"""Reproduce fixed-geometry Step 5 Ni-Al baselines without relaxation.

The script preserves three reviewed execution boundaries:

* ``--load-only`` performs the Step 6B.1 model-loading gate and reads no
  structures.
* ``--phase`` explicitly reproduces one configured phase.  AlNi retains its
  protected Step 6B.2 report path; the four other phases use Step 6B.3 report
  paths.
* ``--all-remaining`` performs Step 6B.3 for exactly Al3Ni, Al3Ni2, Al3Ni5,
  and AlNi3.  It loads one MACE calculator, reuses it across four independent
  source copies, performs four single-point calculations, and publishes four
  reports plus one text and one JSON summary as one transaction.

No mode imports an optimizer, moves atoms, changes a cell, writes a structure,
or performs geometry relaxation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("reproduce_ni_al_mace_baseline")

DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/mace_relaxation.json")
REQUIRED_SELECTED_DIRECTORY_RELATIVE_PATH = Path(
    "data/processed/ni_al_structures/selected"
)
REQUIRED_BASELINE_TABLE_RELATIVE_PATH = Path(
    "results/mace_zero_shot/tables/ni_al_mace_zero_shot.json"
)
REQUIRED_ZERO_SHOT_STRUCTURES_RELATIVE_PATH = Path(
    "results/mace_zero_shot/structures"
)
REQUIRED_COMPARISON_DIRECTORY_RELATIVE_PATH = Path(
    "results/mace_relaxation/comparison"
)

CONFIG_PHASE_ORDER = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
REMAINING_PHASES = ("Al3Ni", "Al3Ni2", "Al3Ni5", "AlNi3")
ALNI_PILOT_PHASE = "AlNi"
EXPECTED_MATERIAL_IDS = {
    "Al3Ni": "mp-622209",
    "Al3Ni2": "mp-1057",
    "AlNi": "mp-1487",
    "Al3Ni5": "mp-16514",
    "AlNi3": "mp-2593",
}
EXPECTED_ATOM_COUNTS = {
    "Al3Ni": 16,
    "Al3Ni2": 5,
    "AlNi": 2,
    "Al3Ni5": 8,
    "AlNi3": 4,
}
EXPECTED_MODEL = {
    "family": "MACE",
    "name": "MACE-MP-0",
    "value": "small",
    "device": "cpu",
    "default_dtype": "float64",
    "dispersion": False,
}

ALLOWED_ELEMENTS = frozenset({"Al", "Ni"})
VALID_DTYPES = frozenset({"float32", "float64"})
REQUIRED_FACTORY_PARAMETERS = frozenset(
    {"model", "device", "default_dtype", "dispersion"}
)
BASELINE_EVALUATION_TYPE = "zero-shot single-point"
STRESS_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
STRESS_FIELDS = tuple(
    f"stress_{label}_eV_per_A3" for label in STRESS_LABELS
)
FORCE_BASELINE_FIELDS = (
    "maximum_force_eV_per_A",
    "mean_force_eV_per_A",
    "rms_force_eV_per_A",
    "minimum_force_eV_per_A",
    "total_force_x_eV_per_A",
    "total_force_y_eV_per_A",
    "total_force_z_eV_per_A",
    "total_force_norm_eV_per_A",
)

ALNI_REPORT_FILENAME = "AlNi_step6b2_baseline_reproduction.txt"
BATCH_SUMMARY_FILENAME = "ni_al_step6b3_baseline_reproduction_summary.txt"
BATCH_JSON_FILENAME = "ni_al_step6b3_baseline_reproduction.json"
IMMUTABILITY_ABSOLUTE_TOLERANCE = 1.0e-12

ENERGY_TOLERANCE = (1.0e-8, 1.0e-10)
ENERGY_PER_ATOM_TOLERANCE = (1.0e-9, 1.0e-10)
FORCE_TOLERANCE = (1.0e-8, 1.0e-8)
STRESS_TOLERANCE = (1.0e-9, 1.0e-8)
VOLUME_TOLERANCE = (1.0e-10, 1.0e-10)


class Step6BError(RuntimeError):
    """Base class for anticipated Step 6B workflow failures."""


class RuntimeEnvironmentError(Step6BError):
    """Raised when the repository's Python environment is not active."""


class ConfigurationError(Step6BError):
    """Raised when workflow configuration is missing or invalid."""


class PhaseScopeError(Step6BError):
    """Raised when no explicit mode or an unsupported phase is requested."""


class DependencyError(Step6BError):
    """Raised when an installed scientific dependency is unavailable."""


class InputValidationError(Step6BError):
    """Raised when a selected structure or metadata file is invalid."""


class BaselineValidationError(Step6BError):
    """Raised when the stored Step 5 baseline is invalid."""


class OutputCollisionError(Step6BError):
    """Raised when one or more targets exist without overwrite consent."""


class CalculatorLoadingError(Step6BError):
    """Raised when the configured MACE calculator cannot be constructed."""


class CalculatorValidationError(Step6BError):
    """Raised when the MACE factory does not return an ASE calculator."""


class CalculationError(Step6BError):
    """Raised when a single-point property request fails."""


class NonFiniteResultError(Step6BError):
    """Raised when MACE or a derived statistic contains NaN or infinity."""


class PublicationError(Step6BError):
    """Raised when transactional output publication cannot be completed."""

    def __init__(
        self,
        message: str,
        publication_state: str = "none",
    ) -> None:
        super().__init__(message)
        self.publication_state = publication_state


class DuplicateJsonKeyError(ValueError):
    """Internal signal used to reject ambiguous duplicate JSON keys."""


@dataclass(frozen=True)
class CommandLineOptions:
    """Validated command-line selections for one invocation."""

    config: Path
    phase: str | None
    all_remaining: bool
    load_only: bool
    overwrite: bool
    verbose: bool


@dataclass(frozen=True)
class PhaseDefinition:
    """Expected identity and selected-cell size for one configured phase."""

    phase_key: str
    material_id: str
    atom_count: int

    @property
    def is_alni_pilot(self) -> bool:
        """Return whether this phase belongs to Step 6B.2."""

        return self.phase_key == ALNI_PILOT_PHASE

    @property
    def step_label(self) -> str:
        """Return the reporting step associated with this phase."""

        return "Step 6B.2" if self.is_alni_pilot else "Step 6B.3"

    @property
    def report_filename(self) -> str:
        """Return the phase's sole individual report filename."""

        if self.is_alni_pilot:
            return ALNI_REPORT_FILENAME
        return f"{self.phase_key}_step6b3_baseline_reproduction.txt"


@dataclass(frozen=True)
class ModelSettings:
    """Validated MACE settings read directly from the relaxation config."""

    family: str
    name: str
    value: str
    device: str
    default_dtype: str
    dispersion: bool


@dataclass(frozen=True)
class MaceDependencies:
    """Focused imports required to construct and validate one calculator."""

    factory: Callable[..., Any]
    ase_calculator_class: type[Any]


@dataclass(frozen=True)
class ScientificDependencies:
    """Scientific objects required only by structure-evaluation modes."""

    numpy: Any
    ase_read: Callable[..., Any]
    composition_class: type[Any]


@dataclass(frozen=True)
class RepositoryLayout:
    """Canonical Step 6B input and output directories."""

    selected_directory: Path
    baseline_table: Path
    annotated_directory: Path
    comparison_directory: Path
    reports_directory: Path
    tables_directory: Path
    protected_alni_report: Path
    batch_summary_report: Path
    batch_json_table: Path


@dataclass(frozen=True)
class PhasePaths:
    """Canonical paths for one explicitly requested phase."""

    structure: Path
    metadata: Path
    annotated_structure: Path
    report: Path


@dataclass(frozen=True)
class FileSnapshot:
    """Content and metadata fingerprint for one read-only source file."""

    label: str
    path: Path
    sha256: str
    size: int
    modification_time_ns: int


@dataclass(frozen=True)
class ValidationCheck:
    """One independently reportable validation or immutability assertion."""

    label: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidatedStructure:
    """Validated source atoms plus identity and read-only fingerprints."""

    phase: PhaseDefinition
    atoms: Any
    material_id: str
    formula: str
    atom_order: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    structure_snapshot: FileSnapshot
    metadata_snapshot: FileSnapshot
    checks: tuple[ValidationCheck, ...]


@dataclass(frozen=True)
class BaselineTable:
    """Validated Step 5 table shared by all requested phases."""

    document: Mapping[str, Any]
    snapshot: FileSnapshot


@dataclass(frozen=True)
class BaselineData:
    """The unique validated Step 5 record for one phase."""

    phase: PhaseDefinition
    record: Mapping[str, Any]
    annotated_structure: Path
    table_snapshot: FileSnapshot
    annotated_snapshot: FileSnapshot
    checks: tuple[ValidationCheck, ...]


@dataclass(frozen=True)
class InitialState:
    """Independent copies used to detect calculator-induced mutation."""

    positions: Any
    cell: Any
    symbols: tuple[str, ...]
    numbers: Any
    pbc: Any
    atom_count: int
    volume: float


@dataclass(frozen=True)
class SinglePointResult:
    """Finite values calculated for one unchanged working copy."""

    total_energy_eV: float
    energy_per_atom_eV: float
    forces_eV_per_A: Any
    force_magnitudes_eV_per_A: Any
    maximum_force_eV_per_A: float
    rms_force_eV_per_A: float
    total_force_eV_per_A: Any
    total_force_norm_eV_per_A: float
    stress_eV_per_A3: Any
    volume_A3: float
    volume_per_atom_A3: float
    number_of_atoms: int


@dataclass(frozen=True)
class NumericComparison:
    """One tolerance-controlled Step 5 versus reproduced comparison."""

    label: str
    field_name: str
    unit: str
    baseline: float
    reproduced: float
    absolute_difference: float
    relative_difference: float | None
    absolute_tolerance: float
    relative_tolerance: float
    effective_tolerance: float
    passed: bool


@dataclass(frozen=True)
class TextComparison:
    """One exact non-numeric comparison."""

    label: str
    baseline: str
    reproduced: str
    passed: bool


@dataclass(frozen=True)
class CalculatorIdentity:
    """Safe class identity reported after calculator construction."""

    module_name: str
    class_name: str

    @property
    def qualified_name(self) -> str:
        """Return the module-qualified calculator class name."""

        return f"{self.module_name}.{self.class_name}"


@dataclass(frozen=True)
class PhaseExecution:
    """Complete reportable result for one phase."""

    structure: ValidatedStructure
    baseline: BaselineData
    result: SinglePointResult
    comparisons: tuple[NumericComparison, ...]
    material_comparison: TextComparison
    immutability_checks: tuple[ValidationCheck, ...]
    source_checks: tuple[ValidationCheck, ...]
    identity_passed: bool
    immutability_passed: bool
    source_files_passed: bool
    reproducibility_passed: bool


def phase_definition(phase_key: str) -> PhaseDefinition:
    """Build one immutable definition from reviewed project constants."""

    return PhaseDefinition(
        phase_key=phase_key,
        material_id=EXPECTED_MATERIAL_IDS[phase_key],
        atom_count=EXPECTED_ATOM_COUNTS[phase_key],
    )


def locate_project_root() -> Path:
    """Locate the repository from this script rather than the shell directory."""

    return Path(__file__).resolve().parents[1]


def parse_arguments(arguments: Sequence[str] | None = None) -> CommandLineOptions:
    """Parse explicit, mutually exclusive calculation modes."""

    parser = argparse.ArgumentParser(
        description=(
            "Load MACE only, reproduce one explicit Ni-Al phase, or reproduce "
            "exactly the four remaining Step 6B.3 phases."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_RELATIVE_PATH,
        help=(
            "Relaxation configuration path. Relative paths resolve from the "
            "repository root (default: configs/mace_relaxation.json)."
        ),
    )
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--phase",
        help=(
            "Explicit single phase. Valid values: "
            + ", ".join(CONFIG_PHASE_ORDER)
            + "."
        ),
    )
    execution_group.add_argument(
        "--all-remaining",
        action="store_true",
        help=(
            "Process exactly Al3Ni, Al3Ni2, Al3Ni5, and AlNi3; AlNi is "
            "excluded."
        ),
    )
    execution_group.add_argument(
        "--load-only",
        action="store_true",
        help=(
            "Run Step 6B.1 model loading only; do not read structures, request "
            "properties, or write repository output."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Intentionally replace only Step 6B.3 outputs targeted by this "
            "invocation. The AlNi Step 6B.2 pilot report is never replaced."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed workflow diagnostics.",
    )
    parsed = parser.parse_args(arguments)
    return CommandLineOptions(
        config=parsed.config,
        phase=(None if parsed.phase is None else str(parsed.phase)),
        all_remaining=bool(parsed.all_remaining),
        load_only=bool(parsed.load_only),
        overwrite=bool(parsed.overwrite),
        verbose=bool(parsed.verbose),
    )


def configure_logging(verbose: bool) -> None:
    """Configure readable script logs without third-party debug noise."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False
    logging.getLogger().setLevel(logging.WARNING)


def validate_runtime_environment(project_root: Path) -> None:
    """Require Python 3.11 from this repository's existing virtual environment."""

    expected_prefix = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    if os.path.normcase(str(active_prefix)) != os.path.normcase(
        str(expected_prefix)
    ):
        raise RuntimeEnvironmentError(
            "Run this workflow with the project interpreter: "
            ".\\.venv\\Scripts\\python.exe"
        )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeEnvironmentError(
            "Step 6B requires Python 3.11; the active interpreter is "
            f"Python {sys.version_info.major}.{sys.version_info.minor}."
        )


def choose_requested_phases(
    options: CommandLineOptions,
) -> tuple[PhaseDefinition, ...]:
    """Require an explicit execution selection and preserve reviewed order."""

    if options.load_only:
        if options.overwrite:
            raise PhaseScopeError("--overwrite is not meaningful with --load-only.")
        return ()
    if options.all_remaining:
        return tuple(phase_definition(key) for key in REMAINING_PHASES)
    if options.phase is None:
        raise PhaseScopeError(
            "No calculation scope was selected. Specify --phase PHASE, "
            "--all-remaining, or --load-only."
        )
    if options.phase not in CONFIG_PHASE_ORDER:
        raise PhaseScopeError(
            f"Unsupported --phase value {options.phase!r}. Valid phases: "
            + ", ".join(CONFIG_PHASE_ORDER)
        )
    if options.phase == ALNI_PILOT_PHASE and options.overwrite:
        raise PhaseScopeError(
            "--overwrite is prohibited for AlNi because the successful "
            "Step 6B.2 pilot report is protected."
        )
    return (phase_definition(options.phase),)


def resolve_config_path(project_root: Path, supplied_path: Path) -> Path:
    """Resolve the config and reject paths outside the repository."""

    resolved = (
        supplied_path.resolve()
        if supplied_path.is_absolute()
        else (project_root / supplied_path).resolve()
    )
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ConfigurationError(
            f"Configuration path must be inside the repository: {resolved}"
        ) from exc
    return resolved


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_nonstandard_json_constant(value: str) -> None:
    """Reject non-standard JSON spellings for NaN and infinity."""

    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def read_json_object(
    path: Path,
    label: str,
    error_class: type[Step6BError],
) -> dict[str, Any]:
    """Read strict UTF-8 JSON and require an object at the document root."""

    if not path.is_file():
        raise error_class(f"{label} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonstandard_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise error_class(f"{label} is not valid UTF-8: {path}") from exc
    except (json.JSONDecodeError, DuplicateJsonKeyError, ValueError) as exc:
        detail = (
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        raise error_class(f"{label} contains invalid JSON: {detail}") from exc
    except OSError as exc:
        raise error_class(f"Could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_class(f"{label} root must be a JSON object: {path}")
    return value


def require_mapping(
    parent: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Require one named configuration section to be an object."""

    value = parent.get(field_name)
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a JSON object."
        )
    return value


def require_nonempty_string(
    parent: Mapping[str, Any],
    field_name: str,
) -> str:
    """Return an exact required string or raise a field-specific error."""

    value = parent.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a non-empty string."
        )
    if value != value.strip():
        raise ConfigurationError(
            f"Configuration field '{field_name}' must not contain leading or "
            "trailing whitespace."
        )
    return value


def validate_model_settings(config: Mapping[str, Any]) -> ModelSettings:
    """Validate and retain the successful Step 6B model settings."""

    model = require_mapping(config, "model")
    family = require_nonempty_string(model, "family")
    name = require_nonempty_string(model, "name")
    value = require_nonempty_string(model, "value")
    device = require_nonempty_string(model, "device")
    default_dtype = require_nonempty_string(model, "default_dtype")
    dispersion = model.get("dispersion")
    if default_dtype not in VALID_DTYPES:
        raise ConfigurationError(
            "Configuration field 'model.default_dtype' must be float32 or "
            f"float64; received {default_dtype!r}."
        )
    if not isinstance(dispersion, bool):
        raise ConfigurationError(
            "Configuration field 'model.dispersion' must be Boolean."
        )
    actual = {
        "family": family,
        "name": name,
        "value": value,
        "device": device,
        "default_dtype": default_dtype,
        "dispersion": dispersion,
    }
    for field_name, expected_value in EXPECTED_MODEL.items():
        if actual[field_name] != expected_value:
            raise ConfigurationError(
                f"Step 6B requires model.{field_name}={expected_value!r}; "
                f"received {actual[field_name]!r}."
            )
    return ModelSettings(
        family=family,
        name=name,
        value=value,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
    )


def resolve_repository_path(
    project_root: Path,
    raw_value: str,
    field_name: str,
) -> Path:
    """Resolve a configured path and prevent it from escaping the repository."""

    supplied = Path(raw_value.replace("/", os.sep))
    resolved = (
        supplied.resolve()
        if supplied.is_absolute()
        else (project_root / supplied).resolve()
    )
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ConfigurationError(
            f"Configuration field '{field_name}' must resolve inside the "
            f"repository: {resolved}"
        ) from exc
    return resolved


def configured_repository_path(
    section: Mapping[str, Any],
    field_name: str,
    project_root: Path,
) -> Path:
    """Read one nonempty configured repository path."""

    return resolve_repository_path(
        project_root,
        require_nonempty_string(section, field_name),
        field_name,
    )


def validate_repository_layout(
    config: Mapping[str, Any],
    project_root: Path,
) -> RepositoryLayout:
    """Validate all canonical paths and expected phase identities."""

    if config.get("schema_version") != "1.0":
        raise ConfigurationError(
            "Configuration schema_version must equal '1.0'."
        )
    raw_phase_order = config.get("phase_order")
    if raw_phase_order != list(CONFIG_PHASE_ORDER):
        raise ConfigurationError(
            "Configuration phase_order must be: "
            + ", ".join(CONFIG_PHASE_ORDER)
        )
    expected_phases = require_mapping(config, "expected_phases")
    if set(expected_phases) != set(CONFIG_PHASE_ORDER):
        raise ConfigurationError(
            "Configuration expected_phases must contain exactly the five "
            "reviewed phase keys."
        )
    for phase_key in CONFIG_PHASE_ORDER:
        settings = expected_phases.get(phase_key)
        if not isinstance(settings, dict):
            raise ConfigurationError(
                f"Configuration expected_phases.{phase_key} must be an object."
            )
        material_id = require_nonempty_string(
            settings, "expected_material_id"
        )
        if material_id != EXPECTED_MATERIAL_IDS[phase_key]:
            raise ConfigurationError(
                f"Configuration material ID for {phase_key} must be "
                f"{EXPECTED_MATERIAL_IDS[phase_key]!r}; received "
                f"{material_id!r}."
            )

    inputs = require_mapping(config, "input")
    outputs = require_mapping(config, "output")
    selected_directory = configured_repository_path(
        inputs, "selected_structure_directory", project_root
    )
    baseline_table = configured_repository_path(
        inputs, "zero_shot_table", project_root
    )
    comparison_directory = configured_repository_path(
        outputs, "comparison_directory", project_root
    )
    required_selected = (
        project_root / REQUIRED_SELECTED_DIRECTORY_RELATIVE_PATH
    ).resolve()
    required_table = (
        project_root / REQUIRED_BASELINE_TABLE_RELATIVE_PATH
    ).resolve()
    required_comparison = (
        project_root / REQUIRED_COMPARISON_DIRECTORY_RELATIVE_PATH
    ).resolve()
    if selected_directory != required_selected:
        raise ConfigurationError(
            "Step 6B requires input.selected_structure_directory to resolve "
            f"to {required_selected}; received {selected_directory}."
        )
    if baseline_table != required_table:
        raise ConfigurationError(
            "Step 6B requires input.zero_shot_table to resolve to "
            f"{required_table}; received {baseline_table}."
        )
    if comparison_directory != required_comparison:
        raise ConfigurationError(
            "Step 6B requires output.comparison_directory to resolve to "
            f"{required_comparison}; received {comparison_directory}."
        )
    annotated_directory = (
        project_root / REQUIRED_ZERO_SHOT_STRUCTURES_RELATIVE_PATH
    ).resolve()
    reports_directory = comparison_directory / "reports"
    tables_directory = comparison_directory / "tables"
    for label, directory in (
        ("selected-structure", selected_directory),
        ("Step 5 annotated-structure", annotated_directory),
        ("comparison", comparison_directory),
        ("comparison reports", reports_directory),
        ("comparison tables", tables_directory),
    ):
        if not directory.is_dir():
            raise ConfigurationError(
                f"Required {label} directory does not exist: {directory}"
            )
    return RepositoryLayout(
        selected_directory=selected_directory,
        baseline_table=baseline_table,
        annotated_directory=annotated_directory,
        comparison_directory=comparison_directory,
        reports_directory=reports_directory,
        tables_directory=tables_directory,
        protected_alni_report=reports_directory / ALNI_REPORT_FILENAME,
        batch_summary_report=reports_directory / BATCH_SUMMARY_FILENAME,
        batch_json_table=tables_directory / BATCH_JSON_FILENAME,
    )


def paths_for_phase(
    layout: RepositoryLayout,
    phase: PhaseDefinition,
) -> PhasePaths:
    """Derive all canonical paths for one phase."""

    return PhasePaths(
        structure=layout.selected_directory / f"{phase.phase_key}.extxyz",
        metadata=layout.selected_directory / f"{phase.phase_key}.metadata.json",
        annotated_structure=(
            layout.annotated_directory
            / f"{phase.phase_key}_mace_zero_shot.extxyz"
        ),
        report=layout.reports_directory / phase.report_filename,
    )


def target_paths(
    requested_phases: Sequence[PhaseDefinition],
    layout: RepositoryLayout,
    all_remaining: bool,
) -> tuple[Path, ...]:
    """Return exactly the persistent outputs authorized by this invocation."""

    reports = tuple(
        paths_for_phase(layout, phase).report for phase in requested_phases
    )
    if not all_remaining:
        return reports
    targets = reports + (
        layout.batch_summary_report,
        layout.batch_json_table,
    )
    if layout.protected_alni_report in targets:
        raise ConfigurationError(
            "Internal safety error: batch targets include the protected AlNi "
            "Step 6B.2 report."
        )
    return targets


def validate_output_collisions(
    targets: Sequence[Path],
    overwrite: bool,
    project_root: Path,
) -> None:
    """List all collisions before importing MACE or reading a structure."""

    existing = [path for path in targets if os.path.lexists(path)]
    invalid = [
        path
        for path in existing
        if path.is_symlink() or not path.is_file()
    ]
    if invalid:
        raise OutputCollisionError(
            "Step 6B output collision(s) include a symlink or non-regular "
            "target. Every existing target is listed; no replacement was "
            "attempted:\n"
            + "\n".join(
                "  - "
                + path.relative_to(project_root).as_posix()
                + (
                    " [INVALID: symlink or non-regular target]"
                    if path in invalid
                    else " [regular-file collision]"
                )
                for path in existing
            )
        )
    if existing and not overwrite:
        raise OutputCollisionError(
            "Step 6B output collision(s) found. Inspect every file and use "
            "--overwrite only when replacement is intentional:\n"
            + "\n".join(
                f"  - {path.relative_to(project_root).as_posix()}"
                for path in existing
            )
        )


def import_mace_dependencies() -> MaceDependencies:
    """Import only the MACE factory and ASE calculator base class."""

    try:
        from ase.calculators.calculator import Calculator
        from mace.calculators import mace_mp
    except (ImportError, OSError) as exc:
        missing_name = (
            exc.name
            if isinstance(exc, ImportError) and exc.name
            else "an installed MACE dependency"
        )
        raise DependencyError(
            "Required MACE loading import is unavailable: "
            f"{missing_name} ({type(exc).__name__}: {exc})."
        ) from exc
    try:
        parameters = inspect.signature(mace_mp).parameters
    except (TypeError, ValueError) as exc:
        raise DependencyError(
            "Could not inspect the installed mace_mp factory."
        ) from exc
    missing_parameters = REQUIRED_FACTORY_PARAMETERS.difference(parameters)
    if missing_parameters:
        raise DependencyError(
            "The installed mace_mp factory is missing required parameters: "
            + ", ".join(sorted(missing_parameters))
        )
    return MaceDependencies(
        factory=mace_mp,
        ase_calculator_class=Calculator,
    )


def import_scientific_dependencies() -> ScientificDependencies:
    """Import structure libraries only for calculation modes."""

    try:
        import numpy as np
        from ase.io import read as ase_read
        from pymatgen.core import Composition
    except (ImportError, OSError) as exc:
        missing_name = (
            exc.name
            if isinstance(exc, ImportError) and exc.name
            else "an installed scientific dependency"
        )
        raise DependencyError(
            "Required Step 6B dependency is unavailable: "
            f"{missing_name} ({type(exc).__name__}: {exc})."
        ) from exc
    return ScientificDependencies(
        numpy=np,
        ase_read=ase_read,
        composition_class=Composition,
    )


def load_calculator_once(
    settings: ModelSettings,
    dependencies: MaceDependencies,
) -> Any:
    """Construct exactly one configured calculator."""

    LOGGER.info(
        "Loading %s %s once (model=%s, device=%s, dtype=%s, dispersion=%s)...",
        settings.family,
        settings.name,
        settings.value,
        settings.device,
        settings.default_dtype,
        str(settings.dispersion).lower(),
    )
    try:
        calculator = dependencies.factory(
            model=settings.value,
            device=settings.device,
            default_dtype=settings.default_dtype,
            dispersion=settings.dispersion,
        )
    except Exception as exc:
        message = str(exc).strip() or "no additional detail was provided"
        raise CalculatorLoadingError(
            "MACE calculator loading failed "
            f"({type(exc).__name__}: {message})"
        ) from exc
    if calculator is None:
        raise CalculatorValidationError(
            "The MACE factory returned None instead of a calculator."
        )
    if not isinstance(calculator, dependencies.ase_calculator_class):
        calculator_type = type(calculator)
        raise CalculatorValidationError(
            "The MACE factory returned an object that is not an ASE-compatible "
            f"calculator: {calculator_type.__module__}."
            f"{calculator_type.__qualname__}"
        )
    return calculator


def calculator_identity(calculator: Any) -> CalculatorIdentity:
    """Extract class identity without requesting calculator results."""

    calculator_type = type(calculator)
    return CalculatorIdentity(
        module_name=calculator_type.__module__,
        class_name=calculator_type.__qualname__,
    )


def file_sha256(path: Path, label: str, error_class: type[Step6BError]) -> str:
    """Hash a source file without loading it all into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise error_class(f"Could not hash {label}: {path}: {exc}") from exc
    return digest.hexdigest()


def capture_file_snapshot(
    path: Path,
    label: str,
    error_class: type[Step6BError],
) -> FileSnapshot:
    """Capture a read-only file's content hash, size, and modification time."""

    if not path.is_file():
        raise error_class(f"{label} does not exist: {path}")
    try:
        stat = path.stat()
    except OSError as exc:
        raise error_class(f"Could not inspect {label}: {path}: {exc}") from exc
    return FileSnapshot(
        label=label,
        path=path,
        sha256=file_sha256(path, label, error_class),
        size=stat.st_size,
        modification_time_ns=stat.st_mtime_ns,
    )


def validate_protected_alni_report(layout: RepositoryLayout) -> FileSnapshot:
    """Require the successful Step 6B.2 pilot before remaining-phase work."""

    snapshot = capture_file_snapshot(
        layout.protected_alni_report,
        "protected AlNi Step 6B.2 report",
        InputValidationError,
    )
    try:
        text = layout.protected_alni_report.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputValidationError(
            "Could not read the protected AlNi Step 6B.2 report."
        ) from exc
    if "Overall reproducibility status: PASS" not in text:
        raise InputValidationError(
            "Protected AlNi Step 6B.2 report does not contain a PASS status."
        )
    return snapshot


def composition_signature(composition: Any) -> tuple[tuple[str, float], ...]:
    """Create an element-order-independent reduced-composition signature."""

    reduced = composition.reduced_composition
    return tuple(
        sorted(
            (str(symbol), float(amount))
            for symbol, amount in reduced.get_el_amt_dict().items()
        )
    )


def metadata_string(
    metadata: Mapping[str, Any],
    field_name: str,
    phase_key: str,
) -> str:
    """Require one nonempty selected-structure provenance string."""

    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(
            f"Metadata for {phase_key} requires nonempty '{field_name}'."
        )
    return value.strip()


def read_one_extxyz(
    path: Path,
    label: str,
    dependencies: ScientificDependencies,
    error_class: type[Step6BError],
) -> Any:
    """Read exactly one EXTXYZ frame with a field-specific error."""

    try:
        frames = dependencies.ase_read(str(path), index=":", format="extxyz")
    except MemoryError:
        raise
    except Exception as exc:
        raise error_class(
            f"Could not read {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise error_class(
            f"{label} must contain exactly one structure; found {len(frames)}."
        )
    return frames[0]


def validate_selected_structure(
    phase: PhaseDefinition,
    paths: PhasePaths,
    project_root: Path,
    dependencies: ScientificDependencies,
) -> ValidatedStructure:
    """Validate one selected phase without modifying either source file."""

    np = dependencies.numpy
    structure_snapshot = capture_file_snapshot(
        paths.structure,
        f"selected {phase.phase_key} EXTXYZ",
        InputValidationError,
    )
    metadata_snapshot = capture_file_snapshot(
        paths.metadata,
        f"selected {phase.phase_key} metadata",
        InputValidationError,
    )
    metadata = read_json_object(
        paths.metadata,
        f"selected {phase.phase_key} metadata",
        InputValidationError,
    )
    metadata_phase = metadata_string(metadata, "phase_key", phase.phase_key)
    material_id = metadata_string(metadata, "material_id", phase.phase_key)
    formula = metadata_string(metadata, "formula_pretty", phase.phase_key)
    if metadata_phase != phase.phase_key:
        raise InputValidationError(
            f"Metadata phase_key for {phase.phase_key} is "
            f"{metadata_phase!r}."
        )
    if material_id != phase.material_id:
        raise InputValidationError(
            f"Metadata material_id for {phase.phase_key} must be "
            f"{phase.material_id!r}; found {material_id!r}."
        )

    expected_composition = dependencies.composition_class(phase.phase_key)
    expected_signature = composition_signature(expected_composition)
    try:
        formula_composition = dependencies.composition_class(formula)
    except Exception as exc:
        raise InputValidationError(
            f"Metadata formula for {phase.phase_key} is invalid: {formula!r}."
        ) from exc
    if composition_signature(formula_composition) != expected_signature:
        raise InputValidationError(
            f"Metadata formula {formula!r} does not match {phase.phase_key}."
        )
    reduced_value = metadata.get("reduced_composition")
    if not isinstance(reduced_value, str) or not reduced_value.strip():
        raise InputValidationError(
            f"Metadata reduced_composition is missing for {phase.phase_key}."
        )
    try:
        reduced_composition = dependencies.composition_class(reduced_value)
    except Exception as exc:
        raise InputValidationError(
            f"Metadata reduced_composition is invalid for {phase.phase_key}."
        ) from exc
    if composition_signature(reduced_composition) != expected_signature:
        raise InputValidationError(
            f"Metadata reduced_composition does not match {phase.phase_key}."
        )
    selected_status = metadata.get("selected_candidate")
    if selected_status is not True:
        raise InputValidationError(
            f"Metadata selected_candidate must be true for {phase.phase_key}."
        )
    selected_path_value = metadata.get("selected_extxyz_path")
    if not isinstance(selected_path_value, str) or not selected_path_value.strip():
        raise InputValidationError(
            f"Metadata selected_extxyz_path is missing for {phase.phase_key}."
        )
    selected_path = resolve_repository_path(
        project_root,
        selected_path_value.strip(),
        f"{phase.phase_key} selected_extxyz_path",
    )
    if selected_path != paths.structure:
        raise InputValidationError(
            f"Metadata selected_extxyz_path does not identify {paths.structure}."
        )

    atoms = read_one_extxyz(
        paths.structure,
        f"selected {phase.phase_key} EXTXYZ",
        dependencies,
        InputValidationError,
    )
    if atoms.calc is not None:
        raise InputValidationError(
            f"Selected {phase.phase_key} structure has a calculator attached "
            "after reading."
        )
    atom_count = len(atoms)
    if atom_count < 1 or atom_count != phase.atom_count:
        raise InputValidationError(
            f"Selected {phase.phase_key} structure must contain "
            f"{phase.atom_count} atoms; found {atom_count}."
        )
    symbols = tuple(atoms.get_chemical_symbols())
    numbers = tuple(int(value) for value in atoms.numbers.tolist())
    unexpected = set(symbols).difference(ALLOWED_ELEMENTS)
    if unexpected:
        raise InputValidationError(
            f"Selected {phase.phase_key} contains unsupported elements: "
            + ", ".join(sorted(unexpected))
        )
    try:
        structure_composition = dependencies.composition_class(Counter(symbols))
    except Exception as exc:
        raise InputValidationError(
            f"Could not determine composition for {phase.phase_key}."
        ) from exc
    if composition_signature(structure_composition) != expected_signature:
        raise InputValidationError(
            f"Selected structure reduces to "
            f"{structure_composition.reduced_formula}; expected "
            f"{phase.phase_key}."
        )

    pbc = np.asarray(atoms.pbc, dtype=bool)
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    if pbc.shape != (3,) or not bool(pbc.all()):
        raise InputValidationError(
            f"Selected {phase.phase_key} must be periodic in x, y, and z."
        )
    if positions.shape != (atom_count, 3) or not bool(
        np.isfinite(positions).all()
    ):
        raise InputValidationError(
            f"Selected {phase.phase_key} has invalid or nonfinite positions."
        )
    if cell.shape != (3, 3) or not bool(np.isfinite(cell).all()):
        raise InputValidationError(
            f"Selected {phase.phase_key} has invalid or nonfinite cell values."
        )
    volume = float(atoms.get_volume())
    if not math.isfinite(volume) or volume <= 0.0:
        raise InputValidationError(
            f"Selected {phase.phase_key} has invalid volume {volume!r}."
        )

    metadata_sites = metadata.get("number_of_sites")
    if (
        isinstance(metadata_sites, bool)
        or not isinstance(metadata_sites, int)
        or metadata_sites != atom_count
    ):
        raise InputValidationError(
            f"Metadata number_of_sites for {phase.phase_key} does not match "
            f"the structure: {metadata_sites!r} versus {atom_count}."
        )
    metadata_volume = metadata.get("volume_A3")
    if isinstance(metadata_volume, bool) or not isinstance(
        metadata_volume, (int, float)
    ):
        raise InputValidationError(
            f"Metadata volume_A3 is invalid for {phase.phase_key}."
        )
    metadata_volume_float = float(metadata_volume)
    if not math.isfinite(metadata_volume_float) or not math.isclose(
        metadata_volume_float,
        volume,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise InputValidationError(
            f"Metadata volume_A3 is inconsistent for {phase.phase_key}."
        )
    if atoms.info.get("phase_key") != phase.phase_key:
        raise InputValidationError(
            f"Selected EXTXYZ phase_key is mismatched for {phase.phase_key}."
        )
    if atoms.info.get("material_id") != material_id:
        raise InputValidationError(
            f"Selected EXTXYZ material_id is mismatched for {phase.phase_key}."
        )
    if atoms.info.get("formula_pretty") != formula:
        raise InputValidationError(
            f"Selected EXTXYZ formula_pretty is mismatched for "
            f"{phase.phase_key}."
        )
    if not bool(atoms.info.get("selected_candidate")):
        raise InputValidationError(
            f"Selected EXTXYZ selected_candidate is not true for "
            f"{phase.phase_key}."
        )
    if atoms.calc is not None:
        raise InputValidationError(
            f"A calculator became attached while reading {phase.phase_key}."
        )

    if capture_file_snapshot(
        paths.structure,
        f"selected {phase.phase_key} EXTXYZ",
        InputValidationError,
    ) != structure_snapshot:
        raise InputValidationError(
            f"Original {phase.phase_key} EXTXYZ changed during validation."
        )
    if capture_file_snapshot(
        paths.metadata,
        f"selected {phase.phase_key} metadata",
        InputValidationError,
    ) != metadata_snapshot:
        raise InputValidationError(
            f"Original {phase.phase_key} metadata changed during validation."
        )

    checks = (
        ValidationCheck(
            "EXTXYZ file exists", True, relative_path(paths.structure, project_root)
        ),
        ValidationCheck(
            "Metadata JSON exists", True, relative_path(paths.metadata, project_root)
        ),
        ValidationCheck("Phase key", True, phase.phase_key),
        ValidationCheck("Materials Project ID", True, material_id),
        ValidationCheck("Metadata formula", True, formula),
        ValidationCheck(
            "Metadata formula composition",
            True,
            formula_composition.reduced_formula,
        ),
        ValidationCheck(
            "Metadata reduced composition",
            True,
            reduced_composition.reduced_formula,
        ),
        ValidationCheck(
            "Metadata selected EXTXYZ path",
            True,
            relative_path(selected_path, project_root),
        ),
        ValidationCheck(
            "Metadata site count", True, str(metadata_sites)
        ),
        ValidationCheck(
            "Metadata volume",
            True,
            f"{format_float(metadata_volume_float)} angstrom^3",
        ),
        ValidationCheck("Atom count", True, str(atom_count)),
        ValidationCheck("Elements", True, ", ".join(sorted(set(symbols)))),
        ValidationCheck(
            "Reduced composition",
            True,
            structure_composition.reduced_formula,
        ),
        ValidationCheck("Atom ordering", True, ", ".join(symbols)),
        ValidationCheck(
            "Atomic numbers", True, ", ".join(map(str, numbers))
        ),
        ValidationCheck("PBC in all directions", True, "true, true, true"),
        ValidationCheck("Finite positions", True, f"shape={positions.shape}"),
        ValidationCheck("Finite cell", True, f"shape={cell.shape}"),
        ValidationCheck("Positive finite volume", True, format_float(volume)),
        ValidationCheck("Calculator absent after reading", True, "None"),
        ValidationCheck("Selected status", True, "selected_candidate=true"),
        ValidationCheck(
            "EXTXYZ identity header",
            True,
            f"phase_key={phase.phase_key}; material_id={material_id}; "
            f"formula_pretty={formula}",
        ),
        ValidationCheck(
            "Original EXTXYZ unchanged while reading",
            True,
            f"sha256={structure_snapshot.sha256}",
        ),
    )
    LOGGER.debug(
        "Validated %s: %s, %d atoms, volume %.17g A^3.",
        phase.phase_key,
        material_id,
        atom_count,
        volume,
    )
    return ValidatedStructure(
        phase=phase,
        atoms=atoms,
        material_id=material_id,
        formula=formula,
        atom_order=symbols,
        atomic_numbers=numbers,
        structure_snapshot=structure_snapshot,
        metadata_snapshot=metadata_snapshot,
        checks=checks,
    )


def finite_baseline_number(
    record: Mapping[str, Any],
    field_name: str,
    phase_key: str,
) -> float:
    """Require one stored Step 5 property to be a finite real number."""

    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineValidationError(
            f"Step 5 field '{field_name}' for {phase_key} is not numeric."
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise BaselineValidationError(
            f"Step 5 field '{field_name}' for {phase_key} is nonfinite."
        )
    return converted


def load_baseline_table(
    layout: RepositoryLayout,
    settings: ModelSettings,
) -> BaselineTable:
    """Validate shared Step 5 provenance once before phase-specific checks."""

    snapshot = capture_file_snapshot(
        layout.baseline_table,
        "Step 5 zero-shot JSON table",
        BaselineValidationError,
    )
    table = read_json_object(
        layout.baseline_table,
        "Step 5 zero-shot JSON table",
        BaselineValidationError,
    )
    if table.get("schema_version") != "1.0":
        raise BaselineValidationError(
            "Step 5 table schema_version must equal '1.0'."
        )
    if table.get("evaluation_type") != BASELINE_EVALUATION_TYPE:
        raise BaselineValidationError(
            "Step 5 table is not a zero-shot single-point evaluation."
        )
    if table.get("overall_status") != "success":
        raise BaselineValidationError(
            "Step 5 table overall_status is not success."
        )
    expected_model = {
        "family": settings.family,
        "name": settings.name,
        "size": settings.value,
        "device": settings.device,
        "dtype": settings.default_dtype,
        "dispersion_enabled": settings.dispersion,
    }
    model = table.get("model")
    if not isinstance(model, dict) or any(
        model.get(field_name) != expected
        for field_name, expected in expected_model.items()
    ):
        raise BaselineValidationError(
            "Step 5 model settings do not match the Step 6B configuration."
        )
    if table.get("stress_component_order") != list(STRESS_LABELS):
        raise BaselineValidationError(
            "Step 5 stress_component_order is not ASE Voigt order "
            "xx, yy, zz, yz, xz, xy."
        )
    records = table.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise BaselineValidationError(
            "Step 5 table records must be an array of objects."
        )
    if capture_file_snapshot(
        layout.baseline_table,
        "Step 5 zero-shot JSON table",
        BaselineValidationError,
    ) != snapshot:
        raise BaselineValidationError(
            "Step 5 JSON table changed while it was being validated."
        )
    return BaselineTable(document=table, snapshot=snapshot)


def validate_annotated_structure(
    phase: PhaseDefinition,
    path: Path,
    source: ValidatedStructure,
    dependencies: ScientificDependencies,
) -> None:
    """Confirm the Step 5 annotated file preserves source identity/geometry."""

    np = dependencies.numpy
    annotated = read_one_extxyz(
        path,
        f"Step 5 annotated {phase.phase_key} EXTXYZ",
        dependencies,
        BaselineValidationError,
    )
    if annotated.calc is not None:
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} unexpectedly has a calculator."
        )
    if len(annotated) != phase.atom_count:
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} atom count is mismatched."
        )
    if tuple(annotated.get_chemical_symbols()) != source.atom_order:
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} atom ordering differs from "
            "the source."
        )
    if not np.array_equal(annotated.numbers, source.atoms.numbers):
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} atomic numbers differ from "
            "the source."
        )
    if not np.allclose(
        annotated.positions,
        source.atoms.positions,
        rtol=0.0,
        atol=IMMUTABILITY_ABSOLUTE_TOLERANCE,
    ):
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} positions differ from source."
        )
    if not np.allclose(
        annotated.cell.array,
        source.atoms.cell.array,
        rtol=0.0,
        atol=IMMUTABILITY_ABSOLUTE_TOLERANCE,
    ):
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} cell differs from source."
        )
    if not np.array_equal(annotated.pbc, source.atoms.pbc):
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} PBC differs from source."
        )
    if not math.isclose(
        float(annotated.get_volume()),
        float(source.atoms.get_volume()),
        rel_tol=0.0,
        abs_tol=IMMUTABILITY_ABSOLUTE_TOLERANCE,
    ):
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} volume differs from source."
        )
    if annotated.info.get("material_id") != phase.material_id:
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} material_id is mismatched."
        )


def validate_phase_baseline(
    phase: PhaseDefinition,
    paths: PhasePaths,
    project_root: Path,
    settings: ModelSettings,
    structure: ValidatedStructure,
    baseline_table: BaselineTable,
    dependencies: ScientificDependencies,
) -> BaselineData:
    """Locate and validate one unique successful Step 5 record."""

    table = baseline_table.document
    completed_phases = table.get("completed_phases")
    failed_phases = table.get("failed_phases")
    if not isinstance(completed_phases, list) or phase.phase_key not in completed_phases:
        raise BaselineValidationError(
            f"Step 5 completed_phases does not contain {phase.phase_key}."
        )
    if not isinstance(failed_phases, list) or phase.phase_key in failed_phases:
        raise BaselineValidationError(
            f"Step 5 failed_phases is invalid for {phase.phase_key}."
        )
    records = table["records"]
    matches = [
        record for record in records if record.get("phase_key") == phase.phase_key
    ]
    if len(matches) != 1:
        raise BaselineValidationError(
            f"Step 5 table must contain exactly one {phase.phase_key} record; "
            f"found {len(matches)}."
        )
    record = matches[0]
    if record.get("material_id") != phase.material_id:
        raise BaselineValidationError(
            f"Step 5 material_id is mismatched for {phase.phase_key}."
        )
    if record.get("evaluation_status") != "success":
        raise BaselineValidationError(
            f"Step 5 evaluation_status is not success for {phase.phase_key}."
        )
    atom_count = record.get("number_of_atoms")
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count != phase.atom_count
        or atom_count != len(structure.atoms)
    ):
        raise BaselineValidationError(
            f"Step 5 atom count is mismatched for {phase.phase_key}."
        )
    record_formula = record.get("formula")
    if not isinstance(record_formula, str) or not record_formula.strip():
        raise BaselineValidationError(
            f"Step 5 formula is missing for {phase.phase_key}."
        )
    try:
        record_composition = dependencies.composition_class(record_formula)
    except Exception as exc:
        raise BaselineValidationError(
            f"Step 5 formula is invalid for {phase.phase_key}."
        ) from exc
    if composition_signature(record_composition) != composition_signature(
        dependencies.composition_class(phase.phase_key)
    ):
        raise BaselineValidationError(
            f"Step 5 formula does not match {phase.phase_key}."
        )
    expected_record_model = {
        "mace_model_name": settings.name,
        "model_size": settings.value,
        "device": settings.device,
        "dtype": settings.default_dtype,
        "dispersion_enabled": settings.dispersion,
    }
    for field_name, expected in expected_record_model.items():
        if record.get(field_name) != expected:
            raise BaselineValidationError(
                f"Step 5 field '{field_name}' is mismatched for "
                f"{phase.phase_key}."
            )

    finite_baseline_number(record, "total_energy_eV", phase.phase_key)
    finite_baseline_number(record, "energy_per_atom_eV", phase.phase_key)
    for field_name in FORCE_BASELINE_FIELDS:
        finite_baseline_number(record, field_name, phase.phase_key)
    for field_name in STRESS_FIELDS:
        finite_baseline_number(record, field_name, phase.phase_key)
    volume = finite_baseline_number(record, "volume_A3", phase.phase_key)
    volume_per_atom = finite_baseline_number(
        record, "volume_per_atom_A3", phase.phase_key
    )
    if volume <= 0.0 or volume_per_atom <= 0.0:
        raise BaselineValidationError(
            f"Step 5 volumes must be positive for {phase.phase_key}."
        )

    input_path_value = record.get("input_structure_path")
    if not isinstance(input_path_value, str) or not input_path_value.strip():
        raise BaselineValidationError(
            f"Step 5 input_structure_path is missing for {phase.phase_key}."
        )
    recorded_input = resolve_repository_path(
        project_root,
        input_path_value.strip(),
        f"Step 5 {phase.phase_key} input_structure_path",
    )
    if recorded_input != paths.structure:
        raise BaselineValidationError(
            f"Step 5 {phase.phase_key} record refers to another input."
        )
    output_path_value = record.get("output_structure_path")
    if not isinstance(output_path_value, str) or not output_path_value.strip():
        raise BaselineValidationError(
            f"Step 5 output_structure_path is missing for {phase.phase_key}."
        )
    recorded_output = resolve_repository_path(
        project_root,
        output_path_value.strip(),
        f"Step 5 {phase.phase_key} output_structure_path",
    )
    if recorded_output != paths.annotated_structure:
        raise BaselineValidationError(
            f"Step 5 {phase.phase_key} output path is not the canonical "
            "annotated structure."
        )
    annotated_snapshot = capture_file_snapshot(
        paths.annotated_structure,
        f"Step 5 annotated {phase.phase_key} EXTXYZ",
        BaselineValidationError,
    )
    if annotated_snapshot.size <= 0:
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} EXTXYZ is empty."
        )
    validate_annotated_structure(
        phase,
        paths.annotated_structure,
        structure,
        dependencies,
    )
    if capture_file_snapshot(
        paths.annotated_structure,
        f"Step 5 annotated {phase.phase_key} EXTXYZ",
        BaselineValidationError,
    ) != annotated_snapshot:
        raise BaselineValidationError(
            f"Step 5 annotated {phase.phase_key} changed during validation."
        )

    checks = (
        ValidationCheck("Step 5 schema version", True, "1.0"),
        ValidationCheck("Step 5 evaluation type", True, BASELINE_EVALUATION_TYPE),
        ValidationCheck("Step 5 overall status", True, "success"),
        ValidationCheck(
            f"Step 5 {phase.phase_key} record count", True, "1"
        ),
        ValidationCheck(
            f"Step 5 {phase.phase_key} status", True, "success"
        ),
        ValidationCheck(
            f"Step 5 {phase.phase_key} material ID", True, phase.material_id
        ),
        ValidationCheck(
            f"Step 5 {phase.phase_key} atom count", True, str(atom_count)
        ),
        ValidationCheck("Step 5 model settings", True, "match configuration"),
        ValidationCheck("Step 5 energy fields finite", True, "2 fields"),
        ValidationCheck("Step 5 force fields finite", True, "8 fields"),
        ValidationCheck("Step 5 stress fields finite", True, "6 fields"),
        ValidationCheck("Step 5 volumes finite and positive", True, "2 fields"),
        ValidationCheck(
            "Step 5 annotated structure exists",
            True,
            relative_path(paths.annotated_structure, project_root),
        ),
        ValidationCheck(
            "Step 5 numerical source",
            True,
            "JSON table (not rounded EXTXYZ force columns)",
        ),
    )
    return BaselineData(
        phase=phase,
        record=record,
        annotated_structure=paths.annotated_structure,
        table_snapshot=baseline_table.snapshot,
        annotated_snapshot=annotated_snapshot,
        checks=checks,
    )


def require_finite_scalar(value: Any, label: str) -> float:
    """Convert one numerical result to a finite Python float."""

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise NonFiniteResultError(
            f"{label} is not a real scalar: {value!r}"
        ) from exc
    if not math.isfinite(converted):
        raise NonFiniteResultError(f"{label} is nonfinite: {converted!r}")
    return converted


def max_absolute_difference(first: Any, second: Any, np: Any) -> float:
    """Return a finite maximum absolute array difference when shapes agree."""

    if np.shape(first) != np.shape(second):
        return math.inf
    difference = np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))
    if difference.size == 0:
        return 0.0
    if not bool(np.isfinite(difference).all()):
        return math.inf
    return float(np.max(difference))


def build_immutability_checks(
    phase_key: str,
    working: Any,
    initial: InitialState,
    dependencies: ScientificDependencies,
) -> list[ValidationCheck]:
    """Evaluate every structural immutability item independently."""

    np = dependencies.numpy
    current_positions = np.asarray(working.positions, dtype=float)
    current_cell = np.asarray(working.cell.array, dtype=float)
    current_symbols = tuple(working.get_chemical_symbols())
    current_numbers = np.asarray(working.numbers)
    current_pbc = np.asarray(working.pbc, dtype=bool)
    current_count = len(working)
    try:
        current_volume = float(working.get_volume())
    except Exception:
        current_volume = math.nan
    positions_passed = (
        current_positions.shape == initial.positions.shape
        and bool(
            np.allclose(
                current_positions,
                initial.positions,
                rtol=0.0,
                atol=IMMUTABILITY_ABSOLUTE_TOLERANCE,
            )
        )
    )
    cell_passed = (
        current_cell.shape == initial.cell.shape
        and bool(
            np.allclose(
                current_cell,
                initial.cell,
                rtol=0.0,
                atol=IMMUTABILITY_ABSOLUTE_TOLERANCE,
            )
        )
    )
    volume_passed = math.isfinite(current_volume) and math.isclose(
        current_volume,
        initial.volume,
        rel_tol=0.0,
        abs_tol=IMMUTABILITY_ABSOLUTE_TOLERANCE,
    )
    return [
        ValidationCheck(
            "Atomic positions unchanged",
            positions_passed,
            "max absolute difference="
            + format_float(
                max_absolute_difference(
                    current_positions, initial.positions, np
                )
            )
            + f"; atol={IMMUTABILITY_ABSOLUTE_TOLERANCE:.1e}, rtol=0",
        ),
        ValidationCheck(
            "Cell vectors unchanged",
            cell_passed,
            "max absolute difference="
            + format_float(
                max_absolute_difference(current_cell, initial.cell, np)
            )
            + f"; atol={IMMUTABILITY_ABSOLUTE_TOLERANCE:.1e}, rtol=0",
        ),
        ValidationCheck(
            "Chemical symbols and atom ordering unchanged",
            current_symbols == initial.symbols,
            f"initial={list(initial.symbols)!r}; final={list(current_symbols)!r}",
        ),
        ValidationCheck(
            "Atomic numbers and atom ordering unchanged",
            bool(np.array_equal(current_numbers, initial.numbers)),
            f"initial={initial.numbers.tolist()!r}; "
            f"final={current_numbers.tolist()!r}",
        ),
        ValidationCheck(
            "Atom count unchanged",
            current_count == initial.atom_count,
            f"initial={initial.atom_count}; final={current_count}",
        ),
        ValidationCheck(
            "Periodic boundary conditions unchanged",
            bool(np.array_equal(current_pbc, initial.pbc)),
            f"initial={initial.pbc.tolist()!r}; final={current_pbc.tolist()!r}",
        ),
        ValidationCheck(
            "Cell volume unchanged",
            volume_passed,
            f"initial={format_float(initial.volume)}; "
            f"final={format_float(current_volume)}; "
            f"absolute difference="
            f"{format_float(abs(current_volume - initial.volume))}; "
            f"atol={IMMUTABILITY_ABSOLUTE_TOLERANCE:.1e}, rtol=0",
        ),
        ValidationCheck(
            "Phase identity retained during calculation",
            True,
            phase_key,
        ),
    ]


def evaluate_single_point(
    structure: ValidatedStructure,
    calculator: Any,
    dependencies: ScientificDependencies,
) -> tuple[SinglePointResult, tuple[ValidationCheck, ...]]:
    """Evaluate one deep copy and prove its complete geometry is unchanged."""

    np = dependencies.numpy
    phase_key = structure.phase.phase_key
    source_atoms = structure.atoms
    working = copy.deepcopy(source_atoms)
    if working is source_atoms:
        raise InputValidationError(
            f"Deep copy did not create an independent {phase_key} Atoms object."
        )
    if working.calc is not None:
        raise InputValidationError(
            f"Deep {phase_key} working copy retained a calculator."
        )
    if np.shares_memory(working.positions, source_atoms.positions):
        raise InputValidationError(
            f"Deep {phase_key} copy shares position memory with source."
        )

    initial = InitialState(
        positions=np.array(working.positions, dtype=np.float64, copy=True),
        cell=np.array(working.cell.array, dtype=np.float64, copy=True),
        symbols=tuple(working.get_chemical_symbols()),
        numbers=np.array(working.numbers, copy=True),
        pbc=np.array(working.pbc, dtype=bool, copy=True),
        atom_count=len(working),
        volume=require_finite_scalar(
            working.get_volume(), f"initial {phase_key} volume"
        ),
    )
    working.calc = calculator
    try:
        try:
            total_energy_raw = working.get_potential_energy()
        except MemoryError:
            raise
        except Exception as exc:
            raise CalculationError(
                f"{phase_key} total-energy request failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        try:
            forces = np.asarray(
                working.get_forces(), dtype=np.float64
            ).copy()
        except MemoryError:
            raise
        except Exception as exc:
            raise CalculationError(
                f"{phase_key} force request failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        try:
            stress = np.asarray(
                working.get_stress(voigt=True), dtype=np.float64
            ).copy()
        except MemoryError:
            raise
        except Exception as exc:
            raise CalculationError(
                f"{phase_key} ASE-Voigt stress request failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        immutability_checks = build_immutability_checks(
            phase_key, working, initial, dependencies
        )
        volume = require_finite_scalar(
            working.get_volume(), f"reproduced {phase_key} volume"
        )
    finally:
        working.calc = None

    if source_atoms.calc is not None:
        raise InputValidationError(
            f"Pristine {phase_key} source acquired a calculator."
        )
    immutability_checks.extend(
        [
            ValidationCheck(
                "Calculator detached from working copy",
                working.calc is None,
                "None",
            ),
            ValidationCheck(
                "Pristine source remains calculator-free",
                source_atoms.calc is None,
                "None",
            ),
        ]
    )
    total_energy = require_finite_scalar(
        total_energy_raw, f"reproduced {phase_key} total energy"
    )
    atom_count = len(working)
    if forces.shape != (atom_count, 3):
        raise CalculationError(
            f"{phase_key} force array has shape {forces.shape}; expected "
            f"({atom_count}, 3)."
        )
    if stress.shape != (6,):
        raise CalculationError(
            f"{phase_key} stress has shape {stress.shape}; expected (6,)."
        )
    if not bool(np.isfinite(forces).all()):
        raise NonFiniteResultError(
            f"{phase_key} atomic forces contain NaN or infinity."
        )
    if not bool(np.isfinite(stress).all()):
        raise NonFiniteResultError(
            f"{phase_key} stress contains NaN or infinity."
        )
    if volume <= 0.0:
        raise NonFiniteResultError(
            f"Reproduced {phase_key} volume is not positive: {volume!r}."
        )
    force_magnitudes = np.linalg.norm(forces, axis=1)
    total_force = np.sum(forces, axis=0, dtype=np.float64)
    if not bool(np.isfinite(force_magnitudes).all()):
        raise NonFiniteResultError(
            f"{phase_key} force magnitudes are nonfinite."
        )
    if not bool(np.isfinite(total_force).all()):
        raise NonFiniteResultError(
            f"{phase_key} total force vector is nonfinite."
        )
    result = SinglePointResult(
        total_energy_eV=total_energy,
        energy_per_atom_eV=require_finite_scalar(
            total_energy / atom_count,
            f"reproduced {phase_key} energy per atom",
        ),
        forces_eV_per_A=forces,
        force_magnitudes_eV_per_A=force_magnitudes,
        maximum_force_eV_per_A=require_finite_scalar(
            np.max(force_magnitudes),
            f"reproduced {phase_key} maximum force",
        ),
        rms_force_eV_per_A=require_finite_scalar(
            np.sqrt(np.mean(np.square(force_magnitudes))),
            f"reproduced {phase_key} RMS force",
        ),
        total_force_eV_per_A=np.asarray(
            total_force, dtype=np.float64
        ).copy(),
        total_force_norm_eV_per_A=require_finite_scalar(
            np.linalg.norm(total_force),
            f"reproduced {phase_key} total force norm",
        ),
        stress_eV_per_A3=stress,
        volume_A3=volume,
        volume_per_atom_A3=require_finite_scalar(
            volume / atom_count,
            f"reproduced {phase_key} volume per atom",
        ),
        number_of_atoms=atom_count,
    )
    return result, tuple(immutability_checks)


def reset_calculator_between_phases(calculator: Any, phase_key: str) -> None:
    """Clear cached results while retaining the same loaded model object."""

    try:
        calculator.reset()
    except Exception as exc:
        raise CalculatorValidationError(
            f"Could not clear calculator state after {phase_key} "
            f"({type(exc).__name__}: {exc})."
        ) from exc


def compare_numeric(
    label: str,
    field_name: str,
    unit: str,
    baseline: float,
    reproduced: float,
    tolerance: tuple[float, float],
) -> NumericComparison:
    """Apply the documented absolute-plus-relative tolerance."""

    absolute_tolerance, relative_tolerance = tolerance
    absolute_difference = abs(reproduced - baseline)
    effective_tolerance = (
        absolute_tolerance + relative_tolerance * abs(baseline)
    )
    relative_difference = (
        None
        if abs(baseline) <= absolute_tolerance
        else absolute_difference / abs(baseline)
    )
    return NumericComparison(
        label=label,
        field_name=field_name,
        unit=unit,
        baseline=baseline,
        reproduced=reproduced,
        absolute_difference=absolute_difference,
        relative_difference=relative_difference,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        effective_tolerance=effective_tolerance,
        passed=absolute_difference <= effective_tolerance,
    )


def build_comparisons(
    baseline: BaselineData,
    result: SinglePointResult,
    material_id: str,
) -> tuple[tuple[NumericComparison, ...], TextComparison]:
    """Compare required scalar, vector, identity, and count values."""

    phase_key = baseline.phase.phase_key
    record = baseline.record
    comparisons: list[NumericComparison] = [
        compare_numeric(
            "Total energy",
            "total_energy_eV",
            "eV",
            finite_baseline_number(record, "total_energy_eV", phase_key),
            result.total_energy_eV,
            ENERGY_TOLERANCE,
        ),
        compare_numeric(
            "Energy per atom",
            "energy_per_atom_eV",
            "eV/atom",
            finite_baseline_number(record, "energy_per_atom_eV", phase_key),
            result.energy_per_atom_eV,
            ENERGY_PER_ATOM_TOLERANCE,
        ),
        compare_numeric(
            "Maximum atomic force magnitude",
            "maximum_force_eV_per_A",
            "eV/angstrom",
            finite_baseline_number(
                record, "maximum_force_eV_per_A", phase_key
            ),
            result.maximum_force_eV_per_A,
            FORCE_TOLERANCE,
        ),
        compare_numeric(
            "RMS atomic force magnitude",
            "rms_force_eV_per_A",
            "eV/angstrom",
            finite_baseline_number(record, "rms_force_eV_per_A", phase_key),
            result.rms_force_eV_per_A,
            FORCE_TOLERANCE,
        ),
    ]
    for index, axis in enumerate(("x", "y", "z")):
        field_name = f"total_force_{axis}_eV_per_A"
        comparisons.append(
            compare_numeric(
                f"Total force {axis}",
                field_name,
                "eV/angstrom",
                finite_baseline_number(record, field_name, phase_key),
                float(result.total_force_eV_per_A[index]),
                FORCE_TOLERANCE,
            )
        )
    comparisons.append(
        compare_numeric(
            "Total force norm",
            "total_force_norm_eV_per_A",
            "eV/angstrom",
            finite_baseline_number(
                record, "total_force_norm_eV_per_A", phase_key
            ),
            result.total_force_norm_eV_per_A,
            FORCE_TOLERANCE,
        )
    )
    for index, stress_label in enumerate(STRESS_LABELS):
        field_name = f"stress_{stress_label}_eV_per_A3"
        comparisons.append(
            compare_numeric(
                f"Stress {stress_label}",
                field_name,
                "eV/angstrom^3",
                finite_baseline_number(record, field_name, phase_key),
                float(result.stress_eV_per_A3[index]),
                STRESS_TOLERANCE,
            )
        )
    comparisons.extend(
        [
            compare_numeric(
                "Volume",
                "volume_A3",
                "angstrom^3",
                finite_baseline_number(record, "volume_A3", phase_key),
                result.volume_A3,
                VOLUME_TOLERANCE,
            ),
            compare_numeric(
                "Volume per atom",
                "volume_per_atom_A3",
                "angstrom^3/atom",
                finite_baseline_number(
                    record, "volume_per_atom_A3", phase_key
                ),
                result.volume_per_atom_A3,
                VOLUME_TOLERANCE,
            ),
            compare_numeric(
                "Atom count",
                "number_of_atoms",
                "atoms",
                float(record["number_of_atoms"]),
                float(result.number_of_atoms),
                (0.0, 0.0),
            ),
        ]
    )
    material_comparison = TextComparison(
        label="Materials Project ID",
        baseline=str(record["material_id"]),
        reproduced=material_id,
        passed=str(record["material_id"]) == material_id,
    )
    return tuple(comparisons), material_comparison


def verify_file_snapshot(snapshot: FileSnapshot) -> tuple[ValidationCheck, ...]:
    """Report content, size, and mtime preservation for one source file."""

    current = capture_file_snapshot(
        snapshot.path,
        snapshot.label,
        InputValidationError,
    )
    return (
        ValidationCheck(
            f"{snapshot.label} content unchanged",
            current.sha256 == snapshot.sha256,
            f"initial_sha256={snapshot.sha256}; "
            f"final_sha256={current.sha256}",
        ),
        ValidationCheck(
            f"{snapshot.label} size unchanged",
            current.size == snapshot.size,
            f"initial={snapshot.size}; final={current.size} bytes",
        ),
        ValidationCheck(
            f"{snapshot.label} modification time unchanged",
            current.modification_time_ns == snapshot.modification_time_ns,
            "initial="
            f"{snapshot.modification_time_ns}; "
            f"final={current.modification_time_ns} ns",
        ),
    )


def verify_phase_source_files(
    structure: ValidatedStructure,
    baseline: BaselineData,
) -> tuple[ValidationCheck, ...]:
    """Prove all four phase-specific/shared source artifacts were unchanged."""

    checks: list[ValidationCheck] = []
    for snapshot in (
        structure.structure_snapshot,
        structure.metadata_snapshot,
        baseline.table_snapshot,
        baseline.annotated_snapshot,
    ):
        checks.extend(verify_file_snapshot(snapshot))
    return tuple(checks)


def build_phase_execution(
    structure: ValidatedStructure,
    baseline: BaselineData,
    result: SinglePointResult,
    comparisons: tuple[NumericComparison, ...],
    material_comparison: TextComparison,
    immutability_checks: tuple[ValidationCheck, ...],
    source_checks: tuple[ValidationCheck, ...],
) -> PhaseExecution:
    """Aggregate all phase status gates without hiding an individual failure."""

    identity_passed = all(
        check.passed for check in (*structure.checks, *baseline.checks)
    ) and material_comparison.passed
    immutability_passed = all(
        check.passed for check in immutability_checks
    )
    source_files_passed = all(check.passed for check in source_checks)
    reproducibility_passed = (
        identity_passed
        and all(comparison.passed for comparison in comparisons)
        and immutability_passed
        and source_files_passed
    )
    return PhaseExecution(
        structure=structure,
        baseline=baseline,
        result=result,
        comparisons=comparisons,
        material_comparison=material_comparison,
        immutability_checks=immutability_checks,
        source_checks=source_checks,
        identity_passed=identity_passed,
        immutability_passed=immutability_passed,
        source_files_passed=source_files_passed,
        reproducibility_passed=reproducibility_passed,
    )


def relative_path(path: Path, project_root: Path) -> str:
    """Return a normalized repository-relative path for reports and logs."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ConfigurationError(
            f"Path is outside the project root: {path}"
        ) from exc


def utc_timestamp() -> str:
    """Return a second-resolution UTC timestamp."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def format_float(value: float) -> str:
    """Retain binary64 round-trip precision in audit artifacts."""

    return f"{float(value):.17g}"


def pass_text(passed: bool) -> str:
    """Format one Boolean result consistently."""

    return "PASS" if passed else "FAIL"


def comparison_by_label(
    execution: PhaseExecution,
    label: str,
) -> NumericComparison:
    """Return one required named comparison."""

    for comparison in execution.comparisons:
        if comparison.label == label:
            return comparison
    raise RuntimeError(f"Internal error: missing comparison {label!r}.")


def build_individual_report(
    project_root: Path,
    config_path: Path,
    settings: ModelSettings,
    paths: PhasePaths,
    calculator_class: CalculatorIdentity,
    execution: PhaseExecution,
    protected_pilot_checks: Sequence[ValidationCheck],
    generated_at_utc: str,
    batch_mode: bool,
) -> str:
    """Build one complete Step 6B.2 or Step 6B.3 phase report."""

    structure = execution.structure
    phase = structure.phase
    baseline = execution.baseline
    result = execution.result
    record = baseline.record
    title = (
        f"{phase.step_label} - {phase.phase_key} Initial Baseline Reproduction"
    )
    lines = [
        title,
        "=" * len(title),
        "",
        "Project and execution",
        "---------------------",
        f"Project root: {project_root}",
        f"Report timestamp (UTC): {generated_at_utc}",
        f"Configuration: {relative_path(config_path, project_root)}",
        f"Phase: {phase.phase_key}",
        f"Materials Project ID: {structure.material_id}",
        f"Execution mode: {'--all-remaining' if batch_mode else '--phase'}",
        "Purpose: computational reproduction of the stored Step 5 "
        "fixed-geometry single-point baseline.",
        "Scientific accuracy conclusion: not evaluated in this sub-step.",
        "",
        "Model settings",
        "--------------",
        f"Family: {settings.family}",
        f"Name: {settings.name}",
        f"Model value: {settings.value}",
        f"Device: {settings.device}",
        f"Default dtype: {settings.default_dtype}",
        f"Dispersion enabled: {str(settings.dispersion).lower()}",
        f"Calculator class: {calculator_class.qualified_name}",
        "Calculator loads for this invocation: 1",
        "Single-point calculations for this phase: 1",
        (
            "Calculator reuse: one shared calculator is reused across four "
            "independent phases."
            if batch_mode
            else "Calculator reuse: not applicable to this single-phase invocation."
        ),
        "",
        "Inputs",
        "------",
        f"Selected structure: {relative_path(paths.structure, project_root)}",
        f"Selected metadata: {relative_path(paths.metadata, project_root)}",
        "Step 5 baseline JSON: "
        f"{relative_path(baseline.table_snapshot.path, project_root)}",
        "Step 5 annotated structure: "
        f"{relative_path(baseline.annotated_structure, project_root)}",
        "Numerical baseline source: Step 5 JSON table",
        "Per-atom force-vector policy: reproduced vectors are recorded, but "
        "the Step 5 JSON has no per-atom force-vector array. Rounded annotated "
        "EXTXYZ force columns are not used as numerical references.",
        "",
        "Identity and metadata validation",
        "--------------------------------",
    ]
    for check in structure.checks:
        lines.append(
            f"[{pass_text(check.passed)}] {check.label}: {check.detail}"
        )
    lines.extend(["", "Step 5 baseline validation", "--------------------------"])
    for check in baseline.checks:
        lines.append(
            f"[{pass_text(check.passed)}] {check.label}: {check.detail}"
        )

    lines.extend(
        [
            "",
            "Reproduced single-point values",
            "------------------------------",
            f"Number of atoms: {result.number_of_atoms}",
            f"Atom ordering: {', '.join(structure.atom_order)}",
            "Atomic numbers: "
            + ", ".join(map(str, structure.atomic_numbers)),
            f"Total energy: {format_float(result.total_energy_eV)} eV",
            "Energy per atom: "
            f"{format_float(result.energy_per_atom_eV)} eV/atom",
            "Maximum atomic force magnitude: "
            f"{format_float(result.maximum_force_eV_per_A)} eV/angstrom",
            "RMS atomic force magnitude: "
            f"{format_float(result.rms_force_eV_per_A)} eV/angstrom",
            "Total force vector (x, y, z): ["
            + ", ".join(
                format_float(value) for value in result.total_force_eV_per_A
            )
            + "] eV/angstrom",
            "Total force norm: "
            f"{format_float(result.total_force_norm_eV_per_A)} eV/angstrom",
            "Stress in ASE Voigt order (xx, yy, zz, yz, xz, xy): ["
            + ", ".join(
                format_float(value) for value in result.stress_eV_per_A3
            )
            + "] eV/angstrom^3",
            f"Volume: {format_float(result.volume_A3)} angstrom^3",
            "Volume per atom: "
            f"{format_float(result.volume_per_atom_A3)} angstrom^3/atom",
            "",
            "Atomic force vectors",
            "--------------------",
        ]
    )
    for index, (symbol, vector, magnitude) in enumerate(
        zip(
            structure.atom_order,
            result.forces_eV_per_A,
            result.force_magnitudes_eV_per_A,
            strict=True,
        ),
        start=1,
    ):
        lines.append(
            f"Atom {index} ({symbol}): ["
            + ", ".join(format_float(value) for value in vector)
            + f"] eV/angstrom; magnitude={format_float(magnitude)} "
            "eV/angstrom"
        )

    lines.extend(
        [
            "",
            "Detailed Step 5 comparisons and tolerances",
            "------------------------------------------",
            "Pass criterion: absolute_difference <= absolute_tolerance + "
            "relative_tolerance * abs(Step 5 value).",
            "Relative difference is N/A when abs(Step 5 value) is at or below "
            "the absolute tolerance.",
            "",
        ]
    )
    for comparison in execution.comparisons:
        relative_difference = (
            "N/A"
            if comparison.relative_difference is None
            else format_float(comparison.relative_difference)
        )
        lines.extend(
            [
                f"[{pass_text(comparison.passed)}] {comparison.label} "
                f"({comparison.unit})",
                f"  Step 5 value: {format_float(comparison.baseline)}",
                f"  Reproduced value: {format_float(comparison.reproduced)}",
                "  Absolute difference: "
                f"{format_float(comparison.absolute_difference)}",
                f"  Relative difference: {relative_difference}",
                "  Absolute tolerance: "
                f"{format_float(comparison.absolute_tolerance)}",
                "  Relative tolerance: "
                f"{format_float(comparison.relative_tolerance)}",
                "  Effective allowed difference: "
                f"{format_float(comparison.effective_tolerance)}",
            ]
        )
    material = execution.material_comparison
    lines.extend(
        [
            f"[{pass_text(material.passed)}] {material.label}",
            f"  Step 5 value: {material.baseline}",
            f"  Reproduced value: {material.reproduced}",
            "  Absolute difference: N/A",
            "  Relative difference: N/A",
            "  Tolerance: exact string equality",
            "",
            "Immutability checks",
            "-------------------",
        ]
    )
    for check in execution.immutability_checks:
        lines.append(
            f"[{pass_text(check.passed)}] {check.label}: {check.detail}"
        )
    lines.extend(["", "Source-file checks", "------------------"])
    for check in execution.source_checks:
        lines.append(
            f"[{pass_text(check.passed)}] {check.label}: {check.detail}"
        )
    if protected_pilot_checks:
        lines.extend(
            [
                "",
                "Protected AlNi Step 6B.2 pilot",
                "--------------------------------",
            ]
        )
        for check in protected_pilot_checks:
            lines.append(
                f"[{pass_text(check.passed)}] {check.label}: {check.detail}"
            )

    stress_passed = all(
        comparison.passed
        for comparison in execution.comparisons
        if comparison.label.startswith("Stress ")
    )
    volume_passed = all(
        comparison.passed
        for comparison in execution.comparisons
        if comparison.label in {"Volume", "Volume per atom"}
    )
    lines.extend(
        [
            "",
            "Execution boundary",
            "------------------",
            "Optimizer imported: No",
            "Optimizer created: No",
            "FIRE executed: No",
            "Relaxation executed: No",
            "Atomic positions deliberately changed: No",
            "Cell vectors deliberately changed: No",
            "Trajectory created: No",
            "Structure output created: No",
            "Formation energy calculated: No",
            "",
            "Phase status",
            "------------",
            f"Identity status: {pass_text(execution.identity_passed)}",
            f"Stress comparison status: {pass_text(stress_passed)}",
            f"Volume comparison status: {pass_text(volume_passed)}",
            "Immutability status: "
            f"{pass_text(execution.immutability_passed)}",
            "Source-file status: "
            f"{pass_text(execution.source_files_passed)}",
            "Phase reproducibility status: "
            f"{pass_text(execution.reproducibility_passed)}",
            "Overall reproducibility status: "
            f"{pass_text(execution.reproducibility_passed)}",
            "",
            "This status tests computational reproducibility only. It is not a "
            "scientific accuracy conclusion.",
            "",
            "Stored Step 5 reference excerpt",
            "-------------------------------",
            "Step 5 total energy: "
            f"{format_float(float(record['total_energy_eV']))} eV",
            "Step 5 energy per atom: "
            f"{format_float(float(record['energy_per_atom_eV']))} eV/atom",
            "Step 5 maximum force: "
            f"{format_float(float(record['maximum_force_eV_per_A']))} "
            "eV/angstrom",
        ]
    )
    return "\n".join(lines) + "\n"


def numeric_comparison_json(comparison: NumericComparison) -> dict[str, Any]:
    """Serialize one complete scalar comparison without losing semantics."""

    return {
        "label": comparison.label,
        "field_name": comparison.field_name,
        "unit": comparison.unit,
        "step5_value": comparison.baseline,
        "reproduced_value": comparison.reproduced,
        "absolute_difference": comparison.absolute_difference,
        "relative_difference": comparison.relative_difference,
        "absolute_tolerance": comparison.absolute_tolerance,
        "relative_tolerance": comparison.relative_tolerance,
        "effective_allowed_difference": comparison.effective_tolerance,
        "status": pass_text(comparison.passed),
    }


def phase_summary_record(execution: PhaseExecution) -> dict[str, Any]:
    """Build one machine-readable combined-summary record."""

    record = execution.baseline.record
    result = execution.result
    total_energy = comparison_by_label(execution, "Total energy")
    energy_per_atom = comparison_by_label(execution, "Energy per atom")
    maximum_force = comparison_by_label(
        execution, "Maximum atomic force magnitude"
    )
    rms_force = comparison_by_label(
        execution, "RMS atomic force magnitude"
    )
    stress_passed = all(
        comparison.passed
        for comparison in execution.comparisons
        if comparison.label.startswith("Stress ")
    )
    volume_passed = all(
        comparison.passed
        for comparison in execution.comparisons
        if comparison.label in {"Volume", "Volume per atom"}
    )
    return {
        "phase_key": execution.structure.phase.phase_key,
        "material_id": execution.structure.material_id,
        "number_of_atoms": result.number_of_atoms,
        "step5_total_energy_eV": float(record["total_energy_eV"]),
        "reproduced_total_energy_eV": result.total_energy_eV,
        "total_energy_absolute_difference_eV": (
            total_energy.absolute_difference
        ),
        "step5_energy_per_atom_eV": float(record["energy_per_atom_eV"]),
        "reproduced_energy_per_atom_eV": result.energy_per_atom_eV,
        "energy_per_atom_absolute_difference_eV": (
            energy_per_atom.absolute_difference
        ),
        "step5_maximum_force_eV_per_A": float(
            record["maximum_force_eV_per_A"]
        ),
        "reproduced_maximum_force_eV_per_A": (
            result.maximum_force_eV_per_A
        ),
        "maximum_force_absolute_difference_eV_per_A": (
            maximum_force.absolute_difference
        ),
        "step5_rms_force_eV_per_A": float(record["rms_force_eV_per_A"]),
        "reproduced_rms_force_eV_per_A": result.rms_force_eV_per_A,
        "rms_force_absolute_difference_eV_per_A": (
            rms_force.absolute_difference
        ),
        "stress_comparison_status": pass_text(stress_passed),
        "volume_comparison_status": pass_text(volume_passed),
        "identity_status": pass_text(execution.identity_passed),
        "immutability_status": pass_text(execution.immutability_passed),
        "source_file_status": pass_text(execution.source_files_passed),
        "phase_reproducibility_status": pass_text(
            execution.reproducibility_passed
        ),
        "comparisons": [
            numeric_comparison_json(comparison)
            for comparison in execution.comparisons
        ]
        + [
            {
                "label": execution.material_comparison.label,
                "field_name": "material_id",
                "step5_value": execution.material_comparison.baseline,
                "reproduced_value": execution.material_comparison.reproduced,
                "absolute_difference": None,
                "relative_difference": None,
                "tolerance": "exact string equality",
                "status": pass_text(execution.material_comparison.passed),
            }
        ],
    }


def build_combined_text_report(
    project_root: Path,
    config_path: Path,
    settings: ModelSettings,
    calculator_class: CalculatorIdentity,
    executions: Sequence[PhaseExecution],
    protected_pilot_checks: Sequence[ValidationCheck],
    generated_at_utc: str,
    calculator_loads: int,
    single_point_calculations: int,
) -> str:
    """Build the human-readable four-phase Step 6B.3 summary."""

    completed = [
        execution.structure.phase.phase_key
        for execution in executions
        if execution.reproducibility_passed
    ]
    failed = [
        execution.structure.phase.phase_key
        for execution in executions
        if not execution.reproducibility_passed
    ]
    overall_passed = (
        not failed
        and calculator_loads == 1
        and single_point_calculations == len(REMAINING_PHASES)
        and all(check.passed for check in protected_pilot_checks)
    )
    lines = [
        "Step 6B.3 - Remaining Ni-Al Baseline Reproduction Summary",
        "=========================================================",
        "",
        "Project and scope",
        "-----------------",
        f"Project root: {project_root}",
        f"Report timestamp (UTC): {generated_at_utc}",
        f"Configuration: {relative_path(config_path, project_root)}",
        "Requested phases: " + ", ".join(REMAINING_PHASES),
        "AlNi pilot included in batch: No",
        "Purpose: reproduce stored Step 5 fixed-geometry single-point values.",
        "Scientific accuracy conclusion: not evaluated in this sub-step.",
        "",
        "Model and execution",
        "-------------------",
        f"Model: {settings.family} {settings.name} {settings.value}",
        f"Device: {settings.device}",
        f"Default dtype: {settings.default_dtype}",
        f"Dispersion enabled: {str(settings.dispersion).lower()}",
        f"Calculator class: {calculator_class.qualified_name}",
        f"Calculator loads: {calculator_loads}",
        f"Single-point calculations: {single_point_calculations}",
        "Calculator reuse: one calculator reused across four independent "
        "original structures.",
        "",
        "Protected AlNi Step 6B.2 pilot",
        "--------------------------------",
    ]
    for check in protected_pilot_checks:
        lines.append(
            f"[{pass_text(check.passed)}] {check.label}: {check.detail}"
        )
    lines.extend(["", "Per-phase summary", "-----------------"])
    for execution in executions:
        record = execution.baseline.record
        result = execution.result
        energy = comparison_by_label(execution, "Total energy")
        max_force = comparison_by_label(
            execution, "Maximum atomic force magnitude"
        )
        rms_force = comparison_by_label(
            execution, "RMS atomic force magnitude"
        )
        stress_passed = all(
            comparison.passed
            for comparison in execution.comparisons
            if comparison.label.startswith("Stress ")
        )
        volume_passed = all(
            comparison.passed
            for comparison in execution.comparisons
            if comparison.label in {"Volume", "Volume per atom"}
        )
        lines.extend(
            [
                "",
                execution.structure.phase.phase_key,
                "~" * len(execution.structure.phase.phase_key),
                f"Material ID: {execution.structure.material_id}",
                f"Atom count: {result.number_of_atoms}",
                "Step 5 total energy: "
                f"{format_float(float(record['total_energy_eV']))} eV",
                "Reproduced total energy: "
                f"{format_float(result.total_energy_eV)} eV",
                "Energy absolute difference: "
                f"{format_float(energy.absolute_difference)} eV",
                "Step 5 energy per atom: "
                f"{format_float(float(record['energy_per_atom_eV']))} eV/atom",
                "Reproduced energy per atom: "
                f"{format_float(result.energy_per_atom_eV)} eV/atom",
                "Step 5 maximum force: "
                f"{format_float(float(record['maximum_force_eV_per_A']))} "
                "eV/angstrom",
                "Reproduced maximum force: "
                f"{format_float(result.maximum_force_eV_per_A)} eV/angstrom",
                "Maximum-force absolute difference: "
                f"{format_float(max_force.absolute_difference)} eV/angstrom",
                "Step 5 RMS force: "
                f"{format_float(float(record['rms_force_eV_per_A']))} "
                "eV/angstrom",
                "Reproduced RMS force: "
                f"{format_float(result.rms_force_eV_per_A)} eV/angstrom",
                "RMS-force absolute difference: "
                f"{format_float(rms_force.absolute_difference)} eV/angstrom",
                f"Stress comparison status: {pass_text(stress_passed)}",
                f"Volume comparison status: {pass_text(volume_passed)}",
                f"Identity status: {pass_text(execution.identity_passed)}",
                "Immutability status: "
                f"{pass_text(execution.immutability_passed)}",
                "Source-file status: "
                f"{pass_text(execution.source_files_passed)}",
                "Phase reproducibility status: "
                f"{pass_text(execution.reproducibility_passed)}",
            ]
        )
    lines.extend(
        [
            "",
            "Execution boundary",
            "------------------",
            "Optimizer imported: No",
            "Optimizer created: No",
            "FIRE executed: No",
            "Relaxation executed: No",
            "Trajectories created: No",
            "Structures written: No",
            "Formation energy calculated: No",
            "Raw MACE energies were not used to rank physical stability.",
            "",
            "Final Step 6B.3 status",
            "----------------------",
            f"Requested phases: {len(REMAINING_PHASES)}",
            f"Completed phases: {len(completed)}",
            f"Failed phases: {len(failed)}",
            "Completed phase keys: "
            + (", ".join(completed) if completed else "none"),
            "Failed phase keys: " + (", ".join(failed) if failed else "none"),
            f"Calculator loads: {calculator_loads}",
            f"Single-point calculations: {single_point_calculations}",
            "Optimizer created: No",
            "Relaxation executed: No",
            f"Overall Step 6B.3 status: {pass_text(overall_passed)}",
            "",
            "This status tests computational reproducibility only. It is not a "
            "scientific model-accuracy conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_combined_json(
    settings: ModelSettings,
    calculator_class: CalculatorIdentity,
    executions: Sequence[PhaseExecution],
    protected_pilot_snapshot: FileSnapshot,
    protected_pilot_checks: Sequence[ValidationCheck],
    generated_at_utc: str,
    calculator_loads: int,
    single_point_calculations: int,
) -> str:
    """Build the strict machine-readable four-phase combined table."""

    completed = [
        execution.structure.phase.phase_key
        for execution in executions
        if execution.reproducibility_passed
    ]
    failed = [
        execution.structure.phase.phase_key
        for execution in executions
        if not execution.reproducibility_passed
    ]
    overall_passed = (
        not failed
        and calculator_loads == 1
        and single_point_calculations == len(REMAINING_PHASES)
        and all(check.passed for check in protected_pilot_checks)
    )
    document = {
        "schema_version": "1.0",
        "project_step": "6B.3",
        "generated_at_utc": generated_at_utc,
        "description": (
            "Fixed-geometry reproduction of the four remaining Ni-Al Step 5 "
            "MACE zero-shot single-point baselines."
        ),
        "evaluation_type": "baseline reproduction single-point",
        "requested_phases": list(REMAINING_PHASES),
        "requested_phase_count": len(REMAINING_PHASES),
        "alni_pilot_included": False,
        "model": {
            "family": settings.family,
            "name": settings.name,
            "size": settings.value,
            "device": settings.device,
            "dtype": settings.default_dtype,
            "dispersion_enabled": settings.dispersion,
            "calculator_class": calculator_class.qualified_name,
        },
        "stress_component_order": list(STRESS_LABELS),
        "tolerances": {
            "total_energy": {
                "absolute_eV": ENERGY_TOLERANCE[0],
                "relative": ENERGY_TOLERANCE[1],
            },
            "energy_per_atom": {
                "absolute_eV_per_atom": ENERGY_PER_ATOM_TOLERANCE[0],
                "relative": ENERGY_PER_ATOM_TOLERANCE[1],
            },
            "force": {
                "absolute_eV_per_A": FORCE_TOLERANCE[0],
                "relative": FORCE_TOLERANCE[1],
            },
            "stress": {
                "absolute_eV_per_A3": STRESS_TOLERANCE[0],
                "relative": STRESS_TOLERANCE[1],
            },
            "volume": {
                "absolute_A3": VOLUME_TOLERANCE[0],
                "relative": VOLUME_TOLERANCE[1],
            },
            "immutability": {
                "absolute": IMMUTABILITY_ABSOLUTE_TOLERANCE,
                "relative": 0.0,
            },
            "identity_and_atom_count": "exact equality",
        },
        "protected_alni_step6b2_report": {
            "sha256": protected_pilot_snapshot.sha256,
            "size_bytes": protected_pilot_snapshot.size,
            "modification_time_ns": (
                protected_pilot_snapshot.modification_time_ns
            ),
            "status": pass_text(
                all(check.passed for check in protected_pilot_checks)
            ),
        },
        "records": [
            phase_summary_record(execution) for execution in executions
        ],
        "completed_phases": completed,
        "completed_phase_count": len(completed),
        "failed_phases": failed,
        "failed_phase_count": len(failed),
        "calculator_loads": calculator_loads,
        "single_point_calculations": single_point_calculations,
        "optimizer_created": False,
        "fire_executed": False,
        "relaxation_executed": False,
        "trajectories_created": False,
        "structures_written": False,
        "overall_status": pass_text(overall_passed),
        "accuracy_conclusion": (
            "Not evaluated; this artifact tests computational reproducibility "
            "only."
        ),
    }
    try:
        return (
            json.dumps(
                document,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise PublicationError(
            f"Could not serialize the Step 6B.3 JSON table: {exc}"
        ) from exc


def stage_output_files(
    staging_root: Path,
    contents: Mapping[Path, str],
) -> dict[Path, Path]:
    """Write and fsync every complete output before any target is changed."""

    staged: dict[Path, Path] = {}
    for index, (target, content) in enumerate(contents.items()):
        if not content.strip():
            raise PublicationError(
                f"Refusing to publish an empty output: {target}"
            )
        staged_path = staging_root / f"{index:02d}_{target.name}"
        try:
            with staged_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise PublicationError(
                f"Could not stage {target.name}: {type(exc).__name__}: {exc}"
            ) from exc
        if not staged_path.is_file() or staged_path.stat().st_size <= 0:
            raise PublicationError(
                f"Staging did not produce a complete output for {target}."
            )
        staged[target] = staged_path
    return staged


def publish_outputs_transactionally(
    contents: Mapping[Path, str],
    overwrite: bool,
    project_root: Path,
    final_validator: Callable[[], None] | None = None,
) -> None:
    """Publish all requested files or restore the complete prior state."""

    targets = tuple(contents)
    if not targets:
        raise PublicationError("No output targets were provided.")
    parent_directories = {target.parent for target in targets}
    if any(not directory.is_dir() for directory in parent_directories):
        missing = [
            directory for directory in parent_directories if not directory.is_dir()
        ]
        raise PublicationError(
            "Output directory does not exist: "
            + ", ".join(str(path) for path in missing)
        )
    staging_parent = min(parent_directories, key=str)
    staging_root: Path | None = None
    committed = False
    retain_for_recovery = False
    try:
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=".step6b_publication_",
                dir=staging_parent,
            )
        )
        staged = stage_output_files(staging_root, contents)
        expected = {
            target: (
                len(content.encode("utf-8")),
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            for target, content in contents.items()
        }
        validate_output_collisions(targets, overwrite, project_root)
        backups_root = staging_root / "backups"
        backups: dict[Path, Path] = {}
        if overwrite:
            for index, target in enumerate(targets):
                if not os.path.lexists(target):
                    continue
                backups_root.mkdir(parents=True, exist_ok=True)
                backup = backups_root / f"{index:02d}_{target.name}"
                original_size = target.stat().st_size
                original_hash = file_sha256(
                    target,
                    f"existing output {target.name}",
                    PublicationError,
                )
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
                if (
                    not backup.is_file()
                    or backup.stat().st_size != original_size
                    or file_sha256(
                        backup,
                        f"backup of {target.name}",
                        PublicationError,
                    )
                    != original_hash
                ):
                    raise PublicationError(
                        f"Could not verify the backup of {target}."
                    )
                backups[target] = backup

        published: list[Path] = []
        try:
            if overwrite:
                for target in targets:
                    os.replace(staged[target], target)
                    published.append(target)
            else:
                for target in targets:
                    os.link(staged[target], target)
                    published.append(target)
            for target in targets:
                expected_size, expected_hash = expected[target]
                if (
                    not target.is_file()
                    or target.stat().st_size != expected_size
                    or file_sha256(
                        target,
                        f"published output {target.name}",
                        PublicationError,
                    )
                    != expected_hash
                ):
                    raise PublicationError(
                        f"Published output could not be verified: {target}"
                    )
            if final_validator is not None:
                final_validator()
            committed = True
        except BaseException as exc:
            rollback_errors: list[str] = []
            for target in reversed(published):
                try:
                    backup = backups.get(target)
                    if backup is not None and backup.is_file():
                        os.replace(backup, target)
                    elif os.path.lexists(target):
                        target.unlink()
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"restore {target}: "
                        f"{type(rollback_exc).__name__}: {rollback_exc}"
                    )
            if rollback_errors:
                retain_for_recovery = True
                raise PublicationError(
                    "Transactional Step 6B publication failed and rollback "
                    "was incomplete. Inspect the targets and retained recovery "
                    f"directory {staging_root}. Original error: "
                    f"{type(exc).__name__}: {exc}; rollback errors: "
                    + "; ".join(rollback_errors),
                    publication_state="indeterminate",
                ) from exc
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise PublicationError(
                "Transactional Step 6B publication failed; the complete prior "
                "output state was restored "
                f"({type(exc).__name__}: {exc}).",
                publication_state="rolled_back",
            ) from exc
    except Step6BError:
        raise
    except OSError as exc:
        raise PublicationError(
            "Could not prepare or execute transactional output publication "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    finally:
        if (
            staging_root is not None
            and staging_root.exists()
            and not retain_for_recovery
        ):
            try:
                shutil.rmtree(staging_root)
            except OSError as cleanup_exc:
                state = "committed" if committed else "not committed"
                LOGGER.warning(
                    "Transactional outputs were %s, but temporary cleanup "
                    "failed for %s (%s: %s).",
                    state,
                    staging_root,
                    type(cleanup_exc).__name__,
                    cleanup_exc,
                )


def print_load_only_summary(
    project_root: Path,
    config_path: Path,
    settings: ModelSettings,
    identity: CalculatorIdentity,
) -> None:
    """Print the preserved Step 6B.1 no-structure result."""

    LOGGER.info("")
    LOGGER.info("=" * 76)
    LOGGER.info("STEP 6B.1 - MACE MODEL LOADING TEST (--load-only)")
    LOGGER.info("=" * 76)
    LOGGER.info("Project root                 : %s", project_root)
    LOGGER.info("Configuration path           : %s", config_path)
    LOGGER.info("Model family                 : %s", settings.family)
    LOGGER.info("Model name                   : %s", settings.name)
    LOGGER.info("Model value                  : %s", settings.value)
    LOGGER.info("Device                       : %s", settings.device)
    LOGGER.info("Default dtype                : %s", settings.default_dtype)
    LOGGER.info("Dispersion                   : %s", str(settings.dispersion).lower())
    LOGGER.info("Calculator class             : %s", identity.qualified_name)
    LOGGER.info("Calculator load count        : 1")
    LOGGER.info("-" * 76)
    LOGGER.info("Structures read              : No")
    LOGGER.info("Calculator attached to atoms : No")
    LOGGER.info("Physical properties requested: No")
    LOGGER.info("Single-point calculations    : 0")
    LOGGER.info("Optimizer created            : No")
    LOGGER.info("Relaxation executed          : No")
    LOGGER.info("Repository outputs created   : No")
    LOGGER.info("Overall status               : SUCCESS")
    LOGGER.info("=" * 76)


def print_single_phase_summary(
    project_root: Path,
    paths: PhasePaths,
    execution: PhaseExecution,
    calculator_loads: int,
    single_point_calculations: int,
) -> None:
    """Print a concise but complete single-phase result."""

    phase = execution.structure.phase
    energy = comparison_by_label(execution, "Total energy")
    maximum_force = comparison_by_label(
        execution, "Maximum atomic force magnitude"
    )
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info(
        "%s - %s BASELINE REPRODUCTION",
        phase.step_label.upper(),
        phase.phase_key,
    )
    LOGGER.info("=" * 78)
    LOGGER.info("Phase                                : %s", phase.phase_key)
    LOGGER.info("Material ID                          : %s", phase.material_id)
    LOGGER.info("Atom count                           : %d", phase.atom_count)
    LOGGER.info(
        "Step 5 total energy                  : %s eV",
        format_float(energy.baseline),
    )
    LOGGER.info(
        "Reproduced total energy              : %s eV",
        format_float(energy.reproduced),
    )
    LOGGER.info(
        "Energy absolute difference           : %s eV",
        format_float(energy.absolute_difference),
    )
    LOGGER.info(
        "Maximum force absolute difference    : %s eV/angstrom",
        format_float(maximum_force.absolute_difference),
    )
    LOGGER.info(
        "Identity status                      : %s",
        pass_text(execution.identity_passed),
    )
    LOGGER.info(
        "Immutability status                  : %s",
        pass_text(execution.immutability_passed),
    )
    LOGGER.info(
        "Source-file status                   : %s",
        pass_text(execution.source_files_passed),
    )
    LOGGER.info("Calculator loads                     : %d", calculator_loads)
    LOGGER.info(
        "Single-point calculations            : %d",
        single_point_calculations,
    )
    LOGGER.info("Optimizer created                    : No")
    LOGGER.info("Relaxation executed                  : No")
    LOGGER.info(
        "Output report                        : %s",
        relative_path(paths.report, project_root),
    )
    LOGGER.info(
        "Phase reproducibility status         : %s",
        pass_text(execution.reproducibility_passed),
    )
    LOGGER.info("=" * 78)


def print_batch_summary(
    project_root: Path,
    layout: RepositoryLayout,
    executions: Sequence[PhaseExecution],
    protected_pilot_checks: Sequence[ValidationCheck],
    calculator_loads: int,
    single_point_calculations: int,
) -> None:
    """Print the required four-phase batch counts and statuses."""

    completed = [
        execution for execution in executions if execution.reproducibility_passed
    ]
    failed = [
        execution for execution in executions if not execution.reproducibility_passed
    ]
    overall_passed = (
        not failed
        and calculator_loads == 1
        and single_point_calculations == len(REMAINING_PHASES)
        and all(check.passed for check in protected_pilot_checks)
    )
    LOGGER.info("")
    LOGGER.info("=" * 82)
    LOGGER.info("STEP 6B.3 - REMAINING NI-AL BASELINE REPRODUCTION")
    LOGGER.info("=" * 82)
    LOGGER.info("Project root                          : %s", project_root)
    LOGGER.info("Requested phases                     : %s", ", ".join(REMAINING_PHASES))
    LOGGER.info("Requested phase count                : %d", len(REMAINING_PHASES))
    LOGGER.info("AlNi processed in batch              : No")
    for execution in executions:
        energy = comparison_by_label(execution, "Total energy")
        max_force = comparison_by_label(
            execution, "Maximum atomic force magnitude"
        )
        LOGGER.info(
            "  %-7s %s | E diff=%s eV | max-F diff=%s eV/angstrom",
            execution.structure.phase.phase_key,
            pass_text(execution.reproducibility_passed),
            format_float(energy.absolute_difference),
            format_float(max_force.absolute_difference),
        )
    LOGGER.info("Completed phases                      : %d", len(completed))
    LOGGER.info("Failed phases                         : %d", len(failed))
    LOGGER.info("Calculator loads                      : %d", calculator_loads)
    LOGGER.info(
        "Single-point calculations             : %d",
        single_point_calculations,
    )
    LOGGER.info("Optimizer created                     : No")
    LOGGER.info("FIRE executed                         : No")
    LOGGER.info("Relaxation executed                   : No")
    LOGGER.info(
        "Protected AlNi pilot unchanged        : %s",
        pass_text(all(check.passed for check in protected_pilot_checks)),
    )
    LOGGER.info(
        "Combined text report                  : %s",
        relative_path(layout.batch_summary_report, project_root),
    )
    LOGGER.info(
        "Combined JSON table                   : %s",
        relative_path(layout.batch_json_table, project_root),
    )
    LOGGER.info(
        "Overall Step 6B.3 status              : %s",
        pass_text(overall_passed),
    )
    LOGGER.info("=" * 82)
    LOGGER.info(
        "This result tests computational reproducibility, not scientific accuracy."
    )


def run(arguments: Sequence[str] | None = None) -> int:
    """Run load-only, one explicit phase, or the four-phase batch."""

    options = parse_arguments(arguments)
    configure_logging(options.verbose)
    project_root = locate_project_root()
    workflow_name = "Step 6B"
    try:
        validate_runtime_environment(project_root)
        requested_phases = choose_requested_phases(options)
        config_path = resolve_config_path(project_root, options.config)
        LOGGER.debug("Resolved project root: %s", project_root)
        LOGGER.debug("Resolved configuration path: %s", config_path)
        config = read_json_object(
            config_path,
            "relaxation configuration",
            ConfigurationError,
        )
        settings = validate_model_settings(config)

        if options.load_only:
            mace_dependencies = import_mace_dependencies()
            calculator = load_calculator_once(settings, mace_dependencies)
            identity = calculator_identity(calculator)
            print_load_only_summary(
                project_root,
                config_path,
                settings,
                identity,
            )
            return 0

        workflow_name = (
            "Step 6B.3"
            if options.all_remaining
            or requested_phases[0].phase_key != ALNI_PILOT_PHASE
            else "Step 6B.2"
        )
        layout = validate_repository_layout(config, project_root)
        targets = target_paths(
            requested_phases, layout, options.all_remaining
        )
        if (
            options.overwrite
            and layout.protected_alni_report in targets
        ):
            raise PhaseScopeError(
                "The AlNi Step 6B.2 pilot report is protected from "
                "--overwrite in every execution mode."
            )
        validate_output_collisions(
            targets, options.overwrite, project_root
        )

        protected_pilot_snapshot: FileSnapshot | None = None
        if all(not phase.is_alni_pilot for phase in requested_phases):
            protected_pilot_snapshot = validate_protected_alni_report(layout)

        scientific_dependencies = import_scientific_dependencies()
        baseline_table = load_baseline_table(layout, settings)
        validated: list[tuple[PhaseDefinition, PhasePaths, ValidatedStructure, BaselineData]] = []
        for phase in requested_phases:
            paths = paths_for_phase(layout, phase)
            structure = validate_selected_structure(
                phase, paths, project_root, scientific_dependencies
            )
            baseline = validate_phase_baseline(
                phase,
                paths,
                project_root,
                settings,
                structure,
                baseline_table,
                scientific_dependencies,
            )
            validated.append((phase, paths, structure, baseline))

        mace_dependencies = import_mace_dependencies()
        calculator = load_calculator_once(settings, mace_dependencies)
        calculator_loads = 1
        identity = calculator_identity(calculator)
        single_point_calculations = 0
        executions: list[PhaseExecution] = []
        for index, (_, _, structure, baseline) in enumerate(validated):
            result, immutability_checks = evaluate_single_point(
                structure,
                calculator,
                scientific_dependencies,
            )
            single_point_calculations += 1
            comparisons, material_comparison = build_comparisons(
                baseline, result, structure.material_id
            )
            source_checks = verify_phase_source_files(structure, baseline)
            executions.append(
                build_phase_execution(
                    structure,
                    baseline,
                    result,
                    comparisons,
                    material_comparison,
                    immutability_checks,
                    source_checks,
                )
            )
            if index < len(validated) - 1:
                reset_calculator_between_phases(
                    calculator, structure.phase.phase_key
                )

        if calculator_loads != 1:
            raise CalculatorValidationError(
                f"Calculator load count is {calculator_loads}; expected 1."
            )
        if single_point_calculations != len(requested_phases):
            raise CalculationError(
                "Single-point calculation count does not match the requested "
                f"phase count: {single_point_calculations} versus "
                f"{len(requested_phases)}."
            )

        endpoint_executions: list[PhaseExecution] = []
        for execution in executions:
            endpoint_source_checks = verify_phase_source_files(
                execution.structure,
                execution.baseline,
            )
            endpoint_executions.append(
                build_phase_execution(
                    execution.structure,
                    execution.baseline,
                    execution.result,
                    execution.comparisons,
                    execution.material_comparison,
                    execution.immutability_checks,
                    endpoint_source_checks,
                )
            )
        executions = endpoint_executions

        protected_pilot_checks: tuple[ValidationCheck, ...] = ()
        if protected_pilot_snapshot is not None:
            protected_pilot_checks = verify_file_snapshot(
                protected_pilot_snapshot
            )
        generated_at_utc = utc_timestamp()
        contents: dict[Path, str] = {}
        for (_, paths, _, _), execution in zip(
            validated, executions, strict=True
        ):
            contents[paths.report] = build_individual_report(
                project_root,
                config_path,
                settings,
                paths,
                identity,
                execution,
                protected_pilot_checks,
                generated_at_utc,
                options.all_remaining,
            )
        if options.all_remaining:
            if protected_pilot_snapshot is None:
                raise ConfigurationError(
                    "Batch mode did not capture the protected AlNi report."
                )
            contents[layout.batch_summary_report] = build_combined_text_report(
                project_root,
                config_path,
                settings,
                identity,
                executions,
                protected_pilot_checks,
                generated_at_utc,
                calculator_loads,
                single_point_calculations,
            )
            contents[layout.batch_json_table] = build_combined_json(
                settings,
                identity,
                executions,
                protected_pilot_snapshot,
                protected_pilot_checks,
                generated_at_utc,
                calculator_loads,
                single_point_calculations,
            )

        def validate_read_only_endpoint() -> None:
            failures: list[str] = []
            for execution in executions:
                for check in verify_phase_source_files(
                    execution.structure,
                    execution.baseline,
                ):
                    if not check.passed:
                        failures.append(
                            f"{execution.structure.phase.phase_key}: "
                            f"{check.label}"
                        )
            if protected_pilot_snapshot is not None:
                for check in verify_file_snapshot(
                    protected_pilot_snapshot
                ):
                    if not check.passed:
                        failures.append(check.label)
            if failures:
                raise PublicationError(
                    "Read-only endpoint validation failed: "
                    + "; ".join(failures)
                )

        publish_outputs_transactionally(
            contents,
            options.overwrite,
            project_root,
            final_validator=validate_read_only_endpoint,
        )

        overall_passed = all(
            execution.reproducibility_passed for execution in executions
        ) and all(check.passed for check in protected_pilot_checks)
        if options.all_remaining:
            print_batch_summary(
                project_root,
                layout,
                executions,
                protected_pilot_checks,
                calculator_loads,
                single_point_calculations,
            )
        else:
            print_single_phase_summary(
                project_root,
                validated[0][1],
                executions[0],
                calculator_loads,
                single_point_calculations,
            )
        return 0 if overall_passed else 1
    except Step6BError as exc:
        LOGGER.error("%s failed (%s): %s", workflow_name, type(exc).__name__, exc)
        LOGGER.info("Optimizer created            : No")
        LOGGER.info("FIRE executed                : No")
        LOGGER.info("Relaxation executed          : No")
        if isinstance(exc, PublicationError):
            LOGGER.info(
                "Publication state             : %s",
                exc.publication_state,
            )
            LOGGER.info(
                "New outputs remain published  : %s",
                (
                    "Unknown; inspect reported targets and recovery directory"
                    if exc.publication_state == "indeterminate"
                    else "No"
                ),
            )
        else:
            LOGGER.info("New outputs published        : No")
        return 1
    except KeyboardInterrupt:
        LOGGER.error("%s was interrupted by the user.", workflow_name)
        return 130


def main() -> None:
    """Propagate the controlled workflow status to the calling shell."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()

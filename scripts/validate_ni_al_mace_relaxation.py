"""Validate the preparation for controlled Ni-Al geometry relaxation.

Step 6A is deliberately limited to configuration, structure, provenance, and
output-plan validation.  It never imports or constructs a MACE calculator,
never requests model properties, and never changes atoms or cell vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("validate_ni_al_mace_relaxation")

SCHEMA_VERSION = "1.0"
DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/mace_relaxation.json")
EXPECTED_PHASE_ORDER = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
EXPECTED_MATERIAL_IDS = {
    "Al3Ni": "mp-622209",
    "Al3Ni2": "mp-1057",
    "AlNi": "mp-1487",
    "Al3Ni5": "mp-16514",
    "AlNi3": "mp-2593",
}
ALLOWED_ELEMENTS = frozenset({"Al", "Ni"})
VALID_MODES = ("atomic_only", "full_cell", "all")
VALID_DTYPES = frozenset({"float32", "float64"})
BASELINE_EVALUATION_TYPE = "zero-shot single-point"

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
STRESS_BASELINE_FIELDS = (
    "stress_xx_eV_per_A3",
    "stress_yy_eV_per_A3",
    "stress_zz_eV_per_A3",
    "stress_yz_eV_per_A3",
    "stress_xz_eV_per_A3",
    "stress_xy_eV_per_A3",
)


class Step6AError(RuntimeError):
    """Base class for readable Step 6A validation failures."""


class DependencyError(Step6AError):
    """Raised when the project Python environment is not suitable."""


class ConfigurationError(Step6AError):
    """Raised when the relaxation configuration is invalid."""


class InputValidationError(Step6AError):
    """Raised when a selected structure or its metadata is invalid."""


class BaselineValidationError(Step6AError):
    """Raised when the Step 5 single-point baseline is inconsistent."""


class DirectoryPlanError(Step6AError):
    """Raised when the planned empty directory tree is unsafe or inconsistent."""


class DuplicateJsonKeyError(ValueError):
    """Internal signal used to reject ambiguous duplicate JSON keys."""


@dataclass(frozen=True)
class CommandLineOptions:
    """Validated command-line selections for one preparation run."""

    config: Path
    phase: str | None
    mode: str
    create_directories: bool
    verbose: bool


@dataclass(frozen=True)
class ScientificDependencies:
    """Non-calculator scientific utilities needed for input validation."""

    numpy: Any
    ase_read: Callable[..., Any]
    composition_class: type[Any]


@dataclass(frozen=True)
class ModelSettings:
    """Model identity planned for later relaxation stages."""

    family: str
    name: str
    value: str
    device: str
    default_dtype: str
    dispersion: bool


@dataclass(frozen=True)
class RelaxationModeSettings:
    """Validated controls for one future relaxation mode."""

    name: str
    enabled: bool
    optimizer: str
    force_threshold_eV_per_A: float
    stress_threshold_eV_per_A3: float | None
    maximum_steps: int
    trajectory_interval: int
    allow_atomic_positions: bool
    allow_cell_shape: bool
    allow_cell_volume: bool
    hydrostatic_strain: bool | None = None
    constant_volume: bool | None = None


@dataclass(frozen=True)
class SafetySettings:
    """Limits that later stages must enforce while relaxation is running."""

    maximum_absolute_volume_change_percent: float
    maximum_atomic_displacement_A: float
    stop_on_nonfinite_value: bool
    preserve_original_structure: bool
    require_periodic_cell: bool


@dataclass(frozen=True)
class RelaxationConfiguration:
    """Complete validated and repository-resolved Step 6 configuration."""

    schema_version: str
    description: str
    model: ModelSettings
    selected_structure_directory: Path
    zero_shot_table: Path
    root_output_directory: Path
    atomic_only_output_directory: Path
    full_cell_output_directory: Path
    comparison_output_directory: Path
    phase_order: tuple[str, ...]
    expected_material_ids: Mapping[str, str]
    atomic_only: RelaxationModeSettings
    full_cell: RelaxationModeSettings
    safety: SafetySettings


@dataclass(frozen=True)
class StructureValidation:
    """Validated facts read without modifying one selected structure."""

    material_id: str
    formula: str
    number_of_atoms: int
    space_group: str
    structure_path: Path


@dataclass(frozen=True)
class PhaseValidationOutcome:
    """Per-phase status used by the complete console summary."""

    phase_key: str
    material_id: str | None
    formula: str | None
    number_of_atoms: int | None
    space_group: str | None
    baseline_status: str
    structure_path: Path
    valid: bool
    error_type: str | None = None
    error_message: str | None = None


def locate_project_root() -> Path:
    """Locate the repository from this script, independent of the shell."""

    return Path(__file__).resolve().parents[1]


def parse_arguments(arguments: Sequence[str] | None = None) -> CommandLineOptions:
    """Parse the documented Step 6A command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the configuration, selected structures, and Step 5 "
            "baseline for later controlled Ni-Al geometry relaxation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_RELATIVE_PATH,
        help=(
            "Relaxation configuration path. Relative paths are resolved from "
            "the repository root (default: configs/mace_relaxation.json)."
        ),
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Validate one exact phase key; omit to validate all five phases.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="all",
        help="Select the planned mode to report (default: all).",
    )
    parser.add_argument(
        "--create-directories",
        action="store_true",
        help="Create only the empty planned Step 6 output directory tree.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed validation logging.",
    )
    parsed = parser.parse_args(arguments)
    return CommandLineOptions(
        config=parsed.config,
        phase=parsed.phase,
        mode=parsed.mode,
        create_directories=parsed.create_directories,
        verbose=parsed.verbose,
    )


def configure_logging(verbose: bool) -> None:
    """Configure concise workflow logging without third-party debug noise."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False
    logging.getLogger().setLevel(logging.WARNING)


def validate_runtime(project_root: Path) -> None:
    """Require the repository's Python 3.11 virtual environment."""

    expected_prefix = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    if os.path.normcase(str(active_prefix)) != os.path.normcase(
        str(expected_prefix)
    ):
        raise DependencyError(
            "Run this script with the project interpreter: "
            ".\\.venv\\Scripts\\python.exe"
        )
    if sys.version_info[:2] != (3, 11):
        raise DependencyError(
            "Step 6A requires Python 3.11; the active interpreter is "
            f"Python {sys.version_info.major}.{sys.version_info.minor}."
        )


def import_scientific_dependencies() -> ScientificDependencies:
    """Import structure-validation packages without importing any calculator."""

    try:
        import numpy as np
        from ase.io import read as ase_read
        from pymatgen.core import Composition
    except ImportError as exc:
        missing = exc.name or "an unknown package"
        raise DependencyError(
            f"Required project dependency is missing: {missing}."
        ) from exc
    return ScientificDependencies(
        numpy=np,
        ase_read=ase_read,
        composition_class=Composition,
    )


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys so configuration meaning cannot be ambiguous."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_nonstandard_json_constant(value: str) -> None:
    """Reject JSON NaN and Infinity values that Python otherwise accepts."""

    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def read_json_object(
    path: Path,
    label: str,
    error_class: type[Step6AError],
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
        raise error_class(f"Invalid JSON in {label}: {detail}") from exc
    except OSError as exc:
        raise error_class(f"Could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_class(f"{label} root must be a JSON object: {path}")
    return value


def require_mapping(parent: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    """Require one named configuration section to be an object."""

    value = parent.get(field)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration field '{field}' must be an object.")
    return value


def require_nonempty_string(parent: Mapping[str, Any], field: str) -> str:
    """Require a nonempty string and return its normalized value."""

    value = parent.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"Configuration field '{field}' must be a non-empty string."
        )
    return value.strip()


def require_boolean(parent: Mapping[str, Any], field: str) -> bool:
    """Require an actual JSON Boolean rather than a truthy substitute."""

    value = parent.get(field)
    if not isinstance(value, bool):
        raise ConfigurationError(
            f"Configuration field '{field}' must be true or false."
        )
    return value


def require_positive_number(parent: Mapping[str, Any], field: str) -> float:
    """Require a finite positive scientific threshold or safety limit."""

    value = parent.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"Configuration field '{field}' must be a positive number."
        )
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(
            f"Configuration field '{field}' must be finite and greater than zero."
        )
    return converted


def require_positive_integer(parent: Mapping[str, Any], field: str) -> int:
    """Require a positive integral step count or trajectory interval."""

    value = parent.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(
            f"Configuration field '{field}' must be a positive integer."
        )
    return value


def resolve_repository_path(
    project_root: Path,
    raw_value: str,
    field: str,
) -> Path:
    """Resolve a configured path and prevent it from escaping the repository."""

    supplied = Path(raw_value.replace("/", os.sep))
    resolved = supplied.resolve() if supplied.is_absolute() else (
        project_root / supplied
    ).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ConfigurationError(
            f"Configuration field '{field}' must resolve inside the repository: "
            f"{resolved}"
        ) from exc
    return resolved


def require_repository_path(
    parent: Mapping[str, Any],
    field: str,
    project_root: Path,
) -> Path:
    """Read a nonempty configuration path and constrain it to the project."""

    return resolve_repository_path(
        project_root,
        require_nonempty_string(parent, field),
        field,
    )


def validate_model(raw: Mapping[str, Any]) -> ModelSettings:
    """Validate the identity and numerical settings planned for later work."""

    model = require_mapping(raw, "model")
    family = require_nonempty_string(model, "family")
    name = require_nonempty_string(model, "name")
    value = require_nonempty_string(model, "value")
    device = require_nonempty_string(model, "device")
    dtype = require_nonempty_string(model, "default_dtype")
    dispersion = require_boolean(model, "dispersion")
    if family != "MACE":
        raise ConfigurationError("Configuration field 'model.family' must be 'MACE'.")
    if dtype not in VALID_DTYPES:
        raise ConfigurationError(
            "Configuration field 'model.default_dtype' must be float32 or float64."
        )
    return ModelSettings(family, name, value, device, dtype, dispersion)


def validate_phase_scope(
    raw: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Keep the configured scope aligned with the five selected structures."""

    phase_order_value = raw.get("phase_order")
    if not isinstance(phase_order_value, list) or not phase_order_value:
        raise ConfigurationError("Configuration field 'phase_order' must be an array.")
    if not all(isinstance(value, str) and value.strip() for value in phase_order_value):
        raise ConfigurationError("Every phase_order entry must be a non-empty string.")
    phase_order = tuple(str(value).strip() for value in phase_order_value)
    if len(set(phase_order)) != len(phase_order):
        raise ConfigurationError("Configuration field 'phase_order' has duplicates.")
    if phase_order != EXPECTED_PHASE_ORDER:
        raise ConfigurationError(
            "Configuration phase_order must match the selected project phases: "
            + ", ".join(EXPECTED_PHASE_ORDER)
        )

    expected = require_mapping(raw, "expected_phases")
    if set(expected) != set(phase_order):
        raise ConfigurationError(
            "Configuration expected_phases keys must exactly match phase_order."
        )
    material_ids: dict[str, str] = {}
    for phase in phase_order:
        phase_settings = expected.get(phase)
        if not isinstance(phase_settings, dict):
            raise ConfigurationError(
                f"Configuration expected_phases.{phase} must be an object."
            )
        material_id = require_nonempty_string(
            phase_settings, "expected_material_id"
        )
        if material_id != EXPECTED_MATERIAL_IDS[phase]:
            raise ConfigurationError(
                f"Expected Materials Project ID for {phase} must be "
                f"{EXPECTED_MATERIAL_IDS[phase]}, not {material_id}."
            )
        material_ids[phase] = material_id
    return phase_order, material_ids


def validate_atomic_only_mode(raw_modes: Mapping[str, Any]) -> RelaxationModeSettings:
    """Ensure the fixed-cell mode can move atoms but cannot change the cell."""

    mode = require_mapping(raw_modes, "atomic_only")
    settings = RelaxationModeSettings(
        name="atomic_only",
        enabled=require_boolean(mode, "enabled"),
        optimizer=require_nonempty_string(mode, "optimizer"),
        force_threshold_eV_per_A=require_positive_number(
            mode, "force_threshold_eV_per_A"
        ),
        stress_threshold_eV_per_A3=None,
        maximum_steps=require_positive_integer(mode, "maximum_steps"),
        trajectory_interval=require_positive_integer(mode, "trajectory_interval"),
        allow_atomic_positions=require_boolean(mode, "allow_atomic_positions"),
        allow_cell_shape=require_boolean(mode, "allow_cell_shape"),
        allow_cell_volume=require_boolean(mode, "allow_cell_volume"),
    )
    if not settings.allow_atomic_positions:
        raise ConfigurationError("atomic_only must allow atomic-position changes.")
    if settings.allow_cell_shape or settings.allow_cell_volume:
        raise ConfigurationError(
            "atomic_only must prohibit both cell-shape and cell-volume changes."
        )
    return settings


def validate_full_cell_mode(raw_modes: Mapping[str, Any]) -> RelaxationModeSettings:
    """Ensure the full-cell mode permits coupled atomic and cell response."""

    mode = require_mapping(raw_modes, "full_cell")
    settings = RelaxationModeSettings(
        name="full_cell",
        enabled=require_boolean(mode, "enabled"),
        optimizer=require_nonempty_string(mode, "optimizer"),
        force_threshold_eV_per_A=require_positive_number(
            mode, "force_threshold_eV_per_A"
        ),
        stress_threshold_eV_per_A3=require_positive_number(
            mode, "stress_threshold_eV_per_A3"
        ),
        maximum_steps=require_positive_integer(mode, "maximum_steps"),
        trajectory_interval=require_positive_integer(mode, "trajectory_interval"),
        allow_atomic_positions=require_boolean(mode, "allow_atomic_positions"),
        allow_cell_shape=require_boolean(mode, "allow_cell_shape"),
        allow_cell_volume=require_boolean(mode, "allow_cell_volume"),
        hydrostatic_strain=require_boolean(mode, "hydrostatic_strain"),
        constant_volume=require_boolean(mode, "constant_volume"),
    )
    if not all(
        (
            settings.allow_atomic_positions,
            settings.allow_cell_shape,
            settings.allow_cell_volume,
        )
    ):
        raise ConfigurationError(
            "full_cell must allow atomic positions, cell shape, and cell volume."
        )
    if settings.constant_volume:
        raise ConfigurationError(
            "full_cell constant_volume must be false when volume changes are allowed."
        )
    return settings


def validate_safety(raw: Mapping[str, Any]) -> SafetySettings:
    """Validate hard limits that protect future optimization runs."""

    safety = require_mapping(raw, "safety")
    settings = SafetySettings(
        maximum_absolute_volume_change_percent=require_positive_number(
            safety, "maximum_absolute_volume_change_percent"
        ),
        maximum_atomic_displacement_A=require_positive_number(
            safety, "maximum_atomic_displacement_A"
        ),
        stop_on_nonfinite_value=require_boolean(safety, "stop_on_nonfinite_value"),
        preserve_original_structure=require_boolean(
            safety, "preserve_original_structure"
        ),
        require_periodic_cell=require_boolean(safety, "require_periodic_cell"),
    )
    if not settings.stop_on_nonfinite_value:
        raise ConfigurationError("safety.stop_on_nonfinite_value must be true.")
    if not settings.preserve_original_structure:
        raise ConfigurationError("safety.preserve_original_structure must be true.")
    if not settings.require_periodic_cell:
        raise ConfigurationError("safety.require_periodic_cell must be true.")
    return settings


def validate_configuration(
    raw: Mapping[str, Any],
    project_root: Path,
) -> RelaxationConfiguration:
    """Validate every configuration section before any directory is created."""

    schema_version = require_nonempty_string(raw, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}."
        )
    description = require_nonempty_string(raw, "description")
    model = validate_model(raw)
    phase_order, material_ids = validate_phase_scope(raw)
    modes = require_mapping(raw, "relaxation_modes")
    atomic_only = validate_atomic_only_mode(modes)
    full_cell = validate_full_cell_mode(modes)
    safety = validate_safety(raw)

    inputs = require_mapping(raw, "input")
    selected_directory = require_repository_path(
        inputs, "selected_structure_directory", project_root
    )
    zero_shot_table = require_repository_path(
        inputs, "zero_shot_table", project_root
    )
    outputs = require_mapping(raw, "output")
    root_output = require_repository_path(outputs, "root_directory", project_root)
    atomic_output = require_repository_path(
        outputs, "atomic_only_directory", project_root
    )
    full_output = require_repository_path(outputs, "full_cell_directory", project_root)
    comparison_output = require_repository_path(
        outputs, "comparison_directory", project_root
    )

    child_outputs = (atomic_output, full_output, comparison_output)
    if len({root_output, *child_outputs}) != 4:
        raise ConfigurationError("All configured output directories must be distinct.")
    for directory in child_outputs:
        try:
            relative = directory.relative_to(root_output)
        except ValueError as exc:
            raise ConfigurationError(
                f"Output directory must be inside root_directory: {directory}"
            ) from exc
        if not relative.parts:
            raise ConfigurationError("An output mode directory cannot equal the root.")
    if root_output.is_relative_to(selected_directory) or selected_directory.is_relative_to(
        root_output
    ):
        raise ConfigurationError("Input structures and relaxation outputs must not overlap.")
    if root_output.is_relative_to(zero_shot_table.parent):
        raise ConfigurationError("Relaxation outputs must not overlap Step 5 tables.")

    return RelaxationConfiguration(
        schema_version=schema_version,
        description=description,
        model=model,
        selected_structure_directory=selected_directory,
        zero_shot_table=zero_shot_table,
        root_output_directory=root_output,
        atomic_only_output_directory=atomic_output,
        full_cell_output_directory=full_output,
        comparison_output_directory=comparison_output,
        phase_order=phase_order,
        expected_material_ids=material_ids,
        atomic_only=atomic_only,
        full_cell=full_cell,
        safety=safety,
    )


def choose_requested_phases(
    phase_order: tuple[str, ...],
    requested_phase: str | None,
) -> tuple[str, ...]:
    """Select one exact phase or retain the configured five-phase order."""

    if requested_phase is None:
        return phase_order
    if requested_phase not in phase_order:
        raise ConfigurationError(
            f"Unknown --phase value {requested_phase!r}. Valid phases: "
            + ", ".join(phase_order)
        )
    return (requested_phase,)


def validate_requested_mode(
    requested_mode: str,
    config: RelaxationConfiguration,
) -> None:
    """Reject a requested future mode if it is disabled in configuration."""

    if requested_mode in {"atomic_only", "all"} and not config.atomic_only.enabled:
        raise ConfigurationError("The requested atomic_only mode is disabled.")
    if requested_mode in {"full_cell", "all"} and not config.full_cell.enabled:
        raise ConfigurationError("The requested full_cell mode is disabled.")


def file_sha256(path: Path) -> str:
    """Hash an input so read-only validation can prove it was preserved."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise InputValidationError(f"Could not hash input structure {path}: {exc}") from exc
    return digest.hexdigest()


def composition_signature(composition: Any) -> tuple[tuple[str, float], ...]:
    """Create an element-order-independent reduced-composition signature."""

    reduced = composition.reduced_composition
    return tuple(
        sorted(
            (str(symbol), float(amount))
            for symbol, amount in reduced.get_el_amt_dict().items()
        )
    )


def require_metadata_string(
    metadata: Mapping[str, Any],
    field: str,
    phase_key: str,
) -> str:
    """Require a provenance string needed to identify a selected structure."""

    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(
            f"Metadata for {phase_key} requires non-empty '{field}'."
        )
    return value.strip()


def validate_selected_structure(
    phase_key: str,
    config: RelaxationConfiguration,
    dependencies: ScientificDependencies,
) -> StructureValidation:
    """Validate one selected EXTXYZ and metadata pair without changing either."""

    structure_path = config.selected_structure_directory / f"{phase_key}.extxyz"
    metadata_path = (
        config.selected_structure_directory / f"{phase_key}.metadata.json"
    )
    if not structure_path.is_file():
        raise InputValidationError(
            f"Selected EXTXYZ does not exist for {phase_key}: {structure_path}"
        )
    if not metadata_path.is_file():
        raise InputValidationError(
            f"Selected metadata does not exist for {phase_key}: {metadata_path}"
        )

    # Hashing before and after ASE parsing makes the read-only promise testable,
    # not merely an assumption about the file-reading library.
    before_hash = file_sha256(structure_path)
    before_stat = structure_path.stat()
    metadata = read_json_object(
        metadata_path,
        f"metadata for {phase_key}",
        InputValidationError,
    )
    metadata_phase = require_metadata_string(metadata, "phase_key", phase_key)
    material_id = require_metadata_string(metadata, "material_id", phase_key)
    formula = require_metadata_string(metadata, "formula_pretty", phase_key)
    symmetry_symbol = require_metadata_string(metadata, "symmetry_symbol", phase_key)
    symmetry_number = metadata.get("symmetry_number")
    if metadata_phase != phase_key:
        raise InputValidationError(
            f"Metadata phase mismatch for {phase_key}: {metadata_phase!r}."
        )
    if material_id != config.expected_material_ids[phase_key]:
        raise InputValidationError(
            f"Metadata material ID mismatch for {phase_key}: expected "
            f"{config.expected_material_ids[phase_key]}, found {material_id}."
        )
    if (
        not isinstance(symmetry_number, int)
        or isinstance(symmetry_number, bool)
        or symmetry_number <= 0
    ):
        raise InputValidationError(
            f"Metadata for {phase_key} has invalid symmetry_number."
        )

    try:
        frames = dependencies.ase_read(
            str(structure_path), index=":", format="extxyz"
        )
    except MemoryError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Could not read selected EXTXYZ for {phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise InputValidationError(
            f"Selected EXTXYZ for {phase_key} must contain exactly one frame; "
            f"found {len(frames)}."
        )
    atoms = frames[0]

    # Any attached calculator could expose stale computed properties and would
    # blur the strict boundary between validation and scientific calculation.
    if atoms.calc is not None:
        raise InputValidationError(
            f"Selected structure for {phase_key} has an attached calculator."
        )
    number_of_atoms = len(atoms)
    if number_of_atoms <= 0:
        raise InputValidationError(f"Selected structure for {phase_key} has no atoms.")
    if not bool(dependencies.numpy.asarray(atoms.pbc, dtype=bool).all()):
        raise InputValidationError(
            f"Selected structure for {phase_key} is not periodic in all directions."
        )

    positions = dependencies.numpy.asarray(atoms.positions, dtype=float)
    cell = dependencies.numpy.asarray(atoms.cell.array, dtype=float)
    if positions.shape != (number_of_atoms, 3) or not bool(
        dependencies.numpy.isfinite(positions).all()
    ):
        raise InputValidationError(
            f"Selected structure for {phase_key} has invalid atomic positions."
        )
    if cell.shape != (3, 3) or not bool(dependencies.numpy.isfinite(cell).all()):
        raise InputValidationError(
            f"Selected structure for {phase_key} has invalid cell vectors."
        )
    volume = float(atoms.get_volume())
    if not math.isfinite(volume) or volume <= 0.0:
        raise InputValidationError(
            f"Selected structure for {phase_key} has invalid cell volume {volume!r}."
        )

    symbols = atoms.get_chemical_symbols()
    unexpected_elements = set(symbols).difference(ALLOWED_ELEMENTS)
    if unexpected_elements:
        raise InputValidationError(
            f"Selected structure for {phase_key} contains unsupported elements: "
            + ", ".join(sorted(unexpected_elements))
        )
    try:
        expected_composition = dependencies.composition_class(phase_key)
        structure_composition = dependencies.composition_class(Counter(symbols))
        metadata_composition = dependencies.composition_class(formula)
    except MemoryError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Could not interpret composition for {phase_key}: {exc}"
        ) from exc
    expected_signature = composition_signature(expected_composition)
    if composition_signature(structure_composition) != expected_signature:
        raise InputValidationError(
            f"Structure composition for {phase_key} reduces to "
            f"{structure_composition.reduced_formula}."
        )
    if composition_signature(metadata_composition) != expected_signature:
        raise InputValidationError(
            f"Metadata formula {formula!r} does not match phase {phase_key}."
        )
    metadata_sites = metadata.get("number_of_sites")
    if (
        not isinstance(metadata_sites, int)
        or isinstance(metadata_sites, bool)
        or metadata_sites != number_of_atoms
    ):
        raise InputValidationError(
            f"Metadata atom count for {phase_key} does not match the structure."
        )
    if atoms.info.get("phase_key") != phase_key:
        raise InputValidationError(
            f"EXTXYZ phase metadata is missing or mismatched for {phase_key}."
        )
    if atoms.info.get("material_id") != material_id:
        raise InputValidationError(
            f"EXTXYZ material ID does not match metadata for {phase_key}."
        )
    if atoms.calc is not None:
        raise InputValidationError(
            f"A calculator became attached while validating {phase_key}."
        )

    after_stat = structure_path.stat()
    after_hash = file_sha256(structure_path)
    if (
        before_hash != after_hash
        or before_stat.st_size != after_stat.st_size
        or before_stat.st_mtime_ns != after_stat.st_mtime_ns
    ):
        raise InputValidationError(
            f"Original selected structure changed during validation: {structure_path}"
        )

    LOGGER.debug(
        "Validated structure %s: %s, %d atoms, volume %.12g A^3.",
        phase_key,
        material_id,
        number_of_atoms,
        volume,
    )
    return StructureValidation(
        material_id=material_id,
        formula=formula,
        number_of_atoms=number_of_atoms,
        space_group=f"{symmetry_symbol} ({symmetry_number})",
        structure_path=structure_path,
    )


def finite_baseline_number(record: Mapping[str, Any], field: str, phase: str) -> float:
    """Require one stored Step 5 property to be a finite real number."""

    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineValidationError(
            f"Step 5 field '{field}' for {phase} is not numeric."
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise BaselineValidationError(
            f"Step 5 field '{field}' for {phase} is nonfinite."
        )
    return converted


def load_and_validate_baseline_table(
    config: RelaxationConfiguration,
    project_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], str]:
    """Load the Step 5 table and validate its single-point run provenance."""

    table = read_json_object(
        config.zero_shot_table,
        "Step 5 zero-shot table",
        BaselineValidationError,
    )
    if table.get("evaluation_type") != BASELINE_EVALUATION_TYPE:
        raise BaselineValidationError(
            "Step 5 table is not a zero-shot single-point evaluation."
        )
    if table.get("overall_status") != "success":
        raise BaselineValidationError("Step 5 table overall_status is not success.")
    model = table.get("model")
    expected_model = {
        "family": config.model.family,
        "name": config.model.name,
        "size": config.model.value,
        "device": config.model.device,
        "dtype": config.model.default_dtype,
        "dispersion_enabled": config.model.dispersion,
    }
    if not isinstance(model, dict) or any(
        model.get(field) != value for field, value in expected_model.items()
    ):
        raise BaselineValidationError(
            "Step 5 model settings do not match the planned relaxation model settings."
        )
    records = table.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise BaselineValidationError("Step 5 table must contain an array of records.")
    by_phase: dict[str, Mapping[str, Any]] = {}
    for record in records:
        phase_key = record.get("phase_key")
        if not isinstance(phase_key, str) or not phase_key:
            raise BaselineValidationError("A Step 5 record has no valid phase_key.")
        if phase_key in by_phase:
            raise BaselineValidationError(
                f"Step 5 table contains duplicate records for {phase_key}."
            )
        by_phase[phase_key] = record
    try:
        config.zero_shot_table.relative_to(project_root)
    except ValueError as exc:
        raise BaselineValidationError(
            "Step 5 table must be located inside the repository."
        ) from exc
    return by_phase, "success"


def validate_phase_baseline(
    phase_key: str,
    structure: StructureValidation,
    records_by_phase: Mapping[str, Mapping[str, Any]],
    config: RelaxationConfiguration,
    project_root: Path,
) -> None:
    """Confirm a finite successful Step 5 record exists for one input."""

    record = records_by_phase.get(phase_key)
    if record is None:
        raise BaselineValidationError(
            f"Step 5 table has no record for requested phase {phase_key}."
        )
    if record.get("material_id") != config.expected_material_ids[phase_key]:
        raise BaselineValidationError(
            f"Step 5 material ID does not match {phase_key}."
        )
    if record.get("evaluation_status") != "success":
        raise BaselineValidationError(
            f"Step 5 evaluation_status is not success for {phase_key}."
        )
    atom_count = record.get("number_of_atoms")
    if (
        not isinstance(atom_count, int)
        or isinstance(atom_count, bool)
        or atom_count != structure.number_of_atoms
    ):
        raise BaselineValidationError(
            f"Step 5 atom count does not match the selected {phase_key} structure."
        )
    finite_baseline_number(record, "total_energy_eV", phase_key)
    finite_baseline_number(record, "energy_per_atom_eV", phase_key)
    for field in FORCE_BASELINE_FIELDS:
        finite_baseline_number(record, field, phase_key)
    for field in STRESS_BASELINE_FIELDS:
        finite_baseline_number(record, field, phase_key)

    output_value = record.get("output_structure_path")
    if not isinstance(output_value, str) or not output_value.strip():
        raise BaselineValidationError(
            f"Step 5 output structure path is missing for {phase_key}."
        )
    try:
        output_path = resolve_repository_path(
            project_root,
            output_value,
            f"Step 5 output_structure_path for {phase_key}",
        )
    except ConfigurationError as exc:
        raise BaselineValidationError(str(exc)) from exc
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise BaselineValidationError(
            f"Step 5 annotated structure is missing or empty for {phase_key}: "
            f"{output_path}"
        )

    input_value = record.get("input_structure_path")
    if not isinstance(input_value, str) or not input_value.strip():
        raise BaselineValidationError(
            f"Step 5 input structure path is missing for {phase_key}."
        )
    try:
        recorded_input = resolve_repository_path(
            project_root,
            input_value,
            f"Step 5 input_structure_path for {phase_key}",
        )
    except ConfigurationError as exc:
        raise BaselineValidationError(str(exc)) from exc
    if recorded_input != structure.structure_path:
        raise BaselineValidationError(
            f"Step 5 record for {phase_key} refers to a different input structure."
        )


def validate_requested_phases(
    requested_phases: Sequence[str],
    config: RelaxationConfiguration,
    dependencies: ScientificDependencies,
    records_by_phase: Mapping[str, Mapping[str, Any]],
    project_root: Path,
) -> list[PhaseValidationOutcome]:
    """Validate every requested structure and baseline while collecting failures."""

    outcomes: list[PhaseValidationOutcome] = []
    for phase_key in requested_phases:
        path = config.selected_structure_directory / f"{phase_key}.extxyz"
        structure: StructureValidation | None = None
        try:
            structure = validate_selected_structure(phase_key, config, dependencies)
            validate_phase_baseline(
                phase_key,
                structure,
                records_by_phase,
                config,
                project_root,
            )
            outcomes.append(
                PhaseValidationOutcome(
                    phase_key=phase_key,
                    material_id=structure.material_id,
                    formula=structure.formula,
                    number_of_atoms=structure.number_of_atoms,
                    space_group=structure.space_group,
                    baseline_status="success",
                    structure_path=structure.structure_path,
                    valid=True,
                )
            )
        except (InputValidationError, BaselineValidationError) as exc:
            LOGGER.error(
                "Phase %s failed (%s): %s",
                phase_key,
                type(exc).__name__,
                exc,
            )
            outcomes.append(
                PhaseValidationOutcome(
                    phase_key=phase_key,
                    material_id=(structure.material_id if structure else None),
                    formula=(structure.formula if structure else None),
                    number_of_atoms=(structure.number_of_atoms if structure else None),
                    space_group=(structure.space_group if structure else None),
                    baseline_status=("failed" if structure else "not checked"),
                    structure_path=path,
                    valid=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
    return outcomes


def planned_output_directories(
    config: RelaxationConfiguration,
) -> tuple[Path, ...]:
    """Return the exact directory-only output plan in creation order."""

    return (
        config.root_output_directory,
        config.atomic_only_output_directory,
        config.atomic_only_output_directory / "structures",
        config.atomic_only_output_directory / "trajectories",
        config.atomic_only_output_directory / "tables",
        config.atomic_only_output_directory / "reports",
        config.full_cell_output_directory,
        config.full_cell_output_directory / "structures",
        config.full_cell_output_directory / "trajectories",
        config.full_cell_output_directory / "tables",
        config.full_cell_output_directory / "reports",
        config.comparison_output_directory,
        config.comparison_output_directory / "tables",
        config.comparison_output_directory / "reports",
    )


def create_empty_directory_plan(config: RelaxationConfiguration) -> int:
    """Create only planned directories, refusing any pre-existing result content."""

    planned = planned_output_directories(config)
    planned_set = set(planned)
    for path in planned:
        if path.exists() and not path.is_dir():
            raise DirectoryPlanError(
                f"Planned output directory path is occupied by a file: {path}"
            )
    root = config.root_output_directory
    if root.is_dir():
        unexpected = [path for path in root.rglob("*") if path not in planned_set]
        if unexpected:
            raise DirectoryPlanError(
                "The Step 6 output root is not an empty planned directory tree; "
                f"unexpected content starts with: {unexpected[0]}"
            )

    missing = [path for path in planned if not path.exists()]
    try:
        for path in planned:
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DirectoryPlanError(
            f"Could not create the planned output directories: {exc}"
        ) from exc

    # A reverse scan confirms this preparation step produced directories only.
    for path in root.rglob("*"):
        if not path.is_dir() or path not in planned_set:
            raise DirectoryPlanError(
                f"Unexpected Step 6 output content exists after creation: {path}"
            )
    return len(missing)


def path_for_console(path: Path, project_root: Path) -> str:
    """Prefer portable repository-relative paths in the execution plan."""

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def print_mode_settings(settings: RelaxationModeSettings, selected: bool) -> None:
    """Print one future mode with all convergence and cell permissions."""

    LOGGER.info("%s%s", settings.name, " [requested]" if selected else "")
    LOGGER.info("  Enabled: %s", str(settings.enabled).lower())
    LOGGER.info("  Optimizer: %s", settings.optimizer)
    LOGGER.info(
        "  Force threshold: %.10g eV/angstrom",
        settings.force_threshold_eV_per_A,
    )
    if settings.stress_threshold_eV_per_A3 is not None:
        LOGGER.info(
            "  Stress threshold: %.10g eV/angstrom^3",
            settings.stress_threshold_eV_per_A3,
        )
    else:
        LOGGER.info("  Stress threshold: not applicable to fixed-cell mode")
    LOGGER.info("  Maximum steps: %d", settings.maximum_steps)
    LOGGER.info("  Trajectory interval: %d", settings.trajectory_interval)
    LOGGER.info("  Allow atomic positions: %s", settings.allow_atomic_positions)
    LOGGER.info("  Allow cell shape: %s", settings.allow_cell_shape)
    LOGGER.info("  Allow cell volume: %s", settings.allow_cell_volume)
    if settings.hydrostatic_strain is not None:
        LOGGER.info("  Hydrostatic strain: %s", settings.hydrostatic_strain)
    if settings.constant_volume is not None:
        LOGGER.info("  Constant volume: %s", settings.constant_volume)


def print_console_summary(
    project_root: Path,
    config_path: Path,
    config: RelaxationConfiguration,
    requested_phases: Sequence[str],
    requested_mode: str,
    outcomes: Sequence[PhaseValidationOutcome],
    baseline_table_status: str,
    create_requested: bool,
    created_count: int,
) -> None:
    """Print the complete validation result and later execution plan."""

    validated_count = sum(outcome.valid for outcome in outcomes)
    failed_count = len(outcomes) - validated_count
    overall_success = failed_count == 0
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info("STEP 6A — GEOMETRY-RELAXATION CONFIGURATION AND VALIDATION")
    LOGGER.info("=" * 78)
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Configuration path: %s", config_path)
    LOGGER.info("Schema version: %s", config.schema_version)
    LOGGER.info("Requested phases: %s", ", ".join(requested_phases))
    LOGGER.info("Requested mode: %s", requested_mode)
    LOGGER.info("Model settings:")
    LOGGER.info("  Family/name/value: %s / %s / %s", config.model.family, config.model.name, config.model.value)
    LOGGER.info("  Device/dtype: %s / %s", config.model.device, config.model.default_dtype)
    LOGGER.info("  Dispersion: %s", str(config.model.dispersion).lower())
    LOGGER.info("")
    LOGGER.info("Phase validation")
    LOGGER.info("----------------")
    for outcome in outcomes:
        LOGGER.info("Phase: %s", outcome.phase_key)
        LOGGER.info("  Material ID: %s", outcome.material_id or "not available")
        LOGGER.info("  Formula: %s", outcome.formula or "not available")
        LOGGER.info(
            "  Number of atoms: %s",
            outcome.number_of_atoms if outcome.number_of_atoms is not None else "not available",
        )
        LOGGER.info("  Space group: %s", outcome.space_group or "not available")
        LOGGER.info("  Step 5 baseline status: %s", outcome.baseline_status)
        LOGGER.info(
            "  Input structure: %s",
            path_for_console(outcome.structure_path, project_root),
        )
        LOGGER.info("  Validation status: %s", "VALID" if outcome.valid else "FAILED")
        if not outcome.valid:
            LOGGER.info("  Error: %s: %s", outcome.error_type, outcome.error_message)

    LOGGER.info("")
    LOGGER.info("Relaxation-mode settings")
    LOGGER.info("------------------------")
    print_mode_settings(
        config.atomic_only,
        requested_mode in {"atomic_only", "all"},
    )
    print_mode_settings(
        config.full_cell,
        requested_mode in {"full_cell", "all"},
    )
    LOGGER.info("")
    LOGGER.info("Safety settings")
    LOGGER.info("---------------")
    LOGGER.info(
        "Maximum absolute volume change: %.10g %%",
        config.safety.maximum_absolute_volume_change_percent,
    )
    LOGGER.info(
        "Maximum atomic displacement: %.10g angstrom",
        config.safety.maximum_atomic_displacement_A,
    )
    LOGGER.info("Stop on nonfinite value: %s", config.safety.stop_on_nonfinite_value)
    LOGGER.info("Preserve original structure: %s", config.safety.preserve_original_structure)
    LOGGER.info("Require periodic cell: %s", config.safety.require_periodic_cell)
    LOGGER.info("")
    LOGGER.info("Planned output paths")
    LOGGER.info("--------------------")
    for path in planned_output_directories(config):
        LOGGER.info("  %s/", path_for_console(path, project_root))
    LOGGER.info("")
    LOGGER.info("Later execution plan (not executed in Step 6A)")
    LOGGER.info("----------------------------------------------")
    LOGGER.info("1. Step 6B will load MACE once and reproduce the initial single-point baseline.")
    LOGGER.info("2. atomic_only will start from an original copy and keep all cell vectors fixed.")
    LOGGER.info("3. full_cell will start independently and may change atoms, shape, and volume.")
    LOGGER.info("4. Each mode will enforce convergence and safety limits in separate outputs.")
    LOGGER.info("5. A later comparison will separate internal-coordinate and cell effects.")
    LOGGER.info("")
    LOGGER.info("Validated phase count: %d", validated_count)
    LOGGER.info("Failed phase count: %d", failed_count)
    LOGGER.info("Step 5 baseline table status: %s", baseline_table_status)
    LOGGER.info("Directory creation requested: %s", str(create_requested).lower())
    LOGGER.info("Directories newly created: %d", created_count)
    LOGGER.info("MACE loaded: No")
    LOGGER.info("Energy, force, or stress calculation requested: No")
    LOGGER.info("Relaxation run: No")
    LOGGER.info("Atomic positions changed: No")
    LOGGER.info("Cell vectors changed: No")
    LOGGER.info("Overall validation status: %s", "SUCCESS" if overall_success else "FAILURE")
    LOGGER.info("=" * 78)


def run(arguments: Sequence[str] | None = None) -> int:
    """Run Step 6A validation and return a process-compatible status code."""

    options = parse_arguments(arguments)
    configure_logging(options.verbose)
    project_root = locate_project_root()
    try:
        validate_runtime(project_root)
        dependencies = import_scientific_dependencies()
        config_path = (
            options.config.resolve()
            if options.config.is_absolute()
            else (project_root / options.config).resolve()
        )
        LOGGER.debug("Resolved project root: %s", project_root)
        LOGGER.debug("Resolved configuration: %s", config_path)
        raw_config = read_json_object(
            config_path,
            "relaxation configuration",
            ConfigurationError,
        )
        config = validate_configuration(raw_config, project_root)
        requested_phases = choose_requested_phases(config.phase_order, options.phase)
        validate_requested_mode(options.mode, config)
        records, baseline_status = load_and_validate_baseline_table(
            config, project_root
        )
        outcomes = validate_requested_phases(
            requested_phases,
            config,
            dependencies,
            records,
            project_root,
        )
        created_count = 0
        if options.create_directories:
            if any(not outcome.valid for outcome in outcomes):
                raise DirectoryPlanError(
                    "Output directories were not created because phase validation failed."
                )
            created_count = create_empty_directory_plan(config)
        print_console_summary(
            project_root,
            config_path,
            config,
            requested_phases,
            options.mode,
            outcomes,
            baseline_status,
            options.create_directories,
            created_count,
        )
        return 0 if all(outcome.valid for outcome in outcomes) else 1
    except Step6AError as exc:
        LOGGER.error("Step 6A failed (%s): %s", type(exc).__name__, exc)
        LOGGER.info("MACE loaded: No")
        LOGGER.info("Relaxation run: No")
        LOGGER.info("Atomic positions changed: No")
        LOGGER.info("Cell vectors changed: No")
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Step 6A validation was interrupted by the user.")
        return 130


def main() -> None:
    """Propagate Step 6A success or failure to the calling shell."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()

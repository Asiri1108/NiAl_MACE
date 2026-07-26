#!/usr/bin/env python3
"""Validate and compare the Ni-Al Step 6 relaxation outputs.

This module is deliberately calculator-free.  It consumes the stored Step 5
single-point table and validated Step 6C/6D phase-result manifests, checks their
companion structures, trajectories, histories, and reports, then optionally
publishes the Step 6E comparison bundle.  It never imports MACE, creates a
calculator, runs an optimizer, or evaluates a physical property.

The public :func:`validate_plan` function accepts ``require_inputs=False`` so
the pipeline's static validation gate can validate future paths before Step 6C
and Step 6D exist.  Standalone ``--validate-only`` and ``--analyze`` use
``require_inputs=True``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger("ni_al_step6_analysis")

SCHEMA_VERSION = "1.0"
PROJECT_STEP = "6E"
RECORD_TYPE = "ni_al_mace_relaxation_comparison"
EXPECTED_PHASE_ORDER = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
EXPECTED_IDENTITIES: Mapping[str, tuple[str, int]] = {
    "Al3Ni": ("mp-622209", 16),
    "Al3Ni2": ("mp-1057", 5),
    "AlNi": ("mp-1487", 2),
    "Al3Ni5": ("mp-16514", 8),
    "AlNi3": ("mp-2593", 4),
}
MODES = ("atomic_only", "full_cell")
STRESS_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
CONVERGED_STATUSES = frozenset({"ALREADY_CONVERGED", "CONVERGED"})
CONVERGENCE_STATUSES = frozenset(
    {"ALREADY_CONVERGED", "CONVERGED", "NOT_CONVERGED"}
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_SYMPREC_A = 0.001
DEFAULT_ANGLE_TOLERANCE_DEG = 5.0

FIGURE_FILENAMES = (
    "atomic_only_energy_convergence.png",
    "atomic_only_force_convergence.png",
    "full_cell_energy_convergence.png",
    "full_cell_force_convergence.png",
    "full_cell_stress_convergence.png",
    "full_cell_volume_convergence.png",
    "final_energy_change_by_phase.png",
    "final_max_force_by_phase.png",
    "full_cell_volume_change_percent.png",
)

CSV_FIELDS = (
    "phase_order_index",
    "phase_key",
    "formula",
    "material_id",
    "number_of_atoms",
    "stage",
    "result_class",
    "diagnostic_only",
    "total_energy_eV",
    "energy_per_atom_eV",
    "energy_change_from_initial_eV",
    "energy_change_from_initial_eV_per_atom",
    "maximum_force_eV_per_A",
    "rms_force_eV_per_A",
    "total_force_x_eV_per_A",
    "total_force_y_eV_per_A",
    "total_force_z_eV_per_A",
    "total_force_norm_eV_per_A",
    "stress_xx_eV_per_A3",
    "stress_yy_eV_per_A3",
    "stress_zz_eV_per_A3",
    "stress_yz_eV_per_A3",
    "stress_xz_eV_per_A3",
    "stress_xy_eV_per_A3",
    "maximum_absolute_stress_eV_per_A3",
    "volume_A3",
    "volume_per_atom_A3",
    "volume_change_from_initial_A3",
    "volume_change_from_initial_percent",
    "a_A",
    "b_A",
    "c_A",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
    "maximum_displacement_A",
    "rms_displacement_A",
    "displacement_definition",
    "maximum_internal_displacement_A",
    "rms_internal_displacement_A",
    "maximum_total_cartesian_displacement_A",
    "rms_total_cartesian_displacement_A",
    "optimizer_steps",
    "wall_clock_duration_s",
    "force_converged",
    "stress_converged",
    "overall_convergence_status",
    "safety_status",
    "space_group_symbol",
    "space_group_number",
    "space_group_symbol_preserved",
    "space_group_number_preserved",
    "symmetry_preserved",
    "source_result_path",
)

HISTORY_NUMERIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "step": ("optimizer_step", "step"),
    "total_energy_eV": ("total_energy_eV", "energy_eV"),
    "energy_per_atom_eV": ("energy_per_atom_eV",),
    "maximum_force_eV_per_A": (
        "maximum_force_eV_per_A",
        "max_force_eV_per_A",
    ),
    "rms_force_eV_per_A": ("rms_force_eV_per_A",),
    "maximum_absolute_stress_eV_per_A3": (
        "maximum_absolute_stress_eV_per_A3",
        "max_abs_stress_eV_per_A3",
    ),
    "volume_A3": ("volume_A3",),
}


class Step6AnalysisError(RuntimeError):
    """Base class for readable Step 6E failures."""


class ConfigurationError(Step6AnalysisError):
    """Raised when the Step 6 configuration is invalid."""


class InputValidationError(Step6AnalysisError):
    """Raised when an upstream result bundle is invalid."""


class DependencyError(Step6AnalysisError):
    """Raised when a non-MACE analysis dependency is unavailable."""


class OutputCollisionError(Step6AnalysisError):
    """Raised before analysis when one or more Step 6E targets exist."""


class PublicationError(Step6AnalysisError):
    """Raised when the complete comparison bundle cannot be published."""


class DuplicateJsonKeyError(ValueError):
    """Raised when strict JSON parsing finds a duplicate object key."""


@dataclass(frozen=True)
class CommandLineOptions:
    """Validated command-line selection."""

    config: Path
    validate_only: bool
    analyze: bool
    overwrite: bool
    verbose: bool


@dataclass(frozen=True)
class FileIdentity:
    """Stable identity for one repository file."""

    path: Path
    relative_path: str
    sha256: str
    size_bytes: int
    modification_time_ns: int

    def as_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "modification_time_ns": self.modification_time_ns,
        }


@dataclass(frozen=True)
class AnalysisConfiguration:
    """Repository-resolved settings needed by Step 6E."""

    project_root: Path
    config_path: Path
    config_identity: FileIdentity
    raw_config: Mapping[str, Any]
    semantic_fingerprint_sha256: str
    phase_order: tuple[str, ...]
    expected_material_ids: Mapping[str, str]
    expected_atom_counts: Mapping[str, int]
    selected_directory: Path
    step5_table: Path
    atomic_directory: Path
    full_cell_directory: Path
    comparison_directory: Path
    force_threshold_eV_per_A: float
    stress_threshold_eV_per_A3: float
    symmetry_symprec_A: float
    symmetry_angle_tolerance_deg: float


@dataclass(frozen=True)
class OutputPlan:
    """Exact authorized Step 6E publication targets."""

    csv_table: Path
    json_table: Path
    text_report: Path
    figures: Mapping[str, Path]

    @property
    def targets(self) -> tuple[Path, ...]:
        """Return every target in deterministic publication order."""

        return (
            self.csv_table,
            self.json_table,
            self.text_report,
            *(self.figures[name] for name in FIGURE_FILENAMES),
        )


@dataclass(frozen=True)
class SymmetryResult:
    """Space-group result evaluated at the common Step 6E tolerances."""

    symbol: str
    number: int

    def as_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "space_group_symbol": self.symbol,
            "space_group_number": self.number,
        }


@dataclass(frozen=True)
class HistoryTable:
    """Strictly parsed convergence history."""

    path: Path
    identity: FileIdentity
    fieldnames: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PhaseResult:
    """Validated upstream result and its companion artifacts."""

    phase_key: str
    mode: str
    manifest_path: Path
    manifest_identity: FileIdentity
    document: Mapping[str, Any]
    identity: Mapping[str, Any]
    execution: Mapping[str, Any]
    initial: Mapping[str, Any]
    final: Mapping[str, Any]
    changes: Mapping[str, Any]
    displacements: Mapping[str, Any]
    strain: Mapping[str, Any] | None
    structure_path: Path
    trajectory_path: Path
    history: HistoryTable
    report_path: Path
    source_structure_path: Path
    final_atoms: Any


@dataclass(frozen=True)
class PhaseInputs:
    """Step 5 baseline plus both independently relaxed outputs."""

    phase_key: str
    step5_record: Mapping[str, Any]
    source_atoms: Any
    source_identity: FileIdentity
    atomic_only: PhaseResult
    full_cell: PhaseResult


@dataclass(frozen=True)
class AnalysisPlan:
    """Complete static or input-validated Step 6E plan."""

    config: AnalysisConfiguration
    outputs: OutputPlan
    collisions: tuple[Path, ...]
    step5_identity: FileIdentity | None
    phases: tuple[PhaseInputs, ...]
    inputs_validated: bool


@dataclass(frozen=True)
class AnalysisData:
    """Derived comparison data ready for serialization and plotting."""

    generated_at_utc: str
    rows: tuple[Mapping[str, Any], ...]
    phase_records: tuple[Mapping[str, Any], ...]
    histories: Mapping[tuple[str, str], HistoryTable]
    analysis_status: str
    diagnostic_phases: Mapping[str, tuple[str, ...]]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys rather than accepting ambiguous input."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject the non-standard JSON constants NaN and Infinity."""

    raise ValueError(f"Non-standard JSON constant is forbidden: {value}")


def read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    """Read one strict UTF-8 JSON object."""

    if not path.is_file():
        raise InputValidationError(f"{label} does not exist: {path}")
    if path.stat().st_size <= 0:
        raise InputValidationError(f"{label} is empty: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError(
            f"Could not read {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise InputValidationError(f"{label} root must be a JSON object.")
    return document


def _canonical_json_sha256(value: Any) -> str:
    """Hash one semantic payload using deterministic strict JSON."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Could not canonicalize the configuration: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise InputValidationError(
            f"Could not hash {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return digest.hexdigest()


def capture_file_identity(
    path: Path,
    project_root: Path,
    label: str,
) -> FileIdentity:
    """Capture content and filesystem metadata for one regular file."""

    if path.is_symlink() or not path.is_file():
        raise InputValidationError(f"{label} must be a regular file: {path}")
    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise InputValidationError(
            f"{label} must stay inside the repository: {path}"
        ) from exc
    stat = path.stat()
    if stat.st_size <= 0:
        raise InputValidationError(f"{label} is empty: {path}")
    return FileIdentity(
        path=path,
        relative_path=relative,
        sha256=_file_sha256(path),
        size_bytes=stat.st_size,
        modification_time_ns=stat.st_mtime_ns,
    )


def _path_for_console(path: Path, root: Path) -> str:
    """Prefer a repository-relative console path."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def locate_project_root() -> Path:
    """Locate the repository independently of the terminal directory."""

    root = Path(__file__).resolve().parents[1]
    if not (root / "configs").is_dir() or not (root / "scripts").is_dir():
        raise ConfigurationError(
            f"Could not identify the project root from {Path(__file__).resolve()}."
        )
    return root


def _optional_shared_utils() -> Any | None:
    """Lazily import ``step6_utils`` when it is complete and calculator-free.

    The analyzer does not depend on this optional module.  During collaborative
    implementation the file may temporarily be absent or incomplete, so any
    import failure falls back to the local, deliberately narrow utilities.
    Importing it must not cause a MACE package to appear.
    """

    if importlib.util.find_spec("step6_utils") is None:
        return None
    before = {name for name in sys.modules if name == "mace" or name.startswith("mace.")}
    try:
        module = importlib.import_module("step6_utils")
    except (ImportError, SyntaxError, AttributeError) as exc:
        LOGGER.debug("Shared step6_utils is unavailable; using local helpers: %s", exc)
        return None
    after = {name for name in sys.modules if name == "mace" or name.startswith("mace.")}
    if after - before:
        raise DependencyError(
            "Importing step6_utils unexpectedly imported MACE; Step 6E refuses "
            "to continue."
        )
    return module


def assert_mace_not_imported() -> None:
    """Enforce the analysis-only execution boundary."""

    imported = sorted(
        name for name in sys.modules if name == "mace" or name.startswith("mace.")
    )
    if imported:
        raise DependencyError(
            "Step 6E must not import MACE; imported modules: "
            + ", ".join(imported)
        )


def parse_arguments(arguments: Sequence[str] | None = None) -> CommandLineOptions:
    """Parse the strict Step 6E command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate stored Step 5/6C/6D Ni-Al results and optionally create "
            "the Step 6E comparison bundle without running MACE."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mace_relaxation.json"),
        help="Relaxation configuration path, resolved from the repository root.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate every required input and output path without writing outputs.",
    )
    modes.add_argument(
        "--analyze",
        action="store_true",
        help="Create the complete Step 6E comparison and figure bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the exact Step 6E output targets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    namespace = parser.parse_args(arguments)
    if namespace.overwrite and not namespace.analyze:
        parser.error("--overwrite is allowed only with --analyze.")
    return CommandLineOptions(
        config=namespace.config,
        validate_only=bool(namespace.validate_only),
        analyze=bool(namespace.analyze),
        overwrite=bool(namespace.overwrite),
        verbose=bool(namespace.verbose),
    )


def configure_logging(verbose: bool) -> None:
    """Configure deterministic console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def _require_mapping(
    parent: Mapping[str, Any],
    field: str,
    label: str,
) -> Mapping[str, Any]:
    """Require one child JSON object."""

    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{label}.{field} must be an object.")
    return value


def _require_string(
    parent: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    """Require a non-empty string."""

    value = parent.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{label}.{field} must be a non-empty string.")
    return value.strip()


def _finite_float(value: Any, label: str) -> float:
    """Convert one non-Boolean finite numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise InputValidationError(f"{label} must be finite.")
    return result


def _positive_float(value: Any, label: str) -> float:
    """Require one strictly positive finite value."""

    result = _finite_float(value, label)
    if result <= 0.0:
        raise InputValidationError(f"{label} must be positive.")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    """Require a nonnegative integer without accepting Boolean values."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputValidationError(f"{label} must be a nonnegative integer.")
    return value


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    """Require a fixed-length finite numeric sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InputValidationError(f"{label} must be a {length}-value array.")
    if len(value) != length:
        raise InputValidationError(
            f"{label} must contain {length} values; found {len(value)}."
        )
    return tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value))


def _resolve_repository_path(
    root: Path,
    value: str | Path,
    label: str,
) -> Path:
    """Resolve and constrain a configured path to the repository."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"{label} escapes the repository: {candidate}") from exc
    return resolved


def _configuration_path(root: Path, supplied: Path) -> Path:
    """Resolve a CLI configuration path from the repository."""

    return _resolve_repository_path(root, supplied, "--config")


def validate_configuration(config_path: Path) -> AnalysisConfiguration:
    """Validate the fields used by Step 6E and resolve all repository paths."""

    project_root = locate_project_root()
    document = read_json_object(config_path, "Step 6 configuration")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Configuration schema_version must equal {SCHEMA_VERSION!r}."
        )

    phase_order_value = document.get("phase_order")
    if (
        not isinstance(phase_order_value, list)
        or tuple(phase_order_value) != EXPECTED_PHASE_ORDER
    ):
        raise ConfigurationError(
            "phase_order must contain exactly: " + ", ".join(EXPECTED_PHASE_ORDER)
        )
    phase_order = tuple(str(value) for value in phase_order_value)
    expected_phases = _require_mapping(document, "expected_phases", "configuration")
    material_ids: dict[str, str] = {}
    atom_counts: dict[str, int] = {}
    for phase in phase_order:
        phase_config = _require_mapping(
            expected_phases, phase, f"configuration.expected_phases"
        )
        material_id = _require_string(
            phase_config, "expected_material_id", f"expected_phases.{phase}"
        )
        expected_id, expected_count = EXPECTED_IDENTITIES[phase]
        if material_id != expected_id:
            raise ConfigurationError(
                f"Expected material ID for {phase} must be {expected_id}; "
                f"found {material_id}."
            )
        material_ids[phase] = material_id
        atom_counts[phase] = expected_count

    input_config = _require_mapping(document, "input", "configuration")
    output_config = _require_mapping(document, "output", "configuration")
    selected_directory = _resolve_repository_path(
        project_root,
        _require_string(
            input_config, "selected_structure_directory", "configuration.input"
        ),
        "input.selected_structure_directory",
    )
    step5_table = _resolve_repository_path(
        project_root,
        _require_string(input_config, "zero_shot_table", "configuration.input"),
        "input.zero_shot_table",
    )
    atomic_directory = _resolve_repository_path(
        project_root,
        _require_string(
            output_config, "atomic_only_directory", "configuration.output"
        ),
        "output.atomic_only_directory",
    )
    full_cell_directory = _resolve_repository_path(
        project_root,
        _require_string(output_config, "full_cell_directory", "configuration.output"),
        "output.full_cell_directory",
    )
    comparison_directory = _resolve_repository_path(
        project_root,
        _require_string(
            output_config, "comparison_directory", "configuration.output"
        ),
        "output.comparison_directory",
    )
    if len({atomic_directory, full_cell_directory, comparison_directory}) != 3:
        raise ConfigurationError("Step 6 mode output directories must be distinct.")

    modes = _require_mapping(document, "relaxation_modes", "configuration")
    atomic_mode = _require_mapping(modes, "atomic_only", "relaxation_modes")
    full_mode = _require_mapping(modes, "full_cell", "relaxation_modes")
    force_atomic = _positive_float(
        atomic_mode.get("force_threshold_eV_per_A"),
        "atomic_only.force_threshold_eV_per_A",
    )
    force_full = _positive_float(
        full_mode.get("force_threshold_eV_per_A"),
        "full_cell.force_threshold_eV_per_A",
    )
    if force_atomic != force_full:
        raise ConfigurationError(
            "Atomic-only and full-cell force thresholds must match for Step 6E."
        )
    stress_threshold = _positive_float(
        full_mode.get("stress_threshold_eV_per_A3"),
        "full_cell.stress_threshold_eV_per_A3",
    )

    execution_value = document.get("step6_execution", {})
    if not isinstance(execution_value, Mapping):
        raise ConfigurationError("step6_execution must be an object when present.")
    symprec = _positive_float(
        execution_value.get("symmetry_symprec_A", DEFAULT_SYMPREC_A),
        "step6_execution.symmetry_symprec_A",
    )
    angle_tolerance = _positive_float(
        execution_value.get(
            "symmetry_angle_tolerance_deg", DEFAULT_ANGLE_TOLERANCE_DEG
        ),
        "step6_execution.symmetry_angle_tolerance_deg",
    )
    if symprec != DEFAULT_SYMPREC_A or angle_tolerance != DEFAULT_ANGLE_TOLERANCE_DEG:
        raise ConfigurationError(
            "Step 6E symmetry tolerances must remain 0.001 angstrom and 5 degrees."
        )

    semantic_payload = {
        "schema_version": document.get("schema_version"),
        "model": document.get("model"),
        "phase_order": phase_order_value,
        "expected_phases": document.get("expected_phases"),
        "relaxation_modes": document.get("relaxation_modes"),
        "safety": document.get("safety"),
        "step6_execution": execution_value,
    }
    config_identity = capture_file_identity(
        config_path, project_root, "Step 6 configuration"
    )
    return AnalysisConfiguration(
        project_root=project_root,
        config_path=config_path,
        config_identity=config_identity,
        raw_config=document,
        semantic_fingerprint_sha256=_canonical_json_sha256(semantic_payload),
        phase_order=phase_order,
        expected_material_ids=material_ids,
        expected_atom_counts=atom_counts,
        selected_directory=selected_directory,
        step5_table=step5_table,
        atomic_directory=atomic_directory,
        full_cell_directory=full_cell_directory,
        comparison_directory=comparison_directory,
        force_threshold_eV_per_A=force_atomic,
        stress_threshold_eV_per_A3=stress_threshold,
        symmetry_symprec_A=symprec,
        symmetry_angle_tolerance_deg=angle_tolerance,
    )


def build_output_plan(config: AnalysisConfiguration) -> OutputPlan:
    """Build the exact Step 6E target inventory."""

    tables = config.comparison_directory / "tables"
    reports = config.comparison_directory / "reports"
    figures_directory = config.comparison_directory / "figures"
    figures = {
        name: figures_directory / name for name in FIGURE_FILENAMES
    }
    return OutputPlan(
        csv_table=tables / "ni_al_relaxation_comparison.csv",
        json_table=tables / "ni_al_relaxation_comparison.json",
        text_report=reports / "ni_al_relaxation_comparison.txt",
        figures=figures,
    )


def _find_collisions(plan: OutputPlan) -> tuple[Path, ...]:
    """Return existing exact targets without following unsafe entries."""

    return tuple(path for path in plan.targets if os.path.lexists(path))


def _validate_output_target_safety(
    plan: OutputPlan,
    config: AnalysisConfiguration,
) -> None:
    """Constrain every output to the configured comparison directory."""

    root = config.comparison_directory.resolve()
    if len(set(plan.targets)) != len(plan.targets):
        raise ConfigurationError("Step 6E output targets are not unique.")
    for target in plan.targets:
        try:
            target.resolve().relative_to(root)
        except ValueError as exc:
            raise ConfigurationError(
                f"Step 6E target escapes comparison_directory: {target}"
            ) from exc
        if target.is_symlink():
            raise OutputCollisionError(
                f"Step 6E refuses a symlink target: {target}"
            )


def _validate_collision_policy(
    plan: OutputPlan,
    config: AnalysisConfiguration,
    overwrite: bool,
) -> tuple[Path, ...]:
    """Reject every collision unless exact Step 6E replacement was requested."""

    collisions = _find_collisions(plan)
    unsafe = [
        path
        for path in collisions
        if path.is_symlink() or not path.is_file()
    ]
    if unsafe:
        listed = "\n".join(
            f"  - {_path_for_console(path, config.project_root)}" for path in unsafe
        )
        raise OutputCollisionError(
            "Step 6E target collision(s) are symlinks or non-regular files:\n"
            + listed
        )
    if collisions and not overwrite:
        listed = "\n".join(
            f"  - {_path_for_console(path, config.project_root)}"
            for path in collisions
        )
        raise OutputCollisionError(
            "Step 6E output collision(s) found. No output was changed. "
            "Inspect every target and use --overwrite intentionally:\n" + listed
        )
    return collisions


def _lazy_import_scientific_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Import only calculator-free analysis APIs."""

    assert_mace_not_imported()
    try:
        import numpy as np
        from ase.io import read as ase_read
        from ase.io.trajectory import Trajectory
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import (
            SpacegroupAnalyzer,
            SymmetryUndeterminedError,
        )
    except ImportError as exc:
        raise DependencyError(
            f"Required Step 6E analysis dependency is unavailable: {exc}"
        ) from exc
    assert_mace_not_imported()
    symmetry_api = (AseAtomsAdaptor, SpacegroupAnalyzer, SymmetryUndeterminedError)
    return np, ase_read, Trajectory, symmetry_api, None


def _read_atoms(path: Path, label: str, ase_read: Any) -> Any:
    """Read exactly one periodic ASE Atoms object without calculation."""

    if not path.is_file() or path.stat().st_size <= 0:
        raise InputValidationError(f"{label} is missing or empty: {path}")
    try:
        atoms = ase_read(path, index=0)
    except Exception as exc:
        raise InputValidationError(
            f"Could not read {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if atoms is None:
        raise InputValidationError(f"{label} did not contain a structure.")
    return atoms


def _validate_atoms_identity(
    atoms: Any,
    phase: str,
    expected_count: int,
    np: Any,
    label: str,
) -> None:
    """Validate finite periodic Al-Ni geometry and stable atom identity."""

    if len(atoms) != expected_count:
        raise InputValidationError(
            f"{label} atom count for {phase} is {len(atoms)}; expected "
            f"{expected_count}."
        )
    symbols = tuple(atoms.get_chemical_symbols())
    if not symbols or set(symbols) - {"Al", "Ni"}:
        raise InputValidationError(f"{label} contains elements other than Al and Ni.")
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    if positions.shape != (expected_count, 3) or not np.all(np.isfinite(positions)):
        raise InputValidationError(f"{label} has invalid positions.")
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise InputValidationError(f"{label} has an invalid cell.")
    determinant = float(np.linalg.det(cell))
    volume = float(atoms.get_volume())
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise InputValidationError(f"{label} cell determinant is not positive.")
    if not math.isfinite(volume) or volume <= 0.0:
        raise InputValidationError(f"{label} volume is not positive.")
    if not bool(np.all(np.asarray(atoms.get_pbc(), dtype=bool))):
        raise InputValidationError(f"{label} is not periodic in all directions.")


def _validate_step5_table(
    config: AnalysisConfiguration,
) -> tuple[FileIdentity, Mapping[str, Mapping[str, Any]]]:
    """Validate the Step 5 table and index its five canonical records."""

    document = read_json_object(config.step5_table, "Step 5 baseline JSON")
    if document.get("schema_version") != "1.0":
        raise InputValidationError("Step 5 schema_version must equal '1.0'.")
    if document.get("evaluation_type") != "zero-shot single-point":
        raise InputValidationError(
            "Step 5 evaluation_type must be 'zero-shot single-point'."
        )
    if str(document.get("overall_status", "")).lower() != "success":
        raise InputValidationError("Step 5 overall_status is not success.")
    records_value = document.get("records")
    if not isinstance(records_value, list):
        raise InputValidationError("Step 5 records must be an array.")
    indexed: dict[str, Mapping[str, Any]] = {}
    numeric_fields = (
        "total_energy_eV",
        "energy_per_atom_eV",
        "maximum_force_eV_per_A",
        "rms_force_eV_per_A",
        "total_force_x_eV_per_A",
        "total_force_y_eV_per_A",
        "total_force_z_eV_per_A",
        "total_force_norm_eV_per_A",
        *(f"stress_{component}_eV_per_A3" for component in STRESS_COMPONENTS),
        "volume_A3",
        "volume_per_atom_A3",
    )
    for raw in records_value:
        if not isinstance(raw, Mapping):
            raise InputValidationError("Every Step 5 record must be an object.")
        phase = _require_string(raw, "phase_key", "Step 5 record")
        if phase not in config.phase_order or phase in indexed:
            raise InputValidationError(
                f"Step 5 contains an unexpected or duplicate phase: {phase}"
            )
        if raw.get("material_id") != config.expected_material_ids[phase]:
            raise InputValidationError(
                f"Step 5 material ID mismatch for {phase}."
            )
        if raw.get("number_of_atoms") != config.expected_atom_counts[phase]:
            raise InputValidationError(
                f"Step 5 atom-count mismatch for {phase}."
            )
        if str(raw.get("evaluation_status", "")).lower() != "success":
            raise InputValidationError(f"Step 5 record for {phase} is not successful.")
        for field in numeric_fields:
            _finite_float(raw.get(field), f"Step 5 {phase}.{field}")
        if _positive_float(raw.get("volume_A3"), f"Step 5 {phase}.volume_A3") <= 0:
            raise AssertionError("unreachable")
        indexed[phase] = raw
    if tuple(indexed) != config.phase_order:
        raise InputValidationError(
            "Step 5 records do not follow the configured deterministic phase order."
        )
    identity = capture_file_identity(
        config.step5_table, config.project_root, "Step 5 baseline JSON"
    )
    return identity, indexed


def _phase_manifest_path(
    config: AnalysisConfiguration,
    phase: str,
    mode: str,
) -> Path:
    """Return the exact machine-readable sidecar path."""

    base = config.atomic_directory if mode == "atomic_only" else config.full_cell_directory
    return base / "checkpoints" / f"{phase}_{mode}_result.json"


def _expected_companion_paths(
    config: AnalysisConfiguration,
    phase: str,
    mode: str,
) -> Mapping[str, Path]:
    """Return exact required paths for one upstream phase bundle."""

    base = config.atomic_directory if mode == "atomic_only" else config.full_cell_directory
    relaxed_name = f"{phase}_{mode}_relaxed.extxyz"
    return {
        "structure": base / "structures" / relaxed_name,
        "trajectory": base / "trajectories" / f"{phase}_{mode}.traj",
        "history": base / "tables" / f"{phase}_{mode}_history.csv",
        "report": base / "reports" / f"{phase}_{mode}_report.txt",
        "optimizer_log": base / "logs" / f"{phase}_{mode}.log",
    }


def _manifest_identity_section(
    document: Mapping[str, Any],
    phase: str,
) -> Mapping[str, Any]:
    """Read the required identity object with a narrow legacy fallback."""

    value = document.get("identity")
    if isinstance(value, Mapping):
        return value
    if all(key in document for key in ("phase_key", "material_id", "number_of_atoms")):
        return document
    raise InputValidationError(f"{phase} result manifest lacks identity.")


def _manifest_configuration_fingerprints(
    document: Mapping[str, Any],
    phase: str,
) -> tuple[str, str | None]:
    """Read exact-file and semantic configuration fingerprints."""

    section = document.get("configuration")
    if not isinstance(section, Mapping):
        section = document
    file_hash = section.get("file_sha256", section.get("configuration_file_sha256"))
    semantic_hash = section.get(
        "semantic_fingerprint_sha256",
        section.get("configuration_fingerprint_sha256"),
    )
    # The shared Step 6 runner uses the exact configuration-file hash as its
    # single resume fingerprint.  Richer manifests may additionally carry the
    # semantic fingerprint proposed by Step 6E.
    if file_hash is None and isinstance(semantic_hash, str):
        file_hash = semantic_hash
        semantic_hash = None
    for value, name in ((file_hash, "configuration file SHA-256"),):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise InputValidationError(f"{phase} has an invalid {name}.")
    if semantic_hash is not None and (
        not isinstance(semantic_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", semantic_hash)
    ):
        raise InputValidationError(
            f"{phase} has an invalid semantic configuration fingerprint."
        )
    return str(file_hash), None if semantic_hash is None else str(semantic_hash)


def _output_descriptor(
    outputs: Mapping[str, Any],
    name: str,
    expected: Path,
    config: AnalysisConfiguration,
    phase: str,
) -> Mapping[str, Any]:
    """Validate one exact companion descriptor and its fingerprint."""

    aliases: Mapping[str, tuple[str, ...]] = {
        "structure": ("structure", "final_structure"),
        "trajectory": ("trajectory",),
        "history": ("history", "history_csv"),
        "report": ("report",),
        "optimizer_log": ("optimizer_log", "log"),
    }
    value: Any = None
    for alias in aliases[name]:
        if alias in outputs:
            value = outputs[alias]
            break
    if not isinstance(value, Mapping):
        raise InputValidationError(
            f"{phase} outputs.{name} must be a file descriptor."
        )
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise InputValidationError(f"{phase} outputs.{name}.path is invalid.")
    resolved = _resolve_repository_path(
        config.project_root, path_value, f"{phase} outputs.{name}.path"
    )
    if resolved != expected.resolve():
        raise InputValidationError(
            f"{phase} {name} path is {resolved}; expected {expected.resolve()}."
        )
    actual = capture_file_identity(expected, config.project_root, f"{phase} {name}")
    recorded_hash = value.get("sha256")
    recorded_size = value.get("size_bytes")
    if recorded_hash != actual.sha256 or recorded_size != actual.size_bytes:
        raise InputValidationError(
            f"{phase} {name} fingerprint does not match its result manifest."
        )
    return value


def _state_section(
    document: Mapping[str, Any],
    field: str,
    phase: str,
    mode: str,
) -> Mapping[str, Any]:
    """Read one initial/final property state."""

    value = document.get(field)
    if not isinstance(value, Mapping):
        raise InputValidationError(
            f"{phase} {mode} manifest.{field} must be an object."
        )
    nested = value.get("metrics")
    if isinstance(nested, Mapping):
        return nested
    return value


def _state_metric(
    state: Mapping[str, Any],
    field: str,
    label: str,
) -> float:
    """Read one required finite state metric."""

    return _finite_float(state.get(field), f"{label}.{field}")


def _state_vector(
    state: Mapping[str, Any],
    vector_field: str,
    scalar_prefix: str,
    length: int,
    label: str,
) -> tuple[float, ...]:
    """Read a vector field, accepting only the established scalar fallback."""

    field_aliases: Mapping[str, tuple[str, ...]] = {
        "total_force_vector_eV_per_A": (
            "total_force_vector_eV_per_A",
            "total_force_eV_per_A",
        ),
        "stress_voigt_eV_per_A3": (
            "stress_voigt_eV_per_A3",
            "stress_eV_per_A3",
        ),
    }
    value = next(
        (
            state[name]
            for name in field_aliases.get(vector_field, (vector_field,))
            if name in state
        ),
        None,
    )
    if value is not None:
        return _finite_vector(value, length, f"{label}.{vector_field}")
    components = (
        ("x", "y", "z")
        if length == 3
        else STRESS_COMPONENTS
    )
    scalar_values = [
        state.get(f"{scalar_prefix}_{component}")
        for component in components
    ]
    if all(value is not None for value in scalar_values):
        return tuple(
            _finite_float(
                value,
                f"{label}.{scalar_prefix}_{components[index]}",
            )
            for index, value in enumerate(scalar_values)
        )
    raise InputValidationError(
        f"{label} lacks {vector_field} and its component fallback."
    )


def _validate_state(
    state: Mapping[str, Any],
    label: str,
    number_of_atoms: int,
) -> dict[str, Any]:
    """Normalize one complete state without calculating new physical values."""

    energy = _state_metric(state, "total_energy_eV", label)
    energy_per_atom = _state_metric(state, "energy_per_atom_eV", label)
    maximum_force = _state_metric(state, "maximum_force_eV_per_A", label)
    rms_force = _state_metric(state, "rms_force_eV_per_A", label)
    total_force = _state_vector(
        state,
        "total_force_vector_eV_per_A",
        "total_force",
        3,
        label,
    )
    total_force_norm = _state_metric(state, "total_force_norm_eV_per_A", label)
    stress = _state_vector(
        state,
        "stress_voigt_eV_per_A3",
        "stress",
        6,
        label,
    )
    max_stress = state.get("maximum_absolute_stress_eV_per_A3")
    if max_stress is None:
        max_stress_value = max(abs(value) for value in stress)
    else:
        max_stress_value = _finite_float(
            max_stress, f"{label}.maximum_absolute_stress_eV_per_A3"
        )
    volume = _positive_float(state.get("volume_A3"), f"{label}.volume_A3")
    volume_per_atom = _positive_float(
        state.get("volume_per_atom_A3"), f"{label}.volume_per_atom_A3"
    )
    lengths = _finite_vector(
        state.get("lattice_lengths_A"), 3, f"{label}.lattice_lengths_A"
    )
    angles = _finite_vector(
        state.get("lattice_angles_deg"), 3, f"{label}.lattice_angles_deg"
    )
    if not math.isclose(
        energy_per_atom,
        energy / number_of_atoms,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise InputValidationError(
            f"{label} energy-per-atom is inconsistent with total energy."
        )
    if not math.isclose(
        volume_per_atom,
        volume / number_of_atoms,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise InputValidationError(
            f"{label} volume-per-atom is inconsistent with volume."
        )
    if not math.isclose(
        max_stress_value,
        max(abs(value) for value in stress),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise InputValidationError(
            f"{label} maximum absolute stress is inconsistent."
        )
    return {
        "total_energy_eV": energy,
        "energy_per_atom_eV": energy_per_atom,
        "maximum_force_eV_per_A": maximum_force,
        "rms_force_eV_per_A": rms_force,
        "total_force_vector_eV_per_A": total_force,
        "total_force_norm_eV_per_A": total_force_norm,
        "stress_voigt_eV_per_A3": stress,
        "maximum_absolute_stress_eV_per_A3": max_stress_value,
        "volume_A3": volume,
        "volume_per_atom_A3": volume_per_atom,
        "lattice_lengths_A": lengths,
        "lattice_angles_deg": angles,
    }


def _parse_history(
    path: Path,
    config: AnalysisConfiguration,
    phase: str,
    mode: str,
) -> HistoryTable:
    """Parse and validate one required convergence-history CSV."""

    identity = capture_file_identity(path, config.project_root, f"{phase} history")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise InputValidationError(
                    f"{phase} {mode} history has missing or duplicate headers."
                )
            raw_rows = list(reader)
            fieldnames = tuple(reader.fieldnames)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputValidationError(
            f"Could not parse {phase} {mode} history: {exc}"
        ) from exc
    if not raw_rows:
        raise InputValidationError(f"{phase} {mode} history contains no rows.")

    selected_names: dict[str, str] = {}
    for canonical, aliases in HISTORY_NUMERIC_ALIASES.items():
        match = next((alias for alias in aliases if alias in fieldnames), None)
        if match is None:
            raise InputValidationError(
                f"{phase} {mode} history lacks required column {canonical!r}."
            )
        selected_names[canonical] = match

    parsed: list[Mapping[str, Any]] = []
    previous_step = -1
    for row_index, raw in enumerate(raw_rows):
        try:
            step_value = int(raw[selected_names["step"]])
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                f"{phase} {mode} history row {row_index} has an invalid step."
            ) from exc
        if step_value != previous_step + 1:
            raise InputValidationError(
                f"{phase} {mode} history steps must be consecutive from zero; "
                f"found {step_value} after {previous_step}."
            )
        previous_step = step_value
        normalized: dict[str, Any] = {"step": step_value}
        for canonical, source in selected_names.items():
            if canonical == "step":
                continue
            try:
                numeric = float(raw[source])
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    f"{phase} {mode} history row {row_index} column {source} "
                    "is not numeric."
                ) from exc
            if not math.isfinite(numeric):
                raise InputValidationError(
                    f"{phase} {mode} history row {row_index} column {source} "
                    "is not finite."
                )
            normalized[canonical] = numeric
        if "phase_key" in fieldnames and raw["phase_key"] != phase:
            raise InputValidationError(
                f"{phase} {mode} history contains a different phase key."
            )
        if "mode" in fieldnames and raw["mode"] != mode:
            raise InputValidationError(
                f"{phase} {mode} history contains a different mode."
            )
        parsed.append(normalized)
    return HistoryTable(
        path=path,
        identity=identity,
        fieldnames=fieldnames,
        rows=tuple(parsed),
    )


def _extract_unique_report_value(text: str, key: str) -> str | None:
    """Extract one unique anchored ``Key: value`` report field."""

    matches = re.findall(
        rf"(?im)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$",
        text,
    )
    if not matches:
        return None
    values = {value.strip() for value in matches}
    if len(values) != 1:
        raise InputValidationError(f"Report contains conflicting {key!r} lines.")
    return next(iter(values))


def _validate_report_agreement(
    path: Path,
    phase: str,
    mode: str,
    execution_status: str,
    convergence_status: str,
    safety_status: str,
    force_converged: bool,
    stress_converged: bool | None,
) -> None:
    """Require the human-readable report to agree with machine status."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InputValidationError(
            f"Could not read {phase} {mode} report: {exc}"
        ) from exc
    if not text.strip():
        raise InputValidationError(f"{phase} {mode} report is empty.")
    # The shared runner deliberately distinguishes safe execution from
    # scientific convergence.  Its human report exposes "Phase status" plus
    # the individual force/stress/overall checks; execution completion remains
    # machine-readable in the manifest.
    for key, expected in (
        ("Phase status", convergence_status),
        ("Safety status", safety_status),
    ):
        observed = _extract_unique_report_value(text, key)
        if observed is None:
            raise InputValidationError(
                f"{phase} {mode} report lacks the required {key!r} line."
            )
        if observed.upper() != expected.upper():
            raise InputValidationError(
                f"{phase} {mode} report {key!r} is {observed!r}; "
                f"manifest says {expected!r}."
            )
    if execution_status != "COMPLETED":
        raise InputValidationError(
            f"{phase} {mode} report cannot confirm an incomplete execution."
        )
    force_expected = "PASS" if force_converged else "FAIL"
    force_observed = _extract_unique_report_value(text, "Force convergence")
    if force_observed != force_expected:
        raise InputValidationError(
            f"{phase} {mode} report force convergence disagrees with manifest."
        )
    overall_observed = _extract_unique_report_value(text, "Overall convergence")
    expected_overall = (
        "PASS" if convergence_status in CONVERGED_STATUSES else "FAIL"
    )
    if overall_observed != expected_overall:
        raise InputValidationError(
            f"{phase} {mode} report overall convergence disagrees with manifest."
        )
    if mode == "atomic_only":
        stress_observed = _extract_unique_report_value(text, "Stress convergence")
        if stress_observed != "Not required in atomic-only mode":
            raise InputValidationError(
                f"{phase} atomic-only report has an invalid stress status."
            )
    else:
        stress_expected = "PASS" if stress_converged else "FAIL"
        stress_observed = _extract_unique_report_value(text, "Stress convergence")
        if stress_observed != stress_expected:
            raise InputValidationError(
                f"{phase} full-cell report stress convergence disagrees with "
                "manifest."
            )
    lower = text.lower()
    if "mace" not in lower or "dft" not in lower:
        raise InputValidationError(
            f"{phase} {mode} report lacks the required MACE/DFT limitation."
        )


def _validate_trajectory(
    path: Path,
    source_atoms: Any,
    final_atoms: Any,
    expected_frames: int,
    phase: str,
    mode: str,
    np: Any,
    trajectory_api: Any,
) -> None:
    """Validate trajectory length, identity, and endpoint geometries."""

    try:
        with trajectory_api(str(path), mode="r") as trajectory:
            frame_count = len(trajectory)
            if frame_count != expected_frames:
                raise InputValidationError(
                    f"{phase} {mode} trajectory has {frame_count} frames; "
                    f"expected {expected_frames}."
                )
            first = trajectory[0]
            last = trajectory[-1]
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Could not validate {phase} {mode} trajectory: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for frame, reference, label in (
        (first, source_atoms, "first"),
        (last, final_atoms, "last"),
    ):
        if tuple(frame.get_atomic_numbers()) != tuple(reference.get_atomic_numbers()):
            raise InputValidationError(
                f"{phase} {mode} trajectory {label} frame changes atom identity."
            )
        if not np.allclose(
            frame.get_positions(),
            reference.get_positions(),
            atol=1e-10,
            rtol=0.0,
        ):
            raise InputValidationError(
                f"{phase} {mode} trajectory {label} positions disagree."
            )
        if not np.allclose(
            frame.cell.array,
            reference.cell.array,
            atol=1e-10,
            rtol=0.0,
        ):
            raise InputValidationError(
                f"{phase} {mode} trajectory {label} cell disagrees."
            )


def _validate_manifest_provenance(
    document: Mapping[str, Any],
    source_identity: FileIdentity,
    config: AnalysisConfiguration,
    phase: str,
    mode: str,
) -> None:
    """Validate that every relaxation started from the original selected file."""

    provenance = document.get("provenance")
    source: Mapping[str, Any] | None = None
    if isinstance(provenance, Mapping):
        role = provenance.get("starting_structure_role")
        if role != "original_selected":
            raise InputValidationError(
                f"{phase} {mode} starting_structure_role must be "
                "'original_selected'."
            )
        candidate = provenance.get("source_structure")
        if isinstance(candidate, Mapping):
            source = candidate
    else:
        # The shared runner records the complete protected-file set.  Locate
        # the exact original selected EXTXYZ within that immutable manifest.
        protected = document.get("protected_sources")
        if not isinstance(protected, list):
            raise InputValidationError(
                f"{phase} {mode} lacks protected source provenance."
            )
        seen_paths: set[Path] = set()
        for candidate in protected:
            if not isinstance(candidate, Mapping):
                raise InputValidationError(
                    f"{phase} {mode} protected source entry is not an object."
                )
            path_value = candidate.get("path")
            if not isinstance(path_value, str):
                raise InputValidationError(
                    f"{phase} {mode} protected source path is invalid."
                )
            resolved = _resolve_repository_path(
                config.project_root,
                path_value,
                f"{phase} {mode} protected source",
            )
            if resolved in seen_paths:
                raise InputValidationError(
                    f"{phase} {mode} protected source path is duplicated: "
                    f"{path_value}"
                )
            seen_paths.add(resolved)
            current = capture_file_identity(
                resolved,
                config.project_root,
                f"{phase} {mode} protected source",
            )
            if (
                candidate.get("sha256") != current.sha256
                or candidate.get("size_bytes") != current.size_bytes
                or candidate.get("modification_time_ns")
                != current.modification_time_ns
            ):
                raise InputValidationError(
                    f"{phase} {mode} protected source changed after relaxation: "
                    f"{path_value}"
                )
            if resolved == source_identity.path.resolve():
                source = candidate
    if source is None:
        raise InputValidationError(
            f"{phase} {mode} provenance does not identify its original "
            "selected EXTXYZ."
        )
    path_value = source.get("path")
    if not isinstance(path_value, str):
        raise InputValidationError(
            f"{phase} {mode} source structure path is missing."
        )
    resolved = _resolve_repository_path(
        config.project_root, path_value, f"{phase} {mode} source structure"
    )
    if resolved != source_identity.path.resolve():
        raise InputValidationError(
            f"{phase} {mode} did not start from the configured original structure."
        )
    if source.get("sha256") != source_identity.sha256:
        raise InputValidationError(
            f"{phase} {mode} source structure content fingerprint changed."
        )
    if source.get("size_bytes") != source_identity.size_bytes:
        raise InputValidationError(
            f"{phase} {mode} source structure size changed."
        )
    recorded_mtime = source.get("modification_time_ns")
    if (
        recorded_mtime is not None
        and recorded_mtime != source_identity.modification_time_ns
    ):
        raise InputValidationError(
            f"{phase} {mode} source structure modification time changed."
        )


def _validate_phase_result(
    config: AnalysisConfiguration,
    phase: str,
    mode: str,
    source_atoms: Any,
    source_identity: FileIdentity,
    np: Any,
    ase_read: Any,
    trajectory_api: Any,
) -> PhaseResult:
    """Validate one complete Step 6C or Step 6D phase bundle."""

    manifest_path = _phase_manifest_path(config, phase, mode)
    document = read_json_object(
        manifest_path, f"{phase} {mode} result manifest"
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise InputValidationError(
            f"{phase} {mode} result schema_version must equal {SCHEMA_VERSION!r}."
        )
    expected_step = "6C" if mode == "atomic_only" else "6D"
    if document.get("project_step") != expected_step:
        raise InputValidationError(
            f"{phase} {mode} project_step must equal {expected_step!r}."
        )
    record_type = document.get("record_type", document.get("artifact_type"))
    if record_type not in {
        "ni_al_mace_relaxation_phase_result",
        "ni_al_relaxation_phase_result",
    }:
        raise InputValidationError(
            f"{phase} {mode} has an unsupported record_type: {record_type!r}."
        )
    if document.get("mode") != mode:
        raise InputValidationError(
            f"{phase} result mode is {document.get('mode')!r}; expected {mode!r}."
        )

    identity = _manifest_identity_section(document, phase)
    if identity.get("phase_key") != phase:
        raise InputValidationError(f"{phase} {mode} manifest phase key mismatch.")
    if identity.get("material_id") != config.expected_material_ids[phase]:
        raise InputValidationError(f"{phase} {mode} material ID mismatch.")
    if identity.get("number_of_atoms") != config.expected_atom_counts[phase]:
        raise InputValidationError(f"{phase} {mode} atom-count mismatch.")

    file_hash, semantic_hash = _manifest_configuration_fingerprints(document, phase)
    if file_hash != config.config_identity.sha256:
        raise InputValidationError(
            f"{phase} {mode} was created from a different configuration file."
        )
    if (
        semantic_hash is not None
        and semantic_hash != config.semantic_fingerprint_sha256
    ):
        raise InputValidationError(
            f"{phase} {mode} semantic configuration fingerprint differs."
        )

    execution_value = document.get("execution")
    if isinstance(execution_value, Mapping):
        execution = dict(execution_value)
    else:
        optimizer_section = document.get("optimizer")
        counts_section = document.get("counts")
        timing_section = document.get("timing")
        history_document = document.get("history")
        if not isinstance(optimizer_section, Mapping):
            optimizer_section = {}
        if not isinstance(counts_section, Mapping):
            counts_section = {}
        if not isinstance(timing_section, Mapping):
            timing_section = {}
        history_count = len(history_document) if isinstance(history_document, list) else 0
        execution = {
            "execution_status": document.get("execution_status"),
            "convergence_status": document.get("convergence_status"),
            "force_converged": document.get("force_converged"),
            "stress_converged": document.get("stress_converged"),
            "safety_status": document.get("safety_status"),
            "optimizer_steps": optimizer_section.get(
                "steps", counts_section.get("optimizer_steps")
            ),
            "history_rows": history_count,
            "trajectory_frames": history_count,
            "wall_clock_duration_s": timing_section.get("wall_time_seconds"),
            "optimizer_created": optimizer_section.get("created"),
        }
    execution_status = _require_string(
        execution, "execution_status", f"{phase} {mode}.execution"
    ).upper()
    convergence_status = _require_string(
        execution, "convergence_status", f"{phase} {mode}.execution"
    ).upper()
    safety_status = _require_string(
        execution, "safety_status", f"{phase} {mode}.execution"
    ).upper()
    if execution_status != "COMPLETED":
        raise InputValidationError(
            f"{phase} {mode} execution is not completed: {execution_status}."
        )
    if convergence_status not in CONVERGENCE_STATUSES:
        raise InputValidationError(
            f"{phase} {mode} convergence status is invalid: {convergence_status}."
        )
    if safety_status != "PASS":
        raise InputValidationError(
            f"{phase} {mode} safety status is not PASS."
        )
    optimizer_steps = _nonnegative_int(
        execution.get("optimizer_steps"),
        f"{phase} {mode}.execution.optimizer_steps",
    )
    history_rows = _nonnegative_int(
        execution.get("history_rows"),
        f"{phase} {mode}.execution.history_rows",
    )
    trajectory_frames = _nonnegative_int(
        execution.get("trajectory_frames"),
        f"{phase} {mode}.execution.trajectory_frames",
    )
    if history_rows != optimizer_steps + 1 or trajectory_frames != history_rows:
        raise InputValidationError(
            f"{phase} {mode} step/history/trajectory counts are inconsistent."
        )
    _finite_float(
        execution.get("wall_clock_duration_s"),
        f"{phase} {mode}.execution.wall_clock_duration_s",
    )

    initial_raw = _state_section(document, "initial", phase, mode)
    final_raw = _state_section(document, "final", phase, mode)
    initial = _validate_state(
        initial_raw,
        f"{phase} {mode}.initial",
        config.expected_atom_counts[phase],
    )
    final = _validate_state(
        final_raw,
        f"{phase} {mode}.final",
        config.expected_atom_counts[phase],
    )
    changes_value = document.get("changes", {})
    displacements_value = document.get(
        "displacements",
        document.get(
            "displacement",
            final_raw.get(
                "displacement",
                changes_value.get("displacement", {})
                if isinstance(changes_value, Mapping)
                else {},
            ),
        ),
    )
    if not isinstance(changes_value, Mapping) or not isinstance(
        displacements_value, Mapping
    ):
        raise InputValidationError(
            f"{phase} {mode} changes/displacements must be objects."
        )
    strain_value = document.get("strain")
    if strain_value is not None and not isinstance(strain_value, Mapping):
        raise InputValidationError(f"{phase} {mode} strain must be object or null.")

    force_ok = final["maximum_force_eV_per_A"] <= config.force_threshold_eV_per_A
    stress_ok = (
        None
        if mode == "atomic_only"
        else final["maximum_absolute_stress_eV_per_A3"]
        <= config.stress_threshold_eV_per_A3
    )
    expected_converged = force_ok and (stress_ok is None or stress_ok)
    recorded_force = execution.get("force_converged")
    recorded_stress = execution.get("stress_converged")
    if recorded_force is not force_ok:
        raise InputValidationError(
            f"{phase} {mode} force convergence flag is inconsistent."
        )
    if mode == "atomic_only":
        if recorded_stress is not None:
            raise InputValidationError(
                f"{phase} atomic-only stress_converged must be null."
            )
    elif recorded_stress is not stress_ok:
        raise InputValidationError(
            f"{phase} full-cell stress convergence flag is inconsistent."
        )
    if (convergence_status in CONVERGED_STATUSES) != expected_converged:
        raise InputValidationError(
            f"{phase} {mode} convergence status disagrees with final criteria."
        )
    if convergence_status == "ALREADY_CONVERGED" and optimizer_steps != 0:
        raise InputValidationError(
            f"{phase} {mode} ALREADY_CONVERGED result has optimizer steps."
        )

    companions = _expected_companion_paths(config, phase, mode)
    outputs_value = document.get("outputs", document.get("artifacts"))
    if not isinstance(outputs_value, Mapping):
        raise InputValidationError(
            f"{phase} {mode} manifest lacks its artifact inventory."
        )
    outputs = outputs_value
    for name, expected in companions.items():
        _output_descriptor(outputs, name, expected, config, phase)

    structure_path = companions["structure"]
    trajectory_path = companions["trajectory"]
    history_path = companions["history"]
    report_path = companions["report"]
    final_atoms = _read_atoms(structure_path, f"{phase} {mode} final structure", ase_read)
    _validate_atoms_identity(
        final_atoms,
        phase,
        config.expected_atom_counts[phase],
        np,
        f"{phase} {mode} final structure",
    )
    if final_atoms.calc is not None:
        raise InputValidationError(
            f"{phase} {mode} final structure retains a calculator."
        )
    if tuple(final_atoms.get_atomic_numbers()) != tuple(source_atoms.get_atomic_numbers()):
        raise InputValidationError(
            f"{phase} {mode} final atom ordering differs from the source."
        )
    if mode == "atomic_only":
        if not np.allclose(
            final_atoms.cell.array,
            source_atoms.cell.array,
            atol=1e-12,
            rtol=0.0,
        ):
            raise InputValidationError(f"{phase} atomic-only cell changed.")
        if not math.isclose(
            float(final_atoms.get_volume()),
            float(source_atoms.get_volume()),
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise InputValidationError(f"{phase} atomic-only volume changed.")

    final_lengths = tuple(float(value) for value in final_atoms.cell.lengths())
    final_angles = tuple(float(value) for value in final_atoms.cell.angles())
    if not np.allclose(
        final_lengths,
        final["lattice_lengths_A"],
        atol=1e-9,
        rtol=1e-10,
    ):
        raise InputValidationError(
            f"{phase} {mode} final lattice lengths disagree with its manifest."
        )
    if not np.allclose(
        final_angles,
        final["lattice_angles_deg"],
        atol=1e-8,
        rtol=1e-10,
    ):
        raise InputValidationError(
            f"{phase} {mode} final lattice angles disagree with its manifest."
        )
    if not math.isclose(
        float(final_atoms.get_volume()),
        final["volume_A3"],
        abs_tol=1e-9,
        rel_tol=1e-10,
    ):
        raise InputValidationError(
            f"{phase} {mode} final volume disagrees with its manifest."
        )

    _validate_manifest_provenance(
        document, source_identity, config, phase, mode
    )
    history = _parse_history(history_path, config, phase, mode)
    if len(history.rows) != history_rows:
        raise InputValidationError(
            f"{phase} {mode} history row count disagrees with its manifest."
        )
    final_history = history.rows[-1]
    for field in (
        "total_energy_eV",
        "energy_per_atom_eV",
        "maximum_force_eV_per_A",
        "rms_force_eV_per_A",
        "maximum_absolute_stress_eV_per_A3",
        "volume_A3",
    ):
        if not math.isclose(
            float(final_history[field]),
            float(final[field]),
            abs_tol=1e-9,
            rel_tol=1e-9,
        ):
            raise InputValidationError(
                f"{phase} {mode} final history {field} disagrees with manifest."
            )
    _validate_report_agreement(
        report_path,
        phase,
        mode,
        execution_status,
        convergence_status,
        safety_status,
        force_ok,
        stress_ok,
    )
    _validate_trajectory(
        trajectory_path,
        source_atoms,
        final_atoms,
        trajectory_frames,
        phase,
        mode,
        np,
        trajectory_api,
    )
    manifest_identity = capture_file_identity(
        manifest_path, config.project_root, f"{phase} {mode} manifest"
    )
    return PhaseResult(
        phase_key=phase,
        mode=mode,
        manifest_path=manifest_path,
        manifest_identity=manifest_identity,
        document=document,
        identity=identity,
        execution={
            **execution,
            "execution_status": execution_status,
            "convergence_status": convergence_status,
            "safety_status": safety_status,
            "optimizer_steps": optimizer_steps,
        },
        initial=initial,
        final=final,
        changes=changes_value,
        displacements=displacements_value,
        strain=strain_value,
        structure_path=structure_path,
        trajectory_path=trajectory_path,
        history=history,
        report_path=report_path,
        source_structure_path=source_identity.path,
        final_atoms=final_atoms,
    )


def _validate_all_inputs(
    config: AnalysisConfiguration,
) -> tuple[FileIdentity, tuple[PhaseInputs, ...]]:
    """Validate Step 5 and every Step 6C/6D phase bundle."""

    step5_identity, step5_records = _validate_step5_table(config)
    np, ase_read, trajectory_api, _symmetry_api, _unused = (
        _lazy_import_scientific_dependencies()
    )
    phases: list[PhaseInputs] = []
    for phase in config.phase_order:
        source_path = config.selected_directory / f"{phase}.extxyz"
        source_identity = capture_file_identity(
            source_path, config.project_root, f"{phase} selected structure"
        )
        source_atoms = _read_atoms(
            source_path, f"{phase} selected structure", ase_read
        )
        _validate_atoms_identity(
            source_atoms,
            phase,
            config.expected_atom_counts[phase],
            np,
            f"{phase} selected structure",
        )
        if source_atoms.calc is not None:
            raise InputValidationError(
                f"{phase} selected structure retains a calculator after reading."
            )
        atomic = _validate_phase_result(
            config,
            phase,
            "atomic_only",
            source_atoms,
            source_identity,
            np,
            ase_read,
            trajectory_api,
        )
        full = _validate_phase_result(
            config,
            phase,
            "full_cell",
            source_atoms,
            source_identity,
            np,
            ase_read,
            trajectory_api,
        )
        phases.append(
            PhaseInputs(
                phase_key=phase,
                step5_record=step5_records[phase],
                source_atoms=source_atoms,
                source_identity=source_identity,
                atomic_only=atomic,
                full_cell=full,
            )
        )
    assert_mace_not_imported()
    return step5_identity, tuple(phases)


def validate_plan(
    config: str | Path = Path("configs/mace_relaxation.json"),
    *,
    require_inputs: bool,
    overwrite: bool = False,
) -> AnalysisPlan:
    """Validate the static or complete Step 6E plan.

    Parameters
    ----------
    config
        Repository-relative or absolute configuration path.
    require_inputs
        ``False`` validates only configuration and future paths.  ``True``
        validates Step 5 and all ten C/D phase bundles.
    overwrite
        Whether existing exact Step 6E targets are authorized for replacement.
        Static validation reports collisions but does not replace anything.
    """

    assert_mace_not_imported()
    _optional_shared_utils()
    project_root = locate_project_root()
    config_path = _configuration_path(project_root, Path(config))
    validated_config = validate_configuration(config_path)
    output_plan = build_output_plan(validated_config)
    _validate_output_target_safety(output_plan, validated_config)
    collisions = _find_collisions(output_plan)
    if any(path.is_symlink() or not path.is_file() for path in collisions):
        _validate_collision_policy(
            output_plan, validated_config, overwrite=overwrite
        )
    step5_identity: FileIdentity | None = None
    phases: tuple[PhaseInputs, ...] = ()
    if require_inputs:
        step5_identity, phases = _validate_all_inputs(validated_config)
    assert_mace_not_imported()
    return AnalysisPlan(
        config=validated_config,
        outputs=output_plan,
        collisions=collisions,
        step5_identity=step5_identity,
        phases=phases,
        inputs_validated=require_inputs,
    )


def validation_plan(
    config: str | Path = Path("configs/mace_relaxation.json"),
    *,
    require_inputs: bool,
    overwrite: bool = False,
) -> AnalysisPlan:
    """Compatibility alias for pipeline callers."""

    return validate_plan(
        config,
        require_inputs=require_inputs,
        overwrite=overwrite,
    )


def _symmetry(atoms: Any, config: AnalysisConfiguration) -> SymmetryResult:
    """Compute one tolerance-dependent space-group identity."""

    _np, _ase_read, _trajectory, symmetry_api, _unused = (
        _lazy_import_scientific_dependencies()
    )
    adaptor, analyzer_class, error_class = symmetry_api
    try:
        structure = adaptor.get_structure(atoms)
        analyzer = analyzer_class(
            structure,
            symprec=config.symmetry_symprec_A,
            angle_tolerance=config.symmetry_angle_tolerance_deg,
        )
        symbol = str(analyzer.get_space_group_symbol())
        number = int(analyzer.get_space_group_number())
    except error_class as exc:
        raise InputValidationError(
            f"Symmetry could not be determined at symprec="
            f"{config.symmetry_symprec_A} A and angle_tolerance="
            f"{config.symmetry_angle_tolerance_deg} deg: {exc}"
        ) from exc
    except Exception as exc:
        raise InputValidationError(
            f"Symmetry analysis failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not symbol or number < 1 or number > 230:
        raise InputValidationError("Symmetry analysis returned an invalid space group.")
    return SymmetryResult(symbol=symbol, number=number)


def _source_geometry_state(phase: PhaseInputs) -> dict[str, Any]:
    """Combine Step 5 numeric values with source-only lattice geometry."""

    record = phase.step5_record
    stress = tuple(
        _finite_float(
            record.get(f"stress_{component}_eV_per_A3"),
            f"Step 5 {phase.phase_key}.stress_{component}",
        )
        for component in STRESS_COMPONENTS
    )
    return {
        "total_energy_eV": _finite_float(
            record.get("total_energy_eV"), f"Step 5 {phase.phase_key}.energy"
        ),
        "energy_per_atom_eV": _finite_float(
            record.get("energy_per_atom_eV"),
            f"Step 5 {phase.phase_key}.energy_per_atom",
        ),
        "maximum_force_eV_per_A": _finite_float(
            record.get("maximum_force_eV_per_A"),
            f"Step 5 {phase.phase_key}.maximum_force",
        ),
        "rms_force_eV_per_A": _finite_float(
            record.get("rms_force_eV_per_A"),
            f"Step 5 {phase.phase_key}.rms_force",
        ),
        "total_force_vector_eV_per_A": tuple(
            _finite_float(
                record.get(f"total_force_{component}_eV_per_A"),
                f"Step 5 {phase.phase_key}.total_force_{component}",
            )
            for component in ("x", "y", "z")
        ),
        "total_force_norm_eV_per_A": _finite_float(
            record.get("total_force_norm_eV_per_A"),
            f"Step 5 {phase.phase_key}.total_force_norm",
        ),
        "stress_voigt_eV_per_A3": stress,
        "maximum_absolute_stress_eV_per_A3": max(abs(value) for value in stress),
        "volume_A3": _positive_float(
            record.get("volume_A3"), f"Step 5 {phase.phase_key}.volume"
        ),
        "volume_per_atom_A3": _positive_float(
            record.get("volume_per_atom_A3"),
            f"Step 5 {phase.phase_key}.volume_per_atom",
        ),
        "lattice_lengths_A": tuple(
            float(value) for value in phase.source_atoms.cell.lengths()
        ),
        "lattice_angles_deg": tuple(
            float(value) for value in phase.source_atoms.cell.angles()
        ),
    }


def _displacement_metrics(result: PhaseResult) -> dict[str, Any]:
    """Normalize mode-specific displacement values from the result manifest."""

    values = result.displacements

    def optional(*names: str) -> float | None:
        for name in names:
            if name in values and values[name] is not None:
                return _finite_float(
                    values[name], f"{result.phase_key} {result.mode}.{name}"
                )
        return None

    max_internal = optional(
        "maximum_internal_displacement_A",
        "maximum_displacement_A",
        "max_displacement_A",
        "maximum_internal_A",
    )
    rms_internal = optional(
        "rms_internal_displacement_A",
        "rms_displacement_A",
        "rms_internal_A",
    )
    max_total = optional(
        "maximum_total_cartesian_displacement_A",
        "maximum_total_A",
    )
    rms_total = optional(
        "rms_total_cartesian_displacement_A",
        "rms_total_A",
    )
    if result.mode == "atomic_only":
        if max_internal is None or rms_internal is None:
            raise InputValidationError(
                f"{result.phase_key} atomic-only displacement metrics are missing."
            )
        max_total = max_internal if max_total is None else max_total
        rms_total = rms_internal if rms_total is None else rms_total
    elif any(value is None for value in (max_internal, rms_internal, max_total, rms_total)):
        raise InputValidationError(
            f"{result.phase_key} full-cell displacement metrics are incomplete."
        )
    return {
        "maximum_internal_displacement_A": max_internal,
        "rms_internal_displacement_A": rms_internal,
        "maximum_total_cartesian_displacement_A": max_total,
        "rms_total_cartesian_displacement_A": rms_total,
    }


def _build_stage_row(
    *,
    phase_index: int,
    phase: PhaseInputs,
    stage: str,
    state: Mapping[str, Any],
    initial: Mapping[str, Any],
    symmetry: SymmetryResult,
    initial_symmetry: SymmetryResult,
    result: PhaseResult | None,
) -> dict[str, Any]:
    """Build one long-form comparison row."""

    stress = tuple(state["stress_voigt_eV_per_A3"])
    lengths = tuple(state["lattice_lengths_A"])
    angles = tuple(state["lattice_angles_deg"])
    total_force = tuple(state["total_force_vector_eV_per_A"])
    if result is None:
        result_class = "BASELINE"
        diagnostic = False
        convergence_status = "BASELINE"
        safety_status = "NOT_APPLICABLE"
        optimizer_steps = 0
        wall_clock = 0.0
        atomic_initial_raw = phase.atomic_only.document.get("initial", {})
        full_initial_raw = phase.full_cell.document.get("initial", {})
        if not isinstance(atomic_initial_raw, Mapping) or not isinstance(
            full_initial_raw, Mapping
        ):
            raise InputValidationError(
                f"{phase.phase_key} manifests lack initial convergence flags."
            )
        force_flag = atomic_initial_raw.get("force_converged")
        stress_flag = full_initial_raw.get("stress_converged")
        if not isinstance(force_flag, bool) or not isinstance(stress_flag, bool):
            raise InputValidationError(
                f"{phase.phase_key} initial convergence flags are invalid."
            )
        force_converged = force_flag
        stress_converged = stress_flag
        displacements = {
            "maximum_internal_displacement_A": 0.0,
            "rms_internal_displacement_A": 0.0,
            "maximum_total_cartesian_displacement_A": 0.0,
            "rms_total_cartesian_displacement_A": 0.0,
        }
        source_result_path = phase.source_identity.relative_path
        displacement_definition = "baseline_zero"
    else:
        convergence_status = str(result.execution["convergence_status"])
        accepted = convergence_status in CONVERGED_STATUSES
        result_class = (
            "ACCEPTED_CONVERGED" if accepted else "DIAGNOSTIC_NOT_CONVERGED"
        )
        diagnostic = not accepted
        safety_status = str(result.execution["safety_status"])
        optimizer_steps = int(result.execution["optimizer_steps"])
        wall_clock = _finite_float(
            result.execution.get("wall_clock_duration_s"),
            f"{phase.phase_key} {stage}.wall_clock_duration_s",
        )
        force_converged = bool(result.execution["force_converged"])
        raw_stress_flag = result.execution.get("stress_converged")
        stress_converged = (
            None if raw_stress_flag is None else bool(raw_stress_flag)
        )
        displacements = _displacement_metrics(result)
        source_result_path = result.manifest_identity.relative_path
        displacement_definition = (
            "fixed-cell periodic minimum-image"
            if stage == "atomic_only"
            else "initial-cell internal-coordinate minimum-image"
        )
    preserved_symbol = symmetry.symbol == initial_symmetry.symbol
    preserved_number = symmetry.number == initial_symmetry.number
    max_displacement = displacements["maximum_internal_displacement_A"]
    rms_displacement = displacements["rms_internal_displacement_A"]
    return {
        "phase_order_index": phase_index,
        "phase_key": phase.phase_key,
        "formula": str(phase.step5_record.get("formula", phase.phase_key)),
        "material_id": str(phase.step5_record["material_id"]),
        "number_of_atoms": int(phase.step5_record["number_of_atoms"]),
        "stage": stage,
        "result_class": result_class,
        "diagnostic_only": diagnostic,
        "total_energy_eV": state["total_energy_eV"],
        "energy_per_atom_eV": state["energy_per_atom_eV"],
        "energy_change_from_initial_eV": (
            state["total_energy_eV"] - initial["total_energy_eV"]
        ),
        "energy_change_from_initial_eV_per_atom": (
            state["energy_per_atom_eV"] - initial["energy_per_atom_eV"]
        ),
        "maximum_force_eV_per_A": state["maximum_force_eV_per_A"],
        "rms_force_eV_per_A": state["rms_force_eV_per_A"],
        "total_force_x_eV_per_A": total_force[0],
        "total_force_y_eV_per_A": total_force[1],
        "total_force_z_eV_per_A": total_force[2],
        "total_force_norm_eV_per_A": state["total_force_norm_eV_per_A"],
        "stress_xx_eV_per_A3": stress[0],
        "stress_yy_eV_per_A3": stress[1],
        "stress_zz_eV_per_A3": stress[2],
        "stress_yz_eV_per_A3": stress[3],
        "stress_xz_eV_per_A3": stress[4],
        "stress_xy_eV_per_A3": stress[5],
        "maximum_absolute_stress_eV_per_A3": state[
            "maximum_absolute_stress_eV_per_A3"
        ],
        "volume_A3": state["volume_A3"],
        "volume_per_atom_A3": state["volume_per_atom_A3"],
        "volume_change_from_initial_A3": (
            state["volume_A3"] - initial["volume_A3"]
        ),
        "volume_change_from_initial_percent": (
            100.0 * (state["volume_A3"] - initial["volume_A3"])
            / initial["volume_A3"]
        ),
        "a_A": lengths[0],
        "b_A": lengths[1],
        "c_A": lengths[2],
        "alpha_deg": angles[0],
        "beta_deg": angles[1],
        "gamma_deg": angles[2],
        "maximum_displacement_A": max_displacement,
        "rms_displacement_A": rms_displacement,
        "displacement_definition": displacement_definition,
        **displacements,
        "optimizer_steps": optimizer_steps,
        "wall_clock_duration_s": wall_clock,
        "force_converged": force_converged,
        "stress_converged": stress_converged,
        "overall_convergence_status": convergence_status,
        "safety_status": safety_status,
        "space_group_symbol": symmetry.symbol,
        "space_group_number": symmetry.number,
        "space_group_symbol_preserved": preserved_symbol,
        "space_group_number_preserved": preserved_number,
        "symmetry_preserved": preserved_symbol and preserved_number,
        "source_result_path": source_result_path,
    }


def derive_analysis(plan: AnalysisPlan) -> AnalysisData:
    """Compute comparisons and symmetry strictly from validated stored data."""

    if not plan.inputs_validated or plan.step5_identity is None:
        raise InputValidationError(
            "Complete input validation is required before deriving Step 6E data."
        )
    rows: list[Mapping[str, Any]] = []
    phase_records: list[Mapping[str, Any]] = []
    histories: dict[tuple[str, str], HistoryTable] = {}
    diagnostics: dict[str, list[str]] = {mode: [] for mode in MODES}
    for phase_index, phase in enumerate(plan.phases):
        initial = _source_geometry_state(phase)
        initial_symmetry = _symmetry(phase.source_atoms, plan.config)
        atomic_symmetry = _symmetry(phase.atomic_only.final_atoms, plan.config)
        full_symmetry = _symmetry(phase.full_cell.final_atoms, plan.config)
        stage_rows = (
            _build_stage_row(
                phase_index=phase_index,
                phase=phase,
                stage="initial",
                state=initial,
                initial=initial,
                symmetry=initial_symmetry,
                initial_symmetry=initial_symmetry,
                result=None,
            ),
            _build_stage_row(
                phase_index=phase_index,
                phase=phase,
                stage="atomic_only",
                state=phase.atomic_only.final,
                initial=initial,
                symmetry=atomic_symmetry,
                initial_symmetry=initial_symmetry,
                result=phase.atomic_only,
            ),
            _build_stage_row(
                phase_index=phase_index,
                phase=phase,
                stage="full_cell",
                state=phase.full_cell.final,
                initial=initial,
                symmetry=full_symmetry,
                initial_symmetry=initial_symmetry,
                result=phase.full_cell,
            ),
        )
        rows.extend(stage_rows)
        atomic_row, full_row = stage_rows[1], stage_rows[2]
        for mode, row in (
            ("atomic_only", atomic_row),
            ("full_cell", full_row),
        ):
            if row["diagnostic_only"]:
                diagnostics[mode].append(phase.phase_key)
        phase_records.append(
            {
                "identity": {
                    "phase_order_index": phase_index,
                    "phase_key": phase.phase_key,
                    "formula": stage_rows[0]["formula"],
                    "material_id": stage_rows[0]["material_id"],
                    "number_of_atoms": stage_rows[0]["number_of_atoms"],
                },
                "initial": dict(stage_rows[0]),
                "atomic_only": dict(atomic_row),
                "full_cell": dict(full_row),
                "deltas": {
                    "atomic_only_minus_initial": {
                        "total_energy_eV": atomic_row[
                            "energy_change_from_initial_eV"
                        ],
                        "energy_per_atom_eV": atomic_row[
                            "energy_change_from_initial_eV_per_atom"
                        ],
                        "volume_A3": atomic_row["volume_change_from_initial_A3"],
                        "volume_percent": atomic_row[
                            "volume_change_from_initial_percent"
                        ],
                    },
                    "full_cell_minus_initial": {
                        "total_energy_eV": full_row[
                            "energy_change_from_initial_eV"
                        ],
                        "energy_per_atom_eV": full_row[
                            "energy_change_from_initial_eV_per_atom"
                        ],
                        "volume_A3": full_row["volume_change_from_initial_A3"],
                        "volume_percent": full_row[
                            "volume_change_from_initial_percent"
                        ],
                    },
                    "full_cell_minus_atomic_only": {
                        "total_energy_eV": (
                            full_row["total_energy_eV"]
                            - atomic_row["total_energy_eV"]
                        ),
                        "energy_per_atom_eV": (
                            full_row["energy_per_atom_eV"]
                            - atomic_row["energy_per_atom_eV"]
                        ),
                        "volume_A3": full_row["volume_A3"] - atomic_row["volume_A3"],
                    },
                },
                "symmetry_comparison": {
                    "symprec_A": plan.config.symmetry_symprec_A,
                    "angle_tolerance_deg": (
                        plan.config.symmetry_angle_tolerance_deg
                    ),
                    "initial": initial_symmetry.as_json(),
                    "atomic_only": atomic_symmetry.as_json(),
                    "full_cell": full_symmetry.as_json(),
                    "atomic_only_preserved": atomic_row["symmetry_preserved"],
                    "full_cell_preserved": full_row["symmetry_preserved"],
                    "limitation": (
                        "Space-group detection is tolerance-dependent; no "
                        "symmetry constraint or refinement was applied."
                    ),
                },
            }
        )
        histories[("atomic_only", phase.phase_key)] = phase.atomic_only.history
        histories[("full_cell", phase.phase_key)] = phase.full_cell.history
    diagnostics_final = {
        mode: tuple(values) for mode, values in diagnostics.items()
    }
    analysis_status = (
        "PARTIAL"
        if any(diagnostics_final[mode] for mode in MODES)
        else "SUCCESS"
    )
    assert_mace_not_imported()
    return AnalysisData(
        generated_at_utc=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        rows=tuple(rows),
        phase_records=tuple(phase_records),
        histories=histories,
        analysis_status=analysis_status,
        diagnostic_phases=diagnostics_final,
    )


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize the long-form comparison table deterministically."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    ""
                    if row[field] is None
                    else str(row[field]).lower()
                    if isinstance(row[field], bool)
                    else row[field]
                )
                for field in CSV_FIELDS
            }
        )
    return buffer.getvalue()


def _json_text(
    plan: AnalysisPlan,
    data: AnalysisData,
) -> str:
    """Serialize the strict Step 6E JSON comparison document."""

    if plan.step5_identity is None:
        raise InputValidationError("Step 5 identity is unavailable.")
    input_results = {
        mode: [
            (
                phase.atomic_only.manifest_identity.as_json()
                if mode == "atomic_only"
                else phase.full_cell.manifest_identity.as_json()
            )
            for phase in plan.phases
        ]
        for mode in MODES
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "project_step": PROJECT_STEP,
        "record_type": RECORD_TYPE,
        "generated_at_utc": data.generated_at_utc,
        "phase_order": list(plan.config.phase_order),
        "stress_component_order": list(STRESS_COMPONENTS),
        "units": {
            "energy": "eV",
            "energy_per_atom": "eV/atom",
            "force": "eV/angstrom",
            "stress": "eV/angstrom^3",
            "volume": "angstrom^3",
            "length": "angstrom",
            "angle": "degree",
            "duration": "second",
        },
        "source_contract": {
            "step5_numeric_source": plan.step5_identity.as_json(),
            "atomic_only_numeric_sources": input_results["atomic_only"],
            "full_cell_numeric_sources": input_results["full_cell"],
            "histories_used_only_for_figures": True,
            "reports_used_only_for_status_agreement": True,
            "mace_rerun": False,
        },
        "configuration": {
            "file": plan.config.config_identity.as_json(),
            "semantic_fingerprint_sha256": (
                plan.config.semantic_fingerprint_sha256
            ),
            "force_threshold_eV_per_A": (
                plan.config.force_threshold_eV_per_A
            ),
            "stress_threshold_eV_per_A3": (
                plan.config.stress_threshold_eV_per_A3
            ),
        },
        "symmetry_settings": {
            "implementation": "pymatgen SpacegroupAnalyzer",
            "symprec_A": plan.config.symmetry_symprec_A,
            "angle_tolerance_deg": (
                plan.config.symmetry_angle_tolerance_deg
            ),
            "symmetry_constraint_applied": False,
            "tolerance_dependent": True,
        },
        "records": list(data.phase_records),
        "accepted_converged_phases": {
            mode: [
                phase.phase_key
                for phase in plan.phases
                if (
                    (
                        phase.atomic_only
                        if mode == "atomic_only"
                        else phase.full_cell
                    ).execution["convergence_status"]
                    in CONVERGED_STATUSES
                )
            ]
            for mode in MODES
        },
        "diagnostic_only_phases_by_mode": {
            mode: list(data.diagnostic_phases[mode]) for mode in MODES
        },
        "analysis_status": data.analysis_status,
        "output_inventory": [
            _path_for_console(path, plan.config.project_root)
            for path in plan.outputs.targets
        ],
        "scientific_interpretation_boundary": {
            "mace_potential_surface_convergence_only": True,
            "dft_accuracy_conclusion": False,
            "experimental_accuracy_conclusion": False,
            "raw_energy_phase_stability_ranking": False,
            "formation_energy_calculated": False,
            "step7_implemented": False,
        },
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
        raise PublicationError(f"Could not serialize comparison JSON: {exc}") from exc


def _report_text(plan: AnalysisPlan, data: AnalysisData) -> str:
    """Build the human-readable Step 6E comparison report."""

    lines = [
        "Step 6E - Ni-Al MACE Relaxation Comparison",
        "===========================================",
        "",
        "Scope and provenance",
        "--------------------",
        f"Generated at (UTC): {data.generated_at_utc}",
        "Initial numeric source: Step 5 zero-shot JSON table",
        "Atomic-only numeric source: validated Step 6C phase-result JSON",
        "Full-cell numeric source: validated Step 6D phase-result JSON",
        "MACE loaded or rerun: No",
        "Formation energy calculated: No",
        "",
        "Convergence criteria",
        "--------------------",
        (
            "Atomic-only force threshold: "
            f"{plan.config.force_threshold_eV_per_A:.16g} eV/angstrom"
        ),
        (
            "Full-cell force threshold: "
            f"{plan.config.force_threshold_eV_per_A:.16g} eV/angstrom"
        ),
        (
            "Full-cell stress threshold: "
            f"{plan.config.stress_threshold_eV_per_A3:.16g} "
            "eV/angstrom^3"
        ),
        (
            "A result is accepted as converged only when its machine-readable "
            "status, text report, and recomputed final criteria agree."
        ),
        "",
        "Symmetry method",
        "---------------",
        "Implementation: pymatgen SpacegroupAnalyzer",
        f"symprec: {plan.config.symmetry_symprec_A:.16g} angstrom",
        (
            "Angle tolerance: "
            f"{plan.config.symmetry_angle_tolerance_deg:.16g} degrees"
        ),
        "No symmetry constraint, refinement, or FixSymmetry was applied.",
        "Space-group detection is tolerance-dependent.",
        "",
        "Per-phase comparison",
        "--------------------",
    ]
    records_by_phase = {
        record["identity"]["phase_key"]: record for record in data.phase_records
    }
    for phase in plan.config.phase_order:
        record = records_by_phase[phase]
        initial = record["initial"]
        atomic = record["atomic_only"]
        full = record["full_cell"]
        symmetry = record["symmetry_comparison"]
        lines.extend(
            [
                "",
                phase,
                "~" * len(phase),
                f"Material ID: {initial['material_id']}",
                f"Atom count: {initial['number_of_atoms']}",
                (
                    "Initial total energy: "
                    f"{initial['total_energy_eV']:.16g} eV"
                ),
                (
                    "Atomic-only energy change: "
                    f"{atomic['energy_change_from_initial_eV']:.16g} eV"
                ),
                (
                    "Full-cell energy change: "
                    f"{full['energy_change_from_initial_eV']:.16g} eV"
                ),
                (
                    "Initial / atomic-only / full-cell maximum force: "
                    f"{initial['maximum_force_eV_per_A']:.16g} / "
                    f"{atomic['maximum_force_eV_per_A']:.16g} / "
                    f"{full['maximum_force_eV_per_A']:.16g} eV/angstrom"
                ),
                (
                    "Initial / atomic-only / full-cell maximum absolute stress: "
                    f"{initial['maximum_absolute_stress_eV_per_A3']:.16g} / "
                    f"{atomic['maximum_absolute_stress_eV_per_A3']:.16g} / "
                    f"{full['maximum_absolute_stress_eV_per_A3']:.16g} "
                    "eV/angstrom^3"
                ),
                (
                    "Atomic-only volume change: "
                    f"{atomic['volume_change_from_initial_percent']:.16g} %"
                ),
                (
                    "Full-cell volume change: "
                    f"{full['volume_change_from_initial_percent']:.16g} %"
                ),
                (
                    "Atomic-only maximum / RMS displacement: "
                    f"{atomic['maximum_displacement_A']:.16g} / "
                    f"{atomic['rms_displacement_A']:.16g} angstrom"
                ),
                (
                    "Full-cell maximum / RMS internal displacement: "
                    f"{full['maximum_internal_displacement_A']:.16g} / "
                    f"{full['rms_internal_displacement_A']:.16g} angstrom"
                ),
                (
                    "Atomic-only optimizer steps and status: "
                    f"{atomic['optimizer_steps']}; "
                    f"{atomic['overall_convergence_status']}"
                ),
                (
                    "Full-cell optimizer steps and status: "
                    f"{full['optimizer_steps']}; "
                    f"{full['overall_convergence_status']}"
                ),
                (
                    "Symmetry initial / atomic-only / full-cell: "
                    f"{symmetry['initial']['space_group_symbol']} "
                    f"({symmetry['initial']['space_group_number']}) / "
                    f"{symmetry['atomic_only']['space_group_symbol']} "
                    f"({symmetry['atomic_only']['space_group_number']}) / "
                    f"{symmetry['full_cell']['space_group_symbol']} "
                    f"({symmetry['full_cell']['space_group_number']})"
                ),
                (
                    "Symmetry preserved atomic-only / full-cell: "
                    f"{str(symmetry['atomic_only_preserved'])} / "
                    f"{str(symmetry['full_cell_preserved'])}"
                ),
                f"Atomic-only safety status: {atomic['safety_status']}",
                f"Full-cell safety status: {full['safety_status']}",
            ]
        )
    lines.extend(
        [
            "",
            "Nonconverged diagnostic results",
            "-------------------------------",
        ]
    )
    for mode in MODES:
        values = data.diagnostic_phases[mode]
        lines.append(
            f"{mode}: {', '.join(values) if values else 'none'}"
        )
    lines.extend(
        [
            "",
            "Scientific limitations",
            "----------------------",
            (
                "These are MACE-potential relaxation results. They do not "
                "establish accuracy against DFT or experiment."
            ),
            (
                "Raw total energies and raw energies per atom were not used "
                "to rank phase stability across different compositions."
            ),
            "No formation energy was calculated and Step 7 was not implemented.",
            "",
            "Output inventory",
            "----------------",
            *(
                f"- {_path_for_console(path, plan.config.project_root)}"
                for path in plan.outputs.targets
            ),
            "",
            "Final Step 6E status",
            "--------------------",
            f"Analysis status: {data.analysis_status}",
            "MACE loaded: No",
            "Optimizer created: No",
            "Relaxation executed: No",
            "Formation energy calculated: No",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_file(path: Path, content: str) -> None:
    """Write, flush, and fsync one staged UTF-8 text artifact."""

    if not content.strip():
        raise PublicationError(f"Refusing to stage an empty file: {path}")
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PublicationError(
            f"Could not stage {path.name}: {type(exc).__name__}: {exc}"
        ) from exc


def _lazy_pyplot() -> tuple[Any, Any]:
    """Load Matplotlib headlessly without importing any calculator package."""

    assert_mace_not_imported()
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise DependencyError(f"Matplotlib plotting is unavailable: {exc}") from exc
    assert_mace_not_imported()
    return plt, np


def _phase_color(index: int) -> str:
    """Return a deterministic color for one configured phase."""

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    return colors[index % len(colors)]


def _save_figure(fig: Any, path: Path, plt: Any) -> None:
    """Save one staged PNG and always release Matplotlib state."""

    try:
        fig.tight_layout()
        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
            metadata={
                "Title": "Ni-Al Step 6 relaxation analysis",
                "Software": "matplotlib",
            },
        )
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
        raise PublicationError(f"Figure was not created correctly: {path.name}")
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise PublicationError(f"Figure is not a valid PNG: {path.name}")


def _plot_history(
    *,
    plan: AnalysisPlan,
    data: AnalysisData,
    mode: str,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    threshold: float | None = None,
    transform: Any | None = None,
) -> None:
    """Plot one deterministic five-phase convergence history."""

    plt, _np = _lazy_pyplot()
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for index, phase in enumerate(plan.config.phase_order):
        history = data.histories[(mode, phase)]
        steps = [row["step"] for row in history.rows]
        values = [float(row[metric]) for row in history.rows]
        if transform is not None:
            values = transform(values)
        result = next(
            row
            for row in data.rows
            if row["phase_key"] == phase and row["stage"] == mode
        )
        suffix = (
            ""
            if not result["diagnostic_only"]
            else " [NOT_CONVERGED]"
        )
        axis.plot(
            steps,
            values,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            color=_phase_color(index),
            label=phase + suffix,
        )
    if threshold is not None:
        axis.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"threshold = {threshold:.7g}",
        )
    axis.set_xlabel("FIRE optimizer step")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path, plt)


def _plot_grouped_bars(
    *,
    plan: AnalysisPlan,
    data: AnalysisData,
    field: str,
    ylabel: str,
    title: str,
    path: Path,
    modes: tuple[str, ...],
    threshold: float | None = None,
) -> None:
    """Plot deterministic grouped per-phase values with diagnostic hatching."""

    plt, np = _lazy_pyplot()
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(plan.config.phase_order), dtype=float)
    width = 0.36 if len(modes) == 2 else 0.56
    mode_colors = {"atomic_only": "#4c78a8", "full_cell": "#f58518"}
    for mode_index, mode in enumerate(modes):
        rows = [
            next(
                row
                for row in data.rows
                if row["phase_key"] == phase and row["stage"] == mode
            )
            for phase in plan.config.phase_order
        ]
        positions = (
            x + (mode_index - (len(modes) - 1) / 2.0) * width
        )
        bars = axis.bar(
            positions,
            [float(row[field]) for row in rows],
            width=width,
            label=mode.replace("_", " "),
            color=mode_colors[mode],
        )
        for bar, row in zip(bars, rows):
            if row["diagnostic_only"]:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
    if threshold is not None:
        axis.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"threshold = {threshold:.7g}",
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(x, plan.config.phase_order)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path, plt)


def _create_figures(
    staging_paths: Mapping[Path, Path],
    plan: AnalysisPlan,
    data: AnalysisData,
) -> None:
    """Generate and validate all nine required figures."""

    figure_stage = {
        name: staging_paths[plan.outputs.figures[name]]
        for name in FIGURE_FILENAMES
    }

    def relative_energy(values: Sequence[float]) -> list[float]:
        first = values[0]
        return [value - first for value in values]

    _plot_history(
        plan=plan,
        data=data,
        mode="atomic_only",
        metric="energy_per_atom_eV",
        ylabel="Energy change from step 0 (eV/atom)",
        title=(
            "Atomic-only MACE energy convergence\n"
            "Within-phase changes only; not a phase-stability ranking"
        ),
        path=figure_stage["atomic_only_energy_convergence.png"],
        transform=relative_energy,
    )
    _plot_history(
        plan=plan,
        data=data,
        mode="atomic_only",
        metric="maximum_force_eV_per_A",
        ylabel="Maximum atomic force (eV/angstrom)",
        title="Atomic-only maximum-force convergence",
        path=figure_stage["atomic_only_force_convergence.png"],
        threshold=plan.config.force_threshold_eV_per_A,
    )
    _plot_history(
        plan=plan,
        data=data,
        mode="full_cell",
        metric="energy_per_atom_eV",
        ylabel="Energy change from step 0 (eV/atom)",
        title=(
            "Full-cell MACE energy convergence\n"
            "Within-phase changes only; not a phase-stability ranking"
        ),
        path=figure_stage["full_cell_energy_convergence.png"],
        transform=relative_energy,
    )
    _plot_history(
        plan=plan,
        data=data,
        mode="full_cell",
        metric="maximum_force_eV_per_A",
        ylabel="Maximum atomic force (eV/angstrom)",
        title="Full-cell maximum-force convergence",
        path=figure_stage["full_cell_force_convergence.png"],
        threshold=plan.config.force_threshold_eV_per_A,
    )
    _plot_history(
        plan=plan,
        data=data,
        mode="full_cell",
        metric="maximum_absolute_stress_eV_per_A3",
        ylabel="Maximum absolute ASE stress (eV/angstrom^3)",
        title="Full-cell stress convergence (six Voigt components)",
        path=figure_stage["full_cell_stress_convergence.png"],
        threshold=plan.config.stress_threshold_eV_per_A3,
    )

    def relative_volume(values: Sequence[float]) -> list[float]:
        first = values[0]
        return [100.0 * (value - first) / first for value in values]

    _plot_history(
        plan=plan,
        data=data,
        mode="full_cell",
        metric="volume_A3",
        ylabel="Volume change from step 0 (%)",
        title="Full-cell volume convergence",
        path=figure_stage["full_cell_volume_convergence.png"],
        transform=relative_volume,
    )
    _plot_grouped_bars(
        plan=plan,
        data=data,
        field="energy_change_from_initial_eV_per_atom",
        ylabel="Final energy change (eV/atom)",
        title=(
            "MACE relaxation energy changes by phase\n"
            "Not a cross-composition phase-stability ranking"
        ),
        path=figure_stage["final_energy_change_by_phase.png"],
        modes=("atomic_only", "full_cell"),
    )
    _plot_grouped_bars(
        plan=plan,
        data=data,
        field="maximum_force_eV_per_A",
        ylabel="Final maximum atomic force (eV/angstrom)",
        title="Final maximum force by phase and relaxation mode",
        path=figure_stage["final_max_force_by_phase.png"],
        modes=("atomic_only", "full_cell"),
        threshold=plan.config.force_threshold_eV_per_A,
    )
    _plot_grouped_bars(
        plan=plan,
        data=data,
        field="volume_change_from_initial_percent",
        ylabel="Full-cell volume change (%)",
        title="Full-cell final volume change by phase",
        path=figure_stage["full_cell_volume_change_percent.png"],
        modes=("full_cell",),
    )


def _prepare_staging_paths(
    staging_root: Path,
    plan: OutputPlan,
) -> Mapping[Path, Path]:
    """Map final targets to unique flat staging paths."""

    return {
        target: staging_root / f"{index:02d}_{target.name}"
        for index, target in enumerate(plan.targets)
    }


def _validate_staged_bundle(
    paths: Mapping[Path, Path],
    plan: AnalysisPlan,
) -> None:
    """Reopen every staged output before publication."""

    for target, staged in paths.items():
        if not staged.is_file() or staged.stat().st_size <= 0:
            raise PublicationError(f"Staged output is missing or empty: {target.name}")
    read_json_object(paths[plan.outputs.json_table], "staged comparison JSON")
    try:
        with paths[plan.outputs.csv_table].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise PublicationError("Staged comparison CSV header is invalid.")
            rows = list(reader)
            if len(rows) != len(plan.config.phase_order) * 3:
                raise PublicationError(
                    "Staged comparison CSV must contain exactly 15 data rows."
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PublicationError(f"Could not validate staged CSV: {exc}") from exc
    report = paths[plan.outputs.text_report].read_text(encoding="utf-8")
    if "Analysis status:" not in report or "Formation energy calculated: No" not in report:
        raise PublicationError("Staged comparison report is incomplete.")
    for name in FIGURE_FILENAMES:
        figure = paths[plan.outputs.figures[name]]
        with figure.open("rb") as handle:
            if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                raise PublicationError(f"Staged figure is not a PNG: {name}")


def _publish_staged_bundle(
    staging: Mapping[Path, Path],
    plan: AnalysisPlan,
    overwrite: bool,
) -> None:
    """Publish all twelve outputs with verified rollback on failure."""

    config = plan.config
    _validate_collision_policy(plan.outputs, config, overwrite)
    expected = {
        target: (
            staged.stat().st_size,
            _file_sha256(staged),
        )
        for target, staged in staging.items()
    }
    staging_root = next(iter(staging.values())).parent
    backups_root = staging_root / "backups"
    backups: dict[Path, Path] = {}
    if overwrite:
        for index, target in enumerate(plan.outputs.targets):
            if not os.path.lexists(target):
                continue
            if target.is_symlink() or not target.is_file():
                raise OutputCollisionError(
                    f"Refusing unsafe overwrite target: {target}"
                )
            backups_root.mkdir(parents=True, exist_ok=True)
            backup = backups_root / f"{index:02d}_{target.name}"
            try:
                os.link(target, backup)
            except OSError:
                shutil.copy2(target, backup)
            if (
                not backup.is_file()
                or backup.stat().st_size != target.stat().st_size
                or _file_sha256(backup) != _file_sha256(target)
            ):
                raise PublicationError(f"Could not verify backup of {target}.")
            backups[target] = backup

    published: list[Path] = []
    try:
        if overwrite:
            for target in plan.outputs.targets:
                os.replace(staging[target], target)
                published.append(target)
        else:
            for target in plan.outputs.targets:
                os.link(staging[target], target)
                published.append(target)
        for target in plan.outputs.targets:
            size, sha256 = expected[target]
            if (
                not target.is_file()
                or target.stat().st_size != size
                or _file_sha256(target) != sha256
            ):
                raise PublicationError(
                    f"Published output verification failed: {target}"
                )
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
                    f"{target}: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        if rollback_errors:
            raise PublicationError(
                "Step 6E publication failed and rollback was incomplete. "
                f"Retain and inspect {staging_root}. Original error: {exc}; "
                "rollback errors: " + "; ".join(rollback_errors)
            ) from exc
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise PublicationError(
            "Step 6E publication failed; the complete prior output state was "
            f"restored ({type(exc).__name__}: {exc})."
        ) from exc


def publish_analysis(
    plan: AnalysisPlan,
    data: AnalysisData,
    overwrite: bool,
) -> None:
    """Stage, validate, and publish the complete Step 6E bundle."""

    _validate_collision_policy(plan.outputs, plan.config, overwrite)
    for target in plan.outputs.targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    staging_root: Path | None = None
    keep_for_recovery = False
    try:
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=".step6e_publication_",
                dir=plan.config.comparison_directory,
            )
        )
        staged = _prepare_staging_paths(staging_root, plan.outputs)
        _write_text_file(
            staged[plan.outputs.csv_table],
            _csv_text(data.rows),
        )
        _write_text_file(
            staged[plan.outputs.json_table],
            _json_text(plan, data),
        )
        _write_text_file(
            staged[plan.outputs.text_report],
            _report_text(plan, data),
        )
        _create_figures(staged, plan, data)
        _validate_staged_bundle(staged, plan)
        try:
            _publish_staged_bundle(staged, plan, overwrite)
        except PublicationError as exc:
            if "rollback was incomplete" in str(exc):
                keep_for_recovery = True
            raise
    finally:
        if (
            staging_root is not None
            and staging_root.exists()
            and not keep_for_recovery
        ):
            try:
                shutil.rmtree(staging_root)
            except OSError as exc:
                LOGGER.warning(
                    "Could not remove Step 6E staging directory %s: %s",
                    staging_root,
                    exc,
                )
    assert_mace_not_imported()


def validate_analysis_inputs(
    config_path: Path | str,
    *,
    require_inputs: bool = True,
    overwrite: bool = False,
) -> AnalysisPlan:
    """Stable orchestrator API for static or complete analysis validation."""

    return validate_plan(
        config_path,
        require_inputs=require_inputs,
        overwrite=overwrite,
    )


def validate_existing_analysis_outputs(
    config_path: Path | str,
) -> AnalysisPlan:
    """Validate an already published complete Step 6E bundle for resume."""

    plan = validate_analysis_inputs(
        config_path,
        require_inputs=True,
        overwrite=False,
    )
    existing = set(_find_collisions(plan.outputs))
    expected = set(plan.outputs.targets)
    if existing != expected:
        missing = sorted(expected.difference(existing))
        unexpected = sorted(existing.difference(expected))
        details = [
            *(f"missing: {_path_for_console(path, plan.config.project_root)}" for path in missing),
            *(
                f"unexpected: {_path_for_console(path, plan.config.project_root)}"
                for path in unexpected
            ),
        ]
        raise OutputCollisionError(
            "Existing Step 6E bundle is incomplete: " + "; ".join(details)
        )
    _validate_staged_bundle(
        {target: target for target in plan.outputs.targets},
        plan,
    )
    document = read_json_object(
        plan.outputs.json_table,
        "existing comparison JSON",
    )
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("project_step") != "6E"
        or document.get("record_type")
        != "ni_al_mace_relaxation_comparison"
    ):
        raise PublicationError(
            "Existing Step 6E comparison JSON identity is invalid."
        )
    assert_mace_not_imported()
    return plan


def run_analysis(
    config_path: Path | str,
    *,
    overwrite: bool = False,
) -> AnalysisData:
    """Validate inputs, derive comparisons, and publish the Step 6E bundle."""

    plan = validate_analysis_inputs(
        config_path,
        require_inputs=True,
        overwrite=overwrite,
    )
    _validate_collision_policy(plan.outputs, plan.config, overwrite)
    data = derive_analysis(plan)
    publish_analysis(plan, data, overwrite)
    return data


def _print_validation_summary(plan: AnalysisPlan) -> None:
    """Print the standalone validation-only result."""

    LOGGER.info("Step 6E validation-only summary")
    LOGGER.info("Configuration: %s", _path_for_console(
        plan.config.config_path, plan.config.project_root
    ))
    LOGGER.info("Input phases validated: %d", len(plan.phases))
    LOGGER.info("Atomic-only result manifests: %d", len(plan.phases))
    LOGGER.info("Full-cell result manifests: %d", len(plan.phases))
    LOGGER.info("Planned Step 6E outputs: %d", len(plan.outputs.targets))
    LOGGER.info("Existing target collisions: %d", len(plan.collisions))
    for path in plan.collisions:
        LOGGER.info("  collision: %s", _path_for_console(path, plan.config.project_root))
    LOGGER.info("MACE loaded: No")
    LOGGER.info("Optimizer imported or created: No")
    LOGGER.info("Relaxation executed: No")
    LOGGER.info("Analysis outputs written: No")
    LOGGER.info("Formation energy calculated: No")
    LOGGER.info("Validation status: SUCCESS")


def run(arguments: Sequence[str] | None = None) -> int:
    """Execute the requested validation-only or analysis workflow."""

    options = parse_arguments(arguments)
    configure_logging(options.verbose)
    assert_mace_not_imported()
    try:
        plan = validate_plan(
            options.config,
            require_inputs=True,
            overwrite=options.overwrite,
        )
        if options.validate_only:
            _print_validation_summary(plan)
            return 0
        data = run_analysis(options.config, overwrite=options.overwrite)
        LOGGER.info("Step 6E analysis status: %s", data.analysis_status)
        LOGGER.info("Published outputs: %d", len(plan.outputs.targets))
        for target in plan.outputs.targets:
            LOGGER.info(
                "  %s", _path_for_console(target, plan.config.project_root)
            )
        LOGGER.info("MACE loaded: No")
        LOGGER.info("Optimizer created: No")
        LOGGER.info("Relaxation executed: No")
        LOGGER.info("Formation energy calculated: No")
        assert_mace_not_imported()
        return 0
    except Step6AnalysisError as exc:
        LOGGER.error("%s", exc)
        LOGGER.info("MACE loaded: No")
        LOGGER.info("Optimizer created: No")
        LOGGER.info("Relaxation executed: No")
        LOGGER.info("Formation energy calculated: No")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point, also callable by tests."""

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())

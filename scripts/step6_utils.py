"""Shared, safety-focused execution utilities for Ni-Al MACE relaxations.

This module implements the common Step 6C (fixed-cell atomic relaxation) and
Step 6D (coupled atom-and-cell relaxation) workflows.  Calculator, optimizer,
and cell-filter imports are deliberately lazy: importing this module or
building a validation context cannot load MACE or create an optimizer.

The public entry points are designed for the thin command-line runners and the
Step 6 pipeline orchestrator:

``locate_project_root``
    Resolve repository paths independently of the terminal working directory.
``validate_mode_cli_plan``
    Enforce safe combinations of validation and execution flags.
``load_and_validate_context``
    Strictly validate configuration, selected sources, metadata, Step 5
    baselines, and Step 6B reproducibility evidence.
``load_calculator_session``
    Lazily construct one reusable MACE calculator.
``execute_mode``
    Execute independent phase relaxations and transactionally publish results.
``validate_phase_bundle`` / ``validate_all_mode_outputs``
    Verify complete output bundles, including resume manifests and hashes.

No function in this module calculates formation energies or compares raw
energies across compositions as a phase-stability ranking.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import logging
import math
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:
    import validate_ni_al_mace_relaxation as _step6a
    import reproduce_ni_al_mace_baseline as _step6b
except ModuleNotFoundError:  # Support ``import scripts.step6_utils`` in tests.
    from scripts import validate_ni_al_mace_relaxation as _step6a
    from scripts import reproduce_ni_al_mace_baseline as _step6b


LOGGER = logging.getLogger("step6_utils")

SCHEMA_VERSION = "1.0"
PHASE_ORDER: tuple[str, ...] = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
ATOMIC_ONLY_MODE = "atomic_only"
FULL_CELL_MODE = "full_cell"
VALID_MODES = (ATOMIC_ONLY_MODE, FULL_CELL_MODE)
ATOMIC_ONLY_PILOT_PHASE = "Al3Ni"
FULL_CELL_PILOT_PHASE = "AlNi"
STRESS_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
ATOMIC_CELL_ATOL_A = 1.0e-12
GENERALIZED_FORCE_AUTO_STOP_FMAX = 0.0

EXPECTED_MATERIAL_IDS: Mapping[str, str] = {
    "Al3Ni": "mp-622209",
    "Al3Ni2": "mp-1057",
    "AlNi": "mp-1487",
    "Al3Ni5": "mp-16514",
    "AlNi3": "mp-2593",
}
EXPECTED_ATOM_COUNTS: Mapping[str, int] = {
    "Al3Ni": 16,
    "Al3Ni2": 5,
    "AlNi": 2,
    "Al3Ni5": 8,
    "AlNi3": 4,
}


class Step6Error(RuntimeError):
    """Base class for controlled Step 6 failures."""


class Step6ConfigurationError(Step6Error):
    """Raised when configuration or command scope is unsafe."""


class Step6InputError(Step6Error):
    """Raised when a protected scientific input is invalid."""


class Step6ReproducibilityError(Step6InputError):
    """Raised when required Step 6B evidence is absent or unsuccessful."""


class Step6DependencyError(Step6Error):
    """Raised when a required installed public API is unavailable."""


class Step6CalculatorError(Step6Error):
    """Raised when MACE calculator creation or reuse fails."""


class Step6CalculationError(Step6Error):
    """Raised when an energy, force, stress, or optimizer call fails."""


class Step6SafetyError(Step6CalculationError):
    """Raised when a configured relaxation safety invariant is violated."""


class Step6CollisionError(Step6Error):
    """Raised when protected output collision handling refuses publication."""


class Step6PublicationError(Step6Error):
    """Raised when transactional publication or rollback fails."""


class Step6ResumeError(Step6Error):
    """Raised when an existing phase bundle is not safe to resume."""


class DuplicateJsonKeyError(ValueError):
    """Internal signal used to reject duplicate JSON keys."""


@dataclass(frozen=True)
class FileSnapshot:
    """Content and filesystem metadata fingerprint for a protected file."""

    label: str
    path: Path
    sha256: str
    size: int
    modification_time_ns: int

    def to_json(self, project_root: Path) -> dict[str, Any]:
        """Return a portable JSON record."""

        return {
            "label": self.label,
            "path": relative_path(self.path, project_root),
            "sha256": self.sha256,
            "size_bytes": self.size,
            "modification_time_ns": self.modification_time_ns,
        }


@dataclass(frozen=True)
class ExecutionSettings:
    """Validated operational settings that do not alter scientific criteria."""

    atomic_only_pilot_phase: str = ATOMIC_ONLY_PILOT_PHASE
    full_cell_pilot_phase: str = FULL_CELL_PILOT_PHASE
    symmetry_symprec_A: float = 0.001
    symmetry_angle_tolerance_deg: float = 5.0
    history_interval: int = 1
    external_pressure_eV_per_A3: float = 0.0


@dataclass(frozen=True)
class PhaseInput:
    """Strictly validated source, metadata, and baseline for one phase."""

    phase_key: str
    material_id: str
    atom_count: int
    structure: Any
    baseline: Any
    step6b_snapshots: tuple[FileSnapshot, ...]


@dataclass(frozen=True)
class ModeOutputPaths:
    """Canonical Step 6C or Step 6D output targets for one phase."""

    structure: Path
    trajectory: Path
    history_csv: Path
    report: Path
    log: Path
    result_json: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every target in deterministic publication order."""

        return (
            self.structure,
            self.trajectory,
            self.history_csv,
            self.report,
            self.log,
            self.result_json,
        )


@dataclass(frozen=True)
class CombinedOutputPaths:
    """Canonical combined mode summary paths."""

    csv: Path
    json: Path
    report: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every summary target."""

        return (self.csv, self.json, self.report)


@dataclass(frozen=True)
class Step6Context:
    """Fully validated, calculation-free context for one relaxation mode."""

    project_root: Path
    config_path: Path
    config_snapshot: FileSnapshot
    config_fingerprint: str
    raw_config: Mapping[str, Any]
    configuration: Any
    mode: str
    mode_settings: Any
    execution: ExecutionSettings
    phase_keys: tuple[str, ...]
    phase_inputs: Mapping[str, PhaseInput]
    protected_snapshots: tuple[FileSnapshot, ...]
    output_root: Path

    @property
    def pilot_phase(self) -> str:
        """Return the configured pilot phase for this mode."""

        if self.mode == ATOMIC_ONLY_MODE:
            return self.execution.atomic_only_pilot_phase
        return self.execution.full_cell_pilot_phase


@dataclass
class CalculatorSession:
    """One reusable calculator and auditable counters for a batch."""

    calculator: Any
    calculator_class: str
    configuration_fingerprint: str
    load_count: int = 1
    state_evaluations: int = 0
    optimizer_steps: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DisplacementMetrics:
    """Periodic atomic displacement metrics relative to the original input."""

    internal_vectors_A: tuple[tuple[float, float, float], ...]
    internal_magnitudes_A: tuple[float, ...]
    maximum_internal_A: float
    rms_internal_A: float
    mean_internal_A: float
    total_vectors_A: tuple[tuple[float, float, float], ...]
    total_magnitudes_A: tuple[float, ...]
    maximum_total_A: float
    rms_total_A: float
    mean_total_A: float
    per_species: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class StateMetrics:
    """Complete finite state measured from the underlying ASE Atoms."""

    step: int
    elapsed_seconds: float
    total_energy_eV: float
    energy_per_atom_eV: float
    forces_eV_per_A: tuple[tuple[float, float, float], ...]
    force_magnitudes_eV_per_A: tuple[float, ...]
    maximum_force_eV_per_A: float
    mean_force_eV_per_A: float
    rms_force_eV_per_A: float
    total_force_eV_per_A: tuple[float, float, float]
    total_force_norm_eV_per_A: float
    stress_eV_per_A3: tuple[float, float, float, float, float, float]
    maximum_absolute_stress_eV_per_A3: float
    volume_A3: float
    volume_per_atom_A3: float
    cell_A: tuple[tuple[float, float, float], ...]
    lattice_lengths_A: tuple[float, float, float]
    lattice_angles_deg: tuple[float, float, float]
    positions_A: tuple[tuple[float, float, float], ...]
    scaled_positions: tuple[tuple[float, float, float], ...]
    displacement: DisplacementMetrics
    volume_change_percent: float
    force_converged: bool
    stress_converged: bool | None
    overall_converged: bool


@dataclass(frozen=True)
class InitialGeometry:
    """Independent arrays used for identity, safety, and displacement checks."""

    positions_A: Any
    scaled_positions: Any
    cell_A: Any
    symbols: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    pbc: tuple[bool, bool, bool]
    atom_count: int
    volume_A3: float


@dataclass(frozen=True)
class PhaseResult:
    """Validated reportable result for one phase and one relaxation mode."""

    phase_key: str
    material_id: str
    mode: str
    status: str
    safety_status: str
    optimizer_created: bool
    optimizer_steps: int
    state_evaluations: int
    calculator_class: str
    calculator_load_count: int
    started_at_utc: str
    completed_at_utc: str
    wall_time_seconds: float
    initial: StateMetrics
    final: StateMetrics
    history: tuple[StateMetrics, ...]
    warnings: tuple[str, ...]
    output_paths: ModeOutputPaths
    source_snapshots: tuple[FileSnapshot, ...]
    configuration_fingerprint: str
    resumed: bool = False

    @property
    def force_converged(self) -> bool:
        """Return final raw atomic-force convergence."""

        return self.final.force_converged

    @property
    def stress_converged(self) -> bool | None:
        """Return final raw stress convergence, or None for atomic-only."""

        return self.final.stress_converged

    @property
    def overall_converged(self) -> bool:
        """Return final scientific convergence."""

        return self.final.overall_converged


@dataclass(frozen=True)
class ModeSummary:
    """Combined result returned to runners and the pipeline."""

    mode: str
    requested_phases: tuple[str, ...]
    results: tuple[PhaseResult, ...]
    calculator_loads: int
    calculator_class: str
    state_evaluations: int
    optimizer_steps: int
    resumed_phases: tuple[str, ...]
    executed_phases: tuple[str, ...]
    overall_status: str
    combined_outputs: CombinedOutputPaths | None


def locate_project_root() -> Path:
    """Locate the repository from this module, never from the shell CWD."""

    return Path(__file__).resolve().parents[1]


def relative_path(path: Path, project_root: Path) -> str:
    """Return a stable POSIX-style repository-relative path when possible."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def utc_timestamp() -> str:
    """Return a second-resolution UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is forbidden: {value}")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read strict UTF-8 JSON while rejecting duplicates and nonfinite values."""

    if not path.is_file():
        raise Step6InputError(f"{label} does not exist: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Step6InputError(
            f"Could not read strict {label} from {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise Step6InputError(f"{label} must have a JSON object at its root.")
    return parsed


def file_sha256(path: Path) -> str:
    """Compute a SHA-256 digest without loading the complete file into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Step6InputError(f"Could not hash protected file {path}: {exc}") from exc
    return digest.hexdigest()


def capture_file_snapshot(path: Path, label: str) -> FileSnapshot:
    """Capture immutable content, size, and timestamp evidence."""

    if not path.is_file():
        raise Step6InputError(f"Protected {label} is missing: {path}")
    try:
        stat = path.stat()
    except OSError as exc:
        raise Step6InputError(f"Could not stat protected {label}: {exc}") from exc
    return FileSnapshot(
        label=label,
        path=path.resolve(),
        sha256=file_sha256(path),
        size=stat.st_size,
        modification_time_ns=stat.st_mtime_ns,
    )


def verify_file_snapshot(snapshot: FileSnapshot) -> None:
    """Raise when any protected file attribute differs from its snapshot."""

    current = capture_file_snapshot(snapshot.path, snapshot.label)
    differences: list[str] = []
    if current.sha256 != snapshot.sha256:
        differences.append(
            f"sha256 {snapshot.sha256} -> {current.sha256}"
        )
    if current.size != snapshot.size:
        differences.append(f"size {snapshot.size} -> {current.size}")
    if current.modification_time_ns != snapshot.modification_time_ns:
        differences.append(
            "mtime_ns "
            f"{snapshot.modification_time_ns} -> {current.modification_time_ns}"
        )
    if differences:
        raise Step6SafetyError(
            f"Protected file changed during calculation ({snapshot.label}): "
            + "; ".join(differences)
        )


def verify_protected_files(context: Step6Context) -> None:
    """Verify every source, baseline, configuration, and Step 6B snapshot."""

    for snapshot in context.protected_snapshots:
        verify_file_snapshot(snapshot)


def validate_mode_cli_plan(
    *,
    validate_only: bool,
    execute: bool,
    create_directories: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    """Validate mutually exclusive runner flags before any side effect."""

    if validate_only == execute:
        raise Step6ConfigurationError(
            "Select exactly one of --validate-only and --execute."
        )
    if overwrite and not execute:
        raise Step6ConfigurationError("--overwrite is allowed only with --execute.")
    if resume and not execute:
        raise Step6ConfigurationError("--resume is allowed only with --execute.")
    if resume and overwrite:
        raise Step6ConfigurationError(
            "--resume and --overwrite are mutually exclusive safety modes."
        )
    if create_directories and not validate_only:
        raise Step6ConfigurationError(
            "--create-directories is allowed only with --validate-only."
        )


def installed_scientific_versions() -> Mapping[str, str]:
    """Return audited package versions without constructing scientific objects."""

    versions: dict[str, str] = {}
    for distribution in (
        "ase",
        "mace-torch",
        "numpy",
        "pymatgen",
        "matplotlib",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Step6DependencyError(
                f"Required distribution is not installed: {distribution}"
            ) from exc
    return versions


def validate_installed_relaxation_api(mode: str) -> Mapping[str, str]:
    """Validate required installed public APIs without importing FIRE.

    Full-cell validation imports the filter class only to inspect its public
    constructor. It does not instantiate the filter. FIRE remains unimported
    until a real non-converged execution needs an optimizer.
    """

    versions = dict(installed_scientific_versions())
    ase_spec = importlib.util.find_spec("ase")
    ase_locations = (
        tuple(ase_spec.submodule_search_locations)
        if ase_spec is not None and ase_spec.submodule_search_locations is not None
        else ()
    )
    fire_available = any(
        (Path(location) / "optimize" / "fire.py").is_file()
        for location in ase_locations
    )
    if not fire_available:
        raise Step6DependencyError(
            f"ase.optimize.FIRE is unavailable in ASE {versions['ase']}."
        )
    if mode == FULL_CELL_MODE:
        try:
            from ase.filters import FrechetCellFilter
        except ImportError as exc:
            raise Step6DependencyError(
                "ase.filters.FrechetCellFilter is unavailable in installed "
                f"ASE {versions['ase']}: {exc}"
            ) from exc
        parameters = inspect.signature(FrechetCellFilter).parameters
        required = {
            "atoms",
            "mask",
            "exp_cell_factor",
            "hydrostatic_strain",
            "constant_volume",
            "scalar_pressure",
        }
        if not required.issubset(parameters):
            raise Step6DependencyError(
                "Installed FrechetCellFilter public signature is incompatible; "
                f"received parameters {tuple(parameters)}."
            )
    return versions


def _resolve_config_path(project_root: Path, config_path: Path) -> Path:
    candidate = config_path
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step6ConfigurationError(
            f"Configuration must remain inside the repository: {resolved}"
        ) from exc
    return resolved


def _validate_phase_keys(phase_keys: Sequence[str] | None) -> tuple[str, ...]:
    if phase_keys is None:
        return PHASE_ORDER
    selected = tuple(phase_keys)
    if not selected:
        raise Step6ConfigurationError("At least one phase must be selected.")
    if len(set(selected)) != len(selected):
        raise Step6ConfigurationError("Requested phase keys contain duplicates.")
    unknown = [phase for phase in selected if phase not in PHASE_ORDER]
    if unknown:
        raise Step6ConfigurationError(
            "Unknown phase key(s): " + ", ".join(unknown)
        )
    return tuple(phase for phase in PHASE_ORDER if phase in selected)


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step6ConfigurationError(f"{label} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise Step6ConfigurationError(f"{label} must be a finite positive number.")
    return number


def _validate_execution_settings(raw_config: Mapping[str, Any]) -> ExecutionSettings:
    raw = raw_config.get("step6_execution", {})
    if not isinstance(raw, Mapping):
        raise Step6ConfigurationError("step6_execution must be a JSON object.")

    atomic_pilot = raw.get("atomic_only_pilot_phase", ATOMIC_ONLY_PILOT_PHASE)
    full_pilot = raw.get("full_cell_pilot_phase", FULL_CELL_PILOT_PHASE)
    if atomic_pilot != ATOMIC_ONLY_PILOT_PHASE:
        raise Step6ConfigurationError(
            f"atomic_only_pilot_phase must be {ATOMIC_ONLY_PILOT_PHASE!r}."
        )
    if full_pilot != FULL_CELL_PILOT_PHASE:
        raise Step6ConfigurationError(
            f"full_cell_pilot_phase must be {FULL_CELL_PILOT_PHASE!r}."
        )
    symprec = _positive_finite(
        raw.get("symmetry_symprec_A", 0.001), "symmetry_symprec_A"
    )
    angle = _positive_finite(
        raw.get("symmetry_angle_tolerance_deg", 5.0),
        "symmetry_angle_tolerance_deg",
    )
    interval = raw.get("history_interval", 1)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise Step6ConfigurationError("history_interval must be a positive integer.")
    pressure = raw.get("external_pressure_eV_per_A3", 0.0)
    if isinstance(pressure, bool) or not isinstance(pressure, (int, float)):
        raise Step6ConfigurationError(
            "external_pressure_eV_per_A3 must be finite numeric zero."
        )
    pressure_value = float(pressure)
    if not math.isfinite(pressure_value) or pressure_value != 0.0:
        raise Step6ConfigurationError(
            "Step 6 requires external_pressure_eV_per_A3 to be exactly 0.0."
        )
    return ExecutionSettings(
        atomic_only_pilot_phase=atomic_pilot,
        full_cell_pilot_phase=full_pilot,
        symmetry_symprec_A=symprec,
        symmetry_angle_tolerance_deg=angle,
        history_interval=interval,
        external_pressure_eV_per_A3=pressure_value,
    )


def _mode_output_root(configuration: Any, mode: str) -> Path:
    if mode == ATOMIC_ONLY_MODE:
        return Path(configuration.atomic_only_output_directory)
    return Path(configuration.full_cell_output_directory)


def _validate_fixed_scientific_controls(configuration: Any) -> None:
    """Reject any silent change to the user-mandated Step 6 criteria."""

    atomic = configuration.atomic_only
    full = configuration.full_cell
    safety = configuration.safety
    errors: list[str] = []
    exact_mode_values = (
        (atomic.optimizer, "FIRE", "atomic_only.optimizer"),
        (
            atomic.force_threshold_eV_per_A,
            0.01,
            "atomic_only.force_threshold_eV_per_A",
        ),
        (atomic.maximum_steps, 500, "atomic_only.maximum_steps"),
        (atomic.trajectory_interval, 1, "atomic_only.trajectory_interval"),
        (full.optimizer, "FIRE", "full_cell.optimizer"),
        (
            full.force_threshold_eV_per_A,
            0.01,
            "full_cell.force_threshold_eV_per_A",
        ),
        (
            full.stress_threshold_eV_per_A3,
            0.0006241509,
            "full_cell.stress_threshold_eV_per_A3",
        ),
        (full.maximum_steps, 1000, "full_cell.maximum_steps"),
        (full.trajectory_interval, 1, "full_cell.trajectory_interval"),
        (full.hydrostatic_strain, False, "full_cell.hydrostatic_strain"),
        (full.constant_volume, False, "full_cell.constant_volume"),
        (
            safety.maximum_absolute_volume_change_percent,
            25.0,
            "safety.maximum_absolute_volume_change_percent",
        ),
        (
            safety.maximum_atomic_displacement_A,
            2.0,
            "safety.maximum_atomic_displacement_A",
        ),
        (safety.stop_on_nonfinite_value, True, "safety.stop_on_nonfinite_value"),
        (
            safety.preserve_original_structure,
            True,
            "safety.preserve_original_structure",
        ),
        (safety.require_periodic_cell, True, "safety.require_periodic_cell"),
    )
    for actual, expected, label in exact_mode_values:
        if actual != expected:
            errors.append(f"{label}: expected {expected!r}, received {actual!r}")
    if errors:
        raise Step6ConfigurationError(
            "Configured scientific controls differ from the mandatory Step 6 "
            "values:\n  - " + "\n  - ".join(errors)
        )


def planned_mode_directories(context: Step6Context) -> tuple[Path, ...]:
    """Return every mode directory that execution may populate."""

    root = context.output_root
    return (
        root,
        root / "structures",
        root / "trajectories",
        root / "tables",
        root / "reports",
        root / "logs",
        root / "checkpoints",
    )


def _snapshot_step6b_evidence(
    project_root: Path, phase_key: str
) -> tuple[FileSnapshot, ...]:
    reports = project_root / "results" / "mace_relaxation" / "comparison" / "reports"
    tables = project_root / "results" / "mace_relaxation" / "comparison" / "tables"
    if phase_key == "AlNi":
        report = reports / "AlNi_step6b2_baseline_reproduction.txt"
        text = report.read_text(encoding="utf-8") if report.is_file() else ""
        if (
            "Overall reproducibility status: PASS" not in text
            or "Phase: AlNi" not in text
            or "Materials Project ID: mp-1487" not in text
        ):
            raise Step6ReproducibilityError(
                "AlNi Step 6B.2 report is missing or does not report PASS."
            )
        return (capture_file_snapshot(report, "AlNi Step 6B.2 report"),)

    individual = reports / f"{phase_key}_step6b3_baseline_reproduction.txt"
    combined_text = reports / "ni_al_step6b3_baseline_reproduction_summary.txt"
    combined_json = tables / "ni_al_step6b3_baseline_reproduction.json"
    individual_text = (
        individual.read_text(encoding="utf-8") if individual.is_file() else ""
    )
    if "Overall reproducibility status: PASS" not in individual_text:
        raise Step6ReproducibilityError(
            f"{phase_key} Step 6B.3 report is missing or does not report PASS."
        )
    document = read_json_object(combined_json, "Step 6B.3 combined JSON")
    remaining = ("Al3Ni", "Al3Ni2", "Al3Ni5", "AlNi3")
    if (
        document.get("schema_version") != "1.0"
        or document.get("project_step") != "6B.3"
        or document.get("overall_status") != "PASS"
        or tuple(document.get("requested_phases", ())) != remaining
        or tuple(document.get("completed_phases", ())) != remaining
        or document.get("failed_phases") != []
        or document.get("calculator_loads") != 1
        or document.get("single_point_calculations") != 4
        or document.get("optimizer_created") is not False
        or document.get("fire_executed") is not False
        or document.get("relaxation_executed") is not False
        or document.get("trajectories_created") is not False
        or document.get("structures_written") is not False
    ):
        raise Step6ReproducibilityError(
            "Step 6B.3 combined JSON scope/status/execution evidence is invalid."
        )
    model = document.get("model")
    if not isinstance(model, Mapping) or any(
        model.get(key) != value
        for key, value in {
            "family": "MACE",
            "name": "MACE-MP-0",
            "size": "small",
            "device": "cpu",
            "dtype": "float64",
            "dispersion_enabled": False,
        }.items()
    ):
        raise Step6ReproducibilityError("Step 6B.3 model evidence is invalid.")
    combined_text_value = (
        combined_text.read_text(encoding="utf-8")
        if combined_text.is_file()
        else ""
    )
    if "Overall Step 6B.3 status: PASS" not in combined_text_value:
        raise Step6ReproducibilityError(
            "Step 6B.3 combined text report is missing its PASS sentinel."
        )
    pilot_path = reports / "AlNi_step6b2_baseline_reproduction.txt"
    pilot_snapshot = capture_file_snapshot(
        pilot_path, "protected AlNi Step 6B.2 report"
    )
    pilot_record = document.get("protected_alni_step6b2_report")
    if not isinstance(pilot_record, Mapping) or (
        pilot_record.get("status"),
        pilot_record.get("sha256"),
        pilot_record.get("size_bytes"),
        pilot_record.get("modification_time_ns"),
    ) != (
        "PASS",
        pilot_snapshot.sha256,
        pilot_snapshot.size,
        pilot_snapshot.modification_time_ns,
    ):
        raise Step6ReproducibilityError(
            "Step 6B.3 protected AlNi Step 6B.2 fingerprint is invalid."
        )
    records = document.get("records")
    if not isinstance(records, list):
        raise Step6ReproducibilityError("Step 6B.3 records must be a JSON array.")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("phase_key") == phase_key
    ]
    if len(matches) != 1 or matches[0].get("phase_reproducibility_status") != "PASS":
        raise Step6ReproducibilityError(
            f"Step 6B.3 has no unique PASS record for {phase_key}."
        )
    return (
        capture_file_snapshot(individual, f"{phase_key} Step 6B.3 report"),
        capture_file_snapshot(combined_text, "Step 6B.3 combined report"),
        capture_file_snapshot(combined_json, "Step 6B.3 combined JSON"),
        pilot_snapshot,
    )


def _collect_global_protected_snapshots(
    project_root: Path,
    config_snapshot: FileSnapshot,
    phase_inputs: Mapping[str, PhaseInput],
) -> tuple[FileSnapshot, ...]:
    snapshots: MutableMapping[Path, FileSnapshot] = {
        config_snapshot.path: config_snapshot
    }
    for phase_input in phase_inputs.values():
        candidates = (
            phase_input.structure.structure_snapshot,
            phase_input.structure.metadata_snapshot,
            phase_input.baseline.table_snapshot,
            phase_input.baseline.annotated_snapshot,
        )
        for candidate in candidates:
            converted = FileSnapshot(
                label=candidate.label,
                path=Path(candidate.path).resolve(),
                sha256=candidate.sha256,
                size=candidate.size,
                modification_time_ns=candidate.modification_time_ns,
            )
            snapshots[converted.path] = converted
        for snapshot in phase_input.step6b_snapshots:
            snapshots[snapshot.path] = snapshot

    # Protect every selected input and every Step 5 artifact, even when a
    # single-phase pilot is selected. A calculation is never authorized to
    # mutate a sibling phase or an earlier result.
    for phase in PHASE_ORDER:
        for path, label in (
            (
                project_root
                / "data"
                / "processed"
                / "ni_al_structures"
                / "selected"
                / f"{phase}.extxyz",
                f"selected {phase} EXTXYZ",
            ),
            (
                project_root
                / "data"
                / "processed"
                / "ni_al_structures"
                / "selected"
                / f"{phase}.metadata.json",
                f"selected {phase} metadata",
            ),
        ):
            resolved = path.resolve()
            if resolved not in snapshots:
                snapshot = capture_file_snapshot(path, label)
                snapshots[snapshot.path] = snapshot
    step5_root = project_root / "results" / "mace_zero_shot"
    if step5_root.is_dir():
        for path in sorted(step5_root.rglob("*")):
            if path.is_file() and path.resolve() not in snapshots:
                snapshot = capture_file_snapshot(
                    path, f"protected Step 5 output {path.name}"
                )
                snapshots[snapshot.path] = snapshot

    # Protect all successful Step 6A/6B evidence and prior environment
    # snapshots, even when a single phase is selected.
    protected_patterns = (
        ("results/mace_relaxation/comparison/reports", "*step6b*.txt"),
        ("results/mace_relaxation/comparison/reports", "*step6b3*.txt"),
        ("results/mace_relaxation/comparison/tables", "*step6b3*.json"),
        ("environment", "requirements_step6a.txt"),
        ("environment", "requirements_step6b*.txt"),
    )
    for directory_name, pattern in protected_patterns:
        directory = project_root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path.resolve() not in snapshots:
                snapshot = capture_file_snapshot(path, f"protected prior output {path.name}")
                snapshots[snapshot.path] = snapshot
    for relative in (
        Path("configs/ni_al_phases.json"),
        Path("configs/mace_zero_shot.json"),
    ):
        path = project_root / relative
        if path.is_file() and path.resolve() not in snapshots:
            snapshot = capture_file_snapshot(path, f"protected {relative.name}")
            snapshots[snapshot.path] = snapshot
    return tuple(sorted(snapshots.values(), key=lambda item: str(item.path)))


def load_and_validate_context(
    config_path: Path | str,
    mode: str,
    phase_keys: Sequence[str] | None,
    create_directories: bool = False,
    require_step6_outputs: bool = False,
) -> Step6Context:
    """Build a strict calculation-free context for a Step 6 mode.

    This function intentionally imports neither MACE, FIRE, nor
    ``FrechetCellFilter``.
    """

    if mode not in VALID_MODES:
        raise Step6ConfigurationError(
            f"mode must be one of {', '.join(VALID_MODES)}; received {mode!r}."
        )
    project_root = locate_project_root()
    selected_phases = _validate_phase_keys(phase_keys)
    resolved_config = _resolve_config_path(project_root, Path(config_path))

    try:
        _step6a.validate_runtime(project_root)
        raw_config = _step6a.read_json_object(
            resolved_config, "MACE relaxation configuration", _step6a.ConfigurationError
        )
        configuration = _step6a.validate_configuration(raw_config, project_root)
        _step6a.validate_requested_mode(mode, configuration)
        scientific = _step6b.import_scientific_dependencies()
        model_settings = _step6b.validate_model_settings(raw_config)
        layout = _step6b.validate_repository_layout(raw_config, project_root)
        baseline_table = _step6b.load_baseline_table(layout, model_settings)
    except (_step6a.Step6AError, _step6b.Step6BError) as exc:
        raise Step6InputError(
            f"Existing Step 6A/6B validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    if tuple(configuration.phase_order) != PHASE_ORDER:
        raise Step6ConfigurationError(
            "Configured phase order does not match the controlled Step 6 order."
        )
    execution = _validate_execution_settings(raw_config)
    _validate_fixed_scientific_controls(configuration)
    validate_installed_relaxation_api(mode)
    config_snapshot = capture_file_snapshot(
        resolved_config, "Step 6 relaxation configuration"
    )
    phase_inputs: dict[str, PhaseInput] = {}
    for phase_key in selected_phases:
        try:
            definition = _step6b.phase_definition(phase_key)
            paths = _step6b.paths_for_phase(layout, definition)
            structure = _step6b.validate_selected_structure(
                definition, paths, project_root, scientific
            )
            baseline = _step6b.validate_phase_baseline(
                definition,
                paths,
                project_root,
                model_settings,
                structure,
                baseline_table,
                scientific,
            )
            evidence = _snapshot_step6b_evidence(project_root, phase_key)
        except (_step6b.Step6BError, OSError, UnicodeError) as exc:
            raise Step6InputError(
                f"Strict input validation failed for {phase_key}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        phase_inputs[phase_key] = PhaseInput(
            phase_key=phase_key,
            material_id=EXPECTED_MATERIAL_IDS[phase_key],
            atom_count=EXPECTED_ATOM_COUNTS[phase_key],
            structure=structure,
            baseline=baseline,
            step6b_snapshots=evidence,
        )

    mode_settings = (
        configuration.atomic_only
        if mode == ATOMIC_ONLY_MODE
        else configuration.full_cell
    )
    output_root = _mode_output_root(configuration, mode)
    provisional = Step6Context(
        project_root=project_root,
        config_path=resolved_config,
        config_snapshot=config_snapshot,
        config_fingerprint=config_snapshot.sha256,
        raw_config=raw_config,
        configuration=configuration,
        mode=mode,
        mode_settings=mode_settings,
        execution=execution,
        phase_keys=selected_phases,
        phase_inputs=phase_inputs,
        protected_snapshots=(),
        output_root=output_root,
    )
    protected = _collect_global_protected_snapshots(
        project_root, config_snapshot, phase_inputs
    )
    context = Step6Context(
        **{
            **provisional.__dict__,
            "protected_snapshots": protected,
        }
    )
    if create_directories:
        for directory in planned_mode_directories(context):
            directory.mkdir(parents=True, exist_ok=True)
    if require_step6_outputs:
        validate_all_mode_outputs(context)
    return context


def phase_output_paths(context: Step6Context, phase_key: str) -> ModeOutputPaths:
    """Resolve the canonical complete bundle for one phase."""

    if phase_key not in PHASE_ORDER:
        raise Step6ConfigurationError(f"Unknown phase key: {phase_key}")
    stem = f"{phase_key}_{context.mode}"
    return ModeOutputPaths(
        structure=context.output_root / "structures" / f"{stem}_relaxed.extxyz",
        trajectory=context.output_root / "trajectories" / f"{stem}.traj",
        history_csv=context.output_root / "tables" / f"{stem}_history.csv",
        report=context.output_root / "reports" / f"{stem}_report.txt",
        log=context.output_root / "logs" / f"{stem}.log",
        result_json=context.output_root / "checkpoints" / f"{stem}_result.json",
    )


def combined_output_paths(context: Step6Context) -> CombinedOutputPaths:
    """Resolve the required combined CSV, JSON, and text targets."""

    if context.mode == ATOMIC_ONLY_MODE:
        prefix = "ni_al_atomic_only_summary"
    else:
        prefix = "ni_al_full_cell_summary"
    return CombinedOutputPaths(
        csv=context.output_root / "tables" / f"{prefix}.csv",
        json=context.output_root / "tables" / f"{prefix}.json",
        report=context.output_root / "reports" / f"{prefix}.txt",
    )


def _numpy() -> Any:
    """Import NumPy without importing any calculator or optimizer."""

    try:
        import numpy as np
    except ImportError as exc:
        raise Step6DependencyError(f"NumPy is unavailable: {exc}") from exc
    return np


def _finite_float(value: Any, label: str) -> float:
    """Convert one numeric scalar and reject booleans and nonfinite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step6CalculationError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise Step6SafetyError(f"{label} is NaN or infinity.")
    return result


def _tuple_vector(values: Any, length: int, label: str) -> tuple[float, ...]:
    """Convert a one-dimensional finite vector of an exact length."""

    np = _numpy()
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not bool(np.all(np.isfinite(array))):
        raise Step6SafetyError(
            f"{label} must be a finite vector with shape ({length},); "
            f"received {array.shape}."
        )
    return tuple(float(value) for value in array)


def _tuple_matrix(
    values: Any, rows: int, columns: int, label: str
) -> tuple[tuple[float, ...], ...]:
    """Convert a finite matrix of an exact shape."""

    np = _numpy()
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, columns) or not bool(np.all(np.isfinite(array))):
        raise Step6SafetyError(
            f"{label} must be finite with shape ({rows}, {columns}); "
            f"received {array.shape}."
        )
    return tuple(
        tuple(float(value) for value in row)
        for row in array
    )


def _capture_initial_geometry(atoms: Any) -> InitialGeometry:
    """Capture independent source geometry and identity arrays."""

    np = _numpy()
    positions = np.asarray(atoms.get_positions(), dtype=float).copy()
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float).copy()
    cell = np.asarray(atoms.cell.array, dtype=float).copy()
    pbc_array = np.asarray(atoms.get_pbc(), dtype=bool)
    volume = float(atoms.get_volume())
    determinant = float(np.linalg.det(cell))
    if (
        positions.shape != (len(atoms), 3)
        or scaled.shape != (len(atoms), 3)
        or cell.shape != (3, 3)
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(scaled)))
        or not bool(np.all(np.isfinite(cell)))
        or not math.isfinite(volume)
        or volume <= 0.0
        or not math.isfinite(determinant)
        or determinant <= 0.0
    ):
        raise Step6InputError("Initial ASE structure geometry is invalid.")
    if pbc_array.shape != (3,) or not bool(np.all(pbc_array)):
        raise Step6InputError("Initial ASE structure must be periodic in x, y, and z.")
    return InitialGeometry(
        positions_A=positions,
        scaled_positions=scaled,
        cell_A=cell,
        symbols=tuple(atoms.get_chemical_symbols()),
        atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers()),
        pbc=tuple(bool(value) for value in pbc_array),
        atom_count=len(atoms),
        volume_A3=volume,
    )


def _wrapped_fractional_delta(current: Any, initial: Any, np: Any) -> Any:
    """Return deterministic minimum-image fractional differences."""

    delta = np.asarray(current, dtype=float) - np.asarray(initial, dtype=float)
    # floor(x + 0.5) maps each component into [-0.5, 0.5), avoiding
    # banker's-rounding ambiguity exactly at half a cell.
    return delta - np.floor(delta + 0.5)


def _displacement_metrics(
    atoms: Any, initial: InitialGeometry
) -> DisplacementMetrics:
    """Calculate internal and total displacement with fixed atom correspondence."""

    np = _numpy()
    current_scaled = np.asarray(
        atoms.get_scaled_positions(wrap=False), dtype=float
    )
    initial_scaled = np.asarray(initial.scaled_positions, dtype=float)
    initial_cell = np.asarray(initial.cell_A, dtype=float)
    current_cell = np.asarray(atoms.cell.array, dtype=float)
    delta_scaled = _wrapped_fractional_delta(current_scaled, initial_scaled, np)
    internal = delta_scaled @ initial_cell

    initial_wrapped = initial_scaled - np.floor(initial_scaled)
    correlated_final_scaled = initial_wrapped + delta_scaled
    total = (
        correlated_final_scaled @ current_cell
        - initial_wrapped @ initial_cell
    )
    internal_magnitudes = np.linalg.norm(internal, axis=1)
    total_magnitudes = np.linalg.norm(total, axis=1)
    if not (
        bool(np.all(np.isfinite(internal)))
        and bool(np.all(np.isfinite(total)))
        and bool(np.all(np.isfinite(internal_magnitudes)))
        and bool(np.all(np.isfinite(total_magnitudes)))
    ):
        raise Step6SafetyError("Atomic displacement metrics contain NaN or infinity.")

    def statistics(values: Any) -> tuple[float, float, float]:
        if len(values) == 0:
            return (0.0, 0.0, 0.0)
        return (
            float(np.max(values)),
            float(np.sqrt(np.mean(np.square(values)))),
            float(np.mean(values)),
        )

    maximum_internal, rms_internal, mean_internal = statistics(
        internal_magnitudes
    )
    maximum_total, rms_total, mean_total = statistics(total_magnitudes)
    per_species: dict[str, dict[str, float]] = {}
    symbols = tuple(atoms.get_chemical_symbols())
    for symbol in sorted(set(symbols)):
        indices = [index for index, item in enumerate(symbols) if item == symbol]
        subset = internal_magnitudes[indices]
        species_max, species_rms, species_mean = statistics(subset)
        per_species[symbol] = {
            "atom_count": float(len(indices)),
            "maximum_internal_displacement_A": species_max,
            "rms_internal_displacement_A": species_rms,
            "mean_internal_displacement_A": species_mean,
        }
    return DisplacementMetrics(
        internal_vectors_A=tuple(
            tuple(float(value) for value in row) for row in internal
        ),
        internal_magnitudes_A=tuple(float(value) for value in internal_magnitudes),
        maximum_internal_A=maximum_internal,
        rms_internal_A=rms_internal,
        mean_internal_A=mean_internal,
        total_vectors_A=tuple(
            tuple(float(value) for value in row) for row in total
        ),
        total_magnitudes_A=tuple(float(value) for value in total_magnitudes),
        maximum_total_A=maximum_total,
        rms_total_A=rms_total,
        mean_total_A=mean_total,
        per_species=per_species,
    )


def _validate_identity_and_geometry(
    atoms: Any,
    initial: InitialGeometry,
    context: Step6Context,
    displacement: DisplacementMetrics,
) -> float:
    """Enforce phase identity, finite geometry, volume, and displacement limits."""

    np = _numpy()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = tuple(bool(value) for value in atoms.get_pbc())
    determinant = float(np.linalg.det(cell))
    volume = float(atoms.get_volume())
    if len(atoms) != initial.atom_count:
        raise Step6SafetyError("Atom count changed during relaxation.")
    if tuple(atoms.get_chemical_symbols()) != initial.symbols:
        raise Step6SafetyError("Atom symbols or ordering changed during relaxation.")
    if tuple(int(value) for value in atoms.get_atomic_numbers()) != initial.atomic_numbers:
        raise Step6SafetyError("Atomic numbers or ordering changed during relaxation.")
    if pbc != initial.pbc:
        raise Step6SafetyError("Periodic-boundary flags changed during relaxation.")
    if (
        positions.shape != (initial.atom_count, 3)
        or cell.shape != (3, 3)
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(cell)))
    ):
        raise Step6SafetyError("Positions or cell contain NaN or infinity.")
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise Step6SafetyError(
            f"Cell determinant is nonpositive or nonfinite: {determinant!r}."
        )
    if not math.isfinite(volume) or volume <= 0.0:
        raise Step6SafetyError(f"Cell volume is invalid: {volume!r}.")

    volume_change_percent = 100.0 * (volume / initial.volume_A3 - 1.0)
    if context.mode == ATOMIC_ONLY_MODE:
        if not bool(
            np.allclose(
                cell,
                initial.cell_A,
                atol=ATOMIC_CELL_ATOL_A,
                rtol=0.0,
            )
        ):
            maximum_difference = float(np.max(np.abs(cell - initial.cell_A)))
            raise Step6SafetyError(
                "Atomic-only cell changed; maximum absolute component "
                f"difference={maximum_difference:.17g} A."
            )
        if not math.isclose(
            volume,
            initial.volume_A3,
            abs_tol=ATOMIC_CELL_ATOL_A,
            rel_tol=0.0,
        ):
            raise Step6SafetyError(
                "Atomic-only volume changed beyond atol=1e-12, rtol=0."
            )
    elif abs(volume_change_percent) > (
        context.configuration.safety.maximum_absolute_volume_change_percent
    ):
        raise Step6SafetyError(
            "Full-cell absolute volume change exceeded the configured limit: "
            f"{volume_change_percent:.17g}%."
        )

    if displacement.maximum_internal_A > (
        context.configuration.safety.maximum_atomic_displacement_A
    ):
        raise Step6SafetyError(
            "Maximum internal atomic displacement exceeded the configured "
            f"limit: {displacement.maximum_internal_A:.17g} A."
        )
    return volume_change_percent


def _evaluate_state(
    atoms: Any,
    initial: InitialGeometry,
    context: Step6Context,
    step: int,
    elapsed_seconds: float,
    session: CalculatorSession,
) -> StateMetrics:
    """Evaluate one complete raw state and apply all safety checks."""

    np = _numpy()
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception as exc:
        raise Step6CalculationError(
            f"State evaluation failed at optimizer step {step}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(energy):
        raise Step6SafetyError(f"Energy is nonfinite at optimizer step {step}.")
    if forces.shape != (initial.atom_count, 3) or not bool(
        np.all(np.isfinite(forces))
    ):
        raise Step6SafetyError(
            f"Forces are invalid at optimizer step {step}: shape={forces.shape}."
        )
    if stress.shape != (6,) or not bool(np.all(np.isfinite(stress))):
        raise Step6SafetyError(
            f"Stress is invalid at optimizer step {step}: shape={stress.shape}."
        )
    displacement = _displacement_metrics(atoms, initial)
    volume_change_percent = _validate_identity_and_geometry(
        atoms, initial, context, displacement
    )
    force_magnitudes = np.linalg.norm(forces, axis=1)
    total_force = np.sum(forces, axis=0)
    maximum_force = float(np.max(force_magnitudes))
    mean_force = float(np.mean(force_magnitudes))
    rms_force = float(np.sqrt(np.mean(np.square(force_magnitudes))))
    total_force_norm = float(np.linalg.norm(total_force))
    maximum_stress = float(np.max(np.abs(stress)))
    volume = float(atoms.get_volume())
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    if not (
        bool(np.all(np.isfinite(force_magnitudes)))
        and math.isfinite(maximum_force)
        and math.isfinite(rms_force)
        and math.isfinite(total_force_norm)
        and math.isfinite(maximum_stress)
        and bool(np.all(np.isfinite(lengths)))
        and bool(np.all(np.isfinite(angles)))
    ):
        raise Step6SafetyError(
            f"Derived state metrics are nonfinite at optimizer step {step}."
        )
    force_converged = (
        maximum_force <= context.mode_settings.force_threshold_eV_per_A
    )
    if context.mode == FULL_CELL_MODE:
        threshold = context.mode_settings.stress_threshold_eV_per_A3
        if threshold is None:
            raise Step6ConfigurationError(
                "Full-cell stress threshold is unexpectedly absent."
            )
        stress_converged: bool | None = maximum_stress <= threshold
        overall_converged = force_converged and stress_converged
    else:
        stress_converged = None
        overall_converged = force_converged

    session.state_evaluations += 1
    return StateMetrics(
        step=step,
        elapsed_seconds=float(elapsed_seconds),
        total_energy_eV=energy,
        energy_per_atom_eV=energy / initial.atom_count,
        forces_eV_per_A=tuple(
            tuple(float(value) for value in row) for row in forces
        ),
        force_magnitudes_eV_per_A=tuple(float(value) for value in force_magnitudes),
        maximum_force_eV_per_A=maximum_force,
        mean_force_eV_per_A=mean_force,
        rms_force_eV_per_A=rms_force,
        total_force_eV_per_A=tuple(float(value) for value in total_force),
        total_force_norm_eV_per_A=total_force_norm,
        stress_eV_per_A3=tuple(float(value) for value in stress),
        maximum_absolute_stress_eV_per_A3=maximum_stress,
        volume_A3=volume,
        volume_per_atom_A3=volume / initial.atom_count,
        cell_A=tuple(tuple(float(value) for value in row) for row in atoms.cell.array),
        lattice_lengths_A=tuple(float(value) for value in lengths),
        lattice_angles_deg=tuple(float(value) for value in angles),
        positions_A=tuple(
            tuple(float(value) for value in row)
            for row in atoms.get_positions()
        ),
        scaled_positions=tuple(
            tuple(float(value) for value in row)
            for row in atoms.get_scaled_positions(wrap=False)
        ),
        displacement=displacement,
        volume_change_percent=volume_change_percent,
        force_converged=force_converged,
        stress_converged=stress_converged,
        overall_converged=overall_converged,
    )


def load_calculator_session(context: Step6Context) -> CalculatorSession:
    """Lazily construct exactly one configured MACE calculator."""

    model = context.configuration.model
    captured: list[str] = []
    try:
        from ase.calculators.calculator import Calculator
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise Step6DependencyError(
            f"MACE/ASE calculator API import failed: {exc}"
        ) from exc

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calculator = mace_mp(
                model=model.value,
                device=model.device,
                default_dtype=model.default_dtype,
                dispersion=model.dispersion,
            )
        captured.extend(
            f"{item.category.__name__}: {item.message}" for item in caught
        )
    except Exception as exc:
        raise Step6CalculatorError(
            "MACE calculator loading failed with the validated settings: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(calculator, Calculator):
        raise Step6CalculatorError(
            "mace_mp did not return an ASE Calculator instance."
        )
    calculator_type = type(calculator)
    return CalculatorSession(
        calculator=calculator,
        calculator_class=(
            f"{calculator_type.__module__}.{calculator_type.__qualname__}"
        ),
        configuration_fingerprint=context.config_fingerprint,
        warnings=captured,
    )


def _reset_calculator(session: CalculatorSession, phase_key: str) -> None:
    """Clear shared calculator state before another independent phase."""

    try:
        session.calculator.reset()
    except Exception as exc:
        raise Step6CalculatorError(
            f"Could not reset calculator after {phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _displacement_to_json(metrics: DisplacementMetrics) -> dict[str, Any]:
    return {
        "internal_vectors_A": [list(row) for row in metrics.internal_vectors_A],
        "internal_magnitudes_A": list(metrics.internal_magnitudes_A),
        "maximum_internal_A": metrics.maximum_internal_A,
        "rms_internal_A": metrics.rms_internal_A,
        "mean_internal_A": metrics.mean_internal_A,
        "total_vectors_A": [list(row) for row in metrics.total_vectors_A],
        "total_magnitudes_A": list(metrics.total_magnitudes_A),
        "maximum_total_A": metrics.maximum_total_A,
        "rms_total_A": metrics.rms_total_A,
        "mean_total_A": metrics.mean_total_A,
        "per_species": {
            symbol: dict(values) for symbol, values in metrics.per_species.items()
        },
    }


def _state_to_json(state: StateMetrics) -> dict[str, Any]:
    """Serialize one complete state without loss of numeric precision."""

    return {
        "step": state.step,
        "elapsed_seconds": state.elapsed_seconds,
        "total_energy_eV": state.total_energy_eV,
        "energy_per_atom_eV": state.energy_per_atom_eV,
        "forces_eV_per_A": [list(row) for row in state.forces_eV_per_A],
        "force_magnitudes_eV_per_A": list(state.force_magnitudes_eV_per_A),
        "maximum_force_eV_per_A": state.maximum_force_eV_per_A,
        "mean_force_eV_per_A": state.mean_force_eV_per_A,
        "rms_force_eV_per_A": state.rms_force_eV_per_A,
        "total_force_eV_per_A": list(state.total_force_eV_per_A),
        "total_force_norm_eV_per_A": state.total_force_norm_eV_per_A,
        "stress_component_order": list(STRESS_COMPONENTS),
        "stress_eV_per_A3": list(state.stress_eV_per_A3),
        "maximum_absolute_stress_eV_per_A3": (
            state.maximum_absolute_stress_eV_per_A3
        ),
        "volume_A3": state.volume_A3,
        "volume_per_atom_A3": state.volume_per_atom_A3,
        "cell_A": [list(row) for row in state.cell_A],
        "lattice_lengths_A": list(state.lattice_lengths_A),
        "lattice_angles_deg": list(state.lattice_angles_deg),
        "positions_A": [list(row) for row in state.positions_A],
        "scaled_positions": [list(row) for row in state.scaled_positions],
        "displacement": _displacement_to_json(state.displacement),
        "volume_change_percent": state.volume_change_percent,
        "force_converged": state.force_converged,
        "stress_converged": state.stress_converged,
        "overall_converged": state.overall_converged,
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Step6ResumeError(f"{label} must be a JSON object.")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Step6ResumeError(f"{label} must be boolean.")
    return value


def _require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Step6ResumeError(f"{label} must be an integer >= {minimum}.")
    return value


def _state_from_json(raw_value: Any, label: str, atom_count: int) -> StateMetrics:
    """Strictly reconstruct one state from a resume manifest."""

    raw = _require_mapping(raw_value, label)
    displacement_raw = _require_mapping(raw.get("displacement"), f"{label}.displacement")
    per_species_raw = _require_mapping(
        displacement_raw.get("per_species"), f"{label}.displacement.per_species"
    )
    per_species: dict[str, dict[str, float]] = {}
    for symbol, values in per_species_raw.items():
        if symbol not in {"Al", "Ni"}:
            raise Step6ResumeError(f"{label} has unexpected species {symbol!r}.")
        mapping = _require_mapping(values, f"{label}.per_species.{symbol}")
        per_species[symbol] = {
            str(key): _finite_float(value, f"{label}.per_species.{symbol}.{key}")
            for key, value in mapping.items()
        }
    displacement = DisplacementMetrics(
        internal_vectors_A=_tuple_matrix(
            displacement_raw.get("internal_vectors_A"),
            atom_count,
            3,
            f"{label}.internal_vectors_A",
        ),
        internal_magnitudes_A=_tuple_vector(
            displacement_raw.get("internal_magnitudes_A"),
            atom_count,
            f"{label}.internal_magnitudes_A",
        ),
        maximum_internal_A=_finite_float(
            displacement_raw.get("maximum_internal_A"),
            f"{label}.maximum_internal_A",
        ),
        rms_internal_A=_finite_float(
            displacement_raw.get("rms_internal_A"), f"{label}.rms_internal_A"
        ),
        mean_internal_A=_finite_float(
            displacement_raw.get("mean_internal_A"), f"{label}.mean_internal_A"
        ),
        total_vectors_A=_tuple_matrix(
            displacement_raw.get("total_vectors_A"),
            atom_count,
            3,
            f"{label}.total_vectors_A",
        ),
        total_magnitudes_A=_tuple_vector(
            displacement_raw.get("total_magnitudes_A"),
            atom_count,
            f"{label}.total_magnitudes_A",
        ),
        maximum_total_A=_finite_float(
            displacement_raw.get("maximum_total_A"), f"{label}.maximum_total_A"
        ),
        rms_total_A=_finite_float(
            displacement_raw.get("rms_total_A"), f"{label}.rms_total_A"
        ),
        mean_total_A=_finite_float(
            displacement_raw.get("mean_total_A"), f"{label}.mean_total_A"
        ),
        per_species=per_species,
    )
    stress_converged_raw = raw.get("stress_converged")
    if stress_converged_raw is not None and not isinstance(
        stress_converged_raw, bool
    ):
        raise Step6ResumeError(f"{label}.stress_converged must be bool or null.")
    return StateMetrics(
        step=_require_int(raw.get("step"), f"{label}.step"),
        elapsed_seconds=_finite_float(
            raw.get("elapsed_seconds"), f"{label}.elapsed_seconds"
        ),
        total_energy_eV=_finite_float(
            raw.get("total_energy_eV"), f"{label}.total_energy_eV"
        ),
        energy_per_atom_eV=_finite_float(
            raw.get("energy_per_atom_eV"), f"{label}.energy_per_atom_eV"
        ),
        forces_eV_per_A=_tuple_matrix(
            raw.get("forces_eV_per_A"),
            atom_count,
            3,
            f"{label}.forces_eV_per_A",
        ),
        force_magnitudes_eV_per_A=_tuple_vector(
            raw.get("force_magnitudes_eV_per_A"),
            atom_count,
            f"{label}.force_magnitudes_eV_per_A",
        ),
        maximum_force_eV_per_A=_finite_float(
            raw.get("maximum_force_eV_per_A"),
            f"{label}.maximum_force_eV_per_A",
        ),
        mean_force_eV_per_A=_finite_float(
            raw.get("mean_force_eV_per_A"), f"{label}.mean_force_eV_per_A"
        ),
        rms_force_eV_per_A=_finite_float(
            raw.get("rms_force_eV_per_A"), f"{label}.rms_force_eV_per_A"
        ),
        total_force_eV_per_A=_tuple_vector(
            raw.get("total_force_eV_per_A"), 3, f"{label}.total_force_eV_per_A"
        ),
        total_force_norm_eV_per_A=_finite_float(
            raw.get("total_force_norm_eV_per_A"),
            f"{label}.total_force_norm_eV_per_A",
        ),
        stress_eV_per_A3=_tuple_vector(
            raw.get("stress_eV_per_A3"), 6, f"{label}.stress_eV_per_A3"
        ),
        maximum_absolute_stress_eV_per_A3=_finite_float(
            raw.get("maximum_absolute_stress_eV_per_A3"),
            f"{label}.maximum_absolute_stress_eV_per_A3",
        ),
        volume_A3=_finite_float(raw.get("volume_A3"), f"{label}.volume_A3"),
        volume_per_atom_A3=_finite_float(
            raw.get("volume_per_atom_A3"), f"{label}.volume_per_atom_A3"
        ),
        cell_A=_tuple_matrix(raw.get("cell_A"), 3, 3, f"{label}.cell_A"),
        lattice_lengths_A=_tuple_vector(
            raw.get("lattice_lengths_A"), 3, f"{label}.lattice_lengths_A"
        ),
        lattice_angles_deg=_tuple_vector(
            raw.get("lattice_angles_deg"), 3, f"{label}.lattice_angles_deg"
        ),
        positions_A=_tuple_matrix(
            raw.get("positions_A"), atom_count, 3, f"{label}.positions_A"
        ),
        scaled_positions=_tuple_matrix(
            raw.get("scaled_positions"),
            atom_count,
            3,
            f"{label}.scaled_positions",
        ),
        displacement=displacement,
        volume_change_percent=_finite_float(
            raw.get("volume_change_percent"), f"{label}.volume_change_percent"
        ),
        force_converged=_require_bool(
            raw.get("force_converged"), f"{label}.force_converged"
        ),
        stress_converged=stress_converged_raw,
        overall_converged=_require_bool(
            raw.get("overall_converged"), f"{label}.overall_converged"
        ),
    )


def _assert_numeric_close(
    actual: float,
    expected: float,
    label: str,
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
) -> None:
    if not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
        raise Step6ResumeError(
            f"{label} is internally inconsistent: {actual:.17g} versus "
            f"{expected:.17g}."
        )


def _validate_state_consistency(
    state: StateMetrics,
    context: Step6Context,
    atom_count: int,
    initial_volume_A3: float,
    label: str,
) -> None:
    """Recompute every key derived/convergence field in a stored state."""

    np = _numpy()
    forces = np.asarray(state.forces_eV_per_A, dtype=float)
    magnitudes = np.linalg.norm(forces, axis=1)
    stored_magnitudes = np.asarray(state.force_magnitudes_eV_per_A, dtype=float)
    if not bool(
        np.allclose(magnitudes, stored_magnitudes, atol=1.0e-12, rtol=1.0e-12)
    ):
        raise Step6ResumeError(f"{label} force magnitudes are inconsistent.")
    _assert_numeric_close(
        state.maximum_force_eV_per_A,
        float(np.max(magnitudes)),
        f"{label}.maximum_force",
    )
    _assert_numeric_close(
        state.mean_force_eV_per_A,
        float(np.mean(magnitudes)),
        f"{label}.mean_force",
    )
    _assert_numeric_close(
        state.rms_force_eV_per_A,
        float(np.sqrt(np.mean(np.square(magnitudes)))),
        f"{label}.rms_force",
    )
    total_force = np.sum(forces, axis=0)
    if not bool(
        np.allclose(
            total_force,
            np.asarray(state.total_force_eV_per_A),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    ):
        raise Step6ResumeError(f"{label} total force vector is inconsistent.")
    _assert_numeric_close(
        state.total_force_norm_eV_per_A,
        float(np.linalg.norm(total_force)),
        f"{label}.total_force_norm",
    )
    _assert_numeric_close(
        state.energy_per_atom_eV,
        state.total_energy_eV / atom_count,
        f"{label}.energy_per_atom",
    )
    _assert_numeric_close(
        state.volume_per_atom_A3,
        state.volume_A3 / atom_count,
        f"{label}.volume_per_atom",
    )
    _assert_numeric_close(
        state.maximum_absolute_stress_eV_per_A3,
        float(np.max(np.abs(np.asarray(state.stress_eV_per_A3)))),
        f"{label}.maximum_absolute_stress",
    )
    _assert_numeric_close(
        state.volume_change_percent,
        100.0 * (state.volume_A3 / initial_volume_A3 - 1.0),
        f"{label}.volume_change_percent",
    )
    internal_vectors = np.asarray(
        state.displacement.internal_vectors_A, dtype=float
    )
    internal_magnitudes = np.linalg.norm(internal_vectors, axis=1)
    total_vectors = np.asarray(state.displacement.total_vectors_A, dtype=float)
    total_magnitudes = np.linalg.norm(total_vectors, axis=1)
    if not bool(
        np.allclose(
            internal_magnitudes,
            np.asarray(state.displacement.internal_magnitudes_A),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    ) or not bool(
        np.allclose(
            total_magnitudes,
            np.asarray(state.displacement.total_magnitudes_A),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    ):
        raise Step6ResumeError(f"{label} displacement magnitudes are inconsistent.")
    for prefix, magnitudes_value, maximum, rms, mean in (
        (
            "internal",
            internal_magnitudes,
            state.displacement.maximum_internal_A,
            state.displacement.rms_internal_A,
            state.displacement.mean_internal_A,
        ),
        (
            "total",
            total_magnitudes,
            state.displacement.maximum_total_A,
            state.displacement.rms_total_A,
            state.displacement.mean_total_A,
        ),
    ):
        _assert_numeric_close(
            maximum, float(np.max(magnitudes_value)), f"{label}.{prefix}.maximum"
        )
        _assert_numeric_close(
            rms,
            float(np.sqrt(np.mean(np.square(magnitudes_value)))),
            f"{label}.{prefix}.rms",
        )
        _assert_numeric_close(
            mean, float(np.mean(magnitudes_value)), f"{label}.{prefix}.mean"
        )
    expected_force_converged = (
        state.maximum_force_eV_per_A
        <= context.mode_settings.force_threshold_eV_per_A
    )
    if state.force_converged != expected_force_converged:
        raise Step6ResumeError(f"{label} force-convergence boolean is invalid.")
    if context.mode == FULL_CELL_MODE:
        stress_threshold = context.mode_settings.stress_threshold_eV_per_A3
        if stress_threshold is None:
            raise Step6ResumeError("Full-cell stress threshold is absent.")
        expected_stress = (
            state.maximum_absolute_stress_eV_per_A3 <= stress_threshold
        )
        if state.stress_converged != expected_stress:
            raise Step6ResumeError(f"{label} stress-convergence boolean is invalid.")
        expected_overall = expected_force_converged and expected_stress
    else:
        if state.stress_converged is not None:
            raise Step6ResumeError(
                f"{label} atomic-only stress convergence must be null."
            )
        expected_overall = expected_force_converged
    if state.overall_converged != expected_overall:
        raise Step6ResumeError(f"{label} overall-convergence boolean is invalid.")


def _cell_change_json(initial: StateMetrics, final: StateMetrics) -> dict[str, Any]:
    """Return lattice, deformation-gradient, and Green-Lagrange strain changes."""

    np = _numpy()
    initial_cell = np.asarray(initial.cell_A, dtype=float)
    final_cell = np.asarray(final.cell_A, dtype=float)
    try:
        deformation = np.linalg.solve(initial_cell, final_cell).T
        strain = 0.5 * (deformation.T @ deformation - np.eye(3))
    except np.linalg.LinAlgError as exc:
        raise Step6SafetyError(
            f"Could not compute deformation gradient: {exc}"
        ) from exc
    if not (
        bool(np.all(np.isfinite(deformation)))
        and bool(np.all(np.isfinite(strain)))
    ):
        raise Step6SafetyError("Cell strain metrics are nonfinite.")
    return {
        "initial_cell_A": [list(row) for row in initial.cell_A],
        "final_cell_A": [list(row) for row in final.cell_A],
        "initial_lattice_lengths_A": list(initial.lattice_lengths_A),
        "final_lattice_lengths_A": list(final.lattice_lengths_A),
        "lattice_length_changes_A": [
            final.lattice_lengths_A[index] - initial.lattice_lengths_A[index]
            for index in range(3)
        ],
        "initial_lattice_angles_deg": list(initial.lattice_angles_deg),
        "final_lattice_angles_deg": list(final.lattice_angles_deg),
        "lattice_angle_changes_deg": [
            final.lattice_angles_deg[index] - initial.lattice_angles_deg[index]
            for index in range(3)
        ],
        "initial_volume_A3": initial.volume_A3,
        "final_volume_A3": final.volume_A3,
        "volume_change_A3": final.volume_A3 - initial.volume_A3,
        "volume_change_percent": final.volume_change_percent,
        "deformation_gradient": [
            [float(value) for value in row] for row in deformation
        ],
        "green_lagrange_strain": [
            [float(value) for value in row] for row in strain
        ],
    }


def _history_fieldnames() -> tuple[str, ...]:
    return (
        "step",
        "elapsed_seconds",
        "total_energy_eV",
        "energy_per_atom_eV",
        "maximum_force_eV_per_A",
        "mean_force_eV_per_A",
        "rms_force_eV_per_A",
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
        "volume_change_percent",
        "lattice_a_A",
        "lattice_b_A",
        "lattice_c_A",
        "angle_alpha_deg",
        "angle_beta_deg",
        "angle_gamma_deg",
        "maximum_internal_displacement_A",
        "rms_internal_displacement_A",
        "maximum_total_displacement_A",
        "rms_total_displacement_A",
        "force_converged",
        "stress_converged",
        "overall_converged",
        "safety_status",
    )


def _history_row(state: StateMetrics) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": state.step,
        "elapsed_seconds": state.elapsed_seconds,
        "total_energy_eV": state.total_energy_eV,
        "energy_per_atom_eV": state.energy_per_atom_eV,
        "maximum_force_eV_per_A": state.maximum_force_eV_per_A,
        "mean_force_eV_per_A": state.mean_force_eV_per_A,
        "rms_force_eV_per_A": state.rms_force_eV_per_A,
        "total_force_norm_eV_per_A": state.total_force_norm_eV_per_A,
        "maximum_absolute_stress_eV_per_A3": (
            state.maximum_absolute_stress_eV_per_A3
        ),
        "volume_A3": state.volume_A3,
        "volume_per_atom_A3": state.volume_per_atom_A3,
        "volume_change_percent": state.volume_change_percent,
        "maximum_internal_displacement_A": (
            state.displacement.maximum_internal_A
        ),
        "rms_internal_displacement_A": state.displacement.rms_internal_A,
        "maximum_total_displacement_A": state.displacement.maximum_total_A,
        "rms_total_displacement_A": state.displacement.rms_total_A,
        "force_converged": state.force_converged,
        "stress_converged": (
            "" if state.stress_converged is None else state.stress_converged
        ),
        "overall_converged": state.overall_converged,
        "safety_status": "PASS",
    }
    for component, value in zip(STRESS_COMPONENTS, state.stress_eV_per_A3):
        row[f"stress_{component}_eV_per_A3"] = value
    for label, value in zip(("a", "b", "c"), state.lattice_lengths_A):
        row[f"lattice_{label}_A"] = value
    for label, value in zip(
        ("alpha", "beta", "gamma"), state.lattice_angles_deg
    ):
        row[f"angle_{label}_deg"] = value
    return row


def _history_csv_bytes(history: Sequence[StateMetrics]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_history_fieldnames())
    writer.writeheader()
    for state in history:
        writer.writerow(_history_row(state))
    return buffer.getvalue().encode("utf-8")


def _format_vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.17g}" for value in values) + "]"


def _phase_report_text(context: Step6Context, result: PhaseResult) -> str:
    """Build the complete human-readable Step 6C/6D phase report."""

    mode_title = (
        "Atomic-Only Relaxation"
        if context.mode == ATOMIC_ONLY_MODE
        else "Full-Cell Relaxation"
    )
    settings = context.mode_settings
    initial = result.initial
    final = result.final
    cell_change = _cell_change_json(initial, final)
    stress_requirement = (
        "Not required in atomic-only mode"
        if final.stress_converged is None
        else ("PASS" if final.stress_converged else "FAIL")
    )
    filter_lines: list[str]
    if context.mode == FULL_CELL_MODE:
        filter_lines = [
            "Filter: ase.filters.FrechetCellFilter",
            "Filter mask: all six strain components",
            f"Hydrostatic strain: {str(settings.hydrostatic_strain).lower()}",
            f"Constant volume: {str(settings.constant_volume).lower()}",
            "External pressure (eV/angstrom^3): 0",
            (
                "ASE generalized auto-stop fmax: 0.0 eV/angstrom "
                "(control sentinel only; actual force and stress criteria "
                "are evaluated explicitly)"
            ),
        ]
    else:
        filter_lines = ["Filter: None; cell fixed exactly"]
    lines = [
        f"Step {'6C' if context.mode == ATOMIC_ONLY_MODE else '6D'} - "
        f"{result.phase_key} {mode_title}",
        "=" * 76,
        "",
        "1. Identity and provenance",
        "--------------------------",
        f"Phase: {result.phase_key}",
        f"Materials Project ID: {result.material_id}",
        f"Atom count: {EXPECTED_ATOM_COUNTS[result.phase_key]}",
        f"Mode: {result.mode}",
        f"Configuration: {relative_path(context.config_path, context.project_root)}",
        f"Configuration SHA-256: {context.config_fingerprint}",
        f"Started (UTC): {result.started_at_utc}",
        f"Completed (UTC): {result.completed_at_utc}",
        "",
        "2. Model settings",
        "-----------------",
        f"Family: {context.configuration.model.family}",
        f"Name: {context.configuration.model.name}",
        f"Model value: {context.configuration.model.value}",
        f"Device: {context.configuration.model.device}",
        f"Default dtype: {context.configuration.model.default_dtype}",
        f"Dispersion enabled: {str(context.configuration.model.dispersion).lower()}",
        f"Calculator class: {result.calculator_class}",
        f"Calculator loads in session: {result.calculator_load_count}",
        f"Recorded state/single-point evaluations: {result.state_evaluations}",
        "",
        "3. Optimizer and filter settings",
        "--------------------------------",
        f"Optimizer: {settings.optimizer}",
        f"Optimizer created: {str(result.optimizer_created)}",
        f"Maximum steps: {settings.maximum_steps}",
        f"Actual optimizer steps: {result.optimizer_steps}",
        f"Force threshold (eV/angstrom): {settings.force_threshold_eV_per_A:.17g}",
    ]
    if settings.stress_threshold_eV_per_A3 is not None:
        lines.append(
            "Stress threshold (eV/angstrom^3): "
            f"{settings.stress_threshold_eV_per_A3:.17g}"
        )
    lines.extend(filter_lines)
    lines.extend(
        [
            "",
            "4. Initial structure values",
            "---------------------------",
            f"Total energy (eV): {initial.total_energy_eV:.17g}",
            f"Energy per atom (eV/atom): {initial.energy_per_atom_eV:.17g}",
            f"Maximum force (eV/angstrom): {initial.maximum_force_eV_per_A:.17g}",
            f"RMS force (eV/angstrom): {initial.rms_force_eV_per_A:.17g}",
            f"Stress [xx, yy, zz, yz, xz, xy] (eV/angstrom^3): "
            f"{_format_vector(initial.stress_eV_per_A3)}",
            f"Volume (angstrom^3): {initial.volume_A3:.17g}",
            "Cell matrix (angstrom): "
            + json.dumps([list(row) for row in initial.cell_A], separators=(",", ":")),
            f"Lattice lengths (angstrom): {_format_vector(initial.lattice_lengths_A)}",
            f"Lattice angles (degrees): {_format_vector(initial.lattice_angles_deg)}",
            f"Initial force convergence: {'PASS' if initial.force_converged else 'FAIL'}",
            "",
            "5. Final structure values",
            "-------------------------",
            f"Total energy (eV): {final.total_energy_eV:.17g}",
            f"Energy per atom (eV/atom): {final.energy_per_atom_eV:.17g}",
            f"Maximum force (eV/angstrom): {final.maximum_force_eV_per_A:.17g}",
            f"RMS force (eV/angstrom): {final.rms_force_eV_per_A:.17g}",
            f"Stress [xx, yy, zz, yz, xz, xy] (eV/angstrom^3): "
            f"{_format_vector(final.stress_eV_per_A3)}",
            f"Volume (angstrom^3): {final.volume_A3:.17g}",
            "Cell matrix (angstrom): "
            + json.dumps([list(row) for row in final.cell_A], separators=(",", ":")),
            f"Lattice lengths (angstrom): {_format_vector(final.lattice_lengths_A)}",
            f"Lattice angles (degrees): {_format_vector(final.lattice_angles_deg)}",
            "",
            "6. Changes",
            "----------",
            f"Energy change (eV): {final.total_energy_eV - initial.total_energy_eV:.17g}",
            f"Maximum-force change (eV/angstrom): "
            f"{final.maximum_force_eV_per_A - initial.maximum_force_eV_per_A:.17g}",
            "Stress change [xx, yy, zz, yz, xz, xy] (eV/angstrom^3): "
            + _format_vector(
                [
                    final.stress_eV_per_A3[index]
                    - initial.stress_eV_per_A3[index]
                    for index in range(6)
                ]
            ),
            f"Volume change (angstrom^3): "
            f"{final.volume_A3 - initial.volume_A3:.17g}",
            f"Volume change (percent): {final.volume_change_percent:.17g}",
            f"Lattice-length changes (angstrom): "
            f"{_format_vector(cell_change['lattice_length_changes_A'])}",
            f"Lattice-angle changes (degrees): "
            f"{_format_vector(cell_change['lattice_angle_changes_deg'])}",
            "",
            "7. Atomic displacement statistics",
            "---------------------------------",
            f"Maximum internal displacement (angstrom): "
            f"{final.displacement.maximum_internal_A:.17g}",
            f"RMS internal displacement (angstrom): "
            f"{final.displacement.rms_internal_A:.17g}",
            f"Mean internal displacement (angstrom): "
            f"{final.displacement.mean_internal_A:.17g}",
            f"Maximum total Cartesian displacement (angstrom): "
            f"{final.displacement.maximum_total_A:.17g}",
            f"RMS total Cartesian displacement (angstrom): "
            f"{final.displacement.rms_total_A:.17g}",
            (
                "Internal displacement uses wrapped fractional-coordinate "
                "differences mapped through the initial cell. Total Cartesian "
                "displacement includes cell deformation."
            ),
            "Per-atom internal displacement vectors (angstrom):",
        ]
    )
    for index, (symbol, vector, magnitude) in enumerate(
        zip(
            context.phase_inputs[result.phase_key].structure.atom_order,
            final.displacement.internal_vectors_A,
            final.displacement.internal_magnitudes_A,
        )
    ):
        lines.append(
            f"  atom {index} {symbol}: {_format_vector(vector)}; "
            f"magnitude={magnitude:.17g}"
        )
    lines.extend(
        [
            "Per-species internal displacement statistics:",
        ]
    )
    for symbol, values in final.displacement.per_species.items():
        lines.append(
            f"  {symbol}: count={int(values['atom_count'])}; "
            f"max={values['maximum_internal_displacement_A']:.17g} A; "
            f"rms={values['rms_internal_displacement_A']:.17g} A; "
            f"mean={values['mean_internal_displacement_A']:.17g} A"
        )
    lines.extend(
        [
            "",
            "8. Safety and provenance checks",
            "-------------------------------",
            f"Safety status: {result.safety_status}",
            "Positions/cell/energy/forces/stress finite: PASS",
            "Positive cell determinant and volume: PASS",
            "Atom identity and ordering preserved: PASS",
            "Periodic-boundary flags preserved: PASS",
            "Configured displacement and volume limits: PASS",
            "Protected source files unchanged: PASS",
            "Protected file fingerprints:",
        ]
    )
    for snapshot in result.source_snapshots:
        lines.append(
            "  "
            f"{relative_path(snapshot.path, context.project_root)}: "
            f"sha256={snapshot.sha256}; size={snapshot.size}; "
            f"mtime_ns={snapshot.modification_time_ns}"
        )
    lines.extend(
        [
            "",
            "9. Output paths",
            "---------------",
        ]
    )
    for name, path in (
        ("Final structure", result.output_paths.structure),
        ("Trajectory", result.output_paths.trajectory),
        ("History CSV", result.output_paths.history_csv),
        ("Report", result.output_paths.report),
        ("Optimizer log", result.output_paths.log),
        ("Resume manifest", result.output_paths.result_json),
    ):
        lines.append(f"{name}: {relative_path(path, context.project_root)}")
    lines.extend(
        [
            "",
            "10. Convergence status",
            "----------------------",
            f"Force convergence: {'PASS' if final.force_converged else 'FAIL'}",
            f"Stress convergence: {stress_requirement}",
            f"Overall convergence: {'PASS' if final.overall_converged else 'FAIL'}",
            f"Phase status: {result.status}",
            "",
            "11. Warnings",
            "------------",
        ]
    )
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "12. Scientific interpretation boundary",
            "--------------------------------------",
            (
                "This is a geometry relaxation on the MACE potential-energy "
                "surface. It is not a DFT-accuracy or experimental-accuracy "
                "conclusion and is not a cross-composition phase-stability ranking."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_record(stage_path: Path, final_path: Path, context: Step6Context) -> dict[str, Any]:
    stat = stage_path.stat()
    return {
        "path": relative_path(final_path, context.project_root),
        "sha256": file_sha256(stage_path),
        "size_bytes": stat.st_size,
    }


def _manifest_document(
    context: Step6Context,
    result: PhaseResult,
    staged: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the machine-readable resume authority for one phase."""

    output_by_name = {
        "final_structure": result.output_paths.structure,
        "trajectory": result.output_paths.trajectory,
        "history_csv": result.output_paths.history_csv,
        "report": result.output_paths.report,
        "optimizer_log": result.output_paths.log,
    }
    artifacts = {
        name: _artifact_record(staged[name], output_by_name[name], context)
        for name in output_by_name
    }
    settings = context.mode_settings
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_relaxation_phase_result",
        "project_step": "6C" if context.mode == ATOMIC_ONLY_MODE else "6D",
        "phase_key": result.phase_key,
        "material_id": result.material_id,
        "number_of_atoms": EXPECTED_ATOM_COUNTS[result.phase_key],
        "mode": result.mode,
        "generated_at_utc": result.completed_at_utc,
        "configuration_path": relative_path(
            context.config_path, context.project_root
        ),
        "configuration_fingerprint_sha256": result.configuration_fingerprint,
        "execution_status": "COMPLETED",
        "convergence_status": result.status,
        "force_converged": result.force_converged,
        "stress_converged": result.stress_converged,
        "overall_converged": result.overall_converged,
        "safety_status": result.safety_status,
        "output_validation_status": "PASS",
        "model": {
            "family": context.configuration.model.family,
            "name": context.configuration.model.name,
            "value": context.configuration.model.value,
            "device": context.configuration.model.device,
            "default_dtype": context.configuration.model.default_dtype,
            "dispersion": context.configuration.model.dispersion,
            "calculator_class": result.calculator_class,
        },
        "optimizer": {
            "name": settings.optimizer,
            "created": result.optimizer_created,
            "steps": result.optimizer_steps,
            "maximum_steps": settings.maximum_steps,
            "force_threshold_eV_per_A": settings.force_threshold_eV_per_A,
            "stress_threshold_eV_per_A3": settings.stress_threshold_eV_per_A3,
            "trajectory_interval": settings.trajectory_interval,
            "history_interval": context.execution.history_interval,
            "ase_generalized_auto_stop_fmax_eV_per_A": (
                GENERALIZED_FORCE_AUTO_STOP_FMAX
                if context.mode == FULL_CELL_MODE
                else settings.force_threshold_eV_per_A
            ),
            "filter": (
                None
                if context.mode == ATOMIC_ONLY_MODE
                else {
                    "name": "FrechetCellFilter",
                    "mask": "all",
                    "exp_cell_factor": float(
                        EXPECTED_ATOM_COUNTS[result.phase_key]
                    ),
                    "hydrostatic_strain": settings.hydrostatic_strain,
                    "constant_volume": settings.constant_volume,
                    "external_pressure_eV_per_A3": (
                        context.execution.external_pressure_eV_per_A3
                    ),
                }
            ),
        },
        "counts": {
            "calculator_loads_in_session": result.calculator_load_count,
            "state_evaluations": result.state_evaluations,
            "optimizer_steps": result.optimizer_steps,
        },
        "timing": {
            "started_at_utc": result.started_at_utc,
            "completed_at_utc": result.completed_at_utc,
            "wall_time_seconds": result.wall_time_seconds,
        },
        "initial": _state_to_json(result.initial),
        "final": _state_to_json(result.final),
        "changes": {
            "total_energy_change_eV": (
                result.final.total_energy_eV - result.initial.total_energy_eV
            ),
            "energy_per_atom_change_eV": (
                result.final.energy_per_atom_eV
                - result.initial.energy_per_atom_eV
            ),
            "maximum_force_change_eV_per_A": (
                result.final.maximum_force_eV_per_A
                - result.initial.maximum_force_eV_per_A
            ),
            "stress_change_eV_per_A3": [
                result.final.stress_eV_per_A3[index]
                - result.initial.stress_eV_per_A3[index]
                for index in range(6)
            ],
            "cell": _cell_change_json(result.initial, result.final),
            "displacement": _displacement_to_json(result.final.displacement),
        },
        "history": [_state_to_json(state) for state in result.history],
        "warnings": list(result.warnings),
        "protected_sources": [
            snapshot.to_json(context.project_root)
            for snapshot in result.source_snapshots
        ],
        "artifacts": artifacts,
        "scientific_limitations": [
            "MACE-potential relaxation; not a DFT validation.",
            "Not an experimental validation.",
            "Raw energies are not ranked across compositions.",
            "No formation energies were calculated.",
        ],
    }


def _stage_path(
    staging_root: Path, output_root: Path, final_path: Path
) -> Path:
    """Map a canonical output target into a same-volume staging tree."""

    try:
        relative = final_path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise Step6PublicationError(
            f"Output target escaped its mode root: {final_path}"
        ) from exc
    staged = staging_root / relative
    staged.parent.mkdir(parents=True, exist_ok=True)
    return staged


def _write_roundtrip_safe_extxyz(atoms: Any, path: Path) -> None:
    """Write one ASE-header EXTXYZ frame with 17-digit Cartesian positions."""

    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
    except ImportError as exc:
        raise Step6DependencyError(f"ASE writer is unavailable: {exc}") from exc
    try:
        ase_write(
            path,
            atoms,
            format="extxyz",
            columns=["symbols", "positions"],
            write_info=True,
            write_results=False,
        )
        # ASE 3.29's EXTXYZ writer formats Cartesian positions with eight
        # decimal places. Preserve its generated count/header verbatim, but
        # replace the simple species+position rows with round-trip-safe values.
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(atoms) + 2:
            raise Step6PublicationError(
                "ASE generated an unexpected EXTXYZ frame layout."
            )
        rewritten = lines[:2]
        for symbol, position in zip(
            atoms.get_chemical_symbols(), atoms.get_positions()
        ):
            rewritten.append(
                f"{symbol} "
                + " ".join(f"{float(value):.17g}" for value in position)
            )
        path.write_text(
            "\n".join(rewritten) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        roundtrip_frames = ase_read(path, index=":", format="extxyz")
        if not isinstance(roundtrip_frames, list) or len(roundtrip_frames) != 1:
            raise Step6PublicationError(
                "High-precision final EXTXYZ did not round-trip as one frame."
            )
        roundtrip = roundtrip_frames[0]
        np = _numpy()
        if not bool(
            np.allclose(
                roundtrip.get_positions(),
                atoms.get_positions(),
                atol=1.0e-12,
                rtol=0.0,
            )
        ) or not bool(
            np.allclose(
                roundtrip.cell.array,
                atoms.cell.array,
                atol=1.0e-12,
                rtol=0.0,
            )
        ):
            raise Step6PublicationError(
                "High-precision final EXTXYZ failed the 1e-12 round-trip check."
            )
    except Step6Error:
        raise
    except Exception as exc:
        raise Step6PublicationError(
            f"Could not write round-trip-safe EXTXYZ: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _write_final_structure(
    context: Step6Context,
    result: PhaseResult,
    atoms: Any,
    path: Path,
) -> None:
    """Write a detached final EXTXYZ with prefixed project metadata."""

    if atoms.calc is not None:
        raise Step6PublicationError(
            "Refusing to write a final structure with an attached calculator."
        )
    output_atoms = atoms.copy()
    output_atoms.calc = None
    output_atoms.info.update(
        {
            "step6_phase": result.phase_key,
            "step6_material_id": result.material_id,
            "step6_mode": result.mode,
            "step6_model": context.configuration.model.name,
            "step6_model_value": context.configuration.model.value,
            "step6_device": context.configuration.model.device,
            "step6_dtype": context.configuration.model.default_dtype,
            "step6_convergence_status": result.status,
            "step6_optimizer_steps": result.optimizer_steps,
            "step6_initial_energy_eV": result.initial.total_energy_eV,
            "step6_final_energy_eV": result.final.total_energy_eV,
            "step6_initial_max_force_eV_per_A": (
                result.initial.maximum_force_eV_per_A
            ),
            "step6_final_max_force_eV_per_A": (
                result.final.maximum_force_eV_per_A
            ),
            "step6_initial_stress_eV_per_A3_json": json.dumps(
                list(result.initial.stress_eV_per_A3), separators=(",", ":")
            ),
            "step6_final_stress_eV_per_A3_json": json.dumps(
                list(result.final.stress_eV_per_A3), separators=(",", ":")
            ),
            "step6_initial_volume_A3": result.initial.volume_A3,
            "step6_final_volume_A3": result.final.volume_A3,
            "step6_maximum_internal_displacement_A": (
                result.final.displacement.maximum_internal_A
            ),
            "step6_source_structure_path": relative_path(
                context.phase_inputs[result.phase_key].structure.structure_snapshot.path,
                context.project_root,
            ),
            "step6_execution_timestamp_utc": result.completed_at_utc,
            "step6_configuration_sha256": result.configuration_fingerprint,
        }
    )
    try:
        _write_roundtrip_safe_extxyz(output_atoms, path)
    except Exception as exc:
        if isinstance(exc, Step6Error):
            raise
        raise Step6PublicationError(
            f"Could not write staged final EXTXYZ for {result.phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _write_already_converged_trajectory(path: Path, atoms: Any) -> None:
    """Write exactly one frame for a phase converged before optimizer creation."""

    try:
        from ase.io.trajectory import Trajectory
        with Trajectory(path, mode="w", atoms=atoms) as trajectory:
            trajectory.write(atoms)
    except Exception as exc:
        raise Step6PublicationError(
            f"Could not write the step-0 trajectory: {type(exc).__name__}: {exc}"
        ) from exc


def _validate_staged_phase_bundle(
    context: Step6Context,
    result: PhaseResult,
    staged: Mapping[str, Path],
) -> None:
    """Fully validate temporary phase artifacts before any final path changes."""

    document = read_json_object(
        staged["result_json"], f"staged {result.phase_key} resume manifest"
    )
    if (
        document.get("artifact_type") != "ni_al_mace_relaxation_phase_result"
        or document.get("phase_key") != result.phase_key
        or document.get("material_id") != result.material_id
        or document.get("mode") != context.mode
        or document.get("configuration_fingerprint_sha256")
        != context.config_fingerprint
        or document.get("convergence_status") != result.status
        or document.get("safety_status") != "PASS"
    ):
        raise Step6PublicationError(
            f"Staged manifest identity/status is invalid for {result.phase_key}."
        )
    atom_count = EXPECTED_ATOM_COUNTS[result.phase_key]
    initial_manifest = _state_from_json(
        document.get("initial"), f"staged {result.phase_key}.initial", atom_count
    )
    final_manifest = _state_from_json(
        document.get("final"), f"staged {result.phase_key}.final", atom_count
    )
    if initial_manifest != result.initial or final_manifest != result.final:
        raise Step6PublicationError(
            f"Staged manifest states disagree for {result.phase_key}."
        )
    for index, state in enumerate(result.history):
        _validate_state_consistency(
            state,
            context,
            atom_count,
            result.initial.volume_A3,
            f"staged {result.phase_key}.history[{index}]",
        )

    artifacts = _require_mapping(
        document.get("artifacts"), f"staged {result.phase_key}.artifacts"
    )
    for name, final_path in (
        ("final_structure", result.output_paths.structure),
        ("trajectory", result.output_paths.trajectory),
        ("history_csv", result.output_paths.history_csv),
        ("report", result.output_paths.report),
        ("optimizer_log", result.output_paths.log),
    ):
        raw = _require_mapping(
            artifacts.get(name), f"staged {result.phase_key}.artifacts.{name}"
        )
        if raw.get("path") != relative_path(final_path, context.project_root):
            raise Step6PublicationError(
                f"Staged artifact path mismatch for {result.phase_key}/{name}."
            )
        if (
            raw.get("sha256") != file_sha256(staged[name])
            or raw.get("size_bytes") != staged[name].stat().st_size
        ):
            raise Step6PublicationError(
                f"Staged artifact hash mismatch for {result.phase_key}/{name}."
            )
    try:
        from ase.io import read as ase_read
        from ase.io.trajectory import Trajectory

        final_frames = ase_read(
            staged["final_structure"], index=":", format="extxyz"
        )
        with Trajectory(staged["trajectory"], mode="r") as trajectory:
            trajectory_frames = [frame for frame in trajectory]
    except Exception as exc:
        raise Step6PublicationError(
            f"Could not read staged scientific artifacts for {result.phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(final_frames, list) or len(final_frames) != 1:
        raise Step6PublicationError(
            f"Staged final EXTXYZ for {result.phase_key} must have one frame."
        )
    final_atoms = final_frames[0]
    final_atoms.calc = None
    source_atoms = context.phase_inputs[result.phase_key].structure.atoms
    if not trajectory_frames:
        raise Step6PublicationError(
            f"Staged trajectory for {result.phase_key} is empty."
        )
    _compare_geometry(
        trajectory_frames[0],
        source_atoms,
        label=f"staged {result.phase_key} trajectory start",
    )
    _compare_geometry(
        trajectory_frames[-1],
        final_atoms,
        label=f"staged {result.phase_key} trajectory end",
    )
    np = _numpy()
    if not bool(
        np.allclose(
            final_atoms.get_positions(),
            np.asarray(result.final.positions_A),
            atol=1.0e-12,
            rtol=0.0,
        )
    ) or not bool(
        np.allclose(
            final_atoms.cell.array,
            np.asarray(result.final.cell_A),
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise Step6PublicationError(
            f"Staged final geometry disagrees for {result.phase_key}."
        )
    if context.mode == ATOMIC_ONLY_MODE and not bool(
        np.allclose(
            final_atoms.cell.array,
            source_atoms.cell.array,
            atol=ATOMIC_CELL_ATOL_A,
            rtol=0.0,
        )
    ):
        raise Step6PublicationError(
            f"Staged atomic-only cell changed for {result.phase_key}."
        )
    report_text = staged["report"].read_text(encoding="utf-8")
    for sentinel in (
        f"Phase: {result.phase_key}",
        f"Configuration SHA-256: {context.config_fingerprint}",
        f"Phase status: {result.status}",
        "Safety status: PASS",
    ):
        if sentinel not in report_text:
            raise Step6PublicationError(
                f"Staged report lacks {sentinel!r} for {result.phase_key}."
            )
    try:
        with staged["history_csv"].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Step6PublicationError(
            f"Could not parse staged history for {result.phase_key}: {exc}"
        ) from exc
    if (
        not rows
        or tuple(rows[0].keys()) != _history_fieldnames()
        or int(rows[0]["step"]) != 0
        or int(rows[-1]["step"]) != result.optimizer_steps
    ):
        raise Step6PublicationError(
            f"Staged history endpoints/schema are invalid for {result.phase_key}."
        )
    verify_protected_files(context)


def _run_relaxation(
    context: Step6Context,
    phase_key: str,
    working: Any,
    initial_geometry: InitialGeometry,
    initial_state: StateMetrics,
    session: CalculatorSession,
    trajectory_path: Path,
    log_path: Path,
    started_monotonic: float,
) -> tuple[str, bool, int, tuple[StateMetrics, ...], tuple[str, ...]]:
    """Run FIRE when needed, enforcing exact externally measured convergence."""

    if initial_state.overall_converged:
        _write_already_converged_trajectory(trajectory_path, working)
        log_path.write_text(
            "Optimizer not created: the initial state already met all "
            "configured convergence criteria.\n",
            encoding="utf-8",
        )
        return (
            "ALREADY_CONVERGED",
            False,
            0,
            (initial_state,),
            (),
        )

    try:
        from ase.io.trajectory import Trajectory
        from ase.optimize import FIRE
    except ImportError as exc:
        raise Step6DependencyError(
            f"Installed ASE FIRE/Trajectory API is unavailable: {exc}"
        ) from exc

    target: Any = working
    if context.mode == FULL_CELL_MODE:
        try:
            from ase.filters import FrechetCellFilter
        except ImportError as exc:
            ase_version = importlib.metadata.version("ase")
            raise Step6DependencyError(
                "ase.filters.FrechetCellFilter is unavailable in installed "
                f"ASE {ase_version}: {exc}"
            ) from exc
        target = FrechetCellFilter(
            working,
            mask=None,
            exp_cell_factor=None,
            hydrostatic_strain=context.mode_settings.hydrostatic_strain,
            constant_volume=context.mode_settings.constant_volume,
            scalar_pressure=context.execution.external_pressure_eV_per_A3,
        )

    history: list[StateMetrics] = []
    last_state: StateMetrics = initial_state
    phase_warnings: list[str] = []
    optimizer: Any = None

    def monitor() -> None:
        """Measure actual Atoms values; never use filter convergence proxies."""

        nonlocal last_state
        if optimizer is None:
            raise Step6CalculationError("Optimizer observer ran before setup.")
        step = int(optimizer.get_number_of_steps())
        if step == 0 and not history:
            state = initial_state
        else:
            state = _evaluate_state(
                working,
                initial_geometry,
                context,
                step,
                time.monotonic() - started_monotonic,
                session,
            )
        last_state = state
        if (
            step % context.execution.history_interval == 0
            or state.overall_converged
            or step == context.mode_settings.maximum_steps
        ):
            if not history or history[-1].step != state.step:
                history.append(state)

    internal_fmax = (
        context.mode_settings.force_threshold_eV_per_A
        if context.mode == ATOMIC_ONLY_MODE
        else GENERALIZED_FORCE_AUTO_STOP_FMAX
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Trajectory(
                trajectory_path, mode="w", atoms=working
            ) as trajectory:
                with FIRE(
                    target,
                    logfile=str(log_path),
                    trajectory=None,
                ) as optimizer_instance:
                    optimizer = optimizer_instance
                    optimizer.attach(monitor, interval=1)
                    optimizer.attach(
                        trajectory.write,
                        interval=context.mode_settings.trajectory_interval,
                    )
                    iterator = optimizer.irun(
                        fmax=internal_fmax,
                        steps=context.mode_settings.maximum_steps,
                    )
                    reached_convergence = False
                    try:
                        for _generalized_convergence in iterator:
                            if last_state.overall_converged:
                                reached_convergence = True
                                break
                    finally:
                        iterator.close()
                    optimizer_steps = int(optimizer.get_number_of_steps())
                    if (
                        history
                        and history[-1].step != optimizer_steps
                    ):
                        history.append(last_state)
                    if (
                        optimizer_steps
                        % context.mode_settings.trajectory_interval
                        != 0
                    ):
                        trajectory.write(working)
                    if reached_convergence:
                        status = "CONVERGED"
                    else:
                        status = "NOT_CONVERGED"
            phase_warnings.extend(
                f"{item.category.__name__}: {item.message}" for item in caught
            )
    except Step6Error:
        raise
    except Exception as exc:
        raise Step6CalculationError(
            f"FIRE relaxation failed for {phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not history:
        raise Step6CalculationError(
            f"FIRE produced no recorded states for {phase_key}."
        )
    # Reassert exact scientific status independent of generator termination.
    if last_state.overall_converged:
        status = "CONVERGED"
    elif optimizer_steps >= context.mode_settings.maximum_steps:
        status = "NOT_CONVERGED"
    else:
        raise Step6CalculationError(
            f"FIRE stopped unexpectedly after {optimizer_steps} steps for "
            f"{phase_key} without satisfying the actual convergence criteria."
        )
    session.optimizer_steps += optimizer_steps
    return (
        status,
        True,
        optimizer_steps,
        tuple(history),
        tuple(phase_warnings),
    )


def _execute_phase_to_staging(
    context: Step6Context,
    phase_key: str,
    session: CalculatorSession,
    staging_root: Path,
) -> tuple[PhaseResult, Mapping[Path, Path]]:
    """Execute one independent original structure into temporary files."""

    if session.configuration_fingerprint != context.config_fingerprint:
        raise Step6CalculatorError(
            "Calculator session configuration fingerprint does not match context."
        )
    phase_input = context.phase_inputs[phase_key]
    outputs = phase_output_paths(context, phase_key)
    staged_by_final = {
        target: _stage_path(staging_root, context.output_root, target)
        for target in outputs.all_paths()
    }
    staged_named = {
        "final_structure": staged_by_final[outputs.structure],
        "trajectory": staged_by_final[outputs.trajectory],
        "history_csv": staged_by_final[outputs.history_csv],
        "report": staged_by_final[outputs.report],
        "optimizer_log": staged_by_final[outputs.log],
        "result_json": staged_by_final[outputs.result_json],
    }

    source_atoms = phase_input.structure.atoms
    if source_atoms.calc is not None:
        raise Step6InputError(
            f"Pristine source {phase_key} unexpectedly has a calculator."
        )
    working = source_atoms.copy()
    working.calc = None
    initial_geometry = _capture_initial_geometry(working)
    started_at = utc_timestamp()
    started_monotonic = time.monotonic()
    state_count_before = session.state_evaluations
    result: PhaseResult | None = None
    try:
        working.calc = session.calculator
        initial_state = _evaluate_state(
            working,
            initial_geometry,
            context,
            0,
            0.0,
            session,
        )
        (
            status,
            optimizer_created,
            optimizer_steps,
            history,
            calculation_warnings,
        ) = _run_relaxation(
            context,
            phase_key,
            working,
            initial_geometry,
            initial_state,
            session,
            staged_named["trajectory"],
            staged_named["optimizer_log"],
            started_monotonic,
        )
        final_state = history[-1]
        if status in {"ALREADY_CONVERGED", "CONVERGED"}:
            if not final_state.overall_converged:
                raise Step6CalculationError(
                    f"{phase_key} was labeled {status} without exact convergence."
                )
        elif final_state.overall_converged:
            raise Step6CalculationError(
                f"{phase_key} was labeled NOT_CONVERGED despite exact convergence."
            )
        completed_at = utc_timestamp()
        result = PhaseResult(
            phase_key=phase_key,
            material_id=phase_input.material_id,
            mode=context.mode,
            status=status,
            safety_status="PASS",
            optimizer_created=optimizer_created,
            optimizer_steps=optimizer_steps,
            state_evaluations=session.state_evaluations - state_count_before,
            calculator_class=session.calculator_class,
            calculator_load_count=session.load_count,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            wall_time_seconds=time.monotonic() - started_monotonic,
            initial=initial_state,
            final=final_state,
            history=history,
            warnings=tuple(session.warnings) + calculation_warnings,
            output_paths=outputs,
            source_snapshots=context.protected_snapshots,
            configuration_fingerprint=context.config_fingerprint,
        )
    finally:
        working.calc = None
        try:
            verify_protected_files(context)
        finally:
            _reset_calculator(session, phase_key)

    if result is None:
        raise Step6CalculationError(f"No result was produced for {phase_key}.")
    if source_atoms.calc is not None or working.calc is not None:
        raise Step6SafetyError(
            f"Calculator detachment failed for {phase_key}."
        )

    _write_final_structure(
        context, result, working, staged_named["final_structure"]
    )
    staged_named["history_csv"].write_bytes(_history_csv_bytes(result.history))
    staged_named["report"].write_text(
        _phase_report_text(context, result), encoding="utf-8", newline="\n"
    )
    manifest = _manifest_document(context, result, staged_named)
    staged_named["result_json"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, path in staged_named.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise Step6PublicationError(
                f"Staged {name} for {phase_key} is absent or empty: {path}"
            )
    # Parse staged machine-readable artifacts before publication.
    read_json_object(staged_named["result_json"], f"staged {phase_key} manifest")
    try:
        with staged_named["history_csv"].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Step6PublicationError(
            f"Could not parse staged history for {phase_key}: {exc}"
        ) from exc
    if not rows or int(rows[0]["step"]) != 0 or int(rows[-1]["step"]) != result.optimizer_steps:
        raise Step6PublicationError(
            f"Staged history endpoints are invalid for {phase_key}."
        )
    _validate_staged_phase_bundle(context, result, staged_named)
    return result, staged_by_final


def _snapshot_from_manifest(
    raw_value: Any, project_root: Path, label: str
) -> FileSnapshot:
    raw = _require_mapping(raw_value, label)
    path_value = raw.get("path")
    sha256 = raw.get("sha256")
    size = raw.get("size_bytes")
    mtime = raw.get("modification_time_ns")
    if not isinstance(path_value, str) or not path_value:
        raise Step6ResumeError(f"{label}.path must be a nonempty string.")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise Step6ResumeError(f"{label}.sha256 is invalid.")
    size_int = _require_int(size, f"{label}.size_bytes")
    mtime_int = _require_int(mtime, f"{label}.modification_time_ns")
    path = (project_root / Path(path_value)).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step6ResumeError(f"{label}.path escapes the repository.") from exc
    return FileSnapshot(
        label=str(raw.get("label", label)),
        path=path,
        sha256=sha256,
        size=size_int,
        modification_time_ns=mtime_int,
    )


def _validate_artifact_record(
    raw_value: Any,
    expected_path: Path,
    context: Step6Context,
    label: str,
) -> None:
    raw = _require_mapping(raw_value, label)
    expected_relative = relative_path(expected_path, context.project_root)
    if raw.get("path") != expected_relative:
        raise Step6ResumeError(
            f"{label} path mismatch: expected {expected_relative!r}."
        )
    sha = raw.get("sha256")
    size = raw.get("size_bytes")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise Step6ResumeError(f"{label} hash/size record is invalid.")
    if not expected_path.is_file():
        raise Step6ResumeError(f"{label} file is missing: {expected_path}")
    stat = expected_path.stat()
    if stat.st_size != size or file_sha256(expected_path) != sha:
        raise Step6ResumeError(f"{label} file fingerprint does not match manifest.")


def _compare_geometry(
    first: Any,
    second: Any,
    *,
    label: str,
    tolerance: float = 1.0e-12,
) -> None:
    np = _numpy()
    if len(first) != len(second):
        raise Step6ResumeError(f"{label}: atom counts differ.")
    if tuple(first.get_chemical_symbols()) != tuple(second.get_chemical_symbols()):
        raise Step6ResumeError(f"{label}: atom identities or ordering differ.")
    if not bool(
        np.allclose(
            first.get_positions(),
            second.get_positions(),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        raise Step6ResumeError(f"{label}: positions differ.")
    if not bool(
        np.allclose(
            first.cell.array,
            second.cell.array,
            atol=tolerance,
            rtol=0.0,
        )
    ):
        raise Step6ResumeError(f"{label}: cells differ.")
    if not bool(np.array_equal(first.get_pbc(), second.get_pbc())):
        raise Step6ResumeError(f"{label}: PBC differs.")


def validate_phase_bundle(context: Step6Context, phase: str) -> PhaseResult:
    """Validate and reconstruct one complete phase bundle without MACE."""

    if phase not in context.phase_inputs:
        raise Step6ResumeError(
            f"Phase {phase!r} was not validated in this context."
        )
    outputs = phase_output_paths(context, phase)
    document = read_json_object(outputs.result_json, f"{phase} resume manifest")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Step6ResumeError(f"{phase} manifest schema version is unsupported.")
    if document.get("artifact_type") != "ni_al_mace_relaxation_phase_result":
        raise Step6ResumeError(f"{phase} manifest artifact_type is invalid.")
    if document.get("phase_key") != phase or document.get("mode") != context.mode:
        raise Step6ResumeError(f"{phase} manifest identity or mode mismatch.")
    if document.get("material_id") != EXPECTED_MATERIAL_IDS[phase]:
        raise Step6ResumeError(f"{phase} manifest material ID mismatch.")
    if document.get("number_of_atoms") != EXPECTED_ATOM_COUNTS[phase]:
        raise Step6ResumeError(f"{phase} manifest atom count mismatch.")
    if document.get("configuration_fingerprint_sha256") != (
        context.config_fingerprint
    ):
        raise Step6ResumeError(
            f"{phase} manifest configuration fingerprint does not match."
        )
    if (
        document.get("execution_status") != "COMPLETED"
        or document.get("safety_status") != "PASS"
        or document.get("output_validation_status") != "PASS"
    ):
        raise Step6ResumeError(
            f"{phase} manifest is not a completed, safe, validated result."
        )
    status = document.get("convergence_status")
    if status not in {"ALREADY_CONVERGED", "CONVERGED", "NOT_CONVERGED"}:
        raise Step6ResumeError(f"{phase} convergence status is invalid.")

    model = _require_mapping(document.get("model"), f"{phase}.model")
    configured_model = context.configuration.model
    expected_model = {
        "family": configured_model.family,
        "name": configured_model.name,
        "value": configured_model.value,
        "device": configured_model.device,
        "default_dtype": configured_model.default_dtype,
        "dispersion": configured_model.dispersion,
    }
    for key, value in expected_model.items():
        if model.get(key) != value:
            raise Step6ResumeError(f"{phase} model setting {key!r} mismatches.")
    calculator_class = model.get("calculator_class")
    if not isinstance(calculator_class, str) or not calculator_class:
        raise Step6ResumeError(f"{phase} calculator class is invalid.")

    optimizer = _require_mapping(document.get("optimizer"), f"{phase}.optimizer")
    if optimizer.get("name") != "FIRE":
        raise Step6ResumeError(f"{phase} optimizer is not FIRE.")
    optimizer_created = _require_bool(
        optimizer.get("created"), f"{phase}.optimizer.created"
    )
    optimizer_steps = _require_int(
        optimizer.get("steps"), f"{phase}.optimizer.steps"
    )
    if optimizer_steps > context.mode_settings.maximum_steps:
        raise Step6ResumeError(f"{phase} optimizer step count exceeds its maximum.")
    if optimizer.get("maximum_steps") != context.mode_settings.maximum_steps:
        raise Step6ResumeError(f"{phase} maximum optimizer steps mismatch.")
    if not math.isclose(
        _finite_float(
            optimizer.get("force_threshold_eV_per_A"),
            f"{phase}.force_threshold",
        ),
        context.mode_settings.force_threshold_eV_per_A,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise Step6ResumeError(f"{phase} force threshold mismatch.")
    if optimizer.get("trajectory_interval") != (
        context.mode_settings.trajectory_interval
    ) or optimizer.get("history_interval") != context.execution.history_interval:
        raise Step6ResumeError(f"{phase} history/trajectory intervals mismatch.")
    if context.mode == FULL_CELL_MODE:
        if optimizer.get("stress_threshold_eV_per_A3") != (
            context.mode_settings.stress_threshold_eV_per_A3
        ):
            raise Step6ResumeError(f"{phase} stress threshold mismatch.")
        if optimizer.get("ase_generalized_auto_stop_fmax_eV_per_A") != 0.0:
            raise Step6ResumeError(
                f"{phase} full-cell generalized auto-stop sentinel mismatch."
            )
        filter_record = _require_mapping(
            optimizer.get("filter"), f"{phase}.optimizer.filter"
        )
        if (
            filter_record.get("name") != "FrechetCellFilter"
            or filter_record.get("mask") != "all"
            or filter_record.get("exp_cell_factor")
            != float(EXPECTED_ATOM_COUNTS[phase])
            or filter_record.get("hydrostatic_strain") is not False
            or filter_record.get("constant_volume") is not False
            or filter_record.get("external_pressure_eV_per_A3") != 0.0
        ):
            raise Step6ResumeError(f"{phase} FrechetCellFilter settings mismatch.")
    elif optimizer.get("filter") is not None:
        raise Step6ResumeError(f"{phase} atomic-only manifest has a cell filter.")

    initial = _state_from_json(
        document.get("initial"), f"{phase}.initial", EXPECTED_ATOM_COUNTS[phase]
    )
    final = _state_from_json(
        document.get("final"), f"{phase}.final", EXPECTED_ATOM_COUNTS[phase]
    )
    history_raw = document.get("history")
    if not isinstance(history_raw, list) or not history_raw:
        raise Step6ResumeError(f"{phase} history must be a nonempty array.")
    history = tuple(
        _state_from_json(
            value,
            f"{phase}.history[{index}]",
            EXPECTED_ATOM_COUNTS[phase],
        )
        for index, value in enumerate(history_raw)
    )
    _validate_state_consistency(
        initial,
        context,
        EXPECTED_ATOM_COUNTS[phase],
        initial.volume_A3,
        f"{phase}.initial",
    )
    for index, state in enumerate(history):
        _validate_state_consistency(
            state,
            context,
            EXPECTED_ATOM_COUNTS[phase],
            initial.volume_A3,
            f"{phase}.history[{index}]",
        )
    _validate_state_consistency(
        final,
        context,
        EXPECTED_ATOM_COUNTS[phase],
        initial.volume_A3,
        f"{phase}.final",
    )
    if history[0].step != 0 or history[-1].step != optimizer_steps:
        raise Step6ResumeError(f"{phase} manifest history endpoints are invalid.")
    if any(
        current.step <= previous.step
        or current.elapsed_seconds < previous.elapsed_seconds
        for previous, current in zip(history, history[1:])
    ):
        raise Step6ResumeError(f"{phase} history steps/times are not increasing.")
    if history[0] != initial or history[-1] != final:
        raise Step6ResumeError(f"{phase} initial/final history records disagree.")
    force_converged = _require_bool(
        document.get("force_converged"), f"{phase}.force_converged"
    )
    overall_converged = _require_bool(
        document.get("overall_converged"), f"{phase}.overall_converged"
    )
    stress_converged_raw = document.get("stress_converged")
    if context.mode == ATOMIC_ONLY_MODE:
        if stress_converged_raw is not None:
            raise Step6ResumeError(
                f"{phase} atomic-only stress convergence must be null."
            )
    elif not isinstance(stress_converged_raw, bool):
        raise Step6ResumeError(
            f"{phase} full-cell stress convergence must be boolean."
        )
    if (
        force_converged != final.force_converged
        or overall_converged != final.overall_converged
        or stress_converged_raw != final.stress_converged
    ):
        raise Step6ResumeError(f"{phase} convergence fields are inconsistent.")
    if status in {"ALREADY_CONVERGED", "CONVERGED"} and not overall_converged:
        raise Step6ResumeError(f"{phase} converged label contradicts metrics.")
    if status == "NOT_CONVERGED" and overall_converged:
        raise Step6ResumeError(f"{phase} NOT_CONVERGED label contradicts metrics.")
    if status == "ALREADY_CONVERGED" and (
        optimizer_created or optimizer_steps != 0
    ):
        raise Step6ResumeError(
            f"{phase} ALREADY_CONVERGED optimizer accounting is invalid."
        )
    if status != "ALREADY_CONVERGED" and (
        not optimizer_created or optimizer_steps <= 0
    ):
        raise Step6ResumeError(f"{phase} optimizer accounting is inconsistent.")

    artifacts = _require_mapping(document.get("artifacts"), f"{phase}.artifacts")
    for name, path in (
        ("final_structure", outputs.structure),
        ("trajectory", outputs.trajectory),
        ("history_csv", outputs.history_csv),
        ("report", outputs.report),
        ("optimizer_log", outputs.log),
    ):
        _validate_artifact_record(
            artifacts.get(name), path, context, f"{phase}.artifacts.{name}"
        )

    protected_raw = document.get("protected_sources")
    if not isinstance(protected_raw, list) or not protected_raw:
        raise Step6ResumeError(f"{phase} protected source records are absent.")
    manifest_snapshots = {
        snapshot.path: snapshot
        for snapshot in (
            _snapshot_from_manifest(
                value, context.project_root, f"{phase}.protected_sources[{index}]"
            )
            for index, value in enumerate(protected_raw)
        )
    }
    for expected in context.protected_snapshots:
        recorded = manifest_snapshots.get(expected.path)
        if recorded is None or (
            recorded.sha256,
            recorded.size,
            recorded.modification_time_ns,
        ) != (
            expected.sha256,
            expected.size,
            expected.modification_time_ns,
        ):
            raise Step6ResumeError(
                f"{phase} protected provenance is absent or mismatched for "
                f"{expected.path.name}."
            )
        verify_file_snapshot(recorded)

    try:
        from ase.io import read as ase_read
        from ase.io.trajectory import Trajectory
        final_frames = ase_read(outputs.structure, index=":", format="extxyz")
    except Exception as exc:
        raise Step6ResumeError(
            f"Could not read {phase} final structure: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(final_frames, list) or len(final_frames) != 1:
        raise Step6ResumeError(f"{phase} final EXTXYZ must contain one frame.")
    final_atoms = final_frames[0]
    final_atoms.calc = None
    source_atoms = context.phase_inputs[phase].structure.atoms
    if len(final_atoms) != EXPECTED_ATOM_COUNTS[phase]:
        raise Step6ResumeError(f"{phase} final structure atom count is invalid.")
    if tuple(final_atoms.get_chemical_symbols()) != tuple(
        source_atoms.get_chemical_symbols()
    ):
        raise Step6ResumeError(f"{phase} final structure ordering is invalid.")
    np = _numpy()
    if not bool(
        np.allclose(
            final_atoms.get_positions(),
            np.asarray(final.positions_A),
            atol=1.0e-12,
            rtol=0.0,
        )
    ) or not bool(
        np.allclose(
            final_atoms.cell.array,
            np.asarray(final.cell_A),
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise Step6ResumeError(
            f"{phase} final EXTXYZ geometry disagrees with its manifest."
        )
    if context.mode == ATOMIC_ONLY_MODE and not bool(
        np.allclose(
            final_atoms.cell.array,
            source_atoms.cell.array,
            atol=ATOMIC_CELL_ATOL_A,
            rtol=0.0,
        )
    ):
        raise Step6ResumeError(f"{phase} atomic-only final cell changed.")

    try:
        with Trajectory(outputs.trajectory, mode="r") as trajectory:
            frames = [frame for frame in trajectory]
    except Exception as exc:
        raise Step6ResumeError(
            f"Could not read {phase} trajectory: {type(exc).__name__}: {exc}"
        ) from exc
    if not frames:
        raise Step6ResumeError(f"{phase} trajectory has no frames.")
    _compare_geometry(frames[0], source_atoms, label=f"{phase} trajectory start")
    _compare_geometry(frames[-1], final_atoms, label=f"{phase} trajectory end")

    try:
        with outputs.history_csv.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            csv_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Step6ResumeError(f"Could not parse {phase} history CSV: {exc}") from exc
    if (
        not csv_rows
        or tuple(csv_rows[0].keys()) != _history_fieldnames()
        or int(csv_rows[0]["step"]) != 0
        or int(csv_rows[-1]["step"]) != optimizer_steps
    ):
        raise Step6ResumeError(f"{phase} history CSV endpoints/schema are invalid.")
    if not math.isclose(
        float(csv_rows[-1]["total_energy_eV"]),
        final.total_energy_eV,
        abs_tol=1.0e-12,
        rel_tol=1.0e-12,
    ):
        raise Step6ResumeError(f"{phase} history CSV final energy mismatches.")
    report_text = outputs.report.read_text(encoding="utf-8")
    for sentinel in (
        f"Phase: {phase}",
        f"Materials Project ID: {EXPECTED_MATERIAL_IDS[phase]}",
        f"Configuration SHA-256: {context.config_fingerprint}",
        f"Phase status: {status}",
        "Safety status: PASS",
    ):
        if sentinel not in report_text:
            raise Step6ResumeError(
                f"{phase} report is missing sentinel: {sentinel!r}"
            )

    counts = _require_mapping(document.get("counts"), f"{phase}.counts")
    timing = _require_mapping(document.get("timing"), f"{phase}.timing")
    warnings_raw = document.get("warnings")
    if not isinstance(warnings_raw, list) or not all(
        isinstance(value, str) for value in warnings_raw
    ):
        raise Step6ResumeError(f"{phase} warnings must be a string array.")
    result = PhaseResult(
        phase_key=phase,
        material_id=EXPECTED_MATERIAL_IDS[phase],
        mode=context.mode,
        status=str(status),
        safety_status="PASS",
        optimizer_created=optimizer_created,
        optimizer_steps=optimizer_steps,
        state_evaluations=_require_int(
            counts.get("state_evaluations"), f"{phase}.state_evaluations", 1
        ),
        calculator_class=calculator_class,
        calculator_load_count=_require_int(
            counts.get("calculator_loads_in_session"),
            f"{phase}.calculator_loads",
            1,
        ),
        started_at_utc=str(timing.get("started_at_utc")),
        completed_at_utc=str(timing.get("completed_at_utc")),
        wall_time_seconds=_finite_float(
            timing.get("wall_time_seconds"), f"{phase}.wall_time_seconds"
        ),
        initial=initial,
        final=final,
        history=history,
        warnings=tuple(warnings_raw),
        output_paths=outputs,
        source_snapshots=tuple(manifest_snapshots.values()),
        configuration_fingerprint=context.config_fingerprint,
        resumed=True,
    )
    verify_protected_files(context)
    return result


def _summary_record(result: PhaseResult) -> dict[str, Any]:
    initial = result.initial
    final = result.final
    return {
        "phase_key": result.phase_key,
        "material_id": result.material_id,
        "number_of_atoms": EXPECTED_ATOM_COUNTS[result.phase_key],
        "status": result.status,
        "safety_status": result.safety_status,
        "force_converged": result.force_converged,
        "stress_converged": result.stress_converged,
        "overall_converged": result.overall_converged,
        "optimizer_steps": result.optimizer_steps,
        "state_evaluations": result.state_evaluations,
        "wall_time_seconds": result.wall_time_seconds,
        "initial_total_energy_eV": initial.total_energy_eV,
        "final_total_energy_eV": final.total_energy_eV,
        "energy_change_eV": final.total_energy_eV - initial.total_energy_eV,
        "initial_energy_per_atom_eV": initial.energy_per_atom_eV,
        "final_energy_per_atom_eV": final.energy_per_atom_eV,
        "initial_maximum_force_eV_per_A": initial.maximum_force_eV_per_A,
        "final_maximum_force_eV_per_A": final.maximum_force_eV_per_A,
        "initial_rms_force_eV_per_A": initial.rms_force_eV_per_A,
        "final_rms_force_eV_per_A": final.rms_force_eV_per_A,
        "initial_total_force_norm_eV_per_A": initial.total_force_norm_eV_per_A,
        "final_total_force_norm_eV_per_A": final.total_force_norm_eV_per_A,
        "initial_stress_eV_per_A3": list(initial.stress_eV_per_A3),
        "final_stress_eV_per_A3": list(final.stress_eV_per_A3),
        "initial_maximum_absolute_stress_eV_per_A3": (
            initial.maximum_absolute_stress_eV_per_A3
        ),
        "final_maximum_absolute_stress_eV_per_A3": (
            final.maximum_absolute_stress_eV_per_A3
        ),
        "initial_volume_A3": initial.volume_A3,
        "final_volume_A3": final.volume_A3,
        "volume_change_A3": final.volume_A3 - initial.volume_A3,
        "volume_change_percent": final.volume_change_percent,
        "initial_lattice_lengths_A": list(initial.lattice_lengths_A),
        "final_lattice_lengths_A": list(final.lattice_lengths_A),
        "initial_lattice_angles_deg": list(initial.lattice_angles_deg),
        "final_lattice_angles_deg": list(final.lattice_angles_deg),
        "maximum_internal_displacement_A": (
            final.displacement.maximum_internal_A
        ),
        "rms_internal_displacement_A": final.displacement.rms_internal_A,
        "maximum_total_displacement_A": final.displacement.maximum_total_A,
        "rms_total_displacement_A": final.displacement.rms_total_A,
        "resumed": result.resumed,
        "result_manifest": relative_path(
            result.output_paths.result_json,
            result.output_paths.result_json.parents[4],
        ),
    }


def _combined_csv_bytes(results: Sequence[PhaseResult]) -> bytes:
    fieldnames = (
        "phase_key",
        "material_id",
        "number_of_atoms",
        "status",
        "safety_status",
        "force_converged",
        "stress_converged",
        "overall_converged",
        "optimizer_steps",
        "state_evaluations",
        "wall_time_seconds",
        "initial_total_energy_eV",
        "final_total_energy_eV",
        "energy_change_eV",
        "initial_maximum_force_eV_per_A",
        "final_maximum_force_eV_per_A",
        "initial_rms_force_eV_per_A",
        "final_rms_force_eV_per_A",
        "initial_maximum_absolute_stress_eV_per_A3",
        "final_maximum_absolute_stress_eV_per_A3",
        "initial_volume_A3",
        "final_volume_A3",
        "volume_change_A3",
        "volume_change_percent",
        "maximum_internal_displacement_A",
        "rms_internal_displacement_A",
        "maximum_total_displacement_A",
        "rms_total_displacement_A",
        "resumed",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        complete = _summary_record(result)
        writer.writerow({field: complete[field] for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def _combined_status(results: Sequence[PhaseResult]) -> str:
    if not results:
        return "FAILED"
    if all(result.overall_converged for result in results):
        return "SUCCESS"
    return "PARTIAL"


def build_combined_summary(
    context: Step6Context,
    results: Sequence[PhaseResult],
    calculator_session: CalculatorSession | None = None,
) -> tuple[ModeSummary, Mapping[Path, bytes]]:
    """Build combined CSV/JSON/text content without publishing it."""

    result_by_phase = {result.phase_key: result for result in results}
    ordered = tuple(
        result_by_phase[phase]
        for phase in PHASE_ORDER
        if phase in result_by_phase
    )
    if len(ordered) != len(results):
        raise Step6PublicationError("Combined results contain duplicate phases.")
    requested = tuple(result.phase_key for result in ordered)
    resumed = tuple(result.phase_key for result in ordered if result.resumed)
    executed = tuple(result.phase_key for result in ordered if not result.resumed)
    calculator_class = (
        calculator_session.calculator_class
        if calculator_session is not None
        else (
            ordered[0].calculator_class if ordered else "not loaded"
        )
    )
    calculator_loads = (
        calculator_session.load_count if calculator_session is not None else 0
    )
    status = _combined_status(ordered)
    paths = combined_output_paths(context)
    summary = ModeSummary(
        mode=context.mode,
        requested_phases=requested,
        results=ordered,
        calculator_loads=calculator_loads,
        calculator_class=calculator_class,
        state_evaluations=sum(result.state_evaluations for result in ordered),
        optimizer_steps=sum(result.optimizer_steps for result in ordered),
        resumed_phases=resumed,
        executed_phases=executed,
        overall_status=status,
        combined_outputs=paths,
    )
    records = [_summary_record(result) for result in ordered]
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"ni_al_{context.mode}_relaxation_summary",
        "project_step": "6C" if context.mode == ATOMIC_ONLY_MODE else "6D",
        "generated_at_utc": utc_timestamp(),
        "mode": context.mode,
        "configuration_path": relative_path(
            context.config_path, context.project_root
        ),
        "configuration_fingerprint_sha256": context.config_fingerprint,
        "requested_phases": list(requested),
        "completed_phases": list(requested),
        "failed_phases": [],
        "converged_phases": [
            result.phase_key for result in ordered if result.overall_converged
        ],
        "not_converged_phases": [
            result.phase_key for result in ordered if not result.overall_converged
        ],
        "overall_status": status,
        "model": {
            "family": context.configuration.model.family,
            "name": context.configuration.model.name,
            "value": context.configuration.model.value,
            "device": context.configuration.model.device,
            "default_dtype": context.configuration.model.default_dtype,
            "dispersion": context.configuration.model.dispersion,
            "calculator_class": calculator_class,
        },
        "execution": {
            "calculator_loads_this_invocation": calculator_loads,
            "calculator_reused_sequentially": bool(executed),
            "state_evaluations": summary.state_evaluations,
            "optimizer_steps": summary.optimizer_steps,
            "resumed_phases": list(resumed),
            "executed_phases": list(executed),
            "optimizer": "FIRE",
            "filter": (
                None
                if context.mode == ATOMIC_ONLY_MODE
                else "FrechetCellFilter"
            ),
        },
        "convergence": {
            "force_threshold_eV_per_A": (
                context.mode_settings.force_threshold_eV_per_A
            ),
            "stress_threshold_eV_per_A3": (
                context.mode_settings.stress_threshold_eV_per_A3
            ),
            "requires_actual_atomic_force": True,
            "requires_actual_six_component_stress": (
                context.mode == FULL_CELL_MODE
            ),
        },
        "stress_component_order": list(STRESS_COMPONENTS),
        "records": records,
        "scientific_limitations": [
            "MACE-potential relaxation; not DFT or experimental validation.",
            "No raw-energy phase-stability ranking across compositions.",
            "No formation energies calculated.",
        ],
    }
    title = (
        "Step 6C - Ni-Al Atomic-Only Relaxation Summary"
        if context.mode == ATOMIC_ONLY_MODE
        else "Step 6D - Ni-Al Full-Cell Relaxation Summary"
    )
    lines = [
        title,
        "=" * len(title),
        "",
        f"Generated (UTC): {document['generated_at_utc']}",
        f"Configuration SHA-256: {context.config_fingerprint}",
        f"Requested phases: {', '.join(requested)}",
        f"Overall status: {status}",
        f"Calculator class: {calculator_class}",
        f"Calculator loads this invocation: {calculator_loads}",
        f"Executed phases: {', '.join(executed) if executed else 'None'}",
        f"Resumed phases: {', '.join(resumed) if resumed else 'None'}",
        f"Total optimizer steps: {summary.optimizer_steps}",
        "",
        "Per-phase results",
        "-----------------",
    ]
    for result in ordered:
        lines.extend(
            [
                f"{result.phase_key}: {result.status}",
                f"  steps={result.optimizer_steps}",
                f"  force_converged={result.force_converged}",
                f"  stress_converged={result.stress_converged}",
                f"  initial_energy_eV={result.initial.total_energy_eV:.17g}",
                f"  final_energy_eV={result.final.total_energy_eV:.17g}",
                f"  final_max_force_eV_per_A={result.final.maximum_force_eV_per_A:.17g}",
                f"  final_max_abs_stress_eV_per_A3="
                f"{result.final.maximum_absolute_stress_eV_per_A3:.17g}",
                f"  volume_change_percent={result.final.volume_change_percent:.17g}",
                f"  maximum_internal_displacement_A="
                f"{result.final.displacement.maximum_internal_A:.17g}",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation boundary",
            "-----------------------",
            (
                "These are MACE-potential relaxation results, not DFT or "
                "experimental validation. Raw energies are not compared across "
                "compositions as a stability ranking. No formation energies "
                "were calculated."
            ),
            "",
        ]
    )
    contents: Mapping[Path, bytes] = {
        paths.csv: _combined_csv_bytes(ordered),
        paths.json: (
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
        paths.report: "\n".join(lines).encode("utf-8"),
    }
    return summary, contents


def _validate_targets_in_output_root(
    context: Step6Context, targets: Iterable[Path]
) -> tuple[Path, ...]:
    validated: list[Path] = []
    root = context.output_root.resolve()
    for target in targets:
        resolved = target.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise Step6PublicationError(
                f"Publication target escapes the controlled mode root: {resolved}"
            ) from exc
        if resolved in validated:
            raise Step6PublicationError(
                f"Duplicate publication target: {resolved}"
            )
        validated.append(resolved)
    return tuple(validated)


def _publish_files_transactionally(
    context: Step6Context,
    staged_by_final: Mapping[Path, Path],
    *,
    overwrite: bool,
    final_validator: Callable[[], None] | None = None,
) -> None:
    """Atomically publish a multi-file bundle and roll back any failure."""

    if not staged_by_final:
        return
    targets = _validate_targets_in_output_root(context, staged_by_final.keys())
    normalized: dict[Path, Path] = {
        final.resolve(): staged.resolve()
        for final, staged in staged_by_final.items()
    }
    collisions: list[Path] = []
    for target in targets:
        if target.exists():
            if target.is_dir():
                raise Step6CollisionError(
                    f"Output target is a directory, not a file: {target}"
                )
            if not overwrite:
                collisions.append(target)
        stage = normalized[target]
        if not stage.is_file() or stage.stat().st_size <= 0:
            raise Step6PublicationError(
                f"Staged publication input is absent or empty: {stage}"
            )
    if collisions:
        details = "\n".join(
            f"  - {relative_path(path, context.project_root)}"
            for path in collisions
        )
        raise Step6CollisionError(
            "Refusing to overwrite existing Step 6 output(s):\n" + details
        )
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    backup_root = Path(
        tempfile.mkdtemp(prefix=".step6-publication-backup-", dir=context.output_root)
    )
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for index, target in enumerate(targets):
            staged = normalized[target]
            if overwrite and target.exists():
                backup = backup_root / f"{index:04d}.backup"
                os.replace(target, backup)
                backups[target] = backup
            if overwrite:
                os.replace(staged, target)
            else:
                # A same-volume hard link gives atomic create-without-replace
                # semantics and closes the race after the collision precheck.
                os.link(staged, target)
            published.append(target)
        if final_validator is not None:
            final_validator()
    except Exception as exc:
        rollback_errors: list[str] = []
        for target in reversed(published):
            try:
                if target.is_file() or target.is_symlink():
                    target.unlink()
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"remove {target}: {type(rollback_exc).__name__}: "
                    f"{rollback_exc}"
                )
        for target, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(backup, target)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"restore {target}: {type(rollback_exc).__name__}: "
                    f"{rollback_exc}"
                )
        detail = (
            f"{type(exc).__name__}: {exc}"
            + (
                "; rollback errors: " + "; ".join(rollback_errors)
                if rollback_errors
                else ""
            )
        )
        raise Step6PublicationError(
            f"Transactional Step 6 publication failed: {detail}"
        ) from exc
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _validate_combined_outputs(
    context: Step6Context, expected_results: Sequence[PhaseResult]
) -> None:
    """Validate combined mode CSV/JSON/text against phase manifests."""

    paths = combined_output_paths(context)
    for path in paths.all_paths():
        if not path.is_file() or path.stat().st_size <= 0:
            raise Step6ResumeError(f"Combined output is absent or empty: {path}")
    document = read_json_object(paths.json, f"{context.mode} combined summary")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("artifact_type")
        != f"ni_al_{context.mode}_relaxation_summary"
        or document.get("mode") != context.mode
        or document.get("configuration_fingerprint_sha256")
        != context.config_fingerprint
    ):
        raise Step6ResumeError("Combined summary identity/configuration mismatch.")
    expected_phases = [result.phase_key for result in expected_results]
    if document.get("requested_phases") != expected_phases:
        raise Step6ResumeError("Combined summary phase scope mismatch.")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != len(expected_results):
        raise Step6ResumeError("Combined summary records are incomplete.")
    for expected, raw in zip(expected_results, records):
        if not isinstance(raw, Mapping):
            raise Step6ResumeError("Combined summary record is not an object.")
        if (
            raw.get("phase_key") != expected.phase_key
            or raw.get("material_id") != expected.material_id
            or raw.get("status") != expected.status
            or raw.get("overall_converged") != expected.overall_converged
            or raw.get("optimizer_steps") != expected.optimizer_steps
        ):
            raise Step6ResumeError(
                f"Combined summary disagrees with {expected.phase_key} manifest."
            )
    try:
        with paths.csv.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Step6ResumeError(f"Combined CSV could not be parsed: {exc}") from exc
    if [row.get("phase_key") for row in csv_rows] != expected_phases:
        raise Step6ResumeError("Combined CSV phase order/scope mismatch.")
    report = paths.report.read_text(encoding="utf-8")
    if (
        f"Configuration SHA-256: {context.config_fingerprint}" not in report
        or f"Requested phases: {', '.join(expected_phases)}" not in report
    ):
        raise Step6ResumeError("Combined text report provenance is incomplete.")


def publish_combined_summary(
    context: Step6Context,
    results: Sequence[PhaseResult],
    calculator_session: CalculatorSession | None = None,
    overwrite: bool = False,
) -> ModeSummary:
    """Build, transactionally publish, and validate combined mode outputs."""

    summary, contents = build_combined_summary(
        context, results, calculator_session
    )
    for directory in planned_mode_directories(context):
        directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".step6-summary-staging-", dir=context.output_root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged: dict[Path, Path] = {}
        for target, payload in contents.items():
            stage = _stage_path(staging_root, context.output_root, target)
            stage.write_bytes(payload)
            staged[target] = stage

        def final_validator() -> None:
            verify_protected_files(context)
            _validate_combined_outputs(context, summary.results)

        _publish_files_transactionally(
            context,
            staged,
            overwrite=overwrite,
            final_validator=final_validator,
        )
    return summary


def validate_all_mode_outputs(context: Step6Context) -> tuple[PhaseResult, ...]:
    """Validate all context-selected phase bundles and their combined outputs."""

    results = tuple(
        validate_phase_bundle(context, phase) for phase in context.phase_keys
    )
    _validate_combined_outputs(context, results)
    verify_protected_files(context)
    return results


def _existing_phase_targets(paths: ModeOutputPaths) -> tuple[Path, ...]:
    return tuple(path for path in paths.all_paths() if path.exists())


def execute_mode(
    context: Step6Context,
    phase_keys: Sequence[str] | None = None,
    overwrite: bool = False,
    resume: bool = False,
    calculator_session: CalculatorSession | None = None,
    publish_summary: bool = True,
) -> ModeSummary:
    """Execute independent phase relaxations with one reusable calculator.

    All new phase files and optional combined summaries are staged before one
    transactional publication. Existing valid converged bundles may be reused
    only when ``resume`` is explicit.
    """

    selected = (
        context.phase_keys
        if phase_keys is None
        else _validate_phase_keys(phase_keys)
    )
    missing_context = [phase for phase in selected if phase not in context.phase_inputs]
    if missing_context:
        raise Step6ConfigurationError(
            "Execution requested phases absent from validated context: "
            + ", ".join(missing_context)
        )
    for directory in planned_mode_directories(context):
        directory.mkdir(parents=True, exist_ok=True)
    verify_protected_files(context)

    resumed_results: dict[str, PhaseResult] = {}
    phases_to_execute: list[str] = []
    collision_paths: list[Path] = []
    for phase in selected:
        paths = phase_output_paths(context, phase)
        existing = _existing_phase_targets(paths)
        if existing and resume:
            try:
                resumed_result = validate_phase_bundle(context, phase)
                if not resumed_result.overall_converged:
                    raise Step6ResumeError(
                        f"{phase} is a valid diagnostic result but is "
                        "NOT_CONVERGED and is not resume-eligible."
                    )
                resumed_results[phase] = resumed_result
                continue
            except Step6Error:
                if not overwrite:
                    raise
        if existing and not overwrite:
            collision_paths.extend(existing)
        phases_to_execute.append(phase)
    summary_paths = combined_output_paths(context)
    existing_summary = (
        tuple(path for path in summary_paths.all_paths() if path.exists())
        if publish_summary
        else ()
    )
    if collision_paths:
        detail = "\n".join(
            f"  - {relative_path(path, context.project_root)}"
            for path in sorted(set(collision_paths))
        )
        raise Step6CollisionError(
            "Existing phase output collisions were found:\n" + detail
        )

    if (
        publish_summary
        and existing_summary
        and not overwrite
        and not phases_to_execute
    ):
        ordered_resumed = tuple(resumed_results[phase] for phase in selected)
        _validate_combined_outputs(context, ordered_resumed)
        summary, _contents = build_combined_summary(
            context, ordered_resumed, calculator_session=None
        )
        return summary
    if publish_summary and existing_summary and not overwrite:
        detail = "\n".join(
            f"  - {relative_path(path, context.project_root)}"
            for path in existing_summary
        )
        raise Step6CollisionError(
            "Existing combined output collisions were found:\n" + detail
        )

    session = calculator_session
    if phases_to_execute:
        if session is None:
            session = load_calculator_session(context)
        elif session.configuration_fingerprint != context.config_fingerprint:
            raise Step6CalculatorError(
                "Provided calculator session does not match this context."
            )
        if session.load_count != 1:
            raise Step6CalculatorError(
                f"Calculator load count must be exactly one; got {session.load_count}."
            )

    with tempfile.TemporaryDirectory(
        prefix=f".step6-{context.mode}-staging-",
        dir=context.output_root,
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        all_results: dict[str, PhaseResult] = dict(resumed_results)
        for phase in phases_to_execute:
            if session is None:
                raise Step6CalculatorError("Calculator session was not created.")
            result, staged = _execute_phase_to_staging(
                context, phase, session, staging_root
            )
            all_results[phase] = result
            staged_by_final.update(staged)

        ordered_results = tuple(all_results[phase] for phase in selected)
        if publish_summary:
            summary, summary_contents = build_combined_summary(
                context, ordered_results, session
            )
            for target, payload in summary_contents.items():
                stage = _stage_path(staging_root, context.output_root, target)
                stage.write_bytes(payload)
                staged_by_final[target] = stage
        else:
            calculator_class = (
                session.calculator_class
                if session is not None
                else ordered_results[0].calculator_class
            )
            summary = ModeSummary(
                mode=context.mode,
                requested_phases=selected,
                results=ordered_results,
                calculator_loads=session.load_count if session is not None else 0,
                calculator_class=calculator_class,
                state_evaluations=sum(
                    result.state_evaluations for result in ordered_results
                ),
                optimizer_steps=sum(
                    result.optimizer_steps for result in ordered_results
                ),
                resumed_phases=tuple(
                    result.phase_key for result in ordered_results if result.resumed
                ),
                executed_phases=tuple(
                    result.phase_key for result in ordered_results if not result.resumed
                ),
                overall_status=_combined_status(ordered_results),
                combined_outputs=None,
            )

        def final_validator() -> None:
            verify_protected_files(context)
            validated_results = tuple(
                validate_phase_bundle(context, phase) for phase in selected
            )
            if publish_summary:
                _validate_combined_outputs(context, validated_results)

        _publish_files_transactionally(
            context,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )
    return summary


__all__ = [
    "ATOMIC_ONLY_MODE",
    "ATOMIC_ONLY_PILOT_PHASE",
    "CalculatorSession",
    "CombinedOutputPaths",
    "ExecutionSettings",
    "FULL_CELL_MODE",
    "FULL_CELL_PILOT_PHASE",
    "FileSnapshot",
    "ModeOutputPaths",
    "ModeSummary",
    "PHASE_ORDER",
    "PhaseResult",
    "Step6CalculationError",
    "Step6CalculatorError",
    "Step6CollisionError",
    "Step6ConfigurationError",
    "Step6Context",
    "Step6DependencyError",
    "Step6Error",
    "Step6InputError",
    "Step6PublicationError",
    "Step6ResumeError",
    "Step6SafetyError",
    "build_combined_summary",
    "combined_output_paths",
    "execute_mode",
    "installed_scientific_versions",
    "load_and_validate_context",
    "load_calculator_session",
    "locate_project_root",
    "phase_output_paths",
    "planned_mode_directories",
    "publish_combined_summary",
    "validate_all_mode_outputs",
    "validate_mode_cli_plan",
    "validate_installed_relaxation_api",
    "validate_phase_bundle",
    "verify_protected_files",
]

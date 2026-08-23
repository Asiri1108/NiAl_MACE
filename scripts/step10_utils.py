"""Shared utilities for the Ni-Al Step 10 static LAMMPS EAM benchmark.

Step 10 executes the Step 9-designed benchmark: the three validated
classical Ni-Al EAM potentials each process independent copies of the same
seven original selected structures through three states (initial fixed
geometry, Stage A fixed-cell minimization, Stage B full-cell zero-pressure
minimization), after which potential-specific formation energies are
compared against the existing MACE and Materials Project DFT-derived
results.  This module provides:

``load_step10_config``
    Strict validation of ``configs/ni_al_lammps_benchmark.json``.
``validate_step9_success`` / ``discover_lammps``
    Step 9 provenance checks and safe LAMMPS executable discovery.
``load_source_structure`` / conversion path helpers
    Validated access to the seven protected source structures.
``build_lammps_input`` / ``parse_thermo_sections`` / ``parse_force_dump``
    Controlled input generation and machine-readable output parsing.
``build_state_record`` / ``validate_state_checkpoint``
    Complete per-state result records with independent scientific
    convergence and safety validation, plus strict resume validation.

Nothing in this module runs molecular dynamics, assigns velocities, loads
MACE, queries Materials Project, or performs DFT.  LAMMPS pressure is
recorded in bar and converted to an explicitly labeled ASE-ordered stress
tensor via ``stress_eV_per_A3 = -pressure_bar / 1.602176634e6`` (LAMMPS
positive pressure is compression; ASE positive stress is tension);
convergence decisions use absolute values so the sign convention cannot
affect pass/fail.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from step6_utils import FileSnapshot
from step7_utils import (
    Step7Error,
    file_sha256,
    read_strict_json,
    relative_path,
    snapshot_file,
    verify_snapshots,
)


LOGGER = logging.getLogger("step10_utils")

SCHEMA_VERSION = "1.0"
POTENTIAL_ORDER: tuple[str, ...] = (
    "pun_mishin_2009",
    "mishin_2004_ipr2",
    "mishin_2002",
)
STRUCTURE_ORDER: tuple[str, ...] = (
    "Al",
    "Ni",
    "Al3Ni",
    "Al3Ni2",
    "AlNi",
    "Al3Ni5",
    "AlNi3",
)
ELEMENT_KEYS: tuple[str, ...] = ("Al", "Ni")
COMPOUND_ORDER: tuple[str, ...] = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
STATE_ORDER: tuple[str, ...] = ("initial", "fixed_cell", "full_cell")
PILOT_POTENTIAL = "pun_mishin_2009"
PILOT_PHASE = "AlNi"

# LAMMPS thermo pressure order used by the generated inputs and the ASE
# Voigt stress order used throughout this project.
LAMMPS_PRESSURE_COMPONENTS: tuple[str, ...] = (
    "pxx",
    "pyy",
    "pzz",
    "pxy",
    "pxz",
    "pyz",
)
ASE_STRESS_COMPONENTS: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
# Explicit component mapping from LAMMPS pressure keys to ASE stress slots.
ASE_FROM_LAMMPS: Mapping[str, str] = {
    "xx": "pxx",
    "yy": "pyy",
    "zz": "pzz",
    "yz": "pyz",
    "xz": "pxz",
    "xy": "pxy",
}
BAR_PER_EV_A3 = 1.602176634e6

THERMO_COLUMNS: tuple[str, ...] = (
    "Step",
    "PotEng",
    "Press",
    "Pxx",
    "Pyy",
    "Pzz",
    "Pxy",
    "Pxz",
    "Pyz",
    "Volume",
    "Lx",
    "Ly",
    "Lz",
    "Xy",
    "Xz",
    "Yz",
    "c_fmax",
)


class Step10Error(RuntimeError):
    """Base class for controlled Step 10 failures."""


class Step10ConfigurationError(Step10Error):
    """Raised when the Step 10 configuration or command scope is unsafe."""


class Step10DependencyError(Step10Error):
    """Raised when a required executable or installed API is unavailable."""


class Step10InputError(Step10Error):
    """Raised when a protected scientific input is invalid."""


class Step10ConversionError(Step10Error):
    """Raised when structure conversion or round-trip validation fails."""


class Step10ExecutionError(Step10Error):
    """Raised when a LAMMPS invocation fails."""


class Step10ParseError(Step10Error):
    """Raised when controlled LAMMPS output cannot be parsed."""


class Step10SafetyError(Step10Error):
    """Raised when a configured safety invariant is violated."""


class Step10CollisionError(Step10Error):
    """Raised when output collision handling refuses publication."""


class Step10ResumeError(Step10Error):
    """Raised when an existing Step 10 bundle is not safe to reuse."""


class Step10CalculationError(Step10Error):
    """Raised when a formation-energy or comparison calculation fails."""


def _require_exact(value: Any, expected: Any, label: str) -> Any:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise Step10ConfigurationError(
            f"{label} must be exactly {expected!r}; received {value!r}."
        )
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Step10ConfigurationError(f"{label} must be a JSON object.")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Step10ConfigurationError(f"{label} must be a nonempty string.")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step10ParseError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise Step10SafetyError(f"{label} is NaN or infinity.")
    return result


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise Step10DependencyError(f"NumPy is unavailable: {exc}") from exc
    return np


def _relative_repo_path(value: Any, label: str, project_root: Path) -> Path:
    text = _require_string(value, label)
    candidate = (project_root / Path(text)).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step10ConfigurationError(
            f"{label} must remain inside the repository: {text}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class PotentialSpec:
    """Validated configuration for one classical potential."""

    key: str
    role: str
    path: Path
    pair_style: str
    pair_coeff_elements: tuple[str, str]


@dataclass(frozen=True)
class StructureSpec:
    """Validated configuration for one source structure."""

    key: str
    kind: str
    path: Path
    material_id: str
    expected_atoms: int
    conversion_directory: str


@dataclass(frozen=True)
class Step10Config:
    """Fully validated Step 10 configuration."""

    project_root: Path
    config_path: Path
    config_snapshot: FileSnapshot
    fingerprint: str
    raw: Mapping[str, Any]
    executable_environment_variable: str
    executable_candidates: tuple[str, ...]
    units: str
    atom_style: str
    boundary: tuple[str, str, str]
    atom_type_order: tuple[str, str]
    potentials: Mapping[str, PotentialSpec]
    structures: Mapping[str, StructureSpec]
    force_threshold_eV_per_A: float
    pressure_threshold_bar: float
    stress_threshold_eV_per_A3: float
    minimization: Mapping[str, Any]
    safety: Mapping[str, Any]
    symmetry_symprec_A: float
    symmetry_angle_tolerance_deg: float
    comparison_sources: Mapping[str, Path]
    envelope_tolerance_eV_per_atom: float
    arithmetic_tolerance_eV_per_atom: float
    error_threshold_bins: tuple[float, ...]
    conversion_root: Path
    run_root: Path
    analysis_root: Path


def load_step10_config(config_path: Path | str) -> Step10Config:
    """Load and strictly validate the Step 10 benchmark configuration."""

    from step7_utils import locate_project_root

    project_root = locate_project_root()
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step10ConfigurationError(
            f"Configuration must remain inside the repository: {resolved}"
        ) from exc
    try:
        raw = read_strict_json(resolved, "Step 10 benchmark configuration")
    except Step7Error as exc:
        raise Step10ConfigurationError(str(exc)) from exc

    _require_exact(raw.get("schema_version"), SCHEMA_VERSION, "schema_version")
    lammps_raw = _require_mapping(raw.get("lammps"), "lammps")
    _require_exact(
        lammps_raw.get("executable_environment_variable"),
        "LAMMPS_EXECUTABLE",
        "lammps.executable_environment_variable",
    )
    _require_exact(
        lammps_raw.get("require_eam_alloy"), True, "lammps.require_eam_alloy"
    )
    _require_exact(lammps_raw.get("units"), "metal", "lammps.units")
    _require_exact(lammps_raw.get("atom_style"), "atomic", "lammps.atom_style")
    _require_exact(
        lammps_raw.get("boundary"), ["p", "p", "p"], "lammps.boundary"
    )
    _require_exact(
        lammps_raw.get("atom_type_order"),
        ["Al", "Ni"],
        "lammps.atom_type_order",
    )
    executable_candidates = tuple(
        str(item) for item in lammps_raw.get("executable_candidates", ())
    )
    if not executable_candidates:
        raise Step10ConfigurationError(
            "lammps.executable_candidates must not be empty."
        )

    potentials_raw = _require_mapping(raw.get("potentials"), "potentials")
    if tuple(potentials_raw.keys()) != POTENTIAL_ORDER:
        raise Step10ConfigurationError(
            f"potentials must define exactly {list(POTENTIAL_ORDER)} in order."
        )
    expected_potentials = {
        "pun_mishin_2009": (
            "primary",
            "data/processed/interatomic_potentials/ni_al/pun_mishin_2009/"
            "Mishin-Ni-Al-2009.eam.alloy",
        ),
        "mishin_2004_ipr2": (
            "secondary",
            "data/processed/interatomic_potentials/ni_al/mishin_2004_ipr2/"
            "NiAl_Mishin_2004.eam.alloy",
        ),
        "mishin_2002": (
            "historical_secondary",
            "data/processed/interatomic_potentials/ni_al/mishin_2002/"
            "NiAl02.eam.alloy",
        ),
    }
    potentials: dict[str, PotentialSpec] = {}
    for key in POTENTIAL_ORDER:
        spec_raw = _require_mapping(potentials_raw.get(key), f"potentials.{key}")
        role, expected_path = expected_potentials[key]
        _require_exact(spec_raw.get("role"), role, f"potentials.{key}.role")
        _require_exact(
            spec_raw.get("pair_style"), "eam/alloy", f"potentials.{key}.pair_style"
        )
        _require_exact(
            spec_raw.get("pair_coeff_elements"),
            ["Al", "Ni"],
            f"potentials.{key}.pair_coeff_elements",
        )
        path = _relative_repo_path(
            _require_exact(
                spec_raw.get("path"), expected_path, f"potentials.{key}.path"
            ),
            f"potentials.{key}.path",
            project_root,
        )
        potentials[key] = PotentialSpec(
            key=key,
            role=role,
            path=path,
            pair_style="eam/alloy",
            pair_coeff_elements=("Al", "Ni"),
        )

    structures_raw = _require_mapping(raw.get("structures"), "structures")
    if tuple(structures_raw.keys()) != STRUCTURE_ORDER:
        raise Step10ConfigurationError(
            f"structures must define exactly {list(STRUCTURE_ORDER)} in order."
        )
    expected_structures = {
        "Al": ("elemental_reference", "mp-134", 1, "pure_Al"),
        "Ni": ("elemental_reference", "mp-23", 1, "pure_Ni"),
        "Al3Ni": ("compound", "mp-622209", 16, "Al3Ni"),
        "Al3Ni2": ("compound", "mp-1057", 5, "Al3Ni2"),
        "AlNi": ("compound", "mp-1487", 2, "AlNi"),
        "Al3Ni5": ("compound", "mp-16514", 8, "Al3Ni5"),
        "AlNi3": ("compound", "mp-2593", 4, "AlNi3"),
    }
    structures: dict[str, StructureSpec] = {}
    for key in STRUCTURE_ORDER:
        spec_raw = _require_mapping(structures_raw.get(key), f"structures.{key}")
        kind, material_id, atoms, directory = expected_structures[key]
        _require_exact(spec_raw.get("kind"), kind, f"structures.{key}.kind")
        _require_exact(
            spec_raw.get("material_id"),
            material_id,
            f"structures.{key}.material_id",
        )
        _require_exact(
            spec_raw.get("expected_atoms"),
            atoms,
            f"structures.{key}.expected_atoms",
        )
        _require_exact(
            spec_raw.get("conversion_directory"),
            directory,
            f"structures.{key}.conversion_directory",
        )
        path = _relative_repo_path(
            spec_raw.get("path"), f"structures.{key}.path", project_root
        )
        structures[key] = StructureSpec(
            key=key,
            kind=kind,
            path=path,
            material_id=material_id,
            expected_atoms=atoms,
            conversion_directory=directory,
        )

    convergence_raw = _require_mapping(
        raw.get("scientific_convergence"), "scientific_convergence"
    )
    force_threshold = _require_exact(
        convergence_raw.get("maximum_force_eV_per_A"),
        0.01,
        "scientific_convergence.maximum_force_eV_per_A",
    )
    pressure_threshold = _require_exact(
        convergence_raw.get("maximum_absolute_pressure_bar"),
        999.999988070,
        "scientific_convergence.maximum_absolute_pressure_bar",
    )
    stress_threshold = _require_exact(
        convergence_raw.get("maximum_absolute_stress_eV_per_A3"),
        0.0006241509,
        "scientific_convergence.maximum_absolute_stress_eV_per_A3",
    )

    minimization_raw = _require_mapping(raw.get("minimization"), "minimization")
    for name, expected in (
        ("style", "cg"),
        ("line_search", "quadratic"),
        ("energy_tolerance", 0.0),
        ("technical_force_tolerance_eV_per_A", 1e-10),
        ("maximum_iterations_per_cycle", 10000),
        ("maximum_force_evaluations_per_cycle", 100000),
        ("maximum_cycles", 5),
        ("box_relax_mode", "tri"),
        ("target_pressure_bar", 0.0),
        ("vmax", 0.001),
        ("neighbor_skin_A", 2.0),
    ):
        _require_exact(minimization_raw.get(name), expected, f"minimization.{name}")

    safety_raw = _require_mapping(raw.get("safety"), "safety")
    for name, expected in (
        ("maximum_absolute_volume_change_percent", 25.0),
        ("maximum_internal_atomic_displacement_A", 2.0),
        ("stop_on_nonfinite_value", True),
        ("require_positive_cell_determinant", True),
        ("preserve_atom_order", True),
        ("preserve_composition", True),
    ):
        _require_exact(safety_raw.get(name), expected, f"safety.{name}")

    symmetry_raw = _require_mapping(raw.get("symmetry"), "symmetry")
    _require_exact(symmetry_raw.get("symprec_A"), 0.001, "symmetry.symprec_A")
    _require_exact(
        symmetry_raw.get("angle_tolerance_deg"),
        5.0,
        "symmetry.angle_tolerance_deg",
    )

    sources_raw = _require_mapping(
        raw.get("comparison_sources"), "comparison_sources"
    )
    comparison_sources = {
        name: _relative_repo_path(
            sources_raw.get(name), f"comparison_sources.{name}", project_root
        )
        for name in (
            "step8_energy_table",
            "step8_structural_table",
            "step8_checkpoint",
            "step7_formation_table",
            "step6_full_cell_summary",
            "step9_manifest",
            "step9_report",
            "step9_plan",
        )
    }

    analysis_raw = _require_mapping(raw.get("analysis"), "analysis")
    _require_exact(
        analysis_raw.get("envelope_tolerance_eV_per_atom"),
        1e-8,
        "analysis.envelope_tolerance_eV_per_atom",
    )
    _require_exact(
        analysis_raw.get("arithmetic_tolerance_eV_per_atom"),
        1e-12,
        "analysis.arithmetic_tolerance_eV_per_atom",
    )
    _require_exact(
        analysis_raw.get("error_threshold_bins_eV_per_atom"),
        [0.05, 0.1],
        "analysis.error_threshold_bins_eV_per_atom",
    )

    outputs_raw = _require_mapping(raw.get("outputs"), "outputs")
    conversion_root = _relative_repo_path(
        _require_exact(
            outputs_raw.get("conversion_root"),
            "data/processed/lammps_structures/ni_al",
            "outputs.conversion_root",
        ),
        "outputs.conversion_root",
        project_root,
    )
    run_root = _relative_repo_path(
        _require_exact(
            outputs_raw.get("run_root"),
            "results/lammps_benchmark/runs",
            "outputs.run_root",
        ),
        "outputs.run_root",
        project_root,
    )
    analysis_root = _relative_repo_path(
        _require_exact(
            outputs_raw.get("analysis_root"),
            "results/lammps_benchmark",
            "outputs.analysis_root",
        ),
        "outputs.analysis_root",
        project_root,
    )

    snapshot = snapshot_file(resolved, "Step 10 configuration")
    return Step10Config(
        project_root=project_root,
        config_path=resolved,
        config_snapshot=snapshot,
        fingerprint=snapshot.sha256,
        raw=raw,
        executable_environment_variable="LAMMPS_EXECUTABLE",
        executable_candidates=executable_candidates,
        units="metal",
        atom_style="atomic",
        boundary=("p", "p", "p"),
        atom_type_order=("Al", "Ni"),
        potentials=potentials,
        structures=structures,
        force_threshold_eV_per_A=float(force_threshold),
        pressure_threshold_bar=float(pressure_threshold),
        stress_threshold_eV_per_A3=float(stress_threshold),
        minimization=dict(minimization_raw),
        safety=dict(safety_raw),
        symmetry_symprec_A=0.001,
        symmetry_angle_tolerance_deg=5.0,
        comparison_sources=comparison_sources,
        envelope_tolerance_eV_per_atom=1e-8,
        arithmetic_tolerance_eV_per_atom=1e-12,
        error_threshold_bins=(0.05, 0.1),
        conversion_root=conversion_root,
        run_root=run_root,
        analysis_root=analysis_root,
    )


def validate_selection(
    values: Sequence[str] | None, universe: tuple[str, ...], label: str
) -> tuple[str, ...]:
    """Normalize a requested subset selection deterministically."""

    if values is None:
        return universe
    selected = tuple(values)
    if not selected:
        raise Step10ConfigurationError(f"At least one {label} must be selected.")
    if len(set(selected)) != len(selected):
        raise Step10ConfigurationError(f"Requested {label}s contain duplicates.")
    unknown = [item for item in selected if item not in universe]
    if unknown:
        raise Step10ConfigurationError(
            f"Unknown {label}(s): " + ", ".join(unknown)
        )
    return tuple(item for item in universe if item in selected)


# ---------------------------------------------------------------------------
# Step 9 provenance and LAMMPS discovery
# ---------------------------------------------------------------------------


def validate_step9_success(
    config: Step10Config,
) -> tuple[Mapping[str, str], tuple[FileSnapshot, ...]]:
    """Confirm Step 9 SUCCESS and verify the three potential fingerprints."""

    report_path = config.comparison_sources["step9_report"]
    manifest_path = config.comparison_sources["step9_manifest"]
    snapshots = [
        snapshot_file(report_path, "Step 9 final report"),
        snapshot_file(manifest_path, "Step 9 potential manifest"),
    ]
    text = report_path.read_text(encoding="utf-8")
    if "OVERALL STEP 9 STATUS: SUCCESS" not in text:
        raise Step10InputError(
            "The Step 9 final report does not record OVERALL STEP 9 STATUS: "
            "SUCCESS; Step 10 cannot run on unfinished Step 9 results."
        )
    try:
        manifest = read_strict_json(manifest_path, "Step 9 potential manifest")
    except Step7Error as exc:
        raise Step10InputError(str(exc)) from exc
    recorded: dict[str, str] = {}
    for entry in manifest.get("files", ()):
        if not isinstance(entry, Mapping):
            continue
        key = str(entry.get("candidate_key"))
        recorded[key] = str(entry.get("sha256"))
    hashes: dict[str, str] = {}
    for key in POTENTIAL_ORDER:
        spec = config.potentials[key]
        snapshots.append(snapshot_file(spec.path, f"{key} potential file"))
        digest = file_sha256(spec.path)
        if recorded.get(key) != digest:
            raise Step10InputError(
                f"{key}: potential file SHA-256 {digest} does not match the "
                f"Step 9 manifest value {recorded.get(key)!r}."
            )
        hashes[key] = digest
    verify_snapshots(snapshots)
    return hashes, tuple(snapshots)


@dataclass(frozen=True)
class LammpsExecutable:
    """Validated local LAMMPS executable."""

    path: str
    discovery_source: str
    version_line: str
    eam_alloy_listed: bool
    sha256: str | None
    size_bytes: int | None


def discover_lammps(config: Step10Config) -> LammpsExecutable:
    """Discover and validate the LAMMPS executable without running physics.

    Discovery order: the ``LAMMPS_EXECUTABLE`` environment variable, the
    executable recorded by the Step 9 plan, then PATH searches.  Only
    ``<executable> -h`` is invoked; no simulation runs.
    """

    candidates: list[tuple[str, str]] = []
    env_value = os.environ.get(config.executable_environment_variable, "").strip()
    if env_value:
        candidates.append((env_value, "environment"))
    try:
        plan = read_strict_json(
            config.comparison_sources["step9_plan"], "Step 9 plan"
        )
        step9_path = plan.get("lammps_availability", {}).get("executable_path")
        if isinstance(step9_path, str) and step9_path.strip():
            candidates.append((step9_path, "step9_plan"))
    except Step7Error:
        pass
    for name in config.executable_candidates:
        located = shutil.which(name)
        if located:
            candidates.append((located, f"which:{name}"))

    last_error = "no candidate executable was found"
    for path_text, source in candidates:
        path = Path(path_text)
        if not path.is_file():
            last_error = f"{path_text} does not exist"
            continue
        try:
            completed = subprocess.run(
                [str(path), "-h"],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = f"'{path_text} -h' failed: {type(exc).__name__}: {exc}"
            continue
        if completed.returncode != 0:
            last_error = (
                f"'{path_text} -h' exited with code {completed.returncode}"
            )
            continue
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        # Prefer the real version banner over the usage example: '-h' prints
        # 'Large-scale Atomic/Molecular Massively Parallel Simulator - <date>'.
        version_line = next(
            (
                line.strip()
                for line in output.splitlines()
                if "Massively Parallel Simulator" in line
                or line.strip().startswith("LAMMPS (")
            ),
            "",
        ) or next(
            (
                line.strip()
                for line in output.splitlines()
                if "LAMMPS" in line and line.strip()
            ),
            "",
        )
        eam_alloy = "eam/alloy" in output
        if not eam_alloy:
            last_error = f"{path_text} does not list eam/alloy"
            continue
        try:
            sha256: str | None = file_sha256(path)
            size: int | None = path.stat().st_size
        except (Step7Error, OSError):
            sha256 = None
            size = None
        return LammpsExecutable(
            path=str(path),
            discovery_source=source,
            version_line=version_line,
            eam_alloy_listed=True,
            sha256=sha256,
            size_bytes=size,
        )
    raise Step10DependencyError(
        "No usable LAMMPS executable with eam/alloy support was found "
        f"({last_error}). Set {config.executable_environment_variable} or "
        "install LAMMPS manually; nothing is installed automatically."
    )


# ---------------------------------------------------------------------------
# Source structures and conversion paths
# ---------------------------------------------------------------------------


def load_source_structure(
    config: Step10Config, key: str
) -> tuple[Any, FileSnapshot]:
    """Read and validate one protected source structure."""

    spec = config.structures[key]
    if not spec.path.is_file():
        raise Step10InputError(f"Source structure is missing: {spec.path}")
    snapshot = snapshot_file(spec.path, f"source structure {key}")
    try:
        from ase.io import read as ase_read

        frames = ase_read(spec.path, index=":", format="extxyz")
    except Exception as exc:
        raise Step10InputError(
            f"Could not read source structure {key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list) or len(frames) != 1:
        raise Step10InputError(f"{key}: source EXTXYZ must contain one frame.")
    atoms = frames[0]
    atoms.calc = None
    np = _numpy()
    symbols = tuple(atoms.get_chemical_symbols())
    if len(atoms) != spec.expected_atoms:
        raise Step10InputError(
            f"{key}: expected {spec.expected_atoms} atoms, found {len(atoms)}."
        )
    if set(symbols) - {"Al", "Ni"}:
        raise Step10InputError(f"{key}: non-Al/Ni species present.")
    info_id = atoms.info.get("material_id")
    if info_id is not None and str(info_id) != spec.material_id:
        raise Step10InputError(
            f"{key}: EXTXYZ material_id {info_id!r} does not match the "
            f"expected {spec.material_id!r}."
        )
    cell = np.asarray(atoms.cell.array, dtype=float)
    determinant = float(np.linalg.det(cell))
    if (
        not bool(np.all(np.isfinite(atoms.get_positions())))
        or not bool(np.all(np.isfinite(cell)))
        or determinant <= 0.0
        or float(atoms.get_volume()) <= 0.0
        or not bool(np.all(np.asarray(atoms.get_pbc(), dtype=bool)))
    ):
        raise Step10InputError(f"{key}: source geometry is invalid.")
    return atoms, snapshot


def conversion_paths(config: Step10Config, key: str) -> tuple[Path, Path, Path]:
    """Return (data file, conversion JSON, conversion report) for one phase."""

    directory = config.conversion_root / config.structures[key].conversion_directory
    return (
        directory / f"{key}.lammps.data",
        directory / f"{key}.conversion.json",
        directory / f"{key}.conversion_report.txt",
    )


def read_lammps_structure(path: Path, label: str) -> Any:
    """Read one LAMMPS data file back into ASE using the fixed conventions."""

    try:
        from ase.io.lammpsdata import read_lammps_data

        with path.open("r", encoding="utf-8") as handle:
            atoms = read_lammps_data(
                handle,
                Z_of_type={1: 13, 2: 28},
                sort_by_id=True,
                units="metal",
                atom_style="atomic",
            )
    except Exception as exc:
        raise Step10ConversionError(
            f"Could not read {label}: {type(exc).__name__}: {exc}"
        ) from exc
    atoms.calc = None
    return atoms


def validate_converted_bundle(config: Step10Config, key: str) -> Mapping[str, Any]:
    """Validate one published converted bundle against its source."""

    data_path, json_path, report_path = conversion_paths(config, key)
    for path, label in (
        (data_path, f"{key} converted data"),
        (json_path, f"{key} conversion JSON"),
        (report_path, f"{key} conversion report"),
    ):
        if not path.is_file():
            raise Step10ConversionError(
                f"{label} does not exist: {path}. Run "
                "convert_ni_al_structures_to_lammps.py --convert first."
            )
    try:
        record = read_strict_json(json_path, f"{key} conversion JSON")
    except Step7Error as exc:
        raise Step10ConversionError(str(exc)) from exc
    if record.get("structure_key") != key:
        raise Step10ConversionError(f"{key}: conversion JSON key mismatch.")
    if record.get("data_file_sha256") != file_sha256(data_path):
        raise Step10ConversionError(
            f"{key}: converted data file hash does not match its record."
        )
    source_sha = record.get("source_sha256")
    current_source = file_sha256(config.structures[key].path)
    if source_sha != current_source:
        raise Step10ConversionError(
            f"{key}: source structure changed since conversion "
            f"({source_sha} versus {current_source})."
        )
    if record.get("round_trip_status") != "PASS":
        raise Step10ConversionError(f"{key}: round-trip status is not PASS.")
    return record


# ---------------------------------------------------------------------------
# LAMMPS input generation
# ---------------------------------------------------------------------------


def _lammps_path(path: Path) -> str:
    """Return an absolute forward-slash path for a LAMMPS input command."""

    text = str(path.resolve()).replace("\\", "/")
    if " " in text:
        raise Step10ConfigurationError(
            f"LAMMPS input paths must not contain spaces: {text}"
        )
    return text


def build_lammps_input(
    config: Step10Config,
    potential_path: Path,
    data_path: Path,
    stage: str,
    dump_name: str,
    final_data_name: str,
) -> str:
    """Generate the explicit LAMMPS input for one stage invocation.

    ``initial`` performs a single ``run 0`` (no motion); ``fixed_cell``
    minimizes atomic positions with the cell fixed; ``full_cell`` minimizes
    atoms plus all six cell degrees of freedom under ``fix box/relax tri``
    at zero target pressure.  No dynamics, velocities, or thermostats are
    ever generated.
    """

    if stage not in STATE_ORDER:
        raise Step10ConfigurationError(f"Unknown stage: {stage!r}")
    minimization = config.minimization
    lines = [
        "# Generated by Step 10; static calculation only - no dynamics.",
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"read_data {_lammps_path(data_path)}",
        "",
        "pair_style eam/alloy",
        f"pair_coeff * * {_lammps_path(potential_path)} Al Ni",
        "",
        f"neighbor {minimization['neighbor_skin_A']} bin",
        "neigh_modify delay 0 every 1 check yes",
        "",
        "variable fmag atom sqrt(fx*fx+fy*fy+fz*fz)",
        "compute fmax all reduce max v_fmag",
        "thermo 1",
        "thermo_style custom step pe press pxx pyy pzz pxy pxz pyz vol "
        "lx ly lz xy xz yz c_fmax",
        "thermo_modify format float %20.15g",
        "",
    ]
    if stage == "initial":
        lines.append("run 0")
    else:
        lines.extend(
            [
                f"min_style {minimization['style']}",
                f"min_modify line {minimization['line_search']}",
            ]
        )
        if stage == "full_cell":
            lines.append(
                "fix boxrelax all box/relax "
                f"{minimization['box_relax_mode']} "
                f"{minimization['target_pressure_bar']} "
                f"vmax {minimization['vmax']}"
            )
        lines.append(
            f"minimize {minimization['energy_tolerance']} "
            f"{minimization['technical_force_tolerance_eV_per_A']} "
            f"{minimization['maximum_iterations_per_cycle']} "
            f"{minimization['maximum_force_evaluations_per_cycle']}"
        )
        if stage == "full_cell":
            lines.append("unfix boxrelax")
    lines.extend(
        [
            "",
            f"write_dump all custom {dump_name} id type x y z fx fy fz "
            "modify format float %20.15g sort id",
            f"write_data {final_data_name}",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LAMMPS output parsing
# ---------------------------------------------------------------------------


def parse_thermo_sections(log_text: str) -> list[list[dict[str, float]]]:
    """Parse every thermo table from a LAMMPS log into numeric rows."""

    sections: list[list[dict[str, float]]] = []
    lines = log_text.splitlines()
    index = 0
    while index < len(lines):
        tokens = lines[index].split()
        if tokens and tokens[0] == "Step":
            header = tokens
            rows: list[dict[str, float]] = []
            index += 1
            while index < len(lines):
                row_tokens = lines[index].split()
                if len(row_tokens) != len(header):
                    break
                try:
                    values = [float(token) for token in row_tokens]
                except ValueError:
                    break
                rows.append(dict(zip(header, values)))
                index += 1
            if rows:
                sections.append(rows)
            continue
        index += 1
    return sections


def extract_final_thermo(log_text: str, label: str) -> dict[str, float]:
    """Return the final thermo row of the final section with validation."""

    sections = parse_thermo_sections(log_text)
    if not sections or not sections[-1]:
        raise Step10ParseError(f"{label}: no thermo table found in the log.")
    row = sections[-1][-1]
    missing = [name for name in THERMO_COLUMNS if name not in row]
    if missing:
        raise Step10ParseError(
            f"{label}: thermo columns missing from the log: {missing}."
        )
    for name in THERMO_COLUMNS:
        if not math.isfinite(row[name]):
            raise Step10SafetyError(
                f"{label}: thermo value {name} is nonfinite."
            )
    return row


def parse_minimization_stats(log_text: str, label: str) -> dict[str, Any]:
    """Parse stopping criterion, iterations, evaluations, and loop time."""

    stop_reason: str | None = None
    iterations: int | None = None
    evaluations: int | None = None
    loop_time_seconds = 0.0
    lines = log_text.splitlines()
    for position, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Stopping criterion ="):
            stop_reason = stripped.split("=", 1)[1].strip()
        if stripped.startswith("Iterations, force evaluations"):
            try:
                numbers = stripped.split("=", 1)[1].split()
                iterations = int(numbers[0])
                evaluations = int(numbers[1])
            except (IndexError, ValueError) as exc:
                raise Step10ParseError(
                    f"{label}: could not parse iteration counts."
                ) from exc
        if stripped.startswith("Loop time of"):
            try:
                loop_time_seconds += float(stripped.split()[3])
            except (IndexError, ValueError):
                pass
        del position
    return {
        "stop_reason": stop_reason,
        "iterations": iterations,
        "force_evaluations": evaluations,
        "loop_time_seconds": loop_time_seconds,
    }


def check_log_for_errors(log_text: str, stdout: str, stderr: str, label: str) -> None:
    """Reject runs whose output contains LAMMPS errors or lost atoms."""

    for name, text in (("log", log_text), ("stdout", stdout), ("stderr", stderr)):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("ERROR"):
                raise Step10ExecutionError(
                    f"{label}: LAMMPS reported an error in {name}: {stripped}"
                )
            if "Lost atoms" in stripped:
                raise Step10ExecutionError(
                    f"{label}: LAMMPS reported lost atoms in {name}."
                )


def parse_force_dump(path: Path, expected_atoms: int, label: str) -> Any:
    """Parse an id/type/x/y/z/fx/fy/fz dump into a sorted numpy array."""

    np = _numpy()
    if not path.is_file():
        raise Step10ParseError(f"{label}: force dump is missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        atoms_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("ITEM: ATOMS")
        )
    except StopIteration as exc:
        raise Step10ParseError(f"{label}: dump has no ATOMS section.") from exc
    columns = lines[atoms_index].split()[2:]
    expected_columns = ["id", "type", "x", "y", "z", "fx", "fy", "fz"]
    if columns != expected_columns:
        raise Step10ParseError(
            f"{label}: unexpected dump columns {columns}."
        )
    rows = []
    for line in lines[atoms_index + 1 : atoms_index + 1 + expected_atoms]:
        values = [float(token) for token in line.split()]
        if len(values) != len(expected_columns):
            raise Step10ParseError(f"{label}: malformed dump row.")
        rows.append(values)
    if len(rows) != expected_atoms:
        raise Step10ParseError(
            f"{label}: dump contains {len(rows)} atoms; expected "
            f"{expected_atoms}."
        )
    array = np.asarray(rows, dtype=float)
    if not bool(np.all(np.isfinite(array))):
        raise Step10SafetyError(f"{label}: dump contains nonfinite values.")
    order = np.argsort(array[:, 0], kind="stable")
    return array[order]


def pressure_row_to_records(row: Mapping[str, float]) -> dict[str, Any]:
    """Convert one thermo row into pressure/stress records with mapping."""

    pressure = {
        "pxx": row["Pxx"],
        "pyy": row["Pyy"],
        "pzz": row["Pzz"],
        "pxy": row["Pxy"],
        "pxz": row["Pxz"],
        "pyz": row["Pyz"],
    }
    stress = {
        component: -pressure[ASE_FROM_LAMMPS[component]] / BAR_PER_EV_A3
        for component in ASE_STRESS_COMPONENTS
    }
    return {
        "pressure_bar": [pressure[name] for name in LAMMPS_PRESSURE_COMPONENTS],
        "pressure_component_order": list(LAMMPS_PRESSURE_COMPONENTS),
        "maximum_absolute_pressure_bar": max(
            abs(value) for value in pressure.values()
        ),
        "stress_eV_per_A3": [stress[name] for name in ASE_STRESS_COMPONENTS],
        "stress_component_order": list(ASE_STRESS_COMPONENTS),
        "maximum_absolute_stress_eV_per_A3": max(
            abs(value) for value in stress.values()
        ),
        "stress_sign_convention": (
            "stress_eV_per_A3 = -pressure_bar / 1.602176634e6; LAMMPS "
            "positive pressure is compression, ASE positive stress is "
            "tension; convergence uses absolute values."
        ),
    }


# ---------------------------------------------------------------------------
# Geometry helpers and symmetry
# ---------------------------------------------------------------------------


def wrapped_displacement_statistics(
    reference_atoms: Any, current_atoms: Any
) -> tuple[float, float]:
    """Return (maximum, RMS) internal displacement via wrapped fractional
    differences mapped through the reference cell."""

    np = _numpy()
    reference_scaled = np.asarray(
        reference_atoms.get_scaled_positions(wrap=False), dtype=float
    )
    current_scaled = np.asarray(
        current_atoms.get_scaled_positions(wrap=False), dtype=float
    )
    if reference_scaled.shape != current_scaled.shape:
        raise Step10SafetyError("Displacement shapes are incompatible.")
    delta = current_scaled - reference_scaled
    delta = delta - np.floor(delta + 0.5)
    internal = delta @ np.asarray(reference_atoms.cell.array, dtype=float)
    magnitudes = np.linalg.norm(internal, axis=1)
    if not bool(np.all(np.isfinite(magnitudes))):
        raise Step10SafetyError("Displacement metrics are nonfinite.")
    return (
        float(np.max(magnitudes)) if len(magnitudes) else 0.0,
        float(np.sqrt(np.mean(np.square(magnitudes)))) if len(magnitudes) else 0.0,
    )


def analyze_symmetry_of_atoms(config: Step10Config, atoms: Any, label: str) -> dict[str, Any]:
    """Analyze the space group of an ASE structure with project tolerances."""

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:
        raise Step10DependencyError(
            f"pymatgen symmetry analysis is unavailable: {exc}"
        ) from exc
    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=config.symmetry_symprec_A,
            angle_tolerance=config.symmetry_angle_tolerance_deg,
        )
        return {
            "space_group_symbol": analyzer.get_space_group_symbol(),
            "space_group_number": int(analyzer.get_space_group_number()),
            "symprec_A": config.symmetry_symprec_A,
            "angle_tolerance_deg": config.symmetry_angle_tolerance_deg,
        }
    except Exception as exc:
        raise Step10CalculationError(
            f"Symmetry analysis failed for {label}: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Run directory layout and state records
# ---------------------------------------------------------------------------


def run_phase_dir(config: Step10Config, potential: str, phase: str) -> Path:
    """Return the run directory for one potential/phase pair."""

    return config.run_root / potential / phase


def stage_dir(config: Step10Config, potential: str, phase: str, stage: str) -> Path:
    """Return the stage directory for one potential/phase/stage triple."""

    return run_phase_dir(config, potential, phase) / stage


def stage_checkpoint_path(
    config: Step10Config, potential: str, phase: str, stage: str
) -> Path:
    """Return the machine-readable checkpoint path for one stage."""

    return (
        run_phase_dir(config, potential, phase)
        / "checkpoints"
        / f"{stage}_result.json"
    )


def stage_report_path(
    config: Step10Config, potential: str, phase: str, stage: str
) -> Path:
    """Return the human-readable stage report path."""

    return (
        run_phase_dir(config, potential, phase) / "reports" / f"{stage}_report.txt"
    )


def build_state_record(
    config: Step10Config,
    potential: str,
    potential_sha256: str,
    phase: str,
    stage: str,
    thermo_row: Mapping[str, float],
    forces: Any,
    final_atoms: Any,
    original_atoms: Any,
    start_atoms: Any,
    cycles: Sequence[Mapping[str, Any]],
    executable: LammpsExecutable,
    wall_time_seconds: float,
    source_data_sha256: str,
    timestamps: Mapping[str, str],
) -> dict[str, Any]:
    """Build one complete validated state record.

    ``original_atoms`` is the converted original structure (displacement and
    volume-change reference); ``start_atoms`` is this stage's actual input
    structure.  Scientific convergence is decided independently from the
    parsed maximum force and pressure components, never from minimizer
    termination alone.
    """

    np = _numpy()
    spec = config.structures[phase]
    symbols = tuple(final_atoms.get_chemical_symbols())
    if len(final_atoms) != spec.expected_atoms:
        raise Step10SafetyError(f"{potential}/{phase}/{stage}: atom count changed.")
    if tuple(original_atoms.get_chemical_symbols()) != symbols:
        raise Step10SafetyError(
            f"{potential}/{phase}/{stage}: species order changed."
        )
    cell = np.asarray(final_atoms.cell.array, dtype=float)
    determinant = float(np.linalg.det(cell))
    volume = float(final_atoms.get_volume())
    if determinant <= 0.0 or volume <= 0.0 or not bool(np.all(np.isfinite(cell))):
        raise Step10SafetyError(
            f"{potential}/{phase}/{stage}: final cell is invalid."
        )
    positions = np.asarray(final_atoms.get_positions(), dtype=float)
    if not bool(np.all(np.isfinite(positions))):
        raise Step10SafetyError(
            f"{potential}/{phase}/{stage}: final positions are nonfinite."
        )
    original_volume = float(original_atoms.get_volume())
    volume_change_percent = 100.0 * (volume / original_volume - 1.0)
    if abs(volume_change_percent) > float(
        config.safety["maximum_absolute_volume_change_percent"]
    ):
        raise Step10SafetyError(
            f"{potential}/{phase}/{stage}: volume change "
            f"{volume_change_percent:.6f}% exceeds the 25% safety limit."
        )
    maximum_displacement, rms_displacement = wrapped_displacement_statistics(
        original_atoms, final_atoms
    )
    if maximum_displacement > float(
        config.safety["maximum_internal_atomic_displacement_A"]
    ):
        raise Step10SafetyError(
            f"{potential}/{phase}/{stage}: internal displacement "
            f"{maximum_displacement:.6f} A exceeds the 2 A safety limit."
        )
    if stage == "fixed_cell":
        start_cell = np.asarray(start_atoms.cell.array, dtype=float)
        if not bool(np.allclose(cell, start_cell, atol=1e-8, rtol=0.0)):
            raise Step10SafetyError(
                f"{potential}/{phase}/{stage}: the cell changed during a "
                "fixed-cell minimization."
            )

    force_vectors = forces[:, 5:8]
    force_magnitudes = np.linalg.norm(force_vectors, axis=1)
    maximum_force = float(np.max(force_magnitudes))
    rms_force = float(np.sqrt(np.mean(np.square(force_magnitudes))))
    thermo_fmax = _finite(thermo_row["c_fmax"], "thermo c_fmax")
    if not math.isclose(maximum_force, thermo_fmax, abs_tol=1e-6, rel_tol=1e-6):
        raise Step10ParseError(
            f"{potential}/{phase}/{stage}: dump-derived maximum force "
            f"{maximum_force:.12g} disagrees with thermo c_fmax "
            f"{thermo_fmax:.12g}."
        )
    pressure_records = pressure_row_to_records(thermo_row)
    energy = _finite(thermo_row["PotEng"], "potential energy")
    thermo_volume = _finite(thermo_row["Volume"], "thermo volume")
    if not math.isclose(thermo_volume, volume, abs_tol=1e-6, rel_tol=1e-9):
        raise Step10ParseError(
            f"{potential}/{phase}/{stage}: thermo volume disagrees with the "
            "read-back structure volume."
        )

    force_converged = maximum_force <= config.force_threshold_eV_per_A
    pressure_converged = (
        pressure_records["maximum_absolute_pressure_bar"]
        <= config.pressure_threshold_bar
    )
    if stage == "initial":
        convergence_status = "COMPLETED"
        scientifically_converged = True
    elif stage == "fixed_cell":
        scientifically_converged = force_converged
        convergence_status = "CONVERGED" if force_converged else "NOT_CONVERGED"
    else:
        scientifically_converged = force_converged and pressure_converged
        convergence_status = (
            "CONVERGED" if scientifically_converged else "NOT_CONVERGED"
        )

    al_count = sum(1 for symbol in symbols if symbol == "Al")
    ni_count = sum(1 for symbol in symbols if symbol == "Ni")
    lengths = [float(value) for value in final_atoms.cell.lengths()]
    angles = [float(value) for value in final_atoms.cell.angles()]
    symmetry = analyze_symmetry_of_atoms(
        config, final_atoms, f"{potential}/{phase}/{stage}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_state_result",
        "project_step": "10",
        "potential_key": potential,
        "potential_role": config.potentials[potential].role,
        "potential_path": relative_path(
            config.potentials[potential].path, config.project_root
        ),
        "potential_sha256": potential_sha256,
        "phase": phase,
        "material_id": spec.material_id,
        "formula": str(final_atoms.symbols),
        "stage": stage,
        "atom_count": spec.expected_atoms,
        "al_count": al_count,
        "ni_count": ni_count,
        "total_energy_eV": energy,
        "energy_per_atom_eV": energy / spec.expected_atoms,
        "force_vectors_eV_per_A": [
            [float(value) for value in row] for row in force_vectors
        ],
        "maximum_force_eV_per_A": maximum_force,
        "rms_force_eV_per_A": rms_force,
        **pressure_records,
        "volume_A3": volume,
        "volume_per_atom_A3": volume / spec.expected_atoms,
        "cell_A": [[float(value) for value in row] for row in cell],
        "lattice_lengths_A": lengths,
        "lattice_angles_deg": angles,
        "volume_change_percent_vs_original": volume_change_percent,
        "maximum_internal_displacement_A": maximum_displacement,
        "rms_internal_displacement_A": rms_displacement,
        "symmetry": symmetry,
        "cycles": [dict(cycle) for cycle in cycles],
        "minimizer_iterations_total": sum(
            int(cycle.get("iterations") or 0) for cycle in cycles
        ),
        "force_evaluations_total": sum(
            int(cycle.get("force_evaluations") or 0) for cycle in cycles
        ),
        "minimizer_stop_reason": (
            cycles[-1].get("stop_reason") if cycles else None
        ),
        "lammps_loop_time_seconds": sum(
            float(cycle.get("loop_time_seconds") or 0.0) for cycle in cycles
        ),
        "force_converged": force_converged,
        "pressure_converged": pressure_converged,
        "convergence_status": convergence_status,
        "scientifically_converged": scientifically_converged,
        "convergence_note": (
            "Scientific convergence checked independently from the parsed "
            "maximum per-atom force and the six pressure components; LAMMPS "
            "minimizer termination alone is never treated as convergence."
        ),
        "safety_status": "PASS",
        "wall_time_seconds": wall_time_seconds,
        "lammps_executable": executable.path,
        "lammps_version": executable.version_line,
        "configuration_fingerprint_sha256": config.fingerprint,
        "source_data_sha256": source_data_sha256,
        "timestamps": dict(timestamps),
    }


def validate_state_checkpoint(
    config: Step10Config,
    potential: str,
    potential_sha256: str,
    phase: str,
    stage: str,
    executable: LammpsExecutable | None = None,
) -> Mapping[str, Any]:
    """Strictly validate one published stage checkpoint for reuse/analysis."""

    path = stage_checkpoint_path(config, potential, phase, stage)
    if not path.is_file():
        raise Step10ResumeError(
            f"{potential}/{phase}/{stage}: checkpoint is missing: {path}"
        )
    try:
        record = read_strict_json(path, f"{potential}/{phase}/{stage} checkpoint")
    except Step7Error as exc:
        raise Step10ResumeError(str(exc)) from exc
    if (
        record.get("artifact_type") != "ni_al_lammps_state_result"
        or record.get("potential_key") != potential
        or record.get("phase") != phase
        or record.get("stage") != stage
        or record.get("configuration_fingerprint_sha256") != config.fingerprint
        or record.get("potential_sha256") != potential_sha256
        or record.get("safety_status") != "PASS"
    ):
        raise Step10ResumeError(
            f"{potential}/{phase}/{stage}: checkpoint identity/fingerprints "
            "do not match the current configuration."
        )
    if executable is not None and record.get("lammps_executable") != (
        executable.path
    ):
        raise Step10ResumeError(
            f"{potential}/{phase}/{stage}: checkpoint was produced by a "
            "different LAMMPS executable; explicit revalidation is required."
        )
    status = record.get("convergence_status")
    if stage == "initial":
        if status != "COMPLETED":
            raise Step10ResumeError(
                f"{potential}/{phase}/{stage}: initial state is not COMPLETED."
            )
    elif status != "CONVERGED":
        raise Step10ResumeError(
            f"{potential}/{phase}/{stage}: status {status!r} is not "
            "resume-eligible."
        )
    energy = _finite(record.get("total_energy_eV"), "checkpoint energy")
    per_atom = _finite(record.get("energy_per_atom_eV"), "checkpoint e/atom")
    atoms = record.get("atom_count")
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms <= 0:
        raise Step10ResumeError(f"{potential}/{phase}/{stage}: bad atom count.")
    if not math.isclose(per_atom, energy / atoms, abs_tol=1e-12, rel_tol=1e-12):
        raise Step10ResumeError(
            f"{potential}/{phase}/{stage}: energy bookkeeping is inconsistent."
        )
    final_extxyz = stage_dir(config, potential, phase, stage) / "final.extxyz"
    if not final_extxyz.is_file():
        raise Step10ResumeError(
            f"{potential}/{phase}/{stage}: final EXTXYZ is missing."
        )
    return record


__all__ = [
    "ASE_FROM_LAMMPS",
    "ASE_STRESS_COMPONENTS",
    "BAR_PER_EV_A3",
    "COMPOUND_ORDER",
    "ELEMENT_KEYS",
    "LAMMPS_PRESSURE_COMPONENTS",
    "PILOT_PHASE",
    "PILOT_POTENTIAL",
    "POTENTIAL_ORDER",
    "SCHEMA_VERSION",
    "STATE_ORDER",
    "STRUCTURE_ORDER",
    "THERMO_COLUMNS",
    "LammpsExecutable",
    "PotentialSpec",
    "Step10CalculationError",
    "Step10CollisionError",
    "Step10Config",
    "Step10ConfigurationError",
    "Step10ConversionError",
    "Step10DependencyError",
    "Step10Error",
    "Step10ExecutionError",
    "Step10InputError",
    "Step10ParseError",
    "Step10ResumeError",
    "Step10SafetyError",
    "StructureSpec",
    "analyze_symmetry_of_atoms",
    "build_lammps_input",
    "build_state_record",
    "check_log_for_errors",
    "conversion_paths",
    "discover_lammps",
    "extract_final_thermo",
    "load_source_structure",
    "load_step10_config",
    "parse_force_dump",
    "parse_minimization_stats",
    "parse_thermo_sections",
    "pressure_row_to_records",
    "read_lammps_structure",
    "run_phase_dir",
    "stage_checkpoint_path",
    "stage_dir",
    "stage_report_path",
    "validate_converted_bundle",
    "validate_selection",
    "validate_state_checkpoint",
    "validate_step9_success",
    "wrapped_displacement_statistics",
]

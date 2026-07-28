"""Shared utilities for the Ni-Al Step 8 MACE-versus-MP-DFT benchmark.

Step 8 compares the validated Step 7 relaxed MACE formation energies and
Step 6 MACE-relaxed structures against Materials Project DFT-derived
processed thermodynamic and structural reference data.  This module
provides:

``load_step8_config``
    Strict validation of ``configs/mace_dft_benchmark.json``.
``validate_step7_sources``
    Read-only validation of the Step 7/6 machine-readable MACE results.
``load_benchmark_records``
    Validation of the retrieved raw Materials Project benchmark bundles.
``calculate_comparisons`` / ``calculate_statistics``
    Formation-energy error records and dataset-level statistics.
``calculate_structural_comparisons``
    Volume, density, symmetry, and standardized-lattice comparison.

No function here queries Materials Project, loads MACE, creates an
optimizer, performs DFT, or modifies Step 6 or Step 7 data.  Raw MACE and
VASP total energies are never compared; the benchmark energy is only the
Materials Project processed ``formation_energy_per_atom``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from step6_utils import FileSnapshot
from step7_utils import (
    PHASE_ORDER,
    Step7Error,
    read_strict_json,
    relative_path,
    snapshot_file,
    verify_snapshots,
)


LOGGER = logging.getLogger("step8_utils")

SCHEMA_VERSION = "1.0"
EXPECTED_MATERIAL_IDS: Mapping[str, str] = {
    "Al3Ni": "mp-622209",
    "Al3Ni2": "mp-1057",
    "AlNi": "mp-1487",
    "Al3Ni5": "mp-16514",
    "AlNi3": "mp-2593",
}
EXPECTED_MODEL: Mapping[str, Any] = {
    "family": "MACE",
    "name": "MACE-MP-0",
    "value": "small",
    "device": "cpu",
    "default_dtype": "float64",
    "dispersion": False,
}
THERMO_TYPE_PREFERENCE: tuple[str, ...] = (
    "GGA_GGA+U_R2SCAN",
    "GGA_GGA+U",
    "R2SCAN",
)

BENCHMARK_LIMITATIONS: tuple[str, ...] = (
    "Materials Project values are processed DFT-derived reference data "
    "under the Materials Project correction and mixing scheme, not "
    "experimental truth and not raw uncorrected DFT total energies.",
    "This five-phase benchmark is not a complete Ni-Al phase diagram.",
    "Materials Project energy above hull is computed against the full "
    "Materials Project Ni-Al entry set and is not directly comparable to "
    "the Step 7 selected-set MACE envelope.",
    "Raw MACE and VASP total energies are never compared.",
)


class Step8Error(RuntimeError):
    """Base class for controlled Step 8 failures."""


class Step8ConfigurationError(Step8Error):
    """Raised when the Step 8 configuration or command scope is unsafe."""


class Step8DependencyError(Step8Error):
    """Raised when a required installed public API is unavailable."""


class Step8ApiError(Step8Error):
    """Raised for Materials Project access, key, or retrieval failures."""


class Step8InputError(Step8Error):
    """Raised when a protected scientific input is invalid."""


class Step8CalculationError(Step8Error):
    """Raised when a comparison or statistical calculation fails."""


class Step8CollisionError(Step8Error):
    """Raised when output collision handling refuses publication."""


class Step8ResumeError(Step8Error):
    """Raised when an existing Step 8 bundle is not safe to reuse."""


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Step8ConfigurationError(f"{label} must be a nonempty string.")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> Any:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise Step8ConfigurationError(
            f"{label} must be exactly {expected!r}; received {value!r}."
        )
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Step8ConfigurationError(f"{label} must be a JSON object.")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step8InputError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise Step8InputError(f"{label} is NaN or infinity.")
    return result


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise Step8DependencyError(f"NumPy is unavailable: {exc}") from exc
    return np


def _relative_repo_path(value: Any, label: str, project_root: Path) -> Path:
    text = _require_string(value, label)
    candidate = (project_root / Path(text)).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step8ConfigurationError(
            f"{label} must remain inside the repository: {text}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class ComparisonSettings:
    """Validated comparison controls."""

    primary_mace_energy_field: str
    initial_mace_energy_field: str
    include_initial_mace_diagnostic: bool
    symmetry_symprec_A: float
    symmetry_angle_tolerance_deg: float
    numerical_tolerance: float
    error_threshold_bins_eV_per_atom: tuple[float, ...]
    relative_error_minimum_denominator_eV_per_atom: float


@dataclass(frozen=True)
class MaceSources:
    """Validated machine-readable Step 6/7 MACE source locations."""

    formation_energy_table: Path
    formation_energy_configuration: Path
    full_cell_summary: Path
    full_cell_checkpoint_directory: Path
    full_cell_structure_directory: Path
    step7_report: Path


@dataclass(frozen=True)
class Step8Config:
    """Fully validated Step 8 configuration."""

    project_root: Path
    config_path: Path
    config_snapshot: FileSnapshot
    fingerprint: str
    raw: Mapping[str, Any]
    api_key_environment_variable: str
    require_non_deprecated: bool
    phases: Mapping[str, str]
    mace_sources: MaceSources
    comparison: ComparisonSettings
    raw_benchmark_root: Path
    result_root: Path


def load_step8_config(config_path: Path | str) -> Step8Config:
    """Load and strictly validate the Step 8 benchmark configuration."""

    from step7_utils import locate_project_root

    project_root = locate_project_root()
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step8ConfigurationError(
            f"Configuration must remain inside the repository: {resolved}"
        ) from exc
    try:
        raw = read_strict_json(resolved, "Step 8 benchmark configuration")
    except Step7Error as exc:
        raise Step8ConfigurationError(str(exc)) from exc

    _require_exact(raw.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_string(raw.get("description"), "description")
    source = _require_mapping(raw.get("benchmark_source"), "benchmark_source")
    _require_exact(
        source.get("provider"), "Materials Project", "benchmark_source.provider"
    )
    _require_exact(
        source.get("api_key_environment_variable"),
        "MP_API_KEY",
        "benchmark_source.api_key_environment_variable",
    )
    _require_exact(
        source.get("require_non_deprecated"),
        True,
        "benchmark_source.require_non_deprecated",
    )
    _require_exact(
        source.get("record_database_version"),
        True,
        "benchmark_source.record_database_version",
    )

    phases_raw = _require_mapping(raw.get("phases"), "phases")
    if tuple(phases_raw.keys()) != PHASE_ORDER:
        raise Step8ConfigurationError(
            f"phases must define exactly {list(PHASE_ORDER)} in order."
        )
    phases: dict[str, str] = {}
    for phase, spec in phases_raw.items():
        mapping = _require_mapping(spec, f"phases.{phase}")
        phases[phase] = _require_exact(
            mapping.get("material_id"),
            EXPECTED_MATERIAL_IDS[phase],
            f"phases.{phase}.material_id",
        )

    sources_raw = _require_mapping(raw.get("mace_sources"), "mace_sources")
    mace_sources = MaceSources(
        formation_energy_table=_relative_repo_path(
            _require_exact(
                sources_raw.get("formation_energy_table"),
                "results/mace_formation_energy/tables/"
                "ni_al_mace_formation_energies.json",
                "mace_sources.formation_energy_table",
            ),
            "mace_sources.formation_energy_table",
            project_root,
        ),
        formation_energy_configuration=_relative_repo_path(
            _require_exact(
                sources_raw.get("formation_energy_configuration"),
                "configs/mace_formation_energy.json",
                "mace_sources.formation_energy_configuration",
            ),
            "mace_sources.formation_energy_configuration",
            project_root,
        ),
        full_cell_summary=_relative_repo_path(
            _require_exact(
                sources_raw.get("full_cell_summary"),
                "results/mace_relaxation/full_cell/tables/"
                "ni_al_full_cell_summary.json",
                "mace_sources.full_cell_summary",
            ),
            "mace_sources.full_cell_summary",
            project_root,
        ),
        full_cell_checkpoint_directory=_relative_repo_path(
            _require_exact(
                sources_raw.get("full_cell_checkpoint_directory"),
                "results/mace_relaxation/full_cell/checkpoints",
                "mace_sources.full_cell_checkpoint_directory",
            ),
            "mace_sources.full_cell_checkpoint_directory",
            project_root,
        ),
        full_cell_structure_directory=_relative_repo_path(
            _require_exact(
                sources_raw.get("full_cell_structure_directory"),
                "results/mace_relaxation/full_cell/structures",
                "mace_sources.full_cell_structure_directory",
            ),
            "mace_sources.full_cell_structure_directory",
            project_root,
        ),
        step7_report=_relative_repo_path(
            _require_exact(
                sources_raw.get("step7_report"),
                "results/mace_formation_energy/reports/"
                "ni_al_step7_final_report.txt",
                "mace_sources.step7_report",
            ),
            "mace_sources.step7_report",
            project_root,
        ),
    )

    comparison_raw = _require_mapping(raw.get("comparison"), "comparison")
    comparison = ComparisonSettings(
        primary_mace_energy_field=_require_exact(
            comparison_raw.get("primary_mace_energy_field"),
            "relaxed_formation_energy_eV_per_atom",
            "comparison.primary_mace_energy_field",
        ),
        initial_mace_energy_field=_require_exact(
            comparison_raw.get("initial_mace_energy_field"),
            "initial_formation_energy_eV_per_atom",
            "comparison.initial_mace_energy_field",
        ),
        include_initial_mace_diagnostic=_require_exact(
            comparison_raw.get("include_initial_mace_diagnostic"),
            True,
            "comparison.include_initial_mace_diagnostic",
        ),
        symmetry_symprec_A=_require_exact(
            comparison_raw.get("symmetry_symprec_A"),
            0.001,
            "comparison.symmetry_symprec_A",
        ),
        symmetry_angle_tolerance_deg=_require_exact(
            comparison_raw.get("symmetry_angle_tolerance_deg"),
            5.0,
            "comparison.symmetry_angle_tolerance_deg",
        ),
        numerical_tolerance=_require_exact(
            comparison_raw.get("numerical_tolerance"),
            1e-12,
            "comparison.numerical_tolerance",
        ),
        error_threshold_bins_eV_per_atom=tuple(
            _require_exact(
                comparison_raw.get("error_threshold_bins_eV_per_atom"),
                [0.05, 0.1, 0.2],
                "comparison.error_threshold_bins_eV_per_atom",
            )
        ),
        relative_error_minimum_denominator_eV_per_atom=_require_exact(
            comparison_raw.get(
                "relative_error_minimum_denominator_eV_per_atom"
            ),
            0.05,
            "comparison.relative_error_minimum_denominator_eV_per_atom",
        ),
    )

    output_raw = _require_mapping(raw.get("output"), "output")
    raw_benchmark_root = _relative_repo_path(
        _require_exact(
            output_raw.get("raw_benchmark_root"),
            "data/raw/materials_project/dft_benchmark",
            "output.raw_benchmark_root",
        ),
        "output.raw_benchmark_root",
        project_root,
    )
    result_root = _relative_repo_path(
        _require_exact(
            output_raw.get("result_root"),
            "results/mace_vs_dft",
            "output.result_root",
        ),
        "output.result_root",
        project_root,
    )

    snapshot = snapshot_file(resolved, "Step 8 configuration")
    return Step8Config(
        project_root=project_root,
        config_path=resolved,
        config_snapshot=snapshot,
        fingerprint=snapshot.sha256,
        raw=raw,
        api_key_environment_variable="MP_API_KEY",
        require_non_deprecated=True,
        phases=phases,
        mace_sources=mace_sources,
        comparison=comparison,
        raw_benchmark_root=raw_benchmark_root,
        result_root=result_root,
    )


def validate_phase_keys(phases: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a requested phase selection deterministically."""

    if phases is None:
        return PHASE_ORDER
    selected = tuple(phases)
    if not selected:
        raise Step8ConfigurationError("At least one phase must be selected.")
    if len(set(selected)) != len(selected):
        raise Step8ConfigurationError("Requested phases contain duplicates.")
    unknown = [phase for phase in selected if phase not in PHASE_ORDER]
    if unknown:
        raise Step8ConfigurationError("Unknown phase(s): " + ", ".join(unknown))
    return tuple(phase for phase in PHASE_ORDER if phase in selected)


# ---------------------------------------------------------------------------
# Output-path resolution
# ---------------------------------------------------------------------------


def benchmark_phase_paths(config: Step8Config, phase: str) -> tuple[Path, Path, Path]:
    """Return (metadata, cif, extxyz) targets for one raw benchmark bundle."""

    directory = config.raw_benchmark_root / phase
    return (
        directory / "metadata.json",
        directory / "structure.cif",
        directory / "structure.extxyz",
    )


@dataclass(frozen=True)
class Step8OutputPaths:
    """Canonical Step 8 comparison output targets."""

    energy_csv: Path
    energy_json: Path
    structural_csv: Path
    structural_json: Path
    comparison_report: Path
    final_report: Path
    checkpoint: Path
    parity_figure: Path
    error_figure: Path
    composition_figure: Path
    initial_vs_relaxed_figure: Path
    volume_figure: Path
    volume_error_figure: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every comparison target in deterministic order."""

        return (
            self.energy_csv,
            self.energy_json,
            self.structural_csv,
            self.structural_json,
            self.comparison_report,
            self.final_report,
            self.checkpoint,
            self.parity_figure,
            self.error_figure,
            self.composition_figure,
            self.initial_vs_relaxed_figure,
            self.volume_figure,
            self.volume_error_figure,
        )


def step8_output_paths(config: Step8Config) -> Step8OutputPaths:
    """Resolve every canonical Step 8 comparison target."""

    root = config.result_root
    return Step8OutputPaths(
        energy_csv=root / "tables" / "ni_al_mace_vs_mp_dft.csv",
        energy_json=root / "tables" / "ni_al_mace_vs_mp_dft.json",
        structural_csv=root / "tables" / "ni_al_structural_comparison.csv",
        structural_json=root / "tables" / "ni_al_structural_comparison.json",
        comparison_report=root / "reports" / "ni_al_mace_vs_mp_dft_report.txt",
        final_report=root / "reports" / "ni_al_step8_final_report.txt",
        checkpoint=root / "checkpoints" / "step8_benchmark_result.json",
        parity_figure=root / "figures" / "formation_energy_parity_plot.png",
        error_figure=root / "figures" / "formation_energy_error_by_phase.png",
        composition_figure=(
            root / "figures" / "mace_vs_dft_formation_energy_by_composition.png"
        ),
        initial_vs_relaxed_figure=(
            root / "figures" / "initial_vs_relaxed_mace_error.png"
        ),
        volume_figure=root / "figures" / "volume_per_atom_mace_vs_mp.png",
        volume_error_figure=(
            root / "figures" / "volume_percent_error_by_phase.png"
        ),
    )


def step8_directories(config: Step8Config) -> tuple[Path, ...]:
    """Return every Step 8 result directory execution may populate."""

    root = config.result_root
    return (
        root,
        root / "tables",
        root / "reports",
        root / "figures",
        root / "checkpoints",
    )


# ---------------------------------------------------------------------------
# MACE (Step 6/7) source validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaceRecord:
    """Validated Step 7/6 MACE result for one phase."""

    phase_key: str
    material_id: str
    reduced_formula: str
    cell_al_count: int
    cell_ni_count: int
    cell_atom_count: int
    ni_fraction: float
    initial_formation_energy_eV_per_atom: float
    relaxed_formation_energy_eV_per_atom: float
    relaxation_effect_eV_per_atom: float
    on_selected_set_envelope: bool
    compound_convergence_status: str
    elemental_convergence_status: str
    safety_status: str
    final_volume_A3: float
    final_volume_per_atom_A3: float
    volume_change_percent: float
    provenance_path: str
    relaxed_structure_path: Path


@dataclass(frozen=True)
class MaceSourceBundle:
    """Complete validated Step 7/6 MACE inputs."""

    records: Mapping[str, MaceRecord]
    model: Mapping[str, Any]
    chemical_potentials: Mapping[str, float]
    step7_database_version: Mapping[str, Any]
    step7_configuration_fingerprint: str
    snapshots: tuple[FileSnapshot, ...]


def validate_step7_sources(config: Step8Config) -> MaceSourceBundle:
    """Validate every machine-readable Step 7/6 MACE source completely."""

    sources = config.mace_sources
    snapshots: list[FileSnapshot] = []
    for path, label in (
        (sources.formation_energy_table, "Step 7 formation-energy table"),
        (sources.formation_energy_configuration, "Step 7 configuration"),
        (sources.full_cell_summary, "Step 6 full-cell summary"),
        (sources.step7_report, "Step 7 final report"),
    ):
        snapshots.append(snapshot_file(path, label))

    report_text = sources.step7_report.read_text(encoding="utf-8")
    if "OVERALL STEP 7 STATUS: SUCCESS" not in report_text:
        raise Step8InputError(
            "The Step 7 final report does not record OVERALL STEP 7 STATUS: "
            "SUCCESS; Step 8 cannot benchmark unfinished Step 7 results."
        )

    document = read_strict_json(
        sources.formation_energy_table, "Step 7 formation-energy table"
    )
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("artifact_type") != "ni_al_mace_formation_energies"
    ):
        raise Step8InputError(
            "Step 7 formation-energy table identity is invalid."
        )
    step7_config_snapshot = snapshot_file(
        sources.formation_energy_configuration, "Step 7 configuration"
    )
    if (
        document.get("configuration_fingerprint_sha256")
        != step7_config_snapshot.sha256
    ):
        raise Step8InputError(
            "The Step 7 formation-energy table was produced by a different "
            "Step 7 configuration than the one currently on disk."
        )
    model = document.get("model")
    if not isinstance(model, Mapping) or any(
        model.get(key) != value for key, value in EXPECTED_MODEL.items()
    ):
        raise Step8InputError(
            "Step 7 MACE model settings are not MACE-MP-0 Small on CPU with "
            "float64 and dispersion disabled."
        )
    potentials = document.get("chemical_potentials_eV_per_atom")
    if not isinstance(potentials, Mapping):
        raise Step8InputError("Step 7 chemical potentials are absent.")
    mu_al = _finite_float(potentials.get("mu_Al_MACE"), "mu_Al_MACE")
    mu_ni = _finite_float(potentials.get("mu_Ni_MACE"), "mu_Ni_MACE")
    database_version = document.get("materials_project_database_version")
    if not isinstance(database_version, Mapping):
        raise Step8InputError("Step 7 database-version record is absent.")

    summary = read_strict_json(
        sources.full_cell_summary, "Step 6 full-cell summary"
    )
    if summary.get("overall_status") != "SUCCESS":
        raise Step8InputError("Step 6 full-cell summary is not SUCCESS.")
    summary_records = {
        record.get("phase_key"): record
        for record in summary.get("records", ())
        if isinstance(record, Mapping)
    }

    table_records = {
        record.get("phase_key"): record
        for record in document.get("records", ())
        if isinstance(record, Mapping)
    }
    records: dict[str, MaceRecord] = {}
    for phase in PHASE_ORDER:
        row = table_records.get(phase)
        summary_row = summary_records.get(phase)
        if row is None:
            raise Step8InputError(
                f"Step 7 table has no record for {phase}."
            )
        if summary_row is None:
            raise Step8InputError(
                f"Step 6 summary has no record for {phase}."
            )
        material_id = row.get("material_id")
        if material_id != config.phases[phase]:
            raise Step8InputError(
                f"{phase} material ID mismatch between Step 7 and Step 8 "
                f"configuration: {material_id!r}."
            )
        for count_field in ("cell_al_count", "cell_ni_count", "cell_atom_count"):
            value = row.get(count_field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Step8InputError(f"{phase} {count_field} is invalid.")
        al_count = int(row["cell_al_count"])
        ni_count = int(row["cell_ni_count"])
        atom_count = int(row["cell_atom_count"])
        if al_count + ni_count != atom_count:
            raise Step8InputError(f"{phase} composition bookkeeping is invalid.")
        ni_fraction = _finite_float(
            row.get("ni_atomic_fraction"), f"{phase} Ni fraction"
        )
        if not math.isclose(
            ni_fraction, ni_count / atom_count, abs_tol=1e-12, rel_tol=0.0
        ):
            raise Step8InputError(f"{phase} Ni fraction is inconsistent.")
        initial = _finite_float(
            row.get(config.comparison.initial_mace_energy_field),
            f"{phase} initial MACE formation energy",
        )
        relaxed = _finite_float(
            row.get(config.comparison.primary_mace_energy_field),
            f"{phase} relaxed MACE formation energy",
        )
        status = row.get("compound_convergence_status")
        if status not in {"CONVERGED", "ALREADY_CONVERGED"}:
            raise Step8InputError(
                f"{phase} compound convergence status {status!r} is invalid."
            )
        if row.get("safety_status") != "PASS":
            raise Step8InputError(f"{phase} safety status is not PASS.")
        elemental_status = row.get("elemental_convergence_status")
        if not isinstance(elemental_status, str) or not all(
            part.split("=")[-1] in {"CONVERGED", "ALREADY_CONVERGED"}
            for part in elemental_status.split("; ")
        ):
            raise Step8InputError(
                f"{phase} elemental reference status is invalid: "
                f"{elemental_status!r}."
            )
        provenance = row.get("compound_checkpoint_path")
        provenance_path = config.project_root / str(provenance)
        if not provenance_path.is_file():
            raise Step8InputError(
                f"{phase} provenance checkpoint is missing: {provenance!r}."
            )
        snapshots.append(
            snapshot_file(provenance_path, f"{phase} full-cell checkpoint")
        )
        structure_path = (
            sources.full_cell_structure_directory
            / f"{phase}_full_cell_relaxed.extxyz"
        )
        snapshots.append(
            snapshot_file(structure_path, f"{phase} MACE relaxed structure")
        )
        final_volume = _finite_float(
            summary_row.get("final_volume_A3"), f"{phase} final volume"
        )
        records[phase] = MaceRecord(
            phase_key=phase,
            material_id=str(material_id),
            reduced_formula=str(row.get("reduced_formula")),
            cell_al_count=al_count,
            cell_ni_count=ni_count,
            cell_atom_count=atom_count,
            ni_fraction=ni_fraction,
            initial_formation_energy_eV_per_atom=initial,
            relaxed_formation_energy_eV_per_atom=relaxed,
            relaxation_effect_eV_per_atom=_finite_float(
                row.get("relaxation_effect_eV_per_atom"),
                f"{phase} relaxation effect",
            ),
            on_selected_set_envelope=bool(row.get("on_selected_set_envelope")),
            compound_convergence_status=str(status),
            elemental_convergence_status=str(elemental_status),
            safety_status="PASS",
            final_volume_A3=final_volume,
            final_volume_per_atom_A3=final_volume / atom_count,
            volume_change_percent=_finite_float(
                summary_row.get("volume_change_percent"),
                f"{phase} volume change",
            ),
            provenance_path=str(provenance),
            relaxed_structure_path=structure_path,
        )
    verify_snapshots(snapshots)
    return MaceSourceBundle(
        records=records,
        model=dict(model),
        chemical_potentials={"mu_Al_MACE": mu_al, "mu_Ni_MACE": mu_ni},
        step7_database_version=dict(database_version),
        step7_configuration_fingerprint=step7_config_snapshot.sha256,
        snapshots=tuple(snapshots),
    )


# ---------------------------------------------------------------------------
# Benchmark record validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRecord:
    """Validated retrieved Materials Project benchmark data for one phase."""

    phase_key: str
    material_id: str
    formula_pretty: str
    reduced_formula: str
    composition: Mapping[str, int]
    number_of_sites: int
    formation_energy_per_atom_eV: float
    energy_above_hull_eV_per_atom: float
    is_stable: bool | None
    deprecated: bool | None
    theoretical: bool | None
    selected_thermo_type: str | None
    thermo_selection_rationale: str
    database_version: str | None
    retrieval_time_utc: str
    space_group_symbol: str | None
    space_group_number: int | None
    volume_A3: float
    volume_per_atom_A3: float
    density_g_cm3: float | None
    metadata_path: Path
    structure_path: Path


def load_benchmark_records(
    config: Step8Config, phases: Sequence[str] | None = None
) -> tuple[Mapping[str, BenchmarkRecord], tuple[FileSnapshot, ...]]:
    """Validate the retrieved raw benchmark bundles completely."""

    try:
        from ase.io import read as ase_read
        from pymatgen.core import Composition
    except ImportError as exc:
        raise Step8DependencyError(
            f"ASE/pymatgen support is unavailable: {exc}"
        ) from exc
    np = _numpy()
    selected = validate_phase_keys(phases)
    snapshots: list[FileSnapshot] = []
    records: dict[str, BenchmarkRecord] = {}
    for phase in selected:
        metadata_path, cif_path, extxyz_path = benchmark_phase_paths(
            config, phase
        )
        for path, label in (
            (metadata_path, f"{phase} benchmark metadata"),
            (cif_path, f"{phase} benchmark CIF"),
            (extxyz_path, f"{phase} benchmark EXTXYZ"),
        ):
            if not path.is_file():
                raise Step8InputError(
                    f"{label} does not exist: {path}. Run "
                    "fetch_ni_al_mp_dft_benchmarks.py --fetch first."
                )
            snapshots.append(snapshot_file(path, label))
        try:
            metadata = read_strict_json(
                metadata_path, f"{phase} benchmark metadata"
            )
        except Step7Error as exc:
            raise Step8InputError(str(exc)) from exc
        if metadata.get("phase") != phase:
            raise Step8InputError(f"{phase} metadata phase mismatch.")
        material_id = metadata.get("material_id")
        if material_id != config.phases[phase]:
            raise Step8InputError(
                f"{phase} benchmark material ID {material_id!r} does not "
                f"match the configured {config.phases[phase]!r}."
            )
        if config.require_non_deprecated and metadata.get("deprecated") is True:
            raise Step8InputError(f"{phase} benchmark record is deprecated.")
        formation = _finite_float(
            metadata.get("formation_energy_per_atom_eV"),
            f"{phase} MP formation energy per atom",
        )
        hull = _finite_float(
            metadata.get("energy_above_hull_eV_per_atom"),
            f"{phase} MP energy above hull",
        )
        if hull < -1e-8:
            raise Step8InputError(
                f"{phase} MP energy above hull is negative: {hull!r}."
            )
        extxyz_sha = metadata.get("structure_extxyz_sha256")
        from step7_utils import file_sha256

        if extxyz_sha != file_sha256(extxyz_path):
            raise Step8InputError(
                f"{phase} benchmark EXTXYZ fingerprint does not match its "
                "metadata."
            )
        try:
            frames = ase_read(extxyz_path, index=":", format="extxyz")
        except Exception as exc:
            raise Step8InputError(
                f"Could not read {phase} benchmark structure: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(frames, list) or len(frames) != 1:
            raise Step8InputError(
                f"{phase} benchmark EXTXYZ must contain one frame."
            )
        atoms = frames[0]
        atoms.calc = None
        symbols = tuple(atoms.get_chemical_symbols())
        if not symbols or set(symbols) - {"Al", "Ni"}:
            raise Step8InputError(
                f"{phase} benchmark structure has non-Al/Ni species."
            )
        composition = Composition("".join(symbols))
        expected = Composition(phase)
        if composition.reduced_composition != expected.reduced_composition:
            raise Step8InputError(
                f"{phase} benchmark reduced composition "
                f"{composition.reduced_formula!r} does not match the phase."
            )
        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = np.asarray(atoms.cell.array, dtype=float)
        determinant = float(np.linalg.det(cell))
        volume = float(atoms.get_volume())
        if (
            not bool(np.all(np.isfinite(positions)))
            or not bool(np.all(np.isfinite(cell)))
            or not math.isfinite(determinant)
            or determinant <= 0.0
            or not math.isfinite(volume)
            or volume <= 0.0
            or not bool(np.all(np.asarray(atoms.get_pbc(), dtype=bool)))
        ):
            raise Step8InputError(
                f"{phase} benchmark structure geometry is invalid."
            )
        symbol = metadata.get("space_group_symbol")
        number = metadata.get("space_group_number")
        counts = {
            element: sum(1 for item in symbols if item == element)
            for element in ("Al", "Ni")
        }
        records[phase] = BenchmarkRecord(
            phase_key=phase,
            material_id=str(material_id),
            formula_pretty=str(metadata.get("formula_pretty")),
            reduced_formula=composition.reduced_formula,
            composition=counts,
            number_of_sites=len(symbols),
            formation_energy_per_atom_eV=formation,
            energy_above_hull_eV_per_atom=hull,
            is_stable=(
                metadata.get("is_stable")
                if isinstance(metadata.get("is_stable"), bool)
                else None
            ),
            deprecated=(
                metadata.get("deprecated")
                if isinstance(metadata.get("deprecated"), bool)
                else None
            ),
            theoretical=(
                metadata.get("theoretical")
                if isinstance(metadata.get("theoretical"), bool)
                else None
            ),
            selected_thermo_type=(
                metadata.get("selected_thermo_type")
                if isinstance(metadata.get("selected_thermo_type"), str)
                else None
            ),
            thermo_selection_rationale=str(
                metadata.get("thermo_selection_rationale", "")
            ),
            database_version=(
                str(metadata.get("materials_project_database_version"))
                if metadata.get("materials_project_database_version")
                is not None
                else None
            ),
            retrieval_time_utc=str(metadata.get("retrieval_time_utc")),
            space_group_symbol=symbol if isinstance(symbol, str) else None,
            space_group_number=(
                number
                if isinstance(number, int) and not isinstance(number, bool)
                else None
            ),
            volume_A3=volume,
            volume_per_atom_A3=volume / len(symbols),
            density_g_cm3=(
                float(metadata.get("density_g_cm3"))
                if isinstance(metadata.get("density_g_cm3"), (int, float))
                and not isinstance(metadata.get("density_g_cm3"), bool)
                else None
            ),
            metadata_path=metadata_path,
            structure_path=extxyz_path,
        )
    verify_snapshots(snapshots)
    return records, tuple(snapshots)


# ---------------------------------------------------------------------------
# Formation-energy comparison and statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRecord:
    """Per-phase formation-energy comparison result."""

    phase_key: str
    material_id: str
    reduced_formula: str
    ni_fraction: float
    mp_database_version: str | None
    mp_thermo_type: str | None
    mp_formation_energy_eV_per_atom: float
    mp_energy_above_hull_eV_per_atom: float
    mp_is_stable: bool | None
    mace_initial_formation_energy_eV_per_atom: float
    mace_relaxed_formation_energy_eV_per_atom: float
    initial_signed_error_eV_per_atom: float
    relaxed_signed_error_eV_per_atom: float
    relaxed_absolute_error_eV_per_atom: float
    squared_error_eV2_per_atom2: float
    relaxation_effect_on_benchmark_error_eV_per_atom: float
    diagnostic_relative_error_percent: float | None
    mace_on_selected_set_envelope: bool
    mace_convergence_status: str
    mace_safety_status: str
    benchmark_provenance_path: str
    mace_provenance_path: str


def calculate_comparisons(
    config: Step8Config,
    benchmarks: Mapping[str, BenchmarkRecord],
    mace: MaceSourceBundle,
) -> tuple[ComparisonRecord, ...]:
    """Calculate the per-phase formation-energy error records.

    The signed error convention is MACE minus Materials Project, so a
    positive relaxed signed error means MACE predicts a less negative
    (weaker) formation energy than the MP processed reference.
    """

    records: list[ComparisonRecord] = []
    minimum_denominator = (
        config.comparison.relative_error_minimum_denominator_eV_per_atom
    )
    for phase in PHASE_ORDER:
        benchmark = benchmarks.get(phase)
        mace_record = mace.records.get(phase)
        if benchmark is None or mace_record is None:
            raise Step8CalculationError(
                f"{phase} is missing from the benchmark or MACE inputs."
            )
        if benchmark.material_id != mace_record.material_id:
            raise Step8CalculationError(
                f"{phase} identity mismatch between benchmark and MACE data."
            )
        initial_error = (
            mace_record.initial_formation_energy_eV_per_atom
            - benchmark.formation_energy_per_atom_eV
        )
        relaxed_error = (
            mace_record.relaxed_formation_energy_eV_per_atom
            - benchmark.formation_energy_per_atom_eV
        )
        if abs(benchmark.formation_energy_per_atom_eV) >= minimum_denominator:
            relative = (
                100.0
                * relaxed_error
                / abs(benchmark.formation_energy_per_atom_eV)
            )
        else:
            # Percentage error is undefined near a zero denominator; the
            # diagnostic is omitted rather than silently misleading.
            relative = None
        records.append(
            ComparisonRecord(
                phase_key=phase,
                material_id=benchmark.material_id,
                reduced_formula=mace_record.reduced_formula,
                ni_fraction=mace_record.ni_fraction,
                mp_database_version=benchmark.database_version,
                mp_thermo_type=benchmark.selected_thermo_type,
                mp_formation_energy_eV_per_atom=(
                    benchmark.formation_energy_per_atom_eV
                ),
                mp_energy_above_hull_eV_per_atom=(
                    benchmark.energy_above_hull_eV_per_atom
                ),
                mp_is_stable=benchmark.is_stable,
                mace_initial_formation_energy_eV_per_atom=(
                    mace_record.initial_formation_energy_eV_per_atom
                ),
                mace_relaxed_formation_energy_eV_per_atom=(
                    mace_record.relaxed_formation_energy_eV_per_atom
                ),
                initial_signed_error_eV_per_atom=initial_error,
                relaxed_signed_error_eV_per_atom=relaxed_error,
                relaxed_absolute_error_eV_per_atom=abs(relaxed_error),
                squared_error_eV2_per_atom2=relaxed_error**2,
                relaxation_effect_on_benchmark_error_eV_per_atom=(
                    abs(relaxed_error) - abs(initial_error)
                ),
                diagnostic_relative_error_percent=relative,
                mace_on_selected_set_envelope=(
                    mace_record.on_selected_set_envelope
                ),
                mace_convergence_status=mace_record.compound_convergence_status,
                mace_safety_status=mace_record.safety_status,
                benchmark_provenance_path=relative_path(
                    benchmark.metadata_path, config.project_root
                ),
                mace_provenance_path=mace_record.provenance_path,
            )
        )
    return tuple(records)


def _rankdata(values: Sequence[float], np: Any) -> Any:
    """Average-tie ranks (1-based) without a SciPy dependency."""

    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    position = 0
    while position < len(array):
        end = position
        while (
            end + 1 < len(array)
            and array[order[end + 1]] == array[order[position]]
        ):
            end += 1
        average_rank = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average_rank
        position = end + 1
    return ranks


def calculate_statistics(
    config: Step8Config, records: Sequence[ComparisonRecord]
) -> dict[str, Any]:
    """Calculate dataset-level statistics across the five selected phases.

    The sample size is five compounds; correlation metrics are exploratory
    and the threshold counts are descriptive bins, not universal accuracy
    standards.
    """

    np = _numpy()
    if len(records) != len(PHASE_ORDER):
        raise Step8CalculationError(
            "Statistics require all five phase comparisons."
        )
    signed = np.asarray(
        [record.relaxed_signed_error_eV_per_atom for record in records]
    )
    absolute = np.abs(signed)
    mace_values = np.asarray(
        [
            record.mace_relaxed_formation_energy_eV_per_atom
            for record in records
        ]
    )
    dft_values = np.asarray(
        [record.mp_formation_energy_eV_per_atom for record in records]
    )
    phases = [record.phase_key for record in records]

    pearson: float | None = None
    spearman: float | None = None
    if float(np.std(mace_values)) > 0.0 and float(np.std(dft_values)) > 0.0:
        pearson = float(np.corrcoef(mace_values, dft_values)[0, 1])
        mace_ranks = _rankdata(mace_values, np)
        dft_ranks = _rankdata(dft_values, np)
        if float(np.std(mace_ranks)) > 0.0 and float(np.std(dft_ranks)) > 0.0:
            spearman = float(np.corrcoef(mace_ranks, dft_ranks)[0, 1])

    mace_order = [phases[index] for index in np.argsort(mace_values, kind="stable")]
    dft_order = [phases[index] for index in np.argsort(dft_values, kind="stable")]
    pair_total = 0
    pair_agreement = 0
    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            pair_total += 1
            mace_sign = math.copysign(
                1.0, mace_values[second] - mace_values[first]
            )
            dft_sign = math.copysign(1.0, dft_values[second] - dft_values[first])
            if mace_sign == dft_sign:
                pair_agreement += 1

    def trend_signs(values: Any) -> list[int]:
        ordered = sorted(
            zip((record.ni_fraction for record in records), values)
        )
        return [
            int(math.copysign(1.0, second[1] - first[1]))
            for first, second in zip(ordered, ordered[1:])
        ]

    thresholds = config.comparison.error_threshold_bins_eV_per_atom
    threshold_counts = {
        f"{threshold:g}": int(np.sum(absolute <= threshold + 1e-15))
        for threshold in thresholds
    }
    return {
        "sample_size": len(records),
        "sample_size_note": (
            "Only five selected compounds; all statistics are descriptive "
            "and correlation metrics are exploratory."
        ),
        "mean_signed_error_eV_per_atom": float(np.mean(signed)),
        "mean_absolute_error_eV_per_atom": float(np.mean(absolute)),
        "root_mean_squared_error_eV_per_atom": float(
            np.sqrt(np.mean(np.square(signed)))
        ),
        "median_absolute_error_eV_per_atom": float(np.median(absolute)),
        "maximum_absolute_error_eV_per_atom": float(np.max(absolute)),
        "phase_with_maximum_absolute_error": phases[int(np.argmax(absolute))],
        "phase_with_minimum_absolute_error": phases[int(np.argmin(absolute))],
        "standard_deviation_of_signed_errors_eV_per_atom": float(
            np.std(signed)
        ),
        "standard_deviation_note": "Population standard deviation over n=5.",
        "pearson_correlation": pearson,
        "spearman_rank_correlation": spearman,
        "correlation_note": (
            "Exploratory only: five data points cannot support strong "
            "statistical claims."
        ),
        "mace_ranking_most_negative_first": mace_order,
        "dft_ranking_most_negative_first": dft_order,
        "exact_ranking_agreement": mace_order == dft_order,
        "pairwise_ordering_agreement": f"{pair_agreement}/{pair_total}",
        "pairwise_ordering_agreement_fraction": pair_agreement / pair_total,
        "most_negative_phase_mace": mace_order[0],
        "most_negative_phase_dft": dft_order[0],
        "composition_trend_signs_match": (
            trend_signs(mace_values) == trend_signs(dft_values)
        ),
        "error_threshold_counts": threshold_counts,
        "error_threshold_note": (
            "Descriptive bins only, not universal accuracy standards."
        ),
        "all_signed_errors_positive": bool(np.all(signed > 0.0)),
        "all_signed_errors_negative": bool(np.all(signed < 0.0)),
    }


# ---------------------------------------------------------------------------
# Structural comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralRecord:
    """Per-phase structural comparison between MP DFT and MACE structures."""

    phase_key: str
    material_id: str
    mp_volume_per_atom_A3: float
    mace_volume_per_atom_A3: float
    signed_volume_per_atom_difference_A3: float
    absolute_volume_per_atom_difference_A3: float
    volume_per_atom_difference_percent: float
    mp_density_g_cm3: float | None
    mace_density_g_cm3: float | None
    mp_space_group_symbol: str
    mp_space_group_number: int
    mace_space_group_symbol: str
    mace_space_group_number: int
    symmetry_preserved: bool
    standardization_method: str
    lattice_comparison_available: bool
    lattice_comparison_note: str
    mp_lattice_abc_A: tuple[float, float, float] | None
    mace_lattice_abc_A: tuple[float, float, float] | None
    lattice_abc_differences_A: tuple[float, float, float] | None
    mp_lattice_angles_deg: tuple[float, float, float] | None
    mace_lattice_angles_deg: tuple[float, float, float] | None
    lattice_angle_differences_deg: tuple[float, float, float] | None


def _pymatgen_structure_from_extxyz(path: Path, label: str) -> Any:
    """Read one EXTXYZ frame and convert it to a pymatgen Structure."""

    try:
        from ase.io import read as ase_read
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise Step8DependencyError(
            f"ASE/pymatgen conversion support is unavailable: {exc}"
        ) from exc
    try:
        frames = ase_read(path, index=":", format="extxyz")
    except Exception as exc:
        raise Step8InputError(
            f"Could not read {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list) or len(frames) != 1:
        raise Step8InputError(f"{label} must contain exactly one frame.")
    atoms = frames[0]
    atoms.calc = None
    return AseAtomsAdaptor.get_structure(atoms)


def _symmetry_of(structure: Any, config: Step8Config, label: str) -> tuple[str, int, Any]:
    """Return (symbol, number, analyzer) for one structure."""

    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:
        raise Step8DependencyError(
            f"pymatgen symmetry analysis is unavailable: {exc}"
        ) from exc
    try:
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=config.comparison.symmetry_symprec_A,
            angle_tolerance=config.comparison.symmetry_angle_tolerance_deg,
        )
        return (
            analyzer.get_space_group_symbol(),
            int(analyzer.get_space_group_number()),
            analyzer,
        )
    except Exception as exc:
        raise Step8CalculationError(
            f"Symmetry analysis failed for {label}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def calculate_structural_comparisons(
    config: Step8Config,
    benchmarks: Mapping[str, BenchmarkRecord],
    mace: MaceSourceBundle,
) -> tuple[StructuralRecord, ...]:
    """Compare MP DFT-relaxed and MACE-relaxed structures per phase.

    Raw lattice matrices are never compared directly because the two
    sources may use different primitive or conventional representations.
    Both structures are standardized with the same documented pymatgen
    conventional-standard procedure before lattice parameters are compared,
    and the comparison is skipped safely when standardization is ambiguous.
    """

    standardization_method = (
        "pymatgen SpacegroupAnalyzer.get_conventional_standard_structure "
        f"(symprec={config.comparison.symmetry_symprec_A} A, "
        f"angle_tolerance={config.comparison.symmetry_angle_tolerance_deg} deg) "
        "applied identically to both structures"
    )
    records: list[StructuralRecord] = []
    for phase in PHASE_ORDER:
        benchmark = benchmarks[phase]
        mace_record = mace.records[phase]
        mp_structure = _pymatgen_structure_from_extxyz(
            benchmark.structure_path, f"{phase} MP benchmark structure"
        )
        mace_structure = _pymatgen_structure_from_extxyz(
            mace_record.relaxed_structure_path,
            f"{phase} MACE relaxed structure",
        )
        mp_volume_per_atom = float(mp_structure.volume) / len(mp_structure)
        mace_volume_per_atom = float(mace_structure.volume) / len(
            mace_structure
        )
        if not math.isclose(
            mace_volume_per_atom,
            mace_record.final_volume_per_atom_A3,
            abs_tol=1e-9,
            rel_tol=0.0,
        ):
            raise Step8InputError(
                f"{phase} MACE structure volume disagrees with the Step 6 "
                "summary record."
            )
        signed = mace_volume_per_atom - mp_volume_per_atom
        percent = 100.0 * signed / mp_volume_per_atom
        try:
            mp_density: float | None = float(mp_structure.density)
            mace_density: float | None = float(mace_structure.density)
        except Exception:
            mp_density = None
            mace_density = None

        mp_symbol, mp_number, mp_analyzer = _symmetry_of(
            mp_structure, config, f"{phase} MP structure"
        )
        mace_symbol, mace_number, mace_analyzer = _symmetry_of(
            mace_structure, config, f"{phase} MACE structure"
        )

        lattice_available = False
        note = ""
        mp_abc = mace_abc = abc_diff = None
        mp_angles = mace_angles = angle_diff = None
        try:
            mp_standard = mp_analyzer.get_conventional_standard_structure()
            mace_standard = mace_analyzer.get_conventional_standard_structure()
            if (
                mp_standard.composition.reduced_composition
                != mace_standard.composition.reduced_composition
            ):
                note = (
                    "Standardized compositions differ; lattice comparison "
                    "skipped safely."
                )
            elif len(mp_standard) != len(mace_standard):
                note = (
                    "Standardized conventional cells contain different atom "
                    "counts; lattice comparison skipped safely."
                )
            else:
                mp_lattice = mp_standard.lattice
                mace_lattice = mace_standard.lattice
                mp_abc = tuple(float(value) for value in mp_lattice.abc)
                mace_abc = tuple(float(value) for value in mace_lattice.abc)
                abc_diff = tuple(
                    mace_abc[index] - mp_abc[index] for index in range(3)
                )
                mp_angles = tuple(float(value) for value in mp_lattice.angles)
                mace_angles = tuple(
                    float(value) for value in mace_lattice.angles
                )
                angle_diff = tuple(
                    mace_angles[index] - mp_angles[index] for index in range(3)
                )
                lattice_available = True
                note = "Standardized conventional cells compared."
        except Exception as exc:
            note = (
                "Standardization failed "
                f"({type(exc).__name__}); lattice comparison skipped safely."
            )

        records.append(
            StructuralRecord(
                phase_key=phase,
                material_id=benchmark.material_id,
                mp_volume_per_atom_A3=mp_volume_per_atom,
                mace_volume_per_atom_A3=mace_volume_per_atom,
                signed_volume_per_atom_difference_A3=signed,
                absolute_volume_per_atom_difference_A3=abs(signed),
                volume_per_atom_difference_percent=percent,
                mp_density_g_cm3=mp_density,
                mace_density_g_cm3=mace_density,
                mp_space_group_symbol=mp_symbol,
                mp_space_group_number=mp_number,
                mace_space_group_symbol=mace_symbol,
                mace_space_group_number=mace_number,
                symmetry_preserved=(
                    mp_symbol == mace_symbol and mp_number == mace_number
                ),
                standardization_method=standardization_method,
                lattice_comparison_available=lattice_available,
                lattice_comparison_note=note,
                mp_lattice_abc_A=mp_abc,
                mace_lattice_abc_A=mace_abc,
                lattice_abc_differences_A=abc_diff,
                mp_lattice_angles_deg=mp_angles,
                mace_lattice_angles_deg=mace_angles,
                lattice_angle_differences_deg=angle_diff,
            )
        )
    return tuple(records)


def calculate_structural_statistics(
    records: Sequence[StructuralRecord],
) -> dict[str, Any]:
    """Summarize the structural comparison across the five phases."""

    np = _numpy()
    percents = np.asarray(
        [record.volume_per_atom_difference_percent for record in records]
    )
    phases = [record.phase_key for record in records]
    all_positive = bool(np.all(percents > 0.0))
    return {
        "sample_size": len(records),
        "mean_signed_volume_percent_error": float(np.mean(percents)),
        "mean_absolute_volume_percent_error": float(np.mean(np.abs(percents))),
        "rmse_volume_percent_error": float(
            np.sqrt(np.mean(np.square(percents)))
        ),
        "maximum_absolute_volume_percent_error": float(np.max(np.abs(percents))),
        "phase_with_maximum_absolute_volume_error": phases[
            int(np.argmax(np.abs(percents)))
        ],
        "symmetry_agreement_count": sum(
            1 for record in records if record.symmetry_preserved
        ),
        "lattice_comparisons_available": sum(
            1 for record in records if record.lattice_comparison_available
        ),
        "all_volume_errors_positive": all_positive,
        "systematic_volume_statement": (
            "All five selected phases show positive MACE volume-per-atom "
            "error relative to the Materials Project structures; the "
            "selected sample suggests systematic expansion, without claiming "
            "universal behavior beyond this dataset."
            if all_positive
            else "The selected sample does not show a single-signed volume "
            "error; no systematic expansion claim is made."
        ),
    }


__all__ = [
    "BENCHMARK_LIMITATIONS",
    "EXPECTED_MATERIAL_IDS",
    "EXPECTED_MODEL",
    "PHASE_ORDER",
    "SCHEMA_VERSION",
    "THERMO_TYPE_PREFERENCE",
    "BenchmarkRecord",
    "ComparisonRecord",
    "ComparisonSettings",
    "MaceRecord",
    "MaceSourceBundle",
    "MaceSources",
    "Step8ApiError",
    "Step8CalculationError",
    "Step8CollisionError",
    "Step8Config",
    "Step8ConfigurationError",
    "Step8DependencyError",
    "Step8Error",
    "Step8InputError",
    "Step8OutputPaths",
    "Step8ResumeError",
    "StructuralRecord",
    "benchmark_phase_paths",
    "calculate_comparisons",
    "calculate_statistics",
    "calculate_structural_comparisons",
    "calculate_structural_statistics",
    "load_benchmark_records",
    "load_step8_config",
    "step8_directories",
    "step8_output_paths",
    "validate_phase_keys",
    "validate_step7_sources",
]

"""Shared, safety-focused utilities for the Ni-Al MACE Step 7 workflow.

Step 7 establishes MACE-consistent pure Al and pure Ni reference states and
then calculates MACE-consistent formation energies for the five selected
Ni-Al phases.  This module provides:

``load_step7_config``
    Strict validation of ``configs/mace_formation_energy.json``.
``validate_selected_elemental_structure``
    Structural, provenance, and symmetry validation of a selected pure
    elemental reference structure.
``execute_elemental_references``
    Independent full-cell MACE relaxation of pure Al and pure Ni with the
    exact Step 6 scientific settings, transactionally published.
``validate_step6_compound_sources``
    Read-only validation of the Step 6 full-cell compound energies.
``load_validated_elemental_results`` / ``extract_chemical_potentials``
    Reconstruction of published elemental results and chemical potentials.
``calculate_formation_energies`` / ``lower_convex_envelope``
    Formation-energy arithmetic and the selected-set lower convex envelope.

Calculator, optimizer, and cell-filter imports are deliberately lazy so that
importing this module or running any validation-only path cannot load MACE,
create an optimizer, or alter a structure.  No function in this module ranks
raw total energies across compositions, compares MACE against DFT formation
energies, or performs any DFT, LAMMPS, or training calculation.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from step6_utils import (
    FileSnapshot,
    Step6Error,
    capture_file_snapshot as _step6_capture_file_snapshot,
    file_sha256,
    locate_project_root,
    read_json_object as _step6_read_json_object,
    relative_path,
    utc_timestamp,
    verify_file_snapshot as _step6_verify_file_snapshot,
)


LOGGER = logging.getLogger("step7_utils")

SCHEMA_VERSION = "1.0"
ELEMENT_ORDER: tuple[str, ...] = ("Al", "Ni")
PHASE_ORDER: tuple[str, ...] = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
STRESS_COMPONENTS: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
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
GENERALIZED_FORCE_AUTO_STOP_FMAX = 0.0
ELEMENT_COLORS: Mapping[str, str] = {"Al": "#1f77b4", "Ni": "#ff7f0e"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

NI_MAGNETIC_LIMITATION: tuple[str, ...] = (
    "Ni is a magnetic element in DFT descriptions.",
    "The current structural MACE workflow does not provide an explicit "
    "user-controlled spin or magnetic-moment input.",
    "Step 7 therefore calculates the reference energy produced by the "
    "configured pretrained MACE model for the selected Ni crystal structure.",
    "This is a MACE-consistent reference, not a direct controlled magnetic "
    "DFT reference.",
    "Magnetic-state sensitivity must be considered during later DFT "
    "comparison.",
)

SELECTED_SET_LIMITATION: tuple[str, ...] = (
    "This selected-set analysis is incomplete: only pure Al, pure Ni, and "
    "the five selected compounds were included.",
    "Untested Ni-Al compositions may lie below the selected envelope.",
    "The selected-set lower convex envelope is not Materials Project energy "
    "above hull and is not a complete Ni-Al convex hull.",
    "This is not a complete phase diagram and is not proof of experimental "
    "stability.",
)


class Step7Error(RuntimeError):
    """Base class for controlled Step 7 failures."""


class Step7ConfigurationError(Step7Error):
    """Raised when the Step 7 configuration or command scope is unsafe."""


class Step7DependencyError(Step7Error):
    """Raised when a required installed public API is unavailable."""


class Step7ApiError(Step7Error):
    """Raised for Materials Project access, key, or retrieval failures."""


class Step7InputError(Step7Error):
    """Raised when a protected scientific input is invalid."""


class Step7CalculatorError(Step7Error):
    """Raised when MACE calculator creation or reuse fails."""


class Step7CalculationError(Step7Error):
    """Raised when an energy, force, stress, or optimizer call fails."""


class Step7SafetyError(Step7CalculationError):
    """Raised when a configured Step 7 safety invariant is violated."""


class Step7CollisionError(Step7Error):
    """Raised when output collision handling refuses publication."""


class Step7PublicationError(Step7Error):
    """Raised when transactional publication or rollback fails."""


class Step7ResumeError(Step7Error):
    """Raised when an existing Step 7 bundle is not safe to reuse."""


def read_strict_json(path: Path, label: str) -> dict[str, Any]:
    """Read strict JSON, converting Step 6 helper failures to Step 7 errors."""

    try:
        return _step6_read_json_object(path, label)
    except Step6Error as exc:
        raise Step7InputError(str(exc)) from exc


def snapshot_file(path: Path, label: str) -> FileSnapshot:
    """Capture a protected-file fingerprint with Step 7 error semantics."""

    try:
        return _step6_capture_file_snapshot(path, label)
    except Step6Error as exc:
        raise Step7InputError(str(exc)) from exc


def verify_snapshots(snapshots: Iterable[FileSnapshot]) -> None:
    """Verify protected-file fingerprints with Step 7 error semantics."""

    for snapshot in snapshots:
        try:
            _step6_verify_file_snapshot(snapshot)
        except Step6Error as exc:
            raise Step7SafetyError(str(exc)) from exc


def assert_mace_not_imported() -> None:
    """Fail when any MACE module has been imported in this process."""

    loaded = sorted(
        name for name in sys.modules if name == "mace" or name.startswith("mace.")
    )
    if loaded:
        raise Step7DependencyError(
            "MACE modules are unexpectedly imported in a no-MACE path: "
            + ", ".join(loaded)
        )


def installed_step7_versions() -> Mapping[str, str]:
    """Return audited package versions without creating scientific objects."""

    versions: dict[str, str] = {}
    for distribution in (
        "ase",
        "mace-torch",
        "numpy",
        "pymatgen",
        "matplotlib",
        "mp-api",
        "python-dotenv",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Step7DependencyError(
                f"Required distribution is not installed: {distribution}"
            ) from exc
    return versions


def _numpy() -> Any:
    """Import NumPy without importing any calculator or optimizer."""

    try:
        import numpy as np
    except ImportError as exc:
        raise Step7DependencyError(f"NumPy is unavailable: {exc}") from exc
    return np


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Step7ConfigurationError(f"{label} must be a nonempty string.")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Step7ConfigurationError(f"{label} must be boolean.")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> Any:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise Step7ConfigurationError(
            f"{label} must be exactly {expected!r}; received {value!r}."
        )
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Step7ConfigurationError(f"{label} must be a JSON object.")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step7CalculationError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise Step7SafetyError(f"{label} is NaN or infinity.")
    return result


def _relative_repo_path(value: Any, label: str, project_root: Path) -> Path:
    text = _require_string(value, label)
    candidate = (project_root / Path(text)).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step7ConfigurationError(
            f"{label} must remain inside the repository: {text}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class ModelSettings:
    """Validated MACE model settings shared by every Step 7 calculation."""

    family: str
    name: str
    value: str
    device: str
    default_dtype: str
    dispersion: bool

    def to_json(self) -> dict[str, Any]:
        """Return a portable JSON record."""

        return {
            "family": self.family,
            "name": self.name,
            "value": self.value,
            "device": self.device,
            "default_dtype": self.default_dtype,
            "dispersion": self.dispersion,
        }


@dataclass(frozen=True)
class MaterialsProjectSettings:
    """Validated retrieval policy for the elemental references."""

    api_key_environment_variable: str
    require_stable_elemental_reference: bool
    maximum_energy_above_hull_eV_per_atom: float
    expected_elemental_space_group_symbol: str
    expected_elemental_space_group_number: int


@dataclass(frozen=True)
class ElementalReferenceSpec:
    """Expected identity for one elemental reference."""

    element: str
    expected_formula: str
    expected_elements: tuple[str, ...]


@dataclass(frozen=True)
class RelaxationSettings:
    """Validated full-cell relaxation criteria matching Step 6 exactly."""

    optimizer: str
    force_threshold_eV_per_A: float
    stress_threshold_eV_per_A3: float
    maximum_steps: int
    trajectory_interval: int
    hydrostatic_strain: bool
    constant_volume: bool
    external_pressure_eV_per_A3: float


@dataclass(frozen=True)
class SafetySettings:
    """Validated calculation safety limits matching Step 6 exactly."""

    maximum_absolute_volume_change_percent: float
    maximum_atomic_displacement_A: float
    stop_on_nonfinite_value: bool
    preserve_original_structure: bool
    require_periodic_cell: bool


@dataclass(frozen=True)
class SymmetrySettings:
    """Validated symmetry-analysis tolerances."""

    symprec_A: float
    angle_tolerance_deg: float


@dataclass(frozen=True)
class CompoundSources:
    """Validated machine-readable Step 5/6 compound energy sources."""

    initial_zero_shot_table: Path
    full_cell_summary: Path
    full_cell_checkpoint_directory: Path
    selected_structure_directory: Path


@dataclass(frozen=True)
class InputSettings:
    """Validated Step 7 elemental-reference input locations."""

    elemental_raw_root: Path
    elemental_selected_directory: Path


@dataclass(frozen=True)
class OutputSettings:
    """Validated Step 7 output roots."""

    elemental_reference_root: Path
    formation_energy_root: Path


@dataclass(frozen=True)
class AnalysisSettings:
    """Validated numerical tolerances for the selected-set analysis."""

    envelope_tolerance_eV_per_atom: float
    arithmetic_tolerance_eV_per_atom: float


@dataclass(frozen=True)
class Step7Config:
    """Fully validated Step 7 configuration."""

    project_root: Path
    config_path: Path
    config_snapshot: FileSnapshot
    fingerprint: str
    raw: Mapping[str, Any]
    model: ModelSettings
    materials_project: MaterialsProjectSettings
    elemental_references: Mapping[str, ElementalReferenceSpec]
    relaxation: RelaxationSettings
    safety: SafetySettings
    symmetry: SymmetrySettings
    compound_sources: CompoundSources
    phase_order: tuple[str, ...]
    input: InputSettings
    output: OutputSettings
    analysis: AnalysisSettings


def load_step7_config(config_path: Path | str) -> Step7Config:
    """Load and strictly validate the Step 7 configuration.

    Every scientific control is validated against the mandatory Step 7 value
    so that a silent configuration edit cannot alter model settings or the
    Step 6-consistent convergence criteria.
    """

    project_root = locate_project_root()
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step7ConfigurationError(
            f"Configuration must remain inside the repository: {resolved}"
        ) from exc
    raw = read_strict_json(resolved, "Step 7 formation-energy configuration")

    _require_exact(raw.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_string(raw.get("description"), "description")

    model_raw = _require_mapping(raw.get("model"), "model")
    model = ModelSettings(
        family=_require_exact(model_raw.get("family"), "MACE", "model.family"),
        name=_require_exact(model_raw.get("name"), "MACE-MP-0", "model.name"),
        value=_require_exact(model_raw.get("value"), "small", "model.value"),
        device=_require_exact(model_raw.get("device"), "cpu", "model.device"),
        default_dtype=_require_exact(
            model_raw.get("default_dtype"), "float64", "model.default_dtype"
        ),
        dispersion=_require_exact(
            model_raw.get("dispersion"), False, "model.dispersion"
        ),
    )

    mp_raw = _require_mapping(raw.get("materials_project"), "materials_project")
    materials_project = MaterialsProjectSettings(
        api_key_environment_variable=_require_exact(
            mp_raw.get("api_key_environment_variable"),
            "MP_API_KEY",
            "materials_project.api_key_environment_variable",
        ),
        require_stable_elemental_reference=_require_exact(
            mp_raw.get("require_stable_elemental_reference"),
            True,
            "materials_project.require_stable_elemental_reference",
        ),
        maximum_energy_above_hull_eV_per_atom=_require_exact(
            mp_raw.get("maximum_energy_above_hull_eV_per_atom"),
            1e-8,
            "materials_project.maximum_energy_above_hull_eV_per_atom",
        ),
        expected_elemental_space_group_symbol=_require_exact(
            mp_raw.get("expected_elemental_space_group_symbol"),
            "Fm-3m",
            "materials_project.expected_elemental_space_group_symbol",
        ),
        expected_elemental_space_group_number=_require_exact(
            mp_raw.get("expected_elemental_space_group_number"),
            225,
            "materials_project.expected_elemental_space_group_number",
        ),
    )

    references_raw = _require_mapping(
        raw.get("elemental_references"), "elemental_references"
    )
    if tuple(references_raw.keys()) != ELEMENT_ORDER:
        raise Step7ConfigurationError(
            "elemental_references must define exactly Al then Ni."
        )
    references: dict[str, ElementalReferenceSpec] = {}
    for element in ELEMENT_ORDER:
        spec_raw = _require_mapping(
            references_raw.get(element), f"elemental_references.{element}"
        )
        _require_exact(
            spec_raw.get("expected_formula"),
            element,
            f"elemental_references.{element}.expected_formula",
        )
        _require_exact(
            spec_raw.get("expected_elements"),
            [element],
            f"elemental_references.{element}.expected_elements",
        )
        references[element] = ElementalReferenceSpec(
            element=element,
            expected_formula=element,
            expected_elements=(element,),
        )

    relaxation_raw = _require_mapping(raw.get("relaxation"), "relaxation")
    relaxation = RelaxationSettings(
        optimizer=_require_exact(
            relaxation_raw.get("optimizer"), "FIRE", "relaxation.optimizer"
        ),
        force_threshold_eV_per_A=_require_exact(
            relaxation_raw.get("force_threshold_eV_per_A"),
            0.01,
            "relaxation.force_threshold_eV_per_A",
        ),
        stress_threshold_eV_per_A3=_require_exact(
            relaxation_raw.get("stress_threshold_eV_per_A3"),
            0.0006241509,
            "relaxation.stress_threshold_eV_per_A3",
        ),
        maximum_steps=_require_exact(
            relaxation_raw.get("maximum_steps"), 1000, "relaxation.maximum_steps"
        ),
        trajectory_interval=_require_exact(
            relaxation_raw.get("trajectory_interval"),
            1,
            "relaxation.trajectory_interval",
        ),
        hydrostatic_strain=_require_exact(
            relaxation_raw.get("hydrostatic_strain"),
            False,
            "relaxation.hydrostatic_strain",
        ),
        constant_volume=_require_exact(
            relaxation_raw.get("constant_volume"),
            False,
            "relaxation.constant_volume",
        ),
        external_pressure_eV_per_A3=_require_exact(
            relaxation_raw.get("external_pressure_eV_per_A3"),
            0.0,
            "relaxation.external_pressure_eV_per_A3",
        ),
    )

    safety_raw = _require_mapping(raw.get("safety"), "safety")
    safety = SafetySettings(
        maximum_absolute_volume_change_percent=_require_exact(
            safety_raw.get("maximum_absolute_volume_change_percent"),
            25.0,
            "safety.maximum_absolute_volume_change_percent",
        ),
        maximum_atomic_displacement_A=_require_exact(
            safety_raw.get("maximum_atomic_displacement_A"),
            2.0,
            "safety.maximum_atomic_displacement_A",
        ),
        stop_on_nonfinite_value=_require_exact(
            safety_raw.get("stop_on_nonfinite_value"),
            True,
            "safety.stop_on_nonfinite_value",
        ),
        preserve_original_structure=_require_exact(
            safety_raw.get("preserve_original_structure"),
            True,
            "safety.preserve_original_structure",
        ),
        require_periodic_cell=_require_exact(
            safety_raw.get("require_periodic_cell"),
            True,
            "safety.require_periodic_cell",
        ),
    )

    symmetry_raw = _require_mapping(raw.get("symmetry"), "symmetry")
    symmetry = SymmetrySettings(
        symprec_A=_require_exact(
            symmetry_raw.get("symprec_A"), 0.001, "symmetry.symprec_A"
        ),
        angle_tolerance_deg=_require_exact(
            symmetry_raw.get("angle_tolerance_deg"),
            5.0,
            "symmetry.angle_tolerance_deg",
        ),
    )

    sources_raw = _require_mapping(
        raw.get("compound_energy_sources"), "compound_energy_sources"
    )
    compound_sources = CompoundSources(
        initial_zero_shot_table=_relative_repo_path(
            _require_exact(
                sources_raw.get("initial_zero_shot_table"),
                "results/mace_zero_shot/tables/ni_al_mace_zero_shot.json",
                "compound_energy_sources.initial_zero_shot_table",
            ),
            "compound_energy_sources.initial_zero_shot_table",
            project_root,
        ),
        full_cell_summary=_relative_repo_path(
            _require_exact(
                sources_raw.get("full_cell_summary"),
                "results/mace_relaxation/full_cell/tables/"
                "ni_al_full_cell_summary.json",
                "compound_energy_sources.full_cell_summary",
            ),
            "compound_energy_sources.full_cell_summary",
            project_root,
        ),
        full_cell_checkpoint_directory=_relative_repo_path(
            _require_exact(
                sources_raw.get("full_cell_checkpoint_directory"),
                "results/mace_relaxation/full_cell/checkpoints",
                "compound_energy_sources.full_cell_checkpoint_directory",
            ),
            "compound_energy_sources.full_cell_checkpoint_directory",
            project_root,
        ),
        selected_structure_directory=_relative_repo_path(
            _require_exact(
                sources_raw.get("selected_structure_directory"),
                "data/processed/ni_al_structures/selected",
                "compound_energy_sources.selected_structure_directory",
            ),
            "compound_energy_sources.selected_structure_directory",
            project_root,
        ),
    )

    phase_order_raw = raw.get("phase_order")
    if tuple(phase_order_raw or ()) != PHASE_ORDER:
        raise Step7ConfigurationError(
            "phase_order must be exactly "
            f"{list(PHASE_ORDER)}; received {phase_order_raw!r}."
        )

    input_raw = _require_mapping(raw.get("input"), "input")
    input_settings = InputSettings(
        elemental_raw_root=_relative_repo_path(
            _require_exact(
                input_raw.get("elemental_raw_root"),
                "data/raw/materials_project/elemental_references",
                "input.elemental_raw_root",
            ),
            "input.elemental_raw_root",
            project_root,
        ),
        elemental_selected_directory=_relative_repo_path(
            _require_exact(
                input_raw.get("elemental_selected_directory"),
                "data/processed/elemental_references/selected",
                "input.elemental_selected_directory",
            ),
            "input.elemental_selected_directory",
            project_root,
        ),
    )

    output_raw = _require_mapping(raw.get("output"), "output")
    output_settings = OutputSettings(
        elemental_reference_root=_relative_repo_path(
            _require_exact(
                output_raw.get("elemental_reference_root"),
                "results/mace_elemental_references",
                "output.elemental_reference_root",
            ),
            "output.elemental_reference_root",
            project_root,
        ),
        formation_energy_root=_relative_repo_path(
            _require_exact(
                output_raw.get("formation_energy_root"),
                "results/mace_formation_energy",
                "output.formation_energy_root",
            ),
            "output.formation_energy_root",
            project_root,
        ),
    )

    analysis_raw = _require_mapping(raw.get("analysis"), "analysis")
    analysis = AnalysisSettings(
        envelope_tolerance_eV_per_atom=_require_exact(
            analysis_raw.get("envelope_tolerance_eV_per_atom"),
            1e-8,
            "analysis.envelope_tolerance_eV_per_atom",
        ),
        arithmetic_tolerance_eV_per_atom=_require_exact(
            analysis_raw.get("arithmetic_tolerance_eV_per_atom"),
            1e-12,
            "analysis.arithmetic_tolerance_eV_per_atom",
        ),
    )

    snapshot = snapshot_file(resolved, "Step 7 configuration")
    return Step7Config(
        project_root=project_root,
        config_path=resolved,
        config_snapshot=snapshot,
        fingerprint=snapshot.sha256,
        raw=raw,
        model=model,
        materials_project=materials_project,
        elemental_references=references,
        relaxation=relaxation,
        safety=safety,
        symmetry=symmetry,
        compound_sources=compound_sources,
        phase_order=PHASE_ORDER,
        input=input_settings,
        output=output_settings,
        analysis=analysis,
    )


def validate_element_keys(elements: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a requested element selection deterministically."""

    if elements is None:
        return ELEMENT_ORDER
    selected = tuple(elements)
    if not selected:
        raise Step7ConfigurationError("At least one element must be selected.")
    if len(set(selected)) != len(selected):
        raise Step7ConfigurationError("Requested elements contain duplicates.")
    unknown = [element for element in selected if element not in ELEMENT_ORDER]
    if unknown:
        raise Step7ConfigurationError(
            "Unknown element(s): " + ", ".join(unknown)
        )
    return tuple(element for element in ELEMENT_ORDER if element in selected)


# ---------------------------------------------------------------------------
# Output-path resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementalOutputPaths:
    """Canonical Step 7 output bundle for one elemental reference."""

    single_point_json: Path
    structure: Path
    trajectory: Path
    history_csv: Path
    report: Path
    log: Path
    checkpoint: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every target in deterministic publication order."""

        return (
            self.single_point_json,
            self.structure,
            self.trajectory,
            self.history_csv,
            self.report,
            self.log,
            self.checkpoint,
        )


@dataclass(frozen=True)
class ElementalCombinedPaths:
    """Canonical combined elemental summary and figure targets."""

    csv: Path
    json: Path
    report: Path
    energy_figure: Path
    force_figure: Path
    stress_figure: Path
    volume_figure: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every combined target."""

        return (
            self.csv,
            self.json,
            self.report,
            self.energy_figure,
            self.force_figure,
            self.stress_figure,
            self.volume_figure,
        )


def elemental_output_paths(config: Step7Config, element: str) -> ElementalOutputPaths:
    """Resolve the canonical per-element output bundle."""

    root = config.output.elemental_reference_root
    return ElementalOutputPaths(
        single_point_json=(
            root / "single_point" / f"{element}_mace_initial_single_point.json"
        ),
        structure=(
            root
            / "full_cell"
            / "structures"
            / f"{element}_mace_full_cell_relaxed.extxyz"
        ),
        trajectory=root / "full_cell" / "trajectories" / f"{element}_mace_full_cell.traj",
        history_csv=(
            root / "full_cell" / "tables" / f"{element}_mace_full_cell_history.csv"
        ),
        report=root / "full_cell" / "reports" / f"{element}_mace_reference_report.txt",
        log=root / "full_cell" / "logs" / f"{element}_mace_reference.log",
        checkpoint=(
            root
            / "full_cell"
            / "checkpoints"
            / f"{element}_mace_reference_result.json"
        ),
    )


def elemental_combined_paths(config: Step7Config) -> ElementalCombinedPaths:
    """Resolve the combined elemental summary and figure targets."""

    root = config.output.elemental_reference_root
    return ElementalCombinedPaths(
        csv=root / "full_cell" / "tables" / "mace_elemental_reference_summary.csv",
        json=root / "full_cell" / "tables" / "mace_elemental_reference_summary.json",
        report=(
            root / "full_cell" / "reports" / "mace_elemental_reference_summary.txt"
        ),
        energy_figure=root / "figures" / "elemental_reference_energy_convergence.png",
        force_figure=root / "figures" / "elemental_reference_force_convergence.png",
        stress_figure=root / "figures" / "elemental_reference_stress_convergence.png",
        volume_figure=root / "figures" / "elemental_reference_volume_convergence.png",
    )


def elemental_directories(config: Step7Config) -> tuple[Path, ...]:
    """Return every elemental-reference directory execution may populate."""

    root = config.output.elemental_reference_root
    return (
        root,
        root / "single_point",
        root / "full_cell",
        root / "full_cell" / "structures",
        root / "full_cell" / "trajectories",
        root / "full_cell" / "tables",
        root / "full_cell" / "reports",
        root / "full_cell" / "checkpoints",
        root / "full_cell" / "logs",
        root / "figures",
    )


@dataclass(frozen=True)
class FormationOutputPaths:
    """Canonical Step 7 formation-energy analysis targets."""

    table_csv: Path
    table_json: Path
    report: Path
    relaxed_figure: Path
    initial_vs_relaxed_figure: Path
    envelope_figure: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every analysis target."""

        return (
            self.table_csv,
            self.table_json,
            self.report,
            self.relaxed_figure,
            self.initial_vs_relaxed_figure,
            self.envelope_figure,
        )


def formation_output_paths(config: Step7Config) -> FormationOutputPaths:
    """Resolve the canonical formation-energy analysis targets."""

    root = config.output.formation_energy_root
    return FormationOutputPaths(
        table_csv=root / "tables" / "ni_al_mace_formation_energies.csv",
        table_json=root / "tables" / "ni_al_mace_formation_energies.json",
        report=root / "reports" / "ni_al_mace_formation_energy_report.txt",
        relaxed_figure=(
            root / "figures" / "mace_relaxed_formation_energy_vs_ni_fraction.png"
        ),
        initial_vs_relaxed_figure=(
            root / "figures" / "mace_initial_vs_relaxed_formation_energy.png"
        ),
        envelope_figure=root / "figures" / "selected_set_lower_convex_envelope.png",
    )


def formation_directories(config: Step7Config) -> tuple[Path, ...]:
    """Return every formation-energy directory execution may populate."""

    root = config.output.formation_energy_root
    return (root, root / "tables", root / "reports", root / "figures")


def final_report_path(config: Step7Config) -> Path:
    """Resolve the authoritative Step 7 final report path."""

    return (
        config.output.formation_energy_root
        / "reports"
        / "ni_al_step7_final_report.txt"
    )


def selected_elemental_paths(
    config: Step7Config, element: str
) -> tuple[Path, Path]:
    """Return the selected EXTXYZ and metadata paths for one element."""

    directory = config.input.elemental_selected_directory
    return (
        directory / f"{element}.extxyz",
        directory / f"{element}.metadata.json",
    )


# ---------------------------------------------------------------------------
# Selected elemental structure validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementalStructureInput:
    """Strictly validated selected elemental reference for one element."""

    element: str
    material_id: str
    atom_count: int
    atoms: Any
    metadata: Mapping[str, Any]
    structure_snapshot: FileSnapshot
    metadata_snapshot: FileSnapshot
    space_group_symbol: str
    space_group_number: int
    crystal_system: str


def analyze_symmetry(atoms: Any, config: Step7Config) -> dict[str, Any]:
    """Analyze the space group of an ASE structure with pymatgen tolerances."""

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:
        raise Step7DependencyError(
            f"pymatgen symmetry analysis is unavailable: {exc}"
        ) from exc
    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=config.symmetry.symprec_A,
            angle_tolerance=config.symmetry.angle_tolerance_deg,
        )
        symbol = analyzer.get_space_group_symbol()
        number = int(analyzer.get_space_group_number())
        crystal_system = str(analyzer.get_crystal_system())
    except Exception as exc:
        raise Step7CalculationError(
            f"Symmetry analysis failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "space_group_symbol": symbol,
        "space_group_number": number,
        "crystal_system": crystal_system,
        "symprec_A": config.symmetry.symprec_A,
        "angle_tolerance_deg": config.symmetry.angle_tolerance_deg,
        "limitation": (
            "Space-group detection is tolerance-dependent; no symmetry "
            "constraint or refinement was applied."
        ),
    }


def validate_selected_elemental_structure(
    config: Step7Config, element: str
) -> ElementalStructureInput:
    """Validate one selected elemental EXTXYZ/metadata pair completely."""

    if element not in ELEMENT_ORDER:
        raise Step7ConfigurationError(f"Unknown element: {element!r}")
    structure_path, metadata_path = selected_elemental_paths(config, element)
    if not structure_path.is_file():
        raise Step7InputError(
            f"Selected {element} EXTXYZ does not exist: {structure_path}. "
            "Run fetch_ni_al_elemental_references.py --fetch first."
        )
    if not metadata_path.is_file():
        raise Step7InputError(
            f"Selected {element} metadata does not exist: {metadata_path}."
        )
    structure_snapshot = snapshot_file(
        structure_path, f"selected {element} EXTXYZ"
    )
    metadata_snapshot = snapshot_file(
        metadata_path, f"selected {element} metadata"
    )
    metadata = read_strict_json(metadata_path, f"selected {element} metadata")
    for key in ("element", "material_id", "structure_sha256"):
        if key not in metadata:
            raise Step7InputError(
                f"Selected {element} metadata is missing {key!r}."
            )
    if metadata.get("element") != element:
        raise Step7InputError(
            f"Selected {element} metadata records element "
            f"{metadata.get('element')!r}."
        )
    material_id = metadata.get("material_id")
    if not isinstance(material_id, str) or not material_id.startswith("mp-"):
        raise Step7InputError(
            f"Selected {element} metadata material_id is invalid: {material_id!r}"
        )
    if metadata.get("structure_sha256") != structure_snapshot.sha256:
        raise Step7InputError(
            f"Selected {element} EXTXYZ fingerprint does not match its "
            "metadata; the structure or metadata was modified."
        )

    try:
        from ase.io import read as ase_read
    except ImportError as exc:
        raise Step7DependencyError(f"ASE reader is unavailable: {exc}") from exc
    try:
        frames = ase_read(structure_path, index=":", format="extxyz")
    except Exception as exc:
        raise Step7InputError(
            f"Could not read selected {element} EXTXYZ: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list) or len(frames) != 1:
        raise Step7InputError(
            f"Selected {element} EXTXYZ must contain exactly one frame."
        )
    atoms = frames[0]
    if atoms.calc is not None:
        # A stray attached calculator would let stored results masquerade as
        # newly calculated MACE values; the pristine input must be inert.
        raise Step7InputError(
            f"Selected {element} structure has an attached calculator."
        )
    symbols = tuple(atoms.get_chemical_symbols())
    if len(symbols) <= 0:
        raise Step7InputError(f"Selected {element} structure has no atoms.")
    if set(symbols) != {element}:
        raise Step7InputError(
            f"Selected {element} structure contains species {sorted(set(symbols))}."
        )
    try:
        from pymatgen.core import Composition
    except ImportError as exc:
        raise Step7DependencyError(f"pymatgen is unavailable: {exc}") from exc
    reduced = Composition("".join(symbols)).reduced_formula
    if reduced != element:
        raise Step7InputError(
            f"Selected {element} reduced composition is {reduced!r}."
        )
    np = _numpy()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    determinant = float(np.linalg.det(cell))
    volume = float(atoms.get_volume())
    if (
        positions.shape != (len(atoms), 3)
        or not bool(np.all(np.isfinite(positions)))
        or cell.shape != (3, 3)
        or not bool(np.all(np.isfinite(cell)))
    ):
        raise Step7InputError(
            f"Selected {element} positions or cell are not finite."
        )
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise Step7InputError(
            f"Selected {element} cell determinant is not positive: {determinant!r}"
        )
    if not math.isfinite(volume) or volume <= 0.0:
        raise Step7InputError(
            f"Selected {element} volume is invalid: {volume!r}"
        )
    if not bool(np.all(np.asarray(atoms.get_pbc(), dtype=bool))):
        raise Step7InputError(
            f"Selected {element} structure must be periodic in x, y, and z."
        )
    info_material_id = atoms.info.get("material_id")
    if info_material_id is not None and info_material_id != material_id:
        raise Step7InputError(
            f"Selected {element} EXTXYZ material_id {info_material_id!r} "
            f"disagrees with metadata {material_id!r}."
        )

    symmetry = analyze_symmetry(atoms, config)
    expected_symbol = (
        config.materials_project.expected_elemental_space_group_symbol
    )
    expected_number = (
        config.materials_project.expected_elemental_space_group_number
    )
    if (
        symmetry["space_group_symbol"] != expected_symbol
        or symmetry["space_group_number"] != expected_number
    ):
        raise Step7InputError(
            f"Selected {element} structure is {symmetry['space_group_symbol']} "
            f"({symmetry['space_group_number']}), not the expected "
            f"{expected_symbol} ({expected_number}). Review the retrieval "
            "metadata and selection rationale before continuing; this "
            "deviation requires explicit project review."
        )

    # The source files must not change while they are being read.
    verify_snapshots((structure_snapshot, metadata_snapshot))
    return ElementalStructureInput(
        element=element,
        material_id=material_id,
        atom_count=len(atoms),
        atoms=atoms,
        metadata=metadata,
        structure_snapshot=structure_snapshot,
        metadata_snapshot=metadata_snapshot,
        space_group_symbol=symmetry["space_group_symbol"],
        space_group_number=symmetry["space_group_number"],
        crystal_system=symmetry["crystal_system"],
    )


# ---------------------------------------------------------------------------
# Calculator session
# ---------------------------------------------------------------------------


@dataclass
class Step7CalculatorSession:
    """One reusable MACE calculator with auditable counters."""

    calculator: Any
    calculator_class: str
    configuration_fingerprint: str
    load_count: int = 1
    state_evaluations: int = 0
    optimizer_steps: int = 0
    warnings: list[str] = field(default_factory=list)


def load_step7_calculator(config: Step7Config) -> Step7CalculatorSession:
    """Lazily construct exactly one configured MACE calculator."""

    try:
        from ase.calculators.calculator import Calculator
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise Step7DependencyError(
            f"MACE/ASE calculator API import failed: {exc}"
        ) from exc
    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calculator = mace_mp(
                model=config.model.value,
                device=config.model.device,
                default_dtype=config.model.default_dtype,
                dispersion=config.model.dispersion,
            )
        captured.extend(
            f"{item.category.__name__}: {item.message}" for item in caught
        )
    except Exception as exc:
        raise Step7CalculatorError(
            "MACE calculator loading failed with the validated settings: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(calculator, Calculator):
        raise Step7CalculatorError(
            "mace_mp did not return an ASE Calculator instance."
        )
    calculator_type = type(calculator)
    return Step7CalculatorSession(
        calculator=calculator,
        calculator_class=(
            f"{calculator_type.__module__}.{calculator_type.__qualname__}"
        ),
        configuration_fingerprint=config.fingerprint,
        warnings=captured,
    )


def reset_calculator(session: Step7CalculatorSession, label: str) -> None:
    """Clear shared calculator state before another independent element."""

    try:
        session.calculator.reset()
    except Exception as exc:
        raise Step7CalculatorError(
            f"Could not reset calculator after {label}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def validate_frechet_cell_filter_api() -> str:
    """Verify the installed public FrechetCellFilter API without creating it."""

    import inspect

    try:
        ase_version = importlib.metadata.version("ase")
    except importlib.metadata.PackageNotFoundError:
        ase_version = "not installed"
    try:
        from ase.filters import FrechetCellFilter
    except ImportError as exc:
        raise Step7DependencyError(
            "ase.filters.FrechetCellFilter is unavailable in installed ASE "
            f"{ase_version}: {exc}. Do not substitute another cell filter "
            "without user review."
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
    missing = sorted(required.difference(parameters))
    if missing:
        raise Step7DependencyError(
            f"Installed FrechetCellFilter public signature in ASE {ase_version} "
            "is missing: " + ", ".join(missing)
        )
    return ase_version


# ---------------------------------------------------------------------------
# State evaluation and full-cell relaxation
# ---------------------------------------------------------------------------


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
class ElementState:
    """Complete finite state measured from the underlying ASE Atoms."""

    step: int
    elapsed_seconds: float
    total_energy_eV: float
    energy_per_atom_eV: float
    forces_eV_per_A: tuple[tuple[float, float, float], ...]
    maximum_force_eV_per_A: float
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
    maximum_internal_displacement_A: float
    rms_internal_displacement_A: float
    maximum_total_displacement_A: float
    rms_total_displacement_A: float
    volume_change_percent: float
    force_converged: bool
    stress_converged: bool
    overall_converged: bool


def capture_initial_geometry(atoms: Any) -> InitialGeometry:
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
        raise Step7InputError("Initial ASE structure geometry is invalid.")
    if pbc_array.shape != (3,) or not bool(np.all(pbc_array)):
        raise Step7InputError(
            "Initial ASE structure must be periodic in x, y, and z."
        )
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


def _displacement_statistics(
    atoms: Any, initial: InitialGeometry
) -> tuple[float, float, float, float]:
    """Return periodic internal and total displacement statistics.

    Internal displacement maps wrapped fractional-coordinate differences
    through the initial cell so pure cell deformation does not count as
    atomic motion.  Total Cartesian displacement includes cell deformation.
    """

    np = _numpy()
    current_scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    initial_scaled = np.asarray(initial.scaled_positions, dtype=float)
    initial_cell = np.asarray(initial.cell_A, dtype=float)
    current_cell = np.asarray(atoms.cell.array, dtype=float)
    delta_scaled = current_scaled - initial_scaled
    # floor(x + 0.5) maps each component into [-0.5, 0.5), avoiding
    # banker's-rounding ambiguity exactly at half a cell.
    delta_scaled = delta_scaled - np.floor(delta_scaled + 0.5)
    internal = delta_scaled @ initial_cell
    initial_wrapped = initial_scaled - np.floor(initial_scaled)
    correlated_final_scaled = initial_wrapped + delta_scaled
    total = (
        correlated_final_scaled @ current_cell - initial_wrapped @ initial_cell
    )
    internal_magnitudes = np.linalg.norm(internal, axis=1)
    total_magnitudes = np.linalg.norm(total, axis=1)
    if not (
        bool(np.all(np.isfinite(internal_magnitudes)))
        and bool(np.all(np.isfinite(total_magnitudes)))
    ):
        raise Step7SafetyError("Displacement metrics contain NaN or infinity.")
    return (
        float(np.max(internal_magnitudes)),
        float(np.sqrt(np.mean(np.square(internal_magnitudes)))),
        float(np.max(total_magnitudes)),
        float(np.sqrt(np.mean(np.square(total_magnitudes)))),
    )


def evaluate_element_state(
    atoms: Any,
    initial: InitialGeometry,
    config: Step7Config,
    element: str,
    step: int,
    elapsed_seconds: float,
    session: Step7CalculatorSession,
) -> ElementState:
    """Evaluate one complete raw state and apply every safety check."""

    np = _numpy()
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception as exc:
        raise Step7CalculationError(
            f"{element} state evaluation failed at optimizer step {step}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(energy):
        raise Step7SafetyError(
            f"{element} energy is nonfinite at optimizer step {step}."
        )
    if forces.shape != (initial.atom_count, 3) or not bool(
        np.all(np.isfinite(forces))
    ):
        raise Step7SafetyError(
            f"{element} forces are invalid at optimizer step {step}."
        )
    if stress.shape != (6,) or not bool(np.all(np.isfinite(stress))):
        raise Step7SafetyError(
            f"{element} stress is invalid at optimizer step {step}."
        )

    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    determinant = float(np.linalg.det(cell))
    volume = float(atoms.get_volume())
    if len(atoms) != initial.atom_count:
        raise Step7SafetyError(f"{element} atom count changed during relaxation.")
    if tuple(atoms.get_chemical_symbols()) != initial.symbols:
        raise Step7SafetyError(
            f"{element} atom symbols or ordering changed during relaxation."
        )
    if (
        tuple(int(value) for value in atoms.get_atomic_numbers())
        != initial.atomic_numbers
    ):
        raise Step7SafetyError(
            f"{element} atomic numbers or ordering changed during relaxation."
        )
    if tuple(bool(value) for value in atoms.get_pbc()) != initial.pbc:
        raise Step7SafetyError(
            f"{element} periodic-boundary flags changed during relaxation."
        )
    if (
        positions.shape != (initial.atom_count, 3)
        or cell.shape != (3, 3)
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(cell)))
    ):
        raise Step7SafetyError(
            f"{element} positions or cell contain NaN or infinity."
        )
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise Step7SafetyError(
            f"{element} cell determinant is nonpositive: {determinant!r}."
        )
    if not math.isfinite(volume) or volume <= 0.0:
        raise Step7SafetyError(f"{element} cell volume is invalid: {volume!r}.")

    volume_change_percent = 100.0 * (volume / initial.volume_A3 - 1.0)
    if abs(volume_change_percent) > (
        config.safety.maximum_absolute_volume_change_percent
    ):
        raise Step7SafetyError(
            f"{element} absolute volume change exceeded the configured limit: "
            f"{volume_change_percent:.17g}%."
        )
    (
        maximum_internal,
        rms_internal,
        maximum_total,
        rms_total,
    ) = _displacement_statistics(atoms, initial)
    if maximum_internal > config.safety.maximum_atomic_displacement_A:
        raise Step7SafetyError(
            f"{element} maximum internal displacement exceeded the limit: "
            f"{maximum_internal:.17g} A."
        )

    force_magnitudes = np.linalg.norm(forces, axis=1)
    total_force = np.sum(forces, axis=0)
    maximum_force = float(np.max(force_magnitudes))
    rms_force = float(np.sqrt(np.mean(np.square(force_magnitudes))))
    total_force_norm = float(np.linalg.norm(total_force))
    maximum_stress = float(np.max(np.abs(stress)))
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    if not (
        math.isfinite(maximum_force)
        and math.isfinite(rms_force)
        and math.isfinite(total_force_norm)
        and math.isfinite(maximum_stress)
        and bool(np.all(np.isfinite(lengths)))
        and bool(np.all(np.isfinite(angles)))
    ):
        raise Step7SafetyError(
            f"{element} derived metrics are nonfinite at step {step}."
        )
    force_converged = (
        maximum_force <= config.relaxation.force_threshold_eV_per_A
    )
    stress_converged = (
        maximum_stress <= config.relaxation.stress_threshold_eV_per_A3
    )
    session.state_evaluations += 1
    return ElementState(
        step=step,
        elapsed_seconds=float(elapsed_seconds),
        total_energy_eV=energy,
        energy_per_atom_eV=energy / initial.atom_count,
        forces_eV_per_A=tuple(
            tuple(float(value) for value in row) for row in forces
        ),
        maximum_force_eV_per_A=maximum_force,
        rms_force_eV_per_A=rms_force,
        total_force_eV_per_A=tuple(float(value) for value in total_force),
        total_force_norm_eV_per_A=total_force_norm,
        stress_eV_per_A3=tuple(float(value) for value in stress),
        maximum_absolute_stress_eV_per_A3=maximum_stress,
        volume_A3=volume,
        volume_per_atom_A3=volume / initial.atom_count,
        cell_A=tuple(tuple(float(value) for value in row) for row in cell),
        lattice_lengths_A=tuple(float(value) for value in lengths),
        lattice_angles_deg=tuple(float(value) for value in angles),
        positions_A=tuple(
            tuple(float(value) for value in row) for row in positions
        ),
        scaled_positions=tuple(
            tuple(float(value) for value in row)
            for row in atoms.get_scaled_positions(wrap=False)
        ),
        maximum_internal_displacement_A=maximum_internal,
        rms_internal_displacement_A=rms_internal,
        maximum_total_displacement_A=maximum_total,
        rms_total_displacement_A=rms_total,
        volume_change_percent=volume_change_percent,
        force_converged=force_converged,
        stress_converged=stress_converged,
        overall_converged=force_converged and stress_converged,
    )


def run_element_relaxation(
    config: Step7Config,
    element: str,
    working: Any,
    initial_geometry: InitialGeometry,
    initial_state: ElementState,
    session: Step7CalculatorSession,
    trajectory_path: Path,
    log_path: Path,
    started_monotonic: float,
) -> tuple[str, bool, int, tuple[ElementState, ...], tuple[str, ...]]:
    """Run FIRE/FrechetCellFilter with exact externally measured convergence.

    Convergence is decided from the underlying ASE Atoms forces and stress,
    never from FIRE termination or filter generalized forces alone.
    """

    if initial_state.overall_converged:
        try:
            from ase.io.trajectory import Trajectory
            with Trajectory(trajectory_path, mode="w", atoms=working) as trajectory:
                trajectory.write(working)
        except Exception as exc:
            raise Step7PublicationError(
                f"Could not write the {element} step-0 trajectory: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        log_path.write_text(
            "Optimizer not created: the initial state already met all "
            "configured convergence criteria.\n",
            encoding="utf-8",
        )
        return ("ALREADY_CONVERGED", False, 0, (initial_state,), ())

    try:
        from ase.io.trajectory import Trajectory
        from ase.optimize import FIRE
    except ImportError as exc:
        raise Step7DependencyError(
            f"Installed ASE FIRE/Trajectory API is unavailable: {exc}"
        ) from exc
    try:
        from ase.filters import FrechetCellFilter
    except ImportError as exc:
        ase_version = importlib.metadata.version("ase")
        raise Step7DependencyError(
            "ase.filters.FrechetCellFilter is unavailable in installed ASE "
            f"{ase_version}: {exc}"
        ) from exc
    target = FrechetCellFilter(
        working,
        mask=None,
        exp_cell_factor=None,
        hydrostatic_strain=config.relaxation.hydrostatic_strain,
        constant_volume=config.relaxation.constant_volume,
        scalar_pressure=config.relaxation.external_pressure_eV_per_A3,
    )

    history: list[ElementState] = [initial_state]
    last_state: ElementState = initial_state
    phase_warnings: list[str] = []
    optimizer: Any = None

    def monitor() -> None:
        """Measure actual Atoms values; never use filter convergence proxies."""

        nonlocal last_state
        if optimizer is None:
            raise Step7CalculationError("Optimizer observer ran before setup.")
        step = int(optimizer.get_number_of_steps())
        if step == 0:
            state = initial_state
        else:
            state = evaluate_element_state(
                working,
                initial_geometry,
                config,
                element,
                step,
                time.monotonic() - started_monotonic,
                session,
            )
        last_state = state
        if not history or history[-1].step != state.step:
            history.append(state)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Trajectory(
                trajectory_path, mode="w", atoms=working
            ) as trajectory:
                with FIRE(
                    target, logfile=str(log_path), trajectory=None
                ) as optimizer_instance:
                    optimizer = optimizer_instance
                    optimizer.attach(monitor, interval=1)
                    optimizer.attach(
                        trajectory.write,
                        interval=config.relaxation.trajectory_interval,
                    )
                    iterator = optimizer.irun(
                        fmax=GENERALIZED_FORCE_AUTO_STOP_FMAX,
                        steps=config.relaxation.maximum_steps,
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
                    if history and history[-1].step != optimizer_steps:
                        history.append(last_state)
                    if (
                        optimizer_steps
                        % config.relaxation.trajectory_interval
                        != 0
                    ):
                        trajectory.write(working)
                    status = (
                        "CONVERGED" if reached_convergence else "NOT_CONVERGED"
                    )
            phase_warnings.extend(
                f"{item.category.__name__}: {item.message}" for item in caught
            )
    except Step7Error:
        raise
    except Exception as exc:
        raise Step7CalculationError(
            f"FIRE relaxation failed for {element}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not history:
        raise Step7CalculationError(
            f"FIRE produced no recorded states for {element}."
        )
    # Reassert exact scientific status independent of generator termination.
    if last_state.overall_converged:
        status = "CONVERGED"
    elif optimizer_steps >= config.relaxation.maximum_steps:
        status = "NOT_CONVERGED"
    else:
        raise Step7CalculationError(
            f"FIRE stopped unexpectedly after {optimizer_steps} steps for "
            f"{element} without satisfying the actual convergence criteria."
        )
    session.optimizer_steps += optimizer_steps
    return (status, True, optimizer_steps, tuple(history), tuple(phase_warnings))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def element_state_to_json(state: ElementState) -> dict[str, Any]:
    """Serialize one complete state without loss of numeric precision."""

    return {
        "step": state.step,
        "elapsed_seconds": state.elapsed_seconds,
        "total_energy_eV": state.total_energy_eV,
        "energy_per_atom_eV": state.energy_per_atom_eV,
        "forces_eV_per_A": [list(row) for row in state.forces_eV_per_A],
        "maximum_force_eV_per_A": state.maximum_force_eV_per_A,
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
        "maximum_internal_displacement_A": state.maximum_internal_displacement_A,
        "rms_internal_displacement_A": state.rms_internal_displacement_A,
        "maximum_total_displacement_A": state.maximum_total_displacement_A,
        "rms_total_displacement_A": state.rms_total_displacement_A,
        "volume_change_percent": state.volume_change_percent,
        "force_converged": state.force_converged,
        "stress_converged": state.stress_converged,
        "overall_converged": state.overall_converged,
    }


def element_state_from_json(
    raw_value: Any, label: str, atom_count: int
) -> ElementState:
    """Strictly reconstruct one state from a published checkpoint."""

    if not isinstance(raw_value, Mapping):
        raise Step7ResumeError(f"{label} must be a JSON object.")
    raw = raw_value
    np = _numpy()

    def finite(name: str) -> float:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Step7ResumeError(f"{label}.{name} is not numeric.")
        result = float(value)
        if not math.isfinite(result):
            raise Step7ResumeError(f"{label}.{name} is nonfinite.")
        return result

    def vector(name: str, length: int) -> tuple[float, ...]:
        array = np.asarray(raw.get(name), dtype=float)
        if array.shape != (length,) or not bool(np.all(np.isfinite(array))):
            raise Step7ResumeError(f"{label}.{name} is invalid.")
        return tuple(float(value) for value in array)

    def matrix(name: str, rows: int) -> tuple[tuple[float, ...], ...]:
        array = np.asarray(raw.get(name), dtype=float)
        if array.shape != (rows, 3) or not bool(np.all(np.isfinite(array))):
            raise Step7ResumeError(f"{label}.{name} is invalid.")
        return tuple(tuple(float(value) for value in row) for row in array)

    def boolean(name: str) -> bool:
        value = raw.get(name)
        if not isinstance(value, bool):
            raise Step7ResumeError(f"{label}.{name} must be boolean.")
        return value

    step = raw.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise Step7ResumeError(f"{label}.step must be an integer >= 0.")
    return ElementState(
        step=step,
        elapsed_seconds=finite("elapsed_seconds"),
        total_energy_eV=finite("total_energy_eV"),
        energy_per_atom_eV=finite("energy_per_atom_eV"),
        forces_eV_per_A=matrix("forces_eV_per_A", atom_count),
        maximum_force_eV_per_A=finite("maximum_force_eV_per_A"),
        rms_force_eV_per_A=finite("rms_force_eV_per_A"),
        total_force_eV_per_A=vector("total_force_eV_per_A", 3),
        total_force_norm_eV_per_A=finite("total_force_norm_eV_per_A"),
        stress_eV_per_A3=vector("stress_eV_per_A3", 6),
        maximum_absolute_stress_eV_per_A3=finite(
            "maximum_absolute_stress_eV_per_A3"
        ),
        volume_A3=finite("volume_A3"),
        volume_per_atom_A3=finite("volume_per_atom_A3"),
        cell_A=matrix("cell_A", 3),
        lattice_lengths_A=vector("lattice_lengths_A", 3),
        lattice_angles_deg=vector("lattice_angles_deg", 3),
        positions_A=matrix("positions_A", atom_count),
        scaled_positions=matrix("scaled_positions", atom_count),
        maximum_internal_displacement_A=finite(
            "maximum_internal_displacement_A"
        ),
        rms_internal_displacement_A=finite("rms_internal_displacement_A"),
        maximum_total_displacement_A=finite("maximum_total_displacement_A"),
        rms_total_displacement_A=finite("rms_total_displacement_A"),
        volume_change_percent=finite("volume_change_percent"),
        force_converged=boolean("force_converged"),
        stress_converged=boolean("stress_converged"),
        overall_converged=boolean("overall_converged"),
    )


def validate_element_state_consistency(
    state: ElementState,
    config: Step7Config,
    atom_count: int,
    initial_volume_A3: float,
    label: str,
) -> None:
    """Recompute derived and convergence fields of a stored state."""

    np = _numpy()
    forces = np.asarray(state.forces_eV_per_A, dtype=float)
    magnitudes = np.linalg.norm(forces, axis=1)
    checks = (
        (state.maximum_force_eV_per_A, float(np.max(magnitudes)), "maximum_force"),
        (
            state.rms_force_eV_per_A,
            float(np.sqrt(np.mean(np.square(magnitudes)))),
            "rms_force",
        ),
        (
            state.total_force_norm_eV_per_A,
            float(np.linalg.norm(np.sum(forces, axis=0))),
            "total_force_norm",
        ),
        (
            state.energy_per_atom_eV,
            state.total_energy_eV / atom_count,
            "energy_per_atom",
        ),
        (
            state.volume_per_atom_A3,
            state.volume_A3 / atom_count,
            "volume_per_atom",
        ),
        (
            state.maximum_absolute_stress_eV_per_A3,
            float(np.max(np.abs(np.asarray(state.stress_eV_per_A3)))),
            "maximum_absolute_stress",
        ),
        (
            state.volume_change_percent,
            100.0 * (state.volume_A3 / initial_volume_A3 - 1.0),
            "volume_change_percent",
        ),
    )
    for actual, expected, name in checks:
        if not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
            raise Step7ResumeError(
                f"{label}.{name} is internally inconsistent: {actual:.17g} "
                f"versus {expected:.17g}."
            )
    expected_force = (
        state.maximum_force_eV_per_A <= config.relaxation.force_threshold_eV_per_A
    )
    expected_stress = (
        state.maximum_absolute_stress_eV_per_A3
        <= config.relaxation.stress_threshold_eV_per_A3
    )
    if (
        state.force_converged != expected_force
        or state.stress_converged != expected_stress
        or state.overall_converged != (expected_force and expected_stress)
    ):
        raise Step7ResumeError(f"{label} convergence booleans are invalid.")


ELEMENT_HISTORY_FIELDNAMES: tuple[str, ...] = (
    "step",
    "elapsed_seconds",
    "total_energy_eV",
    "energy_per_atom_eV",
    "maximum_force_eV_per_A",
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


def element_history_row(state: ElementState) -> dict[str, Any]:
    """Serialize one state to a flat history row."""

    row: dict[str, Any] = {
        "step": state.step,
        "elapsed_seconds": state.elapsed_seconds,
        "total_energy_eV": state.total_energy_eV,
        "energy_per_atom_eV": state.energy_per_atom_eV,
        "maximum_force_eV_per_A": state.maximum_force_eV_per_A,
        "rms_force_eV_per_A": state.rms_force_eV_per_A,
        "total_force_norm_eV_per_A": state.total_force_norm_eV_per_A,
        "maximum_absolute_stress_eV_per_A3": (
            state.maximum_absolute_stress_eV_per_A3
        ),
        "volume_A3": state.volume_A3,
        "volume_per_atom_A3": state.volume_per_atom_A3,
        "volume_change_percent": state.volume_change_percent,
        "maximum_internal_displacement_A": state.maximum_internal_displacement_A,
        "rms_internal_displacement_A": state.rms_internal_displacement_A,
        "maximum_total_displacement_A": state.maximum_total_displacement_A,
        "rms_total_displacement_A": state.rms_total_displacement_A,
        "force_converged": state.force_converged,
        "stress_converged": state.stress_converged,
        "overall_converged": state.overall_converged,
        "safety_status": "PASS",
    }
    for component, value in zip(STRESS_COMPONENTS, state.stress_eV_per_A3):
        row[f"stress_{component}_eV_per_A3"] = value
    for name, value in zip(("a", "b", "c"), state.lattice_lengths_A):
        row[f"lattice_{name}_A"] = value
    for name, value in zip(
        ("alpha", "beta", "gamma"), state.lattice_angles_deg
    ):
        row[f"angle_{name}_deg"] = value
    return row


def element_history_csv_bytes(history: Sequence[ElementState]) -> bytes:
    """Serialize the complete per-step history table."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ELEMENT_HISTORY_FIELDNAMES)
    writer.writeheader()
    for state in history:
        writer.writerow(element_history_row(state))
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Transactional publication
# ---------------------------------------------------------------------------


def stage_path(staging_root: Path, output_root: Path, final_path: Path) -> Path:
    """Map a canonical output target into a same-volume staging tree."""

    try:
        relative = final_path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise Step7PublicationError(
            f"Output target escaped its output root: {final_path}"
        ) from exc
    staged = staging_root / relative
    staged.parent.mkdir(parents=True, exist_ok=True)
    return staged


def publish_files_transactionally(
    project_root: Path,
    output_root: Path,
    staged_by_final: Mapping[Path, Path],
    *,
    overwrite: bool,
    final_validator: Callable[[], None] | None = None,
) -> None:
    """Atomically publish a multi-file bundle and roll back any failure."""

    if not staged_by_final:
        return
    normalized: dict[Path, Path] = {}
    root = output_root.resolve()
    for final, staged in staged_by_final.items():
        resolved = final.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise Step7PublicationError(
                f"Publication target escapes the controlled root: {resolved}"
            ) from exc
        if resolved in normalized:
            raise Step7PublicationError(
                f"Duplicate publication target: {resolved}"
            )
        normalized[resolved] = staged.resolve()
    collisions: list[Path] = []
    for target, staged in normalized.items():
        if target.exists():
            if target.is_dir():
                raise Step7CollisionError(
                    f"Output target is a directory, not a file: {target}"
                )
            if not overwrite:
                collisions.append(target)
        if not staged.is_file() or staged.stat().st_size <= 0:
            raise Step7PublicationError(
                f"Staged publication input is absent or empty: {staged}"
            )
    if collisions:
        details = "\n".join(
            f"  - {relative_path(path, project_root)}"
            for path in sorted(collisions)
        )
        raise Step7CollisionError(
            "Refusing to overwrite existing Step 7 output(s):\n" + details
        )
    for target in normalized:
        target.parent.mkdir(parents=True, exist_ok=True)

    backup_root = Path(
        tempfile.mkdtemp(prefix=".step7-publication-backup-", dir=output_root)
    )
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for index, (target, staged) in enumerate(sorted(normalized.items())):
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
        detail = f"{type(exc).__name__}: {exc}" + (
            "; rollback errors: " + "; ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        raise Step7PublicationError(
            f"Transactional Step 7 publication failed: {detail}"
        ) from exc
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Publish one UTF-8 text file atomically with collision protection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise Step7CollisionError(f"Output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise Step7PublicationError(f"Staged text output is empty: {temporary}")
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def write_strict_json_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize strict JSON output bytes."""

    return (
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Step 6 compound sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompoundRecord:
    """Validated Step 5/6 energies and stoichiometry for one compound."""

    phase_key: str
    material_id: str
    reduced_formula: str
    formula_unit_al: int
    formula_unit_ni: int
    formula_units_in_cell: int
    cell_al_count: int
    cell_ni_count: int
    cell_atom_count: int
    ni_fraction: float
    initial_total_energy_eV: float
    initial_energy_per_atom_eV: float
    final_total_energy_eV: float
    final_energy_per_atom_eV: float
    convergence_status: str
    safety_status: str
    checkpoint_path: Path
    selected_structure_path: Path


def validate_step6_compound_sources(
    config: Step7Config,
) -> tuple[tuple[CompoundRecord, ...], tuple[FileSnapshot, ...]]:
    """Validate the five Step 6 full-cell compound energies without MACE.

    Primary energies come only from the machine-readable full-cell summary
    and per-phase checkpoints; nothing is re-derived from console output or
    formatted text reports, and no compound relaxation is rerun.
    """

    sources = config.compound_sources
    snapshots: list[FileSnapshot] = [
        snapshot_file(sources.full_cell_summary, "Step 6 full-cell summary"),
        snapshot_file(sources.initial_zero_shot_table, "Step 5 zero-shot table"),
    ]
    summary = read_strict_json(
        sources.full_cell_summary, "Step 6 full-cell summary"
    )
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("artifact_type") != "ni_al_full_cell_relaxation_summary"
        or summary.get("mode") != "full_cell"
        or summary.get("overall_status") != "SUCCESS"
        or summary.get("failed_phases") != []
        or list(summary.get("requested_phases", ())) != list(PHASE_ORDER)
        or list(summary.get("completed_phases", ())) != list(PHASE_ORDER)
    ):
        raise Step7InputError(
            "Step 6 full-cell summary identity/status is not a validated "
            "SUCCESS for all five phases."
        )
    model = summary.get("model")
    if not isinstance(model, Mapping) or any(
        model.get(key) != value
        for key, value in config.model.to_json().items()
    ):
        raise Step7InputError(
            "Step 6 full-cell model settings are incompatible with the "
            "Step 7 model settings; energies cannot be mixed."
        )
    convergence = summary.get("convergence")
    if not isinstance(convergence, Mapping) or (
        convergence.get("force_threshold_eV_per_A")
        != config.relaxation.force_threshold_eV_per_A
        or convergence.get("stress_threshold_eV_per_A3")
        != config.relaxation.stress_threshold_eV_per_A3
    ):
        raise Step7InputError(
            "Step 6 convergence criteria differ from the Step 7 elemental "
            "criteria; formation energies would be inconsistent."
        )
    step6_fingerprint = summary.get("configuration_fingerprint_sha256")
    if not isinstance(step6_fingerprint, str) or len(step6_fingerprint) != 64:
        raise Step7InputError(
            "Step 6 summary configuration fingerprint is invalid."
        )

    zero_shot = read_strict_json(
        sources.initial_zero_shot_table, "Step 5 zero-shot table"
    )
    if zero_shot.get("overall_status") != "success":
        raise Step7InputError("Step 5 zero-shot table is not a success record.")
    zero_records = {
        record.get("phase_key"): record
        for record in zero_shot.get("records", ())
        if isinstance(record, Mapping)
        and record.get("evaluation_status") == "success"
    }

    summary_records = {
        record.get("phase_key"): record
        for record in summary.get("records", ())
        if isinstance(record, Mapping)
    }
    try:
        from ase.io import read as ase_read
        from pymatgen.core import Composition
    except ImportError as exc:
        raise Step7DependencyError(
            f"ASE/pymatgen composition support is unavailable: {exc}"
        ) from exc

    compounds: list[CompoundRecord] = []
    for phase in PHASE_ORDER:
        record = summary_records.get(phase)
        zero_record = zero_records.get(phase)
        if record is None:
            raise Step7InputError(f"Step 6 summary has no record for {phase}.")
        if zero_record is None:
            raise Step7InputError(
                f"Step 5 zero-shot table has no successful record for {phase}."
            )
        material_id = record.get("material_id")
        if material_id != EXPECTED_MATERIAL_IDS[phase]:
            raise Step7InputError(
                f"{phase} material ID mismatch: {material_id!r}."
            )
        atom_count = record.get("number_of_atoms")
        if atom_count != EXPECTED_ATOM_COUNTS[phase]:
            raise Step7InputError(f"{phase} atom count mismatch: {atom_count!r}.")
        status = record.get("status")
        if status not in {"CONVERGED", "ALREADY_CONVERGED"}:
            raise Step7InputError(
                f"{phase} full-cell status {status!r} is not a converged result."
            )
        if record.get("safety_status") != "PASS":
            raise Step7InputError(f"{phase} full-cell safety status is not PASS.")
        initial_total = _finite_float(
            record.get("initial_total_energy_eV"),
            f"{phase} initial total energy",
        )
        final_total = _finite_float(
            record.get("final_total_energy_eV"), f"{phase} final total energy"
        )
        initial_per_atom = _finite_float(
            record.get("initial_energy_per_atom_eV"),
            f"{phase} initial energy per atom",
        )
        final_per_atom = _finite_float(
            record.get("final_energy_per_atom_eV"),
            f"{phase} final energy per atom",
        )
        for label, total, per_atom in (
            ("initial", initial_total, initial_per_atom),
            ("final", final_total, final_per_atom),
        ):
            if not math.isclose(
                per_atom, total / atom_count, abs_tol=1e-12, rel_tol=1e-12
            ):
                raise Step7InputError(
                    f"{phase} {label} energy-per-atom bookkeeping is "
                    "internally inconsistent."
                )
        zero_total = _finite_float(
            zero_record.get("total_energy_eV"), f"{phase} Step 5 total energy"
        )
        if not math.isclose(zero_total, initial_total, abs_tol=1e-9, rel_tol=0.0):
            raise Step7InputError(
                f"{phase} Step 5 zero-shot energy disagrees with the Step 6 "
                f"initial energy: {zero_total:.17g} versus {initial_total:.17g}."
            )

        checkpoint_path = (
            sources.full_cell_checkpoint_directory
            / f"{phase}_full_cell_result.json"
        )
        snapshots.append(
            snapshot_file(checkpoint_path, f"{phase} full-cell checkpoint")
        )
        checkpoint = read_strict_json(
            checkpoint_path, f"{phase} full-cell checkpoint"
        )
        if (
            checkpoint.get("phase_key") != phase
            or checkpoint.get("material_id") != EXPECTED_MATERIAL_IDS[phase]
            or checkpoint.get("convergence_status") != status
            or checkpoint.get("safety_status") != "PASS"
            or checkpoint.get("configuration_fingerprint_sha256")
            != step6_fingerprint
        ):
            raise Step7InputError(
                f"{phase} checkpoint identity/status disagrees with the summary."
            )
        final_state = checkpoint.get("final")
        if not isinstance(final_state, Mapping) or not math.isclose(
            _finite_float(
                final_state.get("total_energy_eV"),
                f"{phase} checkpoint final energy",
            ),
            final_total,
            abs_tol=1e-12,
            rel_tol=1e-12,
        ):
            raise Step7InputError(
                f"{phase} checkpoint final energy disagrees with the summary."
            )

        formula = zero_record.get("formula")
        composition = Composition(str(formula)).reduced_composition
        element_counts = {
            str(species): int(round(amount))
            for species, amount in composition.get_el_amt_dict().items()
        }
        if set(element_counts) != {"Al", "Ni"}:
            raise Step7InputError(
                f"{phase} reduced composition {element_counts!r} is not Al-Ni."
            )
        formula_al = element_counts["Al"]
        formula_ni = element_counts["Ni"]
        if formula_al <= 0 or formula_ni <= 0:
            raise Step7InputError(
                f"{phase} stoichiometric counts must be positive integers."
            )
        formula_atoms = formula_al + formula_ni
        if atom_count % formula_atoms != 0:
            raise Step7InputError(
                f"{phase} cell atom count {atom_count} is not an integer "
                f"multiple of the {formula_atoms}-atom formula unit."
            )
        formula_units = atom_count // formula_atoms

        selected_path = (
            sources.selected_structure_directory / f"{phase}.extxyz"
        )
        snapshots.append(
            snapshot_file(selected_path, f"selected {phase} EXTXYZ")
        )
        try:
            frames = ase_read(selected_path, index=":", format="extxyz")
        except Exception as exc:
            raise Step7InputError(
                f"Could not read selected {phase} structure: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(frames, list) or len(frames) != 1:
            raise Step7InputError(
                f"Selected {phase} EXTXYZ must contain one frame."
            )
        symbols = tuple(frames[0].get_chemical_symbols())
        cell_al = sum(1 for symbol in symbols if symbol == "Al")
        cell_ni = sum(1 for symbol in symbols if symbol == "Ni")
        if cell_al + cell_ni != len(symbols) or len(symbols) != atom_count:
            raise Step7InputError(
                f"Selected {phase} structure composition is inconsistent."
            )
        if (
            cell_al != formula_units * formula_al
            or cell_ni != formula_units * formula_ni
        ):
            raise Step7InputError(
                f"{phase} actual cell composition Al{cell_al}Ni{cell_ni} does "
                f"not match {formula_units} x Al{formula_al}Ni{formula_ni}."
            )
        ni_fraction = cell_ni / atom_count
        al_fraction = cell_al / atom_count
        if not (0.0 < ni_fraction < 1.0) or not math.isclose(
            al_fraction + ni_fraction, 1.0, abs_tol=1e-12, rel_tol=0.0
        ):
            raise Step7InputError(f"{phase} atomic fractions are invalid.")

        compounds.append(
            CompoundRecord(
                phase_key=phase,
                material_id=EXPECTED_MATERIAL_IDS[phase],
                reduced_formula=composition.reduced_formula,
                formula_unit_al=formula_al,
                formula_unit_ni=formula_ni,
                formula_units_in_cell=formula_units,
                cell_al_count=cell_al,
                cell_ni_count=cell_ni,
                cell_atom_count=atom_count,
                ni_fraction=ni_fraction,
                initial_total_energy_eV=initial_total,
                initial_energy_per_atom_eV=initial_per_atom,
                final_total_energy_eV=final_total,
                final_energy_per_atom_eV=final_per_atom,
                convergence_status=str(status),
                safety_status="PASS",
                checkpoint_path=checkpoint_path,
                selected_structure_path=selected_path,
            )
        )
    verify_snapshots(snapshots)
    return tuple(compounds), tuple(snapshots)


# ---------------------------------------------------------------------------
# Elemental results and chemical potentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementalReferenceRecord:
    """Validated published elemental-reference result for one element."""

    element: str
    material_id: str
    atom_count: int
    initial_total_energy_eV: float
    initial_energy_per_atom_eV: float
    final_total_energy_eV: float
    final_energy_per_atom_eV: float
    initial_maximum_force_eV_per_A: float
    final_maximum_force_eV_per_A: float
    initial_maximum_absolute_stress_eV_per_A3: float
    final_maximum_absolute_stress_eV_per_A3: float
    initial_volume_A3: float
    final_volume_A3: float
    volume_change_percent: float
    optimizer_steps: int
    convergence_status: str
    safety_status: str
    initial_symmetry: Mapping[str, Any]
    final_symmetry: Mapping[str, Any]
    configuration_fingerprint: str
    model: Mapping[str, Any]
    checkpoint_path: Path


def load_validated_elemental_results(
    config: Step7Config,
) -> tuple[Mapping[str, ElementalReferenceRecord], tuple[FileSnapshot, ...]]:
    """Load and cross-validate the published elemental-reference results."""

    combined = elemental_combined_paths(config)
    snapshots: list[FileSnapshot] = [
        snapshot_file(combined.json, "elemental reference summary JSON")
    ]
    summary = read_strict_json(
        combined.json, "elemental reference summary JSON"
    )
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("artifact_type") != "ni_al_mace_elemental_reference_summary"
        or summary.get("configuration_fingerprint_sha256") != config.fingerprint
        or list(summary.get("elements", ())) != list(ELEMENT_ORDER)
    ):
        raise Step7InputError(
            "Elemental reference summary identity/configuration is invalid. "
            "Run run_ni_al_mace_elemental_references.py --execute first."
        )
    summary_records = {
        record.get("element"): record
        for record in summary.get("records", ())
        if isinstance(record, Mapping)
    }
    results: dict[str, ElementalReferenceRecord] = {}
    for element in ELEMENT_ORDER:
        outputs = elemental_output_paths(config, element)
        snapshots.append(
            snapshot_file(outputs.checkpoint, f"{element} reference checkpoint")
        )
        checkpoint = read_strict_json(
            outputs.checkpoint, f"{element} reference checkpoint"
        )
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("artifact_type")
            != "ni_al_mace_elemental_reference_result"
            or checkpoint.get("element") != element
            or checkpoint.get("configuration_fingerprint_sha256")
            != config.fingerprint
            or checkpoint.get("safety_status") != "PASS"
        ):
            raise Step7InputError(
                f"{element} reference checkpoint identity/configuration is "
                "invalid."
            )
        model = checkpoint.get("model")
        if not isinstance(model, Mapping) or any(
            model.get(key) != value
            for key, value in config.model.to_json().items()
        ):
            raise Step7InputError(
                f"{element} reference model settings do not match Step 7."
            )
        status = checkpoint.get("convergence_status")
        if status not in {"CONVERGED", "ALREADY_CONVERGED"}:
            raise Step7InputError(
                f"{element} reference status {status!r} is not a valid "
                "chemical-potential source; formation energies must not use "
                "a NOT_CONVERGED or FAILED elemental state."
            )
        atom_count = checkpoint.get("number_of_atoms")
        if (
            isinstance(atom_count, bool)
            or not isinstance(atom_count, int)
            or atom_count <= 0
        ):
            raise Step7InputError(f"{element} reference atom count is invalid.")
        initial_state = element_state_from_json(
            checkpoint.get("initial"), f"{element}.initial", atom_count
        )
        final_state = element_state_from_json(
            checkpoint.get("final"), f"{element}.final", atom_count
        )
        validate_element_state_consistency(
            initial_state, config, atom_count, initial_state.volume_A3,
            f"{element}.initial",
        )
        validate_element_state_consistency(
            final_state, config, atom_count, initial_state.volume_A3,
            f"{element}.final",
        )
        if not final_state.overall_converged:
            raise Step7InputError(
                f"{element} final state does not satisfy the convergence "
                "criteria despite its converged label."
            )
        summary_record = summary_records.get(element)
        if not isinstance(summary_record, Mapping) or not math.isclose(
            _finite_float(
                summary_record.get("final_energy_per_atom_eV"),
                f"{element} summary final energy per atom",
            ),
            final_state.energy_per_atom_eV,
            abs_tol=1e-12,
            rel_tol=1e-12,
        ):
            raise Step7InputError(
                f"{element} summary record disagrees with its checkpoint."
            )
        material_id = checkpoint.get("material_id")
        if not isinstance(material_id, str) or not material_id.startswith("mp-"):
            raise Step7InputError(f"{element} reference material ID is invalid.")
        initial_symmetry = checkpoint.get("initial_symmetry")
        final_symmetry = checkpoint.get("final_symmetry")
        if not isinstance(initial_symmetry, Mapping) or not isinstance(
            final_symmetry, Mapping
        ):
            raise Step7InputError(f"{element} symmetry records are invalid.")
        optimizer_steps = checkpoint.get("optimizer", {})
        steps = (
            optimizer_steps.get("steps")
            if isinstance(optimizer_steps, Mapping)
            else None
        )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise Step7InputError(f"{element} optimizer step count is invalid.")
        results[element] = ElementalReferenceRecord(
            element=element,
            material_id=material_id,
            atom_count=atom_count,
            initial_total_energy_eV=initial_state.total_energy_eV,
            initial_energy_per_atom_eV=initial_state.energy_per_atom_eV,
            final_total_energy_eV=final_state.total_energy_eV,
            final_energy_per_atom_eV=final_state.energy_per_atom_eV,
            initial_maximum_force_eV_per_A=initial_state.maximum_force_eV_per_A,
            final_maximum_force_eV_per_A=final_state.maximum_force_eV_per_A,
            initial_maximum_absolute_stress_eV_per_A3=(
                initial_state.maximum_absolute_stress_eV_per_A3
            ),
            final_maximum_absolute_stress_eV_per_A3=(
                final_state.maximum_absolute_stress_eV_per_A3
            ),
            initial_volume_A3=initial_state.volume_A3,
            final_volume_A3=final_state.volume_A3,
            volume_change_percent=final_state.volume_change_percent,
            optimizer_steps=steps,
            convergence_status=str(status),
            safety_status="PASS",
            initial_symmetry=initial_symmetry,
            final_symmetry=final_symmetry,
            configuration_fingerprint=config.fingerprint,
            model=dict(model),
            checkpoint_path=outputs.checkpoint,
        )
    verify_snapshots(snapshots)
    return results, tuple(snapshots)


def extract_chemical_potentials(
    config: Step7Config,
    results: Mapping[str, ElementalReferenceRecord],
) -> dict[str, float]:
    """Extract mu_Al_MACE and mu_Ni_MACE from validated relaxed references.

    Both chemical potentials are relaxed MACE total energies per atom in
    eV/atom.  Materials Project DFT elemental energies are never used here,
    and MACE and DFT energies are never mixed.
    """

    potentials: dict[str, float] = {}
    for element in ELEMENT_ORDER:
        record = results.get(element)
        if record is None:
            raise Step7InputError(f"No validated {element} reference exists.")
        if record.convergence_status not in {"CONVERGED", "ALREADY_CONVERGED"}:
            raise Step7InputError(
                f"{element} reference is {record.convergence_status}; it is "
                "not a valid chemical potential."
            )
        if record.safety_status != "PASS":
            raise Step7InputError(f"{element} reference safety status failed.")
        if record.configuration_fingerprint != config.fingerprint:
            raise Step7InputError(
                f"{element} reference configuration fingerprint mismatch."
            )
        if record.atom_count <= 0:
            raise Step7InputError(f"{element} reference atom count is invalid.")
        mu = record.final_total_energy_eV / record.atom_count
        if not math.isfinite(mu):
            raise Step7InputError(f"{element} chemical potential is nonfinite.")
        if not math.isclose(
            mu, record.final_energy_per_atom_eV, abs_tol=1e-12, rel_tol=1e-12
        ):
            raise Step7InputError(
                f"{element} chemical-potential bookkeeping is inconsistent."
            )
        potentials[element] = mu
    models = {json.dumps(dict(results[el].model), sort_keys=True) for el in ELEMENT_ORDER}
    if len(models) != 1:
        raise Step7InputError(
            "Al and Ni references were not produced by identical MACE model "
            "settings."
        )
    return potentials


# ---------------------------------------------------------------------------
# Formation energies and the selected-set lower convex envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormationEnergyRecord:
    """Complete formation-energy result for one compound."""

    phase_key: str
    material_id: str
    reduced_formula: str
    formula_unit_al: int
    formula_unit_ni: int
    formula_units_in_cell: int
    cell_al_count: int
    cell_ni_count: int
    cell_atom_count: int
    ni_fraction: float
    initial_compound_total_energy_eV: float
    relaxed_compound_total_energy_eV: float
    initial_formation_energy_eV_per_atom: float
    relaxed_formation_energy_eV_per_atom: float
    relaxation_effect_eV_per_atom: float
    envelope_energy_eV_per_atom: float
    energy_above_envelope_eV_per_atom: float
    on_selected_set_envelope: bool
    compound_convergence_status: str
    elemental_convergence_status: str
    safety_status: str
    checkpoint_path: Path
    selected_structure_path: Path


def _formation_energy_per_atom(
    cell_total_energy_eV: float,
    cell_al: int,
    cell_ni: int,
    mu_al: float,
    mu_ni: float,
) -> float:
    """Apply the formation-energy equation using actual cell composition."""

    total_atoms = cell_al + cell_ni
    return (
        cell_total_energy_eV - cell_al * mu_al - cell_ni * mu_ni
    ) / total_atoms


def calculate_formation_energies(
    config: Step7Config,
    compounds: Sequence[CompoundRecord],
    elemental_results: Mapping[str, ElementalReferenceRecord],
) -> tuple[FormationEnergyRecord, ...]:
    """Calculate initial and relaxed formation energies for every compound.

    The primary result subtracts full-cell relaxed elemental references from
    full-cell relaxed compound energies.  The clearly separated diagnostic
    subtracts initial single-point elemental references from initial
    fixed-geometry compound energies.  The two states are never mixed.
    """

    potentials = extract_chemical_potentials(config, elemental_results)
    mu_al = potentials["Al"]
    mu_ni = potentials["Ni"]
    initial_mu_al = elemental_results["Al"].initial_energy_per_atom_eV
    initial_mu_ni = elemental_results["Ni"].initial_energy_per_atom_eV
    tolerance = config.analysis.arithmetic_tolerance_eV_per_atom

    provisional: list[dict[str, Any]] = []
    for compound in compounds:
        relaxed = _formation_energy_per_atom(
            compound.final_total_energy_eV,
            compound.cell_al_count,
            compound.cell_ni_count,
            mu_al,
            mu_ni,
        )
        # The formula-unit route divides the cell energy by the number of
        # formula units first; both routes must agree exactly.
        formula_atoms = compound.formula_unit_al + compound.formula_unit_ni
        relaxed_formula_route = (
            compound.final_total_energy_eV / compound.formula_units_in_cell
            - compound.formula_unit_al * mu_al
            - compound.formula_unit_ni * mu_ni
        ) / formula_atoms
        if not math.isclose(
            relaxed, relaxed_formula_route, abs_tol=tolerance, rel_tol=0.0
        ):
            raise Step7CalculationError(
                f"{compound.phase_key} formula-unit and full-cell relaxed "
                f"formation energies disagree: {relaxed:.17g} versus "
                f"{relaxed_formula_route:.17g}."
            )
        initial = _formation_energy_per_atom(
            compound.initial_total_energy_eV,
            compound.cell_al_count,
            compound.cell_ni_count,
            initial_mu_al,
            initial_mu_ni,
        )
        initial_formula_route = (
            compound.initial_total_energy_eV / compound.formula_units_in_cell
            - compound.formula_unit_al * initial_mu_al
            - compound.formula_unit_ni * initial_mu_ni
        ) / formula_atoms
        if not math.isclose(
            initial, initial_formula_route, abs_tol=tolerance, rel_tol=0.0
        ):
            raise Step7CalculationError(
                f"{compound.phase_key} formula-unit and full-cell initial "
                "formation energies disagree."
            )
        if not (math.isfinite(relaxed) and math.isfinite(initial)):
            raise Step7CalculationError(
                f"{compound.phase_key} formation energy is nonfinite."
            )
        provisional.append(
            {
                "compound": compound,
                "initial": initial,
                "relaxed": relaxed,
            }
        )

    envelope = lower_convex_envelope(
        [(0.0, 0.0)]
        + [
            (entry["compound"].ni_fraction, entry["relaxed"])
            for entry in provisional
        ]
        + [(1.0, 0.0)]
    )
    envelope_tolerance = config.analysis.envelope_tolerance_eV_per_atom
    records: list[FormationEnergyRecord] = []
    for entry in provisional:
        compound = entry["compound"]
        envelope_value = envelope_energy(envelope, compound.ni_fraction)
        above = entry["relaxed"] - envelope_value
        if above < -envelope_tolerance:
            raise Step7CalculationError(
                f"{compound.phase_key} lies below its own selected-set "
                "envelope; the envelope construction is inconsistent."
            )
        records.append(
            FormationEnergyRecord(
                phase_key=compound.phase_key,
                material_id=compound.material_id,
                reduced_formula=compound.reduced_formula,
                formula_unit_al=compound.formula_unit_al,
                formula_unit_ni=compound.formula_unit_ni,
                formula_units_in_cell=compound.formula_units_in_cell,
                cell_al_count=compound.cell_al_count,
                cell_ni_count=compound.cell_ni_count,
                cell_atom_count=compound.cell_atom_count,
                ni_fraction=compound.ni_fraction,
                initial_compound_total_energy_eV=(
                    compound.initial_total_energy_eV
                ),
                relaxed_compound_total_energy_eV=(
                    compound.final_total_energy_eV
                ),
                initial_formation_energy_eV_per_atom=entry["initial"],
                relaxed_formation_energy_eV_per_atom=entry["relaxed"],
                relaxation_effect_eV_per_atom=(
                    entry["relaxed"] - entry["initial"]
                ),
                envelope_energy_eV_per_atom=envelope_value,
                energy_above_envelope_eV_per_atom=max(above, 0.0),
                on_selected_set_envelope=abs(above) <= envelope_tolerance,
                compound_convergence_status=compound.convergence_status,
                elemental_convergence_status=(
                    f"Al={elemental_results['Al'].convergence_status}; "
                    f"Ni={elemental_results['Ni'].convergence_status}"
                ),
                safety_status="PASS",
                checkpoint_path=compound.checkpoint_path,
                selected_structure_path=compound.selected_structure_path,
            )
        )
    return tuple(records)


def lower_convex_envelope(
    points: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], ...]:
    """Build the lower convex envelope of (x, y) points deterministically.

    This is a selected-set construction only.  It uses Andrew's monotone
    chain over the provided points and must never be labeled a complete
    Ni-Al convex hull.
    """

    if len(points) < 2:
        raise Step7CalculationError(
            "The selected-set envelope needs at least two points."
        )
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)) or not (0.0 <= x <= 1.0):
            raise Step7CalculationError(
                f"Envelope point ({x!r}, {y!r}) is invalid."
            )
    ordered = sorted(points)
    hull: list[tuple[float, float]] = []
    for point in ordered:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
            # A non-positive cross product means the middle point is above or
            # on the segment; remove it to keep only the lower boundary.
            if cross <= 0.0:
                hull.pop()
            else:
                break
        hull.append(point)
    return tuple(hull)


def envelope_energy(
    envelope: Sequence[tuple[float, float]], x: float
) -> float:
    """Interpolate the selected-set lower envelope at one composition."""

    if not envelope:
        raise Step7CalculationError("The envelope is empty.")
    if x < envelope[0][0] - 1e-12 or x > envelope[-1][0] + 1e-12:
        raise Step7CalculationError(
            f"Composition {x!r} is outside the envelope domain."
        )
    for (x1, y1), (x2, y2) in zip(envelope, envelope[1:]):
        if x1 - 1e-12 <= x <= x2 + 1e-12:
            if math.isclose(x1, x2, abs_tol=1e-15):
                return min(y1, y2)
            fraction = (x - x1) / (x2 - x1)
            return y1 + fraction * (y2 - y1)
    raise Step7CalculationError(
        f"Composition {x!r} could not be interpolated on the envelope."
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def lazy_pyplot() -> Any:
    """Load Matplotlib headlessly without importing any calculator package."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise Step7DependencyError(
            f"Matplotlib plotting is unavailable: {exc}"
        ) from exc
    return plt


def save_figure(fig: Any, path: Path, plt: Any) -> None:
    """Save one staged PNG and always release Matplotlib state."""

    try:
        fig.tight_layout()
        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
            metadata={
                "Title": "Ni-Al Step 7 formation-energy analysis",
                "Software": "matplotlib",
            },
        )
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
        raise Step7PublicationError(
            f"Figure was not created correctly: {path.name}"
        )
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise Step7PublicationError(
                f"Figure is not a valid PNG: {path.name}"
            )


def render_elemental_convergence_figures(
    config: Step7Config,
    history_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Mapping[str, Path],
) -> None:
    """Render the four elemental convergence figures into staged paths.

    ``history_rows`` maps each element to its flat per-step history rows
    (the same rows serialized in the history CSV and checkpoint), so both
    fresh executions and validated resumed checkpoints can render figures.
    ``targets`` maps figure keys (``energy``, ``force``, ``stress``,
    ``volume``) to staged file paths.  Elements are drawn in deterministic
    order with fixed per-element colors.
    """

    plt = lazy_pyplot()
    ordered = [
        element for element in ELEMENT_ORDER if element in history_rows
    ]
    if not ordered:
        raise Step7CalculationError("No elemental histories were provided.")
    specifications = (
        (
            "energy",
            "energy_per_atom_eV",
            "MACE energy per atom (eV/atom)",
            "Elemental reference energy convergence (MACE-MP-0 Small)",
            None,
        ),
        (
            "force",
            "maximum_force_eV_per_A",
            "Maximum atomic force (eV/angstrom)",
            "Elemental reference force convergence (MACE-MP-0 Small)",
            config.relaxation.force_threshold_eV_per_A,
        ),
        (
            "stress",
            "maximum_absolute_stress_eV_per_A3",
            "Maximum |stress component| (eV/angstrom^3)",
            "Elemental reference stress convergence (MACE-MP-0 Small)",
            config.relaxation.stress_threshold_eV_per_A3,
        ),
        (
            "volume",
            "volume_per_atom_A3",
            "Volume per atom (angstrom^3/atom)",
            "Elemental reference volume convergence (MACE-MP-0 Small)",
            None,
        ),
    )
    for key, metric, ylabel, title, threshold in specifications:
        path = targets.get(key)
        if path is None:
            raise Step7PublicationError(f"Missing staged figure target: {key}")
        fig, axis = plt.subplots(figsize=(8.4, 5.2))
        for element in ordered:
            rows = history_rows[element]
            if not rows:
                raise Step7CalculationError(
                    f"{element} has no history rows for figure rendering."
                )
            axis.plot(
                [int(row["step"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                color=ELEMENT_COLORS[element],
                label=f"{element} (pure, FCC reference)",
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
        save_figure(fig, path, plt)


__all__ = [
    "ELEMENT_COLORS",
    "ELEMENT_HISTORY_FIELDNAMES",
    "ELEMENT_ORDER",
    "EXPECTED_ATOM_COUNTS",
    "EXPECTED_MATERIAL_IDS",
    "GENERALIZED_FORCE_AUTO_STOP_FMAX",
    "NI_MAGNETIC_LIMITATION",
    "PHASE_ORDER",
    "SCHEMA_VERSION",
    "SELECTED_SET_LIMITATION",
    "STRESS_COMPONENTS",
    "CompoundRecord",
    "ElementState",
    "ElementalCombinedPaths",
    "ElementalOutputPaths",
    "ElementalReferenceRecord",
    "ElementalStructureInput",
    "FormationEnergyRecord",
    "FormationOutputPaths",
    "InitialGeometry",
    "Step7ApiError",
    "Step7CalculationError",
    "Step7CalculatorError",
    "Step7CalculatorSession",
    "Step7CollisionError",
    "Step7Config",
    "Step7ConfigurationError",
    "Step7DependencyError",
    "Step7Error",
    "Step7InputError",
    "Step7PublicationError",
    "Step7ResumeError",
    "Step7SafetyError",
    "analyze_symmetry",
    "assert_mace_not_imported",
    "atomic_write_text",
    "calculate_formation_energies",
    "capture_initial_geometry",
    "element_history_csv_bytes",
    "element_history_row",
    "element_state_from_json",
    "element_state_to_json",
    "elemental_combined_paths",
    "elemental_directories",
    "elemental_output_paths",
    "envelope_energy",
    "evaluate_element_state",
    "extract_chemical_potentials",
    "file_sha256",
    "final_report_path",
    "formation_directories",
    "formation_output_paths",
    "installed_step7_versions",
    "lazy_pyplot",
    "load_step7_calculator",
    "load_step7_config",
    "load_validated_elemental_results",
    "locate_project_root",
    "lower_convex_envelope",
    "publish_files_transactionally",
    "read_strict_json",
    "relative_path",
    "render_elemental_convergence_figures",
    "reset_calculator",
    "run_element_relaxation",
    "save_figure",
    "selected_elemental_paths",
    "snapshot_file",
    "stage_path",
    "utc_timestamp",
    "validate_element_keys",
    "validate_element_state_consistency",
    "validate_frechet_cell_filter_api",
    "validate_selected_elemental_structure",
    "validate_step6_compound_sources",
    "verify_snapshots",
    "write_strict_json_bytes",
]

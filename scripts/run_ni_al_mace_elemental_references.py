"""Calculate MACE-consistent pure Al and pure Ni reference states.

For each selected elemental Materials Project structure the workflow:

1. reads the pristine selected structure and validates it completely;
2. creates an independent working copy and records the original geometry;
3. attaches one shared MACE-MP-0 Small calculator (loaded exactly once per
   batch) to the working copy only;
4. calculates the initial fixed-geometry single-point values;
5. performs a full-cell FIRE/FrechetCellFilter relaxation from the original
   selected structure with the exact Step 6 convergence criteria;
6. validates convergence, safety, symmetry, and provenance; and
7. atomically publishes the per-element and combined result bundles.

Al and Ni are processed independently: neither element's final geometry is
ever used as the other element's input.  Validation-only mode performs every
static check without importing MACE or creating an optimizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7_utils import (
    ELEMENT_HISTORY_FIELDNAMES,
    ELEMENT_ORDER,
    GENERALIZED_FORCE_AUTO_STOP_FMAX,
    NI_MAGNETIC_LIMITATION,
    SCHEMA_VERSION,
    ElementState,
    ElementalOutputPaths,
    ElementalStructureInput,
    Step7CalculationError,
    Step7CalculatorSession,
    Step7CollisionError,
    Step7Config,
    Step7ConfigurationError,
    Step7DependencyError,
    Step7Error,
    Step7InputError,
    Step7PublicationError,
    Step7ResumeError,
    Step7SafetyError,
    analyze_symmetry,
    assert_mace_not_imported,
    capture_initial_geometry,
    element_history_csv_bytes,
    element_history_row,
    element_state_from_json,
    element_state_to_json,
    elemental_combined_paths,
    elemental_directories,
    elemental_output_paths,
    evaluate_element_state,
    file_sha256,
    load_step7_calculator,
    load_step7_config,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    render_elemental_convergence_figures,
    reset_calculator,
    run_element_relaxation,
    stage_path,
    utc_timestamp,
    validate_element_keys,
    validate_element_state_consistency,
    validate_frechet_cell_filter_api,
    validate_selected_elemental_structure,
    verify_snapshots,
    write_strict_json_bytes,
)


LOGGER = logging.getLogger("ni_al_step7.elemental_runner")
DEFAULT_CONFIG = Path("configs/mace_formation_energy.json")


@dataclass(frozen=True)
class ElementResult:
    """Complete validated result for one elemental reference."""

    element: str
    material_id: str
    atom_count: int
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
    initial: ElementState
    final: ElementState
    history_rows: tuple[Mapping[str, Any], ...]
    initial_symmetry: Mapping[str, Any]
    final_symmetry: Mapping[str, Any]
    warnings: tuple[str, ...]
    output_paths: ElementalOutputPaths
    resumed: bool = False

    @property
    def overall_converged(self) -> bool:
        """Return final scientific convergence."""

        return self.final.overall_converged


@dataclass(frozen=True)
class ElementalBatchSummary:
    """Combined outcome returned to the command line and the pipeline."""

    elements: tuple[str, ...]
    results: tuple[ElementResult, ...]
    calculator_loads: int
    calculator_class: str
    executed_elements: tuple[str, ...]
    resumed_elements: tuple[str, ...]
    overall_status: str


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the elemental-reference runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute independent full-cell MACE relaxations of "
            "the selected pure Al and pure Ni reference structures."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 7 configuration path, repository-relative by default.",
    )
    parser.add_argument(
        "--element",
        choices=(*ELEMENT_ORDER, "all"),
        default="all",
        help="Process one element or both (default: all).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and selected files without loading MACE.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Run the real MACE elemental-reference calculations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, compatible, validated successful outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 7 elemental-reference calculation outputs.",
    )
    parser.add_argument(
        "--create-directories",
        action="store_true",
        help="Create the planned output directories (validate-only mode).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure deterministic console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _validate_flags(args: argparse.Namespace) -> None:
    """Reject unsafe command-line flag combinations."""

    if args.overwrite and not args.execute:
        raise Step7ConfigurationError("--overwrite is allowed only with --execute.")
    if args.resume and not args.execute:
        raise Step7ConfigurationError("--resume is allowed only with --execute.")
    if args.resume and args.overwrite:
        raise Step7ConfigurationError(
            "--resume and --overwrite are mutually exclusive safety modes."
        )
    if args.create_directories and not args.validate_only:
        raise Step7ConfigurationError(
            "--create-directories is allowed only with --validate-only."
        )


def _requested_elements(option: str) -> tuple[str, ...]:
    """Normalize the --element option."""

    if option == "all":
        return validate_element_keys(None)
    return validate_element_keys((option,))


def _write_roundtrip_safe_extxyz(atoms: Any, path: Path) -> None:
    """Write one ASE-header EXTXYZ frame with 17-digit Cartesian positions."""

    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
    except ImportError as exc:
        raise Step7DependencyError(f"ASE writer is unavailable: {exc}") from exc
    try:
        import numpy as np

        ase_write(
            path,
            atoms,
            format="extxyz",
            columns=["symbols", "positions"],
            write_info=True,
            write_results=False,
        )
        # ASE's EXTXYZ writer formats Cartesian positions with eight decimal
        # places.  Preserve its generated count/header verbatim, but replace
        # the simple species+position rows with round-trip-safe values.
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(atoms) + 2:
            raise Step7PublicationError(
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
            "\n".join(rewritten) + "\n", encoding="utf-8", newline="\n"
        )
        roundtrip_frames = ase_read(path, index=":", format="extxyz")
        if not isinstance(roundtrip_frames, list) or len(roundtrip_frames) != 1:
            raise Step7PublicationError(
                "High-precision final EXTXYZ did not round-trip as one frame."
            )
        roundtrip = roundtrip_frames[0]
        if not bool(
            np.allclose(
                roundtrip.get_positions(),
                atoms.get_positions(),
                atol=1.0e-12,
                rtol=0.0,
            )
        ) or not bool(
            np.allclose(
                roundtrip.cell.array, atoms.cell.array, atol=1.0e-12, rtol=0.0
            )
        ):
            raise Step7PublicationError(
                "High-precision final EXTXYZ failed the 1e-12 round-trip check."
            )
    except Step7Error:
        raise
    except Exception as exc:
        raise Step7PublicationError(
            f"Could not write round-trip-safe EXTXYZ: {type(exc).__name__}: {exc}"
        ) from exc


def _magnetic_limitation_lines(element: str) -> tuple[str, ...]:
    """Return the documented magnetic-limitation statement for one element."""

    if element == "Ni":
        return NI_MAGNETIC_LIMITATION
    return (
        "Al is not treated as a magnetic element in standard DFT references; "
        "no additional magnetic limitation applies beyond the general "
        "MACE-consistency scope of Step 7.",
    )


def _element_report_text(
    config: Step7Config,
    result: ElementResult,
    structure_input: ElementalStructureInput,
) -> str:
    """Build the complete human-readable elemental reference report."""

    initial = result.initial
    final = result.final
    metadata = structure_input.metadata

    def vector(values: Sequence[float]) -> str:
        return "[" + ", ".join(f"{value:.17g}" for value in values) + "]"

    lines = [
        f"Step 7 - Pure {result.element} MACE Elemental Reference",
        "=" * 76,
        "",
        "1. Identity and provenance",
        "--------------------------",
        f"Element: {result.element}",
        f"Materials Project ID: {result.material_id}",
        f"Atom count: {result.atom_count}",
        "Materials Project database version: "
        f"{metadata.get('materials_project_database_version')}",
        f"Retrieval time (UTC): {metadata.get('retrieval_time_utc')}",
        "Selected structure: "
        + relative_path(
            structure_input.structure_snapshot.path, config.project_root
        ),
        f"Configuration: {relative_path(config.config_path, config.project_root)}",
        f"Configuration SHA-256: {config.fingerprint}",
        f"Started (UTC): {result.started_at_utc}",
        f"Completed (UTC): {result.completed_at_utc}",
        "",
        "2. Model settings",
        "-----------------",
        f"Family: {config.model.family}",
        f"Name: {config.model.name}",
        f"Model value: {config.model.value}",
        f"Device: {config.model.device}",
        f"Default dtype: {config.model.default_dtype}",
        f"Dispersion enabled: {str(config.model.dispersion).lower()}",
        f"Calculator class: {result.calculator_class}",
        f"Calculator loads in session: {result.calculator_load_count}",
        f"Recorded state evaluations: {result.state_evaluations}",
        "",
        "3. Optimizer and filter settings",
        "--------------------------------",
        f"Optimizer: {config.relaxation.optimizer}",
        f"Optimizer created: {result.optimizer_created}",
        f"Maximum steps: {config.relaxation.maximum_steps}",
        f"Actual optimizer steps: {result.optimizer_steps}",
        "Force threshold (eV/angstrom): "
        f"{config.relaxation.force_threshold_eV_per_A:.17g}",
        "Stress threshold (eV/angstrom^3): "
        f"{config.relaxation.stress_threshold_eV_per_A3:.17g}",
        "Filter: ase.filters.FrechetCellFilter",
        f"Hydrostatic strain: {str(config.relaxation.hydrostatic_strain).lower()}",
        f"Constant volume: {str(config.relaxation.constant_volume).lower()}",
        "External pressure (eV/angstrom^3): "
        f"{config.relaxation.external_pressure_eV_per_A3:.17g}",
        (
            "ASE generalized auto-stop fmax: 0.0 eV/angstrom (control "
            "sentinel only; actual force and stress criteria are evaluated "
            "explicitly)"
        ),
        "",
        "4. Initial structure values",
        "---------------------------",
        f"Total energy (eV): {initial.total_energy_eV:.17g}",
        f"Energy per atom (eV/atom): {initial.energy_per_atom_eV:.17g}",
        f"Maximum force (eV/angstrom): {initial.maximum_force_eV_per_A:.17g}",
        f"RMS force (eV/angstrom): {initial.rms_force_eV_per_A:.17g}",
        f"Total force norm (eV/angstrom): {initial.total_force_norm_eV_per_A:.17g}",
        "Stress [xx, yy, zz, yz, xz, xy] (eV/angstrom^3): "
        + vector(initial.stress_eV_per_A3),
        "Maximum |stress| (eV/angstrom^3): "
        f"{initial.maximum_absolute_stress_eV_per_A3:.17g}",
        f"Volume (angstrom^3): {initial.volume_A3:.17g}",
        f"Volume per atom (angstrom^3/atom): {initial.volume_per_atom_A3:.17g}",
        "Cell matrix (angstrom): "
        + json.dumps([list(row) for row in initial.cell_A], separators=(",", ":")),
        f"Lattice lengths (angstrom): {vector(initial.lattice_lengths_A)}",
        f"Lattice angles (degrees): {vector(initial.lattice_angles_deg)}",
        "Initial symmetry: "
        f"{result.initial_symmetry.get('space_group_symbol')} "
        f"({result.initial_symmetry.get('space_group_number')})",
        "",
        "5. Final structure values",
        "-------------------------",
        f"Total energy (eV): {final.total_energy_eV:.17g}",
        f"Energy per atom (eV/atom): {final.energy_per_atom_eV:.17g}",
        f"Maximum force (eV/angstrom): {final.maximum_force_eV_per_A:.17g}",
        f"RMS force (eV/angstrom): {final.rms_force_eV_per_A:.17g}",
        "Stress [xx, yy, zz, yz, xz, xy] (eV/angstrom^3): "
        + vector(final.stress_eV_per_A3),
        "Maximum |stress| (eV/angstrom^3): "
        f"{final.maximum_absolute_stress_eV_per_A3:.17g}",
        f"Volume (angstrom^3): {final.volume_A3:.17g}",
        f"Volume per atom (angstrom^3/atom): {final.volume_per_atom_A3:.17g}",
        "Cell matrix (angstrom): "
        + json.dumps([list(row) for row in final.cell_A], separators=(",", ":")),
        f"Lattice lengths (angstrom): {vector(final.lattice_lengths_A)}",
        f"Lattice angles (degrees): {vector(final.lattice_angles_deg)}",
        "Final symmetry: "
        f"{result.final_symmetry.get('space_group_symbol')} "
        f"({result.final_symmetry.get('space_group_number')})",
        "",
        "6. Changes",
        "----------",
        f"Energy change (eV): {final.total_energy_eV - initial.total_energy_eV:.17g}",
        "Energy change per atom (eV/atom): "
        f"{final.energy_per_atom_eV - initial.energy_per_atom_eV:.17g}",
        f"Volume change (angstrom^3): {final.volume_A3 - initial.volume_A3:.17g}",
        f"Volume change (percent): {final.volume_change_percent:.17g}",
        "Lattice-length changes (angstrom): "
        + vector(
            [
                final.lattice_lengths_A[index] - initial.lattice_lengths_A[index]
                for index in range(3)
            ]
        ),
        "Lattice-angle changes (degrees): "
        + vector(
            [
                final.lattice_angles_deg[index] - initial.lattice_angles_deg[index]
                for index in range(3)
            ]
        ),
        "Maximum internal displacement (angstrom): "
        f"{final.maximum_internal_displacement_A:.17g}",
        "Maximum total Cartesian displacement (angstrom): "
        f"{final.maximum_total_displacement_A:.17g}",
        "",
        "7. Chemical-potential candidate",
        "-------------------------------",
        "mu_"
        f"{result.element}_MACE candidate (relaxed total energy / atoms): "
        f"{final.energy_per_atom_eV:.17g} eV/atom",
        (
            "This value becomes a valid chemical potential only because the "
            "convergence and safety checks below passed."
        ),
        "",
        "8. Safety and provenance checks",
        "-------------------------------",
        f"Safety status: {result.safety_status}",
        "Energy/forces/stress/positions/cell finite: PASS",
        "Positive cell determinant and volume: PASS",
        "Atom identity, ordering, and PBC preserved: PASS",
        "Configured displacement and volume limits: PASS",
        "Protected source files unchanged: PASS",
        "Protected file fingerprints:",
        "  "
        + relative_path(structure_input.structure_snapshot.path, config.project_root)
        + f": sha256={structure_input.structure_snapshot.sha256}; "
        + f"size={structure_input.structure_snapshot.size}",
        "  "
        + relative_path(structure_input.metadata_snapshot.path, config.project_root)
        + f": sha256={structure_input.metadata_snapshot.sha256}; "
        + f"size={structure_input.metadata_snapshot.size}",
        "",
        "9. Output paths",
        "---------------",
    ]
    for name, path in (
        ("Initial single point", result.output_paths.single_point_json),
        ("Final structure", result.output_paths.structure),
        ("Trajectory", result.output_paths.trajectory),
        ("History CSV", result.output_paths.history_csv),
        ("Report", result.output_paths.report),
        ("Optimizer log", result.output_paths.log),
        ("Checkpoint", result.output_paths.checkpoint),
    ):
        lines.append(f"{name}: {relative_path(path, config.project_root)}")
    lines.extend(
        [
            "",
            "10. Convergence status",
            "----------------------",
            f"Force convergence: {'PASS' if final.force_converged else 'FAIL'}",
            f"Stress convergence: {'PASS' if final.stress_converged else 'FAIL'}",
            f"Overall convergence: {'PASS' if final.overall_converged else 'FAIL'}",
            f"Element status: {result.status}",
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
            "12. Magnetic limitation",
            "-----------------------",
            *(f"- {line}" for line in _magnetic_limitation_lines(result.element)),
            "",
            "13. Scientific interpretation boundary",
            "--------------------------------------",
            (
                "This is a geometry relaxation on the configured MACE "
                "potential-energy surface. It is not a DFT or experimental "
                "accuracy conclusion, and this raw elemental energy is never "
                "ranked against compound energies across compositions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _element_checkpoint_document(
    config: Step7Config,
    result: ElementResult,
    structure_input: ElementalStructureInput,
    staged_named: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the machine-readable resume authority for one element."""

    outputs = result.output_paths
    artifact_targets = {
        "single_point_json": outputs.single_point_json,
        "final_structure": outputs.structure,
        "trajectory": outputs.trajectory,
        "history_csv": outputs.history_csv,
        "report": outputs.report,
        "optimizer_log": outputs.log,
    }
    artifacts = {
        name: {
            "path": relative_path(artifact_targets[name], config.project_root),
            "sha256": file_sha256(staged_named[name]),
            "size_bytes": staged_named[name].stat().st_size,
        }
        for name in artifact_targets
    }
    initial = result.initial
    final = result.final
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_elemental_reference_result",
        "project_step": "7",
        "element": result.element,
        "material_id": result.material_id,
        "number_of_atoms": result.atom_count,
        "generated_at_utc": result.completed_at_utc,
        "configuration_path": relative_path(config.config_path, config.project_root),
        "configuration_fingerprint_sha256": config.fingerprint,
        "execution_status": "COMPLETED",
        "convergence_status": result.status,
        "force_converged": final.force_converged,
        "stress_converged": final.stress_converged,
        "overall_converged": final.overall_converged,
        "safety_status": result.safety_status,
        "output_validation_status": "PASS",
        "model": {
            **config.model.to_json(),
            "calculator_class": result.calculator_class,
        },
        "optimizer": {
            "name": config.relaxation.optimizer,
            "created": result.optimizer_created,
            "steps": result.optimizer_steps,
            "maximum_steps": config.relaxation.maximum_steps,
            "force_threshold_eV_per_A": (
                config.relaxation.force_threshold_eV_per_A
            ),
            "stress_threshold_eV_per_A3": (
                config.relaxation.stress_threshold_eV_per_A3
            ),
            "trajectory_interval": config.relaxation.trajectory_interval,
            "ase_generalized_auto_stop_fmax_eV_per_A": (
                GENERALIZED_FORCE_AUTO_STOP_FMAX
            ),
            "filter": {
                "name": "FrechetCellFilter",
                "mask": "all",
                "exp_cell_factor": float(result.atom_count),
                "hydrostatic_strain": config.relaxation.hydrostatic_strain,
                "constant_volume": config.relaxation.constant_volume,
                "external_pressure_eV_per_A3": (
                    config.relaxation.external_pressure_eV_per_A3
                ),
            },
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
        "initial": element_state_to_json(initial),
        "final": element_state_to_json(final),
        "initial_symmetry": dict(result.initial_symmetry),
        "final_symmetry": dict(result.final_symmetry),
        "lattice_changes": {
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
        },
        "chemical_potential_candidate_eV_per_atom": final.energy_per_atom_eV,
        "history_rows": [dict(row) for row in result.history_rows],
        "warnings": list(result.warnings),
        "magnetic_limitation": list(_magnetic_limitation_lines(result.element)),
        "source": {
            "selected_structure_path": relative_path(
                structure_input.structure_snapshot.path, config.project_root
            ),
            "selected_metadata_path": relative_path(
                structure_input.metadata_snapshot.path, config.project_root
            ),
            "materials_project_database_version": (
                structure_input.metadata.get(
                    "materials_project_database_version"
                )
            ),
            "retrieval_time_utc": structure_input.metadata.get(
                "retrieval_time_utc"
            ),
        },
        "protected_sources": [
            structure_input.structure_snapshot.to_json(config.project_root),
            structure_input.metadata_snapshot.to_json(config.project_root),
            config.config_snapshot.to_json(config.project_root),
        ],
        "artifacts": artifacts,
        "scientific_limitations": [
            "MACE-potential reference state; not a DFT or experimental "
            "validation.",
            "Raw elemental energies are not ranked against compounds.",
            "The reference is valid only together with its convergence and "
            "safety statuses.",
        ],
    }


def _execute_element_to_staging(
    config: Step7Config,
    structure_input: ElementalStructureInput,
    session: Step7CalculatorSession,
    staging_root: Path,
) -> tuple[ElementResult, dict[Path, Path]]:
    """Execute one independent element into temporary staged files."""

    element = structure_input.element
    outputs = elemental_output_paths(config, element)
    output_root = config.output.elemental_reference_root
    staged_by_final = {
        target: stage_path(staging_root, output_root, target)
        for target in outputs.all_paths()
    }
    staged_named = {
        "single_point_json": staged_by_final[outputs.single_point_json],
        "final_structure": staged_by_final[outputs.structure],
        "trajectory": staged_by_final[outputs.trajectory],
        "history_csv": staged_by_final[outputs.history_csv],
        "report": staged_by_final[outputs.report],
        "optimizer_log": staged_by_final[outputs.log],
        "checkpoint": staged_by_final[outputs.checkpoint],
    }
    source_atoms = structure_input.atoms
    if source_atoms.calc is not None:
        raise Step7InputError(
            f"Pristine source {element} unexpectedly has a calculator."
        )
    working = source_atoms.copy()
    working.calc = None
    initial_geometry = capture_initial_geometry(working)
    started_at = utc_timestamp()
    started_monotonic = time.monotonic()
    state_count_before = session.state_evaluations
    initial_symmetry = {
        "space_group_symbol": structure_input.space_group_symbol,
        "space_group_number": structure_input.space_group_number,
        "crystal_system": structure_input.crystal_system,
        "symprec_A": config.symmetry.symprec_A,
        "angle_tolerance_deg": config.symmetry.angle_tolerance_deg,
    }
    try:
        working.calc = session.calculator
        initial_state = evaluate_element_state(
            working, initial_geometry, config, element, 0, 0.0, session
        )
        (
            status,
            optimizer_created,
            optimizer_steps,
            history,
            calculation_warnings,
        ) = run_element_relaxation(
            config,
            element,
            working,
            initial_geometry,
            initial_state,
            session,
            staged_named["trajectory"],
            staged_named["optimizer_log"],
            started_monotonic,
        )
    finally:
        working.calc = None
        try:
            verify_snapshots(
                (
                    structure_input.structure_snapshot,
                    structure_input.metadata_snapshot,
                    config.config_snapshot,
                )
            )
        finally:
            reset_calculator(session, element)
    if source_atoms.calc is not None or working.calc is not None:
        raise Step7SafetyError(f"Calculator detachment failed for {element}.")

    final_state = history[-1]
    if status in {"ALREADY_CONVERGED", "CONVERGED"}:
        if not final_state.overall_converged:
            raise Step7CalculationError(
                f"{element} was labeled {status} without exact convergence."
            )
    elif final_state.overall_converged:
        raise Step7CalculationError(
            f"{element} was labeled NOT_CONVERGED despite exact convergence."
        )
    final_symmetry = analyze_symmetry(working, config)
    completed_at = utc_timestamp()
    result = ElementResult(
        element=element,
        material_id=structure_input.material_id,
        atom_count=structure_input.atom_count,
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
        history_rows=tuple(element_history_row(state) for state in history),
        initial_symmetry=initial_symmetry,
        final_symmetry=final_symmetry,
        warnings=tuple(session.warnings) + calculation_warnings,
        output_paths=outputs,
    )

    single_point_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_elemental_initial_single_point",
        "project_step": "7",
        "element": element,
        "material_id": result.material_id,
        "number_of_atoms": result.atom_count,
        "generated_at_utc": completed_at,
        "configuration_fingerprint_sha256": config.fingerprint,
        "model": {
            **config.model.to_json(),
            "calculator_class": result.calculator_class,
        },
        "state": element_state_to_json(initial_state),
        "symmetry": dict(initial_symmetry),
        "source_structure_path": relative_path(
            structure_input.structure_snapshot.path, config.project_root
        ),
        "note": (
            "Fixed-geometry MACE single point of the pristine selected "
            "structure before any relaxation."
        ),
    }
    staged_named["single_point_json"].write_bytes(
        write_strict_json_bytes(single_point_document)
    )

    output_atoms = working.copy()
    output_atoms.calc = None
    output_atoms.info.update(
        {
            "step7_element": element,
            "step7_material_id": result.material_id,
            "step7_role": "mace_relaxed_elemental_reference",
            "step7_model": config.model.name,
            "step7_model_value": config.model.value,
            "step7_device": config.model.device,
            "step7_dtype": config.model.default_dtype,
            "step7_convergence_status": status,
            "step7_optimizer_steps": optimizer_steps,
            "step7_initial_energy_eV": initial_state.total_energy_eV,
            "step7_final_energy_eV": final_state.total_energy_eV,
            "step7_final_energy_per_atom_eV": final_state.energy_per_atom_eV,
            "step7_execution_timestamp_utc": completed_at,
            "step7_configuration_sha256": config.fingerprint,
        }
    )
    _write_roundtrip_safe_extxyz(output_atoms, staged_named["final_structure"])
    staged_named["history_csv"].write_bytes(
        element_history_csv_bytes(history)
    )
    staged_named["report"].write_text(
        _element_report_text(config, result, structure_input),
        encoding="utf-8",
        newline="\n",
    )
    checkpoint = _element_checkpoint_document(
        config, result, structure_input, staged_named
    )
    staged_named["checkpoint"].write_bytes(write_strict_json_bytes(checkpoint))
    for name, path in staged_named.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise Step7PublicationError(
                f"Staged {name} for {element} is absent or empty: {path}"
            )
    read_strict_json(staged_named["checkpoint"], f"staged {element} checkpoint")
    return result, staged_by_final


def validate_element_bundle(
    config: Step7Config,
    structure_input: ElementalStructureInput,
) -> ElementResult:
    """Validate one complete published element bundle without MACE."""

    element = structure_input.element
    outputs = elemental_output_paths(config, element)
    missing = [path for path in outputs.all_paths() if not path.is_file()]
    if missing:
        raise Step7ResumeError(
            f"{element} bundle is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    checkpoint = read_strict_json(outputs.checkpoint, f"{element} checkpoint")
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("artifact_type")
        != "ni_al_mace_elemental_reference_result"
        or checkpoint.get("element") != element
        or checkpoint.get("material_id") != structure_input.material_id
        or checkpoint.get("number_of_atoms") != structure_input.atom_count
        or checkpoint.get("configuration_fingerprint_sha256")
        != config.fingerprint
        or checkpoint.get("execution_status") != "COMPLETED"
        or checkpoint.get("safety_status") != "PASS"
        or checkpoint.get("output_validation_status") != "PASS"
    ):
        raise Step7ResumeError(
            f"{element} checkpoint identity/status is not resume-eligible."
        )
    model = checkpoint.get("model")
    if not isinstance(model, Mapping) or any(
        model.get(key) != value for key, value in config.model.to_json().items()
    ):
        raise Step7ResumeError(f"{element} checkpoint model settings mismatch.")
    status = checkpoint.get("convergence_status")
    if status not in {"ALREADY_CONVERGED", "CONVERGED"}:
        raise Step7ResumeError(
            f"{element} status {status!r} is not resume-eligible."
        )
    initial = element_state_from_json(
        checkpoint.get("initial"), f"{element}.initial", structure_input.atom_count
    )
    final = element_state_from_json(
        checkpoint.get("final"), f"{element}.final", structure_input.atom_count
    )
    validate_element_state_consistency(
        initial, config, structure_input.atom_count, initial.volume_A3,
        f"{element}.initial",
    )
    validate_element_state_consistency(
        final, config, structure_input.atom_count, initial.volume_A3,
        f"{element}.final",
    )
    if not final.overall_converged:
        raise Step7ResumeError(
            f"{element} final state does not satisfy the convergence criteria."
        )
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Step7ResumeError(f"{element} checkpoint artifacts are invalid.")
    artifact_targets = {
        "single_point_json": outputs.single_point_json,
        "final_structure": outputs.structure,
        "trajectory": outputs.trajectory,
        "history_csv": outputs.history_csv,
        "report": outputs.report,
        "optimizer_log": outputs.log,
    }
    for name, target in artifact_targets.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise Step7ResumeError(f"{element} artifact record {name} is absent.")
        if record.get("path") != relative_path(target, config.project_root):
            raise Step7ResumeError(f"{element} artifact path mismatch: {name}.")
        if (
            record.get("sha256") != file_sha256(target)
            or record.get("size_bytes") != target.stat().st_size
        ):
            raise Step7ResumeError(
                f"{element} artifact fingerprint mismatch: {name}."
            )
    history_rows = checkpoint.get("history_rows")
    if not isinstance(history_rows, list) or not history_rows:
        raise Step7ResumeError(f"{element} history rows are absent.")
    optimizer = checkpoint.get("optimizer")
    steps = optimizer.get("steps") if isinstance(optimizer, Mapping) else None
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise Step7ResumeError(f"{element} optimizer step count is invalid.")
    if (
        int(history_rows[0].get("step", -1)) != 0
        or int(history_rows[-1].get("step", -1)) != steps
    ):
        raise Step7ResumeError(f"{element} history endpoints are invalid.")

    try:
        from ase.io import read as ase_read
        from ase.io.trajectory import Trajectory

        import numpy as np

        final_frames = ase_read(outputs.structure, index=":", format="extxyz")
        with Trajectory(outputs.trajectory, mode="r") as trajectory:
            trajectory_frames = [frame for frame in trajectory]
    except Exception as exc:
        raise Step7ResumeError(
            f"Could not read {element} published artifacts: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(final_frames, list) or len(final_frames) != 1:
        raise Step7ResumeError(f"{element} final EXTXYZ must contain one frame.")
    final_atoms = final_frames[0]
    final_atoms.calc = None
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
        raise Step7ResumeError(
            f"{element} final EXTXYZ geometry disagrees with its checkpoint."
        )
    if not trajectory_frames:
        raise Step7ResumeError(f"{element} trajectory has no frames.")
    if not bool(
        np.allclose(
            trajectory_frames[0].get_positions(),
            structure_input.atoms.get_positions(),
            atol=1.0e-12,
            rtol=0.0,
        )
    ) or not bool(
        np.allclose(
            trajectory_frames[-1].get_positions(),
            final_atoms.get_positions(),
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise Step7ResumeError(
            f"{element} trajectory endpoints do not match source/final states."
        )
    initial_symmetry = checkpoint.get("initial_symmetry")
    final_symmetry = checkpoint.get("final_symmetry")
    if not isinstance(initial_symmetry, Mapping) or not isinstance(
        final_symmetry, Mapping
    ):
        raise Step7ResumeError(f"{element} symmetry records are invalid.")
    timing = checkpoint.get("timing", {})
    counts = checkpoint.get("counts", {})
    warnings_raw = checkpoint.get("warnings")
    if not isinstance(warnings_raw, list):
        raise Step7ResumeError(f"{element} warnings must be an array.")
    verify_snapshots(
        (structure_input.structure_snapshot, structure_input.metadata_snapshot)
    )
    return ElementResult(
        element=element,
        material_id=structure_input.material_id,
        atom_count=structure_input.atom_count,
        status=str(status),
        safety_status="PASS",
        optimizer_created=bool(optimizer.get("created")),
        optimizer_steps=steps,
        state_evaluations=int(counts.get("state_evaluations", 1)),
        calculator_class=str(model.get("calculator_class")),
        calculator_load_count=int(counts.get("calculator_loads_in_session", 1)),
        started_at_utc=str(timing.get("started_at_utc")),
        completed_at_utc=str(timing.get("completed_at_utc")),
        wall_time_seconds=float(timing.get("wall_time_seconds", 0.0)),
        initial=initial,
        final=final,
        history_rows=tuple(dict(row) for row in history_rows),
        initial_symmetry=initial_symmetry,
        final_symmetry=final_symmetry,
        warnings=tuple(str(item) for item in warnings_raw),
        output_paths=outputs,
        resumed=True,
    )


def _combined_summary_documents(
    config: Step7Config,
    results: Sequence[ElementResult],
    session: Step7CalculatorSession | None,
) -> tuple[dict[str, Any], bytes, str]:
    """Build the combined JSON document, CSV bytes, and text report."""

    import csv as csv_module
    import io as io_module

    overall = (
        "SUCCESS"
        if all(result.overall_converged for result in results)
        else "PARTIAL"
    )
    records = []
    for result in results:
        records.append(
            {
                "element": result.element,
                "material_id": result.material_id,
                "number_of_atoms": result.atom_count,
                "status": result.status,
                "safety_status": result.safety_status,
                "optimizer_steps": result.optimizer_steps,
                "state_evaluations": result.state_evaluations,
                "wall_time_seconds": result.wall_time_seconds,
                "initial_total_energy_eV": result.initial.total_energy_eV,
                "final_total_energy_eV": result.final.total_energy_eV,
                "initial_energy_per_atom_eV": result.initial.energy_per_atom_eV,
                "final_energy_per_atom_eV": result.final.energy_per_atom_eV,
                "initial_maximum_force_eV_per_A": (
                    result.initial.maximum_force_eV_per_A
                ),
                "final_maximum_force_eV_per_A": (
                    result.final.maximum_force_eV_per_A
                ),
                "initial_maximum_absolute_stress_eV_per_A3": (
                    result.initial.maximum_absolute_stress_eV_per_A3
                ),
                "final_maximum_absolute_stress_eV_per_A3": (
                    result.final.maximum_absolute_stress_eV_per_A3
                ),
                "initial_volume_A3": result.initial.volume_A3,
                "final_volume_A3": result.final.volume_A3,
                "volume_change_percent": result.final.volume_change_percent,
                "initial_volume_per_atom_A3": result.initial.volume_per_atom_A3,
                "final_volume_per_atom_A3": result.final.volume_per_atom_A3,
                "initial_space_group": (
                    f"{result.initial_symmetry.get('space_group_symbol')} "
                    f"({result.initial_symmetry.get('space_group_number')})"
                ),
                "final_space_group": (
                    f"{result.final_symmetry.get('space_group_symbol')} "
                    f"({result.final_symmetry.get('space_group_number')})"
                ),
                "chemical_potential_candidate_eV_per_atom": (
                    result.final.energy_per_atom_eV
                ),
                "resumed": result.resumed,
                "checkpoint": relative_path(
                    result.output_paths.checkpoint, config.project_root
                ),
            }
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_elemental_reference_summary",
        "project_step": "7",
        "generated_at_utc": utc_timestamp(),
        "configuration_path": relative_path(
            config.config_path, config.project_root
        ),
        "configuration_fingerprint_sha256": config.fingerprint,
        "elements": [result.element for result in results],
        "overall_status": overall,
        "model": {
            **config.model.to_json(),
            "calculator_class": (
                session.calculator_class
                if session is not None
                else results[0].calculator_class
            ),
        },
        "execution": {
            "calculator_loads_this_invocation": (
                session.load_count if session is not None else 0
            ),
            "state_evaluations": sum(
                result.state_evaluations for result in results
            ),
            "optimizer_steps": sum(result.optimizer_steps for result in results),
            "executed_elements": [
                result.element for result in results if not result.resumed
            ],
            "resumed_elements": [
                result.element for result in results if result.resumed
            ],
            "optimizer": config.relaxation.optimizer,
            "filter": "FrechetCellFilter",
        },
        "convergence": {
            "force_threshold_eV_per_A": (
                config.relaxation.force_threshold_eV_per_A
            ),
            "stress_threshold_eV_per_A3": (
                config.relaxation.stress_threshold_eV_per_A3
            ),
            "requires_actual_atomic_force": True,
            "requires_actual_six_component_stress": True,
        },
        "records": records,
        "magnetic_limitation_Ni": list(NI_MAGNETIC_LIMITATION),
        "scientific_limitations": [
            "MACE-potential reference states; not DFT or experimental "
            "validation.",
            "Raw elemental energies are not ranked across compositions.",
            "These references are consumed only by the Step 7 "
            "formation-energy calculation.",
        ],
    }
    fieldnames = tuple(records[0].keys())
    buffer = io_module.StringIO(newline="")
    writer = csv_module.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    csv_bytes = buffer.getvalue().encode("utf-8")
    lines = [
        "Step 7 - MACE Elemental Reference Summary",
        "=" * 60,
        "",
        f"Generated (UTC): {document['generated_at_utc']}",
        f"Configuration SHA-256: {config.fingerprint}",
        f"Elements: {', '.join(result.element for result in results)}",
        f"Overall status: {overall}",
        "",
        "Per-element results",
        "-------------------",
    ]
    for result in results:
        lines.extend(
            [
                f"{result.element} ({result.material_id}): {result.status}",
                f"  atoms={result.atom_count}",
                f"  optimizer_steps={result.optimizer_steps}",
                "  initial_energy_per_atom_eV="
                f"{result.initial.energy_per_atom_eV:.17g}",
                "  final_energy_per_atom_eV="
                f"{result.final.energy_per_atom_eV:.17g}",
                "  final_max_force_eV_per_A="
                f"{result.final.maximum_force_eV_per_A:.17g}",
                "  final_max_abs_stress_eV_per_A3="
                f"{result.final.maximum_absolute_stress_eV_per_A3:.17g}",
                "  volume_change_percent="
                f"{result.final.volume_change_percent:.17g}",
                "  final_space_group="
                f"{result.final_symmetry.get('space_group_symbol')} "
                f"({result.final_symmetry.get('space_group_number')})",
            ]
        )
    lines.extend(
        [
            "",
            "Magnetic limitation for Ni",
            "--------------------------",
            *(f"- {line}" for line in NI_MAGNETIC_LIMITATION),
            "",
            "Interpretation boundary",
            "-----------------------",
            (
                "These are MACE-potential reference states, not DFT or "
                "experimental validation. Raw elemental energies are never "
                "ranked against compound energies across compositions."
            ),
            "",
        ]
    )
    return document, csv_bytes, "\n".join(lines)


def run_validate_only(
    config: Step7Config,
    elements: Sequence[str],
    create_directories: bool,
) -> None:
    """Validate configuration and selected inputs without loading MACE."""

    ase_version = validate_frechet_cell_filter_api()
    inputs = {
        element: validate_selected_elemental_structure(config, element)
        for element in elements
    }
    collisions: list[Path] = []
    for element in elements:
        collisions.extend(
            path
            for path in elemental_output_paths(config, element).all_paths()
            if path.exists()
        )
    if tuple(elements) == ELEMENT_ORDER:
        collisions.extend(
            path
            for path in elemental_combined_paths(config).all_paths()
            if path.exists()
        )
    if create_directories:
        for directory in elemental_directories(config):
            directory.mkdir(parents=True, exist_ok=True)
    assert_mace_not_imported()
    print("=" * 78)
    print("STEP 7 ELEMENTAL REFERENCE VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Elements: {', '.join(elements)}")
    for element in elements:
        item = inputs[element]
        print(
            f"{element}: {item.material_id}; {item.atom_count} atom(s); "
            f"{item.space_group_symbol} ({item.space_group_number}); "
            "validation PASS"
        )
    print(f"Validated FrechetCellFilter public API in ASE {ase_version}")
    print(
        "Existing Step 7 elemental outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print(f"Directories created: {'Yes' if create_directories else 'No'}")
    print("MACE loaded: No")
    print("Optimizer created: No")
    print("Relaxation executed: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def execute_elements(
    config: Step7Config,
    elements: Sequence[str],
    *,
    overwrite: bool,
    resume: bool,
    session: Step7CalculatorSession | None = None,
) -> ElementalBatchSummary:
    """Execute or resume the requested elemental references and publish."""

    selected = validate_element_keys(elements)
    publish_summary = selected == ELEMENT_ORDER
    inputs = {
        element: validate_selected_elemental_structure(config, element)
        for element in selected
    }
    for directory in elemental_directories(config):
        directory.mkdir(parents=True, exist_ok=True)

    resumed_results: dict[str, ElementResult] = {}
    to_execute: list[str] = []
    collisions: list[Path] = []
    for element in selected:
        outputs = elemental_output_paths(config, element)
        existing = tuple(path for path in outputs.all_paths() if path.exists())
        if existing and resume:
            resumed_results[element] = validate_element_bundle(
                config, inputs[element]
            )
            LOGGER.info("%s: reusing the validated existing bundle.", element)
            continue
        if existing and not overwrite:
            collisions.extend(existing)
        to_execute.append(element)
    combined = elemental_combined_paths(config)
    existing_combined = (
        tuple(path for path in combined.all_paths() if path.exists())
        if publish_summary
        else ()
    )
    if collisions:
        listing = "\n".join(
            f"  - {relative_path(path, config.project_root)}"
            for path in sorted(set(collisions))
        )
        raise Step7CollisionError(
            "Existing Step 7 elemental output collisions were found:\n" + listing
        )
    if publish_summary and existing_combined and not (overwrite or resume):
        listing = "\n".join(
            f"  - {relative_path(path, config.project_root)}"
            for path in existing_combined
        )
        raise Step7CollisionError(
            "Existing combined elemental output collisions were found:\n"
            + listing
        )

    if to_execute:
        if session is None:
            LOGGER.info(
                "Loading the MACE calculator once for element(s): %s",
                ", ".join(to_execute),
            )
            session = load_step7_calculator(config)
        elif session.configuration_fingerprint != config.fingerprint:
            raise Step7CalculationError(
                "Provided calculator session does not match this configuration."
            )

    with tempfile.TemporaryDirectory(
        prefix=".step7-elemental-staging-",
        dir=config.output.elemental_reference_root,
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        all_results: dict[str, ElementResult] = dict(resumed_results)
        for element in to_execute:
            if session is None:
                raise Step7CalculationError("Calculator session was not created.")
            LOGGER.info("Executing %s elemental reference...", element)
            result, staged = _execute_element_to_staging(
                config, inputs[element], session, staging_root
            )
            all_results[element] = result
            staged_by_final.update(staged)
            LOGGER.info(
                "%s: %s after %d step(s); final %0.9f eV/atom.",
                element,
                result.status,
                result.optimizer_steps,
                result.final.energy_per_atom_eV,
            )

        ordered_results = tuple(all_results[element] for element in selected)
        summary_document: dict[str, Any] | None = None
        if publish_summary:
            need_combined = (
                bool(to_execute)
                or overwrite
                or len(existing_combined) != len(combined.all_paths())
            )
            if need_combined:
                summary_document, csv_bytes, report_text = (
                    _combined_summary_documents(config, ordered_results, session)
                )
                output_root = config.output.elemental_reference_root
                for target, payload in (
                    (combined.csv, csv_bytes),
                    (combined.json, write_strict_json_bytes(summary_document)),
                    (combined.report, report_text.encode("utf-8")),
                ):
                    staged = stage_path(staging_root, output_root, target)
                    staged.write_bytes(payload)
                    staged_by_final[target] = staged
                figure_targets = {
                    "energy": stage_path(
                        staging_root, output_root, combined.energy_figure
                    ),
                    "force": stage_path(
                        staging_root, output_root, combined.force_figure
                    ),
                    "stress": stage_path(
                        staging_root, output_root, combined.stress_figure
                    ),
                    "volume": stage_path(
                        staging_root, output_root, combined.volume_figure
                    ),
                }
                render_elemental_convergence_figures(
                    config,
                    {
                        result.element: result.history_rows
                        for result in ordered_results
                    },
                    figure_targets,
                )
                staged_by_final[combined.energy_figure] = figure_targets["energy"]
                staged_by_final[combined.force_figure] = figure_targets["force"]
                staged_by_final[combined.stress_figure] = figure_targets["stress"]
                staged_by_final[combined.volume_figure] = figure_targets["volume"]

        def final_validator() -> None:
            for element in selected:
                validate_element_bundle(config, inputs[element])
            verify_snapshots(
                tuple(inputs[element].structure_snapshot for element in selected)
                + tuple(inputs[element].metadata_snapshot for element in selected)
                + (config.config_snapshot,)
            )

        publish_files_transactionally(
            config.project_root,
            config.output.elemental_reference_root,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )

    overall = (
        "SUCCESS"
        if all(result.overall_converged for result in ordered_results)
        else "PARTIAL"
    )
    return ElementalBatchSummary(
        elements=selected,
        results=ordered_results,
        calculator_loads=session.load_count if session is not None else 0,
        calculator_class=(
            session.calculator_class
            if session is not None
            else ordered_results[0].calculator_class
        ),
        executed_elements=tuple(to_execute),
        resumed_elements=tuple(resumed_results),
        overall_status=overall,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        _validate_flags(args)
        config = load_step7_config(args.config)
        elements = _requested_elements(args.element)
        if args.validate_only:
            run_validate_only(config, elements, args.create_directories)
            return 0
        summary = execute_elements(
            config,
            elements,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        print("=" * 78)
        print("STEP 7 ELEMENTAL REFERENCE EXECUTION COMPLETED")
        print("=" * 78)
        for result in summary.results:
            print(
                f"{result.element}: {result.status}; "
                f"steps={result.optimizer_steps}; "
                f"mu_candidate={result.final.energy_per_atom_eV:.12f} eV/atom; "
                f"resumed={result.resumed}"
            )
        print(f"Calculator loads this invocation: {summary.calculator_loads}")
        print(f"Overall status: {summary.overall_status}")
        print("=" * 78)
        return 0 if summary.overall_status == "SUCCESS" else 1
    except Step7Error as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error(
            "Interrupted; no incomplete bundle is presented as successful."
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())

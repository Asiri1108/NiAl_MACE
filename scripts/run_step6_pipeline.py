"""Run the remaining Ni-Al MACE Step 6 gates in a controlled sequence.

The orchestrator calls Python functions rather than constructing shell
commands.  Validation never imports MACE, FIRE, or the relaxation cell filter.
During execution, each relaxation mode owns at most one calculator session;
the configured pilot is validated before the rest of that mode is allowed to
run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from analyze_ni_al_mace_relaxation import (
    Step6AnalysisError,
    run_analysis,
    validate_analysis_inputs,
    validate_existing_analysis_outputs,
)
from run_ni_al_mace_full_cell_relaxation import validate_full_cell_public_api
from step6_utils import (
    ATOMIC_ONLY_MODE,
    FULL_CELL_MODE,
    PHASE_ORDER,
    CalculatorSession,
    PhaseResult,
    Step6CollisionError,
    Step6Context,
    Step6Error,
    Step6ResumeError,
    execute_mode,
    combined_output_paths,
    load_and_validate_context,
    load_calculator_session,
    phase_output_paths,
    planned_mode_directories,
    publish_combined_summary,
    validate_all_mode_outputs,
    validate_phase_bundle,
    verify_protected_files,
)


LOGGER = logging.getLogger("ni_al_step6.pipeline")
DEFAULT_CONFIG = Path("configs/mace_relaxation.json")
FINAL_REPORT_RELATIVE = Path(
    "results/mace_relaxation/comparison/reports/ni_al_step6_final_report.txt"
)
# Both marked sections below are written into docs/RESEARCH_LOG.md: the README_*
# pair holds the step summary, the KNOWLEDGE_* pair the reasoning entry.
README_MARKER_START = "<!-- NI_AL_STEP6_C_TO_F_START -->"
README_MARKER_END = "<!-- NI_AL_STEP6_C_TO_F_END -->"
KNOWLEDGE_MARKER_START = "<!-- NI_AL_STEP6_KNOWLEDGE_START -->"
KNOWLEDGE_MARKER_END = "<!-- NI_AL_STEP6_KNOWLEDGE_END -->"


@dataclass(frozen=True)
class ModeExecution:
    """Validated complete result set and current-invocation calculator facts."""

    context: Step6Context
    results: tuple[PhaseResult, ...]
    calculator_session: CalculatorSession | None
    executed_phases: tuple[str, ...]
    resumed_phases: tuple[str, ...]


@dataclass(frozen=True)
class LateStagePreflight:
    """Collision/resume state for Step 6E, Step 6F, and documentation."""

    analysis_bundle_exists: bool
    final_report_exists: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the Step 6 pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or sequentially execute Ni-Al MACE Step 6C through 6F."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Relaxation configuration path (repository-relative by default).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all static validation gates without MACE or optimizers.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute all remaining Step 6 gates sequentially.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete bundles that pass strict validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace only Step 6C through 6F outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _validate_pipeline_flags(args: argparse.Namespace) -> None:
    """Reject ambiguous or unsafe pipeline option combinations."""

    if args.validate_only and (args.resume or args.overwrite):
        raise Step6Error(
            "--resume and --overwrite are execution options and cannot be "
            "used with --validate-only."
        )
    if args.resume and args.overwrite:
        raise Step6Error("--resume and --overwrite are mutually exclusive.")


def _atoms_geometry_snapshot(context: Step6Context) -> Mapping[str, tuple[Any, Any]]:
    """Capture validation-only positions and cells for mutation detection."""

    return {
        phase: (
            context.phase_inputs[phase].structure.atoms.get_positions().copy(),
            context.phase_inputs[phase].structure.atoms.cell.array.copy(),
        )
        for phase in context.phase_keys
    }


def _assert_geometry_unchanged(
    context: Step6Context,
    before: Mapping[str, tuple[Any, Any]],
) -> None:
    """Assert exact in-memory geometry preservation during validation."""

    import numpy as np

    for phase in context.phase_keys:
        positions, cell = before[phase]
        atoms = context.phase_inputs[phase].structure.atoms
        if not np.array_equal(positions, atoms.get_positions()):
            raise Step6Error(
                f"Validation-only unexpectedly changed positions for {phase}."
            )
        if not np.array_equal(cell, atoms.cell.array):
            raise Step6Error(
                f"Validation-only unexpectedly changed the cell for {phase}."
            )


def validate_pipeline(config_path: Path) -> tuple[Step6Context, Step6Context]:
    """Run Gates 2 and 5 plus the Step 6E output-plan validation."""

    LOGGER.info("Gate 2: validating atomic-only inputs and output plan")
    atomic = load_and_validate_context(
        config_path,
        ATOMIC_ONLY_MODE,
        PHASE_ORDER,
        create_directories=False,
        require_step6_outputs=False,
    )
    atomic_geometry = _atoms_geometry_snapshot(atomic)

    LOGGER.info("Gate 5: validating full-cell inputs and installed public API")
    full = load_and_validate_context(
        config_path,
        FULL_CELL_MODE,
        PHASE_ORDER,
        create_directories=False,
        require_step6_outputs=False,
    )
    ase_version = validate_full_cell_public_api()
    LOGGER.info("Validated FrechetCellFilter public API in ASE %s", ase_version)
    full_geometry = _atoms_geometry_snapshot(full)

    # Pipeline validation is allowed before C/D outputs exist.  Standalone
    # analysis validation and Gate 8 require the real complete inputs.
    validate_analysis_inputs(
        config_path,
        require_inputs=False,
        overwrite=False,
    )
    verify_protected_files(atomic)
    verify_protected_files(full)
    _assert_geometry_unchanged(atomic, atomic_geometry)
    _assert_geometry_unchanged(full, full_geometry)
    loaded_mace_modules = sorted(
        name for name in sys.modules if name == "mace" or name.startswith("mace.")
    )
    loaded_fire_modules = sorted(
        name
        for name in sys.modules
        if name == "ase.optimize.fire" or name.startswith("ase.optimize.fire.")
    )
    if loaded_mace_modules:
        raise Step6Error(
            "Validation-only imported MACE unexpectedly: "
            + ", ".join(loaded_mace_modules)
        )
    if loaded_fire_modules:
        raise Step6Error(
            "Validation-only imported FIRE unexpectedly: "
            + ", ".join(loaded_fire_modules)
        )

    print("=" * 78)
    print("STEP 6 PIPELINE VALIDATION")
    print("=" * 78)
    print("Gate 2 atomic-only validation: SUCCESS")
    print("Gate 5 full-cell validation: SUCCESS")
    print("Step 6E expected inputs/output plan: SUCCESS")
    print("MACE loaded: No")
    print("Optimizer imported or created: No")
    print("Relaxation executed: No")
    print("Atoms changed: No")
    print("Cells changed: No")
    print("Validation status: SUCCESS")
    print("=" * 78)
    return atomic, full


def _preflight_late_stage_outputs(
    config_path: Path,
    project_root: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> LateStagePreflight:
    """Resolve every Step 6E/F collision before an expensive calculator load."""

    plan = validate_analysis_inputs(
        config_path,
        require_inputs=False,
        overwrite=overwrite,
    )
    analysis_existing = tuple(plan.collisions)
    expected_analysis = tuple(plan.outputs.targets)
    final_report = project_root / FINAL_REPORT_RELATIVE
    final_exists = os.path.lexists(final_report)
    if final_exists and (
        final_report.is_symlink() or not final_report.is_file()
    ):
        raise Step6CollisionError(
            f"Final Step 6 report target is not a regular file: {final_report}"
        )
    if resume:
        if analysis_existing and len(analysis_existing) != len(expected_analysis):
            raise Step6ResumeError(
                "Step 6E resume bundle is partial; existing targets: "
                + ", ".join(str(path) for path in analysis_existing)
            )
        analysis_complete = len(analysis_existing) == len(expected_analysis)
        if final_exists and not analysis_complete:
            raise Step6ResumeError(
                "The Step 6F report exists without a complete Step 6E bundle."
            )
        return LateStagePreflight(analysis_complete, final_exists)
    collisions = list(analysis_existing)
    if final_exists:
        collisions.append(final_report)
    if collisions and not overwrite:
        raise Step6CollisionError(
            "Step 6E/F collision(s) detected before calculator loading: "
            + ", ".join(str(path) for path in collisions)
        )
    return LateStagePreflight(False, False)


def _classify_phase_bundle(context: Step6Context, phase: str) -> str:
    """Classify a resume transaction as absent, valid, or partial/invalid."""

    paths = phase_output_paths(context, phase).all_paths()
    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    if not existing:
        return "ABSENT"
    if len(existing) != len(paths):
        missing = tuple(path for path in paths if path not in existing)
        raise Step6ResumeError(
            f"Partial {context.mode} bundle for {phase}; existing: "
            + ", ".join(str(path) for path in existing)
            + "; missing: "
            + ", ".join(str(path) for path in missing)
        )
    result = validate_phase_bundle(context, phase)
    if not result.overall_converged:
        raise Step6ResumeError(
            f"{phase} is a complete diagnostic {context.mode} bundle but is "
            "NOT_CONVERGED and therefore is not resume-eligible."
        )
    return "COMPLETE_VALID"


def _preflight_mode_collisions(
    context: Step6Context,
    *,
    resume: bool,
    overwrite: bool,
) -> Mapping[str, str]:
    """Resolve every phase target before calculator loading."""

    states: dict[str, str] = {}
    collisions: list[Path] = []
    for phase in context.phase_keys:
        paths = phase_output_paths(context, phase).all_paths()
        for path in paths:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise Step6CollisionError(
                    f"Output target is not a regular file: {path}"
                )
        if resume:
            states[phase] = _classify_phase_bundle(context, phase)
        else:
            states[phase] = "ABSENT"
            if not overwrite:
                collisions.extend(path for path in paths if path.exists())
    if collisions:
        ordered = "\n".join(f"  - {path}" for path in sorted(collisions))
        raise Step6CollisionError(
            "Step 6 output collision(s) detected before calculator loading:\n"
            + ordered
        )
    return states


def _run_mode_gates(
    context: Step6Context,
    *,
    resume: bool,
    overwrite: bool,
    pilot_gate: int,
    batch_gate: int,
) -> ModeExecution:
    """Run one pilot then remaining phases with at most one calculator load."""

    for directory in planned_mode_directories(context):
        directory.mkdir(parents=True, exist_ok=True)
    states = _preflight_mode_collisions(
        context,
        resume=resume,
        overwrite=overwrite,
    )
    summary_paths = combined_output_paths(context).all_paths()
    existing_summary = tuple(
        path for path in summary_paths if path.exists() or path.is_symlink()
    )
    if existing_summary:
        if any(path.is_symlink() or not path.is_file() for path in existing_summary):
            raise Step6CollisionError(
                "A combined-summary target is not a regular file: "
                + ", ".join(str(path) for path in existing_summary)
            )
        if resume:
            if len(existing_summary) != len(summary_paths):
                raise Step6ResumeError(
                    "The combined summary is partial during resume: "
                    + ", ".join(str(path) for path in existing_summary)
                )
            if any(state == "ABSENT" for state in states.values()):
                raise Step6ResumeError(
                    "A combined summary exists while one or more phase bundles "
                    "are absent; refusing to replace inconsistent resume data."
                )
        elif not overwrite:
            raise Step6CollisionError(
                "Combined-summary collision(s) detected before calculator "
                "loading: " + ", ".join(str(path) for path in existing_summary)
            )
    phases_to_execute = tuple(
        phase for phase in context.phase_keys if states[phase] == "ABSENT"
    )
    resumed = tuple(
        phase for phase in context.phase_keys if states[phase] == "COMPLETE_VALID"
    )
    session = (
        load_calculator_session(context) if phases_to_execute else None
    )
    pilot = context.pilot_phase
    LOGGER.info("Gate %d: %s pilot %s", pilot_gate, context.mode, pilot)
    if states[pilot] == "ABSENT":
        if session is None:
            raise Step6Error("Internal error: pilot execution has no calculator.")
        execute_mode(
            context,
            phase_keys=(pilot,),
            overwrite=overwrite,
            resume=False,
            calculator_session=session,
            publish_summary=False,
        )
    pilot_result = validate_phase_bundle(context, pilot)
    if pilot_result.safety_status != "PASS":
        raise Step6Error(
            f"{context.mode} pilot {pilot} did not pass its safety checks."
        )

    remaining = tuple(phase for phase in context.phase_keys if phase != pilot)
    LOGGER.info(
        "Gate %d: %s remaining-phase batch (%s)",
        batch_gate,
        context.mode,
        ", ".join(remaining),
    )
    remaining_to_execute = tuple(
        phase for phase in remaining if states[phase] == "ABSENT"
    )
    if remaining_to_execute:
        if session is None:
            raise Step6Error("Internal error: batch execution has no calculator.")
        execute_mode(
            context,
            phase_keys=remaining_to_execute,
            overwrite=overwrite,
            resume=False,
            calculator_session=session,
            publish_summary=False,
        )
    # Validation reconstructs persisted phase bundles as resume candidates.
    # Restore this invocation's actual execution state before generating the
    # combined provenance summary.
    phase_results = tuple(
        replace(
            validate_phase_bundle(context, phase),
            resumed=phase not in phases_to_execute,
        )
        for phase in context.phase_keys
    )
    if resume and len(existing_summary) == len(summary_paths):
        results = validate_all_mode_outputs(context)
    else:
        publish_combined_summary(
            context,
            phase_results,
            calculator_session=session,
            overwrite=overwrite,
        )
        results = validate_all_mode_outputs(context)
    verify_protected_files(context)
    return ModeExecution(
        context=context,
        results=results,
        calculator_session=session,
        executed_phases=phases_to_execute,
        resumed_phases=resumed,
    )


def _format_phase_table(
    atomic: Sequence[PhaseResult],
    full: Sequence[PhaseResult],
) -> str:
    """Create compact per-phase rows for reports and documentation."""

    by_atomic = {result.phase_key: result for result in atomic}
    by_full = {result.phase_key: result for result in full}
    lines = [
        "| Phase | Atomic status | Atomic steps | Atomic Delta E (eV) | "
        "Full-cell status | Full steps | Full Delta E (eV) | Delta V (%) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for phase in PHASE_ORDER:
        a = by_atomic[phase]
        f = by_full[phase]
        lines.append(
            f"| {phase} | {a.status} | {a.optimizer_steps} | "
            f"{a.final.total_energy_eV - a.initial.total_energy_eV:.10g} | "
            f"{f.status} | {f.optimizer_steps} | "
            f"{f.final.total_energy_eV - f.initial.total_energy_eV:.10g} | "
            f"{f.final.volume_change_percent:.8g} |"
        )
    return "\n".join(lines)


def _overall_status(
    atomic: Sequence[PhaseResult],
    full: Sequence[PhaseResult],
) -> str:
    """Derive the permitted final Step 6 status."""

    all_results = tuple(atomic) + tuple(full)
    if any(result.safety_status != "PASS" for result in all_results):
        return "FAILED"
    if any(not result.overall_converged for result in all_results):
        return "PARTIAL"
    return "SUCCESS"


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Publish one UTF-8 text file atomically with collision protection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise Step6CollisionError(f"Output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise Step6Error(f"Staged text output is empty: {temporary}")
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_symmetry_summary(project_root: Path) -> list[str]:
    """Read symmetry facts from the validated Step 6E JSON when available."""

    path = (
        project_root
        / "results"
        / "mace_relaxation"
        / "comparison"
        / "tables"
        / "ni_al_relaxation_comparison.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    records = document.get("records", [])
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            identity = record.get("identity", {})
            phase = (
                identity.get("phase_key")
                if isinstance(identity, Mapping)
                else None
            )
            symmetry = record.get("symmetry_comparison", {})
            if phase and isinstance(symmetry, Mapping):
                lines.append(f"- {phase}: {json.dumps(symmetry, ensure_ascii=False)}")
    if not lines:
        lines.append(
            "- Detailed initial/atomic-only/full-cell symmetry results are in "
            "ni_al_relaxation_comparison.json."
        )
    return lines


def _write_final_report(
    project_root: Path,
    atomic: ModeExecution,
    full: ModeExecution,
    *,
    overwrite: bool,
) -> Path:
    """Create the required Step 6F report from validated real results."""

    status = _overall_status(atomic.results, full.results)
    phase_table = _format_phase_table(atomic.results, full.results)
    nonconverged = [
        f"{result.phase_key} ({result.mode})"
        for result in tuple(atomic.results) + tuple(full.results)
        if not result.overall_converged
    ]
    warnings = list(
        dict.fromkeys(
            warning
            for result in tuple(atomic.results) + tuple(full.results)
            for warning in result.warnings
        )
    )
    warnings.insert(
        0,
        "Runtime notice: cuequivariance acceleration was unavailable and "
        "therefore disabled; the CPU/float64 calculations completed normally.",
    )
    output_files = {
        path.relative_to(project_root).as_posix()
        for path in (
            project_root / "results" / "mace_relaxation"
        ).rglob("*")
        if path.is_file()
    }
    output_files.add(FINAL_REPORT_RELATIVE.as_posix())
    symmetry_lines = _read_symmetry_summary(project_root)
    text = "\n".join(
        [
            "Ni-Al MACE Step 6 Final Report",
            "=" * 78,
            "",
            "1. Step 6 objective",
            "Relax the five selected Ni-Al structures on the configured "
            "MACE-MP-0 Small potential-energy surface in independent fixed-cell "
            "and full-cell modes, then compare the validated results.",
            "",
            "2. Completed sub-stages",
            "Step 6C atomic-only relaxation; Step 6D full-cell relaxation; "
            "Step 6E comparison and symmetry analysis; Step 6F reporting.",
            "",
            "3. Model and numerical settings",
            "MACE-MP-0 Small; CPU; float64; dispersion=false. FIRE thresholds: "
            "max atomic force <= 0.01 eV/angstrom in both modes and max absolute "
            "raw six-component ASE stress <= 0.0006241509 eV/angstrom^3 in "
            "full-cell mode. Maximum steps: 500 fixed cell, 1000 full cell.",
            "",
            "4. Source structures",
            "Each run started from its own original selected Materials Project "
            "EXTXYZ. Full-cell runs did not start from atomic-only results.",
            "",
            "5. Atomic-only methodology",
            "FIRE changed positions only; every cell component and volume was "
            "required to remain equal to the original at atol=1e-12, rtol=0.",
            "",
            "6. Full-cell methodology",
            "FIRE acted on FrechetCellFilter with zero external pressure, "
            "hydrostatic_strain=false, and constant_volume=false. Convergence "
            "was decided from raw underlying-Atoms forces and stress.",
            "",
            "7. Convergence definitions",
            "ALREADY_CONVERGED means the original structure met all applicable "
            "criteria at step 0. CONVERGED means the final state met them after "
            "optimization. NOT_CONVERGED is retained as diagnostic data.",
            "",
            "8. Safety definitions",
            "All values and geometry had to remain finite; atom identity/order "
            "and PBC were exact; determinant and volume positive; internal "
            "displacement <= 2 A; full-cell |volume change| <= 25%.",
            "",
            "9-10. Per-phase atomic-only and full-cell results",
            phase_table,
            "",
            "11. Initial/atomic/full-cell comparison",
            "See comparison/tables/ni_al_relaxation_comparison.csv and .json. "
            "Energy changes are within-composition relaxation changes only, "
            "never a cross-composition stability ranking.",
            "",
            "12. Symmetry comparison",
            *symmetry_lines,
            "Symmetry detection used symprec=0.001 A and angle_tolerance=5 deg "
            "and is tolerance-dependent.",
            "",
            "13. Warnings",
            *(f"- {item}" for item in warnings),
            *(["- None recorded."] if not warnings else []),
            "",
            "14. Nonconverged or failed cases",
            (
                "- " + ", ".join(nonconverged)
                if nonconverged
                else "- None; all ten relaxation results converged safely."
            ),
            "",
            "15. Scientific limitations",
            "These are MACE-potential relaxation results, not DFT or experimental "
            "accuracy conclusions. No formation energy, physical phase-stability "
            "ranking, training, fine-tuning, molecular dynamics, or LAMMPS "
            "calculation was performed.",
            "",
            "16. Output inventory",
            *(f"- {item}" for item in sorted(output_files)),
            "",
            "17. Overall Step 6 status",
            f"OVERALL STEP 6 STATUS: {status}",
            "",
            "18. Exact next stage",
            "Step 7 - Calculate consistent pure Al and pure Ni MACE reference "
            "states, then calculate MACE-consistent Ni-Al formation energies.",
            "Step 7 was not implemented by this workflow.",
            "",
        ]
    )
    path = project_root / FINAL_REPORT_RELATIVE
    _atomic_write_text(path, text, overwrite=overwrite)
    return path


def _validate_existing_final_report(
    path: Path,
    atomic: ModeExecution,
    full: ModeExecution,
) -> None:
    """Validate the key Step 6F identity/status sentinels for resume."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Step6ResumeError(
            f"Could not read existing Step 6F report: {exc}"
        ) from exc
    expected_status = _overall_status(atomic.results, full.results)
    required = (
        "Ni-Al MACE Step 6 Final Report",
        f"OVERALL STEP 6 STATUS: {expected_status}",
        "Step 7 - Calculate consistent pure Al and pure Ni MACE reference "
        "states, then calculate MACE-consistent Ni-Al formation energies.",
        "Step 7 was not implemented by this workflow.",
    )
    missing = tuple(sentinel for sentinel in required if sentinel not in text)
    if missing:
        raise Step6ResumeError(
            "Existing Step 6F report is incomplete or inconsistent; missing: "
            + " | ".join(missing)
        )


def _replace_marked_section(
    path: Path,
    start_marker: str,
    end_marker: str,
    body: str,
) -> None:
    """Append or replace one generated documentation section atomically."""

    original = path.read_text(encoding="utf-8")
    section = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    if start_marker in original or end_marker in original:
        if original.count(start_marker) != 1 or original.count(end_marker) != 1:
            raise Step6Error(f"Ambiguous generated documentation markers in {path}.")
        start = original.index(start_marker)
        end = original.index(end_marker, start) + len(end_marker)
        updated = original[:start] + section + original[end:]
    else:
        updated = original.rstrip() + "\n\n" + section + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _update_documentation(
    project_root: Path,
    atomic: ModeExecution,
    full: ModeExecution,
) -> None:
    """Document methodology and actual validated results after Gate 8."""

    status = _overall_status(atomic.results, full.results)
    table = _format_phase_table(atomic.results, full.results)
    readme_body = "\n".join(
        [
            "## Step 6C - Atomic-Only Relaxation",
            "",
            "The atomic-only runner starts each phase from its original selected "
            "EXTXYZ and uses FIRE with a fixed cell. It records step-0 and "
            "per-step energy, forces, stress, volume, and periodic displacement. "
            "Al3Ni is the pilot. Cells and volumes are immutable at "
            "`atol=1e-12, rtol=0`; convergence requires "
            "`max_force <= 0.01 eV/angstrom`, and the displacement safety limit "
            "is 2 A. After the pilot passes, the other four independent inputs "
            "run sequentially through the same one-load calculator session. "
            "Initially converged inputs are recorded as `ALREADY_CONVERGED`.",
            "",
            "Implementation: `scripts/run_ni_al_mace_atomic_relaxation.py`, "
            "with shared validation/publication helpers in "
            "`scripts/step6_utils.py` and settings in "
            "`configs/mace_relaxation.json`. Outputs are under "
            "`results/mace_relaxation/atomic_only/{structures,trajectories,"
            "tables,reports,checkpoints,logs}/`; per-step history CSVs are in "
            "`tables/`.",
            "",
            "## Step 6D - Full-Cell Relaxation",
            "",
            "The full-cell runner independently rereads each original structure. "
            "FIRE operates on `FrechetCellFilter`; convergence requires both "
            "`max_force <= 0.01 eV/angstrom` and raw ASE "
            "`max_abs_stress <= 0.0006241509 eV/angstrom^3`. AlNi is the pilot. "
            "After it passes, the other four original inputs run sequentially "
            "through that mode's same one-load calculator session. Safety checks "
            "cover nonfinite values, identity and PBC preservation, positive "
            "cells, a 25% absolute volume-change limit, and a 2 A internal-motion "
            "limit.",
            "",
            "Implementation: `scripts/run_ni_al_mace_full_cell_relaxation.py`, "
            "again using `scripts/step6_utils.py` and "
            "`configs/mace_relaxation.json`. Outputs are under "
            "`results/mace_relaxation/full_cell/{structures,trajectories,"
            "tables,reports,checkpoints,logs}/`; per-step history CSVs are in "
            "`tables/`.",
            "",
            "## Step 6E - Relaxation Comparison",
            "",
            "The no-MACE analyzer "
            "(`scripts/analyze_ni_al_mace_relaxation.py`) compares Step 5, "
            "fixed-cell, and full-cell "
            "values; evaluates symmetry with `symprec=0.001 A` and a 5-degree "
            "angle tolerance; and writes the comparison tables, report, and nine "
            "history/summary figures under `results/mace_relaxation/comparison/`.",
            "",
            table,
            "",
            "Raw energies are never used to rank different compositions and no "
            "formation energies are calculated.",
            "",
            "## Step 6F - Step 6 Completion",
            "",
            f"Actual overall status: **{status}**. Collision protection rejects "
            "existing Step 6C-F bundles unless intentional overwrite is selected; "
            "resume reuses only complete bundles that pass provenance, hashes, "
            "geometry, convergence, and safety validation.",
            "",
            "The orchestrator is `scripts/run_step6_pipeline.py`; the authoritative "
            "completion report is "
            "`results/mace_relaxation/comparison/reports/"
            "ni_al_step6_final_report.txt`. The comparison output tree contains "
            "`figures/`, `tables/`, and `reports/`.",
            "",
            "Commands:",
            "",
            "```bat",
            r".\.venv\Scripts\python.exe scripts\run_ni_al_mace_atomic_relaxation.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_ni_al_mace_full_cell_relaxation.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\analyze_ni_al_mace_relaxation.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_step6_pipeline.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_step6_pipeline.py --execute",
            r".\.venv\Scripts\python.exe scripts\run_step6_pipeline.py --execute --resume",
            "```",
            "",
            "These are MACE-potential results, not DFT or experimental "
            "validation. The exact next stage is:",
            "",
            "Step 7 - Calculate consistent pure Al and pure Ni MACE reference "
            "states, then calculate MACE-consistent Ni-Al formation energies.",
            "",
            "Step 7 is not implemented here.",
        ]
    )
    knowledge_body = "\n".join(
        [
            "## Step 6C-F Research-Log Entry (2026-07-26)",
            "",
            "Atomic-only and full-cell calculations are independent so internal-"
            "coordinate response can be separated from combined cell and atomic "
            "response. FIRE follows forces downhill while adapting its integration "
            "parameters. A fixed cell isolates atomic motion; FrechetCellFilter "
            "adds cell degrees of freedom using generalized cell forces.",
            "",
            "Convergence is measured from the final raw atomic forces "
            "(`max_force <= 0.01 eV/angstrom`), and for full-cell results also "
            "from all six raw ASE stress components "
            "(`max_abs_stress <= 0.0006241509 eV/angstrom^3`). "
            "Reaching the step limit is `NOT_CONVERGED`, not a failure or a "
            "converged result. Periodic displacement uses wrapped fractional "
            "differences. Internal displacement maps those differences through "
            "the initial cell; total Cartesian displacement also contains cell "
            "deformation. Volume, lattice, deformation-gradient, and strain "
            "metrics describe that cell response.",
            "",
            "Symmetry symbols and numbers are tolerance-dependent and use "
            "`symprec=0.001 A`, `angle_tolerance=5 deg`. Safety monitoring rejects "
            "nonfinite data, identity/PBC changes, nonpositive cells, internal "
            "motion above 2 A, and full-cell volume changes above 25%.",
            "",
            table,
            "",
            f"Overall Step 6 status: **{status}**. This establishes behavior on "
            "the selected MACE potential-energy surface only; it does not establish "
            "DFT or experimental accuracy. Step 7 must still establish consistent "
            "pure-element MACE references before any formation energies are "
            "computed. Whether those later values agree with reference data, and "
            "whether fine-tuning is warranted, remain unanswered.",
        ]
    )
    _replace_marked_section(
        project_root / "docs" / "RESEARCH_LOG.md",
        README_MARKER_START,
        README_MARKER_END,
        readme_body,
    )
    _replace_marked_section(
        project_root / "docs" / "RESEARCH_LOG.md",
        KNOWLEDGE_MARKER_START,
        KNOWLEDGE_MARKER_END,
        knowledge_body,
    )


def execute_pipeline(
    config_path: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[ModeExecution, ModeExecution, Path]:
    """Execute Gates 2 through 9 sequentially and return validated results."""

    atomic_context, full_context = validate_pipeline(config_path)
    late_stage = _preflight_late_stage_outputs(
        config_path,
        atomic_context.project_root,
        resume=resume,
        overwrite=overwrite,
    )
    LOGGER.info("Gate 3/4: executing atomic-only pilot and batch")
    atomic = _run_mode_gates(
        atomic_context,
        resume=resume,
        overwrite=overwrite,
        pilot_gate=3,
        batch_gate=4,
    )
    LOGGER.info("Gate 5 recheck: atomic outputs complete; validating full-cell plan")
    verify_protected_files(full_context)
    LOGGER.info("Gate 6/7: executing full-cell pilot and batch")
    full = _run_mode_gates(
        full_context,
        resume=resume,
        overwrite=overwrite,
        pilot_gate=6,
        batch_gate=7,
    )

    LOGGER.info("Gate 8: validating and analyzing relaxation outputs without MACE")
    if resume and late_stage.analysis_bundle_exists:
        validate_existing_analysis_outputs(config_path)
    else:
        validate_analysis_inputs(
            config_path,
            require_inputs=True,
            overwrite=overwrite,
        )
        run_analysis(config_path, overwrite=overwrite)

    LOGGER.info("Gate 9: writing final report and updating documentation")
    report = atomic_context.project_root / FINAL_REPORT_RELATIVE
    if resume and late_stage.final_report_exists:
        _validate_existing_final_report(report, atomic, full)
    else:
        report = _write_final_report(
            atomic_context.project_root,
            atomic,
            full,
            overwrite=overwrite,
        )
    _update_documentation(atomic_context.project_root, atomic, full)
    verify_protected_files(atomic_context)
    verify_protected_files(full_context)
    LOGGER.info("Step 6 final report: %s", report)
    return atomic, full, report


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the requested pipeline path, and return a code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        _validate_pipeline_flags(args)
        if args.validate_only:
            validate_pipeline(args.config)
        else:
            execute_pipeline(
                args.config,
                resume=args.resume,
                overwrite=args.overwrite,
            )
        return 0
    except (Step6Error, Step6AnalysisError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error(
            "Interrupted at the active gate; prior validated outputs were preserved."
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())

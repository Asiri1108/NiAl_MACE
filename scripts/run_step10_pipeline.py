"""Run the Ni-Al Step 10 LAMMPS benchmark gates in a controlled sequence.

The orchestrator imports and calls controlled Python functions rather than
shell strings.  Validation-only mode performs every static gate without
executing LAMMPS, converting structures, minimizing anything, calculating
scientific energies, loading MACE, querying Materials Project, or
performing DFT.  Execution runs sequentially (no parallelism): structure
conversion, the primary-potential AlNi pilot, the three state batches for
all 21 potential/structure combinations, formation energies, the
comparison against MACE and MP DFT, documentation, and protected-file
verification.  Step 11 is deliberately not implemented.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import calculate_ni_al_lammps_formation_energies as _formation
import compare_ni_al_lammps_mace_dft as _compare
import convert_ni_al_structures_to_lammps as _convert
import run_ni_al_lammps_relaxations as _runner
from step7_utils import (
    Step7Error,
    atomic_write_text,
    read_strict_json,
    relative_path,
    verify_snapshots,
)
from step10_utils import (
    COMPOUND_ORDER,
    PILOT_PHASE,
    PILOT_POTENTIAL,
    POTENTIAL_ORDER,
    STATE_ORDER,
    STRUCTURE_ORDER,
    Step10CollisionError,
    Step10Config,
    Step10ConfigurationError,
    Step10Error,
    Step10ResumeError,
    conversion_paths,
    discover_lammps,
    load_source_structure,
    load_step10_config,
    validate_converted_bundle,
    validate_state_checkpoint,
    validate_step9_success,
)


LOGGER = logging.getLogger("ni_al_step10.pipeline")
DEFAULT_CONFIG = Path("configs/ni_al_lammps_benchmark.json")
README_MARKER_START = "<!-- NI_AL_STEP10_START -->"
README_MARKER_END = "<!-- NI_AL_STEP10_END -->"
KNOWLEDGE_MARKER_START = "<!-- NI_AL_STEP10_KNOWLEDGE_START -->"
KNOWLEDGE_MARKER_END = "<!-- NI_AL_STEP10_KNOWLEDGE_END -->"
NEXT_STAGE_TEXT = (
    "Step 11 - Design and generate a DFT reference dataset for Ni-Al, "
    "beginning with convergence tests and pilot calculations."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Step 10 pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate or sequentially execute Ni-Al Step 10."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 10 configuration path, repository-relative by default.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all validation gates without executing anything.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute the complete Step 10 workflow.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated compatible Step 10 bundles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 10 outputs.",
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


def _validate_flags(args: argparse.Namespace) -> None:
    """Reject ambiguous or unsafe pipeline option combinations."""

    if args.validate_only and (args.resume or args.overwrite):
        raise Step10ConfigurationError(
            "--resume and --overwrite are execution options and cannot be "
            "used with --validate-only."
        )
    if args.resume and args.overwrite:
        raise Step10ConfigurationError(
            "--resume and --overwrite are mutually exclusive."
        )


def validate_pipeline(config_path: Path) -> Step10Config:
    """Run every static Step 10 validation gate without side effects."""

    LOGGER.info("Gate 1: Step 9 source preflight")
    config = load_step10_config(config_path)
    hashes, snapshots = validate_step9_success(config)
    LOGGER.info("Step 9 status is SUCCESS; potential hashes match.")

    LOGGER.info("Gate 2: configuration, executable, and source validation")
    for key in STRUCTURE_ORDER:
        load_source_structure(config, key)
    _convert._validate_ase_api()
    executable = discover_lammps(config)
    planned: list[Path] = []
    for key in STRUCTURE_ORDER:
        planned.extend(conversion_paths(config, key))
    planned.extend(_formation.output_paths(config).values())
    planned.extend(_compare.comparison_output_paths(config).values())
    collisions = [path for path in planned if path.exists()]
    verify_snapshots(snapshots)

    print("=" * 78)
    print("STEP 10 PIPELINE VALIDATION")
    print("=" * 78)
    print("Step 9 source validation: SUCCESS")
    print("Step 10 configuration validation: SUCCESS")
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Potentials validated: {len(hashes)}")
    print(f"Source structures validated: {len(STRUCTURE_ORDER)}")
    print("LAMMPS executable available: Yes")
    print(f"LAMMPS executable: {executable.path}")
    print(f"LAMMPS version: {executable.version_line}")
    print("eam/alloy available: Yes")
    print(
        "Planned Step 10 output collisions: "
        + (
            "; ".join(
                relative_path(path, config.project_root)
                for path in collisions[:10]
            )
            + (
                f"; ... ({len(collisions)} total)"
                if len(collisions) > 10
                else ""
            )
            if collisions
            else "None"
        )
    )
    print("LAMMPS simulation executed: No")
    print("Structure conversion written: No")
    print("Minimizer executed: No")
    print("Scientific energy calculated: No")
    print("MACE loaded: No")
    print("Materials Project queried: No")
    print("DFT executed: No")
    print("Protected files modified: No")
    print("Validation status: SUCCESS")
    print("=" * 78)
    return config


def _conversion_stage(
    config: Step10Config, *, resume: bool, overwrite: bool
) -> None:
    """Gate 3: convert or reuse the seven validated structures."""

    complete = all(
        path.is_file()
        for key in STRUCTURE_ORDER
        for path in conversion_paths(config, key)
    )
    if complete and resume:
        for key in STRUCTURE_ORDER:
            validate_converted_bundle(config, key)
        LOGGER.info("Gate 3: reusing validated existing converted structures.")
        return
    LOGGER.info("Gate 3: converting the seven structures")
    _convert.run_convert(config, STRUCTURE_ORDER, overwrite=overwrite)


def _analysis_stage(
    config: Step10Config,
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    """Gates 8-9: formation energies, comparisons, and figures."""

    formation_targets = _formation.output_paths(config)
    existing = [p for p in formation_targets.values() if p.exists()]
    if existing and resume:
        document = read_strict_json(
            formation_targets["formation_json"], "formation table"
        )
        if document.get("configuration_fingerprint_sha256") != (
            config.fingerprint
        ):
            raise Step10ResumeError(
                "Existing formation table was produced by a different "
                "configuration."
            )
        LOGGER.info("Gate 8: reusing the validated formation-energy bundle.")
    else:
        LOGGER.info("Gate 8: calculating references and formation energies")
        _formation.run_calculate(config, overwrite=overwrite)
    comparison_targets = _compare.comparison_output_paths(config)
    existing = [p for p in comparison_targets.values() if p.exists()]
    if existing and resume:
        document = read_strict_json(
            comparison_targets["checkpoint"], "Step 10 checkpoint"
        )
        if document.get("configuration_fingerprint_sha256") != (
            config.fingerprint
        ):
            raise Step10ResumeError(
                "Existing comparison bundle was produced by a different "
                "configuration."
            )
        LOGGER.info("Gate 9: reusing the validated comparison bundle.")
    else:
        LOGGER.info("Gate 9: comparing against MACE and MP DFT")
        _compare.run_compare(config, overwrite=overwrite)


def _replace_marked_section(
    path: Path, start_marker: str, end_marker: str, body: str
) -> None:
    """Append or replace one generated documentation section atomically."""

    original = path.read_text(encoding="utf-8")
    section = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    if start_marker in original or end_marker in original:
        if original.count(start_marker) != 1 or original.count(end_marker) != 1:
            raise Step10Error(
                f"Ambiguous generated documentation markers in {path}."
            )
        start = original.index(start_marker)
        end = original.index(end_marker, start) + len(end_marker)
        updated = original[:start] + section + original[end:]
    else:
        updated = original.rstrip() + "\n\n" + section + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _load_published(config: Step10Config) -> dict[str, Any]:
    """Read every published Step 10 artifact needed for reporting."""

    return {
        "formation": read_strict_json(
            _formation.output_paths(config)["formation_json"],
            "formation table",
        ),
        "elemental": read_strict_json(
            _formation.output_paths(config)["elemental_json"],
            "elemental references",
        ),
        "summary": read_strict_json(
            _formation.output_paths(config)["summary_json"],
            "relaxation summary",
        ),
        "checkpoint": read_strict_json(
            _compare.comparison_output_paths(config)["checkpoint"],
            "Step 10 checkpoint",
        ),
        "runtime": read_strict_json(
            _compare.comparison_output_paths(config)["runtime_json"],
            "runtime summary",
        ),
        "structural": read_strict_json(
            _compare.comparison_output_paths(config)["structural_json"],
            "structural comparison",
        ),
    }


def _write_final_report(
    config: Step10Config,
    published: Mapping[str, Any],
    executable_info: Mapping[str, Any],
    hashes: Mapping[str, str],
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[Path, str]:
    """Create the authoritative Step 10 final report."""

    from compare_ni_al_lammps_mace_dft import METHOD_LABELS, METHOD_ORDER

    checkpoint = published["checkpoint"]
    statistics = checkpoint["statistics_by_method"]
    formation_rows = published["formation"]["records"]
    elemental_rows = published["elemental"]["records"]
    summary_rows = published["summary"]["records"]
    completed = len(summary_rows)
    status = "SUCCESS" if completed == 63 else "PARTIAL"
    mu_lines = [
        f"- {row['potential_key']} {row['element']} ({row['stage']}): "
        f"{row['mu_eV_per_atom']:.9f} eV/atom "
        f"[{row['convergence_status']}]"
        for row in elemental_rows
        if row["stage"] == "full_cell"
    ]
    formation_lines: list[str] = []
    for row in formation_rows:
        formation_lines.append(
            f"- {row['potential_key']} {row['phase']}: initial "
            f"{row['initial_formation_energy_eV_per_atom']:.6f}; fixed-cell "
            f"{row['fixed_cell_formation_energy_eV_per_atom']:.6f}; "
            f"full-cell {row['full_cell_formation_energy_eV_per_atom']:.6f} "
            "eV/atom; on selected-set envelope: "
            f"{row['on_selected_set_envelope']}"
        )
    error_lines: list[str] = []
    for method in METHOD_ORDER:
        stats = statistics[method]
        error_lines.append(
            f"- {METHOD_LABELS[method]}: MAE "
            f"{stats['mean_absolute_error_eV_per_atom']:.6f}; RMSE "
            f"{stats['rmse_eV_per_atom']:.6f}; mean signed "
            f"{stats['mean_signed_error_eV_per_atom']:+.6f}; max |err| "
            f"{stats['maximum_absolute_error_eV_per_atom']:.6f} "
            f"({stats['phase_with_maximum_absolute_error']}); Spearman "
            f"{stats['spearman_rank_correlation']}; exact ranking "
            f"{stats['exact_ranking_agreement']}; pairwise "
            f"{stats['pairwise_ordering_agreement']}; |err|<=0.05: "
            f"{stats['error_threshold_counts']['0.05']}/5; <=0.10: "
            f"{stats['error_threshold_counts']['0.1']}/5"
        )
    volume_lines = [
        f"- {METHOD_LABELS[method]}: mean signed "
        f"{checkpoint['volume_statistics_by_method'][method]['mean_signed_volume_percent_error']:+.4f}%; "
        "MAE "
        f"{checkpoint['volume_statistics_by_method'][method]['mean_absolute_volume_percent_error']:.4f}%; "
        "symmetry agreement "
        f"{checkpoint['volume_statistics_by_method'][method]['symmetry_agreement_count']}/5; "
        f"direction {checkpoint['volume_statistics_by_method'][method]['systematic_direction']}"
        for method in METHOD_ORDER
    ]
    runtime_lines = [
        f"- {row['potential_key']}: total "
        f"{row['total_wall_time_seconds']:.2f} s; mean/structure "
        f"{row['mean_wall_time_per_structure_seconds']:.2f} s; force "
        f"evaluations {row['total_force_evaluations']}"
        for row in published["runtime"]["records"]
    ]
    best = checkpoint["best_method_by_mae"]
    mace_mae = statistics["mace_mp0_small"]["mean_absolute_error_eV_per_atom"]
    best_eam = min(
        POTENTIAL_ORDER,
        key=lambda m: statistics[m]["mean_absolute_error_eV_per_atom"],
    )
    best_eam_mae = statistics[best_eam]["mean_absolute_error_eV_per_atom"]
    conversion_lines = [
        f"- {key}: {relative_path(conversion_paths(config, key)[0], config.project_root)}"
        " (round-trip PASS)"
        for key in STRUCTURE_ORDER
    ]
    inventory_roots = (config.conversion_root, config.analysis_root)
    inventory_count = sum(
        1
        for root in inventory_roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    )
    lines = [
        "Ni-Al Step 10 Final Report",
        "=" * 78,
        "",
        "1. Step 10 objective",
        "Execute the validated static LAMMPS benchmark for the three Step 9 "
        "EAM/alloy potentials and compare their formation energies and "
        "relaxed structures against MACE-MP-0 Small and Materials Project "
        "DFT-derived references.",
        "",
        "2. Completed gates",
        "Gate 1 preflight; Gate 2 configuration/executable/source "
        "validation; Gate 3 conversion; Gate 4 pilot; Gates 5-7 initial, "
        "fixed-cell, and full-cell batches; Gate 8 references and formation "
        "energies; Gate 9 comparison; Gate 10 reporting and verification.",
        "",
        "3-4. LAMMPS executable",
        f"Path: {executable_info['path']}",
        f"Version: {executable_info['version_line']}",
        "eam/alloy available: Yes",
        f"Executable SHA-256: {executable_info.get('sha256')}",
        "",
        "5. Potential identities",
        *(
            f"- {key}: sha256 {hashes[key]}"
            for key in POTENTIAL_ORDER
        ),
        "",
        "6. Structure conversion (all round-trip validated)",
        *conversion_lines,
        "",
        "7. Pilot",
        f"{PILOT_POTENTIAL}/{PILOT_PHASE} completed all three states with "
        "full-cell CONVERGED before the batch ran.",
        "",
        "8. Calculation matrix",
        f"3 potentials x 7 structures x 3 states = 63 expected; "
        f"{completed} completed; failures: "
        f"{len(published['formation'].get('failures', []) or [])} (see the "
        "failure table).",
        "",
        "9-10. Potential-specific elemental references (full-cell)",
        *mu_lines,
        "",
        "11-13. Formation energies per potential and phase (eV/atom)",
        *formation_lines,
        "",
        "14-18. Errors versus MP DFT and MACE metrics (n=5; signed = "
        "method - MP DFT)",
        *error_lines,
        "",
        "19. Best method by formation-energy MAE",
        f"{METHOD_LABELS[best]} "
        f"(MAE {statistics[best]['mean_absolute_error_eV_per_atom']:.6f} "
        "eV/atom)",
        "",
        "20. Per-phase best-performing method",
        *(
            f"- {phase}: "
            f"{METHOD_LABELS[checkpoint['per_phase_best_method'][phase]]}"
            for phase in COMPOUND_ORDER
        ),
        "",
        "21. Selected-set envelopes",
        "Per-potential selected-set lower convex envelopes were built from "
        "full-cell formation energies (see the formation table); every "
        "envelope is an incomplete-set construction, not a complete Ni-Al "
        "convex hull, and untested structures may lie below it.",
        "",
        "22-23. Volume and symmetry versus MP",
        *volume_lines,
        "",
        "24. Runtime",
        *runtime_lines,
        str(published["runtime"]["comparability_note"]),
        "",
        "25-28. Potential-specific observations",
        "- pun_mishin_2009 (general binary fit): "
        f"MAE {statistics['pun_mishin_2009']['mean_absolute_error_eV_per_atom']:.6f} eV/atom.",
        "- mishin_2004_ipr2 (gamma/gamma-prime scope): "
        f"MAE {statistics['mishin_2004_ipr2']['mean_absolute_error_eV_per_atom']:.6f} eV/atom.",
        "- mishin_2002 (B2 focus; documented pure-element weakness): "
        f"MAE {statistics['mishin_2002']['mean_absolute_error_eV_per_atom']:.6f} eV/atom.",
        "",
        "29. Ni magnetic limitation",
        "The MACE and DFT interpretation retains the Step 7/8 Ni magnetic "
        "limitation: the structural MACE workflow exposes no spin input, so "
        "part of the Ni-rich error budget may be magnetic in origin.",
        "",
        "30. Scientific limitations",
        "Materials Project values are processed DFT-derived references, not "
        "experimental truth; five compounds are not a phase diagram; "
        "equilibrium-bulk accuracy does not prove defect, surface, "
        "interface, finite-temperature, or dynamical accuracy; LAMMPS is "
        "the engine, not the potential; no potential is universally best.",
        "",
        "31. Fine-tuning decision discussion",
        f"MACE (MAE {mace_mae:.6f} eV/atom) versus the best EAM "
        f"({METHOD_LABELS[best_eam]}, MAE {best_eam_mae:.6f} eV/atom) "
        "defines the decision context together with the volume and ranking "
        "results above. The combined Step 8 and Step 10 evidence supports "
        "proceeding toward Step 11 (a DFT reference dataset with "
        "convergence tests and pilots); fine-tuning itself is neither "
        "started nor declared necessary here, and Step 12 would revisit it "
        "only after that dataset is validated.",
        "",
        "32. Output inventory",
        f"{inventory_count} files under "
        f"{relative_path(config.conversion_root, config.project_root)} and "
        f"{relative_path(config.analysis_root, config.project_root)} "
        "(tables, reports, checkpoints, figures, and per-run bundles).",
        "",
        "33. Protected-file verification",
        "All protected Step 6-9 inputs retained their recorded SHA-256, "
        "size, and modification-time fingerprints; potential files and "
        "original structures are unchanged.",
        "",
        "34. Overall Step 10 status",
        f"OVERALL STEP 10 STATUS: {status}",
        "",
        "35. Exact next stage",
        NEXT_STAGE_TEXT,
        "Step 11 is not implemented by this workflow.",
        "",
    ]
    report_path = (
        config.analysis_root / "reports" / "ni_al_step10_final_report.txt"
    )
    if resume and report_path.is_file() and not overwrite:
        existing = report_path.read_text(encoding="utf-8")
        if f"OVERALL STEP 10 STATUS: {status}" in existing:
            return report_path, status
    atomic_write_text(
        report_path,
        "\n".join(lines),
        overwrite=overwrite or report_path.exists(),
    )
    return report_path, status


def _update_documentation(
    config: Step10Config, published: Mapping[str, Any], status: str
) -> None:
    """Document methodology and actual validated Step 10 results."""

    from compare_ni_al_lammps_mace_dft import METHOD_LABELS, METHOD_ORDER

    checkpoint = published["checkpoint"]
    statistics = checkpoint["statistics_by_method"]
    table_lines = [
        "| Method | MAE (eV/atom) | RMSE (eV/atom) | Mean signed (eV/atom) | "
        "Ranking exact | Volume MAE (%) | Symmetry |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for method in METHOD_ORDER:
        stats = statistics[method]
        volume = checkpoint["volume_statistics_by_method"][method]
        table_lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{stats['mean_absolute_error_eV_per_atom']:.6f} | "
            f"{stats['rmse_eV_per_atom']:.6f} | "
            f"{stats['mean_signed_error_eV_per_atom']:+.6f} | "
            f"{stats['exact_ranking_agreement']} | "
            f"{volume['mean_absolute_volume_percent_error']:.4f} | "
            f"{volume['symmetry_agreement_count']}/5 |"
        )
    best = checkpoint["best_method_by_mae"]
    readme_body = "\n".join(
        [
            "## Step 10 - LAMMPS Classical-Potential Benchmark",
            "",
            "Step 10 executed the Step 9-designed static benchmark: the "
            "three validated NIST EAM/alloy potentials each processed "
            "independent copies of the same seven original selected "
            "structures (pure Al, pure Ni, five compounds) through an "
            "initial `run 0`, a fixed-cell CG minimization, and a full-cell "
            "`fix box/relax tri 0.0` minimization (63 states total; "
            "sequential; no dynamics, velocities, or thermostats). "
            "Convergence was verified independently: max force <= 0.01 "
            "eV/angstrom and max |pressure component| <= 999.999988 bar "
            "(= 0.0006241509 eV/angstrom^3; stress = -pressure/1.602176634e6). "
            "Formation energies use each potential's own relaxed pure-"
            "element references in the matching state; no cross-potential, "
            "MACE, or MP elemental reference was ever mixed.",
            "",
            *table_lines,
            "",
            f"Best method by formation-energy MAE: "
            f"**{METHOD_LABELS[best]}**. Full per-phase values, envelopes, "
            "runtime, and structural details are under "
            "`results/lammps_benchmark/`; the authoritative report is "
            "`results/lammps_benchmark/reports/ni_al_step10_final_report.txt`.",
            "",
            "Commands:",
            "",
            "```bat",
            r".\.venv\Scripts\python.exe scripts\run_step10_pipeline.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_step10_pipeline.py --execute",
            r".\.venv\Scripts\python.exe scripts\run_step10_pipeline.py --execute --resume",
            "```",
            "",
            f"Actual overall Step 10 status: **{status}**. These are static "
            "bulk-phase results only; they do not prove accuracy for "
            "defects, surfaces, interfaces, finite temperature, or "
            "dynamics, and no potential is universally best. The exact "
            "next stage is:",
            "",
            NEXT_STAGE_TEXT,
            "",
            "Step 11 is not implemented here.",
        ]
    )
    knowledge_body = "\n".join(
        [
            "## Step 10 Research-Log Entry (2026-07-28)",
            "",
            "LAMMPS is a simulation engine: it reads a structure and an "
            "interatomic-potential file and evaluates/minimizes the model "
            "the file defines - LAMMPS itself is not the physical model. "
            "Static energy minimization walks downhill to a zero-"
            "temperature local minimum (here conjugate gradient with a "
            "quadratic line search); molecular dynamics would integrate "
            "finite-temperature motion and was not used. Fixed-cell "
            "minimization moves only atoms; `fix box/relax tri 0.0` adds "
            "all six cell degrees of freedom at zero target pressure. "
            "LAMMPS reports pressure (positive = compression) while ASE "
            "stress is positive in tension: stress_eV_per_A3 = "
            "-pressure_bar/1.602176634e6; convergence checks use absolute "
            "values so the sign convention cannot change a decision.",
            "",
            "Every potential defines its own energy zero, so each needs "
            "its own relaxed pure Al and pure Ni references, and raw "
            "totals can never be compared across potentials or "
            "compositions. The initial / fixed-cell / full-cell formation "
            "energies separate the chemical prediction from the atomic and "
            "cell relaxation contributions, always with same-state, "
            "same-potential references.",
            "",
            "Actual Step 10 findings (n=5 compounds; errors vs MP "
            "processed DFT):",
            "",
            *table_lines,
            "",
            "The combined Step 8 and Step 10 evidence feeds the Step 11 "
            "decision: a DFT reference dataset (convergence tests first) "
            "is the justified next investigation, with any MACE "
            "fine-tuning deferred to Step 12 after that dataset is "
            "validated.",
            "",
            f"Overall Step 10 status: **{status}**.",
        ]
    )
    _replace_marked_section(
        config.project_root / "README.md",
        README_MARKER_START,
        README_MARKER_END,
        readme_body,
    )
    _replace_marked_section(
        config.project_root / "PROJECT_KNOWLEDGE.md",
        KNOWLEDGE_MARKER_START,
        KNOWLEDGE_MARKER_END,
        knowledge_body,
    )


def execute_pipeline(
    config_path: Path, *, resume: bool, overwrite: bool
) -> tuple[Path, str]:
    """Execute every Step 10 gate sequentially."""

    config = validate_pipeline(config_path)
    hashes, snapshots = validate_step9_success(config)
    executable = discover_lammps(config)

    _conversion_stage(config, resume=resume, overwrite=overwrite)
    for key in STRUCTURE_ORDER:
        validate_converted_bundle(config, key)

    LOGGER.info("Gate 4: primary-potential AlNi pilot")
    _runner.execute_selection(
        config,
        (PILOT_POTENTIAL,),
        (PILOT_PHASE,),
        STATE_ORDER,
        overwrite=overwrite,
        resume=True,
        pilot_only=True,
    )
    for stage in STATE_ORDER:
        validate_state_checkpoint(
            config, PILOT_POTENTIAL, hashes[PILOT_POTENTIAL], PILOT_PHASE, stage
        )
    LOGGER.info("Pilot passed; running the remaining batches")

    for gate, stage in ((5, "initial"), (6, "fixed_cell"), (7, "full_cell")):
        LOGGER.info("Gate %d: %s batch for all 21 combinations", gate, stage)
        _runner.execute_selection(
            config,
            POTENTIAL_ORDER,
            STRUCTURE_ORDER,
            (stage,),
            overwrite=overwrite,
            resume=True,
        )
    records, failures = _formation.load_all_state_records(config)
    if failures or len(records) != 63:
        raise Step10Error(
            f"State validation found {len(records)}/63 valid states and "
            f"{len(failures)} failure(s); Step 10 cannot proceed to analysis."
        )
    LOGGER.info("All 63 state records validated.")

    _analysis_stage(config, resume=resume, overwrite=overwrite)

    LOGGER.info("Gate 10: final report, documentation, and verification")
    published = _load_published(config)
    executable_info = {
        "path": executable.path,
        "version_line": executable.version_line,
        "sha256": executable.sha256,
    }
    report, status = _write_final_report(
        config,
        published,
        executable_info,
        hashes,
        overwrite=True,
        resume=resume,
    )
    _update_documentation(config, published, status)
    verify_snapshots(snapshots)
    for key in STRUCTURE_ORDER:
        validate_converted_bundle(config, key)

    print("=" * 78)
    print("STEP 10 PIPELINE EXECUTION COMPLETED")
    print("=" * 78)
    print(f"Overall Step 10 status: {status}")
    print(f"Final report: {relative_path(report, config.project_root)}")
    print(f"Exact next stage: {NEXT_STAGE_TEXT}")
    print("Step 11 is not implemented by this workflow.")
    print("=" * 78)
    return report, status


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the requested pipeline path, and return a code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        _validate_flags(args)
        if args.validate_only:
            validate_pipeline(args.config)
        else:
            _report, status = execute_pipeline(
                args.config, resume=args.resume, overwrite=args.overwrite
            )
            if status != "SUCCESS":
                return 1
        return 0
    except (Step10Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error(
            "Interrupted at the active gate; completed states and logs were "
            "preserved."
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())

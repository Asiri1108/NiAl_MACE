"""Execute the static LAMMPS relaxations for the Ni-Al benchmark.

Every potential/structure combination is processed through three
independent states: an initial fixed-geometry evaluation (``run 0``), a
Stage A fixed-cell conjugate-gradient minimization, and a Stage B
full-cell zero-pressure minimization (``fix box/relax tri 0.0``) starting
from the validated Stage A result.  LAMMPS is driven through subprocess
argument lists (never ``shell=True``) inside isolated run directories,
with every invocation's input, log, stdout, stderr, thermo history, force
dump, and final structure preserved.

Scientific convergence is decided independently from the parsed maximum
per-atom force and the six pressure components; LAMMPS minimizer
termination is never treated as convergence by itself.  No molecular
dynamics runs, no velocities are assigned, and no thermostat exists in
any generated input.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7_utils import (
    Step7Error,
    atomic_write_text,
    file_sha256,
    relative_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step10_utils import (
    PILOT_PHASE,
    PILOT_POTENTIAL,
    POTENTIAL_ORDER,
    STATE_ORDER,
    STRUCTURE_ORDER,
    THERMO_COLUMNS,
    LammpsExecutable,
    Step10CollisionError,
    Step10Config,
    Step10Error,
    Step10ExecutionError,
    Step10ResumeError,
    build_lammps_input,
    build_state_record,
    check_log_for_errors,
    conversion_paths,
    discover_lammps,
    extract_final_thermo,
    load_step10_config,
    parse_force_dump,
    parse_minimization_stats,
    parse_thermo_sections,
    read_lammps_structure,
    stage_checkpoint_path,
    stage_dir,
    stage_report_path,
    validate_converted_bundle,
    validate_selection,
    validate_state_checkpoint,
    validate_step9_success,
)


LOGGER = logging.getLogger("ni_al_step10.runner")
DEFAULT_CONFIG = Path("configs/ni_al_lammps_benchmark.json")
SUBPROCESS_TIMEOUT_SECONDS = 900


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the LAMMPS runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute the static LAMMPS relaxations for the "
            "Ni-Al classical-potential benchmark."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 10 configuration path, repository-relative by default.",
    )
    parser.add_argument(
        "--potential",
        choices=(*POTENTIAL_ORDER, "all"),
        default="all",
        help="Process one potential or all three (default: all).",
    )
    parser.add_argument(
        "--phase",
        choices=(*STRUCTURE_ORDER, "all"),
        default="all",
        help="Process one structure or all seven (default: all).",
    )
    parser.add_argument(
        "--stage",
        choices=("pilot", "initial", "fixed-cell", "full-cell", "all"),
        default="all",
        help="Process one stage, the pilot, or everything (default: all).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all inputs without running LAMMPS.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute the selected calculations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, compatible, validated outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the selected Step 10 outputs.",
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


def _stage_keys(option: str) -> tuple[str, ...]:
    """Map the --stage option onto internal stage keys."""

    mapping = {
        "initial": ("initial",),
        "fixed-cell": ("fixed_cell",),
        "full-cell": ("full_cell",),
        "all": STATE_ORDER,
        "pilot": STATE_ORDER,
    }
    return mapping[option]


def _run_lammps_cycle(
    executable: LammpsExecutable,
    directory: Path,
    input_name: str,
    log_name: str,
) -> tuple[float, str, str, str]:
    """Run one LAMMPS invocation and return (wall time, log, stdout, stderr)."""

    arguments = [executable.path, "-in", input_name, "-log", log_name]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            arguments,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise Step10ExecutionError(
            f"LAMMPS timed out after {SUBPROCESS_TIMEOUT_SECONDS} s in "
            f"{directory}: {exc}"
        ) from exc
    except OSError as exc:
        raise Step10ExecutionError(
            f"LAMMPS could not be started: {type(exc).__name__}: {exc}"
        ) from exc
    wall = time.perf_counter() - started
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    (directory / input_name.replace("in.", "stdout.").replace(
        ".lammps", ".txt"
    )).write_text(stdout, encoding="utf-8", newline="\n")
    (directory / input_name.replace("in.", "stderr.").replace(
        ".lammps", ".txt"
    )).write_text(stderr, encoding="utf-8", newline="\n")
    log_path = directory / log_name
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    if completed.returncode != 0:
        check_log_for_errors(log_text, stdout, stderr, str(directory))
        raise Step10ExecutionError(
            f"LAMMPS exited with code {completed.returncode} in {directory}; "
            "logs were preserved for inspection."
        )
    check_log_for_errors(log_text, stdout, stderr, str(directory))
    return wall, log_text, stdout, stderr


def _thermo_history_csv(cycle_logs: Sequence[tuple[int, str]]) -> bytes:
    """Serialize every thermo row of every cycle into one CSV."""

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["cycle", *THERMO_COLUMNS])
    for cycle_number, log_text in cycle_logs:
        for section in parse_thermo_sections(log_text):
            for row in section:
                writer.writerow(
                    [cycle_number]
                    + [row.get(name, "") for name in THERMO_COLUMNS]
                )
    return buffer.getvalue().encode("utf-8")


def _stage_report_text(config: Step10Config, record: Mapping[str, Any]) -> str:
    """Render the human-readable stage report."""

    lines = [
        f"Step 10 stage report - {record['potential_key']} / "
        f"{record['phase']} / {record['stage']}",
        "=" * 72,
        f"Material ID: {record['material_id']}",
        f"Potential: {record['potential_path']} "
        f"(sha256 {record['potential_sha256'][:16]}...)",
        f"LAMMPS: {record['lammps_version']}",
        f"Atoms: {record['atom_count']} "
        f"(Al {record['al_count']}, Ni {record['ni_count']})",
        f"Total energy: {record['total_energy_eV']:.12f} eV",
        f"Energy per atom: {record['energy_per_atom_eV']:.12f} eV/atom",
        f"Maximum force: {record['maximum_force_eV_per_A']:.9g} eV/angstrom",
        "Maximum |pressure component|: "
        f"{record['maximum_absolute_pressure_bar']:.9g} bar",
        "Maximum |stress component|: "
        f"{record['maximum_absolute_stress_eV_per_A3']:.9g} eV/angstrom^3",
        f"Volume: {record['volume_A3']:.9f} A^3 "
        f"({record['volume_per_atom_A3']:.9f} A^3/atom)",
        "Volume change versus original: "
        f"{record['volume_change_percent_vs_original']:+.6f}%",
        "Maximum internal displacement versus original: "
        f"{record['maximum_internal_displacement_A']:.9g} A",
        f"Symmetry: {record['symmetry']['space_group_symbol']} "
        f"({record['symmetry']['space_group_number']})",
        f"Minimizer iterations: {record['minimizer_iterations_total']}; "
        f"force evaluations: {record['force_evaluations_total']}; "
        f"stop reason: {record['minimizer_stop_reason']}",
        f"Convergence status: {record['convergence_status']} "
        f"(force_converged={record['force_converged']}, "
        f"pressure_converged={record['pressure_converged']})",
        str(record["convergence_note"]),
        f"Safety status: {record['safety_status']}",
        f"Wall time: {record['wall_time_seconds']:.3f} s "
        f"(LAMMPS loop time {record['lammps_loop_time_seconds']:.3f} s)",
        "No dynamics, velocities, or thermostats were used.",
        "",
    ]
    return "\n".join(lines)


def _existing_stage_outputs(
    config: Step10Config, potential: str, phase: str, stage: str
) -> list[Path]:
    """Return existing artifacts for one stage."""

    directory = stage_dir(config, potential, phase, stage)
    existing: list[Path] = []
    if directory.is_dir():
        existing.extend(sorted(p for p in directory.iterdir() if p.is_file()))
    checkpoint = stage_checkpoint_path(config, potential, phase, stage)
    if checkpoint.exists():
        existing.append(checkpoint)
    report = stage_report_path(config, potential, phase, stage)
    if report.exists():
        existing.append(report)
    return existing


def execute_stage(
    config: Step10Config,
    executable: LammpsExecutable,
    potential: str,
    potential_sha256: str,
    phase: str,
    stage: str,
    *,
    overwrite: bool,
    resume: bool,
) -> Mapping[str, Any]:
    """Execute (or resume) one potential/phase/stage state."""

    existing = _existing_stage_outputs(config, potential, phase, stage)
    if existing and resume:
        try:
            record = validate_state_checkpoint(
                config, potential, potential_sha256, phase, stage, executable
            )
            LOGGER.info(
                "%s/%s/%s: reusing the validated existing state.",
                potential,
                phase,
                stage,
            )
            return record
        except Step10ResumeError:
            if not overwrite:
                raise
    if existing and not overwrite:
        listing = "\n".join(
            f"  - {relative_path(path, config.project_root)}"
            for path in existing
        )
        raise Step10CollisionError(
            f"{potential}/{phase}/{stage}: existing outputs were found; "
            "re-run with --overwrite or --resume:\n" + listing
        )
    directory = stage_dir(config, potential, phase, stage)
    if existing and overwrite:
        if directory.is_dir():
            shutil.rmtree(directory)
        for path in (
            stage_checkpoint_path(config, potential, phase, stage),
            stage_report_path(config, potential, phase, stage),
        ):
            if path.exists():
                path.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    (directory.parent / "checkpoints").mkdir(parents=True, exist_ok=True)
    (directory.parent / "reports").mkdir(parents=True, exist_ok=True)

    conversion_record = validate_converted_bundle(config, phase)
    converted_data = conversion_paths(config, phase)[0]
    original_atoms = read_lammps_structure(
        converted_data, f"{phase} converted data"
    )
    if stage == "full_cell":
        validate_state_checkpoint(
            config, potential, potential_sha256, phase, "fixed_cell", executable
        )
        start_data = (
            stage_dir(config, potential, phase, "fixed_cell") / "final.data"
        )
        if not start_data.is_file():
            raise Step10ResumeError(
                f"{potential}/{phase}: fixed_cell final.data is missing."
            )
    else:
        start_data = converted_data
    start_atoms = read_lammps_structure(
        start_data, f"{potential}/{phase}/{stage} start structure"
    )
    source_data_sha = file_sha256(start_data)

    started_at = utc_timestamp()
    total_wall = 0.0
    cycles: list[dict[str, Any]] = []
    cycle_logs: list[tuple[int, str]] = []
    maximum_cycles = (
        1 if stage == "initial" else int(config.minimization["maximum_cycles"])
    )
    current_data = start_data
    final_row: Mapping[str, float] | None = None
    for cycle in range(1, maximum_cycles + 1):
        input_name = f"in.cycle{cycle}.lammps"
        log_name = f"log.cycle{cycle}.lammps"
        dump_name = f"forces.cycle{cycle}.dump"
        data_name = f"final.cycle{cycle}.data"
        input_text = build_lammps_input(
            config,
            config.potentials[potential].path,
            current_data,
            stage,
            dump_name,
            data_name,
        )
        (directory / input_name).write_text(
            input_text, encoding="utf-8", newline="\n"
        )
        wall, log_text, stdout, stderr = _run_lammps_cycle(
            executable, directory, input_name, log_name
        )
        del stdout, stderr
        total_wall += wall
        row = extract_final_thermo(
            log_text, f"{potential}/{phase}/{stage} cycle {cycle}"
        )
        stats = parse_minimization_stats(
            log_text, f"{potential}/{phase}/{stage} cycle {cycle}"
        )
        cycles.append(
            {
                "cycle": cycle,
                "input_file": input_name,
                "log_file": log_name,
                "wall_time_seconds": wall,
                "potential_energy_eV": row["PotEng"],
                "maximum_force_eV_per_A": row["c_fmax"],
                "maximum_absolute_pressure_bar": max(
                    abs(row[name])
                    for name in ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
                ),
                "volume_A3": row["Volume"],
                **stats,
            }
        )
        cycle_logs.append((cycle, log_text))
        final_row = row
        force_ok = row["c_fmax"] <= config.force_threshold_eV_per_A
        pressure_ok = (
            cycles[-1]["maximum_absolute_pressure_bar"]
            <= config.pressure_threshold_bar
        )
        if stage == "initial":
            break
        if stage == "fixed_cell" and force_ok:
            break
        if stage == "full_cell" and force_ok and pressure_ok:
            break
        current_data = directory / data_name
    if final_row is None:
        raise Step10ExecutionError(
            f"{potential}/{phase}/{stage}: no cycle produced thermo output."
        )
    last_cycle = cycles[-1]["cycle"]
    final_data_cycle = directory / f"final.cycle{last_cycle}.data"
    final_dump = directory / f"forces.cycle{last_cycle}.dump"
    canonical_final = directory / "final.data"
    shutil.copyfile(final_data_cycle, canonical_final)
    final_atoms = read_lammps_structure(
        canonical_final, f"{potential}/{phase}/{stage} final structure"
    )
    forces = parse_force_dump(
        final_dump,
        config.structures[phase].expected_atoms,
        f"{potential}/{phase}/{stage}",
    )
    completed_at = utc_timestamp()
    record = build_state_record(
        config,
        potential,
        potential_sha256,
        phase,
        stage,
        final_row,
        forces,
        final_atoms,
        original_atoms,
        start_atoms,
        cycles,
        executable,
        total_wall,
        source_data_sha,
        {"started_at_utc": started_at, "completed_at_utc": completed_at},
    )
    record = dict(record)
    record["source_conversion_sha256"] = conversion_record["data_file_sha256"]
    record["start_structure_path"] = relative_path(
        start_data, config.project_root
    )

    (directory / "thermo_history.csv").write_bytes(
        _thermo_history_csv(cycle_logs)
    )
    final_extxyz = directory / "final.extxyz"
    try:
        from ase.io import write as ase_write

        output_atoms = final_atoms.copy()
        output_atoms.info.update(
            {
                "step10_potential": potential,
                "step10_phase": phase,
                "step10_stage": stage,
                "step10_material_id": record["material_id"],
                "step10_total_energy_eV": record["total_energy_eV"],
                "step10_convergence_status": record["convergence_status"],
                "step10_configuration_sha256": config.fingerprint,
            }
        )
        ase_write(final_extxyz, output_atoms, format="extxyz")
    except Exception as exc:
        raise Step10ExecutionError(
            f"{potential}/{phase}/{stage}: could not write final EXTXYZ: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    record["artifacts"] = {
        "final_data_sha256": file_sha256(canonical_final),
        "final_extxyz_sha256": file_sha256(final_extxyz),
        "final_dump_sha256": file_sha256(final_dump),
        "last_log_sha256": file_sha256(
            directory / f"log.cycle{last_cycle}.lammps"
        ),
    }

    checkpoint = stage_checkpoint_path(config, potential, phase, stage)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        checkpoint.unlink()
    checkpoint.write_bytes(write_strict_json_bytes(record))
    report = stage_report_path(config, potential, phase, stage)
    atomic_write_text(
        report, _stage_report_text(config, record), overwrite=True
    )
    if stage != "initial" and record["convergence_status"] != "CONVERGED":
        raise Step10ExecutionError(
            f"{potential}/{phase}/{stage}: final state is NOT_CONVERGED "
            f"(max force {record['maximum_force_eV_per_A']:.6g} eV/A, max "
            f"|pressure| {record['maximum_absolute_pressure_bar']:.6g} bar) "
            "after the configured cycles; logs and the diagnostic checkpoint "
            "were preserved."
        )
    LOGGER.info(
        "%s/%s/%s: %s (E=%.9f eV/atom; fmax=%.3g eV/A; |P|max=%.4g bar; "
        "%.2f s)",
        potential,
        phase,
        stage,
        record["convergence_status"],
        record["energy_per_atom_eV"],
        record["maximum_force_eV_per_A"],
        record["maximum_absolute_pressure_bar"],
        total_wall,
    )
    return record


def run_validate_only(
    config: Step10Config,
    potentials: Sequence[str],
    phases: Sequence[str],
) -> None:
    """Validate every input for the selected scope without running LAMMPS."""

    hashes, _snapshots = validate_step9_success(config)
    executable = discover_lammps(config)
    conversion_ready = True
    for phase in phases:
        try:
            validate_converted_bundle(config, phase)
        except Step10Error:
            conversion_ready = False
    print("=" * 78)
    print("STEP 10 LAMMPS RUNNER VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Potentials: {', '.join(potentials)} (hashes match Step 9)")
    print(f"Structures: {', '.join(phases)}")
    print(f"LAMMPS executable: {executable.path}")
    print(f"LAMMPS version: {executable.version_line}")
    print("eam/alloy available: Yes")
    print(
        "Converted structures ready: "
        + ("Yes" if conversion_ready else "No (run the conversion first)")
    )
    print("LAMMPS simulation executed: No")
    print("Validation status: SUCCESS")
    print("=" * 78)
    del hashes


def execute_selection(
    config: Step10Config,
    potentials: Sequence[str],
    phases: Sequence[str],
    stages: Sequence[str],
    *,
    overwrite: bool,
    resume: bool,
    pilot_only: bool = False,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Execute the selected scope stage-major and return all records."""

    hashes, _snapshots = validate_step9_success(config)
    executable = discover_lammps(config)
    results: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    if pilot_only:
        combos = [(PILOT_POTENTIAL, PILOT_PHASE)]
    else:
        combos = [
            (potential, phase) for potential in potentials for phase in phases
        ]
    for stage in stages:
        for potential, phase in combos:
            record = execute_stage(
                config,
                executable,
                potential,
                hashes[potential],
                phase,
                stage,
                overwrite=overwrite,
                resume=resume,
            )
            results[(potential, phase, stage)] = record
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.resume and args.overwrite:
        LOGGER.error("--resume and --overwrite are mutually exclusive.")
        return 1
    if (args.resume or args.overwrite) and not args.execute:
        LOGGER.error("--resume/--overwrite are allowed only with --execute.")
        return 1
    try:
        config = load_step10_config(args.config)
        potentials = (
            validate_selection(None, POTENTIAL_ORDER, "potential")
            if args.potential == "all"
            else validate_selection((args.potential,), POTENTIAL_ORDER, "potential")
        )
        phases = (
            validate_selection(None, STRUCTURE_ORDER, "structure")
            if args.phase == "all"
            else validate_selection((args.phase,), STRUCTURE_ORDER, "structure")
        )
        stages = _stage_keys(args.stage)
        if args.validate_only:
            run_validate_only(config, potentials, phases)
            return 0
        results = execute_selection(
            config,
            potentials,
            phases,
            stages,
            overwrite=args.overwrite,
            resume=args.resume,
            pilot_only=args.stage == "pilot",
        )
        print("=" * 78)
        print(f"STEP 10 LAMMPS EXECUTION COMPLETED ({len(results)} state(s))")
        print("=" * 78)
        return 0
    except (Step10Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; completed states and logs were preserved.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

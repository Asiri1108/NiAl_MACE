"""Run the Ni-Al MACE Step 8 benchmark gates in a controlled sequence.

The orchestrator imports and calls controlled Python functions rather than
constructing shell command strings.  Validation-only mode performs every
static gate without network retrieval, without loading MACE, without
creating an optimizer, without executing DFT, and without writing any
comparison output.

Execution order:

1.  Preflight (Step 7 SUCCESS, configuration, installed APIs, key presence).
2.  Retrieve the five MP DFT benchmark records by exact material ID.
3.  Validate the benchmark records (identity, provenance, structures).
4.  Validate the Step 7/6 MACE results.
5.  Calculate formation-energy errors and structural comparisons.
6.  Calculate statistical and ranking summaries.
7.  Create tables, figures, reports, and the checkpoint.
8.  Update documentation from actual results.
9.  Verify every protected file fingerprint.

Step 9 (classical-potential comparison) is deliberately not implemented.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import compare_ni_al_mace_vs_mp_dft as _compare
import fetch_ni_al_mp_dft_benchmarks as _fetch
from step7_utils import (
    Step7Error,
    installed_step7_versions,
    read_strict_json,
    relative_path,
    verify_snapshots,
)
from step8_utils import (
    PHASE_ORDER,
    Step8ApiError,
    Step8Config,
    Step8ConfigurationError,
    Step8Error,
    Step8ResumeError,
    benchmark_phase_paths,
    calculate_comparisons,
    load_benchmark_records,
    load_step8_config,
    step8_output_paths,
    validate_step7_sources,
)


LOGGER = logging.getLogger("ni_al_step8.pipeline")
DEFAULT_CONFIG = Path("configs/mace_dft_benchmark.json")
README_MARKER_START = "<!-- NI_AL_STEP8_START -->"
README_MARKER_END = "<!-- NI_AL_STEP8_END -->"
KNOWLEDGE_MARKER_START = "<!-- NI_AL_STEP8_KNOWLEDGE_START -->"
KNOWLEDGE_MARKER_END = "<!-- NI_AL_STEP8_KNOWLEDGE_END -->"
NEXT_STAGE_TEXT = (
    "Step 9 - Select and document candidate classical Ni-Al interatomic "
    "potentials and design the LAMMPS comparison."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Step 8 pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate or sequentially execute Ni-Al MACE Step 8."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 8 configuration path, repository-relative by default.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all static validation gates without retrieval or writes.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute the complete Step 8 workflow.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only validated, compatible, complete Step 8 bundles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace only Step 8 outputs.",
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
        raise Step8ConfigurationError(
            "--resume and --overwrite are execution options and cannot be "
            "used with --validate-only."
        )
    if args.resume and args.overwrite:
        raise Step8ConfigurationError(
            "--resume and --overwrite are mutually exclusive."
        )


def _assert_heavy_modules_not_imported() -> None:
    """Fail when MACE or FIRE was imported on a validation-only path."""

    loaded_mace = sorted(
        name for name in sys.modules if name == "mace" or name.startswith("mace.")
    )
    loaded_fire = sorted(
        name
        for name in sys.modules
        if name == "ase.optimize.fire" or name.startswith("ase.optimize.fire.")
    )
    if loaded_mace:
        raise Step8Error(
            "Validation-only imported MACE unexpectedly: "
            + ", ".join(loaded_mace)
        )
    if loaded_fire:
        raise Step8Error(
            "Validation-only imported FIRE unexpectedly: "
            + ", ".join(loaded_fire)
        )


def validate_pipeline(config_path: Path) -> Step8Config:
    """Run every static Step 8 validation gate without side effects."""

    LOGGER.info("Gate 1: Step 7 source preflight")
    config = load_step8_config(config_path)
    mace = validate_step7_sources(config)
    LOGGER.info(
        "Validated %d Step 7 MACE phase records; Step 7 status is SUCCESS.",
        len(mace.records),
    )

    LOGGER.info("Gate 2: Step 8 configuration and installed public APIs")
    versions = installed_step7_versions()
    import inspect

    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise Step8Error(
            f"The official mp-api client is unavailable: {exc}"
        ) from exc
    if "api_key" not in inspect.signature(MPRester).parameters:
        raise Step8Error("Installed MPRester does not accept api_key.")

    api_key_line = "Yes (value not printed)"
    api_key_available = True
    try:
        _fetch._load_api_key_safely(config)
    except Step8Error as exc:
        api_key_available = False
        api_key_line = f"No - {exc}"

    benchmarks_present = all(
        path.is_file()
        for phase in PHASE_ORDER
        for path in benchmark_phase_paths(config, phase)
    )
    if benchmarks_present:
        load_benchmark_records(config)
        LOGGER.info("Existing raw benchmark bundles validated.")

    planned = list(step8_output_paths(config).all_paths())
    collisions = [path for path in planned if path.exists()]
    verify_snapshots(mace.snapshots)
    _assert_heavy_modules_not_imported()

    print("=" * 78)
    print("STEP 8 PIPELINE VALIDATION")
    print("=" * 78)
    print("Step 7 source validation: SUCCESS")
    print("Step 8 configuration validation: SUCCESS")
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"API key variable available: {api_key_line}")
    print(
        "Raw benchmark bundles present: "
        + ("Yes (validated)" if benchmarks_present else "No (fetch pending)")
    )
    print(
        "Planned Step 8 output collisions: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print(
        "Installed packages: "
        + "; ".join(
            f"{name}={version}" for name, version in sorted(versions.items())
        )
    )
    print("Materials Project query executed: No")
    print("MACE loaded: No")
    print("Optimizer created: No")
    print("DFT executed: No")
    print("Comparison calculated: No")
    print("Protected files modified: No")
    print(
        "Validation status: " + ("SUCCESS" if api_key_available else "FAILED")
    )
    print("=" * 78)
    if not api_key_available:
        raise Step8ApiError(
            "MP_API_KEY is missing; Step 8 execution cannot retrieve the DFT "
            "benchmark records. Configure the key locally in .env and re-run "
            r".\.venv\Scripts\python.exe scripts\run_step8_pipeline.py "
            "--execute"
        )
    return config


def _fetch_stage(config: Step8Config, *, resume: bool, overwrite: bool) -> None:
    """Gate 3: obtain the benchmark records from Materials Project."""

    present = all(
        path.is_file()
        for phase in PHASE_ORDER
        for path in benchmark_phase_paths(config, phase)
    )
    if present and resume:
        load_benchmark_records(config)
        LOGGER.info("Gate 3: reusing validated existing benchmark bundles.")
        return
    LOGGER.info("Gate 3: fetching DFT benchmark records from Materials Project")
    _fetch.run_fetch(config, PHASE_ORDER, overwrite=overwrite)


def _validate_existing_comparison(config: Step8Config) -> None:
    """Validate that a published comparison bundle matches current inputs."""

    outputs = step8_output_paths(config)
    missing = [path for path in outputs.all_paths() if not path.is_file()]
    if missing:
        raise Step8ResumeError(
            "Step 8 comparison bundle is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    document = read_strict_json(
        outputs.energy_json, "published Step 8 energy JSON"
    )
    if document.get("configuration_fingerprint_sha256") != config.fingerprint:
        raise Step8ResumeError(
            "Published Step 8 energy JSON was produced by a different "
            "configuration."
        )
    benchmarks, _snapshots = load_benchmark_records(config)
    mace = validate_step7_sources(config)
    records = calculate_comparisons(config, benchmarks, mace)
    published = {
        row.get("phase_key"): row
        for row in document.get("records", ())
        if isinstance(row, Mapping)
    }
    for record in records:
        row = published.get(record.phase_key)
        if row is None:
            raise Step8ResumeError(
                f"Published comparison lacks {record.phase_key}."
            )
        stored = row.get("relaxed_signed_error_eV_per_atom")
        if not isinstance(stored, (int, float)) or not math.isclose(
            float(stored),
            record.relaxed_signed_error_eV_per_atom,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise Step8ResumeError(
                f"Published signed error for {record.phase_key} disagrees "
                "with the current validated inputs."
            )


def _comparison_stage(
    config: Step8Config, *, resume: bool, overwrite: bool
) -> None:
    """Gates 4-9: MACE validation, errors, structure, statistics, outputs."""

    outputs = step8_output_paths(config)
    existing = [path for path in outputs.all_paths() if path.exists()]
    if existing and resume:
        LOGGER.info("Gates 4-9: validating the existing comparison bundle.")
        _validate_existing_comparison(config)
        return
    LOGGER.info("Gates 4-9: calculating the comparison bundle")
    _compare.run_compare(config, overwrite=overwrite)


def _replace_marked_section(
    path: Path, start_marker: str, end_marker: str, body: str
) -> None:
    """Append or replace one generated documentation section atomically."""

    original = path.read_text(encoding="utf-8")
    section = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    if start_marker in original or end_marker in original:
        if original.count(start_marker) != 1 or original.count(end_marker) != 1:
            raise Step8Error(
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


def _phase_table_lines(document: Mapping[str, Any]) -> list[str]:
    """Build a compact Markdown table from the published comparison."""

    lines = [
        "| Phase | MP DFT E_f (eV/atom) | MACE relaxed E_f (eV/atom) | "
        "Signed error (eV/atom) | MP hull (eV/atom) | dV/atom (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    structural_by_phase: dict[str, Mapping[str, Any]] = {}
    for row in document.get("structural_records", ()):
        if isinstance(row, Mapping):
            structural_by_phase[str(row.get("phase_key"))] = row
    for row in document.get("records", ()):
        if not isinstance(row, Mapping):
            continue
        phase = str(row.get("phase_key"))
        structural = structural_by_phase.get(phase, {})
        lines.append(
            f"| {phase} | "
            f"{row.get('mp_formation_energy_eV_per_atom'):.6f} | "
            f"{row.get('mace_relaxed_formation_energy_eV_per_atom'):.6f} | "
            f"{row.get('relaxed_signed_error_eV_per_atom'):+.6f} | "
            f"{row.get('mp_energy_above_hull_eV_per_atom'):.6f} | "
            f"{structural.get('volume_per_atom_difference_percent'):+.4f} |"
        )
    return lines


def _update_documentation(config: Step8Config) -> str:
    """Document methodology and actual validated Step 8 results."""

    outputs = step8_output_paths(config)
    checkpoint = read_strict_json(outputs.checkpoint, "Step 8 checkpoint")
    statistics = checkpoint.get("statistics", {})
    structural_summary = checkpoint.get("structural_summary", {})
    status = str(checkpoint.get("overall_status", "PARTIAL"))
    table = _phase_table_lines(checkpoint)
    database_versions = sorted(
        {
            str(row.get("mp_database_version"))
            for row in checkpoint.get("records", ())
            if isinstance(row, Mapping)
            and row.get("mp_database_version") is not None
        }
    )
    readme_body = "\n".join(
        [
            "## Step 8 - MACE vs Materials Project DFT Benchmark",
            "",
            "Step 8 retrieves the five selected phases by exact material ID "
            "from the official Materials Project summary endpoint and "
            "benchmarks the Step 7 relaxed MACE formation energies and "
            "Step 6 MACE-relaxed structures against the MP processed "
            "DFT-derived references. Processed `formation_energy_per_atom` "
            "is used because it is Materials Project's recommended, "
            "correction-consistent thermodynamic quantity; raw MACE and "
            "VASP total energies use incompatible reference scales and are "
            "never compared. The signed error is `MACE - MP DFT` in "
            "eV/atom.",
            "",
            "Material IDs: "
            + ", ".join(
                f"{phase}={config.phases[phase]}" for phase in PHASE_ORDER
            )
            + ". Materials Project database version: "
            + ("; ".join(database_versions) if database_versions else "n/a")
            + ".",
            "",
            *table,
            "",
            "Aggregate (n=5): MAE = "
            f"{statistics.get('mean_absolute_error_eV_per_atom'):.6f} "
            "eV/atom; RMSE = "
            f"{statistics.get('root_mean_squared_error_eV_per_atom'):.6f} "
            "eV/atom; mean signed error = "
            f"{statistics.get('mean_signed_error_eV_per_atom'):+.6f} "
            "eV/atom; exact ranking agreement = "
            f"{statistics.get('exact_ranking_agreement')}; pairwise ordering "
            f"agreement = {statistics.get('pairwise_ordering_agreement')}. "
            "Volume: mean signed error = "
            f"{structural_summary.get('mean_signed_volume_percent_error'):+.4f}%"
            "; symmetry agreement = "
            f"{structural_summary.get('symmetry_agreement_count')}/5 "
            "(symprec 0.001 A, angle tolerance 5 deg).",
            "",
            "MP energy above hull is DFT context computed against the full "
            "MP Ni-Al entry set; it is not comparable to and was never "
            "subtracted from the Step 7 selected-set envelope. Five phases "
            "are a small sample: statistics are descriptive, correlations "
            "exploratory, and no universal MACE accuracy claim is made.",
            "",
            "Implementation: `scripts/step8_utils.py`, "
            "`scripts/fetch_ni_al_mp_dft_benchmarks.py`, "
            "`scripts/compare_ni_al_mace_vs_mp_dft.py`, and "
            "`scripts/run_step8_pipeline.py`, with settings in "
            "`configs/mace_dft_benchmark.json`. Outputs are under "
            "`results/mace_vs_dft/`; the authoritative report is "
            "`results/mace_vs_dft/reports/ni_al_step8_final_report.txt`.",
            "",
            "Commands:",
            "",
            "```bat",
            r".\.venv\Scripts\python.exe scripts\fetch_ni_al_mp_dft_benchmarks.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\fetch_ni_al_mp_dft_benchmarks.py --fetch",
            r".\.venv\Scripts\python.exe scripts\compare_ni_al_mace_vs_mp_dft.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\compare_ni_al_mace_vs_mp_dft.py --compare",
            r".\.venv\Scripts\python.exe scripts\run_step8_pipeline.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_step8_pipeline.py --execute",
            r".\.venv\Scripts\python.exe scripts\run_step8_pipeline.py --execute --resume",
            "```",
            "",
            f"Actual overall Step 8 status: **{status}**. No DFT was run, "
            "no LAMMPS or fine-tuning was implemented, and no automatic "
            "fine-tuning decision was made. The exact next stage is:",
            "",
            NEXT_STAGE_TEXT,
            "",
            "Step 9 is not implemented here.",
        ]
    )
    knowledge_body = "\n".join(
        [
            "## Step 8 Research-Log Entry (2026-07-28)",
            "",
            "A MACE formation energy and an MP DFT formation energy are the "
            "same physical definition evaluated on two different energy "
            "surfaces: each subtracts its own elemental references, so the "
            "two are comparable while raw totals are not. Materials Project "
            "publishes processed thermodynamic entries (its recommended "
            "correction/mixing scheme, recorded per phase via the thermo "
            "endpoint), which is why the processed "
            "`formation_energy_per_atom` is the benchmark rather than any "
            "raw VASP total.",
            "",
            "The signed error (MACE - MP DFT) keeps the direction of the "
            "bias visible; MAE averages magnitudes and RMSE additionally "
            "weights outliers. A systematic bias means the signed errors "
            "share one sign rather than scattering around zero. Ranking "
            "agreement asks whether both methods order the five phases "
            "identically by formation energy - relevant because many alloy "
            "conclusions depend on ordering rather than absolute values.",
            "",
            "MP energy above hull is computed against every Ni-Al entry in "
            "Materials Project, while the Step 7 envelope contains only "
            "seven points on the MACE surface; the two answer different "
            "questions and were never subtracted. Volume-per-atom is "
            "compared directly, while lattice parameters are compared only "
            "after both structures pass through the same pymatgen "
            "conventional standardization, because primitive and "
            "conventional representations would otherwise differ trivially.",
            "",
            "Actual Step 8 findings (n=5): MAE = "
            f"{statistics.get('mean_absolute_error_eV_per_atom'):.6f} "
            "eV/atom; RMSE = "
            f"{statistics.get('root_mean_squared_error_eV_per_atom'):.6f} "
            "eV/atom; mean signed error = "
            f"{statistics.get('mean_signed_error_eV_per_atom'):+.6f} "
            "eV/atom; all signed errors positive = "
            f"{statistics.get('all_signed_errors_positive')}; exact ranking "
            f"agreement = {statistics.get('exact_ranking_agreement')}; "
            "pairwise agreement = "
            f"{statistics.get('pairwise_ordering_agreement')}; mean signed "
            "volume error = "
            f"{structural_summary.get('mean_signed_volume_percent_error'):+.4f}%"
            "; symmetry agreement = "
            f"{structural_summary.get('symmetry_agreement_count')}/5.",
            "",
            *table,
            "",
            "Ni remains a magnetic element in DFT descriptions while the "
            "structural MACE workflow exposes no spin input, so part of the "
            "Ni-rich error budget may be magnetic; this is recorded, not "
            "resolved. The next research decision (whether fine-tuning is "
            "justified) must weigh the formation-energy bias, the "
            "single-signed volume error, the preserved or broken ranking, "
            "the Ni magnetic limitation, and the five-phase sample size - "
            "no undocumented universal threshold decides it.",
            "",
            f"Overall Step 8 status: **{status}**.",
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
    return status


def execute_pipeline(
    config_path: Path, *, resume: bool, overwrite: bool
) -> tuple[Path, str]:
    """Execute every Step 8 gate sequentially and return the final report."""

    config = validate_pipeline(config_path)
    mace = validate_step7_sources(config)

    _fetch_stage(config, resume=resume, overwrite=overwrite)

    LOGGER.info("Gate 4: validating the retrieved benchmark records")
    load_benchmark_records(config)

    _comparison_stage(config, resume=resume, overwrite=overwrite)

    LOGGER.info("Gate 10: documentation update and protected-file verification")
    status = _update_documentation(config)
    verify_snapshots(mace.snapshots)
    load_benchmark_records(config)

    report = step8_output_paths(config).final_report
    print("=" * 78)
    print("STEP 8 PIPELINE EXECUTION COMPLETED")
    print("=" * 78)
    print(f"Overall Step 8 status: {status}")
    print(f"Final report: {relative_path(report, config.project_root)}")
    print(f"Exact next stage: {NEXT_STAGE_TEXT}")
    print("Step 9 is not implemented by this workflow.")
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
    except (Step8Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error(
            "Interrupted at the active gate; prior validated outputs were "
            "preserved."
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())

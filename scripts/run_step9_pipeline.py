"""Run the Ni-Al Step 9 potential-selection gates in a controlled sequence.

The orchestrator imports and calls controlled Python functions rather than
constructing shell command strings.  Validation-only mode performs every
static gate without network retrieval, without LAMMPS execution, without
MACE, without DFT, and without writing any scientific output.

Execution order:

1.  Preflight (Step 8 SUCCESS, configuration, source policy).
2.  Retrieve the three official candidate bundles from NIST.
3.  Validate every setfl file, metadata record, and processed copy.
4.  Review candidate scientific scope and provenance.
5.  Inspect local LAMMPS availability (no simulation).
6.  Build the candidate evaluation matrix and select roles.
7.  Create the Step 10 benchmark design plan.
8.  Write reports, update documentation, and verify protected files.

Step 10 is deliberately not implemented and no LAMMPS simulation runs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import design_ni_al_lammps_benchmark as _design
import fetch_ni_al_classical_potentials as _fetch
from step7_utils import (
    Step7Error,
    installed_step7_versions,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    stage_path,
    utc_timestamp,
    verify_snapshots,
    write_strict_json_bytes,
)
from step9_utils import (
    CANDIDATE_ORDER,
    ROLE_RANKS,
    LammpsAvailability,
    Step9CollisionError,
    Step9Config,
    Step9ConfigurationError,
    Step9Error,
    Step9ResumeError,
    ValidatedCandidate,
    build_evaluation_matrix,
    candidate_bundle_paths,
    ev_per_A3_to_bar,
    inspect_lammps_availability,
    load_step9_config,
    pair_coeff_line,
    step9_directories,
    step9_result_paths,
    validate_candidate_bundle,
    validate_step8_success,
)


LOGGER = logging.getLogger("ni_al_step9.pipeline")
DEFAULT_CONFIG = Path("configs/ni_al_classical_potentials.json")
# Both marked sections below are written into docs/RESEARCH_LOG.md: the README_*
# pair holds the step summary, the KNOWLEDGE_* pair the reasoning entry.
README_MARKER_START = "<!-- NI_AL_STEP9_START -->"
README_MARKER_END = "<!-- NI_AL_STEP9_END -->"
KNOWLEDGE_MARKER_START = "<!-- NI_AL_STEP9_KNOWLEDGE_START -->"
KNOWLEDGE_MARKER_END = "<!-- NI_AL_STEP9_KNOWLEDGE_END -->"
NEXT_STAGE_TEXT = (
    "Step 10 - Execute the designed LAMMPS benchmark: convert structures, "
    "relax with each validated classical potential, and compare formation "
    "energies and structures against MACE and the Materials Project DFT "
    "references."
)
EXCLUDED_CATEGORIES: tuple[str, ...] = (
    "ternary Ni-Al-Co, Ni-Al-H, and Fe-Ni-Al potentials",
    "pure-element-only Al or Ni potentials",
    "arbitrary tutorial or unverified personal-repository files",
    "hybrid combinations of unrelated Al-Al, Ni-Ni, and Al-Ni functions "
    "or mixing rules",
    "Lennard-Jones approximations and ReaxFF",
    "potentials without documented Ni-Al cross interactions or with "
    "unclear license/citation/provenance",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Step 9 pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate or sequentially execute Ni-Al Step 9."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 9 configuration path, repository-relative by default.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation without network retrieval or output creation.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute retrieval, validation, selection, and design.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only fully validated compatible candidate bundles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 9 files.",
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
        raise Step9ConfigurationError(
            "--resume and --overwrite are execution options and cannot be "
            "used with --validate-only."
        )
    if args.resume and args.overwrite:
        raise Step9ConfigurationError(
            "--resume and --overwrite are mutually exclusive."
        )


def validate_pipeline(config_path: Path) -> Step9Config:
    """Run every static Step 9 validation gate without side effects."""

    LOGGER.info("Gate 1: Step 8 source preflight")
    config = load_step9_config(config_path)
    step8_snapshots = validate_step8_success(config)
    LOGGER.info("Step 8 final status is SUCCESS.")

    LOGGER.info("Gate 2/3: configuration and source-policy validation")
    versions = installed_step7_versions()
    lammps = inspect_lammps_availability(config)
    planned = list(step9_result_paths(config).values())
    for key in CANDIDATE_ORDER:
        planned.extend(candidate_bundle_paths(config, key).all_paths())
    collisions = [path for path in planned if path.exists()]
    verify_snapshots(step8_snapshots)

    print("=" * 78)
    print("STEP 9 PIPELINE VALIDATION")
    print("=" * 78)
    print("Step 8 source validation: SUCCESS")
    print("Step 9 configuration validation: SUCCESS")
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Candidate identities validated: {len(CANDIDATE_ORDER)}")
    print(
        "Authoritative source policy: HTTPS-only from "
        + ", ".join(config.source_policy.allowed_download_hosts)
        + " (NIST Interatomic Potentials Repository); unverified sources "
        "rejected"
    )
    print(f"LAMMPS availability: {lammps.status} ({lammps.detail})")
    print(
        "Existing Step 9 outputs: "
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
    print("Network download executed: No")
    print("LAMMPS simulation executed: No")
    print("MACE loaded: No")
    print("Optimizer created: No")
    print("DFT executed: No")
    print("Scientific energy calculated: No")
    print("Protected files modified: No")
    print("Validation status: SUCCESS")
    print("=" * 78)
    return config


def _fetch_stage(
    config: Step9Config, *, resume: bool, overwrite: bool
) -> None:
    """Gate 4: obtain the three official candidate bundles."""

    complete = all(
        path.is_file()
        for key in CANDIDATE_ORDER
        for path in candidate_bundle_paths(config, key).required_paths()
    )
    if complete and resume:
        for key in CANDIDATE_ORDER:
            validate_candidate_bundle(config, key)
        LOGGER.info("Gate 4: reusing validated existing candidate bundles.")
        return
    LOGGER.info("Gate 4: retrieving official candidate bundles from NIST")
    _fetch.run_fetch(config, CANDIDATE_ORDER, overwrite=overwrite)


def _matrix_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize the evaluation matrix to CSV with flattened lists."""

    fieldnames = tuple(rows[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        flattened = {
            key: (
                "; ".join(str(item) for item in value)
                if isinstance(value, list)
                else value
            )
            for key, value in row.items()
        }
        writer.writerow(flattened)
    return buffer.getvalue().encode("utf-8")


def _selection_report_text(
    config: Step9Config,
    validated: Mapping[str, ValidatedCandidate],
    lammps: LammpsAvailability,
) -> str:
    """Build the human-readable candidate selection report."""

    lines = [
        "Step 9 - Classical Ni-Al Potential Selection Report",
        "=" * 76,
        "",
        "Authoritative source policy",
        "---------------------------",
        "All files were retrieved over HTTPS exclusively from the NIST "
        "Interatomic Potentials Repository (www.ctcms.nist.gov); OpenKIM "
        "identifiers and the original peer-reviewed publications provide "
        "cross-reference provenance. Unverified sources were rejected.",
        "",
        "Selection decision (evidence-based, qualitative)",
        "------------------------------------------------",
    ]
    for key in config.candidate_order:
        bundle = validated[key]
        spec = bundle.spec
        rank, rank_label = ROLE_RANKS[spec.role]
        lines.extend(
            [
                f"{rank}. {rank_label}: {key}",
                f"   {spec.potential_name}",
                f"   Citation: {spec.citation}",
                f"   DOI: {spec.doi}",
                f"   NIST implementation: {spec.implementation_identity}",
                f"   OpenKIM: {spec.openkim_extended_id} "
                f"(family {spec.openkim_family_id}; driver "
                f"{spec.openkim_model_driver_family})",
                f"   File: {spec.expected_filename} "
                f"(sha256 {bundle.raw_sha256})",
                f"   File element order: {' '.join(bundle.setfl.elements)}; "
                f"cutoff {bundle.setfl.cutoff_A:.6f} A",
                f"   Planned mapping: {pair_coeff_line(spec, config)}",
                f"   Scope: {spec.fitting_scope}",
            ]
        )
        if spec.known_warnings:
            for warning in spec.known_warnings:
                lines.append(f"   Warning: {warning}")
        lines.append("")
    lines.extend(
        [
            "Primary rationale",
            "-----------------",
            "pun_mishin_2009 is the primary candidate because it is binary "
            "Ni-Al specific, builds on established pure Al and pure Ni "
            "descriptions, fits the cross interaction to B2-NiAl properties "
            "plus ab initio formation energies of multiple intermetallic "
            "structures, and targets broader Ni-Al phase, interface, and "
            "mechanical applications - the closest match to a general "
            "comparison across the five selected compositions. This is a "
            "documented-scope argument, not a claim of universal accuracy.",
            "",
            "mishin_2004_ipr2 is the secondary sensitivity test: an "
            "important gamma/gamma-prime (Ni3Al-focused) model whose "
            "narrower intended scope makes it a probe of how fitting "
            "emphasis affects transferability. Only the corrected ipr2 "
            "implementation is accepted: the superseded ipr1 file has "
            "non-zero isolated-atom energies, while ipr2 explicitly sets "
            "F(rho=0)=0 so isolated atoms have zero energy. Project "
            "formation energies still use consistently relaxed bulk "
            "elemental references, never isolated atoms.",
            "",
            "mishin_2002 is the historical secondary: an established "
            "B2-NiAl EAM retained for model-generation sensitivity. Its "
            "documented warning about weaker pure-element behavior is "
            "recorded evidence, and age alone is not treated as proof of "
            "lower accuracy.",
            "",
            "Excluded or deferred categories",
            "-------------------------------",
            *(f"- {item}" for item in EXCLUDED_CATEGORIES),
            "No cross interactions were created by mixing rules and no "
            "separate pure-element EAM files were combined; every accepted "
            "file provides the alloy interaction explicitly in a supported "
            "multielement setfl form.",
            "",
            "LAMMPS availability",
            "-------------------",
            f"Status: {lammps.status}",
            f"Detail: {lammps.detail}",
            "LAMMPS absence does not affect this selection; it only leaves "
            "Step 10 execution readiness incomplete.",
            "",
        ]
    )
    return "\n".join(lines)


def _final_report_text(
    config: Step9Config,
    validated: Mapping[str, ValidatedCandidate],
    lammps: LammpsAvailability,
    plan: Mapping[str, Any],
    status: str,
) -> str:
    """Build the authoritative Step 9 final report."""

    stress_bar = ev_per_A3_to_bar(0.0006241509)
    inventory_roots = (
        config.raw_root,
        config.selected_root,
        config.result_root,
    )
    inventory = sorted(
        {
            relative_path(path, config.project_root)
            for root in inventory_roots
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file()
        }
    )
    lines = [
        "Ni-Al Step 9 Final Report",
        "=" * 78,
        "",
        "1. Step 9 objective",
        "Select, retrieve, validate, and document candidate classical Ni-Al "
        "EAM potentials from authoritative sources and design the future "
        "Step 10 LAMMPS benchmark. No LAMMPS simulation, MACE calculation, "
        "or DFT calculation was executed.",
        "",
        "2. Completed gates",
        "Gate 1 preflight; Gate 2 source inspection; Gate 3 configuration; "
        "Gate 4 retrieval; Gate 5 setfl validation; Gate 6 scope review; "
        "Gate 7 LAMMPS availability; Gate 8 selection and ranking; Gate 9 "
        "Step 10 design; Gate 10 reporting and verification.",
        "",
        "3. Authoritative source policy",
        "HTTPS-only retrieval from the NIST Interatomic Potentials "
        "Repository (www.ctcms.nist.gov) with redirect confinement, "
        "timeouts, bounded retries, and a size cap; OpenKIM and the "
        "original publications as cross-reference provenance; unverified "
        "sources rejected.",
        "",
        "4-6. Candidates, provenance, and intended scope",
    ]
    for key in config.candidate_order:
        bundle = validated[key]
        spec = bundle.spec
        lines.extend(
            [
                f"- {key} ({spec.role}): {spec.citation} DOI {spec.doi}.",
                f"  NIST: {spec.repository_identity} / "
                f"{spec.implementation_identity}.",
                f"  Scope: {spec.fitting_scope}",
            ]
        )
    lines.extend(
        [
            "",
            "7-9. Files, fingerprints, and setfl validation",
        ]
    )
    for key in config.candidate_order:
        bundle = validated[key]
        paths = candidate_bundle_paths(config, key)
        setfl = bundle.setfl
        lines.extend(
            [
                f"- {key}:",
                f"  raw: {relative_path(paths.raw_potential, config.project_root)}",
                "  processed: "
                + relative_path(paths.processed_potential, config.project_root),
                f"  sha256: {bundle.raw_sha256} "
                f"({bundle.raw_size_bytes} bytes; raw==processed PASS)",
                "  release notes: "
                + (
                    relative_path(
                        paths.raw_release_notes, config.project_root
                    )
                    if bundle.release_notes_available
                    else "not available"
                ),
                f"  setfl: VALID; elements {' '.join(setfl.elements)}; "
                f"Nrho={setfl.nrho}; drho={setfl.drho:.10g}; Nr={setfl.nr}; "
                f"dr={setfl.dr:.10g}; cutoff={setfl.cutoff_A:.10g} A; "
                f"{setfl.total_tabulated_values} tabulated values complete",
            ]
        )
    lines.extend(
        [
            "",
            "10. Element mapping",
            "Project convention: LAMMPS atom type 1 = Al, atom type 2 = Ni. "
            "Planned command for every eam/alloy file: 'pair_style "
            "eam/alloy' with 'pair_coeff * * <file> Al Ni'. The names after "
            "the filename map atom types to elements and need not match the "
            "internal header order textually; LAMMPS resolves the named "
            "elements. Per-pair pair_coeff lines and pair_style hybrid EAM "
            "mixing are prohibited.",
            "",
            "11. Superseded implementation checks",
            "The 2004 ipr1 file (NiAl.eam.alloy; superseded 2020-12-14; "
            "non-zero isolated-atom energies) was neither downloaded nor "
            "accepted; its absence from every bundle directory was "
            "verified. The corrected ipr2 file explicitly sets F(rho=0)=0 "
            "so isolated atoms have zero energy. Formation energies remain "
            "defined from consistently relaxed bulk elemental references, "
            "not isolated atoms.",
            "",
            "12. OpenKIM cross-reference",
        ]
    )
    for key in config.candidate_order:
        spec = validated[key].spec
        lines.append(
            f"- {key}: family {spec.openkim_family_id}; documented current "
            f"extended ID {spec.openkim_extended_id}; driver family "
            f"{spec.openkim_model_driver_family}; species Al, Ni; cites the "
            "same publication as the NIST implementation. OpenKIM is "
            "provenance cross-reference only; no OpenKIM code was "
            "downloaded or executed and pair_style kim is not used. Step 10 "
            "will use the validated NIST eam/alloy files directly."
        )
    lines.extend(
        [
            "",
            "13. LAMMPS availability",
            f"Status: {lammps.status}",
            f"Detail: {lammps.detail}",
            f"Python 'lammps' package present: "
            f"{lammps.python_lammps_package_present}",
            "LAMMPS was not installed automatically; absence is a Step 10 "
            "readiness issue, not a Step 9 failure.",
            "",
            "14-15. Selection",
            "Primary: pun_mishin_2009 (broad binary Ni-Al fitting including "
            "intermetallic formation energies; intended for phases, "
            "interfaces, and mechanical behavior).",
            "Secondary sensitivity test: mishin_2004_ipr2 (gamma/gamma-prime "
            "focus; corrected implementation).",
            "Historical secondary: mishin_2002 (B2-NiAl focus; documented "
            "pure-element weakness recorded).",
            "The expected ranking was confirmed by source inspection and "
            "file validation; no deviation was required.",
            "",
            "16. Excluded or deferred categories",
            *(f"- {item}" for item in EXCLUDED_CATEGORIES),
            "",
            "17. Step 10 comparison design",
            "Each potential independently processes the same seven original "
            "selected structures (pure Al, pure Ni, five compounds): "
            "initial fixed-geometry values, Stage A fixed-cell "
            "minimization, then Stage B full-cell minimization at zero "
            "target pressure (fix box/relax, triclinic where needed); the "
            "primary classical result is full-cell relaxed. Static "
            "minimization only - no finite-temperature MD and no NPT "
            "substitute. Full plan: "
            + relative_path(
                step9_result_paths(config)["plan_json"], config.project_root
            ),
            "",
            "18. Formation-energy consistency rule",
            "mu_Al_P and mu_Ni_P come from potential P's own relaxed pure "
            "elements; E_form_P = (E_compound_P - N_Al*mu_Al_P - "
            "N_Ni*mu_Ni_P)/(N_Al+N_Ni) with actual cell counts and "
            "formula-unit cross-checks. MACE chemical potentials, MP "
            "elemental energies, and cross-potential mixing are prohibited; "
            "raw totals are never compared across compositions.",
            "",
            "19. Convergence design",
            "Force target 0.01 eV/angstrom (native metal units); stress "
            "target 0.0006241509 eV/angstrom^3 = "
            f"{stress_bar:.9f} bar, converted programmatically from exact "
            "SI definitions (1 eV/angstrom^3 = 1.602176634e6 bar). "
            "Convergence is verified from recorded maximum force and "
            "pressure components; minimizer termination alone is never "
            "scientific convergence.",
            "",
            "20. Structure-conversion design",
            "One documented ASE/pymatgen converter from the existing EXTXYZ "
            "structures to LAMMPS data files, preserving triclinic cells, "
            "scaled coordinates, ordering, species, and periodicity, with "
            "volume/composition/coordinate validation and recorded "
            "fingerprints. No production Step 10 data files were written in "
            "Step 9.",
            "",
            "21. Scientific limitations",
            "This stage selected and validated potential files; it produced "
            "no new energies, so nothing here demonstrates classical-"
            "potential accuracy. The 2002 candidate's documented "
            "pure-element weakness and the 2004 candidate's narrower scope "
            "are recorded hypotheses that Step 10 will measure.",
            "",
            "22. Output inventory",
            *(f"- {item}" for item in inventory),
            "",
            "23. Protected-file verification",
            "All protected Step 6, Step 7, and Step 8 inputs retained their "
            "recorded SHA-256, size, and modification-time fingerprints.",
            "",
            "24. Overall Step 9 status",
            f"OVERALL STEP 9 STATUS: {status}",
            "",
            "25. Exact next stage",
            NEXT_STAGE_TEXT,
            "Step 10 is not implemented by this workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def _replace_marked_section(
    path: Path, start_marker: str, end_marker: str, body: str
) -> None:
    """Append or replace one generated documentation section atomically."""

    original = path.read_text(encoding="utf-8")
    section = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    if start_marker in original or end_marker in original:
        if original.count(start_marker) != 1 or original.count(end_marker) != 1:
            raise Step9Error(
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


def _update_documentation(
    config: Step9Config,
    validated: Mapping[str, ValidatedCandidate],
    lammps: LammpsAvailability,
    status: str,
) -> None:
    """Document methodology and actual validated Step 9 results."""

    stress_bar = ev_per_A3_to_bar(0.0006241509)
    file_lines = [
        f"| {key} | {validated[key].spec.role} | "
        f"`{validated[key].spec.expected_filename}` | "
        f"{validated[key].setfl.cutoff_A:.4f} | "
        f"{' '.join(validated[key].setfl.elements)} | "
        f"`{validated[key].raw_sha256[:16]}...` |"
        for key in config.candidate_order
    ]
    readme_body = "\n".join(
        [
            "## Step 9 - Classical Ni-Al Potential Selection",
            "",
            "Step 9 selected, retrieved, and validated three documented "
            "binary Ni-Al EAM potentials from the NIST Interatomic "
            "Potentials Repository (HTTPS-only, redirect-confined, "
            "fingerprinted) and designed the Step 10 LAMMPS benchmark. No "
            "LAMMPS simulation, MACE calculation, or DFT calculation was "
            "executed, and no new scientific energy exists from this step.",
            "",
            "| Candidate | Role | Official file | Cutoff (A) | File element "
            "order | SHA-256 |",
            "|---|---|---|---:|---|---|",
            *file_lines,
            "",
            "**Primary: `pun_mishin_2009`** (Purja Pun & Mishin 2009, DOI "
            "10.1080/14786430903258184) - binary Ni-Al specific, built on "
            "established pure-element descriptions with the cross "
            "interaction fitted to B2-NiAl properties and ab initio "
            "intermetallic formation energies. Secondary: "
            "`mishin_2004_ipr2` (gamma/gamma-prime focus) - only the "
            "corrected ipr2 file `NiAl_Mishin_2004.eam.alloy` is accepted "
            "because the superseded ipr1 file has non-zero isolated-atom "
            "energies, while ipr2 sets F(rho=0)=0. Historical secondary: "
            "`mishin_2002` (B2-optimized; documented pure-element "
            "weakness).",
            "",
            "All files are `eam/alloy` setfl files validated array-by-array "
            "(headers, Al+Ni identity, grids, finiteness, exact counts, no "
            "trailing content) with byte-identical processed copies under "
            "`data/processed/interatomic_potentials/ni_al/`. Planned "
            "mapping: atom type 1 = Al, type 2 = Ni via `pair_coeff * * "
            "<file> Al Ni` (never per-pair pair_coeff and never "
            "pair_style hybrid mixing).",
            "",
            f"Local LAMMPS availability: **{lammps.status}** - "
            f"{lammps.detail} LAMMPS was not installed automatically; "
            "absence only affects Step 10 readiness.",
            "",
            "The Step 10 design (results/lammps_potential_selection/plans/) "
            "specifies: identical starting structures for every potential; "
            "two-stage static minimization (fixed-cell, then full-cell via "
            "`fix box/relax` at zero pressure); per-potential elemental "
            "references with the standard formation-energy equation; force "
            "target 0.01 eV/angstrom and stress target 0.0006241509 "
            f"eV/angstrom^3 = {stress_bar:.6f} bar (converted from exact SI "
            "definitions); and independent convergence verification.",
            "",
            "Commands:",
            "",
            "```bat",
            r".\.venv\Scripts\python.exe scripts\fetch_ni_al_classical_potentials.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\fetch_ni_al_classical_potentials.py --fetch",
            r".\.venv\Scripts\python.exe scripts\validate_ni_al_classical_potentials.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\design_ni_al_lammps_benchmark.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\design_ni_al_lammps_benchmark.py --design",
            r".\.venv\Scripts\python.exe scripts\run_step9_pipeline.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_step9_pipeline.py --execute",
            r".\.venv\Scripts\python.exe scripts\run_step9_pipeline.py --execute --resume",
            "```",
            "",
            f"Actual overall Step 9 status: **{status}**. The exact next "
            "stage is:",
            "",
            NEXT_STAGE_TEXT,
            "",
            "Step 10 is not implemented here.",
        ]
    )
    knowledge_body = "\n".join(
        [
            "## Step 9 Research-Log Entry (2026-07-28)",
            "",
            "A classical interatomic potential is an explicit analytic/"
            "tabulated energy model. In EAM the energy is a sum of pair "
            "terms plus an embedding energy F(rho) evaluated at the host "
            "electron density each atom sits in; the alloy cross "
            "interaction is the fitted Al-Ni pair function plus how each "
            "species' density enters the other's embedding. A setfl "
            "(`eam/alloy`) file tabulates F(rho), rho(r), and r*phi(r) for "
            "every element and pair on shared grids - unlike the older "
            "single-element funcfl (`eam`) format, it defines the "
            "cross-pair explicitly, which is why separate pure Al and pure "
            "Ni files can never be mixed safely: the Al-Ni interaction "
            "would be an undefined guess, not physics.",
            "",
            "Potential scope matters because a fit reproduces what it was "
            "trained on. 2009 (Purja Pun & Mishin) is the broad binary "
            "Ni-Al model (B2 properties plus ab initio intermetallic "
            "formation energies; interfaces and mechanics), 2004 (Mishin) "
            "targets gamma/gamma-prime (Ni3Al), and 2002 (Mishin-Mehl-"
            "Papaconstantopoulos) targets B2-NiAl with documented weaker "
            "pure-element behavior. That is why 2009 is primary and the "
            "others are sensitivity tests. The 2004 ipr1 file is rejected: "
            "its isolated-atom energies are non-zero, so bulk energies are "
            "correct but are not cohesive energies; ipr2 sets F(rho=0)=0. "
            "Our formation energies always use relaxed bulk elemental "
            "references, so each potential needs its own Al and Ni "
            "references and raw totals can never be compared across "
            "potentials - every model has its own arbitrary energy zero.",
            "",
            "LAMMPS is the engine that reads the potential file and "
            "evaluates it; the file is data, not code. Static minimization "
            "follows forces downhill to a zero-temperature local minimum "
            "(the analogue of Step 6's FIRE relaxations), while molecular "
            "dynamics integrates finite-temperature motion - Step 10's "
            "primary benchmark is static minimization only.",
            "",
            f"Actual Step 9 selection: primary pun_mishin_2009, secondary "
            "mishin_2004_ipr2, historical secondary mishin_2002; all three "
            "official NIST files validated array-complete with recorded "
            f"SHA-256; local LAMMPS status {lammps.status}. Overall Step 9 "
            f"status: **{status}**.",
            "",
            "Unanswered questions for Step 10: how large are each "
            "potential's formation-energy and volume errors against MP DFT "
            "and against MACE under identical structures and convergence "
            "targets; does the 2004 model's gamma-prime focus degrade "
            "Al-rich phases; how strong is the 2002 pure-element weakness "
            "in practice; and how do classical costs compare with MACE.",
        ]
    )
    _replace_marked_section(
        config.project_root / "docs" / "RESEARCH_LOG.md",
        README_MARKER_START,
        README_MARKER_END,
        readme_body,
    )
    _replace_marked_section(
        config.project_root / "docs" / "RESEARCH_LOG.md",
        KNOWLEDGE_MARKER_START,
        KNOWLEDGE_MARKER_END,
        knowledge_body,
    )


def execute_pipeline(
    config_path: Path, *, resume: bool, overwrite: bool
) -> tuple[Path, str]:
    """Execute every Step 9 gate sequentially and return the final report."""

    config = validate_pipeline(config_path)
    step8_snapshots = validate_step8_success(config)

    _fetch_stage(config, resume=resume, overwrite=overwrite)

    LOGGER.info("Gate 5/6: validating setfl bundles and scientific scope")
    validated = {
        key: validate_candidate_bundle(config, key) for key in CANDIDATE_ORDER
    }
    LOGGER.info("Gate 7: inspecting local LAMMPS availability")
    lammps = inspect_lammps_availability(config)
    LOGGER.info("LAMMPS availability: %s", lammps.status)

    LOGGER.info("Gate 9: creating the Step 10 benchmark design plan")
    plan_paths = (
        step9_result_paths(config)["plan_json"],
        step9_result_paths(config)["plan_txt"],
    )
    plan_exists = all(path.is_file() for path in plan_paths)
    if plan_exists and resume:
        plan = read_strict_json(plan_paths[0], "Step 10 plan JSON")
        if plan.get("configuration_fingerprint_sha256") != config.fingerprint:
            raise Step9ResumeError(
                "Existing Step 10 plan was produced by a different "
                "configuration."
            )
        LOGGER.info("Reusing the validated existing Step 10 plan.")
    else:
        plan = _design.run_design(config, overwrite=overwrite)

    LOGGER.info("Gate 8/10: evaluation matrix, reports, and documentation")
    matrix = build_evaluation_matrix(config, validated, lammps)
    paths = step9_result_paths(config)
    result_targets = {
        name: paths[name]
        for name in (
            "candidates_csv",
            "candidates_json",
            "selection_report",
            "file_manifest",
            "final_report",
        )
    }
    existing_results = [
        path for path in result_targets.values() if path.exists()
    ]
    if existing_results and not (overwrite or resume):
        listing = "\n".join(
            f"  - {relative_path(path, config.project_root)}"
            for path in existing_results
        )
        raise Step9CollisionError(
            "Existing Step 9 result outputs were found; re-run with "
            "--overwrite after review:\n" + listing
        )
    status = "SUCCESS"
    matrix_document = {
        "schema_version": "1.0",
        "artifact_type": "ni_al_classical_potential_candidates",
        "project_step": "9",
        "generated_at_utc": utc_timestamp(),
        "configuration_fingerprint_sha256": config.fingerprint,
        "lammps_availability": lammps.status,
        "records": matrix,
        "ranking_rule": (
            "Qualitative, evidence-based ranking (1 Primary, 2 Secondary, "
            "3 Historical secondary, 4 Rejected, 5 Unavailable); no "
            "fabricated numeric scores."
        ),
    }
    manifest_document = {
        "schema_version": "1.0",
        "artifact_type": "ni_al_potential_file_manifest",
        "project_step": "9",
        "generated_at_utc": utc_timestamp(),
        "configuration_fingerprint_sha256": config.fingerprint,
        "files": [
            {
                "candidate_key": key,
                "raw_path": relative_path(
                    candidate_bundle_paths(config, key).raw_potential,
                    config.project_root,
                ),
                "processed_path": relative_path(
                    candidate_bundle_paths(config, key).processed_potential,
                    config.project_root,
                ),
                "sha256": validated[key].raw_sha256,
                "size_bytes": validated[key].raw_size_bytes,
                "release_notes_path": (
                    relative_path(
                        candidate_bundle_paths(config, key).raw_release_notes,
                        config.project_root,
                    )
                    if validated[key].release_notes_available
                    else None
                ),
                "release_notes_sha256": validated[key].release_notes_sha256,
                "source_metadata_path": relative_path(
                    candidate_bundle_paths(config, key).source_metadata,
                    config.project_root,
                ),
                "retrieval_manifest_path": relative_path(
                    candidate_bundle_paths(config, key).retrieval_manifest,
                    config.project_root,
                ),
            }
            for key in CANDIDATE_ORDER
        ],
    }
    selection_report = _selection_report_text(config, validated, lammps)
    final_report = _final_report_text(
        config, validated, lammps, plan, status
    )
    for directory in step9_directories(config):
        directory.mkdir(parents=True, exist_ok=True)
    root = config.result_root
    with tempfile.TemporaryDirectory(
        prefix=".step9-results-staging-", dir=root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for target, payload in (
            (result_targets["candidates_csv"], _matrix_csv_bytes(matrix)),
            (
                result_targets["candidates_json"],
                write_strict_json_bytes(matrix_document),
            ),
            (
                result_targets["file_manifest"],
                write_strict_json_bytes(manifest_document),
            ),
            (
                result_targets["selection_report"],
                selection_report.encode("utf-8"),
            ),
            (result_targets["final_report"], final_report.encode("utf-8")),
        ):
            staged = stage_path(staging_root, root, target)
            staged.write_bytes(payload)
            staged_by_final[target] = staged

        def final_validator() -> None:
            verify_snapshots(step8_snapshots)
            reread = read_strict_json(
                result_targets["candidates_json"], "published matrix JSON"
            )
            if len(reread.get("records", ())) != len(CANDIDATE_ORDER):
                raise Step9Error("Published matrix JSON is incomplete.")

        publish_files_transactionally(
            config.project_root,
            root,
            staged_by_final,
            overwrite=overwrite or bool(existing_results and resume),
            final_validator=final_validator,
        )

    _update_documentation(config, validated, lammps, status)
    verify_snapshots(step8_snapshots)
    for key in CANDIDATE_ORDER:
        validate_candidate_bundle(config, key)

    report = result_targets["final_report"]
    print("=" * 78)
    print("STEP 9 PIPELINE EXECUTION COMPLETED")
    print("=" * 78)
    print(f"Overall Step 9 status: {status}")
    print("Primary: pun_mishin_2009; Secondary: mishin_2004_ipr2; "
          "Historical secondary: mishin_2002")
    print(f"LAMMPS availability: {lammps.status}")
    print(f"Final report: {relative_path(report, config.project_root)}")
    print(f"Exact next stage: {NEXT_STAGE_TEXT}")
    print("Step 10 is not implemented by this workflow.")
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
    except (Step9Error, Step7Error) as exc:
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

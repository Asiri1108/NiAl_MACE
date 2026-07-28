"""Run the Ni-Al MACE Step 7 gates in a controlled sequence.

The orchestrator imports and calls controlled Python functions rather than
constructing shell command strings.  Validation-only mode performs every
static gate without network retrieval, without loading MACE, without
creating an optimizer, without calculating any physical property, and
without writing scientific outputs.

Execution order:

1.  Preflight (Step 6 sources, configuration, installed APIs, key presence).
2.  Fetch the Al and Ni elemental references from Materials Project.
3.  Validate the retrieved references (structure, provenance, symmetry).
4.  Load MACE once and calculate the initial Al and Ni single points.
5.  Relax Al and Ni independently with full-cell FIRE/FrechetCellFilter.
6.  Validate elemental convergence and safety; publish elemental results.
7.  Validate the Step 6 compound energies and extract chemical potentials.
8.  Calculate initial and relaxed formation energies and the selected-set
    lower convex envelope; publish tables, reports, and figures.
9.  Update documentation from actual results and write the final report.
10. Verify every protected file fingerprint.

Step 8 is deliberately not implemented here.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import calculate_ni_al_mace_formation_energies as _formation
import fetch_ni_al_elemental_references as _fetch
import run_ni_al_mace_elemental_references as _elemental
from step7_utils import (
    ELEMENT_ORDER,
    NI_MAGNETIC_LIMITATION,
    PHASE_ORDER,
    SELECTED_SET_LIMITATION,
    Step7ApiError,
    Step7CollisionError,
    Step7Config,
    Step7ConfigurationError,
    Step7Error,
    Step7ResumeError,
    atomic_write_text,
    elemental_combined_paths,
    elemental_output_paths,
    final_report_path,
    formation_output_paths,
    installed_step7_versions,
    load_step7_config,
    load_validated_elemental_results,
    read_strict_json,
    relative_path,
    selected_elemental_paths,
    utc_timestamp,
    validate_frechet_cell_filter_api,
    validate_selected_elemental_structure,
    validate_step6_compound_sources,
    verify_snapshots,
)


LOGGER = logging.getLogger("ni_al_step7.pipeline")
DEFAULT_CONFIG = Path("configs/mace_formation_energy.json")
README_MARKER_START = "<!-- NI_AL_STEP7_START -->"
README_MARKER_END = "<!-- NI_AL_STEP7_END -->"
KNOWLEDGE_MARKER_START = "<!-- NI_AL_STEP7_KNOWLEDGE_START -->"
KNOWLEDGE_MARKER_END = "<!-- NI_AL_STEP7_KNOWLEDGE_END -->"
NEXT_STAGE_TEXT = (
    "Step 8 - Select and document candidate classical Ni-Al interatomic "
    "potentials for the future LAMMPS comparison."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Step 7 pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate or sequentially execute Ni-Al MACE Step 7."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 7 configuration path, repository-relative by default.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all static validation gates without retrieval or MACE.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute the complete Step 7 workflow.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, compatible, validated Step 7 outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace only Step 7 outputs.",
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
        raise Step7ConfigurationError(
            "--resume and --overwrite are execution options and cannot be "
            "used with --validate-only."
        )
    if args.resume and args.overwrite:
        raise Step7ConfigurationError(
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
        raise Step7Error(
            "Validation-only imported MACE unexpectedly: " + ", ".join(loaded_mace)
        )
    if loaded_fire:
        raise Step7Error(
            "Validation-only imported FIRE unexpectedly: " + ", ".join(loaded_fire)
        )


def validate_pipeline(config_path: Path) -> Step7Config:
    """Run every static Step 7 validation gate without side effects."""

    LOGGER.info("Gate 1: repository and Step 6 source preflight")
    config = load_step7_config(config_path)
    compounds, compound_snapshots = validate_step6_compound_sources(config)
    LOGGER.info(
        "Validated %d Step 6 compound records.", len(compounds)
    )

    LOGGER.info("Gate 2: Step 7 configuration and installed public APIs")
    versions = installed_step7_versions()
    ase_version = validate_frechet_cell_filter_api()
    LOGGER.info("Validated FrechetCellFilter public API in ASE %s", ase_version)

    api_key_line = "Yes (value not printed)"
    api_key_available = True
    try:
        _fetch._load_api_key_safely(config)
    except Step7Error as exc:
        api_key_available = False
        api_key_line = f"No - {exc}"

    selected_present = all(
        path.is_file()
        for element in ELEMENT_ORDER
        for path in selected_elemental_paths(config, element)
    )
    if selected_present:
        for element in ELEMENT_ORDER:
            validate_selected_elemental_structure(config, element)
        LOGGER.info("Existing selected elemental references validated.")

    planned: list[Path] = []
    for element in ELEMENT_ORDER:
        planned.extend(elemental_output_paths(config, element).all_paths())
    planned.extend(elemental_combined_paths(config).all_paths())
    planned.extend(formation_output_paths(config).all_paths())
    planned.append(final_report_path(config))
    collisions = [path for path in planned if path.exists()]

    verify_snapshots(compound_snapshots)
    _assert_heavy_modules_not_imported()

    print("=" * 78)
    print("STEP 7 PIPELINE VALIDATION")
    print("=" * 78)
    print("Step 6 source validation: SUCCESS")
    print("Configuration validation: SUCCESS")
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"API key variable available: {api_key_line}")
    print(
        "Selected elemental references present: "
        + ("Yes (validated)" if selected_present else "No (fetch stage pending)")
    )
    print(
        "Planned Step 7 output collisions: "
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
        + "; ".join(f"{name}={version}" for name, version in sorted(versions.items()))
    )
    print("Materials Project query executed: No")
    print("MACE loaded: No")
    print("Optimizer created: No")
    print("Relaxation executed: No")
    print("Formation energy calculated: No")
    print("Protected files modified: No")
    print(
        "Validation status: " + ("SUCCESS" if api_key_available else "FAILED")
    )
    print("=" * 78)
    if not api_key_available:
        raise Step7ApiError(
            "MP_API_KEY is missing; Step 7 execution cannot retrieve the "
            "elemental references. Configure the key locally in .env and "
            "re-run."
        )
    return config


def _fetch_stage(
    config: Step7Config, *, resume: bool, overwrite: bool
) -> None:
    """Gate 3: obtain the elemental references from Materials Project."""

    selected_present = all(
        path.is_file()
        for element in ELEMENT_ORDER
        for path in selected_elemental_paths(config, element)
    )
    if selected_present and resume:
        for element in ELEMENT_ORDER:
            validate_selected_elemental_structure(config, element)
        LOGGER.info("Gate 3: reusing validated existing elemental references.")
        return
    LOGGER.info("Gate 3: fetching elemental references from Materials Project")
    _fetch.run_fetch(config, ELEMENT_ORDER, overwrite=overwrite)


def _validate_existing_formation_outputs(config: Step7Config) -> None:
    """Validate that published analysis outputs match current inputs."""

    outputs = formation_output_paths(config)
    missing = [path for path in outputs.all_paths() if not path.is_file()]
    if missing:
        raise Step7ResumeError(
            "Formation-energy bundle is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    document = read_strict_json(
        outputs.table_json, "published formation-energy JSON"
    )
    if document.get("configuration_fingerprint_sha256") != config.fingerprint:
        raise Step7ResumeError(
            "Published formation-energy JSON was produced by a different "
            "configuration."
        )
    compounds, _snapshots = validate_step6_compound_sources(config)
    elemental, _elemental_snapshots = load_validated_elemental_results(config)
    from step7_utils import calculate_formation_energies

    records = calculate_formation_energies(config, compounds, elemental)
    published = {
        row.get("phase_key"): row
        for row in document.get("records", ())
        if isinstance(row, Mapping)
    }
    import math

    for record in records:
        row = published.get(record.phase_key)
        if row is None:
            raise Step7ResumeError(
                f"Published analysis lacks {record.phase_key}."
            )
        stored = row.get("relaxed_formation_energy_eV_per_atom")
        if not isinstance(stored, (int, float)) or not math.isclose(
            float(stored),
            record.relaxed_formation_energy_eV_per_atom,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise Step7ResumeError(
                f"Published relaxed formation energy for {record.phase_key} "
                "disagrees with the current validated inputs."
            )


def _formation_stage(
    config: Step7Config, *, resume: bool, overwrite: bool
) -> None:
    """Gates 7-9: chemical potentials, formation energies, and figures."""

    outputs = formation_output_paths(config)
    existing = [path for path in outputs.all_paths() if path.exists()]
    if existing and resume:
        LOGGER.info("Gate 8/9: validating the existing analysis bundle.")
        _validate_existing_formation_outputs(config)
        return
    LOGGER.info("Gate 8/9: calculating formation energies and figures")
    _formation.run_calculate(config, overwrite=overwrite)


def _replace_marked_section(
    path: Path, start_marker: str, end_marker: str, body: str
) -> None:
    """Append or replace one generated documentation section atomically."""

    original = path.read_text(encoding="utf-8")
    section = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    if start_marker in original or end_marker in original:
        if original.count(start_marker) != 1 or original.count(end_marker) != 1:
            raise Step7Error(
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


def _load_published_results(
    config: Step7Config,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Read the published machine-readable Step 7 results for reporting."""

    elemental_summary = read_strict_json(
        elemental_combined_paths(config).json,
        "elemental reference summary JSON",
    )
    formation_document = read_strict_json(
        formation_output_paths(config).table_json,
        "formation-energy JSON",
    )
    selected_metadata = {
        element: read_strict_json(
            selected_elemental_paths(config, element)[1],
            f"selected {element} metadata",
        )
        for element in ELEMENT_ORDER
    }
    return elemental_summary, formation_document, selected_metadata


def _phase_table_lines(formation_document: Mapping[str, Any]) -> list[str]:
    """Build a compact per-phase Markdown table from published results."""

    lines = [
        "| Phase | x_Ni | Initial E_f (eV/atom) | Relaxed E_f (eV/atom) | "
        "Relaxation effect (eV/atom) | Above envelope (eV/atom) | On envelope |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in formation_document.get("records", ()):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('phase_key')} | {row.get('ni_atomic_fraction'):.6f} | "
            f"{row.get('initial_formation_energy_eV_per_atom'):.9f} | "
            f"{row.get('relaxed_formation_energy_eV_per_atom'):.9f} | "
            f"{row.get('relaxation_effect_eV_per_atom'):.9f} | "
            f"{row.get('energy_above_selected_set_envelope_eV_per_atom'):.9f} | "
            f"{'yes' if row.get('on_selected_set_envelope') else 'no'} |"
        )
    return lines


def _write_final_report(
    config: Step7Config,
    *,
    resume: bool,
    overwrite: bool,
    compound_snapshots: Sequence[Any],
) -> tuple[Path, str]:
    """Create the authoritative Step 7 final report from published results."""

    elemental_summary, formation_document, selected_metadata = (
        _load_published_results(config)
    )
    elemental_records = {
        record.get("element"): record
        for record in elemental_summary.get("records", ())
        if isinstance(record, Mapping)
    }
    formation_records = [
        row
        for row in formation_document.get("records", ())
        if isinstance(row, Mapping)
    ]
    all_elements_converged = all(
        elemental_records.get(element, {}).get("status")
        in {"CONVERGED", "ALREADY_CONVERGED"}
        and elemental_records.get(element, {}).get("safety_status") == "PASS"
        for element in ELEMENT_ORDER
    )
    all_phases_valid = len(formation_records) == len(PHASE_ORDER) and all(
        row.get("safety_status") == "PASS"
        and row.get("compound_convergence_status")
        in {"CONVERGED", "ALREADY_CONVERGED"}
        for row in formation_records
    )
    status = "SUCCESS" if (all_elements_converged and all_phases_valid) else "PARTIAL"

    output_roots = (
        config.output.elemental_reference_root,
        config.output.formation_energy_root,
        config.input.elemental_selected_directory,
        config.input.elemental_raw_root,
    )
    inventory = sorted(
        relative_path(path, config.project_root)
        for root in output_roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    )
    report_target = final_report_path(config)
    inventory.append(relative_path(report_target, config.project_root))
    inventory = sorted(set(inventory))

    mu = formation_document.get("chemical_potentials_eV_per_atom", {})
    envelope = formation_document.get("selected_set_envelope", {})
    warnings_lines = sorted(
        {
            str(warning)
            for element in ELEMENT_ORDER
            for warning in read_strict_json(
                elemental_output_paths(config, element).checkpoint,
                f"{element} checkpoint",
            ).get("warnings", ())
        }
    )

    def element_block(element: str) -> list[str]:
        record = elemental_records.get(element, {})
        metadata = selected_metadata[element]
        return [
            f"{element}:",
            f"  Selected material ID: {record.get('material_id')}",
            "  API documents returned: "
            f"{metadata.get('api_documents_returned')}",
            "  Valid candidates: "
            f"{metadata.get('valid_candidate_count')}",
            f"  Selection rationale: {metadata.get('selection_rationale')}",
            f"  Atom count: {record.get('number_of_atoms')}",
            "  Initial energy per atom (eV/atom): "
            f"{record.get('initial_energy_per_atom_eV'):.17g}",
            "  Relaxed energy per atom (eV/atom): "
            f"{record.get('final_energy_per_atom_eV'):.17g}",
            "  Initial max |stress| (eV/angstrom^3): "
            f"{record.get('initial_maximum_absolute_stress_eV_per_A3'):.17g}",
            "  Final max |stress| (eV/angstrom^3): "
            f"{record.get('final_maximum_absolute_stress_eV_per_A3'):.17g}",
            f"  Optimizer steps: {record.get('optimizer_steps')}",
            f"  Convergence status: {record.get('status')}",
            f"  Volume change (percent): "
            f"{record.get('volume_change_percent'):.17g}",
            f"  Initial symmetry: {record.get('initial_space_group')}",
            f"  Final symmetry: {record.get('final_space_group')}",
        ]

    per_phase_initial = [
        f"- {row.get('phase_key')}: "
        f"{row.get('initial_formation_energy_eV_per_atom'):.9f} eV/atom"
        for row in formation_records
    ]
    per_phase_relaxed = [
        f"- {row.get('phase_key')}: "
        f"{row.get('relaxed_formation_energy_eV_per_atom'):.9f} eV/atom"
        for row in formation_records
    ]
    per_phase_effect = [
        f"- {row.get('phase_key')}: "
        f"{row.get('relaxation_effect_eV_per_atom'):.9f} eV/atom"
        for row in formation_records
    ]
    envelope_lines = [
        f"- ({vertex.get('ni_fraction'):.6f}, "
        f"{vertex.get('formation_energy_eV_per_atom'):.9f})"
        for vertex in envelope.get("vertices", ())
        if isinstance(vertex, Mapping)
    ]
    membership_lines = [
        f"- {row.get('phase_key')}: above envelope "
        f"{row.get('energy_above_selected_set_envelope_eV_per_atom'):.9f} "
        "eV/atom; on envelope: "
        f"{'yes' if row.get('on_selected_set_envelope') else 'no'}"
        for row in formation_records
    ]

    text = "\n".join(
        [
            "Ni-Al MACE Step 7 Final Report",
            "=" * 78,
            "",
            "1. Step 7 objective",
            "Establish MACE-consistent pure Al and pure Ni reference states "
            "with full-cell relaxation, then calculate MACE-consistent "
            "formation energies for the five selected Ni-Al phases.",
            "",
            "2. Completed gates",
            "Gate 1 preflight; Gate 2 configuration; Gate 3 elemental "
            "retrieval; Gate 4 structure validation; Gate 5 elemental single "
            "points; Gate 6 elemental full-cell relaxations; Gate 7 chemical "
            "potentials; Gate 8 formation energies; Gate 9 selected-set "
            "analysis and figures; Gate 10 reporting and verification.",
            "",
            "3. Materials Project retrieval details",
            "Endpoint: mp_api.client.MPRester.materials.summary.search; "
            "queried per element with deprecated=false. Materials Project "
            "supplied crystal structures and provenance only; its DFT "
            "energies were never used as MACE reference energies.",
            "",
            "4-5. Selected elemental structures",
            *element_block("Al"),
            *element_block("Ni"),
            "",
            "6. Materials Project database version",
            "; ".join(
                f"{element}="
                f"{selected_metadata[element].get('materials_project_database_version')}"
                for element in ELEMENT_ORDER
            ),
            "",
            "7. MACE model settings",
            f"{config.model.name} {config.model.value}; device="
            f"{config.model.device}; dtype={config.model.default_dtype}; "
            f"dispersion={str(config.model.dispersion).lower()}",
            "",
            "8. Elemental relaxation settings",
            "FIRE on ase.filters.FrechetCellFilter; force threshold 0.01 "
            "eV/angstrom; raw six-component stress threshold 0.0006241509 "
            "eV/angstrom^3; maximum 1000 steps; trajectory interval 1; "
            "hydrostatic_strain=false; constant_volume=false; zero external "
            "pressure. Identical to the Step 6 full-cell criteria.",
            "",
            "9-12. Al and Ni initial/relaxed values, convergence, symmetry",
            "See section 4-5 blocks above; both references were processed "
            "independently from their original selected structures.",
            "",
            "13. Chemical potentials",
            f"mu_Al_MACE = {mu.get('mu_Al_MACE'):.17g} eV/atom",
            f"mu_Ni_MACE = {mu.get('mu_Ni_MACE'):.17g} eV/atom",
            "",
            "14. Formation-energy equation",
            "E_f[eV/atom] = (E_compound_total - N_Al*mu_Al_MACE "
            "- N_Ni*mu_Ni_MACE) / (N_Al + N_Ni), applied with the actual "
            "cell composition and validated against the formula-unit route "
            "at 1e-12 eV/atom.",
            "",
            "15. Per-phase initial formation energies (diagnostic)",
            *per_phase_initial,
            "",
            "16. Per-phase relaxed formation energies (primary)",
            *per_phase_relaxed,
            "",
            "17. Relaxation effects (relaxed minus initial)",
            *per_phase_effect,
            "",
            "18. Selected-set lower-envelope analysis",
            "Envelope vertices (x_Ni, E_f eV/atom):",
            *envelope_lines,
            "Membership:",
            *membership_lines,
            *(f"- {line}" for line in SELECTED_SET_LIMITATION),
            "",
            "19. Warnings",
            *(
                [f"- {line}" for line in warnings_lines]
                if warnings_lines
                else ["- None recorded."]
            ),
            "",
            "20. Magnetic limitation for Ni",
            *(f"- {line}" for line in NI_MAGNETIC_LIMITATION),
            "",
            "21. Scientific limitations",
            "These are MACE-consistent results on the configured potential "
            "only. They do not show agreement with DFT or experiment, do not "
            "form a complete phase diagram, do not prove MACE accuracy, and "
            "do not decide whether fine-tuning is necessary. Raw total "
            "energies were never used to rank phases across compositions.",
            "",
            "22. Output inventory",
            *(f"- {item}" for item in inventory),
            "",
            "23. Protected-file verification",
            "All protected Step 5 and Step 6 inputs, the selected Ni-Al "
            "structures, and the Step 7 configuration retained their "
            "recorded SHA-256, size, and modification-time fingerprints.",
            "",
            "24. Overall Step 7 status",
            f"OVERALL STEP 7 STATUS: {status}",
            "",
            "25. Exact next stage",
            NEXT_STAGE_TEXT,
            "Step 8 is not implemented by this workflow.",
            "",
        ]
    )
    if resume and report_target.is_file():
        existing = report_target.read_text(encoding="utf-8")
        required = (
            "Ni-Al MACE Step 7 Final Report",
            f"OVERALL STEP 7 STATUS: {status}",
            NEXT_STAGE_TEXT,
        )
        missing = [item for item in required if item not in existing]
        if missing:
            raise Step7ResumeError(
                "Existing Step 7 final report is inconsistent; missing: "
                + " | ".join(missing)
            )
        return report_target, status
    atomic_write_text(report_target, text, overwrite=overwrite)
    verify_snapshots(compound_snapshots)
    return report_target, status


def _update_documentation(config: Step7Config, status: str) -> None:
    """Document methodology and actual validated Step 7 results."""

    _elemental_summary, formation_document, selected_metadata = (
        _load_published_results(config)
    )
    mu = formation_document.get("chemical_potentials_eV_per_atom", {})
    table = _phase_table_lines(formation_document)
    ids = ", ".join(
        f"{element}: {selected_metadata[element].get('material_id')}"
        for element in ELEMENT_ORDER
    )
    database_version = "; ".join(
        f"{element}="
        f"{selected_metadata[element].get('materials_project_database_version')}"
        for element in ELEMENT_ORDER
    )
    readme_body = "\n".join(
        [
            "## Step 7 - MACE Elemental References and Formation Energies",
            "",
            "Step 7 retrieves the stable FCC pure Al and pure Ni reference "
            "structures from Materials Project (structures and provenance "
            "only - never DFT energies), relaxes both independently with the "
            "exact Step 6 full-cell criteria (FIRE + FrechetCellFilter; "
            "max force <= 0.01 eV/angstrom; max |raw ASE stress| <= "
            "0.0006241509 eV/angstrom^3; up to 1000 steps), and defines the "
            "chemical potentials `mu_X_MACE = relaxed total energy / atoms`.",
            "",
            f"Selected structures - {ids}. "
            f"Materials Project database version: {database_version}.",
            "",
            "The MACE-consistent formation energy per atom is",
            "",
            "```text",
            "E_f = (E_compound_total - N_Al*mu_Al_MACE - N_Ni*mu_Ni_MACE)"
            " / (N_Al + N_Ni)",
            "```",
            "",
            "applied with the actual cell composition (validated against the "
            "formula-unit route at 1e-12 eV/atom). The primary result uses "
            "full-cell relaxed compound and elemental energies; the clearly "
            "separated diagnostic uses initial fixed-geometry single points "
            "on both sides. Initial and relaxed states are never mixed.",
            "",
            "Chemical potentials (this executed run): "
            f"mu_Al_MACE = {mu.get('mu_Al_MACE'):.9f} eV/atom; "
            f"mu_Ni_MACE = {mu.get('mu_Ni_MACE'):.9f} eV/atom.",
            "",
            *table,
            "",
            "The selected-set lower convex envelope uses only pure Al, the "
            "five selected compounds, and pure Ni. It is not the complete "
            "Ni-Al convex hull, not Materials Project energy above hull, and "
            "not a phase-diagram or experimental-stability claim. Untested "
            "compositions may lie below it. Ni is magnetic in DFT "
            "descriptions; the structural MACE workflow has no explicit "
            "spin input, so the Ni reference is MACE-consistent, not a "
            "controlled magnetic DFT reference.",
            "",
            "Implementation: `scripts/step7_utils.py`, "
            "`scripts/fetch_ni_al_elemental_references.py`, "
            "`scripts/run_ni_al_mace_elemental_references.py`, "
            "`scripts/calculate_ni_al_mace_formation_energies.py`, and "
            "`scripts/run_step7_pipeline.py`, with settings in "
            "`configs/mace_formation_energy.json`. Outputs are under "
            "`results/mace_elemental_references/` and "
            "`results/mace_formation_energy/`; the authoritative report is "
            "`results/mace_formation_energy/reports/"
            "ni_al_step7_final_report.txt`.",
            "",
            "Commands:",
            "",
            "```bat",
            r".\.venv\Scripts\python.exe scripts\fetch_ni_al_elemental_references.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\fetch_ni_al_elemental_references.py --fetch",
            r".\.venv\Scripts\python.exe scripts\run_ni_al_mace_elemental_references.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_ni_al_mace_elemental_references.py --execute",
            r".\.venv\Scripts\python.exe scripts\calculate_ni_al_mace_formation_energies.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\calculate_ni_al_mace_formation_energies.py --calculate",
            r".\.venv\Scripts\python.exe scripts\run_step7_pipeline.py --validate-only",
            r".\.venv\Scripts\python.exe scripts\run_step7_pipeline.py --execute",
            r".\.venv\Scripts\python.exe scripts\run_step7_pipeline.py --execute --resume",
            "```",
            "",
            f"Actual overall Step 7 status: **{status}**. These are "
            "MACE-consistent results only; no DFT was performed, no "
            "MACE-versus-DFT formation-energy comparison was made, and no "
            "accuracy or fine-tuning conclusion is drawn. The exact next "
            "stage is:",
            "",
            NEXT_STAGE_TEXT,
            "",
            "Step 8 is not implemented here.",
        ]
    )
    knowledge_body = "\n".join(
        [
            "## Step 7 Research-Log Entry (2026-07-28)",
            "",
            "Pure elemental crystals are required because a formation energy "
            "compares a compound against the elements in their reference "
            "crystalline states; isolated atoms would measure atomization "
            "energy instead, which is a different quantity with much larger "
            "magnitudes. Every energy entering the formula must come from "
            "the same model, precision, and convergence convention - mixing "
            "MACE and DFT energies would make the difference meaningless.",
            "",
            "The elemental chemical potential `mu_X_MACE` is the relaxed "
            "MACE total energy per atom of the pure crystal. The formation "
            "energy per atom subtracts composition-weighted chemical "
            "potentials from the compound energy and divides by the total "
            "atom count. Formula-unit counting (x, y per Al_x Ni_y) and "
            "simulation-cell counting (N_Al, N_Ni per cell) must agree after "
            "handling the number of formula units; Step 7 validates the two "
            "routes against each other at 1e-12 eV/atom.",
            "",
            "The initial-versus-relaxed distinction matters: the initial "
            "diagnostic uses fixed DFT geometries on the MACE surface, while "
            "the primary result uses MACE-relaxed geometries on both sides. "
            "Raw total energies across compositions can never be ranked "
            "directly because each composition has a different reference "
            "scale. The selected-set envelope is not a complete convex hull: "
            "only seven points were considered, and untested compositions "
            "may lie below it.",
            "",
            f"Actual results - mu_Al_MACE = {mu.get('mu_Al_MACE'):.9f} "
            f"eV/atom; mu_Ni_MACE = {mu.get('mu_Ni_MACE'):.9f} eV/atom "
            f"({ids}; database version {database_version}).",
            "",
            *table,
            "",
            "Ni magnetic limitation: Ni is magnetic in DFT descriptions; "
            "the structural MACE workflow exposes no user-controlled spin "
            "input, so the Ni reference is the configured pretrained MACE "
            "model's energy for the selected crystal - a MACE-consistent "
            "reference, not a controlled magnetic DFT reference. No Ni "
            "magnetic moment was invented.",
            "",
            "Unanswered questions for Step 8: which classical Ni-Al "
            "potentials should enter the LAMMPS comparison; how MACE and "
            "classical formation energies, lattice constants, and relaxed "
            "structures compare under identical conventions; whether "
            "observed MACE-versus-reference differences are systematic; and "
            "whether fine-tuning is justified.",
            "",
            f"Overall Step 7 status: **{status}**.",
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
    """Execute every Step 7 gate sequentially and return the final report."""

    config = validate_pipeline(config_path)
    compounds, compound_snapshots = validate_step6_compound_sources(config)

    _fetch_stage(config, resume=resume, overwrite=overwrite)

    LOGGER.info("Gate 4: validating the retrieved elemental references")
    for element in ELEMENT_ORDER:
        validate_selected_elemental_structure(config, element)

    LOGGER.info(
        "Gates 5-6: elemental single points and full-cell relaxations"
    )
    summary = _elemental.execute_elements(
        config, ELEMENT_ORDER, overwrite=overwrite, resume=resume
    )
    if summary.overall_status != "SUCCESS":
        raise Step7Error(
            "An elemental reference did not converge safely; primary "
            "formation energies must not be calculated from it. Step 7 is "
            f"{summary.overall_status}."
        )

    LOGGER.info("Gate 7: validating references and chemical potentials")
    elemental_results, _snapshots = load_validated_elemental_results(config)
    from step7_utils import extract_chemical_potentials

    potentials = extract_chemical_potentials(config, elemental_results)
    LOGGER.info(
        "mu_Al_MACE=%.9f eV/atom; mu_Ni_MACE=%.9f eV/atom",
        potentials["Al"],
        potentials["Ni"],
    )

    _formation_stage(config, resume=resume, overwrite=overwrite)

    LOGGER.info("Gate 10: final report, documentation, and verification")
    report, status = _write_final_report(
        config,
        resume=resume,
        overwrite=overwrite,
        compound_snapshots=compound_snapshots,
    )
    _update_documentation(config, status)
    verify_snapshots(compound_snapshots)
    for element in ELEMENT_ORDER:
        validate_selected_elemental_structure(config, element)

    print("=" * 78)
    print("STEP 7 PIPELINE EXECUTION COMPLETED")
    print("=" * 78)
    print(f"Overall Step 7 status: {status}")
    print(f"mu_Al_MACE = {potentials['Al']:.12f} eV/atom")
    print(f"mu_Ni_MACE = {potentials['Ni']:.12f} eV/atom")
    print(f"Final report: {relative_path(report, config.project_root)}")
    print(f"Exact next stage: {NEXT_STAGE_TEXT}")
    print("Step 8 is not implemented by this workflow.")
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
    except Step7Error as exc:
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

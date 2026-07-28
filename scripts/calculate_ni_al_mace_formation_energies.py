"""Calculate MACE-consistent Ni-Al formation energies from validated results.

This analysis command never loads MACE, never reruns any relaxation, never
calls Materials Project, performs no DFT, and never modifies a source
structure.  It consumes only:

* the published Step 7 elemental-reference results (relaxed and initial
  MACE energies for pure Al and pure Ni); and
* the validated Step 6 full-cell compound energies plus the Step 5
  fixed-geometry compound energies.

The primary result subtracts full-cell relaxed elemental references from
full-cell relaxed compound energies.  A clearly separated diagnostic
subtracts initial single-point elemental references from initial
fixed-geometry compound energies.  Initial and relaxed states are never
mixed.  The selected-set lower convex envelope is an incomplete-set
construction and is never presented as a complete Ni-Al convex hull.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7_utils import (
    ELEMENT_ORDER,
    PHASE_ORDER,
    SCHEMA_VERSION,
    SELECTED_SET_LIMITATION,
    NI_MAGNETIC_LIMITATION,
    CompoundRecord,
    ElementalReferenceRecord,
    FormationEnergyRecord,
    FormationOutputPaths,
    Step7CollisionError,
    Step7Config,
    Step7ConfigurationError,
    Step7Error,
    assert_mace_not_imported,
    calculate_formation_energies,
    extract_chemical_potentials,
    formation_directories,
    formation_output_paths,
    installed_step7_versions,
    lazy_pyplot,
    load_step7_config,
    load_validated_elemental_results,
    lower_convex_envelope,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    save_figure,
    selected_elemental_paths,
    stage_path,
    utc_timestamp,
    validate_step6_compound_sources,
    verify_snapshots,
    write_strict_json_bytes,
)


LOGGER = logging.getLogger("ni_al_step7.formation_energies")
DEFAULT_CONFIG = Path("configs/mace_formation_energy.json")

TABLE_FIELDNAMES: tuple[str, ...] = (
    "phase_key",
    "material_id",
    "reduced_formula",
    "cell_al_count",
    "cell_ni_count",
    "cell_atom_count",
    "formula_unit_al",
    "formula_unit_ni",
    "formula_units_in_cell",
    "ni_atomic_fraction",
    "initial_compound_total_energy_eV",
    "relaxed_compound_total_energy_eV",
    "initial_al_reference_eV_per_atom",
    "relaxed_al_reference_eV_per_atom",
    "initial_ni_reference_eV_per_atom",
    "relaxed_ni_reference_eV_per_atom",
    "initial_formation_energy_eV_per_atom",
    "relaxed_formation_energy_eV_per_atom",
    "relaxation_effect_eV_per_atom",
    "selected_set_envelope_eV_per_atom",
    "energy_above_selected_set_envelope_eV_per_atom",
    "on_selected_set_envelope",
    "compound_convergence_status",
    "elemental_convergence_status",
    "safety_status",
    "compound_checkpoint_path",
    "selected_structure_path",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the formation-energy analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate MACE-consistent Ni-Al formation energies from the "
            "validated Step 6 and Step 7 machine-readable results."
        )
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
        help="Validate all input references without writing analysis results.",
    )
    action.add_argument(
        "--calculate",
        action="store_true",
        help="Calculate and publish the Step 7 formation-energy outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 7 formation-energy analysis outputs.",
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


def _database_versions(config: Step7Config) -> dict[str, Any]:
    """Read the recorded Materials Project database versions per element."""

    versions: dict[str, Any] = {}
    for element in ELEMENT_ORDER:
        _structure_path, metadata_path = selected_elemental_paths(config, element)
        metadata = read_strict_json(metadata_path, f"selected {element} metadata")
        versions[element] = metadata.get("materials_project_database_version")
    return versions


def _table_row(record: FormationEnergyRecord, config: Step7Config,
               elemental: Mapping[str, ElementalReferenceRecord]) -> dict[str, Any]:
    """Serialize one formation-energy record to a flat table row."""

    return {
        "phase_key": record.phase_key,
        "material_id": record.material_id,
        "reduced_formula": record.reduced_formula,
        "cell_al_count": record.cell_al_count,
        "cell_ni_count": record.cell_ni_count,
        "cell_atom_count": record.cell_atom_count,
        "formula_unit_al": record.formula_unit_al,
        "formula_unit_ni": record.formula_unit_ni,
        "formula_units_in_cell": record.formula_units_in_cell,
        "ni_atomic_fraction": record.ni_fraction,
        "initial_compound_total_energy_eV": (
            record.initial_compound_total_energy_eV
        ),
        "relaxed_compound_total_energy_eV": (
            record.relaxed_compound_total_energy_eV
        ),
        "initial_al_reference_eV_per_atom": (
            elemental["Al"].initial_energy_per_atom_eV
        ),
        "relaxed_al_reference_eV_per_atom": (
            elemental["Al"].final_energy_per_atom_eV
        ),
        "initial_ni_reference_eV_per_atom": (
            elemental["Ni"].initial_energy_per_atom_eV
        ),
        "relaxed_ni_reference_eV_per_atom": (
            elemental["Ni"].final_energy_per_atom_eV
        ),
        "initial_formation_energy_eV_per_atom": (
            record.initial_formation_energy_eV_per_atom
        ),
        "relaxed_formation_energy_eV_per_atom": (
            record.relaxed_formation_energy_eV_per_atom
        ),
        "relaxation_effect_eV_per_atom": record.relaxation_effect_eV_per_atom,
        "selected_set_envelope_eV_per_atom": (
            record.envelope_energy_eV_per_atom
        ),
        "energy_above_selected_set_envelope_eV_per_atom": (
            record.energy_above_envelope_eV_per_atom
        ),
        "on_selected_set_envelope": record.on_selected_set_envelope,
        "compound_convergence_status": record.compound_convergence_status,
        "elemental_convergence_status": record.elemental_convergence_status,
        "safety_status": record.safety_status,
        "compound_checkpoint_path": relative_path(
            record.checkpoint_path, config.project_root
        ),
        "selected_structure_path": relative_path(
            record.selected_structure_path, config.project_root
        ),
    }


def _json_document(
    config: Step7Config,
    records: Sequence[FormationEnergyRecord],
    elemental: Mapping[str, ElementalReferenceRecord],
    potentials: Mapping[str, float],
    envelope: Sequence[tuple[float, float]],
    database_versions: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable formation-energy document."""

    versions = installed_step7_versions()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_formation_energies",
        "project_step": "7",
        "generated_at_utc": utc_timestamp(),
        "configuration_path": relative_path(
            config.config_path, config.project_root
        ),
        "configuration_fingerprint_sha256": config.fingerprint,
        "model": config.model.to_json(),
        "package_versions": dict(versions),
        "materials_project_database_version": dict(database_versions),
        "units": {
            "total_energy": "eV",
            "energy_per_atom": "eV/atom",
            "formation_energy": "eV/atom",
            "composition": "Ni atomic fraction (dimensionless)",
        },
        "formation_energy_equation": (
            "formation_energy_eV_per_atom = (compound_total_energy_eV "
            "- N_Al * mu_Al_MACE - N_Ni * mu_Ni_MACE) / (N_Al + N_Ni), "
            "using the actual cell composition; the formula-unit route was "
            "validated to agree within 1e-12 eV/atom."
        ),
        "elemental_references": {
            element: {
                "material_id": elemental[element].material_id,
                "atom_count": elemental[element].atom_count,
                "initial_energy_per_atom_eV": (
                    elemental[element].initial_energy_per_atom_eV
                ),
                "relaxed_energy_per_atom_eV": (
                    elemental[element].final_energy_per_atom_eV
                ),
                "convergence_status": elemental[element].convergence_status,
                "optimizer_steps": elemental[element].optimizer_steps,
                "checkpoint": relative_path(
                    elemental[element].checkpoint_path, config.project_root
                ),
            }
            for element in ELEMENT_ORDER
        },
        "chemical_potentials_eV_per_atom": {
            "mu_Al_MACE": potentials["Al"],
            "mu_Ni_MACE": potentials["Ni"],
        },
        "selected_set_envelope": {
            "description": (
                "Selected-set lower convex envelope over pure Al, the five "
                "selected compounds, and pure Ni only."
            ),
            "tolerance_eV_per_atom": (
                config.analysis.envelope_tolerance_eV_per_atom
            ),
            "vertices": [
                {"ni_fraction": x, "formation_energy_eV_per_atom": y}
                for x, y in envelope
            ],
            "limitations": list(SELECTED_SET_LIMITATION),
        },
        "records": [
            _table_row(record, config, elemental) for record in records
        ],
        "magnetic_limitation_Ni": list(NI_MAGNETIC_LIMITATION),
        "scientific_limitations": [
            *SELECTED_SET_LIMITATION,
            "No comparison against Materials Project DFT formation energies "
            "was performed in this step.",
            "MACE and DFT energies were never mixed.",
            "Raw total energies were never ranked across compositions.",
        ],
    }


def _report_text(
    config: Step7Config,
    records: Sequence[FormationEnergyRecord],
    elemental: Mapping[str, ElementalReferenceRecord],
    potentials: Mapping[str, float],
    envelope: Sequence[tuple[float, float]],
    database_versions: Mapping[str, Any],
) -> str:
    """Build the human-readable formation-energy report."""

    lines = [
        "Step 7 - MACE-Consistent Ni-Al Formation Energies",
        "=" * 76,
        "",
        "1. Objective",
        "------------",
        "Calculate MACE-consistent formation energies for the five selected "
        "Ni-Al phases using MACE-relaxed pure Al and pure Ni reference "
        "states from the same model and convergence criteria.",
        "",
        "2. Model and provenance",
        "-----------------------",
        f"Model: {config.model.name} {config.model.value}; "
        f"device={config.model.device}; dtype={config.model.default_dtype}; "
        f"dispersion={str(config.model.dispersion).lower()}",
        f"Configuration SHA-256: {config.fingerprint}",
        "Materials Project database version (structure provenance only): "
        + "; ".join(
            f"{element}={database_versions.get(element)}"
            for element in ELEMENT_ORDER
        ),
        "",
        "3. Elemental chemical potentials (MACE, relaxed)",
        "------------------------------------------------",
    ]
    for element in ELEMENT_ORDER:
        record = elemental[element]
        lines.extend(
            [
                f"mu_{element}_MACE = {potentials[element]:.17g} eV/atom",
                f"  source: {record.material_id}; {record.atom_count} atom(s); "
                f"{record.convergence_status}; "
                f"{record.optimizer_steps} optimizer step(s)",
                f"  initial single-point energy per atom: "
                f"{record.initial_energy_per_atom_eV:.17g} eV/atom",
            ]
        )
    lines.extend(
        [
            "",
            "4. Formation-energy equation",
            "----------------------------",
            "formation_energy_eV_per_atom =",
            "  (compound_total_energy_eV",
            "   - N_Al * mu_Al_MACE_eV_per_atom",
            "   - N_Ni * mu_Ni_MACE_eV_per_atom) / (N_Al + N_Ni)",
            "The actual cell composition was used; the formula-unit route "
            "was validated to agree within 1e-12 eV/atom.",
            "",
            "5. Per-phase results (eV/atom)",
            "------------------------------",
            "phase | x_Ni | initial E_f | relaxed E_f | relaxation effect | "
            "above envelope | on envelope",
        ]
    )
    for record in records:
        lines.append(
            f"{record.phase_key} | {record.ni_fraction:.6f} | "
            f"{record.initial_formation_energy_eV_per_atom:.9f} | "
            f"{record.relaxed_formation_energy_eV_per_atom:.9f} | "
            f"{record.relaxation_effect_eV_per_atom:.9f} | "
            f"{record.energy_above_envelope_eV_per_atom:.9f} | "
            f"{'yes' if record.on_selected_set_envelope else 'no'}"
        )
    lines.extend(
        [
            "",
            "The initial values are a clearly labeled fixed-geometry "
            "diagnostic: initial compound single points minus initial "
            "elemental single points. The relaxed values are the primary "
            "result: full-cell relaxed compounds minus full-cell relaxed "
            "elemental references. Initial and relaxed states were never "
            "mixed.",
            "",
            "6. Selected-set lower convex envelope",
            "-------------------------------------",
            "Vertices (Ni fraction, formation energy eV/atom):",
        ]
    )
    for x, y in envelope:
        lines.append(f"  ({x:.6f}, {y:.9f})")
    lines.extend(
        [
            f"Envelope membership tolerance: "
            f"{config.analysis.envelope_tolerance_eV_per_atom:g} eV/atom",
            "",
            "IMPORTANT LIMITATIONS:",
            *(f"- {line}" for line in SELECTED_SET_LIMITATION),
            "",
            "7. Lower-energy trend within the selected set",
            "---------------------------------------------",
            (
                "Within only the selected five compounds and the two "
                "elemental endpoints, the relaxed MACE formation energies "
                "order as listed above; this ordering is a statement about "
                "the configured MACE potential on this incomplete selected "
                "set, not a complete phase diagram, not a DFT result, and "
                "not an experimental stability proof."
            ),
            "",
            "8. Magnetic limitation for Ni",
            "-----------------------------",
            *(f"- {line}" for line in NI_MAGNETIC_LIMITATION),
            "",
            "9. Scientific boundaries",
            "------------------------",
            "- No DFT calculation was performed.",
            "- No comparison against Materials Project DFT formation "
            "energies was performed in this step.",
            "- MACE and DFT energies were never mixed.",
            "- Raw total energies were never ranked across compositions.",
            "- No conclusion about MACE accuracy or fine-tuning necessity "
            "is made here.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_formation_figures(
    config: Step7Config,
    records: Sequence[FormationEnergyRecord],
    envelope: Sequence[tuple[float, float]],
    targets: Mapping[str, Path],
) -> None:
    """Render the three formation-energy figures into staged paths."""

    plt = lazy_pyplot()
    phases = [record.phase_key for record in records]
    fractions = [record.ni_fraction for record in records]
    relaxed = [
        record.relaxed_formation_energy_eV_per_atom for record in records
    ]
    initial = [
        record.initial_formation_energy_eV_per_atom for record in records
    ]

    # Figure 1: relaxed formation energy versus Ni fraction.
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.scatter(
        [0.0, 1.0],
        [0.0, 0.0],
        marker="s",
        s=60,
        color="#555555",
        zorder=3,
        label="elemental endpoints (Al, Ni; E_f = 0 by definition)",
    )
    axis.scatter(
        fractions,
        relaxed,
        marker="o",
        s=55,
        color="#1f77b4",
        zorder=3,
        label="relaxed MACE formation energy",
    )
    for phase, x, y in zip(phases, fractions, relaxed):
        axis.annotate(
            phase,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    axis.set_xlabel("Ni atomic fraction x in Al(1-x)Ni(x)")
    axis.set_ylabel("Formation energy (eV/atom)")
    axis.set_title(
        "MACE-consistent relaxed formation energies (selected phases only)"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["relaxed"], plt)

    # Figure 2: initial diagnostic versus relaxed primary result.
    import numpy as np

    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    x_positions = np.arange(len(phases), dtype=float)
    width = 0.36
    axis.bar(
        x_positions - width / 2.0,
        initial,
        width=width,
        color="#4c78a8",
        label="initial fixed-geometry diagnostic",
    )
    axis.bar(
        x_positions + width / 2.0,
        relaxed,
        width=width,
        color="#f58518",
        label="relaxed primary result",
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Formation energy (eV/atom)")
    axis.set_title("MACE formation energies: initial diagnostic versus relaxed")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["initial_vs_relaxed"], plt)

    # Figure 3: the selected-set lower convex envelope.
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.axhline(0.0, color="black", linewidth=0.7)
    envelope_x = [x for x, _y in envelope]
    envelope_y = [y for _x, y in envelope]
    axis.plot(
        envelope_x,
        envelope_y,
        color="#2ca02c",
        linewidth=1.6,
        marker="D",
        markersize=5,
        label="selected-set lower convex envelope",
        zorder=2,
    )
    axis.scatter(
        fractions,
        relaxed,
        marker="o",
        s=55,
        color="#1f77b4",
        zorder=3,
        label="relaxed MACE formation energies",
    )
    axis.scatter(
        [0.0, 1.0],
        [0.0, 0.0],
        marker="s",
        s=60,
        color="#555555",
        zorder=3,
        label="elemental endpoints (E_f = 0)",
    )
    for phase, x, y in zip(phases, fractions, relaxed):
        axis.annotate(
            phase,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    axis.set_xlabel("Ni atomic fraction x in Al(1-x)Ni(x)")
    axis.set_ylabel("Formation energy (eV/atom)")
    axis.set_title("Selected-set lower convex envelope (incomplete set)")
    axis.text(
        0.5,
        0.02,
        "Selected-set construction only; NOT the complete Ni-Al convex hull.",
        transform=axis.transAxes,
        fontsize=8,
        ha="center",
        color="#666666",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["envelope"], plt)


def load_and_validate_inputs(
    config: Step7Config,
) -> tuple[
    tuple[CompoundRecord, ...],
    Mapping[str, ElementalReferenceRecord],
    dict[str, float],
    dict[str, Any],
]:
    """Validate every analysis input without writing anything."""

    assert_mace_not_imported()
    compounds, _compound_snapshots = validate_step6_compound_sources(config)
    elemental, _elemental_snapshots = load_validated_elemental_results(config)
    potentials = extract_chemical_potentials(config, elemental)
    database_versions = _database_versions(config)
    assert_mace_not_imported()
    return compounds, elemental, potentials, database_versions


def run_validate_only(config: Step7Config) -> None:
    """Validate all input references and report the analysis plan."""

    compounds, elemental, potentials, database_versions = (
        load_and_validate_inputs(config)
    )
    outputs = formation_output_paths(config)
    collisions = [path for path in outputs.all_paths() if path.exists()]
    print("=" * 78)
    print("STEP 7 FORMATION-ENERGY INPUT VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(
        "Validated compounds: "
        + ", ".join(record.phase_key for record in compounds)
    )
    for element in ELEMENT_ORDER:
        record = elemental[element]
        print(
            f"{element} reference: {record.material_id}; "
            f"{record.convergence_status}; "
            f"mu={potentials[element]:.12f} eV/atom"
        )
    print(
        "Materials Project database version: "
        + "; ".join(
            f"{element}={database_versions.get(element)}"
            for element in ELEMENT_ORDER
        )
    )
    print(
        "Existing analysis outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("MACE loaded: No")
    print("Relaxations rerun: No")
    print("Materials Project queried: No")
    print("Analysis results written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_calculate(
    config: Step7Config, *, overwrite: bool
) -> tuple[FormationEnergyRecord, ...]:
    """Calculate, stage, validate, and publish the analysis bundle."""

    compounds, elemental, potentials, database_versions = (
        load_and_validate_inputs(config)
    )
    records = calculate_formation_energies(config, compounds, elemental)
    envelope = lower_convex_envelope(
        [(0.0, 0.0)]
        + [
            (record.ni_fraction, record.relaxed_formation_energy_eV_per_atom)
            for record in records
        ]
        + [(1.0, 0.0)]
    )
    outputs = formation_output_paths(config)
    if not overwrite:
        collisions = [path for path in outputs.all_paths() if path.exists()]
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step7CollisionError(
                "Existing Step 7 formation-energy outputs were found; re-run "
                "with --overwrite after review:\n" + listing
            )
    for directory in formation_directories(config):
        directory.mkdir(parents=True, exist_ok=True)

    rows = [_table_row(record, config, elemental) for record in records]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=TABLE_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in TABLE_FIELDNAMES})
    csv_bytes = buffer.getvalue().encode("utf-8")
    document = _json_document(
        config, records, elemental, potentials, envelope, database_versions
    )
    report = _report_text(
        config, records, elemental, potentials, envelope, database_versions
    )

    root = config.output.formation_energy_root
    with tempfile.TemporaryDirectory(
        prefix=".step7-formation-staging-", dir=root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for target, payload in (
            (outputs.table_csv, csv_bytes),
            (outputs.table_json, write_strict_json_bytes(document)),
            (outputs.report, report.encode("utf-8")),
        ):
            staged = stage_path(staging_root, root, target)
            staged.write_bytes(payload)
            staged_by_final[target] = staged
        figure_targets = {
            "relaxed": stage_path(staging_root, root, outputs.relaxed_figure),
            "initial_vs_relaxed": stage_path(
                staging_root, root, outputs.initial_vs_relaxed_figure
            ),
            "envelope": stage_path(staging_root, root, outputs.envelope_figure),
        }
        _render_formation_figures(config, records, envelope, figure_targets)
        staged_by_final[outputs.relaxed_figure] = figure_targets["relaxed"]
        staged_by_final[outputs.initial_vs_relaxed_figure] = figure_targets[
            "initial_vs_relaxed"
        ]
        staged_by_final[outputs.envelope_figure] = figure_targets["envelope"]

        staged_json = staged_by_final[outputs.table_json]
        published_document = read_strict_json(
            staged_json, "staged formation-energy JSON"
        )
        if len(published_document.get("records", ())) != len(PHASE_ORDER):
            raise Step7Error(
                "Staged formation-energy JSON does not contain all phases."
            )

        def final_validator() -> None:
            _compounds, compound_snapshots = validate_step6_compound_sources(
                config
            )
            verify_snapshots(compound_snapshots)
            reread = read_strict_json(
                outputs.table_json, "published formation-energy JSON"
            )
            if (
                reread.get("configuration_fingerprint_sha256")
                != config.fingerprint
            ):
                raise Step7Error(
                    "Published formation-energy JSON fingerprint mismatch."
                )

        publish_files_transactionally(
            config.project_root,
            root,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )
    assert_mace_not_imported()
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.calculate:
        LOGGER.error("--overwrite is allowed only with --calculate.")
        return 1
    try:
        config = load_step7_config(args.config)
        if args.validate_only:
            run_validate_only(config)
            return 0
        records = run_calculate(config, overwrite=args.overwrite)
        print("=" * 78)
        print("STEP 7 FORMATION-ENERGY CALCULATION COMPLETED")
        print("=" * 78)
        for record in records:
            print(
                f"{record.phase_key}: initial="
                f"{record.initial_formation_energy_eV_per_atom:.9f}; relaxed="
                f"{record.relaxed_formation_energy_eV_per_atom:.9f}; effect="
                f"{record.relaxation_effect_eV_per_atom:.9f} eV/atom; "
                f"on_envelope={record.on_selected_set_envelope}"
            )
        print("Selected-set envelope is NOT a complete Ni-Al convex hull.")
        print("=" * 78)
        return 0
    except Step7Error as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial analysis bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

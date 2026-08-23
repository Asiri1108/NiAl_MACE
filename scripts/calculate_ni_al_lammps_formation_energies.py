"""Calculate potential-specific elemental references and formation energies.

This analysis command reads only the published Step 10 LAMMPS state
checkpoints.  It never runs LAMMPS, never loads MACE, never queries
Materials Project, and never performs DFT.  For every potential P and
every calculation state s (initial, fixed-cell, full-cell), the elemental
chemical potentials come exclusively from potential P's own pure Al and
pure Ni results in the same state s:

``E_form_P,s = (E_compound_P,s - N_Al*mu_Al_P,s - N_Ni*mu_Ni_P,s) / N``

States are never mixed, no cross-potential reference is ever used, and
raw total energies are never ranked across compositions.  The per-
potential selected-set lower convex envelope (full-cell energies only) is
an incomplete-set construction and is never presented as a complete Ni-Al
convex hull.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7_utils import (
    Step7Error,
    envelope_energy,
    lower_convex_envelope,
    publish_files_transactionally,
    relative_path,
    stage_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step10_utils import (
    COMPOUND_ORDER,
    ELEMENT_KEYS,
    POTENTIAL_ORDER,
    SCHEMA_VERSION,
    STATE_ORDER,
    STRUCTURE_ORDER,
    Step10CalculationError,
    Step10CollisionError,
    Step10Config,
    Step10Error,
    load_step10_config,
    stage_checkpoint_path,
    validate_state_checkpoint,
    validate_step9_success,
)


LOGGER = logging.getLogger("ni_al_step10.formation")
DEFAULT_CONFIG = Path("configs/ni_al_lammps_benchmark.json")

SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "potential_key",
    "potential_role",
    "phase",
    "stage",
    "material_id",
    "atom_count",
    "al_count",
    "ni_count",
    "total_energy_eV",
    "energy_per_atom_eV",
    "maximum_force_eV_per_A",
    "maximum_absolute_pressure_bar",
    "maximum_absolute_stress_eV_per_A3",
    "volume_A3",
    "volume_per_atom_A3",
    "volume_change_percent_vs_original",
    "maximum_internal_displacement_A",
    "space_group",
    "minimizer_iterations_total",
    "force_evaluations_total",
    "convergence_status",
    "safety_status",
    "wall_time_seconds",
)

FORMATION_FIELDNAMES: tuple[str, ...] = (
    "potential_key",
    "potential_role",
    "phase",
    "material_id",
    "reduced_formula",
    "al_count",
    "ni_count",
    "atom_count",
    "ni_atomic_fraction",
    "initial_formation_energy_eV_per_atom",
    "fixed_cell_formation_energy_eV_per_atom",
    "full_cell_formation_energy_eV_per_atom",
    "atomic_relaxation_effect_eV_per_atom",
    "cell_relaxation_effect_eV_per_atom",
    "total_relaxation_effect_eV_per_atom",
    "envelope_energy_eV_per_atom",
    "energy_above_selected_set_envelope_eV_per_atom",
    "on_selected_set_envelope",
    "convergence_status",
    "safety_status",
    "compound_checkpoint",
    "al_reference_checkpoint",
    "ni_reference_checkpoint",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the formation-energy analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate potential-specific Ni-Al elemental references and "
            "formation energies from the published LAMMPS state checkpoints."
        )
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
        help="Validate LAMMPS relaxation bundles without writing analysis.",
    )
    action.add_argument(
        "--calculate",
        action="store_true",
        help="Calculate references and formation energies.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 10 formation-energy analysis outputs.",
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


def output_paths(config: Step10Config) -> dict[str, Path]:
    """Return the canonical formation-analysis targets."""

    tables = config.analysis_root / "tables"
    return {
        "summary_csv": tables / "ni_al_lammps_relaxation_summary.csv",
        "summary_json": tables / "ni_al_lammps_relaxation_summary.json",
        "failures_json": tables / "ni_al_lammps_relaxation_failures.json",
        "elemental_csv": tables / "ni_al_lammps_elemental_references.csv",
        "elemental_json": tables / "ni_al_lammps_elemental_references.json",
        "formation_csv": tables / "ni_al_lammps_formation_energies.csv",
        "formation_json": tables / "ni_al_lammps_formation_energies.json",
    }


def load_all_state_records(
    config: Step10Config,
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, Any]], list[dict[str, Any]]
]:
    """Load and strictly validate all 63 expected state checkpoints."""

    hashes, _snapshots = validate_step9_success(config)
    records: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for potential in POTENTIAL_ORDER:
        for phase in STRUCTURE_ORDER:
            for stage in STATE_ORDER:
                try:
                    records[(potential, phase, stage)] = (
                        validate_state_checkpoint(
                            config, potential, hashes[potential], phase, stage
                        )
                    )
                except Step10Error as exc:
                    failures.append(
                        {
                            "potential_key": potential,
                            "phase": phase,
                            "stage": stage,
                            "checkpoint": relative_path(
                                stage_checkpoint_path(
                                    config, potential, phase, stage
                                ),
                                config.project_root,
                            ),
                            "error": str(exc),
                        }
                    )
    return records, failures


def _summary_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one state record into a summary table row."""

    return {
        "potential_key": record["potential_key"],
        "potential_role": record["potential_role"],
        "phase": record["phase"],
        "stage": record["stage"],
        "material_id": record["material_id"],
        "atom_count": record["atom_count"],
        "al_count": record["al_count"],
        "ni_count": record["ni_count"],
        "total_energy_eV": record["total_energy_eV"],
        "energy_per_atom_eV": record["energy_per_atom_eV"],
        "maximum_force_eV_per_A": record["maximum_force_eV_per_A"],
        "maximum_absolute_pressure_bar": record[
            "maximum_absolute_pressure_bar"
        ],
        "maximum_absolute_stress_eV_per_A3": record[
            "maximum_absolute_stress_eV_per_A3"
        ],
        "volume_A3": record["volume_A3"],
        "volume_per_atom_A3": record["volume_per_atom_A3"],
        "volume_change_percent_vs_original": record[
            "volume_change_percent_vs_original"
        ],
        "maximum_internal_displacement_A": record[
            "maximum_internal_displacement_A"
        ],
        "space_group": (
            f"{record['symmetry']['space_group_symbol']} "
            f"({record['symmetry']['space_group_number']})"
        ),
        "minimizer_iterations_total": record["minimizer_iterations_total"],
        "force_evaluations_total": record["force_evaluations_total"],
        "convergence_status": record["convergence_status"],
        "safety_status": record["safety_status"],
        "wall_time_seconds": record["wall_time_seconds"],
    }


def calculate_formation_tables(
    config: Step10Config,
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build elemental-reference rows, formation rows, and envelopes."""

    try:
        from pymatgen.core import Composition
    except ImportError as exc:
        raise Step10CalculationError(f"pymatgen is unavailable: {exc}") from exc
    tolerance = config.arithmetic_tolerance_eV_per_atom
    elemental_rows: list[dict[str, Any]] = []
    formation_rows: list[dict[str, Any]] = []
    envelopes: dict[str, Any] = {}
    for potential in POTENTIAL_ORDER:
        role = config.potentials[potential].role
        references: dict[tuple[str, str], float] = {}
        for element in ELEMENT_KEYS:
            for stage in STATE_ORDER:
                record = records.get((potential, element, stage))
                if record is None:
                    raise Step10CalculationError(
                        f"{potential}/{element}/{stage}: elemental reference "
                        "state is missing; formation energies cannot be "
                        "calculated."
                    )
                count = record["al_count"] if element == "Al" else record["ni_count"]
                if count != record["atom_count"] or count <= 0:
                    raise Step10CalculationError(
                        f"{potential}/{element}/{stage}: elemental "
                        "composition is invalid."
                    )
                mu = float(record["total_energy_eV"]) / count
                references[(element, stage)] = mu
                elemental_rows.append(
                    {
                        "potential_key": potential,
                        "potential_role": role,
                        "element": element,
                        "stage": stage,
                        "material_id": record["material_id"],
                        "atom_count": record["atom_count"],
                        "total_energy_eV": record["total_energy_eV"],
                        "mu_eV_per_atom": mu,
                        "maximum_force_eV_per_A": record[
                            "maximum_force_eV_per_A"
                        ],
                        "maximum_absolute_pressure_bar": record[
                            "maximum_absolute_pressure_bar"
                        ],
                        "volume_per_atom_A3": record["volume_per_atom_A3"],
                        "space_group": (
                            f"{record['symmetry']['space_group_symbol']} "
                            f"({record['symmetry']['space_group_number']})"
                        ),
                        "convergence_status": record["convergence_status"],
                        "safety_status": record["safety_status"],
                        "checkpoint": relative_path(
                            stage_checkpoint_path(
                                config, potential, element, stage
                            ),
                            config.project_root,
                        ),
                    }
                )
        provisional: list[dict[str, Any]] = []
        for phase in COMPOUND_ORDER:
            composition = Composition(phase).reduced_composition
            counts = {
                str(species): int(round(amount))
                for species, amount in composition.get_el_amt_dict().items()
            }
            formula_al, formula_ni = counts["Al"], counts["Ni"]
            values: dict[str, float] = {}
            for stage in STATE_ORDER:
                record = records.get((potential, phase, stage))
                if record is None:
                    raise Step10CalculationError(
                        f"{potential}/{phase}/{stage}: compound state is "
                        "missing."
                    )
                al_count = int(record["al_count"])
                ni_count = int(record["ni_count"])
                atoms = al_count + ni_count
                formula_atoms = formula_al + formula_ni
                if atoms % formula_atoms != 0 or (
                    al_count * formula_ni != ni_count * formula_al
                ):
                    raise Step10CalculationError(
                        f"{potential}/{phase}: cell composition does not "
                        "match the reduced formula."
                    )
                units = atoms // formula_atoms
                mu_al = references[("Al", stage)]
                mu_ni = references[("Ni", stage)]
                energy = float(record["total_energy_eV"])
                cell_route = (
                    energy - al_count * mu_al - ni_count * mu_ni
                ) / atoms
                formula_route = (
                    energy / units - formula_al * mu_al - formula_ni * mu_ni
                ) / formula_atoms
                if not math.isclose(
                    cell_route, formula_route, abs_tol=tolerance, rel_tol=0.0
                ):
                    raise Step10CalculationError(
                        f"{potential}/{phase}/{stage}: formula-unit and "
                        "cell-count formation routes disagree."
                    )
                values[stage] = cell_route
            full_record = records[(potential, phase, "full_cell")]
            provisional.append(
                {
                    "phase": phase,
                    "material_id": full_record["material_id"],
                    "reduced_formula": composition.reduced_formula,
                    "al_count": int(full_record["al_count"]),
                    "ni_count": int(full_record["ni_count"]),
                    "atom_count": int(full_record["atom_count"]),
                    "ni_fraction": (
                        int(full_record["ni_count"])
                        / int(full_record["atom_count"])
                    ),
                    "values": values,
                    "convergence_status": full_record["convergence_status"],
                    "safety_status": full_record["safety_status"],
                }
            )
        try:
            envelope = lower_convex_envelope(
                [(0.0, 0.0)]
                + [
                    (entry["ni_fraction"], entry["values"]["full_cell"])
                    for entry in provisional
                ]
                + [(1.0, 0.0)]
            )
        except Step7Error as exc:
            raise Step10CalculationError(str(exc)) from exc
        envelopes[potential] = {
            "description": (
                "Selected-set lower convex envelope over pure Al, the five "
                "selected compounds, and pure Ni using full-cell formation "
                "energies from this potential only. Not a complete Ni-Al "
                "convex hull, not the Materials Project hull, and not a "
                "complete phase diagram; untested Ni-Al structures may lie "
                "below it."
            ),
            "vertices": [
                {"ni_fraction": x, "formation_energy_eV_per_atom": y}
                for x, y in envelope
            ],
        }
        for entry in provisional:
            envelope_value = envelope_energy(envelope, entry["ni_fraction"])
            above = entry["values"]["full_cell"] - envelope_value
            formation_rows.append(
                {
                    "potential_key": potential,
                    "potential_role": role,
                    "phase": entry["phase"],
                    "material_id": entry["material_id"],
                    "reduced_formula": entry["reduced_formula"],
                    "al_count": entry["al_count"],
                    "ni_count": entry["ni_count"],
                    "atom_count": entry["atom_count"],
                    "ni_atomic_fraction": entry["ni_fraction"],
                    "initial_formation_energy_eV_per_atom": (
                        entry["values"]["initial"]
                    ),
                    "fixed_cell_formation_energy_eV_per_atom": (
                        entry["values"]["fixed_cell"]
                    ),
                    "full_cell_formation_energy_eV_per_atom": (
                        entry["values"]["full_cell"]
                    ),
                    "atomic_relaxation_effect_eV_per_atom": (
                        entry["values"]["fixed_cell"]
                        - entry["values"]["initial"]
                    ),
                    "cell_relaxation_effect_eV_per_atom": (
                        entry["values"]["full_cell"]
                        - entry["values"]["fixed_cell"]
                    ),
                    "total_relaxation_effect_eV_per_atom": (
                        entry["values"]["full_cell"]
                        - entry["values"]["initial"]
                    ),
                    "envelope_energy_eV_per_atom": envelope_value,
                    "energy_above_selected_set_envelope_eV_per_atom": max(
                        above, 0.0
                    ),
                    "on_selected_set_envelope": (
                        abs(above) <= config.envelope_tolerance_eV_per_atom
                    ),
                    "convergence_status": entry["convergence_status"],
                    "safety_status": entry["safety_status"],
                    "compound_checkpoint": relative_path(
                        stage_checkpoint_path(
                            config, potential, entry["phase"], "full_cell"
                        ),
                        config.project_root,
                    ),
                    "al_reference_checkpoint": relative_path(
                        stage_checkpoint_path(config, potential, "Al", "full_cell"),
                        config.project_root,
                    ),
                    "ni_reference_checkpoint": relative_path(
                        stage_checkpoint_path(config, potential, "Ni", "full_cell"),
                        config.project_root,
                    ),
                }
            )
    return elemental_rows, formation_rows, envelopes


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> bytes:
    """Serialize rows to CSV bytes."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=tuple(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def run_validate_only(config: Step10Config) -> None:
    """Validate every relaxation bundle without writing analysis outputs."""

    records, failures = load_all_state_records(config)
    print("=" * 78)
    print("STEP 10 FORMATION-ENERGY INPUT VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Validated state checkpoints: {len(records)}/63")
    if failures:
        print("Missing or invalid states:")
        for failure in failures:
            print(
                f"  - {failure['potential_key']}/{failure['phase']}/"
                f"{failure['stage']}: {failure['error']}"
            )
    print("LAMMPS executed: No")
    print("Analysis written: No")
    print(
        "Validation status: "
        + ("SUCCESS" if not failures else "FAILED (incomplete states)")
    )
    print("=" * 78)
    if failures:
        raise Step10CalculationError(
            f"{len(failures)} state checkpoint(s) are missing or invalid; "
            "run the LAMMPS relaxations first."
        )


def run_calculate(config: Step10Config, *, overwrite: bool) -> dict[str, Any]:
    """Calculate, stage, and transactionally publish the analysis tables."""

    records, failures = load_all_state_records(config)
    if failures:
        raise Step10CalculationError(
            f"{len(failures)} state checkpoint(s) are missing or invalid; "
            "formation energies are not calculated from incomplete data."
        )
    summary_rows = [
        _summary_row(records[(potential, phase, stage)])
        for potential in POTENTIAL_ORDER
        for phase in STRUCTURE_ORDER
        for stage in STATE_ORDER
    ]
    elemental_rows, formation_rows, envelopes = calculate_formation_tables(
        config, records
    )
    targets = output_paths(config)
    if not overwrite:
        collisions = [path for path in targets.values() if path.exists()]
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step10CollisionError(
                "Existing formation-analysis outputs were found; re-run with "
                "--overwrite after review:\n" + listing
            )
    generated = utc_timestamp()
    summary_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_relaxation_summary",
        "project_step": "10",
        "generated_at_utc": generated,
        "configuration_fingerprint_sha256": config.fingerprint,
        "expected_rows": 63,
        "completed_rows": len(summary_rows),
        "records": summary_rows,
    }
    failures_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_relaxation_failures",
        "project_step": "10",
        "generated_at_utc": generated,
        "failures": failures,
        "note": "Missing or failed states are recorded here explicitly; no "
        "fabricated rows exist in the summary table.",
    }
    elemental_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_elemental_references",
        "project_step": "10",
        "generated_at_utc": generated,
        "configuration_fingerprint_sha256": config.fingerprint,
        "consistency_rule": (
            "Each potential uses only its own relaxed bulk pure Al and pure "
            "Ni references in the matching calculation state; MACE, "
            "Materials Project, cross-potential, and isolated-atom "
            "references are never used."
        ),
        "records": elemental_rows,
    }
    formation_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_formation_energies",
        "project_step": "10",
        "generated_at_utc": generated,
        "configuration_fingerprint_sha256": config.fingerprint,
        "formation_energy_equation": (
            "E_form_P = (E_compound_P - N_Al*mu_Al_P - N_Ni*mu_Ni_P) / "
            "(N_Al + N_Ni), evaluated separately for the initial, "
            "fixed-cell, and full-cell states with same-state references "
            "from the same potential; formula-unit and cell-count routes "
            "validated within 1e-12 eV/atom."
        ),
        "selected_set_envelopes": envelopes,
        "records": formation_rows,
    }
    elemental_fieldnames = tuple(elemental_rows[0].keys())
    root = config.analysis_root
    (root / "tables").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".step10-formation-", dir=root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for target, payload in (
            (
                targets["summary_csv"],
                _csv_bytes(summary_rows, SUMMARY_FIELDNAMES),
            ),
            (
                targets["summary_json"],
                write_strict_json_bytes(summary_document),
            ),
            (
                targets["failures_json"],
                write_strict_json_bytes(failures_document),
            ),
            (
                targets["elemental_csv"],
                _csv_bytes(elemental_rows, elemental_fieldnames),
            ),
            (
                targets["elemental_json"],
                write_strict_json_bytes(elemental_document),
            ),
            (
                targets["formation_csv"],
                _csv_bytes(formation_rows, FORMATION_FIELDNAMES),
            ),
            (
                targets["formation_json"],
                write_strict_json_bytes(formation_document),
            ),
        ):
            staged = stage_path(staging_root, root, target)
            staged.write_bytes(payload)
            staged_by_final[target] = staged
        publish_files_transactionally(
            config.project_root, root, staged_by_final, overwrite=overwrite
        )
    return formation_document


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.calculate:
        LOGGER.error("--overwrite is allowed only with --calculate.")
        return 1
    try:
        config = load_step10_config(args.config)
        if args.validate_only:
            run_validate_only(config)
            return 0
        document = run_calculate(config, overwrite=args.overwrite)
        print("=" * 78)
        print("STEP 10 FORMATION-ENERGY CALCULATION COMPLETED")
        print("=" * 78)
        for row in document["records"]:
            print(
                f"{row['potential_key']} {row['phase']}: full-cell E_f = "
                f"{row['full_cell_formation_energy_eV_per_atom']:.6f} eV/atom"
            )
        print("Selected-set envelopes are NOT complete Ni-Al convex hulls.")
        print("=" * 78)
        return 0
    except (Step10Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial analysis bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

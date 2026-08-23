"""Compare the LAMMPS classical-potential results with MACE and MP DFT.

This analysis command reads only published machine-readable sources: the
Step 10 LAMMPS formation-energy and relaxation tables, the Step 8 MACE
versus Materials Project benchmark, and the Step 10 state checkpoints.
It never runs LAMMPS, never loads MACE, never creates an optimizer, never
queries Materials Project, and never performs DFT.  Step 8 MACE results
are consumed unchanged.  The error convention throughout is
``signed_error = method - MP processed DFT`` in eV/atom; raw total
energies are never ranked across compositions, and the per-potential
selected-set envelopes are never called complete hulls.
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
    NI_MAGNETIC_LIMITATION,
    Step7Error,
    assert_mace_not_imported,
    lazy_pyplot,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    save_figure,
    stage_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step10_utils import (
    COMPOUND_ORDER,
    POTENTIAL_ORDER,
    SCHEMA_VERSION,
    STRUCTURE_ORDER,
    Step10CalculationError,
    Step10CollisionError,
    Step10Config,
    Step10Error,
    Step10InputError,
    load_step10_config,
    read_lammps_structure,
    stage_dir,
    load_source_structure,
    validate_step9_success,
)
from calculate_ni_al_lammps_formation_energies import (
    output_paths as formation_output_paths,
)


LOGGER = logging.getLogger("ni_al_step10.compare")
DEFAULT_CONFIG = Path("configs/ni_al_lammps_benchmark.json")
METHOD_ORDER: tuple[str, ...] = ("mace_mp0_small", *POTENTIAL_ORDER)
METHOD_LABELS: Mapping[str, str] = {
    "mace_mp0_small": "MACE-MP-0 Small",
    "pun_mishin_2009": "Pun-Mishin 2009 EAM",
    "mishin_2004_ipr2": "Mishin 2004 EAM (ipr2)",
    "mishin_2002": "Mishin 2002 EAM",
}
METHOD_COLORS: Mapping[str, str] = {
    "mp_dft": "#2ca02c",
    "mace_mp0_small": "#1f77b4",
    "pun_mishin_2009": "#d62728",
    "mishin_2004_ipr2": "#9467bd",
    "mishin_2002": "#8c564b",
}

ENERGY_FIELDNAMES: tuple[str, ...] = (
    "method",
    "method_label",
    "phase",
    "material_id",
    "ni_atomic_fraction",
    "mp_dft_formation_energy_eV_per_atom",
    "mp_energy_above_hull_eV_per_atom",
    "mp_is_stable",
    "formation_energy_eV_per_atom",
    "signed_error_vs_mp_eV_per_atom",
    "absolute_error_vs_mp_eV_per_atom",
    "on_selected_set_envelope",
)

STRUCTURAL_FIELDNAMES: tuple[str, ...] = (
    "method",
    "phase",
    "material_id",
    "mp_volume_per_atom_A3",
    "volume_per_atom_A3",
    "signed_volume_error_A3_per_atom",
    "volume_error_percent",
    "mp_space_group",
    "relaxed_space_group",
    "symmetry_agreement",
    "maximum_internal_displacement_A",
    "volume_change_percent_vs_original",
    "standardized_lattice_available",
    "standardized_lattice_note",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the comparison analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare LAMMPS classical-potential formation energies and "
            "structures against MACE and Materials Project DFT references."
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
        help="Validate all inputs without writing comparison outputs.",
    )
    action.add_argument(
        "--compare",
        action="store_true",
        help="Create the comparison tables, figures, and reports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 10 comparison outputs.",
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


def comparison_output_paths(config: Step10Config) -> dict[str, Path]:
    """Return the canonical comparison targets."""

    root = config.analysis_root
    figures = root / "figures"
    return {
        "energy_csv": root / "tables" / "ni_al_lammps_vs_mace_mp.csv",
        "energy_json": root / "tables" / "ni_al_lammps_vs_mace_mp.json",
        "structural_csv": (
            root / "tables" / "ni_al_lammps_structural_comparison.csv"
        ),
        "structural_json": (
            root / "tables" / "ni_al_lammps_structural_comparison.json"
        ),
        "runtime_csv": root / "tables" / "ni_al_lammps_runtime_summary.csv",
        "runtime_json": root / "tables" / "ni_al_lammps_runtime_summary.json",
        "checkpoint": root / "checkpoints" / "step10_benchmark_result.json",
        "report": root / "reports" / "ni_al_lammps_benchmark_report.txt",
        "fig_composition": (
            figures / "formation_energy_all_methods_vs_composition.png"
        ),
        "fig_error_by_phase": (
            figures / "formation_energy_error_by_phase_and_method.png"
        ),
        "fig_parity_2009": (
            figures / "formation_energy_parity_pun_mishin_2009.png"
        ),
        "fig_parity_2004": figures / "formation_energy_parity_mishin_2004.png",
        "fig_parity_2002": figures / "formation_energy_parity_mishin_2002.png",
        "fig_mae_rmse": figures / "formation_energy_mae_rmse_by_method.png",
        "fig_volume_error": (
            figures / "volume_percent_error_by_phase_and_method.png"
        ),
        "fig_volume_mae": figures / "volume_mae_by_method.png",
        "fig_runtime": figures / "runtime_by_potential.png",
        "fig_envelope_2009": (
            figures / "selected_set_envelope_pun_mishin_2009.png"
        ),
        "fig_envelope_2004": figures / "selected_set_envelope_mishin_2004.png",
        "fig_envelope_2002": figures / "selected_set_envelope_mishin_2002.png",
    }


def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-tie 1-based ranks without a SciPy dependency."""

    import numpy as np

    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    position = 0
    while position < len(array):
        end = position
        while (
            end + 1 < len(array)
            and array[order[end + 1]] == array[order[position]]
        ):
            end += 1
        for index in range(position, end + 1):
            ranks[order[index]] = (position + end) / 2.0 + 1.0
        position = end + 1
    return [float(value) for value in ranks]


def _method_statistics(
    phases: Sequence[str],
    method_values: Sequence[float],
    dft_values: Sequence[float],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Descriptive error statistics for one method versus MP DFT (n=5)."""

    import numpy as np

    method = np.asarray(method_values, dtype=float)
    dft = np.asarray(dft_values, dtype=float)
    signed = method - dft
    absolute = np.abs(signed)
    pearson = spearman = None
    if float(np.std(method)) > 0.0 and float(np.std(dft)) > 0.0:
        pearson = float(np.corrcoef(method, dft)[0, 1])
        method_ranks = np.asarray(_rankdata(method_values))
        dft_ranks = np.asarray(_rankdata(dft_values))
        if float(np.std(method_ranks)) > 0.0 and float(np.std(dft_ranks)) > 0.0:
            spearman = float(np.corrcoef(method_ranks, dft_ranks)[0, 1])
    method_order = [phases[i] for i in np.argsort(method, kind="stable")]
    dft_order = [phases[i] for i in np.argsort(dft, kind="stable")]
    pair_total = pair_agree = 0
    for first in range(len(phases)):
        for second in range(first + 1, len(phases)):
            pair_total += 1
            if math.copysign(1.0, method[second] - method[first]) == (
                math.copysign(1.0, dft[second] - dft[first])
            ):
                pair_agree += 1
    return {
        "sample_size": len(phases),
        "mean_signed_error_eV_per_atom": float(np.mean(signed)),
        "mean_absolute_error_eV_per_atom": float(np.mean(absolute)),
        "rmse_eV_per_atom": float(np.sqrt(np.mean(np.square(signed)))),
        "median_absolute_error_eV_per_atom": float(np.median(absolute)),
        "maximum_absolute_error_eV_per_atom": float(np.max(absolute)),
        "phase_with_maximum_absolute_error": phases[int(np.argmax(absolute))],
        "minimum_absolute_error_eV_per_atom": float(np.min(absolute)),
        "phase_with_minimum_absolute_error": phases[int(np.argmin(absolute))],
        "standard_deviation_of_signed_errors_eV_per_atom": float(
            np.std(signed)
        ),
        "pearson_correlation": pearson,
        "spearman_rank_correlation": spearman,
        "ranking_most_negative_first": method_order,
        "dft_ranking_most_negative_first": dft_order,
        "exact_ranking_agreement": method_order == dft_order,
        "pairwise_ordering_agreement": f"{pair_agree}/{pair_total}",
        "most_negative_phase": method_order[0],
        "error_threshold_counts": {
            f"{threshold:g}": int(np.sum(absolute <= threshold + 1e-15))
            for threshold in thresholds
        },
        "correlation_note": (
            "Exploratory only: five compounds cannot support strong "
            "statistical claims."
        ),
    }


def load_comparison_inputs(config: Step10Config) -> dict[str, Any]:
    """Load and cross-validate every comparison input source."""

    assert_mace_not_imported()
    validate_step9_success(config)
    formation = read_strict_json(
        formation_output_paths(config)["formation_json"],
        "Step 10 formation-energy table",
    )
    if formation.get("configuration_fingerprint_sha256") != config.fingerprint:
        raise Step10InputError(
            "Step 10 formation table was produced by a different "
            "configuration; rerun the formation-energy calculation."
        )
    summary = read_strict_json(
        formation_output_paths(config)["summary_json"],
        "Step 10 relaxation summary",
    )
    step8_energy = read_strict_json(
        config.comparison_sources["step8_energy_table"], "Step 8 energy table"
    )
    step8_structural = read_strict_json(
        config.comparison_sources["step8_structural_table"],
        "Step 8 structural table",
    )
    step6_summary = read_strict_json(
        config.comparison_sources["step6_full_cell_summary"],
        "Step 6 full-cell summary",
    )
    step8_rows = {
        str(row.get("phase_key")): row
        for row in step8_energy.get("records", ())
        if isinstance(row, Mapping)
    }
    step8_struct_rows = {
        str(row.get("phase_key")): row
        for row in step8_structural.get("records", ())
        if isinstance(row, Mapping)
    }
    for phase in COMPOUND_ORDER:
        if phase not in step8_rows or phase not in step8_struct_rows:
            raise Step10InputError(f"Step 8 records are missing {phase}.")
    formation_rows = {
        (str(row.get("potential_key")), str(row.get("phase"))): row
        for row in formation.get("records", ())
        if isinstance(row, Mapping)
    }
    for potential in POTENTIAL_ORDER:
        for phase in COMPOUND_ORDER:
            if (potential, phase) not in formation_rows:
                raise Step10InputError(
                    f"Step 10 formation records are missing "
                    f"{potential}/{phase}."
                )
    summary_rows = {
        (
            str(row.get("potential_key")),
            str(row.get("phase")),
            str(row.get("stage")),
        ): row
        for row in summary.get("records", ())
        if isinstance(row, Mapping)
    }
    if len(summary_rows) != 63:
        raise Step10InputError(
            f"Relaxation summary contains {len(summary_rows)} rows; 63 "
            "expected."
        )
    return {
        "formation": formation,
        "formation_rows": formation_rows,
        "summary_rows": summary_rows,
        "step8_energy": step8_energy,
        "step8_rows": step8_rows,
        "step8_struct_rows": step8_struct_rows,
        "step6_summary": step6_summary,
    }


def _standardized_lattice_comparison(
    config: Step10Config, potential: str, phase: str
) -> tuple[bool, str, dict[str, Any] | None]:
    """Compare standardized conventional lattices of LAMMPS versus MP."""

    try:
        from ase.io import read as ase_read
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:
        raise Step10CalculationError(f"pymatgen unavailable: {exc}") from exc
    mp_path = (
        config.project_root
        / "data"
        / "raw"
        / "materials_project"
        / "dft_benchmark"
        / phase
        / "structure.extxyz"
    )
    lammps_path = stage_dir(config, potential, phase, "full_cell") / "final.extxyz"
    try:
        mp_atoms = ase_read(mp_path, format="extxyz")
        lammps_atoms = ase_read(lammps_path, format="extxyz")
        analyzer_settings = {
            "symprec": config.symmetry_symprec_A,
            "angle_tolerance": config.symmetry_angle_tolerance_deg,
        }
        mp_standard = SpacegroupAnalyzer(
            AseAtomsAdaptor.get_structure(mp_atoms), **analyzer_settings
        ).get_conventional_standard_structure()
        lammps_standard = SpacegroupAnalyzer(
            AseAtomsAdaptor.get_structure(lammps_atoms), **analyzer_settings
        ).get_conventional_standard_structure()
        if (
            mp_standard.composition.reduced_composition
            != lammps_standard.composition.reduced_composition
            or len(mp_standard) != len(lammps_standard)
        ):
            return (
                False,
                "Standardized cells are incompatible; comparison skipped "
                "safely.",
                None,
            )
        return (
            True,
            "pymatgen conventional-standard structures compared "
            "(symprec 0.001 A, angle tolerance 5 deg).",
            {
                "mp_abc_A": [float(v) for v in mp_standard.lattice.abc],
                "lammps_abc_A": [
                    float(v) for v in lammps_standard.lattice.abc
                ],
                "abc_differences_A": [
                    float(a - b)
                    for a, b in zip(
                        lammps_standard.lattice.abc, mp_standard.lattice.abc
                    )
                ],
                "mp_angles_deg": [
                    float(v) for v in mp_standard.lattice.angles
                ],
                "lammps_angles_deg": [
                    float(v) for v in lammps_standard.lattice.angles
                ],
            },
        )
    except Exception as exc:
        return (
            False,
            f"Standardization failed ({type(exc).__name__}); comparison "
            "skipped safely.",
            None,
        )


def build_comparison(config: Step10Config, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Build every comparison table, statistic, and runtime summary."""

    import numpy as np

    step8_rows = inputs["step8_rows"]
    formation_rows = inputs["formation_rows"]
    summary_rows = inputs["summary_rows"]
    step8_struct_rows = inputs["step8_struct_rows"]
    phases = list(COMPOUND_ORDER)
    dft_values = [
        float(step8_rows[phase]["mp_formation_energy_eV_per_atom"])
        for phase in phases
    ]
    fractions = [
        float(step8_rows[phase]["ni_atomic_fraction"]) for phase in phases
    ]
    method_values: dict[str, list[float]] = {
        "mace_mp0_small": [
            float(step8_rows[phase]["mace_relaxed_formation_energy_eV_per_atom"])
            for phase in phases
        ]
    }
    energy_rows: list[dict[str, Any]] = []
    for phase, fraction, dft in zip(phases, fractions, dft_values):
        energy_rows.append(
            {
                "method": "mace_mp0_small",
                "method_label": METHOD_LABELS["mace_mp0_small"],
                "phase": phase,
                "material_id": step8_rows[phase]["material_id"],
                "ni_atomic_fraction": fraction,
                "mp_dft_formation_energy_eV_per_atom": dft,
                "mp_energy_above_hull_eV_per_atom": step8_rows[phase][
                    "mp_energy_above_hull_eV_per_atom"
                ],
                "mp_is_stable": step8_rows[phase]["mp_is_stable"],
                "formation_energy_eV_per_atom": method_values[
                    "mace_mp0_small"
                ][phases.index(phase)],
                "signed_error_vs_mp_eV_per_atom": step8_rows[phase][
                    "relaxed_signed_error_eV_per_atom"
                ],
                "absolute_error_vs_mp_eV_per_atom": step8_rows[phase][
                    "relaxed_absolute_error_eV_per_atom"
                ],
                "on_selected_set_envelope": step8_rows[phase][
                    "mace_on_selected_set_envelope"
                ],
            }
        )
    for potential in POTENTIAL_ORDER:
        values: list[float] = []
        for phase, fraction, dft in zip(phases, fractions, dft_values):
            row = formation_rows[(potential, phase)]
            value = float(row["full_cell_formation_energy_eV_per_atom"])
            values.append(value)
            energy_rows.append(
                {
                    "method": potential,
                    "method_label": METHOD_LABELS[potential],
                    "phase": phase,
                    "material_id": row["material_id"],
                    "ni_atomic_fraction": fraction,
                    "mp_dft_formation_energy_eV_per_atom": dft,
                    "mp_energy_above_hull_eV_per_atom": step8_rows[phase][
                        "mp_energy_above_hull_eV_per_atom"
                    ],
                    "mp_is_stable": step8_rows[phase]["mp_is_stable"],
                    "formation_energy_eV_per_atom": value,
                    "signed_error_vs_mp_eV_per_atom": value - dft,
                    "absolute_error_vs_mp_eV_per_atom": abs(value - dft),
                    "on_selected_set_envelope": row[
                        "on_selected_set_envelope"
                    ],
                }
            )
        method_values[potential] = values
    statistics = {
        method: _method_statistics(
            phases,
            method_values[method],
            dft_values,
            config.error_threshold_bins,
        )
        for method in METHOD_ORDER
    }
    mae_ranking = sorted(
        METHOD_ORDER,
        key=lambda method: statistics[method][
            "mean_absolute_error_eV_per_atom"
        ],
    )
    per_phase_best: dict[str, str] = {}
    for index, phase in enumerate(phases):
        per_phase_best[phase] = min(
            METHOD_ORDER,
            key=lambda method: abs(
                method_values[method][index] - dft_values[index]
            ),
        )

    structural_rows: list[dict[str, Any]] = []
    for potential in POTENTIAL_ORDER:
        for phase in phases:
            struct = step8_struct_rows[phase]
            summary_row = summary_rows[(potential, phase, "full_cell")]
            mp_volume = float(struct["mp_volume_per_atom_A3"])
            volume = float(summary_row["volume_per_atom_A3"])
            available, note, lattice = _standardized_lattice_comparison(
                config, potential, phase
            )
            # The Step 8 structural table stores pre-formatted space-group
            # strings such as 'Pm-3m (221)'.
            mp_sg = str(struct["mp_space_group"])
            structural_rows.append(
                {
                    "method": potential,
                    "phase": phase,
                    "material_id": summary_row["material_id"],
                    "mp_volume_per_atom_A3": mp_volume,
                    "volume_per_atom_A3": volume,
                    "signed_volume_error_A3_per_atom": volume - mp_volume,
                    "volume_error_percent": 100.0
                    * (volume - mp_volume)
                    / mp_volume,
                    "mp_space_group": mp_sg,
                    "relaxed_space_group": summary_row["space_group"],
                    "symmetry_agreement": (
                        summary_row["space_group"] == mp_sg
                    ),
                    "maximum_internal_displacement_A": summary_row[
                        "maximum_internal_displacement_A"
                    ],
                    "volume_change_percent_vs_original": summary_row[
                        "volume_change_percent_vs_original"
                    ],
                    "standardized_lattice_available": available,
                    "standardized_lattice_note": note,
                    "standardized_lattice": lattice,
                }
            )
    for phase in phases:
        struct = step8_struct_rows[phase]
        mp_volume = float(struct["mp_volume_per_atom_A3"])
        mace_volume = float(struct["mace_volume_per_atom_A3"])
        mp_sg = str(struct["mp_space_group"])
        mace_sg = str(struct["mace_space_group"])
        structural_rows.append(
            {
                "method": "mace_mp0_small",
                "phase": phase,
                "material_id": struct["material_id"],
                "mp_volume_per_atom_A3": mp_volume,
                "volume_per_atom_A3": mace_volume,
                "signed_volume_error_A3_per_atom": mace_volume - mp_volume,
                "volume_error_percent": 100.0
                * (mace_volume - mp_volume)
                / mp_volume,
                "mp_space_group": mp_sg,
                "relaxed_space_group": mace_sg,
                "symmetry_agreement": bool(struct["symmetry_preserved"]),
                "maximum_internal_displacement_A": None,
                "volume_change_percent_vs_original": None,
                "standardized_lattice_available": bool(
                    struct["lattice_comparison_available"]
                ),
                "standardized_lattice_note": "From the Step 8 benchmark.",
                "standardized_lattice": None,
            }
        )
    volume_statistics: dict[str, Any] = {}
    for method in METHOD_ORDER:
        percents = np.asarray(
            [
                row["volume_error_percent"]
                for row in structural_rows
                if row["method"] == method
            ],
            dtype=float,
        )
        agreements = sum(
            1
            for row in structural_rows
            if row["method"] == method and row["symmetry_agreement"]
        )
        all_positive = bool(np.all(percents > 0.0))
        all_negative = bool(np.all(percents < 0.0))
        volume_statistics[method] = {
            "mean_signed_volume_percent_error": float(np.mean(percents)),
            "mean_absolute_volume_percent_error": float(
                np.mean(np.abs(percents))
            ),
            "rmse_volume_percent_error": float(
                np.sqrt(np.mean(np.square(percents)))
            ),
            "maximum_absolute_volume_percent_error": float(
                np.max(np.abs(percents))
            ),
            "symmetry_agreement_count": agreements,
            "systematic_direction": (
                "expansion"
                if all_positive
                else "contraction" if all_negative else "mixed"
            ),
            "systematic_note": (
                "Single-signed volume errors in this five-compound sample "
                "suggest a systematic trend without claiming universal "
                "behavior."
                if all_positive or all_negative
                else "No single-signed volume trend in this sample."
            ),
        }

    runtime_rows: list[dict[str, Any]] = []
    for potential in POTENTIAL_ORDER:
        walls = [
            float(summary_rows[(potential, phase, stage)]["wall_time_seconds"])
            for phase in STRUCTURE_ORDER
            for stage in ("initial", "fixed_cell", "full_cell")
        ]
        per_structure = [
            sum(
                float(
                    summary_rows[(potential, phase, stage)][
                        "wall_time_seconds"
                    ]
                )
                for stage in ("initial", "fixed_cell", "full_cell")
            )
            for phase in STRUCTURE_ORDER
        ]
        evaluations = sum(
            int(summary_rows[(potential, phase, stage)][
                "force_evaluations_total"
            ])
            for phase in STRUCTURE_ORDER
            for stage in ("initial", "fixed_cell", "full_cell")
        )
        runtime_rows.append(
            {
                "potential_key": potential,
                "total_wall_time_seconds": float(np.sum(walls)),
                "mean_wall_time_per_structure_seconds": float(
                    np.mean(per_structure)
                ),
                "median_wall_time_per_structure_seconds": float(
                    np.median(per_structure)
                ),
                "total_force_evaluations": evaluations,
                "states": len(walls),
            }
        )
    mace_context = [
        {
            "phase_key": record.get("phase_key"),
            "wall_time_seconds": record.get("wall_time_seconds"),
        }
        for record in inputs["step6_summary"].get("records", ())
        if isinstance(record, Mapping)
    ]
    runtime_note = (
        "MACE Step 6 full-cell wall times are listed as context only: they "
        "were measured on the same machine but include the Python-side "
        "per-step monitoring of that workflow, so the timing scopes differ "
        "and no precise LAMMPS-versus-MACE speed ratio is claimed."
    )
    return {
        "phases": phases,
        "fractions": fractions,
        "dft_values": dft_values,
        "method_values": method_values,
        "energy_rows": energy_rows,
        "statistics": statistics,
        "mace_statistics_step8": inputs["step8_energy"].get("statistics", {}),
        "mae_ranking": mae_ranking,
        "best_method_by_mae": mae_ranking[0],
        "per_phase_best_method": per_phase_best,
        "structural_rows": structural_rows,
        "volume_statistics": volume_statistics,
        "runtime_rows": runtime_rows,
        "mace_runtime_context": mace_context,
        "runtime_note": runtime_note,
        "envelopes": inputs["formation"].get("selected_set_envelopes", {}),
    }


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


def _render_figures(
    config: Step10Config,
    comparison: Mapping[str, Any],
    targets: Mapping[str, Path],
) -> None:
    """Render the twelve required figures into staged paths."""

    import numpy as np

    plt = lazy_pyplot()
    phases = comparison["phases"]
    fractions = comparison["fractions"]
    dft = comparison["dft_values"]
    method_values = comparison["method_values"]
    statistics = comparison["statistics"]

    # 1. All methods versus composition.
    fig, axis = plt.subplots(figsize=(8.8, 5.6))
    axis.axhline(0.0, color="black", linewidth=0.7)
    order = np.argsort(np.asarray(fractions))
    axis.plot(
        [fractions[i] for i in order],
        [dft[i] for i in order],
        marker="s",
        markersize=6,
        linewidth=1.4,
        color=METHOD_COLORS["mp_dft"],
        label="MP processed DFT",
    )
    for method in METHOD_ORDER:
        axis.plot(
            [fractions[i] for i in order],
            [method_values[method][i] for i in order],
            marker="o",
            markersize=4.5,
            linewidth=1.1,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    for phase, x, y in zip(phases, fractions, dft):
        axis.annotate(
            phase, (x, y), textcoords="offset points", xytext=(5, 6), fontsize=8
        )
    axis.set_xlabel("Ni atomic fraction x in Al(1-x)Ni(x)")
    axis.set_ylabel("Formation energy (eV/atom)")
    axis.set_title(
        "Formation energies by composition: five selected phases only"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["fig_composition"], plt)

    # 2. Signed error by phase and method.
    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    x_positions = np.arange(len(phases), dtype=float)
    width = 0.2
    for offset, method in enumerate(METHOD_ORDER):
        errors = [
            method_values[method][i] - dft[i] for i in range(len(phases))
        ]
        axis.bar(
            x_positions + (offset - 1.5) * width,
            errors,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Signed error vs MP DFT (eV/atom)")
    axis.set_title("Formation-energy error by phase and method (n=5)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["fig_error_by_phase"], plt)

    # 3-5. Parity plots per classical potential.
    for potential, target_key in (
        ("pun_mishin_2009", "fig_parity_2009"),
        ("mishin_2004_ipr2", "fig_parity_2004"),
        ("mishin_2002", "fig_parity_2002"),
    ):
        values = method_values[potential]
        fig, axis = plt.subplots(figsize=(7.0, 7.0))
        low = min(*dft, *values) - 0.06
        high = max(*dft, *values) + 0.06
        axis.plot(
            [low, high],
            [low, high],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="y = x (perfect agreement)",
        )
        axis.scatter(
            dft, values, s=60, color=METHOD_COLORS[potential], zorder=3
        )
        for phase, x, y in zip(phases, dft, values):
            axis.annotate(
                phase,
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("MP processed DFT formation energy (eV/atom)")
        axis.set_ylabel(
            f"{METHOD_LABELS[potential]} formation energy (eV/atom)"
        )
        axis.set_title(f"Parity: {METHOD_LABELS[potential]} vs MP DFT (n=5)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        save_figure(fig, targets[target_key], plt)

    # 6. MAE and RMSE by method.
    fig, axis = plt.subplots(figsize=(8.8, 5.4))
    x_positions = np.arange(len(METHOD_ORDER), dtype=float)
    mae_values = [
        statistics[m]["mean_absolute_error_eV_per_atom"] for m in METHOD_ORDER
    ]
    rmse_values = [statistics[m]["rmse_eV_per_atom"] for m in METHOD_ORDER]
    axis.bar(
        x_positions - 0.18,
        mae_values,
        width=0.36,
        color="#4c78a8",
        label="MAE",
    )
    axis.bar(
        x_positions + 0.18,
        rmse_values,
        width=0.36,
        color="#f58518",
        label="RMSE",
    )
    axis.set_xticks(
        x_positions, [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8
    )
    axis.set_ylabel("Formation-energy error vs MP DFT (eV/atom)")
    axis.set_title("MAE and RMSE by method (n=5 compounds)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["fig_mae_rmse"], plt)

    # 7. Volume percent error by phase and method.
    volume_by_method: dict[str, list[float]] = {}
    for method in METHOD_ORDER:
        rows = {
            row["phase"]: row["volume_error_percent"]
            for row in comparison["structural_rows"]
            if row["method"] == method
        }
        volume_by_method[method] = [rows[phase] for phase in phases]
    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    x_positions = np.arange(len(phases), dtype=float)
    for offset, method in enumerate(METHOD_ORDER):
        axis.bar(
            x_positions + (offset - 1.5) * 0.2,
            volume_by_method[method],
            width=0.2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Volume-per-atom error vs MP (%)")
    axis.set_title("Relaxed volume error by phase and method (n=5)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["fig_volume_error"], plt)

    # 8. Volume MAE by method.
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    x_positions = np.arange(len(METHOD_ORDER), dtype=float)
    axis.bar(
        x_positions,
        [
            comparison["volume_statistics"][m][
                "mean_absolute_volume_percent_error"
            ]
            for m in METHOD_ORDER
        ],
        width=0.55,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
    )
    axis.set_xticks(
        x_positions, [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8
    )
    axis.set_ylabel("Mean absolute volume-per-atom error vs MP (%)")
    axis.set_title("Volume MAE by method (n=5 compounds)")
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(fig, targets["fig_volume_mae"], plt)

    # 9. Runtime by potential.
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    x_positions = np.arange(len(POTENTIAL_ORDER), dtype=float)
    axis.bar(
        x_positions,
        [row["total_wall_time_seconds"] for row in comparison["runtime_rows"]],
        width=0.55,
        color=[METHOD_COLORS[m] for m in POTENTIAL_ORDER],
    )
    axis.set_xticks(
        x_positions, [METHOD_LABELS[m] for m in POTENTIAL_ORDER], fontsize=8
    )
    axis.set_ylabel("Total LAMMPS wall time, 21 states (s)")
    axis.set_title("Runtime by potential (subprocess wall time)")
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(fig, targets["fig_runtime"], plt)

    # 10-12. Selected-set envelopes per potential.
    for potential, target_key in (
        ("pun_mishin_2009", "fig_envelope_2009"),
        ("mishin_2004_ipr2", "fig_envelope_2004"),
        ("mishin_2002", "fig_envelope_2002"),
    ):
        envelope = comparison["envelopes"].get(potential, {})
        vertices = envelope.get("vertices", [])
        fig, axis = plt.subplots(figsize=(8.4, 5.2))
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.plot(
            [vertex["ni_fraction"] for vertex in vertices],
            [vertex["formation_energy_eV_per_atom"] for vertex in vertices],
            color="#2ca02c",
            linewidth=1.6,
            marker="D",
            markersize=5,
            label="selected-set lower convex envelope",
        )
        axis.scatter(
            fractions,
            method_values[potential],
            s=55,
            color=METHOD_COLORS[potential],
            zorder=3,
            label=f"{METHOD_LABELS[potential]} full-cell E_f",
        )
        for phase, x, y in zip(phases, fractions, method_values[potential]):
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
            f"Selected-set lower convex envelope: {METHOD_LABELS[potential]}"
        )
        axis.text(
            0.5,
            0.02,
            "Selected-set construction only; NOT the complete Ni-Al convex "
            "hull.",
            transform=axis.transAxes,
            fontsize=8,
            ha="center",
            color="#666666",
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        save_figure(fig, targets[target_key], plt)


def _benchmark_report_text(
    config: Step10Config, comparison: Mapping[str, Any]
) -> str:
    """Render the human-readable Step 10 benchmark comparison report."""

    statistics = comparison["statistics"]
    lines = [
        "Step 10 - LAMMPS Classical-Potential Benchmark Comparison",
        "=" * 76,
        "",
        "Error convention: signed error = method - MP processed DFT "
        "(eV/atom). Sample size is five compounds; every statistic is "
        "descriptive and correlations are exploratory.",
        "",
        "1. Full-cell formation energies (eV/atom)",
        "phase | MP DFT | " + " | ".join(
            METHOD_LABELS[m] for m in METHOD_ORDER
        ),
    ]
    for index, phase in enumerate(comparison["phases"]):
        lines.append(
            f"{phase} | {comparison['dft_values'][index]:.6f} | "
            + " | ".join(
                f"{comparison['method_values'][m][index]:.6f}"
                for m in METHOD_ORDER
            )
        )
    lines.extend(["", "2. Error metrics versus MP DFT (n=5)"])
    for method in METHOD_ORDER:
        stats = statistics[method]
        lines.extend(
            [
                f"{METHOD_LABELS[method]}:",
                f"  mean signed error: "
                f"{stats['mean_signed_error_eV_per_atom']:+.6f} eV/atom",
                f"  MAE: {stats['mean_absolute_error_eV_per_atom']:.6f}; "
                f"RMSE: {stats['rmse_eV_per_atom']:.6f}; median |err|: "
                f"{stats['median_absolute_error_eV_per_atom']:.6f} eV/atom",
                "  maximum |err|: "
                f"{stats['maximum_absolute_error_eV_per_atom']:.6f} "
                f"({stats['phase_with_maximum_absolute_error']}); minimum: "
                f"{stats['minimum_absolute_error_eV_per_atom']:.6f} "
                f"({stats['phase_with_minimum_absolute_error']})",
                f"  Pearson: {stats['pearson_correlation']}; Spearman: "
                f"{stats['spearman_rank_correlation']}; exact ranking "
                f"agreement: {stats['exact_ranking_agreement']}; pairwise: "
                f"{stats['pairwise_ordering_agreement']}",
                "  |err| <= 0.05 / 0.10 eV/atom: "
                f"{stats['error_threshold_counts']['0.05']}/5, "
                f"{stats['error_threshold_counts']['0.1']}/5",
                "  most negative formation energy among the five selected "
                f"phases under this method: {stats['most_negative_phase']}",
            ]
        )
    lines.extend(
        [
            "",
            "3. Method ranking by formation-energy MAE",
            " > ".join(
                METHOD_LABELS[m] for m in comparison["mae_ranking"]
            )
            + " (best first)",
            f"Best method by MAE: "
            f"{METHOD_LABELS[comparison['best_method_by_mae']]}",
            "Per-phase best method: "
            + "; ".join(
                f"{phase}: {METHOD_LABELS[comparison['per_phase_best_method'][phase]]}"
                for phase in comparison["phases"]
            ),
            "",
            "4. Volume statistics versus MP (per method)",
        ]
    )
    for method in METHOD_ORDER:
        stats = comparison["volume_statistics"][method]
        lines.append(
            f"{METHOD_LABELS[method]}: mean signed "
            f"{stats['mean_signed_volume_percent_error']:+.4f}%; MAE "
            f"{stats['mean_absolute_volume_percent_error']:.4f}%; RMSE "
            f"{stats['rmse_volume_percent_error']:.4f}%; max "
            f"{stats['maximum_absolute_volume_percent_error']:.4f}%; "
            f"symmetry agreement {stats['symmetry_agreement_count']}/5; "
            f"direction: {stats['systematic_direction']}"
        )
    lines.extend(
        [
            "",
            "5. Runtime",
        ]
    )
    for row in comparison["runtime_rows"]:
        lines.append(
            f"{METHOD_LABELS[row['potential_key']]}: total "
            f"{row['total_wall_time_seconds']:.2f} s over {row['states']} "
            f"states; mean per structure "
            f"{row['mean_wall_time_per_structure_seconds']:.2f} s; total "
            f"force evaluations {row['total_force_evaluations']}"
        )
    lines.extend(
        [
            str(comparison["runtime_note"]),
            "",
            "6. Limitations",
            "- Materials Project values are processed DFT-derived references, "
            "not experimental truth.",
            "- Five compounds are not a complete Ni-Al phase diagram; the "
            "per-potential envelopes are selected-set constructions, not "
            "complete hulls.",
            "- Accuracy on equilibrium bulk phases does not prove accuracy "
            "for defects, surfaces, interfaces, finite temperature, or "
            "molecular dynamics.",
            "- LAMMPS is the execution engine; the EAM file is the physical "
            "model.",
            *(f"- {line}" for line in NI_MAGNETIC_LIMITATION),
            "",
        ]
    )
    return "\n".join(lines)


def run_validate_only(config: Step10Config) -> None:
    """Validate every comparison input without writing outputs."""

    inputs = load_comparison_inputs(config)
    targets = comparison_output_paths(config)
    collisions = [path for path in targets.values() if path.exists()]
    print("=" * 78)
    print("STEP 10 COMPARISON INPUT VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(
        "Formation records: "
        f"{len(inputs['formation'].get('records', ()))} (15 expected); "
        f"summary rows: {len(inputs['summary_rows'])}/63"
    )
    print("Step 8 MACE/MP records: 5/5 validated")
    print(
        "Existing comparison outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("LAMMPS executed: No")
    print("MACE loaded: No")
    print("Materials Project queried: No")
    print("Comparison outputs written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_compare(config: Step10Config, *, overwrite: bool) -> dict[str, Any]:
    """Build, stage, and transactionally publish the comparison bundle."""

    inputs = load_comparison_inputs(config)
    comparison = build_comparison(config, inputs)
    targets = comparison_output_paths(config)
    if not overwrite:
        collisions = [path for path in targets.values() if path.exists()]
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step10CollisionError(
                "Existing Step 10 comparison outputs were found; re-run with "
                "--overwrite after review:\n" + listing
            )
    generated = utc_timestamp()
    energy_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_vs_mace_mp",
        "project_step": "10",
        "generated_at_utc": generated,
        "configuration_fingerprint_sha256": config.fingerprint,
        "error_convention": (
            "signed_error = method - MP processed DFT (eV/atom)."
        ),
        "records": comparison["energy_rows"],
        "statistics_by_method": comparison["statistics"],
        "mace_statistics_from_step8": comparison["mace_statistics_step8"],
        "mae_ranking_best_first": comparison["mae_ranking"],
        "best_method_by_mae": comparison["best_method_by_mae"],
        "per_phase_best_method": comparison["per_phase_best_method"],
        "selected_set_envelopes": comparison["envelopes"],
        "limitations": [
            "n=5 compounds; statistics are descriptive and correlations "
            "exploratory.",
            "Materials Project values are processed DFT-derived references, "
            "not experimental truth.",
            "Selected-set envelopes are not complete Ni-Al convex hulls.",
            *NI_MAGNETIC_LIMITATION,
        ],
    }
    structural_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_structural_comparison",
        "project_step": "10",
        "generated_at_utc": generated,
        "configuration_fingerprint_sha256": config.fingerprint,
        "symmetry_settings": {
            "symprec_A": config.symmetry_symprec_A,
            "angle_tolerance_deg": config.symmetry_angle_tolerance_deg,
        },
        "records": comparison["structural_rows"],
        "volume_statistics_by_method": comparison["volume_statistics"],
    }
    runtime_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_lammps_runtime_summary",
        "project_step": "10",
        "generated_at_utc": generated,
        "records": comparison["runtime_rows"],
        "mace_step6_context": comparison["mace_runtime_context"],
        "comparability_note": comparison["runtime_note"],
    }
    checkpoint_document = {
        **energy_document,
        "artifact_type": "ni_al_step10_benchmark_result",
        "structural_records": comparison["structural_rows"],
        "volume_statistics_by_method": comparison["volume_statistics"],
        "runtime_records": comparison["runtime_rows"],
        "overall_status": "SUCCESS",
    }
    report_text = _benchmark_report_text(config, comparison)

    root = config.analysis_root
    for name in ("tables", "checkpoints", "reports", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".step10-comparison-", dir=root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for target, payload in (
            (
                targets["energy_csv"],
                _csv_bytes(comparison["energy_rows"], ENERGY_FIELDNAMES),
            ),
            (targets["energy_json"], write_strict_json_bytes(energy_document)),
            (
                targets["structural_csv"],
                _csv_bytes(
                    comparison["structural_rows"], STRUCTURAL_FIELDNAMES
                ),
            ),
            (
                targets["structural_json"],
                write_strict_json_bytes(structural_document),
            ),
            (
                targets["runtime_csv"],
                _csv_bytes(
                    comparison["runtime_rows"],
                    tuple(comparison["runtime_rows"][0].keys()),
                ),
            ),
            (
                targets["runtime_json"],
                write_strict_json_bytes(runtime_document),
            ),
            (
                targets["checkpoint"],
                write_strict_json_bytes(checkpoint_document),
            ),
            (targets["report"], report_text.encode("utf-8")),
        ):
            staged = stage_path(staging_root, root, target)
            staged.write_bytes(payload)
            staged_by_final[target] = staged
        figure_targets = {
            key: stage_path(staging_root, root, targets[key])
            for key in targets
            if key.startswith("fig_")
        }
        _render_figures(config, comparison, figure_targets)
        for key, staged in figure_targets.items():
            staged_by_final[targets[key]] = staged
        publish_files_transactionally(
            config.project_root, root, staged_by_final, overwrite=overwrite
        )
    assert_mace_not_imported()
    return checkpoint_document


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.compare:
        LOGGER.error("--overwrite is allowed only with --compare.")
        return 1
    try:
        config = load_step10_config(args.config)
        if args.validate_only:
            run_validate_only(config)
            return 0
        document = run_compare(config, overwrite=args.overwrite)
        print("=" * 78)
        print("STEP 10 COMPARISON COMPLETED")
        print("=" * 78)
        for method in METHOD_ORDER:
            stats = document["statistics_by_method"][method]
            print(
                f"{METHOD_LABELS[method]}: MAE="
                f"{stats['mean_absolute_error_eV_per_atom']:.6f}; RMSE="
                f"{stats['rmse_eV_per_atom']:.6f}; mean signed="
                f"{stats['mean_signed_error_eV_per_atom']:+.6f} eV/atom"
            )
        print(
            "Best method by MAE: "
            f"{METHOD_LABELS[document['best_method_by_mae']]}"
        )
        print("=" * 78)
        return 0
    except (Step10Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial comparison bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

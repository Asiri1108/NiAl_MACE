"""Compare validated MACE Ni-Al results against MP DFT benchmark records.

This analysis command never queries Materials Project, never loads MACE,
never creates an optimizer, never reruns any relaxation, performs no DFT,
and never modifies Step 6 or Step 7 data.  It consumes only the published
raw benchmark bundles and the machine-readable Step 6/7 MACE results, and
produces the Step 8 comparison tables, reports, checkpoint, and figures.

The benchmark energy is the Materials Project processed
``formation_energy_per_atom``; raw MACE and VASP total energies are never
compared.  Materials Project energy above hull is reported as context only
and is never subtracted from the Step 7 selected-set envelope values.
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
    NI_MAGNETIC_LIMITATION,
    SELECTED_SET_LIMITATION,
    Step7Error,
    assert_mace_not_imported,
    installed_step7_versions,
    lazy_pyplot,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    save_figure,
    stage_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step8_utils import (
    BENCHMARK_LIMITATIONS,
    PHASE_ORDER,
    SCHEMA_VERSION,
    BenchmarkRecord,
    ComparisonRecord,
    MaceSourceBundle,
    Step8CollisionError,
    Step8Config,
    Step8Error,
    StructuralRecord,
    calculate_comparisons,
    calculate_statistics,
    calculate_structural_comparisons,
    calculate_structural_statistics,
    load_benchmark_records,
    load_step8_config,
    step8_directories,
    step8_output_paths,
    validate_step7_sources,
)


LOGGER = logging.getLogger("ni_al_step8.compare")
DEFAULT_CONFIG = Path("configs/mace_dft_benchmark.json")

ENERGY_TABLE_FIELDNAMES: tuple[str, ...] = (
    "phase_key",
    "material_id",
    "reduced_formula",
    "ni_atomic_fraction",
    "mp_database_version",
    "mp_thermo_type",
    "mp_formation_energy_eV_per_atom",
    "mp_energy_above_hull_eV_per_atom",
    "mp_is_stable",
    "mace_initial_formation_energy_eV_per_atom",
    "mace_relaxed_formation_energy_eV_per_atom",
    "initial_signed_error_eV_per_atom",
    "relaxed_signed_error_eV_per_atom",
    "relaxed_absolute_error_eV_per_atom",
    "squared_error_eV2_per_atom2",
    "relaxation_effect_on_benchmark_error_eV_per_atom",
    "diagnostic_relative_error_percent",
    "mace_on_selected_set_envelope",
    "mace_convergence_status",
    "mace_safety_status",
    "benchmark_provenance_path",
    "mace_provenance_path",
)

STRUCTURAL_TABLE_FIELDNAMES: tuple[str, ...] = (
    "phase_key",
    "material_id",
    "mp_volume_per_atom_A3",
    "mace_volume_per_atom_A3",
    "signed_volume_per_atom_difference_A3",
    "absolute_volume_per_atom_difference_A3",
    "volume_per_atom_difference_percent",
    "mp_density_g_cm3",
    "mace_density_g_cm3",
    "mp_space_group",
    "mace_space_group",
    "symmetry_preserved",
    "lattice_comparison_available",
    "lattice_comparison_note",
    "mp_lattice_abc_A",
    "mace_lattice_abc_A",
    "lattice_abc_differences_A",
    "mp_lattice_angles_deg",
    "mace_lattice_angles_deg",
    "lattice_angle_differences_deg",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the comparison analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare validated MACE Ni-Al formation energies and relaxed "
            "structures against Materials Project DFT benchmark records."
        )
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
        help=(
            "Validate benchmark and MACE inputs without writing comparison "
            "outputs."
        ),
    )
    action.add_argument(
        "--compare",
        action="store_true",
        help="Create the Step 8 comparison outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 8 result outputs.",
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


def _energy_row(record: ComparisonRecord) -> dict[str, Any]:
    """Serialize one comparison record as a flat table row."""

    return {
        "phase_key": record.phase_key,
        "material_id": record.material_id,
        "reduced_formula": record.reduced_formula,
        "ni_atomic_fraction": record.ni_fraction,
        "mp_database_version": record.mp_database_version,
        "mp_thermo_type": record.mp_thermo_type,
        "mp_formation_energy_eV_per_atom": (
            record.mp_formation_energy_eV_per_atom
        ),
        "mp_energy_above_hull_eV_per_atom": (
            record.mp_energy_above_hull_eV_per_atom
        ),
        "mp_is_stable": record.mp_is_stable,
        "mace_initial_formation_energy_eV_per_atom": (
            record.mace_initial_formation_energy_eV_per_atom
        ),
        "mace_relaxed_formation_energy_eV_per_atom": (
            record.mace_relaxed_formation_energy_eV_per_atom
        ),
        "initial_signed_error_eV_per_atom": (
            record.initial_signed_error_eV_per_atom
        ),
        "relaxed_signed_error_eV_per_atom": (
            record.relaxed_signed_error_eV_per_atom
        ),
        "relaxed_absolute_error_eV_per_atom": (
            record.relaxed_absolute_error_eV_per_atom
        ),
        "squared_error_eV2_per_atom2": record.squared_error_eV2_per_atom2,
        "relaxation_effect_on_benchmark_error_eV_per_atom": (
            record.relaxation_effect_on_benchmark_error_eV_per_atom
        ),
        "diagnostic_relative_error_percent": (
            record.diagnostic_relative_error_percent
        ),
        "mace_on_selected_set_envelope": record.mace_on_selected_set_envelope,
        "mace_convergence_status": record.mace_convergence_status,
        "mace_safety_status": record.mace_safety_status,
        "benchmark_provenance_path": record.benchmark_provenance_path,
        "mace_provenance_path": record.mace_provenance_path,
    }


def _structural_row(record: StructuralRecord) -> dict[str, Any]:
    """Serialize one structural record as a flat table row."""

    def fmt(values: tuple[float, float, float] | None) -> str:
        if values is None:
            return "not available"
        return "[" + ", ".join(f"{value:.9f}" for value in values) + "]"

    return {
        "phase_key": record.phase_key,
        "material_id": record.material_id,
        "mp_volume_per_atom_A3": record.mp_volume_per_atom_A3,
        "mace_volume_per_atom_A3": record.mace_volume_per_atom_A3,
        "signed_volume_per_atom_difference_A3": (
            record.signed_volume_per_atom_difference_A3
        ),
        "absolute_volume_per_atom_difference_A3": (
            record.absolute_volume_per_atom_difference_A3
        ),
        "volume_per_atom_difference_percent": (
            record.volume_per_atom_difference_percent
        ),
        "mp_density_g_cm3": record.mp_density_g_cm3,
        "mace_density_g_cm3": record.mace_density_g_cm3,
        "mp_space_group": (
            f"{record.mp_space_group_symbol} ({record.mp_space_group_number})"
        ),
        "mace_space_group": (
            f"{record.mace_space_group_symbol} "
            f"({record.mace_space_group_number})"
        ),
        "symmetry_preserved": record.symmetry_preserved,
        "lattice_comparison_available": record.lattice_comparison_available,
        "lattice_comparison_note": record.lattice_comparison_note,
        "mp_lattice_abc_A": fmt(record.mp_lattice_abc_A),
        "mace_lattice_abc_A": fmt(record.mace_lattice_abc_A),
        "lattice_abc_differences_A": fmt(record.lattice_abc_differences_A),
        "mp_lattice_angles_deg": fmt(record.mp_lattice_angles_deg),
        "mace_lattice_angles_deg": fmt(record.mace_lattice_angles_deg),
        "lattice_angle_differences_deg": fmt(
            record.lattice_angle_differences_deg
        ),
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


def _hull_context_rows(
    comparisons: Sequence[ComparisonRecord],
) -> list[dict[str, Any]]:
    """Build the clearly separated energy-above-hull context table.

    Materials Project energy above hull and the Step 7 selected-set
    envelope membership answer different questions and are never
    subtracted from each other.
    """

    return [
        {
            "phase_key": record.phase_key,
            "mp_energy_above_hull_eV_per_atom": (
                record.mp_energy_above_hull_eV_per_atom
            ),
            "mp_is_stable": record.mp_is_stable,
            "mace_on_selected_set_envelope_step7": (
                record.mace_on_selected_set_envelope
            ),
            "note": (
                "MP hull uses the full MP Ni-Al entry set and its processed "
                "thermodynamic scheme; the Step 7 envelope uses only five "
                "compounds plus elemental endpoints on the MACE surface. "
                "These values answer different questions."
            ),
        }
        for record in comparisons
    ]


def _render_figures(
    config: Step8Config,
    comparisons: Sequence[ComparisonRecord],
    structural: Sequence[StructuralRecord],
    targets: Mapping[str, Path],
) -> None:
    """Render the six Step 8 figures into staged paths."""

    import numpy as np

    plt = lazy_pyplot()
    phases = [record.phase_key for record in comparisons]
    dft = [record.mp_formation_energy_eV_per_atom for record in comparisons]
    mace_relaxed = [
        record.mace_relaxed_formation_energy_eV_per_atom
        for record in comparisons
    ]
    signed = [
        record.relaxed_signed_error_eV_per_atom for record in comparisons
    ]
    initial_signed = [
        record.initial_signed_error_eV_per_atom for record in comparisons
    ]
    fractions = [record.ni_fraction for record in comparisons]

    # Figure 1: parity plot with a y = x reference.
    fig, axis = plt.subplots(figsize=(7.2, 7.2))
    low = min(*dft, *mace_relaxed) - 0.05
    high = max(*dft, *mace_relaxed) + 0.05
    axis.plot(
        [low, high],
        [low, high],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="y = x (perfect agreement)",
    )
    axis.scatter(dft, mace_relaxed, s=60, color="#1f77b4", zorder=3)
    for phase, x, y in zip(phases, dft, mace_relaxed):
        axis.annotate(
            phase, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8
        )
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("MP processed DFT formation energy (eV/atom)")
    axis.set_ylabel("MACE relaxed formation energy (eV/atom)")
    axis.set_title("Formation-energy parity: MACE vs Materials Project (n=5)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["parity"], plt)

    # Figure 2: relaxed signed error by phase.
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    x_positions = np.arange(len(phases), dtype=float)
    axis.bar(x_positions, signed, width=0.56, color="#d62728")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Signed error, MACE - MP DFT (eV/atom)")
    axis.set_title("Relaxed MACE formation-energy error by phase (n=5)")
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(fig, targets["error"], plt)

    # Figure 3: formation energy versus composition for both methods.
    order = np.argsort(np.asarray(fractions))
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.plot(
        [fractions[index] for index in order],
        [dft[index] for index in order],
        marker="s",
        markersize=6,
        linewidth=1.2,
        color="#2ca02c",
        label="MP processed DFT",
    )
    axis.plot(
        [fractions[index] for index in order],
        [mace_relaxed[index] for index in order],
        marker="o",
        markersize=6,
        linewidth=1.2,
        color="#1f77b4",
        label="MACE relaxed",
    )
    for phase, x, y in zip(phases, fractions, mace_relaxed):
        axis.annotate(
            phase, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8
        )
    axis.set_xlabel("Ni atomic fraction x in Al(1-x)Ni(x)")
    axis.set_ylabel("Formation energy (eV/atom)")
    axis.set_title(
        "Formation energies by composition: five selected phases only"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["composition"], plt)

    # Figure 4: initial versus relaxed MACE benchmark error.
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    width = 0.36
    axis.bar(
        x_positions - width / 2.0,
        initial_signed,
        width=width,
        color="#4c78a8",
        label="initial fixed-geometry MACE error",
    )
    axis.bar(
        x_positions + width / 2.0,
        signed,
        width=width,
        color="#f58518",
        label="relaxed MACE error",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Signed error, MACE - MP DFT (eV/atom)")
    axis.set_title("Initial versus relaxed MACE formation-energy error (n=5)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["initial_vs_relaxed"], plt)

    # Figure 5: volume per atom, MACE versus MP.
    mp_volumes = [record.mp_volume_per_atom_A3 for record in structural]
    mace_volumes = [record.mace_volume_per_atom_A3 for record in structural]
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.bar(
        x_positions - width / 2.0,
        mp_volumes,
        width=width,
        color="#2ca02c",
        label="MP DFT-relaxed",
    )
    axis.bar(
        x_positions + width / 2.0,
        mace_volumes,
        width=width,
        color="#1f77b4",
        label="MACE full-cell relaxed",
    )
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Volume per atom (angstrom^3/atom)")
    axis.set_title("Volume per atom: MACE versus Materials Project (n=5)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(fig, targets["volume"], plt)

    # Figure 6: volume percentage error by phase.
    percents = [
        record.volume_per_atom_difference_percent for record in structural
    ]
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.bar(x_positions, percents, width=0.56, color="#9467bd")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, phases)
    axis.set_ylabel("Volume-per-atom error, MACE vs MP (%)")
    axis.set_title("MACE volume-per-atom percentage error by phase (n=5)")
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(fig, targets["volume_error"], plt)


def _comparison_report_text(
    config: Step8Config,
    comparisons: Sequence[ComparisonRecord],
    statistics: Mapping[str, Any],
    structural: Sequence[StructuralRecord],
    structural_statistics: Mapping[str, Any],
    warnings_lines: Sequence[str],
) -> str:
    """Build the human-readable Step 8 comparison report."""

    lines = [
        "Step 8 - MACE versus Materials Project DFT Benchmark",
        "=" * 76,
        "",
        "1. Objective",
        "------------",
        "Quantify how the relaxed MACE-MP-0 Small formation energies and "
        "relaxed structures for the five selected Ni-Al phases differ from "
        "the Materials Project processed DFT-derived references.",
        "",
        "2. Benchmark definition",
        "-----------------------",
        "The benchmark energy is the Materials Project processed "
        "formation_energy_per_atom from the summary endpoint. Raw MACE and "
        "VASP total energies were never compared.",
        "",
        "3. Per-phase formation-energy comparison (eV/atom)",
        "--------------------------------------------------",
        "phase | x_Ni | MP DFT E_f | MACE initial E_f | MACE relaxed E_f | "
        "initial signed err | relaxed signed err | |err|",
    ]
    for record in comparisons:
        lines.append(
            f"{record.phase_key} | {record.ni_fraction:.4f} | "
            f"{record.mp_formation_energy_eV_per_atom:.6f} | "
            f"{record.mace_initial_formation_energy_eV_per_atom:.6f} | "
            f"{record.mace_relaxed_formation_energy_eV_per_atom:.6f} | "
            f"{record.initial_signed_error_eV_per_atom:+.6f} | "
            f"{record.relaxed_signed_error_eV_per_atom:+.6f} | "
            f"{record.relaxed_absolute_error_eV_per_atom:.6f}"
        )
    lines.extend(
        [
            "",
            "Signed error convention: MACE minus MP DFT; positive means MACE "
            "predicts a less negative formation energy than the reference.",
            "",
            "4. Dataset statistics (n=5; descriptive only)",
            "---------------------------------------------",
            f"Mean signed error: "
            f"{statistics['mean_signed_error_eV_per_atom']:+.6f} eV/atom",
            f"Mean absolute error (MAE): "
            f"{statistics['mean_absolute_error_eV_per_atom']:.6f} eV/atom",
            f"Root mean squared error (RMSE): "
            f"{statistics['root_mean_squared_error_eV_per_atom']:.6f} eV/atom",
            f"Median absolute error: "
            f"{statistics['median_absolute_error_eV_per_atom']:.6f} eV/atom",
            "Maximum absolute error: "
            f"{statistics['maximum_absolute_error_eV_per_atom']:.6f} eV/atom "
            f"({statistics['phase_with_maximum_absolute_error']})",
            "Minimum absolute error phase: "
            f"{statistics['phase_with_minimum_absolute_error']}",
            "Standard deviation of signed errors: "
            f"{statistics['standard_deviation_of_signed_errors_eV_per_atom']:.6f}"
            " eV/atom (population, n=5)",
            f"Pearson correlation (exploratory): "
            f"{statistics['pearson_correlation']}",
            f"Spearman rank correlation (exploratory): "
            f"{statistics['spearman_rank_correlation']}",
            "Error-threshold counts (descriptive bins only): "
            + "; ".join(
                f"|err| <= {threshold} eV/atom: {count}/5"
                for threshold, count in statistics[
                    "error_threshold_counts"
                ].items()
            ),
            "All signed errors positive: "
            f"{statistics['all_signed_errors_positive']}",
            "All signed errors negative: "
            f"{statistics['all_signed_errors_negative']}",
            "",
            "5. Ranking and trend",
            "--------------------",
            "MACE ranking (most negative first): "
            + " < ".join(statistics["mace_ranking_most_negative_first"]),
            "MP DFT ranking (most negative first): "
            + " < ".join(statistics["dft_ranking_most_negative_first"]),
            f"Exact ranking agreement: {statistics['exact_ranking_agreement']}",
            "Pairwise ordering agreement: "
            f"{statistics['pairwise_ordering_agreement']}",
            "Composition-trend signs match: "
            f"{statistics['composition_trend_signs_match']}",
            (
                f"{statistics['most_negative_phase_mace']} has the most "
                "negative formation energy among the five selected phases "
                "under both methods."
                if statistics["most_negative_phase_mace"]
                == statistics["most_negative_phase_dft"]
                else (
                    "The most negative formation-energy phase differs: "
                    f"{statistics['most_negative_phase_mace']} (MACE) versus "
                    f"{statistics['most_negative_phase_dft']} (MP DFT), each "
                    "under its specified method."
                )
            ),
            "",
            "6. Energy-above-hull context (not an error metric)",
            "--------------------------------------------------",
        ]
    )
    for record in comparisons:
        lines.append(
            f"{record.phase_key}: MP hull = "
            f"{record.mp_energy_above_hull_eV_per_atom:.6f} eV/atom; "
            f"MP stable = {record.mp_is_stable}; Step 7 selected-set "
            f"envelope member = {record.mace_on_selected_set_envelope}"
        )
    lines.extend(
        [
            "MP energy above hull uses the full Materials Project Ni-Al "
            "entry set; the Step 7 envelope uses only five compounds plus "
            "elemental endpoints on the MACE surface. These values answer "
            "different questions and were never subtracted.",
            "",
            "7. Structural comparison",
            "------------------------",
            "phase | MP V/atom | MACE V/atom | dV (%) | MP SG | MACE SG | "
            "preserved",
        ]
    )
    for record in structural:
        lines.append(
            f"{record.phase_key} | {record.mp_volume_per_atom_A3:.6f} | "
            f"{record.mace_volume_per_atom_A3:.6f} | "
            f"{record.volume_per_atom_difference_percent:+.4f} | "
            f"{record.mp_space_group_symbol} "
            f"({record.mp_space_group_number}) | "
            f"{record.mace_space_group_symbol} "
            f"({record.mace_space_group_number}) | "
            f"{record.symmetry_preserved}"
        )
    lines.extend(
        [
            "",
            f"Standardization: {structural[0].standardization_method}.",
            "Structural summary: "
            f"mean signed volume error "
            f"{structural_statistics['mean_signed_volume_percent_error']:+.4f}%"
            f"; MAE "
            f"{structural_statistics['mean_absolute_volume_percent_error']:.4f}%"
            f"; RMSE "
            f"{structural_statistics['rmse_volume_percent_error']:.4f}%; "
            "maximum "
            f"{structural_statistics['maximum_absolute_volume_percent_error']:.4f}%"
            f" ({structural_statistics['phase_with_maximum_absolute_volume_error']})"
            f"; symmetry agreement "
            f"{structural_statistics['symmetry_agreement_count']}/5.",
            structural_statistics["systematic_volume_statement"],
            "",
            "8. Warnings",
            "-----------",
            *(
                [f"- {line}" for line in warnings_lines]
                if warnings_lines
                else ["- None recorded."]
            ),
            "",
            "9. Limitations",
            "--------------",
            *(f"- {line}" for line in BENCHMARK_LIMITATIONS),
            "- Sample size is five compounds; correlation metrics are "
            "exploratory and threshold counts are descriptive bins only.",
            *(f"- {line}" for line in NI_MAGNETIC_LIMITATION),
            "",
        ]
    )
    return "\n".join(lines)


def _final_report_text(
    config: Step8Config,
    comparisons: Sequence[ComparisonRecord],
    statistics: Mapping[str, Any],
    structural: Sequence[StructuralRecord],
    structural_statistics: Mapping[str, Any],
    mace: MaceSourceBundle,
    benchmarks: Mapping[str, BenchmarkRecord],
    warnings_lines: Sequence[str],
    inventory: Sequence[str],
    status: str,
) -> str:
    """Build the authoritative Step 8 final report."""

    database_versions = sorted(
        {
            str(record.database_version)
            for record in benchmarks.values()
            if record.database_version is not None
        }
    )
    signed = [
        record.relaxed_signed_error_eV_per_atom for record in comparisons
    ]
    common_sign = (
        "All five relaxed signed errors are positive (MACE formation "
        "energies are less negative than the MP references)."
        if statistics["all_signed_errors_positive"]
        else (
            "All five relaxed signed errors are negative (MACE formation "
            "energies are more negative than the MP references)."
            if statistics["all_signed_errors_negative"]
            else "The relaxed signed errors do not share a common sign."
        )
    )
    lines = [
        "Ni-Al MACE Step 8 Final Report",
        "=" * 78,
        "",
        "1. Step 8 objective",
        "Benchmark the Step 7 relaxed MACE formation energies and Step 6 "
        "MACE-relaxed structures against Materials Project DFT-derived "
        "processed reference data for the five selected Ni-Al phases.",
        "",
        "2. Completed gates",
        "Gate 1 preflight; Gate 2 configuration; Gate 3 benchmark "
        "retrieval; Gate 4 benchmark validation; Gate 5 MACE validation; "
        "Gate 6 formation-energy errors; Gate 7 structural comparison; "
        "Gate 8 statistics and ranking; Gate 9 tables, reports, figures; "
        "Gate 10 documentation and verification.",
        "",
        "3. Materials Project retrieval method",
        "Exact material-ID queries against the official public summary "
        "endpoint, with the thermodynamic endpoint inspected to document "
        "the processed entry type. The API key was never printed or stored.",
        "",
        "4. Database version",
        "; ".join(database_versions) if database_versions else "unavailable",
        "Step 7 retrieval version(s): "
        + "; ".join(
            f"{key}={value}"
            for key, value in mace.step7_database_version.items()
        ),
        "",
        "5. Benchmark field definitions",
        "Primary benchmark: Materials Project processed "
        "formation_energy_per_atom (summary endpoint). Context fields: "
        "energy_above_hull, is_stable, symmetry, structure, volume. These "
        "are processed DFT-derived values under the MP correction scheme, "
        "not experimental truth and not raw uncorrected totals.",
        "",
        "6. Thermodynamic entry selection method",
        "All thermo entries per material were retrieved; the entry type "
        "whose formation energy numerically matches the summary processed "
        "value is recorded as its provenance. Incompatible functional "
        "schemes were never averaged or mixed. Per-phase resolution:",
    ]
    for phase in PHASE_ORDER:
        record = benchmarks[phase]
        lines.append(
            f"- {phase}: {record.selected_thermo_type} "
            f"({record.thermo_selection_rationale})"
        )
    lines.extend(
        [
            "",
            "7. MACE source files and model settings",
            "Step 7 table: results/mace_formation_energy/tables/"
            "ni_al_mace_formation_energies.json; Step 6 summary: "
            "results/mace_relaxation/full_cell/tables/"
            "ni_al_full_cell_summary.json. Model: MACE-MP-0 Small; cpu; "
            "float64; dispersion=false. mu_Al_MACE = "
            f"{mace.chemical_potentials['mu_Al_MACE']:.9f} eV/atom; "
            "mu_Ni_MACE = "
            f"{mace.chemical_potentials['mu_Ni_MACE']:.9f} eV/atom.",
            "",
            "8-11. Per-phase energies and errors (eV/atom)",
        ]
    )
    for record in comparisons:
        lines.append(
            f"- {record.phase_key}: MP DFT = "
            f"{record.mp_formation_energy_eV_per_atom:.6f}; MACE initial = "
            f"{record.mace_initial_formation_energy_eV_per_atom:.6f}; MACE "
            f"relaxed = "
            f"{record.mace_relaxed_formation_energy_eV_per_atom:.6f}; "
            f"signed error = {record.relaxed_signed_error_eV_per_atom:+.6f}; "
            f"absolute error = "
            f"{record.relaxed_absolute_error_eV_per_atom:.6f}"
        )
    lines.extend(
        [
            "",
            "12-15. Aggregate error metrics (n=5)",
            f"MAE = {statistics['mean_absolute_error_eV_per_atom']:.6f} "
            "eV/atom",
            f"RMSE = "
            f"{statistics['root_mean_squared_error_eV_per_atom']:.6f} eV/atom",
            "Mean signed error = "
            f"{statistics['mean_signed_error_eV_per_atom']:+.6f} eV/atom",
            "Maximum absolute error = "
            f"{statistics['maximum_absolute_error_eV_per_atom']:.6f} eV/atom "
            f"({statistics['phase_with_maximum_absolute_error']})",
            "",
            "16. Correlation and ranking",
            f"Pearson (exploratory): {statistics['pearson_correlation']}; "
            f"Spearman (exploratory): "
            f"{statistics['spearman_rank_correlation']}; exact ranking "
            f"agreement: {statistics['exact_ranking_agreement']}; pairwise "
            f"ordering agreement: "
            f"{statistics['pairwise_ordering_agreement']}; most negative "
            "formation energy among the five selected phases under each "
            f"method: {statistics['most_negative_phase_mace']} (MACE), "
            f"{statistics['most_negative_phase_dft']} (MP DFT).",
            "",
            "17. Descriptive error-threshold counts",
            "; ".join(
                f"|err| <= {threshold} eV/atom: {count}/5"
                for threshold, count in statistics[
                    "error_threshold_counts"
                ].items()
            )
            + " (descriptive bins only, not universal accuracy standards)",
            "",
            "18. Materials Project energy-above-hull context",
        ]
    )
    for record in comparisons:
        lines.append(
            f"- {record.phase_key}: hull = "
            f"{record.mp_energy_above_hull_eV_per_atom:.6f} eV/atom; stable "
            f"= {record.mp_is_stable}; Step 7 envelope member = "
            f"{record.mace_on_selected_set_envelope}"
        )
    lines.extend(
        [
            "MP hull values and the Step 7 selected-set envelope answer "
            "different questions and were never subtracted from each other.",
            "",
            "19-20. Volume and symmetry comparison",
        ]
    )
    for record in structural:
        lines.append(
            f"- {record.phase_key}: MP V/atom = "
            f"{record.mp_volume_per_atom_A3:.6f} A^3; MACE V/atom = "
            f"{record.mace_volume_per_atom_A3:.6f} A^3; error = "
            f"{record.volume_per_atom_difference_percent:+.4f}%; symmetry "
            f"{record.mp_space_group_symbol} -> "
            f"{record.mace_space_group_symbol} "
            f"({'preserved' if record.symmetry_preserved else 'changed'})"
        )
    lines.extend(
        [
            "Volume summary: mean signed "
            f"{structural_statistics['mean_signed_volume_percent_error']:+.4f}%"
            f"; MAE "
            f"{structural_statistics['mean_absolute_volume_percent_error']:.4f}%"
            f"; RMSE "
            f"{structural_statistics['rmse_volume_percent_error']:.4f}%; "
            "symmetry agreement "
            f"{structural_statistics['symmetry_agreement_count']}/5.",
            "",
            "21. Systematic cell-expansion evidence",
            structural_statistics["systematic_volume_statement"],
            "",
            "22. Ni magnetic limitation",
            *(f"- {line}" for line in NI_MAGNETIC_LIMITATION),
            "",
            "23. Warnings",
            *(
                [f"- {line}" for line in warnings_lines]
                if warnings_lines
                else ["- None recorded."]
            ),
            "",
            "24. Scientific limitations",
            *(f"- {line}" for line in BENCHMARK_LIMITATIONS),
            "- Five compounds are a small sample; no universal accuracy or "
            "inaccuracy claim about MACE is made.",
            "",
            "25. Fine-tuning decision boundary",
            f"{common_sign} The mean signed error of "
            f"{statistics['mean_signed_error_eV_per_atom']:+.6f} eV/atom and "
            f"MAE of {statistics['mean_absolute_error_eV_per_atom']:.6f} "
            "eV/atom are moderate on the explicit numeric scale of the "
            "benchmarked formation energies (roughly -0.40 to -0.69 "
            "eV/atom), and the volume errors share one sign, which "
            "together justify a deeper fine-tuning investigation - but no "
            "undocumented universal threshold is used, and fine-tuning is "
            "neither started nor declared necessary here. The distinct "
            "issues are: (1) formation-energy bias, (2) systematic volume "
            "error, (3) ranking agreement "
            f"({statistics['exact_ranking_agreement']}), (4) the Ni "
            "magnetic limitation, and (5) the five-phase sample size.",
            "",
            "26. Output inventory",
            *(f"- {item}" for item in inventory),
            "",
            "27. Protected-file verification",
            "All Step 6 and Step 7 inputs consumed by Step 8 retained their "
            "recorded SHA-256, size, and modification-time fingerprints; no "
            "MACE calculation was rerun and no source was modified.",
            "",
            "28. Overall Step 8 status",
            f"OVERALL STEP 8 STATUS: {status}",
            "",
            "29. Exact next stage",
            "Step 9 - Select and document candidate classical Ni-Al "
            "interatomic potentials and design the LAMMPS comparison.",
            "Step 9 is not implemented by this workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def _collect_warnings(
    benchmarks: Mapping[str, BenchmarkRecord]
) -> list[str]:
    """Collect retrieval warnings recorded in the benchmark metadata."""

    warnings_lines: list[str] = []
    for phase in PHASE_ORDER:
        record = benchmarks[phase]
        try:
            metadata = read_strict_json(
                record.metadata_path, f"{phase} benchmark metadata"
            )
        except Step7Error:
            continue
        warning = metadata.get("database_version_warning")
        if isinstance(warning, str) and warning:
            warnings_lines.append(warning)
        if record.selected_thermo_type is None:
            warnings_lines.append(
                f"{phase}: the processed thermodynamic entry type could not "
                "be resolved numerically; the summary value remains the "
                "documented primary benchmark."
            )
    return sorted(set(warnings_lines))


def load_and_validate_inputs(
    config: Step8Config,
) -> tuple[Mapping[str, BenchmarkRecord], MaceSourceBundle]:
    """Validate every comparison input without writing anything."""

    assert_mace_not_imported()
    mace = validate_step7_sources(config)
    benchmarks, _snapshots = load_benchmark_records(config)
    assert_mace_not_imported()
    return benchmarks, mace


def run_validate_only(config: Step8Config) -> None:
    """Validate benchmark and MACE inputs and report the analysis plan."""

    benchmarks, mace = load_and_validate_inputs(config)
    outputs = step8_output_paths(config)
    collisions = [path for path in outputs.all_paths() if path.exists()]
    print("=" * 78)
    print("STEP 8 COMPARISON INPUT VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    for phase in PHASE_ORDER:
        benchmark = benchmarks[phase]
        mace_record = mace.records[phase]
        print(
            f"{phase}: MP {benchmark.material_id} "
            f"(E_f={benchmark.formation_energy_per_atom_eV:.6f} eV/atom; "
            f"thermo_type={benchmark.selected_thermo_type}); MACE relaxed "
            f"E_f={mace_record.relaxed_formation_energy_eV_per_atom:.6f} "
            "eV/atom; validation PASS"
        )
    print(
        "Existing Step 8 outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("Materials Project queried: No")
    print("MACE loaded: No")
    print("Comparison outputs written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_compare(
    config: Step8Config, *, overwrite: bool
) -> tuple[Sequence[ComparisonRecord], Mapping[str, Any]]:
    """Calculate, stage, validate, and publish the Step 8 bundle."""

    benchmarks, mace = load_and_validate_inputs(config)
    comparisons = calculate_comparisons(config, benchmarks, mace)
    statistics = calculate_statistics(config, comparisons)
    structural = calculate_structural_comparisons(config, benchmarks, mace)
    structural_statistics = calculate_structural_statistics(structural)
    warnings_lines = _collect_warnings(benchmarks)
    versions = installed_step7_versions()

    outputs = step8_output_paths(config)
    if not overwrite:
        collisions = [path for path in outputs.all_paths() if path.exists()]
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step8CollisionError(
                "Existing Step 8 comparison outputs were found; re-run with "
                "--overwrite after review:\n" + listing
            )
    for directory in step8_directories(config):
        directory.mkdir(parents=True, exist_ok=True)

    energy_rows = [_energy_row(record) for record in comparisons]
    structural_rows = [_structural_row(record) for record in structural]
    generated_at = utc_timestamp()
    energy_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_vs_mp_dft_benchmark",
        "project_step": "8",
        "generated_at_utc": generated_at,
        "configuration_path": relative_path(
            config.config_path, config.project_root
        ),
        "configuration_fingerprint_sha256": config.fingerprint,
        "step7_configuration_fingerprint_sha256": (
            mace.step7_configuration_fingerprint
        ),
        "package_versions": dict(versions),
        "mace_model": dict(mace.model),
        "chemical_potentials_eV_per_atom": dict(mace.chemical_potentials),
        "units": {
            "formation_energy": "eV/atom",
            "errors": "eV/atom",
            "volume": "angstrom^3/atom",
        },
        "error_convention": (
            "signed_error = MACE - Materials Project processed DFT; "
            "positive means MACE is less negative."
        ),
        "records": energy_rows,
        "statistics": dict(statistics),
        "energy_above_hull_context": _hull_context_rows(comparisons),
        "structural_summary": dict(structural_statistics),
        "source_fingerprints": [
            snapshot.to_json(config.project_root)
            for snapshot in mace.snapshots
        ],
        "methodological_limitations": [
            *BENCHMARK_LIMITATIONS,
            "Sample size is five compounds; correlation metrics are "
            "exploratory.",
            *NI_MAGNETIC_LIMITATION,
            *SELECTED_SET_LIMITATION,
        ],
    }
    structural_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ni_al_mace_vs_mp_dft_structural_comparison",
        "project_step": "8",
        "generated_at_utc": generated_at,
        "configuration_fingerprint_sha256": config.fingerprint,
        "symmetry_settings": {
            "symprec_A": config.comparison.symmetry_symprec_A,
            "angle_tolerance_deg": (
                config.comparison.symmetry_angle_tolerance_deg
            ),
        },
        "standardization_method": structural[0].standardization_method,
        "records": structural_rows,
        "summary": dict(structural_statistics),
    }
    comparison_report = _comparison_report_text(
        config,
        comparisons,
        statistics,
        structural,
        structural_statistics,
        warnings_lines,
    )

    root = config.result_root
    with tempfile.TemporaryDirectory(
        prefix=".step8-comparison-staging-", dir=root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for target, payload in (
            (
                outputs.energy_csv,
                _csv_bytes(energy_rows, ENERGY_TABLE_FIELDNAMES),
            ),
            (outputs.energy_json, write_strict_json_bytes(energy_document)),
            (
                outputs.structural_csv,
                _csv_bytes(structural_rows, STRUCTURAL_TABLE_FIELDNAMES),
            ),
            (
                outputs.structural_json,
                write_strict_json_bytes(structural_document),
            ),
            (
                outputs.comparison_report,
                comparison_report.encode("utf-8"),
            ),
        ):
            staged = stage_path(staging_root, root, target)
            staged.write_bytes(payload)
            staged_by_final[target] = staged

        figure_targets = {
            "parity": stage_path(staging_root, root, outputs.parity_figure),
            "error": stage_path(staging_root, root, outputs.error_figure),
            "composition": stage_path(
                staging_root, root, outputs.composition_figure
            ),
            "initial_vs_relaxed": stage_path(
                staging_root, root, outputs.initial_vs_relaxed_figure
            ),
            "volume": stage_path(staging_root, root, outputs.volume_figure),
            "volume_error": stage_path(
                staging_root, root, outputs.volume_error_figure
            ),
        }
        _render_figures(config, comparisons, structural, figure_targets)
        staged_by_final[outputs.parity_figure] = figure_targets["parity"]
        staged_by_final[outputs.error_figure] = figure_targets["error"]
        staged_by_final[outputs.composition_figure] = figure_targets[
            "composition"
        ]
        staged_by_final[outputs.initial_vs_relaxed_figure] = figure_targets[
            "initial_vs_relaxed"
        ]
        staged_by_final[outputs.volume_figure] = figure_targets["volume"]
        staged_by_final[outputs.volume_error_figure] = figure_targets[
            "volume_error"
        ]

        status = (
            "SUCCESS"
            if len(comparisons) == len(PHASE_ORDER)
            and len(structural) == len(PHASE_ORDER)
            else "PARTIAL"
        )
        inventory = sorted(
            relative_path(path, config.project_root)
            for path in outputs.all_paths()
        ) + sorted(
            relative_path(path, config.project_root)
            for phase in PHASE_ORDER
            for path in (
                config.raw_benchmark_root / phase / "metadata.json",
                config.raw_benchmark_root / phase / "structure.cif",
                config.raw_benchmark_root / phase / "structure.extxyz",
            )
        )
        final_report = _final_report_text(
            config,
            comparisons,
            statistics,
            structural,
            structural_statistics,
            mace,
            benchmarks,
            warnings_lines,
            inventory,
            status,
        )
        staged = stage_path(staging_root, root, outputs.final_report)
        staged.write_bytes(final_report.encode("utf-8"))
        staged_by_final[outputs.final_report] = staged

        checkpoint_document = {
            **energy_document,
            "artifact_type": "ni_al_step8_benchmark_result",
            "structural_records": structural_rows,
            "warnings": list(warnings_lines),
            "overall_status": status,
            "artifacts": sorted(
                relative_path(path, config.project_root)
                for path in outputs.all_paths()
            ),
        }
        staged = stage_path(staging_root, root, outputs.checkpoint)
        staged.write_bytes(write_strict_json_bytes(checkpoint_document))
        staged_by_final[outputs.checkpoint] = staged

        def final_validator() -> None:
            from step7_utils import verify_snapshots

            verify_snapshots(mace.snapshots)
            reread = read_strict_json(
                outputs.energy_json, "published Step 8 energy JSON"
            )
            if (
                reread.get("configuration_fingerprint_sha256")
                != config.fingerprint
                or len(reread.get("records", ())) != len(PHASE_ORDER)
            ):
                raise Step8Error(
                    "Published Step 8 energy JSON is inconsistent."
                )

        publish_files_transactionally(
            config.project_root,
            root,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )
    assert_mace_not_imported()
    return comparisons, statistics


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.compare:
        LOGGER.error("--overwrite is allowed only with --compare.")
        return 1
    try:
        config = load_step8_config(args.config)
        if args.validate_only:
            run_validate_only(config)
            return 0
        comparisons, statistics = run_compare(config, overwrite=args.overwrite)
        print("=" * 78)
        print("STEP 8 COMPARISON COMPLETED")
        print("=" * 78)
        for record in comparisons:
            print(
                f"{record.phase_key}: MP={record.mp_formation_energy_eV_per_atom:.6f}; "
                f"MACE={record.mace_relaxed_formation_energy_eV_per_atom:.6f}; "
                f"signed error={record.relaxed_signed_error_eV_per_atom:+.6f} "
                "eV/atom"
            )
        print(
            f"MAE={statistics['mean_absolute_error_eV_per_atom']:.6f} eV/atom; "
            f"RMSE={statistics['root_mean_squared_error_eV_per_atom']:.6f} "
            "eV/atom; mean signed="
            f"{statistics['mean_signed_error_eV_per_atom']:+.6f} eV/atom"
        )
        print("=" * 78)
        return 0
    except (Step8Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial comparison bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

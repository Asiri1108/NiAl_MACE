"""Design (but do not execute) the future Step 10 LAMMPS benchmark.

The design describes a fair comparison workflow: every classical potential
processes independent copies of the same original selected Materials
Project structures used by the MACE workflow, calculates its own pure Al
and pure Ni relaxed elemental references, and forms formation energies
only from energies produced by that same potential.  Convergence targets
are mapped from the exact Step 6 MACE criteria, with the stress-threshold
unit conversion from eV/angstrom^3 to bar computed programmatically from
exact SI definitions rather than hard-coded.

No LAMMPS simulation is executed, no atoms are created, no potential file
is read into a calculation, and no scientific energy is produced here.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7_utils import (
    Step7Error,
    file_sha256,
    publish_files_transactionally,
    relative_path,
    stage_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step9_utils import (
    ELEMENT_ORDER,
    EV_JOULE,
    PHASE_ORDER,
    LammpsAvailability,
    Step9CollisionError,
    Step9Config,
    Step9Error,
    Step9InputError,
    ValidatedCandidate,
    ev_per_A3_to_bar,
    inspect_lammps_availability,
    load_step9_config,
    pair_coeff_line,
    step9_result_paths,
    validate_candidate_bundle,
    validate_candidate_keys,
)


LOGGER = logging.getLogger("ni_al_step9.design_benchmark")
DEFAULT_CONFIG = Path("configs/ni_al_classical_potentials.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the benchmark design."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate inputs and create the Step 10 LAMMPS benchmark design "
            "plan without executing any simulation."
        )
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
        help="Validate input sources and the planned benchmark only.",
    )
    action.add_argument(
        "--design",
        action="store_true",
        help="Create the Step 10 plan outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 9 plan outputs.",
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


def _structure_inventory(config: Step9Config) -> list[dict[str, Any]]:
    """Fingerprint the source structures the Step 10 benchmark will use."""

    design = config.step10_design
    structures: list[dict[str, Any]] = []
    elemental_directory = config.project_root / str(
        design["elemental_reference_directory"]
    )
    for element in ELEMENT_ORDER:
        path = elemental_directory / f"{element}.extxyz"
        if not path.is_file():
            raise Step9InputError(
                f"Elemental reference structure is missing: {path}"
            )
        structures.append(
            {
                "key": element,
                "kind": "elemental_reference",
                "path": relative_path(path, config.project_root),
                "sha256": file_sha256(path),
            }
        )
    selected_directory = config.project_root / str(
        design["selected_structure_directory"]
    )
    for phase in PHASE_ORDER:
        path = selected_directory / f"{phase}.extxyz"
        if not path.is_file():
            raise Step9InputError(f"Selected structure is missing: {path}")
        structures.append(
            {
                "key": phase,
                "kind": "compound",
                "path": relative_path(path, config.project_root),
                "sha256": file_sha256(path),
            }
        )
    return structures


def build_step10_plan(
    config: Step9Config,
    validated: Mapping[str, ValidatedCandidate],
    lammps: LammpsAvailability,
) -> dict[str, Any]:
    """Build the complete machine-readable Step 10 benchmark plan."""

    force_threshold = float(config.step10_design["force_threshold_eV_per_A"])
    stress_threshold = float(
        config.step10_design["stress_threshold_eV_per_A3"]
    )
    stress_threshold_bar = ev_per_A3_to_bar(stress_threshold)
    structures = _structure_inventory(config)
    potentials = []
    for key in config.candidate_order:
        bundle = validated[key]
        potentials.append(
            {
                "candidate_key": key,
                "role": bundle.spec.role,
                "processed_file": relative_path(
                    config.selected_root / key / bundle.spec.expected_filename,
                    config.project_root,
                ),
                "sha256": bundle.raw_sha256,
                "file_element_order": list(bundle.setfl.elements),
                "cutoff_A": bundle.setfl.cutoff_A,
                "pair_style": bundle.spec.pair_style,
                "pair_coeff": pair_coeff_line(bundle.spec, config),
                "known_warnings": list(bundle.spec.known_warnings),
            }
        )
    return {
        "schema_version": "1.0",
        "artifact_type": "ni_al_step10_lammps_benchmark_plan",
        "project_step": "9",
        "generated_at_utc": utc_timestamp(),
        "configuration_fingerprint_sha256": config.fingerprint,
        "objective": (
            "Fair static comparison of the validated classical Ni-Al EAM "
            "potentials against the MACE and Materials Project DFT results "
            "for the five selected compounds plus pure Al and pure Ni, using "
            "identical starting structures and per-potential elemental "
            "references."
        ),
        "execution_status": (
            "DESIGN ONLY: no LAMMPS simulation was executed in Step 9."
        ),
        "lammps_availability": {
            "status": lammps.status,
            "executable_path": lammps.executable_path,
            "version_line": lammps.version_line,
            "eam_alloy_listed": lammps.eam_alloy_listed,
            "python_lammps_package_present": (
                lammps.python_lammps_package_present
            ),
            "detail": lammps.detail,
            "step10_readiness_note": (
                "LAMMPS absence does not affect Step 9 selection but leaves "
                "Step 10 execution readiness incomplete until a LAMMPS build "
                "with the MANYBODY package (eam/alloy) is provided."
            ),
        },
        "global_settings": {
            "units": config.lammps_design.units,
            "atom_style": config.lammps_design.atom_style,
            "boundary": list(config.lammps_design.boundary),
            "pair_style": config.lammps_design.pair_style,
            "atom_type_mapping": dict(config.lammps_design.atom_type_mapping),
            "precision_note": (
                "Use a double-precision LAMMPS build (the standard "
                "distribution default) for the static benchmark."
            ),
            "neighbor_note": (
                "Default metal-units neighbor settings (e.g. 'neighbor 2.0 "
                "bin' with 'neigh_modify every 1 delay 0 check yes') are "
                "appropriate for these EAM cutoffs (~5.7-6.3 A)."
            ),
        },
        "structures": structures,
        "potentials": potentials,
        "structure_conversion_plan": {
            "tool": "ASE public API (ase.io.read EXTXYZ -> ase.io.write "
            "lammps-data) or pymatgen LammpsData; one documented converter "
            "for all structures",
            "requirements": [
                "preserve triclinic cells exactly (LAMMPS restricted "
                "triclinic form via the standard rotation; never manually "
                "approximated)",
                "preserve scaled coordinates and atom ordering",
                "preserve species and periodicity",
                "map Al to atom type 1 and Ni to atom type 2",
                "validate volume before and after conversion",
                "validate composition and atom count",
                "validate Cartesian coordinates modulo periodic wrapping",
                "record conversion SHA-256 fingerprints for every data file",
            ],
            "note": (
                "No Step 10 production data files were written in Step 9."
            ),
        },
        "workflow_per_potential": [
            "1. Convert the seven source structures (Al, Ni, five compounds) "
            "to validated LAMMPS data files; independent copies per "
            "potential and phase.",
            "2. Initial fixed-geometry single point: record energy, forces, "
            "pressure tensor, and volume before any motion.",
            "3. Stage A minimization: atomic positions at fixed cell "
            "(minimize with the cell unchanged).",
            "4. Stage B minimization: atomic positions plus cell at zero "
            "target pressure using 'fix box/relax tri 0.0' (or aniso for "
            "orthogonal cells), allowing triclinic cells where needed; the "
            "primary classical result is the full-cell relaxed state.",
            "5. Record per-stage energies, maximum force, pressure tensor, "
            "volume, minimizer termination reason, and wall time.",
            "6. Verify convergence independently from the recorded maximum "
            "force and pressure components; LAMMPS minimizer termination "
            "alone is never treated as scientific convergence.",
            "7. Calculate mu_Al_P and mu_Ni_P from that potential's own "
            "relaxed pure-element cells; then formation energies for the "
            "five compounds using only that potential's references.",
            "8. Compare against Materials Project DFT and MACE formation "
            "energies, volumes, symmetry, convergence, and computational "
            "cost; never rank raw total energies across compositions.",
        ],
        "proposed_lammps_template": [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "read_data <structure>.lmp",
            "pair_style eam/alloy",
            "pair_coeff * * <potential-file> Al Ni",
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            "thermo 1",
            "thermo_style custom step pe fmax press pxx pyy pzz pxy pxz pyz "
            "vol",
            "min_style cg",
            "# Stage A: atomic positions at fixed cell",
            "minimize 0.0 <ftol> <maxiter> <maxeval>",
            "# Stage B: full-cell relaxation at zero target pressure",
            "fix boxrelax all box/relax tri 0.0",
            "minimize 0.0 <ftol> <maxiter> <maxeval>",
            "unfix boxrelax",
        ],
        "convergence_mapping": {
            "mace_force_threshold_eV_per_A": force_threshold,
            "mace_stress_threshold_eV_per_A3": stress_threshold,
            "lammps_force_threshold_eV_per_A": force_threshold,
            "stress_unit_conversion": {
                "eV_exact_J": EV_JOULE,
                "definition": (
                    "1 eV/angstrom^3 = 1.602176634e-19 J / 1e-30 m^3 = "
                    "1.602176634e11 Pa = 1.602176634e6 bar (exact SI "
                    "definitions)."
                ),
                "factor_bar_per_eV_per_A3": ev_per_A3_to_bar(1.0),
                "mace_stress_threshold_bar": stress_threshold_bar,
            },
            "proposed_lammps_targets": {
                "maximum_force_eV_per_A": force_threshold,
                "maximum_absolute_pressure_component_bar": (
                    stress_threshold_bar
                ),
                "note": (
                    "The minimizer force tolerance <ftol> should be set at "
                    "or below 0.01 eV/angstrom (LAMMPS metal units use "
                    "eV/angstrom natively), and convergence must be "
                    "verified afterwards from the recorded maximum force "
                    "and the six pressure components against these targets; "
                    "energy tolerance is set to 0.0 so the force criterion "
                    "governs, mirroring the Step 6 protocol."
                ),
            },
            "additional_checks": [
                "maximum force",
                "maximum absolute stress/pressure components",
                "energy convergence between final iterations",
                "minimizer termination reason",
                "cell determinant positivity",
                "volume change versus the 25% safety limit",
                "atom identity and count preservation",
                "nonfinite-value rejection",
            ],
        },
        "formation_energy_design": {
            "equations": [
                "mu_Al_P = E(relaxed pure Al with potential P) / N_Al_atoms",
                "mu_Ni_P = E(relaxed pure Ni with potential P) / N_Ni_atoms",
                "E_form_P = (E_compound_P - N_Al*mu_Al_P - N_Ni*mu_Ni_P) / "
                "(N_Al + N_Ni)",
            ],
            "consistency_rules": [
                "Use actual simulation-cell atom counts and verify the "
                "formula-unit route agrees within 1e-12 eV/atom.",
                "Never mix compound energies from one potential with "
                "elemental references from another potential.",
                "Never use MACE chemical potentials or Materials Project "
                "elemental total energies in classical calculations.",
                "Never compare raw classical total energies across "
                "compositions or across potentials.",
            ],
        },
        "limitations": [
            "This is a design document; no LAMMPS result exists yet.",
            "The 2002 candidate carries a documented warning about poor "
            "pure-element behavior that will directly affect its formation "
            "energies.",
            "The 2004 candidate was fitted primarily for gamma/gamma-prime "
            "compositions; transferability across all five phases is part "
            "of what Step 10 measures.",
            "Minimizer tolerances are proposals tied to the Step 6 MACE "
            "criteria and must be recorded verbatim by the Step 10 runner.",
        ],
    }


def _plan_text(plan: Mapping[str, Any]) -> str:
    """Render the human-readable Step 10 plan."""

    lines = [
        "Step 10 LAMMPS Benchmark Plan (designed in Step 9; not executed)",
        "=" * 76,
        "",
        "Objective",
        "---------",
        str(plan["objective"]),
        "",
        f"Generated (UTC): {plan['generated_at_utc']}",
        f"Configuration SHA-256: {plan['configuration_fingerprint_sha256']}",
        str(plan["execution_status"]),
        "",
        "LAMMPS availability",
        "-------------------",
        f"Status: {plan['lammps_availability']['status']}",
        f"Detail: {plan['lammps_availability']['detail']}",
        str(plan["lammps_availability"]["step10_readiness_note"]),
        "",
        "Global settings",
        "---------------",
        f"units {plan['global_settings']['units']}; atom_style "
        f"{plan['global_settings']['atom_style']}; boundary "
        f"{' '.join(plan['global_settings']['boundary'])}; pair_style "
        f"{plan['global_settings']['pair_style']}; atom type 1 = Al, atom "
        "type 2 = Ni",
        "",
        "Potentials",
        "----------",
    ]
    for potential in plan["potentials"]:
        lines.extend(
            [
                f"{potential['candidate_key']} ({potential['role']}):",
                f"  file: {potential['processed_file']}",
                f"  sha256: {potential['sha256']}",
                "  file element order: "
                + " ".join(potential["file_element_order"]),
                f"  cutoff: {potential['cutoff_A']:.6f} A",
                f"  command: pair_style {potential['pair_style']}; "
                f"{potential['pair_coeff']}",
            ]
        )
    lines.extend(
        [
            "",
            "Structures (identical starting points for every potential)",
            "----------------------------------------------------------",
        ]
    )
    for structure in plan["structures"]:
        lines.append(
            f"- {structure['key']} ({structure['kind']}): "
            f"{structure['path']} (sha256 {structure['sha256'][:16]}...)"
        )
    lines.extend(
        [
            "",
            "Workflow per potential",
            "----------------------",
            *plan["workflow_per_potential"],
            "",
            "Proposed LAMMPS command template",
            "--------------------------------",
            *(f"  {line}" for line in plan["proposed_lammps_template"]),
            "",
            "Convergence mapping",
            "-------------------",
            "MACE force threshold: "
            f"{plan['convergence_mapping']['mace_force_threshold_eV_per_A']} "
            "eV/angstrom (used directly; metal units are eV/angstrom)",
            "MACE stress threshold: "
            f"{plan['convergence_mapping']['mace_stress_threshold_eV_per_A3']}"
            " eV/angstrom^3 = "
            f"{plan['convergence_mapping']['stress_unit_conversion']['mace_stress_threshold_bar']:.9f}"
            " bar (conversion computed from exact SI definitions: "
            f"{plan['convergence_mapping']['stress_unit_conversion']['factor_bar_per_eV_per_A3']:.9g}"
            " bar per eV/angstrom^3)",
            str(
                plan["convergence_mapping"]["proposed_lammps_targets"]["note"]
            ),
            "Independent checks: "
            + "; ".join(plan["convergence_mapping"]["additional_checks"]),
            "",
            "Formation-energy consistency",
            "----------------------------",
            *plan["formation_energy_design"]["equations"],
            *(
                f"- {rule}"
                for rule in plan["formation_energy_design"][
                    "consistency_rules"
                ]
            ),
            "",
            "Structure conversion",
            "--------------------",
            f"Tool: {plan['structure_conversion_plan']['tool']}",
            *(
                f"- {item}"
                for item in plan["structure_conversion_plan"]["requirements"]
            ),
            str(plan["structure_conversion_plan"]["note"]),
            "",
            "Limitations",
            "-----------",
            *(f"- {item}" for item in plan["limitations"]),
            "",
        ]
    )
    return "\n".join(lines)


def run_validate_only(config: Step9Config) -> None:
    """Validate design inputs without writing plan outputs."""

    validated = {
        key: validate_candidate_bundle(config, key)
        for key in validate_candidate_keys(None)
    }
    structures = _structure_inventory(config)
    lammps = inspect_lammps_availability(config)
    paths = step9_result_paths(config)
    collisions = [
        path
        for path in (paths["plan_json"], paths["plan_txt"])
        if path.exists()
    ]
    print("=" * 78)
    print("STEP 9 BENCHMARK-DESIGN VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Validated potential bundles: {len(validated)}")
    print(f"Source structures fingerprinted: {len(structures)}")
    print(f"LAMMPS availability: {lammps.status}")
    print(
        "Existing plan outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("LAMMPS simulation executed: No")
    print("Plan outputs written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_design(config: Step9Config, *, overwrite: bool) -> dict[str, Any]:
    """Create and transactionally publish the Step 10 plan outputs."""

    validated = {
        key: validate_candidate_bundle(config, key)
        for key in validate_candidate_keys(None)
    }
    lammps = inspect_lammps_availability(config)
    plan = build_step10_plan(config, validated, lammps)
    paths = step9_result_paths(config)
    targets = (paths["plan_json"], paths["plan_txt"])
    if not overwrite:
        collisions = [path for path in targets if path.exists()]
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step9CollisionError(
                "Existing Step 9 plan outputs were found; re-run with "
                "--overwrite after review:\n" + listing
            )
    root = config.result_root
    for directory in (root, root / "plans"):
        directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".step9-plan-staging-", dir=root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final = {
            paths["plan_json"]: stage_path(
                staging_root, root, paths["plan_json"]
            ),
            paths["plan_txt"]: stage_path(
                staging_root, root, paths["plan_txt"]
            ),
        }
        staged_by_final[paths["plan_json"]].write_bytes(
            write_strict_json_bytes(plan)
        )
        staged_by_final[paths["plan_txt"]].write_text(
            _plan_text(plan), encoding="utf-8", newline="\n"
        )
        publish_files_transactionally(
            config.project_root,
            root,
            staged_by_final,
            overwrite=overwrite,
        )
    LOGGER.info(
        "Step 10 plan published: %s",
        relative_path(paths["plan_json"], config.project_root),
    )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.design:
        LOGGER.error("--overwrite is allowed only with --design.")
        return 1
    try:
        config = load_step9_config(args.config)
        if args.validate_only:
            run_validate_only(config)
            return 0
        plan = run_design(config, overwrite=args.overwrite)
        print("=" * 78)
        print("STEP 10 BENCHMARK PLAN CREATED (design only; nothing executed)")
        print("=" * 78)
        print(f"LAMMPS availability: {plan['lammps_availability']['status']}")
        print(
            "Stress threshold mapping: "
            f"{plan['convergence_mapping']['mace_stress_threshold_eV_per_A3']}"
            " eV/angstrom^3 = "
            f"{plan['convergence_mapping']['stress_unit_conversion']['mace_stress_threshold_bar']:.6f}"
            " bar"
        )
        print("=" * 78)
        return 0
    except (Step9Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial plan bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

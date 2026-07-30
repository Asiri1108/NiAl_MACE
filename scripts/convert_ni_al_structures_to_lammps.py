"""Convert the seven original Ni-Al EXTXYZ structures to LAMMPS data files.

The conversion uses the installed ASE public API with one deterministic
convention: ``units='metal'``, ``atom_style='atomic'``,
``specorder=['Al','Ni']`` (atom type 1 = Al, type 2 = Ni, retained even
when one type has zero atoms), ``masses=True``, ``force_skew=True`` (so
every box is written in the restricted-triclinic form, which the Stage B
``fix box/relax tri`` command requires), and ``reduce_cell=False``.

Every converted file is round-trip validated by reading it back through
ASE and checking physical equivalence: composition, atom count, ordered
species, volume, lattice metric, wrapped fractional positions under the
documented ASE Prism transformation, and all-pairs minimum-image
distances.  Raw cell matrices are deliberately not required to be
textually identical after the valid rotation into the LAMMPS
restricted-triclinic representation.
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
    utc_timestamp,
    write_strict_json_bytes,
)
from step10_utils import (
    STRUCTURE_ORDER,
    Step10CollisionError,
    Step10Config,
    Step10ConversionError,
    Step10DependencyError,
    Step10Error,
    conversion_paths,
    load_source_structure,
    load_step10_config,
    validate_selection,
)


LOGGER = logging.getLogger("ni_al_step10.convert")
DEFAULT_CONFIG = Path("configs/ni_al_lammps_benchmark.json")
GEOMETRY_ATOL = 1e-9


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for structure conversion."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert the original selected Ni-Al structures to validated "
            "LAMMPS data files with atom type 1 = Al and type 2 = Ni."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 10 configuration path, repository-relative by default.",
    )
    parser.add_argument(
        "--phase",
        choices=(*STRUCTURE_ORDER, "all"),
        default="all",
        help="Convert one structure or all seven (default: all).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate source structures and conversion APIs without writing "
            "converted data."
        ),
    )
    action.add_argument(
        "--convert",
        action="store_true",
        help="Create the converted structures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 10 converted structure files.",
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


def _requested(option: str) -> tuple[str, ...]:
    """Normalize the --phase option."""

    if option == "all":
        return validate_selection(None, STRUCTURE_ORDER, "structure")
    return validate_selection((option,), STRUCTURE_ORDER, "structure")


def _validate_ase_api() -> None:
    """Validate the installed ASE lammpsdata public API signatures."""

    import inspect

    try:
        from ase.io.lammpsdata import read_lammps_data, write_lammps_data
    except ImportError as exc:
        raise Step10DependencyError(
            f"ASE lammpsdata API is unavailable: {exc}"
        ) from exc
    write_parameters = inspect.signature(write_lammps_data).parameters
    required_write = {
        "specorder",
        "force_skew",
        "masses",
        "units",
        "atom_style",
        "reduce_cell",
    }
    missing = sorted(required_write.difference(write_parameters))
    if missing:
        raise Step10DependencyError(
            f"write_lammps_data is missing required parameters: {missing}"
        )
    read_parameters = inspect.signature(read_lammps_data).parameters
    required_read = {"Z_of_type", "sort_by_id", "units", "atom_style"}
    missing = sorted(required_read.difference(read_parameters))
    if missing:
        raise Step10DependencyError(
            f"read_lammps_data is missing required parameters: {missing}"
        )


def _pair_distance_matrix(atoms: Any) -> Any:
    """Return the all-pairs minimum-image distance matrix."""

    import numpy as np

    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    delta = scaled[:, None, :] - scaled[None, :, :]
    delta = delta - np.round(delta)
    cartesian = delta @ cell
    return np.linalg.norm(cartesian, axis=2)


def _round_trip_checks(
    source_atoms: Any, roundtrip_atoms: Any, key: str
) -> dict[str, Any]:
    """Validate physical equivalence between source and round-trip atoms."""

    import numpy as np

    checks: dict[str, Any] = {}
    if len(source_atoms) != len(roundtrip_atoms):
        raise Step10ConversionError(f"{key}: atom count changed in round trip.")
    source_symbols = tuple(source_atoms.get_chemical_symbols())
    roundtrip_symbols = tuple(roundtrip_atoms.get_chemical_symbols())
    if source_symbols != roundtrip_symbols:
        raise Step10ConversionError(
            f"{key}: ordered species changed in round trip."
        )
    checks["ordered_species_preserved"] = True
    volume_difference = abs(
        float(source_atoms.get_volume()) - float(roundtrip_atoms.get_volume())
    )
    if volume_difference > 1e-8:
        raise Step10ConversionError(
            f"{key}: volume changed by {volume_difference:.3e} A^3."
        )
    checks["volume_difference_A3"] = volume_difference
    length_difference = float(
        np.max(
            np.abs(
                np.asarray(source_atoms.cell.lengths())
                - np.asarray(roundtrip_atoms.cell.lengths())
            )
        )
    )
    angle_difference = float(
        np.max(
            np.abs(
                np.asarray(source_atoms.cell.angles())
                - np.asarray(roundtrip_atoms.cell.angles())
            )
        )
    )
    if length_difference > GEOMETRY_ATOL or angle_difference > 1e-8:
        raise Step10ConversionError(
            f"{key}: lattice metric changed (dl={length_difference:.3e} A, "
            f"dangle={angle_difference:.3e} deg). Raw matrices are allowed "
            "to differ by the Prism rotation, but the metric must match."
        )
    checks["maximum_lattice_length_difference_A"] = length_difference
    checks["maximum_lattice_angle_difference_deg"] = angle_difference
    source_scaled = np.asarray(
        source_atoms.get_scaled_positions(wrap=True), dtype=float
    )
    roundtrip_scaled = np.asarray(
        roundtrip_atoms.get_scaled_positions(wrap=True), dtype=float
    )
    delta = np.abs(source_scaled - roundtrip_scaled)
    delta = np.minimum(delta, 1.0 - delta)
    scaled_difference = float(delta.max()) if delta.size else 0.0
    if scaled_difference > GEOMETRY_ATOL:
        raise Step10ConversionError(
            f"{key}: wrapped fractional positions changed by "
            f"{scaled_difference:.3e}."
        )
    checks["maximum_wrapped_fractional_difference"] = scaled_difference
    distance_difference = float(
        np.max(
            np.abs(
                _pair_distance_matrix(source_atoms)
                - _pair_distance_matrix(roundtrip_atoms)
            )
        )
    )
    if distance_difference > 1e-8:
        raise Step10ConversionError(
            f"{key}: periodic pair distances changed by "
            f"{distance_difference:.3e} A."
        )
    checks["maximum_pair_distance_difference_A"] = distance_difference
    return checks


def convert_structure(
    config: Step10Config, key: str, staged_data: Path
) -> tuple[dict[str, Any], Any]:
    """Convert one structure into a staged data file and validate it."""

    from ase.io.lammpsdata import read_lammps_data, write_lammps_data

    source_atoms, source_snapshot = load_source_structure(config, key)
    with staged_data.open("w", encoding="utf-8", newline="\n") as handle:
        write_lammps_data(
            handle,
            source_atoms,
            specorder=["Al", "Ni"],
            force_skew=True,
            masses=True,
            units="metal",
            atom_style="atomic",
            reduce_cell=False,
        )
    with staged_data.open("r", encoding="utf-8") as handle:
        roundtrip_atoms = read_lammps_data(
            handle,
            Z_of_type={1: 13, 2: 28},
            sort_by_id=True,
            units="metal",
            atom_style="atomic",
        )
    roundtrip_atoms.calc = None
    text = staged_data.read_text(encoding="utf-8")
    if "2 atom types" not in text:
        raise Step10ConversionError(
            f"{key}: converted data file must declare 2 atom types."
        )
    if "xy xz yz" not in text:
        raise Step10ConversionError(
            f"{key}: converted data file must carry the restricted-"
            "triclinic tilt line (force_skew=True)."
        )
    checks = _round_trip_checks(source_atoms, roundtrip_atoms, key)
    spec = config.structures[key]
    symbols = tuple(source_atoms.get_chemical_symbols())
    record = {
        "schema_version": "1.0",
        "artifact_type": "ni_al_lammps_structure_conversion",
        "project_step": "10",
        "structure_key": key,
        "kind": spec.kind,
        "material_id": spec.material_id,
        "generated_at_utc": utc_timestamp(),
        "source_path": relative_path(spec.path, config.project_root),
        "source_sha256": source_snapshot.sha256,
        "data_file_path": relative_path(
            conversion_paths(config, key)[0], config.project_root
        ),
        "data_file_sha256": file_sha256(staged_data),
        "conversion_settings": {
            "tool": "ase.io.lammpsdata.write_lammps_data",
            "units": "metal",
            "atom_style": "atomic",
            "specorder": ["Al", "Ni"],
            "masses": True,
            "force_skew": True,
            "reduce_cell": False,
            "atom_type_mapping": {"1": "Al", "2": "Ni"},
        },
        "read_back_settings": {
            "tool": "ase.io.lammpsdata.read_lammps_data",
            "Z_of_type": {"1": 13, "2": 28},
            "units": "metal",
            "atom_style": "atomic",
            "sort_by_id": True,
        },
        "atom_count": len(source_atoms),
        "al_count": sum(1 for symbol in symbols if symbol == "Al"),
        "ni_count": sum(1 for symbol in symbols if symbol == "Ni"),
        "volume_A3": float(source_atoms.get_volume()),
        "lattice_lengths_A": [
            float(value) for value in source_atoms.cell.lengths()
        ],
        "lattice_angles_deg": [
            float(value) for value in source_atoms.cell.angles()
        ],
        "round_trip_checks": checks,
        "round_trip_status": "PASS",
        "representation_note": (
            "The data file stores the LAMMPS restricted-triclinic "
            "representation produced by the documented ASE Prism rotation; "
            "physical equivalence is validated through metric, fractional-"
            "coordinate, and pair-distance checks rather than textual cell "
            "identity."
        ),
    }
    return record, source_atoms


def _conversion_report_text(record: Mapping[str, Any]) -> str:
    """Render the human-readable conversion report for one structure."""

    checks = record["round_trip_checks"]
    return "\n".join(
        [
            f"Step 10 conversion report - {record['structure_key']}",
            "=" * 64,
            f"Generated (UTC): {record['generated_at_utc']}",
            f"Source: {record['source_path']}",
            f"Source SHA-256: {record['source_sha256']}",
            f"Data file: {record['data_file_path']}",
            f"Data file SHA-256: {record['data_file_sha256']}",
            f"Material ID: {record['material_id']}",
            f"Atoms: {record['atom_count']} "
            f"(Al {record['al_count']}, Ni {record['ni_count']})",
            f"Volume: {record['volume_A3']:.9f} A^3",
            "Convention: units metal; atom_style atomic; "
            "specorder [Al, Ni]; masses true; force_skew true; "
            "reduce_cell false; atom type 1 = Al, type 2 = Ni.",
            "Round-trip checks:",
            f"  volume difference: {checks['volume_difference_A3']:.3e} A^3",
            "  max lattice length difference: "
            f"{checks['maximum_lattice_length_difference_A']:.3e} A",
            "  max lattice angle difference: "
            f"{checks['maximum_lattice_angle_difference_deg']:.3e} deg",
            "  max wrapped fractional difference: "
            f"{checks['maximum_wrapped_fractional_difference']:.3e}",
            "  max periodic pair-distance difference: "
            f"{checks['maximum_pair_distance_difference_A']:.3e} A",
            "Round-trip status: PASS",
            str(record["representation_note"]),
            "",
        ]
    )


def run_validate_only(config: Step10Config, keys: Sequence[str]) -> None:
    """Validate sources and APIs without writing converted data."""

    _validate_ase_api()
    for key in keys:
        load_source_structure(config, key)
    collisions = [
        path
        for key in keys
        for path in conversion_paths(config, key)
        if path.exists()
    ]
    print("=" * 78)
    print("STEP 10 STRUCTURE-CONVERSION VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Source structures validated: {len(keys)} ({', '.join(keys)})")
    print("ASE lammpsdata API: validated")
    print(
        "Existing conversion outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("Structure conversion written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_convert(
    config: Step10Config, keys: Sequence[str], overwrite: bool
) -> None:
    """Convert, round-trip validate, and transactionally publish."""

    _validate_ase_api()
    if not overwrite:
        collisions = [
            path
            for key in keys
            for path in conversion_paths(config, key)
            if path.exists()
        ]
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step10CollisionError(
                "Existing converted structure files were found; re-run with "
                "--overwrite after review:\n" + listing
            )
    data_root = config.project_root / "data"
    with tempfile.TemporaryDirectory(
        prefix=".step10-conversion-", dir=config.project_root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for key in keys:
            data_path, json_path, report_path = conversion_paths(config, key)
            staged_data = staging_root / data_path.resolve().relative_to(
                data_root.resolve()
            )
            staged_data.parent.mkdir(parents=True, exist_ok=True)
            record, _atoms = convert_structure(config, key, staged_data)
            staged_json = staging_root / json_path.resolve().relative_to(
                data_root.resolve()
            )
            staged_json.write_bytes(write_strict_json_bytes(record))
            staged_report = staging_root / report_path.resolve().relative_to(
                data_root.resolve()
            )
            staged_report.write_text(
                _conversion_report_text(record), encoding="utf-8", newline="\n"
            )
            staged_by_final[data_path] = staged_data
            staged_by_final[json_path] = staged_json
            staged_by_final[report_path] = staged_report
            LOGGER.info(
                "%s: converted and round-trip validated (%d atoms).",
                key,
                record["atom_count"],
            )
        publish_files_transactionally(
            config.project_root,
            data_root,
            staged_by_final,
            overwrite=overwrite,
        )
    print("=" * 78)
    print("STEP 10 STRUCTURE CONVERSION COMPLETED")
    print("=" * 78)
    for key in keys:
        data_path, _json_path, _report_path = conversion_paths(config, key)
        print(f"{key}: {relative_path(data_path, config.project_root)}")
    print("All round-trip validations: PASS")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.convert:
        LOGGER.error("--overwrite is allowed only with --convert.")
        return 1
    try:
        config = load_step10_config(args.config)
        keys = _requested(args.phase)
        if args.validate_only:
            run_validate_only(config, keys)
        else:
            run_convert(config, keys, overwrite=args.overwrite)
        return 0
    except (Step10Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial conversion bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

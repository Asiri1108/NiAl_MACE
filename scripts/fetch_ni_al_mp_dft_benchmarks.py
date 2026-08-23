"""Fetch Materials Project DFT-derived benchmark records for Step 8.

The five selected Ni-Al phases are retrieved by exact material ID from the
official public Materials Project summary endpoint, with the thermodynamic
endpoint inspected to document which processed thermodynamic entry type the
summary value corresponds to.  The primary benchmark quantity is the
Materials Project processed ``formation_energy_per_atom``; raw total
energies are never used and MACE totals are never compared against VASP
totals.

API-key policy follows the established Step 4/7 pattern: the key is read
from ``MP_API_KEY`` (optionally populated from the Git-ignored local
``.env``), is never printed, and is never written into logs, metadata, or
reports.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import fetch_ni_al_structures as _step4
from step7_utils import (
    Step7Error,
    file_sha256,
    installed_step7_versions,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step8_utils import (
    BENCHMARK_LIMITATIONS,
    PHASE_ORDER,
    THERMO_TYPE_PREFERENCE,
    Step8ApiError,
    Step8CollisionError,
    Step8Config,
    Step8DependencyError,
    Step8Error,
    benchmark_phase_paths,
    load_step8_config,
    validate_phase_keys,
)


LOGGER = logging.getLogger("ni_al_step8.fetch_benchmarks")
DEFAULT_CONFIG = Path("configs/mace_dft_benchmark.json")
SUMMARY_ENDPOINT_DESCRIPTION = (
    "mp_api.client.MPRester.materials.summary.search (official public "
    "Materials Project summary endpoint)"
)
THERMO_ENDPOINT_DESCRIPTION = (
    "mp_api.client.MPRester.materials.thermo.search (official public "
    "Materials Project thermodynamic endpoint)"
)
SUMMARY_MANDATORY_FIELDS = (
    "material_id",
    "formula_pretty",
    "structure",
    "formation_energy_per_atom",
    "energy_above_hull",
)
SUMMARY_OPTIONAL_FIELDS = (
    "is_stable",
    "deprecated",
    "theoretical",
    "symmetry",
    "nsites",
    "volume",
    "density",
    "last_updated",
)
THERMO_MANDATORY_FIELDS = ("material_id", "thermo_type")
THERMO_OPTIONAL_FIELDS = (
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "energy_type",
    "decomposes_to",
    "last_updated",
)


@dataclass(frozen=True)
class RetrievedBenchmark:
    """One fully retrieved and documented benchmark phase."""

    phase: str
    material_id: str
    formula_pretty: str
    structure: Any
    formation_energy_per_atom_eV: float
    energy_above_hull_eV_per_atom: float
    is_stable: bool | None
    deprecated: bool | None
    theoretical: bool | None
    mp_space_group_symbol: str | None
    mp_space_group_number: int | None
    mp_crystal_system: str | None
    nsites: int | None
    volume_A3: float | None
    density_g_cm3: float | None
    last_updated: Any
    thermo_entries: tuple[Mapping[str, Any], ...]
    matching_thermo_types: tuple[str, ...]
    selected_thermo_type: str | None
    thermo_selection_rationale: str
    retrieval_time_utc: str


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for benchmark retrieval."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Materials Project DFT-derived benchmark records for "
            "the five selected Ni-Al phases by exact material ID."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 8 configuration path, repository-relative by default.",
    )
    parser.add_argument(
        "--phase",
        choices=(*PHASE_ORDER, "all"),
        default="all",
        help="Retrieve one phase or all five (default: all).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration and API-key presence without querying "
            "the network."
        ),
    )
    action.add_argument(
        "--fetch",
        action="store_true",
        help="Perform the actual Materials Project retrieval.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 8 raw benchmark files.",
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


def _requested_phases(option: str) -> tuple[str, ...]:
    """Normalize the --phase option."""

    if option == "all":
        return validate_phase_keys(None)
    return validate_phase_keys((option,))


def _load_api_key_safely(config: Step8Config) -> str:
    """Load MP_API_KEY using the established pattern; never print it."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise Step8DependencyError(f"python-dotenv is unavailable: {exc}") from exc
    try:
        return _step4.load_api_key(config.project_root, load_dotenv)
    except _step4.Step4Error as exc:
        raise Step8ApiError(str(exc)) from exc


def _filter_fields(
    endpoint: Any,
    mandatory: Sequence[str],
    optional: Sequence[str],
    label: str,
) -> list[str]:
    """Validate the endpoint signature and return only supported fields."""

    try:
        signature = inspect.signature(endpoint.search)
    except (TypeError, ValueError) as exc:
        raise Step8ApiError(
            f"Could not inspect the public {label} search method."
        ) from exc
    required_parameters = {"material_ids", "fields", "all_fields"}
    missing = required_parameters.difference(signature.parameters)
    if missing:
        raise Step8ApiError(
            f"The installed {label} search is missing required parameters: "
            + ", ".join(sorted(missing))
        )
    try:
        available = endpoint.available_fields
    except Exception as exc:
        raise Step8ApiError(
            f"Could not inspect available {label} fields."
        ) from exc
    if callable(available):
        available = available()
    available_set = {str(field) for field in available}
    missing_mandatory = set(mandatory).difference(available_set)
    if missing_mandatory:
        raise Step8ApiError(
            f"The installed {label} endpoint does not support mandatory "
            "fields: " + ", ".join(sorted(missing_mandatory))
        )
    unsupported = [field for field in optional if field not in available_set]
    if unsupported:
        LOGGER.debug(
            "Optional %s fields unavailable in this mp-api version: %s",
            label,
            ", ".join(unsupported),
        )
    return [
        field
        for field in (*mandatory, *optional)
        if field in available_set
    ]


def _existing_targets(
    config: Step8Config, phases: Sequence[str]
) -> tuple[Path, ...]:
    """Return existing raw benchmark targets for collision reporting."""

    existing: list[Path] = []
    for phase in phases:
        for path in benchmark_phase_paths(config, phase):
            if path.exists():
                existing.append(path)
    return tuple(existing)


def _sanitize_thermo_entry(document: Any) -> dict[str, Any]:
    """Extract API-key-free provenance from one thermodynamic document."""

    thermo_type = _step4.optional_string(
        _step4.document_value(document, "thermo_type")
    )
    return {
        "thermo_type": thermo_type,
        "energy_type": _step4.optional_string(
            _step4.document_value(document, "energy_type")
        ),
        "formation_energy_per_atom_eV": _step4.optional_finite_float(
            _step4.document_value(document, "formation_energy_per_atom")
        ),
        "energy_above_hull_eV_per_atom": _step4.optional_finite_float(
            _step4.document_value(document, "energy_above_hull")
        ),
        "is_stable": _step4.optional_bool(
            _step4.document_value(document, "is_stable")
        ),
        "decomposes_to": _step4.safely_convert_optional_provenance(
            _step4.document_value(document, "decomposes_to"), "decomposes_to"
        ),
        "last_updated": _step4.safely_convert_optional_provenance(
            _step4.document_value(document, "last_updated"), "last_updated"
        ),
    }


def _select_thermo_type(
    summary_formation_energy: float,
    thermo_entries: Sequence[Mapping[str, Any]],
    phase: str,
) -> tuple[tuple[str, ...], str | None, str]:
    """Resolve which processed thermodynamic entry the summary value uses.

    Incompatible functional schemes (GGA, GGA+U, r2SCAN, mixed) are never
    averaged or silently mixed: the primary benchmark stays the summary
    processed value, and this resolution documents its provenance by exact
    numerical match.
    """

    matches = tuple(
        sorted(
            {
                str(entry["thermo_type"])
                for entry in thermo_entries
                if entry.get("thermo_type") is not None
                and entry.get("formation_energy_per_atom_eV") is not None
                and math.isclose(
                    float(entry["formation_energy_per_atom_eV"]),
                    summary_formation_energy,
                    abs_tol=1e-6,
                    rel_tol=0.0,
                )
            }
        )
    )
    if not matches:
        return (
            (),
            None,
            f"No thermodynamic entry for {phase} numerically matches the "
            "summary processed formation energy; the summary value remains "
            "the documented primary benchmark and the entry type is "
            "recorded as unresolved.",
        )
    if len(matches) == 1:
        return (
            matches,
            matches[0],
            f"Exactly one thermodynamic entry type ({matches[0]}) matches "
            "the summary processed formation energy; it is recorded as the "
            "provenance of the primary benchmark value.",
        )
    for preferred in THERMO_TYPE_PREFERENCE:
        if preferred in matches:
            return (
                matches,
                preferred,
                f"Multiple thermodynamic entry types ({', '.join(matches)}) "
                "match the summary value with numerically identical "
                "formation energies; the documented Materials Project "
                f"preference order selected {preferred}. No incompatible "
                "entries were averaged or mixed.",
            )
    raise Step8ApiError(
        f"Ambiguous thermodynamic entry selection for {phase}: matching "
        f"types {matches} contain no documented preferred scheme; failing "
        "safely for user review."
    )


def _fetch_phase(
    summary_endpoint: Any,
    thermo_endpoint: Any,
    summary_fields: Sequence[str],
    thermo_fields: Sequence[str],
    config: Step8Config,
    phase: str,
    api_key: str,
) -> RetrievedBenchmark:
    """Retrieve and document one benchmark phase by exact material ID."""

    material_id = config.phases[phase]
    retrieval_time = utc_timestamp()
    LOGGER.info("Querying %s (%s)...", phase, material_id)
    try:
        summary_documents = list(
            summary_endpoint.search(
                material_ids=[material_id],
                fields=list(summary_fields),
                all_fields=False,
            )
        )
    except Exception as exc:
        raise Step8ApiError(
            f"Summary retrieval failed for {phase} ({material_id}): "
            f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
        ) from exc
    if len(summary_documents) != 1:
        raise Step8ApiError(
            f"{phase}: expected exactly one summary record for "
            f"{material_id}; received {len(summary_documents)}."
        )
    document = summary_documents[0]
    returned_id = _step4.optional_string(
        _step4.document_value(document, "material_id")
    )
    if returned_id != material_id:
        raise Step8ApiError(
            f"{phase}: returned material ID {returned_id!r} does not match "
            f"the requested {material_id!r}."
        )
    formation = _step4.optional_finite_float(
        _step4.document_value(document, "formation_energy_per_atom")
    )
    hull = _step4.optional_finite_float(
        _step4.document_value(document, "energy_above_hull")
    )
    structure = _step4.document_value(document, "structure")
    if formation is None or hull is None or structure is None:
        raise Step8ApiError(
            f"{phase}: mandatory summary fields are missing "
            "(formation energy, hull energy, or structure)."
        )
    deprecated = _step4.optional_bool(
        _step4.document_value(document, "deprecated")
    )
    if config.require_non_deprecated and deprecated is True:
        raise Step8ApiError(
            f"{phase}: the requested record {material_id} is deprecated."
        )
    symmetry = _step4.document_value(document, "symmetry")

    try:
        thermo_documents = list(
            thermo_endpoint.search(
                material_ids=[material_id],
                fields=list(thermo_fields),
                all_fields=False,
            )
        )
    except Exception as exc:
        raise Step8ApiError(
            f"Thermodynamic retrieval failed for {phase} ({material_id}): "
            f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
        ) from exc
    thermo_entries = tuple(
        _sanitize_thermo_entry(item) for item in thermo_documents
    )
    matching, selected_type, rationale = _select_thermo_type(
        formation, thermo_entries, phase
    )
    LOGGER.info(
        "  %s: summary E_f=%.6f eV/atom; hull=%.6f eV/atom; %d thermo "
        "entr%s; resolved type: %s",
        phase,
        formation,
        hull,
        len(thermo_entries),
        "y" if len(thermo_entries) == 1 else "ies",
        selected_type,
    )
    return RetrievedBenchmark(
        phase=phase,
        material_id=material_id,
        formula_pretty=str(
            _step4.optional_string(
                _step4.document_value(document, "formula_pretty")
            )
        ),
        structure=structure,
        formation_energy_per_atom_eV=formation,
        energy_above_hull_eV_per_atom=hull,
        is_stable=_step4.optional_bool(
            _step4.document_value(document, "is_stable")
        ),
        deprecated=deprecated,
        theoretical=_step4.optional_bool(
            _step4.document_value(document, "theoretical")
        ),
        mp_space_group_symbol=_step4.optional_string(
            getattr(symmetry, "symbol", None) if symmetry is not None else None
        ),
        mp_space_group_number=_step4.optional_integer(
            getattr(symmetry, "number", None) if symmetry is not None else None
        ),
        mp_crystal_system=_step4.optional_string(
            getattr(symmetry, "crystal_system", None)
            if symmetry is not None
            else None
        ),
        nsites=_step4.optional_integer(
            _step4.document_value(document, "nsites")
        ),
        volume_A3=_step4.optional_finite_float(
            _step4.document_value(document, "volume")
        ),
        density_g_cm3=_step4.optional_finite_float(
            _step4.document_value(document, "density")
        ),
        last_updated=_step4.document_value(document, "last_updated"),
        thermo_entries=thermo_entries,
        matching_thermo_types=matching,
        selected_thermo_type=selected_type,
        thermo_selection_rationale=rationale,
        retrieval_time_utc=retrieval_time,
    )


def _step7_database_version(config: Step8Config) -> Mapping[str, Any]:
    """Read the database versions recorded during Step 7 retrieval."""

    try:
        document = read_strict_json(
            config.mace_sources.formation_energy_table,
            "Step 7 formation-energy table",
        )
    except Step7Error as exc:
        raise Step8Error(str(exc)) from exc
    versions = document.get("materials_project_database_version")
    return versions if isinstance(versions, Mapping) else {}


def _stage_phase_outputs(
    staging_root: Path,
    data_root: Path,
    config: Step8Config,
    retrieved: RetrievedBenchmark,
    database_version: str | None,
    version_warning: str | None,
    summary_fields: Sequence[str],
    thermo_fields: Sequence[str],
    versions: Mapping[str, str],
) -> dict[Path, Path]:
    """Stage the metadata, CIF, and EXTXYZ bundle for one phase."""

    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:
        raise Step8DependencyError(
            f"Structure writers are unavailable: {exc}"
        ) from exc
    metadata_final, cif_final, extxyz_final = benchmark_phase_paths(
        config, retrieved.phase
    )
    staged_by_final: dict[Path, Path] = {}
    for final_path in (metadata_final, cif_final, extxyz_final):
        staged = staging_root / final_path.resolve().relative_to(
            data_root.resolve()
        )
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged_by_final[final_path] = staged

    try:
        CifWriter(retrieved.structure).write_file(
            str(staged_by_final[cif_final])
        )
        atoms = AseAtomsAdaptor.get_atoms(retrieved.structure)
    except Exception as exc:
        raise Step8ApiError(
            f"Structure conversion failed for {retrieved.material_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not bool(atoms.pbc.all()) or abs(float(atoms.cell.volume)) == 0.0:
        raise Step8ApiError(
            f"Converted {retrieved.material_id} lost its periodic cell."
        )
    atoms.info.clear()
    atoms.info.update(
        {
            "source": "Materials Project",
            "material_id": retrieved.material_id,
            "phase_key": retrieved.phase,
            "formula_pretty": retrieved.formula_pretty,
            "retrieval_time_utc": retrieved.retrieval_time_utc,
            "step8_role": "dft_benchmark_structure",
            "formation_energy_per_atom_eV": (
                retrieved.formation_energy_per_atom_eV
            ),
            "energy_above_hull_eV_per_atom": (
                retrieved.energy_above_hull_eV_per_atom
            ),
        }
    )
    if database_version is not None:
        atoms.info["materials_project_database_version"] = database_version
    ase_write(str(staged_by_final[extxyz_final]), atoms, format="extxyz")
    frames = ase_read(
        staged_by_final[extxyz_final], index=":", format="extxyz"
    )
    if not isinstance(frames, list) or len(frames) != 1:
        raise Step8ApiError(
            f"Staged {retrieved.phase} benchmark EXTXYZ did not round-trip."
        )
    roundtrip = frames[0]
    volume = float(roundtrip.get_volume())
    atom_count = len(roundtrip)
    if atom_count <= 0 or not math.isfinite(volume) or volume <= 0.0:
        raise Step8ApiError(
            f"Staged {retrieved.phase} benchmark structure is invalid."
        )

    metadata = {
        "schema_version": "1.0",
        "source": "Materials Project",
        "summary_endpoint": SUMMARY_ENDPOINT_DESCRIPTION,
        "thermo_endpoint": THERMO_ENDPOINT_DESCRIPTION,
        "retrieval_time_utc": retrieved.retrieval_time_utc,
        "phase": retrieved.phase,
        "material_id": retrieved.material_id,
        "formula_pretty": retrieved.formula_pretty,
        "composition": {
            element: sum(
                1
                for symbol in roundtrip.get_chemical_symbols()
                if symbol == element
            )
            for element in ("Al", "Ni")
        },
        "number_of_sites": atom_count,
        "formation_energy_per_atom_eV": (
            retrieved.formation_energy_per_atom_eV
        ),
        "formation_energy_definition": (
            "Materials Project processed formation_energy_per_atom from the "
            "summary endpoint; not a raw uncorrected DFT total-energy "
            "difference computed here."
        ),
        "energy_above_hull_eV_per_atom": (
            retrieved.energy_above_hull_eV_per_atom
        ),
        "is_stable": retrieved.is_stable,
        "deprecated": retrieved.deprecated,
        "theoretical": retrieved.theoretical,
        "selected_thermo_type": retrieved.selected_thermo_type,
        "matching_thermo_types": list(retrieved.matching_thermo_types),
        "thermo_selection_rationale": retrieved.thermo_selection_rationale,
        "thermo_entries": [dict(entry) for entry in retrieved.thermo_entries],
        "materials_project_database_version": database_version,
        "database_version_warning": version_warning,
        "mp_api_version": versions.get("mp-api"),
        "requested_summary_fields": list(summary_fields),
        "requested_thermo_fields": list(thermo_fields),
        "mp_space_group_symbol": retrieved.mp_space_group_symbol,
        "mp_space_group_number": retrieved.mp_space_group_number,
        "mp_crystal_system": retrieved.mp_crystal_system,
        "space_group_symbol": retrieved.mp_space_group_symbol,
        "space_group_number": retrieved.mp_space_group_number,
        "mp_reported_nsites": retrieved.nsites,
        "mp_reported_volume_A3": retrieved.volume_A3,
        "volume_A3": volume,
        "volume_per_atom_A3": volume / atom_count,
        "density_g_cm3": retrieved.density_g_cm3,
        "last_updated": _step4.safely_convert_optional_provenance(
            retrieved.last_updated, "last_updated"
        ),
        "structure_cif_sha256": file_sha256(staged_by_final[cif_final]),
        "structure_extxyz_sha256": file_sha256(
            staged_by_final[extxyz_final]
        ),
        "benchmark_limitation_statement": list(BENCHMARK_LIMITATIONS),
    }
    staged_by_final[metadata_final].write_bytes(
        write_strict_json_bytes(metadata)
    )
    return staged_by_final


def run_validate_only(config: Step8Config, phases: Sequence[str]) -> None:
    """Validate configuration and API readiness without any retrieval."""

    versions = installed_step7_versions()
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise Step8DependencyError(
            f"The official mp-api client is unavailable: {exc}"
        ) from exc
    constructor = inspect.signature(MPRester)
    if "api_key" not in constructor.parameters:
        raise Step8DependencyError(
            "Installed MPRester does not accept an api_key parameter."
        )
    _load_api_key_safely(config)
    collisions = _existing_targets(config, phases)
    print("=" * 78)
    print("STEP 8 BENCHMARK RETRIEVAL VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Phases: {', '.join(phases)}")
    print(
        "Material IDs: "
        + ", ".join(f"{phase}={config.phases[phase]}" for phase in phases)
    )
    print(f"mp-api version: {versions['mp-api']}")
    print("API key variable available: Yes (value not printed)")
    print(
        "Existing raw benchmark outputs: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("Materials Project query executed: No")
    print("Files downloaded or written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_fetch(
    config: Step8Config,
    phases: Sequence[str],
    overwrite: bool,
) -> tuple[RetrievedBenchmark, ...]:
    """Perform the retrieval, staging, and transactional publication."""

    versions = installed_step7_versions()
    api_key = _load_api_key_safely(config)
    if not overwrite:
        collisions = _existing_targets(config, phases)
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step8CollisionError(
                "Existing Step 8 raw benchmark outputs were found; re-run "
                "with --overwrite after review:\n" + listing
            )
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise Step8DependencyError(
            f"The official mp-api client is unavailable: {exc}"
        ) from exc

    step7_versions = _step7_database_version(config)
    data_root = config.project_root / "data"
    retrieved_records: list[RetrievedBenchmark] = []
    with tempfile.TemporaryDirectory(
        prefix=".step8-benchmark-retrieval-", dir=config.project_root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        try:
            rester_context = MPRester(api_key=api_key, mute_progress_bars=True)
        except Exception as exc:
            raise Step8ApiError(
                "Failed to initialize the Materials Project client: "
                f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
            ) from exc
        try:
            with rester_context as mpr:
                summary_endpoint = mpr.materials.summary
                thermo_endpoint = mpr.materials.thermo
                summary_fields = _filter_fields(
                    summary_endpoint,
                    SUMMARY_MANDATORY_FIELDS,
                    SUMMARY_OPTIONAL_FIELDS,
                    "summary",
                )
                thermo_fields = _filter_fields(
                    thermo_endpoint,
                    THERMO_MANDATORY_FIELDS,
                    THERMO_OPTIONAL_FIELDS,
                    "thermo",
                )
                database_version: str | None
                try:
                    database_version = str(getattr(mpr, "db_version"))
                except Exception:
                    database_version = None
                    LOGGER.warning(
                        "Materials Project database version was unavailable."
                    )
                version_warning: str | None = None
                distinct_step7 = {
                    str(value)
                    for value in step7_versions.values()
                    if value is not None
                }
                if (
                    database_version is not None
                    and distinct_step7
                    and {database_version} != distinct_step7
                ):
                    version_warning = (
                        "Benchmark-provenance warning: the current Materials "
                        f"Project database version {database_version} differs "
                        "from the Step 7 retrieval version(s) "
                        f"{sorted(distinct_step7)}. The requested material "
                        "IDs still resolved correctly; both versions are "
                        "recorded and Step 7 results were not modified."
                    )
                    LOGGER.warning("%s", version_warning)
                staged_by_final: dict[Path, Path] = {}
                for phase in phases:
                    retrieved = _fetch_phase(
                        summary_endpoint,
                        thermo_endpoint,
                        summary_fields,
                        thermo_fields,
                        config,
                        phase,
                        api_key,
                    )
                    staged_by_final.update(
                        _stage_phase_outputs(
                            staging_root,
                            data_root,
                            config,
                            retrieved,
                            database_version,
                            version_warning,
                            summary_fields,
                            thermo_fields,
                            versions,
                        )
                    )
                    retrieved_records.append(retrieved)
        except Step8Error:
            raise
        except Exception as exc:
            raise Step8ApiError(
                "Failed while communicating with Materials Project: "
                f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
            ) from exc

        def final_validator() -> None:
            for retrieved in retrieved_records:
                metadata_path, _cif, extxyz_path = benchmark_phase_paths(
                    config, retrieved.phase
                )
                metadata = read_strict_json(
                    metadata_path, f"published {retrieved.phase} metadata"
                )
                if metadata.get("structure_extxyz_sha256") != file_sha256(
                    extxyz_path
                ):
                    raise Step8ApiError(
                        f"Published {retrieved.phase} benchmark EXTXYZ hash "
                        "does not match its metadata."
                    )

        publish_files_transactionally(
            config.project_root,
            data_root,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )

    print("=" * 78)
    print("STEP 8 BENCHMARK RETRIEVAL COMPLETED")
    print("=" * 78)
    for retrieved in retrieved_records:
        print(
            f"{retrieved.phase}: {retrieved.material_id}; "
            f"E_f={retrieved.formation_energy_per_atom_eV:.6f} eV/atom; "
            f"hull={retrieved.energy_above_hull_eV_per_atom:.6f} eV/atom; "
            f"stable={retrieved.is_stable}; "
            f"thermo_type={retrieved.selected_thermo_type}"
        )
    print("API key printed or stored: No")
    print("=" * 78)
    return tuple(retrieved_records)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.fetch:
        LOGGER.error("--overwrite is allowed only with --fetch.")
        return 1
    try:
        config = load_step8_config(args.config)
        phases = _requested_phases(args.phase)
        if args.validate_only:
            run_validate_only(config, phases)
        else:
            run_fetch(config, phases, overwrite=args.overwrite)
        return 0
    except (Step8Error, Step7Error, _step4.Step4Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial benchmark bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

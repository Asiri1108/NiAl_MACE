"""Fetch and validate pure Al and pure Ni elemental reference structures.

Step 7 uses Materials Project only to obtain and document the elemental
crystal structures.  Materials Project DFT energies are never used as MACE
reference energies; MACE calculates every reference energy later in
``run_ni_al_mace_elemental_references.py``.

The command follows the established Step 4 security pattern: the API key is
read from the ``MP_API_KEY`` environment variable (optionally populated from
the Git-ignored local ``.env``), is never printed, and is never written into
logs, metadata, or reports.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import fetch_ni_al_structures as _step4
from step7_utils import (
    ELEMENT_ORDER,
    Step7ApiError,
    Step7CollisionError,
    Step7Config,
    Step7DependencyError,
    Step7Error,
    Step7InputError,
    analyze_symmetry,
    atomic_write_text,
    file_sha256,
    installed_step7_versions,
    load_step7_config,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    selected_elemental_paths,
    utc_timestamp,
    validate_element_keys,
    write_strict_json_bytes,
)


LOGGER = logging.getLogger("ni_al_step7.fetch_elemental")
DEFAULT_CONFIG = Path("configs/mace_formation_energy.json")
SOURCE_ENDPOINT_DESCRIPTION = (
    "mp_api.client.MPRester.materials.summary.search (official public "
    "Materials Project summary endpoint)"
)


@dataclass(frozen=True)
class ElementalCandidate:
    """One sanitized elemental candidate retained from the API response."""

    material_id: str
    formula_pretty: str
    structure: Any
    number_of_sites: int
    energy_above_hull_eV_per_atom: float | None
    formation_energy_per_atom_eV: float | None
    is_stable: bool | None
    theoretical: bool | None
    deprecated: bool | None
    space_group_symbol: str | None
    space_group_number: int | None
    crystal_system: str | None
    volume_A3: float | None
    density_g_cm3: float | None
    last_updated: Any

    def sanitized(self) -> dict[str, Any]:
        """Return API-key-free candidate metadata for records and review."""

        return {
            "material_id": self.material_id,
            "formula_pretty": self.formula_pretty,
            "number_of_sites": self.number_of_sites,
            "energy_above_hull_eV_per_atom": self.energy_above_hull_eV_per_atom,
            "formation_energy_per_atom_eV": self.formation_energy_per_atom_eV,
            "is_stable": self.is_stable,
            "theoretical": self.theoretical,
            "deprecated": self.deprecated,
            "space_group_symbol": self.space_group_symbol,
            "space_group_number": self.space_group_number,
            "crystal_system": self.crystal_system,
            "volume_A3": self.volume_A3,
            "density_g_cm3": self.density_g_cm3,
            "last_updated": _step4.safely_convert_optional_provenance(
                self.last_updated, "last_updated"
            ),
        }


@dataclass(frozen=True)
class ElementalSelection:
    """The deterministic selection outcome for one element."""

    element: str
    selected: ElementalCandidate
    api_documents_returned: int
    valid_candidates: tuple[ElementalCandidate, ...]
    preferred_candidates: tuple[ElementalCandidate, ...]
    selection_rationale: str
    retrieval_time_utc: str
    query_criteria: Mapping[str, Any]
    requested_fields: tuple[str, ...]
    database_version: str | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for elemental-reference retrieval."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve and document the stable FCC pure Al and pure Ni "
            "reference structures from Materials Project for Step 7."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 7 configuration path, repository-relative by default.",
    )
    parser.add_argument(
        "--element",
        choices=(*ELEMENT_ORDER, "all"),
        default="all",
        help="Retrieve one element or both (default: all).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration and API-key availability without "
            "downloading or writing scientific files."
        ),
    )
    action.add_argument(
        "--fetch",
        action="store_true",
        help="Perform the real Materials Project retrieval.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace only Step 7 elemental retrieval outputs.",
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


def _requested_elements(option: str) -> tuple[str, ...]:
    """Normalize the --element option."""

    if option == "all":
        return validate_element_keys(None)
    return validate_element_keys((option,))


def _load_api_key_safely(config: Step7Config) -> str:
    """Load MP_API_KEY using the established Step 4 pattern; never print it."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise Step7DependencyError(
            f"python-dotenv is unavailable: {exc}"
        ) from exc
    try:
        return _step4.load_api_key(config.project_root, load_dotenv)
    except _step4.Step4Error as exc:
        raise Step7ApiError(str(exc)) from exc


def _raw_candidate_directory(
    config: Step7Config, element: str, material_id: str
) -> Path:
    """Resolve one raw candidate bundle directory."""

    return config.input.elemental_raw_root / element / material_id


def _planned_targets(
    config: Step7Config,
    element: str,
    material_ids: Sequence[str],
) -> tuple[Path, ...]:
    """Return every planned retrieval target for one element."""

    targets: list[Path] = []
    for material_id in material_ids:
        raw = _raw_candidate_directory(config, element, material_id)
        targets.extend(
            (raw / "structure.cif", raw / "structure.extxyz", raw / "metadata.json")
        )
    structure_path, metadata_path = selected_elemental_paths(config, element)
    targets.extend((structure_path, metadata_path))
    return tuple(targets)


def _existing_selected_targets(
    config: Step7Config, elements: Sequence[str]
) -> tuple[Path, ...]:
    """Return existing selected and raw targets for collision reporting."""

    existing: list[Path] = []
    for element in elements:
        structure_path, metadata_path = selected_elemental_paths(config, element)
        for path in (structure_path, metadata_path):
            if path.exists():
                existing.append(path)
        raw_root = config.input.elemental_raw_root / element
        if raw_root.is_dir():
            existing.extend(
                sorted(path for path in raw_root.rglob("*") if path.is_file())
            )
    return tuple(existing)


def _candidate_from_document(
    document: Any, element: str
) -> ElementalCandidate | None:
    """Convert one API document into a sanitized validated candidate."""

    material_id = _step4.optional_string(
        _step4.document_value(document, "material_id")
    )
    formula = _step4.optional_string(
        _step4.document_value(document, "formula_pretty")
    )
    structure = _step4.document_value(document, "structure")
    if material_id is None or formula is None or structure is None:
        return None
    symmetry = _step4.document_value(document, "symmetry")
    symbol = _step4.optional_string(
        getattr(symmetry, "symbol", None)
        if symmetry is not None
        else None
    )
    number = _step4.optional_integer(
        getattr(symmetry, "number", None) if symmetry is not None else None
    )
    crystal_system = _step4.optional_string(
        getattr(symmetry, "crystal_system", None)
        if symmetry is not None
        else None
    )
    try:
        number_of_sites = len(structure)
    except TypeError:
        return None
    return ElementalCandidate(
        material_id=material_id,
        formula_pretty=formula,
        structure=structure,
        number_of_sites=int(number_of_sites),
        energy_above_hull_eV_per_atom=_step4.optional_finite_float(
            _step4.document_value(document, "energy_above_hull")
        ),
        formation_energy_per_atom_eV=_step4.optional_finite_float(
            _step4.document_value(document, "formation_energy_per_atom")
        ),
        is_stable=_step4.optional_bool(
            _step4.document_value(document, "is_stable")
        ),
        theoretical=_step4.optional_bool(
            _step4.document_value(document, "theoretical")
        ),
        deprecated=_step4.optional_bool(
            _step4.document_value(document, "deprecated")
        ),
        space_group_symbol=symbol,
        space_group_number=number,
        crystal_system=crystal_system,
        volume_A3=_step4.optional_finite_float(
            _step4.document_value(document, "volume")
        ),
        density_g_cm3=_step4.optional_finite_float(
            _step4.document_value(document, "density")
        ),
        last_updated=_step4.document_value(document, "last_updated"),
    )


def _structure_is_valid_elemental(candidate: ElementalCandidate, element: str) -> bool:
    """Validate the periodic crystal structure of one candidate."""

    structure = candidate.structure
    try:
        composition = structure.composition
        element_amounts = composition.get_el_amt_dict()
        lattice_matrix = structure.lattice.matrix
        volume = float(structure.lattice.volume)
    except Exception:
        return False
    if set(element_amounts) != {element}:
        return False
    if candidate.number_of_sites <= 0:
        return False
    for row in lattice_matrix:
        for value in row:
            if not math.isfinite(float(value)):
                return False
    for site in structure:
        for value in site.coords:
            if not math.isfinite(float(value)):
                return False
    return math.isfinite(volume) and volume > 0.0


def _filter_valid_candidates(
    candidates: Sequence[ElementalCandidate],
    config: Step7Config,
    element: str,
) -> tuple[ElementalCandidate, ...]:
    """Apply the configured elemental validity requirements."""

    settings = config.materials_project
    valid: list[ElementalCandidate] = []
    for candidate in candidates:
        if candidate.deprecated is True:
            continue
        if candidate.formula_pretty != element:
            continue
        if not _structure_is_valid_elemental(candidate, element):
            continue
        if settings.require_stable_elemental_reference and (
            candidate.is_stable is not True
        ):
            continue
        hull = candidate.energy_above_hull_eV_per_atom
        if hull is None or not math.isfinite(hull):
            continue
        if hull > settings.maximum_energy_above_hull_eV_per_atom:
            continue
        valid.append(candidate)
    return tuple(sorted(valid, key=lambda item: item.material_id))


def _select_candidate(
    valid: Sequence[ElementalCandidate],
    config: Step7Config,
    element: str,
) -> tuple[ElementalCandidate, tuple[ElementalCandidate, ...], str]:
    """Select one candidate deterministically or fail safely."""

    settings = config.materials_project
    preferred = tuple(
        candidate
        for candidate in valid
        if candidate.space_group_symbol
        == settings.expected_elemental_space_group_symbol
        and candidate.space_group_number
        == settings.expected_elemental_space_group_number
    )
    if not valid:
        raise Step7ApiError(
            f"No valid stable elemental {element} candidate satisfied the "
            "configured requirements."
        )
    if not preferred:
        listing = "\n".join(
            f"  - {candidate.sanitized()}" for candidate in valid
        )
        raise Step7ApiError(
            f"No stable {element} candidate has the recognized ambient "
            f"{settings.expected_elemental_space_group_symbol} "
            f"({settings.expected_elemental_space_group_number}) reference "
            "structure. Sanitized candidates for user review:\n" + listing
        )
    if len(preferred) == 1:
        selected = preferred[0]
        rationale = (
            f"Exactly one valid, stable, non-deprecated elemental {element} "
            "candidate with energy_above_hull <= "
            f"{settings.maximum_energy_above_hull_eV_per_atom:g} eV/atom and "
            f"the recognized ambient {selected.space_group_symbol} "
            f"({selected.space_group_number}) FCC reference symmetry remained "
            "after deterministic filtering; it was selected without any "
            "tie-breaking. This reproducible project choice documents the "
            "crystal structure only; Materials Project energies are not used "
            "as MACE reference energies."
        )
        return selected, preferred, rationale

    # Multiple preferred candidates: compare energy fields carefully and
    # deterministically. Identical energy evidence is ambiguous and fails.
    def energy_key(candidate: ElementalCandidate) -> tuple[float, float]:
        hull = candidate.energy_above_hull_eV_per_atom
        formation = candidate.formation_energy_per_atom_eV
        return (
            hull if hull is not None else math.inf,
            formation if formation is not None else math.inf,
        )

    ranked = sorted(preferred, key=lambda item: (*energy_key(item), item.material_id))
    best, runner_up = ranked[0], ranked[1]
    if energy_key(best) == energy_key(runner_up):
        listing = "\n".join(
            f"  - {candidate.sanitized()}" for candidate in ranked
        )
        raise Step7ApiError(
            f"Multiple scientifically indistinguishable stable {element} "
            "candidates remained; refusing to resolve the ambiguity by "
            "ordering. Sanitized candidates for user review:\n" + listing
        )
    selected = best
    rationale = (
        f"{len(preferred)} valid stable {element} candidates shared the "
        "preferred symmetry; the candidate with strictly lower "
        "(energy_above_hull, formation_energy_per_atom) evidence was selected "
        "deterministically."
    )
    return selected, preferred, rationale


def _fetch_element_selection(
    endpoint: Any,
    config: Step7Config,
    element: str,
    requested_fields: Sequence[str],
    database_version: str | None,
    api_key: str,
) -> ElementalSelection:
    """Query one elemental system and select its reference deterministically."""

    retrieval_time_utc = utc_timestamp()
    query_criteria = {
        "formula": element,
        "deprecated": False,
        "all_fields": False,
        "fields": list(requested_fields),
    }
    LOGGER.info("Querying Materials Project for elemental %s...", element)
    try:
        documents = list(
            endpoint.search(
                formula=element,
                deprecated=False,
                fields=list(requested_fields),
                all_fields=False,
            )
        )
    except Exception as exc:
        raise Step7ApiError(
            f"Materials Project summary search failed for {element}: "
            f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
        ) from exc
    if not documents:
        raise Step7ApiError(
            f"No current Materials Project results were returned for {element}."
        )
    candidates = [
        candidate
        for candidate in (
            _candidate_from_document(document, element) for document in documents
        )
        if candidate is not None
    ]
    valid = _filter_valid_candidates(candidates, config, element)
    selected, preferred, rationale = _select_candidate(valid, config, element)
    LOGGER.info(
        "  %s: %d documents returned; %d valid; selected %s.",
        element,
        len(documents),
        len(valid),
        selected.material_id,
    )
    return ElementalSelection(
        element=element,
        selected=selected,
        api_documents_returned=len(documents),
        valid_candidates=valid,
        preferred_candidates=preferred,
        selection_rationale=rationale,
        retrieval_time_utc=retrieval_time_utc,
        query_criteria=query_criteria,
        requested_fields=tuple(requested_fields),
        database_version=database_version,
    )


def _structure_to_provenance_atoms(
    selection: ElementalSelection, candidate: ElementalCandidate
) -> Any:
    """Convert one pymatgen Structure to a provenance-annotated Atoms object."""

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise Step7DependencyError(f"pymatgen ASE adaptor is unavailable: {exc}") from exc
    try:
        atoms = AseAtomsAdaptor.get_atoms(candidate.structure)
    except Exception as exc:
        raise Step7ApiError(
            f"Structure conversion failed for {candidate.material_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not bool(atoms.pbc.all()) or abs(float(atoms.cell.volume)) == 0.0:
        raise Step7ApiError(
            f"Converted {candidate.material_id} lost its periodic unit cell."
        )
    atoms.info.clear()
    atoms.info.update(
        {
            "source": "Materials Project",
            "material_id": candidate.material_id,
            "element": selection.element,
            "formula_pretty": candidate.formula_pretty,
            "retrieval_time_utc": selection.retrieval_time_utc,
            "step7_role": "elemental_reference",
        }
    )
    if candidate.energy_above_hull_eV_per_atom is not None:
        atoms.info["energy_above_hull_eV_per_atom"] = (
            candidate.energy_above_hull_eV_per_atom
        )
    if candidate.is_stable is not None:
        atoms.info["is_stable"] = candidate.is_stable
    if candidate.space_group_symbol is not None:
        atoms.info["space_group_symbol"] = candidate.space_group_symbol
    if candidate.space_group_number is not None:
        atoms.info["space_group_number"] = candidate.space_group_number
    if selection.database_version is not None:
        atoms.info["materials_project_database_version"] = (
            selection.database_version
        )
    return atoms


def _write_candidate_bundle(
    staging_root: Path,
    data_root: Path,
    config: Step7Config,
    selection: ElementalSelection,
    candidate: ElementalCandidate,
    staged_by_final: dict[Path, Path],
) -> None:
    """Stage one raw candidate bundle (CIF, EXTXYZ, metadata)."""

    try:
        from ase.io import write as ase_write
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:
        raise Step7DependencyError(f"Structure writers are unavailable: {exc}") from exc
    raw_dir = _raw_candidate_directory(
        config, selection.element, candidate.material_id
    )
    for final_path in (
        raw_dir / "structure.cif",
        raw_dir / "structure.extxyz",
        raw_dir / "metadata.json",
    ):
        staged = staging_root / final_path.resolve().relative_to(data_root.resolve())
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged_by_final[final_path] = staged
    cif_staged = staged_by_final[raw_dir / "structure.cif"]
    extxyz_staged = staged_by_final[raw_dir / "structure.extxyz"]
    metadata_staged = staged_by_final[raw_dir / "metadata.json"]
    try:
        CifWriter(candidate.structure).write_file(str(cif_staged))
    except Exception as exc:
        raise Step7ApiError(
            f"CIF conversion failed for {candidate.material_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    atoms = _structure_to_provenance_atoms(selection, candidate)
    try:
        ase_write(str(extxyz_staged), atoms, format="extxyz")
    except Exception as exc:
        raise Step7ApiError(
            f"EXTXYZ conversion failed for {candidate.material_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    metadata = {
        "schema_version": "1.0",
        "source": "Materials Project",
        "source_endpoint": SOURCE_ENDPOINT_DESCRIPTION,
        "retrieval_time_utc": selection.retrieval_time_utc,
        "element": selection.element,
        "selected_candidate": candidate.material_id
        == selection.selected.material_id,
        "materials_project_database_version": selection.database_version,
        "query_criteria": dict(selection.query_criteria),
        **candidate.sanitized(),
    }
    metadata_staged.write_bytes(write_strict_json_bytes(metadata))


def _stage_element_outputs(
    staging_root: Path,
    data_root: Path,
    config: Step7Config,
    selection: ElementalSelection,
    versions: Mapping[str, str],
) -> dict[Path, Path]:
    """Stage every retrieval output for one element."""

    staged_by_final: dict[Path, Path] = {}
    for candidate in selection.valid_candidates:
        _write_candidate_bundle(
            staging_root, data_root, config, selection, candidate, staged_by_final
        )

    structure_final, metadata_final = selected_elemental_paths(
        config, selection.element
    )
    structure_staged = staging_root / structure_final.resolve().relative_to(
        data_root.resolve()
    )
    metadata_staged = staging_root / metadata_final.resolve().relative_to(
        data_root.resolve()
    )
    structure_staged.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
    except ImportError as exc:
        raise Step7DependencyError(f"ASE I/O is unavailable: {exc}") from exc
    atoms = _structure_to_provenance_atoms(selection, selection.selected)
    ase_write(str(structure_staged), atoms, format="extxyz")

    # Independent read-back validation of the staged selected structure,
    # including the required symmetry confirmation with project tolerances.
    frames = ase_read(structure_staged, index=":", format="extxyz")
    if not isinstance(frames, list) or len(frames) != 1:
        raise Step7ApiError(
            f"Staged selected {selection.element} EXTXYZ did not round-trip."
        )
    roundtrip = frames[0]
    if set(roundtrip.get_chemical_symbols()) != {selection.element}:
        raise Step7ApiError(
            f"Staged selected {selection.element} structure has wrong species."
        )
    symmetry = analyze_symmetry(roundtrip, config)
    expected_symbol = (
        config.materials_project.expected_elemental_space_group_symbol
    )
    expected_number = (
        config.materials_project.expected_elemental_space_group_number
    )
    if (
        symmetry["space_group_symbol"] != expected_symbol
        or symmetry["space_group_number"] != expected_number
    ):
        raise Step7ApiError(
            f"Selected {selection.element} structure analyzed as "
            f"{symmetry['space_group_symbol']} "
            f"({symmetry['space_group_number']}), not the expected "
            f"{expected_symbol} ({expected_number}); failing for user review."
        )

    structure_sha256 = file_sha256(structure_staged)
    selected = selection.selected
    metadata = {
        "schema_version": "1.0",
        "source": "Materials Project",
        "source_endpoint": SOURCE_ENDPOINT_DESCRIPTION,
        "element": selection.element,
        "formula": selected.formula_pretty,
        "material_id": selected.material_id,
        "api_documents_returned": selection.api_documents_returned,
        "valid_candidate_count": len(selection.valid_candidates),
        "preferred_candidate_count": len(selection.preferred_candidates),
        "valid_candidate_ids": [
            candidate.material_id for candidate in selection.valid_candidates
        ],
        "selection_criteria": [
            "exactly one chemical element matching the request",
            "not deprecated",
            "periodic crystal structure with finite lattice and positions",
            "positive volume",
            "is_stable is true",
            "finite energy_above_hull <= "
            f"{config.materials_project.maximum_energy_above_hull_eV_per_atom:g}"
            " eV/atom",
            "preferred ambient elemental reference symmetry "
            f"{expected_symbol} ({expected_number})",
            "deterministic selection; ambiguity fails safely",
        ],
        "selection_rationale": selection.selection_rationale,
        "materials_project_database_version": selection.database_version,
        "mp_api_version": versions.get("mp-api"),
        "retrieval_time_utc": selection.retrieval_time_utc,
        "query_criteria": dict(selection.query_criteria),
        "original_api_fields_used": list(selection.requested_fields),
        "space_group_symbol": selected.space_group_symbol,
        "space_group_number": selected.space_group_number,
        "analyzed_space_group_symbol": symmetry["space_group_symbol"],
        "analyzed_space_group_number": symmetry["space_group_number"],
        "analyzed_crystal_system": symmetry["crystal_system"],
        "symmetry_symprec_A": config.symmetry.symprec_A,
        "symmetry_angle_tolerance_deg": config.symmetry.angle_tolerance_deg,
        "crystal_system": selected.crystal_system,
        "number_of_sites": selected.number_of_sites,
        "energy_above_hull_eV_per_atom": (
            selected.energy_above_hull_eV_per_atom
        ),
        "is_stable": selected.is_stable,
        "theoretical": selected.theoretical,
        "volume_A3": selected.volume_A3,
        "density_g_cm3": selected.density_g_cm3,
        "structure_output_path": relative_path(
            structure_final, config.project_root
        ),
        "raw_bundle_path": relative_path(
            _raw_candidate_directory(
                config, selection.element, selected.material_id
            ),
            config.project_root,
        ),
        "structure_sha256": structure_sha256,
        "energy_usage_note": (
            "Materials Project energies documented here are provenance only; "
            "MACE calculates the Step 7 reference energies."
        ),
    }
    metadata_staged.parent.mkdir(parents=True, exist_ok=True)
    metadata_staged.write_bytes(write_strict_json_bytes(metadata))
    staged_by_final[structure_final] = structure_staged
    staged_by_final[metadata_final] = metadata_staged
    return staged_by_final


def run_validate_only(config: Step7Config, elements: Sequence[str]) -> None:
    """Validate configuration and API readiness without any retrieval."""

    versions = installed_step7_versions()
    try:
        import inspect

        from mp_api.client import MPRester
    except ImportError as exc:
        raise Step7DependencyError(
            f"The official mp-api client is unavailable: {exc}"
        ) from exc
    constructor = inspect.signature(MPRester)
    if "api_key" not in constructor.parameters:
        raise Step7DependencyError(
            "Installed MPRester does not accept an api_key parameter."
        )
    _load_api_key_safely(config)
    collisions = _existing_selected_targets(config, elements)
    print("=" * 78)
    print("STEP 7 ELEMENTAL RETRIEVAL VALIDATION")
    print("=" * 78)
    print(f"Configuration: {relative_path(config.config_path, config.project_root)}")
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Elements: {', '.join(elements)}")
    print(f"mp-api version: {versions['mp-api']}")
    print("API key variable available: Yes (value not printed)")
    print(
        "Existing retrieval outputs: "
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
    config: Step7Config,
    elements: Sequence[str],
    overwrite: bool,
) -> tuple[ElementalSelection, ...]:
    """Perform the real retrieval, staging, and transactional publication."""

    versions = installed_step7_versions()
    api_key = _load_api_key_safely(config)
    if not overwrite:
        collisions = _existing_selected_targets(config, elements)
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step7CollisionError(
                "Existing Step 7 elemental retrieval outputs were found; "
                "re-run with --overwrite after review:\n" + listing
            )
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise Step7DependencyError(
            f"The official mp-api client is unavailable: {exc}"
        ) from exc

    data_root = config.project_root / "data"
    selections: list[ElementalSelection] = []
    with tempfile.TemporaryDirectory(
        prefix=".step7-elemental-retrieval-", dir=config.project_root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        try:
            rester_context = MPRester(api_key=api_key, mute_progress_bars=True)
        except Exception as exc:
            raise Step7ApiError(
                "Failed to initialize the Materials Project client: "
                f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
            ) from exc
        try:
            with rester_context as mpr:
                endpoint = mpr.materials.summary
                try:
                    requested_fields = _step4.public_summary_fields(endpoint)
                except _step4.Step4Error as exc:
                    raise Step7ApiError(str(exc)) from exc
                database_version: str | None
                try:
                    database_version = str(mpr.get_database_version())
                except Exception:
                    database_version = None
                    LOGGER.warning(
                        "Materials Project database version was unavailable."
                    )
                staged_by_final: dict[Path, Path] = {}
                for element in elements:
                    selection = _fetch_element_selection(
                        endpoint,
                        config,
                        element,
                        requested_fields,
                        database_version,
                        api_key,
                    )
                    staged_by_final.update(
                        _stage_element_outputs(
                            staging_root, data_root, config, selection, versions
                        )
                    )
                    selections.append(selection)
        except Step7Error:
            raise
        except Exception as exc:
            raise Step7ApiError(
                "Failed while communicating with Materials Project: "
                f"{type(exc).__name__}: {_step4.safe_error_text(exc, api_key)}"
            ) from exc

        def final_validator() -> None:
            for selection in selections:
                structure_path, metadata_path = selected_elemental_paths(
                    config, selection.element
                )
                metadata = read_strict_json(
                    metadata_path, f"published {selection.element} metadata"
                )
                if metadata.get("structure_sha256") != file_sha256(structure_path):
                    raise Step7InputError(
                        f"Published {selection.element} EXTXYZ hash does not "
                        "match its published metadata."
                    )

        publish_files_transactionally(
            config.project_root,
            data_root,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )

    print("=" * 78)
    print("STEP 7 ELEMENTAL RETRIEVAL COMPLETED")
    print("=" * 78)
    for selection in selections:
        selected = selection.selected
        print(
            f"{selection.element}: {selected.material_id} "
            f"({selected.space_group_symbol} {selected.space_group_number}; "
            f"{selected.number_of_sites} site(s); "
            f"E_hull={selected.energy_above_hull_eV_per_atom!r} eV/atom); "
            f"documents={selection.api_documents_returned}; "
            f"valid={len(selection.valid_candidates)}"
        )
    print(
        "Materials Project database version: "
        + (selections[0].database_version or "unavailable")
    )
    print("API key printed or stored: No")
    print("=" * 78)
    return tuple(selections)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.fetch:
        LOGGER.error("--overwrite is allowed only with --fetch.")
        return 1
    try:
        config = load_step7_config(args.config)
        elements = _requested_elements(args.element)
        if args.validate_only:
            run_validate_only(config, elements)
        else:
            run_fetch(config, elements, overwrite=args.overwrite)
        return 0
    except (Step7Error, _step4.Step4Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial retrieval bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

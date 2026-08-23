"""Download and organize reproducible Ni-Al structures from Materials Project.

This Step 4 workflow queries every current, non-deprecated summary entry for
the configured Ni-Al compositions, verifies exact reduced compositions, saves
all valid candidates, and selects one deterministic working structure per
phase. It does not run MACE, relax structures, create supercells, or perform
LAMMPS calculations.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias


LOGGER = logging.getLogger("fetch_ni_al_structures")

SCHEMA_VERSION = "1.0"
SOURCE_NAME = "Materials Project"
DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/ni_al_phases.json")
RAW_ROOT_RELATIVE_PATH = Path("data/raw/materials_project/ni_al")
PROCESSED_ROOT_RELATIVE_PATH = Path("data/processed/ni_al_structures")
SELECTED_ROOT_RELATIVE_PATH = PROCESSED_ROOT_RELATIVE_PATH / "selected"
CSV_MANIFEST_RELATIVE_PATH = (
    PROCESSED_ROOT_RELATIVE_PATH / "ni_al_phase_manifest.csv"
)
JSON_MANIFEST_RELATIVE_PATH = (
    PROCESSED_ROOT_RELATIVE_PATH / "ni_al_phase_manifest.json"
)

MANDATORY_SUMMARY_FIELDS = ("material_id", "formula_pretty", "structure")
OPTIONAL_SUMMARY_FIELDS = (
    "energy_above_hull",
    "formation_energy_per_atom",
    "is_stable",
    "theoretical",
    "symmetry",
    "volume",
    "density",
    "nsites",
    "database_IDs",
    "origins",
    "last_updated",
    "deprecated",
)

EXPECTED_PHASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Al3Ni": ("Al3Ni", ("Al3Ni",)),
    "Al3Ni2": ("Al3Ni2", ("Al3Ni2",)),
    "AlNi": ("AlNi", ("AlNi", "NiAl")),
    "Al3Ni5": ("Al3Ni5", ("Al3Ni5", "Ni5Al3")),
    "AlNi3": ("AlNi3", ("AlNi3", "Ni3Al")),
}

SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

MANIFEST_FIELDS = (
    "phase_key",
    "query_formula",
    "aliases",
    "material_id",
    "formula_pretty",
    "reduced_composition",
    "number_of_sites",
    "energy_above_hull_eV_per_atom",
    "formation_energy_per_atom_eV",
    "is_stable",
    "theoretical",
    "symmetry_symbol",
    "symmetry_number",
    "crystal_system",
    "volume_A3",
    "density_g_cm3",
    "is_selected",
    "selection_rank",
    "selection_reason",
    "raw_cif_path",
    "raw_extxyz_path",
    "selected_cif_path",
    "selected_extxyz_path",
    "retrieval_time_utc",
)


class Step4Error(RuntimeError):
    """Base class for anticipated Step 4 workflow errors."""


class DependencyError(Step4Error):
    """Raised when the required project environment is unavailable."""


class ConfigurationError(Step4Error):
    """Raised when the phase configuration is missing or invalid."""


class ApiKeyError(Step4Error):
    """Raised when a usable Materials Project API key is unavailable."""


class ApiCompatibilityError(Step4Error):
    """Raised when the installed mp-api lacks required public features."""


class ApiConnectionError(Step4Error):
    """Raised when the Materials Project client cannot be initialized."""


class ApiQueryError(Step4Error):
    """Raised when a Materials Project summary query fails."""


class NoResultsError(Step4Error):
    """Raised when Materials Project returns no documents for a formula."""


class NoExactMatchesError(Step4Error):
    """Raised when returned documents contain no exact target composition."""


class MandatoryFieldError(Step4Error):
    """Raised when a required Materials Project field is absent."""


class StructureConversionError(Step4Error):
    """Raised when a structure cannot be written without modification."""


class SerializationError(Step4Error):
    """Raised when provenance cannot be represented safely as JSON."""


class OutputCollisionError(Step4Error):
    """Raised when an output exists and overwrite was not requested."""


class PartialOutputError(Step4Error):
    """Raised when an existing output bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class ScientificDependencies:
    """References to third-party classes and functions used by the workflow."""

    composition_class: type[Any]
    structure_class: type[Any]
    cif_writer_class: type[Any]
    ase_adaptor_class: type[Any]
    mp_rester_class: type[Any]
    ase_write: Callable[..., None]
    load_dotenv: Callable[..., bool]


@dataclass(frozen=True)
class PhaseDefinition:
    """Validated definition of one target Ni-Al composition."""

    order: int
    phase_key: str
    query_formula: str
    aliases: tuple[str, ...]
    description: str
    is_compound: bool
    is_elemental_reference: bool
    reduced_composition: str
    composition_signature: tuple[tuple[str, float], ...]


@dataclass
class Candidate:
    """Normalized Materials Project summary data for one exact candidate."""

    phase: PhaseDefinition
    material_id: str
    formula_pretty: str
    structure: Any
    reduced_composition: str
    number_of_sites: int
    volume_A3: float | None
    density_g_cm3: float | None
    energy_above_hull_eV_per_atom: float | None
    formation_energy_per_atom_eV: float | None
    is_stable: bool | None
    theoretical: bool | None
    deprecated: bool | None
    symmetry_symbol: str | None
    symmetry_number: int | None
    crystal_system: str | None
    database_ids: JSONValue
    origins: JSONValue
    materials_project_last_updated: str | None
    retrieval_time_utc: str
    selected_candidate: bool = False
    selection_rank: int = 0
    selection_reason: str = ""


@dataclass
class PhaseOutcome:
    """Console-report state for one requested phase."""

    phase: PhaseDefinition
    api_documents_returned: int = 0
    exact_candidates_retained: int = 0
    candidates_saved: int = 0
    selected: Candidate | None = None
    completed: bool = False
    error_type: str | None = None
    error_message: str | None = None


def locate_project_root() -> Path:
    """Locate the repository from this script instead of the shell directory."""

    # The script lives in ``project/scripts``; its second parent is therefore
    # the stable repository root even when the command starts elsewhere.
    return Path(__file__).resolve().parents[1]


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line interface for validation and acquisition."""

    parser = argparse.ArgumentParser(
        description=(
            "Download, validate, and organize Materials Project structures "
            "for the configured Ni-Al intermetallic phases."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Phase configuration path (default: "
            "configs/ni_al_phases.json relative to the repository root)."
        ),
    )
    parser.add_argument(
        "--phase",
        help="Process only this phase key; omit to process every phase.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing downloaded phase files to be replaced.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the environment, configuration, folders, and existing "
            "outputs without connecting to Materials Project."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed workflow logging.",
    )
    return parser.parse_args(arguments)


def configure_logging(verbose: bool) -> None:
    """Configure this script without enabling verbose third-party HTTP logs."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False

    # Verbose mode is intentionally scoped to this workflow. In particular,
    # requests, urllib3, and mp-api must not start emitting HTTP diagnostics
    # that could contain authentication context outside our redaction path.
    logging.getLogger().setLevel(logging.WARNING)


def import_scientific_dependencies() -> ScientificDependencies:
    """Import and return all packages required by validation and acquisition."""

    try:
        from ase.io import write as ase_write
        from dotenv import load_dotenv
        from mp_api.client import MPRester
        from pymatgen.core import Composition, Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.io.cif import CifWriter
    except ImportError as exc:
        missing_name = exc.name or "an unknown package"
        raise DependencyError(
            f"Missing required Python import '{missing_name}'. Install the "
            "Step 4 dependencies with: .\\.venv\\Scripts\\python.exe -m pip "
            "install mp-api python-dotenv"
        ) from exc

    return ScientificDependencies(
        composition_class=Composition,
        structure_class=Structure,
        cif_writer_class=CifWriter,
        ase_adaptor_class=AseAtomsAdaptor,
        mp_rester_class=MPRester,
        ase_write=ase_write,
        load_dotenv=load_dotenv,
    )


def validate_runtime_and_folders(project_root: Path) -> None:
    """Confirm the required project venv and repository folders are in use."""

    expected_venv = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    if os.path.normcase(str(active_prefix)) != os.path.normcase(
        str(expected_venv)
    ):
        raise DependencyError(
            "This script must run with the project virtual environment: "
            ".\\.venv\\Scripts\\python.exe"
        )

    if sys.version_info[:2] != (3, 11):
        raise DependencyError(
            "The project virtual environment must use Python 3.11; the active "
            f"interpreter is Python {sys.version_info.major}.{sys.version_info.minor}."
        )

    # Output roots such as ``data`` may be absent in a fresh Git clone because
    # Git does not track empty directories. They are created only during a real
    # successful publication, not by --validate-only.
    required_directories = ("scripts", "configs", "environment")
    for directory_name in required_directories:
        directory = project_root / directory_name
        if not directory.is_dir():
            raise ConfigurationError(
                f"Required repository directory is missing: {directory}"
            )

    for output_relative_path in (
        RAW_ROOT_RELATIVE_PATH,
        PROCESSED_ROOT_RELATIVE_PATH,
        SELECTED_ROOT_RELATIVE_PATH,
    ):
        output_path = project_root / output_relative_path
        if output_path.exists() and not output_path.is_dir():
            raise ConfigurationError(
                f"Expected an output directory but found a file: {output_path}"
            )


def resolve_config_path(project_root: Path, supplied_path: Path | None) -> Path:
    """Resolve the default config from the repository and custom paths from cwd."""

    if supplied_path is None:
        return project_root / DEFAULT_CONFIG_RELATIVE_PATH
    return supplied_path.expanduser().resolve()


def require_nonempty_string(value: Any, field_name: str) -> str:
    """Return a trimmed string or raise a field-specific configuration error."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def require_safe_component(value: str, field_name: str) -> str:
    """Reject path traversal and separators before values become directory names."""

    if not SAFE_PATH_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ConfigurationError(
            f"'{field_name}' contains an unsafe path component: {value!r}"
        )
    return value


def composition_signature(composition: Any) -> tuple[tuple[str, float], ...]:
    """Create an order-independent signature for a reduced composition."""

    reduced = composition.reduced_composition
    amounts = reduced.get_el_amt_dict()
    return tuple(sorted((str(symbol), float(amount)) for symbol, amount in amounts.items()))


def load_phase_configuration(
    config_path: Path,
    dependencies: ScientificDependencies,
) -> tuple[str, str, list[PhaseDefinition]]:
    """Load JSON and validate the fixed five-phase scientific scope."""

    # Configuration is separated from code so aliases and scientific context
    # remain reviewable without embedding changeable database identifiers.
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in {config_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read configuration file {config_path}: {exc}"
        ) from exc

    if not isinstance(raw_config, dict):
        raise ConfigurationError("The configuration root must be a JSON object.")

    description = require_nonempty_string(
        raw_config.get("description"), "description"
    )
    schema_version = require_nonempty_string(
        raw_config.get("schema_version"), "schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported configuration schema version {schema_version!r}; "
            f"this script supports {SCHEMA_VERSION!r}."
        )
    raw_phases = raw_config.get("phases")
    if not isinstance(raw_phases, list):
        raise ConfigurationError("'phases' must be a JSON array.")
    if len(raw_phases) != len(EXPECTED_PHASES):
        raise ConfigurationError(
            f"The configuration must define exactly {len(EXPECTED_PHASES)} phases."
        )

    phase_definitions: list[PhaseDefinition] = []
    seen_keys: set[str] = set()
    seen_compositions: dict[tuple[tuple[str, float], ...], str] = {}

    for index, raw_phase in enumerate(raw_phases):
        if not isinstance(raw_phase, dict):
            raise ConfigurationError(f"Phase entry {index + 1} must be an object.")

        prefix = f"phases[{index}]"
        phase_key = require_safe_component(
            require_nonempty_string(raw_phase.get("phase_key"), f"{prefix}.phase_key"),
            f"{prefix}.phase_key",
        )
        if phase_key in seen_keys:
            raise ConfigurationError(f"Duplicate phase key: {phase_key}")
        seen_keys.add(phase_key)

        query_formula = require_nonempty_string(
            raw_phase.get("query_formula"), f"{prefix}.query_formula"
        )
        raw_aliases = raw_phase.get("aliases")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise ConfigurationError(
                f"'{prefix}.aliases' must be a non-empty array of strings."
            )
        aliases = tuple(
            require_nonempty_string(alias, f"{prefix}.aliases")
            for alias in raw_aliases
        )
        if len(set(aliases)) != len(aliases):
            raise ConfigurationError(f"Duplicate aliases are defined for {phase_key}.")

        phase_description = require_nonempty_string(
            raw_phase.get("description"), f"{prefix}.description"
        )
        is_compound = raw_phase.get("is_compound")
        is_elemental_reference = raw_phase.get("is_elemental_reference")
        if not isinstance(is_compound, bool) or not isinstance(
            is_elemental_reference, bool
        ):
            raise ConfigurationError(
                f"{phase_key} must define boolean 'is_compound' and "
                "'is_elemental_reference' fields."
            )
        if not is_compound or is_elemental_reference:
            raise ConfigurationError(
                f"{phase_key} must be marked as a compound, not an elemental reference."
            )

        try:
            composition = dependencies.composition_class(query_formula)
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid query formula for {phase_key}: {query_formula!r}"
            ) from exc
        signature = composition_signature(composition)
        if signature in seen_compositions:
            other_phase = seen_compositions[signature]
            raise ConfigurationError(
                "Duplicate reduced compositions are configured for "
                f"{other_phase} and {phase_key}."
            )
        seen_compositions[signature] = phase_key

        expected = EXPECTED_PHASES.get(phase_key)
        if expected is None:
            raise ConfigurationError(
                f"Unexpected phase key {phase_key!r}; expected only the five Ni-Al targets."
            )
        expected_formula, expected_aliases = expected
        if query_formula != expected_formula or aliases != expected_aliases:
            raise ConfigurationError(
                f"{phase_key} must use query formula {expected_formula!r} and aliases "
                f"{list(expected_aliases)!r}."
            )

        phase_definitions.append(
            PhaseDefinition(
                order=index,
                phase_key=phase_key,
                query_formula=query_formula,
                aliases=aliases,
                description=phase_description,
                is_compound=is_compound,
                is_elemental_reference=is_elemental_reference,
                reduced_composition=composition.reduced_formula,
                composition_signature=signature,
            )
        )

    actual_order = [phase.phase_key for phase in phase_definitions]
    expected_order = list(EXPECTED_PHASES)
    if actual_order != expected_order:
        raise ConfigurationError(
            "Phases must remain in the documented stable order: "
            + ", ".join(expected_order)
        )

    return description, schema_version, phase_definitions


def choose_requested_phases(
    phases: Sequence[PhaseDefinition],
    requested_phase_key: str | None,
) -> list[PhaseDefinition]:
    """Select all configured phases or one explicitly requested phase."""

    if requested_phase_key is None:
        return list(phases)
    phase_by_key = {phase.phase_key: phase for phase in phases}
    try:
        return [phase_by_key[requested_phase_key]]
    except KeyError as exc:
        valid_keys = ", ".join(phase_by_key)
        raise ConfigurationError(
            f"Unknown --phase value {requested_phase_key!r}. Valid phase keys: "
            f"{valid_keys}"
        ) from exc


def utc_timestamp() -> str:
    """Return a timezone-aware, second-resolution UTC provenance timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def safe_error_text(exc: BaseException, secret: str | None = None) -> str:
    """Create readable exception text while redacting any in-memory API key."""

    message = str(exc).strip() or "no additional detail was provided"
    if secret:
        message = message.replace(secret, "<redacted>")
    return message


def load_api_key(
    project_root: Path,
    load_dotenv: Callable[..., bool],
) -> str:
    """Load MP_API_KEY from the environment, optionally populated by local .env."""

    # ``override=False`` preserves a key already set as a Windows environment
    # variable; the local .env is only a convenient, Git-ignored fallback.
    dotenv_path = project_root / ".env"
    try:
        load_dotenv(dotenv_path=dotenv_path, override=False)
    except Exception as exc:
        raise ApiKeyError(
            f"Could not load the local .env file ({type(exc).__name__})."
        ) from exc

    api_key = os.environ.get("MP_API_KEY", "").strip()
    placeholder_values = {
        "replace_with_your_materials_project_api_key",
        "your_materials_project_api_key",
        "your_api_key_here",
    }
    if not api_key or api_key.lower() in placeholder_values:
        raise ApiKeyError(
            "MP_API_KEY is missing. Copy .env.example to .env, place the key in "
            ".env, and never commit that file. Configuration validation remains "
            "available with --validate-only."
        )
    return api_key


def to_json_compatible(value: Any, field_name: str) -> JSONValue:
    """Convert supported provenance objects into meaningful JSON-native values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return to_json_compatible(value.value, field_name)
    if isinstance(value, datetime):
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        converted: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(
                    f"{field_name} contains a non-string mapping key of type "
                    f"{type(key).__name__}."
                )
            converted[key] = to_json_compatible(item, f"{field_name}.{key}")
        return converted
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            to_json_compatible(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return to_json_compatible(
            value.model_dump(mode="json"), field_name
        )
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return to_json_compatible(value.as_dict(), field_name)
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_compatible(asdict(value), field_name)
    if hasattr(value, "item") and callable(value.item):
        try:
            scalar_value = value.item()
        except (TypeError, ValueError):
            scalar_value = value
        if scalar_value is not value:
            return to_json_compatible(scalar_value, field_name)
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_json_compatible(value.tolist(), field_name)
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                f"{field_name} could not be converted from {type(value).__name__}."
            ) from exc
    raise SerializationError(
        f"{field_name} has unsupported provenance type {type(value).__name__}."
    )


def safely_convert_optional_provenance(value: Any, field_name: str) -> JSONValue:
    """Serialize an optional structured field, omitting unsafe unsupported forms."""

    if value is None:
        return None
    try:
        return to_json_compatible(value, field_name)
    except SerializationError as exc:
        LOGGER.warning("  Omitting %s: %s", field_name, exc)
        return None


def document_value(document: Any, field_name: str) -> Any:
    """Read a public summary field from either a model or mapping result."""

    if isinstance(document, Mapping):
        return document.get(field_name)
    return getattr(document, field_name, None)


def optional_finite_float(value: Any) -> float | None:
    """Normalize a numeric value, treating missing or non-finite data as absent."""

    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def optional_bool(value: Any) -> bool | None:
    """Return true boolean values without coercing arbitrary API data."""

    return value if isinstance(value, bool) else None


def optional_string(value: Any) -> str | None:
    """Normalize strings and enums without using arbitrary object repr output."""

    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def optional_integer(value: Any) -> int | None:
    """Normalize an integer-valued field while rejecting booleans."""

    if value is None or isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted


def public_summary_fields(summary_endpoint: Any) -> list[str]:
    """Check the installed public endpoint and return only supported fields."""

    try:
        signature = inspect.signature(summary_endpoint.search)
    except (TypeError, ValueError) as exc:
        raise ApiCompatibilityError(
            "Could not inspect the public Materials Project summary search method."
        ) from exc
    required_parameters = {"formula", "deprecated", "fields", "all_fields"}
    missing_parameters = required_parameters.difference(signature.parameters)
    if missing_parameters:
        raise ApiCompatibilityError(
            "The installed mp-api public summary search is missing required "
            "parameters: " + ", ".join(sorted(missing_parameters))
        )

    try:
        available_fields_value = summary_endpoint.available_fields
    except Exception as exc:
        raise ApiCompatibilityError(
            "Could not inspect available Materials Project summary fields."
        ) from exc
    if callable(available_fields_value):
        available_fields_value = available_fields_value()
    if not isinstance(available_fields_value, Sequence) or isinstance(
        available_fields_value, (str, bytes)
    ):
        raise ApiCompatibilityError(
            "The installed mp-api returned an invalid available_fields value."
        )
    available_fields = {str(field) for field in available_fields_value}

    missing_mandatory = set(MANDATORY_SUMMARY_FIELDS).difference(available_fields)
    if missing_mandatory:
        raise MandatoryFieldError(
            "The installed Materials Project summary client does not support "
            "mandatory fields: " + ", ".join(sorted(missing_mandatory))
        )

    requested_fields = [
        field
        for field in (*MANDATORY_SUMMARY_FIELDS, *OPTIONAL_SUMMARY_FIELDS)
        if field in available_fields
    ]
    unsupported_optional = [
        field for field in OPTIONAL_SUMMARY_FIELDS if field not in available_fields
    ]
    if unsupported_optional:
        LOGGER.debug(
            "Optional summary fields unavailable in this mp-api version: %s",
            ", ".join(unsupported_optional),
        )
    LOGGER.debug("Requested summary fields: %s", ", ".join(requested_fields))
    return requested_fields


def query_summary_documents(
    summary_endpoint: Any,
    phase: PhaseDefinition,
    requested_fields: Sequence[str],
) -> list[Any]:
    """Query every current, non-deprecated summary document for one formula."""

    # Formula IDs are deliberately not hard-coded. The supported public search
    # interface is asked for the current set of candidates on every real run.
    try:
        documents = summary_endpoint.search(
            formula=phase.query_formula,
            deprecated=False,
            fields=list(requested_fields),
            all_fields=False,
        )
    except Exception as exc:
        raise ApiQueryError(
            f"Materials Project summary search failed: {type(exc).__name__}: {exc}"
        ) from exc

    result = list(documents)
    if not result:
        raise NoResultsError(
            f"No current Materials Project results were returned for "
            f"{phase.query_formula}."
        )
    return result


def build_exact_candidates(
    documents: Sequence[Any],
    phase: PhaseDefinition,
    retrieval_time_utc: str,
    dependencies: ScientificDependencies,
) -> list[Candidate]:
    """Validate mandatory fields and retain only exact reduced compositions."""

    candidates: list[Candidate] = []
    seen_material_ids: set[str] = set()

    for document_index, document in enumerate(documents, start=1):
        material_id_value = document_value(document, "material_id")
        formula_pretty_value = document_value(document, "formula_pretty")
        structure = document_value(document, "structure")
        missing_fields: list[str] = []
        if material_id_value is None:
            missing_fields.append("material_id")
        if not isinstance(formula_pretty_value, str) or not formula_pretty_value.strip():
            missing_fields.append("formula_pretty")
        if structure is None:
            missing_fields.append("structure")
        if missing_fields:
            raise MandatoryFieldError(
                f"Returned document {document_index} for {phase.phase_key} is "
                "missing mandatory fields: " + ", ".join(missing_fields)
            )

        material_id = require_safe_component(str(material_id_value), "material_id")
        if material_id in seen_material_ids:
            raise MandatoryFieldError(
                f"Materials Project returned duplicate material_id {material_id!r} "
                f"for {phase.phase_key}."
            )
        seen_material_ids.add(material_id)

        if not isinstance(structure, dependencies.structure_class):
            raise MandatoryFieldError(
                f"The structure for {material_id} is not a pymatgen Structure."
            )

        try:
            formula_composition = dependencies.composition_class(
                formula_pretty_value
            )
            formula_signature = composition_signature(formula_composition)
            structure_signature = composition_signature(structure.composition)
        except Exception as exc:
            raise MandatoryFieldError(
                f"Could not validate the reduced composition for {material_id}."
            ) from exc

        deprecated = optional_bool(document_value(document, "deprecated"))
        if deprecated is True:
            LOGGER.warning(
                "  Ignoring %s because it is marked deprecated despite the query filter.",
                material_id,
            )
            continue
        if (
            formula_signature != phase.composition_signature
            or structure_signature != phase.composition_signature
        ):
            LOGGER.warning(
                "  Ignoring %s because its reduced composition does not exactly "
                "match %s.",
                material_id,
                phase.query_formula,
            )
            continue

        symmetry = document_value(document, "symmetry")
        symmetry_symbol = optional_string(document_value(symmetry, "symbol"))
        symmetry_number = optional_integer(document_value(symmetry, "number"))
        crystal_system = optional_string(
            document_value(symmetry, "crystal_system")
        )

        volume = optional_finite_float(document_value(document, "volume"))
        if volume is None:
            volume = optional_finite_float(getattr(structure, "volume", None))
        density = optional_finite_float(document_value(document, "density"))
        if density is None:
            density = optional_finite_float(getattr(structure, "density", None))

        last_updated_value = document_value(document, "last_updated")
        last_updated_json = safely_convert_optional_provenance(
            last_updated_value, "last_updated"
        )
        last_updated = (
            last_updated_json if isinstance(last_updated_json, str) else None
        )

        candidates.append(
            Candidate(
                phase=phase,
                material_id=material_id,
                formula_pretty=formula_pretty_value.strip(),
                structure=structure,
                reduced_composition=phase.reduced_composition,
                number_of_sites=len(structure),
                volume_A3=volume,
                density_g_cm3=density,
                energy_above_hull_eV_per_atom=optional_finite_float(
                    document_value(document, "energy_above_hull")
                ),
                formation_energy_per_atom_eV=optional_finite_float(
                    document_value(document, "formation_energy_per_atom")
                ),
                is_stable=optional_bool(document_value(document, "is_stable")),
                theoretical=optional_bool(document_value(document, "theoretical")),
                deprecated=deprecated,
                symmetry_symbol=symmetry_symbol,
                symmetry_number=symmetry_number,
                crystal_system=crystal_system,
                database_ids=safely_convert_optional_provenance(
                    document_value(document, "database_IDs"), "database_IDs"
                ),
                origins=safely_convert_optional_provenance(
                    document_value(document, "origins"), "origins"
                ),
                materials_project_last_updated=last_updated,
                retrieval_time_utc=retrieval_time_utc,
            )
        )

    if not candidates:
        raise NoExactMatchesError(
            f"Materials Project returned {len(documents)} document(s) for "
            f"{phase.query_formula}, but none matched its exact reduced composition."
        )
    return candidates


def candidate_sort_key(candidate: Candidate) -> tuple[float, int, float, str]:
    """Apply the documented deterministic structure-selection ordering."""

    return (
        candidate.energy_above_hull_eV_per_atom
        if candidate.energy_above_hull_eV_per_atom is not None
        else math.inf,
        0 if candidate.is_stable is True else 1,
        candidate.formation_energy_per_atom_eV
        if candidate.formation_energy_per_atom_eV is not None
        else math.inf,
        candidate.material_id,
    )


def rank_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Rank all candidates and annotate the selected working structure."""

    ranked = sorted(candidates, key=candidate_sort_key)
    selected_id = ranked[0].material_id
    rule = (
        "lowest available energy above hull; stable status on a hull-energy "
        "tie; lowest available formation energy per atom on a further tie; "
        "lexicographical material_id as the final tie-breaker"
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate.selection_rank = rank
        candidate.selected_candidate = rank == 1
        if rank == 1:
            candidate.selection_reason = (
                f"Selected as the current working structure by deterministic "
                f"ranking ({rule}). It ranked 1 of {len(ranked)} candidates. "
                "This reproducible project choice is not a claim of absolute "
                "experimental ground truth."
            )
        else:
            candidate.selection_reason = (
                f"Retained as alternative candidate at rank {rank} of "
                f"{len(ranked)} under the deterministic rule ({rule}). "
                f"{selected_id} is the current working structure; this "
                "alternative was not discarded."
            )
    return ranked


def raw_relative_paths(candidate: Candidate) -> tuple[Path, Path, Path]:
    """Return repository-relative CIF, EXTXYZ, and metadata paths."""

    candidate_directory = (
        RAW_ROOT_RELATIVE_PATH
        / candidate.phase.phase_key
        / candidate.material_id
    )
    return (
        candidate_directory / "structure.cif",
        candidate_directory / "structure.extxyz",
        candidate_directory / "metadata.json",
    )


def selected_relative_paths(phase: PhaseDefinition) -> tuple[Path, Path, Path]:
    """Return repository-relative selected CIF, EXTXYZ, and metadata paths."""

    return (
        SELECTED_ROOT_RELATIVE_PATH / f"{phase.phase_key}.cif",
        SELECTED_ROOT_RELATIVE_PATH / f"{phase.phase_key}.extxyz",
        SELECTED_ROOT_RELATIVE_PATH / f"{phase.phase_key}.metadata.json",
    )


def relative_path_text(path: Path) -> str:
    """Use forward-slash repository-relative paths in portable provenance."""

    if path.is_absolute() or path.drive or ".." in path.parts:
        raise SerializationError(f"Unsafe repository-relative path: {path}")
    return path.as_posix()


def candidate_metadata(candidate: Candidate) -> dict[str, JSONValue]:
    """Create the complete JSON metadata record for one candidate."""

    raw_cif_path, raw_extxyz_path, _ = raw_relative_paths(candidate)
    selected_cif_path, selected_extxyz_path, _ = selected_relative_paths(
        candidate.phase
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "retrieval_time_utc": candidate.retrieval_time_utc,
        "phase_key": candidate.phase.phase_key,
        "query_formula": candidate.phase.query_formula,
        "aliases": list(candidate.phase.aliases),
        "scientific_description": candidate.phase.description,
        "is_compound": candidate.phase.is_compound,
        "is_elemental_reference": candidate.phase.is_elemental_reference,
        "material_id": candidate.material_id,
        "formula_pretty": candidate.formula_pretty,
        "reduced_composition": candidate.reduced_composition,
        "number_of_sites": candidate.number_of_sites,
        "volume_A3": candidate.volume_A3,
        "density_g_cm3": candidate.density_g_cm3,
        "energy_above_hull_eV_per_atom": (
            candidate.energy_above_hull_eV_per_atom
        ),
        "formation_energy_per_atom_eV": (
            candidate.formation_energy_per_atom_eV
        ),
        "is_stable": candidate.is_stable,
        "theoretical": candidate.theoretical,
        "deprecated": candidate.deprecated,
        "symmetry_symbol": candidate.symmetry_symbol,
        "symmetry_number": candidate.symmetry_number,
        "crystal_system": candidate.crystal_system,
        "database_IDs": candidate.database_ids,
        "origins": candidate.origins,
        "materials_project_last_updated": (
            candidate.materials_project_last_updated
        ),
        "selected_candidate": candidate.selected_candidate,
        "selection_rank": candidate.selection_rank,
        "selection_reason": candidate.selection_reason,
        "raw_cif_path": relative_path_text(raw_cif_path),
        "raw_extxyz_path": relative_path_text(raw_extxyz_path),
        "selected_cif_path": (
            relative_path_text(selected_cif_path)
            if candidate.selected_candidate
            else None
        ),
        "selected_extxyz_path": (
            relative_path_text(selected_extxyz_path)
            if candidate.selected_candidate
            else None
        ),
    }


def candidate_manifest_record(candidate: Candidate) -> dict[str, JSONValue]:
    """Create one normalized row shared by the JSON and CSV manifests."""

    raw_cif_path, raw_extxyz_path, _ = raw_relative_paths(candidate)
    selected_cif_path, selected_extxyz_path, _ = selected_relative_paths(
        candidate.phase
    )
    return {
        "phase_key": candidate.phase.phase_key,
        "query_formula": candidate.phase.query_formula,
        "aliases": list(candidate.phase.aliases),
        "material_id": candidate.material_id,
        "formula_pretty": candidate.formula_pretty,
        "reduced_composition": candidate.reduced_composition,
        "number_of_sites": candidate.number_of_sites,
        "energy_above_hull_eV_per_atom": (
            candidate.energy_above_hull_eV_per_atom
        ),
        "formation_energy_per_atom_eV": (
            candidate.formation_energy_per_atom_eV
        ),
        "is_stable": candidate.is_stable,
        "theoretical": candidate.theoretical,
        "symmetry_symbol": candidate.symmetry_symbol,
        "symmetry_number": candidate.symmetry_number,
        "crystal_system": candidate.crystal_system,
        "volume_A3": candidate.volume_A3,
        "density_g_cm3": candidate.density_g_cm3,
        "is_selected": candidate.selected_candidate,
        "selection_rank": candidate.selection_rank,
        "selection_reason": candidate.selection_reason,
        "raw_cif_path": relative_path_text(raw_cif_path),
        "raw_extxyz_path": relative_path_text(raw_extxyz_path),
        "selected_cif_path": (
            relative_path_text(selected_cif_path)
            if candidate.selected_candidate
            else None
        ),
        "selected_extxyz_path": (
            relative_path_text(selected_extxyz_path)
            if candidate.selected_candidate
            else None
        ),
        "retrieval_time_utc": candidate.retrieval_time_utc,
    }


def write_json(path: Path, value: JSONValue) -> None:
    """Write strict, readable UTF-8 JSON within a staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")


def write_candidate_cif(
    candidate: Candidate,
    path: Path,
    dependencies: ScientificDependencies,
) -> None:
    """Write the unmodified pymatgen Structure to CIF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dependencies.cif_writer_class(candidate.structure).write_file(str(path))
    except Exception as exc:
        raise StructureConversionError(
            f"CIF conversion failed for {candidate.material_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def write_candidate_extxyz(
    candidate: Candidate,
    path: Path,
    dependencies: ScientificDependencies,
) -> None:
    """Convert the unchanged periodic Structure and write scalar provenance."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atoms = dependencies.ase_adaptor_class.get_atoms(candidate.structure)
        if not bool(atoms.pbc.all()) or abs(float(atoms.cell.volume)) == 0.0:
            raise ValueError("the converted structure lost its periodic unit cell")

        # The adaptor may copy auxiliary info from pymatgen. Clearing only this
        # metadata mapping lets us guarantee EXTXYZ contains scalar provenance
        # while leaving atomic coordinates, cell vectors, and PBC untouched.
        atoms.info.clear()
        atoms.info.update(
            {
                "source": SOURCE_NAME,
                "material_id": candidate.material_id,
                "phase_key": candidate.phase.phase_key,
                "query_formula": candidate.phase.query_formula,
                "formula_pretty": candidate.formula_pretty,
                "selected_candidate": candidate.selected_candidate,
                "selection_rank": candidate.selection_rank,
                "retrieval_time_utc": candidate.retrieval_time_utc,
            }
        )
        if candidate.energy_above_hull_eV_per_atom is not None:
            atoms.info["energy_above_hull_eV_per_atom"] = (
                candidate.energy_above_hull_eV_per_atom
            )
        if candidate.formation_energy_per_atom_eV is not None:
            atoms.info["formation_energy_per_atom_eV"] = (
                candidate.formation_energy_per_atom_eV
            )
        if candidate.is_stable is not None:
            atoms.info["is_stable"] = candidate.is_stable

        dependencies.ase_write(
            str(path),
            atoms,
            format="extxyz",
        )
    except Exception as exc:
        raise StructureConversionError(
            f"EXTXYZ conversion failed for {candidate.material_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def inspect_target_bundle(
    project_root: Path,
    relative_paths: Sequence[Path],
    overwrite: bool,
    label: str,
) -> None:
    """Detect existing complete and partial bundles before staging a phase."""

    targets = [project_root / relative_path for relative_path in relative_paths]
    for target in targets:
        if target.exists() and not target.is_file():
            raise PartialOutputError(
                f"Expected a file for {label}, but found another object: {target}"
            )
    present = [target.exists() for target in targets]
    if any(present) and not all(present):
        missing = [str(path) for path, exists in zip(targets, present) if not exists]
        message = (
            f"Partially written output detected for {label}; missing: "
            + ", ".join(missing)
        )
        raise PartialOutputError(
            message
            + ". Inspect the incomplete provenance before another acquisition."
        )
    if any(present) and not overwrite:
        existing = [str(path) for path, exists in zip(targets, present) if exists]
        raise OutputCollisionError(
            f"Output already exists for {label}: {', '.join(existing)}. "
            "Use --overwrite to replace existing downloaded files."
        )


def validate_phase_output_targets(
    project_root: Path,
    phase: PhaseDefinition,
    candidates: Sequence[Candidate],
    overwrite: bool,
) -> None:
    """Preflight every candidate and selected bundle before writing a phase."""

    for candidate in candidates:
        inspect_target_bundle(
            project_root,
            raw_relative_paths(candidate),
            overwrite,
            f"{phase.phase_key}/{candidate.material_id}",
        )
    inspect_target_bundle(
        project_root,
        selected_relative_paths(phase),
        overwrite,
        f"selected {phase.phase_key}",
    )


def stage_phase_bundle(
    staging_root: Path,
    ranked_candidates: Sequence[Candidate],
    dependencies: ScientificDependencies,
) -> tuple[list[Path], list[dict[str, JSONValue]]]:
    """Stage all raw candidates and the selected working copies as one bundle."""

    staged_relative_paths: list[Path] = []
    manifest_records: list[dict[str, JSONValue]] = []

    # Every exact candidate is written before publication. A conversion error
    # therefore fails the phase without silently dropping a polymorph.
    for candidate in ranked_candidates:
        raw_cif_path, raw_extxyz_path, raw_metadata_path = raw_relative_paths(
            candidate
        )
        write_candidate_cif(
            candidate,
            staging_root / raw_cif_path,
            dependencies,
        )
        write_candidate_extxyz(
            candidate,
            staging_root / raw_extxyz_path,
            dependencies,
        )
        write_json(
            staging_root / raw_metadata_path,
            candidate_metadata(candidate),
        )
        staged_relative_paths.extend(
            (raw_cif_path, raw_extxyz_path, raw_metadata_path)
        )
        manifest_records.append(candidate_manifest_record(candidate))

    selected_candidate = ranked_candidates[0]
    raw_cif_path, raw_extxyz_path, _ = raw_relative_paths(selected_candidate)
    selected_cif_path, selected_extxyz_path, selected_metadata_path = (
        selected_relative_paths(selected_candidate.phase)
    )
    (staging_root / selected_cif_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staging_root / raw_cif_path, staging_root / selected_cif_path)
    shutil.copyfile(
        staging_root / raw_extxyz_path,
        staging_root / selected_extxyz_path,
    )
    write_json(
        staging_root / selected_metadata_path,
        candidate_metadata(selected_candidate),
    )
    staged_relative_paths.extend(
        (selected_cif_path, selected_extxyz_path, selected_metadata_path)
    )

    for relative_path in staged_relative_paths:
        staged_path = staging_root / relative_path
        if not staged_path.is_file() or staged_path.stat().st_size == 0:
            raise PartialOutputError(
                f"Staging did not produce a complete file: {relative_path_text(relative_path)}"
            )
    return staged_relative_paths, manifest_records


def remove_staged_phase(staging_root: Path, phase: PhaseDefinition) -> None:
    """Remove only temporary files for a failed phase."""

    raw_phase_path = staging_root / RAW_ROOT_RELATIVE_PATH / phase.phase_key
    if raw_phase_path.is_dir():
        shutil.rmtree(raw_phase_path)
    for relative_path in selected_relative_paths(phase):
        staged_path = staging_root / relative_path
        if staged_path.is_file():
            staged_path.unlink()


def manifest_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return the unique phase/material key for manifest consistency checks."""

    phase_key = record.get("phase_key")
    material_id = record.get("material_id")
    if not isinstance(phase_key, str) or not isinstance(material_id, str):
        raise PartialOutputError(
            "Manifest records must contain string phase_key and material_id fields."
        )
    return phase_key, material_id


def safe_manifest_relative_path(value: Any, field_name: str) -> Path:
    """Validate a repository-relative path read from an existing manifest."""

    if not isinstance(value, str) or not value:
        raise PartialOutputError(
            f"Existing manifest field {field_name!r} must contain a relative path."
        )
    path = Path(value.replace("/", os.sep))
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise PartialOutputError(
            f"Existing manifest contains unsafe path in {field_name!r}: {value!r}"
        )
    return path


def validate_existing_manifest_record(
    record: Mapping[str, JSONValue],
    phase_by_key: Mapping[str, PhaseDefinition],
) -> None:
    """Validate one retained manifest record before it can be merged."""

    missing_fields = set(MANIFEST_FIELDS).difference(record)
    if missing_fields:
        raise PartialOutputError(
            "Existing JSON manifest record is missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    phase_key, material_id = manifest_identity(record)
    phase = phase_by_key.get(phase_key)
    if phase is None:
        raise PartialOutputError(
            f"Existing manifest contains unknown phase key {phase_key!r}."
        )
    if not SAFE_PATH_COMPONENT.fullmatch(material_id) or material_id in {".", ".."}:
        raise PartialOutputError(
            f"Existing manifest contains unsafe material_id {material_id!r}."
        )

    expected_values: dict[str, JSONValue] = {
        "query_formula": phase.query_formula,
        "aliases": list(phase.aliases),
        "reduced_composition": phase.reduced_composition,
    }
    for field_name, expected_value in expected_values.items():
        if record.get(field_name) != expected_value:
            raise PartialOutputError(
                f"Existing manifest has a noncanonical {field_name!r} value for "
                f"{phase_key}/{material_id}."
            )

    formula_pretty = record.get("formula_pretty")
    selection_reason = record.get("selection_reason")
    retrieval_time = record.get("retrieval_time_utc")
    if not isinstance(formula_pretty, str) or not formula_pretty.strip():
        raise PartialOutputError(
            f"Existing manifest has no formula_pretty for {phase_key}/{material_id}."
        )
    if not isinstance(selection_reason, str) or not selection_reason.strip():
        raise PartialOutputError(
            f"Existing manifest has no selection reason for {phase_key}/{material_id}."
        )
    if not isinstance(retrieval_time, str) or not retrieval_time.endswith("Z"):
        raise PartialOutputError(
            f"Existing manifest has an invalid UTC retrieval time for "
            f"{phase_key}/{material_id}."
        )
    try:
        datetime.fromisoformat(retrieval_time.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise PartialOutputError(
            f"Existing manifest has an invalid UTC retrieval time for "
            f"{phase_key}/{material_id}."
        ) from exc

    number_of_sites = record.get("number_of_sites")
    selection_rank = record.get("selection_rank")
    is_selected = record.get("is_selected")
    if (
        not isinstance(number_of_sites, int)
        or isinstance(number_of_sites, bool)
        or number_of_sites <= 0
    ):
        raise PartialOutputError(
            f"Existing manifest has an invalid site count for {phase_key}/{material_id}."
        )
    if (
        not isinstance(selection_rank, int)
        or isinstance(selection_rank, bool)
        or selection_rank <= 0
    ):
        raise PartialOutputError(
            f"Existing manifest has an invalid selection rank for "
            f"{phase_key}/{material_id}."
        )
    if not isinstance(is_selected, bool):
        raise PartialOutputError(
            f"Existing manifest has a non-boolean selection flag for "
            f"{phase_key}/{material_id}."
        )
    if is_selected and selection_rank != 1:
        raise PartialOutputError(
            f"The selected candidate for {phase_key} must have selection rank 1."
        )

    for field_name in (
        "energy_above_hull_eV_per_atom",
        "formation_energy_per_atom_eV",
        "volume_A3",
        "density_g_cm3",
    ):
        value = record.get(field_name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise PartialOutputError(
                f"Existing manifest has an invalid {field_name!r} value for "
                f"{phase_key}/{material_id}."
            )
    for field_name in ("is_stable", "theoretical"):
        if record.get(field_name) is not None and not isinstance(
            record.get(field_name), bool
        ):
            raise PartialOutputError(
                f"Existing manifest has an invalid {field_name!r} value for "
                f"{phase_key}/{material_id}."
            )

    raw_directory = RAW_ROOT_RELATIVE_PATH / phase_key / material_id
    expected_raw_cif = raw_directory / "structure.cif"
    expected_raw_extxyz = raw_directory / "structure.extxyz"
    if safe_manifest_relative_path(
        record.get("raw_cif_path"), "raw_cif_path"
    ) != expected_raw_cif or safe_manifest_relative_path(
        record.get("raw_extxyz_path"), "raw_extxyz_path"
    ) != expected_raw_extxyz:
        raise PartialOutputError(
            f"Existing manifest has noncanonical raw paths for "
            f"{phase_key}/{material_id}."
        )

    expected_selected_cif, expected_selected_extxyz, _ = selected_relative_paths(
        phase
    )
    if is_selected:
        if safe_manifest_relative_path(
            record.get("selected_cif_path"), "selected_cif_path"
        ) != expected_selected_cif or safe_manifest_relative_path(
            record.get("selected_extxyz_path"), "selected_extxyz_path"
        ) != expected_selected_extxyz:
            raise PartialOutputError(
                f"Existing manifest has noncanonical selected paths for {phase_key}."
            )
    elif record.get("selected_cif_path") is not None or record.get(
        "selected_extxyz_path"
    ) is not None:
        raise PartialOutputError(
            f"Unselected manifest candidate {phase_key}/{material_id} contains "
            "selected working paths."
        )


def validate_metadata_against_manifest(
    metadata_path: Path,
    record: Mapping[str, JSONValue],
) -> None:
    """Confirm an existing metadata file agrees with its manifest record."""

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartialOutputError(
            f"Could not read candidate metadata {metadata_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise PartialOutputError(f"Candidate metadata is not an object: {metadata_path}")

    direct_fields = tuple(
        field_name
        for field_name in MANIFEST_FIELDS
        if field_name != "is_selected"
    )
    for field_name in direct_fields:
        if metadata.get(field_name) != record.get(field_name):
            raise PartialOutputError(
                f"Metadata and manifest disagree on {field_name!r}: {metadata_path}"
            )
    if metadata.get("selected_candidate") != record.get("is_selected"):
        raise PartialOutputError(
            f"Metadata and manifest disagree on selection state: {metadata_path}"
        )
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get(
        "source"
    ) != SOURCE_NAME:
        raise PartialOutputError(
            f"Metadata has an unsupported schema or source: {metadata_path}"
        )


def load_existing_manifests(project_root: Path) -> list[dict[str, JSONValue]]:
    """Load consistent existing manifests so single-phase runs preserve others."""

    json_path = project_root / JSON_MANIFEST_RELATIVE_PATH
    csv_path = project_root / CSV_MANIFEST_RELATIVE_PATH
    if json_path.exists() != csv_path.exists():
        raise PartialOutputError(
            "Partially written manifest output detected: the JSON and CSV "
            "manifests must either both exist or both be absent."
        )
    if not json_path.exists():
        return []
    if not json_path.is_file() or not csv_path.is_file():
        raise PartialOutputError("Manifest paths must be regular files.")

    try:
        json_value = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartialOutputError(
            f"Could not read existing JSON manifest: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(json_value, list) or not all(
        isinstance(record, dict) for record in json_value
    ):
        raise PartialOutputError(
            "The existing JSON manifest must be an array of candidate records."
        )

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(MANIFEST_FIELDS):
                raise PartialOutputError(
                    "The existing CSV manifest has missing, extra, or reordered "
                    "columns."
                )
            csv_records = list(reader)
    except (OSError, csv.Error) as exc:
        raise PartialOutputError(
            f"Could not read existing CSV manifest: {type(exc).__name__}: {exc}"
        ) from exc

    json_records: list[dict[str, JSONValue]] = [
        {str(key): to_json_compatible(value, str(key)) for key, value in record.items()}
        for record in json_value
    ]
    json_keys = [manifest_identity(record) for record in json_records]
    csv_keys = [manifest_identity(record) for record in csv_records]
    if len(set(json_keys)) != len(json_keys):
        raise PartialOutputError("Duplicate candidate records exist in the JSON manifest.")
    if len(set(csv_keys)) != len(csv_keys):
        raise PartialOutputError("Duplicate candidate records exist in the CSV manifest.")
    if set(json_keys) != set(csv_keys):
        raise PartialOutputError(
            "The existing JSON and CSV manifests describe different candidates."
        )

    csv_by_key = {manifest_identity(record): record for record in csv_records}
    for record in json_records:
        key = manifest_identity(record)
        csv_record = csv_by_key[key]
        for field_name in MANIFEST_FIELDS:
            expected_csv_value = str(csv_cell(record.get(field_name)))
            actual_csv_value = csv_record.get(field_name)
            if actual_csv_value != expected_csv_value:
                raise PartialOutputError(
                    "The existing JSON and CSV manifests disagree on "
                    f"{field_name!r} for {key[0]}/{key[1]}."
                )

    return json_records


def validate_existing_output_integrity(
    project_root: Path,
    phases: Sequence[PhaseDefinition],
    existing_records: Sequence[Mapping[str, JSONValue]],
) -> None:
    """Detect incomplete bundles and broken provenance before any API access."""

    raw_root = project_root / RAW_ROOT_RELATIVE_PATH
    selected_root = project_root / SELECTED_ROOT_RELATIVE_PATH
    phase_by_key = {phase.phase_key: phase for phase in phases}
    selected_counts: dict[str, int] = {}
    ranks_by_phase: dict[str, list[int]] = {}
    manifest_candidate_directories: set[Path] = set()
    manifest_raw_files: set[Path] = set()
    manifest_selected_files: set[Path] = set()

    for record in existing_records:
        phase_key, material_id = manifest_identity(record)
        validate_existing_manifest_record(record, phase_by_key)
        selection_rank = int(record["selection_rank"])
        ranks_by_phase.setdefault(phase_key, []).append(selection_rank)

        raw_cif = safe_manifest_relative_path(record.get("raw_cif_path"), "raw_cif_path")
        raw_extxyz = safe_manifest_relative_path(
            record.get("raw_extxyz_path"), "raw_extxyz_path"
        )
        raw_metadata = raw_cif.parent / "metadata.json"
        manifest_candidate_directories.add(raw_cif.parent)
        manifest_raw_files.update((raw_cif, raw_extxyz, raw_metadata))
        required_raw = (
            project_root / raw_cif,
            project_root / raw_extxyz,
            project_root / raw_metadata,
        )
        if not all(path.is_file() for path in required_raw):
            raise PartialOutputError(
                f"Manifest references an incomplete raw bundle for "
                f"{phase_key}/{material_id}."
            )
        validate_metadata_against_manifest(project_root / raw_metadata, record)

        if record.get("is_selected") is True:
            selected_counts[phase_key] = selected_counts.get(phase_key, 0) + 1
            selected_cif = safe_manifest_relative_path(
                record.get("selected_cif_path"), "selected_cif_path"
            )
            selected_extxyz = safe_manifest_relative_path(
                record.get("selected_extxyz_path"), "selected_extxyz_path"
            )
            selected_metadata = selected_cif.with_suffix(".metadata.json")
            if not all(
                (project_root / path).is_file()
                for path in (selected_cif, selected_extxyz, selected_metadata)
            ):
                raise PartialOutputError(
                    f"Manifest references an incomplete selected bundle for {phase_key}."
                )
            manifest_selected_files.update(
                (selected_cif, selected_extxyz, selected_metadata)
            )
            validate_metadata_against_manifest(
                project_root / selected_metadata,
                record,
            )

    represented_phases = set(ranks_by_phase)
    for phase_key, ranks in ranks_by_phase.items():
        if selected_counts.get(phase_key, 0) != 1:
            raise PartialOutputError(
                f"Existing manifest must select exactly one candidate for {phase_key}."
            )
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise PartialOutputError(
                f"Existing manifest ranks for {phase_key} must be contiguous from 1."
            )

    # Scan in the opposite direction as well: every complete raw directory and
    # selected file must be represented by the manifests. This prevents stale
    # candidates from becoming silent, untracked provenance after a refresh.
    if raw_root.is_dir():
        for candidate_directory in raw_root.glob("*/*"):
            if not candidate_directory.is_dir():
                continue
            expected = (
                candidate_directory / "structure.cif",
                candidate_directory / "structure.extxyz",
                candidate_directory / "metadata.json",
            )
            present = [path.is_file() for path in expected]
            if any(present) and not all(present):
                raise PartialOutputError(
                    f"Partially written raw candidate bundle: {candidate_directory}"
                )
            if all(present):
                relative_directory = candidate_directory.relative_to(project_root)
                if relative_directory not in manifest_candidate_directories:
                    raise PartialOutputError(
                        "Complete raw candidate bundle is absent from the manifests: "
                        f"{candidate_directory}"
                    )
                actual_file_names = {
                    path.name for path in candidate_directory.iterdir() if path.is_file()
                }
                if actual_file_names != {
                    "structure.cif",
                    "structure.extxyz",
                    "metadata.json",
                }:
                    raise PartialOutputError(
                        f"Unexpected files exist in raw candidate bundle: "
                        f"{candidate_directory}"
                    )
        for raw_file in raw_root.rglob("*"):
            if raw_file.is_file():
                relative_file = raw_file.relative_to(project_root)
                if relative_file not in manifest_raw_files:
                    raise PartialOutputError(
                        "Raw output file is absent from the manifests: "
                        f"{raw_file}"
                    )

    for phase in phases:
        selected_relative = selected_relative_paths(phase)
        selected_paths = [project_root / path for path in selected_relative]
        present = [path.is_file() for path in selected_paths]
        if any(present) and not all(present):
            raise PartialOutputError(
                f"Partially written selected bundle for {phase.phase_key}."
            )
        if all(present) and phase.phase_key not in represented_phases:
            raise PartialOutputError(
                f"Selected bundle for {phase.phase_key} is absent from the manifests."
            )

    if selected_root.is_dir():
        for selected_file in selected_root.iterdir():
            if selected_file.is_file():
                relative_file = selected_file.relative_to(project_root)
                if relative_file not in manifest_selected_files:
                    raise PartialOutputError(
                        "Selected output file is absent from the manifests: "
                        f"{selected_file}"
                    )


def merge_manifest_records(
    existing_records: Sequence[dict[str, JSONValue]],
    replacement_records: Mapping[str, Sequence[dict[str, JSONValue]]],
    phases: Sequence[PhaseDefinition],
) -> list[dict[str, JSONValue]]:
    """Replace successful phases while preserving unrequested or failed phases."""

    replaced_phase_keys = set(replacement_records)
    merged = [
        dict(record)
        for record in existing_records
        if record.get("phase_key") not in replaced_phase_keys
    ]
    for records in replacement_records.values():
        merged.extend(dict(record) for record in records)

    phase_order = {phase.phase_key: phase.order for phase in phases}
    merged.sort(
        key=lambda record: (
            phase_order.get(str(record.get("phase_key")), len(phase_order)),
            optional_integer(record.get("selection_rank")) or sys.maxsize,
            str(record.get("material_id", "")),
        )
    )
    identities = [manifest_identity(record) for record in merged]
    if len(set(identities)) != len(identities):
        raise PartialOutputError("Merged manifests would contain duplicate candidates.")
    return merged


def obsolete_candidate_directories(
    existing_records: Sequence[Mapping[str, JSONValue]],
    phase: PhaseDefinition,
    current_candidates: Sequence[Candidate],
) -> list[Path]:
    """Find prior candidate directories superseded by a refreshed API result."""

    current_material_ids = {
        candidate.material_id for candidate in current_candidates
    }
    obsolete: set[Path] = set()
    for record in existing_records:
        if record.get("phase_key") != phase.phase_key:
            continue
        material_id = record.get("material_id")
        if isinstance(material_id, str) and material_id not in current_material_ids:
            obsolete.add(RAW_ROOT_RELATIVE_PATH / phase.phase_key / material_id)
    return sorted(obsolete, key=lambda path: path.as_posix())


def csv_cell(value: JSONValue) -> str | int | float:
    """Convert JSON values into stable, lossless CSV cell representations."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def stage_manifests(
    staging_root: Path,
    records: Sequence[dict[str, JSONValue]],
) -> list[Path]:
    """Create JSON and CSV manifests from the same sorted candidate records."""

    write_json(staging_root / JSON_MANIFEST_RELATIVE_PATH, list(records))
    csv_path = staging_root / CSV_MANIFEST_RELATIVE_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(MANIFEST_FIELDS),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: csv_cell(record.get(field)) for field in MANIFEST_FIELDS}
            )
    return [CSV_MANIFEST_RELATIVE_PATH, JSON_MANIFEST_RELATIVE_PATH]


def publication_priority(relative_path: Path) -> tuple[int, str]:
    """Publish metadata before structures and manifests after all phase files."""

    if relative_path in {
        CSV_MANIFEST_RELATIVE_PATH,
        JSON_MANIFEST_RELATIVE_PATH,
    }:
        priority = 3
    elif relative_path.name.endswith("metadata.json") or relative_path.name == "metadata.json":
        priority = 0
    elif relative_path.suffix.lower() == ".cif":
        priority = 1
    else:
        priority = 2
    return priority, relative_path.as_posix()


def publish_staged_files(
    project_root: Path,
    staging_root: Path,
    relative_paths: Sequence[Path],
    replaceable_paths: set[Path],
    obsolete_relative_paths: Sequence[Path] = (),
) -> None:
    """Publish staged files atomically and restore prior files on any failure."""

    unique_paths = sorted(set(relative_paths), key=publication_priority)
    rollback_root = staging_root / "__rollback__"
    committed: list[tuple[Path, Path | None, bool]] = []
    removed_directories: list[tuple[Path, Path]] = []

    for relative_path in unique_paths:
        staged_path = staging_root / relative_path
        target_path = project_root / relative_path
        if not staged_path.is_file():
            raise PartialOutputError(
                f"Missing staged file before publication: {relative_path_text(relative_path)}"
            )
        if target_path.exists() and relative_path not in replaceable_paths:
            raise OutputCollisionError(
                f"File already exists without --overwrite: {target_path}"
            )
        if target_path.exists() and not target_path.is_file():
            raise PartialOutputError(
                f"Expected a replaceable file but found another object: {target_path}"
            )

    for obsolete_relative_path in obsolete_relative_paths:
        relative_path_text(obsolete_relative_path)
        obsolete_path = project_root / obsolete_relative_path
        if obsolete_path.exists() and not obsolete_path.is_dir():
            raise PartialOutputError(
                f"Expected an obsolete candidate directory: {obsolete_path}"
            )

    try:
        # A refresh may return a different current candidate set. Prior
        # candidates no longer returned are moved into the rollback area first,
        # then removed only when the full publication transaction succeeds.
        for obsolete_relative_path in sorted(
            set(obsolete_relative_paths), key=lambda path: path.as_posix()
        ):
            obsolete_path = project_root / obsolete_relative_path
            if not obsolete_path.exists():
                continue
            backup_path = rollback_root / "obsolete" / obsolete_relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(obsolete_path, backup_path)
            removed_directories.append((obsolete_path, backup_path))

        for relative_path in unique_paths:
            staged_path = staging_root / relative_path
            target_path = project_root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path: Path | None = None
            if target_path.exists():
                backup_path = rollback_root / "files" / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target_path, backup_path)
            # Register the backup before publishing. If the next replace fails,
            # the outer rollback still knows where the only prior copy lives.
            committed.append((target_path, backup_path, False))
            os.replace(staged_path, target_path)
            committed[-1] = (target_path, backup_path, True)
    except Exception as exc:
        rollback_errors: list[str] = []
        for target_path, backup_path, published in reversed(committed):
            try:
                if published and target_path.is_file():
                    target_path.unlink()
                if backup_path is not None and backup_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup_path, target_path)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{target_path}: {type(rollback_exc).__name__}"
                )
        for obsolete_path, backup_path in reversed(removed_directories):
            try:
                if backup_path.exists():
                    obsolete_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup_path, obsolete_path)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{obsolete_path}: {type(rollback_exc).__name__}"
                )

        recovery_detail = ""
        if rollback_errors and rollback_root.exists():
            recovery_path = project_root / (
                ".ni_al_step4_recovery_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                + str(os.getpid())
            )
            try:
                os.replace(rollback_root, recovery_path)
                recovery_detail = f"; preserved rollback data at {recovery_path}"
            except OSError as recovery_exc:
                rollback_errors.append(
                    "rollback preservation: " + type(recovery_exc).__name__
                )
        detail = (
            "; rollback also failed for " + ", ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        raise PartialOutputError(
            f"Atomic publication failed ({type(exc).__name__}: {exc})"
            f"{detail}{recovery_detail}"
        ) from exc


def log_phase_failure(
    outcome: PhaseOutcome,
    exc: BaseException,
    api_key: str,
    verbose: bool,
) -> None:
    """Record and report one phase failure without exposing the API key."""

    outcome.error_type = type(exc).__name__
    outcome.error_message = safe_error_text(exc, api_key)
    LOGGER.error(
        "Phase %s failed (%s): %s",
        outcome.phase.phase_key,
        outcome.error_type,
        outcome.error_message,
    )
    if verbose:
        trace = traceback.format_exc().replace(api_key, "<redacted>")
        LOGGER.debug(trace.rstrip())


def acquire_structures(
    project_root: Path,
    all_phases: Sequence[PhaseDefinition],
    requested_phases: Sequence[PhaseDefinition],
    existing_records: Sequence[dict[str, JSONValue]],
    dependencies: ScientificDependencies,
    api_key: str,
    overwrite: bool,
    verbose: bool,
) -> tuple[list[PhaseOutcome], bool]:
    """Connect, query phases, stage outputs, and publish successful work."""

    outcomes = [PhaseOutcome(phase=phase) for phase in requested_phases]
    retrieval_time_utc = utc_timestamp()
    replacement_records: dict[str, list[dict[str, JSONValue]]] = {}
    staged_phase_paths: dict[str, list[Path]] = {}
    obsolete_phase_paths: dict[str, list[Path]] = {}

    # A shared temporary directory keeps every successful phase and both
    # manifests off the final paths until the complete publish transaction.
    with tempfile.TemporaryDirectory(
        prefix=".ni_al_step4_",
        dir=project_root,
    ) as temporary_directory:
        staging_root = Path(temporary_directory)

        # MPRester is the official public client; construction and endpoint
        # inspection occur only after a real key has been loaded safely.
        try:
            mpr_context = dependencies.mp_rester_class(
                api_key=api_key,
                mute_progress_bars=True,
            )
        except Exception as exc:
            raise ApiConnectionError(
                "Failed to initialize the Materials Project client: "
                f"{type(exc).__name__}: {safe_error_text(exc, api_key)}"
            ) from exc

        try:
            with mpr_context as mpr:
                summary_endpoint = mpr.materials.summary
                requested_fields = public_summary_fields(summary_endpoint)

                for outcome in outcomes:
                    phase = outcome.phase
                    LOGGER.info(
                        "Querying %s (formula %s)...",
                        phase.phase_key,
                        phase.query_formula,
                    )
                    try:
                        documents = query_summary_documents(
                            summary_endpoint,
                            phase,
                            requested_fields,
                        )
                        outcome.api_documents_returned = len(documents)
                        candidates = build_exact_candidates(
                            documents,
                            phase,
                            retrieval_time_utc,
                            dependencies,
                        )
                        outcome.exact_candidates_retained = len(candidates)

                        # Ranking happens before any file is written so raw and
                        # selected metadata agree on every rank and reason.
                        ranked_candidates = rank_candidates(candidates)
                        outcome.selected = ranked_candidates[0]
                        obsolete_paths = (
                            obsolete_candidate_directories(
                                existing_records,
                                phase,
                                ranked_candidates,
                            )
                            if overwrite
                            else []
                        )
                        validate_phase_output_targets(
                            project_root,
                            phase,
                            ranked_candidates,
                            overwrite,
                        )
                        phase_paths, records = stage_phase_bundle(
                            staging_root,
                            ranked_candidates,
                            dependencies,
                        )
                        staged_phase_paths[phase.phase_key] = phase_paths
                        obsolete_phase_paths[phase.phase_key] = obsolete_paths
                        replacement_records[phase.phase_key] = records
                        LOGGER.info(
                            "  Retained %d exact candidate(s); selected %s.",
                            len(ranked_candidates),
                            ranked_candidates[0].material_id,
                        )
                    except Exception as exc:
                        remove_staged_phase(staging_root, phase)
                        replacement_records.pop(phase.phase_key, None)
                        staged_phase_paths.pop(phase.phase_key, None)
                        obsolete_phase_paths.pop(phase.phase_key, None)
                        log_phase_failure(outcome, exc, api_key, verbose)
        except (ApiCompatibilityError, MandatoryFieldError):
            raise
        except Exception as exc:
            raise ApiConnectionError(
                "Failed while communicating with Materials Project: "
                f"{type(exc).__name__}: {safe_error_text(exc, api_key)}"
            ) from exc

        if not replacement_records:
            return outcomes, False

        merged_records = merge_manifest_records(
            existing_records,
            replacement_records,
            all_phases,
        )
        manifest_paths = stage_manifests(staging_root, merged_records)
        all_staged_paths = [
            relative_path
            for paths in staged_phase_paths.values()
            for relative_path in paths
        ] + manifest_paths
        all_obsolete_paths = [
            relative_path
            for paths in obsolete_phase_paths.values()
            for relative_path in paths
        ]
        if all_obsolete_paths:
            LOGGER.info(
                "Refreshing the current dataset will remove %d stale candidate "
                "bundle(s) superseded by this API result.",
                len(all_obsolete_paths),
            )

        # Manifests are expected to change when a new phase is added, so their
        # existing versions are transactionally replaceable. Downloaded phase
        # files remain protected unless --overwrite was explicitly supplied.
        replaceable_paths = set(manifest_paths)
        if overwrite:
            replaceable_paths.update(all_staged_paths)

        try:
            publish_staged_files(
                project_root,
                staging_root,
                all_staged_paths,
                replaceable_paths,
                all_obsolete_paths,
            )
        except Exception as exc:
            for outcome in outcomes:
                if outcome.phase.phase_key in replacement_records:
                    log_phase_failure(outcome, exc, api_key, verbose)
            return outcomes, False

        for outcome in outcomes:
            if outcome.phase.phase_key in replacement_records:
                outcome.completed = True
                outcome.candidates_saved = outcome.exact_candidates_retained
        return outcomes, True


def format_optional_energy(value: float | None) -> str:
    """Format optional per-atom energies for the console report."""

    return "not available" if value is None else f"{value:.8f} eV/atom"


def print_console_report(
    project_root: Path,
    outcomes: Sequence[PhaseOutcome],
    manifests_written: bool,
) -> None:
    """Print per-phase provenance and the final completion summary."""

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("Ni-Al Materials Project structure-acquisition summary")
    LOGGER.info("=" * 72)
    for outcome in outcomes:
        phase = outcome.phase
        LOGGER.info("Phase: %s", phase.phase_key)
        LOGGER.info("  Query formula: %s", phase.query_formula)
        LOGGER.info("  API documents returned: %d", outcome.api_documents_returned)
        LOGGER.info(
            "  Exact-composition candidates retained: %d",
            outcome.exact_candidates_retained,
        )
        if outcome.completed and outcome.selected is not None:
            selected = outcome.selected
            selected_cif, selected_extxyz, _ = selected_relative_paths(phase)
            space_group = selected.symmetry_symbol or "not available"
            if selected.symmetry_number is not None:
                space_group += f" ({selected.symmetry_number})"
            LOGGER.info("  Selected Materials Project ID: %s", selected.material_id)
            LOGGER.info("  Selected formula: %s", selected.formula_pretty)
            LOGGER.info(
                "  Energy above hull: %s",
                format_optional_energy(selected.energy_above_hull_eV_per_atom),
            )
            LOGGER.info(
                "  Formation energy: %s",
                format_optional_energy(selected.formation_energy_per_atom_eV),
            )
            LOGGER.info("  Space group: %s", space_group)
            LOGGER.info("  Selected CIF: %s", relative_path_text(selected_cif))
            LOGGER.info(
                "  Selected EXTXYZ: %s", relative_path_text(selected_extxyz)
            )
        else:
            LOGGER.info(
                "  Status: FAILED (%s: %s)",
                outcome.error_type or "UnknownError",
                outcome.error_message or "no additional detail",
            )

    completed = sum(outcome.completed for outcome in outcomes)
    failed = len(outcomes) - completed
    total_candidates = sum(outcome.candidates_saved for outcome in outcomes)
    LOGGER.info("")
    LOGGER.info("Requested phases: %d", len(outcomes))
    LOGGER.info("Completed phases: %d", completed)
    LOGGER.info("Failed phases: %d", failed)
    LOGGER.info("Total candidates saved: %d", total_candidates)
    if manifests_written:
        LOGGER.info(
            "CSV manifest: %s",
            relative_path_text(CSV_MANIFEST_RELATIVE_PATH),
        )
        LOGGER.info(
            "JSON manifest: %s",
            relative_path_text(JSON_MANIFEST_RELATIVE_PATH),
        )
    else:
        LOGGER.info("Manifests: not written during this run")
    if failed:
        LOGGER.error("Step 4 finished with one or more failures.")
    else:
        LOGGER.info("Step 4 structure acquisition completed successfully.")


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate inputs and run the requested Step 4 workflow."""

    args = parse_arguments(arguments)
    configure_logging(args.verbose)
    project_root = locate_project_root()

    try:
        validate_runtime_and_folders(project_root)
        dependencies = import_scientific_dependencies()
        config_path = resolve_config_path(project_root, args.config)
        description, schema_version, phases = load_phase_configuration(
            config_path,
            dependencies,
        )
        requested_phases = choose_requested_phases(phases, args.phase)
        existing_records = load_existing_manifests(project_root)
        validate_existing_output_integrity(project_root, phases, existing_records)

        if args.validate_only:
            LOGGER.info("Step 4 validation succeeded.")
            LOGGER.info("Project root: %s", project_root)
            LOGGER.info("Configuration: %s", config_path)
            LOGGER.info("Schema version: %s", schema_version)
            LOGGER.info("Description: %s", description)
            LOGGER.info(
                "Validated phases: %s",
                ", ".join(phase.phase_key for phase in requested_phases),
            )
            LOGGER.info("No API key was required and no API connection was made.")
            return 0

        api_key = load_api_key(project_root, dependencies.load_dotenv)
        outcomes, manifests_written = acquire_structures(
            project_root=project_root,
            all_phases=phases,
            requested_phases=requested_phases,
            existing_records=existing_records,
            dependencies=dependencies,
            api_key=api_key,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        print_console_report(project_root, outcomes, manifests_written)
        return 0 if all(outcome.completed for outcome in outcomes) else 1
    except Step4Error as exc:
        LOGGER.error("Step 4 failed (%s): %s", type(exc).__name__, exc)
        if args.verbose:
            LOGGER.debug(traceback.format_exc().rstrip())
        return 1
    except Exception as exc:
        LOGGER.error(
            "Step 4 failed unexpectedly (%s): %s",
            type(exc).__name__,
            exc,
        )
        if args.verbose:
            LOGGER.debug(traceback.format_exc().rstrip())
        return 1


if __name__ == "__main__":
    # Report the main workflow result to CMD, PowerShell, and automation tools.
    raise SystemExit(main())

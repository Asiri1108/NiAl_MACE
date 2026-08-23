"""Evaluate selected Ni-Al structures with pretrained MACE-MP-0.

This Step 5 workflow performs zero-shot, single-point energy, force, and
stress calculations. It deliberately does not train or fine-tune a model,
relax a structure, run molecular dynamics, or invoke LAMMPS.

The workflow is organized so that scientific validation is independent from
model execution:

1. Locate the repository from this file, never from the shell directory.
2. Load and strictly validate the JSON configuration.
3. Discover the configured selected EXTXYZ and metadata pairs.
4. Validate periodicity, finite geometry, provenance, and reduced composition.
5. Load one shared MACE-MP-0 calculator only for a real evaluation run.
6. Attach that calculator to an isolated copy of each input structure.
7. Request single-point energy, forces, and ASE-order stress.
8. Calculate force magnitudes and total-force summary statistics with NumPy.
9. Confirm that calculator use did not change the input geometry.
10. Create a separate calculator-free, annotated output copy.
11. Stage EXTXYZ, CSV, JSON, and text outputs before atomic publication.
12. Report every phase success or failure and return a meaningful exit code.
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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias
from urllib.error import HTTPError, URLError


LOGGER = logging.getLogger("evaluate_ni_al_mace_zero_shot")

SCHEMA_VERSION = "1.0"
EVALUATION_TYPE = "zero-shot single-point"
DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/mace_zero_shot.json")
EXPECTED_PHASE_ORDER = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
ALLOWED_ELEMENTS = frozenset({"Al", "Ni"})
SAFE_PHASE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STRESS_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")

STRUCTURES_SUBDIRECTORY = Path("structures")
TABLES_SUBDIRECTORY = Path("tables")
REPORTS_SUBDIRECTORY = Path("reports")
CSV_FILENAME = "ni_al_mace_zero_shot.csv"
JSON_FILENAME = "ni_al_mace_zero_shot.json"
TEXT_REPORT_FILENAME = "ni_al_mace_zero_shot.txt"
RUN_LOCK_FILENAME = ".ni_al_step5.lock"

SUMMARY_FIELDS = (
    "phase_key",
    "formula",
    "material_id",
    "number_of_atoms",
    "space_group_symbol",
    "space_group_number",
    "mace_model_name",
    "model_size",
    "device",
    "dtype",
    "dispersion_enabled",
    "total_energy_eV",
    "energy_per_atom_eV",
    "volume_A3",
    "volume_per_atom_A3",
    "maximum_force_eV_per_A",
    "mean_force_eV_per_A",
    "rms_force_eV_per_A",
    "minimum_force_eV_per_A",
    "total_force_x_eV_per_A",
    "total_force_y_eV_per_A",
    "total_force_z_eV_per_A",
    "total_force_norm_eV_per_A",
    "stress_xx_eV_per_A3",
    "stress_yy_eV_per_A3",
    "stress_zz_eV_per_A3",
    "stress_yz_eV_per_A3",
    "stress_xz_eV_per_A3",
    "stress_xy_eV_per_A3",
    "input_structure_path",
    "output_structure_path",
    "evaluation_time_utc",
    "evaluation_status",
    "error_type",
    "error_message",
)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class Step5Error(RuntimeError):
    """Base class for anticipated Step 5 workflow errors."""


class DependencyError(Step5Error):
    """Raised when the required project environment is unavailable."""


class ConfigurationError(Step5Error):
    """Raised when the Step 5 configuration is missing or invalid."""


class InputValidationError(Step5Error):
    """Raised when a selected structure or its metadata is invalid."""


class OutputCollisionError(Step5Error):
    """Raised when generated output exists without explicit replacement."""


class PartialOutputError(Step5Error):
    """Raised when an existing or staged output bundle is incomplete."""


class ModelLoadingError(Step5Error):
    """Raised when the pretrained calculator cannot be constructed."""


class ModelDownloadError(ModelLoadingError):
    """Raised when obtaining the requested pretrained checkpoint fails."""


class EnergyCalculationError(Step5Error):
    """Raised when MACE cannot return a total energy."""


class ForceCalculationError(Step5Error):
    """Raised when MACE cannot return atomic forces."""


class StressCalculationError(Step5Error):
    """Raised when MACE cannot return periodic-cell stress."""


class NonFiniteResultError(Step5Error):
    """Raised when a calculated or derived value is NaN or infinite."""


class StructureWriteError(Step5Error):
    """Raised when an annotated EXTXYZ cannot be written and verified."""


class PublicationError(Step5Error):
    """Raised when staged outputs cannot be published or rolled back."""


class DuplicateJsonKeyError(ValueError):
    """Internal signal used to reject ambiguous duplicate JSON keys."""


@dataclass(frozen=True)
class ScientificDependencies:
    """Third-party objects used after dependency validation."""

    numpy: Any
    ase_read: Callable[..., Any]
    ase_write: Callable[..., None]
    composition_class: type[Any]
    mace_factory: Callable[..., Any]


@dataclass(frozen=True)
class PhaseDefinition:
    """One configured phase and its order-independent target composition."""

    order: int
    phase_key: str
    reduced_formula: str
    composition_signature: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated Step 5 configuration resolved against the repository."""

    schema_version: str
    description: str
    model_family: str
    model_name: str
    model_size: str
    device: str
    default_dtype: str
    dispersion: bool
    input_relative_path: Path
    output_relative_path: Path
    input_directory: Path
    output_directory: Path
    phases: tuple[PhaseDefinition, ...]


@dataclass(frozen=True)
class RuntimeOptions:
    """Effective calculator settings after applying command-line overrides."""

    model: str
    device: str
    dtype: str


@dataclass(frozen=True)
class PhaseInput:
    """A completely validated source structure and its provenance."""

    phase: PhaseDefinition
    atoms: Any
    metadata: Mapping[str, JSONValue]
    input_structure_relative_path: Path
    metadata_relative_path: Path
    output_structure_relative_path: Path


@dataclass(frozen=True)
class PhaseResult:
    """All calculated values for one successful single-point evaluation."""

    total_energy_eV: float
    energy_per_atom_eV: float
    forces_eV_per_A: Any
    stress_eV_per_A3: Any
    number_of_atoms: int
    volume_A3: float
    volume_per_atom_A3: float
    maximum_force_eV_per_A: float
    mean_force_eV_per_A: float
    rms_force_eV_per_A: float
    minimum_force_eV_per_A: float
    total_force_eV_per_A: Any
    total_force_norm_eV_per_A: float
    evaluation_time_utc: str


@dataclass
class PhaseOutcome:
    """Mutable run state used to report one phase success or failure."""

    phase: PhaseDefinition
    phase_input: PhaseInput | None = None
    result: PhaseResult | None = None
    completed: bool = False
    error_type: str | None = None
    error_message: str | None = None
    failure_time_utc: str | None = None


def locate_project_root() -> Path:
    """Locate the repository from the stable location of this script."""

    # This file is stored in ``project/scripts``. Resolving its second parent
    # makes the workflow independent of the caller's current working directory.
    return Path(__file__).resolve().parents[1]


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the Step 5 command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or evaluate the five selected Ni-Al structures with "
            "pretrained MACE-MP-0 in zero-shot single-point mode."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON configuration path (default: configs/mace_zero_shot.json "
            "relative to the repository root)."
        ),
    )
    parser.add_argument(
        "--phase",
        help="Process one exact phase key; omit to process all five phases.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="MACE model alias or path (default from config: small).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device (default from config: cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default=None,
        help="Numerical precision (default from config: float64).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing generated result files to be replaced.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration, dependencies, inputs, formulas, and "
            "output paths without loading MACE or calculating properties."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed workflow logging.",
    )
    return parser.parse_args(arguments)


def configure_logging(verbose: bool) -> None:
    """Configure concise console logging without enabling third-party debug logs."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False
    logging.getLogger().setLevel(logging.WARNING)


def validate_runtime(project_root: Path) -> None:
    """Require Python 3.11 from this repository's virtual environment."""

    expected_prefix = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    if os.path.normcase(str(active_prefix)) != os.path.normcase(
        str(expected_prefix)
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


def import_scientific_dependencies() -> ScientificDependencies:
    """Import required packages without constructing or downloading a model."""

    try:
        import numpy as np
        from ase.io import read as ase_read
        from ase.io import write as ase_write
        from mace.calculators import mace_mp
        from pymatgen.core import Composition
    except ImportError as exc:
        missing_name = exc.name or "an unknown package"
        raise DependencyError(
            f"Missing required Python import '{missing_name}' in the project "
            "virtual environment. Install the Step 5 dependencies before running."
        ) from exc

    try:
        parameters = inspect.signature(mace_mp).parameters
    except (TypeError, ValueError) as exc:
        raise DependencyError("Could not inspect the installed mace_mp function.") from exc
    required_parameters = {"model", "device", "default_dtype", "dispersion"}
    missing_parameters = required_parameters.difference(parameters)
    if missing_parameters:
        raise DependencyError(
            "The installed MACE version does not support required mace_mp "
            "arguments: " + ", ".join(sorted(missing_parameters))
        )

    return ScientificDependencies(
        numpy=np,
        ase_read=ase_read,
        ase_write=ase_write,
        composition_class=Composition,
        mace_factory=mace_mp,
    )


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_nonstandard_json_constant(value: str) -> None:
    """Reject JSON NaN and Infinity spellings accepted by Python by default."""

    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read strict UTF-8 JSON and require an object at its root."""

    if not path.is_file():
        raise ConfigurationError(f"{label} not found: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonstandard_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"{label} is not valid UTF-8: {path}") from exc
    except (json.JSONDecodeError, DuplicateJsonKeyError, ValueError) as exc:
        detail = (
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        raise ConfigurationError(f"Invalid JSON in {path}: {detail}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not read {label.lower()} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} root must be a JSON object: {path}")
    return value


def require_nonempty_string(value: Any, field_name: str) -> str:
    """Return a trimmed string or raise a field-specific configuration error."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def repository_relative_path(
    project_root: Path,
    value: Any,
    field_name: str,
) -> tuple[Path, Path]:
    """Resolve a configured path while keeping it inside the repository."""

    text = require_nonempty_string(value, field_name)
    relative_path = Path(text.replace("/", os.sep))
    if (
        relative_path.is_absolute()
        or relative_path.drive
        or ".." in relative_path.parts
    ):
        raise ConfigurationError(
            f"'{field_name}' must be a safe repository-relative path: {text!r}"
        )
    resolved_path = (project_root / relative_path).resolve()
    try:
        resolved_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ConfigurationError(
            f"'{field_name}' resolves outside the repository: {text!r}"
        ) from exc
    return relative_path, resolved_path


def composition_signature(composition: Any) -> tuple[tuple[str, float], ...]:
    """Return an element-order-independent signature of a reduced composition."""

    reduced = composition.reduced_composition
    return tuple(
        sorted(
            (str(symbol), float(amount))
            for symbol, amount in reduced.get_el_amt_dict().items()
        )
    )


def resolve_config_path(project_root: Path, supplied_path: Path | None) -> Path:
    """Resolve the default config from the repository and custom paths from cwd."""

    if supplied_path is None:
        return project_root / DEFAULT_CONFIG_RELATIVE_PATH
    return supplied_path.expanduser().resolve()


def load_configuration(
    project_root: Path,
    config_path: Path,
    dependencies: ScientificDependencies,
) -> EvaluationConfig:
    """Load and validate the fixed five-phase zero-shot configuration."""

    # Keeping model and path choices in JSON makes the scientific run settings
    # reviewable, while strict validation prevents accidental changes of scope.
    raw = read_json_object(config_path, "Configuration file")
    required_fields = {
        "schema_version",
        "description",
        "model_family",
        "model_name",
        "model_size",
        "device",
        "default_dtype",
        "dispersion",
        "input_directory",
        "output_directory",
        "phase_order",
        "expected_phases",
    }
    missing_fields = required_fields.difference(raw)
    if missing_fields:
        raise ConfigurationError(
            "Missing configuration fields: " + ", ".join(sorted(missing_fields))
        )
    unexpected_fields = set(raw).difference(required_fields)
    if unexpected_fields:
        raise ConfigurationError(
            "Unexpected configuration fields: "
            + ", ".join(sorted(unexpected_fields))
        )

    schema_version = require_nonempty_string(
        raw.get("schema_version"), "schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}."
        )
    description = require_nonempty_string(raw.get("description"), "description")
    model_family = require_nonempty_string(raw.get("model_family"), "model_family")
    model_name = require_nonempty_string(raw.get("model_name"), "model_name")
    model_size = require_nonempty_string(raw.get("model_size"), "model_size")
    device = require_nonempty_string(raw.get("device"), "device")
    default_dtype = require_nonempty_string(
        raw.get("default_dtype"), "default_dtype"
    )
    if model_family != "MACE" or model_name != "MACE-MP-0":
        raise ConfigurationError(
            "Step 5 requires model_family 'MACE' and model_name 'MACE-MP-0'."
        )
    if default_dtype not in {"float32", "float64"}:
        raise ConfigurationError("'default_dtype' must be 'float32' or 'float64'.")
    dispersion = raw.get("dispersion")
    if not isinstance(dispersion, bool):
        raise ConfigurationError("'dispersion' must be a JSON boolean.")

    expected_phases = raw.get("expected_phases")
    if (
        not isinstance(expected_phases, int)
        or isinstance(expected_phases, bool)
        or expected_phases != len(EXPECTED_PHASE_ORDER)
    ):
        raise ConfigurationError(
            f"'expected_phases' must equal {len(EXPECTED_PHASE_ORDER)}."
        )
    raw_phase_order = raw.get("phase_order")
    if not isinstance(raw_phase_order, list) or not all(
        isinstance(value, str) and value.strip() for value in raw_phase_order
    ):
        raise ConfigurationError("'phase_order' must be an array of phase keys.")
    phase_keys = [str(value).strip() for value in raw_phase_order]
    if len(set(phase_keys)) != len(phase_keys):
        duplicates = sorted(
            key for key, count in Counter(phase_keys).items() if count > 1
        )
        raise ConfigurationError(
            "Duplicate phases in 'phase_order': " + ", ".join(duplicates)
        )
    if len(phase_keys) != expected_phases:
        raise ConfigurationError(
            "The number of phase definitions does not match 'expected_phases'."
        )
    if tuple(phase_keys) != EXPECTED_PHASE_ORDER:
        raise ConfigurationError(
            "'phase_order' must contain the five documented phases in order: "
            + ", ".join(EXPECTED_PHASE_ORDER)
        )

    phases: list[PhaseDefinition] = []
    for order, phase_key in enumerate(phase_keys):
        if not SAFE_PHASE_KEY.fullmatch(phase_key) or phase_key in {".", ".."}:
            raise ConfigurationError(f"Unsafe phase key: {phase_key!r}")
        try:
            composition = dependencies.composition_class(phase_key)
        except Exception as exc:
            raise ConfigurationError(
                f"Phase key is not a valid chemical composition: {phase_key!r}"
            ) from exc
        if set(composition.get_el_amt_dict()) != ALLOWED_ELEMENTS:
            raise ConfigurationError(
                f"Configured phase {phase_key!r} must contain only Al and Ni."
            )
        phases.append(
            PhaseDefinition(
                order=order,
                phase_key=phase_key,
                reduced_formula=composition.reduced_formula,
                composition_signature=composition_signature(composition),
            )
        )

    input_relative, input_directory = repository_relative_path(
        project_root, raw.get("input_directory"), "input_directory"
    )
    output_relative, output_directory = repository_relative_path(
        project_root, raw.get("output_directory"), "output_directory"
    )
    if input_directory == output_directory:
        raise ConfigurationError("Input and output directories must be different.")
    if output_directory.is_relative_to(input_directory) or input_directory.is_relative_to(
        output_directory
    ):
        raise ConfigurationError("Input and output directories must not overlap.")

    return EvaluationConfig(
        schema_version=schema_version,
        description=description,
        model_family=model_family,
        model_name=model_name,
        model_size=model_size,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        input_relative_path=input_relative,
        output_relative_path=output_relative,
        input_directory=input_directory,
        output_directory=output_directory,
        phases=tuple(phases),
    )


def resolve_runtime_options(
    args: argparse.Namespace,
    config: EvaluationConfig,
) -> RuntimeOptions:
    """Apply optional CLI settings over the validated configuration defaults."""

    model = config.model_size if args.model is None else str(args.model).strip()
    device = config.device if args.device is None else str(args.device).strip()
    dtype = config.default_dtype if args.dtype is None else str(args.dtype)
    if not model:
        raise ConfigurationError("--model must not be empty.")
    if not device:
        raise ConfigurationError("--device must not be empty.")
    if dtype not in {"float32", "float64"}:
        raise ConfigurationError("--dtype must be 'float32' or 'float64'.")
    return RuntimeOptions(model=model, device=device, dtype=dtype)


def model_provenance_text(model: str) -> str:
    """Normalize path separators so EXTXYZ string metadata round-trips safely."""

    return model.replace("\\", "/")


def choose_requested_phases(
    phases: Sequence[PhaseDefinition],
    requested_phase_key: str | None,
) -> list[PhaseDefinition]:
    """Select all phases or one exact, case-sensitive configured phase key."""

    if requested_phase_key is None:
        return list(phases)
    phase_by_key = {phase.phase_key: phase for phase in phases}
    try:
        return [phase_by_key[requested_phase_key]]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown --phase value {requested_phase_key!r}. Valid phases: "
            + ", ".join(phase_by_key)
        ) from exc


def utc_timestamp() -> str:
    """Return a timezone-aware, second-resolution UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def relative_path_text(path: Path) -> str:
    """Serialize a safe repository-relative path with forward slashes."""

    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ConfigurationError(f"Unsafe repository-relative path: {path}")
    return path.as_posix()


def phase_paths(
    config: EvaluationConfig,
    phase: PhaseDefinition,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return absolute and relative source/metadata/output paths for a phase."""

    input_relative = config.input_relative_path / f"{phase.phase_key}.extxyz"
    metadata_relative = (
        config.input_relative_path / f"{phase.phase_key}.metadata.json"
    )
    output_relative = (
        config.output_relative_path
        / STRUCTURES_SUBDIRECTORY
        / f"{phase.phase_key}_mace_zero_shot.extxyz"
    )
    return (
        input_relative,
        metadata_relative,
        output_relative,
        config.input_directory / f"{phase.phase_key}.extxyz",
        config.input_directory / f"{phase.phase_key}.metadata.json",
        config.output_directory
        / STRUCTURES_SUBDIRECTORY
        / f"{phase.phase_key}_mace_zero_shot.extxyz",
    )


def read_metadata_object(path: Path, phase_key: str) -> dict[str, JSONValue]:
    """Read one strict metadata object and translate JSON errors to input errors."""

    if not path.is_file():
        raise InputValidationError(
            f"Missing metadata for {phase_key}: {path}"
        )
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonstandard_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            f"Metadata for {phase_key} is not valid UTF-8: {path}"
        ) from exc
    except (json.JSONDecodeError, DuplicateJsonKeyError, ValueError) as exc:
        detail = (
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        raise InputValidationError(
            f"Invalid metadata JSON for {phase_key}: {detail}"
        ) from exc
    except OSError as exc:
        raise InputValidationError(
            f"Could not read metadata for {phase_key}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InputValidationError(
            f"Metadata root for {phase_key} must be a JSON object."
        )
    return {str(key): value for key, value in raw.items()}


def metadata_string(
    metadata: Mapping[str, JSONValue],
    field_name: str,
    phase_key: str,
) -> str:
    """Require a nonempty string in selected-structure metadata."""

    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(
            f"Metadata for {phase_key} requires nonempty '{field_name}'."
        )
    return value.strip()


def validate_phase_input(
    config: EvaluationConfig,
    phase: PhaseDefinition,
    dependencies: ScientificDependencies,
) -> PhaseInput:
    """Load and validate one selected structure without changing it."""

    (
        input_relative,
        metadata_relative,
        output_relative,
        input_path,
        metadata_path,
        _,
    ) = phase_paths(config, phase)
    if not input_path.is_file():
        raise InputValidationError(
            f"Missing selected structure for {phase.phase_key}: {input_path}"
        )

    metadata = read_metadata_object(metadata_path, phase.phase_key)
    metadata_phase_key = metadata_string(metadata, "phase_key", phase.phase_key)
    if metadata_phase_key != phase.phase_key:
        raise InputValidationError(
            f"Metadata phase mismatch for {phase.phase_key}: "
            f"found {metadata_phase_key!r}."
        )
    material_id = metadata_string(metadata, "material_id", phase.phase_key)
    source_formula = metadata_string(metadata, "formula_pretty", phase.phase_key)
    symmetry_symbol = metadata_string(
        metadata, "symmetry_symbol", phase.phase_key
    )
    symmetry_number = metadata.get("symmetry_number")
    if (
        not isinstance(symmetry_number, int)
        or isinstance(symmetry_number, bool)
        or symmetry_number <= 0
    ):
        raise InputValidationError(
            f"Metadata for {phase.phase_key} has invalid 'symmetry_number'."
        )

    try:
        metadata_composition = dependencies.composition_class(source_formula)
    except MemoryError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Invalid metadata formula for {phase.phase_key}: {source_formula!r}"
        ) from exc
    if composition_signature(metadata_composition) != phase.composition_signature:
        raise InputValidationError(
            f"Metadata formula for {phase.phase_key} does not match its configured "
            f"composition: {source_formula!r}."
        )
    reduced_metadata = metadata.get("reduced_composition")
    if reduced_metadata is not None:
        if not isinstance(reduced_metadata, str) or not reduced_metadata.strip():
            raise InputValidationError(
                f"Metadata for {phase.phase_key} has invalid 'reduced_composition'."
            )
        try:
            reduced_composition = dependencies.composition_class(reduced_metadata)
        except MemoryError:
            raise
        except Exception as exc:
            raise InputValidationError(
                f"Invalid reduced composition metadata for {phase.phase_key}."
            ) from exc
        if composition_signature(reduced_composition) != phase.composition_signature:
            raise InputValidationError(
                f"Reduced composition metadata does not match {phase.phase_key}."
            )

    try:
        frames = dependencies.ase_read(
            str(input_path), index=":", format="extxyz"
        )
    except MemoryError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Could not read selected EXTXYZ for {phase.phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise InputValidationError(
            f"Selected EXTXYZ for {phase.phase_key} must contain exactly one "
            f"structure; found {len(frames)}."
        )
    atoms = frames[0]

    # Selected files are source geometries, not result containers. Rejecting
    # any attached calculator prevents stale ASE or MACE results from being
    # mistaken for the current zero-shot evaluation.
    if atoms.calc is not None:
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} has a pre-attached "
            f"calculator ({type(atoms.calc).__name__})."
        )
    generated_prefixes = ("MACE_", "mace_")
    stale_keys = [
        key
        for key in (*atoms.info.keys(), *atoms.arrays.keys())
        if str(key).startswith(generated_prefixes)
    ]
    if stale_keys:
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} contains prior MACE "
            "annotations: " + ", ".join(sorted(map(str, stale_keys)))
        )

    number_of_atoms = len(atoms)
    if number_of_atoms <= 0:
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} contains no atoms."
        )
    if not bool(dependencies.numpy.asarray(atoms.pbc, dtype=bool).all()):
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} is not periodic in all "
            "three directions."
        )
    positions = dependencies.numpy.asarray(atoms.positions)
    cell = dependencies.numpy.asarray(atoms.cell.array)
    if positions.shape != (number_of_atoms, 3):
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} has invalid positions shape "
            f"{positions.shape}."
        )
    if cell.shape != (3, 3):
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} has invalid cell shape "
            f"{cell.shape}."
        )
    if not bool(dependencies.numpy.isfinite(positions).all()):
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} has nonfinite positions."
        )
    if not bool(dependencies.numpy.isfinite(cell).all()):
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} has nonfinite cell values."
        )
    try:
        volume = float(atoms.get_volume())
    except MemoryError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Could not determine cell volume for {phase.phase_key}: {exc}"
        ) from exc
    if not math.isfinite(volume) or volume <= 0.0:
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} has invalid cell volume "
            f"{volume!r}."
        )

    symbols = atoms.get_chemical_symbols()
    element_set = set(symbols)
    if not element_set.issubset(ALLOWED_ELEMENTS):
        unexpected = sorted(element_set.difference(ALLOWED_ELEMENTS))
        raise InputValidationError(
            f"Selected structure for {phase.phase_key} contains unsupported "
            "elements: " + ", ".join(unexpected)
        )
    try:
        structure_composition = dependencies.composition_class(Counter(symbols))
    except MemoryError:
        raise
    except Exception as exc:
        raise InputValidationError(
            f"Could not determine composition for {phase.phase_key}."
        ) from exc
    if composition_signature(structure_composition) != phase.composition_signature:
        raise InputValidationError(
            f"Composition mismatch for {phase.phase_key}: structure reduces to "
            f"{structure_composition.reduced_formula}, expected "
            f"{phase.reduced_formula}."
        )

    metadata_sites = metadata.get("number_of_sites")
    if (
        not isinstance(metadata_sites, int)
        or isinstance(metadata_sites, bool)
        or metadata_sites != number_of_atoms
    ):
        raise InputValidationError(
            f"Metadata atom count for {phase.phase_key} does not match the "
            f"structure ({metadata_sites!r} versus {number_of_atoms})."
        )
    embedded_phase = atoms.info.get("phase_key")
    embedded_material = atoms.info.get("material_id")
    embedded_formula = atoms.info.get("formula_pretty")
    if embedded_phase != phase.phase_key:
        raise InputValidationError(
            f"EXTXYZ phase metadata for {phase.phase_key} is missing or mismatched."
        )
    if embedded_material != material_id:
        raise InputValidationError(
            f"EXTXYZ material ID for {phase.phase_key} does not match metadata."
        )
    if embedded_formula != source_formula:
        raise InputValidationError(
            f"EXTXYZ formula for {phase.phase_key} does not match metadata."
        )
    LOGGER.debug(
        "Validated %s: %s, %d atoms, %s (%d), %.12g A^3.",
        phase.phase_key,
        material_id,
        number_of_atoms,
        symmetry_symbol,
        symmetry_number,
        volume,
    )

    return PhaseInput(
        phase=phase,
        atoms=atoms,
        metadata=metadata,
        input_structure_relative_path=input_relative,
        metadata_relative_path=metadata_relative,
        output_structure_relative_path=output_relative,
    )


def log_phase_failure(
    outcome: PhaseOutcome,
    exc: BaseException,
    verbose: bool,
) -> None:
    """Record one phase failure and optionally include its traceback."""

    outcome.completed = False
    outcome.result = None
    outcome.error_type = type(exc).__name__
    outcome.error_message = str(exc).strip() or "no additional detail was provided"
    outcome.failure_time_utc = utc_timestamp()
    LOGGER.error(
        "Phase %s failed (%s): %s",
        outcome.phase.phase_key,
        outcome.error_type,
        outcome.error_message,
    )
    if verbose:
        LOGGER.debug(traceback.format_exc().rstrip())


def discover_and_validate_inputs(
    config: EvaluationConfig,
    requested_phases: Sequence[PhaseDefinition],
    dependencies: ScientificDependencies,
    verbose: bool,
) -> list[PhaseOutcome]:
    """Discover each configured pair and collect all safe validation failures."""

    # Files are derived from the configured phase order instead of glob order,
    # so Al3Ni5 cannot be accidentally placed before AlNi in reports.
    outcomes: list[PhaseOutcome] = []
    for phase in requested_phases:
        outcome = PhaseOutcome(phase=phase)
        try:
            outcome.phase_input = validate_phase_input(config, phase, dependencies)
        except MemoryError:
            raise
        except Exception as exc:
            log_phase_failure(outcome, exc, verbose)
        outcomes.append(outcome)
    return outcomes


def summary_relative_paths(config: EvaluationConfig) -> tuple[Path, Path, Path]:
    """Return repository-relative CSV, JSON, and text report paths."""

    return (
        config.output_relative_path / TABLES_SUBDIRECTORY / CSV_FILENAME,
        config.output_relative_path / TABLES_SUBDIRECTORY / JSON_FILENAME,
        config.output_relative_path / REPORTS_SUBDIRECTORY / TEXT_REPORT_FILENAME,
    )


def acquire_run_lock(project_root: Path) -> Path:
    """Atomically exclude concurrent real runs that share fixed output paths."""

    lock_path = project_root / RUN_LOCK_FILENAME
    lock_document = {
        "pid": os.getpid(),
        "started_at_utc": utc_timestamp(),
        "purpose": "Ni-Al MACE Step 5 output publication lock",
    }
    created = False
    try:
        with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
            created = True
            json.dump(lock_document, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise OutputCollisionError(
            f"Another Step 5 run or a stale run lock exists: {lock_path}. "
            "Confirm that no evaluation is active before removing a stale lock."
        ) from exc
    except BaseException as exc:
        if created and lock_path.is_file():
            try:
                lock_path.unlink()
            except OSError:
                pass
        if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
            raise
        raise PublicationError(
            f"Could not create the Step 5 run lock {lock_path}: {exc}"
        ) from exc
    return lock_path


def release_run_lock(lock_path: Path) -> None:
    """Remove the exact lock owned by the current real run."""

    try:
        lock_path.unlink()
    except OSError as exc:
        raise PublicationError(
            f"Could not remove the Step 5 run lock {lock_path}: {exc}"
        ) from exc


def validate_existing_annotated_structure(
    source_path: Path,
    output_path: Path,
    record: Mapping[str, Any],
    dependencies: ScientificDependencies,
) -> None:
    """Verify an existing successful EXTXYZ against its source and JSON row."""

    np = dependencies.numpy
    try:
        source_frames = dependencies.ase_read(
            str(source_path), index=":", format="extxyz"
        )
        output_frames = dependencies.ase_read(
            str(output_path), index=":", format="extxyz"
        )
    except MemoryError:
        raise
    except Exception as exc:
        raise PartialOutputError(
            f"Could not read an existing annotated structure: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(source_frames, list):
        source_frames = [source_frames]
    if not isinstance(output_frames, list):
        output_frames = [output_frames]
    if len(source_frames) != 1 or len(output_frames) != 1:
        raise PartialOutputError(
            f"Existing source/output frame count is invalid for {record['phase_key']}."
        )
    source = source_frames[0]
    output = output_frames[0]
    if output.calc is not None:
        raise PartialOutputError(
            f"Existing annotated structure has a calculator for {record['phase_key']}."
        )
    if (
        not np.array_equal(source.numbers, output.numbers)
        or not np.array_equal(source.pbc, output.pbc)
        or not np.allclose(source.positions, output.positions, rtol=0.0, atol=5.0e-9)
        or not np.allclose(source.cell.array, output.cell.array, rtol=0.0, atol=5.0e-12)
    ):
        raise PartialOutputError(
            f"Existing annotated structure does not preserve {record['phase_key']}."
        )
    forces = output.arrays.get("MACE_forces_eV_per_A")
    if forces is None or np.asarray(forces).shape != (len(source), 3) or not bool(
        np.isfinite(forces).all()
    ):
        raise PartialOutputError(
            f"Existing annotated forces are invalid for {record['phase_key']}."
        )
    required_info = {
        "phase_key": record["phase_key"],
        "source_material_id": record["material_id"],
        "source_formula": record["formula"],
        "model_name": record["mace_model_name"],
        "model_size": record["model_size"],
        "device": record["device"],
        "default_dtype": record["dtype"],
        "dispersion": record["dispersion_enabled"],
        "evaluation_type": EVALUATION_TYPE,
        "evaluation_time_utc": record["evaluation_time_utc"],
    }
    if any(output.info.get(key) != value for key, value in required_info.items()):
        raise PartialOutputError(
            f"Existing annotated provenance is invalid for {record['phase_key']}."
        )
    energy = output.info.get("MACE_total_energy_eV")
    stress = np.asarray(output.info.get("MACE_stress_eV_per_A3"))
    expected_stress = np.asarray(
        [record[f"stress_{label}_eV_per_A3"] for label in STRESS_LABELS],
        dtype=np.float64,
    )
    if (
        not isinstance(energy, (int, float))
        or not math.isclose(
            float(energy), float(record["total_energy_eV"]),
            rel_tol=1.0e-14, abs_tol=1.0e-14,
        )
        or stress.shape != (6,)
        or not np.allclose(stress, expected_stress, rtol=0.0, atol=1.0e-14)
    ):
        raise PartialOutputError(
            f"Existing annotated results disagree with JSON for {record['phase_key']}."
        )


def validate_output_paths(
    project_root: Path,
    config: EvaluationConfig,
    requested_phases: Sequence[PhaseDefinition],
    overwrite: bool,
    validation_only: bool = False,
    dependencies: ScientificDependencies | None = None,
) -> None:
    """Validate output paths and any prior result snapshot before replacement."""

    for directory in (
        config.output_directory,
        config.output_directory / STRUCTURES_SUBDIRECTORY,
        config.output_directory / TABLES_SUBDIRECTORY,
        config.output_directory / REPORTS_SUBDIRECTORY,
    ):
        if directory.exists() and not directory.is_dir():
            raise PartialOutputError(
                f"Expected an output directory but found another object: {directory}"
            )

    all_structure_targets = {
        phase.phase_key: phase_paths(config, phase)[5] for phase in config.phases
    }
    csv_relative, json_relative, report_relative = summary_relative_paths(config)
    summary_targets = [
        project_root / csv_relative,
        project_root / json_relative,
        project_root / report_relative,
    ]
    for target in (*all_structure_targets.values(), *summary_targets):
        if target.exists() and not target.is_file():
            raise PartialOutputError(
                f"Expected a generated result file but found another object: {target}"
            )

    summary_present = [target.is_file() for target in summary_targets]
    existing_structures = {
        phase_key: target
        for phase_key, target in all_structure_targets.items()
        if target.is_file()
    }
    if any(summary_present) and not all(summary_present):
        missing = [
            str(path)
            for path, present in zip(summary_targets, summary_present)
            if not present
        ]
        raise PartialOutputError(
            "Partially written summary output detected; missing: "
            + ", ".join(missing)
        )
    if not any(summary_present) and existing_structures:
        raise PartialOutputError(
            "Annotated structures exist without their summary bundle: "
            + ", ".join(str(path) for path in existing_structures.values())
        )
    if not all(summary_present):
        return

    if any(target.stat().st_size == 0 for target in summary_targets):
        raise PartialOutputError("An existing Step 5 summary file is empty.")
    try:
        previous = json.loads(
            summary_targets[1].read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonstandard_json_constant,
        )
    except Exception as exc:
        raise PartialOutputError(
            f"Could not validate existing JSON summary: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        not isinstance(previous, dict)
        or previous.get("schema_version") != SCHEMA_VERSION
        or not isinstance(previous.get("records"), list)
    ):
        raise PartialOutputError(
            "Existing JSON summary has an unsupported or incomplete schema."
        )
    records = previous["records"]
    if not records or not all(isinstance(record, dict) for record in records):
        raise PartialOutputError(
            "Existing JSON summary must contain one or more phase records."
        )
    if any(set(record) != set(SUMMARY_FIELDS) for record in records):
        raise PartialOutputError(
            "Existing JSON summary records have missing or extra fields."
        )
    record_phase_keys = [record.get("phase_key") for record in records]
    if not all(isinstance(key, str) for key in record_phase_keys):
        raise PartialOutputError(
            "Existing JSON summary has a non-string phase key."
        )
    normalized_phase_keys = [str(key) for key in record_phase_keys]
    if len(set(normalized_phase_keys)) != len(normalized_phase_keys):
        raise PartialOutputError(
            "Existing JSON summary contains duplicate phase records."
        )
    configured_order = [phase.phase_key for phase in config.phases]
    configured_keys = set(configured_order)
    if not set(normalized_phase_keys).issubset(configured_keys):
        raise PartialOutputError(
            "Existing JSON summary contains an unknown phase key."
        )
    expected_record_order = [
        key for key in configured_order if key in set(normalized_phase_keys)
    ]
    if normalized_phase_keys != expected_record_order:
        raise PartialOutputError(
            "Existing JSON summary phase records are not in configured order."
        )
    prior_requested = previous.get("requested_phases")
    if prior_requested != normalized_phase_keys:
        raise PartialOutputError(
            "Existing JSON summary requested phases do not match its records."
        )

    phase_by_key = {phase.phase_key: phase for phase in config.phases}
    expected_completed: list[str] = []
    expected_failed: list[str] = []
    for record in records:
        phase_key = str(record["phase_key"])
        target = all_structure_targets[phase_key]
        expected_input = relative_path_text(
            phase_paths(config, phase_by_key[phase_key])[0]
        )
        expected_output = relative_path_text(
            phase_paths(config, phase_by_key[phase_key])[2]
        )
        if record.get("input_structure_path") != expected_input:
            raise PartialOutputError(
                f"Existing summary has a noncanonical input path for {phase_key}."
            )
        status = record.get("evaluation_status")
        if status == "success":
            expected_completed.append(phase_key)
            if record.get("output_structure_path") != expected_output:
                raise PartialOutputError(
                    f"Existing summary has a noncanonical output path for {phase_key}."
                )
            if not target.is_file() or target.stat().st_size == 0:
                raise PartialOutputError(
                    f"Existing summary marks {phase_key} successful but its "
                    f"annotated structure is missing or empty: {target}"
                )
            if dependencies is not None:
                validate_existing_annotated_structure(
                    project_root / phase_paths(config, phase_by_key[phase_key])[0],
                    target,
                    record,
                    dependencies,
                )
        elif status == "failed":
            expected_failed.append(phase_key)
            if record.get("output_structure_path") is not None:
                raise PartialOutputError(
                    f"Existing failed record for {phase_key} has an output path."
                )
            if target.exists():
                raise PartialOutputError(
                    f"Existing failed record for {phase_key} has a stale "
                    f"annotated structure: {target}"
                )
        else:
            raise PartialOutputError(
                f"Existing summary has invalid status for {phase_key}."
            )
    if previous.get("completed_phases") != expected_completed or previous.get(
        "failed_phases"
    ) != expected_failed:
        raise PartialOutputError(
            "Existing JSON summary completion lists disagree with its records."
        )
    expected_overall = "success" if not expected_failed else "failure"
    if previous.get("overall_status") != expected_overall:
        raise PartialOutputError(
            "Existing JSON summary overall status disagrees with its records."
        )
    orphaned_phases = set(existing_structures).difference(normalized_phase_keys)
    if orphaned_phases:
        raise PartialOutputError(
            "Annotated structures are absent from the existing summary: "
            + ", ".join(sorted(orphaned_phases))
        )

    try:
        with summary_targets[0].open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(SUMMARY_FIELDS):
                raise PartialOutputError(
                    "Existing CSV summary has missing, extra, or reordered columns."
                )
            csv_records = list(reader)
    except (OSError, csv.Error) as exc:
        raise PartialOutputError(
            f"Could not validate existing CSV summary: {type(exc).__name__}: {exc}"
        ) from exc
    if [record.get("phase_key") for record in csv_records] != normalized_phase_keys:
        raise PartialOutputError(
            "Existing CSV and JSON summaries describe different phase records."
        )
    for json_record, csv_record in zip(records, csv_records):
        for field_name in SUMMARY_FIELDS:
            expected_csv = str(csv_cell(json_record.get(field_name)))
            if csv_record.get(field_name) != expected_csv:
                raise PartialOutputError(
                    "Existing CSV and JSON summaries disagree on "
                    f"{field_name!r} for {json_record['phase_key']}."
                )
    try:
        report_text = summary_targets[2].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PartialOutputError(
            f"Could not validate existing text report: {type(exc).__name__}: {exc}"
        ) from exc
    generated_at_utc = previous.get("generated_at_utc")
    model_document = previous.get("model")
    first_record = records[0]
    expected_model_document = {
        "family": config.model_family,
        "name": config.model_name,
        "size": first_record["model_size"],
        "device": first_record["device"],
        "dtype": first_record["dtype"],
        "dispersion_enabled": first_record["dispersion_enabled"],
    }
    if not isinstance(generated_at_utc, str) or model_document != expected_model_document:
        raise PartialOutputError(
            "Existing JSON model or generation provenance is inconsistent."
        )
    for record in records[1:]:
        if any(
            record[field_name] != first_record[field_name]
            for field_name in (
                "mace_model_name",
                "model_size",
                "device",
                "dtype",
                "dispersion_enabled",
            )
        ):
            raise PartialOutputError(
                "Existing JSON summary mixes incompatible model settings."
            )
    prior_options = RuntimeOptions(
        model=str(first_record["model_size"]),
        device=str(first_record["device"]),
        dtype=str(first_record["dtype"]),
    )
    expected_report = build_text_report(
        records,
        config,
        prior_options,
        generated_at_utc,
    )
    if report_text != expected_report:
        raise PartialOutputError(
            "Existing text report does not exactly match the JSON records."
        )

    current_requested = [phase.phase_key for phase in requested_phases]
    if not validation_only and overwrite and current_requested != prior_requested:
        raise OutputCollisionError(
            "The fixed Step 5 summary files currently describe a different phase "
            "set. To preserve a coherent bundle, rerun that same phase set or "
            "remove the reviewed outputs before starting a new selection."
        )
    if not validation_only and not overwrite:
        raise OutputCollisionError(
            "Step 5 summary output already exists. Use --overwrite to replace "
            "the generated result bundle."
        )


def exception_chain(exc: BaseException) -> list[BaseException]:
    """Return an exception and its explicit or implicit causes without loops."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def looks_like_download_failure(exc: BaseException) -> bool:
    """Identify common network/checkpoint errors from a MACE loading failure."""

    network_types = (HTTPError, URLError, TimeoutError, ConnectionError)
    if any(isinstance(item, network_types) for item in exception_chain(exc)):
        return True
    text = " ".join(str(item).lower() for item in exception_chain(exc))
    markers = (
        "download",
        "connection",
        "network",
        "http error",
        "urlopen",
        "name resolution",
        "timed out",
        "checkpoint not found",
    )
    return any(marker in text for marker in markers)


def load_calculator(
    config: EvaluationConfig,
    options: RuntimeOptions,
    dependencies: ScientificDependencies,
) -> Any:
    """Construct the shared pretrained MACE calculator exactly once."""

    LOGGER.info(
        "Loading %s %s model once (model=%s, device=%s, dtype=%s, dispersion=%s)...",
        config.model_family,
        config.model_name,
        options.model,
        options.device,
        options.dtype,
        str(config.dispersion).lower(),
    )
    try:
        calculator = dependencies.mace_factory(
            model=options.model,
            device=options.device,
            default_dtype=options.dtype,
            dispersion=config.dispersion,
        )
    except MemoryError:
        raise
    except Exception as exc:
        message = str(exc).strip() or "no additional detail was provided"
        if looks_like_download_failure(exc):
            raise ModelDownloadError(
                "Failed to download or locate the requested pretrained MACE "
                f"checkpoint: {type(exc).__name__}: {message}"
            ) from exc
        raise ModelLoadingError(
            "Failed to load the requested pretrained MACE model: "
            f"{type(exc).__name__}: {message}"
        ) from exc
    if calculator is None:
        raise ModelLoadingError("mace_mp returned no calculator.")
    LOGGER.info("MACE model loaded successfully.")
    return calculator


def require_finite_scalar(value: Any, label: str) -> float:
    """Convert one numerical result to a finite Python float."""

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise NonFiniteResultError(f"{label} is not a real scalar: {value!r}") from exc
    if not math.isfinite(converted):
        raise NonFiniteResultError(f"{label} is nonfinite: {converted!r}")
    return converted


def assert_structure_unchanged(
    source: Any,
    evaluated: Any,
    dependencies: ScientificDependencies,
    phase_key: str,
) -> None:
    """Confirm calculator evaluation preserved species, geometry, cell, and PBC."""

    np = dependencies.numpy
    if not np.array_equal(source.numbers, evaluated.numbers):
        raise InputValidationError(
            f"Calculator evaluation changed chemical species for {phase_key}."
        )
    if not np.array_equal(source.positions, evaluated.positions):
        raise InputValidationError(
            f"Calculator evaluation changed atomic positions for {phase_key}."
        )
    if not np.array_equal(source.cell.array, evaluated.cell.array):
        raise InputValidationError(
            f"Calculator evaluation changed cell vectors for {phase_key}."
        )
    if not np.array_equal(source.pbc, evaluated.pbc):
        raise InputValidationError(
            f"Calculator evaluation changed periodic boundaries for {phase_key}."
        )


def evaluate_single_point(
    phase_input: PhaseInput,
    calculator: Any,
    dependencies: ScientificDependencies,
) -> PhaseResult:
    """Calculate and validate one energy, force array, and ASE-order stress."""

    np = dependencies.numpy
    source = phase_input.atoms
    working = source.copy()
    if working.calc is not None:
        raise InputValidationError(
            f"Working copy for {phase_input.phase.phase_key} unexpectedly retained "
            "a calculator."
        )
    # Only the isolated working copy receives MACE. The pristine object remains
    # calculator-free and becomes the source for the annotated output copy.
    working.calc = calculator
    try:
        try:
            total_energy = float(working.get_potential_energy())
        except MemoryError:
            raise
        except Exception as exc:
            raise EnergyCalculationError(
                f"Energy calculation failed for {phase_input.phase.phase_key}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            forces = np.asarray(working.get_forces(), dtype=np.float64).copy()
        except MemoryError:
            raise
        except Exception as exc:
            raise ForceCalculationError(
                f"Force calculation failed for {phase_input.phase.phase_key}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            stress = np.asarray(
                working.get_stress(voigt=True), dtype=np.float64
            ).copy()
        except MemoryError:
            raise
        except Exception as exc:
            raise StressCalculationError(
                f"Stress calculation failed for {phase_input.phase.phase_key}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        assert_structure_unchanged(
            source, working, dependencies, phase_input.phase.phase_key
        )
    finally:
        working.calc = None

    number_of_atoms = len(source)
    if forces.shape != (number_of_atoms, 3):
        raise ForceCalculationError(
            f"Force array for {phase_input.phase.phase_key} has shape "
            f"{forces.shape}; expected ({number_of_atoms}, 3)."
        )
    if stress.shape != (6,):
        raise StressCalculationError(
            f"Stress for {phase_input.phase.phase_key} has shape {stress.shape}; "
            "expected six ASE Voigt components."
        )
    total_energy = require_finite_scalar(
        total_energy, f"total energy for {phase_input.phase.phase_key}"
    )
    if not bool(np.isfinite(forces).all()):
        raise NonFiniteResultError(
            f"Forces for {phase_input.phase.phase_key} contain NaN or infinity."
        )
    if not bool(np.isfinite(stress).all()):
        raise NonFiniteResultError(
            f"Stress for {phase_input.phase.phase_key} contains NaN or infinity."
        )

    # Force statistics use per-atom vector magnitudes. In particular, RMS is
    # sqrt(mean(|F_i|^2)), not a component-wise RMS over the N-by-3 array.
    magnitudes = np.linalg.norm(forces, axis=1)
    total_force = np.sum(forces, axis=0, dtype=np.float64)
    volume = require_finite_scalar(
        source.get_volume(), f"cell volume for {phase_input.phase.phase_key}"
    )
    derived_values = {
        "energy per atom": total_energy / number_of_atoms,
        "volume per atom": volume / number_of_atoms,
        "maximum force": np.max(magnitudes),
        "mean force": np.mean(magnitudes),
        "RMS force": np.sqrt(np.mean(np.square(magnitudes))),
        "minimum force": np.min(magnitudes),
        "total force norm": np.linalg.norm(total_force),
    }
    finite_derived = {
        name: require_finite_scalar(
            value, f"{name} for {phase_input.phase.phase_key}"
        )
        for name, value in derived_values.items()
    }
    if not bool(np.isfinite(total_force).all()):
        raise NonFiniteResultError(
            f"Total force for {phase_input.phase.phase_key} is nonfinite."
        )

    return PhaseResult(
        total_energy_eV=total_energy,
        energy_per_atom_eV=finite_derived["energy per atom"],
        forces_eV_per_A=forces,
        stress_eV_per_A3=stress,
        number_of_atoms=number_of_atoms,
        volume_A3=volume,
        volume_per_atom_A3=finite_derived["volume per atom"],
        maximum_force_eV_per_A=finite_derived["maximum force"],
        mean_force_eV_per_A=finite_derived["mean force"],
        rms_force_eV_per_A=finite_derived["RMS force"],
        minimum_force_eV_per_A=finite_derived["minimum force"],
        total_force_eV_per_A=np.asarray(total_force, dtype=np.float64),
        total_force_norm_eV_per_A=finite_derived["total force norm"],
        evaluation_time_utc=utc_timestamp(),
    )


def annotation_info(
    phase_input: PhaseInput,
    result: PhaseResult,
    config: EvaluationConfig,
    options: RuntimeOptions,
) -> dict[str, Any]:
    """Build controlled scalar and vector annotations for one output structure."""

    return {
        "MACE_total_energy_eV": result.total_energy_eV,
        "MACE_energy_per_atom_eV": result.energy_per_atom_eV,
        "MACE_stress_eV_per_A3": result.stress_eV_per_A3,
        "phase_key": phase_input.phase.phase_key,
        "source_material_id": phase_input.metadata["material_id"],
        "source_formula": phase_input.metadata["formula_pretty"],
        "model_family": config.model_family,
        "model_name": config.model_name,
        "model_size": model_provenance_text(options.model),
        "device": options.device,
        "default_dtype": options.dtype,
        "dispersion": config.dispersion,
        "evaluation_type": EVALUATION_TYPE,
        "evaluation_time_utc": result.evaluation_time_utc,
        "ASE_stress_order": "xx yy zz yz xz xy",
    }


def verify_annotated_structure(
    path: Path,
    phase_input: PhaseInput,
    result: PhaseResult,
    config: EvaluationConfig,
    options: RuntimeOptions,
    dependencies: ScientificDependencies,
) -> None:
    """Read back a staged EXTXYZ and verify geometry and MACE annotations."""

    np = dependencies.numpy
    try:
        frames = dependencies.ase_read(str(path), index=":", format="extxyz")
    except MemoryError:
        raise
    except Exception as exc:
        raise StructureWriteError(
            f"Could not read back annotated EXTXYZ for "
            f"{phase_input.phase.phase_key}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise StructureWriteError(
            f"Annotated EXTXYZ for {phase_input.phase.phase_key} contains "
            f"{len(frames)} frames instead of one."
        )
    written = frames[0]
    source = phase_input.atoms
    if written.calc is not None:
        raise StructureWriteError(
            f"Annotated EXTXYZ for {phase_input.phase.phase_key} acquired an "
            "unexpected ASE calculator."
        )
    if not np.array_equal(written.numbers, source.numbers):
        raise StructureWriteError(
            f"Annotated EXTXYZ changed species for {phase_input.phase.phase_key}."
        )
    if not np.array_equal(written.pbc, source.pbc):
        raise StructureWriteError(
            f"Annotated EXTXYZ changed PBC for {phase_input.phase.phase_key}."
        )
    if not np.allclose(
        written.positions, source.positions, rtol=0.0, atol=5.0e-9
    ):
        raise StructureWriteError(
            f"Annotated EXTXYZ changed positions for {phase_input.phase.phase_key}."
        )
    if not np.allclose(
        written.cell.array, source.cell.array, rtol=0.0, atol=5.0e-12
    ):
        raise StructureWriteError(
            f"Annotated EXTXYZ changed cell vectors for {phase_input.phase.phase_key}."
        )
    written_forces = written.arrays.get("MACE_forces_eV_per_A")
    if written_forces is None or np.asarray(written_forces).shape != (
        result.number_of_atoms,
        3,
    ):
        raise StructureWriteError(
            f"Annotated EXTXYZ has missing or invalid MACE forces for "
            f"{phase_input.phase.phase_key}."
        )
    if not np.allclose(
        written_forces, result.forces_eV_per_A, rtol=0.0, atol=5.0e-8
    ):
        raise StructureWriteError(
            f"Annotated EXTXYZ forces failed read-back verification for "
            f"{phase_input.phase.phase_key}."
        )
    written_energy = written.info.get("MACE_total_energy_eV")
    if not isinstance(written_energy, (int, float)) or not math.isclose(
        float(written_energy), result.total_energy_eV, rel_tol=1.0e-14, abs_tol=1.0e-14
    ):
        raise StructureWriteError(
            f"Annotated EXTXYZ energy failed read-back verification for "
            f"{phase_input.phase.phase_key}."
        )
    written_stress = np.asarray(written.info.get("MACE_stress_eV_per_A3"))
    if written_stress.shape != (6,) or not np.allclose(
        written_stress, result.stress_eV_per_A3, rtol=0.0, atol=1.0e-14
    ):
        raise StructureWriteError(
            f"Annotated EXTXYZ stress failed read-back verification for "
            f"{phase_input.phase.phase_key}."
        )
    expected_info = annotation_info(phase_input, result, config, options)
    for key, expected_value in expected_info.items():
        if key not in written.info:
            raise StructureWriteError(
                f"Annotated EXTXYZ is missing required field {key!r} for "
                f"{phase_input.phase.phase_key}."
            )
        actual_value = written.info[key]
        if isinstance(expected_value, np.ndarray):
            values_match = np.allclose(
                np.asarray(actual_value),
                expected_value,
                rtol=0.0,
                atol=1.0e-14,
            )
        elif isinstance(expected_value, float):
            values_match = isinstance(actual_value, (int, float)) and math.isclose(
                float(actual_value), expected_value, rel_tol=1.0e-14, abs_tol=1.0e-14
            )
        else:
            values_match = actual_value == expected_value
        if not bool(values_match):
            raise StructureWriteError(
                f"Annotated EXTXYZ field {key!r} failed read-back verification "
                f"for {phase_input.phase.phase_key}."
            )


def stage_annotated_structure(
    staging_root: Path,
    phase_input: PhaseInput,
    result: PhaseResult,
    config: EvaluationConfig,
    options: RuntimeOptions,
    dependencies: ScientificDependencies,
) -> None:
    """Create and verify an annotated copy without touching the source file."""

    annotated = phase_input.atoms.copy()
    annotated.calc = None
    generated_info = annotation_info(phase_input, result, config, options)
    collisions = sorted(
        key
        for key, value in generated_info.items()
        if key in annotated.info and annotated.info[key] != value
    )
    if "MACE_forces_eV_per_A" in annotated.arrays:
        collisions.append("MACE_forces_eV_per_A")
    if collisions:
        raise StructureWriteError(
            f"Source structure for {phase_input.phase.phase_key} already contains "
            "generated annotation keys: " + ", ".join(collisions)
        )
    annotated.info.update(generated_info)
    annotated.arrays["MACE_forces_eV_per_A"] = (
        dependencies.numpy.asarray(result.forces_eV_per_A).copy()
    )
    staged_path = staging_root / phase_input.output_structure_relative_path
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # write_results=False prevents ASE from adding canonical energy/forces/
        # stress fields that would attach a stale SinglePointCalculator on read.
        dependencies.ase_write(
            str(staged_path),
            annotated,
            format="extxyz",
            write_results=False,
        )
    except MemoryError:
        raise
    except Exception as exc:
        raise StructureWriteError(
            f"Could not write annotated EXTXYZ for {phase_input.phase.phase_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not staged_path.is_file() or staged_path.stat().st_size == 0:
        raise StructureWriteError(
            f"Annotated EXTXYZ staging produced no complete file for "
            f"{phase_input.phase.phase_key}."
        )
    verify_annotated_structure(
        staged_path, phase_input, result, config, options, dependencies
    )


def reset_calculator(calculator: Any, failed_phase: str) -> bool:
    """Reset cached ASE results after a phase failure before calculator reuse."""

    reset = getattr(calculator, "reset", None)
    if not callable(reset):
        LOGGER.error(
            "The shared calculator cannot be reset after %s failed; remaining "
            "phases cannot be evaluated safely.",
            failed_phase,
        )
        return False
    try:
        reset()
    except MemoryError:
        raise
    except Exception as exc:
        LOGGER.error(
            "Could not reset the shared calculator after %s (%s: %s).",
            failed_phase,
            type(exc).__name__,
            exc,
        )
        return False
    return True


def evaluate_phases(
    staging_root: Path,
    outcomes: Sequence[PhaseOutcome],
    calculator: Any,
    config: EvaluationConfig,
    options: RuntimeOptions,
    dependencies: ScientificDependencies,
    verbose: bool,
) -> None:
    """Evaluate valid phases, stage outputs, and isolate safe phase failures."""

    calculator_usable = True
    for outcome in outcomes:
        if outcome.phase_input is None:
            continue
        if not calculator_usable:
            log_phase_failure(
                outcome,
                ModelLoadingError(
                    "The shared MACE calculator became unusable after an earlier "
                    "phase failure."
                ),
                verbose,
            )
            continue
        LOGGER.info("Evaluating %s...", outcome.phase.phase_key)
        try:
            result = evaluate_single_point(
                outcome.phase_input, calculator, dependencies
            )
            # A phase is not successful until its calculator-free annotated
            # structure has been written and passed a full read-back check.
            stage_annotated_structure(
                staging_root,
                outcome.phase_input,
                result,
                config,
                options,
                dependencies,
            )
            outcome.result = result
            outcome.completed = True
            outcome.error_type = None
            outcome.error_message = None
            outcome.failure_time_utc = None
            LOGGER.info("  %s completed.", outcome.phase.phase_key)
        except MemoryError:
            raise
        except Exception as exc:
            log_phase_failure(outcome, exc, verbose)
            if isinstance(
                exc,
                (
                    EnergyCalculationError,
                    ForceCalculationError,
                    StressCalculationError,
                    NonFiniteResultError,
                    InputValidationError,
                ),
            ):
                calculator_usable = reset_calculator(
                    calculator, outcome.phase.phase_key
                )


def empty_record(
    outcome: PhaseOutcome,
    config: EvaluationConfig,
    options: RuntimeOptions,
) -> dict[str, JSONValue]:
    """Create a complete failed-record schema without fabricated numbers."""

    phase_input = outcome.phase_input
    metadata = phase_input.metadata if phase_input is not None else {}
    number_of_atoms = len(phase_input.atoms) if phase_input is not None else None
    volume_A3 = (
        float(phase_input.atoms.get_volume()) if phase_input is not None else None
    )
    input_relative = (
        phase_input.input_structure_relative_path
        if phase_input is not None
        else config.input_relative_path / f"{outcome.phase.phase_key}.extxyz"
    )
    return {
        "phase_key": outcome.phase.phase_key,
        "formula": (
            metadata.get("formula_pretty")
            if isinstance(metadata.get("formula_pretty"), str)
            else outcome.phase.reduced_formula
        ),
        "material_id": (
            metadata.get("material_id")
            if isinstance(metadata.get("material_id"), str)
            else None
        ),
        "number_of_atoms": number_of_atoms,
        "space_group_symbol": (
            metadata.get("symmetry_symbol")
            if isinstance(metadata.get("symmetry_symbol"), str)
            else None
        ),
        "space_group_number": (
            metadata.get("symmetry_number")
            if isinstance(metadata.get("symmetry_number"), int)
            and not isinstance(metadata.get("symmetry_number"), bool)
            else None
        ),
        "mace_model_name": config.model_name,
        "model_size": model_provenance_text(options.model),
        "device": options.device,
        "dtype": options.dtype,
        "dispersion_enabled": config.dispersion,
        "total_energy_eV": None,
        "energy_per_atom_eV": None,
        "volume_A3": volume_A3,
        "volume_per_atom_A3": (
            volume_A3 / number_of_atoms
            if volume_A3 is not None and number_of_atoms is not None
            else None
        ),
        "maximum_force_eV_per_A": None,
        "mean_force_eV_per_A": None,
        "rms_force_eV_per_A": None,
        "minimum_force_eV_per_A": None,
        "total_force_x_eV_per_A": None,
        "total_force_y_eV_per_A": None,
        "total_force_z_eV_per_A": None,
        "total_force_norm_eV_per_A": None,
        "stress_xx_eV_per_A3": None,
        "stress_yy_eV_per_A3": None,
        "stress_zz_eV_per_A3": None,
        "stress_yz_eV_per_A3": None,
        "stress_xz_eV_per_A3": None,
        "stress_xy_eV_per_A3": None,
        "input_structure_path": relative_path_text(input_relative),
        "output_structure_path": None,
        "evaluation_time_utc": outcome.failure_time_utc or utc_timestamp(),
        "evaluation_status": "failed",
        "error_type": outcome.error_type or "UnknownError",
        "error_message": outcome.error_message or "no additional detail",
    }


def outcome_record(
    outcome: PhaseOutcome,
    config: EvaluationConfig,
    options: RuntimeOptions,
) -> dict[str, JSONValue]:
    """Convert one outcome to the common CSV and JSON record schema."""

    if (
        not outcome.completed
        or outcome.result is None
        or outcome.phase_input is None
    ):
        return empty_record(outcome, config, options)

    phase_input = outcome.phase_input
    result = outcome.result
    metadata = phase_input.metadata
    total_force = result.total_force_eV_per_A
    stress = result.stress_eV_per_A3
    record: dict[str, JSONValue] = {
        "phase_key": outcome.phase.phase_key,
        "formula": str(metadata["formula_pretty"]),
        "material_id": str(metadata["material_id"]),
        "number_of_atoms": result.number_of_atoms,
        "space_group_symbol": str(metadata["symmetry_symbol"]),
        "space_group_number": int(metadata["symmetry_number"]),
        "mace_model_name": config.model_name,
        "model_size": model_provenance_text(options.model),
        "device": options.device,
        "dtype": options.dtype,
        "dispersion_enabled": config.dispersion,
        "total_energy_eV": result.total_energy_eV,
        "energy_per_atom_eV": result.energy_per_atom_eV,
        "volume_A3": result.volume_A3,
        "volume_per_atom_A3": result.volume_per_atom_A3,
        "maximum_force_eV_per_A": result.maximum_force_eV_per_A,
        "mean_force_eV_per_A": result.mean_force_eV_per_A,
        "rms_force_eV_per_A": result.rms_force_eV_per_A,
        "minimum_force_eV_per_A": result.minimum_force_eV_per_A,
        "total_force_x_eV_per_A": float(total_force[0]),
        "total_force_y_eV_per_A": float(total_force[1]),
        "total_force_z_eV_per_A": float(total_force[2]),
        "total_force_norm_eV_per_A": result.total_force_norm_eV_per_A,
        "stress_xx_eV_per_A3": float(stress[0]),
        "stress_yy_eV_per_A3": float(stress[1]),
        "stress_zz_eV_per_A3": float(stress[2]),
        "stress_yz_eV_per_A3": float(stress[3]),
        "stress_xz_eV_per_A3": float(stress[4]),
        "stress_xy_eV_per_A3": float(stress[5]),
        "input_structure_path": relative_path_text(
            phase_input.input_structure_relative_path
        ),
        "output_structure_path": relative_path_text(
            phase_input.output_structure_relative_path
        ),
        "evaluation_time_utc": result.evaluation_time_utc,
        "evaluation_status": "success",
        "error_type": None,
        "error_message": None,
    }
    if tuple(record) != SUMMARY_FIELDS:
        raise PartialOutputError("Internal summary record field order is invalid.")
    return record


def csv_cell(value: JSONValue) -> str | int:
    """Serialize CSV values without losing floating-point precision."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteResultError("A nonfinite value reached CSV serialization.")
        return format(value, ".17g")
    return value


def write_json_file(path: Path, value: JSONValue) -> None:
    """Write readable strict UTF-8 JSON inside the staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        path.write_text(serialized + "\n", encoding="utf-8", newline="\n")
    except (OSError, TypeError, ValueError) as exc:
        raise PartialOutputError(
            f"Could not write JSON summary {path}: {type(exc).__name__}: {exc}"
        ) from exc


def write_csv_file(
    path: Path,
    records: Sequence[Mapping[str, JSONValue]],
) -> None:
    """Write the tabular phase records with deterministic columns and precision."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(SUMMARY_FIELDS),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {field: csv_cell(record.get(field)) for field in SUMMARY_FIELDS}
                )
    except (OSError, csv.Error, ValueError) as exc:
        raise PartialOutputError(
            f"Could not write CSV summary {path}: {type(exc).__name__}: {exc}"
        ) from exc


def optional_report_value(value: JSONValue, unit: str = "") -> str:
    """Format one optional report value without inventing a failed result."""

    if value is None:
        return "not available"
    if isinstance(value, float):
        suffix = f" {unit}" if unit else ""
        return f"{value:.12g}{suffix}"
    suffix = f" {unit}" if unit else ""
    return f"{value}{suffix}"


def build_text_report(
    records: Sequence[Mapping[str, JSONValue]],
    config: EvaluationConfig,
    options: RuntimeOptions,
    generated_at_utc: str,
) -> str:
    """Build the human-readable scientific and execution report."""

    completed = sum(record["evaluation_status"] == "success" for record in records)
    failed = len(records) - completed
    lines = [
        "Ni-Al MACE-MP-0 Zero-Shot Single-Point Evaluation",
        "==================================================",
        "",
        "Purpose",
        "-------",
        f"Evaluate whether the pretrained {config.model_name} model value "
        f"'{model_provenance_text(options.model)}' can",
        "compute finite energies, forces, and stresses for the requested",
        f"{len(records)} selected Ni-Al intermetallic phase(s) without additional "
        "training.",
        "",
        "Evaluation definition",
        "---------------------",
        "Zero-shot means the pretrained model is applied directly to structures",
        "that were not used for any project-specific training or fine-tuning.",
        "No training or fine-tuning occurred in this step.",
        "No geometry optimization or atomic/cell relaxation occurred in this step.",
        "Each input was evaluated once at its downloaded Materials Project geometry.",
        "",
        "Model and execution",
        "-------------------",
        f"Model family: {config.model_family}",
        f"Model name: {config.model_name}",
        f"Model value/size: {model_provenance_text(options.model)}",
        f"Execution device: {options.device}",
        f"Numerical precision: {options.dtype}",
        f"Dispersion correction enabled: {str(config.dispersion).lower()}",
        f"Evaluation type: {EVALUATION_TYPE}",
        f"Report generation time (UTC): {generated_at_utc}",
        "",
        "Stress convention",
        "-----------------",
        "Stress is reported in ASE Voigt order: xx, yy, zz, yz, xz, xy.",
        "Units are eV/angstrom^3. ASE uses positive stress for tension and",
        "negative diagonal stress for hydrostatic compression.",
        "",
        "Energy-reference limitation",
        "---------------------------",
        "Materials Project formation energies are not directly compared with raw",
        "MACE total energies here because the reference states and calculation",
        "conventions may differ. No formation energy or elemental-reference",
        "subtraction is performed in Step 5.",
        "",
        "Phase results",
        "=============",
    ]
    for record in records:
        phase_key = str(record["phase_key"])
        lines.extend(
            [
                "",
                phase_key,
                "-" * len(phase_key),
                f"Status: {record['evaluation_status']}",
                f"Formula: {optional_report_value(record['formula'])}",
                f"Materials Project ID: {optional_report_value(record['material_id'])}",
                "Space group: "
                f"{optional_report_value(record['space_group_symbol'])} "
                f"({optional_report_value(record['space_group_number'])})",
                f"Number of atoms: {optional_report_value(record['number_of_atoms'])}",
                "Total energy: "
                f"{optional_report_value(record['total_energy_eV'], 'eV')}",
                "Energy per atom: "
                f"{optional_report_value(record['energy_per_atom_eV'], 'eV/atom')}",
                f"Cell volume: {optional_report_value(record['volume_A3'], 'angstrom^3')}",
                "Volume per atom: "
                f"{optional_report_value(record['volume_per_atom_A3'], 'angstrom^3/atom')}",
                "Maximum force magnitude: "
                f"{optional_report_value(record['maximum_force_eV_per_A'], 'eV/angstrom')}",
                "Mean force magnitude: "
                f"{optional_report_value(record['mean_force_eV_per_A'], 'eV/angstrom')}",
                "RMS force magnitude: "
                f"{optional_report_value(record['rms_force_eV_per_A'], 'eV/angstrom')}",
                "Minimum force magnitude: "
                f"{optional_report_value(record['minimum_force_eV_per_A'], 'eV/angstrom')}",
                "Total force vector (x, y, z): "
                f"[{optional_report_value(record['total_force_x_eV_per_A'])}, "
                f"{optional_report_value(record['total_force_y_eV_per_A'])}, "
                f"{optional_report_value(record['total_force_z_eV_per_A'])}] eV/angstrom",
                "Total force norm: "
                f"{optional_report_value(record['total_force_norm_eV_per_A'], 'eV/angstrom')}",
                "Stress (xx, yy, zz, yz, xz, xy): "
                f"[{optional_report_value(record['stress_xx_eV_per_A3'])}, "
                f"{optional_report_value(record['stress_yy_eV_per_A3'])}, "
                f"{optional_report_value(record['stress_zz_eV_per_A3'])}, "
                f"{optional_report_value(record['stress_yz_eV_per_A3'])}, "
                f"{optional_report_value(record['stress_xz_eV_per_A3'])}, "
                f"{optional_report_value(record['stress_xy_eV_per_A3'])}] eV/angstrom^3",
                f"Input structure: {record['input_structure_path']}",
                "Annotated output: "
                f"{optional_report_value(record['output_structure_path'])}",
                f"Evaluation time (UTC): {record['evaluation_time_utc']}",
            ]
        )
        if record["evaluation_status"] != "success":
            lines.extend(
                [
                    f"Error type: {record['error_type']}",
                    f"Error: {record['error_message']}",
                ]
            )

    lines.extend(
        [
            "",
            "Force interpretation",
            "--------------------",
            "Nonzero forces in this step are not by themselves evidence that the",
            "structure or model is incorrect, because the input geometry was optimized",
            "using a different energy model.",
            "They show that a Materials Project DFT geometry need not be a stationary",
            "point on the MACE potential-energy surface. Maximum force alone from this",
            "single-point calculation does not classify a phase as stable or unstable.",
            "",
            "Scientific limitations",
            "----------------------",
            "This is a software and physical-response baseline, not a complete accuracy",
            "validation against DFT or experiment. No relaxation, training, fine-tuning,",
            "molecular dynamics, LAMMPS, EAM, or MEAM calculation was performed.",
            "",
            "Final summary",
            "-------------",
            f"Requested phases: {len(records)}",
            f"Completed phases: {completed}",
            f"Failed phases: {failed}",
            f"Overall status: {'SUCCESS' if failed == 0 else 'FAILURE'}",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summary_outputs(
    staging_root: Path,
    project_root: Path,
    outcomes: Sequence[PhaseOutcome],
    config: EvaluationConfig,
    options: RuntimeOptions,
) -> tuple[list[dict[str, JSONValue]], tuple[Path, Path, Path]]:
    """Write CSV, JSON, and text reports from the same ordered records."""

    records = [outcome_record(outcome, config, options) for outcome in outcomes]
    records.sort(key=lambda record: EXPECTED_PHASE_ORDER.index(str(record["phase_key"])))
    generated_at_utc = utc_timestamp()
    completed_phases = [
        str(record["phase_key"])
        for record in records
        if record["evaluation_status"] == "success"
    ]
    failed_phases = [
        str(record["phase_key"])
        for record in records
        if record["evaluation_status"] != "success"
    ]
    json_document: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "description": (
            f"Zero-shot single-point evaluation of selected Ni-Al phases with "
            f"{config.model_name} model value "
            f"{model_provenance_text(options.model)}."
        ),
        "evaluation_type": EVALUATION_TYPE,
        "model": {
            "family": config.model_family,
            "name": config.model_name,
            "size": model_provenance_text(options.model),
            "device": options.device,
            "dtype": options.dtype,
            "dispersion_enabled": config.dispersion,
        },
        "stress_component_order": list(STRESS_LABELS),
        "stress_units": "eV/angstrom^3",
        "stress_sign_convention": (
            "ASE convention: positive stress is tensile; hydrostatic compression "
            "has negative diagonal components."
        ),
        "requested_phases": [outcome.phase.phase_key for outcome in outcomes],
        "completed_phases": completed_phases,
        "failed_phases": failed_phases,
        "overall_status": "success" if not failed_phases else "failure",
        "records": records,
    }
    csv_relative, json_relative, report_relative = summary_relative_paths(config)
    write_csv_file(staging_root / csv_relative, records)
    write_json_file(staging_root / json_relative, json_document)
    report_path = staging_root / report_relative
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_path.write_text(
            build_text_report(records, config, options, generated_at_utc),
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        raise PartialOutputError(
            f"Could not write text report {report_path}: {exc}"
        ) from exc
    for relative_path in (csv_relative, json_relative, report_relative):
        staged_path = staging_root / relative_path
        if not staged_path.is_file() or staged_path.stat().st_size == 0:
            raise PartialOutputError(
                f"Summary staging did not produce a complete file: "
                f"{relative_path_text(relative_path)}"
            )
    return records, (csv_relative, json_relative, report_relative)


def publication_priority(relative_path: Path) -> tuple[int, str]:
    """Publish verified structures before tables and the final text report."""

    if STRUCTURES_SUBDIRECTORY.name in relative_path.parts:
        priority = 0
    elif relative_path.suffix.lower() in {".csv", ".json"}:
        priority = 1
    else:
        priority = 2
    return priority, relative_path.as_posix()


def publish_staged_outputs(
    project_root: Path,
    staging_root: Path,
    relative_paths: Sequence[Path],
    obsolete_relative_paths: Sequence[Path],
    overwrite: bool,
) -> None:
    """Publish one run transactionally and restore prior files on failure."""

    unique_paths = sorted(set(relative_paths), key=publication_priority)
    obsolete_paths = sorted(
        set(obsolete_relative_paths), key=lambda path: path.as_posix()
    )
    # Backups live outside TemporaryDirectory so an interrupt or rollback
    # failure cannot let automatic staging cleanup delete the only prior copy.
    rollback_root = project_root / (
        ".ni_al_step5_rollback_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f_")
        + str(os.getpid())
    )
    if rollback_root.exists():
        raise PublicationError(f"Rollback path already exists: {rollback_root}")
    committed: list[tuple[Path, Path | None, Path]] = []
    removed: list[tuple[Path, Path]] = []

    for relative_path in unique_paths:
        staged_path = staging_root / relative_path
        target_path = project_root / relative_path
        if not staged_path.is_file() or staged_path.stat().st_size == 0:
            raise PartialOutputError(
                f"Missing staged file before publication: "
                f"{relative_path_text(relative_path)}"
            )
        if target_path.exists() and not target_path.is_file():
            raise PartialOutputError(
                f"Expected a replaceable file but found another object: {target_path}"
            )
        if target_path.exists() and not overwrite:
            raise OutputCollisionError(
                f"File already exists without --overwrite: {target_path}"
            )
    for relative_path in obsolete_paths:
        target_path = project_root / relative_path
        if target_path.exists() and not target_path.is_file():
            raise PartialOutputError(
                f"Expected an obsolete generated file: {target_path}"
            )

    try:
        # Under --overwrite, a newly failed phase must not leave behind an old
        # annotated file that could be mistaken for this run's result.
        for relative_path in obsolete_paths:
            target_path = project_root / relative_path
            if not target_path.exists():
                continue
            if not overwrite:
                raise OutputCollisionError(
                    f"File appeared without --overwrite: {target_path}"
                )
            backup_path = rollback_root / "obsolete" / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            # Register rollback intent before the atomic move so an interrupt
            # immediately after os.replace cannot orphan an untracked backup.
            removed.append((target_path, backup_path))
            os.replace(target_path, backup_path)

        for relative_path in unique_paths:
            staged_path = staging_root / relative_path
            target_path = project_root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path: Path | None = None
            if target_path.exists():
                # Recheck at commit time to narrow the preflight/commit race;
                # a cooperative no-overwrite run never replaces a new target.
                if not overwrite:
                    raise OutputCollisionError(
                        f"File appeared without --overwrite: {target_path}"
                    )
                backup_path = rollback_root / "files" / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
            # Register both the backup and staged path before either atomic
            # move. Rollback can then infer which move completed by existence.
            committed.append((target_path, backup_path, staged_path))
            if backup_path is not None:
                os.replace(target_path, backup_path)
            os.replace(staged_path, target_path)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target_path, backup_path, staged_path in reversed(committed):
            try:
                if backup_path is not None and backup_path.exists():
                    if target_path.is_file():
                        target_path.unlink()
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup_path, target_path)
                elif backup_path is None and not staged_path.exists() and target_path.is_file():
                    # No prior target existed. A missing staged file means its
                    # move completed, so remove only that newly published file.
                    target_path.unlink()
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{target_path}: {type(rollback_exc).__name__}"
                )
        for target_path, backup_path in reversed(removed):
            try:
                if backup_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup_path, target_path)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{target_path}: {type(rollback_exc).__name__}"
                )

        if rollback_root.exists() and not rollback_errors:
            try:
                shutil.rmtree(rollback_root)
            except OSError as recovery_exc:
                rollback_errors.append(
                    "rollback cleanup: " + type(recovery_exc).__name__
                )
        recovery_detail = (
            f"; unresolved rollback data retained at {rollback_root}"
            if rollback_errors and rollback_root.exists()
            else ""
        )
        rollback_detail = (
            "; rollback also failed for " + ", ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)) and not rollback_errors:
            raise
        raise PublicationError(
            f"Atomic publication failed ({type(exc).__name__}: {exc})"
            f"{rollback_detail}{recovery_detail}"
        ) from exc

    if rollback_root.exists():
        try:
            shutil.rmtree(rollback_root)
        except OSError as exc:
            LOGGER.warning(
                "Published outputs successfully, but prior overwrite backups "
                "remain at %s (%s: %s).",
                rollback_root,
                type(exc).__name__,
                exc,
            )


def print_validation_summary(
    project_root: Path,
    config_path: Path,
    config: EvaluationConfig,
    outcomes: Sequence[PhaseOutcome],
    options: RuntimeOptions,
) -> None:
    """Print the no-calculation validation result."""

    valid = [outcome for outcome in outcomes if outcome.phase_input is not None]
    failed = [outcome for outcome in outcomes if outcome.phase_input is None]
    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("Ni-Al MACE zero-shot validation summary")
    LOGGER.info("=" * 72)
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Configuration: %s", config_path)
    LOGGER.info("Schema version: %s", config.schema_version)
    LOGGER.info("Requested phases: %s", ", ".join(o.phase.phase_key for o in outcomes))
    LOGGER.info("Validated phases: %d", len(valid))
    LOGGER.info("Failed phases: %d", len(failed))
    LOGGER.info("Model value: %s", options.model)
    LOGGER.info("Device: %s", options.device)
    LOGGER.info("Dtype: %s", options.dtype)
    for outcome in outcomes:
        if outcome.phase_input is not None:
            metadata = outcome.phase_input.metadata
            LOGGER.info(
                "  %s: VALID (%s, %d atoms)",
                outcome.phase.phase_key,
                metadata["material_id"],
                len(outcome.phase_input.atoms),
            )
        else:
            LOGGER.error(
                "  %s: FAILED (%s: %s)",
                outcome.phase.phase_key,
                outcome.error_type,
                outcome.error_message,
            )
    LOGGER.info("The MACE model was not loaded and no calculation was run.")
    if failed:
        LOGGER.error("Step 5 validation failed.")
    else:
        LOGGER.info("Step 5 validation succeeded.")


def print_console_summary(
    outcomes: Sequence[PhaseOutcome],
    config: EvaluationConfig,
    options: RuntimeOptions,
    model_loaded: bool,
    outputs_published: bool,
) -> None:
    """Display all required phase metrics and the final run status."""

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("Ni-Al MACE-MP-0 zero-shot evaluation summary")
    LOGGER.info("=" * 72)
    for outcome in outcomes:
        LOGGER.info("Phase: %s", outcome.phase.phase_key)
        if (
            outcome.completed
            and outcome.phase_input is not None
            and outcome.result is not None
        ):
            metadata = outcome.phase_input.metadata
            result = outcome.result
            stress_text = ", ".join(
                f"{label}={float(value):.12g}"
                for label, value in zip(STRESS_LABELS, result.stress_eV_per_A3)
            )
            LOGGER.info("  Source Materials Project ID: %s", metadata["material_id"])
            LOGGER.info("  Number of atoms: %d", result.number_of_atoms)
            LOGGER.info("  Total energy: %.12g eV", result.total_energy_eV)
            LOGGER.info(
                "  Energy per atom: %.12g eV/atom", result.energy_per_atom_eV
            )
            LOGGER.info(
                "  Maximum force: %.12g eV/angstrom",
                result.maximum_force_eV_per_A,
            )
            LOGGER.info(
                "  Mean force: %.12g eV/angstrom", result.mean_force_eV_per_A
            )
            LOGGER.info(
                "  RMS force: %.12g eV/angstrom", result.rms_force_eV_per_A
            )
            LOGGER.info(
                "  Total force norm: %.12g eV/angstrom",
                result.total_force_norm_eV_per_A,
            )
            LOGGER.info(
                "  Volume per atom: %.12g angstrom^3/atom",
                result.volume_per_atom_A3,
            )
            LOGGER.info(
                "  Stress (xx, yy, zz, yz, xz, xy): %s eV/angstrom^3",
                stress_text,
            )
            LOGGER.info(
                "  Annotated structure: %s",
                relative_path_text(outcome.phase_input.output_structure_relative_path),
            )
        else:
            LOGGER.error(
                "  FAILED (%s: %s)",
                outcome.error_type or "UnknownError",
                outcome.error_message or "no additional detail",
            )

    completed = sum(outcome.completed for outcome in outcomes)
    failed = len(outcomes) - completed
    csv_relative, json_relative, report_relative = summary_relative_paths(config)
    LOGGER.info("")
    LOGGER.info("Requested phases: %d", len(outcomes))
    LOGGER.info("Completed phases: %d", completed)
    LOGGER.info("Failed phases: %d", failed)
    LOGGER.info("Model: %s %s (%s)", config.model_family, config.model_name, options.model)
    LOGGER.info("MACE model loaded successfully: %s", str(model_loaded).lower())
    LOGGER.info("Device: %s", options.device)
    LOGGER.info("Dtype: %s", options.dtype)
    if outputs_published:
        LOGGER.info("CSV path: %s", relative_path_text(csv_relative))
        LOGGER.info("JSON path: %s", relative_path_text(json_relative))
        LOGGER.info("Text report path: %s", relative_path_text(report_relative))
    else:
        LOGGER.info("CSV path: not published")
        LOGGER.info("JSON path: not published")
        LOGGER.info("Text report path: not published")
    if failed or not outputs_published:
        LOGGER.error("Overall status: FAILURE")
    else:
        LOGGER.info("Overall status: SUCCESS")


def mark_model_failure(
    outcomes: Sequence[PhaseOutcome],
    exc: BaseException,
    verbose: bool,
) -> None:
    """Apply one global model-loading failure to every otherwise valid phase."""

    for outcome in outcomes:
        if outcome.phase_input is not None:
            log_phase_failure(outcome, exc, verbose)


def run_real_evaluation(
    project_root: Path,
    config: EvaluationConfig,
    requested_phases: Sequence[PhaseDefinition],
    outcomes: Sequence[PhaseOutcome],
    options: RuntimeOptions,
    dependencies: ScientificDependencies,
    overwrite: bool,
    verbose: bool,
) -> tuple[bool, bool]:
    """Run model evaluation and publication while holding the global run lock."""

    lock_path = acquire_run_lock(project_root)
    try:
        # Collision inspection occurs under the same lock held through commit,
        # closing the cooperative concurrent-run race for fixed summary paths.
        validate_output_paths(
            project_root,
            config,
            requested_phases,
            overwrite,
            validation_only=False,
            dependencies=dependencies,
        )
        model_loaded = False
        outputs_published = False
        with tempfile.TemporaryDirectory(
            prefix=".ni_al_step5_", dir=project_root
        ) as temporary_directory:
            staging_root = Path(temporary_directory)
            valid_outcomes = [
                outcome for outcome in outcomes if outcome.phase_input is not None
            ]
            if valid_outcomes:
                try:
                    calculator = load_calculator(config, options, dependencies)
                    model_loaded = True
                    evaluate_phases(
                        staging_root,
                        outcomes,
                        calculator,
                        config,
                        options,
                        dependencies,
                        verbose,
                    )
                except (ModelLoadingError, ModelDownloadError) as exc:
                    LOGGER.error("%s", exc)
                    if verbose:
                        LOGGER.debug(traceback.format_exc().rstrip())
                    mark_model_failure(outcomes, exc, verbose)

            _, summary_paths = stage_summary_outputs(
                staging_root,
                project_root,
                outcomes,
                config,
                options,
            )
            successful_structure_paths = [
                outcome.phase_input.output_structure_relative_path
                for outcome in outcomes
                if outcome.completed and outcome.phase_input is not None
            ]
            failed_structure_paths = [
                phase_paths(config, outcome.phase)[2]
                for outcome in outcomes
                if not outcome.completed
            ]
            publish_staged_outputs(
                project_root=project_root,
                staging_root=staging_root,
                relative_paths=[*successful_structure_paths, *summary_paths],
                obsolete_relative_paths=(failed_structure_paths if overwrite else []),
                overwrite=overwrite,
            )
            outputs_published = True
        return model_loaded, outputs_published
    finally:
        release_run_lock(lock_path)


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate inputs and optionally run the complete Step 5 workflow."""

    args = parse_arguments(arguments)
    configure_logging(args.verbose)
    project_root = locate_project_root()

    try:
        validate_runtime(project_root)
        dependencies = import_scientific_dependencies()
        config_path = resolve_config_path(project_root, args.config)
        config = load_configuration(project_root, config_path, dependencies)
        options = resolve_runtime_options(args, config)
        requested_phases = choose_requested_phases(config.phases, args.phase)
        if options.dtype == "float32":
            LOGGER.warning(
                "WARNING: float32 was selected. Step 5 defaults to float64 for "
                "this static scientific baseline."
            )
        outcomes = discover_and_validate_inputs(
            config, requested_phases, dependencies, args.verbose
        )
        if args.validate_only:
            validate_output_paths(
                project_root,
                config,
                requested_phases,
                args.overwrite,
                validation_only=True,
                dependencies=dependencies,
            )
            print_validation_summary(
                project_root, config_path, config, outcomes, options
            )
            return 0 if all(outcome.phase_input is not None for outcome in outcomes) else 1

        model_loaded, outputs_published = run_real_evaluation(
            project_root=project_root,
            config=config,
            requested_phases=requested_phases,
            outcomes=outcomes,
            options=options,
            dependencies=dependencies,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        print_console_summary(
            outcomes, config, options, model_loaded, outputs_published
        )
        return 0 if all(outcome.completed for outcome in outcomes) else 1
    except Step5Error as exc:
        LOGGER.error("Step 5 failed (%s): %s", type(exc).__name__, exc)
        if args.verbose:
            LOGGER.debug(traceback.format_exc().rstrip())
        return 1
    except Exception as exc:
        LOGGER.error(
            "Step 5 failed unexpectedly (%s): %s", type(exc).__name__, exc
        )
        if args.verbose:
            LOGGER.debug(traceback.format_exc().rstrip())
        return 1


if __name__ == "__main__":
    # Propagate success or failure to CMD, PowerShell, and automation tools.
    raise SystemExit(main())

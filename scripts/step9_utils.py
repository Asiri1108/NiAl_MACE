"""Shared utilities for Ni-Al Step 9 classical-potential selection.

Step 9 selects, retrieves, validates, and documents candidate classical
Ni-Al EAM potentials from authoritative sources and designs the future
Step 10 LAMMPS benchmark.  This module provides:

``load_step9_config``
    Strict validation of ``configs/ni_al_classical_potentials.json``.
``parse_setfl``
    A complete DYNAMO multielement setfl (``eam/alloy``) validator.
``validate_candidate_bundle``
    Validation of one retrieved raw+processed candidate bundle.
``inspect_lammps_availability``
    Safe local LAMMPS executable inspection without running a simulation.
``validate_step8_success``
    Read-only confirmation that Step 8 finished with SUCCESS.
``build_evaluation_matrix``
    The documented qualitative candidate evaluation matrix.

No function here executes LAMMPS, loads MACE, performs DFT, calculates
any new scientific energy, or modifies Step 6, Step 7, or Step 8 data.
Potential files are treated strictly as data, never as executable code.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from step6_utils import FileSnapshot
from step7_utils import (
    Step7Error,
    file_sha256,
    read_strict_json,
    relative_path,
    snapshot_file,
    verify_snapshots,
)


LOGGER = logging.getLogger("step9_utils")

SCHEMA_VERSION = "1.0"
CANDIDATE_ORDER: tuple[str, ...] = (
    "pun_mishin_2009",
    "mishin_2004_ipr2",
    "mishin_2002",
)
PHASE_ORDER: tuple[str, ...] = ("Al3Ni", "Al3Ni2", "AlNi", "Al3Ni5", "AlNi3")
ELEMENT_ORDER: tuple[str, ...] = ("Al", "Ni")
ROLE_RANKS: Mapping[str, tuple[int, str]] = {
    "primary": (1, "Primary"),
    "secondary": (2, "Secondary"),
    "historical_secondary": (3, "Historical secondary"),
}
SUPERSEDED_2004_IPR1_FILENAME = "NiAl.eam.alloy"

# Exact SI definitions: 1 eV = 1.602176634e-19 J (exact CODATA 2019),
# 1 angstrom^3 = 1e-30 m^3, 1 bar = 1e5 Pa.
EV_JOULE = 1.602176634e-19
ANGSTROM3_M3 = 1.0e-30
BAR_PASCAL = 1.0e5


class Step9Error(RuntimeError):
    """Base class for controlled Step 9 failures."""


class Step9ConfigurationError(Step9Error):
    """Raised when the Step 9 configuration or command scope is unsafe."""


class Step9SourceError(Step9Error):
    """Raised when a retrieval source violates the authoritative policy."""


class Step9RetrievalError(Step9Error):
    """Raised when controlled HTTPS retrieval fails."""


class Step9PotentialFormatError(Step9Error):
    """Raised when a potential file fails setfl structural validation."""


class Step9InputError(Step9Error):
    """Raised when a protected scientific input is invalid."""


class Step9CollisionError(Step9Error):
    """Raised when output collision handling refuses publication."""


class Step9ResumeError(Step9Error):
    """Raised when an existing Step 9 bundle is not safe to reuse."""


def ev_per_A3_to_bar(value_eV_per_A3: float) -> float:
    """Convert a stress value from eV/angstrom^3 to bar using exact factors."""

    return value_eV_per_A3 * EV_JOULE / ANGSTROM3_M3 / BAR_PASCAL


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Step9ConfigurationError(f"{label} must be a nonempty string.")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> Any:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise Step9ConfigurationError(
            f"{label} must be exactly {expected!r}; received {value!r}."
        )
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Step9ConfigurationError(f"{label} must be a JSON object.")
    return value


def _require_https_url(value: Any, label: str, allowed_hosts: Sequence[str]) -> str:
    url = _require_string(value, label)
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise Step9SourceError(f"{label} must use HTTPS: {url}")
    if parsed.hostname not in allowed_hosts:
        raise Step9SourceError(
            f"{label} host {parsed.hostname!r} is not in the authoritative "
            f"allow-list {list(allowed_hosts)}."
        )
    return url


def _relative_repo_path(value: Any, label: str, project_root: Path) -> Path:
    text = _require_string(value, label)
    candidate = (project_root / Path(text)).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step9ConfigurationError(
            f"{label} must remain inside the repository: {text}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class SourcePolicy:
    """Validated authoritative retrieval policy."""

    allowed_repositories: tuple[str, ...]
    allowed_download_hosts: tuple[str, ...]
    allow_unverified_sources: bool
    https_required: bool
    connection_timeout_seconds: int
    maximum_file_size_bytes: int
    maximum_attempts_per_file: int
    user_agent: str


@dataclass(frozen=True)
class CandidateSpec:
    """Validated configuration for one candidate potential."""

    key: str
    role: str
    potential_name: str
    formalism: str
    authors: str
    publication_year: int
    citation: str
    doi: str
    repository_identity: str
    implementation_identity: str
    entry_page_url: str
    potential_file_url: str
    release_notes_url: str
    release_notes_filename: str
    expected_filename: str
    pair_style: str
    expected_elements: tuple[str, ...]
    openkim_family_id: str
    openkim_extended_id: str
    openkim_model_driver_family: str
    enabled_for_step10: bool
    fitting_scope: str
    implementation_notes: str
    known_warnings: tuple[str, ...]
    reject_superseded_ipr1: bool
    superseded_ipr1: Mapping[str, Any] | None


@dataclass(frozen=True)
class LammpsDesign:
    """Validated deterministic LAMMPS design conventions."""

    units: str
    atom_style: str
    boundary: tuple[str, str, str]
    pair_style: str
    atom_type_mapping: Mapping[str, str]
    execute_in_step9: bool
    executable_candidates: tuple[str, ...]


@dataclass(frozen=True)
class Step9Config:
    """Fully validated Step 9 configuration."""

    project_root: Path
    config_path: Path
    config_snapshot: FileSnapshot
    fingerprint: str
    raw: Mapping[str, Any]
    source_policy: SourcePolicy
    candidate_order: tuple[str, ...]
    candidates: Mapping[str, CandidateSpec]
    minimum_file_size_bytes: int
    expected_atomic_numbers: Mapping[str, int]
    expected_masses_amu: Mapping[str, float]
    mass_tolerance_amu: float
    lammps_design: LammpsDesign
    step10_design: Mapping[str, Any]
    raw_root: Path
    selected_root: Path
    result_root: Path


def load_step9_config(config_path: Path | str) -> Step9Config:
    """Load and strictly validate the Step 9 configuration."""

    from step7_utils import locate_project_root

    project_root = locate_project_root()
    candidate_path = Path(config_path)
    if not candidate_path.is_absolute():
        candidate_path = project_root / candidate_path
    resolved = candidate_path.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise Step9ConfigurationError(
            f"Configuration must remain inside the repository: {resolved}"
        ) from exc
    try:
        raw = read_strict_json(resolved, "Step 9 potential configuration")
    except Step7Error as exc:
        raise Step9ConfigurationError(str(exc)) from exc

    _require_exact(raw.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_string(raw.get("description"), "description")

    policy_raw = _require_mapping(raw.get("source_policy"), "source_policy")
    _require_exact(
        policy_raw.get("allow_unverified_sources"),
        False,
        "source_policy.allow_unverified_sources",
    )
    _require_exact(
        policy_raw.get("https_required"), True, "source_policy.https_required"
    )
    hosts = policy_raw.get("allowed_download_hosts")
    if hosts != ["www.ctcms.nist.gov"]:
        raise Step9ConfigurationError(
            "source_policy.allowed_download_hosts must be exactly "
            "['www.ctcms.nist.gov'] (NIST Interatomic Potentials Repository)."
        )
    timeout = policy_raw.get("connection_timeout_seconds")
    maximum_size = policy_raw.get("maximum_file_size_bytes")
    attempts = policy_raw.get("maximum_attempts_per_file")
    for name, value, low, high in (
        ("connection_timeout_seconds", timeout, 10, 300),
        ("maximum_file_size_bytes", maximum_size, 100000, 100000000),
        ("maximum_attempts_per_file", attempts, 1, 3),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not low <= value <= high
        ):
            raise Step9ConfigurationError(
                f"source_policy.{name} must be an integer in [{low}, {high}]."
            )
    policy = SourcePolicy(
        allowed_repositories=tuple(
            str(item) for item in policy_raw.get("allowed_repositories", ())
        ),
        allowed_download_hosts=("www.ctcms.nist.gov",),
        allow_unverified_sources=False,
        https_required=True,
        connection_timeout_seconds=int(timeout),
        maximum_file_size_bytes=int(maximum_size),
        maximum_attempts_per_file=int(attempts),
        user_agent=_require_string(
            policy_raw.get("user_agent"), "source_policy.user_agent"
        ),
    )
    if "secret" in policy.user_agent.lower().replace("no-secret", ""):
        raise Step9ConfigurationError("User agent must not embed secrets.")

    if tuple(raw.get("candidate_order", ())) != CANDIDATE_ORDER:
        raise Step9ConfigurationError(
            f"candidate_order must be exactly {list(CANDIDATE_ORDER)}."
        )
    candidates_raw = _require_mapping(raw.get("candidates"), "candidates")
    if tuple(candidates_raw.keys()) != CANDIDATE_ORDER:
        raise Step9ConfigurationError(
            "candidates must define exactly the configured candidate order."
        )
    expected_identity = {
        "pun_mishin_2009": (
            "primary",
            "2009--Purja-Pun-G-P-Mishin-Y--Ni-Al",
            "2009--Purja-Pun-G-P--Ni-Al--LAMMPS--ipr1",
            "Mishin-Ni-Al-2009.eam.alloy",
            "MO_751354403791",
        ),
        "mishin_2004_ipr2": (
            "secondary",
            "2004--Mishin-Y--Ni-Al",
            "2004--Mishin-Y--Ni-Al--LAMMPS--ipr2",
            "NiAl_Mishin_2004.eam.alloy",
            "MO_101214310689",
        ),
        "mishin_2002": (
            "historical_secondary",
            "2002--Mishin-Y-Mehl-M-J-Papaconstantopoulos-D-A--Ni-Al",
            "2002--Mishin-Y--Ni-Al--LAMMPS--ipr1",
            "NiAl02.eam.alloy",
            "MO_109933561507",
        ),
    }
    candidates: dict[str, CandidateSpec] = {}
    for key in CANDIDATE_ORDER:
        spec_raw = _require_mapping(candidates_raw.get(key), f"candidates.{key}")
        role, repository, implementation, filename, family = expected_identity[key]
        _require_exact(spec_raw.get("role"), role, f"candidates.{key}.role")
        _require_exact(
            spec_raw.get("repository_identity"),
            repository,
            f"candidates.{key}.repository_identity",
        )
        _require_exact(
            spec_raw.get("implementation_identity"),
            implementation,
            f"candidates.{key}.implementation_identity",
        )
        _require_exact(
            spec_raw.get("expected_filename"),
            filename,
            f"candidates.{key}.expected_filename",
        )
        _require_exact(
            spec_raw.get("pair_style"), "eam/alloy", f"candidates.{key}.pair_style"
        )
        _require_exact(
            spec_raw.get("expected_elements"),
            ["Al", "Ni"],
            f"candidates.{key}.expected_elements",
        )
        _require_exact(
            spec_raw.get("openkim_family_id"),
            family,
            f"candidates.{key}.openkim_family_id",
        )
        potential_url = _require_https_url(
            spec_raw.get("potential_file_url"),
            f"candidates.{key}.potential_file_url",
            policy.allowed_download_hosts,
        )
        if not potential_url.endswith("/" + filename.replace(" ", "%20")):
            raise Step9SourceError(
                f"candidates.{key} potential URL does not end with the "
                f"official filename {filename!r}."
            )
        notes_url = _require_https_url(
            spec_raw.get("release_notes_url"),
            f"candidates.{key}.release_notes_url",
            policy.allowed_download_hosts,
        )
        _require_https_url(
            spec_raw.get("entry_page_url"),
            f"candidates.{key}.entry_page_url",
            policy.allowed_download_hosts,
        )
        year = spec_raw.get("publication_year")
        if isinstance(year, bool) or not isinstance(year, int) or not (
            2000 <= year <= 2010
        ):
            raise Step9ConfigurationError(
                f"candidates.{key}.publication_year is implausible: {year!r}."
            )
        reject_ipr1 = bool(spec_raw.get("reject_superseded_ipr1", False))
        superseded = spec_raw.get("superseded_ipr1")
        if key == "mishin_2004_ipr2":
            if not reject_ipr1 or not isinstance(superseded, Mapping):
                raise Step9ConfigurationError(
                    "mishin_2004_ipr2 must reject the superseded ipr1 file "
                    "and document it."
                )
            _require_exact(
                superseded.get("filename"),
                SUPERSEDED_2004_IPR1_FILENAME,
                "candidates.mishin_2004_ipr2.superseded_ipr1.filename",
            )
        candidates[key] = CandidateSpec(
            key=key,
            role=role,
            potential_name=_require_string(
                spec_raw.get("potential_name"), f"candidates.{key}.potential_name"
            ),
            formalism=_require_string(
                spec_raw.get("formalism"), f"candidates.{key}.formalism"
            ),
            authors=_require_string(
                spec_raw.get("authors"), f"candidates.{key}.authors"
            ),
            publication_year=year,
            citation=_require_string(
                spec_raw.get("citation"), f"candidates.{key}.citation"
            ),
            doi=_require_string(spec_raw.get("doi"), f"candidates.{key}.doi"),
            repository_identity=repository,
            implementation_identity=implementation,
            entry_page_url=str(spec_raw.get("entry_page_url")),
            potential_file_url=potential_url,
            release_notes_url=notes_url,
            release_notes_filename=_require_string(
                spec_raw.get("release_notes_filename"),
                f"candidates.{key}.release_notes_filename",
            ),
            expected_filename=filename,
            pair_style="eam/alloy",
            expected_elements=("Al", "Ni"),
            openkim_family_id=family,
            openkim_extended_id=_require_string(
                spec_raw.get("openkim_extended_id"),
                f"candidates.{key}.openkim_extended_id",
            ),
            openkim_model_driver_family=_require_string(
                spec_raw.get("openkim_model_driver_family"),
                f"candidates.{key}.openkim_model_driver_family",
            ),
            enabled_for_step10=_require_exact(
                spec_raw.get("enabled_for_step10"),
                True,
                f"candidates.{key}.enabled_for_step10",
            ),
            fitting_scope=_require_string(
                spec_raw.get("fitting_scope"), f"candidates.{key}.fitting_scope"
            ),
            implementation_notes=_require_string(
                spec_raw.get("implementation_notes"),
                f"candidates.{key}.implementation_notes",
            ),
            known_warnings=tuple(
                str(item) for item in spec_raw.get("known_warnings", ())
            ),
            reject_superseded_ipr1=reject_ipr1,
            superseded_ipr1=(
                dict(superseded) if isinstance(superseded, Mapping) else None
            ),
        )

    validation_raw = _require_mapping(raw.get("validation"), "validation")
    minimum_size = validation_raw.get("minimum_file_size_bytes")
    if (
        isinstance(minimum_size, bool)
        or not isinstance(minimum_size, int)
        or minimum_size <= 0
    ):
        raise Step9ConfigurationError(
            "validation.minimum_file_size_bytes must be a positive integer."
        )
    _require_exact(
        validation_raw.get("expected_atomic_numbers"),
        {"Al": 13, "Ni": 28},
        "validation.expected_atomic_numbers",
    )
    masses = _require_mapping(
        validation_raw.get("expected_masses_amu"),
        "validation.expected_masses_amu",
    )
    tolerance = validation_raw.get("mass_tolerance_amu")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 < float(tolerance) <= 2.0
    ):
        raise Step9ConfigurationError(
            "validation.mass_tolerance_amu must be in (0, 2]."
        )

    design_raw = _require_mapping(raw.get("lammps_design"), "lammps_design")
    _require_exact(design_raw.get("units"), "metal", "lammps_design.units")
    _require_exact(
        design_raw.get("atom_style"), "atomic", "lammps_design.atom_style"
    )
    _require_exact(
        design_raw.get("boundary"), ["p", "p", "p"], "lammps_design.boundary"
    )
    _require_exact(
        design_raw.get("pair_style"), "eam/alloy", "lammps_design.pair_style"
    )
    _require_exact(
        design_raw.get("atom_type_mapping"),
        {"1": "Al", "2": "Ni"},
        "lammps_design.atom_type_mapping",
    )
    _require_exact(
        design_raw.get("execute_in_step9"),
        False,
        "lammps_design.execute_in_step9",
    )
    lammps_design = LammpsDesign(
        units="metal",
        atom_style="atomic",
        boundary=("p", "p", "p"),
        pair_style="eam/alloy",
        atom_type_mapping={"1": "Al", "2": "Ni"},
        execute_in_step9=False,
        executable_candidates=tuple(
            str(item) for item in design_raw.get("executable_candidates", ())
        ),
    )
    if not lammps_design.executable_candidates:
        raise Step9ConfigurationError(
            "lammps_design.executable_candidates must not be empty."
        )

    step10_raw = _require_mapping(raw.get("step10_design"), "step10_design")
    _require_exact(
        step10_raw.get("force_threshold_eV_per_A"),
        0.01,
        "step10_design.force_threshold_eV_per_A",
    )
    _require_exact(
        step10_raw.get("stress_threshold_eV_per_A3"),
        0.0006241509,
        "step10_design.stress_threshold_eV_per_A3",
    )
    _require_exact(
        step10_raw.get("phase_order"),
        list(PHASE_ORDER),
        "step10_design.phase_order",
    )
    _require_exact(
        step10_raw.get("elemental_references"),
        list(ELEMENT_ORDER),
        "step10_design.elemental_references",
    )

    output_raw = _require_mapping(raw.get("output"), "output")
    raw_root = _relative_repo_path(
        _require_exact(
            output_raw.get("raw_root"),
            "data/raw/interatomic_potentials/ni_al",
            "output.raw_root",
        ),
        "output.raw_root",
        project_root,
    )
    selected_root = _relative_repo_path(
        _require_exact(
            output_raw.get("selected_root"),
            "data/processed/interatomic_potentials/ni_al",
            "output.selected_root",
        ),
        "output.selected_root",
        project_root,
    )
    result_root = _relative_repo_path(
        _require_exact(
            output_raw.get("result_root"),
            "results/lammps_potential_selection",
            "output.result_root",
        ),
        "output.result_root",
        project_root,
    )

    snapshot = snapshot_file(resolved, "Step 9 configuration")
    return Step9Config(
        project_root=project_root,
        config_path=resolved,
        config_snapshot=snapshot,
        fingerprint=snapshot.sha256,
        raw=raw,
        source_policy=policy,
        candidate_order=CANDIDATE_ORDER,
        candidates=candidates,
        minimum_file_size_bytes=int(minimum_size),
        expected_atomic_numbers={"Al": 13, "Ni": 28},
        expected_masses_amu={
            element: float(masses[element]) for element in ELEMENT_ORDER
        },
        mass_tolerance_amu=float(tolerance),
        lammps_design=lammps_design,
        step10_design=dict(step10_raw),
        raw_root=raw_root,
        selected_root=selected_root,
        result_root=result_root,
    )


def validate_candidate_keys(candidates: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a requested candidate selection deterministically."""

    if candidates is None:
        return CANDIDATE_ORDER
    selected = tuple(candidates)
    if not selected:
        raise Step9ConfigurationError("At least one candidate must be selected.")
    if len(set(selected)) != len(selected):
        raise Step9ConfigurationError("Requested candidates contain duplicates.")
    unknown = [key for key in selected if key not in CANDIDATE_ORDER]
    if unknown:
        raise Step9ConfigurationError(
            "Unknown candidate(s): " + ", ".join(unknown)
        )
    return tuple(key for key in CANDIDATE_ORDER if key in selected)


# ---------------------------------------------------------------------------
# Bundle paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateBundlePaths:
    """Canonical raw and processed targets for one candidate."""

    raw_directory: Path
    raw_potential: Path
    raw_release_notes: Path
    source_metadata: Path
    retrieval_manifest: Path
    processed_directory: Path
    processed_potential: Path

    def required_paths(self) -> tuple[Path, ...]:
        """Return the mandatory bundle files (release notes are optional)."""

        return (
            self.raw_potential,
            self.source_metadata,
            self.retrieval_manifest,
            self.processed_potential,
        )

    def all_paths(self) -> tuple[Path, ...]:
        """Return every possible bundle target."""

        return (
            self.raw_potential,
            self.raw_release_notes,
            self.source_metadata,
            self.retrieval_manifest,
            self.processed_potential,
        )


def candidate_bundle_paths(
    config: Step9Config, key: str
) -> CandidateBundlePaths:
    """Resolve the canonical bundle paths for one candidate."""

    spec = config.candidates[key]
    raw_directory = config.raw_root / key
    processed_directory = config.selected_root / key
    return CandidateBundlePaths(
        raw_directory=raw_directory,
        raw_potential=raw_directory / spec.expected_filename,
        raw_release_notes=raw_directory / spec.release_notes_filename,
        source_metadata=raw_directory / "source_metadata.json",
        retrieval_manifest=raw_directory / "retrieval_manifest.json",
        processed_directory=processed_directory,
        processed_potential=processed_directory / spec.expected_filename,
    )


def step9_result_paths(config: Step9Config) -> dict[str, Path]:
    """Resolve the canonical Step 9 result targets."""

    root = config.result_root
    return {
        "candidates_csv": (
            root / "tables" / "ni_al_classical_potential_candidates.csv"
        ),
        "candidates_json": (
            root / "tables" / "ni_al_classical_potential_candidates.json"
        ),
        "selection_report": (
            root / "reports" / "ni_al_classical_potential_selection_report.txt"
        ),
        "final_report": root / "reports" / "ni_al_step9_final_report.txt",
        "file_manifest": (
            root / "manifests" / "ni_al_potential_file_manifest.json"
        ),
        "plan_json": (
            root / "plans" / "ni_al_step10_lammps_benchmark_plan.json"
        ),
        "plan_txt": root / "plans" / "ni_al_step10_lammps_benchmark_plan.txt",
    }


def step9_directories(config: Step9Config) -> tuple[Path, ...]:
    """Return every Step 9 result directory execution may populate."""

    root = config.result_root
    return (
        root,
        root / "tables",
        root / "reports",
        root / "manifests",
        root / "plans",
    )


# ---------------------------------------------------------------------------
# Setfl parsing and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetflElementRecord:
    """Per-element header data from a setfl file."""

    name: str
    atomic_number: int
    mass_amu: float
    lattice_constant_A: float
    lattice_type: str


@dataclass(frozen=True)
class SetflData:
    """Complete validated structure of a DYNAMO multielement setfl file."""

    path: Path
    comment_lines: tuple[str, ...]
    element_count: int
    elements: tuple[str, ...]
    nrho: int
    drho: float
    nr: int
    dr: float
    cutoff_A: float
    element_records: tuple[SetflElementRecord, ...]
    total_tabulated_values: int
    file_size_bytes: int
    sha256: str


_FORBIDDEN_FLOAT_TOKENS = {"nan", "-nan", "+nan", "inf", "-inf", "+inf",
                           "infinity", "-infinity", "+infinity"}


def _parse_finite_float(token: str, label: str) -> float:
    """Parse one numeric token, rejecting NaN and infinity spellings."""

    if token.lower() in _FORBIDDEN_FLOAT_TOKENS:
        raise Step9PotentialFormatError(
            f"{label} contains a forbidden nonfinite token: {token!r}"
        )
    try:
        value = float(token)
    except ValueError as exc:
        raise Step9PotentialFormatError(
            f"{label} contains an unparseable numeric token: {token!r}"
        ) from exc
    if not math.isfinite(value):
        raise Step9PotentialFormatError(
            f"{label} contains a nonfinite value: {token!r}"
        )
    return value


def parse_setfl(path: Path, config: Step9Config) -> SetflData:
    """Parse and completely validate one DYNAMO multielement setfl file.

    Values are validated in place; the file itself is never edited,
    malformed trailing content is never silently accepted, and missing
    values are never inferred.
    """

    label = path.name
    if not path.is_file():
        raise Step9PotentialFormatError(f"{label}: file does not exist.")
    size = path.stat().st_size
    if size == 0:
        raise Step9PotentialFormatError(f"{label}: file is empty.")
    if size < config.minimum_file_size_bytes:
        raise Step9PotentialFormatError(
            f"{label}: file size {size} bytes is below the plausible minimum "
            f"{config.minimum_file_size_bytes}."
        )
    if size > config.source_policy.maximum_file_size_bytes:
        raise Step9PotentialFormatError(
            f"{label}: file size {size} bytes exceeds the plausible maximum."
        )
    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes:
        raise Step9PotentialFormatError(f"{label}: file contains NUL bytes.")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Step9PotentialFormatError(
            f"{label}: file is not readable as UTF-8/ASCII text: {exc}"
        ) from exc

    lines = text.splitlines()
    if len(lines) < 6:
        raise Step9PotentialFormatError(
            f"{label}: too few lines for a setfl file."
        )
    comments = tuple(lines[0:3])
    element_header = lines[3].split()
    if len(element_header) < 2:
        raise Step9PotentialFormatError(
            f"{label}: element-count header line is malformed."
        )
    try:
        element_count = int(element_header[0])
    except ValueError as exc:
        raise Step9PotentialFormatError(
            f"{label}: element count is not an integer: {element_header[0]!r}"
        ) from exc
    if element_count <= 0:
        raise Step9PotentialFormatError(
            f"{label}: element count must be positive."
        )
    elements = tuple(element_header[1:])
    if len(elements) != element_count:
        raise Step9PotentialFormatError(
            f"{label}: header declares {element_count} elements but lists "
            f"{len(elements)} names."
        )
    if len(set(elements)) != len(elements):
        raise Step9PotentialFormatError(
            f"{label}: element names are not unique: {elements}."
        )
    missing = [name for name in ELEMENT_ORDER if name not in elements]
    if missing:
        raise Step9PotentialFormatError(
            f"{label}: required element(s) missing from the header: {missing}."
        )

    grid_tokens = lines[4].split()
    if len(grid_tokens) != 5:
        raise Step9PotentialFormatError(
            f"{label}: grid header must have exactly 5 fields "
            "(Nrho drho Nr dr cutoff)."
        )
    try:
        nrho = int(grid_tokens[0])
        nr = int(grid_tokens[2])
    except ValueError as exc:
        raise Step9PotentialFormatError(
            f"{label}: Nrho/Nr are not integers."
        ) from exc
    drho = _parse_finite_float(grid_tokens[1], f"{label} drho")
    dr = _parse_finite_float(grid_tokens[3], f"{label} dr")
    cutoff = _parse_finite_float(grid_tokens[4], f"{label} cutoff")
    if nrho <= 0 or nr <= 0:
        raise Step9PotentialFormatError(
            f"{label}: Nrho and Nr must be positive integers."
        )
    if drho <= 0.0 or dr <= 0.0 or cutoff <= 0.0:
        raise Step9PotentialFormatError(
            f"{label}: drho, dr, and cutoff must be positive."
        )

    # Walk the remaining content as one token stream: for each element, a
    # 4-field header (Z, mass, lattice constant, lattice type) followed by
    # Nrho embedding values and Nr density values; then Nr pair values for
    # every i>=j pair. The stream must be consumed exactly.
    tokens: list[str] = []
    for line in lines[5:]:
        tokens.extend(line.split())
    position = 0

    def take(count: int, what: str) -> list[str]:
        nonlocal position
        if position + count > len(tokens):
            raise Step9PotentialFormatError(
                f"{label}: file is truncated while reading {what}; expected "
                f"{count} more token(s) but only {len(tokens) - position} "
                "remain."
            )
        chunk = tokens[position:position + count]
        position += count
        return chunk

    element_records: list[SetflElementRecord] = []
    for name in elements:
        header = take(4, f"{name} element header")
        atomic_number_value = _parse_finite_float(
            header[0], f"{label} {name} atomic number"
        )
        atomic_number = int(round(atomic_number_value))
        if not math.isclose(atomic_number_value, atomic_number, abs_tol=1e-9):
            raise Step9PotentialFormatError(
                f"{label}: {name} atomic number is not an integer: {header[0]!r}"
            )
        mass = _parse_finite_float(header[1], f"{label} {name} mass")
        lattice_constant = _parse_finite_float(
            header[2], f"{label} {name} lattice constant"
        )
        lattice_type = header[3]
        expected_z = config.expected_atomic_numbers.get(name)
        if expected_z is not None and atomic_number != expected_z:
            raise Step9PotentialFormatError(
                f"{label}: {name} atomic number {atomic_number} does not "
                f"match the expected {expected_z}."
            )
        expected_mass = config.expected_masses_amu.get(name)
        if expected_mass is not None and abs(mass - expected_mass) > (
            config.mass_tolerance_amu
        ):
            raise Step9PotentialFormatError(
                f"{label}: {name} mass {mass} amu is outside the plausible "
                f"range around {expected_mass} amu."
            )
        if mass <= 0.0 or lattice_constant <= 0.0:
            raise Step9PotentialFormatError(
                f"{label}: {name} mass/lattice constant must be positive."
            )
        for index, token in enumerate(take(nrho, f"{name} embedding function")):
            _parse_finite_float(token, f"{label} {name} F(rho)[{index}]")
        for index, token in enumerate(take(nr, f"{name} density function")):
            _parse_finite_float(token, f"{label} {name} rho(r)[{index}]")
        element_records.append(
            SetflElementRecord(
                name=name,
                atomic_number=atomic_number,
                mass_amu=mass,
                lattice_constant_A=lattice_constant,
                lattice_type=lattice_type,
            )
        )

    pair_count = element_count * (element_count + 1) // 2
    for pair_index in range(pair_count):
        for index, token in enumerate(
            take(nr, f"pair function {pair_index + 1}/{pair_count}")
        ):
            _parse_finite_float(
                token, f"{label} pair[{pair_index}][{index}]"
            )
    if position != len(tokens):
        raise Step9PotentialFormatError(
            f"{label}: {len(tokens) - position} unexpected trailing token(s) "
            "after the final tabulated array; malformed trailing content is "
            "not accepted."
        )
    total_values = element_count * (nrho + nr) + pair_count * nr
    return SetflData(
        path=path,
        comment_lines=comments,
        element_count=element_count,
        elements=elements,
        nrho=nrho,
        drho=drho,
        nr=nr,
        dr=dr,
        cutoff_A=cutoff,
        element_records=tuple(element_records),
        total_tabulated_values=total_values,
        file_size_bytes=size,
        sha256=file_sha256(path),
    )


# ---------------------------------------------------------------------------
# Candidate bundle validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedCandidate:
    """One completely validated retrieved candidate bundle."""

    spec: CandidateSpec
    setfl: SetflData
    raw_sha256: str
    processed_sha256: str
    raw_size_bytes: int
    release_notes_available: bool
    release_notes_sha256: str | None
    source_metadata: Mapping[str, Any]
    retrieval_manifest: Mapping[str, Any]
    snapshots: tuple[FileSnapshot, ...]


def validate_candidate_bundle(
    config: Step9Config, key: str
) -> ValidatedCandidate:
    """Validate one published raw+processed candidate bundle completely."""

    spec = config.candidates[key]
    paths = candidate_bundle_paths(config, key)
    missing = [path for path in paths.required_paths() if not path.is_file()]
    if missing:
        raise Step9InputError(
            f"{key} bundle is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
            + ". Run fetch_ni_al_classical_potentials.py --fetch first."
        )
    snapshots = [
        snapshot_file(path, f"{key} {path.name}")
        for path in paths.required_paths()
    ]
    release_notes_available = paths.raw_release_notes.is_file()
    release_notes_sha: str | None = None
    if release_notes_available:
        snapshots.append(
            snapshot_file(paths.raw_release_notes, f"{key} release notes")
        )
        release_notes_sha = file_sha256(paths.raw_release_notes)

    # The 2004 superseded ipr1 file must never be present in any bundle.
    for directory in (paths.raw_directory, paths.processed_directory):
        superseded = directory / SUPERSEDED_2004_IPR1_FILENAME
        if superseded.exists():
            raise Step9InputError(
                f"The superseded 2004 ipr1 file {superseded} is present; it "
                "must not be accepted. Remove it and re-run the fetch."
            )

    setfl = parse_setfl(paths.raw_potential, config)
    raw_sha = setfl.sha256
    processed_sha = file_sha256(paths.processed_potential)
    if raw_sha != processed_sha:
        raise Step9InputError(
            f"{key}: processed copy is not byte-identical to the validated "
            f"raw file ({raw_sha} versus {processed_sha})."
        )
    if paths.raw_potential.name != spec.expected_filename:
        raise Step9InputError(
            f"{key}: filename {paths.raw_potential.name!r} does not match the "
            f"official implementation filename {spec.expected_filename!r}."
        )

    try:
        source_metadata = read_strict_json(
            paths.source_metadata, f"{key} source metadata"
        )
        retrieval_manifest = read_strict_json(
            paths.retrieval_manifest, f"{key} retrieval manifest"
        )
    except Step7Error as exc:
        raise Step9InputError(str(exc)) from exc
    for field_name, expected in (
        ("candidate_key", key),
        ("repository_identity", spec.repository_identity),
        ("implementation_identity", spec.implementation_identity),
        ("expected_filename", spec.expected_filename),
        ("openkim_family_id", spec.openkim_family_id),
    ):
        if source_metadata.get(field_name) != expected:
            raise Step9InputError(
                f"{key} source metadata field {field_name!r} does not match "
                "the authoritative candidate configuration."
            )
    recorded_sha = retrieval_manifest.get("potential_file_sha256")
    if recorded_sha != raw_sha:
        raise Step9InputError(
            f"{key}: retrieval manifest SHA-256 does not match the file."
        )
    source_url = retrieval_manifest.get("potential_file_url")
    if source_url != spec.potential_file_url:
        raise Step9InputError(
            f"{key}: retrieval manifest URL does not match the authoritative "
            "configuration."
        )
    verify_snapshots(snapshots)
    return ValidatedCandidate(
        spec=spec,
        setfl=setfl,
        raw_sha256=raw_sha,
        processed_sha256=processed_sha,
        raw_size_bytes=setfl.file_size_bytes,
        release_notes_available=release_notes_available,
        release_notes_sha256=release_notes_sha,
        source_metadata=source_metadata,
        retrieval_manifest=retrieval_manifest,
        snapshots=tuple(snapshots),
    )


def pair_coeff_line(spec: CandidateSpec, config: Step9Config) -> str:
    """Return the planned deterministic pair_coeff command for one candidate.

    ``Al Ni`` after the filename maps LAMMPS atom types 1 and 2 to element
    names; LAMMPS resolves the names against the file header internally, so
    the mapping does not need to match the internal header order textually.
    """

    mapping = config.lammps_design.atom_type_mapping
    return (
        f"pair_coeff * * {spec.expected_filename} "
        f"{mapping['1']} {mapping['2']}"
    )


# ---------------------------------------------------------------------------
# Step 8 source validation
# ---------------------------------------------------------------------------


def validate_step8_success(
    config: Step9Config,
) -> tuple[FileSnapshot, ...]:
    """Confirm Step 8 finished with SUCCESS and fingerprint its evidence."""

    report_path = (
        config.project_root
        / "results"
        / "mace_vs_dft"
        / "reports"
        / "ni_al_step8_final_report.txt"
    )
    checkpoint_path = (
        config.project_root
        / "results"
        / "mace_vs_dft"
        / "checkpoints"
        / "step8_benchmark_result.json"
    )
    snapshots = (
        snapshot_file(report_path, "Step 8 final report"),
        snapshot_file(checkpoint_path, "Step 8 checkpoint"),
    )
    text = report_path.read_text(encoding="utf-8")
    if "OVERALL STEP 8 STATUS: SUCCESS" not in text:
        raise Step9InputError(
            "The Step 8 final report does not record OVERALL STEP 8 STATUS: "
            "SUCCESS; Step 9 cannot proceed from unfinished Step 8 results."
        )
    try:
        checkpoint = read_strict_json(checkpoint_path, "Step 8 checkpoint")
    except Step7Error as exc:
        raise Step9InputError(str(exc)) from exc
    if checkpoint.get("overall_status") != "SUCCESS":
        raise Step9InputError("The Step 8 checkpoint is not SUCCESS.")
    verify_snapshots(snapshots)
    return snapshots


# ---------------------------------------------------------------------------
# LAMMPS availability inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LammpsAvailability:
    """Result of the safe local LAMMPS availability inspection."""

    status: str
    executable_path: str | None
    version_line: str | None
    eam_alloy_listed: bool
    python_lammps_package_present: bool
    detail: str


def inspect_lammps_availability(config: Step9Config) -> LammpsAvailability:
    """Safely inspect local LAMMPS availability without running anything.

    Only ``<executable> -h`` is invoked when an executable is found; no
    simulation runs, no atoms are created, and no potential file is read
    into a calculation.  Absence of LAMMPS never fails Step 9; it only
    marks Step 10 execution readiness as incomplete.
    """

    try:
        python_package = importlib.util.find_spec("lammps") is not None
    except (ImportError, ValueError):
        python_package = False

    executable: str | None = None
    for name in config.lammps_design.executable_candidates:
        located = shutil.which(name)
        if located:
            executable = located
            break
    if executable is None:
        return LammpsAvailability(
            status="NOT_FOUND",
            executable_path=None,
            version_line=None,
            eam_alloy_listed=False,
            python_lammps_package_present=python_package,
            detail=(
                "No LAMMPS executable was found on PATH among "
                f"{list(config.lammps_design.executable_candidates)}. LAMMPS "
                "was not installed automatically."
            ),
        )
    try:
        completed = subprocess.run(
            [executable, "-h"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return LammpsAvailability(
            status="CHECK_FAILED",
            executable_path=executable,
            version_line=None,
            eam_alloy_listed=False,
            python_lammps_package_present=python_package,
            detail=f"'{executable} -h' failed: {type(exc).__name__}: {exc}",
        )
    version_line = next(
        (
            line.strip()
            for line in output.splitlines()
            if "LAMMPS" in line and line.strip()
        ),
        None,
    )
    eam_alloy = "eam/alloy" in output
    status = (
        "AVAILABLE_AND_EAM_ALLOY_CONFIRMED"
        if eam_alloy
        else "AVAILABLE_BUT_EAM_ALLOY_UNCONFIRMED"
    )
    return LammpsAvailability(
        status=status,
        executable_path=executable,
        version_line=version_line,
        eam_alloy_listed=eam_alloy,
        python_lammps_package_present=python_package,
        detail=(
            f"'{executable} -h' completed; eam/alloy "
            f"{'was' if eam_alloy else 'was not'} listed in the help output."
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation matrix
# ---------------------------------------------------------------------------


def build_evaluation_matrix(
    config: Step9Config,
    validated: Mapping[str, ValidatedCandidate],
    lammps: LammpsAvailability,
) -> list[dict[str, Any]]:
    """Build the documented qualitative candidate evaluation matrix.

    The ranking is qualitative and evidence-based (Primary / Secondary /
    Historical secondary); no fabricated numeric scores are assigned.
    """

    rows: list[dict[str, Any]] = []
    for key in config.candidate_order:
        spec = config.candidates[key]
        bundle = validated.get(key)
        rank, rank_label = ROLE_RANKS[spec.role]
        rows.append(
            {
                "candidate_key": key,
                "role": spec.role,
                "potential_name": spec.potential_name,
                "authors": spec.authors,
                "publication_year": spec.publication_year,
                "citation": spec.citation,
                "doi": spec.doi,
                "repository_identity": spec.repository_identity,
                "implementation_identity": spec.implementation_identity,
                "openkim_family_id": spec.openkim_family_id,
                "openkim_extended_id": spec.openkim_extended_id,
                "potential_formalism": spec.formalism,
                "lammps_pair_style": spec.pair_style,
                "official_filename": spec.expected_filename,
                "element_list": list(spec.expected_elements),
                "file_element_order": (
                    list(bundle.setfl.elements) if bundle else None
                ),
                "cutoff_A": bundle.setfl.cutoff_A if bundle else None,
                "nrho": bundle.setfl.nrho if bundle else None,
                "drho": bundle.setfl.drho if bundle else None,
                "nr": bundle.setfl.nr if bundle else None,
                "dr": bundle.setfl.dr if bundle else None,
                "sha256": bundle.raw_sha256 if bundle else None,
                "file_size_bytes": bundle.raw_size_bytes if bundle else None,
                "binary_ni_al_specific": True,
                "official_nist_file_available": bundle is not None,
                "openkim_cross_reference_available": True,
                "lammps_eam_alloy_compatible": True,
                "corrected_non_superseded_implementation": True,
                "fitting_emphasis": spec.fitting_scope,
                "pure_element_basis_documented": key == "pun_mishin_2009",
                "intermetallic_formation_energy_fitting_documented": (
                    key == "pun_mishin_2009"
                ),
                "b2_nial_relevance": True,
                "ni3al_relevance": key in {"pun_mishin_2009", "mishin_2004_ipr2"},
                "broader_composition_relevance": key == "pun_mishin_2009",
                "interface_mechanical_relevance": key == "pun_mishin_2009",
                "known_implementation_warnings": list(spec.known_warnings),
                "file_validation_status": (
                    "VALIDATED" if bundle else "UNAVAILABLE"
                ),
                "superseded_status": (
                    "uses corrected ipr2; superseded ipr1 rejected"
                    if key == "mishin_2004_ipr2"
                    else "not superseded"
                ),
                "release_notes_available": (
                    bundle.release_notes_available if bundle else None
                ),
                "lammps_local_availability": lammps.status,
                "future_benchmark_role": rank_label,
                "recommendation_rank": rank,
            }
        )
    return rows


__all__ = [
    "CANDIDATE_ORDER",
    "ELEMENT_ORDER",
    "EV_JOULE",
    "PHASE_ORDER",
    "ROLE_RANKS",
    "SCHEMA_VERSION",
    "SUPERSEDED_2004_IPR1_FILENAME",
    "CandidateBundlePaths",
    "CandidateSpec",
    "LammpsAvailability",
    "LammpsDesign",
    "SetflData",
    "SetflElementRecord",
    "SourcePolicy",
    "Step9CollisionError",
    "Step9Config",
    "Step9ConfigurationError",
    "Step9Error",
    "Step9InputError",
    "Step9PotentialFormatError",
    "Step9ResumeError",
    "Step9RetrievalError",
    "Step9SourceError",
    "ValidatedCandidate",
    "build_evaluation_matrix",
    "candidate_bundle_paths",
    "ev_per_A3_to_bar",
    "inspect_lammps_availability",
    "load_step9_config",
    "pair_coeff_line",
    "parse_setfl",
    "step9_directories",
    "step9_result_paths",
    "validate_candidate_bundle",
    "validate_candidate_keys",
    "validate_step8_success",
]

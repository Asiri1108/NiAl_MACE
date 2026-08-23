"""Retrieve the candidate classical Ni-Al potentials from NIST.

Files are downloaded only from the authoritative NIST Interatomic
Potentials Repository host over HTTPS, with redirect confinement to the
approved host, explicit timeouts, a bounded retry count, and a maximum
download size.  Downloaded potential files are treated strictly as data:
they are parsed and validated, never executed.  Numerical contents are
never altered; the processed copy is byte-identical to the validated raw
file, and both SHA-256 digests are recorded and compared.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from step7_utils import (
    Step7Error,
    file_sha256,
    publish_files_transactionally,
    read_strict_json,
    relative_path,
    utc_timestamp,
    write_strict_json_bytes,
)
from step9_utils import (
    CANDIDATE_ORDER,
    CandidateSpec,
    Step9CollisionError,
    Step9Config,
    Step9Error,
    Step9RetrievalError,
    Step9SourceError,
    candidate_bundle_paths,
    load_step9_config,
    parse_setfl,
    validate_candidate_keys,
)


LOGGER = logging.getLogger("ni_al_step9.fetch_potentials")
DEFAULT_CONFIG = Path("configs/ni_al_classical_potentials.json")


@dataclass(frozen=True)
class RetrievedFile:
    """One retrieved resource with its provenance record."""

    url: str
    final_url: str
    staged_path: Path
    sha256: str
    size_bytes: int
    retrieval_time_utc: str
    attempts: int


class _ConfinedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects that leave the approved HTTPS hosts."""

    def __init__(self, allowed_hosts: Sequence[str]) -> None:
        super().__init__()
        self._allowed_hosts = tuple(allowed_hosts)

    def redirect_request(  # noqa: D102 - documented by the class docstring.
        self, req, fp, code, msg, headers, newurl
    ):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in self._allowed_hosts:
            raise Step9SourceError(
                f"Redirect to unapproved location was rejected: {newurl!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for potential retrieval."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve the official candidate classical Ni-Al potential files "
            "from the NIST Interatomic Potentials Repository."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 9 configuration path, repository-relative by default.",
    )
    parser.add_argument(
        "--candidate",
        choices=(*CANDIDATE_ORDER, "all"),
        default="all",
        help="Retrieve one candidate or all three (default: all).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration, authoritative URLs, and planned paths "
            "without network retrieval."
        ),
    )
    action.add_argument(
        "--fetch",
        action="store_true",
        help="Retrieve the official files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only Step 9 raw and processed potential resources.",
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


def _requested_candidates(option: str) -> tuple[str, ...]:
    """Normalize the --candidate option."""

    if option == "all":
        return validate_candidate_keys(None)
    return validate_candidate_keys((option,))


def _existing_targets(
    config: Step9Config, candidates: Sequence[str]
) -> tuple[Path, ...]:
    """Return existing bundle targets for collision reporting."""

    existing: list[Path] = []
    for key in candidates:
        paths = candidate_bundle_paths(config, key)
        existing.extend(path for path in paths.all_paths() if path.exists())
    return tuple(existing)


def _download_to(
    config: Step9Config,
    url: str,
    destination: Path,
    label: str,
) -> RetrievedFile:
    """Download one HTTPS resource with confinement, timeout, and size cap."""

    policy = config.source_policy
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in (
        policy.allowed_download_hosts
    ):
        raise Step9SourceError(
            f"{label}: URL violates the authoritative source policy: {url!r}"
        )
    opener = urllib.request.build_opener(
        _ConfinedRedirectHandler(policy.allowed_download_hosts)
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": policy.user_agent}
    )
    last_error: Exception | None = None
    for attempt in range(1, policy.maximum_attempts_per_file + 1):
        try:
            with opener.open(
                request, timeout=policy.connection_timeout_seconds
            ) as response:
                final_url = response.geturl()
                final_parsed = urlparse(final_url)
                if (
                    final_parsed.scheme != "https"
                    or final_parsed.hostname
                    not in policy.allowed_download_hosts
                ):
                    raise Step9SourceError(
                        f"{label}: response resolved to an unapproved "
                        f"location: {final_url!r}"
                    )
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError:
                        declared_size = None
                    else:
                        if declared_size > policy.maximum_file_size_bytes:
                            raise Step9RetrievalError(
                                f"{label}: declared size {declared_size} "
                                "exceeds the configured maximum."
                            )
                destination.parent.mkdir(parents=True, exist_ok=True)
                received = 0
                with destination.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > policy.maximum_file_size_bytes:
                            raise Step9RetrievalError(
                                f"{label}: download exceeded the configured "
                                f"maximum of "
                                f"{policy.maximum_file_size_bytes} bytes."
                            )
                        handle.write(chunk)
            if received == 0:
                raise Step9RetrievalError(f"{label}: downloaded file is empty.")
            return RetrievedFile(
                url=url,
                final_url=final_url,
                staged_path=destination,
                sha256=file_sha256(destination),
                size_bytes=received,
                retrieval_time_utc=utc_timestamp(),
                attempts=attempt,
            )
        except (Step9Error,) as exc:
            raise exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            LOGGER.warning(
                "%s: attempt %d/%d failed: %s",
                label,
                attempt,
                policy.maximum_attempts_per_file,
                exc,
            )
    raise Step9RetrievalError(
        f"{label}: retrieval failed after "
        f"{config.source_policy.maximum_attempts_per_file} attempt(s): "
        f"{type(last_error).__name__}: {last_error}"
    )


def _stage_candidate(
    config: Step9Config,
    key: str,
    staging_root: Path,
    data_root: Path,
) -> tuple[dict[Path, Path], Mapping[str, Any]]:
    """Retrieve, validate, and stage one complete candidate bundle."""

    spec: CandidateSpec = config.candidates[key]
    paths = candidate_bundle_paths(config, key)

    def staged(final_path: Path) -> Path:
        target = staging_root / final_path.resolve().relative_to(
            data_root.resolve()
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    LOGGER.info("Retrieving %s (%s)...", key, spec.implementation_identity)
    potential = _download_to(
        config,
        spec.potential_file_url,
        staged(paths.raw_potential),
        f"{key} potential file",
    )
    release_notes: RetrievedFile | None = None
    release_notes_error: str | None = None
    try:
        release_notes = _download_to(
            config,
            spec.release_notes_url,
            staged(paths.raw_release_notes),
            f"{key} release notes",
        )
    except Step9Error as exc:
        # Release notes are provenance context; their absence is recorded as
        # a warning while the potential file itself remains mandatory.
        release_notes_error = str(exc)
        LOGGER.warning("%s: release notes unavailable: %s", key, exc)

    # Validate the downloaded setfl file completely before anything is
    # published; the potential file is data and is never executed.
    setfl = parse_setfl(potential.staged_path, config)
    LOGGER.info(
        "  %s: setfl valid; elements=%s; Nrho=%d; Nr=%d; cutoff=%.6f A",
        key,
        "/".join(setfl.elements),
        setfl.nrho,
        setfl.nr,
        setfl.cutoff_A,
    )

    processed_staged = staged(paths.processed_potential)
    processed_staged.write_bytes(potential.staged_path.read_bytes())
    processed_sha = file_sha256(processed_staged)
    if processed_sha != potential.sha256:
        raise Step9RetrievalError(
            f"{key}: staged processed copy is not byte-identical to the raw "
            "download."
        )

    source_metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "candidate_key": key,
        "role": spec.role,
        "potential_name": spec.potential_name,
        "formalism": spec.formalism,
        "authors": spec.authors,
        "publication_year": spec.publication_year,
        "citation": spec.citation,
        "doi": spec.doi,
        "source_organization": "NIST Interatomic Potentials Repository",
        "source_page": spec.entry_page_url,
        "repository_identity": spec.repository_identity,
        "implementation_identity": spec.implementation_identity,
        "openkim_family_id": spec.openkim_family_id,
        "openkim_extended_id": spec.openkim_extended_id,
        "openkim_model_driver_family": spec.openkim_model_driver_family,
        "expected_filename": spec.expected_filename,
        "pair_style": spec.pair_style,
        "expected_elements": list(spec.expected_elements),
        "file_element_order": list(setfl.elements),
        "fitting_scope": spec.fitting_scope,
        "implementation_notes": spec.implementation_notes,
        "known_warnings": list(spec.known_warnings),
        "reject_superseded_ipr1": spec.reject_superseded_ipr1,
        "superseded_ipr1": (
            dict(spec.superseded_ipr1) if spec.superseded_ipr1 else None
        ),
        "setfl": {
            "comment_lines": list(setfl.comment_lines),
            "element_count": setfl.element_count,
            "elements": list(setfl.elements),
            "nrho": setfl.nrho,
            "drho": setfl.drho,
            "nr": setfl.nr,
            "dr": setfl.dr,
            "cutoff_A": setfl.cutoff_A,
            "element_records": [
                {
                    "name": record.name,
                    "atomic_number": record.atomic_number,
                    "mass_amu": record.mass_amu,
                    "lattice_constant_A": record.lattice_constant_A,
                    "lattice_type": record.lattice_type,
                }
                for record in setfl.element_records
            ],
            "total_tabulated_values": setfl.total_tabulated_values,
        },
    }
    retrieval_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "candidate_key": key,
        "retrieval_time_utc": potential.retrieval_time_utc,
        "potential_file_url": potential.url,
        "potential_file_final_url": potential.final_url,
        "potential_file_sha256": potential.sha256,
        "potential_file_size_bytes": potential.size_bytes,
        "potential_file_attempts": potential.attempts,
        "potential_file_local_path": relative_path(
            paths.raw_potential, config.project_root
        ),
        "processed_copy_local_path": relative_path(
            paths.processed_potential, config.project_root
        ),
        "processed_copy_sha256": processed_sha,
        "release_notes_url": spec.release_notes_url,
        "release_notes_available": release_notes is not None,
        "release_notes_sha256": (
            release_notes.sha256 if release_notes else None
        ),
        "release_notes_size_bytes": (
            release_notes.size_bytes if release_notes else None
        ),
        "release_notes_local_path": (
            relative_path(paths.raw_release_notes, config.project_root)
            if release_notes
            else None
        ),
        "release_notes_error": release_notes_error,
        "user_agent": config.source_policy.user_agent,
        "https_required": True,
        "allowed_hosts": list(config.source_policy.allowed_download_hosts),
        "configuration_fingerprint_sha256": config.fingerprint,
    }
    metadata_staged = staged(paths.source_metadata)
    metadata_staged.write_bytes(write_strict_json_bytes(source_metadata))
    manifest_staged = staged(paths.retrieval_manifest)
    manifest_staged.write_bytes(write_strict_json_bytes(retrieval_manifest))

    staged_by_final: dict[Path, Path] = {
        paths.raw_potential: potential.staged_path,
        paths.source_metadata: metadata_staged,
        paths.retrieval_manifest: manifest_staged,
        paths.processed_potential: processed_staged,
    }
    if release_notes is not None:
        staged_by_final[paths.raw_release_notes] = release_notes.staged_path
    return staged_by_final, retrieval_manifest


def run_validate_only(config: Step9Config, candidates: Sequence[str]) -> None:
    """Validate configuration, URLs, and planned paths without any network."""

    collisions = _existing_targets(config, candidates)
    print("=" * 78)
    print("STEP 9 POTENTIAL RETRIEVAL VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    print(f"Candidates: {', '.join(candidates)}")
    for key in candidates:
        spec = config.candidates[key]
        print(
            f"{key}: {spec.implementation_identity}; file "
            f"{spec.expected_filename}; host www.ctcms.nist.gov; HTTPS OK"
        )
    print(
        "Existing bundle targets: "
        + (
            "; ".join(
                relative_path(path, config.project_root) for path in collisions
            )
            if collisions
            else "None"
        )
    )
    print("Network download executed: No")
    print("Files written: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run_fetch(
    config: Step9Config,
    candidates: Sequence[str],
    overwrite: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Retrieve, validate, stage, and transactionally publish bundles."""

    if not overwrite:
        collisions = _existing_targets(config, candidates)
        if collisions:
            listing = "\n".join(
                f"  - {relative_path(path, config.project_root)}"
                for path in collisions
            )
            raise Step9CollisionError(
                "Existing Step 9 potential bundle files were found; re-run "
                "with --overwrite after review:\n" + listing
            )
    data_root = config.project_root / "data"
    manifests: list[Mapping[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".step9-potential-retrieval-", dir=config.project_root
    ) as temporary_name:
        staging_root = Path(temporary_name)
        staged_by_final: dict[Path, Path] = {}
        for key in candidates:
            staged, manifest = _stage_candidate(
                config, key, staging_root, data_root
            )
            staged_by_final.update(staged)
            manifests.append(manifest)

        def final_validator() -> None:
            for key in candidates:
                paths = candidate_bundle_paths(config, key)
                manifest = read_strict_json(
                    paths.retrieval_manifest, f"published {key} manifest"
                )
                if manifest.get("potential_file_sha256") != file_sha256(
                    paths.raw_potential
                ):
                    raise Step9RetrievalError(
                        f"Published {key} potential hash does not match its "
                        "manifest."
                    )
                if file_sha256(paths.raw_potential) != file_sha256(
                    paths.processed_potential
                ):
                    raise Step9RetrievalError(
                        f"Published {key} raw and processed copies differ."
                    )

        publish_files_transactionally(
            config.project_root,
            data_root,
            staged_by_final,
            overwrite=overwrite,
            final_validator=final_validator,
        )

    print("=" * 78)
    print("STEP 9 POTENTIAL RETRIEVAL COMPLETED")
    print("=" * 78)
    for manifest in manifests:
        print(
            f"{manifest['candidate_key']}: "
            f"{manifest['potential_file_local_path']} "
            f"({manifest['potential_file_size_bytes']} bytes; sha256 "
            f"{manifest['potential_file_sha256'][:16]}...); release notes "
            f"available: {manifest['release_notes_available']}"
        )
    print("Files were treated as data; nothing downloaded was executed.")
    print("=" * 78)
    return tuple(manifests)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.overwrite and not args.fetch:
        LOGGER.error("--overwrite is allowed only with --fetch.")
        return 1
    try:
        config = load_step9_config(args.config)
        candidates = _requested_candidates(args.candidate)
        if args.validate_only:
            run_validate_only(config, candidates)
        else:
            run_fetch(config, candidates, overwrite=args.overwrite)
        return 0
    except (Step9Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no partial candidate bundle was published.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

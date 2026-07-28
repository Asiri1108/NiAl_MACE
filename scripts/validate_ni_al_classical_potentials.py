"""Validate the retrieved classical Ni-Al potential bundles.

Every candidate `.eam.alloy` file is validated as a complete DYNAMO
multielement setfl file (headers, element identities, grid parameters,
finite parseable tabulated arrays, exact array counts, no trailing
garbage), together with its provenance metadata, retrieval manifest,
byte-identical processed copy, and the explicit rejection of the
superseded 2004 ipr1 file.  Files are only read; nothing is altered and
nothing is executed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from step7_utils import Step7Error, relative_path
from step9_utils import (
    CANDIDATE_ORDER,
    Step9ConfigurationError,
    Step9Error,
    load_step9_config,
    pair_coeff_line,
    validate_candidate_bundle,
    validate_candidate_keys,
)


LOGGER = logging.getLogger("ni_al_step9.validate_potentials")
DEFAULT_CONFIG = Path("configs/ni_al_classical_potentials.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for potential-file validation."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the retrieved classical Ni-Al setfl potential bundles "
            "without altering or executing them."
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
        help="Validate one candidate or all three (default: all).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files without altering them (required mode).",
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


def run_validation(config_path: Path, candidate_option: str) -> None:
    """Validate the requested candidate bundles and print the evidence."""

    config = load_step9_config(config_path)
    candidates = (
        validate_candidate_keys(None)
        if candidate_option == "all"
        else validate_candidate_keys((candidate_option,))
    )
    print("=" * 78)
    print("STEP 9 POTENTIAL-FILE VALIDATION")
    print("=" * 78)
    print(f"Configuration SHA-256: {config.fingerprint}")
    for key in candidates:
        bundle = validate_candidate_bundle(config, key)
        setfl = bundle.setfl
        print(f"{key}: VALIDATED")
        print(f"  file: {bundle.spec.expected_filename}")
        print(f"  sha256: {bundle.raw_sha256}")
        print(f"  size: {bundle.raw_size_bytes} bytes")
        print(f"  file element order: {' '.join(setfl.elements)}")
        print(
            f"  grids: Nrho={setfl.nrho}; drho={setfl.drho:.12g}; "
            f"Nr={setfl.nr}; dr={setfl.dr:.12g}; cutoff={setfl.cutoff_A:.12g} A"
        )
        print(
            "  element records: "
            + "; ".join(
                f"{record.name} (Z={record.atomic_number}, "
                f"m={record.mass_amu:.4f} amu, {record.lattice_type})"
                for record in setfl.element_records
            )
        )
        print(
            f"  tabulated values present: {setfl.total_tabulated_values} "
            "(complete; no truncation; no trailing content)"
        )
        print("  raw and processed copies byte-identical: PASS")
        print(
            "  release notes: "
            + (
                relative_path(
                    config.raw_root / key / bundle.spec.release_notes_filename,
                    config.project_root,
                )
                if bundle.release_notes_available
                else "not available"
            )
        )
        print(f"  planned mapping: {pair_coeff_line(bundle.spec, config)}")
        if key == "mishin_2004_ipr2":
            print(
                "  superseded ipr1 check: PASS (NiAl.eam.alloy is absent; "
                "corrected ipr2 sets F(rho=0)=0 so isolated atoms have zero "
                "energy)"
            )
    print("Potential values edited: No")
    print("Files executed: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        if not args.validate_only:
            raise Step9ConfigurationError(
                "This command only validates; pass --validate-only explicitly."
            )
        run_validation(args.config, args.candidate)
        return 0
    except (Step9Error, Step7Error) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted during validation; nothing was altered.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Validate or execute fixed-cell MACE relaxations for the Ni-Al phase set.

This command is intentionally a thin interface over :mod:`step6_utils`.  In
particular, importing this module does not import MACE or ASE's FIRE optimizer,
which keeps ``--validate-only`` a genuinely static validation path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from step6_utils import (
    ATOMIC_ONLY_MODE,
    PHASE_ORDER,
    Step6Context,
    Step6Error,
    execute_mode,
    load_and_validate_context,
    validate_mode_cli_plan,
)


LOGGER = logging.getLogger("ni_al_step6.atomic_only")
DEFAULT_CONFIG = Path("configs/mace_relaxation.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for atomic-only relaxation."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute fixed-cell FIRE relaxations of the five "
            "selected Ni-Al structures with MACE-MP-0 Small."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Relaxation configuration path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--phase",
        choices=PHASE_ORDER,
        help="Process one configured phase; omit to process all five phases.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and the output plan without importing MACE or FIRE.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute real atomic-only MACE relaxations.",
    )
    parser.add_argument(
        "--create-directories",
        action="store_true",
        help="Create the configured output directories during validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only colliding Step 6C outputs during execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete Step 6C bundles that pass strict validation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure deterministic console logging for the command."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _print_validation_plan(
    context: Step6Context, phase_keys: Sequence[str]
) -> None:
    """Print the scientific plan and explicit validation-only guarantees."""

    mode_settings = context.mode_settings
    safety = context.configuration.safety
    print("=" * 78)
    print("STEP 6C - ATOMIC-ONLY RELAXATION VALIDATION")
    print("=" * 78)
    print(f"Phases: {', '.join(phase_keys)}")
    print("Starting structures: independent original selected EXTXYZ files")
    print(
        "Optimizer plan: FIRE; fmax <= "
        f"{mode_settings.force_threshold_eV_per_A} eV/angstrom"
    )
    print(f"Maximum steps per phase: {mode_settings.maximum_steps}")
    print("Cell shape/volume: fixed; equality tolerance atol=1e-12, rtol=0")
    print(
        "Maximum periodic atomic displacement: "
        f"{safety.maximum_atomic_displacement_A} angstrom"
    )
    print("MACE loaded: No")
    print("Optimizer imported or created: No")
    print("Relaxation executed: No")
    print("Atoms changed: No")
    print("Cells changed: No")
    print("Validation status: SUCCESS")
    print("=" * 78)


def run(
    *,
    config_path: Path,
    phase: str | None,
    validate_only: bool,
    execute: bool,
    create_directories: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> int:
    """Run the selected atomic-only command mode and return a process code."""

    validate_mode_cli_plan(
        validate_only=validate_only,
        execute=execute,
        create_directories=create_directories,
        overwrite=overwrite,
        resume=resume,
    )
    phase_keys: tuple[str, ...] = (phase,) if phase else tuple(PHASE_ORDER)
    context = load_and_validate_context(
        config_path=config_path,
        mode=ATOMIC_ONLY_MODE,
        phase_keys=phase_keys,
        create_directories=create_directories,
        require_step6_outputs=False,
    )
    if validate_only:
        _print_validation_plan(context, phase_keys)
        return 0

    summary = execute_mode(
        context,
        phase_keys=phase_keys,
        overwrite=overwrite,
        resume=resume,
        calculator_session=None,
        publish_summary=phase is None,
    )
    LOGGER.info("Atomic-only execution completed: %s", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report controlled failures, and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return run(
            config_path=args.config,
            phase=args.phase,
            validate_only=args.validate_only,
            execute=args.execute,
            create_directories=args.create_directories,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    except Step6Error as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no incomplete bundle is presented as successful.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

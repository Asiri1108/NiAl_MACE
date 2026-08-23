"""Validate or execute independent full-cell MACE relaxations for Ni-Al.

The command delegates its numerical work to :mod:`step6_utils`.  Heavy
calculator, optimizer, and filter imports remain lazy so that validation-only
execution cannot accidentally evaluate or alter a structure.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import logging
import sys
from pathlib import Path
from typing import Sequence

from step6_utils import (
    FULL_CELL_MODE,
    PHASE_ORDER,
    Step6Context,
    Step6Error,
    execute_mode,
    load_and_validate_context,
    validate_mode_cli_plan,
)


LOGGER = logging.getLogger("ni_al_step6.full_cell")
DEFAULT_CONFIG = Path("configs/mace_relaxation.json")


def validate_full_cell_public_api() -> str:
    """Verify the installed public FrechetCellFilter API without creating it."""

    try:
        from ase.filters import FrechetCellFilter
    except ImportError as exc:
        try:
            ase_version = importlib.metadata.version("ase")
        except importlib.metadata.PackageNotFoundError:
            ase_version = "not installed"
        raise Step6Error(
            "ase.filters.FrechetCellFilter is unavailable in installed ASE "
            f"{ase_version}: {exc}"
        ) from exc
    parameters = inspect.signature(FrechetCellFilter).parameters
    required = {
        "atoms",
        "exp_cell_factor",
        "hydrostatic_strain",
        "constant_volume",
        "scalar_pressure",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise Step6Error(
            "Installed FrechetCellFilter public signature is missing: "
            + ", ".join(missing)
        )
    return importlib.metadata.version("ase")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for full-cell relaxation."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute independent FIRE/FrechetCellFilter relaxations "
            "of the five selected Ni-Al structures with MACE-MP-0 Small."
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
        help="Validate without loading MACE or importing/creating FIRE or a cell filter.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute real full-cell MACE relaxations.",
    )
    parser.add_argument(
        "--create-directories",
        action="store_true",
        help="Create the configured output directories during validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only colliding Step 6D outputs during execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete Step 6D bundles that pass strict validation.",
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
    """Print the plan and guarantees for static full-cell validation."""

    mode_settings = context.mode_settings
    execution = context.execution
    safety = context.configuration.safety
    print("=" * 78)
    print("STEP 6D - FULL-CELL RELAXATION VALIDATION")
    print("=" * 78)
    print(f"Phases: {', '.join(phase_keys)}")
    print("Starting structures: independent original selected EXTXYZ files")
    print("Filter plan: ase.filters.FrechetCellFilter")
    print(
        "Optimizer plan: FIRE; raw max force <= "
        f"{mode_settings.force_threshold_eV_per_A} eV/angstrom"
    )
    print(
        "Raw ASE Voigt stress criterion: max(abs(xx,yy,zz,yz,xz,xy)) <= "
        f"{mode_settings.stress_threshold_eV_per_A3} eV/angstrom^3"
    )
    print(f"Maximum steps per phase: {mode_settings.maximum_steps}")
    print(
        "Filter settings: hydrostatic_strain=false; constant_volume=false; "
        "external pressure="
        f"{execution.external_pressure_eV_per_A3} eV/angstrom^3"
    )
    print(
        "Safety: |volume change| <= "
        f"{safety.maximum_absolute_volume_change_percent}%; "
        "maximum internal displacement <= "
        f"{safety.maximum_atomic_displacement_A} angstrom"
    )
    print("MACE loaded: No")
    print("Optimizer imported or created: No")
    print("FrechetCellFilter instantiated: No")
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
    """Run the selected full-cell command mode and return a process code."""

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
        mode=FULL_CELL_MODE,
        phase_keys=phase_keys,
        create_directories=create_directories,
        require_step6_outputs=False,
    )
    ase_version = validate_full_cell_public_api()
    if validate_only:
        LOGGER.info("Validated FrechetCellFilter public API in ASE %s", ase_version)
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
    LOGGER.info("Full-cell execution completed: %s", summary)
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

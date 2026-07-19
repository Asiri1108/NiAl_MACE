"""
MACE baseline calculation for crystalline aluminum.

Purpose:
    This script verifies that the pretrained MACE-MP-0 model can be
    loaded and used as an ASE calculator on the current computer.

    Two aluminum structures are evaluated:

    1. A perfect FCC aluminum supercell.
    2. The same supercell after displacing one atom.

    Comparing these structures confirms that MACE can calculate:
        - Total potential energy
        - Potential energy per atom
        - Atomic forces
        - Stress

Usage:
    Run this script from the project root directory:

        python scripts/al_mace_baseline.py

Important:
    The pretrained model may be downloaded automatically during the
    first execution. Internet access is therefore required the first time.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from mace.calculators import mace_mp


# Determine the root directory of the project.
#
# __file__ points to:
#     project/scripts/al_mace_baseline.py
#
# parents[1] therefore points to:
#     project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define where generated results will be saved.
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "tables"
STRUCTURES_DIRECTORY = PROJECT_ROOT / "results" / "structures"

# Create the result directories if they do not already exist.
RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
STRUCTURES_DIRECTORY.mkdir(parents=True, exist_ok=True)


@dataclass
class StructureResult:
    """
    Store the calculated properties of one atomic structure.

    Attributes:
        label:
            A human-readable name for the evaluated structure.

        number_of_atoms:
            Number of atoms inside the simulation cell.

        total_energy:
            Total potential energy of the structure in electronvolts.

        energy_per_atom:
            Total energy divided by the number of atoms.

        maximum_force:
            Largest atomic force magnitude in eV/Å.

        mean_force:
            Average atomic force magnitude in eV/Å.

        stress:
            Stress tensor in Voigt notation.
    """

    label: str
    number_of_atoms: int
    total_energy: float
    energy_per_atom: float
    maximum_force: float
    mean_force: float
    stress: np.ndarray


def create_aluminum_supercell() -> Atoms:
    """
    Create a periodic 2 × 2 × 2 FCC aluminum supercell.

    Returns:
        An ASE Atoms object containing 32 aluminum atoms.

    Explanation:
        A conventional FCC unit cell contains four atoms.

        Repeating it twice along x, y, and z gives:

            4 × 2 × 2 × 2 = 32 atoms

        This larger structure allows us to move one atom independently
        and observe the forces predicted by MACE.
    """

    # Define the initial aluminum lattice constant in angstroms.
    lattice_constant = 4.05

    # Create a conventional cubic FCC aluminum cell.
    #
    # cubic=True produces a four-atom conventional cell instead of
    # the one-atom primitive cell used in the previous ASE test.
    conventional_cell = bulk(
        name="Al",
        crystalstructure="fcc",
        a=lattice_constant,
        cubic=True,
    )

    # Repeat the conventional cell twice in every spatial direction.
    supercell = conventional_cell.repeat((2, 2, 2))

    return supercell


def evaluate_structure(
    atoms: Atoms,
    calculator,
    label: str,
) -> tuple[StructureResult, Atoms]:
    """
    Evaluate energy, forces, and stress using a MACE calculator.

    Args:
        atoms:
            Atomic structure that will be evaluated.

        calculator:
            ASE-compatible MACE calculator.

        label:
            Descriptive name for the structure.

    Returns:
        A tuple containing:
            1. A StructureResult object with numerical results.
            2. A copied ASE structure containing the MACE predictions.
    """

    # Attach the MACE calculator to the atomic structure.
    atoms.calc = calculator

    # Calculate the total potential energy in electronvolts.
    total_energy = float(atoms.get_potential_energy())

    # Calculate the force vector acting on every atom.
    #
    # The returned array has shape:
    #     number_of_atoms × 3
    #
    # The three columns represent forces in x, y, and z.
    forces = atoms.get_forces()

    # Calculate the periodic-cell stress in Voigt notation.
    #
    # ASE returns six components:
    #     xx, yy, zz, yz, xz, xy
    stress = atoms.get_stress(voigt=True)

    # Calculate the magnitude of each atomic force vector.
    force_magnitudes = np.linalg.norm(forces, axis=1)

    # Calculate useful summary values.
    number_of_atoms = len(atoms)
    energy_per_atom = total_energy / number_of_atoms
    maximum_force = float(np.max(force_magnitudes))
    mean_force = float(np.mean(force_magnitudes))

    # Store the numerical results in a structured Python object.
    result = StructureResult(
        label=label,
        number_of_atoms=number_of_atoms,
        total_energy=total_energy,
        energy_per_atom=energy_per_atom,
        maximum_force=maximum_force,
        mean_force=mean_force,
        stress=np.asarray(stress),
    )

    # Create an independent copy for saving.
    saved_atoms = atoms.copy()

    # Store metadata inside the extended XYZ structure.
    #
    # We use MACE-specific names so these predictions are not confused
    # later with reference DFT energies and forces.
    saved_atoms.info["configuration_label"] = label
    saved_atoms.info["MACE_total_energy_eV"] = total_energy
    saved_atoms.info["MACE_energy_per_atom_eV"] = energy_per_atom
    saved_atoms.info["MACE_stress"] = np.asarray(stress)

    # Store the force acting on every atom.
    saved_atoms.arrays["MACE_forces"] = np.asarray(forces)

    return result, saved_atoms


def print_result(result: StructureResult) -> None:
    """
    Print one structure result in a readable form.

    Args:
        result:
            Calculated result that will be displayed.
    """

    print("-" * 70)
    print(f"Structure          : {result.label}")
    print(f"Number of atoms    : {result.number_of_atoms}")
    print(f"Total energy       : {result.total_energy:.8f} eV")
    print(f"Energy per atom    : {result.energy_per_atom:.8f} eV/atom")
    print(f"Maximum force      : {result.maximum_force:.8f} eV/Å")
    print(f"Mean force         : {result.mean_force:.8f} eV/Å")
    print("Stress components  :")
    print(result.stress)


def save_text_report(
    perfect_result: StructureResult,
    displaced_result: StructureResult,
) -> Path:
    """
    Save the baseline calculation results to a text file.

    Args:
        perfect_result:
            Result for the undisturbed FCC structure.

        displaced_result:
            Result after moving one aluminum atom.

    Returns:
        Path of the generated report file.
    """

    # Calculate how much the total energy changed after displacement.
    energy_difference = (
        displaced_result.total_energy - perfect_result.total_energy
    )

    # Define the output report path.
    report_path = RESULTS_DIRECTORY / "al_mace_baseline.txt"

    # Build the complete text report.
    report = f"""MACE Aluminum Baseline Calculation
==================================

Model family:
MACE-MP-0

Model size:
small

Execution device:
CPU

Numerical precision:
float32

Perfect FCC aluminum
--------------------
Number of atoms: {perfect_result.number_of_atoms}
Total energy: {perfect_result.total_energy:.10f} eV
Energy per atom: {perfect_result.energy_per_atom:.10f} eV/atom
Maximum force: {perfect_result.maximum_force:.10f} eV/Å
Mean force: {perfect_result.mean_force:.10f} eV/Å
Stress: {perfect_result.stress.tolist()}

Displaced FCC aluminum
----------------------
Number of atoms: {displaced_result.number_of_atoms}
Total energy: {displaced_result.total_energy:.10f} eV
Energy per atom: {displaced_result.energy_per_atom:.10f} eV/atom
Maximum force: {displaced_result.maximum_force:.10f} eV/Å
Mean force: {displaced_result.mean_force:.10f} eV/Å
Stress: {displaced_result.stress.tolist()}

Comparison
----------
Energy change after displacement: {energy_difference:.10f} eV

Interpretation
--------------
The perfect structure tests the model on a symmetric FCC crystal.

The displaced structure tests whether the model responds to a local
change in atomic position by changing the energy and producing atomic
forces.
"""

    # Save the report using UTF-8 text encoding.
    report_path.write_text(report, encoding="utf-8")

    return report_path


def main() -> None:
    """
    Run the complete MACE aluminum baseline calculation.
    """

    print("=" * 70)
    print("MACE Aluminum Baseline Test")
    print("=" * 70)

    print("\nLoading the pretrained MACE-MP-0 small model...")

    # Load the small MACE-MP-0 foundation model as an ASE calculator.
    #
    # model="small":
    #     Explicitly selects the small MACE-MP-0 checkpoint.
    #
    # device="cpu":
    #     Runs calculations using the computer processor.
    #
    # default_dtype="float32":
    #     Uses 32-bit floating-point values to reduce CPU computation cost.
    #
    # dispersion=False:
    #     Does not add an external D3 dispersion correction.
    calculator = mace_mp(
        model="small",
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )

    print("MACE model loaded successfully.")

    # Create the perfect 32-atom FCC aluminum supercell.
    perfect_aluminum = create_aluminum_supercell()

    print(f"\nCreated aluminum supercell with {len(perfect_aluminum)} atoms.")

    # Evaluate the undisturbed aluminum structure.
    perfect_result, perfect_saved_structure = evaluate_structure(
        atoms=perfect_aluminum,
        calculator=calculator,
        label="Perfect FCC aluminum",
    )

    # Create an independent copy for the displaced structure.
    displaced_aluminum = perfect_aluminum.copy()

    # Move the first atom by 0.05 Å along the x direction.
    #
    # This deliberately breaks the perfect crystal symmetry and should
    # produce nonzero forces around the displaced atom.
    displacement = np.array([0.05, 0.0, 0.0])
    displaced_aluminum.positions[0] += displacement

    # Evaluate the displaced structure.
    displaced_result, displaced_saved_structure = evaluate_structure(
        atoms=displaced_aluminum,
        calculator=calculator,
        label="Displaced FCC aluminum",
    )

    # Print both sets of results.
    print("\nCalculation results:")
    print_result(perfect_result)
    print_result(displaced_result)

    # Calculate and display the energy difference.
    energy_difference = (
        displaced_result.total_energy - perfect_result.total_energy
    )

    print("-" * 70)
    print(
        "Energy change after displacement: "
        f"{energy_difference:.8f} eV"
    )

    # Save both structures in extended XYZ format.
    structures_path = (
        STRUCTURES_DIRECTORY / "al_mace_baseline.extxyz"
    )

    write(
        filename=structures_path,
        images=[
            perfect_saved_structure,
            displaced_saved_structure,
        ],
        format="extxyz",
    )

    # Save a readable numerical report.
    report_path = save_text_report(
        perfect_result=perfect_result,
        displaced_result=displaced_result,
    )

    print("\nGenerated files:")
    print(f"Structure file : {structures_path}")
    print(f"Text report    : {report_path}")

    print("\nMACE aluminum baseline test completed successfully.")


if __name__ == "__main__":
    # Run the main workflow only when this script is executed directly.
    main()
"""
ASE installation and structure-generation test.

Purpose:
    This script verifies that the Atomic Simulation Environment (ASE)
    is installed correctly and can generate a periodic FCC aluminum crystal.

Usage:
    Run this script from the project root directory:

        python scripts/check_ase.py
"""

from ase import Atoms
from ase.build import bulk


def print_structure_information(atoms: Atoms) -> None:
    """
    Print basic information about an ASE atomic structure.

    Args:
        atoms:
            An ASE Atoms object containing the chemical elements,
            atomic positions, simulation cell, and periodic boundaries.
    """

    print("=" * 60)
    print("ASE Environment Test")
    print("=" * 60)

    # Print the chemical formula of the generated structure.
    print(f"Chemical formula : {atoms.get_chemical_formula()}")

    # Print the number of atoms inside the primitive simulation cell.
    print(f"Number of atoms  : {len(atoms)}")

    # Print whether the structure repeats periodically in x, y, and z.
    print(f"Periodic boundary conditions: {atoms.get_pbc()}")

    # Print the volume of the primitive cell in cubic angstroms.
    print(f"Cell volume      : {atoms.get_volume():.6f} Å³")

    # Print the three vectors that define the periodic simulation cell.
    print("\nSimulation cell vectors:")
    print(atoms.get_cell())

    # Print Cartesian atomic coordinates in angstroms.
    print("\nAtomic positions in Cartesian coordinates:")
    print(atoms.get_positions())

    # Print atomic coordinates relative to the simulation cell vectors.
    print("\nScaled atomic positions:")
    print(atoms.get_scaled_positions())

    # Print atomic numbers.
    # Aluminum has atomic number 13.
    print("\nAtomic numbers:")
    print(atoms.get_atomic_numbers())


def main() -> None:
    """
    Create a primitive FCC aluminum structure and test ASE functionality.
    """

    # Define the aluminum lattice constant in angstroms.
    lattice_constant = 4.05

    # Create a primitive face-centered cubic aluminum cell.
    #
    # The primitive FCC cell contains one aluminum atom.
    # Periodic boundary conditions repeat this cell in three dimensions,
    # representing a continuous aluminum crystal.
    aluminum = bulk(
        name="Al",
        crystalstructure="fcc",
        a=lattice_constant,
    )

    # Print the generated structure information.
    print_structure_information(aluminum)

    print("\nASE test completed successfully.")


if __name__ == "__main__":
    # Run the main function only when this file is executed directly.
    main()
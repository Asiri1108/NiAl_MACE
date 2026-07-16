# ASE Environment Test 
# Atomic Simulation Environment.
# (function) def bulk
from ase.build import bulk

print("=" * 50)
print("ASE Environment Test")
print("=" * 50)

atoms = bulk("Al", "fcc", a=4.05)

print()
print("Chemical formula:")
print(atoms.get_chemical_formula())

print()
print("Number of atoms:")
print(len(atoms))

print()
print("Cell:")
print(atoms.cell)

print()
print("Atomic positions:")
print(atoms.positions)
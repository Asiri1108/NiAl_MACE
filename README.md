"# NiAl_MACE" 
# Ni–Al MACE Interatomic Potential Project

## Project Overview

This project investigates the use of a universal pretrained MACE interatomic potential for Ni–Al alloy systems.

The main goal is to evaluate the pretrained model on Ni–Al structures and then fine-tune it using consistent reference data. The fine-tuned model will be evaluated using machine-learning error metrics and physical validation tests before being used in molecular dynamics simulations.

## Research Question

Can a universal pretrained MACE model be fine-tuned to provide accurate and physically reliable predictions for Ni–Al alloy structures?

## Main Objectives

1. Evaluate a universal MACE model on Ni–Al structures before fine-tuning.
2. Prepare and validate a consistent Ni–Al reference dataset.
3. Fine-tune the selected MACE foundation model.
4. Evaluate energy, force, and stress predictions on unseen structures.
5. Validate important physical properties of Ni, Al, and Ni–Al phases.
6. Integrate the final model with LAMMPS for molecular dynamics simulations.
7. Visualize and analyze the simulation outputs using OVITO.

## Proposed Comparison

The project will compare:

* Reference calculations.
* The universal MACE model before fine-tuning.
* The Ni–Al fine-tuned MACE model.
* A traditional interatomic potential when an appropriate baseline is available.

## Project Workflow

1. Define the research question and scope.
2. Prepare the software environment.
3. Create basic Ni and Al structures.
4. Test the universal MACE baseline.
5. Collect and inspect reference data.
6. Convert the data into a consistent training format.
7. Create training, validation, and test splits.
8. Fine-tune the MACE model.
9. Evaluate machine-learning accuracy.
10. Validate physical properties.
11. Run molecular dynamics simulations.
12. Analyze results and document conclusions.

## Directory Structure

```text
NiAl_MACE/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── scripts/
├── configs/
├── models/
├── runs/
├── results/
│   ├── figures/
│   └── tables/
├── lammps/
├── ovito/
├── notebooks/
├── environment/
├── docs/
├── references/
├── README.md
└── .gitignore
```

## Data Management Rules

* Original reference data must remain unchanged in `data/raw/`.
* Cleaned and converted data must be stored in `data/processed/`.
* Training, validation, and test datasets must be stored in `data/splits/`.
* Test data must not be used during model training.
* Every dataset must have recorded provenance, units, calculation settings, and preprocessing history.
* Large generated files should not be committed directly to Git.

## Reproducibility Requirements

Every experiment must record:

* Operating system.
* Python version.
* MACE version.
* PyTorch version.
* ASE version.
* CUDA version.
* GPU or CPU type.
* Dataset version.
* Foundation model name and version.
* Training command.
* Configuration file.
* Random seed.
* Model checkpoint.
* Evaluation results.

## Current Status

### Step 0 — Project Definition

Status: Completed

Completed tasks:

* Defined the initial research goal.
* Defined the initial research question.
* Selected Ni–Al as the target alloy system.
* Selected MACE as the machine-learning interatomic potential framework.
* Defined the initial comparison between the universal and fine-tuned models.

### Step 1 — Project and Environment Setup

Status: In Progress

Current tasks:

* Create the project directory.
* Create the standard folder structure.
* Initialize the Git repository.
* Record the computer and software information.
* Prepare an isolated Python environment.

## Important Scientific Principle

A low machine-learning error alone does not prove that an interatomic potential is physically reliable.

The model must also be tested on structures and physical conditions that were not used during training. Physical properties and molecular dynamics stability must be evaluated separately.

## Change Log

### 2026-07-16

* Created the initial project structure.
* Added the research question and objectives.
* Added the proposed workflow.
* Added data management and reproducibility rules.
* Started the environment setup stage.
### Python Environment

- Project Python version: Python 3.11.9
- Virtual environment: `.venv`
- Active interpreter:

## Step 2 – Atomic Simulation Environment (ASE)

Status: Completed

Completed tasks:

- Installed ASE successfully.
- Verified ASE can generate FCC aluminum structures.
- Confirmed atomic positions and simulation cell can be created.
- Prepared the project for integration with MACE.

Lessons learned:

- ASE is not a physics engine.
- ASE is a framework that creates and manipulates atomic structures.
- ASE will become the interface between Python, MACE, and LAMMPS.

## Step 2 — Atomic Simulation Environment

Status: Completed

### Completed Tasks

* Installed the Atomic Simulation Environment.
* Verified that ASE can generate an FCC aluminum crystal.
* Generated the primitive FCC unit cell using a lattice constant of 4.05 Å.
* Confirmed that the primitive cell contains one aluminum atom.
* Confirmed that ASE correctly represents periodic cell vectors and atomic positions.

### ASE Test Result

The generated aluminum structure had:

* Chemical formula: Al
* Number of atoms: 1
* Crystal type: FCC primitive cell
* Lattice constant: 4.05 Å
* Periodic boundary representation: three-dimensional periodic cell

### Interpretation

The single atom does not represent the full aluminum crystal. It represents one primitive periodic cell that is repeated in three dimensions.

ASE will be used as the structure and calculator interface between Python, MACE, Materials Project data, and LAMMPS.

## Planned Data Source

Materials Project will be used to obtain Ni, Al, and Ni–Al structures and calculated properties.

The project will initially use Materials Project data for:

* Selecting relevant Ni–Al phases.
* Obtaining relaxed crystal structures.
* Recording material identifiers and calculation provenance.
* Comparing lattice, energy, formation-energy, and stability results.
* Investigating relaxation trajectories as a possible source of training configurations.

Materials Project summary data alone will not automatically be treated as a complete MACE training dataset. Each candidate training configuration must be checked for consistent energy, atomic-force, stress, structure, and DFT-method information.

Mixing data generated using different DFT functionals will be avoided unless the difference is explicitly understood and handled.

## Current Status

### Step 2 — Atomic Simulation Environment

Status: Completed

### Step 3 — MACE Installation and Baseline Testing

Status: Ready to Begin

### ASE Validation Result

The ASE validation script successfully created a primitive FCC aluminum cell.

Recorded results:

- Chemical formula: Al
- Number of atoms: 1
- Atomic number: 13
- Lattice constant: 4.05 Å
- Primitive-cell volume: 16.607531 Å³
- Periodic boundaries: enabled in all three dimensions

The calculated primitive-cell volume is consistent with an FCC conventional
cell containing four atoms. This confirms that ASE generated the expected
primitive aluminum structure correctly.

## Step 3 — MACE Aluminum Baseline

Status: Completed

### Execution Environment

- Python version: 3.11.9
- Python environment: `.venv`
- Execution device: CPU
- Foundation model: MACE-MP-0 small
- Numerical precision: float32
- Dispersion correction: disabled

### Evaluated System

A 2 × 2 × 2 conventional FCC aluminum supercell was generated using ASE.

- Number of atoms: 32
- Initial lattice constant: 4.05 Å
- First structure: perfect FCC aluminum
- Second structure: one atom displaced by 0.05 Å along the x direction

### Baseline Results

#### Perfect FCC Aluminum

- Total energy: -118.70617676 eV
- Energy per atom: -3.70956802 eV/atom
- Maximum force: 0.00000245 eV/Å
- Mean force: 0.00000140 eV/Å

#### Displaced FCC Aluminum

- Total energy: -118.70216370 eV
- Energy per atom: -3.70944262 eV/atom
- Maximum force: 0.16058496 eV/Å
- Mean force: 0.01531472 eV/Å

### Comparison

Moving one atom by 0.05 Å increased the predicted energy by:

- 0.00401306 eV

The perfect periodic crystal produced forces close to zero. The displaced
structure produced a clear increase in force and energy. This confirms that
the pretrained MACE model responds to changes in the local atomic environment.

This test validates the software workflow only. It does not yet establish
the final physical accuracy of the model.

### Generated Files

```text
scripts/al_mace_baseline.py
results/structures/al_mace_baseline.extxyz
results/tables/al_mace_baseline.txt

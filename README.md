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

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
```

## Step 4 — Ni-Al Phase Structure Acquisition

Status: Completed on 2026-07-20. The authenticated Materials Project
acquisition retained nine exact-composition candidates and selected one
working structure for each of the five target phases.

### Purpose and Scientific Scope

This step prepares a reproducible, provenance-rich structure dataset from
Materials Project. It downloads crystal structures and summary metadata only;
it does not train or fine-tune MACE, relax structures, run molecular dynamics,
or perform EAM, MEAM, or LAMMPS calculations.

The five target intermetallic compositions are:

* `Al3Ni`
* `Al3Ni2`
* `AlNi` / `NiAl`
* `Al3Ni5` / `Ni5Al3`
* `AlNi3` / `Ni3Al`

The pretrained MACE-MP-0 model already supports both Al and Ni. Pure aluminum
and pure nickel are therefore not being trained separately before the
compounds are studied. They may be introduced later as elemental reference
structures for formation-energy calculations, but they are not part of this
download step.

Materials Project summary structures are not automatically a force-training
trajectory dataset. A summary structure does not by itself provide the
consistent collection of energies, atomic forces, stresses, configurations,
and calculation settings needed for force training.

### Candidate Preservation and Selection

The acquisition script queries all current, non-deprecated summary entries for
each configured formula. It uses pymatgen compositions to verify the exact
reduced composition instead of relying on formula-string order. Every exact
candidate is preserved under `data/raw/`; alternative polymorphs are never
silently discarded.

One candidate is copied to the selected working directory using this
deterministic order:

1. Lowest available energy above hull.
2. Stable entries before entries not marked stable when hull energies tie.
3. Lowest available formation energy per atom when the earlier values tie.
4. Lexicographical Materials Project ID as the final tie-breaker.

Missing numerical values rank as positive infinity. The selected candidate is
the project's reproducible current working structure, not a claim of absolute
experimental ground truth. The manifest and metadata record the full ranking
and a human-readable selection reason.

### API-Key Security and Installation

Install only the Step 4 packages into the existing project environment from
Windows CMD:

```bat
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\python.exe -m pip install mp-api python-dotenv
```

Create the local API configuration and run validation before downloading:

```bat
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\activate.bat
copy .env.example .env
python scripts\fetch_ni_al_structures.py --validate-only
python scripts\fetch_ni_al_structures.py
```

Edit `.env` locally and replace the example value with a Materials Project API
key. The script also accepts `MP_API_KEY` when it is already defined as a
Windows environment variable. It never prints the key. **The `.env` file must
never be committed.** `.env.example` contains only a safe placeholder.

To download one phase only:

```bat
python scripts\fetch_ni_al_structures.py --phase AlNi
```

To replace existing downloaded phase files after reviewing the consequences:

```bat
python scripts\fetch_ni_al_structures.py --overwrite
```

Without `--overwrite`, an existing complete candidate bundle causes a clear
failure rather than an implicit replacement. An incomplete bundle always
stops the workflow with a partial-output diagnostic so its provenance can be
inspected before another acquisition is attempted. During an explicit
`--overwrite` refresh, candidate bundles no longer returned by the current API
query are reported and removed in the same rollback-protected transaction that
publishes the replacement files and manifests.

### Output Layout

Directories are created only by a real successful acquisition when needed:

```text
configs/
└── ni_al_phases.json
data/
├── raw/materials_project/ni_al/
│   └── <phase_key>/<material_id>/
│       ├── structure.cif
│       ├── structure.extxyz
│       └── metadata.json
└── processed/ni_al_structures/
    ├── selected/
    │   ├── Al3Ni.cif
    │   ├── Al3Ni.extxyz
    │   ├── Al3Ni.metadata.json
    │   └── ... corresponding files for all five phases
    ├── ni_al_phase_manifest.csv
    └── ni_al_phase_manifest.json
```

Each candidate has one record in both manifests. Records contain composition,
energetic and symmetry fields when available, selection rank and reason,
retrieval time, and repository-relative raw and selected paths. A single-phase
run merges its new records with existing records for other phases instead of
erasing them.

### Success Criteria and Current Project Status

Step 4 acquisition is complete when all five formulas have been queried, all
exact-composition candidates and provenance metadata have been saved, exactly
one working candidate per phase has been selected deterministically, both
manifests agree, and the console reports zero failed phases.

The code, configuration, API-key safeguards, and validation path are
implemented. The authenticated acquisition completed successfully: nine raw
candidates were retained, five working structures were selected, and the JSON
and CSV manifests record their provenance and deterministic rankings.

### Next Step

The selected Ni-Al structures will be evaluated using the pretrained
MACE-MP-0 small model in a zero-shot single-point baseline before any
fine-tuning decision is made.

## Step 5 — MACE-MP-0 Zero-Shot Evaluation of Ni-Al Phases

Status: Completed on 2026-07-21. Validation passed, the cached pretrained model
loaded successfully, and all five requested phases completed with zero failures.

### Scientific Purpose and Scope

Step 5 tests whether the pretrained MACE-MP-0 small foundation model can
successfully calculate finite energies, forces, and stresses for the five
selected Ni-Al intermetallic structures without additional training. This is a
zero-shot software and physical-response baseline, not yet a complete accuracy
validation against DFT or experiment.

Zero-shot evaluation means that the pretrained model is applied directly to
the selected structures without project-specific training or fine-tuning. The
model is not trained or fine-tuned in this step. Atomic positions, cell vectors,
chemical species, and periodic boundary conditions are preserved exactly as
downloaded, and no geometry optimization or relaxation is performed.

The default calculation settings are:

* Foundation model: MACE-MP-0 small.
* Execution device: CPU.
* Numerical precision: float64.
* Dispersion correction: disabled.
* Evaluation type: zero-shot single-point calculation.

The evaluated phases, in report order, are:

1. `Al3Ni`
2. `Al3Ni2`
3. `AlNi`
4. `Al3Ni5`
5. `AlNi3`

### Calculated Properties and Interpretation

For each phase, the script calculates the total energy, energy per atom,
atomic force vectors, periodic-cell stress, cell volume, and volume per atom.
It also reports the maximum, mean, root-mean-square, and minimum magnitudes of
the per-atom force vectors, together with the vector sum of all atomic forces
and the norm of that total-force vector. The RMS statistic is
`sqrt(mean(|F_i|^2))`, where `|F_i|` is the magnitude of one atom's force.

Stress is reported in the ASE Voigt component order `xx, yy, zz, yz, xz, xy`
in eV/angstrom^3. ASE uses positive stress for tension; hydrostatic compression
has negative diagonal stress components under this convention.

Materials Project formation energies must not be compared directly with raw
MACE total energies in this step. The two sources may use different elemental
energy references and calculation conventions. Step 5 therefore does not
calculate formation energies or subtract elemental reference energies.

The selected Materials Project structures were optimized on a DFT energy
surface, not on the MACE energy surface. Nonzero MACE forces are consequently
expected in general and do not by themselves show that either a structure or
the model is incorrect. A maximum force from this single-point calculation is
not sufficient to classify a phase as physically stable or unstable.

### Inputs and Generated Outputs

The input directory is:

```text
data/processed/ni_al_structures/selected/
```

Original selected files are read only. Annotated copies and summaries are
written under:

```text
results/mace_zero_shot/
├── structures/
│   ├── Al3Ni_mace_zero_shot.extxyz
│   ├── Al3Ni2_mace_zero_shot.extxyz
│   ├── AlNi_mace_zero_shot.extxyz
│   ├── Al3Ni5_mace_zero_shot.extxyz
│   └── AlNi3_mace_zero_shot.extxyz
├── tables/
│   ├── ni_al_mace_zero_shot.csv
│   └── ni_al_mace_zero_shot.json
└── reports/
    └── ni_al_mace_zero_shot.txt
```

Each annotated EXTXYZ retains the source structure and adds explicitly named
MACE energy, force, and stress fields plus model and provenance metadata. The
original Materials Project selected structures are never overwritten.

### Windows CMD Commands

Activate the existing project environment and validate all dependencies,
configuration fields, selected structures, formulas, and output paths without
loading the pretrained model:

```bat
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\activate.bat

python scripts\evaluate_ni_al_mace_zero_shot.py --validate-only
```

Run the complete five-phase zero-shot calculation:

```bat
python scripts\evaluate_ni_al_mace_zero_shot.py
```

Evaluate one phase only:

```bat
python scripts\evaluate_ni_al_mace_zero_shot.py --phase AlNi
```

Replace an existing, reviewed Step 5 result bundle:

```bat
python scripts\evaluate_ni_al_mace_zero_shot.py --overwrite
```

### Success Criteria and Current Project Status

Step 5 succeeds when configuration and input validation pass, the pretrained
model loads once, all requested phases return finite energy, force, stress, and
derived statistics, every annotated structure passes read-back verification,
the CSV, JSON, and text reports are published atomically, and the console
reports zero failed phases. Validation-only mode must not load the model, run a
calculation, or create result directories.

Current project status: the Step 5 implementation, configuration,
documentation, environment snapshot, five annotated structures, and three
summary reports are present. The CPU/float64 MACE-MP-0 small run completed all
five phases successfully. The authoritative execution record is
`results/mace_zero_shot/reports/ni_al_mace_zero_shot.txt`.

### Next Step

Reviewing the zero-shot results and preparing controlled geometry-relaxation
tests before comparing MACE with selected Ni-Al potentials in LAMMPS.

## Step 6A — Geometry-Relaxation Configuration and Validation

Status: Completed on 2026-07-22. The relaxation design, selected structures,
metadata, Step 5 baseline, and planned output paths have been validated for all
five phases. No relaxation or new scientific calculation was run.

### Purpose and Scope

Step 6A prepares a controlled and reviewable relaxation workflow before MACE is
allowed to act on a structure. It validates the planned parameters, confirms
that every input is a finite three-dimensional periodic Al-Ni structure with
the expected reduced composition and Materials Project provenance, and checks
that a successful finite Step 5 single-point record and annotated structure
exist for every requested phase.

The two future modes remain separate and each will start from an independent
copy of the original selected structure:

* `atomic_only` may change atomic positions while cell vectors, volume, shape,
  and periodic boundary conditions remain fixed.
* `full_cell` may change atomic positions, cell vectors, volume, and shape while
  periodic boundary conditions remain enabled.

This separation will show whether differences between the Materials Project
DFT geometry and the future MACE geometry arise mainly from internal atomic
coordinates, cell dimensions or shape, or both. Step 6A does not load MACE,
attach an ASE calculator, request energy, forces, or stress, move atoms, change
the cell, or perform geometry optimization.

### Configuration and Initial Parameters

The configuration is stored in:

```text
configs/mace_relaxation.json
```

The preparation and validation script is:

```text
scripts/validate_ni_al_mace_relaxation.py
```

The initial controlled defaults are:

| Setting | `atomic_only` | `full_cell` |
|---|---:|---:|
| Model | MACE-MP-0 small | MACE-MP-0 small |
| Device and precision | CPU, float64 | CPU, float64 |
| Dispersion | disabled | disabled |
| Optimizer | FIRE | FIRE |
| Force threshold | 0.01 eV/angstrom | 0.01 eV/angstrom |
| Stress threshold | not applicable | 0.0006241509 eV/angstrom^3 |
| Maximum steps | 500 | 1000 |
| Trajectory interval | 1 | 1 |
| Cell shape/volume changes | prohibited | allowed |

These are initial controlled parameters, not a claim that convergence behavior
is insensitive to the chosen thresholds or optimizer. Later work may test
parameter sensitivity after the reproducible baseline workflow is established.

The safety design stops later runs on nonfinite values, preserves original
structures, requires a periodic cell, limits the maximum absolute volume change
to 25%, and limits the maximum atomic displacement to 2.0 angstrom.

### Inputs and Planned Outputs

Selected structures and metadata are read from:

```text
data/processed/ni_al_structures/selected/
```

The Step 5 baseline table is read without recalculating its values:

```text
results/mace_zero_shot/tables/ni_al_mace_zero_shot.json
```

With `--create-directories`, the script creates only this empty preparation
tree; it creates no relaxed structures, trajectories, result tables, or
scientific reports:

```text
results/mace_relaxation/
├── atomic_only/
│   ├── structures/
│   ├── trajectories/
│   ├── tables/
│   └── reports/
├── full_cell/
│   ├── structures/
│   ├── trajectories/
│   ├── tables/
│   └── reports/
└── comparison/
    ├── tables/
    └── reports/
```

### Windows CMD Validation Commands

```bat
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\activate.bat

python scripts\validate_ni_al_mace_relaxation.py

python scripts\validate_ni_al_mace_relaxation.py --phase AlNi

python scripts\validate_ni_al_mace_relaxation.py --mode atomic_only

python scripts\validate_ni_al_mace_relaxation.py --create-directories
```

### Success Criteria and Current Project Status

Step 6A succeeds when the JSON configuration and both mode definitions are
valid, all requested EXTXYZ and metadata pairs pass structural and provenance
checks, matching successful Step 5 records contain finite stored energy, force,
and stress values, annotated Step 5 structures exist, original inputs remain
unchanged, and any requested output preparation creates directories only.

Current project status: Step 6A is complete. Five phases validated, zero phases
failed, the empty planned output tree was created, MACE was not loaded, and no
atoms, cell vectors, or scientific results were changed or generated.

### Next Sub-step

Step 6B — Loading MACE once and reproducing the initial single-point energy,
force, stress, and volume values before any relaxation is allowed.

## Step 6B.1 — MACE Model Loading Test

Purpose: verify that the model settings prepared in Step 6A can construct one
valid ASE-compatible MACE calculator before any structure or physical property
enters the workflow.

The executable script is:

```text
scripts/reproduce_ni_al_mace_baseline.py
```

It reads only the `model` section of:

```text
configs/mace_relaxation.json
```

The configured settings are MACE-MP-0 small, CPU execution, float64 precision,
and dispersion disabled. The script imports the installed `mace_mp` factory and
loads the configured calculator exactly once per invocation. It validates that
the returned object is an ASE calculator and reports its class.

Run from Windows CMD with the project environment active:

```bat
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\activate.bat

python scripts\reproduce_ni_al_mace_baseline.py --load-only
python scripts\reproduce_ni_al_mace_baseline.py --load-only --verbose
```

Step 6B.1 does not read CIF or EXTXYZ files, create an ASE `Atoms` object,
attach the calculator to atoms, calculate energy, forces, stress, or volume,
run an optimizer, change geometry, or create scientific output files. Success
means that the configuration is valid, MACE imports, one non-null
ASE-compatible calculator is created, and the process exits successfully.

### Next Sub-step

Step 6B.2 — Read only the AlNi structure, attach the already configured MACE
calculator, and reproduce its initial single-point energy, forces, stress, and
volume without moving atoms or changing the cell.

## Step 6B.2 — AlNi Initial Baseline Reproduction

Status: Completed on 2026-07-26. The reproduced AlNi values matched every
required Step 5 comparison within tolerance, and all geometry, ordering,
periodicity, volume, and read-only source-file checks passed.

### Purpose and Pilot Scope

Step 6B.2 tests computational reproducibility before any geometry relaxation is
allowed. It recalculates the initial MACE energy, forces, stress, and volume for
one unchanged structure and compares the new values with the stored Step 5
record. This is a reproducibility test: it checks whether the same configured
calculation can be repeated consistently. It is not an accuracy comparison
against DFT or experiment.

Only `AlNi` is used first because its two-atom B2 cell is the smallest selected
Ni-Al structure and its identity is unambiguous. Restricting this safety gate
to one simple phase makes it easier to verify the calculation, comparison, and
immutability logic before the workflow is extended to more complex cells.

The selected input and its expected identity are:

```text
data/processed/ni_al_structures/selected/AlNi.extxyz
Materials Project ID: mp-1487
```

The numerical baseline is the successful AlNi record in:

```text
results/mace_zero_shot/tables/ni_al_mace_zero_shot.json
```

The Step 5 JSON record remains the source of full-precision baseline numbers.
The annotated Step 5 EXTXYZ is checked for existence, identity, and preserved
geometry, but its serialized force columns are not used as the numerical
reference.

### Script and Commands

The controlled script is:

```text
scripts/reproduce_ni_al_mace_baseline.py
```

Run the isolated Step 6B.1 loading behavior:

```bat
.\.venv\Scripts\python.exe scripts\reproduce_ni_al_mace_baseline.py --load-only
```

Run the AlNi Step 6B.2 reproduction:

```bat
.\.venv\Scripts\python.exe scripts\reproduce_ni_al_mace_baseline.py --phase AlNi
```

Use `--verbose` for additional validation diagnostics. Explicit `--phase`
execution now accepts any reviewed Ni-Al phase so the same fixed-geometry
workflow can be smoke-tested one phase at a time. The successful AlNi Step
6B.2 report remains specially protected: its existing-file collision stops a
plain `--phase AlNi` invocation, and `--phase AlNi --overwrite` is rejected.
Only Step 6B.3 report targets may be intentionally replaced.

### Calculated Quantities

One MACE-MP-0 Small calculator is loaded on CPU with float64 precision and
dispersion disabled. It is attached only to a deep in-memory copy of the
selected AlNi structure. One fixed-geometry single-point evaluation requests:

* total energy and energy per atom;
* every atomic force vector;
* maximum and root-mean-square atomic force magnitudes;
* the total force vector and its norm;
* stress in ASE Voigt order `xx, yy, zz, yz, xz, xy`;
* volume and volume per atom.

Every returned and derived numerical value must be finite.

### Comparison Tolerances

Each scalar uses the criterion
`absolute_difference <= absolute_tolerance + relative_tolerance * abs(reference)`.
The reference is the stored Step 5 JSON value.

| Quantity | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Total energy | 1e-8 eV | 1e-10 |
| Energy per atom | 1e-9 eV/atom | 1e-10 |
| Force statistics and total-force components | 1e-8 eV/angstrom | 1e-8 |
| Stress components | 1e-9 eV/angstrom^3 | 1e-8 |
| Volume and volume per atom | 1e-10 angstrom^3 | 1e-10 |

Atom count and Materials Project ID must match exactly. Relative differences
are reported as not applicable when a Step 5 value is already at or below the
absolute tolerance; this is important for near-zero total-force and shear-
stress components, where the absolute tolerance controls the comparison.

### Immutability and Success Criteria

Before MACE is attached, independent copies are retained for positions, cell
vectors, symbols, atomic numbers, periodic boundaries, atom count, and volume.
After the property requests, every item is checked independently. Positions,
cell vectors, and volume use zero relative tolerance and an absolute tolerance
of `1e-12`; symbols, atomic numbers, ordering, atom count, and periodic
boundaries must match exactly. Content hashes, file sizes, and modification
times also confirm that the original structure, metadata, Step 5 JSON, and
annotated Step 5 structure remain unchanged.

Step 6B.2 succeeds only when the AlNi identity and baseline validations pass,
all required comparisons pass their tolerances, every immutability and
read-only file check passes, the calculator is loaded once, and exactly one
single-point calculation is completed. No optimizer is imported or created,
no FIRE run occurs, no relaxation is performed, and no structure or trajectory
is written.

The only persistent output is atomically published at:

```text
results/mace_relaxation/comparison/reports/AlNi_step6b2_baseline_reproduction.txt
```

### Next Sub-step

Step 6B.3 — Extend the verified single-point reproduction workflow to the
remaining four Ni-Al phases without allowing any structure or cell changes.

## Step 6B.3 — Remaining Ni-Al Baseline Reproduction

Status: Completed successfully on 2026-07-26. All four requested phases
reproduced every JSON-backed Step 5 value exactly in the verified CPU/float64
run. Identity, immutability, source-file, stress, volume, and overall
reproducibility checks passed for every phase.

### Purpose and Batch Scope

Step 6B.3 extends the reviewed AlNi pilot procedure to exactly these remaining
selected structures:

| Phase | Materials Project ID | Atoms |
|---|---|---:|
| Al3Ni | mp-622209 | 16 |
| Al3Ni2 | mp-1057 | 5 |
| Al3Ni5 | mp-16514 | 8 |
| AlNi3 | mp-2593 | 4 |

AlNi is deliberately absent from `--all-remaining` because it already passed
the Step 6B.2 pilot. The batch neither recalculates AlNi nor targets its report.
The protected report retained SHA-256
`53beefe9e502ac925d2d96cd267dcef039d27d882181b4aa6bbafef1239ed6b2`,
its 11,873-byte size, and its modification timestamp.

### Controlled Calculation and Comparison

The batch validates all four selected EXTXYZ/metadata pairs and their unique
successful Step 5 records before loading MACE. It then loads one MACE-MP-0
Small calculator on CPU in float64 with dispersion disabled. Each phase starts
from a separate deep copy of its own original structure. The calculator is
detached and reset between phases, so no calculated `Atoms` state or cached
calculator result becomes another phase's input. The authoritative batch count
is one calculator load and exactly four fixed-geometry single-point
calculations.

For every phase, the workflow records total energy, energy per atom, all
reproduced atomic force vectors, maximum and RMS force magnitude, total force
vector and norm, six ASE Voigt stress components, volume, and volume per atom.
The full-precision Step 5 JSON table is the sole numerical reference.
Per-atom vectors are recorded for inspection, but the Step 5 JSON does not
contain a full-precision per-atom vector array; rounded annotated-EXTXYZ force
columns are therefore not used as comparison references. All JSON-backed
energy, force aggregate/vector-component, stress, volume, identity, and
atom-count fields are compared.

Numerical comparisons use
`absolute_difference <= absolute_tolerance + relative_tolerance * abs(reference)`.
The Step 6B.2 tolerances are retained: `1e-8` eV plus `1e-10` relative for
total energy; `1e-9` eV/atom plus `1e-10` relative for energy per atom;
`1e-8` eV/angstrom plus `1e-8` relative for force quantities; `1e-9`
eV/angstrom^3 plus `1e-8` relative for stress; and `1e-10` angstrom^3 plus
`1e-10` relative for volume quantities. Identity and atom count require exact
equality.

### Immutability, Publication, and Commands

Positions, cell vectors, symbols/order, atomic numbers/order, atom count, PBC,
and volume are copied before calculation and checked afterward. Positions,
cell, and volume use `atol=1e-12` and `rtol=0`; the discrete properties require
exact equality. The selected EXTXYZ, selected metadata, shared Step 5 JSON, and
annotated Step 5 EXTXYZ are protected per phase by before/after SHA-256, file
size, and modification-time checks. All sources are rechecked after the fourth
calculation and again inside the publication transaction.

Run the preserved load-only gate, a remaining-phase smoke test, and the batch:

```bat
.\.venv\Scripts\python.exe scripts\reproduce_ni_al_mace_baseline.py --load-only
.\.venv\Scripts\python.exe scripts\reproduce_ni_al_mace_baseline.py --phase Al3Ni
.\.venv\Scripts\python.exe scripts\reproduce_ni_al_mace_baseline.py --all-remaining
```

Without `--overwrite`, every existing target is listed and nothing is
published. After inspection, intentional replacement is limited to Step 6B.3
outputs:

```bat
.\.venv\Scripts\python.exe scripts\reproduce_ni_al_mace_baseline.py --all-remaining --overwrite
```

All batch artifacts are staged and verified before transactional publication.
If publication fails, the prior complete target set is restored; incomplete
rollback retains a reported recovery directory instead of discarding the only
backup.

### Verified Results and Outputs

| Phase | Reproduced total energy (eV) | Energy difference (eV) | Maximum-force difference (eV/angstrom) | Result |
|---|---:|---:|---:|---|
| Al3Ni | -74.695710878699174 | 0 | 0 | PASS |
| Al3Ni2 | -25.78055209010038 | 0 | 0 | PASS |
| Al3Ni5 | -44.56807607688442 | 0 | 0 | PASS |
| AlNi3 | -22.836292364226455 | 0 | 0 | PASS |

All 18 comparison entries per phase passed: the 17 numerical comparisons had
zero absolute difference, and the exact material-ID comparison passed. Four
phases completed, zero failed, all structural and source checks passed, and the
overall Step 6B.3 status is PASS. This establishes computational reproducibility
only; it is not evidence of agreement with DFT or experiment and the raw MACE
energies are not a physical-stability ranking.

The atomically published outputs are:

```text
results/mace_relaxation/comparison/reports/Al3Ni_step6b3_baseline_reproduction.txt
results/mace_relaxation/comparison/reports/Al3Ni2_step6b3_baseline_reproduction.txt
results/mace_relaxation/comparison/reports/Al3Ni5_step6b3_baseline_reproduction.txt
results/mace_relaxation/comparison/reports/AlNi3_step6b3_baseline_reproduction.txt
results/mace_relaxation/comparison/reports/ni_al_step6b3_baseline_reproduction_summary.txt
results/mace_relaxation/comparison/tables/ni_al_step6b3_baseline_reproduction.json
```

No optimizer was imported or created, FIRE was not executed, no relaxation
occurred, no positions or cells changed, and no trajectory or structure was
written.

### Success Criteria

Step 6B.3 succeeds only if the batch requests exactly four non-AlNi phases,
loads one calculator, completes exactly four single points, passes every
identity and numerical comparison, passes every in-memory and source-file
immutability check, leaves the protected AlNi report unchanged, and
transactionally publishes all six Step 6B.3 artifacts. The verified run met
every criterion.

### Next Sub-step

Step 6C.1 — Design and validate the atomic-only relaxation runner without executing relaxation.

<!-- NI_AL_STEP6_C_TO_F_START -->
## Step 6C - Atomic-Only Relaxation

The atomic-only runner starts each phase from its original selected EXTXYZ and uses FIRE with a fixed cell. It records step-0 and per-step energy, forces, stress, volume, and periodic displacement. Al3Ni is the pilot. Cells and volumes are immutable at `atol=1e-12, rtol=0`; convergence requires `max_force <= 0.01 eV/angstrom`, and the displacement safety limit is 2 A. After the pilot passes, the other four independent inputs run sequentially through the same one-load calculator session. Initially converged inputs are recorded as `ALREADY_CONVERGED`.

Implementation: `scripts/run_ni_al_mace_atomic_relaxation.py`, with shared validation/publication helpers in `scripts/step6_utils.py` and settings in `configs/mace_relaxation.json`. Outputs are under `results/mace_relaxation/atomic_only/{structures,trajectories,tables,reports,checkpoints,logs}/`; per-step history CSVs are in `tables/`.

## Step 6D - Full-Cell Relaxation

The full-cell runner independently rereads each original structure. FIRE operates on `FrechetCellFilter`; convergence requires both `max_force <= 0.01 eV/angstrom` and raw ASE `max_abs_stress <= 0.0006241509 eV/angstrom^3`. AlNi is the pilot. After it passes, the other four original inputs run sequentially through that mode's same one-load calculator session. Safety checks cover nonfinite values, identity and PBC preservation, positive cells, a 25% absolute volume-change limit, and a 2 A internal-motion limit.

Implementation: `scripts/run_ni_al_mace_full_cell_relaxation.py`, again using `scripts/step6_utils.py` and `configs/mace_relaxation.json`. Outputs are under `results/mace_relaxation/full_cell/{structures,trajectories,tables,reports,checkpoints,logs}/`; per-step history CSVs are in `tables/`.

## Step 6E - Relaxation Comparison

The no-MACE analyzer (`scripts/analyze_ni_al_mace_relaxation.py`) compares Step 5, fixed-cell, and full-cell values; evaluates symmetry with `symprec=0.001 A` and a 5-degree angle tolerance; and writes the comparison tables, report, and nine history/summary figures under `results/mace_relaxation/comparison/`.

| Phase | Atomic status | Atomic steps | Atomic Delta E (eV) | Full-cell status | Full steps | Full Delta E (eV) | Delta V (%) |
|---|---|---:|---:|---|---:|---:|---:|
| Al3Ni | CONVERGED | 28 | -0.03974671177 | CONVERGED | 40 | -0.1145317471 | 2.7396658 |
| Al3Ni2 | CONVERGED | 10 | -0.001103025753 | CONVERGED | 33 | -0.01827268287 | 2.3826129 |
| AlNi | ALREADY_CONVERGED | 0 | 0 | CONVERGED | 5 | -0.008715574573 | 2.6281714 |
| Al3Ni5 | CONVERGED | 24 | -0.003624474698 | CONVERGED | 34 | -0.07120490603 | 3.2800406 |
| AlNi3 | ALREADY_CONVERGED | 0 | 0 | CONVERGED | 14 | -0.02247959088 | 2.8941496 |

Raw energies are never used to rank different compositions and no formation energies are calculated.

## Step 6F - Step 6 Completion

Actual overall status: **SUCCESS**. Collision protection rejects existing Step 6C-F bundles unless intentional overwrite is selected; resume reuses only complete bundles that pass provenance, hashes, geometry, convergence, and safety validation.

The orchestrator is `scripts/run_step6_pipeline.py`; the authoritative completion report is `results/mace_relaxation/comparison/reports/ni_al_step6_final_report.txt`. The comparison output tree contains `figures/`, `tables/`, and `reports/`.

Commands:

```bat
.\.venv\Scripts\python.exe scripts\run_ni_al_mace_atomic_relaxation.py --validate-only
.\.venv\Scripts\python.exe scripts\run_ni_al_mace_full_cell_relaxation.py --validate-only
.\.venv\Scripts\python.exe scripts\analyze_ni_al_mace_relaxation.py --validate-only
.\.venv\Scripts\python.exe scripts\run_step6_pipeline.py --validate-only
.\.venv\Scripts\python.exe scripts\run_step6_pipeline.py --execute
.\.venv\Scripts\python.exe scripts\run_step6_pipeline.py --execute --resume
```

These are MACE-potential results, not DFT or experimental validation. The exact next stage is:

Step 7 - Calculate consistent pure Al and pure Ni MACE reference states, then calculate MACE-consistent Ni-Al formation energies.

Step 7 is not implemented here.
<!-- NI_AL_STEP6_C_TO_F_END -->

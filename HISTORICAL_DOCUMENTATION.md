# Historical Documentation Archive

This archive preserves the complete pre-audit versions of the repository's two primary Markdown documents. It is retained so that no earlier documentation content is lost. For current results, models, systems, and scope, use [README.md](README.md), [PROJECT_KNOWLEDGE.md](PROJECT_KNOWLEDGE.md), and [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

## Original README.md (verbatim)

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
<!-- README_HISTORICAL_PART_1_END -->
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
<!-- README_HISTORICAL_PART_2_END -->
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
<!-- README_HISTORICAL_PART_3_END -->
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
<!-- README_HISTORICAL_PART_4_END -->
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
<!-- README_HISTORICAL_PART_5_END -->
+
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

<!-- NI_AL_STEP7_START -->
## Step 7 - MACE Elemental References and Formation Energies

Step 7 retrieves the stable FCC pure Al and pure Ni reference structures from Materials Project (structures and provenance only - never DFT energies), relaxes both independently with the exact Step 6 full-cell criteria (FIRE + FrechetCellFilter; max force <= 0.01 eV/angstrom; max |raw ASE stress| <= 0.0006241509 eV/angstrom^3; up to 1000 steps), and defines the chemical potentials `mu_X_MACE = relaxed total energy / atoms`.

Selected structures - Al: mp-134, Ni: mp-23. Materials Project database version: Al=2026.04.13; Ni=2026.04.13.

The MACE-consistent formation energy per atom is

```text
E_f = (E_compound_total - N_Al*mu_Al_MACE - N_Ni*mu_Ni_MACE) / (N_Al + N_Ni)
```

applied with the actual cell composition (validated against the formula-unit route at 1e-12 eV/atom). The primary result uses full-cell relaxed compound and elemental energies; the clearly separated diagnostic uses initial fixed-geometry single points on both sides. Initial and relaxed states are never mixed.

Chemical potentials (this executed run): mu_Al_MACE = -3.709587940 eV/atom; mu_Ni_MACE = -5.732347320 eV/atom.

| Phase | x_Ni | Initial E_f (eV/atom) | Relaxed E_f (eV/atom) | Relaxation effect (eV/atom) | Above envelope (eV/atom) | On envelope |
|---|---:|---:|---:|---:|---:|---|
| Al3Ni | 0.250000 | -0.455126193 | -0.460362379 | -0.005236186 | 0.000000000 | yes |
| Al3Ni2 | 0.400000 | -0.640219089 | -0.641073263 | -0.000854174 | 0.000000000 | yes |
| AlNi | 0.500000 | -0.689259153 | -0.690231034 | -0.000971881 | 0.000000000 | yes |
| Al3Ni5 | 0.625000 | -0.601314792 | -0.606097570 | -0.004782778 | 0.000000000 | yes |
| AlNi3 | 0.750000 | -0.487265381 | -0.488035514 | -0.000770133 | 0.000000000 | yes |

The selected-set lower convex envelope uses only pure Al, the five selected compounds, and pure Ni. It is not the complete Ni-Al convex hull, not Materials Project energy above hull, and not a phase-diagram or experimental-stability claim. Untested compositions may lie below it. Ni is magnetic in DFT descriptions; the structural MACE workflow has no explicit spin input, so the Ni reference is MACE-consistent, not a controlled magnetic DFT reference.

Implementation: `scripts/step7_utils.py`, `scripts/fetch_ni_al_elemental_references.py`, `scripts/run_ni_al_mace_elemental_references.py`, `scripts/calculate_ni_al_mace_formation_energies.py`, and `scripts/run_step7_pipeline.py`, with settings in `configs/mace_formation_energy.json`. Outputs are under `results/mace_elemental_references/` and `results/mace_formation_energy/`; the authoritative report is `results/mace_formation_energy/reports/ni_al_step7_final_report.txt`.

Commands:

```bat
.\.venv\Scripts\python.exe scripts\fetch_ni_al_elemental_references.py --validate-only
.\.venv\Scripts\python.exe scripts\fetch_ni_al_elemental_references.py --fetch
.\.venv\Scripts\python.exe scripts\run_ni_al_mace_elemental_references.py --validate-only
.\.venv\Scripts\python.exe scripts\run_ni_al_mace_elemental_references.py --execute
.\.venv\Scripts\python.exe scripts\calculate_ni_al_mace_formation_energies.py --validate-only
.\.venv\Scripts\python.exe scripts\calculate_ni_al_mace_formation_energies.py --calculate
.\.venv\Scripts\python.exe scripts\run_step7_pipeline.py --validate-only
.\.venv\Scripts\python.exe scripts\run_step7_pipeline.py --execute
.\.venv\Scripts\python.exe scripts\run_step7_pipeline.py --execute --resume
```

Actual overall Step 7 status: **SUCCESS**. These are MACE-consistent results only; no DFT was performed, no MACE-versus-DFT formation-energy comparison was made, and no accuracy or fine-tuning conclusion is drawn. The exact next stage is:

Step 8 - Select and document candidate classical Ni-Al interatomic potentials for the future LAMMPS comparison.

Step 8 is not implemented here.
<!-- NI_AL_STEP7_END -->

<!-- NI_AL_STEP8_START -->
## Step 8 - MACE vs Materials Project DFT Benchmark

Step 8 retrieves the five selected phases by exact material ID from the official Materials Project summary endpoint and benchmarks the Step 7 relaxed MACE formation energies and Step 6 MACE-relaxed structures against the MP processed DFT-derived references. Processed `formation_energy_per_atom` is used because it is Materials Project's recommended, correction-consistent thermodynamic quantity; raw MACE and VASP total energies use incompatible reference scales and are never compared. The signed error is `MACE - MP DFT` in eV/atom.

Material IDs: Al3Ni=mp-622209, Al3Ni2=mp-1057, AlNi=mp-1487, Al3Ni5=mp-16514, AlNi3=mp-2593. Materials Project database version: 2026.04.13.

| Phase | MP DFT E_f (eV/atom) | MACE relaxed E_f (eV/atom) | Signed error (eV/atom) | MP hull (eV/atom) | dV/atom (%) |
|---|---:|---:|---:|---:|---:|
| Al3Ni | -0.418776 | -0.460362 | -0.041587 | 0.000000 | +2.7397 |
| Al3Ni2 | -0.644217 | -0.641073 | +0.003143 | 0.000000 | +2.3826 |
| AlNi | -0.684901 | -0.690231 | -0.005330 | 0.000000 | +2.6282 |
| Al3Ni5 | -0.563251 | -0.606098 | -0.042847 | 0.000000 | +3.2800 |
| AlNi3 | -0.426420 | -0.488036 | -0.061616 | 0.000000 | +2.8941 |

Aggregate (n=5): MAE = 0.030905 eV/atom; RMSE = 0.038471 eV/atom; mean signed error = -0.029647 eV/atom; exact ranking agreement = True; pairwise ordering agreement = 10/10. Volume: mean signed error = +2.7849%; symmetry agreement = 5/5 (symprec 0.001 A, angle tolerance 5 deg).

MP energy above hull is DFT context computed against the full MP Ni-Al entry set; it is not comparable to and was never subtracted from the Step 7 selected-set envelope. Five phases are a small sample: statistics are descriptive, correlations exploratory, and no universal MACE accuracy claim is made.

Implementation: `scripts/step8_utils.py`, `scripts/fetch_ni_al_mp_dft_benchmarks.py`, `scripts/compare_ni_al_mace_vs_mp_dft.py`, and `scripts/run_step8_pipeline.py`, with settings in `configs/mace_dft_benchmark.json`. Outputs are under `results/mace_vs_dft/`; the authoritative report is `results/mace_vs_dft/reports/ni_al_step8_final_report.txt`.

Commands:

```bat
.\.venv\Scripts\python.exe scripts\fetch_ni_al_mp_dft_benchmarks.py --validate-only
.\.venv\Scripts\python.exe scripts\fetch_ni_al_mp_dft_benchmarks.py --fetch
.\.venv\Scripts\python.exe scripts\compare_ni_al_mace_vs_mp_dft.py --validate-only
.\.venv\Scripts\python.exe scripts\compare_ni_al_mace_vs_mp_dft.py --compare
.\.venv\Scripts\python.exe scripts\run_step8_pipeline.py --validate-only
.\.venv\Scripts\python.exe scripts\run_step8_pipeline.py --execute
.\.venv\Scripts\python.exe scripts\run_step8_pipeline.py --execute --resume
```

Actual overall Step 8 status: **SUCCESS**. No DFT was run, no LAMMPS or fine-tuning was implemented, and no automatic fine-tuning decision was made. The exact next stage is:

Step 9 - Select and document candidate classical Ni-Al interatomic potentials and design the LAMMPS comparison.

Step 9 is not implemented here.
<!-- NI_AL_STEP8_END -->

<!-- NI_AL_STEP9_START -->
## Step 9 - Classical Ni-Al Potential Selection

Step 9 selected, retrieved, and validated three documented binary Ni-Al EAM potentials from the NIST Interatomic Potentials Repository (HTTPS-only, redirect-confined, fingerprinted) and designed the Step 10 LAMMPS benchmark. No LAMMPS simulation, MACE calculation, or DFT calculation was executed, and no new scientific energy exists from this step.

| Candidate | Role | Official file | Cutoff (A) | File element order | SHA-256 |
|---|---|---|---:|---|---|
| pun_mishin_2009 | primary | `Mishin-Ni-Al-2009.eam.alloy` | 6.2872 | Ni Al | `e0c4b32cbf05f804...` |
| mishin_2004_ipr2 | secondary | `NiAl_Mishin_2004.eam.alloy` | 6.7249 | Ni Al | `15712c13a4728436...` |
| mishin_2002 | historical_secondary | `NiAl02.eam.alloy` | 5.9541 | Ni Al | `68de13eb1b6682bf...` |

**Primary: `pun_mishin_2009`** (Purja Pun & Mishin 2009, DOI 10.1080/14786430903258184) - binary Ni-Al specific, built on established pure-element descriptions with the cross interaction fitted to B2-NiAl properties and ab initio intermetallic formation energies. Secondary: `mishin_2004_ipr2` (gamma/gamma-prime focus) - only the corrected ipr2 file `NiAl_Mishin_2004.eam.alloy` is accepted because the superseded ipr1 file has non-zero isolated-atom energies, while ipr2 sets F(rho=0)=0. Historical secondary: `mishin_2002` (B2-optimized; documented pure-element weakness).

All files are `eam/alloy` setfl files validated array-by-array (headers, Al+Ni identity, grids, finiteness, exact counts, no trailing content) with byte-identical processed copies under `data/processed/interatomic_potentials/ni_al/`. Planned mapping: atom type 1 = Al, type 2 = Ni via `pair_coeff * * <file> Al Ni` (never per-pair pair_coeff and never pair_style hybrid mixing).

Local LAMMPS availability: **AVAILABLE_AND_EAM_ALLOY_CONFIRMED** - 'C:\Users\A\AppData\Local\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.EXE -h' completed; eam/alloy was listed in the help output. LAMMPS was not installed automatically; absence only affects Step 10 readiness.

The Step 10 design (results/lammps_potential_selection/plans/) specifies: identical starting structures for every potential; two-stage static minimization (fixed-cell, then full-cell via `fix box/relax` at zero pressure); per-potential elemental references with the standard formation-energy equation; force target 0.01 eV/angstrom and stress target 0.0006241509 eV/angstrom^3 = 999.999988 bar (converted from exact SI definitions); and independent convergence verification.

Commands:

```bat
.\.venv\Scripts\python.exe scripts\fetch_ni_al_classical_potentials.py --validate-only
.\.venv\Scripts\python.exe scripts\fetch_ni_al_classical_potentials.py --fetch
.\.venv\Scripts\python.exe scripts\validate_ni_al_classical_potentials.py --validate-only
.\.venv\Scripts\python.exe scripts\design_ni_al_lammps_benchmark.py --validate-only
.\.venv\Scripts\python.exe scripts\design_ni_al_lammps_benchmark.py --design
.\.venv\Scripts\python.exe scripts\run_step9_pipeline.py --validate-only
.\.venv\Scripts\python.exe scripts\run_step9_pipeline.py --execute
.\.venv\Scripts\python.exe scripts\run_step9_pipeline.py --execute --resume
```

Actual overall Step 9 status: **SUCCESS**. The exact next stage is:

Step 10 - Execute the designed LAMMPS benchmark: convert structures, relax with each validated classical potential, and compare formation energies and structures against MACE and the Materials Project DFT references.

Step 10 is not implemented here.
<!-- NI_AL_STEP9_END -->

<!-- NI_AL_STEP10_START -->
## Step 10 - LAMMPS Classical-Potential Benchmark

Step 10 executed the Step 9-designed static benchmark: the three validated NIST EAM/alloy potentials each processed independent copies of the same seven original selected structures (pure Al, pure Ni, five compounds) through an initial `run 0`, a fixed-cell CG minimization, and a full-cell `fix box/relax tri 0.0` minimization (63 states total; sequential; no dynamics, velocities, or thermostats). Convergence was verified independently: max force <= 0.01 eV/angstrom and max |pressure component| <= 999.999988 bar (= 0.0006241509 eV/angstrom^3; stress = -pressure/1.602176634e6). Formation energies use each potential's own relaxed pure-element references in the matching state; no cross-potential, MACE, or MP elemental reference was ever mixed.

| Method | MAE (eV/atom) | RMSE (eV/atom) | Mean signed (eV/atom) | Ranking exact | Volume MAE (%) | Symmetry |
<!-- README_HISTORICAL_PART_6_END -->
|---|---:|---:|---:|---|---:|---|
| MACE-MP-0 Small | 0.030905 | 0.038471 | -0.029647 | True | 2.7849 | 5/5 |
| Pun-Mishin 2009 EAM | 0.117265 | 0.153381 | +0.106242 | False | 1.8576 | 5/5 |
| Mishin 2004 EAM (ipr2) | 0.126620 | 0.159870 | +0.118100 | False | 2.6759 | 5/5 |
| Mishin 2002 EAM | 0.149494 | 0.166682 | +0.149494 | False | 1.6380 | 5/5 |

Best method by formation-energy MAE: **MACE-MP-0 Small**. Full per-phase values, envelopes, runtime, and structural details are under `results/lammps_benchmark/`; the authoritative report is `results/lammps_benchmark/reports/ni_al_step10_final_report.txt`.

Commands:

```bat
.\.venv\Scripts\python.exe scripts\run_step10_pipeline.py --validate-only
.\.venv\Scripts\python.exe scripts\run_step10_pipeline.py --execute
.\.venv\Scripts\python.exe scripts\run_step10_pipeline.py --execute --resume
```

Actual overall Step 10 status: **SUCCESS**. These are static bulk-phase results only; they do not prove accuracy for defects, surfaces, interfaces, finite temperature, or dynamics, and no potential is universally best. The exact next stage is:

Step 11 - Design and generate a DFT reference dataset for Ni-Al, beginning with convergence tests and pilot calculations.

Step 11 is not implemented here.
<!-- NI_AL_STEP10_END -->
<!-- README_HISTORICAL_PART_7_END -->
+
## Original PROJECT_KNOWLEDGE.md (verbatim)

# PROJECT_KNOWLEDGE — Ni–Al MACE Research Project

> **Project path:** `D:\Materials_Research\NiAl_MACE`  
> **Current status:** Step 6C–F completed successfully; all ten relaxation results converged safely and the comparison/reporting bundle is validated
> **Next planned step:** Step 7 - Calculate consistent pure Al and pure Ni MACE reference states, then calculate MACE-consistent Ni-Al formation energies.

---

## 1. Purpose of this file

This file is the personal knowledge guide for the project. It explains:

- why each stage exists;
- how the code workflow works;
- what every important file means;
- what was learned scientifically;
- what remains to be done.

It does not replace `README.md`.

- `README.md` explains how to run the project.
- `PROJECT_KNOWLEDGE.md` explains how to understand and discuss the project.

Update this file after every completed stage.

---

## 2. Final research direction

The project studies the pretrained universal machine-learning interatomic potential:

`MACE-MP-0 Small`

on these Ni–Al intermetallic phases:

1. `Al3Ni`
2. `Al3Ni2`
3. `AlNi`
4. `Al3Ni5`
5. `AlNi3`

The long-term plan is to:

1. apply pretrained MACE directly to the selected phases;
2. calculate energies, forces, stresses, and relaxed structures;
3. calculate consistent formation energies;
4. compare MACE with selected Ni–Al classical potentials in LAMMPS;
5. identify where the models agree or differ;
6. decide whether project-specific fine-tuning is scientifically justified.

---

## 3. Important scope decision

The project does not train separate models for pure Al and pure Ni.

MACE-MP-0 is already pretrained and supports both elements.

Pure Al and pure Ni may be calculated later as reference systems for:

- formation energy;
- mixing energy;
- elemental cohesive properties;
- phase comparison.

Therefore, pure Al and Ni are future reference calculations, not separate training stages.

---

## 4. Main research questions

The broad question is:

> How well does pretrained MACE-MP-0 describe selected Ni–Al intermetallic phases, and how does its behavior compare with selected classical Ni–Al potentials?

Supporting questions include:

- Can MACE evaluate all five structures without additional training?
- Which phases have large residual forces at Materials Project geometries?
- How much do atoms and cells change after MACE relaxation?
- How do MACE formation energies compare with consistent references?
- Does model performance differ between Al-rich and Ni-rich phases?
- Are the largest differences in energy, force, stress, volume, or structure?
- Is fine-tuning necessary?

---

## 5. Project map

```text
Step 1 — Project and Python environment
        ↓
Step 2 — ASE structure test
        ↓
Step 3 — MACE baseline test on aluminum
        ↓
Step 4 — Ni–Al structure acquisition from Materials Project
        ↓
Step 5 — MACE zero-shot single-point evaluation
        ↓
Step 6 — Controlled geometry relaxation
        ↓
Step 7 — Pure Al/Ni references and formation energies
        ↓
Step 8 — Selection of Ni–Al LAMMPS potentials
        ↓
Step 9 — Consistent calculations with each potential
        ↓
Step 10 — Comparison and error analysis
        ↓
Step 11 — Optional fine-tuning if justified
        ↓
Step 12 — Final figures, discussion, and research writing
```

---

## 6. Current status

### Completed

- Step 1: Environment and repository setup
- Step 2: ASE test
- Step 3: MACE baseline on aluminum
- Step 4: Ni–Al structure acquisition
- Step 5: MACE zero-shot evaluation
- Step 6A: Relaxation configuration and validation
- Step 6B.1: Isolated MACE model-loading gate
- Step 6B.2: AlNi pilot baseline reproduction
- Step 6B.3: Four-phase remaining baseline reproduction
- Step 6C: Atomic-only relaxation
- Step 6D: Full-cell relaxation
- Step 6E: Comparison and symmetry analysis
- Step 6F: Final reporting and documentation

### Not yet completed

- MACE formation-energy calculation;
- pure Al and Ni reference calculations for formation energy;
- LAMMPS comparison;
- EAM or MEAM calculations;
- molecular dynamics;
- fine-tuning;
- final accuracy conclusions;
- final research paper.

---

## 7. Folder logic

```text
NiAl_MACE/
│
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── environment/
├── results/
├── scripts/
├── .env
├── .env.example
├── .gitignore
├── README.md
└── PROJECT_KNOWLEDGE.md
```

### `configs/`

Stores JSON settings such as:

- phase names;
- model name and size;
- device;
- precision;
- input and output directories;
- phase order.

Keeping settings outside Python makes the workflow easier to review and reproduce.

### `scripts/`

Contains executable Python programs. Important current scripts are:

- `fetch_ni_al_structures.py`
- `evaluate_ni_al_mace_zero_shot.py`
- `validate_ni_al_mace_relaxation.py`
- `reproduce_ni_al_mace_baseline.py`

### `data/raw/`

Stores structures as downloaded from the original source.

Raw data should normally not be edited manually.

### `data/processed/`

Stores selected and organized inputs for calculations.

The selected structures are under:
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_1_END -->
+
```text
data/processed/ni_al_structures/selected/
```

### `results/`

Stores calculation outputs. Step 5 outputs are under:

```text
results/mace_zero_shot/
```

### `environment/`

Stores package-version snapshots, such as:

```text
environment/requirements_step4.txt
environment/requirements_step5.txt
```

---

## 8. Python virtual environment

The project uses:

```text
D:\Materials_Research\NiAl_MACE\.venv
```

Activate it with:

```cmd
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\activate.bat
```

The terminal should show:

```text
(.venv)
```

The virtual environment isolates the project packages from global Python and reduces version conflicts.

Important packages include:

- PyTorch;
- ASE;
- MACE;
- e3nn;
- pymatgen;
- mp-api.

---

## 9. Step 1 — Environment setup

### Purpose

Prepare a reproducible Python and Git project.

### Main outcomes

- project folder created;
- Git repository initialized;
- Python 3.11 virtual environment created;
- required packages installed;
- project folder structure created.

### Success criteria

- `.venv` activates;
- Python runs from `.venv`;
- required packages import successfully;
- secrets and `.venv` are ignored by Git.

---

## 10. Step 2 — ASE test

ASE means:

`Atomic Simulation Environment`

ASE is used to:

- build or read structures;
- attach calculators;
- request energy, forces, and stress;
- write structure files;
- run future geometry optimizations.

General workflow:

```text
Read or create atoms
        ↓
Attach calculator
        ↓
Request energy, forces, or stress
        ↓
Calculator evaluates structure
        ↓
ASE returns results
```

---

## 11. Step 3 — MACE aluminum baseline

### Purpose

Confirm that MACE loads and responds physically to a structural change.

This was not training or fine-tuning.

### Model

```text
MACE-MP-0 Small
```

### What was tested

1. a symmetric FCC aluminum structure;
2. the same structure after slightly moving one atom.

### Observation

For the symmetric structure:

- forces were nearly zero;
- the energy was lower.

After moving one atom:

- energy increased;
- nonzero forces appeared.

### Meaning

The test confirmed that:

- ASE and MACE communicate;
- MACE calculates energy, forces, and stress;
- the model responds to changes in atomic geometry.

It did not prove full accuracy for Ni–Al.

---

## 12. Step 4 — Materials Project structure acquisition

### Purpose

Download, validate, organize, and select structures for:

- `Al3Ni`
- `Al3Ni2`
- `AlNi`
- `Al3Ni5`
- `AlNi3`

### Validation command

```cmd
python scripts\fetch_ni_al_structures.py --validate-only
```

This checked the configuration without using the API or downloading data.

### Full command

```cmd
python scripts\fetch_ni_al_structures.py
```

### Final result

```text
Requested phases: 5
Completed phases: 5
Failed phases: 0
Total candidates saved: 9
```

---

## 13. Selected Materials Project structures

| Phase | Materials Project ID | Space group | Atoms | Energy above hull |
|---|---|---|---:|---:|
| Al3Ni | `mp-622209` | Pnma (62) | 16 | 0 eV/atom |
| Al3Ni2 | `mp-1057` | P-3m1 (164) | 5 | 0 eV/atom |
| AlNi | `mp-1487` | Pm-3m (221) | 2 | 0 eV/atom |
| Al3Ni5 | `mp-16514` | Cmmm (65) | 8 | 0 eV/atom |
| AlNi3 | `mp-2593` | Pm-3m (221) | 4 | 0 eV/atom |
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_2_END -->
+
All selected structures lie on the Materials Project convex hull.

A careful statement is:

> These phases are thermodynamically stable according to the Materials Project calculations and phase-diagram references used.

This does not prove stability under every temperature, pressure, or experimental condition.

---

## 14. Formation energy and energy above hull

### Energy above hull

A value of:

```text
0 eV/atom
```

means the phase lies on the calculated convex hull.

### Formation energy

Negative formation energy generally indicates that the compound is lower in energy than the elemental reference states under the calculation convention used.

However, Materials Project formation energy must not be directly compared with raw MACE energy because the methods and reference energies may differ.

A MACE formation energy must later use MACE energies for:

- the compound;
- pure Al;
- pure Ni.

---

## 15. Step 4 configuration

### File

```text
configs/ni_al_phases.json
```

### Purpose

Define the five phases, their formulas, aliases, and order.

This avoids hard-coding the same phase list into every script.

---

## 16. Step 4 main script

### File

```text
scripts/fetch_ni_al_structures.py
```

### Workflow

```text
Locate project root
        ↓
Load JSON configuration
        ↓
Validate phases
        ↓
Read API key securely
        ↓
Query Materials Project
        ↓
Retain exact-composition candidates
        ↓
Rank candidates deterministically
        ↓
Save raw CIF, EXTXYZ, and metadata
        ↓
Copy selected structures
        ↓
Write CSV and JSON manifests
        ↓
Print summary
```

The script did not choose the first API result randomly.

It used a fixed selection rule based mainly on:

1. energy above hull;
2. stability;
3. formation energy when needed;
4. deterministic material-ID ordering.

---

## 17. Important Step 4 files

### `.env`

Stores the real API key locally:

```text
MP_API_KEY=private_key
```

It must not be committed.

### `.env.example`

Shows the required variable without revealing the key:

```text
MP_API_KEY=replace_with_your_materials_project_api_key
```

### `.gitignore`

Protects:

- `.env`;
- `.venv`;
- local cache files;
- other private or generated files.

### Raw files

Each candidate may contain:

```text
structure.cif
structure.extxyz
metadata.json
```

### Selected files

Under:

```text
data/processed/ni_al_structures/selected/
```

Examples:

```text
AlNi.cif
AlNi.extxyz
AlNi.metadata.json
```

### Manifests

```text
data/processed/ni_al_structures/ni_al_phase_manifest.csv
data/processed/ni_al_structures/ni_al_phase_manifest.json
```

These record all candidates and which one was selected.

---

## 18. CIF, EXTXYZ, and metadata

### CIF

`Crystallographic Information File`

Stores:

- lattice lengths and angles;
- space-group data;
- chemical species;
- atomic coordinates.

### EXTXYZ

`Extended XYZ`

Can store:

- atomic positions;
- cell vectors;
- periodic boundaries;
- energy;
- forces;
- stress;
- additional metadata.

EXTXYZ is convenient for ASE and MACE.

### Metadata JSON

Records provenance such as:

- phase key;
- material ID;
- formula;
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_3_END -->
- number of sites;
- space group;
- energy above hull;
- formation energy;
- acquisition time;
- selected status;
- source paths.

---

## 19. Step 5 — Zero-shot single-point evaluation

### Purpose

Test whether pretrained MACE-MP-0 Small can evaluate all five selected structures directly.

### Zero-shot means

The pretrained model is applied without project-specific training or fine-tuning.

### Single-point means

The model evaluates the current geometry once.

No atoms or cell vectors are changed.

### Step 5 did not perform

- relaxation;
- molecular dynamics;
- LAMMPS;
- EAM;
- MEAM;
- training;
- fine-tuning;
- formation-energy calculation.

---

## 20. Step 5 configuration

### File

```text
configs/mace_zero_shot.json
```

### Main settings

```text
Model: MACE-MP-0 Small
Device: CPU
Precision: float64
Dispersion: false
Input: data/processed/ni_al_structures/selected
Output: results/mace_zero_shot
```

`float64` was chosen for a static scientific baseline because it provides higher numerical precision than `float32`.

---

## 21. Step 5 main script

### File

```text
scripts/evaluate_ni_al_mace_zero_shot.py
```

### Workflow

```text
Locate repository root
        ↓
Load configuration
        ↓
Validate phases and inputs
        ↓
Read EXTXYZ and metadata
        ↓
Check formula, periodicity, volume, and atoms
        ↓
Load MACE once
        ↓
Attach calculator
        ↓
Calculate energy, forces, and stress
        ↓
Calculate force statistics
        ↓
Save annotated EXTXYZ
        ↓
Write CSV, JSON, and text report
        ↓
Print final summary
```

The model is loaded once to avoid repeated loading overhead.

---

## 22. Step 5 validation

Command:

```cmd
python scripts\evaluate_ni_al_mace_zero_shot.py --validate-only
```

Result:

```text
Validated phases: 5
Failed phases: 0
The MACE model was not loaded and no calculation was run.
Step 5 validation succeeded.
```

Validation confirmed that:

- configuration exists;
- all five structures exist;
- metadata exists;
- formulas match;
- periodicity is valid;
- cell volumes are positive;
- atom counts are valid.

---

## 23. Output collision protection

When output files already exist, the script reports:

```text
OutputCollisionError
```

This prevents accidental replacement of previous results.

Intentional replacement uses:

```cmd
python scripts\evaluate_ni_al_mace_zero_shot.py --overwrite
```

This is a safety feature, not a MACE calculation failure.

---

## 24. Step 5 final result

```text
Requested phases: 5
Completed phases: 5
Failed phases: 0
Overall status: SUCCESS
```

All five structures produced finite MACE energies, forces, and stresses.

---

## 25. Step 5 numerical summary

| Phase | Atoms | Energy per atom (eV/atom) | Maximum force (eV/Å) | Volume per atom (Å³/atom) |
|---|---:|---:|---:|---:|
| Al3Ni | 16 | -4.66848192992 | 0.212792948909 | 14.2844210793 |
| Al3Ni2 | 5 | -5.15611041802 | 0.102594155532 | 13.4870121819 |
| AlNi | 2 | -5.40684087649 | 4.88879156031e-08 | 11.6932860591 |
| Al3Ni5 | 8 | -5.57100950961 | 0.160800415761 | 11.2965082264 |
| AlNi3 | 4 | -5.70907309106 | 5.97849521949e-08 | 10.9319635963 |

---

## 26. Interpretation of Step 5 forces

### AlNi and AlNi3

Their maximum force magnitudes are nearly zero:

```text
AlNi  ≈ 4.89 × 10^-8 eV/Å
AlNi3 ≈ 5.98 × 10^-8 eV/Å
```

This indicates that the Materials Project geometries are also very close to stationary points on the MACE energy surface.

It does not by itself prove complete model accuracy.

### Al3Ni, Al3Ni2, and Al3Ni5

Their residual forces are larger:

```text
Al3Ni  ≈ 0.213 eV/Å
Al3Ni2 ≈ 0.103 eV/Å
Al3Ni5 ≈ 0.161 eV/Å
```
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_4_END -->
+
This indicates that MACE would likely move some atoms during relaxation.

The correct interpretation is:

> A geometry optimized with DFT does not have to be a stationary point on the MACE potential-energy surface.

Nonzero forces do not automatically mean the structure or model is wrong.

---

## 27. Individual forces and total force

Individual atoms can have nonzero forces while the total vector sum is almost zero.

Opposing atomic forces can cancel.

Therefore:

- total force near zero is a useful consistency check;
- it does not mean every atom is relaxed;
- maximum atomic force is more useful for relaxation convergence.

---

## 28. Stress interpretation

ASE reports stress in Voigt order:

```text
xx, yy, zz, yz, xz, xy
```

The report uses:

```text
eV/Å³
```

The negative diagonal stresses suggest that the Materials Project cells may not be at the preferred MACE cell size.

This motivates controlled cell relaxation later.

---

## 29. Why raw MACE energy is not formation energy

A value such as:

```text
AlNi3 energy per atom = -5.70907309106 eV/atom
```

is a raw model energy.

It cannot be used alone to rank the stability of phases with different compositions.

Formation energy needs consistent elemental reference energies from the same model and calculation convention.

---

## 30. Step 5 output files

### Annotated structures

```text
results/mace_zero_shot/structures/
```

Contains one annotated EXTXYZ for each phase.

### CSV table

```text
results/mace_zero_shot/tables/ni_al_mace_zero_shot.csv
```

Useful for Excel, pandas, and plotting.

### JSON table

```text
results/mace_zero_shot/tables/ni_al_mace_zero_shot.json
```

Useful for future scripts and structured processing.

### Text report

```text
results/mace_zero_shot/reports/ni_al_mace_zero_shot.txt
```

Contains model settings, phase results, interpretation limits, and final status.

---

## 31. Warnings observed

### `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`

A PyTorch loading warning. It did not stop the calculation.

### `cuequivariance` not available

Optional acceleration is not installed.

MACE still runs; the main effect is performance.

Warnings should be documented and checked, but not automatically treated as errors.

---

## 32. Important commands

Activate environment:

```cmd
cd /d D:\Materials_Research\NiAl_MACE
.venv\Scripts\activate.bat
```

Validate Step 4:

```cmd
python scripts\fetch_ni_al_structures.py --validate-only
```

Run Step 4:

```cmd
python scripts\fetch_ni_al_structures.py
```

Validate Step 5:

```cmd
python scripts\evaluate_ni_al_mace_zero_shot.py --validate-only
```

Run Step 5:

```cmd
python scripts\evaluate_ni_al_mace_zero_shot.py
```

Overwrite Step 5 results intentionally:

```cmd
python scripts\evaluate_ni_al_mace_zero_shot.py --overwrite
```

Display Step 5 report:

```cmd
type results\mace_zero_shot\reports\ni_al_mace_zero_shot.txt
```

Check Git:

```cmd
git status
git diff
```

Check active Python:

```cmd
where python
python --version
```

---

## 33. Git and secret safety

Do not commit:

- `.env`;
- API keys;
- `.venv`;
- credentials;
- temporary caches.

Normally safe to commit:

- `.env.example`;
- scripts;
- configurations;
- README;
- this knowledge file;
- manifests;
- reproducible results;
- environment snapshots.

Always inspect:

```cmd
git status
git diff
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_5_END -->
```

before committing.

---

## 34. Validation versus calculation

### Validation-only mode

Checks workflow readiness:

- paths;
- configuration;
- formulas;
- metadata;
- periodicity;
- cell validity.

It does not necessarily load MACE or calculate properties.

### Full calculation mode

- loads the model;
- evaluates the structures;
- creates scientific outputs.

Validation success does not mean that the full calculation has run.

---

## 35. Single-point versus relaxation

### Single-point

- geometry stays fixed;
- energy, forces, and stress are measured once.

### Relaxation

- atoms move iteratively;
- the cell may also change;
- forces and stress guide the optimization;
- the run stops when convergence criteria are satisfied.

Step 5 was single-point only.

---

## 36. Planned Step 6 — Controlled geometry relaxation

### Purpose

Determine how each Ni–Al phase changes on the MACE potential-energy surface.

### Recommended sequence

#### Test A — Atomic-only relaxation

- keep the cell fixed;
- move atomic positions only;
- evaluate force convergence.

#### Test B — Full-cell relaxation

- allow atoms to move;
- allow cell size and shape to change;
- evaluate force and stress convergence.

Separating these tests helps determine whether the mismatch is mainly caused by:

- internal atomic coordinates;
- cell parameters;
- both.

### Important outputs

For each phase:

- initial and final energy;
- energy change;
- initial and final maximum force;
- optimization step count;
- initial and final volume;
- percentage volume change;
- atomic displacement statistics;
- initial and final stress;
- convergence status;
- relaxed EXTXYZ;
- report.

---

## 37. Questions for Step 6

- Which phase requires the largest atomic movement?
- Which phase has the largest energy decrease?
- Do AlNi and AlNi3 remain almost unchanged?
- Does Al3Ni show the largest relaxation?
- Does the cell shrink or expand?
- Is symmetry preserved?
- Are final forces below the selected threshold?
- Do all phases converge?
- Does atomic-only relaxation differ strongly from full-cell relaxation?

---

## 38. Future formation-energy step

After relaxation, calculate pure reference structures using the same MACE model and conventions.

Likely references:

- FCC Al;
- FCC Ni.

A consistent formation-energy calculation needs:

- compound energy;
- pure Al energy;
- pure Ni energy;
- matching precision;
- matching relaxation convention.

---

## 39. Future LAMMPS comparison

A fair comparison should control:

- structure;
- composition;
- cell convention;
- relaxation settings;
- force threshold;
- stress threshold;
- units;
- reference structures;
- property definitions.

Each potential should be documented with:

- source;
- publication;
- type;
- supported elements;
- fitting database;
- known limitations.

---

## 40. Fine-tuning decision

Fine-tuning should only be performed if the results justify it.

Possible evidence includes:

- systematic formation-energy error;
- large force error against reference DFT;
- incorrect relaxed structures;
- large volume error;
- poor off-equilibrium behavior;
- repeated failure for certain compositions.

Fine-tuning needs reliable DFT labels:

- energy;
- forces;
- stress.

Materials Project summary metadata alone is not a full force-training dataset.

---

## 41. What has been learned

### Technical

- virtual environments;
- Git project organization;
- JSON configuration;
- Materials Project API;
- CIF and EXTXYZ;
- ASE;
- MACE loading;
- zero-shot calculations;
- CSV, JSON, and text reports;
- validation modes;
- output overwrite protection.

### Scientific

- raw energy versus formation energy;
- energy above hull;
- force and stress;
- zero-shot evaluation;
- DFT geometry versus MACE equilibrium;
- total force versus maximum atomic force;
- the importance of data provenance.

<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_6_END -->
---

## 42. Open questions

1. Why are AlNi and AlNi3 forces nearly zero?
2. Why are forces larger for Al3Ni, Al3Ni2, and Al3Ni5?
3. Is the mismatch mainly atomic or volumetric?
4. Will MACE relaxation preserve the space group?
5. How much will cell volume change?
6. How will MACE formation energies compare with consistent references?
7. Which classical potential gives the closest behavior?
8. Does performance vary with composition?
9. Are errors systematic?
10. Is fine-tuning justified?

---

## 43. How to study a new script

1. Identify the script purpose.
2. List its inputs.
3. List its outputs.
4. Start reading from:

```python
if __name__ == "__main__":
    main()
```

5. Follow `main()` and write the workflow using arrows.
6. For each important function, record:
   - purpose;
   - input;
   - output;
   - possible errors.
7. Connect code to science:
   - where is the model loaded?
   - where is energy calculated?
   - where are forces calculated?
   - where is stress calculated?
   - where are atoms or the cell changed?
   - where are outputs saved?

---

## 44. Script study template

### Script name

`script_name.py`

### Project step

Step X

### Purpose

One clear sentence.

### Inputs

- input files;
- configuration;
- command-line options.

### Outputs

- structure files;
- tables;
- reports.

### Main workflow

```text
Start
  ↓
...
  ↓
Finish
```

### Important functions

#### Function name

- Purpose:
- Input:
- Output:
- Possible errors:

### Scientific meaning

Explain the physical quantity and why it matters.

### Validation command

```cmd
...
```

### Full command

```cmd
...
```

### Success criteria

- ...
- ...

### What I learned

- ...
- ...

---

## 45. Daily log template

### Date

`YYYY-MM-DD`

### Step

Step X

### Goal

...

### Commands

```cmd
...
```

### Files created

- ...

### Files modified

- ...

### Results

...

### Warnings or errors

...

### Interpretation

...

### Questions remaining

...

### Next action

...

---

## 46. Current log — 2026-07-21

### Work completed

- activated the project virtual environment;
- validated Step 4;
- downloaded five Ni–Al phase families;
- saved nine candidates;
- selected five structures;
- validated Step 5;
- reviewed the existing Step 5 report;
- confirmed successful zero-shot evaluation for all five phases.

### Main result

MACE-MP-0 Small successfully produced finite energies, forces, and stresses for all five selected Ni–Al structures without project-specific training.

### Important observation

- AlNi and AlNi3 had forces near zero.
- Al3Ni, Al3Ni2, and Al3Ni5 had larger residual forces.
- Controlled relaxation is needed before stronger conclusions.

### Historical next action after Step 5

At the end of Step 5, the planned next work was Step 6:

- atomic-only relaxation;
- full-cell relaxation;
- convergence and structural-change analysis.

That work is now complete.
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_7_END -->
+
---

## 47. One-paragraph project explanation

This project evaluates the pretrained MACE-MP-0 Small universal
machine-learning interatomic potential on five Ni–Al intermetallic phases
obtained from Materials Project. The workflow established a reproducible Python
and ASE environment, selected and documented one structure per phase, produced
zero-shot fixed-geometry energy/force/stress baselines, and independently
reproduced those baselines without training. Step 6 then completed independent
atomic-only and full-cell relaxations for all five phases, followed by
comparison, symmetry analysis, and validated reporting. All ten results
converged safely. The next action is Step 7: calculate consistent pure Al and
pure Ni MACE reference states, then calculate MACE-consistent Ni–Al formation
energies. LAMMPS comparison remains a later, separately reviewed stage.

---

## 48. Short oral explanation

> We prepared the Python environment, tested ASE and MACE on aluminum, and
> selected one documented Materials Project structure for each of five Ni–Al
> phases. MACE-MP-0 Small then produced fixed-geometry energy, force, and stress
> baselines, and a separately controlled workflow reproduced all stored values
> exactly while proving that the structures and source files were unchanged.
> Independent fixed-cell and full-cell relaxations then converged safely for
> all five phases, with symmetry and structural changes reported separately.
> These are MACE-potential results, not DFT or experimental validation. The
> next step is to establish consistent pure Al and pure Ni MACE reference
> states before calculating formation energies.

---

## Step 6A — Relaxation Design and Validation

Step 6 is divided into small sub-steps so model loading, reproduction of the
initial single-point state, atomic relaxation, cell relaxation, and comparison
can each be reviewed before the next operation is allowed. This reduces the
risk of confusing preparation or baseline checks with newly generated
relaxation results.

Two independent future relaxation modes are defined. `atomic_only` permits
atomic positions to change but fixes the complete cell. `full_cell` permits
atomic positions, cell shape, and cell volume to change while retaining
three-dimensional periodicity. Running them separately from copies of the same
original input will distinguish internal-coordinate effects from cell effects.

Step 6A created:

```text
configs/mace_relaxation.json
scripts/validate_ni_al_mace_relaxation.py
environment/requirements_step6a.txt
results/mace_relaxation/  (empty planned directory tree only)
```

The planned model is MACE-MP-0 small on CPU with float64 precision and no
dispersion. Both modes use FIRE and a force threshold of 0.01 eV/angstrom.
`atomic_only` allows at most 500 steps. `full_cell` allows at most 1000 steps
and uses a stress threshold of 0.0006241509 eV/angstrom^3. These are initial
controlled values that may later receive sensitivity tests.

Future runs must stop on nonfinite values, preserve the selected source files,
retain periodicity, stop if the absolute volume change exceeds 25%, and stop if
an atomic displacement exceeds 2.0 angstrom. Step 6A validated strict JSON,
paths confined to the repository, mode permissions, safety parameters, finite
periodic Al-Ni geometries, reduced compositions, metadata IDs and formulas,
calculator-free ASE inputs, and successful finite Step 5 baseline records and
annotated output paths for all five phases.

No MACE module or model was loaded. No calculator was attached, no energy,
force, or stress was calculated, no optimization ran, and no atomic positions
or cell vectors changed. The empty output directories are preparation, not
scientific results.

Before Step 6B, understand that a successful Step 6A result proves only that
the planned relaxation is internally consistent and ready for the next safety
gate. Step 6B will load MACE once and reproduce the initial Step 5 energy,
force, stress, and volume values before any relaxation is permitted.

---

## Step 6B.1 — MACE Model Loading Test

Step 6B.1 isolates model construction from structure handling and physical
calculation. This separation proves that configuration, imports, and model
construction work before a structure is introduced, so a later calculation
failure is not confused with a model-loading failure.

The new file is:

```text
scripts/reproduce_ni_al_mace_baseline.py
```

It reads the model family, name, value, device, numerical dtype, and dispersion
flag from `configs/mace_relaxation.json`. For the current project these settings
are MACE, MACE-MP-0, small, CPU, float64, and no dispersion. The installed MACE
factory is called exactly once and its result must inherit from the ASE
calculator interface.

No CIF or EXTXYZ file is read, no `Atoms` object is created, and the calculator
is not attached to a structure. No energy, force, stress, or volume is
requested; no optimizer, relaxation, molecular dynamics, LAMMPS, training, or
fine-tuning runs; and no scientific output file is created.

Before Step 6B.2, understand that successful calculator creation confirms only
software and model-loading readiness. It provides no physical result and says
nothing yet about the reproduced AlNi values. Step 6B.2 will introduce only the
AlNi input and reproduce its fixed-geometry single-point properties without
moving atoms or changing the cell.

---

## Step 6B.2 — AlNi Initial Baseline Reproduction

Status: Completed on 2026-07-26. The AlNi identity, Step 5 baseline, numerical
comparisons, structural immutability checks, and source-file fingerprints all
passed. The calculator was loaded once and one fixed-geometry single-point
calculation was performed.

### Why AlNi Is the Pilot Phase

AlNi is used as the first reproduction case because the selected B2 structure
contains only two atoms and has a clear one-to-one Al/Ni ordering. A small,
unambiguous cell makes unintended atom reordering, position changes, or cell
changes easier to detect. Verifying the complete workflow on this pilot limits
the scientific and operational scope before the same fixed-geometry procedure
is extended to the other four phases.

### Reproducibility Is Not Accuracy

Reproducibility asks whether the same model, numerical precision, execution
device, input geometry, and property definitions reproduce the stored Step 5
results within declared numerical tolerances. Accuracy asks a different
question: whether the model agrees with trustworthy reference calculations or
experiments. Step 6B.2 tests reproducibility only and must not be interpreted
as a DFT or experimental accuracy result.

### Fixed-Geometry Single-Point Workflow

The workflow reads the original selected AlNi EXTXYZ, its provenance metadata,
and the unique successful AlNi record in the Step 5 JSON table. It validates
the reduced composition with a pymatgen `Composition`, confirms `mp-1487`,
requires two Al/Ni atoms with full three-dimensional periodicity, and checks
finite positions, cell vectors, and positive volume. The annotated Step 5
AlNi EXTXYZ must exist and preserve the source geometry, but the full-precision
JSON record is the numerical baseline.

MACE-MP-0 Small is loaded exactly once on CPU using float64 precision with
dispersion disabled. The calculator is attached only to a deep in-memory copy
of the source. The copy is evaluated once using ASE requests for potential
energy, atomic forces, and six-component Voigt stress. No geometry setter or
optimization interface is used.

The calculated and derived quantities are:

* total energy and energy per atom;
* all atomic force vectors and their per-atom magnitudes;
* maximum and root-mean-square force magnitude;
* total force vector and total-force norm;
* stress in ASE order `xx, yy, zz, yz, xz, xy`;
* volume and volume per atom.

Every raw and derived numerical value must be finite.

### Immutability Checks

Before attaching MACE, the workflow stores independent copies of positions,
cell vectors, chemical symbols, atomic numbers, atom count, periodic boundary
conditions, and volume. After the three property requests, every item is
checked separately. Positions, cell vectors, and volume use a strict absolute
tolerance with zero relative tolerance. Symbols, atomic numbers, atom ordering,
atom count, and periodic boundary conditions require exact equality.

The selected structure, metadata, Step 5 JSON table, and annotated Step 5
structure are also fingerprinted using content hashes, byte sizes, and
modification times before use and checked again after calculation. These tests
distinguish an in-memory property calculation from an unintended change to a
scientific source file.

### Tolerance Concept

Binary floating-point results can differ by tiny amounts even when a
calculation is computationally reproducible. Each numerical comparison
therefore has a quantity-specific absolute and relative tolerance, and passes
when:

```text
absolute difference <= absolute tolerance
                       + relative tolerance * abs(Step 5 value)
```

Absolute tolerances govern values near zero, such as total-force components
and shear stresses, because their relative differences are not meaningful.
Energy, force, stress, and volume use separate tolerances appropriate to their
units and stored precision. Atom count and material ID must match exactly.
Every comparison records its Step 5 value, reproduced value, absolute
difference, relative difference when meaningful, tolerance, and pass/fail
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_8_END -->
status.

### Deliberately Not Executed

Step 6B.2 does not import or create FIRE or any other optimizer. It does not
move atoms, alter the cell, run a relaxation, write an EXTXYZ structure, create
a trajectory, calculate formation energy, run molecular dynamics, invoke
LAMMPS, evaluate EAM or MEAM, train MACE, or fine-tune MACE. Its only
persistent scientific artifact is the atomic text reproducibility report under
`results/mace_relaxation/comparison/reports/`.

### Next Sub-step

Step 6B.3 — Extend the verified single-point reproduction workflow to the
remaining four Ni-Al phases without allowing any structure or cell changes.

---

## Step 6B.3 — Remaining Ni-Al Baseline Reproduction

Status: Completed successfully on 2026-07-26. The batch processed Al3Ni
(`mp-622209`, 16 atoms), Al3Ni2 (`mp-1057`, 5 atoms), Al3Ni5 (`mp-16514`,
8 atoms), and AlNi3 (`mp-2593`, 4 atoms). Four phases completed, zero failed,
and the overall status is PASS.

### Pilot Versus Remaining-Phase Batch

Step 6B.2 established the method on the two-atom AlNi pilot. Step 6B.3 applies
that already reviewed method to the four remaining, more varied cells. AlNi is
excluded because recalculating a passed pilot would add no validation value
and would create an unnecessary risk of replacing its evidence. The batch
target list cannot contain the Step 6B.2 report, and `--overwrite` is rejected
for an explicit AlNi invocation.

The protected pilot report remained byte-for-byte unchanged:

```text
SHA-256: 53beefe9e502ac925d2d96cd267dcef039d27d882181b4aa6bbafef1239ed6b2
Size: 11873 bytes
Modification time (UTC): 2026-07-26T09:20:34.9389708Z
```

### One Model, Four Independent Inputs

Constructing MACE is more expensive than clearing calculator results, and all
four structures use identical model settings. The batch therefore constructs
one MACE-MP-0 Small calculator on CPU in float64 with dispersion disabled,
then reuses that same object. Reuse does not mean reuse of a calculated
structure: every phase starts from its own validated original `Atoms` object
and a separate deep working copy. After its one property evaluation, the
calculator is detached and reset before the next phase.

The authoritative batch recorded:

```text
Calculator loads: 1
Single-point calculations: 4
```

### Comparison and Immutability Logic

The successful Step 5 JSON record supplies the full-precision reference values.
The workflow compares total energy, energy per atom, maximum and RMS force,
total-force components and norm, all six ASE Voigt stress components, volume,
volume per atom, atom count, and material ID using the Step 6B.2
absolute-plus-relative tolerances. A failure in any required comparison fails
the phase, and any failed phase fails the batch.

Reproduced per-atom force vectors are retained in each text report. They are
not compared numerically because the Step 5 JSON does not store a per-atom
vector array and the annotated EXTXYZ columns are rounded. Treating that
rounded serialization as the numerical source would manufacture differences
that are not present in the authoritative Step 5 result.

For every working copy, positions and cell vectors remain equal within
`atol=1e-12, rtol=0`; volume uses the same tolerance; symbols, ordering,
atomic numbers, atom count, and PBC require exact equality; the calculator is
detached; and the pristine input remains calculator-free. The selected
EXTXYZ, metadata JSON, Step 5 JSON table, and annotated Step 5 EXTXYZ are
fingerprinted by SHA-256, byte size, and modification timestamp. All four
phase source sets are rechecked after the final calculation and during atomic
publication.

### Verified Numerical Results

| Phase | Step 5/reproduced total energy (eV) | Largest difference across 17 numerical comparisons | Identity | Immutability | Sources | Phase |
|---|---:|---:|---|---|---|---|
| Al3Ni | -74.695710878699174 | 0 | PASS | PASS | PASS | PASS |
| Al3Ni2 | -25.78055209010038 | 0 | PASS | PASS | PASS | PASS |
| Al3Ni5 | -44.56807607688442 | 0 | PASS | PASS | PASS | PASS |
| AlNi3 | -22.836292364226455 | 0 | PASS | PASS | PASS | PASS |

The shared Step 5 JSON retained SHA-256
`83658efab2902dcb8113f9562c9adebebd963d697adc6791dc8cd4213c912488`.
No optimizer was imported or created, FIRE did not execute, relaxation did not
occur, positions and cells did not change, and no trajectory or calculated
structure was written.

### Artifacts

The four complete phase reports and combined text report are under:

```text
results/mace_relaxation/comparison/reports/
```

The strict combined JSON table is:

```text
results/mace_relaxation/comparison/tables/ni_al_step6b3_baseline_reproduction.json
```

The reproducible environment snapshot is:

```text
environment/requirements_step6b3.txt
```

### Remaining Limitations

This result proves that the configured implementation reproduces its stored
fixed-geometry results; it does not validate MACE against DFT or experiment.
No geometry was relaxed, no formation energy was calculated, and raw total
energies for cells with different compositions and sizes must not be used to
rank physical stability. Full-precision Step 5 per-atom force vectors are not
available in the JSON. Relaxation convergence, structural changes,
formation-energy references, classical-potential comparisons, and dynamical
behavior all remain future work.

### Historical Next Sub-step After Step 6B.3

At that checkpoint, the next sub-step was Step 6C.1 — design and validate the
atomic-only relaxation runner without executing relaxation. Step 6C–F has now
been completed; the current next step is stated at the top of this file.

<!-- NI_AL_STEP6_KNOWLEDGE_START -->
## Step 6C-F Research-Log Entry (2026-07-26)

Atomic-only and full-cell calculations are independent so internal-coordinate response can be separated from combined cell and atomic response. FIRE follows forces downhill while adapting its integration parameters. A fixed cell isolates atomic motion; FrechetCellFilter adds cell degrees of freedom using generalized cell forces.

Convergence is measured from the final raw atomic forces (`max_force <= 0.01 eV/angstrom`), and for full-cell results also from all six raw ASE stress components (`max_abs_stress <= 0.0006241509 eV/angstrom^3`). Reaching the step limit is `NOT_CONVERGED`, not a failure or a converged result. Periodic displacement uses wrapped fractional differences. Internal displacement maps those differences through the initial cell; total Cartesian displacement also contains cell deformation. Volume, lattice, deformation-gradient, and strain metrics describe that cell response.

Symmetry symbols and numbers are tolerance-dependent and use `symprec=0.001 A`, `angle_tolerance=5 deg`. Safety monitoring rejects nonfinite data, identity/PBC changes, nonpositive cells, internal motion above 2 A, and full-cell volume changes above 25%.

| Phase | Atomic status | Atomic steps | Atomic Delta E (eV) | Full-cell status | Full steps | Full Delta E (eV) | Delta V (%) |
|---|---|---:|---:|---|---:|---:|---:|
| Al3Ni | CONVERGED | 28 | -0.03974671177 | CONVERGED | 40 | -0.1145317471 | 2.7396658 |
| Al3Ni2 | CONVERGED | 10 | -0.001103025753 | CONVERGED | 33 | -0.01827268287 | 2.3826129 |
| AlNi | ALREADY_CONVERGED | 0 | 0 | CONVERGED | 5 | -0.008715574573 | 2.6281714 |
| Al3Ni5 | CONVERGED | 24 | -0.003624474698 | CONVERGED | 34 | -0.07120490603 | 3.2800406 |
| AlNi3 | ALREADY_CONVERGED | 0 | 0 | CONVERGED | 14 | -0.02247959088 | 2.8941496 |

Overall Step 6 status: **SUCCESS**. This establishes behavior on the selected MACE potential-energy surface only; it does not establish DFT or experimental accuracy. Step 7 must still establish consistent pure-element MACE references before any formation energies are computed. Whether those later values agree with reference data, and whether fine-tuning is warranted, remain unanswered.
<!-- NI_AL_STEP6_KNOWLEDGE_END -->

<!-- NI_AL_STEP7_KNOWLEDGE_START -->
## Step 7 Research-Log Entry (2026-07-28)

Pure elemental crystals are required because a formation energy compares a compound against the elements in their reference crystalline states; isolated atoms would measure atomization energy instead, which is a different quantity with much larger magnitudes. Every energy entering the formula must come from the same model, precision, and convergence convention - mixing MACE and DFT energies would make the difference meaningless.

The elemental chemical potential `mu_X_MACE` is the relaxed MACE total energy per atom of the pure crystal. The formation energy per atom subtracts composition-weighted chemical potentials from the compound energy and divides by the total atom count. Formula-unit counting (x, y per Al_x Ni_y) and simulation-cell counting (N_Al, N_Ni per cell) must agree after handling the number of formula units; Step 7 validates the two routes against each other at 1e-12 eV/atom.

The initial-versus-relaxed distinction matters: the initial diagnostic uses fixed DFT geometries on the MACE surface, while the primary result uses MACE-relaxed geometries on both sides. Raw total energies across compositions can never be ranked directly because each composition has a different reference scale. The selected-set envelope is not a complete convex hull: only seven points were considered, and untested compositions may lie below it.

Actual results - mu_Al_MACE = -3.709587940 eV/atom; mu_Ni_MACE = -5.732347320 eV/atom (Al: mp-134, Ni: mp-23; database version Al=2026.04.13; Ni=2026.04.13).

| Phase | x_Ni | Initial E_f (eV/atom) | Relaxed E_f (eV/atom) | Relaxation effect (eV/atom) | Above envelope (eV/atom) | On envelope |
|---|---:|---:|---:|---:|---:|---|
| Al3Ni | 0.250000 | -0.455126193 | -0.460362379 | -0.005236186 | 0.000000000 | yes |
| Al3Ni2 | 0.400000 | -0.640219089 | -0.641073263 | -0.000854174 | 0.000000000 | yes |
| AlNi | 0.500000 | -0.689259153 | -0.690231034 | -0.000971881 | 0.000000000 | yes |
| Al3Ni5 | 0.625000 | -0.601314792 | -0.606097570 | -0.004782778 | 0.000000000 | yes |
| AlNi3 | 0.750000 | -0.487265381 | -0.488035514 | -0.000770133 | 0.000000000 | yes |

Ni magnetic limitation: Ni is magnetic in DFT descriptions; the structural MACE workflow exposes no user-controlled spin input, so the Ni reference is the configured pretrained MACE model's energy for the selected crystal - a MACE-consistent reference, not a controlled magnetic DFT reference. No Ni magnetic moment was invented.

Unanswered questions for Step 8: which classical Ni-Al potentials should enter the LAMMPS comparison; how MACE and classical formation energies, lattice constants, and relaxed structures compare under identical conventions; whether observed MACE-versus-reference differences are systematic; and whether fine-tuning is justified.

Overall Step 7 status: **SUCCESS**.
<!-- NI_AL_STEP7_KNOWLEDGE_END -->

<!-- NI_AL_STEP8_KNOWLEDGE_START -->
## Step 8 Research-Log Entry (2026-07-28)

A MACE formation energy and an MP DFT formation energy are the same physical definition evaluated on two different energy surfaces: each subtracts its own elemental references, so the two are comparable while raw totals are not. Materials Project publishes processed thermodynamic entries (its recommended correction/mixing scheme, recorded per phase via the thermo endpoint), which is why the processed `formation_energy_per_atom` is the benchmark rather than any raw VASP total.

The signed error (MACE - MP DFT) keeps the direction of the bias visible; MAE averages magnitudes and RMSE additionally weights outliers. A systematic bias means the signed errors share one sign rather than scattering around zero. Ranking agreement asks whether both methods order the five phases identically by formation energy - relevant because many alloy conclusions depend on ordering rather than absolute values.

MP energy above hull is computed against every Ni-Al entry in Materials Project, while the Step 7 envelope contains only seven points on the MACE surface; the two answer different questions and were never subtracted. Volume-per-atom is compared directly, while lattice parameters are compared only after both structures pass through the same pymatgen conventional standardization, because primitive and conventional representations would otherwise differ trivially.

Actual Step 8 findings (n=5): MAE = 0.030905 eV/atom; RMSE = 0.038471 eV/atom; mean signed error = -0.029647 eV/atom; all signed errors positive = False; exact ranking agreement = True; pairwise agreement = 10/10; mean signed volume error = +2.7849%; symmetry agreement = 5/5.

| Phase | MP DFT E_f (eV/atom) | MACE relaxed E_f (eV/atom) | Signed error (eV/atom) | MP hull (eV/atom) | dV/atom (%) |
|---|---:|---:|---:|---:|---:|
| Al3Ni | -0.418776 | -0.460362 | -0.041587 | 0.000000 | +2.7397 |
| Al3Ni2 | -0.644217 | -0.641073 | +0.003143 | 0.000000 | +2.3826 |
| AlNi | -0.684901 | -0.690231 | -0.005330 | 0.000000 | +2.6282 |
| Al3Ni5 | -0.563251 | -0.606098 | -0.042847 | 0.000000 | +3.2800 |
| AlNi3 | -0.426420 | -0.488036 | -0.061616 | 0.000000 | +2.8941 |

<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_9_END -->
Ni remains a magnetic element in DFT descriptions while the structural MACE workflow exposes no spin input, so part of the Ni-rich error budget may be magnetic; this is recorded, not resolved. The next research decision (whether fine-tuning is justified) must weigh the formation-energy bias, the single-signed volume error, the preserved or broken ranking, the Ni magnetic limitation, and the five-phase sample size - no undocumented universal threshold decides it.

Overall Step 8 status: **SUCCESS**.
<!-- NI_AL_STEP8_KNOWLEDGE_END -->

<!-- NI_AL_STEP9_KNOWLEDGE_START -->
## Step 9 Research-Log Entry (2026-07-28)

A classical interatomic potential is an explicit analytic/tabulated energy model. In EAM the energy is a sum of pair terms plus an embedding energy F(rho) evaluated at the host electron density each atom sits in; the alloy cross interaction is the fitted Al-Ni pair function plus how each species' density enters the other's embedding. A setfl (`eam/alloy`) file tabulates F(rho), rho(r), and r*phi(r) for every element and pair on shared grids - unlike the older single-element funcfl (`eam`) format, it defines the cross-pair explicitly, which is why separate pure Al and pure Ni files can never be mixed safely: the Al-Ni interaction would be an undefined guess, not physics.

Potential scope matters because a fit reproduces what it was trained on. 2009 (Purja Pun & Mishin) is the broad binary Ni-Al model (B2 properties plus ab initio intermetallic formation energies; interfaces and mechanics), 2004 (Mishin) targets gamma/gamma-prime (Ni3Al), and 2002 (Mishin-Mehl-Papaconstantopoulos) targets B2-NiAl with documented weaker pure-element behavior. That is why 2009 is primary and the others are sensitivity tests. The 2004 ipr1 file is rejected: its isolated-atom energies are non-zero, so bulk energies are correct but are not cohesive energies; ipr2 sets F(rho=0)=0. Our formation energies always use relaxed bulk elemental references, so each potential needs its own Al and Ni references and raw totals can never be compared across potentials - every model has its own arbitrary energy zero.

LAMMPS is the engine that reads the potential file and evaluates it; the file is data, not code. Static minimization follows forces downhill to a zero-temperature local minimum (the analogue of Step 6's FIRE relaxations), while molecular dynamics integrates finite-temperature motion - Step 10's primary benchmark is static minimization only.

Actual Step 9 selection: primary pun_mishin_2009, secondary mishin_2004_ipr2, historical secondary mishin_2002; all three official NIST files validated array-complete with recorded SHA-256; local LAMMPS status AVAILABLE_AND_EAM_ALLOY_CONFIRMED. Overall Step 9 status: **SUCCESS**.

Unanswered questions for Step 10: how large are each potential's formation-energy and volume errors against MP DFT and against MACE under identical structures and convergence targets; does the 2004 model's gamma-prime focus degrade Al-rich phases; how strong is the 2002 pure-element weakness in practice; and how do classical costs compare with MACE.
<!-- NI_AL_STEP9_KNOWLEDGE_END -->

<!-- NI_AL_STEP10_KNOWLEDGE_START -->
## Step 10 Research-Log Entry (2026-07-28)

LAMMPS is a simulation engine: it reads a structure and an interatomic-potential file and evaluates/minimizes the model the file defines - LAMMPS itself is not the physical model. Static energy minimization walks downhill to a zero-temperature local minimum (here conjugate gradient with a quadratic line search); molecular dynamics would integrate finite-temperature motion and was not used. Fixed-cell minimization moves only atoms; `fix box/relax tri 0.0` adds all six cell degrees of freedom at zero target pressure. LAMMPS reports pressure (positive = compression) while ASE stress is positive in tension: stress_eV_per_A3 = -pressure_bar/1.602176634e6; convergence checks use absolute values so the sign convention cannot change a decision.

Every potential defines its own energy zero, so each needs its own relaxed pure Al and pure Ni references, and raw totals can never be compared across potentials or compositions. The initial / fixed-cell / full-cell formation energies separate the chemical prediction from the atomic and cell relaxation contributions, always with same-state, same-potential references.

Actual Step 10 findings (n=5 compounds; errors vs MP processed DFT):

| Method | MAE (eV/atom) | RMSE (eV/atom) | Mean signed (eV/atom) | Ranking exact | Volume MAE (%) | Symmetry |
|---|---:|---:|---:|---|---:|---|
| MACE-MP-0 Small | 0.030905 | 0.038471 | -0.029647 | True | 2.7849 | 5/5 |
| Pun-Mishin 2009 EAM | 0.117265 | 0.153381 | +0.106242 | False | 1.8576 | 5/5 |
| Mishin 2004 EAM (ipr2) | 0.126620 | 0.159870 | +0.118100 | False | 2.6759 | 5/5 |
| Mishin 2002 EAM | 0.149494 | 0.166682 | +0.149494 | False | 1.6380 | 5/5 |

The combined Step 8 and Step 10 evidence feeds the Step 11 decision: a DFT reference dataset (convergence tests first) is the justified next investigation, with any MACE fine-tuning deferred to Step 12 after that dataset is validated.

Overall Step 10 status: **SUCCESS**.
<!-- NI_AL_STEP10_KNOWLEDGE_END -->
<!-- PROJECT_KNOWLEDGE_HISTORICAL_PART_10_END -->

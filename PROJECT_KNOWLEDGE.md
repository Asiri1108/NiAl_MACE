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

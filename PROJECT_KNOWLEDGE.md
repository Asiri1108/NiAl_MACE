# PROJECT_KNOWLEDGE — Ni–Al MACE Research Project

> **Project path:** `D:\Materials_Research\NiAl_MACE`  
> **Current status:** Step 5 completed successfully  
> **Next planned step:** Controlled geometry relaxation using MACE

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

### Not yet completed

- geometry relaxation;
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

### Next action

Prepare Step 6:

- atomic-only relaxation;
- full-cell relaxation;
- convergence and structural-change analysis.

---

## 47. One-paragraph project explanation

This project evaluates the pretrained MACE-MP-0 Small universal machine-learning interatomic potential on five Ni–Al intermetallic phases obtained from Materials Project. The workflow established a reproducible Python and ASE environment, tested MACE on aluminum, downloaded and documented candidate Ni–Al structures, selected one stable structure for each phase, and performed zero-shot single-point MACE evaluations without training or relaxation. All five phases were evaluated successfully. The next stage will perform controlled geometry relaxations to determine how the atomic positions and unit cells change on the MACE potential-energy surface before calculating consistent formation energies and comparing MACE with selected Ni–Al classical potentials in LAMMPS.

---

## 48. Short oral explanation

> We first prepared the Python environment and tested ASE and MACE on aluminum. Then we downloaded five Ni–Al intermetallic phases from Materials Project and selected one documented structure for each phase. After that, we used pretrained MACE-MP-0 Small in zero-shot single-point mode to calculate energy, forces, and stress without training or relaxation. All five phases were evaluated successfully. The next step is controlled geometry relaxation before formation-energy calculations and comparison with other Ni–Al potentials.

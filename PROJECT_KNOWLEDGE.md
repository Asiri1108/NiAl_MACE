# Project Knowledge: Ni-Al Interatomic-Potential Benchmark

## 1. What this project actually did

This repository evaluated a pretrained machine-learning interatomic potential and three published classical Ni-Al potentials on five selected bulk intermetallic crystal structures. The results cover zero-shot single points, geometry relaxation, formation energies, structural comparison, and a static LAMMPS EAM benchmark.

The work completed through Step 10. It did **not** train or fine-tune MACE, run a new DFT calculation, run molecular dynamics, or analyse trajectories in OVITO. Earlier planning language in the repository is superseded by this record and by [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

## 2. Research question and scope

The practical question was:

> How does a pretrained MACE-MP-0 Small model compare with three published Ni-Al EAM potentials for static formation energies and equilibrium structures of five selected Ni-Al intermetallic phases?

The scope is deliberately narrow:

- Static, zero-temperature bulk structures only.
- Five selected intermetallic compounds plus pure Al and Ni reference cells.
- Materials Project processed DFT-derived values as a benchmark, not experimental truth.
- Independent elemental references for each potential family before formation energies are calculated.

It is not a phase-diagram calculation, a validation of finite-temperature properties, or proof of behaviour for defects, surfaces, interfaces, deformation, or long molecular-dynamics runs.

## 3. Models used

| Category | Model | Status in this project |
|---|---|---|
| ML potential | **MACE-MP-0 Small** | Main model. Pretrained and used zero-shot only; no Ni-Al fine-tuning. |
| Classical potential | **Purja Pun-Mishin 2009 Ni-Al EAM** | LAMMPS `eam/alloy` primary EAM baseline. |
| Classical potential | **Mishin 2004 Ni-Al EAM (ipr2)** | LAMMPS `eam/alloy` secondary baseline. The superseded `ipr1` implementation was excluded. |
| Classical potential | **Mishin-Mehl-Papaconstantopoulos 2002 B2-NiAl EAM** | LAMMPS `eam/alloy` historical baseline. |
| Reference dataset | **Materials Project processed DFT-derived thermodynamics** | Benchmark data only, database version `2026.04.13`. |

The key model answer is therefore: **MACE-MP-0 Small was used, as a pretrained CPU zero-shot model. It was not fine-tuned.**

## 4. Systems used

| System | MP ID | Space group | Atoms | Role |
|---|---|---|---:|---|
| Al | `mp-134` | Fm-3m (225) | 1 | Elemental reference |
| Ni | `mp-23` | Fm-3m (225) | 1 | Elemental reference |
| Al3Ni | `mp-622209` | Pnma (62) | 16 | Compound benchmark |
| Al3Ni2 | `mp-1057` | P-3m1 (164) | 5 | Compound benchmark |
| AlNi | `mp-1487` | Pm-3m (221) | 2 | Compound benchmark |
| Al3Ni5 | `mp-16514` | Cmmm (65) | 8 | Compound benchmark |
| AlNi3 | `mp-2593` | Pm-3m (221) | 4 | Compound benchmark |

Step 4 retained nine exact-composition candidate structures and chose the five working structures deterministically. The selection rule is recorded in `data/processed/ni_al_structures/ni_al_phase_manifest.csv`; alternative polymorphs remain available under `data/raw/`.

## 5. Software and numerical systems

| Item | Recorded configuration |
|---|---|
| MACE | MACE-MP-0 Small; CPU; float64; dispersion disabled for Steps 5-8 |
| Step 3 sanity check | MACE-MP-0 Small; CPU; float32 |
| Python atomistic framework | ASE |
| MACE relaxer | FIRE, through `FrechetCellFilter` for full-cell optimisation |
| LAMMPS | `22 Jul 2025 - Update 4`; `metal` units; `atomic` atoms; periodic cells; `eam/alloy` |
| LAMMPS minimiser | Conjugate gradient with quadratic line search and `tri` box relaxation |
| MACE force criterion | <= `0.01 eV/Angstrom` |
| MACE full-cell stress criterion | <= `0.0006241509 eV/Angstrom^3` |
| LAMMPS benchmark | 3 potentials x 7 systems x 3 states = 63 calculations; all completed |

The environment snapshots in `environment/requirements_step*_final.txt` list `mace-torch==0.3.16`, `ase==3.29.0`, and `torch==2.13.0+cpu`. The result artifacts document CPU execution but do not identify a processor model, memory, OS edition, CUDA version, or GPU, so none is inferred.

## 6. Methodology

### Structure handling

Materials Project structures were acquired with their provenance and written to `data/raw/`. The selected working structures in `data/processed/` were read as inputs and were not overwritten by calculated structures.

### MACE calculations

The five compound structures were first evaluated as zero-shot single points. They were then relaxed in two intentionally distinct modes:

1. **Fixed cell:** only atomic positions could change.
2. **Full cell:** positions, cell shape, and volume could change at zero external pressure.

Full-cell MACE calculations began from the original selected structure, not from fixed-cell output. All scheduled MACE relaxations converged and preserved the detected phase symmetry under the project tolerance.

### Formation-energy convention

Formation energies are potential-family-specific. MACE values use MACE-relaxed pure Al and pure Ni references:

`E_f = (E_compound - N_Al * mu_Al - N_Ni * mu_Ni) / (N_Al + N_Ni)`.

For MACE, the calculated chemical potentials are:

- `mu_Al = -3.7095879398802807 eV/atom`
- `mu_Ni = -5.7323473199301382 eV/atom`

The EAM comparisons likewise use independently relaxed Al and Ni references from the same EAM potential. Raw total energies from different models are not compared across compositions.

### Materials Project benchmark convention

Each compound is benchmarked against its Materials Project processed `formation_energy_per_atom`, with the documented entry type `GGA_GGA+U_R2SCAN`. The project compares formation energies, not raw DFT and MACE total energies. Error is always:

`signed error = method formation energy - Materials Project processed DFT formation energy`.

## 7. Completed-stage outcomes

| Step | Outcome |
|---|---|
| 3 | FCC-Al response check completed: a 0.05-Angstrom displacement raised the MACE energy by 0.004013062 eV and generated a clear force response. |
| 4 | Nine candidate structures retained; five selected working phases published with full provenance. |
| 5 | Five of five MACE zero-shot single points completed successfully. |
| 6 | Five fixed-cell and five full-cell MACE relaxations completed safely; all converged. |
| 7 | Pure Al and Ni MACE references converged and five MACE formation energies were calculated. |
| 8 | MACE full-cell formation energies and structures benchmarked against Materials Project values. |
| 9 | Three NIST-sourced EAM/alloy potential files verified and prepared. |
| 10 | 63 LAMMPS states completed with zero failures; MACE and all three EAM models compared. |

## 8. Principal numerical results

### MACE against the Materials Project benchmark

| Phase | MP formation energy | MACE full-cell formation energy | MACE absolute error | MACE volume error |
|---|---:|---:|---:|---:|
| Al3Ni | -0.418776 | -0.460362 | 0.041587 | +2.7397% |
| Al3Ni2 | -0.644217 | -0.641073 | 0.003143 | +2.3826% |
| AlNi | -0.684901 | -0.690231 | 0.005330 | +2.6282% |
| Al3Ni5 | -0.563251 | -0.606098 | 0.042847 | +3.2800% |
| AlNi3 | -0.426420 | -0.488036 | 0.061616 | +2.8941% |

All energies in this table are eV/atom. The five-phase aggregate is MAE `0.030905 eV/atom`, RMSE `0.038471 eV/atom`, mean signed error `-0.029647 eV/atom`, exact ranking agreement, and 10/10 pairwise ordering agreement. All five volume errors are positive; their mean absolute value is 2.7849%.

### Four-model full-cell scorecard

| Model | Formation-energy MAE | RMSE | Mean signed error | Pairwise ordering | Volume MAE |
|---|---:|---:|---:|---:|
| **MACE-MP-0 Small** | **0.030905** | **0.038471** | -0.029647 | **10/10** | 2.7849% |
| Purja Pun-Mishin 2009 EAM | 0.117265 | 0.153381 | +0.106242 | 8/10 | 1.8576% |
| Mishin 2004 EAM (ipr2) | 0.126620 | 0.159870 | +0.118100 | 8/10 | 2.6759% |
| Mishin 2002 EAM | 0.149494 | 0.166682 | +0.149494 | 8/10 | **1.6380%** |

MACE has the lowest formation-energy MAE in this selected five-compound test and is the only model with exact energy ordering. The result is not a claim that it is best for every system: Purja Pun-Mishin 2009 has the smallest error for Al3Ni5, and Mishin 2004 ipr2 has the smallest error for AlNi3.

### MACE relaxation outcome

| Phase | Fixed-cell status | Fixed-cell steps | Full-cell status | Full-cell steps | Full-cell volume change |
|---|---|---:|---|---:|---:|
| Al3Ni | CONVERGED | 28 | CONVERGED | 40 | +2.739666% |
| Al3Ni2 | CONVERGED | 10 | CONVERGED | 33 | +2.382613% |
| AlNi | ALREADY_CONVERGED | 0 | CONVERGED | 5 | +2.628171% |
| Al3Ni5 | CONVERGED | 24 | CONVERGED | 34 | +3.280041% |
| AlNi3 | ALREADY_CONVERGED | 0 | CONVERGED | 14 | +2.894150% |

The complete zero-shot, relaxation, elemental-reference, formation-energy, and LAMMPS calculation tables are reproduced in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

## 9. Correct interpretation of the result

The static five-phase data support a narrow conclusion: MACE-MP-0 Small is the best of the four tested models for the pooled formation-energy metric and phase-energy ordering in this benchmark. It also has a consistent positive volume bias across this sample. The EAM models are less accurate on the pooled energy metric but have lower volume MAE for this particular set.

This does not establish that any potential is universally reliable. In particular:

- Five compounds are not a complete Ni-Al phase diagram.
- A selected-set envelope is not a complete convex hull.
- Materials Project processed values are not experimental truth.
- Ni magnetism remains a caveat because this structural MACE workflow exposes no explicit spin or magnetic-moment input.
- Static equilibrium results do not validate dynamical or defect properties.

## 10. Authoritative files

| Topic | Primary file |
|---|---|
| Structure provenance | `data/processed/ni_al_structures/ni_al_phase_manifest.csv` |
| MACE zero-shot table | `results/mace_zero_shot/tables/ni_al_mace_zero_shot.csv` |
| MACE relaxation table | `results/mace_relaxation/comparison/tables/ni_al_relaxation_comparison.csv` |
| MACE formation energies | `results/mace_formation_energy/tables/ni_al_mace_formation_energies.csv` |
| MACE vs MP benchmark | `results/mace_vs_dft/tables/ni_al_mace_vs_mp_dft.csv` |
| EAM formation energies | `results/lammps_benchmark/tables/ni_al_lammps_formation_energies.csv` |
| Four-model scorecard | `results/lammps_benchmark/tables/ni_al_lammps_vs_mace_mp.csv` |
| Final run report | `results/lammps_benchmark/reports/ni_al_step10_final_report.txt` |

## 11. Next research stage

The recorded next stage is a controlled DFT reference-data campaign for Ni-Al. It should include convergence tests and an explicit treatment of nickel magnetic state. A fine-tuning decision belongs after that reference dataset is validated, not before.

---

## Historical project knowledge preserved verbatim

The section below is retained from the repository revision before the 2026-08-19 documentation audit. It records the original learning notes and planning sequence. The current record above and [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) take precedence if a historical plan conflicts with the completed Step 3-10 work.

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

Separating these tests helps determi…3319 tokens truncated…orce norm;
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

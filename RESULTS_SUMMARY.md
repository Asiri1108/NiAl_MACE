# Verified Calculation Results and Systems

## Documentation status

This file is the compact, numerical source of truth for the completed work in this repository. It was reconciled against the generated tables, checkpoints, and final reports on 2026-08-19. No simulation or model evaluation was rerun for this documentation update.

Companion documents: [README.md](README.md) for the project overview, [Show_Case.md](Show_Case.md) for the narrative explanation of the science, and [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the chronological working log.

Completed calculation stages are Steps 3 through 10. Step 11 (a controlled DFT reference-data campaign) has not started. No MACE fine-tuning, new DFT calculation, molecular dynamics simulation, or OVITO analysis is present in the recorded results.

## Models, engines, and data sources used

| Category | Exact item | Recorded use |
|---|---|---|
| ML potential | **MACE-MP-0 Small** | Pretrained universal MACE potential. Used zero-shot only; no project-specific training or fine-tuning. |
| MACE run mode | CPU, float64, no dispersion | Used for the five-phase zero-shot, relaxation, formation-energy, and MP benchmark steps. The initial FCC-Al sanity check used CPU float32. |
| Classical potential | **Purja Pun-Mishin 2009 Ni-Al EAM** | General binary `eam/alloy` potential, run in LAMMPS. |
| Classical potential | **Mishin 2004 Ni-Al EAM (ipr2)** | Corrected `ipr2` `eam/alloy` potential, run in LAMMPS. |
| Classical potential | **Mishin-Mehl-Papaconstantopoulos 2002 B2-NiAl EAM** | Historical B2-focused `eam/alloy` potential, run in LAMMPS. |
| Atomistic interface | ASE | Structure I/O, MACE evaluation and relaxation orchestration. |
| Classical engine | LAMMPS `22 Jul 2025 - Update 4` | Static EAM benchmark; executable SHA-256 `a6fc5963a8538dbf2eb17f06bc9e34306e363931725402d38cb39d4479ddb54c`. |
| Benchmark data | Materials Project processed DFT-derived values | Database version `2026.04.13`; the selected processed entry type was `GGA_GGA+U_R2SCAN` for every compound. |

The MACE result files record CPU execution. They do not record a CPU model, RAM, operating-system edition, CUDA version, or GPU hardware, so those properties are not claimed here. The final environment snapshots list `mace-torch==0.3.16`, `ase==3.29.0`, and `torch==2.13.0+cpu`; see `environment/requirements_step*_final.txt`.

## Atomic systems

| System | MP ID | Space group | Atoms in cell | Ni atomic fraction | Calculations using it |
|---|---|---|---:|---:|---|
| Al | `mp-134` | Fm-3m (225) | 1 | 0.000 | MACE and EAM elemental reference |
| Ni | `mp-23` | Fm-3m (225) | 1 | 1.000 | MACE and EAM elemental reference |
| Al3Ni | `mp-622209` | Pnma (62) | 16 | 0.250 | MACE zero-shot, relaxation, formation energy, MP benchmark, EAM benchmark |
| Al3Ni2 | `mp-1057` | P-3m1 (164) | 5 | 0.400 | MACE zero-shot, relaxation, formation energy, MP benchmark, EAM benchmark |
| AlNi | `mp-1487` | Pm-3m (221) | 2 | 0.500 | MACE zero-shot, relaxation, formation energy, MP benchmark, EAM benchmark |
| Al3Ni5 | `mp-16514` | Cmmm (65) | 8 | 0.625 | MACE zero-shot, relaxation, formation energy, MP benchmark, EAM benchmark |
| AlNi3 | `mp-2593` | Pm-3m (221) | 4 | 0.750 | MACE zero-shot, relaxation, formation energy, MP benchmark, EAM benchmark |

The Step 4 acquisition preserved nine exact-composition candidates and selected one working structure for each of the five compounds. AlNi had three candidates, AlNi3 had three, and the other three compositions each had one. All five selected structures are stable in the Materials Project context (`energy_above_hull = 0.0 eV/atom`).

## Calculation settings

### MACE

- Single-point phase evaluation: MACE-MP-0 Small, CPU, float64, dispersion disabled; no atomic or cell relaxation.
- Fixed-cell relaxation: FIRE; atomic positions allowed to move; force threshold `0.01 eV/Angstrom`; maximum 500 steps.
- Full-cell relaxation: FIRE through ASE `FrechetCellFilter`; zero external pressure; positions, cell shape, and volume allowed to move; force threshold `0.01 eV/Angstrom`; absolute six-component ASE stress threshold `0.0006241509 eV/Angstrom^3`; maximum 1000 steps.
- Safety checks required finite values, unchanged atom identity/order and PBC, positive volume, internal displacement no larger than `2 Angstrom`, and full-cell volume change no larger than 25%.

### LAMMPS EAM

- `metal` units, `atomic` atom style, periodic boundaries, `eam/alloy` pair style, type order `Al Ni`.
- Conjugate-gradient minimisation; quadratic line search; zero target pressure; `tri` box relaxation; technical force tolerance `1e-10 eV/Angstrom`; scientific acceptance force threshold `0.01 eV/Angstrom`; maximum absolute stress `0.0006241509 eV/Angstrom^3` (about 1000 bar).
- Three states were calculated independently for every potential-system pair: initial single point, fixed-cell atomic relaxation, and full-cell relaxation.

## Step 3: FCC-Al MACE sanity check

This was a software-response check, not an accuracy benchmark. A 32-atom 2 x 2 x 2 conventional FCC-Al supercell was evaluated with MACE-MP-0 Small on CPU in float32.

| Structure | Total energy (eV) | Energy/atom (eV) | Max force (eV/Angstrom) | Mean force (eV/Angstrom) |
|---|---:|---:|---:|---:|
| Perfect FCC Al | -118.706176758 | -3.709568024 | 0.000002447 | 0.000001396 |
| One atom displaced by 0.05 Angstrom | -118.702163696 | -3.709442616 | 0.160584956 | 0.015314721 |

The displacement changed the energy by `+0.004013062 eV`.

## Step 4: structure acquisition

| Phase | Selected MP ID | Atoms | MP formation energy (eV/atom) | MP volume (Angstrom^3) |
|---|---|---:|---:|---:|
| Al3Ni | `mp-622209` | 16 | -0.418775703 | 228.550737269 |
| Al3Ni2 | `mp-1057` | 5 | -0.644216628 | 67.435060910 |
| AlNi | `mp-1487` | 2 | -0.684901150 | 23.386572118 |
| Al3Ni5 | `mp-16514` | 8 | -0.563250545 | 90.372065811 |
| AlNi3 | `mp-2593` | 4 | -0.426419777 | 43.727854385 |

## Step 5: MACE zero-shot single points

No relaxation was performed. The input geometry is the selected Materials Project structure.

| Phase | Total energy (eV) | Energy/atom (eV) | Volume/atom (Angstrom^3) | Max force (eV/Angstrom) | Max abs. stress (eV/Angstrom^3) |
|---|---:|---:|---:|---:|---:|
| Al3Ni | -74.695710879 | -4.668481930 | 14.284421079 | 0.212792949 | 0.020251825 |
| Al3Ni2 | -25.780552090 | -5.156110418 | 13.487012182 | 0.102594156 | 0.023518165 |
| AlNi | -10.813681753 | -5.406840876 | 11.693286059 | 0.000000049 | 0.029697286 |
| Al3Ni5 | -44.568076077 | -5.571009510 | 11.296508226 | 0.160800416 | 0.040423595 |
| AlNi3 | -22.836292364 | -5.709073091 | 10.931963596 | 0.000000060 | 0.036641959 |

All five evaluations completed successfully. The raw six-component stress vectors, force statistics, and annotated structures are in `results/mace_zero_shot/`.

## Step 6: MACE relaxations

All ten scheduled relaxations converged safely. Initial, fixed-cell, and full-cell calculations started independently from the original selected structure; full-cell runs did not start from fixed-cell results.

| Phase | Fixed-cell status | Steps | Fixed-cell delta E (eV) | Full-cell status | Steps | Full-cell delta E (eV) | Full-cell delta V (%) |
|---|---|---:|---:|---|---:|---:|---:|
| Al3Ni | CONVERGED | 28 | -0.039746712 | CONVERGED | 40 | -0.114531747 | +2.739666 |
| Al3Ni2 | CONVERGED | 10 | -0.001103026 | CONVERGED | 33 | -0.018272683 | +2.382613 |
| AlNi | ALREADY_CONVERGED | 0 | 0.000000000 | CONVERGED | 5 | -0.008715575 | +2.628171 |
| Al3Ni5 | CONVERGED | 24 | -0.003624475 | CONVERGED | 34 | -0.071204906 | +3.280041 |
| AlNi3 | ALREADY_CONVERGED | 0 | 0.000000000 | CONVERGED | 14 | -0.022479591 | +2.894150 |

Detected symmetry was preserved in all five phases for both relaxation modes at `symprec = 0.001 Angstrom` and `angle_tolerance = 5 degrees`.

## Step 7: MACE elemental references and formation energies

Formation energy is calculated independently for each compound as:

`E_f = (E_compound - N_Al * mu_Al - N_Ni * mu_Ni) / (N_Al + N_Ni)`.

| Reference | Initial energy/atom (eV) | Relaxed energy/atom (eV) | Optimiser steps | Volume change | Status |
|---|---:|---:|---:|---:|---|
| Al (`mp-134`) | -3.709129750 | -3.709587940 | 7 | +1.154351% | CONVERGED |
| Ni (`mp-23`) | -5.726033697 | -5.732347320 | 4 | +3.224709% | CONVERGED |

Therefore `mu_Al = -3.7095879398802807 eV/atom` and `mu_Ni = -5.7323473199301382 eV/atom`.

| Phase | Initial formation energy (eV/atom) | Full-cell formation energy (eV/atom) | Relaxation effect (eV/atom) | On selected-set envelope |
|---|---:|---:|---:|---|
| Al3Ni | -0.455126193 | -0.460362379 | -0.005236186 | Yes |
| Al3Ni2 | -0.640219089 | -0.641073263 | -0.000854174 | Yes |
| AlNi | -0.689259153 | -0.690231034 | -0.000971881 | Yes |
| Al3Ni5 | -0.601314792 | -0.606097570 | -0.004782778 | Yes |
| AlNi3 | -0.487265381 | -0.488035514 | -0.000770133 | Yes |

The selected-set envelope includes only the two elemental references and the five compounds. It is not a complete Ni-Al convex hull or a phase diagram.

## Step 8: MACE versus Materials Project benchmark

The benchmark compares full-cell MACE formation energies against Materials Project processed values; it does not compare raw MACE and DFT total energies.

| Phase | MP DFT formation energy (eV/atom) | MACE full-cell formation energy (eV/atom) | Signed error (eV/atom) | Absolute error (eV/atom) | MACE volume error | Symmetry preserved |
|---|---:|---:|---:|---:|---:|---|
| Al3Ni | -0.418775703 | -0.460362379 | -0.041586676 | 0.041586676 | +2.7397% | Yes |
| Al3Ni2 | -0.644216628 | -0.641073263 | +0.003143365 | 0.003143365 | +2.3826% | Yes |
| AlNi | -0.684901150 | -0.690231034 | -0.005329884 | 0.005329884 | +2.6282% | Yes |
| Al3Ni5 | -0.563250545 | -0.606097570 | -0.042847025 | 0.042847025 | +3.2800% | Yes |
| AlNi3 | -0.426419777 | -0.488035514 | -0.061615736 | 0.061615736 | +2.8941% | Yes |

| Aggregate MACE metric (n=5) | Result |
|---|---:|
| Mean absolute error | 0.030904537 eV/atom |
| RMSE | 0.038471045 eV/atom |
| Mean signed error | -0.029647191 eV/atom |
| Maximum absolute error | 0.061615736 eV/atom (AlNi3) |
| Pearson / Spearman correlation | 0.990950180 / 1.000000000 (exploratory) |
| Exact ranking / pairwise agreement | Yes / 10 of 10 |
| Mean absolute volume error | 2.7849% |

## Step 9: classical-potential selection

Three potential files were retrieved from the NIST Interatomic Potentials Repository, verified, and prepared. This step did not run the potentials; execution occurred in Step 10.

| Key | Potential file | Intended role | SHA-256 |
|---|---|---|---|
| `pun_mishin_2009` | `Mishin-Ni-Al-2009.eam.alloy` | Primary general Ni-Al comparison | `e0c4b32cbf05f8044540fb5ebe220171aa4e2915d98040e6cbce1a1f8e2f582b` |
| `mishin_2004_ipr2` | `NiAl_Mishin_2004.eam.alloy` | Gamma/gamma-prime-focused comparison | `15712c13a472843649cc8550e412f8af1847adc7a7ecf12947535f6b529a5611` |
| `mishin_2002` | `NiAl02.eam.alloy` | Historical B2-focused comparison | `68de13eb1b6682bfdef15ef75248104f9b796eeae7fda5c8e1007872ba767b3b` |

## Step 10: LAMMPS EAM benchmark

All LAMMPS input structures round-trip validated. The calculation matrix was 3 potentials x 7 systems x 3 states = 63 calculations. All 63 completed, with zero failures.

### Elemental references used by each EAM potential

| Potential | Al full-cell energy (eV/atom) | Ni full-cell energy (eV/atom) |
|---|---:|---:|
| Purja Pun-Mishin 2009 | -3.359999988 | -4.449999986 |
| Mishin 2004 (ipr2) | -3.360000023 | -4.449999985 |
| Mishin 2002 | -3.362177320 | -4.501335164 |

### Full formation-energy calculation matrix

| Potential | Phase | Initial (eV/atom) | Fixed-cell (eV/atom) | Full-cell (eV/atom) | On selected-set envelope |
|---|---|---:|---:|---:|---|
| Purja Pun-Mishin 2009 | Al3Ni | -0.225127 | -0.233042 | -0.242708 | No |
| Purja Pun-Mishin 2009 | Al3Ni2 | -0.365739 | -0.366511 | -0.362929 | No |
| Purja Pun-Mishin 2009 | AlNi | -0.606038 | -0.606038 | -0.605871 | Yes |
| Purja Pun-Mishin 2009 | Al3Ni5 | -0.539702 | -0.542002 | -0.540870 | Yes |
| Purja Pun-Mishin 2009 | AlNi3 | -0.460621 | -0.460621 | -0.453978 | Yes |
| Mishin 2004 (ipr2) | Al3Ni | -0.218592 | -0.229659 | -0.243823 | No |
| Mishin 2004 (ipr2) | Al3Ni2 | -0.346027 | -0.347609 | -0.352211 | No |
| Mishin 2004 (ipr2) | AlNi | -0.595217 | -0.595217 | -0.590420 | Yes |
| Mishin 2004 (ipr2) | Al3Ni5 | -0.489207 | -0.494378 | -0.512888 | No |
| Mishin 2004 (ipr2) | AlNi3 | -0.443635 | -0.443635 | -0.447720 | Yes |
| Mishin 2002 | Al3Ni | -0.244728 | -0.261185 | -0.267036 | Yes |
| Mishin 2002 | Al3Ni2 | -0.363757 | -0.369921 | -0.370819 | No |
| Mishin 2002 | AlNi | -0.536236 | -0.536236 | -0.533491 | Yes |
| Mishin 2002 | Al3Ni5 | -0.427070 | -0.429789 | -0.435295 | No |
| Mishin 2002 | AlNi3 | -0.387313 | -0.387313 | -0.383452 | Yes |

### Four-method comparison against Materials Project

| Method | MAE (eV/atom) | RMSE (eV/atom) | Mean signed error (eV/atom) | Maximum absolute error | Exact ranking | Pairwise agreement |
|---|---:|---:|---:|---:|---|---:|
| MACE-MP-0 Small | **0.030905** | **0.038471** | -0.029647 | 0.061616 (AlNi3) | Yes | **10/10** |
| Purja Pun-Mishin 2009 EAM | 0.117265 | 0.153381 | +0.106242 | 0.281287 (Al3Ni2) | No | 8/10 |
| Mishin 2004 EAM (ipr2) | 0.126620 | 0.159870 | +0.118100 | 0.292006 (Al3Ni2) | No | 8/10 |
| Mishin 2002 EAM | 0.149494 | 0.166682 | +0.149494 | 0.273397 (Al3Ni2) | No | 8/10 |

The best full-cell formation-energy method by phase was MACE for Al3Ni, Al3Ni2, and AlNi; Purja Pun-Mishin 2009 for Al3Ni5; and Mishin 2004 ipr2 for AlNi3.

| Method | Mean signed volume error | Volume MAE | Symmetry agreement |
|---|---:|---:|---:|
| MACE-MP-0 Small | +2.7849% | 2.7849% | 5/5 |
| Purja Pun-Mishin 2009 EAM | +0.0931% | 1.8576% | 5/5 |
| Mishin 2004 EAM (ipr2) | +2.6759% | 2.6759% | 5/5 |
| Mishin 2002 EAM | +1.5945% | **1.6380%** | 5/5 |

### EAM runtime record

| Potential | Total wall time | Mean wall time per structure | Force evaluations | States |
|---|---:|---:|---:|---:|
| Purja Pun-Mishin 2009 | 2.714 s | 0.388 s | 972 | 21 |
| Mishin 2004 (ipr2) | 2.663 s | 0.380 s | 1123 | 21 |
| Mishin 2002 | 2.786 s | 0.398 s | 780 | 21 |

The MACE and LAMMPS timings have different monitoring scopes, so no precise cross-engine speed ratio is reported.

## Interpretation boundaries

- The Materials Project values are processed DFT-derived references, not experimental measurements or raw DFT totals.
- The five compounds form a small static-bulk sample. They do not constitute a complete Ni-Al phase diagram, convex hull, or transferability proof.
- All selected-set envelopes are incomplete by construction; untested phases may lie below them.
- The current MACE workflow contains no user-controlled spin or magnetic-moment input. Nickel magnetic-state sensitivity remains a caveat for future DFT comparison.
- No result establishes accuracy for defects, surfaces, interfaces, finite temperature, mechanical deformation, or molecular dynamics.

## Authoritative artifacts

| Content | Path |
|---|---|
| Selected structures and provenance | `data/processed/ni_al_structures/ni_al_phase_manifest.csv` |
| MACE zero-shot results | `results/mace_zero_shot/tables/ni_al_mace_zero_shot.csv` |
| MACE relaxation results | `results/mace_relaxation/comparison/tables/ni_al_relaxation_comparison.csv` |
| MACE elemental references | `results/mace_elemental_references/full_cell/tables/mace_elemental_reference_summary.csv` |
| MACE formation energies | `results/mace_formation_energy/tables/ni_al_mace_formation_energies.csv` |
| MACE versus MP benchmark | `results/mace_vs_dft/tables/ni_al_mace_vs_mp_dft.csv` |
| MACE structural benchmark | `results/mace_vs_dft/tables/ni_al_structural_comparison.csv` |
| EAM calculation matrix | `results/lammps_benchmark/tables/ni_al_lammps_formation_energies.csv` |
| Four-method scorecard | `results/lammps_benchmark/tables/ni_al_lammps_vs_mace_mp.csv` |
| Final Step 10 report | `results/lammps_benchmark/reports/ni_al_step10_final_report.txt` |

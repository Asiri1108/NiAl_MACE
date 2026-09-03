# Ni-Al Interatomic-Potential Benchmark

A completed static, zero-temperature benchmark of a pretrained **MACE-MP-0 Small** machine-learning
potential against **three published Ni-Al EAM potentials**, measured on five Ni-Al intermetallic
phases and referenced to Materials Project DFT-derived data.

**Headline result:** used zero-shot with no Ni-Al fine-tuning, MACE-MP-0 Small reached a
formation-energy MAE of **0.0309 eV/atom** versus **0.1173 eV/atom** for the best classical
EAM potential, and was the only method of the four that reproduced the DFT stability ordering
of all five phases exactly.

This is not a MACE fine-tuning project. No project-specific MACE training, new DFT calculation,
molecular dynamics, or OVITO analysis has been run.

## Documentation map

| If you want | Read |
|---|---|
| Overview, models, systems, headline numbers | **This file** |
| The complete audited numerical record and exact settings | [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) |
| The science explained — why each choice, what each number means | [Show_Case.md](Show_Case.md) |
| The chronological step-by-step working log (Steps 0-10) | [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) |

`RESULTS_SUMMARY.md` is the numerical source of truth. Where any document disagrees with it,
it wins.

## Research question and scope

> How does a pretrained MACE-MP-0 Small model compare with three published Ni-Al EAM potentials
> for static formation energies and equilibrium structures of five selected Ni-Al intermetallic phases?

The scope is deliberately narrow:

- Static, zero-temperature bulk structures only.
- Five selected intermetallic compounds plus pure Al and Ni reference cells.
- Materials Project processed DFT-derived values as a benchmark, not experimental truth.
- Independent elemental references for each potential family before any formation energy is computed.

It is **not** a phase-diagram calculation, nor a validation of finite-temperature, defect, surface,
interface, deformation, or molecular-dynamics behaviour.

## Models used

| Role | Model or system | How it was used |
|---|---|---|
| Machine-learning potential | **MACE-MP-0 Small** | Pretrained, zero-shot; CPU; float64 for Steps 5-10. The Step 3 FCC-Al sanity check used float32. No fine-tuning was performed. |
| Classical potential | **Purja Pun-Mishin (2009) Ni-Al EAM** | `eam/alloy` in LAMMPS; primary general-binary EAM baseline. |
| Classical potential | **Mishin (2004) Ni-Al EAM, ipr2** | `eam/alloy` in LAMMPS; the corrected `ipr2` implementation. The superseded `ipr1` was excluded. |
| Classical potential | **Mishin-Mehl-Papaconstantopoulos (2002) B2-NiAl EAM** | `eam/alloy` in LAMMPS; historical B2-focused baseline. |
| Reference data | **Materials Project processed DFT-derived values** | Benchmark only; database version `2026.04.13`, entry type `GGA_GGA+U_R2SCAN` for all five compounds. Not experimental truth, and never used as a MACE elemental reference. |

## Systems evaluated

| System | MP ID | Structure | Atoms in cell | Ni fraction | Use |
|---|---|---|---:|---:|---|
| Al | `mp-134` | FCC, Fm-3m (225) | 1 | 0.000 | MACE and EAM elemental reference |
| Ni | `mp-23` | FCC, Fm-3m (225) | 1 | 1.000 | MACE and EAM elemental reference |
| Al3Ni | `mp-622209` | Pnma (62) | 16 | 0.250 | Compound benchmark |
| Al3Ni2 | `mp-1057` | P-3m1 (164) | 5 | 0.400 | Compound benchmark |
| AlNi | `mp-1487` | Pm-3m (221) | 2 | 0.500 | Compound benchmark |
| Al3Ni5 | `mp-16514` | Cmmm (65) | 8 | 0.625 | Compound benchmark |
| AlNi3 | `mp-2593` | Pm-3m (221) | 4 | 0.750 | Compound benchmark |

Nine exact-composition Materials Project candidates were retained. The five working structures were
selected reproducibly by the documented ranking rule in
`data/processed/ni_al_structures/ni_al_phase_manifest.csv`; the four alternative polymorphs remain
in `data/raw/`. All five selected structures have `energy_above_hull = 0.0 eV/atom`.

## Method in brief

Materials Project structures were acquired with provenance into `data/raw/`; the selected working
structures in `data/processed/` were read as inputs and never overwritten by calculated output.

Each compound was evaluated as a MACE zero-shot single point, then relaxed in two intentionally
separate modes — **fixed cell** (atoms only) and **full cell** (atoms, cell shape, and volume at zero
pressure). Full-cell relaxation started from the original selected structure, not from fixed-cell
output, so the two modes stay independent.

Formation energies are potential-family-specific:

```
E_f = (E_compound - N_Al * mu_Al - N_Ni * mu_Ni) / (N_Al + N_Ni)
```

Every model supplies its own independently relaxed pure Al and pure Ni references, because every
potential defines its own energy zero. Raw total energies are never compared across models or
compositions. For MACE the calculated chemical potentials are:

- `mu_Al = -3.709587940 eV/atom`
- `mu_Ni = -5.732347320 eV/atom`

Errors are always reported as `signed error = method formation energy - MP processed DFT formation energy`.

## Main results

All five MACE full-cell relaxations converged safely and preserved the detected starting symmetry.

| Phase | MP processed DFT | MACE full-cell | MACE signed error | MACE volume error |
|---|---:|---:|---:|---:|
| Al3Ni | -0.418776 | -0.460362 | -0.041587 | +2.7397% |
| Al3Ni2 | -0.644217 | -0.641073 | +0.003143 | +2.3826% |
| AlNi | -0.684901 | -0.690231 | -0.005330 | +2.6282% |
| Al3Ni5 | -0.563251 | -0.606098 | -0.042847 | +3.2800% |
| AlNi3 | -0.426420 | -0.488036 | -0.061616 | +2.8941% |

Formation energies in eV/atom.

### Four-model scorecard

| Method | Formation-energy MAE | RMSE | Mean signed error | Pairwise ordering | Volume MAE |
|---|---:|---:|---:|---:|---:|
| **MACE-MP-0 Small** | **0.030905** | **0.038471** | -0.029647 | **10/10** | 2.7849% |
| Purja Pun-Mishin 2009 EAM | 0.117265 | 0.153381 | +0.106242 | 8/10 | 1.8576% |
| Mishin 2004 EAM (ipr2) | 0.126620 | 0.159870 | +0.118100 | 8/10 | 2.6759% |
| Mishin 2002 EAM | 0.149494 | 0.166682 | +0.149494 | 8/10 | **1.6380%** |

MACE-MP-0 Small has the lowest formation-energy MAE and is the only model reproducing the Materials
Project ordering exactly. It is **not** best on every individual phase: Purja Pun-Mishin 2009 is best
for Al3Ni5, and Mishin 2004 ipr2 is best for AlNi3. All three EAM potentials have a lower volume MAE
than MACE, which shows a consistent positive volume bias across this sample.

## Completed stages

| Step | Outcome |
|---|---|
| 3 | FCC-Al response check: a 0.05-Angstrom displacement raised the MACE energy by 0.004013062 eV with a clear force response. |
| 4 | Nine candidate structures retained; five working phases published with full provenance. |
| 5 | 5/5 MACE zero-shot single points completed. |
| 6 | 5 fixed-cell and 5 full-cell MACE relaxations completed and converged. |
| 7 | Pure Al and Ni MACE references converged; five MACE formation energies calculated. |
| 8 | MACE full-cell formation energies and structures benchmarked against Materials Project. |
| 9 | Three NIST-sourced `eam/alloy` potential files verified and prepared. |
| 10 | 63 LAMMPS states completed with zero failures; MACE and all three EAM models compared. |

Step 11 (a controlled DFT reference-data campaign) has not started.

## Software and recorded settings

| Item | Recorded setting |
|---|---|
| MACE | CPU, float64, dispersion disabled (Steps 5-10); FIRE optimiser, `FrechetCellFilter` for full cell |
| MACE convergence | max force <= 0.01 eV/Angstrom; full-cell max absolute ASE stress <= 0.0006241509 eV/Angstrom^3 |
| MACE step limits | 500 (fixed cell) / 1000 (full cell) |
| LAMMPS | `22 Jul 2025 - Update 4`; `eam/alloy`; `metal` units; periodic; conjugate-gradient with `fix box/relax tri 0.0` |
| LAMMPS matrix | 3 potentials x 7 systems x 3 states = 63 calculations; 63 completed; 0 failures |
| Environment | `mace-torch==0.3.16`, `ase==3.29.0`, `torch==2.13.0+cpu`; see `environment/requirements_step*_final.txt` |

Result artifacts record CPU execution but preserve no CPU model, memory size, OS edition, CUDA
version, or GPU configuration. Those unrecorded details are deliberately not inferred.

## Repository layout

```
configs/       step configuration files
data/raw/      Materials Project structures as acquired, with provenance
data/processed/  the five selected working structures + phase manifest
scripts/       acquisition, calculation, analysis, and run_step*_pipeline.py drivers
results/       all generated tables, reports, checkpoints, and figures
environment/   pinned requirements snapshots per step
docs/          research log archive
```

`lammps/`, `models/`, `notebooks/`, `ovito/`, `references/`, and `runs/` are empty placeholders
reserved for later stages. LAMMPS run artifacts are written under `results/lammps_benchmark/`.

## Key output files

- `results/mace_zero_shot/tables/ni_al_mace_zero_shot.csv`
- `results/mace_relaxation/comparison/tables/ni_al_relaxation_comparison.csv`
- `results/mace_elemental_references/full_cell/tables/mace_elemental_reference_summary.csv`
- `results/mace_formation_energy/tables/ni_al_mace_formation_energies.csv`
- `results/mace_vs_dft/tables/ni_al_mace_vs_mp_dft.csv`
- `results/lammps_benchmark/tables/ni_al_lammps_formation_energies.csv`
- `results/lammps_benchmark/tables/ni_al_lammps_vs_mace_mp.csv`
- `results/lammps_benchmark/reports/ni_al_step10_final_report.txt`

The full path table is in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md#authoritative-artifacts).

## Scope limits and next stage

The benchmark covers static bulk cells only. It does not establish accuracy for defects, interfaces,
surfaces, finite-temperature properties, dynamical stability, or long molecular-dynamics trajectories.
Five compounds are not a complete Ni-Al phase diagram, and a selected-set envelope is not a complete
convex hull. Nickel magnetism is a further limitation: the structural MACE workflow exposes no
user-controlled spin or magnetic-moment input.

The documented next stage is a controlled Ni-Al DFT reference-data campaign with convergence and
magnetic-state checks. A MACE fine-tuning decision belongs after that dataset is validated, not before.

---

No calculation was rerun during documentation updates. The values above were transcribed from the
completed, versioned result artifacts during the documentation audit of 2026-08-19.

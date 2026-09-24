# ROM_TurbulentFlows

Code for the MMAE 597 research project: **β-VAE reduced-order models of 2-D flat-plate
wake flows**, and whether **POD–Galerkin synthetic snapshots** (from the Navier–Stokes
equations projected on POD modes) improve the β-VAE reconstruction energy **Ek** on real
validation data.

- `rom/` is the shared library. Importing it runs nothing.
- `studies/` holds the scripts that produced the results, in pipeline order 1 → 6.
- `tools/` holds viewers and quick checks.
- The data, checkpoints and results are **not** in the repository (about 138 GB).

## Folder layout

The code finds everything relative to the repository root (`rom/paths.py`):

```
<parent>/
├── ROM_TurbulentFlows/     this repository
├── DATA/                   input .mat files: Alpha0/, 2PlatesGap/, AUGMENTED*/ ...
└── bestModels/             checkpoints (.pt) and POD files (.npz)
<parent>/../convergence/    every results CSV / xlsx / figure
```

To point the code somewhere else (for example at test data), set any of
`ROM_DATA_DIR`, `ROM_MODELS_DIR`, `ROM_RESULTS_DIR`.

## Install

Use the same Python that runs your scripts (Terminal and IDLE):

```bash
cd ROM_TurbulentFlows
python3 -m pip install -r requirements.txt     # pinned versions the refactor was verified with
python3 -m pip install -e .                    # makes `rom` importable from anywhere
```

Keep the `-e` (editable install). `rom/paths.py` finds `DATA/`, `bestModels/` and
`convergence/` from where the repository is, so a normal install would look in the wrong place.

Check the install and the folders:

```bash
python3 -c "from rom import paths; print(paths.DATA, paths.MODELS, paths.RESULTS, sep='\n')"
python3 -m pytest -q             # smoke tests on tiny fake data, ~10 s, no real data needed
```

## Running a study

```bash
python3 studies/<folder>/<script>.py [options]
```

Every study can be checked first without training for hours:

- `--plan`: prints what is left and the time estimate. Nothing runs.
- `--epochs 2`: a quick smoke run.
- `--dry-run` / `--assess-only`: see each study below.

Long runs from Terminal use `scripts/launch.sh`. It caps threads, keeps the Mac awake,
writes a log, can wait for another run with `--after`, and can chain scripts with `--then`:

```bash
scripts/launch.sh studies/4_region_map/run_region_map.py
scripts/launch.sh --after run_lowdata_subsets studies/6_capacity/run_capacity.py \
                  --then studies/6_capacity/capacity_headroom.py
```

Every long study resumes where it stopped. Rows already in its CSV are skipped, so just
launch it again.

## Pipeline

All networks share one recipe (`rom/vae.py`): 500 epochs, β = 5e-3, batch 32, lr 3e-4,
torch seed 7, best-validation checkpoint. Normalisation and the mean field come from the
real training snapshots only. Ek is always measured on real validation snapshots.
The split is the random 10 % validation with seed 7 (`rom/split.py`).
Times are for the MPS Mac. About 1.3 s per training snapshot for 500 epochs.

### 1. β-VAE vs POD — `studies/1_vae_vs_pod/`

How does a β-VAE compare with the linear POD basis on one dataset?

| Command | Writes | Time |
|---|---|---|
| `python3 pod.py [--data FILE]` | `bestModels/pod_<data>.npz` + figures | minutes |
| `python3 beta_vae.py [--data FILE] [--epochs N]` | `bestModels/model_betaVAE_<data>_lat…_b…_ep….pt` + figures | from minutes to hours (1000 epochs by default) |
| `python3 analysis.py [--vae PT] [--pod NPZ] [--data FILE]` | figures only (POD vs VAE modes, errors) | minutes |

All three read every layout through `rom.data.load_data`: the Alpha0 and 2-plates files,
and the older Tensor / channel `U` / `UW` files. The file is set in each script's CONFIG
block, or with `--data`. `beta_vae.py` seeds torch (`TORCH_SEED = 7`), so a run can be repeated.

### 2. Convergence — `studies/2_convergence/`

How many snapshots do POD and the β-VAE need before more data stops helping?

| Command | Writes | Time |
|---|---|---|
| `python3 assess_pod_modes.py [random\|tail\|tail5\|tail2.5]` | `pod_modes_assessment[_tailval…].csv`, `pod_spectra[…].npz` (k for 90/95/99/99.9 % energy) | minutes |
| `python3 run_alpha0_convergence.py [--plan] [--epochs N]` | runs the next three per dataset and split (SPLIT in CONFIG) | POD: minutes per dataset; VAE: many trainings per dataset (one night or more) |
| `python3 pod_convergence.py <file> <cap> <k> [level] [split]` | `DATA/AUGMENTED[_TAILVAL…]/<base>_pod<level>_convergence.mat`, `POD<level>_*` columns of `DATA/<folder>/convergence_last_point[…].csv` | minutes |
| `python3 nn_convergence.py <file> <cap> <latent> [split] [--epochs N]` | `…/<base>_nn_convergence.mat`, `NN_*` columns of the same CSV | hours |
| `python3 nn_latent_test.py [split] [Re60=10,14 …] [--epochs N]` | `nn_latent_test[…].csv` | ~20–30 min per full-pool training |
| `python3 make_alpha0_excel.py [split]` | `Alpha0_convergence[…].xlsx` | seconds |

Result: Alpha0 Re 50–80 capped at 1500 snapshots, and Re 100 capped at 1000 (all of its
record). Latents are Re50 5, Re60 7, Re70 9, Re80 11, Re100 15.

### 3. Re 100 synthetic fraction — `studies/3_re100_fraction/`

For the summer Re 100 setup, how does Ek change with the share of synthetic snapshots
in the training set, the Galerkin model size and the number of real snapshots?

| Command | Writes | Time |
|---|---|---|
| `python3 re100_fraction_sweep.py [galerkin\|galerkin_ns] [--latent L] [--fractions …] [--dry-run]` | `re100_fraction_sweep.csv` | ~3 min generation + 500-epoch trainings (≈ 3–4 h) |
| `python3 re100_conv_fraction_sweep.py [galerkin\|galerkin_ns] [--modes K] [--n-real N] [--assess-only]` | `re100_pod_convergence_k<K>.csv/.png`, `re100_conv_fraction_sweep.csv` | step A ~10 min, then the trainings |
| `python3 run_re100_grid.py [galerkin\|galerkin_ns] [--modes …] [--tols …] [--plan]` | `re100_tol_modes_grid.csv` (+ `re100_tol_modes_grid_failures.log`) | `--plan` prints it |

Result: the Re 100 synthetic-fraction heatmap (`re100_tol_modes_grid.csv`).

### 4. Region map — `studies/4_region_map/`

When do synthetic snapshots help, whatever the generator? The gain is mapped against
**headroom** (POD ceiling of the real subset minus real-only Ek) and **quality** (the
reduced model's error on held-out real windows).

| Command | Writes | Time |
|---|---|---|
| `python3 run_region_map.py [--plan] [--no-train] [--datasets …] [--n …] [--subsets S] [--epochs N]` | `region_map_results.csv` (`region_map_generators.csv` with `--no-train`) | one night |
| `python3 run_round1.py [--exp E1,…] [--plan]` (was `run_experiments.py`, E1–E5) | `experiments_results.csv` | ≈ 12.5 h |
| `python3 run_round2.py [--exp N1,…] [--plan] [--csv FILE]` (was `run_experiments2.py`, N1–N4) | `experiments2_results.csv` | ≈ 12.5 h |

Results: region map and follow-up rounds 1–2. The best configuration is **N4**: Re 100,
400 real + 800 synthetic, latent 25, K = 152, **+6.8 ± 1.3 Ek**, better on 5/5 subsets.

### 5. Low data — `studies/5_low_data/`

Do synthetic snapshots still help with only 25–100 real snapshots?

| Command | Writes | Time |
|---|---|---|
| `python3 run_lowdata.py [--stage A\|B] [--plan] [--datasets …] [--n …] [--subsets S] [--epochs N]` | stage A screen `lowdata_screen.csv`, stage B `lowdata_results.csv` | A ≈ 1 h, B ≈ 4.5 h |
| `python3 run_lowdata_subsets.py [--subsets 8] [--plan] [--epochs N] [--csv FILE]` | more rows in `lowdata_results.csv` (same columns) | `--plan` prints it |

Stage B takes each cell's truncation from the stage-A screen (`pick_trunc`: the highest POD
ceiling among the faithful truncations).

Result: Re 50 with 25 real snapshots over 8 subsets. The outcome depends on whether the
training converges or collapses. Where it converges, synthetic data give about 62 % of the
benefit of the same number of extra real snapshots.

### 6. Capacity — `studies/6_capacity/`

How much of the headroom can the network actually use at its latent size?

| Command | Writes | Time |
|---|---|---|
| `python3 run_capacity.py [--datasets …] [--latents …] [--plan] [--epochs N]` | `capacity_ceilings.csv` | ~20 min per Re 100 training |
| `python3 capacity_headroom.py` | `headroom_corrected.csv`, `headroom_corrected_cells.csv` | seconds |

Results: Re 100 capacity ceilings are **latent 5 = 42.8 %, latent 15 = 89.6 %, latent 25 = 92.2 %**.
The capacity-corrected headroom = min(POD ceiling, capacity ceiling) − real-only Ek.
Run `capacity_headroom.py` again after every new `run_capacity.py`. `launch.sh … --then`
does it automatically.

## `rom/` at a glance

| Module | Contents |
|---|---|
| `paths.py` | `DATA`, `AUGMENTED`, `MODELS`, `RESULTS` |
| `data.py` | dataset registry (file, snapshot cap, latent), `load_data` (all .mat layouts), `load` (fast Alpha0 reader) |
| `split.py` | split modes random / tail / tail5 / tail2.5, seeds, subset draws (blocks, contiguous, Re 100 pool) |
| `pod.py` | POD by snapshots, POD ceiling, modes for an energy level, low-data truncations, POD convergence at K modes |
| `vae.py` | Encoder / Decoder, loss, Ek, det(R), the shared `train()` |
| `galerkin_ns.py` | NS-projected operators l, q (built once, sliced to smaller K), RK4, energy-band screening |
| `galerkin_data.py` | data-identified quadratic Galerkin model |
| `augment.py` | generator settings, summer generator, quality error, the arms: synthetic, jitter, real_proj / real_full |
| `results.py` | CSV append / resume helpers, `lowdata_results.csv` columns, convergence last-point CSV |

## Tools

`tools/inspect_vae.py` re-plots a saved checkpoint. `tools/show_dataset.py` shows one Alpha0 snapshot.
`tools/pod_energy_spectrum.py` plots cumulative POD energy. `tools/view_augmented_movie.py` plays real
vs synthetic snapshots of a bundle.

Removed and superseded files are listed in `REMOVED.md`, with the command to recover each.

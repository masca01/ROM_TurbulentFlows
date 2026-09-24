# Removed files

Everything below was deleted from the tree in the cleanup. Nothing is lost: the whole
project before the cleanup is tagged `pre-cleanup`, so any file comes back with

    git show pre-cleanup:<file> > <file>

(or `git checkout pre-cleanup -- <file>` to restore it in place).

## Deleted

| File | What it did | Why it was removed |
|---|---|---|
| `pod_augment_gaussian.py` | Summer strategy A: synthetic snapshots with each POD coefficient drawn from an independent Gaussian | Superseded by the dynamical (Galerkin) generators; only fed the removed summer testers |
| `pod_augment_gmm.py` | Summer config #3: joint GMM on the leading POD coefficients | Superseded by the Galerkin generators; only fed the removed summer testers |
| `pod_augment_noise.py` | Summer strategy C: real coefficient vectors + small jitter | Superseded; the same idea lives on as the `jitter` arm in `rom/augment.py` |
| `pod_augment_highmodes.py` | Summer strategy B: keep leading modes of a real snapshot, resample the high modes | Superseded by the Galerkin generators |
| `aug_experiment_strategyC.py` | Low-data study of strategy C on the isotropic BOX data | Superseded by `studies/5_low_data/`; BOX data no longer part of the project |
| `test_augmented_vae.py` | Trained the beta-VAE on one summer augmentation bundle, appended to `augment_results*.csv` | Summer tester; every later study trains through `rom.vae.train` |
| `train_augmented_vae.py` | Same as the tester, plus checkpoint and figures | Summer tester; superseded like the above |
| `beta_vae_bsp.py` | beta-VAE with a binned-spectral-power loss (isotropic BOX data) | Dead-end VAE variant |
| `beta_vae_multiscale.py` | Two beta-VAEs on large / small scales (BOX data) | Dead-end VAE variant |
| `beta_vae_nbands.py` | N beta-VAEs on N spectral bands (BOX data) | Dead-end VAE variant |
| `pod_augment_galerkin_ns_ensemble.py` | NS-Galerkin ensemble from very few real snapshots, scored on held-out data | Early NS experiment, superseded by the region map |
| `pod_augment_galerkin_ns_future.py` | NS-Galerkin used as a forward predictor | Early NS experiment (the project never predicts future steps) |
| `view_future_prediction.py` | Movie of the forward-prediction bundle | Only viewed the output of `pod_augment_galerkin_ns_future.py` |
| `pod_augment_galerkin_ns_bestcase.py` | Best-case NS augmentation bundle per dataset | Best-case experiment, superseded by the region map |
| `bestcase_train.py` | Trained on the best-case bundles | Best-case experiment, superseded by the region map |
| `run_bestcase_augmentation.py` | Driver of the best-case experiment | Best-case experiment, superseded by the region map |
| `make_convergence_excel.py` | Excel of the Re50-only convergence bundles | Superseded by `make_alpha0_excel.py` (all Re, all splits) |
| `wait_for_experiments2.sh` | Kept the Mac awake until `run_experiments2.py` finished | One-off watcher; `scripts/launch.sh` does this for any run |
| `wait_for_lowdata.sh` | Kept the Mac awake until `run_lowdata.py` finished | One-off watcher; replaced by `scripts/launch.sh` |
| `wait_and_build_ns_deck.sh` | Waited for the NS grid run, then resumed a Claude Code session to build a deck | One-off watcher, tied to one old session |
| `run_capacity_terminal.sh` | Ran `run_capacity.py` from Terminal with 6 threads, optionally after another run | Replaced by `scripts/launch.sh` (it also never ran `capacity_headroom.py`, see README) |
| `README_scripts.txt` | Per-script notes of the summer code | Replaced by `README.md` |
| `__pycache__/` | Compiled Python files (committed by accident) | Build junk; now in `.gitignore` |
| `.DS_Store` | macOS folder metadata | Junk; now in `.gitignore` |
| `augment_results.csv` | Summer tester results (isotropic run) | Old summer result, not read by any kept script |
| `augment_results_50Re_2plates.csv` | Summer tester results, Re50 2-plates | Old summer result |
| `augment_results_50Re_2plates_lessData.csv` | Summer tester results, Re50 2-plates, less data | Old summer result |
| `augment_results_c_newData.csv` | Summer tester results, strategy C | Old summer result |
| `augment_results_gmm.csv` | Summer tester results, GMM | Old summer result |
| `augment_results_isotropic.csv` | Summer tester results, isotropic data | Old summer result |
| `aug_lowdata_results.csv` | Summer low-data results | Old summer result |
| `aug_lowdata_isotropic_results.csv` | Summer low-data results, isotropic data | Old summer result |

## Merged into `rom/` (the file is gone, its code is not)

| File | Now in | What was dropped |
|---|---|---|
| `convergence_split.py` | `rom/split.py` | nothing |
| `convergence_csv.py` | `rom/results.py` | nothing |
| `pod_augment_galerkin_ns.py` | `rom/galerkin_ns.py` (model), `rom/data.py` (loader), `rom/pod.py` (POD), `rom/augment.py` (`save_bundle`) | its `main()` and CONFIG: the summer bundle generator `<base>_aug_galerkin_ns_5traj.mat` for the removed testers |
| `pod_augment_galerkin.py` | `rom/galerkin_data.py` | its `main()` and CONFIG (summer bundle generator), and its older `load_data` copy (no Alpha0 layout; identical to `rom.data.load_data` on every file it was used for) |

Shared functions that moved out of scripts that are still in `studies/` (the scripts
themselves are kept): `beta_vae.py` -> `rom/vae.py`, `rom/data.py`;
`re100_fraction_sweep.py` (`train`, `generate_pool`, `n_aug_for`) -> `rom/vae.py`, `rom/augment.py`;
`re100_conv_fraction_sweep.py` (`pod_val_error`, `assess_pod`, `draw_subset`) -> `rom/pod.py`, `rom/split.py`;
`run_region_map.py` (`load`, `draw_blocks`, `subset_pod`, `k_for`, `ceiling_and_project`, `fidelity`,
`fidelity_windows`, generator settings) -> `rom/data.py`, `rom/split.py`, `rom/pod.py`, `rom/augment.py`;
`run_lowdata.py` (`draw_subset`, `build_ops`, `model_at`, `k_of`, `TRUNCS`, `B_FIELDS`, `append`, `now`)
-> `rom/split.py`, `rom/galerkin_ns.py`, `rom/pod.py`, `rom/results.py`.

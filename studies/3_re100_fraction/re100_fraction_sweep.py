"""
Reproduce the SUMMER Re100 augmentation result and sweep the synthetic fraction.

Summer setup (Week 4 July report; bundles Data2PlatesGap1Re100_aug_galerkin.npz
and _aug_galerkin_ns.npz; checkpoints model_augVAE_..._lat5_b5e-03_ep500.pt):
    data        Data2PlatesGap1Re100.mat, 1000 snapshots, dt = 1, dx = dy = 0.08
    split       RANDOM 10 % validation with seed 7 -> 100 validation, 900 training
    generators  "galerkin"    data-identified quadratic model, 30 modes, ODE fit
                              (central differences, ridge 1e-6)     -> Ek 50.37 %
                "galerkin_ns" NS-projected model, 20 modes           -> Ek 49.44 %
                both: horizon 20, 20 RK4 sub-steps, IC kick 0.05, energy band 5 %,
                900 synthetic snapshots (1:1)
    baseline    900 real only                                        -> Ek 43.77 %
    network     latent 5, beta 5e-3, batch 32, 500 epochs, lr 3e-4, torch seed 7,
                normalisation + mean field from the real training snapshots,
                best-validation checkpoint, Ek on the real validation set

This script rebuilds exactly that (same seed, same split, same model, same rng
sequence, so the first 900 synthetic states are the summer ones) but generates
a LARGER synthetic pool once, and then trains the same network for several
synthetic FRACTIONS  f = n_aug / (n_real + n_aug), using the first n_aug
synthetic snapshots of the pool (nested sets):
    f = 0      ->    0 synthetic   (the baseline)
    f = 0.25   ->  300
    f = 0.5    ->  900             (the summer configuration)
    f = 0.667  -> 1800
    f = 0.75   -> 2700
One CSV row per fraction: ../../convergence/re100_fraction_sweep.csv

Usage:
    python3 re100_fraction_sweep.py                       # galerkin, latent 5, all fractions
    python3 re100_fraction_sweep.py galerkin_ns
    python3 re100_fraction_sweep.py galerkin --latent 15  # the convergence-study latent
    python3 re100_fraction_sweep.py --fractions 0,0.5     # subset of fractions
    python3 re100_fraction_sweep.py --dry-run             # generate + check vs the summer
                                                          # bundle, no training
Budget: ~3 min generation, then 500-epoch trainings on 900 / 1200 / 1800 / 2700 /
3600 snapshots (about 5 + 25 + 40 + 60 + 80 min on the MPS GPU); ~24 GB RAM peak.
"""
import os, sys, csv, time, datetime
import numpy as np

from rom import paths
from rom import augment                       # summer generator settings, generate_pool(), n_aug_for()
from rom import vae                            # train(): the recipe of every study
from rom.data import load_data
from rom.results import rewrite_with_row

_DATA = paths.DATA
_AUG  = paths.AUGMENTED
_CONV = paths.RESULTS

# ============ CONFIG (summer values; do not change to reproduce) ============
# The summer generator settings (Re 100, dt 1, 30 / 25 modes, ridge 1e-6, horizon 20,
# 20 RK4 sub-steps, IC kick 0.05, energy band 5 %, 20000 trajectories) are
# rom.augment.SUMMER_*; the network recipe (beta 5e-3, batch 32, 500 epochs, lr 3e-4,
# torch seed 7) is rom.vae.
DATA_FILE  = os.path.join(_DATA, augment.SUMMER_FILE)
GENERATOR  = "galerkin"           # "galerkin" (data regression, +6.6 pts) | "galerkin_ns" (NS equations, +5.7)
RNG_SEED   = augment.SUMMER_SEED   # 7: split AND trajectory rng (one generator, as in summer)
VAL_FRAC   = augment.SUMMER_VAL_FRAC
FRACTIONS  = [0.0, 0.25, 0.4, 0.45, 0.5, 0.55, 0.6]     # synthetic fraction of the training set
N_ROM      = dict(augment.SUMMER_N_ROM)   # {"galerkin": 30, "galerkin_ns": 25}

LATENT     = 5                 # summer network (the convergence study used 15 for Re100)
BETA       = vae.BETA
EPOCHS     = vae.EPOCHS
TORCH_SEED = vae.TORCH_SEED

VERIFY_SUMMER = True           # compare the first synthetic states with the summer bundle
SAVE_POOL     = False          # also save the synthetic pool as a bundle (~12 GB)
CSV_OUT       = os.path.join(_CONV, "re100_fraction_sweep.csv")
# ============================================================================


def _cli(argv):
    global GENERATOR, LATENT, FRACTIONS
    dry = False
    it = iter(argv)
    for a in it:
        if a in ("galerkin", "galerkin_ns"): GENERATOR = a
        elif a == "--latent":    LATENT = int(next(it))
        elif a == "--fractions": FRACTIONS = [float(x) for x in next(it).split(",")]
        elif a == "--dry-run":   dry = True
        else: raise SystemExit(f"unknown argument {a!r}")
    return dry


n_aug_for = augment.n_aug_for


def verify_against_summer(train_aug, n_check=5):
    f = os.path.join(_AUG, f"Data2PlatesGap1Re100_aug_{GENERATOR}.npz")
    if not os.path.exists(f):
        print(f"[verify]  no summer bundle {os.path.basename(f)} — skipped"); return None
    S = np.load(f, allow_pickle=True)
    old = np.asarray(S["train_aug"][:n_check], dtype=np.float32)
    new = train_aug[:n_check]
    rel = float(np.abs(old - new).max() / (np.abs(old).max() + 1e-12))
    same = rel < 1e-4
    print(f"[verify]  first {n_check} synthetic snapshots vs summer bundle: max rel. diff "
          f"{rel:.2e} -> {'IDENTICAL (summer states reproduced)' if same else 'DIFFERENT'}")
    return same


FIELDS = ["date", "generator", "fraction", "n_real", "n_aug", "n_train", "n_val", "n_rom",
          "rom_energy_pct", "n_traj", "keep_frac", "latent", "beta", "epochs", "seed",
          "Ek", "e", "detR", "best_val_loss", "gain_vs_f0", "summer_ref_Ek", "wall_s"]


def append_row(row):
    rewrite_with_row(CSV_OUT, FIELDS, row, makedirs=True)


def main():
    dry = _cli(sys.argv[1:])
    T0 = time.time()
    data, _ = load_data(DATA_FILE)
    Nt = len(data)
    rng = np.random.default_rng(RNG_SEED)             # same rng for split AND trajectories
    idx = rng.permutation(Nt)
    n_val = max(1, round(VAL_FRAC * Nt))
    val_idx, tr_idx = np.sort(idx[:n_val]), np.sort(idx[n_val:])
    train_real, val_real = data[tr_idx].copy(), data[val_idx].copy()
    del data
    n_real = len(tr_idx)
    fr = sorted(set(FRACTIONS))
    n_augs = [n_aug_for(f, n_real) for f in fr]
    n_max = max(n_augs)
    print(f"[sweep]  generator {GENERATOR}, latent {LATENT}: {n_real} real, {n_val} validation "
          f"(random, seed {RNG_SEED}); fractions {fr} -> synthetic {n_augs}", flush=True)

    train_aug = None
    info = {"n_rom": "", "rom_energy": float("nan"), "n_traj": "", "keep_frac": float("nan")}
    if n_max > 0:
        train_aug, info = augment.generate_pool(train_real, tr_idx, n_max, rng, GENERATOR, N_ROM[GENERATOR], DATA_FILE)
        if VERIFY_SUMMER: verify_against_summer(train_aug)
        if SAVE_POOL:
            os.makedirs(_AUG, exist_ok=True)
            out = augment.save_bundle(os.path.join(_AUG, f"Data2PlatesGap1Re100_aug_{GENERATOR}_pool{n_max}"),
                                 {"train_real": train_real, "train_aug": train_aug, "val_real": val_real,
                                  "comp_names": np.array(["u", "v"], dtype=object), "strategy": f"{GENERATOR}_pool",
                                  "tr_idx": tr_idx, "val_idx": val_idx, **{k: v for k, v in info.items()}})
            print(f"[sweep]  pool saved -> {out}")
    if dry:
        print(f"[sweep]  DRY RUN — no training  ({time.time()-T0:.0f} s)"); return

    summer = {"galerkin": {0.0: 43.77, 0.5: 50.37}, "galerkin_ns": {0.0: 43.77, 0.5: 49.44}}[GENERATOR]
    ek0 = None
    for f, n_aug in zip(fr, n_augs):
        print(f"\n[sweep]  ==== fraction {f:.3f}: {n_real} real + {n_aug} synthetic = {n_real+n_aug} ====", flush=True)
        aug_n = train_aug[:n_aug] if n_aug else np.empty((0,) + train_real.shape[1:], np.float32)
        ek, detR, vl, dt = vae.train(train_real, aug_n, val_real, tag=f"f={f:.2f} n={n_real+n_aug}",
                                     latent=LATENT, epochs=EPOCHS)
        if n_aug == 0: ek0 = ek
        row = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "generator": GENERATOR,
               "fraction": round(n_aug / (n_real + n_aug), 4), "n_real": n_real, "n_aug": n_aug,
               "n_train": n_real + n_aug, "n_val": n_val, "n_rom": info["n_rom"] if n_aug else "",
               "rom_energy_pct": round(100 * info["rom_energy"], 3) if n_aug else "",
               "n_traj": info["n_traj"] if n_aug else "", "keep_frac": round(info["keep_frac"], 4) if n_aug else "",
               "latent": LATENT, "beta": BETA, "epochs": EPOCHS, "seed": TORCH_SEED,
               "Ek": round(float(ek), 4), "e": round(1 - float(ek) / 100, 6), "detR": round(float(detR), 6),
               "best_val_loss": round(float(vl), 6),
               "gain_vs_f0": round(float(ek) - ek0, 3) if ek0 is not None else "",
               "summer_ref_Ek": summer.get(round(f, 3), ""), "wall_s": round(dt)}
        append_row(row)
        print(f"[sweep]  fraction {f:.3f}: Ek = {ek:.2f}%  detR = {detR:.3f}"
              + (f"   (summer: {summer[round(f,3)]}%)" if round(f, 3) in summer else "")
              + (f"   gain vs real-only {row['gain_vs_f0']:+} pts" if ek0 is not None and n_aug else ""), flush=True)
    print(f"\n[sweep]  done in {(time.time()-T0)/60:.0f} min  ->  {CSV_OUT}")


if __name__ == "__main__":
    main()

"""
GRID over the Galerkin model size and the POD-convergence tolerance for the
Re100 synthetic-fraction experiment (re100_conv_fraction_sweep.py, run for
every combination), with one CSV for everything.

For each number of modes K in MODES:
    step A  POD convergence with exactly K modes on the summer split (random 10 %
            validation, seed 7, pool 900): error vs n, n = 25..900 step 25, three
            draws per n  -> ../../convergence/re100_pod_convergence_k<K>.csv / .png
    for each tolerance in TOLERANCES (1 %, 2 %, ..., 10 %): the converged n is the
            smallest n from which every larger n stays within the tolerance of the
            full-pool error.  Several tolerances usually give the SAME n, so each
            distinct n_real is run only once and the row lists the tolerances it
            serves.
    step B  for each distinct n_real: draw the real subset, build the K-mode
            Galerkin model on it, generate the synthetic pool with the summer
            settings, train the summer network (latent 5, 500 epochs) at every
            synthetic fraction in FRACTIONS.

Output: ../../convergence/re100_tol_modes_grid.csv, one row per
(modes, n_real, fraction).  The driver is RESUMABLE: rows already in the CSV are
skipped, so it can be stopped and restarted at any time.

Usage:
    python3 run_re100_grid.py                       # NS-projected, modes 15 20 25 30, tol 1..10 %
    python3 run_re100_grid.py galerkin              # data regression (the run of 10-12 Sep 2026)
    python3 run_re100_grid.py --plan                # step A for every K and the list of
                                                    # cases + training count, no training
    python3 run_re100_grid.py --modes 20,30 --tols 0.02,0.05 --fractions 0,0.5 --latent 15

Budget: step A ~3 min per K.  Trainings: (#distinct n_real per K) x (#fractions);
each is 500 epochs on n_real (1 + f/(1-f)) snapshots, roughly 1.5 s per epoch per
1000 snapshots on the MPS GPU.  --plan prints the estimate before you commit.
"""
import os, sys, csv, time, datetime
import numpy as np

from rom import paths
from rom import augment                           # summer settings, generate_pool(), n_aug_for()
from rom import vae                               # train(): the recipe of every study
from rom.data import load_data
from rom.pod import assess_pod, pod_val_error
from rom.split import draw_pool_subset, RE100_DRAW_SEED
from rom.results import rewrite_with_row

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))

# ============ CONFIG ============
GENERATOR  = "galerkin_ns"                    # "galerkin_ns" (NS equations, random subset) | "galerkin" (data regression, blocks subset)
MODES      = [15, 20, 25, 30]                 # Galerkin model size = POD truncation of step A
TOLERANCES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
FRACTIONS  = [0.0, 0.25, 0.5, 2/3, 0.75]      # synthetic fraction of the training set
LATENT     = 5                                # summer network
CSV_OUT    = os.path.join(_CONV, "re100_tol_modes_grid.csv")
SEC_PER_EPOCH_PER_1000 = 1.5                  # for the time estimate only
DATA_FILE  = os.path.join(paths.DATA, augment.SUMMER_FILE)
# ================================

FIELDS = ["date", "generator", "modes", "tolerances", "n_real", "subset_mode", "fraction", "n_aug", "n_train",
          "n_val", "rom_energy_pct", "n_traj", "keep_frac", "latent", "epochs", "seed",
          "Ek", "e", "detR", "gain_vs_f0", "POD_Ek_max_nreal", "POD_Ek_max_pool", "wall_s"]


def _cli(argv):
    global GENERATOR, MODES, TOLERANCES, FRACTIONS, LATENT
    plan = False
    it = iter(argv)
    for a in it:
        if a in ("galerkin", "galerkin_ns"): GENERATOR = a
        elif a == "--modes":     MODES = [int(x) for x in next(it).split(",")]
        elif a == "--tols":      TOLERANCES = [float(x) for x in next(it).split(",")]
        elif a == "--fractions": FRACTIONS = [float(x) for x in next(it).split(",")]
        elif a == "--latent":    LATENT = int(next(it))
        elif a == "--plan":      plan = True
        else: raise SystemExit(f"unknown argument {a!r}")
    return plan


def converged_n(rows, tol):
    """Smallest n from which every larger n is within tol of the full-pool error."""
    e_full = rows[-1]["e_mean"]
    for r in rows:
        if all(rr["e_mean"] <= (1 + tol) * e_full for rr in rows if rr["n"] >= r["n"]):
            return r["n"]
    return rows[-1]["n"]


def load_done():
    if not os.path.exists(CSV_OUT):
        return set()
    with open(CSV_OUT, newline="") as f:
        return {(r["generator"], int(r["modes"]), int(r["n_real"]), int(r["n_aug"]), int(r["latent"]))
                for r in csv.DictReader(f)}


def append_row(row):
    rewrite_with_row(CSV_OUT, FIELDS, row)


def main():
    plan = _cli(sys.argv[1:])
    n_rom = dict(augment.SUMMER_N_ROM)                # model size of the shared generator (set per K below)
    fr = sorted(set(FRACTIONS)); tols = sorted(set(TOLERANCES))
    T0 = time.time()
    data, _ = load_data(DATA_FILE)
    Nt = len(data)
    rng = np.random.default_rng(augment.SUMMER_SEED)
    idx = rng.permutation(Nt); n_val = max(1, round(augment.SUMMER_VAL_FRAC * Nt))
    val_idx, pool_idx = np.sort(idx[:n_val]), np.sort(idx[n_val:])
    pool, val = data[pool_idx].copy(), data[val_idx].copy()
    del data
    n_pool = len(pool_idx)
    subset_mode = "blocks" if GENERATOR == "galerkin" else "random"
    done = load_done()
    print(f"[grid]  generator {GENERATOR}, latent {LATENT}, modes {MODES}, tolerances "
          f"{[f'{100*t:g}%' for t in tols]}, fractions {fr}; pool {n_pool}, validation {n_val}; "
          f"{len(done)} case(s) already in {os.path.basename(CSV_OUT)}", flush=True)

    # ---- step A for every K, then the list of cases ----
    cases = []                                        # (K, n_real, [tols], e_pool)
    for K in MODES:
        print(f"\n[grid]  ==== STEP A: POD convergence with {K} modes ====", flush=True)
        rows, _ = assess_pod(pool, val, K)
        by_n = {}
        for t in tols:
            by_n.setdefault(converged_n(rows, t), []).append(t)
        e_pool = rows[-1]["e_mean"]
        for n_real, tl in sorted(by_n.items()):
            cases.append((K, n_real, tl, e_pool))
    n_train_total = 0; secs = 0.0
    print(f"\n[grid]  PLAN ({len(cases)} distinct (modes, n_real) cases x {len(fr)} fractions):")
    for K, n_real, tl, e_pool in cases:
        n_augs = [augment.n_aug_for(f, n_real) for f in fr]
        todo = [(f, na) for f, na in zip(fr, n_augs) if (GENERATOR, K, n_real, na, LATENT) not in done]
        est = sum(vae.EPOCHS * SEC_PER_EPOCH_PER_1000 * (n_real + na) / 1000 for _, na in todo)
        n_train_total += len(todo); secs += est
        print(f"    K={K:2d}  n_real={n_real:3d}  tolerances {', '.join(f'{100*t:g}%' for t in tl):28s} "
              f"synthetic {n_augs}   {len(todo)} training(s) to do (~{est/60:.0f} min)")
    print(f"[grid]  => {n_train_total} trainings, roughly {secs/3600:.1f} h", flush=True)
    if plan:
        print(f"[grid]  --plan: stopping here ({time.time()-T0:.0f} s)"); return

    # ---- step B ----
    for K, n_real, tl, e_pool in cases:
        n_rom[GENERATOR] = K
        n_augs = [augment.n_aug_for(f, n_real) for f in fr]
        todo = [(f, na) for f, na in zip(fr, n_augs) if (GENERATOR, K, n_real, na, LATENT) not in done]
        if not todo:
            print(f"\n[grid]  K={K} n_real={n_real}: all fractions already done — skipping"); continue
        print(f"\n[grid]  ======== K = {K} modes, n_real = {n_real} (tolerances "
              f"{', '.join(f'{100*t:g}%' for t in tl)}), {subset_mode} subset ========", flush=True)
        sub = draw_pool_subset(n_pool, n_real, subset_mode, np.random.default_rng(RE100_DRAW_SEED))
        train_real, sub_idx = pool[sub], pool_idx[sub]
        e_sub = pod_val_error(train_real, val, K)
        n_max = max(na for _, na in todo)
        gen_rng = np.random.default_rng(augment.SUMMER_SEED + K)          # trajectory rng for this case
        try:
            train_aug, info = (augment.generate_pool(train_real, sub_idx, n_max, gen_rng, GENERATOR, n_rom[GENERATOR], DATA_FILE) if n_max else
                               (None, {"rom_energy": float("nan"), "n_traj": "", "keep_frac": float("nan")}))
        except Exception as err:                                  # e.g. the model cannot fill the pool
            msg = (f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {GENERATOR}  K={K}  n_real={n_real}: "
                   f"generation failed ({err}) — case skipped, it will be retried on the next launch")
            print(f"\n[grid]  !! {msg}", flush=True)
            with open(os.path.join(_CONV, "re100_tol_modes_grid_failures.log"), "a") as lf:
                lf.write(msg + "\n")
            continue
        ek0 = None
        # the real-only value for gain_vs_f0: from this run, or from an earlier row of this case
        if os.path.exists(CSV_OUT):
            for r in csv.DictReader(open(CSV_OUT, newline="")):
                if (r["generator"], int(r["modes"]), int(r["n_real"]), int(r["n_aug"]), int(r["latent"])) == (GENERATOR, K, n_real, 0, LATENT):
                    ek0 = float(r["Ek"])
        for f, n_aug in zip(fr, n_augs):
            if (f, n_aug) not in todo: continue
            print(f"\n[grid]  ---- K={K} n_real={n_real} fraction {f:.3f}: + {n_aug} synthetic ----", flush=True)
            aug_n = train_aug[:n_aug] if n_aug else np.empty((0,) + train_real.shape[1:], np.float32)
            ek, detR, vl, dt = vae.train(train_real, aug_n, val, tag=f"K={K} n={n_real} f={f:.2f}",
                                         latent=LATENT, epochs=vae.EPOCHS)
            if n_aug == 0: ek0 = ek
            append_row({"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "generator": GENERATOR,
                        "modes": K, "tolerances": " ".join(f"{100*t:g}%" for t in tl), "n_real": n_real,
                        "subset_mode": subset_mode, "fraction": round(n_aug / (n_real + n_aug), 4),
                        "n_aug": n_aug, "n_train": n_real + n_aug, "n_val": n_val,
                        "rom_energy_pct": round(100 * info["rom_energy"], 3) if n_aug else "",
                        "n_traj": info["n_traj"] if n_aug else "", "keep_frac": round(info["keep_frac"], 4) if n_aug else "",
                        "latent": LATENT, "epochs": vae.EPOCHS, "seed": vae.TORCH_SEED,
                        "Ek": round(float(ek), 4), "e": round(1 - float(ek) / 100, 6), "detR": round(float(detR), 6),
                        "gain_vs_f0": round(float(ek) - ek0, 3) if (ek0 is not None and n_aug) else "",
                        "POD_Ek_max_nreal": round(100 * (1 - e_sub), 3), "POD_Ek_max_pool": round(100 * (1 - e_pool), 3),
                        "wall_s": round(dt)})
            print(f"[grid]  K={K} n_real={n_real} f={f:.3f}: Ek = {ek:.2f}%  detR = {detR:.3f}"
                  + (f"   gain {float(ek)-ek0:+.2f} pts" if ek0 is not None and n_aug else "")
                  + f"   [{(time.time()-T0)/60:.0f} min elapsed]", flush=True)
        del train_aug
    print(f"\n[grid]  all done in {(time.time()-T0)/3600:.1f} h  ->  {CSV_OUT}")


if __name__ == "__main__":
    main()

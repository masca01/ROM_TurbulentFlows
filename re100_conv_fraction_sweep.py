"""
Re100: POD convergence at the SUMMER mode counts, then the synthetic-fraction
sweep from the CONVERGED number of real snapshots.

Step A — POD convergence at a fixed number of modes.  The summer generators
kept 30 modes (data-identified Galerkin, 74 % of the 900-snapshot training
energy) and 20 modes (NS-projected, 63 %).  On the summer split (random 10 %
validation, seed 7, pool of 900) the POD basis of a random subset of n snapshots
is truncated to exactly K modes and the unexplained energy of the validation set
is measured, n = 25, 50, ..., 900, three draws per n.  The converged n is the
smallest n whose mean error is within CONV_TOL of the full-pool error.
    -> ../../convergence/re100_pod_convergence_k<K>.csv  (+ .png)

Step B — fraction sweep from n_conv real snapshots.  N_REAL real snapshots are
drawn from the pool, the Galerkin model is built on THEM ONLY (POD + model),
a synthetic pool is generated with the summer settings, and the summer network
(latent 5, beta 5e-3, batch 32, 500 epochs, lr 3e-4, seed 7) is trained at the
synthetic fractions f = n_aug / (n_real + n_aug) of re100_fraction_sweep.py.
    -> ../../convergence/re100_conv_fraction_sweep.csv

Subset drawing.  The NS-projected generator needs no time derivatives, so the
subset can be a plain random draw.  The data-identified generator fits da/dt by
central differences on CONSECUTIVE snapshots, which a random subset almost never
contains; for it the subset is drawn as random BLOCKS of BLOCK consecutive
snapshots (same count, same random coverage of the record) so the fit has
samples.  The subset mode used is recorded in the CSV.

Usage:
    python3 re100_conv_fraction_sweep.py                    # data regression, 25 modes, A then B
    python3 re100_conv_fraction_sweep.py galerkin_ns        # NS equations, 25 modes, random subset
    python3 re100_conv_fraction_sweep.py --assess-only      # step A only (fast, ~10 min)
    python3 re100_conv_fraction_sweep.py --n-real 300       # skip the auto choice
    python3 re100_conv_fraction_sweep.py galerkin --modes 20  # 30 -> 20 modes (model + step A)
    python3 re100_conv_fraction_sweep.py --latent 15 --tol 0.02 --fractions 0,0.5,0.75
"""
import os, sys, csv, time, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re100_fraction_sweep as rs               # summer settings, generate_pool(), train()
import pod_augment_galerkin as pg

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))

# ============ CONFIG ============
GENERATOR  = "galerkin"         # "galerkin" (data regression, blocks subset) | "galerkin_ns" (NS equations, random subset)
N_MODES    = 25                 # Galerkin modes for the model AND step A (25 = the NS run of experiment 2);
                                # None = the count in re100_fraction_sweep.N_ROM
K_ASSESS   = None               # modes for step A; None = the generator's summer count
N_STEP     = 25                 # step A: n = 25, 50, ... , 900
M_REPS     = 3                  # draws per n
DRAW_SEED  = 11                 # subset draws (step A and step B)
CONV_TOL   = 0.05               # converged when mean error <= (1 + tol) * full-pool error
N_REAL     = None               # step B real snapshots; None = converged n from step A
FRACTIONS  = [0.0, 0.25, 0.5, 2/3, 0.75]
BLOCK      = 10                 # block length for the data-identified generator's subset
LATENT     = 5                  # summer network
CSV_OUT    = os.path.join(_CONV, "re100_conv_fraction_sweep.csv")
# ================================


def _cli(argv):
    global GENERATOR, K_ASSESS, N_REAL, FRACTIONS, LATENT, CONV_TOL, N_MODES
    opts = {"assess_only": False}
    it = iter(argv)
    for a in it:
        if a in ("galerkin", "galerkin_ns"): GENERATOR = a
        elif a == "--modes":      N_MODES = int(next(it))               # generator AND step A
        elif a == "--k":          K_ASSESS = int(next(it))            # step A only
        elif a == "--n-real":     N_REAL = int(next(it))
        elif a == "--latent":     LATENT = int(next(it))
        elif a == "--tol":        CONV_TOL = float(next(it))
        elif a == "--fractions":  FRACTIONS = [float(x) for x in next(it).split(",")]
        elif a == "--assess-only": opts["assess_only"] = True
        else: raise SystemExit(f"unknown argument {a!r}")
    return opts


# ------------------------------ step A: POD ------------------------------

def pod_val_error(train, val, k):
    """Unexplained energy fraction of `val` after projection on the top-k POD
    modes of `train` (subset mean removed), as in pod_convergence.py."""
    k = min(k, len(train) - 1)                       # a subset of n snapshots has at most n-1 modes
    X = train.reshape(len(train), -1).astype(np.float32); m = X.mean(0); Xc = X - m
    G = (Xc @ Xc.T).astype(np.float64); w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1][:k]; w, V = np.clip(w[o], 1e-30, None), V[:, o]
    U = (Xc.T @ (V / np.sqrt(w)[None, :])).astype(np.float32)          # D x k, orthonormal
    Xv = val.reshape(len(val), -1).astype(np.float32) - m
    P = (Xv @ U) @ U.T
    return float(((Xv - P) ** 2).sum() / (Xv ** 2).sum())


def assess_pod(pool, val, k):
    n_pool = len(pool)
    n_vals = list(range(N_STEP, n_pool + 1, N_STEP))
    if n_vals[-1] != n_pool: n_vals.append(n_pool)
    rng = np.random.default_rng(DRAW_SEED)
    rows, t0 = [], time.time()
    for n in n_vals:
        reps = 1 if n == n_pool else M_REPS
        errs = []
        for _ in range(reps):
            sub = np.sort(rng.choice(n_pool, size=n, replace=False)) if n < n_pool else np.arange(n_pool)
            errs.append(pod_val_error(pool[sub], val, k))
        rows.append({"n": n, "k": k, "e_mean": float(np.mean(errs)), "e_std": float(np.std(errs)),
                     "Ek_max": 100 * (1 - float(np.mean(errs))), "reps": reps})
        print(f"  k={k}  n={n:4d}  e = {np.mean(errs):.4f} +/- {np.std(errs):.4f}   [{time.time()-t0:.0f}s]", flush=True)
    e_full = rows[-1]["e_mean"]
    conv = {}
    for tol in (0.01, 0.02, 0.05, 0.10):
        ok = [r["n"] for r in rows if r["e_mean"] <= (1 + tol) * e_full]
        # smallest n from which ALL larger n are also within tol (monotone plateau)
        n_conv = None
        for r in rows:
            if all(rr["e_mean"] <= (1 + tol) * e_full for rr in rows if rr["n"] >= r["n"]):
                n_conv = r["n"]; break
        conv[tol] = n_conv
    out = os.path.join(_CONV, f"re100_pod_convergence_k{k}.csv")
    os.makedirs(_CONV, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.errorbar([r["n"] for r in rows], [r["e_mean"] for r in rows], yerr=[r["e_std"] for r in rows],
                    marker="o", ms=3.5, lw=1.6, color="#2a78d6", capsize=2)
        for tol, ls in ((0.05, "--"), (0.02, ":")):
            ax.axhline((1 + tol) * e_full, color="#9a9a94", lw=0.8, ls=ls)
            if conv[tol]: ax.axvline(conv[tol], color="#eb6834", lw=0.8, ls=ls)
        ax.set_xscale("log"); ax.set_xlabel("n training snapshots (random subset of the 900 pool)")
        ax.set_ylabel(f"unexplained validation energy, top-{k} modes")
        ax.set_title(f"Re 100, POD truncated to {k} modes: converged n = {conv[0.05]} (5%), {conv[0.02]} (2%)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.5); fig.tight_layout(); fig.savefig(out[:-4] + ".png", dpi=150)
    except Exception as err:
        print(f"  (plot skipped: {err})")
    print(f"\n[assess]  k = {k}: full-pool error {e_full:.4f} (Ek_max {100*(1-e_full):.2f}%).  "
          f"Converged n: " + ", ".join(f"{100*t:g}% -> {n}" for t, n in conv.items()))
    print(f"[assess]  saved -> {out}")
    return rows, conv


# ------------------------------ step B: sweep ------------------------------

def draw_subset(n_pool, n_real, mode, rng):
    if mode == "random" or n_real >= n_pool:
        return np.sort(rng.choice(n_pool, size=n_real, replace=False)) if n_real < n_pool else np.arange(n_pool)
    n_blocks = int(np.ceil(n_real / BLOCK))
    starts = np.sort(rng.choice(n_pool - BLOCK + 1, size=n_blocks, replace=False))
    idx = np.unique(np.concatenate([np.arange(s, s + BLOCK) for s in starts]))
    while len(idx) < n_real:                                    # top up overlaps
        extra = rng.choice(np.setdiff1d(np.arange(n_pool), idx), size=n_real - len(idx), replace=False)
        idx = np.unique(np.concatenate([idx, extra]))
    return np.sort(rng.choice(idx, size=n_real, replace=False)) if len(idx) > n_real else idx


FIELDS = ["date", "generator", "subset_mode", "n_real", "n_pool", "fraction", "n_aug", "n_train", "n_val",
          "n_rom", "rom_energy_pct", "n_traj", "keep_frac", "latent", "epochs", "seed",
          "Ek", "e", "detR", "gain_vs_f0", "POD_Ek_max_nreal", "POD_Ek_max_pool", "n_conv_tol", "wall_s"]


def append_row(row):
    rows = list(csv.DictReader(open(CSV_OUT, newline=""))) if os.path.exists(CSV_OUT) else []
    rows.append(row)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in FIELDS})


def main():
    opts = _cli(sys.argv[1:])
    rs.GENERATOR, rs.LATENT = GENERATOR, LATENT          # the shared generator/trainer use these
    if N_MODES:
        rs.N_ROM[GENERATOR] = N_MODES
    k = K_ASSESS or rs.N_ROM[GENERATOR]
    T0 = time.time()
    data, _ = pg.load_data(rs.DATA_FILE)
    Nt = len(data)
    rng = np.random.default_rng(rs.RNG_SEED)             # summer split (same rng recipe)
    idx = rng.permutation(Nt); n_val = max(1, round(rs.VAL_FRAC * Nt))
    val_idx, pool_idx = np.sort(idx[:n_val]), np.sort(idx[n_val:])
    pool, val = data[pool_idx].copy(), data[val_idx].copy()
    del data
    n_pool = len(pool_idx)
    print(f"[conv-sweep]  generator {GENERATOR} (K = {k}), pool {n_pool}, validation {n_val} "
          f"(random, seed {rs.RNG_SEED})", flush=True)

    # ---- step A ----
    print(f"\n[conv-sweep]  STEP A: POD convergence with {k} modes, n = {N_STEP}..{n_pool} step {N_STEP}, "
          f"{M_REPS} draws", flush=True)
    rows, conv = assess_pod(pool, val, k)
    e_by_n = {r["n"]: r["e_mean"] for r in rows}
    if opts["assess_only"]:
        print(f"\n[conv-sweep]  assess-only: done in {time.time()-T0:.0f} s"); return

    # ---- step B ----
    n_real = N_REAL or conv[CONV_TOL] or n_pool
    subset_mode = "blocks" if GENERATOR == "galerkin" else "random"
    sub = draw_subset(n_pool, n_real, subset_mode, np.random.default_rng(DRAW_SEED))
    train_real, sub_idx = pool[sub], pool_idx[sub]        # sub_idx: positions in the record (for the ODE fit)
    n_aug_list = [rs.n_aug_for(f, n_real) for f in sorted(set(FRACTIONS))]
    print(f"\n[conv-sweep]  STEP B: n_real = {n_real} ({subset_mode} subset"
          + (f", blocks of {BLOCK}" if subset_mode == "blocks" else "") +
          f"; converged at tol {CONV_TOL:g}: {conv[CONV_TOL]}), fractions -> synthetic {n_aug_list}", flush=True)
    e_sub = pod_val_error(train_real, val, k)
    print(f"[conv-sweep]  POD ceiling of THIS subset with {k} modes: Ek_max {100*(1-e_sub):.2f}%  "
          f"(pool: {100*(1-e_by_n[n_pool]):.2f}%)", flush=True)

    n_max = max(n_aug_list)
    train_aug, info = (rs.generate_pool(train_real, sub_idx, n_max, rng) if n_max else
                       (None, {"n_rom": "", "rom_energy": float("nan"), "n_traj": "", "keep_frac": float("nan")}))
    ek0 = None
    for f, n_aug in zip(sorted(set(FRACTIONS)), n_aug_list):
        print(f"\n[conv-sweep]  ==== fraction {f:.3f}: {n_real} real + {n_aug} synthetic ====", flush=True)
        aug_n = train_aug[:n_aug] if n_aug else np.empty((0,) + train_real.shape[1:], np.float32)
        ek, detR, vl, dt = rs.train(train_real, aug_n, val, tag=f"n_real={n_real} f={f:.2f}")
        if n_aug == 0: ek0 = ek
        append_row({"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "generator": GENERATOR,
                    "subset_mode": subset_mode, "n_real": n_real, "n_pool": n_pool,
                    "fraction": round(n_aug / (n_real + n_aug), 4), "n_aug": n_aug, "n_train": n_real + n_aug,
                    "n_val": n_val, "n_rom": info["n_rom"] if n_aug else "",
                    "rom_energy_pct": round(100 * info["rom_energy"], 3) if n_aug else "",
                    "n_traj": info["n_traj"] if n_aug else "", "keep_frac": round(info["keep_frac"], 4) if n_aug else "",
                    "latent": LATENT, "epochs": rs.EPOCHS, "seed": rs.TORCH_SEED,
                    "Ek": round(float(ek), 4), "e": round(1 - float(ek) / 100, 6), "detR": round(float(detR), 6),
                    "gain_vs_f0": round(float(ek) - ek0, 3) if ek0 is not None else "",
                    "POD_Ek_max_nreal": round(100 * (1 - e_sub), 3), "POD_Ek_max_pool": round(100 * (1 - e_by_n[n_pool]), 3),
                    "n_conv_tol": f"{conv[CONV_TOL]} @ {CONV_TOL:g}", "wall_s": round(dt)})
        print(f"[conv-sweep]  fraction {f:.3f}: Ek = {ek:.2f}%  detR = {detR:.3f}"
              + (f"   gain vs real-only {float(ek)-ek0:+.2f} pts" if ek0 is not None and n_aug else ""), flush=True)
    print(f"\n[conv-sweep]  done in {(time.time()-T0)/60:.0f} min  ->  {CSV_OUT}")


if __name__ == "__main__":
    main()

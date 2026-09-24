"""
POD helpers shared by the studies.

compute_pod / reconstruct     POD by the method of snapshots (pod_augment_galerkin*.py, identical copies)
subset_pod / k_for /
ceiling_and_project           POD of a real subset, modes for an energy level, POD ceiling (run_region_map.py)
TRUNCS / k_of                 the truncations of the low-data screen (run_lowdata.py)
pod_val_error / assess_pod    POD convergence at a fixed number of modes (re100_conv_fraction_sweep.py)
"""
import os, csv, time
import numpy as np

from . import paths
from .split import RE100_DRAW_SEED


def compute_pod(train, rank=None):
    """POD by the method of snapshots.  When there are far more grid points than
    snapshots (the full 2-plates case), it is much cheaper to eigen-decompose
    the small [Ntr x Ntr] Gram matrix than to SVD the huge data matrix.

    train[Ntr, C, H, W] -> (x_mean, Xc, Wp, S, A, shape), where
      x_mean : temporal mean field (flattened)
      Xc     : mean-subtracted snapshots (flattened), kept so I can rebuild the
               spatial modes as U = Xc.T @ Wp without ever storing U
      Wp, S  : the pieces needed to go between coefficients and snapshots
      A      : the POD coefficients of the training snapshots
    """
    Ntr, C, H, W = train.shape
    X  = train.reshape(Ntr, C * H * W)
    x_mean = X.mean(axis=0)
    Xc = (X - x_mean[None, :]).astype(np.float32)
    G  = (Xc @ Xc.T).astype(np.float64)                  # Ntr x Ntr Gram matrix
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]                           # largest energy first
    w = np.clip(w[order], 0.0, None); V = V[:, order]
    S = np.sqrt(w)
    keep = S > (S[0] * 1e-10 if S.size else 0.0)          # drop numerical zeros
    S, V = S[keep], V[:, keep]
    if rank is not None:
        S, V = S[:rank], V[:, :rank]
    A  = (V * S[None, :])                                 # coefficients
    Wp = (V / S[None, :]).astype(np.float32)              # used to reconstruct
    return x_mean.astype(np.float32), Xc, Wp, S, A, (C, H, W)


def reconstruct(x_mean, Xc, Wp, A_new, shape):
    """Turn a set of POD coefficients back into full snapshots.  Uses
    X_new = x_mean + U @ A_new.T with U = Xc.T @ Wp, so the (large) modes U are
    never actually formed in memory."""
    C, H, W = shape
    M = Wp @ A_new.T.astype(np.float32)
    X_new = Xc.T @ M
    X_new += x_mean[:, None]
    return X_new.T.reshape(-1, C, H, W).astype(np.float32)


# ------------------------------ real subsets (region map) ------------------------------

def subset_pod(train_real):
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    e_cum = np.cumsum(S ** 2) / (S ** 2).sum()
    return dict(x_mean=x_mean, Xc=Xc, Wp=Wp, S=S, A=A, shape=shape, e_cum=e_cum)


def k_for(P, level):
    return int(min(np.searchsorted(P["e_cum"], level) + 1, P["A"].shape[1]))


def ceiling_and_project(P, K, val_real, windows_data):
    """POD ceiling of the validation set with K modes, and K-mode coefficients of the
    fidelity windows (same centring as the network: the subset's mean)."""
    U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float32)                 # D x K, orthonormal
    Xv = val_real.reshape(len(val_real), -1) - P["x_mean"]
    Pv = (Xv @ U) @ U.T
    ceiling = 100.0 * (1.0 - float(((Xv - Pv) ** 2).sum() / (Xv ** 2).sum()))
    del Pv, Xv
    win_A = [((w.reshape(len(w), -1) - P["x_mean"]) @ U).astype(np.float64) for w in windows_data]
    del U
    return ceiling, win_A


# ------------------------------ truncations of the low-data study ------------------------------

TRUNCS = [("95%", None), ("99%", None), ("K5", 5), ("K10", 10), ("K15", 15), ("K20", 20)]


def k_of(P, trunc, n):
    name, fixed = trunc
    K = k_for(P, 0.95) if name == "95%" else k_for(P, 0.99) if name == "99%" else fixed
    return int(min(K, P["A"].shape[1], n - 1))


# ------------------------------ POD convergence at K modes (Re100) ------------------------------

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


ASSESS_N_STEP = 25    # n = 25, 50, ... , |pool|
ASSESS_M_REPS = 3     # draws per n


def assess_pod(pool, val, k, n_step=ASSESS_N_STEP, m_reps=ASSESS_M_REPS, draw_seed=RE100_DRAW_SEED):
    """Unexplained validation energy with the top-k POD modes of random subsets of n pool
    snapshots, n = n_step .. |pool|; converged n per tolerance.  Writes
    RESULTS/re100_pod_convergence_k<k>.csv (+ .png)."""
    _CONV = paths.RESULTS
    n_pool = len(pool)
    n_vals = list(range(n_step, n_pool + 1, n_step))
    if n_vals[-1] != n_pool: n_vals.append(n_pool)
    rng = np.random.default_rng(draw_seed)
    rows, t0 = [], time.time()
    for n in n_vals:
        reps = 1 if n == n_pool else m_reps
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
        from matplotlib.figure import Figure        # no pyplot: the backend of the caller is left alone
        fig = Figure(figsize=(6.5, 4)); ax = fig.subplots()
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

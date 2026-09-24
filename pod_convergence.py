"""
POD spatial-mode CONVERGENCE study.

Question: as I build the POD basis from more and more snapshots, how well does
that basis reconstruct *unseen* snapshots?  I hold out 10% of the record as
validation, then build the POD from n randomly chosen training snapshots and
measure the relative reconstruction error of the validation set.  Sweeping n and
repeating m times (different random draws) traces the convergence curve.

Split (done once, fixed seed):
    - 10% of all snapshots -> validation set  V   (matrix X_V, columns=snapshots)
    - 90% of all snapshots -> training pool    P

For each n in N_LIST and each repetition j = 1..M_REPS:
    - draw a random subset S of n snapshots from P
    - CENTERED POD: subtract the subset mean  x_bar_S,  Xc = X_S - x_bar_S
      thin SVD  Xc = U S W^T  -> U = full spatial-mode basis (rank r <= n-1)
    - project the (same-mean-centered) validation set onto that basis and take
      the SQUARED relative Frobenius error (= unexplained ENERGY fraction, the
      same kind of quantity as the NN's 1 - Ek/100, so the two are comparable):

          e = || Xc_V - U U^T Xc_V ||_F^2 / || Xc_V ||_F^2 ,   Xc_V = X_V - x_bar_S

      (U U^T Xc_V is computed by the method of snapshots so we never form U in
       D-space:  U U^T Xc_V = Xc (Xc^T Xc)^+ Xc^T Xc_V .)

Then e_n = mean over the M_REPS draws, with std for error bars, plotted vs n
(log-y).  As n grows the POD subspace spans more of the attractor and e_n falls
and converges.

Usage:  python3 pod_convergence.py <file.mat> <T_MAX> <N_MODES> [E_LEVEL] [SPLIT]
        E_LEVEL (e.g. 95 or 99) is the energy level that N_MODES retains; it tags
        the outputs so several truncations of one dataset coexist.
        SPLIT = random (default: validation = random 10%), tail (validation =
        the LAST 10% of the record) tail5 (the LAST 5%) or tail2.5 (the LAST 2.5%); see convergence_split.py.

Output: ../DATA/AUGMENTED[_TAILVAL]/<base>_pod<E_LEVEL>_convergence.mat
        + columns POD<E_LEVEL>_* in the dataset folder's
          convergence_last_point[_tailval].csv
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from pod_augment_galerkin_ns import load_data
import convergence_split as cs

# ------------ Folder layout ------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))

# ============ CONFIG ============
# DATA_FILE and the snapshot cap can be overridden on the command line so the
# same script runs over the whole Alpha0 Re-sweep, e.g.
#   python3 pod_convergence.py .../Alpha0/dataRe60Alpha0_2.mat 1500
_DEFAULT_FILE = ("/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/"
                 "DATA/Alpha0/dataRe50Alpha0_2.mat")
DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_FILE
COMP_IDX  = None      # None = auto-detect velocity components

# temporal subsampling (for the big Re/alpha-sweep files; 1 = use every snapshot).
# The Alpha0 datasets have 5000 (Re50) or 1500 (Re60/70/80) snapshots; capping
# every dataset at T_MAX = 1500 gives all of them the SAME snapshot budget so the
# convergence curves are comparable across Re.
T_STRIDE  = 1         # keep every snapshot (dt = 0.2)
T_MAX     = int(sys.argv[2]) if len(sys.argv) > 2 else 1500   # common budget

N_STEP    = 25        # n sweep step: n = 50, 100, 150, ... up to |pool| (fine; POD is fast)
N_LIST    = None      # set an explicit list to override the auto n-sweep
M_REPS    = 5         # random POD draws averaged per n (the parameter m)
# truncate U to the top-k modes (None = keep the full basis).  Per-dataset: the
# driver passes the k that retains >= E_LEVEL % of the pool energy (from
# assess_pod_modes.py -> ../../convergence/pod_modes_assessment.csv).
N_MODES   = int(sys.argv[3]) if len(sys.argv) > 3 else 25
# E_LEVEL = the energy level (%) that N_MODES retains.  It is only a LABEL, used
# to tag the outputs so several truncations of the same dataset coexist on disk:
#   ../DATA/AUGMENTED/<base>_pod95_convergence.mat   and   ..._pod99_...
#   CSV columns  POD95_n / POD95_e_mean / POD95_modes   and   POD99_*
# (no level given -> untagged  <base>_pod_convergence.mat  and  POD_* columns)
E_LEVEL   = sys.argv[4] if len(sys.argv) > 4 else None
# SPLIT: how the validation set is chosen — "random" (10% at random, the original
# results) or "tail" (the LAST 10% of the record).  Each mode writes to its OWN
# folder / CSV so the two never overwrite each other (convergence_split.py).
SPLIT     = cs.check_mode(sys.argv[5] if len(sys.argv) > 5 else "random")
VAL_FRAC = cs.val_frac(SPLIT)   # validation fraction of THIS split mode (0.10 or 0.025)
_AUG_DIR  = cs.aug_dir(SPLIT)

RANK_TOL  = 1e-10     # drop POD directions with eigenvalue < RANK_TOL * lambda_max
SPLIT_SEED = 7        # seed for the 90/10 split (random mode only)
DRAW_SEED  = 11       # seed for the random subset draws
STRATEGY  = f"pod{E_LEVEL}_convergence" if E_LEVEL else "pod_convergence"
CSV_TAG   = f"POD{E_LEVEL}" if E_LEVEL else "POD"     # CSV column prefix
# ================================


def pod_project_error(Xc_sub, Xc_val, n_modes=None):
    """SQUARED relative Frobenius error of projecting Xc_val onto span(Xc_sub)
    (= the fraction of validation ENERGY the POD basis fails to capture, the
    same kind of quantity as the NN's 1 - Ek/100).

    Method of snapshots: the orthogonal projector onto the column space of the
    centered subset Xc_sub is  P = Xc_sub (Xc_sub^T Xc_sub)^+ Xc_sub^T, so
    P Xc_val is built without ever forming U (D x r) explicitly.  Truncating the
    eigen-spectrum of the small n x n Gram matrix at RANK_TOL keeps exactly the
    POD directions with real energy (== keeping all columns of U of the SVD).

    If n_modes is given, U is further truncated to the top-k modes (largest
    eigenvalues == most energetic POD directions); if fewer than k modes carry
    real energy (small n), all available modes are used.  n_modes=None keeps the
    full basis.  Returns (relative_error, modes_kept).
    """
    G = Xc_sub.T @ Xc_sub                              # n x n Gram matrix
    lam, Wv = np.linalg.eigh(G)                        # ascending, symmetric PSD
    keep = lam > (lam[-1] * RANK_TOL)
    lam_k = lam[keep]
    Wk = Wv[:, keep]                                   # n x r  (ascending energy)
    if n_modes is not None and Wk.shape[1] > n_modes:
        lam_k = lam_k[-n_modes:]                        # keep the k most energetic
        Wk = Wk[:, -n_modes:]                           #   (eigh is ascending)
    M = Xc_sub.T @ Xc_val                              # n x N_val
    coeff = Wk @ ((Wk.T @ M) / lam_k[:, None])         # (Xc^T Xc)^+ (Xc^T Xc_val)
    proj = Xc_sub @ coeff                              # D x N_val  = U U^T Xc_val
    resid = Xc_val - proj
    err = np.linalg.norm(resid) ** 2 / (np.linalg.norm(Xc_val) ** 2 + 1e-30)
    return float(err), int(Wk.shape[1])


def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX, T_STRIDE, T_MAX)
    Nt, C, H, W = data.shape
    D = C * H * W
    X = data.reshape(Nt, D).astype(np.float64)         # rows = snapshots
    del data
    print(f"[{STRATEGY}]  {os.path.basename(DATA_FILE)}  |  {Nt} snapshots, "
          f"{C}x{H}x{W}  (D = {D})")

    # ---- one fixed 90/10 split (random or tail, see convergence_split.py) ----
    val_idx, pool_idx, n_val = cs.split_indices(Nt, VAL_FRAC, SPLIT, SPLIT_SEED)
    N_pool = len(pool_idx)
    Xval = X[val_idx].T                                # D x N_val (columns=snaps)
    print(f"[{STRATEGY}]  split = {SPLIT}: {N_pool} training-pool, {n_val} validation "
          f"({cs.describe(SPLIT)}; val idx {val_idx[0]}..{val_idx[-1]})", flush=True)

    # ---- n sweep ----
    if N_LIST is not None:
        n_vals = [int(v) for v in N_LIST if int(v) <= N_pool]
    else:
        n_vals = list(range(N_STEP, N_pool + 1, N_STEP))
        if not n_vals or n_vals[-1] != N_pool:
            n_vals.append(N_pool)                       # always include the full pool
    n_vals = sorted(set(n_vals))
    print(f"[{STRATEGY}]  n sweep = {n_vals}   (m = {M_REPS} draws each)")

    draw_rng = np.random.default_rng(DRAW_SEED)
    e_mean = np.empty(len(n_vals))
    e_std = np.empty(len(n_vals))
    e_all = np.full((len(n_vals), M_REPS), np.nan)
    modes_kept = np.empty(len(n_vals))

    for i, n in enumerate(n_vals):
        reps = min(M_REPS, 1) if n == N_pool else M_REPS   # full pool: only 1 draw
        errs, mks = [], []
        for j in range(reps):
            sub = draw_rng.choice(pool_idx, size=n, replace=False) if n < N_pool else pool_idx
            Xs = X[sub].T                               # D x n
            xbar = Xs.mean(axis=1, keepdims=True)       # subset mean flow
            Xc_sub = Xs - xbar
            Xc_val = Xval - xbar                         # center val by SAME mean
            err, mk = pod_project_error(Xc_sub, Xc_val, N_MODES)
            errs.append(err); mks.append(mk)
        errs = np.array(errs)
        e_all[i, :len(errs)] = errs
        e_mean[i] = errs.mean()
        e_std[i] = errs.std()
        modes_kept[i] = np.mean(mks)
        print(f"  n = {n:4d}  ->  e_n = {e_mean[i]:.4e} +/- {e_std[i]:.2e}   "
              f"(modes kept ~ {modes_kept[i]:.0f}, {len(errs)} draw(s))")

    # ---- plot ----
    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(n_vals, e_mean, yerr=e_std, marker="o", capsize=3,
                lw=1.6, ms=5, color="#1f77b4", ecolor="#1f77b4", label="mean $\\pm$ std")
    ax.set_xlabel("n  (number of snapshots used to build the POD)")
    ax.set_ylabel(r"unexplained energy fraction  $e_n = \|X_V - UU^{\top}X_V\|_F^2 / \|X_V\|_F^2$")
    trunc = "full basis" if N_MODES is None else f"top-{N_MODES} modes"
    if E_LEVEL and N_MODES is not None:
        trunc += f" ({E_LEVEL}% energy)"
    vdesc = "random" if SPLIT == "random" else "last"
    ax.set_title(f"POD spatial-mode convergence — {base}\n"
                 f"(centered POD, {trunc}, {n_val} held-out {vdesc} val snapshots, "
                 f"m = {M_REPS} draws/n)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # ---- save numbers ----
    from scipy.io import savemat
    mat = os.path.join(_AUG_DIR, f"{base}_{STRATEGY}.mat")
    savemat(mat, {
        "n_vals": np.array(n_vals), "e_mean": e_mean, "e_std": e_std,
        "e_all": e_all, "modes_kept": modes_kept, "m_reps": M_REPS,
        "n_modes": (0 if N_MODES is None else N_MODES),
        "energy_level": (E_LEVEL if E_LEVEL else ""),   # e.g. "95" / "99" (label)
        "val_frac": VAL_FRAC, "n_val": n_val, "n_pool": N_pool,
        "val_idx": val_idx, "pool_idx": pool_idx, "centered": True,
        "split": SPLIT,
        "split_seed": SPLIT_SEED, "draw_seed": DRAW_SEED, "strategy": STRATEGY,
    })
    print(f"\n[{STRATEGY}]  saved  ->  {mat}\n")

    # ---- record the last (converged) point in the shared CSV ----
    import re
    from convergence_csv import update_row
    m = re.search(r"Re(\d+)", base)
    csv_path = cs.csv_path(DATA_FILE, SPLIT)
    update_row(csv_path, base, {
        "Re": (int(m.group(1)) if m else ""),
        f"{CSV_TAG}_n": int(n_vals[-1]),
        f"{CSV_TAG}_e_mean": float(e_mean[-1]),
        f"{CSV_TAG}_modes": (0 if N_MODES is None else N_MODES),
    })
    print(f"[{STRATEGY}]  updated  ->  {csv_path}  "
          f"({CSV_TAG}_n={n_vals[-1]}, {CSV_TAG}_e_mean={e_mean[-1]:.4e})\n")


if __name__ == "__main__":
    main()

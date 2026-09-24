"""
beta-VAE CONVERGENCE study (the NN counterpart of pod_convergence.py).

Same question, same split, same error idea as the POD convergence script, but
the reduced model is the beta-VAE instead of a POD basis:

    - hold out 10% of the record as a REAL validation set (random with a fixed
      seed, or the LAST 10% of the record — the SPLIT argument);
    - from the remaining 90% pool draw a random subset of n snapshots;
    - train a FRESH beta-VAE on those n snapshots (same architecture / loss /
      hyperparameters as beta_vae.py);
    - measure the reconstruction quality on the held-out real validation set
      with the relative-energy metric Ek we already use, turned into an ERROR:

          e_n = 1 - Ek/100  =  SSE / EN
              = fraction of the validation fluctuation ENERGY the VAE fails to
                reconstruct   (SSE = sum-sq recon error, EN = sum-sq fluctuation)

    - repeat m times with different random subsets of the same size n, average.

Sweeping n = N_STEP, 2*N_STEP, ... up to |pool| traces the data-convergence
curve of the network: how many snapshots the VAE needs before more data stops
helping.  Normalization + the physical mean field come from the n-snapshot
TRAINING subset only (the validation set is never used to fit anything), exactly
mirroring the subset-mean centering in the POD study.

NOTE this is far heavier than the POD study: it trains (roughly) len(n_vals)*m
networks.  Keep N_STEP / M_REPS / N_EPOCHS modest, or bump them when you have
time.  Nothing is saved except the summary .mat + the shown plot.

Usage:  python3 nn_convergence.py <file.mat> <T_MAX> <LATENT_DIM> [SPLIT]
        SPLIT = random (default: validation = random 10%), tail (validation =
        the LAST 10% of the record) tail5 (the LAST 5%) or tail2.5 (the LAST 2.5%); see convergence_split.py.

Output: ../DATA/AUGMENTED[_TAILVAL]/<base>_nn_convergence.mat  (+ a plot shown)
        + NN_* columns in the dataset folder's convergence_last_point[_tailval].csv
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# The identical network + metrics + losses of beta_vae.py, from rom.
from rom import vae as bv
from rom.data import load_data
from rom import split as cs
from rom import paths

# ------------ Folder layout (rom/paths.py) ------------
_DATA_DIR = paths.DATA

# ============ CONFIG ============
# DATA_FILE and the snapshot cap can be overridden on the command line so the
# same script runs over the whole Alpha0 Re-sweep, e.g.
#   python3 nn_convergence.py .../Alpha0/dataRe60Alpha0_2.mat 1500
_DEFAULT_FILE = ("/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/"
                 "DATA/Alpha0/dataRe50Alpha0_2.mat")
DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_FILE
COMP_IDX  = None      # None = auto-detect velocity components

# temporal subsampling: cap every dataset at T_MAX = 1500 (the minimum snapshot
# count across the Alpha0 Re-sweep) so all four share the SAME snapshot budget.
T_STRIDE  = 1         # keep every snapshot (dt = 0.2)
T_MAX     = int(sys.argv[2]) if len(sys.argv) > 2 else 1500   # common budget

N_HEAD    = [5, 10, 15, 20, 30, 50, 100, 150, 250, 400]  # explicit early sweep points
TAIL_STEP = 250       # after the head, continue in steps of 250 ...
TAIL_MAX  = 1500      # ... up to this many snapshots (or |pool|, whichever smaller)
N_LIST    = None      # set a full explicit list to override N_HEAD/TAIL_STEP
M_REPS    = 3         # random training subsets averaged per n (the parameter m)
PROG_EVERY = 50       # print a training-progress line every this many epochs

# VAE hyperparameters (fixed across the sweep; LATENT_DIM is per-dataset and the
# driver passes it: Re50=5, Re60=7, Re70=9, Re80=11, Re100=15)
LATENT_DIM = int(sys.argv[3]) if len(sys.argv) > 3 else 5
BETA       = 5e-3
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4

# SPLIT: how the validation set is chosen — "random" (10% at random, the original
# results) or "tail" (the LAST 10% of the record).  Each mode writes to its OWN
# folder / CSV so the two never overwrite each other (convergence_split.py).
SPLIT      = cs.check_mode(sys.argv[4] if len(sys.argv) > 4 else "random")
VAL_FRAC = cs.val_frac(SPLIT)   # validation fraction of THIS split mode (0.10 or 0.025)
_AUG_DIR   = cs.aug_dir(SPLIT)

SPLIT_SEED = 7        # seed for the 90/10 split (random mode only; same as POD)
DRAW_SEED  = 11       # seed for the random subset draws + per-run torch seeding
STRATEGY   = "nn_convergence"
# ================================

DEVICE = bv.DEVICE


def train_one_vae(train_data, val_real, C, H, W, epochs, seed, tag=""):
    """Train one beta-VAE on train_data, return best-val relative-energy error.

    Normalization + physical mean come from train_data only.  Validation uses the
    real held-out set.  We track the checkpoint with the lowest validation loss
    (like beta_vae.py) and return its Ek-based error e = 1 - Ek/100, plus Ek.

    A heartbeat line is printed every PROG_EVERY epochs so a long training run is
    visibly making progress (the whole reason nothing showed for hours before).
    """
    t0 = time.time()
    mu_C  = train_data.mean(axis=(0, 2, 3)).astype(np.float32)
    std_C = train_data.std(axis=(0, 2, 3))
    std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    phys_mean = train_data.mean(axis=0).astype(np.float32)          # [C, H, W]

    def norm(x):
        return (x - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    train_n  = norm(train_data)
    val_n    = norm(val_real)
    val_fluc = val_real - phys_mean[None]

    X_tr = torch.tensor(train_n, dtype=torch.float32)
    X_va = torch.tensor(val_n,   dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(seed)
    enc = bv.Encoder(H, W, C, LATENT_DIM).to(DEVICE)
    dec = bv.Decoder(H, W, C, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every = max(1, round(epochs / 10))
    best_val = float("inf")
    best_ek  = float("nan")

    for epoch in range(1, epochs + 1):
        enc.train(); dec.train()
        run_loss = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, _, _ = bv.vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss)
        run_loss /= max(1, len(tr_loader))

        if (epoch % val_every == 0) or (epoch == epochs):
            vl, _, _ = bv.val_loss(enc, dec, va_loader, BETA, DEVICE)
            if vl < best_val:
                _, ek_all = bv.compute_ek(enc, dec, val_n, val_fluc,
                                          mu_C, std_C, phys_mean, DEVICE)
                best_val, best_ek = vl, ek_all

        if (epoch % PROG_EVERY == 0) or (epoch == epochs):
            print(f"      [{tag}] epoch {epoch:4d}/{epochs}  "
                  f"train_loss={run_loss:.4e}  best_Ek={best_ek:5.2f}%  "
                  f"[{time.time()-t0:6.1f}s]", flush=True)

    err = 1.0 - best_ek / 100.0                     # relative-energy error (SSE/EN)
    return err, best_ek


def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX, T_STRIDE, T_MAX)
    Nt, C, H, W = data.shape
    print(f"[{STRATEGY}]  {os.path.basename(DATA_FILE)}  |  {Nt} snapshots, "
          f"{C}x{H}x{W}   device = {DEVICE}", flush=True)

    # ---- one fixed 90/10 split (same recipe as pod_convergence.py) ----
    val_idx, pool_idx, n_val = cs.split_indices(Nt, VAL_FRAC, SPLIT, SPLIT_SEED)
    N_pool = len(pool_idx)
    val_real = data[val_idx].astype(np.float32)
    print(f"[{STRATEGY}]  split = {SPLIT}: {N_pool} training-pool, {n_val} validation "
          f"(REAL; {cs.describe(SPLIT)}; val idx {val_idx[0]}..{val_idx[-1]})",
          flush=True)

    # ---- n sweep ----
    ceiling = min(TAIL_MAX, N_pool)                       # can't train on more than pool
    if N_LIST is not None:
        n_vals = [int(v) for v in N_LIST if int(v) <= N_pool]
    else:
        head = [v for v in N_HEAD if v <= ceiling]        # 50, 100, 200, 400
        start = (head[-1] if head else 0) + TAIL_STEP     # then in steps of TAIL_STEP
        tail = list(range(start, ceiling + 1, TAIL_STEP)) # 650, 900, 1150, 1400, ...
        n_vals = head + tail + [ceiling]                  # always include the ceiling
    n_vals = sorted(set(int(v) for v in n_vals if int(v) <= N_pool))
    n_trainings = sum(1 if n == N_pool else M_REPS for n in n_vals)
    print(f"[{STRATEGY}]  n sweep = {n_vals}   (m = {M_REPS} draws each)", flush=True)
    print(f"[{STRATEGY}]  => {n_trainings} VAE trainings x {N_EPOCHS} epochs "
          f"(latent={LATENT_DIM}, beta={BETA})\n", flush=True)

    draw_rng = np.random.default_rng(DRAW_SEED)
    e_mean = np.empty(len(n_vals)); e_std = np.empty(len(n_vals))
    ek_mean = np.empty(len(n_vals))
    e_all  = np.full((len(n_vals), M_REPS), np.nan)
    t0 = time.time()

    for i, n in enumerate(n_vals):
        reps = 1 if n == N_pool else M_REPS         # full pool: only one subset
        errs, eks = [], []
        print(f"[{STRATEGY}]  ---- sweep point {i+1}/{len(n_vals)}:  "
              f"n = {n}  ({reps} draw(s), {N_EPOCHS} epochs each) ----", flush=True)
        for j in range(reps):
            if n < N_pool:
                sub = draw_rng.choice(pool_idx, size=n, replace=False)
            else:
                sub = pool_idx
            train_data = data[sub].astype(np.float32)
            err, ek = train_one_vae(train_data, val_real, C, H, W,
                                    N_EPOCHS, seed=DRAW_SEED + 1000 * i + j,
                                    tag=f"n={n} draw {j+1}/{reps}")
            errs.append(err); eks.append(ek)
            print(f"  n = {n:4d}  draw {j+1}/{reps}  ->  Ek = {ek:6.2f}%   "
                  f"e = {err:.4e}   [{time.time()-t0:6.1f}s]", flush=True)
        errs = np.array(errs)
        e_all[i, :len(errs)] = errs
        e_mean[i] = errs.mean(); e_std[i] = errs.std()
        ek_mean[i] = np.mean(eks)
        print(f"  n = {n:4d}  ==>  e_n = {e_mean[i]:.4e} +/- {e_std[i]:.2e}   "
              f"(mean Ek = {ek_mean[i]:.2f}%)\n", flush=True)

    # ---- plot (linear y, shown not saved, matching pod_convergence.py) ----
    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(n_vals, e_mean, yerr=e_std, marker="s", capsize=3,
                lw=1.6, ms=6, color="#2ca02c", ecolor="#2ca02c", label="mean $\\pm$ std")
    ax.set_xlabel("n  (number of snapshots used to TRAIN the VAE)")
    ax.set_ylabel(r"relative-energy validation error  $e_n = 1 - E_k/100$")
    vdesc = "random" if SPLIT == "random" else "last"
    ax.set_title(f"beta-VAE data convergence — {base}\n"
                 f"(latent={LATENT_DIM}, beta={BETA}, {n_val} held-out {vdesc} REAL val, "
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
        "e_all": e_all, "ek_mean": ek_mean, "m_reps": M_REPS,
        "val_frac": VAL_FRAC, "n_val": n_val, "n_pool": N_pool,
        "val_idx": val_idx, "pool_idx": pool_idx,
        "latent_dim": LATENT_DIM, "beta": BETA, "n_epochs": N_EPOCHS, "lr": LR,
        "split": SPLIT,
        "split_seed": SPLIT_SEED, "draw_seed": DRAW_SEED, "strategy": STRATEGY,
    })
    print(f"[{STRATEGY}]  total wall time {time.time()-t0:.1f}s")
    print(f"[{STRATEGY}]  saved  ->  {mat}\n")

    # ---- record the last (converged) point in the shared CSV ----
    import re
    from rom.results import update_row
    m = re.search(r"Re(\d+)", base)
    csv_path = cs.csv_path(DATA_FILE, SPLIT)
    update_row(csv_path, base, {
        "Re": (int(m.group(1)) if m else ""),
        "NN_n": int(n_vals[-1]),
        "NN_e_mean": float(e_mean[-1]),
        "NN_Ek": float(ek_mean[-1]),
        "NN_latent": LATENT_DIM,
    })
    print(f"[{STRATEGY}]  updated  ->  {csv_path}  "
          f"(NN_n={n_vals[-1]}, NN_Ek={ek_mean[-1]:.2f}%)\n")


if __name__ == "__main__":
    main()

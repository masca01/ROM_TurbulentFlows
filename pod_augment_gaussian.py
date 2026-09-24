"""
POD data augmentation — STRATEGY A: Independent Gaussian per mode.

"""

import os
import numpy as np
import scipy.io as sio

# load_data is reused from the existing beta_vae.py (same folder).
from beta_vae import load_data

# ── Folder layout ──
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
os.makedirs(_AUG_DIR, exist_ok=True)

# ══ CONFIG (keep SEED / VAL_FRAC / N_AUG identical across all 3 generators) ══
DATA_FILE  = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y1.5708_256x256_Nt5024_UVW.mat"
COMP_IDX   = None     # None = auto-detect components
RNG_SEED   = 7        # MUST match the other generators & the tester
VAL_FRAC   = 0.10     # fraction of REAL snapshots held out for validation
N_AUG      = None     # number of synthetic snapshots (None = same as #train)
RANK       = None     # POD modes to keep (None = all training modes)
STRATEGY   = "gaussian"
# ═══════════════════════════════════════════════════════════════════════════


def compute_pod(train, rank=None):
    """train [Ntr, C, H, W] -> (x_mean [D], U [D,r], S [r], A [Ntr,r], shape).

    A[t,i] = S[i] * V[i,t] are the temporal coefficients (amplitudes)."""
    Ntr, C, H, W = train.shape
    D = C * H * W
    X = train.reshape(Ntr, D).T                  # [D, Ntr] snapshots as columns
    x_mean = X.mean(axis=1)                       # [D] temporal mean
    Xc = X - x_mean[:, None]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # U[D,r] S[r] Vt[r,Ntr]
    if rank is not None:
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    A = Vt.T * S[None, :]                          # [Ntr, r]  A[t,i]=S[i]Vt[i,t]
    return x_mean, U, S, A, (C, H, W)


def reconstruct(x_mean, U, A_new, shape):
    """x_mean [D], U [D,r], A_new [N,r] -> snapshots [N, C, H, W]."""
    C, H, W = shape
    X_new = x_mean[:, None] + U @ A_new.T          # [D, N]
    return X_new.T.reshape(-1, C, H, W).astype(np.float32)


def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape

    # ── Train / validation split (validation stays 100% real) ──
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(Nt)
    n_val   = max(1, round(VAL_FRAC * Nt))
    val_idx = np.sort(idx[:n_val])
    tr_idx  = np.sort(idx[n_val:])
    train_real = data[tr_idx]
    val_real   = data[val_idx]
    Ntr = len(tr_idx)
    n_aug = N_AUG or Ntr

    # ── POD on the training snapshots ONLY ──
    x_mean, U, S, A, shape = compute_pod(train_real, RANK)
    r = A.shape[1]

    # ── STRATEGY A: independent Gaussian per mode ──
    a_mean = A.mean(axis=0)                         # [r]
    a_std  = A.std(axis=0)                          # [r]
    A_new = rng.normal(a_mean[None, :], a_std[None, :], size=(n_aug, r))
    train_aug = reconstruct(x_mean, U, A_new, shape)

    # ── Save bundle ──
    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    out_file = os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}.mat")
    sio.savemat(out_file, {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object),
        "strategy": STRATEGY, "tr_idx": tr_idx, "val_idx": val_idx,
        "rng_seed": RNG_SEED, "val_frac": VAL_FRAC,
        "n_modes": r, "rank": (-1 if RANK is None else RANK),
    }, do_compression=True)

    cum = np.cumsum(S**2) / np.sum(S**2)
    print(f"\n[{STRATEGY}]  {base}  |  {H}x{W}  C={C}")
    print(f"  real train = {Ntr}   real val = {len(val_idx)}   synthetic = {n_aug}")
    print(f"  POD modes  = {r}   (modes for 99% energy: {int(np.searchsorted(cum, 0.99)+1)})")
    print(f"  saved → {out_file}\n")


if __name__ == "__main__":
    main()

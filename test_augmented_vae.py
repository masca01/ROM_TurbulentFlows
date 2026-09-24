"""
Tester for POD-augmented datasets.
==================================

Trains the SAME beta-VAE (imported from beta_vae.py, so the architecture is
identical across runs) on a POD-augmented bundle produced by one of:

    pod_augment_gaussian.py    (strategy A)
    pod_augment_highmodes.py   (strategy B)
    pod_augment_joint.py       (strategy C)

Each bundle contains  train_real, train_aug, val_real.  This tester builds

    training set   =  train_real  (+ train_aug  if USE_AUG=True)
    validation set =  val_real           <-- ALWAYS 100% REAL DATA

so the augmented snapshots are NEVER used for validation, and the POD used to
make them was fit on training snapshots only (no validation leakage).

Normalization statistics and the physical mean field are taken from the REAL
training snapshots only, so the sole thing that changes between runs is the
augmentation strategy.  This makes the comparison fair.

Workflow
--------
1. Run each generator once (they share the same seed/split, so train_real and
   val_real are identical across the three bundles).
2. Point AUG_FILE at a bundle, set a LABEL, run this tester.
3. Optionally run once with USE_AUG=False to get the real-data-only baseline.
4. Results (best validation Ek and det(R)) are appended to a CSV so you can
   compare strategies side by side.
"""

import os, sys
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

# Reuse the identical network + metrics + losses from beta_vae.py (same folder).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta_vae as bv

# ── Folder layout ──
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
_RESULTS  = os.path.join(_HERE, "augment_results_isotropic.csv")

# ══ CONFIG ══
AUG_FILE   = os.path.join(_AUG_DIR, "isotropic1024coarse_xz_y1.5708_256x256_Nt5024_UVW_aug_joint.mat")
LABEL      = None     # None = use the strategy name stored in the bundle
USE_AUG    = True     # False = train on real data only (baseline)

# training hyperparameters
LATENT_DIM = 2
BETA       = 5e-3
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4
RNG_SEED   = 7
VAL_EVERY  = None     # None = auto (every ~10% of epochs)
# ═══════════════════════════════════════════════════════════

DEVICE = bv.DEVICE


def load_bundle(path):
    S = sio.loadmat(path, simplify_cells=True)
    train_real = np.asarray(S["train_real"], dtype=np.float32)
    train_aug  = np.asarray(S["train_aug"],  dtype=np.float32)
    val_real   = np.asarray(S["val_real"],   dtype=np.float32)
    # guard against scipy squeezing a singleton snapshot axis
    def _fix(a):
        return a[None] if a.ndim == 3 else a
    train_real, train_aug, val_real = _fix(train_real), _fix(train_aug), _fix(val_real)
    cn = S.get("comp_names", None)
    if cn is None:
        comp_names = [f"c{i}" for i in range(train_real.shape[1])]
    elif isinstance(cn, str):
        comp_names = [cn]
    else:
        comp_names = [str(x) for x in np.atleast_1d(cn)]
    strategy = str(S.get("strategy", "unknown"))
    return train_real, train_aug, val_real, comp_names, strategy


def main():
    train_real, train_aug, val_real, comp_names, strategy = load_bundle(AUG_FILE)
    label = LABEL or strategy
    C, H, W = train_real.shape[1:]

    # ── Normalization & mean field from REAL training data only ──
    mu_C  = train_real.mean(axis=(0, 2, 3))
    std_C = train_real.std(axis=(0, 2, 3))
    std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    mu_C  = mu_C.astype(np.float32)
    phys_mean = train_real.mean(axis=0)                       # [C, H, W]

    def norm(x):
        return (x - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    # ── Build training / validation sets ──
    train_data = np.concatenate([train_real, train_aug], axis=0) if USE_AUG else train_real
    train_n = norm(train_data)
    val_n   = norm(val_real)
    val_fluc = val_real - phys_mean[None]                    # physical fluctuations

    X_tr = torch.tensor(train_n, dtype=torch.float32)
    X_va = torch.tensor(val_n,   dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(RNG_SEED)
    enc = bv.Encoder(H, W, C, LATENT_DIM).to(DEVICE)
    dec = bv.Decoder(H, W, C, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every = VAL_EVERY or max(1, round(N_EPOCHS / 10))

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION TEST  |  strategy = {label}  |  use_aug = {USE_AUG}")
    print(f"  {H}x{W}  C={C}  |  latent={LATENT_DIM}  beta={BETA}")
    print(f"  train = {len(train_data)}  (real {len(train_real)} + aug "
          f"{len(train_aug) if USE_AUG else 0})   val(real) = {len(val_real)}")
    print(f"  device = {DEVICE}")
    print(f"{'='*60}\n")

    best_val = float("inf")
    best_ek = best_detR = float("nan")

    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        s_l = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, _, _ = bv.vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            s_l += loss.item()
        s_l /= len(tr_loader)

        if (epoch % val_every == 0) or (epoch == N_EPOCHS):
            vl, _, _ = bv.val_loss(enc, dec, va_loader, BETA, DEVICE)
            _, ek_all = bv.compute_ek(enc, dec, val_n, val_fluc,
                                      mu_C, std_C, phys_mean, DEVICE)
            detR = bv.compute_det_R(enc, val_n, LATENT_DIM, DEVICE)
            marker = ""
            if vl < best_val:
                best_val, best_ek, best_detR = vl, ek_all, detR
                marker = "  ★ best"
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l:.5f} | va {vl:.5f} | "
                  f"Ek {ek_all:.2f}% | detR {detR:.4f}{marker}")
        else:
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l:.5f}")

    # ── Append result row ──
    header = "label,use_aug,n_train,n_real,n_aug,n_val,best_val_loss,best_val_Ek,best_detR\n"
    row = (f"{label},{USE_AUG},{len(train_data)},{len(train_real)},"
           f"{len(train_aug) if USE_AUG else 0},{len(val_real)},"
           f"{best_val:.6f},{best_ek:.4f},{best_detR:.6f}\n")
    write_header = not os.path.exists(_RESULTS)
    with open(_RESULTS, "a") as f:
        if write_header:
            f.write(header)
        f.write(row)

    print(f"\n{'='*60}")
    print(f"  RESULT  [{label}]  use_aug={USE_AUG}")
    print(f"  best validation (REAL) Ek = {best_ek:.2f}%   det(R) = {best_detR:.4f}")
    print(f"  appended → {_RESULTS}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

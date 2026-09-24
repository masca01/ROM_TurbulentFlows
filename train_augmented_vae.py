"""
train_augmented_vae.py — Train a beta-VAE on POD-augmented data, validate on REAL
=================================================================================

Same idea as test_augmented_vae.py (fair, leak-free augmentation experiment) but
this script ALSO behaves like beta_vae.py at the end: it saves the best-validation
checkpoint to ../bestModels/ and shows the same figures (learning curves +
TRUE | RECON | ERROR field for a real validation snapshot).

Data flow
---------
Input bundle (from pod_augment_*.py) holds  train_real, train_aug, val_real.

    training set    =  train_real  (+ train_aug  if USE_AUG=True)
    validation set  =  val_real            <-- ALWAYS 100% REAL DATA

so synthetic snapshots are NEVER used for validation, and normalization + the
physical mean field come from the REAL training snapshots only.  The reported
Ek / det(R) — and the figures — are therefore measured on real data only.

What gets saved
---------------
1. ../bestModels/model_augVAE_<base>_<strategy>_lat<L>_b<beta>_ep<E>.pt
   A SELF-CONTAINED checkpoint: best-val weights + mu_C/std_C/fluc_mean +
   val_real baked in, so it can be re-visualized later WITHOUT reloading the
   original (possibly multi-GB) .mat file.
2. One row appended to augment_results.csv (best_val_loss / Ek / det(R)),
   matching test_augmented_vae.py so runs stay comparable.

Visualization matches beta_vae.py: run this file and the figures pop up; the
saved checkpoint reproduces the same Ek you see logged (same model, same real
validation set).
"""

import os, sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# Reuse the identical network + metrics + losses from beta_vae.py (same folder).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta_vae as bv

# ── Folder layout ──
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR    = os.path.join(_DATA_DIR, "AUGMENTED")
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "bestModels"))
_RESULTS    = os.path.join(_HERE, "augment_results_alpha0_bestcase.csv")
os.makedirs(_MODELS_DIR, exist_ok=True)

# ══ CONFIG ══
# best case of the convergence study: Re50 Alpha0, 250 real (+1100 synthetic)
# from pod_augment_galerkin_ns_bestcase.py; validation = the 150 real snapshots
# of the random split.  Baseline to beat (250 real only): e = 0.0496, Ek = 95.0 %
# (nn_convergence.py, same subset); all 1350 real: Ek = 98.3 %.
AUG_FILE   = os.path.join(_AUG_DIR, "dataRe50Alpha0_2_aug_galerkin_ns_n250.npz")
LABEL      = None     # None = use the strategy name stored in the bundle
USE_AUG    = True     # False = train on real data only (baseline)

# training hyperparameters (keep identical across runs you want to compare)
LATENT_DIM = 5        # Re50 latent of the convergence study
BETA       = 5e-3
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4
RNG_SEED   = 7
VAL_EVERY  = None     # None = auto (every ~10% of epochs)
SAVE_MODEL = True     # save best checkpoint as .pt
T_SHOW     = 0        # index into val_real to display in the recon figure
# ═══════════════════════════════════════════════════════════

DEVICE = bv.DEVICE


def load_bundle(path):
    # bundles are .mat (small) or .npz (large, e.g. full 2-plates > 2 GB)
    if path.endswith(".npz"):
        S = np.load(path, allow_pickle=True)
    else:
        S = sio.loadmat(path, simplify_cells=True)
    train_real = np.asarray(S["train_real"], dtype=np.float32)
    train_aug  = np.asarray(S["train_aug"],  dtype=np.float32)
    val_real   = np.asarray(S["val_real"],   dtype=np.float32)
    def _fix(a):                                  # guard squeezed singleton axis
        return a[None] if a.ndim == 3 else a
    train_real, train_aug, val_real = _fix(train_real), _fix(train_aug), _fix(val_real)
    cn = S["comp_names"] if "comp_names" in S else None
    if cn is None:
        comp_names = [f"c{i}" for i in range(train_real.shape[1])]
    elif isinstance(cn, str):
        comp_names = [cn]
    else:
        comp_names = [str(x) for x in np.atleast_1d(cn)]
    strategy = str(S["strategy"]) if "strategy" in S else "unknown"
    return train_real, train_aug, val_real, comp_names, strategy


def main():
    train_real, train_aug, val_real, comp_names, strategy = load_bundle(AUG_FILE)
    label = LABEL or strategy
    C, H, W = train_real.shape[1:]

    # ── Normalization & mean field from REAL training data only ──
    mu_C  = train_real.mean(axis=(0, 2, 3)).astype(np.float32)
    std_C = train_real.std(axis=(0, 2, 3))
    std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    phys_mean = train_real.mean(axis=0).astype(np.float32)        # [C, H, W]

    def norm(x):
        return (x - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    # ── Build training / validation sets ──
    train_data = np.concatenate([train_real, train_aug], axis=0) if USE_AUG else train_real
    train_n = norm(train_data)
    val_n   = norm(val_real)
    val_fluc = val_real - phys_mean[None]                         # physical fluctuations

    X_tr = torch.tensor(train_n, dtype=torch.float32)
    X_va = torch.tensor(val_n,   dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(RNG_SEED)
    enc = bv.Encoder(H, W, C, LATENT_DIM).to(DEVICE)
    dec = bv.Decoder(H, W, C, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every = VAL_EVERY or max(1, round(N_EPOCHS / 10))

    base = os.path.splitext(os.path.basename(AUG_FILE))[0]
    tag  = label if USE_AUG else f"{label}_NOAUG"
    save_path = os.path.join(_MODELS_DIR,
        f"model_augVAE_{base}_{tag}_lat{LATENT_DIM}_b{BETA:.0e}_ep{N_EPOCHS}.pt")

    print(f"\n{'='*60}")
    print(f"  AUG-VAE  |  strategy = {label}  |  use_aug = {USE_AUG}")
    print(f"  {H}x{W}  C={C}  |  latent={LATENT_DIM}  beta={BETA}  lr={LR}")
    print(f"  train = {len(train_data)}  (real {len(train_real)} + aug "
          f"{len(train_aug) if USE_AUG else 0})   val(real) = {len(val_real)}")
    print(f"  device = {DEVICE}")
    print(f"{'='*60}\n")

    hist = {k: [] for k in ["tr_loss","tr_recon","tr_kl",
                            "va_loss","va_recon","va_kl",
                            "va_ek_all","va_det_R","va_epoch"]}
    best_val = float("inf")
    best_ek = best_detR = float("nan")
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        s_l = s_r = s_k = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, r, k = bv.vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            s_l += loss.item(); s_r += r.item(); s_k += k.item()
        nb = len(tr_loader)
        hist["tr_loss"].append(s_l/nb)
        hist["tr_recon"].append(s_r/nb)
        hist["tr_kl"].append(s_k/nb)

        do_val = (epoch % val_every == 0) or (epoch == N_EPOCHS)
        if do_val:
            vl, vr, vk = bv.val_loss(enc, dec, va_loader, BETA, DEVICE)
            _, ek_all = bv.compute_ek(enc, dec, val_n, val_fluc,
                                      mu_C, std_C, phys_mean, DEVICE)
            detR = bv.compute_det_R(enc, val_n, LATENT_DIM, DEVICE)

            hist["va_loss"].append(vl);  hist["va_recon"].append(vr)
            hist["va_kl"].append(vk);    hist["va_ek_all"].append(ek_all)
            hist["va_det_R"].append(detR); hist["va_epoch"].append(epoch)

            marker = ""
            if vl < best_val:
                best_val, best_ek, best_detR = vl, ek_all, detR
                best_state = {
                    "enc": {kk: vv.cpu().clone() for kk, vv in enc.state_dict().items()},
                    "dec": {kk: vv.cpu().clone() for kk, vv in dec.state_dict().items()}}
                marker = "  ★ best"
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l/nb:.5f} | va {vl:.5f} | "
                  f"Ek {ek_all:.2f}% | detR {detR:.4f}{marker}")
        else:
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l/nb:.5f}")

    # ── Save best model (self-contained: includes val_real) ──
    if SAVE_MODEL and best_state is not None:
        torch.save({
            "enc_state": best_state["enc"],
            "dec_state": best_state["dec"],
            "mu_C": mu_C, "std_C": std_C,
            "fluc_mean": phys_mean,
            "H": H, "W": W, "C": C,
            "latent_dim": LATENT_DIM, "beta": BETA,
            "comp_names": comp_names,
            "val_real": val_real,                 # baked in → no need to reload .mat
            "strategy": label, "use_aug": USE_AUG,
            "n_real": len(train_real), "n_aug": len(train_aug) if USE_AUG else 0,
            "aug_file": AUG_FILE,
            "history": hist,
        }, save_path)
        print(f"\nSaved → {save_path}")

    # ── Append result row (same schema as test_augmented_vae.py) ──
    header = "label,use_aug,n_train,n_real,n_aug,n_val,best_val_loss,best_val_Ek,best_detR\n"
    row = (f"{label},{USE_AUG},{len(train_data)},{len(train_real)},"
           f"{len(train_aug) if USE_AUG else 0},{len(val_real)},"
           f"{best_val:.6f},{best_ek:.4f},{best_detR:.6f}\n")
    write_header = not os.path.exists(_RESULTS)
    with open(_RESULTS, "a") as f:
        if write_header:
            f.write(header)
        f.write(row)
    print(f"appended → {_RESULTS}")

    # ── Figure 1: learning curves (matches beta_vae.py) ──
    ep    = np.arange(1, N_EPOCHS + 1)
    va_ep = np.array(hist["va_epoch"])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(ep, hist["tr_recon"], label="train recon")
    if hist["va_recon"]:
        axes[0].plot(va_ep, hist["va_recon"], "o-", label="val recon")
    axes[0].set(xlabel="Epoch", ylabel="MSE (normalized)", title="Reconstruction loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, hist["tr_kl"], label="train KL")
    if hist["va_kl"]:
        axes[1].plot(va_ep, hist["va_kl"], "o-", label="val KL")
    axes[1].set(xlabel="Epoch", ylabel="KL", title="KL divergence")
    axes[1].legend(); axes[1].grid(True)

    if hist["va_ek_all"]:
        axes[2].plot(va_ep, hist["va_ek_all"], "o-", color="green")
    axes[2].set(xlabel="Epoch", ylabel="Ek (%)", title="Validation Ek (REAL, all comps)")
    axes[2].grid(True)

    if hist["va_det_R"]:
        axes[3].plot(va_ep, hist["va_det_R"], "o-", color="purple")
    axes[3].set(xlabel="Epoch", ylabel="det(R)  [0–1]", title="Latent disentanglement")
    axes[3].grid(True)

    plt.suptitle(f"aug-VAE  |  {label}  use_aug={USE_AUG}  |  latent={LATENT_DIM}  beta={BETA}")
    plt.tight_layout(); plt.show()

    # ── Figure 2: TRUE | RECON | ERROR for a REAL validation snapshot ──
    enc.load_state_dict(best_state["enc"]); enc.to(DEVICE)
    dec.load_state_dict(best_state["dec"]); dec.to(DEVICE)
    enc.eval(); dec.eval()

    t_show = int(np.clip(T_SHOW, 0, len(val_real) - 1))
    with torch.no_grad():
        x_in = torch.tensor(val_n[t_show:t_show+1], device=DEVICE)
        mu_z, _ = enc(x_in)
        xhat = dec(mu_z).cpu().numpy()[0]

    rec  = xhat * std_C[:, None, None] + mu_C[:, None, None]
    true = val_real[t_show]

    fig, axes = plt.subplots(C, 3, figsize=(12, 4*C), squeeze=False)
    for c in range(C):
        mx_field = max(abs(float(true[c].min())), abs(float(true[c].max())), 1e-8)
        vmin, vmax = -mx_field, mx_field
        err = rec[c] - true[c]
        mx_err = max(abs(float(err.min())), abs(float(err.max())), 1e-8)
        im0 = axes[c, 0].imshow(true[c], vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower")
        axes[c, 0].set_title(f"{comp_names[c]} TRUE"); axes[c, 0].axis("off")
        plt.colorbar(im0, ax=axes[c, 0])
        im1 = axes[c, 1].imshow(rec[c], vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower")
        axes[c, 1].set_title(f"{comp_names[c]} RECON"); axes[c, 1].axis("off")
        plt.colorbar(im1, ax=axes[c, 1])
        im2 = axes[c, 2].imshow(err, vmin=-mx_err, vmax=mx_err, cmap="RdBu_r", origin="lower")
        axes[c, 2].set_title(f"{comp_names[c]} ERROR"); axes[c, 2].axis("off")
        plt.colorbar(im2, ax=axes[c, 2])
    plt.suptitle(f"REAL val snapshot t={t_show}  |  strategy={label}  use_aug={USE_AUG}")
    plt.tight_layout(); plt.show()

    print(f"\n{'='*60}")
    print(f"  RESULT  [{label}]  use_aug={USE_AUG}")
    print(f"  best validation (REAL) Ek = {best_ek:.2f}%   det(R) = {best_detR:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
beta-VAE for turbulent flow ROM

Usage:
    python3 beta_vae.py                          # DATA_FILE and N_EPOCHS of the CONFIG block
    python3 beta_vae.py --data FILE --epochs 2   # another file / a quick check
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# network, loss, metrics and the loader live in rom (shared with every study)
from rom import paths
from rom.data import load_data
from rom.vae import (DEVICE, Encoder, Decoder, compute_ek, compute_det_R,
                     vae_loss, val_loss)

# ── Folder layout (rom/paths.py) ──
_DATA_DIR   = paths.DATA
_MODELS_DIR = paths.MODELS
os.makedirs(_MODELS_DIR, exist_ok=True)

# ══ CONFIG ══
DATA_FILE  = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/2PlatesGap/Data2PlatesGap1Re100.mat"
COMP_IDX   = None    # None = auto-detect; [0,2] = pick u,w from 3-comp field

LATENT_DIM = 5
BETA       = 8e-4
BATCH_SIZE = 32
N_EPOCHS   = 1000
LR         = 3e-4
VAL_FRAC   = 0.1
RNG_SEED   = 7       # train/val split
TORCH_SEED = 7       # network initialisation and batch order
VAL_EVERY  = None    # None = auto (every ~10% of epochs)
SAVE_MODEL = True    # save best checkpoint as .pt file
# ═══════════════════════════════════════════════════════════

# ─────────────────────────── Data ───────────────────────────

def pick_file(start_dir=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(
            title="Select .mat data file",
            initialdir=start_dir or os.path.expanduser("~"),
            filetypes=[("MAT files", "*.mat"), ("All", "*")])
        root.destroy()
        return p
    except Exception:
        return input("Path to .mat file: ").strip()


def normalize(data):
    """Per-channel z-score. Returns data_n, mu [C], std [C]."""
    mu  = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (data - mu) / std, mu[0, :, 0, 0], std[0, :, 0, 0]


def mean_field(data):
    """Time-averaged mean field [C, H, W]."""
    return data.mean(axis=0)


# ─────────────────────────── Main ───────────────────────────

def _cli(argv):
    """--data FILE and --epochs N override the CONFIG block."""
    global DATA_FILE, N_EPOCHS
    it = iter(argv)
    for a in it:
        if a == "--data": DATA_FILE = next(it)
        elif a == "--epochs": N_EPOCHS = int(next(it))
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")


def main():
    global DATA_FILE
    _cli(sys.argv[1:])

    if not DATA_FILE:
        DATA_FILE = pick_file(_DATA_DIR)
    if not DATA_FILE:
        raise RuntimeError("No file selected.")

    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape

    data_n, mu_C, std_C = normalize(data)
    fluc_mean = mean_field(data)                  # [C, H, W]
    data_fluc = data - fluc_mean[None]

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(Nt)
    n_val  = max(1, round(VAL_FRAC * Nt))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_tr = torch.tensor(data_n[tr_idx], dtype=torch.float32)
    X_va = torch.tensor(data_n[val_idx], dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(TORCH_SEED)
    enc = Encoder(H, W, C, LATENT_DIM).to(DEVICE)
    dec = Decoder(H, W, C, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every = VAL_EVERY or max(1, round(N_EPOCHS / 10))

    hist = {k: [] for k in ["tr_loss","tr_recon","tr_kl",
                             "va_loss","va_recon","va_kl",
                             "va_ek_all","va_det_R","va_epoch"]}

    best_val = float("inf")
    best_state = None

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    save_path = os.path.join(_MODELS_DIR,
        f"model_betaVAE_{base}_lat{LATENT_DIM}_b{BETA:.0e}_ep{N_EPOCHS}.pt")

    print(f"\n{'='*56}")
    print(f"  beta-VAE  |  {base}  |  {H}x{W}  C={C}  Nt={Nt}")
    print(f"  latent={LATENT_DIM}  beta={BETA}  lr={LR}  epochs={N_EPOCHS}")
    print(f"  device={DEVICE}  train={len(tr_idx)}  val={len(val_idx)}")
    print(f"{'='*56}\n")

    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        s_l = s_r = s_k = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, r, k = vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            s_l += loss.item(); s_r += r.item(); s_k += k.item()
        nb = len(tr_loader)
        hist["tr_loss"].append(s_l/nb)
        hist["tr_recon"].append(s_r/nb)
        hist["tr_kl"].append(s_k/nb)

        do_val = (epoch % val_every == 0) or (epoch == N_EPOCHS)
        if do_val:
            vl, vr, vk = val_loss(enc, dec, va_loader, BETA, DEVICE)
            va_fluc = data_fluc[val_idx]
            va_data_n = data_n[val_idx]
            _, ek_all = compute_ek(enc, dec, va_data_n, va_fluc, mu_C, std_C, fluc_mean, DEVICE)
            det_R = compute_det_R(enc, va_data_n, LATENT_DIM, DEVICE)

            hist["va_loss"].append(vl); hist["va_recon"].append(vr)
            hist["va_kl"].append(vk);   hist["va_ek_all"].append(ek_all)
            hist["va_det_R"].append(det_R); hist["va_epoch"].append(epoch)

            marker = ""
            if SAVE_MODEL and vl < best_val:
                best_val = vl
                best_state = {
                    "enc": {k: v.cpu().clone() for k, v in enc.state_dict().items()},
                    "dec": {k: v.cpu().clone() for k, v in dec.state_dict().items()}}
                marker = "  ★ best"

            print(f"Ep {epoch:4d}/{N_EPOCHS} | "
                  f"tr {s_l/nb:.5f} | va {vl:.5f} | Ek {ek_all:.1f}% | detR {det_R:.4f}{marker}")
        else:
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l/nb:.5f}")

    # ── Save best model ──
    if SAVE_MODEL and best_state is not None:
        torch.save({
            "enc_state": best_state["enc"],
            "dec_state": best_state["dec"],
            "mu_C": mu_C, "std_C": std_C,
            "fluc_mean": fluc_mean,
            "H": H, "W": W, "C": C,
            "latent_dim": LATENT_DIM, "beta": BETA,
            "comp_names": comp_names,
            "tr_idx": tr_idx, "val_idx": val_idx,
            "data_file": DATA_FILE,
            "history": hist,
        }, save_path)
        print(f"\nSaved → {save_path}")

    # ── Learning curves ──
    ep = np.arange(1, N_EPOCHS + 1)
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
    axes[2].set(xlabel="Epoch", ylabel="Ek (%)", title="Validation Ek (all comps)")
    axes[2].grid(True)

    if hist["va_det_R"]:
        axes[3].plot(va_ep, hist["va_det_R"], "o-", color="purple")
    axes[3].set(xlabel="Epoch", ylabel="det(R)  [0–1]", title="Latent disentanglement")
    axes[3].grid(True)

    plt.suptitle(f"beta-VAE  |  {base}  |  latent={LATENT_DIM}  beta={BETA}")
    plt.tight_layout(); plt.show()

    # ── Final reconstruction of a val snapshot ──
    enc.load_state_dict(best_state["enc"]); enc.to(DEVICE)
    dec.load_state_dict(best_state["dec"]); dec.to(DEVICE)
    enc.eval(); dec.eval()

    t_show = val_idx[0]
    with torch.no_grad():
        x_in = torch.tensor(data_n[t_show:t_show+1], device=DEVICE)
        mu_z, _ = enc(x_in)
        xhat = dec(mu_z).cpu().numpy()[0]

    rec = xhat * std_C[:, None, None] + mu_C[:, None, None]
    true = data[t_show]

    fig, axes = plt.subplots(C, 3, figsize=(12, 4*C), squeeze=False)
    for c in range(C):
        mx_field = max(abs(float(true[c].min())), abs(float(true[c].max())), 1e-8)
        vmin, vmax = -mx_field, mx_field
        err = rec[c] - true[c]
        mx_err = max(abs(float(err.min())), abs(float(err.max())), 1e-8)
        im0 = axes[c, 0].imshow(true[c], vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower")
        axes[c, 0].set_title(f"{comp_names[c]} TRUE"); axes[c, 0].axis("off")
        plt.colorbar(im0, ax=axes[c, 0])
        im1 = axes[c, 1].imshow(rec[c],  vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower")
        axes[c, 1].set_title(f"{comp_names[c]} RECON"); axes[c, 1].axis("off")
        plt.colorbar(im1, ax=axes[c, 1])
        im2 = axes[c, 2].imshow(err, vmin=-mx_err, vmax=mx_err, cmap="RdBu_r", origin="lower")
        axes[c, 2].set_title(f"{comp_names[c]} ERROR"); axes[c, 2].axis("off")
        plt.colorbar(im2, ax=axes[c, 2])
    plt.suptitle(f"Val snapshot t={t_show}")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()

"""
inspect_vae.py — Reproduce post-training figures from a saved beta-VAE checkpoint.

Loads a .pt file produced by beta_vae.py and plots:
  Figure 1: Learning curves  (recon loss, KL, Ek, det(R))
  Figure 2: TRUE | RECON | ERROR field at a chosen snapshot

Requirements: pip install torch numpy scipy matplotlib
"""

import os, math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from rom import paths
from rom.data import load_data

# ── Folder layout (rom/paths.py) ──────────────────────────────
_DATA_DIR   = paths.DATA
_MODELS_DIR = paths.MODELS

# ══════════════════════════ CONFIG ══════════════════════════
# Path to the saved .pt checkpoint (from beta_vae.py).
VAE_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/bestModels/model_betaVAE_isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW_lat5_b8e-04_ep500.pt"

# Path to the original .mat data file.
# Leave "" to use the path stored inside the checkpoint.
DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.mat"

# Snapshot index to reconstruct.
# None = use the first validation snapshot saved in the checkpoint.
T_SHOW = None
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────── File dialog ────────────────────────

def _pick_file(title, start_dir, ftype):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(
            title=title,
            initialdir=start_dir or os.path.expanduser("~"),
            filetypes=[(ftype, f"*.{ftype}"), ("All", "*")])
        root.destroy()
        return p
    except Exception:
        return input(f"{title}: ").strip()


# ─────────────────────── Data loading ───────────────────────

# load_data: rom.data.load_data (every layout: Tensor, U, UW, the 2-plates DataU/DataV
# and the Alpha0 U/V files; before, this script had its own copy without the last two)


# ─────────────────────── Network ────────────────────────────

def _n_conv(H, W):
    return min(6, max(2, math.floor(math.log2(min(H, W))) - 2))

_CH = [8, 16, 32, 64, 128, 256]


class Encoder(nn.Module):
    def __init__(self, H, W, C_in, latent_dim):
        super().__init__()
        n = _n_conv(H, W)
        chs = _CH[:n]
        layers, in_ch = [], C_in
        for out_ch in chs:
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.ELU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        Hb = math.ceil(H / 2**n)
        Wb = math.ceil(W / 2**n)
        self.fc   = nn.Sequential(nn.Linear(Hb * Wb * chs[-1], 256), nn.ELU())
        self.mu   = nn.Linear(256, latent_dim)
        self.logv = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        h = self.fc(x)
        return self.mu(h), self.logv(h)


class Decoder(nn.Module):
    def __init__(self, H, W, C_out, latent_dim):
        super().__init__()
        n = _n_conv(H, W)
        chs = _CH[:n]
        self.H, self.W = H, W
        Hb = math.ceil(H / 2**n)
        Wb = math.ceil(W / 2**n)
        ch_b = chs[-1]
        self.Hb, self.Wb, self.ch_b = Hb, Wb, ch_b
        self.fc = nn.Sequential(nn.Linear(latent_dim, Hb * Wb * ch_b), nn.ELU())
        rev = list(reversed(chs))
        layers, in_ch = [], rev[0]
        for out_ch in rev[1:]:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2,
                                          padding=1, output_padding=1), nn.ELU()]
            in_ch = out_ch
        layers += [nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2,
                                      padding=1, output_padding=1), nn.ELU(),
                   nn.Conv2d(in_ch, C_out, 3, padding=1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(-1, self.ch_b, self.Hb, self.Wb)
        x = self.deconv(x)
        h, w = x.shape[2], x.shape[3]
        r0 = (h - self.H) // 2
        c0 = (w - self.W) // 2
        return x[:, :, r0:r0+self.H, c0:c0+self.W]


# ─────────────────────── Figures ────────────────────────────

def fig_learning_curves(hist, base, latent_dim, beta):
    ep    = np.arange(1, len(hist["tr_recon"]) + 1)
    va_ep = np.array(hist["va_epoch"])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(ep, hist["tr_recon"], label="train")
    if hist["va_recon"]:
        axes[0].plot(va_ep, hist["va_recon"], "o-", label="val")
    axes[0].set(xlabel="Epoch", ylabel="MSE (normalized)", title="Reconstruction loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, hist["tr_kl"], label="train")
    if hist["va_kl"]:
        axes[1].plot(va_ep, hist["va_kl"], "o-", label="val")
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

    plt.suptitle(f"beta-VAE  |  {base}  |  latent={latent_dim}  beta={beta}")
    plt.tight_layout(); plt.show()


def fig_reconstruction(true, rec, comp_names, t_show):
    C = true.shape[0]
    fig, axes = plt.subplots(C, 3, figsize=(12, 4 * C), squeeze=False)
    for c in range(C):
        mx_field = max(abs(float(true[c].min())), abs(float(true[c].max())), 1e-8)
        vmin, vmax = -mx_field, mx_field
        err = rec[c] - true[c]
        mx_err = max(abs(float(err.min())), abs(float(err.max())), 1e-8)

        im0 = axes[c, 0].imshow(true[c], vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower")
        axes[c, 0].set_title(f"{comp_names[c]}  TRUE"); axes[c, 0].axis("off")
        plt.colorbar(im0, ax=axes[c, 0])

        im1 = axes[c, 1].imshow(rec[c], vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower")
        axes[c, 1].set_title(f"{comp_names[c]}  RECON"); axes[c, 1].axis("off")
        plt.colorbar(im1, ax=axes[c, 1])

        im2 = axes[c, 2].imshow(err, vmin=-mx_err, vmax=mx_err, cmap="RdBu_r", origin="lower")
        axes[c, 2].set_title(f"{comp_names[c]}  ERROR"); axes[c, 2].axis("off")
        plt.colorbar(im2, ax=axes[c, 2])

    plt.suptitle(f"Snapshot  t={t_show}")
    plt.tight_layout(); plt.show()


# ─────────────────────── Main ───────────────────────────────

def main():
    global VAE_FILE, DATA_FILE

    if not VAE_FILE:
        VAE_FILE = _pick_file("Select beta-VAE checkpoint (.pt)", _MODELS_DIR, "pt")
    if not VAE_FILE:
        raise RuntimeError("No checkpoint selected.")

    # Load checkpoint
    ckpt = torch.load(VAE_FILE, map_location="cpu", weights_only=False)
    H, W, C       = ckpt["H"], ckpt["W"], ckpt["C"]
    latent_dim    = ckpt["latent_dim"]
    mu_C          = ckpt["mu_C"]
    std_C         = ckpt["std_C"]
    comp_names    = list(ckpt["comp_names"])
    beta          = ckpt["beta"]
    hist          = ckpt.get("history", {})
    val_idx       = ckpt.get("val_idx", np.array([], dtype=int))
    base          = os.path.splitext(os.path.basename(VAE_FILE))[0]

    print(f"Loaded  {os.path.basename(VAE_FILE)}")
    print(f"  H={H}  W={W}  C={C}  latent={latent_dim}  β={beta}")

    # Rebuild model
    enc = Encoder(H, W, C, latent_dim)
    dec = Decoder(H, W, C, latent_dim)
    enc.load_state_dict(ckpt["enc_state"]); enc.to(DEVICE).eval()
    dec.load_state_dict(ckpt["dec_state"]); dec.to(DEVICE).eval()

    # Load data
    data_path = DATA_FILE or str(ckpt.get("data_file", ""))
    if not data_path or not os.path.isfile(data_path):
        data_path = _pick_file("Select .mat data file", _DATA_DIR, "mat")
    data, _ = load_data(data_path)

    # Normalize with saved statistics
    data_n = (data - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    # Choose snapshot
    t_show = T_SHOW
    if t_show is None:
        t_show = int(val_idx[0]) if len(val_idx) > 0 else data.shape[0] // 2
    print(f"  Snapshot t={t_show}")

    # Reconstruct
    with torch.no_grad():
        x_in = torch.tensor(data_n[t_show:t_show+1], device=DEVICE)
        mu_z, _ = enc(x_in)
        xhat = dec(mu_z).cpu().numpy()[0]
    rec  = xhat * std_C[:, None, None] + mu_C[:, None, None]
    true = data[t_show]

    # Figure 1: learning curves
    if hist:
        print("\n--- Figure 1: Learning curves ---")
        fig_learning_curves(hist, base, latent_dim, beta)
    else:
        print("No training history found in checkpoint — skipping learning curves.")

    # Figure 2: TRUE | RECON | ERROR
    print("--- Figure 2: Field reconstruction ---")
    fig_reconstruction(true, rec, comp_names, t_show)


if __name__ == "__main__":
    main()

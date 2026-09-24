"""
Analysis: POD modes + beta-VAE modes + comparison  (universal)
Combines plotModesLatent and compareMethods from MATLAB

Inputs:  model_betaVAE_*.pt   (from beta_vae.py)
         pod_*.npz             (from pod.py)

"""

import os, math
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

from rom import paths

# ── Folder layout (relative to this script) ──────────────────
_DATA_DIR   = paths.DATA
_MODELS_DIR = paths.MODELS

# ══════════════════════════ CONFIG ══════════════════════════
VAE_FILE  = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/bestModels/model_betaVAE_isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW_lat5_b8e-04_ep500.pt"
POD_FILE  = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/bestModels/pod_isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.npz"
DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.mat"

N_MODES_PLOT = 5  # spatial modes to show per method
USE_DELTA    = True   # VAE mode = D(e_i) - D(0)

# POD rank used for field comparison (None = closest Ek to VAE)
COMPARE_RANK = None
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────── File selection ──────────────────────

def pick_file(title, start_dir="", ftype="*"):
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


# ─────────────────── Data loading ────────────────────────

def _read_field(S, key, is_hdf5):
    arr = np.array(S[key], dtype=np.float32)
    if is_hdf5:
        arr = arr.T
    return arr


def load_data(path, comp_idx=None):
    """Load .mat file (v5 or v7.3/HDF5) → data [Nt, C, H, W] float32, comp_names."""
    import h5py

    fh = None
    try:
        S = sio.loadmat(path, simplify_cells=True)
        is_hdf5 = False
    except NotImplementedError:
        fh = h5py.File(path, "r")
        S = fh
        is_hdf5 = True

    try:
        if "Tensor" in S:
            T = _read_field(S, "Tensor", is_hdf5)
            C, N1, N2, Nt = T.shape
            data = T.transpose(3, 0, 1, 2)
            names = {2: ["u", "v"], 3: ["u", "v", "w"]}.get(C, [f"c{i}" for i in range(C)])
        elif "U" in S:
            V = _read_field(S, "U", is_hdf5)
            Nt, Nz, Nx, C_all = V.shape
            if comp_idx is None:
                comp_idx = [0, 2] if C_all == 3 else list(range(C_all))
            V = V[:, :, :, comp_idx]
            data = V.transpose(0, 3, 1, 2)
            all_names = ["u", "v", "w"]
            names = [all_names[i] for i in comp_idx]
        elif "UW" in S:
            V = _read_field(S, "UW", is_hdf5)
            data = V.transpose(0, 3, 1, 2)
            names = ["u", "w"]
        else:
            visible = [k for k in S.keys() if not k.startswith("#")]
            raise ValueError(f"Unknown format. Fields: {visible}")
    finally:
        if fh is not None:
            fh.close()

    print(f"Loaded data: {data.shape}  comps={names}")
    return data, names


# ─────────────────── Network (same as beta_vae.py) ───────

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
        r0 = (h - self.H) // 2; c0 = (w - self.W) // 2
        return x[:, :, r0:r0+self.H, c0:c0+self.W]


# ─────────────────── Load model ──────────────────────────

def load_vae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    H, W, C = ckpt["H"], ckpt["W"], ckpt["C"]
    latent_dim = ckpt["latent_dim"]

    enc = Encoder(H, W, C, latent_dim)
    dec = Decoder(H, W, C, latent_dim)
    enc.load_state_dict(ckpt["enc_state"])
    dec.load_state_dict(ckpt["dec_state"])
    enc.to(DEVICE).eval()
    dec.to(DEVICE).eval()

    return enc, dec, ckpt


def encode_all(enc, data_n, device):
    """Encode all snapshots → Z [Nt, latent_dim]."""
    Nt = data_n.shape[0]
    Z = []
    with torch.no_grad():
        for t in range(Nt):
            x = torch.tensor(data_n[t:t+1], device=device)
            mu, _ = enc(x)
            Z.append(mu.cpu().numpy()[0])
    return np.array(Z, dtype=np.float32)   # [Nt, latent_dim]


def decode_unit(dec, i, latent_dim, device):
    """Decode standard basis vector e_i. Returns [C, H, W] normalized."""
    z = torch.zeros(1, latent_dim, device=device)
    z[0, i] = 1.0
    with torch.no_grad():
        return dec(z).cpu().numpy()[0]   # [C, H, W]


def decode_zero(dec, latent_dim, device):
    """Decode zero vector. Returns [C, H, W] normalized."""
    z = torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        return dec(z).cpu().numpy()[0]


# ─────────────────── Plots ───────────────────────────────

def redblue(n=256):
    m = n // 2
    r = np.concatenate([np.linspace(0, 1, m), np.ones(n - m)])
    g = np.concatenate([np.linspace(0, 1, m), np.linspace(1, 0, n - m)])
    b = np.concatenate([np.ones(m), np.linspace(1, 0, n - m)])
    return np.stack([r, g, b], axis=1)

RB = redblue()


def _imshow(ax, F, symm=True, cmap=None):
    cmap = cmap or RB
    if symm:
        mx = max(abs(F.max()), abs(F.min()), 1e-8)
        vmin, vmax = -mx, mx
    else:
        vmin, vmax = F.min(), F.max()
    im = ax.imshow(F, vmin=vmin, vmax=vmax,
                   cmap=plt.cm.RdBu_r, origin="lower", aspect="auto")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_spectrum(ax, sig_1d):
    sig_1d = sig_1d - sig_1d.mean()
    N = len(sig_1d)
    F = np.abs(np.fft.rfft(sig_1d)) / N
    F[1:-1] *= 2
    freqs = np.fft.rfftfreq(N)
    ax.semilogy(freqs[1:], np.maximum(F[1:], 1e-12), lw=1)
    ax.set_xlabel("frequency"); ax.set_ylabel("|FFT|")
    ax.grid(True)


def fig_pod_modes(Phi, sing, Vsvd, comp_names, H, W, n_modes):
    """Figure 1: POD spatial modes + temporal spectra."""
    C = len(comp_names)
    n = min(n_modes, Phi.shape[1])
    ncols = C + 1   # one per velocity component + one spectrum column

    fig = plt.figure(figsize=(7 * ncols, 5 * n), constrained_layout=True)
    gs  = gridspec.GridSpec(n, ncols, figure=fig)

    for i in range(n):
        phi = Phi[:, i].reshape(C, H, W)
        a_i = sing[i] * Vsvd[:, i]

        for c in range(C):
            ax = fig.add_subplot(gs[i, c])
            _imshow(ax, phi[c])
            if i == 0:
                ax.set_title(comp_names[c], fontsize=13, pad=8)
            # mode label as text annotation outside the axes (left side)
            if c == 0:
                ax.text(-0.08, 0.5, f"Mode {i+1}", transform=ax.transAxes,
                        fontsize=11, va="center", ha="right", rotation=90,
                        clip_on=False)

        ax = fig.add_subplot(gs[i, C])
        _plot_spectrum(ax, a_i)
        if i == 0:
            ax.set_title("FFT(aᵢ)", fontsize=13, pad=8)
        ax.tick_params(labelsize=9)

    fig.suptitle("POD spatial modes", fontsize=15)
    plt.show()


def fig_vae_modes(dec, latent_dim, Z, comp_names, H, W, n_modes, std_C, use_delta):
    """Figure 2: beta-VAE modes (delta or direct)."""
    C = len(comp_names)
    latent_var = Z.var(axis=0)
    order = np.argsort(latent_var)[::-1]

    n = min(n_modes, latent_dim)
    ncols = C + 1

    X0 = decode_zero(dec, latent_dim, DEVICE) if use_delta else None

    fig = plt.figure(figsize=(7 * ncols, 5 * n), constrained_layout=True)
    gs  = gridspec.GridSpec(n, ncols, figure=fig)

    for k in range(n):
        i = order[k]
        Xe = decode_unit(dec, i, latent_dim, DEVICE)
        mode = (Xe - X0) if use_delta else Xe
        mode_phys = mode * std_C[:, None, None]

        var_pct = 100 * latent_var[i] / latent_var.sum()
        row_label = f"z{i}  ({var_pct:.1f}%)"

        for c in range(C):
            ax = fig.add_subplot(gs[k, c])
            _imshow(ax, mode_phys[c])
            if k == 0:
                ax.set_title(comp_names[c], fontsize=13, pad=8)
            if c == 0:
                ax.text(-0.08, 0.5, row_label, transform=ax.transAxes,
                        fontsize=11, va="center", ha="right", rotation=90,
                        clip_on=False)

        ax = fig.add_subplot(gs[k, C])
        _plot_spectrum(ax, Z[:, i])
        if k == 0:
            ax.set_title("FFT(rᵢ)", fontsize=13, pad=8)
        ax.tick_params(labelsize=9)

    label = "D(eᵢ) − D(0)" if use_delta else "D(eᵢ)"
    fig.suptitle(f"β-VAE modes  [{label}]", fontsize=15)
    plt.show()


def fig_correlations(Z, Vsvd, sing, latent_dim, beta):
    """Figure 3: |correlation| matrices for VAE and POD."""
    def corr_mat(X):
        C = np.corrcoef(X.T)
        return np.abs(C)

    latent_var = Z.var(axis=0)
    order = np.argsort(latent_var)[::-1]
    Z_ord = Z[:, order]

    n = min(latent_dim, Vsvd.shape[1])
    A_pod = Vsvd[:, :n] * sing[:n]   # [Nt, n]

    C_vae = corr_mat(Z_ord)
    C_pod = corr_mat(A_pod)

    det_vae = np.linalg.det(C_vae)
    det_pod = np.linalg.det(C_pod)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(C_vae, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[0].set_title(f"β-VAE  β={beta:.1e}  det={det_vae:.4f}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(C_pod, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[1].set_title(f"POD (top {n} modes)  det={det_pod:.4f}")
    plt.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("mode index"); ax.set_ylabel("mode index")
    plt.suptitle("|Correlation matrix|")
    plt.tight_layout(); plt.show()


def fig_error_vs_modes(pod_ranks, ek_pod_all, Z, dec, data_n, std_C, tr_idx, val_idx,
                        latent_dim, C, H, W):
    """Figure 4: normalized error (%) vs number of modes."""
    latent_var = Z.var(axis=0)
    order = np.argsort(latent_var)[::-1]

    def compute_sse(idx_set, active_dims):
        SSE = 0.0; EN = 0.0
        with torch.no_grad():
            for t in idx_set:
                z_full = torch.zeros(1, latent_dim, device=DEVICE)
                z_full[0, active_dims] = torch.tensor(Z[t, active_dims])
                xhat = dec(z_full).cpu().numpy()[0]   # [C, H, W]
                diff = data_n[t] - xhat
                for c in range(C):
                    SSE += (diff[c] ** 2).sum() * std_C[c]**2
                    EN  += (data_n[t, c] ** 2).sum() * std_C[c]**2
        return SSE / max(EN, 1e-12)

    print("Computing VAE error vs number of active latent dims ...")
    err_tr = np.zeros(latent_dim)
    err_va = np.zeros(latent_dim) if len(val_idx) > 0 else None
    for k in range(1, latent_dim + 1):
        active = order[:k]
        err_tr[k-1] = compute_sse(tr_idx, active)
        if err_va is not None:
            err_va[k-1] = compute_sse(val_idx, active)
        print(f"  k={k}/{latent_dim}  err_tr={err_tr[k-1]*100:.2f}%")

    plt.figure(figsize=(9, 5))
    plt.plot(pod_ranks, 100 - ek_pod_all, "o-", lw=2, label="POD")
    plt.plot(np.arange(1, latent_dim+1), err_tr * 100, "-", lw=2,
             label=f"β-VAE d={latent_dim} (train)")
    if err_va is not None:
        plt.plot(np.arange(1, latent_dim+1), err_va * 100, "--", lw=2,
                 label=f"β-VAE d={latent_dim} (val)")
    plt.xlabel("Number of modes"); plt.ylabel("Normalized error (%)")
    plt.title("Reconstruction error vs modes")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(pod_ranks, ek_pod_all, "o-", lw=2, label="POD")
    plt.plot(np.arange(1, latent_dim+1), (1 - err_tr) * 100, "-", lw=2,
             label=f"β-VAE d={latent_dim} (train)")
    if err_va is not None:
        plt.plot(np.arange(1, latent_dim+1), (1 - err_va) * 100, "--", lw=2,
                 label=f"β-VAE d={latent_dim} (val)")
    plt.xlabel("Number of modes"); plt.ylabel("Captured energy (%)")
    plt.title("Captured energy vs modes")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


def fig_field_comparison(data, data_n, enc, dec, Phi, sing, Vsvd, Umean,
                          pod_rank, t_plot, mu_C, std_C, comp_names, C, H, W):
    """Figure 5: TRUE | POD | VAE for each component."""
    # VAE reconstruction
    with torch.no_grad():
        x = torch.tensor(data_n[t_plot:t_plot+1], device=DEVICE)
        mu_z, _ = enc(x)
        xhat = dec(mu_z).cpu().numpy()[0]   # [C, H, W]
    vae_rec = xhat * std_C[:, None, None] + mu_C[:, None, None]

    # POD reconstruction
    r = min(pod_rank, Phi.shape[1])
    coeff  = sing[:r] * Vsvd[t_plot, :r]
    uf_hat = Phi[:, :r] @ coeff
    u_full = uf_hat + Umean[:, 0]
    pod_rec = u_full.reshape(C, H, W).astype(np.float32)

    true = data[t_plot]   # [C, H, W]

    fig, axes = plt.subplots(C, 3, figsize=(12, 4*C), squeeze=False)
    for c in range(C):
        mx = max(abs(float(true[c].min())), abs(float(true[c].max())), 1e-8)
        vmin, vmax = -mx, mx
        for j, (field, lbl) in enumerate([(true, "TRUE"),
                                           (pod_rec, f"POD r={r}"),
                                           (vae_rec, "β-VAE")]):
            im = axes[c, j].imshow(field[c], vmin=vmin, vmax=vmax,
                                   cmap="RdBu_r", origin="lower", aspect="auto")
            axes[c, j].set_title(f"{comp_names[c]}  {lbl}")
            axes[c, j].axis("off")
            plt.colorbar(im, ax=axes[c, j])
    plt.suptitle(f"Field comparison  t={t_plot}")
    plt.tight_layout(); plt.show()

    # Error maps
    fig, axes = plt.subplots(C, 2, figsize=(8, 4*C), squeeze=False)
    for c in range(C):
        for j, (rec, lbl) in enumerate([(vae_rec, "β-VAE error"),
                                         (pod_rec, f"POD error r={r}")]):
            err = rec[c] - true[c]
            mx  = max(abs(err.max()), abs(err.min()), 1e-8)
            im = axes[c, j].imshow(err, vmin=-mx, vmax=mx,
                                   cmap="RdBu_r", origin="lower", aspect="auto")
            axes[c, j].set_title(f"{comp_names[c]}  {lbl}")
            axes[c, j].axis("off")
            plt.colorbar(im, ax=axes[c, j])
    plt.suptitle(f"Error maps  t={t_plot}")
    plt.tight_layout(); plt.show()


# ─────────────────────────── Main ───────────────────────────

def main():
    global VAE_FILE, POD_FILE, DATA_FILE

    if not VAE_FILE:
        VAE_FILE = pick_file("Select beta-VAE checkpoint (.pt)",
                             start_dir=_MODELS_DIR, ftype="pt")
    if not POD_FILE:
        POD_FILE = pick_file("Select POD results (.npz)",
                             start_dir=_MODELS_DIR, ftype="npz")

    # Load VAE
    enc, dec, ckpt = load_vae(VAE_FILE)
    H, W, C       = ckpt["H"], ckpt["W"], ckpt["C"]
    latent_dim    = ckpt["latent_dim"]
    mu_C          = ckpt["mu_C"]     # [C]
    std_C         = ckpt["std_C"]    # [C]
    comp_names    = list(ckpt["comp_names"])
    beta          = ckpt["beta"]
    tr_idx        = ckpt.get("tr_idx", np.array([], dtype=int))
    val_idx       = ckpt.get("val_idx", np.array([], dtype=int))

    # Load POD
    pod = np.load(POD_FILE, allow_pickle=True)
    Phi   = pod["Phi"].astype(np.float64)     # [Ndof, n_modes]
    sing  = pod["sing"].astype(np.float64)    # [n_modes]
    Vsvd  = pod["Vsvd"].astype(np.float64)    # [Nt, n_modes]
    Umean = pod["Umean"].astype(np.float64)   # [Ndof, 1]  (or flat)
    if Umean.ndim == 1:
        Umean = Umean[:, None]
    pod_ranks  = pod["ranks"].tolist()
    ek_pod_all = pod["ek_all"]
    t_plot = int(pod.get("t_plot", 0))

    # Load data
    data_file = DATA_FILE or str(ckpt.get("data_file", ""))
    if not data_file or not os.path.isfile(data_file):
        data_file = pick_file("Select .mat data file",
                              start_dir=_DATA_DIR, ftype="mat")

    data, _ = load_data(data_file)
    Nt = data.shape[0]

    # Normalize (using VAE mu/std)
    data_n = (data - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    # Encode all snapshots
    print(f"\nEncoding {Nt} snapshots ...")
    Z = encode_all(enc, data_n, DEVICE)   # [Nt, latent_dim]

    # Choose POD rank for field comparison
    if COMPARE_RANK is not None:
        pod_rank = COMPARE_RANK
    else:
        # pick rank whose Ek_all is closest to VAE Ek
        # compute VAE Ek_all on all data
        SSE_vae = 0.0; EN_vae = 0.0
        fluc_mean = data.mean(axis=0)
        with torch.no_grad():
            for t in range(Nt):
                x = torch.tensor(data_n[t:t+1], device=DEVICE)
                mu_z, _ = enc(x)
                xhat = dec(mu_z).cpu().numpy()[0]
                rec  = xhat * std_C[:, None, None] + mu_C[:, None, None]
                fluc_rec  = rec - fluc_mean
                fluc_true = data[t] - fluc_mean
                SSE_vae += ((fluc_true - fluc_rec)**2).sum()
                EN_vae  += (fluc_true**2).sum()
        ek_vae = (1 - SSE_vae / max(EN_vae, 1e-12)) * 100
        print(f"VAE Ek_all = {ek_vae:.2f}%")
        best_i = int(np.argmin(np.abs(ek_pod_all - ek_vae)))
        pod_rank = pod_ranks[best_i]
        print(f"Closest POD rank: {pod_rank}  (Ek_pod={ek_pod_all[best_i]:.2f}%)")

    # ── Figures ──
    print("\n--- Figure 1: POD spatial modes ---")
    fig_pod_modes(Phi, sing, Vsvd, comp_names, H, W, N_MODES_PLOT)

    print("--- Figure 2: beta-VAE modes ---")
    fig_vae_modes(dec, latent_dim, Z, comp_names, H, W, N_MODES_PLOT, std_C, USE_DELTA)

    print("--- Figure 3: Correlation matrices ---")
    fig_correlations(Z, Vsvd, sing, latent_dim, beta)

    print("--- Figure 4: Error vs modes ---")
    fig_error_vs_modes(pod_ranks, ek_pod_all, Z, dec, data_n, std_C,
                        tr_idx, val_idx, latent_dim, C, H, W)

    print(f"--- Figure 5: Field comparison at t={t_plot} ---")
    fig_field_comparison(data, data_n, enc, dec,
                          Phi, sing, Vsvd, Umean,
                          pod_rank, t_plot, mu_C, std_C,
                          comp_names, C, H, W)


if __name__ == "__main__":
    main()

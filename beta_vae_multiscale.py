"""
beta-VAE Multiscale — Isotropic Turbulence (BOX dataset)

Splits the 2D velocity field into large and small scales via a circular
spectral filter at cutoff wavenumber K_CUT, trains one beta-VAE per scale,
then reconstructs the field as the sum of both and compares against a
pre-trained full-field beta-VAE.

Data: BOX_Turbulence / U [Nt, Nz, Nx, 3]  →  uses (u, w)

"""

import os, math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Folder layout ─────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "bestModels"))
os.makedirs(_MODELS_DIR, exist_ok=True)

# ══════════════════════════ CONFIG ══════════════════════════
DATA_FILE = (
    "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/"
    "BOX_Turbulence/isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.mat"
)

# Path to a pre-trained full-field beta-VAE .pt (from beta_vae.py).
# Leave "" to skip the comparison panel.
FULL_MODEL_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/bestModels/model_betaVAE_isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW_lat5_b8e-04_ep500.pt"

K_CUT      = 9    # cutoff wavenumber: large scales k ≤ K_CUT, small scales k > K_CUT

LATENT_DIM_LARGE = 2   # latent dim for the large-scale VAE (0 = skip this scale)
LATENT_DIM_SMALL = 3   # latent dim for the small-scale VAE (0 = skip this scale)
BETA       = 8e-4
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4
VAL_FRAC   = 0.05
RNG_SEED   = 7
VAL_EVERY  = None  # None = auto (every ~10 % of epochs)
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────── Data ───────────────────────────

def _read_field(S, key, is_hdf5):
    arr = np.array(S[key], dtype=np.float32)
    if is_hdf5:
        arr = arr.T
    return arr


def load_data(path):
    """Load BOX .mat → data [Nt, 2, H, W] float32, comp_names=['u','w']."""
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
        if "U" in S:
            V = _read_field(S, "U", is_hdf5)       # [Nt, Nz, Nx, 3]
            V = V[:, :, :, [0, 2]]                  # keep u, w
            data = V.transpose(0, 3, 1, 2)          # [Nt, 2, Nz, Nx]
            names = ["u", "w"]
        else:
            raise ValueError("Expected field 'U' in BOX data.")
    finally:
        if fh is not None:
            fh.close()

    print(f"Loaded  {os.path.basename(path)}  →  {data.shape}  comps={names}")
    return data, names


def spectral_filter(data, k_cut):
    """
    Circular spectral filter on a [Nt, C, H, W] field.
    Returns data_large (k ≤ k_cut) and data_small (k > k_cut).
    Their sum equals the original field up to floating-point precision.
    """
    Nt, C, H, W = data.shape

    kh = np.fft.fftfreq(H) * H
    kw = np.fft.fftfreq(W) * W
    KH, KW = np.meshgrid(kh, kw, indexing="ij")
    low_mask  = (np.sqrt(KH**2 + KW**2) <= k_cut).astype(np.float32)   # [H, W]
    high_mask = 1.0 - low_mask

    # Batch FFT over all (Nt * C) slices at once for speed
    flat  = data.reshape(Nt * C, H, W)
    F     = np.fft.fft2(flat, axes=(-2, -1))

    large = np.real(np.fft.ifft2(F * low_mask,  axes=(-2, -1))).astype(np.float32)
    small = np.real(np.fft.ifft2(F * high_mask, axes=(-2, -1))).astype(np.float32)

    return large.reshape(Nt, C, H, W), small.reshape(Nt, C, H, W)


def normalize(data):
    """Per-channel z-score → (data_n, mu [C], std [C])."""
    mu  = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (data - mu) / std, mu[0, :, 0, 0], std[0, :, 0, 0]


# ─────────────────────────── Network ────────────────────────

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


# ─────────────────────────── Training ───────────────────────

def _vae_loss(enc, dec, x, beta):
    mu, logv = enc(x)
    logv = torch.clamp(logv, -10, 10)
    z    = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
    xhat = dec(z)
    recon = F.mse_loss(xhat, x)
    kl    = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
    return recon + beta * kl, recon, kl


def _val_loss(enc, dec, loader, beta, device):
    enc.eval(); dec.eval()
    tot_l = tot_r = tot_k = 0.0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            mu, logv = enc(xb)
            logv = torch.clamp(logv, -10, 10)
            xhat = dec(mu)
            r = F.mse_loss(xhat, xb)
            k = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
            tot_l += (r + beta * k).item()
            tot_r += r.item(); tot_k += k.item()
    n = len(loader)
    return tot_l / n, tot_r / n, tot_k / n


def train_vae(data_n, mu_C, std_C, C, H, W, label, save_path, latent_dim):
    """
    Train a beta-VAE on normalized data [Nt, C, H, W].
    Saves checkpoint and returns (enc, dec).
    """
    Nt = data_n.shape[0]
    rng = np.random.default_rng(RNG_SEED)
    idx     = rng.permutation(Nt)
    n_val   = max(1, round(VAL_FRAC * Nt))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_tr = torch.tensor(data_n[tr_idx], dtype=torch.float32)
    X_va = torch.tensor(data_n[val_idx], dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    enc = Encoder(H, W, C, latent_dim).to(DEVICE)
    dec = Decoder(H, W, C, latent_dim).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every  = VAL_EVERY or max(1, round(N_EPOCHS / 10))
    best_val   = float("inf")
    best_state = None

    print(f"\n{'='*58}")
    print(f"  β-VAE  [{label}]  |  {H}×{W}  C={C}  Nt={Nt}")
    print(f"  latent={latent_dim}  β={BETA}  lr={LR}  epochs={N_EPOCHS}  device={DEVICE}")
    print(f"{'='*58}\n")

    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        s_l = s_r = s_k = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, r, k = _vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            s_l += loss.item(); s_r += r.item(); s_k += k.item()
        nb = len(tr_loader)

        if (epoch % val_every == 0) or (epoch == N_EPOCHS):
            vl, _, _ = _val_loss(enc, dec, va_loader, BETA, DEVICE)
            marker = ""
            if vl < best_val:
                best_val = vl
                best_state = {
                    "enc": {kk: vv.cpu().clone() for kk, vv in enc.state_dict().items()},
                    "dec": {kk: vv.cpu().clone() for kk, vv in dec.state_dict().items()}}
                marker = "  ★"
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l/nb:.5f} | va {vl:.5f}{marker}")
        else:
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l/nb:.5f}")

    enc.load_state_dict(best_state["enc"]); enc.to(DEVICE).eval()
    dec.load_state_dict(best_state["dec"]); dec.to(DEVICE).eval()

    torch.save({
        "enc_state": best_state["enc"],
        "dec_state": best_state["dec"],
        "mu_C": mu_C, "std_C": std_C,
        "H": H, "W": W, "C": C,
        "latent_dim": latent_dim, "beta": BETA,
        "k_cut": K_CUT, "scale": label,
    }, save_path)
    print(f"Saved → {save_path}\n")

    return enc, dec


# ─────────────────────────── Helpers ────────────────────────

def reconstruct(enc, dec, data_n, t, mu_C, std_C):
    """Reconstruct snapshot t → physical field [C, H, W]."""
    enc.eval(); dec.eval()
    with torch.no_grad():
        x = torch.tensor(data_n[t:t+1], dtype=torch.float32, device=DEVICE)
        mu_z, _ = enc(x)
        xhat = dec(mu_z).cpu().numpy()[0]
    return xhat * std_C[:, None, None] + mu_C[:, None, None]


def compute_ek_dataset(enc, dec, data_n, data_phys, mu_C, std_C, phys_mean):
    """Ek (%) on fluctuations over the full dataset."""
    Nt, C = data_n.shape[:2]
    SSE = np.zeros(C); EN = np.zeros(C)
    enc.eval(); dec.eval()
    with torch.no_grad():
        for t in range(Nt):
            x = torch.tensor(data_n[t:t+1], device=DEVICE)
            mu_z, _ = enc(x)
            xhat = dec(mu_z).cpu().numpy()[0]
            rec_phys  = xhat * std_C[:, None, None] + mu_C[:, None, None]
            true_fluc = data_phys[t] - phys_mean
            rec_fluc  = rec_phys    - phys_mean
            err = true_fluc - rec_fluc
            for c in range(C):
                SSE[c] += (err[c]**2).sum()
                EN[c]  += (true_fluc[c]**2).sum()
    EN = np.where(EN < 1e-12, 1e-12, EN)
    return (1 - SSE.sum() / EN.sum()) * 100


def compute_ek_multiscale(enc_L, dec_L, data_large_n, mu_L, std_L,
                          enc_S, dec_S, data_small_n, mu_S, std_S,
                          data_phys, phys_mean):
    """Ek (%) of the multiscale sum over the full dataset.

    A scale whose encoder is None is skipped: it contributes zero to the
    reconstruction, while the truth remains the full field, so its missing
    energy lowers Ek.
    """
    Nt, C = data_phys.shape[:2]
    SSE = np.zeros(C); EN = np.zeros(C)
    for e, d in ((enc_L, dec_L), (enc_S, dec_S)):
        if e is not None:
            e.eval(); d.eval()
    with torch.no_grad():
        for t in range(Nt):
            rec_phys = np.zeros_like(data_phys[t])
            if enc_L is not None:
                rec_phys = rec_phys + reconstruct(enc_L, dec_L, data_large_n, t, mu_L, std_L)
            if enc_S is not None:
                rec_phys = rec_phys + reconstruct(enc_S, dec_S, data_small_n, t, mu_S, std_S)
            true_fluc = data_phys[t] - phys_mean
            rec_fluc  = rec_phys    - phys_mean
            err = true_fluc - rec_fluc
            for c in range(C):
                SSE[c] += (err[c]**2).sum()
                EN[c]  += (true_fluc[c]**2).sum()
    EN = np.where(EN < 1e-12, 1e-12, EN)
    return (1 - SSE.sum() / EN.sum()) * 100


def ring_spectrum(field_2d):
    """Azimuthally averaged 1D power spectrum of a 2D field."""
    H, W = field_2d.shape
    kh = np.fft.fftfreq(H) * H
    kw = np.fft.fftfreq(W) * W
    KH, KW = np.meshgrid(kh, kw, indexing="ij")
    k_int = np.round(np.sqrt(KH**2 + KW**2)).astype(int).ravel()
    power = np.abs(np.fft.fft2(field_2d)).ravel() ** 2
    k_max = k_int.max() + 1
    E   = np.bincount(k_int, weights=power,          minlength=k_max).astype(float)
    cnt = np.bincount(k_int,                          minlength=k_max).astype(float)
    k_arr = np.arange(k_max)
    return k_arr, E / np.where(cnt > 0, cnt, 1.0)


# ─────────────────────────── Main ───────────────────────────

def main():
    if LATENT_DIM_LARGE <= 0 and LATENT_DIM_SMALL <= 0:
        raise ValueError("Both scales have latent_dim ≤ 0 — nothing to train.")

    # ── Load raw data ──────────────────────────────────────
    data, comp_names = load_data(DATA_FILE)
    Nt, C, H, W = data.shape
    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    phys_mean = data.mean(axis=0)   # [C, H, W]

    # ── Spectral filter ────────────────────────────────────
    print(f"\nApplying spectral filter  k_cut = {K_CUT} ...")
    data_large, data_small = spectral_filter(data, K_CUT)
    e_tot   = float((data**2).mean())
    e_large = float((data_large**2).mean()) / max(e_tot, 1e-12)
    e_small = float((data_small**2).mean()) / max(e_tot, 1e-12)
    print(f"  Large-scale energy fraction : {e_large:.3f}")
    print(f"  Small-scale energy fraction : {e_small:.3f}")

    # ── Normalise each scale independently ─────────────────
    data_large_n, mu_L, std_L = normalize(data_large)
    data_small_n, mu_S, std_S = normalize(data_small)

    # ── Train large-scale VAE (or skip) ────────────────────
    if LATENT_DIM_LARGE > 0:
        path_L = os.path.join(_MODELS_DIR,
            f"ms_large_k{K_CUT}_{base}_lat{LATENT_DIM_LARGE}_ep{N_EPOCHS}.pt")
        enc_L, dec_L = train_vae(data_large_n, mu_L, std_L, C, H, W,
                                 f"LARGE  k ≤ {K_CUT}", path_L, LATENT_DIM_LARGE)
    else:
        print(f"\nLarge scale  k ≤ {K_CUT}  latent=0  →  SKIPPED "
              f"(no VAE; contributes 0 to reconstruction)")
        enc_L = dec_L = None

    # ── Train small-scale VAE (or skip) ────────────────────
    if LATENT_DIM_SMALL > 0:
        path_S = os.path.join(_MODELS_DIR,
            f"ms_small_k{K_CUT}_{base}_lat{LATENT_DIM_SMALL}_ep{N_EPOCHS}.pt")
        enc_S, dec_S = train_vae(data_small_n, mu_S, std_S, C, H, W,
                                 f"SMALL  k > {K_CUT}", path_S, LATENT_DIM_SMALL)
    else:
        print(f"\nSmall scale  k > {K_CUT}  latent=0  →  SKIPPED "
              f"(no VAE; contributes 0 to reconstruction)")
        enc_S = dec_S = None

    # ── Load full model (optional) ─────────────────────────
    enc_F = dec_F = mu_F = std_F = data_full_n = None
    if FULL_MODEL_FILE and os.path.isfile(FULL_MODEL_FILE):
        ckpt = torch.load(FULL_MODEL_FILE, map_location="cpu", weights_only=False)
        enc_F = Encoder(H, W, C, ckpt["latent_dim"])
        dec_F = Decoder(H, W, C, ckpt["latent_dim"])
        enc_F.load_state_dict(ckpt["enc_state"]); enc_F.to(DEVICE).eval()
        dec_F.load_state_dict(ckpt["dec_state"]); dec_F.to(DEVICE).eval()
        mu_F  = ckpt["mu_C"]; std_F = ckpt["std_C"]
        data_full_n = (data - mu_F[None, :, None, None]) / std_F[None, :, None, None]
        print(f"Loaded full model: {os.path.basename(FULL_MODEL_FILE)}")
    else:
        print("No full model provided — skipping comparison.")

    # ── Reconstruct a mid-dataset snapshot ─────────────────
    t_plot    = Nt // 2
    rec_large = (reconstruct(enc_L, dec_L, data_large_n, t_plot, mu_L, std_L)
                 if enc_L is not None else np.zeros((C, H, W), dtype=np.float32))
    rec_small = (reconstruct(enc_S, dec_S, data_small_n, t_plot, mu_S, std_S)
                 if enc_S is not None else np.zeros((C, H, W), dtype=np.float32))
    rec_multi = rec_large + rec_small
    true_field = data[t_plot]

    # ── Ek metrics ─────────────────────────────────────────
    print("\nComputing Ek metrics (full dataset) ...")
    ek_multi = compute_ek_multiscale(enc_L, dec_L, data_large_n, mu_L, std_L,
                                     enc_S, dec_S, data_small_n, mu_S, std_S,
                                     data, phys_mean)
    ek_large = (compute_ek_dataset(enc_L, dec_L, data_large_n, data_large, mu_L, std_L,
                                   data_large.mean(axis=0)) if enc_L is not None else None)
    ek_small = (compute_ek_dataset(enc_S, dec_S, data_small_n, data_small, mu_S, std_S,
                                   data_small.mean(axis=0)) if enc_S is not None else None)

    _fmt = lambda v: "SKIPPED" if v is None else f"{v:.2f}%"
    print(f"\n{'='*52}")
    print(f"  Ek large-scale VAE  (own scale) : {_fmt(ek_large)}")
    print(f"  Ek small-scale VAE  (own scale) : {_fmt(ek_small)}")
    print(f"  Ek multiscale sum   (full field): {ek_multi:.2f}%")

    rec_full = None
    if enc_F is not None:
        ek_full = compute_ek_dataset(enc_F, dec_F, data_full_n, data, mu_F, std_F, phys_mean)
        print(f"  Ek full model       (full field): {ek_full:.2f}%")
        rec_full = reconstruct(enc_F, dec_F, data_full_n, t_plot, mu_F, std_F)
    print(f"{'='*52}\n")

    # ── Figure 1: field panels ─────────────────────────────
    for c in range(C):
        lab_L = "VAE large: skipped" if enc_L is None else f"VAE large  k≤{K_CUT}"
        lab_S = "VAE small: skipped" if enc_S is None else f"VAE small  k>{K_CUT}"
        panels = [
            (true_field[c],  "True"),
            (data_large[t_plot, c], f"Large scale  k≤{K_CUT}"),
            (data_small[t_plot, c], f"Small scale  k>{K_CUT}"),
            (rec_large[c],   lab_L),
            (rec_small[c],   lab_S),
            (rec_multi[c],   f"Multiscale sum  Ek={ek_multi:.1f}%"),
        ]
        if rec_full is not None:
            panels.append((rec_full[c], f"Full model  Ek={ek_full:.1f}%"))

        n_cols = len(panels)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), squeeze=False)

        mx = max(abs(float(true_field[c].min())),
                 abs(float(true_field[c].max())), 1e-8)
        vmin, vmax = -mx, mx

        for j, (field, title) in enumerate(panels):
            im = axes[0, j].imshow(field, vmin=vmin, vmax=vmax,
                                   cmap="RdBu_r", origin="lower")
            axes[0, j].set_title(title, fontsize=9)
            axes[0, j].axis("off")
            plt.colorbar(im, ax=axes[0, j])

        plt.suptitle(f"{comp_names[c]}  —  t={t_plot}  |  k_cut={K_CUT}", fontsize=11)
        plt.tight_layout()
        plt.show()

    # ── Figure 2: error maps (multiscale vs full) ──────────
    if rec_full is not None:
        for c in range(C):
            err_multi = rec_multi[c]  - true_field[c]
            err_full  = rec_full[c]   - true_field[c]
            mx_e = max(abs(float(err_multi.min())), abs(float(err_multi.max())),
                       abs(float(err_full.min())),  abs(float(err_full.max())), 1e-8)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
            for j, (err, title) in enumerate([
                    (err_multi, f"Error — Multiscale  Ek={ek_multi:.1f}%"),
                    (err_full,  f"Error — Full model  Ek={ek_full:.1f}%")]):
                im = axes[0, j].imshow(err, vmin=-mx_e, vmax=mx_e,
                                       cmap="RdBu_r", origin="lower")
                axes[0, j].set_title(title, fontsize=10)
                axes[0, j].axis("off")
                plt.colorbar(im, ax=axes[0, j])

            plt.suptitle(f"{comp_names[c]} error  |  k_cut={K_CUT}  t={t_plot}", fontsize=11)
            plt.tight_layout()
            plt.show()

    # ── Figure 3: energy spectra ───────────────────────────
    fig, axes = plt.subplots(1, C, figsize=(7 * C, 5), squeeze=False)
    for c in range(C):
        ax = axes[0, c]
        k, E_true  = ring_spectrum(true_field[c])
        _, E_multi = ring_spectrum(rec_multi[c])

        ax.semilogy(k[1:], E_true[1:],  "-",  lw=2,   label="True",            color="black")
        if enc_L is not None:
            _, E_large = ring_spectrum(rec_large[c])
            ax.semilogy(k[1:], E_large[1:], "--", lw=1.5, label=f"VAE large k≤{K_CUT}", color="royalblue")
        if enc_S is not None:
            _, E_small = ring_spectrum(rec_small[c])
            ax.semilogy(k[1:], E_small[1:], "-.", lw=1.5, label=f"VAE small k>{K_CUT}", color="tomato")
        ax.semilogy(k[1:], E_multi[1:], ":",  lw=2,   label="Multiscale sum",   color="green")
        if rec_full is not None:
            _, E_full = ring_spectrum(rec_full[c])
            ax.semilogy(k[1:], E_full[1:], "-", lw=1.5, label="Full model",
                        color="orange", alpha=0.8)

        ax.axvline(K_CUT, color="gray", linestyle="--", lw=1, label=f"k_cut={K_CUT}")
        ax.set_xlabel("Wavenumber k"); ax.set_ylabel("Energy")
        ax.set_title(f"Energy spectrum — {comp_names[c]}")
        ax.legend(fontsize=8); ax.grid(True)

    plt.suptitle(f"Energy spectra  |  k_cut={K_CUT}", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

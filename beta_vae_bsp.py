"""
beta-VAE for turbulent flow ROM  —  with Binned Spectral Power (BSP) loss.

Single network (one encoder + one decoder), trained to reconstruct the field.
The standard VAE objective (pixel-wise MSE + beta*KL) is augmented with the
Binned Spectral Power loss of

    Chakraborty, Mohan & Maulik (2026),
    "Binned Spectral Power Loss for Improved Prediction of Chaotic Systems",
    J. Comput. Phys. 114866.  (arXiv:2502.00472v3)

The BSP loss compares the *binned energy spectra* of the reconstruction and the
true field (Eq. 7 / Algorithm 1 of the paper) so that small-scale (high
wavenumber) structures are reproduced, mitigating the spectral bias of the
network.  Final objective (combined form, Eq. 8 of the paper):

    L  =  MSE(xhat, x)  +  beta * KL  +  mu * L_BSP(xhat, x)

with

                1     C   N_k (        E_u^bin(c,i) + eps  )^2
    L_BSP  =  -------  Σ   Σ   ( 1 -  --------------------  )
              N_k * C  c=1 i=1 (        E_v^bin(c,i) + eps  )

where u = reconstruction (numerator), v = true field, E = 1/2 |FFT|^2 binned by
radial wavenumber magnitude k = sqrt(kx^2 + ky^2), and each bin is optionally
weighted by lambda_i (default k^2, Sobolev-like, as used in the paper's
turbulence experiments).
"""

import os, math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Folder layout ──
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "bestModels"))
os.makedirs(_MODELS_DIR, exist_ok=True)

# ══ CONFIG ══
#   DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/CHANNEL_Turbulence/channel_256x256_y0.0000_Nt4000_UVW.mat"
#   DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.mat"
#   DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/ModelFLOW/TensorRe160.mat"
DATA_FILE  = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.mat"
COMP_IDX   = None    # None = auto-detect; [0,2] = pick u,w from 3-comp field

LATENT_DIM = 4
BETA       = 8e-4
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4
VAL_FRAC   = 0.05
RNG_SEED   = 7
VAL_EVERY  = None    # None = auto (every ~10% of epochs)
SAVE_MODEL = True    # save best checkpoint as .pt file

# ── BSP (Binned Spectral Power) loss settings ──
USE_BSP      = True       # add the BSP term to the objective
MU           = 1.0        # weight of the BSP term (mu in Eq. 8)
N_BINS       = 16         # number of radial wavenumber bins (N_k)
BSP_EPS      = 1e-8       # eps in the ratio; smaller -> stronger small-scale focus
LAMBDA_MODE  = "k2"       # "k2" (Sobolev-like, paper's turbulence default) or "unity"
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def _read_field(S, key, is_hdf5):
    """Read a numerical array from scipy dict or h5py file.
    h5py stores MATLAB arrays with all dimensions reversed, so we transpose back."""
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
            T = _read_field(S, "Tensor", is_hdf5)        # [C, N1, N2, Nt]
            C, N1, N2, Nt = T.shape
            data = T.transpose(3, 0, 1, 2)               # [Nt, C, N1, N2]
            names = {2: ["u", "v"], 3: ["u", "v", "w"]}.get(C, [f"c{i}" for i in range(C)])

        elif "U" in S:
            V = _read_field(S, "U", is_hdf5)             # [Nt, Nz, Nx, C_all]
            Nt, Nz, Nx, C_all = V.shape
            if comp_idx is None:
                comp_idx = [0, 2] if C_all == 3 else list(range(C_all))
            V = V[:, :, :, comp_idx]
            data = V.transpose(0, 3, 1, 2)               # [Nt, C, Nz, Nx]
            all_names = ["u", "v", "w"]
            names = [all_names[i] for i in comp_idx]

        elif "UW" in S:
            V = _read_field(S, "UW", is_hdf5)            # [Nt, Nz, Nx, 2]
            data = V.transpose(0, 3, 1, 2)               # [Nt, 2, Nz, Nx]
            names = ["u", "w"]

        else:
            visible = [k for k in S.keys() if not k.startswith("#")]
            raise ValueError(f"Unknown format. Fields: {visible}")

    finally:
        if fh is not None:
            fh.close()

    print(f"Loaded  {os.path.basename(path)}  →  {data.shape}  comps={names}")
    return data, names


def normalize(data):
    """Per-channel z-score. Returns data_n, mu [C], std [C]."""
    mu  = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (data - mu) / std, mu[0, :, 0, 0], std[0, :, 0, 0]


def mean_field(data):
    """Time-averaged mean field [C, H, W]."""
    return data.mean(axis=0)


# ──────────────────── Binned Spectral Power loss ────────────────────

class BSPLoss(nn.Module):
    """Binned Spectral Power loss (Chakraborty, Mohan & Maulik 2026, Eq. 7).

    Compares the radially-binned 2-D energy spectra of a prediction `u`
    (e.g. the reconstruction, kept in the numerator as the paper recommends)
    and a target `v` (the true field).  Both are [B, C, H, W] real tensors.

    The bin membership, mode counts N_i, and per-bin weights lambda_i depend
    only on (H, W, N_bins) — not on the network — so the whole operation is a
    fixed linear map of the (differentiable) per-mode energies and back-props
    cleanly through the FFT.
    """

    def __init__(self, H, W, n_bins=16, eps=1e-8, lambda_mode="k2"):
        super().__init__()
        self.eps = float(eps)

        # Integer wavenumber grid (cycles over the periodic box).
        ky = torch.fft.fftfreq(H) * H          # [H]  values in [-H/2, H/2)
        kx = torch.fft.fftfreq(W) * W          # [W]
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        kmag = torch.sqrt(KX**2 + KY**2)       # [H, W]

        kmax = min(H, W) / 2.0                  # Nyquist radius
        edges = torch.linspace(0.0, kmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])          # [n_bins]

        # Assign each Fourier mode to a bin; modes beyond kmax (spectral
        # corners) are dropped.
        idx = torch.bucketize(kmag.flatten(), edges) - 1   # [H*W], in [-1, n_bins]
        idx = idx.clamp(0, n_bins - 1)
        valid = (kmag.flatten() <= kmax)

        # Membership matrix M [n_bins, H*W]; M[i, m] = 1 if mode m is in bin i.
        M = torch.zeros(n_bins, H * W)
        m_ix = torch.arange(H * W)[valid]
        M[idx[valid], m_ix] = 1.0

        counts = M.sum(dim=1).clamp(min=1.0)               # N_i per bin

        if lambda_mode == "k2":
            lam = centers**2
        elif lambda_mode == "unity":
            lam = torch.ones(n_bins)
        else:
            raise ValueError(f"Unknown LAMBDA_MODE '{lambda_mode}' (use 'k2' or 'unity').")

        # Per-bin coefficient applied to the summed energy:  lambda_i / N_i.
        coef = (lam / counts)                              # [n_bins]

        # Register as buffers so they move with .to(device) and are saved/loaded.
        self.register_buffer("M", M)            # [n_bins, H*W]
        self.register_buffer("coef", coef)      # [n_bins]

    def _binned_energy(self, x):
        """x [B, C, H, W] -> binned, weighted energy [B, C, n_bins]."""
        B, C, H, W = x.shape
        Fx = torch.fft.fft2(x, dim=(-2, -1))               # complex [B,C,H,W]
        E  = 0.5 * (Fx.real**2 + Fx.imag**2)               # energy per mode
        E  = E.reshape(B, C, H * W)
        E_bin = torch.einsum("bcm,im->bci", E, self.M)     # sum energy in each bin
        return E_bin * self.coef                            # apply lambda_i / N_i

    def forward(self, u, v):
        Eu = self._binned_energy(u)
        Ev = self._binned_energy(v)
        ratio = (Eu + self.eps) / (Ev + self.eps)
        return ((1.0 - ratio) ** 2).mean()                  # mean over B, C, bins


# ─────────────────────────── Network ───────────────────────

def _n_conv(H, W):
    """Number of stride-2 conv layers."""
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
        flat = Hb * Wb * chs[-1]
        self.fc   = nn.Sequential(nn.Linear(flat, 256), nn.ELU())
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

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, Hb * Wb * ch_b), nn.ELU())

        rev = list(reversed(chs))          # e.g. [256, 128, 64, 32, 16, 8]
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
        # center-crop to target spatial size
        h, w = x.shape[2], x.shape[3]
        r0 = (h - self.H) // 2
        c0 = (w - self.W) // 2
        return x[:, :, r0:r0+self.H, c0:c0+self.W]


# ─────────────────────────── Metrics ───────────────────────

def compute_ek(enc, dec, data_n, fluc_true, mu_C, std_C, phys_mean, device):
    """Ek (%) = (1 - SSE/EN)*100 on fluctuations for all snapshots.
    phys_mean: [C, H, W] time-averaged physical mean field."""
    enc.eval(); dec.eval()
    Nt, C, H, W = data_n.shape
    SSE = np.zeros(C); EN = np.zeros(C)
    with torch.no_grad():
        for t in range(Nt):
            x = torch.tensor(data_n[t:t+1], device=device)
            mu, _ = enc(x)
            xhat = dec(mu).cpu().numpy()[0]               # [C, H, W]
            rec_phys = xhat * std_C[:, None, None] + mu_C[:, None, None]
            rec_fluc = rec_phys - phys_mean
            err = fluc_true[t] - rec_fluc
            for c in range(C):
                SSE[c] += (err[c]**2).sum()
                EN[c]  += (fluc_true[t, c]**2).sum()
    EN = np.where(EN < 1e-12, 1e-12, EN)
    ek_per_comp = (1 - SSE / EN) * 100
    ek_all = (1 - SSE.sum() / EN.sum()) * 100
    return ek_per_comp, ek_all


def compute_det_R(enc, data_n, latent_dim, device):
    """det of latent correlation matrix (0–1; 1 = fully disentangled)."""
    enc.eval()
    Nt = data_n.shape[0]
    Z = np.zeros((Nt, latent_dim), dtype=np.float32)
    with torch.no_grad():
        for t in range(Nt):
            x = torch.tensor(data_n[t:t+1], device=device)
            mu, _ = enc(x)
            Z[t] = mu.cpu().numpy()[0]
    Cov = np.cov(Z.T)
    d = np.sqrt(np.diag(Cov))
    d[d < 1e-12] = 1.0
    R = Cov / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return float(np.linalg.det(R))


# ─────────────────────────── Training ───────────────────────

def vae_loss(enc, dec, x, beta, bsp=None, mu_bsp=0.0):
    mu, logv = enc(x)
    logv = torch.clamp(logv, -10, 10)
    z = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
    xhat = dec(z)
    recon = F.mse_loss(xhat, x)
    kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
    bsp_val = bsp(xhat, x) if (bsp is not None) else torch.zeros((), device=x.device)
    total = recon + beta * kl + mu_bsp * bsp_val
    return total, recon, kl, bsp_val


def val_loss(enc, dec, loader, beta, device, bsp=None, mu_bsp=0.0):
    enc.eval(); dec.eval()
    tot_l = tot_r = tot_k = tot_b = 0.0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            mu, logv = enc(xb)
            logv = torch.clamp(logv, -10, 10)
            xhat = dec(mu)
            r = F.mse_loss(xhat, xb)
            k = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
            b = bsp(xhat, xb) if (bsp is not None) else torch.zeros((), device=device)
            tot_l += (r + beta * k + mu_bsp * b).item()
            tot_r += r.item(); tot_k += k.item(); tot_b += b.item()
    n = len(loader)
    return tot_l / n, tot_r / n, tot_k / n, tot_b / n


# ─────────────────────────── Main ───────────────────────────

def main():
    global DATA_FILE

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

    enc = Encoder(H, W, C, LATENT_DIM).to(DEVICE)
    dec = Decoder(H, W, C, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    bsp = None
    mu_bsp = 0.0
    if USE_BSP:
        bsp = BSPLoss(H, W, n_bins=N_BINS, eps=BSP_EPS, lambda_mode=LAMBDA_MODE).to(DEVICE)
        mu_bsp = MU

    val_every = VAL_EVERY or max(1, round(N_EPOCHS / 10))

    hist = {k: [] for k in ["tr_loss","tr_recon","tr_kl","tr_bsp",
                             "va_loss","va_recon","va_kl","va_bsp",
                             "va_ek_all","va_det_R","va_epoch"]}

    best_val = float("inf")
    best_state = None

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bsp_tag = f"_bsp{MU:.0e}_nb{N_BINS}_{LAMBDA_MODE}" if USE_BSP else "_nobsp"
    save_path = os.path.join(_MODELS_DIR,
        f"model_betaVAE_BSP_{base}_lat{LATENT_DIM}_b{BETA:.0e}{bsp_tag}_ep{N_EPOCHS}.pt")

    print(f"\n{'='*60}")
    print(f"  beta-VAE + BSP  |  {base}  |  {H}x{W}  C={C}  Nt={Nt}")
    print(f"  latent={LATENT_DIM}  beta={BETA}  lr={LR}  epochs={N_EPOCHS}")
    if USE_BSP:
        print(f"  BSP: mu={MU}  n_bins={N_BINS}  eps={BSP_EPS}  lambda={LAMBDA_MODE}")
    else:
        print(f"  BSP: DISABLED")
    print(f"  device={DEVICE}  train={len(tr_idx)}  val={len(val_idx)}")
    print(f"{'='*60}\n")

    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        s_l = s_r = s_k = s_b = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, r, k, b = vae_loss(enc, dec, xb, BETA, bsp, mu_bsp)
            opt.zero_grad(); loss.backward(); opt.step()
            s_l += loss.item(); s_r += r.item(); s_k += k.item(); s_b += b.item()
        nb = len(tr_loader)
        hist["tr_loss"].append(s_l/nb)
        hist["tr_recon"].append(s_r/nb)
        hist["tr_kl"].append(s_k/nb)
        hist["tr_bsp"].append(s_b/nb)

        do_val = (epoch % val_every == 0) or (epoch == N_EPOCHS)
        if do_val:
            vl, vr, vk, vb = val_loss(enc, dec, va_loader, BETA, DEVICE, bsp, mu_bsp)
            va_fluc = data_fluc[val_idx]
            va_data_n = data_n[val_idx]
            _, ek_all = compute_ek(enc, dec, va_data_n, va_fluc, mu_C, std_C, fluc_mean, DEVICE)
            det_R = compute_det_R(enc, va_data_n, LATENT_DIM, DEVICE)

            hist["va_loss"].append(vl); hist["va_recon"].append(vr)
            hist["va_kl"].append(vk);   hist["va_bsp"].append(vb)
            hist["va_ek_all"].append(ek_all)
            hist["va_det_R"].append(det_R); hist["va_epoch"].append(epoch)

            marker = ""
            if SAVE_MODEL and vl < best_val:
                best_val = vl
                best_state = {
                    "enc": {k: v.cpu().clone() for k, v in enc.state_dict().items()},
                    "dec": {k: v.cpu().clone() for k, v in dec.state_dict().items()}}
                marker = "  ★ best"

            print(f"Ep {epoch:4d}/{N_EPOCHS} | "
                  f"tr {s_l/nb:.5f} | va {vl:.5f} | BSP {vb:.4f} | "
                  f"Ek {ek_all:.1f}% | detR {det_R:.4f}{marker}")
        else:
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l/nb:.5f} | BSP {s_b/nb:.4f}")

    # ── Save best model ──
    if SAVE_MODEL and best_state is not None:
        torch.save({
            "enc_state": best_state["enc"],
            "dec_state": best_state["dec"],
            "mu_C": mu_C, "std_C": std_C,
            "fluc_mean": fluc_mean,
            "H": H, "W": W, "C": C,
            "latent_dim": LATENT_DIM, "beta": BETA,
            "use_bsp": USE_BSP, "mu_bsp": MU, "n_bins": N_BINS,
            "bsp_eps": BSP_EPS, "lambda_mode": LAMBDA_MODE,
            "comp_names": comp_names,
            "tr_idx": tr_idx, "val_idx": val_idx,
            "data_file": DATA_FILE,
            "history": hist,
        }, save_path)
        print(f"\nSaved → {save_path}")

    # ── Learning curves ──
    ep = np.arange(1, N_EPOCHS + 1)
    va_ep = np.array(hist["va_epoch"])

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
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

    axes[2].plot(ep, hist["tr_bsp"], label="train BSP")
    if hist["va_bsp"]:
        axes[2].plot(va_ep, hist["va_bsp"], "o-", label="val BSP")
    axes[2].set(xlabel="Epoch", ylabel="BSP loss", title="Binned Spectral Power loss")
    axes[2].legend(); axes[2].grid(True)

    if hist["va_ek_all"]:
        axes[3].plot(va_ep, hist["va_ek_all"], "o-", color="green")
    axes[3].set(xlabel="Epoch", ylabel="Ek (%)", title="Validation Ek (all comps)")
    axes[3].grid(True)

    if hist["va_det_R"]:
        axes[4].plot(va_ep, hist["va_det_R"], "o-", color="purple")
    axes[4].set(xlabel="Epoch", ylabel="det(R)  [0–1]", title="Latent disentanglement")
    axes[4].grid(True)

    plt.suptitle(f"beta-VAE + BSP  |  {base}  |  latent={LATENT_DIM}  beta={BETA}  mu={MU if USE_BSP else 0}")
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
    plt.suptitle(f"Val snapshot t={t_show}  (beta-VAE + BSP)")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()

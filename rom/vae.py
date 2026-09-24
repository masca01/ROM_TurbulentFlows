"""
beta-VAE shared by every study: network, loss, metrics and the training loop.

Encoder / Decoder / vae_loss / val_loss / compute_ek / compute_det_R   (from beta_vae.py)
train()   the training recipe of every study since the summer (from re100_fraction_sweep.py):
          normalisation + mean field from the REAL training snapshots only, synthetic
          snapshots appended, best-validation checkpoint, Ek and det(R) on the real
          validation set.  latent and epochs are arguments (they used to be set by
          assigning re100_fraction_sweep.LATENT / .EPOCHS before the call).
"""

import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# training recipe (re100_fraction_sweep.py; the same values in every study)
BETA       = 5e-3
BATCH      = 32
EPOCHS     = 500
LR         = 3e-4
TORCH_SEED = 7
PROG_EVERY = 50
SEC_PER_SAMPLE = 1.3             # measured: 500 epochs cost ~1.3 s per training snapshot (time estimates only)


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

def vae_loss(enc, dec, x, beta):
    mu, logv = enc(x)
    logv = torch.clamp(logv, -10, 10)
    z = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
    xhat = dec(z)
    recon = F.mse_loss(xhat, x)
    kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
    return recon + beta * kl, recon, kl


def val_loss(enc, dec, loader, beta, device):
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



# ─────────────────────────── Shared training loop ───────────────────────

def train(train_real, train_aug_n, val_real, tag, latent, epochs=EPOCHS, beta=BETA, batch=BATCH,
          lr=LR, torch_seed=TORCH_SEED, prog_every=PROG_EVERY, device=None):
    """Summer recipe: normalisation + mean field from the real snapshots only,
    best-validation checkpoint, Ek and det(R) on the real validation set.
    Returns (best Ek, its det(R), best validation loss, wall seconds)."""
    from torch.utils.data import DataLoader, TensorDataset
    # local names = the old re100_fraction_sweep globals, so the body below is the verbatim summer code
    LATENT, EPOCHS, BETA, BATCH, LR, TORCH_SEED, PROG_EVERY = latent, epochs, beta, batch, lr, torch_seed, prog_every
    DEVICE = globals()["DEVICE"] if device is None else device
    t0 = time.time()
    n_real, n_aug = len(train_real), len(train_aug_n)
    C, H, W = train_real.shape[1:]
    mu_C  = train_real.mean(axis=(0, 2, 3)).astype(np.float32)
    std_C = train_real.std(axis=(0, 2, 3)); std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    phys_mean = train_real.mean(axis=0).astype(np.float32)
    # one preallocated, in-place normalised array (no concatenate copies)
    X = np.empty((n_real + n_aug, C, H, W), dtype=np.float32)
    X[:n_real] = train_real
    if n_aug: X[n_real:] = train_aug_n
    X -= mu_C[None, :, None, None]; X /= std_C[None, :, None, None]
    val_n = ((val_real - mu_C[None, :, None, None]) / std_C[None, :, None, None]).astype(np.float32)
    val_fluc = val_real - phys_mean[None]
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(val_n)), batch_size=BATCH, shuffle=False)

    torch.manual_seed(TORCH_SEED)
    enc = Encoder(H, W, C, LATENT).to(DEVICE); dec = Decoder(H, W, C, LATENT).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    val_every = max(1, round(EPOCHS / 10))
    best_val, best_ek, best_detR = float("inf"), float("nan"), float("nan")
    for epoch in range(1, EPOCHS + 1):
        enc.train(); dec.train(); run = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, _, _ = vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item()
        run /= max(1, len(tr_loader))
        if epoch % val_every == 0 or epoch == EPOCHS:
            vl, _, _ = val_loss(enc, dec, va_loader, BETA, DEVICE)
            if vl < best_val:
                _, ek = compute_ek(enc, dec, val_n, val_fluc, mu_C, std_C, phys_mean, DEVICE)
                detR = compute_det_R(enc, val_n, LATENT, DEVICE)
                best_val, best_ek, best_detR = vl, ek, detR
        if epoch % PROG_EVERY == 0 or epoch == EPOCHS:
            print(f"      [{tag}] epoch {epoch:4d}/{EPOCHS}  train_loss={run:.4e}  "
                  f"best_Ek={best_ek:5.2f}%  detR={best_detR:.3f}  [{time.time()-t0:6.0f}s]", flush=True)
    del X, tr_loader
    return best_ek, best_detR, best_val, time.time() - t0

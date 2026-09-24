"""
aug_experiment_strategyC.py — Low-data augmentation study (POD strategy C)
==========================================================================

Question answered
-----------------
"If I only have a small percentage of the real snapshots, does augmenting them
 with POD strategy-C synthetic snapshots train a BETTER beta-VAE than using the
 scarce real data alone?"

What this single script does
----------------------------
1. Loads a dataset (e.g. the cylinder wake, ~500 snapshots).
2. One seeded shuffle splits the full dataset into:
       - val_real   : a FIXED real validation set (VAL_FRAC of all snapshots),
                      held out FIRST, never trained on, never augmented;
       - train_real : the SCARCE real training pool (DATA_FRAC of all snapshots,
                      e.g. 0.10 -> 10%);
       - the remaining snapshots are left UNUSED (we pretend we never had them).
3. Fits POD on train_real only and generates synthetic snapshots with
   STRATEGY C  (bootstrap a real coefficient vector + small joint noise):
       a_new = a(t_base) + eps ,   eps_i ~ N(0, (NOISE_LEVEL * std_i)^2)
   These preserve the joint/phase structure of the modes (good for the cylinder).
4. Trains TWO identical beta-VAEs:
       BASELINE  : train_real only
       AUGMENTED : train_real + train_aug
   Both use the SAME normalization (from train_real only) and are validated on
   the SAME real validation set -> a fair head-to-head comparison.
5. Saves comparison FIGURES (PNG) and appends a CSV row per run.

Outputs (in ../bestModels/aug_lowdata/<base>_f<DATA_FRAC>/):
    curves_*.png   learning curves (val loss + val Ek), baseline vs augmented
    fields_*.png   TRUE | RECON | ERROR for a real val snapshot, both models
    bars_*.png     best Ek and det(R) bar chart
    + one row per run in CODE/aug_lowdata_results.csv
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")              # save figures to files (no display needed)
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta_vae as bv
from beta_vae import load_data

# ── Folder layout ──
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_OUT_ROOT   = os.path.normpath(os.path.join(_HERE, "..", "bestModels", "aug_lowdata"))
_RESULTS    = os.path.join(_HERE, "aug_lowdata_isotropic_results.csv")
os.makedirs(_OUT_ROOT, exist_ok=True)

# ══ CONFIG ══
DATA_FILE   = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y1.5708_256x256_Nt5024_UVW.mat"
COMP_IDX    = None       # None = auto-detect components

# how much REAL data we pretend to have
DATA_FRAC   = 0.02       # fraction of ALL snapshots used as the real TRAINING pool
VAL_FRAC    = 0.20       # fraction of ALL snapshots held out as REAL validation

# strategy-C augmentation
NOISE_LEVEL = 0.10       # jitter as a fraction of each POD mode's std
RANK        = None       # POD modes to keep (None = all training modes)
AUG_MULT    = 5.0        # #synthetic = AUG_MULT * #real-train   (used if N_AUG None)
N_AUG       = None       # explicit #synthetic snapshots (overrides AUG_MULT)

# training hyperparameters (identical for both networks)
LATENT_DIM  = 4
BETA        = 5e-3
BATCH_SIZE  = 32
N_EPOCHS    = 500
LR          = 3e-4
RNG_SEED    = 7
VAL_EVERY   = None       # None = auto (every ~10% of epochs)
T_SHOW      = 0          # which real val snapshot to draw in the field figure
SAVE_MODELS = False      # also save the two .pt checkpoints
# ═══════════════════════════════════════════════════════════

DEVICE = bv.DEVICE


# ─────────────────────── POD (strategy C) ───────────────────────

def compute_pod(train, rank=None):
    """train [Ntr,C,H,W] -> (x_mean[D], U[D,r], S[r], A[Ntr,r], shape)."""
    Ntr, C, H, W = train.shape
    X = train.reshape(Ntr, C * H * W).T
    x_mean = X.mean(axis=1)
    Xc = X - x_mean[:, None]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if rank is not None:
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    A = Vt.T * S[None, :]
    return x_mean, U, S, A, (C, H, W)


def reconstruct(x_mean, U, A_new, shape):
    C, H, W = shape
    X_new = x_mean[:, None] + U @ A_new.T
    return X_new.T.reshape(-1, C, H, W).astype(np.float32)


def make_strategyC(train_real, n_aug, noise_level, rank, rng):
    x_mean, U, S, A, shape = compute_pod(train_real, rank)
    r = A.shape[1]
    a_std  = A.std(axis=0)
    base_t = rng.integers(0, len(train_real), size=n_aug)
    noise  = rng.normal(0.0, 1.0, size=(n_aug, r)) * (noise_level * a_std[None, :])
    A_new  = A[base_t] + noise
    return reconstruct(x_mean, U, A_new, shape), r


# ─────────────────────── Training (one network) ───────────────────────

def train_one(tag, train_data, val_n, val_fluc, mu_C, std_C, phys_mean, shape):
    """Train one beta-VAE; return (history, best_metrics, enc, dec)."""
    C, H, W = shape
    train_n = (train_data - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    X_tr = torch.tensor(train_n, dtype=torch.float32)
    X_va = torch.tensor(val_n,   dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(RNG_SEED)        # same init for both runs -> fair
    enc = bv.Encoder(H, W, C, LATENT_DIM).to(DEVICE)
    dec = bv.Decoder(H, W, C, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every = VAL_EVERY or max(1, round(N_EPOCHS / 10))
    hist = {k: [] for k in ["tr_loss", "va_loss", "va_ek_all", "va_det_R", "va_epoch"]}
    best = {"val": float("inf"), "ek": float("nan"), "detR": float("nan"),
            "enc": None, "dec": None}

    print(f"\n--- training [{tag}]  |  train={len(train_data)}  val={len(val_n)} ---")
    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        s_l = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, _, _ = bv.vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            s_l += loss.item()
        s_l /= len(tr_loader)
        hist["tr_loss"].append(s_l)

        if (epoch % val_every == 0) or (epoch == N_EPOCHS):
            vl, _, _ = bv.val_loss(enc, dec, va_loader, BETA, DEVICE)
            _, ek_all = bv.compute_ek(enc, dec, val_n, val_fluc,
                                      mu_C, std_C, phys_mean, DEVICE)
            detR = bv.compute_det_R(enc, val_n, LATENT_DIM, DEVICE)
            hist["va_loss"].append(vl); hist["va_ek_all"].append(ek_all)
            hist["va_det_R"].append(detR); hist["va_epoch"].append(epoch)
            marker = ""
            if vl < best["val"]:
                best.update(val=vl, ek=ek_all, detR=detR,
                            enc={k: v.cpu().clone() for k, v in enc.state_dict().items()},
                            dec={k: v.cpu().clone() for k, v in dec.state_dict().items()})
                marker = "  ★ best"
            print(f"  Ep {epoch:4d}/{N_EPOCHS} | tr {s_l:.5f} | va {vl:.5f} | "
                  f"Ek {ek_all:.2f}% | detR {detR:.4f}{marker}")

    enc.load_state_dict(best["enc"]); dec.load_state_dict(best["dec"])
    enc.to(DEVICE).eval(); dec.to(DEVICE).eval()
    return hist, best, enc, dec


def recon_field(enc, dec, val_n, t, mu_C, std_C):
    with torch.no_grad():
        x_in = torch.tensor(val_n[t:t+1], device=DEVICE)
        mu_z, _ = enc(x_in)
        xhat = dec(mu_z).cpu().numpy()[0]
    return xhat * std_C[:, None, None] + mu_C[:, None, None]


# ─────────────────────── Main ───────────────────────

def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape
    base = os.path.splitext(os.path.basename(DATA_FILE))[0]

    # ── one seeded shuffle: val_real (fixed) | train_real (scarce) | unused ──
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(Nt)
    n_val = max(1, round(VAL_FRAC * Nt))
    n_tr  = max(1, round(DATA_FRAC * Nt))
    if n_val + n_tr > Nt:
        raise ValueError(f"VAL_FRAC+DATA_FRAC too large: need {n_val+n_tr} > {Nt} snapshots")
    val_idx = np.sort(idx[:n_val])
    tr_idx  = np.sort(idx[n_val:n_val + n_tr])
    train_real = data[tr_idx]
    val_real   = data[val_idx]

    # ── normalization & mean field from the SCARCE REAL training pool only ──
    mu_C  = train_real.mean(axis=(0, 2, 3)).astype(np.float32)
    std_C = train_real.std(axis=(0, 2, 3))
    std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    phys_mean = train_real.mean(axis=0).astype(np.float32)
    val_n    = (val_real - mu_C[None, :, None, None]) / std_C[None, :, None, None]
    val_fluc = val_real - phys_mean[None]

    # ── strategy-C synthetic snapshots from train_real ──
    n_aug = int(N_AUG) if N_AUG is not None else int(round(AUG_MULT * len(train_real)))
    train_aug, n_modes = make_strategyC(train_real, n_aug, NOISE_LEVEL, RANK, rng)

    print(f"\n{'='*64}")
    print(f"  LOW-DATA AUG STUDY (strategy C)  |  {base}  |  {H}x{W} C={C}")
    print(f"  total snapshots = {Nt}")
    print(f"  real train = {len(train_real)} ({DATA_FRAC:.0%})   "
          f"real val = {len(val_real)} ({VAL_FRAC:.0%})   unused = {Nt-n_val-n_tr}")
    print(f"  synthetic  = {n_aug}   POD modes = {n_modes}   noise = {NOISE_LEVEL}")
    print(f"  latent = {LATENT_DIM}  beta = {BETA}  epochs = {N_EPOCHS}  device = {DEVICE}")
    print(f"{'='*64}")

    # ── train the two networks ──
    h_base, b_base, enc_b, dec_b = train_one(
        "BASELINE (real only)", train_real, val_n, val_fluc, mu_C, std_C, phys_mean, (C, H, W))
    train_aug_all = np.concatenate([train_real, train_aug], axis=0)
    h_aug, b_aug, enc_a, dec_a = train_one(
        "AUGMENTED (real+synth)", train_aug_all, val_n, val_fluc, mu_C, std_C, phys_mean, (C, H, W))

    # ── output folder ──
    out_dir = os.path.join(_OUT_ROOT, f"{base}_f{DATA_FRAC:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{base}_f{DATA_FRAC:.2f}_lat{LATENT_DIM}_nl{NOISE_LEVEL}"

    # ── Figure 1: learning curves (val loss + val Ek) ──
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(h_base["va_epoch"], h_base["va_loss"], "o-", label="baseline")
    ax[0].plot(h_aug["va_epoch"],  h_aug["va_loss"],  "s-", label="augmented")
    ax[0].set(xlabel="Epoch", ylabel="Validation loss (real)", title="Validation loss")
    ax[0].legend(); ax[0].grid(True)
    ax[1].plot(h_base["va_epoch"], h_base["va_ek_all"], "o-", label="baseline")
    ax[1].plot(h_aug["va_epoch"],  h_aug["va_ek_all"],  "s-", label="augmented")
    ax[1].set(xlabel="Epoch", ylabel="Ek (%)", title="Validation Ek (real)")
    ax[1].legend(); ax[1].grid(True)
    plt.suptitle(f"{base}  |  {DATA_FRAC:.0%} real data  |  strategy C (noise={NOISE_LEVEL})")
    plt.tight_layout()
    f1 = os.path.join(out_dir, f"curves_{suffix}.png")
    plt.savefig(f1, dpi=130); plt.close(fig)

    # ── Figure 2: TRUE | RECON | ERROR for a real val snapshot, both models ──
    t = int(np.clip(T_SHOW, 0, len(val_real) - 1))
    true = val_real[t]
    rec_b = recon_field(enc_b, dec_b, val_n, t, mu_C, std_C)
    rec_a = recon_field(enc_a, dec_a, val_n, t, mu_C, std_C)
    cols = [("TRUE", true, None), ("RECON base", rec_b, None), ("ERR base", rec_b - true, "err"),
            ("RECON aug", rec_a, None), ("ERR aug", rec_a - true, "err")]
    fig, axes = plt.subplots(C, 5, figsize=(20, 4 * C), squeeze=False)
    for c in range(C):
        mx = max(abs(float(true[c].min())), abs(float(true[c].max())), 1e-8)
        for j, (title, fld, kind) in enumerate(cols):
            if kind == "err":
                e = fld[c]; me = max(abs(float(e.min())), abs(float(e.max())), 1e-8)
                im = axes[c, j].imshow(e, vmin=-me, vmax=me, cmap="RdBu_r", origin="lower")
            else:
                im = axes[c, j].imshow(fld[c], vmin=-mx, vmax=mx, cmap="RdBu_r", origin="lower")
            axes[c, j].set_title(f"{comp_names[c]} {title}"); axes[c, j].axis("off")
            plt.colorbar(im, ax=axes[c, j], fraction=0.046)
    plt.suptitle(f"Real val snapshot t={t}  |  {DATA_FRAC:.0%} data  |  "
                 f"Ek base {b_base['ek']:.1f}%  vs  aug {b_aug['ek']:.1f}%")
    plt.tight_layout()
    f2 = os.path.join(out_dir, f"fields_{suffix}.png")
    plt.savefig(f2, dpi=130); plt.close(fig)

    # ── Figure 3: best Ek + det(R) bars ──
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["baseline", "augmented"]
    ax[0].bar(labels, [b_base["ek"], b_aug["ek"]], color=["#4C72B0", "#DD8452"])
    ax[0].set(ylabel="best Ek (%)", title="Energy reconstructed (real val)")
    for i, v in enumerate([b_base["ek"], b_aug["ek"]]):
        ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax[1].bar(labels, [b_base["detR"], b_aug["detR"]], color=["#4C72B0", "#DD8452"])
    ax[1].set(ylabel="det(R)", title="Latent disentanglement")
    for i, v in enumerate([b_base["detR"], b_aug["detR"]]):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.suptitle(f"{base}  |  {DATA_FRAC:.0%} real data  |  strategy C")
    plt.tight_layout()
    f3 = os.path.join(out_dir, f"bars_{suffix}.png")
    plt.savefig(f3, dpi=130); plt.close(fig)

    # ── CSV: one row per run ──
    header = ("dataset,data_frac,run,n_real,n_aug,n_val,noise_level,latent,beta,"
              "best_val_loss,best_val_Ek,best_detR\n")
    write_header = not os.path.exists(_RESULTS)
    with open(_RESULTS, "a") as f:
        if write_header:
            f.write(header)
        for run, bb, na in [("baseline", b_base, 0), ("augmented", b_aug, n_aug)]:
            f.write(f"{base},{DATA_FRAC:.4f},{run},{len(train_real)},{na},{len(val_real)},"
                    f"{NOISE_LEVEL},{LATENT_DIM},{BETA},"
                    f"{bb['val']:.6f},{bb['ek']:.4f},{bb['detR']:.6f}\n")

    # ── optional: save the two checkpoints ──
    if SAVE_MODELS:
        for run, enc, dec, bb in [("baseline", enc_b, dec_b, b_base),
                                  ("augmented", enc_a, dec_a, b_aug)]:
            torch.save({
                "enc_state": {k: v.cpu() for k, v in enc.state_dict().items()},
                "dec_state": {k: v.cpu() for k, v in dec.state_dict().items()},
                "mu_C": mu_C, "std_C": std_C, "fluc_mean": phys_mean,
                "H": H, "W": W, "C": C, "latent_dim": LATENT_DIM, "beta": BETA,
                "comp_names": comp_names, "val_real": val_real,
                "run": run, "data_frac": DATA_FRAC, "noise_level": NOISE_LEVEL,
            }, os.path.join(out_dir, f"model_{run}_{suffix}.pt"))

    delta = b_aug["ek"] - b_base["ek"]
    verdict = "AUGMENTATION HELPS" if delta > 0 else "augmentation did NOT help"
    print(f"\n{'='*64}")
    print(f"  RESULT  |  {DATA_FRAC:.0%} real data ({len(train_real)} snaps) + {n_aug} synth")
    print(f"  baseline  Ek = {b_base['ek']:.2f}%   det(R) = {b_base['detR']:.4f}")
    print(f"  augmented Ek = {b_aug['ek']:.2f}%   det(R) = {b_aug['detR']:.4f}")
    print(f"  ΔEk = {delta:+.2f} %  ->  {verdict}")
    print(f"  figures → {out_dir}")
    print(f"  csv     → {_RESULTS}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()

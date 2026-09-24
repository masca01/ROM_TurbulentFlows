"""
POD data augmentation — JOINT density via Gaussian Mixture Model.
===========================================================================

Assessment config #3 from Data_Augmented_Autoencoders.pdf, §1.1 (joint
statistical sampling of POD coefficients).

Background
----------
POD coefficients are uncorrelated at second order  (<a_i a_j> = lambda_i d_ij)
but NOT statistically independent: for nonlinear flows they carry higher-order
dependence, phase relationships, and attractor constraints.  Independent
marginal sampling (strategy A / pod_augment_gaussian.py) ignores this and uses

        p(a_1,...,a_r) ~= prod_j p_j(a_j).

Here we instead estimate the JOINT density p(a) and sample a* ~ p_hat(a).  The
paper lists GMM / KDE / copula / normalizing-flow / small-VAE as options; we use
a Gaussian Mixture Model — the simplest defensible joint-density estimator.

Low-data design
---------------
Fitting a full joint density in ~r dimensions from few snapshots is ill-posed,
so (as the paper says: "joint distribution of the LEADING modal coefficients")
we split the modes:
    - leading N_JOINT modes  -> fit a joint GMM, sample together (keeps the
      phase coupling / attractor shape of the energetic structures);
    - remaining tail modes   -> sample independently from Gaussian marginals
      (these are low-energy and effectively decorrelated).

Screening (paper §1.1 / §1.4)
-----------------------------
Synthetic coefficient vectors are screened by total modal energy
E = 1/2 * sum_i a_i^2  (POD modes are orthonormal, so this is the fluctuation
kinetic energy).  Samples whose energy falls outside a tolerance band around the
observed range are rejected and resampled, reducing the chance of admissible-
but-unphysical fields.

Output: a bundle in ../DATA/AUGMENTED/<base>_aug_gmm.mat with train_real,
train_aug, val_real — drop-in for train_augmented_vae.py / test_augmented_vae.py.
"""

import os
import numpy as np
import scipy.io as sio
from sklearn.mixture import GaussianMixture

from beta_vae import load_data

# ── Folder layout ──
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
os.makedirs(_AUG_DIR, exist_ok=True)

# ══ CONFIG (keep SEED / VAL_FRAC / N_AUG identical across all generators) ══
DATA_FILE    = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/2PlatesGap/Data2PlatesGap1Re100.mat"
COMP_IDX     = None    # None = auto-detect components
RNG_SEED     = 7       # MUST match the other generators & the tester
VAL_FRAC     = 0.10    # fraction of REAL snapshots held out for validation
N_AUG        = None    # number of synthetic snapshots (None = same as #train)
RANK         = None    # POD modes to keep (None = all training modes)

# joint-density (GMM) settings
N_JOINT      = 8       # leading modes modelled jointly by the GMM
N_COMPONENTS = None    # GMM mixture components (None = auto from #train)
COV_TYPE     = "full"  # 'full' | 'diag' | 'tied' | 'spherical'
REG_COVAR    = 1e-6    # covariance regularization (stabilizes fit)

# physical screening of synthetic coefficient vectors
SCREEN       = True    # reject samples outside the observed energy band
ENERGY_TOL   = 0.25    # allowed fractional slack around observed [min,max] energy
MAX_RESAMPLE = 20      # give up screening after this many resample rounds

STRATEGY     = "gmm"
# ═══════════════════════════════════════════════════════════════════════════


def compute_pod(train, rank=None):
    """Method-of-snapshots POD — efficient when #grid-points >> #snapshots
    (e.g. the full 2-plates case), avoiding the huge [D x Ntr] SVD.

    train [Ntr,C,H,W] -> (x_mean[D], Xc[Ntr,D], Wp[Ntr,r], S[r], A[Ntr,r], shape)
    Spatial modes are U = Xc.T @ Wp (never formed, to save memory); reconstruct()
    uses (x_mean, Xc, Wp) directly.  Same POD as the SVD, just cheaper here.
    """
    Ntr, C, H, W = train.shape
    X  = train.reshape(Ntr, C * H * W)            # [Ntr, D] float32, rows=snapshots
    x_mean = X.mean(axis=0)                        # [D]
    Xc = (X - x_mean[None, :]).astype(np.float32)  # [Ntr, D] mean-subtracted
    G  = (Xc @ Xc.T).astype(np.float64)            # [Ntr, Ntr] temporal Gram (small)
    w, V = np.linalg.eigh(G)                        # ascending eigenpairs
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None); V = V[:, order]
    S = np.sqrt(w)                                  # singular values
    keep = S > (S[0] * 1e-10 if S.size else 0.0)   # drop null modes
    S, V = S[keep], V[:, keep]
    if rank is not None:
        S, V = S[:rank], V[:, :rank]
    A  = (V * S[None, :])                           # [Ntr, r] temporal coefficients
    Wp = (V / S[None, :]).astype(np.float32)        # [Ntr, r] projector for modes
    return x_mean.astype(np.float32), Xc, Wp, S, A, (C, H, W)


def reconstruct(x_mean, Xc, Wp, A_new, shape):
    """Rebuild snapshots from new coefficients without materializing the modes:
    X_new = x_mean + U @ A_new.T ,  U = Xc.T @ Wp."""
    C, H, W = shape
    M = Wp @ A_new.T.astype(np.float32)             # [Ntr, n_aug] (small)
    X_new = Xc.T @ M                                # [D, n_aug]
    X_new += x_mean[:, None]
    return X_new.T.reshape(-1, C, H, W).astype(np.float32)


def save_bundle(out_base, payload):
    """Save the augmentation bundle.  MATLAB v5 (.mat) can't store a single array
    larger than ~2 GB, so for big data (e.g. full 2-plates) we fall back to .npz,
    which handles arbitrary sizes and is read by train_augmented_vae.py.  Returns
    the path written."""
    MAT5_LIMIT = 2_000_000_000
    too_big = any(isinstance(v, np.ndarray) and v.nbytes >= MAT5_LIMIT
                  for v in payload.values())
    if too_big:
        path = out_base + ".npz"
        np.savez(path, **payload)                  # uncompressed: fast, any size
    else:
        path = out_base + ".mat"
        sio.savemat(path, payload, do_compression=True)
    return path


def _fit_gmm_robust(Z, n_components, cov_type, reg_covar, seed):
    """Fit a GMM on already-standardized data Z, backing off automatically if the
    covariance becomes singular (common for phase-locked / low-rank coefficients).
    Returns (gmm, n_components_used, cov_type_used)."""
    ncomp, cov, rc = int(n_components), cov_type, float(reg_covar)
    for _ in range(12):
        try:
            gmm = GaussianMixture(n_components=ncomp, covariance_type=cov,
                                  reg_covar=rc, random_state=seed)
            gmm.fit(Z)
            return gmm, ncomp, cov
        except (ValueError, np.linalg.LinAlgError):
            if rc < 1e-1:                 # 1) strengthen covariance regularization
                rc *= 10.0
            elif ncomp > 1:               # 2) fewer components
                ncomp -= 1; rc = float(reg_covar)
            else:                         # 3) fall back to diagonal covariance
                cov, rc = "diag", max(float(reg_covar), 1e-4)
    # last resort: a single diagonal Gaussian always fits
    gmm = GaussianMixture(n_components=1, covariance_type="diag",
                          reg_covar=1e-3, random_state=seed)
    gmm.fit(Z)
    return gmm, 1, "diag"


def sample_gmm_joint(A, n_aug, n_joint, n_components, cov_type, reg_covar,
                     screen, energy_tol, max_resample, rng):
    """Sample n_aug coefficient vectors: joint GMM on leading modes + Gaussian
    marginals on the tail.  Returns (A_new [n_aug,r], gmm, n_components, n_joint)."""
    A = np.asarray(A, dtype=np.float64)                # float64 -> stable GMM fit
    Ntr, r = A.shape
    n_joint = int(min(n_joint, r))
    if n_components is None:                            # modest for scarce data
        n_components = int(np.clip(Ntr // 8, 1, 5))
    n_components = int(min(max(1, n_components), Ntr))

    # standardize the joint block so reg_covar is meaningful and scales are uniform
    head = A[:, :n_joint]
    h_mu = head.mean(axis=0)
    h_sd = head.std(axis=0)
    h_sd = np.where(h_sd < 1e-12, 1.0, h_sd)
    Z = (head - h_mu) / h_sd

    seed = int(rng.integers(0, 2**31 - 1))
    gmm, n_components, cov_type = _fit_gmm_robust(Z, n_components, cov_type, reg_covar, seed)

    # tail (independent Gaussian marginals)
    a_mean = A.mean(axis=0)
    a_std  = A.std(axis=0)

    # observed total-energy band for screening
    E_obs = 0.5 * (A ** 2).sum(axis=1)
    lo = (1.0 - energy_tol) * E_obs.min()
    hi = (1.0 + energy_tol) * E_obs.max()

    def draw(n):
        """Draw n candidate coefficient vectors (unscreened)."""
        gmm.random_state = int(rng.integers(0, 2**31 - 1))
        z, _ = gmm.sample(n)                           # [n, n_joint] standardized
        z = z[rng.permutation(len(z))]                 # gmm.sample() groups by comp
        cand = np.empty((n, r), dtype=np.float64)
        cand[:, :n_joint] = z * h_sd[None, :] + h_mu[None, :]     # un-standardize
        if n_joint < r:
            cand[:, n_joint:] = rng.normal(a_mean[None, n_joint:], a_std[None, n_joint:],
                                           size=(n, r - n_joint))
        return cand

    out = np.empty((n_aug, r), dtype=np.float64)
    filled, rounds = 0, 0
    while filled < n_aug and rounds < max_resample:
        need = n_aug - filled
        cand = draw(need * 2 if screen else need)      # oversample when screening
        if screen:
            E = 0.5 * (cand ** 2).sum(axis=1)
            cand = cand[(E >= lo) & (E <= hi)]
        take = min(len(cand), need)
        out[filled:filled + take] = cand[:take]
        filled += take
        rounds += 1

    if filled < n_aug:                                 # screening too tight: top up
        out[filled:] = draw(n_aug - filled)
    return out, gmm, n_components, n_joint


def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(Nt)
    n_val   = max(1, round(VAL_FRAC * Nt))
    val_idx = np.sort(idx[:n_val])
    tr_idx  = np.sort(idx[n_val:])
    train_real = data[tr_idx].copy()
    val_real   = data[val_idx].copy()
    del data                                             # free the full dataset
    Ntr = len(tr_idx)
    n_aug = N_AUG or Ntr

    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real, RANK)
    r = A.shape[1]

    A_new, gmm, n_comp, n_joint = sample_gmm_joint(
        A, n_aug, N_JOINT, N_COMPONENTS, COV_TYPE, REG_COVAR,
        SCREEN, ENERGY_TOL, MAX_RESAMPLE, rng)
    train_aug = reconstruct(x_mean, Xc, Wp, A_new, shape)
    del Xc                                               # free mean-subtracted data

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bundle = {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object),
        "strategy": STRATEGY, "tr_idx": tr_idx, "val_idx": val_idx,
        "rng_seed": RNG_SEED, "val_frac": VAL_FRAC,
        "n_modes": r, "n_joint": n_joint, "n_components": n_comp,
        "cov_type": COV_TYPE, "screen": SCREEN, "energy_tol": ENERGY_TOL,
        "rank": (-1 if RANK is None else RANK),
    }
    out_file = save_bundle(os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}"), bundle)

    print(f"\n[{STRATEGY}]  {base}  |  {H}x{W}  C={C}")
    print(f"  real train = {Ntr}   real val = {len(val_idx)}   synthetic = {n_aug}")
    print(f"  POD modes  = {r}   joint GMM on leading {n_joint}   components = {n_comp}")
    print(f"  screening  = {SCREEN}  (energy tol = {ENERGY_TOL})")
    print(f"  saved → {out_file}\n")


if __name__ == "__main__":
    main()

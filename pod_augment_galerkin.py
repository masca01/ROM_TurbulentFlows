"""
POD data augmentation using an unclosed POD-Galerkin model that I identify
directly from the data (the "data-driven" version of the Galerkin augmentation).

This is configuration 4 of the assessment.  The idea is to reduce each snapshot
to a few POD coefficients a_i(t), find a low-order model for how they evolve,
and integrate that model to generate new, temporally-coherent snapshots.

The companion script pod_augment_galerkin_ns.py builds the model by projecting
the Navier-Stokes equations onto the modes.  Here I do NOT use the equations:
I fit the model straight to the coefficient time series a(t_n) that I already
have from the POD.  This is the SINDy-style system identification that the
assessment document lists as a valid alternative to the analytic projection.
The model has the usual quadratic Galerkin form

        da_i/dt = c_i + sum_j L_ij a_j + sum_{j<=k} Q_ijk a_j a_k .

The nice property is that this is LINEAR in the unknowns c, L, Q: all the
nonlinearity sits in the known regressors (1, a_j, a_j a_k).  So identifying the
model for each mode i is just an ordinary least-squares fit
da_i = Theta beta_i, where each row of Theta is [1, a_1..a_r, a_1^2, a_1 a_2, ...]
evaluated at one snapshot.

How the script works:
  1. Split the snapshots into train/val with the same seed as the other
     generators, so val_real is identical across strategies, and compute the
     POD on the training snapshots only.
  2. Estimate da/dt with central finite differences,
     da_i(t_n) ~ [a_i(t_{n+1}) - a_i(t_{n-1})] / (2 dt), using only triples of
     snapshots that are consecutive in the ORIGINAL time order.  The random
     validation holdout leaves gaps in the series, and any triple straddling a
     gap is dropped, so no validation data ever leaks into the fit.
  3. Fit c, L, Q for the leading N_ROM modes by ridge least squares.  I
     standardize each coefficient (b = a/s) before fitting, only to keep the
     linear algebra well conditioned.
  4. Generate synthetic trajectories: start from a real coefficient vector plus
     a small perturbation (strategy C), integrate forward with fixed-step RK4,
     and sample at the snapshot spacing.  I keep the horizon SHORT because a
     truncated Galerkin model is unclosed and slowly drifts, so I only trust it
     for a little while and then restart from a fresh real state.
  5. Screen each trajectory by its total modal energy E = 1/2 sum a_i^2 and cut
     it as soon as it leaves the energy band seen in the real data.  The
     fraction of steps rejected is a direct measure of how much the model
     drifts, i.e. how much a closure/calibration (configuration 5) is needed.
  6. Rebuild the snapshots as mean flow + sum of the kept modes.

About coarse data (MODEL = "map"):
Central differences need enough snapshots per oscillation period (roughly 8+).
The 2-plates data has ~16 and is fine, but the cylinder wake has only ~4, so its
higher harmonics sit at/beyond the Nyquist rate, the estimated derivatives are
aliased, and the continuous-time model blows up almost immediately.  For that
case I instead identify the DISCRETE one-step map
        a_i(n+1) = c_i + sum_j L_ij a_j(n) + sum_{j<=k} Q_ijk a_j(n) a_k(n),
which needs no derivatives (only consecutive PAIRS) and is exact at any sampling
rate; I then generate data by iterating the map instead of RK4.  MODEL="auto"
chooses between the two by measuring the snapshots per period, and if generation
still struggles the code backs off to fewer modes.

Output: ../DATA/AUGMENTED/<base>_aug_galerkin.{mat|npz}, a bundle with
train_real / train_aug / val_real that plugs straight into
train_augmented_vae.py.
"""

import os
import numpy as np
import scipy.io as sio

# ------------ Folder layout ------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
os.makedirs(_AUG_DIR, exist_ok=True)

# ============ CONFIG (keep SEED / VAL_FRAC / N_AUG the same across generators) ============
DATA_FILE   = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/2PlatesGap/Data2PlatesGap1Re50.mat"
COMP_IDX    = None     # None = auto-detect the velocity components in the file
RNG_SEED    = 7        # same seed as the other generators -> identical val_real
VAL_FRAC    = 0.10     # fraction of the real snapshots kept aside for validation
N_AUG       = None     # how many synthetic snapshots (None = as many as #train)

# model identification
MODEL       = "auto"   # "ode"  = fit da/dt with central differences + RK4
                       #          (needs ~8+ snapshots/period; 2-plates ~16)
                       # "map"  = fit the one-step map a(n)->a(n+1), no
                       #          derivatives (coarse data, e.g. cylinder ~4)
                       # "auto" = decide from the measured snapshots-per-period
N_ROM       = 20       # POD modes kept in the dynamical model.
                       # For 2-plates: 15 -> ~55% energy and very stable;
                       # 20 -> ~63% energy with mild screened drift; 25+ starts
                       # to overfit the dynamics.  For the cylinder just 6 modes
                       # already hold ~99.9% of the energy.  Backs off
                       # automatically if generation fails.
DT          = 1.0      # snapshot spacing.  It is not given for this data, so I
                       # measure time in snapshot intervals (=1).  This only
                       # rescales c, L, Q by a constant and does not change the
                       # generated snapshots.
RIDGE_ALPHA = 1e-6     # tiny ridge penalty, just for numerical conditioning

# synthetic-trajectory generation
HORIZON     = 181       # snapshot intervals per trajectory (short on purpose)
SUBSTEPS    = 181       # RK4 sub-steps inside each snapshot interval
IC_NOISE    = 0     # initial perturbation, as a fraction of each mode's std
ENERGY_TOL  = 0.25     # allowed slack around the real [min, max] energy band.
                       # Keep it tight: this is what pins the synthetic energy to
                       # the real one (0.25 let the cylinder wander +-25%).
MAX_TRAJ    = 5     # hard cap on trajectory launches (safety)
MIN_KEEP    = 0.20     # if fewer than this fraction of steps survive, the
                       # trajectories are basically ~1 step long -> back off N_ROM

STRATEGY    = "galerkin_5traj"
# ========================================================================


# ------------------------- Loading and POD -------------------------

def _read_field(S, key, is_hdf5):
    """Read one array out of a .mat file.  h5py (v7.3 files) stores MATLAB
    arrays with the dimensions reversed, so in that case I transpose back."""
    arr = np.array(S[key], dtype=np.float32)
    if is_hdf5:
        arr = arr.T
    return arr


def load_data(path, comp_idx=None):
    """Load a .mat dataset and return it as data[Nt, C, H, W] (float32) plus the
    names of the components.  Handles both old (v5) and new (v7.3/HDF5) .mat
    files, and the couple of different variable layouts I have across datasets."""
    import h5py

    # scipy reads v5 files; for v7.3 it raises and I fall back to h5py
    fh = None
    try:
        S = sio.loadmat(path, simplify_cells=True)
        is_hdf5 = False
    except NotImplementedError:
        fh = h5py.File(path, "r")
        S = fh
        is_hdf5 = True

    try:
        if "Tensor" in S:                                # cylinder-wake format
            T = _read_field(S, "Tensor", is_hdf5)        # [C, N1, N2, Nt]
            C, N1, N2, Nt = T.shape
            data = T.transpose(3, 0, 1, 2)               # -> [Nt, C, N1, N2]
            names = {2: ["u", "v"], 3: ["u", "v", "w"]}.get(C, [f"c{i}" for i in range(C)])

        elif "U" in S:                                   # channel-type format
            V = _read_field(S, "U", is_hdf5)             # [Nt, Nz, Nx, C_all]
            Nt, Nz, Nx, C_all = V.shape
            if comp_idx is None:
                comp_idx = [0, 2] if C_all == 3 else list(range(C_all))
            V = V[:, :, :, comp_idx]
            data = V.transpose(0, 3, 1, 2)               # -> [Nt, C, Nz, Nx]
            all_names = ["u", "v", "w"]
            names = [all_names[i] for i in comp_idx]

        elif "UW" in S:
            V = _read_field(S, "UW", is_hdf5)            # [Nt, Nz, Nx, 2]
            data = V.transpose(0, 3, 1, 2)               # -> [Nt, 2, Nz, Nx]
            names = ["u", "w"]

        elif "DataU" in S and "DataV" in S:              # 2-plates-gap DNS
            # Each row of DataU/DataV is one snapshot with the 2D field
            # flattened (Fortran order, x fastest).  The grid resolution is not
            # the same in every dataset (Re=100 is 1199x349, Re=50 is 599x349),
            # and the coordinate arrays are called DataX/DataY in one file and
            # X/Y in the other, so I infer nx, ny from whichever is present
            # instead of hardcoding them.
            u = np.asarray(S["DataU"], dtype=np.float32)
            v = np.asarray(S["DataV"], dtype=np.float32)
            Nt, Nspace = u.shape
            xkey = "DataX" if "DataX" in S else "X"
            NX = int(np.unique(np.asarray(S[xkey]).ravel()).size)   # points in x
            NY = Nspace // NX                                        # points in y
            if NX * NY != Nspace:
                raise ValueError(f"2PlatesGap: inferred nx*ny {NX*NY} != "
                                 f"Nspace {Nspace}")
            u = u.reshape(Nt, NY, NX)                    # [Nt, H=y, W=x]
            v = v.reshape(Nt, NY, NX)
            data = np.stack([u, v], axis=1)              # [Nt, 2, H, W]
            names = ["u", "v"]

        else:
            visible = [k for k in S.keys() if not k.startswith("#")]
            raise ValueError(f"Unknown format. Fields: {visible}")

    finally:
        if fh is not None:
            fh.close()

    print(f"Loaded  {os.path.basename(path)}  →  {data.shape}  comps={names}")
    return data, names


def compute_pod(train, rank=None):
    """POD by the method of snapshots.  When there are far more grid points than
    snapshots (the full 2-plates case), it is much cheaper to eigen-decompose
    the small [Ntr x Ntr] Gram matrix than to SVD the huge data matrix.

    train[Ntr, C, H, W] -> (x_mean, Xc, Wp, S, A, shape), where
      x_mean : temporal mean field (flattened)
      Xc     : mean-subtracted snapshots (flattened), kept so I can rebuild the
               spatial modes as U = Xc.T @ Wp without ever storing U
      Wp, S  : the pieces needed to go between coefficients and snapshots
      A      : the POD coefficients of the training snapshots
    """
    Ntr, C, H, W = train.shape
    X  = train.reshape(Ntr, C * H * W)
    x_mean = X.mean(axis=0)
    Xc = (X - x_mean[None, :]).astype(np.float32)
    G  = (Xc @ Xc.T).astype(np.float64)                  # Ntr x Ntr Gram matrix
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]                           # largest energy first
    w = np.clip(w[order], 0.0, None); V = V[:, order]
    S = np.sqrt(w)
    keep = S > (S[0] * 1e-10 if S.size else 0.0)          # drop numerical zeros
    S, V = S[keep], V[:, keep]
    if rank is not None:
        S, V = S[:rank], V[:, :rank]
    A  = (V * S[None, :])                                 # coefficients
    Wp = (V / S[None, :]).astype(np.float32)              # used to reconstruct
    return x_mean.astype(np.float32), Xc, Wp, S, A, (C, H, W)


def reconstruct(x_mean, Xc, Wp, A_new, shape):
    """Turn a set of POD coefficients back into full snapshots.  Uses
    X_new = x_mean + U @ A_new.T with U = Xc.T @ Wp, so the (large) modes U are
    never actually formed in memory."""
    C, H, W = shape
    M = Wp @ A_new.T.astype(np.float32)
    X_new = Xc.T @ M
    X_new += x_mean[:, None]
    return X_new.T.reshape(-1, C, H, W).astype(np.float32)


def save_bundle(out_base, payload):
    """Save the augmentation bundle.  A v5 .mat file cannot hold a single array
    larger than ~2 GB, so for the big datasets I switch to .npz (which
    train_augmented_vae.py reads just the same)."""
    MAT5_LIMIT = 2_000_000_000
    too_big = any(isinstance(v, np.ndarray) and v.nbytes >= MAT5_LIMIT
                  for v in payload.values())
    if too_big:
        path = out_base + ".npz"
        np.savez(path, **payload)
    else:
        path = out_base + ".mat"
        sio.savemat(path, payload, do_compression=True)
    return path


# ------------------------- Model identification -------------------------

def _theta(B):
    """Build the regressor matrix for the quadratic model from the coefficients
    B[N, r].  Each row becomes [1, b_1..b_r, and every product b_j*b_k with
    j<=k].  These are the known terms; fitting only has to find their weights."""
    N, r = B.shape
    cols = [np.ones((N, 1)), B]
    for j in range(r):
        cols.append(B[:, j:j + 1] * B[:, j:])            # b_j*b_j .. b_j*b_r
    return np.concatenate(cols, axis=1)                  # [N, 1 + r + r(r+1)/2]


def fit_galerkin(A_rom, tr_idx, dt, ridge_alpha):
    """Identify the continuous model db/dt = c + L b + Q(b,b) from the
    coefficient series.

    A_rom[Ntr, r] : leading-mode POD coefficients of the training snapshots
    tr_idx[Ntr]   : their original time indices, used to keep only the central-
                    difference triples that have no validation gap in the middle.

    I standardize per mode (b = a/s) for conditioning and return the fitted
    weights, the scales, the per-mode R^2, and how many samples were used.
    """
    A_rom = np.asarray(A_rom, dtype=np.float64)
    Ntr, r = A_rom.shape
    s = A_rom.std(axis=0)
    s = np.where(s < 1e-14, 1.0, s)
    B = A_rom / s[None, :]

    # keep row n only if n-1, n, n+1 are consecutive in the original time order
    ok = np.where((np.diff(tr_idx[:-1]) == 1) & (np.diff(tr_idx[1:]) == 1))[0] + 1
    if len(ok) < 10:
        raise RuntimeError(f"only {len(ok)} consecutive triples for finite "
                           "differences — check the split")
    dB = (B[ok + 1] - B[ok - 1]) / (2.0 * dt)            # central-difference da/dt

    Th = _theta(B[ok])                                   # regressors at those times
    n_th = Th.shape[1]
    # Ridge least squares written as an augmented system:
    # minimizing ||Th b - dB||^2 + alpha||b||^2 is the same as a plain least
    # squares on [Th; sqrt(alpha) I] against [dB; 0].
    Th_aug = np.vstack([Th, np.sqrt(ridge_alpha) * np.eye(n_th)])
    dB_aug = np.vstack([dB, np.zeros((n_th, r))])
    beta, *_ = np.linalg.lstsq(Th_aug, dB_aug, rcond=None)

    resid = dB - Th @ beta
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((dB - dB.mean(axis=0)) ** 2).sum(axis=0)
    R2 = 1.0 - ss_res / np.where(ss_tot < 1e-300, 1.0, ss_tot)
    return beta, s, R2, len(ok)


def fit_galerkin_map(A_rom, tr_idx, ridge_alpha):
    """Identify the DISCRETE one-step map b(n+1) = c + L b(n) + Q(b,b)(n).

    This needs no derivative estimate — only consecutive snapshot PAIRS — so it
    works on coarsely-sampled data where the central differences alias (the
    cylinder wake).  Same regressors and same linearity in c, L, Q as the
    continuous fit; the only change is that the target is the next snapshot
    instead of a time derivative."""
    A_rom = np.asarray(A_rom, dtype=np.float64)
    Ntr, r = A_rom.shape
    s = A_rom.std(axis=0)
    s = np.where(s < 1e-14, 1.0, s)
    B = A_rom / s[None, :]

    ok = np.where(np.diff(tr_idx) == 1)[0]               # consecutive pairs (n, n+1)
    if len(ok) < 10:
        raise RuntimeError(f"only {len(ok)} consecutive pairs for the map fit "
                           "— check the split")
    Th = _theta(B[ok])                                   # regressors at time n
    Y  = B[ok + 1]                                       # target = state at n+1
    n_th = Th.shape[1]
    Th_aug = np.vstack([Th, np.sqrt(ridge_alpha) * np.eye(n_th)])
    Y_aug  = np.vstack([Y, np.zeros((n_th, r))])
    beta, *_ = np.linalg.lstsq(Th_aug, Y_aug, rcond=None)

    resid = Y - Th @ beta
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    R2 = 1.0 - ss_res / np.where(ss_tot < 1e-300, 1.0, ss_tot)
    return beta, s, R2, len(ok)


def make_rhs(beta):
    """Return the right-hand side f(b) = c + L b + Q(b,b) of the continuous
    model for a single state vector b."""
    r = beta.shape[1]
    def f(b):
        th = _theta(b[None, :])[0]                       # regressors for this b
        return th @ beta                                 # weighted -> db/dt
    return f


def make_step(beta, model, dt, substeps):
    """Return a function that advances a state by exactly one snapshot interval.
    For the discrete map that is just one application of the map; for the
    continuous model it is a fixed-step RK4 split into `substeps` sub-steps."""
    if model == "map":
        def step(b):
            return _theta(b[None, :])[0] @ beta          # iterate the map once
        return step
    f = make_rhs(beta)
    h = dt / substeps
    def step(b):                                         # RK4 over one interval
        for _ in range(substeps):
            k1 = f(b)
            k2 = f(b + 0.5 * h * k1)
            k3 = f(b + 0.5 * h * k2)
            k4 = f(b + h * k3)
            b = b + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return b
    return step


def integrate_trajectories(step, s, B_real, n_aug, horizon,
                           ic_noise, energy_tol, max_traj, rng):
    """Create synthetic coefficient vectors by integrating many short
    trajectories and keeping only the states that stay physically reasonable.

    step            : advances a state by one snapshot interval (RK4 or map)
    s               : per-mode scale used to standardize the coefficients
    B_real[Ntr, r]  : standardized real coefficients; used both as the pool of
                      initial conditions and to define the allowed energy band
    Returns the accepted states (standardized) and a dict of statistics.
    """
    Ntr, r = B_real.shape

    # Allowed energy band, in the physical units of the kept modes.  A synthetic
    # state whose energy leaves this band is treated as drift and cut.
    E_obs = 0.5 * ((B_real * s[None, :]) ** 2).sum(axis=1)
    lo = (1.0 - energy_tol) * E_obs.min()
    hi = (1.0 + energy_tol) * E_obs.max()
    b_cap = 10.0 * np.abs(B_real).max()                  # obvious blow-up cutoff

    out = np.empty((n_aug, r), dtype=np.float64)
    filled = n_traj = n_reject = 0
    while filled < n_aug and n_traj < max_traj:
        n_traj += 1
        # start from a real snapshot plus a small random kick (strategy C)
        b = B_real[rng.integers(0, Ntr)] + ic_noise * rng.normal(size=r)
        for _ in range(horizon):
            with np.errstate(over="ignore", invalid="ignore"):
                b = step(b)                              # advance one interval
            if not np.all(np.isfinite(b)) or np.abs(b).max() > b_cap:
                n_reject += horizon                      # blew up: drop the rest
                break
            E = 0.5 * ((b * s) ** 2).sum()
            if E < lo or E > hi:                         # left the energy band
                n_reject += 1
                break
            out[filled] = b
            filled += 1
            if filled == n_aug:
                break

    if filled < n_aug:
        raise RuntimeError(
            f"only {filled}/{n_aug} synthetic snapshots survived screening after "
            f"{n_traj} trajectories — the unclosed model drifts too fast")
    stats = {"n_traj": n_traj, "n_reject": n_reject,
             "keep_frac": filled / max(filled + n_reject, 1),
             "E_band": (lo, hi)}
    return out, stats


# ------------------------------- Main -------------------------------

def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape

    # Same split as every other generator (same seed), so val_real matches.
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

    # POD on the training snapshots only.
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    r_full = A.shape[1]

    # Decide continuous-vs-discrete from the temporal resolution.  I estimate the
    # oscillation period of mode 1 from its sign changes (zero crossings); if
    # there are too few snapshots per period the derivatives would be aliased and
    # only the discrete map is trustworthy (this is the cylinder-wake case).
    spp = None
    zc = np.where(np.diff(np.sign(A[:, 0])) != 0)[0]
    if len(zc) > 2:
        spp = 2.0 * np.median(np.diff(zc))               # ~snapshots per period
        print(f"\n[{STRATEGY}]  mode-1 period ~ {spp:.0f} snapshots")
    model = MODEL
    if model == "auto":
        model = "ode" if (spp is None or spp >= 8) else "map"
        print(f"[{STRATEGY}]  MODEL=auto -> '{model}'"
              + ("  (too coarse for finite-difference ID)" if model == "map" else ""))

    # Identify the model and generate, backing off to fewer modes if the
    # trajectories cannot survive the energy screening.
    n_rom0 = int(min(N_ROM, r_full))
    ladder = [n_rom0] + [n for n in (15, 10, 6, 4, 2) if n < n_rom0]
    for n_rom in ladder:
        if model == "map":
            beta, s, R2, n_fit = fit_galerkin_map(A[:, :n_rom], tr_idx, RIDGE_ALPHA)
        else:
            beta, s, R2, n_fit = fit_galerkin(A[:, :n_rom], tr_idx, DT, RIDGE_ALPHA)
        e_rom = (S[:n_rom] ** 2).sum() / (S ** 2).sum()
        print(f"\n[{STRATEGY}]  {model}: {n_rom} modes ({100*e_rom:.2f}% energy), "
              f"{beta.shape[0]} regressors, {n_fit} fit samples")
        print("  fit R^2 per mode: " + "  ".join(f"{v:.3f}" for v in R2))

        # standardized real coefficients (same scaling used inside the fit)
        B_real = np.asarray(A[:, :n_rom], dtype=np.float64) / s[None, :]
        step = make_step(beta, model, DT, SUBSTEPS)
        try:
            B_new, st = integrate_trajectories(step, s, B_real, n_aug, HORIZON,
                                               IC_NOISE, ENERGY_TOL, MAX_TRAJ, rng)
            # if almost nothing survives, this N_ROM is too drifty -> shrink it
            if st["keep_frac"] < MIN_KEEP and n_rom != ladder[-1]:
                print(f"  !! filled, but only {100*st['keep_frac']:.1f}% of steps "
                      f"survived screening (< {100*MIN_KEEP:.0f}%)"
                      "\n  -> backing off to a smaller model")
                continue
            break
        except RuntimeError as err:
            print(f"  !! {err}")
            if n_rom == ladder[-1]:
                raise
            print("  -> backing off to a smaller model")

    A_new = B_new * s[None, :]                           # back to physical units
    train_aug = reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new, shape)
    del Xc

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bundle = {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object),
        "strategy": STRATEGY, "tr_idx": tr_idx, "val_idx": val_idx,
        "rng_seed": RNG_SEED, "val_frac": VAL_FRAC, "model": model,
        "n_modes": r_full, "n_rom": n_rom, "dt": DT, "horizon": HORIZON,
        "substeps": SUBSTEPS, "ic_noise": IC_NOISE, "ridge_alpha": RIDGE_ALPHA,
        "energy_tol": ENERGY_TOL, "fit_R2": R2, "n_traj": st["n_traj"],
    }
    out_file = save_bundle(os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}"), bundle)

    print(f"\n[{STRATEGY}]  {base}  |  {H}x{W}  C={C}  |  model = {model}, {n_rom} modes")
    print(f"  real train = {Ntr}   real val = {len(val_idx)}   synthetic = {n_aug}")
    print(f"  trajectories = {st['n_traj']}  (horizon {HORIZON}, IC noise {IC_NOISE})")
    print(f"  screening: kept {100*st['keep_frac']:.1f}% of integrated steps "
          f"(energy band [{st['E_band'][0]:.3g}, {st['E_band'][1]:.3g}], tol {ENERGY_TOL})")
    print(f"  saved → {out_file}\n")


if __name__ == "__main__":
    main()

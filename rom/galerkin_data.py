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

(rom version: the model-identification functions of pod_augment_galerkin.py.  The bundle
generator main() and its CONFIG were dropped with the summer pipeline; the loader and POD
are in rom.data / rom.pod, and integrate_trajectories was an identical copy of the one in
rom.galerkin_ns, which is re-exported here.)
"""

import numpy as np

from .galerkin_ns import integrate_trajectories      # noqa: F401  (identical copy, used as gd.integrate_trajectories)


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


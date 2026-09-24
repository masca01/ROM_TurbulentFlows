"""
Synthetic-snapshot generation and the augmentation arms.

Settings
    region map and every later study (run_region_map.py): GEN_TIME, SUBSTEPS, IC_NOISE,
        ENERGY_TOL, RIDGE, MAX_TRAJ, GENERATORS (their index is part of the trajectory seeds),
        FID_WINDOWS, FID_CLIP (quality error of the reduced model)
    summer Re100 generators (re100_fraction_sweep.py): SUMMER_*

Functions
    n_aug_for          synthetic count for a synthetic fraction f = n_aug / (n_real + n_aug)
    generate_pool      summer generator (data-identified or NS-projected) on a real set
    fidelity           quality error of a reduced model on held-out real windows
    fidelity_windows   the held-out windows
    synthetic_arm      integrate the reduced model, screen, rebuild the snapshots
    jitter_arm         real training coefficients + noise, rebuilt on the same modes
    real_arm           extra real snapshots from the pool, projected on the same modes
                       (real_proj) or full fields (real_full)
    save_bundle        .mat / .npz bundle writer
"""
import numpy as np
import scipy.io as sio

from .pod import reconstruct, compute_pod
from . import galerkin_ns as pns
from . import galerkin_data as pg

# ---- region map and every later study (run_region_map.py) ----
GENERATORS  = ["galerkin", "galerkin_ns"]
# horizon in CONVECTIVE time so datasets with different snapshot spacing (dt 0.2 vs 1)
# integrate the same physical time
GEN_TIME    = 10.0               # convective time units per synthetic trajectory
SUBSTEPS    = 20
IC_NOISE    = 0.05
ENERGY_TOL  = 0.05
RIDGE       = 1e-6
MAX_TRAJ    = 20000
# fidelity: held-out real windows of GEN_TIME, started from the true projected state
FID_WINDOWS = 20
FID_CLIP    = 2.0                # a window's relative error is clipped at 200 % (blow-ups)

# ---- summer Re100 generators (re100_fraction_sweep.py; do not change to reproduce) ----
SUMMER_FILE       = "2PlatesGap/Data2PlatesGap1Re100.mat"   # relative to paths.DATA
SUMMER_SEED       = 7            # split AND trajectory rng (one generator, as in summer)
SUMMER_VAL_FRAC   = 0.10
SUMMER_RE, SUMMER_DT = 100, 1.0
SUMMER_N_ROM      = {"galerkin": 30, "galerkin_ns": 25}
SUMMER_RIDGE      = 1e-6         # galerkin only
SUMMER_HORIZON    = 20
SUMMER_SUBSTEPS   = 20
SUMMER_IC_NOISE   = 0.05
SUMMER_ENERGY_TOL = 0.05
SUMMER_MAX_TRAJ   = 20000


def n_aug_for(f, n_real):
    return 0 if f <= 0 else int(round(f / (1.0 - f) * n_real))


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


# ------------------------------- summer generator -------------------------------

def generate_pool(train_real, tr_idx, n_aug_max, rng, generator, n_rom, data_file):
    """Synthetic coefficient states with the summer generator settings; returns
    (train_aug [n_aug_max, C, H, W], info).  generator = "galerkin" | "galerkin_ns";
    n_rom = model size (the summer value is SUMMER_N_ROM[generator]); data_file is read
    for the grid spacing (NS only)."""
    # local names = the old re100_fraction_sweep globals, so the body below is the verbatim summer code
    GENERATOR, DATA_FILE, RE, DT, RIDGE, HORIZON, SUBSTEPS, IC_NOISE, ENERGY_TOL, MAX_TRAJ = (
        generator, data_file, SUMMER_RE, SUMMER_DT, SUMMER_RIDGE, SUMMER_HORIZON, SUMMER_SUBSTEPS,
        SUMMER_IC_NOISE, SUMMER_ENERGY_TOL, SUMMER_MAX_TRAJ)
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    C, H, W = shape
    n_rom = min(n_rom, A.shape[1])
    e_rom = float((S[:n_rom] ** 2).sum() / (S ** 2).sum())
    if GENERATOR == "galerkin":
        beta, s, R2, n_fit = pg.fit_galerkin(A[:, :n_rom], tr_idx, DT, RIDGE)
        B_real = np.asarray(A[:, :n_rom], dtype=np.float64) / s[None, :]
        step = pg.make_step(beta, "ode", DT, SUBSTEPS)
        print(f"[{GENERATOR}]  ode fit: {n_rom} modes ({100*e_rom:.2f}% energy), "
              f"{beta.shape[0]} regressors, {n_fit} samples; R^2 min {R2.min():.3f} max {R2.max():.3f}")
        B_new, st = pg.integrate_trajectories(step, s, B_real, n_aug_max, HORIZON,
                                              IC_NOISE, ENERGY_TOL, MAX_TRAJ, rng)
        A_new = B_new * s[None, :]
    else:
        dx, dy = pns.grid_spacing(DATA_FILE); kappa = np.sqrt(dx * dy)
        U = (Xc.T @ Wp[:, :n_rom]).astype(np.float64)
        Pu = np.empty((n_rom + 1, H, W)); Pv = np.empty((n_rom + 1, H, W))
        mean_f = x_mean.astype(np.float64).reshape(C, H, W)
        Pu[0], Pv[0] = mean_f[0], mean_f[1]
        modes = (U / kappa).T.reshape(n_rom, C, H, W); Pu[1:], Pv[1:] = modes[:, 0], modes[:, 1]
        del U, modes
        A_phys = np.asarray(A[:, :n_rom], dtype=np.float64) * kappa
        l, q = pns.galerkin_operators_ns(Pu, Pv, dx, dy, RE); del Pu, Pv
        R2 = pns.derivative_R2(l, q, A_phys, tr_idx, DT)
        s = A_phys.std(axis=0); s = np.where(s < 1e-14, 1.0, s)
        B_real = A_phys / s[None, :]
        step = pns.make_step_ns(l, q, s, DT, SUBSTEPS)
        print(f"[{GENERATOR}]  NS projection: {n_rom} modes ({100*e_rom:.2f}% energy); "
              f"derivative R^2 min {R2.min():.3f} max {R2.max():.3f}")
        B_new, st = pns.integrate_trajectories(step, s, B_real, n_aug_max, HORIZON,
                                               IC_NOISE, ENERGY_TOL, MAX_TRAJ, rng)
        A_new = (B_new * s[None, :]) / kappa
    print(f"[{GENERATOR}]  {n_aug_max} synthetic states from {st['n_traj']} trajectories, "
          f"{100*st['keep_frac']:.1f}% of integrated steps kept", flush=True)
    train_aug = reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new, shape)
    del Xc
    return train_aug, {"n_rom": n_rom, "rom_energy": e_rom, "n_traj": st["n_traj"],
                       "keep_frac": st["keep_frac"], "R2_min": float(R2.min())}


# ------------------------------- quality of the reduced model -------------------------------

def fidelity(step, to_b, from_b, win_A, steps):
    """Relative K-mode error of the model over held-out real windows, started from the
    true projected state: sum |a_pred - a_true|^2 / sum |a_true|^2 in POD-coefficient units
    (energy-weighted, comparable between generators), per window, clipped at FID_CLIP."""
    errs, blown = [], 0
    for Aw in win_A:
        b = to_b(Aw[0]); num = den = 0.0; ok = True
        cap = 50.0 * np.abs(to_b(Aw)).max()
        for t in range(1, steps + 1):
            with np.errstate(over="ignore", invalid="ignore"):
                b = step(b)
            if not np.all(np.isfinite(b)) or np.abs(b).max() > cap:
                ok = False; break
            a_pred = from_b(b)
            num += float(((a_pred - Aw[t]) ** 2).sum()); den += float((Aw[t] ** 2).sum())
        if not ok:
            blown += 1; errs.append(FID_CLIP)
        else:
            errs.append(min(num / max(den, 1e-30), FID_CLIP))
    return float(np.mean(errs)), blown / max(len(win_A), 1)


def fidelity_windows(tr_set, Nt, steps, rng):
    """Up to FID_WINDOWS non-overlapping runs of steps+1 consecutive snapshots that contain
    no training snapshot. They may include validation snapshots: fidelity is a diagnostic of
    the generator and is never used to train or select the network, so nothing leaks."""
    L = steps + 1
    ok = [t for t in range(Nt - L + 1) if all((t + j) not in tr_set for j in range(L))]
    rng.shuffle(ok)
    taken, chosen = set(), []
    for t in ok:
        if taken.isdisjoint(range(t, t + L)):
            chosen.append(t); taken.update(range(t, t + L))
        if len(chosen) == FID_WINDOWS:
            break
    return sorted(chosen)


# ------------------------------- augmentation arms -------------------------------
# (inline in run_experiments*.py = run_round1/2.py and run_lowdata*.py; same operations and rng calls, in the same order)

def synthetic_arm(step, s, to_b, from_b, P, A, K, n, steps, rng,
                  kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ):
    """n synthetic snapshots: integrate the reduced model from kicked real states (screened
    by the energy band) and rebuild them on the K modes of the subset POD P.
    A = the subset's K-mode POD coefficients. Returns (aug [n, C, H, W], screening stats)."""
    B_new, st = pns.integrate_trajectories(step, s, to_b(A), n, steps, kick, energy_tol, max_traj, rng)
    aug = reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], from_b(B_new), P["shape"])
    return aug, st


def jitter_arm(P, A, K, n, grng, kick=IC_NOISE):
    """n real training coefficient vectors + kick * (per-mode std) noise, rebuilt on the same
    K modes: new samples but no dynamics (the lower reference)."""
    rows_ = grng.integers(0, len(A), size=n)
    A_new = A[rows_] + kick * A.std(axis=0)[None, :] * grng.standard_normal((n, K))
    return reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], A_new, P["shape"])


def real_arm(P, K, data, pool_idx, tr_idx, n, grng, project=True, cap=True):
    """n EXTRA real snapshots from the pool that the subset does not use.
    project=True  -> projected on the same K modes (real_proj: what a perfect reduced model gives)
    project=False -> full fields (real_full: the upper bound)
    cap=True      -> take at most what the pool still holds (run_round2.py / run_lowdata*);
    cap=False     -> ask for exactly n (run_round1.py: numpy raises when the pool is short).
    Returns (aug, number taken)."""
    avail = np.setdiff1d(pool_idx, tr_idx)
    take = min(n, len(avail)) if cap else n
    extra = np.sort(grng.choice(avail, size=take, replace=False))
    if not project:
        return data[extra].astype(np.float32), take
    U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float32)
    A_x = (data[extra].reshape(take, -1) - P["x_mean"]) @ U
    del U
    return reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], A_x, P["shape"]), take

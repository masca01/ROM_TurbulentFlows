"""
POD-Galerkin (Navier-Stokes projection) used to GENERATE a synthetic ensemble
from very little real data, then score that ensemble against held-out real data.

The experiment this script runs (Re = 50, 2-plates gap):

  * I keep only the first N_REAL snapshots of the record (40 by default) and split
    them IN TIME: the first (N_REAL - N_VAL) are training, the last N_VAL are a
    contiguous real validation tail.  Nothing from the validation tail is ever
    used to build the model — it exists only to judge the synthetic data.
  * I build the POD basis and project the incompressible Navier-Stokes operators
    (convection + diffusion) onto it using ONLY the training snapshots.  This is
    the same physics-derived (not fitted) Galerkin model as
    pod_augment_galerkin_ns.py.
  * I then launch N_TRAJ synthetic trajectories.  Each one starts from a
    "strategy C" initial point — a real TRAINING snapshot plus a small random
    kick — and is integrated forward for exactly HORIZON snapshot intervals.
    Every trajectory contributes HORIZON snapshots, so the ensemble has
    N_TRAJ * HORIZON synthetic states (N_TRAJ = 10, HORIZON = 20 -> ~200).
    N_TRAJ is the single knob that sets how much synthetic data I create.
  * Finally I ask the real question: how good is this synthetic ensemble?  I
    compare it to the 20 real validation snapshots at the level of the whole
    ENSEMBLE (its energy distribution, its per-mode coefficient statistics, and
    its point-wise fluctuation field), not step by step.  The training set is
    shown alongside as the "best a synthetic set could hope to look like".

Because the model is derived from the equations and truncated, it is unclosed
and can drift over a 20-step horizon; a trajectory that blows up is truncated at
the last finite state and reported, which is itself part of "evaluating the
capability" of the method from only 20 training snapshots.

Output: ../DATA/AUGMENTED/<base>_aug_galerkin_ns_ensemble.{mat|npz}, a bundle with
train_real / train_aug / val_real (train_aug = the synthetic ensemble) plus the
evaluation arrays, so it also plugs straight into train_augmented_vae.py.
"""

import os
import argparse
import numpy as np

from pod_augment_galerkin_ns import (load_data, compute_pod, reconstruct,
                                     save_bundle, grid_spacing,
                                     galerkin_operators_ns, make_step_ns,
                                     derivative_R2)

# ------------ Folder layout ------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
os.makedirs(_AUG_DIR, exist_ok=True)

# ============ CONFIG ============
DATA_FILE   = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/2PlatesGap/Data2PlatesGap1Re50.mat"
COMP_IDX    = None     # None = auto-detect the velocity components in the file

# how much REAL data I let myself use, and how it is split (temporal tail)
N_REAL      = 50       # how many real snapshots are used in total (train + val)
N_VAL       = 20       # how many of those are held out as real validation
                       # -> training = N_REAL - N_VAL snapshots (50 by default)
SPLIT       = "random" # "random"   : pick the N_REAL snapshots at random from the
                       #              whole record and split train/val at random,
                       #              so both sets sample the SAME energy
                       #              distribution (fixes the temporal-drift
                       #              mismatch where a time-tail validation sits
                       #              in a higher-energy part of the run);
                       # "temporal" : first N_REAL snapshots in time, validation =
                       #              the last N_VAL (contiguous tail).
SPLIT_SEED  = 7        # reproducible split (same seed the other generators use)

# physical parameters (MUST match DATA_FILE: Re=50 -> 50)
RE          = 50       # Reynolds number of the simulation
DT          = 1        # snapshot spacing, in convective time units
DX_FALLBACK = 0.08     # grid spacing fallback if coordinates can't be read

# reduced-order model
N_ROM       = 30       # POD modes kept in the model (capped at the POD rank,
                       # which is at most N_TRAIN - 1 with so few snapshots).
                       # With only 20 training snapshots the tail modes have no
                       # reliable dynamics (their projected derivative R^2 goes
                       # negative past ~mode 11), so an unclosed model that keeps
                       # them drifts and trajectories blow up before reaching the
                       # full horizon.  Keeping ~10 modes lets every trajectory
                       # survive all HORIZON steps, so N_TRAJ * HORIZON snapshots
                       # are actually produced.

# synthetic-ensemble generation
N_TRAJ      = 10       # <<< THE PARAMETER >>> number of trajectories to launch.
                       # ensemble size = N_TRAJ * HORIZON  (10 * 20 = 200)
HORIZON     = 20       # snapshot intervals integrated per trajectory (fixed length)
SUBSTEPS    = 20       # RK4 sub-steps taken inside each snapshot interval
IC_NOISE    = 0.05     # strategy-C kick: initial perturbation as a fraction of
                       # each mode's std (0 = start exactly on a real snapshot)
BLOWUP_CAP  = 10.0     # truncate a trajectory once any coefficient exceeds this
                       # multiple of the real coefficient maximum
RNG_SEED    = 7        # reproducible initial points and kicks

MAKE_PLOT   = True     # save a small quality-diagnostics figure next to the bundle
STRATEGY    = "galerkin_ns_ensemble"
# ================================


# ------------------------- Ensemble generation -------------------------

def generate_ensemble(step, B_train, n_traj, horizon, ic_noise, rng, blowup_cap):
    """Launch `n_traj` fixed-length trajectories and keep every snapshot along
    them (no energy screening — the ensemble size is deterministic unless a
    trajectory diverges).

    step             : advances a standardized state by one snapshot interval
    B_train[Ntr, r]  : standardized TRAINING coefficients; the pool of strategy-C
                       initial points (validation coefficients are never used)
    Returns the stacked synthetic states (standardized) and per-trajectory stats.
    """
    Ntr, r = B_train.shape
    snaps, kept_per_traj = [], []
    n_blowup = 0
    for _ in range(n_traj):
        # strategy C: a real training snapshot plus a small random kick
        b = B_train[rng.integers(0, Ntr)] + ic_noise * rng.normal(size=r)
        kept = 0
        for _ in range(horizon):
            with np.errstate(over="ignore", invalid="ignore"):
                b = step(b)
            if not np.all(np.isfinite(b)) or np.abs(b).max() > blowup_cap:
                n_blowup += 1                            # unclosed model drifted
                break
            snaps.append(b.copy())
            kept += 1
        kept_per_traj.append(kept)
    if not snaps:
        raise RuntimeError("every trajectory diverged immediately — try fewer "
                           "modes (N_ROM) or a shorter HORIZON")
    stats = {"n_traj": n_traj, "n_blowup": n_blowup,
             "kept_per_traj": np.array(kept_per_traj, dtype=int),
             "n_snaps": len(snaps)}
    return np.asarray(snaps, dtype=np.float64), stats


# ------------------------- Ensemble scoring -------------------------

def _energy(A_phys):
    """Per-snapshot modal fluctuation energy E = 1/2 * sum_i a_i^2."""
    return 0.5 * (A_phys ** 2).sum(axis=1)


def _describe(name, e):
    return (f"  {name:<12s} n={e.size:4d}  E: min {e.min():8.4g}  "
            f"mean {e.mean():8.4g}  max {e.max():8.4g}  std {e.std():8.4g}")


# ------------------------------- Main -------------------------------

def main(n_traj=N_TRAJ, n_rom=N_ROM, horizon=HORIZON):
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape
    if C != 2:
        raise ValueError(f"NS projection needs 2D (u,v) fields, got C={C}")
    if N_REAL > Nt:
        raise ValueError(f"asked for {N_REAL} real snapshots but the file has {Nt}")
    dx, dy = grid_spacing(DATA_FILE)
    kappa = np.sqrt(dx * dy)                              # vector -> physical modes
    print(f"[{STRATEGY}]  Re = {RE:g}, dt = {DT:g}, dx = {dx:.4f}, dy = {dy:.4f}")

    # Select the N_REAL real snapshots and split them into training / validation.
    # Either way the validation set is used ONLY for scoring, never for fitting,
    # and the strategy-C initial points later come ONLY from training snapshots.
    n_val = int(N_VAL)
    if SPLIT == "random":
        # Draw N_REAL snapshots at random from the whole record, then split at
        # random, so training and validation sample the same distribution.
        split_rng = np.random.default_rng(SPLIT_SEED)
        used = split_rng.permutation(Nt)[:N_REAL]
        tr_idx  = np.sort(used[:N_REAL - n_val])
        val_idx = np.sort(used[N_REAL - n_val:])
        how = "random"
    elif SPLIT == "temporal":
        # First N_REAL snapshots in time; validation = the contiguous tail.
        tr_idx  = np.arange(0, N_REAL - n_val)
        val_idx = np.arange(N_REAL - n_val, N_REAL)
        how = "temporal tail"
    else:
        raise ValueError(f"SPLIT must be 'random' or 'temporal', got {SPLIT!r}")
    train_real = data[tr_idx].copy()
    val_real   = data[val_idx].copy()
    del data
    Ntr = len(tr_idx)
    print(f"[{STRATEGY}]  real snapshots used = {N_REAL}  ->  train {Ntr} / "
          f"val {len(val_idx)}  ({how} split, seed {SPLIT_SEED})")

    # POD on the TRAINING snapshots only.
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    r_full = A.shape[1]
    n_rom = int(min(n_rom, r_full))
    e_rom = (S[:n_rom] ** 2).sum() / (S ** 2).sum()
    print(f"[{STRATEGY}]  POD rank = {r_full} (<= N_train-1);  keeping {n_rom} modes "
          f"({100*e_rom:.2f}% of training energy)")

    # Put the mean flow + leading modes on the grid with the physical
    # normalization the NS projection expects (rescale by 1/sqrt(dx*dy)).
    U = (Xc.T @ Wp[:, :n_rom]).astype(np.float64)         # [D, r], vector-unit
    Pu = np.empty((n_rom + 1, H, W))
    Pv = np.empty((n_rom + 1, H, W))
    mean_f = x_mean.astype(np.float64).reshape(C, H, W)
    Pu[0], Pv[0] = mean_f[0], mean_f[1]                   # mean flow = "mode 0"
    modes = (U / kappa).T.reshape(n_rom, C, H, W)         # phi = U / sqrt(dA)
    Pu[1:], Pv[1:] = modes[:, 0], modes[:, 1]
    del U, modes

    # Physical training coefficients (to match the physically-normalized modes).
    A_phys = np.asarray(A[:, :n_rom], dtype=np.float64) * kappa

    print(f"[{STRATEGY}]  projecting NS operators onto {n_rom} modes ...")
    l, q = galerkin_operators_ns(Pu, Pv, dx, dy, RE)
    del Pu, Pv
    # The derivative-R^2 check needs time-consecutive snapshot triples (central
    # differences of da/dt).  A random split keeps almost none, so I only report
    # it when enough survive; otherwise it is not meaningful and I skip it.
    n_triples = int(((np.diff(tr_idx[:-1]) == 1) & (np.diff(tr_idx[1:]) == 1)).sum())
    if n_triples >= 3:
        R2 = derivative_R2(l, q, A_phys, tr_idx, DT)
        print("  derivative R^2 per mode (projected, NOT fitted): "
              + "  ".join(f"{v:.3f}" for v in R2))
    else:
        R2 = np.full(n_rom, np.nan)
        print(f"  derivative R^2: skipped ({how} split has only {n_triples} "
              "time-consecutive triples — diagnostic needs consecutive snapshots)")

    # Standardize the coefficients (per-mode scale s), as in the generator, so
    # IC_NOISE means "a fraction of each mode's spread".
    s = A_phys.std(axis=0)
    s = np.where(s < 1e-14, 1.0, s)
    B_train = A_phys / s[None, :]                         # strategy-C IC pool

    # ---- generate the synthetic ensemble ----
    rng = np.random.default_rng(RNG_SEED)
    step = make_step_ns(l, q, s, DT, SUBSTEPS)
    blowup_cap = BLOWUP_CAP * np.abs(B_train).max()
    B_new, gst = generate_ensemble(step, B_train, n_traj, horizon, IC_NOISE,
                                   rng, blowup_cap)
    A_syn = B_new * s[None, :]                            # physical synthetic coeffs
    print(f"\n[{STRATEGY}]  launched {gst['n_traj']} trajectories (horizon "
          f"{horizon}, IC noise {IC_NOISE}) -> {gst['n_snaps']} synthetic snapshots")
    if gst["n_blowup"]:
        print(f"  !! {gst['n_blowup']} trajectory(ies) diverged and were truncated; "
              f"kept-per-trajectory = {gst['kept_per_traj'].tolist()}")

    # ---- score the ensemble against the REAL validation tail ----
    # Project the real validation snapshots onto the TRAINING POD modes to get
    # their coefficients (physical units), so all three sets live in the same
    # coordinate system.
    Xv = val_real.reshape(len(val_idx), -1).astype(np.float64)
    A_val = (Wp[:, :n_rom].T @ (Xc @ (Xv - x_mean.astype(np.float64)).T)).T
    A_val_phys = A_val * kappa

    e_tr, e_val, e_syn = _energy(A_phys), _energy(A_val_phys), _energy(A_syn)
    lo, hi = e_val.min(), e_val.max()
    in_val_band  = float(np.mean((e_syn >= lo) & (e_syn <= hi)))
    tlo, thi = e_tr.min(), e_tr.max()
    in_tr_band   = float(np.mean((e_syn >= tlo) & (e_syn <= thi)))

    # Per-mode marginals: how close are the synthetic mean/std of each coefficient
    # to the real validation ones?  Reported as an aggregate relative difference.
    mu_val, sd_val = A_val_phys.mean(0), A_val_phys.std(0)
    mu_syn, sd_syn = A_syn.mean(0),      A_syn.std(0)
    denom = sd_val + 1e-12
    std_ratio = sd_syn / denom                            # 1.0 = perfect spread
    mean_shift = np.abs(mu_syn - mu_val) / denom          # in units of real std
    std_rel_err = np.abs(sd_syn - sd_val) / denom

    # Field-level: compare the point-wise RMS fluctuation field of the synthetic
    # ensemble to that of the real validation snapshots (relative L2 over the grid).
    A_new_vec = A_syn / kappa                             # back to vector units
    train_aug = reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new_vec, shape)
    del Xc
    rms_syn = train_aug.std(axis=0)                       # [C,H,W]
    rms_val = val_real.std(axis=0)
    rms_tr  = train_real.std(axis=0)
    field_rms_rel = (np.linalg.norm(rms_syn - rms_val)
                     / (np.linalg.norm(rms_val) + 1e-12))
    field_rms_tr_rel = (np.linalg.norm(rms_tr - rms_val)
                        / (np.linalg.norm(rms_val) + 1e-12))

    # ---- report ----
    print(f"\n[{STRATEGY}]  ENSEMBLE QUALITY vs the {len(val_idx)} real validation "
          "snapshots")
    print("  modal fluctuation energy (1/2 sum a_i^2):")
    print(_describe("train real", e_tr))
    print(_describe("val real",   e_val))
    print(_describe("synthetic",  e_syn))
    print(f"  synthetic snapshots inside the real VAL energy band [{lo:.4g},{hi:.4g}]: "
          f"{100*in_val_band:.1f}%")
    print(f"  synthetic snapshots inside the real TRAIN energy band [{tlo:.4g},{thi:.4g}]: "
          f"{100*in_tr_band:.1f}%")
    print("  per-mode coefficient statistics (aggregate over the "
          f"{n_rom} modes, in units of the real val std):")
    print(f"     spread    : median std ratio  {np.median(std_ratio):.2f}  "
          f"(1.0 = matches real);  mean |std err| {std_rel_err.mean():.2f}")
    print(f"     centering : mean |mean shift|  {mean_shift.mean():.2f}")
    print("  point-wise RMS fluctuation field, relative L2 error vs real val:")
    print(f"     synthetic ensemble : {field_rms_rel:.3f}")
    print(f"     (reference) train  : {field_rms_tr_rel:.3f}")

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bundle = {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object),
        "strategy": STRATEGY, "tr_idx": tr_idx, "val_idx": val_idx,
        "split": SPLIT, "split_seed": SPLIT_SEED, "model": "ns_ode_ensemble",
        "n_real": N_REAL, "n_val": n_val,
        "n_modes": r_full, "n_rom": n_rom, "dt": DT, "re": RE, "dx": dx, "dy": dy,
        "n_traj": gst["n_traj"], "horizon": horizon, "substeps": SUBSTEPS,
        "ic_noise": IC_NOISE, "n_blowup": gst["n_blowup"],
        "kept_per_traj": gst["kept_per_traj"], "rng_seed": RNG_SEED, "fit_R2": R2,
        # scoring arrays
        "E_train": e_tr, "E_val": e_val, "E_syn": e_syn,
        "in_val_band": in_val_band, "in_train_band": in_tr_band,
        "mode_std_ratio": std_ratio, "mode_std_rel_err": std_rel_err,
        "mode_mean_shift": mean_shift,
        "field_rms_rel": field_rms_rel, "field_rms_train_rel": field_rms_tr_rel,
    }
    out_file = save_bundle(os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}"), bundle)
    print(f"\n[{STRATEGY}]  {base}  |  {H}x{W}  C={C}  |  ns_ode ensemble, {n_rom} modes")
    print(f"  real train = {Ntr}   real val = {len(val_idx)}   synthetic = {gst['n_snaps']}")
    print(f"  saved -> {out_file}")

    # ---- optional quality-diagnostics figure ----
    if MAKE_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            bins = np.linspace(min(e_tr.min(), e_val.min(), e_syn.min()),
                               max(e_tr.max(), e_val.max(), e_syn.max()), 24)
            ax[0].hist(e_syn, bins=bins, alpha=0.55, color="#D6172F",
                       label="synthetic", density=True)
            ax[0].hist(e_val, bins=bins, alpha=0.55, color="#3F3F3F",
                       label="real val", density=True)
            ax[0].axvspan(lo, hi, color="#76777B", alpha=0.12)
            ax[0].set_xlabel("modal fluctuation energy  1/2 sum a$_i^2$")
            ax[0].set_ylabel("density"); ax[0].legend(frameon=False)
            ax[0].set_title("Ensemble energy vs real validation", loc="left")
            ks = np.arange(1, n_rom + 1)
            ax[1].bar(ks - 0.2, sd_val, width=0.4, color="#3F3F3F", label="real val")
            ax[1].bar(ks + 0.2, sd_syn, width=0.4, color="#D6172F", label="synthetic")
            ax[1].set_xlabel("POD mode"); ax[1].set_ylabel("coefficient std")
            ax[1].legend(frameon=False)
            ax[1].set_title("Per-mode spread", loc="left")
            for a in ax:
                a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
            fig.tight_layout()
            png = os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}_quality.png")
            fig.savefig(png, dpi=170, bbox_inches="tight"); plt.close(fig)
            print(f"  quality figure -> {png}")
        except Exception as err:
            print(f"  (skipped diagnostics figure: {err})")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate a synthetic ensemble with the NS-projected POD-Galerkin "
                    "model and score it against the real validation tail.")
    ap.add_argument("--n-traj", type=int, default=N_TRAJ,
                    help=f"number of strategy-C trajectories to launch "
                         f"(ensemble size = n_traj * horizon; default {N_TRAJ})")
    ap.add_argument("--n-rom", type=int, default=N_ROM,
                    help=f"POD modes kept in the reduced model (default {N_ROM})")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help=f"snapshots produced per trajectory (default {HORIZON})")
    args = ap.parse_args()
    main(n_traj=args.n_traj, n_rom=args.n_rom, horizon=args.horizon)

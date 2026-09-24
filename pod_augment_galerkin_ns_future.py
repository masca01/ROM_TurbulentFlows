"""
POD-Galerkin (Navier-Stokes projection) used as a FORWARD PREDICTOR, with the
prediction validated against held-out real data.

Same physics-projected Galerkin model as pod_augment_galerkin_ns.py, but used to
predict the future of the flow instead of building an augmentation ensemble.
The difference from the other generators is the SPLIT: here the validation set
is the last VAL_FRAC fraction of the snapshots IN TIME (a contiguous tail), not
a random subset.  That way I can:

  1. train the POD and project the NS model on the first part of the record;
  2. start a single trajectory from the LAST TRAINING snapshot and integrate it
     forward;
  3. because the held-out snapshots come right after in time, the first n_val
     predicted steps line up with real snapshots and can be VALIDATED against
     them (I never use the validation data to build the model, so this is a
     genuine out-of-sample check).

I can keep predicting past the validation window too (e.g. N_FUTURE = Ntr for a
100%-length forecast), but only the first n_val steps have real data to compare
with; the rest is pure extrapolation.  Since the truncated Galerkin model is
unclosed it will drift eventually, so the validation error is expected to grow
with the horizon.

Note: because the split here is temporal, val_real is NOT the same as in the
augmentation generators (which use a shared random seed) — this script is a
prediction/validation experiment, not a drop-in augmentation comparison.

Output: ../DATA/AUGMENTED/<base>_aug_galerkin_ns_future.{mat|npz}, a bundle with
train_real / train_aug / val_real (train_aug = the predicted future snapshots)
plus the per-step validation errors.
"""

import os
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
COMP_IDX    = None     # None = auto-detect the velocity components
VAL_FRAC    = 0.10     # last fraction of snapshots (IN TIME) held out to validate

# physical parameters (MUST match DATA_FILE: Re=100 -> 100, Re=50 -> 50)
RE          = 50
DT          = 1        # snapshot spacing, in convective time units
DX_FALLBACK = 0.08

# model + prediction
N_ROM       = 20       # POD modes kept in the model
N_FUTURE    = None     # future snapshots to predict (None = n_val, i.e. exactly
                       # the validatable window; set e.g. Ntr for a longer run,
                       # of which only the first n_val steps can be validated)
SUBSTEPS    = 20       # RK4 sub-steps per snapshot interval
BLOWUP_CAP  = 10.0     # stop if any coefficient exceeds this * the real maximum

STRATEGY    = "galerkin_ns_future"
# ================================


def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape
    if C != 2:
        raise ValueError(f"NS projection needs 2D (u,v) fields, got C={C}")
    dx, dy = grid_spacing(DATA_FILE)
    kappa = np.sqrt(dx * dy)                              # vector -> physical modes
    print(f"[{STRATEGY}]  Re = {RE:g}, dt = {DT:g}, dx = {dx:.4f}, dy = {dy:.4f}")

    # TEMPORAL split: the last n_val snapshots (in time) are the validation tail,
    # everything before is training.  This is what lets the forward trajectory be
    # checked against real future data.
    n_val   = max(1, round(VAL_FRAC * Nt))
    tr_idx  = np.arange(0, Nt - n_val)
    val_idx = np.arange(Nt - n_val, Nt)
    train_real = data[tr_idx].copy()
    val_real   = data[val_idx].copy()
    del data
    Ntr = len(tr_idx)
    n_future = int(N_FUTURE) if N_FUTURE else n_val

    # POD on the training snapshots only
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    r_full = A.shape[1]
    n_rom = int(min(N_ROM, r_full))

    # Put the mean flow + leading modes on the grid with the physical
    # normalization the projection needs (same as pod_augment_galerkin_ns.py).
    U = (Xc.T @ Wp[:, :n_rom]).astype(np.float64)        # [D, r], vector-unit
    Pu = np.empty((n_rom + 1, H, W))
    Pv = np.empty((n_rom + 1, H, W))
    mean_f = x_mean.astype(np.float64).reshape(C, H, W)
    Pu[0], Pv[0] = mean_f[0], mean_f[1]                   # mean flow = "mode 0"
    modes = (U / kappa).T.reshape(n_rom, C, H, W)         # phi = U / sqrt(dA)
    Pu[1:], Pv[1:] = modes[:, 0], modes[:, 1]
    del U, modes

    # physical training coefficients (to match the physically-normalized modes)
    A_phys = np.asarray(A[:, :n_rom], dtype=np.float64) * kappa

    print(f"[{STRATEGY}]  projecting NS operators onto {n_rom} modes "
          f"({100*(S[:n_rom]**2).sum()/(S**2).sum():.2f}% energy) ...")
    l, q = galerkin_operators_ns(Pu, Pv, dx, dy, RE)
    del Pu, Pv
    R2 = derivative_R2(l, q, A_phys, tr_idx, DT)
    print("  derivative R^2 per mode (projected, NOT fitted): "
          + "  ".join(f"{v:.3f}" for v in R2))

    # Standardize the coefficients (per-mode scale s), as in the generator.
    s = A_phys.std(axis=0)
    s = np.where(s < 1e-14, 1.0, s)

    # Start from the LAST TRAINING snapshot.  With the temporal split its
    # coefficients are simply the last row of A, so no projection is needed.
    b = A_phys[-1] / s                                   # standardized start state

    # March forward n_future snapshot intervals.
    step  = make_step_ns(l, q, s, DT, SUBSTEPS)
    b_cap = BLOWUP_CAP * np.abs(A_phys / s).max()
    B_future = np.empty((n_future, n_rom), dtype=np.float64)
    made = 0
    for k in range(n_future):
        with np.errstate(over="ignore", invalid="ignore"):
            b = step(b)
        if not np.all(np.isfinite(b)) or np.abs(b).max() > b_cap:
            print(f"  !! prediction blew up after {made} future steps "
                  "(unclosed model) — keeping the finite ones")
            break
        B_future[made] = b
        made += 1
    if made == 0:
        raise RuntimeError("the forward prediction diverged immediately — "
                           "try fewer modes (N_ROM) or a shorter N_FUTURE")
    B_future = B_future[:made]

    # ---- validate against the real held-out tail ----
    # Predicted step k corresponds in time to the real validation snapshot k
    # (both start right after the last training snapshot).  I compare in two
    # ways: in the POD-coefficient space (isolates the DYNAMICS error) and in the
    # full field (includes the truncation error of keeping only n_rom modes).
    n_check = min(made, n_val)
    Xv = val_real[:n_check].reshape(n_check, -1).astype(np.float64)
    A_val = (Wp[:, :n_rom].T @ (Xc @ (Xv - x_mean.astype(np.float64)).T)).T
    B_val_true = (A_val * kappa) / s                     # true standardized coeffs
    coef_err = (np.linalg.norm(B_future[:n_check] - B_val_true, axis=1)
                / (np.linalg.norm(B_val_true, axis=1) + 1e-12))

    # Rebuild all predicted snapshots, then compare the first n_check to the real
    # validation fields (error relative to the real fluctuation magnitude).
    A_new = (B_future * s[None, :]) / kappa
    train_aug = reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new, shape)
    mean_field = x_mean.reshape(C, H, W)
    num = np.linalg.norm((train_aug[:n_check] - val_real[:n_check]).reshape(n_check, -1), axis=1)
    den = np.linalg.norm((val_real[:n_check] - mean_field).reshape(n_check, -1), axis=1)
    field_err = num / (den + 1e-12)
    del Xc

    print(f"\n[{STRATEGY}]  validation over the first {n_check} predicted steps "
          "(vs the real held-out tail):")
    print("  step:        " + "  ".join(f"{k+1:5d}" for k in range(min(n_check, 8))))
    print("  coef err:    " + "  ".join(f"{v:5.2f}" for v in coef_err[:8]))
    print("  field err:   " + "  ".join(f"{v:5.2f}" for v in field_err[:8]))
    print(f"  mean over window: coef {coef_err.mean():.3f}, field {field_err.mean():.3f}")

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bundle = {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object),
        "strategy": STRATEGY, "tr_idx": tr_idx, "val_idx": val_idx,
        "val_frac": VAL_FRAC, "split": "temporal", "model": "ns_ode_future",
        "n_modes": r_full, "n_rom": n_rom, "dt": DT, "re": RE,
        "dx": dx, "dy": dy, "substeps": SUBSTEPS,
        "n_future": made, "start_index": int(tr_idx[-1]), "fit_R2": R2,
        "val_coef_err": coef_err, "val_field_err": field_err,
    }
    out_file = save_bundle(os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}"), bundle)

    print(f"\n[{STRATEGY}]  {base}  |  {H}x{W}  C={C}  |  ns_ode forward, {n_rom} modes")
    print(f"  real train = {Ntr}   real val (temporal tail) = {len(val_idx)}")
    print(f"  future snapshots predicted from snapshot #{tr_idx[-1]} = {made} "
          f"(requested {n_future});  {n_check} of them validated")
    print(f"  saved → {out_file}\n")


if __name__ == "__main__":
    main()

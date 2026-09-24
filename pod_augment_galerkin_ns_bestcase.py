"""
BEST-CASE augmentation experiment for the beta-VAE with the NS-projected
POD-Galerkin generator (pod_augment_galerkin_ns.py), set up so that it plugs
straight into the numbers of the convergence study.

Why this case (from Alpha0_convergence.xlsx, random split, Re = 50):
    n real   NN error (Ek)      POD99 error     -> headroom for the NN
      250    0.045 (95.5 %)       0.011            4x
     1350    0.017 (98.3 %)       0.010            (all the real data)
  * the POD basis is already converged at 250 snapshots (POD99 0.0113 vs 0.0100
    with 1350), so a POD-based generator has a good basis to work with, while the
    network is still far from converged there (0.045 vs 0.017);
  * Re 50 needs only 26 modes for 99 % energy — the smallest ROM of all the
    datasets, so the unclosed Galerkin model is the least likely to drift;
  * the split is the RANDOM one: validation snapshots are statistically like the
    training ones, so the synthetic data are asked to fill the sampling gap, not
    to invent states the record never visited (which no augmentation can do —
    see the tail-split results).

What the script does
    1. loads the first T_MAX = 1500 snapshots (the common cap of the study) and
       splits them exactly like pod_convergence.py / nn_convergence.py
       (convergence_split.py, SPLIT_SEED 7) -> 150 real validation, 1350 pool;
    2. draws N_REAL = 250 training snapshots from the pool REPLAYING the random
       draws of nn_convergence.py (DRAW_SEED 11), so train_real IS the first
       n = 250 subset of the convergence run: its baseline is known
       (e = 0.0496, Ek = 95.0 %; mean of the 3 draws 0.045 / 95.5 %);
    3. POD on those 250 snapshots, NS projection of convection + diffusion onto
       the leading N_ROM modes (Re and dt are read from the .mat file), then
       integrates trajectories from kicked real snapshots (strategy C), keeping
       only states inside the observed energy band;
    4. writes ../DATA/AUGMENTED/<base>_aug_<STRATEGY>.npz with
       train_real (250) / train_aug (N_AUG) / val_real (150)  ->  run
       train_augmented_vae.py on it (USE_AUG True, then False for the baseline).

With N_AUG = 1100 the network trains on 250 real + 1100 synthetic = 1350
snapshots, i.e. the same budget as the last point of the convergence curve
(0.017), so the question is answered directly: how much of the way from 0.045 to
0.017 do the synthetic snapshots carry the network?

    python3 pod_augment_galerkin_ns_bestcase.py            # full run (writes ~5 GB)
    python3 pod_augment_galerkin_ns_bestcase.py --dry-run  # model health only

Command-line overrides (used by run_bestcase_augmentation.py to loop over the Re
sweep; every one is optional and defaults to the CONFIG value):
    --file <path.mat>   --split random|tail|tail5|tail2.5   --t-max 1500
    --n-real 250        --n-aug 1100|pool   ("pool": fill up to the pool size)
    --n-rom 26|k99|k95  ("k99": the modes holding 99 % of the training-subset energy)
    --dry-run
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pod_augment_galerkin_ns import (load_data, compute_pod, reconstruct, save_bundle,
                                     grid_spacing, galerkin_operators_ns, make_step_ns,
                                     derivative_R2, integrate_trajectories)
import convergence_split as cs

_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
os.makedirs(_AUG_DIR, exist_ok=True)

# ============ CONFIG (defaults; see the command-line overrides above) ============
DATA_FILE  = os.path.join(_DATA_DIR, "Alpha0", "dataRe50Alpha0_2.mat")
T_MAX      = 1500       # snapshot cap of the convergence study (first 1500 of 5000)
SPLIT      = "random"   # convergence_split mode: "random" (the best case), or
                        # "tail" / "tail5" / "tail2.5" to try the temporal splits
SPLIT_SEED = 7          # same as the convergence scripts
DRAW_SEED  = 11         # same as nn_convergence.py -> identical random subsets

N_REAL     = 250        # real training snapshots (a point of the convergence sweep:
                        # 5,10,15,20,30,50,100,150,250,400,650,900,1150 or the pool)
N_AUG      = "pool"     # synthetic snapshots; "pool" = pool size - N_REAL, so that
                        # real + synthetic = the full-pool budget (1350 -> 1100)

RE         = None       # None = read from the file ("Re"), else e.g. 50
DT         = None       # None = read from the file ("dt"), else e.g. 0.2
DX_FALLBACK = 0.08

N_ROM      = "k99"      # POD modes in the model: an int, or "k99" / "k95" = the
                        # modes holding 99 % / 95 % of the TRAINING-SUBSET energy
                        # (Re50, 250 snapshots -> 25).  Backs off automatically
                        # if the unclosed model drifts.
HORIZON    = 50         # snapshot intervals per trajectory (50 x dt 0.2 = 10 time
                        # units, a few shedding periods); short on purpose
SUBSTEPS   = 20         # RK4 sub-steps per snapshot interval
IC_NOISE   = 0.05       # strategy-C kick on the initial condition (x mode std)
ENERGY_TOL = 0.05       # allowed slack around the real [min, max] energy band
MAX_TRAJ   = 2000       # cap on trajectory launches
MIN_KEEP   = 0.20       # back off N_ROM if fewer than this fraction of steps survive
# ================================


def _cli(argv):
    """--key value overrides of the CONFIG above (plus the --dry-run flag)."""
    g = globals()
    keys = {"--file": "DATA_FILE", "--split": "SPLIT", "--t-max": "T_MAX",
            "--n-real": "N_REAL", "--n-aug": "N_AUG", "--n-rom": "N_ROM"}
    dry = False
    it = iter(argv)
    for a in it:
        if a == "--dry-run":
            dry = True
        elif a in keys:
            v = next(it, None)
            if v is None:
                raise SystemExit(f"{a} needs a value")
            if keys[a] in ("T_MAX", "N_REAL") or (keys[a] in ("N_AUG", "N_ROM") and v.isdigit()):
                v = int(v)
            g[keys[a]] = v
        else:
            raise SystemExit(f"unknown argument {a!r}; see the docstring")
    cs.check_mode(SPLIT)
    return dry


DRY_RUN  = _cli(sys.argv[1:])
STRATEGY = f"galerkin_ns_n{N_REAL}" + ("" if SPLIT == "random" else f"_{SPLIT}")

# the n sweep of nn_convergence.py (needed to replay its random draws)
_N_HEAD, _TAIL_STEP, _TAIL_MAX, _M_REPS = [5, 10, 15, 20, 30, 50, 100, 150, 250, 400], 250, 1500, 3


def convergence_subset(pool_idx, n_real):
    """Replay nn_convergence.py's draws so the returned subset is exactly its
    FIRST draw at n = n_real (same DRAW_SEED, same sweep order)."""
    N_pool = len(pool_idx)
    ceiling = min(_TAIL_MAX, N_pool)
    head = [v for v in _N_HEAD if v <= ceiling]
    start = (head[-1] if head else 0) + _TAIL_STEP
    n_vals = sorted(set(head + list(range(start, ceiling + 1, _TAIL_STEP)) + [ceiling]))
    if n_real not in n_vals:
        raise SystemExit(f"N_REAL = {n_real} is not a sweep point {n_vals}")
    rng = np.random.default_rng(DRAW_SEED)
    for n in n_vals:
        reps = 1 if n == N_pool else _M_REPS
        for _ in range(reps):
            sub = pool_idx if n == N_pool else rng.choice(pool_idx, size=n, replace=False)
            if n == n_real:
                return np.sort(sub)


def read_scalar(path, key, default):
    """Re / dt of the dataset: the CONFIG value if given, else the variable stored
    in the .mat file (Alpha0 files have "Re" and "dt"), else a fallback: Re parsed
    from the file name, dt = 1 convective time (the 2-plates Re100/Re50 files store
    neither; their Readme / loadData.py give dt = 1)."""
    if default is not None:
        return float(default)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if key in f:
                return float(np.array(f[key]).ravel()[0])
    except Exception:
        pass
    import re as _re
    if key == "Re":
        m = _re.search(r"Re(\d+)", os.path.basename(path))
        if m:
            print(f"[read_scalar]  'Re' not stored in the file -> {m.group(1)} from the file name")
            return float(m.group(1))
    if key == "dt":
        print("[read_scalar]  'dt' not stored in the file -> 1.0 (2-plates data: 1 convective "
              "time between snapshots, see loadData.py)")
        return 1.0
    raise SystemExit(f"could not determine {key!r} for {path} — set it in CONFIG")


def main():
    t0 = time.time()
    re_ = read_scalar(DATA_FILE, "Re", RE)
    dt  = read_scalar(DATA_FILE, "dt", DT)
    data, comp_names = load_data(DATA_FILE, None, 1, T_MAX)
    Nt, C, H, W = data.shape
    if C != 2:
        raise ValueError(f"NS projection needs 2D (u,v) fields, got C={C}")
    dx, dy = grid_spacing(DATA_FILE)
    kappa = np.sqrt(dx * dy)

    # ---- split exactly like the convergence study, then the SAME random subset ----
    val_idx, pool_idx, n_val = cs.split_indices(Nt, None, SPLIT, SPLIT_SEED)
    tr_idx = convergence_subset(pool_idx, N_REAL)
    n_aug = (len(pool_idx) - N_REAL) if N_AUG == "pool" else int(N_AUG)
    if n_aug <= 0:
        raise SystemExit(f"nothing to synthesise: pool {len(pool_idx)}, N_REAL {N_REAL}")
    train_real = data[tr_idx].copy()
    val_real   = data[val_idx].copy()
    del data
    print(f"[{STRATEGY}]  Re = {re_:g}  dt = {dt:g}  dx = {dx:.4f}  dy = {dy:.4f}")
    print(f"[{STRATEGY}]  split = {SPLIT} ({cs.describe(SPLIT)}): pool {len(pool_idx)}, "
          f"val {n_val}; training on {N_REAL} real snapshots drawn as in "
          f"nn_convergence.py (first draw at n = {N_REAL}); synthetic target {n_aug}")

    # ---- POD on the real training subset ----
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    r_full = A.shape[1]
    e_cum = np.cumsum(S ** 2) / (S ** 2).sum()
    k95 = int(np.searchsorted(e_cum, 0.95) + 1)
    k99 = int(np.searchsorted(e_cum, 0.99) + 1)
    n_rom_req = {"k99": k99, "k95": k95}.get(N_ROM, N_ROM)
    n_rom0 = int(min(int(n_rom_req), r_full))
    print(f"[{STRATEGY}]  POD rank {r_full}; N_ROM = {N_ROM} -> {n_rom0} modes hold "
          f"{100*e_cum[n_rom0-1]:.2f}% of the training energy (k95 = {k95}, k99 = {k99})")

    U = (Xc.T @ Wp[:, :n_rom0]).astype(np.float64)
    Pu = np.empty((n_rom0 + 1, H, W)); Pv = np.empty((n_rom0 + 1, H, W))
    mean_f = x_mean.astype(np.float64).reshape(C, H, W)
    Pu[0], Pv[0] = mean_f[0], mean_f[1]
    modes = (U / kappa).T.reshape(n_rom0, C, H, W)
    Pu[1:], Pv[1:] = modes[:, 0], modes[:, 1]
    del U, modes
    A_phys = np.asarray(A[:, :n_rom0], dtype=np.float64) * kappa

    print(f"[{STRATEGY}]  projecting NS operators onto {n_rom0} modes ...", flush=True)
    l_full, q_full = galerkin_operators_ns(Pu, Pv, dx, dy, re_)
    del Pu, Pv

    rng = np.random.default_rng(SPLIT_SEED)
    # back-off ladder: the requested size, then the 95 % basis, then a few small models
    ladder = [n_rom0] + sorted({n for n in (k95, 60, 40, 30, 20, 15, 12, 10, 6, 4)
                                if n < n_rom0}, reverse=True)
    for n_rom in ladder:
        l = l_full[:n_rom, :n_rom + 1]
        q = q_full[:n_rom, :n_rom + 1, :n_rom + 1]
        Ar = A_phys[:, :n_rom]
        R2 = derivative_R2(l, q, Ar, tr_idx, dt)
        print(f"\n[{STRATEGY}]  ns_ode: {n_rom} modes ({100*e_cum[n_rom-1]:.2f}% energy)")
        print("  derivative R^2 per mode (projected, NOT fitted): "
              + "  ".join(f"{v:.2f}" for v in R2))
        s = Ar.std(axis=0); s = np.where(s < 1e-14, 1.0, s)
        B_real = Ar / s[None, :]
        step = make_step_ns(l, q, s, dt, SUBSTEPS)
        try:
            B_new, st = integrate_trajectories(step, s, B_real, n_aug, HORIZON,
                                               IC_NOISE, ENERGY_TOL, MAX_TRAJ, rng)
            if st["keep_frac"] < MIN_KEEP and n_rom != ladder[-1]:
                print(f"  !! only {100*st['keep_frac']:.1f}% of steps survived screening "
                      f"(< {100*MIN_KEEP:.0f}%) -> backing off to a smaller model")
                continue
            break
        except RuntimeError as err:
            print(f"  !! {err}")
            if n_rom == ladder[-1]:
                raise
            print("  -> backing off to a smaller model")

    # ---- quick statistical check of the synthetic coefficients ----
    ratio = B_new.std(axis=0)                      # real std is 1 by construction
    print(f"\n[{STRATEGY}]  kept {n_rom} modes; {st['n_traj']} trajectories, "
          f"{100*st['keep_frac']:.1f}% of integrated steps kept "
          f"(energy band tol {ENERGY_TOL})")
    print(f"  synthetic/real coefficient std per mode: "
          + "  ".join(f"{v:.2f}" for v in ratio))
    print(f"  synthetic/real mean |coef| offset (in std units): "
          f"{np.abs(B_new.mean(axis=0)).max():.2f} (max over modes)")

    if DRY_RUN:
        print(f"\n[{STRATEGY}]  DRY RUN — no bundle written   ({time.time()-t0:.0f} s)")
        print(f"[{STRATEGY}]  (would write {os.path.join(_AUG_DIR, os.path.splitext(os.path.basename(DATA_FILE))[0] + '_aug_' + STRATEGY)}.npz)")
        return

    A_new = (B_new * s[None, :]) / kappa
    train_aug = reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new, shape)
    del Xc

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bundle = {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object), "strategy": STRATEGY,
        "tr_idx": tr_idx, "val_idx": val_idx, "pool_idx": pool_idx,
        "split": SPLIT, "split_seed": SPLIT_SEED, "draw_seed": DRAW_SEED,
        "t_max": T_MAX, "n_real": N_REAL, "n_aug": n_aug, "model": "ns_ode",
        "n_modes": r_full, "n_rom": n_rom, "n_rom_requested": str(N_ROM),
        "k95": k95, "k99": k99, "rom_energy": float(e_cum[n_rom - 1]), "dt": dt, "re": re_, "dx": dx, "dy": dy,
        "horizon": HORIZON, "substeps": SUBSTEPS, "ic_noise": IC_NOISE,
        "energy_tol": ENERGY_TOL, "fit_R2": R2, "n_traj": st["n_traj"],
        "keep_frac": st["keep_frac"],
    }
    out = save_bundle(os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}"), bundle)
    print(f"\n[{STRATEGY}]  real train = {N_REAL}   real val = {n_val}   synthetic = {n_aug}")
    print(f"[{STRATEGY}]  saved -> {out}   ({time.time()-t0:.0f} s)")
    print(f"[{STRATEGY}]  next: train_augmented_vae.py with AUG_FILE = that file, "
          f"LATENT_DIM = 5; USE_AUG = True, then False for the 250-real baseline\n")


if __name__ == "__main__":
    main()

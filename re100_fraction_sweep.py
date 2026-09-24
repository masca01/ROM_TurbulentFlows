"""
Reproduce the SUMMER Re100 augmentation result and sweep the synthetic fraction.

Summer setup (Week 4 July report; bundles Data2PlatesGap1Re100_aug_galerkin.npz
and _aug_galerkin_ns.npz; checkpoints model_augVAE_..._lat5_b5e-03_ep500.pt):
    data        Data2PlatesGap1Re100.mat, 1000 snapshots, dt = 1, dx = dy = 0.08
    split       RANDOM 10 % validation with seed 7 -> 100 validation, 900 training
    generators  "galerkin"    data-identified quadratic model, 30 modes, ODE fit
                              (central differences, ridge 1e-6)     -> Ek 50.37 %
                "galerkin_ns" NS-projected model, 20 modes           -> Ek 49.44 %
                both: horizon 20, 20 RK4 sub-steps, IC kick 0.05, energy band 5 %,
                900 synthetic snapshots (1:1)
    baseline    900 real only                                        -> Ek 43.77 %
    network     latent 5, beta 5e-3, batch 32, 500 epochs, lr 3e-4, torch seed 7,
                normalisation + mean field from the real training snapshots,
                best-validation checkpoint, Ek on the real validation set

This script rebuilds exactly that (same seed, same split, same model, same rng
sequence, so the first 900 synthetic states are the summer ones) but generates
a LARGER synthetic pool once, and then trains the same network for several
synthetic FRACTIONS  f = n_aug / (n_real + n_aug), using the first n_aug
synthetic snapshots of the pool (nested sets):
    f = 0      ->    0 synthetic   (the baseline)
    f = 0.25   ->  300
    f = 0.5    ->  900             (the summer configuration)
    f = 0.667  -> 1800
    f = 0.75   -> 2700
One CSV row per fraction: ../../convergence/re100_fraction_sweep.csv

Usage:
    python3 re100_fraction_sweep.py                       # galerkin, latent 5, all fractions
    python3 re100_fraction_sweep.py galerkin_ns
    python3 re100_fraction_sweep.py galerkin --latent 15  # the convergence-study latent
    python3 re100_fraction_sweep.py --fractions 0,0.5     # subset of fractions
    python3 re100_fraction_sweep.py --dry-run             # generate + check vs the summer
                                                          # bundle, no training
Budget: ~3 min generation, then 500-epoch trainings on 900 / 1200 / 1800 / 2700 /
3600 snapshots (about 5 + 25 + 40 + 60 + 80 min on the MPS GPU); ~24 GB RAM peak.
"""
import os, sys, csv, time, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pod_augment_galerkin as pg              # load_data, compute_pod, reconstruct, fit, step, integrate
import pod_augment_galerkin_ns as pns          # galerkin_operators_ns, make_step_ns, grid_spacing, derivative_R2
import beta_vae as bv

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG  = os.path.join(_DATA, "AUGMENTED")
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))

# ============ CONFIG (summer values; do not change to reproduce) ============
DATA_FILE  = os.path.join(_DATA, "2PlatesGap", "Data2PlatesGap1Re100.mat")
GENERATOR  = "galerkin"           # "galerkin" (data regression, +6.6 pts) | "galerkin_ns" (NS equations, +5.7)
RNG_SEED   = 7                 # split AND trajectory rng (one generator, as in summer)
VAL_FRAC   = 0.10
FRACTIONS  = [0.0, 0.25, 0.4, 0.45, 0.5, 0.55, 0.6]     # synthetic fraction of the training set

RE, DT     = 100, 1.0
N_ROM      = {"galerkin": 30, "galerkin_ns": 25}
RIDGE      = 1e-6              # galerkin only
HORIZON    = 20
SUBSTEPS   = 20
IC_NOISE   = 0.05
ENERGY_TOL = 0.05
MAX_TRAJ   = 20000

LATENT     = 5                 # summer network (the convergence study used 15 for Re100)
BETA       = 5e-3
BATCH      = 32
EPOCHS     = 500
LR         = 3e-4
TORCH_SEED = 7
PROG_EVERY = 50

VERIFY_SUMMER = True           # compare the first synthetic states with the summer bundle
SAVE_POOL     = False          # also save the synthetic pool as a bundle (~12 GB)
CSV_OUT       = os.path.join(_CONV, "re100_fraction_sweep.csv")
# ============================================================================

DEVICE = bv.DEVICE


def _cli(argv):
    global GENERATOR, LATENT, FRACTIONS
    dry = False
    it = iter(argv)
    for a in it:
        if a in ("galerkin", "galerkin_ns"): GENERATOR = a
        elif a == "--latent":    LATENT = int(next(it))
        elif a == "--fractions": FRACTIONS = [float(x) for x in next(it).split(",")]
        elif a == "--dry-run":   dry = True
        else: raise SystemExit(f"unknown argument {a!r}")
    return dry


def n_aug_for(f, n_real):
    return 0 if f <= 0 else int(round(f / (1.0 - f) * n_real))


# ------------------------------- generation -------------------------------

def generate_pool(train_real, tr_idx, n_aug_max, rng):
    """Synthetic coefficient states with the summer generator settings; returns
    (train_aug [n_aug_max, C, H, W], info)."""
    x_mean, Xc, Wp, S, A, shape = pg.compute_pod(train_real)
    C, H, W = shape
    n_rom = min(N_ROM[GENERATOR], A.shape[1])
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
    train_aug = pg.reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new, shape)
    del Xc
    return train_aug, {"n_rom": n_rom, "rom_energy": e_rom, "n_traj": st["n_traj"],
                       "keep_frac": st["keep_frac"], "R2_min": float(R2.min())}


def verify_against_summer(train_aug, n_check=5):
    f = os.path.join(_AUG, f"Data2PlatesGap1Re100_aug_{GENERATOR}.npz")
    if not os.path.exists(f):
        print(f"[verify]  no summer bundle {os.path.basename(f)} — skipped"); return None
    S = np.load(f, allow_pickle=True)
    old = np.asarray(S["train_aug"][:n_check], dtype=np.float32)
    new = train_aug[:n_check]
    rel = float(np.abs(old - new).max() / (np.abs(old).max() + 1e-12))
    same = rel < 1e-4
    print(f"[verify]  first {n_check} synthetic snapshots vs summer bundle: max rel. diff "
          f"{rel:.2e} -> {'IDENTICAL (summer states reproduced)' if same else 'DIFFERENT'}")
    return same


# -------------------------------- training --------------------------------

def train(train_real, train_aug_n, val_real, tag):
    """Summer recipe: normalisation + mean field from the real snapshots only,
    best-validation checkpoint, Ek and det(R) on the real validation set."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
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
    enc = bv.Encoder(H, W, C, LATENT).to(DEVICE); dec = bv.Decoder(H, W, C, LATENT).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    val_every = max(1, round(EPOCHS / 10))
    best_val, best_ek, best_detR = float("inf"), float("nan"), float("nan")
    for epoch in range(1, EPOCHS + 1):
        enc.train(); dec.train(); run = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, _, _ = bv.vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item()
        run /= max(1, len(tr_loader))
        if epoch % val_every == 0 or epoch == EPOCHS:
            vl, _, _ = bv.val_loss(enc, dec, va_loader, BETA, DEVICE)
            if vl < best_val:
                _, ek = bv.compute_ek(enc, dec, val_n, val_fluc, mu_C, std_C, phys_mean, DEVICE)
                detR = bv.compute_det_R(enc, val_n, LATENT, DEVICE)
                best_val, best_ek, best_detR = vl, ek, detR
        if epoch % PROG_EVERY == 0 or epoch == EPOCHS:
            print(f"      [{tag}] epoch {epoch:4d}/{EPOCHS}  train_loss={run:.4e}  "
                  f"best_Ek={best_ek:5.2f}%  detR={best_detR:.3f}  [{time.time()-t0:6.0f}s]", flush=True)
    del X, tr_loader
    return best_ek, best_detR, best_val, time.time() - t0


FIELDS = ["date", "generator", "fraction", "n_real", "n_aug", "n_train", "n_val", "n_rom",
          "rom_energy_pct", "n_traj", "keep_frac", "latent", "beta", "epochs", "seed",
          "Ek", "e", "detR", "best_val_loss", "gain_vs_f0", "summer_ref_Ek", "wall_s"]


def append_row(row):
    os.makedirs(_CONV, exist_ok=True)
    rows = list(csv.DictReader(open(CSV_OUT, newline=""))) if os.path.exists(CSV_OUT) else []
    rows.append(row)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in FIELDS})


def main():
    dry = _cli(sys.argv[1:])
    T0 = time.time()
    data, _ = pg.load_data(DATA_FILE)
    Nt = len(data)
    rng = np.random.default_rng(RNG_SEED)             # same rng for split AND trajectories
    idx = rng.permutation(Nt)
    n_val = max(1, round(VAL_FRAC * Nt))
    val_idx, tr_idx = np.sort(idx[:n_val]), np.sort(idx[n_val:])
    train_real, val_real = data[tr_idx].copy(), data[val_idx].copy()
    del data
    n_real = len(tr_idx)
    fr = sorted(set(FRACTIONS))
    n_augs = [n_aug_for(f, n_real) for f in fr]
    n_max = max(n_augs)
    print(f"[sweep]  generator {GENERATOR}, latent {LATENT}: {n_real} real, {n_val} validation "
          f"(random, seed {RNG_SEED}); fractions {fr} -> synthetic {n_augs}", flush=True)

    train_aug = None
    info = {"n_rom": "", "rom_energy": float("nan"), "n_traj": "", "keep_frac": float("nan")}
    if n_max > 0:
        train_aug, info = generate_pool(train_real, tr_idx, n_max, rng)
        if VERIFY_SUMMER: verify_against_summer(train_aug)
        if SAVE_POOL:
            out = pg.save_bundle(os.path.join(_AUG, f"Data2PlatesGap1Re100_aug_{GENERATOR}_pool{n_max}"),
                                 {"train_real": train_real, "train_aug": train_aug, "val_real": val_real,
                                  "comp_names": np.array(["u", "v"], dtype=object), "strategy": f"{GENERATOR}_pool",
                                  "tr_idx": tr_idx, "val_idx": val_idx, **{k: v for k, v in info.items()}})
            print(f"[sweep]  pool saved -> {out}")
    if dry:
        print(f"[sweep]  DRY RUN — no training  ({time.time()-T0:.0f} s)"); return

    summer = {"galerkin": {0.0: 43.77, 0.5: 50.37}, "galerkin_ns": {0.0: 43.77, 0.5: 49.44}}[GENERATOR]
    ek0 = None
    for f, n_aug in zip(fr, n_augs):
        print(f"\n[sweep]  ==== fraction {f:.3f}: {n_real} real + {n_aug} synthetic = {n_real+n_aug} ====", flush=True)
        aug_n = train_aug[:n_aug] if n_aug else np.empty((0,) + train_real.shape[1:], np.float32)
        ek, detR, vl, dt = train(train_real, aug_n, val_real, tag=f"f={f:.2f} n={n_real+n_aug}")
        if n_aug == 0: ek0 = ek
        row = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "generator": GENERATOR,
               "fraction": round(n_aug / (n_real + n_aug), 4), "n_real": n_real, "n_aug": n_aug,
               "n_train": n_real + n_aug, "n_val": n_val, "n_rom": info["n_rom"] if n_aug else "",
               "rom_energy_pct": round(100 * info["rom_energy"], 3) if n_aug else "",
               "n_traj": info["n_traj"] if n_aug else "", "keep_frac": round(info["keep_frac"], 4) if n_aug else "",
               "latent": LATENT, "beta": BETA, "epochs": EPOCHS, "seed": TORCH_SEED,
               "Ek": round(float(ek), 4), "e": round(1 - float(ek) / 100, 6), "detR": round(float(detR), 6),
               "best_val_loss": round(float(vl), 6),
               "gain_vs_f0": round(float(ek) - ek0, 3) if ek0 is not None else "",
               "summer_ref_Ek": summer.get(round(f, 3), ""), "wall_s": round(dt)}
        append_row(row)
        print(f"[sweep]  fraction {f:.3f}: Ek = {ek:.2f}%  detR = {detR:.3f}"
              + (f"   (summer: {summer[round(f,3)]}%)" if round(f, 3) in summer else "")
              + (f"   gain vs real-only {row['gain_vs_f0']:+} pts" if ek0 is not None and n_aug else ""), flush=True)
    print(f"\n[sweep]  done in {(time.time()-T0)/60:.0f} min  ->  {CSV_OUT}")


if __name__ == "__main__":
    main()

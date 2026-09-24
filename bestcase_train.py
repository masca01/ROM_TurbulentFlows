"""
Train the convergence-study beta-VAE on a best-case augmentation bundle and
report the result in ONE CSV row next to its reference numbers.

Two trainings per bundle (same seed, same recipe as nn_convergence.py /
nn_latent_test.py: 500 epochs, beta 5e-3, batch 32, lr 3e-4, best-val checkpoint,
Ek on the REAL validation set only; normalisation + mean field from the REAL
training snapshots only):
    augmented :  train_real + train_aug
    baseline  :  train_real only   (skip with --no-baseline)

The seed is the one nn_convergence.py used for the FIRST draw at n = n_real, so
the baseline should reproduce that sweep point (up to GPU non-determinism), and
the reference columns are read from the convergence .mat bundles of the same
split so every row is self-contained:
    NN_conv_Ek_nreal_draw1 / _mean   NN Ek at n_real in the convergence sweep
    NN_conv_Ek_full                  NN Ek with the whole pool (the ceiling)
    POD99_e_nreal / POD99_e_full     POD99 error at n_real / whole pool
    Ek_max_POD99_nreal / _full       = 100 (1 - POD99 error): the energy a model living
                                     in the 99 %-energy POD subspace of n_real / of the
                                     whole pool can reconstruct at most on the validation
                                     set.  The synthetic snapshots live in that subspace
                                     (ROM = k99 modes of the subset), so this is the
                                     ceiling the synthetic data can push the NN towards.
    gain_pts   = Ek_aug - Ek_baseline
    gap_closed = (Ek_aug - Ek_baseline) / (NN_conv_Ek_full - Ek_baseline)

Usage:  python3 bestcase_train.py <bundle.npz|.mat> <LATENT_DIM> [--no-baseline]
                                  [--epochs N] [--csv path]
Output: one row appended to ../../convergence/bestcase_augmentation_results.csv
"""
import os, re, sys, csv, time, datetime
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn_latent_test as nlt          # train_full_pool(): identical recipe, latent as argument
import convergence_split as cs

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))
CSV_DEFAULT = os.path.join(_CONV, "bestcase_augmentation_results.csv")

FIELDS = ["date", "dataset", "Re", "split", "strategy", "n_real", "n_aug", "n_train", "n_val",
          "n_rom", "rom_energy_pct", "keep_frac", "n_traj", "latent", "epochs", "seed",
          "Ek_baseline", "e_baseline", "Ek_aug", "e_aug", "gain_pts", "gap_closed",
          "NN_conv_Ek_nreal_draw1", "NN_conv_Ek_nreal_mean", "NN_conv_Ek_full", "NN_conv_n_full",
          "POD99_e_nreal", "POD99_e_full", "POD99_modes",
          "Ek_max_POD99_nreal", "Ek_max_POD99_full",
          "wall_s_baseline", "wall_s_aug", "bundle"]

_N_HEAD, _TAIL_STEP, _TAIL_MAX, _M_REPS = [5, 10, 15, 20, 30, 50, 100, 150, 250, 400], 250, 1500, 3


def _args(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    bundle, latent = argv[0], int(argv[1])
    opts = {"baseline": True, "epochs": None, "csv": CSV_DEFAULT}
    it = iter(argv[2:])
    for a in it:
        if a == "--no-baseline": opts["baseline"] = False
        elif a == "--epochs":    opts["epochs"] = int(next(it))
        elif a == "--csv":       opts["csv"] = next(it)
        else: raise SystemExit(f"unknown argument {a!r}")
    return bundle, latent, opts


def load_bundle(path):
    S = np.load(path, allow_pickle=True) if path.endswith(".npz") else sio.loadmat(path, simplify_cells=True)
    def arr(k, default=None):
        if k not in S: return default
        v = np.asarray(S[k]); return v
    def scal(k, default=None):
        v = arr(k)
        if v is None: return default
        v = np.ravel(v)
        return v[0] if v.size else default
    out = {"train_real": arr("train_real").astype(np.float32),
           "train_aug":  arr("train_aug").astype(np.float32),
           "val_real":   arr("val_real").astype(np.float32),
           "pool_idx":   arr("pool_idx"),
           "strategy": str(scal("strategy", "unknown")), "split": str(scal("split", "random")),
           "n_real": int(scal("n_real", 0)), "n_aug": int(scal("n_aug", 0)),
           "n_rom": int(scal("n_rom", 0)), "rom_energy": float(scal("rom_energy", np.nan)),
           "keep_frac": float(scal("keep_frac", np.nan)), "n_traj": int(scal("n_traj", 0))}
    for k in ("train_real", "train_aug", "val_real"):
        if out[k].ndim == 3: out[k] = out[k][None]
    return out


def conv_seed(N_pool, n_real):
    """Seed nn_convergence.py used for its first draw at n = n_real."""
    ceiling = min(_TAIL_MAX, N_pool)
    head = [v for v in _N_HEAD if v <= ceiling]
    start = (head[-1] if head else 0) + _TAIL_STEP
    n_vals = sorted(set(head + list(range(start, ceiling + 1, _TAIL_STEP)) + [ceiling]))
    i = n_vals.index(n_real) if n_real in n_vals else len(n_vals) - 1
    return nlt.DRAW_SEED + 1000 * i


def references(base, split, n_real):
    """Reference numbers from the convergence .mat bundles of this split (blank if absent)."""
    ref = {}
    d = cs.aug_dir(split)
    f = os.path.join(d, f"{base}_nn_convergence.mat")
    if os.path.exists(f):
        S = sio.loadmat(f, simplify_cells=True)
        n = np.atleast_1d(S["n_vals"]); ek = np.atleast_1d(S["ek_mean"]); ea = np.atleast_2d(S["e_all"])
        if n_real in n:
            i = int(np.where(n == n_real)[0][0])
            ref["NN_conv_Ek_nreal_draw1"] = round(100 * (1 - float(ea[i, 0])), 3)
            ref["NN_conv_Ek_nreal_mean"] = round(float(ek[i]), 3)
        ref["NN_conv_Ek_full"] = round(float(ek[-1]), 3); ref["NN_conv_n_full"] = int(n[-1])
    f = os.path.join(d, f"{base}_pod99_convergence.mat")
    if os.path.exists(f):
        S = sio.loadmat(f, simplify_cells=True)
        n = np.atleast_1d(S["n_vals"]); e = np.atleast_1d(S["e_mean"])
        if n_real in n:
            ref["POD99_e_nreal"] = round(float(e[np.where(n == n_real)[0][0]]), 6)
            ref["Ek_max_POD99_nreal"] = round(100 * (1 - ref["POD99_e_nreal"]), 3)
        ref["POD99_e_full"] = round(float(e[-1]), 6); ref["POD99_modes"] = int(np.ravel(S["n_modes"])[0])
        ref["Ek_max_POD99_full"] = round(100 * (1 - ref["POD99_e_full"]), 3)
    return ref


def append_row(path, row):
    """Append the row; if the existing CSV has an older header, rewrite it with the
    current columns (old rows keep blanks in the new ones)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def main():
    bundle_path, latent, opts = _args(sys.argv[1:])
    if opts["epochs"]:
        nlt.N_EPOCHS = opts["epochs"]
    B = load_bundle(bundle_path)
    base = os.path.basename(bundle_path).split("_aug_")[0]
    m = re.search(r"Re(\d+)", base)
    Re = int(m.group(1)) if m else ""
    C, H, W = B["train_real"].shape[1:]
    N_pool = len(B["pool_idx"]) if B["pool_idx"] is not None else B["n_real"] + B["n_aug"]
    seed = conv_seed(N_pool, B["n_real"])
    ref = references(base, B["split"], B["n_real"])
    print(f"[bestcase_train]  {base}  Re {Re}  split {B['split']}  strategy {B['strategy']}")
    print(f"[bestcase_train]  real {B['n_real']} + synthetic {B['n_aug']} (ROM {B['n_rom']} modes, "
          f"{100*B['rom_energy']:.2f}% energy, {100*B['keep_frac']:.1f}% kept)   val {len(B['val_real'])}"
          f"   latent {latent}   {nlt.N_EPOCHS} epochs   seed {seed}   device {nlt.DEVICE}")
    if ref:
        print(f"[bestcase_train]  references: NN@n_real draw1 {ref.get('NN_conv_Ek_nreal_draw1', '—')}%  "
              f"mean {ref.get('NN_conv_Ek_nreal_mean', '—')}%   NN full {ref.get('NN_conv_Ek_full', '—')}%   "
              f"POD99@n_real e {ref.get('POD99_e_nreal', '—')}   POD99 full e {ref.get('POD99_e_full', '—')}")

    row = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "dataset": base, "Re": Re,
           "split": B["split"], "strategy": B["strategy"], "n_real": B["n_real"], "n_aug": B["n_aug"],
           "n_train": B["n_real"] + B["n_aug"], "n_val": len(B["val_real"]), "n_rom": B["n_rom"],
           "rom_energy_pct": round(100 * B["rom_energy"], 3), "keep_frac": round(B["keep_frac"], 4),
           "n_traj": B["n_traj"], "latent": latent, "epochs": nlt.N_EPOCHS, "seed": seed,
           "bundle": os.path.basename(bundle_path), **ref}

    if opts["baseline"]:
        print(f"\n[bestcase_train]  ---- BASELINE: {B['n_real']} real snapshots ----", flush=True)
        ek, err, _, dt = nlt.train_full_pool(B["train_real"], B["val_real"], C, H, W, latent, seed,
                                             tag=f"Re{Re} baseline n={B['n_real']}")
        row.update(Ek_baseline=round(float(ek), 4), e_baseline=round(float(err), 6), wall_s_baseline=round(dt))
        print(f"[bestcase_train]  baseline  Ek = {ek:.2f}%   e = {err:.4e}   [{dt:.0f} s]", flush=True)

    print(f"\n[bestcase_train]  ---- AUGMENTED: {B['n_real']} real + {B['n_aug']} synthetic ----", flush=True)
    train = np.concatenate([B["train_real"], B["train_aug"]], axis=0)
    del B["train_aug"]
    # normalisation + mean field from the REAL snapshots only (as in train_augmented_vae.py)
    ek, err, _, dt = _train_aug(train, B["train_real"], B["val_real"], C, H, W, latent, seed,
                                tag=f"Re{Re} aug n={len(train)}")
    row.update(Ek_aug=round(float(ek), 4), e_aug=round(float(err), 6), wall_s_aug=round(dt))
    print(f"[bestcase_train]  augmented Ek = {ek:.2f}%   e = {err:.4e}   [{dt:.0f} s]", flush=True)

    base_ek = row.get("Ek_baseline", ref.get("NN_conv_Ek_nreal_draw1"))
    full_ek = ref.get("NN_conv_Ek_full")
    if base_ek is not None:
        row["gain_pts"] = round(float(ek) - float(base_ek), 3)
        if full_ek is not None and float(full_ek) != float(base_ek):
            row["gap_closed"] = round((float(ek) - float(base_ek)) / (float(full_ek) - float(base_ek)), 3)
    append_row(opts["csv"], row)
    print(f"\n[bestcase_train]  {base}: baseline {row.get('Ek_baseline', base_ek)}%  ->  augmented "
          f"{row['Ek_aug']}%   (gain {row.get('gain_pts', '—')} pts, gap to {full_ek}% closed: "
          f"{row.get('gap_closed', '—')})")
    print(f"[bestcase_train]  appended -> {opts['csv']}\n")


def _train_aug(train_all, train_real, val_real, C, H, W, latent, seed, tag):
    """Same as nlt.train_full_pool, but normalisation and the physical mean come
    from the REAL training snapshots only, never from the synthetic ones."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    bv = nlt.bv
    t0 = time.time()
    mu_C  = train_real.mean(axis=(0, 2, 3)).astype(np.float32)
    std_C = train_real.std(axis=(0, 2, 3)); std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    phys_mean = train_real.mean(axis=0).astype(np.float32)
    norm = lambda x: (x - mu_C[None, :, None, None]) / std_C[None, :, None, None]
    train_n, val_n = norm(train_all), norm(val_real)
    val_fluc = val_real - phys_mean[None]
    tr_loader = DataLoader(TensorDataset(torch.tensor(train_n, dtype=torch.float32)),
                           batch_size=nlt.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(val_n, dtype=torch.float32)),
                           batch_size=nlt.BATCH_SIZE, shuffle=False)
    torch.manual_seed(seed)
    enc = bv.Encoder(H, W, C, latent).to(nlt.DEVICE); dec = bv.Decoder(H, W, C, latent).to(nlt.DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=nlt.LR)
    val_every = max(1, round(nlt.N_EPOCHS / 10))
    best_val, best_ek = float("inf"), float("nan")
    for epoch in range(1, nlt.N_EPOCHS + 1):
        enc.train(); dec.train(); run_loss = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(nlt.DEVICE)
            loss, _, _ = bv.vae_loss(enc, dec, xb, nlt.BETA)
            opt.zero_grad(); loss.backward(); opt.step(); run_loss += loss.item()
        run_loss /= max(1, len(tr_loader))
        if (epoch % val_every == 0) or (epoch == nlt.N_EPOCHS):
            vl, _, _ = bv.val_loss(enc, dec, va_loader, nlt.BETA, nlt.DEVICE)
            if vl < best_val:
                _, ek_all = bv.compute_ek(enc, dec, val_n, val_fluc, mu_C, std_C, phys_mean, nlt.DEVICE)
                best_val, best_ek = vl, ek_all
        if (epoch % nlt.PROG_EVERY == 0) or (epoch == nlt.N_EPOCHS):
            print(f"      [{tag}] epoch {epoch:4d}/{nlt.N_EPOCHS}  train_loss={run_loss:.4e}"
                  f"  best_Ek={best_ek:5.2f}%  [{time.time()-t0:6.1f}s]", flush=True)
    return best_ek, 1.0 - best_ek / 100.0, best_val, time.time() - t0


if __name__ == "__main__":
    main()

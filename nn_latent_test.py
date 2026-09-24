"""
beta-VAE LATENT-DIMENSION test at the LAST convergence point only.

Trains ONE beta-VAE per (dataset, latent dimension) on the WHOLE training pool
(1350 snapshots for the Alpha0 Re50/60/70/80 sets, 900 for the Re100 set) and
reports the energy retained Ek (%) on the held-out validation set.  Everything
else (split, cap, architecture, beta, batch, epochs, lr, seeds, Ek metric) is
IDENTICAL to nn_convergence.py, so a run with the same latent dimension
reproduces the last sweep point of the convergence study, and a run with a new
latent dimension tells you what that point would become.  It is the quick way
to pick the latent dimension per Re before re-running the full sweep.

Configure LATENT below (one or several latent dims per dataset), or override
from the command line:
    python3 nn_latent_test.py                      # tail5 split, LATENT dict
    python3 nn_latent_test.py tail Re60=10,14 Re70=12,16,20
    python3 nn_latent_test.py random Re100=20      # random split instead
    python3 nn_latent_test.py tail all=8,12        # same dims for every dataset

Output: one line per training appended to
        ../../convergence/nn_latent_test[_tailval].csv
(dataset, Re, latent, n_train, n_val, Ek, e = 1 - Ek/100, epochs, wall time, date)
and a summary table at the end with the POD 95% / 99% errors of the same split
next to the NN values, for comparison.  Nothing in the convergence folders,
CSVs or Excel of the convergence study is modified.
"""
import os, re, sys, csv, time, datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta_vae as bv
import convergence_split as cs

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))

# ============ CONFIG ============
SPLIT = "tail5"         # default split (first CLI arg overrides): "tail5" | "tail2.5" | "tail" | "random"
CAP   = 1500            # common snapshot budget, same as run_alpha0_convergence.py

# Latent dimension(s) to test per dataset  <- THE PARAMETER TO PLAY WITH.
# An int or a list of ints; every value is trained once on the full pool.
# (values used by the convergence study: Re50=5, Re60=7, Re70=9, Re80=11, Re100=15)
LATENT = {
    50:  [6],
    60:  [12],
    70:  [17],
    80:  [12],
    100: [25],
}
DATASETS = [                     # (Re, relative data file)
    (50,  "Alpha0/dataRe50Alpha0_2.mat"),
    (60,  "Alpha0/dataRe60Alpha0_2.mat"),
    (70,  "Alpha0/dataRe70Alpha0_2.mat"),
    (80,  "Alpha0/dataRe80Alpha0_2.mat"),
    (100, "2PlatesGap/Data2PlatesGap1Re100.mat"),
]

# ---- everything below MUST match nn_convergence.py ----
COMP_IDX   = None
T_STRIDE   = 1
N_HEAD     = [5, 10, 15, 20, 30, 50, 100, 150, 250, 400]
TAIL_STEP  = 250
TAIL_MAX   = 1500
BETA       = 5e-3
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4
SPLIT_SEED = 7
DRAW_SEED  = 11
PROG_EVERY = 50
# ================================

DEVICE = bv.DEVICE


def _parse_cli(argv):
    """[split] [Re60=10,14 ...] [all=8,12]  ->  (split, {Re: [dims]})"""
    split = SPLIT
    latent = {re_: (list(v) if isinstance(v, (list, tuple)) else [v])
              for re_, v in LATENT.items()}
    for a in argv:
        if a in cs.MODES:
            split = a
            continue
        m = re.fullmatch(r"(?i)(all|re(\d+))=([\d,]+)", a)
        if not m:
            raise SystemExit(f"bad argument {a!r}: use 'tail5'/'tail2.5'/'tail'/'random', 'Re60=10,14' "
                             f"or 'all=8,12'")
        dims = [int(x) for x in m.group(3).split(",") if x]
        if m.group(1).lower() == "all":
            for re_ in latent:
                latent[re_] = list(dims)
        else:
            latent[int(m.group(2))] = dims
    return cs.check_mode(split), latent


def _conv_seed(N_pool):
    """Seed nn_convergence.py used for its LAST sweep point (full pool), so the
    same latent dimension reproduces that point (up to GPU non-determinism)."""
    ceiling = min(TAIL_MAX, N_pool)
    head = [v for v in N_HEAD if v <= ceiling]
    start = (head[-1] if head else 0) + TAIL_STEP
    n_vals = sorted(set(head + list(range(start, ceiling + 1, TAIL_STEP)) + [ceiling]))
    i = len(n_vals) - 1                       # index of the full-pool point
    return DRAW_SEED + 1000 * i               # draw j = 0


def train_full_pool(train_data, val_real, C, H, W, latent_dim, seed, tag):
    """Copy of nn_convergence.train_one_vae with the latent dimension as an
    argument.  Returns (Ek %, e = 1 - Ek/100, best val loss, wall time)."""
    t0 = time.time()
    mu_C  = train_data.mean(axis=(0, 2, 3)).astype(np.float32)
    std_C = train_data.std(axis=(0, 2, 3))
    std_C = np.where(std_C < 1e-12, 1.0, std_C).astype(np.float32)
    phys_mean = train_data.mean(axis=0).astype(np.float32)

    def norm(x):
        return (x - mu_C[None, :, None, None]) / std_C[None, :, None, None]

    train_n  = norm(train_data)
    val_n    = norm(val_real)
    val_fluc = val_real - phys_mean[None]

    X_tr = torch.tensor(train_n, dtype=torch.float32)
    X_va = torch.tensor(val_n,   dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(seed)
    enc = bv.Encoder(H, W, C, latent_dim).to(DEVICE)
    dec = bv.Decoder(H, W, C, latent_dim).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    val_every = max(1, round(N_EPOCHS / 10))
    best_val, best_ek = float("inf"), float("nan")
    for epoch in range(1, N_EPOCHS + 1):
        enc.train(); dec.train()
        run_loss = 0.0
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            loss, _, _ = bv.vae_loss(enc, dec, xb, BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss)
        run_loss /= max(1, len(tr_loader))
        if (epoch % val_every == 0) or (epoch == N_EPOCHS):
            vl, _, _ = bv.val_loss(enc, dec, va_loader, BETA, DEVICE)
            if vl < best_val:
                _, ek_all = bv.compute_ek(enc, dec, val_n, val_fluc,
                                          mu_C, std_C, phys_mean, DEVICE)
                best_val, best_ek = vl, ek_all
        if (epoch % PROG_EVERY == 0) or (epoch == N_EPOCHS):
            print(f"      [{tag}] epoch {epoch:4d}/{N_EPOCHS}  train_loss={run_loss:.4e}"
                  f"  best_Ek={best_ek:5.2f}%  [{time.time()-t0:6.1f}s]", flush=True)
    return best_ek, 1.0 - best_ek / 100.0, best_val, time.time() - t0


def _pod_reference(data_file, base, split):
    """POD95 / POD99 last-point errors of this split from the shared CSV (if any)."""
    p = cs.csv_path(data_file, split)
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("dataset") == base:
                return r
    return {}


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    split, latent = _parse_cli(sys.argv[1:])
    out_csv = os.path.join(_CONV, f"nn_latent_test{cs.suffix(split)}.csv")
    os.makedirs(_CONV, exist_ok=True)
    plan = {re_: latent.get(re_, []) for re_, _ in DATASETS}
    n_train = sum(len(v) for v in plan.values())
    print(f"######## nn_latent_test  split = {split} ({cs.describe(split)})   cap {CAP}"
          f"   {N_EPOCHS} epochs, beta={BETA}, lr={LR}   device = {DEVICE} ########")
    print("latent dims to test: " +
          ", ".join(f"Re{re_}: {v}" for re_, v in plan.items() if v))
    print(f"=> {n_train} full-pool trainings.   results -> {out_csv}\n", flush=True)

    results = []
    T0 = time.time()
    for re_, fn in DATASETS:
        dims = plan[re_]
        if not dims:
            continue
        path = os.path.join(_DATA, fn)
        base = os.path.splitext(os.path.basename(fn))[0]
        if not os.path.exists(path):
            print(f"!! missing {path} — skipping"); continue

        data, _ = bv.load_data(path, COMP_IDX, T_STRIDE, CAP)
        Nt, C, H, W = data.shape
        val_idx, pool_idx, n_val = cs.split_indices(Nt, None, split, SPLIT_SEED)
        N_pool = len(pool_idx)
        train_data = data[pool_idx].astype(np.float32)   # the WHOLE pool (no draw)
        val_real   = data[val_idx].astype(np.float32)
        seed = _conv_seed(N_pool)
        pod = _pod_reference(path, base, split)
        print(f"\n================  {base}  (Re {re_}): {Nt} snapshots -> train on ALL "
              f"{N_pool} pool snapshots, {n_val} validation ({cs.describe(split)}; "
              f"val idx {val_idx[0]}..{val_idx[-1]}), seed {seed}  ================",
              flush=True)

        for ld in dims:
            print(f"\n  ---- Re {re_}   latent = {ld}   ({N_EPOCHS} epochs on n = {N_pool}) ----",
                  flush=True)
            ek, err, vl, dt = train_full_pool(train_data, val_real, C, H, W, ld, seed,
                                              tag=f"Re{re_} latent {ld}")
            print(f"  Re {re_}  latent {ld:3d}  ->  Ek = {ek:6.2f}%   e = {err:.4e}   "
                  f"[{dt:.0f} s, total {time.time()-T0:.0f} s]", flush=True)
            row = {"dataset": base, "Re": re_, "split": split, "latent": ld,
                   "n_train": N_pool, "n_val": n_val, "Ek": round(float(ek), 4),
                   "e": float(err), "best_val_loss": float(vl),
                   "epochs": N_EPOCHS, "beta": BETA, "lr": LR, "batch": BATCH_SIZE,
                   "seed": seed, "wall_s": round(dt),
                   "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            _append_csv(out_csv, row)
            results.append((re_, ld, N_pool, ek, err, pod))
        del data, train_data, val_real

    # ---- summary ----
    print(f"\n\n==== SUMMARY  (split = {split}; POD columns = last convergence point "
          f"of the same split, from convergence_last_point{cs.suffix(split)}.csv) ====")
    print(f"{'Re':>4} {'latent':>7} {'n_train':>8} {'NN Ek %':>9} {'NN e':>10} "
          f"{'POD95 e':>10} {'POD99 e':>10}")
    for re_, ld, n_pool, ek, err, pod in results:
        p95 = pod.get("POD95_e_mean", ""); p99 = pod.get("POD99_e_mean", "")
        p95 = f"{float(p95):.4e}" if p95 else "—"
        p99 = f"{float(p99):.4e}" if p99 else "—"
        print(f"{re_:>4} {ld:>7} {n_pool:>8} {ek:>9.2f} {err:>10.4e} {p95:>10} {p99:>10}")
    print(f"\nappended {len(results)} row(s) -> {out_csv}")
    print(f"total wall time {time.time()-T0:.0f} s")


if __name__ == "__main__":
    main()

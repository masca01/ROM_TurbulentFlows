"""
MORE REAL SUBSETS FOR THE 25-SNAPSHOT CELL.

The scarce-data result on slide 12 rests on three real subsets, which is too few: at 15 modes the
mean gain is +4.1 with a 95% interval of +-11.6, so the cell cannot be called either way. This
script extends that cell to N_SUBSETS independent real subsets (default 8) for BOTH truncations,
so the comparison between the three sources of added data is made on the same footing.

Cell:      Re 50 (Alpha0), 25 real snapshots, latent 5, blocks of 10, synthetic fraction 0.8
Arms:      real only  |  POD-Galerkin synthetic  |  POD-coefficient jitter  |  extra real projected
Truncations: K = 10 and K = 15 (the two the screen proposed; the network is the same for both, so
             the real-only baseline is trained once per subset and reused)

Rows are appended to the same convergence/lowdata_results.csv with the same columns, so every
analysis and figure that reads that file picks them up. Finished rows are skipped, so the script
can be stopped and relaunched.

Usage:
    python3 run_lowdata_subsets.py                 # subsets 0..7, both truncations
    python3 run_lowdata_subsets.py --subsets 5     # subsets 0..4
    python3 run_lowdata_subsets.py --plan          # what is missing and the time estimate
    python3 run_lowdata_subsets.py --subsets 4 --epochs 2 --csv /tmp/smoke.csv   # smoke test
"""
import os, sys, csv, time, datetime, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_region_map as rm
import run_lowdata as L
import pod_augment_galerkin_ns as pns
import re100_fraction_sweep as rs

# ============ CONFIG ============
DATASET   = "Re50"
N_REAL    = 25
LATENT    = 5
SAMPLING  = "blocks"
FRACTION  = 0.8
TRUNCS    = ["K10", "K15"]
ARMS      = ["galerkin_ns", "jitter", "real_proj"]
N_SUBSETS = 8
# ================================
CSV_OUT = L.TRAIN_CSV


def _args(argv):
    o = {"plan": False, "subsets": N_SUBSETS, "epochs": rm.EPOCHS}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--csv":                       # smoke tests: write somewhere else
            global CSV_OUT
            CSV_OUT = os.path.abspath(next(it))
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


def done_sets():
    """(baselines already trained, augmented rows already present)."""
    base, aug = {}, set()
    if not os.path.exists(CSV_OUT):
        return base, aug
    for r in csv.DictReader(open(CSV_OUT, newline="")):
        if (r["dataset"], r["n_real"], r["sampling"]) != (DATASET, str(N_REAL), SAMPLING):
            continue
        if r["kind"] == "baseline":
            base[r["subset"]] = float(r["Ek"])
        elif r["kind"] == "aug":
            aug.add((r["subset"], r["trunc"], r["generator"], r["fraction"]))
    return base, aug


def main():
    o = _args(sys.argv[1:])
    rs.EPOCHS = o["epochs"]
    base_done, aug_done = done_sets()
    n_aug = rs.n_aug_for(FRACTION, N_REAL)

    todo_base = [s for s in range(o["subsets"]) if str(s) not in base_done]
    todo_aug = [(s, t, g) for s in range(o["subsets"]) for t in TRUNCS for g in ARMS
                if (str(s), t, g, str(round(FRACTION, 4))) not in aug_done]
    samples = len(todo_base) * N_REAL + len(todo_aug) * (N_REAL + n_aug)
    hours = samples * rm.SEC_PER_SAMPLE * o["epochs"] / 500 / 3600
    print(f"[sub]  {DATASET}, {N_REAL} real, latent {LATENT}, fraction {FRACTION} "
          f"({n_aug} added snapshots), truncations {', '.join(TRUNCS)}")
    print(f"[sub]  subsets 0..{o['subsets'] - 1}: {len(todo_base)} real-only + {len(todo_aug)} augmented "
          f"trainings left, about {hours:.1f} h  ->  {CSV_OUT}", flush=True)
    if o["plan"] or not (todo_base or todo_aug):
        for s, t, g in todo_aug:
            print(f"        subset {s}  {t:4s}  {g}")
        return

    T0 = time.time()
    data, re_, dt, path = rm.load(DATASET)
    Nt = len(data)
    val_idx, pool_idx, _ = rm.cs.split_indices(Nt, None, "random", rm.SPLIT_SEED)
    pool_set = set(int(i) for i in pool_idx)
    val_real = data[val_idx]
    steps = max(1, int(round(rm.GEN_TIME / dt)))
    rs.LATENT = LATENT

    for sub in range(o["subsets"]):
        rng = np.random.default_rng([rm.DRAW_SEED, int(re_), N_REAL, sub])
        tr_idx = L.draw_subset(SAMPLING, pool_set, pool_idx, Nt, N_REAL, rng)
        train_real = data[tr_idx]
        win = rm.fidelity_windows(set(int(i) for i in tr_idx), Nt, steps, rng)
        P = rm.subset_pod(train_real)
        windows = [data[t:t + steps + 1] for t in win]
        print(f"\n[sub]  ===== subset {sub}: {len(win)} quality windows =====", flush=True)

        if str(sub) not in base_done:
            ek, detR, _, wall = rs.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32),
                                         val_real, tag=f"{DATASET} n={N_REAL} s{sub} real only")
            base_done[str(sub)] = float(ek)
            L.append(CSV_OUT, L.B_FIELDS, dict(
                date=L.now(), dataset=DATASET, n_real=N_REAL, sampling=SAMPLING, subset=sub, kind="baseline",
                latent=LATENT, trunc="", K="", n_train=N_REAL, epochs=o["epochs"], Ek=round(float(ek), 4),
                detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
            print(f"[sub]  real-only Ek {ek:.2f}%", flush=True)
        base = base_done[str(sub)]

        for tname in TRUNCS:
            trunc = [t for t in L.TRUNCS if t[0] == tname][0]
            K = L.k_of(P, trunc, N_REAL)
            ceiling, win_A = rm.ceiling_and_project(P, K, val_real, windows)
            A = np.asarray(P["A"][:, :K], dtype=np.float64)
            info = dict(dataset=DATASET, n_real=N_REAL, sampling=SAMPLING, subset=sub, latent=LATENT,
                        trunc=tname, K=K, energy_K_pct=round(100 * P["e_cum"][K - 1], 3),
                        pod_ceiling_Ek=round(ceiling, 3))
            qerr = None
            if "galerkin_ns" in ARMS:
                l, q, kap = L.build_ops(P, K, re_, path)
                step, s, to_b, from_b = L.model_at(l, q, kap, A, K, dt)
                qerr = rm.fidelity(step, to_b, from_b, win_A, steps)[0] if win_A else None
                del l, q
                print(f"[sub]  {tname} (K={K}): ceiling {ceiling:.1f}%, ROM error "
                      f"{'-' if qerr is None else f'{qerr:.3f}'}", flush=True)

            for gen in ARMS:
                if (str(sub), tname, gen, str(round(FRACTION, 4))) in aug_done:
                    continue
                try:
                    if gen == "galerkin_ns":
                        B_new, _ = pns.integrate_trajectories(
                            step, s, to_b(A), n_aug, steps, rm.IC_NOISE, rm.ENERGY_TOL, rm.MAX_TRAJ,
                            np.random.default_rng([rm.DRAW_SEED, int(re_), N_REAL, sub, K]))
                        aug = pns.reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], from_b(B_new), P["shape"])
                    elif gen == "jitter":
                        grng = np.random.default_rng([rm.DRAW_SEED, int(re_), N_REAL, sub, K, 7777])
                        idx = grng.integers(0, len(A), size=n_aug)
                        A_new = A[idx] + rm.IC_NOISE * A.std(axis=0)[None, :] * grng.standard_normal((n_aug, K))
                        aug = pns.reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], A_new, P["shape"])
                    else:
                        grng = np.random.default_rng([rm.DRAW_SEED, int(re_), N_REAL, sub, K, 7777])
                        avail = np.setdiff1d(pool_idx, tr_idx)
                        take = min(n_aug, len(avail))
                        extra = np.sort(grng.choice(avail, size=take, replace=False))
                        U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float32)
                        A_x = (data[extra].reshape(take, -1) - P["x_mean"]) @ U
                        del U
                        aug = pns.reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], A_x, P["shape"])
                except Exception as err:
                    msg = str(err).splitlines()[0][:160]
                    print(f"[sub]  !! {gen} {tname}: generation failed ({msg})", flush=True)
                    L.append(CSV_OUT, L.B_FIELDS, dict(info, date=L.now(), kind="gen_failed", generator=gen,
                                                       fraction=round(FRACTION, 4), epochs=o["epochs"],
                                                       status=f"failed: {msg}"))
                    continue
                ek, detR, _, wall = rs.train(train_real, aug, val_real,
                                             tag=f"{DATASET} n={N_REAL} s{sub} {gen} {tname} f={FRACTION}")
                L.append(CSV_OUT, L.B_FIELDS, dict(
                    info, date=L.now(), kind="aug", generator=gen, fraction=round(FRACTION, 4),
                    epochs=o["epochs"], n_aug=len(aug), n_train=N_REAL + len(aug), baseline_Ek=round(base, 4),
                    headroom=round(ceiling - base, 3),
                    quality_err=round(qerr, 5) if (qerr is not None and gen == "galerkin_ns") else "",
                    Ek=round(float(ek), 4), gain=round(float(ek) - base, 3), detR=round(float(detR), 6),
                    status="ok", wall_s=round(wall)))
                print(f"[sub]  {gen:11s} {tname}: Ek {ek:.2f}%  gain {float(ek) - base:+.2f}   "
                      f"[{(time.time() - T0) / 3600:.1f} h]", flush=True)
                del aug
        del P, windows, train_real
    print(f"\n[sub]  done in {(time.time() - T0) / 3600:.2f} h  ->  {CSV_OUT}")


if __name__ == "__main__":
    main()

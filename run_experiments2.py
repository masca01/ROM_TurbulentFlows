"""
SECOND ROUND OF FOLLOW-UP EXPERIMENTS (the "next steps" of the region-map deck).
Same outcome as always: the beta-VAE's reconstruction Ek on the real 10 % random validation
split, and the gain over the SAME real subset trained alone.  Nothing here predicts future
timesteps; the reduced model is only a source of extra training snapshots.

Rule under test (from the region map + first follow-up round):
    a cell gains when   headroom = POD ceiling - real-only Ek  >= 30 points
                  and   quality error of the reduced model     <= 0.5
A cell counts as improved when the mean gain is at least 1 point and its 95 % interval is
above zero over at least 3 real subsets.

N1  POSITIVE TEST ON NEW FLOWS.  Re60 (latent 7) and Re70 (latent 9) with 400 real snapshots,
      99 % modes, NS-projected, fraction 0.5.  The first round only ever tested these flows
      where the rule said "no gain" (they never reached headroom 30 with 100-250 snapshots);
      400 snapshots push the ceiling up, so the rule now has to predict a GAIN and can fail.
      The prediction is written in the CSV before the network is trained.
N2  MORE DIVERSE SYNTHETIC DATA in the reference cell (Re100, 250 real, latent 15, 99 % modes,
      NS-projected, +3.1 points).  Same physics, different sampling of it:
        kick 0.20              bigger initial-condition kick (more spread around the real states)
        horizon 3              shorter trajectories, so ~3x more distinct starting states
        kick 0.20 + horizon 3  both
        the two above at fraction 0.8 (4 synthetic per real) against the reference generator at
        the same dose, plus projected real snapshots at that dose as the ceiling of the ladder.
N3  ARE THE GAINS AN ARTEFACT OF BLOCK SUBSETS?  The reference cell with the 250 real snapshots
      drawn at RANDOM instead of in blocks of 10 (blocks lower the real-only Ek, which is part
      of the headroom).  NS-projected and projected-real, 3 subsets.
N4  BEST ABSOLUTE Ek.  The most favourable combination found so far - 400 real, latent 25,
      fraction 0.67, 99 % modes, NS-projected - on 5 real subsets.  This is the number to
      report if the strategy is used in practice, not just the gain.

Total: 22 real-only + 42 augmented trainings, about 12.5 h (N1 2.6 h, N2 5.7 h, N3 1.4 h,
N4 2.9 h).  Resumable: rows already in the CSV are skipped.

Output: ../../convergence/experiments2_results.csv

Usage:
    python3 run_experiments2.py                 # everything
    python3 run_experiments2.py --exp N1,N3     # a part (about 4 h)
    python3 run_experiments2.py --plan          # what is left and the time estimate
    python3 run_experiments2.py --exp N1 --subsets 1 --epochs 2 --csv /tmp/smoke.csv   # smoke test
"""
import os, sys, csv, time, datetime, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_region_map as rm            # loader, subsets, POD, ceiling, quality windows, settings
import pod_augment_galerkin as pg
import pod_augment_galerkin_ns as pns
import re100_fraction_sweep as rs      # same training recipe as every study

# ============ CONFIG ============
N_SUBSETS     = 3
RULE_HEADROOM = 30.0                   # rule of the region map
RULE_ERROR    = 0.5                    # widened from 0.3 by the E1 dose-response (0.43 -> +2.2)
LATENT = {"Re50": 5, "Re60": 7, "Re70": 9, "Re80": 11, "Re100": 15}
FILES = dict(rm.DATASETS, Re60=("Alpha0/dataRe60Alpha0_2.mat", 1500, 7),
                          Re70=("Alpha0/dataRe70Alpha0_2.mat", 1500, 9))
# reference cell of the first round (+2.8/+3.8/+2.7 points)
REF = dict(ds="Re100", n=250, lat=15, gen="galerkin_ns", lv=0.99, frac=0.5, noise=0.0,
           ridge=rm.RIDGE, kick=rm.IC_NOISE, horiz=rm.GEN_TIME, smode="blocks", nsub=N_SUBSETS)
CSV_OUT = os.path.join(rm._CONV, "experiments2_results.csv")
# ================================
rm.DATASETS = FILES
GIDX = {"galerkin": 0, "galerkin_ns": 1, "jitter": 2, "real_proj": 3, "real_full": 4}


def conditions():
    """Every augmented condition, tagged with the experiment it serves."""
    C = collections.OrderedDict()
    def add(exp, **kw):
        c = dict(REF, **kw); k = ckey(c)
        C.setdefault(k, dict(c, exps=[]))["exps"].append(exp)
    for ds in ("Re60", "Re70"):                                       # N1: the rule has to say "gain"
        add("N1", ds=ds, n=400, lat=LATENT[ds])
    add("N2", kick=0.20)                                              # N2: same physics, more diverse sampling
    add("N2", horiz=3.0)
    add("N2", kick=0.20, horiz=3.0)
    add("N2", frac=0.8)                                               #     dose control at 4 synthetic per real
    add("N2", kick=0.20, horiz=3.0, frac=0.8)
    add("N2", gen="real_proj", frac=0.72)                             #     ceiling of the ladder: every real
                                                                      #     snapshot the pool still holds (643)
    for gen in ("galerkin_ns", "real_proj"):                          # N3: random instead of block subsets
        add("N3", gen=gen, smode="random")
    add("N4", n=400, lat=25, frac=2 / 3, nsub=5)                      # N4: best absolute Ek
    return C


def ckey(c):
    return (c["ds"], c["n"], c["lat"], c["gen"], c["lv"], round(c["frac"], 4), c["noise"], c["ridge"],
            round(c["kick"], 4), round(c["horiz"], 3), c["smode"])


FIELDS = ["date", "exps", "dataset", "n_real", "subset_mode", "subset", "kind", "latent", "generator",
          "mode_level", "K", "energy_K_pct", "fraction", "ic_kick", "gen_time", "noise", "ridge",
          "n_aug", "n_train", "pod_ceiling_Ek", "baseline_Ek", "headroom", "quality_err",
          "quality_blowup_frac", "quality_windows", "gen_keep_frac", "gen_n_traj", "predicted",
          "Ek", "gain", "detR", "status", "wall_s"]


def _args(argv):
    o = {"plan": False, "exp": None, "subsets": None, "epochs": rm.EPOCHS, "csv": CSV_OUT}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--exp": o["exp"] = set(next(it).upper().split(","))
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--csv": o["csv"] = os.path.abspath(next(it))
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


def rowkey(ds, n, smode, sub, kind, lat, gen="", lv="", frac="", kick="", horiz="", noise="", ridge=""):
    return tuple(str(x) for x in (ds, n, smode, sub, kind, lat, gen, lv, frac, kick, horiz, noise, ridge))


def akey(c, sub):
    return rowkey(c["ds"], c["n"], c["smode"], sub, "aug", c["lat"], c["gen"], c["lv"], round(c["frac"], 4),
                  round(c["kick"], 4), round(c["horiz"], 3), c["noise"], c["ridge"])


def load_done(path):
    if not os.path.exists(path):
        return {}, set()
    rows = list(csv.DictReader(open(path, newline="")))
    base = {(r["dataset"], r["n_real"], r["subset_mode"], r["subset"], r["latent"]): float(r["Ek"])
            for r in rows if r["kind"] == "baseline"}
    done = set()
    for r in rows:
        kind = "aug" if r["kind"] == "gen_failed" else r["kind"]
        done.add(rowkey(r["dataset"], r["n_real"], r["subset_mode"], r["subset"], kind, r["latent"],
                        r["generator"], r["mode_level"], r["fraction"], r["ic_kick"], r["gen_time"],
                        r["noise"], r["ridge"]))
    return base, done


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def draw_subset(smode, pool_set, pool_idx, Nt, n, rng):
    if smode == "blocks":
        return rm.draw_blocks(pool_set, Nt, n, rng)
    return np.sort(rng.choice(pool_idx, size=n, replace=False))       # random snapshots, no blocks


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def main():
    o = _args(sys.argv[1:])
    rs.EPOCHS = o["epochs"]
    out = o["csv"]
    base_done, done = load_done(out)
    conds = [c for c in conditions().values() if o["exp"] is None or o["exp"] & set(c["exps"])]
    todo = [(c, sub) for c in conds for sub in range(o["subsets"] or c["nsub"]) if akey(c, sub) not in done]
    bases = sorted({(c["ds"], c["n"], c["smode"], sub, c["lat"]) for c, sub in todo
                    if (c["ds"], str(c["n"]), c["smode"], str(sub), str(c["lat"])) not in base_done})
    samples = sum(n for _, n, _, _, _ in bases) + sum(c["n"] + rs.n_aug_for(c["frac"], c["n"]) for c, _ in todo)
    print(f"[exp2]  experiments {sorted(o['exp']) if o['exp'] else 'N1-N4'}, {o['epochs']} epochs")
    print(f"[exp2]  to do: {len(bases)} real-only + {len(todo)} augmented trainings, about "
          f"{samples * rm.SEC_PER_SAMPLE * o['epochs'] / 500 / 3600:.1f} h of training  ->  {out}", flush=True)
    if o["plan"] or not todo:
        for c, sub in todo if o["plan"] else []:
            print(f"        {'/'.join(c['exps']):6s} {c['ds']:5s} n={c['n']:3d} {c['smode']:6s} s{sub} lat {c['lat']:2d} "
                  f"{c['gen']:11s} K{c['lv']:.2f} f={c['frac']:.2f} kick {c['kick']:.2f} horizon {c['horiz']:g}")
        return

    T0 = time.time()
    by_ds = collections.OrderedDict()
    for c, sub in todo:
        by_ds.setdefault(c["ds"], collections.OrderedDict()).setdefault((c["n"], c["smode"], sub), []).append(c)
    for ds, groups in by_ds.items():
        data, re_, dt, path = rm.load(ds)
        Nt = len(data)
        val_idx, pool_idx, n_val = rm.cs.split_indices(Nt, None, "random", rm.SPLIT_SEED)
        pool_set = set(int(i) for i in pool_idx)
        val_real = data[val_idx]
        steps_q = max(1, int(round(rm.GEN_TIME / dt)))                # quality always over 10 convective times
        for (n, smode, sub), cs_ in groups.items():
            # block subsets: identical draw and quality windows to run_region_map.py / run_experiments.py
            seed = [rm.DRAW_SEED, int(re_), n, sub] + ([] if smode == "blocks" else [555])
            rng = np.random.default_rng(seed)
            tr_idx = draw_subset(smode, pool_set, pool_idx, Nt, n, rng)
            train_real = data[tr_idx]
            win_starts = rm.fidelity_windows(set(int(i) for i in tr_idx), Nt, steps_q, rng)
            print(f"\n[exp2]  ===== {ds}  {n} real ({smode})  subset {sub}  ({len(cs_)} conditions, "
                  f"{len(win_starts)} quality windows) =====", flush=True)

            for lat in sorted({c["lat"] for c in cs_}):                       # real-only baselines first
                if (ds, str(n), smode, str(sub), str(lat)) in base_done:
                    continue
                rs.LATENT = lat
                ek, detR, _, wall = rs.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32), val_real,
                                             tag=f"{ds} n={n} {smode} s{sub} latent {lat} real")
                base_done[(ds, str(n), smode, str(sub), str(lat))] = float(ek)
                append(out, dict(date=now(), exps=" ".join(sorted({e for c in cs_ if c["lat"] == lat for e in c["exps"]})),
                                 dataset=ds, n_real=n, subset_mode=smode, subset=sub, kind="baseline", latent=lat,
                                 n_train=n, Ek=round(float(ek), 4), detR=round(float(detR), 6), status="ok",
                                 wall_s=round(wall)))
                print(f"[exp2]  latent {lat} real-only Ek {ek:.2f}%", flush=True)

            P = rm.subset_pod(train_real)
            windows = [data[t:t + steps_q + 1] for t in win_starts]
            for lv in sorted({c["lv"] for c in cs_}):
                K = rm.k_for(P, lv)
                ceiling, win_A = rm.ceiling_and_project(P, K, val_real, windows)
                A = np.asarray(P["A"][:, :K], dtype=np.float64)
                ns_ops = None
                variants = collections.OrderedDict()
                for c in [c for c in cs_ if c["lv"] == lv]:
                    variants.setdefault((c["gen"], c["noise"], c["ridge"], c["kick"], c["horiz"]), []).append(c)
                print(f"[exp2]  {lv:.0%} modes: K = {K} ({100 * P['e_cum'][K - 1]:.2f}% energy), "
                      f"POD ceiling {ceiling:.2f}%", flush=True)
                for (gen, eps, ridge, kick, horiz), vcs in variants.items():
                    t0 = time.time()
                    steps_g = max(1, int(round(horiz / dt)))           # trajectory length of THIS variant
                    n_max = max(rs.n_aug_for(c["frac"], n) for c in vcs)
                    info = dict(K=K, energy_K_pct=round(100 * P["e_cum"][K - 1], 3), pod_ceiling_Ek=round(ceiling, 3),
                                quality_windows=len(win_A))
                    try:
                        if gen in ("jitter", "real_proj", "real_full"):
                            grng = np.random.default_rng([rm.DRAW_SEED, int(re_), n, sub, K, 7777])
                            if gen == "jitter":
                                rows_ = grng.integers(0, len(A), size=n_max)
                                A_new = A[rows_] + rm.IC_NOISE * A.std(axis=0)[None, :] * grng.standard_normal((n_max, K))
                                aug = pns.reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], A_new, P["shape"])
                            else:
                                avail = np.setdiff1d(pool_idx, tr_idx)     # real snapshots the subset does not use
                                take = min(n_max, len(avail))
                                if take < n_max:
                                    print(f"[exp2]  {gen}: pool holds only {take} unused real snapshots, "
                                          f"asked for {n_max}", flush=True)
                                extra = np.sort(grng.choice(avail, size=take, replace=False))
                                n_max = take
                                if gen == "real_full":
                                    aug = data[extra].astype(np.float32)
                                else:
                                    U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float32)
                                    A_x = (data[extra].reshape(take, -1) - P["x_mean"]) @ U
                                    del U
                                    aug = pns.reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], A_x, P["shape"])
                            print(f"[exp2]  {gen}: {n_max} snapshots [{time.time() - t0:.0f} s]", flush=True)
                            qerr = None
                        elif gen == "galerkin_ns":
                            if ns_ops is None:
                                C_, H_, W_ = P["shape"]
                                dx, dy = pns.grid_spacing(path); kappa = np.sqrt(dx * dy)
                                U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float64)
                                Pu = np.empty((K + 1, H_, W_)); Pv = np.empty((K + 1, H_, W_))
                                mf = P["x_mean"].astype(np.float64).reshape(C_, H_, W_); Pu[0], Pv[0] = mf[0], mf[1]
                                md = (U / kappa).T.reshape(K, C_, H_, W_); Pu[1:], Pv[1:] = md[:, 0], md[:, 1]
                                del U, md
                                l0, q0 = pns.galerkin_operators_ns(Pu, Pv, dx, dy, re_); del Pu, Pv
                                s = (A * kappa).std(axis=0); s = np.where(s < 1e-14, 1.0, s)
                                ns_ops = (l0, q0, s, kappa)
                            l0, q0, s, kappa = ns_ops
                            step = pns.make_step_ns(l0, q0, s, dt, rm.SUBSTEPS)
                            to_b = lambda a, s=s, k=kappa: a * k / s
                            from_b = lambda b, s=s, k=kappa: b * s / k
                            integ = pns.integrate_trajectories
                        else:
                            beta, s, _, _ = pg.fit_galerkin(A, tr_idx, dt, ridge)
                            step = pg.make_step(beta, "ode", dt, rm.SUBSTEPS)
                            to_b = lambda a, s=s: a / s
                            from_b = lambda b, s=s: b * s
                            integ = pg.integrate_trajectories
                        if gen in ("galerkin_ns", "galerkin"):
                            qerr, qblow = (rm.fidelity(step, to_b, from_b, win_A, steps_q) if win_A else (None, ""))
                            if qerr is not None:
                                info.update(quality_err=round(qerr, 5), quality_blowup_frac=round(qblow, 3))
                            gseed = [rm.DRAW_SEED, int(re_), n, sub, K, GIDX[gen]]
                            if abs(kick - rm.IC_NOISE) > 1e-12 or abs(horiz - rm.GEN_TIME) > 1e-12:
                                gseed += [int(round(kick * 1000)), int(round(horiz * 10))]
                            B_new, st = integ(step, s, to_b(A), n_max, steps_g, kick, rm.ENERGY_TOL, rm.MAX_TRAJ,
                                              np.random.default_rng(gseed))
                            aug = pns.reconstruct(P["x_mean"], P["Xc"], P["Wp"][:, :K], from_b(B_new), P["shape"])
                            info.update(gen_keep_frac=round(st["keep_frac"], 4), gen_n_traj=st["n_traj"])
                            print(f"[exp2]  {gen} kick {kick:g} horizon {horiz:g} ({steps_g} steps): quality error "
                                  f"{'-' if qerr is None else f'{qerr:.3f}'}, {n_max} synthetic from {st['n_traj']} "
                                  f"trajectories, {100 * st['keep_frac']:.0f}% kept [{time.time() - t0:.0f} s]", flush=True)
                    except Exception as err:
                        msg = str(err).splitlines()[0][:160]
                        print(f"[exp2]  !! {gen} kick {kick:g} horizon {horiz:g}: generation failed ({msg})", flush=True)
                        for c in vcs:
                            append(out, dict(info, date=now(), exps=" ".join(c["exps"]), dataset=ds, n_real=n,
                                             subset_mode=smode, subset=sub, kind="gen_failed", latent=c["lat"],
                                             generator=gen, mode_level=lv, fraction=round(c["frac"], 4),
                                             ic_kick=kick, gen_time=horiz, noise=eps, ridge=ridge,
                                             status=f"failed: {msg}"))
                        continue
                    for c in vcs:
                        na = min(rs.n_aug_for(c["frac"], n), len(aug))   # capped for real_proj/real_full
                        base = base_done.get((ds, str(n), smode, str(sub), str(c["lat"])))
                        headroom = ceiling - base
                        pred = "" if qerr is None else ("gain" if (headroom >= RULE_HEADROOM and qerr <= RULE_ERROR) else "no gain")
                        rs.LATENT = c["lat"]
                        ek, detR, _, wall = rs.train(train_real, aug[:na], val_real,
                                                     tag=f"{ds} n={n} {smode} s{sub} lat {c['lat']} {gen} K={K} "
                                                         f"f={c['frac']:.2f} kick {kick:g} horizon {horiz:g}")
                        append(out, dict(info, date=now(), exps=" ".join(c["exps"]), dataset=ds, n_real=n,
                                         subset_mode=smode, subset=sub, kind="aug", latent=c["lat"], generator=gen,
                                         mode_level=lv, fraction=round(c["frac"], 4), ic_kick=kick, gen_time=horiz,
                                         noise=eps, ridge=ridge, n_aug=na, n_train=n + na, baseline_Ek=round(base, 4),
                                         headroom=round(headroom, 3), predicted=pred, Ek=round(float(ek), 4),
                                         gain=round(float(ek) - base, 3), detR=round(float(detR), 6), status="ok",
                                         wall_s=round(wall)))
                        print(f"[exp2]  {' '.join(c['exps'])}: latent {c['lat']} f={c['frac']:.2f}: Ek {ek:.2f}%  "
                              f"gain {float(ek) - base:+.2f}  (headroom {headroom:.1f}, quality "
                              f"{'-' if qerr is None else f'{qerr:.3f}'}, predicted {pred or '-'})   "
                              f"[{(time.time() - T0) / 3600:.1f} h]", flush=True)
                    del aug
            del P, train_real, windows
        del data, val_real
    print(f"\n[exp2]  done in {(time.time() - T0) / 3600:.2f} h  ->  {out}")


if __name__ == "__main__":
    main()

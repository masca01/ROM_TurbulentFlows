"""
FOLLOW-UP EXPERIMENTS to the region map: when do POD-Galerkin synthetic snapshots raise the
beta-VAE's reconstruction Ek?  Everything is on the random 10 % validation split (no temporal
validation), the outcome is always the network's Ek gain over the same real subset alone, and
every condition is repeated on 3 real subsets drawn exactly as in run_region_map.py.

The two explanatory coordinates are the ones of the region map:
    headroom = POD ceiling of the real subset (K modes) - real-only Ek of the network
    quality  = error of the reduced model on held-out real snapshots (a check of how physical the
               synthetic data are; 0 perfect, above 1 the model has left the flow)

Positive cell of the region map ("reference"): Re100, 250 real, NS-projected, 99 % modes,
latent 15, synthetic fraction 0.5  ->  +2.8, +3.8, +2.7 points.

E1  headroom vs quality, one at a time, on the reference subsets
      headroom: latent 5 and 25 (same synthetic data, different network)
      quality : relative noise 20 %, 35 %, 50 % on the projected NS operators (same network)
E2  how much synthetic data: fractions 0.25 and 0.67 in the reference cell
E3  what are the synthetic snapshots worth: in the reference cell, replace the 250 NS synthetic
      snapshots by
        jitter     250 real training coefficient vectors + 5 % noise, rebuilt on the same modes
                   (new samples but no dynamics: the lower reference)
        real_proj  250 EXTRA real snapshots from the pool, projected on the same modes
                   (what a perfect reduced model could give)
        real_full  the same 250 extra real snapshots, full fields (the upper bound)
      (a ridge-regularised data-identified model was checked and stays at quality error 1.7
      for any ridge, so it is not trained here)
E4  how much real data: 150 and 400 real snapshots with the reference model and network
      (100 and 250 are already in the region map with the same subsets)
E5  does the rule transfer to new flows: Re60 (latent 7) and Re70 (latent 9), 100 and 250 real,
      95 % and 99 % modes, NS-projected. Before each training the script writes the predicted
      outcome from the rule (gain if headroom >= RULE_HEADROOM and quality error <= RULE_ERROR).

Conditions shared by several experiments are trained once (the reference serves E1-E4).
Total: 27 real-only + 63 augmented trainings, about 12.5 hours. Run a part with --exp.

Output: ../../convergence/experiments_results.csv (resumable: finished rows are skipped).

Usage:
    python3 run_round1.py                  # everything
    python3 run_round1.py --exp E1,E2,E3   # first night (about 7 h)
    python3 run_round1.py --exp E4,E5      # second night (about 6 h)
    python3 run_round1.py --plan           # what is left, nothing runs
"""
import os, sys, csv, time, datetime, collections
import numpy as np

from rom import paths
from rom import galerkin_data as gd
from rom import galerkin_ns as gns
from rom import vae                    # same training recipe as every study
from rom.data import load
from rom.split import split_indices, draw_blocks, SPLIT_SEED, DRAW_SEED
from rom.pod import subset_pod, k_for, ceiling_and_project
from rom.augment import (n_aug_for, fidelity, fidelity_windows, synthetic_arm, jitter_arm, real_arm,
                         GENERATORS, GEN_TIME, SUBSTEPS, IC_NOISE, ENERGY_TOL, RIDGE, MAX_TRAJ)
from rom.vae import EPOCHS, SEC_PER_SAMPLE
from rom.results import append_row, now

# ============ CONFIG ============
N_SUBSETS     = 3
RULE_HEADROOM = 30.0                   # rule from the region map, used for E5 predictions
RULE_ERROR    = 0.3
NOISE_SEED    = 404
REF = dict(ds="Re100", n=250, lat=15, gen="galerkin_ns", lv=0.99, frac=0.5, noise=0.0, ridge=RIDGE)
LATENT = {"Re50": 5, "Re60": 7, "Re70": 9, "Re80": 11, "Re100": 15}
CSV_OUT = os.path.join(paths.RESULTS, "experiments_results.csv")
# ================================
# files and caps of Re50..Re100: rom.data.DATASETS


def conditions():
    """Every augmented condition, tagged with the experiments it serves."""
    C = collections.OrderedDict()
    def add(exp, **kw):
        c = dict(REF, **kw); k = ckey(c)
        C.setdefault(k, dict(c, exps=[]))["exps"].append(exp)
    add("E1")
    for lat in (5, 25): add("E1", lat=lat)
    for eps in (0.20, 0.35, 0.50): add("E1", noise=eps)
    add("E2")
    for f in (0.25, 2 / 3): add("E2", frac=f)
    add("E3")
    for gen in ("jitter", "real_proj", "real_full"): add("E3", gen=gen)
    add("E4")
    for n in (150, 400): add("E4", n=n)
    for ds in ("Re60", "Re70"):
        for n in (100, 250):
            for lv in (0.95, 0.99):
                add("E5", ds=ds, n=n, lat=LATENT[ds], lv=lv)
    return C


def ckey(c):
    return (c["ds"], c["n"], c["lat"], c["gen"], c["lv"], round(c["frac"], 4), c["noise"], c["ridge"])


FIELDS = ["date", "exps", "dataset", "n_real", "subset", "kind", "latent", "generator", "mode_level", "K", "energy_K_pct",
          "fraction", "noise", "ridge", "n_aug", "n_train", "pod_ceiling_Ek", "baseline_Ek", "headroom",
          "quality_err", "quality_blowup_frac", "gen_keep_frac", "gen_n_traj", "predicted", "Ek", "gain", "detR",
          "status", "wall_s"]


def _args(argv):
    o = {"plan": False, "exp": None, "subsets": N_SUBSETS, "epochs": EPOCHS}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--exp": o["exp"] = set(next(it).upper().split(","))
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


def rowkey(ds, n, sub, kind, lat, gen="", lv="", frac="", noise="", ridge=""):
    return tuple(str(x) for x in (ds, n, sub, kind, lat, gen, lv, frac, noise, ridge))


def load_done():
    if not os.path.exists(CSV_OUT):
        return {}, set()
    rows = list(csv.DictReader(open(CSV_OUT, newline="")))
    base = {(r["dataset"], r["n_real"], r["subset"], r["latent"]): float(r["Ek"]) for r in rows if r["kind"] == "baseline"}
    done = set()
    for r in rows:
        kind = "aug" if r["kind"] == "gen_failed" else r["kind"]
        done.add(rowkey(r["dataset"], r["n_real"], r["subset"], kind, r["latent"], r["generator"], r["mode_level"],
                        r["fraction"], r["noise"], r["ridge"]))
    return base, done


def append(row):
    append_row(CSV_OUT, FIELDS, row)


def akey(c, sub):
    return rowkey(c["ds"], c["n"], sub, "aug", c["lat"], c["gen"], c["lv"], round(c["frac"], 4), c["noise"], c["ridge"])


def main():
    o = _args(sys.argv[1:])
    base_done, done = load_done()
    conds = [c for c in conditions().values() if o["exp"] is None or o["exp"] & set(c["exps"])]
    todo = [(c, sub) for c in conds for sub in range(o["subsets"]) if akey(c, sub) not in done]
    bases = sorted({(c["ds"], c["n"], sub, c["lat"]) for c, sub in todo
                    if (c["ds"], str(c["n"]), str(sub), str(c["lat"])) not in base_done})
    samples = sum(n for _, n, _, _ in bases) + sum(c["n"] + n_aug_for(c["frac"], c["n"]) for c, _ in todo)
    print(f"[exp]  experiments {sorted(o['exp']) if o['exp'] else 'E1-E5'}, {o['subsets']} subsets, {o['epochs']} epochs")
    print(f"[exp]  to do: {len(bases)} real-only + {len(todo)} augmented trainings, about "
          f"{samples * SEC_PER_SAMPLE * o['epochs'] / 500 / 3600:.1f} h of training  ->  {CSV_OUT}", flush=True)
    if o["plan"] or not todo:
        return

    T0 = time.time()
    by_ds = collections.OrderedDict()
    for c, sub in todo:
        by_ds.setdefault(c["ds"], collections.OrderedDict()).setdefault((c["n"], sub), []).append(c)
    for ds, groups in by_ds.items():
        data, re_, dt, path = load(ds)
        Nt = len(data)
        val_idx, pool_idx, n_val = split_indices(Nt, None, "random", SPLIT_SEED)
        pool_set = set(int(i) for i in pool_idx)
        val_real = data[val_idx]
        steps = max(1, int(round(GEN_TIME / dt)))
        for (n, sub), cs_ in groups.items():
            # identical subset and quality windows to run_region_map.py (same seeds, same call order)
            rng = np.random.default_rng([DRAW_SEED, int(re_), n, sub])
            tr_idx = draw_blocks(pool_set, Nt, n, rng)
            train_real = data[tr_idx]
            win_starts = fidelity_windows(set(int(i) for i in tr_idx), Nt, steps, rng)
            print(f"\n[exp]  ===== {ds}  {n} real  subset {sub}  ({len(cs_)} conditions) =====", flush=True)

            for lat in sorted({c["lat"] for c in cs_}):                       # real-only baselines first
                if (ds, str(n), str(sub), str(lat)) in base_done:
                    continue
                ek, detR, _, wall = vae.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32), val_real,
                                              tag=f"{ds} n={n} s{sub} latent {lat} real", latent=lat, epochs=o["epochs"])
                base_done[(ds, str(n), str(sub), str(lat))] = float(ek)
                append(dict(date=now(), exps=" ".join(sorted({e for c in cs_ if c["lat"] == lat for e in c["exps"]})), dataset=ds,
                            n_real=n, subset=sub, kind="baseline", latent=lat, n_train=n, Ek=round(float(ek), 4),
                            detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
                print(f"[exp]  latent {lat} real-only Ek {ek:.2f}%", flush=True)

            P = subset_pod(train_real)
            windows = [data[t:t + steps + 1] for t in win_starts]
            for lv in sorted({c["lv"] for c in cs_}):
                K = k_for(P, lv)
                ceiling, win_A = ceiling_and_project(P, K, val_real, windows)
                A = np.asarray(P["A"][:, :K], dtype=np.float64)
                ns_ops = None
                variants = collections.OrderedDict()
                for c in [c for c in cs_ if c["lv"] == lv]:
                    variants.setdefault((c["gen"], c["noise"], c["ridge"]), []).append(c)
                print(f"[exp]  {lv:.0%} modes: K = {K} ({100 * P['e_cum'][K - 1]:.2f}% energy), POD ceiling {ceiling:.2f}%", flush=True)
                for (gen, eps, ridge), vcs in variants.items():
                    t0 = time.time()
                    info = dict(K=K, energy_K_pct=round(100 * P["e_cum"][K - 1], 3), pod_ceiling_Ek=round(ceiling, 3))
                    try:
                        if gen in ("jitter", "real_proj", "real_full"):
                            n_max = max(n_aug_for(c["frac"], n) for c in vcs)
                            grng = np.random.default_rng([DRAW_SEED, int(re_), n, sub, K, 7777])
                            if gen == "jitter":
                                aug = jitter_arm(P, A, K, n_max, grng, kick=IC_NOISE)
                            else:
                                # exactly n_max extra real snapshots (no cap: a short pool fails as before)
                                aug, _ = real_arm(P, K, data, pool_idx, tr_idx, n_max, grng,
                                                  project=(gen == "real_proj"), cap=False)
                            info.update(gen_n_traj="")
                            print(f"[exp]  {gen}: {n_max} snapshots [{time.time() - t0:.0f} s]", flush=True)
                            qerr = None
                        elif gen == "galerkin_ns":
                            if ns_ops is None:
                                C_, H_, W_ = P["shape"]
                                dx, dy = gns.grid_spacing(path); kappa = np.sqrt(dx * dy)
                                U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float64)
                                Pu = np.empty((K + 1, H_, W_)); Pv = np.empty((K + 1, H_, W_))
                                mf = P["x_mean"].astype(np.float64).reshape(C_, H_, W_); Pu[0], Pv[0] = mf[0], mf[1]
                                md = (U / kappa).T.reshape(K, C_, H_, W_); Pu[1:], Pv[1:] = md[:, 0], md[:, 1]
                                del U, md
                                l0, q0 = gns.galerkin_operators_ns(Pu, Pv, dx, dy, re_); del Pu, Pv
                                s = (A * kappa).std(axis=0); s = np.where(s < 1e-14, 1.0, s)
                                ns_ops = (l0, q0, s, kappa)
                            l0, q0, s, kappa = ns_ops
                            nr = np.random.default_rng([NOISE_SEED, n, sub, int(round(eps * 1000))])
                            l = l0 * (1 + eps * nr.standard_normal(l0.shape)) if eps > 0 else l0
                            q = q0 * (1 + eps * nr.standard_normal(q0.shape)) if eps > 0 else q0
                            step = gns.make_step_ns(l, q, s, dt, SUBSTEPS)
                            to_b = lambda a, s=s, k=kappa: a * k / s
                            from_b = lambda b, s=s, k=kappa: b * s / k
                        else:
                            beta, s, _, _ = gd.fit_galerkin(A, tr_idx, dt, ridge)
                            step = gd.make_step(beta, "ode", dt, SUBSTEPS)
                            to_b = lambda a, s=s: a / s
                            from_b = lambda b, s=s: b * s
                        if gen in ("galerkin_ns", "galerkin"):
                            qerr, qblow = fidelity(step, to_b, from_b, win_A, steps)
                            info.update(quality_err=round(qerr, 5), quality_blowup_frac=round(qblow, 3))
                            n_max = max(n_aug_for(c["frac"], n) for c in vcs)
                            aug, st = synthetic_arm(step, s, to_b, from_b, P, A, K, n_max, steps,
                                                    np.random.default_rng([DRAW_SEED, int(re_), n, sub, K, GENERATORS.index(gen)]),
                                                    kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ)
                            info.update(gen_keep_frac=round(st["keep_frac"], 4), gen_n_traj=st["n_traj"])
                            print(f"[exp]  {gen} noise {eps:g} ridge {ridge:g}: quality error {qerr:.3f} (blow-ups {qblow:.0%}), "
                                  f"{n_max} synthetic, {100 * st['keep_frac']:.0f}% kept [{time.time() - t0:.0f} s]", flush=True)
                    except Exception as err:
                        msg = str(err).splitlines()[0][:160]
                        print(f"[exp]  !! {gen} noise {eps:g} ridge {ridge:g}: generation failed ({msg})", flush=True)
                        for c in vcs:
                            append(dict(info, date=now(), exps=" ".join(c["exps"]), dataset=ds, n_real=n, subset=sub, kind="gen_failed",
                                        latent=c["lat"], generator=gen, mode_level=lv, fraction=round(c["frac"], 4), noise=eps,
                                        ridge=ridge, status=f"failed: {msg}"))
                        continue
                    for c in vcs:
                        na = n_aug_for(c["frac"], n)
                        base = base_done.get((ds, str(n), str(sub), str(c["lat"])))
                        headroom = ceiling - base
                        pred = "" if qerr is None else ("gain" if (headroom >= RULE_HEADROOM and qerr <= RULE_ERROR) else "no gain")
                        ek, detR, _, wall = vae.train(train_real, aug[:na], val_real, latent=c["lat"], epochs=o["epochs"],
                                                     tag=f"{ds} n={n} s{sub} lat {c['lat']} {gen} K={K} f={c['frac']:.2f} noise {eps:g} ridge {ridge:g}")
                        append(dict(info, date=now(), exps=" ".join(c["exps"]), dataset=ds, n_real=n, subset=sub, kind="aug",
                                    latent=c["lat"], generator=gen, mode_level=lv, fraction=round(c["frac"], 4), noise=eps, ridge=ridge,
                                    n_aug=na, n_train=n + na, baseline_Ek=round(base, 4), headroom=round(headroom, 3), predicted=pred,
                                    Ek=round(float(ek), 4), gain=round(float(ek) - base, 3), detR=round(float(detR), 6),
                                    status="ok", wall_s=round(wall)))
                        print(f"[exp]  {' '.join(c['exps'])}: latent {c['lat']} f={c['frac']:.2f}: Ek {ek:.2f}%  gain {float(ek) - base:+.2f}  "
                              f"(headroom {headroom:.1f}, quality {'-' if qerr is None else f'{qerr:.3f}'}, predicted {pred or '-'})   [{(time.time() - T0) / 3600:.1f} h]", flush=True)
                    del aug
            del P, train_real, windows
        del data, val_real
    print(f"\n[exp]  done in {(time.time() - T0) / 3600:.2f} h  ->  {CSV_OUT}")


if __name__ == "__main__":
    main()

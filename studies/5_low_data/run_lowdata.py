"""
VERY LOW DATA: can synthetic data still help the beta-VAE when only 25-100 real snapshots exist?

Two stages, one script.

STAGE A - screen, no network is trained (about 1 h).
    Every flow (Re50, Re60, Re70, Re80, Re100) x 25 / 50 / 100 real snapshots x 3 real subsets
    x two ways of sampling them x six truncations. For each combination it measures the two
    coordinates of the region map before any training:
        POD ceiling   = best Ek the K-mode subspace of that subset allows on the real validation set
        quality error = prediction error of the NS-projected model over 10 convective times on
                        held-out real windows (0 perfect, above 1 the model has left the flow)
    and checks that synthetic snapshots can be generated at all. The projected operators are built
    once per subset at K_MAX and sliced for the smaller truncations (exact: l[:K,:K+1], q[:K,:K+1,:K+1]).
    Output: ../../convergence/lowdata_screen.csv

STAGE B - trainings (about 4.5 h).
    Re50 at 25, 50 and 100 real snapshots, with five arms:
        real only | NS synthetic f = 0.5 | NS synthetic f = 0.8
        jitter f = 0.8      (real coefficients + noise, no dynamics: the lower bound)
        projected real f=0.8 (extra real snapshots projected on the same modes: the upper bound)
    Re80 and Re100 at 50 real snapshots, real only + NS f = 0.5 + f = 0.8   (negative controls)
    Re50 at 50 real, real only with 1500 epochs  (control: a gain must not be just more compute)
    The truncation of each cell is taken from the stage-A screen (lowest quality error that can
    still generate), so the screen decides the design instead of a guess.
    Output: ../../convergence/lowdata_results.csv

Pre-registered predictions (written in the CSV before each network is trained), from the rule as it
stands after 51 cells: gain expected when quality error <= 0.5 and headroom >= 20 points.
Expectation at the time of writing: Re50 gains at all three sizes (its error at 50 snapshots is 0.09,
the lowest measured in the project); Re80 and Re100 at 50 do not (errors 0.81 and 0.61).

Usage:
    python3 run_lowdata.py                 # stage A then stage B
    python3 run_lowdata.py --stage A       # the screen only
    python3 run_lowdata.py --stage B       # the trainings only (uses the screen if it exists)
    python3 run_lowdata.py --plan          # what is left and the time estimate, nothing runs
    python3 run_lowdata.py --stage A --datasets Re50 --n 50 --subsets 1 --suffix smoke   # smoke test
"""
import os, sys, csv, time, datetime, collections
import numpy as np

from rom import paths
from rom import vae                    # train(): the recipe of every study so far
from rom.data import load
from rom.split import split_indices, draw_subset, SPLIT_SEED, DRAW_SEED
from rom.pod import subset_pod, ceiling_and_project, TRUNCS, k_of
from rom.galerkin_ns import build_ops, model_at, integrate_trajectories
from rom.augment import (n_aug_for, fidelity, fidelity_windows, synthetic_arm, jitter_arm, real_arm,
                         GEN_TIME, SUBSTEPS, IC_NOISE, ENERGY_TOL, MAX_TRAJ)
from rom.vae import EPOCHS, SEC_PER_SAMPLE
from rom.results import append_row as append, done_rows, now, LOWDATA_FIELDS as B_FIELDS

# ============ CONFIG ============
DATASETS   = ["Re50", "Re60", "Re70", "Re80", "Re100"]
LATENT     = {"Re50": 5, "Re60": 7, "Re70": 9, "Re80": 11, "Re100": 15}
N_REAL     = [25, 50, 100]
N_SUBSETS  = 3
SAMPLINGS  = ["blocks", "contig"]      # blocks of 10 anywhere in the pool | one stretch of the record
# TRUNCS = 95% / 99% energy, K = 5, 10, 15, 20 modes (rom.pod.TRUNCS, shared with run_lowdata_subsets.py)
K_MAX      = 20                        # operators are built once at this size and sliced
RULE_ERROR = 0.5                       # rule after 51 cells
RULE_HEAD  = 20.0
LONG_EPOCHS = 1500                     # compute control
SCREEN_CSV = os.path.join(paths.RESULTS, "lowdata_screen.csv")
TRAIN_CSV  = os.path.join(paths.RESULTS, "lowdata_results.csv")
FALLBACK   = {"Re50": "K10", "Re60": "K10", "Re70": "95%", "Re80": "95%", "Re100": "95%"}
# ================================
# files and caps of Re50..Re100: rom.data.DATASETS

A_FIELDS = ["date", "dataset", "Re", "n_real", "sampling", "subset", "trunc", "K", "energy_K_pct",
            "pod_ceiling_Ek", "quality_err", "quality_blowup_frac", "quality_windows",
            "gen_keep_frac", "gen_n_traj", "gen_status", "seconds"]
# B_FIELDS (lowdata_results.csv) = rom.results.LOWDATA_FIELDS, shared with run_lowdata_subsets.py


def _args(argv):
    o = {"plan": False, "stage": "AB", "datasets": DATASETS, "n": N_REAL, "subsets": N_SUBSETS,
         "epochs": EPOCHS, "sampling": "blocks"}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--stage": o["stage"] = next(it).upper()
        elif a == "--datasets": o["datasets"] = next(it).split(",")
        elif a == "--n": o["n"] = [int(x) for x in next(it).split(",")]
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--sampling": o["sampling"] = next(it)
        elif a == "--suffix":                      # smoke tests: write to separate files
            global SCREEN_CSV, TRAIN_CSV
            suf = next(it)
            SCREEN_CSV = SCREEN_CSV.replace(".csv", f"_{suf}.csv")
            TRAIN_CSV = TRAIN_CSV.replace(".csv", f"_{suf}.csv")
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


# ------------------------------------------------------------------ stage A

def stage_a(o):
    done = done_rows(SCREEN_CSV, lambda r: (r["dataset"], r["n_real"], r["sampling"], r["subset"], r["trunc"]))
    todo = [(ds, n, sm, sub) for ds in o["datasets"] for n in o["n"] for sm in SAMPLINGS
            for sub in range(o["subsets"])
            if any((ds, str(n), sm, str(sub), t[0]) not in done for t in TRUNCS)]
    print(f"[low]  stage A: {len(todo)} subsets to screen ({len(todo) * len(TRUNCS)} rows), "
          f"about {len(todo) * 0.75:.0f} min  ->  {SCREEN_CSV}", flush=True)
    if o["plan"] or not todo:
        return
    by_ds = collections.OrderedDict()
    for ds, n, sm, sub in todo:
        by_ds.setdefault(ds, []).append((n, sm, sub))
    for ds, items in by_ds.items():
        data, re_, dt, path = load(ds)
        Nt = len(data)
        val_idx, pool_idx, _ = split_indices(Nt, None, "random", SPLIT_SEED)
        pool_set = set(int(i) for i in pool_idx)
        val_real = data[val_idx]
        steps = max(1, int(round(GEN_TIME / dt)))
        for n, sm, sub in items:
            t0 = time.time()
            rng = np.random.default_rng([DRAW_SEED, int(re_), n, sub] + ([] if sm == "blocks" else [911]))
            tr_idx = draw_subset(sm, pool_set, pool_idx, Nt, n, rng)
            P = subset_pod(data[tr_idx])
            win = fidelity_windows(set(int(i) for i in tr_idx), Nt, steps, rng)
            windows = [data[t:t + steps + 1] for t in win]
            Kb = int(min(K_MAX, P["A"].shape[1], n - 1))
            l_big, q_big, kappa = build_ops(P, Kb, re_, path)
            A = np.asarray(P["A"][:, :Kb], dtype=np.float64)
            print(f"[low]  {ds} n={n} {sm} s{sub}: {len(win)} quality windows, operators at K={Kb} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
            for trunc in TRUNCS:
                name = trunc[0]
                if (ds, str(n), sm, str(sub), name) in done:
                    continue
                K = k_of(P, trunc, n)
                if K > Kb or K < 1:                      # needs more modes than the operators hold
                    append(SCREEN_CSV, A_FIELDS, dict(date=now(), dataset=ds, Re=re_, n_real=n, sampling=sm,
                                                      subset=sub, trunc=name, K=K,
                                                      gen_status=f"skipped: K={K} above K_MAX={Kb}"))
                    continue
                ceiling, win_A = ceiling_and_project(P, K, val_real, windows)
                step, s, to_b, from_b = model_at(l_big, q_big, kappa, A, K, dt, SUBSTEPS)
                err, blow = fidelity(step, to_b, from_b, win_A, steps) if win_A else (float("nan"), "")
                try:
                    _, st = integrate_trajectories(step, s, to_b(A[:, :K]), n, steps, IC_NOISE,
                                                   ENERGY_TOL, 4000, np.random.default_rng([7, K, n]))
                    gen, keep, ntraj = "ok", round(st["keep_frac"], 4), st["n_traj"]
                except Exception as e:
                    gen, keep, ntraj = str(e).splitlines()[0][:80], "", ""
                append(SCREEN_CSV, A_FIELDS, dict(
                    date=now(), dataset=ds, Re=re_, n_real=n, sampling=sm, subset=sub, trunc=name, K=K,
                    energy_K_pct=round(100 * P["e_cum"][K - 1], 3), pod_ceiling_Ek=round(ceiling, 3),
                    quality_err=round(err, 5), quality_blowup_frac=blow, quality_windows=len(win_A),
                    gen_keep_frac=keep, gen_n_traj=ntraj, gen_status=gen, seconds=round(time.time() - t0)))
                print(f"[low]      {name:4s} K={K:3d} energy {100 * P['e_cum'][K - 1]:5.2f}%  ceiling "
                      f"{ceiling:5.1f}%  error {err:5.2f}  {gen}", flush=True)
            del P, l_big, q_big, windows
        del data, val_real


# ------------------------------------------------------------------ stage B

def pick_trunc(ds, n, sm):
    """Truncation chosen by the screen: the HIGHEST POD ceiling among the truncations whose model
    is still faithful (quality error <= RULE_ERROR) and that can generate. Choosing the lowest
    error instead collapses to 3-4 modes, whose ceiling the network already reaches, so the cell
    has no headroom left and can only lose. If no truncation is faithful, fall back to the least
    unfaithful one (that cell is a negative control anyway)."""
    by_trunc = {}
    if os.path.exists(SCREEN_CSV):
        for r in csv.DictReader(open(SCREEN_CSV, newline="")):
            if (r["dataset"], r["n_real"], r["sampling"]) != (ds, str(n), sm) or r["gen_status"] != "ok":
                continue
            try:
                e, c = float(r["quality_err"]), float(r["pod_ceiling_Ek"])
            except ValueError:
                continue
            d = by_trunc.setdefault(r["trunc"], {"e": [], "c": []})
            d["e"].append(e); d["c"].append(c)
    if not by_trunc:
        return FALLBACK[ds]
    stats = {t: (sum(d["e"]) / len(d["e"]), sum(d["c"]) / len(d["c"])) for t, d in by_trunc.items()}
    faithful = {t: v for t, v in stats.items() if v[0] <= RULE_ERROR}
    if faithful:
        return max(faithful.items(), key=lambda kv: (kv[1][1], -kv[1][0]))[0]      # highest ceiling
    return min(stats.items(), key=lambda kv: kv[1][0])[0]                          # least unfaithful


def conditions(o):
    """(dataset, n, arms) of stage B; an arm is (generator, fraction, epochs)."""
    C = []
    ladder = [("galerkin_ns", 0.5, None), ("galerkin_ns", 0.8, None),
              ("jitter", 0.8, None), ("real_proj", 0.8, None)]
    for n in o["n"]:
        C.append(("Re50", n, ladder))
    for ds in ("Re80", "Re100"):
        if 50 in o["n"]:
            C.append((ds, 50, [("galerkin_ns", 0.5, None), ("galerkin_ns", 0.8, None)]))
    return [c for c in C if c[0] in o["datasets"]]


def stage_b(o):
    def bkey(r):
        return (r["dataset"], r["n_real"], r["sampling"], r["subset"], r["kind"], r["generator"],
                r["fraction"], r["epochs"], r["trunc"])
    done = done_rows(TRAIN_CSV, bkey)
    base_done, long_done = {}, set()
    if os.path.exists(TRAIN_CSV):
        for r in csv.DictReader(open(TRAIN_CSV, newline="")):
            cell = (r["dataset"], r["n_real"], r["sampling"], r["subset"])
            if r["kind"] == "baseline":
                base_done[cell] = float(r["Ek"])          # real only: does not depend on the truncation
            elif r["kind"] == "baseline_long":
                long_done.add(cell)
    conds = conditions(o)
    sm = o["sampling"]
    samples = 0
    for ds, n, arms in conds:
        for sub in range(o["subsets"]):
            if (ds, str(n), sm, str(sub)) not in base_done:
                samples += n
            for gen, f, ep in arms:
                if (ds, str(n), sm, str(sub), "aug", gen, str(round(f, 4)), str(ep or o["epochs"]),
                        pick_trunc(ds, n, sm)) not in done:
                    samples += n + n_aug_for(f, n)
        if ds == "Re50" and n == 50:                       # compute control
            for sub in range(o["subsets"]):
                if (ds, str(n), sm, str(sub)) not in long_done:
                    samples += n * LONG_EPOCHS / o["epochs"]
    print(f"[low]  stage B: about {samples * SEC_PER_SAMPLE * o['epochs'] / 500 / 3600:.1f} h of "
          f"training  ->  {TRAIN_CSV}", flush=True)
    if o["plan"] or not conds:
        return

    T0 = time.time()
    by_ds = collections.OrderedDict()
    for ds, n, arms in conds:
        by_ds.setdefault(ds, []).append((n, arms))
    for ds, items in by_ds.items():
        data, re_, dt, path = load(ds)
        Nt = len(data)
        val_idx, pool_idx, _ = split_indices(Nt, None, "random", SPLIT_SEED)
        pool_set = set(int(i) for i in pool_idx)
        val_real = data[val_idx]
        steps = max(1, int(round(GEN_TIME / dt)))
        for n, arms in items:
            tname = pick_trunc(ds, n, sm)
            trunc = [t for t in TRUNCS if t[0] == tname][0]
            for sub in range(o["subsets"]):
                rng = np.random.default_rng([DRAW_SEED, int(re_), n, sub] + ([] if sm == "blocks" else [911]))
                tr_idx = draw_subset(sm, pool_set, pool_idx, Nt, n, rng)
                train_real = data[tr_idx]
                win = fidelity_windows(set(int(i) for i in tr_idx), Nt, steps, rng)
                P = subset_pod(train_real)
                K = k_of(P, trunc, n)
                windows = [data[t:t + steps + 1] for t in win]
                ceiling, win_A = ceiling_and_project(P, K, val_real, windows)
                A = np.asarray(P["A"][:, :K], dtype=np.float64)
                info = dict(dataset=ds, n_real=n, sampling=sm, subset=sub, latent=LATENT[ds], trunc=tname, K=K,
                            energy_K_pct=round(100 * P["e_cum"][K - 1], 3), pod_ceiling_Ek=round(ceiling, 3))
                print(f"\n[low]  ===== {ds} {n} real ({sm}) subset {sub}: {tname} K={K}, ceiling "
                      f"{ceiling:.1f}%, {len(win_A)} quality windows =====", flush=True)

                key = (ds, str(n), sm, str(sub))
                if key not in base_done:
                    ek, detR, _, wall = vae.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32),
                                                  val_real, tag=f"{ds} n={n} s{sub} real only",
                                                  latent=LATENT[ds], epochs=o["epochs"])
                    base_done[key] = float(ek)
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="baseline", n_train=n,
                                                     epochs=o["epochs"], Ek=round(float(ek), 4),
                                                     detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
                    print(f"[low]  real-only Ek {ek:.2f}%  (headroom {ceiling - float(ek):.1f})", flush=True)
                base = base_done.get(key)

                if ds == "Re50" and n == 50 and key not in long_done:
                    ek, detR, _, wall = vae.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32),
                                                  val_real, tag=f"{ds} n={n} s{sub} real only {LONG_EPOCHS} epochs",
                                                  latent=LATENT[ds], epochs=LONG_EPOCHS)
                    long_done.add(key)
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="baseline_long", n_train=n,
                                                     epochs=LONG_EPOCHS, Ek=round(float(ek), 4),
                                                     baseline_Ek=round(base, 4) if base else "",
                                                     gain=round(float(ek) - base, 3) if base else "",
                                                     detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
                    print(f"[low]  real-only Ek with {LONG_EPOCHS} epochs: {ek:.2f}%", flush=True)

                qerr = None
                step = s = to_b = from_b = None
                if any(g == "galerkin_ns" for g, _, _ in arms):
                    l_big, q_big, kappa = build_ops(P, K, re_, path)
                    step, s, to_b, from_b = model_at(l_big, q_big, kappa, A, K, dt, SUBSTEPS)
                    qerr, _ = fidelity(step, to_b, from_b, win_A, steps) if win_A else (None, "")
                    del l_big, q_big
                    print(f"[low]  quality error {'-' if qerr is None else f'{qerr:.3f}'}", flush=True)

                for gen, f, ep in arms:
                    ep = ep or o["epochs"]
                    if (ds, str(n), sm, str(sub), "aug", gen, str(round(f, 4)), str(ep), tname) in done:
                        continue
                    na = n_aug_for(f, n)
                    t0 = time.time()
                    try:
                        if gen == "galerkin_ns":
                            aug, st = synthetic_arm(step, s, to_b, from_b, P, A, K, na, steps,
                                                    np.random.default_rng([DRAW_SEED, int(re_), n, sub, K]),
                                                    kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ)
                        elif gen == "jitter":
                            grng = np.random.default_rng([DRAW_SEED, int(re_), n, sub, K, 7777])
                            aug = jitter_arm(P, A, K, na, grng, kick=IC_NOISE)
                        else:
                            grng = np.random.default_rng([DRAW_SEED, int(re_), n, sub, K, 7777])
                            aug, na = real_arm(P, K, data, pool_idx, tr_idx, na, grng, project=True, cap=True)
                    except Exception as err:
                        msg = str(err).splitlines()[0][:160]
                        print(f"[low]  !! {gen} f={f}: generation failed ({msg})", flush=True)
                        append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="gen_failed", generator=gen,
                                                         fraction=round(f, 4), epochs=ep, status=f"failed: {msg}"))
                        continue
                    headroom = ceiling - base
                    pred = "" if (gen != "galerkin_ns" or qerr is None) else \
                        ("gain" if (qerr <= RULE_ERROR and headroom >= RULE_HEAD) else "no gain")
                    ek, detR, _, wall = vae.train(train_real, aug[:na], val_real,
                                                  tag=f"{ds} n={n} s{sub} {gen} f={f:.2f} K={K}",
                                                  latent=LATENT[ds], epochs=o["epochs"])
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="aug", generator=gen,
                                                     fraction=round(f, 4), epochs=ep, n_aug=na, n_train=n + na,
                                                     baseline_Ek=round(base, 4), headroom=round(headroom, 3),
                                                     quality_err=round(qerr, 5) if qerr is not None else "",
                                                     predicted=pred, Ek=round(float(ek), 4),
                                                     gain=round(float(ek) - base, 3), detR=round(float(detR), 6),
                                                     status="ok", wall_s=round(wall)))
                    print(f"[low]  {gen} f={f:.2f}: Ek {ek:.2f}%  gain {float(ek) - base:+.2f}  "
                          f"(predicted {pred or '-'}, {time.time() - t0:.0f} s)   [{(time.time() - T0) / 3600:.1f} h]",
                          flush=True)
                    del aug
                del P, windows, train_real
        del data, val_real
    print(f"\n[low]  stage B done in {(time.time() - T0) / 3600:.2f} h  ->  {TRAIN_CSV}")


def main():
    o = _args(sys.argv[1:])
    if "A" in o["stage"]:
        stage_a(o)
    if "B" in o["stage"]:
        stage_b(o)


if __name__ == "__main__":
    main()

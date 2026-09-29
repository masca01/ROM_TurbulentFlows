"""
DOES THE EDDY-VISCOSITY CLOSURE ALSO WORK ON THE 2-PLATE WAKES?

The channel (study 7, round 2) showed that an eddy viscosity nu_T on the fluctuation modes turns the
NS-projected model from "no better than a frozen flow" into a faithful one. This study asks the same of
the wakes, where every earlier result was obtained WITHOUT a closure, and answers it on the SAME cells:
each cell below is reproduced exactly (same real subset, same held-out windows, same modes, same
synthetic-data seed as the stored run), so the only change is the closure term

    da_i/dt = l_ij c_j + q_ijk c_j c_k  +  nu_T * Lambda_ij a_j     (j >= 1: fluctuation modes only)

Cells (stored plain-NS results in brackets: quality error / gain in Ek):
    Re100_n100_99   Re100, 100 real, latent 15, 99 % modes, f = 0.5, 3 subsets  (0.46 / -1.4)  region map
    Re100_n100_95   Re100, 100 real, latent 15, 95 % modes, f = 0.5, 3 subsets  (0.51 / -0.9)  region map
    Re100_n250_99   Re100, 250 real, latent 15, 99 % modes, f = 0.5, 3 subsets  (0.28 / +3.1)  region map
    Re100_n400_N4   Re100, 400 real, latent 25, 99 % modes, f = 2/3, 5 subsets  (0.20 / +6.8)  round 2, N4
    Re80_n100_99    Re80,  100 real, latent 11, 99 % modes, f = 0.5, 3 subsets  (0.44 / -0.1)  control: 10 points
                    of headroom, so even a perfect model should not gain
The Re100 n = 100 cells are the ones the closure could rescue: 21-27 points of headroom, lost because
the error sat at the 0.5 limit.

STAGE A - screen (no network, minutes). Per cell and subset: the plain model is rebuilt and its quality
error must equal the stored one (reproduction check), then nu_T / nu in {0, 0.03, 0.1, 0.3, 1, 3, 10, 30}
(nu = 1/Re), quality over 10 convective times as in the region map, plus the "nothing changes" error.
    -> convergence/closure_wake_screen.csv

STAGE B - trainings. Per cell: nu_T = the ratio with the lowest mean error over the subsets (a cell whose
best is nu_T = 0 is not trained: the closure does not help it). Per subset, the NS arm with the closure,
same K, fraction and synthetic seed as the stored plain arm; its gain uses the stored real-only Ek.
Reproduction check: on subset 0 of every cell the real-only and the plain-NS arms are retrained; they
must give back the stored Ek (--check-all: on every subset).
Written before training, per row:
    predicted  = gain if closure error <= 0.5 and corrected headroom >= 20 (region-map rule)
    pred_delta = change of gain from the round-1 dose-response, +4.8 points per decade of lower error
    -> convergence/closure_wake_results.csv

Usage (from the repository folder):
    python3 studies/8_closure_wake/run_closure_wake.py --plan        # what is left and the time
    python3 studies/8_closure_wake/run_closure_wake.py               # A then B (or --stage A / --stage B)
    --cells Re100_n100_99,Re80_n100_99   --epochs N   --check-all   --suffix smoke
"""
import os, sys, csv, time, collections
import numpy as np

from rom import paths
from rom import vae
from rom.data import load
from rom.split import split_indices, draw_blocks, SPLIT_SEED, DRAW_SEED
from rom.pod import subset_pod, ceiling_and_project, k_for
from rom.galerkin_ns import build_ops, model_at
from rom.channel import persistence, add_eddy_viscosity
from rom.augment import (n_aug_for, fidelity, fidelity_windows, synthetic_arm,
                         GEN_TIME, SUBSTEPS, IC_NOISE, ENERGY_TOL, MAX_TRAJ)
from rom.vae import EPOCHS, SEC_PER_SAMPLE
from rom.results import append_row as append, done_rows, now

# ============ CONFIG ============
CELLS = {
    "Re100_n100_99": dict(ds="Re100", n=100, lat=15, lv=0.99, f=0.5, nsub=3, src="map"),
    "Re100_n100_95": dict(ds="Re100", n=100, lat=15, lv=0.95, f=0.5, nsub=3, src="map"),
    "Re100_n250_99": dict(ds="Re100", n=250, lat=15, lv=0.99, f=0.5, nsub=3, src="map"),
    "Re100_n400_N4": dict(ds="Re100", n=400, lat=25, lv=0.99, f=2 / 3, nsub=5, src="N4"),
    "Re80_n100_99":  dict(ds="Re80", n=100, lat=11, lv=0.99, f=0.5, nsub=3, src="map"),
}
NU_RATIO   = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]   # nu_T / nu, nu = 1/Re (10, 30: the Re100 optimum lies past 3)
RULE_ERROR = 0.5
RULE_HEAD  = 20.0
DOSE       = 4.8                                  # points of gain per decade of quality error (round 1, E1)
NS_GIDX    = 1                                    # index of galerkin_ns in the stored synthetic seeds
MAP_CSV    = os.path.join(paths.RESULTS, "region_map_results.csv")
EXP2_CSV   = os.path.join(paths.RESULTS, "experiments2_results.csv")
CAP_CSV    = os.path.join(paths.RESULTS, "capacity_ceilings.csv")
SCREEN_CSV = os.path.join(paths.RESULTS, "closure_wake_screen.csv")
TRAIN_CSV  = os.path.join(paths.RESULTS, "closure_wake_results.csv")
CAP_FALLBACK = {("Re100", 15): 89.6, ("Re100", 25): 92.2, ("Re80", 11): 98.9, ("Re50", 5): 98.3}
# ================================

A_FIELDS = ["date", "cell", "dataset", "n_real", "subset", "latent", "K", "K_stored", "pod_ceiling_Ek",
            "nu_ratio", "nu_t", "quality_err", "stored_err", "reproduced", "persist_err", "quality_blowup_frac",
            "quality_windows", "seconds"]
B_FIELDS = ["date", "cell", "dataset", "n_real", "subset", "kind", "latent", "K", "fraction", "epochs", "nu_ratio",
            "nu_t", "quality_err", "stored_err", "pod_ceiling_Ek", "capacity_Ek", "baseline_Ek", "headroom_corrected",
            "predicted", "pred_delta", "Ek", "stored_Ek", "gain", "stored_gain", "delta_gain", "reproduced",
            "gen_keep_frac", "status", "wall_s"]


def _args(argv):
    o = {"plan": False, "stage": "AB", "cells": list(CELLS), "epochs": EPOCHS, "check_all": False}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--stage": o["stage"] = next(it).upper()
        elif a == "--cells": o["cells"] = next(it).split(",")
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--check-all": o["check_all"] = True
        elif a == "--suffix":
            global SCREEN_CSV, TRAIN_CSV
            suf = next(it)
            SCREEN_CSV = SCREEN_CSV.replace(".csv", f"_{suf}.csv")
            TRAIN_CSV = TRAIN_CSV.replace(".csv", f"_{suf}.csv")
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    bad = [c for c in o["cells"] if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cell(s) {bad}; choose from {list(CELLS)}")
    return o


def stored():
    """(cell, subset) -> stored plain-NS row: quality error, K, baseline Ek, Ek, gain (None where absent)."""
    out = {}
    rows_map = list(csv.DictReader(open(MAP_CSV, newline=""))) if os.path.exists(MAP_CSV) else []
    rows_n4 = list(csv.DictReader(open(EXP2_CSV, newline=""))) if os.path.exists(EXP2_CSV) else []
    for cid, c in CELLS.items():
        for r in rows_map if c["src"] == "map" else rows_n4:
            if r["kind"] != "aug" or r["generator"] != "galerkin_ns" or r["dataset"] != c["ds"] \
                    or r["n_real"] != str(c["n"]) or abs(float(r["mode_level"]) - c["lv"]) > 1e-9 \
                    or abs(float(r["fraction"]) - c["f"]) > 1e-3:
                continue
            if c["src"] == "N4" and ("N4" not in r["exps"] or r["latent"] != str(c["lat"]) or r["subset_mode"] != "blocks"):
                continue
            err = r["fid_err"] if c["src"] == "map" else r["quality_err"]
            out[(cid, int(r["subset"]))] = dict(err=float(err), K=int(r["K"]), base=float(r["baseline_Ek"]),
                                                Ek=float(r["Ek"]), gain=float(r["gain"]))
    return out


def capacity(ds, lat):
    if os.path.exists(CAP_CSV):
        for r in csv.DictReader(open(CAP_CSV, newline="")):
            if r.get("dataset") == ds and str(r.get("latent")) == str(lat) and r.get("Ek"):
                return float(r["Ek"])
    return CAP_FALLBACK.get((ds, lat), 100.0)


class Data:
    """One dataset loaded once: data, Re, dt, split, quality horizon."""
    _cache = {}

    @classmethod
    def get(cls, ds):
        if ds not in cls._cache:
            cls._cache.clear()                                   # one wake dataset in memory at a time
            d = cls.__new__(cls)
            d.data, d.re, d.dt, d.path = load(ds)
            d.Nt = len(d.data)
            d.val_idx, d.pool_idx, _ = split_indices(d.Nt, None, "random", SPLIT_SEED)
            d.pool_set = set(int(i) for i in d.pool_idx)
            d.val_real = d.data[d.val_idx]
            d.steps = max(1, int(round(GEN_TIME / d.dt)))
            cls._cache[ds] = d
        return cls._cache[ds]


def cell_setup(c, sub):
    """The stored cell, rebuilt: same draw, windows, POD, K and NS operators as run_region_map.py /
    run_round2.py. Returns (D, tr_idx, P, K, ceiling, win_A, l, q, kappa, lap, A)."""
    D = Data.get(c["ds"])
    rng = np.random.default_rng([DRAW_SEED, int(D.re), c["n"], sub])
    tr_idx = draw_blocks(D.pool_set, D.Nt, c["n"], rng)
    win = fidelity_windows(set(int(i) for i in tr_idx), D.Nt, D.steps, rng)
    P = subset_pod(D.data[tr_idx])
    K = k_for(P, c["lv"])
    ceiling, win_A = ceiling_and_project(P, K, D.val_real, [D.data[t:t + D.steps + 1] for t in win])
    l, q, kappa, lap = build_ops(P, K, D.re, D.path, return_lap=True)
    A = np.asarray(P["A"][:, :K], dtype=np.float64)
    return D, tr_idx, P, K, ceiling, win_A, l, q, kappa, lap, A


def model(D, l, q, kappa, lap, A, K, ratio):
    return model_at(add_eddy_viscosity(l, lap, ratio / D.re), q, kappa, A, K, D.dt, SUBSTEPS)


# ------------------------------------------------------------------ stage A

def stage_a(o):
    ST = stored()
    done = done_rows(SCREEN_CSV, lambda r: (r["cell"], r["subset"], r["nu_ratio"]))
    todo = [(cid, sub) for cid in o["cells"] for sub in range(CELLS[cid]["nsub"])
            if any((cid, str(sub), str(x)) not in done for x in NU_RATIO)]
    print(f"[clw]  stage A: {len(todo)} cell subsets to screen, a few minutes each  ->  {SCREEN_CSV}", flush=True)
    if o["plan"] or not todo:
        return
    for cid, sub in sorted(todo, key=lambda t: CELLS[t[0]]["ds"]):
        c = CELLS[cid]; t0 = time.time()
        D, tr_idx, P, K, ceiling, win_A, l, q, kappa, lap, A = cell_setup(c, sub)
        st = ST.get((cid, sub))
        pe = persistence(win_A, D.steps) if win_A else float("nan")
        print(f"\n[clw]  {cid} s{sub}: K = {K} (stored {st['K'] if st else '-'}), ceiling {ceiling:.1f}%, "
              f"{len(win_A)} windows, no change {pe:.2f}  [{time.time() - t0:.0f} s]", flush=True)
        for x in NU_RATIO:
            if (cid, str(sub), str(x)) in done:
                continue
            step, s, to_b, from_b = model(D, l, q, kappa, lap, A, K, x)
            err, blow = fidelity(step, to_b, from_b, win_A, D.steps) if win_A else (float("nan"), "")
            rep = ""
            if x == 0 and st:
                rep = "yes" if (K == st["K"] and abs(round(err, 5) - st["err"]) < 1e-4) else "NO"
            append(SCREEN_CSV, A_FIELDS, dict(
                date=now(), cell=cid, dataset=c["ds"], n_real=c["n"], subset=sub, latent=c["lat"], K=K,
                K_stored=st["K"] if st else "", pod_ceiling_Ek=round(ceiling, 3), nu_ratio=x, nu_t=x / D.re,
                quality_err=round(err, 5), stored_err=st["err"] if st else "", reproduced=rep,
                persist_err=round(pe, 5), quality_blowup_frac=blow, quality_windows=len(win_A),
                seconds=round(time.time() - t0)))
            print(f"[clw]      nu_T = {x:4g} nu: error {err:.3f}" + (f"  (stored {st['err']:.3f}: reproduced {rep})"
                                                                        if x == 0 and st else ""), flush=True)
        del P, l, q, lap, A


# ------------------------------------------------------------------ stage B

def best_ratio(cid):
    acc = collections.defaultdict(list)
    if os.path.exists(SCREEN_CSV):
        for r in csv.DictReader(open(SCREEN_CSV, newline="")):
            if r["cell"] == cid and r["quality_err"]:
                acc[float(r["nu_ratio"])].append(float(r["quality_err"]))
    n = CELLS[cid]["nsub"]
    full = {x: np.mean(v) for x, v in acc.items() if len(v) == n}
    if not full:
        return None, {}
    return min(full, key=full.get), full


def bkey(r):
    return (r["cell"], r["subset"], r["kind"], r["epochs"])


def stage_b(o):
    ep = o["epochs"]
    ST = stored()
    done = done_rows(TRAIN_CSV, bkey)
    plan, samples = [], 0
    print(f"[clw]  stage B ({ep} epochs):")
    for cid in o["cells"]:
        c = CELLS[cid]
        x, means = best_ratio(cid)
        if x is None:
            print(f"          {cid:15s} no complete screen yet - run stage A first"); continue
        line = "  ".join(f"{k:g}: {v:.3f}" for k, v in sorted(means.items()))
        if x == 0:
            print(f"          {cid:15s} best nu_T = 0: the closure does not lower the error ({line}) - not trained")
            continue
        print(f"          {cid:15s} nu_T = {x:g} nu  (mean error by nu_T/nu  {line})")
        plan.append((cid, x))
        for sub in range(c["nsub"]):
            na = n_aug_for(c["f"], c["n"])
            if (cid, str(sub), "closure", str(ep)) not in done:
                samples += c["n"] + na
            if o["check_all"] or sub == 0:
                if (cid, str(sub), "check_baseline", str(ep)) not in done:
                    samples += c["n"]
                if (cid, str(sub), "check_plain", str(ep)) not in done:
                    samples += c["n"] + na
    print(f"[clw]  stage B: about {samples * SEC_PER_SAMPLE * ep / 500 / 3600:.1f} h of training  ->  {TRAIN_CSV}",
          flush=True)
    if o["plan"] or not plan:
        return

    T0 = time.time()
    for cid, x in sorted(plan, key=lambda t: CELLS[t[0]]["ds"]):
        c = CELLS[cid]
        cap = capacity(c["ds"], c["lat"])
        for sub in range(c["nsub"]):
            kinds = ["closure"] + (["check_baseline", "check_plain"] if (o["check_all"] or sub == 0) else [])
            kinds = [k for k in kinds if (cid, str(sub), k, str(ep)) not in done]
            if not kinds:
                continue
            D, tr_idx, P, K, ceiling, win_A, l, q, kappa, lap, A = cell_setup(c, sub)
            st = ST.get((cid, sub)) or {}
            base = st.get("base")
            train = D.data[tr_idx]
            na = n_aug_for(c["f"], c["n"])
            info = dict(cell=cid, dataset=c["ds"], n_real=c["n"], subset=sub, latent=c["lat"], K=K,
                        fraction=round(c["f"], 4), epochs=ep, pod_ceiling_Ek=round(ceiling, 3), capacity_Ek=cap,
                        baseline_Ek=round(base, 4) if base is not None else "", stored_err=st.get("err", ""))
            print(f"\n[clw]  ===== {cid} subset {sub}: K = {K}, stored plain error {st.get('err', '-')}, "
                  f"stored gain {st.get('gain', '-')} =====", flush=True)
            for kind in kinds:
                t0 = time.time()
                if kind == "check_baseline":
                    ek, detR, _, wall = vae.train(train, np.empty((0,) + train.shape[1:], np.float32), D.val_real,
                                                  tag=f"{cid} s{sub} real only (check)", latent=c["lat"], epochs=ep)
                    rep = "" if base is None else ("yes" if abs(float(ek) - base) < 0.05 else "NO")
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind=kind, Ek=round(float(ek), 4),
                                                     stored_Ek=round(base, 4) if base is not None else "",
                                                     reproduced=rep, status="ok", wall_s=round(wall)))
                    print(f"[clw]  real only Ek {ek:.2f}% (stored {base}: reproduced {rep or '-'})", flush=True)
                    continue
                ratio = 0.0 if kind == "check_plain" else x
                step, s, to_b, from_b = model(D, l, q, kappa, lap, A, K, ratio)
                qerr, _ = fidelity(step, to_b, from_b, win_A, D.steps) if win_A else (None, "")
                head = (min(ceiling, cap) - base) if base is not None else None
                pred = pdelta = ""
                if kind == "closure" and qerr is not None:
                    pred = "gain" if (qerr <= RULE_ERROR and head is not None and head >= RULE_HEAD) else "no gain"
                    if st.get("err"):
                        pdelta = round(DOSE * np.log10(st["err"] / max(qerr, 1e-9)), 2)
                try:
                    aug, gst = synthetic_arm(step, s, to_b, from_b, P, A, K, na, D.steps,
                                             np.random.default_rng([DRAW_SEED, int(D.re), c["n"], sub, K, NS_GIDX]),
                                             kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ)
                except Exception as err:
                    msg = str(err).splitlines()[0][:160]
                    print(f"[clw]  !! {kind}: generation failed ({msg})", flush=True)
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind=kind, nu_ratio=ratio, nu_t=ratio / D.re,
                                                     quality_err=round(qerr, 5) if qerr is not None else "",
                                                     predicted=pred, pred_delta=pdelta, status=f"failed: {msg}"))
                    continue
                if kind == "closure":
                    print(f"[clw]  closure nu_T = {ratio:g} nu: error {qerr:.3f} (plain {st.get('err', '-')}), "
                          f"corrected headroom {head if head is None else round(head, 1)}  ->  predicted {pred}, "
                          f"change of gain {pdelta if pdelta != '' else '-'}", flush=True)
                ek, detR, _, wall = vae.train(train, aug[:na], D.val_real, tag=f"{cid} s{sub} {kind} nu_T {ratio:g} nu",
                                              latent=c["lat"], epochs=ep)
                gain = float(ek) - base if base is not None else None
                sg = st.get("gain")
                rep = ""
                if kind == "check_plain" and st.get("Ek") is not None:
                    rep = "yes" if abs(float(ek) - st["Ek"]) < 0.05 else "NO"
                append(TRAIN_CSV, B_FIELDS, dict(
                    info, date=now(), kind=kind, nu_ratio=ratio, nu_t=ratio / D.re,
                    quality_err=round(qerr, 5) if qerr is not None else "",
                    headroom_corrected=round(head, 3) if head is not None else "", predicted=pred, pred_delta=pdelta,
                    Ek=round(float(ek), 4), stored_Ek=st.get("Ek", ""), gain=round(gain, 3) if gain is not None else "",
                    stored_gain=sg if sg is not None else "",
                    delta_gain=round(gain - sg, 3) if (gain is not None and sg is not None and kind == "closure") else "",
                    reproduced=rep, gen_keep_frac=round(gst["keep_frac"], 4), status="ok", wall_s=round(wall)))
                print(f"[clw]  {kind}: Ek {ek:.2f}%" + (f"  gain {gain:+.2f}" if gain is not None else "")
                      + (f"  (plain stored {sg:+.2f})" if sg is not None else "")
                      + (f"  reproduced {rep}" if rep else "") + f"   [{(time.time() - T0) / 3600:.1f} h]", flush=True)
                del aug
            del P, l, q, lap, A, train
    print(f"\n[clw]  stage B done in {(time.time() - T0) / 3600:.2f} h  ->  {TRAIN_CSV}")


def main():
    o = _args(sys.argv[1:])
    if "A" in o["stage"]:
        stage_a(o)
    if "B" in o["stage"]:
        stage_b(o)


if __name__ == "__main__":
    main()

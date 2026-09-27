"""
CHANNEL FLOW, ROUND 2: the NS-projected generator only, with everything that could make it work.

Round 1 (run_channel.py) left the NS-projected model no better than "nothing changes" (best: n = 500,
K = 10, error 0.21 vs 0.22) and no generator cell gained, while plain jitter at K = 10 gained
+2.8 +- 1.1 on 3/3 subsets. This round changes the model and the way its data are used:

  1. Physics fixes ("phys" variant, against the round-1 "plain" model):
       - mean = x-average of the subset mean (the channel is homogeneous in x). A few hundred
         snapshots freeze turbulent structures into the plain temporal mean; they then act as a
         steady forcing in the projected equations.
       - periodic x-derivatives: the full plane is exactly one period long (8 pi), np.gradient's
         one-sided edges are wrong there.
  2. Closure: eddy viscosity nu_T on the fluctuation modes (Aubry et al. 1988), nu_T in
     {0, 3e-4, 1e-3, 2e-3, 3e-3} (nu = 5e-5; 3e-3 ~ 0.06 u_tau h, the usual outer-layer value). It drains
     the energy the truncated model cannot pass to the missing small scales, the reason a
     truncated Galerkin model blows up. First grid {0, 1e-3, 3e-3, 1e-2} on one n = 500 subset:
     1e-3 best at every K, 1e-2 far too strong, so the grid was refined around 1e-3.
  3. More modes: K = 10, 20, 40, 80, 120 (round 1 stopped at 40).
  4. Random real subsets (the NS model needs no consecutive snapshots; round 1 used blocks for the
     data-identified fit): higher POD ceilings and a better real-only baseline for the same n.
  5. Synthetic trajectories of 0.25 h/U_b besides 2: over a short time the model mostly advects a
     real state by the mean flow, so its snapshots are real states moved in a physical way, where
     jitter only adds noise. Fractions 0.5 and 0.8. n = 500 and 1000.
Everything else as round 1: latent 16, 500 epochs, the same split, the quality windows in the last
25 % of the record, the rule relative to "nothing changes" (faithful: error <= 0.5 x no-change).

STAGE A - screen, no network (~12-20 min per subset: K = 120 is the cost; ~1.5 h in all).
    n in {500, 1000} x 3 random subsets x {plain, phys} x nu_T x K: POD ceiling, quality error
    over 2 h/U_b and over 0.25 h/U_b with their "nothing changes" errors, generation check.
    -> convergence/channel_ns_screen.csv

STAGE B - trainings, chosen by the screen (per n, over the 3 subsets):
    best_long  = (variant, nu_T, K) with the lowest error / no-change ratio over 2 h/U_b
    best_short = the same over 0.25 h/U_b
    best_modes = the best at the largest K screened (120): "more modes" (skipped if it is best_long)
    n = 500, per subset:
        real only | real only at 1000 epochs (control: does the network just need more training?)
        ns_long   best_long,  trajectories of 2 h/U_b,    f = 0.5
        ns_long8  best_long,  2 h/U_b,                    f = 0.8
        ns_short  best_short, trajectories of 0.25 h/U_b, f = 0.5
        ns_short8 best_short, 0.25 h/U_b,                 f = 0.8
        ns_modes  best_modes (K = 120), 2 h/U_b,          f = 0.5
        jitter    at best_long's modes, f = 0.5          (noise, no dynamics: round 1's +2.8)
        real_proj at best_long's modes, f = 0.5          (a perfect model: the upper bound)
    n = 1000, per subset: real only | ns_long | ns_short | jitter
    Capacity at latent 16 is round 1's measurement (65.4 %, channel_results.csv).
    Every NS row carries its prediction, written before training: gain if its error <= 0.5 x
    no-change at its own horizon and the corrected headroom >= 20.
    -> convergence/channel_ns_results.csv

Expectation, written before any training (after the test subset of the screen): the long-horizon
arms gain. With nu_T = 1e-3 the model is faithful at every K (error / no-change 0.33-0.47 on the test
subset: physics fixes + closure, K = 40-120), the first faithful NS model on the channel, and the POD
ceiling at K = 80-120 is 56-61 %, well above the real-only Ek of round 1 (25 % at n = 500, blocks).
The short-horizon arms should behave like jitter or slightly better (+1 to +3): a real state advected
for 0.25 h/U_b is a realistic new snapshot but little new information. More modes help only through
the higher ceiling, and only with the closure (without it the error grows with K).

Usage (from the repository folder):
    python3 studies/7_channel/run_channel_ns.py --plan        # what is left and the time
    python3 studies/7_channel/run_channel_ns.py               # A then B (or --stage A / --stage B)
    --epochs N (all trainings)   --n 500  --subsets 1  --suffix smoke   quick checks
"""
import os, sys, csv, time, collections
import numpy as np

from rom import paths
from rom import vae
from rom.data import CHANNEL_DPDX
from rom.split import DRAW_SEED
from rom.channel import Setup, persistence, channel_pod, add_eddy_viscosity, SEED_TAG, SAMPLINGS
from rom.pod import ceiling_and_project
from rom.galerkin_ns import build_ops, model_at, integrate_trajectories
from rom.augment import (n_aug_for, fidelity, synthetic_arm, jitter_arm, real_arm,
                         SUBSTEPS, IC_NOISE, ENERGY_TOL, MAX_TRAJ)
from rom.vae import EPOCHS
from rom.results import append_row as append, done_rows, now

# ============ CONFIG ============
DATASET     = "chan"
LATENT      = 16
N_REAL      = [500, 1000]
N_SUBSETS   = 3
SAMPLING    = "random"
VARIANTS    = {"plain": (False, False), "phys": (True, True)}   # (x-homogeneous mean, periodic x)
NU_T        = [0.0, 3e-4, 1e-3, 2e-3, 3e-3]   # refined around 1e-3 after the test subset (1e-2 too strong)
KS          = [10, 20, 40, 80, 120]
K_MAX       = 120
SHORT_TIME  = 0.25                      # h/U_b: short quality windows and short trajectories
LONG_TIME   = 2.0                       # h/U_b: the round-1 quality horizon (rom.channel.GEN_TIME)
RULE_SKILL  = 0.5
RULE_HEAD   = 20.0
LONG_EPOCHS = 1000                      # control: real only at n = 500
SEC_PER_SAMPLE_CH = 0.74                # measured in round 1 (4.58 h for what 1.3 s/sample called 8.0 h)
SCREEN_CSV  = os.path.join(paths.RESULTS, "channel_ns_screen.csv")
TRAIN_CSV   = os.path.join(paths.RESULTS, "channel_ns_results.csv")
ROUND1_CSV  = os.path.join(paths.RESULTS, "channel_results.csv")   # capacity at latent 16
# arm: (label, which best config, horizon, fraction); the n = 1000 cells only the lighter ladder
ARMS = {500:  [("ns_long", "long", LONG_TIME, 0.5), ("ns_long8", "long", LONG_TIME, 0.8),
               ("ns_short", "short", SHORT_TIME, 0.5), ("ns_short8", "short", SHORT_TIME, 0.8),
               ("ns_modes", "modes", LONG_TIME, 0.5),
               ("jitter", "long", None, 0.5), ("real_proj", "long", None, 0.5)],
        1000: [("ns_long", "long", LONG_TIME, 0.5), ("ns_short", "short", SHORT_TIME, 0.5),
               ("jitter", "long", None, 0.5)]}
# ================================

A_FIELDS = ["date", "dataset", "n_real", "sampling", "subset", "variant", "nu_t", "K", "energy_K_pct",
            "pod_ceiling_Ek", "quality_err", "persist_err", "ratio", "quality_err_short", "persist_err_short",
            "ratio_short", "quality_blowup_frac", "quality_windows", "gen_keep_frac", "gen_status", "seconds"]
B_FIELDS = ["date", "dataset", "n_real", "sampling", "subset", "kind", "arm", "latent", "variant", "nu_t", "K",
            "horizon", "energy_K_pct", "fraction", "epochs", "n_aug", "n_train", "pod_ceiling_Ek", "capacity_Ek",
            "baseline_Ek", "headroom_corrected", "quality_err", "persist_err", "ratio", "predicted", "Ek", "gain",
            "detR", "gen_keep_frac", "status", "wall_s"]


def _args(argv):
    o = {"plan": False, "stage": "AB", "n": N_REAL, "subsets": N_SUBSETS, "epochs": EPOCHS}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--stage": o["stage"] = next(it).upper()
        elif a == "--n": o["n"] = [int(x) for x in next(it).split(",")]
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--suffix":
            global SCREEN_CSV, TRAIN_CSV
            suf = next(it)
            SCREEN_CSV = SCREEN_CSV.replace(".csv", f"_{suf}.csv")
            TRAIN_CSV = TRAIN_CSV.replace(".csv", f"_{suf}.csv")
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


def ops_for(S, P, variant, Kb):
    """Projected operators of one variant at Kb modes, with the Laplacian for the eddy viscosity."""
    _, periodic = VARIANTS[variant]
    return build_ops(P, Kb, S.re, S.path, forcing_x=CHANNEL_DPDX, periodic_x=periodic, return_lap=True)


def steps_of(S, t):
    return max(1, int(round(t / S.dt)))


# ------------------------------------------------------------------ stage A

def stage_a(o):
    done = done_rows(SCREEN_CSV, lambda r: (r["dataset"], r["n_real"], r["subset"], r["variant"], r["nu_t"], r["K"]))
    todo = [(n, sub) for n in o["n"] for sub in range(o["subsets"])
            if any((DATASET, str(n), str(sub), v, str(nt), str(K)) not in done
                   for v in VARIANTS for nt in NU_T for K in KS)]
    print(f"[ns2]  stage A: {len(todo)} subsets to screen, about {len(todo) * 15} min  ->  {SCREEN_CSV}", flush=True)
    if o["plan"] or not todo:
        return
    S = Setup(DATASET, gen_time=LONG_TIME)
    sh = steps_of(S, SHORT_TIME)
    for n, sub in todo:
        tr_idx, windows = S.subset(n, SAMPLING, sub)
        train = S.data[tr_idx]
        for variant, (homog, _) in VARIANTS.items():
            t0 = time.time()
            P = channel_pod(train, homogeneous_x=homog)
            Kb = int(min(K_MAX, P["A"].shape[1], n - 1))
            l, q, kappa, lap = ops_for(S, P, variant, Kb)
            A = np.asarray(P["A"][:, :Kb], dtype=np.float64)
            print(f"\n[ns2]  n={n} s{sub} {variant}: operators at K={Kb} [{time.time() - t0:.0f} s]", flush=True)
            for K in KS:
                base = dict(dataset=DATASET, n_real=n, sampling=SAMPLING, subset=sub, variant=variant, K=K)
                if K > Kb:
                    for nt in NU_T:
                        if (DATASET, str(n), str(sub), variant, str(nt), str(K)) not in done:
                            append(SCREEN_CSV, A_FIELDS, dict(base, date=now(), nu_t=nt,
                                                              gen_status=f"skipped: K={K} above {Kb}"))
                    continue
                ceiling, win_A = ceiling_and_project(P, K, S.val_real, windows)
                win_s = [w[:sh + 1] for w in win_A]
                pe, pe_s = persistence(win_A, S.steps), persistence(win_s, sh)
                for nt in NU_T:
                    if (DATASET, str(n), str(sub), variant, str(nt), str(K)) in done:
                        continue
                    t1 = time.time()
                    step, s, to_b, from_b = model_at(add_eddy_viscosity(l, lap, nt), q, kappa, A, K, S.dt, SUBSTEPS)
                    err, blow = fidelity(step, to_b, from_b, win_A, S.steps)
                    err_s, _ = fidelity(step, to_b, from_b, win_s, sh)
                    try:
                        _, st = integrate_trajectories(step, s, to_b(A[:, :K]), n, S.steps, IC_NOISE, ENERGY_TOL,
                                                       4000, np.random.default_rng([7, K, n]))
                        gen, keep = "ok", round(st["keep_frac"], 4)
                    except Exception as e:
                        gen, keep = str(e).splitlines()[0][:80], ""
                    append(SCREEN_CSV, A_FIELDS, dict(
                        base, date=now(), nu_t=nt, energy_K_pct=round(100 * P["e_cum"][K - 1], 3),
                        pod_ceiling_Ek=round(ceiling, 3), quality_err=round(err, 5), persist_err=round(pe, 5),
                        ratio=round(err / max(pe, 1e-12), 4), quality_err_short=round(err_s, 5),
                        persist_err_short=round(pe_s, 5), ratio_short=round(err_s / max(pe_s, 1e-12), 4),
                        quality_blowup_frac=blow, quality_windows=len(win_A), gen_keep_frac=keep, gen_status=gen,
                        seconds=round(time.time() - t1)))
                    print(f"[ns2]      K={K:3d} ({100 * P['e_cum'][K - 1]:4.1f}%) ceiling {ceiling:5.1f}%  nu_T {nt:6.0e}  "
                          f"2.0: {err:4.2f}/{pe:4.2f}  0.25: {err_s:4.2f}/{pe_s:4.2f}  {gen}  [{time.time() - t1:.0f} s]",
                          flush=True)
            del P, l, q, lap, A


# ------------------------------------------------------------------ stage B

def best_configs(n):
    """{'long' | 'short' | 'modes': (variant, nu_t, K)} from the screen of this n (means over subsets;
    only configurations that generated on every screened subset)."""
    acc = collections.defaultdict(lambda: {"r": [], "rs": [], "ok": True})
    if os.path.exists(SCREEN_CSV):
        for r in csv.DictReader(open(SCREEN_CSV, newline="")):
            if r["dataset"] != DATASET or r["n_real"] != str(n) or r["gen_status"].startswith("skipped"):
                continue
            d = acc[(r["variant"], float(r["nu_t"]), int(r["K"]))]
            if r["gen_status"] != "ok" or not r["ratio"]:
                d["ok"] = False; continue
            d["r"].append(float(r["ratio"])); d["rs"].append(float(r["ratio_short"]))
    st = {k: (np.mean(d["r"]), np.mean(d["rs"])) for k, d in acc.items() if d["ok"] and d["r"]}
    if not st:
        return {}
    out = {"long": min(st, key=lambda k: st[k][0]), "short": min(st, key=lambda k: st[k][1])}
    kmax = max(k[2] for k in st)
    top = min((k for k in st if k[2] == kmax), key=lambda k: st[k][0])
    if top != out["long"]:
        out["modes"] = top
    return out


def bkey(r):
    return (r["dataset"], r["n_real"], r["subset"], r["kind"], r["arm"], r["epochs"])


def stage_b(o):
    ep = o["epochs"]
    rows = list(csv.DictReader(open(TRAIN_CSV, newline=""))) if os.path.exists(TRAIN_CSV) else []
    done = set(bkey(r) for r in rows)
    cap = [float(r["Ek"]) for r in rows if r["kind"] == "capacity"]
    if not cap and os.path.exists(ROUND1_CSV):
        cap = [float(r["Ek"]) for r in csv.DictReader(open(ROUND1_CSV, newline=""))
               if r["dataset"] == DATASET and r["kind"] == "capacity" and r["latent"] == str(LATENT)
               and r["epochs"] == str(ep)]
    cap_Ek = cap[-1] if cap else None
    plan = {}
    for n in o["n"]:
        best = best_configs(n)
        arms = [a for a in ARMS.get(n, ARMS[1000]) if a[1] in best]
        if best:
            plan[n] = (best, arms)
    if not plan:
        print(f"[ns2]  stage B: no screen results yet ({SCREEN_CSV}) - run stage A first")
        return
    samples = 0 if cap_Ek is not None else 1800
    print(f"[ns2]  stage B (latent {LATENT}, {ep} epochs, capacity "
          f"{'%.1f %%' % cap_Ek if cap_Ek is not None else 'to be measured'}):")
    for n, (best, arms) in plan.items():
        for k, v in best.items():
            print(f"          n={n:5d}  best_{k:5s} = {v[0]}, nu_T {v[1]:g}, K {v[2]}")
        for sub in range(o["subsets"]):
            if (DATASET, str(n), str(sub), "baseline", "", str(ep)) not in done:
                samples += n
            if n == 500 and (DATASET, str(n), str(sub), "baseline_long", "", str(LONG_EPOCHS)) not in done:
                samples += n * LONG_EPOCHS / ep
            for label, _, _, f in arms:
                if (DATASET, str(n), str(sub), "aug", label, str(ep)) not in done:
                    samples += n + n_aug_for(f, n)
    print(f"[ns2]  stage B: about {samples * SEC_PER_SAMPLE_CH * ep / 500 / 3600:.1f} h of training  ->  {TRAIN_CSV}",
          flush=True)
    if o["plan"]:
        return

    S = Setup(DATASET, gen_time=LONG_TIME)
    T0 = time.time()
    if cap_Ek is None:
        pool = S.data[S.pool_all]
        ek, detR, _, wall = vae.train(pool, np.empty((0,) + pool.shape[1:], np.float32), S.val_real,
                                      tag=f"{DATASET} capacity (whole pool)", latent=LATENT, epochs=ep)
        cap_Ek = float(ek)
        append(TRAIN_CSV, B_FIELDS, dict(date=now(), dataset=DATASET, n_real=len(pool), kind="capacity", latent=LATENT,
                                         epochs=ep, n_train=len(pool), Ek=round(cap_Ek, 4), detR=round(float(detR), 6),
                                         status="ok", wall_s=round(wall)))
        del pool
    empty = lambda x: np.empty((0,) + x.shape[1:], np.float32)

    for n, (best, arms) in plan.items():
        for sub in range(o["subsets"]):
            tr_idx, windows = S.subset(n, SAMPLING, sub)
            train = S.data[tr_idx]
            print(f"\n[ns2]  ===== n={n} ({SAMPLING}), subset {sub} =====", flush=True)
            info0 = dict(dataset=DATASET, n_real=n, sampling=SAMPLING, subset=sub, latent=LATENT)

            key = (DATASET, str(n), str(sub), "baseline", "", str(ep))
            base = next((float(r["Ek"]) for r in rows if bkey(r) == key), None)
            if base is None:
                ek, detR, _, wall = vae.train(train, empty(train), S.val_real, tag=f"n={n} s{sub} real only",
                                              latent=LATENT, epochs=ep)
                base = float(ek)
                append(TRAIN_CSV, B_FIELDS, dict(info0, date=now(), kind="baseline", epochs=ep, n_train=n,
                                                 capacity_Ek=round(cap_Ek, 4), Ek=round(base, 4),
                                                 detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
                print(f"[ns2]  real-only Ek {base:.2f}%", flush=True)
            if n == 500 and (DATASET, str(n), str(sub), "baseline_long", "", str(LONG_EPOCHS)) not in done:
                ek, detR, _, wall = vae.train(train, empty(train), S.val_real, tag=f"n={n} s{sub} real only "
                                              f"{LONG_EPOCHS} epochs", latent=LATENT, epochs=LONG_EPOCHS)
                append(TRAIN_CSV, B_FIELDS, dict(info0, date=now(), kind="baseline_long", epochs=LONG_EPOCHS,
                                                 n_train=n, baseline_Ek=round(base, 4), Ek=round(float(ek), 4),
                                                 gain=round(float(ek) - base, 3), detR=round(float(detR), 6),
                                                 status="ok", wall_s=round(wall)))
                print(f"[ns2]  real-only Ek with {LONG_EPOCHS} epochs: {ek:.2f}% ({float(ek) - base:+.2f})", flush=True)

            cache = {}                                    # variant -> (P, l, q, kappa, lap, Kb)
            for ai, (label, which, horizon, f) in enumerate(arms):
                if (DATASET, str(n), str(sub), "aug", label, str(ep)) in done:
                    continue
                variant, nt, K = best[which]
                if variant not in cache:
                    P = channel_pod(train, homogeneous_x=VARIANTS[variant][0])
                    Kb = int(min(K_MAX, P["A"].shape[1], n - 1,
                                 max(v[2] for v in best.values() if v[0] == variant)))
                    cache[variant] = (P,) + tuple(ops_for(S, P, variant, Kb)) + (Kb,)
                P, l, q, kappa, lap, Kb = cache[variant]
                K = min(K, Kb)
                A = np.asarray(P["A"][:, :Kb], dtype=np.float64)
                ceiling, win_A = ceiling_and_project(P, K, S.val_real, windows)
                head_c = min(ceiling, cap_Ek) - base
                na = n_aug_for(f, n)
                info = dict(info0, arm=label, variant=variant, nu_t=nt, K=K, horizon=horizon or "",
                            energy_K_pct=round(100 * P["e_cum"][K - 1], 3), fraction=round(f, 4), epochs=ep,
                            pod_ceiling_Ek=round(ceiling, 3), capacity_Ek=round(cap_Ek, 4), baseline_Ek=round(base, 4),
                            headroom_corrected=round(head_c, 3))
                t0 = time.time()
                qerr = pe = ratio = None; pred = keep = ""
                try:
                    if label.startswith("ns_"):
                        hs = steps_of(S, horizon)
                        step, s, to_b, from_b = model_at(add_eddy_viscosity(l, lap, nt), q, kappa, A, K, S.dt, SUBSTEPS)
                        wh = [w[:hs + 1] for w in win_A]
                        qerr, _ = fidelity(step, to_b, from_b, wh, hs)
                        pe = persistence(wh, hs)
                        ratio = qerr / max(pe, 1e-12)
                        pred = "gain" if (ratio <= 1 - RULE_SKILL and head_c >= RULE_HEAD) else "no gain"
                        aug, st = synthetic_arm(step, s, to_b, from_b, P, A[:, :K], K, na, hs,
                                                np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, K,
                                                                       SAMPLINGS.index(SAMPLING), ai]),
                                                kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ)
                        keep = round(st["keep_frac"], 4)
                    elif label == "jitter":
                        aug = jitter_arm(P, A[:, :K], K, na, np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, K, 7777]))
                    else:
                        aug, na = real_arm(P, K, S.data, S.pool_idx, tr_idx, na,
                                           np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, K, 7777]),
                                           project=True, cap=True)
                except Exception as err:
                    msg = str(err).splitlines()[0][:160]
                    print(f"[ns2]  !! {label}: generation failed ({msg})", flush=True)
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="gen_failed", predicted=pred,
                                                     status=f"failed: {msg}"))
                    continue
                if label.startswith("ns_"):
                    print(f"[ns2]  {label}: {variant}, nu_T {nt:g}, K {K}, {horizon} h/U_b: error {qerr:.2f} vs "
                          f"no change {pe:.2f}, corrected headroom {head_c:.1f}  ->  predicted {pred}", flush=True)
                ek, detR, _, wall = vae.train(train, aug[:na], S.val_real, tag=f"n={n} s{sub} {label} f={f}",
                                              latent=LATENT, epochs=ep)
                append(TRAIN_CSV, B_FIELDS, dict(
                    info, date=now(), kind="aug", n_aug=na, n_train=n + na,
                    quality_err=round(qerr, 5) if qerr is not None else "", persist_err=round(pe, 5) if pe else "",
                    ratio=round(ratio, 4) if ratio is not None else "", predicted=pred, Ek=round(float(ek), 4),
                    gain=round(float(ek) - base, 3), detR=round(float(detR), 6), gen_keep_frac=keep,
                    status="ok", wall_s=round(wall)))
                print(f"[ns2]  {label}: Ek {ek:.2f}%  gain {float(ek) - base:+.2f}  ({time.time() - t0:.0f} s)   "
                      f"[{(time.time() - T0) / 3600:.1f} h]", flush=True)
                del aug
            del train, windows, cache
    print(f"\n[ns2]  stage B done in {(time.time() - T0) / 3600:.2f} h  ->  {TRAIN_CSV}")


def main():
    o = _args(sys.argv[1:])
    if "A" in o["stage"]:
        stage_a(o)
    if "B" in o["stage"]:
        stage_b(o)


if __name__ == "__main__":
    main()

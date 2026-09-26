"""
CHANNEL FLOW: do the two POD-Galerkin generators help the beta-VAE on wall turbulence?

Every study so far is a 2-D plate wake. This one moves the same pipeline to the JHTDB turbulent
channel (Re_tau ~ 1000, Re = 1/nu = 20000 in half-height / bulk-velocity units): one streamwise /
wall-normal plane over the whole domain, x in [0, 8 pi), y wall to wall, 1024 x 256, 2000 snapshots
with dt = 0.013 (getDataCode/GetChannelData_Matlab_JHTDB.m), in-plane components u, v.

Both generators of the region map, on the same real subsets:
    galerkin     DATA-IDENTIFIED: the quadratic model da/dt = c + L a + Q(a, a) fitted to the subset's
                 own POD-coefficient series (ridge least squares on central differences)
    galerkin_ns  NS-PROJECTED: 2-D convection + diffusion projected on the subset's POD modes,
                 plus the imposed mean pressure gradient -dP/dx = 0.0025 that drives the channel
                 (nu = 5e-5, both from the JHTDB channel README); nothing fitted, no pressure
                 fluctuation. (On the half plane the forcing lowers the quality error by ~0.02.)

What I expect, written before any number exists:
    the plane is a slice of a 3-D flow. The projected 2-D equations miss the spanwise transport
    (w du/dz, w dv/dz), and the in-plane velocity is not divergence-free, so the dropped pressure
    term does not vanish either. The NS model should leave the flow fast: quality error well above
    0.5, the limit of the region map, so no gain. The data-identified model does not need the
    equations to hold on the plane, only a smooth low-order dynamics, so it is the one that can pass
    the screen, most likely at few modes. Headroom should not be the problem: a few hundred
    snapshots of turbulence leave the network far from its ceiling.

Half-plane check before the full download (chanHalf, n = 50 / 250, one subset, 2 h/U_b):
    NS-projected   quality error 0.77-2.0, WORSE than "nothing changes" (0.41-0.96) in every cell;
    data-identified 2.0 (blows up) in every cell, with blocks of 10 and with runs of 50: it fits the
    training derivatives perfectly (R^2 = 1.00) and fails on held-out windows, an overfit of short
    trajectory pieces. Nothing passes, so stage B would only contain "tested anyway" cells.

STAGE A - screen, no network (a few minutes per subset).
    n = 100, 250, 500 real snapshots x 2 samplings x 3 subsets x truncations 90% / 95% energy and
    K = 5, 10, 20, 40, for BOTH generators: POD ceiling, quality error over GEN_TIME on held-out real
    windows, derivative R^2, and whether synthetic snapshots can be generated.
    Samplings: blocks10 = blocks of 10 consecutive snapshots anywhere in the pool (every earlier
    study); runs50 = runs of 50 consecutive pool snapshots (the held-out ones skipped). Here 10
    snapshots span only 0.13 h/U_b (2 convective times in the wake), too short a piece of
    trajectory for the data-identified fit, so it gets the longer runs as well.
    Horizon GEN_TIME = 2 h/U_b (154 snapshots), per quality window and per synthetic trajectory.
    Measured on the half plane: the error of "nothing changes" (a(t) = a(0)) is 0.2 at 0.5 h/U_b,
    ~1 at 2 and saturates beyond, so at 2 h/U_b the flow has decorrelated as it had over the 10
    convective times of the wake, and "quality error <= 0.5" again means "the model halves the error
    of assuming nothing changes". (A first screen at 0.5 h/U_b gave NS errors of 0.15-0.35: no better
    than the no-change reference, so it measured nothing.) persist_err is stored next to every error.
    Quality windows: windows that long do not fit between the training snapshots, so the LAST 25 %
    of the record (6.5 h/U_b) is reserved for them. Real subsets and the real_proj snapshots come
    from the pool before it; every subset and both generators are scored on the same 14 windows
    (start every 25 snapshots). Validation (Ek) stays the random 10 % of the whole record.
    -> convergence/channel_screen.csv

STAGE B - trainings where the screen says it is worth it, plus the generators that fail it, anyway.
    1. Capacity: real only on the whole training pool (1800 snapshots), the network's own limit at
       this latent, so headroom = min(POD ceiling, capacity) - real only (capacity_headroom.py).
    2. Cells. Per n and generator, the (sampling, truncation) with the highest POD ceiling among
       those that are FAITHFUL (mean quality error <= half the mean "nothing changes" error, see
       below), that generate and whose ceiling leaves >= 20 points
       (pick_trunc of run_lowdata.py) -> "screen passed". Where a generator has no such cell at
       some n, its least unfaithful cell there (ceiling >= 20) is trained anyway -> "tested anyway",
       so the prediction "no gain" is tested too.
    3. Per cell and subset: real only (shared by the cells of the same n and sampling), the generator
       at f = 0.5 and jitter at the same truncation (real coefficients + noise, no dynamics: tells a
       gain from dynamics apart from a gain from noise); the "screen passed" cells also get real_proj
       (extra real snapshots projected on the same modes: what a perfect reduced model would give).
    Every generator row carries its prediction, written before its network is trained:
    gain if quality error <= 0.5 x "nothing changes" error and corrected headroom >= 20.

The rule is RELATIVE to "nothing changes" on the channel. Full-plane check (one subset per n, before
the screen): at 2 h/U_b the no-change error on the reserved tail windows was only 0.04-0.25, against
0.3-1.2 when the modes come from subsets spread over the whole record. Subsets from the first 75 %
see the tail with a large near-constant offset, so an absolute limit (error <= 0.5, the wake's) would
call a frozen model faithful. Halving the no-change error on the same windows is what error <= 0.5
meant in the wake, where the no-change error over 10 convective times is ~1 or more.
    -> convergence/channel_results.csv

Usage (from the repository folder):
    python3 studies/7_channel/run_channel.py --plan          # what is left and the time, nothing runs
    python3 studies/7_channel/run_channel.py --stage A       # the screen only
    python3 studies/7_channel/run_channel.py --stage B       # capacity + trainings (reads the screen)
    python3 studies/7_channel/run_channel.py                 # A then B
Options:
    --dataset chanHalf      the old half-plane file (x in [0, pi], y centreline -> wall) instead
    --n 50,100  --subsets 2  --epochs 2  --suffix smoke      quick checks (separate CSV files)
"""
import os, sys, csv, time, collections
import numpy as np

from rom import paths
from rom import vae
from rom.data import load_channel, CHANNEL_DPDX
from rom.split import split_indices, draw_blocks, SPLIT_SEED, DRAW_SEED
from rom.pod import subset_pod, ceiling_and_project, k_for
from rom.galerkin_ns import build_ops, model_at, derivative_R2, integrate_trajectories
from rom import galerkin_data as gd
from rom.augment import (n_aug_for, fidelity, synthetic_arm, jitter_arm, real_arm,
                         SUBSTEPS, IC_NOISE, ENERGY_TOL, MAX_TRAJ, RIDGE)
from rom.vae import EPOCHS, SEC_PER_SAMPLE
from rom.results import append_row as append, done_rows, now

# ============ CONFIG ============
DATASET    = "chan"                    # rom.data.CHANNEL: "chan" full plane | "chanHalf" old half plane
LATENT     = 16
N_REAL     = [100, 250, 500]          # 50 dropped: on the full plane its POD ceiling is 3-9 %
N_SUBSETS  = 3
TRUNCS     = [("90%", 0.90), ("95%", 0.95), ("K5", 5), ("K10", 10), ("K20", 20), ("K40", 40)]
K_MAX      = 40                        # NS operators built once at this size and sliced; the
                                       # data-identified fit has 1 + K + K(K+1)/2 regressors per mode
GENS       = ["galerkin", "galerkin_ns"]
GEN_TIME   = 2.0                       # h / U_bulk per quality window and per synthetic trajectory
QUALITY_TAIL = 0.25                    # last fraction of the record kept for the quality windows
WINDOW_STRIDE = 25                     # snapshots between window starts
FRACTION   = 0.5
RULE_SKILL = 0.5                       # faithful: quality error <= (1 - RULE_SKILL) x "nothing changes" error
RULE_HEAD  = 20.0
SEED_TAG   = 1000                      # in the rng seeds where the wake studies have int(Re)
SAMPLINGS  = ["blocks10", "runs50"]
RUN        = 50                        # pool snapshots per run of the runs50 sampling
SCREEN_CSV = os.path.join(paths.RESULTS, "channel_screen.csv")
TRAIN_CSV  = os.path.join(paths.RESULTS, "channel_results.csv")
# ================================

A_FIELDS = ["date", "dataset", "n_real", "sampling", "subset", "trunc", "K", "energy_K_pct", "pod_ceiling_Ek",
            "generator", "quality_err", "persist_err", "quality_blowup_frac", "quality_windows", "deriv_R2_min",
            "deriv_R2_mean", "gen_keep_frac", "gen_n_traj", "gen_status", "seconds"]
B_FIELDS = ["date", "dataset", "n_real", "sampling", "subset", "kind", "latent", "generator", "trunc", "K",
            "energy_K_pct", "fraction", "epochs", "n_aug", "n_train", "pod_ceiling_Ek", "capacity_Ek",
            "baseline_Ek", "headroom", "headroom_corrected", "quality_err", "persist_err", "selection", "predicted",
            "Ek", "gain", "detR", "gen_keep_frac", "status", "wall_s"]


def _args(argv):
    o = {"plan": False, "stage": "AB", "dataset": DATASET, "n": N_REAL, "subsets": N_SUBSETS,
         "epochs": EPOCHS}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--stage": o["stage"] = next(it).upper()
        elif a == "--dataset": o["dataset"] = next(it)
        elif a == "--n": o["n"] = [int(x) for x in next(it).split(",")]
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--suffix":                          # quick checks: write to separate files
            global SCREEN_CSV, TRAIN_CSV
            suf = next(it)
            SCREEN_CSV = SCREEN_CSV.replace(".csv", f"_{suf}.csv")
            TRAIN_CSV = TRAIN_CSV.replace(".csv", f"_{suf}.csv")
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


def k_of(P, trunc, n):
    name, v = trunc
    K = k_for(P, v) if isinstance(v, float) else v
    return int(min(K, P["A"].shape[1], n - 1))


def draw_runs(pool_idx, n, rng, run=RUN):
    """n training indices made of non-overlapping runs of `run` consecutive POOL positions (the
    validation snapshots in between are skipped, as in the contig sampling of run_lowdata.py)."""
    n_runs = int(np.ceil(n / run))
    starts = list(range(len(pool_idx) - run + 1)); rng.shuffle(starts)
    taken, chosen = set(), []
    for st in starts:
        if taken.isdisjoint(range(st, st + run)):
            chosen.append(st); taken.update(range(st, st + run))
        if len(chosen) == n_runs:
            break
    pos = sorted(i for st in chosen for i in range(st, st + run))[:n]
    return np.asarray(pool_idx)[pos]


def persistence(win_A, steps):
    """Quality error of a(t) = a(0): the reference every model has to beat."""
    ident = lambda v: v
    return fidelity(ident, ident, ident, win_A, steps)[0]


def model(gen, l_big, q_big, kappa, A, K, tr_idx, dt):
    """(step, s, to_b, from_b, derivative R^2) of either generator at K modes.
    A = the subset's POD coefficients (vector units), at least K columns."""
    if gen == "galerkin":
        beta, s, R2, _ = gd.fit_galerkin(A[:, :K], tr_idx, dt, RIDGE)
        step = gd.make_step(beta, "ode", dt, SUBSTEPS)
        return step, s, (lambda a: a / s), (lambda b: b * s), R2
    step, s, to_b, from_b = model_at(l_big, q_big, kappa, A, K, dt, SUBSTEPS)
    R2 = derivative_R2(l_big[:K, :K + 1], q_big[:K, :K + 1, :K + 1], A[:, :K] * kappa, tr_idx, dt)
    return step, s, to_b, from_b, R2


class Setup:
    """The dataset, its split, and the real subsets, drawn exactly the same way in both stages."""
    def __init__(self, ds):
        self.ds = ds
        self.data, self.re, self.dt, self.path = load_channel(ds)
        self.Nt = len(self.data)
        self.val_idx, pool_all, _ = split_indices(self.Nt, None, "random", SPLIT_SEED)
        self.pool_all = pool_all                                  # capacity: the whole training pool
        self.t_cut = int(round((1 - QUALITY_TAIL) * self.Nt))     # quality windows live after it
        self.pool_idx = pool_all[pool_all < self.t_cut]           # real subsets + real_proj before it
        self.pool_set = set(int(i) for i in self.pool_idx)
        self.val_real = self.data[self.val_idx]
        self.steps = max(1, int(round(GEN_TIME / self.dt)))
        self.win = list(range(self.t_cut, self.Nt - self.steps, WINDOW_STRIDE))

    def subset(self, n, sm, sub):
        rng = np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, SAMPLINGS.index(sm)])
        tr_idx = draw_blocks(self.pool_set, self.t_cut, n, rng) if sm == "blocks10" else \
            draw_runs(self.pool_idx, n, rng)
        return tr_idx, [self.data[t:t + self.steps + 1] for t in self.win]


# ------------------------------------------------------------------ stage A

def stage_a(o):
    ds = o["dataset"]
    done = done_rows(SCREEN_CSV, lambda r: (r["dataset"], r["n_real"], r["sampling"], r["subset"], r["trunc"],
                                            r["generator"]))
    todo = [(n, sm, sub) for n in o["n"] for sm in SAMPLINGS for sub in range(o["subsets"])
            if any((ds, str(n), sm, str(sub), t[0], g) not in done for t in TRUNCS for g in GENS)]
    print(f"[chan]  stage A: {len(todo)} subsets to screen ({len(todo) * len(TRUNCS) * len(GENS)} rows), "
          f"about {len(todo) * 3:.0f} min  ->  {SCREEN_CSV}", flush=True)
    if o["plan"] or not todo:
        return
    S = Setup(ds)
    for n, sm, sub in todo:
        t0 = time.time()
        tr_idx, windows = S.subset(n, sm, sub)
        P = subset_pod(S.data[tr_idx])
        Kb = int(min(K_MAX, P["A"].shape[1], n - 1))
        l_big, q_big, kappa = build_ops(P, Kb, S.re, S.path, forcing_x=CHANNEL_DPDX)
        A = np.asarray(P["A"][:, :Kb], dtype=np.float64)
        print(f"\n[chan]  {ds} n={n} {sm} s{sub}: {len(windows)} quality windows of {S.steps} steps, "
              f"NS operators at K={Kb}, 95% energy needs {k_for(P, 0.95)} modes [{time.time() - t0:.0f} s]",
              flush=True)
        for trunc in TRUNCS:
            name = trunc[0]
            K = k_of(P, trunc, n)
            base = dict(dataset=ds, n_real=n, sampling=sm, subset=sub, trunc=name, K=K)
            if K > Kb or K < 1:
                for gen in GENS:
                    if (ds, str(n), sm, str(sub), name, gen) not in done:
                        append(SCREEN_CSV, A_FIELDS, dict(base, date=now(), generator=gen,
                                                          gen_status=f"skipped: K={K} above K_MAX={Kb}"))
                print(f"[chan]      {name:4s} K={K:3d}  skipped (above K_MAX)", flush=True)
                continue
            ceiling, win_A = ceiling_and_project(P, K, S.val_real, windows)
            pe = persistence(win_A, S.steps) if win_A else float("nan")
            base.update(energy_K_pct=round(100 * P["e_cum"][K - 1], 3), pod_ceiling_Ek=round(ceiling, 3),
                        persist_err=round(pe, 5))
            for gen in GENS:
                if (ds, str(n), sm, str(sub), name, gen) in done:
                    continue
                try:
                    step, s, to_b, from_b, R2 = model(gen, l_big, q_big, kappa, A, K, tr_idx, S.dt)
                except Exception as e:
                    append(SCREEN_CSV, A_FIELDS, dict(base, date=now(), generator=gen,
                                                      gen_status=f"model failed: {str(e).splitlines()[0][:70]}"))
                    continue
                err, blow = fidelity(step, to_b, from_b, win_A, S.steps) if win_A else (float("nan"), "")
                try:
                    _, st = integrate_trajectories(step, s, to_b(A[:, :K]), n, S.steps, IC_NOISE, ENERGY_TOL,
                                                   4000, np.random.default_rng([7, K, n]))
                    gen_ok, keep, ntraj = "ok", round(st["keep_frac"], 4), st["n_traj"]
                except Exception as e:
                    gen_ok, keep, ntraj = str(e).splitlines()[0][:80], "", ""
                append(SCREEN_CSV, A_FIELDS, dict(
                    base, date=now(), generator=gen, quality_err=round(err, 5), quality_blowup_frac=blow,
                    quality_windows=len(win_A), deriv_R2_min=round(float(np.min(R2)), 4),
                    deriv_R2_mean=round(float(np.mean(R2)), 4), gen_keep_frac=keep, gen_n_traj=ntraj,
                    gen_status=gen_ok, seconds=round(time.time() - t0)))
                print(f"[chan]      {name:4s} K={K:3d} energy {100 * P['e_cum'][K - 1]:5.1f}%  ceiling "
                      f"{ceiling:5.1f}%  {gen:11s} error {err:5.2f} (no change {pe:4.2f})  R2 {np.mean(R2):5.2f}  {gen_ok}", flush=True)
        del P, l_big, q_big, windows


# ------------------------------------------------------------------ stage B

def screen_stats(ds):
    """(n, sampling, generator, trunc) -> (mean quality error / mean no-change error, mean POD ceiling,
    mean K); only truncations that generated on every screened subset."""
    acc = collections.defaultdict(lambda: {"e": [], "c": [], "K": [], "p": [], "ok": True})
    if not os.path.exists(SCREEN_CSV):
        return {}
    for r in csv.DictReader(open(SCREEN_CSV, newline="")):
        if r["dataset"] != ds or r["gen_status"].startswith("skipped"):
            continue
        d = acc[(int(r["n_real"]), r["sampling"], r["generator"], r["trunc"])]
        try:
            e, c, pe = float(r["quality_err"]), float(r["pod_ceiling_Ek"]), float(r["persist_err"])
        except ValueError:
            d["ok"] = False; continue
        if r["gen_status"] != "ok" or not np.isfinite(e):
            d["ok"] = False
        d["e"].append(e); d["c"].append(c); d["K"].append(int(r["K"])); d["p"].append(pe)
    # value: (error / no-change error, mean POD ceiling, mean K); ratio <= 1 - RULE_SKILL is faithful
    return {k: (float(np.mean(d["e"])) / max(float(np.mean(d["p"])), 1e-12), float(np.mean(d["c"])),
                round(float(np.mean(d["K"]))))
            for k, d in acc.items() if d["ok"] and d["e"]}


def cells(o):
    """[(n, sampling, generator, trunc, selection)] of stage B, from the screen. Per n and generator:
    the (sampling, truncation) with the highest POD ceiling among the faithful ones."""
    st = screen_stats(o["dataset"])
    out = []
    for gen in GENS:
        mine = [(n, sm, t, v) for (n, sm, g, t), v in st.items() if g == gen and n in o["n"]]
        passed = []
        for n in o["n"]:
            ok = [(sm, t, v) for nn, sm, t, v in mine if nn == n and v[0] <= 1 - RULE_SKILL and v[1] >= RULE_HEAD]
            if ok:
                sm, t, _ = max(ok, key=lambda x: (x[2][1], -x[2][0]))       # highest ceiling
                passed.append((n, sm, gen, t, "screen passed"))
        out += passed
        for n in o["n"]:                    # no faithful cell at this n: its least unfaithful one, anyway
            if any(c[0] == n for c in passed):
                continue
            room = [(sm, t, v) for nn, sm, t, v in mine if nn == n and v[1] >= RULE_HEAD]
            if room:
                sm, t, _ = min(room, key=lambda x: (x[2][0], -x[2][1]))
                out.append((n, sm, gen, t, "tested anyway"))
    return out


def arms_of(cell_list):
    """(n, sampling) -> ordered unique arms (generator, trunc, selection). Every cell gets the jitter
    reference at its truncation (is a gain dynamics or just noise?); the passed cells also get
    real_proj (what a perfect reduced model would give)."""
    by_n = collections.OrderedDict()
    for n, sm, gen, t, sel in sorted(cell_list, key=lambda c: (c[0], c[1])):
        arms = by_n.setdefault((n, sm), [])
        for a in ([(gen, t, sel), ("jitter", t, "reference")] +
                  ([("real_proj", t, "reference")] if sel == "screen passed" else [])):
            if (a[0], a[1]) not in [(x[0], x[1]) for x in arms]:
                arms.append(a)
    return by_n


def bkey(r):
    return (r["dataset"], r["n_real"], r["sampling"], r["subset"], r["kind"], r["generator"], r["trunc"],
            r["fraction"], r["epochs"])


def stage_b(o):
    ds, ep = o["dataset"], o["epochs"]
    rows = list(csv.DictReader(open(TRAIN_CSV, newline=""))) if os.path.exists(TRAIN_CSV) else []
    done = set(bkey(r) for r in rows)
    base_done = {(r["n_real"], r["sampling"], r["subset"]): float(r["Ek"]) for r in rows
                 if r["dataset"] == ds and r["kind"] == "baseline" and r["epochs"] == str(ep)}
    cap = [float(r["Ek"]) for r in rows if r["dataset"] == ds and r["kind"] == "capacity" and r["epochs"] == str(ep)]
    cap_Ek = cap[-1] if cap else None
    cl = cells(o)
    if not cl:
        print(f"[chan]  stage B: no screen results for {ds} yet ({SCREEN_CSV}) - run stage A first")
        return
    by_n = arms_of(cl)
    fr = str(round(FRACTION, 4))
    samples = 0 if cap_Ek is not None else 1800
    print(f"[chan]  stage B cells ({ds}, latent {LATENT}, f = {FRACTION}):")
    for (n, sm), arms in by_n.items():
        for gen, t, sel in arms:
            print(f"          n={n:4d} {sm:8s}  {gen:11s} {t:4s}  {sel}")
        for sub in range(o["subsets"]):
            if (str(n), sm, str(sub)) not in base_done:
                samples += n
            for gen, t, sel in arms:
                if (ds, str(n), sm, str(sub), "aug", gen, t, fr, str(ep)) not in done:
                    samples += n + n_aug_for(FRACTION, n)
    print(f"[chan]  stage B: {'capacity already measured, ' if cap_Ek is not None else 'capacity + '}"
          f"about {samples * SEC_PER_SAMPLE * ep / 500 / 3600:.1f} h of training  ->  {TRAIN_CSV}", flush=True)
    if o["plan"]:
        return

    S = Setup(ds)
    T0 = time.time()
    if cap_Ek is None:                                   # 1. the network's own ceiling at this latent
        pool = S.data[S.pool_all]
        ek, detR, _, wall = vae.train(pool, np.empty((0,) + pool.shape[1:], np.float32), S.val_real,
                                      tag=f"{ds} capacity (whole pool, {len(pool)})", latent=LATENT, epochs=ep)
        cap_Ek = float(ek)
        append(TRAIN_CSV, B_FIELDS, dict(date=now(), dataset=ds, n_real=len(pool), subset="", kind="capacity",
                                         latent=LATENT, epochs=ep, n_train=len(pool), Ek=round(cap_Ek, 4),
                                         detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
        print(f"[chan]  capacity at latent {LATENT}: Ek {cap_Ek:.2f}%", flush=True)
        del pool

    for (n, sm), arms in by_n.items():
        for sub in range(o["subsets"]):
            tr_idx, windows = S.subset(n, sm, sub)
            train_real = S.data[tr_idx]
            P = subset_pod(train_real)
            Ks = {t: k_of(P, [x for x in TRUNCS if x[0] == t][0], n) for _, t, _ in arms}
            Kb = max(Ks.values())
            A = np.asarray(P["A"][:, :Kb], dtype=np.float64)
            l_big = q_big = kappa = None
            if any(g == "galerkin_ns" for g, _, _ in arms):
                l_big, q_big, kappa = build_ops(P, Kb, S.re, S.path, forcing_x=CHANNEL_DPDX)
            print(f"\n[chan]  ===== {ds} {n} real ({sm}), subset {sub}: {len(windows)} quality windows =====", flush=True)

            key = (str(n), sm, str(sub))
            if key not in base_done:
                ek, detR, _, wall = vae.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32),
                                              S.val_real, tag=f"{ds} n={n} {sm} s{sub} real only", latent=LATENT, epochs=ep)
                base_done[key] = float(ek)
                append(TRAIN_CSV, B_FIELDS, dict(date=now(), dataset=ds, n_real=n, sampling=sm, subset=sub, kind="baseline",
                                                 latent=LATENT, epochs=ep, n_train=n, capacity_Ek=round(cap_Ek, 4),
                                                 Ek=round(float(ek), 4), detR=round(float(detR), 6),
                                                 status="ok", wall_s=round(wall)))
                print(f"[chan]  real-only Ek {ek:.2f}%", flush=True)
            base = base_done[key]

            for gi, (gen, t, sel) in enumerate(arms):
                if (ds, str(n), sm, str(sub), "aug", gen, t, str(round(FRACTION, 4)), str(ep)) in done:
                    continue
                K = Ks[t]
                ceiling, win_A = ceiling_and_project(P, K, S.val_real, windows)
                head = ceiling - base
                head_c = min(ceiling, cap_Ek) - base
                pe = persistence(win_A, S.steps) if win_A else None
                info = dict(dataset=ds, n_real=n, sampling=sm, subset=sub, latent=LATENT, generator=gen, trunc=t, K=K,
                            energy_K_pct=round(100 * P["e_cum"][K - 1], 3), fraction=round(FRACTION, 4),
                            epochs=ep, pod_ceiling_Ek=round(ceiling, 3), capacity_Ek=round(cap_Ek, 4),
                            baseline_Ek=round(base, 4), headroom=round(head, 3),
                            headroom_corrected=round(head_c, 3), selection=sel,
                            persist_err=round(pe, 5) if pe is not None else "")
                na = n_aug_for(FRACTION, n)
                t0 = time.time()
                qerr, pred, keep = None, "", ""
                try:
                    if gen in GENS:
                        step, s, to_b, from_b, _ = model(gen, l_big, q_big, kappa, A, K, tr_idx, S.dt)
                        qerr, _ = fidelity(step, to_b, from_b, win_A, S.steps) if win_A else (None, "")
                        pred = "gain" if (qerr is not None and pe and qerr <= (1 - RULE_SKILL) * pe
                                          and head_c >= RULE_HEAD) else "no gain"
                        aug, st = synthetic_arm(step, s, to_b, from_b, P, A[:, :K], K, na, S.steps,
                                                np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, K,
                                                                       GENS.index(gen), SAMPLINGS.index(sm)]),
                                                kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ)
                        keep = round(st["keep_frac"], 4)
                    elif gen == "jitter":
                        aug = jitter_arm(P, A[:, :K], K, na,
                                         np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, K, 7777, SAMPLINGS.index(sm)]))
                    else:
                        aug, na = real_arm(P, K, S.data, S.pool_idx, tr_idx, na,
                                           np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, K, 7777, SAMPLINGS.index(sm)]),
                                           project=True, cap=True)
                except Exception as err:
                    msg = str(err).splitlines()[0][:160]
                    print(f"[chan]  !! {gen} {t}: generation failed ({msg})", flush=True)
                    append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="gen_failed",
                                                     quality_err=round(qerr, 5) if qerr is not None else "",
                                                     predicted=pred, status=f"failed: {msg}"))
                    continue
                print(f"[chan]  {gen} {t} K={K}: quality error {'-' if qerr is None else f'{qerr:.3f}'}, "
                      f"corrected headroom {head_c:.1f}  ->  predicted {pred or '-'}", flush=True)
                ek, detR, _, wall = vae.train(train_real, aug[:na], S.val_real,
                                              tag=f"{ds} n={n} {sm} s{sub} {gen} {t} f={FRACTION}",
                                              latent=LATENT, epochs=ep)
                append(TRAIN_CSV, B_FIELDS, dict(info, date=now(), kind="aug", n_aug=na, n_train=n + na,
                                                 quality_err=round(qerr, 5) if qerr is not None else "",
                                                 predicted=pred, Ek=round(float(ek), 4),
                                                 gain=round(float(ek) - base, 3), detR=round(float(detR), 6),
                                                 gen_keep_frac=keep, status="ok", wall_s=round(wall)))
                print(f"[chan]  {gen} {t}: Ek {ek:.2f}%  gain {float(ek) - base:+.2f}  "
                      f"({time.time() - t0:.0f} s)   [{(time.time() - T0) / 3600:.1f} h]", flush=True)
                del aug
            del P, train_real, windows, l_big, q_big
    print(f"\n[chan]  stage B done in {(time.time() - T0) / 3600:.2f} h  ->  {TRAIN_CSV}")


def main():
    o = _args(sys.argv[1:])
    if "A" in o["stage"]:
        stage_a(o)
    if "B" in o["stage"]:
        stage_b(o)


if __name__ == "__main__":
    main()

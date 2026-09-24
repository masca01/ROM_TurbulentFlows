"""
REGION MAP: in which conditions does POD-Galerkin synthetic data help the beta-VAE,
independently of how the Galerkin model is built?

The generator is treated as a REPLICATE, not as the thing being compared. Every
condition is run with both generators on the SAME real subsets, and the gain is
later mapped against two generator-independent coordinates:

    headroom = POD ceiling of the real subset with K modes  -  real-only Ek
               (how much the synthetic data could add at most)
    fidelity = short-horizon prediction error of the reduced model on held-out
               real snapshots, in its own K-mode space (how faithful its dynamics are)

Design (defaults, one overnight run):
    dataset   Re50 (latent 5), Re80 (latent 11), Re100 (latent 15)   convergence-study latents
    n real    100, 250                     drawn as random BLOCKS of 10 consecutive snapshots
    subsets   3 per n                      shared by both generators and both mode levels
    modes K   holding 95 % and 99 % of the real subset's energy
    generator data-identified (galerkin) and NS-projected (galerkin_ns)
    fraction  0.5 synthetic                (n synthetic snapshots)
    split     random 10 % validation, seed 7 (convergence_split.py)
    network   500 epochs, beta 5e-3, batch 32, lr 3e-4, torch seed 7, Ek on real validation

Per real subset: 1 real-only training. Per subset x mode level x generator: POD ceiling,
fidelity, synthetic generation, and 1 augmented training per fraction.

Output: ../../convergence/region_map_results.csv, one row per training (kind = baseline or
aug) and one row per failed generation (kind = gen_failed). Resumable: rows already in the
CSV are skipped, so the script can be stopped and relaunched.

Usage:
    python3 run_region_map.py                 # full design
    python3 run_region_map.py --plan          # list the work and the time estimate, nothing runs
    python3 run_region_map.py --no-train      # ceilings + fidelity + generation only (fast check),
                                              # written to region_map_generators.csv
    python3 run_region_map.py --datasets Re100 --n 100 --subsets 1 --epochs 2   # smoke test
"""
import os, sys, csv, time, datetime
import numpy as np

from rom import galerkin_data as gd             # data-identified model: fit_galerkin, make_step
from rom import galerkin_ns as gns              # NS projection, grid_spacing
from rom import vae                             # train(): same recipe as every study so far
from rom.data import DATASETS as REGISTRY, load
from rom.split import split_indices, draw_blocks, SPLIT_SEED, DRAW_SEED, BLOCK
from rom.pod import subset_pod, k_for, ceiling_and_project
from rom.augment import (n_aug_for, fidelity, fidelity_windows, synthetic_arm, GENERATORS, GEN_TIME,
                         SUBSTEPS, IC_NOISE, ENERGY_TOL, RIDGE, MAX_TRAJ)
from rom.vae import EPOCHS, SEC_PER_SAMPLE
from rom.results import append_row

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))

# ============ CONFIG ============
# Files, snapshot caps and latents: rom.data.DATASETS (Re50 latent 5, Re80 11, Re100 15).
# Shared with every later study and imported above: split seed 7, subset seed 2026, blocks of
# 10 (rom.split); generators galerkin / galerkin_ns, 10 convective times per trajectory,
# 20 RK4 sub-steps, IC kick 0.05, energy band 5 %, ridge 1e-6, 20000 trajectories,
# 20 quality windows clipped at 2 (rom.augment); 500 epochs, ~1.3 s per snapshot (rom.vae).
DATASETS    = ["Re50", "Re80", "Re100"]
N_REAL      = [100, 250]
N_SUBSETS   = 3
MODE_LEVELS = [0.95, 0.99]       # K = modes holding this share of the subset's energy
FRACTIONS   = [0.5]              # synthetic fraction of the training set
CSV_OUT     = os.path.join(_CONV, "region_map_results.csv")
CSV_GEN     = os.path.join(_CONV, "region_map_generators.csv")
# ================================

FIELDS = ["date", "dataset", "Re", "dt", "latent", "n_real", "subset", "kind", "generator", "mode_level", "K",
          "energy_K_pct", "fraction", "n_aug", "n_train", "n_val", "Ek", "e", "detR",
          "pod_ceiling_Ek", "baseline_Ek", "headroom", "gain",
          "fid_err", "fid_blowup_frac", "fid_windows", "fid_steps",
          "gen_keep_frac", "gen_n_traj", "gen_R2_min", "gen_seconds", "status", "wall_s"]


def _args(argv):
    o = {"plan": False, "train": True, "datasets": list(DATASETS), "n": N_REAL, "subsets": N_SUBSETS, "epochs": EPOCHS}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--no-train": o["train"] = False
        elif a == "--datasets": o["datasets"] = next(it).split(",")
        elif a == "--n": o["n"] = [int(x) for x in next(it).split(",")]
        elif a == "--subsets": o["subsets"] = int(next(it))
        elif a == "--epochs": o["epochs"] = int(next(it))
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


# ------------------------------ reduced models ------------------------------

def build_model(gen, P, K, tr_idx, re_, dt, path):
    """Returns (step, s, to_b, from_b, R2) for either generator.
    to_b: vector-unit POD coefficients -> the model's standardized state b; from_b: back."""
    A = np.asarray(P["A"][:, :K], dtype=np.float64)
    if gen == "galerkin":
        beta, s, R2, _ = gd.fit_galerkin(A, tr_idx, dt, RIDGE)
        step = gd.make_step(beta, "ode", dt, SUBSTEPS)
        return step, s, (lambda a: a / s), (lambda b: b * s), R2
    C, H, W = P["shape"]
    dx, dy = gns.grid_spacing(path); kappa = np.sqrt(dx * dy)
    U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float64)
    Pu = np.empty((K + 1, H, W)); Pv = np.empty((K + 1, H, W))
    mean_f = P["x_mean"].astype(np.float64).reshape(C, H, W)
    Pu[0], Pv[0] = mean_f[0], mean_f[1]
    modes = (U / kappa).T.reshape(K, C, H, W); Pu[1:], Pv[1:] = modes[:, 0], modes[:, 1]
    del U, modes
    l, q = gns.galerkin_operators_ns(Pu, Pv, dx, dy, re_)
    del Pu, Pv
    A_phys = A * kappa
    R2 = gns.derivative_R2(l, q, A_phys, tr_idx, dt)
    s = A_phys.std(axis=0); s = np.where(s < 1e-14, 1.0, s)
    step = gns.make_step_ns(l, q, s, dt, SUBSTEPS)
    return step, s, (lambda a: a * kappa / s), (lambda b: b * s / kappa), R2


# ------------------------------ bookkeeping ------------------------------

def key(r):
    return (r["dataset"], str(r["n_real"]), str(r["subset"]), r["kind"], r.get("generator", ""),
            str(r.get("mode_level", "")), str(r.get("fraction", "")))


def load_done(path):
    if not os.path.exists(path):
        return {}, set()
    rows = list(csv.DictReader(open(path, newline="")))
    base = {(r["dataset"], r["n_real"], r["subset"]): float(r["Ek"]) for r in rows if r["kind"] == "baseline"}
    done = {key(r) for r in rows}
    # a failed generation is deterministic: count it as done so relaunches do not retry it
    done |= {(r["dataset"], r["n_real"], r["subset"], "aug", r["generator"], r["mode_level"], r["fraction"])
             for r in rows if r["kind"] == "gen_failed"}
    return base, done


def append(path, row):
    append_row(path, FIELDS, row, makedirs=True)


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


# ------------------------------ main ------------------------------

def main():
    o = _args(sys.argv[1:])
    out = CSV_OUT if o["train"] else CSV_GEN
    base_done, done = load_done(out)

    # ---- plan ----
    n_base = n_aug = 0; samples = 0
    for ds in o["datasets"]:
        for n in o["n"]:
            for sub in range(o["subsets"]):
                if (ds, str(n), str(sub), "baseline", "", "", "") not in done:
                    n_base += 1; samples += n
                for lv in MODE_LEVELS:
                    for gen in GENERATORS:
                        for f in FRACTIONS:
                            if (ds, str(n), str(sub), "aug", gen, str(lv), str(f)) not in done:
                                n_aug += 1; samples += n + n_aug_for(f, n)
    hours = samples * SEC_PER_SAMPLE * o["epochs"] / 500 / 3600
    print(f"[plan]  datasets {o['datasets']}, n {o['n']}, {o['subsets']} subsets, modes {MODE_LEVELS}, "
          f"generators {GENERATORS}, fractions {FRACTIONS}, {o['epochs']} epochs")
    print(f"[plan]  to do: {n_base} baseline + {n_aug} augmented trainings"
          + (f", about {hours:.1f} h of training plus generation" if o["train"] else " (no-train: generation checks only)"))
    print(f"[plan]  results -> {out}", flush=True)
    if o["plan"]:
        return

    T0 = time.time()
    for ds in o["datasets"]:
        _, _, latent = REGISTRY[ds]
        data, re_, dt, path = load(ds)
        Nt = len(data)
        val_idx, pool_idx, n_val = split_indices(Nt, None, "random", SPLIT_SEED)
        pool_set = set(int(i) for i in pool_idx)
        val_real = data[val_idx]
        steps = max(1, int(round(GEN_TIME / dt)))
        for n in o["n"]:
            for sub in range(o["subsets"]):
                todo = [(lv, gen, f) for lv in MODE_LEVELS for gen in GENERATORS for f in FRACTIONS
                        if (ds, str(n), str(sub), "aug", gen, str(lv), str(f)) not in done]
                need_base = (ds, str(n), str(sub), "baseline", "", "", "") not in done
                if not todo and not need_base:
                    continue
                rng = np.random.default_rng([DRAW_SEED, int(re_), n, sub])
                tr_idx = draw_blocks(pool_set, Nt, n, rng)
                train_real = data[tr_idx]
                tr_set = set(int(i) for i in tr_idx)
                print(f"\n[map]  ===== {ds}  n_real {n}  subset {sub}  ({len(tr_idx) // BLOCK} blocks, latent {latent}) =====", flush=True)
                common = dict(dataset=ds, Re=re_, dt=dt, latent=latent, n_real=n, subset=sub, n_val=n_val)

                # real-only baseline
                if need_base and o["train"]:
                    ek, detR, _, wall = vae.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32), val_real,
                                                  tag=f"{ds} n={n} s{sub} real", latent=latent, epochs=o["epochs"])
                    base_done[(ds, str(n), str(sub))] = float(ek)
                    append(out, dict(common, date=now(), kind="baseline", fraction=0, n_aug=0, n_train=n, Ek=round(float(ek), 4),
                                     e=round(1 - float(ek) / 100, 6), detR=round(float(detR), 6), status="ok", wall_s=round(wall)))
                    print(f"[map]  baseline Ek {ek:.2f}%", flush=True)
                base_ek = base_done.get((ds, str(n), str(sub)))
                if not todo:
                    continue

                P = subset_pod(train_real)
                win_starts = fidelity_windows(tr_set, Nt, steps, rng)
                windows = [data[t:t + steps + 1] for t in win_starts]
                for lv in MODE_LEVELS:
                    todo_lv = [(gen, f) for (l_, gen, f) in todo if l_ == lv]
                    if not todo_lv:
                        continue
                    K = k_for(P, lv)
                    ceiling, win_A = ceiling_and_project(P, K, val_real, windows)
                    print(f"[map]  modes for {lv:.0%}: K = {K}  ({100 * P['e_cum'][K - 1]:.2f}% subset energy), "
                          f"POD ceiling {ceiling:.2f}%, {len(win_A)} fidelity windows of {steps} steps", flush=True)
                    for gen in GENERATORS:
                        fr = [f for (g_, f) in todo_lv if g_ == gen]
                        if not fr:
                            continue
                        row0 = dict(common, generator=gen, mode_level=lv, K=K, energy_K_pct=round(100 * P["e_cum"][K - 1], 3),
                                    pod_ceiling_Ek=round(ceiling, 3), fid_steps=steps, fid_windows=len(win_A))
                        t0 = time.time()
                        try:
                            step, s, to_b, from_b, R2 = build_model(gen, P, K, tr_idx, re_, dt, path)
                            ferr, fblow = fidelity(step, to_b, from_b, win_A, steps)
                            row0.update(fid_err=round(ferr, 5), fid_blowup_frac=round(fblow, 3), gen_R2_min=round(float(np.min(R2)), 4))
                            n_max = max(n_aug_for(f, n) for f in fr)
                            # (the two integrate_trajectories of the old generators were identical copies)
                            aug, st = synthetic_arm(step, s, to_b, from_b, P, np.asarray(P["A"][:, :K], dtype=np.float64),
                                                    K, n_max, steps,
                                                    np.random.default_rng([DRAW_SEED, int(re_), n, sub, K, GENERATORS.index(gen)]),
                                                    kick=IC_NOISE, energy_tol=ENERGY_TOL, max_traj=MAX_TRAJ)
                            row0.update(gen_keep_frac=round(st["keep_frac"], 4), gen_n_traj=st["n_traj"], gen_seconds=round(time.time() - t0))
                            print(f"[map]  {gen:11s} K={K}: fidelity error {ferr:.3f} (blow-ups {fblow:.0%}), "
                                  f"{n_max} synthetic from {st['n_traj']} trajectories, {100 * st['keep_frac']:.0f}% kept "
                                  f"[{time.time() - t0:.0f} s]", flush=True)
                        except Exception as err:
                            msg = str(err).splitlines()[0][:160]
                            print(f"[map]  !! {gen} K={K}: generation failed ({msg})", flush=True)
                            for f in fr:
                                append(out, dict(row0, date=now(), kind="gen_failed", fraction=f, status=f"failed: {msg}",
                                                 gen_seconds=round(time.time() - t0)))
                                done.add((ds, str(n), str(sub), "aug", gen, str(lv), str(f)))
                            continue
                        for f in fr:
                            na = n_aug_for(f, n)
                            row = dict(row0, date=now(), kind="aug", fraction=f, n_aug=na, n_train=n + na,
                                       baseline_Ek=round(base_ek, 4) if base_ek is not None else "")
                            if base_ek is not None:
                                row["headroom"] = round(ceiling - base_ek, 3)
                            if o["train"]:
                                ek, detR, _, wall = vae.train(train_real, aug[:na], val_real, tag=f"{ds} n={n} s{sub} {gen} K={K} f={f}",
                                                              latent=latent, epochs=o["epochs"])
                                row.update(Ek=round(float(ek), 4), e=round(1 - float(ek) / 100, 6), detR=round(float(detR), 6),
                                           status="ok", wall_s=round(wall),
                                           gain=round(float(ek) - base_ek, 3) if base_ek is not None else "")
                                print(f"[map]  {gen:11s} K={K} f={f}: Ek {ek:.2f}%"
                                      + (f"  gain {float(ek) - base_ek:+.2f}  headroom {ceiling - base_ek:.1f}" if base_ek is not None else "")
                                      + f"   [{(time.time() - T0) / 3600:.1f} h elapsed]", flush=True)
                            else:
                                row.update(status="generated")
                            append(out, row)
                        del aug
                del P, train_real, windows
        del data, val_real
    print(f"\n[map]  done in {(time.time() - T0) / 3600:.2f} h  ->  {out}")


if __name__ == "__main__":
    main()

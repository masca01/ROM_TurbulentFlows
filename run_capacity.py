"""
CAPACITY CEILINGS: what a network of a given latent dimension reaches when data are not the limit.

headroom = min(POD ceiling, capacity ceiling) - real-only Ek needs, for every (flow, latent) pair
used in the study, the Ek that network reaches with the whole training pool. Most pairs already
have that number from the convergence study; two do not:

    Re 100, latent 25  - never trained on the full pool, only bounded below by the latent-15 value
    Re 100, latent  5  - 47.9% at 775 snapshots in the tolerance/modes grid, a curve still rising

This script measures them directly: same recipe as every training in the study (500 epochs,
beta 5e-3, batch 32, lr 3e-4, torch seed 7), real snapshots only, the full training pool of the
random 10% split. Latent 15 is included as a control: it should reproduce the convergence study's
90.0%, which checks that this recipe and that one agree.

Output: ../../convergence/capacity_ceilings.csv (resumable). capacity_headroom.py reads this file
when it exists and falls back to its built-in table otherwise, so recomputing every headroom after
this run is a single command that takes no simulation at all.

Cost: one training per (flow, latent), the full pool each time. Re 100: 900 snapshots, about
20 minutes per training.

Usage:
    python3 run_capacity.py                       # Re 100 at latent 5, 15, 25
    python3 run_capacity.py --plan                # what is missing and the time estimate
    python3 run_capacity.py --datasets Re50 --latents 5,15   # any other pair
    python3 run_capacity.py --latents 25 --epochs 2          # smoke test
"""
import os, sys, csv, time, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_region_map as rm
import re100_fraction_sweep as rs

# ============ CONFIG ============
DATASETS = ["Re100"]
LATENTS  = [5, 15, 25]
FILES    = dict(rm.DATASETS, Re60=("Alpha0/dataRe60Alpha0_2.mat", 1500, 7),
                             Re70=("Alpha0/dataRe70Alpha0_2.mat", 1500, 9))
CSV_OUT  = os.path.join(rm._CONV, "capacity_ceilings.csv")
FIELDS   = ["date", "dataset", "Re", "latent", "n_train", "n_val", "epochs", "Ek", "e", "detR",
            "pod_ceiling_pool", "wall_s", "note"]
# ================================
rm.DATASETS = FILES


def _args(argv):
    o = {"plan": False, "datasets": DATASETS, "latents": LATENTS, "epochs": rm.EPOCHS, "csv": CSV_OUT}
    it = iter(argv)
    for a in it:
        if a == "--plan": o["plan"] = True
        elif a == "--datasets": o["datasets"] = next(it).split(",")
        elif a == "--latents": o["latents"] = [int(x) for x in next(it).split(",")]
        elif a == "--epochs": o["epochs"] = int(next(it))
        elif a == "--csv": o["csv"] = os.path.abspath(next(it))
        else: raise SystemExit(f"unknown argument {a!r}; see the docstring")
    return o


def done(path):
    if not os.path.exists(path):
        return set()
    return {(r["dataset"], r["latent"], r["epochs"]) for r in csv.DictReader(open(path, newline=""))}


def main():
    o = _args(sys.argv[1:])
    rs.EPOCHS = o["epochs"]
    have = done(o["csv"])
    todo = [(ds, lat) for ds in o["datasets"] for lat in o["latents"]
            if (ds, str(lat), str(o["epochs"])) not in have]
    # pool size: 90% of the record (the random split of every study)
    pool = {"Re100": 900}.get(o["datasets"][0], 1350)
    hours = len(todo) * pool * rm.SEC_PER_SAMPLE * o["epochs"] / 500 / 3600
    print(f"[cap]  {len(todo)} trainings on the full pool (~{pool} snapshots each), "
          f"about {hours:.1f} h  ->  {o['csv']}", flush=True)
    for ds, lat in todo:
        print(f"        {ds}  latent {lat}")
    if o["plan"] or not todo:
        return

    T0 = time.time()
    by_ds = {}
    for ds, lat in todo:
        by_ds.setdefault(ds, []).append(lat)
    for ds, lats in by_ds.items():
        data, re_, dt, path = rm.load(ds)
        Nt = len(data)
        val_idx, pool_idx, _ = rm.cs.split_indices(Nt, None, "random", rm.SPLIT_SEED)
        train_real, val_real = data[pool_idx], data[val_idx]
        P = rm.subset_pod(train_real)                         # POD ceiling of the whole pool, for context
        K = rm.k_for(P, 0.99)
        ceiling, _ = rm.ceiling_and_project(P, K, val_real, [])
        print(f"[cap]  {ds}: pool {len(train_real)} snapshots, validation {len(val_real)}, "
              f"POD ceiling of the pool at 99% energy ({K} modes) = {ceiling:.2f}%", flush=True)
        del P
        for lat in lats:
            rs.LATENT = lat
            ek, detR, _, wall = rs.train(train_real, np.empty((0,) + train_real.shape[1:], np.float32),
                                         val_real, tag=f"{ds} full pool latent {lat}")
            with open(o["csv"], "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if f.tell() == 0: w.writeheader()
                w.writerow(dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), dataset=ds, Re=re_,
                                latent=lat, n_train=len(train_real), n_val=len(val_real), epochs=o["epochs"],
                                Ek=round(float(ek), 4), e=round(1 - float(ek) / 100, 6),
                                detR=round(float(detR), 6), pod_ceiling_pool=round(ceiling, 3),
                                wall_s=round(wall), note="real snapshots only, full training pool"))
            print(f"[cap]  {ds} latent {lat}: capacity ceiling {ek:.2f}%   "
                  f"[{(time.time() - T0) / 3600:.1f} h]", flush=True)
        del data, train_real, val_real
    print(f"\n[cap]  done in {(time.time() - T0) / 3600:.2f} h  ->  {o['csv']}")
    print("[cap]  now rerun capacity_headroom.py to refresh every corrected headroom (takes no time)")


if __name__ == "__main__":
    main()

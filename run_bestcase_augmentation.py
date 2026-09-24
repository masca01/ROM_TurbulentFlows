"""
Driver: the BEST-CASE augmentation experiment on every dataset of the Re sweep.

For each dataset (Re = 50, 60, 70, 80 at alpha 0, plus the old Re100 set), on the
random split of the convergence study (validation = the same 10 % real
snapshots, cap 1500):
    1. pod_augment_galerkin_ns_bestcase.py --file <f> --n-real N_REAL --n-aug pool
                                            --n-rom k99 --split SPLIT
       -> POD on the SAME N_REAL-snapshot subset as the convergence sweep's first
          draw, NS-projected Galerkin model on the modes holding 99 % of that
          subset's energy (automatic back-off if it drifts), synthetic snapshots
          filling the pool budget (1350 - N_REAL; Re100: 900 - N_REAL)
       -> ../DATA/AUGMENTED/<base>_aug_galerkin_ns_n<N_REAL>.npz   (~5 GB each)
    2. bestcase_train.py <bundle> <latent>
       -> trains the convergence-study beta-VAE on real + synthetic, and once on
          the real subset alone (baseline), same seed as the sweep point
       -> ONE row in ../../convergence/bestcase_augmentation_results.csv with
          the reference numbers of the convergence study beside them.

Re-running the driver is safe: datasets that already have a row in the CSV (same
split / strategy) are skipped (SKIP_DONE), and existing bundles are reused, so a
second run only does what is missing or failed.

Reading the CSV:  Ek_baseline (real only) -> Ek_aug (real + synthetic);
NN_conv_Ek_full is the ceiling (all real snapshots); gap_closed = the fraction
of the way from the baseline to that ceiling the synthetic data carried the net.

Budget: per dataset ~2 min generation + ~5 min baseline + ~30 min augmented
training (1350 snapshots x 500 epochs), i.e. ~3 h for the five datasets.
"""
import os, sys, subprocess, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG  = os.path.join(_DATA, "AUGMENTED")

# ============ CONFIG ============
SPLIT   = "tail5"      # the best case is the random split (see the assessment)
N_REAL  = 250           # real training snapshots: a sweep point of nn_convergence.py
                        # (5,10,15,20,30,50,100,150,250,400,650,900,1150)
N_AUG   = "pool"        # synthetic snapshots: "pool" = fill up to the pool size
                        # (1350 - N_REAL; Re100: 900 - N_REAL), or an int
N_ROM   = "k99"         # Galerkin model size: "k99" / "k95" of the subset, or an int
DATASETS = [            # (relative data file, NN latent dim of the convergence study)
    ("Alpha0/dataRe50Alpha0_2.mat",          5),
    ("Alpha0/dataRe60Alpha0_2.mat",          7),
    ("Alpha0/dataRe70Alpha0_2.mat",          9),
    ("Alpha0/dataRe80Alpha0_2.mat",         11),
    ("2PlatesGap/Data2PlatesGap1Re100.mat", 15),
]
RUN_BASELINE   = True   # also train on the real subset alone (recommended)
REUSE_BUNDLES  = True   # skip generation if the bundle already exists
SKIP_DONE      = True   # skip a dataset entirely if the results CSV already has a row
                        # for it (same split, strategy and n_real) -> re-running the
                        # driver only does the datasets that are missing or failed
DELETE_BUNDLES = False  # remove the ~5 GB bundle after training
# ================================

_ENV = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")


def _run(script, *args):
    cmd = [sys.executable, os.path.join(_HERE, script), *map(str, args)]
    print(f"\n>>> {' '.join(os.path.basename(c) for c in cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=_HERE, env=_ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"!! {script} failed on {args} (exit {proc.returncode})")
    print(f"    done in {time.time()-t0:.0f} s", flush=True)


_CSV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence",
                                     "bestcase_augmentation_results.csv"))


def _done(base, strategy):
    """True if the results CSV already holds a row for this dataset / split / strategy."""
    if not os.path.exists(_CSV):
        return False
    import csv
    with open(_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("dataset") == base and r.get("split") == SPLIT
                    and r.get("strategy") == strategy and r.get("Ek_aug", "") != ""):
                return True
    return False


def main():
    strategy = f"galerkin_ns_n{N_REAL}" + ("" if SPLIT == "random" else f"_{SPLIT}")
    print(f"######## best-case augmentation  split = {SPLIT}   n_real = {N_REAL}   "
          f"n_aug = {N_AUG}   ROM = {N_ROM}   ########", flush=True)
    T0 = time.time()
    for fn, latent in DATASETS:
        path = os.path.join(_DATA, fn)
        base = os.path.splitext(os.path.basename(fn))[0]
        if not os.path.exists(path):
            print(f"!! missing {path} — skipping"); continue
        print(f"\n================  {base}  (latent {latent})  ================", flush=True)
        if SKIP_DONE and _done(base, strategy):
            print(f"    already in {os.path.basename(_CSV)} (split {SPLIT}, {strategy}) — skipping",
                  flush=True)
            continue
        bundle = os.path.join(_AUG, f"{base}_aug_{strategy}.npz")
        if not (REUSE_BUNDLES and os.path.exists(bundle)):
            _run("pod_augment_galerkin_ns_bestcase.py", "--file", path, "--split", SPLIT,
                 "--n-real", N_REAL, "--n-aug", N_AUG, "--n-rom", N_ROM)
            if not os.path.exists(bundle):                 # small bundles are saved as .mat
                alt = bundle[:-4] + ".mat"
                if os.path.exists(alt): bundle = alt
                else: raise SystemExit(f"!! bundle not found after generation: {bundle}")
        else:
            print(f"    reusing {bundle}")
        args = [bundle, latent] + ([] if RUN_BASELINE else ["--no-baseline"])
        _run("bestcase_train.py", *args)
        if DELETE_BUNDLES:
            os.remove(bundle); print(f"    removed {bundle}")

    csv_path = _CSV
    print(f"\nAll datasets processed in {(time.time()-T0)/60:.0f} min.  Results -> {csv_path}")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"\n{'dataset':24s} {'n_real':>6} {'n_aug':>6} {'ROM':>4} {'Ek_base':>8} {'Ek_aug':>8} "
              f"{'gain':>6} {'Ek_full':>8} {'closed':>7}")
        for r in rows:
            if r.get("split") != SPLIT:
                continue
            print(f"{r['dataset']:24s} {r['n_real']:>6} {r['n_aug']:>6} {r['n_rom']:>4} "
                  f"{r['Ek_baseline']:>8} {r['Ek_aug']:>8} {r['gain_pts']:>6} "
                  f"{r['NN_conv_Ek_full']:>8} {r['gap_closed']:>7}")


if __name__ == "__main__":
    main()

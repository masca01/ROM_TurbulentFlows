"""
Driver: run the POD and beta-VAE convergence studies over the Reynolds-number
sweep (Alpha0: Re = 50, 60, 70, 80 at alpha = 0, plus the old Re100 dataset) on
a COMMON snapshot budget so the curves are comparable across Re.

Every run is capped at CAP = 1500 snapshots via the T_MAX command-line argument
that pod_convergence.py / nn_convergence.py accept (Re50 has 5000 -> first 1500;
Re100 only has 1000 -> it uses all of them, pool 900).

SPLIT picks how the validation set is chosen ("random" = 10% at random, the
original results; "tail" = the last 10% of the record; "tail5" = the last 5%,
"tail2.5" = the last 2.5% of the record); each split has its own output folder, CSV and Excel, so they
never overwrite each other (see convergence_split.py).

Per dataset, DATASETS[SPLIT] fixes the NN latent dimension and the POD truncations:
one k (modes retaining >= L % of the pool energy, from assess_pod_modes.py) per
energy level L.  The POD study is run once per level in RUN[SPLIT]["POD_LEVELS"]
and every level's results are kept side by side (tagged _pod95_, _pod99_, ...).

For each dataset it:
    1. runs pod_convergence.py  <file> <CAP> <k_L> <L> <SPLIT>  per level L
    2. runs nn_convergence.py   <file> <CAP> <latent> <SPLIT>
Each sub-run also appends its own columns (POD95_*, POD99_*, NN_*) to the shared
CSV <dataset folder>/convergence_last_point[_tailval].csv (convergence_csv.py),
so the last-convergence points are recorded as the runs go.  Then, once at the
end (unless BUILD_EXCEL is False):
    3. runs make_alpha0_excel.py <SPLIT>, which builds ONE combined Excel with a
       Summary sheet + a sheet per dataset from the _pod95_/_pod99_/_nn_
       convergence.mat bundles in ../DATA/AUGMENTED[_TAILVAL]/.

Note: the beta-VAE runs are the expensive part (training a VAE at every sweep
point, for every dataset).  Set RUN_NN = False to do just the (much faster) POD
sweeps first.
"""
import os, sys, subprocess, time

from rom import paths

_HERE     = os.path.dirname(os.path.abspath(__file__))     # the sibling scripts it launches
_DATA     = paths.DATA

# ============ CONFIG ============
# SPLIT: how the 10% validation set is chosen (convergence_split.py):
#   "random" -> 10% random snapshots (fixed seed)   [the original results]
#               ../DATA/AUGMENTED/, convergence_last_point.csv, Alpha0_convergence.xlsx
#   "tail"   -> the LAST 10% of the record (e.g. train pool = first 1350 snapshots,
#               validation = last 150); random n-subsets are still drawn from the pool
#               ../DATA/AUGMENTED_TAILVAL/, convergence_last_point_tailval.csv,
#               Alpha0_convergence_tailval.xlsx
#   "tail5"   -> the LAST 5% of the record (train pool = first 1425 snapshots,
#               validation = last 75; Re100: pool 950, validation 50)
#               ../DATA/AUGMENTED_TAILVAL5/, convergence_last_point_tailval5.csv,
#               Alpha0_convergence_tailval5.xlsx
#   "tail2.5" -> the LAST 2.5% of the record (train pool = first 1462 snapshots,
#               validation = last 38; Re100: pool 975, validation 25)
#               ../DATA/AUGMENTED_TAILVAL2.5/, convergence_last_point_tailval2.5.csv,
#               Alpha0_convergence_tailval2.5.xlsx
# The splits write to DIFFERENT files, so running one never alters the others.
SPLIT = "tail5"

# Per split: (relative data file, NN latent dimension, {energy level %: POD modes})
# latent dims chosen per Re; POD modes = k retaining >= L% of the POOL energy, so
# they depend on the split (assess_pod_modes.py [random|tail] ->
# ../../convergence/pod_modes_assessment[_tailval].csv).
DATASETS = {
    "random": [                       #            k_95   k_99
        ("Alpha0/dataRe50Alpha0_2.mat",         5,  {95: 12,  99: 26}),
        ("Alpha0/dataRe60Alpha0_2.mat",         7,  {95: 36,  99: 70}),
        ("Alpha0/dataRe70Alpha0_2.mat",         9,  {95: 50,  99: 89}),
        ("Alpha0/dataRe80Alpha0_2.mat",        11,  {95: 15,  99: 36}),
        ("2PlatesGap/Data2PlatesGap1Re100.mat", 15, {95: 106, 99: 226}),  # 1000 snaps
    ],
    "tail": [                         # pool = first 90% of the record
        ("Alpha0/dataRe50Alpha0_2.mat",         5,  {95: 12,  99: 25}),
        ("Alpha0/dataRe60Alpha0_2.mat",         7,  {95: 35,  99: 66}),
        ("Alpha0/dataRe70Alpha0_2.mat",         9,  {95: 47,  99: 83}),
        ("Alpha0/dataRe80Alpha0_2.mat",        11,  {95: 14,  99: 34}),
        ("2PlatesGap/Data2PlatesGap1Re100.mat", 15, {95: 104, 99: 220}),  # 1000 snaps
    ],
    "tail5": [                        # pool = first 95% of the record
        ("Alpha0/dataRe50Alpha0_2.mat",         5,  {95: 12,  99: 26}),
        ("Alpha0/dataRe60Alpha0_2.mat",         7,  {95: 35,  99: 68}),
        ("Alpha0/dataRe70Alpha0_2.mat",         9,  {95: 49,  99: 86}),
        ("Alpha0/dataRe80Alpha0_2.mat",        11,  {95: 14,  99: 35}),
        ("2PlatesGap/Data2PlatesGap1Re100.mat", 15, {95: 106, 99: 226}),  # 1000 snaps
    ],
    "tail2.5": [                      # pool = first 97.5% of the record
        ("Alpha0/dataRe50Alpha0_2.mat",         5,  {95: 12,  99: 26}),
        ("Alpha0/dataRe60Alpha0_2.mat",         7,  {95: 36,  99: 69}),
        ("Alpha0/dataRe70Alpha0_2.mat",         9,  {95: 50,  99: 87}),
        ("Alpha0/dataRe80Alpha0_2.mat",        11,  {95: 14,  99: 36}),
        ("2PlatesGap/Data2PlatesGap1Re100.mat", 15, {95: 107, 99: 228}),  # 1000 snaps
    ],
}
CAP = 1500              # common snapshot budget (Re100 has 1000 -> uses all of them)

# Per split: what to run.  "random": POD@95 + NN are done and kept, only POD@99
# is pending.  "tail" / "tail5" / "tail2.5": everything (both POD truncations + NN).
RUN = {
    "random": dict(POD_LEVELS=[99],     RUN_POD=True, RUN_NN=False),
    "tail":   dict(POD_LEVELS=[95, 99], RUN_POD=True, RUN_NN=True),
    "tail5":   dict(POD_LEVELS=[95, 99], RUN_POD=True, RUN_NN=True),
    "tail2.5": dict(POD_LEVELS=[95, 99], RUN_POD=True, RUN_NN=True),
}
BUILD_EXCEL = True      # at the end, build the combined Excel of THIS split
# ================================

_ENV = dict(os.environ, MPLBACKEND="Agg",   # never pop up a window in batch mode
            PYTHONUNBUFFERED="1")            # stream child stdout live (no buffering)


def _run(script, *args):
    cmd = [sys.executable, os.path.join(_HERE, script), *map(str, args)]
    print(f"\n>>> {' '.join(os.path.basename(c) for c in cmd)}", flush=True)
    t0 = time.time()
    # Stream the child's stdout line-by-line through THIS process so the output is
    # visible even in IDLE, which does NOT display a subprocess's own stdout.
    proc = subprocess.Popen(cmd, cwd=_HERE, env=_ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"!! {script} failed on {args} (exit {proc.returncode})")
    print(f"    done in {time.time()-t0:.0f} s", flush=True)


def main():
    if SPLIT not in DATASETS:
        raise SystemExit(f"SPLIT must be one of {list(DATASETS)}, got {SPLIT!r}")
    cfg = RUN[SPLIT]
    print(f"######## split = {SPLIT}   POD levels {cfg['POD_LEVELS']}   "
          f"POD {'on' if cfg['RUN_POD'] else 'off'}   NN {'on' if cfg['RUN_NN'] else 'off'}"
          f"   ########", flush=True)
    for fn, latent, modes in DATASETS[SPLIT]:
        path = os.path.join(_DATA, fn)
        base = os.path.splitext(os.path.basename(fn))[0]
        if not os.path.exists(path):
            print(f"!! missing {path} — skipping"); continue
        mstr = ", ".join(f"{lv}%: top-{k}" for lv, k in sorted(modes.items()))
        print(f"\n================  {base}  (split {SPLIT}, cap {CAP}, latent {latent}, "
              f"POD modes {mstr})  ================", flush=True)
        if cfg["RUN_POD"]:
            for lv in cfg["POD_LEVELS"]:
                if modes.get(lv) is None:
                    print(f"!! {base}: no k for {lv}% energy in DATASETS[{SPLIT!r}] "
                          f"(run assess_pod_modes.py {SPLIT} and fill it) — skipping",
                          flush=True)
                    continue
                _run("pod_convergence.py", path, CAP, modes[lv], lv, SPLIT)
        if cfg["RUN_NN"]:
            _run("nn_convergence.py", path, CAP, latent, SPLIT)

    if BUILD_EXCEL:
        _run("make_alpha0_excel.py", SPLIT)  # one combined Excel for THIS split
    print(f"\nAll datasets processed  (split = {SPLIT}).")


if __name__ == "__main__":
    main()

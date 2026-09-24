"""Assess how many POD modes retain >= 90 / 95 / 99 / 99.9 % of the fluctuation
energy, for every dataset in the study (Alpha0 Re-sweep + the old Re100 dataset).

Uses the FULL training pool of the exact same fixed 90/10 split as the
convergence scripts (see convergence_split.py; VAL_FRAC per split mode, cap 1500),
centers by the pool mean, and reads the cumulative eigenvalue spectrum of the
snapshot Gram matrix.  The reported k_95 / k_99 are the truncations used as
N_MODES in pod_convergence.py (driver: run_alpha0_convergence.py DATASETS).

Usage:  python3 assess_pod_modes.py [random|tail]
    random (default)  validation = random 10%  -> pod_modes_assessment.csv
    tail              validation = last 10%    -> pod_modes_assessment_tailval.csv
    tail5             validation = last 5%     -> pod_modes_assessment_tailval5.csv
    tail2.5           validation = last 2.5%   -> pod_modes_assessment_tailval2.5.csv

Output: printed table + ../../convergence/pod_modes_assessment[_tailval].csv
        + ../../convergence/pod_spectra[_tailval].npz (full spectra; new
          thresholds need no recompute — just change LEVELS and re-read the .npz)
"""
import os, sys, csv
import numpy as np

from rom.data import load_data
from rom import split as cs

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "DATA"))

SPLIT = cs.check_mode(sys.argv[1] if len(sys.argv) > 1 else "random")
VAL_FRAC = cs.val_frac(SPLIT)   # validation fraction of THIS split mode (0.10 or 0.025)
_OUT     = cs.assessment_path(SPLIT)
_SPECTRA = cs.spectra_path(SPLIT)

DATASETS = [
    ("Re50",  os.path.join(_DATA, "Alpha0", "dataRe50Alpha0_2.mat")),
    ("Re60",  os.path.join(_DATA, "Alpha0", "dataRe60Alpha0_2.mat")),
    ("Re70",  os.path.join(_DATA, "Alpha0", "dataRe70Alpha0_2.mat")),
    ("Re80",  os.path.join(_DATA, "Alpha0", "dataRe80Alpha0_2.mat")),
    ("Re100", os.path.join(_DATA, "2PlatesGap", "Data2PlatesGap1Re100.mat")),
]
CAP        = 1500     # common snapshot budget (Re100 only has 1000 -> uses all)
SPLIT_SEED = 7        # MUST match pod_convergence.py / nn_convergence.py
LEVELS     = [0.90, 0.95, 0.99, 0.999]


def spectrum_for(path):
    data, _ = load_data(path, None, 1, CAP)
    Nt = data.shape[0]
    X = data.reshape(Nt, -1)                      # float32, rows = snapshots
    del data
    # same fixed split as the convergence scripts -> use the FULL training pool
    _, pool, _ = cs.split_indices(Nt, VAL_FRAC, SPLIT, SPLIT_SEED)
    Xp = X[pool]
    del X
    Xp = Xp - Xp.mean(axis=0, keepdims=True)      # center by the POOL mean
    G = Xp @ Xp.T                                 # n x n Gram (float32 matmul)
    del Xp
    lam = np.linalg.eigvalsh(G.astype(np.float64))[::-1]
    lam = np.clip(lam, 0.0, None)
    return Nt, len(pool), lam


def _klabel(lv):
    return f"k_{100*lv:g}"                     # k_90, k_95, k_99, k_99.9


def main():
    print(f"[assess_pod_modes]  split = {SPLIT}  (validation = {cs.describe(SPLIT)})",
          flush=True)
    rows, spectra = [], {}
    for name, path in DATASETS:
        if not os.path.exists(path):
            print(f"!! {name}: missing {path} — skipped", flush=True)
            continue
        print(f"\n=== {name}  ({os.path.basename(path)}) ===", flush=True)
        Nt, n_pool, lam = spectrum_for(path)
        spectra[name] = lam
        cum = np.cumsum(lam) / lam.sum()
        ks = {lv: int(np.searchsorted(cum, lv) + 1) for lv in LEVELS}
        print(f"  {Nt} snapshots (pool {n_pool})")
        for lv in LEVELS:
            print(f"  k for {100*lv:5.1f}% energy :  {ks[lv]:4d} modes", flush=True)
        row = {"dataset": name, "split": SPLIT, "Nt": Nt, "n_pool": n_pool}
        row.update({_klabel(lv): ks[lv] for lv in LEVELS})
        rows.append(row)

    if rows:
        with open(_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        # full eigenvalue spectra, so new thresholds need no recompute
        np.savez(_SPECTRA, **spectra)
        print(f"\nsaved -> {_OUT}\nsaved -> {_SPECTRA}", flush=True)
        hdr = " ".join(f"{_klabel(lv):>7}" for lv in LEVELS)
        print(f"\n{'dataset':>8} {hdr}")
        for r in rows:
            vals = " ".join(f"{r[_klabel(lv)]:7d}" for lv in LEVELS)
            print(f"{r['dataset']:>8} {vals}")


if __name__ == "__main__":
    main()

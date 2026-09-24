"""
One place that defines HOW the record is split into training pool / validation
for the convergence studies, and WHERE each variant stores its results, so the
POD script, the NN script, the mode assessment, the driver and the Excel builder
all agree.

SPLIT modes
    "random"  (original)  validation = 10% of the snapshots picked at random
                          (fixed seed), pool = the other 90%.
                          -> ../DATA/AUGMENTED/,  convergence_last_point.csv,
                             Alpha0_convergence.xlsx
    "tail"                validation = the LAST 10% of the record (e.g. the last
                          150 of 1500 snapshots), pool = the first 90%.  Random
                          subsets of n snapshots are still drawn from the pool.
                          -> ../DATA/AUGMENTED_TAILVAL/,
                             convergence_last_point_tailval.csv,
                             Alpha0_convergence_tailval.xlsx
    "tail5"               same as "tail" but validation = the LAST 5% only
                          (75 of 1500 snapshots -> pool 1425; 50 of 1000 -> pool 950).
                          -> ../DATA/AUGMENTED_TAILVAL5/,
                             convergence_last_point_tailval5.csv,
                             Alpha0_convergence_tailval5.xlsx
    "tail2.5"             same as "tail" but validation = the LAST 2.5% only
                          (38 of 1500 snapshots -> pool 1462; 25 of 1000 -> pool 975).
                          -> ../DATA/AUGMENTED_TAILVAL2.5/,
                             convergence_last_point_tailval2.5.csv,
                             Alpha0_convergence_tailval2.5.xlsx

Each mode carries its own validation fraction: val_frac(mode).

The two variants never write to each other's files, so the random-split results
that already exist stay untouched.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_CONV = os.path.normpath(os.path.join(_HERE, "..", "..", "convergence"))

#          mode:  (validation fraction, file suffix, is the validation the tail?)
_MODES = {
    "random":  (0.10,  "",             False),
    "tail":    (0.10,  "_tailval",     True),
    "tail5":   (0.05,  "_tailval5",    True),
    "tail2.5": (0.025, "_tailval2.5",  True),
}
MODES = tuple(_MODES)


def check_mode(mode):
    if mode not in MODES:
        raise SystemExit(f"unknown split mode {mode!r} — use one of {MODES}")
    return mode


def val_frac(mode):
    """Fraction of ALL snapshots held out as validation in this mode."""
    return _MODES[check_mode(mode)][0]


def is_tail(mode):
    return _MODES[check_mode(mode)][2]


def split_indices(Nt, val_frac, mode, seed):
    """Return (val_idx, pool_idx, n_val), both index arrays sorted ascending.
    val_frac=None -> the mode's own fraction (val_frac(mode))."""
    check_mode(mode)
    if val_frac is None:
        val_frac = _MODES[mode][0]
    n_val = max(1, round(val_frac * Nt))
    if not _MODES[mode][2]:                     # "random"
        perm = np.random.default_rng(seed).permutation(Nt)
        val_idx, pool_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    else:                                       # "tail": last n_val snapshots
        val_idx = np.arange(Nt - n_val, Nt)
        pool_idx = np.arange(0, Nt - n_val)
    return val_idx, pool_idx, n_val


def describe(mode):
    pct = f"{100 * val_frac(mode):g}%"
    return (f"LAST {pct} of the record (tail)" if is_tail(mode)
            else f"random {pct} of the record")


def suffix(mode):
    return _MODES[check_mode(mode)][1]


def aug_dir(mode):
    """Folder for the _pod*_ / _nn_ convergence .mat bundles."""
    d = os.path.join(_DATA, "AUGMENTED" + suffix(mode).upper())
    os.makedirs(d, exist_ok=True)
    return d


def csv_path(data_file, mode):
    """Shared last-point CSV, next to the dataset file."""
    return os.path.join(os.path.dirname(os.path.abspath(data_file)),
                        f"convergence_last_point{suffix(mode)}.csv")


def excel_path(mode):
    return os.path.join(_CONV, f"Alpha0_convergence{suffix(mode)}.xlsx")


def assessment_path(mode):
    return os.path.join(_CONV, f"pod_modes_assessment{suffix(mode)}.csv")


def spectra_path(mode):
    return os.path.join(_CONV, f"pod_spectra{suffix(mode)}.npz")

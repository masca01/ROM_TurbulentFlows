"""
POD energy spectrum — how much fluctuation energy do the first n modes retain?

Classic POD truncation curve (NOT the same as pod_convergence.py):
    - build ONE POD basis from ALL snapshots (centered / fluctuation POD);
    - the singular values give the energy of each mode, sigma_i^2;
    - cumulative energy retained by the first n modes:

          E(n) = sum_{i=1..n} sigma_i^2  /  sum_i sigma_i^2   (in %)

Uses the method of snapshots (eigen-decomposition of the small Nt x Nt Gram
matrix) so it never forms anything in the full D-dimensional space.

Plot: cumulative retained energy (%) vs number of modes n.  Shown, not saved.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pod_augment_galerkin_ns import load_data

# ============ CONFIG ============
DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/2PlatesGap/Data2PlatesGap1Re50.mat"
COMP_IDX  = None      # None = auto-detect velocity components
MARK      = [90, 95, 99, 99.9]   # energy levels to annotate (in %)
# ================================


def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape
    X = data.reshape(Nt, -1).astype(np.float64)        # rows = snapshots
    del data

    # centered (fluctuation) POD, method of snapshots
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T                                       # Nt x Nt Gram matrix
    lam = np.linalg.eigvalsh(G)                         # eigenvalues = sigma_i^2
    lam = np.clip(lam[::-1], 0, None)                   # descending, non-negative

    cum = np.cumsum(lam) / lam.sum() * 100.0            # cumulative energy (%)
    n = np.arange(1, len(cum) + 1)

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    print(f"[{base}]  {Nt} snapshots  ->  {len(lam)} POD modes")
    for lvl in MARK:
        k = int(np.searchsorted(cum, lvl) + 1)
        print(f"  {k:3d} modes retain {lvl}% of the fluctuation energy")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(n, cum, marker="o", ms=4, lw=1.6, color="#d62728")
    for lvl in MARK:
        ax.axhline(lvl, ls=":", lw=0.8, color="gray")
    ax.set_xlabel("number of POD modes  n")
    ax.set_ylabel("cumulative energy retained  E(n)  [%]")
    ax.set_title(f"POD energy spectrum — {base}\n(centered POD on all {Nt} snapshots)")
    ax.set_ylim(0, 101)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
Visual sanity check of an augmentation bundle — REAL vs SYNTHETIC "video".
===========================================================================

Loads any bundle from ../DATA/AUGMENTED (the output of pod_augment_gaussian /
gmm / noise / galerkin) and plays the snapshots as an animation:

    top-left  : real training snapshots        (train_real)
    top-right : synthetic snapshots            (train_aug)
    bottom    : fluctuation energy of every snapshot in both series, with a
                moving cursor — synthetic energy should stay inside the band
                traced by the real data if the augmentation "makes sense".

Both panels share the color scale (robust percentiles of the REAL data), so
any amplitude drift of the synthetic fields is immediately visible.

For dynamical strategies (galerkin) the synthetic frames are consecutive
states of integrated trajectories, so the animation should look like a
smoothly evolving flow; for statistical strategies (gmm / joint / gaussian)
frames are independent draws, so temporal continuity is NOT expected there.

Set SAVE_PATH to write an .mp4 (needs ffmpeg) or .gif instead of opening the
interactive window.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Folder layout ──
_HERE     = os.path.dirname(os.path.abspath(__file__))
_AUG_DIR  = os.path.normpath(os.path.join(_HERE, "..", "DATA", "AUGMENTED"))

# ══ CONFIG ══════════════════════════════════════════════════════════════════
AUG_FILE  = os.path.join(_AUG_DIR, "Data2PlatesGap1Re50_aug_galerkin_ns_5traj.mat")
COMP      = 0        # component to show (0=u, 1=v, ...)
N_FRAMES  = 200      # frames to animate (None = min(#real, #synthetic))
FPS       = 10       # playback speed
SAVE_PATH = None     # e.g. ".../movie.mp4" or ".gif"; None = interactive window
# ═══════════════════════════════════════════════════════════════════════════

def load_bundle(path):
    if path.endswith(".npz"):
        S = np.load(path, allow_pickle=True)
    else:
        S = sio.loadmat(path, simplify_cells=True)
    train_real = np.asarray(S["train_real"], dtype=np.float32)
    train_aug  = np.asarray(S["train_aug"],  dtype=np.float32)
    def _fix(a):
        return a[None] if a.ndim == 3 else a
    train_real, train_aug = _fix(train_real), _fix(train_aug)
    cn = S["comp_names"] if "comp_names" in S else None
    if cn is None:
        comp_names = [f"c{i}" for i in range(train_real.shape[1])]
    elif isinstance(cn, str):
        comp_names = [cn]
    else:
        comp_names = [str(x) for x in np.atleast_1d(cn)]
    strategy = str(S["strategy"]) if "strategy" in S else "unknown"
    return train_real, train_aug, comp_names, strategy


def fluct_energy(x, mean_field):
    """Per-snapshot fluctuation energy  E_k = 1/2 sum (x_k - mean)^2  (all comps)."""
    n = x.shape[0]
    E = np.empty(n, dtype=np.float64)
    for k in range(n):                       # loop keeps peak memory low
        d = x[k] - mean_field
        E[k] = 0.5 * float((d * d).sum())
    return E


def main():
    train_real, train_aug, comp_names, strategy = load_bundle(AUG_FILE)
    n_real, C, H, W = train_real.shape
    n_aug = train_aug.shape[0]
    comp = min(COMP, C - 1)
    n_frames = min(n_real, n_aug, N_FRAMES or max(n_real, n_aug))

    # shared robust color scale from the REAL data
    sample = train_real[:: max(1, n_real // 50), comp]
    vmin, vmax = np.percentile(sample, [1.0, 99.0])

    # energy traces (fluctuations about the REAL training mean field)
    mean_field = train_real.mean(axis=0)
    E_real = fluct_energy(train_real, mean_field)
    E_aug  = fluct_energy(train_aug,  mean_field)

    fig = plt.figure(figsize=(13, 7))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.08)
    axR, axA = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axE = fig.add_subplot(gs[1, :])

    imR = axR.imshow(train_real[0, comp], origin="lower", cmap="RdBu_r",
                     vmin=vmin, vmax=vmax, aspect="equal")
    imA = axA.imshow(train_aug[0, comp],  origin="lower", cmap="RdBu_r",
                     vmin=vmin, vmax=vmax, aspect="equal")
    axR.set_title("REAL (train)"); axA.set_title(f"SYNTHETIC ({strategy})")
    for ax in (axR, axA):
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(imA, ax=(axR, axA), fraction=0.025, pad=0.01)

    axE.plot(E_real, color="0.55", lw=1.0, label="real")
    axE.plot(E_aug,  color="crimson", lw=1.0, alpha=0.8, label="synthetic")
    axE.axhspan(E_real.min(), E_real.max(), color="0.85", alpha=0.5, zorder=0)
    cursor = axE.axvline(0, color="k", lw=1.2)
    axE.set_xlabel("snapshot index"); axE.set_ylabel("fluct. energy")
    axE.legend(loc="upper right", fontsize=8)
    title = fig.suptitle(f"{os.path.basename(AUG_FILE)}   comp={comp_names[comp]}"
                         f"   frame 0/{n_frames}")

    def update(k):
        imR.set_data(train_real[k, comp])
        imA.set_data(train_aug[k, comp])
        cursor.set_xdata([k, k])
        title.set_text(f"{os.path.basename(AUG_FILE)}   comp={comp_names[comp]}"
                       f"   frame {k}/{n_frames}")
        return imR, imA, cursor, title

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / FPS, blit=False)
    if SAVE_PATH:
        writer = ("pillow" if SAVE_PATH.endswith(".gif") else "ffmpeg")
        anim.save(SAVE_PATH, fps=FPS, writer=writer, dpi=100)
        print(f"saved → {SAVE_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

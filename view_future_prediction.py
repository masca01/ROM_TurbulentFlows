"""
Visualize a forward-prediction bundle from pod_augment_galerkin_ns_future.py.

Unlike view_augmented_movie.py (which pairs synthetic frames against random
training frames), here the two panels are time-aligned: the predicted future
snapshot and the REAL held-out snapshot at the SAME instant, so the animation is
a genuine out-of-sample check of the prediction.  A third panel shows their
difference, and the strip underneath tracks the per-step errors with a moving
cursor.

    top-left   : real held-out future           (val_real)
    top-mid    : predicted future               (train_aug)
    top-right  : difference (predicted - real)
    bottom     : per-step coefficient error (from the bundle, dynamics only) and
                 field error (computed here from the paired frames), with a
                 cursor at the current step.

Only the steps that have a real counterpart (the first n_val predicted frames)
are shown, since those are the ones that can be validated.

Set AUG_FILE, COMP; SAVE_PATH writes an .mp4/.gif instead of a live window.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Folder layout ──
_HERE    = os.path.dirname(os.path.abspath(__file__))
_AUG_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA", "AUGMENTED"))

# ══ CONFIG ══
AUG_FILE = os.path.join(_AUG_DIR, "Data2PlatesGap1Re50_aug_galerkin_ns_future.mat")
COMP     = 0         # component to show (0 = u, 1 = v)
FPS      = 4         # playback speed
SAVE_PATH = None     # e.g. ".../future.gif" or ".mp4"; None = live window
# ════════════


def load_bundle(path):
    S = np.load(path, allow_pickle=True) if path.endswith(".npz") \
        else sio.loadmat(path, simplify_cells=True)
    def arr(key):
        return np.asarray(S[key], dtype=np.float32)
    train_real = arr("train_real")
    train_aug  = arr("train_aug")           # predicted future
    val_real   = arr("val_real")            # real held-out future
    cn = S.get("comp_names") if hasattr(S, "get") else (
         S["comp_names"] if "comp_names" in S else None)
    if cn is None:
        comp_names = [f"c{i}" for i in range(val_real.shape[1])]
    elif isinstance(cn, str):
        comp_names = [cn]
    else:
        comp_names = [str(x) for x in np.atleast_1d(cn)]
    coef_err = np.asarray(S["val_coef_err"]).ravel() if "val_coef_err" in S else None
    return train_real, train_aug, val_real, comp_names, coef_err


def main():
    train_real, train_aug, val_real, comp_names, coef_err = load_bundle(AUG_FILE)
    comp = min(COMP, val_real.shape[1] - 1)

    # only the frames that have a real counterpart can be compared
    n = min(train_aug.shape[0], val_real.shape[0])
    pred = train_aug[:n, comp]
    real = val_real[:n, comp]

    # per-step field error, relative to the real fluctuation about the train mean
    mean_field = train_real.mean(axis=0)[comp]
    num = np.linalg.norm((pred - real).reshape(n, -1), axis=1)
    den = np.linalg.norm((real - mean_field[None]).reshape(n, -1), axis=1)
    field_err = num / (den + 1e-12)
    if coef_err is not None:
        coef_err = coef_err[:n]

    # shared color scale from the real frames; symmetric scale for the difference
    vmin, vmax = np.percentile(real, [1.0, 99.0])
    diff = pred - real
    dlim = np.percentile(np.abs(diff), 99.0)

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.1)
    axR = fig.add_subplot(gs[0, 0]); axP = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[0, 2]); axE = fig.add_subplot(gs[1, :])

    imR = axR.imshow(real[0], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    imP = axP.imshow(pred[0], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    imD = axD.imshow(diff[0], origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim, aspect="equal")
    axR.set_title("REAL future (held out)"); axP.set_title("PREDICTED future")
    axD.set_title("difference")
    for ax in (axR, axP, axD):
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(imP, ax=(axR, axP), fraction=0.025, pad=0.01)
    fig.colorbar(imD, ax=axD, fraction=0.046, pad=0.02)

    steps = np.arange(1, n + 1)
    axE.plot(steps, field_err, color="crimson", lw=1.5, marker="o", ms=3, label="field error")
    if coef_err is not None:
        axE.plot(steps, coef_err, color="0.4", lw=1.5, marker="s", ms=3, label="coef error")
    cursor = axE.axvline(1, color="k", lw=1.2)
    axE.set_xlabel("prediction step (into the future)")
    axE.set_ylabel("relative error")
    axE.set_xlim(0.5, n + 0.5); axE.set_ylim(0, None)
    axE.legend(loc="upper left", fontsize=9)

    title = fig.suptitle("", fontsize=12)
    def set_title(k):
        t = (f"{os.path.basename(AUG_FILE)}   comp={comp_names[comp]}   "
             f"step {k+1}/{n}   field err {field_err[k]:.2f}")
        if coef_err is not None:
            t += f"   coef err {coef_err[k]:.2f}"
        title.set_text(t)
    set_title(0)

    def update(k):
        imR.set_data(real[k]); imP.set_data(pred[k]); imD.set_data(diff[k])
        cursor.set_xdata([k + 1, k + 1])
        set_title(k)
        return imR, imP, imD, cursor, title

    anim = animation.FuncAnimation(fig, update, frames=n,
                                   interval=1000 / FPS, blit=False)
    if SAVE_PATH:
        writer = "pillow" if SAVE_PATH.endswith(".gif") else "ffmpeg"
        anim.save(SAVE_PATH, fps=FPS, writer=writer, dpi=100)
        print(f"saved → {SAVE_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

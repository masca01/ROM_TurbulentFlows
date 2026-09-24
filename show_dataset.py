"""
Quick look at the dataRe50Alpha0_2.mat 2-plates-gap dataset.

This new dataset has a different layout from Data2PlatesGap1Re*.mat:
    U, V : [Nt, nx, ny]  velocity components, already on the 2-D grid (time first)
    X, Y : [nx, ny]      grid coordinates
    Re, alpha, dt        scalars

It just reads ONE snapshot (lazily, so the 16 GB file is not pulled into memory)
and draws u, v and the vorticity so you can see what the flow looks like.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

FILE = ("/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/"
        "2PlatesGap/dataRe50Alpha0_2.mat")
SNAP = 2500            # which snapshot to show (0 .. Nt-1)
XLIM = (-3, 25)        # zoom window in x around the plates + near wake
YLIM = (-6, 6)         # zoom window in y

f = h5py.File(FILE, "r")
Re    = float(np.ravel(f["Re"][()])[0])
alpha = float(np.ravel(f["alpha"][()])[0])
dt    = float(np.ravel(f["dt"][()])[0])
Nt, nx, ny = f["U"].shape
X = np.asarray(f["X"][()])          # [nx, ny]
Y = np.asarray(f["Y"][()])
u = np.asarray(f["U"][SNAP])        # [nx, ny]  (single snapshot, lazy read)
v = np.asarray(f["V"][SNAP])
f.close()

dx = float(np.diff(np.unique(X.ravel()))[0])
dy = float(np.diff(np.unique(Y.ravel()))[0])

print(f"file        : {os.path.basename(FILE)}")
print(f"Re          : {Re:g}")
print(f"alpha       : {alpha:g}   (angle of attack, rad)")
print(f"dt          : {dt:g}   (time between snapshots)")
print(f"snapshots   : {Nt}")
print(f"grid        : nx={nx}  ny={ny}   (dx={dx:.3f}, dy={dy:.3f})")
print(f"domain      : x in [{X.min():.2f}, {X.max():.2f}]   "
      f"y in [{Y.min():.2f}, {Y.max():.2f}]")
print(f"showing snap #{SNAP}  (t = {SNAP*dt:g} convective times)")

# vorticity  w = dv/dx - du/dy   (x is axis 0, y is axis 1 in these arrays)
vort = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)

fig, axs = plt.subplots(3, 1, figsize=(8.5, 8), constrained_layout=True)
panels = [
    (u,    "streamwise velocity  u", "RdBu_r", None),
    (v,    "transverse velocity  v", "RdBu_r", None),
    (vort, "vorticity  ω = ∂v/∂x − ∂u/∂y", "RdBu_r", 3.0),
]
for ax, (field, title, cmap, cap) in zip(axs, panels):
    vmax = cap if cap is not None else np.percentile(np.abs(field), 99)
    pc = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=-vmax, vmax=vmax,
                       shading="auto")
    # the two flat plates: unit length, unit gap, at x = 0
    ax.plot([0, 0], [-1.5, -0.5], "k", lw=2.5)
    ax.plot([0, 0], [ 0.5,  1.5], "k", lw=2.5)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, shrink=0.9)
axs[-1].set_xlabel("x")
fig.suptitle(f"dataRe50Alpha0_2  |  Re={Re:g}, alpha={alpha:g}, dt={dt:g}  "
             f"|  snapshot {SNAP}/{Nt}", fontweight="bold")

out = os.path.join(os.path.dirname(FILE), "dataRe50Alpha0_2_preview.png")
fig.savefig(out, dpi=130)
print(f"\nsaved figure -> {out}")
plt.show()

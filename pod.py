"""
POD for turbulent flow ROM  (universal)
Supports: ModelFLOW (Tensor), BOX (U 3-comp), CHANNEL (UW or U), ChannelEdge

Requirements: pip install numpy scipy matplotlib
"""

import os
import numpy as np
import scipy.io as sio
import scipy.linalg as la
import matplotlib.pyplot as plt

# ── Folder layout (relative to this script) ──────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "bestModels"))
os.makedirs(_MODELS_DIR, exist_ok=True)

# ══════════════════════════ CONFIG ══════════════════════════
# Set the full path to your .mat file, or leave "" to open a file dialog.
# Example paths (DATA is at CODING/DATA/):
#   DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/CHANNEL_Turbulence/channelEdge_256x256_Nt4000_UVW.mat"
#   DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/BOX_Turbulence/isotropic1024coarse_xz_y3.1416_256x256_Nt5024_UVW.mat"
#   DATA_FILE = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/ModelFLOW/TensorRe160.mat"
DATA_FILE   = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/CHANNEL_Turbulence/channelEdge_256x256_Nt4000_UVW.mat"
COMP_IDX    = None     # None = auto; [0,2] = pick u,w from 3-comp field

RANKS       = [1, 5, 20, 50]   # POD ranks to evaluate
EVAL_STRIDE = 1                           # stride for Ek computation (increase to speed up)
MAX_MODES   = 100                         # max POD modes saved to disk (truncated Phi)
# ═══════════════════════════════════════════════════════════


def pick_file(start_dir=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(
            title="Select .mat data file",
            initialdir=start_dir or os.path.expanduser("~"),
            filetypes=[("MAT files", "*.mat"), ("All", "*")])
        root.destroy()
        return p
    except Exception:
        return input("Path to .mat file: ").strip()


def _read_field(S, key, is_hdf5):
    """Read a numerical array from scipy dict or h5py file.
    h5py stores MATLAB arrays with all dimensions reversed, so we transpose back."""
    arr = np.array(S[key], dtype=np.float32)
    if is_hdf5:
        arr = arr.T
    return arr


def load_data(path, comp_idx=None):
    """Load .mat file (v5 or v7.3/HDF5) → data [Nt, C, H, W] float32, comp_names."""
    import h5py

    fh = None
    try:
        S = sio.loadmat(path, simplify_cells=True)
        is_hdf5 = False
    except NotImplementedError:
        fh = h5py.File(path, "r")
        S = fh
        is_hdf5 = True

    try:
        if "Tensor" in S:
            T = _read_field(S, "Tensor", is_hdf5)        # [C, N1, N2, Nt]
            C, N1, N2, Nt = T.shape
            data = T.transpose(3, 0, 1, 2)               # [Nt, C, N1, N2]
            names = {2: ["u", "v"], 3: ["u", "v", "w"]}.get(C, [f"c{i}" for i in range(C)])

        elif "U" in S:
            V = _read_field(S, "U", is_hdf5)             # [Nt, Nz, Nx, C_all]
            Nt, Nz, Nx, C_all = V.shape
            if comp_idx is None:
                comp_idx = [0, 2] if C_all == 3 else list(range(C_all))
            V = V[:, :, :, comp_idx]
            data = V.transpose(0, 3, 1, 2)               # [Nt, C, Nz, Nx]
            all_names = ["u", "v", "w"]
            names = [all_names[i] for i in comp_idx]

        elif "UW" in S:
            V = _read_field(S, "UW", is_hdf5)            # [Nt, Nz, Nx, 2]
            data = V.transpose(0, 3, 1, 2)               # [Nt, 2, Nz, Nx]
            names = ["u", "w"]

        else:
            visible = [k for k in S.keys() if not k.startswith("#")]
            raise ValueError(f"Unknown format. Fields: {visible}")

    finally:
        if fh is not None:
            fh.close()

    print(f"Loaded  {os.path.basename(path)}  →  {data.shape}  comps={names}")
    return data, names


def build_snapshot_matrix(data):
    """Stack all components into snapshot matrix [Ndof, Nt].
    data: [Nt, C, H, W]  →  U: [C*H*W, Nt]
    """
    Nt, C, H, W = data.shape
    U = data.reshape(Nt, -1).T.astype(np.float64)   # [C*H*W, Nt]
    Umean = U.mean(axis=1, keepdims=True)
    Uf = U - Umean
    return Uf, Umean


def compute_ek_ranks(Phi, sing, Vsvd, Uf, data_shape, ranks, stride=1):
    """Ek (%) per rank per component and all-component combined.

    Returns ek_c: [n_ranks, C]  (per component)
            ek_all: [n_ranks]   (all components)
    """
    Nt, C, H, W = data_shape
    Ndof = C * H * W
    dof_c = H * W   # DOFs per component

    ranks = [r for r in ranks if r <= Phi.shape[1]]
    n_ranks = len(ranks)
    ek_c   = np.zeros((n_ranks, C))
    ek_all = np.zeros(n_ranks)

    eval_t = np.arange(0, Nt, stride)

    for ri, r in enumerate(ranks):
        SSE = np.zeros(C); EN = np.zeros(C)
        for k in eval_t:
            coeff  = sing[:r] * Vsvd[k, :r]       # [r]
            uf_hat = Phi[:, :r] @ coeff            # [Ndof]
            uf_t   = Uf[:, k]

            for c in range(C):
                sl = slice(c * dof_c, (c + 1) * dof_c)
                err = uf_t[sl] - uf_hat[sl]
                SSE[c] += (err ** 2).sum()
                EN[c]  += (uf_t[sl] ** 2).sum()

        EN_safe = np.where(EN < 1e-12, 1e-12, EN)
        ek_c[ri]   = (1 - SSE / EN_safe) * 100
        ek_all[ri] = (1 - SSE.sum() / max(EN.sum(), 1e-12)) * 100

        print(f"  rank={r:4d} | Ek_all={ek_all[ri]:.2f}%  per-comp={ek_c[ri]}")

    return ek_c, ek_all


def reconstruct_snapshot(Phi, sing, Vsvd, Umean, r, t, data_shape):
    """Reconstruct physical fields at snapshot t using rank r."""
    Nt, C, H, W = data_shape
    coeff  = sing[:r] * Vsvd[t, :r]
    uf_hat = Phi[:, :r] @ coeff
    u_full = uf_hat + Umean[:, 0]
    return u_full.reshape(C, H, W).astype(np.float32)   # [C, H, W]


def plot_energy_spectrum(sing, title="POD energy spectrum"):
    energy = sing**2 / (sing**2).sum()
    plt.figure(figsize=(8, 4))
    plt.semilogy(np.arange(1, len(energy)+1), energy, "o-", ms=3, lw=1)
    plt.xlabel("Mode"); plt.ylabel("Energy fraction (log)")
    plt.title(title); plt.grid(True); plt.tight_layout(); plt.show()


def plot_reconstructions(true_field, recon_fields, ranks, comp_names, t_idx):
    """Plot true field and reconstructions at each rank."""
    C = true_field.shape[0]
    n_ranks = len(recon_fields)

    for c in range(C):
        fig, axes = plt.subplots(1, n_ranks + 1,
                                 figsize=(4 * (n_ranks + 1), 4), squeeze=False)
        mx = max(abs(float(true_field[c].min())), abs(float(true_field[c].max())), 1e-8)
        vmin, vmax = -mx, mx

        im0 = axes[0, 0].imshow(true_field[c], vmin=vmin, vmax=vmax,
                                cmap="RdBu_r", origin="lower")
        axes[0, 0].set_title(f"{comp_names[c]} TRUE (t={t_idx})")
        axes[0, 0].axis("off")
        plt.colorbar(im0, ax=axes[0, 0])

        for ri, (r, rec) in enumerate(zip(ranks, recon_fields)):
            im = axes[0, ri+1].imshow(rec[c], vmin=vmin, vmax=vmax,
                                      cmap="RdBu_r", origin="lower")
            axes[0, ri+1].set_title(f"rank {r}")
            axes[0, ri+1].axis("off")
            plt.colorbar(im, ax=axes[0, ri+1])

        plt.suptitle(f"POD reconstruction  —  {comp_names[c]}")
        plt.tight_layout(); plt.show()


def plot_ek(ranks, ek_c, ek_all, comp_names):
    plt.figure(figsize=(8, 5))
    for c in range(ek_c.shape[1]):
        plt.plot(ranks, ek_c[:, c], "o-", label=f"Ek_{comp_names[c]}")
    plt.plot(ranks, ek_all, "s--", lw=2, label="Ek_all", color="black")
    plt.xlabel("POD rank"); plt.ylabel("Ek (%)")
    plt.title("POD energy reconstruction vs rank")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# ─────────────────────────── Main ───────────────────────────

def main():
    global DATA_FILE

    if not DATA_FILE:
        DATA_FILE = pick_file(_DATA_DIR)
    if not DATA_FILE:
        raise RuntimeError("No file selected.")

    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape

    ranks_use = sorted(set(r for r in RANKS if 1 <= r <= Nt))

    print(f"\nBuilding snapshot matrix [{C*H*W} × {Nt}] ...")
    Uf, Umean = build_snapshot_matrix(data)

    print("Computing economy SVD ...")
    Phi, Ssvd, Vt = la.svd(Uf, full_matrices=False)
    sing = Ssvd                    # [k] singular values
    Vsvd = Vt.T                    # [Nt, k] right singular vectors

    print(f"SVD done: {Phi.shape[1]} modes available\n")
    plot_energy_spectrum(sing, title=f"POD energy  —  {os.path.basename(DATA_FILE)}")

    # Snapshot to display
    t_plot = Nt // 2

    # Reconstruct at each rank
    print("Computing reconstructions ...")
    recon_fields = [reconstruct_snapshot(Phi, sing, Vsvd, Umean, r, t_plot,
                                         (Nt, C, H, W))
                    for r in ranks_use]
    true_field = data[t_plot]   # [C, H, W]
    plot_reconstructions(true_field, recon_fields, ranks_use, comp_names, t_plot)

    # Ek metrics
    print(f"\nComputing Ek vs rank  (eval every {EVAL_STRIDE} snapshots) ...")
    ek_c, ek_all = compute_ek_ranks(Phi, sing, Vsvd, Uf,
                                     (Nt, C, H, W), ranks_use, EVAL_STRIDE)
    plot_ek(ranks_use, ek_c, ek_all, comp_names)

    # Save results
    base      = os.path.splitext(os.path.basename(DATA_FILE))[0]
    save_path = os.path.join(_MODELS_DIR, f"pod_{base}.npz")

    n_save = min(MAX_MODES, Phi.shape[1])
    np.savez(save_path,
             Phi=Phi[:, :n_save].astype(np.float32),
             sing=sing[:n_save].astype(np.float32),
             Vsvd=Vsvd[:, :n_save].astype(np.float32),
             Umean=Umean.astype(np.float32),
             ranks=np.array(ranks_use),
             ek_c=ek_c.astype(np.float32),
             ek_all=ek_all.astype(np.float32),
             comp_names=np.array(comp_names),
             data_file=np.array(DATA_FILE),
             Nt=Nt, C=C, H=H, W=W,
             t_plot=t_plot,
             true_field=true_field)

    print(f"\nSaved → {save_path}  (Phi truncated to {n_save} modes)")

    for ri, r in enumerate(ranks_use):
        print(f"  rank={r:4d} | Ek_all={ek_all[ri]:.2f}%")


if __name__ == "__main__":
    main()

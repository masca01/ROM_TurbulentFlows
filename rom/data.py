"""
Datasets and loaders.

DATASETS is the registry every study uses: name -> (file relative to paths.DATA,
snapshot cap, beta-VAE latent dimension of the convergence study).

load_data() reads any of the .mat layouts in the project and returns
data[Nt, C, H, W] float32 (from pod_augment_galerkin_ns.py / beta_vae.py, which
had identical copies):
    Alpha0 Re/alpha sweep   U, V [Nt, nx, ny], X, Y [nx, ny], Re, dt   (v7.3 / HDF5)
    2-plates Re100 / Re50   DataU, DataV [Nt, nx*ny], DataX/DataY or X/Y
    older layouts           Tensor, U (channel), UW

load() is the fast reader of the region map and every later study (from
run_region_map.py): it reads only the first `cap` snapshots of the big Alpha0
files straight from disk.
"""
import os
import numpy as np
import scipy.io as sio

from . import paths

#          name:  (file, snapshot cap, latent of the convergence study)
DATASETS = {
    "Re50":  ("Alpha0/dataRe50Alpha0_2.mat",          1500, 5),
    "Re60":  ("Alpha0/dataRe60Alpha0_2.mat",          1500, 7),
    "Re70":  ("Alpha0/dataRe70Alpha0_2.mat",          1500, 9),
    "Re80":  ("Alpha0/dataRe80Alpha0_2.mat",          1500, 11),
    "Re100": ("2PlatesGap/Data2PlatesGap1Re100.mat",  None, 15),
}
LATENT = {name: lat for name, (_, _, lat) in DATASETS.items()}


# ------------------------- Loading -------------------------

def _read_field(S, key, is_hdf5):
    """Read one array out of a .mat file.  h5py (v7.3 files) stores MATLAB
    arrays with the dimensions reversed, so in that case I transpose back."""
    arr = np.array(S[key], dtype=np.float32)
    if is_hdf5:
        arr = arr.T
    return arr


def load_data(path, comp_idx=None, t_stride=1, t_max=None):
    """Load a .mat dataset and return it as data[Nt, C, H, W] (float32) plus the
    names of the components.  Handles both old (v5) and new (v7.3/HDF5) .mat
    files, and the couple of different variable layouts I have across datasets.

    t_stride / t_max optionally subsample in TIME (keep every t_stride-th
    snapshot, then at most t_max of them).  This matters for the big Re/alpha
    sweep files (dataRe50Alpha0_2.mat: 5000 snapshots, ~16 GB) — the read is
    done straight from disk with the stride so the full record is never held in
    memory."""
    import h5py

    # scipy reads v5 files; for v7.3 it raises and I fall back to h5py
    fh = None
    try:
        S = sio.loadmat(path, simplify_cells=True)
        is_hdf5 = False
    except NotImplementedError:
        fh = h5py.File(path, "r")
        S = fh
        is_hdf5 = True

    try:
        subsampled = False
        if "Tensor" in S:                                # cylinder-wake format
            T = _read_field(S, "Tensor", is_hdf5)        # [C, N1, N2, Nt]
            C, N1, N2, Nt = T.shape
            data = T.transpose(3, 0, 1, 2)               # -> [Nt, C, N1, N2]
            names = {2: ["u", "v"], 3: ["u", "v", "w"]}.get(C, [f"c{i}" for i in range(C)])

        elif "U" in S and "V" in S:                      # 2-plates-gap Re/alpha
            # sweep format (dataRe50Alpha0_2.mat): U, V are [Nt, nx, ny] on the
            # grid already (time is the first axis), coords X, Y are [nx, ny].
            # These files are huge, so read straight from disk with the temporal
            # stride/cap rather than pulling the whole 5000-snapshot record in.
            sl = slice(None, None, t_stride)
            u = np.asarray(S["U"][sl], dtype=np.float32)  # [nt, nx, ny]
            v = np.asarray(S["V"][sl], dtype=np.float32)
            if t_max is not None:
                u = u[:t_max]; v = v[:t_max]
            u = np.transpose(u, (0, 2, 1))               # -> [nt, H=ny, W=nx]
            v = np.transpose(v, (0, 2, 1))
            data = np.stack([u, v], axis=1)              # [Nt, 2, H, W]
            names = ["u", "v"]
            subsampled = True                            # stride already applied

        elif "U" in S:                                   # channel-type format
            V = _read_field(S, "U", is_hdf5)             # [Nt, Nz, Nx, C_all]
            Nt, Nz, Nx, C_all = V.shape
            if comp_idx is None:
                comp_idx = [0, 2] if C_all == 3 else list(range(C_all))
            V = V[:, :, :, comp_idx]
            data = V.transpose(0, 3, 1, 2)               # -> [Nt, C, Nz, Nx]
            all_names = ["u", "v", "w"]
            names = [all_names[i] for i in comp_idx]

        elif "UW" in S:
            V = _read_field(S, "UW", is_hdf5)            # [Nt, Nz, Nx, 2]
            data = V.transpose(0, 3, 1, 2)               # -> [Nt, 2, Nz, Nx]
            names = ["u", "w"]

        elif "DataU" in S and "DataV" in S:              # 2-plates-gap DNS
            # Each row of DataU/DataV is one snapshot with the 2D field
            # flattened (Fortran order, x fastest).  The grid resolution is not
            # the same in every dataset (Re=100 is 1199x349, Re=50 is 599x349),
            # and the coordinate arrays are called DataX/DataY in one file and
            # X/Y in the other, so I infer nx, ny from whichever is present
            # instead of hardcoding them.
            u = np.asarray(S["DataU"], dtype=np.float32)
            v = np.asarray(S["DataV"], dtype=np.float32)
            Nt, Nspace = u.shape
            xkey = "DataX" if "DataX" in S else "X"
            NX = int(np.unique(np.asarray(S[xkey]).ravel()).size)   # points in x
            NY = Nspace // NX                                        # points in y
            if NX * NY != Nspace:
                raise ValueError(f"2PlatesGap: inferred nx*ny {NX*NY} != "
                                 f"Nspace {Nspace}")
            u = u.reshape(Nt, NY, NX)                    # [Nt, H=y, W=x]
            v = v.reshape(Nt, NY, NX)
            data = np.stack([u, v], axis=1)              # [Nt, 2, H, W]
            names = ["u", "v"]

        else:
            visible = [k for k in S.keys() if not k.startswith("#")]
            raise ValueError(f"Unknown format. Fields: {visible}")

    finally:
        if fh is not None:
            fh.close()

    if not subsampled and (t_stride != 1 or t_max is not None):
        data = data[::t_stride]
        if t_max is not None:
            data = data[:t_max]

    print(f"Loaded  {os.path.basename(path)}  →  {data.shape}  comps={names}")
    return data, names


def load(name):
    """[Nt, 2, H, W] float32, Re, dt, path. Reads only the first `cap` snapshots of the big
    Alpha0 files straight from disk (the generic loader reads all 5000 first)."""
    fn, cap, _ = DATASETS[name]
    path = os.path.join(paths.DATA, fn)
    import h5py
    with h5py.File(path, "r") as f:
        re_ = float(np.array(f["Re"]).ravel()[0]) if "Re" in f else float(name.replace("Re", ""))
        dt = float(np.array(f["dt"]).ravel()[0]) if "dt" in f else 1.0   # old 2-plates files: 1 convective time
        if "U" in f and "V" in f:
            nt, nx, ny = f["U"].shape
            nt = nt if cap is None else min(nt, cap)
            data = np.empty((nt, 2, ny, nx), dtype=np.float32)     # filled one component at a time
            for c, comp in enumerate(("U", "V")):
                for t0 in range(0, nt, 250):                      # chunks keep the float64 read small
                    data[t0:t0 + 250, c] = np.transpose(f[comp][t0:min(t0 + 250, nt)], (0, 2, 1))
    if "data" not in locals():
        data, _ = load_data(path, None, 1, cap)
    print(f"[load]  {name}: {data.shape}, Re {re_:g}, dt {dt:g}", flush=True)
    return data, re_, dt, path

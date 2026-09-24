"""
Tiny FAKE datasets in the exact layouts of the real ones, so the code can be exercised
without the 138 GB of .mat files.

    Alpha0/dataRe{50,60,70,80}Alpha0_2.mat    MATLAB v7.3 (HDF5): U, V [Nt, nx, ny] (h5py order),
                                              X, Y [nx, ny], Re, alpha, dt
    2PlatesGap/Data2PlatesGap1Re100.mat       MATLAB v7.3 (HDF5): DataU, DataV [Nt, nx*ny]
                                              (x fastest), DataX, DataY; no Re / dt (as the old file)
    CHANNEL/channel_fake_UVW.mat              MATLAB v5: U [Nt, Nz, Nx, 3], the layout the study-1
                                              scripts (pod.py, analysis.py, inspect_vae.py) read

The flow is a mean wake plus a few travelling waves and a little noise, so the POD has a
dominant low-rank part and a full-rank tail, like the real data.

    python3 tests/fake_data.py <DATA dir> [--nt-alpha0 N] [--nt-re100 N]
"""
import os, sys
import numpy as np
import h5py


def _mat73(path, arrays):
    """Write `arrays` to an HDF5 file with the 512-byte MATLAB 7.3 header, so that
    scipy.io.loadmat refuses it with NotImplementedError exactly as it does a real v7.3 file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w", userblock_size=512) as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)
    text = b"MATLAB 7.3 MAT-file, Platform: fake, Created on: fake HDF5 schema 1.00 ."
    header = text.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(path, "r+b") as f:
        f.write(header.ljust(512, b"\x00"))


def _flow(nt, nx, ny, dx, dy, dt, re, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(nx) * dx - 2.0
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")                    # [nx, ny]
    t = np.arange(nt) * dt
    omega = 0.8 * (1 + re / 400)
    env = np.exp(-(Y / (0.3 * ny * dy)) ** 2)
    u = np.empty((nt, nx, ny)); v = np.empty((nt, nx, ny))
    base_u = 1.0 - 0.6 * env
    for n in range(nt):
        ph = 1.3 * X - omega * t[n]
        mod = 1.0 + 0.3 * np.sin(0.23 * omega * t[n])          # slow amplitude modulation
        u[n] = base_u.copy(); v[n] = 0.0
        for m in range(1, 7):                                  # harmonics, decaying amplitude
            a = 0.2 * 0.55 ** (m - 1) * (mod if m % 2 else 1.0)
            u[n] += a * env * np.cos(m * ph + 0.4 * m) * (1 + 0.2 * m * Y / Y.max())
            v[n] += 1.2 * a * env * np.sin(m * ph + 0.7 * m)
    u += 3e-3 * rng.standard_normal(u.shape); v += 3e-3 * rng.standard_normal(v.shape)
    return u, v, X, Y


def make(data_dir, nt_alpha0=1500, nt_re100=1000, nx=24, ny=16, nx100=20, ny100=12, nt_channel=120):
    """Write every fake dataset under data_dir; returns data_dir."""
    for i, re in enumerate((50, 60, 70, 80)):
        dt = 0.2
        u, v, X, Y = _flow(nt_alpha0, nx, ny, 0.25, 0.25, dt, re, seed=100 + i)
        _mat73(os.path.join(data_dir, "Alpha0", f"dataRe{re}Alpha0_2.mat"),
               {"U": u, "V": v, "X": X, "Y": Y, "Re": np.array([[float(re)]]),
                "alpha": np.array([[0.0]]), "dt": np.array([[dt]])})
    u, v, X, Y = _flow(nt_re100, nx100, ny100, 0.08 * 4, 0.08 * 4, 1.0, 100, seed=7)
    # DataU[t, j*nx + i] = u(x_i, y_j): x fastest
    DU = np.transpose(u, (0, 2, 1)).reshape(nt_re100, -1)
    DV = np.transpose(v, (0, 2, 1)).reshape(nt_re100, -1)
    _mat73(os.path.join(data_dir, "2PlatesGap", "Data2PlatesGap1Re100.mat"),
           {"DataU": DU, "DataV": DV, "DataX": X.T.reshape(-1), "DataY": Y.T.reshape(-1)})
    u, v, _, _ = _flow(nt_channel, 16, 16, 0.25, 0.25, 0.5, 180, seed=11)
    w = 0.5 * np.roll(u, 3, axis=1) - 0.5
    import scipy.io as sio
    os.makedirs(os.path.join(data_dir, "CHANNEL"), exist_ok=True)
    sio.savemat(os.path.join(data_dir, "CHANNEL", "channel_fake_UVW.mat"),
                {"U": np.stack([u, v, w], axis=-1).astype(np.float32)})     # [Nt, Nz, Nx, 3]
    return data_dir


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--nt-alpha0" in a: kw["nt_alpha0"] = int(a[a.index("--nt-alpha0") + 1])
    if "--nt-re100" in a: kw["nt_re100"] = int(a[a.index("--nt-re100") + 1])
    print(make(a[0], **kw))

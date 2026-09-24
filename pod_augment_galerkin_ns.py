"""
POD data augmentation using a POD-Galerkin model built from the Navier-Stokes equations.

I reduce each snapshot to a few POD coefficients a_i(t), find a low-order model for how those coefficients evolve in time, and then integrate that model to create new, physically plausible snapshots.
Here I do NOT fit the model to the data; instead I derive it analytically by projecting the incompressible Navier-Stokes operators onto the POD modes.

The two operators I need are the convection (quadratic) and diffusion (linear) terms:

    convection:  q_ijk = - INT dA [ (u_j du_k/dx + v_j du_k/dy) u_i + (u_j dv_k/dx + v_j dv_k/dy) v_i ]
    diffusion:   l_ij  = (1/Re) INT dA [ lap(u_j) u_i + lap(v_j) v_i ]

I include the mean flow as an extra "mode 0" whose coefficient is fixed to a_0 = 1.  Then, the reduced model is simply:

    da_i/dt = sum_{j=0}^r l_ij a_j + sum_{j,k=0}^r q_ijk a_j a_k ,   a_0 = 1,

and the usual constant/linear/quadratic (c, L, Q) terms are all hidden inside l and q.  Time is measured in convective units, which is the spacing of the snapshots for this dataset, so here dt is an actual physical time step.

A few things this version needs:
  * the Reynolds number Re and the grid spacing dx, dy;
  * POD modes that are orthonormal in the PHYSICAL inner product, INT phi_i phi_j dA = delta_ij.  compute_pod gives me modes that are orthonormal as plain vectors, so on this uniform grid I only have to rescale them by 1/sqrt(dx*dy)
  (and the coefficients by sqrt(dx*dy)) to get the physical normalization the projection integrals assume;
  * spatial derivatives of the modes.  I drop the pressure term, and the derivatives are only as accurate as the uniform-grid interpolation of the DNS data near the plates.

Everything after the model is built is shared with pod_augment_galerkin.py.  So the bundle produced here differs from the "galerkin" one ONLY in how the model coefficients were obtained: projected from the equations here, fitted from
the data there.  Comparing the two (plus strategy C) is the whole point of this experiment.

Output: ../DATA/AUGMENTED/<base>_aug_galerkin_ns.{mat|npz}, a bundle with train_real / train_aug / val_real that plugs straight into train_augmented_vae.py.
"""

import os
import numpy as np
import scipy.io as sio

# ------------ Folder layout ------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "DATA"))
_AUG_DIR  = os.path.join(_DATA_DIR, "AUGMENTED")
os.makedirs(_AUG_DIR, exist_ok=True)

# ============ CONFIG ============
DATA_FILE   = "/Users/vmascarilla/Desktop/MMAE/597 - Special Topics/CODING/DATA/2PlatesGap/Data2PlatesGap1Re50.mat"
COMP_IDX    = None     # None = auto-detect the velocity components in the file
RNG_SEED    = 7        # same seed as the other generators, so the train/val split (and therefore val_real) is identical everywhere
VAL_FRAC    = 0.10     # fraction of the real snapshots kept aside for validation
N_AUG       = None     # how many synthetic snapshots to make (None = as many as there are real training snapshots)

# physical parameters of this dataset (needed for the NS projection)
RE          = 50       # Reynolds number of the simulation
DT          = 1        # snapshot spacing, in convective time units
DX_FALLBACK = 0.08     # grid spacing

# reduced-order model
N_ROM       = 20       # number of POD modes kept in the dynamical model.

# synthetic-trajectory generation
HORIZON     = 181      # snapshot intervals integrated per trajectory
SUBSTEPS    = 181      # RK4 sub-steps taken inside each snapshot interval
IC_NOISE    = 0.0     # size of the initial perturbation, as a fraction of each mode's std (this is the strategy-C idea)
ENERGY_TOL  = 0.05     # how much the total energy of a synthetic state is allowed to slip outside the real [min, max] band before I reject it (tighter = synthetic energy stays closer to the real)
MAX_TRAJ    = 5     # hard cap on how many trajectories I launch
MIN_KEEP    = 0.20     # if fewer than this fraction of steps survive screening, the model is drifting too much -> fewer modes

STRATEGY    = "galerkin_ns_5traj"
# ================================


# ------------------------- Loading and POD -------------------------

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


def compute_pod(train, rank=None):
    """POD by the method of snapshots.  When there are far more grid points than
    snapshots (the full 2-plates case), it is much cheaper to eigen-decompose
    the small [Ntr x Ntr] Gram matrix than to SVD the huge data matrix.

    train[Ntr, C, H, W] -> (x_mean, Xc, Wp, S, A, shape), where
      x_mean : temporal mean field (flattened)
      Xc     : mean-subtracted snapshots (flattened), kept so I can rebuild the
               spatial modes as U = Xc.T @ Wp without ever storing U
      Wp, S  : the pieces needed to go between coefficients and snapshots
      A      : the POD coefficients of the training snapshots
    """
    Ntr, C, H, W = train.shape
    X  = train.reshape(Ntr, C * H * W)
    x_mean = X.mean(axis=0)
    Xc = (X - x_mean[None, :]).astype(np.float32)
    G  = (Xc @ Xc.T).astype(np.float64)                  # Ntr x Ntr Gram matrix
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]                           # largest energy first
    w = np.clip(w[order], 0.0, None); V = V[:, order]
    S = np.sqrt(w)
    keep = S > (S[0] * 1e-10 if S.size else 0.0)          # drop numerical zeros
    S, V = S[keep], V[:, keep]
    if rank is not None:
        S, V = S[:rank], V[:, :rank]
    A  = (V * S[None, :])                                 # coefficients
    Wp = (V / S[None, :]).astype(np.float32)              # used to reconstruct
    return x_mean.astype(np.float32), Xc, Wp, S, A, (C, H, W)


def reconstruct(x_mean, Xc, Wp, A_new, shape):
    """Turn a set of POD coefficients back into full snapshots.  Uses
    X_new = x_mean + U @ A_new.T with U = Xc.T @ Wp, so the (large) modes U are
    never actually formed in memory."""
    C, H, W = shape
    M = Wp @ A_new.T.astype(np.float32)
    X_new = Xc.T @ M
    X_new += x_mean[:, None]
    return X_new.T.reshape(-1, C, H, W).astype(np.float32)


def save_bundle(out_base, payload):
    """Save the augmentation bundle.  A v5 .mat file cannot hold a single array
    larger than ~2 GB, so for the big datasets I switch to .npz (which
    train_augmented_vae.py reads just the same)."""
    MAT5_LIMIT = 2_000_000_000
    too_big = any(isinstance(v, np.ndarray) and v.nbytes >= MAT5_LIMIT
                  for v in payload.values())
    if too_big:
        path = out_base + ".npz"
        np.savez(path, **payload)
    else:
        path = out_base + ".mat"
        sio.savemat(path, payload, do_compression=True)
    return path


def grid_spacing(path):
    """Get the grid spacing dx, dy from the coordinate arrays in the file.
    The x/y grids are uniform, so I take the gap between the first two distinct
    coordinate values along each axis.  The arrays are named DataX/DataY in some
    files and X/Y in others; if neither is there I fall back to the Readme value."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            xkey = "DataX" if "DataX" in f else "X"
            ykey = "DataY" if "DataY" in f else "Y"
            X = np.unique(np.asarray(f[xkey]).ravel())
            Y = np.unique(np.asarray(f[ykey]).ravel())
        dx = float(X[1] - X[0])
        dy = float(Y[1] - Y[0])
        if dx <= 0 or dy <= 0:
            raise ValueError(f"non-positive spacing dx={dx}, dy={dy}")
        return dx, dy
    except Exception as err:
        print(f"[{STRATEGY}]  could not read grid coordinates ({err}) "
              f"-> using dx = dy = {DX_FALLBACK}")
        return DX_FALLBACK, DX_FALLBACK


# ------------------------- Trajectory generation -------------------------

def integrate_trajectories(step, s, B_real, n_aug, horizon,
                           ic_noise, energy_tol, max_traj, rng):
    """Create synthetic coefficient vectors by integrating many short
    trajectories and keeping only the states that stay physically reasonable.

    step            : advances a state by one snapshot interval
    s               : per-mode scale used to standardize the coefficients
    B_real[Ntr, r]  : standardized real coefficients; used both as the pool of
                      initial conditions and to define the allowed energy band
    Returns the accepted states (standardized) and a dict of statistics.
    """
    Ntr, r = B_real.shape

    # Allowed energy band, measured in the physical units of the kept modes.
    # A synthetic state whose energy leaves this band is considered drift.
    E_obs = 0.5 * ((B_real * s[None, :]) ** 2).sum(axis=1)
    lo = (1.0 - energy_tol) * E_obs.min()
    hi = (1.0 + energy_tol) * E_obs.max()
    b_cap = 10.0 * np.abs(B_real).max()                  # obvious blow-up cutoff

    out = np.empty((n_aug, r), dtype=np.float64)
    filled = n_traj = n_reject = 0
    while filled < n_aug and n_traj < max_traj:
        n_traj += 1
        # start from a real snapshot plus a small random kick (strategy C)
        b = B_real[rng.integers(0, Ntr)] + ic_noise * rng.normal(size=r)
        for _ in range(horizon):
            with np.errstate(over="ignore", invalid="ignore"):
                b = step(b)                              # advance one interval
            if not np.all(np.isfinite(b)) or np.abs(b).max() > b_cap:
                n_reject += horizon                      # blew up: drop the rest
                break
            E = 0.5 * ((b * s) ** 2).sum()
            if E < lo or E > hi:                         # left the energy band
                n_reject += 1
                break
            out[filled] = b
            filled += 1
            if filled == n_aug:
                break

    if filled < n_aug:
        raise RuntimeError(
            f"only {filled}/{n_aug} synthetic snapshots survived screening after "
            f"{n_traj} trajectories — the unclosed model drifts too fast")
    stats = {"n_traj": n_traj, "n_reject": n_reject,
             "keep_frac": filled / max(filled + n_reject, 1),
             "E_band": (lo, hi)}
    return out, stats


# ------------------------- The NS Galerkin projection -------------------------

def galerkin_operators_ns(Pu, Pv, dx, dy, re):
    """Build the Galerkin operators l and q by projecting the Navier-Stokes
    convection and diffusion terms onto the modes.  This is a vectorized version
    of Dawson's nseGalerkinCoeffsDemo.m (same integrals, just done with matrix
    products instead of the triple loop).

    Pu, Pv[r+1, H, W] : the mean flow in row 0, then the r modes.
    Returns l[r, r+1] and q[r, r+1, r+1] such that
        da_i/dt = l[i,:] @ c + c @ q[i] @ c,   with c = [1, a_1..a_r].
    The image axes are H = y (spacing dy) and W = x (spacing dx).
    """
    r1, H, W = Pu.shape
    r = r1 - 1
    dA = dx * dy                                          # area of one grid cell
    Mu = Pu[1:].reshape(r, -1)                            # the test modes phi_i
    Mv = Pv[1:].reshape(r, -1)

    # Diffusion term l_ij = (1/Re) * INT [lap(u_j) u_i + lap(v_j) v_i] dA.
    # I take the Laplacian of every field (mean + modes) and then integrate it
    # against each test mode.  INT (.) dA is just a sum over grid points times dA.
    lap_u = np.empty_like(Pu)
    lap_v = np.empty_like(Pv)
    for j in range(r1):
        duy, dux = np.gradient(Pu[j], dy, dx)
        dvy, dvx = np.gradient(Pv[j], dy, dx)
        lap_u[j] = np.gradient(dux, dx, axis=1) + np.gradient(duy, dy, axis=0)
        lap_v[j] = np.gradient(dvx, dx, axis=1) + np.gradient(dvy, dy, axis=0)
    l = (dA / re) * (Mu @ lap_u.reshape(r1, -1).T + Mv @ lap_v.reshape(r1, -1).T)
    del lap_u, lap_v

    # Convection term
    #   q_ijk = - INT [(u_j du_k/dx + v_j du_k/dy) u_i
    #                 +(u_j dv_k/dx + v_j dv_k/dy) v_i] dA.
    # I loop over k (the field being differentiated) and handle all j at once:
    # conv_u/conv_v hold (u_j . grad u_k) for every j, then I integrate against
    # the test modes to fill the whole q[:, :, k] slice.
    q = np.empty((r, r1, r1))
    for k in range(r1):
        duky, dukx = np.gradient(Pu[k], dy, dx)
        dvky, dvkx = np.gradient(Pv[k], dy, dx)
        conv_u = Pu * dukx + Pv * duky                   # [r1, H, W], all j
        conv_v = Pu * dvkx + Pv * dvky
        q[:, :, k] = -dA * (Mu @ conv_u.reshape(r1, -1).T
                            + Mv @ conv_v.reshape(r1, -1).T)
    return l, q


def make_rhs_ns(l, q, s):
    """Return the right-hand side da/dt of the reduced model, written in the
    per-mode standardized variable b = a/s.  I standardize so that IC_NOISE and
    the energy screening mean the same thing here as in the fitted version.
    Internally I rebuild the physical a = b*s, prepend a_0 = 1, evaluate the
    physical model l@c + c@q@c, and divide by s to go back to b-units."""
    def f(b):
        c = np.concatenate(([1.0], b * s))               # c = [1, a_1..a_r]
        return (l @ c + (q @ c) @ c) / s
    return f


def make_step_ns(l, q, s, dt, substeps):
    """One classical RK4 step across a full snapshot interval, split into
    `substeps` smaller sub-steps for accuracy.  Returns a function b -> b_next."""
    f = make_rhs_ns(l, q, s)
    h = dt / substeps
    def step(b):
        for _ in range(substeps):
            k1 = f(b)
            k2 = f(b + 0.5 * h * k1)
            k3 = f(b + 0.5 * h * k2)
            k4 = f(b + h * k3)
            b = b + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return b
    return step


def derivative_R2(l, q, A_phys, tr_idx, dt):
    """A diagnostic, not part of the model.  Nothing is fitted here, so I check
    how well the projected model reproduces the coefficient derivatives that are
    actually seen in the data (central differences on consecutive training
    triples).  A high R^2 means the physics alone explains that mode well; a low
    value measures the closure/truncation error for that mode."""
    ok = np.where((np.diff(tr_idx[:-1]) == 1) & (np.diff(tr_idx[1:]) == 1))[0] + 1
    dA = (A_phys[ok + 1] - A_phys[ok - 1]) / (2.0 * dt)
    pred = np.empty_like(dA)
    for n, a in enumerate(A_phys[ok]):
        c = np.concatenate(([1.0], a))
        pred[n] = l @ c + (q @ c) @ c
    ss_res = ((dA - pred) ** 2).sum(axis=0)
    ss_tot = ((dA - dA.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.where(ss_tot < 1e-300, 1.0, ss_tot)


# ------------------------------- Main -------------------------------

def main():
    data, comp_names = load_data(DATA_FILE, COMP_IDX)
    Nt, C, H, W = data.shape
    if C != 2:
        raise ValueError(f"NS projection needs 2D (u,v) fields, got C={C}")
    dx, dy = grid_spacing(DATA_FILE)
    kappa = np.sqrt(dx * dy)                              # converts vector-norm
                                                         # modes to physical ones
    print(f"[{STRATEGY}]  Re = {RE:g}, dt = {DT:g}, dx = {dx:.4f}, dy = {dy:.4f}")

    # Same split as every other generator (same seed), so val_real matches.
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(Nt)
    n_val   = max(1, round(VAL_FRAC * Nt))
    val_idx = np.sort(idx[:n_val])
    tr_idx  = np.sort(idx[n_val:])
    train_real = data[tr_idx].copy()
    val_real   = data[val_idx].copy()
    del data
    Ntr = len(tr_idx)
    n_aug = N_AUG or Ntr

    # POD on the training snapshots only.
    x_mean, Xc, Wp, S, A, shape = compute_pod(train_real)
    r_full = A.shape[1]
    n_rom0 = int(min(N_ROM, r_full))

    # Put the mean flow and the leading modes back on the grid, and give them
    # the physical normalization the projection integrals expect.  On a uniform
    # grid the vector-orthonormal modes only need a 1/sqrt(dx*dy) rescaling.
    U = (Xc.T @ Wp[:, :n_rom0]).astype(np.float64)        # [D, r], vector-unit
    Pu = np.empty((n_rom0 + 1, H, W))
    Pv = np.empty((n_rom0 + 1, H, W))
    mean_f = x_mean.astype(np.float64).reshape(C, H, W)
    Pu[0], Pv[0] = mean_f[0], mean_f[1]                   # mean flow = "mode 0"
    modes = (U / kappa).T.reshape(n_rom0, C, H, W)        # phi = U / sqrt(dA)
    Pu[1:], Pv[1:] = modes[:, 0], modes[:, 1]
    del U, modes

    # Quick check that the rescaled modes really are orthonormal in the
    # physical inner product, i.e. INT phi_i phi_j dA = delta_ij.
    Mflat = np.concatenate([Pu[1:].reshape(n_rom0, -1),
                            Pv[1:].reshape(n_rom0, -1)], axis=1)
    G = (Mflat @ Mflat.T) * dx * dy
    print(f"[{STRATEGY}]  mode orthonormality: max |G - I| = "
          f"{np.abs(G - np.eye(n_rom0)).max():.2e}")
    del Mflat, G

    # Coefficients in physical units, to match the physically-normalized modes.
    A_phys = np.asarray(A[:, :n_rom0], dtype=np.float64) * kappa

    print(f"[{STRATEGY}]  projecting NS operators onto {n_rom0} modes ...")
    l_full, q_full = galerkin_operators_ns(Pu, Pv, dx, dy, RE)
    del Pu, Pv

    # Generation with automatic back-off.  Because each entry of l and q is an
    # independent integral, a model with fewer modes is just a corner block of
    # the tensors I already computed — no re-projection needed.
    ladder = [n_rom0] + [n for n in (15, 10, 6, 4, 2) if n < n_rom0]
    for n_rom in ladder:
        l = l_full[:n_rom, :n_rom + 1]
        q = q_full[:n_rom, :n_rom + 1, :n_rom + 1]
        Ar = A_phys[:, :n_rom]
        R2 = derivative_R2(l, q, Ar, tr_idx, DT)
        e_rom = (S[:n_rom] ** 2).sum() / (S ** 2).sum()
        print(f"\n[{STRATEGY}]  ns_ode: {n_rom} modes ({100*e_rom:.2f}% energy)")
        print("  derivative R^2 per mode (projected, NOT fitted): "
              + "  ".join(f"{v:.3f}" for v in R2))

        # standardize the coefficients before integrating (per-mode scale s)
        s = Ar.std(axis=0)
        s = np.where(s < 1e-14, 1.0, s)
        B_real = Ar / s[None, :]
        step = make_step_ns(l, q, s, DT, SUBSTEPS)
        try:
            B_new, st = integrate_trajectories(step, s, B_real, n_aug, HORIZON,
                                               IC_NOISE, ENERGY_TOL, MAX_TRAJ, rng)
            # if almost nothing survives, this N_ROM is too drifty -> shrink it
            if st["keep_frac"] < MIN_KEEP and n_rom != ladder[-1]:
                print(f"  !! filled, but only {100*st['keep_frac']:.1f}% of steps "
                      f"survived screening (< {100*MIN_KEEP:.0f}%)"
                      "\n  -> backing off to a smaller model")
                continue
            break
        except RuntimeError as err:
            print(f"  !! {err}")
            if n_rom == ladder[-1]:
                raise
            print("  -> backing off to a smaller model")

    # Back to vector units, then rebuild the synthetic snapshots.
    A_new = (B_new * s[None, :]) / kappa
    train_aug = reconstruct(x_mean, Xc, Wp[:, :n_rom], A_new, shape)
    del Xc

    base = os.path.splitext(os.path.basename(DATA_FILE))[0]
    bundle = {
        "train_real": train_real, "train_aug": train_aug, "val_real": val_real,
        "comp_names": np.array(comp_names, dtype=object),
        "strategy": STRATEGY, "tr_idx": tr_idx, "val_idx": val_idx,
        "rng_seed": RNG_SEED, "val_frac": VAL_FRAC, "model": "ns_ode",
        "n_modes": r_full, "n_rom": n_rom, "dt": DT, "re": RE,
        "dx": dx, "dy": dy, "horizon": HORIZON, "substeps": SUBSTEPS,
        "ic_noise": IC_NOISE, "energy_tol": ENERGY_TOL,
        "fit_R2": R2, "n_traj": st["n_traj"],
    }
    out_file = save_bundle(os.path.join(_AUG_DIR, f"{base}_aug_{STRATEGY}"), bundle)

    print(f"\n[{STRATEGY}]  {base}  |  {H}x{W}  C={C}  |  ns_ode, {n_rom} modes")
    print(f"  real train = {Ntr}   real val = {len(val_idx)}   synthetic = {n_aug}")
    print(f"  trajectories = {st['n_traj']}  (horizon {HORIZON}, IC noise {IC_NOISE})")
    print(f"  screening: kept {100*st['keep_frac']:.1f}% of integrated steps "
          f"(energy band [{st['E_band'][0]:.3g}, {st['E_band'][1]:.3g}], tol {ENERGY_TOL})")
    print(f"  saved → {out_file}\n")


if __name__ == "__main__":
    main()

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


(rom version: the functions of pod_augment_galerkin_ns.py, plus build_ops / model_at of
run_lowdata.py.  The bundle generator main() was dropped with the summer pipeline; the
loaders and POD moved to rom.data / rom.pod.)
"""

import numpy as np

# used by grid_spacing() only
DX_FALLBACK = 0.08     # grid spacing when the file has no coordinates
STRATEGY    = "galerkin_ns_5traj"   # tag of the messages below


def grid_spacing(path):
    """Get the grid spacing dx, dy from the coordinate arrays in the file.
    The x/y grids are uniform, so I take the gap between the first two distinct
    coordinate values along each axis.  The arrays are named DataX/DataY in some
    files, X/Y in others and x_points/y_points in the channel planes; if none is
    there I fall back to the Readme value."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            xkey = next(k for k in ("DataX", "X", "x_points") if k in f)    # x_points / y_points:
            ykey = next(k for k in ("DataY", "Y", "y_points") if k in f)    # the channel planes
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

def galerkin_operators_ns(Pu, Pv, dx, dy, re, forcing_x=0.0):
    """Build the Galerkin operators l and q by projecting the Navier-Stokes
    convection and diffusion terms onto the modes.  This is a vectorized version
    of Dawson's nseGalerkinCoeffsDemo.m (same integrals, just done with matrix
    products instead of the triple loop).

    Pu, Pv[r+1, H, W] : the mean flow in row 0, then the r modes.
    Returns l[r, r+1] and q[r, r+1, r+1] such that
        da_i/dt = l[i,:] @ c + c @ q[i] @ c,   with c = [1, a_1..a_r].
    The image axes are H = y (spacing dy) and W = x (spacing dx).
    forcing_x : a uniform body force in x, e.g. the imposed mean pressure gradient -dP/dx that
                drives a channel. It projects to the constant term G INT u_i dA, stored in the
                a_0 = 1 column of l. 0 (every wake study) leaves l unchanged.
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
    if forcing_x:
        l[:, 0] += forcing_x * dA * Mu.sum(axis=1)

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



# ------------------------- Operators built once, sliced to smaller K (run_lowdata.py) -------------------------

def build_ops(P, K, re_, path, forcing_x=0.0):
    """Projected NS operators with K modes, plus the mode scaling. l[:K, :K+1] and
    q[:K, :K+1, :K+1] of a larger build are exactly the operators of a smaller truncation.
    forcing_x: see galerkin_operators_ns (the channel's mean pressure gradient)."""
    C, H, W = P["shape"]
    dx, dy = grid_spacing(path)
    kappa = np.sqrt(dx * dy)
    U = (P["Xc"].T @ P["Wp"][:, :K]).astype(np.float64)
    Pu = np.empty((K + 1, H, W)); Pv = np.empty((K + 1, H, W))
    mf = P["x_mean"].astype(np.float64).reshape(C, H, W); Pu[0], Pv[0] = mf[0], mf[1]
    md = (U / kappa).T.reshape(K, C, H, W); Pu[1:], Pv[1:] = md[:, 0], md[:, 1]
    del U, md
    l, q = galerkin_operators_ns(Pu, Pv, dx, dy, re_, forcing_x)
    del Pu, Pv
    return l, q, kappa


def model_at(l_big, q_big, kappa, A, K, dt, substeps):
    """Slice the operators to K modes and return (step, s, to_b, from_b)."""
    l, q = l_big[:K, :K + 1], q_big[:K, :K + 1, :K + 1]
    s = (A[:, :K] * kappa).std(axis=0); s = np.where(s < 1e-14, 1.0, s)
    step = make_step_ns(l, q, s, dt, substeps)
    return step, s, (lambda a: a * kappa / s), (lambda b: b * s / kappa)

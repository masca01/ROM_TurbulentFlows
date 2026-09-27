"""
The channel-flow setup shared by studies/7_channel (run_channel.py, round 1, and run_channel_ns.py,
round 2): the split, the real-subset draws, the reserved tail of quality windows, the "nothing
changes" reference, and the POD with an x-homogeneous mean.

    Setup(ds)            data, Re, dt, split; pool before the reserved tail; the quality windows
    Setup.subset         n real snapshots: blocks10 | runs50 | random, seeded as in round 1
    persistence          quality error of a(t) = a(0) on the same windows
    channel_pod          subset POD, optionally centred on the x-average of the subset mean
    add_eddy_viscosity   l + nu_T * lap on the fluctuation columns (mean column untouched)
"""
import numpy as np

from .data import load_channel
from .split import split_indices, draw_blocks, SPLIT_SEED, DRAW_SEED
from .pod import subset_pod
from .augment import fidelity

GEN_TIME      = 2.0      # h / U_bulk per quality window: "nothing changes" error ~1 on the plane
QUALITY_TAIL  = 0.25     # last fraction of the record kept for the quality windows
WINDOW_STRIDE = 25       # snapshots between window starts
SEED_TAG      = 1000     # in the rng seeds where the wake studies have int(Re)
SAMPLINGS     = ["blocks10", "runs50", "random"]   # the index is part of the seeds (round 1: 0, 1)
RUN           = 50       # pool snapshots per run of the runs50 sampling


def draw_runs(pool_idx, n, rng, run=RUN):
    """n training indices made of non-overlapping runs of `run` consecutive POOL positions (the
    validation snapshots in between are skipped, as in the contig sampling of run_lowdata.py)."""
    n_runs = int(np.ceil(n / run))
    starts = list(range(len(pool_idx) - run + 1)); rng.shuffle(starts)
    taken, chosen = set(), []
    for st in starts:
        if taken.isdisjoint(range(st, st + run)):
            chosen.append(st); taken.update(range(st, st + run))
        if len(chosen) == n_runs:
            break
    pos = sorted(i for st in chosen for i in range(st, st + run))[:n]
    return np.asarray(pool_idx)[pos]


def persistence(win_A, steps):
    """Quality error of a(t) = a(0): the reference every model has to beat."""
    ident = lambda v: v
    return fidelity(ident, ident, ident, win_A, steps)[0]


class Setup:
    """The dataset, its split, and the real subsets, drawn exactly the same way in every stage."""
    def __init__(self, ds, gen_time=GEN_TIME):
        self.ds = ds
        self.data, self.re, self.dt, self.path = load_channel(ds)
        self.Nt = len(self.data)
        self.val_idx, pool_all, _ = split_indices(self.Nt, None, "random", SPLIT_SEED)
        self.pool_all = pool_all                                  # capacity: the whole training pool
        self.t_cut = int(round((1 - QUALITY_TAIL) * self.Nt))     # quality windows live after it
        self.pool_idx = pool_all[pool_all < self.t_cut]           # real subsets + real_proj before it
        self.pool_set = set(int(i) for i in self.pool_idx)
        self.val_real = self.data[self.val_idx]
        self.steps = max(1, int(round(gen_time / self.dt)))
        self.win = list(range(self.t_cut, self.Nt - self.steps, WINDOW_STRIDE))

    def subset(self, n, sm, sub):
        """(training indices, quality windows [steps+1 snapshots each])."""
        rng = np.random.default_rng([DRAW_SEED, SEED_TAG, n, sub, SAMPLINGS.index(sm)])
        if sm == "blocks10":
            tr_idx = draw_blocks(self.pool_set, self.t_cut, n, rng)
        elif sm == "runs50":
            tr_idx = draw_runs(self.pool_idx, n, rng)
        else:                                                     # random: the NS model needs no sequence
            tr_idx = np.sort(rng.choice(self.pool_idx, size=n, replace=False))
        return tr_idx, [self.data[t:t + self.steps + 1] for t in self.win]


def channel_pod(train, homogeneous_x=False):
    """subset_pod of the real snapshots. homogeneous_x: centre on the x-average of the subset mean,
    i.e. the mean profile u(y), v(y), which is what the mean of a channel is (homogeneous in x).
    A few hundred snapshots leave frozen turbulent structures in the plain temporal mean; with the
    x-average they stay in the fluctuations, where the modes can represent them."""
    if not homogeneous_x:
        return subset_pod(train)
    Ntr, C, H, W = train.shape
    m = np.broadcast_to(train.mean(axis=(0, 3), keepdims=True), (1, C, H, W)).reshape(-1)
    X = train.reshape(Ntr, -1)
    Xc = (X - m[None, :]).astype(np.float32)
    G = (Xc @ Xc.T).astype(np.float64)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None); V = V[:, order]
    S = np.sqrt(w)
    keep = S > (S[0] * 1e-10 if S.size else 0.0)
    S, V = S[keep], V[:, keep]
    A = V * S[None, :]
    Wp = (V / S[None, :]).astype(np.float32)
    e_cum = np.cumsum(S ** 2) / (S ** 2).sum()
    return dict(x_mean=m.astype(np.float32), Xc=Xc, Wp=Wp, S=S, A=A, shape=(C, H, W), e_cum=e_cum)


def add_eddy_viscosity(l, lap, nu_t):
    """l with an eddy viscosity nu_t on the fluctuation modes: l[:, 1:] + nu_t * lap[:, 1:].
    The a_0 = 1 (mean) column is untouched: the mean flow is balanced by the pressure gradient and
    the Reynolds stress, and an eddy viscosity acting on it would add a large spurious forcing.
    lap from build_ops(..., return_lap=True). Slicing to smaller K stays exact."""
    if not nu_t:
        return l
    out = l.copy()
    out[:, 1:] += nu_t * lap[:, 1:]
    return out

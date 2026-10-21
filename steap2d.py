#!/usr/bin/env python3
"""
steap2d.py - Simultaneous Trajectory Estimation and Planning (STEAP) for a 2D point robot.

A compact, dependency-light re-implementation of the machinery in
    Mukadam, Dong, Dellaert, Boots. "STEAP: simultaneous trajectory estimation
    and planning." Autonomous Robots, 2018 (arXiv:1807.10425).

What is implemented
-------------------
* State per support time t_i:  theta_i = [px, py, vx, vy]  (constant-velocity GP).
* Factors (all whitened, so the MAP problem is  argmin 1/2 * sum ||e_f||^2):
    - GPPriorFactor           f_gp(theta_i, theta_i+1)            Eq. (33)-(34)
    - StatePriorFactor        start / goal / measurement / f_curr  Eq. (21)-(23)
    - ObstacleFactor          hinge loss on a signed distance field (GPMP2)
    - InterpObstacleFactor    obstacle cost at t_i + tau, GP-interpolated from
                              theta_i and theta_i+1                Eq. (35)
* A sparse factor graph + Levenberg-Marquardt solver (scipy.sparse).
* STEAP loop (Algorithm 1): plan -> interpolate segment -> collision check ->
  execute with noisy dynamics -> noisy localization -> add measurement factor
  -> re-solve the *whole* start-to-goal trajectory.
* Baselines from the paper: open loop (OL) and SLAP (separate estimation and
  planning graphs, solved in sequence).

What is simplified
------------------
The paper updates a Bayes tree incrementally with iSAM2. Here each step
re-solves the full sparse graph with a few warm-started LM iterations. That
gives the same MAP answer (up to convergence) and is plenty fast for ~20-50
states. For true incremental inference, swap `FactorGraph.optimize` for
gtsam.ISAM2 (the gtsam / gpmp2 Python bindings provide the same factors).

Usage
-----
    python steap2d.py                       # animate one STEAP run + compare OL / SLAP / STEAP
    python steap2d.py --seed 7 --n-dyn 0.3  # harder motion noise
    python steap2d.py --save steap.gif      # write the animation to a file
    python steap2d.py --benchmark 40        # success / error table over 40 random worlds

Requires numpy, scipy, matplotlib (pillow for --save *.gif).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.sparse.linalg import spsolve


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # trajectory / GP prior
    n_states: int = 20            # support states theta_0 .. theta_N  (N = n_states)
    total_time: float = 10.0      # seconds from start to goal
    qc: float = 0.5               # power-spectral density Q_C of the acceleration noise
    # robot / obstacles
    radius: float = 0.20          # robot is a disc of this radius (m)
    eps: float = 0.30             # safety margin of the hinge loss (m)
    sigma_obs: float = 0.05       # obstacle factor std-dev
    n_interp: int = 3             # interpolated obstacle factors between each pair of states
    sigma_fix: float = 1e-3       # start / goal tie-down std-dev
    # stochasticity (matches the paper's n_dyn / n_cam knobs)
    n_dyn: float = 0.20           # uniform bounded velocity noise (m/s), redrawn each segment
    n_cam: float = 0.10           # Gaussian position-measurement noise std (m)
    # solver
    batch_iters: int = 100
    step_iters: int = 10
    substeps: int = 10            # execution resolution per segment (GP upsampling)
    # world
    size: tuple = (12.0, 8.0)
    cell: float = 0.05
    n_obstacles: int = 10
    start: tuple = (1.0, 1.0)
    goal: tuple = (11.0, 7.0)

    @property
    def dt(self) -> float:
        return self.total_time / self.n_states


# =============================================================================
# World: occupancy grid -> signed distance field
# =============================================================================
class SignedDistanceField:
    """Grid SDF (positive outside obstacles) with bilinear lookup and gradient."""

    def __init__(self, occupancy: np.ndarray, cell: float, origin=(0.0, 0.0)):
        self.occ = occupancy
        self.cell = cell
        self.origin = np.asarray(origin, float)
        outside = distance_transform_edt(~occupancy) * cell
        inside = distance_transform_edt(occupancy) * cell
        self.field = outside - inside
        # field is indexed [row = y, col = x]
        self.grad_y, self.grad_x = np.gradient(self.field, cell)

    def _rc(self, p):
        p = np.atleast_2d(p)
        col = (p[:, 0] - self.origin[0]) / self.cell
        row = (p[:, 1] - self.origin[1]) / self.cell
        return np.vstack([row, col])

    def distance(self, p) -> np.ndarray:
        return map_coordinates(self.field, self._rc(p), order=1, mode="nearest")

    def query(self, p):
        rc = self._rc(p)
        d = map_coordinates(self.field, rc, order=1, mode="nearest")
        gx = map_coordinates(self.grad_x, rc, order=1, mode="nearest")
        gy = map_coordinates(self.grad_y, rc, order=1, mode="nearest")
        return d, np.stack([gx, gy], axis=1)

    @property
    def extent(self):
        h, w = self.field.shape
        return (self.origin[0], self.origin[0] + w * self.cell,
                self.origin[1], self.origin[1] + h * self.cell)


def make_world(cfg: Config, seed: int) -> SignedDistanceField:
    """Random circles and boxes, plus boundary walls; start and goal kept clear."""
    rng = np.random.default_rng(seed)
    W, H = cfg.size
    nx, ny = int(round(W / cfg.cell)), int(round(H / cfg.cell))
    ys, xs = np.mgrid[0:ny, 0:nx] * cfg.cell
    occ = np.zeros((ny, nx), bool)
    wall = 0.1
    occ |= (xs < wall) | (xs > W - wall) | (ys < wall) | (ys > H - wall)

    start, goal = np.array(cfg.start), np.array(cfg.goal)
    clear = cfg.radius + cfg.eps + 0.6
    placed, tries = 0, 0
    # one obstacle near the straight line so the planner has to work
    centers = [0.5 * (start + goal) + rng.uniform(-0.4, 0.4, 2)]
    while placed < cfg.n_obstacles and tries < 2000:
        tries += 1
        c = centers.pop() if centers else rng.uniform([1.0, 0.8], [W - 1.0, H - 0.8])
        if kind_circle := rng.random() < 0.5:
            r = rng.uniform(0.35, 0.8)
            mask = (xs - c[0]) ** 2 + (ys - c[1]) ** 2 < r ** 2
            reach = r
        else:
            hw, hh = rng.uniform(0.25, 0.8, 2)
            mask = (np.abs(xs - c[0]) < hw) & (np.abs(ys - c[1]) < hh)
            reach = np.hypot(hw, hh)
        if (np.linalg.norm(c - start) < reach + clear or
                np.linalg.norm(c - goal) < reach + clear):
            continue
        occ |= mask
        placed += 1
    return SignedDistanceField(occ, cfg.cell)


# =============================================================================
# Constant-velocity GP prior
# =============================================================================
def gp_phi(dt: float) -> np.ndarray:
    """State transition Phi(t, s) for dt = t - s."""
    I, Z = np.eye(2), np.zeros((2, 2))
    return np.block([[I, dt * I], [Z, I]])


def gp_Q(dt: float, qc: float) -> np.ndarray:
    """Process covariance Q_{i,i+1}, Eq. (34)."""
    Qc = qc * np.eye(2)
    return np.block([[dt ** 3 / 3 * Qc, dt ** 2 / 2 * Qc],
                     [dt ** 2 / 2 * Qc, dt * Qc]])


def whitener(cov: np.ndarray) -> np.ndarray:
    """Return W with ||W e||^2 = e^T cov^-1 e."""
    return np.linalg.cholesky(np.linalg.inv(cov)).T


def gp_interp_matrices(tau: float, dt: float, qc: float):
    """theta(t_i + tau) = Lam @ theta_i + Psi @ theta_{i+1}  (posterior-mean interpolation, Eq. 35)."""
    Psi = gp_Q(tau, qc) @ gp_phi(dt - tau).T @ np.linalg.inv(gp_Q(dt, qc))
    Lam = gp_phi(tau) - Psi @ gp_phi(dt)
    return Lam, Psi


def interpolate(x: np.ndarray, i: int, tau: float, cfg: Config) -> np.ndarray:
    Lam, Psi = gp_interp_matrices(tau, cfg.dt, cfg.qc)
    return Lam @ x[i] + Psi @ x[i + 1]


# =============================================================================
# Factors. Each returns (whitened error, [Jacobian w.r.t. each connected state]).
# =============================================================================
class GPPriorFactor:
    def __init__(self, i: int, dt: float, qc: float):
        self.states = (i, i + 1)
        self.Phi = gp_phi(dt)
        self.W = whitener(gp_Q(dt, qc))
        self.J = [self.W @ self.Phi, -self.W]

    def evaluate(self, a, b):
        return self.W @ (self.Phi @ a - b), self.J


class StatePriorFactor:
    """Gaussian factor on selected dims of one state: start, goal, measurement, f_curr."""

    def __init__(self, i: int, mean, sigma, dims=(0, 1, 2, 3), kind="prior"):
        self.states = (i,)
        self.dims = list(dims)
        self.mean = np.asarray(mean, float)
        self.w = 1.0 / np.broadcast_to(np.asarray(sigma, float), (len(self.dims),))
        J = np.zeros((len(self.dims), 4))
        J[np.arange(len(self.dims)), self.dims] = self.w
        self.J = [J]
        self.kind = kind

    def evaluate(self, a):
        return (a[self.dims] - self.mean) * self.w, self.J


def _hinge(sdf, p, cfg):
    d, g = sdf.query(p)
    d = d[0] - cfg.radius
    if d >= cfg.eps:
        return 0.0, np.zeros(2), False
    return (cfg.eps - d) / cfg.sigma_obs, -g[0] / cfg.sigma_obs, True


class ObstacleFactor:
    def __init__(self, i: int, sdf: SignedDistanceField, cfg: Config):
        self.states = (i,)
        self.sdf, self.cfg = sdf, cfg

    def evaluate(self, a):
        e, de_dp, _ = _hinge(self.sdf, a[:2], self.cfg)
        J = np.zeros((1, 4))
        J[0, :2] = de_dp
        return np.array([e]), [J]


class InterpObstacleFactor:
    def __init__(self, i: int, tau: float, sdf: SignedDistanceField, cfg: Config):
        self.states = (i, i + 1)
        self.sdf, self.cfg = sdf, cfg
        self.Lam, self.Psi = gp_interp_matrices(tau, cfg.dt, cfg.qc)

    def evaluate(self, a, b):
        p = (self.Lam @ a + self.Psi @ b)[:2]
        e, de_dp, _ = _hinge(self.sdf, p, self.cfg)
        return np.array([e]), [de_dp[None, :] @ self.Lam[:2], de_dp[None, :] @ self.Psi[:2]]


# =============================================================================
# Sparse factor graph + Levenberg-Marquardt
# =============================================================================
class FactorGraph:
    def __init__(self):
        self.factors: list = []

    def add(self, f):
        self.factors.append(f)
        return f

    def extend(self, fs):
        self.factors.extend(fs)

    def remove(self, f):
        self.factors.remove(f)

    def _eval(self, f, x):
        return f.evaluate(*[x[k] for k in f.states])

    def cost(self, x) -> float:
        return 0.5 * sum(float(np.sum(self._eval(f, x)[0] ** 2)) for f in self.factors)

    def linearize(self, x, colmap):
        rows, cols, vals, res = [], [], [], []
        r0 = 0
        for f in self.factors:
            e, Js = self._eval(f, x)
            m = len(e)
            res.append(e)
            for k, J in zip(f.states, Js):
                rr, cc = np.meshgrid(np.arange(r0, r0 + m), np.arange(4 * colmap[k], 4 * colmap[k] + 4),
                                     indexing="ij")
                rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(np.asarray(J).ravel())
            r0 += m
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(r0, 4 * len(colmap)))
        return np.concatenate(res), J

    def optimize(self, x0: np.ndarray, max_iters=50, lam=1e-4, rel_tol=1e-6):
        """LM over the states touched by at least one factor; others are left as-is."""
        x = x0.copy()
        states = sorted({k for f in self.factors for k in f.states})
        colmap = {k: j for j, k in enumerate(states)}
        idx = np.array(states)
        cost = self.cost(x)
        for it in range(max_iters):
            r, J = self.linearize(x, colmap)
            H = (J.T @ J).tocsc()
            g = J.T @ r
            D = sp.diags(H.diagonal() + 1e-9)
            while True:
                dx = spsolve((H + lam * D).tocsc(), -g)
                x_new = x.copy()
                x_new[idx] += dx.reshape(-1, 4)
                c_new = self.cost(x_new)
                if c_new < cost:
                    break
                lam *= 10
                if lam > 1e10:
                    return x, cost
            converged = (cost - c_new) < rel_tol * max(cost, 1e-12)
            x, cost, lam = x_new, c_new, max(lam / 10, 1e-9)
            if converged:
                break
        return x, cost


# =============================================================================
# Factor builders
# =============================================================================
def gp_chain(lo, hi, cfg):
    return [GPPriorFactor(i, cfg.dt, cfg.qc) for i in range(lo, hi)]


def obstacle_factors(lo, hi, sdf, cfg):
    fs = [ObstacleFactor(i, sdf, cfg) for i in range(lo, hi + 1)]
    for i in range(lo, hi):
        for j in range(1, cfg.n_interp + 1):
            fs.append(InterpObstacleFactor(i, j * cfg.dt / (cfg.n_interp + 1), sdf, cfg))
    return fs


def start_goal(cfg):
    s = StatePriorFactor(0, [*cfg.start, 0, 0], cfg.sigma_fix, kind="start")
    g = StatePriorFactor(cfg.n_states, [*cfg.goal, 0, 0], cfg.sigma_fix, kind="goal")
    return s, g


def initial_trajectory(cfg, bend=0.3):
    """Straight line at constant velocity, bent sideways by `bend` metres to break symmetry."""
    N = cfg.n_states
    s, g = np.array(cfg.start), np.array(cfg.goal)
    t = np.linspace(0, 1, N + 1)[:, None]
    normal = np.array([-(g - s)[1], (g - s)[0]]) / np.linalg.norm(g - s)
    pos = s + t * (g - s) + bend * np.sin(np.pi * t) * normal
    vel = np.tile((g - s) / cfg.total_time, (N + 1, 1))
    return np.hstack([pos, vel])


def batch_plan(graph, sdf, cfg, bends=(0.3, -0.3, 1.5, -1.5, 3.0, -3.0)):
    """Initial batch solve. Trajectory optimization is local, so try a few initializations
    (Sec. 10 of the paper) and keep the lowest-cost collision-free result."""
    best = None
    for b in bends:
        x, c = graph.optimize(initial_trajectory(cfg, b), cfg.batch_iters)
        free = all(segment_collision_free(x, i, sdf, cfg) for i in range(cfg.n_states))
        key = (not free, c)
        if best is None or key < best[0]:
            best = (key, x)
        if free and b == bends[0]:
            break              # the default seed usually works; skip the rest
    return best[1]


# =============================================================================
# Robot simulation
# =============================================================================
def segment_collision_free(x, i, sdf, cfg, n=10):
    pts = np.array([interpolate(x, i, k * cfg.dt / n, cfg)[:2] for k in range(n + 1)])
    return bool(np.all(sdf.distance(pts) > cfg.radius))


def execute_segment(x_plan, i, p0, vel_noise, sdf, cfg):
    """Follow the planned velocity profile of segment i with additive velocity noise."""
    h = cfg.dt / cfg.substeps
    p = p0.copy()
    path = [p.copy()]
    for k in range(cfg.substeps):
        v = interpolate(x_plan, i, (k + 0.5) * h, cfg)[2:] + vel_noise
        p = p + v * h
        path.append(p.copy())
    path = np.array(path)
    collided = bool(np.any(sdf.distance(path) < cfg.radius))
    return path, collided


@dataclass
class Noise:
    dyn: np.ndarray   # (N, 2) velocity noise per segment
    cam: np.ndarray   # (N, 2) measurement noise per step

    @staticmethod
    def sample(cfg, seed):
        rng = np.random.default_rng(10_000 + seed)
        N = cfg.n_states
        return Noise(rng.uniform(-cfg.n_dyn, cfg.n_dyn, (N, 2)), rng.normal(0, cfg.n_cam, (N, 2)))


@dataclass
class Result:
    method: str
    status: str = "running"            # "success" | "collision" | "plan-blocked"
    truth: list = field(default_factory=list)       # executed positions, dense
    truth_states: list = field(default_factory=list)  # true position at each t_i
    meas: list = field(default_factory=list)
    estimate: np.ndarray | None = None
    first_plan: np.ndarray | None = None
    frames: list = field(default_factory=list)      # snapshots for animation
    step_times: list = field(default_factory=list)

    @property
    def success(self):
        return self.status == "success"

    def goal_error(self, cfg):
        return float(np.linalg.norm(self.truth_states[-1] - np.array(cfg.goal)))

    def estimation_error(self):
        if self.estimate is None:
            return float("nan")
        T = np.array(self.truth_states)
        return float(np.mean(np.linalg.norm(self.estimate[:len(T), :2] - T, axis=1)))


# =============================================================================
# Methods
# =============================================================================
def run_steap(sdf, cfg, noise, record=True) -> Result:
    """Algorithm 1: one graph, start to goal, re-solved after every measurement."""
    N = cfg.n_states
    res = Result("STEAP")
    graph = FactorGraph()
    f_start, f_goal = start_goal(cfg)
    graph.extend(gp_chain(0, N, cfg))
    graph.extend(obstacle_factors(0, N, sdf, cfg))
    graph.extend([f_start, f_goal])

    t0 = time.perf_counter()
    x = batch_plan(graph, sdf, cfg)                                       # GPMP2 special case
    res.step_times.append(time.perf_counter() - t0)
    res.first_plan = x.copy()

    p = np.array(cfg.start, float)
    res.truth, res.truth_states = [p.copy()], [p.copy()]
    if record:
        res.frames.append(dict(i=0, x=x.copy(), truth=np.array(res.truth), meas=[]))

    for i in range(N):
        if not segment_collision_free(x, i, sdf, cfg):
            res.status = "plan-blocked"
            break
        path, hit = execute_segment(x, i, p, noise.dyn[i], sdf, cfg)
        p = path[-1]
        res.truth.extend(path[1:]); res.truth_states.append(p.copy())
        if hit:
            res.status = "collision"
            break
        z = p + noise.cam[i]                                              # localize()
        res.meas.append(z)
        graph.add(StatePriorFactor(i + 1, z, max(cfg.n_cam, 1e-2), dims=(0, 1), kind="meas"))
        if i + 1 == N:
            graph.remove(f_goal)   # final update: goal factor replaced by the last measurement
        t0 = time.perf_counter()
        x, _ = graph.optimize(x, cfg.step_iters)                          # incrementalInference()
        res.step_times.append(time.perf_counter() - t0)
        if record:
            res.frames.append(dict(i=i + 1, x=x.copy(), truth=np.array(res.truth), meas=list(res.meas)))
    else:
        res.status = "success"
    res.estimate = x
    return res


def run_slap(sdf, cfg, noise, record=False) -> Result:
    """Estimation graph (past only) then planning graph (future only), solved in sequence."""
    N = cfg.n_states
    res = Result("SLAP")
    f_start, f_goal = start_goal(cfg)
    plan_g = FactorGraph()
    plan_g.extend(gp_chain(0, N, cfg) + obstacle_factors(0, N, sdf, cfg) + [f_start, f_goal])
    x = batch_plan(plan_g, sdf, cfg)
    res.first_plan = x.copy()
    est_factors = [StatePriorFactor(0, [*cfg.start, 0, 0], cfg.sigma_fix)]
    x_est = x.copy()

    p = np.array(cfg.start, float)
    res.truth, res.truth_states = [p.copy()], [p.copy()]
    for i in range(N):
        if not segment_collision_free(x, i, sdf, cfg):
            res.status = "plan-blocked"
            break
        path, hit = execute_segment(x, i, p, noise.dyn[i], sdf, cfg)
        p = path[-1]
        res.truth.extend(path[1:]); res.truth_states.append(p.copy())
        if hit:
            res.status = "collision"
            break
        z = p + noise.cam[i]
        res.meas.append(z)
        t0 = time.perf_counter()
        # 1) estimation: f_prior * f_gp * f_meas over states 0..i+1
        est_factors += [GPPriorFactor(i, cfg.dt, cfg.qc),
                        StatePriorFactor(i + 1, z, max(cfg.n_cam, 1e-2), dims=(0, 1))]
        est_g = FactorGraph(); est_g.extend(est_factors)
        x_est, _ = est_g.optimize(np.vstack([x_est[:i + 1], x[i + 1:]]), cfg.step_iters)
        # 2) planning: f_curr * f_gp * f_obs * f_goal over states i+1..N
        if i + 1 < N:
            plan_g = FactorGraph()
            plan_g.extend(gp_chain(i + 1, N, cfg) + obstacle_factors(i + 1, N, sdf, cfg) + [
                StatePriorFactor(i + 1, x_est[i + 1], cfg.sigma_fix), f_goal])
            x_seed = x.copy(); x_seed[i + 1] = x_est[i + 1]
            x, _ = plan_g.optimize(x_seed, cfg.step_iters)
        res.step_times.append(time.perf_counter() - t0)
    else:
        res.status = "success"
    res.estimate = x_est
    return res


def run_open_loop(sdf, cfg, noise, record=False) -> Result:
    """Plan once with GPMP2, then execute blindly."""
    N = cfg.n_states
    res = Result("OL")
    f_start, f_goal = start_goal(cfg)
    g = FactorGraph()
    g.extend(gp_chain(0, N, cfg) + obstacle_factors(0, N, sdf, cfg) + [f_start, f_goal])
    x = batch_plan(g, sdf, cfg)
    res.first_plan = x
    p = np.array(cfg.start, float)
    res.truth, res.truth_states = [p.copy()], [p.copy()]
    for i in range(N):
        path, hit = execute_segment(x, i, p, noise.dyn[i], sdf, cfg)
        p = path[-1]
        res.truth.extend(path[1:]); res.truth_states.append(p.copy())
        if hit:
            res.status = "collision"
            break
    else:
        res.status = "success"
    return res


# =============================================================================
# Plotting
# =============================================================================
COLORS = dict(truth="#17905a", est="#cf3434", plan="#2160c4", meas="#d18f00", ol="#8a96a3", slap="#a26f2b")


def dense_curve(x, cfg, lo, hi, n=12):
    pts = [x[lo, :2]]
    for i in range(lo, hi):
        for k in range(1, n + 1):
            pts.append(interpolate(x, i, k * cfg.dt / n, cfg)[:2])
    return np.array(pts)


def draw_world(ax, sdf, cfg):
    ax.imshow(np.clip(sdf.field, -0.5, 2.0), origin="lower", extent=sdf.extent, cmap="Greys_r",
              alpha=0.35, vmin=-0.5, vmax=2.0)
    ax.contourf(*np.meshgrid(np.linspace(*sdf.extent[:2], sdf.field.shape[1]),
                             np.linspace(*sdf.extent[2:], sdf.field.shape[0])),
                sdf.occ.astype(float), levels=[0.5, 1.5], colors=["#33414f"])
    ax.plot(*cfg.start, "ko", ms=6)
    ax.plot(*cfg.goal, marker="*", color="k", ms=14)
    ax.set_xlim(sdf.extent[:2]); ax.set_ylim(sdf.extent[2:])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def animate_steap(res, sdf, cfg, save=None, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(9, 6.4))
    draw_world(ax, sdf, cfg)
    (l_truth,) = ax.plot([], [], color=COLORS["truth"], lw=2.2, label="ground truth")
    (l_est,) = ax.plot([], [], color=COLORS["est"], lw=2.4, label="estimated past")
    (l_plan,) = ax.plot([], [], color=COLORS["plan"], lw=2.2, label="planned future")
    (l_states,) = ax.plot([], [], "o", mfc="white", mec=COLORS["plan"], ms=5)
    (l_meas,) = ax.plot([], [], "x", color=COLORS["meas"], ms=7, mew=2, label="measurements")
    robot = Circle((0, 0), cfg.radius, color=COLORS["truth"], zorder=5)
    ax.add_patch(robot)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    title = ax.set_title("")

    def update(k):
        f = res.frames[k]
        i, x = f["i"], f["x"]
        T = f["truth"]
        l_truth.set_data(T[:, 0], T[:, 1])
        est = dense_curve(x, cfg, 0, i) if i > 0 else np.empty((0, 2))
        l_est.set_data(est[:, 0], est[:, 1])
        plan = dense_curve(x, cfg, i, cfg.n_states)
        l_plan.set_data(plan[:, 0], plan[:, 1])
        l_states.set_data(x[i:, 0], x[i:, 1])
        if f["meas"]:
            M = np.array(f["meas"]); l_meas.set_data(M[:, 0], M[:, 1])
        else:
            l_meas.set_data([], [])
        robot.center = T[-1]
        last = k == len(res.frames) - 1
        title.set_text(f"STEAP  step {i}/{cfg.n_states}" + (f"   [{res.status}]" if last else ""))
        return l_truth, l_est, l_plan, l_states, l_meas, robot, title

    anim = FuncAnimation(fig, update, frames=len(res.frames), interval=350, blit=False, repeat_delay=1500)
    if save:
        writer = "pillow" if save.endswith(".gif") else None
        anim.save(save, writer=writer, dpi=90)
        print(f"saved animation -> {save}")
    if show:
        plt.show()
    return anim


def plot_comparison(results, sdf, cfg, save=None, show=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.2))
    for ax, r in zip(np.atleast_1d(axes), results):
        draw_world(ax, sdf, cfg)
        fp = dense_curve(r.first_plan, cfg, 0, cfg.n_states)
        ax.plot(fp[:, 0], fp[:, 1], "--", color=COLORS["plan"], lw=1, alpha=0.6, label="initial plan")
        T = np.array(r.truth)
        ax.plot(T[:, 0], T[:, 1], color=COLORS["truth"], lw=2, label="ground truth")
        if r.estimate is not None:
            n = len(r.truth_states) - 1
            est = dense_curve(r.estimate, cfg, 0, n)
            ax.plot(est[:, 0], est[:, 1], color=COLORS["est"], lw=2, label="final estimate")
        if r.meas:
            M = np.array(r.meas); ax.plot(M[:, 0], M[:, 1], "x", color=COLORS["meas"], ms=5)
        if r.status == "collision":
            ax.plot(*T[-1], "x", color="red", ms=14, mew=3)
        sub = f"goal err {r.goal_error(cfg):.2f} m"
        if r.estimate is not None:
            sub += f" · est err {r.estimation_error():.3f} m"
        ax.set_title(f"{r.method}: {r.status}\n{sub}", fontsize=10)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
        print(f"saved comparison -> {save}")
    if show:
        plt.show()


# =============================================================================
# Benchmark
# =============================================================================
def benchmark(cfg, n_seeds):
    methods = [("OL", run_open_loop), ("SLAP", run_slap), ("STEAP", run_steap)]
    stats = {m: dict(succ=[], goal=[], est=[], t=[]) for m, _ in methods}
    for seed in range(n_seeds):
        sdf = make_world(cfg, seed)
        noise = Noise.sample(cfg, seed)
        for name, fn in methods:
            r = fn(sdf, cfg, noise, record=False)
            s = stats[name]
            s["succ"].append(r.success)
            if r.success:
                s["goal"].append(r.goal_error(cfg))
                if r.estimate is not None:
                    s["est"].append(r.estimation_error())
            # per-step update time (STEAP's first entry is the initial batch solve)
            s["t"].extend(r.step_times[1:] if name == "STEAP" else r.step_times)
        print(f"\r  seed {seed + 1}/{n_seeds}", end="", flush=True)
    print(f"\n\nn_dyn = {cfg.n_dyn}, n_cam = {cfg.n_cam}, {cfg.n_states} states, {n_seeds} worlds")
    print(f"{'method':<7}{'success':>9}{'goal err (m)':>14}{'est err (m)':>13}{'ms / step':>11}")
    for name, _ in methods:
        s = stats[name]
        f = lambda v: f"{np.mean(v):.3f}" if v else "  -  "
        t = f"{1e3 * np.mean(s['t']):.1f}" if s["t"] else "  -  "
        print(f"{name:<7}{np.mean(s['succ']):>9.2f}{f(s['goal']):>14}{f(s['est']):>13}{t:>11}")
    print("(goal / estimation errors averaged over successful runs only, as in the paper)")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3, help="world + noise seed")
    ap.add_argument("--states", type=int, default=Config.n_states)
    ap.add_argument("--n-dyn", type=float, default=Config.n_dyn, help="velocity noise bound (m/s)")
    ap.add_argument("--n-cam", type=float, default=Config.n_cam, help="measurement noise std (m)")
    ap.add_argument("--save", help="save STEAP animation (.gif or .mp4); comparison saved alongside as .png")
    ap.add_argument("--no-show", action="store_true", help="don't open plot windows")
    ap.add_argument("--benchmark", type=int, metavar="N", help="run OL / SLAP / STEAP over N random worlds")
    args = ap.parse_args()

    cfg = Config(n_states=args.states, n_dyn=args.n_dyn, n_cam=args.n_cam)
    if args.benchmark:
        benchmark(cfg, args.benchmark)
        return
    if args.no_show:
        import matplotlib; matplotlib.use("Agg")

    sdf = make_world(cfg, args.seed)
    noise = Noise.sample(cfg, args.seed)
    results = [run_open_loop(sdf, cfg, noise), run_slap(sdf, cfg, noise), run_steap(sdf, cfg, noise)]
    for r in results:
        line = f"{r.method:<6} {r.status:<13} goal err {r.goal_error(cfg):.3f} m"
        if r.estimate is not None:
            line += f"   est err {r.estimation_error():.3f} m"
        if r.method == "STEAP":
            line += (f"   batch {1e3 * r.step_times[0]:.0f} ms, "
                     f"update {1e3 * np.mean(r.step_times[1:] or [0]):.1f} ms/step")
        print(line)
    if r.meas:
        raw = np.mean([np.linalg.norm(z - t) for z, t in zip(r.meas, r.truth_states[1:])])
        print(f"raw measurement error {raw:.3f} m  (STEAP smooths this with the GP prior + obstacle costs)")

    show = not args.no_show
    plot_comparison(results, sdf, cfg, save=args.save.rsplit(".", 1)[0] + "_compare.png" if args.save else None,
                    show=False)  # displayed together with the animation by plt.show()
    animate_steap(results[-1], sdf, cfg, save=args.save, show=show)


if __name__ == "__main__":
    main()

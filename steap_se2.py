#!/usr/bin/env python3
"""
steap_se2.py - STEAP for a planar SE(2) robot with a sparse GP prior on the Lie group.

Combines
  * Mukadam, Dong, Dellaert, Boots. "STEAP: simultaneous trajectory estimation and
    planning." Autonomous Robots 2018 (arXiv:1807.10425), Sec. 5-7, and
  * Dong, Boots, Dellaert. "Sparse Gaussian processes for continuous-time trajectory
    estimation on matrix Lie groups." arXiv:1705.06020.

Model
-----
Each support state is  x_i = {T_i, w_i}  with  T_i in SE(2)  (pose: x, y, heading) and
w_i in R^3 the body-frame velocity  w = (T^-1 dT/dt)^vee = [v_x, v_y, omega].
Stored as a row  [x, y, th, v_x, v_y, omega].  Optimisation happens in the tangent space:
    T <- T Exp(d_T),   w <- w + d_w           (right perturbation, 6 DOF per state)

GP prior (constant body velocity, dw/dt = white noise), linearised locally around T_i:
    xi_i(t)     = Log(T_i^-1 T(t))                         local pose
    gamma_i(t)  = [xi_i(t); d xi_i/dt],  d xi_i/dt = J_r(xi_i)^-1 w(t)
    gamma_i(t_i)     = [0;        w_i]
    gamma_i(t_i+1)   = [xi;  J_r(xi)^-1 w_i+1],   xi = Log(T_i^-1 T_i+1)
    error            = Phi(dt) gamma_i(t_i) - gamma_i(t_i+1)
                     = [dt w_i - xi;  w_i - J_r(xi)^-1 w_i+1]       whitened by Q_i
Interpolation at t_i + tau (used to execute, collision-check and add obstacle costs):
    gamma(tau) = Lam gamma_i(t_i) + Psi gamma_i(t_i+1)
    T(tau)     = T_i Exp(gamma_1(tau)),   w(tau) = J_r(gamma_1)^-1 gamma_2(tau)

The robot is a 0.9 m x 0.3 m rectangle covered by four discs (as in GPMP2), so its
heading matters for collisions, unlike the point robot in steap2d.py.

Jacobians of the GP, pose-prior and obstacle factors are analytic (closed-form SE(2)
right/left Jacobians); the small d(J_r^-1 w)/d xi term and the interpolated pose inside
the between-state obstacle factors are differentiated numerically. Run
`--check-jacobians` to compare every factor against finite differences.

As in steap2d.py, each step re-solves the full sparse graph with warm-started
Levenberg-Marquardt instead of the paper's incremental Bayes-tree (iSAM2) update.

Usage
-----
    python steap_se2.py                     # animate STEAP, compare with OL and SLAP
    python steap_se2.py --diff-drive        # add soft non-holonomic constraint v_y ~ 0
    python steap_se2.py --save se2.gif      # write animation (+ _compare.png)
    python steap_se2.py --benchmark 20      # OL / SLAP / STEAP over random worlds
    python steap_se2.py --check-jacobians   # verify analytic Jacobians numerically
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import spsolve

D = 6  # tangent dimension of one state {T in SE(2), w in R^3}


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # trajectory / GP prior
    n_states: int = 20
    total_time: float = 10.0
    qc_lin: float = 0.5              # Q_C for v_x, v_y
    qc_rot: float = 0.5              # Q_C for omega
    # robot body: discs along the body x-axis
    body_length: float = 0.9
    body_width: float = 0.3
    disc_x: tuple = (-0.3, -0.1, 0.1, 0.3)
    disc_r: float = 0.17
    # obstacle cost
    eps: float = 0.35                # hinge safety margin (m)
    sigma_obs: float = 0.05
    n_interp: int = 2                # interpolated obstacle factors between states
    # fixed / soft constraints
    sigma_fix: float = 1e-3
    diff_drive: bool = False
    sigma_nonholo: float = 0.02      # std of the soft v_y = 0 factor when diff_drive
    # stochasticity
    n_dyn: float = 0.20              # uniform body-velocity noise, m/s
    n_dyn_rot: float = 0.10          # uniform yaw-rate noise, rad/s
    n_cam: float = 0.10              # pose measurement noise, m
    n_cam_rot: float = 0.05          # pose measurement noise, rad
    # solver
    batch_iters: int = 100
    step_iters: int = 8
    substeps: int = 10
    # world
    size: tuple = (12.0, 8.0)
    cell: float = 0.05
    n_obstacles: int = 9
    start: tuple = (1.0, 1.0, 0.0)               # x, y, heading
    goal: tuple = (11.0, 7.0, math.pi / 2)

    @property
    def dt(self):
        return self.total_time / self.n_states

    @property
    def Qc(self):
        return np.diag([self.qc_lin, self.qc_lin, self.qc_rot])


# =============================================================================
# SE(2) Lie group. Pose T = (x, y, th); tangent xi = (rho_x, rho_y, phi).
# =============================================================================
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ab(phi):
    """V(phi) = [[a, -b], [b, a]] with a = sin(phi)/phi, b = (1 - cos(phi))/phi."""
    if abs(phi) < 1e-6:
        return 1.0 - phi * phi / 6.0, phi / 2.0
    return math.sin(phi) / phi, (1.0 - math.cos(phi)) / phi


def se2_exp(xi):
    rx, ry, phi = xi
    a, b = _ab(phi)
    return np.array([a * rx - b * ry, b * rx + a * ry, wrap(phi)])


def se2_log(T):
    x, y, th = T
    th = wrap(th)
    a, b = _ab(th)
    det = a * a + b * b
    return np.array([(a * x + b * y) / det, (-b * x + a * y) / det, th])


def se2_compose(A, B):
    c, s = math.cos(A[2]), math.sin(A[2])
    return np.array([A[0] + c * B[0] - s * B[1], A[1] + s * B[0] + c * B[1], wrap(A[2] + B[2])])


def se2_inverse(A):
    c, s = math.cos(A[2]), math.sin(A[2])
    return np.array([-c * A[0] - s * A[1], s * A[0] - c * A[1], -A[2]])


def se2_between(A, B):
    """A^-1 B"""
    return se2_compose(se2_inverse(A), B)


def se2_jr(xi):
    """Right Jacobian:  Exp(xi + d) ~= Exp(xi) Exp(J_r(xi) d)."""
    rx, ry, phi = xi
    if abs(phi) < 1e-6:
        a, b = 1.0 - phi * phi / 6.0, phi / 2.0
        c13 = -ry / 2.0 + rx * phi / 6.0
        c23 = rx / 2.0 + ry * phi / 6.0
    else:
        s, c = math.sin(phi), math.cos(phi)
        a, b = s / phi, (1.0 - c) / phi
        c13 = (phi * rx - ry + ry * c - rx * s) / phi ** 2
        c23 = (rx + phi * ry - rx * c - ry * s) / phi ** 2
    return np.array([[a, b, c13], [-b, a, c23], [0.0, 0.0, 1.0]])


def se2_jr_inv(xi):
    """Closed-form inverse: J_r = [[A, c], [0, 1]] with A = [[a, b], [-b, a]]."""
    J = se2_jr(xi)
    a, b = J[0, 0], J[0, 1]
    k = 1.0 / (a * a + b * b)
    Ai = np.array([[k * a, -k * b], [k * b, k * a]])
    out = np.zeros((3, 3))
    out[:2, :2], out[:2, 2], out[2, 2] = Ai, -Ai @ J[:2, 2], 1.0
    return out


def se2_jl_inv(xi):
    """Left Jacobian inverse; J_l(xi) = J_r(-xi)."""
    return se2_jr_inv(-np.asarray(xi))


def retract(s, d):
    """State retraction: T Exp(d_T), w + d_w."""
    return np.concatenate([se2_compose(s[:3], se2_exp(d[:3])), s[3:] + d[3:]])


def local(s, s2):
    """Inverse of retract: tangent vector taking s to s2."""
    return np.concatenate([se2_log(se2_between(s[:3], s2[:3])), s2[3:] - s[3:]])


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


# =============================================================================
# World (occupancy grid -> signed distance field)
# =============================================================================
class SignedDistanceField:
    def __init__(self, occupancy, cell, origin=(0.0, 0.0)):
        self.occ, self.cell, self.origin = occupancy, cell, np.asarray(origin, float)
        self.field = (distance_transform_edt(~occupancy) - distance_transform_edt(occupancy)) * cell

    def query(self, p):
        """Bilinear SDF value and the exact gradient of that bilinear interpolant."""
        p = np.atleast_2d(p)
        h, w = self.field.shape
        c = np.clip((p[:, 0] - self.origin[0]) / self.cell, 0, w - 1 - 1e-9)
        r = np.clip((p[:, 1] - self.origin[1]) / self.cell, 0, h - 1 - 1e-9)
        c0, r0 = c.astype(int), r.astype(int)
        fc, fr = c - c0, r - r0
        F = self.field
        f00, f01, f10, f11 = F[r0, c0], F[r0, c0 + 1], F[r0 + 1, c0], F[r0 + 1, c0 + 1]
        d = (1 - fr) * ((1 - fc) * f00 + fc * f01) + fr * ((1 - fc) * f10 + fc * f11)
        gx = ((1 - fr) * (f01 - f00) + fr * (f11 - f10)) / self.cell
        gy = ((1 - fc) * (f10 - f00) + fc * (f11 - f01)) / self.cell
        return d, np.stack([gx, gy], axis=1)

    def distance(self, p):
        return self.query(p)[0]

    @property
    def extent(self):
        h, w = self.field.shape
        return (self.origin[0], self.origin[0] + w * self.cell, self.origin[1], self.origin[1] + h * self.cell)


def make_world(cfg, seed):
    rng = np.random.default_rng(seed)
    W, H = cfg.size
    nx, ny = int(round(W / cfg.cell)), int(round(H / cfg.cell))
    ys, xs = np.mgrid[0:ny, 0:nx] * cfg.cell
    occ = (xs < 0.1) | (xs > W - 0.1) | (ys < 0.1) | (ys > H - 0.1)
    start, goal = np.array(cfg.start[:2]), np.array(cfg.goal[:2])
    clear = cfg.body_length / 2 + cfg.eps + 0.7
    centers = [0.5 * (start + goal) + rng.uniform(-0.4, 0.4, 2)]
    placed = tries = 0
    while placed < cfg.n_obstacles and tries < 2000:
        tries += 1
        c = centers.pop() if centers else rng.uniform([1.0, 0.8], [W - 1.0, H - 0.8])
        if rng.random() < 0.5:
            r = rng.uniform(0.35, 0.8)
            mask, reach = (xs - c[0]) ** 2 + (ys - c[1]) ** 2 < r ** 2, r
        else:
            hw, hh = rng.uniform(0.25, 0.8, 2)
            mask, reach = (np.abs(xs - c[0]) < hw) & (np.abs(ys - c[1]) < hh), math.hypot(hw, hh)
        if min(np.linalg.norm(c - start), np.linalg.norm(c - goal)) < reach + clear:
            continue
        occ |= mask
        placed += 1
    return SignedDistanceField(occ, cfg.cell)


# =============================================================================
# GP prior on SE(2)
# =============================================================================
def gp_phi(dt):
    I = np.eye(3)
    return np.block([[I, dt * I], [np.zeros((3, 3)), I]])


def gp_Q(dt, Qc):
    return np.block([[dt ** 3 / 3 * Qc, dt ** 2 / 2 * Qc], [dt ** 2 / 2 * Qc, dt * Qc]])


def whitener(cov):
    return np.linalg.cholesky(np.linalg.inv(cov)).T


def gp_interp_matrices(tau, dt, Qc):
    """gamma(t_i + tau) = Lam gamma(t_i) + Psi gamma(t_i+1)."""
    Psi = gp_Q(tau, Qc) @ gp_phi(dt - tau).T @ np.linalg.inv(gp_Q(dt, Qc))
    Lam = gp_phi(tau) - Psi @ gp_phi(dt)
    return Lam, Psi


def gp_interpolate(a, b, Lam, Psi):
    """Pose and body velocity at t_i + tau from states a = x_i, b = x_i+1."""
    xi = se2_log(se2_between(a[:3], b[:3]))
    g0 = np.concatenate([np.zeros(3), a[3:]])
    g1 = np.concatenate([xi, se2_jr_inv(xi) @ b[3:]])
    g = Lam @ g0 + Psi @ g1
    return se2_compose(a[:3], se2_exp(g[:3])), se2_jr_inv(g[:3]) @ g[3:]


class Interpolator:
    """Caches Lam/Psi per tau for a fixed dt."""

    def __init__(self, cfg):
        self.cfg, self.cache = cfg, {}

    def __call__(self, x, i, tau):
        key = round(tau, 9)
        if key not in self.cache:
            self.cache[key] = gp_interp_matrices(tau, self.cfg.dt, self.cfg.Qc)
        return gp_interpolate(x[i], x[i + 1], *self.cache[key])


# =============================================================================
# Factors: evaluate(*states) -> (whitened error, [d error / d tangent of each state])
# =============================================================================
def numeric_jacobian(fun, states, h=1e-6):
    """Central differences of fun(*states) in each state's tangent space."""
    Js = []
    for k in range(len(states)):
        cols = []
        for j in range(D):
            d = np.zeros(D); d[j] = h
            sp_, sm = list(states), list(states)
            sp_[k], sm[k] = retract(states[k], d), retract(states[k], -d)
            cols.append((fun(*sp_) - fun(*sm)) / (2 * h))
        Js.append(np.stack(cols, axis=1))
    return Js


class GPPriorFactor:
    """Constant-body-velocity GP prior between x_i and x_i+1 (Lie-group version)."""

    def __init__(self, i, cfg):
        self.states = (i, i + 1)
        self.dt = cfg.dt
        self.W = whitener(gp_Q(cfg.dt, cfg.Qc))

    def error(self, a, b):
        xi = se2_log(se2_between(a[:3], b[:3]))
        return np.concatenate([self.dt * a[3:] - xi, a[3:] - se2_jr_inv(xi) @ b[3:]])

    def evaluate(self, a, b):
        dt = self.dt
        xi = se2_log(se2_between(a[:3], b[:3]))
        Jri = se2_jr_inv(xi)
        wb = b[3:]
        e = np.concatenate([dt * a[3:] - xi, a[3:] - Jri @ wb])
        dxi_da = -se2_jl_inv(xi)          # d Log(Exp(-d) T_i^-1 T_i+1) / d d
        dxi_db = Jri                       # d Log(T_i^-1 T_i+1 Exp(d)) / d d
        h = 1e-6                           # d (J_r(xi)^-1 w_b) / d xi, numerically
        dg = np.stack([(se2_jr_inv(xi + h * u) @ wb - se2_jr_inv(xi - h * u) @ wb) / (2 * h)
                       for u in np.eye(3)], axis=1)
        Ja, Jb = np.zeros((6, D)), np.zeros((6, D))
        Ja[:3, :3], Ja[3:, :3] = -dxi_da, -dg @ dxi_da
        Ja[:3, 3:], Ja[3:, 3:] = dt * np.eye(3), np.eye(3)
        Jb[:3, :3], Jb[3:, :3] = -dxi_db, -dg @ dxi_db
        Jb[3:, 3:] = -Jri
        return self.W @ e, [self.W @ Ja, self.W @ Jb]


class PosePriorFactor:
    """Log(z^-1 T_i) ~ N(0, diag(sigmas)):  start / goal / pose measurement / f_curr."""

    def __init__(self, i, z, sigmas, kind="prior"):
        self.states, self.z, self.kind = (i,), np.asarray(z, float), kind
        self.w = 1.0 / np.broadcast_to(np.asarray(sigmas, float), (3,))

    def error(self, a):
        return se2_log(se2_between(self.z, a[:3]))

    def evaluate(self, a):
        e = self.error(a)
        J = np.zeros((3, D)); J[:, :3] = se2_jr_inv(e)
        return self.w * e, [self.w[:, None] * J]


class VelocityPriorFactor:
    """Gaussian prior on selected body-velocity components (start/goal at rest, v_y = 0, ...)."""

    def __init__(self, i, mean, sigma, dims=(0, 1, 2)):
        self.states, self.dims = (i,), [3 + d for d in dims]
        self.mean = np.asarray(mean, float)
        self.w = 1.0 / np.broadcast_to(np.asarray(sigma, float), (len(self.dims),))
        J = np.zeros((len(self.dims), D)); J[np.arange(len(self.dims)), self.dims] = self.w
        self.J = [J]

    def error(self, a):
        return a[self.dims] - self.mean

    def evaluate(self, a):
        return self.w * self.error(a), self.J


class BodyCollision:
    """Hinge cost of every body disc against the SDF, with d error / d(right perturbation of T)."""

    def __init__(self, sdf, cfg):
        self.sdf, self.cfg = sdf, cfg
        self.c = np.array([[x, 0.0] for x in cfg.disc_x])      # disc centres in body frame
        self.perp = np.stack([-self.c[:, 1], self.c[:, 0]], 1)  # d(R(dth) c)/d dth at 0

    def __call__(self, T, need_jac=True):
        cfg, R = self.cfg, rot(T[2])
        P = T[:2] + self.c @ R.T
        d, g = self.sdf.query(P)
        d = d - cfg.disc_r
        active = d < cfg.eps
        e = np.where(active, (cfg.eps - d) / cfg.sigma_obs, 0.0)
        if not need_jac:
            return e, None, active.any()
        J = np.zeros((len(P), 3))
        for k in np.flatnonzero(active):
            dp = np.column_stack([R, R @ self.perp[k]])         # 2x3: dp / d(dx, dy, dth)
            J[k] = -(g[k] @ dp) / cfg.sigma_obs
        return e, J, active.any()


class ObstacleFactor:
    def __init__(self, i, body):
        self.states, self.body = (i,), body

    def error(self, a):
        return self.body(a[:3], need_jac=False)[0]

    def evaluate(self, a):
        e, JT, _ = self.body(a[:3])
        J = np.zeros((len(e), D)); J[:, :3] = JT
        return e, [J]


class InterpObstacleFactor:
    """Obstacle cost at t_i + tau, through the GP interpolation of x_i, x_i+1."""

    def __init__(self, i, tau, body, cfg):
        self.states, self.body = (i, i + 1), body
        self.Lam, self.Psi = gp_interp_matrices(tau, cfg.dt, cfg.Qc)

    def pose(self, a, b):
        return gp_interpolate(a, b, self.Lam, self.Psi)[0]

    def error(self, a, b):
        return self.body(self.pose(a, b), need_jac=False)[0]

    def evaluate(self, a, b):
        T = self.pose(a, b)
        e, JT, any_active = self.body(T)
        if not any_active:                              # hinge inactive: skip the Jacobian work
            z = np.zeros((len(e), D))
            return e, [z, z]
        # d (local perturbation of T(tau)) / d (tangent of a, b), numerically
        Da, Db = numeric_jacobian(lambda a_, b_: se2_log(se2_between(T, self.pose(a_, b_))), [a, b], h=1e-5)
        return e, [JT @ Da, JT @ Db]


# =============================================================================
# Sparse factor graph on the manifold + Levenberg-Marquardt
# =============================================================================
class FactorGraph:
    def __init__(self):
        self.factors = []

    def add(self, f):
        self.factors.append(f); return f

    def extend(self, fs):
        self.factors.extend(fs)

    def remove(self, f):
        self.factors.remove(f)

    def cost(self, x):
        total = 0.0
        for f in self.factors:
            e = f.error(*[x[k] for k in f.states])
            w = getattr(f, "w", None)
            if isinstance(f, GPPriorFactor):
                e = f.W @ e
            elif w is not None:
                e = w * e
            total += float(e @ e)
        return 0.5 * total

    def linearize(self, x, colmap):
        rows, cols, vals, res, r0 = [], [], [], [], 0
        for f in self.factors:
            e, Js = f.evaluate(*[x[k] for k in f.states])
            m = len(e)
            res.append(e)
            for k, J in zip(f.states, Js):
                c0 = D * colmap[k]
                rr, cc = np.meshgrid(np.arange(r0, r0 + m), np.arange(c0, c0 + D), indexing="ij")
                rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(np.asarray(J).ravel())
            r0 += m
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(r0, D * len(colmap)))
        return np.concatenate(res), J

    def optimize(self, x0, max_iters=50, lam=1e-4, rel_tol=1e-6):
        x = x0.copy()
        states = sorted({k for f in self.factors for k in f.states})
        colmap = {k: j for j, k in enumerate(states)}
        cost = self.cost(x)
        for _ in range(max_iters):
            r, J = self.linearize(x, colmap)
            H, g = (J.T @ J).tocsc(), J.T @ r
            Dg = sp.diags(H.diagonal() + 1e-9)
            while True:
                dx = spsolve((H + lam * Dg).tocsc(), -g).reshape(-1, D)
                x_new = x.copy()
                for k, j in colmap.items():
                    x_new[k] = retract(x[k], dx[j])
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
# Graph builders
# =============================================================================
def gp_chain(lo, hi, cfg):
    return [GPPriorFactor(i, cfg) for i in range(lo, hi)]


def obstacle_factors(lo, hi, body, cfg):
    fs = [ObstacleFactor(i, body) for i in range(lo, hi + 1)]
    for i in range(lo, hi):
        for j in range(1, cfg.n_interp + 1):
            fs.append(InterpObstacleFactor(i, j * cfg.dt / (cfg.n_interp + 1), body, cfg))
    return fs


def nonholonomic_factors(lo, hi, cfg):
    if not cfg.diff_drive:
        return []
    return [VelocityPriorFactor(i, [0.0], cfg.sigma_nonholo, dims=(1,)) for i in range(lo, hi + 1)]


def start_factors(cfg):
    return [PosePriorFactor(0, cfg.start, cfg.sigma_fix, kind="start"),
            VelocityPriorFactor(0, np.zeros(3), cfg.sigma_fix)]


def goal_factors(cfg):
    N = cfg.n_states
    return [PosePriorFactor(N, cfg.goal, cfg.sigma_fix, kind="goal"),
            VelocityPriorFactor(N, np.zeros(3), cfg.sigma_fix)]


def initial_trajectory(cfg, bend=0.3):
    """Bent straight line, heading interpolated start -> goal, velocities from finite differences."""
    N = cfg.n_states
    s, g = np.array(cfg.start[:2]), np.array(cfg.goal[:2])
    t = np.linspace(0, 1, N + 1)
    normal = np.array([-(g - s)[1], (g - s)[0]]) / np.linalg.norm(g - s)
    pos = s + t[:, None] * (g - s) + bend * np.sin(np.pi * t)[:, None] * normal
    dth = wrap(cfg.goal[2] - cfg.start[2])
    poses = [np.array([*pos[k], wrap(cfg.start[2] + t[k] * dth)]) for k in range(N + 1)]
    x = np.zeros((N + 1, D))
    for k in range(N + 1):
        x[k, :3] = poses[k]
        if k < N:
            x[k, 3:] = se2_log(se2_between(poses[k], poses[k + 1])) / cfg.dt
    x[N, 3:] = x[N - 1, 3:]
    return x


def disc_centres(T, cfg):
    return T[:2] + np.column_stack([cfg.disc_x, np.zeros(len(cfg.disc_x))]) @ rot(T[2]).T


def segment_collision_free(x, i, sdf, cfg, interp, n=10):
    for k in range(n + 1):
        T = interp(x, i, k * cfg.dt / n)[0] if k < n else x[i + 1, :3]
        if np.any(sdf.distance(disc_centres(T, cfg)) < cfg.disc_r):
            return False
    return True


def batch_plan(graph, sdf, cfg, interp, bends=(0.3, -0.3, 1.5, -1.5, 3.0, -3.0)):
    """First solve (the GPMP2 special case). Local optimiser, so try a few initialisations."""
    best = None
    for b in bends:
        x, c = graph.optimize(initial_trajectory(cfg, b), cfg.batch_iters)
        free = all(segment_collision_free(x, i, sdf, cfg, interp) for i in range(cfg.n_states))
        if best is None or (not free, c) < best[0]:
            best = ((not free, c), x)
        if free and b == bends[0]:
            break
    return best[1]


# =============================================================================
# Robot simulation (exact integration on SE(2))
# =============================================================================
@dataclass
class Noise:
    dyn: np.ndarray   # (N, 3) body-velocity noise per segment
    cam: np.ndarray   # (N, 3) tangent-space measurement noise

    @staticmethod
    def sample(cfg, seed):
        rng = np.random.default_rng(10_000 + seed)
        N = cfg.n_states
        lin, r = cfg.n_dyn, cfg.n_dyn_rot
        dyn = rng.uniform(-1, 1, (N, 3)) * [lin, 0.0 if cfg.diff_drive else lin, r]
        cam = rng.normal(0, 1, (N, 3)) * [cfg.n_cam, cfg.n_cam, cfg.n_cam_rot]
        return Noise(dyn, cam)


def body_hits(T, sdf, cfg):
    return bool(np.any(sdf.distance(disc_centres(T, cfg)) < cfg.disc_r))


def execute_segment(x, i, T0, w_noise, sdf, cfg, interp):
    """Track the planned body-velocity profile of segment i, T <- T Exp(w h)."""
    h = cfg.dt / cfg.substeps
    T, path, hit = T0.copy(), [T0.copy()], False
    for k in range(cfg.substeps):
        _, w = interp(x, i, (k + 0.5) * h)
        T = se2_compose(T, se2_exp((w + w_noise) * h))
        path.append(T.copy())
        hit = hit or body_hits(T, sdf, cfg)
    return np.array(path), hit


@dataclass
class Result:
    method: str
    status: str = "running"
    truth: list = field(default_factory=list)          # dense executed poses
    truth_states: list = field(default_factory=list)   # true pose at each t_i
    meas: list = field(default_factory=list)
    estimate: np.ndarray | None = None
    first_plan: np.ndarray | None = None
    frames: list = field(default_factory=list)
    step_times: list = field(default_factory=list)

    @property
    def success(self):
        return self.status == "success"

    def goal_error(self, cfg):
        T = self.truth_states[-1]
        return float(np.linalg.norm(T[:2] - np.array(cfg.goal[:2]))), abs(wrap(T[2] - cfg.goal[2]))

    def estimation_error(self):
        if self.estimate is None:
            return float("nan"), float("nan")
        T = np.array(self.truth_states)
        E = self.estimate[:len(T)]
        return (float(np.mean(np.linalg.norm(E[:, :2] - T[:, :2], axis=1))),
                float(np.mean(np.abs([wrap(a) for a in E[:, 2] - T[:, 2]]))))


def measure(T, n):
    """Pose measurement z = T Exp(n)."""
    return se2_compose(T, se2_exp(n))


# =============================================================================
# Methods
# =============================================================================
def run_steap(sdf, cfg, noise, record=True):
    """Algorithm 1 of the STEAP paper on SE(2)."""
    N, interp, body = cfg.n_states, Interpolator(cfg), BodyCollision(sdf, cfg)
    res = Result("STEAP")
    graph = FactorGraph()
    f_goal = goal_factors(cfg)
    graph.extend(gp_chain(0, N, cfg) + obstacle_factors(0, N, body, cfg)
                 + nonholonomic_factors(0, N, cfg) + start_factors(cfg) + f_goal)

    t0 = time.perf_counter()
    x = batch_plan(graph, sdf, cfg, interp)
    res.step_times.append(time.perf_counter() - t0)
    res.first_plan = x.copy()

    T = np.array(cfg.start, float)
    res.truth, res.truth_states = [T.copy()], [T.copy()]
    if record:
        res.frames.append(dict(i=0, x=x.copy(), truth=np.array(res.truth), meas=[]))
    for i in range(N):
        if not segment_collision_free(x, i, sdf, cfg, interp):
            res.status = "plan-blocked"; break
        path, hit = execute_segment(x, i, T, noise.dyn[i], sdf, cfg, interp)
        T = path[-1]
        res.truth.extend(path[1:]); res.truth_states.append(T.copy())
        if hit:
            res.status = "collision"; break
        z = measure(T, noise.cam[i])                                       # localize()
        res.meas.append(z)
        graph.add(PosePriorFactor(i + 1, z, [cfg.n_cam, cfg.n_cam, cfg.n_cam_rot], kind="meas"))
        if i + 1 == N:
            graph.remove(f_goal[0])        # last update: estimate the final pose from data
        t0 = time.perf_counter()
        x, _ = graph.optimize(x, cfg.step_iters)                           # incrementalInference()
        res.step_times.append(time.perf_counter() - t0)
        if record:
            res.frames.append(dict(i=i + 1, x=x.copy(), truth=np.array(res.truth), meas=list(res.meas)))
    else:
        res.status = "success"
    res.estimate = x
    return res


def run_slap(sdf, cfg, noise, record=False):
    """Estimation graph (past) then planning graph (future), solved in sequence."""
    N, interp, body = cfg.n_states, Interpolator(cfg), BodyCollision(sdf, cfg)
    res = Result("SLAP")
    g = FactorGraph()
    g.extend(gp_chain(0, N, cfg) + obstacle_factors(0, N, body, cfg) + nonholonomic_factors(0, N, cfg)
             + start_factors(cfg) + goal_factors(cfg))
    x = batch_plan(g, sdf, cfg, interp)
    res.first_plan = x.copy()
    est_factors = start_factors(cfg)
    x_est = x.copy()
    sig_meas = [cfg.n_cam, cfg.n_cam, cfg.n_cam_rot]

    T = np.array(cfg.start, float)
    res.truth, res.truth_states = [T.copy()], [T.copy()]
    for i in range(N):
        if not segment_collision_free(x, i, sdf, cfg, interp):
            res.status = "plan-blocked"; break
        path, hit = execute_segment(x, i, T, noise.dyn[i], sdf, cfg, interp)
        T = path[-1]
        res.truth.extend(path[1:]); res.truth_states.append(T.copy())
        if hit:
            res.status = "collision"; break
        z = measure(T, noise.cam[i])
        res.meas.append(z)
        t0 = time.perf_counter()
        est_factors += [GPPriorFactor(i, cfg), PosePriorFactor(i + 1, z, sig_meas)]
        eg = FactorGraph(); eg.extend(est_factors)
        x_est, _ = eg.optimize(np.vstack([x_est[:i + 1], x[i + 1:]]), cfg.step_iters)
        if i + 1 < N:
            pg = FactorGraph()
            pg.extend(gp_chain(i + 1, N, cfg) + obstacle_factors(i + 1, N, body, cfg)
                      + nonholonomic_factors(i + 1, N, cfg) + goal_factors(cfg)
                      + [PosePriorFactor(i + 1, x_est[i + 1, :3], cfg.sigma_fix),
                         VelocityPriorFactor(i + 1, x_est[i + 1, 3:], cfg.sigma_fix)])
            seed = x.copy(); seed[i + 1] = x_est[i + 1]
            x, _ = pg.optimize(seed, cfg.step_iters)
        res.step_times.append(time.perf_counter() - t0)
    else:
        res.status = "success"
    res.estimate = x_est
    return res


def run_open_loop(sdf, cfg, noise, record=False):
    N, interp, body = cfg.n_states, Interpolator(cfg), BodyCollision(sdf, cfg)
    res = Result("OL")
    g = FactorGraph()
    g.extend(gp_chain(0, N, cfg) + obstacle_factors(0, N, body, cfg) + nonholonomic_factors(0, N, cfg)
             + start_factors(cfg) + goal_factors(cfg))
    x = batch_plan(g, sdf, cfg, interp)
    res.first_plan = x
    T = np.array(cfg.start, float)
    res.truth, res.truth_states = [T.copy()], [T.copy()]
    for i in range(N):
        path, hit = execute_segment(x, i, T, noise.dyn[i], sdf, cfg, interp)
        T = path[-1]
        res.truth.extend(path[1:]); res.truth_states.append(T.copy())
        if hit:
            res.status = "collision"; break
    else:
        res.status = "success"
    return res


# =============================================================================
# Jacobian self-test
# =============================================================================
def check_jacobians(cfg, seed=0):
    rng = np.random.default_rng(seed)
    sdf = make_world(cfg, 1)
    body = BodyCollision(sdf, cfg)

    def rand_state():
        return np.array([*rng.uniform([2, 2], [10, 6]), rng.uniform(-3, 3), *rng.normal(0, 1, 3)])

    a = rand_state()
    b = retract(a, np.concatenate([rng.normal(0, 0.4, 3), rng.normal(0, 0.3, 3)]))
    # put the pair right next to an obstacle so the hinge is active
    occ = np.argwhere(sdf.occ[20:-20, 20:-20])[len(np.argwhere(sdf.occ[20:-20, 20:-20])) // 2] + 20
    near = np.array([occ[1], occ[0]]) * cfg.cell + [0.0, 0.0]
    a_near, b_near = a.copy(), b.copy()
    a_near[:2] = near + [0.45, 0.1]
    b_near = retract(a_near, np.array([0.3, 0.05, 0.4, 0.1, 0.0, 0.2]))

    print("SE(2) identities")
    xi = rng.normal(0, 0.8, 3)
    print(f"  Log(Exp(xi)) - xi         {np.abs(se2_log(se2_exp(xi)) - xi).max():.2e}")
    d = 1e-6 * rng.normal(size=3)
    lhs = se2_log(se2_between(se2_exp(xi), se2_exp(xi + d)))
    print(f"  right Jacobian rel. err   {np.linalg.norm(lhs - se2_jr(xi) @ d) / np.linalg.norm(d):.2e}")
    lhs = se2_log(se2_compose(se2_exp(xi + d), se2_inverse(se2_exp(xi))))
    print(f"  left  Jacobian rel. err   {np.linalg.norm(lhs - se2_jr(-xi) @ d) / np.linalg.norm(d):.2e}")

    tests = [("GP prior", GPPriorFactor(0, cfg), [a, b]),
             ("pose prior", PosePriorFactor(0, measure(a[:3], [0.3, -0.2, 0.5]), 0.1), [a]),
             ("velocity prior", VelocityPriorFactor(0, [0.5, 0, 0.1], 0.1), [a]),
             ("obstacle", ObstacleFactor(0, body), [a_near]),
             ("interp obstacle", InterpObstacleFactor(0, cfg.dt / 3, body, cfg), [a_near, b_near])]
    print("factor Jacobians vs. central differences (max abs diff / max abs entry)")
    for name, f, xs in tests:
        e, Js = f.evaluate(*xs)

        def whitened(*s, f=f):
            return f.evaluate(*s)[0]
        Jn = numeric_jacobian(whitened, xs, h=1e-6)
        err = max(np.abs(J - Jn_).max() for J, Jn_ in zip(Js, Jn))
        scale = max(np.abs(Jn_).max() for Jn_ in Jn) + 1e-12
        print(f"  {name:<16} |e| = {np.linalg.norm(e):8.3f}   {err:.2e} / {scale:.2e}")


# =============================================================================
# Plotting
# =============================================================================
COLORS = dict(truth="#17905a", est="#cf3434", plan="#2160c4", meas="#d18f00")


def dense_poses(x, cfg, interp, lo, hi, n=12):
    P = [x[lo, :3]]
    for i in range(lo, hi):
        for k in range(1, n):
            P.append(interp(x, i, k * cfg.dt / n)[0])
        P.append(x[i + 1, :3])
    return np.array(P)


def footprint(T, cfg):
    L, W = cfg.body_length / 2, cfg.body_width / 2
    corners = np.array([[L, W], [-L, W], [-L, -W], [L, -W], [L, W]])
    return T[:2] + corners @ rot(T[2]).T


def draw_world(ax, sdf, cfg):
    ax.imshow(np.clip(sdf.field, -0.5, 2.0), origin="lower", extent=sdf.extent, cmap="Greys_r",
              alpha=0.35, vmin=-0.5, vmax=2.0)
    xs = np.linspace(*sdf.extent[:2], sdf.field.shape[1])
    ys = np.linspace(*sdf.extent[2:], sdf.field.shape[0])
    ax.contourf(xs, ys, sdf.occ.astype(float), levels=[0.5, 1.5], colors=["#33414f"])
    for T, ls in [(cfg.start, "-"), (cfg.goal, "--")]:
        F = footprint(np.array(T), cfg)
        ax.plot(F[:, 0], F[:, 1], "k", ls=ls, lw=1.2)
    ax.set_xlim(sdf.extent[:2]); ax.set_ylim(sdf.extent[2:])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def heading_quiver(ax, P, color, scale=0.35, **kw):
    return ax.quiver(P[:, 0], P[:, 1], np.cos(P[:, 2]), np.sin(P[:, 2]), color=color,
                     angles="xy", scale_units="xy", scale=1 / scale, width=0.004, **kw)


def animate_steap(res, sdf, cfg, save=None, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon

    interp = Interpolator(cfg)
    fig, ax = plt.subplots(figsize=(9, 6.4))
    draw_world(ax, sdf, cfg)
    (l_truth,) = ax.plot([], [], color=COLORS["truth"], lw=2.2, label="ground truth")
    (l_est,) = ax.plot([], [], color=COLORS["est"], lw=2.4, label="estimated past")
    (l_plan,) = ax.plot([], [], color=COLORS["plan"], lw=2.2, label="planned future")
    (l_meas,) = ax.plot([], [], "x", color=COLORS["meas"], ms=7, mew=2, label="pose measurements")
    robot = Polygon(footprint(np.array(cfg.start), cfg), closed=True, fc=COLORS["truth"], ec="white", zorder=6)
    ax.add_patch(robot)
    ghosts = []
    quivers = []
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    title = ax.set_title("")

    def update(k):
        f = res.frames[k]
        i, x, T = f["i"], f["x"], f["truth"]
        for artist in ghosts + quivers:
            artist.remove()
        ghosts.clear(); quivers.clear()
        l_truth.set_data(T[:, 0], T[:, 1])
        est = dense_poses(x, cfg, interp, 0, i) if i > 0 else np.empty((0, 3))
        l_est.set_data(est[:, 0], est[:, 1])
        plan = dense_poses(x, cfg, interp, i, cfg.n_states)
        l_plan.set_data(plan[:, 0], plan[:, 1])
        for j in range(i + 1, cfg.n_states + 1, 2):          # planned footprints
            F = footprint(x[j, :3], cfg)
            ghosts.append(ax.plot(F[:, 0], F[:, 1], color=COLORS["plan"], lw=0.8, alpha=0.45)[0])
        if i > 0:
            quivers.append(heading_quiver(ax, x[1:i + 1, :3], COLORS["est"]))
        if f["meas"]:
            M = np.array(f["meas"])
            l_meas.set_data(M[:, 0], M[:, 1])
            quivers.append(heading_quiver(ax, M, COLORS["meas"], scale=0.25, alpha=0.7))
        else:
            l_meas.set_data([], [])
        robot.set_xy(footprint(T[-1], cfg))
        last = k == len(res.frames) - 1
        title.set_text(f"STEAP on SE(2)  step {i}/{cfg.n_states}" + (f"   [{res.status}]" if last else ""))

    anim = FuncAnimation(fig, update, frames=len(res.frames), interval=400, repeat_delay=1500)
    if save:
        anim.save(save, writer="pillow" if save.endswith(".gif") else None, dpi=90)
        print(f"saved animation -> {save}")
    if show:
        plt.show()
    return anim


def plot_comparison(results, sdf, cfg, save=None):
    import matplotlib.pyplot as plt
    interp = Interpolator(cfg)
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.3))
    for ax, r in zip(np.atleast_1d(axes), results):
        draw_world(ax, sdf, cfg)
        fp = dense_poses(r.first_plan, cfg, interp, 0, cfg.n_states)
        ax.plot(fp[:, 0], fp[:, 1], "--", color=COLORS["plan"], lw=1, alpha=0.6, label="initial plan")
        T = np.array(r.truth)
        ax.plot(T[:, 0], T[:, 1], color=COLORS["truth"], lw=2, label="ground truth")
        for P in r.truth_states[::3]:
            F = footprint(P, cfg); ax.plot(F[:, 0], F[:, 1], color=COLORS["truth"], lw=0.8)
        if r.estimate is not None:
            n = len(r.truth_states) - 1
            est = dense_poses(r.estimate, cfg, interp, 0, n)
            ax.plot(est[:, 0], est[:, 1], color=COLORS["est"], lw=2, label="final estimate")
        if r.meas:
            M = np.array(r.meas); ax.plot(M[:, 0], M[:, 1], "x", color=COLORS["meas"], ms=5)
        if r.status == "collision":
            F = footprint(T[-1], cfg); ax.fill(F[:, 0], F[:, 1], color="red", alpha=0.7)
        gt, gr = r.goal_error(cfg)
        sub = f"goal {gt:.2f} m / {math.degrees(gr):.1f}°"
        if r.estimate is not None:
            et, er = r.estimation_error()
            sub += f" · est {et:.3f} m / {math.degrees(er):.1f}°"
        ax.set_title(f"{r.method}: {r.status}\n{sub}", fontsize=10)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
        print(f"saved comparison -> {save}")


# =============================================================================
# Benchmark
# =============================================================================
def benchmark(cfg, n_seeds):
    methods = [("OL", run_open_loop), ("SLAP", run_slap), ("STEAP", run_steap)]
    S = {m: dict(succ=[], gt=[], gr=[], et=[], er=[], t=[]) for m, _ in methods}
    for seed in range(n_seeds):
        sdf, noise = make_world(cfg, seed), Noise.sample(cfg, seed)
        for name, fn in methods:
            r = fn(sdf, cfg, noise, record=False)
            s = S[name]
            s["succ"].append(r.success)
            s["t"].extend(r.step_times[1:] if name == "STEAP" else r.step_times)
            if r.success:
                gt, gr = r.goal_error(cfg); s["gt"].append(gt); s["gr"].append(gr)
                if r.estimate is not None:
                    et, er = r.estimation_error(); s["et"].append(et); s["er"].append(er)
        print(f"\r  world {seed + 1}/{n_seeds}", end="", flush=True)
    print(f"\n\nSE(2), {'diff-drive' if cfg.diff_drive else 'holonomic'}, n_dyn={cfg.n_dyn}, "
          f"n_cam={cfg.n_cam}, {cfg.n_states} states, {n_seeds} worlds")
    print(f"{'method':<7}{'success':>9}{'goal m':>9}{'goal rad':>10}{'est m':>9}{'est rad':>9}{'ms/step':>9}")
    f = lambda v: f"{np.mean(v):.3f}" if v else "  -  "
    for name, _ in methods:
        s = S[name]
        t = f"{1e3 * np.mean(s['t']):.0f}" if s["t"] else "  -  "
        print(f"{name:<7}{np.mean(s['succ']):>9.2f}{f(s['gt']):>9}{f(s['gr']):>10}{f(s['et']):>9}{f(s['er']):>9}{t:>9}")
    print("(errors averaged over successful runs only, as in the paper)")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--states", type=int, default=Config.n_states)
    ap.add_argument("--n-dyn", type=float, default=Config.n_dyn)
    ap.add_argument("--n-cam", type=float, default=Config.n_cam)
    ap.add_argument("--diff-drive", action="store_true", help="soft non-holonomic constraint v_y = 0")
    ap.add_argument("--save", help="save the STEAP animation (.gif/.mp4) and a _compare.png")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--benchmark", type=int, metavar="N")
    ap.add_argument("--check-jacobians", action="store_true")
    args = ap.parse_args()

    cfg = Config(n_states=args.states, n_dyn=args.n_dyn, n_cam=args.n_cam, diff_drive=args.diff_drive)
    if args.check_jacobians:
        check_jacobians(cfg); return
    if args.benchmark:
        benchmark(cfg, args.benchmark); return
    if args.no_show:
        import matplotlib; matplotlib.use("Agg")

    sdf, noise = make_world(cfg, args.seed), Noise.sample(cfg, args.seed)
    results = [run_open_loop(sdf, cfg, noise), run_slap(sdf, cfg, noise), run_steap(sdf, cfg, noise)]
    for r in results:
        gt, gr = r.goal_error(cfg)
        line = f"{r.method:<6} {r.status:<13} goal err {gt:.3f} m / {math.degrees(gr):5.1f} deg"
        if r.estimate is not None:
            et, er = r.estimation_error()
            line += f"   est err {et:.3f} m / {math.degrees(er):4.1f} deg"
        if r.method == "STEAP":
            line += f"   batch {1e3 * r.step_times[0]:.0f} ms, update {1e3 * np.mean(r.step_times[1:] or [0]):.0f} ms/step"
        print(line)
    r = results[-1]
    if r.meas:
        raw = [local(np.r_[z, 0, 0, 0], np.r_[t, 0, 0, 0])[:3] for z, t in zip(r.meas, r.truth_states[1:])]
        raw = np.array(raw)
        print(f"raw measurement error {np.mean(np.linalg.norm(raw[:, :2], axis=1)):.3f} m / "
              f"{math.degrees(np.mean(np.abs(raw[:, 2]))):.1f} deg")

    plot_comparison(results, sdf, cfg, save=args.save.rsplit(".", 1)[0] + "_compare.png" if args.save else None)
    animate_steap(r, sdf, cfg, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()

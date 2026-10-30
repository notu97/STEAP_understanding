"""
STELA demo: Simultaneous Trajectory Estimation & Local Adaptation
for a simple 2D car-like robot (kinematic bicycle, second order in speed).

Pipeline (same 8 steps as the slides):
  1. Plan offline with a kinodynamic RRT (an SBMP) using the robot's model.
  2. Discretize the plan into nodes (states) and edges (control, duration).
  3. Build a factor graph over a sliding window of nodes.
  4. Execute the current control on the "real" robot (model gap + noise + ramp).
  5. Estimate the past: add noisy pose observations and re-solve.
  6. Adapt the future: the same solve re-optimizes the next controls/durations.
  7. Slide the window when the current edge's duration has elapsed.
  8. Repeat until the goal.

The factor graph is solved as nonlinear least squares with scipy (a batch
re-solve of the small window each tick). The paper uses GTSAM + iSAM2 for
incremental updates; the factors here are the same ideas, written by hand.

Run:  python stela_demo.py            -> prints results, saves stela_demo.png
      python stela_demo.py --animate  -> also plays the run step by step
      python stela_demo.py --gif      -> also saves the run as stela_demo.gif
"""
import argparse
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# ----------------------------------------------------------------------------
# World and robot
# ----------------------------------------------------------------------------
L = 0.30                       # wheelbase [m]
R_ROBOT = 0.15                 # robot radius for collisions [m]
A_LIM, D_LIM = 1.0, 0.45       # controller limits: |acceleration| [m/s^2], |steering| [rad]
A_PLAN, D_PLAN = 0.7, 0.35     # the planner stays inside them, leaving room to correct
V_MAX = 1.0                    # max speed [m/s]
DT_EDGE = 0.1                  # plan edge duration = model step [s]

WORLD = (6.0, 4.0)
PLAN_MARGIN = 0.12             # extra clearance the planner keeps [m]
OBSTACLES = [(1.8, 1.0, 0.45), (2.4, 2.9, 0.5), (3.6, 1.6, 0.5), (4.8, 2.9, 0.4), (1.0, 2.8, 0.35)]
RAMP = (1.9, 1.65, 0.8, 0.7)    # x, y, w, h : unmodeled region (slows + pushes sideways)
START = np.array([0.4, 0.5, 0.6])
GOAL = np.array([5.5, 3.6])
GOAL_R = 0.25


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def obstacle_clearance(x, y):
    """Signed distance from the robot's edge to the nearest obstacle or wall."""
    walls = min(x, y, WORLD[0] - x, WORLD[1] - y) - R_ROBOT
    return min([walls] + [np.hypot(x - ox, y - oy) - r - R_ROBOT for ox, oy, r in OBSTACLES])


def in_collision(x, y, margin=0.0):
    return obstacle_clearance(x, y) < margin


def in_ramp(x, y):
    rx, ry, rw, rh = RAMP
    return rx <= x <= rx + rw and ry <= y <= ry + rh


# ----------------------------------------------------------------------------
# SE(2) helpers (Lie group operations used by the integration factor)
# ----------------------------------------------------------------------------
def se2_exp(vx, vy, w):
    """Exp map: constant body twist applied for unit time -> relative pose."""
    if abs(w) < 1e-6:
        return np.array([vx, vy, w])
    s, c = np.sin(w), np.cos(w)
    return np.array([(s * vx - (1 - c) * vy) / w, ((1 - c) * vx + s * vy) / w, w])


def se2_compose(q, d):
    c, s = np.cos(q[2]), np.sin(q[2])
    return np.array([q[0] + c * d[0] - s * d[1], q[1] + s * d[0] + c * d[1], wrap(q[2] + d[2])])


def se2_between(a, b):
    """a^-1 * b expressed in a's frame (small-angle Log approximation)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


# ----------------------------------------------------------------------------
# The planning model  f_hat:  state (q, w) with w = (v, omega), control u = (a, delta)
# ----------------------------------------------------------------------------
def model_step(q, w, u, dt):
    """One discrete step of the approximate model (same equations as the factors)."""
    q_next = se2_compose(q, se2_exp(w[0] * dt, 0.0, w[1] * dt))      # integration
    v_next = np.clip(w[0] + u[0] * dt, 0.0, V_MAX)                    # dynamics
    w_next = np.array([v_next, v_next * np.tan(u[1]) / L])
    return q_next, w_next


# ----------------------------------------------------------------------------
# Step 1-2: kinodynamic RRT, then discretize into nodes and edges
# ----------------------------------------------------------------------------
def kinodynamic_rrt(rng, iters=6000, attempts=5):
    for _ in range(attempts):
        plan = _rrt_once(rng, iters)
        if plan is not None:
            return plan
    raise RuntimeError("RRT did not reach the goal; try another seed")


def _rrt_once(rng, iters):
    tree = [dict(q=START.copy(), w=np.zeros(2), parent=-1, edge=None)]
    for _ in range(iters):
        target = GOAL if rng.random() < 0.1 else rng.random(2) * WORLD
        near = min(range(len(tree)), key=lambda i: np.sum((tree[i]['q'][:2] - target) ** 2))
        best = None
        for _ in range(8):                                   # try a few random controls
            u = np.array([rng.uniform(-A_PLAN, A_PLAN), rng.uniform(-D_PLAN, D_PLAN)])
            steps = rng.integers(3, 10)
            q, w, ok, traj = tree[near]['q'], tree[near]['w'], True, []
            for _ in range(steps):
                q, w = model_step(q, w, u, DT_EDGE)
                if in_collision(q[0], q[1], PLAN_MARGIN):
                    ok = False
                    break
                traj.append((q, w))
            if not ok:
                continue
            d = np.sum((q[:2] - target) ** 2)
            if best is None or d < best[0]:
                best = (d, u, traj)
        if best is None:
            continue
        _, u, traj = best
        tree.append(dict(q=traj[-1][0], w=traj[-1][1], parent=near, edge=(u, traj)))
        if np.hypot(*(traj[-1][0][:2] - GOAL)) < GOAL_R:
            # Walk back to the root; every model step becomes one plan node/edge.
            chain, i = [], len(tree) - 1
            while tree[i]['parent'] >= 0:
                chain.append(tree[i])
                i = tree[i]['parent']
            chain.reverse()
            nodes_q, nodes_w, edges_u = [START.copy()], [np.zeros(2)], []
            for n in chain:
                u, traj = n['edge']
                for q, w in traj:
                    edges_u.append(u)
                    nodes_q.append(q)
                    nodes_w.append(w)
            return dict(q=np.array(nodes_q), w=np.array(nodes_w), u=np.array(edges_u),
                        dt=np.full(len(edges_u), DT_EDGE), tree=tree)
    return None


# ----------------------------------------------------------------------------
# Step 4: the "real" robot. Different gains, drag, steering bias, noise, ramp.
# ----------------------------------------------------------------------------
class RealCar:
    def __init__(self, rng, noise=1.0):
        self.q, self.v, self.rng, self.noise = START.copy(), 0.0, rng, noise

    def step(self, u, dt):
        a = 0.9 * u[0] - 0.1 * self.v                      # weaker motor + drag
        delta = 0.95 * u[1] + 0.01                         # steering gain and bias
        slip = np.zeros(2)
        if in_ramp(self.q[0], self.q[1]):                  # unmodeled ramp
            a -= 0.6 * self.v
            slip = np.array([0.0, -0.15])                  # pushed sideways (world frame)
        self.v = float(np.clip(self.v + a * dt + self.noise * 0.05 * np.sqrt(dt) * self.rng.normal(), 0, V_MAX))
        om = self.v * np.tan(delta) / L + self.noise * 0.15 * np.sqrt(dt) * self.rng.normal()
        self.q = se2_compose(self.q, se2_exp(self.v * dt, 0, om * dt))
        self.q[:2] += slip * dt
        return self.q.copy()

    def observe(self):
        """Noisy camera pose measurement."""
        return self.q + self.rng.normal(0, [0.03, 0.03, 0.03])


# ----------------------------------------------------------------------------
# Step 3, 5, 6: the STELA sliding-window factor graph
# ----------------------------------------------------------------------------
SIG = dict(integ=0.01, dyn=0.05, obs=0.04, prior_q=np.array([0.04, 0.04, 0.3]), prior_w=0.3,
           prior_dt=0.05, u_reg=5.0, anchor=0.05, obst=0.02, lim=0.01)
OBST_EPS = 0.25                 # obstacle factor is active within this clearance [m]
DT_RANGE = (0.05, 0.4)


class Stela:
    def __init__(self, plan, n_hist=6, n_fwd=10):
        self.plan, self.n_hist, self.n_fwd = plan, n_hist, n_fwd
        self.N = len(plan['q'])
        self.first = 0                   # plan index of the oldest node in the window
        self.curr = 0                    # plan index of the current node
        k = min(self.N - 1, n_fwd)
        # Window variables, initialized from the plan (step 3).
        self.q = [plan['q'][i].copy() for i in range(k + 1)]
        self.w = [plan['w'][i].copy() for i in range(k + 1)]
        self.u = [plan['u'][i].copy() for i in range(k)]
        self.dt = [plan['dt'][i] for i in range(k)]
        self.obs = {}                    # plan index -> list of (dt_z, z)
        self.anchor = None               # prior on the oldest node (stands in for marginalization)

    # --- helpers --------------------------------------------------------------
    def last(self):
        return self.first + len(self.q) - 1

    def local(self, idx):
        return idx - self.first

    def pack(self):
        c = self.local(self.curr)
        return np.concatenate([np.concatenate(self.q), np.concatenate(self.w),
                               np.concatenate(self.u[c:]) if self.u[c:] else np.zeros(0),
                               np.array(self.dt[c:])])

    def unpack(self, x):
        n, c = len(self.q), self.local(self.curr)
        m = len(self.u) - c
        q = x[:3 * n].reshape(n, 3)
        w = x[3 * n:5 * n].reshape(n, 2)
        u = x[5 * n:5 * n + 2 * m].reshape(m, 2)
        dt = x[5 * n + 2 * m:]
        return q, w, u, dt

    # --- the factors ------------------------------------------------------------
    def factors(self, x, elapsed):
        """Returns a list of (residual block, indices of the variables it touches)."""
        q, w, u_f, dt_f = self.unpack(x)
        n, c = len(q), self.local(self.curr)
        m = len(u_f)
        iq = lambda i: [3 * i, 3 * i + 1, 3 * i + 2]
        iw = lambda i: [3 * n + 2 * i, 3 * n + 2 * i + 1]
        iu = lambda i: [5 * n + 2 * (i - c), 5 * n + 2 * (i - c) + 1] if i >= c else []
        idt = lambda i: [5 * n + 2 * m + (i - c)] if i >= c else []
        u = list(self.u[:c]) + list(u_f)          # history controls are fixed (already applied)
        dt = list(self.dt[:c]) + list(dt_f)
        F = []
        for i in range(n - 1):
            # Integration factor: q_i o Exp(w_i * dt_i) should land on q_{i+1} (SE(2)).
            pred = se2_compose(q[i], se2_exp(w[i][0] * dt[i], 0, w[i][1] * dt[i]))
            F.append((se2_between(pred, q[i + 1]) / SIG['integ'], iq(i) + iw(i) + idt(i) + iq(i + 1)))
            # Dynamics factor: the control should produce the next velocity.
            a, delta = np.clip(u[i], [-A_LIM, -D_LIM], [A_LIM, D_LIM])   # actuators saturate
            v_next = w[i][0] + a * dt[i]
            F.append(((np.array([v_next, v_next * np.tan(delta) / L]) - w[i + 1]) / SIG['dyn'],
                      iw(i) + iu(i) + idt(i) + iw(i + 1)))
        # Observation factors (history + current node): pose seen dt_z after the node.
        for idx in range(self.first, self.curr + 1):
            i = self.local(idx)
            for dz, z in self.obs.get(idx, []):
                pred = se2_compose(q[i], se2_exp(w[i][0] * dz, 0, w[i][1] * dz))
                F.append((se2_between(pred, z) / SIG['obs'], iq(i) + iw(i)))
        if self.anchor is not None:               # stands in for the marginalized past
            F.append((se2_between(self.anchor, q[0]) / SIG['anchor'], iq(0)))
        # Forward horizon (local adaptation): priors to the plan, obstacles. No prior on u.
        for idx in range(self.curr + 1, self.last() + 1):
            i = self.local(idx)
            F.append((se2_between(self.plan['q'][idx], q[i]) / SIG['prior_q'], iq(i)))
            F.append(((w[i] - self.plan['w'][idx]) / SIG['prior_w'], iw(i)))
            clr = obstacle_clearance(q[i][0], q[i][1])
            F.append((np.array([max(0.0, OBST_EPS - clr) / SIG['obst']]), iq(i)))
        for j in range(m):
            i = c + j
            F.append((np.array([(dt_f[j] - self.plan['dt'][self.curr + j]) / SIG['prior_dt']]), idt(i)))
            F.append((np.maximum(0, np.abs(u_f[j]) - [A_LIM, D_LIM]) / SIG['lim'], iu(i)))   # control limits
            # Very weak pull toward the plan's control. Not in the paper (controls get no prior);
            # it only keeps steering well-defined when the speed is ~0 and tan(delta) is unobservable.
            F.append(((u_f[j] - self.plan['u'][self.curr + j]) / SIG['u_reg'], iu(i)))
            lo = max(DT_RANGE[0], elapsed if j == 0 else 0)          # current edge can't end in the past
            F.append((np.array([max(0, lo - dt_f[j]), max(0, dt_f[j] - DT_RANGE[1])]) / SIG['lim'], idt(i)))
        return F

    def residuals(self, x, elapsed):
        return np.concatenate([r for r, _ in self.factors(x, elapsed)])

    def solve(self, elapsed):
        x0 = self.pack()
        F = self.factors(x0, elapsed)
        S = lil_matrix((sum(len(r) for r, _ in F), len(x0)), dtype=int)   # Jacobian sparsity
        row = 0
        for r, idx in F:
            for k in range(len(r)):
                S[row + k, idx] = 1
            row += len(r)
        sol = least_squares(self.residuals, x0, args=(elapsed,), method='trf', max_nfev=30,
                            jac_sparsity=S, x_scale='jac')
        q, w, u_f, dt_f = self.unpack(sol.x)
        c = self.local(self.curr)
        self.q, self.w = [qq for qq in q], [ww for ww in w]
        self.u = self.u[:c] + [uu for uu in u_f]
        self.dt = self.dt[:c] + list(dt_f)

    # --- step 5: observations ---------------------------------------------------
    def add_observation(self, dt_z, z):
        self.obs.setdefault(self.curr, []).append((dt_z, z))

    # --- step 7: slide the window ----------------------------------------------------
    def advance(self, elapsed, u_applied):
        c = self.local(self.curr)
        self.dt[c] = elapsed                               # what actually happened on this edge:
        self.u[c] = u_applied                              # its duration and the (average) control sent
        self.curr += 1
        nxt = self.last() + 1
        if nxt < self.N:                                   # next plan node joins the front
            self.q.append(self.plan['q'][nxt].copy())
            self.w.append(self.plan['w'][nxt].copy())
            self.u.append(self.plan['u'][nxt - 1].copy())
            self.dt.append(self.plan['dt'][nxt - 1])
        while self.curr - self.first > self.n_hist:        # oldest node leaves
            self.anchor = self.q[1].copy()
            self.q.pop(0); self.w.pop(0); self.u.pop(0); self.dt.pop(0)
            self.obs.pop(self.first, None)
            self.first += 1

    def current_control(self):
        c = self.local(self.curr)
        return np.clip(self.u[c], [-A_LIM, -D_LIM], [A_LIM, D_LIM]), self.dt[c]

    def done(self):
        return self.curr >= self.N - 1


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
def run_open_loop(plan, seed):
    car = RealCar(np.random.default_rng(seed))
    path = [car.q.copy()]
    for u, d in zip(plan['u'], plan['dt']):
        for _ in range(int(round(d / 0.01))):
            car.step(u, 0.01)
            path.append(car.q.copy())
            if in_collision(*car.q[:2]):
                return np.array(path), 'collision'
    ok = np.hypot(*(car.q[:2] - GOAL)) < GOAL_R
    return np.array(path), 'goal' if ok else 'missed goal'


def run_stela(plan, seed, record=False):
    rng = np.random.default_rng(seed)
    car, stela = RealCar(rng), Stela(plan)
    t, t_edge, sim_dt = 0.0, 0.0, 0.01
    next_obs, next_ctrl = 0.0, 0.0
    path, frames, status, obs_log = [car.q.copy()], [], 'timeout', []
    u = stela.current_control()[0]
    u_sum = np.zeros(2)
    while t < 60:
        if t >= next_obs:                                  # camera at ~10 Hz, jittered
            z = car.observe()
            stela.add_observation(t - t_edge, z)
            obs_log.append((t, z))
            next_obs = t + 0.1 + rng.uniform(-0.02, 0.02)
        if t >= next_ctrl:                                 # STELA update at 20 Hz
            stela.solve(t - t_edge)
            u, dt_curr = stela.current_control()
            if t - t_edge >= dt_curr:                      # current edge's duration elapsed
                stela.advance(t - t_edge, u_sum / max(t - t_edge, 1e-9))
                u_sum = np.zeros(2)
                t_edge = t
                if stela.done():
                    status = 'goal' if np.hypot(*(car.q[:2] - GOAL)) < GOAL_R + 0.15 else 'missed goal'
                    break
                u, _ = stela.current_control()
            if record:
                frames.append(dict(t=t, true=np.array(path), est=np.array(stela.q),
                                   curr=stela.local(stela.curr),
                                   obs=np.array([z[:2] for tz, z in obs_log if t - tz < 1.0])))
            next_ctrl = t + 0.05
        car.step(u, sim_dt)
        u_sum += u * sim_dt
        path.append(car.q.copy())
        t += sim_dt
        if in_collision(*car.q[:2]):
            status = 'collision'
            break
        if np.hypot(*(car.q[:2] - GOAL)) < GOAL_R:
            status = 'goal'
            break
    return np.array(path), status, frames


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def draw_world(ax, plan):
    for ox, oy, r in OBSTACLES:
        ax.add_patch(Circle((ox, oy), r, color='#C9C2B8'))
    rx, ry, rw, rh = RAMP
    ax.add_patch(Rectangle((rx, ry), rw, rh, fc='#E09F3E', alpha=0.12, ec='#E09F3E', ls='--'))
    ax.text(rx + rw / 2, ry + rh + 0.05, 'unmodeled ramp', ha='center', color='#9A5B12', fontsize=9)
    for n in plan['tree'][1:]:
        pts = np.array([plan['tree'][n['parent']]['q'][:2]] + [q[:2] for q, _ in n['edge'][1]])
        ax.plot(pts[:, 0], pts[:, 1], color='#8E99A4', lw=0.5, alpha=0.4)
    ax.plot(plan['q'][:, 0], plan['q'][:, 1], '-o', color='#E09F3E', ms=2.5, lw=2, label='SBMP plan')
    ax.add_patch(Circle(GOAL, GOAL_R, fill=False, ec='#2A9D8F', ls='--', lw=2))
    ax.plot(*START[:2], 'ko')
    ax.set_xlim(0, WORLD[0]); ax.set_ylim(0, WORLD[1]); ax.set_aspect('equal')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--animate', action='store_true', help='play the run in a window')
    ap.add_argument('--gif', action='store_true', help='save the run as stela_demo.gif')
    args = ap.parse_args()

    plan = kinodynamic_rrt(np.random.default_rng(args.seed))
    print(f"Plan: {len(plan['q'])} nodes, {plan['dt'].sum():.1f} s, from a tree of {len(plan['tree'])} nodes")

    ol_path, ol_status = run_open_loop(plan, args.seed + 100)
    st_path, st_status, frames = run_stela(plan, args.seed + 100, record=args.animate or args.gif)
    print(f"Open loop: {ol_status}")
    print(f"STELA:     {st_status}")

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_world(ax, plan)
    ax.plot(ol_path[:, 0], ol_path[:, 1], '--', color='#B5503A', lw=1.5, label=f'open loop ({ol_status})')
    ax.plot(st_path[:, 0], st_path[:, 1], color='#2A9D8F', lw=2, label=f'STELA ({st_status})')
    ax.legend(loc='lower right')
    ax.set_title('STELA demo: car-like robot following an SBMP plan')
    fig.savefig('stela_demo.png', dpi=150, bbox_inches='tight')
    print('Saved stela_demo.png')

    if (args.animate or args.gif) and frames:
        anim = make_animation(plan, frames, ol_path, ol_status, st_status)
        if args.gif:
            from matplotlib.animation import PillowWriter
            anim.save('stela_demo.gif', writer=PillowWriter(fps=10), dpi=80)
            print('Saved stela_demo.gif')
        if args.animate:
            plt.show()
        return anim


def car_outline(q, length=0.32, width=0.18):
    c, s = np.cos(q[2]), np.sin(q[2])
    pts = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * [length, width]
    return np.c_[q[0] + c * pts[:, 0] - s * pts[:, 1], q[1] + s * pts[:, 0] + c * pts[:, 1]]


def make_animation(plan, frames, ol_path, ol_status, st_status, step=2):
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(9, 6.2))
    draw_world(ax, plan)
    ol_ln, = ax.plot([], [], '--', color='#B5503A', lw=1.5, label='open loop')
    ol_x, = ax.plot([], [], 'x', color='#B5503A', ms=14, mew=3)
    true_ln, = ax.plot([], [], color='#8E99A4', lw=1.5, label='real robot (STELA)')
    obs_pts, = ax.plot([], [], '.', color='#2A9D8F', ms=6, alpha=0.6, label='camera observations')
    hist_ln, = ax.plot([], [], 'o-', color='#2A9D8F', ms=6, lw=2.5, label='estimated history')
    fwd_ln, = ax.plot([], [], 'o-', color='#B5503A', ms=6, lw=2.5, label='adapted horizon')
    curr_pt, = ax.plot([], [], 'o', color='#212A33', ms=9)
    car = Polygon(np.zeros((4, 2)), closed=True, fc='#2A9D8F', ec='#212A33', lw=1.5, alpha=0.45, zorder=1.5)
    ax.add_patch(car)
    ax.legend(loc='lower right', fontsize=8)
    idx = list(range(0, len(frames), step)) + [len(frames) - 1] * 15        # hold the last frame

    def update(k):
        f = frames[k]
        n_ol = min(len(ol_path) - 1, int(round(f['t'] / 0.01)))
        ol_ln.set_data(ol_path[:n_ol + 1, 0], ol_path[:n_ol + 1, 1])
        if n_ol == len(ol_path) - 1 and ol_status == 'collision':
            ol_x.set_data([ol_path[-1, 0]], [ol_path[-1, 1]])
        true_ln.set_data(f['true'][:, 0], f['true'][:, 1])
        if len(f['obs']):
            obs_pts.set_data(f['obs'][:, 0], f['obs'][:, 1])
        c = f['curr']
        hist_ln.set_data(f['est'][:c + 1, 0], f['est'][:c + 1, 1])
        fwd_ln.set_data(f['est'][c:, 0], f['est'][c:, 1])
        curr_pt.set_data([f['est'][c, 0]], [f['est'][c, 1]])
        car.set_xy(car_outline(f['true'][-1]))
        last = k == len(frames) - 1
        ax.set_title(f"STELA demo   t = {f['t']:.1f} s" + (f"   STELA: {st_status}, open loop: {ol_status}" if last else ''))
        return ol_ln, ol_x, true_ln, obs_pts, hist_ln, fwd_ln, curr_pt, car

    return FuncAnimation(fig, update, frames=idx, interval=100, blit=False)


if __name__ == '__main__':
    main()

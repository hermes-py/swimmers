"""
Phase 1 - Physics core (numpy, population-vectorized).

Model
-----
* A creature is a TREE: point 0 = head (root). For every point i>=1 there is a "bone":
  parent[i] -> i. parent[i] < i is required (tree order).
* Mass lives at the points (joints), bones are massless.
* Bone lengths are constrained (PBD / Gauss-Seidel).
* Bone 1 = "neck/spine": defines the head's direction (body angle phi = angle(bone 1) - theta0[1]).
  There is NO independent head-angle degree of freedom; the body angle arises entirely from geometry and physics.
  Bone 1 has no motor of its own (passive); at most 23 motorized joints + 1 passive spine.
* Joint i (i>=2 or ...) = the angle between bone i and the reference bone.
    - if parent[i] == 0 the reference = bone 1
    - otherwise the reference = the bone of point parent[i].
  Joint torque is applied as a two-sided force couple (net force zero, angular momentum conserved).
* Water: per-bone anisotropic (normal >> tangential) quadratic drag, with 2 sub-segments.
* No external cheating: only joint torques and water forces act on the creature.

All arrays have shape (B, N) or (B, N, 2); B = population, N = MAX_POINTS (25 = head + 24 limbs).
Unused points are masked with `active=False`.

Note: the joint angle theta_rel is measured relative to the rest angle (theta0) and limited by [jlo, jhi]
(jlo < 0 < jhi). This avoids the ±pi wrap-around problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

MAX_LIMBS = 24
MAX_POINTS = MAX_LIMBS + 2     # head + 24 limbs + 1 'nose' point (head direction)
NOSE_L, NOSE_M, NOSE_W = 0.5, 1.0, 0.1   # nose: a fixed-length, motorless bone attached to the head; its direction = body direction
# Physics version: incremented with every fix that changes numerical results.
#   1: dt=0.01 (was prone to numerical energy exploitation with light-limb + strong-motor combinations)
#   2: dt=0.005 + torque limit
#   3: LINEAR (viscous) drag, anisotropic (low Reynolds number regime): undulating swimming is much more
#      efficient (ideal wave 0.12 -> ~1 m/s) and when the motors are off the creature stops in ~2 s (old: 14 s).
#   4: joint limit spring/damping and TOTAL joint torque bounded by inertia (reduced inertia): in light + short bone chains
#      the limit spring blew up numerically (torque in the tens of thousands, joint speed hundreds of rad/s, result dependent on dt). The body
#      direction (nose) mode also came with this version (old mode behavior remains the same).
PHYSICS_VERSION = 4


# --------------------------------------------------------------------------- #
#  Parameters and data structures
# --------------------------------------------------------------------------- #
@dataclass
class PhysicsParams:
    dt: float = 0.005
    rho: float = 1.0          # water density scale
    cn: float = 0.0           # QUADRATIC drag: normal direction (off in version 3; for the high Reynolds regime)
    ct: float = 0.0           # quadratic drag: tangential direction
    kn_lin: float = 60.0      # LINEAR (viscous) drag: coefficient for the direction PERPENDICULAR to the bone
    kt_lin: float = 3.0       # direction PARALLEL to the bone (1/20 of kn_lin -> strong anisotropy = thrust)
    c_point: float = 0.05     # point (body) isotropic quadratic drag (small: the body is thin against the current)
    alpha_max: float = 250.0  # maximum angular acceleration (rad/s^2): torque = min(command*tau_max, alpha_max*inertia)
                              # an overly strong motor on a light limb = a numerical regime the time step cannot resolve
    k_limit: float = 40.0     # joint limit spring stiffness (multiplied by torque_max)
    c_limit: float = 4.0      # joint limit damping
    iterations: int = 4       # number of PBD passes (forward+backward sweep)
    safe_joint: bool = True   # version 4: joint torque/limit spring bounded by inertia (False -> version 3 behavior)
    w_limit: float = 80.0     # the largest natural frequency allowed for the limit spring (rad/s); w*dt=0.4 at dt=0.005


@dataclass
class Bodies:
    """Fixed anatomy of a population (B creatures). Will be generated from BodyGenome in Phase 2."""
    parent: np.ndarray        # (B,N) int
    length: np.ndarray        # (B,N) bone length (i>=1)
    width: np.ndarray         # (B,N) bone thickness (drag surface)
    mass: np.ndarray          # (B,N) point mass
    theta0: np.ndarray        # (B,N) rest angle (relative to the reference bone)
    jlo: np.ndarray           # (B,N) lower limit (<=0) relative to rest
    jhi: np.ndarray           # (B,N) upper limit (>=0)
    tau_max: np.ndarray       # (B,N) torque capacity
    jdamp: np.ndarray         # (B,N) joint damping
    active: np.ndarray        # (B,N) bool; [:,0] always True
    sub_mass: np.ndarray = None   # (B,N) point + subtree mass (inertia estimate)
    legacy_spine: np.ndarray = None  # (B,) bool: True = OLD mode (bone 1 is a passive spine, it determines the body direction);
                                     # False = body direction = head-nose bone (last point N-1), all limbs may be motorized

    @property
    def B(self):
        return self.parent.shape[0]

    @property
    def N(self):
        return self.parent.shape[1]

    @property
    def bone_active(self):
        ba = self.active.copy()
        ba[:, 0] = False
        return ba


@dataclass
class State:
    x: np.ndarray       # (B,N,2)
    v: np.ndarray       # (B,N,2)
    phi: np.ndarray     # (B,) body angle (derived)
    omega: np.ndarray   # (B,) body angular velocity (derived)
    t: float = 0.0
    theta: np.ndarray = None   # (B,N) latest joint angles (relative to rest)
    jvel: np.ndarray = None    # (B,N) latest joint angular velocities
    tau: np.ndarray = None     # (B,N) latest total applied joint torque


# --------------------------------------------------------------------------- #
#  Creature setup helpers
# --------------------------------------------------------------------------- #
def make_bodies(specs: list[dict], n_points: int = MAX_POINTS) -> Bodies:
    """
    specs: a dict for each creature:
        parent: list[int]            length n (including point 0; parent[0] is ignored)
        length, width, mass, theta0, jlo, jhi, tau_max, jdamp: lists of length n (index 0 ignored)
    """
    B, N = len(specs), n_points
    leg = np.array([bool(s.get("legacy_spine", True)) for s in specs], dtype=bool)
    z = lambda: np.zeros((B, N))
    parent = np.zeros((B, N), dtype=np.int64)
    arr = {k: z() for k in ("length", "width", "mass", "theta0", "jlo", "jhi", "tau_max", "jdamp")}
    active = np.zeros((B, N), dtype=bool)
    for b, s in enumerate(specs):
        n = len(s["parent"])
        assert n <= (N if leg[b] else N - 1), "at most %d limbs" % (N - 1 - (0 if leg[b] else 1))
        for i in range(1, n):
            assert 0 <= s["parent"][i] < i, "tree order: parent[i] < i is required"
        parent[b, :n] = s["parent"]
        for k in arr:
            arr[k][b, :n] = s[k]
        active[b, :n] = True
        if not leg[b]:                      # nose: the last point that carries the head direction (attached to the head)
            k = N - 1
            parent[b, k] = 0
            arr["length"][b, k], arr["width"][b, k], arr["mass"][b, k] = NOSE_L, NOSE_W, NOSE_M
            active[b, k] = True
    # unused points: safe values
    arr["length"][~active] = 1.0
    arr["mass"][~active] = 1.0
    arr["width"][~active] = 0.0
    sub = arr["mass"].copy()
    for b_ in range(B):
        for i in range(N - 1, 0, -1):
            if active[b_, i]:
                sub[b_, parent[b_, i]] += sub[b_, i]
    return Bodies(parent=parent, active=active, sub_mass=sub, legacy_spine=leg, **arr)


def init_state(bodies: Bodies, pos=None, heading=None) -> State:
    """Initial state from the rest geometry. pos (B,2), heading (B,) optional."""
    B, N = bodies.B, bodies.N
    pos = np.zeros((B, 2)) if pos is None else np.asarray(pos, float).reshape(B, 2)
    heading = np.zeros(B) if heading is None else np.asarray(heading, float).reshape(B)
    x = np.zeros((B, N, 2))
    x[:, 0] = pos
    alpha = np.zeros((B, N))
    ar = np.arange(B)
    leg = _leg(bodies)
    for i in range(1, N):
        p = bodies.parent[:, i]
        if i == 1:
            ref = heading
        else:
            ref = np.where(p == 0, np.where(leg, alpha[:, 1], heading), alpha[ar, p])
        alpha[:, i] = ref + bodies.theta0[:, i]
        d = bodies.length[:, i, None] * np.stack([np.cos(alpha[:, i]), np.sin(alpha[:, i])], -1)
        x[:, i] = x[ar, p] + d
    # unused points stay at the head
    x[~bodies.active] = np.broadcast_to(pos[:, None, :], x.shape)[~bodies.active]
    return State(x=x, v=np.zeros_like(x), phi=heading.copy(), omega=np.zeros(B), t=0.0,
                 theta=np.zeros((B, N)), jvel=np.zeros((B, N)), tau=np.zeros((B, N)))


# --------------------------------------------------------------------------- #
#  Core functions
# --------------------------------------------------------------------------- #
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _leg(bd):
    return np.ones(bd.B, dtype=bool) if bd.legacy_spine is None else bd.legacy_spine


def _heading(x, v, bd):
    """Body angle and angular velocity. Old mode: direction of bone 1 (-theta0). New mode: from the head to the nose bone."""
    leg = _leg(bd)
    k = np.where(leg, 1, bd.N - 1)
    ar = np.arange(bd.B)
    d = x[ar, k] - x[:, 0]
    dv = v[ar, k] - v[:, 0]
    off = np.where(leg, bd.theta0[:, 1], 0.0)
    phi = np.arctan2(d[:, 1], d[:, 0]) - off
    om = (d[:, 0] * dv[:, 1] - d[:, 1] * dv[:, 0]) / np.maximum((d ** 2).sum(-1), 1e-9)
    return phi, om


def _scatter(idx, vals, B, N):
    """Write (B,N,2) values by summing them into the (B,N) target indices -> (B,N,2)."""
    flat = (np.arange(B)[:, None] * N + idx).ravel()
    out = np.empty((B * N, 2))
    for c in range(2):
        out[:, c] = np.bincount(flat, weights=vals[..., c].ravel(), minlength=B * N)
    return out.reshape(B, N, 2)


def _gather(arr, idx):
    """Select rows from arr (B,N,...) using idx (B,N)."""
    if arr.ndim == 3:
        return np.take_along_axis(arr, idx[..., None], axis=1)
    return np.take_along_axis(arr, idx, axis=1)


SPINE_BONE = 1   # the bone that defines the body direction


def motor_mask(bd: Bodies) -> np.ndarray:
    """(B,N) bool: which bones have a motorized joint. Brain I/O and the genome read FROM HERE.
    To change the passive spine, this function and the reference-bone definition (refidx) in
    joint_kinematics/step are enough."""
    m = bd.bone_active.copy()
    leg = _leg(bd)
    m[:, SPINE_BONE] = ~leg & m[:, SPINE_BONE]         # old mode: bone 1 is a passive spine; motorized in the new mode
    m[:, bd.N - 1] = leg & m[:, bd.N - 1]              # new mode: last point = nose (motorless)
    return m


def joint_kinematics(st: State, bd: Bodies):
    """Joint angle (relative to rest) and angular velocity, bone directions, etc."""
    par = bd.parent
    xp = _gather(st.x, par)
    vp = _gather(st.v, par)
    d = st.x - xp
    L = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
    u = d / L[..., None]
    n = np.stack([-u[..., 1], u[..., 0]], -1)
    alpha = np.arctan2(d[..., 1], d[..., 0])
    dv = st.v - vp
    wb = (d[..., 0] * dv[..., 1] - d[..., 1] * dv[..., 0]) / L ** 2
    # reference bone: limbs attached to the head articulate relative to bone 1 in the old mode, to the nose bone (N-1) in the new mode
    refidx = np.where(par == 0, np.where(_leg(bd)[:, None], 1, bd.N - 1), par)
    ref = _gather(alpha, refidx)
    wref = _gather(wb, refidx)
    theta = _wrap(alpha - ref - bd.theta0)
    jvel = wb - wref
    has_joint = motor_mask(bd)
    theta = np.where(has_joint, theta, 0.0)
    jvel = np.where(has_joint, jvel, 0.0)
    return dict(d=d, L=L, u=u, n=n, alpha=alpha, wb=wb, theta=theta, jvel=jvel,
                has_joint=has_joint, refidx=refidx)


def step(st: State, bd: Bodies, ctrl: np.ndarray, pp: PhysicsParams, drag: bool = True):
    """
    One physics step. ctrl: (B,N) in [-1,1]; ctrl[:,i] = motor command of joint i (bone i).
    Returns: nothing (st is updated in place).
    """
    B, N = bd.B, bd.N
    dt = pp.dt
    ar = np.arange(B)
    ba = bd.bone_active
    par = bd.parent
    kin = joint_kinematics(st, bd)
    L, u, n, theta, jvel = kin["L"], kin["u"], kin["n"], kin["theta"], kin["jvel"]
    hj, refidx = kin["has_joint"], kin["refidx"]

    # ---- joint torque ----
    tau = np.clip(ctrl, -1.0, 1.0) * bd.tau_max
    tau_cap = pp.alpha_max * np.maximum(bd.sub_mass * L ** 2, 1e-4)
    tau = np.clip(tau, -tau_cap, tau_cap)
    tau = tau - bd.jdamp * jvel
    over = np.maximum(theta - bd.jhi, 0.0)
    under = np.maximum(bd.jlo - theta, 0.0)
    viol = (over > 0) | (under > 0)
    if pp.safe_joint:
        # Reduced inertia: series combination of the end-point inertias of the bone and the reference bone (a lower-bound estimate).
        L_ref = _gather(L, refidx)
        I_i = bd.mass * L ** 2
        I_r = _gather(bd.mass, refidx) * L_ref ** 2
        if not _leg(bd).all():                      # joints referenced to the nose: reduced inertia of the head+nose pair
            mh, mn = bd.mass[:, 0], bd.mass[:, N - 1]
            i_n = (mh * mn / (mh + mn) * bd.length[:, N - 1] ** 2)[:, None]
            I_r = np.where((refidx == N - 1) & ~_leg(bd)[:, None], i_n, I_r)
        I_red = np.maximum(I_i * I_r / np.maximum(I_i + I_r, 1e-9), 1e-4)
        k_lim = np.minimum(pp.k_limit * bd.tau_max, pp.w_limit ** 2 * I_red)
        c_lim = np.minimum(np.maximum(pp.c_limit * bd.tau_max * 0.1, 1.4 * np.sqrt(k_lim * I_red)), 0.9 * I_red / dt)
        tau += k_lim * (under - over)
        tau -= np.where(viol, c_lim * jvel, 0.0)
        cap2 = pp.alpha_max * I_red
        tau = np.clip(tau, -cap2, cap2)             # motor + damping + limit: the total torque bounds the angular acceleration
    else:
        tau += bd.tau_max * (-pp.k_limit * over + pp.k_limit * under)
        tau -= np.where(viol, pp.c_limit * bd.tau_max * jvel * 0.1, 0.0)
    tau = np.where(hj, tau, 0.0)
    leg = _leg(bd)
    if not leg.all():       # reaction on the nose: if the inertia of the nose+head pair is small, torque is limited accordingly (guards against numerical exploitation)
        mh, mn = bd.mass[:, 0], bd.mass[:, N - 1]
        i_nose = (mh * mn / (mh + mn) * bd.length[:, N - 1] ** 2)[:, None]
        nr = (refidx == N - 1) & ~leg[:, None]
        tc = np.where(nr, pp.alpha_max * i_nose, np.inf)
        tau = np.clip(tau, -tc, tc)
        tot = np.sum(np.abs(tau) * nr, axis=1)        # the total reaction torque of ALL limbs attached to the head is also limited
        cap = pp.alpha_max * i_nose[:, 0]
        scale = np.where(tot > cap, cap / np.maximum(tot, 1e-9), 1.0)
        tau = np.where(nr, tau * scale[:, None], tau)

    F = np.zeros_like(st.x)
    # force couple on bone i: +f n @ i , -f n @ parent[i], f = tau / L
    f = (tau / L)[..., None] * n
    F += f
    F -= _scatter(par, f, B, N)
    # reaction: -tau on the reference bone (including bone 1; there is no separate degree of freedom for the head)
    jp = refidx
    Lj = _gather(L, jp)
    nj = _gather(n, jp)
    gp = _gather(par, jp)           # start point of the reference bone
    fr = np.where(hj, -tau / Lj, 0.0)[..., None] * nj
    F += _scatter(jp, fr, B, N)
    F -= _scatter(gp, fr, B, N)

    # ---- water forces ----
    if drag:
        xp_v = _gather(st.v, par)
        for s in (0.25, 0.75):
            vs = (1 - s) * xp_v + s * st.v
            vt = np.sum(vs * u, -1, keepdims=True) * u
            vn = vs - vt
            seg = (0.5 * L * bd.width)[..., None]
            fd = -pp.rho * seg * (pp.cn * np.linalg.norm(vn, axis=-1, keepdims=True) * vn
                                  + pp.ct * np.linalg.norm(vt, axis=-1, keepdims=True) * vt)
            if pp.kn_lin or pp.kt_lin:
                fd = fd - pp.rho * seg * (pp.kn_lin * vn + pp.kt_lin * vt)
            fd = np.where(ba[..., None], fd, 0.0)
            F += s * fd
            F += _scatter(par, (1 - s) * fd, B, N)
        speed = np.linalg.norm(st.v, axis=-1, keepdims=True)
        F += -pp.rho * pp.c_point * np.sqrt(bd.mass)[..., None] * speed * st.v * bd.active[..., None]

    # ---- integration (semi-implicit Euler) ----
    inv_m = np.where(bd.active, 1.0 / bd.mass, 0.0)
    st.v += F * inv_m[..., None] * dt
    x_old = st.x.copy()
    st.x += st.v * dt

    # ---- length constraints (PBD, Gauss-Seidel) ----
    x = st.x
    order = list(range(1, N))
    for it in range(pp.iterations):
        seq = order if it % 2 == 0 else order[::-1]
        for i in seq:
            if not ba[:, i].any():
                continue
            p = par[:, i]
            xi = x[:, i]
            xpi = x[ar, p]
            d = xi - xpi
            dist = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
            wi = inv_m[:, i]
            wp = inv_m[ar, p]
            k = np.where(ba[:, i], (dist - bd.length[:, i]) / (dist * (wi + wp + 1e-12)), 0.0)
            corr = k[:, None] * d
            x[:, i] = xi - wi[:, None] * corr
            x[ar, p] = xpi + wp[:, None] * corr
    st.v = (x - x_old) / dt
    st.v[~bd.active] = 0.0

    st.t += dt
    st.theta, st.jvel, st.tau = theta, jvel, tau
    st.phi, st.omega = _heading(st.x, st.v, bd)


# --------------------------------------------------------------------------- #
#  Measurement helpers (will be used for sensors and fitness in Phase 2/3)
# --------------------------------------------------------------------------- #
def _meas_mass(bd: Bodies):
    """Masses used in measurements (center of mass, velocity): in the new mode excluding the nose (invisible direction carrier)."""
    m = np.where(bd.active, bd.mass, 0.0)
    m[:, bd.N - 1] = m[:, bd.N - 1] * _leg(bd)
    return m


def center_of_mass(st: State, bd: Bodies):
    m = _meas_mass(bd)
    return np.sum(st.x * m[..., None], axis=1) / m.sum(1, keepdims=True)


def com_velocity(st: State, bd: Bodies):
    m = _meas_mass(bd)
    return np.sum(st.v * m[..., None], axis=1) / m.sum(1, keepdims=True)


def constraint_error(st: State, bd: Bodies):
    """Maximum relative bone length error (B,)."""
    xp = _gather(st.x, bd.parent)
    L = np.linalg.norm(st.x - xp, axis=-1)
    err = np.abs(L - bd.length) / bd.length
    return np.where(bd.bone_active, err, 0.0).max(1)


def momentum(st: State, bd: Bodies):
    """Linear and angular momentum. For dragless tests."""
    m = np.where(bd.active, bd.mass, 0.0)
    p = np.sum(st.v * m[..., None], axis=1)
    Lz = np.sum(m * (st.x[..., 0] * st.v[..., 1] - st.x[..., 1] * st.v[..., 0]), axis=1)
    return p, Lz


# --------------------------------------------------------------------------- #
#  Ready-made test creatures
# --------------------------------------------------------------------------- #
def chain_spec(n_seg=5, seg_len=0.5, mass=1.0, width=0.15, tau_max=3.0,
               jrange=0.9, jdamp=0.15, head_mass=2.0):
    """Fish-like chain: the head faces forward, the body extends backward (theta0 = pi)."""
    n = n_seg + 1
    parent = [0] + list(range(0, n - 1))
    theta0 = [0.0, np.pi] + [0.0] * (n_seg - 1)
    m = [head_mass] + [mass] * n_seg
    lo = [0.0] + [-jrange] * n_seg
    hi = [0.0] + [jrange] * n_seg
    return dict(parent=parent, length=[1.0] + [seg_len] * n_seg, width=[0.0] + [width] * n_seg,
                mass=m, theta0=theta0, jlo=lo, jhi=hi, tau_max=[0.0] + [tau_max] * n_seg,
                jdamp=[0.0] + [jdamp] * n_seg)


def star_spec(n_arms=4, arm_len=0.6, seg=2, **kw):
    """A structure with arms branching from the head: tests branching (multiple limbs on one point)."""
    parent, theta0, length, width, mass, lo, hi, tau, jd = [0], [0.0], [1.0], [0.0], [2.0], [0.0], [0.0], [0.0], [0.0]
    for a in range(n_arms):
        ang = 2 * np.pi * a / n_arms
        prev = 0
        for s in range(seg):
            parent.append(prev)
            prev = len(parent) - 1
            theta0.append(ang if s == 0 else 0.0)
            length.append(arm_len)
            width.append(0.15)
            mass.append(1.0)
            lo.append(-0.9)
            hi.append(0.9)
            tau.append(3.0)
            jd.append(0.15)
    return dict(parent=parent, length=length, width=width, mass=mass, theta0=theta0,
                jlo=lo, jhi=hi, tau_max=tau, jdamp=jd)

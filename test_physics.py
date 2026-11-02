"""Phase 1 validation tests:  python test_physics.py"""
import time
import numpy as np
from physics import *


def wave_ctrl(bd, t, freq=1.0, k=0.9, amp=1.0, direction=+1):
    i = np.arange(bd.N)[None, :]
    return amp * np.sin(2 * np.pi * freq * t - direction * k * i) * np.ones((bd.B, 1))


def run(bd, st, pp, T, ctrl_fn, drag=True):
    maxerr = 0.0
    for _ in range(int(T / pp.dt)):
        step(st, bd, ctrl_fn(st.t), pp, drag=drag)
        maxerr = max(maxerr, constraint_error(st, bd).max())
    return maxerr


def test_constraints_and_finite():
    pp = PhysicsParams()
    bd = make_bodies([chain_spec(5), chain_spec(23, seg_len=0.3), star_spec(4), star_spec(8, seg=3)])
    st = init_state(bd, heading=np.random.uniform(-3, 3, bd.B))
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 6, (bd.B, bd.N))
    err = run(bd, st, pp, 10.0, lambda t: np.sin(5 * t + phase))
    assert np.isfinite(st.x).all() and np.isfinite(st.v).all()
    print(f"[1] finite numbers OK | largest relative bone length error = {err:.4f}")
    assert err < 0.05


def test_momentum_conservation():
    """Without drag, internal torques must not change the total momentum."""
    pp = PhysicsParams()
    bd = make_bodies([chain_spec(6), star_spec(4, seg=2)])
    st = init_state(bd, heading=[0.3, 1.0])
    rng = np.random.default_rng(1)
    phase = rng.uniform(0, 6, (bd.B, bd.N))
    run(bd, st, pp, 5.0, lambda t: np.sin(4 * t + phase), drag=False)
    p, Lz = momentum(st, bd)
    scale = np.sum(bd.mass * np.linalg.norm(st.v, axis=-1), axis=1)
    print(f"[2] dragless momentum: |p|={np.linalg.norm(p, axis=1)}, Lz={Lz}, (speed scale {scale})")
    assert np.all(np.linalg.norm(p, axis=1) < 1e-6 * 1 + 0.02 * scale)
    assert np.all(np.abs(Lz) < 0.05 * scale + 1e-3)


def test_swimming():
    """An angle wave (driven by PD) must produce forward/backward thrust; thrust must reverse when direction changes.
    Note: a raw torque wave is not physically meaningful swimming, so the test uses PD.
    Real swimming will be learned by evolution with raw torque in Phase 3."""
    pp = PhysicsParams()
    bd = make_bodies([chain_spec(6, tau_max=10.0)])
    ii = np.arange(bd.N)[None]
    res = {}
    for dirn in (+1, -1):
        st = init_state(bd, heading=[0.0])
        c0 = center_of_mass(st, bd)[0]
        for _ in range(int(12 / pp.dt)):
            tgt = 0.8 * np.sin(2 * np.pi * 1.0 * st.t - dirn * 0.9 * ii)
            step(st, bd, np.clip(30 * (tgt - st.theta) - 2 * st.jvel, -1, 1), pp)
        d = center_of_mass(st, bd)[0] - c0
        h = np.array([np.cos(st.phi[0]), np.sin(st.phi[0])])
        res[dirn] = float(d @ h)
        print(f"[3] wave direction {dirn:+d}: progress along body direction = {res[dirn]:+.2f} (body length 3.0), phi = {st.phi[0] % (2*np.pi):.2f}")
    assert res[+1] > 0.4 and res[-1] < -0.4, "wave direction must reverse the thrust"
    # a creature without control must stay in place (no cheating)
    st = init_state(bd, heading=[0.0])
    com0 = center_of_mass(st, bd)[0].copy()
    run(bd, st, pp, 5.0, lambda t: np.zeros((1, bd.N)))
    assert np.linalg.norm(center_of_mass(st, bd)[0] - com0) < 1e-6
    print("[3] uncontrolled creature stays in place (no external thrust)")


def test_speed():
    pp = PhysicsParams()
    for B, n in ((1, 6), (64, 12), (256, 24)):
        specs = [chain_spec(min(n, 24)) for _ in range(B)]
        bd = make_bodies(specs)
        st = init_state(bd)
        c = np.zeros((B, bd.N))
        t0 = time.time()
        for _ in range(100):
            step(st, bd, c, pp)
        dt = (time.time() - t0) / 100
        print(f"[4] B={B:4d}, {min(n,24)} limbs: {dt*1e3:6.2f} ms/step  ({B/dt:,.0f} creature-steps/s)")


def jelly_spec(legacy):
    """•-•-O-•-•: head in the middle, 2 segments on each side. Bone 1 = left arm, bone 3 = right arm (both attached to the head)."""
    th = 0.7
    return dict(parent=[0, 0, 1, 0, 3], length=[1.0, .5, .5, .5, .5], width=[0, .15, .15, .15, .15], mass=[2, 1, 1, 1, 1],
                theta0=[0, th, 0, -th, 0], jlo=[0, -.9, -.9, -.9, -.9], jhi=[0, .9, .9, .9, .9],
                tau_max=[0, 3, 3, 3, 3], jdamp=[0, .15, .15, .15, .15], legacy_spine=legacy)


def test_head_axis():
    """New mode: body direction = head direction. A symmetric jellyfish driven symmetrically must not drift in
    direction/lateral velocity (in the old mode bone 1 is passive so it drifts); momentum is conserved; numbers are finite."""
    pp = PhysicsParams()
    res = {}
    for leg in (True, False):
        bd = make_bodies([jelly_spec(leg)], n_points=6)
        st = init_state(bd, heading=[np.pi / 2])
        hs, lat = [], 0.0
        for _ in range(int(6.0 / pp.dt)):
            s = np.sin(2 * np.pi * 1.0 * st.t)
            c = np.array([[0, s, s, -s, -s, 0]])                     # mirror-symmetric drive
            step(st, bd, c, pp)
            hs.append(st.phi[0])
        com = center_of_mass(st, bd)[0] - np.array([0.0, 0.0])
        h = np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])
        res[leg] = (np.ptp(hs), abs(com[0]), constraint_error(st, bd).max(), np.isfinite(st.x).all())
    assert res[False][3] and res[False][2] < 0.05
    assert res[False][0] < 0.05, res[False]                           # new mode: direction drift ~0
    assert res[True][0] > 3 * res[False][0] + 0.02, res               # old mode: passive bone 1 creates asymmetry
    # dragless momentum (internal force)
    bd = make_bodies([jelly_spec(False)], n_points=6)
    st = init_state(bd, heading=[0.3])
    ph = np.random.default_rng(3).uniform(0, 6, (1, 6))
    for _ in range(int(4 / pp.dt)):
        step(st, bd, np.sin(5 * st.t + ph) * motor_mask(bd), pp, drag=False)
    p, Lz = momentum(st, bd)
    assert np.linalg.norm(p) < 0.05 and abs(Lz[0]) < 0.1, (p, Lz)
    print(f"[5] head direction: direction drift in symmetric jellyfish new {res[False][0]:.4f} rad vs old mode {res[True][0]:.4f} rad; "
          f"dragless momentum {np.linalg.norm(p):.4f}")


if __name__ == "__main__":
    test_constraints_and_finite()
    test_momentum_conservation()
    test_swimming()
    test_speed()
    test_head_axis()
    print("DONE")

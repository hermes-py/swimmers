"""Phase 2 validation tests:  python test_genome.py"""
import os, tempfile, time
import numpy as np
from genome import *
from brain import *
from physics import *


def test_random_and_mutation_validity():
    rng = np.random.default_rng(0)
    cfg = MutationConfig(p_add=0.4, p_remove=0.3, p_mirror=0.3, p_gene=0.2)   # aggressive
    g = random_genome(rng)
    sizes = []
    for _ in range(3000):
        g = mutate(g, rng, cfg)
        g.body.validate()
        sizes.append(g.body.n)
    print(f"[1] genome valid after 3000 aggressive mutations | limb count range {min(sizes)}..{max(sizes)}")
    assert max(sizes) <= MAX_LIMBS and min(sizes) >= MIN_LIMBS
    assert max(sizes) > 10, "limb addition/symmetry seems not to be working"


def test_mirror_symmetry():
    rng = np.random.default_rng(3)
    for _ in range(200):
        b = random_body(rng, 3, 6)
        m = mirror_limb(b, rng)
        if m.n == b.n:
            continue
        k = m.n - b.n                          # size of the added subtree
        A = GRID["angle_steps"]
        # the first added limb must be the mirror of the source: theta' = -theta, lo<->hi
        src = [i for i in range(b.first_free, b.n + 1)
               if (A - b.theta[i]) % A == m.theta[b.n + 1] and b.hi[i] == m.lo[b.n + 1]
               and b.lo[i] == m.hi[b.n + 1] and b.parent[i] == m.parent[b.n + 1]]
        assert src, "mirror limb not found"
        m.validate()
        print(f"[2] mirror symmetry OK (subtree size {k})")
        return
    raise AssertionError("no mirror could be produced")


def test_crossover():
    rng = np.random.default_rng(5)
    for _ in range(300):
        a, b = random_genome(rng), random_genome(rng)
        c = crossover(a, b, rng)
        c.body.validate()
        assert c.brain.W1.shape == (N_IN, HIDDEN)
    print("[3] 300 crossovers valid")


def test_json_roundtrip():
    rng = np.random.default_rng(7)
    g = random_genome(rng)
    g.meta = {"fitness": 1.25, "generation": 12}
    p = os.path.join(tempfile.mkdtemp(), "g.json")
    save_json(g, p)
    h = load_json(p)
    for f in ("parent",) + FIELDS:
        assert np.array_equal(getattr(g.body, f), getattr(h.body, f)), f
    assert np.allclose(g.brain.W1, h.brain.W1) and np.allclose(g.brain.W2, h.brain.W2)
    assert h.meta["generation"] == 12
    print(f"[4] JSON save/load exact | file {os.path.getsize(p)/1024:.0f} KB")


def test_brain_and_closed_loop():
    rng = np.random.default_rng(11)
    genomes = [random_genome(rng, n_min=2, n_max=24) for _ in range(32)]
    bd = build_bodies(genomes)
    bb = stack_brains([g.brain for g in genomes])
    pp = PhysicsParams()
    st = init_state(bd, heading=rng.uniform(-3, 3, bd.B))
    target = np.full(bd.B, 0.5)
    mm = motor_mask(bd)
    t0 = time.time()
    for _ in range(500):
        ctrl = act(st, bd, bb, target)
        assert np.all(ctrl[~mm] == 0), "no commands should go to motorless/empty slots"
        step(st, bd, ctrl, pp)
    assert np.isfinite(st.x).all() and np.isfinite(st.v).all()
    err = constraint_error(st, bd).max()
    print(f"[5] 32 random creatures, closed loop 5 s: finite, bone error {err:.3f}, {(time.time()-t0):.1f} s")
    # target is in the body frame: observation must be consistent when phi is rotated
    obs = observe(st, bd, np.full(bd.B, 0.0))
    ang = np.arctan2(obs[:, 73], obs[:, 72])
    exp = (0.0 - st.phi + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(np.cos(ang), np.cos(exp), atol=1e-6) and np.allclose(np.sin(ang), np.sin(exp), atol=1e-6)
    print("[5] target direction is correctly converted to the body frame")


def test_spine_flag_survives():
    rng = np.random.default_rng(4)
    g = random_genome(rng, legacy_spine=True)
    assert g.body.legacy_spine
    cfg = MutationConfig(p_add=0.5, p_remove=0.4, p_mirror=0.4, p_gene=0.3)
    h = g
    for _ in range(200):
        h = mutate(h, rng, cfg)
    assert h.body.legacy_spine and crossover(h, random_genome(rng), rng).body.legacy_spine
    p = os.path.join(tempfile.mkdtemp(), "s.json")
    save_json(g, p)
    assert load_json(p).body.legacy_spine
    d = g.to_dict(); d["body"].pop("legacy_spine")                  # old file: no flag -> OLD mode (behavior preserved)
    assert Genome.from_dict(d).body.legacy_spine
    assert not random_genome(rng).body.legacy_spine and not blank_body().legacy_spine   # new creatures: head direction
    bd = build_bodies([g, random_genome(rng)])
    assert list(bd.legacy_spine) == [True, False] and not motor_mask(bd)[0, 1] and motor_mask(bd)[1, 1]
    assert not motor_mask(bd)[1, bd.N - 1], "nose has no motor"
    print("[6] legacy_spine flag correct across mutation, crossover, JSON and old files; new mode is default")


def test_bone1_is_ordinary_in_nose_mode():
    """In the new mode bone 1 can be deleted/mirrored; in the old mode it is still protected. Limb count can drop to 1, physics works."""
    rng = np.random.default_rng(8)
    b = random_body(rng, 4, 6)
    assert not b.legacy_spine and b.first_free == 1
    r = delete_subtree(b, 1)
    assert r is not None
    nb, remap = r
    nb.validate()
    assert nb.n < b.n and nb.parent[1] == 0
    m = mirror_at(b, 1)
    assert m is not None
    m[0].validate()
    A = GRID["angle_steps"]
    assert (m[0].theta[m[1][1]] + b.theta[1]) % A == 0, "mirror: angle has opposite sign"
    leg = b.copy(); leg.legacy_spine = True
    assert delete_subtree(leg, 1) is None and mirror_at(leg, 1) is None
    cfg = MutationConfig(p_add=0.2, p_remove=0.6, p_mirror=0.3, p_gene=0.2, min_limbs=1)
    g, seen, bone1_removed = random_genome(rng), set(), 0
    for _ in range(2000):
        n0 = g.body.n
        g = mutate(g, rng, cfg)
        g.body.validate()
        seen.add(g.body.n)
    assert 1 in seen, "limb count must be able to drop to 1"
    one = Genome(blank_body(), random_brain(rng))
    r = delete_subtree(one.body, 2) 
    assert r is not None and r[0].n == 1
    bd = build_bodies([Genome(r[0], one.brain)])
    st = init_state(bd, heading=[0.0])
    bb = stack_brains([one.brain])
    for _ in range(300):
        step(st, bd, act(st, bd, bb, np.array([0.0])), PhysicsParams())
    assert np.isfinite(st.x).all()
    print("[7] in the new mode bone 1 is an ordinary limb: deletable, mirrorable, limb count drops to 1; protected in the old mode")


if __name__ == "__main__":
    test_random_and_mutation_validity()
    test_mirror_symmetry()
    test_crossover()
    test_json_roundtrip()
    test_brain_and_closed_loop()
    test_spine_flag_survives()
    test_bone1_is_ordinary_in_nose_mode()
    print("DONE")

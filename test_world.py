"""Chapter 2 / Stage 1 tests:  python test_world.py"""
import time
import numpy as np
from genome import random_genome
from physics import center_of_mass
from genome import random_genome
from world import World, WorldConfig


def small(seed=0, n=12, **kw):
    base = dict(e_init=100.0, e_max=200.0, e_birth=40.0, cost_split=80.0, cost_mate=30.0, carnivore_frac=0.0,
                r_bite=1.2, k_meat=20.0, meat_frac=0.5, cost_bite=2.0, bite_cd=1.0, body_init=0.0)   # fixed values for the mechanics tests
    base.update(kw)
    cfg = WorldConfig(n_init=n, seed=seed, **base)
    return World(cfg)


def test_torus_wrap():
    w = small(n=20)
    w.advance(30.0)
    com = center_of_mass(w.st, w.bd)
    assert np.isfinite(w.st.x).all()
    assert (com >= -1e-9).all() and (com < w.L + 1e-9).all(), (com.min(), com.max())
    print(f"[1] torus: 20 creatures 30 s, centers of mass within [0,{w.L:g}), numbers finite")


def put(w, i, head):
    """Shift creature i so that its head is at position `head`."""
    w.st.x[i] += np.asarray(head) - w.st.x[i, 0]


def test_sensing_and_torus_distance():
    w = small(n=1, source_rate=0.0)
    put(w, 0, [1.0, 20.0])
    w.fpos, w.fage = np.array([[6.0, 20.0]]), np.zeros(1)
    w._eat_and_sense()
    assert w.tmag[0] == 1 and abs(w.target[0]) < 1e-9, (w.target, w.tmag)
    w.fpos = np.array([[w.L - 1.0, 20.0]])                     # wrap-around: the shortest path is to the left (-x), 2 m
    w._eat_and_sense()
    assert w.tmag[0] == 1 and abs(abs(w.target[0]) - np.pi) < 1e-9, w.target
    w.fpos = np.array([[1.0 + 20.0, 20.0]])                    # 20 m > sensing range (12)
    w._eat_and_sense()
    assert w.tmag[0] == 0
    print("[2] sensing: direction correct, wrap-around shortest path, out of range = no command")


def test_eating():
    w = small(n=2, source_rate=0.0)
    w.energy[:] = [50.0, 190.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [30.0, 30.0])
    w.fpos = np.array([[10.5, 10.0], [10.0, 12.0], [30.0, 30.2]])   # 1st near, 2nd far, 3rd next to the second creature
    w.fage = np.zeros(3)
    w._eat_and_sense()
    assert w.eaten_total == 2 and len(w.fpos) == 1
    assert abs(w.energy[0] - 70.0) < 1e-9 and abs(w.energy[1] - 200.0) < 1e-9, w.energy
    assert abs(w.wasted - 10.0) < 1e-9, w.wasted                        # 190+20 -> 200: 10 overflowed
    print("[3] eating: food in range is eaten, energy is added, e_max overflow is wasted")


def test_starvation_and_compaction():
    w = small(n=6, source_rate=0.0)
    w.advance(0.2)
    before = {int(i): w.st.x[k].copy() for k, i in enumerate(w.ids)}
    w.energy[[1, 4]] = -0.1
    ids_dead = {int(w.ids[1]), int(w.ids[4])}
    w._deaths()
    assert w.pop == 4 and {d[1] for d in w.deaths} == ids_dead
    for k, i in enumerate(w.ids):
        assert np.array_equal(w.st.x[k], before[int(i)]), "the survivors' state must not change"
    assert len(w.genomes) == w.bd.B == w.bb.W1.shape[0] == w.st.x.shape[0] == w.ctrl.shape[0] == 4
    assert len(w.gen) == len(w.want) == len(w.fprox) == len(w.ids) == len(w.born) == 4
    w.advance(1.0)
    assert np.isfinite(w.st.x).all()
    print("[4] death by starvation: the dead are removed, the survivors' state is preserved exactly, the simulation continues")


def test_energy_accounting():
    w = small(n=30, source_rate=0.8, seed=3)
    e0 = float(w.energy.sum())
    w.advance(60.0)
    b = w.energy_balance()
    assert abs(w.balance_residual()) < 1e-6 * max(1.0, w.total0), w.balance_residual()
    # a motionless creature (zero-output brain) only spends basal energy
    w2 = small(n=3, source_rate=0.0, seed=1)
    for g in w2.genomes:
        g.brain.W2[:] = 0; g.brain.b2[:] = 0
    from brain import stack_brains
    w2.bb = stack_brains([g.brain for g in w2.genomes])
    e = w2.energy.copy()
    w2.advance(10.0)
    spent = e - w2.energy
    assert np.allclose(spent, w2.cfg.c_basal * w2.mass * 10.0, rtol=1e-6), (spent, w2.cfg.c_basal * w2.mass * 10)
    assert abs(w2.spent_move) < 1e-9
    print(f"[5] energy accounting balances (60 s: eaten {w.eaten_total}, movement cost {b['move']:.1f}, basal {b['basal']:.1f}); "
          f"a motionless creature spends only basal energy")


def test_food_dynamics():
    w = small(n=2, source_rate=20.0, food_max=50, food_life=5.0)
    w.advance(3.0)
    assert 0 < len(w.fpos) <= 50
    assert (w.fpos >= 0).all() and (w.fpos < w.L).all()
    w.cfg.source_rate = 0.0
    w.advance(6.0)
    assert len(w.fpos) == 0, "no unrotted food should remain"
    print("[6] food: cap and rotting work, positions are inside the world")


def test_deterministic():
    a, b = small(seed=7, n=10), small(seed=7, n=10)
    a.advance(15.0); b.advance(15.0)
    assert a.stats() == b.stats() and np.array_equal(a.st.x, b.st.x)
    print("[7] same seed = same result")


def ready(w):
    w.born[:] = -100.0          # all mature


def test_asexual_birth():
    w = small(n=4, source_rate=0.0)
    ready(w)
    w.energy[:] = [150.0, 79.0, 150.0, 150.0]
    w.want[:] = 0
    w.want[0] = [-1, 1, -1]      # wants it, has enough energy
    w.want[1] = [-1, 1, -1]      # wants it but energy 79 < 80
    w.born[2] = w.t - 1.0        # young (min_age=5) and wants it
    w.want[2] = [-1, 1, -1]
    w._reproduce()
    assert w.pop == 5 and w.births_asex == 1 and w.births_sex == 0
    assert abs(w.energy[0] - 70.0) < 1e-9 and w.energy[1] == 79.0 and w.energy[2] == 150.0
    assert abs(w.energy[-1] - 40.0) < 1e-9 and w.gen[-1] == 1 and w.gen[0] == 0
    w.genomes[-1].body.validate()
    assert abs(w.repro_paid - 80.0) < 1e-9 and abs(w.newborn_energy - 40.0) < 1e-9
    assert w.bd.B == w.st.x.shape[0] == w.bb.W1.shape[0] == w.ctrl.shape[0] == len(w.ids) == 5
    assert len(set(w.ids.tolist())) == 5
    w.advance(2.0)
    assert np.isfinite(w.st.x).all()
    print("[9] asexual reproduction: only a creature that wants it + has enough energy + is mature gives birth, the parent pays 80, the offspring is born with 40")


def test_sexual_birth():
    w = small(n=4, source_rate=0.0)
    ready(w)
    w.energy[:] = 100.0
    put(w, 0, [10.0, 10.0]); put(w, 1, [11.0, 10.0]); put(w, 2, [30.0, 30.0]); put(w, 3, [10.5, 25.0])
    w.want[:] = 0
    w.want[[0, 1, 2], 0] = 1.0                    # 0 and 1 are close and willing, 2 is far away
    w._reproduce()
    assert w.births_sex == 1 and w.births_asex == 0 and w.pop == 5
    assert abs(w.energy[0] - 70.0) < 1e-9 and abs(w.energy[1] - 70.0) < 1e-9 and w.energy[2] == 100.0 and w.energy[3] == 100.0
    assert abs(w.repro_paid - 60.0) < 1e-9
    child = w.genomes[-1]
    child.body.validate()
    assert abs(np.linalg.norm(w._mindiff(w.st.x[-1, 0] - np.array([10.5, 10.0])))) < 1e-6, "the offspring is born midway between the mates"
    # only one willing -> no birth; distant pair -> no birth
    w2 = small(n=3, source_rate=0.0); ready(w2); w2.energy[:] = 100.0
    put(w2, 0, [10.0, 10.0]); put(w2, 1, [14.0, 10.0]); put(w2, 2, [10.5, 10.0])
    w2.want[:] = 0; w2.want[[0, 1], 0] = 1.0
    w2._reproduce()
    assert w2.births_sex == 0
    print("[10] sexual reproduction: a close pair that both want it each pay 30, the offspring is born in the middle; a lone willing or distant pair does not give birth")


def test_growth_keeps_state():
    w = small(n=6, source_rate=0.0, init_limbs=(3, 5))
    ready(w)
    w.advance(3.0)
    N0 = w.bd.N
    before = w.st.x.copy(), w.st.v.copy(), w.st.phi.copy(), w.bd.legacy_spine.copy()
    big = random_genome(np.random.default_rng(3), n_min=9, n_max=9, brain_v2=True)
    w._add([big], np.array([[5.0, 5.0]]), np.array([1]), 40.0)
    N1 = w.bd.N
    assert N1 == 11 > N0 and w.pop == 7
    x0, v0, phi0, leg = before
    for i in range(6):
        n = w.genomes[i].body.n
        assert np.array_equal(w.st.x[i, :n + 1], x0[i, :n + 1])
        assert np.array_equal(w.st.v[i, :n + 1], v0[i, :n + 1])
        assert np.array_equal(w.st.x[i, N1 - 1], x0[i, N0 - 1]), "the nose point must be moved to the last index"
        assert w.st.phi[i] == phi0[i]
    w.advance(3.0)
    assert np.isfinite(w.st.x).all() and w.pop >= 1
    print(f"[11] batch growth: {N0} -> {N1} points, the state of existing creatures is preserved exactly, the simulation continues")


def test_accounting_with_births():
    w = small(n=40, source_rate=0.8, seed=5, e_init=150.0, want_bias=0.0)
    e0 = float(w.energy.sum())
    w.advance(90.0)
    b = w.energy_balance()
    assert w.births_asex + w.births_sex > 0, "there must be births in this test"
    assert abs(w.balance_residual()) < 1e-6 * max(1.0, w.total0), w.balance_residual()
    print(f"[12] energy accounting with births balances (90 s: {w.births_asex} asexual, {w.births_sex} sexual births, population {w.pop})")


def test_rescue():
    w = small(n=3, source_rate=0.0, min_pop=6)
    w._rescue()
    assert w.pop == 6 and w.rescued == 3
    e0 = 0.0
    w.advance(2.0)
    assert np.isfinite(w.st.x).all()
    w2 = small(n=3, source_rate=0.0)             # min_pop off: the population never grows automatically
    w2._rescue()
    assert w2.pop == 3
    print("[13] rescue: the population is topped up when min_pop is on, untouched when it is off")


def test_brain_compat():
    import os, tempfile
    from genome import save_json, load_json
    from sim import Session
    w = small(n=2)
    g = w.genomes[0]
    p = os.path.join(tempfile.mkdtemp(), "v2.json")
    save_json(g, p)
    h = load_json(p)
    assert h.brain.W1.shape[0] == 85 and h.brain.W2.shape[1] == 27
    s = Session(h, "v2 brain in v1 code", heading=1.0)
    for _ in range(60):
        s.advance(0.05, 0.5, 1.0, True)
    assert np.isfinite(s.st.x).all()
    print("[14] brain v2 JSON save/load; works in a PLAY (v1) session")


def test_seeded_and_fallback():
    import os
    from genome import load_json
    from world import seeded_genomes
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, "examples", f) for f in ("jellyfish.json", "snake.json", "example_symmetric_fish.json")]
    rng = np.random.default_rng(1)
    gs = seeded_genomes(paths, 20, rng)
    assert len(gs) == 20
    for k in range(3):
        o = load_json(paths[k])
        assert gs[k].body.n == o.body.n and np.array_equal(gs[k].body.length, o.body.length)
        assert np.array_equal(gs[k].brain.W1[:79], o.brain.W1), "the first 79 inputs must be preserved exactly"
        assert np.allclose(gs[k].brain.b2[-3:], [1.0, 0.6, -2.0])
    for g in gs:
        g.body.validate()
        assert g.brain.W1.shape == (85, 24)
    w = World(WorldConfig(n_init=20, seed=4), genomes=gs)
    w.advance(30.0)
    assert np.isfinite(w.st.x).all()
    # one that wants sexual reproduction but cannot find a mate divides on its own
    w = small(n=1, source_rate=0.0); ready(w); w.energy[:] = 100.0
    w.want[0] = [0.76, 0.6, -1.0]
    w._reproduce()
    assert w.births_asex == 1 and w.births_sex == 0 and w.pop == 2
    print("[15] seeded population: the example creatures exactly + mutated copies; one that wants sexual reproduction but finds no mate divides on its own")


def test_predation():
    # the prey dies, the attacker gets the meat energy; out of range/no request/cooldown/lack of energy = no bite
    w = small(n=3, source_rate=0.0, digest_penalty=0.5)
    ready(w); w.energy[:] = [100.0, 60.0, 100.0]
    w.diet[:] = [1.0, 0.0, 0.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.8, 10.0]); put(w, 2, [30.0, 30.0])
    w.want[:] = 0; w.want[0, 2] = 1.0
    m1 = float(w.mass[1])
    killed = w._bite()
    assert killed.tolist() == [False, True, False] and w.bites == 1 and w.kills == 1
    expect = 0.5 * 20.0 * 0 + (20.0 * m1 + 0.5 * 60.0) * (1.0 - 0.5 * (1.0 - 1.0))     # carnivore: meat efficiency 1
    assert abs(w.gained_meat - expect) < 1e-9, (w.gained_meat, expect)
    assert abs(w.energy[0] - min(100.0 - 2.0 + expect, 200.0)) < 1e-9, w.energy[0]
    w._deaths(killed)
    assert w.pop == 2 and w.deaths[-1][2] == "bite" and abs(w.dead_energy - 60.0) < 1e-9
    # a herbivore (diet 0) gets 50% efficiency from the same prey
    w = small(n=2, source_rate=0.0, digest_penalty=0.5)
    ready(w); w.energy[:] = [50.0, 40.0]; w.diet[:] = [0.0, 0.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[0, 2] = 1.0
    m1 = float(w.mass[1]); w._bite()
    assert abs(w.gained_meat - (20.0 * m1 + 0.5 * 40.0) * 0.5) < 1e-9
    # out of range: biting the air is free; cooldown: no new bite right after a kill
    w = small(n=2, source_rate=0.0); ready(w); w.energy[:] = 100.0
    put(w, 0, [10.0, 10.0]); put(w, 1, [20.0, 10.0]); w.want[:] = 0; w.want[0, 2] = 1.0
    k = w._bite(); assert not k.any() and w.energy[0] == 100.0 and w.bite_paid == 0.0
    w = small(n=3, source_rate=0.0); ready(w); w.energy[:] = 100.0
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); put(w, 2, [10.0, 10.6]); w.want[:] = 0; w.want[0, 2] = 1.0
    k = w._bite(); assert k.sum() == 1 and abs(w.bite_paid - 2.0) < 1e-9
    k2 = w._bite(); assert not k2.any() and abs(w.bite_paid - 2.0) < 1e-9, "there must be no new bite before the cooldown expires"
    # mutual bite: the one with more energy wins; if equal both die
    w = small(n=2, source_rate=0.0); ready(w); w.energy[:] = [100.0, 70.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[:, 2] = 1.0
    k = w._bite(); assert k.tolist() == [False, True], k
    w = small(n=2, source_rate=0.0); ready(w); w.energy[:] = [100.0, 100.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[:, 2] = 1.0
    k = w._bite(); assert k.tolist() == [True, True]
    print("[16] biting: the prey dies, meat energy arrives with the digestion efficiency, range/cooldown/mutual-bite rules are correct")


def test_diet_targeting_and_digestion():
    w = small(n=3, source_rate=0.0)
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.0, 15.0]); put(w, 2, [30.0, 30.0])
    w.energy[:] = 100.0
    w.fpos, w.fage = np.array([[16.0, 10.0]]), np.zeros(1)        # food 6 m to the right of 0; creature 1 is 5 m above
    w.diet[:] = [0.0, 0.0, 0.0]; w._sense()
    assert abs(w.target[0]) < 1e-9 and w.tmag[0] == 1                 # a herbivore targets the food
    w.diet[0] = 1.0; w._sense()
    assert abs(w.target[0] - np.pi / 2) < 1e-9 and w.tmag[0] == 1     # a carnivore targets the creature (above)
    w.fpos = np.zeros((0, 2)); w.diet[0] = 0.0; w._sense()
    assert w.tmag[0] == 0                                              # a herbivore with no food has no target (does not go for a creature)
    # digestion: a carnivore eats food at 30% efficiency
    w = small(n=1, source_rate=0.0, digest_penalty=0.7)
    put(w, 0, [10.0, 10.0]); w.energy[:] = 50.0; w.diet[:] = 1.0
    w.fpos, w.fage = np.array([[10.2, 10.0]]), np.zeros(1); w._eat()
    assert abs(w.energy[0] - (50.0 + 20.0 * 0.3)) < 1e-9, w.energy
    print("[17] feeding: a herbivore targets food, a carnivore targets a creature; a carnivore digests food at 30% efficiency")


def test_diet_inheritance_and_accounting():
    w = small(n=2, source_rate=0.0, diet_sigma=0.0); ready(w)
    w.diet[:] = [1.0, 0.0]; w.lin[:] = [3, 5]
    w.energy[:] = 100.0; put(w, 0, [10.0, 10.0]); put(w, 1, [11.0, 10.0])
    w.want[:] = 0; w.want[:, 0] = 1.0
    w._reproduce()
    assert w.pop == 3 and abs(w.diet[-1] - 0.5) < 1e-9 and w.lin[-1] in (3, 5) and w.genomes[-1].meta["diet"] == 0.5
    w = small(n=60, source_rate=0.5, seed=9, want_bias=0.0, carnivore_frac=0.4, e_init=150.0)
    e0 = float(w.energy.sum())
    w.advance(120.0)
    b = w.energy_balance()
    assert w.kills > 0 and b["meat"] > 0, (w.kills, b)
    assert abs(w.balance_residual()) < 1e-6 * max(1.0, w.total0), w.balance_residual()
    print(f"[18] diet inheritance (parents' average), energy accounting with predation balances (120 s: {w.kills} kills, meat gain {b['meat']:.0f})")


def test_no_cannibal_farm():
    """Energy conservation: a carnivore eating its own offspring must not gain MORE than the cost of reproduction (with default settings)."""
    c = WorldConfig()
    gain_per_child = (c.meat_frac * c.e_birth) + c.k_meat * 5.0           # a 5 kg offspring
    assert gain_per_child < c.cost_split, (gain_per_child, c.cost_split)
    assert gain_per_child < 2 * c.cost_mate, (gain_per_child, 2 * c.cost_mate)
    print(f"[19] no farming loophole: eating an offspring yields {gain_per_child:.0f}, reproduction costs {c.cost_split:.0f} (asexual) / {2*c.cost_mate:.0f} (sexual)")


def test_carcass():
    # a creature that starves leaves its body energy as a carcass
    w = small(n=3, source_rate=0.0, body_init=40.0, k_meat=0.0)
    ready(w); w.energy[:] = [100.0, -0.1, 100.0]
    put(w, 1, [12.0, 12.0])
    w.total0 = w.total_energy()
    w._deaths()
    assert w.pop == 2 and len(w.cpos) == 1 and abs(w.cval[0] - 40.0) < 1e-9 and abs(w.carc_created - 40.0) < 1e-9
    assert np.allclose(w.cpos[0], [12.0, 12.0])
    assert abs(w.balance_residual()) < 1e-9
    # killed by a bite: prey V=energy+body; the attacker gets meat_frac·V, the remainder (1-meat_frac)·V becomes a carcass
    w = small(n=2, source_rate=0.0, body_init=40.0, k_meat=0.0, digest_penalty=0.5)
    ready(w); w.energy[:] = [100.0, 60.0]; w.diet[:] = [1.0, 0.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[0, 2] = 1.0
    w.total0 = w.total_energy()
    k = w._bite(); w._deaths(k)
    assert abs(w.gained_meat - 50.0) < 1e-9, w.gained_meat           # 0.5·(60+40)·eff(1)
    assert len(w.cval) == 1 and abs(w.cval[0] - 50.0) < 1e-9
    assert abs(w.balance_residual()) < 1e-9, w.balance_residual()
    # eating a carcass: carnivore 100%, herbivore 30% (digest_penalty 0.7)
    for diet, exp in ((1.0, 50.0), (0.0, 50.0 * 0.3)):
        w = small(n=1, source_rate=0.0, body_init=40.0, digest_penalty=0.7)
        put(w, 0, [10.0, 10.0]); w.energy[:] = 50.0; w.diet[:] = diet
        w.cpos, w.cval, w.cage = np.array([[10.3, 10.0]]), np.array([50.0]), np.zeros(1)
        w.carc_created += 50.0; w.total0 -= 0.0
        w._eat()
        assert abs(w.energy[0] - (50.0 + exp)) < 1e-9 and len(w.cval) == 0 and w.carc_eaten == 1
    # rotting
    w = small(n=1, source_rate=0.0, carcass_life=5.0)
    w.cpos, w.cval, w.cage = np.array([[3.0, 3.0]]), np.array([30.0]), np.zeros(1)
    w._food_step(6.0)
    assert len(w.cval) == 0 and abs(w.carc_decay - 30.0) < 1e-9
    # a carnivore targets the carcass, a herbivore does not
    w = small(n=2, source_rate=0.0, body_init=40.0)
    put(w, 0, [10.0, 10.0]); put(w, 1, [35.0, 35.0])
    w.cpos, w.cval, w.cage = np.array([[10.0, 16.0]]), np.array([50.0]), np.zeros(1)
    w.diet[:] = [1.0, 0.0]; w._sense()
    assert abs(w.target[0] - np.pi / 2) < 1e-9 and w.tmag[0] == 1
    w.diet[0] = 0.0; w._sense(); assert w.tmag[0] == 0
    # body energy at birth: asexual (cost 80 - e_birth 40) · body_frac 0.5 = 20
    w = small(n=1, source_rate=0.0, body_init=40.0)
    ready(w); w.energy[:] = 150.0; w.want[0] = [-1, 1, -1]
    w.total0 = w.total_energy()
    w._reproduce()
    assert w.pop == 2 and abs(w.body[-1] - 20.0) < 1e-9 and abs(w.newborn_body - 20.0) < 1e-9
    assert abs(w.balance_residual()) < 1e-9
    print("[20] carcass: starvation/bite carcasses, carnivore/herbivore digestion, rotting, targeting, body energy at birth; conservation ~0")


def test_speed_100():
    w = World(WorldConfig(seed=2))
    t0 = time.time()
    w.advance(20.0)
    dt = time.time() - t0
    s = w.stats()
    assert np.isfinite(w.st.x).all()
    print(f"[8] 100 random creatures, 20 s simulation: {dt:.1f} s real (1 s sim = {dt/20:.2f} s real); "
          f"population {s['pop']}, food {s['food']}")


if __name__ == "__main__":
    test_torus_wrap()
    test_sensing_and_torus_distance()
    test_eating()
    test_starvation_and_compaction()
    test_energy_accounting()
    test_food_dynamics()
    test_deterministic()
    test_asexual_birth()
    test_sexual_birth()
    test_growth_keeps_state()
    test_accounting_with_births()
    test_rescue()
    test_brain_compat()
    test_seeded_and_fallback()
    test_predation()
    test_diet_targeting_and_digestion()
    test_diet_inheritance_and_accounting()
    test_no_cannibal_farm()
    test_carcass()
    test_speed_100()
    print("DONE")

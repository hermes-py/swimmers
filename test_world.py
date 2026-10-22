"""Chapter 2 / Aşama 1 testleri:  python test_world.py"""
import time
import numpy as np
from genome import random_genome
from physics import center_of_mass
from genome import random_genome
from world import World, WorldConfig


def small(seed=0, n=12, **kw):
    base = dict(e_init=100.0, e_max=200.0, e_birth=40.0, cost_split=80.0, cost_mate=30.0, carnivore_frac=0.0,
                r_bite=1.2, k_meat=20.0, meat_frac=0.5, cost_bite=2.0, bite_cd=1.0, body_init=0.0)   # mekanik testleri için sabit değerler
    base.update(kw)
    cfg = WorldConfig(n_init=n, seed=seed, **base)
    return World(cfg)


def test_torus_wrap():
    w = small(n=20)
    w.advance(30.0)
    com = center_of_mass(w.st, w.bd)
    assert np.isfinite(w.st.x).all()
    assert (com >= -1e-9).all() and (com < w.L + 1e-9).all(), (com.min(), com.max())
    print(f"[1] torus: 20 canlı 30 sn, kütle merkezleri [0,{w.L:g}) içinde, sayılar sonlu")


def put(w, i, head):
    """i. canlıyı kafası `head` konumunda olacak şekilde ötele."""
    w.st.x[i] += np.asarray(head) - w.st.x[i, 0]


def test_sensing_and_torus_distance():
    w = small(n=1, source_rate=0.0)
    put(w, 0, [1.0, 20.0])
    w.fpos, w.fage = np.array([[6.0, 20.0]]), np.zeros(1)
    w._eat_and_sense()
    assert w.tmag[0] == 1 and abs(w.target[0]) < 1e-9, (w.target, w.tmag)
    w.fpos = np.array([[w.L - 1.0, 20.0]])                     # sarmal: en kısa yol sola (-x), 2 m
    w._eat_and_sense()
    assert w.tmag[0] == 1 and abs(abs(w.target[0]) - np.pi) < 1e-9, w.target
    w.fpos = np.array([[1.0 + 20.0, 20.0]])                    # 20 m > algı menzili (12)
    w._eat_and_sense()
    assert w.tmag[0] == 0
    print("[2] algı: yön doğru, sarmal en kısa yol, menzil dışı = komut yok")


def test_eating():
    w = small(n=2, source_rate=0.0)
    w.energy[:] = [50.0, 190.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [30.0, 30.0])
    w.fpos = np.array([[10.5, 10.0], [10.0, 12.0], [30.0, 30.2]])   # 1. yakın, 2. uzak, 3. ikincinin yanında
    w.fage = np.zeros(3)
    w._eat_and_sense()
    assert w.eaten_total == 2 and len(w.fpos) == 1
    assert abs(w.energy[0] - 70.0) < 1e-9 and abs(w.energy[1] - 200.0) < 1e-9, w.energy
    assert abs(w.wasted - 10.0) < 1e-9, w.wasted                        # 190+20 -> 200: 10 taştı
    print("[3] yeme: menzildeki yem yenir, enerji eklenir, e_max aşımı boşa gider")


def test_starvation_and_compaction():
    w = small(n=6, source_rate=0.0)
    w.advance(0.2)
    before = {int(i): w.st.x[k].copy() for k, i in enumerate(w.ids)}
    w.energy[[1, 4]] = -0.1
    ids_dead = {int(w.ids[1]), int(w.ids[4])}
    w._deaths()
    assert w.pop == 4 and {d[1] for d in w.deaths} == ids_dead
    for k, i in enumerate(w.ids):
        assert np.array_equal(w.st.x[k], before[int(i)]), "sağ kalanın durumu değişmemeli"
    assert len(w.genomes) == w.bd.B == w.bb.W1.shape[0] == w.st.x.shape[0] == w.ctrl.shape[0] == 4
    assert len(w.gen) == len(w.want) == len(w.fprox) == len(w.ids) == len(w.born) == 4
    w.advance(1.0)
    assert np.isfinite(w.st.x).all()
    print("[4] açlıktan ölüm: ölenler çıkıyor, sağ kalanların durumu birebir korunuyor, simülasyon sürüyor")


def test_energy_accounting():
    w = small(n=30, source_rate=0.8, seed=3)
    e0 = float(w.energy.sum())
    w.advance(60.0)
    b = w.energy_balance()
    assert abs(w.balance_residual()) < 1e-6 * max(1.0, w.total0), w.balance_residual()
    # hareketsiz (sıfır çıktılı beyin) canlı yalnızca bazal harcar
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
    print(f"[5] enerji muhasebesi tutuyor (60 sn: yenen {w.eaten_total}, hareket gideri {b['move']:.1f}, bazal {b['basal']:.1f}); "
          f"hareketsiz canlı yalnızca bazal harcıyor")


def test_food_dynamics():
    w = small(n=2, source_rate=20.0, food_max=50, food_life=5.0)
    w.advance(3.0)
    assert 0 < len(w.fpos) <= 50
    assert (w.fpos >= 0).all() and (w.fpos < w.L).all()
    w.cfg.source_rate = 0.0
    w.advance(6.0)
    assert len(w.fpos) == 0, "çürümeyen yem kalmamalı"
    print("[6] yem: tavan ve çürüme çalışıyor, konumlar dünya içinde")


def test_deterministic():
    a, b = small(seed=7, n=10), small(seed=7, n=10)
    a.advance(15.0); b.advance(15.0)
    assert a.stats() == b.stats() and np.array_equal(a.st.x, b.st.x)
    print("[7] aynı tohum = aynı sonuç")


def ready(w):
    w.born[:] = -100.0          # hepsi olgun


def test_asexual_birth():
    w = small(n=4, source_rate=0.0)
    ready(w)
    w.energy[:] = [150.0, 79.0, 150.0, 150.0]
    w.want[:] = 0
    w.want[0] = [-1, 1, -1]      # istiyor, yeterli enerjisi var
    w.want[1] = [-1, 1, -1]      # istiyor ama enerjisi 79 < 80
    w.born[2] = w.t - 1.0        # genç (min_age=5) ve istiyor
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
    print("[9] eşeysiz üreme: yalnızca isteyen+yeterli enerjili+olgun canlı doğurur, ebeveyn 80 öder, yavru 40 ile doğar")


def test_sexual_birth():
    w = small(n=4, source_rate=0.0)
    ready(w)
    w.energy[:] = 100.0
    put(w, 0, [10.0, 10.0]); put(w, 1, [11.0, 10.0]); put(w, 2, [30.0, 30.0]); put(w, 3, [10.5, 25.0])
    w.want[:] = 0
    w.want[[0, 1, 2], 0] = 1.0                    # 0 ve 1 yakın ve istekli, 2 uzakta
    w._reproduce()
    assert w.births_sex == 1 and w.births_asex == 0 and w.pop == 5
    assert abs(w.energy[0] - 70.0) < 1e-9 and abs(w.energy[1] - 70.0) < 1e-9 and w.energy[2] == 100.0 and w.energy[3] == 100.0
    assert abs(w.repro_paid - 60.0) < 1e-9
    child = w.genomes[-1]
    child.body.validate()
    assert abs(np.linalg.norm(w._mindiff(w.st.x[-1, 0] - np.array([10.5, 10.0])))) < 1e-6, "yavru eşlerin ortasında doğar"
    # tek istekli -> doğum yok; uzak çift -> doğum yok
    w2 = small(n=3, source_rate=0.0); ready(w2); w2.energy[:] = 100.0
    put(w2, 0, [10.0, 10.0]); put(w2, 1, [14.0, 10.0]); put(w2, 2, [10.5, 10.0])
    w2.want[:] = 0; w2.want[[0, 1], 0] = 1.0
    w2._reproduce()
    assert w2.births_sex == 0
    print("[10] eşeyli üreme: yakın ve ikisi de isteyen çift her biri 30 öder, yavru ortada doğar; tek istekli/uzak çift doğurmaz")


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
        assert np.array_equal(w.st.x[i, N1 - 1], x0[i, N0 - 1]), "burun noktası son indekse taşınmalı"
        assert w.st.phi[i] == phi0[i]
    w.advance(3.0)
    assert np.isfinite(w.st.x).all() and w.pop >= 1
    print(f"[11] batch genişlemesi: {N0} -> {N1} nokta, mevcut canlıların durumu birebir korunuyor, simülasyon sürüyor")


def test_accounting_with_births():
    w = small(n=40, source_rate=0.8, seed=5, e_init=150.0, want_bias=0.0)
    e0 = float(w.energy.sum())
    w.advance(90.0)
    b = w.energy_balance()
    assert w.births_asex + w.births_sex > 0, "bu testte doğum olmalı"
    assert abs(w.balance_residual()) < 1e-6 * max(1.0, w.total0), w.balance_residual()
    print(f"[12] doğumlu enerji muhasebesi tutuyor (90 sn: {w.births_asex} eşeysiz, {w.births_sex} eşeyli doğum, nüfus {w.pop})")


def test_rescue():
    w = small(n=3, source_rate=0.0, min_pop=6)
    w._rescue()
    assert w.pop == 6 and w.rescued == 3
    e0 = 0.0
    w.advance(2.0)
    assert np.isfinite(w.st.x).all()
    w2 = small(n=3, source_rate=0.0)             # min_pop kapalı: nüfus asla otomatik artmaz
    w2._rescue()
    assert w2.pop == 3
    print("[13] kurtarma: min_pop açıkken nüfus tamamlanıyor, kapalıyken dokunmuyor")


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
    s = Session(h, "v2 beyin v1 kodunda", heading=1.0)
    for _ in range(60):
        s.advance(0.05, 0.5, 1.0, True)
    assert np.isfinite(s.st.x).all()
    print("[14] beyin v2 JSON'u kaydet/yükle; DENE (v1) oturumunda çalışıyor")


def test_seeded_and_fallback():
    import os
    from genome import load_json
    from world import seeded_genomes
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, "ornekler", f) for f in ("denizanasi.json", "yilan.json", "ornek_simetrik_balik.json")]
    rng = np.random.default_rng(1)
    gs = seeded_genomes(paths, 20, rng)
    assert len(gs) == 20
    for k in range(3):
        o = load_json(paths[k])
        assert gs[k].body.n == o.body.n and np.array_equal(gs[k].body.length, o.body.length)
        assert np.array_equal(gs[k].brain.W1[:79], o.brain.W1), "ilk 79 giriş aynen korunmalı"
        assert np.allclose(gs[k].brain.b2[-3:], [1.0, 0.6, -2.0])
    for g in gs:
        g.body.validate()
        assert g.brain.W1.shape == (85, 24)
    w = World(WorldConfig(n_init=20, seed=4), genomes=gs)
    w.advance(30.0)
    assert np.isfinite(w.st.x).all()
    # eş bulamayan 'eşeyli isteyen' kendi başına bölünür
    w = small(n=1, source_rate=0.0); ready(w); w.energy[:] = 100.0
    w.want[0] = [0.76, 0.6, -1.0]
    w._reproduce()
    assert w.births_asex == 1 and w.births_sex == 0 and w.pop == 2
    print("[15] tohumlu nüfus: örnek canlılar aynen + mutasyonlu kopyalar; eş bulamayan eşeyli isteyen kendi başına bölünür")


def test_predation():
    # av ölür, saldırgan et enerjisini alır; menzil dışı/istek yok/bekleme/enerji eksikliği = ısırık yok
    w = small(n=3, source_rate=0.0, digest_penalty=0.5)
    ready(w); w.energy[:] = [100.0, 60.0, 100.0]
    w.diet[:] = [1.0, 0.0, 0.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.8, 10.0]); put(w, 2, [30.0, 30.0])
    w.want[:] = 0; w.want[0, 2] = 1.0
    m1 = float(w.mass[1])
    killed = w._bite()
    assert killed.tolist() == [False, True, False] and w.bites == 1 and w.kills == 1
    expect = 0.5 * 20.0 * 0 + (20.0 * m1 + 0.5 * 60.0) * (1.0 - 0.5 * (1.0 - 1.0))     # etçil: et verimi 1
    assert abs(w.gained_meat - expect) < 1e-9, (w.gained_meat, expect)
    assert abs(w.energy[0] - min(100.0 - 2.0 + expect, 200.0)) < 1e-9, w.energy[0]
    w._deaths(killed)
    assert w.pop == 2 and w.deaths[-1][2] == "ısırık" and abs(w.dead_energy - 60.0) < 1e-9
    # otçul (diet 0) aynı avdan %50 verimle alır
    w = small(n=2, source_rate=0.0, digest_penalty=0.5)
    ready(w); w.energy[:] = [50.0, 40.0]; w.diet[:] = [0.0, 0.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[0, 2] = 1.0
    m1 = float(w.mass[1]); w._bite()
    assert abs(w.gained_meat - (20.0 * m1 + 0.5 * 40.0) * 0.5) < 1e-9
    # menzil dışı: havaya ısırmak bedava; bekleme: kill'den hemen sonra yeni ısırık olmaz
    w = small(n=2, source_rate=0.0); ready(w); w.energy[:] = 100.0
    put(w, 0, [10.0, 10.0]); put(w, 1, [20.0, 10.0]); w.want[:] = 0; w.want[0, 2] = 1.0
    k = w._bite(); assert not k.any() and w.energy[0] == 100.0 and w.bite_paid == 0.0
    w = small(n=3, source_rate=0.0); ready(w); w.energy[:] = 100.0
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); put(w, 2, [10.0, 10.6]); w.want[:] = 0; w.want[0, 2] = 1.0
    k = w._bite(); assert k.sum() == 1 and abs(w.bite_paid - 2.0) < 1e-9
    k2 = w._bite(); assert not k2.any() and abs(w.bite_paid - 2.0) < 1e-9, "bekleme süresi dolmadan yeni ısırık olmamalı"
    # karşılıklı ısırma: enerjisi yüksek olan kazanır; eşitse ikisi de ölür
    w = small(n=2, source_rate=0.0); ready(w); w.energy[:] = [100.0, 70.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[:, 2] = 1.0
    k = w._bite(); assert k.tolist() == [False, True], k
    w = small(n=2, source_rate=0.0); ready(w); w.energy[:] = [100.0, 100.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[:, 2] = 1.0
    k = w._bite(); assert k.tolist() == [True, True]
    print("[16] ısırma: av ölür, et enerjisi sindirim verimiyle gelir, menzil/bekleme/karşılıklı ısırma kuralları doğru")


def test_diet_targeting_and_digestion():
    w = small(n=3, source_rate=0.0)
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.0, 15.0]); put(w, 2, [30.0, 30.0])
    w.energy[:] = 100.0
    w.fpos, w.fage = np.array([[16.0, 10.0]]), np.zeros(1)        # 0'ın sağında 6 m'de yem; 1 ise 5 m yukarıda
    w.diet[:] = [0.0, 0.0, 0.0]; w._sense()
    assert abs(w.target[0]) < 1e-9 and w.tmag[0] == 1                 # otçul yemi hedefler
    w.diet[0] = 1.0; w._sense()
    assert abs(w.target[0] - np.pi / 2) < 1e-9 and w.tmag[0] == 1     # etçil canlıyı (yukarıdaki) hedefler
    w.fpos = np.zeros((0, 2)); w.diet[0] = 0.0; w._sense()
    assert w.tmag[0] == 0                                              # otçul yem yoksa hedefsiz (canlıya gitmez)
    # sindirim: etçil yemi %30 verimle yer
    w = small(n=1, source_rate=0.0, digest_penalty=0.7)
    put(w, 0, [10.0, 10.0]); w.energy[:] = 50.0; w.diet[:] = 1.0
    w.fpos, w.fage = np.array([[10.2, 10.0]]), np.zeros(1); w._eat()
    assert abs(w.energy[0] - (50.0 + 20.0 * 0.3)) < 1e-9, w.energy
    print("[17] beslenme: otçul yemi, etçil canlıyı hedefler; etçil yemi %30 verimle sindirir")


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
    print(f"[18] beslenme kalıtımı (ebeveyn ortalaması), yırtıcılı enerji muhasebesi tutuyor (120 sn: {w.kills} öldürme, et kazancı {b['meat']:.0f})")


def test_no_cannibal_farm():
    """Enerji korunumu: kendi yavrusunu yiyen etçil, üreme bedelinden FAZLA kazanmamalı (varsayılan ayarlarda)."""
    c = WorldConfig()
    gain_per_child = (c.meat_frac * c.e_birth) + c.k_meat * 5.0           # 5 kg'lık yavru
    assert gain_per_child < c.cost_split, (gain_per_child, c.cost_split)
    assert gain_per_child < 2 * c.cost_mate, (gain_per_child, 2 * c.cost_mate)
    print(f"[19] çiftlik açığı yok: yavruyu yemek {gain_per_child:.0f} kazandırır, üreme {c.cost_split:.0f} (eşeysiz) / {2*c.cost_mate:.0f} (eşeyli) maliyetli")


def test_carcass():
    # açlıktan ölen canlı vücut enerjisini leş olarak bırakır
    w = small(n=3, source_rate=0.0, body_init=40.0, k_meat=0.0)
    ready(w); w.energy[:] = [100.0, -0.1, 100.0]
    put(w, 1, [12.0, 12.0])
    w.total0 = w.total_energy()
    w._deaths()
    assert w.pop == 2 and len(w.cpos) == 1 and abs(w.cval[0] - 40.0) < 1e-9 and abs(w.carc_created - 40.0) < 1e-9
    assert np.allclose(w.cpos[0], [12.0, 12.0])
    assert abs(w.balance_residual()) < 1e-9
    # ısırıkla ölen: av V=enerji+vücut; saldırgan meat_frac·V alır, kalan (1-meat_frac)·V leş olur
    w = small(n=2, source_rate=0.0, body_init=40.0, k_meat=0.0, digest_penalty=0.5)
    ready(w); w.energy[:] = [100.0, 60.0]; w.diet[:] = [1.0, 0.0]
    put(w, 0, [10.0, 10.0]); put(w, 1, [10.5, 10.0]); w.want[:] = 0; w.want[0, 2] = 1.0
    w.total0 = w.total_energy()
    k = w._bite(); w._deaths(k)
    assert abs(w.gained_meat - 50.0) < 1e-9, w.gained_meat           # 0.5·(60+40)·eff(1)
    assert len(w.cval) == 1 and abs(w.cval[0] - 50.0) < 1e-9
    assert abs(w.balance_residual()) < 1e-9, w.balance_residual()
    # leşi yeme: etçil %100, otçul %30 (digest_penalty 0.7)
    for diet, exp in ((1.0, 50.0), (0.0, 50.0 * 0.3)):
        w = small(n=1, source_rate=0.0, body_init=40.0, digest_penalty=0.7)
        put(w, 0, [10.0, 10.0]); w.energy[:] = 50.0; w.diet[:] = diet
        w.cpos, w.cval, w.cage = np.array([[10.3, 10.0]]), np.array([50.0]), np.zeros(1)
        w.carc_created += 50.0; w.total0 -= 0.0
        w._eat()
        assert abs(w.energy[0] - (50.0 + exp)) < 1e-9 and len(w.cval) == 0 and w.carc_eaten == 1
    # çürüme
    w = small(n=1, source_rate=0.0, carcass_life=5.0)
    w.cpos, w.cval, w.cage = np.array([[3.0, 3.0]]), np.array([30.0]), np.zeros(1)
    w._food_step(6.0)
    assert len(w.cval) == 0 and abs(w.carc_decay - 30.0) < 1e-9
    # etçil leşi hedefler, otçul hedeflemez
    w = small(n=2, source_rate=0.0, body_init=40.0)
    put(w, 0, [10.0, 10.0]); put(w, 1, [35.0, 35.0])
    w.cpos, w.cval, w.cage = np.array([[10.0, 16.0]]), np.array([50.0]), np.zeros(1)
    w.diet[:] = [1.0, 0.0]; w._sense()
    assert abs(w.target[0] - np.pi / 2) < 1e-9 and w.tmag[0] == 1
    w.diet[0] = 0.0; w._sense(); assert w.tmag[0] == 0
    # doğumda vücut enerjisi: eşeysiz (cost 80 - e_birth 40) · body_frac 0.5 = 20
    w = small(n=1, source_rate=0.0, body_init=40.0)
    ready(w); w.energy[:] = 150.0; w.want[0] = [-1, 1, -1]
    w.total0 = w.total_energy()
    w._reproduce()
    assert w.pop == 2 and abs(w.body[-1] - 20.0) < 1e-9 and abs(w.newborn_body - 20.0) < 1e-9
    assert abs(w.balance_residual()) < 1e-9
    print("[20] leş: açlık/ısırık leşi, etçil/otçul sindirimi, çürüme, hedefleme, doğumda vücut enerjisi; korunum ~0")


def test_speed_100():
    w = World(WorldConfig(seed=2))
    t0 = time.time()
    w.advance(20.0)
    dt = time.time() - t0
    s = w.stats()
    assert np.isfinite(w.st.x).all()
    print(f"[8] 100 rastgele canlı, 20 sn simülasyon: {dt:.1f} sn gerçek (1 sn sim = {dt/20:.2f} sn gerçek); "
          f"nüfus {s['pop']}, yem {s['food']}")


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
    print("TAMAM")

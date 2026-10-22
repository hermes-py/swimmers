"""Faz 3 doğrulama testleri:  python test_evolution.py"""
import os, tempfile, time
import numpy as np
from evolution import *
from genome import *
from brain import *


EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ornekler", "ornek_simetrik_balik.json")


def small_cfg():
    return EvalConfig(trials=3, duration=4.0, warmup=0.5, grace=0.5)


def test_zero_brain_does_nothing():
    """Çıktısı hep sıfır olan beyin: ilerleme tam 0 olmalı (dışarıdan itki yok)."""
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    g.brain.W2[:] = 0
    g.brain.b2[:] = 0
    r = evaluate_batch([g], make_trials(rng, 3, 2), small_cfg())
    assert np.allclose(r["progress"], 0, atol=1e-9) and np.allclose(r["energy"], 0)
    print("[1] sıfır çıktılı beyin: ilerleme = 0, enerji = 0")


def test_deterministic_and_parallel_equal():
    rng = np.random.default_rng(1)
    pop = [random_genome(rng) for _ in range(8)]
    tr = make_trials(rng, 3, 2)
    a = evaluate_batch(pop, tr, small_cfg())
    b = evaluate_batch(pop, tr, small_cfg())
    assert np.array_equal(a["fitness"], b["fitness"])
    # batch içindeki konum fark etmemeli (bir birey tek başına = popülasyonda)
    solo = evaluate_batch([pop[5]], tr, small_cfg())
    assert np.allclose(solo["fitness"][0], a["fitness"][5], atol=1e-9), (solo["fitness"], a["fitness"][5])
    ev = Evolution(EvoConfig(pop_size=8), small_cfg(), workers=2, seed=1)
    ev.population = pop
    par = ev.evaluate(tr)
    ev.close()
    assert np.allclose(par["fitness"], a["fitness"], atol=1e-9)
    print("[2] deterministik; tek başına = popülasyonda; 2 işçi = tek süreç")


def test_selection_and_curriculum():
    ev = Evolution(EvoConfig(pop_size=16, elite=2, stage_len=2, curriculum="fixed"), small_cfg(), seed=2)
    stages = []
    prev_best = None
    for _ in range(7):
        stages.append(ev.stage)
        s = ev.step_generation()
        assert len(ev.population) == 16
        for g in ev.population:
            g.body.validate()
        # elit bir sonraki nesilde aynen yer almalı
        champ = s["champion"]
        assert any(np.array_equal(g.brain.W1, champ.brain.W1) for g in ev.population)
    assert stages == [0, 0, 1, 1, 2, 2, 2], stages          # 'dur' aşaması varsayılan KAPALI
    ev = Evolution(EvoConfig(pop_size=8, stage_len=1, curriculum="fixed", stop_command=True), small_cfg(), seed=2)
    assert [min(ev.evo.max_stage, g // 1) for g in range(5)] == [0, 1, 2, 3, 3]
    ev.gen = 5
    assert ev.stage == 3
    print("[3] seçilim geçerli, elit korunuyor, müfredat aşamaları:", stages, "(dur kapalı; açılınca aşama 3 var)")


def test_checkpoint_and_json():
    ev = Evolution(EvoConfig(pop_size=8), small_cfg(), seed=4)
    ev.step_generation()
    d = tempfile.mkdtemp()
    ev.save_checkpoint(os.path.join(d, "c.pkl"))
    ev2 = Evolution(EvoConfig(pop_size=8), small_cfg(), seed=99)
    ev2.load_checkpoint(os.path.join(d, "c.pkl"))
    assert ev2.gen == 1 and len(ev2.population) == 8
    a, b = ev.step_generation(), ev2.step_generation()
    assert np.isclose(a["best"], b["best"]), "devam eden eğitim aynı sonucu vermeli"
    print("[4] checkpoint'ten devam, aynı sonucu üretiyor")


def test_stop_reward():
    """Aşama 3: 'komut yok' segmentinde kımıldamayan canlı, yüzmeye devam edenden daha yüksek puan almalı."""
    rng = np.random.default_rng(0)
    tr = make_trials(rng, 4, 3)
    assert (tr["mag1"] == 0).any() or (tr["mag2"] == 0).any() and tr["mag2"][0] == 0
    stop_all = dict(tr, mag1=np.zeros(4), mag2=np.zeros(4))
    still = random_genome(rng)
    still.brain.W2[:] = 0
    still.brain.b2[:] = 0                                  # hiç hareket etmez
    swimmer = load_json(EXAMPLE) if os.path.exists(EXAMPLE) else None
    cfg = small_cfg()
    r0 = evaluate_batch([still], stop_all, cfg)
    assert np.allclose(r0["progress"], 0, atol=1e-9), "durağan canlıda 'dur' ödülü 0 olmalı (ceza yok)"
    if swimmer is not None:
        r1 = evaluate_batch([swimmer], stop_all, cfg)
        assert r1["progress"].mean() < -0.01, r1["progress"]
        assert r1["fitness"][0] < r0["fitness"][0]
        print(f"[6] 'dur' komutunda: durağan canlı progress=0 (fitness {r0['fitness'][0]:+.3f}), "
              f"yüzmeye devam eden {r1['progress'].mean():+.3f} (fitness {r1['fitness'][0]:+.3f})")
    else:
        print("[6] 'dur' ödülü: durağan canlı progress=0")


def test_fitness_prefers_swimming_over_idling():
    """Eski fitness'ta ön yüze yüzen canlı karışık yönlerde hareketsiz canlıdan KÖTÜ puan alıyordu (-0.195 vs 0).
    Yeni fitness'ta küçük yön aralığında yüzücü açık ara önde; en geniş aralıkta (sadece yanlar/arka) fark sınırlı."""
    if not os.path.exists(EXAMPLE):
        print("[7] atlandı (ornek_simetrik_balik.json yok)")
        return
    sw = load_json(EXAMPLE)
    rng = np.random.default_rng(0)
    still = random_genome(rng); still.brain.W2[:] = 0; still.brain.b2[:] = 0
    h = np.linspace(-3, 3, 4)
    for rng_deg, need in ((22.5, 0.15), (180, -0.4)):     # geniş aralıkta ön yüzücü hâlâ durağanın biraz gerisinde (eski: -0.195)
        o = np.radians(rng_deg) * np.array([1, -1, 0.5, -0.5])
        tr = dict(heading=h, tgt1=h + o, tgt2=h + o, mag1=np.ones(4), mag2=np.ones(4))
        r = evaluate_batch([sw, still], tr, small_cfg() if False else EvalConfig(trials=4))
        gap = r["fitness"][0] - r["fitness"][1]
        print(f"[7] yön aralığı ±{rng_deg}°: yüzücü {r['fitness'][0]:+.3f}, durağan {r['fitness'][1]:+.3f}, fark {gap:+.3f}")
        assert gap > need


def test_adaptive_curriculum_logic():
    ev = Evolution(EvoConfig(pop_size=8, adv_streak=3, adv_min_gens=2, regress_gens=5), small_cfg(), seed=0)
    assert ev.notch == 0 and ev.stage == 0 and ev.dir_range is None
    for _ in range(3):
        ev._update_curriculum(0.5)
    assert ev.notch == 1 and ev.stage == 1 and abs(ev.dir_range - np.pi / 8) < 1e-9
    ev._update_curriculum(0.5); ev._update_curriculum(0.01); ev._update_curriculum(0.5)
    assert ev.notch == 1, "streak kesilince ilerlememeli"
    for _ in range(5):
        ev._update_curriculum(0.0)
    assert ev.notch == 0, "uzun süre çok kötüyse bir kademe geri alınmalı"
    for _ in range(80):
        ev._update_curriculum(1.0)
    assert ev.notch == 9 and ev.stage == 2 and ev.dir_range is None, (ev.notch, ev.stage)   # dur kapalı: tavan 9
    ev = Evolution(EvoConfig(pop_size=8, adv_streak=3, adv_min_gens=2, stop_command=True), small_cfg(), seed=0)
    for _ in range(80):
        ev._update_curriculum(1.0)
    assert ev.notch == 10 and ev.stage == 3
    tr = make_trials(np.random.default_rng(1), 4, 1, dir_range=np.pi / 4)
    off = np.abs(((tr["tgt1"] - tr["heading"]) + np.pi) % (2 * np.pi) - np.pi)
    assert off.max() <= np.pi / 4 + 1e-9 and abs(off[0] - np.pi / 4) < 1e-9
    print("[8] adaptif müfredat: ilerleme, streak kesilmesi, geri alma, 10. kademe ve yön aralığı doğru")


def test_seeded_and_frozen_body():
    from genome import blank_body, Genome as G, random_brain
    rng = np.random.default_rng(3)
    design = G(blank_body(), random_brain(rng))
    ev = Evolution(EvoConfig(pop_size=12, freeze_body=True, immigrants=0.1), small_cfg(), workers=1, seed=1, seed_genome=design)
    for _ in range(3):
        ev.step_generation()
    for g in ev.population:
        for f in ("parent", "theta", "length", "mass", "width", "tau", "damp", "lo", "hi"):
            assert np.array_equal(getattr(g.body, f), getattr(design.body, f)), f
    assert len({g.brain.W1.tobytes() for g in ev.population}) > 6, "beyinler çeşitli olmalı"
    from genome import MutationConfig, FIELDS
    ev2 = Evolution(EvoConfig(pop_size=12, freeze_body=False), small_cfg(), MutationConfig(p_gene=0.5), seed=1, seed_genome=design)
    ident = lambda g: g.body.n == design.body.n and all(np.array_equal(getattr(g.body, f), getattr(design.body, f)) for f in FIELDS)
    same = sum(ident(g) for g in ev2.population)
    assert 5 <= same <= 7, same                       # yarısı aynı vücut, yarısı mutasyonlu
    print(f"[9] tasarımdan eğitim: sabit vücutta {len(ev.population)} bireyin hepsi tasarımla aynı; "
          f"evrimleşende {same}/12 aynı, kalanı mutasyonlu")


def test_penalty_gating():
    """Cezalar ilerlemeyen canlıda küçülür; yüzen canlıda tam güçtedir."""
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    tr = make_trials(rng, 3, 0)
    on = evaluate_batch([g], tr, EvalConfig(trials=3, duration=4.0, warmup=0.5, grace=0.5))
    off = evaluate_batch([g], tr, EvalConfig(trials=3, duration=4.0, warmup=0.5, grace=0.5, gate_penalties=False))
    assert np.allclose(on["progress"], off["progress"])
    assert on["fitness"][0] >= off["fitness"][0] - 1e-9, "kapı cezayı yalnızca azaltabilir"
    if os.path.exists(EXAMPLE):
        sw = load_json(EXAMPLE)
        a = evaluate_batch([sw], make_trials(rng, 4, 0), EvalConfig(trials=4))
        b = evaluate_batch([sw], make_trials(np.random.default_rng(0), 4, 0), EvalConfig(trials=4, gate_penalties=False))
        assert a["progress"].mean() > 0.3
    print("[10] ceza kapısı: ilerleme aynı, fitness yalnızca ilerlemeyen canlıda cezadan kurtuluyor")


def test_stall_boost():
    ev = Evolution(EvoConfig(pop_size=8, stall_gens=3, stall_window=2, boost_step_gens=2), small_cfg(), seed=0)
    assert ev.mut_scale == 1.0
    for f in (0.1, 0.1, 0.1, 0.1, 0.1, 0.1):
        ev._track_stall(f)
    assert ev.mut_scale > 1.0, (ev._stall, ev.mut_scale)
    base = ev.mut.brain_sigma
    assert ev._scaled_mut(ev.mut_scale).brain_sigma > base
    for _ in range(40):
        ev._track_stall(0.1)
    assert ev.mut_scale == ev.evo.boost_max
    ev._track_stall(0.5); ev._track_stall(0.5)
    assert ev.mut_scale == 1.0, "iyileşme olunca çarpan sıfırlanmalı"
    ev._stall = 99
    ev.fitness = np.arange(8, dtype=float)
    ev._breed()                      # boost açıkken üreme hatasız ve geçerli
    assert len(ev.population) == 8
    for g in ev.population:
        g.body.validate()
    ev2 = Evolution(EvoConfig(pop_size=8, stall_boost=False), small_cfg(), seed=0)
    ev2._stall = 99
    assert ev2.mut_scale == 1.0
    print("[11] takılma algılama: çarpan artıyor, iyileşmede sıfırlanıyor, boost'lu üreme geçerli")


def test_learning_signal():
    """Kısa evrim: en iyi fitness ilk nesile göre artmalı."""
    ev = Evolution(EvoConfig(pop_size=32, stage_len=1000, curriculum="fixed"), EvalConfig(trials=3, duration=5.0), seed=3)
    first = ev.step_generation()["best"]
    t0 = time.time()
    for _ in range(18):
        last = ev.step_generation()["best"]
    print(f"[5] 19 nesil: en iyi fitness {first:+.3f} -> {last:+.3f}  ({time.time()-t0:.0f} sn)")
    assert last > first + 0.1


if __name__ == "__main__":
    test_zero_brain_does_nothing()
    test_deterministic_and_parallel_equal()
    test_selection_and_curriculum()
    test_checkpoint_and_json()
    test_stop_reward()
    test_fitness_prefers_swimming_over_idling()
    test_adaptive_curriculum_logic()
    test_seeded_and_frozen_body()
    test_penalty_gating()
    test_stall_boost()
    test_learning_signal()
    print("TAMAM")

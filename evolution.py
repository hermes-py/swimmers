"""
Faz 3 - Evrim döngüsü: fitness, seçilim, mutasyon/çaprazlama, çok çekirdekli değerlendirme.

Değerlendirme
-------------
Her birey K farklı başlangıç koşulunda (rastgele gövde açısı + hedef yön dizisi) simüle edilir.
Popülasyon x deneme tek bir numpy batch'i olarak çalışır; popülasyon ayrıca işçi süreçlere bölünür.
Bir denemenin yarısında hedef yön DEĞİŞİR (yön değiştirme yeteneği ölçülür).

Fitness (deneme başına, aşağıdaki terimlerin toplamı):
    + hedef yönde ilerleme        (COM yer değiştirmesinin hedefe izdüşümü / süre)   <- asıl ödül
    - yanal sapma                 (hedefe dik yer değiştirme)                        w_lat
    - enerji                      (ortalama komut^2)                                  w_energy
    - fiziksel kararsızlık        (ortalama gövde dönme hızı)                         w_spin
    - titreme                     (komutların adımlar arası değişimi)                 w_jitter
Yerinde dönme / titreme: COM yer değiştirmesi ~0 olduğundan ilerleme ödülü yok, cezalar var.
Birey fitness'ı = 0.7 * ortalama + 0.3 * en kötü deneme (farklı koşullarda tutarlılık).

Müfredat: zorluk "aşama" ile artar (stage_len nesilde bir):
    0: hedef = gövdenin önü   1: 4 yön (ön/arka/sağ/sol)   2: 8 yön + deneme ortasında yön değişimi
    3: aşama 2 + "KOMUT YOK" segmentleri (hedef vektörü 0): canlı durmayı öğrenir.
       Durma segmentinde ödül = -stop_scale*(ortalama hız) -> kımıldamadan durmak en yüksek puan.
       (stop_scale=1 iken popülasyon 'hiç kıpırdamama'ya çöküyordu: yüzmek 0.65*+0.3 -0.35*0.3 ~ 0.)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass, asdict, field, replace

import numpy as np

from genome import (Genome, MutationConfig, build_bodies, crossover, mutate, mutate_body, mutate_brain,
                    random_brain, random_genome, save_json)
from brain import stack_brains, act
from physics import (PHYSICS_VERSION, PhysicsParams, center_of_mass, constraint_error, init_state, motor_mask, step)


# --------------------------------------------------------------------------- #
#  Ayarlar
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    duration: float = 8.0       # saniye
    trials: int = 4             # birey başına farklı başlangıç koşulu
    warmup: float = 1.0         # ilk saniye ölçüme dahil değil
    grace: float = 1.0          # yön değişiminden sonra ölçüme dahil olmayan süre
    control_dt: float = 0.02    # beyin kontrol periyodu (sn) -> 50 Hz; fizik adımından bağımsız
    w_lat: float = 0.2          # hedefe dik sapma cezası
    stop_scale: float = 0.5     # 'komut yok' segmentinde hız cezası çarpanı (1.0 -> durma, yüzmeyi tamamen sıfırlıyordu)
    neg_scale: float = 0.25     # hedefin TERSİNE ilerleme cezası = bu çarpanla yumuşatılır (durmak yüzmekten iyi olmasın)
    w_energy: float = 0.02
    w_spin: float = 0.01
    w_jitter: float = 0.03
    worst_weight: float = 0.15  # fitness = (1-w)*ortalama + w*en kötü
    bad_fitness: float = -5.0   # patlayan/NaN bireyler
    gate_penalties: bool = True  # enerji/dönme/titreme cezaları canlı gerçekten ilerleyince tam güce çıkar
    gate_floor: float = 0.25     # ilerlemeyen canlıda cezaların kalan oranı (ceza 'kıpırdama'yı tek başına seçtirmesin)
    gate_speed: float = 0.3      # bu ilerleme hızında (m/s) cezalar tam uygulanır


@dataclass
class EvoConfig:
    pop_size: int = 64
    elite: int = 3
    tournament: int = 3
    p_crossover: float = 0.4
    immigrants: float = 0.04    # her nesil rastgele yeni birey oranı
    stage_len: int = 25         # müfredat aşaması başına nesil
    init_limbs: tuple = (4, 10)
    max_stage: int = 3
    curriculum: str = "adaptive"   # adaptive: başarıya göre zorluk artar/azalır | fixed: stage_len nesilde bir
    adv_fitness: float = 0.15       # şampiyon bunu aşarsa (adv_streak nesil üst üste) zorluk +1
    adv_streak: int = 3
    adv_min_gens: int = 5           # bir zorlukta en az bu kadar nesil kal
    regress_fitness: float = 0.03   # şampiyon regress_gens nesil boyunca bunun altında kalırsa zorluk -1
    regress_gens: int = 25
    freeze_body: bool = False       # True: vücut sabit, sadece beyin evrimleşir
    p_stop: float = 0.35        # aşama 3'te her segmentin 'dur' olma olasılığı
    legacy_spine: bool = False   # True: rastgele/göçmen bireyler ESKİ omurga modunda doğar (varsayılan: gövde yönü = kafa)
    stop_command: bool = False  # 'komut yok = dur' (kademe 10 / aşama 3) şimdilik KAPALI; True -> eski davranış
    stall_boost: bool = True    # takılma algılanınca mutasyon güçlenir, göçmenler agresifleşir
    stall_gens: int = 15        # şampiyon hareketli ortalaması bu kadar nesil iyileşmezse 'takıldı' say
    stall_window: int = 5       # şampiyon fitness'ı bu kadar nesillik ortalamayla izlenir (gürültü için)
    stall_eps: float = 0.01     # bundan az iyileşme 'iyileşme' sayılmaz
    boost_max: float = 3.0      # mutasyon büyüklüğünün en çok çarpanı
    boost_step_gens: int = 10   # takılma sürdükçe çarpan bu kadar nesilde +1 artar


# --------------------------------------------------------------------------- #
#  Denemeler ve değerlendirme
# --------------------------------------------------------------------------- #
NOTCH_MAX = 10   # 0: sadece ön | 1..8: yön aralığı notch*22.5° | 9: 8 yön + yön değişimi | 10: + 'dur' komutu


def notch_to_stage(notch: int) -> int:
    return 0 if notch <= 0 else 1 if notch <= 8 else 2 if notch == 9 else 3


def make_trials(rng: np.random.Generator, K: int, stage: int, p_stop: float = 0.35, dir_range=None) -> dict:
    """dir_range (rad): aşama 1'de hedef yön gövde önünden en fazla bu kadar sapar (uniform ±dir_range).
    Her nesilde bir deneme sınırdadır (±dir_range) -> eğitim cephesini zorlar. None -> eski 4 yön."""
    h = rng.uniform(-np.pi, np.pi, K)
    if stage <= 0:
        o1 = np.zeros(K)
        o2 = o1.copy()
    elif stage == 1:
        if dir_range is None:
            o1 = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2], K)
        else:
            o1 = rng.uniform(-dir_range, dir_range, K)
            o1[0] = dir_range * rng.choice([-1.0, 1.0])
        o2 = o1.copy()
    else:
        o1 = rng.integers(0, 8, K) * np.pi / 4
        o2 = rng.integers(0, 8, K) * np.pi / 4
    mag1, mag2 = np.ones(K), np.ones(K)
    if stage >= 3:
        mag1 = (rng.random(K) >= p_stop).astype(float)
        mag2 = (rng.random(K) >= p_stop).astype(float)
        if K >= 2:                       # en az bir 'dur' segmenti garanti
            mag2[0] = 0.0
            mag1[0] = 1.0
    return dict(heading=h, tgt1=h + o1, tgt2=h + o2, mag1=mag1, mag2=mag2)


def evaluate_batch(genomes: list[Genome], trials: dict, cfg: EvalConfig,
                   pp: PhysicsParams = None) -> dict:
    """Dönüş: fitness (P,), per_trial (P,K), ve ölçüm bileşenleri."""
    pp = pp or PhysicsParams()
    P, K = len(genomes), len(trials["heading"])
    rep = [g for g in genomes for _ in range(K)]
    bd = build_bodies(rep)
    bb = stack_brains([g.brain for g in rep])
    heading = np.tile(trials["heading"], P)
    tg1, tg2 = np.tile(trials["tgt1"], P), np.tile(trials["tgt2"], P)
    mg1 = np.tile(trials.get("mag1", np.ones(K)), P)
    mg2 = np.tile(trials.get("mag2", np.ones(K)), P)
    st = init_state(bd, heading=heading)
    B = bd.B
    steps = int(round(cfg.duration / pp.dt))
    sw = steps // 2
    w0 = int(round(cfg.warmup / pp.dt))
    g1 = sw + int(round(cfg.grace / pp.dt))
    assert w0 < sw < g1 < steps, "duration/warmup/grace uyumsuz"
    every = max(1, int(round(cfg.control_dt / pp.dt)))
    mm = motor_mask(bd)
    n_mot = np.maximum(mm.sum(1), 1)

    ctrl = np.zeros((bd.B, bd.N))
    energy = np.zeros(B)
    jitter = np.zeros(B)
    spin = np.zeros(B)
    n_ctrl = 0
    snap = {}
    with np.errstate(all="ignore"):
        for s in range(steps):
            if s % every == 0:
                new = act(st, bd, bb, np.where(s < sw, tg1, tg2), np.where(s < sw, mg1, mg2))
                jitter += np.abs(new - ctrl).sum(1) / n_mot
                ctrl = new
                energy += (ctrl ** 2).sum(1) / n_mot
                n_ctrl += 1
            step(st, bd, ctrl, pp)
            spin += np.abs(st.omega)
            if s + 1 in (w0, sw, g1, steps):
                snap[s + 1] = center_of_mass(st, bd)
        err = constraint_error(st, bd)
    energy /= n_ctrl
    jitter /= n_ctrl
    spin /= steps

    def seg(a, b, tgt, mag):
        T = (b - a) * pp.dt
        d = (snap[b] - snap[a]) / T
        u = np.stack([np.cos(tgt), np.sin(tgt)], -1)
        prog = np.sum(d * u, -1)
        lat = np.abs(d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0])
        prog = np.where(prog > 0, prog, cfg.neg_scale * prog)   # ters yöne gitmek yumuşak cezalı
        stop = mag < 0.5                          # komut yok: kımıldamamak ödüllü
        prog = np.where(stop, -cfg.stop_scale * np.linalg.norm(d, axis=-1), prog)
        lat = np.where(stop, 0.0, lat)
        return prog, lat

    p1, l1 = seg(w0, sw, tg1, mg1)
    p2, l2 = seg(g1, steps, tg2, mg2)
    prog, lat = 0.5 * (p1 + p2), 0.5 * (l1 + l2)
    pen = cfg.w_energy * energy + cfg.w_spin * spin + cfg.w_jitter * jitter
    if cfg.gate_penalties:      # ilerlemeyen canlıda cezalar küçük: 'kıpırdamamak' çekim noktası olmasın
        pen = pen * np.clip(cfg.gate_floor + (1 - cfg.gate_floor) * np.maximum(prog, 0) / cfg.gate_speed, cfg.gate_floor, 1.0)
    f = prog - cfg.w_lat * lat - pen
    bad = ~np.isfinite(f) | (err > 0.5)
    f = np.where(bad, cfg.bad_fitness, f)
    f = np.maximum(f, cfg.bad_fitness)

    ft = f.reshape(P, K)
    fit = (1 - cfg.worst_weight) * ft.mean(1) + cfg.worst_weight * ft.min(1)

    def r(x):
        return np.nan_to_num(x).reshape(P, K)

    return dict(fitness=fit, per_trial=ft, progress=r(prog), lateral=r(lat), energy=r(energy),
                spin=r(spin), jitter=r(jitter))


def _eval_chunk(args):
    genomes, trials, cfg = args
    return evaluate_batch(genomes, trials, cfg)


# --------------------------------------------------------------------------- #
#  Evrim
# --------------------------------------------------------------------------- #
class Evolution:
    def __init__(self, evo: EvoConfig = None, ev: EvalConfig = None, mut: MutationConfig = None,
                 workers: int = 1, seed: int = 0, seed_genome: Genome = None):
        self.evo = evo or EvoConfig()
        if seed_genome is not None:        # tasarımdan eğitimde mod tasarımdan gelir
            self.evo = replace(self.evo, legacy_spine=bool(seed_genome.body.legacy_spine))
        self.ev = ev or EvalConfig()
        self.mut = mut or MutationConfig()
        self.workers = max(1, workers)
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.notch = 0              # adaptif müfredat: 0..NOTCH_MAX
        self._streak = 0            # şampiyon eşiği üst üste kaç nesildir aşıyor
        self._low = 0               # şampiyon kaç nesildir regress eşiğinin altında
        self._at_notch = 0          # mevcut zorlukta geçirilen nesil
        self._champ_hist = []       # son şampiyon fitness'ları (takılma izleme)
        self._hi = -np.inf          # hareketli ortalamanın görülen en yükseği
        self._stall = 0             # kaç nesildir iyileşme yok
        self.seed_genome = seed_genome
        lo, hi = self.evo.init_limbs
        P = self.evo.pop_size
        if seed_genome is None:
            self.population = [random_genome(self.rng, n_min=lo, n_max=hi, legacy_spine=self.evo.legacy_spine) for _ in range(P)]
        else:
            self.population = self._seeded_population(seed_genome)
        self.fitness = np.full(self.evo.pop_size, -np.inf)
        self.history = []          # nesil başına özet
        self.best = None           # (fitness, Genome) tüm zamanların en iyisi (aşama bazında)
        self._pool = None

    # --- değerlendirme ---
    @property
    def notch_max(self) -> int:
        return NOTCH_MAX if self.evo.stop_command else NOTCH_MAX - 1

    @property
    def stage(self) -> int:
        if self.evo.curriculum == "adaptive":
            return notch_to_stage(min(self.notch, self.notch_max))
        top = self.evo.max_stage if self.evo.stop_command else min(self.evo.max_stage, 2)
        return min(top, self.gen // self.evo.stage_len)

    def _get_pool(self):
        if self._pool is None and self.workers > 1:
            self._pool = mp.Pool(self.workers)
        return self._pool

    def set_workers(self, n: int):
        """İşçi sayısını değiştir (turbo modda artırmak için)."""
        n = max(1, int(n))
        if n != self.workers:
            self.close()
            self.workers = n

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def evaluate(self, trials: dict) -> dict:
        P = len(self.population)
        if self.workers == 1:
            return evaluate_batch(self.population, trials, self.ev)
        # boyuta göre sırala, işçilere dağıt (benzer dolgu boyutu, dengeli yük)
        order = np.argsort([g.body.n for g in self.population])
        groups = [order[i::self.workers] for i in range(self.workers)]
        res = self._get_pool().map(_eval_chunk, [([self.population[i] for i in g], trials, self.ev) for g in groups])
        out = {}
        for key in res[0]:
            arr = np.zeros((P,) + res[0][key].shape[1:])
            for g, r in zip(groups, res):
                arr[g] = r[key]
            out[key] = arr
        return out

    def _seeded_population(self, sg: Genome) -> list:
        """Bir tasarımdan başlangıç popülasyonu. Vücut sabitse hepsi aynı vücut, farklı beyinler.
        Değilse yarısı aynı vücut, yarısı hafif mutasyonlu vücut. Tasarımın kendi beyni (varsa) 1. bireydir."""
        P = self.evo.pop_size
        pop = [sg.copy()]
        for k in range(1, P):
            body = sg.body.copy()
            if not self.evo.freeze_body and k >= P // 2:
                body = mutate_body(body, self.rng, self.mut)
            if k <= max(1, P // 8):
                brain = mutate_brain(sg.brain, self.rng, self.mut)       # tasarımın beyninin türevleri
            else:
                brain = random_brain(self.rng)
            pop.append(Genome(body, brain))
        for g in pop:
            g.meta = {"seeded": True, "physics": PHYSICS_VERSION}
        return pop

    # --- adaptif müfredat ---
    @property
    def dir_range(self):
        """Aşama 1'de geçerli yön aralığı (rad); diğer aşamalarda None."""
        if self.evo.curriculum != "adaptive" or notch_to_stage(self.notch) != 1:
            return None
        return min(self.notch, 8) * np.pi / 8

    def _update_curriculum(self, champ_fitness: float):
        """Şampiyon yeterince iyiyse zorluğu artır; uzun süre çok kötüyse bir kademe geri al."""
        if self.evo.curriculum != "adaptive":
            return
        self._at_notch += 1
        self._streak = self._streak + 1 if champ_fitness >= self.evo.adv_fitness else 0
        self._low = self._low + 1 if champ_fitness < self.evo.regress_fitness else 0
        if self._streak >= self.evo.adv_streak and self._at_notch >= self.evo.adv_min_gens and self.notch < self.notch_max:
            self.notch += 1
            self._streak = self._low = self._at_notch = 0
            self._reset_stall()
        elif self._low >= self.evo.regress_gens and self.notch > 0:
            self.notch -= 1
            self._streak = self._low = self._at_notch = 0
            self._reset_stall()

    # --- takılma algılama ve mutasyon güçlendirme ---
    def _reset_stall(self):
        """Zorluk değişince fitness ölçeği değişir: önceki karşılaştırma geçersiz."""
        self._champ_hist, self._hi, self._stall = [], -np.inf, 0

    def _track_stall(self, champ_fitness: float):
        w = self.evo.stall_window
        self._champ_hist = (self._champ_hist + [champ_fitness])[-w:]
        if len(self._champ_hist) < w:
            return
        r = float(np.mean(self._champ_hist))
        if r > self._hi + self.evo.stall_eps:
            self._hi, self._stall = r, 0
        else:
            self._stall += 1

    @property
    def mut_scale(self) -> float:
        if not self.evo.stall_boost or self._stall < self.evo.stall_gens:
            return 1.0
        return min(self.evo.boost_max, 1.5 + (self._stall - self.evo.stall_gens) / self.evo.boost_step_gens)

    def _scaled_mut(self, k: float) -> MutationConfig:
        if k <= 1.0:
            return self.mut
        m = self.mut
        r = min(k, 2.0)          # olasılıkları en çok 2 kat, adım büyüklüğünü k kat artır
        return replace(m, p_gene=min(0.5, m.p_gene * r), p_add=min(0.3, m.p_add * r), p_remove=min(0.3, m.p_remove * r),
                       p_mirror=min(0.3, m.p_mirror * r), brain_rate=min(0.5, m.brain_rate * r), brain_sigma=m.brain_sigma * k)

    # --- seçilim ve üreme ---
    def _tournament(self) -> Genome:
        idx = self.rng.integers(0, len(self.population), self.evo.tournament)
        return self.population[idx[np.argmax(self.fitness[idx])]]

    def _breed(self):
        P = self.evo.pop_size
        order = np.argsort(-self.fitness)
        nxt = [self.population[i].copy() for i in order[: self.evo.elite]]
        k = self.mut_scale
        mut = self._scaled_mut(k)
        n_imm = int(round((max(self.evo.immigrants, 0.10) if k > 1 else self.evo.immigrants) * P))
        lo, hi = self.evo.init_limbs
        while len(nxt) < P - n_imm:
            a = self._tournament()
            if self.rng.random() < self.evo.p_crossover:
                child = crossover(a, self._tournament(), self.rng, self.evo.freeze_body)
            else:
                child = a.copy()
            nxt.append(mutate(child, self.rng, mut, self.evo.freeze_body))
        heavy = self._scaled_mut(self.evo.boost_max)
        while len(nxt) < P:
            if k > 1 and len(nxt) % 2 == 0:       # takılmada: göçmenlerin yarısı elitin AĞIR mutasyonlu kopyası
                src = self.population[order[int(self.rng.integers(0, self.evo.elite))]].copy()
                nxt.append(mutate(src, self.rng, heavy, self.evo.freeze_body))
            elif self.seed_genome is not None and self.evo.freeze_body:
                nxt.append(Genome(self.seed_genome.body.copy(), random_brain(self.rng)))
            else:
                nxt.append(random_genome(self.rng, n_min=lo, n_max=hi, legacy_spine=self.evo.legacy_spine))
        self.population = nxt

    def step_generation(self) -> dict:
        """Bir nesil: değerlendir -> kaydet -> sonraki nesli üret. Özet sözlüğü döner."""
        t0 = time.time()
        trials = make_trials(self.rng, self.ev.trials, self.stage, self.evo.p_stop, self.dir_range)
        res = self.evaluate(trials)
        self.fitness = res["fitness"]
        i = int(np.argmax(self.fitness))
        self._track_stall(float(self.fitness[i]))
        champ = self.population[i].copy()
        champ.meta = {"generation": self.gen, "stage": self.stage, "fitness": float(self.fitness[i]),
                      "progress": float(res["progress"][i].mean()), "energy": float(res["energy"][i].mean()),
                      "n_limbs": champ.body.n, "physics": PHYSICS_VERSION}
        dr = self.dir_range
        summary = dict(gen=self.gen, stage=self.stage, notch=self.notch, notch_max=self.notch_max,
                       stall=self._stall, mut_scale=self.mut_scale,
                       dir_range_deg=None if dr is None else float(np.degrees(dr)), best=float(self.fitness[i]),
                       mean=float(self.fitness.mean()), median=float(np.median(self.fitness)),
                       best_progress=float(res["progress"][i].mean()),
                       best_limbs=champ.body.n, mean_limbs=float(np.mean([g.body.n for g in self.population])),
                       seconds=time.time() - t0, champion=champ)
        self.history.append({k: v for k, v in summary.items() if k != "champion"})
        self.last_result = res
        self._update_curriculum(float(self.fitness[i]))
        self._breed()
        self.gen += 1
        return summary

    # --- kayıt ---
    def save_checkpoint(self, path: str):
        """Eğitimi sürdürmek için (dahili format). İnsan okunur kayıtlar için save_json kullan."""
        with open(path, "wb") as f:
            pickle.dump(dict(gen=self.gen, population=self.population, fitness=self.fitness,
                             history=self.history, physics=PHYSICS_VERSION, notch=self.notch, evo=asdict(self.evo), ev=asdict(self.ev),
                             rng=self.rng.bit_generator.state), f)

    def load_checkpoint(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.gen, self.population, self.fitness, self.history = d["gen"], d["population"], d["fitness"], d["history"]
        self.notch = min(d.get("notch", 0), self.notch_max)
        self.loaded_physics = d.get("physics", 1)   # eski sürümle eğitilmişse popülasyon yeniden uyum sağlayacak
        self.rng.bit_generator.state = d["rng"]

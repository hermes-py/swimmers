"""
Chapter 2 / Aşama 1 - Ortak su dünyası.

* Dünya SARMAL (torus): kenarlar yok, bir kenardan çıkan öbür kenardan girer. Mesafeler en kısa yoldan ölçülür.
* Tüm canlılar TEK fizik batch'inde ilerler (physics.step bağımsız canlılar içindir; canlılar arası fiziksel etkileşim yok).
  Etkileşimler (yeme, algı, ölüm) fizik adımının dışında, her kontrol adımında (50 Hz) başlar.
* Yem: birkaç yem KAYNAĞI sabit hızla yem üretir, kaynaklar yavaşça kayar. Yem bir süre sonra çürür.
* Enerji: hareket = gerçek MEKANİK GÜÇ (|eklem torku x eklem hızı|) x küçük katsayı; ayrıca kütleyle artan çok küçük bazal
  gider. Yem enerji verir. Enerji <= 0 -> ölüm.
* Algı (beyin v2): "hedef yön" girişi = en yakın yemin yönü (menzil dışı = 0); ayrıca yem yakınlığı, kendi enerjisi ve en
  yakın canlının yönü/yakınlığı/enerjisi.
* Üreme (aşama 2): beynin iki isteği var. EŞEYSİZ: tek başına, ebeveyn cost_split öder (pahalı). EŞEYLİ: iki canlı da isterse ve
  kafaları r_mate içindeyse her biri cost_mate öder, yavru iki ebeveynin çaprazı. Yavru e_birth enerjiyle doğar, vücut ve
  beyin mutasyona uğrar. Kural yok: yalnızca enerji ve (kısa) olgunlaşma süresi sınırlar. Batch gerekirse genişler.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, replace

import numpy as np

from brain import BrainBatch, act2, stack_brains
from genome import N_IN_V2, Genome, load_json, MutationConfig, build_bodies, crossover, mutate, random_genome, upgrade_brain
from evolution import EvalConfig
from physics import (MAX_LIMBS, PHYSICS_VERSION, Bodies, PhysicsParams, State, center_of_mass, com_velocity, init_state,
                     motor_mask, step)


@dataclass
class WorldConfig:
    size: float = 40.0            # torus kenarı (m)
    n_init: int = 100             # başlangıç nüfusu
    init_limbs: tuple = (4, 10)   # rastgele başlangıç canlılarının uzuv sayısı aralığı (batch genişliği = üst sınır + 2)
    n_sources: int = 3
    source_rate: float = 0.5      # kaynak başına yem/sn
    source_sigma: float = 2.5     # yemin kaynak çevresindeki saçılımı (m)
    source_drift: float = 0.08    # kaynak hızı (m/sn)
    food_energy: float = 20.0
    food_life: float = 120.0      # sn; sonra yem çürür
    food_max: int = 300           # dünyadaki en çok yem (üretim doluluktan sonra durur)
    eat_radius: float = 0.8       # kafa-yem mesafesi (m)
    sense_radius: float = 12.0    # algı menzili (m)
    e_init: float = 150.0
    e_max: float = 300.0
    c_move: float = 0.003         # enerji / joule (mekanik iş): yılan (27 W) ~0.08/sn, balık (266 W) ~0.8/sn (0.01'de yılan tekeli oluşuyordu)
    c_basal: float = 0.01         # enerji / (sn·kg) (0.05'te rastgele başlangıç nüfusu yem bulamadan eriyordu)
    # üreme
    e_birth: float = 80.0         # yavrunun doğum enerjisi (E0)
    cost_split: float = 160.0     # eşeysiz üreme: ebeveynin ödediği (2·E0) - fark kaybolur
    cost_mate: float = 60.0       # eşeyli üreme: HER ebeveynin ödediği (0.75·E0)
    r_mate: float = 2.0           # eşleşme için kafa-kafa mesafesi (m)
    min_age: float = 5.0          # olgunlaşma süresi (sn): yeni doğan hemen üremesin
    want_thr: float = 0.5         # beyin isteği bu eşiği aşarsa canlı üremek ister
    max_limbs: int = MAX_LIMBS
    min_pop: int = 0              # >0: nüfus bunun altına inerse hayatta kalanların ağır mutasyonlu kopyaları eklenir (0 = kapalı)
    mut: MutationConfig = field(default_factory=MutationConfig)
    want_bias: float = 0.0        # rastgele başlangıç beyinlerinde üreme/ısırma çıkışlarının bias'ı (negatif: rastgele çoğalıp enerji yakmasınlar)
    # yırtıcılık (aşama 3): beslenme genotipi diet in [0,1] (0 = otçul, 1 = etçil)
    carnivore_frac: float = 0.1   # başlangıçta etçil olanların oranı
    diet_sigma: float = 0.05      # yavruda diet mutasyonu (gauss)
    digest_penalty: float = 0.7   # uzmanlaşma bedeli: yem verimi = 1 - p·diet, et verimi = 1 - p·(1-diet)
    cost_bite: float = 5.0        # her ısırma DENEMESİNİN enerjisi
    bite_cd: float = 3.0          # iki deneme arası bekleme (sn)
    r_bite: float = 0.7           # kafa-kafa ısırma menzili (m)
    k_meat: float = 0.0           # ek 'biyokütle' enerjisi/kg. DİKKAT: >0 enerjiyi yoktan yaratır (kendi yavrusunu yiyen etçil bölünme bedelinden fazla kazanır)
    meat_frac: float = 0.8        # av başına kazanç = meat_frac · av enerjisi (+ k_meat · kütle); <=1 iken enerji korunur
    # leş (enerji korunumlu): doğumda ebeveynin kaybettiği enerjinin body_frac'ı canlının 'vücut enerjisi' olur; ölünce
    # vücut enerjisi (+ ısırıkta avın artan enerjisi) leş olur, carcass_life sn sonra çürür. Etçiller leşi de yer.
    body_init: float = 40.0       # başlangıç canlılarının vücut enerjisi
    body_frac: float = 0.5
    carcass_life: float = 60.0
    brain_gain: float = 1.0       # rastgele başlangıç beyinlerinin ağırlık çarpanı (>1: ilk nesiller daha hareketli)
    seed: int = 0


def select_bodies(bd: Bodies, idx) -> Bodies:
    return Bodies(**{f.name: (None if getattr(bd, f.name) is None else getattr(bd, f.name)[idx])
                     for f in dataclasses.fields(bd)})


def select_state(st: State, idx) -> State:
    g = lambda a: None if a is None else a[idx]
    return State(x=st.x[idx], v=st.v[idx], phi=st.phi[idx], omega=st.omega[idx], t=st.t,
                 theta=g(st.theta), jvel=g(st.jvel), tau=g(st.tau))


def concat_bodies(a: Bodies, b: Bodies) -> Bodies:
    return Bodies(**{f.name: np.concatenate([getattr(a, f.name), getattr(b, f.name)]) for f in dataclasses.fields(a)})


def concat_state(a: State, b: State) -> State:
    cat = lambda p, q: np.concatenate([p, q])
    return State(x=cat(a.x, b.x), v=cat(a.v, b.v), phi=cat(a.phi, b.phi), omega=cat(a.omega, b.omega), t=a.t,
                 theta=cat(a.theta, b.theta), jvel=cat(a.jvel, b.jvel), tau=cat(a.tau, b.tau))


def concat_brains(a: BrainBatch, b: BrainBatch) -> BrainBatch:
    return BrainBatch(np.concatenate([a.W1, b.W1]), np.concatenate([a.b1, b.b1]), np.concatenate([a.W2, b.W2]),
                      np.concatenate([a.b2, b.b2]))


def select_brains(bb: BrainBatch, idx) -> BrainBatch:
    return BrainBatch(bb.W1[idx], bb.b1[idx], bb.W2[idx], bb.b2[idx])


class World:
    def __init__(self, cfg: WorldConfig = None, genomes: list = None):
        self.cfg = cfg or WorldConfig()
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.L = float(c.size)
        if genomes is None:
            lo, hi = c.init_limbs
            genomes = [random_genome(self.rng, n_min=lo, n_max=hi, brain_v2=True) for _ in range(c.n_init)]
            for g in genomes:
                g.brain.W1 *= c.brain_gain
                g.brain.W2 *= c.brain_gain
                g.brain.b2[-3:] = c.want_bias
        genomes = [g if g.brain.W1.shape[0] == N_IN_V2 else Genome(g.body, upgrade_brain(g.brain), g.meta) for g in genomes]
        self.n_points = max(c.init_limbs[1], max(g.body.n for g in genomes)) + 2
        B = len(genomes)
        self.genomes = list(genomes)
        self.ids = np.arange(B)
        self.next_id = B
        self.born = np.zeros(B)
        self.gen = np.zeros(B, dtype=int)
        self.want = np.zeros((B, 3))
        self.fprox = np.zeros(B)
        self.diet = np.array([g.meta.get("diet", float(self.rng.random() < c.carnivore_frac)) for g in genomes], float)
        self.lin = np.array([int(g.meta.get("lineage", 0)) for g in genomes], dtype=int)
        self.bite_t = np.full(B, -1e9)
        self.body = np.full(B, c.body_init)
        self._leftover = np.zeros(B)
        self.cpos = np.zeros((0, 2))
        self.cval = np.zeros(0)
        self.cage = np.zeros(0)
        self.energy = np.full(B, c.e_init)
        self.total0 = float(self.energy.sum() + c.body_init * B)
        self.pp = PhysicsParams()
        self.every = max(1, int(round(EvalConfig().control_dt / self.pp.dt)))
        self.bd = build_bodies(self.genomes, n_points=self.n_points)
        self.bb = stack_brains([g.brain for g in self.genomes])
        pos = self.rng.uniform(0, self.L, (B, 2))
        self.st = init_state(self.bd, pos=pos, heading=self.rng.uniform(-math.pi, math.pi, B))
        self._refresh_static()
        self.ctrl = np.zeros((B, self.bd.N))
        self.target = np.zeros(B)
        self.tmag = np.zeros(B)
        # yem
        K = c.n_sources
        self.src = self.rng.uniform(0, self.L, (K, 2))
        self.src_dir = self.rng.uniform(0, 2 * math.pi, K)
        self.fpos = np.zeros((0, 2))
        self.fage = np.zeros(0)
        # sayaçlar
        self.t = 0.0
        self.steps = 0
        self.acc = 0.0
        self.eaten_total = 0
        self.gained = 0.0            # yemekten gelen toplam enerji (e_max taşması dahil)
        self.wasted = 0.0            # e_max'ı aşıp boşa giden
        self.spent_move = 0.0
        self.spent_basal = 0.0
        self.deaths = []             # (zaman, id, neden)
        self.dead_energy = 0.0       # ölenlerin ölüm anındaki enerjisi (<= 0), muhasebe için
        self.repro_paid = 0.0        # üreme için ödenen toplam enerji
        self.newborn_energy = 0.0    # yavrulara/kurtarılanlara verilen toplam enerji
        self.gained_meat = 0.0
        self.gained_carc = 0.0
        self.carc_created = 0.0
        self.carc_eaten_pool = 0.0
        self.carc_decay = 0.0
        self.dead_body = 0.0
        self.newborn_body = 0.0
        self.carc_eaten = 0
        self.bite_paid = 0.0
        self.bites = 0
        self.kills = 0
        self.births_sex = 0
        self.births_asex = 0
        self.rescued = 0
        self.history = []
        self._last_hist = -1.0

    # ------------------------------------------------------------------ yardımcılar
    def _refresh_static(self):
        self.mm = motor_mask(self.bd)
        self.mass = np.where(self.bd.active, self.bd.mass, 0.0).sum(1)

    @property
    def pop(self) -> int:
        return len(self.genomes)

    def _wrap_bodies(self):
        com = center_of_mass(self.st, self.bd)
        shift = np.floor(com / self.L) * self.L
        if np.any(shift != 0):
            self.st.x -= shift[:, None, :]

    def _mindiff(self, d):
        return d - self.L * np.round(d / self.L)

    # ------------------------------------------------------------------ yem
    def _food_step(self, dt):
        c = self.cfg
        self.fage += dt
        if len(self.cpos):
            self.cage += dt
            old = self.cage >= c.carcass_life
            if old.any():
                self.carc_decay += float(self.cval[old].sum())
                self.cpos, self.cval, self.cage = self.cpos[~old], self.cval[~old], self.cage[~old]
        keep = self.fage < c.food_life
        if not keep.all():
            self.fpos, self.fage = self.fpos[keep], self.fage[keep]
        self.src_dir += self.rng.normal(0, 0.5 * math.sqrt(dt), len(self.src))
        self.src = (self.src + c.source_drift * dt * np.stack([np.cos(self.src_dir), np.sin(self.src_dir)], 1)) % self.L
        room = c.food_max - len(self.fpos)
        if room > 0:
            k = self.rng.poisson(c.source_rate * dt, len(self.src))
            tot = int(min(k.sum(), room))
            if tot > 0:
                si = np.repeat(np.arange(len(self.src)), k)[:tot]
                newp = (self.src[si] + self.rng.normal(0, c.source_sigma, (tot, 2))) % self.L
                self.fpos = np.concatenate([self.fpos, newp])
                self.fage = np.concatenate([self.fage, np.zeros(tot)])

    # ------------------------------------------------------------------ yeme, algı, ölüm
    def _eat(self):
        """Yemi en yakın canlı (kafa, eat_radius) yer. Verim = 1 - digest_penalty·diet (etçiller yemi kötü sindirir)."""
        c = self.cfg
        B = self.pop
        if B == 0:
            return
        head = self.st.x[:, 0]
        if len(self.cpos):                                                      # leş: verim = 1 - p·(1-diet)
            dc = self._mindiff(self.cpos[None, :, :] - head[:, None, :])
            distc = np.hypot(dc[..., 0], dc[..., 1])
            jc = distc.argmin(0)
            ec = distc[jc, np.arange(distc.shape[1])] <= c.eat_radius
            if ec.any():
                add = np.zeros(B)
                np.add.at(add, jc[ec], self.cval[ec] * (1.0 - c.digest_penalty * (1.0 - self.diet[jc[ec]])))
                before = self.energy + 0.0
                self.energy = np.minimum(self.energy + add, c.e_max)
                self.gained_carc += float(add.sum())
                self.wasted += float((before + add - self.energy).sum())
                self.carc_eaten_pool += float(self.cval[ec].sum())
                self.carc_eaten += int(ec.sum())
                self.cpos, self.cval, self.cage = self.cpos[~ec], self.cval[~ec], self.cage[~ec]
        if len(self.fpos) == 0:
            return
        d = self._mindiff(self.fpos[None, :, :] - head[:, None, :])           # (B,F,2)
        dist = np.hypot(d[..., 0], d[..., 1])                                   # (B,F)
        F = dist.shape[1]
        j = dist.argmin(0)
        dm = dist[j, np.arange(F)]
        eaten = dm <= c.eat_radius
        if eaten.any():
            add = np.zeros(B)
            je = j[eaten]
            np.add.at(add, je, c.food_energy * (1.0 - c.digest_penalty * self.diet[je]))
            before = self.energy + 0.0
            self.energy = np.minimum(self.energy + add, c.e_max)
            self.gained += float(add.sum())
            self.wasted += float((before + add - self.energy).sum())
            self.eaten_total += int(eaten.sum())
            self.fpos, self.fage = self.fpos[~eaten], self.fage[~eaten]

    def _sense(self):
        """Beynin 'hedef yön' girişi: otçul en yakın yemi, etçil en değerli/yakın canlıyı hedefler (omnivor: skora göre)."""
        c = self.cfg
        B = self.pop
        self.target = np.zeros(B)
        self.tmag = np.zeros(B)
        self.fprox = np.zeros(B)
        if B == 0:
            return
        ar = np.arange(B)
        head = self.st.x[:, 0]
        fd = np.full(B, np.inf)
        fvec = np.zeros((B, 2))
        if len(self.fpos):
            d = self._mindiff(self.fpos[None, :, :] - head[:, None, :])
            dist = np.hypot(d[..., 0], d[..., 1])
            k = dist.argmin(1)
            fd, fvec = dist[ar, k], d[ar, k]
        fnear = fd <= c.sense_radius
        self.fprox = np.where(fnear, np.clip(1 - fd / c.sense_radius, 0, 1), 0.0)
        pd = np.full(B, np.inf)
        pvec = np.zeros((B, 2))
        pval = np.zeros(B)
        if B >= 2 and (self.diet > 0).any():
            d = self._mindiff(head[None, :, :] - head[:, None, :])
            dist = np.hypot(d[..., 0], d[..., 1])
            np.fill_diagonal(dist, np.inf)
            j = dist.argmin(1)
            pd, pvec = dist[ar, j], d[ar, j]
            pval = c.k_meat * self.mass[j] + c.meat_frac * (self.energy[j] + self.body[j])
        pnear = pd <= c.sense_radius
        cd = np.full(B, np.inf)
        cvec = np.zeros((B, 2))
        cvalue = np.zeros(B)
        if len(self.cpos) and (self.diet > 0).any():
            d = self._mindiff(self.cpos[None, :, :] - head[:, None, :])
            dist = np.hypot(d[..., 0], d[..., 1])
            k = dist.argmin(1)
            cd, cvec, cvalue = dist[ar, k], d[ar, k], self.cval[k]
        cnear = cd <= c.sense_radius
        sf = np.where(fnear, (1 - self.diet) * c.food_energy / (fd + 1.0), -np.inf)
        sp = np.where(pnear, self.diet * pval / (pd + 1.0), -np.inf)
        sc = np.where(cnear, self.diet * cvalue / (cd + 1.0), -np.inf)
        best = np.argmax(np.stack([sf, sp, sc]), axis=0)
        vec = np.where((best == 0)[:, None], fvec, np.where((best == 1)[:, None], pvec, cvec))
        has = np.maximum(np.maximum(sf, sp), sc) > 0
        self.target = np.where(has, np.arctan2(vec[:, 1], vec[:, 0]), 0.0)
        self.tmag = has.astype(float)

    def _eat_and_sense(self):          # (aşama 1-2 uyumluluğu: testler ve eski çağrılar)
        self._eat()
        self._sense()

    def _bite(self):
        """Isırma: istek > eşik, bekleme bitmiş, enerjisi yeterli VE menzilde (r_bite, kafa-kafa) bir canlı varsa ısırma olur
        (cost_bite ödenir; havaya ısırmak bedava). Av ölür (kafası yenir), saldırgan et enerjisini alır. Karşılıklı ısırmada enerjisi yüksek olan kazanır (eşitse ikisi de ölür).
        Dönüş: ölen canlıların maskesi."""
        c, B = self.cfg, self.pop
        killed = np.zeros(B, bool)
        self._leftover = np.zeros(B)
        if B < 2:
            return killed
        ok = (self.want[:, 2] > c.want_thr) & (self.t - self.bite_t >= c.bite_cd) & (self.energy > c.cost_bite)
        att = np.nonzero(ok)[0]
        if len(att) == 0:
            return killed
        head = self.st.x[:, 0]
        d = self._mindiff(head[None, :, :] - head[att][:, None, :])
        dist = np.hypot(d[..., 0], d[..., 1])
        dist[np.arange(len(att)), att] = np.inf
        v = dist.argmin(1)
        hit = dist[np.arange(len(att)), v] <= c.r_bite
        if not hit.any():                      # menzilde canlı yoksa ısırma olayı yok (havaya ısırmak bedava)
            return killed
        att, v = att[hit], v[hit]
        self.energy[att] -= c.cost_bite
        self.bite_paid += c.cost_bite * len(att)
        self.bite_t[att] = self.t
        target = {int(a): int(b) for a, b in zip(att, v)}
        if not target:
            return killed
        out = []
        for a in self.rng.permutation(list(target)):
            a = int(a)
            b = target[a]
            if killed[a] or killed[b]:
                continue
            if target.get(b) == a:                              # karşılıklı
                if self.energy[a] > self.energy[b]:
                    out.append((a, b))
                elif self.energy[a] < self.energy[b]:
                    out.append((b, a))
                else:
                    killed[a] = killed[b] = True
                    self.kills += 2
                    self._leftover[a] = self.energy[a] + self.body[a]      # ikisi de ölür: hepsi leş olur
                    self._leftover[b] = self.energy[b] + self.body[b]
                    continue
                w, l = out[-1]
                killed[l] = True
                self.kills += 1
                continue
            out.append((a, b))
            killed[b] = True
            self.kills += 1
        for a, b in out:
            if killed[a]:
                continue
            V = self.energy[b] + self.body[b]
            eff = 1.0 - c.digest_penalty * (1.0 - self.diet[a])
            gain = (c.k_meat * self.mass[b] + c.meat_frac * V) * eff
            self._leftover[b] = (1.0 - c.meat_frac) * V
            before = self.energy[a]
            self.energy[a] = min(before + gain, c.e_max)
            self.gained_meat += gain
            self.wasted += before + gain - self.energy[a]
            self.bites += 1
        return killed

    def _remove(self, alive):
        """alive (B,) bool dışındakileri sil (satır seçimi: batch genişliği değişmez)."""
        idx = np.nonzero(alive)[0]
        self.genomes = [self.genomes[i] for i in idx]
        self.ids, self.born, self.energy = self.ids[idx], self.born[idx], self.energy[idx]
        self.gen, self.want, self.fprox = self.gen[idx], self.want[idx], self.fprox[idx]
        self.diet, self.lin, self.bite_t = self.diet[idx], self.lin[idx], self.bite_t[idx]
        self.body = self.body[idx]
        self.bd, self.bb, self.st = select_bodies(self.bd, idx), select_brains(self.bb, idx), select_state(self.st, idx)
        self.ctrl = self.ctrl[idx]
        self.target, self.tmag = self.target[idx], self.tmag[idx]
        self._refresh_static()

    def _deaths(self, killed=None):
        dead = self.energy <= 0
        if killed is not None:
            dead = dead | killed
        if dead.any():
            for i in np.nonzero(dead)[0]:
                cause = "ısırık" if (killed is not None and killed[i]) else "açlık"
                self.deaths.append((self.t, int(self.ids[i]), cause))
            carc = np.where(killed if killed is not None else np.zeros(self.pop, bool), self._leftover, self.body)
            carc = np.where(dead, carc, 0.0)
            mk = dead & (carc > 1e-9)
            if mk.any():
                self.cpos = np.concatenate([self.cpos, self.st.x[mk, 0]])
                self.cval = np.concatenate([self.cval, carc[mk]])
                self.cage = np.concatenate([self.cage, np.zeros(int(mk.sum()))])
                self.carc_created += float(carc[mk].sum())
            self.dead_energy += float(self.energy[dead].sum())
            self.dead_body += float(self.body[dead].sum())
            self._remove(~dead)

    # ------------------------------------------------------------------ algı: en yakın diğer canlı
    def _extras(self):
        """Beyin v2 ek girişleri (B,6): yem yakınlığı, kendi enerjisi, en yakın canlının yönü (gövde çerçevesi, 2),
        yakınlığı, enerjisi."""
        c, B = self.cfg, self.pop
        ex = np.zeros((B, 6))
        ex[:, 0] = self.fprox
        ex[:, 1] = np.clip(self.energy / c.e_max, 0, 1)
        if B >= 2:
            head = self.st.x[:, 0]
            d = self._mindiff(head[None, :, :] - head[:, None, :])
            dist = np.hypot(d[..., 0], d[..., 1])
            np.fill_diagonal(dist, np.inf)
            ar = np.arange(B)
            j = dist.argmin(1)
            dm = dist[ar, j]
            near = dm <= c.sense_radius
            vec = d[ar, j]
            ph = self.st.phi
            bx = np.cos(ph) * vec[:, 0] + np.sin(ph) * vec[:, 1]
            by = -np.sin(ph) * vec[:, 0] + np.cos(ph) * vec[:, 1]
            u = np.where(near, 1.0 / np.maximum(dm, 1e-9), 0.0)
            ex[:, 2], ex[:, 3] = bx * u, by * u
            ex[:, 4] = np.where(near, np.clip(1 - dm / c.sense_radius, 0, 1), 0.0)
            ex[:, 5] = np.where(near, np.clip(self.energy[j] / c.e_max, 0, 1), 0.0)
        return ex

    # ------------------------------------------------------------------ batch genişletme ve doğum
    def _grow(self, n_new: int):
        """Batch genişliğini (nokta sayısı) n_new'e çıkar; mevcut canlıların durumu korunur (burun noktası son indekse taşınır)."""
        N0, N1, B = self.bd.N, n_new, self.pop
        if N1 <= N0:
            return
        leg = self.bd.legacy_spine

        def remap(a):
            out = np.zeros((B, N1) + a.shape[2:], dtype=a.dtype)
            out[:, :N0 - 1] = a[:, :N0 - 1]
            out[~leg, N1 - 1] = a[~leg, N0 - 1]
            out[leg, N0 - 1] = a[leg, N0 - 1]
            return out

        x, v, th, jv, ta, ct = (remap(a) for a in (self.st.x, self.st.v, self.st.theta, self.st.jvel, self.st.tau, self.ctrl))
        bd = build_bodies(self.genomes, n_points=N1)
        free = ~bd.active
        head = np.broadcast_to(x[:, None, 0, :], x.shape)
        x[free] = head[free]
        for a in (v, th, jv, ta, ct):
            a[free] = 0.0
        self.bd = bd
        self.st = State(x=x, v=v, phi=self.st.phi, omega=self.st.omega, t=self.st.t, theta=th, jvel=jv, tau=ta)
        self.ctrl = ct
        self.n_points = N1
        self._refresh_static()

    def _make_child(self, a: Genome, b: Genome = None) -> Genome:
        c = self.cfg
        base = crossover(a, b, self.rng) if b is not None else a.copy()
        for _ in range(4):
            m = mutate(base, self.rng, c.mut)
            if m.body.n <= c.max_limbs:
                break
        else:
            m = mutate(base, self.rng, c.mut, freeze_body=True)
        m.meta = {"physics": PHYSICS_VERSION}
        return m

    def _add(self, children: list, pos: np.ndarray, gens: np.ndarray, energy: float, diet=None, lin=None, body=None):
        """Yeni canlıları batch'in sonuna ekle."""
        k = len(children)
        if k == 0:
            return
        need = max(g.body.n + (1 if g.body.legacy_spine else 2) for g in children)
        if need > self.bd.N:
            self._grow(need)
        N = self.bd.N
        bd_c = build_bodies(children, n_points=N)
        st_c = init_state(bd_c, pos=pos, heading=self.rng.uniform(-math.pi, math.pi, k))
        self.genomes += list(children)
        self.bd, self.st = concat_bodies(self.bd, bd_c), concat_state(self.st, st_c)
        self.bb = concat_brains(self.bb, stack_brains([g.brain for g in children]))
        self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + k)])
        self.next_id += k
        self.born = np.concatenate([self.born, np.full(k, self.t)])
        self.energy = np.concatenate([self.energy, np.full(k, float(energy))])
        self.gen = np.concatenate([self.gen, gens])
        diet = np.zeros(k) if diet is None else np.asarray(diet, float)
        lin = np.zeros(k, dtype=int) if lin is None else np.asarray(lin, dtype=int)
        for g, dv, lv in zip(children, diet, lin):
            g.meta["diet"], g.meta["lineage"] = float(dv), int(lv)
        body = np.zeros(k) if body is None else np.asarray(body, float)
        self.body = np.concatenate([self.body, body])
        self.newborn_body += float(body.sum())
        self.diet, self.lin, self.bite_t = np.concatenate([self.diet, diet]), np.concatenate([self.lin, lin]), np.concatenate([self.bite_t, np.full(k, -1e9)])
        self.want = np.concatenate([self.want, np.zeros((k, 3))])
        self.target, self.tmag = np.concatenate([self.target, np.zeros(k)]), np.concatenate([self.tmag, np.zeros(k)])
        self.fprox = np.concatenate([self.fprox, np.zeros(k)])
        self.ctrl = np.concatenate([self.ctrl, np.zeros((k, N))])
        self.newborn_energy += float(energy) * k
        self._refresh_static()

    def _reproduce(self):
        c, B = self.cfg, self.pop
        if B == 0:
            return
        ok = (self.t - self.born) >= c.min_age
        w_sex, w_asx = self.want[:, 0], self.want[:, 1]
        mate = ok & (w_sex > c.want_thr) & (w_sex >= w_asx) & (self.energy > c.cost_mate)
        children, pos, gens, dts, lns, bds = [], [], [], [], [], []
        self._paired = np.zeros(B, bool)
        head = self.st.x[:, 0]
        E = np.nonzero(mate)[0]
        if len(E) >= 2:
            d = self._mindiff(head[E][None, :, :] - head[E][:, None, :])
            dist = np.hypot(d[..., 0], d[..., 1])
            np.fill_diagonal(dist, np.inf)
            used = np.zeros(len(E), bool)
            for a in self.rng.permutation(len(E)):
                if used[a]:
                    continue
                cand = np.where(~used & (dist[a] <= c.r_mate))[0]
                if len(cand) == 0:
                    continue
                b = int(cand[dist[a, cand].argmin()])
                used[a] = used[b] = True
                i, j = int(E[a]), int(E[b])
                self._paired[i] = self._paired[j] = True
                self.energy[i] -= c.cost_mate
                self.energy[j] -= c.cost_mate
                self.repro_paid += 2 * c.cost_mate
                children.append(self._make_child(self.genomes[i], self.genomes[j]))
                pos.append(head[i] + 0.5 * self._mindiff(head[j] - head[i]))
                gens.append(max(self.gen[i], self.gen[j]) + 1)
                dts.append(0.5 * (self.diet[i] + self.diet[j]))
                bds.append(max(0.0, 2 * c.cost_mate - c.e_birth) * c.body_frac)
                lns.append(self.lin[i])
                self.births_sex += 1
        split = ok & (w_asx > c.want_thr) & (self.energy > c.cost_split) & ~self._paired
        for i in np.nonzero(split)[0]:
            self.energy[i] -= c.cost_split
            self.repro_paid += c.cost_split
            ang = self.rng.uniform(0, 2 * math.pi)
            children.append(self._make_child(self.genomes[i]))
            pos.append(head[i] + np.array([math.cos(ang), math.sin(ang)]))
            gens.append(self.gen[i] + 1)
            dts.append(self.diet[i])
            bds.append(max(0.0, c.cost_split - c.e_birth) * c.body_frac)
            lns.append(self.lin[i])
            self.births_asex += 1
        if children:
            self.want[np.concatenate([E, np.nonzero(split)[0]]).astype(int)] = 0.0
            dd = np.clip(np.array(dts) + self.rng.normal(0, c.diet_sigma, len(dts)), 0.0, 1.0)
            self._add(children, np.array(pos) % self.L, np.array(gens, dtype=int), c.e_birth, dd, np.array(lns, dtype=int), np.array(bds))

    def _rescue(self):
        c = self.cfg
        if c.min_pop <= 0 or self.pop >= c.min_pop:
            return
        k = c.min_pop - self.pop
        m = c.mut
        heavy = replace(m, p_gene=min(0.5, 2 * m.p_gene), p_add=min(0.3, 2 * m.p_add), p_remove=min(0.3, 2 * m.p_remove),
                        brain_rate=min(0.5, 2 * m.brain_rate), brain_sigma=2 * m.brain_sigma)
        kids, kd, kl = [], [], []
        for _ in range(k):
            if self.pop > 0:
                pi = int(self.rng.integers(0, self.pop))
                p = self.genomes[pi]
                kd.append(self.diet[pi]); kl.append(self.lin[pi])
                kid = mutate(p, self.rng, heavy)
                if kid.body.n > c.max_limbs:
                    kid = mutate(p, self.rng, heavy, freeze_body=True)
            else:
                lo, hi = c.init_limbs
                kid = random_genome(self.rng, n_min=lo, n_max=hi, brain_v2=True)
                kd.append(float(self.rng.random() < c.carnivore_frac)); kl.append(0)
            kid.meta = {"physics": PHYSICS_VERSION}
            kids.append(kid)
        self.rescued += k
        self._add(kids, self.rng.uniform(0, self.L, (k, 2)), np.zeros(k, dtype=int), c.e_init, kd, kl, np.full(k, c.body_init))

    # ------------------------------------------------------------------ kontrol adımı (50 Hz)
    def _control(self):
        dt = self.every * self.pp.dt
        self._wrap_bodies()
        self._food_step(dt)
        self._eat()
        self._deaths(self._bite())
        self._rescue()
        self._reproduce()
        if self.pop == 0:
            return
        self._sense()
        self.ctrl, self.want = act2(self.st, self.bd, self.bb, self.target, self.tmag, self._extras())
        if self.t - self._last_hist >= 1.0:
            self._last_hist = self.t
            self.history.append(self.stats())

    def _account(self):
        c, dt = self.cfg, self.pp.dt
        power = np.sum(np.abs(self.st.tau * self.st.jvel) * self.mm, axis=1)
        mv = c.c_move * power * dt
        bs = c.c_basal * self.mass * dt
        self.energy -= mv + bs
        self.spent_move += float(mv.sum())
        self.spent_basal += float(bs.sum())

    def advance(self, dt: float, max_steps: int = 100000):
        """dt saniye (gerçek zamanlı) ilerlet."""
        self.acc += dt
        n = min(int(self.acc / self.pp.dt), max_steps)
        self.acc -= n * self.pp.dt
        for _ in range(n):
            if self.steps % self.every == 0:
                self._control()
            if self.pop == 0:
                break
            step(self.st, self.bd, self.ctrl, self.pp)
            self._account()
            self.steps += 1
            self.t = self.steps * self.pp.dt

    # ------------------------------------------------------------------ istatistik
    def stats(self) -> dict:
        B = self.pop
        sp = np.linalg.norm(com_velocity(self.st, self.bd), axis=1) if B else np.zeros(0)
        return dict(t=self.t, pop=B, food=len(self.fpos), mean_energy=float(self.energy.mean()) if B else 0.0,
                    eaten=self.eaten_total, mean_limbs=float(np.mean([g.body.n for g in self.genomes])) if B else 0.0,
                    mean_speed=float(sp.mean()) if B else 0.0, births_sex=self.births_sex, births_asex=self.births_asex,
                    mean_gen=float(self.gen.mean()) if B else 0.0, max_gen=int(self.gen.max()) if B else 0,
                    rescued=self.rescued, carn=int((self.diet > 0.5).sum()) if B else 0, bites=self.bites, kills=self.kills,
                    mean_diet=float(self.diet.mean()) if B else 0.0, carcasses=len(self.cpos), carc_eaten=self.carc_eaten)

    def total_energy(self) -> float:
        return float(self.energy.sum() + self.body.sum() + self.cval.sum())

    def balance_residual(self) -> float:
        """Enerji korunumu: ~0 olmalı. Sistem = canlı enerjisi + vücut enerjisi + leşler."""
        b = self.energy_balance()
        inflow = b["gained"] + b["meat"] + self.gained_carc + self.carc_created + b["newborn"] + self.newborn_body
        outflow = b["move"] + b["basal"] + b["bite"] + b["repro"] + b["wasted"] + self.carc_eaten_pool + self.carc_decay
        return self.total0 + inflow - outflow - (self.total_energy() + self.dead_energy + self.dead_body)

    def energy_balance(self) -> dict:
        """Muhasebe: başlangıç + yemek + yavru/kurtarma enerjisi - harcamalar - üreme bedeli - e_max taşması = şu anki toplam + ölenlerin son enerjisi."""
        return dict(gained=self.gained, wasted=self.wasted, move=self.spent_move, basal=self.spent_basal,
                    repro=self.repro_paid, newborn=self.newborn_energy, meat=self.gained_meat, bite=self.bite_paid)


def run_headless(world: World, seconds: float, log=None, every: float = 10.0):
    """Gerçek zamandan bağımsız çalıştır; her `every` sim-sn'de bir satır yazdır."""
    import time
    t0, nxt = time.time(), 0.0
    while world.t < seconds and world.pop > 0:
        world.advance(0.5)
        if world.t >= nxt:
            s = world.stats()
            line = (f"t={s['t']:6.1f}s  nüfus {s['pop']:3d}  yem {s['food']:3d}  ort.enerji {s['mean_energy']:6.1f}  "
                    f"yenen {s['eaten']:4d}  ort.hız {s['mean_speed']:.3f}  doğum {s['births_asex']}+{s['births_sex']}  "
                    f"nesil {s['mean_gen']:.1f}/{s['max_gen']}  uzuv {s['mean_limbs']:.1f}  ({(time.time()-t0):.0f} sn gerçek)")
            if log:
                log(line)
            nxt += every
    return world


def seeded_genomes(paths, n, rng, mut=None, want_bias=(1.0, 0.6, -2.0), max_limbs=24, carnivore_frac=0.2):
    """Eğitilmiş canlı dosyalarından başlangıç nüfusu. İlk len(paths) birey dosyaların aynısı (beyin v2'ye yükseltilir:
    üreme istekleri başlangıçta açık, ısırma kapalı), kalanlar bunların mutasyonlu kopyaları (sırayla dağıtılır)."""
    mut = mut or MutationConfig()
    base = []
    for p in paths:
        g = load_json(p)
        if g.brain.W1.shape[0] != N_IN_V2:
            g = Genome(g.body, upgrade_brain(g.brain, want_bias), g.meta)
        g.meta = {"physics": PHYSICS_VERSION, "seed": str(p), "lineage": len(base)}
        base.append(g)
    out = [g.copy() for g in base[:n]]
    k = 0
    while len(out) < n:
        src = base[k % len(base)]
        kid = mutate(src, rng, mut)
        if kid.body.n > max_limbs:
            kid = mutate(src, rng, mut, freeze_body=True)
        kid.meta = {"physics": PHYSICS_VERSION, "seed": src.meta.get("seed", ""), "lineage": src.meta["lineage"]}
        out.append(kid)
        k += 1
    for g in out:      # beslenme: etçillerin ısırma isteği başlangıçta açık, otçullarınki kapalı
        carn = bool(rng.random() < carnivore_frac)
        g.meta["diet"] = 1.0 if carn else 0.0
        g.brain.b2[-1] = 1.0 if carn else -2.0
    return out

"""
Evrimleşen Yüzücüler - Chapter 2 (ekosistem) arayüzü.  Aşama 1: sarmal dünya, yem kaynakları, enerji, açlıktan ölüm.

    python eco.py                                  # pencereli izleme (rastgele 100 canlı)
    python eco.py --headless --seconds 600         # grafiksiz; satır satır özet + log.csv + hayatta kalanların JSON'u
    python eco.py --n 60 --seed 3 --size 50

Tuşlar:  SPACE turbo · P duraklat · +/- hız · tıkla: canlı seç · S: seçili canlıyı kaydet (DENE/EDİTÖR'de açılır)
         F: seçiliyi takip et · Z: tüm dünya · tekerlek: yakınlaştır · R: yeni dünya · H: yardım · ESC çıkış
Pencere 1280x720 başlar, yeniden boyutlandırılabilir.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from genome import save_json
from sim import Camera
from ui import *
from world import World, WorldConfig, run_headless, seeded_genomes, select_bodies, select_state

PANEL_W = 300


class EcoApp:
    def __init__(self, screen, cfg: WorldConfig, out="runs/eco1", make=None):
        self.screen = screen
        self.cfg = cfg
        self.out = out
        self.make = make or (lambda c: World(c))
        self.world = self.make(cfg)
        self.fonts = Fonts(screen.get_height())
        self.cam = Camera()
        self.cam.auto = False
        self.speed = 1.0
        self.turbo = False
        self.paused = False
        self.show_help = True
        self.sel = None                # seçili canlının kimliği
        self.follow = False
        self.running = True
        self.msg, self.msg_until = "", 0.0
        self.fit_world()
        self.fps = deque(maxlen=30)

    # ---------------- yerleşim ----------------
    def layout(self):
        w, h = self.screen.get_size()
        pw = int(np.clip(w * 0.24, 240, PANEL_W + 60))
        bar = int(self.fonts.small.get_height() * 1.9)
        return pygame.Rect(0, 0, w - pw, h - bar), pygame.Rect(w - pw, 0, pw, h - bar), pygame.Rect(0, h - bar, w, bar)

    def fit_world(self):
        view, _, _ = self.layout()
        L = self.world.L
        self.cam.c = np.array([L / 2, L / 2])
        self.cam.scale = 0.94 * min(view.w, view.h) / L
        self.follow = False

    def say(self, s, secs=3.0):
        self.msg, self.msg_until = s, time.time() + secs

    # ---------------- olaylar ----------------
    def idx_of(self, cid):
        w = self.world
        k = np.nonzero(w.ids == cid)[0]
        return int(k[0]) if len(k) else None

    def handle_event(self, e):
        w = self.world
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.get_surface()
        elif e.type == pygame.MOUSEWHEEL:
            self.cam.scale = float(np.clip(self.cam.scale * (1.12 ** e.y), 3, 300))
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            view, _, _ = self.layout()
            if view.collidepoint(e.pos) and w.pop:
                p = self.s2w(e.pos, view.size)
                d = w._mindiff(w.st.x[:, 0] - p)
                dist = np.hypot(d[:, 0], d[:, 1])
                k = int(dist.argmin())
                self.sel = int(w.ids[k]) if dist[k] < max(1.5, 20 / self.cam.scale) else None
        elif e.type == pygame.KEYDOWN:
            k = e.key
            if k == pygame.K_ESCAPE:
                self.running = False
            elif k == pygame.K_SPACE:
                self.turbo = not self.turbo
            elif k == pygame.K_p:
                self.paused = not self.paused
            elif k == pygame.K_h:
                self.show_help = not self.show_help
            elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.speed = min(16.0, self.speed * 1.5)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(0.125, self.speed / 1.5)
            elif k == pygame.K_z:
                self.fit_world()
            elif k == pygame.K_f:
                self.follow = not self.follow and self.sel is not None
            elif k == pygame.K_r:
                self.cfg = WorldConfig(**{**self.cfg.__dict__, "seed": self.cfg.seed + 1})
                self.world = self.make(self.cfg)
                self.sel = None
                self.fit_world()
                self.say(f"yeni dünya (tohum {self.cfg.seed})")
            elif k == pygame.K_s and self.sel is not None:
                i = self.idx_of(self.sel)
                if i is not None:
                    os.makedirs(self.out, exist_ok=True)
                    g = w.genomes[i].copy()
                    g.meta.update(dict(eco_id=self.sel, energy=float(w.energy[i]), age=float(w.t - w.born[i]), physics=4, diet=float(w.diet[i]), lineage=int(w.lin[i])))
                    path = os.path.join(self.out, f"eco_{self.sel:04d}.json")
                    save_json(g, path)
                    self.say(f"kaydedildi: {path}", 5.0)

    def s2w(self, p, wh):
        c, sc = self.cam.c, self.cam.scale
        return np.array([c[0] + (p[0] - wh[0] / 2) / sc, c[1] - (p[1] - wh[1] / 2) / sc])

    # ---------------- güncelleme ----------------
    def update(self, dt):
        t0 = time.time()
        if not self.paused and self.world.pop > 0:
            if self.turbo:
                while time.time() - t0 < 0.03:
                    self.world.advance(0.25)
            else:
                self.world.advance(min(dt, 0.1) * self.speed, max_steps=int(400 * max(1.0, self.speed)))
        if self.follow and self.sel is not None:
            i = self.idx_of(self.sel)
            if i is None:
                self.follow = False
            else:
                self.cam.c += (self.world.st.x[i, 0] - self.cam.c) * (1 - math.exp(-6 * max(dt, 1e-3)))
        if self.sel is not None and self.idx_of(self.sel) is None:
            self.sel = None
            self.follow = False

    # ---------------- çizim ----------------
    def draw(self):
        scr = self.screen
        h = scr.get_height()
        if self.fonts.h != h:
            self.fonts = Fonts(h)
        scr.fill(C_PANEL)
        view, panel, bar = self.layout()
        if self.turbo:
            self.draw_turbo(scr)
        else:
            self.draw_world(scr.subsurface(view))
            self.draw_panel(scr.subsurface(panel))
        self.draw_bar(scr.subsurface(bar))
        pygame.display.flip()

    def draw_world(self, surf):
        w, f = self.world, self.fonts
        vw, vh = surf.get_size()
        wh = (vw, vh)
        cam, L, sc = self.cam, w.L, self.cam.scale
        surf.fill((7, 28, 52))
        # dünya sınırı ve ızgara (5 m)
        def sp(p):
            return cam.w2s(p, wh)
        x0, y0 = sp((0, L))
        x1, y1 = sp((L, 0))
        pygame.draw.rect(surf, (10, 38, 66), (x0, y0, x1 - x0, y1 - y0))
        if sc > 5:
            for g in range(0, int(L) + 1, 5):
                pygame.draw.line(surf, C_GRID_MAJOR if g % 10 == 0 else C_GRID_MINOR, sp((g, 0)), sp((g, L)))
                pygame.draw.line(surf, C_GRID_MAJOR if g % 10 == 0 else C_GRID_MINOR, sp((0, g)), sp((L, g)))
        pygame.draw.rect(surf, (90, 150, 200), (x0, y0, x1 - x0, y1 - y0), 2)
        # yem kaynakları (halka) ve yem
        for s in w.src:
            pygame.draw.circle(surf, (40, 90, 60), sp(s), max(6, int(w.cfg.source_sigma * sc)), 1)
        for p, v in zip(w.cpos, w.cval):                                           # leş: kahverengi, büyüklük = enerji
            pygame.draw.circle(surf, (170, 110, 70), sp(p), max(3, int(0.12 * sc * math.sqrt(max(v, 1.0) / 10))))
        fr = max(2, int(0.22 * sc))
        for p, a in zip(w.fpos, w.fage):
            k = 1 - a / w.cfg.food_life
            pygame.draw.circle(surf, (int(90 + 120 * k), int(170 + 70 * k), int(80 + 40 * k)), sp(p), fr)
        # canlılar (kenar yakınında sarılmış kopyalarıyla; kopyalar yalnızca dünya içinde görünür)
        surf.set_clip(pygame.Rect(x0, y0, x1 - x0, y1 - y0).clip(surf.get_rect()))
        com = w.st.x[:, 0]
        margin = 4.0
        for i in range(w.pop):
            sx = [0.0] + ([L] if com[i, 0] < margin else []) + ([-L] if com[i, 0] > L - margin else [])
            sy = [0.0] + ([L] if com[i, 1] < margin else []) + ([-L] if com[i, 1] > L - margin else [])
            shifts = [(a, b) for a in sx for b in sy]
            v = SimpleNamespace(bd=select_bodies(w.bd, [i]), st=select_state(w.st, [i]), ctrl=w.ctrl[[i]], mm=w.mm[i])
            for dx, dy in shifts:
                c2 = Camera()
                c2.c = cam.c - np.array([dx, dy])
                c2.scale = sc
                draw_creature(surf, v, c2, wh)
                hp = c2.w2s(com[i], wh)
                frac = float(np.clip(w.energy[i] / w.cfg.e_max, 0, 1))
                bw = 24
                dv = float(w.diet[i])                                   # beslenme: yeşil = otçul, kırmızı = etçil
                pygame.draw.circle(surf, (int(90 + 165 * dv), int(230 - 140 * dv), int(120 - 30 * dv)), (hp[0] + bw / 2 + 5, hp[1] - 14), 4)
                pygame.draw.rect(surf, (20, 30, 40), (hp[0] - bw / 2, hp[1] - 16, bw, 4))
                col = (90, 230, 120) if frac > 0.3 else (255, 190, 70) if frac > 0.12 else (255, 90, 90)
                pygame.draw.rect(surf, col, (hp[0] - bw / 2, hp[1] - 16, bw * frac, 4))
                if int(w.ids[i]) == self.sel:
                    pygame.draw.circle(surf, (255, 220, 80), hp, 18, 2)
                    if w.tmag[i] > 0:         # seçili canlının algıladığı yem yönü
                        a = w.target[i]
                        arrow(surf, C_TARGET, hp, (hp[0] + math.cos(a) * 40, hp[1] - math.sin(a) * 40), 2)
        surf.set_clip(None)
        s = w.stats()
        text(surf, f.normal, f"t = {s['t']:.0f} s   nüfus {s['pop']}   yem {s['food']}   ort. enerji {s['mean_energy']:.0f}",
             (10, 8), C_TEXT)
        if self.msg and time.time() < self.msg_until:
            text(surf, f.normal, self.msg, (10, 8 + f.normal.get_height() + 4), (255, 220, 120))
        if w.pop == 0:
            text(surf, f.huge, "NESİL TÜKENDİ", (vw // 2, vh // 2), (255, 120, 120), "center")
            text(surf, f.normal, "R: yeni dünya", (vw // 2, vh // 2 + f.huge.get_height()), C_DIM, "center")
        if self.show_help:
            text(surf, f.small, "yem: yeşil nokta · enerji çubuğu: yeşil>turuncu>kırmızı · mavi ok: seçilinin algıladığı yem yönü",
                 (10, vh - f.small.get_height() - 6), C_DIM)
        text(surf, f.small, f"hız ×{self.speed:g}" + ("  DURAKLATILDI" if self.paused else ""), (vw - 10, 8), C_DIM, "topright")

    def draw_series(self, surf, rect, series, title):
        f = self.fonts
        pygame.draw.rect(surf, (20, 28, 40), rect)
        pygame.draw.rect(surf, C_PANEL_LINE, rect, 1)
        text(surf, f.small, title, (rect.x + 4, rect.y + 2), C_DIM)
        H = self.world.history
        if len(H) < 2:
            return
        inner = pygame.Rect(rect.x + 4, rect.y + f.small.get_height() + 4, rect.w - 8, rect.h - f.small.get_height() - 8)
        t = np.array([s["t"] for s in H])
        for key, col in series:
            v = np.array([s[key] for s in H], float)
            hi = max(v.max(), 1e-9)
            pts = [(inner.x + (ti - t[0]) / max(t[-1] - t[0], 1e-9) * inner.w, inner.bottom - vi / hi * inner.h) for ti, vi in zip(t, v)]
            pygame.draw.lines(surf, col, False, pts, 2)

    def draw_panel(self, surf):
        w, f = self.world, self.fonts
        pw, ph = surf.get_size()
        pygame.draw.line(surf, C_PANEL_LINE, (0, 0), (0, ph), 2)
        y = 8
        text(surf, f.big, "EKOSİSTEM", (12, y), C_BEST)
        y += f.big.get_height() + 4
        s = w.stats()
        b = w.energy_balance()
        for k, v in (("nüfus", f"{s['pop']}"), ("yem", f"{s['food']}"), ("toplam yenen", f"{s['eaten']}"),
                     ("ölen (açlık)", f"{len(w.deaths)}"),
                     ("doğum eşeysiz / eşeyli", f"{s['births_asex']} / {s['births_sex']}"),
                     ("etçil / otçul", f"{s['carn']} / {s['pop'] - s['carn']}"),
                     ("ısırık / öldürme", f"{s['bites']} / {s['kills']}"),
                     ("leş (şimdi / yenen)", f"{s['carcasses']} / {s['carc_eaten']}"),
                     ("soy (tohum 0/1/2/…)", " / ".join(str(int(x)) for x in np.bincount(w.lin, minlength=3)[:6])),
                     ("nesil ort / en yüksek", f"{s['mean_gen']:.1f} / {s['max_gen']}"), ("ort. uzuv", f"{s['mean_limbs']:.1f}"),
                     ("ort. hız", f"{s['mean_speed']:.2f} m/s"),
                     ("giderler (hareket/bazal/üreme)", f"{b['move']:.0f} / {b['basal']:.0f} / {b['repro']:.0f}"),
                     ("et enerjisi kazancı", f"{b['meat']:.0f}"),
                     ("yemekten kazanç", f"{b['gained']:.0f}")):
            text(surf, f.small, k, (12, y), C_DIM)
            text(surf, f.small, v, (pw - 12, y), C_TEXT, "topright")
            y += f.small.get_height() + 2
        y += 6
        gh = int(ph * 0.14)
        self.draw_series(surf, pygame.Rect(12, y, pw - 24, gh), [("pop", C_BEST)], "nüfus")
        y += gh + 6
        self.draw_series(surf, pygame.Rect(12, y, pw - 24, gh), [("mean_energy", C_POS)], "ortalama enerji")
        y += gh + 10
        i = self.idx_of(self.sel) if self.sel is not None else None
        if i is None:
            text(surf, f.small, "bir canlıya tıkla", (12, y), C_DIM)
            return
        g = w.genomes[i]
        sp = float(np.linalg.norm(w.st.v[i, 0]))
        pw_ = float(np.sum(np.abs(w.st.tau[i] * w.st.jvel[i]) * w.mm[i]))
        text(surf, f.normal, f"canlı #{self.sel}", (12, y), (255, 220, 80))
        y += f.normal.get_height() + 2
        for k, v in (("nesil", f"{w.gen[i]}"), ("beslenme (0 otçul, 1 etçil)", f"{w.diet[i]:.2f}"), ("soy", f"{w.lin[i]}"), ("uzuv", f"{g.body.n}"), ("yaş", f"{w.t - w.born[i]:.0f} sn"),
                     ("üreme isteği (eşeyli/eşeysiz)", f"{w.want[i, 0]:+.1f} / {w.want[i, 1]:+.1f}"), ("enerji", f"{w.energy[i]:.0f} / {w.cfg.e_max:.0f}"),
                     ("kafa hızı", f"{sp:.2f} m/s"), ("mekanik güç", f"{pw_:.1f} W"),
                     ("gider", f"{w.cfg.c_move * pw_ + w.cfg.c_basal * w.mass[i]:.2f} /sn"), ("kütle", f"{w.mass[i]:.1f} kg"),
                     ("yem algısı", "var" if w.tmag[i] > 0 else "yok")):
            text(surf, f.small, k, (12, y), C_DIM)
            text(surf, f.small, v, (pw - 12, y), C_TEXT, "topright")
            y += f.small.get_height() + 2

    def draw_bar(self, surf):
        surf.fill((10, 14, 20))
        w, h = surf.get_size()
        s = ("SPACE turbo · P duraklat · +/- hız · tıkla seç · F takip · S kaydet · Z tüm dünya · R yeni dünya · H yardım"
             if self.show_help else "H: yardım")
        text(surf, self.fonts.small, s, (8, h // 2), C_DIM, "midleft")

    def draw_turbo(self, scr):
        f, w = self.fonts, self.world
        W, H = scr.get_size()
        text(scr, f.huge, "TURBO", (W // 2, int(H * 0.08)), C_POS, "midtop")
        s = w.stats()
        text(scr, f.big, f"t = {s['t']:.0f} s   nüfus {s['pop']}   yem {s['food']}   ort. enerji {s['mean_energy']:.0f}",
             (W // 2, int(H * 0.3)), C_TEXT, "midtop")
        text(scr, f.normal, "SPACE: geri dön", (W // 2, int(H * 0.38)), C_DIM, "midtop")
        self.draw_series(scr, pygame.Rect(int(W * 0.08), int(H * 0.46), int(W * 0.84), int(H * 0.4)),
                         [("pop", C_BEST)], "nüfus")


def write_log(world: World, out: str):
    os.makedirs(out, exist_ok=True)
    cols = ("t", "pop", "food", "mean_energy", "eaten", "mean_limbs", "mean_speed", "births_sex", "births_asex", "mean_gen", "max_gen", "rescued",
            "carn", "bites", "kills", "mean_diet", "carcasses", "carc_eaten")
    with open(os.path.join(out, "log.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for s in world.history:
            fh.write(",".join(f"{s[c]:.4f}" if isinstance(s[c], float) else str(s[c]) for c in cols) + "\n")
    order = np.argsort(-world.energy)[:5]
    for k in order:
        g = world.genomes[int(k)].copy()
        g.meta.update(dict(eco_id=int(world.ids[k]), energy=float(world.energy[k]), age=float(world.t - world.born[k]), physics=4,
                           diet=float(world.diet[k]), lineage=int(world.lin[k])))
        save_json(g, os.path.join(out, f"survivor_{int(world.ids[k]):04d}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=300.0, help="--headless için simülasyon süresi (sn)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=float, default=40.0, help="dünya kenarı (m)")
    ap.add_argument("--sources", type=int, default=5)
    ap.add_argument("--min-pop", type=int, default=0, help="nüfus bunun altına inerse hayatta kalanların mutasyonlu kopyaları eklenir (0=kapalı)")
    ap.add_argument("--random", action="store_true", help="eğitilmiş canlılar yerine RASTGELE beyinli başlangıç (varsayılan: örnek canlılarla tohumla)")
    ap.add_argument("--seed-from", nargs="+", default=None, help="başlangıç nüfusunu bu canlı JSON'larından üret (varsayılan: ornekler/ içindeki 3 canlı)")
    ap.add_argument("--out", default="runs/eco1")
    ap.add_argument("--window", default="1280x720")
    a = ap.parse_args()
    cfg = WorldConfig(n_init=a.n, seed=a.seed, size=a.size, n_sources=a.sources, min_pop=a.min_pop)
    here = os.path.dirname(os.path.abspath(__file__))
    paths = None if a.random else (a.seed_from or [os.path.join(here, "ornekler", f) for f in
                                                  ("denizanasi.json", "yilan.json", "ornek_simetrik_balik.json")])

    def make(c):
        if paths:
            return World(c, genomes=seeded_genomes(paths, c.n_init, np.random.default_rng(c.seed + 12345), c.mut,
                                                   max_limbs=c.max_limbs, carnivore_frac=c.carnivore_frac))
        return World(c)

    if a.headless:
        w = make(cfg)
        run_headless(w, a.seconds, log=print)
        write_log(w, a.out)
        print(f"bitti: nüfus {w.pop}, ölen {len(w.deaths)}, yenen yem {w.eaten_total} -> {a.out}")
        return
    pygame.display.init()
    pygame.font.init()
    W, H = (int(v) for v in a.window.lower().split("x"))
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Evrimleşen Yüzücüler - Ekosistem")
    app = EcoApp(screen, cfg, a.out, make)
    clock = pygame.time.Clock()
    while app.running:
        for e in pygame.event.get():
            app.handle_event(e)
        dt = clock.tick(60) / 1000.0
        app.update(dt)
        app.draw()
    pygame.quit()


if __name__ == "__main__":
    main()

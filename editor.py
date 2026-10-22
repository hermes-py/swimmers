"""
Faz 5 - Canlı tasarım editörü (pygame). app.py içinde üçüncü mod olarak çalışır (TAB ile ya da E tuşuyla).

Fare:
    sol tık (canlı noktası)     seç;  sürükle = o uzvun uzunluğunu ve açısını değiştir (GRİDE YAPIŞIR)
    sol tık (boş yer)           seçili noktaya yeni uzuv ekle (önizleme hayaleti gösterilir)
    sağ tık (nokta)             o uzvu ve alt ağacını sil;  sağ tık (boş) seçimi kaldırır
    orta tuş sürükle / tekerlek kaydır / yakınlaştır (panel üstünde tekerlek: seçili özelliği değiştirir)
Klavye:
    Y simetri aç/kapat    O eski omurga modu aç/kapat    M seçili uzvun aynasını ekle    DEL seçili uzvu sil
    1-8 özellik seç, ↑/↓ değeri değiştir (←/→ özellik değiştir)
    T yüzme testi         V dalga yönü   Q/E dalga frekansı
    N yeni canlı          K/L kütüphaneden canlı yükle (kopyası düzenlenir)
    Ctrl+Z geri al   Ctrl+Y yinele   Ctrl+S kaydet   ENTER: DENE modunda aç (WASD ile kullan)   Z: ortala

Grid: açı 24 adım, uzunluk/kütle/kalınlık/tork/sönüm/sınırlar kademelidir (genome.GRID). Kutupsal kılavuz
(halkalar = uzunluk kademeleri, ışınlar = açı adımları) seçili noktanın etrafında görünür.

Not: Beyin vücuttan bağımsız bir MLP'dir. Yeni tasarlanan gövdeye kopyalanan beyin onu bilmediği için
beklenen gibi yüzmez; 'Yüzme testi' beyinsiz, açı dalgasıyla gövdenin itki üretip üretmediğini gösterir.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pygame

from genome import (FIELDS, GRID, MAX_LIMBS, Genome, add_limb_at, blank_body, delete_set, level_value,
                    mirror_at, n_levels, random_brain, save_json, subtree)
from physics import PHYSICS_VERSION
from sim import Camera, Session, WaveSession
from ui import *

PROPS = [("theta", "Açı (dinlenme)"), ("length", "Uzunluk"), ("mass", "Kütle"), ("width", "Kalınlık"),
         ("tau", "Tork kapasitesi"), ("damp", "Eklem sönümü"), ("lo", "Alt sınır"), ("hi", "Üst sınır")]
ANGLE_STEP = 2 * math.pi / GRID["angle_steps"]


def fmt(field: str, k: int) -> str:
    v = level_value(field, int(k))
    if field == "theta":
        return f"{math.degrees(v):.0f}°"
    if field == "lo":
        return f"-{math.degrees(v):.0f}°"
    if field == "hi":
        return f"+{math.degrees(v):.0f}°"
    if field == "length":
        return f"{v:.2f} m"
    if field == "mass":
        return f"{v:.2f} kg"
    if field == "width":
        return f"{v:.2f} m"
    return f"{v:g}"


class Editor:
    def __init__(self, app):
        self.app = app
        self.cam = Camera()
        self.cam.auto = False
        self.cam.scale = 120.0
        self.tcam = Camera()
        self.body = blank_body()
        self.brain = random_brain(app.rng)
        self.name = "yeni tasarım"
        self.brain_from = "rastgele beyin"
        self.sel = -1
        self.hover = None
        self.prop = "length"
        self.sym = False
        self.pairs: dict[int, int] = {}
        self.undo_stack, self.redo_stack = [], []
        self.drag = None            # (indeks, başlangıç ekran konumu)
        self.drag_moved = False
        self.panning = False
        self.test = None
        self.test_dir, self.test_freq = +1, 1.5
        self.msg, self.msg_until = "", 0.0
        self._sess = None
        self._dirty = True
        self.mouse = (0, 0)
        self.lib_i = 0
        self.confirm = False         # 'Eğitime gönder' onay penceresi açık mı

    # ------------------------------------------------------------------ durum
    def say(self, s, secs=3.5):
        self.msg, self.msg_until = s, time.time() + secs

    def sess(self) -> Session:
        if self._dirty or self._sess is None:
            self._sess = Session(Genome(self.body, self.brain), self.name, heading=HEAD_UP)
            self._dirty = False
        return self._sess

    def open_from(self, genome: Genome, name: str):
        self.body, self.brain = genome.body.copy(), genome.brain.copy()
        self.name = f"{name} (kopya)"
        self.brain_from = name
        self.sel, self.pairs = -1, {}
        self.undo_stack, self.redo_stack = [], []
        self.test = None
        self._dirty = True
        self.center()

    def new_blank(self):
        self.body, self.brain = blank_body(), random_brain(self.app.rng)
        self.name, self.brain_from = "yeni tasarım", "rastgele beyin"
        self.sel, self.pairs = -1, {}
        self.undo_stack, self.redo_stack = [], []
        self.test = None
        self._dirty = True
        self.center()
        self.say("yeni canlı: kafa + 2 uzuv (gövde yönü = kafa)")

    def center(self):
        x = self.sess().st.x[0][: self.body.n + 1]
        self.cam.c = x.mean(0).astype(float)
        ext = float(np.max(np.linalg.norm(x - self.cam.c, axis=1))) if len(x) else 1.0
        w, h = self.app.view_size()
        self.cam.scale = float(np.clip(0.38 * min(w, h) / max(ext, 0.4), 30, 260))

    def push(self):
        self.undo_stack.append((self.body.copy(), dict(self.pairs), self.sel))
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()

    def _restore(self, snap):
        self.body, self.pairs, self.sel = snap[0].copy(), dict(snap[1]), snap[2]
        if self.sel > self.body.n:
            self.sel = -1
        self._dirty = True

    def do_undo(self):
        if self.undo_stack:
            self.redo_stack.append((self.body.copy(), dict(self.pairs), self.sel))
            self._restore(self.undo_stack.pop())
        else:
            self.say("geri alınacak bir şey yok")

    def do_redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.body.copy(), dict(self.pairs), self.sel))
            self._restore(self.redo_stack.pop())

    # ------------------------------------------------------------------ geometri
    def pos(self):
        return self.sess().st.x[0]

    def bone_angle(self, i):
        x = self.pos()
        d = x[i] - x[self.body.parent[i]]
        return math.atan2(d[1], d[0])

    def ref_for_parent(self, p):
        """Parent noktasına eklenecek bir uzvun açı referansı (mutlak yön)."""
        if p == 0:
            return self.bone_angle(1) if self.body.legacy_spine else HEAD_UP    # yeni modda kafa yönü = referans
        return self.bone_angle(p)

    def ref_of(self, i):
        if self.body.legacy_spine:
            return HEAD_UP if i == 1 else self.ref_for_parent(int(self.body.parent[i]))
        return self.ref_for_parent(int(self.body.parent[i]))

    def snap(self, pivot, ref, pw):
        """Dünya konumunu (açı kademesi, uzunluk kademesi)ne yapıştır."""
        v = np.asarray(pw) - self.pos()[pivot]
        dist = float(np.hypot(*v))
        if dist < 0.04:
            return None
        rel = wrap_pi(math.atan2(v[1], v[0]) - ref)
        k = int(round(rel / ANGLE_STEP)) % GRID["angle_steps"]
        li = int(np.argmin(np.abs(np.array(GRID["length"]) - dist)))
        return k, li

    def s2w(self, p, wh):
        return np.array([self.cam.c[0] + (p[0] - wh[0] / 2) / self.cam.scale,
                         self.cam.c[1] - (p[1] - wh[1] / 2) / self.cam.scale])

    def hit(self, p, wh):
        x = self.pos()
        best, bd_ = None, 1e9
        for i in range(self.body.n + 1):
            sp = self.cam.w2s(x[i], wh)
            d = math.hypot(sp[0] - p[0], sp[1] - p[1])
            r = max(13, 0.1 * math.sqrt(level_value("mass", int(self.body.mass[i]))) * self.cam.scale + 4)
            if d <= r and d < bd_:
                best, bd_ = i, d
        return best

    # ------------------------------------------------------------------ düzenleme işlemleri
    def set_field(self, i, field, level):
        b = self.body
        arr = getattr(b, field)
        arr[i] = int(level) % n_levels(field) if field == "theta" else int(np.clip(level, 0, n_levels(field) - 1))
        self._dirty = True
        if self.sym and i >= 1 and i in self.pairs:
            self._twin(i, self.pairs[i], field)

    def _twin(self, i, j, field):
        b = self.body
        if field == "theta":
            b.theta[j] = (GRID["angle_steps"] - b.theta[i]) % GRID["angle_steps"]
        elif field == "lo":
            b.hi[j] = b.lo[i]
        elif field == "hi":
            b.lo[j] = b.hi[i]
        else:
            getattr(b, field)[j] = getattr(b, field)[i]

    def change_level(self, delta):
        if self.sel < 0:
            self.say("önce bir nokta seç")
            return
        f = self.prop
        if self.sel == 0 and f != "mass":
            self.say("kafanın sadece kütlesi ayarlanır")
            return
        if self.sel == 1 and f in ("tau", "damp", "lo", "hi") and self.body.legacy_spine:
            self.say("kemik 1 pasif omurgadır: motor/eklem özellikleri kullanılmaz")
        self.push()
        self.set_field(self.sel, f, int(getattr(self.body, f)[self.sel]) + delta)

    def add_at(self, pw):
        b = self.body
        if b.n >= MAX_LIMBS:
            self.say(f"en fazla {MAX_LIMBS} uzuv")
            return
        s = self.sel
        sn = self.snap(s, self.ref_for_parent(s), pw)
        if sn is None:
            return
        k, li = sn
        A = GRID["angle_steps"]
        self.push()
        tmpl = s if s >= 1 else (b.n if b.n >= 2 else None)
        nb = add_limb_at(b, s, k, li, template=tmpl)
        new = nb.n
        if self.sym and nb.n < MAX_LIMBS:
            tp = self.pairs.get(s, s)
            tk = (A - k) % A
            if not (tp == s and tk == k):                  # eksen üzerindeyse ayna aynı yere düşer
                nb = add_limb_at(nb, tp, tk, li, template=new)
                t = nb.n
                nb.lo[t], nb.hi[t] = nb.hi[new], nb.lo[new]
                self.pairs[new], self.pairs[t] = t, new
        self.body, self.sel = nb, new
        self._dirty = True

    def mirror_selected(self):
        if self.sel < self.body.first_free:
            self.say("ayna için bir uzuv seç" if not self.body.legacy_spine else "ayna için omurga dışında bir uzuv seç")
            return
        r = mirror_at(self.body, self.sel)
        if r is None:
            self.say(f"aynalanamadı (en fazla {MAX_LIMBS} uzuv)")
            return
        self.push()
        self.body = r[0]
        for old, new in r[1].items():
            self.pairs[old], self.pairs[new] = new, old
        self._dirty = True
        self.say("ayna eklendi")

    def delete_at(self, i):
        if i < self.body.first_free:
            self.say("kafa silinemez" if not self.body.legacy_spine else "kafa ve kemik 1 (omurga) silinemez")
            return
        rm = set(subtree(self.body, i))
        if self.sym and i in self.pairs:
            rm |= set(subtree(self.body, self.pairs[i]))
        r = delete_set(self.body, rm)
        if r is None:
            self.say(f"silinemez (en az {self.body.min_n} uzuv kalmalı)")
            return
        self.push()
        self.body, remap = r
        self.pairs = {remap[a]: remap[c] for a, c in self.pairs.items() if a in remap and c in remap}
        self.sel = remap.get(self.sel, -1)
        self._dirty = True

    def toggle_sym(self):
        self.sym = not self.sym
        self.say("simetri AÇIK: eklenen/değiştirilen uzuvların aynası da güncellenir (eksen: kafanın yönü)" if self.sym
                 else "simetri kapalı")

    def toggle_spine(self):
        self.push()
        self.body.legacy_spine = not self.body.legacy_spine
        self._dirty = True
        self.say("ESKİ omurga modu: kemik 1 pasif, gövde yönünü o belirler (kafaya bağlı uzuvların açıları değişir!)"
                 if self.body.legacy_spine else "yeni mod: gövde yönü = kafanın yönü, tüm uzuvlar motorlu")

    def toggle_test(self):
        if self.test is not None:
            self.test = None
            return
        g = Genome(self.body.copy(), self.brain.copy())
        self.test = WaveSession(g, "yüzme testi", heading=HEAD_UP, freq=self.test_freq, direction=self.test_dir)
        self.tcam = Camera()

    def save(self):
        os.makedirs(self.app.out, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(self.app.out, f"design_{i:03d}.json")):
            i += 1
        path = os.path.join(self.app.out, f"design_{i:03d}.json")
        save_json(self.genome(), path)
        self.app.rescan()
        self.say(f"kaydedildi: {path}", 5.0)

    def genome(self) -> Genome:
        return Genome(self.body.copy(), self.brain.copy(), {"designed": True, "physics": PHYSICS_VERSION})

    def send_to_play(self):
        self.app.open_in_play(self.genome(), self.name)

    def ask_training(self):
        if self.test is not None:
            self.test = None
        self.confirm = True

    def start_training(self, freeze: bool):
        self.confirm = False
        self.app.start_training_from(self.genome(), freeze, self.name)

    def modal_rects(self, view_size):
        vw, vh = view_size
        f = self.app.fonts
        w = int(min(640, vw * 0.86))
        bh = int(f.normal.get_height() * 2.0)
        h = int(f.big.get_height() + f.small.get_height() * 6 + 3 * bh + 70)
        box = pygame.Rect((vw - w) // 2, (vh - h) // 2, w, h)
        y = box.bottom - 3 * bh - 36
        b1 = pygame.Rect(box.x + 18, y, w - 36, bh)
        b2 = pygame.Rect(box.x + 18, y + bh + 8, w - 36, bh)
        b3 = pygame.Rect(box.x + 18, y + 2 * (bh + 8), w - 36, bh)
        return box, b1, b2, b3

    def lib_step(self, d):
        lib = self.app.library
        if not lib:
            return
        self.lib_i = (self.lib_i + d) % len(lib)
        item = lib[self.lib_i]
        self.open_from(item["genome"], item["name"])
        self.say(f"yüklendi: {item['name']} (kopya düzenleniyor)")

    # ------------------------------------------------------------------ panel yerleşimi
    def layout_panel(self, panel):
        f = self.app.fonts
        pw = panel.w
        y = 8 + f.big.get_height() + f.small.get_height() * 2 + 14
        bh = int(f.normal.get_height() * 1.45)
        gap = 5
        bw = (pw - 24 - gap) // 2
        sym_label = f"Simetri: {'AÇIK' if self.sym else 'kapalı'} (Y)"
        specs = [(sym_label, self.toggle_sym), ("Ayna ekle (M)", self.mirror_selected),
                 (("Testi durdur" if self.test else "Yüzme testi") + " (T)", self.toggle_test), ("Yeni (N)", self.new_blank),
                 (f"Eski omurga modu: {'AÇIK' if self.body.legacy_spine else 'kapalı'} (O)", self.toggle_spine),
                 ("Geri al", self.do_undo), ("Yinele", self.do_redo),
                 ("Kaydet", self.save), ("DENE'de aç ↵", self.send_to_play),
                 ("Eğitime gönder", self.ask_training), ("Ortala (Z)", self.center)]
        buttons = []
        for n, (label, fn) in enumerate(specs):
            r = pygame.Rect(12 + (n % 2) * (bw + gap), y + (n // 2) * (bh + gap), bw, bh)
            buttons.append((r, label, fn))
        y += ((len(specs) + 1) // 2) * (bh + gap) + 6
        rows = []
        if self.test is None:
            y += f.small.get_height() + 4         # "Seçili: ..." başlığı için yer
            rh = int(f.normal.get_height() * 1.35)
            fields = PROPS if self.sel >= 1 else (PROPS[2:3] if self.sel == 0 else [])
            for field, label in fields:
                r = pygame.Rect(12, y, pw - 24, rh)
                bsz = rh - 4
                minus = pygame.Rect(r.right - 2 * bsz - 8, r.y + 2, bsz, bsz)
                plus = pygame.Rect(r.right - bsz - 2, r.y + 2, bsz, bsz)
                rows.append((field, label, r, minus, plus))
                y += rh + 2
        return buttons, rows, y

    # ------------------------------------------------------------------ olaylar
    def handle_event(self, e, view, panel) -> bool:
        wh = view.size
        if self.confirm:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                box, b1, b2, b3 = self.modal_rects(wh)
                if b1.collidepoint(e.pos):
                    self.start_training(True)
                elif b2.collidepoint(e.pos):
                    self.start_training(False)
                elif b3.collidepoint(e.pos) or not box.collidepoint(e.pos):
                    self.confirm = False
                return True
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_1:
                    self.start_training(True)
                elif e.key == pygame.K_2:
                    self.start_training(False)
                elif e.key in (pygame.K_n, pygame.K_c, pygame.K_BACKSPACE):
                    self.confirm = False
                return True
            return True
        if e.type == pygame.MOUSEMOTION:
            self.mouse = e.pos
            self.hover = self.hit(e.pos, wh) if view.collidepoint(e.pos) and self.test is None else None
            if self.panning:
                self.cam.c[0] -= e.rel[0] / self.cam.scale
                self.cam.c[1] += e.rel[1] / self.cam.scale
            if self.drag and self.test is None and e.buttons[0]:
                i, start = self.drag
                if not self.drag_moved and math.hypot(e.pos[0] - start[0], e.pos[1] - start[1]) > 6:
                    self.push()
                    self.drag_moved = True
                if self.drag_moved:
                    sn = self.snap(int(self.body.parent[i]), self.ref_of(i), self.s2w(e.pos, wh))
                    if sn is not None:
                        self.set_field(i, "theta", sn[0])
                        self.set_field(i, "length", sn[1])
            return True
        if e.type == pygame.MOUSEBUTTONUP:
            if e.button == 1:
                self.drag, self.drag_moved = None, False
            if e.button == 2:
                self.panning = False
            return True
        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 2:
                self.panning = True
                return True
            in_panel = panel.collidepoint(e.pos)
            if in_panel:
                if e.button == 1:
                    buttons, rows, _ = self.layout_panel(panel)
                    lp = (e.pos[0] - panel.x, e.pos[1] - panel.y)
                    for r, label, fn in buttons:
                        if r.collidepoint(lp):
                            fn()
                            return True
                    for field, label, r, minus, plus in rows:
                        if minus.collidepoint(lp):
                            self.prop = field
                            self.change_level(-1)
                            return True
                        if plus.collidepoint(lp):
                            self.prop = field
                            self.change_level(+1)
                            return True
                        if r.collidepoint(lp):
                            self.prop = field
                            return True
                return True
            if not view.collidepoint(e.pos) or self.test is not None:
                return True
            i = self.hit(e.pos, wh)
            if e.button == 1:
                if i is not None:
                    self.sel = i
                    self.drag, self.drag_moved = ((i, e.pos) if i >= 1 else None), False
                elif self.sel >= 0:
                    self.add_at(self.s2w(e.pos, wh))
            elif e.button == 3:
                if i is not None:
                    self.delete_at(i)
                else:
                    self.sel = -1
            return True
        if e.type == pygame.MOUSEWHEEL:
            if panel.collidepoint(pygame.mouse.get_pos()) and self.test is None:
                self.change_level(+1 if e.y > 0 else -1)
            else:
                self.cam.scale = float(np.clip(self.cam.scale * (1.12 ** e.y), 20, 500))
            return True
        if e.type == pygame.KEYDOWN:
            return self.handle_key(e)
        return False

    def handle_key(self, e) -> bool:
        k = e.key
        ctrl = bool(getattr(e, "mod", 0) & pygame.KMOD_CTRL)
        shift = bool(getattr(e, "mod", 0) & pygame.KMOD_SHIFT)
        if ctrl:
            if k == pygame.K_z and not shift:
                self.do_undo()
            elif k == pygame.K_y or (k == pygame.K_z and shift):
                self.do_redo()
            elif k == pygame.K_s:
                self.save()
            else:
                return False
            return True
        if k == pygame.K_y:
            self.toggle_sym()
        elif k == pygame.K_m:
            self.mirror_selected()
        elif k == pygame.K_o:
            self.toggle_spine()
        elif k == pygame.K_t:
            self.toggle_test()
        elif k == pygame.K_v:
            self.test_dir = -self.test_dir
            if self.test:
                self.test.direction = self.test_dir
        elif k == pygame.K_q:
            self.test_freq = max(0.25, self.test_freq - 0.25)
            if self.test:
                self.test.freq = self.test_freq
        elif k == pygame.K_e:
            self.test_freq = min(6.0, self.test_freq + 0.25)
            if self.test:
                self.test.freq = self.test_freq
        elif k == pygame.K_n:
            self.new_blank()
        elif k == pygame.K_k:
            self.lib_step(-1)
        elif k == pygame.K_l:
            self.lib_step(+1)
        elif k == pygame.K_RETURN:
            self.send_to_play()
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE):
            if self.sel >= 0 and self.test is None:
                self.delete_at(self.sel)
        elif k == pygame.K_UP and self.test is None:
            self.change_level(+1)
        elif k == pygame.K_DOWN and self.test is None:
            self.change_level(-1)
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            keys = [p[0] for p in PROPS]
            self.prop = keys[(keys.index(self.prop) + (1 if k == pygame.K_RIGHT else -1)) % len(keys)]
        elif pygame.K_1 <= k <= pygame.K_8:
            self.prop = PROPS[k - pygame.K_1][0]
        elif k == pygame.K_z:
            self.center()
        else:
            return False
        return True

    def update(self, dt, view_wh):
        if self.test is not None:
            self.test.advance(min(dt, 0.1) * self.app.time_scale, 0.0, 1.0, True)
            self.tcam.update(self.test, view_wh, min(dt, 0.1))

    # ------------------------------------------------------------------ çizim
    def draw(self, vsurf, psurf, view, panel):
        if self.test is not None:
            self.app.draw_world(vsurf, self.test, self.tcam)
            self.draw_test_hud(vsurf)
        else:
            self.draw_design(vsurf)
        self.draw_panel(psurf, panel)
        if self.confirm:
            self.draw_modal(vsurf)

    def draw_design(self, surf):
        f = self.app.fonts
        vw, vh = surf.get_size()
        wh = (vw, vh)
        surf.fill((7, 22, 40))
        cam, sc = self.cam, self.cam.scale
        x0, x1 = cam.c[0] - vw / 2 / sc, cam.c[0] + vw / 2 / sc
        y0, y1 = cam.c[1] - vh / 2 / sc, cam.c[1] + vh / 2 / sc
        for stp, col, minsc in ((0.1, (14, 40, 66), 55), (0.5, (22, 62, 98), 22), (1.0, (34, 90, 134), 8)):
            if sc < minsc:
                continue
            for gx in range(int(math.floor(x0 / stp)), int(math.ceil(x1 / stp)) + 1):
                px = vw / 2 + (gx * stp - cam.c[0]) * sc
                pygame.draw.line(surf, (60, 130, 180) if gx == 0 else col, (px, 0), (px, vh))
            for gy in range(int(math.floor(y0 / stp)), int(math.ceil(y1 / stp)) + 1):
                py = vh / 2 - (gy * stp - cam.c[1]) * sc
                pygame.draw.line(surf, (60, 130, 180) if gy == 0 else col, (0, py), (vw, py))
        sess = self.sess()
        x = self.pos()
        b = self.body
        # kutupsal kılavuz (seçili nokta / sürüklenen uzvun pivotu)
        pivot = ref = None
        if self.drag and self.drag_moved:
            pivot, ref = int(b.parent[self.drag[0]]), self.ref_of(self.drag[0])
        elif self.sel >= 0:
            pivot, ref = self.sel, self.ref_for_parent(self.sel)
        snapped = None
        if pivot is not None and self.mouse and self.hover is None:
            snapped = self.snap(pivot, ref, self.s2w(self.mouse, wh))
        if pivot is not None:
            pc = cam.w2s(x[pivot], wh)
            Rmax = GRID["length"][-1] * sc
            for li, L in enumerate(GRID["length"]):
                hot = snapped is not None and snapped[1] == li
                pygame.draw.circle(surf, (230, 200, 90) if hot else (40, 80, 110), pc, max(2, int(L * sc)), 2 if hot else 1)
            for k in range(GRID["angle_steps"]):
                a = ref + k * ANGLE_STEP
                hot = snapped is not None and snapped[0] == k
                col = (230, 200, 90) if hot else ((70, 120, 160) if k % 6 == 0 else (34, 64, 92))
                pygame.draw.line(surf, col, pc, (pc[0] + math.cos(a) * Rmax, pc[1] - math.sin(a) * Rmax), 2 if hot else 1)
        # seçili eklemin açı sınırları (kama)
        if self.sel >= self.body.first_free:
            i = self.sel
            p = int(b.parent[i])
            pc = cam.w2s(x[p], wh)
            base = self.ref_of(i) + level_value("theta", int(b.theta[i]))
            lo, hi = -level_value("lo", int(b.lo[i])), level_value("hi", int(b.hi[i]))
            R = level_value("length", int(b.length[i])) * sc * 0.85
            pts = [pc] + [(pc[0] + math.cos(base + lo + (hi - lo) * t / 24) * R,
                           pc[1] - math.sin(base + lo + (hi - lo) * t / 24) * R) for t in range(25)]
            wedge = pygame.Surface((vw, vh), pygame.SRCALPHA)
            pygame.draw.polygon(wedge, (255, 210, 90, 55), pts)
            surf.blit(wedge, (0, 0))
        # hayalet (eklenecek uzuv önizlemesi)
        if self.sel >= 0 and self.hover is None and snapped is not None and not (self.drag and self.drag_moved):
            k, li = snapped
            a = self.ref_for_parent(self.sel) + k * ANGLE_STEP
            L = GRID["length"][li]
            p0 = cam.w2s(x[self.sel], wh)
            p1 = (p0[0] + math.cos(a) * L * sc, p0[1] - math.sin(a) * L * sc)
            col = (255, 110, 110) if b.n >= MAX_LIMBS else (120, 255, 160)
            pygame.draw.line(surf, col, p0, p1, 3)
            pygame.draw.circle(surf, col, p1, 6, 2)
            text(surf, f.small, f"{L:.2f} m · {math.degrees(wrap_pi(k * ANGLE_STEP)):.0f}°", (p1[0] + 8, p1[1] - 8), col)
        draw_creature(surf, sess, cam, wh)
        # seçim, hover, indeksler
        for i in range(b.n + 1):
            sp = cam.w2s(x[i], wh)
            if i == self.sel:
                pygame.draw.circle(surf, (255, 220, 80), sp, max(13, int(0.12 * sc)), 3)
            elif i == self.hover:
                pygame.draw.circle(surf, (160, 200, 255), sp, max(12, int(0.11 * sc)), 2)
            label = "K" if i == 0 else str(i)
            text(surf, f.small, label, (sp[0] + 9, sp[1] - 18), (255, 240, 170) if i == self.sel else C_DIM)
            if i in self.pairs and self.sym and i >= 1:
                pygame.draw.circle(surf, (190, 140, 255), (sp[0] + 4, sp[1] + 14), 3)
        # ipuçları
        hint = ("sol tık: seç/sürükle · boşa tık: seçili noktaya uzuv ekle · sağ tık: sil · "
                "tekerlek: zoom · orta tuş: kaydır")
        text(surf, f.small, hint, (10, 8), C_DIM)
        if self.msg and time.time() < self.msg_until:
            text(surf, f.normal, self.msg, (10, 8 + f.small.get_height() + 6), (255, 220, 120))
        if self.sym:
            text(surf, f.normal, "SİMETRİ AÇIK", (vw - 10, 8), (190, 140, 255), "topright")

    def draw_modal(self, surf):
        f = self.app.fonts
        dim = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        box, b1, b2, b3 = self.modal_rects(surf.get_size())
        pygame.draw.rect(surf, (18, 26, 40), box, border_radius=10)
        pygame.draw.rect(surf, (190, 140, 255), box, 2, border_radius=10)
        text(surf, f.big, "Eğitime gönder", (box.x + 18, box.y + 12), (190, 140, 255))
        y = box.y + 14 + f.big.get_height()
        lines = [f"Tasarım ({self.body.n} uzuv) yeni bir eğitim koşusunun başlangıcı olur.",
                 "Şu anki eğitim durdurulup kaydedilir; yeni koşu ayrı bir klasöre yazılır.",
                 "Sabit vücut: sadece beyin evrimleşir (tasarımını aynen korur).",
                 "Evrimleşen vücut: tasarım başlangıç noktasıdır, uzuvlar mutasyona uğrar."]
        for ln in lines:
            text(surf, f.small, ln, (box.x + 18, y + 4), C_TEXT)
            y += f.small.get_height() + 3
        mp = pygame.mouse.get_pos()
        for r, label in ((b1, "1 · Vücut SABİT — sadece beyin evrimleşsin"), (b2, "2 · Vücut da evrimleşsin"), (b3, "İptal")):
            on = r.collidepoint(mp)
            pygame.draw.rect(surf, (52, 70, 104) if on else (30, 42, 62), r, border_radius=6)
            pygame.draw.rect(surf, C_PANEL_LINE, r, 1, border_radius=6)
            text(surf, f.normal, label, r.center, C_TEXT, "center")

    def draw_test_hud(self, surf):
        f = self.app.fonts
        vw, vh = surf.get_size()
        t = self.test
        tl = t.tele
        hd = np.array([math.cos(tl["heading"]), math.sin(tl["heading"])])
        fwd = float(tl["vel"] @ hd)
        text(surf, f.big, "YÜZME TESTİ (beyinsiz açı dalgası)", (10, 8), C_VEL)
        text(surf, f.normal, f"hız {tl['speed']:.2f} m/s · gövde yönünde {fwd:+.2f} m/s · dalga frekansı {self.test_freq:.2f} Hz · "
                             f"yön {'baştan uca' if self.test_dir > 0 else 'uçtan başa'}",
             (10, 8 + f.big.get_height() + 4), C_TEXT)
        text(surf, f.small, "T: testi bitir · V: dalga yönü · Q/E: frekans · hız tuşları +/-", (10, 8 + f.big.get_height() + f.normal.get_height() + 10), C_DIM)
        if self.msg and time.time() < self.msg_until:
            text(surf, f.normal, self.msg, (10, vh - f.normal.get_height() - 10), (255, 220, 120))

    def draw_panel(self, surf, panel):
        f = self.app.fonts
        pw, ph = surf.get_size()
        pygame.draw.line(surf, C_PANEL_LINE, (0, 0), (0, ph), 2)
        text(surf, f.big, "EDİTÖR", (12, 8), (190, 140, 255))
        y = 8 + f.big.get_height() + 2
        text(surf, f.small, self.name[:44], (12, y), C_DIM)
        y += f.small.get_height()
        b = self.body
        mass = sum(level_value("mass", int(b.mass[i])) for i in range(b.n + 1))
        text(surf, f.small, f"uzuv {b.n}/{MAX_LIMBS} · toplam kütle {mass:.1f} kg · beyin: {self.brain_from}"[:60], (12, y), C_DIM)
        buttons, rows, y = self.layout_panel(panel)
        mp = pygame.mouse.get_pos()
        lp = (mp[0] - panel.x, mp[1] - panel.y)
        for r, label, fn in buttons:
            on = r.collidepoint(lp)
            pygame.draw.rect(surf, (44, 58, 84) if on else (28, 38, 56), r, border_radius=6)
            pygame.draw.rect(surf, C_PANEL_LINE, r, 1, border_radius=6)
            text(surf, f.small, label, r.center, C_TEXT, "center")
        if self.test is None:
            title = "Kafa" if self.sel == 0 else (f"Seçili uzuv {self.sel}" + (("  (eski mod: pasif omurga)" if self.body.legacy_spine else "") if self.sel == 1 else "")) if self.sel > 0 else "Bir nokta seç"
            ty = rows[0][2].y - f.small.get_height() - 3 if rows else y + 4
            text(surf, f.small, title, (12, ty), C_HEAD)
            for field, label, r, minus, plus in rows:
                active = field == self.prop
                pygame.draw.rect(surf, (38, 52, 76) if active else (22, 30, 44), r, border_radius=4)
                if active:
                    pygame.draw.rect(surf, (190, 140, 255), r, 1, border_radius=4)
                k = int(getattr(self.body, field)[self.sel])
                disabled = self.sel == 1 and field in ("tau", "damp", "lo", "hi") and self.body.legacy_spine
                text(surf, f.small, label, (r.x + 6, r.centery), C_DIM if disabled else C_TEXT, "midleft")
                text(surf, f.small, f"{fmt(field, k)}  ({k + 1}/{n_levels(field)})", (minus.x - 8, r.centery), C_VEL, "midright")
                for rc, s_ in ((minus, "−"), (plus, "+")):
                    pygame.draw.rect(surf, (52, 70, 100) if rc.collidepoint(lp) else (34, 46, 68), rc, border_radius=4)
                    text(surf, f.normal, s_, rc.center, C_TEXT, "center")
            y = (rows[-1][2].bottom if rows else y) + 10
            for ln in ("Yapışma: açı 24 adım (15°), uzunluk 8 kademe.",
                       "Boşa tıklayınca seçili noktaya uzuv eklenir.",
                       "↑/↓ ya da tekerlek: seçili özellik değeri;",
                       "←/→ ya da 1-8: özellik seç.",
                       "K/L: kütüphaneden canlı yükle (kopya)."):
                if y + f.small.get_height() > ph - 4:
                    break
                text(surf, f.small, ln, (12, y), C_DIM)
                y += f.small.get_height() + 1
        else:
            tl = self.test.tele
            lines = [f"hız: {tl['speed']:.2f} m/s", f"t = {tl['t']:.1f} s", "",
                     "Bu test beyin kullanmaz: her kolda", "gövdeden uca yayılan açı dalgası", "uygulanır. İtki üretiyorsa",
                     "gövde suda yüzebilir demektir.", "", "T: testi bitir ve düzenlemeye dön"]
            for ln in lines:
                text(surf, f.small, ln, (12, y), C_TEXT)
                y += f.small.get_height() + 2

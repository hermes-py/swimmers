"""
Phase 5 - Creature design editor (pygame). Runs inside app.py as the third mode (with TAB or the E key).

Mouse:
    left click (creature point)  select;  drag = change that limb's length and angle (SNAPS TO THE GRID)
    left click (empty space)     add a new limb to the selected point (a preview ghost is shown)
    right click (point)          delete that limb and its subtree;  right click (empty) clears the selection
    middle button drag / wheel   pan / zoom (wheel over the panel: changes the selected property)
Keyboard:
    Y toggle symmetry    O toggle old spine mode    M add mirror of the selected limb    DEL delete the selected limb
    1-8 select a property, ↑/↓ change the value (←/→ change property)
    T swim test          V wave direction   Q/E wave frequency
    N new creature       K/L load a creature from the library (a copy is edited)
    Ctrl+Z undo   Ctrl+Y redo   Ctrl+S save   ENTER: open in PLAY mode (use with WASD)   Z: center

Grid: angle 24 steps; length/mass/width/torque/damping/limits are discrete levels (genome.GRID). A polar guide
(rings = length levels, rays = angle steps) is shown around the selected point.

Note: The brain is an MLP independent of the body. A brain copied onto a newly designed body does not know it,
so it will not swim as expected; the 'Swim test' shows, without a brain and with an angle wave, whether the body produces thrust.
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

PROPS = [("theta", "Angle (rest)"), ("length", "Length"), ("mass", "Mass"), ("width", "Thickness"),
         ("tau", "Torque capacity"), ("damp", "Joint damping"), ("lo", "Lower limit"), ("hi", "Upper limit")]
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
        self.name = "new design"
        self.brain_from = "random brain"
        self.sel = -1
        self.hover = None
        self.prop = "length"
        self.sym = False
        self.pairs: dict[int, int] = {}
        self.undo_stack, self.redo_stack = [], []
        self.drag = None            # (index, starting screen position)
        self.drag_moved = False
        self.panning = False
        self.test = None
        self.test_dir, self.test_freq = +1, 1.5
        self.msg, self.msg_until = "", 0.0
        self._sess = None
        self._dirty = True
        self.mouse = (0, 0)
        self.lib_i = 0
        self.confirm = False         # is the 'Send to training' confirmation dialog open

    # ------------------------------------------------------------------ state
    def say(self, s, secs=3.5):
        self.msg, self.msg_until = s, time.time() + secs

    def sess(self) -> Session:
        if self._dirty or self._sess is None:
            self._sess = Session(Genome(self.body, self.brain), self.name, heading=HEAD_UP)
            self._dirty = False
        return self._sess

    def open_from(self, genome: Genome, name: str):
        self.body, self.brain = genome.body.copy(), genome.brain.copy()
        self.name = f"{name} (copy)"
        self.brain_from = name
        self.sel, self.pairs = -1, {}
        self.undo_stack, self.redo_stack = [], []
        self.test = None
        self._dirty = True
        self.center()

    def new_blank(self):
        self.body, self.brain = blank_body(), random_brain(self.app.rng)
        self.name, self.brain_from = "new design", "random brain"
        self.sel, self.pairs = -1, {}
        self.undo_stack, self.redo_stack = [], []
        self.test = None
        self._dirty = True
        self.center()
        self.say("new creature: head + 2 limbs (body direction = head)")

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
            self.say("nothing to undo")

    def do_redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.body.copy(), dict(self.pairs), self.sel))
            self._restore(self.redo_stack.pop())

    # ------------------------------------------------------------------ geometry
    def pos(self):
        return self.sess().st.x[0]

    def bone_angle(self, i):
        x = self.pos()
        d = x[i] - x[self.body.parent[i]]
        return math.atan2(d[1], d[0])

    def ref_for_parent(self, p):
        """Angle reference (absolute direction) of a limb to be added to the parent point."""
        if p == 0:
            return self.bone_angle(1) if self.body.legacy_spine else HEAD_UP    # in the new mode head direction = reference
        return self.bone_angle(p)

    def ref_of(self, i):
        if self.body.legacy_spine:
            return HEAD_UP if i == 1 else self.ref_for_parent(int(self.body.parent[i]))
        return self.ref_for_parent(int(self.body.parent[i]))

    def snap(self, pivot, ref, pw):
        """Snap a world position to (angle level, length level)."""
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

    # ------------------------------------------------------------------ editing operations
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
            self.say("select a point first")
            return
        f = self.prop
        if self.sel == 0 and f != "mass":
            self.say("only the head's mass can be adjusted")
            return
        if self.sel == 1 and f in ("tau", "damp", "lo", "hi") and self.body.legacy_spine:
            self.say("bone 1 is a passive spine: motor/joint properties are not used")
        self.push()
        self.set_field(self.sel, f, int(getattr(self.body, f)[self.sel]) + delta)

    def add_at(self, pw):
        b = self.body
        if b.n >= MAX_LIMBS:
            self.say(f"at most {MAX_LIMBS} limbs")
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
            if not (tp == s and tk == k):                  # if it is on the axis the mirror falls in the same place
                nb = add_limb_at(nb, tp, tk, li, template=new)
                t = nb.n
                nb.lo[t], nb.hi[t] = nb.hi[new], nb.lo[new]
                self.pairs[new], self.pairs[t] = t, new
        self.body, self.sel = nb, new
        self._dirty = True

    def mirror_selected(self):
        if self.sel < self.body.first_free:
            self.say("select a limb to mirror" if not self.body.legacy_spine else "select a limb other than the spine to mirror")
            return
        r = mirror_at(self.body, self.sel)
        if r is None:
            self.say(f"could not be mirrored (at most {MAX_LIMBS} limbs)")
            return
        self.push()
        self.body = r[0]
        for old, new in r[1].items():
            self.pairs[old], self.pairs[new] = new, old
        self._dirty = True
        self.say("mirror added")

    def delete_at(self, i):
        if i < self.body.first_free:
            self.say("the head cannot be deleted" if not self.body.legacy_spine else "the head and bone 1 (spine) cannot be deleted")
            return
        rm = set(subtree(self.body, i))
        if self.sym and i in self.pairs:
            rm |= set(subtree(self.body, self.pairs[i]))
        r = delete_set(self.body, rm)
        if r is None:
            self.say(f"cannot be deleted (at least {self.body.min_n} limbs must remain)")
            return
        self.push()
        self.body, remap = r
        self.pairs = {remap[a]: remap[c] for a, c in self.pairs.items() if a in remap and c in remap}
        self.sel = remap.get(self.sel, -1)
        self._dirty = True

    def toggle_sym(self):
        self.sym = not self.sym
        self.say("symmetry ON: the mirror of added/changed limbs is updated too (axis: head direction)" if self.sym
                 else "symmetry off")

    def toggle_spine(self):
        self.push()
        self.body.legacy_spine = not self.body.legacy_spine
        self._dirty = True
        self.say("OLD spine mode: bone 1 is passive, it determines the body direction (angles of limbs attached to the head change!)"
                 if self.body.legacy_spine else "new mode: body direction = head direction, all limbs are motorized")

    def toggle_test(self):
        if self.test is not None:
            self.test = None
            return
        g = Genome(self.body.copy(), self.brain.copy())
        self.test = WaveSession(g, "swim test", heading=HEAD_UP, freq=self.test_freq, direction=self.test_dir)
        self.tcam = Camera()

    def save(self):
        os.makedirs(self.app.out, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(self.app.out, f"design_{i:03d}.json")):
            i += 1
        path = os.path.join(self.app.out, f"design_{i:03d}.json")
        save_json(self.genome(), path)
        self.app.rescan()
        self.say(f"saved: {path}", 5.0)

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
        self.say(f"loaded: {item['name']} (editing a copy)")

    # ------------------------------------------------------------------ panel layout
    def layout_panel(self, panel):
        f = self.app.fonts
        pw = panel.w
        y = 8 + f.big.get_height() + f.small.get_height() * 2 + 14
        bh = int(f.normal.get_height() * 1.45)
        gap = 5
        bw = (pw - 24 - gap) // 2
        sym_label = f"Symmetry: {'ON' if self.sym else 'off'} (Y)"
        specs = [(sym_label, self.toggle_sym), ("Add mirror (M)", self.mirror_selected),
                 (("Stop test" if self.test else "Swim test") + " (T)", self.toggle_test), ("New (N)", self.new_blank),
                 (f"Old spine mode: {'ON' if self.body.legacy_spine else 'off'} (O)", self.toggle_spine),
                 ("Undo", self.do_undo), ("Redo", self.do_redo),
                 ("Save", self.save), ("Open in PLAY ↵", self.send_to_play),
                 ("Send to training", self.ask_training), ("Center (Z)", self.center)]
        buttons = []
        for n, (label, fn) in enumerate(specs):
            r = pygame.Rect(12 + (n % 2) * (bw + gap), y + (n // 2) * (bh + gap), bw, bh)
            buttons.append((r, label, fn))
        y += ((len(specs) + 1) // 2) * (bh + gap) + 6
        rows = []
        if self.test is None:
            y += f.small.get_height() + 4         # room for the "Selected: ..." heading
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

    # ------------------------------------------------------------------ events
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

    # ------------------------------------------------------------------ drawing
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
        # polar guide (the selected point / the pivot of the limb being dragged)
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
        # angle limits of the selected joint (wedge)
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
        # ghost (preview of the limb to be added)
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
        # selection, hover, indices
        for i in range(b.n + 1):
            sp = cam.w2s(x[i], wh)
            if i == self.sel:
                pygame.draw.circle(surf, (255, 220, 80), sp, max(13, int(0.12 * sc)), 3)
            elif i == self.hover:
                pygame.draw.circle(surf, (160, 200, 255), sp, max(12, int(0.11 * sc)), 2)
            label = "H" if i == 0 else str(i)
            text(surf, f.small, label, (sp[0] + 9, sp[1] - 18), (255, 240, 170) if i == self.sel else C_DIM)
            if i in self.pairs and self.sym and i >= 1:
                pygame.draw.circle(surf, (190, 140, 255), (sp[0] + 4, sp[1] + 14), 3)
        # hints
        hint = ("left click: select/drag · click on empty space: add a limb to the selected point · right click: delete · "
                "wheel: zoom · middle button: pan")
        text(surf, f.small, hint, (10, 8), C_DIM)
        if self.msg and time.time() < self.msg_until:
            text(surf, f.normal, self.msg, (10, 8 + f.small.get_height() + 6), (255, 220, 120))
        if self.sym:
            text(surf, f.normal, "SYMMETRY ON", (vw - 10, 8), (190, 140, 255), "topright")

    def draw_modal(self, surf):
        f = self.app.fonts
        dim = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        box, b1, b2, b3 = self.modal_rects(surf.get_size())
        pygame.draw.rect(surf, (18, 26, 40), box, border_radius=10)
        pygame.draw.rect(surf, (190, 140, 255), box, 2, border_radius=10)
        text(surf, f.big, "Send to training", (box.x + 18, box.y + 12), (190, 140, 255))
        y = box.y + 14 + f.big.get_height()
        lines = [f"The design ({self.body.n} limbs) becomes the starting point of a new training run.",
                 "The current training is stopped and saved; the new run is written to a separate folder.",
                 "Fixed body: only the brain evolves (your design is preserved exactly).",
                 "Evolving body: the design is the starting point, limbs get mutated."]
        for ln in lines:
            text(surf, f.small, ln, (box.x + 18, y + 4), C_TEXT)
            y += f.small.get_height() + 3
        mp = pygame.mouse.get_pos()
        for r, label in ((b1, "1 · FIXED body — let only the brain evolve"), (b2, "2 · Let the body evolve too"), (b3, "Cancel")):
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
        text(surf, f.big, "SWIM TEST (brainless angle wave)", (10, 8), C_VEL)
        text(surf, f.normal, f"speed {tl['speed']:.2f} m/s · along the body direction {fwd:+.2f} m/s · wave frequency {self.test_freq:.2f} Hz · "
                             f"direction {'base to tip' if self.test_dir > 0 else 'tip to base'}",
             (10, 8 + f.big.get_height() + 4), C_TEXT)
        text(surf, f.small, "T: end test · V: wave direction · Q/E: frequency · speed keys +/-", (10, 8 + f.big.get_height() + f.normal.get_height() + 10), C_DIM)
        if self.msg and time.time() < self.msg_until:
            text(surf, f.normal, self.msg, (10, vh - f.normal.get_height() - 10), (255, 220, 120))

    def draw_panel(self, surf, panel):
        f = self.app.fonts
        pw, ph = surf.get_size()
        pygame.draw.line(surf, C_PANEL_LINE, (0, 0), (0, ph), 2)
        text(surf, f.big, "EDITOR", (12, 8), (190, 140, 255))
        y = 8 + f.big.get_height() + 2
        text(surf, f.small, self.name[:44], (12, y), C_DIM)
        y += f.small.get_height()
        b = self.body
        mass = sum(level_value("mass", int(b.mass[i])) for i in range(b.n + 1))
        text(surf, f.small, f"limbs {b.n}/{MAX_LIMBS} · total mass {mass:.1f} kg · brain: {self.brain_from}"[:60], (12, y), C_DIM)
        buttons, rows, y = self.layout_panel(panel)
        mp = pygame.mouse.get_pos()
        lp = (mp[0] - panel.x, mp[1] - panel.y)
        for r, label, fn in buttons:
            on = r.collidepoint(lp)
            pygame.draw.rect(surf, (44, 58, 84) if on else (28, 38, 56), r, border_radius=6)
            pygame.draw.rect(surf, C_PANEL_LINE, r, 1, border_radius=6)
            text(surf, f.small, label, r.center, C_TEXT, "center")
        if self.test is None:
            title = "Head" if self.sel == 0 else (f"Selected limb {self.sel}" + (("  (old mode: passive spine)" if self.body.legacy_spine else "") if self.sel == 1 else "")) if self.sel > 0 else "Select a point"
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
            for ln in ("Snapping: angle 24 steps (15°), length 8 levels.",
                       "Clicking on empty space adds a limb to the selected point.",
                       "↑/↓ or wheel: value of the selected property;",
                       "←/→ or 1-8: select a property.",
                       "K/L: load a creature from the library (copy)."):
                if y + f.small.get_height() > ph - 4:
                    break
                text(surf, f.small, ln, (12, y), C_DIM)
                y += f.small.get_height() + 1
        else:
            tl = self.test.tele
            lines = [f"speed: {tl['speed']:.2f} m/s", f"t = {tl['t']:.1f} s", "",
                     "This test does not use a brain: an angle", "wave travelling from the body to the tip", "is applied to every arm. If it produces",
                     "thrust, the body can swim in water.", "", "T: end the test and return to editing"]
            for ln in lines:
                text(surf, f.small, ln, (12, y), C_TEXT)
                y += f.small.get_height() + 2

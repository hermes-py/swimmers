"""
Evolving Swimmers - pygame UI.

    python app.py                       # starts training in the background, you watch the creature
    python app.py --load runs/x/best.json
    python app.py --no-train            # only watching/playing/editor (no training)

Modes (cycled with TAB): TRAIN (watch) -> PLAY (control with WASD) -> EDITOR (design a creature)

General keys:
    TAB      change mode             E   open the editor (with a copy of the selected creature)
    SPACE    toggle TURBO (graphics off, training uses all cores)
    P        pause / resume training
    + / -    simulation speed        Mouse wheel: zoom   Z: auto zoom
    F11      fullscreen              H: toggle help       ESC: quit
PLAY mode:
    W A S D  target direction.  WORLD control: W up, S down, A left, D right (including diagonals)
             TANK control: W forward, S backward, A turn left, D turn right (relative to the body)
    M        control scheme: WORLD <-> TANK
    (no key) the creature stops: G toggles between "cut the motors" <-> "give the brain no command (0)"
    K / L    previous / next creature in the library   B live champion   N new random creature
    ENTER    reset the creature      R rescan saved creatures
For the EDITOR keys see editor.py (also shown with H).
The window can be freely resized; the whole layout is computed from the window size.
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import queue
import time
from collections import deque

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from brain import act, stack_brains
from evolution import EvalConfig
from genome import Genome, build_bodies, load_json, random_genome
from physics import (PhysicsParams, center_of_mass, com_velocity, init_state, motor_mask, step)
from trainer import run_training
from ui import *          # colors, Fonts, text, arrow, deg, wrap_pi, draw_creature
from sim import Session, Camera
from editor import Editor
from physics import PHYSICS_VERSION

# --------------------------------------------------------------------------- #
#  Wrapper that manages the training process
# --------------------------------------------------------------------------- #
class TrainerHandle:
    def __init__(self, out, pop, trials, duration, stage_len, seed, resume, workers, seed_genome=None, freeze_body=False,
                 curriculum="adaptive", min_limbs=4, stop_command=False, legacy_spine=False):
        self.cmd_q, self.msg_q = mp.Queue(), mp.Queue()
        self.cpu = os.cpu_count() or 1
        self.normal_workers = max(1, self.cpu - 1)
        self.proc = mp.Process(target=run_training, kwargs=dict(
            out=out, pop=pop, gens=0, trials=trials, duration=duration,
            workers=workers or self.normal_workers, stage_len=stage_len, seed=seed, save_every=25,
            resume=resume, cmd_q=self.cmd_q, msg_q=self.msg_q, verbose=False,
            seed_genome=seed_genome, freeze_body=freeze_body, curriculum=curriculum, min_limbs=min_limbs,
            stop_command=stop_command, legacy_spine=legacy_spine))
        self.proc.start()

    def send(self, *cmd):
        self.cmd_q.put(cmd)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.msg_q.get_nowait())
            except queue.Empty:
                return out

    def alive(self):
        return self.proc.is_alive()

    def shutdown(self, timeout=25.0):
        try:
            self.send("stop")
            self.proc.join(timeout)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(2)


# --------------------------------------------------------------------------- #
#  Application
# --------------------------------------------------------------------------- #
MODE_TRAIN, MODE_PLAY, MODE_EDIT = 0, 1, 2
MODE_NAMES = ["TRAIN", "PLAY", "EDITOR"]


class App:
    def __init__(self, screen, out="runs/app1", train=True, pop=64, trials=4, duration=8.0, stage_len=25,
                 seed=0, resume=False, load=None, workers=0, rng_seed=123, curriculum="adaptive", stop_command=False, legacy_spine=False):
        self.screen = screen
        self.out = out
        self.fonts = Fonts(screen.get_height())
        self.mode = MODE_TRAIN if train else MODE_PLAY
        self.turbo = False
        self.show_help = True
        self.cam = Camera()
        self.time_scale = 1.0
        self.rng = np.random.default_rng(rng_seed)
        self.keys = pygame.key.get_pressed
        self.fullscreen = False
        self.windowed_size = screen.get_size()
        self.running = True

        # training state
        self.trainer = None
        self.paused = False
        self.history = []             # summary dictionaries
        self.live = None              # live champion (Genome)
        self.live_gen = -1
        self.pending = None
        self.stage = 0
        self.trainer_error = None
        self.gen_times = deque(maxlen=20)
        self.info_msg = ""
        self.train_args = dict(pop=pop, trials=trials, duration=duration, stage_len=stage_len, seed=seed, workers=workers,
                               curriculum=curriculum, stop_command=stop_command, legacy_spine=legacy_spine)
        self.dir_range = None
        self.seed_note = ""
        if train:
            self.trainer = TrainerHandle(out, pop, trials, duration, stage_len, seed, resume, workers, curriculum=curriculum,
                                         stop_command=stop_command, legacy_spine=legacy_spine)

        # library (PLAY mode)
        self.library = []             # {name, genome}
        self.extra = []
        if load:
            self.extra.append((os.path.basename(load), load_json(load)))
        self.rescan()
        self.lib_idx = 0
        self.watch = None             # session watched in TRAIN mode
        self.play = None
        self.target = 0.0             # PLAY mode target
        self.have_key = False
        self.scheme = "world"         # world | tank
        self.idle = "motor"           # when no key: motor (cut the motors) | brain (no command = 0)
        self.editor = Editor(self)
        if load:
            self.mode = MODE_PLAY
        self.select_library(0)
        self.editor_inited = False

    # ---------------- library ----------------
    def rescan(self):
        lib = []
        if self.live is not None:
            lib.append(dict(name=f"creature champion (generation {self.live_gen})", genome=self.live))
        for name, g in self.extra:
            lib.append(dict(name=self.tag(name, g), genome=g))
        files = []
        if os.path.isdir(self.out):
            files = ([os.path.join(self.out, "best.json")]
                     + sorted(glob.glob(os.path.join(self.out, "design_*.json")), reverse=True)
                     + sorted(glob.glob(os.path.join(self.out, "gen_*.json")), reverse=True))
        for f in files:
            if os.path.exists(f):
                try:
                    g = load_json(f)
                    lib.append(dict(name=self.tag(os.path.basename(f), g), genome=g))
                except Exception:
                    pass
        if not lib:
            lib.append(dict(name="random #1 (untrained)", genome=self._rand_genome(3, 8)))
        self.library = lib

    @staticmethod
    def tag(name, g):
        """Mark files trained with an old physics version (their behavior may differ in the new physics)."""
        if g.meta.get("physics", 1) != PHYSICS_VERSION and not g.meta.get("designed"):
            return name + " [old physics]"
        return name

    def open_in_play(self, genome, name):
        self.library.append(dict(name=name, genome=genome))
        self.select_library(len(self.library) - 1)
        self.mode = MODE_PLAY

    def select_library(self, idx):
        if not self.library:
            return
        self.lib_idx = idx % len(self.library)
        item = self.library[self.lib_idx]
        self.play = Session(item["genome"], item["name"], heading=HEAD_UP)
        self.target = HEAD_UP

    def _rand_genome(self, lo, hi):
        g = random_genome(self.rng, n_min=lo, n_max=hi)
        g.meta = {"physics": PHYSICS_VERSION}
        return g

    # ---------------- events ----------------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.get_surface()
            if not self.fullscreen:
                self.windowed_size = self.screen.get_size()
        elif e.type == pygame.MOUSEWHEEL and self.mode == MODE_EDIT:
            view, panel, _ = self.layout()
            self.editor.handle_event(e, view, panel)
        elif e.type == pygame.MOUSEWHEEL:
            self.cam.auto = False
            self.cam.scale = float(np.clip(self.cam.scale * (1.12 ** e.y), 8, 400))
        elif self.mode == MODE_EDIT and e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            view, panel, _ = self.layout()
            self.editor.handle_event(e, view, panel)
        elif e.type == pygame.KEYDOWN:
            k = e.key
            if self.mode == MODE_EDIT and self.editor.confirm and k == pygame.K_ESCAPE:
                self.editor.confirm = False
                return
            if self.mode == MODE_EDIT and not self.turbo:
                view, panel, _ = self.layout()
                if k not in (pygame.K_ESCAPE, pygame.K_TAB, pygame.K_SPACE, pygame.K_F11, pygame.K_h, pygame.K_p) \
                        and self.editor.handle_event(e, view, panel):
                    return
            if k == pygame.K_ESCAPE:
                self.running = False
            elif k == pygame.K_TAB:
                self.set_mode((self.mode + 1) % 3)
            elif k == pygame.K_e and self.mode != MODE_EDIT:
                self.set_mode(MODE_EDIT)
            elif k == pygame.K_SPACE:
                self.set_turbo(not self.turbo)
            elif k == pygame.K_p and self.trainer:
                self.paused = not self.paused
                self.trainer.send("pause" if self.paused else "resume")
            elif k == pygame.K_h:
                self.show_help = not self.show_help
            elif k == pygame.K_z:
                self.cam.auto = True
            elif k == pygame.K_r:
                self.rescan()
            elif k == pygame.K_F11:
                self.toggle_fullscreen()
            elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.time_scale = min(8.0, self.time_scale * 1.5)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.time_scale = max(0.125, self.time_scale / 1.5)
            elif k == pygame.K_RETURN:
                if self.mode == MODE_TRAIN:
                    self.swap_watch()
                elif self.play:
                    self.play.reset(HEAD_UP)
                    self.target = HEAD_UP
            elif self.mode == MODE_PLAY:
                if k == pygame.K_l:
                    self.select_library(self.lib_idx + 1)
                elif k == pygame.K_k:
                    self.select_library(self.lib_idx - 1)
                elif k == pygame.K_m:
                    self.scheme = "tank" if self.scheme == "world" else "world"
                elif k == pygame.K_g:
                    self.idle = "brain" if self.idle == "motor" else "motor"
                elif k == pygame.K_b:
                    self.rescan()
                    self.select_library(0)
                elif k == pygame.K_n:
                    g = self._rand_genome(3, 10)
                    self.library.append(dict(name=f"random #{len(self.library)} (untrained)", genome=g))
                    self.select_library(len(self.library) - 1)

    def start_training_from(self, genome, freeze_body, name=""):
        """From the editor: save and stop the current training, start a new training run from the design."""
        if self.trainer:
            self.trainer.shutdown()
        base = os.path.dirname(os.path.normpath(self.out)) or "."
        k = 1
        while os.path.exists(os.path.join(base, f"design_{k:03d}")):
            k += 1
        self.out = os.path.join(base, f"design_{k:03d}")
        os.makedirs(self.out, exist_ok=True)
        from genome import save_json
        save_json(genome, os.path.join(self.out, "design.json"))
        self.history, self.live, self.live_gen, self.pending, self.watch = [], None, -1, None, None
        self.stage, self.dir_range, self.trainer_error, self.paused = 0, None, None, False
        self.gen_times.clear()
        a_ = self.train_args
        self.trainer = TrainerHandle(self.out, a_["pop"], a_["trials"], a_["duration"], a_["stage_len"], a_["seed"],
                                     False, a_["workers"], seed_genome=genome, freeze_body=freeze_body,
                                     curriculum=a_["curriculum"], stop_command=a_["stop_command"], legacy_spine=a_["legacy_spine"])
        self.seed_note = f"training from a design ({genome.body.n} limbs, body {'FIXED' if freeze_body else 'evolving'}) → {self.out}"
        self.info_msg = ""
        self.rescan()
        self.mode = MODE_TRAIN

    def set_mode(self, mode):
        self.mode = mode
        if mode == MODE_EDIT and not self.editor_inited:
            self.editor_inited = True
            if self.play is not None:
                self.editor.open_from(self.play.genome, self.play.label)
            else:
                self.editor.new_blank()

    def set_turbo(self, on):
        self.turbo = on
        if self.trainer:
            self.trainer.send("workers", self.trainer.cpu if on else self.trainer.normal_workers)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)

    # ---------------- messages from training ----------------
    def swap_watch(self):
        if self.pending is not None:
            g, gen = self.pending
            self.watch = Session(g, f"champion · generation {gen}", heading=float(self.rng.uniform(-math.pi, math.pi)))
            self.pending = None

    def poll_trainer(self):
        if not self.trainer:
            return
        for msg in self.trainer.poll():
            kind = msg[0]
            if kind == "gen":
                s, champ = msg[1], Genome.from_dict(msg[2])
                self.history.append(s)
                self.stage = s["stage"]
                self.dir_range = s.get("dir_range_deg")
                self.live, self.live_gen = champ, s["gen"]
                self.pending = (champ, s["gen"])
                self.gen_times.append((time.time(), s["gen"]))
                if self.library and self.library[0]["name"].startswith("creature"):
                    self.library[0] = dict(name=f"creature champion (generation {s['gen']})", genome=champ)
                elif self.live is not None:
                    self.library.insert(0, dict(name=f"creature champion (generation {s['gen']})", genome=champ))
                    self.lib_idx += 1
            elif kind == "info":
                self.info_msg = msg[1]
            elif kind == "error":
                self.trainer_error = msg[1]

    def gens_per_min(self):
        if len(self.gen_times) < 2:
            return 0.0
        (t0, g0), (t1, g1) = self.gen_times[0], self.gen_times[-1]
        return 60.0 * (g1 - g0) / max(t1 - t0, 1e-6)

    # ---------------- input ----------------
    def move_vector(self):
        k = self.keys()
        dx = int(bool(k[pygame.K_d])) - int(bool(k[pygame.K_a]))
        dy = int(bool(k[pygame.K_w])) - int(bool(k[pygame.K_s]))
        return dx, dy

    def auto_target(self, sess: Session):
        """Command sequence shown to the champion watched in TRAIN mode: (angle, magnitude)."""
        if self.stage >= 3:
            seq = [(i * math.pi / 4, 1.0) for i in range(8)]
            seq.insert(3, (0.0, 0.0))
            seq.insert(7, (0.0, 0.0))
        elif self.stage == 2:
            seq = [(i * math.pi / 4, 1.0) for i in range(8)]
        elif self.stage == 1 and self.dir_range:
            r = math.radians(self.dir_range)
            seq = [(0.0, 1.0), (r, 1.0), (-r, 1.0)]
        elif self.stage == 1:
            seq = [(0.0, 1.0), (math.pi / 2, 1.0), (math.pi, 1.0), (-math.pi / 2, 1.0)]
        else:
            seq = [(0.0, 1.0)]
        a, m = seq[int(sess.t / 5.0) % len(seq)]
        return sess.h0 + a, m

    # ---------------- update ----------------
    def update(self, dt):
        self.poll_trainer()
        if self.trainer and not self.trainer.alive() and not self.trainer_error and self.running:
            self.trainer_error = "the training process closed unexpectedly"
        if self.turbo:
            return
        dt = min(dt, 0.1) * self.time_scale
        view_wh = self.view_size()
        if self.mode == MODE_TRAIN:
            if self.watch is None:
                self.swap_watch()
            elif self.pending is not None and self.watch.t > 20.0:
                self.swap_watch()
            if self.watch is not None:
                ang, mag = self.auto_target(self.watch)
                self.watch.advance(dt, ang, mag, True)
                self.cam.update(self.watch, view_wh, dt)
        elif self.mode == MODE_EDIT:
            self.editor.update(dt, view_wh)
        else:
            self.update_play(dt, view_wh)

    def update_play(self, dt, view_wh):
        if not self.play:
            return
        dx, dy = self.move_vector()
        self.have_key = (dx != 0 or dy != 0)
        mag, motors = 1.0, True
        if self.have_key:
            if self.scheme == "world":
                self.target = math.atan2(dy, dx)
            else:        # tank: target is in the BODY frame (W forward, S backward, A left, D right)
                self.target = self.play.tele["heading"] + math.atan2(-dx, dy)
        elif self.idle == "motor":
            motors = False
        else:
            mag = 0.0
        self.play.advance(dt, self.target, mag, motors)
        self.cam.update(self.play, view_wh, dt)

    # ---------------- layout ----------------
    def layout(self):
        w, h = self.screen.get_size()
        bar_h = int(self.fonts.small.get_height() * 1.9)
        panel_w = int(np.clip(w * 0.30, 280, 520))
        view = pygame.Rect(0, 0, w - panel_w, h - bar_h)
        panel = pygame.Rect(w - panel_w, 0, panel_w, h - bar_h)
        bar = pygame.Rect(0, h - bar_h, w, bar_h)
        return view, panel, bar

    def view_size(self):
        v, _, _ = self.layout()
        return v.size

    # ---------------- drawing ----------------
    def draw(self):
        w, h = self.screen.get_size()
        if self.fonts.h != h:
            self.fonts = Fonts(h)
        scr = self.screen
        scr.fill(C_PANEL)
        if self.turbo:
            self.draw_turbo()
            pygame.display.flip()
            return
        view, panel, bar = self.layout()
        vsurf = scr.subsurface(view)
        if self.mode == MODE_EDIT:
            self.editor.draw(vsurf, scr.subsurface(panel), view, panel)
        else:
            sess = self.watch if self.mode == MODE_TRAIN else self.play
            self.draw_world(vsurf, sess)
            if self.mode == MODE_TRAIN:
                self.draw_train_hud(vsurf)
            else:
                self.draw_play_hud(vsurf)
            self.draw_panel(scr.subsurface(panel), sess)
        self.draw_bar(scr.subsurface(bar))
        if self.trainer_error:
            self.draw_error(vsurf)
        pygame.display.flip()

    # --- world ---
    def draw_world(self, surf, sess, cam=None):
        vw, vh = surf.get_size()
        # water gradient
        grad = getattr(self, "_grad", None)
        if grad is None or grad.get_size() != (vw, vh):
            grad = pygame.Surface((1, vh))
            for y in range(vh):
                t = y / max(vh - 1, 1)
                grad.set_at((0, y), tuple(int(C_BG_TOP[i] * (1 - t) + C_BG_BOT[i] * t) for i in range(3)))
            self._grad = grad = pygame.transform.scale(grad, (vw, vh))
        surf.blit(grad, (0, 0))
        if sess is None:
            text(surf, self.fonts.big, "Waiting for the first champion from training…", (vw // 2, vh // 2), C_DIM, "center")
            return
        cam = cam or self.cam
        sc = cam.scale
        # grid (1 m and 5 m)
        x0, x1 = cam.c[0] - vw / 2 / sc, cam.c[0] + vw / 2 / sc
        y0, y1 = cam.c[1] - vh / 2 / sc, cam.c[1] + vh / 2 / sc
        for step_m, col in ((1, C_GRID_MINOR), (5, C_GRID_MAJOR)):
            if step_m == 1 and sc < 14:
                continue
            if step_m == 5 and sc < 3:
                continue
            for gx in range(int(math.floor(x0 / step_m)), int(math.ceil(x1 / step_m)) + 1):
                px = vw / 2 + (gx * step_m - cam.c[0]) * sc
                if step_m == 1 and gx % 5 == 0:
                    continue
                pygame.draw.line(surf, col, (px, 0), (px, vh))
            for gy in range(int(math.floor(y0 / step_m)), int(math.ceil(y1 / step_m)) + 1):
                py = vh / 2 - (gy * step_m - cam.c[1]) * sc
                if step_m == 1 and gy % 5 == 0:
                    continue
                pygame.draw.line(surf, col, (0, py), (vw, py))
        wh = (vw, vh)
        # trail
        if len(sess.trail) > 2:
            pts = [cam.w2s(p, wh) for p in sess.trail]
            pygame.draw.lines(surf, (60, 120, 160), False, pts, 2)
        draw_creature(surf, sess, cam, wh)
        # vectors (on the COM)
        tl = sess.tele
        p = cam.w2s(tl["com"], wh)
        R = max(40, min(vw, vh) * 0.12)
        if tl.get("active", True):
            arrow(surf, C_TARGET, p, (p[0] + math.cos(tl["target"]) * R, p[1] - math.sin(tl["target"]) * R), 3)
        arrow(surf, C_HEAD, p, (p[0] + math.cos(tl["heading"]) * R * 0.75, p[1] - math.sin(tl["heading"]) * R * 0.75), 3)
        v = tl["vel"]
        vl = min(R * 1.5, float(np.linalg.norm(v)) * sc)
        if vl > 3:
            n = v / max(np.linalg.norm(v), 1e-9)
            arrow(surf, C_VEL, p, (p[0] + n[0] * vl, p[1] - n[1] * vl), 3)

    # --- HUDs ---
    def draw_graph(self, surf, rect, big=False):
        f = self.fonts
        bg = pygame.Surface(rect.size, pygame.SRCALPHA)
        bg.fill((8, 14, 22, 190))
        surf.blit(bg, rect.topleft)
        pygame.draw.rect(surf, C_PANEL_LINE, rect, 1)
        H = self.history
        text(surf, f.small, "Best fitness / mean", (rect.x + 8, rect.y + 4), C_DIM)
        if len(H) < 2:
            text(surf, f.small, "waiting for data…", rect.center, C_DIM, "center")
            return
        inner = pygame.Rect(rect.x + 44, rect.y + f.small.get_height() + 10, rect.w - 56, rect.h - f.small.get_height() - 24)
        best = np.array([s["best"] for s in H])
        mean = np.array([s["mean"] for s in H])
        stages = [s["stage"] for s in H]
        lo, hi = min(best.min(), mean.min(), 0.0), max(best.max(), mean.max(), 0.05)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        gens = np.array([s["gen"] for s in H])
        gx = lambda g: inner.x + (g - gens[0]) / max(gens[-1] - gens[0], 1) * inner.w
        gy = lambda v: inner.bottom - (v - lo) / (hi - lo) * inner.h
        pygame.draw.line(surf, C_PANEL_LINE, (inner.x, gy(0)), (inner.right, gy(0)))
        # stage transitions
        for k in range(1, len(H)):
            if stages[k] != stages[k - 1]:
                pygame.draw.line(surf, (120, 90, 160), (gx(gens[k]), inner.y), (gx(gens[k]), inner.bottom))
                text(surf, f.small, f"stage {stages[k]}", (gx(gens[k]) + 3, inner.y), (170, 140, 210))
        step = max(1, len(H) // max(inner.w, 1))
        idx = list(range(0, len(H), step))
        if idx[-1] != len(H) - 1:
            idx.append(len(H) - 1)
        for series, col in ((mean, C_MEAN), (best, C_BEST)):
            pts = [(gx(gens[i]), gy(series[i])) for i in idx]
            if len(pts) > 1:
                pygame.draw.lines(surf, col, False, pts, 2)
        text(surf, f.small, f"{hi:.2f}", (rect.x + 4, inner.y), C_DIM)
        text(surf, f.small, f"{lo:.2f}", (rect.x + 4, inner.bottom - f.small.get_height()), C_DIM)
        text(surf, f.small, f"generation {gens[0]}…{gens[-1]}", (inner.right, rect.bottom - f.small.get_height() - 2), C_DIM, "topright")

    def draw_train_hud(self, surf):
        f = self.fonts
        vw, vh = surf.get_size()
        H = self.history
        lines = []
        if H:
            s = H[-1]
            diff = f"difficulty {s.get('notch', s['stage'])}/{s.get('notch_max', 10)}" + (f" · direction range ±{s['dir_range_deg']:.0f}°" if s.get("dir_range_deg") else "")
            lines = [f"Generation {s['gen']}   ·   {diff}",
                     f"best {s['best']:+.3f}   mean {s['mean']:+.3f}",
                     f"best limb count {s['best_limbs']}   mean {s['mean_limbs']:.1f}",
                     f"{self.gens_per_min():.1f} generations/min" + ("   · PAUSED" if self.paused else "")]
            if s.get("mut_scale", 1.0) > 1:
                lines.append(f"stall: {s['stall']} generations without improvement → mutation ×{s['mut_scale']:.1f}")
            val = [h_ for h_ in H if h_.get("validity") is not None]
            if val:
                v = val[-1]["validity"]
                lines.append(f"physics validation (generation {val[-1]['gen']}): {v:.2f}" + ("" if v >= 0.7 else "   SUSPICIOUS: the speed may come from numerical error"))
            if self.seed_note:
                lines.append(self.seed_note[:110])
            if self.info_msg:
                lines.append(self.info_msg[:110])
        else:
            lines = (["Training is starting… (the first generation takes a few seconds)"] + ([self.seed_note[:110]] if self.seed_note else [])) if self.trainer else ["Training is off (--no-train)"]
        y = 8
        for ln in lines:
            text(surf, f.normal, ln, (10, y), C_TEXT)
            y += f.normal.get_height() + 2
        if self.pending is not None and self.watch is not None:
            text(surf, f.small, f"new champion ready (generation {self.pending[1]}) — switch with ENTER, automatic switch soon",
                 (10, y + 4), C_DIM)
        gw, gh = int(vw * 0.46), int(vh * 0.28)
        self.draw_graph(surf, pygame.Rect(10, vh - gh - 10, gw, gh))
        text(surf, f.small, f"speed ×{self.time_scale:g}   zoom {self.cam.scale:.0f} px/m" + ("" if self.cam.auto else " (manual)"),
             (vw - 10, 8), C_DIM, "topright")

    def draw_play_hud(self, surf):
        f = self.fonts
        vw, vh = surf.get_size()
        sess = self.play
        text(surf, f.normal, f"{sess.label}   ·   {sess.genome.body.n} limbs", (10, 8), C_TEXT)
        y = 8 + f.normal.get_height() + 4
        if self.scheme == "world":
            keys = "W↑ S↓ A← D→ (world directions)"
        else:
            keys = "W forward · S backward · A turn left · D turn right (relative to the body)"
        text(surf, f.small, f"control: {self.scheme.upper()} (M)   ·   {keys}", (10, y), C_DIM)
        y += f.small.get_height() + 2
        if self.have_key:
            st_ = "command: active"
        elif self.idle == "motor":
            st_ = "IDLE: motors off, the creature coasts to a stop   (G: give the brain 'no command')"
        else:
            st_ = "IDLE: the brain is given 'no command' (0) — it stops if the creature learned to stop in training   (G: cut the motors)"
        text(surf, f.small, st_, (10, y), C_VEL if not self.have_key else C_TARGET)
        y += f.small.get_height() + 2
        if "[old physics]" in sess.label:
            text(surf, f.small, "this creature was trained with an old physics version: its behavior may differ (check with python validate.py)", (10, y), (255, 150, 120))
        text(surf, f.small, f"speed ×{self.time_scale:g}   zoom {self.cam.scale:.0f} px/m" + ("" if self.cam.auto else " (manual)"),
             (vw - 10, 8), C_DIM, "topright")
        # WASD indicator
        ks = self.keys()
        size = max(22, int(vh * 0.05))
        base = (vw - 3 * size - 24, vh - 2 * size - 24)
        for ch, key, (cx, cy) in (("W", pygame.K_w, (1, 0)), ("A", pygame.K_a, (0, 1)),
                                   ("S", pygame.K_s, (1, 1)), ("D", pygame.K_d, (2, 1))):
            r = pygame.Rect(base[0] + cx * (size + 3), base[1] + cy * (size + 3), size, size)
            on = bool(ks[key])
            pygame.draw.rect(surf, C_TARGET if on else (24, 40, 58), r, border_radius=5)
            pygame.draw.rect(surf, C_PANEL_LINE, r, 1, border_radius=5)
            text(surf, f.normal, ch, r.center, (10, 20, 30) if on else C_DIM, "center")

    def draw_error(self, surf):
        r = surf.get_rect().inflate(-40, -40)
        pygame.draw.rect(surf, (60, 16, 16), r, border_radius=8)
        pygame.draw.rect(surf, (220, 80, 80), r, 2, border_radius=8)
        msg = (self.trainer_error or "")[-900:].splitlines()[-12:]
        y = r.y + 10
        text(surf, self.fonts.normal, "Training error:", (r.x + 12, y), (255, 160, 160))
        y += self.fonts.normal.get_height() + 4
        for ln in msg:
            text(surf, self.fonts.small, ln[:140], (r.x + 12, y), C_TEXT)
            y += self.fonts.small.get_height() + 1

    # --- right panel (telemetry) ---
    def draw_panel(self, surf, sess):
        f = self.fonts
        pw, ph = surf.get_size()
        pygame.draw.line(surf, C_PANEL_LINE, (0, 0), (0, ph), 2)
        y = 8
        text(surf, f.big, f"{MODE_NAMES[self.mode]}", (12, y), C_BEST if self.mode == MODE_TRAIN else C_TARGET)
        if sess is None:
            return
        y += f.big.get_height() + 2
        text(surf, f.small, sess.label, (12, y), C_DIM)
        y += f.small.get_height() + 8

        tl = sess.tele
        # compass
        R = int(min(pw * 0.2, ph * 0.13))
        cc = (12 + R + 4, y + R + 4)
        pygame.draw.circle(surf, (20, 30, 44), cc, R)
        pygame.draw.circle(surf, C_PANEL_LINE, cc, R, 2)
        pygame.draw.line(surf, C_PANEL_LINE, (cc[0] - R, cc[1]), (cc[0] + R, cc[1]))
        pygame.draw.line(surf, C_PANEL_LINE, (cc[0], cc[1] - R), (cc[0], cc[1] + R))
        a2p = lambda a, L: (cc[0] + math.cos(a) * L, cc[1] - math.sin(a) * L)
        active = tl.get("active", True)
        if active:
            arrow(surf, C_TARGET, cc, a2p(tl["target"], R * 0.92), 3)
        arrow(surf, C_HEAD, cc, a2p(tl["heading"], R * 0.70), 3)
        if tl["speed"] > 0.02:
            va = math.atan2(tl["vel"][1], tl["vel"][0])
            arrow(surf, C_VEL, cc, a2p(va, R * min(0.95, 0.15 + tl["speed"] / 1.2)), 3)

        tx = cc[0] + R + 16
        err = math.degrees(wrap_pi(tl["target"] - tl["heading"]))
        rows = [("Target direction", f"{deg(tl['target']):5.0f}°" if active else "  —  (none)", C_TARGET),
                ("Body direction", f"{deg(tl['heading']):5.0f}°", C_HEAD),
                ("Direction error", f"{err:+5.0f}°" if active else "  —", C_TEXT),
                ("Speed", f"{tl['speed']:5.2f} m/s", C_VEL),
                ("Progress to target", f"{tl['progress']:+5.2f} m/s", C_TEXT),
                ("Lateral deviation", f"{tl['lateral']:+5.2f} m/s", C_TEXT)]
        ry = y
        rh = f.small.get_height() + 3
        for k, v, col in rows:
            text(surf, f.small, k, (tx, ry), C_DIM)
            text(surf, f.small, v, (pw - 12, ry), col, "topright")
            ry += rh
        y = max(cc[1] + R + 10, ry + 6)

        # energy
        text(surf, f.small, "Energy", (12, y), C_DIM)
        bw = pw - 24
        y += f.small.get_height() + 2
        pygame.draw.rect(surf, (24, 34, 48), (12, y, bw, 8), border_radius=3)
        pygame.draw.rect(surf, C_POS, (12, y, int(bw * min(1.0, tl["energy"])), 8), border_radius=3)
        y += 11
        text(surf, f.small, f"mean command² {tl['energy']:.2f}   power {tl['power']:.1f}   total {tl['total']:.0f}   t={tl['t']:.1f}s",
             (12, y), C_TEXT)
        y += f.small.get_height() + 8

        # progress graph (last ~12 s)
        gh = int(ph * 0.09)
        gr = pygame.Rect(12, y, pw - 24, gh)
        pygame.draw.rect(surf, (20, 28, 40), gr)
        pygame.draw.rect(surf, C_PANEL_LINE, gr, 1)
        H = list(sess.progress_hist)
        if len(H) > 2:
            vmax = max(0.5, max(abs(v) for v in H))
            zero = gr.centery
            pygame.draw.line(surf, C_PANEL_LINE, (gr.x, zero), (gr.right, zero))
            pts = [(gr.x + i / (sess.progress_hist.maxlen - 1) * gr.w, zero - v / vmax * (gr.h / 2 - 2)) for i, v in enumerate(H)]
            pygame.draw.lines(surf, C_VEL, False, pts, 2)
        text(surf, f.small, "progress to target (last 12 s)", (gr.x + 4, gr.y + 1), C_DIM)
        y = gr.bottom + 8

        # joints
        text(surf, f.small, "Joints: angle (relative to limits) · command", (12, y), C_DIM)
        y += f.small.get_height() + 3
        self.draw_joints(surf, sess, pygame.Rect(12, y, pw - 24, ph - y - 6))

    def draw_joints(self, surf, sess, rect):
        bd, st = sess.bd, sess.st
        idx = [i for i in range(1, bd.N) if sess.mm[i]]
        if not idx or rect.h < 20:
            return
        rh = int(np.clip(rect.h / len(idx), 4, 16))
        label_w = self.fonts.small.size("00")[0] + 4 if rh >= 10 else 0
        ax, aw = rect.x + label_w, int((rect.w - label_w) * 0.62)
        cx0, cw = ax + aw + 8, rect.w - label_w - aw - 8
        y = rect.y
        for i in idx:
            if y + rh > rect.bottom + 1:
                break
            th = float(st.theta[0, i])
            lo, hi = float(bd.jlo[0, i]), float(bd.jhi[0, i])
            if label_w:
                text(surf, self.fonts.small, f"{i}", (rect.x, y - 1), C_DIM)
            bar = pygame.Rect(ax, y + 1, aw, rh - 2)
            pygame.draw.rect(surf, (24, 34, 48), bar)
            zx = bar.x + (0 - lo) / max(hi - lo, 1e-6) * bar.w
            pygame.draw.line(surf, C_PANEL_LINE, (zx, bar.y), (zx, bar.bottom))
            mx = bar.x + float(np.clip((th - lo) / max(hi - lo, 1e-6), 0, 1)) * bar.w
            atlimit = th <= lo + 0.02 or th >= hi - 0.02
            pygame.draw.rect(surf, (255, 90, 90) if atlimit else C_HEAD, (mx - 2, bar.y, 4, bar.h))
            c = float(np.clip(sess.ctrl[0, i], -1, 1))
            cb = pygame.Rect(cx0, y + 1, cw, rh - 2)
            pygame.draw.rect(surf, (24, 34, 48), cb)
            mid = cb.centerx
            wpx = int(c * cb.w / 2)
            pygame.draw.rect(surf, C_POS if c >= 0 else C_NEG, (min(mid, mid + wpx), cb.y, abs(wpx), cb.h))
            pygame.draw.line(surf, C_PANEL_LINE, (mid, cb.y), (mid, cb.bottom))
            y += rh

    # --- bottom bar ---
    def draw_bar(self, surf):
        f = self.fonts
        surf.fill((10, 14, 20))
        w, h = surf.get_size()
        if self.show_help and self.mode == MODE_EDIT:
            s = ("TAB mode · Y symmetry · M mirror · T test · N new · K/L load · ↑/↓ value · 1-8 property · DEL delete · "
                 "Ctrl+Z/Y undo/redo · Ctrl+S save · ENTER open in PLAY · H help")
        elif self.show_help:
            s = ("TAB mode · E editor · SPACE turbo · P pause · WASD direction · M world/tank · G idle · K/L creature · B champion · "
                 "N random · ENTER reset · +/- speed · F11 · H help")
        else:
            s = "H: help"
        text(surf, f.small, s, (8, h // 2), C_DIM, "midleft")
        status = []
        if self.trainer:
            status.append("training: " + ("paused" if self.paused else "running" if self.trainer.alive() else "stopped"))
        text(surf, f.small, "  ".join(status), (w - 8, h // 2), C_BEST, "midright")

    # --- turbo ---
    def draw_turbo(self):
        f = self.fonts
        w, h = self.screen.get_size()
        scr = self.screen
        text(scr, f.huge, "TURBO", (w // 2, int(h * 0.06)), C_POS, "midtop")
        text(scr, f.normal, "Graphics/simulation off — training is using all cores.  SPACE: go back",
             (w // 2, int(h * 0.06) + f.huge.get_height() + 4), C_DIM, "midtop")
        H = self.history
        if H:
            s = H[-1]
            line = (f"Generation {s['gen']}   ·   stage {s['stage']}   ·   best {s['best']:+.3f}   ·   mean {s['mean']:+.3f}"
                    f"   ·   {self.gens_per_min():.1f} generations/min")
        else:
            line = "waiting for the first generation…"
        text(scr, f.big, line, (w // 2, int(h * 0.26)), C_TEXT, "midtop")
        self.draw_graph(scr, pygame.Rect(int(w * 0.07), int(h * 0.36), int(w * 0.86), int(h * 0.52)))
        if self.trainer_error:
            self.draw_error(scr)

    # ---------------- shutdown ----------------
    def close(self):
        if self.trainer:
            self.trainer.shutdown()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/app1", help="save folder (best.json, gen_*.json, log.csv, checkpoint)")
    ap.add_argument("--no-train", action="store_true", help="do not start training")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--stage-len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = number of cores - 1")
    ap.add_argument("--load", default=None, help="open a creature JSON (starts in PLAY mode)")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--curriculum", choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--stop-command", action="store_true", help="enable the 'no command = stop' notch (default OFF)")
    ap.add_argument("--legacy-spine", action="store_true", help="OLD spine mode: bone 1 is passive and determines the body direction (default: body direction = head direction)")
    a = ap.parse_args()

    pygame.display.init()
    pygame.font.init()
    w, h = (int(v) for v in a.size.lower().split("x"))
    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    pygame.display.set_caption("Evolving Swimmers")
    app = App(screen, out=a.out, train=not a.no_train, pop=a.pop, trials=a.trials, duration=a.duration,
              stage_len=a.stage_len, seed=a.seed, resume=a.resume, load=a.load, workers=a.workers,
              curriculum=a.curriculum, stop_command=a.stop_command, legacy_spine=a.legacy_spine)
    clock = pygame.time.Clock()
    try:
        while app.running:
            for e in pygame.event.get():
                app.handle_event(e)
            dt = clock.tick(10 if app.turbo else 60) / 1000.0
            app.update(dt)
            app.draw()
    finally:
        pygame.display.set_caption("Shutting down… (saving training)")
        app.close()
        pygame.quit()


if __name__ == "__main__":
    mp.freeze_support()
    main()

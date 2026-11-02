"""Ecosystem UI tests (headless):  python test_eco.py  -> /tmp/shots/eco_*.png"""
import os, shutil, tempfile
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import numpy as np
import pygame
from eco import EcoApp, write_log
from world import WorldConfig
from genome import load_json

SHOTS = os.environ.get("SHOTS", "/tmp/shots")
os.makedirs(SHOTS, exist_ok=True)


def frames(app, n, dt=1 / 60):
    for _ in range(n):
        app.update(dt)
        app.draw()


def test_viewer():
    out = tempfile.mkdtemp()
    pygame.display.init(); pygame.font.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    app = EcoApp(screen, WorldConfig(n_init=40, seed=1, source_rate=1.0), out)
    frames(app, 120)
    pygame.image.save(app.screen, os.path.join(SHOTS, "eco_1280x720.png"))
    w = app.world
    # select a creature (click at its head's screen position)
    view, _, _ = app.layout()
    p = app.cam.w2s(w.st.x[3, 0], view.size)
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(p[0]), int(p[1])), button=1))
    assert app.sel == int(w.ids[3]), (app.sel, w.ids[3])
    frames(app, 30)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0)); assert app.follow
    frames(app, 30)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0))
    files = [f for f in os.listdir(out) if f.startswith("eco_")]
    assert files and load_json(os.path.join(out, files[0])).body.n >= 1
    pygame.image.save(app.screen, os.path.join(SHOTS, "eco_selected.png"))
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)); frames(app, 5)
    assert app.turbo
    pygame.image.save(app.screen, os.path.join(SHOTS, "eco_turbo.png"))
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0))
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=0))
    for size in ((800, 500), (1920, 1080)):
        app.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, size=size, w=size[0], h=size[1]))
        frames(app, 3)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0)); frames(app, 3)
    # soup depletion rendering
    app.world.energy[:] = -1; app.world._deaths(); frames(app, 3)
    assert app.world.pop == 0
    write_log(app.world, out)
    print("[1] ecosystem UI: drawing, selection, follow, saving, turbo, resizing, extinction screen all error-free")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_viewer()
    print("DONE ->", SHOTS)

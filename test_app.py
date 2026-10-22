"""Arayüz testleri (başsız/dummy sürücü):  python test_app.py [canlı.json]   -> /tmp/shots/*.png"""
import os, sys, time, math, shutil, tempfile
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import multiprocessing as mp
import numpy as np
import pygame
from app import App, MODE_PLAY, MODE_TRAIN, MODE_EDIT
from genome import GRID, level_value

SHOTS = os.environ.get("SHOTS", "/tmp/shots")
os.makedirs(SHOTS, exist_ok=True)


class FakeKeys:
    def __init__(self, down=()):
        self.down = set(down)
    def __getitem__(self, k):
        return k in self.down


def frames(app, n, dt=1 / 60, keys=()):
    app.keys = lambda: FakeKeys(keys)
    for _ in range(n):
        app.update(dt)
        app.draw()


def shot(app, name):
    pygame.image.save(app.screen, os.path.join(SHOTS, name))


def key(app, k, mod=0):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod))


def make_app(size, **kw):
    pygame.display.init(); pygame.font.init()
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    return App(screen, **kw)


def speed(app):
    return app.play.tele["speed"]


def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def motor_mask_ok(ed):
    from physics import motor_mask
    return bool(motor_mask(ed.sess().bd)[0, 1])


def test_play_controls(load):
    out = tempfile.mkdtemp()
    app = make_app((1280, 720), train=False, load=load, out=out)
    assert app.mode == MODE_PLAY
    assert abs(app.play.tele["heading"] - math.pi / 2) < 1e-6, "başlangıçta kafa yukarı bakmalı (90°)"
    # --- dünya kontrolü ---
    frames(app, 120, keys=[pygame.K_w, pygame.K_d])
    assert abs(math.degrees(app.play.tele["target"]) - 45) < 1e-6
    shot(app, "play_1280x720.png")
    print(f"[1] DÜNYA: W+D -> hedef 45°, hız {speed(app):.2f} m/s")
    # --- K / L ---
    app.rescan()
    app.select_library(0)
    n0 = app.lib_idx
    key(app, pygame.K_l)
    assert app.lib_idx == (n0 + 1) % len(app.library)
    key(app, pygame.K_k)
    assert app.lib_idx == n0
    print(f"[1] K/L kütüphane gezinmesi OK ({len(app.library)} canlı)")
    # --- tank kontrolü ---
    key(app, pygame.K_m)
    assert app.scheme == "tank"
    frames(app, 30, keys=[pygame.K_w])
    assert abs(angdiff(app.target, app.play.tele["heading"])) < 0.2          # gövde yalpalar: bir kare gecikmeli
    frames(app, 30, keys=[pygame.K_a])
    assert abs(angdiff(app.target, app.play.tele["heading"] + math.pi / 2)) < 0.2     # A: sola (+90°)
    frames(app, 30, keys=[pygame.K_s])
    assert abs(angdiff(app.target, app.play.tele["heading"] + math.pi)) < 0.2         # S: geri
    frames(app, 30, keys=[pygame.K_w, pygame.K_d])
    assert abs(angdiff(app.target, app.play.tele["heading"] - math.pi / 4)) < 0.2     # W+D: -45°
    shot(app, "play_tank.png")
    print("[1] TANK: W=ön, S=arka, A=+90° (sol), W+D=-45° (gövdeye göre) OK")
    # --- tuş yokken dur ---
    key(app, pygame.K_m)
    assert app.scheme == "dünya"
    frames(app, 600, keys=[pygame.K_d])
    v0 = speed(app)
    frames(app, 1200, keys=[])
    v1 = speed(app)
    assert not app.play.tele["active"] and np.all(app.play.ctrl == 0)
    print(f"[1] BOŞTA (motor kes): hız {v0:.2f} -> {v1:.3f} m/s (20 sn sonra), motor komutları sıfır")
    assert v1 < max(0.5 * v0, 0.03), (v0, v1)        # süzülme: 20 sn sonra hız en az yarıya inmeli
    key(app, pygame.K_g)
    assert app.idle == "beyin"
    frames(app, 30, keys=[])
    assert app.play.mag == 0.0 and not app.play.tele["active"]
    shot(app, "play_idle_brain.png")
    print("[1] BOŞTA (beyin): beyne komut yok (0) veriliyor; hedef oku gizli")
    for size in ((800, 500), (1920, 1080), (640, 400)):
        app.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, size=size, w=size[0], h=size[1]))
        frames(app, 5, keys=[pygame.K_w])
    print("[1] yeniden boyutlandırma hatasız")
    app.close()
    shutil.rmtree(out, ignore_errors=True)


def click(app, pos, button=1):
    app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button))


def drag(app, p0, p1):
    app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=p0, rel=(0, 0), buttons=(0, 0, 0)))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=p0, button=1))
    mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
    for p in (mid, p1):
        app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=p, rel=(1, 1), buttons=(1, 0, 0)))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=p1, button=1))


def w2s(app, w):
    view, _, _ = app.layout()
    p = app.editor.cam.w2s(np.array(w), view.size)
    return (int(p[0]), int(p[1]))


def sp(app, i):
    return w2s(app, app.editor.pos()[i])


def test_editor(load):
    out = tempfile.mkdtemp()
    app = make_app((1280, 720), train=False, load=load, out=out)
    key(app, pygame.K_TAB)                                  # DENE -> EDİTÖR
    assert app.mode == MODE_EDIT
    ed = app.editor
    frames(app, 3)
    shot(app, "editor_open.png")
    print(f"[2] editör açıldı: {ed.name}, {ed.body.n} uzuv")

    key(app, pygame.K_n); frames(app, 2)                    # boş canlı: kafa + omurga + kuyruk
    n1 = ed.body.n
    assert n1 == 2
    click(app, sp(app, n1)); assert ed.sel == n1
    tip = ed.pos()[n1]
    click(app, w2s(app, tip + np.array([0.52, 0.0])))          # kuyruk ucunun sağına
    assert ed.body.n == n1 + 1 and ed.sel == n1 + 1, (ed.body.n, ed.sel)
    ed.body.validate()
    new = ed.sel
    L = level_value("length", int(ed.body.length[new]))
    assert abs(L - 0.5) < 0.06, L
    assert abs(np.hypot(*(ed.pos()[new] - ed.pos()[n1])) - L) < 1e-6
    print(f"[2] boşa tık -> uzuv eklendi (uzunluk {L:.2f} m, açı {math.degrees(level_value('theta', int(ed.body.theta[new]))):.0f}°)")

    before = (int(ed.body.theta[new]), int(ed.body.length[new]))
    dst = ed.pos()[n1] + np.array([0.82, 0.0])
    drag(app, sp(app, new), w2s(app, dst))
    after = (int(ed.body.theta[new]), int(ed.body.length[new]))
    assert after != before
    assert level_value("length", after[1]) in (0.7, 0.85), after
    print(f"[2] sürükle: (açı,uzunluk) kademesi {before} -> {after} (gride yapıştı)")

    key(app, pygame.K_z, pygame.KMOD_CTRL)
    assert (int(ed.body.theta[new]), int(ed.body.length[new])) == before
    key(app, pygame.K_y, pygame.KMOD_CTRL)
    assert (int(ed.body.theta[new]), int(ed.body.length[new])) == after
    print("[2] Ctrl+Z / Ctrl+Y OK")

    ed.prop = "mass"; m0 = int(ed.body.mass[new])
    key(app, pygame.K_UP); assert int(ed.body.mass[new]) == m0 + 1
    key(app, pygame.K_DOWN); assert int(ed.body.mass[new]) == m0
    view, panel, _ = app.layout()
    buttons, rows, _ = ed.layout_panel(panel)
    for field, label, r, minus, plus in rows:
        if field == "length":
            l0 = int(ed.body.length[new])
            click(app, (panel.x + plus.centerx, panel.y + plus.centery))
            assert int(ed.body.length[new]) == min(l0 + 1, len(GRID["length"]) - 1)
    print("[2] ↑/↓ ve panel '+' düğmesi özellik değiştiriyor")

    n_before = ed.body.n
    click(app, sp(app, new), button=3)
    assert ed.body.n == n_before - 1
    ed.body.validate()
    key(app, pygame.K_z, pygame.KMOD_CTRL); assert ed.body.n == n_before
    print("[2] sağ tık silme + geri alma OK")

    view, panel, _ = app.layout()
    buttons, rows, _ = ed.layout_panel(panel)
    for r, label, fn in buttons:
        if label.startswith("Simetri"):
            click(app, (panel.x + r.centerx, panel.y + r.centery))
    assert ed.sym
    click(app, sp(app, 0)); assert ed.sel == 0
    nb = ed.body.n
    click(app, w2s(app, ed.pos()[0] + np.array([0.5, 0.0])))          # kafanın yanına (eksenin dışına)
    assert ed.body.n == nb + 2, (ed.body.n, nb)
    a, b = nb + 1, nb + 2
    A = GRID["angle_steps"]
    assert ed.pairs[a] == b and ed.pairs[b] == a
    assert (int(ed.body.theta[a]) + int(ed.body.theta[b])) % A == 0
    assert ed.body.lo[a] == ed.body.hi[b] and ed.body.hi[a] == ed.body.lo[b]
    ed.prop = "length"; ed.sel = a
    key(app, pygame.K_UP)
    assert ed.body.length[a] == ed.body.length[b]
    print("[2] simetri: eşleşen ayna uzuv eklendi, özellik değişimi eşine yansıyor")
    ed.sel = a; nm = ed.body.n
    key(app, pygame.K_m)
    assert ed.body.n == nm + 1
    ed.body.validate()
    frames(app, 2); shot(app, "editor_symmetry.png")

    assert not ed.body.legacy_spine and motor_mask_ok(ed)           # varsayılan: kafa yönü, kemik 1 motorlu
    key(app, pygame.K_o); assert ed.body.legacy_spine and not motor_mask_ok(ed)
    frames(app, 2); shot(app, "editor_legacy_spine.png")
    key(app, pygame.K_z, pygame.KMOD_CTRL); assert not ed.body.legacy_spine
    print("[2] gövde yönü = kafa (varsayılan); O ile eski omurga moduna geçiliyor, geri alınabiliyor")
    key(app, pygame.K_t); assert ed.test is not None
    frames(app, 240)
    assert np.isfinite(ed.test.st.x).all()
    shot(app, "editor_test.png")
    print(f"[2] yüzme testi çalışıyor (hız {ed.test.tele['speed']:.2f} m/s)")
    key(app, pygame.K_v); key(app, pygame.K_e)
    key(app, pygame.K_t); assert ed.test is None

    key(app, pygame.K_s, pygame.KMOD_CTRL)
    files = [f for f in os.listdir(out) if f.startswith("design_")]
    assert files, "kayıt oluşmadı"
    from genome import load_json
    g = load_json(os.path.join(out, files[0])); g.body.validate()
    assert any(it["name"].startswith("design_") for it in app.library)
    key(app, pygame.K_RETURN)
    assert app.mode == MODE_PLAY and app.play.genome.body.n == ed.body.n
    frames(app, 30, keys=[pygame.K_w]); assert np.isfinite(app.play.st.x).all()
    shot(app, "play_designed.png")
    print(f"[2] kaydedildi ({files[0]}), DENE modunda açıldı, WASD ile çalışıyor")
    key(app, pygame.K_e)
    app.screen = pygame.display.set_mode((800, 500), pygame.RESIZABLE)
    app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, size=(800, 500), w=800, h=500))
    frames(app, 3); shot(app, "editor_800x500.png")
    app.close()
    shutil.rmtree(out, ignore_errors=True)


def test_nose_bone1_deletable(load):
    out = tempfile.mkdtemp()
    app = make_app((1280, 720), train=False, load=load, out=out)
    key(app, pygame.K_TAB)
    ed = app.editor
    key(app, pygame.K_n); frames(app, 2)
    assert ed.body.n == 2 and not ed.body.legacy_spine
    click(app, sp(app, 0)); assert ed.sel == 0
    click(app, w2s(app, ed.pos()[0] + np.array([0.5, 0.0])))      # kafaya yeni uzuv (kemik 3)
    assert ed.body.n == 3 and ed.body.parent[3] == 0
    click(app, sp(app, 1), button=3)                              # kemik 1 + alt ağacı (kemik 2) silinir
    assert ed.body.n == 1, ed.body.n
    ed.body.validate()
    key(app, pygame.K_z, pygame.KMOD_CTRL); assert ed.body.n == 3
    key(app, pygame.K_o); assert ed.body.legacy_spine           # eski modda kemik 1 hâlâ korunur
    n = ed.body.n
    click(app, sp(app, 1), button=3); assert ed.body.n == n
    key(app, pygame.K_o)
    print("[2b] burun modunda kemik 1 silinebiliyor (geri alınır); eski modda korunuyor")
    app.close()
    shutil.rmtree(out, ignore_errors=True)


def test_with_trainer():
    out = tempfile.mkdtemp()
    app = make_app((1280, 720), train=True, out=out, pop=16, trials=2, duration=4.0, stage_len=1, workers=1, curriculum="fixed")
    t0 = time.time()
    while time.time() - t0 < 60 and not (len(app.history) >= 5 and any(h.get("validity") is not None for h in app.history)):
        frames(app, 6)
        time.sleep(0.1)
    n = len(app.history)
    print(f"[3] eğitim süreci: {n} nesil alındı, aşama {app.stage}, "
          f"doğrulama {[round(h['validity'], 2) for h in app.history if h.get('validity') is not None]}")
    assert n >= 4 and app.stage >= 2, (n, app.stage)
    frames(app, 100)
    shot(app, "train_1280x720.png")
    key(app, pygame.K_SPACE); frames(app, 3); shot(app, "turbo.png"); key(app, pygame.K_SPACE)
    key(app, pygame.K_p); time.sleep(1.5); app.update(0.01); assert app.paused
    key(app, pygame.K_p)
    key(app, pygame.K_TAB)
    frames(app, 30)
    assert app.library[0]["name"].startswith("canlı")
    t = time.time(); app.close()
    assert not app.trainer.alive()
    print(f"[3] aşama 2 + doğrulama mesajı, duraklat/sürdür, kapanış {time.time()-t:.1f} sn OK")
    shutil.rmtree(out, ignore_errors=True)


def test_send_to_training(load):
    base = tempfile.mkdtemp()
    out = os.path.join(base, "app1")
    app = make_app((1280, 720), train=False, load=load, out=out, pop=12, trials=2, duration=4.0, workers=1)
    key(app, pygame.K_TAB)
    assert app.mode == MODE_EDIT
    ed = app.editor
    key(app, pygame.K_n)                                     # boş canlı: 2 uzuv
    click(app, sp(app, 2))
    click(app, w2s(app, ed.pos()[2] + np.array([0.5, 0.0])))
    click(app, sp(app, 3))
    click(app, w2s(app, ed.pos()[3] + np.array([0.0, -0.5])))
    n = ed.body.n
    assert n >= 4
    view, panel, _ = app.layout()
    buttons, rows, _ = ed.layout_panel(panel)
    for r, label, fn in buttons:
        if label.startswith("Eğitime"):
            click(app, (panel.x + r.centerx, panel.y + r.centery))
    assert ed.confirm
    frames(app, 2); shot(app, "editor_send_training.png")
    key(app, pygame.K_ESCAPE); assert not ed.confirm and app.running      # ESC sadece pencereyi kapatır
    ed.ask_training()
    key(app, pygame.K_1)                                     # vücut SABİT
    assert app.mode == MODE_TRAIN and app.trainer is not None and app.trainer.alive()
    assert os.path.basename(app.out).startswith("tasarim_")
    assert os.path.exists(os.path.join(app.out, "tasarim.json"))
    t0 = time.time()
    while time.time() - t0 < 60 and len(app.history) < 3:
        frames(app, 6); time.sleep(0.1)
    assert len(app.history) >= 3, "yeni eğitimden mesaj gelmiyor"
    assert all(h["best_limbs"] == n for h in app.history), [h["best_limbs"] for h in app.history]
    assert abs(app.history[-1]["mean_limbs"] - n) < 1e-9
    frames(app, 30); shot(app, "train_from_design.png")
    print(f"[4] tasarımdan eğitim: {len(app.history)} nesil geldi, tüm bireylerde uzuv sayısı {n} (vücut sabit), "
          f"klasör {os.path.basename(app.out)}")
    app.close()
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    mp.freeze_support()
    load = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fish.json"
    test_play_controls(load)
    test_editor(load)
    test_nose_bone1_deletable(load)
    test_with_trainer()
    test_send_to_training(load)
    print("TAMAM ->", SHOTS)

"""Gerçek zamanlı canlı oturumu (fizik + beyin + telemetri) ve kamera."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from brain import act, stack_brains
from evolution import EvalConfig
from genome import Genome, build_bodies
from physics import PhysicsParams, center_of_mass, com_velocity, init_state, motor_mask, step

# --------------------------------------------------------------------------- #
#  Canlı oturumu: tek bir canlının gerçek zamanlı simülasyonu + telemetri
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, genome: Genome, label: str, heading: float = 0.0):
        self.genome, self.label = genome, label
        self.bd = build_bodies([genome], n_points=genome.body.n + 2)
        self.bb = stack_brains([genome.brain])
        self.pp = PhysicsParams()
        self.control_every = max(1, int(round(EvalConfig().control_dt / self.pp.dt)))   # eğitimle aynı kontrol hızı
        self.mm = motor_mask(self.bd)[0]
        self.reset(heading)

    def reset(self, heading: float = 0.0):
        self.st = init_state(self.bd, heading=[heading])
        self.ctrl = np.zeros((1, self.bd.N))
        self.steps = 0
        self.acc = 0.0
        self.t = 0.0
        self.h0 = heading
        self.energy_total = 0.0
        self.trail = deque(maxlen=240)
        self.progress_hist = deque(maxlen=120)      # 0.1 sn örnek, son 12 sn
        self._last_sample = -1.0
        self.target = heading
        self.mag = 1.0
        self.motors = True
        self.tele = {}
        self._telemetry()

    def compute_ctrl(self, target: float, mag: float) -> np.ndarray:
        """Beyin çıktısı. mag=0: 'komut yok' (beyne sıfır hedef vektörü)."""
        return act(self.st, self.bd, self.bb, np.array([target]), np.array([mag]))

    def advance(self, dt: float, target: float, mag: float = 1.0, motors: bool = True):
        """dt kadar gerçek zamanlı ilerlet. motors=False: motorlar kapalı (beyin devre dışı, canlı süzülür)."""
        self.target, self.mag, self.motors = target, mag, motors
        self.acc += dt
        n = int(self.acc / self.pp.dt)
        self.acc -= n * self.pp.dt
        n = min(n, 80)
        for _ in range(n):
            if self.steps % self.control_every == 0:
                self.ctrl = self.compute_ctrl(target, mag) if motors else np.zeros((1, self.bd.N))
            step(self.st, self.bd, self.ctrl, self.pp)
            self.steps += 1
        self.t = self.steps * self.pp.dt
        self._telemetry(n * self.pp.dt)
        if self.t - self._last_sample >= 0.1:
            self._last_sample = self.t
            self.trail.append(self.tele["com"].copy())
            self.progress_hist.append(self.tele["progress"])

    def _telemetry(self, sim_dt=0.0):
        st, bd = self.st, self.bd
        com = center_of_mass(st, bd)[0]
        vel = com_velocity(st, bd)[0]
        u = np.array([math.cos(self.target), math.sin(self.target)]) * self.mag
        mm = self.mm
        c = self.ctrl[0]
        nm = max(int(mm.sum()), 1)
        power = float(np.sum(np.abs(c * bd.tau_max[0] * st.jvel[0])[mm]))
        self.energy_total += power * sim_dt
        self.tele = dict(com=com, vel=vel, speed=float(np.linalg.norm(vel)), heading=float(st.phi[0]),
                         target=self.target, progress=float(vel @ u),
                         lateral=float(vel[0] * u[1] - vel[1] * u[0]),
                         energy=float(np.sum(c[mm] ** 2) / nm), power=power, total=self.energy_total,
                         t=self.t, active=bool(self.mag > 0.5 and self.motors))


# --------------------------------------------------------------------------- #
#  Kamera
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self):
        self.c = np.zeros(2)
        self.scale = 60.0
        self.auto = True

    def update(self, sess: Session, view_wh, dt):
        com = sess.tele["com"]
        k = 1 - math.exp(-6 * dt)
        self.c += (com - self.c) * k
        if self.auto:
            pts = sess.st.x[0][sess.bd.active[0]]
            radius = float(np.max(np.linalg.norm(pts - com, axis=1))) if len(pts) else 1.0
            tgt = np.clip(0.30 * min(view_wh) / max(radius, 0.8), 22.0, 140.0)
            self.scale += (tgt - self.scale) * (1 - math.exp(-3 * dt))

    def w2s(self, p, view_wh):
        return (view_wh[0] / 2 + (p[0] - self.c[0]) * self.scale,
                view_wh[1] / 2 - (p[1] - self.c[1]) * self.scale)




class WaveSession(Session):
    """Editörde 'yüzme testi': beyin yerine her kolda gövdeden uca yayılan açı dalgası (PD ile sürülür).
    Morfolojinin fiziksel olarak itki üretip üretmediğini beyinden bağımsız gösterir."""

    def __init__(self, genome, label, heading=0.0, freq=1.5, direction=+1, k=0.9):
        super().__init__(genome, label, heading)
        self.freq, self.direction, self.k = freq, direction, k
        par = self.bd.parent[0]
        depth = np.zeros(self.bd.N)
        for i in range(1, self.bd.N):
            depth[i] = depth[par[i]] + 1
        self.depth = depth

    def compute_ctrl(self, target, mag):
        bd, st = self.bd, self.st
        amp = 0.7 * np.minimum(-bd.jlo[0], bd.jhi[0])
        tgt = amp * np.sin(2 * math.pi * self.freq * st.t - self.direction * self.k * self.depth)
        c = np.clip(30.0 * (tgt - st.theta[0]) - 2.0 * st.jvel[0], -1.0, 1.0)
        return (c * self.mm)[None, :]

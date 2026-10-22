"""
Faz 2 - Algılar ve beyin ileri geçişi (popülasyon vektörize).

Girdi vektörü (N_IN = 79):
    [0:24]   eklem açıları (dinlenmeye göre, sınır aralığına normalize)
    [24:48]  eklem açısal hızları (tanh ölçekli)
    [48:72]  uzuv yuvası maskesi (1 = bu yuvada gerçek uzuv var)
    [72:74]  hedef yön, GÖVDE çerçevesinde birim vektör (x ileri); (0,0) = komut yok / dur
    [74:76]  gövde çerçevesinde COM hızı (tanh ölçekli)
    [76]     gövde açısal hızı (tanh ölçekli)
    [77:79]  saat sin/cos (ritmi hazır vermez, sadece zaman bilgisi)
Çıktı: 24 motor komutu (tanh -> [-1,1]); motorlu olmayan yuvalar sıfırlanır (motor_mask).
Yuva s, kemik/nokta s+1'e karşılık gelir.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from genome import BrainGenome, N_IN, HIDDEN, N_OUT, N_SLOTS
from physics import Bodies, State, com_velocity, motor_mask

CLOCK_HZ = 1.0
VEL_SCALE = 1.0       # COM hızı normalizasyonu
JVEL_SCALE = 4.0
OMEGA_SCALE = 2.0


@dataclass
class BrainBatch:
    W1: np.ndarray   # (B,N_IN,H)
    b1: np.ndarray   # (B,H)
    W2: np.ndarray   # (B,H,O)
    b2: np.ndarray   # (B,O)


def stack_brains(brains: list[BrainGenome]) -> BrainBatch:
    return BrainBatch(np.stack([b.W1 for b in brains]), np.stack([b.b1 for b in brains]),
                      np.stack([b.W2 for b in brains]), np.stack([b.b2 for b in brains]))


def _to_body_frame(vec, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * vec[:, 0] + s * vec[:, 1], -s * vec[:, 0] + c * vec[:, 1]], -1)


def observe(st: State, bd: Bodies, target_angle: np.ndarray, target_mag=None) -> np.ndarray:
    """target_angle (B,): DÜNYA çerçevesinde hedef yön açısı (WASD'den gelir). Dönüş: (B,N_IN).
    target_mag (B,) 1 = komut var, 0 = KOMUT YOK (dur). Verilmezse 1.
    bd.N < 25 ise kullanılmayan yuvalar sıfırla doldurulur."""
    B = bd.B
    sl = slice(1, bd.N - 1)                          # son nokta (burun / dolgu) uzuv yuvası değil
    rng_ = np.maximum(np.maximum(-bd.jlo[:, sl], bd.jhi[:, sl]), 0.1)
    theta = np.clip(st.theta[:, sl] / rng_, -1.5, 1.5)
    jvel = np.tanh(st.jvel[:, sl] / JVEL_SCALE)
    mask = motor_mask(bd)[:, sl].astype(float)
    theta, jvel = theta * mask, jvel * mask
    act_ = bd.active[:, sl].astype(float)
    pad = N_SLOTS - (bd.N - 2)
    if pad:
        theta, jvel, act_ = (np.pad(a, ((0, 0), (0, pad))) for a in (theta, jvel, act_))
    tgt_world = np.stack([np.cos(target_angle), np.sin(target_angle)], -1)
    tgt = _to_body_frame(tgt_world, st.phi)
    if target_mag is not None:
        tgt = tgt * np.asarray(target_mag, float).reshape(-1, 1)
    vel = np.tanh(_to_body_frame(com_velocity(st, bd), st.phi) / VEL_SCALE)
    om = np.tanh(st.omega / OMEGA_SCALE)[:, None]
    clk = np.stack([np.sin(2 * np.pi * CLOCK_HZ * st.t) * np.ones(B), np.cos(2 * np.pi * CLOCK_HZ * st.t) * np.ones(B)], -1)
    obs = np.concatenate([theta, jvel, act_, tgt, vel, om, clk], axis=1)
    assert obs.shape[1] == N_IN
    return obs


def forward(bb: BrainBatch, obs: np.ndarray) -> np.ndarray:
    n_in = bb.W1.shape[1]
    if obs.shape[1] < n_in:                    # v2 beyin v1 gözlemiyle çalışırsa fazladan girişler 0
        obs = np.pad(obs, ((0, 0), (0, n_in - obs.shape[1])))
    h = np.tanh(np.einsum("bi,bih->bh", obs, bb.W1) + bb.b1)
    return np.tanh(np.einsum("bh,bho->bo", h, bb.W2) + bb.b2)


def act(st: State, bd: Bodies, bb: BrainBatch, target_angle: np.ndarray, target_mag=None) -> np.ndarray:
    """Fizik adımına verilecek (B,N) komut dizisi. [:,0] ve motorsuz kemikler 0."""
    out = forward(bb, observe(st, bd, target_angle, target_mag))
    ctrl = np.zeros((bd.B, bd.N))
    ctrl[:, 1:bd.N - 1] = out[:, :bd.N - 2]
    return ctrl * motor_mask(bd)


def act2(st: State, bd: Bodies, bb: BrainBatch, target_angle: np.ndarray, target_mag: np.ndarray, extra: np.ndarray):
    """Beyin v2 (ekosistem). extra (B,6): [yem yakınlığı, kendi enerjisi, canlı yönü x/y (gövde çerçevesi), canlı yakınlığı,
    canlının enerjisi]. Dönüş: (ctrl (B,N), istekler (B,3) = [eşeyli üreme, eşeysiz üreme, ısırma], hepsi [-1,1])."""
    obs = np.concatenate([observe(st, bd, target_angle, target_mag), extra], axis=1)
    out = forward(bb, obs)
    ctrl = np.zeros((bd.B, bd.N))
    ctrl[:, 1:bd.N - 1] = out[:, :bd.N - 2]
    return ctrl * motor_mask(bd), out[:, N_OUT:N_OUT + 3]

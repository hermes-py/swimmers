"""
Phase 2 - Senses and brain forward pass (population-vectorized).

Input vector (N_IN = 79):
    [0:24]   joint angles (relative to rest, normalized to the limit range)
    [24:48]  joint angular velocities (tanh-scaled)
    [48:72]  limb slot mask (1 = a real limb exists in this slot)
    [72:74]  target direction, unit vector in the BODY frame (x forward); (0,0) = no command / stop
    [74:76]  COM velocity in the body frame (tanh-scaled)
    [76]     body angular velocity (tanh-scaled)
    [77:79]  clock sin/cos (does not hand over the rhythm, only time information)
Output: 24 motor commands (tanh -> [-1,1]); slots without a motor are zeroed (motor_mask).
Slot s corresponds to bone/point s+1.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from genome import BrainGenome, N_IN, HIDDEN, N_OUT, N_SLOTS
from physics import Bodies, State, com_velocity, motor_mask

CLOCK_HZ = 1.0
VEL_SCALE = 1.0       # COM velocity normalization
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
    """target_angle (B,): target direction angle in the WORLD frame (comes from WASD). Returns: (B,N_IN).
    target_mag (B,) 1 = command present, 0 = NO COMMAND (stop). Defaults to 1 if not given.
    If bd.N < 25, unused slots are filled with zeros."""
    B = bd.B
    sl = slice(1, bd.N - 1)                          # the last point (nose / padding) is not a limb slot
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
    if obs.shape[1] < n_in:                    # if a v2 brain runs with v1 observations, the extra inputs are 0
        obs = np.pad(obs, ((0, 0), (0, n_in - obs.shape[1])))
    h = np.tanh(np.einsum("bi,bih->bh", obs, bb.W1) + bb.b1)
    return np.tanh(np.einsum("bh,bho->bo", h, bb.W2) + bb.b2)


def act(st: State, bd: Bodies, bb: BrainBatch, target_angle: np.ndarray, target_mag=None) -> np.ndarray:
    """(B,N) command array to feed to the physics step. [:,0] and motorless bones are 0."""
    out = forward(bb, observe(st, bd, target_angle, target_mag))
    ctrl = np.zeros((bd.B, bd.N))
    ctrl[:, 1:bd.N - 1] = out[:, :bd.N - 2]
    return ctrl * motor_mask(bd)


def act2(st: State, bd: Bodies, bb: BrainBatch, target_angle: np.ndarray, target_mag: np.ndarray, extra: np.ndarray):
    """Brain v2 (ecosystem). extra (B,6): [food proximity, own energy, other creature direction x/y (body frame),
    other creature proximity, other creature's energy]. Returns: (ctrl (B,N), requests (B,3) = [sexual reproduction,
    asexual reproduction, bite], all in [-1,1])."""
    obs = np.concatenate([observe(st, bd, target_angle, target_mag), extra], axis=1)
    out = forward(bb, obs)
    ctrl = np.zeros((bd.B, bd.N))
    ctrl[:, 1:bd.N - 1] = out[:, :bd.N - 2]
    return ctrl * motor_mask(bd), out[:, N_OUT:N_OUT + 3]

"""Shared UI helpers: colors, fonts, arrow/text drawing, creature drawing."""
from __future__ import annotations

import math

import numpy as np
import pygame

# --------------------------------------------------------------------------- #
#  Colors
# --------------------------------------------------------------------------- #
C_BG_TOP, C_BG_BOT = (10, 38, 66), (4, 16, 34)
C_GRID_MINOR, C_GRID_MAJOR = (18, 58, 92), (30, 84, 126)
C_PANEL, C_PANEL_LINE = (14, 20, 30), (44, 58, 78)
C_TEXT, C_DIM = (226, 232, 240), (130, 146, 166)
C_TARGET, C_HEAD, C_VEL = (70, 220, 255), (90, 230, 120), (255, 214, 70)
C_BONE, C_SPINE = (120, 205, 225), (150, 160, 175)
C_POS, C_NEG = (255, 150, 60), (90, 140, 255)
C_BEST, C_MEAN = (90, 230, 120), (255, 170, 70)


HEAD_UP = math.pi / 2     # initial body direction: head faces UP (consistent with W = up)


def deg(a):
    return math.degrees(a) % 360.0


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
#  Fonts (scale with window height)
# --------------------------------------------------------------------------- #
class Fonts:
    NAMES = "segoeui,arial,dejavusans,freesans,helvetica,verdana"

    def __init__(self, h):
        self.h = h
        mk = lambda s: pygame.font.SysFont(self.NAMES, s)
        self.small = mk(max(11, h // 52))
        self.normal = mk(max(13, h // 42))
        self.big = mk(max(18, h // 24))
        self.huge = mk(max(28, h // 12))


def text(surf, font, s, pos, color=C_TEXT, anchor="topleft"):
    img = font.render(s, True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def arrow(surf, color, p0, p1, width=3, head=9):
    pygame.draw.line(surf, color, p0, p1, width)
    d = (p1[0] - p0[0], p1[1] - p0[1])
    L = math.hypot(*d)
    if L < 1:
        return
    ux, uy = d[0] / L, d[1] / L
    h = min(head, L * 0.6)
    a = (p1[0] - ux * h - uy * h * 0.55, p1[1] - uy * h + ux * h * 0.55)
    b = (p1[0] - ux * h + uy * h * 0.55, p1[1] - uy * h - ux * h * 0.55)
    pygame.draw.polygon(surf, color, [p1, a, b])




def draw_creature(surf, sess, cam, wh):
    """Draws the creature in a session (Session) relative to the camera. Bones are tinted orange/blue according to the command.
    Also works for padded creatures within a batch (with unused points): only active points are drawn."""
    sc = cam.scale
    bd, st = sess.bd, sess.st
    x = st.x[0]
    par = bd.parent[0]
    ctrl = sess.ctrl[0]
    legacy = bool(bd.legacy_spine[0])
    nose = None if legacy else int(bd.N - 1)           # in the new mode the last point = nose (invisible direction carrier)
    idx = [int(i) for i in np.nonzero(bd.active[0])[0]]
    S = {i: cam.w2s(x[i], wh) for i in idx}
    limbs = [i for i in idx if i > 0 and i != nose]
    for i in limbs:
        wpx = max(3, int(bd.width[0, i] * sc))
        if i == 1 and legacy:
            col = C_SPINE
        else:
            c = float(np.clip(ctrl[i], -1, 1))
            tgt = C_POS if c > 0 else C_NEG
            a = abs(c)
            col = tuple(int(C_BONE[k] * (1 - a) + tgt[k] * a) for k in range(3))
        pygame.draw.line(surf, col, S[int(par[i])], S[i], wpx)
        pygame.draw.circle(surf, col, S[i], wpx // 2)
    for i in limbs:
        r = max(3, int(0.055 * math.sqrt(bd.mass[0, i]) * sc))
        pygame.draw.circle(surf, (235, 245, 250), S[i], r)
        pygame.draw.circle(surf, (20, 40, 60), S[i], r, 1)
    rh = max(5, int(0.10 * math.sqrt(bd.mass[0, 0]) * sc))
    pygame.draw.circle(surf, (250, 250, 255), S[0], rh)
    pygame.draw.circle(surf, C_HEAD, S[0], rh, 2)
    ph = float(st.phi[0])
    e = (S[0][0] + math.cos(ph) * rh * 0.55, S[0][1] - math.sin(ph) * rh * 0.55)
    pygame.draw.circle(surf, (10, 20, 30), e, max(2, rh // 4))

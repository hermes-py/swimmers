"""
Phase 2 - Genome: BodyGenome + BrainGenome, grid quantization, mutation, crossover, JSON.

The body genome consists entirely of INTEGER levels (grid); real values are read from the GRID table.
Changing the table (number of levels / values) is done in one place, in GRID.

Tree rule: parent[i] < i. Bone 1 is always attached to the head (0) and defines the body direction.
This rule is enforced only in the generation/mutation code; the data structure is a general "bone list".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
import numpy as np

from physics import MAX_LIMBS, make_bodies

# --------------------------------------------------------------------------- #
#  GRID: discrete values (change as you like)
# --------------------------------------------------------------------------- #
GRID = dict(
    angle_steps=24,                                   # 24 steps = 15 degrees
    length=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],  # 8 levels
    mass=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0],             # 6 levels
    width=[0.05, 0.10, 0.15, 0.25],                   # bone thickness (drag surface)
    tau=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0],              # torque capacity
    damp=[0.05, 0.15, 0.3, 0.6],                      # joint damping
    limit_step=0.2, limit_levels=8,                   # joint angle limit: (k+1)*0.2 rad, k=0..7
)
FIELDS = ("theta", "length", "mass", "width", "tau", "damp", "lo", "hi")
MIN_LIMBS = 2


def n_levels(f: str) -> int:
    if f == "theta":
        return GRID["angle_steps"]
    if f in ("lo", "hi"):
        return GRID["limit_levels"]
    return len(GRID[f])


def level_value(f: str, k: int) -> float:
    if f == "theta":
        a = k * 2 * np.pi / GRID["angle_steps"]
        return float((a + np.pi) % (2 * np.pi) - np.pi)
    if f in ("lo", "hi"):
        return float((k + 1) * GRID["limit_step"])
    return float(GRID[f][k])


# --------------------------------------------------------------------------- #
#  BodyGenome
# --------------------------------------------------------------------------- #
@dataclass
class BodyGenome:
    parent: np.ndarray            # (n+1,) int; [0] unused
    theta: np.ndarray             # (n+1,) angle level (rest angle relative to the reference bone)
    length: np.ndarray
    mass: np.ndarray              # [0] = head mass
    width: np.ndarray
    tau: np.ndarray
    damp: np.ndarray
    lo: np.ndarray                # lower limit = -(lo+1)*step
    hi: np.ndarray                # upper limit = +(hi+1)*step
    legacy_spine: bool = True      # True: OLD mode (bone 1 is a passive spine, it determines the body direction). New creatures False:
                                   # body direction is the head's direction, all limbs including bone 1 are motorized. Old files without the flag count as True.

    @property
    def n(self) -> int:
        return len(self.parent) - 1

    @property
    def first_free(self) -> int:
        """First limb that can be deleted/mirrored. In the old mode bone 1 is the passive spine (2); in the new
        (head direction) mode bone 1 is an ordinary limb too (1)."""
        return 2 if self.legacy_spine else 1

    @property
    def min_n(self) -> int:
        """Structural minimum limb count (1 in the new mode; MIN_LIMBS in the old mode)."""
        return MIN_LIMBS if self.legacy_spine else 1

    def copy(self) -> "BodyGenome":
        return BodyGenome(legacy_spine=self.legacy_spine, **{k: getattr(self, k).copy() for k in ("parent",) + FIELDS})

    def children(self, i: int):
        return [j for j in range(1, self.n + 1) if self.parent[j] == i]

    def validate(self):
        assert 1 <= self.n <= MAX_LIMBS, f"limb count {self.n}"
        assert self.parent[1] == 0
        for i in range(1, self.n + 1):
            assert 0 <= self.parent[i] < i, "tree order broken"
        for f in FIELDS:
            a = getattr(self, f)
            assert len(a) == self.n + 1
            assert a.min() >= 0 and a.max() < n_levels(f), f"{f} outside the grid"

    # --- JSON ---
    def to_dict(self) -> dict:
        limbs = []
        for i in range(1, self.n + 1):
            d = {"id": i, "parent": int(self.parent[i])}
            for f in FIELDS:
                d[f] = int(getattr(self, f)[i])
            d["values"] = {
                "angle_deg": round(float(np.degrees(level_value("theta", self.theta[i]))), 1),
                "length": level_value("length", self.length[i]), "mass": level_value("mass", self.mass[i]),
                "width": level_value("width", self.width[i]), "tau_max": level_value("tau", self.tau[i]),
                "damping": level_value("damp", self.damp[i]),
                "limit_lo_rad": -level_value("lo", self.lo[i]), "limit_hi_rad": level_value("hi", self.hi[i]),
            }
            limbs.append(d)
        return {"n_limbs": self.n, "legacy_spine": bool(self.legacy_spine), "head_mass": int(self.mass[0]),
                "head_mass_value": level_value("mass", self.mass[0]), "limbs": limbs,
                "_note": "Level (grid) indices are authoritative; 'values' is for readability only."}

    @staticmethod
    def from_dict(d: dict) -> "BodyGenome":
        n = len(d["limbs"])
        arr = {k: np.zeros(n + 1, dtype=np.int64) for k in ("parent",) + FIELDS}
        arr["mass"][0] = d["head_mass"]
        for L in sorted(d["limbs"], key=lambda x: x["id"]):
            i = L["id"]
            arr["parent"][i] = L["parent"]
            for f in FIELDS:
                arr[f][i] = L[f]
        b = BodyGenome(legacy_spine=bool(d.get("legacy_spine", True)), **arr)
        b.validate()
        return b


def random_body(rng: np.random.Generator, n_min=3, n_max=8, legacy_spine: bool = False) -> BodyGenome:
    n = int(rng.integers(n_min, n_max + 1))
    parent = np.zeros(n + 1, dtype=np.int64)
    for i in range(2, n + 1):
        parent[i] = i - 1 if rng.random() < 0.6 else rng.integers(0, i)
    arr = {f: rng.integers(0, n_levels(f), n + 1) for f in FIELDS}
    for f in FIELDS:
        if f != "mass":
            arr[f][0] = 0          # only the head's mass is meaningful
    b = BodyGenome(parent=parent, legacy_spine=legacy_spine, **arr)
    b.validate()
    return b


# --- structural mutations --------------------------------------------------- #
def _append_limb(b: BodyGenome, parent: int, vals: dict) -> BodyGenome:
    out = BodyGenome(parent=np.append(b.parent, parent), legacy_spine=b.legacy_spine,
                     **{f: np.append(getattr(b, f), vals[f]) for f in FIELDS})
    return out


def add_limb(b: BodyGenome, rng) -> BodyGenome:
    if b.n >= MAX_LIMBS:
        return b
    src = int(rng.integers(1, b.n + 1))
    vals = {f: int(getattr(b, f)[src]) for f in FIELDS}
    vals["theta"] = int(rng.integers(0, n_levels("theta")))
    return _append_limb(b, int(rng.integers(0, b.n + 1)), vals)


def remove_limb(b: BodyGenome, rng, min_limbs: int = MIN_LIMBS) -> BodyGenome:
    """Only a leaf (childless) limb that is not bone 1 is deleted; indices are shifted."""
    if b.n <= max(min_limbs, b.min_n):
        return b
    leaves = [i for i in range(b.first_free, b.n + 1) if not b.children(i)]
    if not leaves:
        return b
    r = int(rng.choice(leaves))
    keep = [i for i in range(b.n + 1) if i != r]
    newp = b.parent[keep].copy()
    newp = np.where(newp > r, newp - 1, newp)
    out = BodyGenome(parent=newp, legacy_spine=b.legacy_spine, **{f: getattr(b, f)[keep] for f in FIELDS})
    return out


def subtree(b: BodyGenome, s: int) -> list:
    """Limb s and all its descendants (in increasing index order)."""
    sub = [s]
    for i in range(s + 1, b.n + 1):
        if b.parent[i] in sub:
            sub.append(i)
    return sub


def mirror_at(b: BodyGenome, s: int):
    """Add limb s (with its subtree) at the same joint point as a MIRROR image -> bilateral symmetry.
    Returns: (new_body, {old_index: mirror_index}) or None if it doesn't fit. Bone 1 cannot be mirrored (in the old mode)."""
    if s < b.first_free or s > b.n:
        return None
    sub = subtree(b, s)
    if b.n + len(sub) > MAX_LIMBS:
        return None
    out, idx = b, {}
    A = GRID["angle_steps"]
    for i in sub:
        vals = {f: int(getattr(b, f)[i]) for f in FIELDS}
        vals["theta"] = (A - vals["theta"]) % A
        vals["lo"], vals["hi"] = vals["hi"], vals["lo"]
        p = int(b.parent[s]) if i == s else idx[int(b.parent[i])]
        out = _append_limb(out, p, vals)
        idx[i] = out.n
    return out, idx


def mirror_limb(b: BodyGenome, rng) -> BodyGenome:
    """Adds the mirror-symmetric counterpart of a random limb (for mutation)."""
    if b.n >= MAX_LIMBS or b.n < b.first_free:
        return b
    r = mirror_at(b, int(rng.integers(b.first_free, b.n + 1)))
    return b if r is None else r[0]


def delete_set(b: BodyGenome, rm: set):
    """Delete the limbs in the set rm, shift the indices. Returns: (body, {old: new}) or None if invalid.
    In the old mode bone 1 (spine) cannot be deleted; the limb count cannot drop below b.min_n."""
    if (1 in rm and b.legacy_spine) or 0 in rm or not rm or b.n - len(rm) < b.min_n:
        return None
    keep = [j for j in range(b.n + 1) if j not in rm]
    remap = {old: new for new, old in enumerate(keep)}
    newp = np.array([remap[int(b.parent[j])] if j > 0 else 0 for j in keep], dtype=np.int64)
    return BodyGenome(parent=newp, legacy_spine=b.legacy_spine, **{f: getattr(b, f)[keep] for f in FIELDS}), remap


def delete_subtree(b: BodyGenome, i: int):
    return delete_set(b, set(subtree(b, i))) if b.first_free <= i <= b.n else None


def add_limb_at(b: BodyGenome, parent: int, theta: int, length: int, template: int = None) -> BodyGenome:
    """Add a new limb at point `parent` (with grid levels). template: the limb whose properties are copied."""
    assert b.n < MAX_LIMBS
    if template is not None and template >= 1:
        vals = {f: int(getattr(b, f)[template]) for f in FIELDS}
    else:
        vals = dict(theta=0, length=3, mass=3, width=2, tau=2, damp=1, lo=3, hi=3)
    vals["theta"], vals["length"] = int(theta), int(length)
    return _append_limb(b, int(parent), vals)


def blank_body() -> BodyGenome:
    """The editor's starting creature: head + 2 limbs (bone 1 on the head, bone 2 attached to it). In the new mode both are ordinary limbs."""
    def arr(vals):
        return np.array(vals, dtype=np.int64)
    return BodyGenome(legacy_spine=False, parent=arr([0, 0, 1]), theta=arr([0, 12, 0]), length=arr([0, 4, 3]), mass=arr([3, 3, 3]),
                      width=arr([0, 2, 2]), tau=arr([0, 2, 2]), damp=arr([0, 1, 1]), lo=arr([0, 3, 3]),
                      hi=arr([0, 3, 3]))


@dataclass
class MutationConfig:
    p_gene: float = 0.08          # probability of a level change for each gene
    p_add: float = 0.05
    p_remove: float = 0.04
    p_mirror: float = 0.03
    brain_rate: float = 0.08      # mutation probability for each weight
    brain_sigma: float = 0.25
    brain_reset: float = 0.005    # probability of completely redrawing a weight
    min_limbs: int = 4            # minimum limb count that deletion by mutation can reach (prevents 2-limb 'V' fish)


def mutate_body(b: BodyGenome, rng, cfg: MutationConfig) -> BodyGenome:
    b = b.copy()
    if rng.random() < cfg.p_add:
        b = add_limb(b, rng)
    if rng.random() < cfg.p_remove:
        b = remove_limb(b, rng, cfg.min_limbs)
    if rng.random() < cfg.p_mirror:
        b = mirror_limb(b, rng)
    for f in FIELDS:
        a = getattr(b, f)
        start = 0 if f == "mass" else 1       # the head only has a mass
        for i in range(start, b.n + 1):
            if rng.random() < cfg.p_gene:
                step = int(rng.choice([-1, 1]))
                if f == "theta":
                    a[i] = (a[i] + step) % n_levels("theta")
                else:
                    a[i] = int(np.clip(a[i] + step, 0, n_levels(f) - 1))
    b.validate()
    return b


def crossover_body(a: BodyGenome, b: BodyGenome, rng) -> BodyGenome:
    """Topology from A; for the shared limb indices each gene comes separately from A or B."""
    c = a.copy()
    m = min(a.n, b.n)
    for f in FIELDS:
        mask = rng.random(m + 1) < 0.5
        arr = getattr(c, f)
        arr[: m + 1][mask] = getattr(b, f)[: m + 1][mask]
    c.validate()
    return c


# --------------------------------------------------------------------------- #
#  BrainGenome (fixed-topology MLP: input -> hidden(tanh) -> output(tanh))
# --------------------------------------------------------------------------- #
N_SLOTS = MAX_LIMBS            # number of limb slots
N_GLOBAL = 7                   # target(2) + body velocity(2) + angular velocity(1) + clock sin/cos(2)
N_IN = 3 * N_SLOTS + N_GLOBAL  # 79
HIDDEN = 24
N_OUT = N_SLOTS                # 24
# Brain v2 (ecosystem): 6 new inputs (food proximity, own energy, nearest creature's direction(2)/proximity/energy) and
# 3 new outputs (sexual reproduction request, asexual reproduction request, bite request - biting will be used in stage 3).
# v1 brains can be upgraded to v2 (new inputs are ignored, new outputs are off) and v2 brains work in v1 code.
N_IN_V2 = N_IN + 6
N_OUT_V2 = N_OUT + 3


@dataclass
class BrainGenome:
    W1: np.ndarray   # (N_IN, HIDDEN)
    b1: np.ndarray   # (HIDDEN,)
    W2: np.ndarray   # (HIDDEN, N_OUT)
    b2: np.ndarray   # (N_OUT,)

    def copy(self):
        return BrainGenome(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def to_dict(self):
        return {"n_in": int(self.W1.shape[0]), "hidden": HIDDEN, "n_out": int(self.W2.shape[1]),
                "W1": self.W1.tolist(), "b1": self.b1.tolist(), "W2": self.W2.tolist(), "b2": self.b2.tolist()}

    @staticmethod
    def from_dict(d):
        assert (d["n_in"], d["hidden"], d["n_out"]) in ((N_IN, HIDDEN, N_OUT), (N_IN_V2, HIDDEN, N_OUT_V2)), "brain size mismatch"
        return BrainGenome(*(np.array(d[k], dtype=float) for k in ("W1", "b1", "W2", "b2")))


def random_brain(rng, n_in: int = N_IN, n_out: int = N_OUT) -> BrainGenome:
    return BrainGenome(rng.normal(0, 1 / np.sqrt(n_in), (n_in, HIDDEN)), np.zeros(HIDDEN),
                       rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, n_out)), np.zeros(n_out))


def upgrade_brain(br: BrainGenome, want_bias=(-2.0, -2.0, -2.0)) -> BrainGenome:
    """v1 -> v2: weights of the new inputs are 0 (ignored), weights of the new outputs are 0 and their bias is want_bias
    (sexual reproduction, asexual reproduction, bite). Default: all off."""
    if br.W1.shape[0] == N_IN_V2:
        return br.copy()
    W1 = np.vstack([br.W1, np.zeros((N_IN_V2 - br.W1.shape[0], br.W1.shape[1]))])
    W2 = np.hstack([br.W2, np.zeros((br.W2.shape[0], N_OUT_V2 - br.W2.shape[1]))])
    b2 = np.concatenate([br.b2, np.asarray(want_bias, float)])
    return BrainGenome(W1, br.b1.copy(), W2, b2)


def mutate_brain(br: BrainGenome, rng, cfg: MutationConfig) -> BrainGenome:
    out = br.copy()
    for name in ("W1", "b1", "W2", "b2"):
        w = getattr(out, name)
        m = rng.random(w.shape) < cfg.brain_rate
        w[m] += rng.normal(0, cfg.brain_sigma, m.sum())
        r = rng.random(w.shape) < cfg.brain_reset
        w[r] = rng.normal(0, 0.5, r.sum())
    return out


def crossover_brain(a: BrainGenome, b: BrainGenome, rng) -> BrainGenome:
    """Per neuron: the incoming/outgoing weights of each hidden neuron come as a whole from A or B."""
    c = a.copy()
    assert a.W1.shape == b.W1.shape and a.W2.shape == b.W2.shape, "brain versions do not match"
    pick = rng.random(HIDDEN) < 0.5
    c.W1[:, pick] = b.W1[:, pick]
    c.b1[pick] = b.b1[pick]
    c.W2[pick, :] = b.W2[pick, :]
    mo = rng.random(a.b2.shape[0]) < 0.5
    c.b2[mo] = b.b2[mo]
    return c


# --------------------------------------------------------------------------- #
#  Genome = body + brain
# --------------------------------------------------------------------------- #
@dataclass
class Genome:
    body: BodyGenome
    brain: BrainGenome
    meta: dict = field(default_factory=dict)   # fitness results, generation, settings, etc.

    def copy(self):
        return Genome(self.body.copy(), self.brain.copy(), json.loads(json.dumps(self.meta)))

    def to_dict(self):
        return {"version": 1, "meta": self.meta, "body": self.body.to_dict(), "brain": self.brain.to_dict()}

    @staticmethod
    def from_dict(d):
        assert d.get("version") == 1
        return Genome(BodyGenome.from_dict(d["body"]), BrainGenome.from_dict(d["brain"]), d.get("meta", {}))


def random_genome(rng, brain_v2: bool = False, **kw) -> Genome:
    br = random_brain(rng, N_IN_V2, N_OUT_V2) if brain_v2 else random_brain(rng)
    return Genome(random_body(rng, **kw), br)


def mutate(g: Genome, rng, cfg: MutationConfig = None, freeze_body: bool = False) -> Genome:
    """freeze_body=True: the body stays exactly the same, only the brain is mutated."""
    cfg = cfg or MutationConfig()
    body = g.body.copy() if freeze_body else mutate_body(g.body, rng, cfg)
    return Genome(body, mutate_brain(g.brain, rng, cfg))


def crossover(a: Genome, b: Genome, rng, freeze_body: bool = False) -> Genome:
    body = a.body.copy() if freeze_body else crossover_body(a.body, b.body, rng)
    return Genome(body, crossover_brain(a.brain, b.brain, rng))


def save_json(g: Genome, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g.to_dict(), f, indent=1, ensure_ascii=False)


def load_json(path: str) -> Genome:
    with open(path, encoding="utf-8") as f:
        return Genome.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
#  Genome -> physics body
# --------------------------------------------------------------------------- #
def body_spec(b: BodyGenome) -> dict:
    n = b.n
    V = lambda f, zero_head=True: [0.0 if (zero_head and i == 0) else level_value(f, int(getattr(b, f)[i]))
                                   for i in range(n + 1)]
    mass = [level_value("mass", int(b.mass[i])) for i in range(n + 1)]
    return dict(parent=[int(p) for p in b.parent], length=[1.0] + V("length")[1:], width=V("width"),
                legacy_spine=bool(b.legacy_spine), mass=mass, theta0=V("theta"), jlo=[0.0] + [-level_value("lo", int(k)) for k in b.lo[1:]],
                jhi=V("hi"), tau_max=V("tau"), jdamp=V("damp"))


def build_bodies(genomes: list[Genome], n_points: int = None):
    """If n_points is not given, padding is based on the largest creature in the batch (physics is faster)."""
    if n_points is None:
        n_points = max(g.body.n for g in genomes) + 2     # + nose point
    return make_bodies([body_spec(g.body) for g in genomes], n_points=n_points)

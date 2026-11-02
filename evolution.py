"""
Phase 3 - Evolution loop: fitness, selection, mutation/crossover, multi-core evaluation.

Evaluation
----------
Each individual is simulated under K different initial conditions (random body angle + target direction sequence).
Population x trials runs as a single numpy batch; the population is additionally split across worker processes.
In half of each trial the target direction CHANGES (the ability to change direction is measured).

Fitness (per trial, the sum of the terms below):
    + progress in the target direction   (projection of the COM displacement onto the target / duration)   <- the main reward
    - lateral deviation                  (displacement perpendicular to the target)                        w_lat
    - energy                             (mean command^2)                                                   w_energy
    - physical instability               (mean body rotation rate)                                          w_spin
    - jitter                             (change of commands between steps)                                 w_jitter
Spinning in place / jitter: since the COM displacement is ~0 there is no progress reward, but there are penalties.
Individual fitness = 0.7 * mean + 0.3 * worst trial (consistency across different conditions).

Curriculum: difficulty increases with "stage" (once every stage_len generations):
    0: target = front of the body   1: 4 directions (front/back/right/left)   2: 8 directions + direction change mid-trial
    3: stage 2 + "NO COMMAND" segments (target vector 0): the creature learns to stop.
       Reward in a stop segment = -stop_scale*(mean speed) -> standing perfectly still scores highest.
       (with stop_scale=1 the population collapsed to 'never moving at all': swimming 0.65*+0.3 -0.35*0.3 ~ 0.)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass, asdict, field, replace

import numpy as np

from genome import (Genome, MutationConfig, build_bodies, crossover, mutate, mutate_body, mutate_brain,
                    random_brain, random_genome, save_json)
from brain import stack_brains, act
from physics import (PHYSICS_VERSION, PhysicsParams, center_of_mass, constraint_error, init_state, motor_mask, step)


# --------------------------------------------------------------------------- #
#  Settings
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    duration: float = 8.0       # seconds
    trials: int = 4             # different initial conditions per individual
    warmup: float = 1.0         # the first second is not included in the measurement
    grace: float = 1.0          # time after a direction change that is not included in the measurement
    control_dt: float = 0.02    # brain control period (s) -> 50 Hz; independent of the physics step
    w_lat: float = 0.2          # penalty for deviation perpendicular to the target
    stop_scale: float = 0.5     # speed penalty multiplier in a 'no command' segment (1.0 -> stopping, wiped out swimming entirely)
    neg_scale: float = 0.25     # penalty for progress OPPOSITE to the target = softened by this multiplier (so that standing still is not better than swimming)
    w_energy: float = 0.02
    w_spin: float = 0.01
    w_jitter: float = 0.03
    worst_weight: float = 0.15  # fitness = (1-w)*mean + w*worst
    bad_fitness: float = -5.0   # exploded/NaN individuals
    gate_penalties: bool = True  # energy/spin/jitter penalties reach full strength when the creature really progresses
    gate_floor: float = 0.25     # fraction of the penalties that remains for a non-progressing creature (so the penalty alone doesn't select for 'staying still')
    gate_speed: float = 0.3      # penalties apply in full at this progress speed (m/s)


@dataclass
class EvoConfig:
    pop_size: int = 64
    elite: int = 3
    tournament: int = 3
    p_crossover: float = 0.4
    immigrants: float = 0.04    # fraction of new random individuals each generation
    stage_len: int = 25         # generations per curriculum stage
    init_limbs: tuple = (4, 10)
    max_stage: int = 3
    curriculum: str = "adaptive"   # adaptive: difficulty rises/falls with success | fixed: once every stage_len generations
    adv_fitness: float = 0.15       # if the champion exceeds this (adv_streak generations in a row), difficulty +1
    adv_streak: int = 3
    adv_min_gens: int = 5           # stay at least this many generations at a difficulty
    regress_fitness: float = 0.03   # if the champion stays below this for regress_gens generations, difficulty -1
    regress_gens: int = 25
    freeze_body: bool = False       # True: body is fixed, only the brain evolves
    p_stop: float = 0.35        # probability that each segment is 'stop' in stage 3
    legacy_spine: bool = False   # True: random/immigrant individuals are born in the OLD spine mode (default: body direction = head)
    stop_command: bool = False  # 'no command = stop' (notch 10 / stage 3) is OFF for now; True -> old behavior
    stall_boost: bool = True    # when a stall is detected mutation gets stronger, immigrants get more aggressive
    stall_gens: int = 15        # count as 'stalled' if the champion's moving average does not improve for this many generations
    stall_window: int = 5       # champion fitness is tracked with an average over this many generations (for noise)
    stall_eps: float = 0.01     # improvement smaller than this does not count as 'improvement'
    boost_max: float = 3.0      # maximum multiplier of the mutation size
    boost_step_gens: int = 10   # while the stall continues, the multiplier rises by +1 every this many generations


# --------------------------------------------------------------------------- #
#  Trials and evaluation
# --------------------------------------------------------------------------- #
NOTCH_MAX = 10   # 0: front only | 1..8: direction range notch*22.5° | 9: 8 directions + direction change | 10: + 'stop' command


def notch_to_stage(notch: int) -> int:
    return 0 if notch <= 0 else 1 if notch <= 8 else 2 if notch == 9 else 3


def make_trials(rng: np.random.Generator, K: int, stage: int, p_stop: float = 0.35, dir_range=None) -> dict:
    """dir_range (rad): in stage 1 the target direction deviates from the body's front by at most this much (uniform ±dir_range).
    In each generation one trial is at the boundary (±dir_range) -> pushes the training frontier. None -> old 4 directions."""
    h = rng.uniform(-np.pi, np.pi, K)
    if stage <= 0:
        o1 = np.zeros(K)
        o2 = o1.copy()
    elif stage == 1:
        if dir_range is None:
            o1 = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2], K)
        else:
            o1 = rng.uniform(-dir_range, dir_range, K)
            o1[0] = dir_range * rng.choice([-1.0, 1.0])
        o2 = o1.copy()
    else:
        o1 = rng.integers(0, 8, K) * np.pi / 4
        o2 = rng.integers(0, 8, K) * np.pi / 4
    mag1, mag2 = np.ones(K), np.ones(K)
    if stage >= 3:
        mag1 = (rng.random(K) >= p_stop).astype(float)
        mag2 = (rng.random(K) >= p_stop).astype(float)
        if K >= 2:                       # guarantee at least one 'stop' segment
            mag2[0] = 0.0
            mag1[0] = 1.0
    return dict(heading=h, tgt1=h + o1, tgt2=h + o2, mag1=mag1, mag2=mag2)


def evaluate_batch(genomes: list[Genome], trials: dict, cfg: EvalConfig,
                   pp: PhysicsParams = None) -> dict:
    """Returns: fitness (P,), per_trial (P,K), and the measurement components."""
    pp = pp or PhysicsParams()
    P, K = len(genomes), len(trials["heading"])
    rep = [g for g in genomes for _ in range(K)]
    bd = build_bodies(rep)
    bb = stack_brains([g.brain for g in rep])
    heading = np.tile(trials["heading"], P)
    tg1, tg2 = np.tile(trials["tgt1"], P), np.tile(trials["tgt2"], P)
    mg1 = np.tile(trials.get("mag1", np.ones(K)), P)
    mg2 = np.tile(trials.get("mag2", np.ones(K)), P)
    st = init_state(bd, heading=heading)
    B = bd.B
    steps = int(round(cfg.duration / pp.dt))
    sw = steps // 2
    w0 = int(round(cfg.warmup / pp.dt))
    g1 = sw + int(round(cfg.grace / pp.dt))
    assert w0 < sw < g1 < steps, "duration/warmup/grace incompatible"
    every = max(1, int(round(cfg.control_dt / pp.dt)))
    mm = motor_mask(bd)
    n_mot = np.maximum(mm.sum(1), 1)

    ctrl = np.zeros((bd.B, bd.N))
    energy = np.zeros(B)
    jitter = np.zeros(B)
    spin = np.zeros(B)
    n_ctrl = 0
    snap = {}
    with np.errstate(all="ignore"):
        for s in range(steps):
            if s % every == 0:
                new = act(st, bd, bb, np.where(s < sw, tg1, tg2), np.where(s < sw, mg1, mg2))
                jitter += np.abs(new - ctrl).sum(1) / n_mot
                ctrl = new
                energy += (ctrl ** 2).sum(1) / n_mot
                n_ctrl += 1
            step(st, bd, ctrl, pp)
            spin += np.abs(st.omega)
            if s + 1 in (w0, sw, g1, steps):
                snap[s + 1] = center_of_mass(st, bd)
        err = constraint_error(st, bd)
    energy /= n_ctrl
    jitter /= n_ctrl
    spin /= steps

    def seg(a, b, tgt, mag):
        T = (b - a) * pp.dt
        d = (snap[b] - snap[a]) / T
        u = np.stack([np.cos(tgt), np.sin(tgt)], -1)
        prog = np.sum(d * u, -1)
        lat = np.abs(d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0])
        prog = np.where(prog > 0, prog, cfg.neg_scale * prog)   # going in the opposite direction has a softened penalty
        stop = mag < 0.5                          # no command: standing still is rewarded
        prog = np.where(stop, -cfg.stop_scale * np.linalg.norm(d, axis=-1), prog)
        lat = np.where(stop, 0.0, lat)
        return prog, lat

    p1, l1 = seg(w0, sw, tg1, mg1)
    p2, l2 = seg(g1, steps, tg2, mg2)
    prog, lat = 0.5 * (p1 + p2), 0.5 * (l1 + l2)
    pen = cfg.w_energy * energy + cfg.w_spin * spin + cfg.w_jitter * jitter
    if cfg.gate_penalties:      # penalties are small for a non-progressing creature: 'staying still' must not become an attractor
        pen = pen * np.clip(cfg.gate_floor + (1 - cfg.gate_floor) * np.maximum(prog, 0) / cfg.gate_speed, cfg.gate_floor, 1.0)
    f = prog - cfg.w_lat * lat - pen
    bad = ~np.isfinite(f) | (err > 0.5)
    f = np.where(bad, cfg.bad_fitness, f)
    f = np.maximum(f, cfg.bad_fitness)

    ft = f.reshape(P, K)
    fit = (1 - cfg.worst_weight) * ft.mean(1) + cfg.worst_weight * ft.min(1)

    def r(x):
        return np.nan_to_num(x).reshape(P, K)

    return dict(fitness=fit, per_trial=ft, progress=r(prog), lateral=r(lat), energy=r(energy),
                spin=r(spin), jitter=r(jitter))


def _eval_chunk(args):
    genomes, trials, cfg = args
    return evaluate_batch(genomes, trials, cfg)


# --------------------------------------------------------------------------- #
#  Evolution
# --------------------------------------------------------------------------- #
class Evolution:
    def __init__(self, evo: EvoConfig = None, ev: EvalConfig = None, mut: MutationConfig = None,
                 workers: int = 1, seed: int = 0, seed_genome: Genome = None):
        self.evo = evo or EvoConfig()
        if seed_genome is not None:        # when training from a design, the mode comes from the design
            self.evo = replace(self.evo, legacy_spine=bool(seed_genome.body.legacy_spine))
        self.ev = ev or EvalConfig()
        self.mut = mut or MutationConfig()
        self.workers = max(1, workers)
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.notch = 0              # adaptive curriculum: 0..NOTCH_MAX
        self._streak = 0            # for how many generations in a row the champion has exceeded the threshold
        self._low = 0               # for how many generations the champion has been below the regress threshold
        self._at_notch = 0          # generations spent at the current difficulty
        self._champ_hist = []       # recent champion fitnesses (stall tracking)
        self._hi = -np.inf          # the highest moving average seen
        self._stall = 0             # for how many generations there has been no improvement
        self.seed_genome = seed_genome
        lo, hi = self.evo.init_limbs
        P = self.evo.pop_size
        if seed_genome is None:
            self.population = [random_genome(self.rng, n_min=lo, n_max=hi, legacy_spine=self.evo.legacy_spine) for _ in range(P)]
        else:
            self.population = self._seeded_population(seed_genome)
        self.fitness = np.full(self.evo.pop_size, -np.inf)
        self.history = []          # per-generation summary
        self.best = None           # (fitness, Genome) best of all time (per stage)
        self._pool = None

    # --- evaluation ---
    @property
    def notch_max(self) -> int:
        return NOTCH_MAX if self.evo.stop_command else NOTCH_MAX - 1

    @property
    def stage(self) -> int:
        if self.evo.curriculum == "adaptive":
            return notch_to_stage(min(self.notch, self.notch_max))
        top = self.evo.max_stage if self.evo.stop_command else min(self.evo.max_stage, 2)
        return min(top, self.gen // self.evo.stage_len)

    def _get_pool(self):
        if self._pool is None and self.workers > 1:
            self._pool = mp.Pool(self.workers)
        return self._pool

    def set_workers(self, n: int):
        """Change the number of workers (to increase it in turbo mode)."""
        n = max(1, int(n))
        if n != self.workers:
            self.close()
            self.workers = n

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def evaluate(self, trials: dict) -> dict:
        P = len(self.population)
        if self.workers == 1:
            return evaluate_batch(self.population, trials, self.ev)
        # sort by size, distribute to workers (similar padding size, balanced load)
        order = np.argsort([g.body.n for g in self.population])
        groups = [order[i::self.workers] for i in range(self.workers)]
        res = self._get_pool().map(_eval_chunk, [([self.population[i] for i in g], trials, self.ev) for g in groups])
        out = {}
        for key in res[0]:
            arr = np.zeros((P,) + res[0][key].shape[1:])
            for g, r in zip(groups, res):
                arr[g] = r[key]
            out[key] = arr
        return out

    def _seeded_population(self, sg: Genome) -> list:
        """Initial population from a design. If the body is fixed, all share the same body with different brains.
        Otherwise half have the same body, half a slightly mutated body. The design's own brain (if any) is individual 1."""
        P = self.evo.pop_size
        pop = [sg.copy()]
        for k in range(1, P):
            body = sg.body.copy()
            if not self.evo.freeze_body and k >= P // 2:
                body = mutate_body(body, self.rng, self.mut)
            if k <= max(1, P // 8):
                brain = mutate_brain(sg.brain, self.rng, self.mut)       # derivatives of the design's brain
            else:
                brain = random_brain(self.rng)
            pop.append(Genome(body, brain))
        for g in pop:
            g.meta = {"seeded": True, "physics": PHYSICS_VERSION}
        return pop

    # --- adaptive curriculum ---
    @property
    def dir_range(self):
        """Direction range (rad) valid in stage 1; None in other stages."""
        if self.evo.curriculum != "adaptive" or notch_to_stage(self.notch) != 1:
            return None
        return min(self.notch, 8) * np.pi / 8

    def _update_curriculum(self, champ_fitness: float):
        """Increase difficulty if the champion is good enough; step back one notch if it is very poor for a long time."""
        if self.evo.curriculum != "adaptive":
            return
        self._at_notch += 1
        self._streak = self._streak + 1 if champ_fitness >= self.evo.adv_fitness else 0
        self._low = self._low + 1 if champ_fitness < self.evo.regress_fitness else 0
        if self._streak >= self.evo.adv_streak and self._at_notch >= self.evo.adv_min_gens and self.notch < self.notch_max:
            self.notch += 1
            self._streak = self._low = self._at_notch = 0
            self._reset_stall()
        elif self._low >= self.evo.regress_gens and self.notch > 0:
            self.notch -= 1
            self._streak = self._low = self._at_notch = 0
            self._reset_stall()

    # --- stall detection and mutation boosting ---
    def _reset_stall(self):
        """When the difficulty changes the fitness scale changes: the previous comparison is invalid."""
        self._champ_hist, self._hi, self._stall = [], -np.inf, 0

    def _track_stall(self, champ_fitness: float):
        w = self.evo.stall_window
        self._champ_hist = (self._champ_hist + [champ_fitness])[-w:]
        if len(self._champ_hist) < w:
            return
        r = float(np.mean(self._champ_hist))
        if r > self._hi + self.evo.stall_eps:
            self._hi, self._stall = r, 0
        else:
            self._stall += 1

    @property
    def mut_scale(self) -> float:
        if not self.evo.stall_boost or self._stall < self.evo.stall_gens:
            return 1.0
        return min(self.evo.boost_max, 1.5 + (self._stall - self.evo.stall_gens) / self.evo.boost_step_gens)

    def _scaled_mut(self, k: float) -> MutationConfig:
        if k <= 1.0:
            return self.mut
        m = self.mut
        r = min(k, 2.0)          # increase probabilities by at most 2x, step size by k x
        return replace(m, p_gene=min(0.5, m.p_gene * r), p_add=min(0.3, m.p_add * r), p_remove=min(0.3, m.p_remove * r),
                       p_mirror=min(0.3, m.p_mirror * r), brain_rate=min(0.5, m.brain_rate * r), brain_sigma=m.brain_sigma * k)

    # --- selection and reproduction ---
    def _tournament(self) -> Genome:
        idx = self.rng.integers(0, len(self.population), self.evo.tournament)
        return self.population[idx[np.argmax(self.fitness[idx])]]

    def _breed(self):
        P = self.evo.pop_size
        order = np.argsort(-self.fitness)
        nxt = [self.population[i].copy() for i in order[: self.evo.elite]]
        k = self.mut_scale
        mut = self._scaled_mut(k)
        n_imm = int(round((max(self.evo.immigrants, 0.10) if k > 1 else self.evo.immigrants) * P))
        lo, hi = self.evo.init_limbs
        while len(nxt) < P - n_imm:
            a = self._tournament()
            if self.rng.random() < self.evo.p_crossover:
                child = crossover(a, self._tournament(), self.rng, self.evo.freeze_body)
            else:
                child = a.copy()
            nxt.append(mutate(child, self.rng, mut, self.evo.freeze_body))
        heavy = self._scaled_mut(self.evo.boost_max)
        while len(nxt) < P:
            if k > 1 and len(nxt) % 2 == 0:       # when stalled: half of the immigrants are HEAVILY mutated copies of the elite
                src = self.population[order[int(self.rng.integers(0, self.evo.elite))]].copy()
                nxt.append(mutate(src, self.rng, heavy, self.evo.freeze_body))
            elif self.seed_genome is not None and self.evo.freeze_body:
                nxt.append(Genome(self.seed_genome.body.copy(), random_brain(self.rng)))
            else:
                nxt.append(random_genome(self.rng, n_min=lo, n_max=hi, legacy_spine=self.evo.legacy_spine))
        self.population = nxt

    def step_generation(self) -> dict:
        """One generation: evaluate -> record -> produce the next generation. Returns the summary dictionary."""
        t0 = time.time()
        trials = make_trials(self.rng, self.ev.trials, self.stage, self.evo.p_stop, self.dir_range)
        res = self.evaluate(trials)
        self.fitness = res["fitness"]
        i = int(np.argmax(self.fitness))
        self._track_stall(float(self.fitness[i]))
        champ = self.population[i].copy()
        champ.meta = {"generation": self.gen, "stage": self.stage, "fitness": float(self.fitness[i]),
                      "progress": float(res["progress"][i].mean()), "energy": float(res["energy"][i].mean()),
                      "n_limbs": champ.body.n, "physics": PHYSICS_VERSION}
        dr = self.dir_range
        summary = dict(gen=self.gen, stage=self.stage, notch=self.notch, notch_max=self.notch_max,
                       stall=self._stall, mut_scale=self.mut_scale,
                       dir_range_deg=None if dr is None else float(np.degrees(dr)), best=float(self.fitness[i]),
                       mean=float(self.fitness.mean()), median=float(np.median(self.fitness)),
                       best_progress=float(res["progress"][i].mean()),
                       best_limbs=champ.body.n, mean_limbs=float(np.mean([g.body.n for g in self.population])),
                       seconds=time.time() - t0, champion=champ)
        self.history.append({k: v for k, v in summary.items() if k != "champion"})
        self.last_result = res
        self._update_curriculum(float(self.fitness[i]))
        self._breed()
        self.gen += 1
        return summary

    # --- saving ---
    def save_checkpoint(self, path: str):
        """For resuming training (internal format). Use save_json for human-readable records."""
        with open(path, "wb") as f:
            pickle.dump(dict(gen=self.gen, population=self.population, fitness=self.fitness,
                             history=self.history, physics=PHYSICS_VERSION, notch=self.notch, evo=asdict(self.evo), ev=asdict(self.ev),
                             rng=self.rng.bit_generator.state), f)

    def load_checkpoint(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.gen, self.population, self.fitness, self.history = d["gen"], d["population"], d["fitness"], d["history"]
        self.notch = min(d.get("notch", 0), self.notch_max)
        self.loaded_physics = d.get("physics", 1)   # if trained with an old version the population will have to re-adapt
        self.rng.bit_generator.state = d["rng"]

"""
Physics validity check: does a creature's performance depend on the time step?

A real swimmer's speed should not change when dt is halved. If the speed drops substantially, the creature is
exploiting energy from the physics engine's numerical error (e.g. light limb + very strong motor).

    python validate.py runs/app1/best.json runs/app1/gen_0800.json
    python validate.py x.json --dt 0.01        # compare against the old/coarse step
"""
import argparse

import numpy as np

from evolution import EvalConfig, evaluate_batch, make_trials
from genome import load_json
from physics import PhysicsParams

OK_RATIO = 0.7


def convergence(genome, trials: int = 3, stage: int = 2, seed: int = 0, base_dt: float = None) -> dict:
    """Runs the same trials with dt and dt/2. ratio = fine_progress / coarse_progress (1 = consistent)."""
    base_dt = base_dt or PhysicsParams().dt
    tr = make_trials(np.random.default_rng(seed), trials, stage)
    cfg = EvalConfig(trials=trials)
    coarse = evaluate_batch([genome], tr, cfg, PhysicsParams(dt=base_dt))["progress"][0].mean()
    fine = evaluate_batch([genome], tr, cfg, PhysicsParams(dt=base_dt / 2))["progress"][0].mean()
    ratio = float(fine / coarse) if coarse > 0.05 else 1.0     # a creature that barely progresses has nothing to exploit
    return dict(coarse=float(coarse), fine=float(fine), ratio=ratio, ok=ratio >= OK_RATIO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dt", type=float, default=None)
    a = ap.parse_args()
    for f in a.files:
        r = convergence(load_json(f), base_dt=a.dt)
        flag = "CONSISTENT" if r["ok"] else "SUSPICIOUS (may be exploiting numerical error)"
        print(f"{f}: {r['coarse']:+.2f} m/s at dt, {r['fine']:+.2f} m/s at dt/2, ratio {r['ratio']:.2f} -> {flag}")


if __name__ == "__main__":
    main()

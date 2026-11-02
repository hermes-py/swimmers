"""
Headless (no graphics) training command.

Examples:
    python train.py --out runs/experiment1 --pop 64 --gens 200 --workers 4
    python train.py --out runs/experiment1 --resume          # continue where it left off
Outputs (in the --out folder):
    best.json              best individual so far (human-readable; can be loaded with app.py)
    gen_0050.json ...      champion of that generation, saved every --save-every generations
    checkpoint.pkl         the whole population for resuming training (internal format)
    log.csv                per-generation summary
Can be stopped safely with Ctrl+C.
(For graphical use: python app.py)
"""
import argparse
import os

from genome import load_json
from trainer import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--gens", type=int, default=200, help="0 = unlimited")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--stage-len", type=int, default=25, help="generations per curriculum stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed-from", default=None, help="create the initial population from a design/creature JSON")
    ap.add_argument("--freeze-body", action="store_true", help="body is fixed, only the brain evolves (with --seed-from)")
    ap.add_argument("--min-limbs", type=int, default=4, help="minimum number of limbs mutation may reduce to")
    ap.add_argument("--stop-command", action="store_true", help="enable the 'no command = stop' notch (default OFF)")
    ap.add_argument("--legacy-spine", action="store_true", help="OLD spine mode: bone 1 is passive and determines the body direction (default: body direction = head direction)")
    ap.add_argument("--no-stall-boost", action="store_true", help="disable strengthening mutation when a stall is detected")
    ap.add_argument("--curriculum", choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--validate-every", type=int, default=10, help="physics validation of the champion (0 = off)")
    a = ap.parse_args()
    run_training(out=a.out, pop=a.pop, gens=a.gens, trials=a.trials, duration=a.duration, workers=a.workers,
                 stage_len=a.stage_len, seed=a.seed, save_every=a.save_every, resume=a.resume,
                 validate_every=a.validate_every, freeze_body=a.freeze_body, min_limbs=a.min_limbs,
                 curriculum=a.curriculum, stop_command=a.stop_command, stall_boost=not a.no_stall_boost, legacy_spine=a.legacy_spine,
                 seed_genome=load_json(a.seed_from) if a.seed_from else None)


if __name__ == "__main__":
    main()

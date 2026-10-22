"""Bir canlının 8 hedef yöndeki (gövde önüne göre) gerçek performansı + fizik doğrulaması.

    python eval_dirs.py runs/x/best.json runs/x/gen_0150.json
"""
import sys

import numpy as np

from evolution import EvalConfig, evaluate_batch
from genome import load_json
from validate import convergence


def main(paths):
    for path in paths:
        g = load_json(path)
        meta = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.meta.items()
                if k in ("generation", "stage", "fitness", "progress")}
        print(path, "| uzuv", g.body.n, "|", meta)
        cfg = EvalConfig(trials=8)
        h = np.linspace(-3, 3, 8)
        row = []
        for off_deg in (0, 45, 90, 135, 180, -135, -90, -45):
            o = np.radians(off_deg)
            tr = dict(heading=h, tgt1=h + o, tgt2=h + o, mag1=np.ones(8), mag2=np.ones(8))
            r = evaluate_batch([g], tr, cfg)
            row.append((off_deg, float(r["progress"][0].mean()), float(r["lateral"][0].mean())))
        print("   yön(°): ilerleme m/s (yanal)  ->  " + "  ".join(f"{d:+4d}:{p:+.2f}({l:.2f})" for d, p, l in row))
        c = convergence(g, trials=4, stage=2, seed=3)
        print(f"   ortalama ilerleme {np.mean([p for _, p, _ in row]):+.3f} | fizik doğrulama oranı {c['ratio']:.2f} "
              f"({'tutarlı' if c['ok'] else 'ŞÜPHELİ'})")


if __name__ == "__main__":
    main(sys.argv[1:])

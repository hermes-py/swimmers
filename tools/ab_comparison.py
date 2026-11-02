import sys, json, time, numpy as np
from evolution import *
variant, seed, gens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
new = variant == "new"
ev = Evolution(EvoConfig(pop_size=32, stall_boost=new), EvalConfig(trials=3, duration=5.0, gate_penalties=new), seed=seed)
rows = []
for _ in range(gens):
    s = ev.step_generation(); c = s.pop("champion")
    rows.append((s["gen"], s["notch"], s["best"], s["best_progress"], s["stall"], s["mut_scale"]))
# common, fair measurement: champion + the population's top 3, progress with fixed trials (independent of the penalty gate)
cfg = EvalConfig(trials=8, gate_penalties=False)
h = np.linspace(-3, 3, 8)
res = {}
for name, off in (("forward", np.zeros(8)), ("8dir", np.arange(8) * np.pi / 4)):
    tr = dict(heading=h, tgt1=h + off, tgt2=h + off, mag1=np.ones(8), mag2=np.ones(8))
    r = evaluate_batch(ev.population, tr, cfg)
    res[name] = float(np.sort(r["progress"].mean(1))[-3:].mean())
print(json.dumps(dict(variant=variant, seed=seed, final_notch=ev.notch, max_notch=max(r[1] for r in rows),
      eval_forward_top3=res["forward"], eval_8dir_top3=res["8dir"],
      boosted_gens=sum(r[5] > 1 for r in rows), limbs=float(np.mean([g.body.n for g in ev.population])))), flush=True)

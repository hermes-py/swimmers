import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from evolution import EvalConfig, evaluate_batch, make_trials
from genome import load_json
out = {}
for name in ("jellyfish", "snake", "example_symmetric_fish"):
    g = load_json(f"{sys.argv[1]}/examples/{name}.json")
    for stage in (0, 2):
        tr = make_trials(np.random.default_rng(7), 4, stage)
        r = evaluate_batch([g], tr, EvalConfig(trials=4, duration=6.0, warmup=0.5, grace=0.5))
        out[f"{name}-s{stage}"] = [float(x) for x in r["progress"][0]]
print(json.dumps(out))

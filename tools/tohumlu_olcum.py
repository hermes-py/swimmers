import sys, time, json, os, numpy as np
sys.path.insert(0,"/home/claude/sw")
from world import *
name=sys.argv[1]; kw=json.loads(sys.argv[2]); T=float(sys.argv[3]); seed=int(sys.argv[4])
cfg=WorldConfig(seed=seed,**kw)
paths=[f"/home/claude/sw/ornekler/{f}.json" for f in ("denizanasi","yilan","ornek_simetrik_balik")]
w=World(cfg, genomes=seeded_genomes(paths,cfg.n_init,np.random.default_rng(seed+12345),cfg.mut)); t0=time.time(); nxt=50
print(name,flush=True)
while w.t<T and w.pop>0:
    w.advance(1.0)
    if w.t>=nxt:
        s=w.stats(); nxt+=50
        print(f"t={s['t']:5.0f} pop {s['pop']:3d} yenen {s['eaten']:4d} doğum {s['births_asex']}+{s['births_sex']} nesil {s['mean_gen']:.1f}/{s['max_gen']} hız {s['mean_speed']:.2f} enerji {s['mean_energy']:.0f} uzuv {s['mean_limbs']:.1f} ({time.time()-t0:.0f}s)",flush=True)

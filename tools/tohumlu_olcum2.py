import sys, time, json, numpy as np
sys.path.insert(0,"/home/claude/sw")
from world import *
name=sys.argv[1]; kw=json.loads(sys.argv[2]); T=float(sys.argv[3]); seed=int(sys.argv[4])
cfg=WorldConfig(seed=seed,**kw)
paths=[f"/home/claude/sw/ornekler/{f}.json" for f in ("denizanasi","yilan","ornek_simetrik_balik")]
w=World(cfg, genomes=seeded_genomes(paths,cfg.n_init,np.random.default_rng(seed+12345),cfg.mut,carnivore_frac=cfg.carnivore_frac)); t0=time.time(); nxt=100
print(name,flush=True)
while w.t<T and w.pop>0:
    w.advance(1.0)
    if w.t>=nxt:
        s=w.stats(); nxt+=100
        lc=np.bincount(w.lin,minlength=3)[:3]
        print(f"t={s['t']:5.0f} pop {s['pop']:3d} (etçil {s['carn']:3d}) enerji {s['mean_energy']:4.0f} yenen {s['eaten']:4d} öldürme {s['kills']:4d} doğum {s['births_asex']}+{s['births_sex']} nesil {s['mean_gen']:.0f} hız {s['mean_speed']:.2f} uzuv {s['mean_limbs']:.1f} diet {s['mean_diet']:.2f} soy {lc.tolist()} ({time.time()-t0:.0f}s)",flush=True)

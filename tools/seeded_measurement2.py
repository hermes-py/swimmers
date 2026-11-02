import sys, time, json, numpy as np
sys.path.insert(0,"/home/claude/sw")
from world import *
name=sys.argv[1]; kw=json.loads(sys.argv[2]); T=float(sys.argv[3]); seed=int(sys.argv[4])
cfg=WorldConfig(seed=seed,**kw)
paths=[f"/home/claude/sw/examples/{f}.json" for f in ("jellyfish","snake","example_symmetric_fish")]
w=World(cfg, genomes=seeded_genomes(paths,cfg.n_init,np.random.default_rng(seed+12345),cfg.mut,carnivore_frac=cfg.carnivore_frac)); t0=time.time(); nxt=100
print(name,flush=True)
while w.t<T and w.pop>0:
    w.advance(1.0)
    if w.t>=nxt:
        s=w.stats(); nxt+=100
        lc=np.bincount(w.lin,minlength=3)[:3]
        print(f"t={s['t']:5.0f} pop {s['pop']:3d} (carnivore {s['carn']:3d}) energy {s['mean_energy']:4.0f} eaten {s['eaten']:4d} kills {s['kills']:4d} births {s['births_asex']}+{s['births_sex']} gen {s['mean_gen']:.0f} speed {s['mean_speed']:.2f} limbs {s['mean_limbs']:.1f} diet {s['mean_diet']:.2f} lineage {lc.tolist()} ({time.time()-t0:.0f}s)",flush=True)

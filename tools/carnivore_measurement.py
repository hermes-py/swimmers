import sys, time, json, numpy as np
sys.path.insert(0,"/home/claude/sw")
from world import *
name=sys.argv[1]; kw=json.loads(sys.argv[2]); T=float(sys.argv[3]); seed=int(sys.argv[4])
cfg=WorldConfig(seed=seed,n_init=60,**kw)
paths=[f"/home/claude/sw/examples/{f}.json" for f in ("jellyfish","snake","example_symmetric_fish")]
w=World(cfg, genomes=seeded_genomes(paths,cfg.n_init,np.random.default_rng(seed+12345),cfg.mut,carnivore_frac=cfg.carnivore_frac)); t0=time.time(); nxt=50
print(name,flush=True)
k0=0
while w.t<T and w.pop>0:
    w.advance(1.0)
    if w.t>=nxt:
        s=w.stats(); nxt+=50
        c=w.diet>0.5; h=~c
        ce=w.energy[c].mean() if c.any() else 0; he=w.energy[h].mean() if h.any() else 0
        lc=np.bincount(w.lin,minlength=3)[:3].tolist()
        print(f"t={s['t']:4.0f} pop {s['pop']:3d} carnivore {int(c.sum()):2d} (E {ce:3.0f}) herbivore {int(h.sum()):3d} (E {he:3.0f}) kills +{s['kills']-k0:2d} bites {s['bites']} lineage {lc} ({time.time()-t0:.0f}s)",flush=True)
        k0=s['kills']

import sys, time, json, numpy as np
sys.path.insert(0,"/home/claude/sw")
from world import *
name=sys.argv[1]; kw=json.loads(sys.argv[2]); T=float(sys.argv[3]); seed=int(sys.argv[4])
cfg=WorldConfig(seed=seed,carnivore_frac=0.0,**kw)
paths=[f"/home/claude/sw/examples/{f}.json" for f in ("jellyfish","snake","example_symmetric_fish")]
w=World(cfg, genomes=seeded_genomes(paths,cfg.n_init,np.random.default_rng(seed+12345),cfg.mut,carnivore_frac=0.0)); t0=time.time(); nxt=100
print(name,flush=True)
def lin_report():
    out=[]
    sp=np.linalg.norm(com_velocity(w.st,w.bd),axis=1)
    for l,n in zip(range(3),("jelly","snake","fish")):
        m=w.lin==l
        out.append(f"{n} {int(m.sum()):3d} (speed {sp[m].mean() if m.any() else 0:.2f}, E {w.energy[m].mean() if m.any() else 0:.0f})")
    return " | ".join(out)
while w.t<T and w.pop>0:
    w.advance(1.0)
    if w.t>=nxt:
        s=w.stats(); nxt+=100
        print(f"t={s['t']:4.0f} pop {s['pop']:3d} eaten {s['eaten']:4d} gen {s['mean_gen']:.0f} | {lin_report()} ({time.time()-t0:.0f}s)",flush=True)

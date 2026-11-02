import numpy as np, sys
sys.path.insert(0,"/home/claude/sw")
from genome import *; from brain import *; from physics import *
def run(legacy, seed=11, n=48, steps=1000):
    rng=np.random.default_rng(seed); gs=[]
    for _ in range(n):
        g=random_genome(rng,n_min=4,n_max=24,legacy_spine=legacy); b=g.body
        b.tau[:]=n_levels("tau")-1; b.mass[0]=0; b.mass[1:]=0; b.length[1:]=0
        for i in range(2,b.n+1): b.parent[i]=0 if rng.random()<0.6 else b.parent[i]
        b.validate(); gs.append(g)
    bd=build_bodies(gs); bb=stack_brains([g.brain for g in gs]); pp=PhysicsParams()
    st=init_state(bd,heading=rng.uniform(-3,3,n))
    with np.errstate(all="ignore"):
        for k in range(steps): step(st,bd,act(st,bd,bb,np.full(n,.5)),pp)
    bad=~np.isfinite(st.x).all(axis=(1,2)) | (constraint_error(st,bd)>0.5)
    return int(bad.sum()), n, float(np.nanmedian(constraint_error(st,bd)))
print("OLD mode: exploded/broken", run(True))
print("NEW mode: exploded/broken", run(False))

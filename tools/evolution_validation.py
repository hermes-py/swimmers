import sys, json, math, numpy as np
sys.path.insert(0,"/home/claude/sw")
from evolution import *; from validate import convergence
from genome import build_bodies; from brain import act, stack_brains; from physics import *
seed=int(sys.argv[1]); gens=int(sys.argv[2])
ev=Evolution(EvoConfig(pop_size=32),EvalConfig(trials=3,duration=5.0),seed=seed)
mjs=[]
for g in range(gens):
    s=ev.step_generation()
def maxjv(genome,dt):
    bd=build_bodies([genome]); bb=stack_brains([genome.brain]); pp=PhysicsParams(dt=dt)
    st=init_state(bd,heading=[1.0]); every=max(1,int(round(0.02/dt))); c=np.zeros((1,bd.N)); m=0.0
    for k in range(int(8/dt)):
        if k%every==0: c=act(st,bd,bb,np.array([1.0]))
        step(st,bd,c,pp); m=max(m,float(np.abs(st.jvel).max()))
    return m
order=np.argsort(-ev.fitness)[:5]
rs=[convergence(ev.population[i],trials=4,stage=min(s["stage"],2),seed=1) for i in order]
print(json.dumps(dict(seed=seed,notch=ev.notch,best=round(float(ev.fitness[order[0]]),3),limbs=round(float(np.mean([g.body.n for g in ev.population])),1),
  validity=[round(r["ratio"],2) for r in rs],progress=[round(r["coarse"],2) for r in rs],
  max_jvel_dt=[round(maxjv(ev.population[i],0.005)) for i in order],max_jvel_half=[round(maxjv(ev.population[i],0.0025)) for i in order])),flush=True)

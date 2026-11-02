"""
Training loop (no graphics). Used in two ways:
  * train.py (command line)    -> run_training(...) is called directly
  * app.py   (pygame UI)       -> runs in a separate process; talks via command/message queues

Commands (cmd_q): ("pause",) ("resume",) ("workers", n) ("stop",)
Messages (msg_q): ("gen", summary_dict, champion_dict) ("info", text) ("error", text) ("done",)
"""
import os
import queue
import time

from evolution import Evolution, EvoConfig, EvalConfig
from physics import PHYSICS_VERSION
from genome import MutationConfig, load_json, save_json
from validate import convergence

CSV_COLS = ("gen", "stage", "best", "mean", "median", "best_progress", "best_limbs", "mean_limbs", "seconds")


def run_training(out="runs/run1", pop=64, gens=0, trials=4, duration=8.0, workers=1, stage_len=25,
                 seed=0, save_every=25, resume=False, cmd_q=None, msg_q=None, verbose=True, validate_every=10,
                 seed_genome=None, freeze_body=False, min_limbs=4, curriculum="adaptive", stop_command=False, stall_boost=True, legacy_spine=False):
    """gens=0 -> unlimited (until stopped)."""
    def say(kind, *payload):
        if msg_q is not None:
            msg_q.put((kind,) + payload)

    try:
        os.makedirs(out, exist_ok=True)
        evo = Evolution(EvoConfig(pop_size=pop, stage_len=stage_len, freeze_body=freeze_body, curriculum=curriculum,
                                stop_command=stop_command, stall_boost=stall_boost, legacy_spine=legacy_spine),
                        EvalConfig(trials=trials, duration=duration), MutationConfig(min_limbs=min_limbs),
                        workers=workers, seed=seed, seed_genome=seed_genome)
        ck = os.path.join(out, "checkpoint.pkl")
        if resume and os.path.exists(ck):
            evo.load_checkpoint(ck)
            say("info", f"resuming from generation {evo.gen}")
            if getattr(evo, "loaded_physics", 2) != PHYSICS_VERSION:
                msg = ("WARNING: this checkpoint was trained with an old physics version; behaviors that exploited numerical "
                       "error will not work in the new physics, the population will have to re-adapt.")
                say("info", msg)
                if verbose:
                    print(msg)
            if verbose:
                print(f"resuming from generation {evo.gen}")
        log = open(os.path.join(out, "log.csv"), "a", encoding="utf-8")
        if log.tell() == 0:
            log.write(",".join(CSV_COLS) + "\n")
        best_by_stage = {}
        paused = False
        stop = False

        def handle(cmd):
            nonlocal paused, stop
            if cmd[0] == "pause":
                paused = True
            elif cmd[0] == "resume":
                paused = False
            elif cmd[0] == "workers":
                evo.set_workers(cmd[1])
            elif cmd[0] == "stop":
                stop = True

        try:
            while not stop and (gens == 0 or evo.gen < gens):
                if cmd_q is not None:
                    while True:
                        try:
                            handle(cmd_q.get_nowait())
                        except queue.Empty:
                            break
                    while paused and not stop:
                        try:
                            handle(cmd_q.get(timeout=0.1))
                        except queue.Empty:
                            pass
                    if stop:
                        break
                s = evo.step_generation()
                champ = s.pop("champion")
                if validate_every and s["gen"] % validate_every == 0:
                    r = convergence(champ, stage=min(s["stage"], 2), seed=s["gen"])
                    s["validity"] = r["ratio"]
                    champ.meta["validity"] = r["ratio"]
                    with open(os.path.join(out, "validity.csv"), "a", encoding="utf-8") as vf:
                        vf.write(f"{s['gen']},{r['coarse']:.4f},{r['fine']:.4f},{r['ratio']:.3f}\n")
                    if verbose:
                        print(f"   physics validation: {r['coarse']:+.2f} at dt, {r['fine']:+.2f} at dt/2 -> ratio {r['ratio']:.2f}"
                              + ("" if r["ok"] else "  <-- SUSPICIOUS"), flush=True)
                log.write(",".join(str(round(s[k], 5)) if isinstance(s[k], float) else str(s[k]) for k in CSV_COLS) + "\n")
                log.flush()
                if verbose:
                    print(f"generation {s['gen']:4d} | difficulty {s['notch']:2d}/{s['notch_max']} (stage {s['stage']}) | best {s['best']:+.3f} "
                          f"(progress {s['best_progress']:+.3f}) | mean {s['mean']:+.3f} | limbs {s['best_limbs']:2d} "
                          f"| {s['seconds']:.1f}s" + (f" | STALL {s['stall']} generations, mutation x{s['mut_scale']:.1f}" if s['mut_scale'] > 1 else ""), flush=True)
                if s["best"] > best_by_stage.get(s["stage"], -1e9):
                    best_by_stage[s["stage"]] = s["best"]
                    save_json(champ, os.path.join(out, "best.json"))
                if s["gen"] % save_every == 0:
                    save_json(champ, os.path.join(out, f"gen_{s['gen']:04d}.json"))
                if s["gen"] % 5 == 0:
                    evo.save_checkpoint(ck)
                say("gen", s, champ.to_dict())
        except KeyboardInterrupt:
            if verbose:
                print("stopped")
        finally:
            evo.save_checkpoint(ck)
            evo.close()
            log.close()
        say("done")
    except Exception as e:  # report to the UI
        import traceback
        say("error", traceback.format_exc())
        raise

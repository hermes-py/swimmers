"""
Eğitim döngüsü (grafiksiz). İki şekilde kullanılır:
  * train.py (komut satırı)    -> run_training(...) doğrudan çağrılır
  * app.py   (pygame arayüzü)  -> ayrı bir süreçte çalışır; komut/mesaj kuyruklarıyla konuşur

Komutlar (cmd_q): ("pause",) ("resume",) ("workers", n) ("stop",)
Mesajlar (msg_q): ("gen", özet_sözlüğü, şampiyon_sözlüğü) ("info", metin) ("error", metin) ("done",)
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
    """gens=0 -> sınırsız (durdurulana kadar)."""
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
            say("info", f"nesil {evo.gen}'den devam ediliyor")
            if getattr(evo, "loaded_physics", 2) != PHYSICS_VERSION:
                msg = ("UYARI: bu checkpoint eski fizik sürümüyle eğitilmiş; sayısal hatadan yararlanan davranışlar "
                       "yeni fizikte çalışmayacak, popülasyon yeniden uyum sağlayacak.")
                say("info", msg)
                if verbose:
                    print(msg)
            if verbose:
                print(f"nesil {evo.gen}'den devam ediliyor")
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
                        print(f"   fizik doğrulama: dt'de {r['coarse']:+.2f}, dt/2'de {r['fine']:+.2f} -> oran {r['ratio']:.2f}"
                              + ("" if r["ok"] else "  <-- ŞÜPHELİ"), flush=True)
                log.write(",".join(str(round(s[k], 5)) if isinstance(s[k], float) else str(s[k]) for k in CSV_COLS) + "\n")
                log.flush()
                if verbose:
                    print(f"nesil {s['gen']:4d} | zorluk {s['notch']:2d}/{s['notch_max']} (aşama {s['stage']}) | en iyi {s['best']:+.3f} "
                          f"(ilerleme {s['best_progress']:+.3f}) | ort {s['mean']:+.3f} | uzuv {s['best_limbs']:2d} "
                          f"| {s['seconds']:.1f}s" + (f" | TAKILMA {s['stall']} nesil, mutasyon x{s['mut_scale']:.1f}" if s['mut_scale'] > 1 else ""), flush=True)
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
                print("durduruldu")
        finally:
            evo.save_checkpoint(ck)
            evo.close()
            log.close()
        say("done")
    except Exception as e:  # arayüze bildir
        import traceback
        say("error", traceback.format_exc())
        raise

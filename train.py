"""
Headless (grafiksiz) eğitim komutu.

Örnek:
    python train.py --out runs/deney1 --pop 64 --gens 200 --workers 4
    python train.py --out runs/deney1 --resume          # kaldığı yerden devam
Çıktılar (--out klasörü):
    best.json              şimdiye kadarki en iyi birey (insan okunur; app.py ile yüklenebilir)
    gen_0050.json ...      her --save-every nesilde o nesil şampiyonu
    checkpoint.pkl         eğitimi sürdürmek için tüm popülasyon (dahili format)
    log.csv                nesil başına özet
Ctrl+C ile güvenle durdurulabilir.
(Grafikli kullanım için: python app.py)
"""
import argparse
import os

from genome import load_json
from trainer import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--gens", type=int, default=200, help="0 = sınırsız")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--stage-len", type=int, default=25, help="müfredat aşaması başına nesil")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed-from", default=None, help="bir tasarım/canlı JSON'undan başlangıç popülasyonu üret")
    ap.add_argument("--freeze-body", action="store_true", help="vücut sabit, sadece beyin evrimleşsin (--seed-from ile)")
    ap.add_argument("--min-limbs", type=int, default=4, help="mutasyonla inilebilecek en az uzuv sayısı")
    ap.add_argument("--stop-command", action="store_true", help="'komut yok = dur' kademesini aç (varsayılan KAPALI)")
    ap.add_argument("--legacy-spine", action="store_true", help="ESKİ omurga modu: kemik 1 pasif, gövde yönünü o belirler (varsayılan: gövde yönü = kafanın yönü)")
    ap.add_argument("--no-stall-boost", action="store_true", help="takılma algılayınca mutasyonu güçlendirmeyi kapat")
    ap.add_argument("--curriculum", choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--validate-every", type=int, default=10, help="şampiyonun fizik doğrulaması (0 = kapalı)")
    a = ap.parse_args()
    run_training(out=a.out, pop=a.pop, gens=a.gens, trials=a.trials, duration=a.duration, workers=a.workers,
                 stage_len=a.stage_len, seed=a.seed, save_every=a.save_every, resume=a.resume,
                 validate_every=a.validate_every, freeze_body=a.freeze_body, min_limbs=a.min_limbs,
                 curriculum=a.curriculum, stop_command=a.stop_command, stall_boost=not a.no_stall_boost, legacy_spine=a.legacy_spine,
                 seed_genome=load_json(a.seed_from) if a.seed_from else None)


if __name__ == "__main__":
    main()

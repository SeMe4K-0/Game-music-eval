"""
Калибровка порогов вырожденности по референсному корпусу.

Пороги по умолчанию (src/metrics/degeneracy.py) выбраны по синтетике и
консервативны. Правильный порог — тот, ниже которого не опускается реальная
ретро-игровая музыка: считаем признаки на окнах NES-VMDB и берём 1-й
перцентиль. Это та же логика самокалибрующегося порога, что у шва (p95
внутритрековых переходов), и она защищает от произвольных констант в статье.

Запуск (CPU, MacBook): python -m src.calibrate_degeneracy --ref-dir results/references
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .audio_utils import load_for_analysis
from .metrics import degeneracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", type=Path, default=Path("results/references"))
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--out", type=Path, default=Path("results/degeneracy_thresholds.json"))
    ap.add_argument("--percentile", type=float, default=1.0)
    ap.add_argument("--max-files-per-duration", type=int, default=300)
    a = ap.parse_args()

    dcfg = yaml.safe_load(open(a.config_dir / "durations.yaml", encoding="utf-8"))
    sr_a, peak = int(dcfg["analysis_sr"]), float(dcfg.get("peak_normalize_to", 0.9))

    feats = []
    for d in sorted(a.ref_dir.glob("*s")):
        files = sorted(d.glob("*.wav"))[: a.max_files_per_duration]
        for f in files:
            y, sr = load_for_analysis(f, sr_a, peak)
            # референс нормализован так же, как генерации; orig_peak не нужен
            feats.append(degeneracy.features(y, sr))
        print(f"{d.name}: {len(files)} файлов")

    if not feats:
        raise SystemExit(f"в {a.ref_dir} нет .wav — сначала python -m src.corpus_prep")

    th = degeneracy.calibrate(feats, q=a.percentile)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(th, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(th, ensure_ascii=False, indent=1))

    n_flagged = sum(degeneracy.is_degenerate(f, th)["degenerate"] for f in feats)
    print(f"\nконтроль: порогами помечено {n_flagged}/{len(feats)} окон САМОГО референса "
          f"({100*n_flagged/len(feats):.1f} %) — ожидается около {a.percentile} % на признак")


if __name__ == "__main__":
    main()

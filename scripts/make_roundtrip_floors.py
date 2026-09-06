#!/usr/bin/env python3
"""
Подготовка КРУГОВОГО пола (codec round-trip) для FAD/KAD.

Зачем. Пол split_half отвечает на вопрос «сколько дал бы идеальный
генератор» — это предел конечной выборки. Круговой пол отвечает на другой
вопрос: «сколько дало бы само представление модели, если бы генерации не было
вовсе». Референс прогоняется через кодек/VAE модели туда-обратно, и
получившееся расстояние — потолок, ниже которого эта модель опуститься не
может в принципе. У каждой модели своё узкое горлышко, поэтому пол считается
per-model: единый EnCodec-пол был бы полом только для MusicGen.

Почему отдельным скриптом. `compute_distributional` ищет уже готовые папки
`results/references/<D>s_rt_<codec>` и молча пропускает их, если папок нет.
Реализации в `src/metrics/codec_floor.py` были написаны, но никто их не
вызывал — из решения D.3 работала ровно половина, и это не было видно ни в
логе, ни в отчёте.

Запуск (нужен GPU; для riffusion — своё окружение .venv-riffusion):
    python -m scripts.make_roundtrip_floors --codecs encodec stable_audio
    python -m scripts.make_roundtrip_floors --codecs riffusion
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import codec_floor                                # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("roundtrip")

# Имя папки -> (функция, к какой модели относится пол). Имя после "_rt_"
# попадает в floors.jsonl как идентификатор пола, поэтому оно должно читаться
# в таблице статьи без расшифровки.
CODECS = {
    "encodec": (codec_floor.roundtrip_encodec, "musicgen_small"),
    "stable_audio": (codec_floor.roundtrip_stable_audio_vae, "stable_audio_open_1_0"),
    "riffusion": (codec_floor.roundtrip_riffusion_spectrogram, "riffusion_v1"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--codecs", nargs="*", default=["encodec", "stable_audio"],
                    help=f"из: {', '.join(CODECS)}")
    ap.add_argument("--durations", nargs="*", type=float,
                    help="по умолчанию все, для которых есть референс")
    ap.add_argument("--limit", type=int, default=None,
                    help="сколько окон референса прогонять (по умолчанию все)")
    a = ap.parse_args()

    ref_root = a.out_dir / "references"
    dirs = sorted(d for d in ref_root.glob("*s") if d.is_dir() and "_half_" not in d.name)
    if a.durations:
        want = {f"{int(D)}s" for D in a.durations}
        dirs = [d for d in dirs if d.name in want]
    if not dirs:
        log.error("нет референса в %s — сначала python -m src.corpus_prep", ref_root)
        return 1

    for name in a.codecs:
        if name not in CODECS:
            log.error("неизвестный кодек %s (есть: %s)", name, ", ".join(CODECS))
            return 1
        fn, model_id = CODECS[name]
        for ref_dir in dirs:
            out = ref_root / f"{ref_dir.name}_rt_{name}"
            files = sorted(ref_dir.glob("*.wav"))
            if a.limit:
                files = files[:a.limit]
            done = len(list(out.glob("*.wav")))
            if done >= len(files):
                log.info("%s %s: уже готово (%d файлов)", name, ref_dir.name, done)
                continue
            log.info("%s (пол для %s) %s: %d окон...", name, model_id, ref_dir.name, len(files))
            t0 = time.time()
            try:
                n = fn(files, out)
                log.info("  готово %d за %.1f мин -> %s", n, (time.time() - t0) / 60, out)
            except Exception as e:                       # один кодек не должен ронять остальные
                log.error("  %s на %s упал: %r", name, ref_dir.name, e)
    log.info("дальше: python -m src.compute_distributional (полы подхватятся сами)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

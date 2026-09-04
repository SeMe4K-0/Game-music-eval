"""
Слитый цикл для арендованной/бесплатной GPU-сессии: генерация -> метрики ->
(опционально) эмбеддинги -> удаление аудио, всё в одной сессии.

Зачем. Полный план даёт около 85 ГБ аудио (PCM_24): musicgen 14.5, ace_step
43.5, audioldm2 6.0, riffusion 16.6, stable_audio 4.9. Ни Kaggle (20 ГБ
сохраняемого вывода), ни бесплатный Google Drive (15 ГБ) столько не примут, а
качать это между площадками бессмысленно. При этом всё, что реально нужно для
статьи, — это per-track метрики (десятки МБ) и CLAP-эмбеддинги (около 1 ГБ
float32 на весь план, 0.5 ГБ во float16). DSP-метрики стоят 0.7 с CPU на трек
(измерено), то есть считаются на простаивающих ядрах, пока GPU занят
следующим треком, и почти ничего не добавляют к времени сессии.

Политика аудио (--keep-audio):
  none   — удалять сразу после метрик (по умолчанию для бесплатных площадок);
  sample — оставить первые --sample-per-cell треков каждой клетки (архив для
           перепроверки метрик без полной перегенерации);
  all    — оставить всё (только если места хватает; тогда лучше --audio-ext .flac).

Важно: аудио НЕ воспроизводится побитово на другом железе даже при том же
сиде, поэтому архивный сэмпл — единственный способ пересчитать метрики позже.

Запуск (пример для Kaggle, одна клетка целиком):
    python -m src.remote_worker --models riffusion_v1 --durations 30 \
        --keep-audio sample --sample-per-cell 2 --out-dir /kaggle/working/results
"""
from __future__ import annotations

import argparse
import json
import logging
import tarfile
from collections import defaultdict
from pathlib import Path

import yaml

from . import run_experiment
from .audio_utils import load_for_analysis
from .env_info import describe
from .metrics import degeneracy, game_criteria, structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("remote_worker")


def build_callback(out_dir: Path, config_dir: Path, keep_audio: str, sample_per_cell: int,
                   with_clap: bool):
    dcfg = yaml.safe_load(open(config_dir / "durations.yaml", encoding="utf-8"))
    analysis_sr = int(dcfg["analysis_sr"])
    peak = float(dcfg.get("peak_normalize_to", 0.9))
    prompts = {p["id"]: p["text"] for p in
               yaml.safe_load(open(config_dir / "prompts.yaml", encoding="utf-8"))["prompts"]}

    thresholds = None
    tpath = out_dir / "degeneracy_thresholds.json"
    if tpath.exists():
        thresholds = json.load(open(tpath, encoding="utf-8"))
    else:
        log.warning("degeneracy_thresholds.json нет — пороги по умолчанию (не откалиброваны по референсу)")

    scorer = None
    if with_clap:
        from .metrics.clap_score import ClapScorer
        scorer = ClapScorer()

    metrics_path = out_dir / "per_track_metrics.jsonl"
    kept = defaultdict(int)

    def on_track(record: dict, wav: Path) -> None:
        y, sr = load_for_analysis(wav, analysis_sr, peak)
        rec = {**{k: record[k] for k in ("model_id", "duration_s", "prompt_id", "prompt_category",
                                          "loop_eval", "set_idx", "track_idx", "seed",
                                          "actual_duration_s")},
               "wav_path": str(wav), "status": "done", "analysis_sr": sr, "env": record.get("env")}
        try:
            rec["degeneracy"] = degeneracy.report(y, sr, orig_peak=record.get("orig_peak"),
                                                   orig_rms=record.get("orig_rms"), thresholds=thresholds)
            rec["game"] = game_criteria.game_loop_report(y, sr)
            rec["structure"] = structure.structure_report(y, sr)
        except Exception as e:
            rec["error"] = repr(e)
            log.warning("метрики упали на %s: %r", wav.name, e)
        if scorer is not None:
            rec["clap_score"] = scorer.score(y, sr, prompts[record["prompt_id"]], str(wav)).score

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")

        cell = (record["model_id"], record["duration_s"])
        if keep_audio == "all":
            return
        if keep_audio == "sample" and kept[cell] < sample_per_cell:
            kept[cell] += 1
            return
        wav.unlink(missing_ok=True)

    return on_track


def _default(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def pack(out_dir: Path, dest: Path) -> Path:
    """Небольшой архив для выгрузки: манифест, метрики, эмбеддинги, архивный
    сэмпл аудио. Именно он переносится с площадки, а не всё аудио."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        for name in ("generation_manifest.jsonl", "per_track_metrics.jsonl",
                     "degeneracy_thresholds.json"):
            p = out_dir / name
            if p.exists():
                tar.add(p, arcname=name)
        for sub in ("embeddings", "audio"):
            p = out_dir / sub
            if p.exists():
                tar.add(p, arcname=sub)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--durations", nargs="*", type=float)
    ap.add_argument("--sets", nargs="*", type=int)
    ap.add_argument("--shard", help="i/n — своя доля клеток")
    ap.add_argument("--max-tracks", type=int)
    ap.add_argument("--keep-audio", choices=["none", "sample", "all"], default="sample")
    ap.add_argument("--sample-per-cell", type=int, default=2)
    ap.add_argument("--audio-ext", choices=[".wav", ".flac"], default=".wav")
    ap.add_argument("--with-clap", action="store_true")
    ap.add_argument("--pack", type=Path, help="упаковать результаты в этот .tar.gz по завершении")
    a = ap.parse_args()

    log.info("среда: %s", describe())
    shard = None
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        shard = (i, n)

    cb = build_callback(a.out_dir, a.config_dir, a.keep_audio, a.sample_per_cell, a.with_clap)
    run_experiment.run(a.config_dir, a.out_dir, a.models, a.durations, a.max_tracks, False,
                       shard, set(a.sets) if a.sets else None, on_track=cb, audio_ext=a.audio_ext)
    if a.pack:
        log.info("упаковано: %s", pack(a.out_dir, a.pack))


if __name__ == "__main__":
    main()

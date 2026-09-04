"""
Стадия 2: per-track метрики групп 2-4 по wav из generation_manifest.jsonl.
CPU (группы 3-4); CLAP — при --with-clap (GPU или CPU, медленно).
Идемпотентно по ключу трека. Можно запускать на MacBook параллельно генерации.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from .audio_utils import load_for_analysis
from .metrics import degeneracy, game_criteria, structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("track_metrics")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def iter_manifest(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "done":
                yield r


def key_of(r: dict) -> tuple:
    return (r["model_id"], r["duration_s"], r["prompt_id"], r["set_idx"], r["track_idx"])


def load_thresholds(out_dir: Path) -> dict | None:
    """Пороги вырожденности, откалиброванные по референсу
    (src/calibrate_degeneracy.py). Без файла берутся консервативные значения
    по умолчанию — это фиксируется в отчёте."""
    p = out_dir / "degeneracy_thresholds.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return None


def run(out_dir: Path, config_dir: Path, with_clap: bool) -> None:
    dcfg = yaml.safe_load(open(config_dir / "durations.yaml", encoding="utf-8"))
    analysis_sr, peak = int(dcfg["analysis_sr"]), float(dcfg.get("peak_normalize_to", 0.9))
    prompts = {p["id"]: p["text"] for p in yaml.safe_load(open(config_dir / "prompts.yaml", encoding="utf-8"))["prompts"]}

    thresholds = load_thresholds(out_dir)
    if thresholds is None:
        log.warning("degeneracy_thresholds.json не найден — пороги по умолчанию, "
                    "не откалиброванные по референсу")
    out_path = out_dir / "per_track_metrics.jsonl"
    done = set()
    if out_path.exists():
        for r in iter_manifest(out_path):
            if (not with_clap) or ("clap_score" in r):
                done.add(key_of(r))

    scorer = None
    if with_clap:
        from .metrics.clap_score import ClapScorer
        scorer = ClapScorer()

    for r in iter_manifest(out_dir / "generation_manifest.jsonl"):
        k = key_of(r)
        if k in done:
            continue
        wav = Path(r["wav_path"])
        if not wav.exists():
            log.warning("нет файла %s", wav)
            continue
        y, sr = load_for_analysis(wav, analysis_sr, peak)
        rec = {**{f: r[f] for f in ("model_id", "duration_s", "prompt_id", "prompt_category", "loop_eval",
                                     "set_idx", "track_idx", "seed", "wav_path", "actual_duration_s")},
               "status": "done", "analysis_sr": sr, "env": r.get("env")}
        try:
            rec["degeneracy"] = degeneracy.report(y, sr, orig_peak=r.get("orig_peak"),
                                                   orig_rms=r.get("orig_rms"), thresholds=thresholds)
            rec["game"] = game_criteria.game_loop_report(y, sr)
            rec["structure"] = structure.structure_report(y, sr)
        except Exception as e:  # трек слишком короткий/тихий и т.п. — фиксируем, не роняем прогон
            rec["error"] = repr(e)
        if scorer is not None:
            rec["clap_score"] = scorer.score(y, sr, prompts[r["prompt_id"]], str(wav)).score
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        done.add(k)
    log.info("per-track метрики: %d треков", len(done))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--with-clap", action="store_true")
    a = ap.parse_args()
    run(a.out_dir, a.config_dir, a.with_clap)


if __name__ == "__main__":
    main()

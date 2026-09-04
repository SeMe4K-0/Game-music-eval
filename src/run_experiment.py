"""
Стадия 1: генерация. Только GPU-работа; метрики считаются отдельными
стадиями (compute_track_metrics.py, compute_distributional.py), чтобы
CPU-анализ не простаивал GPU и мог идти на другой машине (MacBook).

Пишет results/audio/<model>/<D>s/<prompt>/set<k>/seed<seed>.wav и
results/generation_manifest.jsonl (идемпотентно: ключ (model, D, prompt,
set, track) пропускается, если уже записан; клетки NOT_APPLICABLE
регистрируются один раз).

Запуск: python -m src.run_experiment [--models m1 m2] [--durations 15 30]
        [--max-tracks N] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import yaml

from .audio_utils import write_wav
from .env_info import describe, fingerprint
from .generate import ModelRegistry, NotApplicable, generate, load_models_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_experiment")

# Отпечаток среды пишется в КАЖДУЮ запись: при раздаче клеток плана по разным
# площадкам (3060 / T4 / L4) одинаковый сид НЕ даёт одинаковое аудио, и
# «железо» не должно оказаться скрытым фактором, спутанным с моделью.
ENV = fingerprint()


def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def k_for_cell(model_id: str, duration_s: float, plan: dict) -> int:
    r = plan["reduced_K_cells"]
    if model_id in r["models"] and duration_s in r["durations_s"]:
        return int(plan["reduced_K"])
    if model_id in plan["full_K_cells"]["models"]:
        return int(plan["full_K"])
    return int(plan["default_K"])


def make_seed(base: int, set_idx: int, prompt_idx: int, track_idx: int) -> int:
    return int(base + 100000 * set_idx + 1000 * prompt_idx + track_idx)


def load_done(manifest: Path) -> tuple[set, set]:
    done, na = set(), set()
    if not manifest.exists():
        return done, na
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "done":
                done.add((r["model_id"], r["duration_s"], r["prompt_id"], r["set_idx"], r["track_idx"]))
            elif r.get("status") == "not_applicable":
                na.add((r["model_id"], r["duration_s"]))
    return done, na


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


def cell_in_shard(model_id: str, duration_s: float, shard: tuple[int, int] | None) -> bool:
    """Шардирование по КЛЕТКАМ (модель × длительность), а не по трекам: клетка
    целиком считается на одной площадке, иначе железо окажется спутанным с
    моделью внутри одного сравнения (см. src/env_info.py)."""
    if shard is None:
        return True
    i, n = shard
    h = int(hashlib.md5(f"{model_id}|{duration_s}".encode()).hexdigest(), 16)
    return h % n == i


def run(config_dir: Path, out_dir: Path, only_models=None, only_durations=None,
        max_tracks: int | None = None, dry_run: bool = False,
        shard: tuple[int, int] | None = None, only_sets=None,
        on_track=None, audio_ext: str = ".wav") -> None:
    """on_track(record, wav_path) вызывается после записи каждого трека —
    через него src/remote_worker.py считает метрики и удаляет аудио, не
    дублируя обход плана."""
    models = load_models_config(config_dir / "models.yaml")
    prompts = load_yaml(config_dir / "prompts.yaml")["prompts"]
    dcfg = load_yaml(config_dir / "durations.yaml")
    durations = [float(d) for d in dcfg["durations_s"]]
    plan, n_tracks = dcfg["factorial_plan"], int(dcfg["N_tracks_per_prompt_per_set"])
    seed_base, peak = int(dcfg["seed_base"]), float(dcfg.get("peak_normalize_to", 0.9))
    if max_tracks:
        n_tracks = min(n_tracks, max_tracks)

    manifest = out_dir / "generation_manifest.jsonl"
    done, na = load_done(manifest)
    registry = ModelRegistry()

    for cfg in models:
        mid = cfg["id"]
        if only_models and mid not in only_models:
            continue
        for D in durations:
            if only_durations and D not in only_durations:
                continue
            if (mid, D) in na:
                continue
            if not cell_in_shard(mid, D, shard):
                continue
            if cfg.get("long_duration_strategy") == "not_applicable_above_native" and D > float(cfg["native_max_duration_s"]):
                log.info("NOT_APPLICABLE %s @ %.0fs (нативный максимум %.0fs)", mid, D, cfg["native_max_duration_s"])
                append(manifest, {"status": "not_applicable", "model_id": mid, "duration_s": D,
                                  "reason": f"native_max={cfg['native_max_duration_s']}"})
                na.add((mid, D))
                continue
            K = k_for_cell(mid, D, plan)
            for set_idx in range(K):
                if only_sets is not None and set_idx not in only_sets:
                    continue
                for p_idx, p in enumerate(prompts):
                    for t_idx in range(n_tracks):
                        key = (mid, D, p["id"], set_idx, t_idx)
                        if key in done:
                            continue
                        seed = make_seed(seed_base, set_idx, p_idx, t_idx)
                        wav = out_dir / "audio" / mid / f"{int(D)}s" / p["id"] / f"set{set_idx}" / f"seed{seed}{audio_ext}"
                        log.info("gen %s D=%.0f %s set=%d track=%d seed=%d", mid, D, p["id"], set_idx, t_idx, seed)
                        if dry_run:
                            continue
                        try:
                            res = generate(registry, cfg, p["text"], D, seed)
                        except NotApplicable as e:
                            log.warning("%s", e)
                            append(manifest, {"status": "not_applicable", "model_id": mid, "duration_s": D, "reason": str(e)})
                            na.add((mid, D))
                            break
                        wrote = write_wav(wav, res.audio, res.sr, peak)
                        record = {
                            "status": "done", "model_id": mid, "duration_s": D, "prompt_id": p["id"],
                            "prompt_category": p.get("category"), "loop_eval": bool(p.get("loop_eval", True)),
                            "set_idx": set_idx, "track_idx": t_idx, "seed": seed, "wav_path": str(wav),
                            "sr": res.sr, "channels": 1 if res.audio.ndim == 1 else int(res.audio.shape[0]),
                            "requested_duration_s": D,
                            "actual_duration_s": wrote["actual_duration_s"],
                            "orig_peak": wrote["orig_peak"], "orig_rms": wrote["orig_rms"],
                            "wall_time_s": res.wall_time_s, "peak_vram_gb": res.peak_vram_gb,
                            "env": ENV,
                        }
                        append(manifest, record)
                        done.add(key)
                        if on_track is not None:
                            on_track(record, wav)
                    if (mid, D) in na:
                        break
                if (mid, D) in na:
                    break
        registry.release()
    log.info("генерация завершена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--durations", nargs="*", type=float)
    ap.add_argument("--max-tracks", type=int, help="ограничить N на промпт (фаза 0 / smoke)")
    ap.add_argument("--sets", nargs="*", type=int, help="только эти наборы (set_idx)")
    ap.add_argument("--shard", help="i/n — взять только свою долю клеток (для раздачи по площадкам)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    shard = None
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {a.shard}: нужно 0 <= i < n")
        shard = (i, n)
    log.info("среда: %s", describe())
    run(a.config_dir, a.out_dir, a.models, a.durations, a.max_tracks, a.dry_run,
        shard, set(a.sets) if a.sets else None)


if __name__ == "__main__":
    main()

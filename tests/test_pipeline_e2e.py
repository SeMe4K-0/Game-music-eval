"""
Сквозной прогон всего пайплайна на подставных бэкендах: план -> генерация ->
метрики -> вырожденность -> сводка -> решающее правило. Без GPU и весов.

Смысл: проверить связки (манифест, идемпотентность, шардирование, колбэк,
политика хранения аудио, gating по вырожденности, построение таблиц), которые
нельзя проверить юнит-тестами метрик, и которые дороже всего чинить, обнаружив
поломку через сутки прогона на GPU.

Ключевая проверка: модель, выдающая постоянный тон, НЕ должна проходить
решающее правило, даже если её шов и дрейф темпа формально идеальны.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generate as gen_mod
from src import remote_worker, run_experiment, summarize

SR = 22050


def _melody(dur, sr=SR, seed=0):
    """Нормальный музыкальный выход: смена нот, огибающая, удары."""
    rng = np.random.default_rng(seed)
    notes = [261.6, 329.6, 392.0, 523.3, 440.0, 349.2]
    seg, out = 0.5, []
    for k in range(int(np.ceil(dur / seg))):
        f = notes[(k + seed) % len(notes)]
        t = np.arange(int(seg * sr)) / sr
        env = np.minimum(1.0, np.exp(-3 * t) + 0.2)
        out.append((0.5 * np.sin(2 * np.pi * f * t) * env).astype(np.float32))
    y = np.concatenate(out)[: int(dur * sr)]
    return y + rng.normal(0, 0.005, len(y)).astype(np.float32)


class _MelodyBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, prompt, duration_s, seed):
        return _melody(duration_s, SR, seed % 6), SR


class _DroneBackend:
    """Вырожденный выход: постоянный тон. Формально даёт отличный шов и
    нулевой дрейф темпа — ровно тот случай, ради которого сделан детектор."""

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, prompt, duration_s, seed):
        t = np.arange(int(duration_s * SR)) / SR
        return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    models = {"models": [
        {"id": "good", "hf_repo": "x/good", "year": 2025, "architecture": "test",
         "license": "test", "sample_rate": SR, "channels": 1, "native_max_duration_s": 120,
         "long_duration_strategy": "direct_call", "dtype": "float32", "backend": "melody"},
        {"id": "drone", "hf_repo": "x/drone", "year": 2023, "architecture": "test",
         "license": "test", "sample_rate": SR, "channels": 1, "native_max_duration_s": 20,
         "long_duration_strategy": "not_applicable_above_native", "dtype": "float32",
         "backend": "drone"},
    ]}
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(models), encoding="utf-8")
    (cfg_dir / "prompts.yaml").write_text(yaml.safe_dump({"prompts": [
        {"id": "battle", "text": "battle theme", "category": "combat"},
        {"id": "fanfare", "text": "victory fanfare", "category": "stinger", "loop_eval": False},
    ]}), encoding="utf-8")
    (cfg_dir / "durations.yaml").write_text(yaml.safe_dump({
        "durations_s": [15, 30],
        "analysis_sr": SR, "peak_normalize_to": 0.9,
        "N_tracks_per_prompt_per_set": 2,
        "factorial_plan": {"full_K": 2, "reduced_K": 1,
                            "full_K_cells": {"models": ["good"], "durations_s": [15, 30]},
                            "reduced_K_cells": {"models": ["drone"], "durations_s": [15]},
                            "default_K": 2},
        "seed_base": 1000, "bootstrap_resamples": 50,
        "decision_rule": {"loop_prompt_categories": ["combat"],
                           "seam_pass_fraction_min": 0.5,
                           "seam_gap_beat_error_max": 0.10,
                           "bar_fraction_error_max": 0.10,
                           "grid_pass_fraction_min": 0.5},
    }), encoding="utf-8")
    monkeypatch.setitem(gen_mod._BACKENDS, "melody", _MelodyBackend)
    monkeypatch.setitem(gen_mod._BACKENDS, "drone", _DroneBackend)
    return cfg_dir, tmp_path / "results"


def test_end_to_end(env):
    cfg_dir, out_dir = env
    cb = remote_worker.build_callback(out_dir, cfg_dir, keep_audio="sample",
                                       sample_per_cell=1, with_clap=False)
    run_experiment.run(cfg_dir, out_dir, on_track=cb)

    manifest = [json.loads(l) for l in open(out_dir / "generation_manifest.jsonl", encoding="utf-8")]
    done = [r for r in manifest if r["status"] == "done"]
    na = [r for r in manifest if r["status"] == "not_applicable"]

    # good: 2 длительности x 2 промпта x 2 трека x K=2 = 16
    # drone: 15 с (reduced K=1) x 2 промпта x 2 трека = 4; 30 с > native 20 -> NOT_APPLICABLE
    assert len([r for r in done if r["model_id"] == "good"]) == 16
    assert len([r for r in done if r["model_id"] == "drone"]) == 4
    assert len(done) == 20, {m: sum(1 for r in done if r["model_id"] == m) for m in ("good", "drone")}
    assert len(na) == 1 and na[0]["model_id"] == "drone" and na[0]["duration_s"] == 30.0

    for r in done:            # фактическая длительность и уровни записаны
        assert abs(r["actual_duration_s"] - r["duration_s"]) < 0.1
        assert r["orig_peak"] > 0 and "env" in r

    # политика хранения: по одному треку на клетку (good x 2 длительности + drone x 15 с)
    kept = list((out_dir / "audio").rglob("*.wav"))
    assert len(kept) == 3, [p.name for p in kept]

    tracks = [json.loads(l) for l in open(out_dir / "per_track_metrics.jsonl", encoding="utf-8")]
    assert len(tracks) == len(done)
    assert all("game" in t and "structure" in t and "degeneracy" in t for t in tracks)

    # детектор: гудок вырожден, мелодия — нет
    deg = {m: [t["degeneracy"]["degenerate"] for t in tracks if t["model_id"] == m]
           for m in ("good", "drone")}
    assert all(deg["drone"]), "постоянный тон обязан быть помечен вырожденным"
    assert not any(deg["good"]), "мелодия не должна помечаться вырожденной"

    summarize.run(out_dir, cfg_dir)
    decision = json.load(open(out_dir / "summary" / "decision.json", encoding="utf-8"))
    for key, v in decision["verdicts"].items():
        if key.startswith("drone|"):
            assert not v["seam"] and not v["grid"], (key, v)
            assert not v["pass"], f"вырожденная модель прошла правило: {key} {v}"
    assert (out_dir / "summary" / "summary.md").exists()
    md = (out_dir / "summary" / "summary.md").read_text(encoding="utf-8")
    assert "вырожд." in md and "drone" in md

    # доля вырожденных попала в сводку
    s15 = json.load(open(out_dir / "summary" / "15s.json", encoding="utf-8"))
    assert s15["drone"]["tracks"]["degenerate"]["p"] == 1.0
    assert s15["good"]["tracks"]["degenerate"]["p"] == 0.0


def test_idempotent_resume(env):
    cfg_dir, out_dir = env
    cb = remote_worker.build_callback(out_dir, cfg_dir, "none", 0, False)
    run_experiment.run(cfg_dir, out_dir, only_models=["good"], only_durations=[15.0], on_track=cb)
    n1 = sum(1 for _ in open(out_dir / "generation_manifest.jsonl", encoding="utf-8"))
    run_experiment.run(cfg_dir, out_dir, only_models=["good"], only_durations=[15.0], on_track=cb)
    n2 = sum(1 for _ in open(out_dir / "generation_manifest.jsonl", encoding="utf-8"))
    assert n1 == n2 > 0, "повторный запуск не должен генерировать заново"


def test_not_applicable_recorded(env):
    """Длительность выше нативного максимума модели не генерируется молча
    обрезанной, а фиксируется как «не применимо» — это результат по оси
    длительности (случай Stable Audio Open выше 47 с)."""
    cfg_dir, out_dir = env
    run_experiment.run(cfg_dir, out_dir, only_models=["drone"], only_durations=[30.0])
    manifest = [json.loads(l) for l in open(out_dir / "generation_manifest.jsonl", encoding="utf-8")]
    assert manifest and all(r["status"] == "not_applicable" for r in manifest)
    assert manifest[0]["duration_s"] == 30.0
    assert not list(out_dir.glob("audio/**/*.wav"))


def test_shards_partition_cells_exactly_once():
    cells = [("m1", 15.0), ("m1", 30.0), ("m2", 15.0), ("m2", 120.0), ("m3", 60.0), ("m3", 90.0)]
    for n in (2, 3, 4):
        counts = [sum(run_experiment.cell_in_shard(m, d, (i, n)) for i in range(n)) for m, d in cells]
        assert counts == [1] * len(cells), (n, counts)
    assert all(run_experiment.cell_in_shard(m, d, None) for m, d in cells)


def test_seeds_unique_across_sets():
    seeds = {run_experiment.make_seed(1000, s, p, t)
             for s in range(5) for p in range(8) for t in range(12)}
    assert len(seeds) == 5 * 8 * 12, "сиды обязаны различаться между наборами"

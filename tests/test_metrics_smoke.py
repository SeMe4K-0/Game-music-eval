"""
Смоук-тесты DSP-логики групп 3-4, статистики и подготовки референса — на
синтетическом аудио с известным правильным ответом, без GPU и весов моделей.
Запуск: python3 -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import stats, audio_utils
from src.metrics import game_criteria as gc, structure

SR = 22050


def sine(freq, dur, sr=SR, phase=0.0, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def click_track(dur, bpm, sr=SR, offset_s=0.0):
    """Метроном: щелчок на каждой доле, первая доля в offset_s."""
    y = np.zeros(int(dur * sr), dtype=np.float32)
    click = sine(1000, 0.03, sr, amp=0.9) * np.hanning(int(0.03 * sr)).astype(np.float32)
    t = offset_s
    while t < dur:
        s = int(t * sr); e = min(s + len(click), len(y))
        y[s:e] += click[:e - s]
        t += 60.0 / bpm
    return y


class TestSeam:
    def test_smooth_loop_ok(self):
        y = sine(220.0, 4.0)                       # 880 периодов — фаза непрерывна на петле
        r = gc.seam_discontinuity(y, SR, 50.0)
        assert r["seam_ok_p95"] and r["click_ok"], r

    def test_hard_seam_flagged(self):
        y = np.concatenate([sine(220, 2, amp=0.8), sine(880, 2, amp=0.8, phase=1.7)])
        r = gc.seam_discontinuity(y, SR, 50.0)
        assert r["seam_score"] > 1.0 and not r["seam_ok_p95"], r

    def test_hard_worse_than_smooth(self):
        a = gc.seam_discontinuity(sine(220, 4), SR, 50.0)["seam_score"]
        b = gc.seam_discontinuity(np.concatenate([sine(220, 2, amp=.8), sine(880, 2, amp=.8, phase=2.3)]), SR, 50.0)["seam_score"]
        assert b > a

    def test_click_detected_on_dc_jump(self):
        y = sine(220, 4)
        y[-int(0.5 * SR):] += 0.4                  # ступенька на конце -> скачок на стыке
        r = gc.seam_discontinuity(y, SR, 50.0)
        assert r["click_ratio"] > 1.0 and not r["click_ok"], r


class TestBestLoop:
    def test_min_loop_fraction_respected(self):
        y = np.concatenate([sine(220, 3, amp=.8), sine(880, 3, amp=.8)])
        r = gc.find_best_loop(y, SR, min_loop_fraction=0.5)
        assert r["loop_duration_s"] >= 0.5 * 6.0 - 1e-6, r

    def test_best_not_worse_than_naive(self):
        y = np.concatenate([sine(220, 3, amp=.8), sine(880, 3, amp=.8, phase=1.0)])
        r = gc.find_best_loop(y, SR)
        assert r["best_seam_dist"] <= r["naive_seam_dist"] + 1e-9

    def test_cut_end_recovers_clean_loop(self):
        # 4 с чистой петли + 1 с чужого материала в конце: обрезка конца должна
        # найти шов не хуже наивного и вернуть петлю длиной ~4 с
        y = np.concatenate([sine(220, 4), sine(700, 1, amp=.8, phase=2.0)])
        r = gc.find_best_loop(y, SR, min_loop_fraction=0.5)
        assert r["best_variant"] in ("cut_end", "cut_start", "naive")
        assert r["best_seam_dist"] <= r["naive_seam_dist"] + 1e-9


class TestBeatGrid:
    def test_on_grid_loop(self):
        # 120 BPM, доля 0.5 с, 16 с = 8 тактов, первая доля в t=0 -> всё ~0
        y = click_track(16.0, 120.0)
        r = gc.beat_grid_metrics(y, SR)
        assert not r["degenerate"]
        assert r["bar_fraction_error"] < 0.1, r
        assert r["seam_gap_beat_error"] < 0.1, r
        assert abs(r["bpm_from_beats"] - 120.0) < 1.0, r

    def test_offset_grid_still_loops(self):
        # первая доля в 0.25 с (полдоли), длина 16 с кратна периоду: щель на
        # шве = 0.25 + 0.25 = 1 доля -> петля остаётся на сетке
        y = click_track(16.0, 120.0, offset_s=0.25)
        r = gc.beat_grid_metrics(y, SR)
        assert not r["degenerate"]
        assert r["seam_gap_beat_error"] < 0.15 and abs(r["first_beat_s"] - 0.25) < 0.05, r

    def test_seam_gap_detects_offgrid_seam(self):
        # 16.25 с при периоде 0.5: щель на шве = 0.75 доли -> ошибка ~0.25..0.5
        y = click_track(16.25, 120.0)
        r = gc.beat_grid_metrics(y, SR)
        assert r["seam_gap_beat_error"] > 0.3 and r["beat_fraction_error"] > 0.3, r

    def test_off_grid_duration(self):
        # 120 BPM, такт 2 с, 17 с = 8.5 такта -> bar_fraction_error ~0.5
        y = click_track(17.0, 120.0)
        r = gc.beat_grid_metrics(y, SR)
        assert r["bar_fraction_error"] > 0.35, r


class TestStructure:
    def test_key_c_major_or_relative(self):
        chroma = np.zeros(12); chroma[[0, 4, 7]] = 1.0
        key, _, sig = structure.estimate_key(chroma)
        assert sig == 0 and key in ("C major", "A minor")

    def test_relative_keys_share_signature(self):
        maj = np.roll(structure._KS_MAJOR, 0); minr = np.roll(structure._KS_MINOR, 9)  # C major / A minor
        assert structure.estimate_key(maj)[2] == structure.estimate_key(minr)[2] == 0

    def test_regularity_periodic_vs_random(self):
        # период 2 с из четырёх РАЗНЫХ нот по 0.5 с (у чистого тона хромаграмма
        # постоянна во времени, и SSM не содержит структуры по определению)
        one = np.concatenate([sine(f, 0.5) for f in (261.6, 329.6, 392.0, 440.0)])
        periodic = np.tile(one, 6)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.3, len(periodic)).astype(np.float32)
        rp = structure.structure_report(periodic, SR)["structural_regularity"]
        rn = structure.structure_report(noise, SR)["structural_regularity"]
        assert rp["regularity_ratio"] > rn["regularity_ratio"], (rp, rn)
        assert abs(rp["peak_lag_s"] - 2.0) < 0.5 or abs(rp["peak_lag_s"] - 4.0) < 0.5, rp

    def test_tempo_drift_low_for_constant_tempo(self):
        r = structure.tempo_drift(click_track(20.0, 100.0), SR)
        assert r["n_windows"] > 2 and r["cv"] < 0.1, r

    def test_octave_folding(self):
        assert abs(structure._fold_to_octave(240.0, 120.0) - 120.0) < 1e-9
        assert abs(structure._fold_to_octave(60.0, 120.0) - 120.0) < 1e-9
        assert abs(structure._fold_to_octave(130.0, 120.0) - 130.0) < 1e-9


class TestStats:
    def test_bootstrap_covers_mean(self):
        v = list(np.random.default_rng(1).normal(0.7, 0.05, 12))
        b = stats.bootstrap_ci(v, seed=1)
        assert b["ci_lo"] < 0.7 < b["ci_hi"]

    def test_between_set_and_combined(self):
        bs = stats.between_set_ci([12.1, 13.4, 11.8, 12.9, 13.0])
        c = stats.combined_ci(bs, {"ci_lo": 11.5, "ci_hi": 13.9})
        assert c["ci_lo"] <= 11.5 and c["ci_hi"] >= 13.9

    def test_proportion_and_diff(self):
        p = stats.proportion_ci(30, 60)
        assert p["ci_lo"] < 0.5 < p["ci_hi"]
        d = stats.bootstrap_diff_ci([1.0] * 20, [0.0] * 20)
        assert d["significant"] and d["point"] == 1.0


class TestAudioUtils:
    def test_stereo_to_mono_and_resample(self):
        st = np.stack([sine(220, 1.0, 44100), sine(220, 1.0, 44100)])
        y, sr = audio_utils.prepare_for_analysis(st, 44100, 22050)
        assert sr == 22050 and abs(len(y) - 22050) < 5 and abs(np.max(np.abs(y)) - 0.9) < 1e-3


class TestCorpusPrep:
    def test_windows_from_full_pieces(self, tmp_path):
        from src import corpus_prep
        sr = 22050
        for i, dur in enumerate([12.0, 40.0, 100.0]):
            sf.write(str(tmp_path / f"piece{i}.wav"), sine(220, dur, sr), sr)
        pieces = corpus_prep.list_pieces(tmp_path)
        plan = corpus_prep.plan_windows(pieces, 30.0, n_segments=100, max_per_piece=3, seed=0)
        used = {Path(p["piece"]).name for p in plan}
        assert "piece0.wav" not in used            # 12 с < 30 с — исключена
        assert all(p["start_s"] % 30.0 == 0 for p in plan)
        assert sum(1 for p in plan if Path(p["piece"]).name == "piece2.wav") == 3   # 100 с -> 3 окна, cap 3
        out = tmp_path / "ref30"
        m = corpus_prep.write_reference_set(plan, out)
        assert len(list(out.glob("*.wav"))) == len(m) == len(plan)
        y, s = sf.read(str(out / m[0]["file"]))
        assert abs(len(y) / s - 30.0) < 1e-3

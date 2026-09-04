"""
Игровые критерии пригодности генерации (группа 4, docs/metrics.md).
Реализованы с нуля под эту статью; готовых библиотек нет.

  1. seam_discontinuity — разрыв на шве "конец трека -> начало" (наивная петля)
     относительно распределения внутритрековых переходов того же масштаба.
  2. find_best_loop      — наилучшая ИЗВЛЕКАЕМАЯ петля: обрезка начала или
     конца так, чтобы шов был минимален; петля не короче min_loop_fraction
     исходной длительности (иначе "лучшая петля" вырождается в 2-секундный
     обрывок стационарного участка).
  3. beat_grid_metrics   — темп по beat-трекеру, кратность длительности такту
     (4/4) и доле, и ФАЗОВАЯ ошибка шва относительно сетки долей (длительность
     может быть кратна такту, но первая доля начинается не в t=0).

Вход: моно-сигнал на общей analysis_sr (см. src/audio_utils.prepare_for_analysis),
пиково нормализованный. Стерео-разброс не моделируется (ограничение в статье).
"""
from __future__ import annotations

import numpy as np
import librosa


# --------------------------------------------------------------------------
# Спектральные расстояния между кадрами
# --------------------------------------------------------------------------

def _frame_log_spectra(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """frames: (n_frames, w) -> (n_frames, n_fft//2+1) log1p-магнитуд (Hann)."""
    win = np.hanning(frames.shape[1]).astype(np.float32)
    spec = np.fft.rfft(frames * win, n=n_fft, axis=1)
    return np.log1p(np.abs(spec)).astype(np.float32)


def _spectral_distance(sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Нормированное евклидово расстояние между log-спектрами; поддерживает
    батч (n, F) против (F,) или (n, F) против (n, F)."""
    diff = np.linalg.norm(sa - sb, axis=-1)
    denom = np.linalg.norm(sa, axis=-1) + np.linalg.norm(sb, axis=-1) + 1e-8
    return diff / denom


def _window_samples(sr: int, window_ms: float) -> int:
    return max(32, int(sr * window_ms / 1000))


def _n_fft_for(w: int) -> int:
    return 1 << (2 * w - 1).bit_length()


def _internal_transition_stats(y: np.ndarray, w: int, n_fft: int, n_pairs: int = 200, seed: int = 0) -> dict:
    """Распределение спектральных расстояний между СОСЕДНИМИ внутренними
    окнами размера w (типичная внутритрековая изменчивость). Возвращает
    медиану, p95 и сами значения."""
    n = len(y)
    margin = 2 * w
    if n <= margin * 4:
        return {"median": np.nan, "p95": np.nan, "values": np.array([]), "rms_median": np.nan}
    rng = np.random.default_rng(seed)
    starts = rng.integers(margin, n - margin - 2 * w, size=min(n_pairs, max(1, (n - 2 * margin) // w)))
    f1 = np.stack([y[s:s + w] for s in starts])
    f2 = np.stack([y[s + w:s + 2 * w] for s in starts])
    d = _spectral_distance(_frame_log_spectra(f1, n_fft), _frame_log_spectra(f2, n_fft))
    r1 = np.sqrt(np.mean(f1 ** 2, axis=1) + 1e-12)
    r2 = np.sqrt(np.mean(f2 ** 2, axis=1) + 1e-12)
    rms_jump = np.abs(r1 - r2) / (r1 + r2 + 1e-8)
    return {
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "values": d,
        "rms_median": float(np.median(rms_jump)),
        "rms_p95": float(np.percentile(rms_jump, 95)),
    }


def _click_ratio(y: np.ndarray, seam_signal: np.ndarray, boundary_idx: int, sr: int) -> float:
    """Скачок сигнала на стыке (макс |Δ| в ±1 мс вокруг границы) относительно
    99-го перцентиля |Δ| внутри трека (самые сильные транзиенты самого трека).
    <= 1: щелчок на шве не громче того, что уже есть в треке."""
    d = np.abs(np.diff(seam_signal))
    r = max(1, int(sr * 0.001))
    lo, hi = max(0, boundary_idx - r), min(len(d), boundary_idx + r)
    boundary = float(np.max(d[lo:hi])) if hi > lo else 0.0
    interior = float(np.percentile(np.abs(np.diff(y)), 99))
    return boundary / (interior + 1e-8)


# --------------------------------------------------------------------------
# 1. Шов наивной петли
# --------------------------------------------------------------------------

def seam_discontinuity(y: np.ndarray, sr: int, window_ms: float = 50.0, n_pairs: int = 200, seed: int = 0) -> dict:
    """
    seam_score  = seam_dist / median(internal)  — >1: шов заметнее типичного перехода
    seam_ok_p95 = seam_dist <= p95(internal)     — шов неотличим от 95% собственных переходов
    click_ratio, click_ok                        — щелчок на стыке
    """
    n = len(y)
    w = _window_samples(sr, window_ms)
    if n < 4 * w:
        raise ValueError(f"Трек слишком короткий ({n} сэмплов) для окна {window_ms} мс")
    n_fft = _n_fft_for(w)

    end_frame, start_frame = y[-w:], y[:w]
    spec = _frame_log_spectra(np.stack([end_frame, start_frame]), n_fft)
    seam_dist = float(_spectral_distance(spec[0], spec[1]))

    rms_end = float(np.sqrt(np.mean(end_frame ** 2) + 1e-12))
    rms_start = float(np.sqrt(np.mean(start_frame ** 2) + 1e-12))
    seam_rms = abs(rms_end - rms_start) / (rms_end + rms_start + 1e-8)

    seam_signal = np.concatenate([end_frame, start_frame])
    click = _click_ratio(y, seam_signal, w, sr)

    base = _internal_transition_stats(y, w, n_fft, n_pairs, seed)
    med = base["median"] if np.isfinite(base["median"]) else seam_dist
    p95 = base["p95"] if np.isfinite(base["p95"]) else seam_dist

    return {
        "window_ms": window_ms,
        "seam_spectral_dist": seam_dist,
        "internal_median": med,
        "internal_p95": p95,
        "seam_score": seam_dist / (med + 1e-8),
        "seam_ok_p95": bool(seam_dist <= p95),
        "seam_rms_jump": seam_rms,
        "rms_score": seam_rms / (base["rms_median"] + 1e-8) if np.isfinite(base["rms_median"]) else np.nan,
        "click_ratio": click,
        "click_ok": bool(click <= 1.0),
        "n_internal_pairs": int(len(base["values"])),
    }


# --------------------------------------------------------------------------
# 2. Наилучшая извлекаемая петля (обрезка начала ИЛИ конца)
# --------------------------------------------------------------------------

def find_best_loop(
    y: np.ndarray,
    sr: int,
    window_ms: float = 50.0,
    hop_ms: float = 10.0,
    min_loop_fraction: float = 0.5,
    n_pairs: int = 200,
    seed: int = 0,
) -> dict:
    """
    Вариант A (обрезать начало): петля y[t:], шов = (конец трека -> y[t:t+w]).
    Вариант B (обрезать конец):  петля y[:e], шов = (y[e-w:e] -> начало трека).
    Ограничение: длина петли >= min_loop_fraction * длина трека.
    Возвращает лучший вариант с расстоянием, нормированным на ту же
    внутритрековую статистику, что и seam_discontinuity (сопоставимые score).
    """
    n = len(y)
    w = _window_samples(sr, window_ms)
    hop = max(1, int(sr * hop_ms / 1000))
    n_fft = _n_fft_for(w)
    min_len = int(min_loop_fraction * n)

    base = _internal_transition_stats(y, w, n_fft, n_pairs, seed)
    med = base["median"] if np.isfinite(base["median"]) else 1.0
    p95 = base["p95"] if np.isfinite(base["p95"]) else np.inf

    start_spec = _frame_log_spectra(y[:w][None, :], n_fft)[0]
    end_spec = _frame_log_spectra(y[-w:][None, :], n_fft)[0]
    naive = float(_spectral_distance(end_spec, start_spec))

    best = {"variant": "naive", "cut_sample": 0, "loop_start_s": 0.0, "loop_end_s": n / sr, "dist": naive}

    # A: кандидаты начала t в [hop, n - min_len], кадр y[t:t+w]
    t_max = n - min_len
    if t_max > hop:
        ts = np.arange(hop, t_max - w, hop)
        if len(ts):
            frames = np.stack([y[t:t + w] for t in ts])
            d = _spectral_distance(_frame_log_spectra(frames, n_fft), end_spec)
            i = int(np.argmin(d))
            if d[i] < best["dist"]:
                best = {"variant": "cut_start", "cut_sample": int(ts[i]), "loop_start_s": ts[i] / sr,
                        "loop_end_s": n / sr, "dist": float(d[i])}

    # B: кандидаты конца e в [min_len, n - hop], кадр y[e-w:e]
    e_min = max(min_len, w)
    if n - hop > e_min:
        es = np.arange(n - hop, e_min, -hop)
        if len(es):
            frames = np.stack([y[e - w:e] for e in es])
            d = _spectral_distance(_frame_log_spectra(frames, n_fft), start_spec)
            i = int(np.argmin(d))
            if d[i] < best["dist"]:
                best = {"variant": "cut_end", "cut_sample": int(es[i]), "loop_start_s": 0.0,
                        "loop_end_s": es[i] / sr, "dist": float(d[i])}

    loop_len_s = best["loop_end_s"] - best["loop_start_s"]
    return {
        "naive_seam_dist": naive,
        "naive_seam_score": naive / (med + 1e-8),
        "best_variant": best["variant"],
        "best_cut_sample": best["cut_sample"],
        "loop_start_s": best["loop_start_s"],
        "loop_end_s": best["loop_end_s"],
        "loop_duration_s": loop_len_s,
        "best_seam_dist": best["dist"],
        "best_seam_score": best["dist"] / (med + 1e-8),
        "best_seam_ok_p95": bool(best["dist"] <= p95),
        "improvement_over_naive": naive - best["dist"],
        "min_loop_fraction": min_loop_fraction,
    }


# --------------------------------------------------------------------------
# 3. Тактовая сетка: кратность и фаза шва
# --------------------------------------------------------------------------

def beat_grid_metrics(y: np.ndarray, sr: int, loop_start_s: float = 0.0, loop_end_s: float | None = None,
                      beats_per_bar: int = 4) -> dict:
    """
    Оценивает петлю [loop_start_s, loop_end_s) по сетке долей, найденной
    beat-трекером на ВСЁМ треке (устойчивее, чем на обрезке).

    Период доли и положения крайних долей оцениваются линейной регрессией
    времени доли по её индексу (beat-трекер квантует доли по кадрам ~23 мс;
    медиана интервалов даёт систематическую ошибку, которая на 30+ долях
    накапливается до половины такта).

    bar_fraction_error   = |D/T_bar - round(D/T_bar)|   в [0, 0.5]  (допущение 4/4;
                           положение сильной доли неизвестно, петля может
                           начинаться посреди такта — это НЕ дефект)
    beat_fraction_error  = |D/T_beat - round(D/T_beat)| в [0, 0.5]  (глобальный период)
    seam_gap_beat_error  — то же условие, но ЛОКАЛЬНО на шве: щель
        (конец петли − последняя доля) + (первая доля − начало петли),
        выраженная в долях, должна быть целой. Совпадает с beat_fraction_error
        при идеально ровной сетке и отличается при дрейфе темпа; для длинных
        петель — основной критерий. Постоянный сдвиг первой доли относительно
        t=0 на него не влияет (и не должен: при зацикливании он неслышен).
    Известное ограничение: октавная неоднозначность темпа (x2 / /2) —
    кратность инвариантна к удвоению темпа, но не к половинному.
    """
    duration = len(y) / sr
    if loop_end_s is None:
        loop_end_s = duration
    D = loop_end_s - loop_start_s

    # onset-огибающая считается явно со средним по полосам: beat_track(y=...)
    # внутри агрегирует медианой, и на разреженном сигнале (тишина между
    # ударами) огибающая обнуляется -> 0 долей. trim=False: иначе трекер
    # отбрасывает "слабые" крайние доли, а для фазы шва крайние доли — главное.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    beats_in = beats[(beats >= loop_start_s) & (beats < loop_end_s)]

    if len(beats_in) < 4:
        return {"bpm": tempo, "n_beats": int(len(beats_in)), "bar_fraction_error": np.nan,
                "beat_fraction_error": np.nan, "seam_gap_beat_error": np.nan, "degenerate": True}

    idx = np.arange(len(beats_in))
    period, first = np.polyfit(idx, beats_in, 1)      # наклон = период, пересечение = первая доля
    period = float(period)
    if period <= 0:
        return {"bpm": tempo, "n_beats": int(len(beats_in)), "bar_fraction_error": np.nan,
                "beat_fraction_error": np.nan, "seam_gap_beat_error": np.nan, "degenerate": True}
    last = float(first + period * (len(beats_in) - 1))
    bpm_from_beats = 60.0 / period
    bar = beats_per_bar * period

    r_bar = D / bar
    r_beat = D / period
    gap = (loop_end_s - last) + (float(first) - loop_start_s)
    r_gap = gap / period

    return {
        "bpm": tempo,
        "bpm_from_beats": bpm_from_beats,
        "beat_period_s": period,
        "n_beats": int(len(beats_in)),
        "loop_duration_s": D,
        "n_bars_estimate": r_bar,
        "bar_fraction_error": float(abs(r_bar - round(r_bar))),
        "beat_fraction_error": float(abs(r_beat - round(r_beat))),
        "first_beat_s": float(first),
        "last_beat_s": last,
        "seam_gap_beat_error": float(abs(r_gap - round(r_gap))),
        "degenerate": False,
    }


def game_loop_report(y: np.ndarray, sr: int, min_loop_fraction: float = 0.5) -> dict:
    """Сводный отчёт по треку (уже моно/analysis_sr/нормализован):
    шов наивной петли на двух шкалах, наилучшая извлекаемая петля, тактовые
    метрики для наивной и для извлекаемой петли — раздельно."""
    seam_50 = seam_discontinuity(y, sr, window_ms=50.0)
    seam_200 = seam_discontinuity(y, sr, window_ms=200.0)
    best = find_best_loop(y, sr, window_ms=50.0, min_loop_fraction=min_loop_fraction)
    grid_naive = beat_grid_metrics(y, sr)
    grid_best = beat_grid_metrics(y, sr, best["loop_start_s"], best["loop_end_s"])
    return {
        "actual_duration_s": len(y) / sr,
        "seam_50ms": seam_50,
        "seam_200ms": seam_200,
        "seam_naive_ok": bool(seam_50["seam_ok_p95"] and seam_200["seam_ok_p95"] and seam_50["click_ok"]),
        "best_loop": best,
        "grid_naive": grid_naive,
        "grid_best_loop": grid_best,
    }

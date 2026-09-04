"""
Детектор вырожденного выхода — защита от того, что модель, выдающая тишину,
постоянный тон или белый шум, получит ИДЕАЛЬНЫЕ оценки по группам 3-4.

Проверено эмпирически (2026-09-04) на синтетике: тишина даёт seam_naive_ok=True,
tempo_cv≈0, key_drift=0; постоянный тон — seam_naive_ok=True и
bar_fraction_error=0.039 (проходит порог 0.1). Без этой проверки решающее
правило поощряло бы деградацию: чем меньше в сигнале музыки, тем «лучше» шов,
темп и тональность. Риск не гипотетический — AudioLDM2 на 120 с работает в 12
раз дальше своей обучающей длины (10 с), Riffusion склеивается из 24 клипов.

Пороги калибруются по референсному корпусу (`calibrate` ниже), а не берутся из
воздуха: вырожденным считается трек, выпадающий за нижний перцентиль реальной
ретро-игровой музыки. Это та же логика самокалибрующегося порога, что и у шва
(p95 внутритрековых переходов).

Вырожденный трек в решающем правиле считается НЕ ПРОШЕДШИМ (а не исключается):
модель не справилась с задачей, и исключение позволило бы модели с 90 %
тишины отчитаться по оставшимся 10 %.
"""
from __future__ import annotations

import numpy as np
import librosa

# Значения по умолчанию до калибровки на референсе. Консервативные: срабатывают
# только на явно вырожденном сигнале. После `calibrate` заменяются перцентилями
# референсного корпуса.
# Проверено на синтетике (см. tests): различающая способность признаков —
#   spectral_flux:     тишина 0.00, постоянный тон 0.011 | мелодия 0.69, аккорд+удары 2.7
#   spectral_flatness: белый шум 0.56                    | тональная музыка <= 0.001
#   orig_peak_dbfs:    почти тишина -57                  | нормальный выход > -20
# dynamic_range_db как критерий НЕ используется: у аккорда с ровными ударами он
# 1.3 дБ при 6.9 у мелодии, а чиптюн NES с постоянной амплитудой квадратной
# волны динамически сжат по своей природе — это дало бы массовые ложные
# срабатывания. Признак пишется в отчёт, но в правило не входит (калибруется
# отдельно, если референс покажет, что он разделяет).
# onset_rate тоже не в правиле: на постоянном тоне детектор онсетов даёт
# ложные 9.3 события/с (реагирует на шум), т.е. признак не защищает от
# статичного сигнала.
DEFAULT_THRESHOLDS = {
    "peak_dbfs_min": -40.0,        # исходный (до нормализации) пик тише -40 dBFS
    "silent_frac_max": 0.90,       # доля кадров тише -50 dBFS
    "spectral_flatness_max": 0.30, # ближе к 1 — белый шум, а не музыка
    "spectral_flux_min": 0.05,     # ~0 — статичный сигнал (постоянный тон/гудок)
}


def _frame_rms_db(y: np.ndarray, sr: int, frame_ms: float = 50.0) -> np.ndarray:
    n = max(64, int(sr * frame_ms / 1000))
    rms = librosa.feature.rms(y=y, frame_length=n, hop_length=n // 2)[0]
    return 20 * np.log10(np.maximum(rms, 1e-12))


def features(y: np.ndarray, sr: int, orig_peak: float | None = None,
             orig_rms: float | None = None) -> dict:
    """
    y — моно-сигнал ПОСЛЕ приведения к анализу (нормализованный).
    orig_peak / orig_rms — линейные пик и RMS ДО нормализации (из манифеста
    генерации): без них near-silence не отличить, т.к. нормализация усиливает
    сигнал -57 dBFS до полной шкалы.
    """
    dur = len(y) / sr
    db = _frame_rms_db(y, sr)
    # абсолютный порог по нормализованному сигналу (пик приведён к 0.9), а не
    # относительно собственного пика: у полной тишины все кадры равны, и
    # относительный порог давал бы silent_frac = 0 на чистом нуле
    silent_frac = float(np.mean(db < -50.0)) if db.size else 1.0
    dyn_range = float(np.percentile(db, 95) - np.percentile(db, 5)) if db.size else 0.0

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_rate = float(len(onsets) / dur) if dur > 0 else 0.0

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    # спектральный поток: средняя положительная покадровая разность
    # нормированного логспектра; ~0 у статичного сигнала
    logS = np.log1p(S)
    logS = logS / (np.linalg.norm(logS, axis=0, keepdims=True) + 1e-8)
    flux = float(np.mean(np.maximum(np.diff(logS, axis=1), 0).sum(axis=0))) if logS.shape[1] > 1 else 0.0

    out = {
        "duration_s": dur,
        "silent_frac": silent_frac,
        "dynamic_range_db": dyn_range,
        "onset_rate_hz": onset_rate,
        "spectral_flatness": flatness,
        "spectral_flux": flux,
    }
    if orig_peak is not None:
        out["orig_peak_dbfs"] = float(20 * np.log10(max(orig_peak, 1e-12)))
    if orig_rms is not None:
        out["orig_rms_dbfs"] = float(20 * np.log10(max(orig_rms, 1e-12)))
    return out


def is_degenerate(feats: dict, thresholds: dict | None = None) -> dict:
    """-> {"degenerate": bool, "reasons": [...]}. Каждая причина фиксируется
    отдельно: в статье важно, ЧЕМ именно выродился выход модели."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons = []
    if "orig_peak_dbfs" in feats and feats["orig_peak_dbfs"] < t["peak_dbfs_min"]:
        reasons.append("silent_output")
    if feats["silent_frac"] > t["silent_frac_max"]:
        reasons.append("mostly_silence")
    if feats["spectral_flatness"] > t["spectral_flatness_max"]:
        reasons.append("noise_like")
    if feats["spectral_flux"] < t["spectral_flux_min"]:
        reasons.append("static_signal")
    return {"degenerate": bool(reasons), "reasons": reasons}


def report(y: np.ndarray, sr: int, orig_peak: float | None = None,
           orig_rms: float | None = None, thresholds: dict | None = None) -> dict:
    f = features(y, sr, orig_peak, orig_rms)
    return {**f, **is_degenerate(f, thresholds)}


def calibrate(reference_features: list[dict], q: float = 1.0) -> dict:
    """
    Пороги как q-й (по умолчанию 1-й) перцентиль по референсному корпусу:
    вырожденным считается то, что лежит ниже почти всей реальной музыки NES.
    Для «максимальных» порогов (flatness) берётся (100-q)-й перцентиль.
    Уровневый порог peak_dbfs_min не калибруется — референс нормализован.
    """
    def pct(key, p):
        vals = [f[key] for f in reference_features if np.isfinite(f.get(key, np.nan))]
        return float(np.percentile(vals, p)) if vals else DEFAULT_THRESHOLDS.get(key)

    return {
        "peak_dbfs_min": DEFAULT_THRESHOLDS["peak_dbfs_min"],
        "silent_frac_max": pct("silent_frac", 100 - q),
        "spectral_flatness_max": pct("spectral_flatness", 100 - q),
        "spectral_flux_min": pct("spectral_flux", q),
        "_calibration": {"n_reference": len(reference_features), "percentile": q,
                          "note": "dynamic_range_db и onset_rate_hz пишутся в отчёт, но в правило не входят"},
    }

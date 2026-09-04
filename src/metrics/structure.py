"""
Группа 3: структура (docs/metrics.md).

Все признаки (хромаграмма, onset-огибающая) считаются ОДИН раз на весь трек
и нарезаются по окнам — пересчёт CQT на каждое окно был бы в десятки раз
дороже и давал бы краевые артефакты.

- structural_regularity: профиль средней схожести по лагам матрицы
  самоподобия (SSM) в диапазоне [min_lag_s, n/2]; пик/среднее. Малые лаги
  (<1 с) исключены: соседние кадры тривиально похожи, и без этого исключения
  "пик" всегда оказывался на лаге 1 (смежные кадры), т.е. метрика измеряла
  гладкость признаков, а не периодичность.
- tempo_drift: CV локального темпа по скользящему окну; локальные оценки
  предварительно сворачиваются в октаву глобального темпа (x2 / /2), иначе
  CV определяется октавными скачками детектора, а не дрейфом.
- key_drift: доля окон, чья тональность отличается от глобальной; относительные
  мажор/минор (одна ключевая сигнатура) считаются одной тональностью.
"""
from __future__ import annotations

import numpy as np
import librosa

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(chroma_vector: np.ndarray) -> tuple[str, float, int]:
    """-> (имя тональности, корреляция, индекс ключевой сигнатуры 0..11).
    Индекс сигнатуры = тоника относительного мажора; A minor и C major дают 0."""
    if np.std(chroma_vector) < 1e-8:
        return "undetermined", 0.0, -1
    best_score, best_key, best_sig = -np.inf, "undetermined", -1
    for shift in range(12):
        s_maj = np.corrcoef(chroma_vector, np.roll(_KS_MAJOR, shift))[0, 1]
        s_min = np.corrcoef(chroma_vector, np.roll(_KS_MINOR, shift))[0, 1]
        if s_maj > best_score:
            best_score, best_key, best_sig = s_maj, f"{_PC[shift]} major", shift
        if s_min > best_score:
            best_score, best_key, best_sig = s_min, f"{_PC[shift]} minor", (shift + 3) % 12
    return best_key, float(best_score), best_sig


def chroma_frames(y: np.ndarray, sr: int, hop_length: int = 2048) -> tuple[np.ndarray, int]:
    """(12, n_frames) при ~10 кадрах/с на 22 050 Гц. hop кратен 64 (требование CQT)."""
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length), hop_length


def self_similarity_matrix(feat: np.ndarray) -> np.ndarray:
    f = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-8)
    # np.errstate: Accelerate BLAS на macOS выставляет FP-флаги divide-by-zero /
    # overflow / invalid даже на заведомо конечном входе (проверено: warning
    # воспроизводится на случайной конечной матрице, результат при этом
    # конечен). Флаги гасятся, но конечность результата проверяется явно, чтобы
    # реальный NaN не проскочил вместе с ложной тревогой.
    with np.errstate(all="ignore"):
        ssm = (f.T @ f).astype(np.float32)
    if not np.isfinite(ssm).all():
        ssm = np.nan_to_num(ssm, nan=0.0, posinf=1.0, neginf=-1.0)
    return ssm


def structural_regularity(ssm: np.ndarray, frames_per_s: float, min_lag_s: float = 1.0) -> dict:
    n = ssm.shape[0]
    min_lag = max(1, int(round(min_lag_s * frames_per_s)))
    max_lag = n // 2
    if max_lag <= min_lag + 2:
        return {"regularity_ratio": np.nan, "peak_lag_s": np.nan, "n_lags": 0}
    lags = np.arange(min_lag, max_lag)
    profile = np.array([np.trace(ssm, offset=int(k)) / (n - k) for k in lags])
    peak_i = int(np.argmax(profile))
    return {
        "regularity_ratio": float(profile[peak_i] / (np.mean(profile) + 1e-8)),
        "peak_lag_s": float(lags[peak_i] / frames_per_s),
        "n_lags": int(len(lags)),
    }


def _fold_to_octave(local: float, reference: float) -> float:
    """Сворачивает локальный темп в октаву референсного (x2, /2, пока ближе)."""
    if local <= 0 or reference <= 0:
        return local
    while local / reference > np.sqrt(2):
        local /= 2
    while local / reference < 1 / np.sqrt(2):
        local *= 2
    return local


def tempo_drift(y: np.ndarray, sr: int, window_s: float = 5.0, hop_s: float = 1.0, hop_length: int = 512) -> dict:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    fps = sr / hop_length
    global_bpm = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length,
                                             aggregate=np.median)[0])
    win, hop = int(window_s * fps), int(hop_s * fps)
    if len(onset_env) < 2 * win:
        return {"cv": np.nan, "global_bpm": global_bpm, "n_windows": 0}
    tempos = []
    for s in range(0, len(onset_env) - win, hop):
        t = librosa.feature.tempo(onset_envelope=onset_env[s:s + win], sr=sr, hop_length=hop_length,
                                  aggregate=np.median)[0]
        if t > 0:
            tempos.append(_fold_to_octave(float(t), global_bpm))
    if len(tempos) < 2:
        return {"cv": np.nan, "global_bpm": global_bpm, "n_windows": len(tempos)}
    tempos = np.array(tempos)
    return {
        "cv": float(np.std(tempos) / (np.mean(tempos) + 1e-8)),
        "global_bpm": global_bpm,
        "mean_local_bpm": float(np.mean(tempos)),
        "std_local_bpm": float(np.std(tempos)),
        "n_windows": int(len(tempos)),
    }


def key_drift(chroma: np.ndarray, frames_per_s: float, window_s: float = 5.0, hop_s: float = 2.5) -> dict:
    global_key, conf, global_sig = estimate_key(chroma.mean(axis=1))
    win, hop = int(window_s * frames_per_s), int(hop_s * frames_per_s)
    n = chroma.shape[1]
    if n < 2 * win or win < 1:
        return {"global_key": global_key, "global_key_confidence": conf, "drift_fraction": np.nan, "n_windows": 0}
    mism, total = 0, 0
    for s in range(0, n - win, hop):
        _, _, sig = estimate_key(chroma[:, s:s + win].mean(axis=1))
        total += 1
        if sig != global_sig:
            mism += 1
    return {
        "global_key": global_key,
        "global_key_confidence": conf,
        "drift_fraction": mism / total if total else np.nan,
        "n_windows": total,
    }


def structure_report(y: np.ndarray, sr: int) -> dict:
    chroma, hop = chroma_frames(y, sr)
    fps = sr / hop
    ssm = self_similarity_matrix(chroma)
    return {
        "structural_regularity": structural_regularity(ssm, fps),
        "tempo_drift": tempo_drift(y, sr),
        "key_drift": key_drift(chroma, fps),
    }

"""
Общая подготовка аудио для метрик групп 3-4 и для записи wav.

Все модели отдают разные форматы (моно 16/32/44.1 кГц, стерео 44.1/48 кГц,
разная громкость, пики > 1.0). Чтобы per-track метрики были сопоставимы между
моделями, каждый трек приводится к одному виду:
  моно -> ресемплинг на analysis_sr -> пиковая нормализация.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(samples,) | (channels, samples) | (samples, channels) -> (samples,)"""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.ndim != 2:
        raise ValueError(f"неожиданная размерность аудио: {a.shape}")
    # каналов заведомо мало (1-2), сэмплов много: меньшая ось = каналы
    if a.shape[0] <= 8 and a.shape[0] < a.shape[1]:
        return a.mean(axis=0)
    return a.mean(axis=1)


def peak_normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if m < 1e-8:
        return a
    return a * (peak / m)


def prepare_for_analysis(audio: np.ndarray, sr: int, analysis_sr: int = 22050, peak: float = 0.9) -> tuple[np.ndarray, int]:
    y = to_mono(audio)
    if sr != analysis_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=analysis_sr, res_type="soxr_hq")
    return peak_normalize(y, peak), analysis_sr


def load_for_analysis(path: Path, analysis_sr: int = 22050, peak: float = 0.9) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    return prepare_for_analysis(y.T, sr, analysis_sr, peak)


def write_wav(path: Path, audio: np.ndarray, sr: int, peak: float = 0.9) -> dict:
    """Пишет wav (PCM_24) и возвращает длительность вместе с уровнями ДО
    нормализации. Уровни обязаны сохраняться здесь: после пиковой нормализации
    трек с исходным пиком -57 dBFS неотличим от нормального, и детектор
    вырожденного выхода (metrics/degeneracy.py) без них не работает."""
    a = np.asarray(audio, dtype=np.float32)
    orig_peak = float(np.max(np.abs(a))) if a.size else 0.0
    orig_rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    a = peak_normalize(a, peak)
    data = a.T if a.ndim == 2 else a
    path.parent.mkdir(parents=True, exist_ok=True)
    # FLAC сжимает без потерь примерно вдвое и поддерживает только целочисленные
    # форматы; для архивного хранения этого достаточно (сигнал уже нормализован)
    subtype = "PCM_16" if path.suffix.lower() == ".flac" else "PCM_24"
    sf.write(str(path), data, sr, subtype=subtype)
    return {"actual_duration_s": data.shape[0] / sr, "orig_peak": orig_peak, "orig_rms": orig_rms}

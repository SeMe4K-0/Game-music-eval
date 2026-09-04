"""
Группа 2: CLAP Score — косинусное сходство эмбеддингов промпта и аудио
(laion/larger_clap_music).

Два факта, проверенных по документации transformers (ClapFeatureExtractor):
  1. Экстрактор ожидает 48 000 Гц и НЕ ресемплирует сам (параметр
     sampling_rate "only serves to warn"). Ресемплируем здесь явно.
  2. max_length_s = 10, truncation="fusion": для клипов длиннее 10 с модель
     видит 3 случайных 10-секундных фрагмента + даунсэмплированный мел всего
     клипа. То есть CLAP score 120-секундного трека — это оценка по
     фрагментам, а не по всему треку. Это фиксируется в статье как свойство
     метрики. Для fusion-чекпойнта обязательно передавать is_longer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa


@dataclass
class ClapScoreResult:
    text: str
    audio_path: str
    score: float


class ClapScorer:
    def __init__(self, model_id: str = "laion/larger_clap_music", device: str | None = None, seed: int = 0):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.target_sr = int(self.processor.feature_extractor.sampling_rate)  # 48000
        self._torch = torch
        self._seed = seed  # fusion выбирает СЛУЧАЙНЫЕ фрагменты — фиксируем генератор

    def score(self, audio_mono: np.ndarray, sr: int, text: str, audio_path: str = "") -> ClapScoreResult:
        torch = self._torch
        y = np.asarray(audio_mono, dtype=np.float32)
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr, res_type="soxr_hq")

        np.random.seed(self._seed)  # _random_mel_fusion использует numpy RNG
        inputs = self.processor(text=[text], audio=[y], sampling_rate=self.target_sr,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            t = self.model.get_text_features(input_ids=inputs["input_ids"],
                                             attention_mask=inputs.get("attention_mask"))
            a = self.model.get_audio_features(input_features=inputs["input_features"],
                                              is_longer=inputs.get("is_longer"))
        t = t / t.norm(dim=-1, keepdim=True)
        a = a / a.norm(dim=-1, keepdim=True)
        return ClapScoreResult(text=text, audio_path=audio_path, score=float((t @ a.T).item()))

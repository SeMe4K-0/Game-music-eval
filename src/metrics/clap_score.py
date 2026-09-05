"""
Группа 2: CLAP Score — косинусное сходство эмбеддингов промпта и аудио
(laion/larger_clap_music).

Факты, проверенные ЗАПУСКОМ на transformers 4.50.0 (2026-09-05), а не по
документации — прежняя редакция этого файла опиралась на описание fusion-режима
и была неверна для взятого чекпойнта:

  1. Экстрактор ожидает 48 000 Гц и НЕ ресемплирует сам (параметр
     sampling_rate "only serves to warn"). Ресемплируем здесь явно.
  2. `laion/larger_clap_music` — НЕ fusion-чекпойнт: в его конфиге
     `enable_fusion=False`, а свёртка аудио-энкодера принимает 1 канал.
     С truncation="fusion" экстрактор отдаёт 4 мел-канала, и модель падает
     ("expected input[1, 4, 256, 256] to have 1 channels"). Правильный режим —
     `rand_trunc`: один случайный 10-секундный фрагмент.
  3. Отсюда главное свойство метрики для статьи: **CLAP score трека любой
     длины считается по одному случайному 10-секундному окну**, а не по всему
     треку. Фрагмент выбирается numpy-генератором, поэтому сид фиксируется —
     иначе оценка одного и того же трека плавает между запусками.
  4. Текст и аудио обрабатываются РАЗДЕЛЬНО (tokenizer и feature_extractor),
     а не одним вызовом процессора: `truncation` у процессора уходит сразу в
     оба, и токенизатор падает на значении "rand_trunc", которого не знает.
  5. `get_text_features`/`get_audio_features` возвращают уже нормированные
     векторы (проверено: норма 1.0), но нормировку оставляем явной — она
     ничего не стоит и не зависит от версии.

Абсолютные значения косинуса у этого чекпойнта малы (0.001-0.01), но порядок
устойчив: на 8 из 8 треков NES-VMDB «8-bit chiptune retro video game music»
опережает «оперу», «грозу» и «лай собаки». Решающее правило сравнивает модели
между собой на одном промпте, поэтому абсолютная шкала роли не играет.
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
        self._seed = seed  # окно выбирается случайно — фиксируем генератор
        # Режим усечения определяется чекпойнтом, а не задаётся на глаз:
        # fusion-модель ждёт 4 мел-канала, обычная — 1, и перепутать нельзя.
        self._fusion = bool(getattr(self.model.config.audio_config, "enable_fusion", False))
        self._truncation = "fusion" if self._fusion else "rand_trunc"

    def score(self, audio_mono: np.ndarray, sr: int, text: str, audio_path: str = "") -> ClapScoreResult:
        torch = self._torch
        y = np.asarray(audio_mono, dtype=np.float32)
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr, res_type="soxr_hq")

        np.random.seed(self._seed)   # выбор 10-секундного окна идёт numpy RNG
        af = self.processor.feature_extractor(
            [y], sampling_rate=self.target_sr, truncation=self._truncation,
            return_tensors="pt")
        tf = self.processor.tokenizer([text], return_tensors="pt", padding=True)
        af = {k: v.to(self.device) for k, v in af.items()}
        tf = {k: v.to(self.device) for k, v in tf.items()}

        with torch.no_grad():
            t = self.model.get_text_features(input_ids=tf["input_ids"],
                                             attention_mask=tf.get("attention_mask"))
            # is_longer имеет смысл только для fusion-чекпойнта; для нашего он
            # приходит True даже при одном канале и модель его игнорирует.
            a = (self.model.get_audio_features(input_features=af["input_features"],
                                               is_longer=af.get("is_longer"))
                 if self._fusion else
                 self.model.get_audio_features(input_features=af["input_features"]))
        t = t / t.norm(dim=-1, keepdim=True)
        a = a / a.norm(dim=-1, keepdim=True)
        return ClapScoreResult(text=text, audio_path=audio_path, score=float((t @ a.T).item()))

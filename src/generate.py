"""
Генерация: единый интерфейс поверх пяти бэкендов + кэш загруженных моделей.

Исправления относительно первой версии (ревизия 2026-09-04):
  * Модель грузится ОДИН раз на прогон (ModelRegistry), а не на каждый трек —
    иначе 10 000+ загрузок весов по несколько ГБ.
  * MusicGen: настоящая continuation (аудио-промпт через processor(audio=...)),
    а не независимые чанки; лимит 30 с на окно ВКЛЮЧАЯ промпт (1503 токена).
  * Riffusion: реальный API (riffuse -> PIL image -> SpectrogramImageConverter
    -> pydub), интерполяция alpha по цепочке клипов с общим seed-image.
  * ACE-Step: реальный API (manual_seeds строкой, save_path, lyrics="[inst]").
  * Stable Audio Open: D > 47 с — NotApplicable (у модели нет канонического
    механизма продолжения), а не тихий обрез до 47 с под чужой меткой.
  * Возвращается ФАКТИЧЕСКАЯ длительность; все метрики используют её.

API тулкитов сверены с исходниками 2026-09-04; помеченные TODO места —
перепроверить с установленными версиями в фазе 0.
"""
from __future__ import annotations

import gc
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


class NotApplicable(Exception):
    """Модель не может выдать запрошенную длительность нативно."""


@dataclass
class GenerationResult:
    model_id: str
    prompt: str
    requested_duration_s: float
    seed: int
    audio: np.ndarray            # (samples,) или (channels, samples)
    sr: int
    wall_time_s: float
    peak_vram_gb: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def actual_duration_s(self) -> float:
        n = self.audio.shape[-1]
        return n / self.sr


def load_models_config(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(cfg: dict):
    """Тип из конфига, с проверкой поддержки железом (см. src/env_info.py)."""
    from .env_info import resolve_dtype
    return resolve_dtype(cfg.get("dtype", "float16"))


# --------------------------------------------------------------------------
# Бэкенды
# --------------------------------------------------------------------------

class MusicGenBackend:
    def __init__(self, cfg: dict):
        import torch
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        self.cfg = cfg
        self.device = _device()
        self.processor = AutoProcessor.from_pretrained(cfg["hf_repo"])
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            cfg["hf_repo"], torch_dtype=_dtype(cfg)).to(self.device)
        self.sr = int(self.model.config.audio_encoder.sampling_rate)
        self.tps = int(cfg.get("tokens_per_second", 50))
        self.dtype = _dtype(cfg)

    def _to_model(self, inputs):
        """Фаза 0 (2026-09-04, transformers 4.50.0): у continuation-вызова
        `processor(audio=...)` отдаёт `input_values` во float32 независимо от
        точности модели, а `BatchEncoding.to(device)` тип не приводит. Свёртка
        энкодера EnCodec в fp16 падает на float32-входе ("Input type (float)
        and bias type (struct c10::Half) should be the same"). Приводим только
        вещественные тензоры: `input_ids`/`padding_mask` обязаны остаться
        целочисленными. Точность модели при этом остаётся ровно той, что
        прибита в configs/models.yaml."""
        inputs = inputs.to(self.device)
        for k, v in inputs.items():
            if hasattr(v, "is_floating_point") and v.is_floating_point():
                inputs[k] = v.to(self.dtype)
        return inputs

    def _gen(self, inputs, seconds: float):
        import torch
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=True, guidance_scale=3,
                                      max_new_tokens=int(seconds * self.tps))
        return out[0, 0].float().cpu().numpy()

    def generate(self, prompt: str, duration_s: float, seed: int):
        import torch
        torch.manual_seed(seed)
        native = float(self.cfg["native_max_duration_s"])
        prompt_s = float(self.cfg.get("continuation_prompt_s", 10))
        sr = self.sr

        inputs = self._to_model(self.processor(text=[prompt], padding=True, return_tensors="pt"))
        audio = self._gen(inputs, min(duration_s, native))

        while len(audio) / sr < duration_s - 0.05:
            tail = audio[-int(prompt_s * sr):]
            new_s = min(native - prompt_s, duration_s - len(audio) / sr)
            inputs = self._to_model(self.processor(audio=tail, sampling_rate=sr, text=[prompt],
                                                   padding=True, return_tensors="pt"))
            gen = self._gen(inputs, new_s)
            # transformers возвращает промпт + продолжение; если длина выхода
            # заметно больше запрошенного нового куска — отрезаем промпт.
            new_part = gen[len(tail):] if len(gen) > 1.5 * new_s * sr else gen
            if len(new_part) < int(0.5 * sr):
                break  # защита от зацикливания при пустом продолжении
            audio = np.concatenate([audio, new_part])
        return audio[:int(duration_s * sr)], sr


class AudioLDM2Backend:
    def __init__(self, cfg: dict):
        import torch
        from diffusers import AudioLDM2Pipeline
        self.cfg = cfg
        self.device = _device()
        self.pipe = AudioLDM2Pipeline.from_pretrained(cfg["hf_repo"], torch_dtype=_dtype(cfg)).to(self.device)
        self.sr = int(self.pipe.vocoder.config.sampling_rate)

    def generate(self, prompt: str, duration_s: float, seed: int):
        import torch
        g = torch.Generator(self.device).manual_seed(seed)
        audio = self.pipe(prompt=prompt, audio_length_in_s=float(duration_s),
                          num_inference_steps=int(self.cfg.get("num_inference_steps", 200)),
                          generator=g).audios[0]
        return np.asarray(audio, dtype=np.float32), self.sr


class RiffusionBackend:
    """riffusion-hobby. Один клип = 512 px x 10 мс = 5.12 с. Длинная генерация
    — цепочка клипов с интерполяцией alpha 0->1 между двумя PromptInput при
    общем seed-image (img2img от og_beat) — канонический сценарий riffusion."""

    def __init__(self, cfg: dict):
        import torch
        from PIL import Image
        from riffusion.riffusion_pipeline import RiffusionPipeline
        from riffusion.spectrogram_image_converter import SpectrogramImageConverter
        from riffusion.spectrogram_params import SpectrogramParams
        self.cfg = cfg
        self.device = _device()
        self.pipe = RiffusionPipeline.load_checkpoint(
            checkpoint=cfg["hf_repo"], dtype=_dtype(cfg), device=self.device)
        self.params = SpectrogramParams()           # 44100 Гц, моно, <=10 кГц
        self.sr = int(self.params.sample_rate)
        self.converter = SpectrogramImageConverter(params=self.params, device=self.device)
        seed_img = cfg.get("seed_image_path")
        if not seed_img:
            raise FileNotFoundError("models.yaml: riffusion_v1.seed_image_path (seed_images/og_beat.png из репозитория riffusion-hobby)")
        self.seed_image = Image.open(seed_img).convert("RGB")

    def generate(self, prompt: str, duration_s: float, seed: int):
        from riffusion.datatypes import InferenceInput, PromptInput
        native = float(self.cfg["native_max_duration_s"])
        n_chunks = max(1, int(np.ceil(duration_s / native)))
        den, gui = float(self.cfg.get("denoising", 0.75)), float(self.cfg.get("guidance", 7.0))
        start = PromptInput(prompt=prompt, seed=seed, denoising=den, guidance=gui)
        end = PromptInput(prompt=prompt, seed=seed + 1, denoising=den, guidance=gui)
        chunks = []
        for i in range(n_chunks):
            alpha = i / (n_chunks - 1) if n_chunks > 1 else 0.0
            inp = InferenceInput(alpha=alpha, num_inference_steps=int(self.cfg.get("num_inference_steps", 50)),
                                 seed_image_id=self.cfg.get("seed_image_id", "og_beat"), start=start, end=end)
            image = self.pipe.riffuse(inp, init_image=self.seed_image)
            seg = self.converter.audio_from_spectrogram_image(image)
            arr = np.array(seg.get_array_of_samples(), dtype=np.float32) / float(2 ** (8 * seg.sample_width - 1))
            if seg.channels > 1:
                arr = arr.reshape(-1, seg.channels).mean(axis=1)
            chunks.append(arr)
        audio = np.concatenate(chunks)[:int(duration_s * self.sr)]
        return audio, self.sr


class StableAudioOpenBackend:
    def __init__(self, cfg: dict):
        import torch
        from diffusers import StableAudioPipeline
        self.cfg = cfg
        self.device = _device()
        self.pipe = StableAudioPipeline.from_pretrained(cfg["hf_repo"], torch_dtype=_dtype(cfg)).to(self.device)
        self.sr = int(getattr(self.pipe.vae.config, "sampling_rate", cfg["sample_rate"]))

    def generate(self, prompt: str, duration_s: float, seed: int):
        import torch
        if duration_s > float(self.cfg["native_max_duration_s"]) + 1e-6:
            raise NotApplicable(f"Stable Audio Open: {duration_s}s > нативного максимума {self.cfg['native_max_duration_s']}s")
        g = torch.Generator(self.device).manual_seed(seed)
        audio = self.pipe(prompt=prompt, negative_prompt="low quality, noisy",
                          num_inference_steps=int(self.cfg.get("num_inference_steps", 100)),
                          audio_end_in_s=float(duration_s), num_waveforms_per_prompt=1, generator=g).audios[0]
        return np.asarray(audio, dtype=np.float32), self.sr   # (2, samples)


class ACEStepBackend:
    """ACE-Step v1-3.5B. Пайплайн пишет файл по save_path и не возвращает
    массив; читаем файл обратно. Значения guidance/scheduler — дефолты infer.py
    репозитория (TODO фаза 0: сверить с установленной версией)."""

    def __init__(self, cfg: dict):
        from acestep.pipeline_ace_step import ACEStepPipeline
        self.cfg = cfg
        self.device = _device()
        _dtype(cfg)   # проверка поддержки железом до загрузки весов
        self.pipe = ACEStepPipeline(checkpoint_dir=cfg.get("checkpoint_dir", ""),
                                    dtype=cfg.get("dtype", "bfloat16"),
                                    torch_compile=False, cpu_offload=bool(cfg.get("cpu_offload", False)),
                                    overlapped_decode=False)
        self.tmp = Path(tempfile.mkdtemp(prefix="acestep_"))

    def generate(self, prompt: str, duration_s: float, seed: int):
        out = self.tmp / f"gen_{seed}.wav"
        self.pipe(audio_duration=float(duration_s), prompt=prompt, lyrics="[inst]",
                  infer_step=int(self.cfg.get("infer_step", 27)), guidance_scale=15.0,
                  scheduler_type="euler", cfg_type="apg", omega_scale=10.0, manual_seeds=str(seed),
                  guidance_interval=0.5, guidance_interval_decay=0.0, min_guidance_scale=3.0,
                  use_erg_tag=True, use_erg_lyric=True, use_erg_diffusion=True, oss_steps="",
                  guidance_scale_text=0.0, guidance_scale_lyric=0.0, save_path=str(out))
        audio, sr = sf.read(str(out), dtype="float32", always_2d=True)   # (samples, ch)
        out.unlink(missing_ok=True)
        return audio.T, int(sr)


_BACKENDS = {
    "transformers_musicgen": MusicGenBackend,
    "diffusers_audioldm2": AudioLDM2Backend,
    "riffusion": RiffusionBackend,
    "diffusers_stable_audio": StableAudioOpenBackend,
    "ace_step": ACEStepBackend,
}


# --------------------------------------------------------------------------
# Реестр + профилирование
# --------------------------------------------------------------------------

class ModelRegistry:
    def __init__(self):
        self._cache: dict[str, object] = {}

    def get(self, cfg: dict):
        mid = cfg["id"]
        if mid not in self._cache:
            self.release()  # на 12 ГБ держим одну модель за раз
            self._cache[mid] = _BACKENDS[cfg["backend"]](cfg)
        return self._cache[mid]

    def release(self):
        self._cache.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def generate(registry: ModelRegistry, cfg: dict, prompt: str, duration_s: float, seed: int) -> GenerationResult:
    backend = registry.get(cfg)
    peak = None
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        cuda = False
    t0 = time.time()
    audio, sr = backend.generate(prompt, duration_s, seed)
    wall = time.time() - t0
    if cuda:
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return GenerationResult(model_id=cfg["id"], prompt=prompt, requested_duration_s=duration_s, seed=seed,
                            audio=np.asarray(audio, dtype=np.float32), sr=int(sr), wall_time_s=wall,
                            peak_vram_gb=peak, meta={"backend": cfg["backend"], "hf_repo": cfg["hf_repo"]})

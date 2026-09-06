"""
Нижние границы (полы) для FAD/KAD на данном референсе и эмбеддере.

Три вида пола — все считаются как FAD/KAD(референс, X):
  1. split_half — X = вторая половина референса (по разным партитурам).
     Пол конечной выборки: то, что получил бы ИДЕАЛЬНЫЙ генератор. Самый
     важный и самый дешёвый; без него абсолютные FAD неинтерпретируемы.
  2. codec round-trip — X = референс, пропущенный через кодек/VAE модели
     туда-обратно. Потолок представления: ниже него данная модель опуститься
     не может. У каждой модели СВОЁ узкое горлышко, поэтому пол per-model:
       musicgen  -> EnCodec 32 kHz (2.2 kbps, 4 кодбука)        [реализовано]
       stable_audio_open -> Oobleck VAE (diffusers, subfolder vae) [реализовано]
       riffusion -> мел-спектрограмма 512 бинов (<=10 кГц) + Griffin-Lim [реализовано]
       audioldm2 -> мел -> VAE -> мел -> HiFi-GAN вокодер           [TODO фаза 0]
       ace_step  -> DCAE + вокодер                                  [TODO фаза 0]
     Единый EnCodec-пол для всех моделей был бы полом только для MusicGen.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def split_half_indices(files: Iterable[Path], piece_of: dict[str, str]) -> tuple[list[int], list[int]]:
    """Те же половины, что и split_half, но БЕЗ копирования файлов — только
    номера позиций в исходном списке.

    Копирование в отдельные папки заставляло считать эмбеддинги повторно: кэш
    fadtk привязан к пути файла, а половины — это те же самые аудиофайлы под
    новыми именами. На референсе это ровно удвоенная работа, самая дорогая
    часть расчёта FAD. Хеш партитуры и правило чётности совпадают со
    split_half, поэтому разбиение то же самое.
    """
    idx_a: list[int] = []
    idx_b: list[int] = []
    for i, f in enumerate(files):
        h = int(hashlib.md5(piece_of[Path(f).name].encode()).hexdigest(), 16)
        (idx_a if h % 2 == 0 else idx_b).append(i)
    return idx_a, idx_b


def split_half(files: Iterable[Path], piece_of: dict[str, str], out_a: Path, out_b: Path) -> tuple[int, int]:
    """Делит референс на две половины по ПАРТИТУРАМ (piece_of: имя файла ->
    id партитуры), чтобы окна одной партитуры не попали в обе половины."""
    na = nb = 0
    for f in files:
        f = Path(f)
        h = int(hashlib.md5(piece_of[f.name].encode()).hexdigest(), 16)
        if h % 2 == 0:
            _link_or_copy(f, out_a / f.name); na += 1
        else:
            _link_or_copy(f, out_b / f.name); nb += 1
    return na, nb


def roundtrip_encodec(files: Iterable[Path], out_dir: Path, codec_id: str = "facebook/encodec_32khz") -> int:
    import torch, torchaudio
    from transformers import EncodecModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.from_pretrained(codec_id).to(device).eval()
    proc = AutoProcessor.from_pretrained(codec_id)
    sr_m = proc.sampling_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        wav, sr = torchaudio.load(str(f))
        wav = wav.mean(0, keepdim=True)
        if sr != sr_m:
            wav = torchaudio.functional.resample(wav, sr, sr_m)
        inp = proc(raw_audio=wav.squeeze(0).numpy(), sampling_rate=sr_m, return_tensors="pt")
        with torch.no_grad():
            enc = model.encode(inp["input_values"].to(device), inp["padding_mask"].to(device))
            dec = model.decode(enc.audio_codes, enc.audio_scales, inp["padding_mask"].to(device))[0]
        torchaudio.save(str(out_dir / Path(f).name), dec.squeeze(0).cpu(), sr_m)
        n += 1
    return n


def roundtrip_stable_audio_vae(files: Iterable[Path], out_dir: Path,
                               repo: str = "stabilityai/stable-audio-open-1.0") -> int:
    """Oobleck VAE (44.1 кГц, стерео). Моно референс дублируется в 2 канала."""
    import torch, torchaudio
    from diffusers import AutoencoderOobleck

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderOobleck.from_pretrained(repo, subfolder="vae").to(device).eval()
    sr_m = int(vae.config.sampling_rate)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        wav, sr = torchaudio.load(str(f))
        if sr != sr_m:
            wav = torchaudio.functional.resample(wav, sr, sr_m)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        with torch.no_grad():
            z = vae.encode(wav[None].to(device)).latent_dist.mode()
            dec = vae.decode(z).sample[0].cpu()
        torchaudio.save(str(out_dir / Path(f).name), dec, sr_m)
        n += 1
    return n


def roundtrip_riffusion_spectrogram(files: Iterable[Path], out_dir: Path) -> int:
    """Мел-спектрограмма Riffusion (512 бинов, 0-10 кГц) -> Griffin-Lim: пол
    представления "спектрограмма как картинка" (потеря фазы и полосы >10 кГц)."""
    import torch
    from pydub import AudioSegment
    from riffusion.spectrogram_image_converter import SpectrogramImageConverter
    from riffusion.spectrogram_params import SpectrogramParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conv = SpectrogramImageConverter(params=SpectrogramParams(), device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        seg = AudioSegment.from_file(str(f)).set_channels(1)
        image = conv.spectrogram_image_from_audio(seg)
        back = conv.audio_from_spectrogram_image(image)
        back.export(str(out_dir / (Path(f).stem + ".wav")), format="wav")
        n += 1
    return n


# TODO (фаза 0): roundtrip_audioldm2_vae(files, out_dir) — mel (AudioLDM2
# feature extractor) -> pipe.vae.encode/decode -> pipe.vocoder; и
# roundtrip_ace_step_dcae(files, out_dir) — через DCAE из acestep.

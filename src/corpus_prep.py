"""
Референсный корпус NES-VMDB (Cardoso, Moraes, Ferreira; arXiv:2404.04420).

ПРОВЕРЕНО по README репозитория (2026-09-04): аудио датасета — это
ПОЛНОРАЗМЕРНЫЕ MP3, синтезированные из MIDI NES-MDB оригинальным NES-синтом
(папка Audio/), а 15-секундные фрагменты — это ВИДЕО геймплея (папка Videos/),
их звуковая дорожка содержит игровые звуковые эффекты и как референс музыки не
годится. Метаданные (Video-Midi CSV): game id, game name, video fragment id,
matched MIDI, два confidence-столбца Dejavu — нужны только для сопоставления
видео и в подготовке референса не используются.

Поэтому референс длительности D строится ОКНАМИ длиной D из полноразмерных
MP3: непересекающиеся окна по каждой партитуре, затем случайная выборка по
всем кандидатам (с ограничением окон на партитуру, чтобы длинные пьесы не
доминировали). Партитуры короче D исключаются из референса этой D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import peak_normalize

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg"}


def list_pieces(audio_root: Path) -> list[Path]:
    return sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def piece_duration_s(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        import librosa
        return float(librosa.get_duration(path=str(path)))


def load_piece(path: Path) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return y.mean(axis=1), sr
    except Exception:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


def plan_windows(pieces: list[Path], target_duration_s: float, n_segments: int,
                 max_per_piece: int = 3, seed: int = 0) -> list[dict]:
    """Кандидаты = непересекающиеся окна каждой партитуры (не более
    max_per_piece на партитуру, выбранных случайно), затем случайная выборка
    n_segments по всем партитурам. Сначала план, потом загрузка аудио —
    чтобы выборка не была смещена в сторону первых по алфавиту игр."""
    rng = np.random.default_rng(seed)
    candidates = []
    for p in pieces:
        dur = piece_duration_s(p)
        n_win = int(dur // target_duration_s)
        if n_win == 0:
            continue
        starts = np.arange(n_win) * target_duration_s
        if n_win > max_per_piece:
            starts = rng.choice(starts, max_per_piece, replace=False)
        for s in starts:
            candidates.append({"piece": str(p), "start_s": float(s), "duration_s": target_duration_s})
    if len(candidates) > n_segments:
        idx = rng.choice(len(candidates), n_segments, replace=False)
        candidates = [candidates[i] for i in sorted(idx)]
    return candidates


def write_reference_set(plan: list[dict], out_dir: Path, peak: float = 0.9) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, cache = [], {}
    for i, item in enumerate(plan):
        p = Path(item["piece"])
        if str(p) not in cache:
            cache = {str(p): load_piece(p)}  # держим в памяти только текущую партитуру
        y, sr = cache[str(p)]
        a, b = int(item["start_s"] * sr), int((item["start_s"] + item["duration_s"]) * sr)
        seg = peak_normalize(y[a:b], peak)
        if len(seg) < b - a:
            continue
        name = f"{p.stem}_{int(item['start_s'])}s_{int(item['duration_s'])}s_{i:05d}.wav"
        sf.write(str(out_dir / name), seg, sr, subtype="PCM_24")
        manifest.append({**item, "file": name, "sr": sr})
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Построение референсных наборов NES-VMDB по длительностям")
    ap.add_argument("--audio-root", type=Path, required=True, help="папка Audio/ NES-VMDB (полноразмерные MP3)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/references"))
    ap.add_argument("--durations", type=float, nargs="+", default=[15, 30, 60, 90, 120])
    ap.add_argument("--n-segments", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pieces = list_pieces(args.audio_root)
    print(f"партитур: {len(pieces)}")
    for D in args.durations:
        plan = plan_windows(pieces, D, args.n_segments, seed=args.seed)
        out = args.out_dir / f"{int(D)}s"
        m = write_reference_set(plan, out)
        n_pieces = len({x["piece"] for x in m})
        print(f"D={D:>5.0f}s: окон {len(m)} из {n_pieces} партитур -> {out}")


if __name__ == "__main__":
    main()

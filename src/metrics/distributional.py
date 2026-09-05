"""
Группа 1: FAD и KAD.

Схема:
  * Эмбеддинги извлекаются ОДИН раз через fadtk (Microsoft) на эмбеддере
    clap-laion-music и кэшируются в .npy рядом с аудио. fadtk режет каждый файл
    на 10-секундные окна с шагом 1 с и возвращает (T, D) на файл; единица
    статистики FAD/KAD в тулкитах — окно, не файл (сильно коррелированные
    окна одного файла). Это документируется в статье; в бутстрапе
    передискретизируются ФАЙЛЫ (все окна файла целиком), что корректно
    учитывает корреляцию окон внутри файла.
  * FAD и KAD считаются здесь на numpy по ОДНИМ И ТЕМ ЖЕ эмбеддингам — иначе
    сравнение "FAD vs KAD" смешивалось бы с различием эмбеддеров.
  * Точечные значения перепроверяются вызовом самих тулкитов (fadtk.score,
    kadtk CLI) на нескольких клетках — консистентность (до масштабного
    множителя у KAD, если kadtk масштабирует MMD²) фиксируется в логе.

Проверено по исходникам (2026-09-04): fadtk.fad.FrechetAudioDistance(ml,
audio_load_worker, load_model); .cache_embedding_file(path); ._load_embeddings
(files, max_count, concat); .score(baseline_dir, eval_dir); .score_inf(baseline,
eval_files, steps=25, min_n=500) -> FADInfResults(score, slope, r2, points);
kadtk CLI: kadtk <model> <ref> <eval> [--fad] [--inf] [--csv file], печатает
"Score: <x>". Точное имя модели в kadtk — сверить `kadtk --help` (в fadtk
имя clap-laion-music подтверждено).
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import linalg

log = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "clap-laion-music"
MAX_REF_CHUNKS = 6000   # подвыборка окон референса для ядра KAD (память O(n·m))


# --------------------------------------------------------------------------
# Извлечение эмбеддингов через fadtk
# --------------------------------------------------------------------------

_FAD_CACHE: dict = {}


def get_fadtk(model_name: str = DEFAULT_EMBEDDER):
    if model_name in _FAD_CACHE:
        return _FAD_CACHE[model_name]
    from fadtk.fad import FrechetAudioDistance
    from fadtk.model_loader import get_all_models

    models = {m.name: m for m in get_all_models()}
    if model_name not in models:
        raise KeyError(f"Эмбеддер {model_name!r} не найден в fadtk; доступны: {sorted(models)}")
    fad = FrechetAudioDistance(models[model_name], audio_load_worker=8, load_model=True)
    _FAD_CACHE[model_name] = fad
    return fad


def embed_files(files: Sequence[Path], model_name: str = DEFAULT_EMBEDDER) -> list[np.ndarray]:
    """Кэширует и возвращает список (T_i, D) — по одному массиву на файл, в
    порядке files."""
    fad = get_fadtk(model_name)
    files = [Path(f) for f in files]
    for f in files:
        fad.cache_embedding_file(f)
    embs = fad._load_embeddings(files, concat=False)
    return [np.asarray(e, dtype=np.float64) for e in embs]


# --------------------------------------------------------------------------
# Переносимый кэш эмбеддингов
# --------------------------------------------------------------------------
# Слитый цикл на чужой площадке (src/remote_worker.py) удаляет аудио сразу
# после метрик: 85 ГБ вывозить неоткуда. Но FAD/KAD и CLAP считаются ПО аудио,
# а перегенерировать его локально нельзя — на другом железе тот же сид даёт
# другой сигнал (review_log.md, F.31). Значит эмбеддинг обязан извлекаться в
# той же сессии, пока файл ещё есть, и уезжать вместо аудио (~1 ГБ на план).
#
# Кэш fadtk для этого не годится: он адресуется путём аудиофайла, а путь на
# площадке (/kaggle/temp/...) не совпадает с локальным (D:/...). Поэтому свой
# кэш — по ОТНОСИТЕЛЬНОМУ пути трека внутри results/audio, он одинаков везде.


def embedding_cache_path(wav_path: Path, out_dir: Path, model_name: str = DEFAULT_EMBEDDER) -> Path:
    """results/audio/<model>/<D>s/<prompt>/set<k>/seed<N>.wav ->
    results/embeddings/<embedder>/<model>/<D>s/<prompt>/set<k>/seed<N>.npy"""
    wav_path, out_dir = Path(wav_path), Path(out_dir)
    parts = wav_path.resolve().parts
    rel = Path(*parts[parts.index("audio") + 1:]) if "audio" in parts else Path(wav_path.name)
    return out_dir / "embeddings" / model_name / rel.with_suffix(".npy")


def cache_embedding(wav_path: Path, out_dir: Path, model_name: str = DEFAULT_EMBEDDER) -> Path:
    """Извлечь эмбеддинг одного трека и положить в переносимый кэш."""
    dst = embedding_cache_path(wav_path, out_dir, model_name)
    if dst.exists():
        return dst
    emb = embed_files([wav_path], model_name)[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, np.asarray(emb, dtype=np.float32))   # float32: вдвое меньше везти
    return dst


def embed_files_cached(files: Sequence[Path], out_dir: Path,
                       model_name: str = DEFAULT_EMBEDDER) -> list[np.ndarray]:
    """Как embed_files, но сперва берёт готовое из переносимого кэша. Треки,
    сгенерированные на площадке без вывода аудио, считаются только так."""
    files = [Path(f) for f in files]
    out, missing = {}, []
    for f in files:
        c = embedding_cache_path(f, out_dir, model_name)
        if c.exists():
            out[f] = np.asarray(np.load(c), dtype=np.float64)
        elif f.exists():
            missing.append(f)
        else:
            raise FileNotFoundError(
                f"нет ни аудио, ни эмбеддинга для {f}: трек сгенерирован на площадке "
                f"без --with-embeddings, а перегенерация на другом железе не "
                f"воспроизводит сигнал (review_log.md F.31)")
    if missing:
        for f, e in zip(missing, embed_files(missing, model_name)):
            out[f] = e
    return [out[f] for f in files]


# --------------------------------------------------------------------------
# FAD
# --------------------------------------------------------------------------

@dataclass
class GaussStats:
    mu: np.ndarray
    sigma: np.ndarray
    n: int


def gauss_stats(chunks: np.ndarray) -> GaussStats:
    return GaussStats(mu=chunks.mean(axis=0), sigma=np.cov(chunks, rowvar=False), n=chunks.shape[0])


def frechet_distance(a: GaussStats, b: GaussStats) -> float:
    diff = a.mu - b.mu
    covmean, _ = linalg.sqrtm(a.sigma @ b.sigma, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(a.sigma.shape[0]) * 1e-6
        covmean, _ = linalg.sqrtm((a.sigma + eps) @ (b.sigma + eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(a.sigma) + np.trace(b.sigma) - 2 * np.trace(covmean))


def fad_from_embeddings(ref: np.ndarray, ev: np.ndarray) -> float:
    return frechet_distance(gauss_stats(ref), gauss_stats(ev))


# --------------------------------------------------------------------------
# KAD (MMD² с гауссовым ядром, ширина = медиана попарных расстояний референса)
# --------------------------------------------------------------------------

def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = (x * x).sum(1)[:, None]
    yy = (y * y).sum(1)[None, :]
    return np.maximum(xx + yy - 2 * x @ y.T, 0.0)


def median_bandwidth(ref: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = ref if len(ref) <= max_n else ref[rng.choice(len(ref), max_n, replace=False)]
    d2 = _sq_dists(x, x)
    iu = np.triu_indices(len(x), k=1)
    return float(np.sqrt(np.median(d2[iu])))


def _kernel_mean(d2: np.ndarray, bw: float, unbiased_diag: bool) -> float:
    k = np.exp(-d2 / (2 * bw * bw))
    if unbiased_diag:
        n = k.shape[0]
        return float((k.sum() - np.trace(k)) / (n * (n - 1)))
    return float(k.mean())


def kad_from_embeddings(ref: np.ndarray, ev: np.ndarray, bandwidth: float | None = None,
                        k_rr: float | None = None) -> float:
    """Несмещённая оценка MMD². k_rr (член референс-референс) можно передать
    заранее — он не меняется при бутстрапе по оцениваемому набору."""
    bw = bandwidth or median_bandwidth(ref)
    if k_rr is None:
        k_rr = _kernel_mean(_sq_dists(ref, ref), bw, True)
    k_ee = _kernel_mean(_sq_dists(ev, ev), bw, True)
    k_re = _kernel_mean(_sq_dists(ref, ev), bw, False)
    return float(k_rr + k_ee - 2 * k_re)


# --------------------------------------------------------------------------
# Набор + бутстрап по файлам
# --------------------------------------------------------------------------

@dataclass
class DistributionalSetResult:
    fad: float
    kad: float
    fad_boot: dict
    kad_boot: dict
    n_files: int
    n_chunks: int
    bandwidth: float


class ReferenceContext:
    """Предвычисленные статистики референса (подвыборка окон для ядра KAD)."""

    def __init__(self, ref_embs: list[np.ndarray], seed: int = 0):
        rng = np.random.default_rng(seed)
        all_chunks = np.concatenate(ref_embs)
        self.gauss = gauss_stats(all_chunks)
        self.ref_chunks = all_chunks if len(all_chunks) <= MAX_REF_CHUNKS else \
            all_chunks[rng.choice(len(all_chunks), MAX_REF_CHUNKS, replace=False)]
        self.bandwidth = median_bandwidth(self.ref_chunks)
        self.k_rr = _kernel_mean(_sq_dists(self.ref_chunks, self.ref_chunks), self.bandwidth, True)
        self.n_files = len(ref_embs)
        self.n_chunks = int(len(all_chunks))


def evaluate_set(ref: ReferenceContext, eval_embs: list[np.ndarray], n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> DistributionalSetResult:
    ev = np.concatenate(eval_embs)
    fad = frechet_distance(ref.gauss, gauss_stats(ev))
    kad = kad_from_embeddings(ref.ref_chunks, ev, ref.bandwidth, ref.k_rr)

    rng = np.random.default_rng(seed)
    n = len(eval_embs)
    fads, kads = np.empty(n_resamples), np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        e = np.concatenate([eval_embs[j] for j in idx])
        fads[i] = frechet_distance(ref.gauss, gauss_stats(e))
        kads[i] = kad_from_embeddings(ref.ref_chunks, e, ref.bandwidth, ref.k_rr)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]

    def boot(arr, point):
        lo, hi = np.percentile(arr, q)
        return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "boot_std": float(arr.std()),
                "n_observations": n, "n_resamples": n_resamples}

    return DistributionalSetResult(fad=fad, kad=kad, fad_boot=boot(fads, fad), kad_boot=boot(kads, kad),
                                   n_files=n, n_chunks=int(len(ev)), bandwidth=ref.bandwidth)


# --------------------------------------------------------------------------
# Кросс-проверка тулкитами (точечные значения)
# --------------------------------------------------------------------------

def toolkit_fad(ref_dir: Path, eval_dir: Path, model_name: str = DEFAULT_EMBEDDER) -> float:
    return float(get_fadtk(model_name).score(str(ref_dir), str(eval_dir)))


def toolkit_fad_inf(ref_dir: Path, eval_files: Sequence[Path], model_name: str = DEFAULT_EMBEDDER) -> dict:
    r = get_fadtk(model_name).score_inf(str(ref_dir), [Path(f) for f in eval_files])
    return {"score": float(r.score), "slope": float(r.slope), "r2": float(r.r2)}


def toolkit_kad_cli(ref_dir: Path, eval_dir: Path, model_name: str = DEFAULT_EMBEDDER) -> float:
    """kadtk <model> <ref> <eval> — печатает 'Score: <x>'. TODO фаза 0: имя модели."""
    out = subprocess.run(["kadtk", model_name, str(ref_dir), str(eval_dir)],
                         capture_output=True, text=True, check=True).stdout
    for line in reversed(out.strip().splitlines()):
        if "Score" in line:
            return float(line.split(":")[-1].strip())
    raise ValueError(f"kadtk: не найдена строка 'Score:' в выводе:\n{out}")

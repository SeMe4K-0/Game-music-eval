"""
Оценка разброса для сравнений моделей. См. docs/experiment_design.md
"Почему сид — не уровень фактора".

Две независимые оценки разброса метрик группы 1 (FAD/KAD, считаются на
наборах, а не на отдельных треках):
  1. between_set_ci   — разброс между K независимыми наборами.
  2. bootstrap_ci      — бутстрап по трекам внутри одного набора.
  3. combined_ci        — консервативное объединение: берём более широкий
     из двух интервалов, т.к. любой из источников разброса может доминировать
     в зависимости от K и N.

Для метрик, которые считаются per-track (группы 2-4: CLAP score, seam
discontinuity, beat grid alignment, tempo/key drift) — там сид ЯВЛЯЕТСЯ
уровнем фактора (каждый трек даёт одно наблюдение метрики), и для них
используется обычный bootstrap_ci по трекам без разделения на "между/внутри
наборов".
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats as st


def bootstrap_ci(
    values: Sequence,
    statistic_fn: Callable[[Sequence], float] | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Percentile-бутстрап доверительный интервал.

    values: список наблюдений (либо готовые скаляры метрики per-track, либо
        произвольные объекты — если statistic_fn считает метрику по
        произвольному подмножеству, напр. FAD по подмножеству эмбеддингов).
    statistic_fn: функция(subset) -> float. По умолчанию — np.mean (для
        per-track метрик группы 2-4).
    """
    if statistic_fn is None:
        statistic_fn = lambda subset: float(np.mean(subset))

    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap_ci: пустой список наблюдений")

    point = statistic_fn(values)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = [values[j] for j in idx]
        boots[i] = statistic_fn(sample)

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot_std": float(np.std(boots)),
        "n_observations": n,
        "n_resamples": n_resamples,
    }


def between_set_ci(set_values: Sequence[float], alpha: float = 0.05) -> dict:
    """
    CI по K независимым наборам метрики (K обычно 3 или 5, см.
    configs/durations.yaml::factorial_plan). При K=1 CI вырождается в точку —
    это сигнал, что для данной клетки плана нужно достроить хотя бы K=2, иначе
    сравнение с другой моделью в этой клетке не имеет обоснования.
    """
    arr = np.asarray(set_values, dtype=float)
    k = len(arr)
    mean = float(arr.mean())
    if k > 1:
        se = float(arr.std(ddof=1) / np.sqrt(k))
        tval = float(st.t.ppf(1 - alpha / 2, df=k - 1))
        lo, hi = mean - tval * se, mean + tval * se
    else:
        se, lo, hi = 0.0, mean, mean
    return {"mean": mean, "se": se, "ci_lo": lo, "ci_hi": hi, "k": k}


def combined_ci(between: dict, bootstrap: dict) -> dict:
    """Консервативное объединение: шире из двух интервалов, а не их среднее —
    доверительный интервал должен покрывать оба источника неопределённости."""
    width_between = between["ci_hi"] - between["ci_lo"]
    width_boot = bootstrap["ci_hi"] - bootstrap["ci_lo"]
    dominant = "between_set" if width_between >= width_boot else "bootstrap"
    lo = min(between["ci_lo"], bootstrap["ci_lo"])
    hi = max(between["ci_hi"], bootstrap["ci_hi"])
    return {"ci_lo": lo, "ci_hi": hi, "dominant_source": dominant,
            "width_between": width_between, "width_bootstrap": width_boot}


def cis_overlap(ci_a: dict, ci_b: dict) -> bool:
    """Простая проверка непересечения доверительных интервалов двух моделей —
    используется как порог "статистически различимого" сравнения в таблицах
    статьи (не заменяет полноценный тест значимости, но достаточен как явный,
    воспроизводимый критерий для решающего правила веток 1/2)."""
    return not (ci_a["ci_hi"] < ci_b["ci_lo"] or ci_b["ci_hi"] < ci_a["ci_lo"])


def bootstrap_diff_ci(values_a: Sequence[float], values_b: Sequence[float], n_resamples: int = 1000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """Бутстрап-CI на РАЗНОСТЬ средних двух независимых выборок per-track
    метрики (модель A − модель B). Мощнее, чем проверка пересечения двух
    отдельных CI; используется как дополнение к cis_overlap в таблицах."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("bootstrap_diff_ci: пустая выборка")
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        diffs[i] = a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(a.mean() - b.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def proportion_ci(successes: int, n: int, alpha: float = 0.05) -> dict:
    """Wilson-интервал для доли (доля треков, прошедших шовное/тактовое условие)."""
    if n == 0:
        return {"p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    z = float(st.norm.ppf(1 - alpha / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"p": p, "ci_lo": float(centre - half), "ci_hi": float(centre + half), "n": n}

"""
Отпечаток вычислительной среды и разрешение типа данных.

Зачем это нужно, если раньше не было. Как только клетки плана раздаются по
разным площадкам (RTX 3060 / Kaggle T4 / Colab T4 / Lightning L4), возникают
две ловушки, каждая из которых портит сравнение моделей:

1. **Один сид не даёт одинаковое аудио на разном железе.** Разные ядра CUDA,
   разные алгоритмы cuDNN, разный порядок редукций. Схема common random numbers
   (одинаковые сиды между моделями и длительностями) на разном железе не
   работает. Статистика при этом остаётся корректной — наборы независимы, — но
   железо не должно оказаться скрытым фактором, спутанным с моделью. Поэтому
   отпечаток пишется в каждую запись манифеста, и правило простое: одна клетка
   (модель × длительность) генерируется целиком на одной площадке.

2. **bfloat16 есть не везде.** T4 (Turing, sm_75), P100 (Pascal, sm_60) и M4000
   (Maxwell) не имеют аппаратного bf16 — он появился в Ampere (sm_80). ACE-Step
   по умолчанию просит bf16. Молчаливый переход на fp16 на T4 означал бы, что
   модель на разных площадках считается в разной точности — это уже другая
   конфигурация модели, и сравнение с 3060 становится некорректным. Поэтому
   dtype ПРИБИТ в configs/models.yaml, а при несовместимости с железом
   генерация падает с внятной ошибкой вместо тихого понижения точности.
"""
from __future__ import annotations

import platform
import subprocess


class DtypeUnsupported(RuntimeError):
    """Железо не поддерживает тип, прибитый в конфиге модели."""


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda": False, "reason": "torch не установлен"}
    if not torch.cuda.is_available():
        return {"cuda": False, "mps": bool(_try(lambda: torch.backends.mps.is_available(), False))}
    cap = torch.cuda.get_device_capability(0)
    return {
        "cuda": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": f"{cap[0]}.{cap[1]}",
        "capability_major": cap[0],
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        "bf16_native": cap[0] >= 8,          # Ampere и новее
        "n_devices": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "driver": _try(lambda: subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()),
    }


def versions() -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "diffusers", "librosa", "numpy", "soundfile"):
        v[mod] = _try(lambda m=mod: __import__(m).__version__)
    return v


def fingerprint() -> dict:
    """Компактный отпечаток для записи в манифест каждой генерации."""
    g = gpu_info()
    v = versions()
    return {
        "host": platform.node(),
        "gpu": g.get("gpu_name", "cpu"),
        "capability": g.get("capability"),
        "vram_gb": g.get("vram_gb"),
        "bf16_native": g.get("bf16_native", False),
        "torch": v.get("torch"),
        "transformers": v.get("transformers"),
        "diffusers": v.get("diffusers"),
    }


def resolve_dtype(requested: str):
    """
    requested — строка из configs/models.yaml ("bfloat16" | "float16" | "float32").
    Возвращает torch.dtype. Падает, если железо не тянет запрошенный тип:
    молчаливое понижение точности сделало бы модель на разных площадках
    разной конфигурацией.
    """
    import torch

    table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if requested not in table:
        raise ValueError(f"неизвестный dtype {requested!r}; ожидается один из {sorted(table)}")
    g = gpu_info()
    if requested == "bfloat16" and g.get("cuda") and not g.get("bf16_native"):
        raise DtypeUnsupported(
            f"{g.get('gpu_name')} (compute capability {g.get('capability')}) не имеет "
            f"аппаратного bfloat16 — он появился в Ampere (8.0). Модель прибита к "
            f"bfloat16 в configs/models.yaml. Варианты: (а) генерировать эту модель "
            f"только на Ampere и новее (RTX 3060, L4, A100); (б) сменить dtype на "
            f"float16 в конфиге ДЛЯ ВСЕХ площадок сразу и перегенерировать уже "
            f"посчитанные клетки этой модели. Тихо переключиться нельзя: это "
            f"сделало бы точность скрытым фактором, спутанным с моделью."
        )
    if requested == "float16" and not g.get("cuda"):
        return torch.float32  # на CPU/MPS fp16 бессмыслен, это только смоук-прогон
    return table[requested]


def describe() -> str:
    g, v = gpu_info(), versions()
    if not g.get("cuda"):
        return f"CPU/MPS: {v['platform']}, python {v['python']}, torch {v.get('torch')}"
    return (f"{g['gpu_name']} ({g['vram_gb']} ГБ, cc {g['capability']}, "
            f"bf16 {'есть' if g['bf16_native'] else 'НЕТ'}), x{g['n_devices']}, "
            f"CUDA {g['cuda_version']}, torch {v.get('torch')}")


if __name__ == "__main__":
    import json
    print(describe())
    print(json.dumps({"gpu": gpu_info(), "versions": versions()}, ensure_ascii=False, indent=1))

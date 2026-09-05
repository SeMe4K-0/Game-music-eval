#!/usr/bin/env python3
"""
Пробный кернел Kaggle: проверяет площадку до того, как отправлять туда
десятки часов генерации.

Что именно проверяется (каждый пункт — отдельный риск, который дешевле снять
за 15 минут пробы, чем поймать на седьмом часу основного прогона):
  1. какая GPU выдана и есть ли на ней bfloat16 (на T4 его нет — значит
     ace_step там недоступен, см. review_log.md F.30);
  2. работает ли интернет (без него не поставить пакеты и не скачать веса);
  3. ставится ли рабочая связка версий (transformers 4.50 + diffusers 0.39:
     на transformers 5.x пайплайн AudioLDM2 падает на приватном методе GPT2);
  4. проходят ли 26 тестов репозитория на чужой машине;
  5. генерируется ли реальный трек, сколько это стоит по времени и VRAM.

Пятый пункт — главный: числа фазы 0 с RTX 3060 нельзя переносить на T4, а
именно по ним планируется, что уносить на Kaggle, а что считать дома.

Вывод пишется в /kaggle/working/probe_report.json.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = "https://github.com/SeMe4K-0/Game-music-eval.git"
WORK = Path("/kaggle/working")
SRC = Path("/kaggle/temp/Game-music-eval")   # вне /kaggle/working: не попадёт в output
REPORT = WORK / "probe_report.json"

report = {"steps": {}}


def step(name):
    """Каждый шаг изолирован: падение одного не мешает собрать остальные."""
    def deco(fn):
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            out = fn()
            report["steps"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1), **(out or {})}
        except Exception as e:
            report["steps"][name] = {"status": "failed", "seconds": round(time.time() - t0, 1),
                                     "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
            print(f"ОШИБКА на шаге {name}: {e!r}", flush=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return fn
    return deco


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.stdout:
        print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise RuntimeError(f"команда вернула {r.returncode}: {cmd}")
    return r.stdout


@step("gpu")
def _gpu():
    import torch
    ok = torch.cuda.is_available()
    info = {"cuda": ok, "torch": torch.__version__}
    if ok:
        info.update(gpu=torch.cuda.get_device_name(0),
                    capability=".".join(map(str, torch.cuda.get_device_capability(0))),
                    vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
                    bf16_native=torch.cuda.get_device_capability(0)[0] >= 8,   # sm_80+, как в src/env_info.py
                    bf16_torch_claims=torch.cuda.is_bf16_supported(),   # на P100 врёт: True при отсутствии аппаратного bf16
                    n_devices=torch.cuda.device_count())
        print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


@step("internet")
def _internet():
    import urllib.request
    with urllib.request.urlopen("https://huggingface.co/api/models/facebook/musicgen-small", timeout=30) as r:
        return {"huggingface_reachable": r.status == 200}


@step("clone")
def _clone():
    if not SRC.exists():
        sh(f"git clone --depth 1 {REPO} {SRC}")
    sha = sh(f"git -C {SRC} rev-parse --short HEAD").strip()
    return {"commit": sha}


@step("deps")
def _deps():
    # Верхняя граница на transformers обязательна: на 5.x diffusers-пайплайн
    # AudioLDM2 падает с AttributeError на _update_model_kwargs_for_generation.
    sh("pip install -q 'transformers==4.50.0' 'diffusers==0.39.0' 'accelerate' "
       "'librosa' 'soundfile' 'pyyaml' 'pandas' 'pytest' 2>&1 | tail -5")
    import importlib
    vers = {}
    for m in ("transformers", "diffusers", "librosa", "numpy", "soundfile"):
        try:
            vers[m] = importlib.import_module(m).__version__
        except Exception as e:
            vers[m] = f"ERR {e}"
    print(json.dumps(vers, ensure_ascii=False), flush=True)
    return {"versions": vers}


@step("tests")
def _tests():
    out = sh(f"cd {SRC} && python -m pytest tests -q 2>&1 | tail -5")
    return {"tail": out.strip()[-300:]}


@step("generate")
def _generate():
    """Реальная генерация: 15 и 120 с, замер времени и пика VRAM на этой карте."""
    sys.path.insert(0, str(SRC))
    os.chdir(SRC)
    from src.generate import ModelRegistry, generate, load_models_config
    from src import audio_utils
    from src.metrics import degeneracy
    import numpy as np

    cfgs = {m["id"]: m for m in load_models_config(SRC / "configs" / "models.yaml")}
    cfg = cfgs["audioldm2_music"]
    reg = ModelRegistry()
    t0 = time.time()
    reg.get(cfg)
    runs = {"load_time_s": round(time.time() - t0, 1), "durations": []}
    for D in (15.0, 120.0):
        g = generate(reg, cfg, "upbeat 8-bit chiptune battle theme", D, 20260904)
        mono, sr = audio_utils.prepare_for_analysis(g.audio, g.sr, 22050)
        deg = degeneracy.report(mono, sr, orig_peak=float(np.max(np.abs(g.audio))))
        r = {"duration_s": D, "wall_time_s": round(g.wall_time_s, 1),
             "peak_vram_gb": round(g.peak_vram_gb, 2) if g.peak_vram_gb else None,
             "actual_duration_s": round(g.actual_duration_s, 2),
             "degenerate": deg["degenerate"]}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        runs["durations"].append(r)
    reg.release()
    return runs


print("\n" + "=" * 60)
print("ИТОГ ПРОБЫ")
for name, s in report["steps"].items():
    print(f"  {name:<12} {s['status']:>8}  {s['seconds']:>7.1f} c")
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nотчёт: {REPORT}")

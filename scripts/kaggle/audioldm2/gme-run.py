#!/usr/bin/env python3
"""
Боевой кернел Kaggle: генерация одной модели слитым циклом remote_worker.

Площадка: T4 (sm_75, bf16 НЕТ) — годятся только float16-модели. ace_step и
HeartMuLa просят bfloat16 и сюда не поедут (см. review_log.md F.30).

Три ограничения Kaggle, под которые всё и построено:
  1. Сессия обрывается на 12 часах. Поэтому работа идёт под будильник
     (--budget-min): за 25 минут до конца цикл останавливается сам и
     упаковывает результат, а не теряет его вместе с сессией.
  2. /kaggle/working между сессиями НЕ сохраняется. Манифест и метрики
     предыдущей сессии подкладываются входным датасетом (см. PRIOR ниже) и
     копируются в рабочую папку до старта: remote_worker идемпотентен по
     ключу (model, D, prompt, set, track) и пропустит уже сделанное.
  3. Сохраняемый вывод — 20 ГБ. Аудио тут не хранится: --keep-audio sample
     оставляет по паре треков на клетку (архивный сэмпл нужен потому, что
     на другом железе тот же сид НЕ даёт то же аудио, F.31), остальное
     удаляется сразу после подсчёта метрик.

Две карты. T4 выдаётся парой, и обе используются: клетки плана делятся
пополам по --shard, каждый процесс видит свою карту через CUDA_VISIBLE_DEVICES.
Это удваивает выработку за тот же час квоты.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("GME_MODEL", "audioldm2_music")
BUDGET_MIN = int(os.environ.get("GME_BUDGET_MIN", "660"))     # 11 ч из 12, запас на упаковку
# Дымовой прогон: несколько треков на клетку вместо полного плана. Нужен перед
# каждой БОЕВОЙ сессией, где что-то поменялось в конвейере метрик — потерять
# 15 минут дешевле, чем 11 часов, как это уже случилось с группами 1-2.
MAX_TRACKS = os.environ.get("GME_MAX_TRACKS", "").strip()
REPO = "https://github.com/SeMe4K-0/Game-music-eval.git"
SRC = Path("/kaggle/temp/Game-music-eval")                     # вне working: не попадёт в output
WORK = Path("/kaggle/working")
OUT = WORK / "results"
INPUT = Path("/kaggle/input")

t_start = time.time()


def sh(cmd, check=True, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"вернула {r.returncode}: {cmd}")
    return r


# --- 1. Код и зависимости ------------------------------------------------
if not SRC.exists():
    sh(f"git clone --depth 1 {REPO} {SRC}")
sh(f"git -C {SRC} rev-parse --short HEAD")

# Верхняя граница на transformers обязательна: на 5.x diffusers-пайплайн
# AudioLDM2 падает на приватном _update_model_kwargs_for_generation у GPT2.
sh("pip install -q 'transformers==4.50.0' 'diffusers==0.39.0' accelerate "
   "librosa soundfile pyyaml pandas")
# fadtk тянет свой torch (2.14) и librosa 0.10 — на локальной машине он этим
# сломал основное окружение. Ставим без зависимостей: всё нужное уже стоит.
sh("pip install -q --no-deps fadtk laion-clap torchlibrosa msclap nnaudio "
   "braceexpand ftfy webdataset wget h5py progressbar hypy-utils")

# --- 2. Возобновление: подкладываем прошлую сессию -----------------------
OUT.mkdir(parents=True, exist_ok=True)
PRIOR = OUT / "_prior"          # сюда собираем всё, что нашлось от прошлых сессий
PRIOR.mkdir(exist_ok=True)
restored = {}
for d in sorted(INPUT.glob("*")):
    for name in ("generation_manifest.jsonl", "per_track_metrics.jsonl", "degeneracy_thresholds.json"):
        for cand in (d / name, d / "results" / name, *d.glob(f"*/{name}")):
            if not cand.exists():
                continue
            dst = PRIOR / name
            if name.endswith(".jsonl"):
                with open(dst, "a", encoding="utf-8") as f, open(cand, encoding="utf-8") as g:
                    f.write(g.read())
            else:
                shutil.copy2(cand, dst)
            break
for name in ("generation_manifest.jsonl", "per_track_metrics.jsonl", "degeneracy_thresholds.json"):
    f = PRIOR / name
    if f.exists():
        restored[name] = sum(1 for _ in open(f, encoding="utf-8")) if name.endswith(".jsonl") else 1
print("восстановлено из прошлых сессий:", json.dumps(restored, ensure_ascii=False), flush=True)

# --- 3. Сколько карт --------------------------------------------------------
sys.path.insert(0, str(SRC))
import torch                                                    # noqa: E402
n_gpu = torch.cuda.device_count()
print(f"карт: {n_gpu}, {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)
del torch

# --- 4. Запуск: по процессу на карту, клетки делятся шардированием -----------
budget_s = BUDGET_MIN * 60 - (time.time() - t_start)
n_gpu = max(1, n_gpu)

# У каждого воркера СВОЯ папка вывода: два процесса, дописывающих один
# generation_manifest.jsonl, рано или поздно порвут строку друг другу.
# Шарды не пересекаются по клеткам, поэтому лишние строки из чужого шарда
# в подложенном манифесте безвредны — они лишь помечают чужие клетки
# сделанными, а этот воркер к ним и не подойдёт.
procs = []
for i in range(n_gpu):
    wdir = OUT / f"gpu{i}"
    wdir.mkdir(parents=True, exist_ok=True)
    for name in ("generation_manifest.jsonl", "per_track_metrics.jsonl", "degeneracy_thresholds.json"):
        src = PRIOR / name
        if src.exists():
            shutil.copy2(src, wdir / name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1")
    shard = f"--shard {i}/{n_gpu}" if n_gpu > 1 else ""
    log = WORK / f"worker{i}.log"
    # --with-clap и --with-embeddings ОБЯЗАТЕЛЬНЫ: аудио удаляется в этой же
    # сессии, и обе GPU-зависимые группы метрик (CLAP score и эмбеддинги для
    # FAD/KAD) после этого посчитать будет уже не по чему — перегенерация на
    # другом железе не воспроизводит сигнал (review_log.md, F.31).
    limit = f"--max-tracks {MAX_TRACKS}" if MAX_TRACKS else ""
    cmd = (f"cd {SRC} && python -m src.remote_worker --models {MODEL} "
           f"--keep-audio sample --sample-per-cell 2 --with-clap --with-embeddings "
           f"--out-dir {wdir} --config-dir {SRC}/configs {shard} {limit}")
    print(f"[gpu{i}] {cmd}", flush=True)
    procs.append((i, subprocess.Popen(cmd, shell=True, env=env,
                                      stdout=open(log, "w"), stderr=subprocess.STDOUT), log))

# --- 5. Будильник: остановить до обрыва сессии и упаковать -------------------
deadline = t_start + budget_s
while any(p.poll() is None for _, p, _ in procs):
    if time.time() > deadline:
        print("бюджет времени исчерпан — останавливаю воркеры, результат сохраняется", flush=True)
        for _, p, _ in procs:
            p.terminate()
        for _, p, _ in procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
        break
    time.sleep(30)

for i, p, log in procs:
    print(f"\n=== хвост лога gpu{i} (код {p.poll()}) ===", flush=True)
    if log.exists():
        print("".join(open(log, encoding="utf-8", errors="replace").readlines()[-25:]), flush=True)

# --- 6. Итог ----------------------------------------------------------------
# Слияние результатов воркеров в общие файлы для выгрузки
man, met = OUT / "generation_manifest.jsonl", OUT / "per_track_metrics.jsonl"
for wdir in sorted(OUT.glob("gpu*")):          # эмбеддинги: свести в одну папку
    src_emb = wdir / "embeddings"
    if src_emb.exists():
        for f in src_emb.rglob("*.npy"):
            dst_f = OUT / "embeddings" / f.relative_to(src_emb)
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            if not dst_f.exists():
                shutil.copy2(f, dst_f)

for name, dst in (("generation_manifest.jsonl", man), ("per_track_metrics.jsonl", met)):
    seen, lines = set(), []
    for wdir in sorted(OUT.glob("gpu*")):
        f = wdir / name
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip() and line not in seen:      # дедуп: подложенное из прошлых сессий совпадает у воркеров
                seen.add(line); lines.append(line)
    if lines:
        dst.write_text("".join(lines), encoding="utf-8")
        print(f"слито {name}: {len(lines)} строк", flush=True)
# Контроль конвейера метрик: без этих двух чисел сессия бессмысленна —
# аудио удалено, и CLAP с эмбеддингами уже не восстановить (F.31).
n_emb = len(list((OUT / "embeddings").rglob("*.npy"))) if (OUT / "embeddings").exists() else 0
n_clap = 0
if met.exists():
    n_clap = sum(1 for l in open(met, encoding="utf-8") if '"clap_score"' in l)

summary = {
    "model": MODEL,
    "embeddings_saved": n_emb,
    "tracks_with_clap": n_clap,
    "elapsed_min": round((time.time() - t_start) / 60, 1),
    "manifest_rows": sum(1 for _ in open(man, encoding="utf-8")) if man.exists() else 0,
    "metrics_rows": sum(1 for _ in open(met, encoding="utf-8")) if met.exists() else 0,
    "restored": restored,
}
# аудио-сэмпл оставляем, но следим за размером сохраняемого вывода (лимит 20 ГБ)
sz = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1024 ** 3
summary["output_gb"] = round(sz, 2)
(WORK / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n" + json.dumps(summary, ensure_ascii=False, indent=1), flush=True)

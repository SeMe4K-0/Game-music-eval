#!/usr/bin/env python3
"""
Переупаковка вывода прошлого кернела в один архив.

Зачем: Kaggle API качает вывод пофайлово, и на 1291 мелком .npy соединение
рвётся с "SSL: UNEXPECTED_EOF" — за четыре прохода доехало 2 файла из 1291.
Перегенерировать 1292 трека ради этого нельзя, это 11 GPU-часов. Кернел
подключает вывод предыдущего запуска через kernel_sources, собирает его в
один tar.gz и отдаёт одним файлом.

GPU не нужен: только чтение и упаковка.
"""
import json
import shutil
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

src = None
for d in sorted(INPUT.glob("*")):
    cand = d / "results" if (d / "results").exists() else d
    if any(cand.rglob("*.jsonl")):
        src = cand
        break
if src is None:
    raise SystemExit(f"не нашёл вывод прошлого кернела в {list(INPUT.glob('*'))}")

stats = {
    "источник": str(src),
    "эмбеддингов": len(list(src.rglob("*.npy"))),
    "аудио": len(list(src.rglob("*.wav"))),
}
for f in src.rglob("*.jsonl"):
    stats[f.name] = sum(1 for _ in open(f, encoding="utf-8"))
print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)

arc = WORK / "gme_audioldm2_results.tar.gz"
with tarfile.open(arc, "w:gz") as tar:
    tar.add(src, arcname="results")
print(f"архив: {arc} ({arc.stat().st_size / 1024**2:.1f} МБ)", flush=True)
(WORK / "repack_summary.json").write_text(
    json.dumps({**stats, "archive_mb": round(arc.stat().st_size / 1024**2, 1)},
               ensure_ascii=False, indent=1), encoding="utf-8")

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

# Вывод сессии бывает в двух видах: распакованные файлы (ранние прогоны) или
# один tar.gz (после того, как боевой кернел стал паковать сам). Обрабатываем
# оба: точка монтирования у Kaggle тоже плавает — то /kaggle/input/notebooks,
# то /kaggle/input/<имя-кернела>.
src = None
for d in sorted(INPUT.glob("*")):
    cand = d / "results" if (d / "results").exists() else d
    if any(cand.rglob("*.jsonl")):
        src = cand
        break
if src is None:
    arcs = sorted(INPUT.rglob("*.tar.gz"))
    if arcs:
        print(f"вход — архив {arcs[0]} ({arcs[0].stat().st_size / 1024**3:.2f} ГБ), распаковываю", flush=True)
        tmp = Path("/kaggle/temp/unpacked")
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(arcs[0]) as t:
            t.extractall(tmp)
        inner = next((x for x in tmp.rglob("*") if x.is_dir() and any(x.glob("*.jsonl"))), None)
        src = inner or tmp
if src is None:
    raise SystemExit(f"не нашёл ни файлов, ни архива в {list(INPUT.glob('*'))}")

stats = {
    "источник": str(src),
    "эмбеддингов": len(list(src.rglob("*.npy"))),
    "аудио": len(list(src.rglob("*.wav"))),
}
for f in src.rglob("*.jsonl"):
    stats[f.name] = sum(1 for _ in open(f, encoding="utf-8"))
print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)

# Разбор по типам: первый архив вышел 4.9 ГБ, и надо понять, чем именно.
by_ext = {}
for f in src.rglob("*"):
    if f.is_file():
        e = f.suffix or "(без расширения)"
        v = by_ext.setdefault(e, [0, 0])
        v[0] += 1
        v[1] += f.stat().st_size
print("состав вывода:", flush=True)
for e, (n, sz) in sorted(by_ext.items(), key=lambda x: -x[1][1]):
    print(f"  {e:<20} {n:>6} шт  {sz/1024**2:>9.1f} МБ", flush=True)

# Аудио в архив НЕ кладём: качать его по рвущемуся соединению незачем, а
# метрики по нему уже посчитаны в той же сессии. Исключение — архивный
# сэмпл: пары треков на клетку хватает, чтобы позже перепроверить метрики
# (перегенерация на другом железе сигнал не воспроизводит, F.31).
kept_wav, sample_per_cell = {}, 2
def keep(ti: tarfile.TarInfo):
    if not ti.name.endswith(".wav"):
        return ti
    cell = "/".join(ti.name.split("/")[:5])
    kept_wav[cell] = kept_wav.get(cell, 0) + 1
    return ti if kept_wav[cell] <= sample_per_cell else None

arc = WORK / "gme_results.tar.gz"
with tarfile.open(arc, "w:gz") as tar:
    tar.add(src, arcname="results", filter=keep)
print(f"аудио оставлено: {sum(min(v, sample_per_cell) for v in kept_wav.values())} из {sum(kept_wav.values())}", flush=True)
print(f"архив: {arc} ({arc.stat().st_size / 1024**2:.1f} МБ)", flush=True)
(WORK / "repack_summary.json").write_text(
    json.dumps({**stats, "archive_mb": round(arc.stat().st_size / 1024**2, 1)},
               ensure_ascii=False, indent=1), encoding="utf-8")

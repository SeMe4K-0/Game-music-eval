#!/usr/bin/env python3
"""
Загрузка аудио NES-VMDB с Google Drive (docs/corpus.md).

Ссылка на датасет есть только в README репозитория авторов
(github.com/rubensolv/NES-VMDB) и ведёт на папку Drive целиком. Нужна ТОЛЬКО
папка `Audio` — полноразмерные MP3, синтезированные из MIDI NES-MDB
оригинальным NES-синтом. Папка `Videos` (98 940 фрагментов по 15 с) не нужна:
её звуковая дорожка содержит игровые звуковые эффекты и как референс музыки не
годится (см. review_log.md, п. A.3).

Почему не `gdown --folder`: в папке ~5 300 мелких файлов, и обход целиком
падает по таймауту Google на середине, теряя прогресс перечисления (~4 минуты).
Здесь перечисление делается один раз и кэшируется в JSON, а докачка идёт
пофайлово с ретраями и пропуском уже скачанного — прогон можно прерывать и
возобновлять.

Запуск:
    python -m scripts.fetch_nesvmdb --out D:/NES-VMDB/Audio
    python -m scripts.fetch_nesvmdb --out D:/NES-VMDB/Audio --retries 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

AUDIO_FOLDER_ID = "1UMrehR6_JNEHfNmTKI1HCPZxLJzuLKhh"   # NES-VMDB/Audio


def enumerate_files(folder_id: str, cache: Path) -> list[dict]:
    """Перечисление файлов папки Drive с кэшем: обход стоит несколько минут."""
    if cache.exists():
        items = json.loads(cache.read_text(encoding="utf-8"))
        print(f"перечисление из кэша {cache}: {len(items)} файлов", flush=True)
        return items

    import gdown
    print("обход папки Drive (несколько минут, результат кэшируется)...", flush=True)
    entries = gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        skip_download=True, quiet=True, use_cookies=False)
    if not entries:
        raise SystemExit("не удалось перечислить папку: пустой ответ Drive")

    items = [{"id": e.id, "path": e.local_path} for e in entries]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"перечислено {len(items)} файлов -> {cache}", flush=True)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("D:/NES-VMDB/Audio"))
    ap.add_argument("--cache", type=Path, default=Path("results/nesvmdb_file_list.json"))
    ap.add_argument("--retries", type=int, default=3, help="проходов докачки по списку")
    ap.add_argument("--delay", type=float, default=0.3, help="пауза между файлами, с")
    ap.add_argument("--folder-id", default=AUDIO_FOLDER_ID)
    a = ap.parse_args()

    import gdown

    items = enumerate_files(a.folder_id, a.cache)
    a.out.mkdir(parents=True, exist_ok=True)

    # Drive отдаёт пути вида "Audio/001/001_1942_01_02MainBGM.mp3" — первый
    # сегмент отбрасываем, чтобы --out сам был корнем Audio.
    for it in items:
        rel = Path(it["path"])
        it["dest"] = a.out / Path(*rel.parts[1:]) if len(rel.parts) > 1 else a.out / rel.name

    # Порядок закачки перемешивается с фиксированным сидом. Drive отдаёт файлы
    # по возрастанию id игры, и если загрузка оборвётся на середине (а Drive
    # душит запросы), скачанным окажется алфавитный префикс: референс сместится
    # к первым по алфавиту играм. Это ровно то смещение, от которого
    # corpus_prep.py защищается порядком «сначала план, потом чтение аудио».
    # После перемешивания любой префикс загрузки — несмещённая выборка по играм.
    random.Random(20260904).shuffle(items)

    for attempt in range(1, a.retries + 1):
        todo = [it for it in items if not it["dest"].exists() or it["dest"].stat().st_size == 0]
        if not todo:
            break
        print(f"\n=== проход {attempt}/{a.retries}: осталось {len(todo)} из {len(items)} ===", flush=True)
        failed = ok = 0
        for i, it in enumerate(todo, 1):
            it["dest"].parent.mkdir(parents=True, exist_ok=True)
            try:
                gdown.download(id=it["id"], output=str(it["dest"]), quiet=True)
                ok += 1
                time.sleep(a.delay)                     # Drive душит частые запросы
            except Exception as e:                      # сеть, таймаут, квота
                failed += 1
                if failed <= 5:
                    print(f"  ошибка на {it['dest'].name}: {type(e).__name__}", flush=True)
                time.sleep(a.delay * 4)                 # после отказа — пауза длиннее
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}: скачано {ok}, ошибок {failed}", flush=True)

    have = sum(1 for it in items if it["dest"].exists() and it["dest"].stat().st_size > 0)
    size_gb = sum(it["dest"].stat().st_size for it in items
                  if it["dest"].exists()) / 1024 ** 3
    print(f"\nскачано {have} из {len(items)} файлов ({size_gb:.2f} ГБ) в {a.out}")
    if have < len(items):
        print("не всё: запустить ещё раз, уже скачанное пропустится")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

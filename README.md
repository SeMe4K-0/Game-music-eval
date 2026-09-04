# game-music-eval

Код и методика к статье «Оценка пригодности открытых моделей генерации музыки
по тексту для игровых систем» (ИИАСУ 2026, МГТУ им. Баумана, секция «Игровые,
обучающие и тестирующие системы»).

**Статус (2026-09-04):** методика, код и черновик статьи прошли две ревизии —
см. `docs/review_log.md` (32 исправления, 4 решения за автором). Реальных
прогонов на GPU ещё не было; пайплайн целиком проверен сквозными тестами на
подставных бэкендах. Раздача работы по бесплатным GPU — `docs/free_gpu_plan.md`.

## Структура
- `docs/review_log.md` — что было не так и что исправлено; открытые решения.
- `docs/free_gpu_plan.md` — какие бесплатные GPU для чего годятся и что куда унести.
- `docs/model_specs.md` — пять моделей, проверенные характеристики, сверенные API.
- `docs/corpus.md` — NES-VMDB (полноразмерные MP3, окна по D), полы метрик.
- `docs/metrics.md` — определения всех метрик и решающее правило.
- `docs/experiment_design.md` — единицы наблюдения, сиды, дробный план, стадии, фаза 0.
- `docs/related_work.md` — позиционирование, что не заявляется как новизна.
- `paper/paper_draft.md` — черновик статьи (введение, обзор, методы — готовы;
  результаты — шаблон до прогона).
- `configs/` — модели, промпты, длительности, план, сиды, решающее правило.
- `src/` — пайплайн по стадиям; `tests/` — 21 смоук-тест без GPU.

## Стадии
```
python3 -m pytest tests -v                                   # 0. локальная проверка (26 тестов, без GPU)
python -m scripts.phase0_check --isolate                     # 0б. проверка API/VRAM на любой GPU
python -m src.corpus_prep --audio-root <NES-VMDB>/Audio     # 1. референсы по D
python -m src.calibrate_degeneracy                           # 1б. пороги вырожденности по референсу
python -m src.run_experiment [--models ...] [--shard i/n]    # 2. генерация
python -m src.compute_track_metrics [--with-clap]            # 3. per-track метрики (CPU/MacBook)
python -m src.compute_distributional --crosscheck            # 4. FAD/KAD + полы (GPU для эмбеддингов)
python -m src.summarize                                      # 5. таблицы с CI, decision.json
```
На чужой/бесплатной площадке вместо шагов 2–3 — слитый цикл, который удаляет
аудио сразу после метрик (85 ГБ аудио возить между площадками нельзя):
```
python -m src.remote_worker --models riffusion_v1 --keep-audio sample --pack out.tar.gz
```
Все стадии идемпотентны: прерванный прогон возобновляется без пересчёта.

## Что требует GPU (RTX 3060, 12 ГБ, fp16)
`src/generate.py` (5 бэкендов), `src/metrics/distributional.py` (эмбеддинги
fadtk), `src/metrics/clap_score.py`, `src/metrics/codec_floor.py`.

## Фаза 0 — до основного прогона
1. `python -m src.run_experiment --max-tracks 1 --durations 120` — пик VRAM всех
   моделей на самой длинной D; AudioLDM2 при 120 с — первый кандидат на OOM.
2. Сверить API с установленными версиями (список — `docs/experiment_design.md`,
   раздел «Фаза 0»); `riffusion_v1.seed_image_path` в `configs/models.yaml`
   указать на `seed_images/og_beat.png` из репозитория riffusion-hobby;
   `ace_step_v1_3_5b.checkpoint_dir` — на локальную папку весов.
3. `python -m src.compute_distributional --crosscheck` на одной клетке: FAD
   numpy vs fadtk должны совпасть; KAD vs kadtk — до масштабного множителя.
4. `python -m src.run_experiment --dry-run` — обход плана без генерации.

## Железо
Генерация и эмбеддинги — RTX 3060 (CUDA); per-track метрики, статистика,
таблицы и текст — MacBook M4 Pro. Оценка бюджета генерации — около 5 суток
непрерывной работы (`docs/experiment_design.md`).

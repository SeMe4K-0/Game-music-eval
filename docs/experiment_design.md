# План эксперимента (после ревизии 2026-09-04)

## Факторы
- Модель: 5 (docs/model_specs.md).
- Длительность D ∈ {15, 30, 60, 90, 120} с.
- Промпт: 8 фиксированных игровых формулировок (configs/prompts.yaml);
  `victory_fanfare` — стингер, `loop_eval: false`.

## Единицы наблюдения и разброс
- **Per-track метрики (группы 2–4)**: наблюдение = трек. Бутстрап-CI по
  трекам, Wilson-CI для долей прошедших треков, бутстрап-CI на разность
  средних двух моделей (`src/stats.py`).
- **FAD/KAD (группа 1)**: наблюдение = набор (модель × D × set_idx), в
  наборе все 8 промптов × N=12 треков = 96 файлов. Ковариация 512-мерных
  эмбеддингов по 12 файлам одного промпта вырождена — поэтому промпт не
  является уровнем фактора для FAD/KAD. K независимых наборов: разброс
  между наборами (t, K−1) + бутстрап по файлам внутри набора → консервативное
  объединение (`combined_ci`).
- **Сиды**: `seed = 20260904 + 100000·set + 1000·prompt + track` — уникальны
  между наборами (иначе для детерминированных пайплайнов наборы совпадают
  побайтно), одинаковы между D и моделями (common random numbers).

## Дробный факторный план
K=5 везде, кроме (AudioLDM2, Riffusion, Stable Audio Open) × (15, 120 с): K=3.
Сетка D и промпты не режутся ни для одной модели.
Stable Audio Open при D>47 с — NOT_APPLICABLE (нативный максимум; результат,
не пропуск).

Число генераций: MusicGen 5D·8·12·5 = 2400; ACE-Step 2400; AudioLDM2 и
Riffusion по (3·5 + 2·3)·96 = 2016; SAO (15 с: K=3; 30 с: K=5) = (3+5)·96 =
768. Итого **9 600** (было 24 000 в полном плане).

Оценка времени на RTX 3060 (грубо, по скоростям из карточек моделей,
уточнить в фазе 0): MusicGen ~35 с/трек → 23 ч; ACE-Step ~8 с → 5 ч; AudioLDM2
(200 шагов, до 120 с) ~90 с → 50 ч; Riffusion (до 24 клипов × 50 шагов) ~60 с →
34 ч; SAO ~40 с → 9 ч. **≈120 ч ≈ 5 суток** непрерывной генерации + метрики.
Если бюджет не позволяет — снижать N (12 → 8), не K и не сетку D.

## Стадии (все идемпотентны, возобновляемы)
1. `python -m src.corpus_prep --audio-root <NES-VMDB/Audio>` → референсы.
2. `python -m src.run_experiment` (RTX 3060) → wav + generation_manifest.jsonl
   (фактическая длительность, время, пик VRAM).
3. `python -m src.compute_track_metrics [--with-clap]` (CPU/MacBook параллельно).
4. `python -m src.compute_distributional --crosscheck` (GPU для эмбеддингов).
5. `python -m src.summarize` → таблицы с CI, decision.json (ветка 1/2).

## Решающее правило — см. metrics.md и configs/durations.yaml. Фиксируется до
прогона; применяется механически (`src/summarize.py::decide`).

## Фаза 0 (обязательна)
1. Профилировать пик VRAM всех пяти моделей на самой длинной D (`--max-tracks 1`);
   AudioLDM2 при 120 с — первый кандидат на OOM.
2. Сверить API с установленными версиями: `fadtk` (`cache_embedding_file`,
   `_load_embeddings`, `score_inf`), `kadtk --help` (имя модели), riffusion-hobby
   (`load_checkpoint`, `riffuse`, `SpectrogramImageConverter`), ACE-Step
   (`ACEStepPipeline.__call__`, тег `[inst]`), transformers MusicGen
   (включает ли выход аудио-промпт — код обрабатывает оба случая).
3. Кросс-проверка FAD/KAD numpy vs тулкиты на одной клетке (ожидается
   совпадение FAD до 1e-3; KAD — с точностью до масштабного множителя).
4. Dry-run плана: `python -m src.run_experiment --dry-run`.

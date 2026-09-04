# Реестр моделей: проверенные характеристики

Дата проверки: 2026-09-04. Источник — карточки моделей на HuggingFace, официальные
репозитории и статьи. Числа, помеченные «(уточнить на месте)», нужно перепроверить
эмпирически на RTX 3060 в фазе 0 (профилирование VRAM через
`torch.cuda.max_memory_allocated()`), т.к. вторичные источники расходятся.

## Итоговый состав — 5 моделей максимального различия

| # | Модель (HF id) | Год | Архитектура | Лицензия | Частота, кГц | Каналы | Заявл. макс. длительность | VRAM fp16 |
|---|---|---|---|---|---|---|---|---|
| 1 | `facebook/musicgen-small` | 2023 | Авторегрессионный трансформер (300M) над EnCodec (4 кодбука, 50 Гц) | CC-BY-NC-4.0 | 32 | моно | жёсткий лимит 30 с (1503 токена) на окно **включая аудио-промпт**; длиннее — цепочка continuation (промпт 10 с + 20 с) | ~2–4 ГБ |
| 2 | `cvssp/audioldm2-music` | 2023 | Латентная диффузия, UNet 350M (всего 1.1B), LOA-conditioning | CC-BY-NC-SA-4.0 | 16 | моно | по умолчанию 10 с, конфигурируемо; за пределами обучающей длины качество деградирует | ~8 ГБ (уточнить на месте) |
| 3 | `riffusion/riffusion-model-v1` | 2022, фактически "волна 2023" по использованию | Диффузия по спектрограмме (fine-tune Stable Diffusion 1.5, фаза — Griffin-Lim) | CreativeML OpenRAIL-M | 44.1 (SpectrogramParams; **эффективная полоса ≤10 кГц**, 512 мел-бинов) | моно | нативный клип 5.12 с (512 px × 10 мс); длиннее — цепочка alpha-интерполяции при общем seed-image | ~4–6 ГБ (класс SD1.5 UNet) |
| 4 | `stabilityai/stable-audio-open-1.0` | 2024 | DiT-латентная диффузия: автоэнкодер 156M + T5-conditioning 109M + DiT 1057M | Stability AI Community License | 44.1 | стерео | до 47 с; **D>47 с — не применимо** (нет канонического продолжения в diffusers) | ~8–10 ГБ (уточнить на месте) |
| 5 | `ACE-Step/ACE-Step-v1-3.5B` | 2025 (открыт 7 мая 2025, StepFun + ACE Studio) | Диффузия на латенте DCAE (Deep Compression AutoEncoder, заимствован из Sana) + лёгкий линейный трансформер; REPA-выравнивание с MERT/mHuBERT | Apache 2.0 | 48 (по конфигу VAE релизных чекпойнтов) | стерео | практически до ~4 минут при сохранении когерентности | 3.5B в bf16 ≈ 7 ГБ весов; заявлена работа при 8 ГБ с `cpu_offload`; в 12 ГБ — проверить без offload, при OOM включить |

Все пять укладываются в жёсткое ограничение 12 ГБ VRAM (fp16), с запасом у моделей 1, 3, 5
и вероятным запасом у 2 и 4 (требует подтверждения профилированием).

## Фактические числа фазы 0 (RTX 3060, замерено 2026-09-04)

Замерено `scripts/phase0_check.py` на реальном железе: RTX 3060 12 ГБ, sm_86,
bf16 native, driver 581.80, CUDA 12.1, torch 2.5.1+cu121, transformers 4.50.0,
diffusers 0.39.0. Пик VRAM — `torch.cuda.max_memory_allocated()`, один трек,
batch=1. Промпт «upbeat 8-bit chiptune battle theme», seed 20260904.
**Эти числа заменяют оценки из таблицы выше** — оценки VRAM по вторичным
источникам оказались завышены в 2–3 раза.

| Модель | Загрузка, с | D=15 с | D=120 с | Пик VRAM (D=120) | Выход | Вырожден? |
|---|---|---|---|---|---|---|
| `musicgen_small` | 32.6 | 21.4 с | 122.8 с | **1.69 ГБ** | 32 кГц моно | нет на обеих D |
| `audioldm2_music` | 6.3 | 19.6 с | 163.3 с | **4.05 ГБ** | 16 кГц моно | нет на обеих D |
| `riffusion_v1` | 19.7 | 36.3 с | 152.2 с | **3.10 ГБ** | 44.1 кГц моно | нет на обеих D |
| `ace_step_v1_3_5b` | 3.5 + 143.5 (лениво) | ~14 с | **29.1 с** | **9.52 ГБ** | 48 кГц стерео | нет на обеих D |
| `stable_audio_open_1_0` | — | — | — | — | — | не проверено: гейтед репозиторий |

Фактические длительности: musicgen 14.94 / 119.76 с (округление до кодовых
кадров EnCodec 50 Гц), audioldm2 15.0 / 120.0 с, riffusion 15.0 / 120.0 с,
ace_step 14.95 / 119.91 с. Частоты и каналы совпали с заявленными.

**ACE-Step грузит веса лениво, в первом `__call__`, а не в конструкторе**:
`load_time_s` = 3.5 с (конструктор), а первая генерация на 15 с заняла 157.6 с,
из которых 143.5 с — загрузка модели («Model loaded in 143.52 seconds» в логе
пайплайна). Первый трек каждой сессии дороже примерно на 2.5 минуты.

**Числа фазы 0 для ACE-Step завышены — рабочие взяты из пилота.** Даже за
вычетом загрузки первый вызов несёт прогрев, поэтому фаза 0 дала 14.1 с на
D=15 против фактических 5.4 с в установившемся режиме (расхождение 2.6×; на
D=120 — 29.1 против 27.2 с, всего 1.1×, там прогрев размазан по длинному
треку). Замерено на 80 треках пилота (`--max-tracks 2 --sets 0`, 16 треков на
каждую D):

| D | 15 с | 30 с | 60 с | 90 с | 120 с |
|---|---|---|---|---|---|
| с/трек | **5.4** | **6.4** | **12.6** | **18.5** | **27.2** |
| пик VRAM | 7.70 ГБ | 7.96 ГБ | 8.48 ГБ | 9.00 ГБ | 9.53 ГБ |

Фактические длительности пилота: 14.95 / 29.91 / 59.91 / 89.91 / 119.91 с.
Полный план ACE-Step по этим числам — **9.3 ч**, а не 14.0 ч по фазе 0.

**ACE-Step на D=120 идёт по верхней границе 12 ГБ.** Пик по
`max_memory_allocated()` — 9.52 ГБ, но `nvidia-smi` в момент декодирования DCAE
показывал 11 994 МБ из 12 288 (97.6 %) с учётом фрагментации и резерва
аллокатора. Без `cpu_offload` проходит, но запаса почти нет: на этой карте
нельзя одновременно занимать VRAM ничем другим (в прогоне фон рабочего стола
занимал ~0.7 ГБ и это уже учтено). При OOM в основном прогоне — штатная опция
`cpu_offload: true` в `configs/models.yaml`.

### Закрытые открытые вопросы

1. **AudioLDM2 при D=120 с — OOM НЕТ** (решение D.2 в `review_log.md` закрыто
   по факту): пик 4.05 ГБ из 12 ГБ при `audio_length_in_s=120`,
   `num_inference_steps=200`. Chunk-стратегия (latent_chunking) не нужна;
   модель считается прямым вызовом, как и заложено в `long_duration_strategy`.
2. **MusicGen continuation**: выход `generate()` при аудио-промпте **включает
   промпт** — код отрезает его по длине (проверено: 10 с промпта + 2 с нового
   дают 382 080 отсчётов при 32 кГц). Цепочка на 120 с отработала целиком.
3. **Riffusion**: `load_checkpoint` принимает HF-id репозитория; фактический
   выход — 44 100 Гц моно, как в `SpectrogramParams`.
4. **ACE-Step без `cpu_offload` в 12 ГБ проходит** (см. оговорку выше). API
   сверен с установленной версией `ace_step` 0.2.0: сигнатуры
   `ACEStepPipeline(checkpoint_dir, device_id, dtype, ..., torch_compile,
   cpu_offload, ..., overlapped_decode)` и `__call__(audio_duration, prompt,
   lyrics, infer_step, guidance_scale, scheduler_type, cfg_type, omega_scale,
   manual_seeds, ..., save_path)` совпадают с вызовом в `src/generate.py`;
   `manual_seeds` строкой парсится штатно (`isdigit() → int`, `set_seeds`).
5. **Бюджет генерации пересчитан по факту** (было — оценки по карточкам
   моделей): ACE-Step оказался в 3.5 раза дороже оценки (29.1 с против ~8 с),
   AudioLDM2 вдвое дороже на длинных D, а MusicGen и Riffusion — примерно
   вчетверо дешевле оценки на коротких D. См. пересчёт в
   `docs/experiment_design.md`.

### Версии библиотек, при которых бэкенды работают

Верхняя граница обязательна — она не «пожелание», а условие работоспособности:

- **`transformers` 4.x, НЕ 5.x.** На `transformers==5.16.1` (ставится по
  открытому `>=4.45`) diffusers-пайплайн AudioLDM2 падает с
  `AttributeError: 'GPT2Model' object has no attribute
  '_update_model_kwargs_for_generation'` — приватный метод генерации убран в
  5.x, а `diffusers` вызывает его напрямую. Рабочая связка: `transformers`
  4.50.0 + `diffusers` 0.39.0 (обе версии тянет `ace-step`).
- **Riffusion — отдельный venv с `torch` 2.0.x.** Пакет на PyPI называется
  `riffusion` (не `riffusion-hobby`), его пины (torch<2.0, streamlit<1.18,
  pillow<10) несовместимы с остальным окружением. Рабочая связка:
  `riffusion` 0.0.5 (`--no-deps`) + `torch` 2.0.1+cu118 / `torchaudio` 2.0.2 +
  `diffusers` 0.21.4 + `transformers` 4.35.2 + `huggingface_hub` 0.19.4.
  На `torchaudio>=2.1` падает с `TypeError: InverseMelScale.__init__() got an
  unexpected keyword argument 'max_iter'` (параметр удалён вместе со сменой
  солвера инверсии мел-спектра); на `huggingface_hub>=0.20` — diffusers 0.21.4
  не находит `cached_download`.
- **Кэш HF должен лежать по ASCII-пути.** `torch.jit.load` (traced UNet
  riffusion) на Windows открывает файл через ANSI-API и падает на кириллице в
  пути профиля: `open file failed because of errno 2 on fopen`. Решение —
  `HF_HOME` на ASCII-путь (в прогоне: `D:\hf-cache`).

### Почему это максимальное различие, а не популярность

- **Архитектура**: AR-трансформер / UNet-диффузия / диффузия-по-картинке / DiT-диффузия /
  диффузия+линейный-трансформер+REPA — пять разных семейств, ни одна пара не повторяет
  архитектурный класс другой.
- **Поколение**: 2022–2023 (модели 1–3), 2024 (модель 4), 2025 (модель 5) — оба требуемых
  среза представлены, ось поколений разрешима.
- **Максимальная длительность**: короткие нативные окна (риффузия ~5 с, AudioLDM2-music
  ~10 с, MusicGen ~30 с) против длинных (Stable Audio Open ~47 с, ACE-Step ~4 мин).
- **Формат выхода**: моно 16/32 кГц (модели 1–3) против стерео 44.1/48 кГц (модели 4–5).

### Обязательное условие про модель 2025–2026: выполнено

ACE-Step v1-3.5B — открытая модель 2025 года, архитектурно не похожая ни на одну из
остальных четырёх (латентная диффузия по компрессированному мел-спектрограммному
пространству через DCAE + линейное внимание + многоэнкодерное REPA-выравнивание, а не
классический DiT с кросс-вниманием, как у Stable Audio Open). Лицензия Apache 2.0 —
самая мягкая в наборе. Обоснование выбора именно этой модели вместо
`stable-audio-open-small` (тоже 2025, но та же диффузионная DiT-парадигма и то же
семейство Stability, т.е. не даёт архитектурной оси) и вместо YuE (полноразмерная
lyrics-to-song модель, заведомо не укладывается в 12 ГБ VRAM в fp16).

## Источники (проверено WebSearch/WebFetch 2026-09-04)

- MusicGen: карточка `facebook/musicgen-large` на HuggingFace (архитектура,
  32 кГц, моно, CC-BY-NC-4.0); официальный README audiocraft
  (facebookresearch.github.io/audiocraft/docs/MUSICGEN.html) — VRAM-таблица по
  размерам моделей (small эксплуатируется на малых GPU).
- AudioLDM2-music: карточка `cvssp/audioldm2-music` на HuggingFace (350M UNet,
  1.1B итого, 16 кГц, CC-BY-NC-SA-4.0, обучающие данные 665k часов музыки).
- Riffusion: карточка `riffusion/riffusion-model-v1` на HuggingFace + issue
  riffusion-hobby#121 (природа 5-секундного клипа и интерполяции сидов).
- Stable Audio Open: README `stabilityai/stable-audio-open-1.0` на HuggingFace +
  Evans et al., "Stable Audio Open", arXiv:2407.14358 (архитектура, параметры
  компонентов, 44.1 кГц стерео, до 47 с).
- ACE-Step: GitHub `ace-step/ACE-Step`, карточка `ACE-Step/ACE-Step-v1-3.5B` на
  HuggingFace, arXiv:2506.00045 (архитектура DCAE + линейный трансформер + REPA,
  Apache 2.0, скорость генерации по GPU, поддержка 8 ГБ VRAM).

## API, сверенные с исходниками (2026-09-04)

- ACE-Step: `ACEStepPipeline(checkpoint_dir, dtype, torch_compile, cpu_offload, overlapped_decode)`;
  вызов с `audio_duration, prompt, lyrics, infer_step, guidance_scale, scheduler_type,
  cfg_type, omega_scale, manual_seeds (строка), ..., save_path` — пишет файл, не возвращает массив.
- Riffusion: `RiffusionPipeline.load_checkpoint(checkpoint, dtype, device)`; `riffuse(inputs,
  init_image) -> PIL.Image`; `SpectrogramImageConverter(params, device).audio_from_spectrogram_image(img) -> pydub`.
- MusicGen (transformers): аудио-промпт через `processor(audio=, sampling_rate=, text=)`;
  лимит 30 с включает промпт.
- fadtk: `FrechetAudioDistance(ml, audio_load_worker, load_model)`, `cache_embedding_file`,
  `_load_embeddings(files, concat)`, `score`, `score_inf(..., min_n=500)`; эмбеддер `clap-laion-music`.
- kadtk: CLI `kadtk <model> <ref> <eval> [--fad] [--inf] [--csv]`, печатает `Score: x`.
- CLAP (transformers): экстрактор ждёт 48 000 Гц, не ресемплирует; fusion по 10 с; `is_longer`.

## Открытые вопросы для фазы 0 (профилирование на RTX 3060)

1. Точный пик VRAM (fp16, batch=1) для AudioLDM2-music и Stable Audio Open —
   вторичные источники дают вилку, нужен `torch.cuda.max_memory_allocated()`.
2. ~~Частота Riffusion~~ — подтверждено по `spectrogram_params.py`: 44 100 Гц, моно,
   max_frequency 10 000 Гц, клип 5.12 с. Проверить только, что `load_checkpoint`
   принимает HF-id репозитория.
2а. AudioLDM2 при `audio_length_in_s=120`: риск OOM (внимание UNet по длинному
   латенту) — проверить первым.
3. Версия ACE-Step: зафиксировать `ACE-Step-v1-3.5B` (arXiv:2506.00045, есть
   цитируемая статья) как основную; ACE-Step 1.5 — упомянуть в разделе
   "дальнейшая работа", т.к. на дату проверки не имеет отдельной рецензируемой
   публикации с полными техническими деталями.

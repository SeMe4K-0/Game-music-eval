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

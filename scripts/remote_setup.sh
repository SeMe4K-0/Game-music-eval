#!/usr/bin/env bash
# Установка окружения на бесплатной площадке (Kaggle / Colab / Lightning).
# Запуск из корня проекта:  bash scripts/remote_setup.sh [модель ...]
#
# Ставится только то, что нужно указанным моделям: у riffusion-hobby старые
# пины, и ставить его рядом с новыми diffusers без нужды — напрашиваться на
# конфликт версий.
set -euo pipefail

MODELS="${*:-musicgen_small audioldm2_music stable_audio_open_1_0}"
echo "== окружение для: $MODELS =="

pip install -q --upgrade pip
pip install -q numpy scipy librosa soundfile pandas pyyaml

for m in $MODELS; do
  case "$m" in
    musicgen_small)          pip install -q "transformers>=4.45" ;;
    audioldm2_music|stable_audio_open_1_0)
                             pip install -q "diffusers>=0.30" "transformers>=4.45" ;;
    riffusion_v1)            pip install -q "riffusion-hobby" pydub || {
                               echo "!! riffusion-hobby не встал — ставить в отдельной сессии/venv"; }
                             ;;
    ace_step_v1_3_5b)        pip install -q "git+https://github.com/ace-step/ACE-Step.git" ;;
  esac
done

python -c "import sys; sys.path.insert(0,'.'); from src.env_info import describe; print(describe())"
echo "== готово. Проверка API и VRAM: python -m scripts.phase0_check --isolate =="

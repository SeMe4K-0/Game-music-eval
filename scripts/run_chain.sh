#!/usr/bin/env bash
# Очередь работ после завершения генерации Stable Audio Open.
#
# Каждый шаг ждёт освобождения GPU и логируется отдельно; падение одного шага
# не останавливает остальные — иначе ночной прогон теряется целиком из-за
# одной осечки.
#
# Почему шаги разнесены по venv:
#   .venv       — генерация и метрики (torch cu121, librosa 0.11.0)
#   .venv-sa3   — Stable Audio 3: своя библиотека Stability тянет свой torch
#   .venv-fad   — fadtk: тянет torch 2.14 и librosa 0.10, ломает основной venv
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
LOG="$ROOT/chain.log"
export HF_HOME=/d/hf-cache PYTHONIOENCODING=utf-8 TMPDIR=D:/tmp TEMP=D:/tmp TMP=D:/tmp

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_for_gpu() {
    # Ждём, пока освободится карта: пока идёт SAO, DLL заняты и venv не починить.
    say "жду освобождения GPU..."
    while true; do
        local n
        n=$(powershell -NoProfile -Command \
            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_experiment' } | Measure-Object).Count" 2>/dev/null | tr -d '\r ')
        [ "${n:-0}" = "0" ] && break
        sleep 60
    done
    sleep 20            # дать Windows отпустить файловые дескрипторы
    say "GPU свободна"
}

step() {  # step <имя> <venv> <команда...>
    local name="$1" venv="$2"; shift 2
    say "=== $name ==="
    # shellcheck disable=SC1090
    source "$ROOT/$venv/Scripts/activate" || { say "$name: venv $venv не найден"; return 1; }
    if "$@" >> "$ROOT/chain_${name}.log" 2>&1; then
        say "$name: готово"
    else
        say "$name: ОШИБКА (код $?), см. chain_${name}.log"
    fi
    deactivate 2>/dev/null || true
}

say "########## очередь запущена ##########"
wait_for_gpu

# 1. Починка основного venv: fadtk увёл torch в CPU-сборку и откатил librosa.
#    Версия librosa влияет на onset-детекцию и хромаграммы, то есть стала бы
#    скрытым фактором, спутанным с моделью — метрики ACE-Step считались на 0.11.0.
step repair_venv .venv python -m pip install --cache-dir D:/tmp/pipcache \
    --force-reinstall --no-deps torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
step repair_libs .venv python -m pip install --cache-dir D:/tmp/pipcache \
    "librosa==0.11.0" "numpy>=2.0"
step tests .venv python -m pytest tests -q

# 2. Stable Audio 3 (2026): фаза 0 штатным скриптом, затем полный план.
step phase0_sa3 .venv-sa3 python -m scripts.phase0_check --models stable_audio_3_small_music
step gen_sa3 .venv-sa3 python -m src.run_experiment --models stable_audio_3_small_music

# 3. Метрики: сперва по всем сгенерированным трекам (CPU), затем FAD/KAD (GPU).
step metrics .venv python -m src.compute_track_metrics
step distributional .venv-fad python -m src.compute_distributional --crosscheck
step summarize .venv python -m src.summarize

say "########## очередь завершена ##########"

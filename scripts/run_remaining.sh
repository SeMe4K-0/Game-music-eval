#!/usr/bin/env bash
# Очередь на всю оставшуюся локальную работу.
#
# Порядок продиктован зависимостями, а не удобством:
#   генерация -> метрики -> круговые полы -> FAD/KAD -> сводка.
# Метрики требуют аудио, полы требуют GPU и должны считаться до FAD (иначе
# compute_distributional молча пропустит их, как это уже было), сводка
# требует и того и другого.
#
# Kaggle в очередь не входит: audioldm2 идёт там отдельными сессиями и
# ограничен квотой 30 GPU-часов в неделю.
#
# Каждый шаг логируется отдельно, падение одного не останавливает остальные:
# ночной прогон не должен теряться целиком из-за одной осечки.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
LOG="$ROOT/queue.log"
export HF_HOME=/d/hf-cache PYTHONIOENCODING=utf-8 TMPDIR=D:/tmp TEMP=D:/tmp TMP=D:/tmp

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

busy() {   # заняты ли GPU-этапы (по процессам python, а не по имени скрипта)
    powershell -NoProfile -Command \
        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_experiment|compute_distributional|make_roundtrip' } | Measure-Object).Count" \
        2>/dev/null | tr -cd '0-9'
}

wait_gpu() {
    say "жду освобождения GPU..."
    while true; do
        n=$(busy); [ "${n:-0}" = "0" ] && break
        sleep 60
    done
    sleep 20                      # дать Windows отпустить дескрипторы
}

step() {  # step <имя> <venv> <команда...>
    local name="$1" venv="$2"; shift 2
    say "=== $name ==="
    # shellcheck disable=SC1090
    if ! source "$ROOT/$venv/Scripts/activate" 2>/dev/null; then
        say "$name: ПРОПУЩЕН, нет окружения $venv"; return 1
    fi
    local t0=$SECONDS
    if "$@" >> "$ROOT/queue_${name}.log" 2>&1; then
        say "$name: готово за $(( (SECONDS - t0) / 60 )) мин"
    else
        say "$name: ОШИБКА (код $?), см. queue_${name}.log"
    fi
    deactivate 2>/dev/null || true
}

say "########## очередь оставшегося запущена ##########"

# 1. Генерация. musicgen уже идёт — ждём его, затем riffusion.
#    Riffusion в СВОЁМ окружении: её пины (torch 2.0.1, diffusers 0.21.4,
#    transformers 4.35.2) несовместимы с основным venv.
wait_gpu
step gen_riffusion .venv-riffusion python -m src.run_experiment --models riffusion_v1

# 2. Метрики по всему, что сгенерировано (CPU + CLAP на GPU).
#    Идемпотентно: треки с уже посчитанным clap_score пропускаются.
step metrics .venv python -m src.compute_track_metrics --with-clap

# 3. Круговые полы. ДО FAD: compute_distributional подхватывает только
#    готовые папки references/<D>s_rt_*, иначе тихо считает без них.
step floors_gpu .venv python -m scripts.make_roundtrip_floors --codecs encodec stable_audio
step floors_riff .venv-riffusion python -m scripts.make_roundtrip_floors --codecs riffusion

# 4. FAD/KAD. В своём окружении: fadtk тянет torch 2.14 и librosa 0.10 и
#    ломает основной venv, если ставить его туда.
step distributional .venv-fad python -m src.compute_distributional --crosscheck

# 5. Сводка и решающее правило.
step summarize .venv python -m src.summarize

say "########## очередь завершена ##########"
say "осталось вручную: вторая сессия audioldm2 на Kaggle"

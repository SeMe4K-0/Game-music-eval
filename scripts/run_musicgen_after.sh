#!/usr/bin/env bash
# Запуск musicgen после того, как очередь освободит GPU.
#
# Почему не параллельно с очередью: её шаг FAD/KAD тоже занимает GPU. По VRAM
# обе задачи ужились бы (musicgen 1.69 ГБ), но wall_time_s первых сотен треков
# оказался бы завышен из-за конкуренции, а эти числа идут в бюджет плана и в
# docs/model_specs.md.
#
# Ждём по GPU-этапам, а не по имени скрипта очереди: метрики на CPU не мешают,
# а pgrep в Git Bash не видит процессы, запущенные через powershell.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
LOG="$ROOT/chain.log"
export HF_HOME=/d/hf-cache PYTHONIOENCODING=utf-8

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

gpu_busy() {
    powershell -NoProfile -Command \
        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'compute_distributional|phase0_check|run_experiment' } | Measure-Object).Count" \
        2>/dev/null | tr -cd '0-9'
}

say "musicgen: жду освобождения GPU..."
while true; do
    n=$(gpu_busy)
    [ "${n:-0}" = "0" ] && break
    sleep 60
done
sleep 30
say "musicgen: GPU свободна, старт"

say "=== gen_musicgen ==="
# shellcheck disable=SC1091
source "$ROOT/.venv/Scripts/activate"
if python -m src.run_experiment --models musicgen_small >> "$ROOT/chain_gen_musicgen.log" 2>&1; then
    say "gen_musicgen: готово"
else
    say "gen_musicgen: ОШИБКА (код $?), см. chain_gen_musicgen.log"
fi

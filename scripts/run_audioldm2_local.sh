#!/usr/bin/env bash
# Досчёт audioldm2 локально, после riffusion.
#
# Kaggle отпал: недельная квота 30 GPU-часов исчерпана, и обе сессии ушли
# впустую (первая — без CLAP и эмбеддингов, вторая — не подхватила манифест и
# сгенерировала план заново). Осталось 724 трека: D=60 — 23, D=90 — вся клетка
# 480, D=120 — 221. Это самые дорогие длительности, около 27 часов.
#
# Скрипт отдельный, а не строка в run_remaining.sh, потому что тот уже
# выполняется: bash дочитывает файл по мере работы, и правка в середине
# сломала бы текущий прогон.
#
# После генерации переигрываются зависящие шаги — метрики, FAD/KAD и сводка.
# Все они идемпотентны (пропускают посчитанное по ключу), поэтому повтор
# стоит дёшево, а без него данные audioldm2 не попали бы в итог: основная
# очередь дойдёт до них раньше, чем эта генерация закончится.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
LOG="$ROOT/queue.log"
export HF_HOME=/d/hf-cache PYTHONIOENCODING=utf-8 TMPDIR=D:/tmp TEMP=D:/tmp TMP=D:/tmp

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Ждём именно генерацию: метрики на CPU не мешают, а вот две модели на одной
# карте замедлят друг друга и исказят wall_time_s, который идёт в бюджет.
gen_running() {
    powershell -NoProfile -Command \
        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_experiment' } | Measure-Object).Count" \
        2>/dev/null | tr -cd '0-9'
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

say "audioldm2: жду окончания генерации riffusion..."
while true; do
    n=$(gen_running); [ "${n:-0}" = "0" ] && break
    sleep 120
done

# Пауза на остывание перед следующими сутками нагрузки. Ждём не по таймеру, а
# по факту: карта считается остывшей, когда температура опустилась ниже порога.
# Нижняя граница по времени всё равно выдерживается — датчик показывает ядро,
# а остыть должны ещё и цепи питания с памятью, у них инерция больше.
COOL_C=${GME_COOL_C:-50}
COOL_MIN=${GME_COOL_MIN:-30}
say "перерыв: минимум ${COOL_MIN} мин и до ${COOL_C} °C"
t_start=$SECONDS
while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -cd '0-9')
    mins=$(( (SECONDS - t_start) / 60 ))
    if [ "$mins" -ge "$COOL_MIN" ] && [ "${t:-99}" -le "$COOL_C" ]; then
        say "перерыв окончен: ${mins} мин, ${t} °C"
        break
    fi
    if [ "$mins" -ge 120 ]; then          # страховка: не ждать вечно
        say "перерыв окончен по лимиту 2 ч: ${t} °C"
        break
    fi
    [ $(( mins % 10 )) -eq 0 ] && say "  остывает: ${mins} мин, ${t} °C"
    sleep 60
done
say "audioldm2: старт"

# Генерация идёт обычным run_experiment: аудио остаётся на диске, метрики
# считаются отдельным шагом. Это не Kaggle — места хватает, а сохранённое
# аудио позволяет пересчитать метрики без перегенерации.
step gen_audioldm2 .venv python -m src.run_experiment --models audioldm2_music

# Переигрываем то, что зависит от полноты данных.
step metrics_after_audioldm2 .venv python -m src.compute_track_metrics --with-clap
step distributional_after .venv-fad python -m src.compute_distributional --crosscheck
step summarize_after .venv python -m src.summarize

say "########## audioldm2 и пересчёт завершены ##########"

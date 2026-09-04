#!/usr/bin/env python3
"""
Фаза 0: проверка каждой модели по отдельности на арендованной/бесплатной GPU.

Самое ценное, что можно сделать на чужой карте: за минуты GPU-времени выяснить,
какие API отвалились с момента написания кода, сколько на самом деле нужно VRAM
и сколько секунд стоит трек. Без этих чисел планировать основной прогон нельзя,
а получить их можно на любой бесплатной T4.

Для каждой модели: загрузка -> одна короткая генерация -> одна на максимальной
длительности -> замер пика VRAM и времени -> проверка, что выход не вырожден.
Каждая модель проверяется в отдельном процессе (--isolate), иначе первая же
ошибка загрузки весов уронит проверку остальных.

Запуск:
    python -m scripts.phase0_check                      # все модели
    python -m scripts.phase0_check --models riffusion_v1
    python -m scripts.phase0_check --isolate            # каждая в своём процессе
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audio_utils                      # noqa: E402
from src.env_info import describe, fingerprint   # noqa: E402
from src.generate import ModelRegistry, NotApplicable, generate, load_models_config  # noqa: E402
from src.metrics import degeneracy               # noqa: E402


def check_model(cfg: dict, durations: list[float], out_dir: Path) -> dict:
    res = {"model_id": cfg["id"], "hf_repo": cfg["hf_repo"], "dtype": cfg.get("dtype"),
           "env": fingerprint(), "runs": [], "status": "ok"}
    registry = ModelRegistry()
    t_load = time.time()
    try:
        registry.get(cfg)
        res["load_time_s"] = round(time.time() - t_load, 1)
    except Exception as e:
        res.update(status="load_failed", error=repr(e), traceback=traceback.format_exc()[-2000:])
        return res

    for D in durations:
        run = {"duration_s": D}
        try:
            g = generate(registry, cfg, "upbeat 8-bit chiptune battle theme", D, 20260904)
            mono, sr = audio_utils.prepare_for_analysis(g.audio, g.sr, 22050)
            deg = degeneracy.report(mono, sr, orig_peak=float(np.max(np.abs(g.audio))))
            run.update(status="ok", wall_time_s=round(g.wall_time_s, 1),
                       peak_vram_gb=round(g.peak_vram_gb, 2) if g.peak_vram_gb else None,
                       actual_duration_s=round(g.actual_duration_s, 2),
                       sr=g.sr, channels=1 if g.audio.ndim == 1 else int(g.audio.shape[0]),
                       degenerate=deg["degenerate"], degeneracy_reasons=deg["reasons"],
                       spectral_flux=round(deg["spectral_flux"], 3),
                       spectral_flatness=round(deg["spectral_flatness"], 4))
            if out_dir:
                p = out_dir / f"{cfg['id']}_{int(D)}s.wav"
                audio_utils.write_wav(p, g.audio, g.sr)
                run["sample"] = str(p)
        except NotApplicable as e:
            run.update(status="not_applicable", reason=str(e))
        except Exception as e:
            run.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-2000:])
            res["status"] = "partial"
        res["runs"].append(run)
        print(json.dumps(run, ensure_ascii=False), flush=True)

    registry.release()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--out", type=Path, default=Path("results/phase0_report.json"))
    ap.add_argument("--samples-dir", type=Path, default=Path("results/phase0_samples"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--durations", nargs="*", type=float, default=[15.0, 120.0],
                    help="короткая и самая длинная — вторая ловит OOM и деградацию")
    ap.add_argument("--isolate", action="store_true",
                    help="каждую модель в отдельном процессе (падение одной не мешает остальным)")
    a = ap.parse_args()

    print("среда:", describe(), flush=True)
    models = load_models_config(a.config_dir / "models.yaml")
    if a.models:
        models = [m for m in models if m["id"] in a.models]
    a.samples_dir.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    reports = []
    if a.out.exists():
        reports = json.load(open(a.out, encoding="utf-8")).get("models", [])
    seen = {r["model_id"] for r in reports}

    for cfg in models:
        if cfg["id"] in seen:
            print(f"пропуск {cfg['id']} — уже в отчёте", flush=True)
            continue
        print(f"\n=== {cfg['id']} ({cfg['hf_repo']}, {cfg.get('dtype')}) ===", flush=True)
        if a.isolate:
            cmd = [sys.executable, "-m", "scripts.phase0_check", "--models", cfg["id"],
                   "--config-dir", str(a.config_dir), "--out", str(a.out),
                   "--samples-dir", str(a.samples_dir),
                   "--durations", *[str(d) for d in a.durations]]
            subprocess.run(cmd, check=False)
            if a.out.exists():
                reports = json.load(open(a.out, encoding="utf-8")).get("models", [])
            continue
        reports.append(check_model(cfg, a.durations, a.samples_dir))
        json.dump({"env": fingerprint(), "models": reports}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    print("\n" + "=" * 72)
    print(f"{'модель':<24}{'загр.':>7}{'статус':>14}{'VRAM пик':>10}{'с/трек':>9}")
    for r in reports:
        ok = [x for x in r["runs"] if x["status"] == "ok"]
        vram = max((x.get("peak_vram_gb") or 0) for x in ok) if ok else 0
        t = np.mean([x["wall_time_s"] for x in ok]) if ok else float("nan")
        print(f"{r['model_id']:<24}{r.get('load_time_s', '—'):>7}{r['status']:>14}"
              f"{vram:>9.2f}Г{t:>9.1f}")
        for x in r["runs"]:
            if x["status"] != "ok":
                print(f"    {int(x['duration_s'])}с: {x['status']} — {x.get('error', x.get('reason'))[:110]}")
            elif x["degenerate"]:
                print(f"    {int(x['duration_s'])}с: ВЫРОЖДЕН ({', '.join(x['degeneracy_reasons'])}) — "
                      f"модель на этой длительности не даёт музыки")
    print(f"\nотчёт: {a.out}")


if __name__ == "__main__":
    main()

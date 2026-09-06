"""
Стадия 4: сводные таблицы «модель × критерий» с CI по каждой длительности,
кривые по длительности (данные для графиков), решающее правило веток 1/2 —
механически по configs/durations.yaml::decision_rule.

Вход:  per_track_metrics.jsonl, distributional_metrics.jsonl, floors.jsonl
Выход: results/summary/<D>s.json, results/summary/summary.md, decision.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .stats import between_set_ci, bootstrap_ci, combined_ci, proportion_ci


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# Ключ трека: одна клетка плана + номер набора и трека внутри неё.
_TRACK_KEY = ("model_id", "duration_s", "prompt_id", "set_idx", "track_idx")


def _load_latest(path: Path) -> list[dict]:
    """Как _load, но при повторах ключа оставляет ПОСЛЕДНЮЮ запись.

    per_track_metrics.jsonl дописывается, а не переписывается: догон
    отсутствующей метрики (например, CLAP по уже сгенерированным трекам)
    добавляет новую строку рядом со старой. Без схлопывания такие треки
    считались бы дважды, причём доля вырожденных и все доли прохождения
    критериев смещались бы в сторону старой, неполной записи.
    """
    latest = {}
    for r in _load(path):
        try:
            latest[tuple(r[k] for k in _TRACK_KEY)] = r
        except KeyError:                     # запись не про трек (пол, набор)
            latest[len(latest)] = r
    return list(latest.values())


def _get(d: dict, *keys, default=np.nan):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def _finite(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def per_track_summary(tracks: list[dict], rule: dict) -> dict:
    """tracks — все треки одной (модель, D)."""
    loop = [t for t in tracks if t.get("loop_eval", True) and "game" in t]
    n_deg = sum(1 for t in tracks if t.get("degeneracy", {}).get("degenerate"))
    reasons = {}
    for t in tracks:
        for r in t.get("degeneracy", {}).get("reasons", []):
            reasons[r] = reasons.get(r, 0) + 1
    out = {"n_tracks": len(tracks), "n_loop_tracks": len(loop),
           "degenerate": proportion_ci(n_deg, len(tracks)), "degenerate_reasons": reasons}

    def is_deg(t) -> bool:
        return bool(t.get("degeneracy", {}).get("degenerate", False))

    def cond_seam(t, key):     # key: "naive" | "best"
        if is_deg(t):
            return False       # вырожденный выход (тишина/гудок/шум) не проходит,
                                # а не исключается: модель не решила задачу
        g = t["game"]
        if key == "naive":
            return bool(g["seam_naive_ok"])
        return bool(g["best_loop"]["best_seam_ok_p95"])

    def cond_grid(t, key):
        if is_deg(t):
            return False
        g = t["game"]["grid_naive" if key == "naive" else "grid_best_loop"]
        if g.get("degenerate"):
            return False
        return (g["seam_gap_beat_error"] <= rule["seam_gap_beat_error_max"]
                and g["bar_fraction_error"] <= rule["bar_fraction_error_max"])

    for key in ("naive", "best"):
        s = sum(cond_seam(t, key) for t in loop)
        g = sum(cond_grid(t, key) for t in loop)
        both = sum(cond_seam(t, key) and cond_grid(t, key) for t in loop)
        out[f"seam_pass_{key}"] = proportion_ci(s, len(loop))
        out[f"grid_pass_{key}"] = proportion_ci(g, len(loop))
        out[f"seam_and_grid_pass_{key}"] = proportion_ci(both, len(loop))

    def mean_ci(vals):
        v = _finite(vals)
        return bootstrap_ci(v) if v else {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_observations": 0}

    out["seam_score_50ms"] = mean_ci(_get(t, "game", "seam_50ms", "seam_score") for t in loop)
    out["seam_score_200ms"] = mean_ci(_get(t, "game", "seam_200ms", "seam_score") for t in loop)
    out["best_seam_score"] = mean_ci(_get(t, "game", "best_loop", "best_seam_score") for t in loop)
    out["best_loop_duration_s"] = mean_ci(_get(t, "game", "best_loop", "loop_duration_s") for t in loop)
    out["seam_gap_beat_error_naive"] = mean_ci(_get(t, "game", "grid_naive", "seam_gap_beat_error") for t in loop)
    out["bar_fraction_error_naive"] = mean_ci(_get(t, "game", "grid_naive", "bar_fraction_error") for t in loop)
    out["regularity"] = mean_ci(_get(t, "structure", "structural_regularity", "regularity_ratio") for t in tracks)
    out["tempo_cv"] = mean_ci(_get(t, "structure", "tempo_drift", "cv") for t in tracks)
    out["key_drift"] = mean_ci(_get(t, "structure", "key_drift", "drift_fraction") for t in tracks)
    out["actual_duration_s"] = mean_ci(t.get("actual_duration_s") for t in tracks)
    out["clap"] = mean_ci(t.get("clap_score") for t in tracks)
    out["clap_by_prompt"] = {}
    by_p = defaultdict(list)
    for t in tracks:
        if "clap_score" in t:
            by_p[t["prompt_id"]].append(t["clap_score"])
    for p, v in by_p.items():
        out["clap_by_prompt"][p] = float(np.mean(v))
    return out


def distributional_summary(rows: list[dict]) -> dict:
    out = {}
    for m in ("fad", "kad"):
        vals = [r[m] for r in rows]
        between = between_set_ci(vals)
        boots = [r[f"{m}_boot"] for r in rows]
        # бутстрап-CI объединённого набора аппроксимируем средним по наборам
        boot = {"ci_lo": float(np.mean([b["ci_lo"] for b in boots])), "ci_hi": float(np.mean([b["ci_hi"] for b in boots]))}
        out[m] = {"mean": between["mean"], "k": between["k"], "between": between, "bootstrap_avg": boot,
                  "combined": combined_ci(between, boot)}
    return out


def decide(summary: dict, rule: dict) -> dict:
    """Ветка 2, если хотя бы одна модель на хотя бы одной D проходит все три
    условия; иначе ветка 1 с перечнем непройденных условий по каждой модели."""
    verdicts = {}
    for D, models in summary.items():
        clap_medians = {}
        prompts = set()
        for m in models.values():
            prompts |= set(m["tracks"]["clap_by_prompt"])
        for p in prompts:
            vals = [m["tracks"]["clap_by_prompt"][p] for m in models.values() if p in m["tracks"]["clap_by_prompt"]]
            clap_medians[p] = float(np.median(vals)) if vals else np.nan
        for mid, m in models.items():
            t = m["tracks"]
            for key in ("naive", "best"):
                seam_ok = t[f"seam_pass_{key}"]["ci_lo"] >= rule["seam_pass_fraction_min"]
                grid_ok = t[f"grid_pass_{key}"]["ci_lo"] >= rule["grid_pass_fraction_min"]
                above = [t["clap_by_prompt"][p] > clap_medians[p] for p in t["clap_by_prompt"] if np.isfinite(clap_medians[p])]
                clap_ok = bool(above) and (sum(above) / len(above) >= 0.5)
                verdicts[(mid, D, key)] = {"seam": bool(seam_ok), "grid": bool(grid_ok), "clap": bool(clap_ok),
                                          "pass": bool(seam_ok and grid_ok and clap_ok)}
    branch = 2 if any(v["pass"] for v in verdicts.values()) else 1
    return {"branch": branch,
            "verdicts": {f"{m}|{D}|{k}": v for (m, D, k), v in verdicts.items()},
            "passing": [f"{m}|{D}|{k}" for (m, D, k), v in verdicts.items() if v["pass"]]}


def fmt(ci: dict, digits=3) -> str:
    if ci is None or not np.isfinite(ci.get("point", ci.get("p", ci.get("mean", np.nan)))):
        return "—"
    p = ci.get("point", ci.get("p", ci.get("mean")))
    return f"{p:.{digits}f} [{ci['ci_lo']:.{digits}f}, {ci['ci_hi']:.{digits}f}]"


def run(out_dir: Path, config_dir: Path) -> None:
    rule = yaml.safe_load(open(config_dir / "durations.yaml", encoding="utf-8"))["decision_rule"]
    tracks = _load_latest(out_dir / "per_track_metrics.jsonl")
    dist = _load(out_dir / "distributional_metrics.jsonl")
    floors = _load(out_dir / "floors.jsonl")

    by_cell = defaultdict(list)
    for t in tracks:
        by_cell[(t["model_id"], float(t["duration_s"]))].append(t)
    by_dist = defaultdict(list)
    for r in dist:
        by_dist[(r["model_id"], float(r["duration_s"]))].append(r)

    summary = defaultdict(dict)
    for (mid, D), ts in by_cell.items():
        summary[D][mid] = {"tracks": per_track_summary(ts, rule),
                           "dist": distributional_summary(by_dist[(mid, D)]) if by_dist[(mid, D)] else None}

    sdir = out_dir / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    decision = decide(summary, rule)
    json.dump(decision, open(sdir / "decision.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    lines = [f"# Сводка (решающее правило: ветка {decision['branch']})", ""]
    for D in sorted(summary):
        json.dump(summary[D], open(sdir / f"{int(D)}s.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        lines += [f"## D = {int(D)} с", "",
                  "| модель | FAD [CI] | KAD [CI] | CLAP | вырожд. | шов ok (наив.) | шов ok (извл.) | такт ok (наив.) | такт ok (извл.) | tempo CV | key drift | факт. длит. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for mid, m in sorted(summary[D].items()):
            t, d = m["tracks"], m["dist"]
            lines.append("| " + " | ".join([
                mid,
                fmt(d["fad"]["combined"] | {"point": d["fad"]["mean"]}, 2) if d else "—",
                fmt(d["kad"]["combined"] | {"point": d["kad"]["mean"]}, 4) if d else "—",
                fmt(t["clap"]), fmt(t["degenerate"], 2), fmt(t["seam_pass_naive"], 2), fmt(t["seam_pass_best"], 2),
                fmt(t["grid_pass_naive"], 2), fmt(t["grid_pass_best"], 2),
                fmt(t["tempo_cv"]), fmt(t["key_drift"]), fmt(t["actual_duration_s"], 1)]) + " |")
        fl = [f for f in floors if float(f["duration_s"]) == D]
        if fl:
            lines += ["", "Полы: " + "; ".join(f"{f['floor']}: FAD {f['fad']:.2f}, KAD {f['kad']:.4f}" for f in fl)]
        lines.append("")
    if decision["passing"]:
        lines += ["## Прошли все три условия", ""] + [f"- {p}" for p in decision["passing"]]
    open(sdir / "summary.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    a = ap.parse_args()
    run(a.out_dir, a.config_dir)


if __name__ == "__main__":
    main()

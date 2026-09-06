"""
Стадия 3: FAD/KAD на уровне набора (модель × длительность × set) — набор
объединяет все промпты (см. configs/durations.yaml). Плюс полы:
split-half и codec round-trip (по подготовленным папкам).

Вход:  results/generation_manifest.jsonl, results/references/<D>s/*.wav
       (+ results/references/<D>s_half_A|B, results/references/<D>s_rt_<codec>)
Выход: results/distributional_metrics.jsonl, results/floors.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from .metrics import distributional as dm
from .metrics.codec_floor import split_half, split_half_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("distributional")


def _sets(manifest: Path) -> dict:
    sets = defaultdict(list)
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "done":
                sets[(r["model_id"], float(r["duration_s"]), int(r["set_idx"]))].append(Path(r["wav_path"]))
    return sets


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["model_id"], r["duration_s"], r["set_idx"]) for r in map(json.loads, open(path, encoding="utf-8"))}


def run(out_dir: Path, config_dir: Path, embedder: str, n_resamples: int, crosscheck: bool) -> None:
    dcfg = yaml.safe_load(open(config_dir / "durations.yaml", encoding="utf-8"))
    ref_root = out_dir / "references"
    out_path, floors_path = out_dir / "distributional_metrics.jsonl", out_dir / "floors.jsonl"
    sets, done = _sets(out_dir / "generation_manifest.jsonl"), _done(out_path)

    ref_ctx: dict[float, dm.ReferenceContext] = {}
    for D in sorted({d for (_, d, _) in sets}):
        ref_dir = ref_root / f"{int(D)}s"
        files = sorted(ref_dir.glob("*.wav"))
        if not files:
            log.error("нет референса %s — сначала python -m src.corpus_prep", ref_dir)
            continue
        log.info("референс D=%.0f: %d файлов, эмбеддинг...", D, len(files))
        ref_embs = dm.embed_files(files, embedder)
        ref_ctx[D] = dm.ReferenceContext(ref_embs)

        # --- полы ---
        manifest = json.load(open(ref_dir / "manifest.json", encoding="utf-8"))
        piece_of = {m["file"]: m["piece"] for m in manifest}
        # Половины берутся как ПОДМНОЖЕСТВА уже посчитанных эмбеддингов.
        # Раньше файлы копировались в отдельные папки и эмбеддились заново —
        # кэш fadtk привязан к пути, поэтому на референс уходило вдвое больше
        # работы, чем нужно, и это была самая дорогая часть всего расчёта.
        idx_a, idx_b = split_half_indices(files, piece_of)
        log.info("пол split_half D=%.0f: половины %d и %d окон", D, len(idx_a), len(idx_b))
        ctx_a = dm.ReferenceContext([ref_embs[i] for i in idx_a])
        res = dm.evaluate_set(ctx_a, [ref_embs[i] for i in idx_b], n_resamples)
        with open(floors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"duration_s": D, "floor": "split_half", "fad": res.fad, "kad": res.kad,
                                "fad_boot": res.fad_boot, "kad_boot": res.kad_boot}) + "\n")
        for rt in sorted(ref_root.glob(f"{int(D)}s_rt_*")):
            rt_files = sorted(rt.glob("*.wav"))
            if not rt_files:
                continue
            res = dm.evaluate_set(ref_ctx[D], dm.embed_files(rt_files, embedder), n_resamples)
            with open(floors_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"duration_s": D, "floor": rt.name.split("_rt_")[1], "fad": res.fad,
                                    "kad": res.kad, "fad_boot": res.fad_boot, "kad_boot": res.kad_boot}) + "\n")

    for (mid, D, k), files in sorted(sets.items()):
        if (mid, D, k) in done or D not in ref_ctx:
            continue
        log.info("набор %s D=%.0f set=%d: %d файлов", mid, D, k, len(files))
        # cached: треки с площадки без вывода аудио считаются по привезённым
        # эмбеддингам — самого аудио для них уже нет (remote_worker).
        embs = dm.embed_files_cached(sorted(files), out_dir, embedder)
        res = dm.evaluate_set(ref_ctx[D], embs, n_resamples, seed=k)
        rec = {"model_id": mid, "duration_s": D, "set_idx": k, "embedder": embedder,
               "fad": res.fad, "kad": res.kad, "fad_boot": res.fad_boot, "kad_boot": res.kad_boot,
               "n_files": res.n_files, "n_chunks": res.n_chunks, "bandwidth": res.bandwidth}
        if crosscheck and k == 0:
            eval_dir = out_dir / "sets" / mid / f"{int(D)}s" / f"set{k}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                dst = eval_dir / f"{f.parent.parent.name}_{f.name}"
                if not dst.exists():
                    dst.symlink_to(f.resolve())
            try:
                rec["toolkit_fad"] = dm.toolkit_fad(ref_root / f"{int(D)}s", eval_dir, embedder)
                rec["toolkit_fad_inf"] = dm.toolkit_fad_inf(ref_root / f"{int(D)}s", sorted(eval_dir.glob("*.wav")), embedder)
                rec["toolkit_kad"] = dm.toolkit_kad_cli(ref_root / f"{int(D)}s", eval_dir, embedder)
            except Exception as e:
                rec["toolkit_error"] = repr(e)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--embedder", default=dm.DEFAULT_EMBEDDER)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--crosscheck", action="store_true", help="сверить точечные FAD/KAD с fadtk/kadtk на set0")
    a = ap.parse_args()
    run(a.out_dir, a.config_dir, a.embedder, a.n_resamples, a.crosscheck)


if __name__ == "__main__":
    main()

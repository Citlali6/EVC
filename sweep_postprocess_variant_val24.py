"""Per-video threshold sweep under a postprocessing variant on Val24.

Each variant overrides fields of the official C00 postprocessing chain
(P0 / P0c / P18).  Each video is swept over a window around its prior best
threshold (from the merged selection) using its prior best score source,
so runtime stays bounded (~5 min per variant with 10 workers).

Output schema matches the refined scan (single source per video).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
EVC_ROOT = Path(__file__).resolve().parent
M20_CACHE = WORKSPACE / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
M10_CACHE = WORKSPACE / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"
M13_CACHE = WORKSPACE / "experiments" / "20260813_per_video_threshold_val24" / "m13_val24_raw.pt"


def decimal_grid(minimum: str, maximum: str, step: str) -> list[float]:
    lower = Decimal(minimum)
    upper = Decimal(maximum)
    increment = Decimal(step)
    count = int((upper - lower) / increment)
    return [float(lower + index * increment) for index in range(count + 1)]


def _worker_video(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay

    torch.set_num_threads(1)
    file_name = task["file_name"]
    thresholds = task["thresholds"]
    source = task["source"]
    cfg = SimpleNamespace(
        **{**routed.c00_config().__dict__, **task["cfg_overrides"]},
        roc=True, correct_thresh=0.0001,
    )
    scale_value = task.get("score_scale")
    scale = 1.0 if scale_value is None else float(scale_value)
    blend_weight = task.get("blend_weight")
    if blend_weight is not None:
        # scores = w * M10 + (1 - w) * M20
        w = float(blend_weight)
        scores_in = torch.as_tensor(task["scores_m10"]).contiguous() * w + \
                    torch.as_tensor(task["scores_m20"]).contiguous() * (1.0 - w)
    else:
        scores_in = torch.as_tensor(task["scores"]).contiguous()
    if scale != 1.0:
        scores_in = torch.clamp(scores_in * scale, 0.0, 1.0)
    record = replay.RoutedRecord(
        file_name=file_name,
        event_count=int(task["event_count"]),
        scores=scores_in,
        seg_label=torch.as_tensor(task["seg_label"]).contiguous(),
        locs=torch.as_tensor(task["locs"]).contiguous(),
        idx_label=np.asarray(task["idx_label"]),
        source_sha256=str(task["source_sha256"]),
        score_source=source,
    )
    counts = {
        float(threshold): asdict(replay.evaluate_cached_video(record, threshold, cfg))
        for threshold in thresholds
    }
    return {"file_name": file_name, "source": source, "counts_by_threshold": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--name", required=True, help="variant name")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--window", default="0.045")
    parser.add_argument("--step", default="0.0005")
    parser.add_argument("--lo", default="0.45")
    parser.add_argument("--hi", default="0.95")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-source", default=None, help="override per-video source")
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw  # bare string override (e.g. restore_mode=component)
        overrides[key] = value
    score_scale = overrides.pop("score_scale", None)
    if score_scale is not None:
        score_scale = float(score_scale)
    blend_weight = overrides.pop("blend_weight", None)
    if blend_weight is not None:
        blend_weight = float(blend_weight)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["final"]["selection"]
    if any(len(v) == 3 for v in selection.values()):
        # v2 selection: [variant, source, threshold]
        selection = {fn: (v[1], v[2]) for fn, v in selection.items()}
    window = Decimal(args.window)
    lo = Decimal(args.lo)
    hi = Decimal(args.hi)

    sys.path.insert(0, str(EVC_ROOT))
    import replay_temporal_memory_validation as replay

    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    m13 = replay.load_cache(M13_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}
    recs13 = {r["file_name"]: r for r in m13["records"]}

    tasks = []
    for file_name in sorted(selection):
        source, thr = selection[file_name]
        if args.force_source:
            source = args.force_source
        rec10 = recs10[file_name]
        rec20 = recs20[file_name]
        if source == "m10":
            scores = rec10["scores"].numpy().copy()
        elif source == "max3":
            scores = np.maximum(
                np.maximum(rec10["scores"].numpy(), rec20["scores"].numpy()),
                recs13[file_name]["scores"].numpy(),
            ).astype(np.float32, copy=True)
        elif source == "max":
            scores = np.maximum(
                rec10["scores"].numpy(), rec20["scores"].numpy()
            ).astype(np.float32, copy=True)
        else:
            scores = rec20["scores"].numpy().copy()
        coarse_min = Decimal(str(float(thr))) - window
        coarse_max = Decimal(str(float(thr))) + window
        if float(thr) >= 0.9:
            coarse_max = max(coarse_max, hi)
        if float(thr) <= 0.55:
            coarse_min = min(coarse_min, lo)
        thresholds = decimal_grid(
            str(max(coarse_min, lo)), str(min(coarse_max, hi)), args.step
        )
        tasks.append({
            "file_name": file_name,
            "source": source,
            "event_count": int(rec10["event_count"]),
            "scores": scores,
            "scores_m10": rec10["scores"].numpy().copy(),
            "scores_m20": rec20["scores"].numpy().copy(),
            "seg_label": rec10["seg_label"].numpy().copy(),
            "locs": rec10["locs"].numpy().copy(),
            "idx_label": np.asarray(rec10["idx_label"]).copy(),
            "source_sha256": str(rec10["source_sha256"]),
            "thresholds": thresholds,
            "cfg_overrides": overrides,
            "score_scale": score_scale,
            "blend_weight": blend_weight,
        })

    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker_video, task): task["file_name"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("done {}/{} {} ({}s)".format(
                len(results), len(tasks), result["file_name"], int(time.time() - started)),
                flush=True)
    results.sort(key=lambda item: item["file_name"])
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("variant_{}_scan.json".format(args.name))
    out_path.write_text(json.dumps({
        "schema": "ev-uav-postprocess-variant-scan-v1",
        "variant": args.name,
        "overrides": overrides,
        "per_video": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out_path)
    print("elapsed: {:.1f}s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

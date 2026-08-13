"""Refined per-video threshold sweep around a coarse selection on Val24.

Each video keeps the score source chosen by the coarse coordinate ascent and
sweeps thresholds in a window around the coarse threshold with a fine step
(0.0005).  Windows extend toward the low/high boundary when the coarse value
hits the coarse grid edge, so high-threshold videos can be probed up to 0.95.

Output: per_video_refined_scan.json with the same schema as the coarse scan
(only the selected source per video), ready for a second coordinate ascent.
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
    cfg = SimpleNamespace(**routed.c00_config().__dict__, roc=True, correct_thresh=0.0001)
    record = replay.RoutedRecord(
        file_name=file_name,
        event_count=int(task["event_count"]),
        scores=torch.as_tensor(task["scores"]).contiguous(),
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
    parser.add_argument("--window", default="0.06", help="half-window around coarse threshold")
    parser.add_argument("--step", default="0.0005")
    parser.add_argument("--lo", default="0.45")
    parser.add_argument("--hi", default="0.95")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["final"]["selection"]
    window = Decimal(args.window)
    lo = Decimal(args.lo)
    hi = Decimal(args.hi)

    sys.path.insert(0, str(EVC_ROOT))
    import replay_temporal_memory_validation as replay

    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}

    tasks = []
    for file_name in sorted(selection):
        source, thr = selection[file_name]
        rec10 = recs10[file_name]
        rec20 = recs20[file_name]
        record_scores = rec10 if source == "m10" else rec20
        coarse_min = Decimal(str(float(thr))) - window
        coarse_max = Decimal(str(float(thr))) + window
        # extend toward edges if the coarse value sat at a coarse grid edge
        if float(thr) >= 0.846:
            coarse_max = max(coarse_max, hi)
        if float(thr) <= 0.552:
            coarse_min = min(coarse_min, lo)
        thresholds = decimal_grid(
            str(max(coarse_min, lo)), str(min(coarse_max, hi)), args.step
        )
        tasks.append({
            "file_name": file_name,
            "source": source,
            "event_count": int(rec10["event_count"]),
            "scores": record_scores["scores"].numpy().copy(),
            "seg_label": rec10["seg_label"].numpy().copy(),
            "locs": rec10["locs"].numpy().copy(),
            "idx_label": np.asarray(rec10["idx_label"]).copy(),
            "source_sha256": str(rec10["source_sha256"]),
            "thresholds": thresholds,
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
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": "ev-uav-per-video-threshold-refined-scan-v1",
        "window": float(window),
        "step": float(Decimal(args.step)),
        "selection_used": selection,
        "per_video": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out_path)
    print("elapsed: {:.1f}s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Full-grid threshold sweep on per-event max(M10, M20) blended scores.

The blended score gives every event the higher of the two released checkpoints'
raw probabilities, so a video's best recall is never worse than its best
single-model route while the threshold still controls FP admission.  Uses the
exact official C00 postprocessing chain and evaluator.  Output schema matches
the coarse scan so the merged coordinate ascent can consume it directly.
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
    cfg = SimpleNamespace(**routed.c00_config().__dict__, roc=True, correct_thresh=0.0001)
    record = replay.RoutedRecord(
        file_name=file_name,
        event_count=int(task["event_count"]),
        scores=torch.as_tensor(task["scores"]).contiguous(),
        seg_label=torch.as_tensor(task["seg_label"]).contiguous(),
        locs=torch.as_tensor(task["locs"]).contiguous(),
        idx_label=np.asarray(task["idx_label"]),
        source_sha256=str(task["source_sha256"]),
        score_source="max",
    )
    counts = {
        float(threshold): asdict(replay.evaluate_cached_video(record, threshold, cfg))
        for threshold in thresholds
    }
    return {"file_name": file_name, "source": "max", "counts_by_threshold": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", default="0.45", dest="min_threshold")
    parser.add_argument("--max", default="0.95", dest="max_threshold")
    parser.add_argument("--step", default="0.002")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    thresholds = decimal_grid(args.min_threshold, args.max_threshold, args.step)
    sys.path.insert(0, str(EVC_ROOT))
    import replay_temporal_memory_validation as replay

    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}

    tasks = []
    for file_name in sorted(recs20):
        r10 = recs10[file_name]
        r20 = recs20[file_name]
        assert r10["event_count"] == r20["event_count"]
        blended = np.maximum(
            r10["scores"].numpy(), r20["scores"].numpy()
        ).astype(np.float32, copy=True)
        tasks.append({
            "file_name": file_name,
            "event_count": int(r10["event_count"]),
            "scores": blended,
            "seg_label": r10["seg_label"].numpy().copy(),
            "locs": r10["locs"].numpy().copy(),
            "idx_label": np.asarray(r10["idx_label"]).copy(),
            "source_sha256": str(r10["source_sha256"]),
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
        "schema": "ev-uav-per-video-max-blend-threshold-scan-v1",
        "thresholds": thresholds,
        "per_video": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out_path)
    print("elapsed: {:.1f}s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Val24 per-video threshold + score-source sweep on cached raw probabilities.

Strategy (user-authorized Val24-tuned model selection):
  * For every validation video, evaluate the complete official C00
    postprocessing chain (P0 + P0c + P18) at many thresholds, using either the
    M10 or the M20 raw score cache as the per-video score source.
  * The golden routed identity (low<=30000 -> M10@0.718, else M20@0.719) must
    reproduce the official golden counts exactly before any selection starts.
  * A coordinate-ascent pass then picks the best (source, threshold) per video
    against the pooled official Challenge 2 score.

Everything runs on CPU from the frozen raw caches; no model inference and no
training.  The result is a selection table that can be replayed into TXT
submissions by generate_submission_from_selection.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal
from types import SimpleNamespace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
EVC_ROOT = Path(__file__).resolve().parent
M20_CACHE = WORKSPACE / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
M10_CACHE = WORKSPACE / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"
GOLDEN_REPORT = WORKSPACE / "results" / "submission_m20_golden" / "offline_score_report.json"
DEFAULT_OUT = WORKSPACE / "experiments" / "20260813_per_video_threshold_val24"

SOURCES = ("m10", "m20")
COUNT_KEYS = (
    "true_positive_events",
    "false_positive_events",
    "positive_events",
    "detected_target_frames",
    "target_frames",
    "false_components",
    "frame_count",
)
GOLDEN_COUNTS = {
    "true_positive_events": 63981,
    "false_positive_events": 2396,
    "positive_events": 65506,
    "detected_target_frames": 4649,
    "target_frames": 4762,
    "false_components": 1584,
    "frame_count": 3752,
}


def decimal_grid(minimum: str, maximum: str, step: str) -> list[float]:
    lower = Decimal(minimum)
    upper = Decimal(maximum)
    increment = Decimal(step)
    if not (Decimal("0") < lower <= upper < Decimal("1")):
        raise ValueError("invalid threshold bounds")
    count = int((upper - lower) / increment)
    return [float(lower + index * increment) for index in range(count + 1)]


def add_counts(*items: dict) -> dict:
    return {name: sum(int(item[name]) for item in items) for name in COUNT_KEYS}


def metrics_from_counts(counts: dict) -> dict:
    tp = int(counts["true_positive_events"])
    fp = int(counts["false_positive_events"])
    positive = int(counts["positive_events"])
    detected = int(counts["detected_target_frames"])
    objects = int(counts["target_frames"])
    false_components = int(counts["false_components"])
    frames = int(counts["frame_count"])
    acc = float(np.float32(tp) / np.float32(positive))
    iou = float(np.float32(tp) / np.float32(positive + fp))
    pd = float(detected / objects)
    fa = float(false_components / (frames * 346 * 260))
    score_fa = float(math.exp(-10000.0 * fa))
    score = float(0.4 * pd + 0.3 * score_fa + 0.2 * iou + 0.1 * acc)
    return {"iou": iou, "acc": acc, "pd": pd, "fa": fa, "score_fa": score_fa, "score": score}


def _worker_video(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay
    from utils.density_threshold import ChallengeCountTotals

    torch.set_num_threads(1)
    file_name = task["file_name"]
    thresholds = task["thresholds"]
    cfg = SimpleNamespace(**routed.c00_config().__dict__, roc=True, correct_thresh=0.0001)
    counts_by_source = {}
    for source in SOURCES:
        scores = torch.as_tensor(task["scores_" + source]).contiguous()
        seg_label = torch.as_tensor(task["seg_label"]).contiguous()
        locs = torch.as_tensor(task["locs"]).contiguous()
        idx_label = np.asarray(task["idx_label"])
        record = replay.RoutedRecord(
            file_name=file_name,
            event_count=int(task["event_count"]),
            scores=scores,
            seg_label=seg_label,
            locs=locs,
            idx_label=idx_label,
            source_sha256=str(task["source_sha256"]),
            score_source=source,
        )
        counts_by_source[source] = {
            float(threshold): asdict(
                replay.evaluate_cached_video(record, threshold, cfg)
            )
            for threshold in thresholds
        }
    return {
        "file_name": file_name,
        "event_count": int(task["event_count"]),
        "counts_by_source": counts_by_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", default="0.55", dest="min_threshold")
    parser.add_argument("--max", default="0.85", dest="max_threshold")
    parser.add_argument("--step", default="0.002")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="debug: only first N videos")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    thresholds = decimal_grid(args.min_threshold, args.max_threshold, args.step)
    out_dir = Path(args.output).resolve()
    if out_dir.exists() and not args.force:
        raise FileExistsError("output exists: {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay

    m20, m20_sha = replay.load_cache_snapshot(M20_CACHE)
    m10, m10_sha = replay.load_cache_snapshot(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}
    assert set(recs10) == set(recs20)

    # ---- golden identity replay at routed baseline thresholds ----
    cfg = SimpleNamespace(**routed.c00_config().__dict__, roc=True, correct_thresh=0.0001)
    identity = {}
    for file_name in sorted(recs20):
        rec10 = recs10[file_name]
        rec20 = recs20[file_name]
        assert rec10["event_count"] == rec20["event_count"]
        source = rec10 if rec10["event_count"] <= 30000 else rec20
        thr = 0.718 if rec10["event_count"] <= 30000 else 0.719
        record = replay.RoutedRecord(
            file_name=file_name,
            event_count=int(rec10["event_count"]),
            scores=source["scores"].clone(),
            seg_label=rec10["seg_label"].clone(),
            locs=rec10["locs"].clone(),
            idx_label=np.ascontiguousarray(rec10["idx_label"]),
            source_sha256=str(rec10["source_sha256"]),
            score_source="secondary" if source is rec10 else "primary",
        )
        identity[file_name] = asdict(replay.evaluate_cached_video(record, thr, cfg))
    identity_pooled = add_counts(*identity.values())
    identity_metrics = metrics_from_counts(identity_pooled)
    golden_metrics = json.loads(GOLDEN_REPORT.read_text(encoding="utf-8"))["metrics"]
    matches_golden = all(
        identity_pooled[name] == GOLDEN_COUNTS[name] for name in COUNT_KEYS
    )
    print("identity pooled:", json.dumps(identity_pooled, indent=1))
    print("identity metrics:", json.dumps(identity_metrics, indent=1))
    print("golden metrics  :", json.dumps(golden_metrics, indent=1))
    print("identity matches golden counts:", matches_golden)
    if not matches_golden:
        raise RuntimeError("identity replay does not reproduce golden counts")

    # ---- per-video tasks ----
    tasks = []
    for file_name in sorted(recs20):
        rec10 = recs10[file_name]
        rec20 = recs20[file_name]
        assert rec10["event_count"] == rec20["event_count"]
        tasks.append({
            "file_name": file_name,
            "event_count": int(rec10["event_count"]),
            "scores_m10": rec10["scores"].numpy().copy(),
            "scores_m20": rec20["scores"].numpy().copy(),
            "seg_label": rec10["seg_label"].numpy().copy(),
            "locs": rec10["locs"].numpy().copy(),
            "idx_label": np.asarray(rec10["idx_label"]).copy(),
            "source_sha256": str(rec10["source_sha256"]),
            "thresholds": thresholds,
        })
    if args.limit:
        tasks = tasks[: args.limit]

    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker_video, task): task["file_name"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            elapsed = time.time() - started
            print(
                "done {}/{} {} ({}s)".format(
                    len(results), len(tasks), result["file_name"], int(elapsed)
                ),
                flush=True,
            )
    results.sort(key=lambda item: item["file_name"])
    scan_payload = {
        "schema": "ev-uav-per-video-threshold-val24-scan-v1",
        "thresholds": thresholds,
        "golden_counts": GOLDEN_COUNTS,
        "identity_pooled": identity_pooled,
        "identity_matches_golden": matches_golden,
        "per_video": results,
    }
    scan_path = out_dir / "per_video_scan.json"
    scan_path.write_text(
        json.dumps(scan_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("scan saved:", scan_path)
    print("elapsed total: {:.1f}s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

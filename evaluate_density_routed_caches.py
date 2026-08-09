"""Evaluate a three-model validation route using event count only.

The intended Challenge 2 route is:

* ``event_count <= low_max_events``: low-density cache (M10)
* ``low_max_events < event_count <= high_min_events``: middle cache (M20)
* ``event_count > high_min_events``: high-density expert cache

All caches must describe the same complete 24-video validation split and have
identical inference/code provenance.  Labels are used only by the official
validation evaluator, never by routing.
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import replay_temporal_memory_validation as replay


def _aligned_record(cache, index, reference, cache_name):
    record = cache["records"][index]
    for field in ("file_name", "event_count", "source_sha256"):
        if record[field] != reference[field]:
            raise ValueError(
                "{} alignment mismatch at record {} field {}.".format(
                    cache_name, index, field
                )
            )
    if tuple(record["scores"].shape) != tuple(reference["scores"].shape):
        raise ValueError(
            "{} score shape mismatch for {}.".format(
                cache_name, reference["file_name"]
            )
        )
    return record


def route_three_caches(low_cache, middle_cache, high_cache, low_max, high_min):
    """Build aligned routed records without consulting names or labels."""

    if int(low_max) < 0 or int(high_min) <= int(low_max):
        raise ValueError("cutoffs must satisfy 0 <= low_max < high_min.")
    replay.validate_cache_payload(low_cache, "low cache")
    replay.validate_cache_payload(middle_cache, "middle cache")
    replay.validate_cache_payload(high_cache, "high cache")
    replay._validate_cache_compatibility(middle_cache, low_cache)
    replay._validate_cache_compatibility(middle_cache, high_cache)

    routed = []
    for index, middle in enumerate(middle_cache["records"]):
        low = _aligned_record(low_cache, index, middle, "low cache")
        high = _aligned_record(high_cache, index, middle, "high cache")
        event_count = int(middle["event_count"])
        if event_count <= int(low_max):
            selected, source = low, "low"
        elif event_count > int(high_min):
            selected, source = high, "high"
        else:
            selected, source = middle, "middle"
        routed.append(
            replay.RoutedRecord(
                file_name=str(middle["file_name"]),
                event_count=event_count,
                scores=torch.as_tensor(selected["scores"])
                .reshape(-1)
                .cpu()
                .contiguous(),
                seg_label=torch.as_tensor(middle["seg_label"])
                .reshape(-1)
                .cpu()
                .contiguous(),
                locs=torch.as_tensor(middle["locs"]).cpu().contiguous(),
                idx_label=np.ascontiguousarray(middle["idx_label"]),
                source_sha256=str(middle["source_sha256"]),
                score_source=source,
            )
        )
    return routed


def evaluate_route(records, cfg, thresholds):
    """Evaluate the complete route and each density stratum from exact counts."""

    counts_by_source = {"low": [], "middle": [], "high": []}
    per_video = []
    for index, record in enumerate(records, start=1):
        threshold = float(thresholds[record.score_source])
        counts = replay.evaluate_cached_video(record, threshold, cfg)
        counts_by_source[record.score_source].append(counts)
        per_video.append(
            {
                "file_name": record.file_name,
                "event_count": record.event_count,
                "score_source": record.score_source,
                "threshold": threshold,
                "counts": asdict(counts),
            }
        )
        print(
            "evaluate {}/{}: {} -> {}".format(
                index, len(records), record.file_name, record.score_source
            ),
            flush=True,
        )

    all_counts = replay._sum_counts(
        count for values in counts_by_source.values() for count in values
    )
    strata = {}
    for source, values in counts_by_source.items():
        if not values:
            raise RuntimeError("Route produced no {}-density records.".format(source))
        counts = replay._sum_counts(values)
        strata[source] = {
            "video_count": len(values),
            "counts": asdict(counts),
            "metrics": replay.metrics_from_counts_exact(counts, cfg).to_dict(),
        }
    return {
        "metrics": replay.metrics_from_counts_exact(all_counts, cfg).to_dict(),
        "counts": asdict(all_counts),
        "strata": strata,
        "per_video": per_video,
    }


def _atomic_write_json(path, payload, force=False):
    path = Path(path).resolve()
    if path.exists() and not force:
        raise FileExistsError("Output exists; pass --force to replace: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--low-cache", type=Path, required=True)
    parser.add_argument("--middle-cache", type=Path, required=True)
    parser.add_argument("--high-cache", type=Path, required=True)
    parser.add_argument("--low-max-events", type=int, default=30000)
    parser.add_argument("--high-min-events", type=int, default=100000)
    parser.add_argument("--low-threshold", type=float, default=0.718)
    parser.add_argument("--middle-threshold", type=float, default=0.719)
    parser.add_argument("--high-threshold", type=float, default=0.719)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = {
        "low": args.low_cache.resolve(),
        "middle": args.middle_cache.resolve(),
        "high": args.high_cache.resolve(),
    }
    output_path = args.output_json.resolve()
    if any(output_path == path for path in paths.values()):
        raise ValueError("Output JSON must not overwrite an input cache.")

    cfg = replay.load_flat_config(args.config, args.override)
    snapshots = {}
    cache_hashes = {}
    for name, path in paths.items():
        snapshots[name], cache_hashes[name] = replay.load_cache_snapshot(path)
    records = route_three_caches(
        snapshots["low"],
        snapshots["middle"],
        snapshots["high"],
        args.low_max_events,
        args.high_min_events,
    )
    thresholds = {
        "low": args.low_threshold,
        "middle": args.middle_threshold,
        "high": args.high_threshold,
    }
    result = evaluate_route(records, cfg, thresholds)
    result.update(
        {
            "schema": "ev-uav-three-density-route-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision_rule": "event_count_only",
            "cutoffs": {
                "low_max_events_inclusive": int(args.low_max_events),
                "high_min_events_exclusive": int(args.high_min_events),
            },
            "thresholds": thresholds,
            "caches": {
                name: {
                    "path": str(path),
                    "file_sha256": cache_hashes[name],
                    "checkpoint_sha256": snapshots[name]["metadata"][
                        "checkpoint_sha256"
                    ],
                }
                for name, path in paths.items()
            },
            "dataset_signature": snapshots["middle"]["metadata"][
                "dataset_signature"
            ],
            "config_path": str(args.config.resolve()),
            "config_overrides": list(args.override),
            "command": list(sys.argv),
        }
    )
    _atomic_write_json(output_path, result, force=args.force)
    print(json.dumps(result["metrics"], sort_keys=True))
    print("result:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

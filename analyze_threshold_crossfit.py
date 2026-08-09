"""Pre-registered five-fold stability audit for the M20 high-density threshold.

This script deliberately recomputes per-video sufficient counts through the
project's real postprocessor.  It never averages fold Scores: held-out integer
counts are pooled and passed through the official metric implementation once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import replay_temporal_memory_validation as replay


LOW_THRESHOLD = 0.718
REFERENCE_HIGH_THRESHOLD = 0.719
DENSITY_CUTOFF = 30000
HIGH_THRESHOLDS = replay.decimal_grid("0.7150", "0.7250", "0.0001")
EXPECTED_BASELINE = {
    "iou": 0.9422550201,
    "acc": 0.9767196774,
    "pd": 0.9762704746,
    "fa": 4.6929172975e-06,
    "score_fa": 0.9541549752,
    "score": 0.9628776542,
}


def fold_for_file(file_name: str) -> int:
    stem = Path(file_name).stem
    if not stem.startswith("val_") or len(stem) != len("val_000"):
        raise ValueError("Non-canonical validation file name: {}".format(file_name))
    suffix = stem[-3:]
    if not suffix.isdigit() or not 0 <= int(suffix) < 24:
        raise ValueError("Non-canonical validation file name: {}".format(file_name))
    return int(suffix) % 5


def _selected_counts(items: Sequence[Mapping], high_threshold: float):
    selected = []
    for item in items:
        threshold = (
            high_threshold
            if int(item["event_count"]) > DENSITY_CUTOFF
            else LOW_THRESHOLD
        )
        selected.append(item["counts_by_threshold"][float(threshold)])
    return replay._sum_counts(selected)


def _metrics(items: Sequence[Mapping], high_threshold: float, cfg):
    return replay.metrics_from_counts_exact(_selected_counts(items, high_threshold), cfg)


def choose_threshold(items: Sequence[Mapping], cfg) -> tuple[float, dict]:
    candidates = []
    for threshold in HIGH_THRESHOLDS:
        metrics = _metrics(items, threshold, cfg).to_dict()
        candidates.append((float(threshold), metrics))
    candidates.sort(
        key=lambda item: (
            -item[1]["score"],
            abs(item[0] - REFERENCE_HIGH_THRESHOLD),
            item[0],
        )
    )
    return candidates[0]


def _delta(candidate: Mapping, baseline: Mapping) -> dict:
    return {
        name: float(candidate[name]) - float(baseline[name])
        for name in replay.METRIC_NAMES
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--primary-cache", type=Path, required=True)
    parser.add_argument("--secondary-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_path = args.output_json.resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError("Output already exists; pass --force: {}".format(output_path))

    cfg = replay.load_flat_config(args.config, args.override)
    primary, primary_sha256 = replay.load_cache_snapshot(args.primary_cache)
    secondary, secondary_sha256 = replay.load_cache_snapshot(args.secondary_cache)
    records = replay.route_cache_records(primary, secondary, DENSITY_CUTOFF)
    expected_names = {"val_{:03d}.npz".format(index) for index in range(24)}
    actual_names = {record.file_name for record in records}
    if actual_names != expected_names or len(records) != 24:
        raise ValueError("Cross-fit requires exactly canonical val_000..val_023 records.")

    prepared = replay.precompute_video_counts(
        records,
        DENSITY_CUTOFF,
        (LOW_THRESHOLD,),
        HIGH_THRESHOLDS,
        cfg,
    )
    baseline_counts = _selected_counts(prepared, REFERENCE_HIGH_THRESHOLD)
    baseline = replay.metrics_from_counts_exact(baseline_counts, cfg).to_dict()
    replay.verify_formatted_metrics(baseline, EXPECTED_BASELINE)

    full_threshold, full_metrics = choose_threshold(prepared, cfg)
    full_delta = _delta(full_metrics, baseline)

    held_selected_counts = []
    held_baseline_counts = []
    fold_results = []
    selected_thresholds = []
    for fold in range(5):
        train_items = [item for item in prepared if fold_for_file(item["file_name"]) != fold]
        held_items = [item for item in prepared if fold_for_file(item["file_name"]) == fold]
        selected_threshold, train_metrics = choose_threshold(train_items, cfg)
        selected_thresholds.append(selected_threshold)
        selected_counts = _selected_counts(held_items, selected_threshold)
        reference_counts = _selected_counts(held_items, REFERENCE_HIGH_THRESHOLD)
        held_selected_counts.append(selected_counts)
        held_baseline_counts.append(reference_counts)
        selected_metrics = replay.metrics_from_counts_exact(selected_counts, cfg).to_dict()
        reference_metrics = replay.metrics_from_counts_exact(reference_counts, cfg).to_dict()
        fold_results.append(
            {
                "fold": fold,
                "files": [item["file_name"] for item in held_items],
                "train_selected_high_threshold": selected_threshold,
                "train_metrics": train_metrics,
                "held_metrics": selected_metrics,
                "held_baseline_metrics": reference_metrics,
                "held_delta": _delta(selected_metrics, reference_metrics),
                "held_counts": asdict(selected_counts),
                "held_baseline_counts": asdict(reference_counts),
            }
        )

    oof_counts = replay._sum_counts(held_selected_counts)
    oof_baseline_counts = replay._sum_counts(held_baseline_counts)
    oof_metrics = replay.metrics_from_counts_exact(oof_counts, cfg).to_dict()
    oof_baseline = replay.metrics_from_counts_exact(oof_baseline_counts, cfg).to_dict()
    oof_delta = _delta(oof_metrics, oof_baseline)

    threshold_counts = Counter(selected_thresholds)
    modal_threshold, modal_count = sorted(
        threshold_counts.items(),
        key=lambda item: (-item[1], abs(item[0] - REFERENCE_HIGH_THRESHOLD), item[0]),
    )[0]
    modal_metrics = _metrics(prepared, modal_threshold, cfg).to_dict()
    modal_delta = _delta(modal_metrics, baseline)
    fold_score_deltas = [item["held_delta"]["score"] for item in fold_results]

    gates = {
        "oof_score_delta_at_least_1e-4": oof_delta["score"] >= 1e-4,
        "at_least_four_nonnegative_folds": sum(value >= 0 for value in fold_score_deltas) >= 4,
        "at_least_three_positive_folds": sum(value > 0 for value in fold_score_deltas) >= 3,
        "worst_fold_at_least_minus_1e-4": min(fold_score_deltas) >= -1e-4,
        "modal_threshold_selected_at_least_three_times": modal_count >= 3,
        "oof_pd_not_lower": oof_delta["pd"] >= 0,
        "oof_fa_delta_at_most_1e-8": oof_delta["fa"] <= 1e-8,
        "oof_iou_delta_at_least_minus_2e-4": oof_delta["iou"] >= -2e-4,
        "modal_full_score_at_least_baseline_plus_1e-4": modal_delta["score"] >= 1e-4,
    }
    promoted = all(gates.values())

    payload = {
        "schema": "evc-threshold-crossfit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_note": (
            "Nested five-fold OOF on the same labeled validation set; even a passing "
            "candidate requires hidden-test or new held-out confirmation."
        ),
        "protocol": {
            "fold_rule": "int(val_NNN suffix) mod 5",
            "low_threshold": LOW_THRESHOLD,
            "reference_high_threshold": REFERENCE_HIGH_THRESHOLD,
            "high_threshold_min": HIGH_THRESHOLDS[0],
            "high_threshold_max": HIGH_THRESHOLDS[-1],
            "high_threshold_step": 0.0001,
            "high_threshold_count": len(HIGH_THRESHOLDS),
            "density_cutoff": DENSITY_CUTOFF,
            "tie_break": "highest Score, nearest 0.719, then lower threshold",
        },
        "inputs": {
            "primary_cache": str(args.primary_cache.resolve()),
            "primary_cache_sha256": primary_sha256,
            "secondary_cache": str(args.secondary_cache.resolve()),
            "secondary_cache_sha256": secondary_sha256,
            "dataset_signature": primary["metadata"]["dataset_signature"],
            "config": str(args.config.resolve()),
            "config_overrides": list(args.override),
            "script_sha256": _sha256_file(Path(__file__)),
        },
        "baseline": baseline,
        "baseline_counts": asdict(baseline_counts),
        "full_validation_descriptive": {
            "selected_high_threshold": full_threshold,
            "metrics": full_metrics,
            "delta": full_delta,
        },
        "folds": fold_results,
        "selected_threshold_counts": {
            "{:.4f}".format(key): value for key, value in sorted(threshold_counts.items())
        },
        "modal_threshold": modal_threshold,
        "modal_threshold_count": modal_count,
        "modal_full_metrics": modal_metrics,
        "modal_full_delta": modal_delta,
        "oof_metrics": oof_metrics,
        "oof_baseline_metrics": oof_baseline,
        "oof_delta": oof_delta,
        "oof_counts": asdict(oof_counts),
        "oof_baseline_counts": asdict(oof_baseline_counts),
        "gates": gates,
        "promoted_for_hidden_test": promoted,
    }
    replay._write_json(output_path, payload)
    print("full winner: high={:.4f} Score={:.10f} delta={:+.10f}".format(
        full_threshold, full_metrics["score"], full_delta["score"]
    ))
    print("OOF Score={:.10f} delta={:+.10f}".format(oof_metrics["score"], oof_delta["score"]))
    print("selected thresholds:", ", ".join("{:.4f}".format(x) for x in selected_thresholds))
    print("promoted for hidden test:", promoted)
    print("report:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

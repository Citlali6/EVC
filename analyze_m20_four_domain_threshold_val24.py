"""Validation-aware, cache-only M20 domain-threshold diagnostic.

This is explicitly a competition-model-selection diagnostic, not held-out
evidence.  It never loads a model and has no test or submission code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent
EVC_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = EVC_ROOT / "protocols" / "m20_four_domain_threshold_val24_competition_v1.json"
EXPECTED_PROTOCOL_SHA256 = "6fc964b58bb2efc04b5fd2d93e33dab913150594a001c2efc2d31cdfe352315d"
M20_VAL_CACHE = WORKSPACE / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
M10_VAL_CACHE = WORKSPACE / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"
V2_REPORT = WORKSPACE / "experiments" / "20260811_metric_aux_task_arithmetic_wfull_val24_v2_recovery" / "frozen_validation_report.json"
OUTPUT_ROOT = WORKSPACE / "experiments" / "20260811_m20_four_domain_threshold_val24_competition_v1"
CURVE_PATH = OUTPUT_ROOT / "threshold_curves.json"
REPORT_PATH = OUTPUT_ROOT / "threshold_scan_report.json"
DOMAINS = ("middle", "h1", "h2")
COUNT_KEYS = (
    "true_positive_events",
    "false_positive_events",
    "positive_events",
    "detected_target_frames",
    "target_frames",
    "false_components",
    "frame_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol() -> dict:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Frozen threshold diagnostic protocol SHA-256 differs.")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol.get("schema") != "ev-uav-m20-four-domain-threshold-val24-competition-science-v1":
        raise ValueError("Threshold diagnostic protocol schema differs.")
    return protocol


def grid_from_protocol(protocol: dict) -> list[float]:
    grid = protocol["threshold_search"]["grid"]
    values = [index / int(grid["integer_scale"]) for index in range(int(grid["integer_start"]), int(grid["integer_stop_inclusive"]) + 1)]
    if len(values) != int(grid["count"]):
        raise RuntimeError("Threshold-grid count differs.")
    return values


def add_counts(*items: dict) -> dict:
    return {name: sum(int(item[name]) for item in items) for name in COUNT_KEYS}


def replace_domain_counts(total: dict, baseline_domain: dict, candidate_domain: dict) -> dict:
    return {name: int(total[name]) - int(baseline_domain[name]) + int(candidate_domain[name]) for name in COUNT_KEYS}


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


def _worker_domain(domain: str, thresholds: list[float]) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import replay_temporal_memory_validation as replay
    import evaluate_metric_aux_task_arithmetic_validation as frozen_val

    torch.set_num_threads(1)
    primary, primary_sha = replay.load_cache_snapshot(M20_VAL_CACHE)
    secondary, secondary_sha = replay.load_cache_snapshot(M10_VAL_CACHE)
    if primary_sha != "6c9b4a8e33217aac7a05c78590a7feb6db6e6fc332b6411d7603264687710304":
        raise RuntimeError("M20 validation cache identity differs.")
    if secondary_sha != "96a9dfa8833e6f609d29f4db9d8f7196c84c7e92c7026cce734b97ddf133622f":
        raise RuntimeError("M10 validation cache identity differs.")
    records = replay.route_cache_records(primary, secondary, 30000)
    report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    route = {item["file_name"]: item["route"]["domain"] for item in report["per_video"]}
    selected = [record for record in records if route[Path(record.file_name).name] == domain]
    cfg = frozen_val._c00_config()
    results = []
    for threshold in thresholds:
        pooled = None
        for record in selected:
            current = asdict(replay.evaluate_cached_video(record, threshold, cfg))
            pooled = current if pooled is None else add_counts(pooled, current)
        results.append(pooled)
    return {
        "domain": domain,
        "record_names": [Path(record.file_name).name for record in selected],
        "counts": results,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


def _fixed_low_counts() -> tuple[dict, list[str]]:
    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import replay_temporal_memory_validation as replay
    import evaluate_metric_aux_task_arithmetic_validation as frozen_val

    primary, _ = replay.load_cache_snapshot(M20_VAL_CACHE)
    secondary, _ = replay.load_cache_snapshot(M10_VAL_CACHE)
    records = replay.route_cache_records(primary, secondary, 30000)
    report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    route = {item["file_name"]: item["route"]["domain"] for item in report["per_video"]}
    selected = [record for record in records if route[Path(record.file_name).name] == "low"]
    cfg = frozen_val._c00_config()
    counts = [asdict(replay.evaluate_cached_video(record, 0.718, cfg)) for record in selected]
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only threshold diagnostic initialized CUDA.")
    return add_counts(*counts), [Path(record.file_name).name for record in selected]


def choose_best(curve: list[dict], baseline: float = 0.719) -> dict:
    return min(
        curve,
        key=lambda item: (
            -float(item["metrics"]["score"]),
            abs(float(item["threshold"]) - baseline),
            float(item["threshold"]),
        ),
    )


def intervals(indices: list[int], thresholds: list[float]) -> list[dict]:
    if not indices:
        return []
    groups = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            groups.append({"start": thresholds[start], "end": thresholds[previous], "grid_points": previous - start + 1})
            start = index
        previous = index
    groups.append({"start": thresholds[start], "end": thresholds[previous], "grid_points": previous - start + 1})
    return groups


def _joint_best(thresholds: list[float], low_counts: dict, domain_counts: dict) -> dict:
    arrays = {
        domain: {name: np.asarray([row[name] for row in domain_counts[domain]], dtype=np.int64) for name in COUNT_KEYS}
        for domain in DOMAINS
    }
    best = None
    for middle_index, middle_threshold in enumerate(thresholds):
        tp = low_counts["true_positive_events"] + arrays["middle"]["true_positive_events"][middle_index] + arrays["h1"]["true_positive_events"][:, None] + arrays["h2"]["true_positive_events"][None, :]
        fp = low_counts["false_positive_events"] + arrays["middle"]["false_positive_events"][middle_index] + arrays["h1"]["false_positive_events"][:, None] + arrays["h2"]["false_positive_events"][None, :]
        detected = low_counts["detected_target_frames"] + arrays["middle"]["detected_target_frames"][middle_index] + arrays["h1"]["detected_target_frames"][:, None] + arrays["h2"]["detected_target_frames"][None, :]
        false_components = low_counts["false_components"] + arrays["middle"]["false_components"][middle_index] + arrays["h1"]["false_components"][:, None] + arrays["h2"]["false_components"][None, :]
        positive = low_counts["positive_events"] + arrays["middle"]["positive_events"][middle_index] + arrays["h1"]["positive_events"][:, None] + arrays["h2"]["positive_events"][None, :]
        objects = low_counts["target_frames"] + arrays["middle"]["target_frames"][middle_index] + arrays["h1"]["target_frames"][:, None] + arrays["h2"]["target_frames"][None, :]
        frames = low_counts["frame_count"] + arrays["middle"]["frame_count"][middle_index] + arrays["h1"]["frame_count"][:, None] + arrays["h2"]["frame_count"][None, :]
        acc = (tp.astype(np.float32) / positive.astype(np.float32)).astype(np.float64)
        iou = (tp.astype(np.float32) / (positive + fp).astype(np.float32)).astype(np.float64)
        pd = detected / objects
        fa = false_components / (frames * 346 * 260)
        score = 0.4 * pd + 0.3 * np.exp(-10000.0 * fa) + 0.2 * iou + 0.1 * acc
        local_max = float(score.max())
        for h1_index, h2_index in np.argwhere(np.isclose(score, local_max, rtol=0.0, atol=1e-15)):
            threshold_tuple = (middle_threshold, thresholds[int(h1_index)], thresholds[int(h2_index)])
            key = (-local_max, sum(abs(value - 0.719) for value in threshold_tuple), threshold_tuple)
            if best is None or key < best[0]:
                counts = {
                    name: int(low_counts[name] + arrays["middle"][name][middle_index] + arrays["h1"][name][int(h1_index)] + arrays["h2"][name][int(h2_index)])
                    for name in COUNT_KEYS
                }
                best = (key, threshold_tuple, counts)
    return {"thresholds": dict(zip(DOMAINS, best[1])), "counts": best[2], "metrics": metrics_from_counts(best[2])}


def run_val(workers: int) -> dict:
    protocol = load_protocol()
    thresholds = grid_from_protocol(protocol)
    if OUTPUT_ROOT.exists():
        raise FileExistsError("Refusing to overwrite threshold diagnostic output: {}".format(OUTPUT_ROOT))
    started = time.time()
    domain_results = {}
    with ProcessPoolExecutor(max_workers=min(max(1, workers), 3)) as pool:
        futures = {pool.submit(_worker_domain, domain, thresholds): domain for domain in DOMAINS}
        for future in as_completed(futures):
            result = future.result()
            if result["cuda_initialized"]:
                raise RuntimeError("A threshold worker initialized CUDA.")
            domain_results[result["domain"]] = result
            print("completed val domain {}".format(result["domain"]), flush=True)
    low_counts, low_names = _fixed_low_counts()
    baseline_index = thresholds.index(0.719)
    domain_counts = {domain: domain_results[domain]["counts"] for domain in DOMAINS}
    baseline_domain = {domain: domain_counts[domain][baseline_index] for domain in DOMAINS}
    baseline_counts = add_counts(low_counts, *(baseline_domain[domain] for domain in DOMAINS))
    baseline_metrics = metrics_from_counts(baseline_counts)
    v2_report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    if baseline_counts != v2_report["aggregate"]["baseline"]["counts"] or baseline_metrics != v2_report["aggregate"]["baseline"]["metrics"]:
        raise RuntimeError("Released-threshold cache replay does not match the frozen golden Val24 baseline.")

    curves = {}
    optima = {}
    for domain in DOMAINS:
        rows = []
        for threshold, counts in zip(thresholds, domain_counts[domain]):
            full_counts = replace_domain_counts(baseline_counts, baseline_domain[domain], counts)
            metrics = metrics_from_counts(full_counts)
            rows.append({"threshold": threshold, "domain_counts": counts, "full_counts": full_counts, "metrics": metrics, "score_delta": metrics["score"] - baseline_metrics["score"]})
        curves[domain] = rows
        optima[domain] = choose_best(rows)

    combined_counts = baseline_counts
    for domain in DOMAINS:
        combined_counts = replace_domain_counts(combined_counts, baseline_domain[domain], optima[domain]["domain_counts"])
    combined = {"thresholds": {domain: optima[domain]["threshold"] for domain in DOMAINS}, "counts": combined_counts, "metrics": metrics_from_counts(combined_counts)}
    joint = _joint_best(thresholds, low_counts, domain_counts)
    target = float(protocol["objective"]["target_full_val24_score_delta"])
    summary = {
        "schema": "ev-uav-m20-four-domain-threshold-val24-competition-report-v1",
        "created_utc_epoch": time.time(),
        "evidence_class": protocol["evidence_class"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "input_sha256": {
            "m20_val24_raw_cache": sha256_file(M20_VAL_CACHE),
            "m10_val24_raw_cache": sha256_file(M10_VAL_CACHE),
            "v2_validation_report": sha256_file(V2_REPORT),
        },
        "grid": protocol["threshold_search"]["grid"],
        "route_population": {"low": len(low_names), **{domain: len(domain_results[domain]["record_names"]) for domain in DOMAINS}},
        "route_names": {"low": low_names, **{domain: domain_results[domain]["record_names"] for domain in DOMAINS}},
        "baseline": {"thresholds": {"low": 0.718, "middle": 0.719, "h1": 0.719, "h2": 0.719}, "counts": baseline_counts, "metrics": baseline_metrics},
        "one_at_a_time_optima": {domain: optima[domain] for domain in DOMAINS},
        "combined_one_at_a_time_optima": {**combined, "score_delta": combined["metrics"]["score"] - baseline_metrics["score"]},
        "joint_cartesian_grid_optimum": {**joint, "score_delta": joint["metrics"]["score"] - baseline_metrics["score"]},
        "target_score_delta": target,
        "target_met": joint["metrics"]["score"] - baseline_metrics["score"] >= target,
        "target_full_score_0_9700_met": joint["metrics"]["score"] >= 0.9700,
        "finite_grid_upper_bound_only": True,
        "boundary_hit": any(value in (thresholds[0], thresholds[-1]) for value in joint["thresholds"].values()),
        "elapsed_seconds": time.time() - started,
        "training_or_model_inference_performed": False,
        "test_read": False,
        "submission_or_default_changed": False,
    }
    curve_payload = {
        "schema": "ev-uav-m20-four-domain-threshold-val24-curves-v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "thresholds": thresholds,
        "baseline": summary["baseline"],
        "domains": {domain: {"record_names": domain_results[domain]["record_names"], "curve": curves[domain]} for domain in DOMAINS},
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    with CURVE_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(curve_payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    summary["curve_report_sha256"] = sha256_file(CURVE_PATH)
    with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"report": str(REPORT_PATH), "report_sha256": sha256_file(REPORT_PATH), "baseline_score": baseline_metrics["score"], "one_at_a_time": {d: (optima[d]["threshold"], optima[d]["score_delta"]) for d in DOMAINS}, "combined": (combined["thresholds"], summary["combined_one_at_a_time_optima"]["score_delta"]), "joint": (joint["thresholds"], summary["joint_cartesian_grid_optimum"]["score_delta"], joint["metrics"]["score"]), "target_met": summary["target_met"]}, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-val",))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run_val(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

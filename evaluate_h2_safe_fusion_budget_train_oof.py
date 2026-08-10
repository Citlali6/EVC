"""CPU-only adaptive train confirmation for H2 full/T32 safe fusion v2.

The candidate is the frozen v1 k=4 component increment followed by a complete-
source, prediction-only changed-event budget.  This experiment may read only the
official 99-source train population and the formal train-v3 score cache.  The v2
protocol explicitly records that this is post-v1 adaptive train evidence rather
than an independent OOF estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time
from typing import Optional, Sequence

import numpy as np

import evaluate_h2_safe_fusion_train_oof as base


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
PROTOCOL_PATH = (
    PROJECT_ROOT / "protocols" / "h2_safe_fusion_train_oof_v2.json"
).resolve()
EXPECTED_PROTOCOL_SHA256 = "695855adfce9d9b61fb45629d088afe073d9e77da39482daa07d64bea29986d8"
EXPECTED_MANIFEST_SHA256 = base.EXPECTED_MANIFEST_SHA256
EXPECTED_TRAIN_PROTOCOL_SHA256 = base.EXPECTED_TRAIN_PROTOCOL_SHA256
EXPECTED_V1_PROTOCOL_SHA256 = base.EXPECTED_PROTOCOL_SHA256
EXPECTED_V1_REPORT_SHA256 = "6be94c3d2aff9175ace33552e22a26ff9a3b10377208a4976fbe85eca88c771e"
EXPECTED_C00_SHA256 = base.EXPECTED_C00_SHA256
REPORT_SCHEMA = "ev-uav-h2-safe-fusion-train-oof-report-v2"

OFFICIAL_NAMES = base.OFFICIAL_NAMES
GROUPS = base.GROUPS
H2_NAMES = base.H2_NAMES
ALPHAS = base.ALPHAS
MAX_CHANGED_EVENTS = (32, 64, 128, 256)
ANCHOR_MIN = 4
THRESHOLD = base.THRESHOLD
TEMPORAL_FRAME_SIZE = base.TEMPORAL_FRAME_SIZE
WIDTH = base.WIDTH
HEIGHT = base.HEIGHT
COUNT_KEYS = base.COUNT_KEYS


def candidate_catalog():
    candidates = []
    for alpha in ALPHAS:
        suffix = "{:03d}".format(int(round(alpha * 100)))
        for max_changed_events in MAX_CHANGED_EVENTS:
            candidates.append(
                {
                    "candidate_id": "budget_a{}_m{:03d}".format(
                        suffix, max_changed_events
                    ),
                    "family": "component_increment_change_budget",
                    "alpha": alpha,
                    "anchor_min": ANCHOR_MIN,
                    "max_changed_events": max_changed_events,
                }
            )
    return tuple(candidates)


def apply_changed_event_budget(
    full_scores, raw_increment_scores, max_changed_events
):
    """Apply the frozen prediction-only complete-source abstention rule."""
    if int(max_changed_events) not in MAX_CHANGED_EVENTS:
        raise ValueError("max_changed_events is outside the frozen grid.")
    full = np.asarray(full_scores, dtype=np.float32).reshape(-1)
    raw = np.asarray(raw_increment_scores, dtype=np.float32).reshape(-1)
    if full.shape != raw.shape:
        raise ValueError("Full and increment score vectors must be aligned.")
    if not np.isfinite(full).all() or not np.isfinite(raw).all():
        raise ValueError("Score vectors contain non-finite values.")
    if np.any(raw < full):
        raise ValueError("Increment candidate lowered a full score.")
    changed_event_count = int(np.count_nonzero(raw != full))
    abstained = changed_event_count > int(max_changed_events)
    output = full if abstained else raw
    if abstained and not np.array_equal(output, full):
        raise RuntimeError("Budget abstention did not return exact full scores.")
    return output, {
        "changed_event_count": changed_event_count,
        "max_changed_events": int(max_changed_events),
        "video_abstained": abstained,
        "output_changed_events": 0 if abstained else changed_event_count,
    }


def _workspace_input(relative_path, description):
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("{} must be workspace-relative.".format(description))
    resolved = (WORKSPACE_ROOT / relative).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError("{} escapes the workspace root.".format(description)) from error
    return resolved


def validate_protocol():
    protocol = base._load_json_snapshot(
        PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "v2 protocol"
    )
    if (
        protocol.get("schema") != "ev-uav-h2-safe-fusion-train-oof-protocol-v2"
        or protocol.get("status") != "frozen_before_v2_score_recomputation"
        or protocol.get("evidence_class")
        != "adaptive_train_only_confirmation_after_v1_not_independent_oof"
        or protocol.get("split_access", {}).get("gpu_allowed") is not False
    ):
        raise ValueError("Frozen adaptive train-only protocol identity differs.")
    if protocol["population"]["grouped_folds"] != {
        group: list(names) for group, names in GROUPS.items()
    }:
        raise ValueError("Grouped folds differ from the frozen 4/3/4 split.")
    generation = protocol["candidate_generation"]
    if (
        generation["alpha_grid"] != list(ALPHAS)
        or generation["fixed_anchor_min"] != ANCHOR_MIN
        or generation["max_changed_events_grid"] != list(MAX_CHANGED_EVENTS)
        or generation["candidate_ids"]
        != [item["candidate_id"] for item in candidate_catalog()]
    ):
        raise ValueError("Candidate catalog differs from the frozen protocol.")
    if protocol["fixed_evaluation"] != {
        "prediction_threshold": THRESHOLD,
        "temporal_frame_size": TEMPORAL_FRAME_SIZE,
        "resolution": [WIDTH, HEIGHT],
        "postprocess_profile": "released_M20_C00_fixed",
        "postprocess_sha256": EXPECTED_C00_SHA256,
        "metric_implementation": "run_temporal_memory_input_route_train.evaluate_one/evaluation",
        "reported_metrics": ["score", "pd", "fa", "iou", "acc", "score_fa"],
        "reported_counts": list(COUNT_KEYS),
    }:
        raise ValueError("Frozen evaluation settings differ.")
    expected_input_hashes = {
        "formal_train_v3_cache_manifest": EXPECTED_MANIFEST_SHA256,
        "formal_train_v3_protocol": EXPECTED_TRAIN_PROTOCOL_SHA256,
        "v1_protocol": EXPECTED_V1_PROTOCOL_SHA256,
        "v1_train_only_report": EXPECTED_V1_REPORT_SHA256,
    }
    for key, expected_hash in expected_input_hashes.items():
        if protocol["inputs"][key]["sha256"] != expected_hash:
            raise ValueError("Frozen input hash differs: {}".format(key))
    return protocol


def select_candidate(
    group_baseline_counts, group_candidate_counts, development_groups
):
    """Select using only the supplied development groups and frozen v2 order."""
    development_groups = tuple(development_groups)
    if not development_groups or any(group not in GROUPS for group in development_groups):
        raise ValueError("development_groups must be a non-empty frozen-group subset.")
    baseline_by_group = {
        group: base.evaluation(group_baseline_counts[group])
        for group in development_groups
    }
    pooled_baseline = base.evaluation(
        base.add_counts(*(group_baseline_counts[group] for group in development_groups))
    )
    rows = []
    for candidate in candidate_catalog():
        candidate_id = candidate["candidate_id"]
        candidate_by_group = {
            group: base.evaluation(group_candidate_counts[candidate_id][group])
            for group in development_groups
        }
        pooled_candidate = base.evaluation(
            base.add_counts(
                *(group_candidate_counts[candidate_id][group] for group in development_groups)
            )
        )
        group_deltas = {
            group: base.evaluation_delta(
                baseline_by_group[group], candidate_by_group[group]
            )
            for group in development_groups
        }
        pooled_delta = base.evaluation_delta(pooled_baseline, pooled_candidate)
        group_gates = {
            group: base.comparison_gates(
                baseline_by_group[group], candidate_by_group[group], strict_score=True
            )
            for group in development_groups
        }
        pooled_gates = base.comparison_gates(
            pooled_baseline, pooled_candidate, strict_score=True
        )
        eligible = all(pooled_gates.values()) and all(
            all(gates.values()) for gates in group_gates.values()
        )
        rows.append(
            {
                **candidate,
                "eligible": eligible,
                "development_groups": list(development_groups),
                "group_deltas": group_deltas,
                "group_gates": group_gates,
                "pooled": {
                    "baseline": pooled_baseline,
                    "candidate": pooled_candidate,
                    "delta": pooled_delta,
                    "gates": pooled_gates,
                },
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        return {
            "selected_candidate_id": "full_abstain",
            "abstained": True,
            "development_groups": list(development_groups),
            "candidate_rows": rows,
        }

    def rank(row):
        group_score_deltas = [
            row["group_deltas"][group]["metrics"]["score"]
            for group in development_groups
        ]
        pooled = row["pooled"]["delta"]["metrics"]
        return (
            -min(group_score_deltas),
            -pooled["score"],
            -pooled["pd"],
            -pooled["iou"],
            pooled["fa"],
            row["alpha"],
            row["max_changed_events"],
            row["candidate_id"],
        )

    chosen = sorted(eligible_rows, key=rank)[0]
    return {
        "selected_candidate_id": chosen["candidate_id"],
        "abstained": False,
        "development_groups": list(development_groups),
        "selected_rank_inputs": {
            "minimum_group_score_delta": min(
                chosen["group_deltas"][group]["metrics"]["score"]
                for group in development_groups
            ),
            "pooled_delta": chosen["pooled"]["delta"],
            "alpha": chosen["alpha"],
            "anchor_min": chosen["anchor_min"],
            "max_changed_events": chosen["max_changed_events"],
        },
        "candidate_rows": rows,
    }


def run(output_directory):
    started = time.monotonic()
    protocol = validate_protocol()
    inputs = protocol["inputs"]
    manifest_path = _workspace_input(
        inputs["formal_train_v3_cache_manifest"]["workspace_relative_path"],
        "cache manifest",
    )
    train_protocol_path = _workspace_input(
        inputs["formal_train_v3_protocol"]["workspace_relative_path"],
        "formal train protocol",
    )
    v1_protocol_path = _workspace_input(
        inputs["v1_protocol"]["workspace_relative_path"], "v1 protocol"
    )
    v1_report_path = _workspace_input(
        inputs["v1_train_only_report"]["workspace_relative_path"], "v1 report"
    )
    train_root = _workspace_input(
        inputs["train_root_workspace_relative_path"], "train root"
    )
    manifest = base._load_json_snapshot(
        manifest_path, EXPECTED_MANIFEST_SHA256, "cache manifest"
    )
    fixed_inputs = (
        (train_protocol_path, EXPECTED_TRAIN_PROTOCOL_SHA256, "formal train protocol"),
        (v1_protocol_path, EXPECTED_V1_PROTOCOL_SHA256, "v1 protocol"),
        (v1_report_path, EXPECTED_V1_REPORT_SHA256, "v1 train-only report"),
    )
    for path, expected_hash, description in fixed_inputs:
        if base.sha256_file(path) != expected_hash:
            raise ValueError("{} SHA-256 differs.".format(description))
    records = manifest.get("records")
    if (
        manifest.get("schema") != "ev-uav-temporal-input-route-train-cache-v1"
        or manifest.get("complete") is not True
        or manifest.get("video_count") != 99
        or manifest.get("event_count") != 9324544
        or tuple(record.get("source_name") for record in records or ()) != OFFICIAL_NAMES
    ):
        raise ValueError("Formal train-v3 cache population differs.")
    actual_h2 = tuple(
        record["source_name"]
        for record in records
        if record.get("decision", {}).get("domain") == "h2"
    )
    if actual_h2 != H2_NAMES:
        raise ValueError("Formal train-v3 H2 population differs.")
    code_binding = manifest.get("code", {}).get("sha256", {})
    for relative in (
        "run_temporal_memory_input_route_train.py",
        "utils/temporal_memory_input_router.py",
        "utils/postprocess.py",
        "utils/eval.py",
        "utils/challenge_eval.py",
    ):
        if base.sha256_file(PROJECT_ROOT / relative) != code_binding.get(relative):
            raise ValueError("Metric/cache dependency differs: {}".format(relative))

    import torch

    from dataset.temporal_frame import load_temporal_frame_video
    from run_temporal_memory_input_route_train import evaluate_one as project_evaluate_one
    from utils.temporal_memory_input_router import select_temporal_memory_input_route

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized; this experiment is CPU-only.")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for prediction-only components.") from error

    paths = tuple(sorted(train_root.glob("train_*.npz")))
    if tuple(path.name for path in paths) != OFFICIAL_NAMES:
        raise ValueError("Train root is not the exact official 99-source population.")
    records_by_name = {record["source_name"]: record for record in records}
    cache_root = manifest_path.parent
    catalog = candidate_catalog()
    candidate_ids = tuple(item["candidate_id"] for item in catalog)
    candidate_by_group = {
        candidate_id: {group: base.empty_counts() for group in GROUPS}
        for candidate_id in candidate_ids
    }
    baseline_by_group = {group: base.empty_counts() for group in GROUPS}
    non_h2_baseline = base.empty_counts()
    full99_baseline = base.empty_counts()
    per_source = []

    def group_for_name(name):
        return next((group for group, names in GROUPS.items() if name in names), None)

    for index, path in enumerate(paths, start=1):
        record = records_by_name[path.name]
        expected_source_sha = record.get("source_sha256")
        if base.sha256_file(path) != expected_source_sha:
            raise ValueError("Train source SHA-256 differs: {}".format(path.name))
        record_path = (cache_root / record["record"]).resolve()
        try:
            record_path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError("Cache record escapes cache root.") from error
        if base.sha256_file(record_path) != record.get("record_sha256"):
            raise ValueError("Cache record SHA-256 differs: {}".format(path.name))
        with np.load(record_path, allow_pickle=False) as payload:
            if set(payload.files) != {"baseline_scores", "candidate_scores"}:
                raise ValueError("Unexpected cache arrays for {}".format(path.name))
            full_scores = np.asarray(payload["baseline_scores"], dtype=np.float32).reshape(-1)
            t32_scores = np.asarray(payload["candidate_scores"], dtype=np.float32).reshape(-1)
        video = load_temporal_frame_video(path, TEMPORAL_FRAME_SIZE, 8000)
        if full_scores.size != video.locations.shape[0] or t32_scores.size != full_scores.size:
            raise ValueError("Score/source length differs: {}".format(path.name))
        decision = select_temporal_memory_input_route(
            video.polarities, len(video.event_indices_by_bin)
        )
        if decision.to_metadata() != record.get("decision"):
            raise ValueError("Runtime/cache route decision differs: {}".format(path.name))
        baseline_counts, baseline_postprocess = project_evaluate_one(
            video, full_scores, THRESHOLD
        )
        baseline_counts = {key: int(baseline_counts[key]) for key in COUNT_KEYS}
        full99_baseline = base.add_counts(full99_baseline, baseline_counts)
        group = group_for_name(path.name)
        source_report = {
            "source_name": path.name,
            "source_sha256": expected_source_sha,
            "record_sha256": record["record_sha256"],
            "domain": decision.domain,
            "baseline": base.evaluation(baseline_counts),
            "baseline_postprocess": baseline_postprocess,
        }
        if group is None:
            if not np.array_equal(full_scores, t32_scores):
                raise RuntimeError(
                    "Non-H2 cache is not bitwise identical: {}".format(path.name)
                )
            non_h2_baseline = base.add_counts(non_h2_baseline, baseline_counts)
        else:
            if decision.domain != "h2":
                raise RuntimeError("Frozen group member is not H2: {}".format(path.name))
            baseline_by_group[group] = base.add_counts(
                baseline_by_group[group], baseline_counts
            )
            source_report["group"] = group
            source_report["candidates"] = {}
            for alpha in ALPHAS:
                suffix = "{:03d}".format(int(round(alpha * 100)))
                increment_scores, increment_stats = base.component_increment_candidates(
                    video.locations,
                    full_scores,
                    t32_scores,
                    alpha,
                    event_indices_by_frame=video.event_indices_by_bin,
                    cv2_module=cv2,
                )
                raw_increment = increment_scores[ANCHOR_MIN]
                for max_changed_events in MAX_CHANGED_EVENTS:
                    candidate_id = "budget_a{}_m{:03d}".format(
                        suffix, max_changed_events
                    )
                    candidate_scores, budget_stats = apply_changed_event_budget(
                        full_scores, raw_increment, max_changed_events
                    )
                    counts, postprocess = project_evaluate_one(
                        video, candidate_scores, THRESHOLD
                    )
                    counts = {key: int(counts[key]) for key in COUNT_KEYS}
                    candidate_by_group[candidate_id][group] = base.add_counts(
                        candidate_by_group[candidate_id][group], counts
                    )
                    source_report["candidates"][candidate_id] = {
                        "evaluation": base.evaluation(counts),
                        "postprocess": postprocess,
                        "prediction_only_component_stats": increment_stats[ANCHOR_MIN],
                        "prediction_only_budget_stats": budget_stats,
                    }
        per_source.append(source_report)
        print("[{}/99] {} ({})".format(index, path.name, decision.domain), flush=True)

    candidate_group_summary = {}
    for candidate_id in candidate_ids:
        candidate_group_summary[candidate_id] = {}
        for group in GROUPS:
            baseline_eval = base.evaluation(baseline_by_group[group])
            candidate_eval = base.evaluation(candidate_by_group[candidate_id][group])
            candidate_group_summary[candidate_id][group] = {
                "baseline": baseline_eval,
                "candidate": candidate_eval,
                "delta": base.evaluation_delta(baseline_eval, candidate_eval),
                "gates": base.comparison_gates(
                    baseline_eval, candidate_eval, strict_score=False
                ),
            }

    folds = []
    selected_h2_counts = base.empty_counts()
    h2_baseline_counts = base.add_counts(
        *(baseline_by_group[group] for group in GROUPS)
    )
    for held_group in GROUPS:
        development_groups = tuple(group for group in GROUPS if group != held_group)
        selection = select_candidate(
            baseline_by_group, candidate_by_group, development_groups
        )
        selected_id = selection["selected_candidate_id"]
        held_baseline_counts = baseline_by_group[held_group]
        held_candidate_counts = (
            held_baseline_counts
            if selected_id == "full_abstain"
            else candidate_by_group[selected_id][held_group]
        )
        selected_h2_counts = base.add_counts(selected_h2_counts, held_candidate_counts)
        held_baseline = base.evaluation(held_baseline_counts)
        held_candidate = base.evaluation(held_candidate_counts)
        gates = base.comparison_gates(
            held_baseline, held_candidate, strict_score=False
        )
        folds.append(
            {
                "held_group": held_group,
                "held_sources": list(GROUPS[held_group]),
                "development_groups": list(development_groups),
                "selected_candidate_id": selected_id,
                "selection": selection,
                "baseline": held_baseline,
                "candidate": held_candidate,
                "delta": base.evaluation_delta(held_baseline, held_candidate),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )

    h2_baseline = base.evaluation(h2_baseline_counts)
    h2_candidate = base.evaluation(selected_h2_counts)
    h2_gates = base.comparison_gates(h2_baseline, h2_candidate, strict_score=True)
    full99_candidate_counts = base.add_counts(non_h2_baseline, selected_h2_counts)
    full99_baseline_eval = base.evaluation(full99_baseline)
    full99_candidate_eval = base.evaluation(full99_candidate_counts)
    full99_gates = base.comparison_gates(
        full99_baseline_eval, full99_candidate_eval, strict_score=True
    )
    final_selection = select_candidate(
        baseline_by_group, candidate_by_group, tuple(GROUPS)
    )
    all_passed = (
        all(fold["passed"] for fold in folds)
        and all(h2_gates.values())
        and all(full99_gates.values())
    )

    code_paths = (
        "evaluate_h2_safe_fusion_budget_train_oof.py",
        "evaluate_h2_safe_fusion_train_oof.py",
        "protocols/h2_safe_fusion_train_oof_v2.json",
        "protocols/h2_safe_fusion_train_oof_v1.json",
        "run_temporal_memory_input_route_train.py",
        "dataset/temporal_frame.py",
        "utils/challenge_eval.py",
        "utils/component_reranker.py",
        "utils/eval.py",
        "utils/postprocess.py",
        "utils/temporal_memory_input_router.py",
    )
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": base.utc_now(),
        "passed": all_passed,
        "evidence_class": "adaptive_train_only_confirmation_after_v1_not_independent_oof",
        "interpretation_limit": protocol["interpretation_limit"],
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "input_integrity": {
            "train_only": True,
            "validation_or_test_read": False,
            "gpu_used": False,
            "formal_train_cache_manifest": {
                "path": str(manifest_path),
                "sha256": EXPECTED_MANIFEST_SHA256,
            },
            "formal_train_protocol": {
                "path": str(train_protocol_path),
                "sha256": EXPECTED_TRAIN_PROTOCOL_SHA256,
            },
            "v1_protocol": {
                "path": str(v1_protocol_path),
                "sha256": EXPECTED_V1_PROTOCOL_SHA256,
            },
            "v1_train_only_report": {
                "path": str(v1_report_path),
                "sha256": EXPECTED_V1_REPORT_SHA256,
            },
            "train_source_count": 99,
            "h2_source_count": 11,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": "cpu",
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "elapsed_seconds": time.monotonic() - started,
        },
        "code_sha256": {
            relative: base.sha256_file(PROJECT_ROOT / relative) for relative in code_paths
        },
        "candidate_catalog": list(catalog),
        "candidate_by_group": candidate_group_summary,
        "folds": folds,
        "pooled_oof_h2": {
            "baseline": h2_baseline,
            "candidate": h2_candidate,
            "delta": base.evaluation_delta(h2_baseline, h2_candidate),
            "gates": h2_gates,
            "passed": all(h2_gates.values()),
        },
        "pooled_oof_full99_route": {
            "baseline": full99_baseline_eval,
            "candidate": full99_candidate_eval,
            "delta": base.evaluation_delta(full99_baseline_eval, full99_candidate_eval),
            "gates": full99_gates,
            "passed": all(full99_gates.values()),
        },
        "final_all_train_selection": final_selection,
        "per_source": per_source,
        "promotion_gates": {
            "each_held_fold_passed": all(fold["passed"] for fold in folds),
            "pooled_oof_h2_passed": all(h2_gates.values()),
            "pooled_oof_full99_route_passed": all(full99_gates.values()),
            "all_required": all_passed,
        },
    }
    output_directory = Path(output_directory).resolve()
    report_path = output_directory / protocol["outputs"]["report"]
    report_sha = base._atomic_json_no_clobber(report_path, report)
    sidecar_path = output_directory / protocol["outputs"]["report_sha256_sidecar"]
    base._atomic_json_no_clobber(
        sidecar_path,
        {"path": str(report_path), "sha256": report_sha, "schema": REPORT_SCHEMA},
    )
    return report_path, report_sha, report


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default=str(WORKSPACE_ROOT / "experiments" / "20260810_h2_safe_fusion_train_oof_v2"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    report_path, digest, report = run(args.output_directory)
    print("report:", report_path)
    print("sha256:", digest)
    print("passed:", report["passed"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

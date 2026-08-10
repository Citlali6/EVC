"""CPU-only grouped-OOF evaluation of preregistered H2 full/T32 fusion.

Only the official 99-source train population and the frozen train-v3 score
cache are accepted.  Candidate generation is label-free; labels and target ids
enter only the fixed C00 metric evaluation after the protocol is validated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = (PROJECT_ROOT / "protocols" / "h2_safe_fusion_train_oof_v1.json").resolve()
EXPECTED_PROTOCOL_SHA256 = "d05b3c0c3a146f7d76c6d18ddfa8095df4840ee0f9e17930dd921a76d1559f85"
EXPECTED_MANIFEST_SHA256 = "78ca63efd1fd8fda62dcccb1203f0e69000007454a391b7d46455f9952cf2dc7"
EXPECTED_TRAIN_PROTOCOL_SHA256 = "ddd027961bc36f2756a62cd62914c5be3400a2ddd965d53ab2ff066b331f36d1"
EXPECTED_C00_SHA256 = "a79c05cf80d0315a8110a3f88ca2987856d855ba4dde308371b12c2dcd4f32b8"
REPORT_SCHEMA = "ev-uav-h2-safe-fusion-train-oof-report-v1"

OFFICIAL_NAMES = tuple("train_{:03d}.npz".format(index) for index in range(99))
GROUPS = {
    "g1": tuple("train_{:03d}.npz".format(index) for index in range(88, 92)),
    "g2": tuple("train_{:03d}.npz".format(index) for index in range(92, 95)),
    "g3": tuple("train_{:03d}.npz".format(index) for index in range(95, 99)),
}
H2_NAMES = tuple(name for group in GROUPS.values() for name in group)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
ANCHOR_MINIMUMS = (1, 2, 4)
THRESHOLD = 0.719
TEMPORAL_FRAME_SIZE = 50
WIDTH = 346
HEIGHT = 260
COUNT_KEYS = (
    "true_positive_events",
    "false_positive_events",
    "false_negative_events",
    "true_negative_events",
    "correct_target_groups",
    "target_groups",
    "false_components",
    "frame_count",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_catalog():
    candidates = []
    for alpha in ALPHAS:
        suffix = "{:03d}".format(int(round(alpha * 100)))
        candidates.append(
            {
                "candidate_id": "convex_a{}".format(suffix),
                "family": "convex",
                "alpha": alpha,
                "anchor_min": 0,
            }
        )
    for alpha in ALPHAS:
        suffix = "{:03d}".format(int(round(alpha * 100)))
        for anchor_min in ANCHOR_MINIMUMS:
            candidates.append(
                {
                    "candidate_id": "inc_a{}_k{}".format(suffix, anchor_min),
                    "family": "component_increment_abstain",
                    "alpha": alpha,
                    "anchor_min": anchor_min,
                }
            )
    return tuple(candidates)


def convex_blend(full_scores, t32_scores, alpha):
    full = np.asarray(full_scores, dtype=np.float32).reshape(-1)
    t32 = np.asarray(t32_scores, dtype=np.float32).reshape(-1)
    if full.shape != t32.shape:
        raise ValueError("Full and T32 score vectors must be aligned.")
    alpha = float(alpha)
    if alpha not in ALPHAS:
        raise ValueError("alpha is outside the frozen grid.")
    result = np.asarray(
        np.float32(1.0 - alpha) * full + np.float32(alpha) * t32,
        dtype=np.float32,
    )
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("Convex blend produced non-probability scores.")
    return result


def component_increment_candidates(
    locations,
    full_scores,
    t32_scores,
    alpha,
    event_indices_by_frame=None,
    cv2_module=None,
):
    """Return all preregistered anchor thresholds without consuming labels."""
    if cv2_module is None:
        import cv2 as cv2_module

    locations = np.asarray(locations)
    full = np.asarray(full_scores, dtype=np.float32).reshape(-1)
    if locations.ndim != 2 or locations.shape[1] < 3 or locations.shape[0] != full.size:
        raise ValueError("locations must be aligned [N,3+] x/y/t values.")
    if np.any(locations[:, 0] < 0) or np.any(locations[:, 0] >= WIDTH):
        raise ValueError("x coordinate is outside the frozen resolution.")
    if np.any(locations[:, 1] < 0) or np.any(locations[:, 1] >= HEIGHT):
        raise ValueError("y coordinate is outside the frozen resolution.")

    blend = convex_blend(full, t32_scores, alpha)
    proposal = np.maximum(full, blend).astype(np.float32, copy=False)
    outputs = {anchor: full.copy() for anchor in ANCHOR_MINIMUMS}
    stats = {
        anchor: {
            "proposal_components": 0,
            "incremental_components": 0,
            "accepted_incremental_components": 0,
            "abstained_incremental_components": 0,
            "accepted_component_events": 0,
            "changed_events": 0,
        }
        for anchor in ANCHOR_MINIMUMS
    }

    if event_indices_by_frame is None:
        frame_ids = np.floor_divide(
            locations[:, 2].astype(np.int64, copy=False), TEMPORAL_FRAME_SIZE
        )
        event_indices_by_frame = tuple(
            np.flatnonzero(frame_ids == frame).astype(np.int64, copy=False)
            for frame in range(int(frame_ids.max()) + 1)
        )

    for event_indices in event_indices_by_frame:
        event_indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
        if event_indices.size == 0:
            continue
        proposal_positive = proposal[event_indices] >= THRESHOLD
        if not np.any(proposal_positive):
            continue
        positive_indices = event_indices[proposal_positive]
        positive_locations = locations[positive_indices]
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        mask[
            positive_locations[:, 1].astype(np.int64, copy=False),
            positive_locations[:, 0].astype(np.int64, copy=False),
        ] = 1
        component_count, component_labels, _, _ = cv2_module.connectedComponentsWithStats(
            mask,
            connectivity=8,
            ltype=cv2_module.CV_32S,
        )
        full_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        full_positive_indices = event_indices[full[event_indices] >= THRESHOLD]
        if full_positive_indices.size:
            full_locations = locations[full_positive_indices]
            full_mask[
                full_locations[:, 1].astype(np.int64, copy=False),
                full_locations[:, 0].astype(np.int64, copy=False),
            ] = 1
        local_locations = locations[event_indices]
        event_component = component_labels[
            local_locations[:, 1].astype(np.int64, copy=False),
            local_locations[:, 0].astype(np.int64, copy=False),
        ]

        for component_id in range(1, int(component_count)):
            component_pixels = component_labels == component_id
            incremental = bool(np.any(component_pixels & (full_mask == 0)))
            anchors = int(np.count_nonzero(component_pixels & (full_mask > 0)))
            component_events = event_indices[event_component == component_id]
            for anchor_min in ANCHOR_MINIMUMS:
                stats[anchor_min]["proposal_components"] += 1
                if not incremental:
                    continue
                stats[anchor_min]["incremental_components"] += 1
                if anchors >= anchor_min:
                    outputs[anchor_min][component_events] = proposal[component_events]
                    stats[anchor_min]["accepted_incremental_components"] += 1
                    stats[anchor_min]["accepted_component_events"] += int(
                        component_events.size
                    )
                else:
                    stats[anchor_min]["abstained_incremental_components"] += 1

    for anchor_min in ANCHOR_MINIMUMS:
        output = outputs[anchor_min]
        if np.any(output < full):
            raise RuntimeError("Increment-only candidate lowered a full score.")
        stats[anchor_min]["changed_events"] = int(np.count_nonzero(output != full))
    return outputs, stats


def empty_counts():
    return {key: 0 for key in COUNT_KEYS}


def add_counts(*values):
    total = empty_counts()
    for value in values:
        for key in COUNT_KEYS:
            total[key] += int(value[key])
    return total


def evaluation(counts):
    counts = {key: int(counts[key]) for key in COUNT_KEYS}
    tp = counts["true_positive_events"]
    fp = counts["false_positive_events"]
    fn = counts["false_negative_events"]
    union = tp + fp + fn
    positives = tp + fn
    targets = counts["target_groups"]
    frames = counts["frame_count"]
    if min(union, positives, targets, frames) <= 0:
        raise ValueError("A metric denominator is zero.")
    iou = float(np.float32(tp) / np.float32(union))
    acc = float(np.float32(tp) / np.float32(positives))
    pd = counts["correct_target_groups"] / targets
    fa = counts["false_components"] / (frames * WIDTH * HEIGHT)
    score_fa = math.exp(-10000.0 * fa)
    score = 0.4 * pd + 0.3 * score_fa + 0.2 * iou + 0.1 * acc
    metrics = {
        "iou": iou,
        "acc": acc,
        "pd": pd,
        "fa": fa,
        "score_fa": score_fa,
        "score": score,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite metric produced.")
    return {"metrics": metrics, "counts": counts}


def evaluation_delta(baseline, candidate):
    return {
        "metrics": {
            key: float(candidate["metrics"][key]) - float(baseline["metrics"][key])
            for key in baseline["metrics"]
        },
        "counts": {
            key: int(candidate["counts"][key]) - int(baseline["counts"][key])
            for key in COUNT_KEYS
        },
    }


def comparison_gates(baseline, candidate, strict_score):
    return {
        "score": (
            candidate["metrics"]["score"] > baseline["metrics"]["score"]
            if strict_score
            else candidate["metrics"]["score"] >= baseline["metrics"]["score"]
        ),
        "pd_not_lower": candidate["metrics"]["pd"] >= baseline["metrics"]["pd"],
        "iou_not_lower": candidate["metrics"]["iou"] >= baseline["metrics"]["iou"],
        "fa_not_higher": candidate["metrics"]["fa"] <= baseline["metrics"]["fa"],
    }


def select_candidate(group_baseline_counts, group_candidate_counts, development_groups):
    """Select using only the explicitly supplied development groups."""
    development_groups = tuple(development_groups)
    if not development_groups or any(group not in GROUPS for group in development_groups):
        raise ValueError("development_groups must be a non-empty subset of frozen groups.")
    baseline_by_group = {
        group: evaluation(group_baseline_counts[group]) for group in development_groups
    }
    pooled_baseline = evaluation(
        add_counts(*(group_baseline_counts[group] for group in development_groups))
    )
    catalog = {item["candidate_id"]: item for item in candidate_catalog()}
    rows = []
    for candidate_id, candidate in catalog.items():
        candidate_by_group = {
            group: evaluation(group_candidate_counts[candidate_id][group])
            for group in development_groups
        }
        pooled_candidate = evaluation(
            add_counts(
                *(group_candidate_counts[candidate_id][group] for group in development_groups)
            )
        )
        group_deltas = {
            group: evaluation_delta(baseline_by_group[group], candidate_by_group[group])
            for group in development_groups
        }
        pooled_delta = evaluation_delta(pooled_baseline, pooled_candidate)
        group_gates = {
            group: comparison_gates(
                baseline_by_group[group], candidate_by_group[group], strict_score=True
            )
            for group in development_groups
        }
        pooled_gates = comparison_gates(
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

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
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
            -row["anchor_min"],
            row["candidate_id"],
        )

    chosen = sorted(eligible, key=rank)[0]
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
        },
        "candidate_rows": rows,
    }


def _load_json_snapshot(path, expected_sha, name):
    path = Path(path).resolve()
    before = sha256_file(path)
    if before != expected_sha:
        raise ValueError("{} SHA-256 differs.".format(name))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while being read.".format(name))
    return payload


def validate_protocol():
    protocol = _load_json_snapshot(PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "protocol")
    if (
        protocol.get("schema") != "ev-uav-h2-safe-fusion-train-oof-protocol-v1"
        or protocol.get("status") != "frozen_before_h2_fusion_score_or_label_access"
        or protocol.get("split_access", {}).get("gpu_allowed") is not False
    ):
        raise ValueError("Frozen train-only protocol identity differs.")
    if protocol["population"]["grouped_folds"] != {
        group: list(names) for group, names in GROUPS.items()
    }:
        raise ValueError("Grouped folds differ from the frozen 4/3/4 split.")
    candidate_ids = [item["candidate_id"] for item in candidate_catalog()]
    frozen_ids = (
        protocol["candidate_generation"]["families"]["convex"]["candidate_ids"]
        + protocol["candidate_generation"]["families"]["component_increment_abstain"][
            "candidate_ids"
        ]
    )
    if candidate_ids != frozen_ids:
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
    return protocol


def _atomic_json_no_clobber(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Refusing to overwrite report: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def run(output_directory):
    started = time.monotonic()
    protocol = validate_protocol()
    inputs = protocol["inputs"]
    manifest_path = Path(inputs["formal_train_v3_cache_manifest"]["path"]).resolve()
    train_protocol_path = Path(inputs["formal_train_v3_protocol"]["path"]).resolve()
    train_root = Path(inputs["train_root"]).resolve()
    manifest = _load_json_snapshot(manifest_path, EXPECTED_MANIFEST_SHA256, "cache manifest")
    if sha256_file(train_protocol_path) != EXPECTED_TRAIN_PROTOCOL_SHA256:
        raise ValueError("Formal train-v3 protocol SHA-256 differs.")
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
        if sha256_file(PROJECT_ROOT / relative) != code_binding.get(relative):
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
        candidate_id: {group: empty_counts() for group in GROUPS}
        for candidate_id in candidate_ids
    }
    baseline_by_group = {group: empty_counts() for group in GROUPS}
    non_h2_baseline = empty_counts()
    full99_baseline = empty_counts()
    per_source = []

    def group_for_name(name):
        return next((group for group, names in GROUPS.items() if name in names), None)

    for index, path in enumerate(paths, start=1):
        record = records_by_name[path.name]
        expected_source_sha = record.get("source_sha256")
        if sha256_file(path) != expected_source_sha:
            raise ValueError("Train source SHA-256 differs: {}".format(path.name))
        record_path = (cache_root / record["record"]).resolve()
        try:
            record_path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError("Cache record escapes cache root.") from error
        if sha256_file(record_path) != record.get("record_sha256"):
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
        full99_baseline = add_counts(full99_baseline, baseline_counts)
        group = group_for_name(path.name)
        source_report = {
            "source_name": path.name,
            "source_sha256": expected_source_sha,
            "record_sha256": record["record_sha256"],
            "domain": decision.domain,
            "baseline": evaluation(baseline_counts),
            "baseline_postprocess": baseline_postprocess,
        }
        if group is None:
            if not np.array_equal(full_scores, t32_scores):
                raise RuntimeError("Non-H2 cache is not bitwise identical: {}".format(path.name))
            non_h2_baseline = add_counts(non_h2_baseline, baseline_counts)
        else:
            if decision.domain != "h2":
                raise RuntimeError("Frozen group member is not H2: {}".format(path.name))
            baseline_by_group[group] = add_counts(
                baseline_by_group[group], baseline_counts
            )
            source_report["group"] = group
            source_report["candidates"] = {}
            for alpha in ALPHAS:
                suffix = "{:03d}".format(int(round(alpha * 100)))
                convex_id = "convex_a{}".format(suffix)
                convex_scores = convex_blend(full_scores, t32_scores, alpha)
                convex_counts, convex_postprocess = project_evaluate_one(
                    video, convex_scores, THRESHOLD
                )
                convex_counts = {key: int(convex_counts[key]) for key in COUNT_KEYS}
                candidate_by_group[convex_id][group] = add_counts(
                    candidate_by_group[convex_id][group], convex_counts
                )
                source_report["candidates"][convex_id] = {
                    "evaluation": evaluation(convex_counts),
                    "postprocess": convex_postprocess,
                    "prediction_only_component_stats": None,
                }
                increment_scores, increment_stats = component_increment_candidates(
                    video.locations,
                    full_scores,
                    t32_scores,
                    alpha,
                    event_indices_by_frame=video.event_indices_by_bin,
                    cv2_module=cv2,
                )
                for anchor_min in ANCHOR_MINIMUMS:
                    candidate_id = "inc_a{}_k{}".format(suffix, anchor_min)
                    counts, postprocess = project_evaluate_one(
                        video, increment_scores[anchor_min], THRESHOLD
                    )
                    counts = {key: int(counts[key]) for key in COUNT_KEYS}
                    candidate_by_group[candidate_id][group] = add_counts(
                        candidate_by_group[candidate_id][group], counts
                    )
                    source_report["candidates"][candidate_id] = {
                        "evaluation": evaluation(counts),
                        "postprocess": postprocess,
                        "prediction_only_component_stats": increment_stats[anchor_min],
                    }
        per_source.append(source_report)
        print(
            "[{}/99] {} ({})".format(index, path.name, decision.domain),
            flush=True,
        )

    candidate_group_summary = {}
    for candidate_id in candidate_ids:
        candidate_group_summary[candidate_id] = {}
        for group in GROUPS:
            baseline_eval = evaluation(baseline_by_group[group])
            candidate_eval = evaluation(candidate_by_group[candidate_id][group])
            candidate_group_summary[candidate_id][group] = {
                "baseline": baseline_eval,
                "candidate": candidate_eval,
                "delta": evaluation_delta(baseline_eval, candidate_eval),
                "gates": comparison_gates(
                    baseline_eval, candidate_eval, strict_score=False
                ),
            }

    folds = []
    selected_h2_counts = empty_counts()
    h2_baseline_counts = add_counts(*(baseline_by_group[group] for group in GROUPS))
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
        selected_h2_counts = add_counts(selected_h2_counts, held_candidate_counts)
        held_baseline = evaluation(held_baseline_counts)
        held_candidate = evaluation(held_candidate_counts)
        gates = comparison_gates(held_baseline, held_candidate, strict_score=False)
        folds.append(
            {
                "held_group": held_group,
                "held_sources": list(GROUPS[held_group]),
                "development_groups": list(development_groups),
                "selected_candidate_id": selected_id,
                "selection": selection,
                "baseline": held_baseline,
                "candidate": held_candidate,
                "delta": evaluation_delta(held_baseline, held_candidate),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )

    h2_baseline = evaluation(h2_baseline_counts)
    h2_candidate = evaluation(selected_h2_counts)
    h2_gates = comparison_gates(h2_baseline, h2_candidate, strict_score=True)
    full99_candidate_counts = add_counts(non_h2_baseline, selected_h2_counts)
    full99_baseline_eval = evaluation(full99_baseline)
    full99_candidate_eval = evaluation(full99_candidate_counts)
    full99_gates = comparison_gates(
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
        "evaluate_h2_safe_fusion_train_oof.py",
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
        "created_utc": utc_now(),
        "passed": all_passed,
        "evidence_class": "train_only_three_group_out_of_fold_candidate_selection",
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
            "train_source_count": 99,
            "h2_source_count": 11,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": "cpu",
            "elapsed_seconds": time.monotonic() - started,
        },
        "code_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative) for relative in code_paths
        },
        "candidate_catalog": list(catalog),
        "candidate_by_group": candidate_group_summary,
        "folds": folds,
        "pooled_oof_h2": {
            "baseline": h2_baseline,
            "candidate": h2_candidate,
            "delta": evaluation_delta(h2_baseline, h2_candidate),
            "gates": h2_gates,
            "passed": all(h2_gates.values()),
        },
        "pooled_oof_full99_route": {
            "baseline": full99_baseline_eval,
            "candidate": full99_candidate_eval,
            "delta": evaluation_delta(full99_baseline_eval, full99_candidate_eval),
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
    report_sha = _atomic_json_no_clobber(report_path, report)
    sidecar_path = output_directory / protocol["outputs"]["report_sha256_sidecar"]
    sidecar_payload = {
        "path": str(report_path),
        "sha256": report_sha,
        "schema": REPORT_SCHEMA,
    }
    _atomic_json_no_clobber(sidecar_path, sidecar_payload)
    return report_path, report_sha, report


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default="F:/小目标检测/experiments/20260810_h2_safe_fusion_train_oof_v1",
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

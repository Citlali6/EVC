"""CPU-only train-only capacity audit for the M10 low-density route.

The released route sends videos with at most 30,000 input events through M10
at a 0.718 decision threshold.  This audit deliberately stops before fitting a
new model.  It replays that frozen M10+C00 route on all 45 low-density train
sources and measures how much of the remaining error is addressable by
complete post-C00 component removal.

Labels and target IDs are read only after the label-free C00 component
partition has been built.  No validation/test paths, model inference, CUDA, or
source identity features are used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

from dataset.temporal_frame import load_temporal_frame_video
import run_temporal_memory_input_route_train as routed
from utils.atomic_component_deletion import (
    extract_atomic_components,
    pure_false_positive_targets,
)
from utils.challenge_eval import challenge_score
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor, P0ClusterFilter, P18ScoreTrackRecovery


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXPERIMENT_ROOT = WORKSPACE / "experiments" / "20260812_low_m10_component_oracle_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "train_only_capacity_report.json"
CACHE_DIR = (
    WORKSPACE
    / "experiments"
    / "20260810_temporal_input_route_v1"
    / "formal_train_score_cache_v3"
)
MANIFEST_PATH = CACHE_DIR / "manifest.json"
PROTOCOL_PATH = (
    WORKSPACE
    / "experiments"
    / "20260810_temporal_input_route_v1"
    / "frozen_train_cache_eval_protocol_v3.json"
)
TRAIN_ROOT = WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train"

EXPECTED_MANIFEST_SHA256 = "78ca63efd1fd8fda62dcccb1203f0e69000007454a391b7d46455f9952cf2dc7"
EXPECTED_PROTOCOL_SHA256 = "ddd027961bc36f2756a62cd62914c5be3400a2ddd965d53ab2ff066b331f36d1"
LOW_THRESHOLD = np.float32(0.718)
WIDTH = 346
HEIGHT = 260
TEMPORAL_BIN = 50
FULL_TARGET_GAP = 0.9700 - 0.9628776541559201

LOW_FAMILIES = {
    "low_f1_015_027": tuple(f"train_{index:03d}.npz" for index in range(15, 28)),
    "low_f2_033_039": tuple(f"train_{index:03d}.npz" for index in range(33, 40)),
    "low_f3_048_058": tuple(f"train_{index:03d}.npz" for index in range(48, 59)),
    "low_f4_066": ("train_066.npz",),
    "low_f5_075_087": tuple(f"train_{index:03d}.npz" for index in range(75, 88)),
}
LOW_NAMES = tuple(name for values in LOW_FAMILIES.values() for name in values)


@dataclass(frozen=True)
class Counts:
    true_positive_events: int = 0
    false_positive_events: int = 0
    false_negative_events: int = 0
    correct_target_frames: int = 0
    target_frames: int = 0
    false_components: int = 0
    frame_count: int = 0
    event_count: int = 0

    def __add__(self, other):
        if not isinstance(other, Counts):
            return NotImplemented
        return Counts(
            **{
                field: int(getattr(self, field) + getattr(other, field))
                for field in self.__dataclass_fields__
            }
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class PreparedLowVideo:
    source_name: str
    family: str
    source_sha256: str
    cache_sha256: str
    event_count: int
    raw_scores: np.ndarray
    p0_scores: np.ndarray
    final_scores: np.ndarray
    locations4: np.ndarray
    labels: np.ndarray
    target_ids: np.ndarray
    components: tuple[np.ndarray, ...]
    pure_fp: np.ndarray
    baseline: Counts


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_exclusive(path: Path, payload) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        "{}  {}\n".format(digest, path.name), encoding="ascii"
    )
    return digest


def family_for_name(source_name: str) -> str:
    matches = [name for name, values in LOW_FAMILIES.items() if source_name in values]
    if len(matches) != 1:
        raise ValueError("source does not belong to exactly one low family: {}".format(source_name))
    return matches[0]


def metrics_from_counts(counts: Counts) -> dict:
    positives = counts.true_positive_events + counts.false_negative_events
    union = positives + counts.false_positive_events
    denominator = counts.frame_count * WIDTH * HEIGHT
    if min(positives, union, counts.target_frames, denominator) <= 0:
        raise ValueError("invalid sufficient counts")
    iou = float(
        (torch.tensor(counts.true_positive_events, dtype=torch.float32)
         / torch.tensor(union, dtype=torch.float32)).item()
    )
    acc = float(
        (torch.tensor(counts.true_positive_events, dtype=torch.float32)
         / torch.tensor(positives, dtype=torch.float32)).item()
    )
    pd = counts.correct_target_frames / counts.target_frames
    fa = counts.false_components / denominator
    score_fa, score = challenge_score(iou, acc, pd, fa)
    return {
        "iou": iou,
        "acc": acc,
        "pd": pd,
        "fa": fa,
        "score_fa": score_fa,
        "score": score,
    }


def record(counts: Counts) -> dict:
    return {"counts": counts.to_dict(), "metrics": metrics_from_counts(counts)}


def count_delta(candidate: Counts, baseline: Counts) -> dict:
    return {
        field: int(getattr(candidate, field) - getattr(baseline, field))
        for field in baseline.__dataclass_fields__
    }


def metric_delta(candidate: Counts, baseline: Counts) -> dict:
    candidate_metrics = metrics_from_counts(candidate)
    baseline_metrics = metrics_from_counts(baseline)
    return {key: float(candidate_metrics[key] - baseline_metrics[key]) for key in baseline_metrics}


def official_counts(scores, labels, target_ids, locations4) -> Counts:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    ids = np.asarray(target_ids).reshape(-1)
    locations = np.asarray(locations4, dtype=np.int64)
    if not (values.size == truth.size == ids.size == locations.shape[0]):
        raise ValueError("official-count inputs differ in length")
    evaluator = evalute(type("Config", (), {"roc": True, "pd_detT": 50, "correct_thresh": 0.0001})())
    evaluator.roc_update(
        torch.from_numpy(locations[:, 3].copy()),
        torch.from_numpy(values.copy()),
        ids,
        torch.from_numpy(truth.astype(np.float32, copy=False)),
        torch.from_numpy(locations.copy()),
        thresh=float(LOW_THRESHOLD),
    )
    predicted = values >= LOW_THRESHOLD
    positive = truth > 0
    return Counts(
        true_positive_events=int(np.count_nonzero(predicted & positive)),
        false_positive_events=int(np.count_nonzero(predicted & ~positive)),
        false_negative_events=int(np.count_nonzero(~predicted & positive)),
        correct_target_frames=int(evaluator.correct_num),
        target_frames=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
        event_count=int(values.size),
    )


def sum_counts(values) -> Counts:
    total = Counts()
    for value in values:
        total = total + value
    return total


def validate_inputs():
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("low-route cache manifest changed")
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen low-route protocol changed")
    manifest = read_json(MANIFEST_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if manifest.get("schema") != "ev-uav-temporal-input-route-train-cache-v1":
        raise RuntimeError("unexpected route cache schema")
    if manifest.get("split_access", {}).get("validation_or_test_read") is not False:
        raise RuntimeError("cache is not train-only")
    if protocol.get("split_access", {}).get("validation_or_test_permitted") is not False:
        raise RuntimeError("protocol permits validation/test")
    records = [item for item in manifest.get("records", ()) if item.get("decision", {}).get("domain") == "low"]
    if tuple(item.get("source_name") for item in records) != LOW_NAMES:
        raise RuntimeError("low source population differs from frozen continuous families")
    if len(records) != 45 or sum(int(item["event_count"]) for item in records) != 768782:
        raise RuntimeError("low population count/event total differs")
    for item in records:
        decision = item["decision"]
        if not (
            decision.get("checkpoint_role") == "m10"
            and decision.get("mode") == "full_stream"
            and int(decision.get("temporal_bin_count", -1)) == 160
            and np.float32(decision.get("prediction_threshold")) == LOW_THRESHOLD
            and int(item["event_count"]) <= 30000
            and item.get("bitwise_equal_to_baseline") is True
        ):
            raise RuntimeError("low M10 route identity differs for {}".format(item["source_name"]))
    return manifest, records


def p0_and_p18(raw_scores: np.ndarray, locations4: np.ndarray, event_count: int):
    cfg = routed.c00_config()
    raw_tensor = torch.from_numpy(np.asarray(raw_scores, dtype=np.float32).copy())
    location_tensor = torch.from_numpy(np.asarray(locations4, dtype=np.int64).copy())
    full_processor = ChallengePostprocessor.from_cfg(
        cfg, float(LOW_THRESHOLD), event_count=int(event_count)
    )
    final_scores, stats = full_processor.apply(raw_tensor.clone(), location_tensor)
    p0_scores, _ = P0ClusterFilter.from_cfg(
        cfg, float(LOW_THRESHOLD), event_count=int(event_count)
    ).apply(raw_tensor.clone(), location_tensor)
    chained, _ = P18ScoreTrackRecovery.from_cfg(cfg, float(LOW_THRESHOLD)).apply(
        p0_scores.clone(), location_tensor
    )
    if not torch.equal(chained, final_scores):
        raise RuntimeError("manual P0->P18 chain differs from ChallengePostprocessor")
    return (
        p0_scores.numpy().astype(np.float32, copy=True),
        final_scores.numpy().astype(np.float32, copy=True),
        stats,
    )


def prepare_video(metadata: dict) -> PreparedLowVideo:
    source_name = str(metadata["source_name"])
    source_path = TRAIN_ROOT / source_name
    if not source_path.is_file() or sha256_file(source_path) != metadata["source_sha256"]:
        raise RuntimeError("raw train source changed: {}".format(source_name))
    cache_path = (CACHE_DIR / metadata["record"]).resolve()
    if not cache_path.is_file() or sha256_file(cache_path) != metadata["record_sha256"]:
        raise RuntimeError("score cache changed: {}".format(source_name))
    with np.load(cache_path, allow_pickle=False) as archive:
        if set(archive.files) != {"baseline_scores", "candidate_scores"}:
            raise RuntimeError("low cache arrays differ: {}".format(source_name))
        baseline_scores = np.asarray(archive["baseline_scores"], dtype=np.float32).copy()
        candidate_scores = np.asarray(archive["candidate_scores"], dtype=np.float32).copy()
    if not np.array_equal(baseline_scores.view(np.uint32), candidate_scores.view(np.uint32)):
        raise RuntimeError("low M10 cache is not bitwise identity: {}".format(source_name))

    # Construct C00 inputs before truth columns are accessed.
    with np.load(source_path, allow_pickle=False) as archive:
        locations3 = np.asarray(archive["ev_loc"], dtype=np.int64).copy()
    if locations3.shape != (int(metadata["event_count"]), 3):
        raise RuntimeError("input locations differ from cache event count: {}".format(source_name))
    locations4 = np.column_stack((np.zeros(locations3.shape[0], dtype=np.int64), locations3))
    p0_scores, final_scores, _ = p0_and_p18(
        baseline_scores, locations4, int(metadata["event_count"])
    )
    components = extract_atomic_components(
        final_scores,
        locations4,
        float(LOW_THRESHOLD),
        spatial_radius=2,
        temporal_bin_size=50,
        temporal_radius_bins=1,
    ).event_indices

    # Truth enters only after the observable route, C00 state, and partition exist.
    video = load_temporal_frame_video(source_path, TEMPORAL_BIN, 8000)
    if not np.array_equal(video.locations, locations3):
        raise RuntimeError("canonical truth loader locations differ: {}".format(source_name))
    labels = video.labels.astype(np.uint8, copy=True)
    target_ids = video.target_ids.copy()
    pure_fp = pure_false_positive_targets(components, labels)
    baseline = official_counts(final_scores, labels, target_ids, locations4)
    return PreparedLowVideo(
        source_name=source_name,
        family=family_for_name(source_name),
        source_sha256=str(metadata["source_sha256"]),
        cache_sha256=str(metadata["record_sha256"]),
        event_count=int(metadata["event_count"]),
        raw_scores=baseline_scores,
        p0_scores=p0_scores,
        final_scores=final_scores,
        locations4=locations4,
        labels=labels,
        target_ids=target_ids,
        components=components,
        pure_fp=pure_fp,
        baseline=baseline,
    )


def delete_components(video: PreparedLowVideo, delete_mask: np.ndarray) -> np.ndarray:
    decisions = np.asarray(delete_mask, dtype=bool).reshape(-1)
    if decisions.size != len(video.components):
        raise ValueError("delete mask and components differ")
    output = video.final_scores.copy()
    assigned = np.zeros(output.size, dtype=bool)
    for component_index, indices in enumerate(video.components):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(assigned[indices]):
            raise RuntimeError("atomic C00 components overlap")
        assigned[indices] = True
        if decisions[component_index]:
            output[indices] = np.float32(0.0)
    if not np.array_equal(output[~assigned].view(np.uint32), video.final_scores[~assigned].view(np.uint32)):
        raise RuntimeError("non-component scores changed")
    for component_index, indices in enumerate(video.components):
        if decisions[component_index]:
            if not np.all(output[indices] == np.float32(0.0)):
                raise RuntimeError("component deletion is partial")
        elif not np.array_equal(output[indices].view(np.uint32), video.final_scores[indices].view(np.uint32)):
            raise RuntimeError("retained component lost bitwise identity")
    return output


def all_pure_fp_oracle(videos):
    candidates = {}
    for video in videos:
        scores = delete_components(video, video.pure_fp.astype(bool))
        candidates[video.source_name] = official_counts(
            scores, video.labels, video.target_ids, video.locations4
        )
    baseline = sum_counts(video.baseline for video in videos)
    candidate = sum_counts(candidates.values())
    return {
        "action": "delete_every_post_C00_pure_false_positive_component",
        "baseline": record(baseline),
        "candidate": record(candidate),
        "count_delta": count_delta(candidate, baseline),
        "metric_delta": metric_delta(candidate, baseline),
        "per_source": {
            video.source_name: {
                "family": video.family,
                "component_count": len(video.components),
                "pure_fp_component_count": int(video.pure_fp.sum()),
                "candidate": record(candidates[video.source_name]),
                "count_delta": count_delta(candidates[video.source_name], video.baseline),
                "metric_delta": metric_delta(candidates[video.source_name], video.baseline),
            }
            for video in videos
        },
    }


def singleton_positive_gain_oracle(videos):
    """Exact best subset for all-PF candidate components when there are <=20.

    Large sources use a safe greedy lower bound and are labelled as such.  This
    stage is diagnostic only; it never becomes a runtime policy.
    """
    baseline = sum_counts(video.baseline for video in videos)
    selected_masks = {}
    exact_source_count = 0
    lower_bound_source_count = 0
    source_details = {}
    for video in videos:
        candidate_indices = np.flatnonzero(video.pure_fp)
        if candidate_indices.size <= 20:
            best_counts = video.baseline
            best_mask = np.zeros(len(video.components), dtype=bool)
            best_score = metrics_from_counts(best_counts)["score"]
            for state in range(1, 1 << candidate_indices.size):
                chosen = np.zeros(len(video.components), dtype=bool)
                chosen[candidate_indices[np.flatnonzero(
                    (state >> np.arange(candidate_indices.size)) & 1
                )]] = True
                counts = official_counts(
                    delete_components(video, chosen), video.labels, video.target_ids, video.locations4
                )
                score = metrics_from_counts(counts)["score"]
                if score > best_score + 1e-15:
                    best_score = score
                    best_counts = counts
                    best_mask = chosen
            selected_masks[video.source_name] = best_mask
            exact_source_count += 1
            method = "exact_bruteforce_all_pure_fp_subsets"
        else:
            # A deterministic, truth-only greedy lower bound, never claimed as an upper bound.
            selected = np.zeros(len(video.components), dtype=bool)
            current = video.baseline
            improved = True
            while improved:
                improved = False
                best_local = None
                best_local_score = metrics_from_counts(current)["score"]
                for component_index in candidate_indices:
                    if selected[component_index]:
                        continue
                    trial = selected.copy()
                    trial[component_index] = True
                    counts = official_counts(
                        delete_components(video, trial), video.labels, video.target_ids, video.locations4
                    )
                    score = metrics_from_counts(counts)["score"]
                    if score > best_local_score + 1e-15:
                        best_local = (component_index, counts)
                        best_local_score = score
                if best_local is not None:
                    selected[best_local[0]] = True
                    current = best_local[1]
                    improved = True
            selected_masks[video.source_name] = selected
            best_counts = current
            lower_bound_source_count += 1
            method = "deterministic_greedy_truth_only_lower_bound_not_exact"
        source_details[video.source_name] = {
            "method": method,
            "pure_fp_component_count": int(candidate_indices.size),
            "selected_component_count": int(selected_masks[video.source_name].sum()),
            "candidate": record(best_counts),
            "count_delta": count_delta(best_counts, video.baseline),
            "metric_delta": metric_delta(best_counts, video.baseline),
        }
    candidate = sum_counts(
        Counts(**source_details[video.source_name]["candidate"]["counts"])
        for video in videos
    )
    return {
        "action": "per_source_best_pure_FP_subset",
        "exactness": (
            "exact only for sources with <=20 pure-FP components; otherwise a "
            "truth-only greedy lower bound.  It is not a deployable policy or a global oracle."
        ),
        "exact_source_count": exact_source_count,
        "greedy_lower_bound_source_count": lower_bound_source_count,
        "baseline": record(baseline),
        "candidate": record(candidate),
        "count_delta": count_delta(candidate, baseline),
        "metric_delta": metric_delta(candidate, baseline),
        "per_source": source_details,
    }


def family_summaries(videos, global_baseline: Counts, global_oracle: dict):
    candidate_records = global_oracle["per_source"]
    summaries = {}
    for family, names in LOW_FAMILIES.items():
        family_videos = [video for video in videos if video.source_name in names]
        baseline = sum_counts(video.baseline for video in family_videos)
        candidate = sum_counts(
            Counts(**candidate_records[video.source_name]["candidate"]["counts"])
            for video in family_videos
        )
        summaries[family] = {
            "source_names": list(names),
            "baseline": record(baseline),
            "all_pure_FP_candidate": record(candidate),
            "count_delta": count_delta(candidate, baseline),
            "metric_delta": metric_delta(candidate, baseline),
            "global_score_after_replacing_low_domain_only": None,
        }
    return summaries


def val24_low_ceiling(all_pure_fp: dict):
    """Map a low-domain train oracle gain to the known Val24 low-count ceiling.

    This is intentionally not a prediction.  It states the optimistic outcome
    if the same low-domain count improvements transferred exactly to Val24,
    leaving all other domains unchanged.
    """
    baseline = {
        "true_positive_events": 63981,
        "false_positive_events": 2396,
        "positive_events": 65506,
        "detected_target_frames": 4649,
        "target_frames": 4762,
        "false_components": 1584,
        "frame_count": 3752,
    }
    low_baseline = {
        "true_positive_events": 23666,
        "false_positive_events": 806,
        "positive_events": 24074,
        "detected_target_frames": 2169,
        "target_frames": 2211,
        "false_components": 418,
        "frame_count": 1526,
    }
    train_delta = all_pure_fp["count_delta"]
    optimistic_low = dict(low_baseline)
    optimistic_low["true_positive_events"] += int(train_delta["true_positive_events"])
    optimistic_low["false_positive_events"] += int(train_delta["false_positive_events"])
    optimistic_low["detected_target_frames"] += int(train_delta["correct_target_frames"])
    optimistic_low["false_components"] += int(train_delta["false_components"])
    for name in ("true_positive_events", "false_positive_events", "detected_target_frames", "false_components"):
        optimistic_low[name] = max(0, optimistic_low[name])
    optimistic_low["true_positive_events"] = min(optimistic_low["true_positive_events"], optimistic_low["positive_events"])
    optimistic_low["detected_target_frames"] = min(optimistic_low["detected_target_frames"], optimistic_low["target_frames"])
    candidate = dict(baseline)
    for name in ("true_positive_events", "false_positive_events", "detected_target_frames", "false_components"):
        candidate[name] = baseline[name] - low_baseline[name] + optimistic_low[name]
    positive = candidate["positive_events"]
    iou = float(np.float32(candidate["true_positive_events"]) / np.float32(positive + candidate["false_positive_events"]))
    acc = float(np.float32(candidate["true_positive_events"]) / np.float32(positive))
    pd = candidate["detected_target_frames"] / candidate["target_frames"]
    fa = candidate["false_components"] / (candidate["frame_count"] * WIDTH * HEIGHT)
    score_fa, score = challenge_score(iou, acc, pd, fa)
    return {
        "meaning": "optimistic transfer thought experiment only; not validation evidence or a deployment estimate",
        "baseline_counts": baseline,
        "low_baseline_counts": low_baseline,
        "transferred_train_all_pure_FP_count_delta": train_delta,
        "optimistic_replaced_low_counts": optimistic_low,
        "full_val24_counts_if_delta_transferred_exactly": candidate,
        "full_val24_metrics_if_delta_transferred_exactly": {
            "iou": iou, "acc": acc, "pd": pd, "fa": fa, "score_fa": score_fa, "score": score,
        },
        "target_0_9700_reached_under_optimistic_transfer": bool(score >= 0.9700),
    }


def run(output: Path):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only low oracle refuses an initialized CUDA process")
    manifest, records = validate_inputs()
    videos = []
    for index, metadata in enumerate(records, start=1):
        video = prepare_video(metadata)
        videos.append(video)
        print(
            "prepared {}/{} {}: {} C00 components ({} pure FP)".format(
                index, len(records), video.source_name, len(video.components), int(video.pure_fp.sum())
            ),
            flush=True,
        )
    baseline = sum_counts(video.baseline for video in videos)
    all_pure_fp = all_pure_fp_oracle(videos)
    subset = singleton_positive_gain_oracle(videos)
    payload = {
        "schema": "ev-uav-low-m10-component-oracle-train-only-v1",
        "created_utc": utc_now(),
        "dataset_split": "train",
        "validation_or_test_read": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "model_training_or_inference_performed": False,
        "source_identity_is_feature": False,
        "action_space": {
            "route": "frozen M10 full-stream only, event_count <= 30000, threshold .718",
            "baseline": "raw M10 scores -> frozen P0/P0c -> frozen P18",
            "editable_unit": "complete post-C00 component using exact P0 topology rxy=2, bin=50, rt=1",
            "truth_usage": "labels/target IDs attached after C00 component partition; never used in a runtime action",
            "raw_recovery_oracle": "not run in v1; component deletion capacity is isolated first",
        },
        "inputs": {
            "manifest_path": str(MANIFEST_PATH),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "protocol_path": str(PROTOCOL_PATH),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "train_root": str(TRAIN_ROOT),
            "runner_sha256": sha256_file(Path(__file__)),
            "postprocess_sha256": sha256_file(ROOT / "utils" / "postprocess.py"),
            "atomic_component_sha256": sha256_file(ROOT / "utils" / "atomic_component_deletion.py"),
            "evaluator_sha256": sha256_file(ROOT / "utils" / "eval.py"),
        },
        "families": {name: list(values) for name, values in LOW_FAMILIES.items()},
        "baseline": record(baseline),
        "all_pure_false_positive_component_oracle": all_pure_fp,
        "best_pure_false_positive_subset_diagnostic": subset,
        "family_all_pure_fp_summaries": family_summaries(videos, baseline, all_pure_fp),
        "optimistic_val24_transfer_ceiling": val24_low_ceiling(all_pure_fp),
        "gates_for_low_expert": {
            "all_pure_fp_train_score_gain_positive": all_pure_fp["metric_delta"]["score"] > 0.0,
            "all_pure_fp_train_pd_not_lower": all_pure_fp["metric_delta"]["pd"] >= 0.0,
            "all_pure_fp_train_correct_frames_not_lower": all_pure_fp["count_delta"]["correct_target_frames"] >= 0,
            "optimistic_transfer_can_reach_0_9700": val24_low_ceiling(all_pure_fp)["target_0_9700_reached_under_optimistic_transfer"],
            "required_before_model_training": "cross-family feature separability and nested OOF must still pass",
        },
    }
    digest = write_json_exclusive(output, payload)
    print(json.dumps({
        "output": str(Path(output).resolve()),
        "sha256": digest,
        "baseline_score": payload["baseline"]["metrics"]["score"],
        "all_pure_fp_delta": all_pure_fp["metric_delta"],
        "subset_delta": subset["metric_delta"],
        "optimistic_val24_score": payload["optimistic_val24_transfer_ceiling"]["full_val24_metrics_if_delta_transferred_exactly"]["score"],
        "gates": payload["gates_for_low_expert"],
    }, ensure_ascii=False, indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

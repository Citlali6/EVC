"""Preregister and cross-fit the train-only track-edge recovery MVP.

The script accepts only the immutable official-train M20 cache.  It has no
validation argument and never loads a validation path.  Runtime integration
is intentionally absent: even a passing report is only train-side evidence
for a later, separately reviewed decision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import replay_temporal_memory_validation as replay
from crossfit_component_reranker import (
    CORRECT_THRESHOLD,
    EXPECTED_SELECTED_EVENT_COUNT,
    EXPECTED_SELECTED_VIDEO_COUNT,
    H1_NAMES,
    H2_NAMES,
    PD_DETECTION_INTERVAL,
    PREDICTION_THRESHOLD,
    RELEASED_M20_CHECKPOINT_SHA256,
    RESOLUTION,
    SufficientCounts,
    _validate_cache_population,
    metrics_from_counts,
    sufficient_counts_for_video,
    validate_c00_config,
)
from train_component_reranker import (
    _atomic_json,
    _load_cache_record,
    _require_new_output,
    load_train_cache,
)
from utils.component_reranker import (
    TRAIN_CACHE_SCHEMA,
    sha256_file,
    sha256_json,
    temporal_memory_inference_mapping,
)
from utils.postprocess import ChallengePostprocessor
from utils.track_edge_recovery import (
    FEATURE_NAMES,
    FEATURE_SEMANTICS_VERSION,
    FROZEN_TOPOLOGY,
    TrackEdgeCandidate,
    TrackEdgeTrainingTarget,
    attach_training_targets,
    extract_track_edge_candidates,
    select_endpoint_recoveries,
)


PROTOCOL_SCHEMA = "ev-uav-track-edge-recovery-crossfit-protocol-v1"
REPORT_SCHEMA = "ev-uav-track-edge-recovery-crossfit-report-v1"
PROTOCOL_DOCUMENT = "docs/TRACK_EDGE_RECOVERY_PROTOCOL.md"
CODE_PATHS = (
    "train_track_edge_recovery.py",
    "utils/track_edge_recovery.py",
    "crossfit_component_reranker.py",
    "train_component_reranker.py",
    "utils/postprocess.py",
    "utils/component_reranker.py",
    "utils/challenge_eval.py",
    "utils/eval.py",
    "replay_temporal_memory_validation.py",
)
FOLD_PLAN = (
    {
        "fold_id": "holdout_h1",
        "fit_high_block": "h2",
        "held_block": "h1",
    },
    {
        "fold_id": "holdout_h2",
        "fit_high_block": "h1",
        "held_block": "h2",
    },
)
HIDDEN_WIDTH = 8
TRAINABLE_PARAMETER_COUNT = 137
TRAIN_SEED = 53
TRAIN_STEPS = 200
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.001
ADAM_BETAS = (0.9, 0.999)
DECISION_LOGIT = 0.0
POOLED_SCORE_DELTA_GATE = 0.0002
MAX_FALSE_COMPONENTS_PER_PD_FOLD = 4.0
MAX_FALSE_COMPONENTS_PER_PD_POOLED = 3.0
MIN_POSITIVE_CANDIDATE_VIDEOS_PER_HELD_BLOCK = 2
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_sha256(value, name):
    value = str(value).strip().lower()
    if HEX64.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA-256.".format(name))
    return value


def _code_sha256(project_root):
    result = {}
    for relative in CODE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError("Track-edge source is missing: {}".format(path))
        result[relative] = sha256_file(path)
    return result


def _safe_new_output(path, cache_dir, kind, forbidden=()):
    """Keep audit outputs outside the immutable cache and input files."""
    path = Path(path).resolve()
    cache_dir = Path(cache_dir).resolve()
    try:
        path.relative_to(cache_dir)
    except ValueError:
        pass
    else:
        raise ValueError("{} must stay outside the immutable train cache.".format(kind))
    for input_path in forbidden:
        if path == Path(input_path).resolve():
            raise ValueError("{} must not alias an input file.".format(kind))
    return _require_new_output(path, kind)


@dataclass
class PreparedTrackVideo:
    source_name: str
    block: str
    event_count: int
    candidates: tuple
    targets: tuple
    features: np.ndarray
    baseline_counts: SufficientCounts
    baseline_scores: np.ndarray | None
    locations: np.ndarray | None
    event_labels: np.ndarray | None
    target_ids: np.ndarray | None


class TinyTrackEdgeMLP(nn.Module):
    """The frozen 15 -> 8 -> 1, 137-parameter utility classifier."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(len(FEATURE_NAMES), HIDDEN_WIDTH)
        self.output = nn.Linear(HIDDEN_WIDTH, 1)

    def forward(self, features):
        return self.output(torch.tanh(self.input(features))).squeeze(-1)


def _state_mapping(model):
    return {
        name: tensor.detach().cpu().double().numpy().tolist()
        for name, tensor in sorted(model.state_dict().items())
    }


def _flat_parameters(model):
    return torch.cat(
        [parameter.detach().reshape(-1).cpu().double() for parameter in model.parameters()]
    )


def marginal_score_utility(baseline_counts, target):
    """Return the exact one-action train-side Challenge-score marginal."""
    if not isinstance(baseline_counts, SufficientCounts):
        raise TypeError("baseline_counts must be SufficientCounts.")
    if not isinstance(target, TrackEdgeTrainingTarget):
        raise TypeError("target must be TrackEdgeTrainingTarget.")
    values = baseline_counts.to_dict()
    if target.label:
        values["true_positive_events"] += 1
        values["false_negative_events"] -= 1
        if target.recovers_target_group:
            values["correct_objects"] += 1
    else:
        values["false_positive_events"] += 1
        # Never reward a false event for accidentally merging two existing
        # false components; semantic IoU damage is still retained.
        values["false_components"] += max(0, int(target.false_component_delta))
    modified = SufficientCounts(**values)
    return (
        metrics_from_counts(modified)["score"]
        - metrics_from_counts(baseline_counts)["score"]
    )


def _weighted_standardizer(features, base_weights):
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    normalized = weights / weights.sum()
    mean = np.sum(features * normalized[:, None], axis=0)
    centered = features - mean
    scale = np.sqrt(np.sum(centered * centered * normalized[:, None], axis=0))
    scale[scale < 1e-8] = 1.0
    return mean, scale


def fit_metric_weighted_mlp(features, labels, base_weights, utilities):
    """Perform exactly 200 real CPU AdamW updates and return full evidence."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    utilities = np.asarray(utilities, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape != (labels.size, len(FEATURE_NAMES)):
        raise ValueError("MLP features/labels have incompatible shapes.")
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("MLP fitting requires both binary classes.")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("MLP labels must be binary.")
    if not (
        base_weights.size == utilities.size == labels.size
        and np.isfinite(features).all()
        and np.isfinite(base_weights).all()
        and np.isfinite(utilities).all()
    ):
        raise ValueError("MLP fit inputs must be finite and aligned.")
    if (base_weights <= 0.0).any() or (np.abs(utilities) <= 0.0).any():
        raise ValueError("Base weights and absolute utilities must be positive.")
    if not np.all(utilities[labels > 0.5] > 0.0):
        raise ValueError("Positive recovery utilities must be positive.")
    if not np.all(utilities[labels < 0.5] < 0.0):
        raise ValueError("False-event recovery utilities must be negative.")

    feature_mean, feature_scale = _weighted_standardizer(features, base_weights)
    standardized = (features - feature_mean) / feature_scale
    training_weights = base_weights * np.abs(utilities)
    training_weights /= training_weights.sum()
    x = torch.from_numpy(standardized).double()
    y = torch.from_numpy(labels).double()
    w = torch.from_numpy(training_weights).double()

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    model = TinyTrackEdgeMLP().double().cpu()
    if sum(parameter.numel() for parameter in model.parameters()) != TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("Frozen MLP parameter count is not 137.")
    initial_state = _state_mapping(model)
    initial_parameters = _flat_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    def loss_value():
        losses = F.binary_cross_entropy_with_logits(
            model(x), y, reduction="none"
        )
        return torch.sum(w * losses)

    initial_loss = float(loss_value().detach().item())
    trace = [{"step": 0, "loss": initial_loss}]
    try:
        for step in range(1, TRAIN_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value()
            if not torch.isfinite(loss):
                raise RuntimeError("Track-edge MLP produced non-finite loss.")
            loss.backward()
            optimizer.step()
            if step % 25 == 0 or step == TRAIN_STEPS:
                trace.append({"step": step, "loss": float(loss_value().detach().item())})
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    final_loss = float(loss_value().detach().item())
    final_parameters = _flat_parameters(model)
    parameter_delta_l2 = float(torch.linalg.vector_norm(
        final_parameters - initial_parameters
    ).item())
    optimizer_steps = []
    moment_l2 = 0.0
    moment_tensor_count = 0
    finite_optimizer_state = True
    for state in optimizer.state.values():
        step_value = state.get("step", 0)
        optimizer_steps.append(int(step_value.item() if torch.is_tensor(step_value) else step_value))
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(name)
            if tensor is None:
                continue
            moment_tensor_count += 1
            finite_optimizer_state = finite_optimizer_state and bool(
                torch.isfinite(tensor).all().item()
            )
            moment_l2 += float(torch.sum(tensor.detach().double() ** 2).item())
    moment_l2 = math.sqrt(moment_l2)
    if (
        not final_loss < initial_loss
        or parameter_delta_l2 <= 0.0
        or moment_tensor_count != 8
        or set(optimizer_steps) != {TRAIN_STEPS}
        or not finite_optimizer_state
        or moment_l2 <= 0.0
    ):
        raise RuntimeError("Real AdamW training-evidence checks failed.")
    final_state = _state_mapping(model)
    result = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "model_state": final_state,
        "training_evidence": {
            "device": "cpu",
            "seed": TRAIN_SEED,
            "optimizer": "AdamW",
            "optimizer_steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(ADAM_BETAS),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_trace": trace,
            "parameter_count": TRAINABLE_PARAMETER_COUNT,
            "parameter_delta_l2": parameter_delta_l2,
            "optimizer_moment_tensor_count": moment_tensor_count,
            "optimizer_moment_l2": moment_l2,
            "optimizer_state_finite": finite_optimizer_state,
            "initial_state_sha256": sha256_json(initial_state),
            "final_state_sha256": sha256_json(final_state),
            "training_example_count": int(labels.size),
            "positive_example_count": int(labels.sum()),
            "negative_example_count": int(labels.size - labels.sum()),
            "weighted_positive_mass": float(training_weights[labels > 0.5].sum()),
            "weighted_negative_mass": float(training_weights[labels < 0.5].sum()),
        },
    }
    return result


def predict_mlp_logits(features, fitted):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Prediction feature width differs from frozen schema.")
    mean = np.asarray(fitted["feature_mean"], dtype=np.float64)
    scale = np.asarray(fitted["feature_scale"], dtype=np.float64)
    state = fitted["model_state"]
    w1 = np.asarray(state["input.weight"], dtype=np.float64)
    b1 = np.asarray(state["input.bias"], dtype=np.float64)
    w2 = np.asarray(state["output.weight"], dtype=np.float64)
    b2 = np.asarray(state["output.bias"], dtype=np.float64)
    hidden = np.tanh(((features - mean) / scale) @ w1.T + b1)
    logits = hidden @ w2.T + b2
    logits = logits.reshape(-1)
    if not np.isfinite(logits).all():
        raise RuntimeError("Track-edge MLP produced non-finite logits.")
    return logits


def _block_for_name(name, middle_names):
    if name in H1_NAMES:
        return "h1"
    if name in H2_NAMES:
        return "h2"
    if name in middle_names:
        return "middle"
    raise ValueError("Train source is outside frozen blocks: {}".format(name))


def _prepare_videos(cache_dir, manifest, cfg, middle_names):
    videos = []
    for ordinal, metadata in enumerate(manifest["records"], start=1):
        record = _load_cache_record(cache_dir, metadata)
        event_count = int(metadata["event_count"])
        raw_scores = record["scores"].reshape(-1).astype(np.float32, copy=False)
        locations = np.column_stack(
            (
                np.zeros(event_count, dtype=np.int64),
                record["locs"].astype(np.int64, copy=False),
            )
        )
        processor = ChallengePostprocessor.from_cfg(
            cfg, PREDICTION_THRESHOLD, event_count=event_count
        )
        baseline_tensor, _ = processor.apply(
            torch.from_numpy(raw_scores.copy()),
            torch.from_numpy(locations).to(torch.int64).contiguous(),
        )
        baseline_scores = baseline_tensor.numpy().astype(np.float32, copy=False)
        candidates = extract_track_edge_candidates(
            raw_scores,
            baseline_scores,
            locations,
            event_count,
            FROZEN_TOPOLOGY,
        )
        targets = attach_training_targets(
            candidates,
            record["labels"],
            record["target_ids"],
            baseline_scores,
            locations,
            FROZEN_TOPOLOGY,
            CORRECT_THRESHOLD,
        )
        features = (
            np.stack([candidate.features for candidate in candidates]).astype(
                np.float64, copy=False
            )
            if candidates
            else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        )
        baseline_counts = sufficient_counts_for_video(
            baseline_scores,
            record["labels"],
            record["target_ids"],
            locations,
        )
        block = _block_for_name(metadata["source_name"], middle_names)
        keep_runtime = block in {"h1", "h2"}
        videos.append(
            PreparedTrackVideo(
                source_name=metadata["source_name"],
                block=block,
                event_count=event_count,
                candidates=candidates,
                targets=targets,
                features=features,
                baseline_counts=baseline_counts,
                baseline_scores=(baseline_scores.copy() if keep_runtime else None),
                locations=(locations.copy() if keep_runtime else None),
                event_labels=(record["labels"].astype(np.uint8, copy=True) if keep_runtime else None),
                target_ids=(record["target_ids"].copy() if keep_runtime else None),
            )
        )
        print(
            "prepare {}/{}: {} [{}] candidates={} positive={} pd_recoveries={}".format(
                ordinal,
                len(manifest["records"]),
                metadata["source_name"],
                block,
                len(candidates),
                sum(target.label for target in targets),
                sum(target.recovers_target_group for target in targets),
            ),
            flush=True,
        )
    return videos


def _sum_counts(videos):
    counts = SufficientCounts()
    for video in videos:
        counts = counts + video.baseline_counts
    return counts


def partition_fold_videos(fold, videos):
    held = [video for video in videos if video.block == fold["held_block"]]
    fit = [
        video
        for video in videos
        if video.block in {fold["fit_high_block"], "middle"}
    ]
    if not held or not fit:
        raise RuntimeError("Frozen track-edge fold is empty.")
    if {video.source_name for video in held} & {video.source_name for video in fit}:
        raise RuntimeError("Track-edge fit/held source leakage detected.")
    return fit, held


def balanced_training_dataset(fit_videos, fit_high_block):
    """Return hierarchical domain/video/endpoint/candidate weights."""
    domains = (
        [video for video in fit_videos if video.block == fit_high_block and video.candidates],
        [video for video in fit_videos if video.block == "middle" and video.candidates],
    )
    if not all(domains):
        raise ValueError("Both high and middle fit domains need candidate-bearing videos.")
    aggregate_counts = _sum_counts(fit_videos)
    feature_batches = []
    label_batches = []
    utility_batches = []
    weight_batches = []
    source_names = []
    endpoint_names = []
    for domain_videos in domains:
        video_mass = 0.5 / len(domain_videos)
        for video in domain_videos:
            endpoint_positions = {}
            for position, candidate in enumerate(video.candidates):
                endpoint_positions.setdefault(candidate.endpoint_key, []).append(position)
            endpoint_mass = video_mass / len(endpoint_positions)
            video_weights = np.empty(len(video.candidates), dtype=np.float64)
            for endpoint_key, positions in endpoint_positions.items():
                candidate_mass = endpoint_mass / len(positions)
                video_weights[positions] = candidate_mass
                endpoint_names.extend(
                    ["{}:{}:{}".format(video.source_name, *endpoint_key)] * len(positions)
                )
            feature_batches.append(video.features)
            label_batches.append(
                np.asarray([target.label for target in video.targets], dtype=np.float64)
            )
            utility_batches.append(
                np.asarray(
                    [
                        marginal_score_utility(aggregate_counts, target)
                        for target in video.targets
                    ],
                    dtype=np.float64,
                )
            )
            weight_batches.append(video_weights)
            source_names.extend([video.source_name] * len(video.candidates))
    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches)
    utilities = np.concatenate(utility_batches)
    base_weights = np.concatenate(weight_batches)
    if not math.isclose(float(base_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Track-edge hierarchical weights do not sum to one.")
    return {
        "features": features,
        "labels": labels,
        "utilities": utilities,
        "base_weights": base_weights,
        "source_names": tuple(source_names),
        "endpoint_names": tuple(endpoint_names),
        "aggregate_baseline_counts": aggregate_counts,
    }


def _evaluate_held_video(video, fitted):
    if any(
        value is None
        for value in (
            video.baseline_scores,
            video.locations,
            video.event_labels,
            video.target_ids,
        )
    ):
        raise ValueError("Held runtime arrays are unavailable.")
    logits = predict_mlp_logits(video.features, fitted)
    selected_indices = select_endpoint_recoveries(
        video.candidates, logits, DECISION_LOGIT
    )
    recovered_scores = video.baseline_scores.copy()
    recovered_scores[selected_indices] = PREDICTION_THRESHOLD
    counts = sufficient_counts_for_video(
        recovered_scores,
        video.event_labels,
        video.target_ids,
        video.locations,
    )
    selected_set = set(selected_indices.tolist())
    selected_targets = [
        target
        for candidate, target in zip(video.candidates, video.targets)
        if candidate.event_index in selected_set
    ]
    return counts, {
        "selected_event_count": len(selected_indices),
        "selected_true_events": sum(target.label for target in selected_targets),
        "selected_false_events": sum(not target.label for target in selected_targets),
        "selected_pd_recovery_candidates": sum(
            target.recovers_target_group for target in selected_targets
        ),
    }


def _evaluate_fold(fold, videos):
    fit_videos, held_videos = partition_fold_videos(fold, videos)
    dataset = balanced_training_dataset(fit_videos, fold["fit_high_block"])
    fitted = fit_metric_weighted_mlp(
        dataset["features"],
        dataset["labels"],
        dataset["base_weights"],
        dataset["utilities"],
    )
    baseline_counts = _sum_counts(held_videos)
    recovered_counts = SufficientCounts()
    action_totals = {
        "selected_event_count": 0,
        "selected_true_events": 0,
        "selected_false_events": 0,
        "selected_pd_recovery_candidates": 0,
    }
    per_video = []
    for video in held_videos:
        counts, actions = _evaluate_held_video(video, fitted)
        recovered_counts = recovered_counts + counts
        for name in action_totals:
            action_totals[name] += actions[name]
        per_video.append(
            {
                "source_name": video.source_name,
                "candidate_count": len(video.candidates),
                "positive_candidate_count": sum(
                    target.label for target in video.targets
                ),
                "pd_recovery_candidate_count": sum(
                    target.recovers_target_group for target in video.targets
                ),
                "actions": actions,
                "baseline_counts": video.baseline_counts.to_dict(),
                "recovered_counts": counts.to_dict(),
            }
        )
    baseline_metrics = metrics_from_counts(baseline_counts)
    recovered_metrics = metrics_from_counts(recovered_counts)
    positive_candidate_videos = sum(
        any(target.label for target in video.targets) for video in held_videos
    )
    new_pd_groups = recovered_counts.correct_objects - baseline_counts.correct_objects
    false_component_delta = (
        recovered_counts.false_components - baseline_counts.false_components
    )
    false_per_pd = (
        max(0, false_component_delta) / new_pd_groups
        if new_pd_groups > 0
        else None
    )
    return {
        "fold_id": fold["fold_id"],
        "fit_high_block": fold["fit_high_block"],
        "held_block": fold["held_block"],
        "fit_video_names": sorted(set(dataset["source_names"])),
        "held_video_names": [video.source_name for video in held_videos],
        "fit_example_count": int(dataset["features"].shape[0]),
        "fit_positive_count": int(dataset["labels"].sum()),
        "fit_negative_count": int(dataset["labels"].size - dataset["labels"].sum()),
        "fit_endpoint_count": len(set(dataset["endpoint_names"])),
        "model": fitted,
        "positive_candidate_videos": positive_candidate_videos,
        "baseline": {
            "counts": baseline_counts.to_dict(),
            "metrics": baseline_metrics,
        },
        "recovered": {
            "counts": recovered_counts.to_dict(),
            "metrics": recovered_metrics,
            "delta": {
                name: recovered_metrics[name] - baseline_metrics[name]
                for name in recovered_metrics
            },
        },
        "actions": action_totals,
        "new_pd_groups": new_pd_groups,
        "false_component_delta": false_component_delta,
        "false_components_per_new_pd_group": false_per_pd,
        "per_video": per_video,
    }


def evaluate_gates(fold_results, middle_counts):
    if len(fold_results) != 2:
        raise ValueError("Track-edge gates require exactly two folds.")
    by_held = {fold["held_block"]: fold for fold in fold_results}
    if set(by_held) != {"h1", "h2"}:
        raise ValueError("Track-edge gates require held H1 and H2.")
    baseline_high = SufficientCounts()
    recovered_high = SufficientCounts()
    for fold in fold_results:
        baseline_high += SufficientCounts(**fold["baseline"]["counts"])
        recovered_high += SufficientCounts(**fold["recovered"]["counts"])
    baseline_pooled = baseline_high + middle_counts
    recovered_pooled = recovered_high + middle_counts
    baseline_metrics = metrics_from_counts(baseline_pooled)
    recovered_metrics = metrics_from_counts(recovered_pooled)
    pooled_new_pd = recovered_high.correct_objects - baseline_high.correct_objects
    pooled_false_delta = recovered_high.false_components - baseline_high.false_components
    pooled_false_per_pd = (
        max(0, pooled_false_delta) / pooled_new_pd
        if pooled_new_pd > 0
        else None
    )
    checks = {}
    for block in ("h1", "h2"):
        fold = by_held[block]
        checks["{}_score_delta_positive".format(block)] = (
            fold["recovered"]["delta"]["score"] > 0.0
        )
        checks["{}_pd_group_recovered".format(block)] = fold["new_pd_groups"] >= 1
        checks["{}_iou_nondecrease".format(block)] = (
            fold["recovered"]["delta"]["iou"] >= 0.0
        )
        checks["{}_false_components_per_pd_at_most_4".format(block)] = (
            fold["false_components_per_new_pd_group"] is not None
            and fold["false_components_per_new_pd_group"]
            <= MAX_FALSE_COMPONENTS_PER_PD_FOLD
        )
        checks["{}_positive_candidates_span_two_videos".format(block)] = (
            fold["positive_candidate_videos"]
            >= MIN_POSITIVE_CANDIDATE_VIDEOS_PER_HELD_BLOCK
        )
        evidence = fold["model"]["training_evidence"]
        checks["{}_real_training_evidence".format(block)] = (
            evidence["optimizer_steps"] == TRAIN_STEPS
            and evidence["parameter_delta_l2"] > 0.0
            and evidence["final_loss"] < evidence["initial_loss"]
            and evidence["optimizer_moment_l2"] > 0.0
        )
    checks["pooled_false_components_per_pd_at_most_3"] = (
        pooled_false_per_pd is not None
        and pooled_false_per_pd <= MAX_FALSE_COMPONENTS_PER_PD_POOLED
    )
    checks["pooled_pd_strict_increase"] = recovered_metrics["pd"] > baseline_metrics["pd"]
    checks["pooled_iou_nondecrease"] = recovered_metrics["iou"] >= baseline_metrics["iou"]
    checks["pooled_score_delta_at_least_0p0002"] = (
        recovered_metrics["score"] - baseline_metrics["score"]
        >= POOLED_SCORE_DELTA_GATE
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_pooled": {
            "counts": baseline_pooled.to_dict(),
            "metrics": baseline_metrics,
        },
        "recovered_pooled_oof": {
            "counts": recovered_pooled.to_dict(),
            "metrics": recovered_metrics,
            "delta": {
                name: recovered_metrics[name] - baseline_metrics[name]
                for name in recovered_metrics
            },
        },
        "held_high_tradeoff": {
            "new_pd_groups": pooled_new_pd,
            "false_component_delta": pooled_false_delta,
            "false_components_per_new_pd_group": pooled_false_per_pd,
        },
        "runtime_artifact_emitted": False,
    }


def _build_protocol_definition(cache_dir, config_path, overrides):
    project_root = Path(__file__).resolve().parent
    if FROZEN_TOPOLOGY.temporal_bin_size != PD_DETECTION_INTERVAL:
        raise RuntimeError(
            "Track-edge temporal bins must equal the official Pd detection interval."
        )
    cache_dir, _, manifest_sha256, manifest = load_train_cache(cache_dir)
    selected_names, middle_names = _validate_cache_population(manifest, cache_dir)
    config_path = Path(config_path).resolve()
    cfg = replay.load_flat_config(config_path, list(overrides))
    postprocess_contract = validate_c00_config(cfg, PREDICTION_THRESHOLD)
    inference_settings = temporal_memory_inference_mapping(cfg)
    if manifest.get("inference_settings") != inference_settings:
        raise ValueError("Track-edge config inference settings differ from cache.")
    document_path = project_root / PROTOCOL_DOCUMENT
    if not document_path.is_file():
        raise FileNotFoundError("Frozen track-edge protocol document is missing.")
    return {
        "dataset": {
            "dataset_split": "train",
            "cache_schema": TRAIN_CACHE_SCHEMA,
            "cache_manifest_sha256": manifest_sha256,
            "expected_selected_video_count": EXPECTED_SELECTED_VIDEO_COUNT,
            "expected_selected_event_count": EXPECTED_SELECTED_EVENT_COUNT,
            "selected_video_names": list(selected_names),
            "selected_source_identities": [
                {
                    "source_name": item["source_name"],
                    "source_sha256": item["source_sha256"],
                    "event_count": int(item["event_count"]),
                    "record": item["record"],
                    "record_sha256": item["record_sha256"],
                }
                for item in manifest["records"]
            ],
            "official_train_source_manifest_sha256": manifest[
                "official_train_source_manifest_sha256"
            ],
            "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        },
        "blocks": {
            "h1": list(H1_NAMES),
            "h2": list(H2_NAMES),
            "middle": list(middle_names),
        },
        "fold_plan": [dict(fold) for fold in FOLD_PLAN],
        "candidate_extraction": {
            "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "topology": FROZEN_TOPOLOGY.to_dict(),
            "absolute_coordinates_are_features": False,
            "absolute_timestamps_are_features": False,
            "source_or_file_identity_is_feature": False,
            "labels_or_target_ids_are_features": False,
            "maximum_recoveries_per_track_endpoint": 1,
        },
        "fit": {
            "architecture": "Linear(15,8)->tanh->Linear(8,1)",
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "optimizer": "AdamW",
            "device": "cpu",
            "seed": TRAIN_SEED,
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(ADAM_BETAS),
            "decision_logit": DECISION_LOGIT,
            "loss": "official_marginal_utility_weighted_binary_cross_entropy",
            "domain_base_weight": {"fit_high": 0.5, "middle": 0.5},
            "hierarchy": "candidate-bearing video -> endpoint -> candidate equal weight",
            "negative_false_component_merge_reward_clipped": True,
            "model_selection": "none",
        },
        "gates": {
            "each_held_block_score_delta_strictly_positive": True,
            "each_held_block_new_pd_groups_minimum": 1,
            "each_held_block_iou_delta_minimum": 0.0,
            "each_held_block_false_components_per_pd_maximum": MAX_FALSE_COMPONENTS_PER_PD_FOLD,
            "pooled_false_components_per_pd_maximum": MAX_FALSE_COMPONENTS_PER_PD_POOLED,
            "pooled_score_delta_minimum": POOLED_SCORE_DELTA_GATE,
            "pooled_pd_strictly_increases": True,
            "pooled_iou_delta_minimum": 0.0,
            "positive_candidate_videos_per_held_block_minimum": MIN_POSITIVE_CANDIDATE_VIDEOS_PER_HELD_BLOCK,
            "real_adamw_training_evidence_required": True,
        },
        "scoring": {
            "prediction_threshold": PREDICTION_THRESHOLD,
            "pd_detection_interval": PD_DETECTION_INTERVAL,
            "correct_threshold": CORRECT_THRESHOLD,
            "resolution": list(RESOLUTION),
            "pd_fa_time_intervals": "unchanged official open intervals",
        },
        "promotion": {
            "runtime_integration_in_this_mvp": False,
            "validation_access_in_this_mvp": False,
            "artifact_emission_in_this_mvp": False,
            "next_action_if_gates_fail": "archive negative report without tuning",
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "overrides": list(overrides),
            "postprocess_contract": postprocess_contract,
            "postprocess_contract_sha256": sha256_json(postprocess_contract),
            "inference_settings": inference_settings,
            "inference_settings_sha256": sha256_json(inference_settings),
        },
        "frozen_document": {
            "path": PROTOCOL_DOCUMENT,
            "sha256": sha256_file(document_path),
        },
        "code_sha256": _code_sha256(project_root),
        "software_versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def preregister_protocol(args):
    output_path = _safe_new_output(
        args.output_protocol, args.cache_dir, "Track-edge protocol"
    )
    definition = _build_protocol_definition(
        args.cache_dir, args.config, args.override
    )
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "created_utc": _utc_now(),
        "definition": definition,
        "definition_sha256": sha256_json(definition),
    }
    _atomic_json(output_path, payload)
    print("wrote track-edge protocol:", output_path)
    print("protocol file sha256:", sha256_file(output_path))
    print("protocol definition sha256:", payload["definition_sha256"])
    return 0


def _load_protocol(path, expected_sha256):
    path = Path(path).resolve()
    expected_sha256 = _require_sha256(expected_sha256, "expected protocol SHA-256")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Track-edge protocol file SHA-256 mismatch.")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported track-edge protocol schema.")
    if sha256_json(payload.get("definition")) != payload.get("definition_sha256"):
        raise ValueError("Track-edge protocol canonical SHA-256 mismatch.")
    return payload, expected_sha256


def run_crossfit(args):
    output_report = _safe_new_output(
        args.output_report,
        args.cache_dir,
        "Track-edge report",
        forbidden=(args.protocol,),
    )
    protocol, protocol_sha256 = _load_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    definition = protocol["definition"]
    current_definition = _build_protocol_definition(
        args.cache_dir,
        definition["config"]["path"],
        definition["config"]["overrides"],
    )
    if sha256_json(current_definition) != protocol["definition_sha256"]:
        raise ValueError("Track-edge code/config/cache changed after preregistration.")
    cache_dir, _, manifest_sha256, manifest = load_train_cache(args.cache_dir)
    _, middle_names = _validate_cache_population(manifest, cache_dir)
    cfg = replay.load_flat_config(
        definition["config"]["path"], definition["config"]["overrides"]
    )
    videos = _prepare_videos(cache_dir, manifest, cfg, middle_names)
    fold_results = [_evaluate_fold(fold, videos) for fold in FOLD_PLAN]
    middle_counts = _sum_counts(
        [video for video in videos if video.block == "middle"]
    )
    gates = evaluate_gates(fold_results, middle_counts)
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": _utc_now(),
        "protocol_path": str(Path(args.protocol).resolve()),
        "protocol_file_sha256": protocol_sha256,
        "protocol_definition_sha256": protocol["definition_sha256"],
        "cache_manifest_sha256": manifest_sha256,
        "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "dataset_split": manifest["dataset_split"],
        "software_versions": dict(definition["software_versions"]),
        "fold_results": fold_results,
        "gates": gates,
        "conclusion": (
            "train_only_gate_passed_runtime_still_disabled"
            if gates["passed"]
            else "train_only_gate_failed_no_runtime_or_validation"
        ),
    }
    _atomic_json(output_report, report)
    print("wrote track-edge report:", output_report)
    print("report sha256:", sha256_file(output_report))
    print("train-only gates passed:", gates["passed"])
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--cache-dir", required=True)
    preregister.add_argument("--config", required=True)
    preregister.add_argument("--override", action="append", default=[])
    preregister.add_argument("--output-protocol", required=True)
    preregister.set_defaults(handler=preregister_protocol)
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", required=True)
    run.add_argument("--expected-protocol-sha256", required=True)
    run.add_argument("--cache-dir", required=True)
    run.add_argument("--output-report", required=True)
    run.set_defaults(handler=run_crossfit)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

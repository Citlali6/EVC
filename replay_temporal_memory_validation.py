"""Cache raw temporal-memory probabilities and replay Challenge 2 policies.

This utility separates expensive model inference from validation-only policy
analysis:

* ``cache`` runs one temporal-memory checkpoint over the complete ``val``
  split and stores raw per-event probabilities plus the evaluator source
  fields.  The split is intentionally fixed to ``val``; there is no test-label
  input path.
* ``replay`` optionally routes a secondary cache by observable event count,
  applies the project's real P6/P0/P0c/P18 implementations, and sweeps a
  generic low/high threshold grid without another model forward.

Video names are retained solely for provenance/alignment and are never used to
select a model, threshold, or post-processing rule.  Challenge metrics are
computed through ``utils.challenge_eval.evaluate_challenge_metrics`` rather
than a local copy of the scoring formula.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from utils.challenge_eval import (
    ChallengeMetrics,
    add_batch_to_evaluator,
    evaluate_challenge_metrics,
)
from utils.density_threshold import ChallengeCountTotals, select_density_threshold
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


CACHE_SCHEMA = "evc-temporal-memory-raw-probabilities-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "evisseg_evuav.yaml"
METRIC_NAMES = ("iou", "acc", "pd", "fa", "score_fa", "score")
CODE_PROVENANCE_PATHS = (
    "dataset/ev_uav.py",
    "dataset/temporal_frame.py",
    "model/temporal_memory_net.py",
    "utils/temporal_frame_inference.py",
    "utils/temporal_memory_inference.py",
)


@dataclass(frozen=True)
class RoutedRecord:
    """One validation record with scores selected by event-count routing."""

    file_name: str
    event_count: int
    scores: torch.Tensor
    seg_label: torch.Tensor
    locs: torch.Tensor
    idx_label: np.ndarray
    source_sha256: str
    score_source: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_for_digest(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(value))


def source_digest(locs, seg_label, idx_label) -> str:
    """Fingerprint evaluator source fields without influencing decisions."""

    digest = hashlib.sha256()
    for field_name, value in (
        ("locs", locs),
        ("seg_label", seg_label),
        ("idx_label", idx_label),
    ):
        array = _array_for_digest(value)
        digest.update(field_name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _apply_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(
            "Invalid --override {!r}; expected SECTION.KEY=value.".format(override)
        )
    path, raw_value = override.split("=", 1)
    keys = path.split(".")
    if len(keys) < 2 or any(not key for key in keys):
        raise ValueError(
            "Invalid --override path {!r}; expected SECTION.KEY=value.".format(path)
        )
    target = config
    for key in keys[:-1]:
        if not isinstance(target, dict) or key not in target:
            raise KeyError("Unknown configuration section in --override: {}".format(path))
        target = target[key]
    final_key = keys[-1]
    if not isinstance(target, dict) or final_key not in target:
        raise KeyError("Unknown configuration option in --override: {}".format(path))
    target[final_key] = yaml.safe_load(raw_value)


def load_flat_config(config_path: Path, overrides: Sequence[str]) -> SimpleNamespace:
    """Load the project YAML into the same flat attribute view used by cfg."""

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The configuration root must be a YAML mapping.")
    # Round-trip through JSON-compatible data so command overrides never mutate
    # an object shared with another caller.
    config = json.loads(json.dumps(config, ensure_ascii=False))
    for override in overrides:
        _apply_override(config, override)

    flattened = {
        option: value
        for section in config.values()
        for option, value in section.items()
    }
    flattened["resolved_config"] = config
    flattened["config_path"] = str(config_path)
    flattened["config_overrides"] = list(overrides)
    return SimpleNamespace(**flattened)


def decimal_grid(minimum: str, maximum: str, step: str) -> Tuple[float, ...]:
    """Build an inclusive threshold grid without cumulative float drift."""

    lower = Decimal(str(minimum))
    upper = Decimal(str(maximum))
    increment = Decimal(str(step))
    if increment <= 0:
        raise ValueError("threshold step must be positive.")
    if not (Decimal("0") < lower <= upper < Decimal("1")):
        raise ValueError("threshold bounds must satisfy 0 < min <= max < 1.")
    span = upper - lower
    if span % increment != 0:
        raise ValueError("threshold range must be exactly divisible by the step.")
    count = int(span / increment)
    return tuple(float(lower + index * increment) for index in range(count + 1))


def _inference_settings(cfg) -> dict:
    return {
        "temporal_memory_bin_size": int(cfg.temporal_memory_bin_size),
        "temporal_memory_context_bins": int(cfg.temporal_memory_context_bins),
        "temporal_memory_width": int(cfg.temporal_memory_width),
        "temporal_memory_sequence_length": int(cfg.temporal_memory_sequence_length),
        "temporal_memory_inference_batch_size": int(
            cfg.temporal_memory_inference_batch_size
        ),
        "temporal_memory_log_count_clip": float(cfg.temporal_memory_log_count_clip),
        "whole_t": int(cfg.whole_t),
        "resolution": [int(cfg.res[0]), int(cfg.res[1])],
    }


def _code_provenance(project_root: Path) -> dict:
    result = {}
    for relative_path in CODE_PROVENANCE_PATHS:
        path = project_root / relative_path
        result[relative_path] = sha256_file(path) if path.is_file() else None
    return result


def _dataset_signature(records: Sequence[Mapping]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["file_name"]).encode("utf-8"))
        digest.update(str(int(record["event_count"])).encode("ascii"))
        digest.update(str(record["source_sha256"]).encode("ascii"))
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, output_path: Path) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_raw_cache(
    checkpoint_path: Path,
    output_path: Path,
    cfg,
    expected_video_count: int = 24,
    device_name: str = "cuda:0",
) -> dict:
    """Run one checkpoint once per complete validation video and save raw scores."""

    # Imports stay inside the GPU-only operation so CPU replay and unit tests do
    # not initialize model code or require optional sparse-convolution modules.
    from dataset.ev_uav import EvUAV
    from utils.inference_chunks import evaluation_batch_from_sample
    from utils.temporal_frame_inference import temporal_frame_video_from_sample
    from utils.temporal_memory_inference import (
        load_temporal_memory_model,
        predict_temporal_memory_scores,
    )

    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint_path))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build a raw probability cache.")
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError("temporal_memory_context_bins must be odd.")
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError("temporal_memory_sequence_length must exceed one.")

    checkpoint_digest = sha256_file(checkpoint_path)
    device = torch.device(device_name)
    model, checkpoint = load_temporal_memory_model(
        str(checkpoint_path),
        device,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        cfg.temporal_memory_sequence_length,
    )
    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError("No validation files found in: {}".format(dataset.root))
    if expected_video_count and len(dataset.file_list) != int(expected_video_count):
        raise RuntimeError(
            "Expected {} validation videos, found {} in {}.".format(
                expected_video_count, len(dataset.file_list), dataset.root
            )
        )

    records = []
    started = datetime.now(timezone.utc)
    for video_index, file_name in enumerate(dataset.file_list, start=1):
        sample = dataset[video_index - 1]
        batch = evaluation_batch_from_sample(sample)
        event_count = int(len(sample["ev_loc"]))
        frame_video = temporal_frame_video_from_sample(
            sample,
            cfg.temporal_memory_bin_size,
            cfg.whole_t,
        )
        scores = predict_temporal_memory_scores(
            model,
            frame_video,
            device,
            cfg.temporal_memory_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
        ).reshape(-1).detach().cpu().to(torch.float32).contiguous()
        locs = batch["locs"].detach().cpu().to(torch.int64).contiguous()
        seg_label = batch["seg_label"].detach().cpu().contiguous()
        idx_label = np.ascontiguousarray(batch["idx_label"].copy())
        if not (
            scores.numel()
            == seg_label.numel()
            == locs.shape[0]
            == idx_label.shape[0]
            == event_count
        ):
            raise RuntimeError(
                "Prediction/source count mismatch for {}: scores={}, labels={}, "
                "locations={}, target_ids={}, events={}.".format(
                    file_name,
                    scores.numel(),
                    seg_label.numel(),
                    locs.shape[0],
                    idx_label.shape[0],
                    event_count,
                )
            )
        if not torch.isfinite(scores).all() or bool((scores < 0).any()) or bool(
            (scores > 1).any()
        ):
            raise RuntimeError("Non-probability score found for {}.".format(file_name))
        record_digest = source_digest(locs, seg_label, idx_label)
        records.append(
            {
                "file_name": str(file_name),
                "event_count": event_count,
                "scores": scores,
                "seg_label": seg_label,
                "locs": locs,
                "idx_label": idx_label,
                "source_sha256": record_digest,
            }
        )
        print(
            "cache {}/{}: {} ({} events)".format(
                video_index, len(dataset.file_list), file_name, event_count
            ),
            flush=True,
        )
        del sample, batch, frame_video, scores, locs, seg_label, idx_label
        torch.cuda.empty_cache()

    finished = datetime.now(timezone.utc)
    if sha256_file(checkpoint_path) != checkpoint_digest:
        raise RuntimeError(
            "Checkpoint changed while inference was running; refusing to save a mixed cache."
        )
    project_root = Path(__file__).resolve().parent
    metadata = {
        "schema": CACHE_SCHEMA,
        "created_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_split": "val",
        "dataset_root": str(Path(dataset.root).resolve()),
        "dataset_signature": _dataset_signature(records),
        "video_count": len(records),
        "event_count": sum(record["event_count"] for record in records),
        "inference_settings": _inference_settings(cfg),
        "config_path": str(getattr(cfg, "config_path", "")),
        "config_overrides": list(getattr(cfg, "config_overrides", ())),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "code_sha256": _code_provenance(project_root),
    }
    payload = {"metadata": metadata, "records": records}
    _atomic_torch_save(payload, output_path)
    return payload


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.x/early 2.x compatibility.
        return torch.load(path, map_location="cpu")


def validate_cache_payload(payload: Mapping, cache_name: str = "cache") -> None:
    """Reject stale, malformed, incomplete, or non-probability cache data."""

    if not isinstance(payload, Mapping):
        raise TypeError("{} payload must be a mapping.".format(cache_name))
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("{} has an unsupported cache schema.".format(cache_name))
    if metadata.get("dataset_split") != "val":
        raise ValueError("{} is not a validation cache.".format(cache_name))
    if not isinstance(records, list) or not records:
        raise ValueError("{} contains no validation records.".format(cache_name))
    seen_names = set()
    for index, record in enumerate(records):
        required = {
            "file_name",
            "event_count",
            "scores",
            "seg_label",
            "locs",
            "idx_label",
            "source_sha256",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(
                "{} record {} is missing fields: {}.".format(
                    cache_name, index, ", ".join(sorted(missing))
                )
            )
        file_name = str(record["file_name"])
        if file_name in seen_names:
            raise ValueError("{} contains duplicate video {!r}.".format(cache_name, file_name))
        seen_names.add(file_name)
        event_count = int(record["event_count"])
        scores = torch.as_tensor(record["scores"]).reshape(-1)
        labels = torch.as_tensor(record["seg_label"]).reshape(-1)
        locs = torch.as_tensor(record["locs"])
        target_ids = np.asarray(record["idx_label"]).reshape(-1)
        if locs.ndim != 2 or locs.shape[1] < 4:
            raise ValueError("{} {} locations must have shape [N, 4+].".format(cache_name, file_name))
        if not (event_count == scores.numel() == labels.numel() == locs.shape[0] == target_ids.size):
            raise ValueError("{} {} source lengths do not match.".format(cache_name, file_name))
        if not torch.isfinite(scores).all() or bool((scores < 0).any()) or bool(
            (scores > 1).any()
        ):
            raise ValueError("{} {} contains non-probability scores.".format(cache_name, file_name))
        expected_digest = source_digest(locs, labels, target_ids)
        if record["source_sha256"] != expected_digest:
            raise ValueError("{} {} source digest does not match.".format(cache_name, file_name))
    if int(metadata.get("video_count", -1)) != len(records):
        raise ValueError("{} video count metadata does not match.".format(cache_name))
    if int(metadata.get("event_count", -1)) != sum(int(r["event_count"]) for r in records):
        raise ValueError("{} event count metadata does not match.".format(cache_name))
    if metadata.get("dataset_signature") != _dataset_signature(records):
        raise ValueError("{} dataset signature does not match.".format(cache_name))


def load_cache(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Raw probability cache does not exist: {}".format(path))
    payload = _torch_load_cpu(path)
    validate_cache_payload(payload, str(path))
    return payload


def route_cache_records(
    primary_payload: Mapping,
    secondary_payload: Optional[Mapping] = None,
    secondary_max_events: int = 0,
) -> List[RoutedRecord]:
    """Route aligned raw scores using event count only (never file identity)."""

    validate_cache_payload(primary_payload, "primary cache")
    if int(secondary_max_events) < 0:
        raise ValueError("secondary_max_events must be non-negative.")
    primary_records = primary_payload["records"]
    secondary_records = None
    if secondary_payload is not None:
        validate_cache_payload(secondary_payload, "secondary cache")
        secondary_records = secondary_payload["records"]
        if len(primary_records) != len(secondary_records):
            raise ValueError("Primary and secondary cache video counts differ.")
    elif secondary_max_events:
        raise ValueError("secondary_max_events requires a secondary cache.")

    routed = []
    for index, primary in enumerate(primary_records):
        secondary = secondary_records[index] if secondary_records is not None else None
        if secondary is not None:
            for field in ("file_name", "event_count", "source_sha256"):
                if primary[field] != secondary[field]:
                    raise ValueError(
                        "Cache alignment mismatch at record {} field {}.".format(index, field)
                    )
            if tuple(primary["scores"].shape) != tuple(secondary["scores"].shape):
                raise ValueError(
                    "Cache score shape mismatch for {}.".format(primary["file_name"])
                )
        use_secondary = bool(
            secondary is not None
            and int(secondary_max_events) > 0
            and int(primary["event_count"]) <= int(secondary_max_events)
        )
        selected = secondary if use_secondary else primary
        routed.append(
            RoutedRecord(
                file_name=str(primary["file_name"]),
                event_count=int(primary["event_count"]),
                scores=torch.as_tensor(selected["scores"]).reshape(-1).cpu().contiguous(),
                seg_label=torch.as_tensor(primary["seg_label"]).reshape(-1).cpu().contiguous(),
                locs=torch.as_tensor(primary["locs"]).cpu().contiguous(),
                idx_label=np.ascontiguousarray(primary["idx_label"]),
                source_sha256=str(primary["source_sha256"]),
                score_source="secondary" if use_secondary else "primary",
            )
        )
    return routed


def evaluate_cached_video(record: RoutedRecord, threshold: float, cfg) -> ChallengeCountTotals:
    """Return exact per-video sufficient counts after real project postprocessing."""

    threshold = float(threshold)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
    predictions, _ = postprocessor.apply(record.scores.clone(), record.locs)
    evaluator = evalute(cfg)
    batch = {
        "seg_label": record.seg_label,
        "locs": record.locs,
        "idx_label": record.idx_label,
    }
    add_batch_to_evaluator(
        evaluator,
        batch,
        predictions,
        sample_number=0,
        prediction_threshold=threshold,
        collect_roc=True,
    )
    labels = record.seg_label.float().reshape(-1)
    positive_mask = labels > 0.5
    binary_predictions = predictions.reshape(-1) >= threshold
    return ChallengeCountTotals(
        true_positive_events=int((binary_predictions & positive_mask).sum().item()),
        false_positive_events=int((binary_predictions & ~positive_mask).sum().item()),
        positive_events=int(positive_mask.sum().item()),
        detected_target_frames=int(evaluator.correct_num),
        target_frames=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
    )


def _sum_counts(counts: Iterable[ChallengeCountTotals]) -> ChallengeCountTotals:
    counts = tuple(counts)
    if not counts:
        raise ValueError("At least one count record is required.")
    return ChallengeCountTotals(
        true_positive_events=sum(item.true_positive_events for item in counts),
        false_positive_events=sum(item.false_positive_events for item in counts),
        positive_events=sum(item.positive_events for item in counts),
        detected_target_frames=sum(item.detected_target_frames for item in counts),
        target_frames=sum(item.target_frames for item in counts),
        false_components=sum(item.false_components for item in counts),
        frame_count=sum(item.frame_count for item in counts),
    )


def metrics_from_counts_exact(counts: ChallengeCountTotals, cfg) -> ChallengeMetrics:
    """Replay counts through the official evaluator with test2's float semantics.

    Semantic tensors are compressed to TP/FN/FP events.  True negatives do not
    enter either positive-class IoU or positive-class accuracy, so this retains
    the evaluator's exact integer counts and float32 division without allocating
    all validation events for every threshold pair.
    """

    false_negative_events = counts.positive_events - counts.true_positive_events
    if false_negative_events < 0:
        raise ValueError("true positive events exceed positive events.")
    ground_truth = torch.cat(
        (
            torch.ones(counts.positive_events, dtype=torch.float32),
            torch.zeros(counts.false_positive_events, dtype=torch.float32),
        )
    )
    predictions = torch.cat(
        (
            torch.ones(counts.true_positive_events, dtype=torch.float32),
            torch.zeros(false_negative_events, dtype=torch.float32),
            torch.ones(counts.false_positive_events, dtype=torch.float32),
        )
    )
    evaluator = evalute(cfg)
    evaluator.matches["0"] = {"seg_pred": predictions, "seg_gt": ground_truth}
    evaluator.correct_num = counts.detected_target_frames
    evaluator.obj_num = counts.target_frames
    evaluator.false_num = counts.false_components
    evaluator.frame_num = counts.frame_count
    return evaluate_challenge_metrics(evaluator, prediction_threshold=0.5)


def precompute_video_counts(
    records: Sequence[RoutedRecord],
    density_cutoff: int,
    low_thresholds: Sequence[float],
    high_thresholds: Sequence[float],
    cfg,
) -> List[dict]:
    """Run each threshold/postprocessor once for each eligible video."""

    if int(density_cutoff) < 0:
        raise ValueError("density_cutoff must be non-negative.")
    prepared = []
    for index, record in enumerate(records, start=1):
        thresholds = (
            high_thresholds if record.event_count > int(density_cutoff) else low_thresholds
        )
        counts_by_threshold = {
            float(threshold): evaluate_cached_video(record, threshold, cfg)
            for threshold in thresholds
        }
        prepared.append(
            {
                "file_name": record.file_name,
                "event_count": record.event_count,
                "score_source": record.score_source,
                "counts_by_threshold": counts_by_threshold,
            }
        )
        print(
            "postprocess {}/{}: {} ({} thresholds)".format(
                index, len(records), record.file_name, len(thresholds)
            ),
            flush=True,
        )
    return prepared


def evaluate_threshold_pair(
    prepared_counts: Sequence[Mapping],
    density_cutoff: int,
    low_threshold: float,
    high_threshold: float,
    cfg,
) -> ChallengeMetrics:
    selected = []
    for item in prepared_counts:
        threshold = select_density_threshold(
            item["event_count"],
            density_cutoff,
            low_threshold,
            high_threshold,
        )
        selected.append(item["counts_by_threshold"][float(threshold)])
    return metrics_from_counts_exact(_sum_counts(selected), cfg)


def sweep_threshold_pairs(
    records: Sequence[RoutedRecord],
    density_cutoff: int,
    low_thresholds: Sequence[float],
    high_thresholds: Sequence[float],
    cfg,
) -> List[dict]:
    """Evaluate the full P6 grid after one offline postprocess pass per threshold."""

    prepared = precompute_video_counts(
        records,
        density_cutoff,
        low_thresholds,
        high_thresholds,
        cfg,
    )
    results = []
    for low_threshold, high_threshold in itertools.product(
        low_thresholds, high_thresholds
    ):
        metrics = evaluate_threshold_pair(
            prepared,
            density_cutoff,
            low_threshold,
            high_threshold,
            cfg,
        )
        result = {
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
        }
        result.update(metrics.to_dict())
        results.append(result)
    results.sort(
        key=lambda item: (
            -item["score"],
            item["low_threshold"],
            item["high_threshold"],
        )
    )
    return results


def _metric_display(name: str, value: float) -> str:
    return "{:.10e}".format(value) if name == "fa" else "{:.10f}".format(value)


def parse_expected_metrics(values: Sequence[str]) -> dict:
    expected = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Expected metric must use name=value syntax.")
        name, raw_value = value.split("=", 1)
        name = name.strip().lower()
        if name not in METRIC_NAMES:
            raise ValueError("Unknown expected metric: {}".format(name))
        expected[name] = float(raw_value)
    return expected


def verify_formatted_metrics(metrics: Mapping, expected: Mapping) -> None:
    """Require the exact precision printed by test2.py for supplied metrics."""

    mismatches = []
    for name, expected_value in expected.items():
        actual_text = _metric_display(name, float(metrics[name]))
        expected_text = _metric_display(name, float(expected_value))
        if actual_text != expected_text:
            mismatches.append("{}: {} != {}".format(name, actual_text, expected_text))
    if mismatches:
        raise RuntimeError(
            "Cached replay does not match the reference metric display: {}".format(
                "; ".join(mismatches)
            )
        )


def _postprocess_settings(cfg) -> dict:
    names = (
        "p0_enabled",
        "p0_spatial_radius",
        "p0_temporal_bin_size",
        "p0_temporal_radius_bins",
        "p0_min_cluster_events",
        "p0_min_duration_bins",
        "p0c_high_confidence_recovery_enabled",
        "p0c_retain_min_score",
        "p0b_enabled",
        "p18_score_track_recovery_enabled",
        "p18_event_count_cutoff",
        "p18_max_event_count",
        "p18_candidate_floor",
        "p18_spatial_radius",
        "p18_temporal_bin_size",
        "p18_max_link_distance",
        "p18_max_gap_bins",
        "p18_min_track_bins",
        "p18_restore_mode",
        "p18_max_restore_events_per_component",
    )
    return {name: getattr(cfg, name, None) for name in names}


def _write_json(path: Path, payload: Mapping) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, results: Sequence[Mapping]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("rank", "low_threshold", "high_threshold") + METRIC_NAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow({"rank": rank, **result})


def _common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Typed YAML override; repeat this flag for multiple values.",
    )


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("cache", help="Run one checkpoint on val.")
    _common_config_arguments(cache_parser)
    cache_parser.add_argument("--checkpoint", type=Path, required=True)
    cache_parser.add_argument("--output-cache", type=Path, required=True)
    cache_parser.add_argument("--expected-video-count", type=int, default=24)
    cache_parser.add_argument("--device", default="cuda:0")
    cache_parser.add_argument("--force", action="store_true")

    replay_parser = subparsers.add_parser(
        "replay", help="Replay event-count routing and sweep P6 thresholds on CPU."
    )
    _common_config_arguments(replay_parser)
    replay_parser.add_argument("--primary-cache", type=Path, required=True)
    replay_parser.add_argument("--secondary-cache", type=Path)
    replay_parser.add_argument("--secondary-max-events", type=int, default=0)
    replay_parser.add_argument("--density-cutoff", type=int, default=30000)
    replay_parser.add_argument("--low-min", default="0.710")
    replay_parser.add_argument("--low-max", default="0.730")
    replay_parser.add_argument("--high-min", default="0.710")
    replay_parser.add_argument("--high-max", default="0.730")
    replay_parser.add_argument("--threshold-step", default="0.001")
    replay_parser.add_argument("--output-json", type=Path, required=True)
    replay_parser.add_argument("--output-csv", type=Path, required=True)
    replay_parser.add_argument("--top", type=int, default=10)
    replay_parser.add_argument("--reference-low", type=float)
    replay_parser.add_argument("--reference-high", type=float)
    replay_parser.add_argument(
        "--expect-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Verify a reference pair at test2.py's printed precision.",
    )
    return parser.parse_args(argv)


def _find_pair(results: Sequence[Mapping], low: float, high: float) -> Mapping:
    for result in results:
        if math.isclose(result["low_threshold"], low, abs_tol=1e-12) and math.isclose(
            result["high_threshold"], high, abs_tol=1e-12
        ):
            return result
    raise ValueError(
        "Reference threshold pair ({}, {}) is outside the sweep grid.".format(low, high)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_flat_config(args.config, args.override)
    if args.command == "cache":
        output_path = args.output_cache.resolve()
        if output_path.exists() and not args.force:
            raise FileExistsError(
                "Output cache already exists; use a new path or --force: {}".format(
                    output_path
                )
            )
        payload = build_raw_cache(
            args.checkpoint,
            output_path,
            cfg,
            expected_video_count=args.expected_video_count,
            device_name=args.device,
        )
        print("raw cache:", output_path)
        print("checkpoint sha256:", payload["metadata"]["checkpoint_sha256"])
        print("dataset signature:", payload["metadata"]["dataset_signature"])
        print("videos:", payload["metadata"]["video_count"])
        print("events:", payload["metadata"]["event_count"])
        return 0

    primary = load_cache(args.primary_cache)
    secondary = load_cache(args.secondary_cache) if args.secondary_cache else None
    records = route_cache_records(primary, secondary, args.secondary_max_events)
    low_thresholds = decimal_grid(args.low_min, args.low_max, args.threshold_step)
    high_thresholds = decimal_grid(args.high_min, args.high_max, args.threshold_step)
    results = sweep_threshold_pairs(
        records,
        args.density_cutoff,
        low_thresholds,
        high_thresholds,
        cfg,
    )
    routed_secondary = sum(record.score_source == "secondary" for record in records)
    output_payload = {
        "tool_schema": "evc-temporal-memory-replay-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_cache": str(args.primary_cache.resolve()),
        "primary_checkpoint_sha256": primary["metadata"]["checkpoint_sha256"],
        "secondary_cache": str(args.secondary_cache.resolve()) if args.secondary_cache else None,
        "secondary_checkpoint_sha256": (
            secondary["metadata"]["checkpoint_sha256"] if secondary else None
        ),
        "secondary_max_events": int(args.secondary_max_events),
        "secondary_routed_videos": routed_secondary,
        "dataset_signature": primary["metadata"]["dataset_signature"],
        "density_cutoff": int(args.density_cutoff),
        "low_thresholds": list(low_thresholds),
        "high_thresholds": list(high_thresholds),
        "postprocess": _postprocess_settings(cfg),
        "config_path": str(Path(args.config).resolve()),
        "config_overrides": list(args.override),
        "results": results,
    }
    _write_json(args.output_json, output_payload)
    _write_csv(args.output_csv, results)

    print("\nTop threshold pairs")
    for rank, result in enumerate(results[: max(0, args.top)], start=1):
        print(
            "#{:02d} low={:.3f} high={:.3f} Score={:.10f} Pd={:.10f} "
            "IoU={:.10f} Acc={:.10f} Fa={:.10e}".format(
                rank,
                result["low_threshold"],
                result["high_threshold"],
                result["score"],
                result["pd"],
                result["iou"],
                result["acc"],
                result["fa"],
            )
        )

    expected = parse_expected_metrics(args.expect_metric)
    if expected:
        if args.reference_low is None or args.reference_high is None:
            raise ValueError(
                "--expect-metric requires --reference-low and --reference-high."
            )
        reference = _find_pair(results, args.reference_low, args.reference_high)
        verify_formatted_metrics(reference, expected)
        print(
            "reference pair low={:.3f} high={:.3f}: test2 display matches".format(
                args.reference_low, args.reference_high
            )
        )
    print("results json:", args.output_json.resolve())
    print("results csv:", args.output_csv.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from utils.component_reranker import (
    FEATURE_SEMANTICS_VERSION,
    load_artifact_payload,
    sha256_json,
    temporal_memory_inference_mapping,
    validate_artifact_training_provenance,
)
from utils.density_threshold import ChallengeCountTotals, select_density_threshold
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


CACHE_SCHEMA = "evc-temporal-memory-raw-probabilities-v2"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "evisseg_evuav.yaml"
METRIC_NAMES = ("iou", "acc", "pd", "fa", "score_fa", "score")
INFERENCE_SETTING_NAMES = (
    "temporal_memory_bin_size",
    "temporal_memory_context_bins",
    "temporal_memory_width",
    "temporal_memory_sequence_length",
    "temporal_memory_inference_batch_size",
    "temporal_memory_log_count_clip",
    "whole_t",
    "resolution",
)
OFFICIAL_VALIDATION_VIDEO_COUNT = 24
OFFICIAL_VALIDATION_STEMS = tuple(
    "val_{:03d}".format(index) for index in range(OFFICIAL_VALIDATION_VIDEO_COUNT)
)
CACHE_CODE_PROVENANCE_PATHS = (
    "dataset/basedataset.py",
    "dataset/ev_uav.py",
    "dataset/temporal_frame.py",
    "model/modules/confidence_head.py",
    "model/temporal_frame_net.py",
    "model/temporal_memory_net.py",
    "utils/temporal_frame_inference.py",
    "utils/temporal_memory_inference.py",
)
REPLAY_CODE_PROVENANCE_PATHS = (
    "replay_temporal_memory_validation.py",
    "utils/challenge_eval.py",
    "utils/density_threshold.py",
    "utils/eval.py",
    "utils/postprocess.py",
    "utils/component_reranker.py",
)
# Backward-compatible name for callers that imported the original constant.
CODE_PROVENANCE_PATHS = CACHE_CODE_PROVENANCE_PATHS
WARM_ROUTE_SECONDARY_MAX_EVENTS = 30000
WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH = 32
WARM_ROUTE_SECONDARY_SEQUENCE_LENGTH = 16
WARM_ROUTE_MIGRATION_NAME = (
    "temporal_memory_sequence_length_t16_to_t32_strict_state"
)
RELEASED_M20_SHA256 = (
    "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
)
RELEASED_M20_STATE_SHA256 = (
    "6600662013abae77ba397e45338850b0017b5a99121fcc88a251b67e99122b50"
)
RELEASED_M20_FROZEN_STATE_SHA256 = (
    "78e659e275d7351e040bfd65c89b63459cd6c66f4da5393918fa5f618e9c5190"
)
WARM_ROUTE_MUTABLE_STATE_KEYS = frozenset(
    {
        "temporal_attn.output_projection.weight",
        "temporal_attn.output_projection.bias",
    }
)
RELEASED_M10_SHA256 = (
    "5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935"
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
    return temporal_memory_inference_mapping(cfg)


def _code_provenance(
    project_root: Path,
    relative_paths: Sequence[str] = CACHE_CODE_PROVENANCE_PATHS,
) -> dict:
    result = {}
    for relative_path in relative_paths:
        path = project_root / relative_path
        result[relative_path] = sha256_file(path) if path.is_file() else None
    return result


def _validate_official_validation_names(
    file_names: Sequence[str],
    context: str,
) -> None:
    """Require the complete, canonical Challenge 2 validation split."""

    names = tuple(str(name) for name in file_names)
    if len(names) != OFFICIAL_VALIDATION_VIDEO_COUNT:
        raise ValueError(
            "{} must contain exactly {} official validation videos; found {}.".format(
                context, OFFICIAL_VALIDATION_VIDEO_COUNT, len(names)
            )
        )
    stems = tuple(Path(name).stem for name in names)
    if len(set(stems)) != len(stems):
        raise ValueError("{} contains duplicate validation stems.".format(context))
    expected = set(OFFICIAL_VALIDATION_STEMS)
    actual = set(stems)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        unexpected = ", ".join(sorted(actual - expected)) or "none"
        raise ValueError(
            "{} does not contain the canonical val_000..val_023 stems "
            "(missing: {}; unexpected: {}).".format(context, missing, unexpected)
        )


def _validate_expected_video_count(expected_video_count: int) -> None:
    if int(expected_video_count) != OFFICIAL_VALIDATION_VIDEO_COUNT:
        raise ValueError(
            "Complete official validation is mandatory; expected_video_count must be {}.".format(
                OFFICIAL_VALIDATION_VIDEO_COUNT
            )
        )


def _dataset_signature(records: Sequence[Mapping]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["file_name"]).encode("utf-8"))
        digest.update(str(int(record["event_count"])).encode("ascii"))
        digest.update(str(record["source_sha256"]).encode("ascii"))
    return digest.hexdigest()


def _atomic_torch_save(
    payload: dict,
    output_path: Path,
    overwrite: bool = False,
) -> None:
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
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(str(temporary_path), str(output_path))
        else:
            # Publish by exclusive hard link.  Unlike an exists-then-replace
            # sequence, this cannot clobber a concurrently created cache.
            os.link(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_raw_cache(
    checkpoint_path: Path,
    output_path: Path,
    cfg,
    expected_video_count: int = 24,
    device_name: str = "cuda:0",
    overwrite: bool = False,
) -> dict:
    """Run one checkpoint once per complete validation video and save raw scores."""

    _validate_expected_video_count(expected_video_count)

    # Imports stay inside the GPU-only operation so CPU replay and unit tests do
    # not initialize model code or require optional sparse-convolution modules.
    from dataset.ev_uav import EvUAV
    from utils.inference_chunks import evaluation_batch_from_sample
    from utils.temporal_frame_inference import temporal_frame_video_from_sample
    from utils.temporal_memory_inference import (
        load_temporal_memory_model,
        predict_temporal_memory_scores,
    )

    project_root = Path(__file__).resolve().parent
    code_provenance = _code_provenance(project_root)
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
    _validate_official_validation_names(dataset.file_list, "validation dataset")

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
    if _code_provenance(project_root) != code_provenance:
        raise RuntimeError(
            "Inference code changed while caching; refusing to save ambiguous provenance."
        )
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
        "code_sha256": code_provenance,
    }
    payload = {"metadata": metadata, "records": records}
    validate_cache_payload(payload, "new raw cache")
    _atomic_torch_save(payload, output_path, overwrite=overwrite)
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
    _validate_official_validation_names(
        [record.get("file_name", "") for record in records if isinstance(record, Mapping)],
        cache_name,
    )
    inference_settings = metadata.get("inference_settings")
    if not isinstance(inference_settings, Mapping):
        raise ValueError("{} is missing inference settings metadata.".format(cache_name))
    missing_settings = set(INFERENCE_SETTING_NAMES).difference(inference_settings)
    if missing_settings:
        raise ValueError(
            "{} is missing inference settings: {}.".format(
                cache_name, ", ".join(sorted(missing_settings))
            )
        )
    checkpoint_digest = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_digest, str) or len(checkpoint_digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in checkpoint_digest
    ):
        raise ValueError("{} has invalid checkpoint provenance.".format(cache_name))
    code_sha256 = metadata.get("code_sha256")
    if not isinstance(code_sha256, Mapping):
        raise ValueError("{} is missing inference code provenance.".format(cache_name))
    for relative_path in CACHE_CODE_PROVENANCE_PATHS:
        digest = code_sha256.get(relative_path)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ValueError(
                "{} has invalid code provenance for {}.".format(cache_name, relative_path)
            )
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
        if scores.dtype != torch.float32:
            raise ValueError("{} {} scores must have dtype float32.".format(cache_name, file_name))
        if not torch.isfinite(scores).all() or bool((scores < 0).any()) or bool(
            (scores > 1).any()
        ):
            raise ValueError("{} {} contains non-probability scores.".format(cache_name, file_name))
        if not torch.isfinite(labels).all() or bool(
            ((labels != 0) & (labels != 1)).any()
        ):
            raise ValueError("{} {} labels must be finite binary values.".format(cache_name, file_name))
        expected_digest = source_digest(locs, labels, target_ids)
        if record["source_sha256"] != expected_digest:
            raise ValueError("{} {} source digest does not match.".format(cache_name, file_name))
    if int(metadata.get("video_count", -1)) != OFFICIAL_VALIDATION_VIDEO_COUNT:
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


def load_cache_snapshot(path: Path) -> Tuple[dict, str]:
    """Load one immutable cache snapshot and return its file digest."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Raw probability cache does not exist: {}".format(path))
    digest_before = sha256_file(path)
    payload = load_cache(path)
    digest_after = sha256_file(path)
    if digest_before != digest_after:
        raise RuntimeError("Raw probability cache changed while it was being loaded: {}".format(path))
    return payload, digest_before


def _mapping_differences(left: Mapping, right: Mapping) -> Tuple[str, ...]:
    keys = sorted(set(left).union(right))
    return tuple(key for key in keys if left.get(key) != right.get(key))


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frozen_model_state_sha256(
    state: Mapping,
    mutable_state_keys: Iterable[str],
) -> str:
    """Mirror training's canonical state hash for the immutable tensors."""

    mutable_state_keys = frozenset(mutable_state_keys)
    digest = hashlib.sha256()
    for name in sorted(state):
        if name in mutable_state_keys:
            continue
        tensor = torch.as_tensor(state[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_exact_mapping(
    actual: Mapping,
    expected: Mapping,
    context: str,
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError("{} must be a mapping.".format(context))
    differences = tuple(
        name
        for name in sorted(set(actual).union(expected))
        if name not in actual
        or name not in expected
        or type(actual[name]) is not type(expected[name])
        or actual[name] != expected[name]
    )
    if differences:
        raise ValueError(
            "{} differs: {}.".format(context, ", ".join(differences))
        )


def _load_cache_checkpoint(
    payload: Mapping,
    cache_name: str,
) -> Tuple[Mapping, Path, str]:
    metadata = payload["metadata"]
    checkpoint_path_value = metadata.get("checkpoint_path")
    if not isinstance(checkpoint_path_value, str) or not checkpoint_path_value.strip():
        raise ValueError(
            "{} is missing its checkpoint path provenance.".format(cache_name)
        )
    checkpoint_path = Path(checkpoint_path_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "{} checkpoint does not exist: {}".format(cache_name, checkpoint_path)
        )
    expected_sha256 = str(metadata.get("checkpoint_sha256", "")).lower()
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "{} checkpoint SHA-256 does not match its cache metadata.".format(
                cache_name
            )
        )
    checkpoint = _torch_load_cpu(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("{} checkpoint payload must be a mapping.".format(cache_name))
    return checkpoint, checkpoint_path, actual_sha256


def _validate_checkpoint_inference_settings(
    checkpoint_memory: Mapping,
    inference_settings: Mapping,
    cache_name: str,
) -> None:
    checkpoint_to_inference = {
        "temporal_bin_size": "temporal_memory_bin_size",
        "context_bins": "temporal_memory_context_bins",
        "width": "temporal_memory_width",
        "sequence_length": "temporal_memory_sequence_length",
        "log_count_clip": "temporal_memory_log_count_clip",
    }
    differences = []
    for checkpoint_name, inference_name in checkpoint_to_inference.items():
        if checkpoint_name not in checkpoint_memory or inference_name not in inference_settings:
            differences.append(inference_name)
            continue
        checkpoint_value = checkpoint_memory[checkpoint_name]
        inference_value = inference_settings[inference_name]
        if (
            type(checkpoint_value) is not type(inference_value)
            or checkpoint_value != inference_value
        ):
            differences.append(inference_name)
    if differences:
        raise ValueError(
            "{} checkpoint metadata differs from cached inference settings: {}."
            .format(cache_name, ", ".join(differences))
        )


def _validate_warm_primary_checkpoint(
    payload: Mapping,
) -> dict:
    checkpoint, checkpoint_path, checkpoint_sha256 = _load_cache_checkpoint(
        payload, "primary cache"
    )
    if checkpoint.get("checkpoint_format_version") != 2:
        raise ValueError("Warm primary checkpoint must use format version 2.")
    required_checkpoint_fields = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "temporal_memory",
        "provenance",
    )
    missing_checkpoint_fields = tuple(
        name for name in required_checkpoint_fields if name not in checkpoint
    )
    if missing_checkpoint_fields:
        raise ValueError(
            "Warm primary checkpoint is missing fields: {}.".format(
                ", ".join(missing_checkpoint_fields)
            )
        )
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping) or len(model_state) != 89:
        raise ValueError(
            "Warm primary checkpoint must contain the complete 89-key model state."
        )
    frozen_state_sha256 = _frozen_model_state_sha256(
        model_state,
        WARM_ROUTE_MUTABLE_STATE_KEYS,
    )
    if frozen_state_sha256 != RELEASED_M20_FROZEN_STATE_SHA256:
        raise ValueError(
            "Warm primary frozen model state does not match released M20."
        )
    expected_memory = {
        "temporal_bin_size": 50,
        "context_bins": 5,
        "width": 16,
        "sequence_length": WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH,
        "log_count_clip": 4.0,
        "density_calibration_enabled": True,
        "density_calibration_version": 1,
        "trajectory_extrapolation_enabled": False,
        "confidence_head_enabled": False,
        "confidence_only_enabled": False,
        "freeze_base_enabled": False,
        "head_only_enabled": False,
        "dacc_v2_only_enabled": False,
        "attention_projection_only_enabled": True,
        "init_sequence_length_warm_start_enabled": True,
        "temporal_attention_enabled": True,
    }
    checkpoint_memory = checkpoint.get("temporal_memory")
    _require_exact_mapping(
        checkpoint_memory,
        expected_memory,
        "Warm primary temporal-memory metadata",
    )
    _validate_checkpoint_inference_settings(
        checkpoint_memory,
        payload["metadata"]["inference_settings"],
        "primary cache",
    )

    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Warm primary checkpoint provenance must be a mapping.")
    initialized_sha256 = str(
        provenance.get("initialized_from_sha256", "")
    ).lower()
    if initialized_sha256 != RELEASED_M20_SHA256:
        raise ValueError(
            "Warm primary checkpoint is not initialized from released M20."
        )
    migrations = provenance.get("initialization_migrations")
    if not isinstance(migrations, list) or len(migrations) != 1:
        raise ValueError(
            "Warm primary checkpoint must contain one sole initialization migration."
        )
    migration = migrations[0]
    if not isinstance(migration, Mapping):
        raise ValueError("Warm primary checkpoint migration must be a mapping.")
    expected_migration = {
        "name": WARM_ROUTE_MIGRATION_NAME,
        "source_sequence_length": WARM_ROUTE_SECONDARY_SEQUENCE_LENGTH,
        "target_sequence_length": WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH,
        "metadata_difference_allowlist": ["sequence_length"],
        "state_dict_strict": True,
    }
    for name, expected_value in expected_migration.items():
        if (
            name not in migration
            or type(migration[name]) is not type(expected_value)
            or migration[name] != expected_value
        ):
            raise ValueError(
                "Warm primary checkpoint migration field {} is invalid.".format(name)
            )
    migration_parent_sha256 = str(
        migration.get("parent_checkpoint_sha256", "")
    ).lower()
    if migration_parent_sha256 != RELEASED_M20_SHA256:
        raise ValueError("Warm primary migration parent is not released M20.")
    initialized_path = provenance.get("initialized_from")
    migration_parent_path = migration.get("parent_checkpoint")
    if not isinstance(initialized_path, str) or not initialized_path.strip():
        raise ValueError("Warm primary initialized-from path is missing.")
    if not isinstance(migration_parent_path, str):
        raise ValueError("Warm primary migration parent path is missing.")
    initialized_path = Path(initialized_path).expanduser().resolve()
    migration_parent_path = Path(migration_parent_path).expanduser().resolve()
    if initialized_path != migration_parent_path:
        raise ValueError("Warm primary parent checkpoint path provenance differs.")
    if not initialized_path.is_file():
        raise FileNotFoundError(
            "Warm primary released M20 parent does not exist: {}".format(
                initialized_path
            )
        )
    if sha256_file(initialized_path) != RELEASED_M20_SHA256:
        raise ValueError(
            "Warm primary parent checkpoint file is not released M20."
        )
    source_state_sha256 = str(
        migration.get("source_model_state_sha256", "")
    ).lower()
    loaded_state_sha256 = str(
        migration.get("loaded_model_state_sha256", "")
    ).lower()
    if (
        not _is_sha256(source_state_sha256)
        or source_state_sha256 != RELEASED_M20_STATE_SHA256
        or loaded_state_sha256 != RELEASED_M20_STATE_SHA256
    ):
        raise ValueError("Warm primary inherited state provenance is invalid.")

    resolved_config = provenance.get("resolved_config")
    resolved_temporal_memory = (
        resolved_config.get("TEMPORAL_MEMORY", {})
        if isinstance(resolved_config, Mapping)
        else {}
    )
    expected_resolved = {
        "temporal_memory_sequence_length": WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH,
        "temporal_memory_init_sequence_length_warm_start_enabled": True,
        "temporal_memory_attention_projection_only_enabled": True,
    }
    for name, expected_value in expected_resolved.items():
        if (
            name not in resolved_temporal_memory
            or type(resolved_temporal_memory[name]) is not type(expected_value)
            or resolved_temporal_memory[name] != expected_value
        ):
            raise ValueError(
                "Warm primary resolved configuration field {} is invalid.".format(
                    name
                )
            )
    training_scope = provenance.get("training_scope")
    expected_mutable_keys = sorted(WARM_ROUTE_MUTABLE_STATE_KEYS)
    if (
        not isinstance(training_scope, Mapping)
        or training_scope.get("name") != "temporal_attention_projection_only"
        or training_scope.get("trainable_parameter_count") != 9312
        or training_scope.get("mutable_state_keys") != expected_mutable_keys
        or str(training_scope.get("frozen_state_reference_sha256", "")).lower()
        != RELEASED_M20_FROZEN_STATE_SHA256
    ):
        raise ValueError("Warm primary training-scope provenance is invalid.")
    from utils.temporal_memory_inference import load_temporal_memory_model

    strict_model, _ = load_temporal_memory_model(
        checkpoint_path,
        torch.device("cpu"),
        context_bins=5,
        width=16,
        sequence_length=WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH,
    )
    if len(strict_model.state_dict()) != 89:
        raise ValueError("Warm primary strict model schema is not the released 89-key form.")
    del strict_model
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError(
            "Warm primary checkpoint changed during strict replay validation."
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "parent_checkpoint_sha256": initialized_sha256,
        "sequence_length": WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH,
    }


def _validate_released_m10_secondary_checkpoint(payload: Mapping) -> dict:
    checkpoint, checkpoint_path, checkpoint_sha256 = _load_cache_checkpoint(
        payload, "secondary cache"
    )
    if checkpoint_sha256 != RELEASED_M10_SHA256:
        raise ValueError("Secondary cache checkpoint is not released M10.")
    expected_memory = {
        "temporal_bin_size": 50,
        "context_bins": 5,
        "width": 16,
        "sequence_length": WARM_ROUTE_SECONDARY_SEQUENCE_LENGTH,
        "log_count_clip": 4.0,
        "density_calibration_enabled": True,
        "trajectory_extrapolation_enabled": True,
        "confidence_head_enabled": False,
        "confidence_only_enabled": False,
        "temporal_attention_enabled": False,
    }
    checkpoint_memory = checkpoint.get("temporal_memory")
    _require_exact_mapping(
        checkpoint_memory,
        expected_memory,
        "Released M10 temporal-memory metadata",
    )
    _validate_checkpoint_inference_settings(
        checkpoint_memory,
        payload["metadata"]["inference_settings"],
        "secondary cache",
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "sequence_length": WARM_ROUTE_SECONDARY_SEQUENCE_LENGTH,
    }


def _validate_warm_primary_m10_route(
    primary_payload: Mapping,
    secondary_payload: Mapping,
    secondary_max_events: int,
    runtime_inference_settings: Mapping,
) -> dict:
    if int(secondary_max_events) != WARM_ROUTE_SECONDARY_MAX_EVENTS:
        raise ValueError(
            "Warm T32/M10 replay requires secondary_max_events exactly {}."
            .format(WARM_ROUTE_SECONDARY_MAX_EVENTS)
        )
    if not isinstance(runtime_inference_settings, Mapping):
        raise ValueError(
            "Warm T32/M10 replay requires complete runtime inference settings."
        )
    primary_settings = primary_payload["metadata"]["inference_settings"]
    secondary_settings = secondary_payload["metadata"]["inference_settings"]
    runtime_differences = _mapping_differences(
        primary_settings, runtime_inference_settings
    )
    if runtime_differences:
        raise ValueError(
            "Warm primary cache inference settings differ from runtime: {}."
            .format(", ".join(runtime_differences))
        )
    setting_differences = _mapping_differences(
        primary_settings, secondary_settings
    )
    if setting_differences != ("temporal_memory_sequence_length",):
        raise ValueError(
            "Warm T32/M10 caches may differ only in temporal_memory_sequence_length; "
            "changed: {}.".format(", ".join(setting_differences) or "none")
        )
    if (
        primary_settings["temporal_memory_sequence_length"]
        != WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH
        or secondary_settings["temporal_memory_sequence_length"]
        != WARM_ROUTE_SECONDARY_SEQUENCE_LENGTH
    ):
        raise ValueError(
            "Warm route requires primary T32 and secondary M10 T16 settings."
        )
    primary_binding = _validate_warm_primary_checkpoint(primary_payload)
    secondary_binding = _validate_released_m10_secondary_checkpoint(
        secondary_payload
    )
    return {
        "enabled": True,
        "secondary_max_events": WARM_ROUTE_SECONDARY_MAX_EVENTS,
        "metadata_difference_allowlist": [
            "temporal_memory_sequence_length"
        ],
        "primary": primary_binding,
        "secondary": secondary_binding,
    }


def _validate_cache_compatibility(
    primary_payload: Mapping,
    secondary_payload: Mapping,
    secondary_max_events: int = 0,
    allow_warm_primary_t32_secondary_m10_t16: bool = False,
    runtime_inference_settings: Optional[Mapping] = None,
) -> Optional[dict]:
    primary_metadata = primary_payload["metadata"]
    secondary_metadata = secondary_payload["metadata"]
    if primary_metadata["dataset_signature"] != secondary_metadata["dataset_signature"]:
        raise ValueError("Primary and secondary cache dataset signatures differ.")

    primary_settings = primary_metadata["inference_settings"]
    secondary_settings = secondary_metadata["inference_settings"]
    setting_differences = _mapping_differences(primary_settings, secondary_settings)
    if setting_differences and not allow_warm_primary_t32_secondary_m10_t16:
        raise ValueError(
            "Primary and secondary cache inference settings differ: {}.".format(
                ", ".join(setting_differences)
            )
        )

    primary_code = primary_metadata["code_sha256"]
    secondary_code = secondary_metadata["code_sha256"]
    code_differences = (
        _mapping_differences(primary_code, secondary_code)
        if allow_warm_primary_t32_secondary_m10_t16
        else tuple(
            path
            for path in CACHE_CODE_PROVENANCE_PATHS
            if primary_code.get(path) != secondary_code.get(path)
        )
    )
    if code_differences:
        raise ValueError(
            "Primary and secondary cache inference code differs: {}.".format(
                ", ".join(code_differences)
            )
        )
    if allow_warm_primary_t32_secondary_m10_t16:
        return _validate_warm_primary_m10_route(
            primary_payload,
            secondary_payload,
            secondary_max_events,
            runtime_inference_settings,
        )
    return None


def route_cache_records(
    primary_payload: Mapping,
    secondary_payload: Optional[Mapping] = None,
    secondary_max_events: int = 0,
    allow_warm_primary_t32_secondary_m10_t16: bool = False,
    runtime_inference_settings: Optional[Mapping] = None,
) -> List[RoutedRecord]:
    """Route aligned raw scores using event count only (never file identity)."""

    validate_cache_payload(primary_payload, "primary cache")
    if int(secondary_max_events) < 0:
        raise ValueError("secondary_max_events must be non-negative.")
    if allow_warm_primary_t32_secondary_m10_t16 and secondary_payload is None:
        raise ValueError(
            "Warm T32/M10 replay requires a secondary M10 cache."
        )
    primary_records = primary_payload["records"]
    secondary_records = None
    if secondary_payload is not None:
        validate_cache_payload(secondary_payload, "secondary cache")
        _validate_cache_compatibility(
            primary_payload,
            secondary_payload,
            secondary_max_events=secondary_max_events,
            allow_warm_primary_t32_secondary_m10_t16=(
                allow_warm_primary_t32_secondary_m10_t16
            ),
            runtime_inference_settings=runtime_inference_settings,
        )
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
        if allow_warm_primary_t32_secondary_m10_t16:
            expected_secondary = (
                int(primary["event_count"])
                <= WARM_ROUTE_SECONDARY_MAX_EVENTS
            )
            if use_secondary != expected_secondary:
                raise RuntimeError(
                    "Warm replay routing violated the <=30000 M10 / >30000 T32 "
                    "boundary for {}.".format(primary["file_name"])
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


def validate_component_reranker_cache_binding(
    cfg,
    primary_payload: Mapping,
    secondary_payload: Optional[Mapping] = None,
    secondary_max_events: int = 0,
) -> Optional[dict]:
    """Fail closed when replay scores are not the artifact's bound M20 scores.

    Runtime inference validates the configured checkpoint on disk.  Replay has
    an additional trust boundary: cached probabilities may have been produced
    by a different checkpoint or inference configuration.  This guard binds
    the *actual primary cache metadata* to the train-only artifact before any
    labels are evaluated.
    """
    if not bool(getattr(cfg, "component_reranker_enabled", False)):
        return None

    validate_cache_payload(primary_payload, "primary cache")
    if secondary_payload is not None:
        validate_cache_payload(secondary_payload, "secondary cache")
        _validate_cache_compatibility(primary_payload, secondary_payload)

    artifact_path = str(getattr(cfg, "component_reranker_model_path", ""))
    expected_artifact_sha256 = str(
        getattr(cfg, "component_reranker_expected_sha256", "")
    )
    artifact, artifact_sha256 = load_artifact_payload(
        artifact_path, expected_artifact_sha256
    )
    if artifact.get("feature_semantics_version") != FEATURE_SEMANTICS_VERSION:
        raise ValueError(
            "Component reranker feature semantics version does not match replay."
        )
    provenance = artifact.get("provenance")
    provenance = validate_artifact_training_provenance(provenance)
    base_checkpoint_sha256 = str(
        provenance.get("base_checkpoint_sha256", "")
    ).lower()
    if len(base_checkpoint_sha256) != 64:
        raise ValueError(
            "Component reranker artifact has invalid base checkpoint provenance."
        )
    primary_checkpoint_sha256 = str(
        primary_payload["metadata"].get("checkpoint_sha256", "")
    ).lower()
    if primary_checkpoint_sha256 != base_checkpoint_sha256:
        raise ValueError(
            "Primary replay cache checkpoint SHA-256 {} does not match component "
            "reranker base {}.".format(
                primary_checkpoint_sha256, base_checkpoint_sha256
            )
        )

    configured_checkpoint_path = Path(
        str(getattr(cfg, "temporal_memory_model_path", ""))
    ).expanduser().resolve()
    if not configured_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Configured M20 checkpoint does not exist: {}".format(
                configured_checkpoint_path
            )
        )
    configured_checkpoint_sha256 = sha256_file(configured_checkpoint_path)
    if configured_checkpoint_sha256 != base_checkpoint_sha256:
        raise ValueError(
            "Configured M20 checkpoint SHA-256 {} does not match component "
            "reranker base {}.".format(
                configured_checkpoint_sha256, base_checkpoint_sha256
            )
        )

    artifact_inference_settings = provenance.get("inference_settings")
    if not isinstance(artifact_inference_settings, Mapping):
        raise ValueError(
            "Component reranker artifact is missing inference_settings provenance."
        )
    artifact_inference_settings = dict(artifact_inference_settings)
    inference_settings_sha256 = str(
        provenance.get("inference_settings_sha256", "")
    ).lower()
    if inference_settings_sha256 != sha256_json(artifact_inference_settings):
        raise ValueError(
            "Component reranker artifact inference settings signature is invalid."
        )
    runtime_inference_settings = _inference_settings(cfg)
    runtime_differences = _mapping_differences(
        artifact_inference_settings, runtime_inference_settings
    )
    if runtime_differences:
        raise ValueError(
            "Runtime inference settings differ from component reranker artifact: {}."
            .format(", ".join(runtime_differences))
        )
    primary_inference_settings = dict(
        primary_payload["metadata"]["inference_settings"]
    )
    cache_differences = _mapping_differences(
        artifact_inference_settings, primary_inference_settings
    )
    if cache_differences:
        raise ValueError(
            "Primary replay cache inference settings differ from component "
            "reranker artifact: {}.".format(", ".join(cache_differences))
        )

    cutoff = int(getattr(cfg, "component_reranker_event_count_cutoff", 100000))
    trained_cutoff = int(provenance.get("deployment_event_count_cutoff", -1))
    if cutoff != trained_cutoff:
        raise ValueError(
            "Component reranker replay cutoff {} does not match artifact {}."
            .format(cutoff, trained_cutoff)
        )
    secondary_max_events = int(secondary_max_events)
    if secondary_max_events < 0 or secondary_max_events > cutoff:
        raise ValueError(
            "Replay secondary routing must stay entirely at or below component "
            "reranker cutoff {}.".format(cutoff)
        )

    return {
        "artifact_sha256": artifact_sha256,
        "artifact_base_checkpoint_sha256": base_checkpoint_sha256,
        "configured_checkpoint_sha256": configured_checkpoint_sha256,
        "primary_cache_checkpoint_sha256": primary_checkpoint_sha256,
        "inference_settings": artifact_inference_settings,
        "inference_settings_sha256": inference_settings_sha256,
        "event_count_cutoff": cutoff,
        "secondary_max_events": secondary_max_events,
    }


def validate_component_reranker_dense_routes(
    cfg,
    records: Sequence[RoutedRecord],
) -> None:
    """Ensure every reranker-eligible cached video came from primary M20."""
    if not bool(getattr(cfg, "component_reranker_enabled", False)):
        return
    cutoff = int(getattr(cfg, "component_reranker_event_count_cutoff", 100000))
    invalid = [
        record.file_name
        for record in records
        if int(record.event_count) > cutoff and record.score_source != "primary"
    ]
    if invalid:
        raise ValueError(
            "Component reranker-eligible replay records must come from primary "
            "M20 cache; secondary routed: {}.".format(", ".join(invalid))
        )


def evaluate_cached_video(record: RoutedRecord, threshold: float, cfg) -> ChallengeCountTotals:
    """Return exact per-video sufficient counts after real project postprocessing."""

    threshold = float(threshold)
    postprocessor = ChallengePostprocessor.from_cfg(
        cfg,
        threshold,
        event_count=record.event_count,
    )
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
        "p0c_density_retain_enabled",
        "p0c_density_event_count_cutoff",
        "p0c_density_retain_min_score",
        "component_reranker_enabled",
        "component_reranker_event_count_cutoff",
        "component_reranker_model_path",
        "component_reranker_expected_sha256",
        "p0b_enabled",
        "p0b_spatial_radius",
        "p0b_temporal_bin_size",
        "p0b_max_link_distance",
        "p0b_max_gap_bins",
        "p0b_min_track_events",
        "p0b_min_track_frames",
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
    settings = {name: getattr(cfg, name, None) for name in names}
    artifact_path = str(
        getattr(cfg, "component_reranker_model_path", "") or ""
    ).strip()
    settings["component_reranker_actual_sha256"] = (
        sha256_file(Path(artifact_path).expanduser().resolve())
        if bool(getattr(cfg, "component_reranker_enabled", False))
        and artifact_path
        and Path(artifact_path).expanduser().is_file()
        else None
    )
    return settings


def _evaluation_settings(cfg) -> dict:
    return {
        "pd_detT": int(cfg.pd_detT),
        "correct_thresh": float(cfg.correct_thresh),
        "resolution": [int(cfg.res[0]), int(cfg.res[1])],
    }


def _atomic_text_write(path: Path, write_callback) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            write_callback(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, payload: Mapping) -> None:
    def write(handle):
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    _atomic_text_write(path, write)


def _write_csv(path: Path, results: Sequence[Mapping]) -> None:
    fields = ("rank", "low_threshold", "high_threshold") + METRIC_NAMES

    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow({"rank": rank, **result})

    _atomic_text_write(path, write)


def _paths_alias(left: Path, right: Path) -> bool:
    left = Path(left).resolve()
    right = Path(right).resolve()
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            return False
    return False


def _require_distinct_paths(named_paths: Sequence[Tuple[str, Optional[Path]]]) -> None:
    present = [(name, Path(path).resolve()) for name, path in named_paths if path is not None]
    for index, (left_name, left_path) in enumerate(present):
        for right_name, right_path in present[index + 1 :]:
            if _paths_alias(left_path, right_path):
                raise ValueError(
                    "Path conflict: {} and {} resolve to the same file: {}.".format(
                        left_name, right_name, left_path
                    )
                )


def _require_outputs_available(
    named_paths: Sequence[Tuple[str, Path]],
    force: bool,
) -> None:
    if force:
        return
    existing = [
        "{}={}".format(name, Path(path).resolve())
        for name, path in named_paths
        if Path(path).resolve().exists()
    ]
    if existing:
        raise FileExistsError(
            "Output already exists; choose a new path or use --force: {}.".format(
                ", ".join(existing)
            )
        )


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
    cache_parser.add_argument(
        "--expected-video-count",
        type=int,
        choices=(OFFICIAL_VALIDATION_VIDEO_COUNT,),
        default=OFFICIAL_VALIDATION_VIDEO_COUNT,
        help="Safety assertion; complete official validation is always required.",
    )
    cache_parser.add_argument("--device", default="cuda:0")
    cache_parser.add_argument("--force", action="store_true")

    replay_parser = subparsers.add_parser(
        "replay", help="Replay event-count routing and sweep P6 thresholds on CPU."
    )
    _common_config_arguments(replay_parser)
    replay_parser.add_argument("--primary-cache", type=Path, required=True)
    replay_parser.add_argument("--secondary-cache", type=Path)
    replay_parser.add_argument("--secondary-max-events", type=int, default=0)
    replay_parser.add_argument(
        "--allow-warm-primary-t32-secondary-m10-t16",
        action="store_true",
        help=(
            "Explicitly allow only the audited warm-lineage T32 primary with "
            "released M10 T16 at the fixed <=30000 route boundary."
        ),
    )
    replay_parser.add_argument("--density-cutoff", type=int, default=30000)
    replay_parser.add_argument("--low-min", default="0.710")
    replay_parser.add_argument("--low-max", default="0.730")
    replay_parser.add_argument("--high-min", default="0.710")
    replay_parser.add_argument("--high-max", default="0.730")
    replay_parser.add_argument("--threshold-step", default="0.001")
    replay_parser.add_argument("--output-json", type=Path, required=True)
    replay_parser.add_argument("--output-csv", type=Path, required=True)
    replay_parser.add_argument("--force", action="store_true")
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
        _validate_expected_video_count(args.expected_video_count)
        _require_distinct_paths(
            (
                ("config", args.config),
                ("checkpoint", args.checkpoint),
                ("output-cache", output_path),
            )
        )
        _require_outputs_available((("output-cache", output_path),), args.force)
        payload = build_raw_cache(
            args.checkpoint,
            output_path,
            cfg,
            expected_video_count=args.expected_video_count,
            device_name=args.device,
            overwrite=args.force,
        )
        print("raw cache:", output_path)
        print("checkpoint sha256:", payload["metadata"]["checkpoint_sha256"])
        print("dataset signature:", payload["metadata"]["dataset_signature"])
        print("videos:", payload["metadata"]["video_count"])
        print("events:", payload["metadata"]["event_count"])
        return 0

    _require_distinct_paths(
        (
            ("config", args.config),
            ("primary-cache", args.primary_cache),
            ("secondary-cache", args.secondary_cache),
            ("output-json", args.output_json),
            ("output-csv", args.output_csv),
        )
    )
    _require_outputs_available(
        (("output-json", args.output_json), ("output-csv", args.output_csv)),
        args.force,
    )
    expected = parse_expected_metrics(args.expect_metric)
    if expected and (args.reference_low is None or args.reference_high is None):
        raise ValueError(
            "--expect-metric requires --reference-low and --reference-high."
        )

    primary, primary_cache_sha256 = load_cache_snapshot(args.primary_cache)
    if args.secondary_cache:
        secondary, secondary_cache_sha256 = load_cache_snapshot(args.secondary_cache)
    else:
        secondary = None
        secondary_cache_sha256 = None
    checkpoint_paths = []
    for cache_name, payload in (("primary", primary), ("secondary", secondary)):
        if payload is None:
            continue
        checkpoint_path = payload["metadata"].get("checkpoint_path")
        if checkpoint_path:
            checkpoint_paths.append((cache_name + "-checkpoint", Path(checkpoint_path)))
    _require_distinct_paths(
        tuple(checkpoint_paths)
        + (("output-json", args.output_json), ("output-csv", args.output_csv))
    )
    reranker_cache_binding = validate_component_reranker_cache_binding(
        cfg,
        primary,
        secondary,
        args.secondary_max_events,
    )
    records = route_cache_records(
        primary,
        secondary,
        args.secondary_max_events,
        allow_warm_primary_t32_secondary_m10_t16=(
            args.allow_warm_primary_t32_secondary_m10_t16
        ),
        runtime_inference_settings=_inference_settings(cfg),
    )
    validate_component_reranker_dense_routes(cfg, records)
    low_thresholds = decimal_grid(args.low_min, args.low_max, args.threshold_step)
    high_thresholds = decimal_grid(args.high_min, args.high_max, args.threshold_step)
    results = sweep_threshold_pairs(
        records,
        args.density_cutoff,
        low_thresholds,
        high_thresholds,
        cfg,
    )
    reference = None
    if expected:
        reference = _find_pair(results, args.reference_low, args.reference_high)
        # A failed golden-baseline check must not leave files that look like a
        # successful replay.  Verify before either output is created/replaced.
        verify_formatted_metrics(reference, expected)

    routed_secondary = sum(record.score_source == "secondary" for record in records)
    project_root = Path(__file__).resolve().parent
    output_payload = {
        "tool_schema": "evc-temporal-memory-replay-results-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_cache": str(args.primary_cache.resolve()),
        "primary_cache_sha256": primary_cache_sha256,
        "primary_checkpoint_sha256": primary["metadata"]["checkpoint_sha256"],
        "secondary_cache": str(args.secondary_cache.resolve()) if args.secondary_cache else None,
        "secondary_cache_sha256": secondary_cache_sha256,
        "secondary_checkpoint_sha256": (
            secondary["metadata"]["checkpoint_sha256"] if secondary else None
        ),
        "secondary_max_events": int(args.secondary_max_events),
        "secondary_routed_videos": routed_secondary,
        "dataset_split": "val",
        "dataset_signature": primary["metadata"]["dataset_signature"],
        "video_count": len(records),
        "event_count": sum(record.event_count for record in records),
        "inference_settings": dict(primary["metadata"]["inference_settings"]),
        "density_cutoff": int(args.density_cutoff),
        "low_thresholds": list(low_thresholds),
        "high_thresholds": list(high_thresholds),
        "postprocess": _postprocess_settings(cfg),
        "component_reranker_cache_binding": reranker_cache_binding,
        "evaluation": _evaluation_settings(cfg),
        "replay_code_sha256": _code_provenance(
            project_root, REPLAY_CODE_PROVENANCE_PATHS
        ),
        "config_path": str(Path(args.config).resolve()),
        "config_overrides": list(args.override),
        "reference_verification": (
            {
                "low_threshold": float(args.reference_low),
                "high_threshold": float(args.reference_high),
                "expected": expected,
                "matched_test2_display": True,
            }
            if expected
            else None
        ),
        "selection_note": (
            "Threshold pairs are ranked on the same labeled validation split; "
            "the best row is validation-tuned and is not an independent estimate."
        ),
        "results": results,
    }
    if args.allow_warm_primary_t32_secondary_m10_t16:
        output_payload["warm_primary_t32_secondary_m10_t16_route"] = {
            "enabled": True,
            "secondary_max_events": WARM_ROUTE_SECONDARY_MAX_EVENTS,
            "metadata_difference_allowlist": [
                "temporal_memory_sequence_length"
            ],
            "primary_sequence_length": WARM_ROUTE_PRIMARY_SEQUENCE_LENGTH,
            "secondary_sequence_length": WARM_ROUTE_SECONDARY_SEQUENCE_LENGTH,
            "released_m20_parent_sha256": RELEASED_M20_SHA256,
            "released_m10_sha256": RELEASED_M10_SHA256,
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
    print(
        "selection note: grid ranking uses labeled val and is not an independent score"
    )

    if expected:
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

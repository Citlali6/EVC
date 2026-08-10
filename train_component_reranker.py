"""Build a train-only cache and fit the dense component reranker.

This utility has two explicit phases:

``cache``
    Run one temporal-memory checkpoint on complete official *training* videos
    selected only by a configurable event-count rule.  No validation path is
    accepted.  Compact per-video ``.npz`` records and a strict JSON manifest
    preserve the scores, labels, locations, and provenance.

``fit``
    Apply the configured P0/P0c stage, derive label-free component/short-track
    features, and fit a deterministic weighted logistic model.  Labels are
    used only to construct train-time component targets.  Hyperparameters are
    explicit command-line inputs; this script performs no validation sweep or
    automatic cross-validation selection.

The output model is strict JSON rather than pickle.  Runtime loading also
requires the artifact SHA-256 and checks that it is bound to the active M20
primary checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

import replay_temporal_memory_validation as replay
from utils.component_reranker import (
    ARTIFACT_SCHEMA,
    FEATURE_SEMANTICS_VERSION,
    FEATURE_NAMES,
    TRAIN_CACHE_SCHEMA,
    ComponentTopology,
    extract_component_examples,
    input_postprocess_mapping,
    sha256_file,
    sha256_json,
    temporal_memory_inference_mapping,
)


CACHE_SCHEMA = TRAIN_CACHE_SCHEMA
TRAIN_SOURCE_MANIFEST_SCHEME = (
    "sha256_name_utf8_nul_raw_file_sha256_bytes_canonical_order_v1"
)
OFFICIAL_TRAIN_SOURCE_MANIFEST_SHA256 = (
    "e94aaeae451113943a464feec7b1500968601a835ce8eeee914129ed2456625f"
)
CACHE_CODE_PATHS = (
    "train_component_reranker.py",
    "dataset/temporal_frame.py",
    "model/modules/confidence_head.py",
    "model/temporal_frame_net.py",
    "model/temporal_memory_net.py",
    "utils/component_reranker.py",
    "utils/temporal_frame_inference.py",
    "utils/temporal_memory_inference.py",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _require_new_output(path, kind):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("{} already exists: {}".format(kind, path))
    return path


def _code_sha256(project_root, paths):
    return {
        relative: sha256_file(project_root / relative)
        for relative in paths
        if (project_root / relative).is_file()
    }


def source_manifest_sha256(entries):
    """Hash canonical ``(source name, raw file SHA)`` entries."""
    digest = hashlib.sha256()
    expected_name = 0
    seen_source_sha256 = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Train source manifest entries must be JSON objects.")
        name = str(entry.get("source_name", ""))
        if name != "train_{:03d}.npz".format(expected_name):
            raise ValueError(
                "Train source manifest is not canonical at index {}: {!r}.".format(
                    expected_name, name
                )
            )
        raw_sha256 = str(entry.get("source_sha256", "")).lower()
        if len(raw_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in raw_sha256
        ):
            raise ValueError("Train source manifest contains an invalid raw SHA-256.")
        if raw_sha256 in seen_source_sha256:
            raise ValueError("Official train raw source SHA-256 values must be unique.")
        seen_source_sha256.add(raw_sha256)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(raw_sha256))
        expected_name += 1
    if expected_name != 99:
        raise ValueError("Official train source manifest must contain exactly 99 files.")
    return digest.hexdigest()


def hash_train_sources(
    source_paths,
    expected_sha256=OFFICIAL_TRAIN_SOURCE_MANIFEST_SHA256,
):
    """Hash all 99 raw train files and require the official semantic digest.

    ``expected_sha256`` is injectable for isolated synthetic unit tests.  The
    cache CLI never exposes it and always uses the official constant.
    """
    paths = [Path(path).resolve() for path in source_paths]
    entries = [
        {
            "source_name": path.name,
            "source_sha256": sha256_file(path),
        }
        for path in paths
    ]
    actual_sha256 = source_manifest_sha256(entries)
    expected_sha256 = str(expected_sha256).strip().lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Expected train source manifest SHA-256 is invalid.")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Official train source semantic SHA-256 {} does not match required {}."
            .format(actual_sha256, expected_sha256)
        )
    return actual_sha256, entries


def _compact_integer_array(value, minimum_dtype, name):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.integer):
        if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
            raise ValueError("{} must contain finite integers.".format(name))
    minimum = int(value.min()) if value.size else 0
    maximum = int(value.max()) if value.size else 0
    for dtype in (minimum_dtype, np.int32, np.int64):
        info = np.iinfo(dtype)
        if info.min <= minimum and maximum <= info.max:
            return np.ascontiguousarray(value, dtype=dtype)
    raise ValueError("{} values exceed int64 range.".format(name))


def _load_train_source(path):
    with np.load(path, allow_pickle=False) as source:
        required = {"ev_loc", "evs_norm"}
        if not required.issubset(source.files):
            raise ValueError(
                "Training source {} is missing {}.".format(
                    path, sorted(required.difference(source.files))
                )
            )
        locations = np.ascontiguousarray(source["ev_loc"])
        normalized = np.ascontiguousarray(source["evs_norm"])
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("{} ev_loc must have shape [N, 3].".format(path.name))
    if normalized.ndim != 2 or normalized.shape[1] < 6:
        raise ValueError("{} evs_norm must have at least six columns.".format(path.name))
    if normalized.shape[0] != locations.shape[0]:
        raise ValueError("{} ev_loc/evs_norm lengths differ.".format(path.name))
    labels = np.ascontiguousarray(normalized[:, 4] > 0.5, dtype=np.uint8)
    target_ids = _compact_integer_array(normalized[:, 5], np.int16, "target_ids")
    sample = {
        "file_name": path.name,
        "ev_loc": locations,
        "evs_norm": np.ascontiguousarray(normalized[:, :4], dtype=np.float32),
    }
    return sample, labels, target_ids


def build_train_cache(args):
    import torch
    from utils.temporal_frame_inference import temporal_frame_video_from_sample
    from utils.temporal_memory_inference import (
        load_temporal_memory_model,
        predict_temporal_memory_scores,
    )

    project_root = Path(__file__).resolve().parent
    output_dir = _require_new_output(args.output_cache_dir, "Output cache directory")
    data_root = Path(args.data_root).resolve()
    train_dir = data_root / "train"
    checkpoint_path = Path(args.checkpoint).resolve()
    if not train_dir.is_dir():
        raise NotADirectoryError("Official train directory does not exist: {}".format(train_dir))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Base checkpoint does not exist: {}".format(checkpoint_path))
    source_paths = sorted(train_dir.glob("*.npz"))
    if args.expected_total_videos and len(source_paths) != args.expected_total_videos:
        raise ValueError(
            "Expected {} train videos, found {}.".format(
                args.expected_total_videos, len(source_paths)
            )
        )
    for index, path in enumerate(source_paths):
        expected_name = "train_{:03d}.npz".format(index)
        if args.require_canonical_names and path.name != expected_name:
            raise ValueError(
                "Canonical train identity mismatch: expected {}, found {}.".format(
                    expected_name, path.name
                )
            )

    # This gate runs before the output directory is created or any model/GPU
    # work begins.  Canonical names alone are insufficient: bind all 99 raw
    # official train files to the independently verified semantic manifest.
    train_source_manifest_sha256, train_source_entries = hash_train_sources(
        source_paths
    )
    source_sha256_by_name = {
        entry["source_name"]: entry["source_sha256"]
        for entry in train_source_entries
    }

    selected = []
    for path in source_paths:
        with np.load(path, allow_pickle=False) as source:
            event_count = int(len(source["ev_loc"]))
        if event_count > args.min_event_count_exclusive:
            selected.append((path, event_count))
    if not selected:
        raise ValueError("No train video passed the event-count selection rule.")
    if args.expected_selected_videos and len(selected) != args.expected_selected_videos:
        raise ValueError(
            "Expected {} selected train videos, found {}.".format(
                args.expected_selected_videos, len(selected)
            )
        )

    cfg = replay.load_flat_config(args.config, args.override)
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError("temporal_memory_context_bins must be odd.")
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError("temporal_memory_sequence_length must exceed one.")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the requested cache device.")

    output_dir.mkdir(parents=True, exist_ok=False)
    records_dir = output_dir / "records"
    records_dir.mkdir()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    config_path = Path(args.config).resolve()
    config_sha256 = sha256_file(config_path)
    code_before = _code_sha256(project_root, CACHE_CODE_PATHS)
    device = torch.device(args.device)
    model, checkpoint = load_temporal_memory_model(
        str(checkpoint_path),
        device,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        cfg.temporal_memory_sequence_length,
    )
    records = []
    started = datetime.now(timezone.utc)
    for record_index, (source_path, expected_event_count) in enumerate(selected, start=1):
        source_sha256 = source_sha256_by_name[source_path.name]
        sample, labels, target_ids = _load_train_source(source_path)
        if len(sample["ev_loc"]) != expected_event_count:
            raise RuntimeError("Training source changed while scanning: {}".format(source_path))
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
        ).reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
        if scores.size != expected_event_count:
            raise RuntimeError(
                "Prediction/source length mismatch for {}.".format(source_path.name)
            )
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise RuntimeError("Non-probability score found for {}.".format(source_path.name))
        if sha256_file(source_path) != source_sha256:
            raise RuntimeError(
                "Training source changed during inference: {}".format(source_path)
            )
        compact_locations = _compact_integer_array(
            sample["ev_loc"], np.int16, "ev_loc"
        )
        record_path = records_dir / "{:03d}.npz".format(record_index - 1)
        _atomic_npz(
            record_path,
            scores=np.ascontiguousarray(scores),
            locs=compact_locations,
            labels=labels,
            target_ids=target_ids,
        )
        records.append(
            {
                "record": record_path.relative_to(output_dir).as_posix(),
                "record_sha256": sha256_file(record_path),
                "source_name": source_path.name,
                "source_sha256": source_sha256,
                "event_count": expected_event_count,
                "positive_event_count": int(labels.sum()),
            }
        )
        print(
            "cache {}/{}: {} ({} events)".format(
                record_index, len(selected), source_path.name, expected_event_count
            ),
            flush=True,
        )
        del sample, labels, target_ids, frame_video, scores, compact_locations
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("Base checkpoint changed while train cache was generated.")
    if sha256_file(config_path) != config_sha256:
        raise RuntimeError("Config changed while train cache was generated.")
    if _code_sha256(project_root, CACHE_CODE_PATHS) != code_before:
        raise RuntimeError("Cache code changed while train cache was generated.")
    final_source_manifest_sha256, final_source_entries = hash_train_sources(
        source_paths
    )
    if (
        final_source_manifest_sha256 != train_source_manifest_sha256
        or final_source_entries != train_source_entries
    ):
        raise RuntimeError("Official train sources changed during cache generation.")
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": CACHE_SCHEMA,
        "created_utc": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "dataset_split": "train",
        "dataset_root": str(data_root),
        "selection": {
            "observable": "complete_video_event_count",
            "operator": ">",
            "min_event_count_exclusive": int(args.min_event_count_exclusive),
        },
        "total_train_video_count": len(source_paths),
        "official_train_source_manifest_scheme": TRAIN_SOURCE_MANIFEST_SCHEME,
        "official_train_source_manifest_sha256": train_source_manifest_sha256,
        "official_train_sources": train_source_entries,
        "selected_video_count": len(records),
        "selected_event_count": sum(record["event_count"] for record in records),
        "base_checkpoint_path": str(checkpoint_path),
        "base_checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint_epoch": checkpoint.get("epoch"),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "config_overrides": list(args.override),
        "inference_settings": temporal_memory_inference_mapping(cfg),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "code_sha256": code_before,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    print("wrote train cache:", output_dir)
    print("manifest sha256:", sha256_file(manifest_path))
    return 0


def load_train_cache(cache_dir):
    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Train-cache manifest does not exist: {}".format(manifest_path))
    manifest_sha256 = sha256_file(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("Unsupported train-cache schema.")
    if manifest.get("dataset_split") != "train":
        raise ValueError("Component reranker fitting accepts only dataset_split=train.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Train-cache manifest contains no records.")
    if int(manifest.get("selected_video_count", -1)) != len(records):
        raise ValueError("Train-cache selected_video_count does not match records.")
    return cache_dir, manifest_path, manifest_sha256, manifest


def _load_cache_record(cache_dir, metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Train-cache record metadata must be a JSON object.")
    relative = Path(str(metadata.get("record", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Train-cache record path must stay inside cache directory.")
    path = (cache_dir / relative).resolve()
    try:
        path.relative_to(cache_dir)
    except ValueError as exc:
        raise ValueError("Train-cache record escaped cache directory.") from exc
    if not path.is_file():
        raise FileNotFoundError("Train-cache record does not exist: {}".format(path))
    if sha256_file(path) != str(metadata.get("record_sha256", "")).lower():
        raise ValueError("Train-cache record SHA-256 mismatch: {}".format(path))
    with np.load(path, allow_pickle=False) as record:
        required = {"scores", "locs", "labels", "target_ids"}
        if set(record.files) != required:
            raise ValueError("Train-cache record fields differ: {}".format(path))
        values = {name: np.ascontiguousarray(record[name]) for name in required}
    event_count = int(metadata.get("event_count", -1))
    if not (
        values["scores"].reshape(-1).size
        == values["locs"].shape[0]
        == values["labels"].reshape(-1).size
        == values["target_ids"].reshape(-1).size
        == event_count
    ):
        raise ValueError("Train-cache record lengths differ: {}".format(path))
    return values


def _weighted_logistic_loss(design, labels, sample_weights, parameters, l2):
    logits = design @ parameters
    losses = np.logaddexp(0.0, logits) - labels * logits
    return float(
        np.dot(sample_weights, losses) / sample_weights.sum()
        + 0.5 * l2 * np.dot(parameters[:-1], parameters[:-1])
    )


def fit_weighted_logistic(features, labels, positive_weight, l2, max_iterations):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("features/labels have incompatible shapes.")
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Unexpected component feature width.")
    if not np.isin(labels, (0.0, 1.0)).all() or np.unique(labels).size != 2:
        raise ValueError("Component training labels must contain both binary classes.")
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive.")
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("l2 must be finite and positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack((standardized, np.ones(labels.size, dtype=np.float64)))
    sample_weights = np.where(labels > 0.5, positive_weight, 1.0)
    weighted_positives = float(np.dot(sample_weights, labels))
    weighted_negatives = float(np.dot(sample_weights, 1.0 - labels))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[-1] = math.log(
        max(weighted_positives, 1e-12) / max(weighted_negatives, 1e-12)
    )
    converged = False
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        logits = design @ parameters
        probabilities = np.empty_like(logits)
        nonnegative = logits >= 0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exp_negative = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = exp_negative / (1.0 + exp_negative)
        normalization = sample_weights.sum()
        gradient = design.T @ (sample_weights * (probabilities - labels)) / normalization
        gradient[:-1] += l2 * parameters[:-1]
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / normalization
        hessian[:-1, :-1] += np.eye(features.shape[1]) * l2
        hessian[-1, -1] += 1e-12
        try:
            newton_step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.max(np.abs(newton_step)) < 1e-9:
            converged = True
            break
        current_loss = _weighted_logistic_loss(
            design, labels, sample_weights, parameters, l2
        )
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0 ** -20:
            candidate = parameters - step_scale * newton_step
            candidate_loss = _weighted_logistic_loss(
                design, labels, sample_weights, candidate, l2
            )
            if candidate_loss <= current_loss:
                parameters = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if abs(current_loss - candidate_loss) < 1e-12:
            converged = True
            break
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficients": parameters[:-1],
        "intercept": float(parameters[-1]),
        "iterations": iterations,
        "converged": converged,
        "weighted_loss": _weighted_logistic_loss(
            design, labels, sample_weights, parameters, l2
        ),
    }


def _probabilities(features, fitted):
    standardized = (
        np.asarray(features, dtype=np.float64) - fitted["feature_mean"]
    ) / fitted["feature_scale"]
    logits = standardized @ fitted["coefficients"] + fitted["intercept"]
    probabilities = np.empty_like(logits)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    negative_exp = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = negative_exp / (1.0 + negative_exp)
    return probabilities


def fit_artifact(args):
    import torch
    from utils.postprocess import P0ClusterFilter

    output_path = _require_new_output(args.output_model, "Output model artifact")
    cache_dir, manifest_path, manifest_sha256, manifest = load_train_cache(
        args.cache_dir
    )
    cfg = replay.load_flat_config(args.config, args.override)
    if bool(getattr(cfg, "component_reranker_enabled", False)):
        raise ValueError("Disable component_reranker while fitting its artifact.")
    if not bool(getattr(cfg, "p0_enabled", False)):
        raise ValueError("Artifact fitting requires the upstream P0 stage enabled.")
    if bool(getattr(cfg, "p0b_enabled", False)):
        raise ValueError("Artifact fitting supports P0/P0c, not P0b.")
    cache_inference_settings = manifest.get("inference_settings")
    runtime_inference_settings = temporal_memory_inference_mapping(cfg)
    if cache_inference_settings != runtime_inference_settings:
        raise ValueError(
            "Fit config inference settings do not match the train-cache manifest."
        )
    prediction_threshold = float(args.prediction_threshold)
    if not 0.0 < prediction_threshold < 1.0:
        raise ValueError("prediction_threshold must lie strictly inside (0, 1).")
    topology = ComponentTopology(
        spatial_radius=args.spatial_radius,
        temporal_bin_size=args.temporal_bin_size,
        max_link_distance=args.max_link_distance,
        max_gap_bins=args.max_gap_bins,
        max_component_events=args.max_component_events,
    )

    feature_batches = []
    label_batches = []
    per_video = []
    for metadata in manifest["records"]:
        record = _load_cache_record(cache_dir, metadata)
        scores = torch.from_numpy(record["scores"].reshape(-1).astype(np.float32))
        locations = np.column_stack(
            (
                np.zeros(scores.numel(), dtype=np.int64),
                record["locs"].astype(np.int64, copy=False),
            )
        )
        locations_tensor = torch.from_numpy(locations).to(torch.int64).contiguous()
        event_count = int(metadata["event_count"])
        p0_filter = P0ClusterFilter.from_cfg(
            cfg,
            prediction_threshold,
            event_count=event_count,
        )
        p0_scores, p0_stats = p0_filter.apply(scores, locations_tensor)
        examples = extract_component_examples(
            p0_scores.numpy(),
            locations,
            prediction_threshold,
            topology,
            event_count,
            labels=record["labels"],
        )
        if examples:
            feature_batches.append(np.stack([item.features for item in examples]))
            label_batches.append(np.asarray([item.label for item in examples], dtype=np.uint8))
        per_video.append(
            {
                "source_name": metadata["source_name"],
                "event_count": event_count,
                "candidate_components": len(examples),
                "positive_components": int(sum(item.label for item in examples)),
                "p0_input_positive_events": p0_stats.input_positive_events,
                "p0_output_positive_events": p0_stats.output_positive_events,
            }
        )
        print(
            "features {}/{}: {} -> {} candidates".format(
                len(per_video),
                len(manifest["records"]),
                metadata["source_name"],
                len(examples),
            ),
            flush=True,
        )
    if not feature_batches:
        raise RuntimeError("P0/P0c produced no component reranker candidates.")
    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    fitted = fit_weighted_logistic(
        features,
        labels,
        positive_weight=float(args.positive_weight),
        l2=float(args.l2),
        max_iterations=int(args.max_iterations),
    )
    probabilities = _probabilities(features, fitted)
    keep = probabilities >= float(args.keep_probability)
    true_positive = int(np.sum(keep & (labels == 1)))
    false_positive = int(np.sum(keep & (labels == 0)))
    false_negative = int(np.sum(~keep & (labels == 1)))
    true_negative = int(np.sum(~keep & (labels == 0)))
    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    if recall < float(args.minimum_train_component_recall):
        raise RuntimeError(
            "Train component recall {:.6f} is below required {:.6f}; refusing artifact."
            .format(recall, args.minimum_train_component_recall)
        )

    project_root = Path(__file__).resolve().parent
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "created_utc": _utc_now(),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": fitted["feature_mean"].tolist(),
        "feature_scale": fitted["feature_scale"].tolist(),
        "coefficients": fitted["coefficients"].tolist(),
        "intercept": fitted["intercept"],
        "keep_probability": float(args.keep_probability),
        "prediction_threshold": prediction_threshold,
        "topology": topology.to_dict(),
        "fit": {
            "algorithm": "deterministic_weighted_logistic_newton",
            "positive_weight": float(args.positive_weight),
            "l2": float(args.l2),
            "max_iterations": int(args.max_iterations),
            "iterations": fitted["iterations"],
            "converged": fitted["converged"],
            "weighted_loss": fitted["weighted_loss"],
            "hyperparameter_selection": "explicit_cli_no_validation_sweep",
        },
        "provenance": {
            "dataset_split": "train",
            "training_selection": manifest["selection"],
            "deployment_event_count_cutoff": int(args.deployment_event_count_cutoff),
            "input_postprocess": input_postprocess_mapping(
                P0ClusterFilter.from_cfg(
                    cfg,
                    prediction_threshold,
                    event_count=int(args.deployment_event_count_cutoff) + 1,
                ).config
            ),
            "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
            "inference_settings": cache_inference_settings,
            "inference_settings_sha256": sha256_json(
                cache_inference_settings
            ),
            "train_cache_manifest_sha256": manifest_sha256,
            "train_cache_schema": manifest["schema"],
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(Path(args.config).resolve()),
            "config_overrides": list(args.override),
            "fit_script_sha256": sha256_file(Path(__file__)),
            "component_module_sha256": sha256_file(
                project_root / "utils" / "component_reranker.py"
            ),
        },
        "train_diagnostics_in_sample_only": {
            "video_count": len(per_video),
            "component_count": int(labels.size),
            "positive_components": int(labels.sum()),
            "negative_components": int((labels == 0).sum()),
            "kept_true_positive_components": true_positive,
            "kept_false_positive_components": false_positive,
            "removed_true_positive_components": false_negative,
            "removed_false_positive_components": true_negative,
            "component_recall": recall,
            "component_precision": precision,
            "minimum_required_component_recall": float(
                args.minimum_train_component_recall
            ),
            "note": "Training diagnostics are not validation or leaderboard evidence.",
            "per_video": per_video,
        },
    }
    artifact["provenance"]["input_postprocess_sha256"] = sha256_json(
        artifact["provenance"]["input_postprocess"]
    )
    _atomic_json(output_path, artifact)
    print("wrote component reranker artifact:", output_path)
    print("artifact sha256:", sha256_file(output_path))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="Build train-only dense raw-score cache.")
    cache.add_argument("--config", type=Path, required=True)
    cache.add_argument("--override", action="append", default=[])
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--output-cache-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--min-event-count-exclusive", type=int, required=True)
    cache.add_argument("--expected-total-videos", type=int, default=99)
    cache.add_argument("--expected-selected-videos", type=int, default=0)
    cache.add_argument(
        "--require-canonical-names",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    fit = subparsers.add_parser("fit", help="Fit strict JSON reranker from train cache.")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--override", action="append", default=[])
    fit.add_argument("--cache-dir", type=Path, required=True)
    fit.add_argument("--output-model", type=Path, required=True)
    fit.add_argument("--prediction-threshold", type=float, required=True)
    fit.add_argument("--deployment-event-count-cutoff", type=int, default=100000)
    fit.add_argument("--spatial-radius", type=int, default=1)
    fit.add_argument("--temporal-bin-size", type=int, default=50)
    fit.add_argument("--max-link-distance", type=float, default=6.0)
    fit.add_argument("--max-gap-bins", type=int, default=1)
    fit.add_argument("--max-component-events", type=int, default=3)
    fit.add_argument("--positive-weight", type=float, required=True)
    fit.add_argument("--l2", type=float, default=0.1)
    fit.add_argument("--keep-probability", type=float, required=True)
    fit.add_argument("--minimum-train-component-recall", type=float, default=0.995)
    fit.add_argument("--max-iterations", type=int, default=100)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "cache":
        if args.min_event_count_exclusive < 0:
            raise ValueError("min_event_count_exclusive must be non-negative.")
        return build_train_cache(args)
    if not 0.0 <= args.keep_probability <= 1.0:
        raise ValueError("keep_probability must be in [0, 1].")
    if not 0.0 <= args.minimum_train_component_recall <= 1.0:
        raise ValueError("minimum_train_component_recall must be in [0, 1].")
    if args.deployment_event_count_cutoff < 0:
        raise ValueError("deployment_event_count_cutoff must be non-negative.")
    return fit_artifact(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Fresh-G2 nested-group runner for atomic H2 component deletion V3.

There are no CLI knobs for data, architecture, optimization, thresholds, or
cutoffs.  ``train-g2`` may read only the frozen G1/G3 fit sources; it writes and
hashes the final two-model checkpoint plus fit-only cutoff before ``evaluate-g2``
is allowed to open G2.  Validation and test paths are not accepted anywhere.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.temporal_frame import (
    build_temporal_context_frame,
    temporal_frame_video_from_events,
)
from model.h2_atomic_component_deletion_net import (
    ActivityFirstComponentScorer,
    PATCH_CHANNELS,
    balanced_component_bce,
    component_scorer_parameter_count,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.atomic_component_deletion import (
    H2_EVENT_COUNT_CUTOFF,
    H2_POLARITY_MINORITY_CUTOFF,
    atomic_delete_or_identity,
    build_component_patch_queries,
    complete_input_polarity_minority_fraction,
    derive_strict_safe_cutoff,
    extract_atomic_components,
    pure_false_positive_targets,
    use_h2_atomic_deletion,
)
from utils.postprocess import ChallengePostprocessor


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = EVC_ROOT / "protocols" / "h2_atomic_component_deletion_g2_science_v3.json"
TRAIN_ROOT = WORKSPACE_ROOT / "datasets" / "EV-UAV-Challenge2" / "train"
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260811_h2_atomic_component_deletion_g2_v3"
M20_PATH = EVC_ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
GPU_AUTHORIZATION_FLAG = "--root-authorized-gpu"
EXPECTED_SCHEMA = "ev-uav-frozen-m20-h2-atomic-component-deletion-g2-v3"

WHOLE_T = 8000
TEMPORAL_BIN_SIZE = 50
CONTEXT_BINS = 5
WIDTH = 346
HEIGHT = 260
LOG_COUNT_CLIP = 4.0
INFERENCE_BATCH_SIZE = 16

C00_OVERRIDES = [
    "TEST.prediction_threshold=0.719",
    "TEMPORAL_FRAME.temporal_frame_enabled=false",
    "TEMPORAL_MEMORY.temporal_memory_enabled=true",
    "TEMPORAL_MEMORY.temporal_memory_sequence_length=16",
    "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0",
    "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true",
    "POSTPROCESS.p0_enabled=true",
    "POSTPROCESS.p0_spatial_radius=2",
    "POSTPROCESS.p0_temporal_bin_size=50",
    "POSTPROCESS.p0_temporal_radius_bins=1",
    "POSTPROCESS.p0_min_cluster_events=3",
    "POSTPROCESS.p0_min_duration_bins=5",
    "POSTPROCESS.p0c_high_confidence_recovery_enabled=true",
    "POSTPROCESS.p0c_retain_min_score=0.95",
    "POSTPROCESS.p0c_density_retain_enabled=false",
    "POSTPROCESS.p0c_density_event_count_cutoff=100000",
    "POSTPROCESS.p0c_density_retain_min_score=0.97",
    "POSTPROCESS.p0b_enabled=false",
    "POSTPROCESS.p18_score_track_recovery_enabled=true",
    "POSTPROCESS.p18_event_count_cutoff=1",
    "POSTPROCESS.p18_max_event_count=35000",
    "POSTPROCESS.p18_candidate_floor=0.53",
    "POSTPROCESS.p18_spatial_radius=5",
    "POSTPROCESS.p18_temporal_bin_size=50",
    "POSTPROCESS.p18_max_link_distance=8.0",
    "POSTPROCESS.p18_max_gap_bins=1",
    "POSTPROCESS.p18_min_track_bins=4",
    "POSTPROCESS.p18_restore_mode=best",
    "POSTPROCESS.p18_max_restore_events_per_component=0",
    "POSTPROCESS.p6_density_threshold_enabled=true",
    "POSTPROCESS.p6_event_count_cutoff=30000",
    "POSTPROCESS.p6_low_density_threshold=0.718",
    "POSTPROCESS.p6_high_density_threshold=0.719",
]


@dataclass
class PreparedSource:
    source_name: str
    group_id: str
    event_count: int
    polarity_minority_fraction: float
    input_only_h2_route: bool
    prediction_threshold: float
    base_raw_scores: np.ndarray
    base_scores: np.ndarray
    locations: np.ndarray
    component_event_indices: tuple[np.ndarray, ...]
    component_patches: tuple[np.ndarray, ...]
    pure_fp_targets: np.ndarray | None
    labels: np.ndarray | None
    target_ids: np.ndarray | None
    postprocess_stats: dict
    rich_cache_reference_scores: np.ndarray
    rich_cache_record_sha256: str
    rich_cache_comparison: dict


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_float32(values):
    array = np.asarray(values, dtype="<f4").reshape(-1)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def write_json_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def save_torch_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def write_npz_exclusive(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _rich_cache_manifest(protocol):
    cache = protocol["rich_m20_cache"]
    manifest_path = WORKSPACE_ROOT / cache["manifest_workspace_relative_path"]
    if not manifest_path.is_file() or sha256_file(manifest_path) != cache["manifest_sha256"]:
        raise RuntimeError("rich M20 cache manifest changed")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema") != "ev-uav-component-reranker-train-cache-v1"
        or manifest.get("dataset_split") != "train"
        or manifest.get("base_checkpoint_sha256") != sha256_file(M20_PATH)
    ):
        raise RuntimeError("rich M20 cache provenance is incompatible")
    return manifest_path, manifest


def _load_rich_cache_input_reference(protocol, source_name):
    manifest_path, manifest = _rich_cache_manifest(protocol)
    matches = [record for record in manifest["records"] if record["source_name"] == source_name]
    if len(matches) != 1:
        raise RuntimeError("rich M20 cache source record is missing or duplicate")
    metadata = matches[0]
    record_path = manifest_path.parent / metadata["record"]
    if not record_path.is_file() or sha256_file(record_path) != metadata["record_sha256"]:
        raise RuntimeError("rich M20 cache source record changed")
    if metadata["source_sha256"] != protocol["sources"][source_name]["sha256"]:
        raise RuntimeError("rich cache source provenance mismatch")
    with np.load(record_path, allow_pickle=False) as archive:
        scores = np.asarray(archive["scores"], dtype=np.float32).copy()
        locations = np.asarray(archive["locs"], dtype=np.int64).copy()
    return scores, locations, metadata["record_sha256"]


def _component_offsets(event_indices):
    lengths = np.asarray([len(indices) for indices in event_indices], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flattened = (
        np.concatenate(event_indices).astype(np.int64, copy=False)
        if event_indices
        else np.empty(0, dtype=np.int64)
    )
    return offsets, flattened


def persist_source_feature_artifact(source, path):
    """Persist immutable input-only patches and atomic IDs before scoring."""

    component_offsets, component_events = _component_offsets(
        source.component_event_indices
    )
    patch_lengths = np.asarray(
        [patches.shape[0] for patches in source.component_patches], dtype=np.int64
    )
    patch_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(patch_lengths))
    )
    flattened_patches = (
        np.concatenate(source.component_patches, axis=0).astype(np.float16, copy=False)
        if source.component_patches
        else np.empty((0, PATCH_CHANNELS, 1, 1), dtype=np.float16)
    )
    event_component_ids = np.full(source.event_count, -1, dtype=np.int32)
    for component_id, indices in enumerate(source.component_event_indices):
        event_component_ids[np.asarray(indices, dtype=np.int64)] = component_id
    write_npz_exclusive(
        path,
        artifact_schema=np.asarray("ev-uav-h2-atomic-component-input-artifact-v3"),
        event_count=np.asarray(source.event_count, dtype=np.int64),
        m20_raw_scores=source.base_raw_scores.astype(np.float32, copy=False),
        c00_post_scores=source.base_scores.astype(np.float32, copy=False),
        rich_cache_reference_scores=source.rich_cache_reference_scores.astype(
            np.float32, copy=False
        ),
        locations=source.locations.astype(np.int16, copy=False),
        event_component_ids=event_component_ids,
        component_offsets=component_offsets,
        component_event_indices=component_events,
        component_patch_offsets=patch_offsets,
        component_patches=flattened_patches,
        activity_polarity_raw_patches=flattened_patches[:, 17:20],
        component_query_masks=flattened_patches[:, 20],
    )
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "component_count": len(source.component_event_indices),
        "component_patch_bin_count": int(patch_lengths.sum()),
        "contains_labels_or_target_ids": False,
        "rich_cache_record_sha256": source.rich_cache_record_sha256,
        "rich_cache_comparison": source.rich_cache_comparison,
    }


def persist_component_score_artifact(
    path,
    source,
    *,
    model_group_ids,
    score_records,
    consensus_probabilities,
    cutoff,
    enabled,
    include_fit_targets,
):
    component_count = len(source.component_event_indices)
    if len(score_records) != len(model_group_ids):
        raise ValueError("score records and model group IDs differ")
    arrays = {
        "artifact_schema": np.asarray("ev-uav-h2-atomic-component-score-artifact-v3"),
        "component_ids": np.arange(component_count, dtype=np.int32),
        "model_group_ids": np.asarray(model_group_ids),
        "model_pure_fp_probabilities": np.stack(
            [record["probabilities"] for record in score_records]
        ).astype(np.float64, copy=False),
        "activity_adapter_embeddings": np.stack(
            [record["activity_embeddings"] for record in score_records]
        ).astype(np.float32, copy=False),
        "fused_component_embeddings": np.stack(
            [record["fused_embeddings"] for record in score_records]
        ).astype(np.float32, copy=False),
        "consensus_pure_fp_probability": np.asarray(
            consensus_probabilities, dtype=np.float64
        ),
        "strict_safe_cutoff": np.asarray(cutoff, dtype=np.float64),
        "deletion_enabled": np.asarray(bool(enabled), dtype=np.bool_),
        "delete_component": (
            np.asarray(consensus_probabilities, dtype=np.float64) >= float(cutoff)
        ) & bool(enabled),
    }
    if include_fit_targets:
        if source.pure_fp_targets is None:
            raise RuntimeError("fit score artifact requested without fit targets")
        arrays["fit_only_pure_fp_targets"] = source.pure_fp_targets.astype(
            np.uint8, copy=False
        )
    write_npz_exclusive(path, **arrays)
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "component_count": component_count,
        "model_group_ids": list(model_group_ids),
        "contains_fit_only_targets": bool(include_fit_targets),
    }


def load_checkpoint_file(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_protocol():
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError("unexpected V3 protocol schema")
    if protocol.get("status") != "frozen_before_any_v3_gpu_probe_or_g2_prediction":
        raise RuntimeError("V3 protocol is not frozen in its pre-GPU state")
    return protocol


def source_manifest(protocol):
    return protocol["source_groups"]


def source_group(protocol, source_name):
    matches = [
        group_id
        for group_id, group in source_manifest(protocol).items()
        if source_name in group["sources"]
    ]
    if len(matches) != 1:
        raise RuntimeError("source does not belong to exactly one frozen group")
    return matches[0]


def source_path(protocol, source_name):
    all_names = set().union(
        *(set(group["sources"]) for group in source_manifest(protocol).values())
    )
    if source_name not in all_names:
        raise RuntimeError("source lies outside the frozen H2 manifest")
    path = (TRAIN_ROOT / source_name).resolve()
    if path.parent != TRAIN_ROOT.resolve():
        raise RuntimeError("source path escaped the official train root")
    return path


def require_gpu_authorization(args):
    if not bool(getattr(args, "root_authorized_gpu", False)):
        raise RuntimeError("GPU execution requires {}".format(GPU_AUTHORIZATION_FLAG))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


@contextmanager
def gpu_run_lock(purpose):
    """Best-effort shared-workspace lock; parent still performs the GPU process gate."""

    lock_path = WORKSPACE_ROOT / "experiments" / ".codex_gpu_exclusive.lock"
    descriptor = None
    try:
        descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        os.write(descriptor, json.dumps({"pid": os.getpid(), "purpose": purpose}).encode())
        os.fsync(descriptor)
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def build_released_m20(device):
    payload = load_checkpoint_file(M20_PATH, map_location="cpu")
    metadata = payload.get("temporal_memory", {})
    required = {
        "temporal_bin_size": TEMPORAL_BIN_SIZE,
        "context_bins": CONTEXT_BINS,
        "width": 16,
        "sequence_length": 16,
    }
    for key, expected in required.items():
        if int(metadata.get(key, -1)) != expected:
            raise RuntimeError("released M20 metadata differs for {}".format(key))
    model = BidirectionalTemporalMemoryNet(
        input_channels=CONTEXT_BINS * 2,
        width=16,
        density_calibration_enabled=bool(metadata.get("density_calibration_enabled", False)),
        density_calibration_v2_enabled=bool(metadata.get("density_calibration_v2_enabled", False)),
        confidence_head_enabled=bool(metadata.get("confidence_head_enabled", False)),
        temporal_attention_enabled=bool(metadata.get("temporal_attention_enabled", False)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _frame_tensor(video, bins, device):
    frames = np.stack(
        [
            build_temporal_context_frame(
                video,
                int(temporal_bin),
                CONTEXT_BINS,
                WIDTH,
                HEIGHT,
                LOG_COUNT_CLIP,
            )
            for temporal_bin in bins
        ],
        axis=0,
    )
    return torch.from_numpy(frames).float().to(device)


def _decode_frozen_features(model, frames, residual):
    with torch.no_grad():
        level0, level1, level2, bottleneck = model._encode(frames)
        if residual.shape != bottleneck.shape:
            raise ValueError("M20 memory residual does not align")
        base = model.base
        decoded2 = base.decoder2(bottleneck + residual, level2)
        decoded1 = base.decoder1(decoded2, level1)
        decoded0 = base.decoder0(decoded1, level0)
        if base.density_calibration_enabled:
            decoded0 = base.density_calibrator(decoded0, frames[:, : model.input_channels])
        logits = base.head(decoded0)
    return decoded0.detach(), logits.detach()


def _centre_polarity_activity(frames):
    start = (CONTEXT_BINS // 2) * 2
    negative = frames[:, start : start + 1]
    positive = frames[:, start + 1 : start + 2]
    return torch.cat((negative, positive, negative + positive), dim=1)


def _load_input_only(path):
    with np.load(path, allow_pickle=False) as archive:
        events = np.asarray(archive["evs_norm"])
        locations3 = np.asarray(archive["ev_loc"]).astype(np.int64, copy=False)
        polarities = events[:, 3].astype(np.float32, copy=True)
    video = temporal_frame_video_from_events(
        name="",
        locations=locations3,
        polarities=polarities,
        temporal_bin_size=TEMPORAL_BIN_SIZE,
        whole_t=WHOLE_T,
        labels=None,
        target_ids=None,
    )
    locations4 = np.column_stack(
        (np.zeros(locations3.shape[0], dtype=np.int64), locations3)
    )
    return video, polarities, locations4


def _load_truth(path):
    with np.load(path, allow_pickle=False) as archive:
        events = np.asarray(archive["evs_norm"])
    return (
        events[:, 4].astype(np.uint8, copy=True),
        events[:, 5].astype(np.int64, copy=True),
    )


def _build_c00(protocol):
    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay

    if protocol["baseline"]["fixed_config_overrides"] != C00_OVERRIDES:
        raise RuntimeError("C00 overrides differ from protocol")
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml", C00_OVERRIDES
    )
    c00 = component_crossfit.validate_c00_config(
        cfg, float(protocol["baseline"]["prediction_threshold"])
    )
    if component_crossfit.sha256_json(c00) != protocol["baseline"]["effective_c00_sha256"]:
        raise RuntimeError("effective C00 contract changed")
    return cfg, c00


def _m20_raw_scores_and_memory(model, video, device):
    temporal_count = len(video.event_indices_by_bin)
    bottlenecks = []
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            bins = range(start, min(start + INFERENCE_BATCH_SIZE, temporal_count))
            bottlenecks.append(model.encode_bottleneck(_frame_tensor(video, bins, device)))
        memory = model.temporal_residual(torch.cat(bottlenecks, dim=0))
    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, temporal_count)
            bins = range(start, stop)
            frames = _frame_tensor(video, bins, device)
            _, logits = _decode_frozen_features(model, frames, memory[start:stop])
            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            for temporal_bin in range(start, stop):
                indices = video.event_indices_by_bin[temporal_bin]
                if indices.size == 0:
                    continue
                local = temporal_bin - start
                locations = video.locations[indices]
                scores[indices] = probabilities[local, locations[:, 1], locations[:, 0]]
    if not np.isfinite(scores).all():
        raise RuntimeError("M20 produced a non-finite event score")
    return scores, memory


def _component_patch_sequences(
    model,
    video,
    memory,
    queries,
    patch_radius,
    device,
):
    temporal_count = len(video.event_indices_by_bin)
    patch_size = 2 * int(patch_radius) + 1
    sequences = [
        np.empty((len(component), PATCH_CHANNELS, patch_size, patch_size), dtype=np.float16)
        for component in queries
    ]
    by_bin = {}
    for component_index, component in enumerate(queries):
        for sequence_index, query in enumerate(component):
            by_bin.setdefault(query.temporal_bin, []).append(
                (component_index, sequence_index, query)
            )
    written = np.zeros(len(queries), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, temporal_count)
            bins = range(start, stop)
            frames = _frame_tensor(video, bins, device)
            decoder, logits = _decode_frozen_features(model, frames, memory[start:stop])
            dense = torch.cat(
                (decoder, logits, _centre_polarity_activity(frames)), dim=1
            )
            if dense.shape[1] != PATCH_CHANNELS - 1:
                raise RuntimeError("dense V3 patch channel contract changed")
            padded = F.pad(
                dense, (patch_radius, patch_radius, patch_radius, patch_radius)
            )
            selected = []
            metadata = []
            for temporal_bin in range(start, stop):
                for component_index, sequence_index, query in by_bin.get(temporal_bin, ()): 
                    local = temporal_bin - start
                    patch = padded[
                        local,
                        :,
                        query.center_y : query.center_y + patch_size,
                        query.center_x : query.center_x + patch_size,
                    ]
                    if patch.shape[-2:] != (patch_size, patch_size):
                        raise RuntimeError("component crop escaped padded dense map")
                    selected.append(patch)
                    metadata.append((component_index, sequence_index, query.component_mask))
            if selected:
                values = torch.stack(selected).to(dtype=torch.float16).cpu().numpy()
                for value, (component_index, sequence_index, mask) in zip(values, metadata):
                    sequences[component_index][sequence_index, :-1] = value
                    sequences[component_index][sequence_index, -1] = mask.astype(
                        np.float16, copy=False
                    )
                    written[component_index] += 1
    expected = np.asarray([len(component) for component in queries], dtype=np.int64)
    if not np.array_equal(written, expected):
        raise RuntimeError("not every component patch query was materialized")
    for sequence in sequences:
        if not np.isfinite(sequence).all() or not np.all(sequence[:, -1].sum(axis=(-2, -1)) > 0):
            raise RuntimeError("invalid component patch sequence")
    return tuple(sequences)


def prepare_source(protocol, model, cfg, source_name, device, *, include_truth):
    path = source_path(protocol, source_name)
    metadata = protocol["sources"][source_name]
    if sha256_file(path) != metadata["sha256"]:
        raise RuntimeError("source SHA-256 mismatch: {}".format(source_name))
    video, polarities, locations = _load_input_only(path)
    event_count = int(len(polarities))
    minority = complete_input_polarity_minority_fraction(polarities)
    if event_count != int(metadata["event_count"]) or not use_h2_atomic_deletion(
        event_count, polarities
    ):
        raise RuntimeError("source no longer satisfies frozen H2 route")
    expected_minority = float(metadata["polarity_minority_fraction"])
    if not np.isclose(minority, expected_minority, rtol=0.0, atol=1e-15):
        raise RuntimeError("source polarity fraction changed")

    raw_scores, memory = _m20_raw_scores_and_memory(model, video, device)
    cache_scores, cache_locations, cache_record_sha256 = _load_rich_cache_input_reference(
        protocol, source_name
    )
    if cache_scores.shape != raw_scores.shape or not np.array_equal(
        cache_locations, locations[:, 1:4]
    ):
        raise RuntimeError("rich M20 cache event alignment changed")
    cache_delta = np.abs(
        raw_scores.astype(np.float64) - cache_scores.astype(np.float64)
    )
    threshold_for_reference = np.float32(
        protocol["baseline"]["prediction_threshold"]
    )
    rich_cache_comparison = {
        "reference_is_model_input": False,
        "reference_inference_batch_size": int(
            protocol["rich_m20_cache"]["reference_inference_batch_size"]
        ),
        "v3_inference_batch_size": INFERENCE_BATCH_SIZE,
        "reference_scores_sha256": sha256_float32(cache_scores),
        "v3_scores_sha256": sha256_float32(raw_scores),
        "maximum_absolute_difference": float(cache_delta.max() if cache_delta.size else 0.0),
        "mean_absolute_difference": float(cache_delta.mean() if cache_delta.size else 0.0),
        "threshold_disagreement_events": int(
            np.count_nonzero(
                (cache_scores >= threshold_for_reference)
                != (raw_scores >= threshold_for_reference)
            )
        ),
    }
    threshold = float(protocol["baseline"]["prediction_threshold"])
    processed, postprocess_stats = ChallengePostprocessor.from_cfg(
        cfg, threshold, event_count=event_count
    ).apply(torch.from_numpy(raw_scores.copy()), torch.from_numpy(locations).long())
    base_scores = processed.numpy().astype(np.float32, copy=True)
    topology = protocol["atomic_components"]
    components = extract_atomic_components(
        base_scores,
        locations,
        threshold,
        spatial_radius=int(topology["spatial_radius"]),
        temporal_bin_size=int(topology["temporal_bin_size"]),
        temporal_radius_bins=int(topology["temporal_radius_bins"]),
    )
    queries = build_component_patch_queries(
        components.event_indices,
        locations,
        patch_radius=int(protocol["architecture"]["patch_radius"]),
        temporal_bin_size=int(topology["temporal_bin_size"]),
    )
    patches = _component_patch_sequences(
        model,
        video,
        memory,
        queries,
        int(protocol["architecture"]["patch_radius"]),
        device,
    )
    del memory
    labels = target_ids = targets = None
    if include_truth:
        labels, target_ids = _load_truth(path)
        targets = pure_false_positive_targets(components.event_indices, labels)
        if targets.size == 0 or not np.any(targets == 0) or not np.any(targets == 1):
            raise RuntimeError("fit source lacks both component classes")
    return PreparedSource(
        source_name=source_name,
        group_id=source_group(protocol, source_name),
        event_count=event_count,
        polarity_minority_fraction=minority,
        input_only_h2_route=True,
        prediction_threshold=threshold,
        base_raw_scores=raw_scores,
        base_scores=base_scores,
        locations=locations,
        component_event_indices=components.event_indices,
        component_patches=patches,
        pure_fp_targets=targets,
        labels=labels,
        target_ids=target_ids,
        postprocess_stats=asdict(postprocess_stats),
        rich_cache_reference_scores=cache_scores,
        rich_cache_record_sha256=cache_record_sha256,
        rich_cache_comparison=rich_cache_comparison,
    )


class ComponentSequenceDataset(Dataset):
    """Variable-length patch sequences with fit-derived weights."""

    def __init__(self, sources):
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("component training requires at least one source")
        self.rows = []
        base_weights = []
        targets = []
        source_count = len(self.sources)
        for source_index, source in enumerate(self.sources):
            if source.pure_fp_targets is None:
                raise ValueError("component training source has no train-only targets")
            component_count = len(source.component_patches)
            if component_count != source.pure_fp_targets.size or component_count <= 0:
                raise ValueError("component patches and targets do not align")
            for component_index in range(component_count):
                self.rows.append((source_index, component_index))
                base_weights.append(1.0 / (source_count * component_count))
                targets.append(int(source.pure_fp_targets[component_index]))
        targets = np.asarray(targets, dtype=np.uint8)
        weights = np.asarray(base_weights, dtype=np.float64)
        class_zero = targets == 0
        class_one = targets == 1
        mass_zero = float(weights[class_zero].sum())
        mass_one = float(weights[class_one].sum())
        if mass_zero <= 0.0 or mass_one <= 0.0:
            raise RuntimeError("fit group must contain both component classes")
        weights[class_zero] *= 0.5 / mass_zero
        weights[class_one] *= 0.5 / mass_one
        weights *= weights.size / weights.sum()
        self.targets = targets
        self.weights = weights.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source_index, component_index = self.rows[int(index)]
        source = self.sources[source_index]
        return {
            "patches": source.component_patches[component_index],
            "target": np.float32(self.targets[index]),
            "weight": np.float32(self.weights[index]),
        }


def component_sequence_collate(items):
    if not items:
        raise ValueError("cannot collate an empty component batch")
    lengths = np.asarray([item["patches"].shape[0] for item in items], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("component patch sequence must not be empty")
    max_length = int(lengths.max())
    example = items[0]["patches"]
    output = np.empty(
        (len(items), max_length, example.shape[1], example.shape[2], example.shape[3]),
        dtype=np.float16,
    )
    for row, item in enumerate(items):
        patches = np.asarray(item["patches"], dtype=np.float16)
        if patches.shape[1:] != example.shape[1:]:
            raise ValueError("component patch shapes differ")
        length = patches.shape[0]
        output[row, :length] = patches
        # Repeating the final valid slice makes padded masks nonempty.  Packed
        # GRU lengths guarantee these repeated values never affect the score.
        if length < max_length:
            output[row, length:] = patches[-1]
    return {
        "patches": torch.from_numpy(output),
        "lengths": torch.from_numpy(lengths),
        "targets": torch.from_numpy(
            np.asarray([item["target"] for item in items], dtype=np.float32)
        ),
        "weights": torch.from_numpy(
            np.asarray([item["weight"] for item in items], dtype=np.float32)
        ),
    }


def _new_component_model(protocol, device):
    architecture = protocol["architecture"]
    return ActivityFirstComponentScorer(
        decoder_channels=int(architecture["decoder_channels"]),
        activity_width=int(architecture["activity_width"]),
        semantic_width=int(architecture["semantic_width"]),
        temporal_width=int(architecture["temporal_width"]),
    ).to(device)


def train_group_model(protocol, group_id, sources, device, *, max_steps=None):
    training = protocol["training"]
    seed = int(training["group_seeds"][group_id])
    seed_everything(seed)
    model = _new_component_model(protocol, device).train()
    dataset = ComponentSequenceDataset(sources)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=component_sequence_collate,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training["mixed_precision"]))
    records = []
    step = 0
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        for batch in loader:
            patches = batch["patches"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(training["mixed_precision"]),
            ):
                logits = model(patches, lengths)
                loss = balanced_component_bce(logits, targets, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            activity_grad = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in model.activity_adapter.parameters()
                if parameter.grad is not None
            )
            semantic_grad = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in model.semantic_adapter.parameters()
                if parameter.grad is not None
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            record = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "gradient_norm": float(grad_norm.detach()),
                "activity_adapter_gradient_l1": activity_grad,
                "semantic_adapter_gradient_l1": semantic_grad,
                "batch_components": int(logits.numel()),
                "batch_pure_fp_fraction": float(targets.float().mean()),
            }
            if not all(np.isfinite(float(record[key])) for key in (
                "loss",
                "gradient_norm",
                "activity_adapter_gradient_l1",
                "semantic_adapter_gradient_l1",
                "batch_pure_fp_fraction",
            )):
                raise RuntimeError("non-finite V3 training diagnostic")
            records.append(record)
            if max_steps is not None and step >= int(max_steps):
                break
        if max_steps is not None and step >= int(max_steps):
            break
    torch.cuda.synchronize(device)
    return {
        "group_id": group_id,
        "model": model.eval(),
        "records": records,
        "step_count": step,
        "dataset_component_count": len(dataset),
        "elapsed_seconds": time.perf_counter() - started,
    }


def score_source(model, source, protocol, device):
    batch_size = int(protocol["inference"]["component_batch_size"])
    probabilities = []
    activity_embeddings = []
    fused_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source.component_patches), batch_size):
            items = [
                {"patches": patches, "target": np.float32(0), "weight": np.float32(1)}
                for patches in source.component_patches[start : start + batch_size]
            ]
            batch = component_sequence_collate(items)
            patches = batch["patches"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(protocol["training"]["mixed_precision"]),
            ):
                activity_embedding, fused_embedding = model.component_embeddings(
                    patches, lengths
                )
                logits = model.classifier(fused_embedding).squeeze(1)
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            activity_embeddings.append(activity_embedding.float().cpu().numpy())
            fused_embeddings.append(fused_embedding.float().cpu().numpy())
    output = np.concatenate(probabilities).astype(np.float64, copy=False)
    if output.size != len(source.component_patches) or not np.isfinite(output).all():
        raise RuntimeError("component probability output is invalid")
    activity_output = np.concatenate(activity_embeddings).astype(np.float32, copy=False)
    fused_output = np.concatenate(fused_embeddings).astype(np.float32, copy=False)
    if activity_output.shape[0] != output.size or fused_output.shape[0] != output.size:
        raise RuntimeError("component embedding output is invalid")
    return {
        "probabilities": output,
        "activity_embeddings": activity_output,
        "fused_embeddings": fused_output,
    }


def _sum_counts(values):
    from crossfit_component_reranker import SufficientCounts

    total = SufficientCounts()
    for value in values:
        total = total + value
    return total


def _counts(source, scores):
    from crossfit_component_reranker import sufficient_counts_for_video

    if source.labels is None or source.target_ids is None:
        raise RuntimeError("metrics requested before truth was attached")
    return sufficient_counts_for_video(
        np.asarray(scores, dtype=np.float32),
        source.labels,
        source.target_ids,
        source.locations,
        float(source.prediction_threshold),
    )


def _metrics(counts):
    from crossfit_component_reranker import metrics_from_counts

    return metrics_from_counts(counts)


def _metric_record(counts):
    return {"counts": counts.to_dict(), "metrics": _metrics(counts)}


def _safety_gates(candidate, baseline):
    candidate_metrics = _metrics(candidate)
    baseline_metrics = _metrics(baseline)
    return {
        "score_not_lower": candidate_metrics["score"] >= baseline_metrics["score"],
        "iou_not_lower": candidate_metrics["iou"] >= baseline_metrics["iou"],
        "pd_not_lower": candidate_metrics["pd"] >= baseline_metrics["pd"],
        "fa_not_higher": candidate_metrics["fa"] <= baseline_metrics["fa"],
        "true_positive_events_not_lower": candidate.true_positive_events >= baseline.true_positive_events,
        "false_positive_events_not_higher": candidate.false_positive_events <= baseline.false_positive_events,
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "false_components_not_higher": candidate.false_components <= baseline.false_components,
    }


def _evaluate_prepared_sources(sources, probability_by_source, cutoff, enabled):
    per_source = []
    baselines = []
    candidates = []
    atomic_integrity = True
    for source in sources:
        probabilities = probability_by_source[source.source_name]
        candidate_scores, receipt = atomic_delete_or_identity(
            source.base_scores,
            source.component_event_indices,
            probabilities,
            cutoff,
            enabled=enabled,
        )
        baseline = _counts(source, source.base_scores)
        candidate = _counts(source, candidate_scores)
        baselines.append(baseline)
        candidates.append(candidate)
        atomic_integrity &= bool(
            receipt.complete_components_only and receipt.retained_scores_bitwise_equal
        )
        per_source.append(
            {
                "source_name": source.source_name,
                "component_count": len(source.component_event_indices),
                "atomic_edit": asdict(receipt),
                "base": _metric_record(baseline),
                "candidate": _metric_record(candidate),
                "base_scores_sha256": sha256_float32(source.base_scores),
                "candidate_scores_sha256": sha256_float32(candidate_scores),
            }
        )
    pooled_base = _sum_counts(baselines)
    pooled_candidate = _sum_counts(candidates)
    return {
        "sources": per_source,
        "baseline": _metric_record(pooled_base),
        "candidate": _metric_record(pooled_candidate),
        "gates": _safety_gates(pooled_candidate, pooled_base),
        "score_delta": _metrics(pooled_candidate)["score"] - _metrics(pooled_base)["score"],
        "atomic_integrity": atomic_integrity,
    }


def nested_fit_calibration(protocol, models, prepared_by_group, device):
    fit_groups = tuple(protocol["formal_g2"]["fit_groups"])
    if len(fit_groups) != 2:
        raise RuntimeError("fresh G2 nested calibration requires exactly two fit groups")
    left, right = fit_groups
    # Each fit component is scored only by the model trained on the other
    # complete source group.  No self-source or self-group prediction enters
    # calibration.
    probability_by_source = {}
    score_record_by_source = {}
    for source in prepared_by_group[left]:
        score_record_by_source[source.source_name] = score_source(
            models[right], source, protocol, device
        )
        probability_by_source[source.source_name] = score_record_by_source[
            source.source_name
        ]["probabilities"]
    for source in prepared_by_group[right]:
        score_record_by_source[source.source_name] = score_source(
            models[left], source, protocol, device
        )
        probability_by_source[source.source_name] = score_record_by_source[
            source.source_name
        ]["probabilities"]
    all_sources = tuple(prepared_by_group[left]) + tuple(prepared_by_group[right])
    probabilities = np.concatenate(
        [probability_by_source[source.source_name] for source in all_sources]
    )
    targets = np.concatenate([source.pure_fp_targets for source in all_sources])
    cutoff, has_safe_candidate, calibration = derive_strict_safe_cutoff(
        probabilities, targets
    )
    group_diagnostics = {
        group_id: _evaluate_prepared_sources(
            prepared_by_group[group_id], probability_by_source, cutoff, has_safe_candidate
        )
        for group_id in fit_groups
    }
    pooled = _evaluate_prepared_sources(
        all_sources, probability_by_source, cutoff, has_safe_candidate
    )
    all_inner_gates = all(
        diagnostic["atomic_integrity"] and all(diagnostic["gates"].values())
        for diagnostic in group_diagnostics.values()
    ) and pooled["atomic_integrity"] and all(pooled["gates"].values())
    enabled = bool(has_safe_candidate and all_inner_gates)
    return {
        "cutoff": cutoff,
        "enabled": enabled,
        "calibration": calibration,
        "group_diagnostics": group_diagnostics,
        "pooled": pooled,
        "score_record_by_source": score_record_by_source,
        "all_inner_safety_gates_passed": all_inner_gates,
        "inference_aggregation": "minimum_pure_fp_probability_requires_both_fit_group_models_to_agree",
    }


def audit_protocol(run_tests=True):
    protocol = load_protocol()
    if sha256_file(M20_PATH) != protocol["released_m20"]["sha256"]:
        raise RuntimeError("released M20 SHA-256 mismatch")
    if protocol["baseline"]["fixed_config_overrides"] != C00_OVERRIDES:
        raise RuntimeError("baseline override contract changed")
    route = protocol["input_only_route"]
    if (
        int(route["event_count_cutoff_exclusive"]) != H2_EVENT_COUNT_CUTOFF
        or float(route["polarity_minority_cutoff"]) != H2_POLARITY_MINORITY_CUTOFF
    ):
        raise RuntimeError("input-only H2 route changed")
    route_evidence = route["route_evidence_protocol"]
    route_path = WORKSPACE_ROOT / route_evidence["workspace_relative_path"]
    if not route_path.is_file() or sha256_file(route_path) != route_evidence["sha256"]:
        raise RuntimeError("input-only H2 route evidence changed")
    groups = source_manifest(protocol)
    if tuple(groups) != ("g1_088_091", "g2_092_094", "g3_095_098"):
        raise RuntimeError("frozen source-group order changed")
    all_names = [name for group in groups.values() for name in group["sources"]]
    if len(all_names) != 11 or len(set(all_names)) != 11 or set(all_names) != set(protocol["sources"]):
        raise RuntimeError("source groups do not exactly partition H2")
    formal = protocol["formal_g2"]
    fit_names = set().union(*(set(groups[group]["sources"]) for group in formal["fit_groups"]))
    held_names = set(groups[formal["held_group"]]["sources"])
    if fit_names & held_names or fit_names | held_names != set(all_names):
        raise RuntimeError("formal G2 fit/held partition is invalid")
    if formal["held_group"] != "g2_092_094" or "g2_092_094" in formal["fit_groups"]:
        raise RuntimeError("fresh G2 held group leaked into fit groups")
    for name, metadata in protocol["sources"].items():
        path = source_path(protocol, name)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise RuntimeError("frozen train source changed: {}".format(name))
        if int(metadata["event_count"]) <= H2_EVENT_COUNT_CUTOFF:
            raise RuntimeError("source metadata lies outside H2")
        if float(metadata["polarity_minority_fraction"]) < H2_POLARITY_MINORITY_CUTOFF:
            raise RuntimeError("source polarity metadata lies outside H2")
    cache_manifest_path, cache_manifest = _rich_cache_manifest(protocol)
    cache_records = {
        record["source_name"]: record
        for record in cache_manifest["records"]
        if record["source_name"] in protocol["sources"]
    }
    if set(cache_records) != set(protocol["sources"]):
        raise RuntimeError("rich cache does not cover all V3 H2 sources")
    for name, record in cache_records.items():
        record_path = cache_manifest_path.parent / record["record"]
        if (
            record["source_sha256"] != protocol["sources"][name]["sha256"]
            or not record_path.is_file()
            or sha256_file(record_path) != record["record_sha256"]
        ):
            raise RuntimeError("rich cache record provenance changed: {}".format(name))
    _cfg, c00 = _build_c00(protocol)

    base, payload = build_released_m20(torch.device("cpu"))
    if len(payload["model_state_dict"]) != int(protocol["released_m20"]["state_tensor_count"]):
        raise RuntimeError("M20 state tensor count mismatch")
    if sum(parameter.numel() for parameter in base.parameters()) != int(
        protocol["released_m20"]["parameter_count"]
    ):
        raise RuntimeError("M20 parameter count mismatch")
    scorer = _new_component_model(protocol, torch.device("cpu"))
    expected_parameters = int(protocol["architecture"]["trainable_parameter_count"])
    if component_scorer_parameter_count(scorer) != expected_parameters:
        raise RuntimeError("V3 component scorer parameter count mismatch")
    if PATCH_CHANNELS != int(protocol["architecture"]["patch_channels"]):
        raise RuntimeError("V3 patch channel count mismatch")

    evidence = protocol["development_evidence"]
    for record in evidence["artifacts"]:
        path = WORKSPACE_ROOT / record["workspace_relative_path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("development-evidence artifact changed")
    test_output = None
    if run_tests:
        command = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(EVC_ROOT / "tests"),
            "-p",
            "test_h2_atomic_component_deletion_v3.py",
            "-v",
        ]
        completed = subprocess.run(command, cwd=EVC_ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError("V3 CPU tests failed:\n{}".format(completed.stdout + completed.stderr))
        test_output = (completed.stdout + completed.stderr).strip()
    return {
        "schema": "ev-uav-h2-atomic-component-deletion-cpu-audit-v3",
        "created_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": sha256_file(EVC_ROOT / "model" / "h2_atomic_component_deletion_net.py"),
        "utility_sha256": sha256_file(EVC_ROOT / "utils" / "atomic_component_deletion.py"),
        "tests_sha256": sha256_file(EVC_ROOT / "tests" / "test_h2_atomic_component_deletion_v3.py"),
        "released_m20_sha256": sha256_file(M20_PATH),
        "released_m20_state_sha256": state_sha256(base.state_dict()),
        "released_m20_parameter_count": sum(parameter.numel() for parameter in base.parameters()),
        "component_scorer_parameter_count": component_scorer_parameter_count(scorer),
        "effective_c00": c00,
        "rich_m20_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "fit_groups": formal["fit_groups"],
        "held_group": formal["held_group"],
        "held_g2_array_read": False,
        "validation_or_test_read": False,
        "gpu_used": False,
        "cpu_tests": test_output,
    }


def run_audit(_args):
    print(json.dumps(audit_protocol(run_tests=True), indent=2, ensure_ascii=False))


def run_probe(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    result_path = OUTPUT_ROOT / "resource_probe" / "eight_step_probe.json"
    if result_path.exists() or result_path.parent.exists():
        raise FileExistsError("refusing to overwrite V3 probe output")
    fit_source = protocol["probe"]["fit_source"]
    fit_group = source_group(protocol, fit_source)
    if fit_group not in protocol["formal_g2"]["fit_groups"]:
        raise RuntimeError("probe source is outside fresh-G2 fit groups")
    with gpu_run_lock("h2_atomic_component_deletion_v3_probe"):
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        m20, _ = build_released_m20(device)
        m20_before = state_sha256(m20.state_dict())
        cfg, _ = _build_c00(protocol)
        source = prepare_source(
            protocol, m20, cfg, fit_source, device, include_truth=True
        )
        feature_artifact = persist_source_feature_artifact(
            source,
            result_path.parent / "immutable_fit_input" / (Path(fit_source).stem + ".npz"),
        )
        result = train_group_model(
            protocol,
            fit_group,
            [source],
            device,
            max_steps=int(protocol["probe"]["optimizer_steps"]),
        )
        if result["step_count"] != int(protocol["probe"]["optimizer_steps"]):
            raise RuntimeError("V3 probe step count mismatch")
        if result["records"][0]["activity_adapter_gradient_l1"] <= 0.0:
            raise RuntimeError("activity adapter was not reachable on probe step one")
        if result["records"][0]["semantic_adapter_gradient_l1"] <= 0.0:
            raise RuntimeError("semantic adapter was not reachable on probe step one")
        probe_scores = score_source(result["model"], source, protocol, device)
        score_artifact = persist_component_score_artifact(
            result_path.parent / "immutable_probe_scores" / (Path(fit_source).stem + ".npz"),
            source,
            model_group_ids=[fit_group],
            score_records=[probe_scores],
            consensus_probabilities=probe_scores["probabilities"],
            cutoff=1.0,
            enabled=False,
            include_fit_targets=True,
        )
        m20_after = state_sha256(m20.state_dict())
        if m20_after != m20_before:
            raise RuntimeError("released M20 changed during V3 probe")
        torch.cuda.synchronize(device)
        payload = {
            "schema": "ev-uav-h2-atomic-component-deletion-eight-step-probe-v3",
            "created_utc": utc_now(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "fit_source": fit_source,
            "fit_group": fit_group,
            "g2_held_array_read": False,
            "optimizer_steps": result["step_count"],
            "component_count": result["dataset_component_count"],
            "component_class_counts": {
                "target_bearing": int(np.count_nonzero(source.pure_fp_targets == 0)),
                "pure_fp": int(np.count_nonzero(source.pure_fp_targets == 1)),
            },
            "immutable_input_artifact": feature_artifact,
            "immutable_score_and_embedding_artifact": score_artifact,
            "all_step_diagnostics": result["records"],
            "elapsed_seconds": time.perf_counter() - started,
            "training_seconds": result["elapsed_seconds"],
            "peak_cuda_mib": torch.cuda.max_memory_allocated(0) / (1024.0 ** 2),
            "m20_state_sha256_before": m20_before,
            "m20_state_sha256_after": m20_after,
            "validation_or_test_read": False,
        }
        write_json_exclusive(result_path, payload)
        del result, source, m20
        torch.cuda.empty_cache()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_train_g2(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    fold_root = OUTPUT_ROOT / "formal_training" / "hold_g2"
    checkpoint_path = fold_root / "final_atomic_component_models.pt"
    result_path = fold_root / "training_result.json"
    if fold_root.exists() or checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite formal V3 G2 output")
    with gpu_run_lock("h2_atomic_component_deletion_v3_train_g2"):
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        m20, _ = build_released_m20(device)
        m20_before = state_sha256(m20.state_dict())
        cfg, _ = _build_c00(protocol)
        prepared_by_group = {}
        immutable_fit_input_artifacts = {}
        for group_id in protocol["formal_g2"]["fit_groups"]:
            prepared_by_group[group_id] = []
            for source_name in protocol["source_groups"][group_id]["sources"]:
                source = prepare_source(
                    protocol, m20, cfg, source_name, device, include_truth=True
                )
                prepared_by_group[group_id].append(source)
                immutable_fit_input_artifacts[source_name] = persist_source_feature_artifact(
                    source,
                    fold_root / "immutable_fit_input" / (Path(source_name).stem + ".npz"),
                )
                torch.cuda.empty_cache()
                print("fit source prepared:", source_name, flush=True)
        models = {}
        training_results = {}
        for group_id in protocol["formal_g2"]["fit_groups"]:
            result = train_group_model(
                protocol, group_id, prepared_by_group[group_id], device
            )
            models[group_id] = result["model"]
            training_results[group_id] = result
            print("fit group trained:", group_id, flush=True)
        calibration = nested_fit_calibration(
            protocol, models, prepared_by_group, device
        )
        score_record_by_source = calibration.pop("score_record_by_source")
        immutable_inner_oof_score_artifacts = {}
        fit_groups = tuple(protocol["formal_g2"]["fit_groups"])
        left, right = fit_groups
        for group_id in fit_groups:
            scorer_group = right if group_id == left else left
            for source in prepared_by_group[group_id]:
                record = score_record_by_source[source.source_name]
                immutable_inner_oof_score_artifacts[source.source_name] = (
                    persist_component_score_artifact(
                        fold_root
                        / "immutable_inner_oof_scores"
                        / (Path(source.source_name).stem + ".npz"),
                        source,
                        model_group_ids=[scorer_group],
                        score_records=[record],
                        consensus_probabilities=record["probabilities"],
                        cutoff=calibration["cutoff"],
                        enabled=calibration["enabled"],
                        include_fit_targets=True,
                    )
                )
        m20_after = state_sha256(m20.state_dict())
        if m20_after != m20_before:
            raise RuntimeError("released M20 changed during formal V3 G2 fit")
        torch.cuda.synchronize(device)
        checkpoint = {
            "schema": "ev-uav-h2-atomic-component-deletion-checkpoint-v3",
            "created_utc": utc_now(),
            "fold_id": "hold_g2",
            "fit_groups": protocol["formal_g2"]["fit_groups"],
            "fit_sources": [
                source_name
                for group_id in protocol["formal_g2"]["fit_groups"]
                for source_name in protocol["source_groups"][group_id]["sources"]
            ],
            "g2_held_array_read": False,
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "released_m20_sha256": sha256_file(M20_PATH),
            "released_m20_state_sha256": m20_after,
            "architecture": protocol["architecture"],
            "training": protocol["training"],
            "strict_safe_cutoff": calibration["cutoff"],
            "deletion_enabled": calibration["enabled"],
            "inner_oof_calibration": calibration,
            "immutable_fit_input_artifacts": immutable_fit_input_artifacts,
            "immutable_inner_oof_score_artifacts": immutable_inner_oof_score_artifacts,
            "group_model_state_dicts": {
                group_id: OrderedDict(
                    (name, value.detach().cpu())
                    for name, value in models[group_id].state_dict().items()
                )
                for group_id in protocol["formal_g2"]["fit_groups"]
            },
        }
        save_torch_exclusive(checkpoint_path, checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        training_payload = {
            "schema": "ev-uav-h2-atomic-component-deletion-training-result-v3",
            "created_utc": utc_now(),
            "fold_id": "hold_g2",
            "fit_groups": protocol["formal_g2"]["fit_groups"],
            "g2_held_array_read": False,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "strict_safe_cutoff": calibration["cutoff"],
            "deletion_enabled": calibration["enabled"],
            "inner_oof_calibration": calibration,
            "immutable_fit_input_artifacts": immutable_fit_input_artifacts,
            "immutable_inner_oof_score_artifacts": immutable_inner_oof_score_artifacts,
            "group_training": {
                group_id: {
                    "optimizer_steps": training_results[group_id]["step_count"],
                    "component_count": training_results[group_id]["dataset_component_count"],
                    "elapsed_seconds": training_results[group_id]["elapsed_seconds"],
                    "all_step_diagnostics": training_results[group_id]["records"],
                }
                for group_id in protocol["formal_g2"]["fit_groups"]
            },
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_mib": torch.cuda.max_memory_allocated(0) / (1024.0 ** 2),
            "m20_state_sha256_before": m20_before,
            "m20_state_sha256_after": m20_after,
            "validation_or_test_read": False,
        }
        write_json_exclusive(result_path, training_payload)
        del models, training_results, prepared_by_group, m20
        torch.cuda.empty_cache()
    print(json.dumps(training_payload, indent=2, ensure_ascii=False))


def run_evaluate_g2(args):
    require_gpu_authorization(args)
    protocol = load_protocol()
    checkpoint_path = (
        OUTPUT_ROOT / "formal_training" / "hold_g2" / "final_atomic_component_models.pt"
    )
    training_result_path = (
        OUTPUT_ROOT / "formal_training" / "hold_g2" / "training_result.json"
    )
    result_path = (
        OUTPUT_ROOT / "held_train_evaluation" / "hold_g2" / "paired_evaluation.json"
    )
    if result_path.exists() or result_path.parent.exists():
        raise FileExistsError("refusing to overwrite V3 G2 held evaluation")
    if not checkpoint_path.is_file() or not training_result_path.is_file():
        raise FileNotFoundError("formal V3 G2 checkpoint and training receipt are required")
    with training_result_path.open("r", encoding="utf-8") as stream:
        training_receipt = json.load(stream)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if training_receipt.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("training receipt/checkpoint hash mismatch")
    if training_receipt.get("g2_held_array_read") is not False:
        raise RuntimeError("training receipt does not prove G2 isolation")
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    if (
        checkpoint.get("schema") != "ev-uav-h2-atomic-component-deletion-checkpoint-v3"
        or checkpoint.get("fold_id") != "hold_g2"
        or checkpoint.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
        or checkpoint.get("g2_held_array_read") is not False
    ):
        raise RuntimeError("formal V3 checkpoint provenance mismatch")

    with gpu_run_lock("h2_atomic_component_deletion_v3_evaluate_g2"):
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        m20, _ = build_released_m20(device)
        m20_before = state_sha256(m20.state_dict())
        cfg, c00 = _build_c00(protocol)
        models = {}
        for group_id in protocol["formal_g2"]["fit_groups"]:
            model = _new_component_model(protocol, device).eval()
            model.load_state_dict(
                checkpoint["group_model_state_dicts"][group_id], strict=True
            )
            models[group_id] = model
        cutoff = float(checkpoint["strict_safe_cutoff"])
        deletion_enabled = bool(checkpoint["deletion_enabled"])
        records = []
        baseline_counts = []
        candidate_counts = []
        atomic_integrity = True
        immutable_held_input_artifacts = {}
        immutable_held_score_artifacts = {}
        fit_groups = tuple(protocol["formal_g2"]["fit_groups"])
        for source_name in protocol["source_groups"]["g2_092_094"]["sources"]:
            # This is the first command permitted to open any G2 source array.
            source = prepare_source(
                protocol, m20, cfg, source_name, device, include_truth=False
            )
            feature_artifact = persist_source_feature_artifact(
                source,
                result_path.parent
                / "immutable_held_input"
                / (Path(source_name).stem + ".npz"),
            )
            immutable_held_input_artifacts[source_name] = feature_artifact
            group_score_records = [
                score_source(models[group_id], source, protocol, device)
                for group_id in fit_groups
            ]
            consensus_probability = np.minimum.reduce(
                [record["probabilities"] for record in group_score_records]
            )
            routed = source.input_only_h2_route
            candidate_scores, edit_receipt = atomic_delete_or_identity(
                source.base_scores,
                source.component_event_indices,
                consensus_probability,
                cutoff,
                enabled=bool(deletion_enabled and routed),
            )
            score_artifact = persist_component_score_artifact(
                result_path.parent
                / "immutable_held_scores"
                / (Path(source_name).stem + ".npz"),
                source,
                model_group_ids=list(fit_groups),
                score_records=group_score_records,
                consensus_probabilities=consensus_probability,
                cutoff=cutoff,
                enabled=bool(deletion_enabled and routed),
                include_fit_targets=False,
            )
            immutable_held_score_artifacts[source_name] = score_artifact
            # Truth is attached only after all model scores and the atomic
            # candidate have been produced from input-only tensors.
            source.labels, source.target_ids = _load_truth(
                source_path(protocol, source_name)
            )
            base_count = _counts(source, source.base_scores)
            candidate_count = _counts(source, candidate_scores)
            baseline_counts.append(base_count)
            candidate_counts.append(candidate_count)
            atomic_integrity &= bool(
                edit_receipt.complete_components_only
                and edit_receipt.retained_scores_bitwise_equal
            )
            records.append(
                {
                    "source_name": source_name,
                    "source_sha256": protocol["sources"][source_name]["sha256"],
                    "event_count": source.event_count,
                    "polarity_minority_fraction": source.polarity_minority_fraction,
                    "input_only_h2_route": routed,
                    "component_count": len(source.component_event_indices),
                    "pure_fp_probability_aggregation": "minimum_across_g1_and_g3_models",
                    "atomic_edit": asdict(edit_receipt),
                    "immutable_input_artifact": feature_artifact,
                    "immutable_score_and_embedding_artifact": score_artifact,
                    "base_raw_scores_sha256": sha256_float32(source.base_raw_scores),
                    "base_post_c00_scores_sha256": sha256_float32(source.base_scores),
                    "candidate_scores_sha256": sha256_float32(candidate_scores),
                    "postprocess_stats": source.postprocess_stats,
                    "baseline": _metric_record(base_count),
                    "candidate": _metric_record(candidate_count),
                    "gates": _safety_gates(candidate_count, base_count),
                }
            )
            del source
            torch.cuda.empty_cache()
            print("held G2 paired:", source_name, flush=True)
        m20_after = state_sha256(m20.state_dict())
        if m20_after != m20_before:
            raise RuntimeError("released M20 changed during V3 G2 evaluation")
        torch.cuda.synchronize(device)
        pooled_base = _sum_counts(baseline_counts)
        pooled_candidate = _sum_counts(candidate_counts)
        pooled_base_metrics = _metrics(pooled_base)
        pooled_candidate_metrics = _metrics(pooled_candidate)
        metric_delta = {
            key: float(pooled_candidate_metrics[key] - pooled_base_metrics[key])
            for key in pooled_base_metrics
        }
        count_delta = {
            key: int(value - getattr(pooled_base, key))
            for key, value in pooled_candidate.to_dict().items()
        }
        held_gates = _safety_gates(pooled_candidate, pooled_base)
        promotion_gates = {
            "inner_g1_and_g3_grouped_oof_all_metrics_safe": bool(
                checkpoint["inner_oof_calibration"]["all_inner_safety_gates_passed"]
            ),
            "held_g2_atomic_integrity": atomic_integrity,
            **{"held_g2_" + key: value for key, value in held_gates.items()},
            "held_g2_score_gain_at_least_0_02": metric_delta["score"]
            >= float(protocol["promotion_gates"]["substantive_h2_score_gain"]),
        }
        payload = {
            "schema": "ev-uav-h2-atomic-component-deletion-held-g2-paired-evaluation-v3",
            "created_utc": utc_now(),
            "fold_id": "hold_g2",
            "held_group": "g2_092_094",
            "held_sources": protocol["source_groups"]["g2_092_094"]["sources"],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_and_cutoff_written_before_first_held_open": True,
            "strict_safe_cutoff": cutoff,
            "deletion_enabled_from_fit_only_inner_oof": deletion_enabled,
            "prediction_threshold": float(protocol["baseline"]["prediction_threshold"]),
            "effective_c00": c00,
            "records": records,
            "immutable_held_input_artifacts": immutable_held_input_artifacts,
            "immutable_held_score_artifacts": immutable_held_score_artifacts,
            "pooled_base_counts": pooled_base.to_dict(),
            "pooled_candidate_counts": pooled_candidate.to_dict(),
            "pooled_count_delta": count_delta,
            "pooled_base_metrics": pooled_base_metrics,
            "pooled_candidate_metrics": pooled_candidate_metrics,
            "pooled_metric_delta": metric_delta,
            "promotion_gates": promotion_gates,
            "promotion_passed": all(promotion_gates.values()),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_mib": torch.cuda.max_memory_allocated(0) / (1024.0 ** 2),
            "m20_state_sha256_before": m20_before,
            "m20_state_sha256_after": m20_after,
            "validation_or_test_read": False,
        }
        write_json_exclusive(result_path, payload)
        del models, m20
        torch.cuda.empty_cache()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="CPU-only protocol and unit-test audit")
    audit.set_defaults(function=run_audit)
    probe = subparsers.add_parser("probe", help="authorized eight-step fit-only GPU probe")
    probe.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true")
    probe.set_defaults(function=run_probe)
    train = subparsers.add_parser("train-g2", help="authorized fresh-G2 nested fit")
    train.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true")
    train.set_defaults(function=run_train_g2)
    evaluate = subparsers.add_parser(
        "evaluate-g2", help="authorized one-time fresh-G2 held paired evaluation"
    )
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true")
    evaluate.set_defaults(function=run_evaluate_g2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())

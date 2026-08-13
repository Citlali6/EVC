"""Submission-time inference for promoted H2 pyramid recovery V2.

The public ``apply`` method consumes only complete-input polarities, aligned
locations, a label-free temporal video, and the already released M20 post-C00
score vector.  Source names, paths, hashes, fold/component ordinals, labels,
and target ids are deliberately absent from the inference signature and from
the 96-dimensional recovery-head features.

Routing is fail-closed and happens before the expensive H2 backend is called:
only inputs with more than 200000 events and polarity minority fraction at
least 0.20 enter H2.  Every other input returns the exact input M20 ndarray
object.  H2 performs frozen Stage1, applies C00 separately to M20 and Stage1,
then makes exactly one whole-component overlay from Stage1 post-C00 to M20
post-C00.  There is no attenuation, interpolation, partial restore, or second
C00 pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset.temporal_frame import TemporalFrameVideo, build_temporal_context_frame
from model.h2_multiscale_temporal_pyramid_expert import (
    FrozenM20MultiScalePyramidAdapter,
    downsample_frozen_observations,
    fixed_multiscale_temporal_moments,
)
from model.h2_pyramid_component_recovery import H2PyramidComponentRecoveryHead
from utils.atomic_component_deletion import extract_atomic_components
from utils.h2_pyramid_component_recovery import restore_whole_components_bitwise
from utils.postprocess import ChallengePostprocessor
from utils.target_preserving_residual import (
    complete_input_polarity_minority_fraction,
    use_h2_residual_refiner,
)


PREDICTION_THRESHOLD = 0.719
TEMPORAL_COUNT = 160
TEMPORAL_BIN_SIZE = 50
CONTEXT_BINS = 5
WIDTH = 346
HEIGHT = 260
LOG_COUNT_CLIP = 4.0
INFERENCE_BATCH = 8
COMPONENT_MICROBATCH = 64
NODE_FEATURE_DIM = 96

PROMOTED_SCIENCE_PROTOCOL_SHA256 = "4c4c260b66bf4c77fb314432bd2c72432a3273917347a8f5bf943d8489933c70"
FINAL_REFIT_PROTOCOL_SHA256 = "85ff0ef0e363ae0bec59580962ab26f1de6810840d5b664c4191978f87a3eb0b"
EFFECTIVE_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
STAGE1_CHECKPOINT_SHA256 = "0629d0aa109eddc36638f0309d5f1105981a9f73a44c8b90aca390650e48b772"
RECOVERY_CHECKPOINT_SHA256 = "5fbe14b05da6a53b7a2b140752d9027e9510c0256b0a0a198bb99d6bb38a6671"
OUTER_DECISION_SHA256 = "298a8ad299a68c66987b849685fbd3993fdf1b9bef0f2bda7137b0d245a1f334"
EXPECTED_STAGE1_STATE_SHA256 = "fdb0a3d3de7b9554573a9d759012fc74b36c0ac54fc9295ed7c84b42db60ffba"
EXPECTED_RECOVERY_STATE_SHA256 = "3039bd835548beb7c5fb53b131a6a4a4a50cae40e3873eb27a8af8e1465a3024"
EXPECTED_RELEASED_M20_STATE_SHA256 = "feb234e530688a11865e0d49b58a9f54806f69ea63a9de3e01e8b9f714a6113d"
EXPECTED_FIT_SOURCES = tuple(
    "train_{:03d}.npz".format(value) for value in range(92, 99)
)
EXPECTED_ALL11_SOURCES = tuple(
    "train_{:03d}.npz".format(value) for value in range(88, 99)
)
EXPECTED_RELEASED_M20_CHECKPOINT_SHA256 = (
    "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
)

EVC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EVC_ROOT.parent
EXPERIMENT_ROOT = (
    WORKSPACE_ROOT / "experiments" / "20260811_h2_pyramid_component_recovery_v2"
)
DEFAULT_STAGE1_CHECKPOINT = (
    EXPERIMENT_ROOT / "formal_stage1" / "fresh_hold_g1" / "final_expert.pt"
)
DEFAULT_RECOVERY_CHECKPOINT = (
    EXPERIMENT_ROOT / "nested_recovery" / "hold_g1" / "final_recovery_head.pt"
)
DEFAULT_OUTER_DECISION = (
    EXPERIMENT_ROOT
    / "held_train_evaluation"
    / "fresh_hold_g1"
    / "branch_decision.json"
)


@dataclass(frozen=True)
class Stage1ComponentPayload:
    """Truth-free tensors produced by frozen Stage1 for one complete input."""

    internal_m20_post_scores: np.ndarray
    stage1_post_scores: np.ndarray
    components: tuple[np.ndarray, ...]
    node_features: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class H2V2InferenceReceipt:
    event_count: int
    polarity_minority_fraction: float
    routed_to_h2: bool
    disagreement_component_count: int
    restored_component_count: int
    restored_event_count: int
    recovery_cutoff: float | None
    non_h2_m20_object_identity: bool
    internal_m20_post_bitwise_verified: bool
    whole_components_only: bool
    second_c00_applied: bool


@dataclass(frozen=True)
class H2V2InferenceResult:
    scores: np.ndarray
    receipt: H2V2InferenceReceipt


@dataclass(frozen=True)
class FrozenCheckpointPayloads:
    stage1_state_dict: dict[str, torch.Tensor]
    recovery_state_dict: dict[str, torch.Tensor]
    recovery_cutoff: float
    released_m20_state_sha256: str
    effective_c00_sha256: str | None


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_float32_vector(values, description: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError("{} must be a numpy ndarray".format(description))
    if values.ndim != 1 or values.dtype != np.float32:
        raise ValueError("{} must be a one-dimensional float32 vector".format(description))
    if not np.isfinite(values).all():
        raise ValueError("{} must be finite".format(description))
    return values


def _bitwise_float32_equal(left, right) -> bool:
    left = _require_float32_vector(left, "left scores")
    right = _require_float32_vector(right, "right scores")
    return bool(
        left.shape == right.shape
        and np.array_equal(left.view(np.uint32), right.view(np.uint32))
    )


def use_h2_pyramid_recovery_v2(event_count, polarities) -> bool:
    """Frozen, source-free H2 route: count > 200000 and minority >= 0.20."""

    return use_h2_residual_refiner(event_count, polarities)


def apply_atomic_stage2(
    stage1_post_scores,
    released_m20_post_scores,
    components,
    component_probabilities,
    recovery_cutoff,
):
    """Apply the only permitted Stage2 action and return scores plus decisions."""

    stage1 = _require_float32_vector(stage1_post_scores, "Stage1 post-C00 scores")
    m20 = _require_float32_vector(released_m20_post_scores, "released M20 post-C00 scores")
    if stage1.shape != m20.shape:
        raise ValueError("Stage1 and released M20 scores must align")
    components = tuple(components)
    probabilities = np.asarray(component_probabilities, dtype=np.float64).reshape(-1)
    if probabilities.size != len(components) or not np.isfinite(probabilities).all():
        raise ValueError("one finite recovery probability is required per component")
    cutoff = float(recovery_cutoff)
    if not np.isfinite(cutoff):
        raise ValueError("recovery cutoff must be finite")
    decisions = np.asarray(probabilities >= cutoff, dtype=np.bool_)
    output = restore_whole_components_bitwise(stage1, m20, components, decisions)
    return output, decisions


def _verify_sidecar(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError("frozen artifact hash changed: {}".format(path))
    sidecar = Path(str(path) + ".sha256")
    tokens = sidecar.read_text(encoding="ascii").split()
    if len(tokens) != 2 or tokens[0] != actual or tokens[1] != path.name:
        raise RuntimeError("frozen artifact sidecar mismatch: {}".format(path))


def _verify_outer_promotion(path: Path = DEFAULT_OUTER_DECISION) -> dict:
    _verify_sidecar(path, OUTER_DECISION_SHA256)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != "ev-uav-h2-pyramid-recovery-v2-held-G1-branch-decision-v1":
        raise RuntimeError("unexpected V2 outer decision schema")
    if payload.get("outer_passed") is not True or not all(
        payload.get("outer_gates", {}).values()
    ):
        raise RuntimeError("V2 was not promoted by every fresh held-G1 gate")
    if payload.get("decision") != "promote_after_fresh_G1_but_stop_no_other_fold_or_tuning":
        raise RuntimeError("unexpected V2 outer decision")
    if payload.get("protocol_sha256") != PROMOTED_SCIENCE_PROTOCOL_SHA256:
        raise RuntimeError("V2 outer decision protocol binding changed")
    if payload.get("validation_or_test_read") is not False:
        raise RuntimeError("V2 promotion does not prove val/test unread")
    return payload


def _validate_checkpoint_metadata(stage1, recovery) -> None:
    if stage1.get("schema") != "ev-uav-h2-pyramid-v2-fresh-stage1-checkpoint-v1":
        raise RuntimeError("unexpected frozen Stage1 checkpoint schema")
    if recovery.get("schema") != "ev-uav-h2-pyramid-recovery-v2-final-head-checkpoint-v1":
        raise RuntimeError("unexpected frozen recovery checkpoint schema")
    for payload in (stage1, recovery):
        if payload.get("protocol_sha256") != PROMOTED_SCIENCE_PROTOCOL_SHA256:
            raise RuntimeError("frozen checkpoint protocol binding changed")
        if tuple(payload.get("fit_sources", ())) != EXPECTED_FIT_SOURCES:
            raise RuntimeError("frozen checkpoint fit set changed")
        if payload.get("validation_or_test_read") is not False:
            raise RuntimeError("frozen checkpoint does not prove val/test unread")
        if payload.get("fresh_initialization") is not True:
            raise RuntimeError("frozen checkpoint is not a fresh fit")
    if stage1.get("V1_checkpoint_reused") is not False:
        raise RuntimeError("V1 Stage1 reuse is forbidden")
    if int(stage1.get("optimizer_steps", -1)) != 56:
        raise RuntimeError("frozen Stage1 step count changed")
    if int(recovery.get("optimizer_steps", -1)) != 56:
        raise RuntimeError("frozen recovery step count changed")
    stage1_state_hash = state_sha256(stage1["expert_state_dict"])
    recovery_state_hash = state_sha256(recovery["head_state_dict"])
    if stage1_state_hash != stage1.get("expert_state_sha256"):
        raise RuntimeError("Stage1 state hash/metadata mismatch")
    if recovery_state_hash != recovery.get("head_state_sha256"):
        raise RuntimeError("recovery state hash/metadata mismatch")
    if stage1_state_hash != EXPECTED_STAGE1_STATE_SHA256:
        raise RuntimeError("unexpected promoted Stage1 state")
    if recovery_state_hash != EXPECTED_RECOVERY_STATE_SHA256:
        raise RuntimeError("unexpected promoted recovery state")
    if stage1.get("released_m20_state_sha256") != EXPECTED_RELEASED_M20_STATE_SHA256:
        raise RuntimeError("frozen Stage1 released-M20 binding changed")


def load_frozen_checkpoint_payloads(
    *,
    stage1_checkpoint_path=DEFAULT_STAGE1_CHECKPOINT,
    recovery_checkpoint_path=DEFAULT_RECOVERY_CHECKPOINT,
    outer_decision_path=DEFAULT_OUTER_DECISION,
) -> FrozenCheckpointPayloads:
    """Load the two promoted formal artifacts after complete CPU hash checks."""

    stage1_path = Path(stage1_checkpoint_path)
    recovery_path = Path(recovery_checkpoint_path)
    outer = _verify_outer_promotion(Path(outer_decision_path))
    _verify_sidecar(stage1_path, STAGE1_CHECKPOINT_SHA256)
    _verify_sidecar(recovery_path, RECOVERY_CHECKPOINT_SHA256)
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_metadata(stage1, recovery)
    if stage1.get("runner_sha256") != recovery.get("runner_sha256"):
        raise RuntimeError("Stage1/recovery runner bindings differ")
    if stage1.get("runner_sha256") != outer.get("runner_sha256"):
        raise RuntimeError("checkpoint/outer-decision runner bindings differ")
    cutoff = float(recovery["OOF_cutoff"])
    if not np.isfinite(cutoff):
        raise RuntimeError("fitted recovery cutoff is non-finite")
    return FrozenCheckpointPayloads(
        stage1_state_dict=stage1["expert_state_dict"],
        recovery_state_dict=recovery["head_state_dict"],
        recovery_cutoff=cutoff,
        released_m20_state_sha256=stage1["released_m20_state_sha256"],
        effective_c00_sha256=EFFECTIVE_C00_SHA256,
    )


def _first_present(mapping, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise RuntimeError("final package is missing {}".format("/".join(keys)))


def _is_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_final_package_payload(
    final_package_path,
    *,
    verify_wrapper_hash=True,
    outer_decision_path=None,
) -> FrozenCheckpointPayloads:
    """Load the all11 package written after the promoted wrapper was frozen."""

    package_path = Path(final_package_path)
    if outer_decision_path is not None:
        _verify_outer_promotion(Path(outer_decision_path))
    sidecar = Path(str(package_path) + ".sha256")
    actual_package_hash = sha256_file(package_path)
    tokens = sidecar.read_text(encoding="ascii").split()
    if len(tokens) != 2 or tokens[0] != actual_package_hash or tokens[1] != package_path.name:
        raise RuntimeError("final package sidecar mismatch")
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    if package.get("schema") != "ev-uav-h2-pyramid-recovery-v2-all11-final-package-v1":
        raise RuntimeError("unexpected H2 V2 final package schema")
    if package.get("protocol_sha256") != FINAL_REFIT_PROTOCOL_SHA256:
        raise RuntimeError("final package protocol binding changed")
    if package.get("outer_decision_sha256") != OUTER_DECISION_SHA256:
        raise RuntimeError("final package outer-promotion binding changed")
    if package.get("validation_or_test_read") is not False:
        raise RuntimeError("final package does not prove val/test unread")
    if float(package.get("prediction_threshold", float("nan"))) != PREDICTION_THRESHOLD:
        raise RuntimeError("final package decision threshold changed")
    route = package.get("route", {})
    if not isinstance(route, dict):
        raise RuntimeError("final package route must be a mapping")
    count_cutoff = _first_present(route, "event_count_cutoff_exclusive", "event_count_cutoff")
    minority_cutoff = _first_present(route, "polarity_minority_cutoff_inclusive", "polarity_minority_cutoff")
    if int(count_cutoff) != 200000 or float(minority_cutoff) != 0.20:
        raise RuntimeError("final package H2 route changed")
    if verify_wrapper_hash:
        expected_wrapper_hash = package.get("inference_wrapper_sha256")
        if expected_wrapper_hash != sha256_file(Path(__file__)):
            raise RuntimeError("final package inference-wrapper binding changed")
    runner_hash = package.get("runner_sha256")
    if not _is_sha256(runner_hash):
        raise RuntimeError("final package runner SHA-256 is invalid")
    if tuple(package.get("fit_sources", ())) != EXPECTED_ALL11_SOURCES:
        raise RuntimeError("final package all11 fit set changed")
    if package.get("released_M20_checkpoint_sha256") != (
        EXPECTED_RELEASED_M20_CHECKPOINT_SHA256
    ):
        raise RuntimeError("final package released-M20 checkpoint binding changed")
    if package.get("promoted_outer_weights_or_cutoff_reused") is not False:
        raise RuntimeError("final package reused promoted outer weights or cutoff")
    if package.get("optimizer_state_included") is not False:
        raise RuntimeError("deployment package unexpectedly includes optimizer state")
    fresh = package.get("fresh_initialization", {})
    if fresh != {"Stage1": True, "OOF_heads": True, "final_head": True}:
        raise RuntimeError("final package fresh-initialization receipt changed")
    dependencies = package.get("execution_dependency_sha256")
    if not isinstance(dependencies, dict) or not dependencies:
        raise RuntimeError("final package execution dependency manifest is missing")
    for relative_path, expected_hash in dependencies.items():
        if not isinstance(relative_path, str) or not _is_sha256(expected_hash):
            raise RuntimeError("invalid final package execution dependency receipt")
        dependency_path = EVC_ROOT / relative_path
        if sha256_file(dependency_path) != expected_hash:
            raise RuntimeError("final package dependency changed: {}".format(relative_path))
    stage1_state = package["fresh_Stage1_state_dict"]
    recovery_state = package["fresh_final_recovery_head_state_dict"]
    stage1_hash = state_sha256(stage1_state)
    recovery_hash = state_sha256(recovery_state)
    packaged_stage1_hash = package.get("fresh_Stage1_state_sha256")
    packaged_recovery_hash = package.get("fresh_final_recovery_head_state_sha256")
    if packaged_stage1_hash != stage1_hash or packaged_recovery_hash != recovery_hash:
        raise RuntimeError("final package state receipt mismatch")
    m20_hash = package.get(
        "released_m20_state_sha256", EXPECTED_RELEASED_M20_STATE_SHA256
    )
    if m20_hash != EXPECTED_RELEASED_M20_STATE_SHA256:
        raise RuntimeError("final package released-M20 binding changed")
    cutoff = float(package["recovery_cutoff"])
    if not np.isfinite(cutoff):
        raise RuntimeError("final package recovery cutoff is non-finite")
    effective_c00_sha = package.get("effective_C00_sha256")
    if effective_c00_sha != EFFECTIVE_C00_SHA256:
        raise RuntimeError("final package effective-C00 binding changed")
    return FrozenCheckpointPayloads(
        stage1_state_dict=stage1_state,
        recovery_state_dict=recovery_state,
        recovery_cutoff=cutoff,
        released_m20_state_sha256=m20_hash,
        effective_c00_sha256=effective_c00_sha,
    )


def _component_node_geometry(component, video, stage1_post_scores):
    """Exact 10-value geometry used to train the promoted formal head."""

    component = np.asarray(component, dtype=np.int64)
    locations = video.locations[component].astype(np.float64, copy=False)
    temporal_bins = np.floor_divide(locations[:, 2].astype(np.int64), TEMPORAL_BIN_SIZE)
    unique_bins = np.unique(temporal_bins)
    component_centroid = locations[:, :2].mean(axis=0)
    component_width = max(float(np.ptp(locations[:, 0]) + 1.0), 1.0)
    component_height = max(float(np.ptp(locations[:, 1]) + 1.0), 1.0)
    midpoint = 0.5 * (float(unique_bins[0]) + float(unique_bins[-1]))
    span = max(float(unique_bins[-1] - unique_bins[0]), 1.0)
    output = []
    previous = None
    for temporal_bin in unique_bins:
        indices = component[temporal_bins == temporal_bin]
        values = video.locations[indices].astype(np.float64, copy=False)
        centroid = values[:, :2].mean(axis=0)
        delta = np.zeros(2, dtype=np.float64) if previous is None else centroid - previous
        previous = centroid
        geometry = np.asarray(
            (
                np.log1p(indices.size),
                indices.size / component.size,
                (float(temporal_bin) - midpoint) / span,
                (centroid[0] - component_centroid[0]) / component_width,
                (centroid[1] - component_centroid[1]) / component_height,
                float(np.ptp(values[:, 0]) + 1.0) / component_width,
                float(np.ptp(values[:, 1]) + 1.0) / component_height,
                delta[0] / float(WIDTH),
                delta[1] / float(HEIGHT),
                float(
                    np.mean(
                        stage1_post_scores[indices]
                        >= np.float32(PREDICTION_THRESHOLD)
                    )
                ),
            ),
            dtype=np.float32,
        )
        output.append((int(temporal_bin), indices, geometry))
    return tuple(output)


class _FrozenStage1Executor:
    def __init__(self, adapter, c00_cfg, device):
        self.adapter = adapter
        self.c00_cfg = c00_cfg
        self.device = torch.device(device)

    def _frames(self, video, bins):
        values = np.stack(
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
        return torch.from_numpy(values).to(device=self.device, dtype=torch.float32)

    def _apply_c00(self, scores, locations4):
        values = torch.from_numpy(
            np.asarray(scores, dtype=np.float32).copy()
        )
        locations = torch.from_numpy(
            np.asarray(locations4, dtype=np.int64)
        ).long()
        processed, _ = ChallengePostprocessor.from_cfg(
            self.c00_cfg,
            PREDICTION_THRESHOLD,
            event_count=len(scores),
        ).apply(values, locations)
        output = processed.detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(output).all():
            raise RuntimeError("C00 produced non-finite scores")
        return output

    def _validate_input(self, video, locations4):
        if not isinstance(video, TemporalFrameVideo):
            raise TypeError("video must be a label-free TemporalFrameVideo")
        if len(video.event_indices_by_bin) != TEMPORAL_COUNT:
            raise ValueError("H2 V2 requires one complete T160 stream")
        locations = np.asarray(locations4)
        event_count = int(video.locations.shape[0])
        if locations.shape != (event_count, 4):
            raise ValueError("locations4 must have shape [E,4] batch/x/y/t")
        if locations.dtype.kind not in "iu":
            raise ValueError("locations4 must be integer-valued")
        if not np.array_equal(locations[:, 1:4], video.locations):
            raise ValueError("locations4 and temporal video event order differ")
        return locations.astype(np.int64, copy=False)

    def __call__(self, video, locations4):
        locations4 = self._validate_input(video, locations4)
        event_count = int(video.locations.shape[0])
        memory_parts = []
        with torch.no_grad():
            for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
                stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
                frames = self._frames(video, range(start, stop))
                memory_parts.append(self.adapter.released_m20.encode_bottleneck(frames))
            bottlenecks = torch.cat(memory_parts, dim=0)
            memory = self.adapter.released_m20.temporal_residual(bottlenecks)
            del memory_parts, bottlenecks

            observations = []
            base_raw = np.empty(event_count, dtype=np.float32)
            for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
                stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
                frames = self._frames(video, range(start, stop))
                decoder, base_logits, centre = self.adapter.decode_frozen_features(
                    frames, memory[start:stop]
                )
                pooled = downsample_frozen_observations(
                    decoder.unsqueeze(0),
                    base_logits.unsqueeze(0),
                    centre.unsqueeze(0),
                ).squeeze(0)
                observations.append(pooled.to(device="cpu", dtype=torch.float16))
                probabilities = torch.sigmoid(base_logits).squeeze(1).cpu().numpy()
                for temporal_bin in range(start, stop):
                    indices = video.event_indices_by_bin[temporal_bin]
                    if indices.size:
                        xy = video.locations[indices]
                        base_raw[indices] = probabilities[
                            temporal_bin - start, xy[:, 1], xy[:, 0]
                        ]
            observations_cpu = torch.cat(observations, dim=0).contiguous()
            observations_device = observations_cpu.unsqueeze(0).to(
                device=self.device, dtype=torch.float32
            )
            summaries_device = fixed_multiscale_temporal_moments(observations_device)
            summaries = tuple(
                value.squeeze(0).to(device="cpu", dtype=torch.float16).contiguous()
                for value in summaries_device
            )
            del observations, observations_cpu, observations_device, summaries_device

            stage1_raw = np.empty(event_count, dtype=np.float32)
            for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
                stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
                frames = self._frames(video, range(start, stop))
                decoder, base_logits, centre = self.adapter.decode_frozen_features(
                    frames, memory[start:stop]
                )
                summary_views = tuple(
                    value[start:stop].to(device=self.device, dtype=torch.float32)
                    for value in summaries
                )
                refined = self.adapter.expert(
                    decoder.unsqueeze(0),
                    base_logits.unsqueeze(0),
                    centre.unsqueeze(0),
                    tuple(value.unsqueeze(0) for value in summary_views),
                ).squeeze(0)
                probabilities = torch.sigmoid(refined).squeeze(1).cpu().numpy()
                for temporal_bin in range(start, stop):
                    indices = video.event_indices_by_bin[temporal_bin]
                    if indices.size:
                        xy = video.locations[indices]
                        stage1_raw[indices] = probabilities[
                            temporal_bin - start, xy[:, 1], xy[:, 0]
                        ]

        if not np.isfinite(base_raw).all() or not np.isfinite(stage1_raw).all():
            raise RuntimeError("frozen Stage1 produced non-finite event scores")
        base_post = self._apply_c00(base_raw, locations4)
        stage1_post = self._apply_c00(stage1_raw, locations4)
        all_components = extract_atomic_components(
            base_post,
            locations4,
            PREDICTION_THRESHOLD,
            spatial_radius=2,
            temporal_bin_size=TEMPORAL_BIN_SIZE,
            temporal_radius_bins=1,
        ).event_indices
        components = tuple(
            np.asarray(component, dtype=np.int64)
            for component in all_components
            if np.any(stage1_post[component] < np.float32(PREDICTION_THRESHOLD))
        )
        if not components:
            return Stage1ComponentPayload(
                internal_m20_post_scores=base_post,
                stage1_post_scores=stage1_post,
                components=(),
                node_features=(),
            )

        node_specs = tuple(
            _component_node_geometry(component, video, stage1_post)
            for component in components
        )
        node_lookup = {}
        for component_index, nodes in enumerate(node_specs):
            for node_index, (temporal_bin, indices, geometry) in enumerate(nodes):
                node_lookup.setdefault(temporal_bin, []).append(
                    (component_index, node_index, indices, geometry)
                )
        node_dense = [[None for _ in nodes] for nodes in node_specs]
        with torch.no_grad():
            for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
                stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
                frames = self._frames(video, range(start, stop))
                decoder, base_logits, centre = self.adapter.decode_frozen_features(
                    frames, memory[start:stop]
                )
                summary_views = tuple(
                    value[start:stop].to(device=self.device, dtype=torch.float32)
                    for value in summaries
                )
                parts = self.adapter.expert(
                    decoder.unsqueeze(0),
                    base_logits.unsqueeze(0),
                    centre.unsqueeze(0),
                    tuple(value.unsqueeze(0) for value in summary_views),
                    return_parts=True,
                )
                encoded_scales = []
                for scale_index, summary in enumerate(summary_views):
                    encoded = self.adapter.expert.scale_encoder(summary)
                    encoded = encoded + self.adapter.expert.scale_tokens[
                        scale_index
                    ].view(1, -1, 1, 1)
                    encoded_scales.append(
                        F.interpolate(
                            encoded,
                            size=decoder.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                for temporal_bin in range(start, stop):
                    local = temporal_bin - start
                    for component_index, node_index, indices, geometry in node_lookup.get(
                        temporal_bin, ()
                    ):
                        xy = video.locations[indices]
                        xs = torch.from_numpy(xy[:, 0]).to(
                            device=self.device, dtype=torch.long
                        )
                        ys = torch.from_numpy(xy[:, 1]).to(
                            device=self.device, dtype=torch.long
                        )
                        decoder_mean = decoder[local, :, ys, xs].mean(dim=1)
                        scale_means = [
                            encoded[local, :, ys, xs].mean(dim=1)
                            for encoded in encoded_scales
                        ]
                        base_mean = base_logits[local, 0, ys, xs].mean().reshape(1)
                        stage1_mean = parts.refined_logits[
                            0, local, 0, ys, xs
                        ].mean().reshape(1)
                        centre_mean = centre[local, :, ys, xs].mean(dim=1)
                        dense = torch.cat(
                            (
                                decoder_mean,
                                *scale_means,
                                base_mean,
                                stage1_mean,
                                (stage1_mean - base_mean).reshape(1),
                                centre_mean,
                            )
                        ).detach().cpu().float().numpy()
                        node_dense[component_index][node_index] = np.concatenate(
                            (dense, geometry), axis=0
                        ).astype(np.float32, copy=False)
        output_nodes = []
        for values in node_dense:
            if any(value is None for value in values):
                raise RuntimeError("not every recovery node received frozen features")
            joined = np.stack(values).astype(np.float32, copy=False)
            if joined.ndim != 2 or joined.shape[1] != NODE_FEATURE_DIM:
                raise RuntimeError("recovery node feature schema changed")
            if joined.shape[0] > TEMPORAL_COUNT or not np.isfinite(joined).all():
                raise RuntimeError("invalid recovery node feature matrix")
            output_nodes.append(joined)
        return Stage1ComponentPayload(
            internal_m20_post_scores=base_post,
            stage1_post_scores=stage1_post,
            components=components,
            node_features=tuple(output_nodes),
        )


class _FrozenRecoveryPredictor:
    def __init__(self, head, device):
        self.head = head
        self.device = torch.device(device)

    def __call__(self, node_features):
        values = tuple(np.asarray(value, dtype=np.float32) for value in node_features)
        if not values:
            return np.zeros(0, dtype=np.float64)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), COMPONENT_MICROBATCH):
                chunk = values[start : start + COMPONENT_MICROBATCH]
                maximum = max(value.shape[0] for value in chunk)
                batch = np.zeros((len(chunk), maximum, NODE_FEATURE_DIM), dtype=np.float32)
                mask = np.zeros((len(chunk), maximum), dtype=np.bool_)
                for row, value in enumerate(chunk):
                    if value.ndim != 2 or value.shape[1] != NODE_FEATURE_DIM:
                        raise ValueError("node features must have shape [N,96]")
                    if value.shape[0] < 1 or value.shape[0] > TEMPORAL_COUNT:
                        raise ValueError("node sequence length must be in [1,160]")
                    batch[row, : value.shape[0]] = value
                    mask[row, : value.shape[0]] = True
                feature_tensor = torch.from_numpy(batch).to(
                    device=self.device, dtype=torch.float32
                )
                mask_tensor = torch.from_numpy(mask).to(
                    device=self.device, dtype=torch.bool
                )
                probabilities = torch.sigmoid(
                    self.head(feature_tensor, mask_tensor)
                ).cpu().numpy().astype(np.float64)
                outputs.append(probabilities)
        return np.concatenate(outputs)


class H2PyramidRecoveryV2Inference:
    """Source-free router around frozen Stage1 and atomic Stage2 recovery."""

    def __init__(
        self,
        stage1_executor: Callable,
        component_probability_predictor: Callable,
        recovery_cutoff: float,
    ):
        if not callable(stage1_executor) or not callable(component_probability_predictor):
            raise TypeError("Stage1 executor and probability predictor must be callable")
        cutoff = float(recovery_cutoff)
        if not np.isfinite(cutoff):
            raise ValueError("recovery cutoff must be finite")
        self._stage1_executor = stage1_executor
        self._component_probability_predictor = component_probability_predictor
        self.recovery_cutoff = cutoff

    @classmethod
    def _from_payloads(cls, released_m20, c00_cfg, device, payloads):
        from crossfit_component_reranker import sha256_json, validate_c00_config

        effective_c00 = validate_c00_config(c00_cfg, PREDICTION_THRESHOLD)
        actual_c00_sha256 = sha256_json(effective_c00)
        if actual_c00_sha256 != payloads.effective_c00_sha256:
            raise RuntimeError("provided C00 config does not match the frozen package")
        if state_sha256(released_m20.state_dict()) != payloads.released_m20_state_sha256:
            raise RuntimeError("provided released M20 state does not match the promoted fit")
        adapter = FrozenM20MultiScalePyramidAdapter(
            released_m20, context_bins=CONTEXT_BINS
        ).to(device)
        adapter.expert.load_state_dict(payloads.stage1_state_dict, strict=True)
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        head = H2PyramidComponentRecoveryHead().to(device)
        head.load_state_dict(payloads.recovery_state_dict, strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        return cls(
            _FrozenStage1Executor(adapter, c00_cfg, device),
            _FrozenRecoveryPredictor(head, device),
            payloads.recovery_cutoff,
        )

    @classmethod
    def from_final_package(
        cls,
        released_m20,
        c00_cfg,
        device,
        final_package_path,
        *,
        verify_wrapper_hash=True,
        outer_decision_path=None,
    ):
        payloads = load_final_package_payload(
            final_package_path,
            verify_wrapper_hash=verify_wrapper_hash,
            outer_decision_path=outer_decision_path,
        )
        return cls._from_payloads(released_m20, c00_cfg, device, payloads)

    @classmethod
    def from_frozen_checkpoints(
        cls,
        released_m20,
        c00_cfg,
        device,
        *,
        stage1_checkpoint_path=DEFAULT_STAGE1_CHECKPOINT,
        recovery_checkpoint_path=DEFAULT_RECOVERY_CHECKPOINT,
        outer_decision_path=DEFAULT_OUTER_DECISION,
    ):
        payloads = load_frozen_checkpoint_payloads(
            stage1_checkpoint_path=stage1_checkpoint_path,
            recovery_checkpoint_path=recovery_checkpoint_path,
            outer_decision_path=outer_decision_path,
        )
        return cls._from_payloads(released_m20, c00_cfg, device, payloads)

    def apply(self, released_m20_post_scores, video, polarities, locations4):
        """Route and infer one complete input without any source/truth argument."""

        m20 = _require_float32_vector(
            released_m20_post_scores, "released M20 post-C00 scores"
        )
        polarity_values = np.asarray(polarities)
        if polarity_values.ndim != 1 or polarity_values.size != m20.size:
            raise ValueError("complete-input polarities must align with M20 scores")
        minority = complete_input_polarity_minority_fraction(polarity_values)
        routed = use_h2_pyramid_recovery_v2(m20.size, polarity_values)
        if not routed:
            return H2V2InferenceResult(
                scores=released_m20_post_scores,
                receipt=H2V2InferenceReceipt(
                    event_count=m20.size,
                    polarity_minority_fraction=minority,
                    routed_to_h2=False,
                    disagreement_component_count=0,
                    restored_component_count=0,
                    restored_event_count=0,
                    recovery_cutoff=None,
                    non_h2_m20_object_identity=True,
                    internal_m20_post_bitwise_verified=False,
                    whole_components_only=True,
                    second_c00_applied=False,
                ),
            )
        payload = self._stage1_executor(video, locations4)
        if not isinstance(payload, Stage1ComponentPayload):
            raise TypeError("Stage1 executor returned an unexpected payload")
        if not _bitwise_float32_equal(payload.internal_m20_post_scores, m20):
            raise RuntimeError(
                "internal frozen M20 post-C00 scores differ bitwise from released M20"
            )
        stage1 = _require_float32_vector(
            payload.stage1_post_scores, "Stage1 post-C00 scores"
        )
        if stage1.shape != m20.shape:
            raise ValueError("Stage1 scores do not align with released M20")
        components = tuple(payload.components)
        nodes = tuple(payload.node_features)
        if len(components) != len(nodes):
            raise RuntimeError("one node sequence is required per disagreement component")
        probabilities = self._component_probability_predictor(nodes)
        stage2, decisions = apply_atomic_stage2(
            stage1,
            m20,
            components,
            probabilities,
            self.recovery_cutoff,
        )
        restored_event_count = int(
            sum(
                np.asarray(component, dtype=np.int64).size
                for component, decision in zip(components, decisions)
                if bool(decision)
            )
        )
        return H2V2InferenceResult(
            scores=stage2,
            receipt=H2V2InferenceReceipt(
                event_count=m20.size,
                polarity_minority_fraction=minority,
                routed_to_h2=True,
                disagreement_component_count=len(components),
                restored_component_count=int(np.count_nonzero(decisions)),
                restored_event_count=restored_event_count,
                recovery_cutoff=self.recovery_cutoff,
                non_h2_m20_object_identity=False,
                internal_m20_post_bitwise_verified=True,
                whole_components_only=True,
                second_c00_applied=False,
            ),
        )


__all__ = (
    "DEFAULT_OUTER_DECISION",
    "DEFAULT_RECOVERY_CHECKPOINT",
    "DEFAULT_STAGE1_CHECKPOINT",
    "FrozenCheckpointPayloads",
    "H2PyramidRecoveryV2Inference",
    "H2V2InferenceReceipt",
    "H2V2InferenceResult",
    "PREDICTION_THRESHOLD",
    "Stage1ComponentPayload",
    "apply_atomic_stage2",
    "load_final_package_payload",
    "load_frozen_checkpoint_payloads",
    "sha256_file",
    "state_sha256",
    "use_h2_pyramid_recovery_v2",
)

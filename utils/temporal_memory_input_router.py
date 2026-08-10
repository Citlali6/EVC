"""Fail-closed, label-free input routing for the frozen M20 candidate.

This module is intentionally opt-in.  It does not alter the released
submission path.  The route consumes the complete event count first and, only
inside the frozen high-density population, every event polarity in a complete
160-bin input video.  It never consumes a source name, event label, or target
identifier:

* event count <= 30,000 -> released M10 full-stream T160;
* 30,000 < event count <= 200,000 -> released M20 full-stream T160;
* event count > 200,000 and polarity minority fraction < 0.20 -> M20 T160;
* event count > 200,000 and fraction >= 0.20 -> M20 T32, stride 16.

The prediction threshold and C00 postprocessor are recorded in the policy
identity because they are part of the frozen evaluation protocol, although
postprocessing itself remains outside this inference-only module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch

from utils.temporal_memory_inference import predict_temporal_memory_scores
from utils.temporal_memory_windowed_inference import (
    predict_temporal_memory_scores_windowed,
)


ROUTE_POLICY_SCHEMA = "ev-uav-m20-polarity-temporal-route-v1"
POLARITY_MINORITY_CUTOFF = 0.20
EXPECTED_TEMPORAL_BIN_COUNT = 160
LOW_DENSITY_MAX_EVENT_COUNT = 30_000
# Reuse the frozen training-density boundary from
# TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff.  On official
# train it selects exactly the same 15 sources used by the T32 diagnostic.
HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE = 200_000
H2_WINDOW_LENGTH = 32
H2_WINDOW_STRIDE = 16
M10_PREDICTION_THRESHOLD = 0.718
M20_PREDICTION_THRESHOLD = 0.719
# Backwards-friendly name for the M20/H1/H2 candidate threshold.
PREDICTION_THRESHOLD = M20_PREDICTION_THRESHOLD
POSTPROCESS_PROFILE = "released_M20_C00_fixed"
PERSISTENCE_STAGE_STATUS = "disabled_pending_routed_train_oof_interaction"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def route_policy_definition() -> dict:
    """Return the immutable, JSON-serializable route definition."""

    return {
        "schema": ROUTE_POLICY_SCHEMA,
        "observable": "complete-video event polarity minority fraction",
        "observable_definition": (
            "min(mean(polarity > 0.5), 1 - mean(polarity > 0.5))"
        ),
        "observable_event_scope": "all input events exactly once",
        "low_density": {
            "condition": "event_count <= 30000",
            "checkpoint_role": "m10",
            "mode": "full_stream",
            "temporal_length": EXPECTED_TEMPORAL_BIN_COUNT,
        },
        "middle_density": {
            "condition": "30000 < event_count <= 200000",
            "checkpoint_role": "m20",
            "mode": "full_stream",
            "temporal_length": EXPECTED_TEMPORAL_BIN_COUNT,
        },
        "high_density": {
            "condition": "event_count > 200000",
            "checkpoint_role": "m20",
            "secondary_observable": "polarity minority fraction",
        },
        "low_density_max_event_count": LOW_DENSITY_MAX_EVENT_COUNT,
        "high_density_min_event_count_exclusive": (
            HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
        ),
        "cutoff": POLARITY_MINORITY_CUTOFF,
        "cutoff_operator": "<",
        "h1": {
            "mode": "full_stream",
            "temporal_length": EXPECTED_TEMPORAL_BIN_COUNT,
        },
        "h2": {
            "mode": "window_t32_stride16",
            "window_length": H2_WINDOW_LENGTH,
            "stride": H2_WINDOW_STRIDE,
            "stitch": "nearest_window_center_ties_to_earlier_window",
        },
        "expected_temporal_bin_count": EXPECTED_TEMPORAL_BIN_COUNT,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "prediction_threshold_by_checkpoint": {
            "m10": M10_PREDICTION_THRESHOLD,
            "m20": M20_PREDICTION_THRESHOLD,
        },
        "postprocess_profile": POSTPROCESS_PROFILE,
        "labels_used_for_route": False,
        "source_name_used_for_route": False,
        "persistent_pixel_second_stage": {
            "enabled": False,
            "status": PERSISTENCE_STAGE_STATUS,
        },
    }


def route_policy_sha256() -> str:
    """Return a stable digest binding every deployed route choice."""

    return _canonical_sha256(route_policy_definition())


def polarity_minority_fraction(polarities) -> float:
    """Compute the route observable from the complete polarity vector only.

    A one-dimensional, non-empty, finite vector in [0, 1] is required.  The
    strict shape check avoids accidentally routing on one batch, time slice,
    or multi-column array instead of the complete event stream.
    """

    values = np.asarray(polarities)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("Complete-video polarities must be a non-empty 1D vector.")
    if values.dtype.kind not in "biuf":
        raise TypeError("Polarities must be numeric.")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Polarities must be finite.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Normalized polarities must lie in [0, 1].")
    positive_count = int(np.count_nonzero(values > 0.5))
    negative_count = int(values.size - positive_count)
    return float(min(positive_count, negative_count) / values.size)


@dataclass(frozen=True)
class TemporalInputRouteDecision:
    """Auditable route decision derived without labels or source identity."""

    domain: str
    checkpoint_role: str
    mode: str
    polarity_minority_fraction: float
    event_count: int
    temporal_bin_count: int
    window_length: int | None
    stride: int | None
    prediction_threshold: float
    policy_sha256: str

    def __post_init__(self):
        if self.domain not in {"low", "middle", "h1", "h2"}:
            raise ValueError("domain must be low, middle, h1, or h2.")
        expected_mode = "window_t32" if self.domain == "h2" else "full_stream"
        if self.mode != expected_mode:
            raise ValueError("Route domain and mode disagree.")
        expected_checkpoint = "m10" if self.domain == "low" else "m20"
        if self.checkpoint_role != expected_checkpoint:
            raise ValueError("Route domain and checkpoint role disagree.")
        expected_threshold = (
            M10_PREDICTION_THRESHOLD
            if self.checkpoint_role == "m10"
            else M20_PREDICTION_THRESHOLD
        )
        if self.prediction_threshold != expected_threshold:
            raise ValueError("Route threshold and checkpoint role disagree.")
        if self.event_count <= 0:
            raise ValueError("event_count must be positive.")
        if self.temporal_bin_count != EXPECTED_TEMPORAL_BIN_COUNT:
            raise ValueError("The frozen route requires exactly 160 temporal bins.")
        if not 0.0 <= self.polarity_minority_fraction <= 0.5:
            raise ValueError("polarity_minority_fraction must lie in [0, 0.5].")
        if self.domain == "low" and self.event_count > LOW_DENSITY_MAX_EVENT_COUNT:
            raise ValueError("Low-density decision exceeds the M10 event-count gate.")
        if self.domain == "middle" and not (
            LOW_DENSITY_MAX_EVENT_COUNT
            < self.event_count
            <= HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
        ):
            raise ValueError("Middle-density decision is outside its frozen gate.")
        if self.domain in {"h1", "h2"} and (
            self.event_count <= HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
        ):
            raise ValueError("H1/H2 decisions require the frozen high-density gate.")
        if self.domain in {"low", "middle", "h1"} and (
            self.window_length is not None or self.stride is not None
        ):
            raise ValueError("Full-stream routes must not carry window metadata.")
        if self.domain == "h1" and (
            self.polarity_minority_fraction >= POLARITY_MINORITY_CUTOFF
        ):
            raise ValueError("Invalid H1 decision metadata.")
        if self.domain == "h2" and (
            self.polarity_minority_fraction < POLARITY_MINORITY_CUTOFF
            or self.window_length != H2_WINDOW_LENGTH
            or self.stride != H2_WINDOW_STRIDE
        ):
            raise ValueError("Invalid H2 decision metadata.")
        if self.policy_sha256 != route_policy_sha256():
            raise ValueError("Route policy digest mismatch.")

    def to_metadata(self) -> dict:
        return asdict(self)


def select_temporal_memory_input_route(
    polarities,
    temporal_bin_count,
) -> TemporalInputRouteDecision:
    """Select M10/M20 and H1/H2 using only complete-video input statistics.

    ``temporal_bin_count`` is structural input metadata.  No video object is
    accepted here, which makes accidental access to ``name`` or ``labels``
    impossible at the routing boundary.
    """

    if isinstance(temporal_bin_count, bool):
        raise ValueError("temporal_bin_count must be an integer.")
    try:
        temporal_bin_count = int(temporal_bin_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("temporal_bin_count must be an integer.") from error
    if temporal_bin_count != EXPECTED_TEMPORAL_BIN_COUNT:
        raise ValueError(
            "The frozen route requires exactly {} temporal bins; got {}.".format(
                EXPECTED_TEMPORAL_BIN_COUNT,
                temporal_bin_count,
            )
        )
    values = np.asarray(polarities)
    fraction = polarity_minority_fraction(values)
    event_count = int(values.size)
    common = {
        "polarity_minority_fraction": fraction,
        "event_count": event_count,
        "temporal_bin_count": temporal_bin_count,
        "policy_sha256": route_policy_sha256(),
    }
    if event_count <= LOW_DENSITY_MAX_EVENT_COUNT:
        return TemporalInputRouteDecision(
            domain="low",
            checkpoint_role="m10",
            mode="full_stream",
            window_length=None,
            stride=None,
            prediction_threshold=M10_PREDICTION_THRESHOLD,
            **common,
        )
    if event_count <= HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE:
        return TemporalInputRouteDecision(
            domain="middle",
            checkpoint_role="m20",
            mode="full_stream",
            window_length=None,
            stride=None,
            prediction_threshold=M20_PREDICTION_THRESHOLD,
            **common,
        )
    if fraction < POLARITY_MINORITY_CUTOFF:
        return TemporalInputRouteDecision(
            domain="h1",
            checkpoint_role="m20",
            mode="full_stream",
            window_length=None,
            stride=None,
            prediction_threshold=M20_PREDICTION_THRESHOLD,
            **common,
        )
    return TemporalInputRouteDecision(
        domain="h2",
        checkpoint_role="m20",
        mode="window_t32",
        window_length=H2_WINDOW_LENGTH,
        stride=H2_WINDOW_STRIDE,
        prediction_threshold=M20_PREDICTION_THRESHOLD,
        **common,
    )


def _validate_scores(scores, event_count):
    if not torch.is_tensor(scores):
        raise TypeError("Temporal-memory predictor must return a torch tensor.")
    scores = scores.detach().cpu().float().reshape(-1)
    if scores.numel() != int(event_count):
        raise RuntimeError(
            "Routed score count {} does not match event count {}.".format(
                scores.numel(),
                event_count,
            )
        )
    if not torch.isfinite(scores).all() or bool((scores < 0.0).any()) or bool(
        (scores > 1.0).any()
    ):
        raise RuntimeError("Routed inference produced non-probability scores.")
    return scores


def predict_temporal_memory_scores_input_routed(
    m10_model,
    m20_model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
):
    """Run the frozen input-only route and return ``(scores, decision)``.

    Only ``video.polarities``, ``video.event_indices_by_bin``, and the event
    count implied by ``video.locations`` are used outside the existing
    predictors.  In particular, neither ``video.name`` nor ``video.labels`` is
    read by this function or by the route selector.
    """

    temporal_bin_count = len(video.event_indices_by_bin)
    decision = select_temporal_memory_input_route(
        video.polarities,
        temporal_bin_count,
    )
    common = {
        "video": video,
        "device": device,
        "context_bins": context_bins,
        "width": width,
        "height": height,
        "inference_batch_size": inference_batch_size,
        "log_count_clip": log_count_clip,
    }
    if decision.checkpoint_role == "m10":
        common["model"] = m10_model
    else:
        common["model"] = m20_model
    if common["model"] is None:
        raise RuntimeError(
            "The {} checkpoint model required by the route is unavailable.".format(
                decision.checkpoint_role.upper()
            )
        )
    if decision.mode == "full_stream":
        scores = predict_temporal_memory_scores(**common)
    else:
        scores = predict_temporal_memory_scores_windowed(
            **common,
            window_length=decision.window_length,
            stride=decision.stride,
        )
    return _validate_scores(scores, decision.event_count), decision


def assert_full_window_identity(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
) -> dict:
    """Fail unless the L=full window path is bitwise identical to full-stream."""

    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count != EXPECTED_TEMPORAL_BIN_COUNT:
        raise ValueError("Identity audit requires exactly 160 temporal bins.")
    common = {
        "model": model,
        "video": video,
        "device": device,
        "context_bins": context_bins,
        "width": width,
        "height": height,
        "inference_batch_size": inference_batch_size,
        "log_count_clip": log_count_clip,
    }
    full_scores = _validate_scores(
        predict_temporal_memory_scores(**common),
        len(video.polarities),
    )
    identity_scores = _validate_scores(
        predict_temporal_memory_scores_windowed(
            **common,
            window_length=temporal_bin_count,
        ),
        len(video.polarities),
    )
    exact = bool(torch.equal(full_scores, identity_scores))
    max_delta = float(torch.max(torch.abs(full_scores - identity_scores)).item())
    if not exact:
        raise RuntimeError(
            "L=full identity failed (max absolute score delta {}).".format(max_delta)
        )
    return {
        "temporal_bin_count": temporal_bin_count,
        "bitwise_equal": True,
        "max_absolute_score_delta": max_delta,
        "event_count": int(full_scores.numel()),
    }


def require_persistence_second_stage_disabled(enabled) -> None:
    """Prevent an unaudited persistence model from silently entering the route."""

    if bool(enabled):
        raise RuntimeError(
            "Persistent-pixel postprocessing is not deployable yet: the available "
            "grouped OOF audit used full-stream M20 scores, so its interaction with "
            "H2 T32 routed scores must be evaluated and frozen first."
        )

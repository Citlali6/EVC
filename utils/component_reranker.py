"""Train-only-fitted component reranking for dense Challenge 2 videos.

The runtime path in this module is deliberately label-free.  It receives the
scores left by P0/P0c and derives features only from those scores, ``x/y/t``
locations, and the complete-video event count.  Labels are accepted solely by
``extract_component_examples`` for the separate training utility.

The reranker is conservative: it can suppress small retained components, but
it never raises a score.  ``ChallengePostprocessor`` places it between P0/P0c
and P18 so the stage ordering is explicit and reproducible.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


ARTIFACT_SCHEMA = "ev-uav-component-reranker-v1"
TRAIN_CACHE_SCHEMA = "ev-uav-component-reranker-train-cache-v1"
FEATURE_SEMANTICS_VERSION = "p0-per-bin-8conn-greedy-short-track-v1"
FEATURE_NAMES = (
    "log_video_events",
    "log_component_events",
    "score_max",
    "score_mean",
    "score_min",
    "score_std",
    "score_margin_max",
    "log_unique_cells",
    "bbox_diagonal",
    "track_bin_count",
    "log_track_events",
    "track_score_max",
    "track_score_mean",
    "track_displacement_per_bin",
)


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError("Expected a boolean value, got {!r}.".format(value))
    return bool(value)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_postprocess_mapping(config):
    """Serialize the effective P0/P0c contract seen by the reranker."""
    attributes = (
        "enabled",
        "spatial_radius",
        "temporal_bin_size",
        "temporal_radius_bins",
        "min_cluster_events",
        "min_duration_bins",
        "high_confidence_recovery_enabled",
        "retain_min_score",
        "density_retain_enabled",
        "density_event_count_cutoff",
        "density_retain_min_score",
    )
    missing = [name for name in attributes if not hasattr(config, name)]
    if missing:
        raise ValueError(
            "Component reranker input P0/P0c config is missing: {}.".format(
                ", ".join(missing)
            )
        )
    return {
        "enabled": bool(config.enabled),
        "spatial_radius": int(config.spatial_radius),
        "temporal_bin_size": int(config.temporal_bin_size),
        "temporal_radius_bins": int(config.temporal_radius_bins),
        "min_cluster_events": int(config.min_cluster_events),
        "min_duration_bins": int(config.min_duration_bins),
        "high_confidence_recovery_enabled": bool(
            config.high_confidence_recovery_enabled
        ),
        "retain_min_score": float(config.retain_min_score),
        "density_retain_enabled": bool(config.density_retain_enabled),
        "density_event_count_cutoff": int(config.density_event_count_cutoff),
        "density_retain_min_score": float(config.density_retain_min_score),
    }


def temporal_memory_inference_mapping(cfg):
    """Canonical numeric settings that determine M20 raw probabilities."""
    required = (
        "temporal_memory_bin_size",
        "temporal_memory_context_bins",
        "temporal_memory_width",
        "temporal_memory_sequence_length",
        "temporal_memory_inference_batch_size",
        "temporal_memory_log_count_clip",
        "whole_t",
        "res",
    )
    missing = [name for name in required if not hasattr(cfg, name)]
    if missing:
        raise ValueError(
            "Component reranker inference config is missing: {}.".format(
                ", ".join(missing)
            )
        )
    resolution = getattr(cfg, "res")
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("Component reranker inference resolution must contain two values.")
    mapping = {
        "temporal_memory_bin_size": int(cfg.temporal_memory_bin_size),
        "temporal_memory_context_bins": int(cfg.temporal_memory_context_bins),
        "temporal_memory_width": int(cfg.temporal_memory_width),
        "temporal_memory_sequence_length": int(cfg.temporal_memory_sequence_length),
        "temporal_memory_inference_batch_size": int(
            cfg.temporal_memory_inference_batch_size
        ),
        "temporal_memory_log_count_clip": float(
            cfg.temporal_memory_log_count_clip
        ),
        "whole_t": int(cfg.whole_t),
        "resolution": [int(resolution[0]), int(resolution[1])],
    }
    if mapping["temporal_memory_bin_size"] <= 0:
        raise ValueError("temporal_memory_bin_size must be positive.")
    if mapping["temporal_memory_context_bins"] <= 0:
        raise ValueError("temporal_memory_context_bins must be positive.")
    if mapping["temporal_memory_width"] <= 0:
        raise ValueError("temporal_memory_width must be positive.")
    if mapping["temporal_memory_sequence_length"] <= 1:
        raise ValueError("temporal_memory_sequence_length must exceed one.")
    if mapping["temporal_memory_inference_batch_size"] <= 0:
        raise ValueError("temporal_memory_inference_batch_size must be positive.")
    if not math.isfinite(mapping["temporal_memory_log_count_clip"]):
        raise ValueError("temporal_memory_log_count_clip must be finite.")
    if mapping["whole_t"] <= 0 or min(mapping["resolution"]) <= 0:
        raise ValueError("whole_t and resolution must be positive.")
    return mapping


@dataclass(frozen=True)
class ComponentTopology:
    # Radius one matches the evaluator's 8-connected per-bin components.
    spatial_radius: int = 1
    temporal_bin_size: int = 50
    max_link_distance: float = 6.0
    max_gap_bins: int = 1
    max_component_events: int = 3

    def __post_init__(self):
        if self.spatial_radius < 0:
            raise ValueError("component reranker spatial_radius must be non-negative.")
        if self.temporal_bin_size <= 0:
            raise ValueError("component reranker temporal_bin_size must be positive.")
        if not math.isfinite(self.max_link_distance) or self.max_link_distance < 0:
            raise ValueError("component reranker max_link_distance must be finite and non-negative.")
        if self.max_gap_bins < 1:
            raise ValueError("component reranker max_gap_bins must be positive.")
        if self.max_component_events < 1:
            raise ValueError("component reranker max_component_events must be positive.")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict):
            raise ValueError("component reranker topology must be a JSON object.")
        required = {
            "spatial_radius",
            "temporal_bin_size",
            "max_link_distance",
            "max_gap_bins",
            "max_component_events",
        }
        if set(value) != required:
            raise ValueError(
                "component reranker topology keys differ: expected {}.".format(
                    sorted(required)
                )
            )
        return cls(
            spatial_radius=int(value["spatial_radius"]),
            temporal_bin_size=int(value["temporal_bin_size"]),
            max_link_distance=float(value["max_link_distance"]),
            max_gap_bins=int(value["max_gap_bins"]),
            max_component_events=int(value["max_component_events"]),
        )

    def to_dict(self):
        return {
            "spatial_radius": self.spatial_radius,
            "temporal_bin_size": self.temporal_bin_size,
            "max_link_distance": self.max_link_distance,
            "max_gap_bins": self.max_gap_bins,
            "max_component_events": self.max_component_events,
        }


@dataclass(frozen=True)
class ComponentRerankerConfig:
    enabled: bool = False
    event_count_cutoff: int = 100000
    model_path: str = ""
    expected_sha256: str = ""
    base_checkpoint_path: str = ""
    event_count: Optional[int] = None

    def __post_init__(self):
        if self.event_count_cutoff < 0:
            raise ValueError("component_reranker_event_count_cutoff must be non-negative.")
        if self.event_count is not None and self.event_count < 0:
            raise ValueError("event_count must be non-negative.")
        if self.enabled and self.event_count is None:
            raise ValueError(
                "component reranker requires a complete-video event_count."
            )
        if self.enabled:
            if not self.model_path:
                raise ValueError(
                    "component_reranker_model_path is required when enabled."
                )
            if len(self.expected_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.expected_sha256
            ):
                raise ValueError(
                    "component_reranker_expected_sha256 must be a 64-character SHA-256."
                )
            if not self.base_checkpoint_path:
                raise ValueError(
                    "temporal_memory_model_path is required by the component reranker."
                )

    @property
    def eligible(self):
        return bool(
            self.enabled
            and self.event_count is not None
            and self.event_count > self.event_count_cutoff
        )

    @classmethod
    def from_cfg(cls, cfg, event_count=None):
        enabled = _as_bool(getattr(cfg, "component_reranker_enabled", False))
        event_count_cutoff = int(
            getattr(cfg, "component_reranker_event_count_cutoff", 100000)
        )
        if enabled:
            if not _as_bool(getattr(cfg, "temporal_memory_enabled", False)):
                raise ValueError(
                    "Component reranker currently supports only temporal-memory M20 inference."
                )
            if float(getattr(cfg, "temporal_memory_sparse_weight", 0.5)) != 0.0:
                raise ValueError(
                    "Component reranker requires pure temporal-memory inference "
                    "(temporal_memory_sparse_weight=0)."
                )
            if _as_bool(getattr(cfg, "temporal_frame_enabled", False)):
                raise ValueError(
                    "Component reranker does not support temporal-frame blending."
                )
            if str(getattr(cfg, "temporal_memory_blend_model_path", "")).strip():
                raise ValueError(
                    "Component reranker does not support a temporal-memory high-density blend model."
                )
            secondary_path = str(
                getattr(cfg, "temporal_memory_secondary_model_path", "")
            ).strip()
            secondary_max = int(
                getattr(cfg, "temporal_memory_secondary_max_event_count", 0)
            )
            if secondary_path and (
                secondary_max <= 0 or secondary_max > event_count_cutoff
            ):
                raise ValueError(
                    "Component reranker requires the secondary temporal-memory model "
                    "to be routed entirely below its dense event-count cutoff."
                )
            if _as_bool(getattr(cfg, "dense_expert_enabled", False)):
                raise ValueError(
                    "Component reranker does not support ENSEMBLE dense_expert routing."
                )
            if _as_bool(getattr(cfg, "ensemble_enabled", False)):
                raise ValueError(
                    "Component reranker does not support sparse-model ensemble blending."
                )
        return cls(
            enabled=enabled,
            event_count_cutoff=event_count_cutoff,
            model_path=str(getattr(cfg, "component_reranker_model_path", "")),
            expected_sha256=str(
                getattr(cfg, "component_reranker_expected_sha256", "")
            ).lower(),
            base_checkpoint_path=str(
                getattr(cfg, "temporal_memory_model_path", "")
            ),
            event_count=None if event_count is None else int(event_count),
        )


def load_artifact_payload(path, expected_sha256):
    """Load a strict JSON artifact after verifying its external file digest."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Component reranker JSON artifact does not exist: {}".format(path)
        )
    expected_sha256 = str(expected_sha256).strip().lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(
            "component_reranker_expected_sha256 must be a 64-character SHA-256."
        )
    artifact_sha256 = sha256_file(path)
    if artifact_sha256 != expected_sha256:
        raise ValueError(
            "Component reranker artifact SHA-256 {} does not match expected {}."
            .format(artifact_sha256, expected_sha256)
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(
            "Unsupported component reranker artifact schema: {!r}.".format(
                payload.get("schema") if isinstance(payload, dict) else None
            )
        )
    return payload, artifact_sha256


def validate_artifact_training_provenance(provenance):
    """Require an artifact lineage rooted in the official training cache."""
    if not isinstance(provenance, dict):
        raise ValueError("Component reranker provenance must be a JSON object.")
    if provenance.get("dataset_split") != "train":
        raise ValueError(
            "Component reranker artifact provenance must use dataset_split=train."
        )
    if provenance.get("train_cache_schema") != TRAIN_CACHE_SCHEMA:
        raise ValueError(
            "Component reranker artifact has invalid train_cache_schema provenance."
        )
    train_cache_manifest_sha256 = str(
        provenance.get("train_cache_manifest_sha256", "")
    ).lower()
    if len(train_cache_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in train_cache_manifest_sha256
    ):
        raise ValueError(
            "Component reranker artifact has invalid "
            "train_cache_manifest_sha256 provenance."
        )
    return provenance


@dataclass(frozen=True)
class ComponentLinearModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    keep_probability: float
    prediction_threshold: float
    topology: ComponentTopology
    provenance: dict
    artifact_sha256: str

    @classmethod
    def load(
        cls,
        path,
        expected_sha256,
        base_checkpoint_path,
        prediction_threshold,
        event_count_cutoff,
        input_postprocess,
        inference_settings,
    ):
        payload, artifact_sha256 = load_artifact_payload(path, expected_sha256)
        if payload.get("feature_semantics_version") != FEATURE_SEMANTICS_VERSION:
            raise ValueError(
                "Component reranker feature semantics version does not match runtime."
            )
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Component reranker feature schema does not match runtime.")

        feature_mean = _finite_vector(payload.get("feature_mean"), "feature_mean")
        feature_scale = _finite_vector(payload.get("feature_scale"), "feature_scale")
        coefficients = _finite_vector(payload.get("coefficients"), "coefficients")
        expected_width = len(FEATURE_NAMES)
        if not (
            feature_mean.size == feature_scale.size == coefficients.size == expected_width
        ):
            raise ValueError(
                "Component reranker vector width must equal {}.".format(expected_width)
            )
        if np.any(feature_scale <= 0):
            raise ValueError("Component reranker feature_scale values must be positive.")
        intercept = _finite_scalar(payload.get("intercept"), "intercept")
        keep_probability = _finite_scalar(
            payload.get("keep_probability"), "keep_probability"
        )
        artifact_threshold = _finite_scalar(
            payload.get("prediction_threshold"), "prediction_threshold"
        )
        if not 0.0 <= keep_probability <= 1.0:
            raise ValueError("Component reranker keep_probability must be in [0, 1].")
        if not math.isclose(
            artifact_threshold,
            float(prediction_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Component reranker was trained at prediction_threshold={} but runtime uses {}."
                .format(artifact_threshold, prediction_threshold)
            )

        provenance = payload.get("provenance")
        provenance = validate_artifact_training_provenance(provenance)
        trained_checkpoint_sha256 = str(
            provenance.get("base_checkpoint_sha256", "")
        ).lower()
        if len(trained_checkpoint_sha256) != 64:
            raise ValueError(
                "Component reranker provenance is missing base_checkpoint_sha256."
            )
        base_checkpoint_path = Path(base_checkpoint_path).expanduser().resolve()
        if not base_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Component reranker base checkpoint does not exist: {}".format(
                    base_checkpoint_path
                )
            )
        runtime_checkpoint_sha256 = sha256_file(base_checkpoint_path)
        if runtime_checkpoint_sha256 != trained_checkpoint_sha256:
            raise ValueError(
                "Component reranker base checkpoint SHA-256 {} does not match trained {}."
                .format(runtime_checkpoint_sha256, trained_checkpoint_sha256)
            )
        trained_cutoff = int(provenance.get("deployment_event_count_cutoff", -1))
        if trained_cutoff != int(event_count_cutoff):
            raise ValueError(
                "Component reranker deployment cutoff {} does not match trained {}."
                .format(event_count_cutoff, trained_cutoff)
            )
        runtime_input_postprocess = input_postprocess_mapping(input_postprocess)
        trained_input_postprocess = provenance.get("input_postprocess")
        if trained_input_postprocess != runtime_input_postprocess:
            raise ValueError(
                "Component reranker P0/P0c input contract differs from its artifact."
            )
        trained_input_sha256 = str(
            provenance.get("input_postprocess_sha256", "")
        ).lower()
        if (
            len(trained_input_sha256) != 64
            or trained_input_sha256 != sha256_json(runtime_input_postprocess)
        ):
            raise ValueError(
                "Component reranker P0/P0c input signature is invalid."
            )
        runtime_inference_settings = dict(inference_settings)
        trained_inference_settings = provenance.get("inference_settings")
        if trained_inference_settings != runtime_inference_settings:
            raise ValueError(
                "Component reranker temporal-memory inference settings differ "
                "from its artifact."
            )
        trained_inference_sha256 = str(
            provenance.get("inference_settings_sha256", "")
        ).lower()
        if (
            len(trained_inference_sha256) != 64
            or trained_inference_sha256 != sha256_json(runtime_inference_settings)
        ):
            raise ValueError(
                "Component reranker temporal-memory inference settings signature is invalid."
            )

        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            coefficients=coefficients,
            intercept=intercept,
            keep_probability=keep_probability,
            prediction_threshold=artifact_threshold,
            topology=ComponentTopology.from_mapping(payload.get("topology")),
            provenance=provenance,
            artifact_sha256=artifact_sha256,
        )

    def predict_keep_probability(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                "component reranker features must have shape [N, {}].".format(
                    len(FEATURE_NAMES)
                )
            )
        logits = (
            ((features - self.feature_mean) / self.feature_scale)
            @ self.coefficients
            + self.intercept
        )
        probabilities = np.empty_like(logits)
        nonnegative = logits >= 0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        negative_exp = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = negative_exp / (1.0 + negative_exp)
        return probabilities


def _finite_vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Component reranker {} must be a finite vector.".format(name))
    return array


def _finite_scalar(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Component reranker {} must be a finite scalar.".format(name)
        ) from exc
    if not math.isfinite(scalar):
        raise ValueError(
            "Component reranker {} must be a finite scalar.".format(name)
        )
    return scalar


@dataclass
class ComponentRerankerStats:
    enabled: bool
    eligible_videos: int = 0
    candidate_components: int = 0
    kept_components: int = 0
    removed_components: int = 0
    removed_events: int = 0

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError("Cannot merge enabled and disabled reranker statistics.")
        self.eligible_videos += other.eligible_videos
        self.candidate_components += other.candidate_components
        self.kept_components += other.kept_components
        self.removed_components += other.removed_components
        self.removed_events += other.removed_events

    def summary(self):
        if not self.enabled:
            return "disabled (predictions unchanged)"
        return (
            "enabled, eligible videos: {}; candidates: {} kept / {} removed; "
            "removed events: {}"
        ).format(
            self.eligible_videos,
            self.kept_components,
            self.removed_components,
            self.removed_events,
        )


@dataclass(frozen=True)
class ComponentExample:
    event_indices: np.ndarray
    features: np.ndarray
    label: Optional[int] = None


def _spatial_components(coordinates, event_indices, spatial_radius):
    """Return deterministic components over unique x/y cells in one time bin."""
    unique_cells, inverse = np.unique(coordinates[:, :2], axis=0, return_inverse=True)
    cell_lookup = {
        (int(cell[0]), int(cell[1])): index
        for index, cell in enumerate(unique_cells)
    }
    cell_events = [[] for _ in range(len(unique_cells))]
    for local_index, cell_index in enumerate(inverse):
        cell_events[int(cell_index)].append(local_index)
    offsets = tuple(
        (dx, dy)
        for dx in range(-spatial_radius, spatial_radius + 1)
        for dy in range(-spatial_radius, spatial_radius + 1)
        if (dx, dy) != (0, 0)
    )
    visited = np.zeros(len(unique_cells), dtype=bool)
    components = []
    for start in range(len(unique_cells)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        local_events = []
        while stack:
            cell_index = stack.pop()
            local_events.extend(cell_events[cell_index])
            x, y = unique_cells[cell_index]
            for dx, dy in offsets:
                neighbor = cell_lookup.get((int(x + dx), int(y + dy)))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        local_events = np.asarray(sorted(local_events), dtype=np.int64)
        components.append(event_indices[local_events])
    return components


def _build_components_and_tracks(scores, coordinates, topology):
    positive_indices = np.flatnonzero(scores >= 0.0)
    # The caller provides positive-only arrays.  Keeping this assertion local
    # catches accidental use of a sentinel mask in future refactors.
    if positive_indices.size != scores.size:
        raise ValueError("_build_components_and_tracks expects positive-only scores.")
    temporal_bins = np.floor_divide(coordinates[:, 2], topology.temporal_bin_size)
    components = []
    for temporal_bin in np.unique(temporal_bins):
        indices = np.flatnonzero(temporal_bins == temporal_bin)
        for component_indices in _spatial_components(
            coordinates[indices], indices, topology.spatial_radius
        ):
            component_coordinates = coordinates[component_indices]
            component_scores = scores[component_indices]
            components.append(
                {
                    "event_indices": component_indices,
                    "temporal_bin": int(temporal_bin),
                    "centroid": component_coordinates[:, :2].mean(axis=0),
                    "event_count": int(component_indices.size),
                    "score_max": float(component_scores.max()),
                    "score_mean": float(component_scores.mean()),
                }
            )

    tracks = []
    component_to_track = np.full(len(components), -1, dtype=np.int64)
    components_by_bin = {}
    for component_index, component in enumerate(components):
        components_by_bin.setdefault(component["temporal_bin"], []).append(component_index)
    for temporal_bin in sorted(components_by_bin):
        current = components_by_bin[temporal_bin]
        candidate_links = []
        for track_index, track in enumerate(tracks):
            gap = int(temporal_bin - track["last_bin"])
            if not 1 <= gap <= topology.max_gap_bins:
                continue
            for component_index in current:
                distance = float(
                    np.linalg.norm(
                        components[component_index]["centroid"] - track["last_centroid"]
                    )
                )
                if distance <= topology.max_link_distance:
                    candidate_links.append((distance, track_index, component_index))
        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index in sorted(candidate_links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue
            track = tracks[track_index]
            track["component_indices"].append(component_index)
            track["last_bin"] = temporal_bin
            track["last_centroid"] = components[component_index]["centroid"]
            component_to_track[component_index] = track_index
            assigned_tracks.add(track_index)
            assigned_components.add(component_index)
        for component_index in current:
            if component_index in assigned_components:
                continue
            component = components[component_index]
            track_index = len(tracks)
            tracks.append(
                {
                    "component_indices": [component_index],
                    "first_bin": temporal_bin,
                    "last_bin": temporal_bin,
                    "first_centroid": component["centroid"],
                    "last_centroid": component["centroid"],
                }
            )
            component_to_track[component_index] = track_index
    return components, tracks, component_to_track


def extract_component_examples(
    prediction_scores,
    locations,
    prediction_threshold,
    topology,
    video_event_count,
    labels=None,
):
    """Extract reranker candidates and optional train-only component labels.

    ``locations`` are ordered ``[batch, x, y, t]``.  If ``labels`` is given,
    a component is positive when at least one of its retained events is a
    target event.  Labels never enter ``features``.
    """
    scores = np.asarray(prediction_scores, dtype=np.float64).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError("locations must have shape [N, 4+] ordered [batch, x, y, t].")
    if scores.size != locations.shape[0]:
        raise ValueError("prediction_scores and locations must have matching lengths.")
    if not np.isfinite(scores).all():
        raise ValueError("prediction_scores must be finite.")
    if not 0.0 <= float(prediction_threshold) <= 1.0:
        raise ValueError("prediction_threshold must be in [0, 1].")
    if int(video_event_count) < 0:
        raise ValueError("video_event_count must be non-negative.")
    if not isinstance(topology, ComponentTopology):
        raise TypeError("topology must be a ComponentTopology.")
    label_values = None
    if labels is not None:
        label_values = np.asarray(labels).reshape(-1)
        if label_values.size != scores.size:
            raise ValueError("labels and prediction_scores must have matching lengths.")

    examples = []
    positive_mask = scores >= float(prediction_threshold)
    for batch_id in np.unique(locations[:, 0]):
        video_mask = locations[:, 0] == batch_id
        positive_indices = np.flatnonzero(video_mask & positive_mask)
        if positive_indices.size == 0:
            continue
        positive_scores = scores[positive_indices]
        positive_coordinates = locations[positive_indices, 1:4].astype(
            np.int64, copy=False
        )
        components, tracks, component_to_track = _build_components_and_tracks(
            positive_scores,
            positive_coordinates,
            topology,
        )
        for component_index, component in enumerate(components):
            if component["event_count"] > topology.max_component_events:
                continue
            local_indices = component["event_indices"]
            event_indices = positive_indices[local_indices]
            component_scores = positive_scores[local_indices]
            component_coordinates = positive_coordinates[local_indices]
            track = tracks[int(component_to_track[component_index])]
            track_component_indices = track["component_indices"]
            track_event_indices = np.concatenate(
                [components[index]["event_indices"] for index in track_component_indices]
            )
            track_scores = positive_scores[track_event_indices]
            track_event_count = int(track_event_indices.size)
            bin_span = max(int(track["last_bin"] - track["first_bin"]), 1)
            track_displacement = float(
                np.linalg.norm(track["last_centroid"] - track["first_centroid"])
                / bin_span
            )
            unique_cells = np.unique(component_coordinates[:, :2], axis=0).shape[0]
            spatial_extent = component_coordinates[:, :2].max(axis=0) - component_coordinates[
                :, :2
            ].min(axis=0)
            feature_values = np.asarray(
                (
                    math.log1p(int(video_event_count)),
                    math.log1p(component["event_count"]),
                    float(component_scores.max()),
                    float(component_scores.mean()),
                    float(component_scores.min()),
                    float(component_scores.std()),
                    float(component_scores.max() - prediction_threshold),
                    math.log1p(int(unique_cells)),
                    float(np.linalg.norm(spatial_extent)),
                    float(len(track_component_indices)),
                    math.log1p(track_event_count),
                    float(track_scores.max()),
                    float(track_scores.mean()),
                    track_displacement,
                ),
                dtype=np.float64,
            )
            if not np.isfinite(feature_values).all():
                raise RuntimeError("Non-finite component reranker feature was derived.")
            component_label = None
            if label_values is not None:
                component_label = int(np.any(label_values[event_indices] > 0.5))
            examples.append(
                ComponentExample(
                    event_indices=event_indices.astype(np.int64, copy=False),
                    features=feature_values,
                    label=component_label,
                )
            )
    return examples


class ComponentReranker:
    """Suppress low-ranked small components in eligible dense videos."""

    def __init__(self, config, prediction_threshold=0.9, model=None):
        if not isinstance(config, ComponentRerankerConfig):
            raise TypeError("config must be a ComponentRerankerConfig.")
        self.config = config
        self.prediction_threshold = float(prediction_threshold)
        self.model = model
        if self.config.eligible and self.model is None:
            raise ValueError("Eligible component reranker requires a loaded model.")

    @classmethod
    def from_cfg(
        cls,
        cfg,
        prediction_threshold=0.9,
        event_count=None,
        input_postprocess=None,
    ):
        config = ComponentRerankerConfig.from_cfg(cfg, event_count=event_count)
        model = None
        if config.eligible:
            if not config.model_path:
                raise ValueError(
                    "component_reranker_model_path is required for eligible videos."
                )
            model = ComponentLinearModel.load(
                config.model_path,
                config.expected_sha256,
                config.base_checkpoint_path,
                prediction_threshold,
                config.event_count_cutoff,
                input_postprocess,
                temporal_memory_inference_mapping(cfg),
            )
        return cls(config, prediction_threshold, model)

    @property
    def enabled(self):
        return self.config.enabled

    @property
    def eligible(self):
        return self.config.eligible

    def new_stats(self):
        return ComponentRerankerStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return "disabled"
        if not self.eligible:
            return "enabled but ineligible (event_count <= {})".format(
                self.config.event_count_cutoff
            )
        return (
            "enabled (event_count > {}, artifact_sha256={}, keep_probability={}, "
            "max_component_events={})"
        ).format(
            self.config.event_count_cutoff,
            self.model.artifact_sha256,
            self.model.keep_probability,
            self.model.topology.max_component_events,
        )

    def apply(self, predictions, locations):
        if not self.enabled or not self.eligible:
            return predictions, ComponentRerankerStats(enabled=self.enabled)

        import torch

        flattened = predictions.reshape(-1)
        if locations.ndim != 2 or locations.shape[1] < 4:
            raise ValueError(
                "Component reranker locations must have shape [N, 4+] "
                "ordered [batch, x, y, t]."
            )
        if flattened.numel() != locations.shape[0]:
            raise ValueError(
                "Prediction and location counts do not match: {} and {}.".format(
                    flattened.numel(), locations.shape[0]
                )
            )
        if int(locations.shape[0]) != int(self.config.event_count):
            raise ValueError(
                "Component reranker complete-video event_count {} does not match "
                "location rows {}.".format(
                    self.config.event_count, locations.shape[0]
                )
            )
        batch_ids = locations[:, 0].detach().cpu().numpy()
        if np.unique(batch_ids).size != 1:
            raise ValueError(
                "Component reranker requires exactly one complete-video batch id."
            )
        examples = extract_component_examples(
            flattened.detach().cpu().numpy(),
            locations.detach().cpu().numpy(),
            self.prediction_threshold,
            self.model.topology,
            self.config.event_count,
        )
        stats = ComponentRerankerStats(
            enabled=True,
            eligible_videos=1,
            candidate_components=len(examples),
        )
        if not examples:
            return predictions, stats
        features = np.stack([example.features for example in examples], axis=0)
        probabilities = self.model.predict_keep_probability(features)
        keep = probabilities >= self.model.keep_probability
        stats.kept_components = int(keep.sum())
        stats.removed_components = int((~keep).sum())
        if keep.all():
            return predictions, stats
        removed_indices = np.concatenate(
            [
                example.event_indices
                for example, keep_component in zip(examples, keep)
                if not keep_component
            ]
        )
        stats.removed_events = int(removed_indices.size)
        output = flattened.clone()
        output[
            torch.from_numpy(removed_indices).to(device=output.device, dtype=torch.long)
        ] = 0.0
        return output.reshape_as(predictions), stats

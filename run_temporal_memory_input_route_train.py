"""Freeze, cache, and evaluate the opt-in temporal input route on train only.

This entry point is deliberately separate from ``submit_challenge2.py`` and
the released validation replay.  Its three stages are fail-closed:

``freeze``
    Bind the complete label-free train input audit, pinned M10/M20 checkpoint
    hashes, inference settings, C00 settings, and code hashes.
``cache``
    Read only event coordinates and polarity, run both the released baseline
    and the frozen candidate where they differ, and create immutable score
    records.  No labels or target ids are indexed.
``evaluate``
    Reopen the same hashed train sources, read labels only after every cache
    and route identity check passes, apply the per-route frozen threshold plus
    C00, and aggregate sufficient counts before computing the official metric.

No stage accepts validation or test paths or source names.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import tempfile
import time

import numpy as np
import torch

from audit_temporal_memory_input_route_train import (
    OFFICIAL_TRAIN_NAMES,
    SCHEMA as AUDIT_SCHEMA,
    discover_official_train_sources,
    sha256_file,
)
from dataset.temporal_frame import (
    load_temporal_frame_video,
    temporal_frame_video_from_events,
)
from utils.challenge_eval import challenge_score
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor
from utils.temporal_memory_inference import (
    load_temporal_memory_model,
    predict_temporal_memory_scores,
)
from utils.temporal_memory_input_router import (
    EXPECTED_TEMPORAL_BIN_COUNT,
    POSTPROCESS_PROFILE,
    assert_full_window_identity,
    predict_temporal_memory_scores_input_routed,
    require_persistence_second_stage_disabled,
    route_policy_definition,
    route_policy_sha256,
    select_temporal_memory_input_route,
)


PROTOCOL_SCHEMA = "ev-uav-temporal-input-route-train-protocol-v3"
CACHE_SCHEMA = "ev-uav-temporal-input-route-train-cache-v1"
EVALUATION_SCHEMA = "ev-uav-temporal-input-route-train-evaluation-v1"
ATTEMPT_FAILURE_SCHEMA = "ev-uav-temporal-input-route-attempt-failure-v1"
SUPERSEDED_PROTOCOL_SHA256 = (
    "b703e3a1a1f2a2441f9d8a51a298a31ce6d5fef844348d5ee2a90631e4a458f1"
)
M10_SHA256 = "5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935"
M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
WHOLE_T = 8000
TEMPORAL_BIN_SIZE = 50
CONTEXT_BINS = 5
MODEL_WIDTH = 16
WIDTH = 346
HEIGHT = 260
INFERENCE_BATCH_SIZE = 8
LOG_COUNT_CLIP = 4.0

# This is the complete scientific route identity frozen per source.  Explicit
# projection prevents future non-scientific dataclass fields from invalidating
# a cache, while inclusion of temporal_bin_count closes attempt1's omission.
ROUTE_DECISION_FIELDS = (
    "event_count",
    "polarity_minority_fraction",
    "domain",
    "checkpoint_role",
    "mode",
    "temporal_bin_count",
    "window_length",
    "stride",
    "prediction_threshold",
    "policy_sha256",
)

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

C00_DEFINITION = {
    "pd_detT": 50,
    "p0_enabled": True,
    "p0_spatial_radius": 2,
    "p0_temporal_bin_size": 50,
    "p0_temporal_radius_bins": 1,
    "p0_min_cluster_events": 3,
    "p0_min_duration_bins": 5,
    "p0c_high_confidence_recovery_enabled": True,
    "p0c_retain_min_score": 0.95,
    "p0c_density_retain_enabled": False,
    "p0b_enabled": False,
    "p18_score_track_recovery_enabled": True,
    "p18_event_count_cutoff": 1,
    "p18_max_event_count": 35000,
    "p18_candidate_floor": 0.53,
    "p18_spatial_radius": 5,
    "p18_temporal_bin_size": 50,
    "p18_max_link_distance": 8.0,
    "p18_max_gap_bins": 1,
    "p18_min_track_bins": 4,
    "p18_restore_mode": "best",
    "p18_max_restore_events_per_component": 0,
    "component_reranker_enabled": False,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_route_decision(value):
    """Project and validate the complete per-source scientific route identity."""

    if not isinstance(value, dict):
        raise TypeError("Route decision must be a mapping.")
    missing = [field for field in ROUTE_DECISION_FIELDS if field not in value]
    if missing:
        raise ValueError("Route decision lacks frozen fields: {}".format(missing))
    projected = {field: value[field] for field in ROUTE_DECISION_FIELDS}
    if projected["temporal_bin_count"] != EXPECTED_TEMPORAL_BIN_COUNT:
        raise ValueError("Frozen route decision must use 160 temporal bins.")
    if projected["policy_sha256"] != route_policy_sha256():
        raise ValueError("Frozen route decision policy hash mismatch.")
    return projected


def c00_sha256():
    return canonical_sha256(C00_DEFINITION)


def c00_config():
    return SimpleNamespace(**C00_DEFINITION)


def evaluator_config():
    return SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001)


def atomic_json(path, payload):
    path = Path(path).resolve()
    sidecar = Path(str(path) + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite immutable output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = sha256_file(path)
    sidecar.write_text("{}  {}\n".format(digest, path.name), encoding="ascii")
    return path, digest, sidecar


def load_json(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream), path


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_identity(path, role):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    expected_digest = M10_SHA256 if role == "m10" else M20_SHA256
    if digest != expected_digest:
        raise ValueError("{} checkpoint SHA-256 mismatch.".format(role.upper()))
    payload = torch_load_cpu(path)
    memory = payload.get("temporal_memory")
    if not isinstance(memory, dict):
        raise ValueError("{} checkpoint lacks temporal_memory metadata.".format(role))
    expected = {
        "temporal_bin_size": TEMPORAL_BIN_SIZE,
        "context_bins": CONTEXT_BINS,
        "width": MODEL_WIDTH,
        "sequence_length": 16,
        "log_count_clip": LOG_COUNT_CLIP,
    }
    mismatches = {
        key: {"expected": value, "actual": memory.get(key)}
        for key, value in expected.items()
        if memory.get(key) != value
    }
    if mismatches:
        raise ValueError("{} checkpoint metadata mismatch: {}".format(role, mismatches))
    expected_attention = role == "m20"
    if bool(memory.get("temporal_attention_enabled", False)) != expected_attention:
        raise ValueError("{} checkpoint attention identity mismatch.".format(role))
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "temporal_memory": {
            key: memory.get(key)
            for key in (
                "temporal_bin_size",
                "context_bins",
                "width",
                "sequence_length",
                "log_count_clip",
                "density_calibration_enabled",
                "trajectory_extrapolation_enabled",
                "confidence_head_enabled",
                "temporal_attention_enabled",
            )
        },
    }


def code_paths():
    root = Path(__file__).resolve().parent
    return {
        "run_temporal_memory_input_route_train.py": Path(__file__).resolve(),
        "audit_temporal_memory_input_route_train.py": root
        / "audit_temporal_memory_input_route_train.py",
        "utils/temporal_memory_input_router.py": root
        / "utils"
        / "temporal_memory_input_router.py",
        "utils/temporal_memory_windowed_inference.py": root
        / "utils"
        / "temporal_memory_windowed_inference.py",
        "utils/temporal_memory_inference.py": root
        / "utils"
        / "temporal_memory_inference.py",
        "utils/postprocess.py": root / "utils" / "postprocess.py",
        "utils/eval.py": root / "utils" / "eval.py",
        "utils/challenge_eval.py": root / "utils" / "challenge_eval.py",
    }


def current_code_identity():
    paths = code_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Route protocol code missing: {}".format(missing))
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def validate_audit(payload, path):
    if payload.get("schema") != AUDIT_SCHEMA:
        raise ValueError("Input audit schema mismatch: {}".format(path))
    split = payload.get("split_access", {})
    if split.get("dataset_split") != "train" or split.get(
        "validation_or_test_read"
    ) is not False:
        raise ValueError("Input audit is not train-only.")
    independence = payload.get("route_independence", {})
    if independence.get("labels_used") is not False or independence.get(
        "source_name_used"
    ) is not False:
        raise ValueError("Input audit does not prove label/name-independent routing.")
    if payload.get("policy_sha256") != route_policy_sha256():
        raise ValueError("Input audit route policy has drifted.")
    if payload.get("policy") != route_policy_definition():
        raise ValueError("Input audit policy definition has drifted.")
    population = payload.get("population", {})
    if (
        population.get("video_count") != len(OFFICIAL_TRAIN_NAMES)
        or population.get("event_count_gt_30000") != 54
        or population.get("event_count_gt_200000") != 15
        or population.get("event_count_200001_to_250000") != 0
        or population.get("gt_200000_matches_existing_15_source_evidence") is not True
    ):
        raise ValueError("Input audit population gates are not the frozen 99/54/15 split.")
    records = payload.get("records")
    if not isinstance(records, list) or tuple(
        record.get("source_name") for record in records
    ) != OFFICIAL_TRAIN_NAMES:
        raise ValueError("Input audit lacks the canonical ordered train population.")
    protection = payload.get("density_gate_protection", {})
    if protection.get("unassessed_below_200k_sources_sent_to_t32") != 0:
        raise ValueError("Input audit routes unassessed sources to T32.")
    return records


def freeze_protocol(args):
    require_persistence_second_stage_disabled(False)
    audit, audit_path = load_json(args.audit)
    audit_records = validate_audit(audit, audit_path)
    attempt_failure, attempt_failure_path = load_json(args.amendment_record)
    if (
        attempt_failure.get("schema") != ATTEMPT_FAILURE_SCHEMA
        or attempt_failure.get("attempt_id") != "attempt2"
        or attempt_failure.get("result")
        != "fail_closed_before_model_load_or_score_cache"
        or attempt_failure.get("frozen_protocol", {}).get("sha256")
        != SUPERSEDED_PROTOCOL_SHA256
        or attempt_failure.get("side_effects", {}).get("score_records_written") != 0
        or attempt_failure.get("side_effects", {}).get("models_loaded") is not False
        or attempt_failure.get("scientific_protocol", {}).get(
            "scientific_route_changed"
        )
        is not False
    ):
        raise ValueError("Attempt2 amendment record is incomplete or inconsistent.")
    m10 = checkpoint_identity(args.m10_checkpoint, "m10")
    m20 = checkpoint_identity(args.m20_checkpoint, "m20")
    code = current_code_identity()
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_revision": 3,
        "created_utc": utc_now(),
        "status": "frozen_before_routed_score_cache",
        "evidence_class": "preregistered_complete_train_cache_and_evaluation_protocol",
        "split_access": {
            "dataset_split": "train",
            "validation_or_test_permitted": False,
            "cache_stage_labels_permitted": False,
            "evaluation_stage_train_labels_permitted": True,
        },
        "route": route_policy_definition(),
        "route_policy_sha256": route_policy_sha256(),
        "route_decision_fields": list(ROUTE_DECISION_FIELDS),
        "amendment": {
            "kind": "cuda_peak_memory_api_compatibility_recovery",
            "supersedes_protocol_sha256": SUPERSEDED_PROTOCOL_SHA256,
            "recovery_record_path": str(attempt_failure_path),
            "recovery_record_sha256": sha256_file(attempt_failure_path),
            "change": (
                "pass the frozen cuda:0 integer device index to Torch 2.5.1 "
                "peak-memory reset/read APIs"
            ),
            "cuda_compatibility_probe_passed": True,
            "scientific_route_changed": False,
            "unchanged": [
                "99 train source hashes",
                "M10 and M20 checkpoint hashes",
                "M10 0.718 and M20 0.719 thresholds",
                "event-count gates 30000 and 200000",
                "polarity minority cutoff 0.20",
                "H2 T32 stride16 nearest-center stitching",
                "C00 postprocess",
                "complete explicit per-source route decision fields",
            ],
        },
        "checkpoints": {"m10": m10, "m20": m20},
        "inference": {
            "whole_t": WHOLE_T,
            "temporal_bin_size": TEMPORAL_BIN_SIZE,
            "temporal_bin_count": EXPECTED_TEMPORAL_BIN_COUNT,
            "context_bins": CONTEXT_BINS,
            "model_width": MODEL_WIDTH,
            "resolution": [WIDTH, HEIGHT],
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "log_count_clip": LOG_COUNT_CLIP,
            "l_equals_full_identity_required_per_checkpoint": True,
        },
        "postprocess": {
            "profile": POSTPROCESS_PROFILE,
            "definition": C00_DEFINITION,
            "sha256": c00_sha256(),
        },
        "population": {
            "names": list(OFFICIAL_TRAIN_NAMES),
            "source_sha256": {
                record["source_name"]: record["source_sha256"]
                for record in audit_records
            },
            "route_decisions": {
                record["source_name"]: frozen_route_decision(record)
                for record in audit_records
            },
        },
        "cache_contract": {
            "schema": CACHE_SCHEMA,
            "records": "baseline and candidate raw float32 probability vectors",
            "candidate_difference": "H2 only; all other routes must be bitwise equal",
            "labels_consumed": False,
        },
        "evaluation_contract": {
            "schema": EVALUATION_SCHEMA,
            "metric": "official sufficient-count pooling after per-video threshold+C00",
            "candidate_selection_or_threshold_search": False,
            "promotion_claim": False,
        },
        "persistence_second_stage": {
            "enabled": False,
            "status": "pending_routed_train_oof_interaction",
        },
        "provenance": {
            "audit_path": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "code": code,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
    }
    output, digest, sidecar = atomic_json(args.output, protocol)
    print("protocol:", output)
    print("protocol_sha256:", digest)
    print("sha256_sidecar:", sidecar)
    return protocol


def validate_protocol(payload, path, require_current_code=True):
    if payload.get("schema") != PROTOCOL_SCHEMA or payload.get("status") != (
        "frozen_before_routed_score_cache"
    ):
        raise ValueError("Route protocol is not frozen or has an unexpected schema.")
    if payload.get("protocol_revision") != 3 or tuple(
        payload.get("route_decision_fields", [])
    ) != ROUTE_DECISION_FIELDS:
        raise ValueError("Route protocol lacks the complete v3 decision field contract.")
    amendment = payload.get("amendment", {})
    if (
        amendment.get("supersedes_protocol_sha256")
        != SUPERSEDED_PROTOCOL_SHA256
        or amendment.get("scientific_route_changed") is not False
        or amendment.get("cuda_compatibility_probe_passed") is not True
        or not Path(amendment.get("recovery_record_path", "")).is_file()
        or sha256_file(amendment["recovery_record_path"])
        != amendment.get("recovery_record_sha256")
    ):
        raise ValueError("Route protocol amendment identity mismatch.")
    split = payload.get("split_access", {})
    if split.get("dataset_split") != "train" or split.get(
        "validation_or_test_permitted"
    ) is not False:
        raise ValueError("Route protocol is not train-only.")
    if payload.get("route_policy_sha256") != route_policy_sha256() or payload.get(
        "route"
    ) != route_policy_definition():
        raise ValueError("Route protocol policy drift detected.")
    if payload.get("postprocess", {}).get("definition") != C00_DEFINITION or payload.get(
        "postprocess", {}
    ).get("sha256") != c00_sha256():
        raise ValueError("Route protocol C00 drift detected.")
    checkpoints = payload.get("checkpoints", {})
    if checkpoints.get("m10", {}).get("sha256") != M10_SHA256 or checkpoints.get(
        "m20", {}
    ).get("sha256") != M20_SHA256:
        raise ValueError("Route protocol checkpoint identity mismatch.")
    population = payload.get("population", {})
    if tuple(population.get("names", [])) != OFFICIAL_TRAIN_NAMES:
        raise ValueError("Route protocol train population mismatch.")
    decisions = population.get("route_decisions", {})
    if set(decisions) != set(OFFICIAL_TRAIN_NAMES):
        raise ValueError("Route protocol decision population mismatch.")
    for name in OFFICIAL_TRAIN_NAMES:
        if decisions[name] != frozen_route_decision(decisions[name]):
            raise ValueError("Route protocol decision projection drift: {}".format(name))
    if require_current_code and payload.get("provenance", {}).get("code") != (
        current_code_identity()
    ):
        raise ValueError("Route protocol code hashes no longer match the workspace.")
    return population


def load_input_only_video(path):
    """Construct a temporal video without reading label or target-id columns."""

    with np.load(path, allow_pickle=False) as payload:
        if "evs_norm" not in payload.files or "ev_loc" not in payload.files:
            raise ValueError("Train source lacks evs_norm/ev_loc: {}".format(path))
        inputs = np.asarray(payload["evs_norm"])
        locations = np.asarray(payload["ev_loc"])
        if inputs.ndim != 2 or inputs.shape[1] < 4:
            raise ValueError("evs_norm must contain x/y/t/p input columns.")
        if locations.ndim != 2 or locations.shape[1] < 3:
            raise ValueError("ev_loc must have three columns.")
        if inputs.shape[0] != locations.shape[0]:
            raise ValueError("Input location and polarity counts disagree.")
        polarities = np.asarray(inputs[:, 3], dtype=np.float32).copy()
        locations = np.asarray(locations[:, :3], dtype=np.int64).copy()
    # A constant name and implicit zero labels/ids make the cache-stage data
    # dependency explicit: runtime routing cannot learn source identity.
    return temporal_frame_video_from_events(
        name="input_video",
        locations=locations,
        polarities=polarities,
        temporal_bin_size=TEMPORAL_BIN_SIZE,
        whole_t=WHOLE_T,
    )


def validate_probability_scores(scores, event_count, label):
    scores = torch.as_tensor(scores).detach().cpu().float().reshape(-1)
    if scores.numel() != int(event_count):
        raise RuntimeError("{} score count mismatch.".format(label))
    if not torch.isfinite(scores).all() or bool((scores < 0).any()) or bool(
        (scores > 1).any()
    ):
        raise RuntimeError("{} contains non-probability scores.".format(label))
    return scores


def atomic_npz(path, **arrays):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_scores(args):
    require_persistence_second_stage_disabled(False)
    protocol, protocol_path = load_json(args.protocol)
    population = validate_protocol(protocol, protocol_path)
    train_root, paths = discover_official_train_sources(args.train_root)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse cache directory: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    records_dir = output_dir / "records"
    records_dir.mkdir()

    expected_hashes = population["source_sha256"]
    expected_decisions = population["route_decisions"]
    for path in paths:
        if sha256_file(path) != expected_hashes.get(path.name):
            raise ValueError("Train source SHA-256 mismatch: {}".format(path))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("Formal CUDA device must have an explicit integer index.")
        torch.cuda.set_device(device.index)
        torch.cuda.reset_peak_memory_stats(device.index)
    m10_info = protocol["checkpoints"]["m10"]
    m20_info = protocol["checkpoints"]["m20"]
    if sha256_file(m10_info["path"]) != m10_info["sha256"] or sha256_file(
        m20_info["path"]
    ) != m20_info["sha256"]:
        raise ValueError("A frozen checkpoint file has changed.")
    m10_model, _ = load_temporal_memory_model(
        m10_info["path"], device, CONTEXT_BINS, MODEL_WIDTH, 16
    )
    m20_model, _ = load_temporal_memory_model(
        m20_info["path"], device, CONTEXT_BINS, MODEL_WIDTH, 16
    )

    records = []
    identity_by_checkpoint = {}
    started = time.perf_counter()
    for index, path in enumerate(paths):
        video = load_input_only_video(path)
        decision = select_temporal_memory_input_route(
            video.polarities, len(video.event_indices_by_bin)
        )
        runtime_frozen_decision = frozen_route_decision(decision.to_metadata())
        if runtime_frozen_decision != expected_decisions.get(path.name):
            raise RuntimeError("Frozen input route changed for {}.".format(path.name))
        model = m10_model if decision.checkpoint_role == "m10" else m20_model
        if decision.checkpoint_role not in identity_by_checkpoint:
            identity_by_checkpoint[decision.checkpoint_role] = {
                "source_name": path.name,
                **assert_full_window_identity(
                    model=model,
                    video=video,
                    device=device,
                    context_bins=CONTEXT_BINS,
                    width=WIDTH,
                    height=HEIGHT,
                    inference_batch_size=INFERENCE_BATCH_SIZE,
                    log_count_clip=LOG_COUNT_CLIP,
                ),
            }
        synchronize(device)
        inference_started = time.perf_counter()
        candidate_scores, runtime_decision = (
            predict_temporal_memory_scores_input_routed(
                m10_model=m10_model,
                m20_model=m20_model,
                video=video,
                device=device,
                context_bins=CONTEXT_BINS,
                width=WIDTH,
                height=HEIGHT,
                inference_batch_size=INFERENCE_BATCH_SIZE,
                log_count_clip=LOG_COUNT_CLIP,
            )
        )
        if runtime_decision != decision:
            raise RuntimeError("Route decision changed within one inference call.")
        if decision.mode == "window_t32":
            baseline_scores = predict_temporal_memory_scores(
                model=m20_model,
                video=video,
                device=device,
                context_bins=CONTEXT_BINS,
                width=WIDTH,
                height=HEIGHT,
                inference_batch_size=INFERENCE_BATCH_SIZE,
                log_count_clip=LOG_COUNT_CLIP,
            )
            baseline_scores = validate_probability_scores(
                baseline_scores, decision.event_count, "baseline"
            )
        else:
            baseline_scores = candidate_scores.clone()
        synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        if decision.mode != "window_t32" and not torch.equal(
            baseline_scores, candidate_scores
        ):
            raise RuntimeError("An unchanged route produced different cache vectors.")
        record_path = records_dir / "{:03d}.npz".format(index)
        atomic_npz(
            record_path,
            baseline_scores=baseline_scores.numpy().astype(np.float32, copy=False),
            candidate_scores=candidate_scores.numpy().astype(np.float32, copy=False),
        )
        records.append(
            {
                "source_name": path.name,
                "source_sha256": expected_hashes[path.name],
                "record": str(record_path.relative_to(output_dir)).replace("\\", "/"),
                "record_sha256": sha256_file(record_path),
                "event_count": decision.event_count,
                "decision": runtime_frozen_decision,
                "baseline_mode": "full_stream",
                "candidate_mode": decision.mode,
                "bitwise_equal_to_baseline": bool(
                    torch.equal(baseline_scores, candidate_scores)
                ),
                "inference_seconds": inference_seconds,
            }
        )
        print(
            "[{}/{}] {} {}/{} equal={}".format(
                index + 1,
                len(paths),
                path.name,
                decision.checkpoint_role,
                decision.mode,
                records[-1]["bitwise_equal_to_baseline"],
            ),
            flush=True,
        )

    if set(identity_by_checkpoint) != {"m10", "m20"} or not all(
        item["bitwise_equal"] for item in identity_by_checkpoint.values()
    ):
        raise RuntimeError("L=full identity was not proven for both checkpoints.")
    actual_route_counts = Counter(
        "{}/{}".format(
            record["decision"]["checkpoint_role"],
            record["decision"]["mode"],
        )
        for record in records
    )
    expected_route_counts = Counter(
        "{}/{}".format(decision["checkpoint_role"], decision["mode"])
        for decision in expected_decisions.values()
    )
    if actual_route_counts != expected_route_counts or actual_route_counts != Counter(
        {"m10/full_stream": 45, "m20/full_stream": 43, "m20/window_t32": 11}
    ):
        raise RuntimeError("Formal route counts differ from frozen 45/43/11 gates.")
    unchanged_records = [record for record in records if record["candidate_mode"] != "window_t32"]
    if len(unchanged_records) != 88 or not all(
        record["bitwise_equal_to_baseline"] for record in unchanged_records
    ):
        raise RuntimeError("The 88 unchanged M10/M20 full routes are not bitwise exact.")
    if sum(record["candidate_mode"] == "window_t32" for record in records) != 11:
        raise RuntimeError("The formal cache does not contain exactly 11 T32 routes.")
    manifest = {
        "schema": CACHE_SCHEMA,
        "created_utc": utc_now(),
        "complete": True,
        "split_access": {
            "dataset_split": "train",
            "validation_or_test_read": False,
            "labels_or_target_ids_indexed": False,
            "consumed": ["ev_loc[:,0:3]", "evs_norm[:,3] polarity"],
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "route_policy_sha256": route_policy_sha256(),
        "device": {
            "requested": str(device),
            "name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device.index))
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device.index))
                if device.type == "cuda"
                else None
            ),
        },
        "identity_checks": identity_by_checkpoint,
        "video_count": len(records),
        "event_count": int(sum(record["event_count"] for record in records)),
        "route_counts": dict(actual_route_counts),
        "route_gates": {
            "expected_45_m10_full": actual_route_counts["m10/full_stream"] == 45,
            "expected_43_m20_full": actual_route_counts["m20/full_stream"] == 43,
            "expected_11_m20_t32": actual_route_counts["m20/window_t32"] == 11,
            "unchanged_88_bitwise_equal": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "code": current_code_identity(),
    }
    manifest_path, digest, sidecar = atomic_json(output_dir / "manifest.json", manifest)
    print("manifest:", manifest_path)
    print("manifest_sha256:", digest)
    print("sha256_sidecar:", sidecar)
    return manifest


def confusion_counts(labels, scores, threshold):
    labels = np.asarray(labels).reshape(-1) > 0.5
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    predicted = scores >= float(threshold)
    return {
        "true_positive_events": int(np.count_nonzero(predicted & labels)),
        "false_positive_events": int(np.count_nonzero(predicted & ~labels)),
        "false_negative_events": int(np.count_nonzero(~predicted & labels)),
        "true_negative_events": int(np.count_nonzero(~predicted & ~labels)),
    }


def add_counts(target, source):
    for key, value in source.items():
        target[key] += int(value)


def evaluate_one(video, scores, threshold):
    scores = validate_probability_scores(scores, video.locations.shape[0], "evaluation")
    locations = torch.from_numpy(
        np.column_stack(
            (
                np.zeros(video.locations.shape[0], dtype=np.int64),
                video.locations.astype(np.int64, copy=False),
            )
        )
    ).long()
    postprocessor = ChallengePostprocessor.from_cfg(
        c00_config(), float(threshold), event_count=int(video.locations.shape[0])
    )
    postprocessed, stats = postprocessor.apply(scores.clone(), locations)
    labels = video.labels.astype(np.float32, copy=False)
    confusion = confusion_counts(labels, postprocessed.numpy(), threshold)
    evaluator = evalute(evaluator_config())
    label_tensor = torch.from_numpy(labels).float()
    evaluator.roc_update(
        locations[:, 3],
        postprocessed.clone(),
        video.target_ids.astype(np.int64, copy=False),
        label_tensor,
        locations,
        thresh=float(threshold),
    )
    return {
        **confusion,
        "correct_target_groups": int(evaluator.correct_num),
        "target_groups": int(evaluator.obj_num),
        "false_components": int(evaluator.false_num),
        "frame_count": int(evaluator.frame_num),
    }, stats.summary()


def normalized_counts(value):
    counts = {key: int(value[key]) for key in COUNT_KEYS}
    if any(number < 0 for number in counts.values()):
        raise ValueError("Metric sufficient counts must be non-negative.")
    return counts


def metrics_from_counts(value):
    counts = normalized_counts(value)
    tp = counts["true_positive_events"]
    fp = counts["false_positive_events"]
    fn = counts["false_negative_events"]
    union = tp + fp + fn
    positive = tp + fn
    if union <= 0 or positive <= 0 or counts["target_groups"] <= 0 or counts[
        "frame_count"
    ] <= 0:
        raise ValueError("Metric denominator is zero.")
    iou = float(
        (torch.tensor(tp, dtype=torch.float32) / torch.tensor(union, dtype=torch.float32)).item()
    )
    acc = float(
        (
            torch.tensor(tp, dtype=torch.float32)
            / torch.tensor(positive, dtype=torch.float32)
        ).item()
    )
    pd = counts["correct_target_groups"] / counts["target_groups"]
    fa = counts["false_components"] / (counts["frame_count"] * WIDTH * HEIGHT)
    score_fa, score = challenge_score(iou, acc, pd, fa)
    values = (iou, acc, pd, fa, score_fa, score)
    if not all(math.isfinite(number) for number in values):
        raise RuntimeError("Non-finite challenge metric.")
    return {
        "iou": iou,
        "acc": acc,
        "pd": pd,
        "fa": fa,
        "score_fa": score_fa,
        "score": score,
    }


def evaluation(counts):
    counts = normalized_counts(counts)
    return {"metrics": metrics_from_counts(counts), "counts": counts}


def evaluation_delta(baseline, candidate):
    return {
        "metrics": {
            key: candidate["metrics"][key] - baseline["metrics"][key]
            for key in baseline["metrics"]
        },
        "counts": {
            key: candidate["counts"][key] - baseline["counts"][key]
            for key in COUNT_KEYS
        },
    }


def evaluate_cache(args):
    require_persistence_second_stage_disabled(False)
    protocol, protocol_path = load_json(args.protocol)
    population = validate_protocol(protocol, protocol_path)
    manifest, manifest_path = load_json(Path(args.cache_dir) / "manifest.json")
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Input route cache is incomplete or has an unexpected schema.")
    if manifest.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("Cache was not created from this frozen protocol.")
    if manifest.get("route_policy_sha256") != route_policy_sha256():
        raise ValueError("Cache route policy drift detected.")
    if manifest.get("code") != current_code_identity():
        raise ValueError("Cache code identity differs from current evaluation code.")
    identities = manifest.get("identity_checks", {})
    if set(identities) != {"m10", "m20"} or not all(
        item.get("bitwise_equal") is True for item in identities.values()
    ):
        raise ValueError("Cache lacks both checkpoint L=full identity checks.")
    records = manifest.get("records")
    if not isinstance(records, list) or tuple(
        record.get("source_name") for record in records
    ) != OFFICIAL_TRAIN_NAMES:
        raise ValueError("Cache source population is incomplete or reordered.")

    train_root, paths = discover_official_train_sources(args.train_root)
    baseline_total = defaultdict(int)
    candidate_total = defaultdict(int)
    by_domain = {
        domain: {"baseline": defaultdict(int), "candidate": defaultdict(int)}
        for domain in ("low", "middle", "h1", "h2")
    }
    per_video = []
    cache_root = Path(args.cache_dir).resolve()
    for index, (path, record) in enumerate(zip(paths, records), start=1):
        expected_source_sha = population["source_sha256"].get(path.name)
        if sha256_file(path) != expected_source_sha or record.get(
            "source_sha256"
        ) != expected_source_sha:
            raise ValueError("Train source SHA mismatch: {}".format(path))
        record_path = (cache_root / record["record"]).resolve()
        try:
            record_path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError("Cache record escapes its cache directory.") from error
        if not record_path.is_file() or sha256_file(record_path) != record.get(
            "record_sha256"
        ):
            raise ValueError("Cache record hash mismatch: {}".format(record_path))
        with np.load(record_path, allow_pickle=False) as payload:
            if set(payload.files) != {"baseline_scores", "candidate_scores"}:
                raise ValueError("Cache record has unexpected arrays.")
            baseline_scores = np.asarray(payload["baseline_scores"], dtype=np.float32)
            candidate_scores = np.asarray(payload["candidate_scores"], dtype=np.float32)
        video = load_temporal_frame_video(path, TEMPORAL_BIN_SIZE, WHOLE_T)
        decision = select_temporal_memory_input_route(
            video.polarities, len(video.event_indices_by_bin)
        )
        runtime_frozen_decision = frozen_route_decision(decision.to_metadata())
        if runtime_frozen_decision != record.get("decision") or (
            runtime_frozen_decision != population["route_decisions"].get(path.name)
        ):
            raise RuntimeError("Evaluation route identity mismatch for {}.".format(path.name))
        if decision.mode != "window_t32" and not np.array_equal(
            baseline_scores, candidate_scores
        ):
            raise RuntimeError("Unchanged route cache differs for {}.".format(path.name))
        baseline_counts, baseline_postprocess = evaluate_one(
            video, baseline_scores, decision.prediction_threshold
        )
        candidate_counts, candidate_postprocess = evaluate_one(
            video, candidate_scores, decision.prediction_threshold
        )
        add_counts(baseline_total, baseline_counts)
        add_counts(candidate_total, candidate_counts)
        add_counts(by_domain[decision.domain]["baseline"], baseline_counts)
        add_counts(by_domain[decision.domain]["candidate"], candidate_counts)
        baseline_eval = evaluation(baseline_counts)
        candidate_eval = evaluation(candidate_counts)
        per_video.append(
            {
                "source_name": path.name,
                "source_sha256": expected_source_sha,
                "record_sha256": record["record_sha256"],
                "decision": runtime_frozen_decision,
                "baseline": baseline_eval,
                "candidate": candidate_eval,
                "delta": evaluation_delta(baseline_eval, candidate_eval),
                "postprocess": {
                    "baseline": baseline_postprocess,
                    "candidate": candidate_postprocess,
                },
            }
        )
        print(
            "[{}/{}] evaluated {} {}/{}".format(
                index,
                len(paths),
                path.name,
                decision.checkpoint_role,
                decision.mode,
            ),
            flush=True,
        )

    baseline = evaluation(baseline_total)
    candidate = evaluation(candidate_total)
    delta = evaluation_delta(baseline, candidate)
    domain_results = {}
    for domain, counts in by_domain.items():
        domain_baseline = evaluation(counts["baseline"])
        domain_candidate = evaluation(counts["candidate"])
        domain_results[domain] = {
            "baseline": domain_baseline,
            "candidate": domain_candidate,
            "delta": evaluation_delta(domain_baseline, domain_candidate),
        }
    report = {
        "schema": EVALUATION_SCHEMA,
        "created_utc": utc_now(),
        "evidence_class": "complete_train_fixed_route_diagnostic_not_validation_or_oof",
        "split_access": {
            "dataset_split": "train",
            "validation_or_test_read": False,
            "route_uses_labels_or_source_name": False,
            "train_labels_used_after_cache_verification_for_metrics": True,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "cache": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "evaluation": {
            "thresholds": protocol["route"]["prediction_threshold_by_checkpoint"],
            "postprocess_profile": POSTPROCESS_PROFILE,
            "postprocess_sha256": c00_sha256(),
            "pooling": "sum sufficient counts, then compute official metrics",
        },
        "pooled": {
            "baseline_released_route": baseline,
            "candidate_input_route": candidate,
            "delta": delta,
        },
        "by_domain": domain_results,
        "per_video": per_video,
        "promotion": {
            "claim": False,
            "reason": (
                "This is a fixed complete-train diagnostic; independent held-data "
                "confirmation is still required."
            ),
        },
        "persistence_second_stage": {"enabled": False},
        "provenance": {
            "train_root": str(train_root),
            "code": current_code_identity(),
        },
    }
    output, digest, sidecar = atomic_json(args.output, report)
    print("report:", output)
    print("report_sha256:", digest)
    print("sha256_sidecar:", sidecar)
    print("baseline_score:", baseline["metrics"]["score"])
    print("candidate_score:", candidate["metrics"]["score"])
    print("delta_score:", delta["metrics"]["score"])
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze the train-only protocol")
    freeze.add_argument("--audit", type=Path, required=True)
    freeze.add_argument("--amendment-record", type=Path, required=True)
    freeze.add_argument("--m10-checkpoint", type=Path, required=True)
    freeze.add_argument("--m20-checkpoint", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=freeze_protocol)

    cache = subparsers.add_parser("cache", help="create label-free routed score cache")
    cache.add_argument("--protocol", type=Path, required=True)
    cache.add_argument("--train-root", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.set_defaults(handler=cache_scores)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a verified train cache")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--train-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=evaluate_cache)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

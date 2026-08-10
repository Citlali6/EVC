"""Train-only grouped OOF for a tiny released-M20/confidence score blend.

The experiment is deliberately isolated from the released inference path.  It
has two fail-closed stages:

``cache``
    Read only complete train inputs (x/y/t/p), run one frozen confidence-only
    checkpoint on the preregistered 15 H1/H2 sources, and write raw scores.
    Labels and target identifiers are never indexed in this stage.

``evaluate``
    Verify every source/checkpoint/cache/protocol hash before opening labels,
    select alpha on fit source groups, and score the disjoint held groups with
    the unchanged official sufficient-count pooling and released C00 chain.

No command accepts validation or test paths.  This is candidate-selection
evidence, not an independent final estimate and not a deployment mutation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time

import numpy as np
import torch

import run_temporal_memory_input_route_train as route_train
from audit_temporal_memory_input_route_train import sha256_file
from utils.temporal_memory_inference import (
    load_temporal_memory_model,
    predict_temporal_memory_scores,
)
from utils.temporal_memory_input_router import select_temporal_memory_input_route


PROTOCOL_SCHEMA = "ev-uav-confidence-score-blend-grouped-oof-science-v1"
CACHE_SCHEMA = "ev-uav-confidence-score-blend-train-cache-v1"
REPORT_SCHEMA = "ev-uav-confidence-score-blend-grouped-oof-report-v1"
DATASET_SPLIT = "train"
PREDICTION_THRESHOLD = 0.719
CONTEXT_BINS = 5
MODEL_WIDTH = 16
WIDTH = 346
HEIGHT = 260
INFERENCE_BATCH_SIZE = 8
LOG_COUNT_CLIP = 4.0
EXPECTED_M20_SHA256 = (
    "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
)
EXPECTED_CONFIDENCE_SHA256 = (
    "c578e27cd4e05d1837fcf969b989cd3911dcbe979a0926aa386f4be51f4ceaa5"
)
EXPECTED_BASELINE_MANIFEST_SHA256 = (
    "05a707dcfeb8487fafdb99599abfff81b452c6fac9d1938da47f711097257f82"
)
EXPECTED_IDENTITY_MANIFEST_SHA256 = (
    "78ca63efd1fd8fda62dcccb1203f0e69000007454a391b7d46455f9952cf2dc7"
)
ALPHAS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
IDENTITY_ALPHA = 0.0
H1_NAMES = tuple("train_{:03d}.npz".format(i) for i in range(44, 48))
H2_NAMES = tuple("train_{:03d}.npz".format(i) for i in range(88, 99))
SOURCE_NAMES = H1_NAMES + H2_NAMES
FORBIDDEN_SPLITS = frozenset({"val", "validation", "test"})
COUNT_KEYS = route_train.COUNT_KEYS


FOLD_PLAN = (
    {
        "fold_id": "h1_holdout_044_045",
        "domain": "h1",
        "fit_names": H1_NAMES[2:],
        "held_names": H1_NAMES[:2],
    },
    {
        "fold_id": "h1_holdout_046_047",
        "domain": "h1",
        "fit_names": H1_NAMES[:2],
        "held_names": H1_NAMES[2:],
    },
    {
        "fold_id": "h2_holdout_088_091",
        "domain": "h2",
        "fit_names": H2_NAMES[4:],
        "held_names": H2_NAMES[:4],
    },
    {
        "fold_id": "h2_holdout_092_094",
        "domain": "h2",
        "fit_names": H2_NAMES[:4] + H2_NAMES[7:],
        "held_names": H2_NAMES[4:7],
    },
    {
        "fold_id": "h2_holdout_095_098",
        "domain": "h2",
        "fit_names": H2_NAMES[:7],
        "held_names": H2_NAMES[7:],
    },
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Refusing to overwrite: {}".format(path))
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
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(digest + "  " + path.name + "\n", encoding="ascii")
    return path, digest


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


def load_json(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream), path


def _reject_forbidden_path(path, label, allow_train_leaf=False):
    path = Path(path).resolve()
    parts = [part.lower() for part in path.parts]
    if allow_train_leaf and parts and parts[-1] == DATASET_SPLIT:
        parts = parts[:-1]
    forbidden = sorted(set(parts) & FORBIDDEN_SPLITS)
    if forbidden:
        raise ValueError(
            "{} contains a forbidden split component {}: {}".format(
                label, forbidden, path
            )
        )
    return path


def validate_train_root(path):
    path = _reject_forbidden_path(path, "train root", allow_train_leaf=True)
    if path.name.lower() != DATASET_SPLIT or not path.is_dir():
        raise ValueError("--train-root must be the official train directory.")
    return path


def _expected_fold_plan_json():
    return [
        {
            "fold_id": fold["fold_id"],
            "domain": fold["domain"],
            "fit_names": list(fold["fit_names"]),
            "held_names": list(fold["held_names"]),
        }
        for fold in FOLD_PLAN
    ]


def validate_protocol(path, require_runner_hash=True):
    protocol, path = load_json(path)
    _reject_forbidden_path(path, "protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unexpected confidence-blend protocol schema.")
    access = protocol.get("split_access", {})
    if access.get("dataset_split") != DATASET_SPLIT or access.get(
        "validation_or_test_allowed"
    ) is not False:
        raise ValueError("Protocol is not strictly train-only.")
    baseline = protocol.get("baseline", {})
    candidate = protocol.get("candidate", {})
    if baseline.get("checkpoint_sha256") != EXPECTED_M20_SHA256:
        raise ValueError("Released M20 hash drift.")
    if candidate.get("checkpoint_sha256") != EXPECTED_CONFIDENCE_SHA256:
        raise ValueError("Confidence checkpoint hash drift.")
    if baseline.get("label_score_cache_manifest_sha256") != (
        EXPECTED_BASELINE_MANIFEST_SHA256
    ):
        raise ValueError("Baseline train-cache binding drift.")
    if baseline.get("identity_score_cache_manifest_sha256") != (
        EXPECTED_IDENTITY_MANIFEST_SHA256
    ):
        raise ValueError("M20 identity-cache binding drift.")
    if float(protocol.get("inference", {}).get("prediction_threshold", -1)) != (
        PREDICTION_THRESHOLD
    ):
        raise ValueError("Prediction threshold is not frozen at 0.719.")
    blend = protocol.get("blend", {})
    if tuple(float(value) for value in blend.get("alpha_grid", [])) != ALPHAS:
        raise ValueError("Alpha grid drift.")
    if float(blend.get("identity_control_alpha", -1)) != IDENTITY_ALPHA:
        raise ValueError("Identity control drift.")
    if blend.get("arithmetic") != (
        "float32((1-alpha)*released_m20_full + alpha*confidence_full)"
    ):
        raise ValueError("Blend arithmetic drift.")
    population = protocol.get("source_population", {})
    if tuple(population.get("names", [])) != SOURCE_NAMES:
        raise ValueError("Frozen 15-source population drift.")
    source_sha = population.get("source_sha256", {})
    if set(source_sha) != set(SOURCE_NAMES):
        raise ValueError("Source hash population is incomplete.")
    if protocol.get("fold_plan") != _expected_fold_plan_json():
        raise ValueError("Grouped OOF fold plan drift.")
    if require_runner_hash and protocol.get("code", {}).get("runner_sha256") != (
        sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("Frozen runner hash differs from current code.")
    for label, item, expected in (
        ("M20", baseline, EXPECTED_M20_SHA256),
        ("confidence", candidate, EXPECTED_CONFIDENCE_SHA256),
    ):
        checkpoint = _reject_forbidden_path(item.get("checkpoint_path", ""), label)
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
            raise ValueError("{} checkpoint file/hash mismatch.".format(label))
    for key, expected in (
        ("label_score_cache_manifest", EXPECTED_BASELINE_MANIFEST_SHA256),
        ("identity_score_cache_manifest", EXPECTED_IDENTITY_MANIFEST_SHA256),
    ):
        manifest = _reject_forbidden_path(baseline.get(key, ""), key)
        if not manifest.is_file() or sha256_file(manifest) != expected:
            raise ValueError("{} file/hash mismatch.".format(key))
    for key, expected_key in (
        ("training_config_path", "training_config_sha256"),
        ("run_summary_path", "run_summary_sha256"),
    ):
        artifact = _reject_forbidden_path(candidate.get(key, ""), key)
        if not artifact.is_file() or sha256_file(artifact) != candidate.get(expected_key):
            raise ValueError("Confidence training artifact hash mismatch: {}".format(key))
    return protocol, path


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_confidence_only_identity(protocol):
    m20 = _torch_load_cpu(protocol["baseline"]["checkpoint_path"])
    confidence = _torch_load_cpu(protocol["candidate"]["checkpoint_path"])
    m20_state = m20.get("model_state_dict", m20)
    confidence_state = confidence.get("model_state_dict", confidence)
    common = set(m20_state) & set(confidence_state)
    changed = sorted(
        name for name in common if not torch.equal(m20_state[name], confidence_state[name])
    )
    extras = sorted(set(confidence_state) - set(m20_state))
    missing = sorted(set(m20_state) - set(confidence_state))
    expected_extras = [
        "base.confidence_head.layers.0.weight",
        "base.confidence_head.layers.1.bias",
        "base.confidence_head.layers.1.weight",
        "base.confidence_head.layers.3.bias",
        "base.confidence_head.layers.3.weight",
    ]
    if changed or missing or extras != expected_extras or len(common) != 89:
        raise RuntimeError("Confidence-only tensor identity audit failed.")
    saved = confidence.get("temporal_memory", {})
    provenance = confidence.get("provenance", {})
    if saved.get("confidence_head_enabled") is not True or provenance.get(
        "initialized_from_sha256"
    ) != EXPECTED_M20_SHA256:
        raise RuntimeError("Confidence checkpoint metadata identity failed.")
    scope = provenance.get("training_scope", {})
    if scope.get("name") != "confidence_only" or int(
        scope.get("trainable_parameter_count", -1)
    ) != 2353:
        raise RuntimeError("Confidence training scope metadata failed.")
    return {
        "common_tensor_count": len(common),
        "bitwise_unchanged_common_tensor_count": len(common) - len(changed),
        "confidence_only_tensor_names": extras,
        "missing_tensor_names": missing,
        "changed_common_tensor_names": changed,
        "training_scope": scope,
    }


def _manifest_records(manifest, expected_names):
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Cache manifest records are invalid.")
    by_name = {item.get("source_name"): item for item in records}
    if not set(expected_names).issubset(by_name):
        raise ValueError("Cache manifest lacks preregistered sources.")
    return by_name


def _load_score_only(cache_root, metadata, key):
    cache_root = Path(cache_root).resolve()
    record_path = (cache_root / metadata["record"]).resolve()
    try:
        record_path.relative_to(cache_root)
    except ValueError as error:
        raise ValueError("Cache record escapes cache root.") from error
    if not record_path.is_file() or sha256_file(record_path) != metadata.get(
        "record_sha256"
    ):
        raise ValueError("Cache record hash mismatch: {}".format(record_path))
    with np.load(record_path, allow_pickle=False) as payload:
        scores = np.asarray(payload[key], dtype=np.float32).reshape(-1).copy()
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Cached scores are not finite probabilities.")
    return scores, record_path


def _domain_for_name(name):
    if name in H1_NAMES:
        return "h1"
    if name in H2_NAMES:
        return "h2"
    raise ValueError("Source is outside the frozen H1/H2 population.")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_scores(args):
    protocol, protocol_path = validate_protocol(args.protocol)
    train_root = validate_train_root(args.train_root)
    output_dir = _reject_forbidden_path(args.output_dir, "output cache")
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse output cache: {}".format(output_dir))
    for name in SOURCE_NAMES:
        if not (train_root / name).is_file():
            raise FileNotFoundError(train_root / name)
    identity_audit = verify_confidence_only_identity(protocol)

    baseline_manifest_path = Path(
        protocol["baseline"]["label_score_cache_manifest"]
    ).resolve()
    baseline_manifest, _ = load_json(baseline_manifest_path)
    if baseline_manifest.get("dataset_split") != DATASET_SPLIT:
        raise ValueError("Baseline cache is not train-only.")
    baseline_root = baseline_manifest_path.parent
    baseline_by_name = _manifest_records(baseline_manifest, SOURCE_NAMES)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is None:
            raise RuntimeError("Formal cache requires an explicit available CUDA device.")
        torch.cuda.set_device(device.index)
        torch.cuda.reset_peak_memory_stats(device.index)
    model, checkpoint = load_temporal_memory_model(
        protocol["candidate"]["checkpoint_path"],
        device,
        CONTEXT_BINS,
        MODEL_WIDTH,
        16,
    )
    if not bool(getattr(model, "confidence_head_enabled", False)):
        raise RuntimeError("Loaded candidate does not enable its confidence head.")
    if checkpoint.get("temporal_memory", {}).get("confidence_only_enabled") is not True:
        raise RuntimeError("Loaded candidate lacks confidence-only metadata.")

    output_dir.mkdir(parents=True)
    records_dir = output_dir / "records"
    records_dir.mkdir()
    records = []
    started = time.perf_counter()
    source_hashes = protocol["source_population"]["source_sha256"]
    for index, name in enumerate(SOURCE_NAMES):
        path = train_root / name
        if sha256_file(path) != source_hashes[name]:
            raise ValueError("Official train source hash mismatch: {}".format(name))
        video = route_train.load_input_only_video(path)
        decision = select_temporal_memory_input_route(
            video.polarities, len(video.event_indices_by_bin)
        )
        expected_domain = _domain_for_name(name)
        if decision.domain != expected_domain or decision.checkpoint_role != "m20":
            raise RuntimeError("Input-only route population drift: {}".format(name))
        _sync(device)
        inference_started = time.perf_counter()
        confidence_scores = predict_temporal_memory_scores(
            model=model,
            video=video,
            device=device,
            context_bins=CONTEXT_BINS,
            width=WIDTH,
            height=HEIGHT,
            inference_batch_size=INFERENCE_BATCH_SIZE,
            log_count_clip=LOG_COUNT_CLIP,
        ).numpy().astype(np.float32, copy=False)
        _sync(device)
        inference_seconds = time.perf_counter() - inference_started
        confidence_scores = route_train.validate_probability_scores(
            confidence_scores, decision.event_count, "confidence"
        ).numpy()
        baseline_metadata = baseline_by_name[name]
        if baseline_metadata.get("source_sha256") != source_hashes[name]:
            raise ValueError("Baseline cache source hash mismatch.")
        baseline_scores, _ = _load_score_only(
            baseline_root, baseline_metadata, "scores"
        )
        if baseline_scores.shape != confidence_scores.shape:
            raise RuntimeError("Candidate/baseline event alignment mismatch.")
        increase = confidence_scores.astype(np.float64) - baseline_scores.astype(
            np.float64
        )
        if float(np.max(increase)) > 1e-7:
            raise RuntimeError("Confidence score exceeds its unchanged M20 base.")
        positive_base = baseline_scores > 0
        ratio = np.divide(
            confidence_scores,
            baseline_scores,
            out=np.ones_like(confidence_scores),
            where=positive_base,
        )
        if positive_base.any() and (
            float(np.min(ratio[positive_base])) < -1e-7
            or float(np.max(ratio[positive_base])) > 1.0000002
        ):
            raise RuntimeError("Recovered confidence multiplier is outside [0,1].")
        record_path = records_dir / "{:03d}.npz".format(index)
        atomic_npz(record_path, confidence_scores=confidence_scores)
        records.append(
            {
                "source_name": name,
                "source_sha256": source_hashes[name],
                "domain": decision.domain,
                "event_count": int(decision.event_count),
                "record": str(record_path.relative_to(output_dir)).replace("\\", "/"),
                "record_sha256": sha256_file(record_path),
                "inference_seconds": inference_seconds,
                "mechanism_audit": {
                    "max_candidate_minus_baseline": float(np.max(increase)),
                    "mean_baseline_minus_candidate": float(
                        np.mean(
                            baseline_scores.astype(np.float64)
                            - confidence_scores.astype(np.float64)
                        )
                    ),
                    "strictly_attenuated_event_count": int(
                        np.count_nonzero(confidence_scores < baseline_scores)
                    ),
                    "min_recovered_confidence": float(
                        np.min(ratio[positive_base]) if positive_base.any() else 1.0
                    ),
                    "max_recovered_confidence": float(
                        np.max(ratio[positive_base]) if positive_base.any() else 1.0
                    ),
                },
            }
        )
        print(
            "[{}/{}] {} domain={} events={} seconds={:.3f}".format(
                index + 1,
                len(SOURCE_NAMES),
                name,
                decision.domain,
                decision.event_count,
                inference_seconds,
            ),
            flush=True,
        )

    manifest = {
        "schema": CACHE_SCHEMA,
        "created_utc": utc_now(),
        "complete": True,
        "dataset_split": DATASET_SPLIT,
        "validation_or_test_read": False,
        "labels_or_target_ids_read": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "checkpoint": {
            "path": protocol["candidate"]["checkpoint_path"],
            "sha256": EXPECTED_CONFIDENCE_SHA256,
        },
        "inference": protocol["inference"],
        "identity_audit": identity_audit,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    path, digest = atomic_json(output_dir / "manifest.json", manifest)
    print("manifest:", path)
    print("manifest_sha256:", digest)
    return manifest


def blend_scores(baseline_scores, confidence_scores, alpha):
    baseline_scores = np.asarray(baseline_scores, dtype=np.float32).reshape(-1)
    confidence_scores = np.asarray(confidence_scores, dtype=np.float32).reshape(-1)
    if baseline_scores.shape != confidence_scores.shape:
        raise ValueError("Blend score vectors differ in shape.")
    alpha32 = np.float32(alpha)
    if not np.isfinite(alpha32) or alpha32 < 0 or alpha32 > 1:
        raise ValueError("alpha must lie in [0,1].")
    blended = (
        (np.float32(1.0) - alpha32) * baseline_scores
        + alpha32 * confidence_scores
    ).astype(np.float32, copy=False)
    if float(alpha) == IDENTITY_ALPHA and not np.array_equal(
        blended, baseline_scores
    ):
        raise RuntimeError("alpha=0 failed exact score identity.")
    if np.any(blended > baseline_scores + np.float32(1e-7)):
        raise RuntimeError("Attenuation blend increased a baseline score.")
    return blended


def _zero_counts():
    return {key: 0 for key in COUNT_KEYS}


def add_counts(target, source):
    for key in COUNT_KEYS:
        target[key] += int(source[key])


def sum_counts(names, table, alpha=None):
    total = _zero_counts()
    for name in names:
        source = table[name] if alpha is None else table[name][alpha]
        add_counts(total, source)
    return total


def evaluation(counts):
    counts = {key: int(counts[key]) for key in COUNT_KEYS}
    return {"counts": counts, "metrics": route_train.metrics_from_counts(counts)}


def evaluation_delta(baseline, candidate):
    return route_train.evaluation_delta(baseline, candidate)


def fit_gate(delta):
    metrics = delta["metrics"]
    counts = delta["counts"]
    checks = {
        "score_positive": metrics["score"] > 0.0,
        "pd_nonnegative": metrics["pd"] >= 0.0,
        "iou_nonnegative": metrics["iou"] >= 0.0,
        "fa_nonpositive": metrics["fa"] <= 0.0,
        "true_positive_events_zero_loss": counts["true_positive_events"] == 0,
        "correct_target_groups_zero_loss": counts["correct_target_groups"] == 0,
        "false_alarm_evidence": (
            counts["false_positive_events"] <= -1
            or counts["false_components"] <= -1
        ),
    }
    return all(checks.values()), checks


def held_gate(delta):
    metrics = delta["metrics"]
    counts = delta["counts"]
    checks = {
        "score_nonnegative": metrics["score"] >= 0.0,
        "pd_nonnegative": metrics["pd"] >= 0.0,
        "iou_nonnegative": metrics["iou"] >= 0.0,
        "fa_nonpositive": metrics["fa"] <= 0.0,
        "true_positive_events_zero_loss": counts["true_positive_events"] == 0,
        "correct_target_groups_zero_loss": counts["correct_target_groups"] == 0,
    }
    return all(checks.values()), checks


def pooled_gate(delta):
    metrics = delta["metrics"]
    counts = delta["counts"]
    checks = {
        "score_delta_at_least_0p0002": metrics["score"] >= 0.0002,
        "pd_nonnegative": metrics["pd"] >= 0.0,
        "iou_nonnegative": metrics["iou"] >= 0.0,
        "fa_nonpositive": metrics["fa"] <= 0.0,
        "true_positive_events_zero_loss": counts["true_positive_events"] == 0,
        "correct_target_groups_zero_loss": counts["correct_target_groups"] == 0,
        "false_positive_events_reduced": counts["false_positive_events"] <= -1,
        "false_components_reduced": counts["false_components"] <= -1,
    }
    return all(checks.values()), checks


def select_alpha(fit_results):
    passing = []
    for alpha in ALPHAS:
        result = fit_results[alpha]
        passed, checks = fit_gate(result["delta"])
        result["fit_gate"] = {"passed": passed, "checks": checks}
        if passed:
            passing.append((alpha, result))
    if not passing:
        return None
    passing.sort(key=lambda item: (-item[1]["delta"]["metrics"]["score"], item[0]))
    return float(passing[0][0])


def _load_label_record(cache_root, metadata):
    cache_root = Path(cache_root).resolve()
    record_path = (cache_root / metadata["record"]).resolve()
    try:
        record_path.relative_to(cache_root)
    except ValueError as error:
        raise ValueError("Label cache record escapes cache root.") from error
    if sha256_file(record_path) != metadata.get("record_sha256"):
        raise ValueError("Label cache record hash mismatch.")
    with np.load(record_path, allow_pickle=False) as payload:
        required = {"scores", "locs", "labels", "target_ids"}
        if not required.issubset(payload.files):
            raise ValueError("Official train cache record schema drift.")
        return {
            "scores": np.asarray(payload["scores"], dtype=np.float32).reshape(-1).copy(),
            "locs": np.asarray(payload["locs"], dtype=np.int64).copy(),
            "labels": np.asarray(payload["labels"], dtype=np.uint8).reshape(-1).copy(),
            "target_ids": np.asarray(payload["target_ids"], dtype=np.int64).reshape(-1).copy(),
        }


def evaluate(args):
    protocol, protocol_path = validate_protocol(args.protocol)
    train_root = validate_train_root(args.train_root)
    output = _reject_forbidden_path(args.output, "evaluation output")
    if output.exists():
        raise FileExistsError(output)

    candidate_manifest, candidate_manifest_path = load_json(
        Path(args.cache_dir).resolve() / "manifest.json"
    )
    _reject_forbidden_path(candidate_manifest_path, "candidate cache")
    if (
        candidate_manifest.get("schema") != CACHE_SCHEMA
        or candidate_manifest.get("complete") is not True
        or candidate_manifest.get("dataset_split") != DATASET_SPLIT
        or candidate_manifest.get("labels_or_target_ids_read") is not False
        or candidate_manifest.get("protocol", {}).get("sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError("Candidate cache/protocol identity failed.")
    candidate_root = candidate_manifest_path.parent
    candidate_by_name = _manifest_records(candidate_manifest, SOURCE_NAMES)

    baseline_manifest_path = Path(
        protocol["baseline"]["label_score_cache_manifest"]
    ).resolve()
    baseline_manifest, _ = load_json(baseline_manifest_path)
    if (
        sha256_file(baseline_manifest_path) != EXPECTED_BASELINE_MANIFEST_SHA256
        or baseline_manifest.get("dataset_split") != DATASET_SPLIT
    ):
        raise ValueError("Official train baseline cache identity failed.")
    baseline_root = baseline_manifest_path.parent
    baseline_by_name = _manifest_records(baseline_manifest, SOURCE_NAMES)

    source_hashes = protocol["source_population"]["source_sha256"]
    confidence_score_by_name = {}
    # Pre-label integrity pass: raw input route, every file/cache hash, event
    # alignment and monotone confidence mechanism must pass for all 15 sources.
    for name in SOURCE_NAMES:
        path = train_root / name
        if sha256_file(path) != source_hashes[name]:
            raise ValueError("Official train source hash mismatch: {}".format(name))
        video = route_train.load_input_only_video(path)
        decision = select_temporal_memory_input_route(
            video.polarities, len(video.event_indices_by_bin)
        )
        if decision.domain != _domain_for_name(name) or decision.checkpoint_role != "m20":
            raise RuntimeError("Input-only route drift before label access.")
        candidate_meta = candidate_by_name[name]
        baseline_meta = baseline_by_name[name]
        if (
            candidate_meta.get("source_sha256") != source_hashes[name]
            or baseline_meta.get("source_sha256") != source_hashes[name]
            or candidate_meta.get("domain") != decision.domain
        ):
            raise RuntimeError("Cache source/domain identity drift.")
        confidence_scores, _ = _load_score_only(
            candidate_root, candidate_meta, "confidence_scores"
        )
        baseline_scores, _ = _load_score_only(
            baseline_root, baseline_meta, "scores"
        )
        if confidence_scores.size != decision.event_count or not np.all(
            confidence_scores <= baseline_scores + np.float32(1e-7)
        ):
            raise RuntimeError("Confidence mechanism/alignment failed before labels.")
        confidence_score_by_name[name] = confidence_scores

    baseline_counts_by_name = {}
    candidate_counts_by_name = {name: {} for name in SOURCE_NAMES}
    per_source = {}
    for index, name in enumerate(SOURCE_NAMES, start=1):
        record = _load_label_record(baseline_root, baseline_by_name[name])
        confidence_scores = confidence_score_by_name[name]
        event_count = record["scores"].size
        if (
            record["locs"].shape != (event_count, 3)
            or record["labels"].size != event_count
            or record["target_ids"].size != event_count
            or confidence_scores.size != event_count
        ):
            raise RuntimeError("Official train label record alignment failed.")
        video = SimpleNamespace(
            locations=record["locs"],
            labels=record["labels"],
            target_ids=record["target_ids"],
        )
        baseline_counts, _ = route_train.evaluate_one(
            video, record["scores"], PREDICTION_THRESHOLD
        )
        identity_counts, _ = route_train.evaluate_one(
            video,
            blend_scores(record["scores"], confidence_scores, IDENTITY_ALPHA),
            PREDICTION_THRESHOLD,
        )
        if identity_counts != baseline_counts:
            raise RuntimeError("alpha=0 failed exact official-count identity.")
        baseline_counts_by_name[name] = baseline_counts
        per_alpha = {}
        for alpha in ALPHAS:
            candidate_counts, _ = route_train.evaluate_one(
                video,
                blend_scores(record["scores"], confidence_scores, alpha),
                PREDICTION_THRESHOLD,
            )
            candidate_counts_by_name[name][alpha] = candidate_counts
            baseline_eval = evaluation(baseline_counts)
            candidate_eval = evaluation(candidate_counts)
            per_alpha[str(alpha)] = {
                "candidate": candidate_eval,
                "delta": evaluation_delta(baseline_eval, candidate_eval),
            }
        per_source[name] = {
            "domain": _domain_for_name(name),
            "event_count": event_count,
            "baseline": evaluation(baseline_counts),
            "by_alpha": per_alpha,
        }
        print(
            "[{}/{}] evaluated {}".format(index, len(SOURCE_NAMES), name),
            flush=True,
        )

    folds = []
    pooled_baseline_counts = _zero_counts()
    pooled_candidate_counts = _zero_counts()
    held_seen = []
    all_folds_selected_positive = True
    all_held_passed = True
    for fold in FOLD_PLAN:
        fit_baseline = evaluation(
            sum_counts(fold["fit_names"], baseline_counts_by_name)
        )
        fit_results = {}
        for alpha in ALPHAS:
            candidate_eval = evaluation(
                sum_counts(fold["fit_names"], candidate_counts_by_name, alpha)
            )
            fit_results[alpha] = {
                "candidate": candidate_eval,
                "delta": evaluation_delta(fit_baseline, candidate_eval),
            }
        selected_alpha = select_alpha(fit_results)
        selection_status = "selected_positive_alpha"
        if selected_alpha is None:
            all_folds_selected_positive = False
            selection_status = "no_fit_candidate_fail_closed_identity"
            selected_alpha = IDENTITY_ALPHA
        held_baseline_counts = sum_counts(
            fold["held_names"], baseline_counts_by_name
        )
        if selected_alpha == IDENTITY_ALPHA:
            held_candidate_counts = dict(held_baseline_counts)
        else:
            held_candidate_counts = sum_counts(
                fold["held_names"], candidate_counts_by_name, selected_alpha
            )
        held_baseline = evaluation(held_baseline_counts)
        held_candidate = evaluation(held_candidate_counts)
        held_delta = evaluation_delta(held_baseline, held_candidate)
        held_passed, held_checks = held_gate(held_delta)
        all_held_passed = all_held_passed and held_passed
        add_counts(pooled_baseline_counts, held_baseline_counts)
        add_counts(pooled_candidate_counts, held_candidate_counts)
        held_seen.extend(fold["held_names"])
        folds.append(
            {
                "fold_id": fold["fold_id"],
                "domain": fold["domain"],
                "fit_names": list(fold["fit_names"]),
                "held_names": list(fold["held_names"]),
                "selection_status": selection_status,
                "selected_alpha": selected_alpha,
                "fit": {
                    "baseline": fit_baseline,
                    "by_alpha": {str(alpha): fit_results[alpha] for alpha in ALPHAS},
                },
                "held": {
                    "baseline": held_baseline,
                    "candidate": held_candidate,
                    "delta": held_delta,
                    "gate": {"passed": held_passed, "checks": held_checks},
                },
            }
        )
    if tuple(held_seen) != SOURCE_NAMES or len(set(held_seen)) != len(SOURCE_NAMES):
        raise RuntimeError("Held partitions do not cover each source exactly once.")

    pooled_baseline = evaluation(pooled_baseline_counts)
    pooled_candidate = evaluation(pooled_candidate_counts)
    pooled_delta = evaluation_delta(pooled_baseline, pooled_candidate)
    pooled_passed, pooled_checks = pooled_gate(pooled_delta)

    full_baseline = evaluation(sum_counts(SOURCE_NAMES, baseline_counts_by_name))
    full_results = {}
    for alpha in ALPHAS:
        candidate_eval = evaluation(
            sum_counts(SOURCE_NAMES, candidate_counts_by_name, alpha)
        )
        result = {
            "candidate": candidate_eval,
            "delta": evaluation_delta(full_baseline, candidate_eval),
        }
        passed, checks = fit_gate(result["delta"])
        result["fit_gate"] = {"passed": passed, "checks": checks}
        full_results[alpha] = result
    final_alpha = select_alpha(full_results)
    final_pooled_passed = False
    final_pooled_checks = {}
    if final_alpha is not None:
        final_pooled_passed, final_pooled_checks = pooled_gate(
            full_results[final_alpha]["delta"]
        )
    promotion_passed = bool(
        all_folds_selected_positive
        and all_held_passed
        and pooled_passed
        and final_alpha is not None
        and final_pooled_passed
    )
    failure_reasons = []
    if not all_folds_selected_positive:
        failure_reasons.append("at_least_one_fold_had_no_fit_gated_positive_alpha")
    if not all_held_passed:
        failure_reasons.append("at_least_one_held_fold_failed_hard_gates")
    if not pooled_passed:
        failure_reasons.append("pooled_oof_failed_promotion_gates")
    if final_alpha is None or not final_pooled_passed:
        failure_reasons.append("no_deployable_all_train_alpha_passed_pooled_gates")

    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "evidence_class": (
            "train_only_grouped_oof_candidate_selection_not_independent_final_estimate"
        ),
        "split_access": {
            "dataset_split": DATASET_SPLIT,
            "validation_or_test_read": False,
            "candidate_scores_cached_before_label_access": True,
            "prelabel_integrity_passed_for_all_15_sources": True,
        },
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "candidate_cache": {
            "manifest_path": str(candidate_manifest_path),
            "manifest_sha256": sha256_file(candidate_manifest_path),
        },
        "evaluation": {
            "prediction_threshold": PREDICTION_THRESHOLD,
            "postprocess_profile": "released_M20_C00_fixed",
            "pooling": "sum official sufficient counts, then compute metrics",
            "blend": protocol["blend"],
        },
        "folds": folds,
        "pooled_oof": {
            "baseline": pooled_baseline,
            "candidate": pooled_candidate,
            "delta": pooled_delta,
            "gate": {"passed": pooled_passed, "checks": pooled_checks},
        },
        "all_train_refit": {
            "warning": "used only to instantiate a deployable alpha after OOF gates",
            "baseline": full_baseline,
            "by_alpha": {str(alpha): full_results[alpha] for alpha in ALPHAS},
            "selected_alpha": final_alpha,
            "pooled_gate": {
                "passed": final_pooled_passed,
                "checks": final_pooled_checks,
            },
        },
        "promotion": {
            "passed": promotion_passed,
            "failure_reasons": failure_reasons,
            "on_failure": "eliminate_confidence_score_blend_without_validation_attempt",
            "on_pass": (
                "eligible_for_separately_authorized_frozen_validation_once_only"
            ),
        },
        "per_source": per_source,
        "provenance": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        },
    }
    path, digest = atomic_json(output, report)
    print("report:", path)
    print("report_sha256:", digest)
    print("promotion_passed:", promotion_passed)
    print("pooled_score_delta:", pooled_delta["metrics"]["score"])
    print("final_alpha:", final_alpha)
    return report


def verify(args):
    protocol, path = validate_protocol(args.protocol)
    identity = verify_confidence_only_identity(protocol)
    print("protocol:", path)
    print("protocol_sha256:", sha256_file(path))
    print("runner_sha256:", sha256_file(Path(__file__).resolve()))
    print("confidence_identity:", json.dumps(identity, sort_keys=True))
    return identity


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="CPU-only frozen audit")
    verify_parser.add_argument("--protocol", type=Path, required=True)
    verify_parser.set_defaults(handler=verify)

    cache_parser = subparsers.add_parser("cache", help="label-free 15-source cache")
    cache_parser.add_argument("--protocol", type=Path, required=True)
    cache_parser.add_argument("--train-root", type=Path, required=True)
    cache_parser.add_argument("--output-dir", type=Path, required=True)
    cache_parser.add_argument("--device", default="cuda:0")
    cache_parser.set_defaults(handler=cache_scores)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="verify cache, then run grouped OOF on train labels"
    )
    evaluate_parser.add_argument("--protocol", type=Path, required=True)
    evaluate_parser.add_argument("--train-root", type=Path, required=True)
    evaluate_parser.add_argument("--cache-dir", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

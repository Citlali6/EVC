"""Frozen first-fold train/evaluate overlay for the middle temporal expert.

The overlay reuses the already-audited multi-scale training/inference core, but
binds it to the middle-domain science contract, fit families F2--F5, held family
F1, and a released-M20 train-cache baseline.  CPU audit and GPU train/evaluate
are separate commands.  Held arrays cannot be opened by the training command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import crossfit_component_reranker as crossfit


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SCIENCE_PATH = ROOT / "protocols" / "middle_multiscale_temporal_summary_expert_science_v1.json"
EXECUTION_PATH = ROOT / "protocols" / "middle_multiscale_temporal_summary_first_fold_execution_v1.json"
CORE_PATH = ROOT / "run_h2_multiscale_temporal_pyramid_formal.py"
TRAIN_CACHE_MANIFEST_PATH = (
    WORKSPACE
    / "experiments"
    / "20260810_component_reranker_crosssource_v1"
    / "train_cache_gt30000"
    / "manifest.json"
)
TRAIN_CACHE_ROOT = TRAIN_CACHE_MANIFEST_PATH.parent
TRAIN_DATA_ROOT = (WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train").resolve()
OUTPUT_ROOT = (
    WORKSPACE
    / "experiments"
    / "20260811_middle_multiscale_temporal_summary_grouped_oof_v1"
    / "first_fold_formal"
)
CPU_AUDIT_PATH = OUTPUT_ROOT / "cpu_preflight.json"
TRAIN_OUTPUT_ROOT = OUTPUT_ROOT / "formal_training" / "middle_hold_f1_000_014"
CHECKPOINT_PATH = TRAIN_OUTPUT_ROOT / "final_expert.pt"
TRAINING_RECEIPT_PATH = TRAIN_OUTPUT_ROOT / "training_result.json"
MIDDLE_TRAINING_GATE_PATH = TRAIN_OUTPUT_ROOT / "middle_training_gate.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation" / "middle_hold_f1_000_014"
EVALUATION_PATH = EVALUATION_ROOT / "paired_evaluation.json"
DECISION_PATH = EVALUATION_ROOT / "branch_decision.json"
MIDDLE_REPORT_PATH = EVALUATION_ROOT / "middle_first_fold_report.json"
FAILURE_PATH = OUTPUT_ROOT / "failure_receipt.json"

EXPECTED_SCIENCE_SHA256 = "f17c689186fbfff1763460907f2ae9a5093e992315d004e0b4db537754c5dafe"
EXPECTED_EXECUTION_SHA256 = "13dbc2efab6f54a4fa85bf86f012d994901c62b36c7fc14b1ecc459131bb6b96"
EXPECTED_CORE_SHA256 = "33ca6078c810f52c8a549376bb8d862bae1068561bd7ccd8e5c1479e1f8fe89f"
EXPECTED_MANIFEST_SHA256 = "05a707dcfeb8487fafdb99599abfff81b452c6fac9d1938da47f711097257f82"
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EXPECTED_MODEL_SHA256 = "4d4ea4a365be49ad1b6c7cf1c7c96c2369caf3e12841bbcd781cf109105a6a98"
EXPECTED_LOSS_SHA256 = "f74e145b04b25f2e7478f5c8fd370bc4e9d96123ef6f75ea5833acc210d2c5e9"
EXPECTED_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
EXPECTED_PARENT_AUDIT_SHA256 = "f3feb7bc612dff982b3807eda9538c09949c8d280001a23e914a38f8fdae201e"
EXPECTED_PROBE_SHA256 = "e6e6c4ac650502c3932b59e271ac86eeeff49a000e39f7be81f7e46d77b98cde"
EXPECTED_TEST_SHA256 = "5976e64039127334bf28d9061181ce9d65efa2cc115d65ddf56db11354be8837"

FIT_SOURCES = tuple(
    ["train_{:03d}.npz".format(i) for i in range(28, 33)]
    + ["train_{:03d}.npz".format(i) for i in range(40, 44)]
    + ["train_{:03d}.npz".format(i) for i in range(59, 66)]
    + ["train_{:03d}.npz".format(i) for i in range(67, 75)]
)
HELD_SOURCES = tuple("train_{:03d}.npz".format(i) for i in range(15))
EPOCHS = 2
VIEWS_PER_SOURCE_PER_EPOCH = 2
EXPECTED_STEPS = 96
SEED = 79
GPU_FLAG = "--root-authorized-gpu"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def canonical_json(payload):
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_bytes_exclusive(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(values)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_json_sidecar_exclusive(path, payload):
    values = canonical_json(payload)
    write_bytes_exclusive(path, values)
    digest = hashlib.sha256(values).hexdigest()
    write_bytes_exclusive(
        Path(str(path) + ".sha256"),
        (digest + "  " + Path(path).name + "\n").encode("ascii"),
    )
    return digest


def verify_sidecar(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    actual = sha256_file(path)
    if len(tokens) != 2 or tokens[0] != actual or tokens[1] != path.name:
        raise RuntimeError("sidecar mismatch: {}".format(path))
    return actual


def workspace_path(relative):
    value = (WORKSPACE / relative).resolve()
    if value != WORKSPACE.resolve() and WORKSPACE.resolve() not in value.parents:
        raise RuntimeError("artifact escaped workspace")
    return value


def middle_route(event_count):
    return 30000 < int(event_count) <= 200000


def _load_private_core():
    if sha256_file(CORE_PATH) != EXPECTED_CORE_SHA256:
        raise RuntimeError("generic formal core changed")
    name = "_middle_first_fold_private_formal_core"
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = _load_private_core()


def _load_manifest_sources(science):
    if sha256_file(TRAIN_CACHE_MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("released M20 train-cache manifest changed")
    manifest = read_json(TRAIN_CACHE_MANIFEST_PATH)
    if manifest.get("schema") != "ev-uav-component-reranker-train-cache-v1":
        raise RuntimeError("unexpected train-cache manifest schema")
    records = {record["source_name"]: record for record in manifest["records"]}
    official = {
        record["source_name"]: record["source_sha256"]
        for record in manifest["official_train_sources"]
    }
    middle = []
    for values in science["continuous_source_families"].values():
        middle.extend(values)
    if len(middle) != 39 or len(set(middle)) != 39:
        raise RuntimeError("middle source population changed")
    result = {}
    for name in middle:
        record = records.get(name)
        if record is None or official.get(name) != record.get("source_sha256"):
            raise RuntimeError("manifest evidence missing for {}".format(name))
        if not middle_route(record["event_count"]):
            raise RuntimeError("manifest source left middle route: {}".format(name))
        result[name] = {
            "sha256": record["source_sha256"],
            "event_count": int(record["event_count"]),
            "record": record["record"],
            "record_sha256": record["record_sha256"],
        }
    return manifest, result


def load_frozen_contract():
    if sha256_file(SCIENCE_PATH) != EXPECTED_SCIENCE_SHA256:
        raise RuntimeError("science protocol changed")
    if sha256_file(EXECUTION_PATH) != EXPECTED_EXECUTION_SHA256:
        raise RuntimeError("execution protocol changed")
    science = read_json(SCIENCE_PATH)
    execution = read_json(EXECUTION_PATH)
    if execution.get("schema") != "ev-uav-middle-multiscale-temporal-summary-first-fold-execution-v1":
        raise RuntimeError("unexpected execution schema")
    if execution.get("status") != "frozen_before_first_fold_training_or_held_f1_array_access":
        raise RuntimeError("execution protocol is not frozen")
    if execution["parent_science"]["sha256"] != EXPECTED_SCIENCE_SHA256:
        raise RuntimeError("execution points to another science protocol")
    if tuple(execution["scope"]["fit_sources"]) != FIT_SOURCES:
        raise RuntimeError("fit sources changed")
    if tuple(execution["scope"]["held_sources"]) != HELD_SOURCES:
        raise RuntimeError("held sources changed")
    if set(FIT_SOURCES) & set(HELD_SOURCES):
        raise RuntimeError("fit/held leakage")
    if len(FIT_SOURCES) * EPOCHS * VIEWS_PER_SOURCE_PER_EPOCH != EXPECTED_STEPS:
        raise RuntimeError("optimizer-step arithmetic changed")
    if execution["training"]["optimizer_steps"] != EXPECTED_STEPS:
        raise RuntimeError("execution optimizer steps changed")
    if science["fold_order"][0]["fold_id"] != execution["scope"]["fold_id"]:
        raise RuntimeError("first fold changed")
    if science["fold_order"][0]["optimizer_steps"] != EXPECTED_STEPS:
        raise RuntimeError("science optimizer steps changed")
    if execution["scope"]["validation_read_allowed"] is not False or execution["scope"]["test_read_allowed"] is not False:
        raise RuntimeError("validation/test access is forbidden")
    if execution["scope"]["source_name_path_hash_or_fold_inference_feature_allowed"] is not False:
        raise RuntimeError("source identity inference feature is forbidden")
    if sha256_file(core.atomic.M20_PATH) != EXPECTED_M20_SHA256:
        raise RuntimeError("released M20 changed")
    if sha256_file(ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("expert implementation changed")
    if sha256_file(ROOT / "utils" / "h2_multiscale_pyramid_loss.py") != EXPECTED_LOSS_SHA256:
        raise RuntimeError("loss implementation changed")
    if sha256_file(ROOT / "tests" / "test_middle_multiscale_temporal_summary_first_fold.py") != EXPECTED_TEST_SHA256:
        raise RuntimeError("formal CPU tests changed")
    for key, expected in (
        ("command_audit", EXPECTED_PARENT_AUDIT_SHA256),
        ("eight_step_probe", EXPECTED_PROBE_SHA256),
    ):
        item = execution["cpu_and_probe_evidence"][key]
        path = workspace_path(item["path"])
        if sha256_file(path) != expected or item["sha256"] != expected:
            raise RuntimeError("prior evidence changed: {}".format(key))
    probe_payload = read_json(workspace_path(execution["cpu_and_probe_evidence"]["eight_step_probe"]["path"]))
    if probe_payload.get("passed") is not True or probe_payload.get("optimizer_steps") != 8:
        raise RuntimeError("eight-step probe no longer passes")
    if probe_payload.get("formal_optimizer_steps") != 0 or probe_payload.get("source") != "train_028.npz":
        raise RuntimeError("probe/formal boundary changed")
    _, sources = _load_manifest_sources(science)
    runtime_contract = dict(execution)
    runtime_contract["source_manifest"] = {"sources": sources}
    runtime_contract["numeric_recovery"] = {
        "frozen_M20_observation_precision": "FP32",
        "expert_forward_precision": "FP32",
        "event_and_constraint_loss_precision": "FP32",
        "backward_precision": "FP32",
        "optimizer": "AdamW_FP32",
        "grad_scaler_used": False,
        "optimizer_steps": EXPECTED_STEPS,
        "epochs": EPOCHS,
        "views_per_fit_source_per_epoch": VIEWS_PER_SOURCE_PER_EPOCH,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
    }
    runtime_contract["evaluation_contract"] = {
        "effective_C00_sha256": EXPECTED_C00_SHA256
    }
    return science, runtime_contract


def build_and_validate_c00(_contract):
    cfg, effective = core.probe.build_c00()
    if crossfit.sha256_json(effective) != EXPECTED_C00_SHA256:
        raise RuntimeError("effective C00 changed")
    return cfg, effective


def load_middle_input_and_truth(path, expected_event_count):
    video, polarities, locations4 = core.atomic._load_input_only(path)
    labels, target_ids = core.atomic._load_truth(path)
    if len(polarities) != int(expected_event_count):
        raise RuntimeError("source event count changed")
    if len(video.event_indices_by_bin) != 160:
        raise RuntimeError("middle expert requires complete T160")
    if not middle_route(len(polarities)):
        raise RuntimeError("source left frozen middle route")
    if labels.size != len(polarities) or target_ids.size != len(polarities):
        raise RuntimeError("source truth vectors do not align")
    return video, polarities, locations4, labels, target_ids


def middle_safety_gates(base, candidate, *, require_effect_size):
    co_retention = 1.0 if base["CO"] == 0 else candidate["CO"] / base["CO"]
    gates = {
        "Score_strictly_positive": candidate["Score"] > base["Score"],
        "IoU_not_lower": candidate["IoU"] >= base["IoU"],
        "Fa_not_higher": candidate["Fa"] <= base["Fa"],
        "Pd_delta_at_least_minus_0_002": candidate["Pd"] - base["Pd"] >= -0.002,
        "correct_object_retention_at_least_0_995": co_retention >= 0.995,
    }
    if require_effect_size:
        gates["Score_gain_at_least_0_01"] = candidate["Score"] - base["Score"] >= 0.01
        gates["FP_or_FC_strictly_lower"] = candidate["FP"] < base["FP"] or candidate["FC"] < base["FC"]
    return gates


def configure_core():
    core.__file__ = str(Path(__file__).resolve())
    core.ROOT = ROOT
    core.WORKSPACE = WORKSPACE
    core.TRAIN_DATA_ROOT = TRAIN_DATA_ROOT
    core.SCIENCE_PATH = SCIENCE_PATH
    core.EXECUTION_PATH = EXECUTION_PATH
    core.SOURCE_MANIFEST_PROTOCOL_PATH = TRAIN_CACHE_MANIFEST_PATH
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.TRAIN_OUTPUT_ROOT = TRAIN_OUTPUT_ROOT
    core.CHECKPOINT_PATH = CHECKPOINT_PATH
    core.CHECKPOINT_SIDECAR_PATH = Path(str(CHECKPOINT_PATH) + ".sha256")
    core.TRAINING_RECEIPT_PATH = TRAINING_RECEIPT_PATH
    core.TRAINING_RECEIPT_SIDECAR_PATH = Path(str(TRAINING_RECEIPT_PATH) + ".sha256")
    core.EVALUATION_ROOT = EVALUATION_ROOT
    core.EVALUATION_PATH = EVALUATION_PATH
    core.DECISION_PATH = DECISION_PATH
    core.CPU_AUDIT_PATH = CPU_AUDIT_PATH
    core.EXPECTED_SCIENCE_SHA256 = EXPECTED_SCIENCE_SHA256
    core.EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256 = EXPECTED_MANIFEST_SHA256
    core.EXPECTED_MODEL_SHA256 = EXPECTED_MODEL_SHA256
    core.EXPECTED_LOSS_SHA256 = EXPECTED_LOSS_SHA256
    core.EXPECTED_M20_SHA256 = EXPECTED_M20_SHA256
    core.EXPECTED_C00_SHA256 = EXPECTED_C00_SHA256
    core.FIT_SOURCES = FIT_SOURCES
    core.HELD_SOURCES = HELD_SOURCES
    core.SEED = SEED
    core.EPOCHS = EPOCHS
    core.VIEWS_PER_SOURCE_PER_EPOCH = VIEWS_PER_SOURCE_PER_EPOCH
    core.EXPECTED_STEPS = EXPECTED_STEPS
    core.GPU_FLAG = GPU_FLAG
    core.load_frozen_contract = load_frozen_contract
    core.build_and_validate_c00 = build_and_validate_c00
    core.load_input_and_truth = load_middle_input_and_truth
    core.use_h2_residual_refiner = lambda count, _polarities: middle_route(count)
    core.safety_gates = middle_safety_gates


configure_core()
_CORE_VERIFY_TRAINING_GATE = core.verify_training_gate_before_held


def _require_cpu_audit():
    digest = verify_sidecar(CPU_AUDIT_PATH)
    payload = read_json(CPU_AUDIT_PATH)
    if payload.get("passed") is not True:
        raise RuntimeError("CPU preflight does not pass")
    if payload.get("formal_runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("formal runner changed after CPU preflight")
    if payload.get("execution_protocol_sha256") != EXPECTED_EXECUTION_SHA256:
        raise RuntimeError("CPU preflight execution binding changed")
    if payload.get("held_arrays_read") is not False or payload.get("cuda_initialized_or_used") is not False:
        raise RuntimeError("CPU preflight scope changed")
    return digest


def _require_middle_training_gate(contract):
    checkpoint, receipt, checkpoint_sha, receipt_sha = _CORE_VERIFY_TRAINING_GATE(contract)
    gate_sha = verify_sidecar(MIDDLE_TRAINING_GATE_PATH)
    gate = read_json(MIDDLE_TRAINING_GATE_PATH)
    expected = {
        "passed": True,
        "checkpoint_sha256": checkpoint_sha,
        "training_receipt_sha256": receipt_sha,
        "held_f1_arrays_read": False,
        "optimizer_steps": EXPECTED_STEPS,
        "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
        "execution_protocol_sha256": EXPECTED_EXECUTION_SHA256,
        "formal_runner_sha256": sha256_file(Path(__file__)),
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError("middle training gate changed: {}".format(key))
    return checkpoint, receipt, checkpoint_sha, receipt_sha, gate_sha


def _fit_cache_preflight(sources, cfg):
    from types import SimpleNamespace

    diagnostics = []
    for position, name in enumerate(FIT_SOURCES):
        expected = sources[name]
        record_path = (TRAIN_CACHE_ROOT / expected["record"]).resolve()
        if sha256_file(record_path) != expected["record_sha256"]:
            raise RuntimeError("fit cache record changed: {}".format(name))
        with np.load(record_path, allow_pickle=False) as record:
            scores = np.asarray(record["scores"], dtype=np.float32)
            locs = np.asarray(record["locs"], dtype=np.int64)
            labels = np.asarray(record["labels"], dtype=np.uint8)
            targets = np.asarray(record["target_ids"], dtype=np.int16)
        locations4 = np.column_stack((np.zeros(len(locs), dtype=np.int64), locs))
        bins = [np.flatnonzero(np.floor_divide(locs[:, 2], 50) == value) for value in range(160)]
        video = SimpleNamespace(locations=locs, event_indices_by_bin=bins)
        hard, _, count = core.extract_fit_hard_negatives(cfg, scores, locations4, labels)
        views0, eligible0 = core.prepare_training_views(video, labels, targets, hard, epoch=0, source_position=position)
        views1, eligible1 = core.prepare_training_views(video, labels, targets, hard, epoch=1, source_position=position)
        if len(views0) != 2 or len(views1) != 2:
            raise RuntimeError("fit source does not yield two views per epoch")
        diagnostics.append({
            "source": name,
            "event_count": len(scores),
            "base_component_count": count,
            "pure_FP_component_count": len(hard),
            "eligible_joint_views_epoch0": eligible0,
            "eligible_joint_views_epoch1": eligible1,
            "starts_epoch0": [item["start"] for item in views0],
            "starts_epoch1": [item["start"] for item in views1],
        })
    return diagnostics


def cpu_audit(_args):
    if CPU_AUDIT_PATH.exists() or TRAIN_OUTPUT_ROOT.exists() or EVALUATION_ROOT.exists():
        raise FileExistsError("first-fold formal output already exists")
    science, contract = load_frozen_contract()
    cfg, effective = build_and_validate_c00(contract)
    sources = contract["source_manifest"]["sources"]
    fit_diagnostics = _fit_cache_preflight(sources, cfg)
    held_paths = [(TRAIN_DATA_ROOT / name).resolve() for name in HELD_SOURCES]
    if not all(path.parent == TRAIN_DATA_ROOT and path.is_file() for path in held_paths):
        raise RuntimeError("held F1 path membership/existence changed")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight initialized CUDA")
    payload = {
        "schema": "ev-uav-middle-multiscale-temporal-summary-first-fold-cpu-preflight-v1",
        "created_utc": utc_now(),
        "passed": True,
        "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
        "execution_protocol_sha256": EXPECTED_EXECUTION_SHA256,
        "formal_runner_sha256": sha256_file(Path(__file__)),
        "generic_formal_core_sha256": EXPECTED_CORE_SHA256,
        "command_audit_sha256": EXPECTED_PARENT_AUDIT_SHA256,
        "probe_receipt_sha256": EXPECTED_PROBE_SHA256,
        "train_cache_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "released_m20_sha256": EXPECTED_M20_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "loss_sha256": EXPECTED_LOSS_SHA256,
        "tests_sha256": EXPECTED_TEST_SHA256,
        "effective_C00": effective,
        "effective_C00_sha256": EXPECTED_C00_SHA256,
        "fold_id": "middle_hold_f1_000_014",
        "fit_sources": list(FIT_SOURCES),
        "held_sources_reserved_unread": list(HELD_SOURCES),
        "fit_held_overlap": 0,
        "fit_cache_preflight": fit_diagnostics,
        "optimizer_steps": EXPECTED_STEPS,
        "epochs": EPOCHS,
        "views_per_fit_source_per_epoch": VIEWS_PER_SOURCE_PER_EPOCH,
        "held_arrays_read": False,
        "validation_or_test_read": False,
        "cuda_initialized_or_used": False,
    }
    digest = write_json_sidecar_exclusive(CPU_AUDIT_PATH, payload)
    print(json.dumps({"cpu_preflight": str(CPU_AUDIT_PATH), "sha256": digest, "passed": True, "held_arrays_read": False}, indent=2))


def train_first_fold(args):
    audit_sha = _require_cpu_audit()
    core.train_formal(args)
    checkpoint_sha = verify_sidecar(CHECKPOINT_PATH)
    receipt_sha = verify_sidecar(TRAINING_RECEIPT_PATH)
    receipt = read_json(TRAINING_RECEIPT_PATH)
    gate = {
        "schema": "ev-uav-middle-multiscale-temporal-summary-first-fold-training-gate-v1",
        "created_utc": utc_now(),
        "passed": True,
        "fold_id": "middle_hold_f1_000_014",
        "fit_sources": list(FIT_SOURCES),
        "held_sources_reserved_unread": list(HELD_SOURCES),
        "held_f1_arrays_read": False,
        "validation_or_test_read": False,
        "optimizer_steps": receipt["optimizer_steps"],
        "checkpoint_selection": "final_epoch_only",
        "checkpoint_sha256": checkpoint_sha,
        "training_receipt_sha256": receipt_sha,
        "cpu_preflight_sha256": audit_sha,
        "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
        "execution_protocol_sha256": EXPECTED_EXECUTION_SHA256,
        "formal_runner_sha256": sha256_file(Path(__file__)),
        "released_m20_state_unchanged": receipt["released_m20_state_sha256_before"] == receipt["released_m20_state_sha256_after"],
        "initial_bitwise_m20_identity": receipt["initial_actual_M20_bitwise_identity"],
    }
    if gate["optimizer_steps"] != EXPECTED_STEPS or not gate["released_m20_state_unchanged"] or not gate["initial_bitwise_m20_identity"]:
        raise RuntimeError("post-training immutable gate failed")
    gate_sha = write_json_sidecar_exclusive(MIDDLE_TRAINING_GATE_PATH, gate)
    print(json.dumps({"middle_training_gate": str(MIDDLE_TRAINING_GATE_PATH), "sha256": gate_sha, "passed": True}, indent=2))


def verify_training_gate_before_held(contract):
    checkpoint, receipt, checkpoint_sha, receipt_sha, _ = _require_middle_training_gate(contract)
    return checkpoint, receipt, checkpoint_sha, receipt_sha


core.verify_training_gate_before_held = verify_training_gate_before_held


def _cache_record_for_source(source, sources):
    expected = sources[source]
    path = (TRAIN_CACHE_ROOT / expected["record"]).resolve()
    before = sha256_file(path)
    if before != expected["record_sha256"]:
        raise RuntimeError("held anchor cache changed: {}".format(source))
    with np.load(path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float32).copy()
    if sha256_file(path) != before:
        raise RuntimeError("held anchor cache changed during read")
    return scores, before


def _build_middle_report(core_payload):
    _, contract = load_frozen_contract()
    sources = contract["source_manifest"]["sources"]
    anchor_checks = []
    for record in core_payload["records"]:
        source = record["source_name"]
        artifact_path = Path(record["score_artifact_path"])
        with np.load(artifact_path, allow_pickle=False) as artifact:
            computed_base = np.asarray(artifact["base_raw_scores"], dtype=np.float32)
        anchor, anchor_file_sha = _cache_record_for_source(source, sources)
        if not np.array_equal(computed_base, anchor):
            raise RuntimeError("candidate-internal M20 base differs from frozen cache anchor: {}".format(source))
        anchor_checks.append({
            "source": source,
            "cache_record_sha256": anchor_file_sha,
            "event_count": len(anchor),
            "base_raw_scores_bitwise_equal_frozen_cache_anchor": True,
            "candidate_inference_calls": 1,
        })
    base = core_payload["pooled_base_report"]
    candidate = core_payload["pooled_candidate_report"]
    delta = core_payload["pooled_delta_report"]
    gates = middle_safety_gates(base, candidate, require_effect_size=True)
    invariants = {
        "event_count_equal": core_payload["pooled_base_counts"]["event_count"] == core_payload["pooled_candidate_counts"]["event_count"],
        "frame_count_equal": core_payload["pooled_base_counts"]["frame_count"] == core_payload["pooled_candidate_counts"]["frame_count"],
        "object_count_equal": core_payload["pooled_base_counts"]["object_count"] == core_payload["pooled_candidate_counts"]["object_count"],
        "positive_event_population_equal": core_payload["pooled_base_counts"]["true_positive_events"] + core_payload["pooled_base_counts"]["false_negative_events"] == core_payload["pooled_candidate_counts"]["true_positive_events"] + core_payload["pooled_candidate_counts"]["false_negative_events"],
    }
    gates["population_invariants_equal"] = all(invariants.values())
    score_delta = float(delta["Score"])
    co_retention = 1.0 if base["CO"] == 0 else candidate["CO"] / base["CO"]
    major = (
        candidate["IoU"] < base["IoU"]
        or candidate["Fa"] > base["Fa"]
        or candidate["Pd"] - base["Pd"] < -0.002
        or co_retention < 0.995
        or not all(invariants.values())
    )
    promoted = all(gates.values())
    hard_archive = score_delta < 0.005 or major
    decision = "eligible_for_next_middle_fold_but_not_started" if promoted else "archive_without_other_folds_or_tuning"
    return {
        "schema": "ev-uav-middle-multiscale-temporal-summary-held-f1-report-v1",
        "created_utc": utc_now(),
        "fold_id": "middle_hold_f1_000_014",
        "family": "f1_000_014",
        "fit_sources": list(FIT_SOURCES),
        "held_sources": list(HELD_SOURCES),
        "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
        "execution_protocol_sha256": EXPECTED_EXECUTION_SHA256,
        "formal_runner_sha256": sha256_file(Path(__file__)),
        "core_evaluation_path": str(EVALUATION_PATH.resolve()),
        "core_evaluation_sha256": sha256_file(EVALUATION_PATH),
        "checkpoint_sha256": core_payload["checkpoint_sha256"],
        "training_receipt_sha256": core_payload["training_receipt_sha256"],
        "middle_training_gate_sha256": verify_sidecar(MIDDLE_TRAINING_GATE_PATH),
        "anchor_checks": anchor_checks,
        "records": core_payload["records"],
        "family_pooled_base_counts": core_payload["pooled_base_counts"],
        "family_pooled_candidate_counts": core_payload["pooled_candidate_counts"],
        "family_pooled_base_metrics": core_payload["pooled_base_metrics"],
        "family_pooled_candidate_metrics": core_payload["pooled_candidate_metrics"],
        "family_pooled_base_report": base,
        "family_pooled_candidate_report": candidate,
        "family_pooled_delta_report": delta,
        "population_invariants": invariants,
        "promotion_gates": gates,
        "promoted": promoted,
        "early_archive_triggered": hard_archive,
        "decision": decision,
        "true_positive_and_correct_object_losses_are_reported_not_hidden": True,
        "candidate_inference_calls": len(HELD_SOURCES),
        "other_fold_started": False,
        "validation_or_test_read": False,
        "default_checkpoint_or_submission_changed": False,
    }


def evaluate_first_fold(args):
    _require_cpu_audit()
    core.evaluate_held_g3(args)
    core_payload = read_json(EVALUATION_PATH)
    report = _build_middle_report(core_payload)
    digest = write_json_sidecar_exclusive(MIDDLE_REPORT_PATH, report)
    print(json.dumps({
        "middle_report": str(MIDDLE_REPORT_PATH),
        "sha256": digest,
        "base": report["family_pooled_base_report"],
        "candidate": report["family_pooled_candidate_report"],
        "delta": report["family_pooled_delta_report"],
        "gates": report["promotion_gates"],
        "promoted": report["promoted"],
        "decision": report["decision"],
    }, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.set_defaults(func=cpu_audit)
    train = sub.add_parser("train-first-fold")
    train.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
    train.set_defaults(func=train_first_fold)
    evaluate = sub.add_parser("evaluate-first-fold")
    evaluate.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
    evaluate.set_defaults(func=evaluate_first_fold)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as error:
        if not FAILURE_PATH.exists():
            try:
                write_json_sidecar_exclusive(FAILURE_PATH, {
                    "schema": "ev-uav-middle-multiscale-temporal-summary-first-fold-failure-v1",
                    "created_utc": utc_now(),
                    "command": args.command,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
                    "execution_protocol_sha256": EXPECTED_EXECUTION_SHA256,
                    "formal_runner_sha256": sha256_file(Path(__file__)),
                    "training_output_exists": TRAIN_OUTPUT_ROOT.exists(),
                    "evaluation_output_exists": EVALUATION_ROOT.exists(),
                    "validation_or_test_read": False,
                })
            except Exception:
                pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

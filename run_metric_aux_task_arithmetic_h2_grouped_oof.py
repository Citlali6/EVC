"""Adaptive train-only alpha=1 metric-aux task-arithmetic experiment.

For each frozen source-group fold this runner synthesizes exactly one
inference-only checkpoint::

    W_iso = W_released_m20 + (W_metric_aux_e3 - W_baseline_e3)

No training, alpha grid, threshold search, endpoint re-inference, validation
or test access is exposed.  Three new held-train candidate inferences are
compared with the immutable V4 released-M20 artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
RUNNER_PATH = Path(__file__).resolve()

_PRIVATE_NAMES = (
    "_metric_aux_h2_grouped_oof_v1_core_for_v2",
    "_metric_aux_h2_grouped_oof_v2_for_v3",
    "_metric_aux_h2_grouped_oof_v3_for_v4",
    "_metric_aux_h2_grouped_oof_v4_for_task_arithmetic",
)
_PREVIOUS_PRIVATE = {name: sys.modules.get(name) for name in _PRIVATE_NAMES}
_V4_PATH = EVC_ROOT / "run_metric_aux_h2_grouped_oof_v4.py"
_V4_SPEC = importlib.util.spec_from_file_location(_PRIVATE_NAMES[-1], _V4_PATH)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError("Unable to create a private V4 module for task arithmetic.")
v4 = importlib.util.module_from_spec(_V4_SPEC)
sys.modules[_PRIVATE_NAMES[-1]] = v4
try:
    _V4_SPEC.loader.exec_module(v4)
finally:
    for _name, _previous in _PREVIOUS_PRIVATE.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

core = v4.core

PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "metric_aux_task_arithmetic_h2_grouped_oof_science_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = "4586e1e20a501a4b5219c2d5953a6666c1fc0d4ac08da2a7b29d6cea70938266"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-task-arithmetic-h2-grouped-oof-v1"
OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260810_metric_aux_task_arithmetic_h2_grouped_oof_v1"
)
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
SYNTHESIS_ROOT = OUTPUT_ROOT / "synthesis"
SYNTHESIS_MANIFEST_PATH = SYNTHESIS_ROOT / "synthesis_manifest.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"

V4_PROTOCOL_SHA256 = "589e7d35075e31ad5d85b946ce444adeb88b9ca35cb7c8772b385bc61cfc96b5"
V4_RUNNER_SHA256 = "463fc34024e668786ff502e799badba6919385b8e365f73dc1b15710f18fa32b"
V4_TEST_SHA256 = "33f30fd67d6143a6650a7707ad8c3d6d31e1a2c6dd1adebdc743c2dce98a9f0f"
V4_COMMAND_AUDIT_SHA256 = "cb53edcdaf39b0fbc8ee16f23710279d941a6a63c834824d36c260222839de12"
V4_REPORT_SHA256 = "523cffd42854adc48efa4890f5e8beb8d3f8042d450f07829943471b06fbcf91"
M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EFFECTIVE_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"

_V4_LOAD_PROTOCOL = v4.load_protocol
_BASE_EVALUATE_SPEC = v4._BASE_EVALUATE_SPEC
_BASE_LOAD_EVALUATION_RESULT = v4._BASE_LOAD_EVALUATION_RESULT
_BASE_VERIFY_ASSETS = v4._BASE_VERIFY_ASSETS
_BASE_WRITE_NEW_JSON = v4._BASE_WRITE_NEW_JSON


def _expect(actual, expected, label):
    core._expect_equal(actual, expected, label)


def _require_bound_file(record, expected_sha256, label):
    path = core.workspace_path(record["workspace_relative_path"])
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    actual = core.sha256_file(path)
    if actual != expected_sha256 or actual != record["sha256"]:
        raise RuntimeError("{} SHA-256 differs from the frozen contract.".format(label))
    return path


def _load_bound_json(record, expected_sha256, label):
    path = _require_bound_file(record, expected_sha256, label)
    payload, digest = core.load_json_snapshot(path)
    _expect(digest, expected_sha256, "{} snapshot".format(label))
    return payload, path


def _validate_v4_evaluation(eval_id, payload, protocol):
    fold_id, variant = eval_id[:7], eval_id[8:]
    expected_names = [
        item["name"]
        for item in core.held_items(
            protocol,
            next(fold for fold in protocol["dataset"]["folds"] if fold["fold_id"] == fold_id),
        )
    ]
    expected = {
        "schema": "ev-uav-metric-aux-held-train-evaluation-v1",
        "eval_id": eval_id,
        "fold_id": fold_id,
        "variant": variant,
        "dataset_split": "train",
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "runner_sha256": V4_RUNNER_SHA256,
        "t32_read_or_combined": False,
    }
    for key, value in expected.items():
        _expect(payload.get(key), value, "V4 {} {}".format(eval_id, key))
    _expect(
        [record["source_name"] for record in payload["records"]],
        expected_names,
        "V4 {} source order".format(eval_id),
    )
    _expect(
        payload.get("evaluation_recovery_v4", {}).get("mode"),
        "evaluation_only_import_route_recovery_v4",
        "V4 recovery provenance",
    )
    if variant == "released_m20":
        _expect(payload.get("checkpoint_sha256"), M20_SHA256, "V4 M20 checkpoint")


def _validate_overlay(overlay, digest, effective_v4):
    _expect(overlay.get("schema"), EXPECTED_SCHEMA, "task-arithmetic protocol schema")
    _expect(
        overlay.get("status"),
        "frozen_after_v4_train_only_results_before_any_task_arithmetic_checkpoint_synthesis_or_inference",
        "task-arithmetic protocol status",
    )
    _expect(digest, EXPECTED_PROTOCOL_SHA256, "task-arithmetic protocol SHA-256")
    inheritance = overlay["v4_inheritance"]
    _require_bound_file(inheritance["v4_protocol"], V4_PROTOCOL_SHA256, "V4 protocol")
    _require_bound_file(inheritance["v4_runner"], V4_RUNNER_SHA256, "V4 runner")
    _require_bound_file(inheritance["v4_tests"], V4_TEST_SHA256, "V4 tests")
    _require_bound_file(
        inheritance["v4_command_audit"], V4_COMMAND_AUDIT_SHA256, "V4 command audit"
    )
    report, _ = _load_bound_json(inheritance["v4_report"], V4_REPORT_SHA256, "V4 report")
    _expect(report.get("status"), "failed", "V4 report status")
    _expect(report.get("passed"), False, "V4 report passed")
    _expect(report.get("protocol_sha256"), V4_PROTOCOL_SHA256, "V4 report protocol")
    _expect(report.get("runner_sha256"), V4_RUNNER_SHA256, "V4 report runner")
    _expect(
        report["pooled"]["metric_aux"]["delta_vs_baseline"]["score"],
        overlay["observed_v4_mechanism_evidence"]["metric_aux_vs_paired_baseline_pooled"]["score_delta"],
        "V4 observed score mechanism",
    )

    artifacts = overlay["v4_evaluation_artifacts"]
    expected_ids = {
        "{}_{}".format(fold, variant)
        for fold in ("hold_g1", "hold_g2", "hold_g3")
        for variant in ("released_m20", "baseline", "metric_aux")
    }
    if set(artifacts) != expected_ids:
        raise RuntimeError("V4 evaluation evidence is not exactly nine artifacts.")
    evaluations = {}
    for eval_id in sorted(artifacts):
        payload, _ = _load_bound_json(
            artifacts[eval_id], artifacts[eval_id]["sha256"], "V4 {}".format(eval_id)
        )
        _validate_v4_evaluation(eval_id, payload, effective_v4)
        evaluations[eval_id] = payload

    disclosure = overlay["adaptive_selection_disclosure"]
    for key in (
        "v4_held_train_results_observed_before_candidate_definition",
        "candidate_selected_because_metric_aux_improved_each_fold_vs_paired_baseline_but_shared_full_finetune_drift_harmed_g2_g3_vs_m20",
        "same_three_train_source_groups_reused",
        "new_results_must_not_be_called_independent_held_or_unbiased_oof",
        "shared_parent_pretraining_exposure",
    ):
        _expect(disclosure.get(key), True, "adaptive disclosure {}".format(key))
    candidate = overlay["task_arithmetic_candidate"]
    _expect(candidate.get("candidate_count"), 1, "candidate count")
    _expect(candidate.get("alpha"), 1.0, "alpha")
    for key in ("parameter_grid_allowed", "threshold_search_allowed", "weight_search_allowed"):
        _expect(candidate.get(key), False, "candidate {}".format(key))
    _expect(candidate.get("new_training_optimizer_steps"), 0, "new training steps")
    _expect(
        overlay["evaluation_contract"]["prediction_threshold"],
        effective_v4["evaluation"]["prediction_threshold"],
        "inherited threshold",
    )
    _expect(
        overlay["evaluation_contract"]["effective_c00_canonical_sha256"],
        EFFECTIVE_C00_SHA256,
        "effective C00",
    )
    _expect(
        effective_v4["evaluation"]["effective_c00_canonical_sha256"],
        EFFECTIVE_C00_SHA256,
        "V4 effective C00",
    )
    policy = overlay["promotion_policy_revision"]
    _expect(policy.get("adaptive_policy_revision_disclosed"), True, "adaptive gate revision")
    _expect(
        policy["per_fold_against_released_m20"]["raw_true_positive_events_not_lower_required"],
        False,
        "fold raw TP policy",
    )
    _expect(
        policy["pooled_against_released_m20"]["raw_true_positive_events_not_lower_required"],
        False,
        "pooled raw TP policy",
    )
    _expect(overlay.get("validation_or_test_read_allowed"), False, "split policy")
    _expect(overlay.get("t32_allowed"), False, "T32 policy")
    return {"report": report, "evaluations": evaluations}


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Task-arithmetic protocol SHA-256 {} differs from frozen {}.".format(
                actual, EXPECTED_PROTOCOL_SHA256
            )
        )
    overlay, snapshot_sha = core.load_json_snapshot(PROTOCOL_PATH)
    _expect(snapshot_sha, actual, "task-arithmetic protocol snapshot")
    effective_v4, inherited_sha = _V4_LOAD_PROTOCOL()
    _expect(inherited_sha, V4_PROTOCOL_SHA256, "inherited V4 protocol")
    evidence = _validate_overlay(overlay, actual, effective_v4)
    effective = copy.deepcopy(effective_v4)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["adaptive_selection_disclosure_task_arithmetic"] = copy.deepcopy(
        overlay["adaptive_selection_disclosure"]
    )
    effective["task_arithmetic_candidate"] = copy.deepcopy(
        overlay["task_arithmetic_candidate"]
    )
    effective["input_checkpoint_contract_task_arithmetic"] = copy.deepcopy(
        overlay["input_checkpoint_contract"]
    )
    effective["v4_evaluation_artifacts_task_arithmetic"] = copy.deepcopy(
        overlay["v4_evaluation_artifacts"]
    )
    effective["promotion_policy_revision_task_arithmetic"] = copy.deepcopy(
        overlay["promotion_policy_revision"]
    )
    effective["task_arithmetic_output_contract"] = copy.deepcopy(
        overlay["synthesis_output_contract"]
    )
    effective["audit_amendment"]["claim_scope"] = overlay[
        "adaptive_selection_disclosure"
    ]["claim_scope"]
    effective["revision_history"] = list(effective["revision_history"]) + [
        {
            "adaptive_protocol_sha256": actual,
            "v4_report_sha256": V4_REPORT_SHA256,
            "reason": overlay["observed_v4_mechanism_evidence"]["mechanism_hypothesis"],
            "alpha": 1.0,
            "new_training_optimizer_steps": 0,
            "raw_tp_gate_policy_revised_before_new_results": True,
        }
    ]
    effective["outputs"]["workspace_relative_directory"] = overlay[
        "synthesis_output_contract"
    ]["workspace_relative_directory"]
    effective["_task_arithmetic_bound_v4_evidence"] = evidence
    return effective, actual


def _checkpoint_path(record):
    return core.workspace_path(record["workspace_relative_path"])


def synthesis_specs(protocol):
    contract = protocol["input_checkpoint_contract_task_arithmetic"]
    parent = contract["released_m20"]
    output = []
    for fold in protocol["dataset"]["folds"]:
        fold_id = fold["fold_id"]
        pair = contract["fold_pairs"][fold_id]
        output.append(
            {
                "fold_id": fold_id,
                "held_group": fold["held_group"],
                "parent": str(_checkpoint_path(parent).resolve()),
                "parent_sha256": parent["sha256"],
                "baseline": str(_checkpoint_path(pair["baseline_e3"]).resolve()),
                "baseline_sha256": pair["baseline_e3"]["sha256"],
                "metric_aux": str(_checkpoint_path(pair["metric_aux_e3"]).resolve()),
                "metric_aux_sha256": pair["metric_aux_e3"]["sha256"],
                "output": str(
                    (SYNTHESIS_ROOT / fold_id / "isolated_metric_aux_alpha1.pt").resolve()
                ),
            }
        )
    return output


def _load_checkpoint(path):
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _name_shape_canonical_sha256(state):
    items = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(state.items())
    ]
    payload = json.dumps(
        items, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), items


def model_state_canonical_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
        digest.update(
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        digest.update(b"\n")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def synthesize_state_dict(parent, baseline, metric_aux, alpha=1.0):
    import torch

    if list(parent) != list(baseline) or list(parent) != list(metric_aux):
        raise RuntimeError("Task-arithmetic state-dict key order differs.")
    output = {}
    for name in parent:
        w0, wb, wa = parent[name], baseline[name], metric_aux[name]
        if (
            w0.shape != wb.shape
            or w0.shape != wa.shape
            or w0.dtype != wb.dtype
            or w0.dtype != wa.dtype
            or w0.dtype != torch.float32
        ):
            raise RuntimeError("Task-arithmetic tensor metadata differs: {}".format(name))
        value = (
            w0.detach().cpu().to(torch.float64)
            + float(alpha)
            * (
                wa.detach().cpu().to(torch.float64)
                - wb.detach().cpu().to(torch.float64)
            )
        ).to(dtype=w0.dtype)
        if not torch.isfinite(value).all():
            raise RuntimeError("Task-arithmetic output is non-finite: {}".format(name))
        output[name] = value.contiguous()
    return output


def _state_equal(left, right):
    import torch

    return list(left) == list(right) and all(
        left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and torch.equal(left[name], right[name])
        for name in left
    )


def _strict_model_state_load_cpu(state):
    import torch
    from model.temporal_memory_net import BidirectionalTemporalMemoryNet

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before strict CPU model-state load.")
    model = BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=16,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=False,
        confidence_head_enabled=False,
        temporal_attention_enabled=True,
    )
    model.load_state_dict(state, strict=True)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    tensor_count = sum(1 for _ in model.parameters())
    _expect(tensor_count, 89, "strict-load tensor count")
    _expect(parameter_count, 1924716, "strict-load parameter count")
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "passed": True,
    }


def _task_vector_stats(parent, baseline, metric_aux, isolated):
    import torch

    sums = {"parent": 0.0, "baseline_drift": 0.0, "task": 0.0}
    max_abs = 0.0
    changed_elements = 0
    changed_tensors = 0
    for name in parent:
        w0 = parent[name].to(torch.float64)
        wb = baseline[name].to(torch.float64)
        wa = metric_aux[name].to(torch.float64)
        wi = isolated[name].to(torch.float64)
        drift = wb - w0
        task = wa - wb
        sums["parent"] += float(torch.sum(w0 * w0))
        sums["baseline_drift"] += float(torch.sum(drift * drift))
        sums["task"] += float(torch.sum(task * task))
        maximum = float(torch.max(torch.abs(task))) if task.numel() else 0.0
        max_abs = max(max_abs, maximum)
        count = int(torch.count_nonzero(task))
        changed_elements += count
        changed_tensors += int(count > 0)
        expected = synthesize_state_dict(
            {name: parent[name]}, {name: baseline[name]}, {name: metric_aux[name]}, 1.0
        )[name]
        if not torch.equal(wi, expected):
            raise RuntimeError("Alpha=1 recomputation differs: {}".format(name))
    parent_norm = math.sqrt(sums["parent"])
    drift_norm = math.sqrt(sums["baseline_drift"])
    task_norm = math.sqrt(sums["task"])
    return {
        "parent_l2": parent_norm,
        "baseline_drift_l2": drift_norm,
        "task_vector_l2": task_norm,
        "task_over_baseline_drift": task_norm / drift_norm,
        "task_over_parent": task_norm / parent_norm,
        "task_max_abs": max_abs,
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
    }


def task_arithmetic_preflight(protocol):
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before task-arithmetic CPU preflight.")
    expected = protocol["input_checkpoint_contract_task_arithmetic"][
        "model_state_dict_contract"
    ]
    records = []
    for spec in synthesis_specs(protocol):
        for key in ("parent", "baseline", "metric_aux"):
            path = Path(spec[key])
            _expect(core.sha256_file(path), spec["{}_sha256".format(key)], "{} {} SHA".format(spec["fold_id"], key))
        parent_payload = _load_checkpoint(spec["parent"])
        baseline_payload = _load_checkpoint(spec["baseline"])
        aux_payload = _load_checkpoint(spec["metric_aux"])
        states = [
            payload["model_state_dict"]
            for payload in (parent_payload, baseline_payload, aux_payload)
        ]
        for state in states:
            canonical, items = _name_shape_canonical_sha256(state)
            _expect(canonical, expected["name_shape_canonical_sha256"], "name/shape canonical")
            _expect(len(items), expected["ordered_key_count"], "state key count")
            _expect(sum(item["numel"] for item in items), expected["element_count"], "state elements")
            if any(item["dtype"] != expected["dtype_each"] for item in items):
                raise RuntimeError("Task-arithmetic input contains a non-float32 tensor.")
        isolated = synthesize_state_dict(*states, alpha=1.0)
        alpha_zero = synthesize_state_dict(*states, alpha=0.0)
        if not _state_equal(alpha_zero, states[0]):
            raise RuntimeError("Alpha=0 is not bitwise released-M20 identity.")
        stats = _task_vector_stats(*states, isolated)
        strict_load = _strict_model_state_load_cpu(isolated)
        if not math.isfinite(stats["task_over_baseline_drift"]) or not (
            stats["task_over_baseline_drift"] > 0.0
        ):
            raise RuntimeError("Task vector ratio is not finite and strictly positive.")
        records.append(
            {
                "fold_id": spec["fold_id"],
                "input_model_state_sha256": {
                    "parent": model_state_canonical_sha256(states[0]),
                    "baseline": model_state_canonical_sha256(states[1]),
                    "metric_aux": model_state_canonical_sha256(states[2]),
                },
                "isolated_model_state_sha256": model_state_canonical_sha256(isolated),
                "alpha_zero_parent_identity": True,
                "alpha_one_formula_bitwise": True,
                "strict_model_load_cpu": strict_load,
                "stats": stats,
            }
        )
    return {"records": records, "cuda_not_initialized": not torch.cuda.is_initialized(), "passed": True}


def command_audit_payload(protocol, protocol_sha, assets, preflight):
    specs = synthesis_specs(protocol)
    synthesis_commands = {
        "all_three_folds": [sys.executable, str(RUNNER_PATH), "synthesize"]
    }
    evaluation_commands = {
        "{}_isolated_metric_aux".format(spec["fold_id"]): [
            sys.executable,
            str(RUNNER_PATH),
            "evaluate",
            "--eval-id",
            "{}_isolated_metric_aux".format(spec["fold_id"]),
            GPU_AUTHORIZATION_FLAG,
        ]
        for spec in specs
    }
    return {
        "schema": "ev-uav-metric-aux-task-arithmetic-command-audit-v1",
        "created_utc": core.utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "assets": assets,
        "task_arithmetic_preflight": preflight,
        "synthesis_specs": specs,
        "synthesis_commands": synthesis_commands,
        "evaluation_commands": evaluation_commands,
        "report_command": [sys.executable, str(RUNNER_PATH), "report"],
        "allowed_cli_commands": [
            "audit",
            "synthesize",
            "evaluate",
            "report",
            "all-evaluate-report",
        ],
        "new_training_optimizer_steps": 0,
        "candidate_count": 1,
        "alpha": 1.0,
        "new_candidate_evaluation_count": 3,
        "released_m20_anchor_reinference": False,
        "gpu_or_cuda_initialized": False,
        "data_use_statement": (
            "Only frozen train identities, train-derived checkpoints and V4 train-only artifacts were read. "
            "No validation/test, T32 or persistence-formal artifact was opened; CUDA remained uninitialized."
        ),
    }


def run_audit():
    protocol, protocol_sha = load_protocol()
    assets = _BASE_VERIFY_ASSETS(protocol)
    preflight = task_arithmetic_preflight(protocol)
    payload = command_audit_payload(protocol, protocol_sha, assets, preflight)
    _BASE_WRITE_NEW_JSON(COMMAND_AUDIT_PATH, payload)
    print("task-arithmetic command audit:", COMMAND_AUDIT_PATH)
    print("command audit sha256:", core.sha256_file(COMMAND_AUDIT_PATH))
    print("GPU not initialized; synthesis is CPU-only and evaluation awaits root authorization.")
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the immutable task-arithmetic CPU audit first.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-command-audit-v1", "command schema")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "command protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "command runner")
    _expect(payload.get("new_training_optimizer_steps"), 0, "command training steps")
    _expect(payload.get("alpha"), 1.0, "command alpha")
    _expect(payload.get("candidate_count"), 1, "command candidate count")
    _expect(payload.get("new_candidate_evaluation_count"), 3, "command evaluation count")
    _expect(payload.get("released_m20_anchor_reinference"), False, "command anchor policy")
    _expect(payload.get("gpu_or_cuda_initialized"), False, "command CUDA")
    _expect(payload.get("task_arithmetic_preflight", {}).get("passed"), True, "command preflight")
    return payload, digest


def _atomic_torch_save(payload, destination):
    import torch

    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Refusing to overwrite synthesis output: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def run_synthesis():
    import torch

    protocol, _ = load_protocol()
    _, command_audit_sha = load_command_audit()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU checkpoint synthesis.")
    specs = synthesis_specs(protocol)
    if SYNTHESIS_MANIFEST_PATH.exists() or any(Path(spec["output"]).exists() for spec in specs):
        raise FileExistsError("Refusing to overwrite task-arithmetic synthesis evidence.")
    records = []
    for spec in specs:
        before = {
            key: core.sha256_file(Path(spec[key]))
            for key in ("parent", "baseline", "metric_aux")
        }
        for key, digest in before.items():
            _expect(digest, spec["{}_sha256".format(key)], "{} {} before".format(spec["fold_id"], key))
        parent_payload = _load_checkpoint(spec["parent"])
        baseline_payload = _load_checkpoint(spec["baseline"])
        aux_payload = _load_checkpoint(spec["metric_aux"])
        states = [
            payload["model_state_dict"]
            for payload in (parent_payload, baseline_payload, aux_payload)
        ]
        isolated = synthesize_state_dict(*states, alpha=1.0)
        alpha_zero = synthesize_state_dict(*states, alpha=0.0)
        if not _state_equal(alpha_zero, states[0]):
            raise RuntimeError("Alpha=0 identity failed during synthesis.")
        stats = _task_vector_stats(*states, isolated)
        output_payload = {
            "checkpoint_format_version": 2,
            "epoch": -1,
            "next_epoch": -1,
            "loss": 0.0,
            "model_state_dict": isolated,
            "temporal_memory": copy.deepcopy(parent_payload["temporal_memory"]),
            "provenance": {
                "artifact_kind": "inference_only_metric_aux_task_arithmetic",
                "fold_id": spec["fold_id"],
                "formula": "released_m20 + 1.0 * (metric_aux_e3 - baseline_e3)",
                "arithmetic_dtype": "torch.float64_then_single_cast_to_torch.float32",
                "alpha": 1.0,
                "input_checkpoint_sha256": before,
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": core.sha256_file(RUNNER_PATH),
                "new_training_optimizer_steps": 0,
            },
        }
        _atomic_torch_save(output_payload, spec["output"])
        output_sha = core.sha256_file(Path(spec["output"]))
        reloaded = _load_checkpoint(spec["output"])
        if not _state_equal(reloaded["model_state_dict"], isolated):
            raise RuntimeError("Reloaded isolated checkpoint differs from synthesized state.")
        strict_load = _strict_model_state_load_cpu(reloaded["model_state_dict"])
        after = {
            key: core.sha256_file(Path(spec[key]))
            for key in ("parent", "baseline", "metric_aux")
        }
        if before != after:
            raise RuntimeError("An input checkpoint changed during synthesis.")
        records.append(
            {
                "fold_id": spec["fold_id"],
                "output_path": spec["output"],
                "output_sha256": output_sha,
                "model_state_canonical_sha256": model_state_canonical_sha256(isolated),
                "input_checkpoint_sha256": before,
                "input_before_after_equal": True,
                "alpha_zero_parent_identity": True,
                "alpha_one_formula_bitwise": True,
                "strict_reload_bitwise": True,
                "strict_model_load_cpu": strict_load,
                "all_values_finite": True,
                "stats": stats,
            }
        )
    manifest = {
        "schema": "ev-uav-metric-aux-task-arithmetic-synthesis-manifest-v1",
        "created_utc": core.utc_now(),
        "status": "passed",
        "passed": True,
        "alpha": 1.0,
        "candidate_count": 1,
        "new_training_optimizer_steps": 0,
        "records": records,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "command_audit_sha256": command_audit_sha,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    _BASE_WRITE_NEW_JSON(SYNTHESIS_MANIFEST_PATH, manifest)
    print("synthesis manifest:", SYNTHESIS_MANIFEST_PATH)
    print("synthesis manifest sha256:", core.sha256_file(SYNTHESIS_MANIFEST_PATH))
    return manifest


def load_synthesis_manifest(verify_formula=True):
    import torch

    protocol, _ = load_protocol()
    _, audit_sha = load_command_audit()
    payload, digest = core.load_json_snapshot(SYNTHESIS_MANIFEST_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-synthesis-manifest-v1", "manifest schema")
    _expect(payload.get("passed"), True, "manifest passed")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "manifest protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "manifest runner")
    _expect(payload.get("command_audit_sha256"), audit_sha, "manifest command audit")
    _expect(payload.get("alpha"), 1.0, "manifest alpha")
    _expect(payload.get("new_training_optimizer_steps"), 0, "manifest training steps")
    _expect(payload.get("cuda_not_initialized"), True, "manifest CUDA")
    by_fold = {record["fold_id"]: record for record in payload.get("records", [])}
    if set(by_fold) != {"hold_g1", "hold_g2", "hold_g3"}:
        raise RuntimeError("Synthesis manifest does not contain exactly three folds.")
    for spec in synthesis_specs(protocol):
        record = by_fold[spec["fold_id"]]
        for key in ("parent", "baseline", "metric_aux"):
            _expect(
                core.sha256_file(Path(spec[key])),
                spec["{}_sha256".format(key)],
                "manifest {} input SHA".format(key),
            )
            _expect(
                record["input_checkpoint_sha256"][key],
                spec["{}_sha256".format(key)],
                "manifest {} input binding".format(key),
            )
        output_path = Path(record["output_path"])
        _expect(output_path.resolve(), Path(spec["output"]).resolve(), "manifest output path")
        _expect(core.sha256_file(output_path), record["output_sha256"], "manifest output SHA")
        if verify_formula:
            states = [
                _load_checkpoint(spec[key])["model_state_dict"]
                for key in ("parent", "baseline", "metric_aux")
            ]
            expected = synthesize_state_dict(*states, alpha=1.0)
            output_payload = _load_checkpoint(output_path)
            actual = output_payload["model_state_dict"]
            provenance = output_payload.get("provenance", {})
            _expect(provenance.get("alpha"), 1.0, "checkpoint alpha")
            _expect(
                provenance.get("protocol_sha256"),
                EXPECTED_PROTOCOL_SHA256,
                "checkpoint protocol",
            )
            _expect(
                provenance.get("runner_sha256"),
                core.sha256_file(RUNNER_PATH),
                "checkpoint runner",
            )
            _expect(
                provenance.get("input_checkpoint_sha256"),
                record["input_checkpoint_sha256"],
                "checkpoint inputs",
            )
            if not _state_equal(expected, actual):
                raise RuntimeError("Manifest checkpoint does not satisfy alpha=1 formula.")
            strict_load = _strict_model_state_load_cpu(actual)
            _expect(
                strict_load,
                record["strict_model_load_cpu"],
                "manifest strict model load",
            )
            _expect(
                model_state_canonical_sha256(actual),
                record["model_state_canonical_sha256"],
                "manifest model-state canonical",
            )
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized while verifying synthesized CPU artifacts.")
    return payload, digest


def evaluation_specs(protocol, manifest):
    by_fold = {record["fold_id"]: record for record in manifest["records"]}
    output = []
    for fold in protocol["dataset"]["folds"]:
        fold_id = fold["fold_id"]
        record = by_fold[fold_id]
        output.append(
            {
                "eval_id": "{}_isolated_metric_aux".format(fold_id),
                "fold_id": fold_id,
                "variant": "isolated_metric_aux",
                "held_group": fold["held_group"],
                "held_source_names": [
                    item["name"] for item in core.held_items(protocol, fold)
                ],
                "checkpoint": str(Path(record["output_path"]).resolve()),
                "checkpoint_sha256": record["output_sha256"],
                "training_result_path": None,
                "result_path": str(
                    (
                        EVALUATION_ROOT
                        / "{}_isolated_metric_aux".format(fold_id)
                        / "evaluation.json"
                    ).resolve()
                ),
            }
        )
    return output


def _task_arithmetic_recovery_record(manifest_sha):
    return {
        "mode": "adaptive_metric_aux_task_arithmetic_alpha1_v1",
        "formula": "released_m20 + 1.0 * (metric_aux_e3 - baseline_e3)",
        "alpha": 1.0,
        "synthesis_manifest_sha256": manifest_sha,
        "v4_report_sha256": V4_REPORT_SHA256,
        "new_training_optimizer_steps": 0,
        "adaptive_train_only_not_independent_oof": True,
        "released_m20_anchor_reinference": False,
    }


def _verify_inference_helper(protocol):
    import utils.temporal_frame_inference as frame_inference
    import utils.temporal_memory_inference as memory_inference

    if hasattr(memory_inference, "temporal_frame_video_from_sample"):
        raise RuntimeError("Memory inference already exposes the temporary helper.")
    helper = frame_inference.temporal_frame_video_from_sample
    _expect(
        list(inspect.signature(helper).parameters),
        ["sample", "temporal_bin_size", "whole_t"],
        "full-stream helper signature",
    )
    _expect(
        core.sha256_file(Path(memory_inference.__file__).resolve()),
        v4.MEMORY_INFERENCE_SHA256,
        "memory inference SHA",
    )
    _expect(
        core.sha256_file(Path(frame_inference.__file__).resolve()),
        v4.TEMPORAL_FRAME_INFERENCE_SHA256,
        "frame inference SHA",
    )
    return memory_inference, helper


def evaluate_spec(protocol, spec, manifest_sha):
    memory_inference, helper = _verify_inference_helper(protocol)
    original_writer = core.write_new_json
    recovery = _task_arithmetic_recovery_record(manifest_sha)

    def guarded_writer(path, payload):
        _expect(Path(path).resolve(), Path(spec["result_path"]).resolve(), "candidate output path")
        _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "candidate protocol")
        _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "candidate runner")
        _expect(payload.get("checkpoint_sha256"), spec["checkpoint_sha256"], "candidate checkpoint")
        _expect(payload.get("t32_read_or_combined"), False, "candidate T32")
        payload["task_arithmetic_recovery"] = recovery
        return original_writer(path, payload)

    setattr(memory_inference, "temporal_frame_video_from_sample", helper)
    core.write_new_json = guarded_writer
    try:
        payload = _BASE_EVALUATE_SPEC(protocol, spec)
    finally:
        core.write_new_json = original_writer
        if hasattr(memory_inference, "temporal_frame_video_from_sample"):
            delattr(memory_inference, "temporal_frame_video_from_sample")
    _expect(payload.get("task_arithmetic_recovery"), recovery, "candidate recovery")
    return payload


def load_evaluation_result(spec, manifest_sha):
    payload, digest = _BASE_LOAD_EVALUATION_RESULT(spec)
    _expect(payload.get("variant"), "isolated_metric_aux", "candidate variant")
    _expect(
        payload.get("task_arithmetic_recovery"),
        _task_arithmetic_recovery_record(manifest_sha),
        "candidate recovery provenance",
    )
    return payload, digest


def load_m20_anchor(protocol, fold_id):
    record = protocol["v4_evaluation_artifacts_task_arithmetic"][
        "{}_released_m20".format(fold_id)
    ]
    payload, digest = core.load_json_snapshot(
        core.workspace_path(record["workspace_relative_path"])
    )
    _expect(digest, record["sha256"], "{} M20 anchor SHA".format(fold_id))
    _validate_v4_evaluation("{}_released_m20".format(fold_id), payload, protocol)
    return payload, digest


def run_evaluation(eval_id=None, authorized=False):
    core.require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    load_command_audit()
    manifest, manifest_sha = load_synthesis_manifest(verify_formula=True)
    specs = evaluation_specs(protocol, manifest)
    if eval_id is not None:
        matches = [spec for spec in specs if spec["eval_id"] == eval_id]
        if len(matches) != 1:
            raise KeyError("Unknown task-arithmetic evaluation ID: {}".format(eval_id))
        return [evaluate_spec(protocol, matches[0], manifest_sha)]
    results = []
    for spec in specs:
        if Path(spec["result_path"]).is_file():
            payload, _ = load_evaluation_result(spec, manifest_sha)
            results.append(payload)
            continue
        subprocess.run(
            [
                sys.executable,
                str(RUNNER_PATH),
                "evaluate",
                "--eval-id",
                spec["eval_id"],
                GPU_AUTHORIZATION_FLAG,
            ],
            cwd=str(EVC_ROOT),
            check=True,
        )
        payload, _ = load_evaluation_result(spec, manifest_sha)
        results.append(payload)
    return results


def _official_gate(candidate_counts, candidate_metrics, anchor_counts, anchor_metrics, pooled):
    checks = {
        "score_not_lower": float(candidate_metrics["score"]) >= float(anchor_metrics["score"]),
        "pd_not_lower": float(candidate_metrics["pd"]) >= float(anchor_metrics["pd"]),
        "iou_not_lower": float(candidate_metrics["iou"]) >= float(anchor_metrics["iou"]),
        "fa_not_higher": float(candidate_metrics["fa"]) <= float(anchor_metrics["fa"]),
        "correct_objects_not_lower": int(candidate_counts["correct_objects"])
        >= int(anchor_counts["correct_objects"]),
        "population_invariants_equal": core.population_invariants(candidate_counts)
        == core.population_invariants(anchor_counts),
    }
    if pooled:
        checks["score_delta_at_least_0p0002"] = (
            float(candidate_metrics["score"]) - float(anchor_metrics["score"])
            >= 0.0002
        )
        checks["false_positive_events_or_false_components_strictly_lower"] = (
            int(candidate_counts["false_positive_events"])
            < int(anchor_counts["false_positive_events"])
            or int(candidate_counts["false_components"])
            < int(anchor_counts["false_components"])
        )
    return {"checks": checks, "passed": all(checks.values())}


def build_report(protocol, protocol_sha, manifest, manifest_sha, specs):
    import torch
    from crossfit_component_reranker import SufficientCounts, metrics_from_counts

    if torch.cuda.is_initialized():
        raise RuntimeError("Task-arithmetic report must start CUDA-uninitialized.")
    folds = []
    candidate_counts = []
    anchor_counts = []
    evaluation_artifacts = {}
    anchor_artifacts = {}
    held_seen = []
    fold_gates = {}
    for spec in specs:
        candidate, candidate_sha = load_evaluation_result(spec, manifest_sha)
        anchor, anchor_sha = load_m20_anchor(protocol, spec["fold_id"])
        expected_names = spec["held_source_names"]
        _expect([item["source_name"] for item in candidate["records"]], expected_names, "candidate source order")
        _expect([item["source_name"] for item in anchor["records"]], expected_names, "anchor source order")
        held_seen.extend(expected_names)
        c_counts, a_counts = candidate["pooled_counts"], anchor["pooled_counts"]
        c_metrics, a_metrics = candidate["pooled_metrics"], anchor["pooled_metrics"]
        candidate_record_counts = core.add_count_dicts(
            [record["counts"] for record in candidate["records"]]
        )
        anchor_record_counts = core.add_count_dicts(
            [record["counts"] for record in anchor["records"]]
        )
        _expect(candidate_record_counts, c_counts, "candidate record-to-fold pooling")
        _expect(anchor_record_counts, a_counts, "anchor record-to-fold pooling")
        _expect(
            metrics_from_counts(SufficientCounts(**candidate_record_counts)),
            c_metrics,
            "candidate fold metrics",
        )
        _expect(
            metrics_from_counts(SufficientCounts(**anchor_record_counts)),
            a_metrics,
            "anchor fold metrics",
        )
        source_contract = core.source_index(protocol)
        for record in candidate["records"] + anchor["records"]:
            _expect(
                record["source_sha256"],
                source_contract[record["source_name"]]["sha256"],
                "report source SHA",
            )
        gate = _official_gate(c_counts, c_metrics, a_counts, a_metrics, pooled=False)
        fold_gates[spec["fold_id"]] = gate
        folds.append(
            {
                "fold_id": spec["fold_id"],
                "held_group": spec["held_group"],
                "released_m20": {"counts": a_counts, "metrics": a_metrics},
                "isolated_metric_aux": {"counts": c_counts, "metrics": c_metrics},
                "metric_delta": core.metric_delta(c_metrics, a_metrics),
                "count_delta": {
                    key: int(c_counts[key]) - int(a_counts[key]) for key in core.COUNT_FIELDS
                },
                "raw_true_positive_events_delta_reported_not_gated": int(
                    c_counts["true_positive_events"]
                )
                - int(a_counts["true_positive_events"]),
            }
        )
        candidate_counts.append(c_counts)
        anchor_counts.append(a_counts)
        evaluation_artifacts[spec["eval_id"]] = {
            "path": spec["result_path"],
            "sha256": candidate_sha,
        }
        anchor_artifacts["{}_released_m20".format(spec["fold_id"])] = {
            "path": protocol["v4_evaluation_artifacts_task_arithmetic"][
                "{}_released_m20".format(spec["fold_id"])
            ]["workspace_relative_path"],
            "sha256": anchor_sha,
        }
    expected_held_union_order = [
        item["name"]
        for fold in protocol["dataset"]["folds"]
        for item in core.held_items(protocol, fold)
    ]
    if held_seen != expected_held_union_order:
        raise RuntimeError("Task-arithmetic pooled sources are not the exact frozen order.")
    candidate_pooled_counts = core.add_count_dicts(candidate_counts)
    anchor_pooled_counts = core.add_count_dicts(anchor_counts)
    candidate_pooled_metrics = metrics_from_counts(
        SufficientCounts(**candidate_pooled_counts)
    )
    anchor_pooled_metrics = metrics_from_counts(SufficientCounts(**anchor_pooled_counts))
    pooled_gate = _official_gate(
        candidate_pooled_counts,
        candidate_pooled_metrics,
        anchor_pooled_counts,
        anchor_pooled_metrics,
        pooled=True,
    )
    gates = {
        "fold_checks": fold_gates,
        "pooled_check": pooled_gate,
        "checks": {
            "every_fold_official_metrics_against_released_m20": all(
                gate["passed"] for gate in fold_gates.values()
            ),
            "pooled_official_metrics_and_improvement": pooled_gate["passed"],
            "alpha_exactly_one": True,
            "single_candidate_no_grid": True,
            "raw_tp_reported_not_gated_policy_predeclared": True,
        },
    }
    gates["passed"] = all(gates["checks"].values())
    _, command_audit_sha = load_command_audit()
    payload = {
        "schema": "ev-uav-metric-aux-task-arithmetic-h2-grouped-oof-report-v1",
        "created_utc": core.utc_now(),
        "status": "passed" if gates["passed"] else "failed",
        "passed": gates["passed"],
        "evidence_class": protocol["evidence_class"],
        "adaptive_selection_disclosure": protocol[
            "adaptive_selection_disclosure_task_arithmetic"
        ],
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
        "shared_parent_pretraining_exposure": True,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "command_audit_sha256": command_audit_sha,
        "synthesis_manifest_sha256": manifest_sha,
        "synthesis_records": manifest["records"],
        "v4_report_sha256": V4_REPORT_SHA256,
        "fold_results": folds,
        "pooled": {
            "released_m20": {
                "counts": anchor_pooled_counts,
                "metrics": anchor_pooled_metrics,
            },
            "isolated_metric_aux": {
                "counts": candidate_pooled_counts,
                "metrics": candidate_pooled_metrics,
                "metric_delta": core.metric_delta(
                    candidate_pooled_metrics, anchor_pooled_metrics
                ),
                "count_delta": {
                    key: int(candidate_pooled_counts[key])
                    - int(anchor_pooled_counts[key])
                    for key in core.COUNT_FIELDS
                },
                "raw_true_positive_events_delta_reported_not_gated": int(
                    candidate_pooled_counts["true_positive_events"]
                )
                - int(anchor_pooled_counts["true_positive_events"]),
            },
        },
        "promotion_policy_revision": protocol[
            "promotion_policy_revision_task_arithmetic"
        ],
        "promotion_gates": gates,
        "evaluation_artifacts": evaluation_artifacts,
        "reused_released_m20_anchor_artifacts": anchor_artifacts,
        "released_m20_anchor_reinference": False,
        "new_training_optimizer_steps": 0,
        "t32_read_or_combined": False,
        "decision": (
            "eligible_for_preregistered_all11_paired_fit_before_any_validation"
            if gates["passed"]
            else protocol["promotion_policy_revision_task_arithmetic"][
                "failure_action"
            ]
        ),
        "no_default_submission_or_validation_change": True,
    }
    _BASE_WRITE_NEW_JSON(REPORT_PATH, payload)
    return payload


def run_report():
    protocol, protocol_sha = load_protocol()
    load_command_audit()
    manifest, manifest_sha = load_synthesis_manifest(verify_formula=True)
    specs = evaluation_specs(protocol, manifest)
    payload = build_report(protocol, protocol_sha, manifest, manifest_sha, specs)
    print("task-arithmetic report:", REPORT_PATH)
    print("promotion passed:", payload["passed"])
    return payload


def _patch_core_for_task_arithmetic():
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.EXPECTED_PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.COMMAND_AUDIT_PATH = COMMAND_AUDIT_PATH
    core.EVALUATION_ROOT = EVALUATION_ROOT
    core.REPORT_PATH = REPORT_PATH
    core.__file__ = str(RUNNER_PATH)
    core.load_protocol = load_protocol
    core.load_command_audit = load_command_audit


_patch_core_for_task_arithmetic()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("synthesize")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--eval-id", default=None)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("report")
    all_eval = subparsers.add_parser("all-evaluate-report")
    all_eval.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit()
    if args.command == "synthesize":
        return run_synthesis()
    if args.command == "evaluate":
        return run_evaluation(args.eval_id, args.authorized)
    if args.command == "report":
        return run_report()
    if args.command == "all-evaluate-report":
        core.require_gpu_authorization(args.authorized)
        run_evaluation(authorized=True)
        return run_report()
    raise RuntimeError("Unsupported task-arithmetic command: {}".format(args.command))


if __name__ == "__main__":
    main()

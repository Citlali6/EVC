"""CPU-only V2 geometry audit and alpha=1 synthesis for the frozen all11 pair."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
RUNNER_PATH = Path(__file__).resolve()

_PRIVATE_NAME = "_metric_aux_all11_final_v1_for_v2_geometry"
_previous = sys.modules.get(_PRIVATE_NAME)
_V1_PATH = EVC_ROOT / "run_metric_aux_task_arithmetic_all11_final_refit.py"
_SPEC = importlib.util.spec_from_file_location(_PRIVATE_NAME, _V1_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Unable to import the frozen all11 V1 runner privately.")
v1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_PRIVATE_NAME] = v1
try:
    _SPEC.loader.exec_module(v1)
finally:
    if _previous is None:
        sys.modules.pop(_PRIVATE_NAME, None)
    else:
        sys.modules[_PRIVATE_NAME] = _previous

core = v1.core

PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_task_arithmetic_all11_final_refit_science_v2.json"
EXPECTED_PROTOCOL_SHA256 = "e6493681b4265620966fb1a6ea400de69a3926ab71c9a0c493f82824922cbe92"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-task-arithmetic-all11-final-refit-v2"
BASE_PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_h2_grouped_oof_science_v1.json"
BASE_PROTOCOL_SHA256 = "d30d0e45c85ee2f1a8b3c9533ed85f7c5525a421e252d7c1ad2d635fa804aa2b"
V1_PROTOCOL_SHA256 = "570f0bedfc76794ebdbdebefbc7dbac4a00c9c21fe6b7660c1f6bdf7adb05f19"
V1_RUNNER_SHA256 = "04c31193556ee276666c9dbafd7aef8e27c529e197db24a3c8e4071abc2967d5"
V1_FAILURE_SHA256 = "5a2291e81c99b1cb3e1e0d97d05b6afb76d19fa5e38bf77f809b094cda3d3576"

OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260811_metric_aux_task_arithmetic_all11_final_refit_v2"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
PAIR_AUDIT_PATH = OUTPUT_ROOT / "pair_audit.json"
FINAL_CHECKPOINT_PATH = OUTPUT_ROOT / "synthesis" / "metric_aux_task_arithmetic_all11_alpha1.pt"
SYNTHESIS_MANIFEST_PATH = OUTPUT_ROOT / "synthesis" / "synthesis_manifest.json"


def _expect(actual, expected, label):
    core._expect_equal(actual, expected, label)


def _close(actual, expected, label, tolerance=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=1e-15):
        raise RuntimeError("{} differs: {!r} != {!r}".format(label, actual, expected))


def _path(record):
    return core.workspace_path(record["workspace_relative_path"])


def _require_file(record, label):
    path = _path(record)
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    _expect(core.sha256_file(path), record["sha256"], "{} SHA-256".format(label))
    return path


def _load_json(record, label):
    path = _require_file(record, label)
    payload, digest = core.load_json_snapshot(path)
    _expect(digest, record["sha256"], "{} snapshot".format(label))
    return payload, path


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    _expect(actual, EXPECTED_PROTOCOL_SHA256, "V2 protocol SHA-256")
    overlay, digest = core.load_json_snapshot(PROTOCOL_PATH)
    _expect(digest, actual, "V2 protocol snapshot")
    _expect(overlay.get("schema"), EXPECTED_SCHEMA, "V2 schema")
    _expect(overlay.get("status"), "frozen_after_v1_geometry_gate_failure_before_any_v2_pair_receipt_or_synthesis", "V2 status")
    _expect(overlay["cli_contract"]["allowed_commands"], ["audit", "audit-training", "synthesize"], "V2 CLI")
    _expect(overlay["cli_contract"]["train_probe_evaluate_report_validation_or_submission_commands_exposed"], False, "V2 forbidden CLI")
    _expect(overlay.get("validation_or_test_read_allowed"), False, "V2 split access")
    _expect(overlay["amendment_disclosure"]["v1_attempt_remains_failed"], True, "V1 remains failed")
    _expect(overlay["amendment_disclosure"]["new_training_optimizer_steps"], 0, "V2 training steps")

    v1_evidence = overlay["v1_evidence"]
    _expect(v1_evidence["protocol"]["sha256"], V1_PROTOCOL_SHA256, "V1 protocol contract")
    _expect(v1_evidence["runner"]["sha256"], V1_RUNNER_SHA256, "V1 runner contract")
    for key in ("protocol", "runner", "tests"):
        _require_file(v1_evidence[key], "V1 {}".format(key))
    v1_command_audit, _ = _load_json(v1_evidence["command_audit"], "V1 command audit")
    _expect(v1_command_audit.get("schema"), "ev-uav-metric-aux-task-arithmetic-all11-command-audit-v1", "V1 audit schema")
    _expect(v1_command_audit.get("protocol_sha256"), V1_PROTOCOL_SHA256, "V1 audit protocol")
    _expect(v1_command_audit.get("runner_sha256"), V1_RUNNER_SHA256, "V1 audit runner")
    _expect(v1_command_audit.get("expected_optimizer_steps_each_arm"), 264, "V1 audit arm steps")
    _expect(v1_command_audit.get("expected_optimizer_steps_paired_total"), 528, "V1 audit pair steps")
    _expect(v1_command_audit.get("evaluation_commands"), [], "V1 audit evaluation commands")
    failure, _ = _load_json(v1_evidence["failure_receipt"], "V1 failure receipt")
    _expect(v1_evidence["failure_receipt"]["sha256"], V1_FAILURE_SHA256, "V1 failure contract")
    _expect(failure.get("status"), "failed", "V1 failure status")
    _expect(failure.get("passed"), False, "V1 failure passed")
    _expect(failure.get("failed_gate"), "task_over_drift_open_interval", "V1 failed gate")
    _expect(failure.get("v1_attempt_must_remain_failed"), True, "V1 immutable failure")
    failure_evidence = failure["v1_evidence"]
    _expect(failure_evidence["protocol_sha256"], V1_PROTOCOL_SHA256, "failure V1 protocol")
    _expect(failure_evidence["runner_sha256"], V1_RUNNER_SHA256, "failure V1 runner")
    _expect(failure_evidence["command_audit_sha256"], v1_evidence["command_audit"]["sha256"], "failure V1 audit")
    for key in ("pair_audit_must_remain_absent", "final_checkpoint_must_remain_absent", "synthesis_manifest_must_remain_absent"):
        if core.workspace_path(v1_evidence[key]).exists():
            raise RuntimeError("V1 forbidden artifact now exists: {}".format(v1_evidence[key]))

    _expect(core.sha256_file(BASE_PROTOCOL_PATH), BASE_PROTOCOL_SHA256, "base science protocol")
    base, base_sha = core.load_json_snapshot(BASE_PROTOCOL_PATH)
    _expect(base_sha, BASE_PROTOCOL_SHA256, "base science protocol snapshot")
    effective = copy.deepcopy(base)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["audit_amendment"] = {
        "shared_parent_pretraining_exposure": True,
        "claim_scope": overlay["claim_scope"],
        "revision_reason": overlay["amendment_disclosure"]["reason_for_replacement"],
    }
    effective["all11_v2"] = overlay
    effective["_v1_failure"] = failure
    effective["_v1_command_audit"] = v1_command_audit
    return effective, actual


def _validate_runtime(protocol, variant):
    inputs = protocol["all11_v2"]["all11_pair_inputs"]
    record = inputs["{}_runtime_result".format(variant)]
    payload, path = _load_json(record, "{} runtime result".format(variant))
    _expect(payload.get("schema"), "ev-uav-metric-aux-training-result-v1", "runtime schema")
    _expect(payload.get("status"), "completed", "runtime status")
    _expect(payload.get("run_id"), "all11_{}".format(variant), "runtime id")
    _expect(payload.get("variant"), variant, "runtime variant")
    _expect(payload.get("protocol_sha256"), V1_PROTOCOL_SHA256, "runtime V1 protocol")
    _expect(payload.get("runner_sha256"), V1_RUNNER_SHA256, "runtime V1 runner")
    _expect(payload.get("expected_source_names"), inputs["expected_source_names_in_order"], "runtime sources")
    _expect(payload.get("expected_optimizer_steps"), 264, "runtime steps")
    _expect(payload.get("held_group"), None, "runtime held group")
    _expect(payload.get("input_source_sha256_before_after_equal"), True, "runtime sources stable")
    _expect(payload.get("core_sha256_before_after_equal"), True, "runtime core stable")
    _expect(payload.get("auxiliary_loss_audit", {}).get("passed"), True, "runtime auxiliary audit")
    checkpoints = inputs["{}_checkpoints".format(variant)]
    for epoch in ("e1", "e2", "e3"):
        path_value = _require_file(checkpoints[epoch], "{} {}".format(variant, epoch))
        _expect(payload["checkpoints"][epoch]["sha256"], checkpoints[epoch]["sha256"], "runtime checkpoint SHA")
        _expect(Path(payload["checkpoints"][epoch]["path"]).resolve(), path_value.resolve(), "runtime checkpoint path")
        _expect(
            protocol["_v1_failure"]["v1_evidence"]["{}_checkpoints".format(variant)][epoch],
            checkpoints[epoch]["sha256"],
            "failure {} {} checkpoint".format(variant, epoch),
        )
    _expect(
        protocol["_v1_failure"]["v1_evidence"]["{}_runtime_result_sha256".format(variant)],
        record["sha256"],
        "failure {} runtime".format(variant),
    )
    return payload, record["sha256"], path


def _state(payload):
    return payload["model_state_dict"]


def _delta(left, right):
    import torch
    a, b = _state(left), _state(right)
    if list(a) != list(b):
        raise RuntimeError("Model-state key order differs.")
    vectors = {}
    squared = base_squared = 0.0
    maximum = 0.0
    changed_tensors = changed_elements = 0
    finite = True
    module_squared = {}
    for name in a:
        if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
            raise RuntimeError("Model-state metadata differs: {}".format(name))
        value = (a[name].detach().cpu().to(torch.float64) - b[name].detach().cpu().to(torch.float64)).reshape(-1)
        vectors[name] = value
        square = float(torch.sum(value * value))
        squared += square
        base = b[name].detach().cpu().to(torch.float64)
        base_squared += float(torch.sum(base * base))
        maximum = max(maximum, float(torch.max(torch.abs(value))) if value.numel() else 0.0)
        count = int(torch.count_nonzero(value))
        changed_elements += count
        changed_tensors += int(count > 0)
        finite = finite and bool(torch.isfinite(value).all())
        module = name.split(".")[0]
        module_squared[module] = module_squared.get(module, 0.0) + square
    norm = math.sqrt(squared)
    shares = {name: value / squared for name, value in module_squared.items() if squared > 0.0}
    return {
        "l2": norm,
        "max_abs": maximum,
        "relative_l2": norm / math.sqrt(base_squared),
        "base_l2": math.sqrt(base_squared),
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "finite": finite,
        "module_energy_share": shares,
        "vectors": vectors,
    }


def _cosine(left, right):
    import torch
    if list(left) != list(right):
        raise RuntimeError("Task-vector key order differs.")
    dot = left_sq = right_sq = 0.0
    for name in left:
        a, b = left[name], right[name]
        dot += float(torch.sum(a * b))
        left_sq += float(torch.sum(a * a))
        right_sq += float(torch.sum(b * b))
    if left_sq <= 0.0 or right_sq <= 0.0:
        raise RuntimeError("Cannot compute cosine for a zero task vector.")
    return dot / math.sqrt(left_sq * right_sq)


def _public_stats(value):
    return {key: item for key, item in value.items() if key != "vectors"}


def geometry_preflight(protocol):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before V2 CPU geometry audit.")
    overlay = protocol["all11_v2"]
    inputs = overlay["all11_pair_inputs"]
    baseline, baseline_sha, _ = _validate_runtime(protocol, "baseline")
    metric_aux, metric_sha, _ = _validate_runtime(protocol, "metric_aux")
    parent_path = _require_file(inputs["released_m20"], "released M20")
    parent = core.load_torch_checkpoint(parent_path)

    pair = core.compare_pair_checkpoints(baseline, metric_aux)
    _expect(pair.get("audit_version"), "numeric_near_identity_v2", "numeric pair version")
    _expect(pair.get("passed"), True, "all inherited numeric pair gates")
    scope_reaudit = {}
    for variant, result in (("baseline", baseline), ("metric_aux", metric_aux)):
        scope_reaudit[variant] = {}
        for epoch in ("e1", "e2", "e3"):
            scope_reaudit[variant][epoch] = core.checkpoint_scope_audit(protocol, Path(result["checkpoints"][epoch]["path"]))
    auxiliary_reaudit = {
        "baseline": core.validate_auxiliary_stats(protocol, baseline["auxiliary_loss_stats"], "baseline", 3, 88),
        "metric_aux": core.validate_auxiliary_stats(protocol, metric_aux["auxiliary_loss_stats"], "metric_aux", 3, 88),
    }
    b_overrides = core.override_mapping(baseline["overrides"])
    a_overrides = core.override_mapping(metric_aux["overrides"])
    difference_paths = sorted(key for key in set(b_overrides) | set(a_overrides) if b_overrides.get(key) != a_overrides.get(key))
    active_epochs = [epoch for epoch in range(3) if int(metric_aux["auxiliary_loss_stats"]["epochs"][str(epoch)]["target"]["calls"]) or int(metric_aux["auxiliary_loss_stats"]["epochs"][str(epoch)]["component"]["calls"])]

    b_e3 = core.load_torch_checkpoint(_path(inputs["baseline_checkpoints"]["e3"]))
    a_e3 = core.load_torch_checkpoint(_path(inputs["metric_aux_checkpoints"]["e3"]))
    task = _delta(a_e3, b_e3)
    drift = _delta(b_e3, parent)
    e1 = _delta(
        core.load_torch_checkpoint(_path(inputs["metric_aux_checkpoints"]["e1"])),
        core.load_torch_checkpoint(_path(inputs["baseline_checkpoints"]["e1"])),
    )
    e3_over_e1 = task["l2"] / e1["l2"] if e1["l2"] > 0.0 else math.inf
    parent_l2 = drift["base_l2"]
    step_normalized = task["l2"] / math.sqrt(264.0)
    task_over_m20 = task["l2"] / parent_l2
    task_over_drift = task["l2"] / drift["l2"]

    references = {}
    ref_vectors = {}
    reference_contract = overlay["frozen_oof_geometry_reference"]
    for fold_id, record in reference_contract["folds"].items():
        b_path = _require_file(record["baseline_e3"], "{} baseline E3".format(fold_id))
        a_path = _require_file(record["metric_aux_e3"], "{} metric E3".format(fold_id))
        value = _delta(core.load_torch_checkpoint(a_path), core.load_torch_checkpoint(b_path))
        normalized = value["l2"] / math.sqrt(float(record["optimizer_steps"]))
        _close(value["l2"], record["task_l2"], "{} task L2".format(fold_id))
        _close(normalized, record["step_normalized_task_l2"], "{} normalized L2".format(fold_id))
        _close(value["l2"] / parent_l2, record["task_over_m20"], "{} task/M20".format(fold_id))
        ref_vectors[fold_id] = value["vectors"]
        references[fold_id] = {
            "task_l2": value["l2"],
            "step_normalized_task_l2": normalized,
            "task_over_m20": value["l2"] / parent_l2,
        }
    pairwise = {
        "g1_g2": _cosine(ref_vectors["hold_g1"], ref_vectors["hold_g2"]),
        "g1_g3": _cosine(ref_vectors["hold_g1"], ref_vectors["hold_g3"]),
        "g2_g3": _cosine(ref_vectors["hold_g2"], ref_vectors["hold_g3"]),
    }
    for key, value in pairwise.items():
        _close(value, reference_contract["pairwise_cosines"][key], "OOF {} cosine".format(key))
    cosines = {fold_id: _cosine(task["vectors"], vector) for fold_id, vector in ref_vectors.items()}

    gates_contract = overlay["v2_replacement_safety_gates"]
    lower, upper = gates_contract["step_normalized_task_l2_inclusive_oof_envelope"]
    module_bounds = reference_contract["module_energy_share_bounds"]
    module_gates = {
        name: name in task["module_energy_share"] and float(bounds[0]) <= task["module_energy_share"][name] <= float(bounds[1])
        for name, bounds in module_bounds.items()
    }
    old_checks = {
        "numeric_pair_passed": pair["passed"],
        "source_names_and_order_exact": baseline["expected_source_names"] == metric_aux["expected_source_names"] == inputs["expected_source_names_in_order"],
        "expected_steps_each_arm_exact": int(baseline["expected_optimizer_steps"]) == int(metric_aux["expected_optimizer_steps"]) == 264,
        "command_difference_allowlist_exact": difference_paths == ["TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled", "TRAIN.model_save_root"],
        "training_commands_exact": baseline["overrides"] == protocol["_v1_command_audit"]["training_commands"]["all11_baseline"]["overrides"] and metric_aux["overrides"] == protocol["_v1_command_audit"]["training_commands"]["all11_metric_aux"]["overrides"],
        "resolved_preflights_passed": baseline["resolved_preflight"]["passed"] is True and metric_aux["resolved_preflight"]["passed"] is True,
        "active_epochs_exact": active_epochs == [1, 2],
        "both_auxiliary_reaudits_passed": all(value["passed"] for value in auxiliary_reaudit.values()),
        "all_six_scope_reaudits_passed": all(value["passed"] for values in scope_reaudit.values() for value in values.values()),
        "sources_and_core_unchanged": baseline["input_source_sha256_before_after_equal"] is True and metric_aux["input_source_sha256_before_after_equal"] is True and baseline["core_sha256_before_after_equal"] is True and metric_aux["core_sha256_before_after_equal"] is True,
        "task_and_drift_finite_nonzero": task["finite"] and drift["finite"] and task["l2"] > 0.0 and drift["l2"] > 0.0,
    }
    replacement_gates = {
        "step_normalized_task_l2_in_oof_envelope": float(lower) <= step_normalized <= float(upper),
        "cosine_with_every_oof_task_above_floor": all(value >= float(gates_contract["cosine_with_every_oof_task_at_least_oof_pairwise_minimum"]) for value in cosines.values()),
        "task_over_m20_below_scaled_cap": task_over_m20 <= float(gates_contract["task_over_m20_maximum_scaled_by_sqrt_264_over_168"]),
        "all_module_energy_shares_in_oof_bounds": all(module_gates.values()) and set(task["module_energy_share"]) == set(module_bounds),
        "task_max_abs_not_above_oof_maximum": task["max_abs"] <= float(gates_contract["task_max_abs_not_above_oof_maximum"]),
        "changed_model_tensor_count_exact": task["changed_tensor_count"] == int(gates_contract["changed_model_tensor_count_exact"]),
        "e3_signal_over_e1_floor": e3_over_e1 >= float(gates_contract["e3_model_l2_over_e1_numeric_floor_minimum"]),
    }
    checks = {**old_checks, **replacement_gates}
    if not all(checks.values()):
        raise RuntimeError("V2 geometry preflight failed: {}".format(checks))
    if not (task_over_drift >= 0.1):
        raise RuntimeError("V2 recovery no longer reproduces the exact V1 failure condition.")
    return {
        "baseline_runtime_result_sha256": baseline_sha,
        "metric_aux_runtime_result_sha256": metric_sha,
        "numeric_pair_audit": pair,
        "scope_reaudit": scope_reaudit,
        "auxiliary_reaudit": auxiliary_reaudit,
        "old_v1_checks_except_replaced_ratio": old_checks,
        "replacement_gates": replacement_gates,
        "module_gates": module_gates,
        "checks": checks,
        "task": _public_stats(task),
        "baseline_drift": _public_stats(drift),
        "e1_numeric_floor": _public_stats(e1),
        "e3_model_l2_over_e1_numeric_floor": e3_over_e1,
        "step_normalized_task_l2": step_normalized,
        "task_over_m20": task_over_m20,
        "task_over_drift_reported_not_gated": task_over_drift,
        "cosines_with_oof_tasks": cosines,
        "oof_recomputed": references,
        "oof_pairwise_cosines_recomputed": pairwise,
        "active_zero_based_epochs": active_epochs,
        "command_difference_paths": difference_paths,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "passed": True,
    }


def run_audit():
    protocol, protocol_sha = load_protocol()
    preflight = geometry_preflight(protocol)
    payload = {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-v2-command-audit-v1",
        "created_utc": core.utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "geometry_preflight": preflight,
        "allowed_cli_commands": ["audit", "audit-training", "synthesize"],
        "new_training_optimizer_steps": 0,
        "gpu_commands": [],
        "evaluation_or_score_commands": [],
        "v1_attempt_remains_failed": True,
        "data_use_statement": "Only frozen train-derived checkpoint geometry and receipts were read; no evaluation/report/validation/test artifact was opened and CUDA stayed uninitialized.",
    }
    core.write_new_json(COMMAND_AUDIT_PATH, payload)
    print("V2 immutable command audit:", COMMAND_AUDIT_PATH)
    print("command audit sha256:", core.sha256_file(COMMAND_AUDIT_PATH))
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the immutable V2 CPU audit first.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-all11-v2-command-audit-v1", "audit schema")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "audit protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "audit runner")
    _expect(payload.get("new_training_optimizer_steps"), 0, "audit training steps")
    _expect(payload.get("gpu_commands"), [], "audit GPU commands")
    _expect(payload.get("evaluation_or_score_commands"), [], "audit evaluation commands")
    _expect(payload.get("geometry_preflight", {}).get("passed"), True, "audit geometry")
    _expect(all(payload["geometry_preflight"]["checks"].values()), True, "audit gates")
    return payload, digest


def run_pair_audit():
    protocol, _ = load_protocol()
    _, command_audit_sha = load_command_audit()
    if PAIR_AUDIT_PATH.exists():
        raise FileExistsError("Refusing to overwrite the V2 pair audit.")
    preflight = geometry_preflight(protocol)
    payload = {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-pair-audit-v2",
        "created_utc": core.utc_now(),
        "status": "passed",
        "passed": True,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "command_audit_sha256": command_audit_sha,
        "v1_failure_receipt_sha256": V1_FAILURE_SHA256,
        "v1_attempt_remains_failed": True,
        "new_training_optimizer_steps": 0,
        "geometry_audit": preflight,
        "checks": preflight["checks"],
    }
    core.write_new_json(PAIR_AUDIT_PATH, payload)
    print("V2 pair audit:", PAIR_AUDIT_PATH)
    print("pair audit sha256:", core.sha256_file(PAIR_AUDIT_PATH))
    return payload


def load_pair_audit():
    if not PAIR_AUDIT_PATH.is_file():
        raise FileNotFoundError("V2 pair audit has not passed.")
    payload, digest = core.load_json_snapshot(PAIR_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-all11-pair-audit-v2", "pair schema")
    _expect(payload.get("passed"), True, "pair passed")
    _expect(all(payload.get("checks", {}).values()), True, "pair gates")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "pair protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "pair runner")
    _, command_sha = load_command_audit()
    _expect(payload.get("command_audit_sha256"), command_sha, "pair command audit")
    return payload, digest


def run_synthesis():
    import torch
    protocol, _ = load_protocol()
    _, command_sha = load_command_audit()
    pair, pair_sha = load_pair_audit()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before V2 CPU synthesis.")
    if FINAL_CHECKPOINT_PATH.exists() or SYNTHESIS_MANIFEST_PATH.exists():
        raise FileExistsError("Refusing to overwrite V2 synthesis evidence.")
    inputs = protocol["all11_v2"]["all11_pair_inputs"]
    paths = {
        "parent": _require_file(inputs["released_m20"], "released M20"),
        "baseline_e3": _require_file(inputs["baseline_checkpoints"]["e3"], "baseline E3"),
        "metric_aux_e3": _require_file(inputs["metric_aux_checkpoints"]["e3"], "metric E3"),
    }
    before = {name: core.sha256_file(path) for name, path in paths.items()}
    payloads = {name: core.load_torch_checkpoint(path) for name, path in paths.items()}
    states = [_state(payloads[name]) for name in ("parent", "baseline_e3", "metric_aux_e3")]
    output_state = v1._V5_SYNTHESIZE_STATE_DICT(*states, alpha=1.0)
    alpha_zero = v1._V5_SYNTHESIZE_STATE_DICT(*states, alpha=0.0)
    independent = v1._V5_SYNTHESIZE_STATE_DICT(*states, alpha=1.0)
    if not v1._V5_STATE_EQUAL(alpha_zero, states[0]):
        raise RuntimeError("V2 alpha=0 parent identity failed.")
    if not v1._V5_STATE_EQUAL(output_state, independent):
        raise RuntimeError("V2 alpha=1 independent recomputation failed.")
    strict_load = v1._V5_STRICT_LOAD(output_state)
    checkpoint = {
        "checkpoint_format_version": 2,
        "epoch": -1,
        "next_epoch": -1,
        "loss": 0.0,
        "model_state_dict": output_state,
        "temporal_memory": copy.deepcopy(payloads["parent"]["temporal_memory"]),
        "provenance": {
            "artifact_kind": "inference_only_metric_aux_task_arithmetic_all11_v2",
            "formula": "released_m20 + 1.0 * (metric_aux_all11_e3 - baseline_all11_e3)",
            "alpha": 1.0,
            "arithmetic_dtype": "torch.float64_then_single_cast_to_parent_torch.float32",
            "input_checkpoint_sha256": before,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": core.sha256_file(RUNNER_PATH),
            "command_audit_sha256": command_sha,
            "pair_audit_sha256": pair_sha,
            "v1_attempt_remains_failed": True,
            "new_training_optimizer_steps": 0,
        },
    }
    v1._V5_ATOMIC_TORCH_SAVE(checkpoint, FINAL_CHECKPOINT_PATH)
    output_sha = core.sha256_file(FINAL_CHECKPOINT_PATH)
    reloaded = core.load_torch_checkpoint(FINAL_CHECKPOINT_PATH)
    if not v1._V5_STATE_EQUAL(reloaded["model_state_dict"], output_state):
        raise RuntimeError("Reloaded V2 output differs.")
    reload_strict = v1._V5_STRICT_LOAD(reloaded["model_state_dict"])
    after = {name: core.sha256_file(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("A synthesis input changed.")
    manifest = {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-synthesis-manifest-v2",
        "created_utc": core.utc_now(),
        "status": "completed",
        "passed": True,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "command_audit_sha256": command_sha,
        "pair_audit_sha256": pair_sha,
        "v1_failure_receipt_sha256": V1_FAILURE_SHA256,
        "v1_attempt_remains_failed": True,
        "input_checkpoint_sha256": before,
        "input_checkpoint_sha256_before_after_equal": True,
        "output_path": str(FINAL_CHECKPOINT_PATH.resolve()),
        "output_sha256": output_sha,
        "output_model_state_sha256": v1._V5_MODEL_STATE_SHA(output_state),
        "alpha_zero_parent_bitwise_identity": True,
        "alpha_one_formula_bitwise_recompute": True,
        "strict_cpu_model_load": strict_load,
        "reload_strict_cpu_model_load": reload_strict,
        "temporal_memory_metadata_source": "released_m20",
        "optimizer_scheduler_rng_copied": False,
        "new_training_optimizer_steps": 0,
        "evaluation_or_score_run": False,
        "validation_or_test_read": False,
        "default_submission_changed": False,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "geometry_summary": pair["geometry_audit"],
    }
    core.write_new_json(SYNTHESIS_MANIFEST_PATH, manifest)
    print("V2 final checkpoint:", FINAL_CHECKPOINT_PATH)
    print("checkpoint sha256:", output_sha)
    print("manifest sha256:", core.sha256_file(SYNTHESIS_MANIFEST_PATH))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    commands.add_parser("audit-training")
    commands.add_parser("synthesize")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit()
    if args.command == "audit-training":
        return run_pair_audit()
    if args.command == "synthesize":
        return run_synthesis()
    raise RuntimeError("Unsupported V2 command: {}".format(args.command))


if __name__ == "__main__":
    main()

"""V2 recovery runner for the frozen metric-aux H2 grouped OOF experiment.

V2 inherits the complete V1 scientific definition by SHA-256.  Its sole
change is a pre-registered numerical-near-identity audit for paired E1 CUDA
checkpoints.  V1 attempt 1 remains failed; no V1 output is overwritten or
retroactively reclassified.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
from pathlib import Path
import sys

EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
_CORE_PATH = EVC_ROOT / "run_metric_aux_h2_grouped_oof.py"
_CORE_SPEC = importlib.util.spec_from_file_location(
    "_metric_aux_h2_grouped_oof_v1_core_for_v2", _CORE_PATH
)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise ImportError("Unable to create a private V1 core module for V2 recovery.")
core = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules[_CORE_SPEC.name] = core
_CORE_SPEC.loader.exec_module(core)


PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_h2_grouped_oof_science_v2.json"
EXPECTED_PROTOCOL_SHA256 = "29de33fd9412d5b1fd2349acfb9d1dbfb9a109ace1288768d9100113b982ad06"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-h2-grouped-oof-recovery-v2"
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260810_metric_aux_h2_grouped_oof_v2"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
PROBE_RESULT_PATH = OUTPUT_ROOT / "resource_probe" / "runtime_result.json"
PROBE_FAILURE_PATH = OUTPUT_ROOT / "resource_probe" / "probe_failure.json"
FORMAL_ROOT = OUTPUT_ROOT / "formal_training"
PAIR_AUDIT_PATH = FORMAL_ROOT / "pair_audit.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"

V1_PROTOCOL_SHA256 = "d30d0e45c85ee2f1a8b3c9533ed85f7c5525a421e252d7c1ad2d635fa804aa2b"
V1_RUNNER_SHA256 = "9b3f58a8ed4676e78bf75b3daaf33e96b7218f752088e4e7726da7dbeaa3cf5a"
V1_COMMAND_AUDIT_SHA256 = "ffbd79888f96666a04b69154e150d342acb836c8b577300fe5a55bf963f49fe6"
V1_FAILURE_SHA256 = "88eb445f9dc9252e452e4a7ede26921b7fe319479f2b382e9aaa699541ee88c7"

_BASE_VALIDATE_PROTOCOL = core.validate_protocol
_BASE_COMMAND_AUDIT_PAYLOAD = core.command_audit_payload


def _require_bound_file(record, expected_sha256, label):
    path = core.workspace_path(record["workspace_relative_path"])
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    actual = core.sha256_file(path)
    if actual != expected_sha256 or actual != record["sha256"]:
        raise RuntimeError("{} SHA-256 differs from the V2 recovery contract.".format(label))
    return path


def _validate_overlay(overlay, overlay_sha256):
    core._expect_equal(overlay.get("schema"), EXPECTED_SCHEMA, "V2 protocol schema")
    core._expect_equal(
        overlay.get("status"),
        "frozen_before_any_v2_probe_formal_training_or_held_evaluation",
        "V2 protocol status",
    )
    core._expect_equal(overlay_sha256, EXPECTED_PROTOCOL_SHA256, "V2 protocol SHA-256")
    inheritance = overlay["inheritance"]
    core._expect_equal(
        inheritance["entire_scientific_definition_inherited"],
        True,
        "V1 scientific inheritance",
    )
    _require_bound_file(
        inheritance["base_science_protocol"], V1_PROTOCOL_SHA256, "V1 protocol"
    )
    _require_bound_file(inheritance["base_runner"], V1_RUNNER_SHA256, "V1 runner")
    _require_bound_file(
        inheritance["base_command_audit"],
        V1_COMMAND_AUDIT_SHA256,
        "V1 command audit",
    )
    failure_path = _require_bound_file(
        inheritance["attempt1_failure_receipt"],
        V1_FAILURE_SHA256,
        "V1 attempt-1 failure receipt",
    )
    failure, failure_sha = core.load_json_snapshot(failure_path)
    core._expect_equal(failure_sha, V1_FAILURE_SHA256, "V1 failure snapshot")
    core._expect_equal(failure.get("status"), "failed", "V1 attempt-1 status")
    core._expect_equal(failure.get("passed"), False, "V1 attempt-1 passed flag")
    core._expect_equal(
        failure.get("formal_training_started"), False, "V1 formal-training state"
    )
    core._expect_equal(
        failure.get("held_train_evaluation_started"),
        False,
        "V1 held-evaluation state",
    )
    core._expect_equal(
        failure["recovery_policy"]["retroactive_reclassification_forbidden"],
        True,
        "V1 retroactive recovery policy",
    )

    m23 = overlay["historical_m23_sampling_audit"]
    summary_path = _require_bound_file(
        m23["run_summary"], m23["run_summary"]["sha256"], "M23 run summary"
    )
    log_path = _require_bound_file(
        m23["training_log"], m23["training_log"]["sha256"], "M23 training log"
    )
    summary, summary_sha = core.load_json_snapshot(summary_path)
    core._expect_equal(summary_sha, m23["run_summary"]["sha256"], "M23 summary snapshot")
    expected_summary = m23["expected_resolved_config"]
    resolved = summary["resolved_config"]
    for section, expected_values in expected_summary.items():
        for key, expected_value in expected_values.items():
            core._expect_equal(
                resolved[section][key],
                expected_value,
                "M23 resolved {}.{}".format(section, key),
            )
    core._expect_equal(summary["seed"], 49, "M23 summary seed")
    core._expect_equal(summary["start_epoch"], 0, "M23 start epoch")
    core._expect_equal(summary["next_epoch"], 4, "M23 completed epochs")

    log_before = core.sha256_file(log_path)
    log_bytes = log_path.read_bytes()
    log_after = core.sha256_file(log_path)
    core._expect_equal(log_before, log_after, "M23 log before/after SHA-256")
    core._expect_equal(
        core.hashlib.sha256(log_bytes).hexdigest(),
        m23["training_log"]["sha256"],
        "M23 log snapshot SHA-256",
    )
    core._expect_equal(m23["training_log_encoding"], "utf-16", "M23 log encoding")
    log_lines = log_bytes.decode(m23["training_log_encoding"]).splitlines()
    required_lines = m23["required_training_log_lines"]
    for label, expected_line in required_lines.items():
        core._expect_equal(
            sum(line.strip() == expected_line for line in log_lines),
            1,
            "M23 log record {}".format(label),
        )
    sampling_lines = [
        line.strip()[len("sampling summary: ") :]
        for line in log_lines
        if line.strip().startswith("sampling summary: ")
    ]
    core._expect_equal(len(sampling_lines), 1, "M23 sampling-summary record count")
    sampling_summary = core.json.loads(sampling_lines[0])
    expected_sampling_summary = {
        "dense_event_count_cutoff": 200000,
        "dense_video_count": 15,
        "dense_view_multiplier": 8,
        "extra_dense_views": 210,
        "mode": "dense_multiplier",
        "sequence_count": 408,
        "video_count": 99,
        "views_per_video": 2,
    }
    for key, expected_value in expected_sampling_summary.items():
        core._expect_equal(
            sampling_summary[key], expected_value, "M23 sampling {}".format(key)
        )
    expected_counts = {
        "source_video_count": 99,
        "sequence_count_per_epoch": 408,
        "dense_video_count_over_200k": 15,
        "h2_dense_video_count": 11,
        "non_h2_dense_video_count": 4,
        "non_dense_video_count": 84,
        "base_views_per_video": 2,
        "dense_view_multiplier": 8,
        "h2_dense_sequences_per_epoch": 176,
        "non_h2_dense_sequences_per_epoch": 64,
        "non_dense_sequences_per_epoch": 168,
        "epochs": 4,
        "optimizer_steps_total": 1632,
    }
    for key, expected_value in expected_counts.items():
        core._expect_equal(m23[key], expected_value, "M23 {}".format(key))
    core._expect_equal(
        m23["h2_dense_video_count"] + m23["non_h2_dense_video_count"],
        m23["dense_video_count_over_200k"],
        "M23 dense-population partition",
    )
    core._expect_equal(
        m23["dense_video_count_over_200k"] + m23["non_dense_video_count"],
        m23["source_video_count"],
        "M23 full-population partition",
    )
    core._expect_equal(
        m23["h2_dense_sequences_per_epoch"]
        + m23["non_h2_dense_sequences_per_epoch"]
        + m23["non_dense_sequences_per_epoch"],
        m23["sequence_count_per_epoch"],
        "M23 per-epoch sequence partition",
    )
    core._expect_equal(
        m23["sequence_count_per_epoch"] * m23["epochs"],
        m23["optimizer_steps_total"],
        "M23 optimizer-step formula",
    )

    recovery = overlay["recovery_amendment"]
    core._expect_equal(recovery["fresh_v2_probe_required"], True, "fresh V2 probe")
    core._expect_equal(
        recovery["attempt1_checkpoint_reuse_as_v2_pass_forbidden"],
        True,
        "attempt-1 reuse policy",
    )
    core._expect_equal(
        recovery["scientific_candidate_or_training_change"],
        False,
        "scientific candidate change",
    )
    limits = overlay["numeric_near_identity_contract"]
    e1 = limits["e1_zero_based_epoch0"]
    expected_e1 = {
        "model_max_abs_maximum": 1e-6,
        "model_relative_l2_maximum": 1e-7,
        "optimizer_max_abs_maximum": 1e-5,
        "optimizer_global_l2_maximum": 1e-4,
        "epoch_loss_abs_delta_maximum": 1e-7,
    }
    for key, value in expected_e1.items():
        core._expect_equal(e1[key], value, "V2 {}".format(key))
    core._expect_equal(
        limits["e2_zero_based_epoch1"][
            "candidate_model_global_l2_over_e1_numerical_floor_minimum"
        ],
        10.0,
        "V2 E2/E1 signal floor",
    )
    step_contract = limits["optimizer_step_contract"]
    core._expect_equal(
        step_contract["optimizer_param_group_parameter_count"],
        89,
        "V2 optimizer parameter count",
    )
    core._expect_equal(
        step_contract["optimizer_state_entry_count"],
        89,
        "V2 optimizer state count",
    )
    core._expect_equal(step_contract["probe_e1_step"], 8, "V2 probe E1 step")
    core._expect_equal(step_contract["probe_e2_step"], 16, "V2 probe E2 step")
    core._expect_equal(
        overlay["output_contract"]["workspace_relative_directory"],
        "experiments/20260810_metric_aux_h2_grouped_oof_v2",
        "V2 output directory",
    )
    core._expect_equal(overlay["validation_or_test_read_allowed"], False, "V2 split policy")
    core._expect_equal(overlay["t32_allowed"], False, "V2 T32 policy")
    return failure


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "V2 protocol SHA-256 {} differs from frozen {}.".format(
                actual, EXPECTED_PROTOCOL_SHA256
            )
        )
    overlay, snapshot_sha = core.load_json_snapshot(PROTOCOL_PATH)
    core._expect_equal(snapshot_sha, actual, "V2 protocol snapshot")
    _validate_overlay(overlay, actual)
    base_path = core.workspace_path(
        overlay["inheritance"]["base_science_protocol"]["workspace_relative_path"]
    )
    base, base_sha = core.load_json_snapshot(base_path)
    core._expect_equal(base_sha, V1_PROTOCOL_SHA256, "inherited V1 protocol")
    _BASE_VALIDATE_PROTOCOL(base)
    effective = copy.deepcopy(base)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["recovery_amendment_v2"] = copy.deepcopy(overlay["recovery_amendment"])
    effective["numeric_near_identity_contract"] = copy.deepcopy(
        overlay["numeric_near_identity_contract"]
    )
    effective["v2_inheritance"] = copy.deepcopy(overlay["inheritance"])
    effective["historical_m23_sampling_audit_v2"] = copy.deepcopy(
        overlay["historical_m23_sampling_audit"]
    )
    effective["revision_history"] = list(base["revision_history"]) + [
        {
            "recovery_protocol_sha256": actual,
            "reason": overlay["recovery_amendment"]["reason"],
            "attempt1_failure_sha256": V1_FAILURE_SHA256,
            "attempt1_remains_failed": True,
        }
    ]
    effective["outputs"]["workspace_relative_directory"] = overlay[
        "output_contract"
    ]["workspace_relative_directory"]
    return effective, actual


def command_audit_payload(protocol, protocol_sha256, assets, views):
    payload = _BASE_COMMAND_AUDIT_PAYLOAD(
        protocol, protocol_sha256, assets, views
    )
    payload["schema"] = "ev-uav-metric-aux-h2-grouped-oof-command-audit-v2"
    payload["recovery_contract"] = {
        "attempt1_failure_sha256": V1_FAILURE_SHA256,
        "attempt1_remains_failed": True,
        "fresh_v2_probe_required": True,
        "numeric_near_identity_contract": protocol[
            "numeric_near_identity_contract"
        ],
    }
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the V2 CPU audit before any V2 GPU command.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-h2-grouped-oof-command-audit-v2":
        raise RuntimeError("V2 command-audit schema mismatch.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V2 command audit protocol identity mismatch.")
    if payload.get("runner_sha256") != core.sha256_file(Path(__file__).resolve()):
        raise RuntimeError("V2 runner changed after command audit.")
    if payload.get("gpu_or_cuda_initialized") is not False:
        raise RuntimeError("V2 command audit did not remain CPU-only.")
    recovery = payload.get("recovery_contract", {})
    if (
        recovery.get("attempt1_failure_sha256") != V1_FAILURE_SHA256
        or recovery.get("attempt1_remains_failed") is not True
        or recovery.get("fresh_v2_probe_required") is not True
    ):
        raise RuntimeError("V2 command audit recovery binding mismatch.")
    return payload, digest


def _tensor_model_difference(baseline_state, candidate_state):
    import torch

    if set(baseline_state) != set(candidate_state):
        return {"structure_exact": False, "reason": "state_key_set"}
    squared = 0.0
    baseline_squared = 0.0
    maximum = 0.0
    different_tensors = 0
    different_elements = 0
    for name in sorted(baseline_state):
        baseline = baseline_state[name]
        candidate = candidate_state[name]
        if (
            not torch.is_tensor(baseline)
            or not torch.is_tensor(candidate)
            or baseline.shape != candidate.shape
            or baseline.dtype != candidate.dtype
        ):
            return {"structure_exact": False, "reason": "tensor_metadata", "name": name}
        if not baseline.dtype.is_floating_point:
            if not torch.equal(baseline, candidate):
                return {"structure_exact": False, "reason": "nonfloating_value", "name": name}
            continue
        baseline64 = baseline.detach().cpu().to(torch.float64)
        difference = candidate.detach().cpu().to(torch.float64) - baseline64
        nonzero = int(torch.count_nonzero(difference).item())
        if nonzero:
            different_tensors += 1
            different_elements += nonzero
        squared += float(torch.sum(difference * difference).item())
        baseline_squared += float(torch.sum(baseline64 * baseline64).item())
        maximum = max(maximum, float(torch.max(torch.abs(difference)).item()))
    global_l2 = math.sqrt(squared)
    baseline_l2 = math.sqrt(baseline_squared)
    relative_l2 = math.inf if baseline_l2 == 0.0 else global_l2 / baseline_l2
    return {
        "structure_exact": True,
        "finite": all(math.isfinite(value) for value in (maximum, global_l2, relative_l2)),
        "max_abs": maximum,
        "global_l2": global_l2,
        "baseline_global_l2": baseline_l2,
        "relative_l2": relative_l2,
        "different_tensor_count": different_tensors,
        "different_element_count": different_elements,
    }


def _optimizer_difference(baseline, candidate):
    import torch

    checks = {
        "top_level_keys_exact": set(baseline) == set(candidate),
        "param_groups_exact": core.recursive_bitwise_equal(
            baseline.get("param_groups"), candidate.get("param_groups")
        ),
        "state_keys_exact": set(baseline.get("state", {}))
        == set(candidate.get("state", {})),
    }
    if not all(checks.values()):
        return {"structure_and_nonfloating_state_exact": False, "checks": checks}
    squared = 0.0
    maximum = 0.0
    different_tensors = 0
    different_elements = 0
    for state_key in sorted(baseline["state"]):
        left = baseline["state"][state_key]
        right = candidate["state"][state_key]
        if set(left) != set(right):
            checks["per_parameter_field_keys_exact"] = False
            return {"structure_and_nonfloating_state_exact": False, "checks": checks}
        for field in sorted(left):
            left_value = left[field]
            right_value = right[field]
            if torch.is_tensor(left_value) or torch.is_tensor(right_value):
                if (
                    not torch.is_tensor(left_value)
                    or not torch.is_tensor(right_value)
                    or left_value.shape != right_value.shape
                    or left_value.dtype != right_value.dtype
                ):
                    checks["tensor_metadata_exact"] = False
                    return {"structure_and_nonfloating_state_exact": False, "checks": checks}
                if field == "step":
                    if not torch.equal(left_value, right_value):
                        checks["step_state_exact"] = False
                        return {
                            "structure_and_nonfloating_state_exact": False,
                            "checks": checks,
                        }
                    continue
                if not left_value.dtype.is_floating_point:
                    if not torch.equal(left_value, right_value):
                        checks["nonfloating_tensor_state_exact"] = False
                        return {"structure_and_nonfloating_state_exact": False, "checks": checks}
                    continue
                difference = (
                    right_value.detach().cpu().to(torch.float64)
                    - left_value.detach().cpu().to(torch.float64)
                )
                nonzero = int(torch.count_nonzero(difference).item())
                if nonzero:
                    different_tensors += 1
                    different_elements += nonzero
                squared += float(torch.sum(difference * difference).item())
                maximum = max(maximum, float(torch.max(torch.abs(difference)).item()))
            elif not core.recursive_bitwise_equal(left_value, right_value):
                checks["non_tensor_state_exact"] = False
                return {"structure_and_nonfloating_state_exact": False, "checks": checks}
    checks.update(
        {
            "per_parameter_field_keys_exact": True,
            "tensor_metadata_exact": True,
            "nonfloating_tensor_state_exact": True,
            "step_state_exact": True,
            "non_tensor_state_exact": True,
        }
    )
    global_l2 = math.sqrt(squared)
    return {
        "structure_and_nonfloating_state_exact": True,
        "finite": math.isfinite(maximum) and math.isfinite(global_l2),
        "max_abs": maximum,
        "global_l2": global_l2,
        "different_state_tensor_count": different_tensors,
        "different_element_count": different_elements,
        "checks": checks,
    }


def _optimizer_step_audit(optimizer_state, expected_step, contract):
    import torch

    param_count = sum(
        len(group.get("params", []))
        for group in optimizer_state.get("param_groups", [])
    )
    states = optimizer_state.get("state", {})
    step_values = []
    finite_integer = True
    for state in states.values():
        step = state.get("step")
        if not torch.is_tensor(step) or step.numel() != 1:
            finite_integer = False
            continue
        value = float(step.detach().cpu().item())
        if not math.isfinite(value) or value != int(value):
            finite_integer = False
        step_values.append(value)
    checks = {
        "optimizer_param_group_parameter_count": param_count
        == int(contract["optimizer_param_group_parameter_count"]),
        "optimizer_state_entry_count": len(states)
        == int(contract["optimizer_state_entry_count"]),
        "every_state_has_finite_integer_scalar_step": finite_integer
        and len(step_values) == len(states),
        "every_state_step_matches_expected": len(step_values) == len(states)
        and all(value == float(expected_step) for value in step_values),
    }
    return {
        "expected_step": int(expected_step),
        "observed_unique_steps": sorted(set(step_values)),
        "optimizer_param_group_parameter_count": param_count,
        "optimizer_state_entry_count": len(states),
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_pair_checkpoints(baseline_result, candidate_result):
    baseline_e1 = core.load_torch_checkpoint(
        baseline_result["checkpoints"]["e1"]["path"]
    )
    candidate_e1 = core.load_torch_checkpoint(
        candidate_result["checkpoints"]["e1"]["path"]
    )
    baseline_e2 = core.load_torch_checkpoint(
        baseline_result["checkpoints"]["e2"]["path"]
    )
    candidate_e2 = core.load_torch_checkpoint(
        candidate_result["checkpoints"]["e2"]["path"]
    )
    has_e3 = "e3" in baseline_result["checkpoints"] or "e3" in candidate_result[
        "checkpoints"
    ]
    if has_e3 and not (
        "e3" in baseline_result["checkpoints"]
        and "e3" in candidate_result["checkpoints"]
    ):
        raise RuntimeError("Only one paired training result contains E3.")
    baseline_e3 = (
        core.load_torch_checkpoint(baseline_result["checkpoints"]["e3"]["path"])
        if has_e3
        else None
    )
    candidate_e3 = (
        core.load_torch_checkpoint(candidate_result["checkpoints"]["e3"]["path"])
        if has_e3
        else None
    )
    e1_model = _tensor_model_difference(
        baseline_e1["model_state_dict"], candidate_e1["model_state_dict"]
    )
    e1_optimizer = _optimizer_difference(
        baseline_e1["optimizer_state_dict"], candidate_e1["optimizer_state_dict"]
    )
    e2_model = _tensor_model_difference(
        baseline_e2["model_state_dict"], candidate_e2["model_state_dict"]
    )
    limits = load_protocol()[0]["numeric_near_identity_contract"]
    e1_limits = limits["e1_zero_based_epoch0"]
    e2_limits = limits["e2_zero_based_epoch1"]
    step_contract = limits["optimizer_step_contract"]
    epoch_count = len(baseline_result["auxiliary_loss_stats"]["epochs"])
    if (
        epoch_count <= 0
        or int(baseline_result["expected_optimizer_steps"]) % epoch_count != 0
        or int(candidate_result["expected_optimizer_steps"]) != int(
            baseline_result["expected_optimizer_steps"]
        )
    ):
        raise RuntimeError("Paired optimizer-step denominator is inconsistent.")
    steps_per_epoch = int(baseline_result["expected_optimizer_steps"]) // epoch_count
    baseline_e1_steps = _optimizer_step_audit(
        baseline_e1["optimizer_state_dict"], steps_per_epoch, step_contract
    )
    candidate_e1_steps = _optimizer_step_audit(
        candidate_e1["optimizer_state_dict"], steps_per_epoch, step_contract
    )
    baseline_e2_steps = _optimizer_step_audit(
        baseline_e2["optimizer_state_dict"], steps_per_epoch * 2, step_contract
    )
    candidate_e2_steps = _optimizer_step_audit(
        candidate_e2["optimizer_state_dict"], steps_per_epoch * 2, step_contract
    )
    baseline_e3_steps = (
        _optimizer_step_audit(
            baseline_e3["optimizer_state_dict"], steps_per_epoch * 3, step_contract
        )
        if has_e3
        else None
    )
    candidate_e3_steps = (
        _optimizer_step_audit(
            candidate_e3["optimizer_state_dict"], steps_per_epoch * 3, step_contract
        )
        if has_e3
        else None
    )
    e1_floor = float(e1_model.get("global_l2", math.inf))
    e2_signal = float(e2_model.get("global_l2", 0.0))
    signal_ratio = (
        math.inf if e1_floor == 0.0 and e2_signal > 0.0
        else 0.0 if e1_floor == 0.0
        else e2_signal / e1_floor
    )
    baseline_overrides = core.override_mapping(baseline_result["overrides"])
    candidate_overrides = core.override_mapping(candidate_result["overrides"])
    differences = {
        key
        for key in set(baseline_overrides) | set(candidate_overrides)
        if baseline_overrides.get(key) != candidate_overrides.get(key)
    }
    expected_differences = {
        "TRAIN.model_save_root",
        "TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled",
    }
    baseline_stats = baseline_result["auxiliary_loss_stats"]["epochs"]
    candidate_stats = candidate_result["auxiliary_loss_stats"]["epochs"]
    epoch0_aux_zero = all(
        float(stats["0"][loss_name]["raw_loss_sum"]) == 0.0
        and int(stats["0"][loss_name]["calls"]) == 0
        for stats in (baseline_stats, candidate_stats)
        for loss_name in ("target", "component")
    )
    epoch1_candidate = candidate_stats["1"]
    candidate_aux_positive = all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (
            epoch1_candidate["target"]["raw_loss_mean"],
            epoch1_candidate["target"]["weighted_loss_mean"],
            epoch1_candidate["component"]["raw_loss_mean"],
            epoch1_candidate["component"]["weighted_loss_mean"],
        )
    )
    loss_delta = abs(float(candidate_e1["loss"]) - float(baseline_e1["loss"]))
    checks = {
        "source_names_and_order_exact": baseline_result["expected_source_names"]
        == candidate_result["expected_source_names"],
        "source_before_after_sha_flags_exact": baseline_result[
            "input_source_sha256_before_after_equal"
        ] is True
        and candidate_result["input_source_sha256_before_after_equal"] is True,
        "command_difference_allowlist_exact": differences == expected_differences,
        "e1_epoch0_aux_exact_zero": epoch0_aux_zero,
        "e1_model_structure_exact": e1_model.get("structure_exact") is True,
        "e1_model_finite": e1_model.get("finite") is True,
        "e1_model_max_abs_within_limit": float(e1_model.get("max_abs", math.inf))
        <= float(e1_limits["model_max_abs_maximum"]),
        "e1_model_relative_l2_within_limit": float(
            e1_model.get("relative_l2", math.inf)
        )
        <= float(e1_limits["model_relative_l2_maximum"]),
        "e1_optimizer_structure_and_nonfloating_exact": e1_optimizer.get(
            "structure_and_nonfloating_state_exact"
        ) is True,
        "e1_baseline_optimizer_steps_exact": baseline_e1_steps["passed"],
        "e1_candidate_optimizer_steps_exact": candidate_e1_steps["passed"],
        "e1_optimizer_finite": e1_optimizer.get("finite") is True,
        "e1_optimizer_max_abs_within_limit": float(
            e1_optimizer.get("max_abs", math.inf)
        )
        <= float(e1_limits["optimizer_max_abs_maximum"]),
        "e1_optimizer_global_l2_within_limit": float(
            e1_optimizer.get("global_l2", math.inf)
        )
        <= float(e1_limits["optimizer_global_l2_maximum"]),
        "e1_epoch_loss_abs_delta_within_limit": loss_delta
        <= float(e1_limits["epoch_loss_abs_delta_maximum"]),
        "e1_scheduler_state_exact": core.recursive_bitwise_equal(
            baseline_e1["scheduler_state_dict"], candidate_e1["scheduler_state_dict"]
        ),
        "e1_rng_state_exact": core.recursive_bitwise_equal(
            baseline_e1["rng_state"], candidate_e1["rng_state"]
        ),
        "e2_model_structure_exact": e2_model.get("structure_exact") is True,
        "e2_model_finite": e2_model.get("finite") is True,
        "e2_model_global_l2_positive": e2_signal > 0.0,
        "e2_model_signal_over_e1_floor": signal_ratio
        >= float(
            e2_limits[
                "candidate_model_global_l2_over_e1_numerical_floor_minimum"
            ]
        ),
        "e2_baseline_optimizer_steps_exact": baseline_e2_steps["passed"],
        "e2_candidate_optimizer_steps_exact": candidate_e2_steps["passed"],
        "e2_scheduler_state_exact": core.recursive_bitwise_equal(
            baseline_e2["scheduler_state_dict"], candidate_e2["scheduler_state_dict"]
        ),
        "e2_rng_state_exact": core.recursive_bitwise_equal(
            baseline_e2["rng_state"], candidate_e2["rng_state"]
        ),
        "e2_candidate_aux_losses_finite_positive": candidate_aux_positive,
        "e2_candidate_target_groups_positive": int(
            epoch1_candidate["target"]["group_count"]
        )
        > 0,
        "e2_candidate_component_counts_positive": int(
            epoch1_candidate["component"]["candidate_cell_count"]
        )
        > 0
        and int(epoch1_candidate["component"]["hard_cell_count"]) > 0,
    }
    if has_e3:
        checks.update(
            {
                "e3_baseline_optimizer_steps_exact": baseline_e3_steps["passed"],
                "e3_candidate_optimizer_steps_exact": candidate_e3_steps["passed"],
            }
        )
    return {
        "audit_version": "numeric_near_identity_v2",
        "e1_model": e1_model,
        "e1_optimizer": e1_optimizer,
        "e1_epoch_loss_abs_delta": loss_delta,
        "optimizer_step_audits": {
            "baseline_e1": baseline_e1_steps,
            "candidate_e1": candidate_e1_steps,
            "baseline_e2": baseline_e2_steps,
            "candidate_e2": candidate_e2_steps,
            "baseline_e3": baseline_e3_steps,
            "candidate_e3": candidate_e3_steps,
        },
        "e2_model": e2_model,
        "e2_model_global_l2_over_e1_numerical_floor": signal_ratio,
        "command_difference_paths": sorted(differences),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_probe(authorized=False):
    core.require_gpu_authorization(authorized)
    core.require_idle_gpu()
    protocol, _ = load_protocol()
    command_audit, command_audit_sha = load_command_audit()
    views = command_audit["data_views"]
    if PROBE_RESULT_PATH.exists() or PROBE_FAILURE_PATH.exists():
        raise FileExistsError("Refusing to overwrite completed V2 probe evidence.")
    results = {}
    for spec in core.probe_specs(protocol, views):
        print("starting fresh V2 paired probe:", spec["run_id"], flush=True)
        results[spec["variant"]] = core.run_training_spec(protocol, spec)
    pair = compare_pair_checkpoints(results["baseline"], results["metric_aux"])
    if not pair["passed"]:
        failure_payload = {
            "schema": "ev-uav-metric-aux-resource-probe-failure-v2",
            "created_utc": core.utc_now(),
            "status": "failed",
            "passed": False,
            "failure_gate": "fresh_v2_paired_numeric_near_identity_or_e2_signal",
            "paired_numeric_near_identity_audit": pair,
            "paired_training_results": {
                spec["variant"]: {
                    "path": spec["result_path"],
                    "sha256": core.sha256_file(spec["result_path"]),
                }
                for spec in core.probe_specs(protocol, views)
            },
            "v1_attempt1_failure_sha256": V1_FAILURE_SHA256,
            "v1_attempt1_remains_failed": True,
            "formal_training_started": False,
            "held_train_evaluation_started": False,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        }
        core.write_new_json(PROBE_FAILURE_PATH, failure_payload)
        raise RuntimeError("Fresh V2 paired numeric-identity audit failed: {}".format(pair))
    synthetic = core.synthetic_metric_gradient_probe(protocol, device_name="cpu")
    real_batch = core.real_batch_metric_gradient_probe(
        protocol, results["metric_aux"], views["probe"]["root"]
    )
    candidate_stats = results["metric_aux"]["auxiliary_loss_stats"]["epochs"]
    checks = {
        "fresh_v2_outputs_only": all(
            str(OUTPUT_ROOT.resolve()).lower()
            in str(Path(result["run_directory"]).resolve()).lower()
            for result in results.values()
        ),
        "attempt1_not_reused": True,
        "paired_numeric_near_identity_and_signal": pair["passed"],
        "candidate_epoch0_target_exact_zero": float(
            candidate_stats["0"]["target"]["raw_loss_sum"]
        )
        == 0.0,
        "candidate_epoch0_component_exact_zero": float(
            candidate_stats["0"]["component"]["raw_loss_sum"]
        )
        == 0.0,
        "candidate_epoch1_target_positive": float(
            candidate_stats["1"]["target"]["raw_loss_mean"]
        )
        > 0.0,
        "candidate_epoch1_component_positive": float(
            candidate_stats["1"]["component"]["raw_loss_mean"]
        )
        > 0.0,
        "candidate_epoch1_target_groups_positive": int(
            candidate_stats["1"]["target"]["group_count"]
        )
        > 0,
        "candidate_epoch1_candidate_cells_positive": int(
            candidate_stats["1"]["component"]["candidate_cell_count"]
        )
        > 0,
        "candidate_epoch1_hard_cells_positive": int(
            candidate_stats["1"]["component"]["hard_cell_count"]
        )
        > 0,
        "synthetic_autograd_and_fresh_optimizer": synthetic["passed"],
        "real_epoch1_batch_parameter_gradients_and_fresh_optimizer": real_batch[
            "passed"
        ],
    }
    payload = {
        "schema": "ev-uav-metric-aux-resource-probe-result-v2",
        "created_utc": core.utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed": all(checks.values()),
        "v1_attempt1_failure_sha256": V1_FAILURE_SHA256,
        "v1_attempt1_remains_failed": True,
        "paired_numeric_near_identity_audit": pair,
        "synthetic_metric_gradient_probe": synthetic,
        "real_epoch1_batch_metric_gradient_probe": real_batch,
        "paired_training_results": {
            variant: {
                "result_path": next(
                    spec["result_path"]
                    for spec in core.probe_specs(protocol, views)
                    if spec["variant"] == variant
                ),
                "result_sha256": core.sha256_file(
                    next(
                        spec["result_path"]
                        for spec in core.probe_specs(protocol, views)
                        if spec["variant"] == variant
                    )
                ),
                "e2_checkpoint_sha256": result["checkpoints"]["e2"]["sha256"],
            }
            for variant, result in results.items()
        },
        "command_audit_sha256": command_audit_sha,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "shared_parent_pretraining_exposure": True,
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
    }
    core.write_new_json(PROBE_RESULT_PATH, payload)
    if not payload["passed"]:
        raise RuntimeError("V2 resource probe failed: {}".format(checks))
    print("V2 resource probe passed:", PROBE_RESULT_PATH)
    return payload


def require_probe_passed():
    payload, digest = core.load_json_snapshot(PROBE_RESULT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-resource-probe-result-v2":
        raise RuntimeError("V2 resource-probe schema mismatch.")
    if payload.get("passed") is not True or not all(payload.get("checks", {}).values()):
        raise RuntimeError("Fresh V2 resource probe has not passed every gate.")
    if payload.get("v1_attempt1_remains_failed") is not True:
        raise RuntimeError("V2 receipt retroactively changed V1 attempt 1.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V2 resource-probe protocol identity mismatch.")
    if payload.get("runner_sha256") != core.sha256_file(Path(__file__).resolve()):
        raise RuntimeError("V2 resource-probe runner identity mismatch.")
    return payload, digest


def _patch_core_for_v2():
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.EXPECTED_PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.COMMAND_AUDIT_PATH = COMMAND_AUDIT_PATH
    core.PROBE_RESULT_PATH = PROBE_RESULT_PATH
    core.FORMAL_ROOT = FORMAL_ROOT
    core.PAIR_AUDIT_PATH = PAIR_AUDIT_PATH
    core.EVALUATION_ROOT = EVALUATION_ROOT
    core.REPORT_PATH = REPORT_PATH
    core.__file__ = str(Path(__file__).resolve())
    core.load_protocol = load_protocol
    core.command_audit_payload = command_audit_payload
    core.load_command_audit = load_command_audit
    core.compare_pair_checkpoints = compare_pair_checkpoints
    core.require_probe_passed = require_probe_passed


_patch_core_for_v2()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="CPU-only V2 recovery audit.")
    probe = subparsers.add_parser("probe", help="Run a fresh paired V2 GPU probe.")
    probe.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    train = subparsers.add_parser("train", help="Run all or one V2 formal E3 training.")
    train.add_argument("--run-id", default=None)
    train.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("audit-training", help="CPU-audit completed V2 formal pairs.")
    evaluate = subparsers.add_parser("evaluate", help="Run V2 held-train evaluation.")
    evaluate.add_argument("--eval-id", default=None)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("report", help="Apply inherited held-only double-anchor gates.")
    all_after_probe = subparsers.add_parser("all-after-probe")
    all_after_probe.add_argument(
        GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return core.run_audit()
    if args.command == "probe":
        return run_probe(args.authorized)
    if args.command == "train":
        return core.run_formal_training(args.run_id, args.authorized)
    if args.command == "audit-training":
        return core.run_formal_pair_audit_command()
    if args.command == "evaluate":
        return core.run_formal_evaluation(args.eval_id, args.authorized)
    if args.command == "report":
        return core.run_report()
    if args.command == "all-after-probe":
        core.require_gpu_authorization(args.authorized)
        core.run_formal_training(authorized=True)
        core.run_formal_evaluation(authorized=True)
        return core.run_report()
    raise RuntimeError("Unsupported command: {}".format(args.command))


if __name__ == "__main__":
    main()

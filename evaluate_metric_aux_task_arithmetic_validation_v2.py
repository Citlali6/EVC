"""Governance-approved V2 recovery for the W_full one-shot validation replay.

V1 remains failed and consumed.  This version is a new adaptive attempt whose
only implementation amendment is the golden-report JSON schema adapter.  The
candidate, input-only route, thresholds, full-T160 inference, C00, safety
gates, materiality reporting, and authorization boundary are inherited from
and checked bitwise against V1.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
V1_RUNNER_PATH = (PROJECT_ROOT / "evaluate_metric_aux_task_arithmetic_validation.py").resolve()
V1_SCIENCE_PATH = (
    PROJECT_ROOT / "protocols" / "metric_aux_task_arithmetic_wfull_val24_science_v1.json"
).resolve()
V2_SCIENCE_PATH = (
    PROJECT_ROOT / "protocols" / "metric_aux_task_arithmetic_wfull_val24_science_v2.json"
).resolve()
V2_EXPERIMENT_DIRECTORY = (
    WORKSPACE_ROOT / "experiments" / "20260811_metric_aux_task_arithmetic_wfull_val24_v2_recovery"
).resolve()

V1_RUNNER_SHA256 = "0dae6d90d3a70837d1c01a0798743135d22bd4b99b4515340e1d4e34c74e3a95"
V1_SCIENCE_SHA256 = "fabf6c622a0b4d07905a522c52dee67eb76b50f6a625be4acdc349638ab5b1e0"
V2_SCIENCE_SHA256 = "92e181cc5e89c041b6f17610cd0ca00107b4ee770210e6f182aef897dab3f47b"

OVERLAY_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-recovery-overlay-v2"
EFFECTIVE_SCIENCE_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-science-v2"
EFFECTIVE_STATUS = "frozen_after_v1_parser_failure_before_any_v2_validation_read"
EVIDENCE_CLASS = "second_adaptive_attempt_after_v1_implementation_failure_not_v1_resume_or_independent_held"

V1_FAILURE_INPUT_NAMES = (
    "v1_execution",
    "v1_cpu_receipt",
    "v1_runtime_receipt",
    "v1_claim",
    "v1_failure_report",
)
V1_FAILURE_SHA256 = {
    "v1_execution": "35f3b02c3a7c8c9af3e93afabdb4859501f18b984eb288e891552210ae4d750d",
    "v1_cpu_receipt": "0b462b5f7ed85047103b01e54fde8e23bb7cbf15924a622f5a99f54b8def13f0",
    "v1_runtime_receipt": "9de08cc5e0320ece43d73c921285197478c59dcef589518b9d96f15a54648db4",
    "v1_claim": "d8c3af89b091969a1d3c48ffb91757ff1e9670b72c8d344a7b686ac9b762bcaa",
    "v1_failure_report": "1bca942ada0a40dc676f1255cf831167b50169c424e6302edba1eb380d391ba9",
}

RAW_TO_GOLDEN_COUNT = {
    "event_true_positives": "true_positive_events",
    "event_false_positives": "false_positive_events",
    "ground_truth_positive_events": "positive_events",
    "evaluator_detected_objects": "detected_target_frames",
    "evaluator_objects": "target_frames",
    "evaluator_false_components": "false_components",
    "evaluator_frames": "frame_count",
}


def _load_private_v1_core():
    name = "_wfull_val24_v1_core_for_v2_recovery"
    spec = importlib.util.spec_from_file_location(name, V1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot create the private V1 runner module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = _load_private_v1_core()
_v1_load_json_snapshot = core._load_json_snapshot
_v1_validate_science_protocol = core.validate_science_protocol
_v1_validate_train_only_evidence = core._validate_train_only_evidence


def _load_raw_overlay():
    payload, digest = _v1_load_json_snapshot(
        V2_SCIENCE_PATH,
        V2_SCIENCE_SHA256,
        "V2 recovery overlay",
    )
    validate_recovery_overlay(payload)
    return payload, digest


def validate_recovery_overlay(overlay):
    if overlay.get("schema") != OVERLAY_SCHEMA:
        raise ValueError("Unexpected V2 recovery-overlay schema.")
    if overlay.get("status") != EFFECTIVE_STATUS:
        raise ValueError("V2 overlay is not frozen at its blind boundary.")
    if overlay.get("candidate_id") != "metric_aux_task_arithmetic_all11_alpha1_wfull_h2_full_t160":
        raise ValueError("V2 overlay candidate id differs from V1.")
    if overlay.get("base_v1_science") != {
        "workspace_relative_path": "EVC-work/protocols/metric_aux_task_arithmetic_wfull_val24_science_v1.json",
        "sha256": V1_SCIENCE_SHA256,
    }:
        raise ValueError("V2 overlay does not bind the exact V1 science protocol.")
    if overlay.get("base_v1_runner") != {
        "workspace_relative_path": "EVC-work/evaluate_metric_aux_task_arithmetic_validation.py",
        "sha256": V1_RUNNER_SHA256,
    }:
        raise ValueError("V2 overlay does not bind the exact V1 runner.")
    governance = overlay.get("governance_exception", {})
    required_true = (
        "explicitly_approved_by_root",
        "v1_attempt_remains_failed_and_consumed",
        "v2_is_not_a_v1_resume_or_retry_under_the_v1_contract",
        "v2_is_a_new_second_adaptive_attempt_after_an_implementation_failure",
        "candidate_performance_observed_before_v2_definition",
        "formal_validation_wfull_inference_completed_in_v1",
        "formal_validation_candidate_cache_created_in_v1",
        "scientific_candidate_route_threshold_c00_and_gates_changed",
    )
    for name in required_true[:4]:
        if governance.get(name) is not True:
            raise ValueError("Required governance disclosure is false: {}".format(name))
    for name in required_true[4:]:
        if governance.get(name) is not False:
            raise ValueError("Required no-result/no-change disclosure is false: {}".format(name))
    if governance.get("failure_happened_before_golden_cache_load_raw_npz_load_or_candidate_inference") is not True:
        raise ValueError("V1 failure stage disclosure differs.")
    if governance.get("independent_held_or_unbiased_claim_allowed") is not False:
        raise ValueError("Independent-held wording must remain forbidden.")
    evidence = overlay.get("v1_failure_evidence", {})
    if set(evidence) != set(V1_FAILURE_INPUT_NAMES) | {"v1_h2_cache"}:
        raise ValueError("V1 failure evidence key set differs.")
    for name in V1_FAILURE_INPUT_NAMES:
        if evidence[name].get("sha256") != V1_FAILURE_SHA256[name]:
            raise ValueError("V1 failure evidence SHA differs for {}.".format(name))
    if evidence["v1_h2_cache"].get("required_absent") is not True:
        raise ValueError("V1 H2 cache absence is not required.")
    amendment = overlay.get("sole_implementation_amendment", {})
    if amendment.get("scope") != "golden_report_schema_adapter_only":
        raise ValueError("V2 amendment scope differs.")
    if amendment.get("candidate_or_scientific_change_allowed") is not False:
        raise ValueError("V2 overlay permits a scientific change.")
    if amendment.get("raw_counts_to_golden_counts") != RAW_TO_GOLDEN_COUNT:
        raise ValueError("V2 count-key mapping differs.")
    budget = overlay.get("v2_attempt_budget", {})
    if (
        budget.get("full_val24_replays") != 1
        or budget.get("prior_v1_attempts") != 1
        or budget.get("claim_is_irreversible") is not True
    ):
        raise ValueError("V2 attempt budget differs.")
    if overlay.get("original_train_only_evidence_entries_required_bitwise_equal_to_v1") is not True:
        raise ValueError("Original train-only evidence equality is not required.")
    return overlay


def _materialize_science(overlay):
    validate_recovery_overlay(overlay)
    if core.sha256_file(V1_RUNNER_PATH) != V1_RUNNER_SHA256:
        raise ValueError("V1 runner changed before V2 materialization.")
    base, _ = _v1_load_json_snapshot(V1_SCIENCE_PATH, V1_SCIENCE_SHA256, "V1 science protocol")
    effective = deepcopy(base)
    effective["schema"] = EFFECTIVE_SCIENCE_SCHEMA
    effective["created_utc"] = overlay["created_utc"]
    effective["status"] = EFFECTIVE_STATUS
    effective["evidence_class"] = EVIDENCE_CLASS
    effective["sequence_disclosure"] = dict(base["sequence_disclosure"])
    effective["sequence_disclosure"].update(
        {
            "v1_attempt_failed_and_consumed_before_v2_definition": True,
            "v2_is_explicit_governance_exception": True,
            "v2_is_not_v1_resume": True,
            "second_adaptive_attempt": True,
            "candidate_performance_observed_from_v1": False,
            "independent_held_or_unbiased_claim_allowed": False,
            "required_v2_wording": overlay["governance_exception"]["required_wording"],
        }
    )
    effective["attempt_budget"] = {
        "full_val24_replays": 1,
        "prior_v1_attempts": 1,
        "claim_required_before_any_validation_npz_cache_label_manifest_or_golden_report_read": True,
        "claim_is_irreversible": True,
        "failure_action": overlay["v2_attempt_budget"]["failure_action"],
    }
    effective["split_access"] = deepcopy(base["split_access"])
    effective["split_access"]["before_claim_allowed"] = list(
        effective["split_access"]["before_claim_allowed"]
    ) + [
        "V2 recovery overlay and exact V1 science/runner",
        "immutable V1 execution CPU receipt runtime receipt claim and failure report",
        "absence check for the V1 H2 candidate cache",
    ]
    effective["train_only_evidence"] = deepcopy(base["train_only_evidence"])
    for name in V1_FAILURE_INPUT_NAMES:
        effective["train_only_evidence"][name] = deepcopy(
            overlay["v1_failure_evidence"][name]
        )
    effective["outputs"] = deepcopy(overlay["outputs"])
    effective["governance_exception"] = deepcopy(overlay["governance_exception"])
    effective["sole_implementation_amendment"] = deepcopy(
        overlay["sole_implementation_amendment"]
    )
    effective["base_v1_science"] = deepcopy(overlay["base_v1_science"])
    effective["base_v1_runner"] = deepcopy(overlay["base_v1_runner"])
    effective["v1_failure_evidence"] = deepcopy(overlay["v1_failure_evidence"])
    effective["recovery_overlay"] = deepcopy(overlay)
    return effective


def _v2_load_json_snapshot(path, expected_sha256=None, description="JSON"):
    payload, digest = _v1_load_json_snapshot(path, expected_sha256, description)
    if Path(path).resolve() == V2_SCIENCE_PATH:
        return _materialize_science(payload), digest
    return payload, digest


def validate_science_protocol(protocol):
    if protocol.get("schema") != EFFECTIVE_SCIENCE_SCHEMA or protocol.get("status") != EFFECTIVE_STATUS:
        raise ValueError("Unexpected effective V2 science identity.")
    overlay = protocol.get("recovery_overlay")
    validate_recovery_overlay(overlay)
    proxy = deepcopy(protocol)
    proxy["status"] = "frozen_before_any_wfull_val24_npz_cache_label_or_golden_report_access"
    _v1_validate_science_protocol(proxy)
    base, _ = _v1_load_json_snapshot(V1_SCIENCE_PATH, V1_SCIENCE_SHA256, "V1 science protocol")
    for name in overlay["inherited_scientific_fields_required_bitwise_equal_to_v1"]:
        if protocol.get(name) != base.get(name):
            raise ValueError("V2 changed inherited scientific field {}.".format(name))
    for name, value in base["train_only_evidence"].items():
        if protocol.get("train_only_evidence", {}).get(name) != value:
            raise ValueError("V2 changed original train evidence {}.".format(name))
    if set(protocol["train_only_evidence"]) != set(base["train_only_evidence"]) | set(V1_FAILURE_INPUT_NAMES):
        raise ValueError("Effective V2 train/recovery evidence key set differs.")
    if protocol.get("governance_exception") != overlay["governance_exception"]:
        raise ValueError("Effective governance exception differs from the overlay.")
    if protocol.get("sole_implementation_amendment") != overlay["sole_implementation_amendment"]:
        raise ValueError("Effective parser amendment differs from the overlay.")
    if protocol.get("outputs") != overlay["outputs"]:
        raise ValueError("Effective V2 output paths differ from the overlay.")
    return protocol


def _validate_v1_failure_chain(science, input_paths, expected):
    overlay = science["recovery_overlay"]
    for name in V1_FAILURE_INPUT_NAMES:
        if core.sha256_file(input_paths[name]) != V1_FAILURE_SHA256[name]:
            raise ValueError("V1 artifact changed: {}".format(name))
    execution, _ = _v1_load_json_snapshot(
        input_paths["v1_execution"], expected["v1_execution"], "V1 execution"
    )
    cpu, _ = _v1_load_json_snapshot(
        input_paths["v1_cpu_receipt"], expected["v1_cpu_receipt"], "V1 CPU receipt"
    )
    runtime, _ = _v1_load_json_snapshot(
        input_paths["v1_runtime_receipt"], expected["v1_runtime_receipt"], "V1 runtime receipt"
    )
    claim, _ = _v1_load_json_snapshot(
        input_paths["v1_claim"], expected["v1_claim"], "V1 claim"
    )
    failure, _ = _v1_load_json_snapshot(
        input_paths["v1_failure_report"], expected["v1_failure_report"], "V1 failure report"
    )
    if (
        execution.get("schema") != "ev-uav-metric-aux-task-arithmetic-wfull-val24-execution-v1"
        or execution.get("attempt_budget") != 1
        or execution.get("science_protocol", {}).get("sha256") != V1_SCIENCE_SHA256
        or execution.get("repository", {}).get("code_sha256", {}).get(
            "evaluate_metric_aux_task_arithmetic_validation.py"
        ) != V1_RUNNER_SHA256
        or execution.get("inputs", {}).get("wfull_checkpoint", {}).get("sha256")
        != core.WFULL_CHECKPOINT_SHA256
    ):
        raise ValueError("V1 execution evidence differs.")
    if (
        cpu.get("schema") != "ev-uav-metric-aux-task-arithmetic-wfull-val24-cpu-preflight-v1"
        or cpu.get("passed") is not True
        or cpu.get("execution_protocol_sha256") != V1_FAILURE_SHA256["v1_execution"]
        or cpu.get("validation_npz_cache_label_manifest_or_golden_report_read") is not False
        or cpu.get("attempt_claimed") is not False
    ):
        raise ValueError("V1 CPU receipt evidence differs.")
    if (
        runtime.get("schema") != "ev-uav-metric-aux-task-arithmetic-wfull-val24-runtime-preflight-v1"
        or runtime.get("passed") is not True
        or runtime.get("authorized_by_root") is not True
        or runtime.get("execution_protocol_sha256") != V1_FAILURE_SHA256["v1_execution"]
        or runtime.get("cpu_preflight_receipt_sha256") != V1_FAILURE_SHA256["v1_cpu_receipt"]
        or runtime.get("validation_npz_cache_label_manifest_or_golden_report_read") is not False
        or runtime.get("smoke", {}).get("api") != "predict_temporal_memory_scores"
        or runtime.get("smoke", {}).get("mode") != "full_stream_t160"
        or runtime.get("smoke", {}).get("t32_called") is not False
        or runtime.get("smoke", {}).get("persistence_called") is not False
    ):
        raise ValueError("V1 runtime receipt evidence differs.")
    if (
        claim.get("schema") != "ev-uav-metric-aux-task-arithmetic-wfull-val24-claim-v1"
        or claim.get("attempt") != 1
        or claim.get("attempt_budget") != 1
        or claim.get("execution_protocol_sha256") != V1_FAILURE_SHA256["v1_execution"]
        or claim.get("cpu_preflight_receipt_sha256") != V1_FAILURE_SHA256["v1_cpu_receipt"]
        or claim.get("runtime_preflight_receipt_sha256") != V1_FAILURE_SHA256["v1_runtime_receipt"]
    ):
        raise ValueError("V1 claim evidence differs.")
    if (
        failure.get("schema") != "ev-uav-metric-aux-task-arithmetic-wfull-val24-report-v1"
        or failure.get("status") != "failed_during_claimed_attempt"
        or failure.get("passed") is not False
        or failure.get("error_type") != "JSONDecodeError"
        or failure.get("error") != "Expecting value: line 2 column 12 (char 13)"
        or failure.get("execution_protocol_sha256") != V1_FAILURE_SHA256["v1_execution"]
        or failure.get("cpu_preflight_receipt_sha256") != V1_FAILURE_SHA256["v1_cpu_receipt"]
        or failure.get("runtime_preflight_receipt_sha256") != V1_FAILURE_SHA256["v1_runtime_receipt"]
        or failure.get("attempt_claim", {}).get("sha256") != V1_FAILURE_SHA256["v1_claim"]
        or failure.get("artifact_observation", {}).get("h2_cache")
        != {"exists": False, "sha256": None}
        or failure.get("failure_action")
        != "archive_without_validation_retuning_threshold_search_route_change_or_second_attempt"
    ):
        raise ValueError("V1 failure report evidence differs.")
    cache_path = core._workspace_path(
        overlay["v1_failure_evidence"]["v1_h2_cache"]["workspace_relative_path"],
        "V1 H2 cache",
    )
    if cache_path.exists():
        raise FileExistsError("V1 H2 cache must remain absent.")
    return {
        "v1_execution_sha256": V1_FAILURE_SHA256["v1_execution"],
        "v1_cpu_receipt_sha256": V1_FAILURE_SHA256["v1_cpu_receipt"],
        "v1_runtime_receipt_sha256": V1_FAILURE_SHA256["v1_runtime_receipt"],
        "v1_claim_sha256": V1_FAILURE_SHA256["v1_claim"],
        "v1_failure_report_sha256": V1_FAILURE_SHA256["v1_failure_report"],
        "v1_h2_cache_absent": True,
        "formal_validation_candidate_inference_completed": False,
        "candidate_performance_observed": False,
        "passed": True,
    }


def _validate_train_only_evidence(science, input_paths, expected):
    result = _v1_validate_train_only_evidence(science, input_paths, expected)
    result["v1_failure_chain"] = _validate_v1_failure_chain(
        science, input_paths, expected
    )
    return result


def _validate_golden_report_after_claim(path, expected_sha):
    payload, digest = _v1_load_json_snapshot(
        path, expected_sha, "golden validation report"
    )
    if not isinstance(payload, dict):
        raise ValueError("Golden report must be a top-level JSON object.")
    raw_counts = payload.get("counts")
    metrics = payload.get("metrics")
    if not isinstance(raw_counts, dict) or not isinstance(metrics, dict):
        raise ValueError("Golden report is missing top-level counts or metrics.")
    normalized_counts = {
        golden_name: raw_counts.get(raw_name)
        for raw_name, golden_name in RAW_TO_GOLDEN_COUNT.items()
    }
    if normalized_counts != core.GOLDEN_COUNTS:
        raise ValueError("Golden report normalized sufficient counts differ.")
    if metrics != core.GOLDEN_METRICS:
        raise ValueError("Golden report top-level metrics differ.")
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest,
        "counts": normalized_counts,
        "metrics": metrics,
        "raw_counts_to_golden_counts": dict(RAW_TO_GOLDEN_COUNT),
        "parser": "top_level_json_load_with_explicit_count_key_mapping_v2",
    }


V2_CODE_PATHS = (
    "evaluate_metric_aux_task_arithmetic_validation_v2.py",
    "protocols/metric_aux_task_arithmetic_wfull_val24_science_v2.json",
    "evaluate_metric_aux_task_arithmetic_validation.py",
    "protocols/metric_aux_task_arithmetic_wfull_val24_science_v1.json",
) + tuple(
    path
    for path in core.CODE_PATHS
    if path
    not in {
        "evaluate_metric_aux_task_arithmetic_validation.py",
        "protocols/metric_aux_task_arithmetic_wfull_val24_science_v1.json",
    }
)

core.SCIENCE_PROTOCOL_PATH = V2_SCIENCE_PATH
core.EXPECTED_SCIENCE_PROTOCOL_SHA256 = V2_SCIENCE_SHA256
core.EXPERIMENT_DIRECTORY = V2_EXPERIMENT_DIRECTORY
core.SCIENCE_SCHEMA = EFFECTIVE_SCIENCE_SCHEMA
core.EXECUTION_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-execution-v2"
core.CPU_RECEIPT_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-cpu-preflight-v2"
core.RUNTIME_RECEIPT_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-runtime-preflight-v2"
core.CLAIM_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-claim-v2"
core.H2_CACHE_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-h2-cache-v2"
core.REPORT_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-report-v2"
core.CODE_PATHS = V2_CODE_PATHS
core.TRAIN_INPUT_NAMES = tuple(core.TRAIN_INPUT_NAMES) + V1_FAILURE_INPUT_NAMES
core._load_json_snapshot = _v2_load_json_snapshot
core.validate_science_protocol = validate_science_protocol
core._validate_train_only_evidence = _validate_train_only_evidence
core._validate_golden_report_after_claim = _validate_golden_report_after_claim


sha256_file = core.sha256_file
canonical_sha256 = core.canonical_sha256
route_policy_sha256 = core.route_policy_sha256
classify_wfull_route = core.classify_wfull_route
choose_candidate_scores = core.choose_candidate_scores
promotion_gate_results = core.promotion_gate_results
build_execution_protocol = core.build_execution_protocol
validate_execution_protocol = core.validate_execution_protocol
freeze_execution_protocol = core.freeze_execution_protocol
preflight_execution = core.preflight_execution
runtime_preflight_execution = core.runtime_preflight_execution
run_execution = core.run_execution
parse_args = core.parse_args
_paths = core._paths
_code_sha256 = core._code_sha256
_load_published_validation_contract = core._load_published_validation_contract
_canonical_inputs = core._canonical_inputs
_expected_input_sha256 = core._expected_input_sha256
_strict_cpu_wfull_load = core._strict_cpu_wfull_load
_synthetic_route_preflight = core._synthetic_route_preflight
_effective_c00_sha256 = core._effective_c00_sha256


def load_effective_science_protocol():
    overlay, digest = _load_raw_overlay()
    effective = _materialize_science(overlay)
    validate_science_protocol(effective)
    return effective, digest


def main(argv=None):
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

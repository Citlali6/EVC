"""Evaluation-only V4 recovery for the frozen metric-aux H2 grouped OOF study.

V4 reuses the immutable, passed V3 probe and all six completed V3 formal
training runs.  It changes only the import route for the established
``temporal_frame_video_from_sample`` full-stream constructor.  V4 exposes no
probe or training command and writes evaluations/report under a new root.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
RUNNER_PATH = Path(__file__).resolve()

# Import V3 privately, while restoring all private slots that V3/V2 use.  The
# frozen function objects are retained, but public V1/V2/V3 modules are never
# replaced or modified.
_V1_PRIVATE_NAME = "_metric_aux_h2_grouped_oof_v1_core_for_v2"
_V2_PRIVATE_NAME = "_metric_aux_h2_grouped_oof_v2_for_v3"
_V3_PRIVATE_NAME = "_metric_aux_h2_grouped_oof_v3_for_v4"
_PREVIOUS_PRIVATE_MODULES = {
    name: sys.modules.get(name)
    for name in (_V1_PRIVATE_NAME, _V2_PRIVATE_NAME, _V3_PRIVATE_NAME)
}
_V3_PATH = EVC_ROOT / "run_metric_aux_h2_grouped_oof_v3.py"
_V3_SPEC = importlib.util.spec_from_file_location(_V3_PRIVATE_NAME, _V3_PATH)
if _V3_SPEC is None or _V3_SPEC.loader is None:
    raise ImportError("Unable to create a private V3 module for V4 recovery.")
v3 = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_PRIVATE_NAME] = v3
try:
    _V3_SPEC.loader.exec_module(v3)
finally:
    for _name, _previous in _PREVIOUS_PRIVATE_MODULES.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

core = v3.core

PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_h2_grouped_oof_science_v4.json"
EXPECTED_PROTOCOL_SHA256 = "589e7d35075e31ad5d85b946ce444adeb88b9ca35cb7c8772b385bc61cfc96b5"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-h2-grouped-oof-evaluation-recovery-v4"
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260810_metric_aux_h2_grouped_oof_v4"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"

V3_PROTOCOL_SHA256 = "a4039cdba26ed1f950d62b40edc4b13c9868ac281c0dcd6b5a37e2062cd79875"
V3_RUNNER_SHA256 = "968d2fa0bc32756c5f8b059d44d0b9413daf75396c7d55733a68d126d2148236"
V3_TEST_SHA256 = "379ab2dcdeb1a7c291b2cf2ef623492cf024f6df19a1c7b22f6e807929ed278d"
V3_COMMAND_AUDIT_SHA256 = "a369096bf1b716b40284e7da7d9ec9397ca82e6ecd783ec44b47404c0749c2ea"
V3_PROBE_SHA256 = "09e05f582b439525fb7d04006766258173c502af23b9b1acb6fe5846e95e2b06"
V3_PAIR_AUDIT_SHA256 = "fcd4f877597ac1176d7215bcfd2ee82533f53fadfc2ac6558fc8563ae3ad2c9d"
V3_EVALUATION_FAILURE_SHA256 = "03c9061bd0313355411d06f1f6027faf9777b4ee14906d6f0d32d5848735ce82"
BASE_EVALUATE_OWNER_SHA256 = "9b3f58a8ed4676e78bf75b3daaf33e96b7218f752088e4e7726da7dbeaa3cf5a"
MEMORY_INFERENCE_SHA256 = "8b388094b1cf4504db5f823a5de6cc4d41793f69b1cf8822e16bb10eb59a9610"
TEMPORAL_FRAME_INFERENCE_SHA256 = "2203dd11d69041352ddc5ccb9bb7148c7dbcda4803554fcd2cc120345a05e479"

_V3_LOAD_PROTOCOL = v3.load_protocol
_BASE_EVALUATE_SPEC = core.evaluate_spec
_BASE_LOAD_EVALUATION_RESULT = core.load_evaluation_result
_BASE_BUILD_REPORT = core.build_report
_BASE_VERIFY_ASSETS = core.verify_assets
_BASE_WRITE_NEW_JSON = core.write_new_json


def _expect(value, expected, label):
    core._expect_equal(value, expected, label)


def _require_bound_file(record, expected_sha256, label):
    path = core.workspace_path(record["workspace_relative_path"])
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    actual = core.sha256_file(path)
    if actual != expected_sha256 or actual != record["sha256"]:
        raise RuntimeError("{} SHA-256 differs from the V4 contract.".format(label))
    return path


def _load_bound_json(record, expected_sha256, label):
    path = _require_bound_file(record, expected_sha256, label)
    payload, digest = core.load_json_snapshot(path)
    _expect(digest, expected_sha256, "{} snapshot".format(label))
    return payload, path


def _expected_formal_run_ids():
    return {
        "hold_g1_baseline",
        "hold_g1_metric_aux",
        "hold_g2_baseline",
        "hold_g2_metric_aux",
        "hold_g3_baseline",
        "hold_g3_metric_aux",
    }


def _validate_v3_failure(failure):
    expected = {
        "schema": "ev-uav-metric-aux-held-train-evaluation-failure-v3",
        "status": "failed",
        "passed": False,
        "attempted_eval_id": "hold_g1_released_m20",
        "failure_stage": "evaluate_spec_imports_before_gpu_preflight_result_path_model_or_source_loop",
        "formal_training_completed": True,
        "formal_training_reuse_required": True,
        "formal_retraining_allowed": False,
        "validation_or_test_read": False,
    }
    for key, value in expected.items():
        _expect(failure.get(key), value, "V3 evaluation failure {}".format(key))
    disk = failure["disk_and_control_flow_evidence"]
    exact_zero = (
        "held_source_load_count",
        "model_load_count",
        "full_stream_prediction_call_count",
        "score_tensor_count",
        "postprocess_call_count",
        "sufficient_count_call_count",
    )
    for key in exact_zero:
        _expect(disk.get(key), 0, "V3 failure {}".format(key))
    _expect(disk.get("held_train_evaluation_directory_existed_after_failure"), False, "V3 held root absent")
    _expect(disk.get("evaluation_json_existed_after_failure"), False, "V3 evaluation absent")
    policy = failure["recovery_policy"]
    _expect(policy.get("v3_evaluation_attempt_remains_failed"), True, "V3 attempt remains failed")
    _expect(policy.get("new_probe_or_training_forbidden"), True, "V4 no probe/training")


def _validate_v3_probe(probe):
    _expect(probe.get("schema"), "ev-uav-metric-aux-audit-only-probe-result-v3", "V3 probe schema")
    _expect(probe.get("passed"), True, "V3 probe passed")
    if not probe.get("checks") or not all(probe["checks"].values()):
        raise RuntimeError("V3 probe does not retain every passed gate.")
    _expect(probe.get("protocol_sha256"), V3_PROTOCOL_SHA256, "V3 probe protocol")
    _expect(probe.get("runner_sha256"), V3_RUNNER_SHA256, "V3 probe runner")
    _expect(probe.get("command_audit_sha256"), V3_COMMAND_AUDIT_SHA256, "V3 probe command audit")
    _expect(probe.get("new_pair_training_optimizer_steps"), 0, "V3 probe new training")


def _validate_v3_pair_audit(pair):
    _expect(pair.get("schema"), "ev-uav-metric-aux-formal-pair-audit-v1", "V3 pair schema")
    _expect(pair.get("passed"), True, "V3 pair passed")
    _expect(pair.get("protocol_sha256"), V3_PROTOCOL_SHA256, "V3 pair protocol")
    _expect(pair.get("runner_sha256"), V3_RUNNER_SHA256, "V3 pair runner")
    _expect(pair.get("claim_scope"), "incremental_finetune_transfer_not_fold_clean_model_generalization", "V3 pair claim scope")
    _expect(pair.get("shared_parent_pretraining_exposure"), True, "V3 pair shared parent")
    checks = pair.get("checks", {})
    if set(checks) != {"all_pairs_passed", "all_three_pairs_present"} or not all(checks.values()):
        raise RuntimeError("V3 formal pair top-level gates are incomplete.")
    folds = pair.get("fold_records", [])
    if [item.get("fold_id") for item in folds] != ["hold_g1", "hold_g2", "hold_g3"]:
        raise RuntimeError("V3 pair fold order differs from the frozen definition.")
    if not all(item.get("passed") is True and all(item.get("checks", {}).values()) for item in folds):
        raise RuntimeError("A V3 formal pair nested gate is not passed.")


def _load_v3_formal_runs(overlay, pair):
    evidence = overlay["v3_formal_training_evidence"]
    if set(evidence) != _expected_formal_run_ids():
        raise RuntimeError("V3 formal evidence does not contain exactly six run IDs.")
    pair_by_fold = {item["fold_id"]: item for item in pair["fold_records"]}
    results = {}
    for run_id in sorted(evidence):
        record = evidence[run_id]
        result, _ = _load_bound_json(
            record["training_result"], record["training_result"]["sha256"], "{} result".format(run_id)
        )
        if run_id.endswith("_metric_aux"):
            fold_id, variant = run_id[: -len("_metric_aux")], "metric_aux"
        elif run_id.endswith("_baseline"):
            fold_id, variant = run_id[: -len("_baseline")], "baseline"
        else:
            raise RuntimeError("Unexpected V3 formal run ID: {}".format(run_id))
        required = {
            "schema": "ev-uav-metric-aux-training-result-v1",
            "status": "completed",
            "run_id": run_id,
            "fold_id": fold_id,
            "variant": variant,
            "protocol_sha256": V3_PROTOCOL_SHA256,
            "runner_sha256": V3_RUNNER_SHA256,
            "expected_optimizer_steps": int(record["expected_optimizer_steps"]),
            "claim_scope": "incremental_finetune_transfer_not_fold_clean_model_generalization",
        }
        for key, value in required.items():
            _expect(result.get(key), value, "{} {}".format(run_id, key))
        _expect(result.get("core_sha256_before_after_equal"), True, "{} core identity".format(run_id))
        if result.get("input_source_sha256_before_after_equal") is not True:
            raise RuntimeError("{} source before/after identity failed.".format(run_id))
        e3 = result.get("checkpoints", {}).get("e3", {})
        if e3.get("passed") is not True or e3.get("epoch") != 2:
            raise RuntimeError("{} E3 scope audit is incomplete.".format(run_id))
        checkpoint_path = _require_bound_file(
            record["e3_checkpoint"], record["e3_checkpoint"]["sha256"], "{} E3".format(run_id)
        )
        _expect(Path(e3.get("path", "")).resolve(), checkpoint_path.resolve(), "{} E3 path".format(run_id))
        _expect(e3.get("sha256"), record["e3_checkpoint"]["sha256"], "{} E3 result binding".format(run_id))
        _expect(e3.get("name_shape_audit", {}).get("passed"), True, "{} E3 name/shape".format(run_id))
        pair_record = pair_by_fold[fold_id]
        pair_key = "baseline_result_sha256" if variant == "baseline" else "candidate_result_sha256"
        _expect(pair_record.get(pair_key), record["training_result"]["sha256"], "{} pair result".format(run_id))
        results[run_id] = result
    return results


def _validate_overlay(overlay, overlay_sha256):
    _expect(overlay.get("schema"), EXPECTED_SCHEMA, "V4 protocol schema")
    _expect(
        overlay.get("status"),
        "frozen_after_v3_probe_and_six_formal_training_before_any_v4_held_evaluation",
        "V4 protocol status",
    )
    _expect(overlay_sha256, EXPECTED_PROTOCOL_SHA256, "V4 protocol SHA-256")
    inheritance = overlay["inheritance"]
    _expect(
        inheritance.get("entire_v3_scientific_numeric_training_evaluation_and_promotion_definition_inherited"),
        True,
        "V3 exact scientific inheritance",
    )
    _require_bound_file(inheritance["v3_protocol"], V3_PROTOCOL_SHA256, "V3 protocol")
    _require_bound_file(inheritance["v3_runner"], V3_RUNNER_SHA256, "V3 runner")
    _require_bound_file(inheritance["v3_tests"], V3_TEST_SHA256, "V3 tests")
    command_audit, _ = _load_bound_json(
        inheritance["v3_command_audit"], V3_COMMAND_AUDIT_SHA256, "V3 command audit"
    )
    _expect(command_audit.get("schema"), "ev-uav-metric-aux-h2-grouped-oof-command-audit-v3", "V3 command schema")
    _expect(command_audit.get("protocol_sha256"), V3_PROTOCOL_SHA256, "V3 command protocol")
    _expect(command_audit.get("runner_sha256"), V3_RUNNER_SHA256, "V3 command runner")
    _expect(command_audit.get("gpu_or_cuda_initialized"), False, "V3 command CPU-only")
    probe, _ = _load_bound_json(
        inheritance["v3_resource_probe_receipt"], V3_PROBE_SHA256, "V3 resource probe"
    )
    _validate_v3_probe(probe)
    pair, _ = _load_bound_json(
        inheritance["v3_formal_pair_audit"], V3_PAIR_AUDIT_SHA256, "V3 formal pair audit"
    )
    _validate_v3_pair_audit(pair)
    failure, _ = _load_bound_json(
        inheritance["v3_evaluation_failure_receipt"], V3_EVALUATION_FAILURE_SHA256, "V3 evaluation failure"
    )
    _validate_v3_failure(failure)
    formal_results = _load_v3_formal_runs(overlay, pair)

    recovery = overlay["recovery_amendment"]
    expected_recovery = {
        "v3_evaluation_attempt_remains_failed": True,
        "retroactive_v3_evaluation_pass_forbidden": True,
        "reuse_v3_probe_required": True,
        "reuse_all_six_v3_formal_training_runs_required": True,
        "new_probe_allowed": False,
        "new_training_allowed": False,
        "new_training_optimizer_steps": 0,
        "scientific_candidate_training_evaluation_settings_folds_or_promotion_change": False,
        "changed_contract_only": "evaluation import routing for temporal_frame_video_from_sample",
        "claim_scope": "incremental_finetune_transfer_not_fold_clean_model_generalization",
        "shared_parent_pretraining_exposure": True,
    }
    for key, value in expected_recovery.items():
        _expect(recovery.get(key), value, "V4 recovery {}".format(key))
    api = overlay["evaluation_api_contract"]
    _require_bound_file(api["frozen_evaluate_owner"], BASE_EVALUATE_OWNER_SHA256, "frozen evaluate owner")
    _require_bound_file(api["memory_inference_module"], MEMORY_INFERENCE_SHA256, "memory inference")
    _require_bound_file(api["temporal_frame_inference_module"], TEMPORAL_FRAME_INFERENCE_SHA256, "frame inference")
    _expect(api.get("full_stream_semantics_unchanged"), True, "full-stream semantics")
    _expect(api.get("predictor_api_unchanged"), True, "predictor API")
    _expect(api.get("postprocess_and_sufficient_count_api_unchanged"), True, "metric APIs")
    cli = overlay["cli_contract"]
    _expect(cli.get("allowed_commands"), ["audit", "evaluate", "report", "all-evaluate-report"], "V4 CLI")
    for key in (
        "train_command_exposed",
        "probe_command_exposed",
        "audit_training_command_exposed",
        "all_after_probe_command_exposed",
    ):
        _expect(cli.get(key), False, "V4 {}".format(key))
    _expect(overlay["output_contract"]["workspace_relative_directory"], "experiments/20260810_metric_aux_h2_grouped_oof_v4", "V4 output root")
    _expect(overlay.get("validation_or_test_read_allowed"), False, "V4 split policy")
    _expect(overlay.get("t32_allowed"), False, "V4 T32 policy")
    _expect(overlay.get("prior_persistence_formal_artifact_read_allowed"), False, "V4 persistence policy")
    return {
        "v3_command_audit": command_audit,
        "v3_probe": probe,
        "v3_pair_audit": pair,
        "v3_failure": failure,
        "formal_results": formal_results,
    }


def _validate_formal_sources_against_effective(protocol, results):
    for fold in protocol["dataset"]["folds"]:
        held = {item["name"] for item in core.held_items(protocol, fold)}
        fit = [item["name"] for item in core.fit_items(protocol, fold)]
        for variant in ("baseline", "metric_aux"):
            result = results["{}_{}".format(fold["fold_id"], variant)]
            _expect(result["expected_source_names"], fit, "{} {} fit order".format(fold["fold_id"], variant))
            if held.intersection(result["expected_source_names"]):
                raise RuntimeError("Held sources leaked into a V3 formal fit result.")


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V4 protocol SHA-256 {} differs from frozen {}.".format(actual, EXPECTED_PROTOCOL_SHA256))
    overlay, snapshot_sha = core.load_json_snapshot(PROTOCOL_PATH)
    _expect(snapshot_sha, actual, "V4 protocol snapshot")
    evidence = _validate_overlay(overlay, actual)
    effective, inherited_sha = _V3_LOAD_PROTOCOL()
    _expect(inherited_sha, V3_PROTOCOL_SHA256, "inherited V3 protocol")
    effective = copy.deepcopy(effective)
    _validate_formal_sources_against_effective(effective, evidence["formal_results"])
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["recovery_amendment_v4"] = copy.deepcopy(overlay["recovery_amendment"])
    effective["v4_inheritance"] = copy.deepcopy(overlay["inheritance"])
    effective["evaluation_api_contract_v4"] = copy.deepcopy(overlay["evaluation_api_contract"])
    effective["v3_formal_training_evidence_v4"] = copy.deepcopy(overlay["v3_formal_training_evidence"])
    effective["evaluation_and_report_contract_v4"] = copy.deepcopy(overlay["evaluation_and_report_contract"])
    effective["cli_contract_v4"] = copy.deepcopy(overlay["cli_contract"])
    effective["revision_history"] = list(effective["revision_history"]) + [
        {
            "recovery_protocol_sha256": actual,
            "reason": overlay["recovery_amendment"]["reason"],
            "v3_evaluation_failure_sha256": V3_EVALUATION_FAILURE_SHA256,
            "v3_attempt_remains_failed": True,
            "new_probe_or_training_optimizer_steps": 0,
        }
    ]
    effective["outputs"]["workspace_relative_directory"] = overlay["output_contract"][
        "workspace_relative_directory"
    ]
    return effective, actual


def _load_evidence_for_effective(protocol):
    overlay, digest = core.load_json_snapshot(PROTOCOL_PATH)
    _expect(digest, EXPECTED_PROTOCOL_SHA256, "V4 evidence protocol")
    evidence = _validate_overlay(overlay, digest)
    _validate_formal_sources_against_effective(protocol, evidence["formal_results"])
    return evidence


def require_v3_prerequisites(protocol=None):
    if protocol is None:
        protocol, _ = load_protocol()
    evidence = _load_evidence_for_effective(protocol)
    return evidence, V3_PAIR_AUDIT_SHA256


def evaluation_specs(protocol, ignored_training_specs=None):
    evidence = protocol["v3_formal_training_evidence_v4"]
    output = []
    for fold in protocol["dataset"]["folds"]:
        held_names = [item["name"] for item in core.held_items(protocol, fold)]
        for variant in ("released_m20", "baseline", "metric_aux"):
            eval_id = "{}_{}".format(fold["fold_id"], variant)
            if variant == "released_m20":
                checkpoint = core.workspace_path(protocol["parent_checkpoint"]["workspace_relative_path"])
                checkpoint_sha = protocol["parent_checkpoint"]["sha256"]
                training_result_path = None
            else:
                record = evidence["{}_{}".format(fold["fold_id"], variant)]
                checkpoint = core.workspace_path(record["e3_checkpoint"]["workspace_relative_path"])
                checkpoint_sha = record["e3_checkpoint"]["sha256"]
                training_result_path = str(
                    core.workspace_path(record["training_result"]["workspace_relative_path"])
                )
            output.append(
                {
                    "eval_id": eval_id,
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "held_group": fold["held_group"],
                    "held_source_names": held_names,
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "training_result_path": training_result_path,
                    "result_path": str((EVALUATION_ROOT / eval_id / "evaluation.json").resolve()),
                }
            )
    if len(output) != 9 or len({item["eval_id"] for item in output}) != 9:
        raise RuntimeError("V4 evaluation plan is not exactly nine unique specifications.")
    return output


def _verify_api_surface(protocol):
    import utils.temporal_frame_inference as frame_inference
    import utils.temporal_memory_inference as memory_inference

    api = protocol["evaluation_api_contract_v4"]
    memory_path = Path(memory_inference.__file__).resolve()
    frame_path = Path(frame_inference.__file__).resolve()
    _expect(
        memory_path,
        core.workspace_path(api["memory_inference_module"]["workspace_relative_path"]).resolve(),
        "runtime memory inference path",
    )
    _expect(
        frame_path,
        core.workspace_path(api["temporal_frame_inference_module"]["workspace_relative_path"]).resolve(),
        "runtime frame inference path",
    )
    _expect(core.sha256_file(memory_path), MEMORY_INFERENCE_SHA256, "runtime memory inference")
    _expect(core.sha256_file(frame_path), TEMPORAL_FRAME_INFERENCE_SHA256, "runtime frame inference")
    if hasattr(memory_inference, "temporal_frame_video_from_sample"):
        raise RuntimeError("Memory inference unexpectedly already exposes the injected helper.")
    helper = getattr(frame_inference, api["temporal_frame_inference_module"]["required_symbol"], None)
    if not callable(helper):
        raise RuntimeError("The frozen temporal-frame full-stream helper is unavailable.")
    positional = [
        parameter.name
        for parameter in inspect.signature(helper).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    _expect(
        positional,
        api["temporal_frame_inference_module"]["required_signature_positional_parameters"],
        "temporal-frame helper signature",
    )
    return memory_inference, helper


def _evaluation_recovery_record():
    return {
        "mode": "evaluation_only_import_route_recovery_v4",
        "v3_evaluation_failure_sha256": V3_EVALUATION_FAILURE_SHA256,
        "v3_probe_sha256": V3_PROBE_SHA256,
        "v3_formal_pair_audit_sha256": V3_PAIR_AUDIT_SHA256,
        "base_evaluate_owner_sha256": BASE_EVALUATE_OWNER_SHA256,
        "memory_inference_sha256": MEMORY_INFERENCE_SHA256,
        "temporal_frame_inference_sha256": TEMPORAL_FRAME_INFERENCE_SHA256,
        "temporary_injection_restored_in_finally": True,
        "new_probe_or_training_optimizer_steps": 0,
        "scientific_evaluation_change": False,
    }


def evaluate_spec(protocol, spec):
    memory_inference, helper = _verify_api_surface(protocol)
    expected_runner_sha = core.sha256_file(RUNNER_PATH)
    original_writer = core.write_new_json

    def guarded_writer(path, payload):
        _expect(Path(path).resolve(), Path(spec["result_path"]).resolve(), "V4 evaluation output path")
        _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "V4 evaluation protocol")
        _expect(payload.get("runner_sha256"), expected_runner_sha, "V4 evaluation runner")
        _expect(payload.get("eval_id"), spec["eval_id"], "V4 evaluation ID")
        _expect(payload.get("checkpoint_sha256"), spec["checkpoint_sha256"], "V4 evaluation checkpoint")
        _expect(payload.get("t32_read_or_combined"), False, "V4 T32 exclusion")
        payload["evaluation_recovery_v4"] = _evaluation_recovery_record()
        return original_writer(path, payload)

    if hasattr(memory_inference, "temporal_frame_video_from_sample"):
        raise RuntimeError("V4 helper injection target was not clean.")
    setattr(memory_inference, "temporal_frame_video_from_sample", helper)
    core.write_new_json = guarded_writer
    try:
        payload = _BASE_EVALUATE_SPEC(protocol, spec)
    finally:
        core.write_new_json = original_writer
        if hasattr(memory_inference, "temporal_frame_video_from_sample"):
            delattr(memory_inference, "temporal_frame_video_from_sample")
    if hasattr(memory_inference, "temporal_frame_video_from_sample"):
        raise RuntimeError("V4 helper injection survived the finally boundary.")
    _expect(payload.get("evaluation_recovery_v4"), _evaluation_recovery_record(), "V4 evaluation recovery record")
    return payload


def load_evaluation_result(spec):
    payload, digest = _BASE_LOAD_EVALUATION_RESULT(spec)
    _expect(payload.get("fold_id"), spec["fold_id"], "V4 evaluation fold")
    _expect(payload.get("variant"), spec["variant"], "V4 evaluation variant")
    _expect(payload.get("checkpoint_sha256"), spec["checkpoint_sha256"], "V4 evaluation checkpoint")
    _expect(payload.get("evaluation_recovery_v4"), _evaluation_recovery_record(), "V4 recovery provenance")
    return payload, digest


def command_audit_payload(protocol, protocol_sha256, assets):
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the V4 CPU audit.")
    evidence, _ = require_v3_prerequisites(protocol)
    specs = evaluation_specs(protocol)
    commands = {
        spec["eval_id"]: [
            sys.executable,
            str(RUNNER_PATH),
            "evaluate",
            "--eval-id",
            spec["eval_id"],
            GPU_AUTHORIZATION_FLAG,
        ]
        for spec in specs
    }
    api_memory, _ = _verify_api_surface(protocol)
    if hasattr(api_memory, "temporal_frame_video_from_sample"):
        raise RuntimeError("V4 CPU audit left a helper injection behind.")
    formal = {
        run_id: {
            "training_result_sha256": protocol["v3_formal_training_evidence_v4"][run_id]["training_result"]["sha256"],
            "e3_checkpoint_sha256": protocol["v3_formal_training_evidence_v4"][run_id]["e3_checkpoint"]["sha256"],
            "expected_optimizer_steps": int(protocol["v3_formal_training_evidence_v4"][run_id]["expected_optimizer_steps"]),
        }
        for run_id in sorted(evidence["formal_results"])
    }
    return {
        "schema": "ev-uav-metric-aux-h2-grouped-oof-eval-only-command-audit-v4",
        "created_utc": core.utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256,
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "assets": assets,
        "v3_evidence": {
            "command_audit_sha256": V3_COMMAND_AUDIT_SHA256,
            "probe_sha256": V3_PROBE_SHA256,
            "formal_pair_audit_sha256": V3_PAIR_AUDIT_SHA256,
            "evaluation_failure_sha256": V3_EVALUATION_FAILURE_SHA256,
            "formal_runs": formal,
        },
        "evaluation_specs": specs,
        "evaluation_commands": commands,
        "allowed_cli_commands": ["audit", "evaluate", "report", "all-evaluate-report"],
        "forbidden_cli_commands": ["probe", "train", "audit-training", "all-after-probe"],
        "new_probe_optimizer_steps": 0,
        "new_training_optimizer_steps": 0,
        "evaluation_count": len(specs),
        "gpu_or_cuda_initialized": False,
        "data_use_statement": (
            "Only frozen train identities, V3 probe/formal evidence and checkpoint file identities were read. "
            "No validation/test path, T32 cache or persistence-formal artifact was opened; CUDA remained uninitialized."
        ),
    }


def run_audit():
    protocol, protocol_sha = load_protocol()
    assets = _BASE_VERIFY_ASSETS(protocol)
    payload = command_audit_payload(protocol, protocol_sha, assets)
    _expect(payload["evaluation_count"], 9, "V4 audit evaluation count")
    _expect(payload["new_probe_optimizer_steps"], 0, "V4 audit probe steps")
    _expect(payload["new_training_optimizer_steps"], 0, "V4 audit training steps")
    _BASE_WRITE_NEW_JSON(COMMAND_AUDIT_PATH, payload)
    print("V4 command audit:", COMMAND_AUDIT_PATH)
    print("V4 command audit sha256:", core.sha256_file(COMMAND_AUDIT_PATH))
    print("GPU not initialized; waiting for explicit root authorization.")
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the immutable V4 CPU audit before any V4 evaluation.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-h2-grouped-oof-eval-only-command-audit-v4", "V4 command schema")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "V4 command protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "V4 command runner")
    _expect(payload.get("gpu_or_cuda_initialized"), False, "V4 command CUDA")
    _expect(payload.get("new_probe_optimizer_steps"), 0, "V4 command probe steps")
    _expect(payload.get("new_training_optimizer_steps"), 0, "V4 command training steps")
    _expect(payload.get("evaluation_count"), 9, "V4 command evaluation count")
    _expect(payload.get("allowed_cli_commands"), ["audit", "evaluate", "report", "all-evaluate-report"], "V4 command CLI")
    expected_v3 = {
        "command_audit_sha256": V3_COMMAND_AUDIT_SHA256,
        "probe_sha256": V3_PROBE_SHA256,
        "formal_pair_audit_sha256": V3_PAIR_AUDIT_SHA256,
        "evaluation_failure_sha256": V3_EVALUATION_FAILURE_SHA256,
    }
    for key, value in expected_v3.items():
        _expect(payload.get("v3_evidence", {}).get(key), value, "V4 command {}".format(key))
    if set(payload.get("evaluation_commands", {})) != {
        "{}_{}".format(fold, variant)
        for fold in ("hold_g1", "hold_g2", "hold_g3")
        for variant in ("released_m20", "baseline", "metric_aux")
    }:
        raise RuntimeError("V4 command audit does not contain exactly nine evaluations.")
    return payload, digest


def require_v3_formal_pair_audit():
    protocol, _ = load_protocol()
    evidence, _ = require_v3_prerequisites(protocol)
    return evidence["v3_pair_audit"], V3_PAIR_AUDIT_SHA256


def run_evaluation(eval_id=None, authorized=False):
    core.require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    load_command_audit()
    require_v3_prerequisites(protocol)
    specs = evaluation_specs(protocol)
    if eval_id is not None:
        matches = [spec for spec in specs if spec["eval_id"] == eval_id]
        if len(matches) != 1:
            raise KeyError("Unknown V4 evaluation id: {}".format(eval_id))
        return [evaluate_spec(protocol, matches[0])]
    results = []
    for spec in specs:
        if Path(spec["result_path"]).is_file():
            result, _ = load_evaluation_result(spec)
            print("retaining completed V4 held-train evaluation:", spec["eval_id"])
            results.append(result)
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
        result, _ = load_evaluation_result(spec)
        results.append(result)
    return results


def run_report():
    protocol, protocol_sha = load_protocol()
    load_command_audit()
    require_v3_prerequisites(protocol)
    specs = evaluation_specs(protocol)
    original_writer = core.write_new_json

    def guarded_writer(path, payload):
        _expect(Path(path).resolve(), REPORT_PATH.resolve(), "V4 report output path")
        _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "V4 report protocol")
        _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "V4 report runner")
        _expect(payload.get("formal_pair_audit_sha256"), V3_PAIR_AUDIT_SHA256, "V4 report pair audit")
        payload["evaluation_recovery_v4"] = _evaluation_recovery_record()
        return original_writer(path, payload)

    core.write_new_json = guarded_writer
    try:
        payload = _BASE_BUILD_REPORT(protocol, protocol_sha, specs)
    finally:
        core.write_new_json = original_writer
    _expect(payload.get("evaluation_recovery_v4"), _evaluation_recovery_record(), "V4 report recovery")
    print("V4 grouped OOF report:", REPORT_PATH)
    print("promotion passed:", payload["passed"])
    return payload


def _patch_core_for_v4():
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.EXPECTED_PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.COMMAND_AUDIT_PATH = COMMAND_AUDIT_PATH
    core.PROBE_RESULT_PATH = core.workspace_path(
        "experiments/20260810_metric_aux_h2_grouped_oof_v3/resource_probe/runtime_result.json"
    )
    core.FORMAL_ROOT = core.workspace_path("experiments/20260810_metric_aux_h2_grouped_oof_v3/formal_training")
    core.PAIR_AUDIT_PATH = core.workspace_path(
        "experiments/20260810_metric_aux_h2_grouped_oof_v3/formal_training/pair_audit.json"
    )
    core.EVALUATION_ROOT = EVALUATION_ROOT
    core.REPORT_PATH = REPORT_PATH
    core.__file__ = str(RUNNER_PATH)
    core.load_protocol = load_protocol
    core.load_command_audit = load_command_audit
    core.require_probe_passed = lambda: require_v3_prerequisites()[0]["v3_probe"]
    core.require_formal_pair_audit = require_v3_formal_pair_audit
    core.evaluation_specs = evaluation_specs
    core.evaluate_spec = evaluate_spec
    core.load_evaluation_result = load_evaluation_result


_patch_core_for_v4()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Run the immutable CPU-only V4 evaluation recovery audit.")
    evaluate = subparsers.add_parser("evaluate", help="Run all or one V4 held-train evaluation.")
    evaluate.add_argument("--eval-id", default=None)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("report", help="Apply the unchanged inherited double-anchor gates.")
    all_eval = subparsers.add_parser("all-evaluate-report")
    all_eval.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit()
    if args.command == "evaluate":
        return run_evaluation(args.eval_id, args.authorized)
    if args.command == "report":
        return run_report()
    if args.command == "all-evaluate-report":
        core.require_gpu_authorization(args.authorized)
        run_evaluation(authorized=True)
        return run_report()
    raise RuntimeError("Unsupported V4 command: {}".format(args.command))


if __name__ == "__main__":
    main()

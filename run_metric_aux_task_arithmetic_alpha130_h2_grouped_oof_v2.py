"""Mechanical V2 recovery for the alpha=1.30 grouped-OOF CPU publisher.

V1 failed before its exclusive hard-link publish because this Windows runtime
rejects ``fsync`` on a read-only Python file handle.  V2 inherits the entire
frozen V1 science definition and changes only that handle from ``rb`` to
``rb+``.  V1 artifacts are retained and never overwritten.
"""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = EVC_ROOT / "tests" / "test_run_metric_aux_task_arithmetic_alpha130_h2_grouped_oof_v2.py"
PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_task_arithmetic_alpha130_h2_grouped_oof_science_v2.json"
EXPECTED_PROTOCOL_SHA256 = "7d2ffc6418c0e937b1b05fe20d2f5dc7b4fd5dfae2b6a363304e0833b8a231c5"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-task-arithmetic-alpha130-h2-grouped-oof-cpu-publish-recovery-v2"
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260811_metric_aux_task_arithmetic_alpha130_h2_grouped_oof_v2"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
SYNTHESIS_ROOT = OUTPUT_ROOT / "synthesis"
SYNTHESIS_MANIFEST_PATH = SYNTHESIS_ROOT / "synthesis_manifest.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"

V1_PROTOCOL_SHA256 = "50a20e36f08c6683f816458ba41b8357678e51c01913aa77484d9fa035df5406"
V1_RUNNER_SHA256 = "f9fe4e1572d1580b80299ae6c3c8387aee4d66997026a67e099273cb37656924"
V1_TEST_SHA256 = "06da230bbe1a78f8c9770aa603250fe91d64803efb99ecf7f18e84ee333a752c"
V1_COMMAND_AUDIT_SHA256 = "5b35e555352b084a156876c2dac650d246c0247deceb93903060a5bf9502aa1a"
V1_OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260811_metric_aux_task_arithmetic_alpha130_h2_grouped_oof_v1"

_PRIVATE_NAME = "_metric_aux_task_arithmetic_alpha130_v1_for_cpu_publish_recovery_v2"
_PREVIOUS_PRIVATE = sys.modules.get(_PRIVATE_NAME)
_V1_PATH = EVC_ROOT / "run_metric_aux_task_arithmetic_alpha130_h2_grouped_oof.py"
_SPEC = importlib.util.spec_from_file_location(_PRIVATE_NAME, _V1_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Unable to create a private alpha130 V1 module.")
v1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_PRIVATE_NAME] = v1
try:
    _SPEC.loader.exec_module(v1)
finally:
    if _PREVIOUS_PRIVATE is None:
        sys.modules.pop(_PRIVATE_NAME, None)
    else:
        sys.modules[_PRIVATE_NAME] = _PREVIOUS_PRIVATE

core = v1.core
ALPHA = v1.ALPHA
GPU_AUTHORIZATION_FLAG = v1.GPU_AUTHORIZATION_FLAG

_V1_LOAD_PROTOCOL = v1.load_protocol
_V1_COMMAND_AUDIT_PAYLOAD = v1.command_audit_payload
_V1_PROTOCOL_PATH = v1.PROTOCOL_PATH
_V1_EXPECTED_PROTOCOL_SHA256 = v1.EXPECTED_PROTOCOL_SHA256
_V1_EXPECTED_SCHEMA = v1.EXPECTED_SCHEMA


def _expect(actual, expected, label):
    core._expect_equal(actual, expected, label)


def _require_bound(record, expected_sha, label):
    path = core.workspace_path(record["workspace_relative_path"])
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    digest = core.sha256_file(path)
    if digest != expected_sha or digest != record["sha256"]:
        raise RuntimeError("{} SHA-256 differs from V2 recovery.".format(label))
    return path


def _load_v1_effective():
    current = (v1.PROTOCOL_PATH, v1.EXPECTED_PROTOCOL_SHA256, v1.EXPECTED_SCHEMA)
    try:
        v1.PROTOCOL_PATH = _V1_PROTOCOL_PATH
        v1.EXPECTED_PROTOCOL_SHA256 = _V1_EXPECTED_PROTOCOL_SHA256
        v1.EXPECTED_SCHEMA = _V1_EXPECTED_SCHEMA
        return _V1_LOAD_PROTOCOL()
    finally:
        v1.PROTOCOL_PATH, v1.EXPECTED_PROTOCOL_SHA256, v1.EXPECTED_SCHEMA = current


def _validate_v2_overlay(overlay, digest):
    _expect(digest, EXPECTED_PROTOCOL_SHA256, "V2 protocol SHA-256")
    _expect(overlay.get("schema"), EXPECTED_SCHEMA, "V2 schema")
    _expect(
        overlay.get("status"),
        "frozen_after_v1_cpu_publish_failure_before_any_alpha130_checkpoint_or_inference",
        "V2 status",
    )
    inheritance = overlay["v1_science_inheritance"]
    _require_bound(inheritance["protocol"], V1_PROTOCOL_SHA256, "V1 protocol")
    _require_bound(inheritance["runner"], V1_RUNNER_SHA256, "V1 runner")
    _require_bound(inheritance["tests"], V1_TEST_SHA256, "V1 tests")
    _require_bound(inheritance["command_audit"], V1_COMMAND_AUDIT_SHA256, "V1 command audit")
    failure = overlay["v1_failure_receipt"]
    required_failure = {
        "failed_command": "synthesize",
        "failure_phase": "temporary_torch_file_fsync_before_exclusive_hard_link_publish",
        "exception_type": "OSError",
        "exception_message": "[Errno 9] Bad file descriptor",
        "candidate_checkpoint_publish_count": 0,
        "synthesis_manifest_published": False,
        "formal_alpha130_inference_count": 0,
        "anchor_content_read": False,
        "validation_or_test_read": False,
        "gpu_or_cuda_used": False,
        "input_checkpoint_mutation": False,
        "v1_output_root_must_remain_unmodified": True,
    }
    for key, expected in required_failure.items():
        _expect(failure.get(key), expected, "V1 failure {}".format(key))
    expected_v1_outputs = [
        V1_OUTPUT_ROOT / "synthesis" / "synthesis_manifest.json",
        V1_OUTPUT_ROOT / "grouped_oof_report.json",
        *[
            V1_OUTPUT_ROOT / "synthesis" / fold_id / "isolated_metric_aux_alpha130.pt"
            for fold_id in ("hold_g1", "hold_g2", "hold_g3")
        ],
        *[
            V1_OUTPUT_ROOT / "held_train_evaluation" / "{}_alpha130".format(fold_id) / "evaluation.json"
            for fold_id in ("hold_g1", "hold_g2", "hold_g3")
        ],
    ]
    if any(path.exists() for path in expected_v1_outputs):
        raise RuntimeError("V1 unexpectedly published a candidate, manifest, evaluation, or report.")
    recovery = overlay["recovery_amendment"]
    _expect(recovery.get("only_code_change_allowed"), "open_the_fully_written_temporary_torch_file_as_rb_plus_instead_of_rb_before_fsync", "recovery change")
    _expect(recovery.get("science_candidate_alpha_threshold_c00_fold_source_geometry_and_gates_changed"), False, "science unchanged")
    _expect(recovery.get("alpha"), 1.3, "recovery alpha")
    _expect(recovery.get("candidate_count"), 1, "recovery candidate count")
    _expect(recovery.get("new_training_optimizer_steps"), 0, "recovery training steps")
    _expect(recovery.get("new_formal_candidate_inference_count"), 3, "recovery inference count")
    _expect(recovery.get("alpha_grid_module_projection_threshold_or_c00_search_allowed"), False, "recovery search")
    _expect(overlay["cli_contract"]["allowed_commands"], ["audit", "synthesize", "evaluate", "report", "all-evaluate-report"], "recovery CLI")
    for key in (
        "validation_or_test_read_allowed",
        "current_failed_validation_report_read_allowed",
        "persistence_formal_artifact_read_allowed",
        "new_training_allowed",
        "platform_submission_allowed",
    ):
        _expect(overlay.get(key), False, key)


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V2 protocol SHA-256 differs from frozen recovery.")
    overlay, digest = core.load_json_snapshot(PROTOCOL_PATH)
    _validate_v2_overlay(overlay, digest)
    effective, inherited_sha = _load_v1_effective()
    _expect(inherited_sha, V1_PROTOCOL_SHA256, "inherited V1 protocol")
    effective = copy.deepcopy(effective)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["alpha130_contract"]["output_contract"] = copy.deepcopy(overlay["output_contract"])
    effective["alpha130_contract"]["cpu_publish_recovery_v2"] = copy.deepcopy(overlay)
    effective["outputs"]["workspace_relative_directory"] = overlay["output_contract"]["workspace_relative_directory"]
    return effective, actual


def _atomic_torch_save_exclusive(payload, destination):
    import torch

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite synthesis output: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".{}-".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def command_audit_payload(protocol, protocol_sha, assets, preflight):
    payload = _V1_COMMAND_AUDIT_PAYLOAD(protocol, protocol_sha, assets, preflight)
    payload["schema"] = "ev-uav-metric-aux-task-arithmetic-alpha130-cpu-publish-recovery-v2-command-audit-v1"
    payload["cpu_publish_recovery_v2"] = {
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_runner_sha256": V1_RUNNER_SHA256,
        "v1_tests_sha256": V1_TEST_SHA256,
        "v1_command_audit_sha256": V1_COMMAND_AUDIT_SHA256,
        "v1_candidate_checkpoint_publish_count": 0,
        "v1_synthesis_manifest_published": False,
        "only_execution_change": "temporary_torch_fsync_handle_rb_to_rb_plus",
        "science_definition_unchanged": True,
    }
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the immutable V2 CPU recovery audit first.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-alpha130-cpu-publish-recovery-v2-command-audit-v1", "V2 audit schema")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "V2 audit protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "V2 audit runner")
    _expect(payload.get("tests_sha256"), core.sha256_file(TEST_PATH), "V2 audit tests")
    _expect(payload.get("alpha"), 1.3, "V2 audit alpha")
    _expect(payload.get("new_training_optimizer_steps"), 0, "V2 audit training")
    _expect(payload.get("task_arithmetic_preflight", {}).get("passed"), True, "V2 audit preflight")
    _expect(payload.get("cpu_publish_recovery_v2", {}).get("science_definition_unchanged"), True, "V2 science unchanged")
    return payload, digest


# Parameterize the private V1 implementation with V2 artifact identities.
for name, value in {
    "RUNNER_PATH": RUNNER_PATH,
    "TEST_PATH": TEST_PATH,
    "PROTOCOL_PATH": PROTOCOL_PATH,
    "EXPECTED_PROTOCOL_SHA256": EXPECTED_PROTOCOL_SHA256,
    "EXPECTED_SCHEMA": EXPECTED_SCHEMA,
    "OUTPUT_ROOT": OUTPUT_ROOT,
    "COMMAND_AUDIT_PATH": COMMAND_AUDIT_PATH,
    "SYNTHESIS_ROOT": SYNTHESIS_ROOT,
    "SYNTHESIS_MANIFEST_PATH": SYNTHESIS_MANIFEST_PATH,
    "EVALUATION_ROOT": EVALUATION_ROOT,
    "REPORT_PATH": REPORT_PATH,
    "load_protocol": load_protocol,
    "load_command_audit": load_command_audit,
    "command_audit_payload": command_audit_payload,
    "_atomic_torch_save_exclusive": _atomic_torch_save_exclusive,
}.items():
    setattr(v1, name, value)
v1._patch_core_for_alpha130()

synthesis_specs = v1.synthesis_specs
synthesize_state_dict = v1.synthesize_state_dict
task_arithmetic_preflight = v1.task_arithmetic_preflight
evaluation_specs = v1.evaluation_specs
dual_anchor_gate = v1.dual_anchor_gate
run_audit = v1.run_audit
run_synthesis = v1.run_synthesis
load_synthesis_manifest = v1.load_synthesis_manifest
run_evaluation = v1.run_evaluation
run_report = v1.run_report
build_parser = v1.build_parser


def main(argv=None):
    return v1.main(argv)


if __name__ == "__main__":
    main()

"""Train-only all11 paired refit and alpha=1 task-arithmetic synthesis.

This versioned runner exposes no evaluation, validation, test, threshold-search,
submission, or probe command.  It reuses the frozen V3 resource receipt, trains
one baseline and one metric-aux E3 arm on exactly train_088..train_098, audits
the pair, and then creates exactly one inference-only checkpoint::

    W_full = W_M20 + (W_all11_metric_aux_E3 - W_all11_baseline_E3)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
RUNNER_PATH = Path(__file__).resolve()

_V5_NAME = "_metric_aux_task_arithmetic_h2_grouped_oof_for_all11_final"
_previous_v5 = sys.modules.get(_V5_NAME)
_V5_PATH = EVC_ROOT / "run_metric_aux_task_arithmetic_h2_grouped_oof.py"
_V5_SPEC = importlib.util.spec_from_file_location(_V5_NAME, _V5_PATH)
if _V5_SPEC is None or _V5_SPEC.loader is None:
    raise ImportError("Unable to create a private V5 task-arithmetic module.")
v5 = importlib.util.module_from_spec(_V5_SPEC)
sys.modules[_V5_NAME] = v5
try:
    _V5_SPEC.loader.exec_module(v5)
finally:
    if _previous_v5 is None:
        sys.modules.pop(_V5_NAME, None)
    else:
        sys.modules[_V5_NAME] = _previous_v5

core = v5.core
_V5_LOAD_PROTOCOL = v5.load_protocol
_V5_SYNTHESIZE_STATE_DICT = v5.synthesize_state_dict
_V5_STATE_EQUAL = v5._state_equal
_V5_STRICT_LOAD = v5._strict_model_state_load_cpu
_V5_ATOMIC_TORCH_SAVE = v5._atomic_torch_save
_V5_MODEL_STATE_SHA = v5.model_state_canonical_sha256

PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "metric_aux_task_arithmetic_all11_final_refit_science_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = "570f0bedfc76794ebdbdebefbc7dbac4a00c9c21fe6b7660c1f6bdf7adb05f19"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-task-arithmetic-all11-final-refit-v1"
V5_PROTOCOL_SHA256 = "4586e1e20a501a4b5219c2d5953a6666c1fc0d4ac08da2a7b29d6cea70938266"
V5_RUNNER_SHA256 = "26dc7b53402afa9681add0448ef9e87c5dc297c6361066512d241bfbe8142e86"
V5_REPORT_SHA256 = "df4d8d6bf123e75ae1be7f9f53ef03442daf50f9683042fdfd0343356ce5110e"
V3_PROTOCOL_SHA256 = "a4039cdba26ed1f950d62b40edc4b13c9868ac281c0dcd6b5a37e2062cd79875"
V3_RUNNER_SHA256 = "968d2fa0bc32756c5f8b059d44d0b9413daf75396c7d55733a68d126d2148236"
V3_COMMAND_AUDIT_SHA256 = "a369096bf1b716b40284e7da7d9ec9397ca82e6ecd783ec44b47404c0749c2ea"
V3_PROBE_SHA256 = "09e05f582b439525fb7d04006766258173c502af23b9b1acb6fe5846e95e2b06"
M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"

OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260810_metric_aux_task_arithmetic_all11_final_refit_v1"
)
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
TRAINING_ROOT = OUTPUT_ROOT / "paired_training"
PAIR_AUDIT_PATH = TRAINING_ROOT / "pair_audit.json"
SYNTHESIS_ROOT = OUTPUT_ROOT / "synthesis"
FINAL_CHECKPOINT_PATH = SYNTHESIS_ROOT / "metric_aux_task_arithmetic_all11_alpha1.pt"
SYNTHESIS_MANIFEST_PATH = SYNTHESIS_ROOT / "synthesis_manifest.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"


def _expect(actual, expected, label):
    core._expect_equal(actual, expected, label)


def _bound_path(record):
    return core.workspace_path(record["workspace_relative_path"])


def _require_bound_file(record, label):
    path = _bound_path(record)
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    digest = core.sha256_file(path)
    _expect(digest, record["sha256"], "{} SHA-256".format(label))
    return path


def _load_bound_json(record, label):
    path = _require_bound_file(record, label)
    payload, digest = core.load_json_snapshot(path)
    _expect(digest, record["sha256"], "{} stable snapshot".format(label))
    return payload, path


def _validate_v5_selection(overlay):
    selection = overlay["selection_evidence"]
    expected = {
        "v5_protocol": V5_PROTOCOL_SHA256,
        "v5_runner": V5_RUNNER_SHA256,
        "v5_report": V5_REPORT_SHA256,
    }
    for key, digest in expected.items():
        _expect(selection[key]["sha256"], digest, "{} contract".format(key))
    for key in (
        "v5_protocol",
        "v5_runner",
        "v5_tests",
        "v5_command_audit",
        "v5_synthesis_manifest",
    ):
        _require_bound_file(selection[key], key)
    report, _ = _load_bound_json(selection["v5_report"], "V5 report")
    _expect(report.get("schema"), "ev-uav-metric-aux-task-arithmetic-h2-grouped-oof-report-v1", "V5 report schema")
    _expect(report.get("protocol_sha256"), V5_PROTOCOL_SHA256, "V5 report protocol")
    _expect(report.get("runner_sha256"), V5_RUNNER_SHA256, "V5 report runner")
    _expect(report.get("passed"), selection["required_v5_passed"], "V5 promotion")
    _expect(report.get("status"), "passed", "V5 report status")
    _expect(report.get("decision"), selection["required_v5_decision"], "V5 decision")
    _expect(report.get("released_m20_anchor_reinference"), selection["required_v5_released_m20_anchor_reinference"], "V5 M20 anchor reuse")
    _expect(
        report["pooled"]["isolated_metric_aux"]["metric_delta"]["score"],
        selection["required_v5_pooled_delta_score"],
        "V5 pooled score delta",
    )
    evaluations = {}
    report_artifacts = report["evaluation_artifacts"]
    for fold_id, record in selection["v5_candidate_evaluations"].items():
        payload, path = _load_bound_json(record, "V5 {} evaluation".format(fold_id))
        eval_id = "{}_isolated_metric_aux".format(fold_id)
        _expect(payload.get("eval_id"), eval_id, "V5 evaluation id")
        _expect(payload.get("fold_id"), fold_id, "V5 fold id")
        _expect(payload.get("protocol_sha256"), V5_PROTOCOL_SHA256, "V5 evaluation protocol")
        _expect(payload.get("runner_sha256"), V5_RUNNER_SHA256, "V5 evaluation runner")
        _expect(report_artifacts[eval_id]["sha256"], record["sha256"], "V5 report evaluation hash")
        _expect(Path(report_artifacts[eval_id]["path"]).resolve(), path.resolve(), "V5 report evaluation path")
        evaluations[fold_id] = payload
    _expect(len(report["fold_results"]), 3, "V5 fold count")
    _expect(all(report["promotion_gates"].values()), True, "V5 all promotion gates")
    return {"report": report, "evaluations": evaluations}


def _validate_resource_evidence(overlay):
    evidence = overlay["resource_feasibility_evidence"]
    expected = {
        "v3_protocol": V3_PROTOCOL_SHA256,
        "v3_runner": V3_RUNNER_SHA256,
        "v3_command_audit": V3_COMMAND_AUDIT_SHA256,
        "v3_probe_receipt": V3_PROBE_SHA256,
    }
    for key, digest in expected.items():
        _expect(evidence[key]["sha256"], digest, "{} contract".format(key))
    for key in ("v3_protocol", "v3_runner", "v3_command_audit", "v3_pair_audit"):
        _require_bound_file(evidence[key], key)
    probe, _ = _load_bound_json(evidence["v3_probe_receipt"], "V3 probe receipt")
    _expect(probe.get("schema"), "ev-uav-metric-aux-audit-only-probe-result-v3", "V3 probe schema")
    _expect(probe.get("passed"), True, "V3 probe passed")
    _expect(all(probe.get("checks", {}).values()), True, "V3 probe gates")
    _expect(probe.get("protocol_sha256"), V3_PROTOCOL_SHA256, "V3 probe protocol")
    _expect(probe.get("runner_sha256"), V3_RUNNER_SHA256, "V3 probe runner")
    _expect(probe.get("command_audit_sha256"), V3_COMMAND_AUDIT_SHA256, "V3 probe audit")
    _expect(probe.get("new_pair_training_optimizer_steps"), 0, "V3 reused training steps")
    _expect(probe["corrected_real_batch_metric_gradient_probe"].get("passed"), True, "V3 real-batch gradient")
    return probe


def _validate_overlay(overlay, digest, inherited):
    _expect(overlay.get("schema"), EXPECTED_SCHEMA, "all11 schema")
    _expect(digest, EXPECTED_PROTOCOL_SHA256, "all11 protocol SHA-256")
    _expect(
        overlay.get("status"),
        "frozen_after_adaptive_v5_train_only_promotion_before_any_all11_training_or_synthesis",
        "all11 protocol status",
    )
    _expect(overlay.get("validation_or_test_read_allowed"), False, "split access")
    _expect(overlay.get("t32_allowed"), False, "T32 access")
    _expect(overlay["cli_contract"]["allowed_commands"], ["audit", "train", "audit-training", "synthesize"], "CLI commands")
    _expect(overlay["cli_contract"]["evaluate_or_report_command_exposed"], False, "evaluation CLI")
    _expect(overlay["cli_contract"]["probe_command_exposed"], False, "probe CLI")
    selection = _validate_v5_selection(overlay)
    probe = _validate_resource_evidence(overlay)
    for key in ("m23_run_summary", "m23_training_log"):
        _require_bound_file(overlay["historical_hyperparameter_anchor"][key], key)
    parent = overlay["parent_checkpoint"]
    _expect(parent["sha256"], M20_SHA256, "M20 contract")
    _require_bound_file(parent, "released M20")

    source = overlay["source_contract"]
    names = [
        item["name"]
        for group in source["source_groups_in_order"]
        for item in inherited["dataset"]["source_groups"][group]
    ]
    _expect(names, source["source_names_in_order"], "all11 source order")
    _expect(len(names), source["source_count"], "all11 source count")
    _expect(source["held_source_count"], 0, "all11 held count")

    frozen = overlay["paired_refit_contract"]
    training = inherited["training"]
    candidate = training["candidate"]
    exact = {
        "seed": training["seed"],
        "epochs": training["epochs"],
        "sequence_length": training["sequence_length"],
        "views_per_video": training["views_per_video"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "scheduler": training["scheduler"],
        "scheduler_min_lr": training["scheduler_min_lr"],
        "metric_target_weight": candidate["metric_target_weight"],
        "metric_component_weight": candidate["metric_component_weight"],
        "metric_warmup_epochs": candidate["metric_warmup_epochs"],
        "metric_spatial_cell_size": candidate["metric_spatial_cell_size"],
        "metric_min_cell_events": candidate["metric_min_cell_events"],
        "metric_component_ratio": candidate["metric_component_ratio"],
        "metric_activation_threshold": candidate["metric_activation_threshold"],
        "metric_activation_temperature": candidate["metric_activation_temperature"],
    }
    for key, value in exact.items():
        _expect(frozen[key], value, "all11 inherited {}".format(key))
    scope = training["training_scope_audit"]
    _expect(frozen["trainable_state_tensor_count"], scope["trainable_state_tensor_count"], "scope tensors")
    _expect(frozen["trainable_parameter_count"], scope["trainable_parameter_count"], "scope params")
    _expect(frozen["frozen_parameter_count"], scope["frozen_parameter_count"], "frozen params")
    _expect(frozen["trainable_name_shape_canonical_sha256"], scope["trainable_name_shape_canonical_sha256"], "scope canonical")
    _expect(frozen["expected_sequences_per_epoch_each_arm"], 88, "sequences per arm epoch")
    _expect(frozen["expected_optimizer_steps_each_arm"], 264, "steps per arm")
    _expect(frozen["expected_optimizer_steps_paired_total"], 528, "paired steps")
    _expect(frozen["only_resolved_config_differences"], training["paired_difference_allowlist"], "pair difference allowlist")
    return {"v5": selection, "v3_probe": probe}


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("All11 protocol SHA-256 {} differs from frozen {}.".format(actual, EXPECTED_PROTOCOL_SHA256))
    overlay, snapshot_sha = core.load_json_snapshot(PROTOCOL_PATH)
    _expect(snapshot_sha, actual, "all11 protocol snapshot")
    inherited, inherited_sha = _V5_LOAD_PROTOCOL()
    _expect(inherited_sha, V5_PROTOCOL_SHA256, "inherited V5 protocol")
    evidence = _validate_overlay(overlay, actual, inherited)
    effective = copy.deepcopy(inherited)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["audit_amendment"] = {
        "revision_reason": "V5 adaptive train-only task arithmetic passed; fit one all11 pair for a single deployment candidate.",
        "shared_parent_pretraining_exposure": True,
        "claim_scope": overlay["claim_scope"],
        "paired_causal_scope": "Within one shared M20 initialization and one exact all11 data view, isolate only the metric-aux enable treatment.",
        "fold_clean_parent_retraining_required": False,
    }
    effective["training"]["formal_optimizer_steps_total"] = 528
    effective["training"]["sampling_contract"]["current_grouped_oof"] = "all11 final paired refit: uniform eight views for every one of the eleven H2 sources, dense sampling disabled"
    effective["outputs"] = {
        "workspace_relative_directory": overlay["output_contract"]["workspace_relative_directory"],
        "command_audit": overlay["output_contract"]["command_audit"],
        "formal_training_directory": overlay["output_contract"]["training_directory"],
    }
    effective["all11_final_refit_contract"] = copy.deepcopy(overlay)
    effective["_all11_bound_evidence"] = evidence
    return effective, actual


def all11_items(protocol):
    overlay = protocol["all11_final_refit_contract"]
    index = core.source_index(protocol)
    names = overlay["source_contract"]["source_names_in_order"]
    if set(index) != set(names) or len(index) != 11:
        raise RuntimeError("Effective source union is not exactly the frozen all11 set.")
    return [index[name] for name in names]


def training_specs(protocol, view):
    frozen = protocol["all11_final_refit_contract"]["paired_refit_contract"]
    specs = []
    for variant in frozen["variants_in_order"]:
        run_id = "all11_{}".format(variant)
        output = TRAINING_ROOT / run_id
        specs.append(
            {
                "run_id": run_id,
                "fold_id": "all11_final",
                "variant": variant,
                "fit_groups": list(protocol["all11_final_refit_contract"]["source_contract"]["source_groups_in_order"]),
                "held_group": None,
                "data_root": view["root"],
                "expected_source_names": list(protocol["all11_final_refit_contract"]["source_contract"]["source_names_in_order"]),
                "expected_videos": 11,
                "expected_sequences_per_epoch": 88,
                "expected_optimizer_steps": 264,
                "epochs": 3,
                "output_root": str(output.resolve()),
                "model_root": str((output / "model").resolve()),
                "result_path": str((output / "runtime_result.json").resolve()),
            }
        )
    return specs


def _pair_command_diff(protocol, specs, commands):
    by_variant = {spec["variant"]: spec for spec in specs}
    if set(by_variant) != {"baseline", "metric_aux"}:
        raise RuntimeError("All11 pair specification is incomplete.")
    baseline = core.override_mapping(commands[by_variant["baseline"]["run_id"]]["overrides"])
    candidate = core.override_mapping(commands[by_variant["metric_aux"]["run_id"]]["overrides"])
    differences = {
        key: {"baseline": baseline.get(key), "metric_aux": candidate.get(key)}
        for key in sorted(set(baseline) | set(candidate))
        if baseline.get(key) != candidate.get(key)
    }
    expected = set(protocol["all11_final_refit_contract"]["paired_refit_contract"]["only_resolved_config_differences"])
    if set(differences) != expected:
        raise RuntimeError("All11 pair differs outside the frozen two-path allowlist: {}".format(sorted(differences)))
    return differences


def command_audit_payload(protocol, protocol_sha, assets, view):
    specs = training_specs(protocol, view)
    commands = {}
    resolved = {}
    for spec in specs:
        argv, overrides = core.training_argv(protocol, spec["data_root"], spec["model_root"], spec["variant"], spec["epochs"])
        commands[spec["run_id"]] = {
            "argv": [sys.executable, *argv],
            "overrides": overrides,
            "variant": spec["variant"],
            "expected_source_names": spec["expected_source_names"],
            "expected_sequences_per_epoch": 88,
            "expected_optimizer_steps": 264,
        }
        resolved[spec["run_id"]] = core.resolve_training_config(protocol, spec)
    differences = _pair_command_diff(protocol, specs, commands)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during all11 CPU audit.")
    return {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-command-audit-v1",
        "created_utc": core.utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "assets": assets,
        "data_view": view,
        "training_specs": specs,
        "training_commands": commands,
        "resolved_preflights": resolved,
        "paired_difference_audit": differences,
        "resource_probe_reused": {
            "path": str(_bound_path(protocol["all11_final_refit_contract"]["resource_feasibility_evidence"]["v3_probe_receipt"])),
            "sha256": V3_PROBE_SHA256,
            "passed": True,
            "new_probe_optimizer_steps": 0,
        },
        "expected_optimizer_steps_each_arm": 264,
        "expected_optimizer_steps_paired_total": 528,
        "allowed_cli_commands": ["audit", "train", "audit-training", "synthesize"],
        "train_commands": {
            spec["run_id"]: [sys.executable, str(RUNNER_PATH), "train", "--run-id", spec["run_id"], GPU_AUTHORIZATION_FLAG]
            for spec in specs
        },
        "pair_audit_command": [sys.executable, str(RUNNER_PATH), "audit-training"],
        "synthesis_command": [sys.executable, str(RUNNER_PATH), "synthesize"],
        "evaluation_commands": [],
        "gpu_or_cuda_initialized": False,
        "data_use_statement": "Only frozen train identities and train-derived evidence were read; no validation/test, T32 or persistence-formal artifact was opened.",
    }


def run_audit():
    protocol, protocol_sha = load_protocol()
    assets = core.verify_assets(protocol)
    view = core.materialize_view(protocol, "all11_fit", all11_items(protocol))
    payload = command_audit_payload(protocol, protocol_sha, assets, view)
    _expect(sum(item["expected_optimizer_steps"] for item in payload["training_specs"]), 528, "paired optimizer budget")
    core.write_new_json(COMMAND_AUDIT_PATH, payload)
    print("all11 command audit:", COMMAND_AUDIT_PATH)
    print("command audit sha256:", core.sha256_file(COMMAND_AUDIT_PATH))
    print("CPU audit complete; GPU training remains authorization-gated.")
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the immutable all11 CPU audit first.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-all11-command-audit-v1", "command schema")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "command protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "command runner")
    _expect(payload.get("expected_optimizer_steps_each_arm"), 264, "command arm steps")
    _expect(payload.get("expected_optimizer_steps_paired_total"), 528, "command paired steps")
    _expect(payload.get("evaluation_commands"), [], "command evaluation prohibition")
    _expect(payload.get("gpu_or_cuda_initialized"), False, "command CUDA")
    _expect(set(payload["paired_difference_audit"]), {"TRAIN.model_save_root", "TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled"}, "command differences")
    expected_names = load_protocol()[0]["all11_final_refit_contract"]["source_contract"]["source_names_in_order"]
    records = payload["data_view"]["records"]
    _expect([item["name"] for item in records], expected_names, "command source order")
    for item in records:
        source, destination = Path(item["source"]), Path(item["destination"])
        if not source.is_file() or not destination.is_file() or not os.path.samefile(source, destination):
            raise RuntimeError("All11 data-view hardlink changed: {}".format(destination))
        _expect(core.sha256_file(destination), item["sha256"], "all11 data-view source SHA")
    return payload, digest


def _specs_from_audit(protocol, audit):
    specs = training_specs(protocol, audit["data_view"])
    _expect(specs, audit["training_specs"], "training specs frozen in command audit")
    return specs


def run_training(run_id=None, authorized=False):
    core.require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    audit, _ = load_command_audit()
    _validate_resource_evidence(protocol["all11_final_refit_contract"])
    specs = _specs_from_audit(protocol, audit)
    if run_id is not None:
        matches = [spec for spec in specs if spec["run_id"] == run_id]
        if len(matches) != 1:
            raise KeyError("Unknown all11 run id: {}".format(run_id))
        return [core.run_training_spec(protocol, matches[0])]
    results = []
    for spec in specs:
        if Path(spec["result_path"]).is_file():
            result, _ = core.load_training_result(spec)
            results.append(result)
            print("retaining completed all11 training:", spec["run_id"], flush=True)
            continue
        subprocess.run(
            [sys.executable, str(RUNNER_PATH), "train", "--run-id", spec["run_id"], GPU_AUTHORIZATION_FLAG],
            cwd=str(EVC_ROOT),
            check=True,
        )
        result, _ = core.load_training_result(spec)
        results.append(result)
    if PAIR_AUDIT_PATH.exists():
        raise FileExistsError("Refusing to overwrite all11 pair audit.")
    run_pair_audit(write=True)
    return results


def _model_difference(left, right):
    import torch
    left_state = left["model_state_dict"] if "model_state_dict" in left else left
    right_state = right["model_state_dict"] if "model_state_dict" in right else right
    if list(left_state) != list(right_state):
        raise RuntimeError("Model-state key order differs.")
    squared = 0.0
    max_abs = 0.0
    changed = 0
    finite = True
    for name in left_state:
        a, b = left_state[name], right_state[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("Model-state metadata differs: {}".format(name))
        delta = a.detach().cpu().to(torch.float64) - b.detach().cpu().to(torch.float64)
        finite = finite and bool(torch.isfinite(delta).all())
        squared += float(torch.sum(delta * delta))
        if delta.numel():
            max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
        changed += int(torch.count_nonzero(delta))
    return {"global_l2": math.sqrt(squared), "max_abs": max_abs, "changed_elements": changed, "finite": finite}


def _load_pair_results(protocol, audit):
    specs = _specs_from_audit(protocol, audit)
    by_variant = {spec["variant"]: spec for spec in specs}
    baseline, baseline_sha = core.load_training_result(by_variant["baseline"])
    metric_aux, metric_aux_sha = core.load_training_result(by_variant["metric_aux"])
    return by_variant, baseline, baseline_sha, metric_aux, metric_aux_sha


def run_pair_audit(write=True):
    protocol, _ = load_protocol()
    audit, command_audit_sha = load_command_audit()
    by_variant, baseline, baseline_sha, metric_aux, metric_aux_sha = _load_pair_results(protocol, audit)
    pair = core.compare_pair_checkpoints(baseline, metric_aux)
    if pair.get("audit_version") != "numeric_near_identity_v2" or pair.get("passed") is not True:
        raise RuntimeError("All11 numeric paired checkpoint gate failed.")
    b_e3_path = Path(baseline["checkpoints"]["e3"]["path"])
    a_e3_path = Path(metric_aux["checkpoints"]["e3"]["path"])
    parent_path = _bound_path(protocol["all11_final_refit_contract"]["parent_checkpoint"])
    before = {
        "parent": core.sha256_file(parent_path),
        "baseline_e3": core.sha256_file(b_e3_path),
        "metric_aux_e3": core.sha256_file(a_e3_path),
    }
    _expect(before["parent"], M20_SHA256, "pair M20 SHA")
    parent = core.load_torch_checkpoint(parent_path)
    b_e3 = core.load_torch_checkpoint(b_e3_path)
    a_e3 = core.load_torch_checkpoint(a_e3_path)
    task = _model_difference(a_e3, b_e3)
    drift = _model_difference(b_e3, parent)
    ratio = task["global_l2"] / drift["global_l2"] if drift["global_l2"] > 0.0 else math.inf
    active_epochs = [
        epoch
        for epoch in range(3)
        if int(metric_aux["auxiliary_loss_stats"]["epochs"][str(epoch)]["target"]["calls"]) > 0
        or int(metric_aux["auxiliary_loss_stats"]["epochs"][str(epoch)]["component"]["calls"]) > 0
    ]
    scope_reaudit = {}
    for variant, result in (("baseline", baseline), ("metric_aux", metric_aux)):
        scope_reaudit[variant] = {}
        for epoch_key in ("e1", "e2", "e3"):
            current = core.checkpoint_scope_audit(
                protocol, Path(result["checkpoints"][epoch_key]["path"])
            )
            _expect(
                current["sha256"],
                result["checkpoints"][epoch_key]["sha256"],
                "{} {} scope checkpoint SHA".format(variant, epoch_key),
            )
            scope_reaudit[variant][epoch_key] = current
    auxiliary_reaudit = {
        "baseline": core.validate_auxiliary_stats(
            protocol, baseline["auxiliary_loss_stats"], "baseline", 3, 88
        ),
        "metric_aux": core.validate_auxiliary_stats(
            protocol, metric_aux["auxiliary_loss_stats"], "metric_aux", 3, 88
        ),
    }
    frozen = protocol["all11_final_refit_contract"]
    checks = {
        "numeric_pair_passed": pair["passed"],
        "source_names_and_order_exact": baseline["expected_source_names"] == metric_aux["expected_source_names"] == frozen["source_contract"]["source_names_in_order"],
        "expected_steps_each_arm_exact": int(baseline["expected_optimizer_steps"]) == int(metric_aux["expected_optimizer_steps"]) == 264,
        "training_commands_exact": baseline["overrides"] == audit["training_commands"]["all11_baseline"]["overrides"] and metric_aux["overrides"] == audit["training_commands"]["all11_metric_aux"]["overrides"],
        "resolved_preflights_passed": baseline["resolved_preflight"]["passed"] is True and metric_aux["resolved_preflight"]["passed"] is True,
        "active_epochs_exact": active_epochs == [1, 2],
        "baseline_auxiliary_audit_passed": baseline["auxiliary_loss_audit"]["passed"] is True,
        "metric_auxiliary_audit_passed": metric_aux["auxiliary_loss_audit"]["passed"] is True,
        "input_sources_unchanged": baseline["input_source_sha256_before_after_equal"] is True and metric_aux["input_source_sha256_before_after_equal"] is True,
        "core_files_unchanged": baseline["core_sha256_before_after_equal"] is True and metric_aux["core_sha256_before_after_equal"] is True,
        "all_six_checkpoint_scope_reaudits_passed": all(item["passed"] for values in scope_reaudit.values() for item in values.values()),
        "both_auxiliary_stats_reaudits_passed": all(item["passed"] for item in auxiliary_reaudit.values()),
        "task_vector_finite_nonzero": task["finite"] and task["global_l2"] > 0.0,
        "baseline_drift_finite_nonzero": drift["finite"] and drift["global_l2"] > 0.0,
        "task_over_drift_open_interval": math.isfinite(ratio) and 0.0 < ratio < 0.1,
    }
    if not all(checks.values()):
        raise RuntimeError("All11 paired post-run audit failed: {}".format(checks))
    after = {
        "parent": core.sha256_file(parent_path),
        "baseline_e3": core.sha256_file(b_e3_path),
        "metric_aux_e3": core.sha256_file(a_e3_path),
    }
    if before != after:
        raise RuntimeError("Pair-audit checkpoint changed while being read.")
    payload = {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-pair-audit-v1",
        "created_utc": core.utc_now(),
        "status": "passed",
        "passed": True,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "command_audit_sha256": command_audit_sha,
        "baseline_result_sha256": baseline_sha,
        "metric_aux_result_sha256": metric_aux_sha,
        "input_checkpoint_sha256": before,
        "numeric_pair_audit": pair,
        "checkpoint_scope_reaudit": scope_reaudit,
        "auxiliary_stats_reaudit": auxiliary_reaudit,
        "active_zero_based_epochs": active_epochs,
        "task_vector_stats": task,
        "baseline_drift_stats": drift,
        "task_vector_over_baseline_drift": ratio,
        "checks": checks,
        "claim_scope": frozen["claim_scope"],
        "all11_has_no_held_source": True,
    }
    if write:
        core.write_new_json(PAIR_AUDIT_PATH, payload)
        print("all11 pair audit:", PAIR_AUDIT_PATH)
    return payload


def load_pair_audit():
    if not PAIR_AUDIT_PATH.is_file():
        raise FileNotFoundError("All11 pair audit has not passed.")
    payload, digest = core.load_json_snapshot(PAIR_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-all11-pair-audit-v1", "pair schema")
    _expect(payload.get("passed"), True, "pair passed")
    _expect(all(payload.get("checks", {}).values()), True, "pair gates")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "pair protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "pair runner")
    protocol, _ = load_protocol()
    audit, command_audit_sha = load_command_audit()
    _expect(payload.get("command_audit_sha256"), command_audit_sha, "pair command audit")
    _, _, baseline_sha, _, metric_aux_sha = _load_pair_results(protocol, audit)
    _expect(payload.get("baseline_result_sha256"), baseline_sha, "pair baseline result")
    _expect(payload.get("metric_aux_result_sha256"), metric_aux_sha, "pair metric result")
    for key, path in {
        "parent": _bound_path(protocol["all11_final_refit_contract"]["parent_checkpoint"]),
        "baseline_e3": Path(_load_pair_results(protocol, audit)[1]["checkpoints"]["e3"]["path"]),
        "metric_aux_e3": Path(_load_pair_results(protocol, audit)[3]["checkpoints"]["e3"]["path"]),
    }.items():
        _expect(core.sha256_file(path), payload["input_checkpoint_sha256"][key], "pair {} current SHA".format(key))
    return payload, digest


def _arithmetic_stats(parent, baseline, metric_aux):
    task = _model_difference(metric_aux, baseline)
    drift = _model_difference(baseline, parent)
    ratio = task["global_l2"] / drift["global_l2"] if drift["global_l2"] > 0.0 else math.inf
    return {"task_vector": task, "baseline_drift": drift, "task_vector_over_baseline_drift": ratio}


def run_synthesis():
    import torch
    protocol, _ = load_protocol()
    audit, command_audit_sha = load_command_audit()
    pair_audit, pair_audit_sha = load_pair_audit()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU all11 synthesis.")
    if FINAL_CHECKPOINT_PATH.exists() or SYNTHESIS_MANIFEST_PATH.exists():
        raise FileExistsError("Refusing to overwrite all11 synthesis evidence.")
    _, baseline, baseline_result_sha, metric_aux, metric_aux_result_sha = _load_pair_results(protocol, audit)
    paths = {
        "parent": _bound_path(protocol["all11_final_refit_contract"]["parent_checkpoint"]),
        "baseline_e3": Path(baseline["checkpoints"]["e3"]["path"]),
        "metric_aux_e3": Path(metric_aux["checkpoints"]["e3"]["path"]),
    }
    before = {key: core.sha256_file(path) for key, path in paths.items()}
    _expect(before, pair_audit["input_checkpoint_sha256"], "synthesis inputs bound to pair audit")
    payloads = {key: core.load_torch_checkpoint(path) for key, path in paths.items()}
    states = [payloads[key]["model_state_dict"] for key in ("parent", "baseline_e3", "metric_aux_e3")]
    output_state = _V5_SYNTHESIZE_STATE_DICT(*states, alpha=1.0)
    alpha_zero = _V5_SYNTHESIZE_STATE_DICT(*states, alpha=0.0)
    if not _V5_STATE_EQUAL(alpha_zero, states[0]):
        raise RuntimeError("All11 alpha=0 parent identity failed.")
    independently_recomputed = _V5_SYNTHESIZE_STATE_DICT(*states, alpha=1.0)
    if not _V5_STATE_EQUAL(output_state, independently_recomputed):
        raise RuntimeError("All11 alpha=1 independent recomputation failed.")
    stats = _arithmetic_stats(*states)
    if not (stats["task_vector"]["finite"] and 0.0 < stats["task_vector_over_baseline_drift"] < 0.1):
        raise RuntimeError("All11 task-vector safety gate failed during synthesis.")
    strict_load = _V5_STRICT_LOAD(output_state)
    output_payload = {
        "checkpoint_format_version": 2,
        "epoch": -1,
        "next_epoch": -1,
        "loss": 0.0,
        "model_state_dict": output_state,
        "temporal_memory": copy.deepcopy(payloads["parent"]["temporal_memory"]),
        "provenance": {
            "artifact_kind": "inference_only_metric_aux_task_arithmetic_all11",
            "formula": "released_m20 + 1.0 * (metric_aux_all11_e3 - baseline_all11_e3)",
            "alpha": 1.0,
            "arithmetic_dtype": "torch.float64_then_single_cast_to_parent_torch.float32",
            "input_checkpoint_sha256": before,
            "baseline_result_sha256": baseline_result_sha,
            "metric_aux_result_sha256": metric_aux_result_sha,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": core.sha256_file(RUNNER_PATH),
            "command_audit_sha256": command_audit_sha,
            "pair_audit_sha256": pair_audit_sha,
            "all11_source_count": 11,
            "training_optimizer_steps_each_arm": 264,
            "training_optimizer_steps_paired_total": 528,
            "validation_or_test_read": False,
        },
    }
    _V5_ATOMIC_TORCH_SAVE(output_payload, FINAL_CHECKPOINT_PATH)
    output_sha = core.sha256_file(FINAL_CHECKPOINT_PATH)
    reloaded = core.load_torch_checkpoint(FINAL_CHECKPOINT_PATH)
    if not _V5_STATE_EQUAL(reloaded["model_state_dict"], output_state):
        raise RuntimeError("Reloaded all11 checkpoint differs from synthesized state.")
    reload_strict = _V5_STRICT_LOAD(reloaded["model_state_dict"])
    after = {key: core.sha256_file(path) for key, path in paths.items()}
    if before != after:
        raise RuntimeError("An all11 synthesis input changed during synthesis.")
    manifest = {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-synthesis-manifest-v1",
        "created_utc": core.utc_now(),
        "status": "completed",
        "passed": True,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "command_audit_sha256": command_audit_sha,
        "pair_audit_sha256": pair_audit_sha,
        "input_checkpoint_sha256_before_after_equal": True,
        "input_checkpoint_sha256": before,
        "output_path": str(FINAL_CHECKPOINT_PATH.resolve()),
        "output_sha256": output_sha,
        "output_model_state_sha256": _V5_MODEL_STATE_SHA(output_state),
        "alpha_zero_parent_bitwise_identity": True,
        "alpha_one_formula_bitwise_recompute": True,
        "strict_cpu_model_load": strict_load,
        "reload_strict_cpu_model_load": reload_strict,
        "task_arithmetic_stats": stats,
        "temporal_memory_metadata_source": "released_m20",
        "optimizer_scheduler_rng_copied_from_pair": False,
        "candidate_count": 1,
        "evaluation_run": False,
        "validation_or_test_read": False,
        "default_submission_changed": False,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    core.write_new_json(SYNTHESIS_MANIFEST_PATH, manifest)
    print("all11 final checkpoint:", FINAL_CHECKPOINT_PATH)
    print("checkpoint sha256:", output_sha)
    print("synthesis manifest sha256:", core.sha256_file(SYNTHESIS_MANIFEST_PATH))
    return manifest


def _patch_core():
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.EXPECTED_PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.COMMAND_AUDIT_PATH = COMMAND_AUDIT_PATH
    core.FORMAL_ROOT = TRAINING_ROOT
    core.PAIR_AUDIT_PATH = PAIR_AUDIT_PATH
    core.__file__ = str(RUNNER_PATH)
    core.load_protocol = load_protocol
    core.load_command_audit = load_command_audit


_patch_core()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Run immutable train-only CPU preflight.")
    train = subparsers.add_parser("train", help="Run one or both authorization-gated all11 E3 arms.")
    train.add_argument("--run-id", default=None)
    train.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("audit-training", help="Audit the completed all11 pair on CPU.")
    subparsers.add_parser("synthesize", help="Create the one alpha=1 inference-only checkpoint on CPU.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit()
    if args.command == "train":
        return run_training(args.run_id, args.authorized)
    if args.command == "audit-training":
        return run_pair_audit(write=True)
    if args.command == "synthesize":
        return run_synthesis()
    raise RuntimeError("Unsupported all11 command: {}".format(args.command))


if __name__ == "__main__":
    main()

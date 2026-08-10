"""V3 audit-only recovery for the frozen metric-aux H2 grouped OOF study.

V3 reuses the immutable V2 paired 32-step training evidence.  It changes only
the failed real-batch gradient audit from an erroneous 128x128 spatial frame
to the same frozen 346x260 ``DATA.res`` used by formal training.  V2 remains
failed, and no V2 output is overwritten or retroactively promoted.
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

_V1_PRIVATE_NAME = "_metric_aux_h2_grouped_oof_v1_core_for_v2"
_PREVIOUS_V1_PRIVATE = sys.modules.get(_V1_PRIVATE_NAME)
_V2_PATH = EVC_ROOT / "run_metric_aux_h2_grouped_oof_v2.py"
_V2_SPEC = importlib.util.spec_from_file_location(
    "_metric_aux_h2_grouped_oof_v2_for_v3", _V2_PATH
)
if _V2_SPEC is None or _V2_SPEC.loader is None:
    raise ImportError("Unable to create a private V2 module for V3 recovery.")
v2 = importlib.util.module_from_spec(_V2_SPEC)
sys.modules[_V2_SPEC.name] = v2
_V2_SPEC.loader.exec_module(v2)
if _PREVIOUS_V1_PRIVATE is None:
    sys.modules.pop(_V1_PRIVATE_NAME, None)
else:
    sys.modules[_V1_PRIVATE_NAME] = _PREVIOUS_V1_PRIVATE

core = v2.core

PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_h2_grouped_oof_science_v3.json"
EXPECTED_PROTOCOL_SHA256 = "a4039cdba26ed1f950d62b40edc4b13c9868ac281c0dcd6b5a37e2062cd79875"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-h2-grouped-oof-audit-recovery-v3"
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260810_metric_aux_h2_grouped_oof_v3"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
PROBE_RESULT_PATH = OUTPUT_ROOT / "resource_probe" / "runtime_result.json"
PROBE_FAILURE_PATH = OUTPUT_ROOT / "resource_probe" / "probe_failure.json"
FORMAL_ROOT = OUTPUT_ROOT / "formal_training"
PAIR_AUDIT_PATH = FORMAL_ROOT / "pair_audit.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"

V2_PROTOCOL_SHA256 = "29de33fd9412d5b1fd2349acfb9d1dbfb9a109ace1288768d9100113b982ad06"
V2_RUNNER_SHA256 = "7bdb64a13d23a7e261e6e3128f908f4e0660224f4620338ac4c43412b1c422d1"
V2_TEST_SHA256 = "eaa2e4ba9822394006859c680097b384cdbd299356239d1ff2dd7083b3ab9dac"
V2_COMMAND_AUDIT_SHA256 = "b94a53c8d7da56ecc5fc717aaf52685784476de46730a64d4b5437c49bb0e57c"
V2_FAILURE_SHA256 = "cd0125cc3422a5bdc012c8a521a5f28e4b7fd4d1f2b9d064e514552d555efe99"
V2_BASELINE_RESULT_SHA256 = "49ea77d98fa8bb09ccc56bdfe501ac6e8a4aaac3e22fdb0d494cf7e277ef38cd"
V2_CANDIDATE_RESULT_SHA256 = "19ca70f5c042d7920d6d71ded7a3620708e5b88f3d15a1a7879469beaa587533"

_V2_LOAD_PROTOCOL = v2.load_protocol
_V2_COMMAND_AUDIT_PAYLOAD = v2.command_audit_payload
_V2_COMPARE_PAIR = v2.compare_pair_checkpoints


def _require_bound_file(record, expected_sha256, label):
    path = core.workspace_path(record["workspace_relative_path"])
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    actual = core.sha256_file(path)
    if actual != expected_sha256 or actual != record["sha256"]:
        raise RuntimeError("{} SHA-256 differs from the V3 contract.".format(label))
    return path


def _load_bound_json(record, expected_sha256, label):
    path = _require_bound_file(record, expected_sha256, label)
    payload, digest = core.load_json_snapshot(path)
    core._expect_equal(digest, expected_sha256, "{} snapshot".format(label))
    return payload, path


def _validate_overlay(overlay, overlay_sha256):
    core._expect_equal(overlay.get("schema"), EXPECTED_SCHEMA, "V3 protocol schema")
    core._expect_equal(
        overlay.get("status"),
        "frozen_before_any_v3_gpu_gradient_audit_formal_training_or_held_evaluation",
        "V3 protocol status",
    )
    core._expect_equal(overlay_sha256, EXPECTED_PROTOCOL_SHA256, "V3 protocol SHA-256")
    inheritance = overlay["inheritance"]
    core._expect_equal(
        inheritance["entire_v2_scientific_and_numeric_definition_inherited"],
        True,
        "V2 scientific inheritance",
    )
    _require_bound_file(inheritance["v2_protocol"], V2_PROTOCOL_SHA256, "V2 protocol")
    _require_bound_file(inheritance["v2_runner"], V2_RUNNER_SHA256, "V2 runner")
    _require_bound_file(inheritance["v2_tests"], V2_TEST_SHA256, "V2 tests")
    _require_bound_file(
        inheritance["v2_command_audit"],
        V2_COMMAND_AUDIT_SHA256,
        "V2 command audit",
    )
    failure, _ = _load_bound_json(
        inheritance["v2_failure_receipt"], V2_FAILURE_SHA256, "V2 failure receipt"
    )
    required_failure = {
        "status": "failed",
        "passed": False,
        "formal_training_started": False,
        "held_train_evaluation_started": False,
        "candidate_or_training_failure": False,
    }
    for key, expected in required_failure.items():
        core._expect_equal(failure[key], expected, "V2 failure {}".format(key))
    core._expect_equal(
        failure["completed_numeric_pair_audit"]["passed"],
        True,
        "V2 completed numeric pair gate",
    )
    core._expect_equal(
        failure["recovery_policy"]["repeat_paired_32_step_training_forbidden"],
        True,
        "V2 pair-repeat policy",
    )

    recovery = overlay["recovery_amendment"]
    expected_recovery = {
        "v2_attempt_remains_failed": True,
        "retroactive_v2_pass_forbidden": True,
        "v2_fresh_pair_training_reuse_allowed": True,
        "repeat_32_step_pair_training_forbidden": True,
        "scientific_candidate_training_evaluation_or_promotion_change": False,
        "claim_scope": "incremental_finetune_transfer_not_fold_clean_model_generalization",
        "shared_parent_pretraining_exposure": True,
    }
    for key, expected in expected_recovery.items():
        core._expect_equal(recovery[key], expected, "V3 recovery {}".format(key))

    evidence = overlay["v2_pair_evidence"]
    baseline, _ = _load_bound_json(
        evidence["baseline_training_result"],
        V2_BASELINE_RESULT_SHA256,
        "V2 baseline training result",
    )
    candidate, _ = _load_bound_json(
        evidence["candidate_training_result"],
        V2_CANDIDATE_RESULT_SHA256,
        "V2 candidate training result",
    )
    for label, result, variant in (
        ("baseline", baseline, "baseline"),
        ("candidate", candidate, "metric_aux"),
    ):
        core._expect_equal(result["protocol_sha256"], V2_PROTOCOL_SHA256, "{} protocol".format(label))
        core._expect_equal(result["runner_sha256"], V2_RUNNER_SHA256, "{} runner".format(label))
        core._expect_equal(result["variant"], variant, "{} variant".format(label))
        core._expect_equal(result["expected_source_names"], ["train_096.npz"], "{} source".format(label))
        core._expect_equal(result["expected_optimizer_steps"], 16, "{} optimizer steps".format(label))
        if "e3" in result["checkpoints"]:
            raise RuntimeError("V2 resource evidence unexpectedly contains E3.")
    checkpoint_records = (
        ("baseline_e1_checkpoint", baseline, "e1"),
        ("baseline_e2_checkpoint", baseline, "e2"),
        ("candidate_e1_checkpoint", candidate, "e1"),
        ("candidate_e2_checkpoint", candidate, "e2"),
    )
    for key, result, epoch_key in checkpoint_records:
        path = _require_bound_file(evidence[key], evidence[key]["sha256"], key)
        core._expect_equal(
            Path(result["checkpoints"][epoch_key]["path"]).resolve(),
            path.resolve(),
            "{} path".format(key),
        )
        core._expect_equal(
            result["checkpoints"][epoch_key]["sha256"],
            evidence[key]["sha256"],
            "{} result binding".format(key),
        )
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        summary, summary_sha = core.load_json_snapshot(result["run_summary"])
        core._expect_equal(summary_sha, result["run_summary_sha256"], "{} summary".format(label))
        core._expect_equal(summary["resolved_config"]["DATA"]["res"], [346, 260], "{} DATA.res".format(label))
        core._expect_equal(summary["sampling"]["sequence_count"], 8, "{} sequence count".format(label))
        core._expect_equal(summary["sampling"]["video_count"], 1, "{} video count".format(label))

    pair_gate = evidence["required_numeric_pair_gate"]
    core._expect_equal(pair_gate["passed"], True, "V2 numeric pair reuse gate")
    core._expect_equal(pair_gate["audit_version"], "numeric_near_identity_v2", "V2 numeric audit version")
    core._expect_equal(pair_gate["all_89_optimizer_steps_e1"], 8, "V2 E1 steps")
    core._expect_equal(pair_gate["all_89_optimizer_steps_e2"], 16, "V2 E2 steps")
    failure_pair = failure["completed_numeric_pair_audit"]
    numeric_pairs = {
        "e1_model_max_abs": failure_pair["e1_model"]["max_abs"],
        "e1_model_global_l2": failure_pair["e1_model"]["global_l2"],
        "e1_model_relative_l2": failure_pair["e1_model"]["relative_l2"],
        "e1_optimizer_max_abs": failure_pair["e1_optimizer"]["max_abs"],
        "e1_optimizer_global_l2": failure_pair["e1_optimizer"]["global_l2"],
        "e1_epoch_loss_abs_delta": failure_pair["e1_epoch_loss_abs_delta"],
        "e2_model_global_l2": failure_pair["e2_model"]["global_l2"],
        "e2_model_global_l2_over_e1_floor": failure_pair[
            "e2_model_global_l2_over_e1_numerical_floor"
        ],
    }
    for key, expected in numeric_pairs.items():
        core._expect_equal(pair_gate[key], expected, "V2 pair metric {}".format(key))

    resolution = overlay["audit_resolution_contract"]
    core._expect_equal(resolution["spatial_width"], 346, "audit spatial width")
    core._expect_equal(resolution["spatial_height"], 260, "audit spatial height")
    core._expect_equal(resolution["model_feature_width"], 16, "model feature width")
    core._expect_equal(resolution["sequence_length"], 16, "audit sequence length")
    core._expect_equal(resolution["input_channels"], 10, "audit input channels")
    for key in (
        "config",
        "formal_dataset_builder",
        "dataset",
        "frame_builder",
        "model",
        "loss",
        "model_loader",
    ):
        _require_bound_file(resolution[key], resolution[key]["sha256"], "resolution {}".format(key))
    if not all(word in resolution["event_handling"] for word in ("clamped", "masked", "filtered", "rescaled")):
        raise RuntimeError("V3 no-event-mutation contract is incomplete.")

    bounds = overlay["fixed_cpu_bounds_evidence"]
    core._expect_equal(bounds["source_name"], "train_096.npz", "bounds source")
    core._expect_equal(bounds["view_count"], 8, "bounds view count")
    core._expect_equal(bounds["formal_frame_shape_each"], [16, 10, 260, 346], "bounds frame shape")
    core._expect_equal(sum(bounds["event_counts"]), bounds["total_event_count"], "bounds event total")
    core._expect_equal(
        sum(bounds["outside_rejected_128x128_count_each"]),
        bounds["total_outside_rejected_128x128_count"],
        "bounds rejected-event total",
    )
    core._expect_equal(bounds["outside_formal_resolution_count_each"], [0] * 8, "formal bounds")
    if not math.isclose(
        bounds["rejected_128x128_fraction"],
        bounds["total_outside_rejected_128x128_count"] / bounds["total_event_count"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Rejected 128x128 fraction differs from exact counts.")
    probe = overlay["v3_resource_probe"]
    core._expect_equal(probe["new_training_optimizer_steps"], 0, "V3 new probe training")
    core._expect_equal(probe["model_output_shape_before_index"], [1, 16, 1, 260, 346], "V3 output shape")
    core._expect_equal(
        overlay["output_contract"]["workspace_relative_directory"],
        "experiments/20260810_metric_aux_h2_grouped_oof_v3",
        "V3 output root",
    )
    core._expect_equal(overlay["validation_or_test_read_allowed"], False, "V3 split policy")
    core._expect_equal(overlay["t32_allowed"], False, "V3 T32 policy")
    core._expect_equal(
        overlay["prior_persistence_formal_artifact_read_allowed"],
        False,
        "V3 persistence policy",
    )
    return failure, baseline, candidate


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "V3 protocol SHA-256 {} differs from frozen {}.".format(
                actual, EXPECTED_PROTOCOL_SHA256
            )
        )
    overlay, snapshot_sha = core.load_json_snapshot(PROTOCOL_PATH)
    core._expect_equal(snapshot_sha, actual, "V3 protocol snapshot")
    _validate_overlay(overlay, actual)
    effective, inherited_sha = _V2_LOAD_PROTOCOL()
    core._expect_equal(inherited_sha, V2_PROTOCOL_SHA256, "inherited V2 protocol")
    effective = copy.deepcopy(effective)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["recovery_amendment_v3"] = copy.deepcopy(overlay["recovery_amendment"])
    effective["v3_inheritance"] = copy.deepcopy(overlay["inheritance"])
    effective["v2_pair_evidence_v3"] = copy.deepcopy(overlay["v2_pair_evidence"])
    effective["audit_resolution_contract_v3"] = copy.deepcopy(
        overlay["audit_resolution_contract"]
    )
    effective["fixed_cpu_bounds_evidence_v3"] = copy.deepcopy(
        overlay["fixed_cpu_bounds_evidence"]
    )
    effective["v3_resource_probe"] = copy.deepcopy(overlay["v3_resource_probe"])
    effective["revision_history"] = list(effective["revision_history"]) + [
        {
            "recovery_protocol_sha256": actual,
            "reason": overlay["recovery_amendment"]["reason"],
            "v2_failure_sha256": V2_FAILURE_SHA256,
            "v2_attempt_remains_failed": True,
            "new_pair_training_optimizer_steps": 0,
        }
    ]
    effective["outputs"]["workspace_relative_directory"] = overlay[
        "output_contract"
    ]["workspace_relative_directory"]
    return effective, actual


def _load_v2_pair_results(protocol):
    evidence = protocol["v2_pair_evidence_v3"]
    baseline, baseline_sha = core.load_json_snapshot(
        core.workspace_path(evidence["baseline_training_result"]["workspace_relative_path"])
    )
    candidate, candidate_sha = core.load_json_snapshot(
        core.workspace_path(evidence["candidate_training_result"]["workspace_relative_path"])
    )
    core._expect_equal(baseline_sha, V2_BASELINE_RESULT_SHA256, "V2 baseline result")
    core._expect_equal(candidate_sha, V2_CANDIDATE_RESULT_SHA256, "V2 candidate result")
    for result in (baseline, candidate):
        core._expect_equal(result["protocol_sha256"], V2_PROTOCOL_SHA256, "V2 result protocol")
        core._expect_equal(result["runner_sha256"], V2_RUNNER_SHA256, "V2 result runner")
    return baseline, candidate


def _build_probe_dataset(protocol, probe_data_root):
    from dataset.temporal_memory import TemporalMemoryTrainDataset

    resolution = protocol["audit_resolution_contract_v3"]
    return TemporalMemoryTrainDataset(
        root=Path(probe_data_root) / "train",
        whole_t=8000,
        temporal_bin_size=50,
        context_bins=5,
        sequence_length=int(resolution["sequence_length"]),
        width=int(resolution["spatial_width"]),
        height=int(resolution["spatial_height"]),
        views_per_video=8,
        positive_frame_probability=0.75,
        random_seed=int(protocol["training"]["seed"]),
        log_count_clip=4.0,
        cache_all_videos=False,
        cache_video_count=2,
        dense_sampling_enabled=False,
        dense_event_count_cutoff=200000,
        dense_view_multiplier=2,
        density_bucket_boundaries=[],
        density_bucket_views=[],
        min_event_count_exclusive=200000,
        sparse_target_support_sampling_enabled=False,
        sparse_target_support_max_events=3,
        sparse_target_support_probability=0.75,
    )


def cpu_resolution_audit(protocol, probe_data_root):
    import numpy as np
    import torch
    from dataset.temporal_memory import temporal_memory_collate

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU bounds audit.")
    source_name = protocol["resource_probe"]["source"]
    source_path = (Path(probe_data_root) / "train" / source_name).resolve()
    official_source = core.require_official_train_source(
        core.official_train_root(protocol) / source_name, protocol, source_name
    )
    if not source_path.is_file() or not core.os.path.samefile(source_path, official_source):
        raise RuntimeError("V3 probe source is not the frozen official hard link.")
    source_sha_before = core.sha256_file(source_path)
    expected = protocol["fixed_cpu_bounds_evidence_v3"]
    core._expect_equal(source_sha_before, expected["source_sha256"], "V3 probe source SHA-256")
    dataset = _build_probe_dataset(protocol, probe_data_root)
    core._expect_equal(len(dataset), int(expected["view_count"]), "V3 bounds dataset length")
    core._expect_equal([path.name for path in dataset.file_paths], [source_name], "V3 bounds source membership")
    dataset.set_epoch(int(expected["epoch"]))
    resolution = protocol["audit_resolution_contract_v3"]
    width = int(resolution["spatial_width"])
    height = int(resolution["spatial_height"])
    records = []
    for view_index in range(len(dataset)):
        sample = dataset[view_index]
        batch = temporal_memory_collate([sample])
        frames = batch["frames"]
        time_indices = batch["event_time_indices"]
        event_x = batch["event_x"]
        event_y = batch["event_y"]
        labels = batch["labels"]
        target_ids = batch["target_ids"]
        event_timestamps = batch["event_timestamps"]
        event_count = int(labels.numel())
        lengths = [
            int(value.numel())
            for value in (time_indices, event_x, event_y, labels, target_ids, event_timestamps)
        ]
        if event_count <= 0 or len(set(lengths)) != 1:
            raise RuntimeError("V3 CPU bounds fields do not have one positive common length.")
        frame_shape = [int(value) for value in frames.shape]
        expected_shape = [
            int(resolution["sequence_length"]),
            int(resolution["input_channels"]),
            height,
            width,
        ]
        if frame_shape != expected_shape:
            raise RuntimeError("V3 CPU frame shape differs: {}".format(frame_shape))
        t_min, t_max = int(time_indices.min()), int(time_indices.max())
        x_min, x_max = int(event_x.min()), int(event_x.max())
        y_min, y_max = int(event_y.min()), int(event_y.max())
        outside_formal = int(
            torch.count_nonzero(
                (time_indices < 0)
                | (time_indices >= frames.shape[0])
                | (event_x < 0)
                | (event_x >= width)
                | (event_y < 0)
                | (event_y >= height)
            ).item()
        )
        outside_128 = int(
            torch.count_nonzero(
                (event_x < 0)
                | (event_x >= 128)
                | (event_y < 0)
                | (event_y >= 128)
            ).item()
        )
        record = {
            "view_index": view_index,
            "frame_shape": frame_shape,
            "event_count": event_count,
            "field_lengths": lengths,
            "event_time_index_range": [t_min, t_max],
            "event_x_range": [x_min, x_max],
            "event_y_range": [y_min, y_max],
            "outside_formal_resolution_count": outside_formal,
            "outside_rejected_128x128_count": outside_128,
        }
        records.append(record)
    checks = {
        "event_counts_exact": [record["event_count"] for record in records]
        == expected["event_counts"],
        "frame_shapes_exact": all(
            record["frame_shape"] == expected["formal_frame_shape_each"]
            for record in records
        ),
        "time_ranges_exact": all(
            record["event_time_index_range"] == expected["event_time_index_range_each"]
            for record in records
        ),
        "x_ranges_exact": all(
            record["event_x_range"] == expected["event_x_range_each"]
            for record in records
        ),
        "y_ranges_exact": all(
            record["event_y_range"] == expected["event_y_range_each"]
            for record in records
        ),
        "formal_bounds_zero_exact": [
            record["outside_formal_resolution_count"] for record in records
        ]
        == expected["outside_formal_resolution_count_each"],
        "rejected_128_counts_exact": [
            record["outside_rejected_128x128_count"] for record in records
        ]
        == expected["outside_rejected_128x128_count_each"],
        "all_fields_aligned": all(len(set(record["field_lengths"])) == 1 for record in records),
        "source_before_after_equal": source_sha_before == core.sha256_file(source_path),
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    if not all(checks.values()):
        raise RuntimeError("V3 CPU resolution audit failed: {}".format(checks))
    return {
        "source_name": source_name,
        "source_sha256": source_sha_before,
        "epoch": int(expected["epoch"]),
        "spatial_resolution": [width, height],
        "records": records,
        "total_event_count": sum(record["event_count"] for record in records),
        "total_outside_rejected_128x128_count": sum(
            record["outside_rejected_128x128_count"] for record in records
        ),
        "checks": checks,
        "passed": True,
    }


def command_audit_payload(protocol, protocol_sha256, assets, views):
    payload = _V2_COMMAND_AUDIT_PAYLOAD(protocol, protocol_sha256, assets, views)
    inherited_probe_commands = payload.pop("probe_commands", {})
    baseline, candidate = _load_v2_pair_results(protocol)
    pair = _V2_COMPARE_PAIR(baseline, candidate)
    if not pair["passed"]:
        raise RuntimeError("Bound V2 numeric pair no longer passes V2 gates.")
    synthetic = core.synthetic_metric_gradient_probe(protocol, device_name="cpu")
    bounds = cpu_resolution_audit(protocol, views["probe"]["root"])
    payload["schema"] = "ev-uav-metric-aux-h2-grouped-oof-command-audit-v3"
    payload["v3_audit_only_recovery"] = {
        "v2_failure_sha256": V2_FAILURE_SHA256,
        "v2_attempt_remains_failed": True,
        "v2_pair_training_reused": True,
        "new_pair_training_optimizer_steps": 0,
        "discarded_inherited_probe_training_command_count": len(inherited_probe_commands),
        "repeat_v2_pair_training_forbidden": True,
        "v2_numeric_pair_audit": pair,
        "synthetic_metric_gradient_probe": synthetic,
        "cpu_resolution_and_bounds_audit": bounds,
    }
    payload["probe_command"] = {
        "mode": "audit_only",
        "action": "corrected 346x260 real-batch gradient reachability from bound V2 candidate E1 checkpoint",
        "new_training_optimizer_steps": 0,
        "gpu_authorization_required": True,
    }
    payload["data_use_statement"] = (
        "Only frozen official train_088..train_098 identities and the train_096 "
        "epoch-1 probe arrays were read. No validation/test, T32 cache or "
        "persistence-formal artifact was opened; CUDA remained uninitialized."
    )
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the V3 CPU audit before any V3 GPU command.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-h2-grouped-oof-command-audit-v3":
        raise RuntimeError("V3 command-audit schema mismatch.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V3 command audit protocol identity mismatch.")
    if payload.get("runner_sha256") != core.sha256_file(Path(__file__).resolve()):
        raise RuntimeError("V3 runner changed after command audit.")
    if payload.get("gpu_or_cuda_initialized") is not False:
        raise RuntimeError("V3 command audit did not remain CPU-only.")
    recovery = payload.get("v3_audit_only_recovery", {})
    checks = {
        "v2_failure_bound": recovery.get("v2_failure_sha256") == V2_FAILURE_SHA256,
        "v2_remains_failed": recovery.get("v2_attempt_remains_failed") is True,
        "no_new_pair_training": recovery.get("new_pair_training_optimizer_steps") == 0,
        "pair_passed": recovery.get("v2_numeric_pair_audit", {}).get("passed") is True,
        "synthetic_passed": recovery.get("synthetic_metric_gradient_probe", {}).get("passed") is True,
        "bounds_passed": recovery.get("cpu_resolution_and_bounds_audit", {}).get("passed") is True,
    }
    if not all(checks.values()):
        raise RuntimeError("V3 command-audit recovery gates failed: {}".format(checks))
    return payload, digest


def corrected_real_batch_metric_gradient_probe(protocol, candidate_result, probe_data_root):
    import torch
    from dataset.temporal_memory import temporal_memory_collate
    from utils.component_hard_negative import (
        component_hard_negative_loss,
        target_frame_activation_loss,
    )
    from utils.temporal_memory_inference import load_temporal_memory_model

    cpu_bounds = cpu_resolution_audit(protocol, probe_data_root)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before the corrected CPU bounds gate completed.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the V3 real-batch gradient audit.")
    device = torch.device("cuda:0")
    torch.manual_seed(int(protocol["training"]["seed"]))
    torch.cuda.manual_seed_all(int(protocol["training"]["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint_path = Path(candidate_result["checkpoints"]["e1"]["path"])
    checkpoint_sha_before = core.sha256_file(checkpoint_path)
    core._expect_equal(
        checkpoint_sha_before,
        protocol["v2_pair_evidence_v3"]["candidate_e1_checkpoint"]["sha256"],
        "V3 candidate E1 checkpoint",
    )
    source_name = protocol["resource_probe"]["source"]
    source_path = (Path(probe_data_root) / "train" / source_name).resolve()
    source_sha_before = core.sha256_file(source_path)
    dataset = _build_probe_dataset(protocol, probe_data_root)
    dataset.set_epoch(1)
    resolution = protocol["audit_resolution_contract_v3"]
    model, _ = load_temporal_memory_model(
        checkpoint_path,
        device,
        context_bins=5,
        width=int(resolution["model_feature_width"]),
        sequence_length=int(resolution["sequence_length"]),
    )
    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable_parameters) != 89 or sum(p.numel() for p in trainable_parameters) != 1924716:
        raise RuntimeError("V3 real-batch model does not match the frozen full scope.")
    candidate = protocol["training"]["candidate"]
    chosen = None
    observed_output_shapes = []
    expected_output_shape = protocol["v3_resource_probe"]["model_output_shape_before_index"]
    for view_index in range(len(dataset)):
        sample = dataset[view_index]
        batch = temporal_memory_collate([sample])
        frames_cpu = batch["frames"]
        time_cpu = batch["event_time_indices"]
        x_cpu = batch["event_x"]
        y_cpu = batch["event_y"]
        cpu_preindex_checks = {
            "time_nonnegative": int(time_cpu.min()) >= 0,
            "time_below_sequence": int(time_cpu.max()) < int(frames_cpu.shape[0]),
            "x_nonnegative": int(x_cpu.min()) >= 0,
            "x_below_width": int(x_cpu.max()) < int(frames_cpu.shape[3]),
            "y_nonnegative": int(y_cpu.min()) >= 0,
            "y_below_height": int(y_cpu.max()) < int(frames_cpu.shape[2]),
        }
        if not all(cpu_preindex_checks.values()):
            raise RuntimeError("V3 CPU pre-index bounds failed: {}".format(cpu_preindex_checks))
        frames = frames_cpu.to(device).unsqueeze(0)
        time_indices = time_cpu.to(device)
        event_x = x_cpu.to(device)
        event_y = y_cpu.to(device)
        labels = batch["labels"].to(device)
        target_ids = batch["target_ids"].to(device)
        model_output = model(frames)
        if not torch.is_tensor(model_output):
            raise RuntimeError("V3 expected a tensor model output without confidence head.")
        output_shape = [int(value) for value in model_output.shape]
        observed_output_shapes.append(output_shape)
        if output_shape != expected_output_shape:
            raise RuntimeError(
                "V3 model output shape {} differs before event indexing.".format(output_shape)
            )
        logit_maps = model_output.squeeze(0)
        event_logits = logit_maps[time_indices, 0, event_y, event_x]
        scores = torch.sigmoid(event_logits)
        locations = torch.stack(
            (
                torch.zeros_like(event_x),
                event_x,
                event_y,
                time_indices * 50 + 1,
            ),
            dim=1,
        )
        target_loss, target_groups, missed_groups = target_frame_activation_loss(
            scores,
            labels,
            target_ids,
            locations,
            50,
            candidate["metric_activation_threshold"],
            candidate["metric_activation_temperature"],
        )
        component_loss, candidate_cells, hard_cells = component_hard_negative_loss(
            scores,
            labels,
            locations,
            candidate["metric_spatial_cell_size"],
            50,
            candidate["metric_min_cell_events"],
            candidate["metric_component_ratio"],
            candidate["metric_activation_threshold"],
            candidate["metric_activation_temperature"],
        )
        if int(target_groups) > 0 and int(candidate_cells) > 0 and int(hard_cells) > 0:
            chosen = {
                "view_index": view_index,
                "event_count": int(labels.numel()),
                "target_loss": target_loss,
                "component_loss": component_loss,
                "target_groups": int(target_groups),
                "missed_groups": int(missed_groups),
                "candidate_cells": int(candidate_cells),
                "hard_cells": int(hard_cells),
                "cpu_preindex_checks": cpu_preindex_checks,
                "model_output_shape": output_shape,
            }
            break
        del frames, time_indices, event_x, event_y, labels, target_ids
        del model_output, logit_maps, event_logits, scores, locations
        torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("No V3 epoch-1 view exercised both auxiliary losses.")

    target_gradients = torch.autograd.grad(
        chosen["target_loss"], trainable_parameters, retain_graph=True, allow_unused=True
    )
    component_gradients = torch.autograd.grad(
        chosen["component_loss"], trainable_parameters, retain_graph=False, allow_unused=True
    )

    def gradient_summary(gradients):
        squared = 0.0
        tensor_count = 0
        finite = True
        for gradient in gradients:
            if gradient is None:
                continue
            tensor_count += 1
            finite = finite and bool(torch.isfinite(gradient).all().item())
            squared += float(torch.sum(gradient.detach().double() ** 2).item())
        return math.sqrt(squared), tensor_count, finite

    target_norm, target_tensor_count, target_finite = gradient_summary(target_gradients)
    component_norm, component_tensor_count, component_finite = gradient_summary(component_gradients)
    before = [parameter.detach().clone() for parameter in trainable_parameters]
    optimizer = torch.optim.SGD(trainable_parameters, lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    combined_squared = 0.0
    combined_gradient_tensor_count = 0
    target_weight = float(candidate["metric_target_weight"])
    component_weight = float(candidate["metric_component_weight"])
    for parameter, target_gradient, component_gradient in zip(
        trainable_parameters, target_gradients, component_gradients
    ):
        combined = None
        if target_gradient is not None:
            combined = target_weight * target_gradient.detach()
        if component_gradient is not None:
            component_part = component_weight * component_gradient.detach()
            combined = component_part if combined is None else combined + component_part
        if combined is not None:
            parameter.grad = combined
            combined_gradient_tensor_count += 1
            combined_squared += float(torch.sum(combined.double() ** 2).item())
    optimizer.step()
    update_squared = 0.0
    update_tensor_count = 0
    for prior, parameter in zip(before, trainable_parameters):
        difference = parameter.detach() - prior
        if int(torch.count_nonzero(difference).item()) > 0:
            update_tensor_count += 1
        update_squared += float(torch.sum(difference.double() ** 2).item())
    combined_norm = math.sqrt(combined_squared)
    update_norm = math.sqrt(update_squared)
    torch.cuda.synchronize()
    checks = {
        "cpu_bounds_passed": cpu_bounds["passed"],
        "cpu_preindex_checks_passed": all(chosen["cpu_preindex_checks"].values()),
        "model_output_shape_exact": chosen["model_output_shape"] == expected_output_shape,
        "target_loss_finite_positive": math.isfinite(float(chosen["target_loss"].detach().cpu().item()))
        and float(chosen["target_loss"].detach().cpu().item()) > 0.0,
        "component_loss_finite_positive": math.isfinite(float(chosen["component_loss"].detach().cpu().item()))
        and float(chosen["component_loss"].detach().cpu().item()) > 0.0,
        "target_groups_positive": chosen["target_groups"] > 0,
        "candidate_cells_positive": chosen["candidate_cells"] > 0,
        "hard_cells_positive": chosen["hard_cells"] > 0,
        "target_parameter_gradient_finite_positive": target_finite and target_norm > 0.0,
        "component_parameter_gradient_finite_positive": component_finite and component_norm > 0.0,
        "target_gradient_reaches_parameters": target_tensor_count > 0,
        "component_gradient_reaches_parameters": component_tensor_count > 0,
        "combined_gradient_finite_positive": math.isfinite(combined_norm) and combined_norm > 0.0,
        "fresh_zero_state_optimizer_update_nonzero": math.isfinite(update_norm)
        and update_norm > 0.0
        and update_tensor_count > 0,
        "source_before_after_equal": source_sha_before == core.sha256_file(source_path),
        "checkpoint_before_after_equal": checkpoint_sha_before
        == core.sha256_file(checkpoint_path),
    }
    if not all(checks.values()):
        raise RuntimeError("V3 real-batch gradient audit failed: {}".format(checks))
    return {
        "source_name": source_name,
        "source_sha256": source_sha_before,
        "checkpoint_sha256": checkpoint_sha_before,
        "epoch": 1,
        "view_index": chosen["view_index"],
        "event_count": chosen["event_count"],
        "spatial_resolution": [
            int(resolution["spatial_width"]),
            int(resolution["spatial_height"]),
        ],
        "model_feature_width": int(resolution["model_feature_width"]),
        "observed_model_output_shapes": observed_output_shapes,
        "target_loss": float(chosen["target_loss"].detach().cpu().item()),
        "component_loss": float(chosen["component_loss"].detach().cpu().item()),
        "target_group_count": chosen["target_groups"],
        "missed_target_group_count": chosen["missed_groups"],
        "candidate_cell_count": chosen["candidate_cells"],
        "hard_cell_count": chosen["hard_cells"],
        "target_parameter_gradient_norm": target_norm,
        "target_parameter_gradient_tensor_count": target_tensor_count,
        "component_parameter_gradient_norm": component_norm,
        "component_parameter_gradient_tensor_count": component_tensor_count,
        "combined_weighted_gradient_norm": combined_norm,
        "combined_weighted_gradient_tensor_count": combined_gradient_tensor_count,
        "fresh_optimizer_update_norm": update_norm,
        "fresh_optimizer_updated_tensor_count": update_tensor_count,
        "cpu_resolution_and_bounds_audit": cpu_bounds,
        "checks": checks,
        "passed": True,
    }


def run_probe(authorized=False):
    core.require_gpu_authorization(authorized)
    core.require_idle_gpu()
    protocol, _ = load_protocol()
    command_audit, command_audit_sha = load_command_audit()
    if PROBE_RESULT_PATH.exists() or PROBE_FAILURE_PATH.exists():
        raise FileExistsError("Refusing to overwrite completed V3 probe evidence.")
    _, candidate = _load_v2_pair_results(protocol)
    try:
        real_batch = corrected_real_batch_metric_gradient_probe(
            protocol, candidate, command_audit["data_views"]["probe"]["root"]
        )
        gpu_postflight = core.require_idle_gpu()
    except Exception as error:
        failure = {
            "schema": "ev-uav-metric-aux-audit-only-probe-failure-v3",
            "created_utc": core.utc_now(),
            "status": "failed",
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "v2_failure_sha256": V2_FAILURE_SHA256,
            "v2_attempt_remains_failed": True,
            "v2_pair_training_reused": True,
            "new_pair_training_optimizer_steps": 0,
            "formal_training_started": False,
            "held_train_evaluation_started": False,
            "command_audit_sha256": command_audit_sha,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        }
        core.write_new_json(PROBE_FAILURE_PATH, failure)
        raise
    checks = {
        "v2_attempt_remains_failed": True,
        "v2_numeric_pair_reused_and_passed": command_audit[
            "v3_audit_only_recovery"
        ]["v2_numeric_pair_audit"]["passed"],
        "synthetic_cpu_gate_reused_and_passed": command_audit[
            "v3_audit_only_recovery"
        ]["synthetic_metric_gradient_probe"]["passed"],
        "corrected_cpu_bounds_gate_passed": command_audit[
            "v3_audit_only_recovery"
        ]["cpu_resolution_and_bounds_audit"]["passed"],
        "corrected_real_batch_gradient_gate_passed": real_batch["passed"],
        "no_new_pair_training_optimizer_steps": True,
    }
    payload = {
        "schema": "ev-uav-metric-aux-audit-only-probe-result-v3",
        "created_utc": core.utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed": all(checks.values()),
        "v2_failure_sha256": V2_FAILURE_SHA256,
        "v2_attempt_remains_failed": True,
        "v2_pair_training_reused": True,
        "v2_baseline_training_result_sha256": V2_BASELINE_RESULT_SHA256,
        "v2_candidate_training_result_sha256": V2_CANDIDATE_RESULT_SHA256,
        "new_pair_training_optimizer_steps": 0,
        "corrected_real_batch_metric_gradient_probe": real_batch,
        "gpu_idle_postflight": gpu_postflight,
        "command_audit_sha256": command_audit_sha,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "shared_parent_pretraining_exposure": True,
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
    }
    core.write_new_json(PROBE_RESULT_PATH, payload)
    if not payload["passed"]:
        raise RuntimeError("V3 audit-only probe failed: {}".format(checks))
    print("V3 audit-only resource probe passed:", PROBE_RESULT_PATH)
    return payload


def require_probe_passed():
    payload, digest = core.load_json_snapshot(PROBE_RESULT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-audit-only-probe-result-v3":
        raise RuntimeError("V3 resource-probe schema mismatch.")
    if payload.get("passed") is not True or not all(payload.get("checks", {}).values()):
        raise RuntimeError("V3 audit-only resource probe has not passed every gate.")
    if payload.get("v2_attempt_remains_failed") is not True:
        raise RuntimeError("V3 receipt retroactively changed V2.")
    if payload.get("new_pair_training_optimizer_steps") != 0:
        raise RuntimeError("V3 receipt contains unauthorized pair training.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V3 resource-probe protocol identity mismatch.")
    if payload.get("runner_sha256") != core.sha256_file(Path(__file__).resolve()):
        raise RuntimeError("V3 resource-probe runner identity mismatch.")
    command_audit, command_audit_sha = load_command_audit()
    if payload.get("command_audit_sha256") != command_audit_sha:
        raise RuntimeError("V3 resource probe command-audit identity mismatch.")
    if command_audit["v3_audit_only_recovery"]["v2_failure_sha256"] != V2_FAILURE_SHA256:
        raise RuntimeError("V3 resource probe lost V2 failure binding.")
    return payload, digest


def _patch_core_for_v3():
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
    core.compare_pair_checkpoints = _V2_COMPARE_PAIR
    core.require_probe_passed = require_probe_passed


_patch_core_for_v3()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="CPU-only V3 audit and bounds preflight.")
    probe = subparsers.add_parser("probe", help="Run only the corrected V3 GPU gradient audit.")
    probe.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    train = subparsers.add_parser("train", help="Run all or one V3 formal E3 training.")
    train.add_argument("--run-id", default=None)
    train.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("audit-training", help="CPU-audit completed V3 formal pairs.")
    evaluate = subparsers.add_parser("evaluate", help="Run V3 held-train evaluation.")
    evaluate.add_argument("--eval-id", default=None)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("report", help="Apply inherited held-only double-anchor gates.")
    all_after_probe = subparsers.add_parser("all-after-probe")
    all_after_probe.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
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

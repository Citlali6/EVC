"""Frozen all-11 final refit for the promoted H2 pyramid recovery V2.

This runner exposes no source, threshold, model, loss, epoch, or optimizer
knobs.  It performs train-only final refitting and package audits; it has no
validation, test, scoring, or submission command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

import crossfit_component_reranker as crossfit
import run_h2_atomic_component_deletion_v3 as atomic
import run_h2_multiscale_temporal_pyramid_formal as stage1_parent
import run_h2_multiscale_temporal_pyramid_probe as probe
import run_h2_pyramid_component_recovery_v2 as promoted
from model.h2_multiscale_temporal_pyramid_expert import (
    FrozenM20MultiScalePyramidAdapter,
    pyramid_expert_parameter_count,
)
from model.h2_pyramid_component_recovery import (
    H2PyramidComponentRecoveryHead,
    NODE_FEATURE_DIM,
    component_recovery_parameter_count,
)
from utils.h2_multiscale_pyramid_loss import (
    PyramidDualState,
    multiscale_pyramid_constrained_loss,
    validate_pyramid_step_diagnostics,
)
from utils.h2_pyramid_component_recovery import (
    BreakpointRecord,
    exact_risk_controlled_breakpoint,
    restore_whole_components_bitwise,
)
from utils.target_preserving_residual import (
    H2_EVENT_COUNT_CUTOFF,
    H2_POLARITY_MINORITY_CUTOFF,
    input_only_routed_scores,
    use_h2_residual_refiner,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TRAIN_ROOT = (WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train").resolve()
PROTOCOL_PATH = (
    ROOT
    / "protocols"
    / "h2_pyramid_component_recovery_v2_all11_final_refit_science_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = "85ff0ef0e363ae0bec59580962ab26f1de6810840d5b664c4191978f87a3eb0b"
SOURCE_PROTOCOL_PATH = (
    ROOT / "protocols" / "h2_spatiotemporal_residual_refiner_oof_science_v1.json"
)
EXPECTED_SOURCE_PROTOCOL_SHA256 = (
    "7edec461f2ccc8047156f08c57389319a5defd59d0afcea69cbfcf32e81d2207"
)
EXPECTED_PROMOTED_PROTOCOL_SHA256 = (
    "4c4c260b66bf4c77fb314432bd2c72432a3273917347a8f5bf943d8489933c70"
)
EXPECTED_PROMOTED_RUNNER_SHA256 = (
    "3fa4b7a66339366414a11a0a8b96e53bbd83f6878a9f3f8241dedc6028e47ad5"
)
EXPECTED_INNER_RESULT_SHA256 = (
    "58065abca0d92f2b623b86d3514ff92f84d90f512edb71ed68d89a77235ea671"
)
EXPECTED_OUTER_RESULT_SHA256 = (
    "7a96b13e2cff340d39b145b142565f648d5eaff8ac442d320c8fbf7a33709754"
)
EXPECTED_DECISION_SHA256 = (
    "298a8ad299a68c66987b849685fbd3993fdf1b9bef0f2bda7137b0d245a1f334"
)
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EXPECTED_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
EXPECTED_WRAPPER_SHA256 = (
    "862bb0ff47235d99740a2ad19ac577ec78762922d9e12810d669eecfb3994a8e"
)

EXPECTED_EXECUTION_DEPENDENCIES = {
    "model/h2_multiscale_temporal_pyramid_expert.py": "4d4ea4a365be49ad1b6c7cf1c7c96c2369caf3e12841bbcd781cf109105a6a98",
    "utils/h2_multiscale_pyramid_loss.py": "f74e145b04b25f2e7478f5c8fd370bc4e9d96123ef6f75ea5833acc210d2c5e9",
    "model/h2_pyramid_component_recovery.py": "f55224d30b9d8ddeab463dd433f77e0dc961d45d2ecf60cf255a6ddcd0447352",
    "utils/h2_pyramid_component_recovery.py": "cd8fb85dda17ef177c174b9051a6d089c7a703d67f393dbf80aca656012f9f78",
    "utils/target_preserving_residual.py": "e4377446fb8834a6e2860da228fcabcdeaa46f04ab036dddb1217c9f1e364b0e",
    "run_h2_multiscale_temporal_pyramid_probe.py": "2cd894d046a1433be7b8fc06b95bb616ed82b53ce2078aac082d08ba80eb1df4",
    "run_h2_multiscale_temporal_pyramid_formal.py": "33ca6078c810f52c8a549376bb8d862bae1068561bd7ccd8e5c1479e1f8fe89f",
    "run_h2_atomic_component_deletion_v3.py": "8c04e72306f8ed9a5944491fe13567518bb6d62f5d384eab0b7879bb34e6f884",
    "run_h2_pyramid_component_recovery_v2.py": EXPECTED_PROMOTED_RUNNER_SHA256,
    "crossfit_component_reranker.py": "ec4d0af1abf8dab88037d8731ad1b142aa8cdb76511287d866113865d70e9561",
    "utils/postprocess.py": "a43e2a22947f5307cdb01b810a4f1dc8d4fb624b14bd068baf1e9087ca5d1aa4",
    "utils/atomic_component_deletion.py": "b9d86aa7203686c512ecfd9d143074c06e367fb6b1b7cb658a262b8c04f148c5",
    "utils/eval.py": "cb9de56b54a7153d80a2fd32432188665e830e0a9b945965f2cbd6765bb354e5",
    "utils/challenge_eval.py": "a09e9a4b74abb06864e4551778addd624e53654041c98a41dbe61a1f67fbc952",
    "model/temporal_memory_net.py": "d90c550bc2928482684c1216f52c4752700c27b79935a9552208bc2bca00f32f",
    "dataset/temporal_frame.py": "1f6bed24ebdbae671459028d8f2ff37f179dea9bb2ef23191d45c9e010ac83ee",
    "utils/h2_pyramid_recovery_v2_inference.py": EXPECTED_WRAPPER_SHA256,
}

ALL11 = tuple("train_{:03d}.npz".format(value) for value in range(88, 99))
G1 = ALL11[:4]
G2 = ALL11[4:7]
G3 = ALL11[7:]
GROUPS = {"G1": G1, "G2": G2, "G3": G3}
OOF_FOLDS = (
    ("hold_G1", G2 + G3, G1, 56),
    ("hold_G2", G1 + G3, G2, 64),
    ("hold_G3", G1 + G2, G3, 56),
)
THRESHOLD = 0.719
SEED = 67
STAGE1_EPOCHS = 2
STAGE1_STEPS = 88
RECOVERY_EPOCHS = 8
FINAL_HEAD_STEPS = 88
TOTAL_RECOVERY_STEPS = 264
MAX_CUDA_MIB = 2048.0
GPU_FLAG = "--root-authorized-gpu"

OUTPUT_ROOT = (
    WORKSPACE / "experiments" / "20260811_h2_pyramid_component_recovery_v2_all11_final"
)
CPU_AUDIT_PATH = OUTPUT_ROOT / "cpu_audit" / "report.json"
STAGE1_ROOT = OUTPUT_ROOT / "stage1_all11"
STAGE1_CHECKPOINT = STAGE1_ROOT / "final_stage1.pt"
STAGE1_RECEIPT = STAGE1_ROOT / "training_receipt.json"
FEATURE_ROOT = OUTPUT_ROOT / "all11_input_features"
FEATURE_MANIFEST = FEATURE_ROOT / "manifest.json"
STAGE2_ROOT = OUTPUT_ROOT / "stage2_all11"
OOF_RESULT = STAGE2_ROOT / "oof_calibration.json"
FINAL_PACKAGE = OUTPUT_ROOT / "final_package" / "h2_pyramid_recovery_v2_all11.pt"
FINAL_MANIFEST = OUTPUT_ROOT / "final_package" / "manifest.json"
FINAL_AUDIT = OUTPUT_ROOT / "final_package" / "cpu_audit.json"

PROMOTED_INNER_RESULT = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_pyramid_component_recovery_v2"
    / "nested_recovery"
    / "hold_g1"
    / "inner_result.json"
)
PROMOTED_OUTER_RESULT = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_pyramid_component_recovery_v2"
    / "held_train_evaluation"
    / "fresh_hold_g1"
    / "paired_evaluation.json"
)
PROMOTED_DECISION = PROMOTED_OUTER_RESULT.parent / "branch_decision.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    return promoted.sha256_file(Path(path))


def read_json(path):
    return promoted.read_json(Path(path))


def write_json_exclusive(path, payload):
    return promoted.write_json_exclusive(Path(path), payload)


def write_torch_exclusive(path, payload):
    return promoted.write_torch_exclusive(Path(path), payload)


def verify_sidecar(path):
    return promoted.verify_sidecar(Path(path))


def state_sha256(state):
    return atomic.state_sha256(state)


def _expect(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_protocol():
    _expect(sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256, "final protocol changed")
    protocol = read_json(PROTOCOL_PATH)
    _expect(
        protocol.get("schema")
        == "ev-uav-h2-pyramid-component-recovery-v2-all11-final-refit-science-v1",
        "unexpected final protocol schema",
    )
    _expect(
        protocol.get("status")
        == "frozen_after_fresh_G1_train_only_promotion_before_any_final_refit_GPU_or_final_checkpoint",
        "final protocol is not frozen",
    )
    _expect(protocol["resource_budget"]["GPU_authorized_now"] is False, "protocol must stay GPU-false")
    _expect(tuple(protocol["data_scope"]["all11_sources_in_order"]) == ALL11, "all11 order changed")
    _expect(tuple(protocol["data_scope"]["groups"]["G1"]) == G1, "G1 changed")
    _expect(tuple(protocol["data_scope"]["groups"]["G2"]) == G2, "G2 changed")
    _expect(tuple(protocol["data_scope"]["groups"]["G3"]) == G3, "G3 changed")
    _expect(protocol["Stage1_all11"]["optimizer_steps"] == STAGE1_STEPS, "Stage1 steps changed")
    _expect(
        protocol["Stage2_all11"]["total_recovery_optimizer_steps_excluding_any_mechanical_probe"]
        == TOTAL_RECOVERY_STEPS,
        "Stage2 steps changed",
    )
    _expect(protocol["promotion_evidence"]["weight_reuse_for_final_refit_allowed"] is False, "weight reuse changed")
    _expect(protocol["data_scope"]["validation_read_allowed"] is False, "validation access changed")
    _expect(protocol["data_scope"]["test_read_allowed"] is False, "test access changed")
    _expect(protocol["postprocess"]["effective_C00_sha256"] == EXPECTED_C00_SHA256, "C00 changed")
    _expect(protocol["deployment_route"]["event_count_cutoff_exclusive"] == 200000, "route count changed")
    _expect(protocol["deployment_route"]["polarity_minority_cutoff_inclusive"] == 0.2, "route polarity changed")

    fixed = {
        SOURCE_PROTOCOL_PATH: EXPECTED_SOURCE_PROTOCOL_SHA256,
        ROOT / "protocols" / "h2_pyramid_component_recovery_v2_science_v1.json": EXPECTED_PROMOTED_PROTOCOL_SHA256,
        atomic.M20_PATH: EXPECTED_M20_SHA256,
        PROMOTED_INNER_RESULT: EXPECTED_INNER_RESULT_SHA256,
        PROMOTED_OUTER_RESULT: EXPECTED_OUTER_RESULT_SHA256,
        PROMOTED_DECISION: EXPECTED_DECISION_SHA256,
    }
    for path, expected in fixed.items():
        _expect(sha256_file(path) == expected, "frozen evidence changed: {}".format(path))
    for relative, expected in EXPECTED_EXECUTION_DEPENDENCIES.items():
        _expect(sha256_file(ROOT / relative) == expected, "execution dependency changed: {}".format(relative))

    inner = read_json(PROMOTED_INNER_RESULT)
    outer = read_json(PROMOTED_OUTER_RESULT)
    decision = read_json(PROMOTED_DECISION)
    _expect(inner.get("inner_passed") is True and all(inner.get("gates", {}).values()), "promotion inner gates changed")
    _expect(outer.get("outer_passed") is True and all(outer.get("outer_gates", {}).values()), "promotion outer gates changed")
    _expect(decision.get("decision") == "promote_after_fresh_G1_but_stop_no_other_fold_or_tuning", "promotion decision changed")
    _expect(inner.get("validation_or_test_read") is False, "promotion inner val/test receipt changed")
    _expect(outer.get("validation_or_test_read") is False, "promotion outer val/test receipt changed")
    _expect(decision.get("validation_or_test_read") is False, "promotion decision val/test receipt changed")
    return protocol


def source_manifest():
    return read_json(SOURCE_PROTOCOL_PATH)["h2_sources"]


def verify_source(source_name, manifest):
    _expect(source_name in manifest and source_name in ALL11, "source outside frozen all11")
    path = (TRAIN_ROOT / source_name).resolve()
    _expect(path.parent == TRAIN_ROOT, "source escaped official train root")
    _expect(sha256_file(path) == manifest[source_name]["sha256"], "source SHA changed: {}".format(source_name))
    return path


def build_c00():
    cfg, effective = promoted.build_c00()
    _expect(crossfit.sha256_json(effective) == EXPECTED_C00_SHA256, "effective C00 changed")
    return cfg, effective


def seed_all(seed=SEED):
    promoted.seed_all(seed)


def formal_cpu_audit_gate():
    digest = verify_sidecar(CPU_AUDIT_PATH)
    payload = read_json(CPU_AUDIT_PATH)
    _expect(payload.get("passed") is True, "final CPU audit did not pass")
    _expect(payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "CPU audit protocol changed")
    _expect(payload.get("runner_sha256") == sha256_file(Path(__file__)), "runner changed after CPU audit")
    _expect(payload.get("dataset_arrays_read") is False, "CPU audit read dataset arrays")
    _expect(payload.get("validation_or_test_read") is False, "CPU audit read val/test")
    _expect(payload.get("CUDA_initialized") is False, "CPU audit initialized CUDA")
    return payload, digest


def require_gpu(args):
    if not bool(getattr(args, "root_authorized_gpu", False)):
        raise PermissionError("GPU command requires {}".format(GPU_FLAG))
    formal_cpu_audit_gate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")


def source_cache_path(source_name):
    return FEATURE_ROOT / source_name.replace(".npz", "_input_scores.pt")


def node_cache_path(source_name):
    return FEATURE_ROOT / source_name.replace(".npz", "_node_features.pt")


def load_score_cache(source_name):
    payload = torch.load(source_cache_path(source_name), map_location="cpu", weights_only=False)
    _expect(payload.get("source_name_for_provenance_only") == source_name, "score cache provenance changed")
    _expect(payload.get("contains_labels_or_target_ids") is False, "score cache contains truth")
    _expect(payload.get("runner_sha256") == sha256_file(Path(__file__)), "score cache runner changed")
    return payload


def load_node_cache(source_name):
    payload = torch.load(node_cache_path(source_name), map_location="cpu", weights_only=False)
    _expect(payload.get("source_name_for_provenance_only") == source_name, "node cache provenance changed")
    _expect(payload.get("contains_labels_or_target_ids") is False, "node cache contains truth")
    _expect(payload.get("runner_sha256") == sha256_file(Path(__file__)), "node cache runner changed")
    _expect(len(payload["node_features"]) == int(payload["component_count"]), "node cache alignment changed")
    return payload


def cpu_audit(_args):
    if CPU_AUDIT_PATH.parent.exists():
        raise FileExistsError("refusing to overwrite final CPU audit")
    if any(path.exists() for path in (STAGE1_ROOT, FEATURE_ROOT, STAGE2_ROOT, FINAL_PACKAGE.parent)):
        raise RuntimeError("final refit output exists before CPU audit")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before final CPU audit")
    protocol = load_protocol()
    manifest = source_manifest()
    _expect(tuple(manifest) == ALL11, "source manifest order changed")
    _expect(H2_EVENT_COUNT_CUTOFF == 200000, "route count constant changed")
    _expect(H2_POLARITY_MINORITY_CUTOFF == 0.20, "route polarity constant changed")
    _expect(not use_h2_residual_refiner(200000, np.zeros(200000, dtype=np.uint8)), "count boundary changed")
    balanced = np.tile(np.asarray([0, 1], dtype=np.uint8), 100001)[:200001]
    _expect(use_h2_residual_refiner(200001, balanced), "H2 positive boundary changed")
    base = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    candidate = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    routed = input_only_routed_scores(base, candidate, np.zeros(3, dtype=np.uint8))
    _expect(np.array_equal(routed, base), "non-H2 bitwise identity changed")
    from utils.h2_pyramid_recovery_v2_inference import (
        apply_atomic_stage2,
        use_h2_pyramid_recovery_v2,
    )

    _expect(
        not use_h2_pyramid_recovery_v2(200000, np.zeros(200000, dtype=np.uint8)),
        "deployment wrapper count boundary changed",
    )
    _expect(
        use_h2_pyramid_recovery_v2(200001, balanced),
        "deployment wrapper H2 positive boundary changed",
    )
    restored = restore_whole_components_bitwise(
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
        (np.asarray([0, 1], dtype=np.int64),),
        np.asarray([True]),
    )
    _expect(np.array_equal(restored, np.asarray([0.8, 0.9, 0.3], dtype=np.float32)), "atomic restore changed")
    wrapper_restored, wrapper_decisions = apply_atomic_stage2(
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
        (np.asarray([0, 1], dtype=np.int64),),
        np.asarray([0.75], dtype=np.float64),
        0.5,
    )
    _expect(
        bool(wrapper_decisions[0])
        and np.array_equal(
            wrapper_restored,
            np.asarray([0.8, 0.9, 0.3], dtype=np.float32),
        ),
        "deployment wrapper atomic restore changed",
    )
    head = H2PyramidComponentRecoveryHead().cpu().eval()
    _expect(component_recovery_parameter_count(head) == 14081, "head parameter count changed")
    features = torch.zeros((2, 3, NODE_FEATURE_DIM), dtype=torch.float32)
    mask = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)
    with torch.no_grad():
        values = head(features, mask)
    _expect(values.shape == (2,) and torch.isfinite(values).all(), "head CPU forward failed")
    released_m20, _ = atomic.build_released_m20(torch.device("cpu"))
    adapter = FrozenM20MultiScalePyramidAdapter(
        released_m20, context_bins=5
    ).cpu()
    _expect(
        pyramid_expert_parameter_count(adapter) == 3381,
        "Stage1 CPU parameter count changed",
    )
    _expect(
        not any(parameter.requires_grad for parameter in released_m20.parameters()),
        "CPU-loaded released M20 is not frozen",
    )
    expected_steps = sum(item[3] for item in OOF_FOLDS) + FINAL_HEAD_STEPS
    _expect(expected_steps == TOTAL_RECOVERY_STEPS, "recovery schedule changed")
    coverage = [name for _, _, held, _ in OOF_FOLDS for name in held]
    _expect(tuple(coverage) == ALL11 and len(set(coverage)) == len(ALL11), "OOF coverage changed")
    for _, fit, held, steps in OOF_FOLDS:
        _expect(not set(fit).intersection(held), "OOF fold overlaps")
        _expect(set(fit).union(held) == set(ALL11), "OOF fold is incomplete")
        _expect(steps == RECOVERY_EPOCHS * len(fit), "OOF step count changed")
    _expect(torch.cuda.is_initialized() is False, "CPU audit initialized CUDA")
    payload = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-all11-final-cpu-audit-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_status": protocol["status"],
        "all11_sources": list(ALL11),
        "Stage1_optimizer_steps": STAGE1_STEPS,
        "Stage2_optimizer_steps": TOTAL_RECOVERY_STEPS,
        "OOF_folds": [
            {"fold": name, "fit": list(fit), "held": list(held), "steps": steps}
            for name, fit, held, steps in OOF_FOLDS
        ],
        "route_boundaries_passed": True,
        "non_H2_bitwise_identity_passed": True,
        "atomic_restore_passed": True,
        "deployment_wrapper_route_and_atomic_restore_passed": True,
        "head_CPU_forward_passed": True,
        "Stage1_CPU_strict_M20_load_and_parameter_count_passed": True,
        "resource_budget": protocol["resource_budget"],
        "dataset_arrays_read": False,
        "old_promoted_weights_loaded": False,
        "validation_or_test_read": False,
        "CUDA_initialized": False,
        "GPU_authorized": False,
        "passed": True,
    }
    del adapter, released_m20, head
    digest = write_json_exclusive(CPU_AUDIT_PATH, payload)
    print(json.dumps({"stage": "CPU_audit_complete", "sha256": digest, "passed": True}, indent=2))


def stage1_gate():
    checkpoint_sha = verify_sidecar(STAGE1_CHECKPOINT)
    receipt_sha = verify_sidecar(STAGE1_RECEIPT)
    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location="cpu", weights_only=False)
    receipt = read_json(STAGE1_RECEIPT)
    runner_sha = sha256_file(Path(__file__))
    _expect(checkpoint.get("schema") == "ev-uav-h2-pyramid-recovery-v2-all11-stage1-v1", "Stage1 schema changed")
    _expect(checkpoint.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "Stage1 protocol changed")
    _expect(checkpoint.get("runner_sha256") == runner_sha, "runner changed after Stage1")
    _expect(tuple(checkpoint.get("fit_sources", ())) == ALL11, "Stage1 fit set changed")
    _expect(checkpoint.get("optimizer_steps") == STAGE1_STEPS, "Stage1 step count changed")
    _expect(checkpoint.get("fresh_initialization") is True, "Stage1 is not fresh")
    _expect(checkpoint.get("promoted_checkpoint_reused") is False, "promoted Stage1 was reused")
    _expect(checkpoint.get("validation_or_test_read") is False, "Stage1 val/test receipt changed")
    _expect(state_sha256(checkpoint["expert_state_dict"]) == checkpoint["expert_state_sha256"], "Stage1 state hash changed")
    _expect(receipt.get("checkpoint_sha256") == checkpoint_sha, "Stage1 receipt checkpoint changed")
    _expect(receipt.get("runner_sha256") == runner_sha, "Stage1 receipt runner changed")
    _expect(receipt.get("all_expert_parameter_tensors_updated") is True, "Stage1 update audit failed")
    _expect(receipt.get("validation_or_test_read") is False, "Stage1 receipt val/test changed")
    return checkpoint, receipt, checkpoint_sha, receipt_sha


def train_stage1(args):
    require_gpu(args)
    load_protocol()
    cfg, effective_c00 = build_c00()
    if STAGE1_ROOT.exists():
        raise FileExistsError("refusing to overwrite all11 Stage1")
    if FEATURE_ROOT.exists() or STAGE2_ROOT.exists() or FINAL_PACKAGE.parent.exists():
        raise RuntimeError("later final-refit output exists before Stage1")
    manifest = source_manifest()
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_all11_final_stage1"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_all()
        m20, _ = atomic.build_released_m20(device)
        m20_before = state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        _expect(pyramid_expert_parameter_count(adapter) == 3381, "Stage1 parameter count changed")
        _expect(not any(parameter.requires_grad for parameter in m20.parameters()), "M20 is not frozen")
        adapter.train()
        optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=0.0003, weight_decay=0.0001)
        dual = PyramidDualState()
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in adapter.expert.named_parameters()
        }
        source_hashes_before = {}
        records = []
        source_records = []
        step = 0
        initial_identity = None
        for epoch_zero in range(STAGE1_EPOCHS):
            for source_position, source_name in enumerate(ALL11):
                path = verify_source(source_name, manifest)
                expected = manifest[source_name]
                source_hashes_before.setdefault(source_name, expected["sha256"])
                video, polarities, locations4, labels, target_ids = stage1_parent.load_input_and_truth(
                    path, expected["event_count"]
                )
                memory = probe.full_stream_memory(m20, video, device)
                observations, base_raw, decoded_bins = probe.stream_observations_and_scores(
                    adapter, video, memory, device
                )
                summaries = probe.build_summary_cache(observations, device)
                del observations
                components, c00_stats, base_component_count = stage1_parent.extract_fit_hard_negatives(
                    cfg, base_raw, locations4, labels
                )
                views, eligible = stage1_parent.prepare_training_views(
                    video,
                    labels,
                    target_ids,
                    components,
                    epoch=epoch_zero,
                    source_position=source_position,
                )
                for metadata in views:
                    step += 1
                    start, stop = metadata["start"], metadata["stop"]
                    frames = atomic._frame_tensor(video, range(start, stop), device)
                    decoder, base_logits, centre = adapter.decode_frozen_features(
                        frames, memory[start:stop]
                    )
                    summary_views = tuple(
                        value[start:stop].to(device=device, dtype=torch.float32)
                        for value in summaries
                    )
                    optimizer.zero_grad(set_to_none=True)
                    parts = adapter.expert(
                        decoder.unsqueeze(0),
                        base_logits.unsqueeze(0),
                        centre.unsqueeze(0),
                        tuple(value.unsqueeze(0) for value in summary_views),
                        return_parts=True,
                    )
                    refined, sampled = probe.sample_dense_event_logits(
                        parts.refined_logits.squeeze(0), video, start, stop
                    )
                    base_events, base_sampled = probe.sample_dense_event_logits(
                        base_logits, video, start, stop
                    )
                    _expect(np.array_equal(sampled, metadata["global_indices"]), "Stage1 event order changed")
                    _expect(np.array_equal(base_sampled, sampled), "Stage1 paired event order changed")
                    label_tensor = torch.from_numpy(metadata["labels"]).to(device=device, dtype=torch.float32)
                    target_tensor = torch.from_numpy(metadata["target_ids"]).to(device=device, dtype=torch.long)
                    time_tensor = torch.from_numpy(metadata["times"]).to(device=device, dtype=torch.long)
                    loss, recall, suppression, diagnostics = multiscale_pyramid_constrained_loss(
                        refined.float(),
                        base_events.float(),
                        label_tensor,
                        target_tensor,
                        time_tensor,
                        metadata["hard_negative_components"],
                        dual,
                    )
                    _expect(bool(torch.isfinite(loss)), "Stage1 loss is non-finite")
                    if step == 1:
                        initial_identity = bool(
                            torch.equal(parts.refined_logits.detach(), base_logits.unsqueeze(0))
                            and torch.count_nonzero(parts.correction.detach()) == 0
                        )
                        _expect(initial_identity, "Stage1 zero-init identity failed")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), 5.0)
                    gradient_l1 = {}
                    for name, parameter in adapter.expert.named_parameters():
                        _expect(parameter.grad is not None and torch.isfinite(parameter.grad).all(), "Stage1 gradient failed: {}".format(name))
                        gradient_l1[name] = float(parameter.grad.detach().abs().sum())
                    if step == 1:
                        _expect(gradient_l1["output_projection.weight"] > 0.0, "Stage1 output projection unreachable")
                    optimizer.step()
                    for name, parameter in adapter.expert.named_parameters():
                        _expect(torch.isfinite(parameter).all(), "Stage1 parameter failed: {}".format(name))
                    dual.update(recall, suppression)
                    weights = parts.mixture_weights.detach().float()
                    records.append(
                        {
                            "step": step,
                            "epoch": epoch_zero + 1,
                            "source_name": source_name,
                            "view_purpose": metadata["purpose"],
                            "view_start_bin": start,
                            "view_stop_bin_exclusive": stop,
                            **diagnostics,
                            "gradient_norm": float(gradient_norm),
                            "output_projection_gradient_l1": gradient_l1["output_projection.weight"],
                            "dual_target_time_recall_after": float(dual.target_time_recall),
                            "dual_hard_negative_suppression_after": float(dual.hard_negative_suppression),
                            "mixture_entropy": float((-(weights * weights.clamp_min(1e-12).log()).sum(dim=2)).mean()),
                            "correction_abs_mean": float(parts.correction.detach().float().abs().mean()),
                            "event_count": int(refined.numel()),
                        }
                    )
                    del frames, decoder, base_logits, centre, summary_views, parts, refined, base_events
                    del label_tensor, target_tensor, time_tensor, loss
                source_records.append(
                    {
                        "epoch": epoch_zero + 1,
                        "source_name": source_name,
                        "source_sha256": expected["sha256"],
                        "event_count": int(expected["event_count"]),
                        "eligible_view_count": eligible,
                        "selected_starts": [value["start"] for value in views],
                        "M20_C00_component_count": base_component_count,
                        "pure_FP_component_count": len(components),
                        "first_decoder_bins": decoded_bins,
                        "C00_stats": c00_stats,
                    }
                )
                del video, polarities, locations4, labels, target_ids, memory, summaries, base_raw, components, views
                torch.cuda.empty_cache()
                print("all11 Stage1 epoch {}/2 {} step {}/88".format(epoch_zero + 1, source_name, step), flush=True)
        _expect(step == STAGE1_STEPS, "Stage1 step count mismatch")
        validate_pyramid_step_diagnostics(records, STAGE1_STEPS)
        _expect(all(item["target_time_group_count"] > 0 for item in records), "Stage1 target-time constraint missing")
        _expect(all(item["hard_negative_component_count"] > 0 for item in records), "Stage1 hard-negative constraint missing")
        source_hashes_after = {}
        for source_name in ALL11:
            source_hashes_after[source_name] = sha256_file(verify_source(source_name, manifest))
            _expect(source_hashes_after[source_name] == source_hashes_before[source_name], "Stage1 source changed")
        updates = {
            name: float((parameter.detach().cpu() - before[name]).abs().sum())
            for name, parameter in adapter.expert.named_parameters()
        }
        _expect(all(value > 0.0 for value in updates.values()), "not every Stage1 tensor updated")
        m20_after = state_sha256(m20.state_dict())
        _expect(m20_after == m20_before, "M20 changed during Stage1")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        _expect(peak_mib <= MAX_CUDA_MIB, "Stage1 exceeded 2GiB")
        expert_state = {
            name: value.detach().cpu().clone() for name, value in adapter.expert.state_dict().items()
        }
        checkpoint = {
            "schema": "ev-uav-h2-pyramid-recovery-v2-all11-stage1-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "fit_sources": list(ALL11),
            "fresh_initialization": True,
            "promoted_checkpoint_reused": False,
            "optimizer_steps": step,
            "dual_state": dual.to_dict(),
            "released_m20_state_sha256": m20_after,
            "expert_state_sha256": state_sha256(expert_state),
            "expert_state_dict": expert_state,
            "validation_or_test_read": False,
        }
        checkpoint_sha = write_torch_exclusive(STAGE1_CHECKPOINT, checkpoint)
        receipt = {
            "schema": "ev-uav-h2-pyramid-recovery-v2-all11-stage1-receipt-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "checkpoint_path": str(STAGE1_CHECKPOINT.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "fit_sources": list(ALL11),
            "optimizer_steps": step,
            "initial_M20_bitwise_identity": bool(initial_identity),
            "all_expert_parameter_tensors_updated": True,
            "expert_parameter_update_l1": updates,
            "constraint_trends": stage1_parent.constraint_trends(records),
            "all_step_diagnostics": records,
            "source_epoch_diagnostics": source_records,
            "source_sha256_before": source_hashes_before,
            "source_sha256_after": source_hashes_after,
            "released_m20_state_sha256_before": m20_before,
            "released_m20_state_sha256_after": m20_after,
            "effective_C00": effective_c00,
            "effective_C00_sha256": EXPECTED_C00_SHA256,
            "peak_CUDA_MiB": peak_mib,
            "elapsed_seconds": time.perf_counter() - started,
            "promoted_checkpoint_reused": False,
            "validation_or_test_read": False,
        }
        receipt_sha = write_json_exclusive(STAGE1_RECEIPT, receipt)
        del checkpoint, expert_state, adapter, m20, optimizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)
    print(json.dumps({"stage": "all11_Stage1_complete", "checkpoint_sha256": checkpoint_sha, "receipt_sha256": receipt_sha, "optimizer_steps": step, "peak_CUDA_MiB": peak_mib, "CUDA_after_release_MiB": after_mib}, indent=2))


def extract_all11_features(args):
    require_gpu(args)
    load_protocol()
    checkpoint, _, checkpoint_sha, receipt_sha = stage1_gate()
    cfg, effective_c00 = build_c00()
    if FEATURE_ROOT.exists():
        raise FileExistsError("refusing to overwrite all11 feature cache")
    if STAGE2_ROOT.exists() or FINAL_PACKAGE.parent.exists():
        raise RuntimeError("later final output exists before feature extraction")
    manifest = source_manifest()
    started = time.perf_counter()
    records = []
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_all11_input_features"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m20, _ = atomic.build_released_m20(device)
        m20_before = state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        adapter.expert.load_state_dict(checkpoint["expert_state_dict"], strict=True)
        adapter.eval()
        for source_name in ALL11:
            path = verify_source(source_name, manifest)
            video, polarities, locations4 = atomic._load_input_only(path)
            _expect(len(polarities) == int(manifest[source_name]["event_count"]), "feature event count changed")
            _expect(use_h2_residual_refiner(len(polarities), polarities), "all11 source left H2 route")
            cache = promoted.build_input_only_source_cache(adapter, video, polarities, locations4, cfg, device)
            node_payload = {
                "schema": "ev-uav-h2-pyramid-recovery-v2-all11-node-features-v1",
                "source_name_for_provenance_only": source_name,
                "source_sha256": manifest[source_name]["sha256"],
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__)),
                "event_count": cache["event_count"],
                "node_features": cache.pop("node_features"),
                "component_count": len(cache["components"]),
                "contains_labels_or_target_ids": False,
            }
            cache.update(
                {
                    "source_name_for_provenance_only": source_name,
                    "source_sha256": manifest[source_name]["sha256"],
                    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                    "runner_sha256": sha256_file(Path(__file__)),
                    "Stage1_checkpoint_sha256": checkpoint_sha,
                    "effective_C00_sha256": EXPECTED_C00_SHA256,
                }
            )
            node_sha = write_torch_exclusive(node_cache_path(source_name), node_payload)
            score_sha = write_torch_exclusive(source_cache_path(source_name), cache)
            _expect(verify_sidecar(node_cache_path(source_name)) == node_sha, "node cache verify failed")
            _expect(verify_sidecar(source_cache_path(source_name)) == score_sha, "score cache verify failed")
            _expect(sha256_file(path) == manifest[source_name]["sha256"], "source changed during feature extraction")
            records.append(
                {
                    "source_name": source_name,
                    "event_count": cache["event_count"],
                    "score_cache_path": str(source_cache_path(source_name).resolve()),
                    "score_cache_sha256": score_sha,
                    "node_cache_path": str(node_cache_path(source_name).resolve()),
                    "node_cache_sha256": node_sha,
                    "M20_component_count": cache["M20_component_count"],
                    "action_component_count": len(cache["components"]),
                    "temporal_node_count": int(sum(len(value) for value in node_payload["node_features"])),
                    "contains_labels_or_target_ids": False,
                }
            )
            del video, polarities, locations4, cache, node_payload
            torch.cuda.empty_cache()
            print("all11 input-only features {}".format(source_name), flush=True)
        _expect(state_sha256(m20.state_dict()) == m20_before, "M20 changed during feature extraction")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        _expect(peak_mib <= MAX_CUDA_MIB, "feature extraction exceeded 2GiB")
        del adapter, m20
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)
    payload = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-all11-feature-manifest-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "Stage1_checkpoint_sha256": checkpoint_sha,
        "Stage1_receipt_sha256": receipt_sha,
        "source_order": list(ALL11),
        "records": records,
        "all_caches_input_only": True,
        "effective_C00": effective_c00,
        "effective_C00_sha256": EXPECTED_C00_SHA256,
        "peak_CUDA_MiB": peak_mib,
        "CUDA_after_release_MiB": after_mib,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_or_test_read": False,
    }
    digest = write_json_exclusive(FEATURE_MANIFEST, payload)
    print(json.dumps({"stage": "all11_features_complete", "manifest_sha256": digest, "records": records, "peak_CUDA_MiB": peak_mib}, indent=2))


def feature_gate():
    _, _, stage1_sha, _ = stage1_gate()
    manifest_sha = verify_sidecar(FEATURE_MANIFEST)
    payload = read_json(FEATURE_MANIFEST)
    runner_sha = sha256_file(Path(__file__))
    _expect(payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "feature protocol changed")
    _expect(payload.get("runner_sha256") == runner_sha, "runner changed after features")
    _expect(payload.get("Stage1_checkpoint_sha256") == stage1_sha, "feature Stage1 changed")
    _expect(tuple(payload.get("source_order", ())) == ALL11, "feature order changed")
    _expect(payload.get("all_caches_input_only") is True, "feature caches are not input-only")
    _expect(payload.get("validation_or_test_read") is False, "feature val/test receipt changed")
    for record in payload["records"]:
        _expect(verify_sidecar(Path(record["score_cache_path"])) == record["score_cache_sha256"], "score cache changed")
        _expect(verify_sidecar(Path(record["node_cache_path"])) == record["node_cache_sha256"], "node cache changed")
    return payload, manifest_sha


def load_component_targets(source_names, manifest):
    output = {}
    for source_name in source_names:
        cache = load_score_cache(source_name)
        path = verify_source(source_name, manifest)
        labels, target_ids = atomic._load_truth(path)
        _expect(labels.size == int(cache["event_count"]), "component truth count changed")
        output[source_name] = promoted.component_labels(cache, labels)
        del cache, labels, target_ids
    return output


def train_recovery_head(source_names, targets, device):
    seed_all()
    head = H2PyramidComponentRecoveryHead().to(device)
    _expect(component_recovery_parameter_count(head) == 14081, "head parameter count changed")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.0001)
    before = {name: parameter.detach().cpu().clone() for name, parameter in head.named_parameters()}
    records = []
    step = 0
    for epoch in range(1, RECOVERY_EPOCHS + 1):
        for source_name in source_names:
            step += 1
            node_cache = load_node_cache(source_name)
            features, mask = promoted.padded_component_batch(node_cache, device)
            labels = torch.from_numpy(targets[source_name]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features, mask)
            loss = promoted.balanced_component_bce(logits, labels)
            _expect(bool(torch.isfinite(loss)), "recovery loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            for name, parameter in head.named_parameters():
                _expect(parameter.grad is not None and torch.isfinite(parameter.grad).all(), "recovery gradient failed: {}".format(name))
            optimizer.step()
            for name, parameter in head.named_parameters():
                _expect(torch.isfinite(parameter).all(), "recovery parameter failed: {}".format(name))
            records.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "source_name": source_name,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "positive_component_count": int(torch.count_nonzero(labels > 0.5)),
                    "negative_component_count": int(torch.count_nonzero(labels < 0.5)),
                }
            )
            del node_cache, features, mask, labels, logits, loss
    updates = {
        name: float((parameter.detach().cpu() - before[name]).abs().sum())
        for name, parameter in head.named_parameters()
    }
    _expect(all(value > 0.0 for value in updates.values()), "not every recovery tensor updated")
    state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    del optimizer, head
    return state, records, updates


def predict_recovery_head(state, source_names, device):
    head = H2PyramidComponentRecoveryHead().to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    output = {}
    with torch.no_grad():
        for source_name in source_names:
            node_cache = load_node_cache(source_name)
            features, mask = promoted.padded_component_batch(node_cache, device)
            output[source_name] = torch.sigmoid(head(features, mask)).cpu().numpy().astype(np.float64)
            del node_cache, features, mask
    del head
    return output


def challenge_report(counts):
    return promoted.challenge_report(counts)


def report_delta(reference, candidate):
    return promoted.report_delta(reference, candidate)


def counts_for_scores(cache, labels, target_ids, scores):
    return crossfit.sufficient_counts_for_video(
        scores, labels, target_ids, cache["locations4"], THRESHOLD
    )


def exact_oof_cutoff(probabilities, manifest):
    joined = np.concatenate([np.asarray(probabilities[name], dtype=np.float64) for name in ALL11])
    _expect(joined.size > 0 and np.isfinite(joined).all(), "OOF probabilities invalid")
    unique = np.unique(joined)
    identity = float(np.nextafter(unique.max(), np.inf))
    cutoffs = np.concatenate(([identity], unique[::-1]))
    pooled_m20 = crossfit.SufficientCounts()
    pooled_stage1 = crossfit.SufficientCounts()
    pooled_stage2 = [crossfit.SufficientCounts() for _ in cutoffs]
    per_source = {}
    for source_name in ALL11:
        cache = load_score_cache(source_name)
        path = verify_source(source_name, manifest)
        labels, target_ids = atomic._load_truth(path)
        m20_counts = counts_for_scores(cache, labels, target_ids, cache["base_post"])
        stage1_counts = counts_for_scores(cache, labels, target_ids, cache["stage1_post"])
        source_stage2 = []
        source_probabilities = np.asarray(probabilities[source_name], dtype=np.float64)
        _expect(source_probabilities.size == len(cache["components"]), "OOF component alignment changed")
        for cutoff_index, cutoff in enumerate(cutoffs):
            decisions = source_probabilities >= float(cutoff)
            if np.any(decisions):
                scores = restore_whole_components_bitwise(
                    cache["stage1_post"], cache["base_post"], cache["components"], decisions
                )
                stage2_counts = counts_for_scores(cache, labels, target_ids, scores)
                del scores
            else:
                stage2_counts = stage1_counts
            promoted.assert_paired_count_invariants(m20_counts, stage1_counts, stage2_counts)
            pooled_stage2[cutoff_index] = pooled_stage2[cutoff_index] + stage2_counts
            source_stage2.append(stage2_counts)
        pooled_m20 = pooled_m20 + m20_counts
        pooled_stage1 = pooled_stage1 + stage1_counts
        per_source[source_name] = {"M20": m20_counts, "Stage1": stage1_counts, "Stage2": source_stage2}
        _expect(sha256_file(path) == manifest[source_name]["sha256"], "source changed during OOF calibration")
        del cache, labels, target_ids, source_stage2
    promoted.assert_paired_count_invariants(pooled_m20, pooled_stage1, *pooled_stage2)
    m20_report = challenge_report(pooled_m20)
    stage1_report = challenge_report(pooled_stage1)
    breakpoints = []
    feasible = []
    for cutoff_index, cutoff in enumerate(cutoffs):
        stage2_report = challenge_report(pooled_stage2[cutoff_index])
        recovery = report_delta(stage1_report, stage2_report)
        is_feasible = bool(
            recovery["TP"] >= 0
            and recovery["CO"] >= 0
            and (recovery["TP"] > 0 or recovery["CO"] > 0)
        )
        summary = {
            "cutoff": float(cutoff),
            "restored_component_count": int(sum(np.sum(probabilities[name] >= cutoff) for name in ALL11)),
            "Stage2_Score": stage2_report["Score"],
            "Score_gain_vs_M20": stage2_report["Score"] - m20_report["Score"],
            "TP_recovery_vs_Stage1": recovery["TP"],
            "CO_recovery_vs_Stage1": recovery["CO"],
            "feasible": is_feasible,
        }
        breakpoints.append(summary)
        if is_feasible:
            source_records = []
            for source_name in ALL11:
                stored = per_source[source_name]
                source_records.append(
                    {
                        "source_name": source_name,
                        "component_count": int(np.asarray(probabilities[source_name]).size),
                        "restored_component_count": int(np.sum(probabilities[source_name] >= cutoff)),
                        "M20": challenge_report(stored["M20"]),
                        "Stage1": challenge_report(stored["Stage1"]),
                        "Stage2": challenge_report(stored["Stage2"][cutoff_index]),
                    }
                )
            evaluation = {
                "cutoff": float(cutoff),
                "source_records": source_records,
                "M20": m20_report,
                "Stage1": stage1_report,
                "Stage2": stage2_report,
                "Stage1_delta_vs_M20": report_delta(m20_report, stage1_report),
                "Stage2_delta_vs_M20": report_delta(m20_report, stage2_report),
                "Stage2_recovery_vs_Stage1": recovery,
            }
            feasible.append((summary, evaluation))
    if not feasible:
        return None, breakpoints
    selected_summary, selected_evaluation = max(
        feasible,
        key=lambda item: (
            item[0]["Stage2_Score"],
            item[0]["CO_recovery_vs_Stage1"],
            item[0]["TP_recovery_vs_Stage1"],
            item[0]["cutoff"],
        ),
    )
    independent = exact_risk_controlled_breakpoint(
        BreakpointRecord(
            cutoff=value[0]["cutoff"],
            score_gain_vs_m20=value[0]["Score_gain_vs_M20"],
            stage2_true_positive_events=value[1]["Stage2"]["TP"],
            stage1_true_positive_events=value[1]["Stage1"]["TP"],
            stage2_correct_objects=value[1]["Stage2"]["CO"],
            stage1_correct_objects=value[1]["Stage1"]["CO"],
        )
        for value in feasible
    )
    _expect(
        independent is not None
        and float(independent.cutoff) == float(selected_summary["cutoff"]),
        "independent exact cutoff selector disagreed",
    )
    return {"summary": selected_summary, "evaluation": selected_evaluation}, breakpoints


def train_stage2_and_freeze(args):
    require_gpu(args)
    load_protocol()
    feature_payload, feature_manifest_sha = feature_gate()
    stage1_checkpoint, stage1_receipt, stage1_sha, stage1_receipt_sha = stage1_gate()
    if STAGE2_ROOT.exists() or FINAL_PACKAGE.parent.exists():
        raise FileExistsError("refusing to overwrite final Stage2/package")
    manifest = source_manifest()
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_all11_oof_and_final_head"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        targets = load_component_targets(ALL11, manifest)
        oof_probabilities = {}
        fold_records = []
        total_steps = 0
        for fold_name, fit_sources, held_sources, expected_steps in OOF_FOLDS:
            state, records, updates = train_recovery_head(fit_sources, targets, device)
            _expect(len(records) == expected_steps, "{} step count changed".format(fold_name))
            oof_probabilities.update(predict_recovery_head(state, held_sources, device))
            total_steps += len(records)
            fold_records.append(
                {
                    "fold": fold_name,
                    "fit_sources": list(fit_sources),
                    "held_sources": list(held_sources),
                    "optimizer_steps": len(records),
                    "loss_first": records[0]["loss"],
                    "loss_last": records[-1]["loss"],
                    "all_parameter_tensors_updated": all(value > 0.0 for value in updates.values()),
                    "state_sha256_for_audit_only_not_packaged": state_sha256(state),
                }
            )
            del state, records, updates
        _expect(tuple(oof_probabilities) == ALL11, "OOF prediction coverage/order changed")
        selected, breakpoints = exact_oof_cutoff(oof_probabilities, manifest)
        if selected is None:
            failure = {
                "schema": "ev-uav-h2-pyramid-recovery-v2-all11-oof-calibration-v1",
                "created_utc": utc_now(),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__)),
                "feature_manifest_sha256": feature_manifest_sha,
                "fold_records": fold_records,
                "breakpoints": breakpoints,
                "feasible_cutoff_exists": False,
                "final_head_or_package_created": False,
                "decision": "permanent_stop_identity_no_final_package",
                "validation_or_test_read": False,
            }
            write_json_exclusive(OOF_RESULT, failure)
            print(json.dumps({"stage": "all11_OOF_failed", "feasible_cutoff_exists": False}, indent=2))
            return
        final_state, final_records, final_updates = train_recovery_head(ALL11, targets, device)
        _expect(len(final_records) == FINAL_HEAD_STEPS, "final head step count changed")
        total_steps += len(final_records)
        _expect(total_steps == TOTAL_RECOVERY_STEPS, "total recovery steps changed")
        cutoff = float(selected["summary"]["cutoff"])
        oof_payload = {
            "schema": "ev-uav-h2-pyramid-recovery-v2-all11-oof-calibration-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "feature_manifest_sha256": feature_manifest_sha,
            "fold_records": fold_records,
            "OOF_prediction_source_order": list(oof_probabilities),
            "OOF_predictions_cover_each_source_once": True,
            "OOF_held_fold_by_source": {
                source_name: fold_name
                for fold_name, _, held_sources, _ in OOF_FOLDS
                for source_name in held_sources
            },
            "OOF_component_probabilities": {
                source_name: np.asarray(oof_probabilities[source_name], dtype=np.float64).tolist()
                for source_name in ALL11
            },
            "cutoff_rule_unchanged": True,
            "selected_cutoff": selected,
            "breakpoints": breakpoints,
            "feasible_cutoff_exists": True,
            "train_only_calibration_not_unbiased_evaluation": True,
            "promoted_recovery_head_or_cutoff_reused": False,
            "final_head_optimizer_steps": len(final_records),
            "total_recovery_optimizer_steps": total_steps,
            "final_head_loss_first": final_records[0]["loss"],
            "final_head_loss_last": final_records[-1]["loss"],
            "all_final_head_parameter_tensors_updated": all(value > 0.0 for value in final_updates.values()),
            "validation_or_test_read": False,
        }
        oof_sha = write_json_exclusive(OOF_RESULT, oof_payload)
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        _expect(peak_mib <= MAX_CUDA_MIB, "Stage2 exceeded 2GiB")
        package = {
            "schema": "ev-uav-h2-pyramid-recovery-v2-all11-final-package-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "inference_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
            "fit_sources": list(ALL11),
            "source_manifest_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
            "execution_dependency_sha256": EXPECTED_EXECUTION_DEPENDENCIES,
            "released_M20_checkpoint_sha256": EXPECTED_M20_SHA256,
            "released_m20_state_sha256": stage1_checkpoint[
                "released_m20_state_sha256"
            ],
            "outer_decision_sha256": EXPECTED_DECISION_SHA256,
            "fresh_Stage1_checkpoint_sha256": stage1_sha,
            "fresh_Stage1_state_sha256": stage1_checkpoint["expert_state_sha256"],
            "fresh_Stage1_state_dict": stage1_checkpoint["expert_state_dict"],
            "fresh_final_recovery_head_state_sha256": state_sha256(final_state),
            "fresh_final_recovery_head_state_dict": final_state,
            "all11_OOF_calibration_sha256": oof_sha,
            "recovery_cutoff": cutoff,
            "recovery_cutoff_record": selected,
            "prediction_threshold": THRESHOLD,
            "effective_C00": feature_payload["effective_C00"],
            "effective_C00_sha256": EXPECTED_C00_SHA256,
            "route": {
                "event_count_cutoff_exclusive": H2_EVENT_COUNT_CUTOFF,
                "polarity_minority_cutoff_inclusive": H2_POLARITY_MINORITY_CUTOFF,
                "non_H2_behavior": "bitwise_released_M20_identity",
            },
            "fresh_initialization": {"Stage1": True, "OOF_heads": True, "final_head": True},
            "promoted_outer_weights_or_cutoff_reused": False,
            "optimizer_state_included": False,
            "train_only_calibration_not_unbiased_evaluation": True,
            "validation_or_test_read": False,
        }
        package_sha = write_torch_exclusive(FINAL_PACKAGE, package)
        reloaded = torch.load(FINAL_PACKAGE, map_location="cpu", weights_only=False)
        _expect(state_sha256(reloaded["fresh_Stage1_state_dict"]) == reloaded["fresh_Stage1_state_sha256"], "reloaded Stage1 state changed")
        _expect(state_sha256(reloaded["fresh_final_recovery_head_state_dict"]) == reloaded["fresh_final_recovery_head_state_sha256"], "reloaded final head changed")
        _expect(verify_sidecar(FINAL_PACKAGE) == package_sha, "final package SHA verify failed")
        manifest_payload = {
            "schema": "ev-uav-h2-pyramid-recovery-v2-all11-final-manifest-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "inference_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
            "package_path": str(FINAL_PACKAGE.resolve()),
            "package_sha256": package_sha,
            "Stage1_checkpoint_sha256": stage1_sha,
            "Stage1_receipt_sha256": stage1_receipt_sha,
            "feature_manifest_sha256": feature_manifest_sha,
            "OOF_calibration_sha256": oof_sha,
            "all11_source_sha256_before_after_equal": stage1_receipt["source_sha256_before"] == stage1_receipt["source_sha256_after"],
            "Stage1_optimizer_steps": STAGE1_STEPS,
            "Stage2_optimizer_steps": total_steps,
            "recovery_cutoff": cutoff,
            "strict_CPU_reload": True,
            "promoted_outer_weights_or_cutoff_reused": False,
            "peak_CUDA_MiB": peak_mib,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_or_test_read": False,
            "default_submission_changed": False,
        }
        manifest_sha = write_json_exclusive(FINAL_MANIFEST, manifest_payload)
        del reloaded, package, final_state, final_records, final_updates
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)
    print(json.dumps({"stage": "all11_final_package_complete", "OOF_calibration_sha256": oof_sha, "package_sha256": package_sha, "manifest_sha256": manifest_sha, "selected_cutoff": cutoff, "train_only_OOF": selected["evaluation"], "peak_CUDA_MiB": peak_mib, "CUDA_after_release_MiB": after_mib}, indent=2))


def final_package_gate():
    package_sha = verify_sidecar(FINAL_PACKAGE)
    manifest_sha = verify_sidecar(FINAL_MANIFEST)
    oof_sha = verify_sidecar(OOF_RESULT)
    package = torch.load(FINAL_PACKAGE, map_location="cpu", weights_only=False)
    manifest = read_json(FINAL_MANIFEST)
    oof = read_json(OOF_RESULT)
    runner_sha = sha256_file(Path(__file__))
    _expect(package.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "package protocol changed")
    _expect(package.get("runner_sha256") == runner_sha, "package runner changed")
    _expect(package.get("inference_wrapper_sha256") == EXPECTED_WRAPPER_SHA256, "package wrapper changed")
    _expect(tuple(package.get("fit_sources", ())) == ALL11, "package fit set changed")
    _expect(package.get("promoted_outer_weights_or_cutoff_reused") is False, "package reused promoted weights")
    _expect(package.get("validation_or_test_read") is False, "package val/test receipt changed")
    _expect(state_sha256(package["fresh_Stage1_state_dict"]) == package["fresh_Stage1_state_sha256"], "package Stage1 state changed")
    _expect(state_sha256(package["fresh_final_recovery_head_state_dict"]) == package["fresh_final_recovery_head_state_sha256"], "package head state changed")
    _expect(manifest.get("package_sha256") == package_sha, "manifest package changed")
    _expect(manifest.get("runner_sha256") == runner_sha, "manifest runner changed")
    _expect(manifest.get("validation_or_test_read") is False, "manifest val/test receipt changed")
    _expect(oof.get("feasible_cutoff_exists") is True, "OOF cutoff is infeasible")
    _expect(oof.get("promoted_recovery_head_or_cutoff_reused") is False, "OOF reused promoted artifacts")
    _expect(float(package["recovery_cutoff"]) == float(oof["selected_cutoff"]["summary"]["cutoff"]), "package cutoff changed")
    _expect(package.get("all11_OOF_calibration_sha256") == oof_sha, "package OOF binding changed")
    return package, manifest, oof, package_sha, manifest_sha, oof_sha


def audit_final_package(_args):
    load_protocol()
    formal_cpu_audit_gate()
    if FINAL_AUDIT.exists():
        raise FileExistsError("refusing to overwrite final package audit")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before final package CPU audit")
    package, manifest, oof, package_sha, manifest_sha, oof_sha = final_package_gate()
    from utils.h2_pyramid_recovery_v2_inference import load_final_package_payload

    deployment_payload = load_final_package_payload(
        FINAL_PACKAGE,
        verify_wrapper_hash=True,
    )
    _expect(
        state_sha256(deployment_payload.stage1_state_dict)
        == package["fresh_Stage1_state_sha256"],
        "deployment wrapper Stage1 package load changed",
    )
    _expect(
        state_sha256(deployment_payload.recovery_state_dict)
        == package["fresh_final_recovery_head_state_sha256"],
        "deployment wrapper recovery package load changed",
    )
    _expect(
        float(deployment_payload.recovery_cutoff) == float(package["recovery_cutoff"]),
        "deployment wrapper cutoff package load changed",
    )
    m20, _ = atomic.build_released_m20(torch.device("cpu"))
    adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).cpu()
    adapter.expert.load_state_dict(package["fresh_Stage1_state_dict"], strict=True)
    head = H2PyramidComponentRecoveryHead().cpu()
    head.load_state_dict(package["fresh_final_recovery_head_state_dict"], strict=True)
    base = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    candidate = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    routed = input_only_routed_scores(base, candidate, np.zeros(3, dtype=np.uint8))
    _expect(np.array_equal(routed, base), "final non-H2 identity failed")
    _expect(not torch.cuda.is_initialized(), "final CPU audit initialized CUDA")
    payload = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-all11-final-package-cpu-audit-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "package_sha256": package_sha,
        "manifest_sha256": manifest_sha,
        "OOF_calibration_sha256": oof_sha,
        "strict_CPU_M20_Stage1_and_head_load": True,
        "strict_deployment_wrapper_package_load": True,
        "non_H2_bitwise_identity": True,
        "OOF_cutoff_feasible": oof["feasible_cutoff_exists"],
        "promoted_weights_reused": False,
        "validation_or_test_read": False,
        "CUDA_initialized": False,
        "passed": True,
    }
    digest = write_json_exclusive(FINAL_AUDIT, payload)
    print(json.dumps({"stage": "final_package_CPU_audit_complete", "sha256": digest, "passed": True}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("cpu-audit")
    audit.set_defaults(handler=cpu_audit)
    for name, handler in (
        ("train-stage1", train_stage1),
        ("extract-all11-features", extract_all11_features),
        ("train-stage2-and-freeze", train_stage2_and_freeze),
    ):
        child = commands.add_parser(name)
        child.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
        child.set_defaults(handler=handler)
    final_audit = commands.add_parser("audit-final-package")
    final_audit.set_defaults(handler=audit_final_package)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

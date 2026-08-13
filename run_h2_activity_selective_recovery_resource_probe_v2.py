"""Resource-only Stage2 probe on fixed development source train089.

This runner may never be used as formal evidence.  It freezes an old activity
checkpoint whose fit included the new experiment's sealed G2 and uses it only
to create real train089 disagreement components for an eight-step mechanical
and CUDA-resource check of the unchanged recovery network.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional

import crossfit_component_reranker as component_crossfit
import replay_temporal_memory_validation as replay
from crossfit_component_reranker import metrics_from_counts, sufficient_counts_for_video
from model.h2_activity_selective_recovery import (
    DisagreementRecoveryNet,
    recovery_parameter_count,
    recovery_sequence_collate,
)
from run_h2_activity_selective_recovery_probe import (
    apply_c00,
    build_recovery_features,
    component_offsets,
    dense_full_stream,
    python_gpu_processes,
    sequence_offsets,
    setup_seed,
    utc_now,
    write_json_exclusive,
    write_npz_exclusive,
    save_torch_exclusive,
)
from run_h2_atomic_component_deletion_v3 import (
    C00_OVERRIDES,
    EVC_ROOT,
    _load_input_only,
    _load_truth,
    build_released_m20,
    gpu_run_lock,
    sha256_file,
    sha256_float32,
    state_sha256,
)
from run_high_density_dual_expert_grouped_oof import (
    _load_trained_expert_model,
    tensor_state_sha256,
)
from utils.activity_selective_recovery import (
    atomic_recover_or_identity,
    extract_disagreement_components,
    marginal_recovery_targets,
    negative_reference_conformal_confidence,
)
from utils.atomic_component_deletion import (
    complete_input_polarity_minority_fraction,
    use_h2_atomic_deletion,
)


WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "h2_activity_selective_recovery_resource_probe_v2.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "891a41b36bc372bfc7ee6033940183bfbc73b3391c9e123afe727a4e33c3073e"
)
HELPER_RUNNER_PATH = EVC_ROOT / "run_h2_activity_selective_recovery_probe.py"
EXPECTED_HELPER_RUNNER_SHA256 = (
    "fef4c50b716e5b10b9fd2cf0364e3cda56e7c5f2ca5925a8b105d08f9006b9fc"
)
BASE_SCIENCE_PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "h2_activity_suppress_selective_recovery_g2_science_v1.json"
)
EXPECTED_BASE_SCIENCE_PROTOCOL_SHA256 = (
    "ca61ec2777be57703c0c949d75e1457876ac419efc93df51a17efeb5a5229f23"
)
M20_PATH = EVC_ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
TRAIN_ROOT = WORKSPACE_ROOT / "datasets" / "EV-UAV-Challenge2" / "train"
SOURCE_NAME = "train_089.npz"
OLD_ACTIVITY_CHECKPOINT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_high_density_dual_expert_grouped_oof_v1"
    / "paired_training"
    / "h2_hold_088_091__baseline"
    / "epoch_001_seed49.pt"
)
OLD_ACTIVITY_RUNTIME = OLD_ACTIVITY_CHECKPOINT.parent / "runtime_result.json"
OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_h2_activity_suppress_selective_recovery_g2_v1"
    / "resource_probe_v2_train089"
)
AUTHORIZATION_FLAG = "--root-authorized-resource-gpu"


RUNTIME_CONTEXT = {
    "started_epoch_seconds": None,
    "source_arrays_opened": [],
    "milestone": "not_started",
    "disagreement": None,
}


def load_protocol():
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen resource-only protocol changed")
    if sha256_file(HELPER_RUNNER_PATH) != EXPECTED_HELPER_RUNNER_SHA256:
        raise RuntimeError("frozen helper runner changed")
    if (
        sha256_file(BASE_SCIENCE_PROTOCOL_PATH)
        != EXPECTED_BASE_SCIENCE_PROTOCOL_SHA256
    ):
        raise RuntimeError("base science protocol changed")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol["status"] != "frozen_cpu_resource_design_awaiting_separate_gpu_authorization":
        raise RuntimeError("resource protocol status changed")
    if protocol["role"] != "resource_and_mechanical_evidence_only":
        raise RuntimeError("resource-only role changed")
    if protocol["formal_scientific_evidence"]:
        raise RuntimeError("resource probe was promoted to formal evidence")
    if protocol["execution"]["gpu_authorized"]:
        raise RuntimeError("authorization may not be embedded in the protocol")
    if protocol["scope_firewall"]["source_arrays_allowed"] != [SOURCE_NAME]:
        raise RuntimeError("source firewall changed")
    if protocol["fixed_development_source"]["name"] != SOURCE_NAME:
        raise RuntimeError("fixed resource source changed")
    stage = protocol["stage2_resource_training"]
    exact_stage = {
        "architecture_parameter_count": 7910,
        "input_patch_channels": 57,
        "seed": 73,
        "optimizer_steps": 8,
        "component_batch_size": 16,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "required_real_class_count": 2,
    }
    if any(stage[key] != value for key, value in exact_stage.items()):
        raise RuntimeError("Stage2 resource mechanics changed")
    return protocol


def build_c00(protocol):
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml", list(C00_OVERRIDES)
    )
    threshold = float(protocol["frozen_inference_and_components"]["prediction_threshold"])
    c00 = component_crossfit.validate_c00_config(cfg, threshold)
    if component_crossfit.sha256_json(c00) != protocol[
        "frozen_inference_and_components"
    ]["effective_c00_sha256"]:
        raise RuntimeError("effective C00 changed")
    return cfg, c00


def verify_old_checkpoint_payload(protocol, model, payload):
    frozen = protocol["frozen_old_activity_checkpoint"]
    metadata = payload.get("high_density_expert", {})
    provenance = payload.get("provenance", {})
    training_scope = provenance.get("training_scope", {})
    checks = {
        "checkpoint_format": int(payload.get("checkpoint_format_version", -1))
        == int(frozen["checkpoint_format_version"]),
        "checkpoint_epoch": int(payload.get("epoch", -1))
        == int(frozen["checkpoint_epoch"]),
        "activity_control_mode": metadata.get("input_mode") == "activity_control",
        "h2_domain": metadata.get("domain") == "h2",
        "level1_insertion": metadata.get("insertion_point") == "level1",
        "fit_sources_exact": provenance.get("fit_names") == frozen["fit_sources"],
        "old_held_sources_exact": provenance.get("held_names")
        == frozen["old_held_sources"],
        "source_protocol": provenance.get("protocol_sha256")
        == frozen["source_protocol_sha256"],
        "source_runner": provenance.get("runner_sha256")
        == frozen["source_runner_sha256"],
        "released_m20": provenance.get("released_m20_sha256")
        == protocol["released_m20"]["sha256"],
        "expert_only_scope": training_scope.get("name")
        == "high_density_expert_only",
        "expert_tensor_count": int(
            training_scope.get("trainable_state_tensor_count", -1)
        )
        == int(frozen["expert_state_tensor_count"]),
        "expert_parameter_count": int(
            training_scope.get("trainable_parameter_count", -1)
        )
        == int(frozen["expert_parameter_count"]),
        "inherited_m20_frozen": bool(
            training_scope.get("inherited_m20_bitwise_frozen", False)
        ),
        "state_tensor_count": len(payload["model_state_dict"])
        == int(frozen["model_state_tensor_count"]),
        "model_parameter_count": sum(p.numel() for p in model.parameters())
        == int(frozen["model_parameter_count"]),
        "final_state_hash": tensor_state_sha256(payload["model_state_dict"])
        == frozen["source_final_model_state_sha256"],
        "train089_was_old_held": SOURCE_NAME in provenance.get("held_names", []),
        "sealed_g2_was_old_fit": all(
            name in provenance.get("fit_names", [])
            for name in ("train_092.npz", "train_093.npz", "train_094.npz")
        ),
        "formal_reuse_disabled": not bool(
            frozen["formal_weight_initialization_allowed"]
            or frozen["formal_feature_generation_allowed"]
            or frozen["formal_cutoff_or_score_evidence_allowed"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("old activity checkpoint provenance failed: {}".format(failed))
    return checks


def persist_feature_artifact(
    path,
    locations,
    m20_raw,
    m20_post,
    activity_raw,
    activity_post,
    disagreement,
    patches,
    trajectories,
):
    offsets, component_events = component_offsets(disagreement.event_indices)
    patch_offsets = sequence_offsets(patches)
    write_npz_exclusive(
        path,
        artifact_schema=np.asarray(
            "ev-uav-h2-activity-selective-recovery-resource-input-v2"
        ),
        source_name=np.asarray(SOURCE_NAME),
        resource_only=np.asarray(True),
        formal_use_forbidden=np.asarray(True),
        m20_raw_scores=np.asarray(m20_raw, dtype=np.float32),
        m20_c00_scores=np.asarray(m20_post, dtype=np.float32),
        activity_raw_scores=np.asarray(activity_raw, dtype=np.float32),
        activity_c00_scores=np.asarray(activity_post, dtype=np.float32),
        locations=np.asarray(locations, dtype=np.int16),
        disagreement_m20_component_ids=disagreement.m20_component_ids,
        disagreement_component_offsets=offsets,
        disagreement_component_event_indices=component_events,
        patch_offsets=patch_offsets,
        recovery_patches=np.concatenate(patches, axis=0),
        trajectory_context=np.concatenate(trajectories, axis=0).astype(np.float32),
        contains_labels_or_target_ids=np.asarray(False),
    )
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def deterministic_dual_class_indices(targets, step, batch_size):
    targets = np.asarray(targets, dtype=np.uint8)
    negatives = np.flatnonzero(targets == 0)
    positives = np.flatnonzero(targets == 1)
    if negatives.size == 0 or positives.size == 0:
        raise RuntimeError("real train089 disagreement lacks both marginal classes")
    positive_slots = batch_size // 2
    negative_slots = batch_size - positive_slots
    selected_positive = [
        int(positives[(step * positive_slots + offset) % positives.size])
        for offset in range(positive_slots)
    ]
    selected_negative = [
        int(negatives[(step * negative_slots + offset) % negatives.size])
        for offset in range(negative_slots)
    ]
    indices = []
    for offset in range(max(len(selected_positive), len(selected_negative))):
        if offset < len(selected_positive):
            indices.append(selected_positive[offset])
        if offset < len(selected_negative):
            indices.append(selected_negative[offset])
    if len(indices) != batch_size:
        raise RuntimeError("deterministic resource batch size changed")
    return indices


def train_stage2_resource(protocol, patches, trajectories, targets, device):
    stage = protocol["stage2_resource_training"]
    setup_seed(stage["seed"])
    targets = np.asarray(targets, dtype=np.uint8)
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise RuntimeError("real train089 disagreement lacks both marginal classes")
    class_counts = np.bincount(targets, minlength=2).astype(np.float64)
    weights = 1.0 / class_counts[targets]
    weights /= weights.mean()
    items = [
        {
            "patches": patches[index],
            "trajectory": trajectories[index],
            "target": float(targets[index]),
            "weight": float(weights[index]),
        }
        for index in range(len(patches))
    ]
    model = DisagreementRecoveryNet().to(device)
    if recovery_parameter_count(model) != int(stage["architecture_parameter_count"]):
        raise RuntimeError("Stage2 parameter count changed")
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    gradient_seen = {name: False for name, _ in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    branch_prefixes = (
        "semantic_stem.",
        "context_stem.",
        "spatial_fusion.",
        "trajectory_encoder.",
        "temporal.",
        "temporal_attention.",
        "classifier.",
    )
    diagnostics = []
    batch_size = int(stage["component_batch_size"])
    for step in range(int(stage["optimizer_steps"])):
        indices = deterministic_dual_class_indices(targets, step, batch_size)
        batch = recovery_sequence_collate([items[index] for index in indices])
        patch_tensor = batch["patches"].to(device)
        trajectory_tensor = batch["trajectory"].to(device)
        lengths = batch["lengths"].to(device)
        batch_targets = batch["targets"].to(device)
        batch_weights = batch["weights"].to(device)
        if int(torch.count_nonzero(batch_targets == 0).item()) == 0 or int(
            torch.count_nonzero(batch_targets == 1).item()
        ) == 0:
            raise RuntimeError("Stage2 resource batch lost class diversity")
        optimizer.zero_grad(set_to_none=True)
        logits = model(patch_tensor, trajectory_tensor, lengths)
        losses = functional.binary_cross_entropy_with_logits(
            logits, batch_targets, reduction="none"
        )
        loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Stage2 resource loss is non-finite")
        loss.backward()
        branch_gradient_l1 = {prefix: 0.0 for prefix in branch_prefixes}
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Stage2 resource gradient is missing or non-finite")
            value = float(gradient.detach().abs().sum().item())
            gradient_seen[name] |= value > 0.0
            for prefix in branch_gradient_l1:
                if name.startswith(prefix):
                    branch_gradient_l1[prefix] += value
        if not all(value > 0.0 for value in branch_gradient_l1.values()):
            raise RuntimeError("a Stage2 branch had zero gradient in a resource step")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(stage["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Stage2 resource gradient norm is non-finite")
        optimizer.step()
        diagnostics.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().item()),
                "preclip_gradient_norm": float(gradient_norm.detach().item()),
                "branch_gradient_l1": branch_gradient_l1,
                "batch_component_slots": len(indices),
                "batch_unique_components": len(set(indices)),
                "batch_positive_slots": int(
                    torch.count_nonzero(batch_targets == 1).item()
                ),
                "batch_negative_slots": int(
                    torch.count_nonzero(batch_targets == 0).item()
                ),
            }
        )
    if not all(gradient_seen.values()):
        missing = [name for name, seen in gradient_seen.items() if not seen]
        raise RuntimeError("Stage2 tensors never received nonzero gradient: {}".format(missing))
    updated = {
        name: not torch.equal(parameter.detach().cpu(), initial[name])
        for name, parameter in model.named_parameters()
    }
    if not all(updated.values()):
        missing = [name for name, changed in updated.items() if not changed]
        raise RuntimeError("Stage2 tensors did not update: {}".format(missing))
    model.eval()
    raw_probabilities = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = recovery_sequence_collate(items[start : start + batch_size])
            logits, embedding, _ = model(
                batch["patches"].to(device),
                batch["trajectory"].to(device),
                batch["lengths"].to(device),
                return_embedding=True,
            )
            raw_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    raw_probabilities = np.concatenate(raw_probabilities).astype(np.float64)
    embeddings = np.concatenate(embeddings).astype(np.float32)
    negative_reference = raw_probabilities[targets == 0]
    conformal = negative_reference_conformal_confidence(
        raw_probabilities, negative_reference
    )
    checkpoint_path = OUTPUT_ROOT / "stage2_train089_resource_step8.pt"
    save_torch_exclusive(
        checkpoint_path,
        {
            "schema": "ev-uav-h2-activity-selective-recovery-stage2-resource-v2",
            "resource_only": True,
            "formal_reuse_forbidden": True,
            "model_state_dict": model.state_dict(),
            "optimizer_steps": int(stage["optimizer_steps"]),
            "fit_negative_reference_probabilities": negative_reference,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "source": SOURCE_NAME,
            "old_activity_checkpoint_sha256": protocol[
                "frozen_old_activity_checkpoint"
            ]["sha256"],
        },
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "raw_probabilities": raw_probabilities,
        "conformal_confidences": conformal,
        "embeddings": embeddings,
        "negative_reference": negative_reference,
        "diagnostics": diagnostics,
        "gradient_seen": gradient_seen,
        "updated": updated,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def safe_cuda_peak_mib():
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_reserved(0) / (1024 ** 2))
    except Exception:
        pass
    return None


def safe_cuda_allocated_mib():
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated(0) / (1024 ** 2))
    except Exception:
        pass
    return None


def run_resource_probe(authorized):
    if not authorized:
        raise PermissionError("explicit parent resource-GPU authorization is required")
    protocol = load_protocol()
    if OUTPUT_ROOT.exists():
        raise FileExistsError("refusing to overwrite resource probe v2")
    if python_gpu_processes():
        raise RuntimeError("another Python GPU process exists before resource probe")
    source_path = TRAIN_ROOT / SOURCE_NAME
    source_metadata = protocol["fixed_development_source"]
    if not source_path.is_file() or sha256_file(source_path) != source_metadata["sha256"]:
        raise RuntimeError("fixed train089 identity changed")
    if sha256_file(M20_PATH) != protocol["released_m20"]["sha256"]:
        raise RuntimeError("released M20 identity changed")
    if sha256_file(OLD_ACTIVITY_CHECKPOINT) != protocol[
        "frozen_old_activity_checkpoint"
    ]["sha256"]:
        raise RuntimeError("old activity checkpoint identity changed")
    if sha256_file(OLD_ACTIVITY_RUNTIME) != protocol[
        "frozen_old_activity_checkpoint"
    ]["runtime_result_sha256"]:
        raise RuntimeError("old activity runtime provenance changed")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    RUNTIME_CONTEXT["started_epoch_seconds"] = started
    RUNTIME_CONTEXT["milestone"] = "output_created"
    device = torch.device("cuda:0")
    with gpu_run_lock("h2_activity_selective_recovery_resource_probe_v2"):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        cfg, c00 = build_c00(protocol)
        video, polarities, locations = _load_input_only(source_path)
        RUNTIME_CONTEXT["source_arrays_opened"] = [SOURCE_NAME]
        RUNTIME_CONTEXT["milestone"] = "train089_input_loaded"
        if (
            int(len(polarities)) != int(source_metadata["event_count"])
            or not use_h2_atomic_deletion(len(polarities), polarities)
            or not np.isclose(
                complete_input_polarity_minority_fraction(polarities),
                float(source_metadata["polarity_minority_fraction"]),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise RuntimeError("fixed train089 no longer satisfies exact H2 route")
        threshold = float(
            protocol["frozen_inference_and_components"]["prediction_threshold"]
        )

        activity_model, activity_payload = _load_trained_expert_model(
            OLD_ACTIVITY_CHECKPOINT, device
        )
        for parameter in activity_model.parameters():
            parameter.requires_grad_(False)
        activity_checks = verify_old_checkpoint_payload(
            protocol, activity_model, activity_payload
        )
        activity_hash_before = state_sha256(activity_model.state_dict())
        activity_raw, activity_decoder, activity_logits = dense_full_stream(
            activity_model, video, device
        )
        activity_hash_after = state_sha256(activity_model.state_dict())
        if activity_hash_before != activity_hash_after:
            raise RuntimeError("old activity checkpoint changed during inference")
        activity_post, activity_c00_stats = apply_c00(
            activity_raw, locations, cfg, threshold
        )
        activity_model.to("cpu")
        del activity_model
        gc.collect()
        torch.cuda.empty_cache()

        m20_model, m20_payload = build_released_m20(device)
        for parameter in m20_model.parameters():
            parameter.requires_grad_(False)
        m20_hash_before = state_sha256(m20_model.state_dict())
        inherited_parent_equal = all(
            name in activity_payload["model_state_dict"]
            and torch.equal(
                activity_payload["model_state_dict"][name].detach().cpu(),
                value.detach().cpu(),
            )
            for name, value in m20_model.state_dict().items()
        )
        if (
            len(m20_model.state_dict()) != int(protocol["released_m20"]["state_tensor_count"])
            or not inherited_parent_equal
        ):
            raise RuntimeError("old activity inherited M20 differs from released M20")
        m20_raw, m20_decoder, m20_logits = dense_full_stream(
            m20_model, video, device
        )
        m20_hash_after = state_sha256(m20_model.state_dict())
        if m20_hash_before != m20_hash_after:
            raise RuntimeError("released M20 changed during inference")
        m20_post, m20_c00_stats = apply_c00(m20_raw, locations, cfg, threshold)
        m20_model.to("cpu")
        del m20_model, m20_payload
        gc.collect()
        torch.cuda.empty_cache()
        RUNTIME_CONTEXT["milestone"] = "dual_full_stream_complete"

        disagreement = extract_disagreement_components(
            m20_post,
            activity_post,
            locations,
            threshold,
            spatial_radius=2,
            temporal_bin_size=50,
            temporal_radius_bins=1,
        )
        if not disagreement.event_indices:
            raise RuntimeError("train089 produced no real disagreement component")
        patches, trajectories = build_recovery_features(
            video,
            locations,
            m20_post,
            activity_post,
            disagreement.event_indices,
            m20_decoder,
            activity_decoder,
            m20_logits,
            activity_logits,
            threshold,
        )
        input_artifact = persist_feature_artifact(
            OUTPUT_ROOT / "immutable_resource_input.npz",
            locations,
            m20_raw,
            m20_post,
            activity_raw,
            activity_post,
            disagreement,
            patches,
            trajectories,
        )
        del m20_decoder, activity_decoder, m20_logits, activity_logits
        gc.collect()
        RUNTIME_CONTEXT["milestone"] = "label_free_input_persisted"

        labels, target_ids = _load_truth(source_path)

        def official_score(scores):
            counts = sufficient_counts_for_video(
                scores,
                labels,
                target_ids,
                locations,
                prediction_threshold=threshold,
            )
            return metrics_from_counts(counts)["score"]

        recovery_targets, marginal_score_deltas = marginal_recovery_targets(
            m20_post,
            activity_post,
            disagreement.event_indices,
            threshold,
            official_score,
        )
        positive_count = int(np.count_nonzero(recovery_targets == 1))
        negative_count = int(np.count_nonzero(recovery_targets == 0))
        RUNTIME_CONTEXT["disagreement"] = {
            "component_count": len(disagreement.event_indices),
            "positive_count": positive_count,
            "negative_count": negative_count,
        }
        label_artifact_path = OUTPUT_ROOT / "immutable_fit_only_resource_labels.npz"
        write_npz_exclusive(
            label_artifact_path,
            artifact_schema=np.asarray(
                "ev-uav-h2-activity-selective-recovery-resource-labels-v2"
            ),
            source_name=np.asarray(SOURCE_NAME),
            resource_only=np.asarray(True),
            formal_use_forbidden=np.asarray(True),
            marginal_recovery_targets=np.asarray(recovery_targets, dtype=np.uint8),
            marginal_official_score_deltas=np.asarray(
                marginal_score_deltas, dtype=np.float64
            ),
            contains_fit_only_labels=np.asarray(True),
        )
        label_artifact = {
            "path": str(label_artifact_path.resolve()),
            "sha256": sha256_file(label_artifact_path),
        }
        if positive_count == 0 or negative_count == 0:
            raise RuntimeError("real train089 disagreement lacks both marginal classes")
        RUNTIME_CONTEXT["milestone"] = "real_dual_classes_verified"

        stage2 = train_stage2_resource(
            protocol, patches, trajectories, recovery_targets, device
        )
        RUNTIME_CONTEXT["milestone"] = "stage2_eight_steps_complete"
        mechanical_cutoff = float(np.max(stage2["conformal_confidences"]))
        candidate, recovery_receipt = atomic_recover_or_identity(
            m20_post,
            activity_post,
            disagreement.event_indices,
            stage2["conformal_confidences"],
            mechanical_cutoff,
            threshold,
            enabled=True,
        )
        if not (
            recovery_receipt.enabled
            and recovery_receipt.complete_components_only
            and recovery_receipt.activity_outside_recovery_bitwise_equal
            and recovery_receipt.recovered_m20_scores_bitwise_equal
            and recovery_receipt.recovered_component_count > 0
        ):
            raise RuntimeError("resource-only atomic recovery gate failed")
        score_artifact_path = OUTPUT_ROOT / "immutable_resource_scores_actions.npz"
        write_npz_exclusive(
            score_artifact_path,
            artifact_schema=np.asarray(
                "ev-uav-h2-activity-selective-recovery-resource-actions-v2"
            ),
            resource_only=np.asarray(True),
            formal_use_forbidden=np.asarray(True),
            raw_recovery_probabilities=stage2["raw_probabilities"],
            conformal_confidences=stage2["conformal_confidences"],
            negative_reference_probabilities=stage2["negative_reference"],
            recovery_embeddings=stage2["embeddings"],
            mechanical_integrity_cutoff=np.asarray(mechanical_cutoff),
            recovered_component=np.asarray(
                stage2["conformal_confidences"] >= mechanical_cutoff,
                dtype=np.bool_,
            ),
            final_candidate_scores=np.asarray(candidate, dtype=np.float32),
            candidate_officially_scored=np.asarray(False),
            cutoff_is_scientific_or_formal=np.asarray(False),
        )
        score_artifact = {
            "path": str(score_artifact_path.resolve()),
            "sha256": sha256_file(score_artifact_path),
        }
        torch.cuda.synchronize()
        peak_mib = safe_cuda_peak_mib()
        if peak_mib is None or peak_mib > float(
            protocol["resource_budget"]["hard_peak_cuda_mib"]
        ):
            raise RuntimeError("resource probe exceeded or could not measure CUDA budget")

        del stage2["optimizer"], stage2["model"]
        gc.collect()
        torch.cuda.empty_cache()
        allocated_after_cleanup = safe_cuda_allocated_mib()
        result = {
            "schema": "ev-uav-h2-activity-selective-recovery-resource-result-v2",
            "created_utc": utc_now(),
            "status": "completed_resource_mechanics_only",
            "formal_scientific_evidence": False,
            "formal_started": False,
            "formal_remains_blocked": True,
            "source_arrays_opened": [SOURCE_NAME],
            "held_g2_array_read": False,
            "other_g1_or_g3_array_read": False,
            "validation_or_test_read": False,
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "helper_runner_sha256": sha256_file(HELPER_RUNNER_PATH),
            "source_sha256": sha256_file(source_path),
            "effective_c00_sha256": component_crossfit.sha256_json(c00),
            "old_activity_checkpoint": {
                "path": str(OLD_ACTIVITY_CHECKPOINT.resolve()),
                "sha256": sha256_file(OLD_ACTIVITY_CHECKPOINT),
                "provenance_checks": activity_checks,
                "state_sha256_before": activity_hash_before,
                "state_sha256_after": activity_hash_after,
                "formal_reuse_forbidden": True,
            },
            "frozen_parents": {
                "old_activity_inherited_m20_bitwise_equal_to_released": inherited_parent_equal,
                "released_m20_state_sha256_before": m20_hash_before,
                "released_m20_state_sha256_after": m20_hash_after,
            },
            "full_stream": {
                "temporal_bins": 160,
                "m20_raw_scores_sha256": sha256_float32(m20_raw),
                "m20_c00_scores_sha256": sha256_float32(m20_post),
                "activity_raw_scores_sha256": sha256_float32(activity_raw),
                "activity_c00_scores_sha256": sha256_float32(activity_post),
                "m20_c00_stats": m20_c00_stats,
                "activity_c00_stats": activity_c00_stats,
            },
            "disagreement": {
                "m20_component_count": disagreement.m20_component_count,
                "activity_component_count": disagreement.activity_component_count,
                "component_count": len(disagreement.event_indices),
                "component_bin_patch_count": int(sum(len(value) for value in patches)),
                "marginal_positive_count": positive_count,
                "marginal_negative_count": negative_count,
            },
            "stage2": {
                "trainable_parameter_count": 7910,
                "optimizer_steps": 8,
                "all_tensors_finite_nonzero_gradient": all(
                    stage2["gradient_seen"].values()
                ),
                "all_tensors_updated": all(stage2["updated"].values()),
                "checkpoint": str(stage2["checkpoint_path"].resolve()),
                "checkpoint_sha256": stage2["checkpoint_sha256"],
                "checkpoint_formal_reuse_forbidden": True,
                "diagnostics": stage2["diagnostics"],
            },
            "atomic_recovery": asdict(recovery_receipt),
            "mechanical_cutoff_not_scientific_or_formal": mechanical_cutoff,
            "official_candidate_metric_reported": False,
            "immutable_artifacts": {
                "input": input_artifact,
                "fit_labels": label_artifact,
                "scores_actions": score_artifact,
            },
            "peak_cuda_mib": peak_mib,
            "peak_cuda_budget_mib": float(
                protocol["resource_budget"]["hard_peak_cuda_mib"]
            ),
            "cuda_allocated_after_in_process_cleanup_mib": allocated_after_cleanup,
            "elapsed_seconds": time.time() - started,
        }
        result_path = OUTPUT_ROOT / "resource_probe_result.json"
        write_json_exclusive(result_path, result)
        result["result_path"] = str(result_path.resolve())
        result["result_sha256"] = sha256_file(result_path)
        RUNTIME_CONTEXT["milestone"] = "completed"
    return result


def write_failure_receipt(error):
    if not OUTPUT_ROOT.exists():
        return None
    peak_before_cleanup = safe_cuda_peak_mib()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    allocated_after_cleanup = safe_cuda_allocated_mib()
    path = OUTPUT_ROOT / "resource_probe_failure.json"
    if path.exists():
        return path
    started = RUNTIME_CONTEXT.get("started_epoch_seconds")
    payload = {
        "schema": "ev-uav-h2-activity-selective-recovery-resource-failure-v2",
        "created_utc": utc_now(),
        "status": "failed_closed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "milestone": RUNTIME_CONTEXT.get("milestone"),
        "source_arrays_opened": RUNTIME_CONTEXT.get("source_arrays_opened", []),
        "disagreement": RUNTIME_CONTEXT.get("disagreement"),
        "held_g2_array_read": False,
        "other_g1_or_g3_array_read": False,
        "validation_or_test_read": False,
        "formal_started": False,
        "formal_remains_blocked": True,
        "protocol_sha256": (
            sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "peak_cuda_mib_before_cleanup": peak_before_cleanup,
        "peak_cuda_budget_mib": 3600.0,
        "cuda_allocated_after_in_process_cleanup_mib": allocated_after_cleanup,
        "elapsed_seconds": None if started is None else time.time() - started,
    }
    write_json_exclusive(path, payload)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(AUTHORIZATION_FLAG, action="store_true")
    args = parser.parse_args()
    authorized = getattr(args, "root_authorized_resource_gpu")
    if not authorized:
        parser.error("explicit parent resource-GPU authorization flag is required")
    try:
        result = run_resource_probe(authorized)
    except Exception as error:
        write_failure_receipt(error)
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

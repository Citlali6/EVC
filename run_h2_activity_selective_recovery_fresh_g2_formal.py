"""Fresh-G2 formal runner for activity suppression plus atomic recovery.

The two GPU subcommands are intentionally separate. ``train-and-freeze`` may
open only G1/G3 and must persist an inner-pass receipt plus final checkpoint and
strategy hashes. ``evaluate-held-g2-once`` verifies those immutable artifacts,
creates a one-shot held-open receipt, and only then opens train092..094.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

import crossfit_component_reranker as component_crossfit
import replay_temporal_memory_validation as replay
from crossfit_component_reranker import (
    SufficientCounts,
    metrics_from_counts,
    sufficient_counts_for_video,
)
from dataset.temporal_memory import TemporalMemoryTrainDataset, temporal_memory_collate
from model.h2_activity_selective_recovery import (
    DisagreementRecoveryNet,
    RECOVERY_PATCH_CHANNELS,
    recovery_parameter_count,
    recovery_sequence_collate,
)
from model.high_density_polarity_expert import (
    build_expert_model_from_m20,
    configure_expert_only_training,
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
    CONTEXT_BINS,
    EVC_ROOT,
    HEIGHT,
    LOG_COUNT_CLIP,
    TEMPORAL_BIN_SIZE,
    WHOLE_T,
    WIDTH,
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
from utils.activity_selective_recovery_formal import (
    assess_inner_replay,
    deterministic_epoch_batches,
    exact_confidence_cutoffs,
    select_qualifying_inner_replay,
    source_class_balanced_weights,
)
from utils.atomic_component_deletion import (
    complete_input_polarity_minority_fraction,
    use_h2_atomic_deletion,
)
from utils.temporal_frame_loss import frame_balanced_event_bce


WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "h2_activity_selective_recovery_fresh_g2_formal_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "d9ed1d0d58ade4f0ec5f43a5cb16c2f61d6e3af765356b9eca2b0eeddb0cc2a7"
)
HELPER_RUNNER_PATH = EVC_ROOT / "run_h2_activity_selective_recovery_probe.py"
M20_PATH = EVC_ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
TRAIN_ROOT = WORKSPACE_ROOT / "datasets" / "EV-UAV-Challenge2" / "train"
EXPERIMENT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_h2_activity_selective_recovery_fresh_g2_formal_v1"
)
FIT_OUTPUT = EXPERIMENT_ROOT / "inner_and_final_fit"
HELD_OUTPUT = EXPERIMENT_ROOT / "held_g2_once"
TRAIN_RECEIPT_PATH = FIT_OUTPUT / "training_receipt.json"
STRATEGY_PATH = FIT_OUTPUT / "frozen_strategy.json"
TRAIN_AUTH_FLAG = "--root-authorized-formal-gpu"
HELD_AUTH_FLAG = "--root-authorized-held-g2-once"


GROUP_G1 = "g1_088_091"
GROUP_G3 = "g3_095_098"
GROUP_HELD = "g2_092_094_sealed"
FIT_GROUPS = (GROUP_G1, GROUP_G3)


PROCESS_CONTEXT = {
    "command": None,
    "started": None,
    "milestone": "not_started",
    "source_arrays_opened": [],
    "held_opened": False,
}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_protocol():
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen fresh-G2 protocol changed")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["status"] != "frozen_cpu_implementation_awaiting_priority_and_gpu_authorization":
        raise RuntimeError("formal protocol status changed")
    if protocol["execution"]["gpu_authorized"] or protocol["execution"][
        "priority_authorized"
    ]:
        raise RuntimeError("GPU/priority authorization may not be embedded")
    code_paths = {
        "stage1": EVC_ROOT / "model" / "high_density_polarity_expert.py",
        "stage1_loss": EVC_ROOT / "utils" / "temporal_frame_loss.py",
        "stage2": EVC_ROOT / "model" / "h2_activity_selective_recovery.py",
        "mechanics": EVC_ROOT / "utils" / "activity_selective_recovery.py",
        "formal_helpers": EVC_ROOT
        / "utils"
        / "activity_selective_recovery_formal.py",
        "helper_runner": HELPER_RUNNER_PATH,
        "base_science": EVC_ROOT
        / "protocols"
        / "h2_activity_suppress_selective_recovery_g2_science_v1.json",
    }
    expected = {
        "stage1": protocol["stage1"]["implementation_sha256"],
        "stage1_loss": protocol["stage1"]["training"][
            "loss_implementation_sha256"
        ],
        "stage2": protocol["stage2"]["implementation_sha256"],
        "mechanics": protocol["stage2"]["mechanics_implementation_sha256"],
        "formal_helpers": protocol["stage2"]["formal_helpers_sha256"],
        "helper_runner": protocol["execution"]["helper_runner_sha256"],
        "base_science": protocol["base_science_protocol"]["sha256"],
    }
    for name, path in code_paths.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError("frozen {} hash changed".format(name))
    if sha256_file(M20_PATH) != protocol["released_m20_and_postprocess"][
        "checkpoint_sha256"
    ]:
        raise RuntimeError("released M20 changed")
    if set(protocol["scope"]["sealed_outer_held_sources"]) != set(
        protocol["source_groups"][GROUP_HELD]
    ):
        raise RuntimeError("sealed G2 membership changed")
    if set(protocol["scope"]["inner_fit_sources"]) & set(
        protocol["scope"]["sealed_outer_held_sources"]
    ):
        raise RuntimeError("fit and held groups overlap")
    return protocol


def build_c00(protocol):
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml", list(C00_OVERRIDES)
    )
    frozen = protocol["released_m20_and_postprocess"]
    threshold = float(frozen["prediction_threshold"])
    c00 = component_crossfit.validate_c00_config(cfg, threshold)
    if component_crossfit.sha256_json(c00) != frozen["effective_c00_sha256"]:
        raise RuntimeError("effective C00 changed")
    return cfg, c00


def safe_peak_mib():
    try:
        return float(torch.cuda.max_memory_reserved(0) / (1024 ** 2))
    except Exception:
        return None


def forbidden_hashes(protocol):
    return {
        item["sha256"] for item in protocol["permanently_forbidden_old_or_resource_artifacts"]
    }


def assert_not_forbidden(path, protocol):
    digest = sha256_file(path)
    if digest in forbidden_hashes(protocol):
        raise RuntimeError("formal artifact matches a forbidden old/resource checkpoint")
    return digest


def source_path_and_metadata(protocol, name, allowed):
    if name not in allowed:
        raise RuntimeError("source escaped command firewall")
    path = TRAIN_ROOT / name
    metadata = protocol["sources"][name]
    if not path.is_file() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError("source identity changed: {}".format(name))
    return path, metadata


def validate_h2_input(metadata, polarities):
    if (
        int(len(polarities)) != int(metadata["event_count"])
        or not use_h2_atomic_deletion(len(polarities), polarities)
        or not np.isclose(
            complete_input_polarity_minority_fraction(polarities),
            float(metadata["polarity_minority_fraction"]),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise RuntimeError("exact input-only H2 route changed")


def materialize_fit_view(protocol, fit_id, fit_names):
    view_root = FIT_OUTPUT / "views" / fit_id
    view_root.mkdir(parents=True, exist_ok=False)
    for name in fit_names:
        source, _ = source_path_and_metadata(
            protocol, name, set(protocol["scope"]["inner_fit_sources"])
        )
        destination = view_root / name
        os.link(source, destination)
        if sha256_file(destination) != protocol["sources"][name]["sha256"]:
            raise RuntimeError("hard-link fit view identity changed")
    return view_root


def train_stage1(protocol, fit_id, fit_names, device):
    stage = protocol["stage1"]["training"]
    setup_seed(stage["seed"])
    view_root = materialize_fit_view(protocol, fit_id, fit_names)
    model, parent_payload = build_expert_model_from_m20(
        M20_PATH, input_mode="activity_control", device=device
    )
    trainable_names = configure_expert_only_training(model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if (
        len(trainable) != int(protocol["stage1"]["trainable_tensor_count"])
        or sum(parameter.numel() for parameter in trainable.values())
        != int(protocol["stage1"]["trainable_parameter_count"])
        or tuple(sorted(trainable)) != trainable_names
    ):
        raise RuntimeError("Stage1 trainable scope changed")
    initial_expert = {
        name: parameter.detach().cpu().clone() for name, parameter in trainable.items()
    }
    parent_state = parent_payload["model_state_dict"]
    parent_hash_before = state_sha256(parent_state)
    optimizer = torch.optim.AdamW(
        trainable.values(),
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(stage["epochs"]),
        eta_min=float(stage["scheduler_min_lr"]),
    )
    dataset = TemporalMemoryTrainDataset(
        root=view_root,
        whole_t=WHOLE_T,
        temporal_bin_size=TEMPORAL_BIN_SIZE,
        context_bins=CONTEXT_BINS,
        sequence_length=int(stage["sequence_length"]),
        width=WIDTH,
        height=HEIGHT,
        views_per_video=int(stage["views_per_source"]),
        positive_frame_probability=float(stage["positive_frame_probability"]),
        random_seed=int(stage["seed"]),
        log_count_clip=LOG_COUNT_CLIP,
        cache_all_videos=False,
        cache_video_count=int(stage["cache_video_count"]),
        dense_sampling_enabled=bool(stage["dense_sampling_enabled"]),
    )
    if [path.name for path in dataset.file_paths] != list(fit_names):
        raise RuntimeError("Stage1 dataset membership/order changed")
    # The dataset will open every fit array during training.  Register the
    # scope before the first batch so a mid-Stage1 failure receipt is exact.
    for name in fit_names:
        if name not in PROCESS_CONTEXT["source_arrays_opened"]:
            PROCESS_CONTEXT["source_arrays_opened"].append(name)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_memory_collate,
        pin_memory=True,
    )
    expected_steps = int(stage["epochs"]) * len(fit_names) * int(
        stage["views_per_source"]
    )
    gradient_seen = {name: False for name in trainable}
    diagnostics = []
    steps = 0
    for epoch in range(int(stage["epochs"])):
        dataset.set_epoch(epoch)
        model.eval()
        model.high_density_expert.train()
        losses = []
        gradient_norms = []
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).unsqueeze(0)
            time_indices = batch["event_time_indices"].to(device, non_blocking=True)
            event_x = batch["event_x"].to(device, non_blocking=True)
            event_y = batch["event_y"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            maps = model(frames).squeeze(0)
            logits = maps[time_indices, 0, event_y, event_x]
            loss, _ = frame_balanced_event_bce(
                logits,
                labels,
                time_indices,
                target_positive_loss_mass=float(stage["target_positive_loss_mass"]),
                max_positive_weight=float(stage["max_positive_weight"]),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Stage1 loss is non-finite")
            loss.backward()
            for name, parameter in trainable.items():
                gradient = parameter.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError("Stage1 gradient is missing or non-finite")
                gradient_seen[name] |= float(gradient.detach().abs().sum().item()) > 0.0
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable.values(), float(stage["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Stage1 gradient norm is non-finite")
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
        scheduler.step()
        diagnostics.append(
            {
                "epoch": epoch,
                "steps": len(losses),
                "mean_loss": float(np.mean(losses)),
                "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
            }
        )
    if steps != expected_steps or not all(gradient_seen.values()):
        raise RuntimeError("Stage1 optimizer-step or gradient gate failed")
    trained_state = model.state_dict()
    updated = {
        name: not torch.equal(parameter.detach().cpu(), initial_expert[name])
        for name, parameter in trainable.items()
    }
    inherited_equal = all(
        name in trained_state
        and torch.equal(trained_state[name].detach().cpu(), value.detach().cpu())
        for name, value in parent_state.items()
    )
    if not all(updated.values()) or not inherited_equal or state_sha256(parent_state) != parent_hash_before:
        raise RuntimeError("Stage1 update/frozen-parent gate failed")
    checkpoint_path = FIT_OUTPUT / "checkpoints" / "stage1_{}_final.pt".format(fit_id)
    save_torch_exclusive(
        checkpoint_path,
        {
            "checkpoint_format_version": 3,
            "schema": "ev-uav-activity-selective-recovery-fresh-stage1-formal-v1",
            "model_state_dict": trained_state,
            "temporal_memory": parent_payload["temporal_memory"],
            "high_density_expert": {
                "schema": "ev-uav-high-density-dual-expert-v1",
                "input_mode": "activity_control",
                "domain": "h2",
                "insertion_point": "level1",
                "hidden_channels": 16,
            },
            "provenance": {
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "released_m20_sha256": sha256_file(M20_PATH),
                "fit_id": fit_id,
                "fit_names": list(fit_names),
                "formal_fresh": True,
                "old_or_resource_checkpoint_reused": False,
                "training_scope": {
                    "name": "high_density_expert_only",
                    "trainable_state_tensor_count": 14,
                    "trainable_parameter_count": 1712,
                    "trainable_names": list(trainable_names),
                    "inherited_m20_bitwise_frozen": True,
                },
            },
        },
    )
    checkpoint_sha = assert_not_forbidden(checkpoint_path, protocol)
    model.to("cpu")
    del model, optimizer, scheduler, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "fit_id": fit_id,
        "fit_names": list(fit_names),
        "optimizer_steps": steps,
        "expected_optimizer_steps": expected_steps,
        "all_14_tensors_nonzero_finite_gradient": all(gradient_seen.values()),
        "all_14_tensors_updated": all(updated.values()),
        "inherited_m20_bitwise_frozen": inherited_equal,
        "parent_state_sha256": parent_hash_before,
        "diagnostics": diagnostics,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
    }


def verify_fresh_stage1_payload(protocol, payload, fit_id, fit_names):
    provenance = payload.get("provenance", {})
    checks = {
        "schema": payload.get("schema")
        == "ev-uav-activity-selective-recovery-fresh-stage1-formal-v1",
        "format": int(payload.get("checkpoint_format_version", -1)) == 3,
        "mode": payload.get("high_density_expert", {}).get("input_mode")
        == "activity_control",
        "protocol": provenance.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "runner": provenance.get("runner_sha256")
        == sha256_file(Path(__file__).resolve()),
        "m20": provenance.get("released_m20_sha256")
        == protocol["released_m20_and_postprocess"]["checkpoint_sha256"],
        "fit_id": provenance.get("fit_id") == fit_id,
        "fit_names": provenance.get("fit_names") == list(fit_names),
        "formal_fresh": provenance.get("formal_fresh") is True,
        "no_old_reuse": provenance.get("old_or_resource_checkpoint_reused")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "fresh Stage1 checkpoint provenance failed: {}".format(
                [name for name, value in checks.items() if not value]
            )
        )
    return checks


def flatten_sequences(sequences, channels, spatial_size):
    if sequences:
        return np.concatenate(sequences, axis=0)
    return np.empty((0, channels, spatial_size, spatial_size), dtype=np.float16)


def persist_label_free_source(
    path,
    source_name,
    generator_fit_id,
    generator_checkpoint_sha,
    locations,
    m20_raw,
    m20_post,
    activity_raw,
    activity_post,
    disagreement,
    patches,
    trajectories,
):
    offsets, events = component_offsets(disagreement.event_indices)
    patch_offsets = sequence_offsets(patches)
    flat_patches = flatten_sequences(patches, RECOVERY_PATCH_CHANNELS, 15)
    flat_trajectory = (
        np.concatenate(trajectories, axis=0).astype(np.float32)
        if trajectories
        else np.empty((0, 8), dtype=np.float32)
    )
    write_npz_exclusive(
        path,
        artifact_schema=np.asarray(
            "ev-uav-activity-selective-recovery-formal-source-features-v1"
        ),
        source_name=np.asarray(source_name),
        generator_fit_id=np.asarray(generator_fit_id),
        generator_checkpoint_sha256=np.asarray(generator_checkpoint_sha),
        formal_fresh=np.asarray(True),
        contains_labels_or_target_ids=np.asarray(False),
        m20_raw_scores=np.asarray(m20_raw, dtype=np.float32),
        m20_c00_scores=np.asarray(m20_post, dtype=np.float32),
        activity_raw_scores=np.asarray(activity_raw, dtype=np.float32),
        activity_c00_scores=np.asarray(activity_post, dtype=np.float32),
        locations=np.asarray(locations, dtype=np.int16),
        disagreement_m20_component_ids=disagreement.m20_component_ids,
        disagreement_missing_event_counts=disagreement.missing_event_counts,
        disagreement_activity_supported_event_counts=(
            disagreement.activity_supported_event_counts
        ),
        disagreement_component_offsets=offsets,
        disagreement_component_event_indices=events,
        patch_offsets=patch_offsets,
        recovery_patches=flat_patches,
        trajectory_context=flat_trajectory,
    )
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def extract_source_label_free(
    protocol,
    source_name,
    allowed_names,
    generator_fit_id,
    generator_checkpoint_sha,
    activity_model,
    m20_model,
    cfg,
    output_dir,
    device,
):
    source_path, metadata = source_path_and_metadata(
        protocol, source_name, set(allowed_names)
    )
    video, polarities, locations = _load_input_only(source_path)
    if source_name not in PROCESS_CONTEXT["source_arrays_opened"]:
        PROCESS_CONTEXT["source_arrays_opened"].append(source_name)
    validate_h2_input(metadata, polarities)
    threshold = float(
        protocol["released_m20_and_postprocess"]["prediction_threshold"]
    )
    m20_raw, m20_decoder, m20_logits = dense_full_stream(m20_model, video, device)
    activity_raw, activity_decoder, activity_logits = dense_full_stream(
        activity_model, video, device
    )
    m20_post, m20_c00_stats = apply_c00(m20_raw, locations, cfg, threshold)
    activity_post, activity_c00_stats = apply_c00(
        activity_raw, locations, cfg, threshold
    )
    disagreement = extract_disagreement_components(
        m20_post,
        activity_post,
        locations,
        threshold,
        spatial_radius=2,
        temporal_bin_size=50,
        temporal_radius_bins=1,
    )
    if disagreement.event_indices:
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
    else:
        patches, trajectories = tuple(), tuple()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = persist_label_free_source(
        output_dir / "{}_label_free_features.npz".format(source_name[:-4]),
        source_name,
        generator_fit_id,
        generator_checkpoint_sha,
        locations,
        m20_raw,
        m20_post,
        activity_raw,
        activity_post,
        disagreement,
        patches,
        trajectories,
    )
    del m20_decoder, m20_logits, activity_decoder, activity_logits
    gc.collect()
    return {
        "source_name": source_name,
        "source_path": source_path,
        "source_sha256": metadata["sha256"],
        "generator_fit_id": generator_fit_id,
        "generator_checkpoint_sha256": generator_checkpoint_sha,
        "locations": locations,
        "m20_raw": m20_raw,
        "m20_post": m20_post,
        "activity_raw": activity_raw,
        "activity_post": activity_post,
        "disagreement": disagreement,
        "patches": patches,
        "trajectories": trajectories,
        "m20_c00_stats": m20_c00_stats,
        "activity_c00_stats": activity_c00_stats,
        "label_free_artifact": artifact,
    }


def counts_payload(counts):
    return {"counts": counts.to_dict(), "metrics": metrics_from_counts(counts)}


def attach_fit_truth(protocol, record, output_dir):
    labels, target_ids = _load_truth(record["source_path"])
    threshold = float(
        protocol["released_m20_and_postprocess"]["prediction_threshold"]
    )

    def official_score(scores):
        return metrics_from_counts(
            sufficient_counts_for_video(
                scores,
                labels,
                target_ids,
                record["locations"],
                prediction_threshold=threshold,
            )
        )["score"]

    targets, deltas = marginal_recovery_targets(
        record["m20_post"],
        record["activity_post"],
        record["disagreement"].event_indices,
        threshold,
        official_score,
    )
    label_path = output_dir / "{}_fit_only_labels.npz".format(
        record["source_name"][:-4]
    )
    write_npz_exclusive(
        label_path,
        artifact_schema=np.asarray(
            "ev-uav-activity-selective-recovery-formal-fit-labels-v1"
        ),
        source_name=np.asarray(record["source_name"]),
        generator_fit_id=np.asarray(record["generator_fit_id"]),
        contains_fit_only_labels=np.asarray(True),
        formal_inference_feature_allowed=np.asarray(False),
        marginal_recovery_targets=np.asarray(targets, dtype=np.uint8),
        marginal_official_score_deltas=np.asarray(deltas, dtype=np.float64),
    )
    record.update(
        {
            "labels": labels,
            "target_ids": target_ids,
            "recovery_targets": targets,
            "marginal_score_deltas": deltas,
            "fit_label_artifact": {
                "path": str(label_path.resolve()),
                "sha256": sha256_file(label_path),
            },
            "m20_counts_payload": counts_payload(
                sufficient_counts_for_video(
                    record["m20_post"],
                    labels,
                    target_ids,
                    record["locations"],
                    prediction_threshold=threshold,
                )
            ),
            "activity_counts_payload": counts_payload(
                sufficient_counts_for_video(
                    record["activity_post"],
                    labels,
                    target_ids,
                    record["locations"],
                    prediction_threshold=threshold,
                )
            ),
        }
    )
    return record


def collect_true_oof_group(
    protocol,
    target_group,
    source_names,
    generator_fit_id,
    generator_fit_names,
    generator_checkpoint_path,
    cfg,
    device,
):
    checkpoint_sha = assert_not_forbidden(generator_checkpoint_path, protocol)
    activity_model, activity_payload = _load_trained_expert_model(
        generator_checkpoint_path, device
    )
    verify_fresh_stage1_payload(
        protocol, activity_payload, generator_fit_id, generator_fit_names
    )
    for parameter in activity_model.parameters():
        parameter.requires_grad_(False)
    m20_model, _ = build_released_m20(device)
    for parameter in m20_model.parameters():
        parameter.requires_grad_(False)
    inherited_equal = all(
        name in activity_payload["model_state_dict"]
        and torch.equal(
            activity_payload["model_state_dict"][name].detach().cpu(),
            value.detach().cpu(),
        )
        for name, value in m20_model.state_dict().items()
    )
    if not inherited_equal:
        raise RuntimeError("fresh Stage1 inherited parent differs from released M20")
    activity_before = state_sha256(activity_model.state_dict())
    m20_before = state_sha256(m20_model.state_dict())
    records = []
    feature_dir = FIT_OUTPUT / "oof_features" / target_group
    label_dir = FIT_OUTPUT / "oof_fit_labels" / target_group
    feature_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir(parents=True, exist_ok=False)
    for source_name in source_names:
        record = extract_source_label_free(
            protocol,
            source_name,
            protocol["scope"]["inner_fit_sources"],
            generator_fit_id,
            checkpoint_sha,
            activity_model,
            m20_model,
            cfg,
            feature_dir,
            device,
        )
        records.append(attach_fit_truth(protocol, record, label_dir))
    activity_after = state_sha256(activity_model.state_dict())
    m20_after = state_sha256(m20_model.state_dict())
    if activity_before != activity_after or m20_before != m20_after:
        raise RuntimeError("frozen model changed during OOF feature generation")
    activity_model.to("cpu")
    m20_model.to("cpu")
    del activity_model, m20_model, activity_payload
    gc.collect()
    torch.cuda.empty_cache()
    return records, {
        "target_group": target_group,
        "generator_fit_id": generator_fit_id,
        "generator_fit_names": list(generator_fit_names),
        "generator_checkpoint_sha256": checkpoint_sha,
        "source_names": list(source_names),
        "activity_state_sha256_before": activity_before,
        "activity_state_sha256_after": activity_after,
        "m20_state_sha256_before": m20_before,
        "m20_state_sha256_after": m20_after,
        "inherited_m20_bitwise_equal": inherited_equal,
        "component_count": int(
            sum(len(record["disagreement"].event_indices) for record in records)
        ),
        "marginal_positive_count": int(
            sum(np.count_nonzero(record["recovery_targets"] == 1) for record in records)
        ),
        "marginal_negative_count": int(
            sum(np.count_nonzero(record["recovery_targets"] == 0) for record in records)
        ),
    }


def recovery_items(records, include_targets=True):
    items = []
    source_ids = []
    targets = []
    component_map = []
    for source_id, record in enumerate(records):
        component_count = len(record["disagreement"].event_indices)
        if include_targets and len(record["recovery_targets"]) != component_count:
            raise RuntimeError("component/target count mismatch")
        for component_id in range(component_count):
            target = (
                int(record["recovery_targets"][component_id])
                if include_targets
                else 0
            )
            items.append(
                {
                    "patches": record["patches"][component_id],
                    "trajectory": record["trajectories"][component_id],
                    "target": float(target),
                    "weight": 1.0,
                }
            )
            source_ids.append(source_id)
            targets.append(target)
            component_map.append((source_id, component_id))
    return (
        items,
        np.asarray(source_ids, dtype=np.int64),
        np.asarray(targets, dtype=np.uint8),
        tuple(component_map),
    )


def predict_recovery_items(model, items, batch_size, device):
    raw_probabilities = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), int(batch_size)):
            batch = recovery_sequence_collate(items[start : start + int(batch_size)])
            logits, embedding, _ = model(
                batch["patches"].to(device),
                batch["trajectory"].to(device),
                batch["lengths"].to(device),
                return_embedding=True,
            )
            raw_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    if not raw_probabilities:
        return np.empty(0, dtype=np.float64), np.empty((0, 64), dtype=np.float32)
    probabilities = np.concatenate(raw_probabilities).astype(np.float64)
    embeddings = np.concatenate(embeddings).astype(np.float32)
    if not np.isfinite(probabilities).all() or not np.isfinite(embeddings).all():
        raise RuntimeError("Stage2 prediction is non-finite")
    return probabilities, embeddings


def train_stage2(protocol, fit_id, records, device):
    stage = protocol["stage2"]["training"]
    items, source_ids, targets, _ = recovery_items(records, include_targets=True)
    if not items or set(np.unique(targets).tolist()) != {0, 1}:
        raise RuntimeError("Stage2 fit OOF lacks both real marginal classes")
    weights = source_class_balanced_weights(source_ids, targets)
    for index, weight in enumerate(weights):
        items[index]["weight"] = float(weight)
    setup_seed(stage["seed"])
    model = DisagreementRecoveryNet().to(device)
    if recovery_parameter_count(model) != int(protocol["stage2"]["parameter_count"]):
        raise RuntimeError("Stage2 parameter count changed")
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    gradient_seen = {name: False for name, _ in model.named_parameters()}
    branch_seen = {
        prefix: False
        for prefix in (
            "semantic_stem.",
            "context_stem.",
            "spatial_fusion.",
            "trajectory_encoder.",
            "temporal.",
            "temporal_attention.",
            "classifier.",
        )
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    schedule = deterministic_epoch_batches(
        len(items),
        int(stage["component_batch_size"]),
        int(stage["epochs"]),
        int(stage["seed"]),
    )
    diagnostics_by_epoch = []
    current_epoch = None
    epoch_losses = []
    epoch_norms = []
    for entry in schedule:
        if current_epoch is None:
            current_epoch = int(entry["epoch"])
        if int(entry["epoch"]) != current_epoch:
            diagnostics_by_epoch.append(
                {
                    "epoch": current_epoch,
                    "steps": len(epoch_losses),
                    "mean_loss": float(np.mean(epoch_losses)),
                    "mean_preclip_gradient_norm": float(np.mean(epoch_norms)),
                }
            )
            current_epoch = int(entry["epoch"])
            epoch_losses = []
            epoch_norms = []
        indices = entry["indices"].tolist()
        batch = recovery_sequence_collate([items[index] for index in indices])
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["patches"].to(device),
            batch["trajectory"].to(device),
            batch["lengths"].to(device),
        )
        losses = functional.binary_cross_entropy_with_logits(
            logits, batch["targets"].to(device), reduction="none"
        )
        batch_weights = batch["weights"].to(device)
        loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Stage2 formal loss is non-finite")
        loss.backward()
        branch_gradient = {prefix: 0.0 for prefix in branch_seen}
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Stage2 gradient is missing or non-finite")
            magnitude = float(gradient.detach().abs().sum().item())
            gradient_seen[name] |= magnitude > 0.0
            for prefix in branch_gradient:
                if name.startswith(prefix):
                    branch_gradient[prefix] += magnitude
        for prefix, magnitude in branch_gradient.items():
            branch_seen[prefix] |= magnitude > 0.0
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(stage["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Stage2 gradient norm is non-finite")
        optimizer.step()
        epoch_losses.append(float(loss.detach().item()))
        epoch_norms.append(float(gradient_norm.detach().item()))
    diagnostics_by_epoch.append(
        {
            "epoch": current_epoch,
            "steps": len(epoch_losses),
            "mean_loss": float(np.mean(epoch_losses)),
            "mean_preclip_gradient_norm": float(np.mean(epoch_norms)),
        }
    )
    updated = {
        name: not torch.equal(parameter.detach().cpu(), initial[name])
        for name, parameter in model.named_parameters()
    }
    if not all(gradient_seen.values()) or not all(branch_seen.values()) or not all(
        updated.values()
    ):
        raise RuntimeError("Stage2 gradient/update scope gate failed")
    fit_probabilities, fit_embeddings = predict_recovery_items(
        model, items, int(stage["component_batch_size"]), device
    )
    negative_reference = fit_probabilities[targets == 0]
    if negative_reference.size == 0:
        raise RuntimeError("Stage2 fit negative reference is empty")
    checkpoint_path = FIT_OUTPUT / "checkpoints" / "stage2_{}_final.pt".format(
        fit_id
    )
    save_torch_exclusive(
        checkpoint_path,
        {
            "checkpoint_format_version": 1,
            "schema": "ev-uav-activity-selective-recovery-fresh-stage2-formal-v1",
            "model_state_dict": model.state_dict(),
            "fit_negative_reference_probabilities": negative_reference,
            "provenance": {
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "fit_id": fit_id,
                "fit_source_names": [record["source_name"] for record in records],
                "fit_features_are_true_oof": True,
                "generator_fit_ids": [
                    record["generator_fit_id"] for record in records
                ],
                "old_or_resource_checkpoint_reused": False,
                "optimizer_steps": len(schedule),
                "final_epoch_only": True,
            },
        },
    )
    checkpoint_sha = assert_not_forbidden(checkpoint_path, protocol)
    return {
        "model": model,
        "optimizer": optimizer,
        "negative_reference": negative_reference,
        "fit_probabilities": fit_probabilities,
        "fit_embeddings": fit_embeddings,
        "audit": {
            "fit_id": fit_id,
            "fit_source_names": [record["source_name"] for record in records],
            "component_count": len(items),
            "positive_count": int(np.count_nonzero(targets == 1)),
            "negative_count": int(np.count_nonzero(targets == 0)),
            "optimizer_steps": len(schedule),
            "expected_steps_formula": int(stage["epochs"])
            * int(math.ceil(len(items) / int(stage["component_batch_size"]))),
            "all_parameters_nonzero_finite_gradient": all(gradient_seen.values()),
            "all_branches_nonzero_gradient": all(branch_seen.values()),
            "all_parameters_updated": all(updated.values()),
            "diagnostics": diagnostics_by_epoch,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "fit_negative_reference_count": int(negative_reference.size),
        },
    }


def predict_records_with_recovery(
    protocol, model, negative_reference, records, prediction_id, device
):
    batch_size = int(protocol["stage2"]["training"]["component_batch_size"])
    output_dir = FIT_OUTPUT / "nested_oof_predictions" / prediction_id
    output_dir.mkdir(parents=True, exist_ok=False)
    for record in records:
        items, _, _, _ = recovery_items([record], include_targets=False)
        raw, embeddings = predict_recovery_items(model, items, batch_size, device)
        confidence = negative_reference_conformal_confidence(raw, negative_reference)
        if confidence.size != len(record["disagreement"].event_indices):
            raise RuntimeError("nested OOF confidence count changed")
        artifact_path = output_dir / "{}_predictions.npz".format(
            record["source_name"][:-4]
        )
        write_npz_exclusive(
            artifact_path,
            artifact_schema=np.asarray(
                "ev-uav-activity-selective-recovery-nested-oof-predictions-v1"
            ),
            source_name=np.asarray(record["source_name"]),
            prediction_id=np.asarray(prediction_id),
            contains_labels_or_target_ids=np.asarray(False),
            raw_recovery_probabilities=raw,
            conformal_confidences=confidence,
            recovery_embeddings=embeddings,
            disagreement_m20_component_ids=(
                record["disagreement"].m20_component_ids
            ),
        )
        record["raw_recovery_probabilities"] = raw
        record["recovery_embeddings"] = embeddings
        record["confidences"] = confidence
        record["prediction_artifact"] = {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
        }
    return records


def load_stage2_checkpoint(protocol, checkpoint_path, expected_fit_id, device):
    checkpoint_sha = assert_not_forbidden(checkpoint_path, protocol)
    payload = torch.load(checkpoint_path, map_location="cpu")
    provenance = payload.get("provenance", {})
    if not (
        payload.get("schema")
        == "ev-uav-activity-selective-recovery-fresh-stage2-formal-v1"
        and provenance.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
        and provenance.get("runner_sha256")
        == sha256_file(Path(__file__).resolve())
        and provenance.get("fit_id") == expected_fit_id
        and provenance.get("fit_features_are_true_oof") is True
        and provenance.get("old_or_resource_checkpoint_reused") is False
    ):
        raise RuntimeError("fresh Stage2 checkpoint provenance failed")
    model = DisagreementRecoveryNet().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    reference = np.asarray(
        payload["fit_negative_reference_probabilities"], dtype=np.float64
    )
    if reference.size == 0 or not np.isfinite(reference).all():
        raise RuntimeError("frozen negative reference is invalid")
    return model, reference, payload, checkpoint_sha


def add_payload_counts(payloads):
    total = SufficientCounts()
    for payload in payloads:
        total = total + SufficientCounts(**payload["counts"])
    return counts_payload(total)


def build_inner_replay(protocol, records_by_group, cutoff):
    threshold = float(
        protocol["released_m20_and_postprocess"]["prediction_threshold"]
    )
    group_payloads = {}
    source_payloads = []
    recovered_count = 0
    all_atomic = True
    for group in FIT_GROUPS:
        group_sources = []
        for record in records_by_group[group]:
            candidate, receipt = atomic_recover_or_identity(
                record["m20_post"],
                record["activity_post"],
                record["disagreement"].event_indices,
                record["confidences"],
                float(cutoff),
                threshold,
                enabled=True,
            )
            integrity = bool(
                receipt.complete_components_only
                and receipt.activity_outside_recovery_bitwise_equal
                and receipt.recovered_m20_scores_bitwise_equal
                and receipt.fallback_reason is None
            )
            all_atomic &= integrity
            recovered_count += int(receipt.recovered_component_count)
            candidate_payload = counts_payload(
                sufficient_counts_for_video(
                    candidate,
                    record["labels"],
                    record["target_ids"],
                    record["locations"],
                    prediction_threshold=threshold,
                )
            )
            source_payload = {
                "source_name": record["source_name"],
                "group": group,
                "m20": record["m20_counts_payload"],
                "activity": record["activity_counts_payload"],
                "candidate": candidate_payload,
                "atomic_integrity": integrity,
                "atomic_recovery": asdict(receipt),
            }
            group_sources.append(source_payload)
            source_payloads.append(source_payload)
        group_payloads[group] = {
            "m20": add_payload_counts([item["m20"] for item in group_sources]),
            "activity": add_payload_counts(
                [item["activity"] for item in group_sources]
            ),
            "candidate": add_payload_counts(
                [item["candidate"] for item in group_sources]
            ),
            "atomic_integrity": all(item["atomic_integrity"] for item in group_sources),
        }
    return {
        "cutoff": float(cutoff),
        "recovered_component_count": recovered_count,
        "all_atomic_integrity": all_atomic,
        "sources": source_payloads,
        "groups": group_payloads,
        "pooled": {
            stream: add_payload_counts(
                [item[stream] for item in source_payloads]
            )
            for stream in ("m20", "activity", "candidate")
        },
    }


def compact_record_artifacts(records_by_group):
    output = []
    for group, records in records_by_group.items():
        for record in records:
            output.append(
                {
                    "group": group,
                    "source_name": record["source_name"],
                    "generator_fit_id": record["generator_fit_id"],
                    "generator_checkpoint_sha256": record[
                        "generator_checkpoint_sha256"
                    ],
                    "component_count": len(record["disagreement"].event_indices),
                    "marginal_positive_count": int(
                        np.count_nonzero(record["recovery_targets"] == 1)
                    ),
                    "marginal_negative_count": int(
                        np.count_nonzero(record["recovery_targets"] == 0)
                    ),
                    "label_free_artifact": record["label_free_artifact"],
                    "fit_label_artifact": record["fit_label_artifact"],
                    "prediction_artifact": record["prediction_artifact"],
                }
            )
    return output


def write_training_failure(error):
    if not FIT_OUTPUT.exists():
        return None
    path = FIT_OUTPUT / "training_failure.json"
    if path.exists():
        return path
    write_json_exclusive(
        path,
        {
            "schema": "ev-uav-activity-selective-recovery-formal-training-failure-v1",
            "created_utc": utc_now(),
            "status": "failed_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "milestone": PROCESS_CONTEXT["milestone"],
            "source_arrays_opened": PROCESS_CONTEXT["source_arrays_opened"],
            "held_g2_array_read": PROCESS_CONTEXT["held_opened"],
            "validation_or_test_read": False,
            "formal_held_started": False,
            "protocol_sha256": (
                sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "peak_cuda_mib": safe_peak_mib(),
            "elapsed_seconds": (
                None
                if PROCESS_CONTEXT["started"] is None
                else time.time() - PROCESS_CONTEXT["started"]
            ),
        },
    )
    return path


def run_train_and_freeze(protocol, authorized):
    if not authorized:
        raise PermissionError("explicit formal GPU authorization is required")
    if FIT_OUTPUT.exists() or HELD_OUTPUT.exists():
        raise FileExistsError("refusing to overwrite formal activity-recovery output")
    if python_gpu_processes():
        raise RuntimeError("another Python GPU process exists before formal fit")
    FIT_OUTPUT.mkdir(parents=True, exist_ok=False)
    PROCESS_CONTEXT.update(
        {
            "command": "train-and-freeze",
            "started": time.time(),
            "milestone": "fit_output_created",
            "source_arrays_opened": [],
            "held_opened": False,
        }
    )
    device = torch.device("cuda:0")
    runner_sha = sha256_file(Path(__file__).resolve())
    with gpu_run_lock("h2_activity_selective_recovery_fresh_g2_fit"):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        cfg, c00 = build_c00(protocol)
        group_sources = protocol["source_groups"]

        stage1_g1 = train_stage1(
            protocol, "fit_g1", group_sources[GROUP_G1], device
        )
        PROCESS_CONTEXT["milestone"] = "fresh_stage1_g1_complete"
        stage1_g3 = train_stage1(
            protocol, "fit_g3", group_sources[GROUP_G3], device
        )
        PROCESS_CONTEXT["milestone"] = "fresh_stage1_g3_complete"

        # True OOF: the Stage1 generator never trained on its target group.
        g1_records, g1_oof_audit = collect_true_oof_group(
            protocol,
            GROUP_G1,
            group_sources[GROUP_G1],
            "fit_g3",
            group_sources[GROUP_G3],
            Path(stage1_g3["checkpoint_path"]),
            cfg,
            device,
        )
        PROCESS_CONTEXT["milestone"] = "g1_true_oof_complete"
        g3_records, g3_oof_audit = collect_true_oof_group(
            protocol,
            GROUP_G3,
            group_sources[GROUP_G3],
            "fit_g1",
            group_sources[GROUP_G1],
            Path(stage1_g1["checkpoint_path"]),
            cfg,
            device,
        )
        PROCESS_CONTEXT["milestone"] = "g3_true_oof_complete"
        records_by_group = {GROUP_G1: g1_records, GROUP_G3: g3_records}

        # Recovery fit G1-OOF predicts G3-OOF.
        stage2_g1 = train_stage2(protocol, "fit_g1_oof", g1_records, device)
        predict_records_with_recovery(
            protocol,
            stage2_g1["model"],
            stage2_g1["negative_reference"],
            g3_records,
            "fit_g1_oof_predict_g3_oof",
            device,
        )
        stage2_g1["model"].to("cpu")
        del stage2_g1["model"], stage2_g1["optimizer"]
        gc.collect()
        torch.cuda.empty_cache()
        PROCESS_CONTEXT["milestone"] = "stage2_g1_predict_g3_complete"

        # Recovery fit G3-OOF predicts G1-OOF.
        stage2_g3 = train_stage2(protocol, "fit_g3_oof", g3_records, device)
        predict_records_with_recovery(
            protocol,
            stage2_g3["model"],
            stage2_g3["negative_reference"],
            g1_records,
            "fit_g3_oof_predict_g1_oof",
            device,
        )
        stage2_g3["model"].to("cpu")
        del stage2_g3["model"], stage2_g3["optimizer"]
        gc.collect()
        torch.cuda.empty_cache()
        PROCESS_CONTEXT["milestone"] = "nested_oof_predictions_complete"

        cutoffs = exact_confidence_cutoffs(
            [record["confidences"] for records in records_by_group.values() for record in records]
        )
        replays = [
            build_inner_replay(protocol, records_by_group, cutoff)
            for cutoff in cutoffs
        ]
        selected, replay_audit = select_qualifying_inner_replay(
            replays,
            groups=FIT_GROUPS,
            pooled_score_gain_minimum=float(
                protocol["inner_cutoff_and_gate"]["pooled"][
                    "score_gain_at_least"
                ]
            ),
            absolute_pd_delta_maximum=float(
                protocol["inner_cutoff_and_gate"]["each_group"][
                    "absolute_pd_delta_at_most"
                ]
            ),
        )
        replay_path = FIT_OUTPUT / "inner_exact_breakpoint_replay.json"
        replay_payload = {
            "schema": "ev-uav-activity-selective-recovery-inner-replay-v1",
            "created_utc": utc_now(),
            "candidate_count": len(cutoffs),
            "candidate_cutoffs": cutoffs.tolist(),
            "candidates": [
                {"replay": item["replay"], "gate": item["gate"]}
                for item in replay_audit
            ],
            "selected_cutoff": (
                None if selected is None else selected["replay"]["cutoff"]
            ),
            "selected_gate": None if selected is None else selected["gate"],
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": runner_sha,
            "held_g2_array_read": False,
        }
        write_json_exclusive(replay_path, replay_payload)
        replay_sha = sha256_file(replay_path)
        PROCESS_CONTEXT["milestone"] = "inner_exact_replay_persisted"

        if selected is None:
            peak_mib = safe_peak_mib()
            hard_peak = float(
                protocol["formal_step_and_resource_budget"]["hard_peak_cuda_mib"]
            )
            if peak_mib is None or peak_mib > hard_peak:
                raise RuntimeError(
                    "failed-inner formal fit exceeded or could not measure CUDA budget"
                )
            result_path = FIT_OUTPUT / "training_receipt.json"
            write_json_exclusive(
                result_path,
                {
                    "schema": "ev-uav-activity-selective-recovery-formal-training-v1",
                    "created_utc": utc_now(),
                    "status": "inner_failed_no_qualifying_cutoff",
                    "inner_passed": False,
                    "final_fit_started": False,
                    "held_g2_array_read": False,
                    "held_evaluation_allowed": False,
                    "source_arrays_opened": PROCESS_CONTEXT[
                        "source_arrays_opened"
                    ],
                    "stage1": [stage1_g1, stage1_g3],
                    "true_oof": [g1_oof_audit, g3_oof_audit],
                    "stage2": [stage2_g1["audit"], stage2_g3["audit"]],
                    "oof_artifacts": compact_record_artifacts(records_by_group),
                    "inner_replay_path": str(replay_path.resolve()),
                    "inner_replay_sha256": replay_sha,
                    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                    "runner_sha256": runner_sha,
                    "peak_cuda_mib": peak_mib,
                    "peak_cuda_budget_mib": hard_peak,
                    "elapsed_seconds": time.time() - PROCESS_CONTEXT["started"],
                },
            )
            PROCESS_CONTEXT["milestone"] = "inner_failed_closed"
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "status": "inner_failed_no_qualifying_cutoff",
                "training_receipt_path": str(result_path.resolve()),
                "training_receipt_sha256": sha256_file(result_path),
            }

        PROCESS_CONTEXT["milestone"] = "inner_gate_passed"
        union_sources = list(group_sources[GROUP_G1]) + list(group_sources[GROUP_G3])
        final_stage1 = train_stage1(protocol, "final_union_g1_g3", union_sources, device)
        PROCESS_CONTEXT["milestone"] = "final_stage1_frozen"
        final_stage2 = train_stage2(
            protocol, "final_union_true_oof", g1_records + g3_records, device
        )
        final_stage2["model"].to("cpu")
        del final_stage2["model"], final_stage2["optimizer"]
        gc.collect()
        torch.cuda.empty_cache()
        PROCESS_CONTEXT["milestone"] = "final_stage2_frozen"

        final_stage1_sha = assert_not_forbidden(
            Path(final_stage1["checkpoint_path"]), protocol
        )
        final_stage2_sha = assert_not_forbidden(
            Path(final_stage2["audit"]["checkpoint_path"]), protocol
        )
        strategy = {
            "schema": "ev-uav-activity-selective-recovery-frozen-strategy-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": runner_sha,
            "inner_replay_path": str(replay_path.resolve()),
            "inner_replay_sha256": replay_sha,
            "selected_cutoff": float(selected["replay"]["cutoff"]),
            "selected_gate": selected["gate"],
            "selected_group_and_pooled_metrics": {
                "groups": selected["replay"]["groups"],
                "pooled": selected["replay"]["pooled"],
                "recovered_component_count": selected["replay"][
                    "recovered_component_count"
                ],
            },
            "final_stage1_checkpoint": final_stage1["checkpoint_path"],
            "final_stage1_checkpoint_sha256": final_stage1_sha,
            "final_stage2_checkpoint": final_stage2["audit"]["checkpoint_path"],
            "final_stage2_checkpoint_sha256": final_stage2_sha,
            "final_stage2_negative_reference_count": final_stage2["audit"][
                "fit_negative_reference_count"
            ],
            "final_stage2_fit_features_are_true_oof": True,
            "old_or_resource_checkpoint_reused": False,
            "held_g2_array_read": False,
            "cutoff_adjustment_after_this_file_allowed": False,
        }
        write_json_exclusive(STRATEGY_PATH, strategy)
        strategy_sha = sha256_file(STRATEGY_PATH)
        peak_mib = safe_peak_mib()
        hard_peak = float(
            protocol["formal_step_and_resource_budget"]["hard_peak_cuda_mib"]
        )
        if peak_mib is None or peak_mib > hard_peak:
            raise RuntimeError("formal fit exceeded or could not measure CUDA budget")
        training_receipt = {
            "schema": "ev-uav-activity-selective-recovery-formal-training-v1",
            "created_utc": utc_now(),
            "status": "inner_passed_final_frozen",
            "inner_passed": True,
            "final_fit_completed": True,
            "held_g2_array_read": False,
            "held_evaluation_allowed": True,
            "source_arrays_opened": PROCESS_CONTEXT["source_arrays_opened"],
            "stage1": [stage1_g1, stage1_g3, final_stage1],
            "true_oof": [g1_oof_audit, g3_oof_audit],
            "stage2": [
                stage2_g1["audit"],
                stage2_g3["audit"],
                final_stage2["audit"],
            ],
            "oof_artifacts": compact_record_artifacts(records_by_group),
            "inner_replay_path": str(replay_path.resolve()),
            "inner_replay_sha256": replay_sha,
            "strategy_path": str(STRATEGY_PATH.resolve()),
            "strategy_sha256": strategy_sha,
            "final_stage1_checkpoint_sha256": final_stage1_sha,
            "final_stage2_checkpoint_sha256": final_stage2_sha,
            "forbidden_checkpoint_hashes_absent": all(
                digest not in forbidden_hashes(protocol)
                for digest in (final_stage1_sha, final_stage2_sha)
            ),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": runner_sha,
            "effective_c00_sha256": component_crossfit.sha256_json(c00),
            "peak_cuda_mib": peak_mib,
            "peak_cuda_budget_mib": hard_peak,
            "elapsed_seconds": time.time() - PROCESS_CONTEXT["started"],
        }
        write_json_exclusive(TRAIN_RECEIPT_PATH, training_receipt)
        PROCESS_CONTEXT["milestone"] = "inner_passed_final_hashes_frozen"
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "inner_passed_final_frozen",
            "training_receipt_path": str(TRAIN_RECEIPT_PATH.resolve()),
            "training_receipt_sha256": sha256_file(TRAIN_RECEIPT_PATH),
            "strategy_path": str(STRATEGY_PATH.resolve()),
            "strategy_sha256": strategy_sha,
            "peak_cuda_mib": peak_mib,
        }


def verify_training_chain(protocol):
    if not TRAIN_RECEIPT_PATH.is_file() or not STRATEGY_PATH.is_file():
        raise RuntimeError("final training receipt/strategy is absent")
    receipt = read_json(TRAIN_RECEIPT_PATH)
    strategy = read_json(STRATEGY_PATH)
    current_runner_sha = sha256_file(Path(__file__).resolve())
    if not (
        receipt.get("status") == "inner_passed_final_frozen"
        and receipt.get("inner_passed") is True
        and receipt.get("held_g2_array_read") is False
        and receipt.get("held_evaluation_allowed") is True
        and receipt.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
        and receipt.get("runner_sha256") == current_runner_sha
        and receipt.get("strategy_sha256") == sha256_file(STRATEGY_PATH)
        and strategy.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
        and strategy.get("runner_sha256") == current_runner_sha
        and strategy.get("inner_replay_sha256")
        == sha256_file(Path(strategy["inner_replay_path"]))
        and strategy.get("final_stage2_fit_features_are_true_oof") is True
        and strategy.get("old_or_resource_checkpoint_reused") is False
        and strategy.get("cutoff_adjustment_after_this_file_allowed") is False
    ):
        raise RuntimeError("training receipt/strategy chain failed")
    stage1_path = Path(strategy["final_stage1_checkpoint"])
    stage2_path = Path(strategy["final_stage2_checkpoint"])
    stage1_sha = assert_not_forbidden(stage1_path, protocol)
    stage2_sha = assert_not_forbidden(stage2_path, protocol)
    if (
        stage1_sha != strategy["final_stage1_checkpoint_sha256"]
        or stage2_sha != strategy["final_stage2_checkpoint_sha256"]
        or stage1_sha != receipt["final_stage1_checkpoint_sha256"]
        or stage2_sha != receipt["final_stage2_checkpoint_sha256"]
    ):
        raise RuntimeError("final checkpoint hash chain failed")
    return receipt, strategy, stage1_path, stage2_path


def persist_held_action_artifact(
    path,
    record,
    raw_probabilities,
    confidences,
    embeddings,
    cutoff,
    candidate,
    receipt,
):
    write_npz_exclusive(
        path,
        artifact_schema=np.asarray(
            "ev-uav-activity-selective-recovery-held-action-v1"
        ),
        source_name=np.asarray(record["source_name"]),
        contains_labels_or_target_ids=np.asarray(False),
        raw_recovery_probabilities=np.asarray(raw_probabilities, dtype=np.float64),
        conformal_confidences=np.asarray(confidences, dtype=np.float64),
        recovery_embeddings=np.asarray(embeddings, dtype=np.float32),
        frozen_cutoff=np.asarray(float(cutoff), dtype=np.float64),
        recovered_component=np.asarray(
            confidences >= float(cutoff), dtype=np.bool_
        ),
        final_candidate_scores=np.asarray(candidate, dtype=np.float32),
        complete_components_only=np.asarray(receipt.complete_components_only),
        activity_outside_recovery_bitwise_equal=np.asarray(
            receipt.activity_outside_recovery_bitwise_equal
        ),
        recovered_m20_scores_bitwise_equal=np.asarray(
            receipt.recovered_m20_scores_bitwise_equal
        ),
    )
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def held_promotion_gate(protocol, m20_payload, candidate_payload, atomic_integrity):
    gate = protocol["outer_held_once"]["promotion_pooled"]
    m20_counts = m20_payload["counts"]
    candidate_counts = candidate_payload["counts"]
    m20_metrics = m20_payload["metrics"]
    candidate_metrics = candidate_payload["metrics"]
    checks = {
        "score_gain_at_least": float(
            candidate_metrics["score"] - m20_metrics["score"]
        )
        >= float(gate["score_gain_at_least"]),
        "iou_not_lower": float(candidate_metrics["iou"])
        >= float(m20_metrics["iou"]),
        "absolute_pd_delta_at_most": abs(
            float(candidate_metrics["pd"] - m20_metrics["pd"])
        )
        <= float(gate["absolute_pd_delta_at_most"]),
        "false_positive_events_strictly_lower": int(
            candidate_counts["false_positive_events"]
        )
        < int(m20_counts["false_positive_events"]),
        "false_components_strictly_lower": int(
            candidate_counts["false_components"]
        )
        < int(m20_counts["false_components"]),
        "atomic_integrity": bool(atomic_integrity),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "score_gain": float(candidate_metrics["score"] - m20_metrics["score"]),
    }


def write_held_failure(error):
    if not HELD_OUTPUT.exists():
        return None
    path = HELD_OUTPUT / "held_failure.json"
    if path.exists():
        return path
    write_json_exclusive(
        path,
        {
            "schema": "ev-uav-activity-selective-recovery-held-failure-v1",
            "created_utc": utc_now(),
            "status": "failed_closed_held_consumed_no_retry",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "milestone": PROCESS_CONTEXT["milestone"],
            "source_arrays_opened": PROCESS_CONTEXT["source_arrays_opened"],
            "held_g2_array_read": PROCESS_CONTEXT["held_opened"],
            "validation_or_test_read": False,
            "retry_allowed": False,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "peak_cuda_mib": safe_peak_mib(),
            "elapsed_seconds": (
                None
                if PROCESS_CONTEXT["started"] is None
                else time.time() - PROCESS_CONTEXT["started"]
            ),
        },
    )
    return path


def run_evaluate_held_once(protocol, authorized):
    if not authorized:
        raise PermissionError("explicit one-shot held-G2 authorization is required")
    if HELD_OUTPUT.exists():
        raise FileExistsError("held G2 was already opened or attempted; retry forbidden")
    if python_gpu_processes():
        raise RuntimeError("another Python GPU process exists before held evaluation")
    receipt, strategy, stage1_path, stage2_path = verify_training_chain(protocol)
    held_names = list(protocol["source_groups"][GROUP_HELD])
    # Hash verification above completes before any held source file is touched.
    HELD_OUTPUT.mkdir(parents=True, exist_ok=False)
    PROCESS_CONTEXT.update(
        {
            "command": "evaluate-held-g2-once",
            "started": time.time(),
            "milestone": "held_output_created_after_hash_chain",
            "source_arrays_opened": [],
            "held_opened": False,
        }
    )
    held_open_receipt = HELD_OUTPUT / "held_open_receipt.json"
    write_json_exclusive(
        held_open_receipt,
        {
            "schema": "ev-uav-activity-selective-recovery-held-open-v1",
            "created_utc": utc_now(),
            "status": "held_consumed_no_retry_even_on_failure",
            "held_sources_about_to_open": held_names,
            "training_receipt_sha256": sha256_file(TRAIN_RECEIPT_PATH),
            "strategy_sha256": sha256_file(STRATEGY_PATH),
            "final_stage1_checkpoint_sha256": sha256_file(stage1_path),
            "final_stage2_checkpoint_sha256": sha256_file(stage2_path),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "retry_allowed": False,
        },
    )
    PROCESS_CONTEXT["held_opened"] = True
    PROCESS_CONTEXT["milestone"] = "held_open_receipt_persisted"
    device = torch.device("cuda:0")
    with gpu_run_lock("h2_activity_selective_recovery_held_g2_once"):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        cfg, c00 = build_c00(protocol)
        activity_model, activity_payload = _load_trained_expert_model(
            stage1_path, device
        )
        union_names = list(protocol["source_groups"][GROUP_G1]) + list(
            protocol["source_groups"][GROUP_G3]
        )
        verify_fresh_stage1_payload(
            protocol, activity_payload, "final_union_g1_g3", union_names
        )
        for parameter in activity_model.parameters():
            parameter.requires_grad_(False)
        recovery_model, negative_reference, _, _ = load_stage2_checkpoint(
            protocol, stage2_path, "final_union_true_oof", device
        )
        m20_model, _ = build_released_m20(device)
        for parameter in m20_model.parameters():
            parameter.requires_grad_(False)
        inherited_equal = all(
            name in activity_payload["model_state_dict"]
            and torch.equal(
                activity_payload["model_state_dict"][name].detach().cpu(),
                value.detach().cpu(),
            )
            for name, value in m20_model.state_dict().items()
        )
        if not inherited_equal:
            raise RuntimeError("final activity inherited M20 differs")
        state_before = {
            "activity": state_sha256(activity_model.state_dict()),
            "recovery": state_sha256(recovery_model.state_dict()),
            "m20": state_sha256(m20_model.state_dict()),
        }
        cutoff = float(strategy["selected_cutoff"])
        threshold = float(
            protocol["released_m20_and_postprocess"]["prediction_threshold"]
        )
        source_results = []
        pooled = {
            "m20": SufficientCounts(),
            "activity": SufficientCounts(),
            "candidate": SufficientCounts(),
        }
        all_atomic = True
        feature_dir = HELD_OUTPUT / "label_free_features"
        action_dir = HELD_OUTPUT / "label_free_actions"
        feature_dir.mkdir(parents=True, exist_ok=False)
        action_dir.mkdir(parents=True, exist_ok=False)
        batch_size = int(protocol["stage2"]["training"]["component_batch_size"])
        for source_name in held_names:
            record = extract_source_label_free(
                protocol,
                source_name,
                held_names,
                "final_union_g1_g3",
                sha256_file(stage1_path),
                activity_model,
                m20_model,
                cfg,
                feature_dir,
                device,
            )
            items, _, _, _ = recovery_items([record], include_targets=False)
            raw, embeddings = predict_recovery_items(
                recovery_model, items, batch_size, device
            )
            confidences = negative_reference_conformal_confidence(
                raw, negative_reference
            )
            candidate, atomic_receipt = atomic_recover_or_identity(
                record["m20_post"],
                record["activity_post"],
                record["disagreement"].event_indices,
                confidences,
                cutoff,
                threshold,
                enabled=True,
            )
            atomic_integrity = bool(
                atomic_receipt.complete_components_only
                and atomic_receipt.activity_outside_recovery_bitwise_equal
                and atomic_receipt.recovered_m20_scores_bitwise_equal
                and atomic_receipt.fallback_reason is None
            )
            all_atomic &= atomic_integrity
            action_artifact = persist_held_action_artifact(
                action_dir / "{}_actions.npz".format(source_name[:-4]),
                record,
                raw,
                confidences,
                embeddings,
                cutoff,
                candidate,
                atomic_receipt,
            )
            # Truth opens only after both label-free feature and action artifacts.
            labels, target_ids = _load_truth(record["source_path"])
            stream_payloads = {}
            for stream, scores in (
                ("m20", record["m20_post"]),
                ("activity", record["activity_post"]),
                ("candidate", candidate),
            ):
                counts = sufficient_counts_for_video(
                    scores,
                    labels,
                    target_ids,
                    record["locations"],
                    prediction_threshold=threshold,
                )
                pooled[stream] = pooled[stream] + counts
                stream_payloads[stream] = counts_payload(counts)
            source_results.append(
                {
                    "source_name": source_name,
                    "source_sha256": record["source_sha256"],
                    "m20": stream_payloads["m20"],
                    "activity": stream_payloads["activity"],
                    "candidate": stream_payloads["candidate"],
                    "atomic_recovery": asdict(atomic_receipt),
                    "atomic_integrity": atomic_integrity,
                    "label_free_feature_artifact": record[
                        "label_free_artifact"
                    ],
                    "label_free_action_artifact": action_artifact,
                }
            )
            PROCESS_CONTEXT["milestone"] = "held_source_{}_complete".format(
                source_name
            )
        state_after = {
            "activity": state_sha256(activity_model.state_dict()),
            "recovery": state_sha256(recovery_model.state_dict()),
            "m20": state_sha256(m20_model.state_dict()),
        }
        if state_before != state_after:
            raise RuntimeError("frozen held model state changed")
        pooled_payload = {
            stream: counts_payload(counts) for stream, counts in pooled.items()
        }
        promotion = held_promotion_gate(
            protocol, pooled_payload["m20"], pooled_payload["candidate"], all_atomic
        )
        peak_mib = safe_peak_mib()
        hard_peak = float(
            protocol["formal_step_and_resource_budget"]["hard_peak_cuda_mib"]
        )
        if peak_mib is None or peak_mib > hard_peak:
            raise RuntimeError("held evaluation exceeded or could not measure CUDA budget")
        result = {
            "schema": "ev-uav-activity-selective-recovery-held-g2-result-v1",
            "created_utc": utc_now(),
            "status": "completed_promoted" if promotion["passed"] else "completed_eliminated",
            "held_consumed": True,
            "retry_allowed": False,
            "source_arrays_opened": PROCESS_CONTEXT["source_arrays_opened"],
            "validation_or_test_read": False,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "training_receipt_sha256": sha256_file(TRAIN_RECEIPT_PATH),
            "strategy_sha256": sha256_file(STRATEGY_PATH),
            "frozen_cutoff": cutoff,
            "source_results": source_results,
            "pooled": pooled_payload,
            "promotion": promotion,
            "all_atomic_integrity": all_atomic,
            "frozen_state_sha256_before": state_before,
            "frozen_state_sha256_after": state_after,
            "inherited_m20_bitwise_equal": inherited_equal,
            "effective_c00_sha256": component_crossfit.sha256_json(c00),
            "peak_cuda_mib": peak_mib,
            "peak_cuda_budget_mib": hard_peak,
            "elapsed_seconds": time.time() - PROCESS_CONTEXT["started"],
        }
        result_path = HELD_OUTPUT / "held_g2_result.json"
        write_json_exclusive(result_path, result)
        activity_model.to("cpu")
        recovery_model.to("cpu")
        m20_model.to("cpu")
        del activity_model, recovery_model, m20_model
        gc.collect()
        torch.cuda.empty_cache()
        PROCESS_CONTEXT["milestone"] = "held_completed_no_retry"
        return {
            "status": result["status"],
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "promotion": promotion,
            "peak_cuda_mib": peak_mib,
        }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train-and-freeze")
    train_parser.add_argument(TRAIN_AUTH_FLAG, action="store_true")
    held_parser = subparsers.add_parser("evaluate-held-g2-once")
    held_parser.add_argument(HELD_AUTH_FLAG, action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    try:
        if args.command == "train-and-freeze":
            result = run_train_and_freeze(
                protocol, getattr(args, "root_authorized_formal_gpu")
            )
        else:
            result = run_evaluate_held_once(
                protocol, getattr(args, "root_authorized_held_g2_once")
            )
    except Exception as error:
        if args.command == "train-and-freeze":
            write_training_failure(error)
        else:
            write_held_failure(error)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

"""Authorized train095-only GPU probe for activity suppression + recovery."""

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
from crossfit_component_reranker import metrics_from_counts, sufficient_counts_for_video
from dataset.temporal_frame import build_temporal_context_frame
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
from run_h2_atomic_component_deletion_v3 import (
    C00_OVERRIDES,
    CONTEXT_BINS,
    EVC_ROOT,
    HEIGHT,
    INFERENCE_BATCH_SIZE,
    LOG_COUNT_CLIP,
    TEMPORAL_BIN_SIZE,
    WHOLE_T,
    WIDTH,
    _decode_frozen_features,
    _frame_tensor,
    _load_input_only,
    _load_truth,
    build_released_m20,
    gpu_run_lock,
    sha256_file,
    sha256_float32,
    state_sha256,
)
from utils.activity_selective_recovery import (
    atomic_recover_or_identity,
    disagreement_trajectory_context,
    extract_disagreement_components,
    marginal_recovery_targets,
    negative_reference_conformal_confidence,
)
from utils.atomic_component_deletion import (
    complete_input_polarity_minority_fraction,
    use_h2_atomic_deletion,
)
from utils.postprocess import ChallengePostprocessor
from utils.temporal_frame_loss import frame_balanced_event_bce


WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "h2_activity_suppress_selective_recovery_g2_science_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "ca61ec2777be57703c0c949d75e1457876ac419efc93df51a17efeb5a5229f23"
)
M20_PATH = EVC_ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
TRAIN_ROOT = WORKSPACE_ROOT / "datasets" / "EV-UAV-Challenge2" / "train"
SOURCE_NAME = "train_095.npz"
OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_h2_activity_suppress_selective_recovery_g2_v1"
    / "resource_probe"
)
AUTHORIZATION_FLAG = "--root-authorized-gpu"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_npz_exclusive(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def save_torch_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def setup_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_protocol():
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen activity-recovery protocol changed")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol["status"] != "frozen_cpu_design_before_any_gpu_probe_or_fresh_g2_open":
        raise RuntimeError("protocol status differs")
    if protocol["gpu_probe_budget"]["status"] != "not_authorized":
        raise RuntimeError("scientific protocol probe status was mutated")
    if protocol["gpu_probe_budget"]["source"] != SOURCE_NAME + " only":
        raise RuntimeError("probe source differs")
    if (
        int(protocol["gpu_probe_budget"]["stage1_optimizer_steps"]) != 8
        or int(protocol["gpu_probe_budget"]["stage2_optimizer_steps"]) != 8
    ):
        raise RuntimeError("probe step budget differs")
    if set(protocol["scope"]["fresh_outer_held_sources"]) != {
        "train_092.npz",
        "train_093.npz",
        "train_094.npz",
    }:
        raise RuntimeError("held G2 declaration differs")
    return protocol


def python_gpu_processes():
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if "python" in line.lower():
                rows.append(line.strip())
    return rows


def build_c00(protocol):
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml", list(C00_OVERRIDES)
    )
    threshold = float(protocol["baseline"]["prediction_threshold"])
    c00 = component_crossfit.validate_c00_config(cfg, threshold)
    if component_crossfit.sha256_json(c00) != protocol["baseline"][
        "effective_c00_sha256"
    ]:
        raise RuntimeError("effective C00 changed")
    return cfg, c00


def materialize_probe_view(source_path):
    view_root = OUTPUT_ROOT / "fit_view_train095"
    view_root.mkdir(parents=True, exist_ok=False)
    destination = view_root / SOURCE_NAME
    os.link(source_path, destination)
    if not destination.is_file() or sha256_file(destination) != sha256_file(source_path):
        raise RuntimeError("probe hard-link view changed source identity")
    return view_root, destination


def train_stage1(protocol, source_path, device):
    stage = protocol["stage1_activity_suppression"]["training"]
    setup_seed(stage["seed"])
    view_root, linked_source = materialize_probe_view(source_path)
    model, parent_checkpoint = build_expert_model_from_m20(
        M20_PATH, input_mode="activity_control", device=device
    )
    trainable_names = configure_expert_only_training(model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(trainable) != 14 or sum(p.numel() for p in trainable.values()) != 1712:
        raise RuntimeError("Stage1 trainable scope changed")
    initial_expert = {
        name: parameter.detach().cpu().clone() for name, parameter in trainable.items()
    }
    parent_state = parent_checkpoint["model_state_dict"]
    parent_hash_before = state_sha256(parent_state)
    optimizer = torch.optim.AdamW(
        list(trainable.values()),
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
        positive_frame_probability=0.75,
        random_seed=int(stage["seed"]),
        log_count_clip=LOG_COUNT_CLIP,
        cache_all_videos=False,
        cache_video_count=1,
        dense_sampling_enabled=False,
    )
    if [path.name for path in dataset.file_paths] != [SOURCE_NAME]:
        raise RuntimeError("Stage1 dataset escaped train095")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_memory_collate,
        pin_memory=True,
    )
    gradient_seen = {name: False for name in trainable}
    diagnostics = []
    step = 0
    for epoch in range(int(stage["epochs"])):
        dataset.set_epoch(epoch)
        model.eval()
        model.high_density_expert.train()
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).unsqueeze(0)
            time_indices = batch["event_time_indices"].to(device, non_blocking=True)
            event_x = batch["event_x"].to(device, non_blocking=True)
            event_y = batch["event_y"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            maps = model(frames).squeeze(0)
            logits = maps[time_indices, 0, event_y, event_x]
            loss, loss_diagnostics = frame_balanced_event_bce(
                logits,
                labels,
                time_indices,
                target_positive_loss_mass=float(stage["target_positive_loss_mass"]),
                max_positive_weight=float(stage["max_positive_weight"]),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Stage1 loss is non-finite")
            loss.backward()
            tensor_gradient_l1 = {}
            for name, parameter in trainable.items():
                gradient = parameter.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError("Stage1 gradient is missing or non-finite")
                value = float(gradient.detach().abs().sum().item())
                tensor_gradient_l1[name] = value
                gradient_seen[name] |= value > 0.0
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(trainable.values()), float(stage["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Stage1 gradient norm is non-finite")
            optimizer.step()
            step += 1
            diagnostics.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    "gradient_norm": float(gradient_norm.detach().item()),
                    "nonzero_gradient_tensor_count": int(
                        sum(value > 0.0 for value in tensor_gradient_l1.values())
                    ),
                    "loss_diagnostics": {
                        key: float(value) if isinstance(value, (float, int)) else value
                        for key, value in loss_diagnostics.items()
                    },
                }
            )
        scheduler.step()
    if step != 8 or not all(gradient_seen.values()):
        raise RuntimeError("Stage1 eight-step gradient gate failed")
    updated = {
        name: not torch.equal(parameter.detach().cpu(), initial_expert[name])
        for name, parameter in trainable.items()
    }
    if not all(updated.values()):
        raise RuntimeError("not every Stage1 tensor updated")
    trained_state = model.state_dict()
    parent_equal = all(
        name in trained_state
        and torch.equal(trained_state[name].detach().cpu(), value.detach().cpu())
        for name, value in parent_state.items()
    )
    if not parent_equal or state_sha256(parent_state) != parent_hash_before:
        raise RuntimeError("M20 parent changed during Stage1")
    checkpoint_path = OUTPUT_ROOT / "stage1_train095_step8.pt"
    save_torch_exclusive(
        checkpoint_path,
        {
            "schema": "ev-uav-h2-activity-selective-recovery-stage1-probe-v1",
            "model_state_dict": trained_state,
            "temporal_memory": parent_checkpoint["temporal_memory"],
            "input_mode": "activity_control",
            "optimizer_steps": step,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "source": SOURCE_NAME,
        },
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "dataset": dataset,
        "loader": loader,
        "linked_source": linked_source,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "diagnostics": diagnostics,
        "gradient_seen": gradient_seen,
        "updated": updated,
        "m20_parent_state_sha256": parent_hash_before,
        "m20_parent_bitwise_equal": parent_equal,
        "trainable_names": list(trainable_names),
    }


def dense_full_stream(model, video, device):
    temporal_count = len(video.event_indices_by_bin)
    if temporal_count != 160:
        raise RuntimeError("probe source is not full T160")
    bottlenecks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, temporal_count)
            frames = _frame_tensor(video, range(start, stop), device)
            bottlenecks.append(model.encode_bottleneck(frames))
        memory = model.temporal_residual(torch.cat(bottlenecks, dim=0))
    decoder = np.empty((temporal_count, 16, HEIGHT, WIDTH), dtype=np.float16)
    logits_cpu = np.empty((temporal_count, 1, HEIGHT, WIDTH), dtype=np.float16)
    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, temporal_count)
            frames = _frame_tensor(video, range(start, stop), device)
            decoded, logits = _decode_frozen_features(
                model, frames, memory[start:stop]
            )
            decoder[start:stop] = decoded.to(torch.float16).cpu().numpy()
            logits_cpu[start:stop] = logits.to(torch.float16).cpu().numpy()
            probabilities = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
            for temporal_bin in range(start, stop):
                indices = video.event_indices_by_bin[temporal_bin]
                if indices.size == 0:
                    continue
                local = temporal_bin - start
                locations = video.locations[indices]
                scores[indices] = probabilities[
                    local, locations[:, 1], locations[:, 0]
                ]
    if not (
        np.isfinite(scores).all()
        and np.isfinite(decoder).all()
        and np.isfinite(logits_cpu).all()
    ):
        raise RuntimeError("full-stream feature output is non-finite")
    del memory, bottlenecks
    return scores, decoder, logits_cpu


def apply_c00(scores, locations, cfg, threshold):
    processed, stats = ChallengePostprocessor.from_cfg(
        cfg, threshold, event_count=int(len(scores))
    ).apply(
        torch.from_numpy(np.asarray(scores, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(locations, dtype=np.int64)).long(),
    )
    return processed.numpy().astype(np.float32, copy=True), asdict(stats)


def crop_with_padding(array, center_x, center_y, radius):
    size = 2 * int(radius) + 1
    padded = np.pad(
        np.asarray(array),
        ((0, 0), (radius, radius), (radius, radius)),
        mode="constant",
    )
    output = padded[
        :, int(center_y) : int(center_y) + size, int(center_x) : int(center_x) + size
    ]
    if output.shape[-2:] != (size, size):
        raise RuntimeError("dense patch crop escaped padding")
    return output


def build_recovery_features(
    video,
    locations,
    m20_post,
    activity_post,
    components,
    m20_decoder,
    activity_decoder,
    m20_logits,
    activity_logits,
    threshold,
):
    radius = 7
    queries_by_component = []
    trajectory_by_component = []
    for indices in components:
        queries, trajectory = disagreement_trajectory_context(
            indices,
            locations,
            m20_post,
            activity_post,
            threshold,
            patch_radius=radius,
            temporal_bin_size=TEMPORAL_BIN_SIZE,
            stream_bin_count=160,
            width=WIDTH,
            height=HEIGHT,
        )
        queries_by_component.append(queries)
        trajectory_by_component.append(trajectory)
    activity_masks = {}
    patches = []
    for queries in queries_by_component:
        sequence = []
        for query in queries:
            temporal_bin = int(query.temporal_bin)
            raw = build_temporal_context_frame(
                video,
                temporal_bin,
                CONTEXT_BINS,
                WIDTH,
                HEIGHT,
                LOG_COUNT_CLIP,
            ).astype(np.float32, copy=False)
            negative = raw[0::2]
            positive = raw[1::2]
            activity_maps = 0.5 * (negative + positive)
            signed = positive - negative
            semantic = np.concatenate(
                (
                    m20_decoder[temporal_bin].astype(np.float32),
                    activity_decoder[temporal_bin].astype(np.float32),
                    m20_logits[temporal_bin].astype(np.float32),
                    activity_logits[temporal_bin].astype(np.float32),
                    activity_logits[temporal_bin].astype(np.float32)
                    - m20_logits[temporal_bin].astype(np.float32),
                    raw,
                    activity_maps,
                    signed,
                ),
                axis=0,
            )
            if semantic.shape[0] != 55:
                raise RuntimeError("recovery dense feature channel contract changed")
            dense_patch = crop_with_padding(
                semantic, query.center_x, query.center_y, radius
            )
            if temporal_bin not in activity_masks:
                mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                event_indices = video.event_indices_by_bin[temporal_bin]
                selected = event_indices[
                    activity_post[event_indices] >= np.float32(threshold)
                ]
                if selected.size:
                    local = video.locations[selected]
                    mask[local[:, 1], local[:, 0]] = 1.0
                activity_masks[temporal_bin] = mask
            activity_mask_patch = crop_with_padding(
                activity_masks[temporal_bin][None],
                query.center_x,
                query.center_y,
                radius,
            )
            patch = np.concatenate(
                (
                    dense_patch,
                    query.component_mask[None].astype(np.float32),
                    activity_mask_patch,
                ),
                axis=0,
            ).astype(np.float16)
            if patch.shape != (RECOVERY_PATCH_CHANNELS, 15, 15):
                raise RuntimeError("recovery patch shape changed")
            sequence.append(patch)
        patches.append(np.stack(sequence, axis=0))
    return tuple(patches), tuple(trajectory_by_component)


def component_offsets(components):
    lengths = np.asarray([len(indices) for indices in components], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flattened = (
        np.concatenate(components).astype(np.int64, copy=False)
        if components
        else np.empty(0, dtype=np.int64)
    )
    return offsets, flattened


def sequence_offsets(sequences):
    lengths = np.asarray([len(values) for values in sequences], dtype=np.int64)
    return np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))


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
    component_offsets_value, component_events = component_offsets(
        disagreement.event_indices
    )
    patch_offsets = sequence_offsets(patches)
    flat_patches = np.concatenate(patches, axis=0)
    flat_trajectory = np.concatenate(trajectories, axis=0).astype(np.float32)
    write_npz_exclusive(
        path,
        artifact_schema=np.asarray(
            "ev-uav-h2-activity-selective-recovery-probe-input-v1"
        ),
        source_name=np.asarray(SOURCE_NAME),
        m20_raw_scores=np.asarray(m20_raw, dtype=np.float32),
        m20_c00_scores=np.asarray(m20_post, dtype=np.float32),
        activity_raw_scores=np.asarray(activity_raw, dtype=np.float32),
        activity_c00_scores=np.asarray(activity_post, dtype=np.float32),
        locations=np.asarray(locations, dtype=np.int16),
        disagreement_m20_component_ids=disagreement.m20_component_ids,
        disagreement_component_offsets=component_offsets_value,
        disagreement_component_event_indices=component_events,
        patch_offsets=patch_offsets,
        recovery_patches=flat_patches,
        trajectory_context=flat_trajectory,
        contains_labels_or_target_ids=np.asarray(False),
    )
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def train_stage2(protocol, patches, trajectories, targets, device):
    stage = protocol["stage2_training"]
    setup_seed(stage["seed"])
    targets = np.asarray(targets, dtype=np.uint8)
    if not np.any(targets == 0) or not np.any(targets == 1):
        raise RuntimeError("real train095 disagreement lacks both marginal classes")
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
    if recovery_parameter_count(model) != 7910:
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
    diagnostics = []
    batch_size = int(stage["component_batch_size"])
    for step in range(8):
        start = (step * batch_size) % len(items)
        indices = [(start + offset) % len(items) for offset in range(min(batch_size, len(items)))]
        batch = recovery_sequence_collate([items[index] for index in indices])
        patch_tensor = batch["patches"].to(device)
        trajectory_tensor = batch["trajectory"].to(device)
        lengths = batch["lengths"].to(device)
        batch_targets = batch["targets"].to(device)
        batch_weights = batch["weights"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(patch_tensor, trajectory_tensor, lengths)
        losses = functional.binary_cross_entropy_with_logits(
            logits, batch_targets, reduction="none"
        )
        loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Stage2 loss is non-finite")
        loss.backward()
        branch_gradient_l1 = {
            prefix: 0.0
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
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Stage2 gradient is missing or non-finite")
            value = float(gradient.detach().abs().sum().item())
            gradient_seen[name] |= value > 0.0
            for prefix in branch_gradient_l1:
                if name.startswith(prefix):
                    branch_gradient_l1[prefix] += value
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(stage["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Stage2 gradient norm is non-finite")
        optimizer.step()
        diagnostics.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().item()),
                "gradient_norm": float(gradient_norm.detach().item()),
                "branch_gradient_l1": branch_gradient_l1,
                "batch_components": len(indices),
                "batch_positive_fraction": float(batch_targets.mean().item()),
            }
        )
    if not all(gradient_seen.values()):
        missing = [name for name, value in gradient_seen.items() if not value]
        raise RuntimeError("Stage2 tensors never received gradient: {}".format(missing))
    updated = {
        name: not torch.equal(parameter.detach().cpu(), initial[name])
        for name, parameter in model.named_parameters()
    }
    if not all(updated.values()):
        missing = [name for name, value in updated.items() if not value]
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
    reference = raw_probabilities[targets == 0]
    conformal = negative_reference_conformal_confidence(raw_probabilities, reference)
    checkpoint_path = OUTPUT_ROOT / "stage2_train095_step8.pt"
    save_torch_exclusive(
        checkpoint_path,
        {
            "schema": "ev-uav-h2-activity-selective-recovery-stage2-probe-v1",
            "model_state_dict": model.state_dict(),
            "optimizer_steps": 8,
            "fit_negative_reference_probabilities": reference,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "source": SOURCE_NAME,
        },
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "raw_probabilities": raw_probabilities,
        "conformal_confidences": conformal,
        "embeddings": embeddings,
        "negative_reference": reference,
        "diagnostics": diagnostics,
        "gradient_seen": gradient_seen,
        "updated": updated,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def run_probe(authorized):
    if not authorized:
        raise PermissionError("explicit root GPU authorization flag is required")
    if OUTPUT_ROOT.exists():
        raise FileExistsError("refusing to overwrite activity-recovery probe")
    protocol = load_protocol()
    if python_gpu_processes():
        raise RuntimeError("another Python GPU process exists before probe")
    source_path = TRAIN_ROOT / SOURCE_NAME
    metadata = protocol["sources"][SOURCE_NAME]
    if not source_path.is_file() or sha256_file(source_path) != metadata["sha256"]:
        raise RuntimeError("train095 identity changed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device("cuda:0")
    result = None
    with gpu_run_lock("h2_activity_selective_recovery_probe"):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        cfg, c00 = build_c00(protocol)
        video, polarities, locations = _load_input_only(source_path)
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
            raise RuntimeError("train095 no longer satisfies exact H2 route")
        threshold = float(protocol["baseline"]["prediction_threshold"])

        stage1 = train_stage1(protocol, source_path, device)
        activity_raw, activity_decoder, activity_logits = dense_full_stream(
            stage1["model"], video, device
        )
        activity_post, activity_c00_stats = apply_c00(
            activity_raw, locations, cfg, threshold
        )
        # Drop all Stage1 training objects before loading the separate M20 pass.
        del stage1["optimizer"], stage1["scheduler"], stage1["loader"], stage1["dataset"]
        stage1["model"].to("cpu")
        del stage1["model"]
        gc.collect()
        torch.cuda.empty_cache()

        m20_model, m20_payload = build_released_m20(device)
        m20_hash_before = state_sha256(m20_model.state_dict())
        m20_raw, m20_decoder, m20_logits = dense_full_stream(m20_model, video, device)
        m20_hash_after = state_sha256(m20_model.state_dict())
        if m20_hash_before != m20_hash_after:
            raise RuntimeError("M20 changed during feature inference")
        m20_post, m20_c00_stats = apply_c00(m20_raw, locations, cfg, threshold)
        m20_model.to("cpu")
        del m20_model, m20_payload
        gc.collect()
        torch.cuda.empty_cache()

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
            raise RuntimeError("train095 produced no real disagreement component")
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
            OUTPUT_ROOT / "immutable_probe_input.npz",
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
        label_artifact_path = OUTPUT_ROOT / "immutable_fit_only_probe_labels.npz"
        write_npz_exclusive(
            label_artifact_path,
            artifact_schema=np.asarray(
                "ev-uav-h2-activity-selective-recovery-probe-fit-labels-v1"
            ),
            marginal_recovery_targets=recovery_targets,
            marginal_official_score_deltas=marginal_score_deltas,
            contains_fit_only_labels=np.asarray(True),
            source_name=np.asarray(SOURCE_NAME),
        )
        label_artifact = {
            "path": str(label_artifact_path.resolve()),
            "sha256": sha256_file(label_artifact_path),
        }
        stage2 = train_stage2(
            protocol, patches, trajectories, recovery_targets, device
        )
        # Probe-only mechanical action: the exact maximum observed confidence.
        # This is neither saved as a scientific cutoff nor used for formal work.
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
            raise RuntimeError("probe atomic recovery gate failed")
        score_artifact_path = OUTPUT_ROOT / "immutable_probe_recovery_scores.npz"
        write_npz_exclusive(
            score_artifact_path,
            artifact_schema=np.asarray(
                "ev-uav-h2-activity-selective-recovery-probe-scores-v1"
            ),
            raw_recovery_probabilities=stage2["raw_probabilities"],
            conformal_confidences=stage2["conformal_confidences"],
            negative_reference_probabilities=stage2["negative_reference"],
            recovery_embeddings=stage2["embeddings"],
            mechanical_integrity_cutoff=np.asarray(mechanical_cutoff),
            recovered_component=np.asarray(
                stage2["conformal_confidences"] >= mechanical_cutoff,
                dtype=np.bool_,
            ),
            final_candidate_scores=candidate.astype(np.float32),
            cutoff_is_scientific_or_formal=np.asarray(False),
        )
        score_artifact = {
            "path": str(score_artifact_path.resolve()),
            "sha256": sha256_file(score_artifact_path),
        }
        torch.cuda.synchronize()
        peak_mib = float(torch.cuda.max_memory_reserved(0) / (1024 ** 2))
        if peak_mib > float(protocol["gpu_probe_budget"]["estimated_peak_cuda_mib"][1]):
            raise RuntimeError("probe exceeded frozen 3.6 GiB CUDA budget")
        activity_counts = sufficient_counts_for_video(
            activity_post,
            labels,
            target_ids,
            locations,
            prediction_threshold=threshold,
        )
        m20_counts = sufficient_counts_for_video(
            m20_post,
            labels,
            target_ids,
            locations,
            prediction_threshold=threshold,
        )
        candidate_counts = sufficient_counts_for_video(
            candidate,
            labels,
            target_ids,
            locations,
            prediction_threshold=threshold,
        )
        result = {
            "schema": "ev-uav-h2-activity-selective-recovery-probe-result-v1",
            "created_utc": utc_now(),
            "status": "completed",
            "source_arrays_opened": [SOURCE_NAME],
            "held_g2_array_read": False,
            "g1_array_read": False,
            "validation_or_test_read": False,
            "formal_started": False,
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_sha256": sha256_file(source_path),
            "effective_c00_sha256": component_crossfit.sha256_json(c00),
            "stage1": {
                "trainable_parameter_count": 1712,
                "optimizer_steps": 8,
                "all_14_tensors_finite_nonzero_gradient": all(
                    stage1["gradient_seen"].values()
                ),
                "all_14_tensors_updated": all(stage1["updated"].values()),
                "m20_parent_bitwise_equal": stage1["m20_parent_bitwise_equal"],
                "m20_parent_state_sha256": stage1["m20_parent_state_sha256"],
                "checkpoint": str(stage1["checkpoint_path"].resolve()),
                "checkpoint_sha256": stage1["checkpoint_sha256"],
                "diagnostics": stage1["diagnostics"],
            },
            "full_stream": {
                "temporal_bins": 160,
                "m20_raw_scores_sha256": sha256_float32(m20_raw),
                "m20_c00_scores_sha256": sha256_float32(m20_post),
                "activity_raw_scores_sha256": sha256_float32(activity_raw),
                "activity_c00_scores_sha256": sha256_float32(activity_post),
                "m20_c00_stats": m20_c00_stats,
                "activity_c00_stats": activity_c00_stats,
                "m20_state_sha256_before": m20_hash_before,
                "m20_state_sha256_after": m20_hash_after,
            },
            "disagreement": {
                "m20_component_count": disagreement.m20_component_count,
                "activity_component_count": disagreement.activity_component_count,
                "component_count": len(disagreement.event_indices),
                "component_bin_patch_count": int(sum(len(value) for value in patches)),
                "marginal_positive_count": int(
                    np.count_nonzero(recovery_targets == 1)
                ),
                "marginal_negative_count": int(
                    np.count_nonzero(recovery_targets == 0)
                ),
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
                "diagnostics": stage2["diagnostics"],
            },
            "atomic_recovery": asdict(recovery_receipt),
            "mechanical_cutoff_not_formal": mechanical_cutoff,
            "metrics_are_probe_diagnostics_not_selection": {
                "m20": {
                    "counts": m20_counts.to_dict(),
                    "metrics": metrics_from_counts(m20_counts),
                },
                "activity": {
                    "counts": activity_counts.to_dict(),
                    "metrics": metrics_from_counts(activity_counts),
                },
                "mechanical_recovery": {
                    "counts": candidate_counts.to_dict(),
                    "metrics": metrics_from_counts(candidate_counts),
                },
            },
            "immutable_artifacts": {
                "input": input_artifact,
                "fit_labels": label_artifact,
                "recovery_scores": score_artifact,
            },
            "peak_cuda_mib": peak_mib,
            "peak_cuda_budget_mib": 3600.0,
            "elapsed_seconds": time.time() - started,
        }
        result_path = OUTPUT_ROOT / "probe_result.json"
        write_json_exclusive(result_path, result)
        result["result_path"] = str(result_path.resolve())
        result["result_sha256"] = sha256_file(result_path)
        del stage2["optimizer"], stage2["model"]
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(AUTHORIZATION_FLAG, action="store_true")
    args = parser.parse_args()
    try:
        result = run_probe(getattr(args, "root_authorized_gpu"))
    except Exception as error:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        failure_path = OUTPUT_ROOT / "probe_failure.json"
        if not failure_path.exists():
            write_json_exclusive(
                failure_path,
                {
                    "schema": "ev-uav-h2-activity-selective-recovery-probe-failure-v1",
                    "created_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "source_arrays_allowed": [SOURCE_NAME],
                    "held_g2_array_read": False,
                    "g1_array_read": False,
                    "validation_or_test_read": False,
                    "formal_started": False,
                    "protocol_sha256": (
                        sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None
                    ),
                    "runner_sha256": sha256_file(Path(__file__).resolve()),
                },
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

"""Authorized unique eight-step GPU probe for the frozen H2 pyramid protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import crossfit_component_reranker as crossfit
import replay_temporal_memory_validation as replay
import run_h2_atomic_component_deletion_v3 as atomic
from model.h2_multiscale_temporal_pyramid_expert import (
    FrozenM20MultiScalePyramidAdapter,
    downsample_frozen_observations,
    fixed_multiscale_temporal_moments,
    pyramid_expert_parameter_count,
)
from utils.atomic_component_deletion import (
    extract_atomic_components,
    pure_false_positive_targets,
)
from utils.h2_multiscale_pyramid_loss import (
    PyramidDualState,
    multiscale_pyramid_constrained_loss,
    validate_pyramid_step_diagnostics,
)
from utils.postprocess import ChallengePostprocessor
from utils.target_preserving_residual import use_h2_residual_refiner


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROTOCOL_PATH = ROOT / "protocols" / "h2_multiscale_temporal_pyramid_expert_science_v1.json"
OUTPUT_PATH = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_multiscale_temporal_pyramid_expert_v1"
    / "resource_probe"
    / "eight_step_probe.json"
)
SOURCE_NAME = "train_092.npz"
SOURCE_PATH = WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train" / SOURCE_NAME
SOURCE_SHA256 = "9b8707627ac1bbcd471bbff7fc9b02fafbe5c6dd0f54857dbb158fb8916eeca6"
SOURCE_EVENT_COUNT = 595519
EXPECTED_PROTOCOL_SHA256 = "0bdb6e0657483e253b363462ffad6969dcd85df52ef5707d32ed93a914268155"
GPU_FLAG = "--root-authorized-gpu"
TEMPORAL_COUNT = 160
VIEW_BINS = 16
INFERENCE_BATCH = 8
PROBE_STEPS = 8
SEED = 67
PREDICTION_THRESHOLD = 0.719


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_file(path):
    return sha256_file(path)


def write_bytes_exclusive(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(values)
        stream.flush()
        os.fsync(stream.fileno())


def load_protocol():
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen pyramid protocol SHA-256 changed")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol["eight_step_probe"]["GPU_authorized"] is not False:
        raise RuntimeError("protocol must remain frozen as unauthorized; CLI carries root authorization")
    if protocol["eight_step_probe"]["source"] != SOURCE_NAME:
        raise RuntimeError("probe source changed")
    if int(protocol["eight_step_probe"]["optimizer_steps"]) != PROBE_STEPS:
        raise RuntimeError("probe step count changed")
    if protocol["science_scope"]["validation_read_allowed"] is not False:
        raise RuntimeError("validation access is forbidden")
    if protocol["science_scope"]["test_read_allowed"] is not False:
        raise RuntimeError("test access is forbidden")
    first_fold = protocol["fold_order"][0]
    if SOURCE_NAME not in protocol["source_groups"]["g2_092_094"]:
        raise RuntimeError("probe source is outside frozen H2 groups")
    if "g2_092_094" not in first_fold["fit_groups"] or first_fold["held_group"] != "g3_095_098":
        raise RuntimeError("probe source is not a first-fold fit member")
    return protocol


def build_c00():
    cfg = replay.load_flat_config(
        ROOT / "configs" / "evisseg_evuav.yaml", atomic.C00_OVERRIDES
    )
    effective = crossfit.validate_c00_config(cfg, PREDICTION_THRESHOLD)
    return cfg, effective


def full_stream_memory(model, video, device):
    temporal_count = len(video.event_indices_by_bin)
    if temporal_count != TEMPORAL_COUNT:
        raise RuntimeError("probe requires the complete T160 stream")
    bottlenecks = []
    with torch.no_grad():
        for start in range(0, temporal_count, INFERENCE_BATCH):
            stop = min(start + INFERENCE_BATCH, temporal_count)
            frames = atomic._frame_tensor(video, range(start, stop), device)
            bottlenecks.append(model.encode_bottleneck(frames))
            del frames
        joined = torch.cat(bottlenecks, dim=0)
        memory = model.temporal_residual(joined)
    del bottlenecks, joined
    return memory


def stream_observations_and_scores(adapter, video, memory, device):
    observations = []
    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    decoded_bin_count = 0
    with torch.no_grad():
        for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
            stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
            frames = atomic._frame_tensor(video, range(start, stop), device)
            decoder, logits, centre = adapter.decode_frozen_features(
                frames, memory[start:stop]
            )
            pooled = downsample_frozen_observations(
                decoder.unsqueeze(0), logits.unsqueeze(0), centre.unsqueeze(0)
            ).squeeze(0)
            observations.append(pooled.to(device="cpu", dtype=torch.float16))
            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            for temporal_bin in range(start, stop):
                indices = video.event_indices_by_bin[temporal_bin]
                if indices.size == 0:
                    continue
                local = temporal_bin - start
                xy = video.locations[indices]
                scores[indices] = probabilities[local, xy[:, 1], xy[:, 0]]
            decoded_bin_count += stop - start
            del frames, decoder, logits, centre, pooled, probabilities
    if decoded_bin_count != TEMPORAL_COUNT or not np.isfinite(scores).all():
        raise RuntimeError("first streaming decoder pass is incomplete")
    return torch.cat(observations, dim=0).contiguous(), scores, decoded_bin_count


def build_summary_cache(observations_cpu, device):
    with torch.no_grad():
        observations_gpu = observations_cpu.unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        summaries_gpu = fixed_multiscale_temporal_moments(observations_gpu)
        summaries_cpu = tuple(
            value.squeeze(0).to(device="cpu", dtype=torch.float16).contiguous()
            for value in summaries_gpu
        )
    del observations_gpu, summaries_gpu
    return summaries_cpu


def sample_dense_event_logits(dense_logits, video, start, stop):
    values = []
    global_indices = []
    for temporal_bin in range(start, stop):
        indices = video.event_indices_by_bin[temporal_bin]
        if indices.size == 0:
            continue
        xy = video.locations[indices]
        values.append(dense_logits[temporal_bin - start, 0, xy[:, 1], xy[:, 0]])
        global_indices.append(indices)
    if not values:
        raise RuntimeError("probe view has no events")
    return torch.cat(values), np.concatenate(global_indices).astype(np.int64, copy=False)


def prepare_view_metadata(video, labels, target_ids, pure_fp_components):
    positive_bins = np.floor_divide(
        video.locations[np.flatnonzero(labels > 0), 2].astype(np.int64), 50
    )
    component_bins = []
    for component in pure_fp_components:
        bins = np.unique(
            np.floor_divide(video.locations[component, 2].astype(np.int64), 50)
        )
        if bins.size != 1:
            raise RuntimeError("probe hard-negative components must be per-bin")
        component_bins.append(int(bins[0]))
    component_bins = np.asarray(component_bins, dtype=np.int64)
    eligible = []
    components_by_start = {}
    for start in range(0, TEMPORAL_COUNT - VIEW_BINS + 1):
        stop = start + VIEW_BINS
        if not np.any((positive_bins >= start) & (positive_bins < stop)):
            continue
        rows = np.flatnonzero((component_bins >= start) & (component_bins < stop))
        if rows.size == 0:
            continue
        eligible.append(start)
        components_by_start[start] = tuple(pure_fp_components[int(row)] for row in rows)
    if not eligible:
        raise RuntimeError("train_092 has no joint target/hard-negative probe view")
    generator = np.random.default_rng(SEED)
    order = generator.permutation(len(eligible))
    selected = [eligible[int(order[index % len(order)])] for index in range(PROBE_STEPS)]
    metadata = []
    for start in selected:
        stop = start + VIEW_BINS
        global_indices = np.concatenate(
            [video.event_indices_by_bin[temporal_bin] for temporal_bin in range(start, stop)]
        ).astype(np.int64, copy=False)
        global_to_local = np.full(labels.size, -1, dtype=np.int64)
        global_to_local[global_indices] = np.arange(global_indices.size, dtype=np.int64)
        local_components = []
        for component in components_by_start[start]:
            local = global_to_local[component]
            if np.any(local < 0):
                raise RuntimeError("hard-negative component escaped its selected view")
            local_components.append(local)
        selected_labels = labels[global_indices]
        selected_target_ids = target_ids[global_indices]
        selected_times = np.floor_divide(
            video.locations[global_indices, 2].astype(np.int64), 50
        )
        if not np.any(selected_labels > 0) or not local_components:
            raise RuntimeError("probe view lacks a required loss class")
        metadata.append(
            {
                "start": int(start),
                "stop": int(stop),
                "global_indices": global_indices,
                "labels": selected_labels,
                "target_ids": selected_target_ids,
                "times": selected_times,
                "hard_negative_components": tuple(local_components),
            }
        )
    return metadata, len(eligible)


def parameter_snapshot(module):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def parameter_update_l1(module, before):
    return {
        name: float((parameter.detach().cpu() - before[name]).abs().sum())
        for name, parameter in module.named_parameters()
    }


def run_probe(args):
    if not getattr(args, "root_authorized_gpu"):
        raise PermissionError("unique pyramid probe requires explicit root GPU authorization")
    protocol = load_protocol()
    if OUTPUT_PATH.exists() or OUTPUT_PATH.parent.exists():
        raise FileExistsError("refusing to overwrite immutable pyramid probe receipt")
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("train_092 source SHA-256 changed")

    started = time.perf_counter()
    payload = None
    with atomic.gpu_run_lock("h2_multiscale_temporal_pyramid_unique_probe"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        m20, checkpoint = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        if pyramid_expert_parameter_count(adapter) != 3381:
            raise RuntimeError("pyramid parameter count changed")
        if any(parameter.requires_grad for parameter in m20.parameters()):
            raise RuntimeError("released M20 is not frozen")

        video, polarities, locations4 = atomic._load_input_only(SOURCE_PATH)
        labels, target_ids = atomic._load_truth(SOURCE_PATH)
        if len(polarities) != SOURCE_EVENT_COUNT or not use_h2_residual_refiner(
            SOURCE_EVENT_COUNT, polarities
        ):
            raise RuntimeError("train_092 no longer satisfies frozen H2 route")
        if len(video.event_indices_by_bin) != TEMPORAL_COUNT:
            raise RuntimeError("train_092 no longer has T160")

        memory = full_stream_memory(m20, video, device)
        observations_cpu, raw_scores, observation_decoder_bins = (
            stream_observations_and_scores(adapter, video, memory, device)
        )
        summary_cache = build_summary_cache(observations_cpu, device)
        del observations_cpu
        torch.cuda.empty_cache()

        cfg, effective_c00 = build_c00()
        processed, postprocess_stats = ChallengePostprocessor.from_cfg(
            cfg, PREDICTION_THRESHOLD, event_count=SOURCE_EVENT_COUNT
        ).apply(torch.from_numpy(raw_scores.copy()), torch.from_numpy(locations4).long())
        base_scores = processed.numpy().astype(np.float32, copy=True)
        components = extract_atomic_components(
            base_scores,
            locations4,
            PREDICTION_THRESHOLD,
            spatial_radius=2,
            temporal_bin_size=50,
            temporal_radius_bins=0,
        )
        component_targets = pure_false_positive_targets(components.event_indices, labels)
        pure_fp_components = tuple(
            components.event_indices[index]
            for index in np.flatnonzero(component_targets == 1)
        )
        view_metadata, eligible_view_count = prepare_view_metadata(
            video, labels, target_ids, pure_fp_components
        )

        adapter.train()
        optimizer = torch.optim.AdamW(
            adapter.trainable_parameters(),
            lr=float(protocol["training"]["learning_rate"]),
            weight_decay=float(protocol["training"]["weight_decay"]),
        )
        dual_state = PyramidDualState()
        expert_before = parameter_snapshot(adapter.expert)
        cumulative_gradient_l1 = {
            name: 0.0 for name, _ in adapter.expert.named_parameters()
        }
        records = []
        initial_identity = None
        second_decoder_bins = 0

        # Authorized numerical recovery gate: run the exact first fit-only
        # batch in FP32, check every loss/gradient for finiteness, and discard
        # gradients without updating any parameter or dual multiplier.
        audit_metadata = view_metadata[0]
        audit_start = audit_metadata["start"]
        audit_stop = audit_metadata["stop"]
        audit_frames = atomic._frame_tensor(
            video, range(audit_start, audit_stop), device
        )
        audit_decoder, audit_base_logits, audit_centre = adapter.decode_frozen_features(
            audit_frames, memory[audit_start:audit_stop]
        )
        audit_summaries = tuple(
            value[audit_start:audit_stop].to(device=device, dtype=torch.float32)
            for value in summary_cache
        )
        optimizer.zero_grad(set_to_none=True)
        audit_parts = adapter.expert(
            audit_decoder.unsqueeze(0),
            audit_base_logits.unsqueeze(0),
            audit_centre.unsqueeze(0),
            tuple(value.unsqueeze(0) for value in audit_summaries),
            return_parts=True,
        )
        audit_refined, audit_global = sample_dense_event_logits(
            audit_parts.refined_logits.squeeze(0), video, audit_start, audit_stop
        )
        audit_base, audit_base_global = sample_dense_event_logits(
            audit_base_logits, video, audit_start, audit_stop
        )
        if not np.array_equal(audit_global, audit_metadata["global_indices"]) or not np.array_equal(
            audit_base_global, audit_global
        ):
            raise RuntimeError("FP32 gradient audit event sampling order changed")
        audit_labels = torch.from_numpy(audit_metadata["labels"]).to(
            device=device, dtype=torch.float32
        )
        audit_target_ids = torch.from_numpy(audit_metadata["target_ids"]).to(
            device=device, dtype=torch.long
        )
        audit_times = torch.from_numpy(audit_metadata["times"]).to(
            device=device, dtype=torch.long
        )
        audit_loss, audit_recall, audit_suppression, audit_diagnostics = (
            multiscale_pyramid_constrained_loss(
                audit_refined.float(),
                audit_base.float(),
                audit_labels,
                audit_target_ids,
                audit_times,
                audit_metadata["hard_negative_components"],
                dual_state,
            )
        )
        if not torch.isfinite(audit_loss):
            raise RuntimeError("FP32 no-update gradient audit loss is non-finite")
        audit_loss.backward()
        audit_gradient_l1 = {}
        for name, parameter in adapter.expert.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    "FP32 no-update gradient audit failed for {}".format(name)
                )
            audit_gradient_l1[name] = float(parameter.grad.detach().abs().sum())
        audit_global_gradient_l1 = sum(audit_gradient_l1.values())
        if audit_global_gradient_l1 <= 0.0 or audit_gradient_l1["output_projection.weight"] <= 0.0:
            raise RuntimeError("FP32 no-update gradient audit has no reachable output gradient")
        fp32_gradient_audit = {
            "view_start_bin": audit_start,
            "view_stop_bin_exclusive": audit_stop,
            **audit_diagnostics,
            "loss_terms_finite": True,
            "all_parameter_tensor_gradients_finite": True,
            "global_gradient_l1": audit_global_gradient_l1,
            "output_projection_gradient_l1": audit_gradient_l1[
                "output_projection.weight"
            ],
            "structural_zero_init_upstream_zero_gradient_expected": True,
            "zero_gradient_parameter_tensors": sorted(
                name for name, value in audit_gradient_l1.items() if value == 0.0
            ),
            "optimizer_step_executed": False,
            "dual_update_executed": False,
        }
        optimizer.zero_grad(set_to_none=True)
        del (
            audit_frames,
            audit_decoder,
            audit_base_logits,
            audit_centre,
            audit_summaries,
            audit_parts,
            audit_refined,
            audit_base,
            audit_labels,
            audit_target_ids,
            audit_times,
            audit_loss,
        )

        for step, metadata in enumerate(view_metadata, start=1):
            start = metadata["start"]
            stop = metadata["stop"]
            frames = atomic._frame_tensor(video, range(start, stop), device)
            decoder, base_logits, centre = adapter.decode_frozen_features(
                frames, memory[start:stop]
            )
            summaries = tuple(
                value[start:stop].to(device=device, dtype=torch.float32)
                for value in summary_cache
            )
            optimizer.zero_grad(set_to_none=True)
            parts = adapter.expert(
                decoder.unsqueeze(0),
                base_logits.unsqueeze(0),
                centre.unsqueeze(0),
                tuple(value.unsqueeze(0) for value in summaries),
                return_parts=True,
            )
            refined_events, sampled_global = sample_dense_event_logits(
                parts.refined_logits.squeeze(0), video, start, stop
            )
            base_events, base_global = sample_dense_event_logits(
                base_logits, video, start, stop
            )
            if not np.array_equal(sampled_global, metadata["global_indices"]) or not np.array_equal(
                base_global, sampled_global
            ):
                raise RuntimeError("probe event sampling order changed")
            label_tensor = torch.from_numpy(metadata["labels"]).to(
                device=device, dtype=torch.float32
            )
            target_tensor = torch.from_numpy(metadata["target_ids"]).to(
                device=device, dtype=torch.long
            )
            time_tensor = torch.from_numpy(metadata["times"]).to(
                device=device, dtype=torch.long
            )
            loss, recall, suppression, diagnostics = (
                multiscale_pyramid_constrained_loss(
                    refined_events.float(),
                    base_events.float(),
                    label_tensor,
                    target_tensor,
                    time_tensor,
                    metadata["hard_negative_components"],
                    dual_state,
                )
            )
            if step == 1:
                initial_identity = bool(
                    torch.equal(parts.refined_logits.detach(), base_logits.unsqueeze(0))
                    and torch.count_nonzero(parts.correction.detach()) == 0
                )
                if not initial_identity:
                    raise RuntimeError("zero-init pyramid is not bitwise M20 identity")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter.trainable_parameters(),
                float(protocol["training"]["gradient_clip_norm"]),
            )
            step_gradient_l1 = {}
            for name, parameter in adapter.expert.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("missing/non-finite pyramid gradient: {}".format(name))
                value = float(parameter.grad.detach().abs().sum())
                step_gradient_l1[name] = value
                cumulative_gradient_l1[name] += value
            if step == 1 and step_gradient_l1["output_projection.weight"] <= 0.0:
                raise RuntimeError("zero output projection was unreachable on step one")
            optimizer.step()
            for name, parameter in adapter.expert.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError("non-finite pyramid parameter after step: {}".format(name))
            dual_state.update(recall, suppression)
            weights = parts.mixture_weights.detach().float()
            mixture_entropy = float(
                (-(weights * weights.clamp_min(torch.finfo(weights.dtype).eps).log()).sum(dim=2)).mean()
            )
            records.append(
                {
                    "step": step,
                    "view_start_bin": start,
                    "view_stop_bin_exclusive": stop,
                    **diagnostics,
                    "gradient_norm": float(gradient_norm),
                    "output_projection_gradient_l1": step_gradient_l1["output_projection.weight"],
                    "scale_encoder_gradient_l1": sum(
                        value for name, value in step_gradient_l1.items() if name.startswith("scale_encoder.")
                    ),
                    "mixture_projection_gradient_l1": sum(
                        value for name, value in step_gradient_l1.items() if name.startswith("mixture_projection.")
                    ),
                    "dual_target_time_recall_after": float(dual_state.target_time_recall),
                    "dual_hard_negative_suppression_after": float(
                        dual_state.hard_negative_suppression
                    ),
                    "mixture_entropy": mixture_entropy,
                    "correction_abs_mean": float(parts.correction.detach().float().abs().mean()),
                    "event_count": int(refined_events.numel()),
                    "hard_negative_component_count": len(metadata["hard_negative_components"]),
                }
            )
            second_decoder_bins += stop - start
            del (
                frames,
                decoder,
                base_logits,
                centre,
                summaries,
                parts,
                refined_events,
                base_events,
                label_tensor,
                target_tensor,
                time_tensor,
                loss,
            )
        validate_pyramid_step_diagnostics(records, PROBE_STEPS)
        updates = parameter_update_l1(adapter.expert, expert_before)
        updated_tensors = [name for name, value in updates.items() if value > 0.0]
        reached_tensors = [
            name for name, value in cumulative_gradient_l1.items() if value > 0.0
        ]
        parameter_tensor_count = len(expert_before)
        long_context_reached = bool(
            sum(value for name, value in cumulative_gradient_l1.items() if name.startswith("scale_encoder.")) > 0
            and sum(value for name, value in cumulative_gradient_l1.items() if name.startswith("mixture_projection.")) > 0
        )
        all_parameter_tensors_updated = len(updated_tensors) == parameter_tensor_count
        all_parameter_tensors_reached = len(reached_tensors) == parameter_tensor_count

        m20_after = atomic.state_sha256(m20.state_dict())
        if m20_after != m20_before:
            raise RuntimeError("released M20 changed during pyramid probe")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        memory_gate = peak_mib <= 3.5 * 1024.0
        mechanical_gates = {
            "FP32_no_update_gradient_audit_passed": (
                fp32_gradient_audit["loss_terms_finite"]
                and fp32_gradient_audit["all_parameter_tensor_gradients_finite"]
                and fp32_gradient_audit["global_gradient_l1"] > 0.0
            ),
            "full_T160_memory_pass": True,
            "first_streaming_decoder_pass_all_160_bins": observation_decoder_bins == 160,
            "second_streaming_decoder_training_views_all_128_bins": second_decoder_bins == 128,
            "initial_actual_M20_bitwise_identity": bool(initial_identity),
            "trainable_parameter_count_3381": pyramid_expert_parameter_count(adapter) == 3381,
            "both_dynamic_constraints_present_every_step": all(
                record["target_time_group_count"] > 0
                and record["hard_negative_component_count"] > 0
                for record in records
            ),
            "output_projection_reached_step_one": records[0]["output_projection_gradient_l1"] > 0,
            "long_context_scale_and_mixture_reached_by_step_eight": long_context_reached,
            "all_parameter_tensors_have_finite_gradient_and_update": (
                all_parameter_tensors_reached and all_parameter_tensors_updated
            ),
            "released_M20_bitwise_unchanged": m20_after == m20_before,
            "peak_CUDA_not_above_3_5_GiB": memory_gate,
            "no_held_G3_or_other_source_read": True,
            "no_validation_or_test_read": True,
        }
        payload = {
            "schema": "ev-uav-h2-multiscale-temporal-pyramid-eight-step-probe-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "model_sha256": sha256_file(
                ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py"
            ),
            "loss_sha256": sha256_file(ROOT / "utils" / "h2_multiscale_pyramid_loss.py"),
            "released_m20_sha256": sha256_file(atomic.M20_PATH),
            "released_m20_state_sha256_before": m20_before,
            "released_m20_state_sha256_after": m20_after,
            "source": SOURCE_NAME,
            "source_sha256": SOURCE_SHA256,
            "source_event_count": SOURCE_EVENT_COUNT,
            "source_is_fit_only_for_first_fold": True,
            "held_group": "g3_095_098",
            "held_G3_array_read": False,
            "other_source_array_read": False,
            "validation_or_test_read": False,
            "complete_temporal_bins": TEMPORAL_COUNT,
            "first_streaming_decoder_bins": observation_decoder_bins,
            "second_streaming_decoder_bins": second_decoder_bins,
            "eligible_fit_only_view_count": eligible_view_count,
            "selected_view_starts": [record["view_start_bin"] for record in records],
            "base_C00": effective_c00,
            "base_C00_stats": asdict(postprocess_stats),
            "base_component_count_per_bin_topology": len(components.event_indices),
            "pure_FP_component_count_per_bin_topology": len(pure_fp_components),
            "optimizer": protocol["training"]["optimizer"],
            "frozen_M20_numeric_precision": "FP32",
            "expert_loss_backward_optimizer_numeric_precision": "FP32_no_GradScaler",
            "optimizer_steps": PROBE_STEPS,
            "FP32_no_update_gradient_audit": fp32_gradient_audit,
            "trainable_parameter_count": pyramid_expert_parameter_count(adapter),
            "parameter_tensor_count": parameter_tensor_count,
            "updated_parameter_tensors": updated_tensors,
            "reached_parameter_tensors": reached_tensors,
            "parameter_update_l1": updates,
            "cumulative_gradient_l1": cumulative_gradient_l1,
            "all_step_diagnostics": records,
            "dual_state_after": dual_state.to_dict(),
            "peak_CUDA_MiB": peak_mib,
            "peak_budget_MiB": 3.5 * 1024.0,
            "mechanical_gates": mechanical_gates,
            "mechanical_passed": all(mechanical_gates.values()),
            "formal_started": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not payload["mechanical_passed"]:
            raise RuntimeError(
                "pyramid probe mechanical gate failed: {}".format(
                    {name: value for name, value in mechanical_gates.items() if not value}
                )
            )
        del (
            adapter,
            m20,
            checkpoint,
            memory,
            summary_cache,
            raw_scores,
            base_scores,
            processed,
            optimizer,
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        payload["CUDA_allocated_after_release_MiB"] = torch.cuda.memory_allocated() / (
            1024.0 ** 2
        )

    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    write_bytes_exclusive(OUTPUT_PATH, serialized)
    receipt_sha256 = hashlib.sha256(serialized).hexdigest()
    write_bytes_exclusive(
        Path(str(OUTPUT_PATH) + ".sha256"),
        (receipt_sha256 + "  " + OUTPUT_PATH.name + "\n").encode("ascii"),
    )
    print(
        json.dumps(
            {
                "receipt": str(OUTPUT_PATH.resolve()),
                "receipt_sha256": receipt_sha256,
                "mechanical_passed": payload["mechanical_passed"],
                "peak_CUDA_MiB": payload["peak_CUDA_MiB"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "formal_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
    return value


if __name__ == "__main__":
    raise SystemExit(run_probe(parser().parse_args()))

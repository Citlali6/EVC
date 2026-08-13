"""Formal fresh-held-G1 chain for pyramid suppression plus atomic recovery V2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import crossfit_component_reranker as crossfit
import run_h2_atomic_component_deletion_v3 as atomic
import run_h2_multiscale_temporal_pyramid_formal as stage1_parent
import run_h2_multiscale_temporal_pyramid_probe as probe
from model.h2_multiscale_temporal_pyramid_expert import (
    FrozenM20MultiScalePyramidAdapter,
    pyramid_expert_parameter_count,
)
from model.h2_pyramid_component_recovery import (
    H2PyramidComponentRecoveryHead,
    NODE_FEATURE_DIM,
    component_recovery_parameter_count,
)
from utils.atomic_component_deletion import extract_atomic_components
from utils.h2_multiscale_pyramid_loss import (
    PyramidDualState,
    multiscale_pyramid_constrained_loss,
    validate_pyramid_step_diagnostics,
)
from utils.h2_pyramid_component_recovery import restore_whole_components_bitwise
from utils.postprocess import ChallengePostprocessor
from utils.target_preserving_residual import use_h2_residual_refiner


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TRAIN_ROOT = (WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train").resolve()
PROTOCOL_PATH = ROOT / "protocols" / "h2_pyramid_component_recovery_v2_science_v1.json"
SOURCE_PROTOCOL_PATH = ROOT / "protocols" / "h2_spatiotemporal_residual_refiner_oof_science_v1.json"
EXPECTED_PROTOCOL_SHA256 = "4c4c260b66bf4c77fb314432bd2c72432a3273917347a8f5bf943d8489933c70"
EXPECTED_SOURCE_PROTOCOL_SHA256 = "7edec461f2ccc8047156f08c57389319a5defd59d0afcea69cbfcf32e81d2207"
EXPECTED_STAGE1_SCIENCE_SHA256 = "0bdb6e0657483e253b363462ffad6969dcd85df52ef5707d32ed93a914268155"
EXPECTED_STAGE1_MODEL_SHA256 = "4d4ea4a365be49ad1b6c7cf1c7c96c2369caf3e12841bbcd781cf109105a6a98"
EXPECTED_STAGE1_LOSS_SHA256 = "f74e145b04b25f2e7478f5c8fd370bc4e9d96123ef6f75ea5833acc210d2c5e9"
EXPECTED_RECOVERY_MODEL_SHA256 = "f55224d30b9d8ddeab463dd433f77e0dc961d45d2ecf60cf255a6ddcd0447352"
EXPECTED_RECOVERY_UTILITY_SHA256 = "cd8fb85dda17ef177c174b9051a6d089c7a703d67f393dbf80aca656012f9f78"
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EXPECTED_EXECUTION_DEPENDENCIES = {
    "run_h2_multiscale_temporal_pyramid_probe.py": "2cd894d046a1433be7b8fc06b95bb616ed82b53ce2078aac082d08ba80eb1df4",
    "run_h2_multiscale_temporal_pyramid_formal.py": "33ca6078c810f52c8a549376bb8d862bae1068561bd7ccd8e5c1479e1f8fe89f",
    "run_h2_atomic_component_deletion_v3.py": "8c04e72306f8ed9a5944491fe13567518bb6d62f5d384eab0b7879bb34e6f884",
    "crossfit_component_reranker.py": "ec4d0af1abf8dab88037d8731ad1b142aa8cdb76511287d866113865d70e9561",
    "utils/postprocess.py": "a43e2a22947f5307cdb01b810a4f1dc8d4fb624b14bd068baf1e9087ca5d1aa4",
    "utils/atomic_component_deletion.py": "b9d86aa7203686c512ecfd9d143074c06e367fb6b1b7cb658a262b8c04f148c5",
    "utils/target_preserving_residual.py": "e4377446fb8834a6e2860da228fcabcdeaa46f04ab036dddb1217c9f1e364b0e",
    "utils/eval.py": "cb9de56b54a7153d80a2fd32432188665e830e0a9b945965f2cbd6765bb354e5",
    "utils/challenge_eval.py": "a09e9a4b74abb06864e4551778addd624e53654041c98a41dbe61a1f67fbc952",
    "model/temporal_memory_net.py": "d90c550bc2928482684c1216f52c4752700c27b79935a9552208bc2bca00f32f",
    "dataset/temporal_frame.py": "1f6bed24ebdbae671459028d8f2ff37f179dea9bb2ef23191d45c9e010ac83ee",
}

G1 = tuple("train_{:03d}.npz".format(value) for value in range(88, 92))
G2 = tuple("train_{:03d}.npz".format(value) for value in range(92, 95))
G3 = tuple("train_{:03d}.npz".format(value) for value in range(95, 99))
OUTER_FIT = G2 + G3
FEATURE_ORDER = G3 + G2
TEMPORAL_COUNT = 160
VIEW_BINS = 16
INFERENCE_BATCH = 8
THRESHOLD = 0.719
STAGE1_SEED = 67
RECOVERY_SEED = 67
STAGE1_EPOCHS = 2
STAGE1_VIEWS = 4
STAGE1_STEPS = 56
RECOVERY_EPOCHS = 8
RECOVERY_FULL_STEPS = 56
MAX_COMPONENTS_PER_SOURCE = 64
MAX_TEMPORAL_NODES_PER_COMPONENT = 160
GPU_FLAG = "--root-authorized-gpu"

OUTPUT_ROOT = WORKSPACE / "experiments" / "20260811_h2_pyramid_component_recovery_v2"
STAGE1_ROOT = OUTPUT_ROOT / "formal_stage1" / "fresh_hold_g1"
STAGE1_CHECKPOINT = STAGE1_ROOT / "final_expert.pt"
STAGE1_RECEIPT = STAGE1_ROOT / "training_result.json"
FEATURE_ROOT = OUTPUT_ROOT / "fit_feature_cache"
FEATURE_RECEIPT = FEATURE_ROOT / "manifest.json"
PROBE_ROOT = OUTPUT_ROOT / "recovery_head_mechanical_probe"
PROBE_RECEIPT = PROBE_ROOT / "eight_step_probe.json"
INNER_ROOT = OUTPUT_ROOT / "nested_recovery" / "hold_g1"
INNER_RESULT = INNER_ROOT / "inner_result.json"
INNER_DECISION = INNER_ROOT / "inner_decision.json"
G3_HEAD_CHECKPOINT = INNER_ROOT / "fit_G3_head.pt"
FINAL_HEAD_CHECKPOINT = INNER_ROOT / "final_recovery_head.pt"
OUTER_ROOT = OUTPUT_ROOT / "held_train_evaluation" / "fresh_hold_g1"
OUTER_INFERENCE_MANIFEST = OUTER_ROOT / "input_only_inference_manifest.json"
OUTER_RESULT = OUTER_ROOT / "paired_evaluation.json"
OUTER_DECISION = OUTER_ROOT / "branch_decision.json"
FORMAL_AUDIT_ROOT = OUTPUT_ROOT / "formal_runner_cpu_audit"
FORMAL_AUDIT_RECEIPT = FORMAL_AUDIT_ROOT / "report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    return probe.sha256_file(Path(path))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def json_bytes(payload):
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_bytes_exclusive(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(values)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_json_exclusive(path, payload):
    values = json_bytes(payload)
    write_bytes_exclusive(path, values)
    digest = hashlib.sha256(values).hexdigest()
    write_bytes_exclusive(
        Path(str(path) + ".sha256"),
        (digest + "  " + Path(path).name + "\n").encode("ascii"),
    )
    return digest


def write_torch_exclusive(path, payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    values = buffer.getvalue()
    write_bytes_exclusive(path, values)
    digest = hashlib.sha256(values).hexdigest()
    write_bytes_exclusive(
        Path(str(path) + ".sha256"),
        (digest + "  " + Path(path).name + "\n").encode("ascii"),
    )
    return digest


def write_npz_exclusive(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    values = buffer.getvalue()
    write_bytes_exclusive(path, values)
    digest = hashlib.sha256(values).hexdigest()
    write_bytes_exclusive(
        Path(str(path) + ".sha256"),
        (digest + "  " + Path(path).name + "\n").encode("ascii"),
    )
    return digest


def array_receipt(value):
    array = np.ascontiguousarray(np.asarray(value))
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "canonical_content_sha256": digest,
    }


def verify_sidecar(path):
    path = Path(path)
    actual = sha256_file(path)
    tokens = Path(str(path) + ".sha256").read_text(encoding="ascii").split()
    if len(tokens) != 2 or tokens[0] != actual or tokens[1] != path.name:
        raise RuntimeError("sidecar mismatch: {}".format(path))
    return actual


def load_protocol():
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen V2 protocol changed")
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != (
        "frozen_after_G3_development_before_any_G2_or_G1_array_read_or_V2_GPU"
    ):
        raise RuntimeError("V2 protocol is not frozen")
    if protocol["resource_budget_before_GPU"]["authorized_GPU"] is not False:
        raise RuntimeError("protocol file must remain GPU-unauthorized; CLI carries authorization")
    fixed = {
        ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py": EXPECTED_STAGE1_MODEL_SHA256,
        ROOT / "utils" / "h2_multiscale_pyramid_loss.py": EXPECTED_STAGE1_LOSS_SHA256,
        ROOT / "model" / "h2_pyramid_component_recovery.py": EXPECTED_RECOVERY_MODEL_SHA256,
        ROOT / "utils" / "h2_pyramid_component_recovery.py": EXPECTED_RECOVERY_UTILITY_SHA256,
        atomic.M20_PATH: EXPECTED_M20_SHA256,
        SOURCE_PROTOCOL_PATH: EXPECTED_SOURCE_PROTOCOL_SHA256,
        ROOT / "protocols" / "h2_multiscale_temporal_pyramid_expert_science_v1.json": EXPECTED_STAGE1_SCIENCE_SHA256,
    }
    for path, expected in fixed.items():
        if sha256_file(path) != expected:
            raise RuntimeError("frozen dependency changed: {}".format(path))
    for relative_path, expected in EXPECTED_EXECUTION_DEPENDENCIES.items():
        path = ROOT / relative_path
        if sha256_file(path) != expected:
            raise RuntimeError("frozen execution dependency changed: {}".format(path))
    if protocol["Stage1_pyramid"]["V1_checkpoint_reuse_allowed"] is not False:
        raise RuntimeError("V2 unexpectedly permits V1 checkpoint reuse")
    if tuple(protocol["source_manifest"]["groups"]["G1_fresh_outer_held"]) != G1:
        raise RuntimeError("G1 source group changed")
    if tuple(protocol["source_manifest"]["groups"]["G2_outer_fit"]) != G2:
        raise RuntimeError("G2 source group changed")
    if tuple(protocol["source_manifest"]["groups"]["G3_outer_fit_design_exposed"]) != G3:
        raise RuntimeError("G3 source group changed")
    return protocol


def source_manifest():
    return read_json(SOURCE_PROTOCOL_PATH)["h2_sources"]


def source_path(source_name, manifest):
    if source_name not in manifest:
        raise RuntimeError("source lies outside frozen H2 manifest")
    path = (TRAIN_ROOT / source_name).resolve()
    if path.parent != TRAIN_ROOT:
        raise RuntimeError("source escaped official train root")
    return path


def verify_source(source_name, manifest):
    path = source_path(source_name, manifest)
    if sha256_file(path) != manifest[source_name]["sha256"]:
        raise RuntimeError("source SHA mismatch: {}".format(source_name))
    return path


def require_gpu(args):
    if not bool(getattr(args, "root_authorized_gpu", False)):
        raise PermissionError("GPU command requires {}".format(GPU_FLAG))
    formal_audit_gate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def seed_all(seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_c00():
    cfg, effective = probe.build_c00()
    if crossfit.sha256_json(effective) != stage1_parent.EXPECTED_C00_SHA256:
        raise RuntimeError("effective C00 changed")
    return cfg, effective


def load_input_truth(path, expected_count):
    return stage1_parent.load_input_and_truth(path, expected_count)


def stage1_checkpoint_gate():
    checkpoint_sha = verify_sidecar(STAGE1_CHECKPOINT)
    receipt_sha = verify_sidecar(STAGE1_RECEIPT)
    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location="cpu", weights_only=False)
    receipt = read_json(STAGE1_RECEIPT)
    if checkpoint.get("schema") != "ev-uav-h2-pyramid-v2-fresh-stage1-checkpoint-v1":
        raise RuntimeError("unexpected fresh Stage1 checkpoint")
    if tuple(checkpoint.get("fit_sources", ())) != OUTER_FIT:
        raise RuntimeError("fresh Stage1 fit set changed")
    if checkpoint.get("held_G1_arrays_read") is not False:
        raise RuntimeError("fresh Stage1 checkpoint does not prove G1 unread")
    if checkpoint.get("validation_or_test_read") is not False:
        raise RuntimeError("fresh Stage1 checkpoint does not prove val/test unread")
    if checkpoint.get("optimizer_steps") != STAGE1_STEPS:
        raise RuntimeError("fresh Stage1 step count changed")
    if checkpoint.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("fresh Stage1 protocol binding changed")
    if checkpoint.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V2 runner changed after fresh Stage1")
    if receipt.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("fresh Stage1 receipt checkpoint mismatch")
    if receipt.get("schema") != (
        "ev-uav-h2-pyramid-v2-fresh-stage1-training-result-v1"
    ):
        raise RuntimeError("unexpected fresh Stage1 receipt")
    if receipt.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("fresh Stage1 receipt protocol binding changed")
    if receipt.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("fresh Stage1 receipt runner binding changed")
    if tuple(receipt.get("fit_sources", ())) != OUTER_FIT:
        raise RuntimeError("fresh Stage1 receipt fit set changed")
    if receipt.get("held_G1_arrays_read") is not False:
        raise RuntimeError("fresh Stage1 receipt does not prove G1 unread")
    if receipt.get("validation_or_test_read") is not False:
        raise RuntimeError("fresh Stage1 receipt does not prove val/test unread")
    if receipt.get("all_expert_parameter_tensors_updated") is not True:
        raise RuntimeError("fresh Stage1 receipt lacks complete parameter update audit")
    if atomic.state_sha256(checkpoint["expert_state_dict"]) != checkpoint[
        "expert_state_sha256"
    ]:
        raise RuntimeError("fresh Stage1 expert state hash mismatch")
    return checkpoint, receipt, checkpoint_sha, receipt_sha


def train_stage1(args):
    require_gpu(args)
    protocol = load_protocol()
    cfg, effective_c00 = build_c00()
    if STAGE1_ROOT.exists():
        raise FileExistsError("refusing to overwrite fresh V2 Stage1")
    if FEATURE_ROOT.exists() or INNER_ROOT.exists() or OUTER_ROOT.exists():
        raise RuntimeError("later V2 stage exists before fresh Stage1")
    manifest = source_manifest()
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_fresh_stage1_hold_g1"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_all(STAGE1_SEED)
        m20, _ = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        if pyramid_expert_parameter_count(adapter) != 3381:
            raise RuntimeError("fresh Stage1 parameter count changed")
        if any(parameter.requires_grad for parameter in m20.parameters()):
            raise RuntimeError("released M20 is not frozen")
        adapter.train()
        optimizer = torch.optim.AdamW(
            adapter.trainable_parameters(),
            lr=0.0003,
            weight_decay=0.0001,
        )
        dual = PyramidDualState()
        records = []
        source_records = []
        fit_initial_hashes = {}
        expert_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in adapter.expert.named_parameters()
        }
        step = 0
        initial_identity = None
        for epoch_zero in range(STAGE1_EPOCHS):
            for source_position, source_name in enumerate(OUTER_FIT):
                path = verify_source(source_name, manifest)
                expected = manifest[source_name]
                fit_initial_hashes.setdefault(source_name, expected["sha256"])
                video, polarities, locations4, labels, target_ids = load_input_truth(
                    path, expected["event_count"]
                )
                memory = probe.full_stream_memory(m20, video, device)
                observations, base_raw, decoded_bins = probe.stream_observations_and_scores(
                    adapter, video, memory, device
                )
                summaries = probe.build_summary_cache(observations, device)
                del observations
                components, c00_stats, base_component_count = (
                    stage1_parent.extract_fit_hard_negatives(
                        cfg, base_raw, locations4, labels
                    )
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
                    if not np.array_equal(sampled, metadata["global_indices"]):
                        raise RuntimeError("fresh Stage1 candidate event order changed")
                    if not np.array_equal(base_sampled, sampled):
                        raise RuntimeError("fresh Stage1 paired event order changed")
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
                            refined.float(),
                            base_events.float(),
                            label_tensor,
                            target_tensor,
                            time_tensor,
                            metadata["hard_negative_components"],
                            dual,
                        )
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("fresh Stage1 FP32 loss is non-finite")
                    if step == 1:
                        initial_identity = bool(
                            torch.equal(
                                parts.refined_logits.detach(),
                                base_logits.unsqueeze(0),
                            )
                            and torch.count_nonzero(parts.correction.detach()) == 0
                        )
                        if not initial_identity:
                            raise RuntimeError("fresh Stage1 is not zero-init M20 identity")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        adapter.trainable_parameters(), 5.0
                    )
                    gradient_l1 = {}
                    for name, parameter in adapter.expert.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("fresh Stage1 gradient failure: {}".format(name))
                        gradient_l1[name] = float(parameter.grad.detach().abs().sum())
                    if step == 1 and gradient_l1["output_projection.weight"] <= 0.0:
                        raise RuntimeError("fresh Stage1 output projection is unreachable")
                    optimizer.step()
                    for name, parameter in adapter.expert.named_parameters():
                        if not torch.isfinite(parameter).all():
                            raise RuntimeError("fresh Stage1 parameter failure: {}".format(name))
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
                            "output_projection_gradient_l1": gradient_l1[
                                "output_projection.weight"
                            ],
                            "scale_encoder_gradient_l1": sum(
                                value
                                for name, value in gradient_l1.items()
                                if name.startswith("scale_encoder.")
                            ),
                            "mixture_projection_gradient_l1": sum(
                                value
                                for name, value in gradient_l1.items()
                                if name.startswith("mixture_projection.")
                            ),
                            "dual_target_time_recall_after": float(dual.target_time_recall),
                            "dual_hard_negative_suppression_after": float(
                                dual.hard_negative_suppression
                            ),
                            "mixture_entropy": float(
                                (-(weights * weights.clamp_min(1e-12).log()).sum(dim=2)).mean()
                            ),
                            "correction_abs_mean": float(
                                parts.correction.detach().float().abs().mean()
                            ),
                            "event_count": int(refined.numel()),
                        }
                    )
                    del (
                        frames,
                        decoder,
                        base_logits,
                        centre,
                        summary_views,
                        parts,
                        refined,
                        base_events,
                        label_tensor,
                        target_tensor,
                        time_tensor,
                        loss,
                    )
                source_records.append(
                    {
                        "epoch": epoch_zero + 1,
                        "source_name": source_name,
                        "eligible_view_count": eligible,
                        "selected_starts": [value["start"] for value in views],
                        "M20_C00_component_count": base_component_count,
                        "pure_FP_component_count": len(components),
                        "first_decoder_bins": decoded_bins,
                        "C00_stats": c00_stats,
                    }
                )
                del (
                    video,
                    polarities,
                    locations4,
                    labels,
                    target_ids,
                    memory,
                    summaries,
                    base_raw,
                    components,
                    views,
                )
                torch.cuda.empty_cache()
                print(
                    "V2 fresh Stage1 epoch {}/2 {} step {}/56".format(
                        epoch_zero + 1, source_name, step
                    ),
                    flush=True,
                )
        if step != STAGE1_STEPS:
            raise RuntimeError("fresh Stage1 optimizer-step count mismatch")
        validate_pyramid_step_diagnostics(records, STAGE1_STEPS)
        if not all(record["target_time_group_count"] > 0 for record in records):
            raise RuntimeError("fresh Stage1 step lacks target-time constraint")
        if not all(record["hard_negative_component_count"] > 0 for record in records):
            raise RuntimeError("fresh Stage1 step lacks hard-negative constraint")
        fit_final_hashes = {}
        for source_name in OUTER_FIT:
            path = source_path(source_name, manifest)
            fit_final_hashes[source_name] = sha256_file(path)
            if fit_final_hashes[source_name] != fit_initial_hashes[source_name]:
                raise RuntimeError("fit source changed during fresh Stage1")
        expert_updates = {
            name: float((parameter.detach().cpu() - expert_before[name]).abs().sum())
            for name, parameter in adapter.expert.named_parameters()
        }
        if not all(value > 0 for value in expert_updates.values()):
            raise RuntimeError("not every fresh Stage1 parameter tensor updated")
        m20_after = atomic.state_sha256(m20.state_dict())
        if m20_after != m20_before:
            raise RuntimeError("M20 changed during fresh Stage1")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        expert_state = {
            name: value.detach().cpu().clone()
            for name, value in adapter.expert.state_dict().items()
        }
        checkpoint = {
            "schema": "ev-uav-h2-pyramid-v2-fresh-stage1-checkpoint-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "fit_sources": list(OUTER_FIT),
            "held_sources_reserved_unread": list(G1),
            "held_G1_arrays_read": False,
            "validation_or_test_read": False,
            "fresh_initialization": True,
            "V1_checkpoint_reused": False,
            "optimizer_steps": step,
            "dual_state": dual.to_dict(),
            "released_m20_state_sha256": m20_after,
            "expert_state_sha256": atomic.state_sha256(expert_state),
            "expert_state_dict": expert_state,
        }
        checkpoint_sha = write_torch_exclusive(STAGE1_CHECKPOINT, checkpoint)
        reloaded = torch.load(STAGE1_CHECKPOINT, map_location="cpu", weights_only=False)
        if atomic.state_sha256(reloaded["expert_state_dict"]) != reloaded[
            "expert_state_sha256"
        ]:
            raise RuntimeError("persisted fresh Stage1 state failed verification")
        receipt = {
            "schema": "ev-uav-h2-pyramid-v2-fresh-stage1-training-result-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "fit_sources": list(OUTER_FIT),
            "held_G1_arrays_read": False,
            "validation_or_test_read": False,
            "checkpoint_path": str(STAGE1_CHECKPOINT.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "optimizer_steps": step,
            "initial_M20_bitwise_identity": bool(initial_identity),
            "all_expert_parameter_tensors_updated": True,
            "expert_parameter_update_l1": expert_updates,
            "dual_final": dual.to_dict(),
            "constraint_trends": stage1_parent.constraint_trends(records),
            "all_step_diagnostics": records,
            "source_epoch_diagnostics": source_records,
            "fit_source_sha256_before": fit_initial_hashes,
            "fit_source_sha256_after": fit_final_hashes,
            "effective_C00": effective_c00,
            "released_m20_state_sha256_before": m20_before,
            "released_m20_state_sha256_after": m20_after,
            "peak_CUDA_MiB": peak_mib,
            "elapsed_seconds": time.perf_counter() - started,
        }
        receipt_sha = write_json_exclusive(STAGE1_RECEIPT, receipt)
        verify_sidecar(STAGE1_CHECKPOINT)
        verify_sidecar(STAGE1_RECEIPT)
        del reloaded, checkpoint, expert_state, adapter, m20, optimizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)
    print(
        json.dumps(
            {
                "stage": "fresh_Stage1_complete",
                "checkpoint_sha256": checkpoint_sha,
                "receipt_sha256": receipt_sha,
                "optimizer_steps": step,
                "constraint_trends": receipt["constraint_trends"],
                "peak_CUDA_MiB": peak_mib,
                "CUDA_after_release_MiB": after_mib,
                "held_G1_arrays_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def apply_c00(cfg, scores, locations4):
    processed, stats = ChallengePostprocessor.from_cfg(
        cfg, THRESHOLD, event_count=len(scores)
    ).apply(
        torch.from_numpy(np.asarray(scores, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(locations4, dtype=np.int64)).long(),
    )
    return processed.numpy().astype(np.float32, copy=True), asdict(stats)


def _node_geometry(component, video, stage1_post):
    component = np.asarray(component, dtype=np.int64)
    locations = video.locations[component].astype(np.float64, copy=False)
    bins = np.floor_divide(locations[:, 2].astype(np.int64), 50)
    unique_bins = np.unique(bins)
    component_centroid = locations[:, :2].mean(axis=0)
    component_width = max(float(np.ptp(locations[:, 0]) + 1.0), 1.0)
    component_height = max(float(np.ptp(locations[:, 1]) + 1.0), 1.0)
    midpoint = 0.5 * (float(unique_bins[0]) + float(unique_bins[-1]))
    span = max(float(unique_bins[-1] - unique_bins[0]), 1.0)
    output = []
    previous = None
    for temporal_bin in unique_bins:
        indices = component[bins == temporal_bin]
        values = video.locations[indices].astype(np.float64, copy=False)
        centroid = values[:, :2].mean(axis=0)
        if previous is None:
            delta = np.zeros(2, dtype=np.float64)
        else:
            delta = centroid - previous
        previous = centroid
        geometry = np.asarray(
            [
                np.log1p(indices.size),
                indices.size / component.size,
                (float(temporal_bin) - midpoint) / span,
                (centroid[0] - component_centroid[0]) / component_width,
                (centroid[1] - component_centroid[1]) / component_height,
                float(np.ptp(values[:, 0]) + 1.0) / component_width,
                float(np.ptp(values[:, 1]) + 1.0) / component_height,
                delta[0] / 346.0,
                delta[1] / 260.0,
                float(np.mean(stage1_post[indices] >= np.float32(THRESHOLD))),
            ],
            dtype=np.float32,
        )
        output.append((int(temporal_bin), indices, geometry))
    return output


def build_input_only_source_cache(adapter, video, polarities, locations4, cfg, device):
    memory = probe.full_stream_memory(adapter.released_m20, video, device)
    observations, base_raw, first_bins = probe.stream_observations_and_scores(
        adapter, video, memory, device
    )
    summaries = probe.build_summary_cache(observations, device)
    del observations
    stage1_raw = np.empty_like(base_raw)
    with torch.no_grad():
        for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
            stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
            frames = atomic._frame_tensor(video, range(start, stop), device)
            decoder, base_logits, centre = adapter.decode_frozen_features(
                frames, memory[start:stop]
            )
            summary_views = tuple(
                value[start:stop].to(device=device, dtype=torch.float32)
                for value in summaries
            )
            refined = adapter.expert(
                decoder.unsqueeze(0),
                base_logits.unsqueeze(0),
                centre.unsqueeze(0),
                tuple(value.unsqueeze(0) for value in summary_views),
            ).squeeze(0)
            probabilities = torch.sigmoid(refined).squeeze(1).cpu().numpy()
            for temporal_bin in range(start, stop):
                indices = video.event_indices_by_bin[temporal_bin]
                if indices.size == 0:
                    continue
                xy = video.locations[indices]
                stage1_raw[indices] = probabilities[
                    temporal_bin - start, xy[:, 1], xy[:, 0]
                ]
            del frames, decoder, base_logits, centre, summary_views, refined, probabilities
    if not np.isfinite(stage1_raw).all():
        raise RuntimeError("input-only Stage1 raw scores are non-finite")
    base_post, base_stats = apply_c00(cfg, base_raw, locations4)
    stage1_post, stage1_stats = apply_c00(cfg, stage1_raw, locations4)
    all_components = extract_atomic_components(
        base_post,
        locations4,
        THRESHOLD,
        spatial_radius=2,
        temporal_bin_size=50,
        temporal_radius_bins=1,
    ).event_indices
    components = tuple(
        component
        for component in all_components
        if np.any(stage1_post[component] < np.float32(THRESHOLD))
    )
    if not components:
        raise RuntimeError("fit source has no Stage1/M20 threshold-disagreement component")
    if len(components) > MAX_COMPONENTS_PER_SOURCE:
        raise RuntimeError("component source batch exceeded frozen 64-component budget")
    node_specs = [_node_geometry(component, video, stage1_post) for component in components]
    node_lookup = {}
    for component_index, nodes in enumerate(node_specs):
        for node_index, (temporal_bin, indices, geometry) in enumerate(nodes):
            node_lookup.setdefault(temporal_bin, []).append(
                (component_index, node_index, indices, geometry)
            )
    node_dense = [
        [None for _ in nodes]
        for nodes in node_specs
    ]
    feature_decoder_bins = 0
    with torch.no_grad():
        for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
            stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
            frames = atomic._frame_tensor(video, range(start, stop), device)
            decoder, base_logits, centre = adapter.decode_frozen_features(
                frames, memory[start:stop]
            )
            summary_views = tuple(
                value[start:stop].to(device=device, dtype=torch.float32)
                for value in summaries
            )
            parts = adapter.expert(
                decoder.unsqueeze(0),
                base_logits.unsqueeze(0),
                centre.unsqueeze(0),
                tuple(value.unsqueeze(0) for value in summary_views),
                return_parts=True,
            )
            encoded_scales = []
            for scale_index, summary in enumerate(summary_views):
                encoded = adapter.expert.scale_encoder(summary)
                encoded = encoded + adapter.expert.scale_tokens[scale_index].view(
                    1, -1, 1, 1
                )
                encoded_scales.append(
                    F.interpolate(
                        encoded,
                        size=decoder.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            for temporal_bin in range(start, stop):
                local = temporal_bin - start
                for component_index, node_index, indices, geometry in node_lookup.get(
                    temporal_bin, ()
                ):
                    xy = video.locations[indices]
                    xs = torch.from_numpy(xy[:, 0]).to(device=device, dtype=torch.long)
                    ys = torch.from_numpy(xy[:, 1]).to(device=device, dtype=torch.long)
                    decoder_mean = decoder[local, :, ys, xs].mean(dim=1)
                    scale_means = [
                        encoded[local, :, ys, xs].mean(dim=1)
                        for encoded in encoded_scales
                    ]
                    base_mean = base_logits[local, 0, ys, xs].mean().reshape(1)
                    stage1_mean = parts.refined_logits[0, local, 0, ys, xs].mean().reshape(1)
                    delta_mean = (stage1_mean - base_mean).reshape(1)
                    centre_mean = centre[local, :, ys, xs].mean(dim=1)
                    dense = torch.cat(
                        (
                            decoder_mean,
                            *scale_means,
                            base_mean,
                            stage1_mean,
                            delta_mean,
                            centre_mean,
                        )
                    ).detach().cpu().float().numpy()
                    if dense.size != 86:
                        raise RuntimeError("dense component-node feature count changed")
                    node_dense[component_index][node_index] = np.concatenate(
                        (dense, geometry), axis=0
                    ).astype(np.float32, copy=False)
            feature_decoder_bins += stop - start
            del (
                frames,
                decoder,
                base_logits,
                centre,
                summary_views,
                parts,
                encoded_scales,
            )
    node_features = []
    for values in node_dense:
        if any(value is None for value in values):
            raise RuntimeError("not every component temporal node received features")
        joined = np.stack(values).astype(np.float32, copy=False)
        if joined.shape[1] != NODE_FEATURE_DIM or not np.isfinite(joined).all():
            raise RuntimeError("invalid component node feature matrix")
        if joined.shape[0] > MAX_TEMPORAL_NODES_PER_COMPONENT:
            raise RuntimeError("component exceeded frozen 160-node budget")
        node_features.append(joined)
    del memory, summaries
    return {
        "schema": "ev-uav-h2-pyramid-recovery-input-only-feature-cache-v1",
        "event_count": len(polarities),
        "locations4": locations4.astype(np.int64, copy=True),
        "polarities": np.asarray(polarities, dtype=np.float32).copy(),
        "base_raw": base_raw.astype(np.float32, copy=True),
        "stage1_raw": stage1_raw.astype(np.float32, copy=True),
        "base_post": base_post,
        "stage1_post": stage1_post,
        "components": tuple(np.asarray(value, dtype=np.int64) for value in components),
        "node_features": tuple(node_features),
        "contains_labels_or_target_ids": False,
        "base_C00_stats": base_stats,
        "stage1_C00_stats": stage1_stats,
        "M20_component_count": len(all_components),
        "disagreement_component_count": len(components),
        "first_decoder_bins": first_bins,
        "feature_decoder_bins": feature_decoder_bins,
    }


def extract_fit_features(args):
    require_gpu(args)
    load_protocol()
    checkpoint, _, checkpoint_sha, receipt_sha = stage1_checkpoint_gate()
    cfg, effective_c00 = build_c00()
    if FEATURE_ROOT.exists():
        raise FileExistsError("refusing to overwrite V2 fit feature cache")
    if INNER_ROOT.exists() or OUTER_ROOT.exists():
        raise RuntimeError("later V2 stage exists before fit feature extraction")
    manifest = source_manifest()
    started = time.perf_counter()
    cache_records = []
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_fit_input_features"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m20, _ = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        adapter.expert.load_state_dict(checkpoint["expert_state_dict"], strict=True)
        adapter.eval()
        for source_name in FEATURE_ORDER:
            path = verify_source(source_name, manifest)
            video, polarities, locations4 = atomic._load_input_only(path)
            if len(polarities) != int(manifest[source_name]["event_count"]):
                raise RuntimeError("fit feature source event count changed")
            if not use_h2_residual_refiner(len(polarities), polarities):
                raise RuntimeError("fit feature source left H2 route")
            cache = build_input_only_source_cache(
                adapter, video, polarities, locations4, cfg, device
            )
            cache.update(
                {
                    "source_name_for_provenance_only": source_name,
                    "source_sha256": manifest[source_name]["sha256"],
                    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                    "runner_sha256": sha256_file(Path(__file__)),
                    "Stage1_checkpoint_sha256": checkpoint_sha,
                    "effective_C00_sha256": crossfit.sha256_json(effective_c00),
                }
            )
            node_payload = {
                "schema": "ev-uav-h2-pyramid-recovery-node-features-v1",
                "source_name_for_provenance_only": source_name,
                "source_sha256": manifest[source_name]["sha256"],
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__)),
                "event_count": cache["event_count"],
                "node_features": cache.pop("node_features"),
                "component_count": len(cache["components"]),
                "contains_labels_or_target_ids": False,
            }
            node_path = FEATURE_ROOT / (
                source_name.replace(".npz", "_node_features.pt")
            )
            node_sha = write_torch_exclusive(node_path, node_payload)
            cache_path = FEATURE_ROOT / (
                source_name.replace(".npz", "_input_scores.pt")
            )
            cache_sha = write_torch_exclusive(cache_path, cache)
            reloaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            node_reloaded = torch.load(node_path, map_location="cpu", weights_only=False)
            if reloaded.get("contains_labels_or_target_ids") is not False:
                raise RuntimeError("fit feature cache truth isolation failed")
            if node_reloaded.get("contains_labels_or_target_ids") is not False:
                raise RuntimeError("fit node cache truth isolation failed")
            if len(node_reloaded["node_features"]) != len(reloaded["components"]):
                raise RuntimeError("fit feature cache component alignment failed")
            cache_records.append(
                {
                    "source_name": source_name,
                    "cache_path": str(cache_path.resolve()),
                    "cache_sha256": cache_sha,
                    "node_cache_path": str(node_path.resolve()),
                    "node_cache_sha256": node_sha,
                    "event_count": cache["event_count"],
                    "M20_component_count": cache["M20_component_count"],
                    "disagreement_component_count": cache[
                        "disagreement_component_count"
                    ],
                    "temporal_node_count": int(
                        sum(len(value) for value in node_payload["node_features"])
                    ),
                    "contains_labels_or_target_ids": False,
                }
            )
            del (
                video,
                polarities,
                locations4,
                cache,
                node_payload,
                reloaded,
                node_reloaded,
            )
            torch.cuda.empty_cache()
            print("V2 fit input-only features {}".format(source_name), flush=True)
        if atomic.state_sha256(m20.state_dict()) != m20_before:
            raise RuntimeError("M20 changed during V2 feature extraction")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        del adapter, m20
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)
    receipt = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-fit-feature-manifest-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "Stage1_checkpoint_sha256": checkpoint_sha,
        "Stage1_receipt_sha256": receipt_sha,
        "source_order": list(FEATURE_ORDER),
        "records": cache_records,
        "all_caches_input_only": True,
        "G1_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
        "effective_C00": effective_c00,
        "peak_CUDA_MiB": peak_mib,
        "CUDA_after_release_MiB": after_mib,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = write_json_exclusive(FEATURE_RECEIPT, receipt)
    print(
        json.dumps(
            {
                "stage": "fit_input_only_features_complete",
                "manifest_sha256": digest,
                "records": cache_records,
                "peak_CUDA_MiB": peak_mib,
                "G1_arrays_or_predictions_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def feature_manifest_gate():
    _, _, stage1_sha, _ = stage1_checkpoint_gate()
    manifest_sha = verify_sidecar(FEATURE_RECEIPT)
    payload = read_json(FEATURE_RECEIPT)
    if payload.get("Stage1_checkpoint_sha256") != stage1_sha:
        raise RuntimeError("feature manifest Stage1 checkpoint mismatch")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("feature manifest protocol binding changed")
    if payload.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V2 runner changed after feature extraction")
    if tuple(payload.get("source_order", ())) != FEATURE_ORDER:
        raise RuntimeError("feature source order changed")
    if payload.get("all_caches_input_only") is not True:
        raise RuntimeError("feature manifest is not input-only")
    if payload.get("G1_arrays_or_predictions_read") is not False:
        raise RuntimeError("feature manifest does not prove G1 unread")
    if payload.get("validation_or_test_read") is not False:
        raise RuntimeError("feature manifest does not prove val/test unread")
    for record in payload["records"]:
        path = Path(record["cache_path"])
        if verify_sidecar(path) != record["cache_sha256"]:
            raise RuntimeError("feature cache SHA mismatch")
        node_path = Path(record["node_cache_path"])
        if verify_sidecar(node_path) != record["node_cache_sha256"]:
            raise RuntimeError("node feature cache SHA mismatch")
    return payload, manifest_sha


def cache_path_for(source_name):
    return FEATURE_ROOT / (source_name.replace(".npz", "_input_scores.pt"))


def node_cache_path_for(source_name):
    return FEATURE_ROOT / (source_name.replace(".npz", "_node_features.pt"))


def load_cache(source_name):
    cache = torch.load(cache_path_for(source_name), map_location="cpu", weights_only=False)
    if cache.get("source_name_for_provenance_only") != source_name:
        raise RuntimeError("feature cache provenance mismatch")
    if cache.get("contains_labels_or_target_ids") is not False:
        raise RuntimeError("feature cache unexpectedly contains truth")
    if cache.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("feature cache runner binding changed")
    return cache


def load_node_cache(source_name):
    cache = torch.load(
        node_cache_path_for(source_name), map_location="cpu", weights_only=False
    )
    if cache.get("source_name_for_provenance_only") != source_name:
        raise RuntimeError("node cache provenance mismatch")
    if cache.get("contains_labels_or_target_ids") is not False:
        raise RuntimeError("node cache unexpectedly contains truth")
    if cache.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("node cache runner binding changed")
    if len(cache["node_features"]) != int(cache["component_count"]):
        raise RuntimeError("node cache component alignment changed")
    return cache


def load_truth_for_cache(source_name, cache, manifest):
    path = verify_source(source_name, manifest)
    labels, target_ids = atomic._load_truth(path)
    if labels.size != int(cache["event_count"]):
        raise RuntimeError("truth/cache event count mismatch")
    return labels, target_ids


def component_labels(cache, labels):
    output = []
    for component in cache["components"]:
        component = np.asarray(component, dtype=np.int64)
        lost = cache["stage1_post"][component] < np.float32(THRESHOLD)
        if not np.any(lost):
            raise RuntimeError("cached action component has no threshold disagreement")
        output.append(bool(np.any(labels[component[lost]] > 0)))
    values = np.asarray(output, dtype=np.float32)
    if not np.any(values > 0.5) or not np.any(values < 0.5):
        raise RuntimeError("recovery component batch needs both truth classes")
    return values


def padded_component_batch(cache, device):
    features = tuple(np.asarray(value, dtype=np.float32) for value in cache["node_features"])
    if not features:
        raise RuntimeError("component batch is empty")
    if len(features) > MAX_COMPONENTS_PER_SOURCE:
        raise RuntimeError("component batch exceeded frozen 64-component budget")
    maximum = max(value.shape[0] for value in features)
    if maximum > MAX_TEMPORAL_NODES_PER_COMPONENT:
        raise RuntimeError("component batch exceeded frozen 160-node budget")
    batch = np.zeros((len(features), maximum, NODE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(features), maximum), dtype=np.bool_)
    for row, value in enumerate(features):
        if value.ndim != 2 or value.shape[1] != NODE_FEATURE_DIM:
            raise RuntimeError("component feature matrix changed")
        batch[row, : value.shape[0]] = value
        mask[row, : value.shape[0]] = True
    return (
        torch.from_numpy(batch).to(device=device, dtype=torch.float32),
        torch.from_numpy(mask).to(device=device, dtype=torch.bool),
    )


def balanced_component_bce(logits, labels):
    positive = labels > 0.5
    negative = ~positive
    if not bool(torch.any(positive)) or not bool(torch.any(negative)):
        raise RuntimeError("balanced component BCE needs both classes")
    return 0.5 * (
        F.softplus(-logits[positive]).mean() + F.softplus(logits[negative]).mean()
    )


def mechanical_probe(args):
    require_gpu(args)
    load_protocol()
    feature_payload, feature_sha = feature_manifest_gate()
    if PROBE_ROOT.exists():
        raise FileExistsError("refusing to overwrite recovery-head mechanical probe")
    if INNER_ROOT.exists() or OUTER_ROOT.exists():
        raise RuntimeError("later V2 stage exists before recovery-head probe")
    manifest = source_manifest()
    source_name = G3[0]
    cache = load_cache(source_name)
    node_cache = load_node_cache(source_name)
    labels, _ = load_truth_for_cache(source_name, cache, manifest)
    targets = component_labels(cache, labels)
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_head_real_batch_probe"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_all(RECOVERY_SEED)
        head = H2PyramidComponentRecoveryHead().to(device)
        if component_recovery_parameter_count(head) != 14081:
            raise RuntimeError("recovery-head parameter count changed")
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.0001)
        features, mask = padded_component_batch(node_cache, device)
        label_tensor = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, mask)
        audit_loss = balanced_component_bce(logits, label_tensor)
        if not torch.isfinite(audit_loss):
            raise RuntimeError("real-batch recovery audit loss is non-finite")
        audit_loss.backward()
        gradient_l1 = {}
        for name, parameter in head.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("real-batch recovery gradient failure: {}".format(name))
            gradient_l1[name] = float(parameter.grad.detach().abs().sum())
        if sum(gradient_l1.values()) <= 0 or gradient_l1["output.3.weight"] <= 0:
            raise RuntimeError("real-batch recovery head is unreachable")
        optimizer.zero_grad(set_to_none=True)
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in head.named_parameters()
        }
        records = []
        for step in range(1, 9):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features, mask)
            loss = balanced_component_bce(logits, label_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError("recovery probe loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            for name, parameter in head.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("recovery probe gradient failure: {}".format(name))
            optimizer.step()
            for name, parameter in head.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError("recovery probe parameter failure: {}".format(name))
            records.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "logit_mean": float(logits.detach().mean()),
                    "positive_component_count": int(np.sum(targets > 0.5)),
                    "negative_component_count": int(np.sum(targets < 0.5)),
                }
            )
        updates = {
            name: float((parameter.detach().cpu() - before[name]).abs().sum())
            for name, parameter in head.named_parameters()
        }
        if not all(value > 0 for value in updates.values()):
            raise RuntimeError("not every recovery-head parameter tensor updated")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        if peak_mib > 2.0 * 1024.0:
            raise RuntimeError("recovery-head mechanical probe exceeded 2GiB")
        del head, optimizer, features, mask, label_tensor, logits, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)
    payload = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-real-batch-eight-step-probe-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "feature_manifest_sha256": feature_sha,
        "source_name_for_probe_provenance_only": source_name,
        "source_is_G3_design_fit": True,
        "G2_recovery_truth_or_metrics_read": False,
        "Stage1_training_used_G2_truth_as_outer_fit": True,
        "G1_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
        "real_component_count": len(cache["components"]),
        "real_temporal_node_count": int(
            sum(len(value) for value in node_cache["node_features"])
        ),
        "positive_component_count": int(np.sum(targets > 0.5)),
        "negative_component_count": int(np.sum(targets < 0.5)),
        "FP32_no_update_gradient_audit": {
            "loss": float(audit_loss.detach().cpu()),
            "global_gradient_l1": sum(gradient_l1.values()),
            "all_parameter_gradients_finite": True,
            "optimizer_update": False,
        },
        "optimizer_steps": 8,
        "all_step_diagnostics": records,
        "all_parameter_tensors_updated": True,
        "parameter_update_l1": updates,
        "peak_CUDA_MiB": peak_mib,
        "peak_CUDA_budget_MiB": 2048.0,
        "CUDA_after_release_MiB": after_mib,
        "mechanical_passed": True,
        "formal_inner_started": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = write_json_exclusive(PROBE_RECEIPT, payload)
    print(
        json.dumps(
            {
                "stage": "recovery_head_mechanical_probe_complete",
                "receipt_sha256": digest,
                "mechanical_passed": True,
                "peak_CUDA_MiB": peak_mib,
                "loss_first": records[0]["loss"],
                "loss_last": records[-1]["loss"],
                "G1_arrays_or_predictions_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def probe_gate():
    feature_payload, feature_sha = feature_manifest_gate()
    probe_sha = verify_sidecar(PROBE_RECEIPT)
    payload = read_json(PROBE_RECEIPT)
    if payload.get("mechanical_passed") is not True:
        raise RuntimeError("recovery-head mechanical probe did not pass")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("mechanical probe protocol binding changed")
    if payload.get("formal_inner_started") is not False:
        raise RuntimeError("mechanical receipt says formal inner already started")
    if payload.get("feature_manifest_sha256") != feature_sha:
        raise RuntimeError("mechanical probe feature binding changed")
    if payload.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V2 runner changed after mechanical probe")
    if payload.get("G1_arrays_or_predictions_read") is not False:
        raise RuntimeError("mechanical probe does not prove G1 unread")
    if payload.get("G2_recovery_truth_or_metrics_read") is not False:
        raise RuntimeError("mechanical probe unexpectedly opened fresh G2 truth")
    if payload.get("validation_or_test_read") is not False:
        raise RuntimeError("mechanical probe does not prove val/test unread")
    return payload, probe_sha, feature_payload, feature_sha


def component_target_bundle(source_names, manifest):
    output = {}
    for source_name in source_names:
        cache = load_cache(source_name)
        labels, target_ids = load_truth_for_cache(source_name, cache, manifest)
        output[source_name] = component_labels(cache, labels)
        del cache, labels, target_ids
    return output


def train_head(source_names, component_targets, device, *, seed=RECOVERY_SEED):
    seed_all(seed)
    head = H2PyramidComponentRecoveryHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.0001)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in head.named_parameters()
    }
    records = []
    step = 0
    for epoch in range(1, RECOVERY_EPOCHS + 1):
        for source_name in source_names:
            step += 1
            node_cache = load_node_cache(source_name)
            features, mask = padded_component_batch(node_cache, device)
            labels = torch.from_numpy(component_targets[source_name]).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            logits = head(features, mask)
            loss = balanced_component_bce(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("formal recovery-head loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            for name, parameter in head.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("formal recovery gradient failure: {}".format(name))
            optimizer.step()
            for name, parameter in head.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError("formal recovery parameter failure: {}".format(name))
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
    if not all(value > 0 for value in updates.values()):
        raise RuntimeError("not every formal recovery-head parameter tensor updated")
    state = {
        name: value.detach().cpu().clone()
        for name, value in head.state_dict().items()
    }
    del optimizer, head
    return state, records, updates


def predict_head_from_node_cache(state, node_cache, device):
    head = H2PyramidComponentRecoveryHead().to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    with torch.no_grad():
        features, mask = padded_component_batch(node_cache, device)
        output = torch.sigmoid(head(features, mask)).cpu().numpy().astype(np.float64)
        del features, mask
    del head
    return output


def predict_head(state, source_names, device):
    output = {}
    for source_name in source_names:
        node_cache = load_node_cache(source_name)
        output[source_name] = predict_head_from_node_cache(state, node_cache, device)
        del node_cache
    return output


def challenge_report(counts):
    metrics = crossfit.metrics_from_counts(counts)
    return {
        "Score": float(metrics["score"]),
        "IoU": float(metrics["iou"]),
        "Pd": float(metrics["pd"]),
        "Fa": float(metrics["fa"]),
        "TP": int(counts.true_positive_events),
        "FP": int(counts.false_positive_events),
        "CO": int(counts.correct_objects),
        "FC": int(counts.false_components),
    }


def report_delta(reference, candidate):
    return {
        key: (
            float(candidate[key] - reference[key])
            if key in {"Score", "IoU", "Pd", "Fa"}
            else int(candidate[key] - reference[key])
        )
        for key in reference
    }


def counts_for_scores(cache, truth, scores):
    return crossfit.sufficient_counts_for_video(
        scores,
        truth["labels"],
        truth["target_ids"],
        cache["locations4"],
        THRESHOLD,
    )


def evaluate_probability_actions(source_names, probabilities, cutoff, manifest):
    pooled_m20 = crossfit.SufficientCounts()
    pooled_stage1 = crossfit.SufficientCounts()
    pooled_stage2 = crossfit.SufficientCounts()
    source_records = []
    for source_name in source_names:
        cache = load_cache(source_name)
        labels, target_ids = load_truth_for_cache(source_name, cache, manifest)
        truth = {"labels": labels, "target_ids": target_ids}
        decisions = np.asarray(probabilities[source_name] >= float(cutoff), dtype=np.bool_)
        stage2_scores = restore_whole_components_bitwise(
            cache["stage1_post"],
            cache["base_post"],
            cache["components"],
            decisions,
        )
        m20_counts = counts_for_scores(cache, truth, cache["base_post"])
        stage1_counts = counts_for_scores(cache, truth, cache["stage1_post"])
        stage2_counts = counts_for_scores(cache, truth, stage2_scores)
        assert_paired_count_invariants(m20_counts, stage1_counts, stage2_counts)
        pooled_m20 = pooled_m20 + m20_counts
        pooled_stage1 = pooled_stage1 + stage1_counts
        pooled_stage2 = pooled_stage2 + stage2_counts
        source_records.append(
            {
                "source_name": source_name,
                "component_count": len(cache["components"]),
                "restored_component_count": int(np.sum(decisions)),
                "M20": challenge_report(m20_counts),
                "Stage1": challenge_report(stage1_counts),
                "Stage2": challenge_report(stage2_counts),
            }
        )
        del cache, labels, target_ids, truth, decisions, stage2_scores
    m20 = challenge_report(pooled_m20)
    stage1 = challenge_report(pooled_stage1)
    stage2 = challenge_report(pooled_stage2)
    return {
        "cutoff": float(cutoff),
        "source_records": source_records,
        "M20": m20,
        "Stage1": stage1,
        "Stage2": stage2,
        "Stage1_delta_vs_M20": report_delta(m20, stage1),
        "Stage2_delta_vs_M20": report_delta(m20, stage2),
        "Stage2_recovery_vs_Stage1": report_delta(stage1, stage2),
    }


def select_exact_cutoff(source_names, probabilities, manifest):
    joined = np.concatenate([np.asarray(probabilities[name], dtype=np.float64) for name in source_names])
    if not np.isfinite(joined).all() or joined.size == 0:
        raise RuntimeError("invalid fit-only component probabilities")
    unique = np.unique(joined)
    identity = float(np.nextafter(unique.max(), np.inf))
    cutoffs = np.concatenate(([identity], unique[::-1]))
    pooled_m20 = crossfit.SufficientCounts()
    pooled_stage1 = crossfit.SufficientCounts()
    pooled_stage2 = [crossfit.SufficientCounts() for _ in cutoffs]
    per_source = {}
    for source_name in source_names:
        cache = load_cache(source_name)
        labels, target_ids = load_truth_for_cache(source_name, cache, manifest)
        truth = {"labels": labels, "target_ids": target_ids}
        m20_counts = counts_for_scores(cache, truth, cache["base_post"])
        stage1_counts = counts_for_scores(cache, truth, cache["stage1_post"])
        source_stage2 = []
        source_probabilities = np.asarray(
            probabilities[source_name], dtype=np.float64
        )
        if source_probabilities.size != len(cache["components"]):
            raise RuntimeError("component probability/cache count mismatch")
        for cutoff_index, cutoff in enumerate(cutoffs):
            decisions = np.asarray(
                source_probabilities >= float(cutoff), dtype=np.bool_
            )
            if np.any(decisions):
                stage2_scores = restore_whole_components_bitwise(
                    cache["stage1_post"],
                    cache["base_post"],
                    cache["components"],
                    decisions,
                )
                stage2_counts = counts_for_scores(cache, truth, stage2_scores)
                del stage2_scores
            else:
                stage2_counts = stage1_counts
            assert_paired_count_invariants(m20_counts, stage1_counts, stage2_counts)
            pooled_stage2[cutoff_index] = (
                pooled_stage2[cutoff_index] + stage2_counts
            )
            source_stage2.append(stage2_counts)
        pooled_m20 = pooled_m20 + m20_counts
        pooled_stage1 = pooled_stage1 + stage1_counts
        per_source[source_name] = {
            "M20": m20_counts,
            "Stage1": stage1_counts,
            "Stage2": source_stage2,
        }
        del cache, labels, target_ids, truth, source_stage2
    assert_paired_count_invariants(
        pooled_m20, pooled_stage1, *pooled_stage2
    )
    records = []
    feasible = []
    pooled_m20_report = challenge_report(pooled_m20)
    pooled_stage1_report = challenge_report(pooled_stage1)
    for cutoff_index, cutoff in enumerate(cutoffs):
        pooled_stage2_report = challenge_report(pooled_stage2[cutoff_index])
        source_records = []
        for source_name in source_names:
            stored = per_source[source_name]
            source_records.append(
                {
                    "source_name": source_name,
                    "component_count": int(
                        np.asarray(probabilities[source_name]).size
                    ),
                    "restored_component_count": int(
                        np.sum(probabilities[source_name] >= cutoff)
                    ),
                    "M20": challenge_report(stored["M20"]),
                    "Stage1": challenge_report(stored["Stage1"]),
                    "Stage2": challenge_report(stored["Stage2"][cutoff_index]),
                }
            )
        result = {
            "cutoff": float(cutoff),
            "source_records": source_records,
            "M20": pooled_m20_report,
            "Stage1": pooled_stage1_report,
            "Stage2": pooled_stage2_report,
            "Stage1_delta_vs_M20": report_delta(
                pooled_m20_report, pooled_stage1_report
            ),
            "Stage2_delta_vs_M20": report_delta(
                pooled_m20_report, pooled_stage2_report
            ),
            "Stage2_recovery_vs_Stage1": report_delta(
                pooled_stage1_report, pooled_stage2_report
            ),
        }
        recovery = result["Stage2_recovery_vs_Stage1"]
        is_feasible = bool(
            recovery["TP"] >= 0
            and recovery["CO"] >= 0
            and (recovery["TP"] > 0 or recovery["CO"] > 0)
        )
        summary = {
            "cutoff": float(cutoff),
            "restored_component_count": int(
                sum(np.sum(probabilities[name] >= cutoff) for name in source_names)
            ),
            "Stage2_Score": result["Stage2"]["Score"],
            "Score_gain_vs_M20": result["Stage2_delta_vs_M20"]["Score"],
            "TP_recovery_vs_Stage1": recovery["TP"],
            "CO_recovery_vs_Stage1": recovery["CO"],
            "feasible": is_feasible,
        }
        records.append(summary)
        if is_feasible:
            feasible.append((summary, result))
    if not feasible:
        return None, records
    selected_summary, selected_result = max(
        feasible,
        key=lambda pair: (
            pair[0]["Stage2_Score"],
            pair[0]["CO_recovery_vs_Stage1"],
            pair[0]["TP_recovery_vs_Stage1"],
            pair[0]["cutoff"],
        ),
    )
    return {"summary": selected_summary, "evaluation": selected_result}, records


def lso_predictions(group, component_targets, device):
    output = {}
    model_records = []
    for held_name in group:
        fit_names = tuple(name for name in group if name != held_name)
        state, training_records, updates = train_head(
            fit_names, component_targets, device
        )
        output.update(predict_head(state, (held_name,), device))
        model_records.append(
            {
                "held_source": held_name,
                "fit_sources": list(fit_names),
                "optimizer_steps": len(training_records),
                "loss_first": training_records[0]["loss"],
                "loss_last": training_records[-1]["loss"],
                "all_parameter_tensors_updated": all(value > 0 for value in updates.values()),
            }
        )
    return output, model_records


def state_sha256(state):
    return atomic.state_sha256(state)


def run_inner(args):
    require_gpu(args)
    load_protocol()
    probe_payload, probe_sha, _, feature_sha = probe_gate()
    if INNER_ROOT.exists():
        raise FileExistsError("refusing to overwrite V2 nested recovery")
    if OUTER_ROOT.exists():
        raise RuntimeError("outer G1 output exists before V2 inner gate")
    manifest = source_manifest()
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_nested_heads"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # G3 is already design-exposed.  Fit and freeze its complete inner
        # route before opening G2 recovery labels or metrics.
        targets_g3 = component_target_bundle(G3, manifest)
        g3_lso_probabilities, g3_lso_models = lso_predictions(
            G3, targets_g3, device
        )
        g3_cutoff, g3_breakpoints = select_exact_cutoff(
            G3, g3_lso_probabilities, manifest
        )
        if g3_cutoff is None:
            payload = {
                "schema": "ev-uav-h2-pyramid-recovery-v2-inner-result-v1",
                "created_utc": utc_now(),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__)),
                "probe_receipt_sha256": probe_sha,
                "feature_manifest_sha256": feature_sha,
                "failure_stage": "G3_fit_only_cutoff_has_no_risk_feasible_breakpoint",
                "G3_breakpoints": g3_breakpoints,
                "fresh_G2_recovery_truth_or_metrics_read": False,
                "Stage1_training_used_G2_truth_as_outer_fit": True,
                "G1_arrays_or_predictions_read": False,
                "validation_or_test_read": False,
                "inner_passed": False,
            }
            result_sha = write_json_exclusive(INNER_RESULT, payload)
            decision_sha = write_json_exclusive(
                INNER_DECISION,
                {
                    "schema": "ev-uav-h2-pyramid-recovery-v2-inner-decision-v1",
                    "created_utc": utc_now(),
                    "inner_result_sha256": result_sha,
                    "decision": "archive_without_G2_or_G1_open_or_tuning",
                    "G1_arrays_or_predictions_read": False,
                    "validation_or_test_read": False,
                },
            )
            print(json.dumps({"inner_passed": False, "decision_sha256": decision_sha}, indent=2))
            return
        g3_state, g3_training_records, g3_updates = train_head(
            G3, targets_g3, device
        )
        g3_checkpoint = {
            "schema": "ev-uav-h2-pyramid-recovery-v2-fit-G3-head-v1",
            "created_utc": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "fit_sources": list(G3),
            "held_G2_recovery_truth_or_metrics_read": False,
            "held_G1_arrays_or_predictions_read": False,
            "validation_or_test_read": False,
            "fit_only_cutoff": g3_cutoff["summary"]["cutoff"],
            "fit_only_cutoff_record": g3_cutoff,
            "optimizer_steps": len(g3_training_records),
            "head_state_sha256": state_sha256(g3_state),
            "head_state_dict": g3_state,
        }
        g3_checkpoint_sha = write_torch_exclusive(G3_HEAD_CHECKPOINT, g3_checkpoint)
        if verify_sidecar(G3_HEAD_CHECKPOINT) != g3_checkpoint_sha:
            raise RuntimeError("G3 recovery-head checkpoint verification failed")
        g3_to_g2_probabilities = predict_head(g3_state, G2, device)

        # First recovery-specific truth/metric access to fresh G2 occurs only
        # after the G3 model and its fit-only cutoff have been committed.
        targets_g2 = component_target_bundle(G2, manifest)
        inner_fresh_g2 = evaluate_probability_actions(
            G2,
            g3_to_g2_probabilities,
            g3_cutoff["summary"]["cutoff"],
            manifest,
        )

        g2_lso_probabilities, g2_lso_models = lso_predictions(
            G2, targets_g2, device
        )
        g2_cutoff, g2_breakpoints = select_exact_cutoff(
            G2, g2_lso_probabilities, manifest
        )
        if g2_cutoff is None:
            g2_state = None
            g2_training_records = []
            g2_updates = {}
            inner_design_g3 = None
            mutual_cutoff = None
            mutual_breakpoints = []
            pooled_mutual = None
        else:
            g2_state, g2_training_records, g2_updates = train_head(
                G2, targets_g2, device
            )
            g2_to_g3_probabilities = predict_head(g2_state, G3, device)
            inner_design_g3 = evaluate_probability_actions(
                G3,
                g2_to_g3_probabilities,
                g2_cutoff["summary"]["cutoff"],
                manifest,
            )
            mutual_probabilities = {
                **g3_to_g2_probabilities,
                **g2_to_g3_probabilities,
            }
            mutual_cutoff, mutual_breakpoints = select_exact_cutoff(
                OUTER_FIT, mutual_probabilities, manifest
            )
            pooled_mutual = (
                None if mutual_cutoff is None else mutual_cutoff["evaluation"]
            )

        gates = {
            "G3_fit_cutoff_exists": g3_cutoff is not None,
            "G2_fit_cutoff_exists": g2_cutoff is not None,
            "fresh_G2_inner_Stage2_score_gain_vs_M20_positive": (
                inner_fresh_g2["Stage2_delta_vs_M20"]["Score"] > 0.0
            ),
            "design_G3_inner_Stage2_score_gain_vs_M20_positive": bool(
                inner_design_g3 is not None
                and inner_design_g3["Stage2_delta_vs_M20"]["Score"] > 0.0
            ),
            "mutual_pooled_cutoff_exists": mutual_cutoff is not None,
            "mutual_pooled_Stage2_score_gain_vs_M20_at_least_0_02": bool(
                pooled_mutual is not None
                and pooled_mutual["Stage2_delta_vs_M20"]["Score"] >= 0.02
            ),
            "mutual_pooled_Stage2_TP_not_below_Stage1": bool(
                pooled_mutual is not None
                and pooled_mutual["Stage2_recovery_vs_Stage1"]["TP"] >= 0
            ),
            "mutual_pooled_Stage2_CO_not_below_Stage1": bool(
                pooled_mutual is not None
                and pooled_mutual["Stage2_recovery_vs_Stage1"]["CO"] >= 0
            ),
            "mutual_pooled_strictly_recovers_Stage1_TP_or_CO": bool(
                pooled_mutual is not None
                and (
                    pooled_mutual["Stage2_recovery_vs_Stage1"]["TP"] > 0
                    or pooled_mutual["Stage2_recovery_vs_Stage1"]["CO"] > 0
                )
            ),
        }
        inner_passed = all(gates.values())
        final_state = None
        final_records = []
        final_updates = {}
        final_checkpoint_sha = None
        if inner_passed:
            all_targets = {**targets_g2, **targets_g3}
            final_state, final_records, final_updates = train_head(
                OUTER_FIT, all_targets, device
            )
            if len(final_records) != RECOVERY_FULL_STEPS:
                raise RuntimeError("final recovery-head optimizer-step count mismatch")
            final_checkpoint = {
                "schema": "ev-uav-h2-pyramid-recovery-v2-final-head-checkpoint-v1",
                "created_utc": utc_now(),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": sha256_file(Path(__file__)),
                "fit_sources": list(OUTER_FIT),
                "held_G1_arrays_or_predictions_read": False,
                "validation_or_test_read": False,
                "fresh_initialization": True,
                "optimizer_steps": len(final_records),
                "OOF_cutoff": mutual_cutoff["summary"]["cutoff"],
                "OOF_cutoff_record": mutual_cutoff,
                "head_state_sha256": state_sha256(final_state),
                "head_state_dict": final_state,
            }
            final_checkpoint_sha = write_torch_exclusive(
                FINAL_HEAD_CHECKPOINT, final_checkpoint
            )
            reloaded = torch.load(
                FINAL_HEAD_CHECKPOINT, map_location="cpu", weights_only=False
            )
            if state_sha256(reloaded["head_state_dict"]) != reloaded[
                "head_state_sha256"
            ]:
                raise RuntimeError("final recovery-head state verification failed")
            if verify_sidecar(FINAL_HEAD_CHECKPOINT) != final_checkpoint_sha:
                raise RuntimeError("final recovery-head checkpoint SHA verification failed")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)

    payload = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-inner-result-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "probe_receipt_sha256": probe_sha,
        "feature_manifest_sha256": feature_sha,
        "G3_design_exposure_disclosure": True,
        "G3_fit_checkpoint_sha256_before_G2_recovery_truth_metrics": g3_checkpoint_sha,
        "G3_fit_source_LOO_models": g3_lso_models,
        "G3_fit_cutoff": g3_cutoff,
        "G3_fit_breakpoints": g3_breakpoints,
        "G3_full_training": {
            "optimizer_steps": len(g3_training_records),
            "loss_first": g3_training_records[0]["loss"],
            "loss_last": g3_training_records[-1]["loss"],
            "all_parameter_tensors_updated": all(value > 0 for value in g3_updates.values()),
        },
        "fresh_G2_inner_evaluation": inner_fresh_g2,
        "G2_fit_source_LOO_models": g2_lso_models,
        "G2_fit_cutoff": g2_cutoff,
        "G2_fit_breakpoints": g2_breakpoints,
        "G2_full_training": {
            "optimizer_steps": len(g2_training_records),
            "loss_first": None if not g2_training_records else g2_training_records[0]["loss"],
            "loss_last": None if not g2_training_records else g2_training_records[-1]["loss"],
            "all_parameter_tensors_updated": bool(g2_updates)
            and all(value > 0 for value in g2_updates.values()),
        },
        "design_G3_inner_evaluation": inner_design_g3,
        "mutual_OOF_cutoff": mutual_cutoff,
        "mutual_OOF_breakpoints": mutual_breakpoints,
        "mutual_OOF_evaluation": pooled_mutual,
        "gates": gates,
        "inner_passed": inner_passed,
        "final_head_checkpoint_path": (
            str(FINAL_HEAD_CHECKPOINT.resolve()) if inner_passed else None
        ),
        "final_head_checkpoint_sha256": final_checkpoint_sha,
        "final_head_training": {
            "optimizer_steps": len(final_records),
            "loss_first": None if not final_records else final_records[0]["loss"],
            "loss_last": None if not final_records else final_records[-1]["loss"],
            "all_parameter_tensors_updated": bool(final_updates)
            and all(value > 0 for value in final_updates.values()),
        },
        "G1_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
        "peak_CUDA_MiB": peak_mib,
        "CUDA_after_release_MiB": after_mib,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_sha = write_json_exclusive(INNER_RESULT, payload)
    decision = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-inner-decision-v1",
        "created_utc": utc_now(),
        "inner_result_sha256": result_sha,
        "gates": gates,
        "inner_passed": inner_passed,
        "decision": (
            "eligible_for_unique_fresh_G1_evaluation_but_not_started"
            if inner_passed
            else "archive_without_G1_open_or_tuning"
        ),
        "G1_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
    }
    decision_sha = write_json_exclusive(INNER_DECISION, decision)
    print(
        json.dumps(
            {
                "stage": "nested_recovery_complete",
                "inner_result_sha256": result_sha,
                "inner_decision_sha256": decision_sha,
                "fresh_G2_inner": inner_fresh_g2,
                "design_G3_inner": inner_design_g3,
                "mutual_OOF": pooled_mutual,
                "gates": gates,
                "inner_passed": inner_passed,
                "final_head_checkpoint_sha256": final_checkpoint_sha,
                "peak_CUDA_MiB": peak_mib,
                "G1_arrays_or_predictions_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def inner_gate():
    probe_gate()
    result_sha = verify_sidecar(INNER_RESULT)
    decision_sha = verify_sidecar(INNER_DECISION)
    checkpoint_sha = verify_sidecar(FINAL_HEAD_CHECKPOINT)
    result = read_json(INNER_RESULT)
    decision = read_json(INNER_DECISION)
    checkpoint = torch.load(
        FINAL_HEAD_CHECKPOINT, map_location="cpu", weights_only=False
    )
    runner_sha = sha256_file(Path(__file__))
    if result.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("inner result protocol binding changed")
    if result.get("runner_sha256") != runner_sha:
        raise RuntimeError("V2 runner changed after nested recovery")
    if result.get("inner_passed") is not True or not all(
        result.get("gates", {}).values()
    ):
        raise RuntimeError("nested recovery did not pass every frozen inner gate")
    if result.get("G1_arrays_or_predictions_read") is not False:
        raise RuntimeError("inner result does not prove G1 unread")
    if result.get("validation_or_test_read") is not False:
        raise RuntimeError("inner result does not prove validation/test unread")
    if result.get("final_head_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("inner result final-head binding changed")
    if decision.get("inner_result_sha256") != result_sha:
        raise RuntimeError("inner decision result binding changed")
    if decision.get("inner_passed") is not True or not all(
        decision.get("gates", {}).values()
    ):
        raise RuntimeError("inner decision is not eligible for G1")
    if decision.get("decision") != (
        "eligible_for_unique_fresh_G1_evaluation_but_not_started"
    ):
        raise RuntimeError("inner decision does not authorize G1 evaluation")
    if decision.get("G1_arrays_or_predictions_read") is not False:
        raise RuntimeError("inner decision does not prove G1 unread")
    if decision.get("validation_or_test_read") is not False:
        raise RuntimeError("inner decision does not prove validation/test unread")
    if checkpoint.get("schema") != (
        "ev-uav-h2-pyramid-recovery-v2-final-head-checkpoint-v1"
    ):
        raise RuntimeError("unexpected final recovery-head checkpoint")
    if checkpoint.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("final recovery-head protocol binding changed")
    if checkpoint.get("runner_sha256") != runner_sha:
        raise RuntimeError("final recovery-head runner binding changed")
    if tuple(checkpoint.get("fit_sources", ())) != OUTER_FIT:
        raise RuntimeError("final recovery-head fit set changed")
    if checkpoint.get("held_G1_arrays_or_predictions_read") is not False:
        raise RuntimeError("final recovery-head does not prove G1 unread")
    if checkpoint.get("validation_or_test_read") is not False:
        raise RuntimeError("final recovery-head does not prove validation/test unread")
    if checkpoint.get("fresh_initialization") is not True:
        raise RuntimeError("final recovery head is not fresh")
    if checkpoint.get("optimizer_steps") != RECOVERY_FULL_STEPS:
        raise RuntimeError("final recovery-head step count changed")
    if state_sha256(checkpoint["head_state_dict"]) != checkpoint[
        "head_state_sha256"
    ]:
        raise RuntimeError("final recovery-head state hash changed")
    cutoff = float(checkpoint["OOF_cutoff"])
    if not np.isfinite(cutoff):
        raise RuntimeError("final recovery cutoff is non-finite")
    result_cutoff = float(result["mutual_OOF_cutoff"]["summary"]["cutoff"])
    if cutoff != result_cutoff:
        raise RuntimeError("final recovery cutoff/result mismatch")
    return (
        result,
        decision,
        checkpoint,
        result_sha,
        decision_sha,
        checkpoint_sha,
    )


def assert_atomic_overlay(cache, stage2_scores, decisions):
    stage2_scores = np.asarray(stage2_scores, dtype=np.float32)
    stage1_scores = np.asarray(cache["stage1_post"], dtype=np.float32)
    base_scores = np.asarray(cache["base_post"], dtype=np.float32)
    covered = np.zeros(stage2_scores.size, dtype=np.bool_)
    for component, restore in zip(cache["components"], decisions):
        indices = np.asarray(component, dtype=np.int64)
        if np.any(covered[indices]):
            raise RuntimeError("held component overlay is not disjoint")
        covered[indices] = True
        expected = base_scores[indices] if bool(restore) else stage1_scores[indices]
        if not np.array_equal(stage2_scores[indices], expected):
            raise RuntimeError("held component overlay is not bitwise atomic")
    if not np.array_equal(stage2_scores[~covered], stage1_scores[~covered]):
        raise RuntimeError("held Stage2 changed scores outside action components")


def held_artifact_arrays(cache, probabilities, decisions, stage2_scores, cutoff):
    components = tuple(np.asarray(value, dtype=np.int64) for value in cache["components"])
    lengths = np.asarray([value.size for value in components], dtype=np.int64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
    )
    joined = (
        np.concatenate(components).astype(np.int64, copy=False)
        if components
        else np.zeros(0, dtype=np.int64)
    )
    return {
        "artifact_schema_version": np.asarray([1], dtype=np.int64),
        "event_count": np.asarray([cache["event_count"]], dtype=np.int64),
        "event_index": np.arange(cache["event_count"], dtype=np.int64),
        "locations4": np.asarray(cache["locations4"], dtype=np.int64),
        "M20_raw_scores": np.asarray(cache["base_raw"], dtype=np.float32),
        "Stage1_raw_scores": np.asarray(cache["stage1_raw"], dtype=np.float32),
        "M20_post_C00_scores": np.asarray(cache["base_post"], dtype=np.float32),
        "Stage1_post_C00_scores": np.asarray(cache["stage1_post"], dtype=np.float32),
        "Stage2_post_atomic_scores": np.asarray(stage2_scores, dtype=np.float32),
        "component_offsets": offsets,
        "component_event_indices": joined,
        "component_recovery_probability": np.asarray(
            probabilities, dtype=np.float64
        ),
        "component_recovery_decision": np.asarray(decisions, dtype=np.bool_),
        "OOF_cutoff": np.asarray([cutoff], dtype=np.float64),
    }


def assert_paired_count_invariants(*counts):
    if len(counts) < 2:
        raise ValueError("paired count audit needs at least two arms")
    reference = counts[0]
    for candidate in counts[1:]:
        for field in ("event_count", "frame_count", "object_count"):
            if getattr(candidate, field) != getattr(reference, field):
                raise RuntimeError("paired count invariant changed: {}".format(field))
        reference_positive = (
            reference.true_positive_events + reference.false_negative_events
        )
        candidate_positive = (
            candidate.true_positive_events + candidate.false_negative_events
        )
        if candidate_positive != reference_positive:
            raise RuntimeError("paired truth-positive count changed")


def evaluate_g1(args):
    require_gpu(args)
    load_protocol()
    (
        inner_result,
        _,
        head_checkpoint,
        inner_result_sha,
        inner_decision_sha,
        head_checkpoint_sha,
    ) = inner_gate()
    stage1_checkpoint, _, stage1_checkpoint_sha, _ = stage1_checkpoint_gate()
    if OUTER_ROOT.exists():
        raise FileExistsError("refusing to overwrite unique fresh G1 evaluation")
    manifest = source_manifest()
    cfg, effective_c00 = build_c00()
    cutoff = float(head_checkpoint["OOF_cutoff"])
    started = time.perf_counter()
    artifact_records = []

    # All held predictions and input-only action artifacts are committed before
    # any held label or target id is attached to the evaluation process.
    with atomic.gpu_run_lock("h2_pyramid_recovery_v2_unique_held_G1_inference"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m20, _ = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        adapter.expert.load_state_dict(
            stage1_checkpoint["expert_state_dict"], strict=True
        )
        adapter.eval()
        for source_name in G1:
            path = verify_source(source_name, manifest)
            video, polarities, locations4 = atomic._load_input_only(path)
            if len(polarities) != int(manifest[source_name]["event_count"]):
                raise RuntimeError("held source event count changed")
            if not use_h2_residual_refiner(len(polarities), polarities):
                raise RuntimeError("held source left H2 route")
            cache = build_input_only_source_cache(
                adapter, video, polarities, locations4, cfg, device
            )
            probabilities = predict_head_from_node_cache(
                head_checkpoint["head_state_dict"],
                {"node_features": cache["node_features"]},
                device,
            )
            decisions = np.asarray(probabilities >= cutoff, dtype=np.bool_)
            stage2_scores = restore_whole_components_bitwise(
                cache["stage1_post"],
                cache["base_post"],
                cache["components"],
                decisions,
            )
            assert_atomic_overlay(cache, stage2_scores, decisions)
            arrays = held_artifact_arrays(
                cache, probabilities, decisions, stage2_scores, cutoff
            )
            artifact_path = OUTER_ROOT / (
                source_name.replace(".npz", "_input_only_scores.npz")
            )
            artifact_sha = write_npz_exclusive(artifact_path, arrays)
            if verify_sidecar(artifact_path) != artifact_sha:
                raise RuntimeError("held input-only artifact SHA verification failed")
            with np.load(artifact_path, allow_pickle=False) as reloaded:
                if set(reloaded.files) != set(arrays):
                    raise RuntimeError("held input-only artifact key mismatch")
                for key, value in arrays.items():
                    if not np.array_equal(reloaded[key], value):
                        raise RuntimeError(
                            "held input-only artifact array mismatch: {}".format(key)
                        )
            artifact_records.append(
                {
                    "source_name_for_provenance_only": source_name,
                    "source_sha256": manifest[source_name]["sha256"],
                    "artifact_path": str(artifact_path.resolve()),
                    "artifact_sha256": artifact_sha,
                    "arrays": {
                        key: array_receipt(value) for key, value in arrays.items()
                    },
                    "M20_C00_stats": cache["base_C00_stats"],
                    "Stage1_C00_stats": cache["stage1_C00_stats"],
                    "M20_component_count": cache["M20_component_count"],
                    "action_component_count": len(cache["components"]),
                    "restored_component_count": int(np.sum(decisions)),
                    "contains_labels_or_target_ids": False,
                }
            )
            del (
                video,
                polarities,
                locations4,
                cache,
                probabilities,
                decisions,
                stage2_scores,
                arrays,
            )
            torch.cuda.empty_cache()
            print("V2 held input-only inference {}".format(source_name), flush=True)
        if atomic.state_sha256(m20.state_dict()) != m20_before:
            raise RuntimeError("M20 changed during held G1 inference")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        del adapter, m20
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)

    inference_manifest = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-held-G1-input-only-manifest-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "fresh_Stage1_checkpoint_sha256": stage1_checkpoint_sha,
        "recovery_head_checkpoint_sha256": head_checkpoint_sha,
        "inner_result_sha256": inner_result_sha,
        "inner_decision_sha256": inner_decision_sha,
        "OOF_cutoff": cutoff,
        "held_source_order": list(G1),
        "records": artifact_records,
        "all_predictions_committed_before_truth_attachment": True,
        "contains_labels_or_target_ids": False,
        "validation_or_test_read": False,
        "effective_C00": effective_c00,
        "peak_CUDA_MiB": peak_mib,
        "CUDA_after_release_MiB": after_mib,
    }
    inference_manifest_sha = write_json_exclusive(
        OUTER_INFERENCE_MANIFEST, inference_manifest
    )

    pooled_m20 = crossfit.SufficientCounts()
    pooled_stage1 = crossfit.SufficientCounts()
    pooled_stage2 = crossfit.SufficientCounts()
    source_records = []
    for artifact_record in artifact_records:
        source_name = artifact_record["source_name_for_provenance_only"]
        artifact_path = Path(artifact_record["artifact_path"])
        if verify_sidecar(artifact_path) != artifact_record["artifact_sha256"]:
            raise RuntimeError("held artifact changed before truth scoring")
        with np.load(artifact_path, allow_pickle=False) as artifact:
            locations4 = artifact["locations4"].astype(np.int64, copy=True)
            m20_scores = artifact["M20_post_C00_scores"].astype(
                np.float32, copy=True
            )
            stage1_scores = artifact["Stage1_post_C00_scores"].astype(
                np.float32, copy=True
            )
            stage2_scores = artifact["Stage2_post_atomic_scores"].astype(
                np.float32, copy=True
            )
            event_count = int(artifact["event_count"][0])
        path = verify_source(source_name, manifest)
        labels, target_ids = atomic._load_truth(path)
        if labels.size != event_count or target_ids.size != event_count:
            raise RuntimeError("held truth/artifact event count mismatch")
        m20_counts = crossfit.sufficient_counts_for_video(
            m20_scores, labels, target_ids, locations4, THRESHOLD
        )
        stage1_counts = crossfit.sufficient_counts_for_video(
            stage1_scores, labels, target_ids, locations4, THRESHOLD
        )
        stage2_counts = crossfit.sufficient_counts_for_video(
            stage2_scores, labels, target_ids, locations4, THRESHOLD
        )
        assert_paired_count_invariants(m20_counts, stage1_counts, stage2_counts)
        pooled_m20 = pooled_m20 + m20_counts
        pooled_stage1 = pooled_stage1 + stage1_counts
        pooled_stage2 = pooled_stage2 + stage2_counts
        source_records.append(
            {
                "source_name": source_name,
                "artifact_sha256": artifact_record["artifact_sha256"],
                "action_component_count": artifact_record[
                    "action_component_count"
                ],
                "restored_component_count": artifact_record[
                    "restored_component_count"
                ],
                "M20": challenge_report(m20_counts),
                "Stage1": challenge_report(stage1_counts),
                "Stage2": challenge_report(stage2_counts),
                "Stage1_delta_vs_M20": report_delta(
                    challenge_report(m20_counts), challenge_report(stage1_counts)
                ),
                "Stage2_delta_vs_M20": report_delta(
                    challenge_report(m20_counts), challenge_report(stage2_counts)
                ),
                "Stage2_recovery_vs_Stage1": report_delta(
                    challenge_report(stage1_counts), challenge_report(stage2_counts)
                ),
            }
        )
        if sha256_file(path) != manifest[source_name]["sha256"]:
            raise RuntimeError("held source changed during paired scoring")
        del labels, target_ids, locations4, m20_scores, stage1_scores, stage2_scores
    assert_paired_count_invariants(pooled_m20, pooled_stage1, pooled_stage2)
    pooled_reports = {
        "M20": challenge_report(pooled_m20),
        "Stage1": challenge_report(pooled_stage1),
        "Stage2": challenge_report(pooled_stage2),
    }
    pooled_reports["Stage1_delta_vs_M20"] = report_delta(
        pooled_reports["M20"], pooled_reports["Stage1"]
    )
    pooled_reports["Stage2_delta_vs_M20"] = report_delta(
        pooled_reports["M20"], pooled_reports["Stage2"]
    )
    pooled_reports["Stage2_recovery_vs_Stage1"] = report_delta(
        pooled_reports["Stage1"], pooled_reports["Stage2"]
    )
    recovery = pooled_reports["Stage2_recovery_vs_Stage1"]
    gates = {
        "held_G1_Stage2_score_gain_vs_M20_at_least_0_02": (
            pooled_reports["Stage2_delta_vs_M20"]["Score"] >= 0.02
        ),
        "held_G1_Stage2_TP_not_below_Stage1": recovery["TP"] >= 0,
        "held_G1_Stage2_CO_not_below_Stage1": recovery["CO"] >= 0,
        "held_G1_Stage2_strictly_recovers_Stage1_TP_or_CO": (
            recovery["TP"] > 0 or recovery["CO"] > 0
        ),
    }
    outer_passed = all(gates.values())
    result = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-held-G1-paired-evaluation-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "fresh_Stage1_checkpoint_sha256": stage1_checkpoint_sha,
        "recovery_head_checkpoint_sha256": head_checkpoint_sha,
        "inner_result_sha256": inner_result_sha,
        "inner_gates": inner_result["gates"],
        "input_only_inference_manifest_sha256": inference_manifest_sha,
        "OOF_cutoff": cutoff,
        "source_records": source_records,
        "pooled": pooled_reports,
        "outer_gates": gates,
        "outer_passed": outer_passed,
        "single_outer_held_open_complete": True,
        "no_other_fold_threshold_or_model_tuning_started": True,
        "validation_or_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_sha = write_json_exclusive(OUTER_RESULT, result)
    decision = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-held-G1-branch-decision-v1",
        "created_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "paired_evaluation_sha256": result_sha,
        "outer_gates": gates,
        "outer_passed": outer_passed,
        "decision": (
            "promote_after_fresh_G1_but_stop_no_other_fold_or_tuning"
            if outer_passed
            else "archive_without_other_fold_threshold_or_model_tuning"
        ),
        "validation_or_test_read": False,
    }
    decision_sha = write_json_exclusive(OUTER_DECISION, decision)
    print(
        json.dumps(
            {
                "stage": "unique_fresh_G1_evaluation_complete",
                "paired_evaluation_sha256": result_sha,
                "branch_decision_sha256": decision_sha,
                "pooled": pooled_reports,
                "outer_gates": gates,
                "outer_passed": outer_passed,
                "validation_or_test_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def formal_cpu_audit(_args):
    if FORMAL_AUDIT_ROOT.exists():
        raise FileExistsError("refusing to overwrite formal-runner CPU audit")
    if any(
        path.exists()
        for path in (STAGE1_ROOT, FEATURE_ROOT, PROBE_ROOT, INNER_ROOT, OUTER_ROOT)
    ):
        raise RuntimeError("formal output exists before formal-runner CPU audit")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before CPU audit")
    protocol = load_protocol()
    manifest = source_manifest()
    if tuple(manifest) != G1 + G2 + G3:
        raise RuntimeError("frozen 11-source manifest order changed")
    head = H2PyramidComponentRecoveryHead().cpu().eval()
    if component_recovery_parameter_count(head) != 14081:
        raise RuntimeError("recovery-head parameter count changed")
    synthetic = torch.zeros((3, 5, NODE_FEATURE_DIM), dtype=torch.float32)
    mask = torch.tensor(
        [[True, True, True, False, False], [True] * 5, [True, False, False, False, False]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        logits = head(synthetic, mask)
    if logits.shape != (3,) or not torch.isfinite(logits).all():
        raise RuntimeError("recovery-head CPU forward audit failed")
    base = np.asarray([0.8, 0.9, 0.7, 0.95], dtype=np.float32)
    stage1 = np.asarray([0.1, 0.2, 0.7, 0.3], dtype=np.float32)
    components = (np.asarray([0, 1]), np.asarray([3]))
    restored = restore_whole_components_bitwise(
        stage1, base, components, np.asarray([True, False])
    )
    if not np.array_equal(restored, np.asarray([0.8, 0.9, 0.7, 0.3], dtype=np.float32)):
        raise RuntimeError("atomic recovery CPU identity audit failed")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit unexpectedly initialized CUDA")
    payload = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-formal-runner-cpu-audit-v1",
        "created_utc": utc_now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_status": protocol["status"],
        "execution_dependency_sha256": EXPECTED_EXECUTION_DEPENDENCIES,
        "source_groups": {"G1": list(G1), "G2": list(G2), "G3": list(G3)},
        "recovery_head_parameter_count": 14081,
        "synthetic_head_forward_finite": True,
        "whole_component_bitwise_identity_passed": True,
        "formal_stage_outputs_absent": True,
        "G2_or_G1_dataset_arrays_read": False,
        "old_G1_predictions_or_metrics_read": False,
        "validation_or_test_read": False,
        "CUDA_initialized": False,
        "CPU_audit_passed": True,
    }
    digest = write_json_exclusive(FORMAL_AUDIT_RECEIPT, payload)
    print(
        json.dumps(
            {
                "stage": "formal_runner_CPU_audit_complete",
                "receipt_sha256": digest,
                "runner_sha256": payload["runner_sha256"],
                "CPU_audit_passed": True,
                "CUDA_initialized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def formal_audit_gate():
    audit_sha = verify_sidecar(FORMAL_AUDIT_RECEIPT)
    audit = read_json(FORMAL_AUDIT_RECEIPT)
    if audit.get("CPU_audit_passed") is not True:
        raise RuntimeError("formal-runner CPU audit did not pass")
    if audit.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V2 runner changed after formal CPU audit")
    if audit.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("formal CPU audit protocol binding changed")
    if audit.get("G2_or_G1_dataset_arrays_read") is not False:
        raise RuntimeError("formal CPU audit does not prove G2/G1 unread")
    if audit.get("validation_or_test_read") is not False:
        raise RuntimeError("formal CPU audit does not prove validation/test unread")
    return audit, audit_sha


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frozen V2 pyramid suppression plus atomic recovery chain."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("cpu-audit")
    audit_parser.set_defaults(handler=formal_cpu_audit)
    for command, handler in (
        ("train-stage1", train_stage1),
        ("extract-fit-features", extract_fit_features),
        ("mechanical-probe", mechanical_probe),
        ("run-inner", run_inner),
        ("evaluate-g1", evaluate_g1),
    ):
        child = commands.add_parser(command)
        child.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
        child.set_defaults(handler=handler)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

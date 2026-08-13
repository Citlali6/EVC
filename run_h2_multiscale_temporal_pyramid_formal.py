"""Frozen hold-G3 formal run for the H2 multi-scale temporal pyramid expert.

The train and held-evaluation commands are intentionally separate processes.
Training can open only fit G1+G2 arrays.  Held evaluation refuses to start
until the final checkpoint, its SHA sidecar, and the fit-only training receipt
all exist and agree.
"""

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

import crossfit_component_reranker as crossfit
import run_h2_atomic_component_deletion_v3 as atomic
import run_h2_multiscale_temporal_pyramid_probe as probe
from model.h2_multiscale_temporal_pyramid_expert import (
    FrozenM20MultiScalePyramidAdapter,
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
TRAIN_DATA_ROOT = (
    WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train"
).resolve()
SCIENCE_PATH = ROOT / "protocols" / "h2_multiscale_temporal_pyramid_expert_science_v1.json"
EXECUTION_PATH = (
    ROOT / "protocols" / "h2_multiscale_temporal_pyramid_expert_formal_execution_v1.json"
)
SOURCE_MANIFEST_PROTOCOL_PATH = (
    ROOT / "protocols" / "h2_spatiotemporal_residual_refiner_oof_science_v1.json"
)
OUTPUT_ROOT = (
    WORKSPACE / "experiments" / "20260811_h2_multiscale_temporal_pyramid_expert_v1"
)
TRAIN_OUTPUT_ROOT = OUTPUT_ROOT / "formal_training" / "hold_g3"
CHECKPOINT_PATH = TRAIN_OUTPUT_ROOT / "final_expert.pt"
CHECKPOINT_SIDECAR_PATH = TRAIN_OUTPUT_ROOT / "final_expert.pt.sha256"
TRAINING_RECEIPT_PATH = TRAIN_OUTPUT_ROOT / "training_result.json"
TRAINING_RECEIPT_SIDECAR_PATH = TRAIN_OUTPUT_ROOT / "training_result.json.sha256"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation" / "hold_g3"
EVALUATION_PATH = EVALUATION_ROOT / "paired_evaluation.json"
DECISION_PATH = EVALUATION_ROOT / "branch_decision.json"
CPU_AUDIT_PATH = OUTPUT_ROOT / "formal_cpu_audit" / "report.json"

EXPECTED_SCIENCE_SHA256 = "0bdb6e0657483e253b363462ffad6969dcd85df52ef5707d32ed93a914268155"
EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256 = (
    "7edec461f2ccc8047156f08c57389319a5defd59d0afcea69cbfcf32e81d2207"
)
EXPECTED_MODEL_SHA256 = "4d4ea4a365be49ad1b6c7cf1c7c96c2369caf3e12841bbcd781cf109105a6a98"
EXPECTED_LOSS_SHA256 = "f74e145b04b25f2e7478f5c8fd370bc4e9d96123ef6f75ea5833acc210d2c5e9"
EXPECTED_TEST_SHA256 = "0043addebd42f22735398223260c79c2df4eff4f16629c0fbef01beb9e4cb2fd"
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EXPECTED_PROBE_SHA256 = "488073e916be00360f31ccb1ae8cf52ed3a415091043ad23518b8ca9f893e4eb"
EXPECTED_PROBE_RUNNER_SHA256 = "2cd894d046a1433be7b8fc06b95bb616ed82b53ce2078aac082d08ba80eb1df4"
EXPECTED_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"

FIT_SOURCES = tuple("train_{:03d}.npz".format(index) for index in range(88, 95))
HELD_SOURCES = tuple("train_{:03d}.npz".format(index) for index in range(95, 99))
TEMPORAL_COUNT = 160
VIEW_BINS = 16
INFERENCE_BATCH = 8
SEED = 67
EPOCHS = 2
VIEWS_PER_SOURCE_PER_EPOCH = 4
EXPECTED_STEPS = len(FIT_SOURCES) * EPOCHS * VIEWS_PER_SOURCE_PER_EPOCH
PREDICTION_THRESHOLD = 0.719
GPU_FLAG = "--root-authorized-gpu"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    return probe.sha256_file(Path(path))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def serialize_json(payload):
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
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


def write_json_with_sidecar_exclusive(path, payload):
    values = serialize_json(payload)
    write_bytes_exclusive(path, values)
    digest = hashlib.sha256(values).hexdigest()
    write_bytes_exclusive(
        Path(str(path) + ".sha256"),
        (digest + "  " + Path(path).name + "\n").encode("ascii"),
    )
    return digest


def write_torch_with_sidecar_exclusive(path, payload):
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


def write_npz_with_sidecar_exclusive(path, arrays):
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


def verify_sidecar(path, sidecar_path):
    actual = sha256_file(path)
    tokens = Path(sidecar_path).read_text(encoding="ascii").strip().split()
    if len(tokens) != 2 or tokens[0] != actual or tokens[1] != Path(path).name:
        raise RuntimeError("SHA sidecar mismatch: {}".format(path))
    return actual


def workspace_artifact_path(relative):
    value = (WORKSPACE / str(relative)).resolve()
    if value != WORKSPACE.resolve() and WORKSPACE.resolve() not in value.parents:
        raise RuntimeError("amendment artifact escaped the workspace")
    return value


def load_frozen_contract():
    science = probe.load_protocol()
    if sha256_file(SCIENCE_PATH) != EXPECTED_SCIENCE_SHA256:
        raise RuntimeError("science protocol SHA changed")
    contract = read_json(EXECUTION_PATH)
    if contract.get("schema") != (
        "ev-uav-h2-multiscale-temporal-pyramid-formal-execution-amendment-v1"
    ):
        raise RuntimeError("unexpected formal execution amendment schema")
    if contract.get("status") != (
        "frozen_before_formal_hold_g3_training_or_held_array_access"
    ):
        raise RuntimeError("formal execution amendment is not frozen")
    if contract["parent_science_protocol"]["sha256"] != EXPECTED_SCIENCE_SHA256:
        raise RuntimeError("execution amendment points to another science protocol")
    if tuple(contract["scope"]["fit_sources"]) != FIT_SOURCES:
        raise RuntimeError("fit source order changed")
    if tuple(contract["scope"]["held_sources"]) != HELD_SOURCES:
        raise RuntimeError("held source order changed")
    if set(FIT_SOURCES) & set(HELD_SOURCES):
        raise RuntimeError("fit/held source overlap")
    if contract["scope"]["validation_read_allowed"] is not False:
        raise RuntimeError("validation access is forbidden")
    if contract["scope"]["test_read_allowed"] is not False:
        raise RuntimeError("test access is forbidden")
    numeric = contract["numeric_recovery"]
    expected_numeric = {
        "frozen_M20_observation_precision": "FP32",
        "expert_forward_precision": "FP32",
        "event_and_constraint_loss_precision": "FP32",
        "backward_precision": "FP32",
        "optimizer": "AdamW_FP32",
        "grad_scaler_used": False,
        "optimizer_steps": EXPECTED_STEPS,
        "epochs": EPOCHS,
        "views_per_fit_source_per_epoch": VIEWS_PER_SOURCE_PER_EPOCH,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
    }
    for key, expected in expected_numeric.items():
        if numeric.get(key) != expected:
            raise RuntimeError("numeric recovery contract changed: {}".format(key))
    if science["training"]["first_fold_optimizer_steps"] != EXPECTED_STEPS:
        raise RuntimeError("science protocol optimizer-step count changed")
    if science["training"]["epochs"] != EPOCHS:
        raise RuntimeError("science protocol epoch count changed")
    if science["training"]["views_per_fit_source_per_epoch"] != (
        VIEWS_PER_SOURCE_PER_EPOCH
    ):
        raise RuntimeError("science protocol view count changed")

    fixed_paths = {
        ROOT / "run_h2_multiscale_temporal_pyramid_probe.py": EXPECTED_PROBE_RUNNER_SHA256,
        ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py": EXPECTED_MODEL_SHA256,
        ROOT / "utils" / "h2_multiscale_pyramid_loss.py": EXPECTED_LOSS_SHA256,
        ROOT / "tests" / "test_h2_multiscale_temporal_pyramid_expert.py": EXPECTED_TEST_SHA256,
        atomic.M20_PATH: EXPECTED_M20_SHA256,
        SOURCE_MANIFEST_PROTOCOL_PATH: EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256,
    }
    for path, expected in fixed_paths.items():
        if sha256_file(path) != expected:
            raise RuntimeError("frozen dependency SHA changed: {}".format(path))

    evidence = contract["immutable_probe_evidence"]
    for record in evidence["failure_receipts"]:
        path = workspace_artifact_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("probe failure evidence changed")
    successful = evidence["successful_probe"]
    successful_path = workspace_artifact_path(successful["path"])
    if sha256_file(successful_path) != EXPECTED_PROBE_SHA256:
        raise RuntimeError("successful probe receipt changed")
    successful_payload = read_json(successful_path)
    if not successful_payload.get("mechanical_passed"):
        raise RuntimeError("successful probe no longer passes")
    if successful_payload.get("formal_started") is not False:
        raise RuntimeError("probe receipt says formal already started")
    if successful_payload.get("runner_sha256") != EXPECTED_PROBE_RUNNER_SHA256:
        raise RuntimeError("successful probe runner changed")
    successful_sidecar = Path(str(successful_path) + ".sha256")
    if sha256_file(successful_sidecar) != successful["sidecar_file_sha256"]:
        raise RuntimeError("successful probe sidecar file changed")
    if verify_sidecar(successful_path, successful_sidecar) != EXPECTED_PROBE_SHA256:
        raise RuntimeError("successful probe sidecar content changed")

    inherited = read_json(SOURCE_MANIFEST_PROTOCOL_PATH)["h2_sources"]
    if contract["source_manifest"]["inherited_protocol_sha256"] != (
        EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256
    ):
        raise RuntimeError("execution amendment source-manifest binding changed")
    if inherited != contract["source_manifest"]["sources"]:
        raise RuntimeError("embedded 11-source manifest differs from inherited protocol")
    return science, contract


def build_and_validate_c00(contract):
    cfg, effective = probe.build_c00()
    if list(atomic.C00_OVERRIDES) != contract["evaluation_contract"]["C00_overrides"]:
        raise RuntimeError("C00 override list changed")
    if crossfit.sha256_json(effective) != EXPECTED_C00_SHA256:
        raise RuntimeError("effective C00 contract changed")
    if contract["evaluation_contract"]["effective_C00_sha256"] != EXPECTED_C00_SHA256:
        raise RuntimeError("amendment C00 SHA changed")
    return cfg, effective


def require_gpu_authorization(args):
    if not bool(getattr(args, "root_authorized_gpu", False)):
        raise PermissionError("formal GPU execution requires {}".format(GPU_FLAG))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def source_path_from_manifest(source_name, manifest):
    if source_name not in manifest:
        raise RuntimeError("source outside the frozen manifest")
    path = (TRAIN_DATA_ROOT / source_name).resolve()
    if path.parent != TRAIN_DATA_ROOT:
        raise RuntimeError("source path escaped official train root")
    return path


def verify_source_file(source_name, manifest):
    path = source_path_from_manifest(source_name, manifest)
    expected = manifest[source_name]
    if sha256_file(path) != expected["sha256"]:
        raise RuntimeError("source SHA changed: {}".format(source_name))
    return path, expected


def load_input_and_truth(path, expected_event_count):
    video, polarities, locations4 = atomic._load_input_only(path)
    labels, target_ids = atomic._load_truth(path)
    if len(polarities) != int(expected_event_count):
        raise RuntimeError("source event count changed")
    if labels.size != len(polarities) or target_ids.size != len(polarities):
        raise RuntimeError("source truth vectors do not align")
    if len(video.event_indices_by_bin) != TEMPORAL_COUNT:
        raise RuntimeError("formal pyramid requires complete T160")
    if not use_h2_residual_refiner(len(polarities), polarities):
        raise RuntimeError("source no longer satisfies the frozen input-only H2 route")
    return video, polarities, locations4, labels, target_ids


def extract_fit_hard_negatives(cfg, raw_scores, locations4, labels):
    processed, stats = ChallengePostprocessor.from_cfg(
        cfg, PREDICTION_THRESHOLD, event_count=len(labels)
    ).apply(
        torch.from_numpy(raw_scores.copy()),
        torch.from_numpy(locations4).long(),
    )
    c00_scores = processed.numpy().astype(np.float32, copy=True)
    components = extract_atomic_components(
        c00_scores,
        locations4,
        PREDICTION_THRESHOLD,
        spatial_radius=2,
        temporal_bin_size=50,
        temporal_radius_bins=0,
    )
    component_targets = pure_false_positive_targets(components.event_indices, labels)
    pure_fp_components = tuple(
        components.event_indices[int(index)]
        for index in np.flatnonzero(component_targets == 1)
    )
    if not pure_fp_components:
        raise RuntimeError("fit source has no pure-FP C00 component")
    return pure_fp_components, asdict(stats), len(components.event_indices)


def prepare_training_views(
    video,
    labels,
    target_ids,
    pure_fp_components,
    *,
    epoch,
    source_position,
):
    positive_target = (labels > 0) & (target_ids > 0)
    positive_bins = np.floor_divide(
        video.locations[np.flatnonzero(positive_target), 2].astype(np.int64), 50
    )
    component_bins = []
    for component in pure_fp_components:
        bins = np.unique(
            np.floor_divide(video.locations[component, 2].astype(np.int64), 50)
        )
        if bins.size != 1:
            raise RuntimeError("hard-negative training topology must be per-bin")
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
        global_indices = np.concatenate(
            [video.event_indices_by_bin[value] for value in range(start, stop)]
        )
        if not np.any(labels[global_indices] == 0):
            continue
        eligible.append(start)
        components_by_start[start] = tuple(
            pure_fp_components[int(row)] for row in rows
        )
    if not eligible:
        raise RuntimeError("fit source has no joint target/hard-negative T16 view")

    generator = np.random.default_rng(
        np.random.SeedSequence([SEED, int(epoch), int(source_position)])
    )
    orders = (
        [eligible[int(index)] for index in generator.permutation(len(eligible))],
        [eligible[int(index)] for index in generator.permutation(len(eligible))],
    )
    selected = []
    used = set()
    for view_index in range(VIEWS_PER_SOURCE_PER_EPOCH):
        purpose_index = view_index % 2
        order = orders[purpose_index]
        available = [value for value in order if value not in used]
        start = available[0] if available else order[view_index % len(order)]
        selected.append(
            (
                "target_bearing" if purpose_index == 0 else "hard_negative_component",
                start,
            )
        )
        used.add(start)

    metadata = []
    for purpose, start in selected:
        stop = start + VIEW_BINS
        global_indices = np.concatenate(
            [video.event_indices_by_bin[value] for value in range(start, stop)]
        ).astype(np.int64, copy=False)
        global_to_local = np.full(labels.size, -1, dtype=np.int64)
        global_to_local[global_indices] = np.arange(global_indices.size, dtype=np.int64)
        local_components = []
        for component in components_by_start[start]:
            local = global_to_local[component]
            if np.any(local < 0):
                raise RuntimeError("hard-negative component escaped selected view")
            local_components.append(local)
        selected_labels = labels[global_indices]
        selected_targets = target_ids[global_indices]
        selected_times = np.floor_divide(
            video.locations[global_indices, 2].astype(np.int64), 50
        )
        if not np.any((selected_labels > 0) & (selected_targets > 0)):
            raise RuntimeError("training view lacks a positive target/time group")
        if not np.any(selected_labels == 0) or not local_components:
            raise RuntimeError("training view lacks event negatives or pure-FP components")
        metadata.append(
            {
                "purpose": purpose,
                "start": int(start),
                "stop": int(stop),
                "global_indices": global_indices,
                "labels": selected_labels,
                "target_ids": selected_targets,
                "times": selected_times,
                "hard_negative_components": tuple(local_components),
            }
        )
    return metadata, len(eligible)


def constraint_trends(records):
    fields = (
        "classification_normalized",
        "target_time_recall_violation",
        "hard_negative_suppression_violation",
    )
    result = {}
    for field in fields:
        values = np.asarray([record[field] for record in records], dtype=np.float64)
        result[field] = {
            "first": float(values[0]),
            "last": float(values[-1]),
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "epoch_means": [
                float(
                    np.mean(
                        [
                            record[field]
                            for record in records
                            if int(record["epoch"]) == epoch
                        ]
                    )
                )
                for epoch in range(1, EPOCHS + 1)
            ],
        }
    return result


def train_formal(args):
    require_gpu_authorization(args)
    science, contract = load_frozen_contract()
    cfg, effective_c00 = build_and_validate_c00(contract)
    if TRAIN_OUTPUT_ROOT.exists():
        raise FileExistsError("refusing to overwrite formal hold-G3 training")
    if EVALUATION_ROOT.exists():
        raise FileExistsError("held evaluation exists before formal training")
    manifest = contract["source_manifest"]["sources"]
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_multiscale_pyramid_formal_hold_g3_training"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        m20, _ = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        if pyramid_expert_parameter_count(adapter) != 3381:
            raise RuntimeError("pyramid trainable parameter count changed")
        if any(parameter.requires_grad for parameter in m20.parameters()):
            raise RuntimeError("released M20 is not frozen")
        adapter.train()
        optimizer = torch.optim.AdamW(
            adapter.trainable_parameters(),
            lr=float(science["training"]["learning_rate"]),
            weight_decay=float(science["training"]["weight_decay"]),
        )
        dual_state = PyramidDualState()
        records = []
        source_records = []
        fit_initial_hashes = {}
        initial_identity = None
        step = 0

        for epoch_zero in range(EPOCHS):
            epoch = epoch_zero + 1
            for source_position, source_name in enumerate(FIT_SOURCES):
                path, expected = verify_source_file(source_name, manifest)
                fit_initial_hashes.setdefault(source_name, expected["sha256"])
                video, polarities, locations4, labels, target_ids = load_input_and_truth(
                    path, expected["event_count"]
                )
                memory = probe.full_stream_memory(m20, video, device)
                observations_cpu, raw_scores, first_decoder_bins = (
                    probe.stream_observations_and_scores(
                        adapter, video, memory, device
                    )
                )
                summary_cache = probe.build_summary_cache(observations_cpu, device)
                del observations_cpu
                pure_fp_components, post_stats, base_component_count = (
                    extract_fit_hard_negatives(cfg, raw_scores, locations4, labels)
                )
                views, eligible_count = prepare_training_views(
                    video,
                    labels,
                    target_ids,
                    pure_fp_components,
                    epoch=epoch_zero,
                    source_position=source_position,
                )
                source_step_start = step + 1
                for metadata in views:
                    step += 1
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
                    refined_events, sampled = probe.sample_dense_event_logits(
                        parts.refined_logits.squeeze(0), video, start, stop
                    )
                    base_events, base_sampled = probe.sample_dense_event_logits(
                        base_logits, video, start, stop
                    )
                    if not np.array_equal(sampled, metadata["global_indices"]):
                        raise RuntimeError("candidate event order changed")
                    if not np.array_equal(base_sampled, sampled):
                        raise RuntimeError("paired base event order changed")
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
                    if not torch.isfinite(loss):
                        raise RuntimeError("formal FP32 loss is non-finite")
                    if step == 1:
                        initial_identity = bool(
                            torch.equal(
                                parts.refined_logits.detach(),
                                base_logits.unsqueeze(0),
                            )
                            and torch.count_nonzero(parts.correction.detach()) == 0
                        )
                        if not initial_identity:
                            raise RuntimeError("formal zero-init is not bitwise M20 identity")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        adapter.trainable_parameters(),
                        float(science["training"]["gradient_clip_norm"]),
                    )
                    gradient_l1 = {}
                    for name, parameter in adapter.expert.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(
                                "missing/non-finite formal gradient: {}".format(name)
                            )
                        gradient_l1[name] = float(parameter.grad.detach().abs().sum())
                    if step == 1 and gradient_l1["output_projection.weight"] <= 0.0:
                        raise RuntimeError("formal output projection unreachable at step one")
                    optimizer.step()
                    for name, parameter in adapter.expert.named_parameters():
                        if not torch.isfinite(parameter).all():
                            raise RuntimeError(
                                "non-finite formal parameter after update: {}".format(name)
                            )
                    dual_state.update(recall, suppression)
                    weights = parts.mixture_weights.detach().float()
                    mixture_entropy = float(
                        (
                            -(
                                weights
                                * weights.clamp_min(
                                    torch.finfo(weights.dtype).eps
                                ).log()
                            ).sum(dim=2)
                        ).mean()
                    )
                    records.append(
                        {
                            "step": step,
                            "epoch": epoch,
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
                            "dual_target_time_recall_after": float(
                                dual_state.target_time_recall
                            ),
                            "dual_hard_negative_suppression_after": float(
                                dual_state.hard_negative_suppression
                            ),
                            "mixture_entropy": mixture_entropy,
                            "correction_abs_mean": float(
                                parts.correction.detach().float().abs().mean()
                            ),
                            "event_count": int(refined_events.numel()),
                        }
                    )
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
                source_records.append(
                    {
                        "epoch": epoch,
                        "source_name": source_name,
                        "source_sha256": expected["sha256"],
                        "event_count": int(expected["event_count"]),
                        "full_temporal_bins": len(video.event_indices_by_bin),
                        "first_streaming_decoder_bins": first_decoder_bins,
                        "eligible_joint_view_count": eligible_count,
                        "selected_view_starts": [value["start"] for value in views],
                        "selected_view_purposes": [value["purpose"] for value in views],
                        "base_C00_component_count": base_component_count,
                        "pure_FP_component_count": len(pure_fp_components),
                        "base_C00_stats": post_stats,
                        "optimizer_step_start": source_step_start,
                        "optimizer_step_stop": step,
                    }
                )
                del (
                    video,
                    polarities,
                    locations4,
                    labels,
                    target_ids,
                    memory,
                    summary_cache,
                    raw_scores,
                    pure_fp_components,
                    views,
                )
                torch.cuda.empty_cache()
                print(
                    "formal train epoch {}/{} source {} steps {}/{}".format(
                        epoch, EPOCHS, source_name, step, EXPECTED_STEPS
                    ),
                    flush=True,
                )

        if step != EXPECTED_STEPS:
            raise RuntimeError("formal optimizer-step count mismatch")
        validate_pyramid_step_diagnostics(records, EXPECTED_STEPS)
        if not all(record["target_time_group_count"] > 0 for record in records):
            raise RuntimeError("a formal step lacks the target-time constraint")
        if not all(record["hard_negative_component_count"] > 0 for record in records):
            raise RuntimeError("a formal step lacks the hard-negative constraint")
        fit_final_hashes = {}
        for source_name in FIT_SOURCES:
            path = source_path_from_manifest(source_name, manifest)
            fit_final_hashes[source_name] = sha256_file(path)
            if fit_final_hashes[source_name] != fit_initial_hashes[source_name]:
                raise RuntimeError("fit source changed during formal training")
        m20_after = atomic.state_sha256(m20.state_dict())
        if m20_after != m20_before:
            raise RuntimeError("released M20 changed during formal training")
        torch.cuda.synchronize()
        peak_cuda_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        expert_state = {
            key: value.detach().cpu().clone()
            for key, value in adapter.expert.state_dict().items()
        }
        expert_state_sha256 = atomic.state_sha256(expert_state)
        training_elapsed = time.perf_counter() - started
        checkpoint = {
            "schema": "ev-uav-h2-multiscale-temporal-pyramid-hold-g3-checkpoint-v1",
            "created_utc": utc_now(),
            "fold_id": "hold_g3_first",
            "fit_sources": list(FIT_SOURCES),
            "held_sources_reserved_unread": list(HELD_SOURCES),
            "held_arrays_read": False,
            "validation_or_test_read": False,
            "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
            "execution_amendment_sha256": sha256_file(EXECUTION_PATH),
            "formal_runner_sha256": sha256_file(Path(__file__)),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "loss_sha256": EXPECTED_LOSS_SHA256,
            "released_m20_sha256": EXPECTED_M20_SHA256,
            "released_m20_state_sha256": m20_after,
            "source_manifest_protocol_sha256": EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256,
            "effective_C00_sha256": EXPECTED_C00_SHA256,
            "numeric_strategy": {
                "frozen_M20": "FP32",
                "expert_forward_constraint_loss_backward_optimizer": "FP32",
                "grad_scaler_used": False,
            },
            "checkpoint_selection": "final_epoch_only",
            "optimizer_steps": step,
            "epochs": EPOCHS,
            "dual_state": dual_state.to_dict(),
            "expert_state_sha256": expert_state_sha256,
            "expert_state_dict": expert_state,
        }
        checkpoint_sha256 = write_torch_with_sidecar_exclusive(
            CHECKPOINT_PATH, checkpoint
        )
        reloaded = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if reloaded.get("expert_state_sha256") != atomic.state_sha256(
            reloaded["expert_state_dict"]
        ):
            raise RuntimeError("persisted formal expert state failed hash verification")
        if reloaded.get("optimizer_steps") != EXPECTED_STEPS:
            raise RuntimeError("persisted formal checkpoint step count changed")
        if tuple(reloaded.get("fit_sources", ())) != FIT_SOURCES:
            raise RuntimeError("persisted formal checkpoint fit set changed")
        checkpoint_verified_sha256 = verify_sidecar(
            CHECKPOINT_PATH, CHECKPOINT_SIDECAR_PATH
        )
        if checkpoint_verified_sha256 != checkpoint_sha256:
            raise RuntimeError("formal checkpoint SHA changed after write")
        trends = constraint_trends(records)
        training_receipt = {
            "schema": "ev-uav-h2-multiscale-temporal-pyramid-hold-g3-training-result-v1",
            "created_utc": utc_now(),
            "fold_id": "hold_g3_first",
            "fit_sources": list(FIT_SOURCES),
            "held_sources_reserved_unread": list(HELD_SOURCES),
            "held_G3_arrays_read": False,
            "validation_or_test_read": False,
            "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_reloaded_and_verified_before_process_exit": True,
            "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
            "execution_amendment_sha256": sha256_file(EXECUTION_PATH),
            "formal_runner_sha256": sha256_file(Path(__file__)),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "loss_sha256": EXPECTED_LOSS_SHA256,
            "released_m20_sha256": EXPECTED_M20_SHA256,
            "released_m20_state_sha256_before": m20_before,
            "released_m20_state_sha256_after": m20_after,
            "source_manifest_protocol_sha256": EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256,
            "effective_C00": effective_c00,
            "effective_C00_sha256": EXPECTED_C00_SHA256,
            "numeric_strategy": checkpoint["numeric_strategy"],
            "optimizer_steps": step,
            "epochs": EPOCHS,
            "views_per_fit_source_per_epoch": VIEWS_PER_SOURCE_PER_EPOCH,
            "initial_actual_M20_bitwise_identity": bool(initial_identity),
            "dual_final": dual_state.to_dict(),
            "constraint_trends": trends,
            "all_step_diagnostics": records,
            "source_epoch_diagnostics": source_records,
            "fit_source_sha256_before": fit_initial_hashes,
            "fit_source_sha256_after": fit_final_hashes,
            "training_elapsed_seconds": training_elapsed,
            "peak_CUDA_MiB": peak_cuda_mib,
            "formal_held_evaluation_started": False,
        }
        training_receipt_sha256 = write_json_with_sidecar_exclusive(
            TRAINING_RECEIPT_PATH, training_receipt
        )
        del (
            reloaded,
            checkpoint,
            expert_state,
            adapter,
            m20,
            optimizer,
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        cuda_after_release_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)

    print(
        json.dumps(
            {
                "checkpoint": str(CHECKPOINT_PATH.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "training_receipt": str(TRAINING_RECEIPT_PATH.resolve()),
                "training_receipt_sha256": training_receipt_sha256,
                "optimizer_steps": EXPECTED_STEPS,
                "constraint_trends": trends,
                "peak_CUDA_MiB": peak_cuda_mib,
                "CUDA_allocated_after_release_MiB": cuda_after_release_mib,
                "held_G3_arrays_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def verify_training_gate_before_held(contract):
    required = (
        CHECKPOINT_PATH,
        CHECKPOINT_SIDECAR_PATH,
        TRAINING_RECEIPT_PATH,
        TRAINING_RECEIPT_SIDECAR_PATH,
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("final checkpoint/receipt gate is incomplete")
    checkpoint_sha256 = verify_sidecar(CHECKPOINT_PATH, CHECKPOINT_SIDECAR_PATH)
    receipt_sha256 = verify_sidecar(
        TRAINING_RECEIPT_PATH, TRAINING_RECEIPT_SIDECAR_PATH
    )
    receipt = read_json(TRAINING_RECEIPT_PATH)
    if receipt.get("held_G3_arrays_read") is not False:
        raise RuntimeError("training receipt does not prove held unread")
    if receipt.get("formal_held_evaluation_started") is not False:
        raise RuntimeError("training receipt says held evaluation already started")
    if receipt.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("training receipt checkpoint SHA mismatch")
    if receipt.get("science_protocol_sha256") != EXPECTED_SCIENCE_SHA256:
        raise RuntimeError("training receipt science protocol mismatch")
    if receipt.get("execution_amendment_sha256") != sha256_file(EXECUTION_PATH):
        raise RuntimeError("training receipt execution amendment mismatch")
    if receipt.get("optimizer_steps") != EXPECTED_STEPS:
        raise RuntimeError("training receipt step count mismatch")
    if tuple(receipt.get("fit_sources", ())) != FIT_SOURCES:
        raise RuntimeError("training receipt fit source mismatch")
    if tuple(receipt.get("held_sources_reserved_unread", ())) != HELD_SOURCES:
        raise RuntimeError("training receipt held source mismatch")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("held_arrays_read") is not False:
        raise RuntimeError("checkpoint does not prove held unread")
    if checkpoint.get("checkpoint_selection") != "final_epoch_only":
        raise RuntimeError("checkpoint selection changed")
    if checkpoint.get("optimizer_steps") != EXPECTED_STEPS:
        raise RuntimeError("checkpoint optimizer steps changed")
    if tuple(checkpoint.get("fit_sources", ())) != FIT_SOURCES:
        raise RuntimeError("checkpoint fit set changed")
    if tuple(checkpoint.get("held_sources_reserved_unread", ())) != HELD_SOURCES:
        raise RuntimeError("checkpoint held set changed")
    if checkpoint.get("science_protocol_sha256") != EXPECTED_SCIENCE_SHA256:
        raise RuntimeError("checkpoint science protocol changed")
    if checkpoint.get("execution_amendment_sha256") != sha256_file(EXECUTION_PATH):
        raise RuntimeError("checkpoint execution amendment changed")
    if checkpoint.get("formal_runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("formal runner changed between training and held evaluation")
    state_sha256 = atomic.state_sha256(checkpoint["expert_state_dict"])
    if checkpoint.get("expert_state_sha256") != state_sha256:
        raise RuntimeError("checkpoint expert state hash mismatch")
    if EVALUATION_ROOT.exists():
        raise FileExistsError("refusing to overwrite held-G3 evaluation")
    return checkpoint, receipt, checkpoint_sha256, receipt_sha256


def predict_paired_full_stream(adapter, video, device):
    memory = probe.full_stream_memory(adapter.released_m20, video, device)
    observations_cpu, base_scores, first_decoder_bins = (
        probe.stream_observations_and_scores(adapter, video, memory, device)
    )
    summary_cache = probe.build_summary_cache(observations_cpu, device)
    del observations_cpu
    candidate_scores = np.empty_like(base_scores)
    second_decoder_bins = 0
    correction_abs_sum = 0.0
    correction_element_count = 0
    mixture_sum = np.zeros(4, dtype=np.float64)
    mixture_element_count = 0
    entropy_sum = 0.0
    entropy_element_count = 0
    with torch.no_grad():
        for start in range(0, TEMPORAL_COUNT, INFERENCE_BATCH):
            stop = min(start + INFERENCE_BATCH, TEMPORAL_COUNT)
            frames = atomic._frame_tensor(video, range(start, stop), device)
            decoder, base_logits, centre = adapter.decode_frozen_features(
                frames, memory[start:stop]
            )
            summaries = tuple(
                value[start:stop].to(device=device, dtype=torch.float32)
                for value in summary_cache
            )
            parts = adapter.expert(
                decoder.unsqueeze(0),
                base_logits.unsqueeze(0),
                centre.unsqueeze(0),
                tuple(value.unsqueeze(0) for value in summaries),
                return_parts=True,
            )
            probabilities = torch.sigmoid(parts.refined_logits).squeeze(0).squeeze(1)
            for temporal_bin in range(start, stop):
                indices = video.event_indices_by_bin[temporal_bin]
                if indices.size == 0:
                    continue
                local = temporal_bin - start
                xy = video.locations[indices]
                candidate_scores[indices] = (
                    probabilities[local, xy[:, 1], xy[:, 0]].cpu().numpy()
                )
            correction = parts.correction.detach().float()
            correction_abs_sum += float(correction.abs().sum())
            correction_element_count += correction.numel()
            weights = parts.mixture_weights.detach().float()
            mixture_sum += weights.sum(dim=(0, 1, 3, 4)).cpu().numpy()
            mixture_element_count += int(
                weights.shape[0] * weights.shape[1] * weights.shape[3] * weights.shape[4]
            )
            entropy = -(
                weights
                * weights.clamp_min(torch.finfo(weights.dtype).eps).log()
            ).sum(dim=2)
            entropy_sum += float(entropy.sum())
            entropy_element_count += entropy.numel()
            second_decoder_bins += stop - start
            del (
                frames,
                decoder,
                base_logits,
                centre,
                summaries,
                parts,
                probabilities,
                correction,
                weights,
                entropy,
            )
    if first_decoder_bins != TEMPORAL_COUNT or second_decoder_bins != TEMPORAL_COUNT:
        raise RuntimeError("held paired streaming decode did not cover T160")
    if not np.isfinite(base_scores).all() or not np.isfinite(candidate_scores).all():
        raise RuntimeError("held paired raw scores contain non-finite values")
    diagnostics = {
        "first_streaming_decoder_bins": first_decoder_bins,
        "second_streaming_decoder_bins": second_decoder_bins,
        "correction_abs_dense_mean": correction_abs_sum
        / max(correction_element_count, 1),
        "mixture_scale_mean": (mixture_sum / max(mixture_element_count, 1)).tolist(),
        "mixture_entropy_mean": entropy_sum / max(entropy_element_count, 1),
    }
    del memory, summary_cache
    return base_scores, candidate_scores, diagnostics


def challenge_view(counts, metrics):
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


def delta_view(base, candidate):
    return {
        key: float(candidate[key] - base[key])
        if key in {"Score", "IoU", "Pd", "Fa"}
        else int(candidate[key] - base[key])
        for key in base
    }


def safety_gates(base, candidate, *, require_effect_size):
    gates = {
        "Score_not_lower": candidate["Score"] >= base["Score"],
        "IoU_not_lower": candidate["IoU"] >= base["IoU"],
        "Pd_not_lower": candidate["Pd"] >= base["Pd"],
        "TP_not_lower": candidate["TP"] >= base["TP"],
        "CO_not_lower": candidate["CO"] >= base["CO"],
        "Fa_not_higher": candidate["Fa"] <= base["Fa"],
        "FP_not_higher": candidate["FP"] <= base["FP"],
        "FC_not_higher": candidate["FC"] <= base["FC"],
    }
    if require_effect_size:
        gates["Score_gain_at_least_0_01"] = (
            candidate["Score"] - base["Score"] >= 0.01
        )
    return gates


def evaluate_held_g3(args):
    require_gpu_authorization(args)
    _, contract = load_frozen_contract()
    cfg, effective_c00 = build_and_validate_c00(contract)
    checkpoint, training_receipt, checkpoint_sha256, training_receipt_sha256 = (
        verify_training_gate_before_held(contract)
    )
    manifest = contract["source_manifest"]["sources"]
    pooled_base = crossfit.SufficientCounts()
    pooled_candidate = crossfit.SufficientCounts()
    records = []
    held_access_ledger = []
    started = time.perf_counter()
    with atomic.gpu_run_lock("h2_multiscale_pyramid_formal_hold_g3_evaluation"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m20, _ = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        adapter.expert.load_state_dict(checkpoint["expert_state_dict"], strict=True)
        adapter.eval()
        if atomic.state_sha256(adapter.expert.state_dict()) != checkpoint[
            "expert_state_sha256"
        ]:
            raise RuntimeError("loaded held expert state hash mismatch")

        for source_name in HELD_SOURCES:
            source_open_utc = utc_now()
            path, expected = verify_source_file(source_name, manifest)
            video, polarities, locations4 = atomic._load_input_only(path)
            if len(polarities) != int(expected["event_count"]):
                raise RuntimeError("held source event count changed")
            if len(video.event_indices_by_bin) != TEMPORAL_COUNT:
                raise RuntimeError("held source is not T160")
            if not use_h2_residual_refiner(len(polarities), polarities):
                raise RuntimeError("held source no longer satisfies input-only H2 route")
            base_raw, candidate_raw, inference_diagnostics = (
                predict_paired_full_stream(adapter, video, device)
            )
            locations_tensor = torch.from_numpy(locations4).long().contiguous()
            base_processed, base_stats = ChallengePostprocessor.from_cfg(
                cfg, PREDICTION_THRESHOLD, event_count=len(polarities)
            ).apply(torch.from_numpy(base_raw.copy()), locations_tensor)
            candidate_processed, candidate_stats = ChallengePostprocessor.from_cfg(
                cfg, PREDICTION_THRESHOLD, event_count=len(polarities)
            ).apply(torch.from_numpy(candidate_raw.copy()), locations_tensor)
            base_post = base_processed.numpy().astype(np.float32, copy=True)
            candidate_post = candidate_processed.numpy().astype(np.float32, copy=True)
            artifact_path = EVALUATION_ROOT / "artifacts" / (
                source_name.replace(".npz", "_paired_scores.npz")
            )
            artifact_arrays = {
                "artifact_schema": np.asarray(
                    ["ev-uav-h2-pyramid-paired-input-only-scores-v1"], dtype="<U52"
                ),
                "event_count": np.asarray([len(polarities)], dtype=np.int64),
                "event_index": np.arange(len(polarities), dtype=np.int64),
                "locations4": locations4.astype(np.int64, copy=False),
                "base_raw_scores": base_raw.astype(np.float32, copy=False),
                "candidate_raw_scores": candidate_raw.astype(np.float32, copy=False),
                "base_post_C00_scores": base_post,
                "candidate_post_C00_scores": candidate_post,
            }
            artifact_sha256 = write_npz_with_sidecar_exclusive(
                artifact_path, artifact_arrays
            )
            artifact_manifest = {
                "schema": "ev-uav-h2-pyramid-paired-score-artifact-manifest-v1",
                "created_utc": utc_now(),
                "source_name": source_name,
                "source_sha256": expected["sha256"],
                "event_count": int(expected["event_count"]),
                "checkpoint_sha256": checkpoint_sha256,
                "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
                "execution_amendment_sha256": sha256_file(EXECUTION_PATH),
                "formal_runner_sha256": sha256_file(Path(__file__)),
                "released_m20_sha256": EXPECTED_M20_SHA256,
                "model_sha256": EXPECTED_MODEL_SHA256,
                "loss_sha256": EXPECTED_LOSS_SHA256,
                "effective_C00_sha256": EXPECTED_C00_SHA256,
                "artifact_path": str(artifact_path.resolve()),
                "artifact_sha256": artifact_sha256,
                "contains_labels_or_target_ids": False,
                "truth_fields_accessed_before_artifact_write": False,
                "array_content_sha256": {
                    "base_raw_scores": atomic.sha256_float32(base_raw),
                    "candidate_raw_scores": atomic.sha256_float32(candidate_raw),
                    "base_post_C00_scores": atomic.sha256_float32(base_post),
                    "candidate_post_C00_scores": atomic.sha256_float32(candidate_post),
                },
                "base_postprocess": asdict(base_stats),
                "candidate_postprocess": asdict(candidate_stats),
                "inference_diagnostics": inference_diagnostics,
            }
            artifact_manifest_path = Path(str(artifact_path) + ".manifest.json")
            artifact_manifest_sha256 = write_json_with_sidecar_exclusive(
                artifact_manifest_path, artifact_manifest
            )

            truth_access_utc = utc_now()
            labels, target_ids = atomic._load_truth(path)
            if labels.size != len(polarities) or target_ids.size != len(polarities):
                raise RuntimeError("held truth does not align with score artifact")
            base_counts = crossfit.sufficient_counts_for_video(
                base_post,
                labels,
                target_ids,
                locations4,
                PREDICTION_THRESHOLD,
            )
            candidate_counts = crossfit.sufficient_counts_for_video(
                candidate_post,
                labels,
                target_ids,
                locations4,
                PREDICTION_THRESHOLD,
            )
            if base_counts.event_count != candidate_counts.event_count:
                raise RuntimeError("paired held event-count metric invariant failed")
            if base_counts.frame_count != candidate_counts.frame_count:
                raise RuntimeError("paired held frame-count metric invariant failed")
            if base_counts.object_count != candidate_counts.object_count:
                raise RuntimeError("paired held object-count metric invariant failed")
            if (
                base_counts.true_positive_events + base_counts.false_negative_events
                != candidate_counts.true_positive_events
                + candidate_counts.false_negative_events
            ):
                raise RuntimeError("paired held positive-event invariant failed")
            base_metrics = crossfit.metrics_from_counts(base_counts)
            candidate_metrics = crossfit.metrics_from_counts(candidate_counts)
            base_view = challenge_view(base_counts, base_metrics)
            candidate_view = challenge_view(candidate_counts, candidate_metrics)
            per_source_gates = safety_gates(
                base_view, candidate_view, require_effect_size=False
            )
            records.append(
                {
                    "source_name": source_name,
                    "source_sha256": expected["sha256"],
                    "event_count": int(expected["event_count"]),
                    "input_only_route": {
                        "event_count": len(polarities),
                        "polarity_minority_fraction": float(
                            min(
                                np.mean(polarities > 0),
                                np.mean(polarities <= 0),
                            )
                        ),
                        "candidate": "h2_multiscale_temporal_pyramid",
                    },
                    "score_artifact_path": str(artifact_path.resolve()),
                    "score_artifact_sha256": artifact_sha256,
                    "score_artifact_manifest_path": str(
                        artifact_manifest_path.resolve()
                    ),
                    "score_artifact_manifest_sha256": artifact_manifest_sha256,
                    "base_counts": base_counts.to_dict(),
                    "candidate_counts": candidate_counts.to_dict(),
                    "base_metrics": base_metrics,
                    "candidate_metrics": candidate_metrics,
                    "base_report": base_view,
                    "candidate_report": candidate_view,
                    "delta_report": delta_view(base_view, candidate_view),
                    "per_source_safety_gates": per_source_gates,
                    "all_per_source_safety_gates_passed": all(
                        per_source_gates.values()
                    ),
                }
            )
            held_access_ledger.append(
                {
                    "source_name": source_name,
                    "input_array_open_utc": source_open_utc,
                    "input_only_score_artifact_committed_utc": artifact_manifest[
                        "created_utc"
                    ],
                    "truth_field_access_utc": truth_access_utc,
                    "artifact_preceded_truth_field_access": True,
                }
            )
            pooled_base = pooled_base + base_counts
            pooled_candidate = pooled_candidate + candidate_counts
            del (
                video,
                polarities,
                locations4,
                base_raw,
                candidate_raw,
                base_processed,
                candidate_processed,
                base_post,
                candidate_post,
                artifact_arrays,
                labels,
                target_ids,
            )
            torch.cuda.empty_cache()
            print("formal held paired {}".format(source_name), flush=True)

        m20_after = atomic.state_sha256(m20.state_dict())
        if m20_after != m20_before or m20_after != checkpoint[
            "released_m20_state_sha256"
        ]:
            raise RuntimeError("released M20 changed during held evaluation")
        torch.cuda.synchronize()
        peak_cuda_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        del adapter, m20
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        cuda_after_release_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)

    pooled_base_metrics = crossfit.metrics_from_counts(pooled_base)
    pooled_candidate_metrics = crossfit.metrics_from_counts(pooled_candidate)
    pooled_base_view = challenge_view(pooled_base, pooled_base_metrics)
    pooled_candidate_view = challenge_view(pooled_candidate, pooled_candidate_metrics)
    pooled_delta = delta_view(pooled_base_view, pooled_candidate_view)
    pooled_gates = safety_gates(
        pooled_base_view, pooled_candidate_view, require_effect_size=True
    )
    promoted = all(pooled_gates.values())
    payload = {
        "schema": "ev-uav-h2-multiscale-temporal-pyramid-hold-g3-paired-evaluation-v1",
        "created_utc": utc_now(),
        "fold_id": "hold_g3_first",
        "fit_sources": list(FIT_SOURCES),
        "held_sources": list(HELD_SOURCES),
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_receipt_path": str(TRAINING_RECEIPT_PATH.resolve()),
        "training_receipt_sha256": training_receipt_sha256,
        "checkpoint_and_receipt_verified_before_first_held_open": True,
        "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
        "execution_amendment_sha256": sha256_file(EXECUTION_PATH),
        "formal_runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "loss_sha256": EXPECTED_LOSS_SHA256,
        "released_m20_sha256": EXPECTED_M20_SHA256,
        "released_m20_state_sha256_before": m20_before,
        "released_m20_state_sha256_after": m20_after,
        "source_manifest_protocol_sha256": EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "effective_C00": effective_c00,
        "effective_C00_sha256": EXPECTED_C00_SHA256,
        "numeric_strategy": checkpoint["numeric_strategy"],
        "training_constraint_trends": training_receipt["constraint_trends"],
        "held_access_ledger": held_access_ledger,
        "records": records,
        "pooled_base_counts": pooled_base.to_dict(),
        "pooled_candidate_counts": pooled_candidate.to_dict(),
        "pooled_base_metrics": pooled_base_metrics,
        "pooled_candidate_metrics": pooled_candidate_metrics,
        "pooled_base_report": pooled_base_view,
        "pooled_candidate_report": pooled_candidate_view,
        "pooled_delta_report": pooled_delta,
        "promotion_gates": pooled_gates,
        "promoted": promoted,
        "failure_action_if_not_promoted": (
            "permanently_archive_without_other_folds_or_tuning"
        ),
        "other_fold_started": False,
        "validation_or_test_read": False,
        "evaluation_elapsed_seconds": time.perf_counter() - started,
        "peak_CUDA_MiB": peak_cuda_mib,
        "CUDA_allocated_after_release_MiB": cuda_after_release_mib,
    }
    evaluation_sha256 = write_json_with_sidecar_exclusive(EVALUATION_PATH, payload)
    decision = {
        "schema": "ev-uav-h2-multiscale-temporal-pyramid-branch-decision-v1",
        "created_utc": utc_now(),
        "fold_id": "hold_g3_first",
        "paired_evaluation_path": str(EVALUATION_PATH.resolve()),
        "paired_evaluation_sha256": evaluation_sha256,
        "promotion_gates": pooled_gates,
        "promoted": promoted,
        "decision": (
            "eligible_for_next_frozen_fold_but_not_started"
            if promoted
            else "permanently_archived_no_other_folds_no_tuning"
        ),
        "validation_or_test_read": False,
    }
    decision_sha256 = write_json_with_sidecar_exclusive(DECISION_PATH, decision)
    print(
        json.dumps(
            {
                "evaluation": str(EVALUATION_PATH.resolve()),
                "evaluation_sha256": evaluation_sha256,
                "decision": str(DECISION_PATH.resolve()),
                "decision_sha256": decision_sha256,
                "pooled_base": pooled_base_view,
                "pooled_candidate": pooled_candidate_view,
                "pooled_delta": pooled_delta,
                "promotion_gates": pooled_gates,
                "promoted": promoted,
                "validation_or_test_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cpu_audit(_args):
    science, contract = load_frozen_contract()
    _, effective_c00 = build_and_validate_c00(contract)
    if TRAIN_OUTPUT_ROOT.exists() or EVALUATION_ROOT.exists():
        raise RuntimeError("formal output already exists; CPU preflight is no longer pristine")
    report = {
        "schema": "ev-uav-h2-multiscale-temporal-pyramid-formal-cpu-audit-v1",
        "created_utc": utc_now(),
        "science_protocol_path": str(SCIENCE_PATH.resolve()),
        "science_protocol_sha256": EXPECTED_SCIENCE_SHA256,
        "execution_amendment_path": str(EXECUTION_PATH.resolve()),
        "execution_amendment_sha256": sha256_file(EXECUTION_PATH),
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "loss_sha256": EXPECTED_LOSS_SHA256,
        "tests_sha256": EXPECTED_TEST_SHA256,
        "released_m20_sha256": EXPECTED_M20_SHA256,
        "source_manifest_protocol_sha256": EXPECTED_SOURCE_MANIFEST_PROTOCOL_SHA256,
        "effective_C00": effective_c00,
        "effective_C00_sha256": EXPECTED_C00_SHA256,
        "fold_id": "hold_g3_first",
        "fit_sources": list(FIT_SOURCES),
        "held_sources_reserved_unread": list(HELD_SOURCES),
        "optimizer_steps": EXPECTED_STEPS,
        "epochs": EPOCHS,
        "views_per_fit_source_per_epoch": VIEWS_PER_SOURCE_PER_EPOCH,
        "numeric_strategy": contract["numeric_recovery"],
        "train_and_held_are_separate_processes": True,
        "checkpoint_hash_gate_before_held": True,
        "held_arrays_read": False,
        "validation_or_test_read": False,
        "cuda_initialized_or_used": False,
        "science_training_steps_match": science["training"][
            "first_fold_optimizer_steps"
        ]
        == EXPECTED_STEPS,
        "cpu_audit_passed": True,
    }
    digest = write_json_with_sidecar_exclusive(CPU_AUDIT_PATH, report)
    print(
        json.dumps(
            {
                "report": str(CPU_AUDIT_PATH.resolve()),
                "report_sha256": digest,
                "cpu_audit_passed": True,
                "held_arrays_read": False,
                "cuda_initialized_or_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.set_defaults(func=cpu_audit)
    train = subparsers.add_parser("train-hold-g3")
    train.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
    train.set_defaults(func=train_formal)
    evaluate = subparsers.add_parser("evaluate-hold-g3")
    evaluate.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
    evaluate.set_defaults(func=evaluate_held_g3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

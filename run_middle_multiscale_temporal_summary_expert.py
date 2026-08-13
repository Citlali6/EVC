"""Train-only CPU audit and eight-step resource probe for the middle expert.

The frozen candidate reuses the already-audited 16/32/64/full-T160 pyramid
head and constrained loss without modifying any H2 source or artifact.  This
runner intentionally stops at the GPU-probe authorization boundary; formal
LOFO training/evaluation is a later, separately authorized phase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROTOCOL_PATH = ROOT / "protocols" / "middle_multiscale_temporal_summary_expert_science_v1.json"
OUTPUT_ROOT = WORKSPACE / "experiments" / "20260811_middle_multiscale_temporal_summary_grouped_oof_v1"
AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
PROBE_PATH = OUTPUT_ROOT / "resource_probe" / "eight_step_probe.json"
EXPECTED_PROTOCOL_SHA256 = "f17c689186fbfff1763460907f2ae9a5093e992315d004e0b4db537754c5dafe"
EXPECTED_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EXPECTED_MODEL_SHA256 = "4d4ea4a365be49ad1b6c7cf1c7c96c2369caf3e12841bbcd781cf109105a6a98"
EXPECTED_LOSS_SHA256 = "f74e145b04b25f2e7478f5c8fd370bc4e9d96123ef6f75ea5833acc210d2c5e9"
PROBE_STEPS = 8
TEMPORAL_COUNT = 160
VIEW_BINS = 16
PREDICTION_THRESHOLD = 0.719
GPU_FLAG = "--root-authorized-gpu"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload):
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = json_bytes(payload)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(values)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(values).hexdigest()


def load_json_snapshot(path):
    path = Path(path)
    before = sha256_file(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("JSON changed while it was read: {}".format(path))
    return payload, after


def workspace_path(relative):
    path = (WORKSPACE / relative).resolve()
    try:
        path.relative_to(WORKSPACE.resolve())
    except ValueError as error:
        raise RuntimeError("protocol path escaped the workspace") from error
    return path


def middle_route(event_count):
    event_count = int(event_count)
    return 30000 < event_count <= 200000


def load_protocol():
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("middle science protocol SHA-256 changed")
    protocol, after = load_json_snapshot(PROTOCOL_PATH)
    if after != actual:
        raise RuntimeError("middle science protocol changed while loading")
    if protocol.get("schema") != "ev-uav-middle-multiscale-temporal-summary-grouped-oof-science-v1":
        raise RuntimeError("middle science schema changed")
    scope = protocol["science_scope"]
    if scope["validation_read_allowed"] or scope["test_read_allowed"]:
        raise RuntimeError("validation/test access must remain forbidden")
    route = protocol["input_only_route"]
    if route["lower_bound_exclusive"] != 30000 or route["upper_bound_inclusive"] != 200000:
        raise RuntimeError("middle input-only route changed")
    if route["observable_inputs"] != ["event_count"] or route["file_name_or_source_identity_used"]:
        raise RuntimeError("middle route is not source-free")
    if protocol["eight_step_resource_probe"]["GPU_authorized"] is not False:
        raise RuntimeError("science file must remain unauthorized; CLI carries authorization")
    return protocol, actual


def family_sources(protocol):
    return {
        name: tuple(values)
        for name, values in protocol["continuous_source_families"].items()
    }


def all_middle_sources(protocol):
    return tuple(source for values in family_sources(protocol).values() for source in values)


def first_fold_spec(protocol):
    folds = protocol["fold_order"]
    first = [fold for fold in folds if fold.get("is_first_fold")]
    if len(first) != 1 or first[0] is not folds[0]:
        raise RuntimeError("exactly the first frozen fold must be marked first")
    families = family_sources(protocol)
    fold = first[0]
    fit = tuple(source for name in fold["fit_families"] for source in families[name])
    held = families[fold["held_family"]]
    return {**fold, "fit_sources": fit, "held_sources": held}


def _manifest_evidence(protocol):
    cache_root = workspace_path(protocol["released_m20_train_cache"]["workspace_relative_path"])
    manifest_path = cache_root / "manifest.json"
    manifest, digest = load_json_snapshot(manifest_path)
    if digest != protocol["released_m20_train_cache"]["manifest_sha256"]:
        raise RuntimeError("released M20 train-cache manifest changed")
    if manifest.get("schema") != protocol["released_m20_train_cache"]["schema"]:
        raise RuntimeError("released M20 train-cache schema changed")
    by_name = {record["source_name"]: record for record in manifest["records"]}
    return cache_root, manifest_path, digest, manifest, by_name


def _validate_sources_and_folds(protocol, cache_root, manifest, by_name):
    families = family_sources(protocol)
    expected_family_names = (
        "f1_000_014",
        "f2_028_032",
        "f3_040_043",
        "f4_059_065",
        "f5_067_074",
    )
    if tuple(families) != expected_family_names:
        raise RuntimeError("continuous family order changed")
    sources = all_middle_sources(protocol)
    if len(sources) != 39 or len(set(sources)) != 39:
        raise RuntimeError("middle source membership is not exactly 39 unique videos")
    evidence = {}
    for family_name, names in families.items():
        indices = [int(name[6:9]) for name in names]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise RuntimeError("family is not contiguous: {}".format(family_name))
        for source in names:
            if source not in by_name:
                raise RuntimeError("middle source is absent from cache manifest: {}".format(source))
            record = by_name[source]
            if not middle_route(record["event_count"]):
                raise RuntimeError("source escaped middle event-count route: {}".format(source))
            record_path = cache_root / record["record"]
            if sha256_file(record_path) != record["record_sha256"]:
                raise RuntimeError("middle cache record changed: {}".format(source))
            evidence[source] = {
                "family": family_name,
                "event_count": int(record["event_count"]),
                "positive_event_count": int(record["positive_event_count"]),
                "source_sha256": record["source_sha256"],
                "record": record["record"],
                "record_sha256": record["record_sha256"],
            }
    held_seen = []
    folds = []
    all_set = set(sources)
    for fold in protocol["fold_order"]:
        held = tuple(families[fold["held_family"]])
        fit = tuple(source for name in fold["fit_families"] for source in families[name])
        if set(held) & set(fit) or set(held) | set(fit) != all_set:
            raise RuntimeError("fold fit/held partition is not exact")
        if len(held) != fold["held_source_count"] or len(fit) != fold["fit_source_count"]:
            raise RuntimeError("fold source count changed")
        expected_steps = len(fit) * protocol["training"]["epochs"] * protocol["training"]["views_per_fit_source_per_epoch"]
        if expected_steps != fold["optimizer_steps"]:
            raise RuntimeError("fold optimizer-step arithmetic changed")
        held_seen.extend(held)
        folds.append({"fold_id": fold["fold_id"], "fit_sources": list(fit), "held_sources": list(held), "optimizer_steps": expected_steps})
    if len(held_seen) != 39 or len(set(held_seen)) != 39 or set(held_seen) != all_set:
        raise RuntimeError("LOFO held union is incomplete or duplicated")
    return evidence, folds


def _validate_dependencies(protocol):
    paths = {
        "model/h2_multiscale_temporal_pyramid_expert.py": ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py",
        "utils/h2_multiscale_pyramid_loss.py": ROOT / "utils" / "h2_multiscale_pyramid_loss.py",
        "run_h2_multiscale_temporal_pyramid_probe.py": ROOT / "run_h2_multiscale_temporal_pyramid_probe.py",
        "run_h2_atomic_component_deletion_v3.py": ROOT / "run_h2_atomic_component_deletion_v3.py",
        "utils/atomic_component_deletion.py": ROOT / "utils" / "atomic_component_deletion.py",
        "crossfit_component_reranker.py": ROOT / "crossfit_component_reranker.py",
        "train_component_reranker.py": ROOT / "train_component_reranker.py",
        "utils/postprocess.py": ROOT / "utils" / "postprocess.py",
    }
    expected = {
        "model/h2_multiscale_temporal_pyramid_expert.py": protocol["architecture"]["implementation_sha256"],
        "utils/h2_multiscale_pyramid_loss.py": protocol["architecture"]["loss_sha256"],
        **protocol["dependencies"],
    }
    evidence = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        if digest != expected[name]:
            raise RuntimeError("dependency changed: {}".format(name))
        evidence[name] = {"path": str(path.resolve()), "sha256": digest}
    parent = workspace_path(protocol["released_m20"]["workspace_relative_path"])
    if sha256_file(parent) != EXPECTED_M20_SHA256:
        raise RuntimeError("released M20 checkpoint changed")
    evidence["released_m20"] = {"path": str(parent), "sha256": EXPECTED_M20_SHA256}
    return evidence


def _synthetic_cpu_audit(protocol):
    import torch

    import crossfit_component_reranker as crossfit
    from model.h2_multiscale_temporal_pyramid_expert import (
        MultiScaleTemporalPyramidHead,
        downsample_frozen_observations,
        fixed_multiscale_temporal_moments,
        pyramid_expert_parameter_count,
    )
    from utils.h2_multiscale_pyramid_loss import (
        PyramidDualState,
        multiscale_pyramid_constrained_loss,
    )

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit started after CUDA initialization")
    torch.manual_seed(79)
    head = MultiScaleTemporalPyramidHead()
    decoder = torch.randn(1, 5, 16, 8, 8)
    base = torch.randn(1, 5, 1, 8, 8)
    centre = torch.randn(1, 5, 3, 8, 8)
    observations = downsample_frozen_observations(decoder, base, centre)
    summaries = fixed_multiscale_temporal_moments(observations)
    parts = head(decoder, base, centre, summaries, return_parts=True)
    identity = torch.equal(parts.refined_logits, base) and int(torch.count_nonzero(parts.correction)) == 0
    parameter_count = pyramid_expert_parameter_count(head)
    refined = torch.tensor((0.4, 0.2, -0.1, 0.3), requires_grad=True)
    base_events = torch.tensor((0.5, 0.3, 0.2, 0.4))
    labels = torch.tensor((1.0, 1.0, 0.0, 0.0))
    target_ids = torch.tensor((1, 1, 0, 0))
    times = torch.tensor((0, 1, 0, 0))
    loss, recall, suppression, diagnostics = multiscale_pyramid_constrained_loss(
        refined,
        base_events,
        labels,
        target_ids,
        times,
        (torch.tensor((2, 3)),),
        PyramidDualState(),
    )
    loss.backward()
    counts = crossfit.SufficientCounts(**protocol["parent_error_capacity_diagnostic"]["M20_counts"])
    metrics = crossfit.metrics_from_counts(counts)
    expected_metrics = protocol["parent_error_capacity_diagnostic"]["M20_metrics"]
    capacity_metrics_match = all(
        abs(float(metrics[key]) - float(expected_metrics[key])) <= 1e-12
        for key in expected_metrics
    )
    checks = {
        "cuda_uninitialized": not torch.cuda.is_initialized(),
        "zero_init_bitwise_M20_identity": bool(identity),
        "trainable_parameter_count_3381": parameter_count == 3381,
        "summary_scales_exact": tuple(head.scales) == (16, 32, 64, 160),
        "loss_finite": bool(torch.isfinite(loss)),
        "target_recall_finite": bool(torch.isfinite(recall)),
        "hard_negative_finite_positive": bool(torch.isfinite(suppression) and suppression > 0),
        "event_gradient_finite_nonzero": bool(torch.isfinite(refined.grad).all() and torch.count_nonzero(refined.grad) > 0),
        "capacity_metric_arithmetic_exact": capacity_metrics_match,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parameter_count": parameter_count,
        "loss_diagnostics": diagnostics,
    }


def _probe_source_cpu_preflight(protocol, cache_root, by_name, source_evidence):
    """Prove the frozen probe source has eight legal joint loss views on CPU."""
    import numpy as np
    import torch

    import run_h2_atomic_component_deletion_v3 as atomic
    import run_h2_multiscale_temporal_pyramid_probe as helper
    from utils.atomic_component_deletion import extract_atomic_components, pure_false_positive_targets
    from utils.postprocess import ChallengePostprocessor

    name = protocol["eight_step_resource_probe"]["source"]
    official = WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train" / name
    if sha256_file(official) != source_evidence[name]["source_sha256"]:
        raise RuntimeError("probe source changed before CPU preflight")
    video, polarities, locations4 = atomic._load_input_only(official)
    labels, target_ids = atomic._load_truth(official)
    record = by_name[name]
    with np.load(cache_root / record["record"], allow_pickle=False) as values:
        raw = values["scores"].astype(np.float32, copy=True)
    if raw.size != len(polarities) or not middle_route(len(polarities)):
        raise RuntimeError("probe source/cache route mismatch")
    cfg, effective = helper.build_c00()
    processed, _ = ChallengePostprocessor.from_cfg(
        cfg, PREDICTION_THRESHOLD, event_count=len(labels)
    ).apply(torch.from_numpy(raw), torch.from_numpy(locations4).long())
    components = extract_atomic_components(
        processed.numpy(),
        locations4,
        PREDICTION_THRESHOLD,
        spatial_radius=2,
        temporal_bin_size=50,
        temporal_radius_bins=0,
    )
    component_targets = pure_false_positive_targets(components.event_indices, labels)
    pure_fp = tuple(
        components.event_indices[index]
        for index in np.flatnonzero(component_targets == 1)
    )
    old_values = (helper.SEED, helper.PROBE_STEPS, helper.TEMPORAL_COUNT, helper.VIEW_BINS)
    helper.SEED = int(protocol["training"]["seed"])
    helper.PROBE_STEPS = PROBE_STEPS
    helper.TEMPORAL_COUNT = TEMPORAL_COUNT
    helper.VIEW_BINS = VIEW_BINS
    try:
        views, eligible = helper.prepare_view_metadata(
            video, labels, target_ids, pure_fp
        )
    finally:
        helper.SEED, helper.PROBE_STEPS, helper.TEMPORAL_COUNT, helper.VIEW_BINS = old_values
    checks = {
        "middle_route_true": middle_route(len(polarities)),
        "complete_T160": len(video.event_indices_by_bin) == TEMPORAL_COUNT,
        "eight_selected_views": len(views) == PROBE_STEPS,
        "joint_target_and_hard_negative_each_view": all(
            bool(np.any(view["labels"] > 0))
            and bool(view["hard_negative_components"])
            for view in views
        ),
        "effective_C00_exact": hashlib.sha256(
            json.dumps(effective, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        == EXPECTED_C00_SHA256,
        "cuda_uninitialized": not torch.cuda.is_initialized(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source": name,
        "event_count": len(polarities),
        "base_component_count": len(components.event_indices),
        "pure_FP_component_count": len(pure_fp),
        "eligible_joint_view_count": eligible,
        "selected_view_starts": [view["start"] for view in views],
    }


def build_audit_payload():
    protocol, protocol_sha = load_protocol()
    cache_root, manifest_path, manifest_sha, manifest, by_name = _manifest_evidence(protocol)
    source_evidence, folds = _validate_sources_and_folds(protocol, cache_root, manifest, by_name)
    dependencies = _validate_dependencies(protocol)
    synthetic = _synthetic_cpu_audit(protocol)
    probe_preflight = _probe_source_cpu_preflight(
        protocol, cache_root, by_name, source_evidence
    )
    probe_source = protocol["eight_step_resource_probe"]["source"]
    first = first_fold_spec(protocol)
    checks = {
        "source_count_39": len(source_evidence) == 39,
        "five_continuous_families": len(family_sources(protocol)) == 5,
        "five_exact_LOFO_folds": len(folds) == 5,
        "first_fold_fit_24_held_15_steps_96": len(first["fit_sources"]) == 24 and len(first["held_sources"]) == 15 and first["optimizer_steps"] == 96,
        "probe_source_is_first_fold_fit_only": probe_source in first["fit_sources"] and probe_source not in first["held_sources"],
        "probe_steps_8_formal_steps_0": protocol["eight_step_resource_probe"]["optimizer_steps"] == 8 and protocol["eight_step_resource_probe"]["formal_optimizer_steps"] == 0,
        "route_boundaries_exact": not middle_route(30000) and middle_route(30001) and middle_route(200000) and not middle_route(200001),
        "synthetic_cpu_audit_passed": synthetic["passed"],
        "probe_source_cpu_preflight_passed": probe_preflight["passed"],
    }
    return {
        "schema": "ev-uav-middle-multiscale-temporal-summary-command-audit-v1",
        "created_utc": utc_now(),
        "passed": all(checks.values()),
        "checks": checks,
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": protocol_sha,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cache_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
        "source_evidence": source_evidence,
        "folds": folds,
        "first_fold": first,
        "dependencies": dependencies,
        "synthetic_cpu_audit": synthetic,
        "probe_source_cpu_preflight": probe_preflight,
        "validation_or_test_read": False,
        "gpu_or_cuda_initialized": False,
        "formal_training_steps": 0,
    }


def run_audit():
    if AUDIT_PATH.exists() or PROBE_PATH.exists():
        raise FileExistsError("refusing to overwrite middle command audit/probe")
    payload = build_audit_payload()
    if not payload["passed"]:
        raise RuntimeError("middle CPU audit failed")
    digest = write_new_json(AUDIT_PATH, payload)
    return {**payload, "receipt_sha256": digest}


def require_audit():
    payload, digest = load_json_snapshot(AUDIT_PATH)
    if payload.get("schema") != "ev-uav-middle-multiscale-temporal-summary-command-audit-v1" or payload.get("passed") is not True:
        raise RuntimeError("middle command audit is absent or failed")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("middle command audit protocol mismatch")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("middle command audit runner mismatch")
    return payload, digest


def _python_gpu_rows():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return rows, [line for line in rows if "python.exe" in line.lower() or "pythonw.exe" in line.lower()]


def _parameter_snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.named_parameters()}


def run_probe(root_authorized_gpu=False):
    if not root_authorized_gpu:
        raise PermissionError("middle eight-step GPU probe requires explicit root authorization")
    protocol, protocol_sha = load_protocol()
    audit, audit_sha = require_audit()
    if PROBE_PATH.exists() or PROBE_PATH.parent.exists():
        raise FileExistsError("refusing to overwrite immutable middle probe")
    rows, python_rows = _python_gpu_rows()
    if python_rows:
        raise RuntimeError("another Python GPU process is active: {}".format(python_rows))

    import numpy as np
    import torch

    import run_h2_atomic_component_deletion_v3 as atomic
    import run_h2_multiscale_temporal_pyramid_probe as helper
    from model.h2_multiscale_temporal_pyramid_expert import (
        FrozenM20MultiScalePyramidAdapter,
        pyramid_expert_parameter_count,
    )
    from utils.atomic_component_deletion import extract_atomic_components, pure_false_positive_targets
    from utils.h2_multiscale_pyramid_loss import (
        PyramidDualState,
        multiscale_pyramid_constrained_loss,
        validate_pyramid_step_diagnostics,
    )
    from utils.postprocess import ChallengePostprocessor

    probe = protocol["eight_step_resource_probe"]
    evidence = audit["source_evidence"][probe["source"]]
    source_path = WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train" / probe["source"]
    if sha256_file(source_path) != evidence["source_sha256"]:
        raise RuntimeError("middle probe source SHA changed")
    started = time.perf_counter()
    payload = None
    with atomic.gpu_run_lock("middle_multiscale_temporal_summary_unique_probe"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(protocol["training"]["seed"])
        np.random.seed(protocol["training"]["seed"])
        m20, checkpoint = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        if pyramid_expert_parameter_count(adapter) != 3381:
            raise RuntimeError("middle pyramid parameter count changed")
        if any(parameter.requires_grad for parameter in m20.parameters()):
            raise RuntimeError("released M20 is not frozen")
        video, polarities, locations4 = atomic._load_input_only(source_path)
        labels, target_ids = atomic._load_truth(source_path)
        if len(polarities) != evidence["event_count"] or not middle_route(len(polarities)):
            raise RuntimeError("probe source escaped the middle input-only route")
        if len(video.event_indices_by_bin) != TEMPORAL_COUNT:
            raise RuntimeError("middle probe source is not complete T160")
        memory = helper.full_stream_memory(m20, video, device)
        observations_cpu, raw_scores, first_decoder_bins = helper.stream_observations_and_scores(adapter, video, memory, device)
        summary_cache = helper.build_summary_cache(observations_cpu, device)
        del observations_cpu
        cfg, effective_c00 = helper.build_c00()
        if hashlib.sha256(json.dumps(effective_c00, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() != EXPECTED_C00_SHA256:
            raise RuntimeError("effective C00 changed")
        processed, post_stats = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD, event_count=len(labels)).apply(
            torch.from_numpy(raw_scores.copy()), torch.from_numpy(locations4).long()
        )
        base_scores = processed.numpy().astype(np.float32, copy=True)
        components = extract_atomic_components(base_scores, locations4, PREDICTION_THRESHOLD, spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=0)
        targets = pure_false_positive_targets(components.event_indices, labels)
        pure_fp = tuple(components.event_indices[index] for index in np.flatnonzero(targets == 1))
        old_values = (helper.SEED, helper.PROBE_STEPS, helper.TEMPORAL_COUNT, helper.VIEW_BINS)
        helper.SEED = int(protocol["training"]["seed"])
        helper.PROBE_STEPS = PROBE_STEPS
        helper.TEMPORAL_COUNT = TEMPORAL_COUNT
        helper.VIEW_BINS = VIEW_BINS
        try:
            views, eligible_view_count = helper.prepare_view_metadata(video, labels, target_ids, pure_fp)
        finally:
            helper.SEED, helper.PROBE_STEPS, helper.TEMPORAL_COUNT, helper.VIEW_BINS = old_values
        adapter.train()
        optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=protocol["training"]["learning_rate"], weight_decay=protocol["training"]["weight_decay"])
        dual = PyramidDualState()
        before = _parameter_snapshot(adapter.expert)
        cumulative = {name: 0.0 for name, _ in adapter.expert.named_parameters()}
        records = []
        initial_identity = False
        second_decoder_bins = 0
        for step, metadata in enumerate(views, start=1):
            start, stop = metadata["start"], metadata["stop"]
            frames = atomic._frame_tensor(video, range(start, stop), device)
            decoder, base_logits, centre = adapter.decode_frozen_features(frames, memory[start:stop])
            summaries = tuple(value[start:stop].to(device=device, dtype=torch.float32) for value in summary_cache)
            optimizer.zero_grad(set_to_none=True)
            parts = adapter.expert(decoder.unsqueeze(0), base_logits.unsqueeze(0), centre.unsqueeze(0), tuple(value.unsqueeze(0) for value in summaries), return_parts=True)
            refined, sampled = helper.sample_dense_event_logits(parts.refined_logits.squeeze(0), video, start, stop)
            base_events, base_sampled = helper.sample_dense_event_logits(base_logits, video, start, stop)
            if not np.array_equal(sampled, metadata["global_indices"]) or not np.array_equal(base_sampled, sampled):
                raise RuntimeError("middle probe event sampling order changed")
            label_tensor = torch.from_numpy(metadata["labels"]).to(device=device, dtype=torch.float32)
            target_tensor = torch.from_numpy(metadata["target_ids"]).to(device=device, dtype=torch.long)
            time_tensor = torch.from_numpy(metadata["times"]).to(device=device, dtype=torch.long)
            loss, recall, suppression, diagnostics = multiscale_pyramid_constrained_loss(
                refined.float(), base_events.float(), label_tensor, target_tensor, time_tensor, metadata["hard_negative_components"], dual
            )
            if step == 1:
                initial_identity = bool(torch.equal(parts.refined_logits.detach(), base_logits.unsqueeze(0)) and int(torch.count_nonzero(parts.correction.detach())) == 0)
            if not torch.isfinite(loss):
                raise RuntimeError("middle probe loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), protocol["training"]["gradient_clip_norm"])
            step_grad = {}
            for name, parameter in adapter.expert.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("missing/non-finite expert gradient: {}".format(name))
                step_grad[name] = float(parameter.grad.detach().abs().sum())
                cumulative[name] += step_grad[name]
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in adapter.expert.parameters()):
                raise RuntimeError("middle probe produced a non-finite parameter")
            dual.update(recall, suppression)
            weights = parts.mixture_weights.detach().float()
            entropy = float((-(weights * weights.clamp_min(torch.finfo(weights.dtype).eps).log()).sum(dim=2)).mean())
            records.append({
                "step": step,
                "view_start_bin": start,
                "view_stop_bin_exclusive": stop,
                **diagnostics,
                "gradient_norm": float(gradient_norm),
                "output_projection_gradient_l1": step_grad["output_projection.weight"],
                "scale_encoder_gradient_l1": sum(value for name, value in step_grad.items() if name.startswith("scale_encoder.")),
                "mixture_projection_gradient_l1": sum(value for name, value in step_grad.items() if name.startswith("mixture_projection.")),
                "dual_target_time_recall_after": float(dual.target_time_recall),
                "dual_hard_negative_suppression_after": float(dual.hard_negative_suppression),
                "mixture_entropy": entropy,
                "correction_abs_mean": float(parts.correction.detach().float().abs().mean()),
            })
            second_decoder_bins += stop - start
        validate_pyramid_step_diagnostics(records, PROBE_STEPS)
        updates = {name: float((parameter.detach().cpu() - before[name]).abs().sum()) for name, parameter in adapter.expert.named_parameters()}
        m20_after = atomic.state_sha256(m20.state_dict())
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        checks = {
            "middle_input_route_true": middle_route(len(polarities)),
            "complete_T160_first_pass": first_decoder_bins == 160,
            "eight_views_second_pass_128_bins": second_decoder_bins == 128,
            "initial_bitwise_M20_identity": initial_identity,
            "released_M20_state_unchanged": m20_after == m20_before,
            "all_steps_have_target_and_hard_negative_constraints": all(record["target_time_group_count"] > 0 and record["hard_negative_component_count"] > 0 for record in records),
            "all_parameter_tensors_reached": all(value > 0 for value in cumulative.values()),
            "all_parameter_tensors_updated": all(value > 0 for value in updates.values()),
            "long_context_reached_by_step_8": sum(value for name, value in cumulative.items() if name.startswith("scale_encoder.")) > 0 and sum(value for name, value in cumulative.items() if name.startswith("mixture_projection.")) > 0,
            "peak_CUDA_within_budget": peak_mib <= probe["peak_CUDA_memory_limit_MiB"],
            "no_validation_or_test_read": True,
            "formal_optimizer_steps_zero": True,
        }
        payload = {
            "schema": "ev-uav-middle-multiscale-temporal-summary-eight-step-probe-v1",
            "created_utc": utc_now(),
            "passed": all(checks.values()),
            "checks": checks,
            "protocol_sha256": protocol_sha,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "command_audit_sha256": audit_sha,
            "source": probe["source"],
            "source_sha256": evidence["source_sha256"],
            "source_event_count": evidence["event_count"],
            "source_is_first_fold_fit_only": True,
            "other_source_array_read": False,
            "validation_or_test_read": False,
            "optimizer_steps": PROBE_STEPS,
            "formal_optimizer_steps": 0,
            "eligible_fit_only_view_count": eligible_view_count,
            "selected_view_starts": [record["view_start_bin"] for record in records],
            "effective_C00": effective_c00,
            "base_C00_stats": asdict(post_stats),
            "base_component_count": len(components.event_indices),
            "pure_FP_component_count": len(pure_fp),
            "trainable_parameter_count": pyramid_expert_parameter_count(adapter),
            "parameter_tensor_count": len(cumulative),
            "cumulative_gradient_l1": cumulative,
            "parameter_update_l1": updates,
            "all_step_diagnostics": records,
            "dual_state_after": dual.to_dict(),
            "released_m20_state_sha256_before": m20_before,
            "released_m20_state_sha256_after": m20_after,
            "gpu_preflight": {"snapshot": rows, "other_python_compute_processes": python_rows, "passed": not python_rows},
            "peak_CUDA_MiB": peak_mib,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not payload["passed"]:
            raise RuntimeError("middle eight-step probe failed: {}".format({key: value for key, value in checks.items() if not value}))
        del adapter, m20, checkpoint, memory, summary_cache, optimizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        payload["CUDA_allocated_after_release_MiB"] = torch.cuda.memory_allocated() / (1024.0 ** 2)
    digest = write_new_json(PROBE_PATH, payload)
    return {**payload, "receipt_sha256": digest}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Run the immutable CPU-only middle protocol audit.")
    subparsers.add_parser("plan-first-fold", help="Print the frozen first-fold plan without reading arrays.")
    probe = subparsers.add_parser("probe", help="Run the unique eight-step fit-only GPU resource probe.")
    probe.add_argument(GPU_FLAG, action="store_true", dest="root_authorized_gpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = run_audit()
    elif args.command == "plan-first-fold":
        protocol, protocol_sha = load_protocol()
        result = {"protocol_sha256": protocol_sha, "first_fold": first_fold_spec(protocol), "formal_training_started": False}
    elif args.command == "probe":
        result = run_probe(args.root_authorized_gpu)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

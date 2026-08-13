"""One-shot local Val24 replay for the train-selected H2 Stage1-only candidate.

The all11 H2 OOF calibration found that the multiscale Stage1 suppressor was
strongly positive while the optional component-recovery overlay lowered OOF
Score.  This runner therefore deploys only the fresh all11 Stage1 state.  It
has no threshold, source, or model-selection CLI knobs: the decision to omit
Stage2 is fixed by the saved train-only OOF result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import crossfit_component_reranker as crossfit
import replay_temporal_memory_validation as replay
import run_h2_atomic_component_deletion_v3 as atomic
import run_h2_pyramid_component_recovery_v2 as promoted
import run_h2_pyramid_component_recovery_v2_all11_final as final_refit
from model.h2_multiscale_temporal_pyramid_expert import FrozenM20MultiScalePyramidAdapter
from utils.target_preserving_residual import use_h2_residual_refiner


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
VAL_ROOT = (WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "val").resolve()
OUTPUT_ROOT = WORKSPACE / "experiments" / "20260812_h2_pyramid_stage1_only_val24"
CPU_AUDIT = OUTPUT_ROOT / "cpu_audit.json"
RUN_RESULT = OUTPUT_ROOT / "run_result.json"
H2_CACHE = OUTPUT_ROOT / "h2_stage1_raw_cache.pt"

M10_CACHE = (
    WORKSPACE
    / "experiments"
    / "20260810_dacc_v2_projection_only_seed49"
    / "replay"
    / "m10_val24_raw.pt"
)
M20_CACHE = (
    WORKSPACE
    / "experiments"
    / "20260810_baseline_fine_sweep"
    / "m20_val24_raw.pt"
)
STAGE1_CHECKPOINT = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_pyramid_component_recovery_v2_all11_final"
    / "stage1_all11"
    / "final_stage1.pt"
)
STAGE1_RECEIPT = STAGE1_CHECKPOINT.parent / "training_receipt.json"
OOF_RESULT = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_pyramid_component_recovery_v2_all11_final"
    / "stage2_all11"
    / "oof_calibration.json"
)

EXPECTED_STAGE1_SHA256 = "3d9a331feaf45c9c0aeb559b4265e0e921df3e53cbaeae3204eacd031cfa42ba"
EXPECTED_STAGE1_RECEIPT_SHA256 = "f09a7bc29032b9714014771e827bf29e5d739a08b798f2ea3a32fd275c6113dd"
EXPECTED_OOF_SHA256 = "c207204bed3b809696a93560ed262ac98498a7090d9993b0282b402eb78fa5a5"
EXPECTED_M10_CACHE_SHA256 = "96a9dfa8833e6f609d29f4db9d8f7196c84c7e92c7026cce734b97ddf133622f"
EXPECTED_M20_CACHE_SHA256 = "6c9b4a8e33217aac7a05c78590a7feb6db6e6fc332b6411d7603264687710304"
EXPECTED_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
EXPECTED_M20_STATE_SHA256 = "feb234e530688a11865e0d49b58a9f54806f69ea63a9de3e01e8b9f714a6113d"
EXPECTED_VAL_STEMS = tuple("val_{:03d}.npz".format(index) for index in range(24))
LOW_EVENT_COUNT_MAX = 30000
LOW_THRESHOLD = 0.718
M20_THRESHOLD = 0.719
GPU_FLAG = "--root-authorized-gpu"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    digest = sha256_file(path)
    with Path(str(path) + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write("{}  {}\n".format(digest, path.name))
    return digest


def _expect(condition, message):
    if not condition:
        raise RuntimeError(message)


def _verify_sidecar(path, expected=None):
    path = Path(path)
    digest = sha256_file(path)
    if expected is not None:
        _expect(digest == expected, "frozen artifact changed: {}".format(path))
    tokens = Path(str(path) + ".sha256").read_text(encoding="ascii").split()
    _expect(len(tokens) == 2 and tokens[0] == digest and tokens[1] == path.name, "sidecar mismatch: {}".format(path))
    return digest


def _delta(candidate, baseline):
    return {name: float(candidate[name]) - float(baseline[name]) for name in baseline}


def _load_stage1():
    _verify_sidecar(STAGE1_CHECKPOINT, EXPECTED_STAGE1_SHA256)
    _verify_sidecar(STAGE1_RECEIPT, EXPECTED_STAGE1_RECEIPT_SHA256)
    _verify_sidecar(OOF_RESULT, EXPECTED_OOF_SHA256)
    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location="cpu", weights_only=False)
    receipt = json.loads(STAGE1_RECEIPT.read_text(encoding="utf-8"))
    oof = json.loads(OOF_RESULT.read_text(encoding="utf-8"))
    _expect(checkpoint.get("schema") == "ev-uav-h2-pyramid-recovery-v2-all11-stage1-v1", "unexpected Stage1 schema")
    _expect(tuple(checkpoint.get("fit_sources", ())) == tuple("train_{:03d}.npz".format(index) for index in range(88, 99)), "Stage1 source set changed")
    _expect(checkpoint.get("optimizer_steps") == 88, "Stage1 optimizer steps changed")
    _expect(receipt.get("all_expert_parameter_tensors_updated") is True, "Stage1 update audit failed")
    _expect(receipt.get("released_m20_state_sha256_before") == EXPECTED_M20_STATE_SHA256, "Stage1 M20 binding changed")
    _expect(receipt.get("released_m20_state_sha256_after") == EXPECTED_M20_STATE_SHA256, "Stage1 changed M20")
    _expect(receipt.get("validation_or_test_read") is False, "Stage1 receipt val/test flag changed")
    selected = oof.get("selected_cutoff", {}).get("evaluation", {})
    recovery = selected.get("Stage2_recovery_vs_Stage1", {})
    _expect(float(recovery.get("Score", 0.0)) < 0.0, "Stage2-only omission is no longer train-only OOF justified")
    stage1_gain = selected.get("Stage1_delta_vs_M20", {})
    _expect(float(stage1_gain.get("Score", 0.0)) > 0.0, "Stage1-only candidate lacks positive train-only OOF evidence")
    return checkpoint, receipt, oof


def _load_baseline_records():
    _expect(sha256_file(M10_CACHE) == EXPECTED_M10_CACHE_SHA256, "M10 cache changed")
    _expect(sha256_file(M20_CACHE) == EXPECTED_M20_CACHE_SHA256, "M20 cache changed")
    m10, _ = replay.load_cache_snapshot(M10_CACHE)
    m20, _ = replay.load_cache_snapshot(M20_CACHE)
    records = replay.route_cache_records(m20, m10, LOW_EVENT_COUNT_MAX)
    _expect(tuple(Path(record.file_name).name for record in records) == EXPECTED_VAL_STEMS, "Val24 cache order changed")
    return records


def _input_path(file_name):
    path = (VAL_ROOT / Path(file_name).name).resolve()
    _expect(path.parent == VAL_ROOT and path.is_file(), "validation source missing: {}".format(file_name))
    return path


def _h2_decision(polarities):
    return bool(use_h2_residual_refiner(int(np.asarray(polarities).size), polarities))


def cpu_audit(_args):
    if CPU_AUDIT.exists() or RUN_RESULT.exists() or H2_CACHE.exists():
        raise FileExistsError("Stage1-only validation output already exists")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU audit")
    checkpoint, receipt, oof = _load_stage1()
    _expect(sha256_file(M10_CACHE) == EXPECTED_M10_CACHE_SHA256, "M10 cache changed")
    _expect(sha256_file(M20_CACHE) == EXPECTED_M20_CACHE_SHA256, "M20 cache changed")
    cfg, effective_c00 = final_refit.build_c00()
    _expect(crossfit.sha256_json(effective_c00) == EXPECTED_C00_SHA256, "effective C00 changed")
    m20, _ = atomic.build_released_m20(torch.device("cpu"))
    adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).cpu()
    adapter.expert.load_state_dict(checkpoint["expert_state_dict"], strict=True)
    _expect(atomic.state_sha256(m20.state_dict()) == EXPECTED_M20_STATE_SHA256, "CPU M20 state changed")
    _expect(not any(parameter.requires_grad for parameter in m20.parameters()), "M20 should be frozen")
    _expect(all(not parameter.requires_grad for parameter in adapter.released_m20.parameters()), "adapter exposed trainable M20")
    _expect(torch.cuda.is_initialized() is False, "CPU audit initialized CUDA")
    payload = {
        "schema": "ev-uav-h2-pyramid-stage1-only-val24-cpu-audit-v1",
        "created_utc": utc_now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "stage1_checkpoint_sha256": EXPECTED_STAGE1_SHA256,
        "stage1_receipt_sha256": EXPECTED_STAGE1_RECEIPT_SHA256,
        "all11_oof_sha256": EXPECTED_OOF_SHA256,
        "stage1_oof_score_gain_vs_M20": oof["selected_cutoff"]["evaluation"]["Stage1_delta_vs_M20"]["Score"],
        "stage2_oof_score_delta_vs_Stage1": oof["selected_cutoff"]["evaluation"]["Stage2_recovery_vs_Stage1"]["Score"],
        "stage2_deployment_disabled_by_train_only_oof": True,
        "M10_cache_sha256": EXPECTED_M10_CACHE_SHA256,
        "M20_cache_sha256": EXPECTED_M20_CACHE_SHA256,
        "effective_C00_sha256": EXPECTED_C00_SHA256,
        "dataset_arrays_read": False,
        "validation_or_test_read": False,
        "CUDA_initialized": False,
        "passed": True,
    }
    del adapter, m20, receipt
    digest = write_json_exclusive(CPU_AUDIT, payload)
    print(json.dumps({"stage": "cpu_audit_complete", "sha256": digest, "passed": True}, indent=2))


def _cpu_audit_gate():
    digest = _verify_sidecar(CPU_AUDIT)
    audit = json.loads(CPU_AUDIT.read_text(encoding="utf-8"))
    _expect(audit.get("passed") is True, "CPU audit failed")
    _expect(audit.get("runner_sha256") == sha256_file(Path(__file__)), "runner changed after CPU audit")
    _expect(audit.get("validation_or_test_read") is False, "CPU audit touched val/test")
    return digest


def run(_args):
    if RUN_RESULT.exists() or H2_CACHE.exists():
        raise FileExistsError("Stage1-only Val24 replay already consumed")
    audit_sha = _cpu_audit_gate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint, receipt, oof = _load_stage1()
    baseline_records = _load_baseline_records()
    cfg, effective_c00 = final_refit.build_c00()
    _expect(crossfit.sha256_json(effective_c00) == EXPECTED_C00_SHA256, "effective C00 changed")
    source_hashes = {record.file_name: sha256_file(_input_path(record.file_name)) for record in baseline_records}
    candidate_counts = []
    baseline_counts = []
    per_video = []
    h2_records = []
    h2_calls = 0
    non_h2_identity = True
    started = utc_now()

    with atomic.gpu_run_lock("h2_pyramid_stage1_only_val24"):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m20, _ = atomic.build_released_m20(device)
        m20_before = atomic.state_sha256(m20.state_dict())
        adapter = FrozenM20MultiScalePyramidAdapter(m20, context_bins=5).to(device)
        adapter.expert.load_state_dict(checkpoint["expert_state_dict"], strict=True)
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)

        for index, baseline in enumerate(baseline_records, start=1):
            path = _input_path(baseline.file_name)
            video, polarities, locations4 = atomic._load_input_only(path)
            _expect(locations4.shape[0] == baseline.event_count, "event count mismatch: {}".format(baseline.file_name))
            _expect(np.array_equal(locations4, baseline.locs.detach().cpu().numpy()), "location mismatch: {}".format(baseline.file_name))
            h2 = _h2_decision(polarities)
            expected_h2 = int(baseline.event_count) > 200000 and min(float(np.mean(np.asarray(polarities) > 0.5)), 1.0 - float(np.mean(np.asarray(polarities) > 0.5))) >= 0.20
            _expect(h2 == expected_h2, "H2 input route mismatch: {}".format(baseline.file_name))
            if h2:
                h2_calls += 1
                cache = promoted.build_input_only_source_cache(adapter, video, polarities, locations4, cfg, device)
                base_raw = np.asarray(cache["base_raw"], dtype=np.float32)
                raw_cache_scores = baseline.scores.detach().cpu().numpy().astype(np.float32, copy=False)
                _expect(np.array_equal(base_raw.view(np.uint32), raw_cache_scores.view(np.uint32)), "released M20 raw mismatch: {}".format(baseline.file_name))
                candidate_scores = torch.from_numpy(np.asarray(cache["stage1_raw"], dtype=np.float32).copy()).contiguous()
                h2_records.append(
                    {
                        "file_name": baseline.file_name,
                        "event_count": int(baseline.event_count),
                        "source_file_sha256": source_hashes[baseline.file_name],
                        "base_raw_scores": base_raw,
                        "stage1_raw_scores": candidate_scores.numpy().copy(),
                        "contains_labels_or_target_ids": False,
                        "stage1_C00_stats": cache["stage1_C00_stats"],
                    }
                )
                preserved = False
                del cache
            else:
                candidate_scores = baseline.scores
                preserved = candidate_scores.data_ptr() == baseline.scores.data_ptr() and torch.equal(candidate_scores, baseline.scores)
                non_h2_identity = non_h2_identity and preserved
            candidate = replay.RoutedRecord(
                file_name=baseline.file_name,
                event_count=baseline.event_count,
                scores=candidate_scores,
                seg_label=baseline.seg_label,
                locs=baseline.locs,
                idx_label=baseline.idx_label,
                source_sha256=baseline.source_sha256,
                score_source="h2_stage1" if h2 else baseline.score_source,
            )
            threshold = LOW_THRESHOLD if baseline.event_count <= LOW_EVENT_COUNT_MAX else M20_THRESHOLD
            baseline_counts.append(replay.evaluate_cached_video(baseline, threshold, cfg))
            candidate_counts.append(replay.evaluate_cached_video(candidate, threshold, cfg))
            per_video.append(
                {
                    "file_name": baseline.file_name,
                    "event_count": int(baseline.event_count),
                    "routed_to_h2_stage1": h2,
                    "raw_scores_bitwise_preserved": preserved,
                    "baseline_counts": asdict(baseline_counts[-1]),
                    "candidate_counts": asdict(candidate_counts[-1]),
                }
            )
            print("stage1 route/evaluate {}/24: {} -> {}".format(index, baseline.file_name, "H2" if h2 else "identity"), flush=True)
            del video, polarities, locations4, candidate_scores, candidate
            torch.cuda.empty_cache()
        _expect(atomic.state_sha256(m20.state_dict()) == m20_before, "M20 changed during Stage1 Val24 replay")
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        del adapter, m20
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mib = torch.cuda.memory_allocated() / (1024.0 ** 2)

    _expect(h2_calls == 1, "expected exactly one H2 validation input")
    _expect(non_h2_identity, "a non-H2 score vector changed")
    cache_payload = {
        "schema": "ev-uav-h2-pyramid-stage1-only-val24-h2-cache-v1",
        "created_utc": utc_now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "stage1_checkpoint_sha256": EXPECTED_STAGE1_SHA256,
        "records": h2_records,
        "record_count": len(h2_records),
        "validation_or_test_read": True,
    }
    replay._atomic_torch_save(cache_payload, H2_CACHE, overwrite=False)
    h2_cache_sha = sha256_file(H2_CACHE)
    baseline_total = replay._sum_counts(baseline_counts)
    candidate_total = replay._sum_counts(candidate_counts)
    baseline_metrics = replay.metrics_from_counts_exact(baseline_total, cfg).to_dict()
    candidate_metrics = replay.metrics_from_counts_exact(candidate_total, cfg).to_dict()
    delta = _delta(candidate_metrics, baseline_metrics)
    gates = {
        "baseline_matches_golden": baseline_metrics["score"] == 0.9628776541559201,
        "non_h2_bitwise_identity": non_h2_identity,
        "exactly_one_h2_call": h2_calls == 1,
        "candidate_score_improves": delta["score"] > 0.0,
        "candidate_iou_not_lower": delta["iou"] >= 0.0,
        "candidate_fa_not_higher": delta["fa"] <= 0.0,
    }
    payload = {
        "schema": "ev-uav-h2-pyramid-stage1-only-val24-report-v1",
        "created_utc": utc_now(),
        "started_utc": started,
        "runner_sha256": sha256_file(Path(__file__)),
        "cpu_audit_sha256": audit_sha,
        "stage1_checkpoint_sha256": EXPECTED_STAGE1_SHA256,
        "stage1_receipt_sha256": EXPECTED_STAGE1_RECEIPT_SHA256,
        "all11_oof_sha256": EXPECTED_OOF_SHA256,
        "stage2_disabled_by_train_only_oof": True,
        "source_hashes": source_hashes,
        "h2_cache_sha256": h2_cache_sha,
        "route_summary": {"h2_calls": h2_calls, "non_h2_bitwise_identity": non_h2_identity},
        "per_video": per_video,
        "aggregate": {
            "baseline": {"counts": asdict(baseline_total), "metrics": baseline_metrics},
            "candidate": {"counts": asdict(candidate_total), "metrics": candidate_metrics},
            "delta": delta,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "peak_CUDA_MiB": peak_mib,
        "CUDA_after_release_MiB": after_mib,
        "validation_or_test_read": True,
        "test_read": False,
    }
    digest = write_json_exclusive(RUN_RESULT, payload)
    print(json.dumps({"stage": "val24_stage1_only_complete", "report_sha256": digest, "delta": delta, "gates": gates, "passed": all(gates.values())}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("cpu-audit")
    audit.set_defaults(handler=cpu_audit)
    execute = commands.add_parser("run")
    execute.add_argument(GPU_FLAG, dest="root_authorized_gpu", action="store_true")
    execute.set_defaults(handler=run)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run" and not args.root_authorized_gpu:
        raise PermissionError("run requires {}".format(GPU_FLAG))
    args.handler(args)


if __name__ == "__main__":
    main()

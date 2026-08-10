"""One-shot validation-blind replay for the train-selected W_full checkpoint.

Stages are deliberately separated. ``freeze`` and ``preflight`` are CPU-only
and must not touch a validation NPZ, either golden score cache, the official
manifest payload, labels, or the golden report. ``runtime-preflight`` and
``run`` require an explicit root authorization flag. ``run`` creates the sole
O_EXCL attempt claim before the first deferred read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence


SCIENCE_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-science-v1"
EXECUTION_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-execution-v1"
CPU_RECEIPT_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-cpu-preflight-v1"
RUNTIME_RECEIPT_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-runtime-preflight-v1"
CLAIM_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-claim-v1"
H2_CACHE_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-h2-cache-v1"
REPORT_SCHEMA = "ev-uav-metric-aux-task-arithmetic-wfull-val24-report-v1"

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SCIENCE_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "protocols"
    / "metric_aux_task_arithmetic_wfull_val24_science_v1.json"
).resolve()
EXPECTED_SCIENCE_PROTOCOL_SHA256 = (
    "fabf6c622a0b4d07905a522c52dee67eb76b50f6a625be4acdc349638ab5b1e0"
)
EXPERIMENT_DIRECTORY = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_metric_aux_task_arithmetic_wfull_val24_v1"
).resolve()

OFFICIAL_VIDEO_COUNT = 24
OFFICIAL_EVENT_COUNT = 1_424_330
OFFICIAL_STEMS = tuple("val_{:03d}".format(index) for index in range(24))
OFFICIAL_DATASET_SIGNATURE = "bedba93c1d523f58c35da6399219df1b98e6240f92d093520fa0f4961d927274"
OFFICIAL_MANIFEST_SHA256 = "c7c574b5dfa8336fe50917581544b5e4991b2cde197f31c9a5bee05a29e336d4"
OFFICIAL_SEMANTIC_SHA256 = "d780da17e69446b988b1b5fae7954855d5ce66a32aa7b9581eeb3e4a0563f83f"
M10_CHECKPOINT_SHA256 = "5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935"
M20_CHECKPOINT_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
M10_CACHE_SHA256 = "96a9dfa8833e6f609d29f4db9d8f7196c84c7e92c7026cce734b97ddf133622f"
M20_CACHE_SHA256 = "6c9b4a8e33217aac7a05c78590a7feb6db6e6fc332b6411d7603264687710304"
GOLDEN_REPORT_SHA256 = "da6004ddd22731b8e848c9ed0c561961abbc04b4e3f66cd07b1e085d26f9f383"
WFULL_CHECKPOINT_SHA256 = "614999c09f82ec1911620ee35dae3f1f6362cb92d59a82e2e539e9b2ad2432ee"
PUBLISHED_VALIDATION_CONTRACT_SHA256 = (
    "54dcb6d9b8e535110113a05a31c66bf50af39919d82a6d4d63c676d5d491187f"
)
EXPECTED_EFFECTIVE_C00_SHA256 = (
    "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
)

LOW_EVENT_COUNT_MAX = 30_000
HIGH_EVENT_COUNT_MAX = 200_000
POLARITY_MINORITY_CUTOFF = 0.20
LOW_THRESHOLD = 0.718
M20_THRESHOLD = 0.719
MATERIALITY_SCORE_DELTA = 0.00005

GOLDEN_COUNTS = {
    "true_positive_events": 63981,
    "false_positive_events": 2396,
    "positive_events": 65506,
    "detected_target_frames": 4649,
    "target_frames": 4762,
    "false_components": 1584,
    "frame_count": 3752,
}
GOLDEN_METRICS = {
    "iou": 0.9422550201416016,
    "acc": 0.9767196774482727,
    "pd": 0.9762704745905082,
    "fa": 4.69291729752432e-06,
    "score_fa": 0.9541549751552311,
    "score": 0.9628776541559201,
}

EXPECTED_RUNTIME = {
    "python_version": "3.9.25",
    "numpy_version": "1.26.4",
    "opencv_version": "4.8.1",
    "opencv_build_sha256": "173c080cc486d36465d7dcbe73e6a921c4c55fb56ee67b0cc2dad09ddd43f4f4",
    "torch_version": "2.5.1+cu121",
    "platform": "Windows-10-10.0.26100-SP0",
}

C00_SETTINGS = {
    "prediction_threshold": M20_THRESHOLD,
    "roc": True,
    "correct_thresh": 0.0001,
    "res": [346, 260],
    "pd_detT": 50,
    "temporal_memory_enabled": True,
    "temporal_memory_sparse_weight": 0.0,
    "temporal_memory_temporal_attention_enabled": True,
    "temporal_frame_enabled": False,
    "dense_expert_enabled": False,
    "ensemble_enabled": False,
    "temporal_memory_blend_model_path": "",
    "temporal_memory_secondary_model_path": "",
    "temporal_memory_secondary_max_event_count": 0,
    "temporal_memory_primary_weight": 1.0,
    "p0_enabled": True,
    "p0_spatial_radius": 2,
    "p0_temporal_bin_size": 50,
    "p0_temporal_radius_bins": 1,
    "p0_min_cluster_events": 3,
    "p0_min_duration_bins": 5,
    "p0c_high_confidence_recovery_enabled": True,
    "p0c_retain_min_score": 0.95,
    "p0c_density_retain_enabled": False,
    "p0c_density_event_count_cutoff": 100000,
    "p0c_density_retain_min_score": 0.97,
    "component_reranker_enabled": False,
    "component_reranker_event_count_cutoff": 100000,
    "p0b_enabled": False,
    "p18_score_track_recovery_enabled": True,
    "p18_event_count_cutoff": 1,
    "p18_max_event_count": 35000,
    "p18_candidate_floor": 0.53,
    "p18_spatial_radius": 5,
    "p18_temporal_bin_size": 50,
    "p18_max_link_distance": 8.0,
    "p18_max_gap_bins": 1,
    "p18_min_track_bins": 4,
    "p18_restore_mode": "best",
    "p18_max_restore_events_per_component": 0,
    "p6_density_threshold_enabled": True,
    "p6_event_count_cutoff": LOW_EVENT_COUNT_MAX,
    "p6_low_density_threshold": LOW_THRESHOLD,
    "p6_high_density_threshold": M20_THRESHOLD,
}

CODE_PATHS = (
    "evaluate_metric_aux_task_arithmetic_validation.py",
    "protocols/metric_aux_task_arithmetic_wfull_val24_science_v1.json",
    "crossfit_component_reranker.py",
    "train_component_reranker.py",
    "replay_temporal_memory_validation.py",
    "dataset/temporal_frame.py",
    "model/modules/confidence_head.py",
    "model/temporal_frame_net.py",
    "model/temporal_memory_net.py",
    "utils/challenge_eval.py",
    "utils/component_reranker.py",
    "utils/density_threshold.py",
    "utils/eval.py",
    "utils/multiscale_motion.py",
    "utils/postprocess.py",
    "utils/temporal_memory_inference.py",
)

TRAIN_INPUT_NAMES = (
    "v5_protocol",
    "v5_runner",
    "v5_command_audit",
    "v5_synthesis_manifest",
    "v5_grouped_oof_report",
    "all11_v2_protocol",
    "all11_v2_runner",
    "all11_v2_command_audit",
    "all11_v2_pair_audit",
    "all11_v2_synthesis_manifest",
    "wfull_checkpoint",
)
BASELINE_INPUT_NAMES = ("m10_checkpoint", "m20_checkpoint")
DEFERRED_INPUT_NAMES = frozenset(
    {"m10_golden_cache", "m20_golden_cache", "golden_report", "official_manifest"}
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value, name):
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("{} must be a lowercase SHA-256 digest.".format(name))
    return value


def _workspace_path(relative_path, description):
    text = str(relative_path).replace("\\", "/")
    path = (WORKSPACE_ROOT / text).resolve()
    if path != WORKSPACE_ROOT and WORKSPACE_ROOT not in path.parents:
        raise ValueError("{} escapes the workspace: {}".format(description, path))
    return path


def _load_json_snapshot(path, expected_sha256=None, description="JSON"):
    path = Path(path).resolve()
    before = sha256_file(path)
    if expected_sha256 is not None and before != _require_sha256(expected_sha256, description):
        raise ValueError("{} differs from its frozen SHA-256.".format(description))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    after = sha256_file(path)
    if after != before:
        raise RuntimeError("{} changed while being read.".format(description))
    return payload, before


def _atomic_json_no_clobber(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite immutable JSON: {}".format(path))
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _git_state():
    def run(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
        )
        return result.stdout

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": run("rev-parse", "HEAD").decode("ascii").strip().lower(),
        "clean": not bool(status.strip()),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _paths():
    directory = Path(EXPERIMENT_DIRECTORY).resolve()
    return {
        "science_protocol": Path(SCIENCE_PROTOCOL_PATH).resolve(),
        "execution_protocol": directory / "execution_protocol.json",
        "cpu_receipt": directory / "cpu_preflight_receipt.json",
        "runtime_receipt": directory / "runtime_preflight_receipt.json",
        "claim": directory / "validation_attempt_claim.json",
        "h2_cache": directory / "raw_wfull_full_t160_h2_only.pt",
        "report": directory / "frozen_validation_report.json",
    }


def _code_sha256():
    result = {}
    for relative in CODE_PATHS:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError("Required code file is missing: {}".format(path))
        result[relative] = sha256_file(path)
    return result


def _semantic_manifest_sha256(entries):
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(Path(entry["path"]).name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(entry["sha256"])))
    return digest.hexdigest()


def _c00_config():
    return SimpleNamespace(**dict(C00_SETTINGS))


def _effective_c00_sha256():
    import crossfit_component_reranker as component_crossfit

    cfg = _c00_config()
    component_crossfit.validate_c00_config(cfg)
    return canonical_sha256(component_crossfit._postprocess_contract(cfg))


def _route_policy_definition():
    return {
        "inputs": ["complete_video_polarities", "temporal_bin_count"],
        "temporal_bin_count": 160,
        "low_event_count_max_inclusive": LOW_EVENT_COUNT_MAX,
        "high_event_count_max_inclusive": HIGH_EVENT_COUNT_MAX,
        "polarity_positive_test": "value>0.5",
        "polarity_minority_cutoff_inclusive_for_h2": POLARITY_MINORITY_CUTOFF,
        "low": {"score_source": "golden_m10", "threshold": LOW_THRESHOLD},
        "middle": {"score_source": "golden_m20", "threshold": M20_THRESHOLD},
        "h1": {"score_source": "golden_m20", "threshold": M20_THRESHOLD},
        "h2": {"score_source": "wfull", "mode": "full_stream_t160", "threshold": M20_THRESHOLD},
        "t32": False,
        "persistence": False,
    }


def route_policy_sha256():
    return canonical_sha256(_route_policy_definition())


@dataclass(frozen=True)
class WFullRouteDecision:
    domain: str
    event_count: int
    polarity_minority_fraction: float
    temporal_bin_count: int
    baseline_score_source: str
    candidate_score_source: str
    candidate_action: str
    prediction_threshold: float
    policy_sha256: str

    def to_metadata(self):
        return asdict(self)


def polarity_minority_fraction(polarities):
    import numpy as np

    values = np.asarray(polarities)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("Complete-video polarities must be a non-empty 1D vector.")
    if values.dtype.kind not in "biuf":
        raise TypeError("Polarities must be numeric.")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Polarities must be finite.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Normalized polarities must lie in [0, 1].")
    positives = int(np.count_nonzero(values > 0.5))
    return float(min(positives, int(values.size) - positives) / int(values.size))


def classify_wfull_route(polarities, temporal_bin_count):
    if isinstance(temporal_bin_count, bool) or int(temporal_bin_count) != 160:
        raise ValueError("The frozen route requires exactly 160 temporal bins.")
    fraction = polarity_minority_fraction(polarities)
    event_count = int(len(polarities))
    common = {
        "event_count": event_count,
        "polarity_minority_fraction": fraction,
        "temporal_bin_count": 160,
        "policy_sha256": route_policy_sha256(),
    }
    if event_count <= LOW_EVENT_COUNT_MAX:
        return WFullRouteDecision(
            domain="low", baseline_score_source="golden_m10", candidate_score_source="golden_m10",
            candidate_action="reuse_same_tensor", prediction_threshold=LOW_THRESHOLD, **common
        )
    if event_count <= HIGH_EVENT_COUNT_MAX:
        return WFullRouteDecision(
            domain="middle", baseline_score_source="golden_m20", candidate_score_source="golden_m20",
            candidate_action="reuse_same_tensor", prediction_threshold=M20_THRESHOLD, **common
        )
    if fraction < POLARITY_MINORITY_CUTOFF:
        return WFullRouteDecision(
            domain="h1", baseline_score_source="golden_m20", candidate_score_source="golden_m20",
            candidate_action="reuse_same_tensor", prediction_threshold=M20_THRESHOLD, **common
        )
    return WFullRouteDecision(
        domain="h2", baseline_score_source="golden_m20", candidate_score_source="wfull_full_t160",
        candidate_action="infer_wfull_full_stream_t160", prediction_threshold=M20_THRESHOLD, **common
    )


def choose_candidate_scores(decision, golden_scores, h2_predictor):
    domain = decision.domain if hasattr(decision, "domain") else decision["domain"]
    if domain == "h2":
        return h2_predictor(), False
    if domain not in {"low", "middle", "h1"}:
        raise ValueError("Unknown route domain: {}".format(domain))
    return golden_scores, True


def promotion_gate_results(
    baseline_counts,
    baseline_metrics,
    candidate_counts,
    candidate_metrics,
    h2_baseline_counts,
    h2_baseline_metrics,
    h2_candidate_counts,
    h2_candidate_metrics,
    preservation,
    inference_calls,
    h2_count,
    only_h2_called,
):
    population_names = ("positive_events", "target_frames", "frame_count")
    gates = {
        "golden_baseline_exact_match": baseline_counts == GOLDEN_COUNTS and baseline_metrics == GOLDEN_METRICS,
        "aggregate_score_strictly_greater_than_golden": float(candidate_metrics["score"]) > float(GOLDEN_METRICS["score"]),
        "aggregate_pd_not_lower_than_golden": float(candidate_metrics["pd"]) >= float(GOLDEN_METRICS["pd"]),
        "aggregate_iou_not_lower_than_golden": float(candidate_metrics["iou"]) >= float(GOLDEN_METRICS["iou"]),
        "aggregate_fa_not_higher_than_golden": float(candidate_metrics["fa"]) <= float(GOLDEN_METRICS["fa"]),
        "validation_h2_correct_objects_not_lower_than_golden_h2": int(h2_candidate_counts["detected_target_frames"]) >= int(h2_baseline_counts["detected_target_frames"]),
        "validation_h2_pd_not_lower_than_golden_h2": float(h2_candidate_metrics["pd"]) >= float(h2_baseline_metrics["pd"]),
        "population_invariants_equal": all(candidate_counts[name] == baseline_counts[name] for name in population_names),
        "all_non_h2_golden_tensor_object_storage_layout_bits_and_counts_preserved": bool(preservation),
        "wfull_full_t160_calls_equal_h2_count": int(h2_count) > 0 and int(inference_calls) == int(h2_count),
        "only_h2_calls_wfull": bool(only_h2_called),
        "route_population_exactly_one_h2_and_23_non_h2": int(h2_count) == 1,
        "t32_not_read_or_combined": True,
        "persistence_not_read_or_combined": True,
    }
    materiality_delta = float(candidate_metrics["score"]) - float(GOLDEN_METRICS["score"])
    return gates, {
        "aggregate_score_delta": materiality_delta,
        "threshold": MATERIALITY_SCORE_DELTA,
        "met": materiality_delta >= MATERIALITY_SCORE_DELTA,
        "included_in_safety_pass": False,
    }


def validate_science_protocol(protocol):
    if protocol.get("schema") != SCIENCE_SCHEMA:
        raise ValueError("Unexpected W_full validation science schema.")
    if protocol.get("status") != "frozen_before_any_wfull_val24_npz_cache_label_or_golden_report_access":
        raise ValueError("Science protocol is not frozen at the blind boundary.")
    if protocol.get("candidate_id") != "metric_aux_task_arithmetic_all11_alpha1_wfull_h2_full_t160":
        raise ValueError("Unexpected candidate id.")
    if protocol.get("attempt_budget", {}).get("full_val24_replays") != 1:
        raise ValueError("Exactly one validation replay must be preregistered.")
    if protocol.get("train_only_evidence", {}).get("wfull_checkpoint", {}).get("sha256") != WFULL_CHECKPOINT_SHA256:
        raise ValueError("W_full checkpoint identity differs.")
    if protocol.get("published_validation_contract", {}).get("sha256") != PUBLISHED_VALIDATION_CONTRACT_SHA256:
        raise ValueError("Published validation contract identity differs.")
    route = protocol.get("route_policy", {})
    if (
        route.get("polarity_minority_cutoff") != POLARITY_MINORITY_CUTOFF
        or route.get("t32_allowed") is not False
        or route.get("persistence_allowed") is not False
        or route.get("threshold_search_allowed") is not False
        or route.get("non_h2_golden_tensor_same_object_storage_layout_and_bits") is not True
        or route.get("expected_h2_video_count") != 1
        or route.get("expected_non_h2_video_count") != 23
    ):
        raise ValueError("Route policy differs from the frozen input-only policy.")
    inference = protocol.get("inference", {})
    expected_inference = {
        "mode": "full_stream_t160", "checkpoint": "wfull_checkpoint", "device": "cuda:0",
        "temporal_bin_size": 50, "temporal_bin_count": 160, "whole_t": 8000,
        "context_bins": 5, "network_width": 16, "sequence_length": 16,
        "resolution": [346, 260], "inference_batch_size": 8, "log_count_clip": 4.0,
        "prediction_threshold": M20_THRESHOLD, "window_length": None, "stride": None,
        "forbidden_apis": ["predict_temporal_memory_scores_windowed", "predict_persistence_component_keep_probabilities"],
    }
    if inference != expected_inference:
        raise ValueError("Full-stream T160 inference contract differs.")
    if protocol.get("postprocess", {}).get("effective_c00_canonical_sha256") != EXPECTED_EFFECTIVE_C00_SHA256:
        raise ValueError("Effective C00 identity differs.")
    if protocol.get("golden") != {"routing": protocol["golden"]["routing"], "counts": GOLDEN_COUNTS, "metrics": GOLDEN_METRICS}:
        raise ValueError("Golden baseline constants differ.")
    materiality = protocol.get("promotion_gates", {}).get("materiality_report_only", {})
    if materiality != {"aggregate_score_delta_at_least": MATERIALITY_SCORE_DELTA, "included_in_safety_pass": False}:
        raise ValueError("Materiality must remain report-only at 0.00005.")
    if protocol.get("preflight", {}).get("runtime_preflight_requires_explicit_root_authorization") is not True:
        raise ValueError("GPU runtime preflight must require explicit root authorization.")
    if protocol.get("preflight", {}).get("formal_run_requires_explicit_root_authorization") is not True:
        raise ValueError("Formal replay must require explicit root authorization.")
    return protocol


def _load_published_validation_contract(science):
    spec = science["published_validation_contract"]
    path = _workspace_path(spec["workspace_relative_path"], "published validation contract")
    payload, digest = _load_json_snapshot(path, spec["sha256"], "published validation contract")
    if payload.get("validation_dataset", {}).get("video_count") != OFFICIAL_VIDEO_COUNT:
        raise ValueError("Published validation video count differs.")
    if payload.get("validation_dataset", {}).get("event_count") != OFFICIAL_EVENT_COUNT:
        raise ValueError("Published validation event count differs.")
    if payload.get("validation_dataset", {}).get("dataset_signature") != OFFICIAL_DATASET_SIGNATURE:
        raise ValueError("Published validation dataset signature differs.")
    if payload.get("validation_dataset", {}).get("semantic_manifest_sha256") != OFFICIAL_SEMANTIC_SHA256:
        raise ValueError("Published validation semantic digest differs.")
    if payload.get("golden", {}).get("counts") != GOLDEN_COUNTS or payload.get("golden", {}).get("metrics") != GOLDEN_METRICS:
        raise ValueError("Published golden constants differ.")
    deferred = payload.get("deferred_validation_inputs", {})
    expected = {
        "m10_checkpoint": M10_CHECKPOINT_SHA256,
        "m20_checkpoint": M20_CHECKPOINT_SHA256,
        "m10_golden_cache": M10_CACHE_SHA256,
        "m20_golden_cache": M20_CACHE_SHA256,
        "golden_report": GOLDEN_REPORT_SHA256,
        "official_manifest": OFFICIAL_MANIFEST_SHA256,
    }
    if set(deferred) != set(expected):
        raise ValueError("Published deferred input key set differs.")
    for name, digest_expected in expected.items():
        if deferred[name].get("sha256") != digest_expected:
            raise ValueError("Published {} SHA-256 differs.".format(name))
    entries = payload["validation_dataset"].get("manifest_files", [])
    if len(entries) != OFFICIAL_VIDEO_COUNT or _semantic_manifest_sha256(entries) != OFFICIAL_SEMANTIC_SHA256:
        raise ValueError("Published validation manifest entries differ.")
    return payload, path, digest


def _canonical_inputs(science, published, published_path):
    result = {
        name: _workspace_path(science["train_only_evidence"][name]["workspace_relative_path"], name)
        for name in TRAIN_INPUT_NAMES
    }
    result["published_validation_contract"] = published_path
    for name, spec in published["deferred_validation_inputs"].items():
        result[name] = _workspace_path(spec["workspace_relative_path"], name)
    return result


def _expected_input_sha256(science, published):
    result = {name: science["train_only_evidence"][name]["sha256"] for name in TRAIN_INPUT_NAMES}
    result["published_validation_contract"] = science["published_validation_contract"]["sha256"]
    result.update({name: spec["sha256"] for name, spec in published["deferred_validation_inputs"].items()})
    return result


def _validate_train_only_evidence(science, input_paths, expected):
    verified = {}
    for name in TRAIN_INPUT_NAMES:
        digest = sha256_file(input_paths[name])
        if digest != expected[name]:
            raise ValueError("{} differs from the train-only frozen hash.".format(name))
        verified[name] = digest
    v5, _ = _load_json_snapshot(input_paths["v5_grouped_oof_report"], expected["v5_grouped_oof_report"], "v5 grouped OOF report")
    v5_spec = science["train_only_evidence"]["v5_grouped_oof_report"]
    for key, wanted in {
        "schema": v5_spec["required_schema"], "status": v5_spec["required_status"],
        "passed": True, "decision": v5_spec["required_decision"],
        "protocol_sha256": expected["v5_protocol"], "runner_sha256": expected["v5_runner"],
        "command_audit_sha256": expected["v5_command_audit"],
        "synthesis_manifest_sha256": expected["v5_synthesis_manifest"],
        "new_training_optimizer_steps": 0, "t32_read_or_combined": False,
    }.items():
        if v5.get(key) != wanted:
            raise ValueError("v5 report differs at {}.".format(key))
    if v5.get("promotion_gates", {}).get("passed") is not True:
        raise ValueError("v5 promotion gates did not pass.")
    pair, _ = _load_json_snapshot(input_paths["all11_v2_pair_audit"], expected["all11_v2_pair_audit"], "all11 v2 pair audit")
    for key, wanted in {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-pair-audit-v2",
        "status": "passed", "passed": True, "new_training_optimizer_steps": 0,
        "protocol_sha256": expected["all11_v2_protocol"],
        "runner_sha256": expected["all11_v2_runner"],
        "command_audit_sha256": expected["all11_v2_command_audit"],
    }.items():
        if pair.get(key) != wanted:
            raise ValueError("all11 pair audit differs at {}.".format(key))
    manifest, _ = _load_json_snapshot(input_paths["all11_v2_synthesis_manifest"], expected["all11_v2_synthesis_manifest"], "all11 v2 synthesis manifest")
    for key, wanted in {
        "schema": "ev-uav-metric-aux-task-arithmetic-all11-synthesis-manifest-v2",
        "status": "completed", "passed": True, "new_training_optimizer_steps": 0,
        "protocol_sha256": expected["all11_v2_protocol"],
        "runner_sha256": expected["all11_v2_runner"],
        "command_audit_sha256": expected["all11_v2_command_audit"],
        "pair_audit_sha256": expected["all11_v2_pair_audit"],
        "output_sha256": WFULL_CHECKPOINT_SHA256,
        "alpha_one_formula_bitwise_recompute": True,
        "alpha_zero_parent_bitwise_identity": True,
        "validation_or_test_read": False,
        "evaluation_or_score_run": False,
        "default_submission_changed": False,
    }.items():
        if manifest.get(key) != wanted:
            raise ValueError("all11 synthesis manifest differs at {}.".format(key))
    if Path(manifest.get("output_path", "")).resolve() != input_paths["wfull_checkpoint"]:
        raise ValueError("all11 synthesis manifest output path differs.")
    return {"verified_sha256": verified, "v5_passed": True, "all11_pair_passed": True, "all11_manifest_passed": True}


def build_execution_protocol(science, science_sha, published, published_path, code, git):
    validate_science_protocol(science)
    if science_sha != EXPECTED_SCIENCE_PROTOCOL_SHA256:
        raise ValueError("Science protocol SHA-256 differs from the runner constant.")
    inputs = _canonical_inputs(science, published, published_path)
    expected = _expected_input_sha256(science, published)
    protocol = {
        "schema": EXECUTION_SCHEMA,
        "created_utc": utc_now(),
        "attempt_budget": 1,
        "science_protocol": {"path": str(SCIENCE_PROTOCOL_PATH), "sha256": science_sha, "payload": science},
        "repository": {"project_root": str(PROJECT_ROOT), "expected_clean_git_head": git["head"], "code_sha256": code},
        "inputs": {name: {"path": str(path), "sha256": expected[name], "deferred_until_after_claim": name in DEFERRED_INPUT_NAMES} for name, path in inputs.items()},
        "validation_dataset": published["validation_dataset"],
        "golden": science["golden"],
        "route_policy": {"definition": _route_policy_definition(), "sha256": route_policy_sha256()},
        "inference": science["inference"],
        "postprocess": science["postprocess"],
        "promotion_gates": science["promotion_gates"],
        "runtime_contract": science["runtime_contract"],
        "outputs": {name: str(path) for name, path in _paths().items() if name not in {"science_protocol", "execution_protocol"}},
        "evidence_class": science["evidence_class"],
        "sequence_disclosure": science["sequence_disclosure"],
        "side_effect_limits": science["side_effect_limits"],
    }
    validate_execution_protocol(protocol)
    return protocol


def validate_execution_protocol(protocol):
    if protocol.get("schema") != EXECUTION_SCHEMA or protocol.get("attempt_budget") != 1:
        raise ValueError("Invalid execution protocol schema or attempt budget.")
    science = protocol.get("science_protocol", {})
    if science.get("sha256") != EXPECTED_SCIENCE_PROTOCOL_SHA256 or science.get("payload") is None:
        raise ValueError("Execution protocol does not bind the frozen science protocol.")
    validate_science_protocol(science["payload"])
    if Path(science.get("path", "")).resolve() != SCIENCE_PROTOCOL_PATH:
        raise ValueError("Execution science protocol path differs.")
    if Path(protocol.get("repository", {}).get("project_root", "")).resolve() != PROJECT_ROOT:
        raise ValueError("Execution project root differs.")
    published, published_path, _ = _load_published_validation_contract(science["payload"])
    canonical_inputs = _canonical_inputs(science["payload"], published, published_path)
    canonical_hashes = _expected_input_sha256(science["payload"], published)
    expected_names = set(TRAIN_INPUT_NAMES) | set(BASELINE_INPUT_NAMES) | set(DEFERRED_INPUT_NAMES) | {"published_validation_contract"}
    if set(protocol.get("inputs", {})) != expected_names:
        raise ValueError("Execution input key set differs.")
    for name, spec in protocol["inputs"].items():
        _require_sha256(spec.get("sha256"), name)
        if Path(spec.get("path", "")).resolve() != canonical_inputs[name]:
            raise ValueError("Execution input path differs for {}.".format(name))
        if spec.get("sha256") != canonical_hashes[name]:
            raise ValueError("Execution input SHA-256 differs for {}.".format(name))
        if bool(spec.get("deferred_until_after_claim")) != (name in DEFERRED_INPUT_NAMES):
            raise ValueError("Deferred flag differs for {}.".format(name))
    if protocol.get("repository", {}).get("code_sha256", {}).keys() != dict.fromkeys(CODE_PATHS).keys():
        raise ValueError("Execution code path set differs.")
    if protocol.get("route_policy") != {"definition": _route_policy_definition(), "sha256": route_policy_sha256()}:
        raise ValueError("Execution route policy differs.")
    if protocol.get("inference") != science["payload"]["inference"]:
        raise ValueError("Execution inference settings differ.")
    if protocol.get("golden") != science["payload"]["golden"]:
        raise ValueError("Execution golden constants differ.")
    if protocol.get("postprocess") != science["payload"]["postprocess"]:
        raise ValueError("Execution C00 contract differs.")
    if protocol.get("promotion_gates") != science["payload"]["promotion_gates"]:
        raise ValueError("Execution promotion gates differ.")
    dataset = protocol.get("validation_dataset", {})
    if dataset != published["validation_dataset"]:
        raise ValueError("Execution validation dataset differs from the published contract.")
    if dataset.get("video_count") != OFFICIAL_VIDEO_COUNT or dataset.get("event_count") != OFFICIAL_EVENT_COUNT:
        raise ValueError("Execution validation population differs.")
    if dataset.get("dataset_signature") != OFFICIAL_DATASET_SIGNATURE or dataset.get("semantic_manifest_sha256") != OFFICIAL_SEMANTIC_SHA256:
        raise ValueError("Execution validation identity differs.")
    if len(dataset.get("manifest_files", [])) != OFFICIAL_VIDEO_COUNT or _semantic_manifest_sha256(dataset["manifest_files"]) != OFFICIAL_SEMANTIC_SHA256:
        raise ValueError("Execution validation manifest entries differ.")
    expected_outputs = {
        name: str(path)
        for name, path in _paths().items()
        if name not in {"science_protocol", "execution_protocol"}
    }
    if protocol.get("outputs") != expected_outputs:
        raise ValueError("Execution output paths differ.")
    if protocol.get("runtime_contract") != science["payload"]["runtime_contract"]:
        raise ValueError("Execution runtime contract differs.")
    if protocol.get("evidence_class") != science["payload"]["evidence_class"]:
        raise ValueError("Execution evidence class differs.")
    if protocol.get("sequence_disclosure") != science["payload"]["sequence_disclosure"]:
        raise ValueError("Execution sequence disclosure differs.")
    if protocol.get("side_effect_limits") != science["payload"]["side_effect_limits"]:
        raise ValueError("Execution side-effect limits differ.")
    return protocol


def freeze_execution_protocol():
    paths = _paths()
    if any(paths[name].exists() for name in ("execution_protocol", "cpu_receipt", "runtime_receipt", "claim", "h2_cache", "report")):
        raise FileExistsError("A canonical W_full validation output path is already occupied.")
    science, science_sha = _load_json_snapshot(paths["science_protocol"], EXPECTED_SCIENCE_PROTOCOL_SHA256, "science protocol")
    validate_science_protocol(science)
    published, published_path, _ = _load_published_validation_contract(science)
    inputs = _canonical_inputs(science, published, published_path)
    expected = _expected_input_sha256(science, published)
    train_evidence = _validate_train_only_evidence(science, inputs, expected)
    verified_non_deferred = {}
    for name in BASELINE_INPUT_NAMES + ("published_validation_contract",):
        digest = sha256_file(inputs[name])
        if digest != expected[name]:
            raise ValueError("{} differs before freeze.".format(name))
        verified_non_deferred[name] = digest
    if _effective_c00_sha256() != EXPECTED_EFFECTIVE_C00_SHA256:
        raise ValueError("Live effective C00 differs before freeze.")
    git = _git_state()
    if not git["clean"]:
        raise RuntimeError("Commit all candidate code before freezing the execution protocol.")
    code = _code_sha256()
    protocol = build_execution_protocol(science, science_sha, published, published_path, code, git)
    digest = _atomic_json_no_clobber(paths["execution_protocol"], protocol)
    return {
        "execution_protocol": str(paths["execution_protocol"]), "sha256": digest,
        "train_only_evidence": train_evidence, "verified_non_deferred": verified_non_deferred,
        "deferred_until_after_claim": sorted(DEFERRED_INPUT_NAMES),
        "validation_npz_cache_label_manifest_or_golden_report_read": False,
        "attempt_claimed": False, "cuda_initialized": False,
    }


def _load_execution(expected_execution_sha256):
    expected = _require_sha256(expected_execution_sha256, "execution protocol")
    protocol, digest = _load_json_snapshot(_paths()["execution_protocol"], expected, "execution protocol")
    validate_execution_protocol(protocol)
    return protocol, _paths(), digest


def _preclaim_validate(expected_execution_sha256, stage):
    protocol, paths, protocol_sha = _load_execution(expected_execution_sha256)
    allowed = {
        "preflight": {"execution_protocol"},
        "runtime-preflight": {"execution_protocol", "cpu_receipt"},
        "run": {"execution_protocol", "cpu_receipt", "runtime_receipt"},
    }[stage]
    for name in ("execution_protocol", "cpu_receipt", "runtime_receipt", "claim", "h2_cache", "report"):
        exists = paths[name].exists()
        if name in allowed and not exists:
            raise FileNotFoundError("Required {} is missing.".format(name))
        if name not in allowed and exists:
            raise FileExistsError("Unexpected immutable output exists before {}: {}".format(stage, paths[name]))
    git = _git_state()
    if not git["clean"] or git["head"] != protocol["repository"]["expected_clean_git_head"]:
        raise RuntimeError("Git identity differs from the frozen execution protocol.")
    code = _code_sha256()
    if code != protocol["repository"]["code_sha256"]:
        raise RuntimeError("Code differs from the frozen execution protocol.")
    science, science_sha = _load_json_snapshot(SCIENCE_PROTOCOL_PATH, EXPECTED_SCIENCE_PROTOCOL_SHA256, "science protocol")
    if science_sha != protocol["science_protocol"]["sha256"] or science != protocol["science_protocol"]["payload"]:
        raise RuntimeError("Science protocol differs from the execution protocol.")
    input_paths = {name: Path(spec["path"]).resolve() for name, spec in protocol["inputs"].items()}
    verified = {}
    for name, path in input_paths.items():
        if name in DEFERRED_INPUT_NAMES:
            continue
        digest = sha256_file(path)
        if digest != protocol["inputs"][name]["sha256"]:
            raise ValueError("{} differs before claim.".format(name))
        verified[name] = digest
    evidence = _validate_train_only_evidence(science, input_paths, {name: spec["sha256"] for name, spec in protocol["inputs"].items()})
    if _effective_c00_sha256() != EXPECTED_EFFECTIVE_C00_SHA256:
        raise ValueError("Live effective C00 differs before claim.")
    return protocol, paths, protocol_sha, git, code, input_paths, verified, evidence


def _runtime_identity(torch, np, require_cuda):
    import cv2

    actual = {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode("utf-8")).hexdigest(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device_name": torch.cuda.get_device_name(0) if require_cuda and torch.cuda.is_available() else None,
    }
    differences = {name: {"expected": expected, "actual": actual.get(name)} for name, expected in EXPECTED_RUNTIME.items() if actual.get(name) != expected}
    if differences:
        raise RuntimeError("Runtime differs from the frozen contract: {}".format(json.dumps(differences, sort_keys=True)))
    if require_cuda:
        if (
            not actual["cuda_available"]
            or not actual["cuda_initialized"]
            or actual["cuda_runtime"] != "12.1"
            or actual["cuda_device_name"] != "NVIDIA GeForce RTX 4060 Laptop GPU"
        ):
            raise RuntimeError("Frozen CUDA runtime/device is unavailable.")
    elif actual["cuda_initialized"]:
        raise RuntimeError("CUDA must remain uninitialized during CPU preflight.")
    return actual


def _strict_cpu_wfull_load(path):
    import numpy as np
    import torch
    from utils.temporal_memory_inference import load_temporal_memory_model

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU W_full load.")
    before = sha256_file(path)
    model, checkpoint = load_temporal_memory_model(path, torch.device("cpu"), context_bins=5, width=16, sequence_length=16)
    after = sha256_file(path)
    if before != WFULL_CHECKPOINT_SHA256 or after != before:
        raise RuntimeError("W_full changed during CPU strict load.")
    named = list(model.named_parameters())
    if len(named) != 89 or sum(int(parameter.numel()) for _, parameter in named) != 1_924_716:
        raise RuntimeError("W_full model scope differs from 89 tensors / 1,924,716 parameters.")
    if any(not bool(torch.isfinite(parameter.detach()).all()) for _, parameter in named):
        raise RuntimeError("W_full contains non-finite model parameters.")
    temporal = checkpoint.get("temporal_memory", {})
    if temporal.get("context_bins") != 5 or temporal.get("width") != 16 or temporal.get("sequence_length") != 16:
        raise ValueError("W_full temporal-memory metadata differs.")
    runtime = _runtime_identity(torch, np, require_cuda=False)
    return {
        "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
        "tensor_count": len(named), "parameter_count": 1_924_716,
        "temporal_memory": {key: temporal.get(key) for key in ("context_bins", "width", "sequence_length")},
        "strict_load_passed": True, "cuda_initialized": bool(torch.cuda.is_initialized()),
        "runtime": runtime,
    }


def _synthetic_route_preflight():
    import numpy as np

    cases = {
        "low_30000": classify_wfull_route(np.zeros(30_000, dtype=np.float32), 160),
        "middle_30001": classify_wfull_route(np.zeros(30_001, dtype=np.float32), 160),
        "h1_200001": classify_wfull_route(np.zeros(200_001, dtype=np.float32), 160),
        "h2_exact_cutoff": classify_wfull_route(np.r_[np.ones(40_001, dtype=np.float32), np.zeros(160_004, dtype=np.float32)], 160),
        "h2_above_cutoff": classify_wfull_route(np.r_[np.ones(100_003, dtype=np.float32), np.zeros(100_002, dtype=np.float32)], 160),
    }
    expected_domains = {"low_30000": "low", "middle_30001": "middle", "h1_200001": "h1", "h2_exact_cutoff": "h2", "h2_above_cutoff": "h2"}
    if {name: decision.domain for name, decision in cases.items()} != expected_domains:
        raise RuntimeError("Synthetic route boundary smoke failed.")
    sentinel = object()
    calls = {"count": 0}
    def infer():
        calls["count"] += 1
        return object()
    for name, decision in cases.items():
        selected, preserved = choose_candidate_scores(decision, sentinel, infer)
        if decision.domain == "h2":
            if preserved or selected is sentinel:
                raise RuntimeError("H2 did not call the candidate predictor.")
        elif not preserved or selected is not sentinel:
            raise RuntimeError("Non-H2 failed same-object identity.")
    if calls["count"] != 2:
        raise RuntimeError("Only synthetic H2 routes may call the predictor.")
    return {
        "route_policy_sha256": route_policy_sha256(),
        "domains": expected_domains,
        "h2_predictor_calls": calls["count"],
        "non_h2_same_object_identity": True,
        "t32_called": False,
        "persistence_called": False,
        "passed": True,
    }


def preflight_execution(expected_execution_sha256):
    state = _preclaim_validate(expected_execution_sha256, "preflight")
    protocol, paths, protocol_sha, git, code, input_paths, verified, evidence = state
    smoke = _synthetic_route_preflight()
    wfull = _strict_cpu_wfull_load(input_paths["wfull_checkpoint"])
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU preflight.")
    receipt = {
        "schema": CPU_RECEIPT_SCHEMA, "created_utc": utc_now(), "passed": True,
        "execution_protocol_sha256": protocol_sha, "git": git, "code_sha256": code,
        "verified_non_deferred_inputs": verified, "train_only_evidence": evidence,
        "synthetic_route_smoke": smoke, "wfull_cpu_strict_load": wfull,
        "effective_c00_sha256": _effective_c00_sha256(),
        "deferred_until_after_claim": sorted(DEFERRED_INPUT_NAMES),
        "validation_npz_cache_label_manifest_or_golden_report_read": False,
        "attempt_claimed": False, "cuda_initialized": False,
    }
    digest = _atomic_json_no_clobber(paths["cpu_receipt"], receipt)
    return {"path": str(paths["cpu_receipt"]), "sha256": digest, "payload": receipt}


def _load_cpu_receipt(protocol_sha, paths, code, expected_receipt_sha256):
    receipt, digest = _load_json_snapshot(
        paths["cpu_receipt"],
        _require_sha256(expected_receipt_sha256, "CPU preflight receipt"),
        "CPU preflight receipt",
    )
    if (
        receipt.get("schema") != CPU_RECEIPT_SCHEMA or receipt.get("passed") is not True
        or receipt.get("execution_protocol_sha256") != protocol_sha
        or receipt.get("code_sha256") != code
        or receipt.get("validation_npz_cache_label_manifest_or_golden_report_read") is not False
        or receipt.get("attempt_claimed") is not False or receipt.get("cuda_initialized") is not False
        or receipt.get("synthetic_route_smoke", {}).get("passed") is not True
        or receipt.get("wfull_cpu_strict_load", {}).get("strict_load_passed") is not True
        or receipt.get("wfull_cpu_strict_load", {}).get("checkpoint_sha256_before") != WFULL_CHECKPOINT_SHA256
        or receipt.get("effective_c00_sha256") != EXPECTED_EFFECTIVE_C00_SHA256
        or receipt.get("deferred_until_after_claim") != sorted(DEFERRED_INPUT_NAMES)
    ):
        raise ValueError("CPU preflight receipt is incomplete or differs.")
    return receipt, digest


def _prepare_runtime_before_claim(protocol, input_paths):
    import numpy as np
    import torch
    import replay_temporal_memory_validation as replay
    from dataset.temporal_frame import load_temporal_frame_video, temporal_frame_video_from_events
    from utils.temporal_memory_inference import load_temporal_memory_model, predict_temporal_memory_scores

    forbidden_before = {
        "windowed": "utils.temporal_memory_windowed_inference" in sys.modules,
        "persistence": "utils.persistence_component_suppressor" in sys.modules,
    }
    if any(forbidden_before.values()):
        raise RuntimeError("A forbidden T32 or persistence runtime module was imported.")
    if sha256_file(input_paths["wfull_checkpoint"]) != WFULL_CHECKPOINT_SHA256:
        raise ValueError("W_full checkpoint differs before GPU load.")
    device = torch.device(protocol["inference"]["device"])
    model, checkpoint = load_temporal_memory_model(
        input_paths["wfull_checkpoint"], device,
        context_bins=protocol["inference"]["context_bins"],
        width=protocol["inference"]["network_width"],
        sequence_length=protocol["inference"]["sequence_length"],
    )
    runtime = _runtime_identity(torch, np, require_cuda=True)
    locations = np.asarray([[index % 346, (index * 3) % 260, index * 50] for index in range(160)], dtype=np.int64)
    polarities = (np.arange(160) % 2).astype(np.float32)
    video = temporal_frame_video_from_events("synthetic_full_t160", locations, polarities, 50, 8000)
    if len(video.event_indices_by_bin) != 160:
        raise RuntimeError("Synthetic video does not have exactly 160 bins.")
    scores = predict_temporal_memory_scores(
        model=model, video=video, device=device,
        context_bins=protocol["inference"]["context_bins"],
        width=protocol["inference"]["resolution"][0], height=protocol["inference"]["resolution"][1],
        inference_batch_size=protocol["inference"]["inference_batch_size"],
        log_count_clip=protocol["inference"]["log_count_clip"],
    ).detach().cpu().to(torch.float32).reshape(-1).contiguous()
    forbidden_after = {
        "windowed": "utils.temporal_memory_windowed_inference" in sys.modules,
        "persistence": "utils.persistence_component_suppressor" in sys.modules,
    }
    if scores.numel() != 160 or not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()) or bool((scores > 1).any()):
        raise RuntimeError("Synthetic W_full full-T160 inference failed.")
    if any(forbidden_after.values()):
        raise RuntimeError("Synthetic smoke imported a forbidden T32 or persistence module.")
    smoke = {
        "temporal_bin_count": 160, "event_count": 160, "score_count": int(scores.numel()),
        "scores_finite_probabilities": True, "api": "predict_temporal_memory_scores",
        "mode": "full_stream_t160", "window_length": None, "stride": None,
        "t32_called": False, "persistence_called": False,
        "checkpoint_sha256": WFULL_CHECKPOINT_SHA256,
        "checkpoint_metadata": {key: checkpoint.get("temporal_memory", {}).get(key) for key in ("context_bins", "width", "sequence_length")},
        "passed": True,
    }
    bundle = {
        "torch": torch, "numpy": np, "replay": replay, "model": model,
        "load_temporal_frame_video": load_temporal_frame_video,
        "predict_temporal_memory_scores": predict_temporal_memory_scores,
    }
    return runtime, smoke, bundle


def runtime_preflight_execution(
    expected_execution_sha256,
    expected_cpu_preflight_receipt_sha256,
    authorized_by_root=False,
):
    if not authorized_by_root:
        raise PermissionError("Explicit root authorization is required for GPU runtime preflight.")
    state = _preclaim_validate(expected_execution_sha256, "runtime-preflight")
    protocol, paths, protocol_sha, git, code, input_paths, verified, _ = state
    cpu_receipt, cpu_sha = _load_cpu_receipt(
        protocol_sha, paths, code, expected_cpu_preflight_receipt_sha256
    )
    runtime, smoke, _ = _prepare_runtime_before_claim(protocol, input_paths)
    receipt = {
        "schema": RUNTIME_RECEIPT_SCHEMA, "created_utc": utc_now(), "passed": True,
        "authorized_by_root": True, "execution_protocol_sha256": protocol_sha,
        "cpu_preflight_receipt_sha256": cpu_sha, "git": git, "code_sha256": code,
        "verified_non_deferred_inputs": verified, "runtime": runtime, "smoke": smoke,
        "deferred_until_after_claim": sorted(DEFERRED_INPUT_NAMES),
        "validation_npz_cache_label_manifest_or_golden_report_read": False,
        "attempt_claimed": False,
    }
    digest = _atomic_json_no_clobber(paths["runtime_receipt"], receipt)
    return {"path": str(paths["runtime_receipt"]), "sha256": digest, "payload": receipt, "cpu_receipt": cpu_receipt}


def _load_runtime_receipt(protocol_sha, paths, code, cpu_sha, expected_receipt_sha256):
    receipt, digest = _load_json_snapshot(
        paths["runtime_receipt"],
        _require_sha256(expected_receipt_sha256, "runtime preflight receipt"),
        "runtime preflight receipt",
    )
    if (
        receipt.get("schema") != RUNTIME_RECEIPT_SCHEMA or receipt.get("passed") is not True
        or receipt.get("authorized_by_root") is not True
        or receipt.get("execution_protocol_sha256") != protocol_sha
        or receipt.get("cpu_preflight_receipt_sha256") != cpu_sha
        or receipt.get("code_sha256") != code
        or receipt.get("validation_npz_cache_label_manifest_or_golden_report_read") is not False
        or receipt.get("attempt_claimed") is not False
        or receipt.get("smoke", {}).get("passed") is not True
        or receipt.get("smoke", {}).get("api") != "predict_temporal_memory_scores"
        or receipt.get("smoke", {}).get("t32_called") is not False
        or receipt.get("smoke", {}).get("persistence_called") is not False
    ):
        raise ValueError("Runtime preflight receipt is incomplete or differs.")
    return receipt, digest


def _atomic_claim(path, execution_sha, cpu_sha, runtime_sha):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CLAIM_SCHEMA, "created_utc": utc_now(), "attempt": 1, "attempt_budget": 1,
        "execution_protocol_sha256": execution_sha, "cpu_preflight_receipt_sha256": cpu_sha,
        "runtime_preflight_receipt_sha256": runtime_sha,
        "claim_wording": "Irreversible attempt 1/1 claimed before any validation NPZ cache label manifest or golden-report read.",
    }
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return payload, hashlib.sha256(data).hexdigest()


def _require_claim_before_deferred(paths, execution_sha, cpu_sha, runtime_sha):
    claim, digest = _load_json_snapshot(paths["claim"], description="attempt claim")
    if (
        claim.get("schema") != CLAIM_SCHEMA or claim.get("attempt") != 1 or claim.get("attempt_budget") != 1
        or claim.get("execution_protocol_sha256") != execution_sha
        or claim.get("cpu_preflight_receipt_sha256") != cpu_sha
        or claim.get("runtime_preflight_receipt_sha256") != runtime_sha
    ):
        raise RuntimeError("Attempt claim is absent, incomplete, or differs before deferred IO.")
    return claim, digest


def _extract_json_member(text, member):
    decoder = json.JSONDecoder()
    marker = json.dumps(str(member))
    offset = text.find(marker)
    if offset < 0:
        raise ValueError("JSON member {!r} is missing.".format(member))
    colon = text.find(":", offset + len(marker))
    if colon < 0:
        raise ValueError("JSON member {!r} has no value.".format(member))
    value, _ = decoder.raw_decode(text, colon + 1)
    return value


def _validate_golden_report_after_claim(path, expected_sha):
    before = sha256_file(path)
    if before != expected_sha:
        raise ValueError("Golden report differs after claim.")
    text = Path(path).read_text(encoding="utf-8")
    if _extract_json_member(text, "counts") != GOLDEN_COUNTS or _extract_json_member(text, "metrics") != GOLDEN_METRICS:
        raise ValueError("Golden report sufficient counts or metrics differ.")
    if sha256_file(path) != before:
        raise RuntimeError("Golden report changed while being read.")
    return {"path": str(Path(path).resolve()), "sha256": before, "counts": GOLDEN_COUNTS, "metrics": GOLDEN_METRICS}


def _manifest_val_entries_after_claim(path, expected_sha):
    before = sha256_file(path)
    if before != expected_sha:
        raise ValueError("Official manifest differs after claim.")
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("Official manifest changed while being read.")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Official manifest schema differs.")
    selected = {}
    for entry in manifest["files"]:
        relative = str(entry.get("path", "")).replace("\\", "/")
        if relative.startswith("val/"):
            name = Path(relative).name
            if relative != "val/" + name or name in selected:
                raise ValueError("Official manifest validation paths are noncanonical.")
            selected[name] = {"path": relative, "size": int(entry.get("size", -1)), "sha256": str(entry.get("sha256", "")).lower()}
    expected_names = tuple(stem + ".npz" for stem in OFFICIAL_STEMS)
    if tuple(sorted(selected)) != expected_names:
        raise ValueError("Official manifest does not contain exactly val_000..val_023.")
    return [selected[name] for name in expected_names]


def _validate_validation_files_after_claim(protocol, input_paths):
    manifest_entries = _manifest_val_entries_after_claim(input_paths["official_manifest"], protocol["inputs"]["official_manifest"]["sha256"])
    expected_entries = protocol["validation_dataset"]["manifest_files"]
    if manifest_entries != expected_entries:
        raise ValueError("Official manifest entries differ from the execution protocol.")
    val_root = (input_paths["official_manifest"].parent / "val").resolve()
    actual_names = tuple(sorted(path.name for path in val_root.glob("*.npz") if path.is_file()))
    expected_names = tuple(Path(entry["path"]).name for entry in expected_entries)
    if actual_names != expected_names:
        raise ValueError("Validation directory population differs.")
    evidence = []
    for entry in expected_entries:
        path = (val_root / Path(entry["path"]).name).resolve()
        if path.parent != val_root or not path.is_file():
            raise ValueError("Validation path is noncanonical: {}".format(path))
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(entry["size"]) or digest != entry["sha256"]:
            raise ValueError("Validation source differs: {}".format(path.name))
        evidence.append({"name": path.name, "size": size, "sha256": digest})
    if _semantic_manifest_sha256(expected_entries) != OFFICIAL_SEMANTIC_SHA256:
        raise RuntimeError("Validation semantic manifest differs.")
    return {"val_root": str(val_root), "manifest_sha256": OFFICIAL_MANIFEST_SHA256, "semantic_sha256": OFFICIAL_SEMANTIC_SHA256, "video_count": len(evidence), "files": evidence}


def _validate_cache_contract(replay, primary, secondary, input_paths):
    expected_inference = {
        "temporal_memory_bin_size": 50, "temporal_memory_context_bins": 5,
        "temporal_memory_width": 16, "temporal_memory_sequence_length": 16,
        "temporal_memory_inference_batch_size": 8, "temporal_memory_log_count_clip": 4.0,
        "whole_t": 8000, "resolution": [346, 260],
    }
    for name, payload, checkpoint_name, checkpoint_sha in (
        ("m20", primary, "m20_checkpoint", M20_CHECKPOINT_SHA256),
        ("m10", secondary, "m10_checkpoint", M10_CHECKPOINT_SHA256),
    ):
        metadata = payload["metadata"]
        if (
            metadata.get("dataset_split") != "val" or metadata.get("video_count") != OFFICIAL_VIDEO_COUNT
            or metadata.get("event_count") != OFFICIAL_EVENT_COUNT or metadata.get("dataset_signature") != OFFICIAL_DATASET_SIGNATURE
            or metadata.get("checkpoint_sha256") != checkpoint_sha
            or Path(metadata.get("checkpoint_path", "")).resolve() != input_paths[checkpoint_name]
            or metadata.get("inference_settings") != expected_inference
        ):
            raise ValueError("{} golden cache contract differs.".format(name))
    binding = replay._validate_cache_compatibility(primary, secondary, secondary_max_events=LOW_EVENT_COUNT_MAX)
    records = replay.route_cache_records(primary, secondary, LOW_EVENT_COUNT_MAX)
    if tuple(Path(record.file_name).stem for record in records) != OFFICIAL_STEMS:
        raise ValueError("Golden cache record order differs.")
    return binding, records


def _validate_raw_alignment(video, record, np):
    cached_locs = record.locs.detach().cpu().numpy()
    if cached_locs.ndim != 2 or cached_locs.shape[1] != 4 or not np.array_equal(cached_locs[:, 1:4], video.locations):
        raise ValueError("Raw validation locations differ from the golden cache.")
    if not np.array_equal(record.seg_label.detach().cpu().numpy().reshape(-1), video.labels):
        raise ValueError("Raw validation labels differ from the golden cache.")
    if not np.array_equal(np.asarray(record.idx_label).reshape(-1), video.target_ids):
        raise ValueError("Raw validation target ids differ from the golden cache.")


def _tensor_bytes_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _delta(candidate, baseline):
    return {name: float(candidate[name]) - float(baseline[name]) for name in baseline}


def _run_claimed(protocol, paths, input_paths, runtime_bundle, execution_sha, cpu_sha, runtime_sha):
    claim, claim_sha = _require_claim_before_deferred(paths, execution_sha, cpu_sha, runtime_sha)
    torch = runtime_bundle["torch"]
    np = runtime_bundle["numpy"]
    replay = runtime_bundle["replay"]
    load_video = runtime_bundle["load_temporal_frame_video"]
    predict = runtime_bundle["predict_temporal_memory_scores"]
    model = runtime_bundle["model"]

    validation_before = _validate_validation_files_after_claim(protocol, input_paths)
    golden_report = _validate_golden_report_after_claim(input_paths["golden_report"], GOLDEN_REPORT_SHA256)
    deferred_hashes_before = {name: sha256_file(input_paths[name]) for name in DEFERRED_INPUT_NAMES}
    for name, digest in deferred_hashes_before.items():
        if digest != protocol["inputs"][name]["sha256"]:
            raise ValueError("{} differs after claim.".format(name))
    secondary, secondary_sha = replay.load_cache_snapshot(input_paths["m10_golden_cache"])
    primary, primary_sha = replay.load_cache_snapshot(input_paths["m20_golden_cache"])
    if secondary_sha != deferred_hashes_before["m10_golden_cache"] or primary_sha != deferred_hashes_before["m20_golden_cache"]:
        raise RuntimeError("A golden score cache changed while being loaded.")
    binding, baseline_records = _validate_cache_contract(replay, primary, secondary, input_paths)
    cfg = _c00_config()
    if _effective_c00_sha256() != EXPECTED_EFFECTIVE_C00_SHA256:
        raise RuntimeError("Effective C00 differs during the formal replay.")

    device = torch.device(protocol["inference"]["device"])
    baseline_counts = []
    candidate_counts = []
    h2_baseline_counts = []
    h2_candidate_counts = []
    per_video = []
    h2_records = []
    inference_calls = 0
    all_non_h2_preserved = True
    only_h2_called = True
    file_sha = {entry["name"]: entry["sha256"] for entry in validation_before["files"]}
    val_root = Path(validation_before["val_root"])

    for index, baseline_record in enumerate(baseline_records, start=1):
        file_name = Path(baseline_record.file_name).name
        raw_path = (val_root / file_name).resolve()
        raw_before = sha256_file(raw_path)
        video = load_video(raw_path, protocol["inference"]["temporal_bin_size"], protocol["inference"]["whole_t"])
        raw_after = sha256_file(raw_path)
        if raw_after != raw_before or raw_before != file_sha[file_name]:
            raise RuntimeError("Raw validation source changed while loading: {}".format(file_name))
        decision = classify_wfull_route(video.polarities, len(video.event_indices_by_bin))
        _validate_raw_alignment(video, baseline_record, np)
        if decision.event_count != baseline_record.event_count:
            raise ValueError("Route event count differs from the golden cache.")
        expected_baseline = "secondary" if decision.domain == "low" else "primary"
        if baseline_record.score_source != expected_baseline:
            raise RuntimeError("Golden baseline route source differs.")

        called = {"value": False}
        def infer_h2():
            nonlocal inference_calls
            called["value"] = True
            inference_calls += 1
            return predict(
                model=model, video=video, device=device,
                context_bins=protocol["inference"]["context_bins"],
                width=protocol["inference"]["resolution"][0], height=protocol["inference"]["resolution"][1],
                inference_batch_size=protocol["inference"]["inference_batch_size"],
                log_count_clip=protocol["inference"]["log_count_clip"],
            )

        scores, identity_route = choose_candidate_scores(decision, baseline_record.scores, infer_h2)
        if decision.domain == "h2":
            scores = scores.detach().cpu().to(torch.float32).reshape(-1).contiguous()
            preserved = False
            if not called["value"]:
                raise RuntimeError("H2 route did not call W_full.")
        else:
            if called["value"]:
                only_h2_called = False
            preserved = (
                identity_route and scores is baseline_record.scores
                and scores.data_ptr() == baseline_record.scores.data_ptr()
                and scores.storage_offset() == baseline_record.scores.storage_offset()
                and scores.stride() == baseline_record.scores.stride()
                and torch.equal(scores, baseline_record.scores)
            )
            all_non_h2_preserved = all_non_h2_preserved and preserved
        if scores.numel() != decision.event_count or not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()) or bool((scores > 1).any()):
            raise RuntimeError("Candidate scores are malformed for {}.".format(file_name))

        candidate_record = replay.RoutedRecord(
            file_name=baseline_record.file_name, event_count=baseline_record.event_count,
            scores=scores, seg_label=baseline_record.seg_label, locs=baseline_record.locs,
            idx_label=baseline_record.idx_label, source_sha256=baseline_record.source_sha256,
            score_source=decision.candidate_score_source,
        )
        base_count = replay.evaluate_cached_video(baseline_record, decision.prediction_threshold, cfg)
        cand_count = replay.evaluate_cached_video(candidate_record, decision.prediction_threshold, cfg)
        baseline_counts.append(base_count)
        candidate_counts.append(cand_count)
        if decision.domain == "h2":
            h2_baseline_counts.append(base_count)
            h2_candidate_counts.append(cand_count)
            h2_records.append({
                "file_name": file_name, "event_count": decision.event_count,
                "source_sha256": baseline_record.source_sha256, "source_file_sha256": file_sha[file_name],
                "route": decision.to_metadata(), "scores": scores,
                "scores_dtype": str(scores.dtype), "scores_shape": list(scores.shape),
                "scores_bytes_sha256": _tensor_bytes_sha256(scores),
            })
        elif asdict(base_count) != asdict(cand_count):
            all_non_h2_preserved = False
        per_video.append({
            "index": index, "file_name": file_name, "event_count": decision.event_count,
            "polarity_minority_fraction": decision.polarity_minority_fraction,
            "route": decision.to_metadata(), "threshold": decision.prediction_threshold,
            "wfull_called": called["value"], "scores_identity_preserved": preserved,
            "baseline_counts": asdict(base_count), "candidate_counts": asdict(cand_count),
        })
        print("route/evaluate {}/24: {} -> {}".format(index, file_name, decision.domain), flush=True)

    h2_count = sum(item["route"]["domain"] == "h2" for item in per_video)
    if inference_calls != h2_count or len(h2_records) != h2_count or h2_count <= 0:
        raise RuntimeError("W_full inference count differs from the positive H2 route count.")
    if any(item["wfull_called"] != (item["route"]["domain"] == "h2") for item in per_video):
        only_h2_called = False
    if not only_h2_called:
        raise RuntimeError("W_full was called outside H2.")

    h2_payload = {
        "schema": H2_CACHE_SCHEMA, "created_utc": utc_now(),
        "execution_protocol_sha256": execution_sha, "cpu_preflight_receipt_sha256": cpu_sha,
        "runtime_preflight_receipt_sha256": runtime_sha, "attempt_claim_sha256": claim_sha,
        "candidate_checkpoint_path": str(input_paths["wfull_checkpoint"]),
        "candidate_checkpoint_sha256": WFULL_CHECKPOINT_SHA256,
        "route_policy_sha256": route_policy_sha256(),
        "inference_settings": protocol["inference"],
        "inference_settings_sha256": canonical_sha256(protocol["inference"]),
        "effective_c00_sha256": EXPECTED_EFFECTIVE_C00_SHA256,
        "record_count": h2_count, "records": h2_records,
        "t32_read_or_combined": False, "persistence_read_or_combined": False,
    }
    replay._atomic_torch_save(h2_payload, paths["h2_cache"], overwrite=False)
    h2_cache_sha = sha256_file(paths["h2_cache"])

    baseline_total = replay._sum_counts(baseline_counts)
    candidate_total = replay._sum_counts(candidate_counts)
    h2_baseline_total = replay._sum_counts(h2_baseline_counts)
    h2_candidate_total = replay._sum_counts(h2_candidate_counts)
    baseline_count_dict = asdict(baseline_total)
    candidate_count_dict = asdict(candidate_total)
    h2_baseline_count_dict = asdict(h2_baseline_total)
    h2_candidate_count_dict = asdict(h2_candidate_total)
    baseline_metrics = replay.metrics_from_counts_exact(baseline_total, cfg).to_dict()
    candidate_metrics = replay.metrics_from_counts_exact(candidate_total, cfg).to_dict()
    h2_baseline_metrics = replay.metrics_from_counts_exact(h2_baseline_total, cfg).to_dict()
    h2_candidate_metrics = replay.metrics_from_counts_exact(h2_candidate_total, cfg).to_dict()
    gates, materiality = promotion_gate_results(
        baseline_count_dict, baseline_metrics, candidate_count_dict, candidate_metrics,
        h2_baseline_count_dict, h2_baseline_metrics, h2_candidate_count_dict, h2_candidate_metrics,
        all_non_h2_preserved, inference_calls, h2_count, only_h2_called,
    )

    validation_after = _validate_validation_files_after_claim(protocol, input_paths)
    if validation_after != validation_before:
        raise RuntimeError("Validation evidence changed during the one-shot replay.")
    deferred_hashes_after = {name: sha256_file(input_paths[name]) for name in DEFERRED_INPUT_NAMES}
    if deferred_hashes_after != deferred_hashes_before:
        raise RuntimeError("Deferred validation evidence changed during replay.")
    return {
        "claim": claim, "claim_sha256": claim_sha,
        "validation_integrity": {"before": validation_before, "after": validation_after, "equal": True},
        "golden_report": golden_report, "deferred_sha256": deferred_hashes_before,
        "golden_cache_binding": binding,
        "h2_cache": {"path": str(paths["h2_cache"]), "sha256": h2_cache_sha, "record_count": h2_count},
        "route_summary": {
            "counts": {domain: sum(item["route"]["domain"] == domain for item in per_video) for domain in ("low", "middle", "h1", "h2")},
            "wfull_full_t160_calls": inference_calls, "only_h2_called_wfull": only_h2_called,
            "non_h2_scores_identity_preserved": all_non_h2_preserved,
        },
        "per_video": per_video,
        "aggregate": {
            "baseline": {"counts": baseline_count_dict, "metrics": baseline_metrics},
            "candidate": {"counts": candidate_count_dict, "metrics": candidate_metrics},
            "delta": {"counts": {name: candidate_count_dict[name] - baseline_count_dict[name] for name in baseline_count_dict}, "metrics": _delta(candidate_metrics, baseline_metrics)},
        },
        "validation_h2": {
            "baseline": {"counts": h2_baseline_count_dict, "metrics": h2_baseline_metrics},
            "candidate": {"counts": h2_candidate_count_dict, "metrics": h2_candidate_metrics},
            "delta": {"counts": {name: h2_candidate_count_dict[name] - h2_baseline_count_dict[name] for name in h2_baseline_count_dict}, "metrics": _delta(h2_candidate_metrics, h2_baseline_metrics)},
        },
        "safety_gates": gates, "materiality_report_only": materiality,
        "passed": all(gates.values()),
        "t32_read_or_combined": False, "persistence_read_or_combined": False,
    }


def _failure_report(protocol_sha, cpu_sha, runtime_sha, claim, claim_sha, stage, error, paths):
    return {
        "schema": REPORT_SCHEMA, "created_utc": utc_now(), "status": "failed_during_claimed_attempt",
        "passed": False, "stage": stage, "error_type": type(error).__name__, "error": str(error),
        "execution_protocol_sha256": protocol_sha, "cpu_preflight_receipt_sha256": cpu_sha,
        "runtime_preflight_receipt_sha256": runtime_sha, "attempt_claim": {"payload": claim, "sha256": claim_sha},
        "artifact_observation": {name: {"exists": paths[name].exists(), "sha256": sha256_file(paths[name]) if paths[name].is_file() else None} for name in ("claim", "h2_cache")},
        "failure_action": "archive_without_validation_retuning_threshold_search_route_change_or_second_attempt",
        "submission_zip_created": False, "platform_upload_performed": False,
    }


def run_execution(
    expected_execution_sha256,
    expected_cpu_preflight_receipt_sha256,
    expected_runtime_preflight_receipt_sha256,
    authorized_by_root=False,
):
    if not authorized_by_root:
        raise PermissionError("Explicit root authorization is required for the formal validation replay.")
    state = _preclaim_validate(expected_execution_sha256, "run")
    protocol, paths, protocol_sha, git_before, code_before, input_paths, verified_before, _ = state
    cpu_receipt, cpu_sha = _load_cpu_receipt(
        protocol_sha, paths, code_before, expected_cpu_preflight_receipt_sha256
    )
    runtime_receipt, runtime_sha = _load_runtime_receipt(
        protocol_sha,
        paths,
        code_before,
        cpu_sha,
        expected_runtime_preflight_receipt_sha256,
    )
    runtime, smoke, bundle = _prepare_runtime_before_claim(protocol, input_paths)
    if runtime != runtime_receipt["runtime"] or smoke != runtime_receipt["smoke"]:
        raise RuntimeError("Live runtime smoke differs from the immutable runtime receipt.")
    claim, claim_sha = _atomic_claim(paths["claim"], protocol_sha, cpu_sha, runtime_sha)
    stage = "after_claim_before_first_deferred_read"
    try:
        stage = "claimed_h2_only_wfull_full_t160_and_golden_c00"
        outcome = _run_claimed(protocol, paths, input_paths, bundle, protocol_sha, cpu_sha, runtime_sha)
        stage = "postrun_immutability_check"
        git_after = _git_state()
        code_after = _code_sha256()
        verified_after = {name: sha256_file(path) for name, path in input_paths.items() if name not in DEFERRED_INPUT_NAMES}
        if git_after != git_before or code_after != code_before or verified_after != verified_before:
            raise RuntimeError("Repository code or a non-deferred input changed during replay.")
        if sha256_file(paths["execution_protocol"]) != protocol_sha or sha256_file(paths["cpu_receipt"]) != cpu_sha or sha256_file(paths["runtime_receipt"]) != runtime_sha or sha256_file(paths["claim"]) != claim_sha:
            raise RuntimeError("A frozen protocol/receipt/claim changed during replay.")
        report = {
            "schema": REPORT_SCHEMA, "created_utc": utc_now(), "status": "completed",
            "passed": outcome["passed"], "evidence_class": protocol["evidence_class"],
            "sequence_disclosure": protocol["sequence_disclosure"],
            "execution_protocol": {"path": str(paths["execution_protocol"]), "sha256": protocol_sha},
            "cpu_preflight_receipt": {"path": str(paths["cpu_receipt"]), "sha256": cpu_sha, "payload": cpu_receipt},
            "runtime_preflight_receipt": {"path": str(paths["runtime_receipt"]), "sha256": runtime_sha, "payload": runtime_receipt},
            "attempt_claim": {"payload": claim, "sha256": claim_sha},
            "repository": {"before": git_before, "after": git_after, "code_sha256": code_before},
            "candidate": {"checkpoint_path": str(input_paths["wfull_checkpoint"]), "checkpoint_sha256": WFULL_CHECKPOINT_SHA256, "train_only_evidence": protocol["science_protocol"]["payload"]["train_only_evidence"]},
            "route_policy": protocol["route_policy"], "inference": protocol["inference"], "postprocess": protocol["postprocess"],
            "validation_integrity": outcome["validation_integrity"], "golden_report": outcome["golden_report"],
            "deferred_sha256": outcome["deferred_sha256"], "golden_cache_binding": outcome["golden_cache_binding"],
            "h2_cache": outcome["h2_cache"], "route_summary": outcome["route_summary"], "per_video": outcome["per_video"],
            "aggregate": outcome["aggregate"], "validation_h2": outcome["validation_h2"],
            "safety_gates": outcome["safety_gates"], "materiality_report_only": outcome["materiality_report_only"],
            "t32_read_or_combined": False, "persistence_read_or_combined": False,
            "failure_action": None if outcome["passed"] else "archive_without_validation_retuning_threshold_search_route_change_or_second_attempt",
            "submission_zip_created": False, "platform_upload_performed": False,
        }
    except BaseException as error:
        report = _failure_report(protocol_sha, cpu_sha, runtime_sha, claim, claim_sha, stage, error, paths)
        _atomic_json_no_clobber(paths["report"], report)
        raise
    _atomic_json_no_clobber(paths["report"], report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    for name in ("preflight", "runtime-preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--expected-execution-protocol-sha256", required=True)
        if name in {"runtime-preflight", "run"}:
            command.add_argument("--expected-cpu-preflight-receipt-sha256", required=True)
            command.add_argument("--authorized-by-root", action="store_true")
        if name == "run":
            command.add_argument("--expected-runtime-preflight-receipt-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.command == "freeze":
        result = freeze_execution_protocol()
    elif args.command == "preflight":
        result = preflight_execution(args.expected_execution_protocol_sha256)
    elif args.command == "runtime-preflight":
        result = runtime_preflight_execution(
            args.expected_execution_protocol_sha256,
            args.expected_cpu_preflight_receipt_sha256,
            args.authorized_by_root,
        )
    else:
        result = run_execution(
            args.expected_execution_protocol_sha256,
            args.expected_cpu_preflight_receipt_sha256,
            args.expected_runtime_preflight_receipt_sha256,
            args.authorized_by_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

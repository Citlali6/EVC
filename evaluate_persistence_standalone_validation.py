"""Freeze, preflight, and later run one sequence-adaptive persistence val24 replay.

``freeze`` and ``preflight`` are deliberately unable to open the official
validation manifest, validation NPZ files, either golden score cache, or the
golden report.  ``run`` creates an irreversible exclusive claim before any of
those deferred inputs are hashed or opened.  This file is not a T32 runner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np

from utils.component_reranker import sha256_file, sha256_json
from utils.persistence_component_suppressor import (
    DEFAULT_TOPOLOGY,
    FEATURE_NAMES,
    PersistenceArtifact,
    PersistenceComponentSuppressor,
    observable_route,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SCIENCE_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "protocols"
    / "persistence_standalone_val24_sequence_science_v1.json"
).resolve()
EXPECTED_SCIENCE_PROTOCOL_SHA256 = (
    "54dcb6d9b8e535110113a05a31c66bf50af39919d82a6d4d63c676d5d491187f"
)
EXPERIMENT_DIRECTORY = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260810_persistence_standalone_val24_sequence_v1"
).resolve()

EXECUTION_PROTOCOL_SCHEMA = "ev-uav-persistence-standalone-val24-execution-v1"
PREFLIGHT_RECEIPT_SCHEMA = "ev-uav-persistence-standalone-val24-preflight-v1"
CLAIM_SCHEMA = "ev-uav-persistence-standalone-val24-attempt-claim-v1"
REPORT_SCHEMA = "ev-uav-persistence-standalone-val24-report-v1"
EXPECTED_EFFECTIVE_C00_SHA256 = (
    "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
)
OFFICIAL_VIDEO_COUNT = 24
OFFICIAL_EVENT_COUNT = 1_424_330
OFFICIAL_STEMS = tuple("val_{:03d}".format(index) for index in range(24))
OFFICIAL_DATASET_SIGNATURE = "bedba93c1d523f58c35da6399219df1b98e6240f92d093520fa0f4961d927274"
OFFICIAL_MANIFEST_SHA256 = "c7c574b5dfa8336fe50917581544b5e4991b2cde197f31c9a5bee05a29e336d4"
OFFICIAL_SEMANTIC_SHA256 = "d780da17e69446b988b1b5fae7954855d5ce66a32aa7b9581eeb3e4a0563f83f"
M10_CACHE_SHA256 = "96a9dfa8833e6f609d29f4db9d8f7196c84c7e92c7026cce734b97ddf133622f"
M20_CACHE_SHA256 = "6c9b4a8e33217aac7a05c78590a7feb6db6e6fc332b6411d7603264687710304"
GOLDEN_REPORT_SHA256 = "da6004ddd22731b8e848c9ed0c561961abbc04b4e3f66cd07b1e085d26f9f383"
M10_CHECKPOINT_SHA256 = "5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935"
M20_CHECKPOINT_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
ARTIFACT_SHA256 = "7ffdb37352ed3a5be0b9f4b22b67fbbbb7bc66dd3befbe0c5b1ffd1823d8f07e"
TRAIN_REPLAY_SHA256 = "d9d1f232baaa4877177538c9d97ace7729d2d3913a80cf690b28bcb85f513d4e"
TRAIN_OOF_SHA256 = "acfdda1910305834ad217117b3c237fb496050aa1c6d3f0669594feaac3e96ae"
TRAIN_FIT_PROTOCOL_SHA256 = "d7e064a5b941453e0a940d3401c6734b3fc25fe2461e8758a392f78810b69878"
MINIMUM_SCORE_DELTA = 0.0001
LOW_EVENT_COUNT_MAX = 30_000
LOW_THRESHOLD = 0.718
M20_THRESHOLD = 0.719
EXPECTED_RUNTIME = {
    "python_version": "3.9.25",
    "numpy_version": "1.26.4",
    "opencv_version": "4.8.1",
    "opencv_build_sha256": (
        "173c080cc486d36465d7dcbe73e6a921c4c55fb56ee67b0cc2dad09ddd43f4f4"
    ),
    "torch_version": "2.5.1+cu121",
    "platform": "Windows-10-10.0.26100-SP0",
    "cpu_only": True,
    "cuda_must_remain_uninitialized": True,
}

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
    "p6_event_count_cutoff": 30000,
    "p6_low_density_threshold": LOW_THRESHOLD,
    "p6_high_density_threshold": M20_THRESHOLD,
}
CODE_PATHS = (
    "evaluate_persistence_standalone_validation.py",
    "crossfit_component_reranker.py",
    "train_component_reranker.py",
    "protocols/persistence_standalone_val24_sequence_science_v1.json",
    "utils/persistence_component_suppressor.py",
    "utils/component_reranker.py",
    "utils/postprocess.py",
    "utils/temporal_memory_input_router.py",
    "replay_temporal_memory_validation.py",
    "dataset/temporal_frame.py",
    "utils/challenge_eval.py",
    "utils/density_threshold.py",
    "utils/eval.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _paths():
    return {
        "execution_protocol": EXPERIMENT_DIRECTORY / "execution_protocol.json",
        "preflight_receipt": EXPERIMENT_DIRECTORY / "preflight_receipt.json",
        "claim": EXPERIMENT_DIRECTORY / "validation_attempt_claim.json",
        "h2_cache": EXPERIMENT_DIRECTORY / "h2_p0_persistence_p18_cache.pt",
        "report": EXPERIMENT_DIRECTORY / "frozen_validation_report.json",
    }


def _workspace_path(relative_path, description):
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("{} must be workspace-relative.".format(description))
    path = (WORKSPACE_ROOT / relative).resolve()
    try:
        path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError("{} escapes the workspace.".format(description)) from error
    return path


def _load_json_snapshot(path, expected_sha256, description):
    path = Path(path).resolve()
    before = sha256_file(path)
    if before != expected_sha256:
        raise ValueError("{} SHA-256 differs.".format(description))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while being read.".format(description))
    return payload


def _atomic_json_no_clobber(path, payload):
    import tempfile

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Refusing to overwrite immutable output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _semantic_manifest_sha256(entries):
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(Path(entry["path"]).name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry["sha256"]))
    return digest.hexdigest()


def _code_sha256():
    return {relative: sha256_file(PROJECT_ROOT / relative) for relative in CODE_PATHS}


def _c00_config():
    return SimpleNamespace(**dict(C00_SETTINGS))


def _effective_c00_sha256():
    import crossfit_component_reranker as component_crossfit

    cfg = _c00_config()
    component_crossfit.validate_c00_config(cfg)
    return sha256_json(component_crossfit._postprocess_contract(cfg))


def _runtime_identity():
    """Return and fail closed on the frozen CPU feature/runtime environment."""
    import cv2
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain uninitialized for persistence replay.")
    actual = {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "opencv_build_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "cpu_only": True,
        "cuda_must_remain_uninitialized": True,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    differences = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in EXPECTED_RUNTIME.items()
        if actual.get(name) != expected
    }
    if differences:
        raise RuntimeError(
            "Persistence runtime differs from the frozen contract: {}.".format(
                json.dumps(differences, sort_keys=True)
            )
        )
    return actual


def validate_science_protocol(protocol):
    if (
        protocol.get("schema")
        != "ev-uav-persistence-standalone-val24-sequence-science-v1"
        or protocol.get("status")
        != "frozen_before_any_persistence_val24_data_or_cache_access"
        or protocol.get("candidate_id") != "persistence_pw08_kp050"
        or protocol.get("evidence_class")
        != "single_sequence_adaptive_val24_replay_not_independent_held"
    ):
        raise ValueError("Persistence val24 science protocol identity differs.")
    disclosure = protocol["sequence_disclosure"]
    if (
        disclosure["candidate_selected_before_t32_validation_failure"] is not True
        or disclosure["this_replay_ordered_after_t32_validation_failure"] is not True
        or disclosure["independent_held_claim_allowed"] is not False
        or disclosure["train_oof_candidate_grid_count"] != 7
        or disclosure["reported_train_oof_delta_is_selection_affected"] is not True
    ):
        raise ValueError("Sequence-adaptive disclosure differs.")
    budget = protocol["attempt_budget"]
    if (
        budget["full_val24_replays"] != 1
        or budget["claim_required_before_any_validation_npz_cache_label_or_report_read"]
        is not True
        or budget["claim_is_irreversible"] is not True
    ):
        raise ValueError("Persistence validation attempt budget differs.")
    split_access = protocol["split_access"]
    if (
        "H2-only final-fit protocol" not in split_access["before_claim_allowed"]
        or set(split_access["deferred_until_after_claim"])
        != {
            "official validation manifest",
            "24 validation NPZ files",
            "M10 and M20 golden validation caches",
            "golden validation report",
        }
        or split_access["test_forbidden"] is not True
        or split_access["gpu_allowed"] is not False
    ):
        raise ValueError("Persistence split-access contract differs.")
    candidate = protocol["candidate_inputs"]
    expected_candidate_hashes = {
        "artifact": ARTIFACT_SHA256,
        "train_replay_report": TRAIN_REPLAY_SHA256,
        "train_oof_report": TRAIN_OOF_SHA256,
        "train_fit_protocol": TRAIN_FIT_PROTOCOL_SHA256,
    }
    for name, expected in expected_candidate_hashes.items():
        if candidate[name]["sha256"] != expected:
            raise ValueError("Candidate input binding differs: {}".format(name))
    deferred = protocol["deferred_validation_inputs"]
    expected_deferred = {
        "m10_checkpoint": M10_CHECKPOINT_SHA256,
        "m20_checkpoint": M20_CHECKPOINT_SHA256,
        "m10_golden_cache": M10_CACHE_SHA256,
        "m20_golden_cache": M20_CACHE_SHA256,
        "golden_report": GOLDEN_REPORT_SHA256,
        "official_manifest": OFFICIAL_MANIFEST_SHA256,
    }
    for name, expected in expected_deferred.items():
        if deferred[name]["sha256"] != expected:
            raise ValueError("Deferred validation binding differs: {}".format(name))
    dataset = protocol["validation_dataset"]
    entries = dataset["manifest_files"]
    if (
        dataset["video_count"] != OFFICIAL_VIDEO_COUNT
        or dataset["event_count"] != OFFICIAL_EVENT_COUNT
        or dataset["dataset_signature"] != OFFICIAL_DATASET_SIGNATURE
        or tuple(Path(item["path"]).stem for item in entries) != OFFICIAL_STEMS
        or _semantic_manifest_sha256(entries) != OFFICIAL_SEMANTIC_SHA256
    ):
        raise ValueError("Frozen official val24 metadata differs.")
    chain = protocol["candidate_chain"]
    if (
        chain["t32_allowed"] is not False
        or chain["effective_c00_canonical_sha256"]
        != EXPECTED_EFFECTIVE_C00_SHA256
        or chain["h2_stage_order"]
        != [
            "released raw M20 full-stream T160 probabilities",
            "frozen P0/P0c postprocess",
            "extract topology r1/tbin50/link6/gap1/max_events3 candidates",
            "derive the frozen ordered float64 14-feature persistence vector per component",
            "keep probability >=0.5; zero only rejected component events",
            "frozen P18 recovery",
        ]
    ):
        raise ValueError("Frozen persistence stage chain differs.")
    if protocol["golden"] != {
        "routing": "M10 for event_count<=30000; released full-stream M20 otherwise",
        "counts": GOLDEN_COUNTS,
        "metrics": GOLDEN_METRICS,
    }:
        raise ValueError("Frozen golden baseline differs.")
    gates = protocol["promotion_gates"]
    if (
        gates["candidate_score_strictly_greater_than_golden_plus"]
        != MINIMUM_SCORE_DELTA
        or gates["true_positive_event_delta_equals_zero"] is not True
        or gates["correct_target_delta_equals_zero"] is not True
        or gates["positive_event_count_unchanged"] is not True
        or gates["target_frame_count_unchanged"] is not True
        or gates["frame_count_unchanged"] is not True
        or gates["each_h2_zero_true_positive_and_correct_target_loss"] is not True
        or gates["candidate_runtime_calls_equal_h2_count"] is not True
        or gates["t32_not_read_or_combined"] is not True
        or gates["all_required"] is not True
    ):
        raise ValueError("Frozen persistence promotion gates differ.")
    if protocol.get("runtime_contract") != EXPECTED_RUNTIME:
        raise ValueError("Frozen persistence runtime contract differs.")
    if protocol.get("outputs") != {
        "workspace_relative_directory": (
            "experiments/20260810_persistence_standalone_val24_sequence_v1"
        ),
        "execution_protocol": "execution_protocol.json",
        "preflight_receipt": "preflight_receipt.json",
        "attempt_claim": "validation_attempt_claim.json",
        "h2_cache": "h2_p0_persistence_p18_cache.pt",
        "report": "frozen_validation_report.json",
    }:
        raise ValueError("Frozen persistence output contract differs.")
    if _effective_c00_sha256() != EXPECTED_EFFECTIVE_C00_SHA256:
        raise ValueError("Effective C00 canonical SHA-256 differs.")
    return protocol


def _validate_candidate_inputs(science):
    paths = {
        name: _workspace_path(item["workspace_relative_path"], name)
        for name, item in science["candidate_inputs"].items()
    }
    for name, path in paths.items():
        expected = science["candidate_inputs"][name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError("Candidate input changed: {}".format(name))
    artifact = PersistenceArtifact.load(paths["artifact"], ARTIFACT_SHA256)
    train_report = _load_json_snapshot(
        paths["train_replay_report"],
        TRAIN_REPLAY_SHA256,
        "corrected H2-only train replay report",
    )
    if (
        train_report.get("passed") is not True
        or train_report.get("input_integrity", {}).get("h2_fit_source_count") != 11
        or train_report.get("runtime_replay", {}).get("h1_identity") is not True
        or train_report.get("runtime_replay", {}).get("only_h2_component_calls")
        is not True
        or train_report.get("runtime_replay", {}).get(
            "h2_zero_true_positive_and_correct_object_loss"
        )
        is not True
        or train_report.get("selection_disclosure", {}).get(
            "pooled_oof_delta_is_selection_affected_not_independent"
        )
        is not True
    ):
        raise ValueError("Corrected H2-only train replay evidence differs.")
    oof = _load_json_snapshot(
        paths["train_oof_report"], TRAIN_OOF_SHA256, "train-only OOF report"
    )
    if oof.get("pooled_oof", {}).get("conservative_winner_candidate_id") != "persistence_pw08_kp050":
        raise ValueError("Train OOF winner differs.")
    fit_protocol = _load_json_snapshot(
        paths["train_fit_protocol"],
        TRAIN_FIT_PROTOCOL_SHA256,
        "H2-only train-fit protocol",
    )
    if (
        fit_protocol.get("status") != "frozen_before_h2_only_final_fit"
        or fit_protocol.get("population", {}).get("fit_source_count") != 11
        or fit_protocol.get("estimator_correction", {}).get(
            "superseded_outputs_must_not_enter_validation"
        )
        is not True
    ):
        raise ValueError("Corrected train-fit protocol differs.")
    return paths, artifact


def build_execution_protocol(science, code):
    paths = _paths()
    return {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "created_utc": utc_now(),
        "status": "frozen_before_persistence_val24_preflight_or_claim",
        "evidence_class": science["evidence_class"],
        "science_protocol": {
            "path": str(SCIENCE_PROTOCOL_PATH),
            "sha256": EXPECTED_SCIENCE_PROTOCOL_SHA256,
        },
        "candidate_inputs": science["candidate_inputs"],
        "deferred_validation_inputs": science["deferred_validation_inputs"],
        "validation_dataset": science["validation_dataset"],
        "golden": science["golden"],
        "candidate_chain": science["candidate_chain"],
        "promotion_gates": science["promotion_gates"],
        "runtime_contract": science["runtime_contract"],
        "attempt_budget": science["attempt_budget"],
        "sequence_disclosure": science["sequence_disclosure"],
        "code_sha256": code,
        "outputs": {name: str(path) for name, path in paths.items() if name != "execution_protocol"},
        "preflight_contract": {
            "validation_npz_read": False,
            "validation_cache_read": False,
            "validation_label_read": False,
            "golden_report_read": False,
            "attempt_claimed": False,
        },
    }


def freeze_execution_protocol():
    paths = _paths()
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Persistence val24 protocol/receipt/claim/report path is occupied.")
    science = validate_science_protocol(
        _load_json_snapshot(
            SCIENCE_PROTOCOL_PATH,
            EXPECTED_SCIENCE_PROTOCOL_SHA256,
            "persistence val24 science protocol",
        )
    )
    _validate_candidate_inputs(science)
    payload = build_execution_protocol(science, _code_sha256())
    digest = _atomic_json_no_clobber(paths["execution_protocol"], payload)
    return {
        "path": str(paths["execution_protocol"]),
        "sha256": digest,
        "validation_npz_read": False,
        "validation_cache_read": False,
        "golden_report_read": False,
        "attempt_claimed": False,
    }


def validate_execution_protocol(protocol):
    if (
        protocol.get("schema") != EXECUTION_PROTOCOL_SCHEMA
        or protocol.get("status")
        != "frozen_before_persistence_val24_preflight_or_claim"
        or protocol.get("science_protocol", {}).get("sha256")
        != EXPECTED_SCIENCE_PROTOCOL_SHA256
        or protocol.get("science_protocol", {}).get("path")
        != str(SCIENCE_PROTOCOL_PATH)
        or protocol.get("code_sha256") != _code_sha256()
    ):
        raise ValueError("Persistence execution protocol identity differs.")
    science = validate_science_protocol(
        _load_json_snapshot(
            SCIENCE_PROTOCOL_PATH,
            EXPECTED_SCIENCE_PROTOCOL_SHA256,
            "persistence val24 science protocol",
        )
    )
    for key in (
        "candidate_inputs",
        "deferred_validation_inputs",
        "validation_dataset",
        "golden",
        "candidate_chain",
        "promotion_gates",
        "runtime_contract",
        "attempt_budget",
        "sequence_disclosure",
    ):
        if protocol[key] != science[key]:
            raise ValueError("Execution/science protocol field differs: {}".format(key))
    if protocol.get("evidence_class") != science["evidence_class"]:
        raise ValueError("Execution/science evidence class differs.")
    if protocol.get("preflight_contract") != {
        "validation_npz_read": False,
        "validation_cache_read": False,
        "validation_label_read": False,
        "golden_report_read": False,
        "attempt_claimed": False,
    }:
        raise ValueError("Execution preflight contract differs.")
    if set(protocol) != {
        "schema",
        "created_utc",
        "status",
        "evidence_class",
        "science_protocol",
        "candidate_inputs",
        "deferred_validation_inputs",
        "validation_dataset",
        "golden",
        "candidate_chain",
        "promotion_gates",
        "runtime_contract",
        "attempt_budget",
        "sequence_disclosure",
        "code_sha256",
        "outputs",
        "preflight_contract",
    }:
        raise ValueError("Execution protocol fields differ.")
    expected_outputs = {
        name: str(path)
        for name, path in _paths().items()
        if name != "execution_protocol"
    }
    if protocol["outputs"] != expected_outputs:
        raise ValueError("Execution output paths differ.")
    return protocol


def _load_execution(expected_sha256):
    return validate_execution_protocol(
        _load_json_snapshot(
            _paths()["execution_protocol"],
            str(expected_sha256).lower(),
            "persistence execution protocol",
        )
    )


def choose_persistence_scores(route, baseline_scores, h2_predictor):
    """Call the persistence component/model chain only on the observable H2 route."""
    eligible = bool(route["eligible"] if isinstance(route, dict) else route.eligible)
    if eligible:
        return h2_predictor(), False
    return baseline_scores, True


def classify_full_stream_route(event_count, polarities, temporal_bin_count=160):
    """Classify the released full-stream source and standalone H2 stage.

    This intentionally has no T32 mode or window metadata.
    """
    event_count = int(event_count)
    if int(temporal_bin_count) != 160:
        raise ValueError("Persistence validation requires exactly 160 temporal bins.")
    persistence = observable_route(event_count, polarities)
    if event_count <= LOW_EVENT_COUNT_MAX:
        domain = "low"
        score_source = "secondary"
        checkpoint_role = "m10"
    elif event_count <= 200_000:
        domain = "middle"
        score_source = "primary"
        checkpoint_role = "m20"
    elif persistence["eligible"]:
        domain = "h2"
        score_source = "primary"
        checkpoint_role = "m20"
    else:
        domain = "h1"
        score_source = "primary"
        checkpoint_role = "m20"
    return {
        "domain": domain,
        "event_count": event_count,
        "temporal_bin_count": int(temporal_bin_count),
        "polarity_minority_fraction": persistence[
            "polarity_minority_fraction"
        ],
        "checkpoint_role": checkpoint_role,
        "score_source": score_source,
        "mode": "full_stream",
        "persistence_eligible": persistence["eligible"],
        "t32_read_or_combined": False,
    }


def _synthetic_preflight(artifact):
    import torch
    from utils.postprocess import P0ClusterFilter, P18ScoreTrackRecovery

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized during CPU-only preflight.")
    suppressor = PersistenceComponentSuppressor(artifact)
    cfg = _c00_config()

    non_h2_scores = torch.linspace(0.0, 1.0, 16)
    non_h2_route = {"eligible": False}
    predictor_calls = 0

    def forbidden_predictor():
        nonlocal predictor_calls
        predictor_calls += 1
        raise RuntimeError("Non-H2 route invoked the persistence predictor.")

    non_h2_output, preserved = choose_persistence_scores(
        non_h2_route, non_h2_scores, forbidden_predictor
    )
    non_h2_identity = bool(
        preserved
        and non_h2_output is non_h2_scores
        and torch.equal(non_h2_output, non_h2_scores)
        and predictor_calls == 0
    )

    count = 200001
    polarities = np.arange(count, dtype=np.uint8) % 2
    x = np.arange(count, dtype=np.int64) % 346
    y = np.arange(count, dtype=np.int64) % 260
    t = np.arange(count, dtype=np.int64) % 8000
    x[:5] = 10
    y[:5] = 10
    t[:5] = np.arange(5, dtype=np.int64) * 50
    locations = np.column_stack((x, y, t))
    locations4 = torch.from_numpy(
        np.column_stack((np.zeros(count, dtype=np.int64), locations))
    )
    raw_scores = torch.zeros(count, dtype=torch.float32)
    raw_scores[:5] = 0.99
    p0 = P0ClusterFilter.from_cfg(cfg, M20_THRESHOLD, event_count=count)
    p0_scores, _ = p0.apply(raw_scores, locations4)
    route = observable_route(count, polarities)
    calls = 0

    def predict_h2():
        nonlocal calls
        calls += 1
        return suppressor.apply(p0_scores, locations, polarities)

    routed, preserved_h2 = choose_persistence_scores(route, p0_scores, predict_h2)
    suppressed_scores, stats = routed
    p18 = P18ScoreTrackRecovery.from_cfg(cfg, M20_THRESHOLD)
    final_scores, _ = p18.apply(suppressed_scores, locations4)
    h2_ok = bool(
        route["eligible"]
        and not preserved_h2
        and calls == 1
        and stats.component_chain_called
        and stats.candidate_component_count > 0
        and final_scores.shape == raw_scores.shape
        and torch.isfinite(final_scores).all()
        and not bool((final_scores > p0_scores).any())
    )
    cuda_initialized = bool(torch.cuda.is_initialized())
    return {
        "passed": non_h2_identity and h2_ok and not cuda_initialized,
        "non_h2_same_tensor_object_and_bits": non_h2_identity,
        "non_h2_predictor_calls": predictor_calls,
        "h2_predictor_calls": calls,
        "h2_component_chain_called": stats.component_chain_called,
        "h2_candidate_component_count": stats.candidate_component_count,
        "h2_kept_candidate_components": stats.kept_candidate_components,
        "h2_removed_candidate_components": stats.removed_candidate_components,
        "stage_order": [
            "raw M20 full-stream T160 probabilities",
            "P0/P0c",
            "topology r1/tbin50/link6/gap1/max_events3 component extraction",
            "ordered float64 14-feature persistence scoring",
            "keep>=0.5 and zero only rejected component events",
            "P18",
        ],
        "effective_c00_canonical_sha256": _effective_c00_sha256(),
        "t32_read_or_combined": False,
        "cuda_initialized": cuda_initialized,
        "validation_or_cache_read": False,
    }


def preflight_execution(expected_execution_sha256):
    paths = _paths()
    if paths["preflight_receipt"].exists():
        raise FileExistsError("Canonical preflight receipt already exists.")
    if any(paths[name].exists() for name in ("claim", "h2_cache", "report")):
        raise FileExistsError("Claim/cache/report path is already occupied.")
    protocol = _load_execution(expected_execution_sha256)
    science = _load_json_snapshot(
        SCIENCE_PROTOCOL_PATH,
        EXPECTED_SCIENCE_PROTOCOL_SHA256,
        "persistence val24 science protocol",
    )
    _, artifact = _validate_candidate_inputs(science)
    runtime = _runtime_identity()
    smoke = _synthetic_preflight(artifact)
    if smoke["passed"] is not True:
        raise RuntimeError("Persistence synthetic preflight failed.")
    payload = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "created_utc": utc_now(),
        "passed": True,
        "execution_protocol_sha256": str(expected_execution_sha256).lower(),
        "science_protocol_sha256": EXPECTED_SCIENCE_PROTOCOL_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "code_sha256": protocol["code_sha256"],
        "runtime": runtime,
        "synthetic_smoke": smoke,
        "deferred_until_after_claim": sorted(
            protocol["deferred_validation_inputs"].keys()
        ),
        "validation_npz_read": False,
        "validation_cache_read": False,
        "validation_label_read": False,
        "golden_report_read": False,
        "attempt_claimed": False,
    }
    digest = _atomic_json_no_clobber(paths["preflight_receipt"], payload)
    return {"path": str(paths["preflight_receipt"]), "sha256": digest, **payload}


def _load_preflight_receipt(execution_sha256):
    path = _paths()["preflight_receipt"]
    actual = sha256_file(path)
    receipt = _load_json_snapshot(path, actual, "persistence preflight receipt")
    runtime = _runtime_identity()
    smoke = receipt.get("synthetic_smoke", {})
    expected_stage_order = [
        "raw M20 full-stream T160 probabilities",
        "P0/P0c",
        "topology r1/tbin50/link6/gap1/max_events3 component extraction",
        "ordered float64 14-feature persistence scoring",
        "keep>=0.5 and zero only rejected component events",
        "P18",
    ]
    if (
        receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA
        or receipt.get("passed") is not True
        or receipt.get("execution_protocol_sha256") != execution_sha256
        or receipt.get("science_protocol_sha256") != EXPECTED_SCIENCE_PROTOCOL_SHA256
        or receipt.get("artifact_sha256") != ARTIFACT_SHA256
        or receipt.get("code_sha256") != _code_sha256()
        or receipt.get("runtime") != runtime
        or receipt.get("deferred_until_after_claim")
        != sorted(
            (
                "m10_checkpoint",
                "m20_checkpoint",
                "m10_golden_cache",
                "m20_golden_cache",
                "golden_report",
                "official_manifest",
            )
        )
        or smoke.get("passed") is not True
        or smoke.get("non_h2_same_tensor_object_and_bits") is not True
        or smoke.get("non_h2_predictor_calls") != 0
        or smoke.get("h2_predictor_calls") != 1
        or smoke.get("h2_component_chain_called") is not True
        or smoke.get("h2_candidate_component_count", 0) <= 0
        or smoke.get("h2_kept_candidate_components", -1)
        + smoke.get("h2_removed_candidate_components", -1)
        != smoke.get("h2_candidate_component_count")
        or smoke.get("stage_order") != expected_stage_order
        or smoke.get("effective_c00_canonical_sha256")
        != EXPECTED_EFFECTIVE_C00_SHA256
        or smoke.get("t32_read_or_combined") is not False
        or smoke.get("cuda_initialized") is not False
        or smoke.get("validation_or_cache_read") is not False
        or any(
            receipt.get(name) is not False
            for name in (
                "validation_npz_read",
                "validation_cache_read",
                "validation_label_read",
                "golden_report_read",
                "attempt_claimed",
            )
        )
    ):
        raise ValueError("Immutable persistence preflight receipt differs.")
    return receipt, actual


def _atomic_claim(path, execution_sha256, preflight_sha256):
    payload = {
        "schema": CLAIM_SCHEMA,
        "claimed_utc": utc_now(),
        "execution_protocol_sha256": execution_sha256,
        "preflight_receipt_sha256": preflight_sha256,
        "attempt": "1/1",
        "state": "irreversibly_claimed_before_any_validation_input_read",
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return payload, sha256_file(path)


def _deferred_paths(protocol):
    return {
        name: _workspace_path(item["workspace_relative_path"], name)
        for name, item in protocol["deferred_validation_inputs"].items()
    }


def _validate_validation_files_after_claim(protocol, deferred):
    if sha256_file(deferred["official_manifest"]) != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("Official validation manifest differs after claim.")
    val_root = deferred["official_manifest"].parent / "val"
    entries = protocol["validation_dataset"]["manifest_files"]
    actual = tuple(sorted(path.name for path in val_root.glob("*.npz") if path.is_file()))
    expected = tuple(Path(item["path"]).name for item in entries)
    if actual != expected:
        raise ValueError("Official validation directory population differs.")
    evidence = []
    for item in entries:
        path = (val_root / Path(item["path"]).name).resolve()
        if path.parent != val_root.resolve() or not path.is_file():
            raise ValueError("Noncanonical validation path after claim.")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != item["size"] or digest != item["sha256"]:
            raise ValueError("Validation source differs after claim: {}".format(path.name))
        evidence.append({"name": path.name, "size": size, "sha256": digest})
    if _semantic_manifest_sha256(entries) != OFFICIAL_SEMANTIC_SHA256:
        raise RuntimeError("Official validation semantic digest differs.")
    return val_root.resolve(), evidence


def _validate_cache_contract(primary, secondary, deferred):
    import replay_temporal_memory_validation as replay

    expected_inference = {
        "temporal_memory_bin_size": 50,
        "temporal_memory_context_bins": 5,
        "temporal_memory_width": 16,
        "temporal_memory_sequence_length": 16,
        "temporal_memory_inference_batch_size": 8,
        "temporal_memory_log_count_clip": 4.0,
        "whole_t": 8000,
        "resolution": [346, 260],
    }
    for name, payload, checkpoint_name, checkpoint_sha in (
        ("m20", primary, "m20_checkpoint", M20_CHECKPOINT_SHA256),
        ("m10", secondary, "m10_checkpoint", M10_CHECKPOINT_SHA256),
    ):
        metadata = payload["metadata"]
        if (
            metadata.get("dataset_split") != "val"
            or metadata.get("video_count") != OFFICIAL_VIDEO_COUNT
            or metadata.get("event_count") != OFFICIAL_EVENT_COUNT
            or metadata.get("dataset_signature") != OFFICIAL_DATASET_SIGNATURE
            or metadata.get("checkpoint_sha256") != checkpoint_sha
            or Path(metadata.get("checkpoint_path", "")).resolve()
            != deferred[checkpoint_name]
            or metadata.get("inference_settings") != expected_inference
        ):
            raise ValueError("{} golden cache contract differs.".format(name))
    binding = replay._validate_cache_compatibility(
        primary, secondary, secondary_max_events=LOW_EVENT_COUNT_MAX
    )
    records = replay.route_cache_records(primary, secondary, LOW_EVENT_COUNT_MAX)
    if tuple(Path(item.file_name).stem for item in records) != OFFICIAL_STEMS:
        raise ValueError("Golden cache record order differs.")
    return binding, records


def _validate_raw_alignment(video, record):
    cached_locations = record.locs.detach().cpu().numpy()
    if not np.array_equal(cached_locations[:, 1:4], video.locations):
        raise ValueError("Raw/cache validation locations differ.")
    if not np.array_equal(record.seg_label.detach().cpu().numpy().reshape(-1), video.labels):
        raise ValueError("Raw/cache validation labels differ.")
    if not np.array_equal(np.asarray(record.idx_label).reshape(-1), video.target_ids):
        raise ValueError("Raw/cache validation target ids differ.")


def _counts_from_postprocessed(record, scores):
    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay

    counts = component_crossfit.sufficient_counts_for_video(
        scores.detach().cpu().numpy(),
        record.seg_label.detach().cpu().numpy(),
        record.idx_label,
        record.locs.detach().cpu().numpy(),
        prediction_threshold=M20_THRESHOLD,
    )
    return replay.ChallengeCountTotals(
        true_positive_events=counts.true_positive_events,
        false_positive_events=counts.false_positive_events,
        positive_events=counts.true_positive_events + counts.false_negative_events,
        detected_target_frames=counts.correct_objects,
        target_frames=counts.object_count,
        false_components=counts.false_components,
        frame_count=counts.frame_count,
    )


def _gate_results(
    baseline_counts,
    baseline,
    candidate,
    candidate_counts,
    preservation,
    calls,
    h2_count,
    each_h2_zero_loss,
):
    return {
        "golden_baseline_exact_match": baseline_counts == GOLDEN_COUNTS
        and baseline == GOLDEN_METRICS,
        "candidate_score_strictly_greater_than_golden_plus_0p0001": candidate[
            "score"
        ]
        > GOLDEN_METRICS["score"] + MINIMUM_SCORE_DELTA,
        "candidate_pd_not_lower_than_golden": candidate["pd"] >= GOLDEN_METRICS["pd"],
        "candidate_iou_not_lower_than_golden": candidate["iou"] >= GOLDEN_METRICS["iou"],
        "candidate_fa_not_higher_than_golden": candidate["fa"] <= GOLDEN_METRICS["fa"],
        "true_positive_event_delta_equals_zero": candidate_counts[
            "true_positive_events"
        ]
        == baseline_counts["true_positive_events"],
        "correct_target_delta_equals_zero": candidate_counts[
            "detected_target_frames"
        ]
        == baseline_counts["detected_target_frames"],
        "positive_event_count_unchanged": candidate_counts["positive_events"]
        == baseline_counts["positive_events"],
        "target_frame_count_unchanged": candidate_counts["target_frames"]
        == baseline_counts["target_frames"],
        "frame_count_unchanged": candidate_counts["frame_count"]
        == baseline_counts["frame_count"],
        "each_h2_zero_true_positive_and_correct_target_loss": bool(
            each_h2_zero_loss
        ),
        "all_non_h2_scores_bitwise_and_object_preserved": bool(preservation),
        "candidate_runtime_calls_equal_h2_count": int(calls) == int(h2_count),
        "t32_not_read_or_combined": True,
    }


def _run_claimed(protocol, deferred, artifact, execution_sha256):
    import torch
    import replay_temporal_memory_validation as replay
    from dataset.temporal_frame import load_temporal_frame_video
    from utils.postprocess import P0ClusterFilter, P18ScoreTrackRecovery

    val_root, validation_evidence = _validate_validation_files_after_claim(
        protocol, deferred
    )
    expected_hashes = {
        name: protocol["deferred_validation_inputs"][name]["sha256"]
        for name in protocol["deferred_validation_inputs"]
    }
    observed_hashes = {name: sha256_file(path) for name, path in deferred.items()}
    if observed_hashes != expected_hashes:
        raise ValueError("A deferred validation input differs after claim.")
    secondary, secondary_sha = replay.load_cache_snapshot(deferred["m10_golden_cache"])
    primary, primary_sha = replay.load_cache_snapshot(deferred["m20_golden_cache"])
    if secondary_sha != M10_CACHE_SHA256 or primary_sha != M20_CACHE_SHA256:
        raise RuntimeError("Golden validation cache changed while loading.")
    binding, records = _validate_cache_contract(primary, secondary, deferred)
    cfg = _c00_config()
    suppressor = PersistenceComponentSuppressor(artifact)
    baseline_counts = []
    candidate_counts = []
    per_video = []
    runtime_calls = 0
    non_h2_preserved = True
    h2_count = 0
    each_h2_zero_loss = True
    h2_cache_records = []
    validation_sha_by_name = {
        item["name"]: item["sha256"] for item in validation_evidence
    }
    for index, record in enumerate(records, start=1):
        file_name = Path(record.file_name).name
        raw_path = val_root / file_name
        raw_sha_before = sha256_file(raw_path)
        if raw_sha_before != validation_sha_by_name[file_name]:
            raise RuntimeError("Validation source changed before load: {}".format(file_name))
        video = load_temporal_frame_video(raw_path, 50, 8000)
        raw_sha_after = sha256_file(raw_path)
        if raw_sha_after != raw_sha_before:
            raise RuntimeError("Validation source changed while loading: {}".format(file_name))
        decision = classify_full_stream_route(
            record.event_count,
            video.polarities,
            len(video.event_indices_by_bin),
        )
        persistence_route = observable_route(record.event_count, video.polarities)
        if persistence_route["eligible"] != (decision["domain"] == "h2"):
            raise RuntimeError(
                "Input-domain and persistence routes differ: {}".format(file_name)
            )
        _validate_raw_alignment(video, record)
        if decision["event_count"] != record.event_count:
            raise ValueError("Raw/cache event counts differ: {}".format(file_name))
        expected_source = decision["score_source"]
        if record.score_source != expected_source:
            raise RuntimeError("Golden route source differs: {}".format(file_name))
        threshold = LOW_THRESHOLD if decision["domain"] == "low" else M20_THRESHOLD
        baseline_count = replay.evaluate_cached_video(record, threshold, cfg)
        if decision["domain"] == "h2":
            h2_count += 1
            p0 = P0ClusterFilter.from_cfg(cfg, M20_THRESHOLD, event_count=record.event_count)
            p0_scores, _ = p0.apply(record.scores.clone(), record.locs)
            def predict_h2():
                nonlocal runtime_calls
                runtime_calls += 1
                return suppressor.apply(p0_scores, video.locations, video.polarities)

            routed, preserved = choose_persistence_scores(
                persistence_route, p0_scores, predict_h2
            )
            suppressed, stats = routed
            if preserved or not stats.component_chain_called:
                raise RuntimeError("H2 persistence runtime was not called.")
            p18 = P18ScoreTrackRecovery.from_cfg(cfg, M20_THRESHOLD)
            candidate_scores, _ = p18.apply(suppressed, record.locs)
            candidate_count = _counts_from_postprocessed(record, candidate_scores)
            video_zero_loss = bool(
                candidate_count.true_positive_events
                == baseline_count.true_positive_events
                and candidate_count.detected_target_frames
                == baseline_count.detected_target_frames
            )
            each_h2_zero_loss = each_h2_zero_loss and video_zero_loss
            h2_cache_records.append(
                {
                    "file_name": file_name,
                    "source_sha256": record.source_sha256,
                    "event_count": int(record.event_count),
                    "route": dict(persistence_route),
                    "p0_p0c_scores": p0_scores.detach()
                    .cpu()
                    .to(torch.float32)
                    .reshape(-1)
                    .contiguous(),
                    "persistence_scores": suppressed.detach()
                    .cpu()
                    .to(torch.float32)
                    .reshape(-1)
                    .contiguous(),
                    "p18_final_scores": candidate_scores.detach()
                    .cpu()
                    .to(torch.float32)
                    .reshape(-1)
                    .contiguous(),
                    "runtime_stats": stats.to_dict(),
                    "zero_true_positive_and_correct_target_loss": video_zero_loss,
                }
            )
        else:
            h2_predictor_calls = 0

            def forbidden():
                nonlocal h2_predictor_calls
                h2_predictor_calls += 1
                raise RuntimeError("Non-H2 source called the persistence runtime.")

            candidate_scores, preserved = choose_persistence_scores(
                persistence_route, record.scores, forbidden
            )
            preserved = bool(
                preserved
                and candidate_scores is record.scores
                and candidate_scores.data_ptr() == record.scores.data_ptr()
                and torch.equal(candidate_scores, record.scores)
                and h2_predictor_calls == 0
            )
            non_h2_preserved = non_h2_preserved and preserved
            candidate_count = baseline_count
            stats = None
        baseline_counts.append(baseline_count)
        candidate_counts.append(candidate_count)
        per_video.append(
            {
                "index": index,
                "file_name": file_name,
                "route": dict(decision),
                "persistence_observable_route": dict(persistence_route),
                "candidate_mode": (
                    "full_stream_m20_p0_persistence_p18"
                    if decision["domain"] == "h2"
                    else "released_full_stream_identity"
                ),
                "t32_read_or_combined": False,
                "baseline_score_source": record.score_source,
                "candidate_score_source": (
                    "m20_full_p0_persistence_p18"
                    if decision["domain"] == "h2"
                    else record.score_source
                ),
                "persistence_runtime_called": decision["domain"] == "h2",
                "scores_bitwise_and_object_preserved": preserved,
                "runtime_stats": None if stats is None else stats.to_dict(),
                "baseline_counts": asdict(baseline_count),
                "candidate_counts": asdict(candidate_count),
            }
        )
        print(
            "persistence/evaluate {}/24: {} -> {}".format(
                index, file_name, decision["domain"]
            ),
            flush=True,
        )

    h2_cache_payload = {
        "schema": "ev-uav-persistence-standalone-h2-score-cache-v1",
        "created_utc": utc_now(),
        "execution_protocol_sha256": execution_sha256,
        "science_protocol_sha256": EXPECTED_SCIENCE_PROTOCOL_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "effective_c00_canonical_sha256": EXPECTED_EFFECTIVE_C00_SHA256,
        "runtime": _runtime_identity(),
        "stage_order": protocol["candidate_chain"]["h2_stage_order"],
        "t32_read_or_combined": False,
        "video_count": len(h2_cache_records),
        "records": h2_cache_records,
    }
    replay._atomic_torch_save(h2_cache_payload, _paths()["h2_cache"], overwrite=False)
    h2_cache_sha256 = sha256_file(_paths()["h2_cache"])

    baseline_total = replay._sum_counts(baseline_counts)
    candidate_total = replay._sum_counts(candidate_counts)
    baseline_count_dict = asdict(baseline_total)
    candidate_count_dict = asdict(candidate_total)
    baseline_metrics = replay.metrics_from_counts_exact(baseline_total, cfg).to_dict()
    candidate_metrics = replay.metrics_from_counts_exact(candidate_total, cfg).to_dict()
    gates = _gate_results(
        baseline_count_dict,
        baseline_metrics,
        candidate_metrics,
        candidate_count_dict,
        non_h2_preserved,
        runtime_calls,
        h2_count,
        each_h2_zero_loss,
    )
    val_root_after, validation_evidence_after = (
        _validate_validation_files_after_claim(protocol, deferred)
    )
    observed_hashes_after = {
        name: sha256_file(path) for name, path in deferred.items()
    }
    runtime_after = _runtime_identity()
    integrity_preserved = bool(
        val_root_after == val_root
        and validation_evidence_after == validation_evidence
        and observed_hashes_after == observed_hashes
        and sha256_file(_paths()["h2_cache"]) == h2_cache_sha256
        and runtime_after["cuda_initialized"] is False
    )
    if not integrity_preserved:
        raise RuntimeError("Validation/cache/runtime integrity changed during replay.")
    return {
        "passed": all(gates.values()),
        "validation_integrity": {
            "before": validation_evidence,
            "after": validation_evidence_after,
            "unchanged": True,
        },
        "deferred_input_sha256": {
            "before": observed_hashes,
            "after": observed_hashes_after,
            "unchanged": True,
        },
        "runtime_after": runtime_after,
        "h2_cache": {
            "path": str(_paths()["h2_cache"]),
            "sha256": h2_cache_sha256,
            "schema": h2_cache_payload["schema"],
            "video_count": len(h2_cache_records),
        },
        "golden_cache_binding": binding,
        "baseline": {"counts": baseline_count_dict, "metrics": baseline_metrics},
        "candidate": {"counts": candidate_count_dict, "metrics": candidate_metrics},
        "metric_delta": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
        },
        "count_delta": {
            key: candidate_count_dict[key] - baseline_count_dict[key]
            for key in baseline_count_dict
        },
        "gates": gates,
        "h2_count": h2_count,
        "runtime_calls": runtime_calls,
        "per_video": per_video,
    }


def run_execution(expected_execution_sha256):
    paths = _paths()
    if any(paths[name].exists() for name in ("claim", "h2_cache", "report")):
        raise FileExistsError("Persistence attempt was already claimed, cached, or reported.")
    protocol = _load_execution(expected_execution_sha256)
    science = _load_json_snapshot(
        SCIENCE_PROTOCOL_PATH,
        EXPECTED_SCIENCE_PROTOCOL_SHA256,
        "persistence val24 science protocol",
    )
    _, artifact = _validate_candidate_inputs(science)
    receipt, receipt_sha = _load_preflight_receipt(expected_execution_sha256)
    live_smoke = _synthetic_preflight(artifact)
    if live_smoke != receipt["synthetic_smoke"]:
        raise RuntimeError(
            "Live CPU persistence smoke differs from the immutable preflight receipt."
        )
    if sha256_file(paths["preflight_receipt"]) != receipt_sha:
        raise RuntimeError("Preflight receipt changed before the attempt claim.")
    claim, claim_sha = _atomic_claim(
        paths["claim"], expected_execution_sha256, receipt_sha
    )
    try:
        deferred = _deferred_paths(protocol)
        outcome = _run_claimed(
            protocol, deferred, artifact, str(expected_execution_sha256).lower()
        )
        report = {
            "schema": REPORT_SCHEMA,
            "created_utc": utc_now(),
            "passed": outcome["passed"],
            "evidence_class": protocol["evidence_class"],
            "sequence_disclosure": protocol["sequence_disclosure"],
            "execution_protocol": {
                "path": str(paths["execution_protocol"]),
                "sha256": expected_execution_sha256,
            },
            "preflight_receipt": {"payload": receipt, "sha256": receipt_sha},
            "attempt_claim": {"payload": claim, "sha256": claim_sha},
            "artifact_sha256": ARTIFACT_SHA256,
            "t32_read_or_combined": False,
            **outcome,
            "failure_action": (
                None
                if outcome["passed"]
                else "archive_without_validation_retuning_or_second_persistence_attempt"
            ),
        }
    except BaseException as error:
        report = {
            "schema": REPORT_SCHEMA,
            "created_utc": utc_now(),
            "passed": False,
            "evidence_class": protocol["evidence_class"],
            "sequence_disclosure": protocol["sequence_disclosure"],
            "execution_protocol_sha256": expected_execution_sha256,
            "preflight_receipt_sha256": receipt_sha,
            "attempt_claim": {"payload": claim, "sha256": claim_sha},
            "stage": "after_claim",
            "error_type": type(error).__name__,
            "error": str(error),
            "t32_read_or_combined": False,
            "failure_action": "archive_without_validation_retuning_or_second_persistence_attempt",
        }
    digest = _atomic_json_no_clobber(paths["report"], report)
    return {"path": str(paths["report"]), "sha256": digest, **report}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    for name in ("preflight", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("expected_execution_protocol_sha256")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "freeze":
        result = freeze_execution_protocol()
    elif args.command == "preflight":
        result = preflight_execution(args.expected_execution_protocol_sha256)
    else:
        result = run_execution(args.expected_execution_protocol_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""One-shot frozen 24-val replay for the train-selected temporal input route.

The four CLI stages deliberately have different access budgets:

``freeze``
    Reads only repository code and previously published, non-cache provenance.
    It does not open the validation manifest, a validation NPZ, or either raw
    score cache.  It writes the sole canonical execution protocol.
``preflight``
    Verifies the frozen protocol, clean git/code identity, checkpoints and
    published reports.  It still does not open a validation/cache payload.
``runtime-preflight``
    Uses the frozen GPU Python environment to import the complete replay stack,
    validate CUDA/model compatibility, and execute a synthetic M20 T32 smoke
    inference.  It writes an immutable receipt without opening validation data,
    labels, manifests, or raw score caches and does not consume attempt 1/1.
``run``
    Repeats the synthetic runtime smoke check and only then creates a durable
    O_EXCL 1/1 attempt claim.  Only after the claim may it hash
    validation files, load the two frozen golden caches, and run M20 T32/stride
    16 inference for videos whose input-only route is H2.  Every other candidate
    score tensor retains the exact baseline cache storage and bits.

This is an opt-in experiment runner.  It does not modify the released
submission path, create a submission ZIP, or upload anything.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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


SCIENCE_PROTOCOL_SCHEMA = "ev-uav-temporal-input-route-val24-science-protocol-v1"
EXECUTION_PROTOCOL_SCHEMA = "ev-uav-temporal-input-route-val24-execution-protocol-v1"
CLAIM_SCHEMA = "ev-uav-temporal-input-route-val24-attempt-claim-v1"
REPORT_SCHEMA = "ev-uav-temporal-input-route-val24-report-v1"
H2_CACHE_SCHEMA = "ev-uav-temporal-input-route-val24-h2-cache-v1"
RUNTIME_RECEIPT_SCHEMA = "ev-uav-temporal-input-route-val24-runtime-preflight-v1"

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SCIENCE_PROTOCOL_PATH = (
    PROJECT_ROOT / "protocols" / "temporal_input_route_val24_science.json"
).resolve()
FROZEN_EXPERIMENT_DIRECTORY = (
    WORKSPACE_ROOT / "experiments" / "20260810_temporal_input_route_frozen_val24"
).resolve()

M10_CHECKPOINT_SHA256 = "5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935"
M20_CHECKPOINT_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
M10_CACHE_SHA256 = "96a9dfa8833e6f609d29f4db9d8f7196c84c7e92c7026cce734b97ddf133622f"
M20_CACHE_SHA256 = "6c9b4a8e33217aac7a05c78590a7feb6db6e6fc332b6411d7603264687710304"
GOLDEN_REPORT_SHA256 = "da6004ddd22731b8e848c9ed0c561961abbc04b4e3f66cd07b1e085d26f9f383"
PRIOR_PROTOCOL_SHA256 = "8523a411e35bcfb0d0c78cbffd8a98a7983d4eeedd7f09d8a7b10e8a6200477d"
OFFICIAL_MANIFEST_SHA256 = "c7c574b5dfa8336fe50917581544b5e4991b2cde197f31c9a5bee05a29e336d4"
OFFICIAL_SEMANTIC_SHA256 = "d780da17e69446b988b1b5fae7954855d5ce66a32aa7b9581eeb3e4a0563f83f"
OFFICIAL_DATASET_SIGNATURE = "bedba93c1d523f58c35da6399219df1b98e6240f92d093520fa0f4961d927274"
TRAIN_V3_CACHE_MANIFEST_SHA256 = "78ca63efd1fd8fda62dcccb1203f0e69000007454a391b7d46455f9952cf2dc7"
TRAIN_V3_EVALUATION_SHA256 = "4e3610a057c9c330b18f9d0712d57fe80a1463bc4ae567f64172986409d6e956"
TRAIN_V3_PROTOCOL_SHA256 = "ddd027961bc36f2756a62cd62914c5be3400a2ddd965d53ab2ff066b331f36d1"

EXPECTED_RUNTIME = {
    "python_version": "3.9.25",
    "torch_version": "2.5.1+cu121",
    "numpy_version": "1.26.4",
    "platform": "Windows-10-10.0.26100-SP0",
    "cuda_runtime": "12.1",
    "cuda_device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
}

OFFICIAL_VIDEO_COUNT = 24
OFFICIAL_EVENT_COUNT = 1_424_330
OFFICIAL_STEMS = tuple("val_{:03d}".format(index) for index in range(24))
LOW_EVENT_COUNT_MAX = 30_000
HIGH_EVENT_COUNT_MAX = 200_000
LOW_THRESHOLD = 0.718
M20_THRESHOLD = 0.719
POLARITY_MINORITY_CUTOFF = 0.20
H2_WINDOW_LENGTH = 32
H2_STRIDE = 16
MINIMUM_SCORE_DELTA = 0.0001

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
    "pd_detT": 50,
    "p0_enabled": True,
    "p0_spatial_radius": 2,
    "p0_temporal_bin_size": 50,
    "p0_temporal_radius_bins": 1,
    "p0_min_cluster_events": 3,
    "p0_min_duration_bins": 5,
    "p0c_high_confidence_recovery_enabled": True,
    "p0c_retain_min_score": 0.95,
    "p0c_density_retain_enabled": False,
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
    "component_reranker_enabled": False,
}

CODE_PATHS = (
    "evaluate_temporal_input_route_validation.py",
    "protocols/temporal_input_route_val24_science.json",
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
    "utils/temporal_memory_input_router.py",
    "utils/temporal_memory_windowed_inference.py",
)

INPUT_NAMES = (
    "m10_checkpoint",
    "m20_checkpoint",
    "m10_golden_cache",
    "m20_golden_cache",
    "golden_report",
    "prior_t32_execution_protocol",
    "official_manifest",
    "train_v3_cache_manifest",
    "train_v3_evaluation",
    "train_v3_protocol",
)
DEFERRED_VAL_INPUTS = frozenset(
    {"m10_golden_cache", "m20_golden_cache", "official_manifest"}
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value, name):
    value = str(value).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("{} must be a lowercase SHA-256.".format(name))
    return value


def _canonical_path(value, name):
    path = Path(str(value))
    if not path.is_absolute() or str(path) != str(path.resolve()):
        raise ValueError("{} must be an absolute resolved path.".format(name))
    return path.resolve()


def _canonical_paths():
    directory = Path(FROZEN_EXPERIMENT_DIRECTORY).resolve()
    return {
        "science_protocol": Path(SCIENCE_PROTOCOL_PATH).resolve(),
        "execution_protocol": directory / "preregistered_execution_protocol.json",
        "runtime_receipt": directory / "runtime_preflight_receipt.json",
        "claim": directory / "validation_attempt_claim.json",
        "h2_cache": directory / "raw_m20_t32_stride16_h2_only.pt",
        "report": directory / "frozen_validation_report.json",
    }


def _canonical_inputs():
    return {
        "m10_checkpoint": (
            PROJECT_ROOT / "checkpoints" / "m10_dense_views2_epoch_002_seed42.pt"
        ).resolve(),
        "m20_checkpoint": (
            PROJECT_ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
        ).resolve(),
        "m10_golden_cache": (
            WORKSPACE_ROOT
            / "experiments"
            / "20260810_dacc_v2_projection_only_seed49"
            / "replay"
            / "m10_val24_raw.pt"
        ).resolve(),
        "m20_golden_cache": (
            WORKSPACE_ROOT
            / "experiments"
            / "20260810_baseline_fine_sweep"
            / "m20_val24_raw.pt"
        ).resolve(),
        "golden_report": (
            WORKSPACE_ROOT
            / "results"
            / "submission_m20_golden"
            / "offline_score_report.json"
        ).resolve(),
        "prior_t32_execution_protocol": (
            WORKSPACE_ROOT
            / "experiments"
            / "20260810_t32_attention_projection_seed49"
            / "validation"
            / "preregistered_execution_protocol.json"
        ).resolve(),
        "official_manifest": (
            WORKSPACE_ROOT
            / "datasets"
            / "EV-UAV-Challenge2"
            / "official_google_drive_manifest.json"
        ).resolve(),
        "train_v3_cache_manifest": (
            WORKSPACE_ROOT
            / "experiments"
            / "20260810_temporal_input_route_v1"
            / "formal_train_score_cache_v3"
            / "manifest.json"
        ).resolve(),
        "train_v3_evaluation": (
            WORKSPACE_ROOT
            / "experiments"
            / "20260810_temporal_input_route_v1"
            / "formal_train_route_evaluation_v3.json"
        ).resolve(),
        "train_v3_protocol": (
            WORKSPACE_ROOT
            / "experiments"
            / "20260810_temporal_input_route_v1"
            / "frozen_train_cache_eval_protocol_v3.json"
        ).resolve(),
    }


def _expected_input_sha256():
    return {
        "m10_checkpoint": M10_CHECKPOINT_SHA256,
        "m20_checkpoint": M20_CHECKPOINT_SHA256,
        "m10_golden_cache": M10_CACHE_SHA256,
        "m20_golden_cache": M20_CACHE_SHA256,
        "golden_report": GOLDEN_REPORT_SHA256,
        "prior_t32_execution_protocol": PRIOR_PROTOCOL_SHA256,
        "official_manifest": OFFICIAL_MANIFEST_SHA256,
        "train_v3_cache_manifest": TRAIN_V3_CACHE_MANIFEST_SHA256,
        "train_v3_evaluation": TRAIN_V3_EVALUATION_SHA256,
        "train_v3_protocol": TRAIN_V3_PROTOCOL_SHA256,
    }


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


def _code_sha256():
    result = {}
    for relative in CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError("Required code file is missing: {}".format(path))
        result[relative] = sha256_file(path)
    return result


def _load_json_snapshot(path, expected_sha256=None, name="JSON"):
    path = Path(path).resolve()
    before = sha256_file(path)
    if expected_sha256 is not None and before != _require_sha256(expected_sha256, name + " SHA-256"):
        raise ValueError("{} SHA-256 differs from the frozen value.".format(name))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while it was read.".format(name))
    return payload, before


def _extract_json_member(text, member):
    """Decode one member even when unrelated legacy Windows paths are malformed."""
    marker = json.dumps(member) + ":"
    start = text.find(marker)
    if start < 0:
        raise ValueError("JSON evidence lacks {}.".format(member))
    value_start = start + len(marker)
    value_start += len(text[value_start:]) - len(text[value_start:].lstrip())
    value, _ = json.JSONDecoder().raw_decode(text, value_start)
    return value


def _validate_train_prerequisite_files(protocol, input_paths):
    """Require the exact completed 99-train v3 evidence before val is claimable."""
    expected = _expected_input_sha256()
    for name in ("train_v3_cache_manifest", "train_v3_evaluation", "train_v3_protocol"):
        if sha256_file(input_paths[name]) != expected[name]:
            raise ValueError("{} differs from frozen train-v3 evidence.".format(name))

    manifest, manifest_sha = _load_json_snapshot(
        input_paths["train_v3_cache_manifest"],
        TRAIN_V3_CACHE_MANIFEST_SHA256,
        "train-v3 cache manifest",
    )
    if (
        manifest.get("schema") != "ev-uav-temporal-input-route-train-cache-v1"
        or manifest.get("complete") is not True
        or manifest.get("video_count") != 99
        or manifest.get("route_policy_sha256") != protocol["route_policy"]["sha256"]
        or manifest.get("protocol", {}).get("sha256") != TRAIN_V3_PROTOCOL_SHA256
    ):
        raise ValueError("Train-v3 cache manifest identity is incomplete.")
    split = manifest.get("split_access", {})
    if (
        split.get("dataset_split") != "train"
        or split.get("validation_or_test_read") is not False
        or split.get("labels_or_target_ids_indexed") is not False
    ):
        raise ValueError("Train-v3 cache was not label-free train-only inference.")
    if manifest.get("route_counts") != {
        "m20/full_stream": 43,
        "m10/full_stream": 45,
        "m20/window_t32": 11,
    }:
        raise ValueError("Train-v3 route population differs from frozen 45/43/11.")
    if manifest.get("route_gates") != {
        "expected_45_m10_full": True,
        "expected_43_m20_full": True,
        "expected_11_m20_t32": True,
        "unchanged_88_bitwise_equal": True,
    }:
        raise ValueError("Train-v3 cache route gates did not all pass.")
    shared_runtime = (
        "utils/temporal_memory_input_router.py",
        "utils/temporal_memory_windowed_inference.py",
        "utils/temporal_memory_inference.py",
        "utils/postprocess.py",
        "utils/eval.py",
        "utils/challenge_eval.py",
    )
    manifest_code = manifest.get("code", {}).get("sha256", {})
    drift = [
        relative
        for relative in shared_runtime
        if manifest_code.get(relative) != sha256_file(PROJECT_ROOT / relative)
    ]
    if drift:
        raise ValueError(
            "Formal val runtime differs from train-v3 evidence: {}.".format(
                ", ".join(drift)
            )
        )
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 99:
        raise ValueError("Train-v3 cache manifest does not cover 99 sources.")
    h2 = [record for record in records if record.get("decision", {}).get("domain") == "h2"]
    non_h2 = [record for record in records if record.get("decision", {}).get("domain") != "h2"]
    if (
        len(h2) != 11
        or len(non_h2) != 88
        or not all(record.get("bitwise_equal_to_baseline") is True for record in non_h2)
        or not all(record.get("bitwise_equal_to_baseline") is False for record in h2)
    ):
        raise ValueError("Train-v3 88 identity / 11 changed contract did not pass.")

    evaluation_path = input_paths["train_v3_evaluation"]
    before = sha256_file(evaluation_path)
    text = evaluation_path.read_text(encoding="utf-8")
    if sha256_file(evaluation_path) != before or before != TRAIN_V3_EVALUATION_SHA256:
        raise RuntimeError("Train-v3 evaluation changed while being read.")
    if _extract_json_member(text, "schema") != "ev-uav-temporal-input-route-train-evaluation-v1":
        raise ValueError("Train-v3 evaluation schema differs.")
    evaluation_split = _extract_json_member(text, "split_access")
    if (
        evaluation_split.get("dataset_split") != "train"
        or evaluation_split.get("validation_or_test_read") is not False
        or evaluation_split.get("route_uses_labels_or_source_name") is not False
    ):
        raise ValueError("Train-v3 evaluation is not train-only/route-independent.")
    if _extract_json_member(text, "protocol").get("sha256") != TRAIN_V3_PROTOCOL_SHA256:
        raise ValueError("Train-v3 evaluation protocol binding differs.")
    if _extract_json_member(text, "cache").get("manifest_sha256") != TRAIN_V3_CACHE_MANIFEST_SHA256:
        raise ValueError("Train-v3 evaluation cache binding differs.")
    pooled = _extract_json_member(text, "pooled")
    delta = pooled.get("delta", {}).get("metrics")
    expected_delta = protocol["train_prerequisite"]["pooled_delta"]
    if delta != expected_delta:
        raise ValueError("Train-v3 pooled metric delta differs from frozen evidence.")
    gates = {
        "score_delta_strictly_positive": delta["score"] > 0.0,
        "pd_delta_nondecreasing": delta["pd"] >= 0.0,
        "iou_delta_nondecreasing": delta["iou"] >= 0.0,
        "fa_delta_nonincreasing": delta["fa"] <= 0.0,
        "non_h2_identity_complete": len(non_h2) == 88
        and all(record["bitwise_equal_to_baseline"] for record in non_h2),
        "h2_changed_complete": len(h2) == 11
        and all(not record["bitwise_equal_to_baseline"] for record in h2),
        "all_required": True,
    }
    if not all(value for name, value in gates.items() if name != "all_required"):
        raise ValueError("Train-v3 prerequisite promotion gates did not all pass.")
    return {
        "passed": True,
        "manifest_sha256": manifest_sha,
        "evaluation_sha256": before,
        "protocol_sha256": TRAIN_V3_PROTOCOL_SHA256,
        "source_count": 99,
        "route_counts": {"non_h2": 88, "h2": 11},
        "pooled_delta": delta,
        "gates": gates,
    }


def _atomic_json_no_clobber(path, payload):
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
        name = Path(entry["path"]).name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry["sha256"]))
    return digest.hexdigest()


def validate_science_protocol(protocol):
    if not isinstance(protocol, Mapping) or protocol.get("schema") != SCIENCE_PROTOCOL_SCHEMA:
        raise ValueError("Unsupported science protocol schema.")
    if protocol.get("status") != "frozen_before_any_new_val24_access":
        raise ValueError("Science protocol is not frozen before validation access.")
    if protocol.get("candidate_id") != "m10_m20_polarity_h2_t32_stride16_c00_no_persistence":
        raise ValueError("Science protocol candidate identity differs.")
    if protocol.get("attempt_budget") != 1:
        raise ValueError("Science protocol must have attempt_budget=1.")

    dataset = protocol.get("validation_dataset", {})
    if (
        dataset.get("split") != "val"
        or dataset.get("video_count") != OFFICIAL_VIDEO_COUNT
        or dataset.get("event_count") != OFFICIAL_EVENT_COUNT
        or tuple(dataset.get("canonical_stems", ())) != OFFICIAL_STEMS
        or dataset.get("dataset_signature") != OFFICIAL_DATASET_SIGNATURE
        or dataset.get("official_manifest_sha256") != OFFICIAL_MANIFEST_SHA256
        or dataset.get("semantic_sha256") != OFFICIAL_SEMANTIC_SHA256
    ):
        raise ValueError("Science protocol validation population differs from official val24.")
    evidence = dataset.get("manifest_evidence_source", {})
    if evidence != {
        "path_role": "prior_t32_execution_protocol",
        "sha256": PRIOR_PROTOCOL_SHA256,
        "member": "validation_dataset.manifest_files",
        "new_dataset_read_required_to_freeze": False,
    }:
        raise ValueError("Science protocol manifest evidence is not the prior frozen list.")
    entries = dataset.get("manifest_files")
    if not isinstance(entries, list) or len(entries) != OFFICIAL_VIDEO_COUNT:
        raise ValueError("Science protocol must bind 24 manifest entries.")
    expected_paths = tuple("val/{}.npz".format(stem) for stem in OFFICIAL_STEMS)
    if tuple(entry.get("path") for entry in entries) != expected_paths:
        raise ValueError("Science protocol manifest paths are not canonical and ordered.")
    for entry in entries:
        if set(entry) != {"path", "size", "sha256"} or int(entry["size"]) <= 0:
            raise ValueError("Science protocol contains a malformed manifest entry.")
        _require_sha256(entry["sha256"], "manifest member")
    if _semantic_manifest_sha256(entries) != OFFICIAL_SEMANTIC_SHA256:
        raise ValueError("Science protocol manifest member list has drifted.")

    expected_inputs = {
        "m10_checkpoint": M10_CHECKPOINT_SHA256,
        "m20_checkpoint": M20_CHECKPOINT_SHA256,
        "m10_golden_cache": M10_CACHE_SHA256,
        "m20_golden_cache": M20_CACHE_SHA256,
        "golden_report": GOLDEN_REPORT_SHA256,
        "train_v3_cache_manifest": TRAIN_V3_CACHE_MANIFEST_SHA256,
        "train_v3_evaluation": TRAIN_V3_EVALUATION_SHA256,
        "train_v3_protocol": TRAIN_V3_PROTOCOL_SHA256,
    }
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_inputs):
        raise ValueError("Science protocol input bindings are incomplete.")
    for name, expected in expected_inputs.items():
        if inputs[name].get("sha256") != expected:
            raise ValueError("Science protocol {} SHA-256 differs.".format(name))

    expected_train_prerequisite = {
        "split": "train",
        "complete_source_count": 99,
        "validation_or_test_read": False,
        "route_uses_labels_or_source_name": False,
        "route_policy_sha256": (
            "f7835bc89d2ad4c3d29b46acc23f237bafd89e05cc1d983208fddf492480a836"
        ),
        "route_counts": {"non_h2": 88, "h2": 11},
        "identity_requirement": (
            "all 88 non-H2 candidate score vectors bitwise equal baseline"
        ),
        "change_requirement": (
            "all 11 H2 candidate score vectors differ from baseline"
        ),
        "pooled_delta": {
            "iou": 0.014221429824829102,
            "acc": 0.0013789534568786621,
            "pd": 0.00014801657785668,
            "fa": -8.313546995842769e-07,
            "score_fa": 0.007428358228430798,
            "score": 0.005269895410325631,
        },
        "required_gates": {
            "score_delta_strictly_positive": True,
            "pd_delta_nondecreasing": True,
            "iou_delta_nondecreasing": True,
            "fa_delta_nonincreasing": True,
            "non_h2_identity_complete": True,
            "h2_changed_complete": True,
            "all_required": True,
        },
        "provenance_limitation": {
            "severity": "P2",
            "retroactive_full_train_to_val_code_equivalence_provable": False,
            "unrecorded_transitive_files": [
                "dataset/temporal_frame.py",
                "model/modules/confidence_head.py",
                "model/temporal_frame_net.py",
                "model/temporal_memory_net.py",
                "utils/multiscale_motion.py",
                "utils/density_threshold.py",
                "utils/component_reranker.py",
            ],
            "mitigation": (
                "formal val freezes a clean git HEAD and expanded current runtime "
                "code hashes; train-v3 is not rerun or retroactively altered"
            ),
        },
    }
    if protocol.get("train_prerequisite") != expected_train_prerequisite:
        raise ValueError("Science protocol train-v3 prerequisite differs.")

    route = protocol.get("route_policy", {})
    definition = route.get("definition")
    if not isinstance(definition, Mapping) or route.get("sha256") != canonical_sha256(definition):
        raise ValueError("Science protocol route policy hash differs.")
    expected_route_members = {
        "low_density": {
            "condition": "event_count <= 30000",
            "checkpoint_role": "m10",
            "mode": "full_stream",
            "temporal_length": 160,
        },
        "middle_density": {
            "condition": "30000 < event_count <= 200000",
            "checkpoint_role": "m20",
            "mode": "full_stream",
            "temporal_length": 160,
        },
        "h1": {"mode": "full_stream", "temporal_length": 160},
        "h2": {
            "mode": "window_t32_stride16",
            "window_length": 32,
            "stride": 16,
            "stitch": "nearest_window_center_ties_to_earlier_window",
        },
        "prediction_threshold_by_checkpoint": {"m10": 0.718, "m20": 0.719},
        "persistent_pixel_second_stage": {
            "enabled": False,
            "status": "disabled_pending_routed_train_oof_interaction",
        },
    }
    for name, expected in expected_route_members.items():
        if definition.get(name) != expected:
            raise ValueError("Science protocol route differs at {}.".format(name))
    if (
        definition.get("cutoff") != POLARITY_MINORITY_CUTOFF
        or definition.get("cutoff_operator") != "<"
        or definition.get("labels_used_for_route") is not False
        or definition.get("source_name_used_for_route") is not False
    ):
        raise ValueError("Science protocol route independence/cutoff differs.")

    inference = protocol.get("inference", {})
    expected_inference = {
        "whole_t": 8000,
        "temporal_bin_size": 50,
        "temporal_bin_count": 160,
        "context_bins": 5,
        "model_width": 16,
        "resolution": [346, 260],
        "inference_batch_size": 8,
        "log_count_clip": 4.0,
        "h2_window_length": 32,
        "h2_stride": 16,
        "device": "cuda:0",
    }
    if inference != expected_inference:
        raise ValueError("Science protocol inference settings differ.")
    if protocol.get("runtime_preflight") != {
        "required_before_attempt_claim": True,
        "synthetic_only": True,
        "validation_or_cache_read": False,
        "expected_environment": EXPECTED_RUNTIME,
    }:
        raise ValueError("Science protocol runtime preflight contract differs.")
    if protocol.get("thresholds") != {
        "low_event_count_max": LOW_EVENT_COUNT_MAX,
        "low": LOW_THRESHOLD,
        "otherwise": M20_THRESHOLD,
    }:
        raise ValueError("Science protocol thresholds differ.")
    if protocol.get("postprocess") != {"profile": "C00", "settings": C00_SETTINGS}:
        raise ValueError("Science protocol is not exact C00.")
    if protocol.get("persistence", {}).get("enabled") is not False:
        raise ValueError("Persistent-pixel stage must remain disabled.")
    if protocol.get("golden") != {"counts": GOLDEN_COUNTS, "metrics": GOLDEN_METRICS}:
        raise ValueError("Science protocol golden baseline differs.")
    expected_gates = {
        "golden_baseline_exact_match": True,
        "candidate_score_strictly_greater_than_golden_plus": MINIMUM_SCORE_DELTA,
        "candidate_pd_not_lower_than_golden": True,
        "candidate_iou_not_lower_than_golden": True,
        "candidate_fa_not_higher_than_golden": True,
        "all_non_h2_scores_bitwise_preserved": True,
        "h2_only_reinference": True,
        "all_gates_required": True,
        "failure_action": "archive_without_validation_retuning_or_second_attempt",
    }
    if protocol.get("promotion_gates") != expected_gates:
        raise ValueError("Science protocol promotion gates differ.")
    if protocol.get("side_effect_limits") != {
        "submission_zip": False,
        "platform_upload": False,
        "threshold_search": False,
    }:
        raise ValueError("Science protocol side-effect limits differ.")
    return protocol


def validate_execution_protocol(protocol):
    if not isinstance(protocol, Mapping) or protocol.get("schema") != EXECUTION_PROTOCOL_SCHEMA:
        raise ValueError("Unsupported execution protocol schema.")
    expected_top = {
        "schema",
        "created_utc",
        "attempt_budget",
        "science_protocol",
        "repository",
        "inputs",
        "train_prerequisite",
        "validation_dataset",
        "route_policy",
        "inference",
        "runtime_preflight",
        "thresholds",
        "postprocess",
        "persistence",
        "cache_reuse",
        "golden",
        "promotion_gates",
        "side_effect_limits",
        "outputs",
    }
    if set(protocol) != expected_top or protocol.get("attempt_budget") != 1:
        raise ValueError("Execution protocol top-level schema or attempt budget differs.")
    science = protocol.get("science_protocol", {})
    if set(science) != {"path", "sha256", "payload"}:
        raise ValueError("Execution protocol science binding is malformed.")
    if _canonical_path(science["path"], "science protocol") != SCIENCE_PROTOCOL_PATH:
        raise ValueError("Execution protocol uses a noncanonical science protocol.")
    _require_sha256(science["sha256"], "science protocol")
    validate_science_protocol(science["payload"])

    repository = protocol.get("repository", {})
    if set(repository) != {"project_root", "expected_clean_git_head", "code_sha256"}:
        raise ValueError("Execution protocol repository binding is malformed.")
    if _canonical_path(repository["project_root"], "project root") != PROJECT_ROOT:
        raise ValueError("Execution protocol project root differs.")
    head = str(repository["expected_clean_git_head"]).lower()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ValueError("Execution protocol git HEAD is invalid.")
    code = repository.get("code_sha256")
    if not isinstance(code, Mapping) or set(code) != set(CODE_PATHS):
        raise ValueError("Execution protocol code binding is incomplete.")
    for relative in CODE_PATHS:
        _require_sha256(code[relative], "code hash for " + relative)

    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_NAMES):
        raise ValueError("Execution protocol input binding is incomplete.")
    canonical_inputs = _canonical_inputs()
    expected_hashes = _expected_input_sha256()
    for name in INPUT_NAMES:
        binding = inputs[name]
        if set(binding) != {"path", "sha256"}:
            raise ValueError("Execution protocol {} binding is malformed.".format(name))
        if _canonical_path(binding["path"], name) != canonical_inputs[name]:
            raise ValueError("Execution protocol {} path differs.".format(name))
        if binding["sha256"] != expected_hashes[name]:
            raise ValueError("Execution protocol {} SHA-256 differs.".format(name))

    payload = science["payload"]
    copied = (
        "validation_dataset",
        "train_prerequisite",
        "route_policy",
        "inference",
        "runtime_preflight",
        "thresholds",
        "postprocess",
        "persistence",
        "cache_reuse",
        "golden",
        "promotion_gates",
        "side_effect_limits",
    )
    for name in copied:
        if protocol.get(name) != payload.get(name):
            raise ValueError("Execution protocol changed frozen {}.".format(name))
    outputs = protocol.get("outputs", {})
    canonical_outputs = _canonical_paths()
    if set(outputs) != {"runtime_receipt", "claim", "h2_cache", "report"}:
        raise ValueError("Execution protocol outputs are malformed.")
    for name in outputs:
        if _canonical_path(outputs[name], "output " + name) != canonical_outputs[name]:
            raise ValueError("Execution protocol output {} differs.".format(name))
    return protocol


def build_execution_protocol(science_protocol, science_sha256, git, code_sha256):
    """Build a frozen protocol without touching validation files or caches."""
    validate_science_protocol(science_protocol)
    if not git.get("clean"):
        raise RuntimeError("Git worktree must be clean before protocol freeze.")
    paths = _canonical_paths()
    inputs = _canonical_inputs()
    hashes = _expected_input_sha256()
    protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "created_utc": _utc_now(),
        "attempt_budget": 1,
        "science_protocol": {
            "path": str(paths["science_protocol"]),
            "sha256": science_sha256,
            "payload": science_protocol,
        },
        "repository": {
            "project_root": str(PROJECT_ROOT),
            "expected_clean_git_head": git["head"],
            "code_sha256": code_sha256,
        },
        "inputs": {
            name: {"path": str(inputs[name]), "sha256": hashes[name]}
            for name in INPUT_NAMES
        },
        "outputs": {
            name: str(paths[name])
            for name in ("runtime_receipt", "claim", "h2_cache", "report")
        },
    }
    for name in (
        "validation_dataset",
        "train_prerequisite",
        "route_policy",
        "inference",
        "runtime_preflight",
        "thresholds",
        "postprocess",
        "persistence",
        "cache_reuse",
        "golden",
        "promotion_gates",
        "side_effect_limits",
    ):
        protocol[name] = science_protocol[name]
    validate_execution_protocol(protocol)
    return protocol


def freeze_execution_protocol():
    paths = _canonical_paths()
    if any(
        paths[name].exists()
        for name in ("execution_protocol", "runtime_receipt", "claim", "h2_cache", "report")
    ):
        raise FileExistsError("Canonical protocol/claim/cache/report path is already occupied.")
    science, science_sha = _load_json_snapshot(paths["science_protocol"], name="science protocol")
    validate_science_protocol(science)
    git = _git_state()
    if not git["clean"]:
        raise RuntimeError("Commit all candidate code before freezing the execution protocol.")
    code = _code_sha256()
    # Deliberately verify only non-validation/non-cache evidence at freeze time.
    inputs = _canonical_inputs()
    expected = _expected_input_sha256()
    for name in INPUT_NAMES:
        if name in DEFERRED_VAL_INPUTS:
            continue
        if sha256_file(inputs[name]) != expected[name]:
            raise ValueError("{} differs from its preregistered hash.".format(name))
    train_prerequisite = _validate_train_prerequisite_files(science, inputs)
    protocol = build_execution_protocol(science, science_sha, git, code)
    digest = _atomic_json_no_clobber(paths["execution_protocol"], protocol)
    return {
        "execution_protocol": str(paths["execution_protocol"]),
        "sha256": digest,
        "validation_npz_read": False,
        "validation_cache_read": False,
        "attempt_claimed": False,
        "train_prerequisite": train_prerequisite,
    }


def _load_execution_protocol(expected_protocol_sha256):
    paths = _canonical_paths()
    expected = _require_sha256(expected_protocol_sha256, "execution protocol")
    protocol, actual = _load_json_snapshot(
        paths["execution_protocol"], expected, "execution protocol"
    )
    validate_execution_protocol(protocol)
    return protocol, paths, actual


def _preclaim_validate(expected_protocol_sha256):
    protocol, paths, protocol_sha = _load_execution_protocol(expected_protocol_sha256)
    for name in ("claim", "h2_cache", "report"):
        if paths[name].exists():
            raise FileExistsError("Canonical {} already exists: {}".format(name, paths[name]))
    if not paths["execution_protocol"].parent.is_dir():
        raise FileNotFoundError("Frozen experiment directory is missing.")
    git = _git_state()
    if not git["clean"] or git["head"] != protocol["repository"]["expected_clean_git_head"]:
        raise RuntimeError("Git state differs from the frozen execution protocol.")
    code = _code_sha256()
    if code != protocol["repository"]["code_sha256"]:
        raise RuntimeError("Code hashes differ from the frozen execution protocol.")
    science, science_sha = _load_json_snapshot(
        paths["science_protocol"], protocol["science_protocol"]["sha256"], "science protocol"
    )
    if science != protocol["science_protocol"]["payload"]:
        raise ValueError("Science protocol payload differs from execution protocol.")
    validate_science_protocol(science)

    # No official manifest, validation NPZ, or raw score cache is touched here.
    verified_non_val = {}
    input_paths = {name: Path(protocol["inputs"][name]["path"]) for name in INPUT_NAMES}
    for name in INPUT_NAMES:
        if name in DEFERRED_VAL_INPUTS:
            continue
        digest = sha256_file(input_paths[name])
        if digest != protocol["inputs"][name]["sha256"]:
            raise ValueError("{} differs from execution protocol.".format(name))
        verified_non_val[name] = digest
    if science_sha != protocol["science_protocol"]["sha256"]:
        raise RuntimeError("Science protocol changed during preclaim checks.")
    train_prerequisite = _validate_train_prerequisite_files(protocol, input_paths)
    return (
        protocol,
        paths,
        protocol_sha,
        git,
        code,
        input_paths,
        verified_non_val,
        train_prerequisite,
    )


def preflight_execution(expected_protocol_sha256):
    state = _preclaim_validate(expected_protocol_sha256)
    protocol, _, protocol_sha, git, code, _, verified, train_prerequisite = state
    return {
        "protocol_sha256": protocol_sha,
        "attempt_budget": protocol["attempt_budget"],
        "git": git,
        "code_sha256": code,
        "verified_non_validation_inputs": verified,
        "deferred_until_after_claim": sorted(DEFERRED_VAL_INPUTS),
        "validation_npz_read": False,
        "validation_cache_read": False,
        "validation_scored": False,
        "claim_created": False,
        "train_prerequisite": train_prerequisite,
    }


def _runtime_environment_identity(torch, np):
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 1:
        raise RuntimeError("CUDA device 0 is unavailable for the frozen H2 replay.")
    actual = {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
    }
    differences = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in EXPECTED_RUNTIME.items()
        if actual.get(name) != expected
    }
    if differences:
        raise RuntimeError(
            "Runtime differs from frozen train-v3 environment: {}.".format(
                json.dumps(differences, sort_keys=True)
            )
        )
    return actual


def _synthetic_runtime_arrays(np, protocol):
    """Build label-free 0/1-polarity inputs for the pre-claim smoke check."""
    temporal_bins = np.arange(
        protocol["inference"]["temporal_bin_count"], dtype=np.int64
    )
    locations = np.column_stack(
        (
            temporal_bins % protocol["inference"]["resolution"][0],
            (temporal_bins * 3) % protocol["inference"]["resolution"][1],
            temporal_bins * protocol["inference"]["temporal_bin_size"] + 1,
        )
    ).astype(np.int64, copy=False)
    polarities = (temporal_bins % 2).astype(np.float32, copy=False)
    route_probe = np.ones(HIGH_EVENT_COUNT_MAX + 2, dtype=np.float32)
    route_probe[::2] = 0.0
    return locations, polarities, route_probe


def _prepare_runtime_before_claim(protocol, input_paths):
    """Load and smoke-test the exact GPU stack without reading validation inputs."""
    import numpy as np
    import torch

    import replay_temporal_memory_validation as replay
    from dataset.temporal_frame import (
        load_temporal_frame_video,
        temporal_frame_video_from_events,
    )
    from utils.temporal_memory_inference import load_temporal_memory_model
    from utils.temporal_memory_input_router import (
        route_policy_definition,
        route_policy_sha256,
        select_temporal_memory_input_route,
    )
    from utils.temporal_memory_windowed_inference import (
        predict_temporal_memory_scores_windowed,
    )

    if protocol["persistence"]["enabled"] is not False:
        raise RuntimeError("Persistence stage was not disabled.")
    if route_policy_definition() != protocol["route_policy"]["definition"]:
        raise RuntimeError("Runtime route definition differs from frozen policy.")
    if route_policy_sha256() != protocol["route_policy"]["sha256"]:
        raise RuntimeError("Runtime route SHA-256 differs from frozen policy.")
    for owner, names in (
        (
            replay,
            (
                "RoutedRecord",
                "_atomic_torch_save",
                "_sum_counts",
                "evaluate_cached_video",
                "load_cache_snapshot",
                "metrics_from_counts_exact",
                "route_cache_records",
            ),
        ),
    ):
        missing = [name for name in names if not callable(getattr(owner, name, None))]
        if missing:
            raise RuntimeError("Replay API is incomplete: {}.".format(", ".join(missing)))

    runtime = _runtime_environment_identity(torch, np)
    device = torch.device(protocol["inference"]["device"])
    torch.cuda.set_device(device)
    model, _ = load_temporal_memory_model(
        str(input_paths["m20_checkpoint"]),
        device,
        protocol["inference"]["context_bins"],
        protocol["inference"]["model_width"],
        16,
    )

    locations, polarities, route_probe = _synthetic_runtime_arrays(np, protocol)
    synthetic_video = temporal_frame_video_from_events(
        name="synthetic_runtime_preflight",
        locations=locations,
        polarities=polarities,
        temporal_bin_size=protocol["inference"]["temporal_bin_size"],
        whole_t=protocol["inference"]["whole_t"],
        labels=None,
        target_ids=None,
    )

    decision = select_temporal_memory_input_route(
        route_probe, len(synthetic_video.event_indices_by_bin)
    )
    decision_metadata = decision.to_metadata()
    if (
        decision.domain != "h2"
        or decision.checkpoint_role != "m20"
        or decision.window_length != H2_WINDOW_LENGTH
        or decision.stride != H2_STRIDE
        or decision.prediction_threshold != M20_THRESHOLD
    ):
        raise RuntimeError("Synthetic H2 route does not match the frozen candidate.")

    scores = predict_temporal_memory_scores_windowed(
        model=model,
        video=synthetic_video,
        device=device,
        context_bins=protocol["inference"]["context_bins"],
        width=protocol["inference"]["resolution"][0],
        height=protocol["inference"]["resolution"][1],
        inference_batch_size=protocol["inference"]["inference_batch_size"],
        log_count_clip=protocol["inference"]["log_count_clip"],
        window_length=H2_WINDOW_LENGTH,
        stride=H2_STRIDE,
    ).detach().cpu().to(torch.float32).reshape(-1).contiguous()
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(scores).all())
    probabilities = finite and not bool((scores < 0).any()) and not bool((scores > 1).any())
    if scores.numel() != locations.shape[0] or not probabilities:
        raise RuntimeError("Synthetic M20 T32 smoke scores are malformed.")

    smoke = {
        "synthetic_only": True,
        "validation_or_cache_read": False,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "checkpoint_sha256": protocol["inputs"]["m20_checkpoint"]["sha256"],
        "temporal_bin_count": len(synthetic_video.event_indices_by_bin),
        "event_score_count": int(scores.numel()),
        "scores_finite": finite,
        "scores_in_probability_range": probabilities,
        "route": decision_metadata,
    }
    bundle = {
        "torch": torch,
        "numpy": np,
        "replay": replay,
        "load_temporal_frame_video": load_temporal_frame_video,
        "select_temporal_memory_input_route": select_temporal_memory_input_route,
        "predict_temporal_memory_scores_windowed": predict_temporal_memory_scores_windowed,
        "model": model,
    }
    return runtime, smoke, bundle


def _runtime_receipt_payload(protocol_sha, git, code, runtime, smoke, paths):
    evidence = {
        "execution_protocol_sha256": protocol_sha,
        "repository": {"git": git, "code_sha256": code},
        "runtime": runtime,
        "smoke": smoke,
    }
    return {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "created_utc": _utc_now(),
        "passed": True,
        "execution_protocol": {
            "path": str(paths["execution_protocol"]),
            "sha256": protocol_sha,
        },
        "repository": evidence["repository"],
        "runtime": runtime,
        "smoke": smoke,
        "evidence_sha256": canonical_sha256(evidence),
        "validation_npz_read": False,
        "validation_cache_read": False,
        "validation_labels_read": False,
        "attempt_claimed": False,
    }


def runtime_preflight_execution(expected_protocol_sha256):
    state = _preclaim_validate(expected_protocol_sha256)
    protocol, paths, protocol_sha, git, code, input_paths = state[:6]
    if paths["runtime_receipt"].exists():
        raise FileExistsError(
            "Canonical runtime preflight receipt already exists: {}".format(
                paths["runtime_receipt"]
            )
        )
    runtime, smoke, bundle = _prepare_runtime_before_claim(protocol, input_paths)
    payload = _runtime_receipt_payload(
        protocol_sha, git, code, runtime, smoke, paths
    )
    digest = _atomic_json_no_clobber(paths["runtime_receipt"], payload)
    # The standalone command exits immediately; explicitly release its smoke model.
    del bundle["model"]
    bundle["torch"].cuda.empty_cache()
    return {
        "runtime_receipt": str(paths["runtime_receipt"]),
        "sha256": digest,
        "runtime": runtime,
        "smoke": smoke,
        "validation_npz_read": False,
        "validation_cache_read": False,
        "validation_labels_read": False,
        "attempt_claimed": False,
    }


def _load_runtime_receipt(protocol_sha, git, code, paths):
    receipt, digest = _load_json_snapshot(
        paths["runtime_receipt"], name="runtime preflight receipt"
    )
    if receipt.get("schema") != RUNTIME_RECEIPT_SCHEMA or receipt.get("passed") is not True:
        raise ValueError("Runtime preflight receipt is not a passing v1 receipt.")
    if receipt.get("execution_protocol") != {
        "path": str(paths["execution_protocol"]),
        "sha256": protocol_sha,
    }:
        raise ValueError("Runtime receipt protocol binding differs.")
    if receipt.get("repository") != {"git": git, "code_sha256": code}:
        raise ValueError("Runtime receipt repository binding differs.")
    for name in (
        "validation_npz_read",
        "validation_cache_read",
        "validation_labels_read",
        "attempt_claimed",
    ):
        if receipt.get(name) is not False:
            raise ValueError("Runtime receipt violates the no-validation/no-claim contract.")
    runtime = receipt.get("runtime")
    smoke = receipt.get("smoke")
    if not isinstance(runtime, Mapping) or not isinstance(smoke, Mapping):
        raise ValueError("Runtime receipt lacks runtime/smoke evidence.")
    for name, expected in EXPECTED_RUNTIME.items():
        if runtime.get(name) != expected:
            raise ValueError("Runtime receipt differs at {}.".format(name))
    if (
        smoke.get("synthetic_only") is not True
        or smoke.get("validation_or_cache_read") is not False
        or smoke.get("scores_finite") is not True
        or smoke.get("scores_in_probability_range") is not True
        or smoke.get("route", {}).get("domain") != "h2"
    ):
        raise ValueError("Runtime receipt smoke contract did not pass.")
    evidence = {
        "execution_protocol_sha256": protocol_sha,
        "repository": receipt["repository"],
        "runtime": runtime,
        "smoke": smoke,
    }
    if receipt.get("evidence_sha256") != canonical_sha256(evidence):
        raise ValueError("Runtime receipt evidence digest differs.")
    return receipt, digest


def _atomic_claim(path, protocol_sha256, report_path, runtime_receipt_sha256):
    payload = {
        "schema": CLAIM_SCHEMA,
        "claimed_utc": _utc_now(),
        "attempt": 1,
        "attempt_budget": 1,
        "execution_protocol_path": str(_canonical_paths()["execution_protocol"]),
        "execution_protocol_sha256": protocol_sha256,
        "runtime_preflight_receipt_sha256": _require_sha256(
            runtime_receipt_sha256, "runtime preflight receipt"
        ),
        "report_path": str(Path(report_path).resolve()),
        "state": "irreversibly_claimed_before_any_val_or_cache_read",
    }
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(Path(path).resolve()), flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive path intentionally remains and consumes attempt 1/1.
        raise
    return payload, sha256_file(path)


def _validate_validation_files_after_claim(protocol, input_paths):
    if sha256_file(input_paths["official_manifest"]) != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("Official manifest differs after claim.")
    dataset_root = input_paths["official_manifest"].parent
    val_root = (dataset_root / "val").resolve()
    entries = protocol["validation_dataset"]["manifest_files"]
    expected_names = tuple(Path(entry["path"]).name for entry in entries)
    actual_names = tuple(sorted(path.name for path in val_root.glob("*.npz") if path.is_file()))
    if actual_names != expected_names:
        raise ValueError("Validation directory is not exactly val_000..val_023.")
    evidence = []
    for entry in entries:
        name = Path(entry["path"]).name
        path = (val_root / name).resolve()
        if path.parent != val_root or not path.is_file():
            raise ValueError("Validation path is noncanonical: {}".format(path))
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(entry["size"]) or digest != entry["sha256"]:
            raise ValueError("Validation file differs from frozen manifest: {}".format(name))
        evidence.append({"name": name, "size": size, "sha256": digest})
    if _semantic_manifest_sha256(entries) != OFFICIAL_SEMANTIC_SHA256:
        raise RuntimeError("Validation semantic digest differs after claim.")
    return {
        "manifest_path": str(input_paths["official_manifest"]),
        "manifest_sha256": OFFICIAL_MANIFEST_SHA256,
        "semantic_sha256": OFFICIAL_SEMANTIC_SHA256,
        "video_count": len(evidence),
        "files": evidence,
    }


def _c00_config():
    values = dict(C00_SETTINGS)
    values.update(
        {
            "roc": True,
            "correct_thresh": 0.0001,
            "res": [346, 260],
            "p0c_density_event_count_cutoff": 100000,
            "p0c_density_retain_min_score": 0.97,
        }
    )
    return SimpleNamespace(**values)


def _delta(candidate, baseline):
    return {name: float(candidate[name]) - float(baseline[name]) for name in baseline}


def _candidate_score_source(decision):
    if decision.domain == "low":
        return "golden_m10_cache"
    if decision.domain in {"middle", "h1"}:
        return "golden_m20_cache"
    if decision.domain == "h2":
        return "m20_t32_stride16"
    raise ValueError("Unknown route domain: {}".format(decision.domain))


def choose_candidate_scores(decision, baseline_scores, h2_predictor):
    """Pure routing boundary used to audit that only H2 can call inference."""
    domain = decision.domain if hasattr(decision, "domain") else decision["domain"]
    if domain == "h2":
        return h2_predictor(), False
    if domain not in {"low", "middle", "h1"}:
        raise ValueError("Unknown route domain: {}".format(domain))
    return baseline_scores, True


def promotion_gate_results(baseline_counts, baseline, candidate, preservation, calls, h2_count):
    return {
        "golden_baseline_exact_match": (
            baseline_counts == GOLDEN_COUNTS and baseline == GOLDEN_METRICS
        ),
        "candidate_score_strictly_greater_than_golden_plus_0p0001": (
            float(candidate["score"]) > float(GOLDEN_METRICS["score"]) + MINIMUM_SCORE_DELTA
        ),
        "candidate_pd_not_lower_than_golden": candidate["pd"] >= GOLDEN_METRICS["pd"],
        "candidate_iou_not_lower_than_golden": candidate["iou"] >= GOLDEN_METRICS["iou"],
        "candidate_fa_not_higher_than_golden": candidate["fa"] <= GOLDEN_METRICS["fa"],
        "all_non_h2_scores_bitwise_preserved": bool(preservation),
        "h2_only_reinference": int(calls) == int(h2_count),
        "persistence_disabled": True,
    }


def _validate_loaded_golden_caches(replay, protocol, primary, secondary, input_paths):
    inference = protocol["inference"]
    expected_inference_settings = {
        "temporal_memory_bin_size": inference["temporal_bin_size"],
        "temporal_memory_context_bins": inference["context_bins"],
        "temporal_memory_width": inference["model_width"],
        # Both released full-stream caches and the M20 model are T16-trained;
        # T32 is solely the frozen reset/stitch inference window length.
        "temporal_memory_sequence_length": 16,
        "temporal_memory_inference_batch_size": inference["inference_batch_size"],
        "temporal_memory_log_count_clip": inference["log_count_clip"],
        "whole_t": inference["whole_t"],
        "resolution": inference["resolution"],
    }
    for name, payload, checkpoint_name in (
        ("m20", primary, "m20_checkpoint"),
        ("m10", secondary, "m10_checkpoint"),
    ):
        metadata = payload["metadata"]
        if (
            metadata.get("dataset_split") != "val"
            or int(metadata.get("video_count", -1)) != OFFICIAL_VIDEO_COUNT
            or int(metadata.get("event_count", -1)) != OFFICIAL_EVENT_COUNT
            or metadata.get("dataset_signature") != OFFICIAL_DATASET_SIGNATURE
        ):
            raise ValueError("{} golden cache is not the official val24 population.".format(name))
        if metadata.get("checkpoint_sha256") != protocol["inputs"][checkpoint_name]["sha256"]:
            raise ValueError("{} golden cache checkpoint hash differs.".format(name))
        if Path(str(metadata.get("checkpoint_path", ""))).resolve() != input_paths[checkpoint_name]:
            raise ValueError("{} golden cache checkpoint path differs.".format(name))
        if metadata.get("inference_settings") != expected_inference_settings:
            raise ValueError(
                "{} golden cache inference settings differ from the frozen protocol.".format(
                    name
                )
            )
    binding = replay._validate_cache_compatibility(
        primary, secondary, secondary_max_events=LOW_EVENT_COUNT_MAX
    )
    records = replay.route_cache_records(primary, secondary, LOW_EVENT_COUNT_MAX)
    if tuple(Path(record.file_name).stem for record in records) != OFFICIAL_STEMS:
        raise ValueError("Golden cache record order is not canonical.")
    for record in records:
        expected = "secondary" if record.event_count <= LOW_EVENT_COUNT_MAX else "primary"
        if record.score_source != expected:
            raise RuntimeError("Released <=30k M10 / >30k M20 route differs.")
    return binding, records


def _validate_raw_alignment(video, record, np):
    cached_locs = record.locs.detach().cpu().numpy()
    if cached_locs.ndim != 2 or cached_locs.shape[1] != 4:
        raise ValueError("Golden cache locations must be [batch,x,y,t].")
    if not np.array_equal(cached_locs[:, 1:4], video.locations):
        raise ValueError("Raw validation locations differ from golden cache.")
    if not np.array_equal(record.seg_label.detach().cpu().numpy().reshape(-1), video.labels):
        raise ValueError("Raw validation labels differ from golden cache.")
    if not np.array_equal(np.asarray(record.idx_label).reshape(-1), video.target_ids):
        raise ValueError("Raw validation target ids differ from golden cache.")


def _run_claimed(protocol, paths, input_paths, runtime_bundle):
    # All imports, CUDA checks, model loading, and API smoke checks completed
    # before the irreversible claim.  Only validation/cache access starts here.
    torch = runtime_bundle["torch"]
    np = runtime_bundle["numpy"]
    replay = runtime_bundle["replay"]
    load_temporal_frame_video = runtime_bundle["load_temporal_frame_video"]
    select_temporal_memory_input_route = runtime_bundle[
        "select_temporal_memory_input_route"
    ]
    predict_temporal_memory_scores_windowed = runtime_bundle[
        "predict_temporal_memory_scores_windowed"
    ]

    validation_before = _validate_validation_files_after_claim(protocol, input_paths)
    cache_hashes_before = {}
    for name in ("m10_golden_cache", "m20_golden_cache"):
        digest = sha256_file(input_paths[name])
        if digest != protocol["inputs"][name]["sha256"]:
            raise ValueError("{} differs after claim.".format(name))
        cache_hashes_before[name] = digest
    secondary, secondary_sha = replay.load_cache_snapshot(input_paths["m10_golden_cache"])
    primary, primary_sha = replay.load_cache_snapshot(input_paths["m20_golden_cache"])
    if secondary_sha != cache_hashes_before["m10_golden_cache"] or primary_sha != cache_hashes_before["m20_golden_cache"]:
        raise RuntimeError("Golden cache changed while loading.")
    binding, baseline_records = _validate_loaded_golden_caches(
        replay, protocol, primary, secondary, input_paths
    )

    cfg = _c00_config()
    device = torch.device(protocol["inference"]["device"])
    model = runtime_bundle["model"]
    inference_calls = 0
    h2_records = []
    baseline_counts = []
    candidate_counts = []
    per_video = []
    all_non_h2_preserved = True
    val_root = input_paths["official_manifest"].parent / "val"
    file_hash_by_name = {
        entry["name"]: entry["sha256"] for entry in validation_before["files"]
    }

    for index, baseline_record in enumerate(baseline_records, start=1):
        file_name = Path(baseline_record.file_name).name
        video = load_temporal_frame_video(
            val_root / file_name,
            protocol["inference"]["temporal_bin_size"],
            protocol["inference"]["whole_t"],
        )
        decision = select_temporal_memory_input_route(
            video.polarities, len(video.event_indices_by_bin)
        )
        # This integrity check is after the input-only decision and cannot affect it.
        _validate_raw_alignment(video, baseline_record, np)
        if decision.event_count != baseline_record.event_count:
            raise ValueError("Route event count differs from golden cache for {}.".format(file_name))
        expected_baseline_source = "secondary" if decision.domain == "low" else "primary"
        if baseline_record.score_source != expected_baseline_source:
            raise RuntimeError("Golden route source differs for {}.".format(file_name))

        def infer_h2():
            nonlocal inference_calls
            inference_calls += 1
            return predict_temporal_memory_scores_windowed(
                model=model,
                video=video,
                device=device,
                context_bins=protocol["inference"]["context_bins"],
                width=protocol["inference"]["resolution"][0],
                height=protocol["inference"]["resolution"][1],
                inference_batch_size=protocol["inference"]["inference_batch_size"],
                log_count_clip=protocol["inference"]["log_count_clip"],
                window_length=H2_WINDOW_LENGTH,
                stride=H2_STRIDE,
            )

        scores, preserved = choose_candidate_scores(
            decision, baseline_record.scores, infer_h2
        )
        scores = scores.detach().cpu().to(torch.float32).reshape(-1).contiguous()
        if scores.numel() != decision.event_count or not torch.isfinite(scores).all():
            raise RuntimeError("Candidate scores are malformed for {}.".format(file_name))
        if bool((scores < 0).any()) or bool((scores > 1).any()):
            raise RuntimeError("Candidate scores are not probabilities for {}.".format(file_name))
        if preserved:
            preserved = scores.data_ptr() == baseline_record.scores.data_ptr() and torch.equal(
                scores, baseline_record.scores
            )
            all_non_h2_preserved = all_non_h2_preserved and preserved
        else:
            h2_records.append(
                {
                    "file_name": file_name,
                    "event_count": decision.event_count,
                    "scores": scores,
                    "source_sha256": baseline_record.source_sha256,
                    "source_file_sha256": file_hash_by_name[file_name],
                    "route": decision.to_metadata(),
                }
            )

        candidate_record = replay.RoutedRecord(
            file_name=baseline_record.file_name,
            event_count=baseline_record.event_count,
            scores=scores,
            seg_label=baseline_record.seg_label,
            locs=baseline_record.locs,
            idx_label=baseline_record.idx_label,
            source_sha256=baseline_record.source_sha256,
            score_source=_candidate_score_source(decision),
        )
        threshold = decision.prediction_threshold
        expected_threshold = LOW_THRESHOLD if decision.domain == "low" else M20_THRESHOLD
        if threshold != expected_threshold:
            raise RuntimeError("Route threshold differs for {}.".format(file_name))
        base_count = replay.evaluate_cached_video(baseline_record, threshold, cfg)
        cand_count = replay.evaluate_cached_video(candidate_record, threshold, cfg)
        baseline_counts.append(base_count)
        candidate_counts.append(cand_count)
        per_video.append(
            {
                "index": index,
                "file_name": file_name,
                "event_count": decision.event_count,
                "polarity_minority_fraction": decision.polarity_minority_fraction,
                "route": decision.to_metadata(),
                "baseline_score_source": baseline_record.score_source,
                "candidate_score_source": candidate_record.score_source,
                "threshold": threshold,
                "reinferred": decision.domain == "h2",
                "scores_bitwise_preserved": preserved,
                "baseline_counts": asdict(base_count),
                "candidate_counts": asdict(cand_count),
            }
        )
        print("route/evaluate {}/24: {} -> {}".format(index, file_name, decision.domain), flush=True)

    h2_count = sum(item["route"]["domain"] == "h2" for item in per_video)
    if inference_calls != h2_count or len(h2_records) != h2_count:
        raise RuntimeError("Inference call count differs from H2 route count.")
    h2_payload = {
        "schema": H2_CACHE_SCHEMA,
        "created_utc": _utc_now(),
        "execution_protocol_sha256": sha256_file(paths["execution_protocol"]),
        "checkpoint_path": str(input_paths["m20_checkpoint"]),
        "checkpoint_sha256": M20_CHECKPOINT_SHA256,
        "route_policy_sha256": protocol["route_policy"]["sha256"],
        "window_length": H2_WINDOW_LENGTH,
        "stride": H2_STRIDE,
        "record_count": h2_count,
        "records": h2_records,
    }
    replay._atomic_torch_save(h2_payload, paths["h2_cache"], overwrite=False)
    h2_cache_sha = sha256_file(paths["h2_cache"])

    baseline_total = replay._sum_counts(baseline_counts)
    candidate_total = replay._sum_counts(candidate_counts)
    baseline_count_dict = asdict(baseline_total)
    candidate_count_dict = asdict(candidate_total)
    baseline_metrics = replay.metrics_from_counts_exact(baseline_total, cfg).to_dict()
    candidate_metrics = replay.metrics_from_counts_exact(candidate_total, cfg).to_dict()
    gates = promotion_gate_results(
        baseline_count_dict,
        baseline_metrics,
        candidate_metrics,
        all_non_h2_preserved,
        inference_calls,
        h2_count,
    )

    validation_after = _validate_validation_files_after_claim(protocol, input_paths)
    if validation_after != validation_before:
        raise RuntimeError("Validation files changed during the one-shot replay.")
    cache_hashes_after = {
        name: sha256_file(input_paths[name])
        for name in ("m10_golden_cache", "m20_golden_cache")
    }
    if cache_hashes_after != cache_hashes_before:
        raise RuntimeError("Golden score caches changed during the one-shot replay.")
    return {
        "validation_integrity": {
            "before": validation_before,
            "after": validation_after,
            "equal": True,
        },
        "golden_cache_sha256": cache_hashes_before,
        "golden_cache_binding": binding,
        "h2_cache": {
            "path": str(paths["h2_cache"]),
            "sha256": h2_cache_sha,
            "record_count": h2_count,
        },
        "route_summary": {
            "counts": {
                domain: sum(item["route"]["domain"] == domain for item in per_video)
                for domain in ("low", "middle", "h1", "h2")
            },
            "h2_inference_calls": inference_calls,
            "non_h2_scores_bitwise_preserved": all_non_h2_preserved,
        },
        "per_video": per_video,
        "aggregate": {
            "baseline": {"counts": baseline_count_dict, "metrics": baseline_metrics},
            "candidate": {"counts": candidate_count_dict, "metrics": candidate_metrics},
            "delta": _delta(candidate_metrics, baseline_metrics),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _artifact_observation(paths):
    result = {}
    for name in ("runtime_receipt", "claim", "h2_cache", "report"):
        path = paths[name]
        item = {"path": str(path), "exists": path.is_file()}
        if item["exists"]:
            item.update({"size": path.stat().st_size, "sha256": sha256_file(path)})
        result[name] = item
    return result


def _failure_report(
    protocol_sha,
    runtime_receipt,
    runtime_receipt_sha,
    claim_payload,
    claim_sha,
    stage,
    error,
    paths,
):
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": _utc_now(),
        "status": "failed",
        "passed": False,
        "evidence_class": "single_claimed_frozen_24_validation_route_replay",
        "execution_protocol": {
            "path": str(paths["execution_protocol"]),
            "sha256": protocol_sha,
        },
        "runtime_preflight_receipt": {
            "path": str(paths["runtime_receipt"]),
            "sha256": runtime_receipt_sha,
            "payload": runtime_receipt,
        },
        "attempt_claim": {"payload": claim_payload, "sha256": claim_sha},
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        },
        "artifacts": _artifact_observation(paths),
        "failure_action": "archive_without_validation_retuning_or_second_attempt",
        "submission_zip_created": False,
        "platform_upload_performed": False,
    }


def run_execution(expected_protocol_sha256):
    state = _preclaim_validate(expected_protocol_sha256)
    protocol, paths, protocol_sha, git_before, code_before, input_paths = state[:6]
    verified_non_val_before, train_prerequisite = state[6], state[7]
    runtime_receipt, runtime_receipt_sha = _load_runtime_receipt(
        protocol_sha, git_before, code_before, paths
    )
    runtime, smoke, runtime_bundle = _prepare_runtime_before_claim(protocol, input_paths)
    if runtime != runtime_receipt["runtime"] or smoke != runtime_receipt["smoke"]:
        raise RuntimeError("Live runtime smoke differs from the immutable preflight receipt.")
    if sha256_file(paths["runtime_receipt"]) != runtime_receipt_sha:
        raise RuntimeError("Runtime preflight receipt changed before the claim.")
    claim_payload, claim_sha = _atomic_claim(
        paths["claim"],
        protocol_sha,
        paths["report"],
        runtime_receipt_sha,
    )
    stage = "after_claim_before_val_or_cache_read"
    try:
        stage = "claimed_h2_only_inference_and_fixed_c00_evaluation"
        outcome = _run_claimed(protocol, paths, input_paths, runtime_bundle)
        stage = "postrun_immutability_check"
        git_after = _git_state()
        code_after = _code_sha256()
        if git_after != git_before or code_after != code_before:
            raise RuntimeError("Git or code changed during the one-shot replay.")
        if sha256_file(paths["execution_protocol"]) != protocol_sha:
            raise RuntimeError("Execution protocol changed during replay.")
        if sha256_file(paths["claim"]) != claim_sha:
            raise RuntimeError("Attempt claim changed during replay.")
        if sha256_file(paths["runtime_receipt"]) != runtime_receipt_sha:
            raise RuntimeError("Runtime preflight receipt changed during replay.")
        verified_non_val_after = {
            name: sha256_file(input_paths[name])
            for name in INPUT_NAMES
            if name not in DEFERRED_VAL_INPUTS
        }
        if verified_non_val_after != verified_non_val_before:
            raise RuntimeError("A non-validation prerequisite changed during replay.")
        report = {
            "schema": REPORT_SCHEMA,
            "created_utc": _utc_now(),
            "status": "completed",
            "passed": outcome["passed"],
            "evidence_class": "single_claimed_frozen_24_validation_route_replay",
            "execution_protocol": {
                "path": str(paths["execution_protocol"]),
                "sha256": protocol_sha,
            },
            "runtime_preflight_receipt": {
                "path": str(paths["runtime_receipt"]),
                "sha256": runtime_receipt_sha,
                "payload": runtime_receipt,
            },
            "attempt_claim": {"payload": claim_payload, "sha256": claim_sha},
            "repository": {
                "before": git_before,
                "after": git_after,
                "code_sha256": code_before,
            },
            "inputs": protocol["inputs"],
            "train_prerequisite": train_prerequisite,
            "route_policy": protocol["route_policy"],
            "persistence": protocol["persistence"],
            "validation_integrity": outcome["validation_integrity"],
            "golden_cache_sha256": outcome["golden_cache_sha256"],
            "golden_cache_binding": outcome["golden_cache_binding"],
            "h2_cache": outcome["h2_cache"],
            "route_summary": outcome["route_summary"],
            "per_video": outcome["per_video"],
            "aggregate": outcome["aggregate"],
            "gates": outcome["gates"],
            "failure_action": (
                None
                if outcome["passed"]
                else "archive_without_validation_retuning_or_second_attempt"
            ),
            "submission_zip_created": False,
            "platform_upload_performed": False,
        }
    except BaseException as error:
        report = _failure_report(
            protocol_sha,
            runtime_receipt,
            runtime_receipt_sha,
            claim_payload,
            claim_sha,
            stage,
            error,
            paths,
        )
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.command == "freeze":
        result = freeze_execution_protocol()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = preflight_execution(args.expected_execution_protocol_sha256)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "runtime-preflight":
        result = runtime_preflight_execution(
            args.expected_execution_protocol_sha256
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    report = run_execution(args.expected_execution_protocol_sha256)
    print("report:", _canonical_paths()["report"])
    print("promotion gates passed:", report["passed"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

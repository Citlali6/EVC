"""One-shot, protocol-bound validation of the audited T32 E3 checkpoint.

``preflight`` hashes and validates only immutable audit/config/manifest files.
It neither imports torch/replay nor opens validation NPZ/cache payloads.
``run`` accepts only the canonical protocol and its externally supplied hash,
durably consumes the sole attempt, then creates exactly one M10/T16 and one
E3/T32 cache and evaluates the frozen C00 and C09 profiles.  It never creates a
submission archive and never uploads anything.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping, Optional, Sequence


PROTOCOL_SCHEMA = "ev-uav-t32-e3-validation-execution-protocol-v1"
CLAIM_SCHEMA = "ev-uav-t32-e3-validation-attempt-claim-v1"
REPORT_SCHEMA = "ev-uav-t32-e3-frozen-validation-report-v1"
FORMAL_PROTOCOL_SCHEMA = "ev-uav-t32-attention-projection-preregistered-protocol-v1"
FORMAL_AUDIT_SCHEMA = "ev-uav-t32-formal-training-audit-v1"

OFFICIAL_VIDEO_COUNT = 24
OFFICIAL_EVENT_COUNT = 1424330
OFFICIAL_STEMS = tuple("val_{:03d}".format(index) for index in range(24))
ROUTE_CUTOFF = 30000
LOW_THRESHOLD = 0.718
HIGH_THRESHOLD = 0.719

REQUIRED_REPLAY_ANCESTOR = "b5775d91e51817bda6f94d382f1da33bd64a142f"
FORMAL_TRAINING_COMMIT = "4b3e3eb48371d418e150f034c36b6949848e4b0d"
M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
M10_SHA256 = "5c89c89a165469c0a4e8286d4644d60d2f82cf5775edbb724f626e24e67d8935"
E1_SHA256 = "907cf5778a942d3cdbdebf3d07ab22eec2e2fe125552aa2d7a1d3d0a43bf39b2"
E2_SHA256 = "0781af6c822a50dfc2896375ffbf7d6bced1de40adc6eaaa526d6cf83bb4cfbc"
E3_SHA256 = "943f3179fc873c0ade96d4b73e779ac79899c64927a783a68e447709454e4c8d"
OFFICIAL_MANIFEST_SHA256 = "c7c574b5dfa8336fe50917581544b5e4991b2cde197f31c9a5bee05a29e336d4"
OFFICIAL_VAL_SEMANTIC_SHA256 = "d780da17e69446b988b1b5fae7954855d5ce66a32aa7b9581eeb3e4a0563f83f"
OFFICIAL_CACHE_DATASET_SIGNATURE = "bedba93c1d523f58c35da6399219df1b98e6240f92d093520fa0f4961d927274"
SOURCE_CONFIG_SHA256 = "c4157f6e04fb96be1fe9bef6ed87004b1e7da0d72507a43091e9f929345f2ec9"
FORMAL_PROTOCOL_SHA256 = "9acc8cf3fc1cf036ec6be1d5dc8d30370ac846fafd7609b537f38972ac3797a5"
FORMAL_AUDIT_SHA256 = "5cf205efe52c61e767be8283abde79c29c2dcd9675fea23843a08122b3d45a9a"
FORMAL_RUNTIME_SHA256 = "b2ccadd7660e472caf258f3bafca792456c9a6c399d35ecc8fd3a14182f5ac4b"
FORMAL_RUNNER_SHA256 = "7dfcf94a83a9f6431cd3b62ab66f0a61a059bca8c305f350639b0530e582bbb1"
RUN_SUMMARY_SHA256 = "357e0a7d96afef657634fbb6694ba6af226c4a8e061734015d50d2061a70550e"
TRAINING_CONFIG_SHA256 = "b41dadb648b913645be12b7931e4934567477bf714ec0663b38af8913ff7296f"
GOLDEN_REPORT_SHA256 = "da6004ddd22731b8e848c9ed0c561961abbc04b4e3f66cd07b1e085d26f9f383"
C09_REPORT_SHA256 = "5c299b9f69e42e8781a06ce113ce0eb1d4b879517a8295288c497878a3093a14"

GOLDEN_C00 = {
    "iou": 0.9422550201416016,
    "acc": 0.9767196774482727,
    "pd": 0.9762704745905082,
    "fa": 4.69291729752432e-06,
    "score_fa": 0.9541549751552311,
    "score": 0.9628776541559201,
}
C09_ACTUAL = {
    "iou": 0.942417323589325,
    "acc": 0.9766433835029602,
    "pd": 0.9762704745905082,
    "fa": 4.672178395325665e-06,
    "score_fa": 0.9543528769429719,
    "score": 0.9629618559872559,
}
GATE_LIMITS = {
    "pd": 0.9762704745905082,
    "iou": 0.9423673236,
    "fa": 4.6821783953e-06,
}
GOLDEN_COUNTS = {
    "evaluator_detected_objects": 4649,
    "evaluator_false_components": 1584,
    "evaluator_frames": 3752,
    "evaluator_objects": 4762,
    "event_false_negatives": 1525,
    "event_false_positives": 2396,
    "event_true_positives": 63981,
    "events": 1424330,
    "ground_truth_positive_events": 65506,
    "predicted_positive_events": 66377,
    "videos": 24,
}
C09_COUNTS = {
    "evaluator_detected_objects": 4649,
    "evaluator_false_components": 1577,
    "evaluator_frames": 3752,
    "evaluator_objects": 4762,
    "event_false_negatives": 1530,
    "event_false_positives": 2379,
    "event_true_positives": 63976,
    "events": 1424330,
    "ground_truth_positive_events": 65506,
    "predicted_positive_events": 66355,
    "videos": 24,
}
PROMOTION_GATES = {
    "minimum_score_delta_over_c09": 0.00001,
    "minimum_pd": GATE_LIMITS["pd"],
    "minimum_iou": GATE_LIMITS["iou"],
    "maximum_fa": GATE_LIMITS["fa"],
    "all_gates_required": True,
    "failure_action": "archive_without_e1_e2_or_validation_retuning",
}

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
FROZEN_EXPERIMENT_DIRECTORY = (
    WORKSPACE_ROOT / "experiments" / "20260810_t32_attention_projection_seed49"
).resolve()
FORMAL_RUN_RELATIVE = Path("formal_run")
RUN_RELATIVE = Path("model/runs/20260810-133043_seed49_pid48684")

CODE_PATHS = (
    "evaluate_t32_e3_validation.py",
    "replay_temporal_memory_validation.py",
    "dataset/basedataset.py",
    "dataset/event_features.py",
    "dataset/event_frame.py",
    "dataset/ev_uav.py",
    "dataset/sampling.py",
    "dataset/temporal_chunks.py",
    "dataset/temporal_frame.py",
    "model/modules/confidence_head.py",
    "model/temporal_frame_net.py",
    "model/temporal_memory_net.py",
    "utils/challenge_eval.py",
    "utils/component_reranker.py",
    "utils/density_threshold.py",
    "utils/eval.py",
    "utils/inference_chunks.py",
    "utils/postprocess.py",
    "utils/spatial_tta.py",
    "utils/temporal_frame_inference.py",
    "utils/temporal_memory_inference.py",
)


def _canonical_paths() -> dict:
    experiment = Path(FROZEN_EXPERIMENT_DIRECTORY).resolve()
    formal = experiment / FORMAL_RUN_RELATIVE
    run = formal / RUN_RELATIVE
    validation = experiment / "validation"
    return {
        "formal_protocol": formal / "preregistered_protocol.json",
        "formal_audit": formal / "formal_training_audit.json",
        "formal_runtime": formal / "formal_runtime_result.json",
        "formal_runner": formal / "run_formal.py",
        "run_summary": run / "run_summary.json",
        "training_config": run / "config.yaml",
        "e1_checkpoint": run / "epoch_001_seed49.pt",
        "e2_checkpoint": run / "epoch_002_seed49.pt",
        "e3_checkpoint": run / "epoch_003_seed49.pt",
        "m20_parent": PROJECT_ROOT / "checkpoints/m20_attn_dense_views8_epoch_003_seed48.pt",
        "m10_checkpoint": PROJECT_ROOT / "checkpoints/m10_dense_views2_epoch_002_seed42.pt",
        "source_config": PROJECT_ROOT / "configs/evisseg_evuav.yaml",
        "dataset_manifest": WORKSPACE_ROOT / "datasets/EV-UAV-Challenge2/official_google_drive_manifest.json",
        "golden_report": WORKSPACE_ROOT / "results/submission_m20_golden/offline_score_report.json",
        "c09_report": WORKSPACE_ROOT / "results/exploratory_p0c_density_097/offline_score_report.json",
        "protocol": validation / "preregistered_execution_protocol.json",
        "claim": validation / "validation_attempt_claim.json",
        "m10_cache": validation / "raw_m10_t16_val24.pt",
        "e3_cache": validation / "raw_e3_t32_val24.pt",
        "report": validation / "frozen_validation_report.json",
    }


def _expected_input_sha256() -> dict:
    return {
        "formal_protocol": FORMAL_PROTOCOL_SHA256,
        "formal_audit": FORMAL_AUDIT_SHA256,
        "formal_runtime": FORMAL_RUNTIME_SHA256,
        "formal_runner": FORMAL_RUNNER_SHA256,
        "run_summary": RUN_SUMMARY_SHA256,
        "training_config": TRAINING_CONFIG_SHA256,
        "e1_checkpoint": E1_SHA256,
        "e2_checkpoint": E2_SHA256,
        "e3_checkpoint": E3_SHA256,
        "m20_parent": M20_SHA256,
        "m10_checkpoint": M10_SHA256,
        "source_config": SOURCE_CONFIG_SHA256,
        "dataset_manifest": OFFICIAL_MANIFEST_SHA256,
        "golden_report": GOLDEN_REPORT_SHA256,
        "c09_report": C09_REPORT_SHA256,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("{} must be a lowercase SHA-256.".format(name))
    return value


def _git_state() -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=str(PROJECT_ROOT), capture_output=True, check=True
        )
        return result.stdout

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_REPLAY_ANCESTOR, "HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
    ).returncode == 0
    return {
        "head": run("rev-parse", "HEAD").decode("ascii").strip().lower(),
        "clean": not bool(status.strip()),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "required_replay_ancestor_present": ancestor,
    }


def _code_sha256() -> dict:
    result = {}
    for relative in CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError("Required code file is missing: {}".format(path))
        result[relative] = sha256_file(path)
    return result


def _load_json(path: Path, expected_sha256: str, name: str):
    before = sha256_file(path)
    if before != _require_sha256(expected_sha256, name + " SHA-256"):
        raise ValueError("{} SHA-256 differs from frozen value.".format(name))
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while being read.".format(name))
    return payload


def _extract_json_member(text: str, member: str):
    """Decode one JSON member even when unrelated legacy paths are malformed."""
    marker = json.dumps(member) + ":"
    start = text.find(marker)
    if start < 0:
        raise ValueError("Offline report lacks {}.".format(member))
    value_start = start + len(marker)
    value_start += len(text[value_start:]) - len(text[value_start:].lstrip())
    value, _ = json.JSONDecoder().raw_decode(text, value_start)
    return value


def _validate_offline_report(path: Path, expected_sha256: str, metrics, counts, name: str) -> None:
    before = sha256_file(path)
    if before != expected_sha256:
        raise ValueError("{} SHA-256 differs from the frozen value.".format(name))
    text = Path(path).read_text(encoding="utf-8")
    if _extract_json_member(text, "metrics") != metrics:
        raise ValueError("{} metrics differ from the frozen baseline.".format(name))
    if _extract_json_member(text, "counts") != counts:
        raise ValueError("{} sufficient counts differ from the frozen baseline.".format(name))
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while being read.".format(name))


def _manifest_val_entries(manifest_path: Path, expected_manifest_sha256: str) -> list:
    """Parse only the frozen manifest; this never opens an NPZ."""
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Official Drive manifest SHA-256 differs.")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Official Drive manifest schema is invalid.")
    selected = {}
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping):
            raise ValueError("Official Drive manifest contains a malformed entry.")
        relative = str(entry.get("path", "")).replace("\\", "/")
        if relative.startswith("val/"):
            name = Path(relative).name
            if relative != "val/" + name or name in selected:
                raise ValueError("Official Drive manifest val paths are not canonical and unique.")
            selected[name] = entry
    expected_names = tuple(stem + ".npz" for stem in OFFICIAL_STEMS)
    if tuple(sorted(selected)) != expected_names:
        raise ValueError("Official Drive manifest does not contain exactly val_000..val_023.")
    result = []
    for name in expected_names:
        entry = selected[name]
        result.append({
            "path": "val/" + name,
            "size": int(entry.get("size", -1)),
            "sha256": _require_sha256(entry.get("sha256"), "manifest SHA-256 for " + name),
        })
    return result


def _validate_validation_files(paths: Mapping[str, Path]) -> dict:
    """Hash the exact official val NPZ population; call only after the claim."""
    manifest_path = paths["dataset_manifest"]
    entries = _manifest_val_entries(manifest_path, OFFICIAL_MANIFEST_SHA256)
    val_root = (manifest_path.parent / "val").resolve()
    expected_names = tuple(stem + ".npz" for stem in OFFICIAL_STEMS)
    disk_names = tuple(sorted(path.name for path in val_root.glob("*.npz") if path.is_file()))
    if disk_names != expected_names:
        raise ValueError("Validation directory does not contain exactly val_000..val_023.")
    digest = hashlib.sha256()
    evidence = []
    for entry in entries:
        name = Path(entry["path"]).name
        expected_sha = entry["sha256"]
        expected_size = entry["size"]
        path = val_root / name
        if path.resolve().parent != val_root or not path.is_file():
            raise ValueError("Validation path is not canonical: {}".format(path))
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError("Validation file size differs from manifest: {}".format(name))
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError("Validation file SHA-256 differs from manifest: {}".format(name))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(actual_sha))
        evidence.append({"name": name, "size": actual_size, "sha256": actual_sha})
    semantic_sha = digest.hexdigest()
    if semantic_sha != OFFICIAL_VAL_SEMANTIC_SHA256:
        raise ValueError("Validation semantic SHA-256 is not the official population.")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": OFFICIAL_MANIFEST_SHA256,
        "semantic_sha256_scheme": "sha256(name_utf8 + NUL + member_sha256_bytes), canonical order",
        "semantic_sha256": semantic_sha,
        "video_count": len(evidence),
        "files": evidence,
    }


def _require_canonical(path, expected: Path, name: str) -> Path:
    supplied = Path(str(path))
    if not supplied.is_absolute() or supplied.resolve() != expected.resolve():
        raise ValueError("{} is not the canonical path {}.".format(name, expected))
    if str(supplied) != str(supplied.resolve()):
        raise ValueError("{} is not stored in resolved form.".format(name))
    return supplied.resolve()


def _require_distinct(named_paths) -> None:
    items = [(name, Path(path).resolve()) for name, path in named_paths]
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            aliases = os.path.normcase(str(left)) == os.path.normcase(str(right))
            if not aliases and left.exists() and right.exists():
                try:
                    aliases = left.samefile(right)
                except OSError:
                    pass
            if aliases:
                raise ValueError("{} aliases {} at {}.".format(left_name, right_name, left))


def _atomic_no_clobber_json(path: Path, payload: Mapping) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Output already exists: {}".format(path))
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
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


def _atomic_claim(path: Path, protocol_path: Path, protocol_sha256: str, report_path: Path):
    payload = {
        "schema": CLAIM_SCHEMA,
        "claimed_utc": _utc_now(),
        "attempt": 1,
        "attempt_budget": 1,
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": protocol_sha256,
        "report_path": str(Path(report_path).resolve()),
        "state": "irreversibly_claimed_before_any_validation_or_cache_load",
    }
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(str(Path(path).resolve()), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return payload, sha256_file(path)


def _validate_formal_lineage(paths: Mapping[str, Path], inputs: Mapping) -> None:
    formal = _load_json(paths["formal_protocol"], inputs["formal_protocol"]["sha256"], "formal protocol")
    audit = _load_json(paths["formal_audit"], inputs["formal_audit"]["sha256"], "formal audit")
    summary = _load_json(paths["run_summary"], inputs["run_summary"]["sha256"], "run summary")
    if formal.get("schema") != FORMAL_PROTOCOL_SCHEMA or formal.get("created_before_formal_training") is not True:
        raise ValueError("Formal preregistration is invalid.")
    if formal.get("git_commit") != FORMAL_TRAINING_COMMIT:
        raise ValueError("Formal preregistration git lineage is invalid.")
    parent = formal.get("parent_checkpoint", {})
    if parent.get("sha256") != M20_SHA256 or parent.get("sequence_length") != 16:
        raise ValueError("Formal preregistration parent is not released M20/T16.")
    dataset = formal.get("dataset", {})
    if dataset.get("official_drive_manifest_sha256") != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("Formal preregistration dataset manifest differs.")
    if dataset.get("source_video_count") != 99 or dataset.get("selected_video_count") != 54:
        raise ValueError("Formal preregistration training population differs.")
    training = formal.get("training", {})
    if training.get("selection_checkpoint") != "epoch_003_seed49.pt" or training.get("e1_e2_must_not_be_used_for_model_selection") is not True:
        raise ValueError("Formal protocol does not select E3 exclusively.")
    if training.get("source_sequence_length") != 16 or training.get("target_sequence_length") != 32:
        raise ValueError("Formal protocol is not the frozen T16-to-T32 experiment.")
    plan = formal.get("frozen_validation_plan", {})
    expected_plan = {
        "checkpoint": "E3 only",
        "raw_model_inference_runs": 1,
        "low_density_route": "released M10 for event_count <= 30000",
        "low_threshold": LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
        "no_threshold_or_hyperparameter_search": True,
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise ValueError("Formal validation plan differs at {}.".format(key))
    comparison = plan.get("primary_comparison", {})
    expected_comparison = {
        "baseline_score": C09_ACTUAL["score"],
        "minimum_score_delta": PROMOTION_GATES["minimum_score_delta_over_c09"],
        "minimum_pd": GATE_LIMITS["pd"],
        "minimum_iou": GATE_LIMITS["iou"],
        "maximum_fa": GATE_LIMITS["fa"],
    }
    for key, expected in expected_comparison.items():
        if comparison.get(key) != expected:
            raise ValueError("Formal promotion gate differs at {}.".format(key))
    if formal.get("test_labels_must_not_be_read") is not True or formal.get("platform_submission_authorized") is not False:
        raise ValueError("Formal data/upload restrictions are invalid.")
    if audit.get("schema") != FORMAL_AUDIT_SCHEMA or audit.get("status") != "passed" or audit.get("independent_review") != "passed":
        raise ValueError("Formal training audit did not pass.")
    if audit.get("p1_findings") != 0 or audit.get("p2_findings") != 0:
        raise ValueError("Formal training audit contains blocking findings.")
    audit_protocol = audit.get("protocol", {})
    if audit_protocol.get("sha256") != inputs["formal_protocol"]["sha256"] or audit_protocol.get("git_commit") != FORMAL_TRAINING_COMMIT:
        raise ValueError("Formal audit protocol/git binding differs.")
    artifacts = audit.get("artifacts", {})
    expected_artifacts = {
        "config_sha256": inputs["training_config"]["sha256"],
        "run_summary_sha256": inputs["run_summary"]["sha256"],
        "epoch_001_sha256": inputs["e1_checkpoint"]["sha256"],
        "epoch_002_sha256": inputs["e2_checkpoint"]["sha256"],
        "epoch_003_sha256": inputs["e3_checkpoint"]["sha256"],
    }
    for key, expected in expected_artifacts.items():
        if artifacts.get(key) != expected:
            raise ValueError("Formal audit artifact differs at {}.".format(key))
    selection = audit.get("selection", {})
    if selection.get("only_allowed_checkpoint") != "epoch_003_seed49.pt" or selection.get("only_allowed_checkpoint_sha256") != E3_SHA256:
        raise ValueError("Formal audit does not exclusively authorize E3.")
    for key in ("epoch_001_must_not_be_evaluated_or_selected", "epoch_002_must_not_be_evaluated_or_selected", "best_loss_must_not_be_evaluated_or_selected"):
        if selection.get(key) is not True:
            raise ValueError("Formal audit fails the E1/E2 rejection contract.")
    access = audit.get("data_access", {})
    if access != {"validation_accessed_by_audit": False, "test_accessed_by_audit": False, "platform_upload_performed": False}:
        raise ValueError("Formal audit data-access record is invalid.")
    if audit.get("runtime", {}).get("result_sha256") != inputs["formal_runtime"]["sha256"]:
        raise ValueError("Formal runtime result is not audit-bound.")
    if audit.get("protocol", {}).get("runner_sha256") != inputs["formal_runner"]["sha256"]:
        raise ValueError("Formal runner is not audit-bound.")
    truth = audit.get("training_truth", {})
    if truth.get("changed_state_keys") != [
        "temporal_attn.output_projection.bias",
        "temporal_attn.output_projection.weight",
    ] or truth.get("changed_element_total") != 9312 or truth.get("frozen_state_mismatch_count") != 0:
        raise ValueError("Formal audit training-state evidence differs.")
    if summary.get("seed") != 49 or summary.get("training_scope") != "temporal_attention_projection_only":
        raise ValueError("Run summary is not the frozen seed49 projection-only run.")
    if summary.get("initialized_from_sha256") != M20_SHA256 or summary.get("git", {}) != {"commit": FORMAL_TRAINING_COMMIT, "dirty": False}:
        raise ValueError("Run summary parent/git lineage is invalid.")
    migrations = summary.get("initialization_migrations")
    if not isinstance(migrations, list) or len(migrations) != 1:
        raise ValueError("Run summary must contain the sole T16-to-T32 migration.")
    migration = migrations[0]
    for key, expected in {
        "name": "temporal_memory_sequence_length_t16_to_t32_strict_state",
        "source_sequence_length": 16,
        "target_sequence_length": 32,
        "metadata_difference_allowlist": ["sequence_length"],
        "state_dict_strict": True,
        "parent_checkpoint_sha256": M20_SHA256,
    }.items():
        if migration.get(key) != expected:
            raise ValueError("Run summary migration differs at {}.".format(key))
    _validate_offline_report(
        paths["golden_report"], inputs["golden_report"]["sha256"],
        GOLDEN_C00, GOLDEN_COUNTS, "golden C00 report",
    )
    _validate_offline_report(
        paths["c09_report"], inputs["c09_report"]["sha256"],
        C09_ACTUAL, C09_COUNTS, "C09 report",
    )


def _runtime_contract(paths: Mapping[str, Path]) -> dict:
    dataset_root = paths["dataset_manifest"].parent.as_posix()
    shared_cache = [
        "DATA.root={}".format(json.dumps(dataset_root)),
        "TEMPORAL_FRAME.temporal_frame_enabled=false",
        "TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0",
        "TEMPORAL_MEMORY.temporal_memory_bin_size=50",
        "TEMPORAL_MEMORY.temporal_memory_context_bins=5",
        "TEMPORAL_MEMORY.temporal_memory_width=16",
        "TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8",
        "TEMPORAL_MEMORY.temporal_memory_log_count_clip=4.0",
    ]
    m10 = shared_cache + [
        "TEMPORAL_MEMORY.temporal_memory_sequence_length=16",
        "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=false",
        "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true",
    ]
    e3 = shared_cache + [
        "TEMPORAL_MEMORY.temporal_memory_sequence_length=32",
        "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_attention_projection_only_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_init_sequence_length_warm_start_enabled=true",
        "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false",
    ]
    c00 = e3 + [
        "TEST.prediction_threshold=0.719",
        "TEST.roc=true",
        "TEST.pd_detT=50",
        "TEST.correct_thresh=0.0001",
        "POSTPROCESS.p0_enabled=true",
        "POSTPROCESS.p0_spatial_radius=2",
        "POSTPROCESS.p0_temporal_bin_size=50",
        "POSTPROCESS.p0_temporal_radius_bins=1",
        "POSTPROCESS.p0_min_cluster_events=3",
        "POSTPROCESS.p0_min_duration_bins=5",
        "POSTPROCESS.p0c_high_confidence_recovery_enabled=true",
        "POSTPROCESS.p0c_retain_min_score=0.95",
        "POSTPROCESS.p0c_density_retain_enabled=false",
        "POSTPROCESS.p0c_density_event_count_cutoff=100000",
        "POSTPROCESS.p0c_density_retain_min_score=0.97",
        "POSTPROCESS.component_reranker_enabled=false",
        "POSTPROCESS.p0b_enabled=false",
        "POSTPROCESS.p18_score_track_recovery_enabled=true",
        "POSTPROCESS.p18_event_count_cutoff=1",
        "POSTPROCESS.p18_max_event_count=35000",
        "POSTPROCESS.p18_candidate_floor=0.53",
        "POSTPROCESS.p18_spatial_radius=5",
        "POSTPROCESS.p18_temporal_bin_size=50",
        "POSTPROCESS.p18_max_link_distance=8.0",
        "POSTPROCESS.p18_max_gap_bins=1",
        "POSTPROCESS.p18_min_track_bins=4",
        "POSTPROCESS.p18_restore_mode=best",
        "POSTPROCESS.p18_max_restore_events_per_component=0",
        "POSTPROCESS.p6_density_threshold_enabled=true",
        "POSTPROCESS.p6_event_count_cutoff=30000",
        "POSTPROCESS.p6_low_density_threshold=0.718",
        "POSTPROCESS.p6_high_density_threshold=0.719",
    ]
    c09 = [item if item != "POSTPROCESS.p0c_density_retain_enabled=false" else
           "POSTPROCESS.p0c_density_retain_enabled=true" for item in c00]
    return {
        "device": "cuda:0",
        "cache_generation_order": ["m10_t16", "e3_t32"],
        "m10_cache_overrides": m10,
        "e3_cache_overrides": e3,
        "route": {"secondary": "released_m10_t16", "secondary_max_events_inclusive": 30000,
                  "primary": "audited_e3_t32", "primary_min_events_exclusive": 30000,
                  "metadata_difference_allowlist": ["temporal_memory_sequence_length"]},
        "thresholds": {"low": LOW_THRESHOLD, "high": HIGH_THRESHOLD, "density_cutoff": ROUTE_CUTOFF},
        "postprocess_profiles": {"C00": c00, "C09": c09},
        "no_grid_search": True,
        "raw_inference_runs_per_checkpoint": 1,
        "zip_creation": False,
        "platform_upload": False,
    }


def build_execution_protocol() -> dict:
    paths, expected = _canonical_paths(), _expected_input_sha256()
    git = _git_state()
    if not git["clean"] or not git["required_replay_ancestor_present"]:
        raise RuntimeError("Preflight requires a clean reviewed replay-compatible HEAD.")
    inputs = {}
    for name, expected_sha in expected.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError("Frozen input is missing: {}".format(path))
        digest = sha256_file(path)
        if digest != expected_sha:
            raise ValueError("Frozen input {} has unexpected SHA-256.".format(name))
        inputs[name] = {"path": str(path.resolve()), "sha256": digest}
    _require_distinct([(name, value["path"]) for name, value in inputs.items()])
    _validate_formal_lineage(paths, inputs)
    manifest_entries = _manifest_val_entries(
        paths["dataset_manifest"], inputs["dataset_manifest"]["sha256"]
    )
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "created_utc": _utc_now(),
        "attempt_budget": 1,
        "repository": {"project_root": str(PROJECT_ROOT), "expected_clean_git_head": git["head"],
                       "required_replay_ancestor": REQUIRED_REPLAY_ANCESTOR, "code_sha256": _code_sha256()},
        "inputs": inputs,
        "validation_dataset": {"split": "val", "manifest_sha256": OFFICIAL_MANIFEST_SHA256,
                               "dataset_signature": OFFICIAL_CACHE_DATASET_SIGNATURE,
                               "video_count": OFFICIAL_VIDEO_COUNT, "event_count": OFFICIAL_EVENT_COUNT,
                               "canonical_stems": list(OFFICIAL_STEMS),
                               "manifest_files": manifest_entries},
        "runtime": _runtime_contract(paths),
        "baselines": {"C00_released_m20": GOLDEN_C00, "C09_released_m20": C09_ACTUAL},
        "promotion_gates": PROMOTION_GATES,
        "outputs": {name: str(paths[name].resolve()) for name in ("claim", "m10_cache", "e3_cache", "report")},
    }
    validate_execution_protocol(protocol)
    return protocol


def validate_execution_protocol(protocol: Mapping) -> None:
    expected_top = {"schema", "created_utc", "attempt_budget", "repository", "inputs",
                    "validation_dataset", "runtime", "baselines", "promotion_gates", "outputs"}
    if not isinstance(protocol, Mapping) or set(protocol) != expected_top or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported or malformed T32 validation protocol.")
    if protocol.get("attempt_budget") != 1:
        raise ValueError("Protocol must authorize exactly one attempt.")
    paths, expected = _canonical_paths(), _expected_input_sha256()
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected):
        raise ValueError("Protocol input bindings are incomplete.")
    for name, digest in expected.items():
        binding = inputs[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError("Malformed input binding {}.".format(name))
        _require_canonical(binding["path"], paths[name], "input " + name)
        if binding["sha256"] != digest:
            raise ValueError("Input {} is not frozen to the reviewed SHA-256.".format(name))
    repository = protocol.get("repository")
    if not isinstance(repository, Mapping) or set(repository) != {"project_root", "expected_clean_git_head", "required_replay_ancestor", "code_sha256"}:
        raise ValueError("Repository binding is malformed.")
    _require_canonical(repository["project_root"], PROJECT_ROOT, "repository.project_root")
    if repository["required_replay_ancestor"] != REQUIRED_REPLAY_ANCESTOR:
        raise ValueError("Replay ancestor binding differs.")
    head = str(repository["expected_clean_git_head"])
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ValueError("Protocol git HEAD is invalid.")
    code = repository["code_sha256"]
    if not isinstance(code, Mapping) or set(code) != set(CODE_PATHS):
        raise ValueError("Protocol code binding is incomplete.")
    for name in CODE_PATHS:
        _require_sha256(code[name], "code " + name)
    dataset = protocol.get("validation_dataset")
    manifest_entries = _manifest_val_entries(
        paths["dataset_manifest"], protocol["inputs"]["dataset_manifest"]["sha256"]
    )
    expected_dataset = {"split": "val", "manifest_sha256": OFFICIAL_MANIFEST_SHA256,
                        "dataset_signature": OFFICIAL_CACHE_DATASET_SIGNATURE,
                        "video_count": OFFICIAL_VIDEO_COUNT, "event_count": OFFICIAL_EVENT_COUNT,
                        "canonical_stems": list(OFFICIAL_STEMS),
                        "manifest_files": manifest_entries}
    if dataset != expected_dataset:
        raise ValueError("Protocol validation dataset binding differs.")
    if protocol.get("runtime") != _runtime_contract(paths):
        raise ValueError("Protocol runtime is not the frozen singleton runtime.")
    if protocol.get("baselines") != {"C00_released_m20": GOLDEN_C00, "C09_released_m20": C09_ACTUAL}:
        raise ValueError("Protocol baselines differ.")
    if protocol.get("promotion_gates") != PROMOTION_GATES:
        raise ValueError("Protocol promotion gates differ.")
    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"claim", "m10_cache", "e3_cache", "report"}:
        raise ValueError("Protocol outputs are malformed.")
    for name in outputs:
        _require_canonical(outputs[name], paths[name], "output " + name)
    _require_distinct([(name, binding["path"]) for name, binding in inputs.items()] +
                      [(name, value) for name, value in outputs.items()])


def preflight_execution() -> dict:
    """Create the canonical immutable protocol without opening val/cache payloads."""
    paths = _canonical_paths()
    if paths["protocol"].exists():
        raise FileExistsError("Canonical protocol already exists: {}".format(paths["protocol"]))
    for name in ("claim", "m10_cache", "e3_cache", "report"):
        if paths[name].exists():
            raise FileExistsError("Canonical {} already exists: {}".format(name, paths[name]))
    protocol = build_execution_protocol()
    paths["protocol"].parent.mkdir(parents=False, exist_ok=False)
    digest = _atomic_no_clobber_json(paths["protocol"], protocol)
    return {"protocol_path": str(paths["protocol"]), "protocol_sha256": digest,
            "claim_created": False, "validation_or_cache_loaded": False}


def _preclaim_validate(protocol_path: Path, expected_protocol_sha256: str):
    paths = _canonical_paths()
    _require_canonical(protocol_path, paths["protocol"], "execution protocol")
    expected_protocol_sha256 = _require_sha256(expected_protocol_sha256, "protocol SHA-256")
    protocol = _load_json(paths["protocol"], expected_protocol_sha256, "execution protocol")
    validate_execution_protocol(protocol)
    for name in ("claim", "m10_cache", "e3_cache", "report"):
        if paths[name].exists():
            raise FileExistsError("Canonical {} already exists: {}".format(name, paths[name]))
    git = _git_state()
    if not git["clean"] or not git["required_replay_ancestor_present"]:
        raise RuntimeError("Run requires a clean replay-compatible worktree.")
    if git["head"] != protocol["repository"]["expected_clean_git_head"]:
        raise RuntimeError("Git HEAD differs from the protocol.")
    code = _code_sha256()
    if code != protocol["repository"]["code_sha256"]:
        raise RuntimeError("Code differs from the protocol.")
    input_hashes = {}
    for name, binding in protocol["inputs"].items():
        path = paths[name]
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise ValueError("Frozen input {} changed.".format(name))
        input_hashes[name] = digest
    _validate_formal_lineage(paths, protocol["inputs"])
    return protocol, paths, git, code, input_hashes, expected_protocol_sha256


def _cache_cli(replay, protocol, paths, checkpoint_name: str, cache_name: str, override_name: str) -> None:
    arguments = ["cache", "--config", str(paths["source_config"])]
    for override in protocol["runtime"][override_name]:
        arguments.extend(("--override", override))
    arguments.extend(("--checkpoint", str(paths[checkpoint_name]), "--output-cache", str(paths[cache_name]),
                      "--expected-video-count", str(OFFICIAL_VIDEO_COUNT), "--device", "cuda:0"))
    if replay.main(arguments) != 0:
        raise RuntimeError("Cache generation failed for {}.".format(checkpoint_name))


def _validate_and_route(replay, protocol, paths, primary, secondary, t32_cfg):
    for name, payload, checkpoint_name in (
        ("primary", primary, "e3_checkpoint"), ("secondary", secondary, "m10_checkpoint")
    ):
        metadata = payload["metadata"]
        if metadata.get("dataset_split") != "val" or int(metadata.get("video_count", -1)) != OFFICIAL_VIDEO_COUNT:
            raise ValueError("{} cache is not complete official validation.".format(name))
        if int(metadata.get("event_count", -1)) != OFFICIAL_EVENT_COUNT:
            raise ValueError("{} cache event count is not official.".format(name))
        if metadata.get("dataset_signature") != OFFICIAL_CACHE_DATASET_SIGNATURE:
            raise ValueError("{} cache dataset signature is not official.".format(name))
        if Path(str(metadata.get("checkpoint_path", ""))).resolve() != paths[checkpoint_name]:
            raise ValueError("{} cache checkpoint path differs.".format(name))
        if metadata.get("checkpoint_sha256") != protocol["inputs"][checkpoint_name]["sha256"]:
            raise ValueError("{} cache checkpoint SHA-256 differs.".format(name))
        cache_code = metadata.get("code_sha256")
        if not isinstance(cache_code, Mapping):
            raise ValueError("{} cache lacks code provenance.".format(name))
        for relative in replay.CACHE_CODE_PROVENANCE_PATHS:
            if cache_code.get(relative) != protocol["repository"]["code_sha256"].get(relative):
                raise ValueError("{} cache inference code differs at {}.".format(name, relative))
    names = tuple(Path(item["file_name"]).stem for item in primary["records"])
    if names != OFFICIAL_STEMS:
        raise ValueError("Primary cache record order is not canonical.")
    binding = replay._validate_cache_compatibility(
        primary, secondary, secondary_max_events=ROUTE_CUTOFF,
        allow_warm_primary_t32_secondary_m10_t16=True,
        runtime_inference_settings=replay._inference_settings(t32_cfg),
    )
    records = replay.route_cache_records(
        primary, secondary, ROUTE_CUTOFF,
        allow_warm_primary_t32_secondary_m10_t16=True,
        runtime_inference_settings=replay._inference_settings(t32_cfg),
    )
    for record in records:
        expected = "secondary" if record.event_count <= ROUTE_CUTOFF else "primary"
        if record.score_source != expected:
            raise RuntimeError("Fixed <=30000 M10 / >30000 E3 route was violated.")
    return binding, records


def _evaluate_profile(replay, records, cfg) -> dict:
    prepared = replay.precompute_video_counts(
        records, ROUTE_CUTOFF, (LOW_THRESHOLD,), (HIGH_THRESHOLD,), cfg
    )
    per_video, counts = [], []
    for item in prepared:
        threshold = LOW_THRESHOLD if item["event_count"] <= ROUTE_CUTOFF else HIGH_THRESHOLD
        count = item["counts_by_threshold"][threshold]
        counts.append(count)
        per_video.append({"file_name": item["file_name"], "event_count": item["event_count"],
                          "score_source": item["score_source"], "threshold": threshold,
                          "counts": asdict(count)})
    total = replay._sum_counts(counts)
    metrics = replay.metrics_from_counts_exact(total, cfg).to_dict()
    return {"counts": asdict(total), "metrics": metrics, "per_video": per_video}


def _run_claimed(protocol, paths):
    # Deliberately lazy: this is the first torch/replay import and is reached
    # only after the durable O_EXCL claim exists.
    import replay_temporal_memory_validation as replay

    validation_before = _validate_validation_files(paths)
    _cache_cli(replay, protocol, paths, "m10_checkpoint", "m10_cache", "m10_cache_overrides")
    validation_after_m10 = _validate_validation_files(paths)
    if validation_after_m10 != validation_before:
        raise RuntimeError("Official validation files changed during M10 cache generation.")
    _cache_cli(replay, protocol, paths, "e3_checkpoint", "e3_cache", "e3_cache_overrides")
    validation_after_e3 = _validate_validation_files(paths)
    if validation_after_e3 != validation_before:
        raise RuntimeError("Official validation files changed during E3 cache generation.")
    secondary, secondary_sha = replay.load_cache_snapshot(paths["m10_cache"])
    primary, primary_sha = replay.load_cache_snapshot(paths["e3_cache"])
    c00_cfg = replay.load_flat_config(paths["source_config"], protocol["runtime"]["postprocess_profiles"]["C00"])
    c09_cfg = replay.load_flat_config(paths["source_config"], protocol["runtime"]["postprocess_profiles"]["C09"])
    binding, records = _validate_and_route(replay, protocol, paths, primary, secondary, c00_cfg)
    c00 = _evaluate_profile(replay, records, c00_cfg)
    c09 = _evaluate_profile(replay, records, c09_cfg)
    metrics = c09["metrics"]
    gates = {
        "minimum_score_delta_over_c09": metrics["score"] - C09_ACTUAL["score"] >= PROMOTION_GATES["minimum_score_delta_over_c09"],
        "minimum_pd": metrics["pd"] >= PROMOTION_GATES["minimum_pd"],
        "minimum_iou": metrics["iou"] >= PROMOTION_GATES["minimum_iou"],
        "maximum_fa": metrics["fa"] <= PROMOTION_GATES["maximum_fa"],
    }
    comparisons = {
        "C00_vs_released_m20": {
            name: float(c00["metrics"][name]) - float(GOLDEN_C00[name])
            for name in GOLDEN_C00
        },
        "C09_vs_frozen_C09": {
            name: float(c09["metrics"][name]) - float(C09_ACTUAL[name])
            for name in C09_ACTUAL
        },
    }
    return {"validation_dataset_stages": {
                "before_m10": validation_before,
                "after_m10": validation_after_m10,
                "after_e3": validation_after_e3,
            },
            "cache_sha256": {"m10_t16": secondary_sha, "e3_t32": primary_sha},
            "route_binding": binding, "profiles": {"C00": c00, "C09": c09},
            "comparisons": comparisons, "gates": gates, "passed": all(gates.values())}


def _failure_report(protocol_path, protocol_sha, claim_payload, claim_sha, stage, error, integrity):
    return {
        "schema": REPORT_SCHEMA, "created_utc": _utc_now(), "status": "failed",
        "passed": False, "evidence_class": "single_claimed_frozen_t32_e3_validation_attempt",
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "claim": {"payload": claim_payload, "sha256": claim_sha},
        "failure": {"stage": stage, "type": type(error).__name__, "message": str(error)},
        "post_failure_integrity_observation": integrity,
        "failure_action": PROMOTION_GATES["failure_action"],
        "submission_zip_created": False, "platform_upload_performed": False,
    }


def _cache_artifact_observation(paths: Mapping[str, Path]) -> dict:
    result = {}
    for name in ("m10_cache", "e3_cache"):
        path = paths[name]
        item = {"path": str(path.resolve()), "exists": path.is_file()}
        if item["exists"]:
            item["size"] = path.stat().st_size
            item["sha256"] = sha256_file(path)
        result[name] = item
    return result


def run_execution(protocol_path: Path, expected_protocol_sha256: str) -> dict:
    protocol, paths, git_before, code_before, hashes_before, protocol_sha = _preclaim_validate(
        protocol_path, expected_protocol_sha256
    )
    claim_payload, claim_sha = _atomic_claim(paths["claim"], paths["protocol"], protocol_sha, paths["report"])
    stage = "after_claim_before_cache_generation"
    try:
        stage = "cache_generation_and_fixed_evaluation"
        outcome = _run_claimed(protocol, paths)
        stage = "post_run_immutability_check"
        hashes_after = {name: sha256_file(paths[name]) for name in protocol["inputs"]}
        git_after, code_after = _git_state(), _code_sha256()
        if hashes_after != hashes_before or git_after != git_before or code_after != code_before:
            raise RuntimeError("Immutable inputs, code, or git changed during validation.")
        cache_hashes_after = {
            "m10_t16": sha256_file(paths["m10_cache"]),
            "e3_t32": sha256_file(paths["e3_cache"]),
        }
        if cache_hashes_after != outcome["cache_sha256"]:
            raise RuntimeError("A generated raw cache changed during fixed evaluation.")
        validation_after = _validate_validation_files(paths)
        validation_before = outcome["validation_dataset_stages"]["before_m10"]
        if validation_after != validation_before:
            raise RuntimeError("Official validation files changed during the one-shot run.")
        if sha256_file(paths["protocol"]) != protocol_sha or sha256_file(paths["claim"]) != claim_sha:
            raise RuntimeError("Protocol or attempt claim changed during validation.")
        report = {
            "schema": REPORT_SCHEMA, "created_utc": _utc_now(), "status": "completed",
            "passed": outcome["passed"], "evidence_class": "single_claimed_frozen_t32_e3_validation_attempt",
            "protocol": {"path": str(paths["protocol"]), "sha256": protocol_sha},
            "claim": {"payload": claim_payload, "sha256": claim_sha},
            "inputs": protocol["inputs"], "repository": {"before": git_before, "after": git_after,
                                                                "code_sha256": code_before},
            "runtime": protocol["runtime"], "baselines": protocol["baselines"],
            "validation_dataset_integrity": {**outcome["validation_dataset_stages"],
                                               "before_report": validation_after,
                                               "all_stages_equal": True},
            "cache_artifacts": _cache_artifact_observation(paths),
            "cache_sha256": outcome["cache_sha256"], "route_binding": outcome["route_binding"],
            "profiles": outcome["profiles"], "comparisons": outcome["comparisons"],
            "gates": outcome["gates"],
            "failure_action": None if outcome["passed"] else PROMOTION_GATES["failure_action"],
            "submission_zip_created": False, "platform_upload_performed": False,
        }
    except BaseException as error:
        integrity = {}
        try:
            observed_inputs = {name: sha256_file(paths[name]) for name in protocol["inputs"]}
            integrity["inputs_equal"] = observed_inputs == hashes_before
            integrity["observed_input_sha256"] = observed_inputs
        except Exception as observation_error:
            integrity["input_observation_error"] = repr(observation_error)
        try:
            observed_git, observed_code = _git_state(), _code_sha256()
            integrity["git_equal"] = observed_git == git_before
            integrity["code_equal"] = observed_code == code_before
            integrity["observed_git"] = observed_git
            integrity["observed_code_sha256"] = observed_code
        except Exception as observation_error:
            integrity["repository_observation_error"] = repr(observation_error)
        try:
            protocol_observed_sha = sha256_file(paths["protocol"])
            claim_observed_sha = sha256_file(paths["claim"])
            integrity["protocol"] = {
                "path": str(paths["protocol"]), "sha256": protocol_observed_sha,
                "equals_claimed": protocol_observed_sha == protocol_sha,
            }
            integrity["claim"] = {
                "path": str(paths["claim"]), "sha256": claim_observed_sha,
                "equals_created": claim_observed_sha == claim_sha,
            }
        except Exception as observation_error:
            integrity["protocol_claim_observation_error"] = repr(observation_error)
        try:
            integrity["cache_artifacts"] = _cache_artifact_observation(paths)
        except Exception as observation_error:
            integrity["cache_artifact_observation_error"] = repr(observation_error)
        try:
            integrity["validation_dataset"] = _validate_validation_files(paths)
        except Exception as observation_error:
            integrity["validation_dataset_observation_error"] = repr(observation_error)
        report = _failure_report(
            paths["protocol"], protocol_sha, claim_payload, claim_sha,
            stage, error, integrity,
        )
    _atomic_no_clobber_json(paths["report"], report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    run = commands.add_parser("run")
    run.add_argument("--execution-protocol", type=Path, required=True)
    run.add_argument("--expected-execution-protocol-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight_execution(), indent=2, sort_keys=True))
        return 0
    report = run_execution(args.execution_protocol, args.expected_execution_protocol_sha256)
    print("report:", _canonical_paths()["report"])
    print("promotion gates passed:", report["passed"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

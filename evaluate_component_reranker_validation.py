"""Execute one protocol-bound Challenge 2 component-reranker validation.

``run`` has no path, threshold, routing, or output choices.  Every such value
comes from one immutable execution protocol.  After all label-free checks pass,
an exclusive claim file is durably created *before* either validation cache is
loaded.  The claim is never removed, including on crashes and failed gates.

``preflight`` performs the same immutable-input checks and additionally checks
cache metadata, but neither scores validation labels nor creates a claim.
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
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import torch

import replay_temporal_memory_validation as replay
from utils.challenge_eval import add_batch_to_evaluator
from utils.component_reranker import load_artifact_payload, sha256_file
from utils.density_threshold import ChallengeCountTotals, select_density_threshold
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


EXECUTION_PROTOCOL_SCHEMA = (
    "ev-uav-component-reranker-validation-execution-protocol-v1"
)
REPORT_SCHEMA = "ev-uav-component-reranker-frozen-validation-report-v2"
CLAIM_SCHEMA = "ev-uav-component-reranker-validation-attempt-claim-v1"
POLICY_SCHEMA = "ev-uav-component-reranker-singleton-validation-policy-v1"

OFFICIAL_VALIDATION_VIDEO_COUNT = 24
OFFICIAL_VALIDATION_EVENT_COUNT = 1424330
OFFICIAL_VALIDATION_STEMS = tuple("val_{:03d}".format(i) for i in range(24))
LOW_THRESHOLD = 0.718
HIGH_THRESHOLD = 0.719
DENSITY_CUTOFF = 30000
SECONDARY_MAX_EVENTS = 30000
RERANKER_CUTOFF = 100000
MINIMUM_SCORE_DELTA = 0.0001

FROZEN_CONFIG_OVERRIDES = (
    "TEST.prediction_threshold=0.719",
    "TEMPORAL_FRAME.temporal_frame_enabled=false",
    "TEMPORAL_MEMORY.temporal_memory_enabled=true",
    "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0",
    "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true",
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
)

CODE_PATHS = (
    "evaluate_component_reranker_validation.py",
    "replay_temporal_memory_validation.py",
    "utils/challenge_eval.py",
    "utils/component_reranker.py",
    "utils/density_threshold.py",
    "utils/eval.py",
    "utils/postprocess.py",
)
INPUT_NAMES = (
    "policy",
    "artifact",
    "primary_cache",
    "secondary_cache",
    "config",
    "primary_checkpoint",
    "secondary_checkpoint",
)
FROZEN_EXPERIMENT_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "20260810_component_reranker_posthoc_singleton_v2"
).resolve()


def _canonical_experiment_paths():
    """Return the one non-selectable path set for this frozen experiment."""
    directory = Path(FROZEN_EXPERIMENT_DIRECTORY).resolve()
    return {
        "execution_protocol": directory / "validation_execution_protocol.json",
        "policy": directory / "validation_acceptance_policy.json",
        "artifact": directory / "component_reranker_full54_v2.json",
        "claim": directory / "validation_attempt_claim.json",
        "report": directory / "frozen_validation_report.json",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(value, name):
    value = str(value).strip().lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("{} must be a lowercase 64-character SHA-256.".format(name))
    return value


def _canonical_path(value, name):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError("{} must be an absolute path.".format(name))
    resolved = path.resolve()
    if str(path) != str(resolved):
        raise ValueError("{} is not stored as its canonical resolved path.".format(name))
    return resolved


def _paths_alias(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            pass
    return False


def _require_distinct_paths(named_paths):
    items = [(name, Path(path).resolve()) for name, path in named_paths]
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1 :]:
            if _paths_alias(left_path, right_path):
                raise ValueError(
                    "Path conflict: {} and {} alias {}.".format(
                        left_name, right_name, left_path
                    )
                )


def _git_state(project_root):
    def run(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=str(project_root),
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


def _code_sha256(project_root):
    result = {}
    for relative in CODE_PATHS:
        path = Path(project_root) / relative
        if not path.is_file():
            raise FileNotFoundError("Required code file is missing: {}".format(path))
        result[relative] = sha256_file(path)
    return result


def _load_json_snapshot(path, expected_sha256, name):
    path = Path(path).resolve()
    expected_sha256 = _require_sha256(expected_sha256, name + " SHA-256")
    before = sha256_file(path)
    if before != expected_sha256:
        raise ValueError(
            "{} SHA-256 {} does not match expected {}.".format(
                name, before, expected_sha256
            )
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if sha256_file(path) != before:
        raise RuntimeError("{} changed while it was loaded.".format(name))
    return payload


def validate_policy(policy):
    if not isinstance(policy, Mapping) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("Unsupported validation policy schema.")
    if policy.get("status") != "frozen_before_v2_artifact_and_before_validation_replay":
        raise ValueError("Validation policy was not frozen at the required time.")
    budget = policy.get("evaluation_budget")
    if not isinstance(budget, Mapping) or budget.get("full_validation_replays") != 1:
        raise ValueError("Validation policy must authorize exactly one replay.")
    if budget.get("threshold_or_hyperparameter_search_after_replay") is not False:
        raise ValueError("Validation policy must prohibit post-replay tuning.")
    contract = policy.get("frozen_inference_contract")
    required_contract = {
        "low_model_route": "M10 when event_count <= 30000",
        "primary_model_route": "released M20 when event_count > 30000",
        "component_reranker_route": "enabled only when event_count > 100000",
        "low_threshold": LOW_THRESHOLD,
        "primary_threshold": HIGH_THRESHOLD,
        "p0c_retain_min_score": 0.95,
        "p0c_density_retain_enabled": False,
        "postprocess_order": "P0/P0c -> component reranker -> P18",
    }
    if not isinstance(contract, Mapping):
        raise ValueError("Policy lacks a frozen inference contract.")
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError("Policy inference contract differs at {}.".format(key))
    golden, c09, gates = (
        policy.get("golden_baseline"),
        policy.get("existing_exploratory_c09"),
        policy.get("promotion_gates"),
    )
    metrics = {"iou", "acc", "pd", "fa", "score_fa", "score"}
    if not isinstance(golden, Mapping) or not metrics.issubset(golden):
        raise ValueError("Policy golden baseline is incomplete.")
    if not isinstance(c09, Mapping) or "score" not in c09:
        raise ValueError("Policy C09 baseline is incomplete.")
    required_gates = {
        "minimum_score": float(golden["score"]) + MINIMUM_SCORE_DELTA,
        "minimum_score_delta_over_golden": MINIMUM_SCORE_DELTA,
        "must_exceed_existing_c09": True,
        "minimum_pd": float(golden["pd"]),
        "minimum_iou": float(golden["iou"]),
        "maximum_fa_exclusive": float(golden["fa"]),
        "noneligible_videos_bitwise_unchanged": True,
        "eligible_video_rule": "event_count > 100000",
        "each_eligible_video_score_delta_nonnegative": True,
        "at_least_one_eligible_video_score_delta_strictly_positive": True,
        "all_gates_required": True,
    }
    if not isinstance(gates, Mapping) or set(gates) != set(required_gates):
        raise ValueError("Policy promotion-gate keys differ from the frozen schema.")
    for key, expected in required_gates.items():
        if gates.get(key) != expected:
            raise ValueError("Policy promotion gate differs at {}.".format(key))


def _expected_runtime_mapping():
    return {
        "density_cutoff": DENSITY_CUTOFF,
        "secondary_max_events": SECONDARY_MAX_EVENTS,
        "component_reranker_cutoff": RERANKER_CUTOFF,
        "low_threshold": LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
        "low_model_route": "secondary_m10_if_event_count_lte_30000",
        "primary_model_route": "primary_m20_if_event_count_gt_30000",
        "component_reranker_route": "enabled_if_event_count_gt_100000",
        "config_overrides": list(FROZEN_CONFIG_OVERRIDES),
    }


def validate_execution_protocol(protocol):
    if not isinstance(protocol, Mapping) or protocol.get("schema") != EXECUTION_PROTOCOL_SCHEMA:
        raise ValueError("Unsupported execution protocol schema.")
    expected_top = {
        "schema",
        "created_utc",
        "attempt_budget",
        "repository",
        "inputs",
        "validation_dataset",
        "runtime",
        "outputs",
    }
    if set(protocol) != expected_top:
        raise ValueError("Execution protocol top-level keys differ from the schema.")
    if protocol.get("attempt_budget") != 1:
        raise ValueError("Execution protocol attempt_budget must be exactly one.")
    repository = protocol.get("repository")
    if not isinstance(repository, Mapping) or set(repository) != {
        "project_root",
        "expected_clean_git_head",
        "code_sha256",
    }:
        raise ValueError("Execution protocol repository binding is incomplete.")
    _canonical_path(repository["project_root"], "repository.project_root")
    head = str(repository["expected_clean_git_head"]).lower()
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ValueError("Execution protocol git HEAD is invalid.")
    code = repository.get("code_sha256")
    if not isinstance(code, Mapping) or set(code) != set(CODE_PATHS):
        raise ValueError("Execution protocol must bind every CODE_PATHS file.")
    for relative in CODE_PATHS:
        _require_sha256(code[relative], "code SHA-256 for " + relative)

    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_NAMES):
        raise ValueError("Execution protocol input bindings are incomplete.")
    for name in INPUT_NAMES:
        binding = inputs[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError("Execution protocol input {} is malformed.".format(name))
        _canonical_path(binding["path"], "inputs.{}.path".format(name))
        _require_sha256(binding["sha256"], "inputs.{}.sha256".format(name))

    dataset = protocol.get("validation_dataset")
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "dataset_signature",
        "video_count",
        "event_count",
        "canonical_stems",
    }:
        raise ValueError("Execution protocol validation dataset binding is incomplete.")
    _require_sha256(dataset["dataset_signature"], "validation dataset signature")
    if int(dataset["video_count"]) != OFFICIAL_VALIDATION_VIDEO_COUNT:
        raise ValueError("Execution protocol must bind exactly 24 validation videos.")
    if int(dataset["event_count"]) != OFFICIAL_VALIDATION_EVENT_COUNT:
        raise ValueError("Execution protocol validation event count is not official.")
    if tuple(dataset["canonical_stems"]) != OFFICIAL_VALIDATION_STEMS:
        raise ValueError("Execution protocol validation stems are not canonical.")
    if protocol.get("runtime") != _expected_runtime_mapping():
        raise ValueError("Execution protocol runtime differs from frozen constants.")
    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"report_path", "claim_path"}:
        raise ValueError("Execution protocol output binding is incomplete.")
    _canonical_path(outputs["report_path"], "outputs.report_path")
    _canonical_path(outputs["claim_path"], "outputs.claim_path")
    canonical = _canonical_experiment_paths()
    fixed_bindings = {
        "policy": protocol["inputs"]["policy"]["path"],
        "artifact": protocol["inputs"]["artifact"]["path"],
        "report": outputs["report_path"],
        "claim": outputs["claim_path"],
    }
    for name, stored in fixed_bindings.items():
        if Path(stored) != canonical[name]:
            raise ValueError(
                "Execution protocol {} path differs from the canonical frozen experiment path."
                .format(name)
            )


def _protocol_paths(protocol, protocol_path):
    paths = {name: Path(protocol["inputs"][name]["path"]) for name in INPUT_NAMES}
    paths["execution_protocol"] = Path(protocol_path).resolve()
    paths["report"] = Path(protocol["outputs"]["report_path"])
    paths["claim"] = Path(protocol["outputs"]["claim_path"])
    return paths


def _validate_policy_artifact_contract(protocol, policy, artifact, code_hashes):
    validate_policy(policy)
    policy_sha = protocol["inputs"]["policy"]["sha256"]
    provenance = artifact.get("provenance", {})
    policy_binding = provenance.get("crossfit_hypothesis", {}).get(
        "validation_acceptance_policy", {}
    )
    if policy_binding.get("schema") != POLICY_SCHEMA or policy_binding.get("sha256") != policy_sha:
        raise ValueError("Artifact is not bound to the frozen validation policy.")
    if artifact.get("keep_probability") != 0.4 or artifact.get("prediction_threshold") != HIGH_THRESHOLD:
        raise ValueError("Artifact is not the frozen kp=0.4 singleton.")
    fit = artifact.get("fit")
    if not isinstance(fit, Mapping) or fit.get("positive_weight") != 4.0 or fit.get("l2") != 0.1:
        raise ValueError("Artifact is not the frozen pw=4/l2=0.1 singleton.")
    if provenance.get("crossfit_candidate_profile") != "posthoc_pw4_kp040_v2":
        raise ValueError("Artifact profile differs from the frozen singleton.")
    if tuple(provenance.get("config_overrides", ())) != FROZEN_CONFIG_OVERRIDES:
        raise ValueError("Artifact config overrides differ from the frozen contract.")
    if provenance.get("config_sha256") != protocol["inputs"]["config"]["sha256"]:
        raise ValueError("Artifact config SHA-256 differs from the protocol.")
    if provenance.get("base_checkpoint_sha256") != protocol["inputs"]["primary_checkpoint"]["sha256"]:
        raise ValueError("Artifact base checkpoint differs from protocol M20.")
    training_code = provenance.get("crossfit_code_sha256")
    shared = (
        "replay_temporal_memory_validation.py",
        "utils/challenge_eval.py",
        "utils/component_reranker.py",
        "utils/eval.py",
        "utils/postprocess.py",
    )
    if not isinstance(training_code, Mapping) or any(
        training_code.get(path) != code_hashes[path] for path in shared
    ):
        raise ValueError("Runtime core differs from artifact training provenance.")


def _preclaim_validate(protocol_path, expected_protocol_sha256):
    project_root = Path(__file__).resolve().parent
    protocol_path = Path(protocol_path).expanduser().resolve()
    canonical_protocol = _canonical_experiment_paths()["execution_protocol"]
    if protocol_path != canonical_protocol:
        raise ValueError(
            "Only the canonical frozen execution protocol may be used: {}".format(
                canonical_protocol
            )
        )
    protocol = _load_json_snapshot(
        protocol_path, expected_protocol_sha256, "execution protocol"
    )
    validate_execution_protocol(protocol)
    paths = _protocol_paths(protocol, protocol_path)
    if Path(protocol["repository"]["project_root"]) != project_root:
        raise ValueError("Execution protocol is bound to a different project root.")
    _require_distinct_paths(paths.items())
    # The canonical claim is the authoritative 1/1 budget guard.  Check it
    # first even when a successful report also exists.
    for name in ("claim", "report"):
        if paths[name].exists():
            raise FileExistsError("{} already exists: {}".format(name, paths[name]))
        if not paths[name].parent.is_dir():
            raise FileNotFoundError("{} parent directory does not exist.".format(name))
    git = _git_state(project_root)
    if not git["clean"]:
        raise RuntimeError("Git worktree must be clean before claiming validation.")
    if git["head"] != protocol["repository"]["expected_clean_git_head"]:
        raise RuntimeError("Git HEAD differs from the execution protocol.")
    code = _code_sha256(project_root)
    if code != protocol["repository"]["code_sha256"]:
        raise RuntimeError("Code hashes differ from the execution protocol.")
    input_hashes = {}
    for name in INPUT_NAMES:
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError("Protocol input does not exist: {}".format(path))
        digest = sha256_file(path)
        if digest != protocol["inputs"][name]["sha256"]:
            raise ValueError("{} hash differs from execution protocol.".format(name))
        input_hashes[name] = digest
    policy = _load_json_snapshot(paths["policy"], input_hashes["policy"], "policy")
    artifact, _ = load_artifact_payload(paths["artifact"], input_hashes["artifact"])
    _validate_policy_artifact_contract(protocol, policy, artifact, code)
    return protocol, paths, policy, artifact, git, code, input_hashes


def _validate_loaded_cache_contract(protocol, primary, secondary):
    replay._validate_cache_compatibility(primary, secondary)
    dataset = protocol["validation_dataset"]
    for name, payload, checkpoint_name in (
        ("primary", primary, "primary_checkpoint"),
        ("secondary", secondary, "secondary_checkpoint"),
    ):
        metadata = payload["metadata"]
        if metadata["dataset_signature"] != dataset["dataset_signature"]:
            raise ValueError("{} cache dataset signature differs from protocol.".format(name))
        if int(metadata["video_count"]) != dataset["video_count"]:
            raise ValueError("{} cache video count differs from protocol.".format(name))
        if int(metadata["event_count"]) != dataset["event_count"]:
            raise ValueError("{} cache event count differs from protocol.".format(name))
        if str(Path(metadata.get("checkpoint_path", "")).resolve()) != protocol["inputs"][checkpoint_name]["path"]:
            raise ValueError("{} cache checkpoint path differs from protocol.".format(name))
        if metadata["checkpoint_sha256"].lower() != protocol["inputs"][checkpoint_name]["sha256"]:
            raise ValueError("{} cache checkpoint SHA-256 differs from protocol.".format(name))
    names = [Path(record["file_name"]).stem for record in primary["records"]]
    if tuple(names) != OFFICIAL_VALIDATION_STEMS:
        raise ValueError("Loaded validation records are not in canonical order.")


def _atomic_claim(path, protocol_path, protocol_sha256, report_path):
    payload = {
        "schema": CLAIM_SCHEMA,
        "claimed_utc": _utc_now(),
        "attempt": 1,
        "attempt_budget": 1,
        "execution_protocol_path": str(Path(protocol_path).resolve()),
        "execution_protocol_sha256": protocol_sha256,
        "report_path": str(Path(report_path).resolve()),
        "state": "claimed_irreversibly_before_validation_cache_load",
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
        # The exclusive path deliberately remains and consumes the attempt even
        # if durability or subsequent validation fails.
        raise
    return payload, sha256_file(path)


def _atomic_no_clobber_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Output already exists: {}".format(path))
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


def _counts_for_predictions(record, predictions, threshold, cfg):
    evaluator = evalute(cfg)
    add_batch_to_evaluator(
        evaluator,
        {"seg_label": record.seg_label, "locs": record.locs, "idx_label": record.idx_label},
        predictions,
        sample_number=0,
        prediction_threshold=threshold,
        collect_roc=True,
    )
    labels = record.seg_label.float().reshape(-1)
    positive = labels > 0.5
    binary = predictions.reshape(-1) >= threshold
    return ChallengeCountTotals(
        true_positive_events=int((binary & positive).sum().item()),
        false_positive_events=int((binary & ~positive).sum().item()),
        positive_events=int(positive.sum().item()),
        detected_target_frames=int(evaluator.correct_num),
        target_frames=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
    )


def _evaluate_one(record, threshold, cfg):
    processor = ChallengePostprocessor.from_cfg(
        cfg, threshold, event_count=record.event_count
    )
    predictions, _ = processor.apply(record.scores.clone(), record.locs)
    predictions = predictions.reshape(-1).cpu().contiguous()
    counts = _counts_for_predictions(record, predictions, threshold, cfg)
    metrics = replay.metrics_from_counts_exact(counts, cfg)
    return predictions, counts, metrics


def _delta(candidate, baseline):
    return {
        name: float(candidate[name]) - float(baseline[name])
        for name in replay.METRIC_NAMES
    }


def _runtime_configs(protocol, paths):
    baseline = replay.load_flat_config(
        paths["config"], protocol["runtime"]["config_overrides"]
    )
    baseline.temporal_memory_model_path = str(paths["primary_checkpoint"])
    baseline.temporal_memory_secondary_model_path = str(paths["secondary_checkpoint"])
    baseline.temporal_memory_secondary_max_event_count = SECONDARY_MAX_EVENTS
    baseline.component_reranker_enabled = False
    baseline.component_reranker_event_count_cutoff = RERANKER_CUTOFF
    baseline.component_reranker_model_path = ""
    baseline.component_reranker_expected_sha256 = ""
    candidate = SimpleNamespace(**vars(baseline))
    candidate.component_reranker_enabled = True
    candidate.component_reranker_model_path = str(paths["artifact"])
    candidate.component_reranker_expected_sha256 = protocol["inputs"]["artifact"]["sha256"]
    return baseline, candidate


def preflight_execution(protocol_path, expected_protocol_sha256):
    """Check immutable bindings and cache metadata without scoring or claiming."""
    expected_protocol_sha256 = _require_sha256(
        expected_protocol_sha256, "execution protocol SHA-256"
    )
    state = _preclaim_validate(protocol_path, expected_protocol_sha256)
    protocol, paths, git_before, code_before, hashes_before = (
        state[0], state[1], state[4], state[5], state[6]
    )
    primary, primary_sha = replay.load_cache_snapshot(paths["primary_cache"])
    secondary, secondary_sha = replay.load_cache_snapshot(paths["secondary_cache"])
    _validate_loaded_cache_contract(protocol, primary, secondary)
    if primary_sha != protocol["inputs"]["primary_cache"]["sha256"] or secondary_sha != protocol["inputs"]["secondary_cache"]["sha256"]:
        raise RuntimeError("Cache changed during preflight.")
    hashes_after = {name: sha256_file(paths[name]) for name in INPUT_NAMES}
    if hashes_after != hashes_before:
        raise RuntimeError("An immutable input changed during preflight.")
    if sha256_file(protocol_path) != expected_protocol_sha256:
        raise RuntimeError("Execution protocol changed during preflight.")
    if _code_sha256(Path(__file__).resolve().parent) != code_before:
        raise RuntimeError("Code changed during preflight.")
    if _git_state(Path(__file__).resolve().parent) != git_before:
        raise RuntimeError("Git state changed during preflight.")
    return {
        "protocol_sha256": expected_protocol_sha256,
        "dataset_signature": protocol["validation_dataset"]["dataset_signature"],
        "video_count": protocol["validation_dataset"]["video_count"],
        "event_count": protocol["validation_dataset"]["event_count"],
        "claim_created": False,
        "validation_scored": False,
    }


def run_execution(protocol_path, expected_protocol_sha256):
    """Consume the sole claim, then run baseline and singleton exactly once."""
    expected_protocol_sha256 = _require_sha256(
        expected_protocol_sha256, "execution protocol SHA-256"
    )
    protocol, paths, policy, artifact, git_before, code_before, hashes_before = (
        _preclaim_validate(protocol_path, expected_protocol_sha256)
    )
    claim_payload, claim_sha256 = _atomic_claim(
        paths["claim"], protocol_path, expected_protocol_sha256, paths["report"]
    )

    # No validation cache is loaded before the durable claim above.
    primary, primary_sha = replay.load_cache_snapshot(paths["primary_cache"])
    secondary, secondary_sha = replay.load_cache_snapshot(paths["secondary_cache"])
    if primary_sha != hashes_before["primary_cache"] or secondary_sha != hashes_before["secondary_cache"]:
        raise RuntimeError("Validation cache changed after claim.")
    _validate_loaded_cache_contract(protocol, primary, secondary)
    baseline_cfg, candidate_cfg = _runtime_configs(protocol, paths)
    binding = replay.validate_component_reranker_cache_binding(
        candidate_cfg, primary, secondary, SECONDARY_MAX_EVENTS
    )
    records = replay.route_cache_records(primary, secondary, SECONDARY_MAX_EVENTS)
    replay.validate_component_reranker_dense_routes(candidate_cfg, records)

    per_video, baseline_counts, candidate_counts = [], [], []
    for index, record in enumerate(records, start=1):
        threshold = select_density_threshold(
            record.event_count, DENSITY_CUTOFF, LOW_THRESHOLD, HIGH_THRESHOLD
        )
        base_pred, base_count, base_metrics = _evaluate_one(record, threshold, baseline_cfg)
        cand_pred, cand_count, cand_metrics = _evaluate_one(record, threshold, candidate_cfg)
        eligible = record.event_count > RERANKER_CUTOFF
        unchanged = torch.equal(base_pred, cand_pred)
        base_dict, cand_dict = base_metrics.to_dict(), cand_metrics.to_dict()
        per_video.append(
            {
                "index": index,
                "file_name": record.file_name,
                "event_count": record.event_count,
                "score_source": record.score_source,
                "threshold": threshold,
                "reranker_eligible": eligible,
                "predictions_bitwise_equal": unchanged,
                "baseline": {"counts": asdict(base_count), "metrics": base_dict},
                "candidate": {"counts": asdict(cand_count), "metrics": cand_dict},
                "delta": _delta(cand_dict, base_dict),
            }
        )
        baseline_counts.append(base_count)
        candidate_counts.append(cand_count)
        print("evaluate {}/24: {}".format(index, record.file_name), flush=True)

    base_count = replay._sum_counts(baseline_counts)
    cand_count = replay._sum_counts(candidate_counts)
    base_metrics = replay.metrics_from_counts_exact(base_count, baseline_cfg).to_dict()
    cand_metrics = replay.metrics_from_counts_exact(cand_count, candidate_cfg).to_dict()
    aggregate_delta = _delta(cand_metrics, base_metrics)
    golden, policy_gates = policy["golden_baseline"], policy["promotion_gates"]
    eligible = [item for item in per_video if item["reranker_eligible"]]
    gates = {
        "golden_baseline_exact_match": all(
            abs(float(base_metrics[name]) - float(golden[name])) <= 1e-12
            for name in replay.METRIC_NAMES
        ),
        "minimum_score": cand_metrics["score"] >= policy_gates["minimum_score"],
        "minimum_score_delta_over_golden": aggregate_delta["score"] >= MINIMUM_SCORE_DELTA,
        "must_exceed_existing_c09": cand_metrics["score"] > policy["existing_exploratory_c09"]["score"],
        "minimum_pd": cand_metrics["pd"] >= policy_gates["minimum_pd"],
        "minimum_iou": cand_metrics["iou"] >= policy_gates["minimum_iou"],
        "maximum_fa_exclusive": cand_metrics["fa"] < policy_gates["maximum_fa_exclusive"],
        "noneligible_videos_bitwise_unchanged": all(
            item["predictions_bitwise_equal"] for item in per_video if not item["reranker_eligible"]
        ),
        "each_eligible_video_score_delta_nonnegative": bool(eligible)
        and all(item["delta"]["score"] >= 0.0 for item in eligible),
        "at_least_one_eligible_video_score_delta_strictly_positive": any(
            item["delta"]["score"] > 0.0 for item in eligible
        ),
    }
    passed = all(gates.values())

    hashes_after = {name: sha256_file(paths[name]) for name in INPUT_NAMES}
    code_after = _code_sha256(Path(__file__).resolve().parent)
    git_after = _git_state(Path(__file__).resolve().parent)
    if hashes_after != hashes_before:
        raise RuntimeError("An immutable input changed after claim.")
    if sha256_file(protocol_path) != expected_protocol_sha256:
        raise RuntimeError("Execution protocol changed after claim.")
    if code_after != code_before or git_after != git_before:
        raise RuntimeError("Code or git state changed after claim.")
    if sha256_file(paths["claim"]) != claim_sha256:
        raise RuntimeError("Attempt claim changed after creation.")

    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": _utc_now(),
        "execution_protocol": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": expected_protocol_sha256,
        },
        "attempt_claim": {
            "path": str(paths["claim"]),
            "sha256": claim_sha256,
            "payload": claim_payload,
        },
        "evidence_class": "single_claimed_frozen_24_validation_replay",
        "inputs": {
            name: {"path": str(paths[name]), "sha256": hashes_before[name]}
            for name in INPUT_NAMES
        },
        "validation_dataset": protocol["validation_dataset"],
        "runtime": protocol["runtime"],
        "repository": {
            "git": git_before,
            "code_sha256": code_before,
            "before_after_equal": True,
        },
        "component_reranker_cache_binding": binding,
        "per_video": per_video,
        "aggregate": {
            "baseline": {"counts": asdict(base_count), "metrics": base_metrics},
            "candidate": {"counts": asdict(cand_count), "metrics": cand_metrics},
            "delta": aggregate_delta,
        },
        "gates": gates,
        "passed": passed,
        "failure_action": None if passed else "archive_without_validation_tuning",
    }
    _atomic_no_clobber_json(paths["report"], report)
    return report


def build_execution_protocol(*, inputs):
    """Build (but do not write) a protocol from reviewed immutable inputs.

    This helper is intentionally absent from the CLI.  It performs metadata
    inspection but no scoring and is meant for the one-time preregistration
    step after this code is committed.
    """
    project_root = Path(__file__).resolve().parent
    git = _git_state(project_root)
    if not git["clean"]:
        raise RuntimeError("Commit all code before building an execution protocol.")
    normalized = {}
    for name in INPUT_NAMES:
        path = _canonical_path(inputs[name], "builder input " + name)
        normalized[name] = {"path": str(path), "sha256": sha256_file(path)}
    primary = replay.load_cache(Path(normalized["primary_cache"]["path"]))
    secondary = replay.load_cache(Path(normalized["secondary_cache"]["path"]))
    replay._validate_cache_compatibility(primary, secondary)
    protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "created_utc": _utc_now(),
        "attempt_budget": 1,
        "repository": {
            "project_root": str(project_root),
            "expected_clean_git_head": git["head"],
            "code_sha256": _code_sha256(project_root),
        },
        "inputs": normalized,
        "validation_dataset": {
            "dataset_signature": primary["metadata"]["dataset_signature"],
            "video_count": int(primary["metadata"]["video_count"]),
            "event_count": int(primary["metadata"]["event_count"]),
            "canonical_stems": [Path(r["file_name"]).stem for r in primary["records"]],
        },
        "runtime": _expected_runtime_mapping(),
        "outputs": {
            "report_path": str(_canonical_experiment_paths()["report"]),
            "claim_path": str(_canonical_experiment_paths()["claim"]),
        },
    }
    validate_execution_protocol(protocol)
    _validate_loaded_cache_contract(protocol, primary, secondary)
    return protocol


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--execution-protocol", type=Path, required=True)
        command.add_argument("--expected-execution-protocol-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.command == "preflight":
        result = preflight_execution(
            args.execution_protocol, args.expected_execution_protocol_sha256
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    report = run_execution(
        args.execution_protocol, args.expected_execution_protocol_sha256
    )
    print("report:", _canonical_experiment_paths()["report"])
    print("promotion gates passed:", report["passed"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

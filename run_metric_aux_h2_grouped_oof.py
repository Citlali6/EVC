"""Run the frozen train-only H2 grouped OOF metric-aux experiment.

The CPU ``audit`` command is the only command intended before explicit GPU
coordination.  All data paths are fail-closed to the eleven frozen
``train_088``...``train_098`` sources.  Validation, test, T32 and prior
persistence-formal artifacts are outside this runner's accepted path contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_h2_grouped_oof_science_v1.json"
EXPECTED_PROTOCOL_SHA256 = "d30d0e45c85ee2f1a8b3c9533ed85f7c5525a421e252d7c1ad2d635fa804aa2b"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-h2-grouped-oof-science-v1"
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260810_metric_aux_h2_grouped_oof_v1"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
PROBE_RESULT_PATH = OUTPUT_ROOT / "resource_probe" / "runtime_result.json"
FORMAL_ROOT = OUTPUT_ROOT / "formal_training"
PAIR_AUDIT_PATH = FORMAL_ROOT / "pair_audit.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_float32(array):
    import numpy as np

    value = np.asarray(array, dtype=np.float32).reshape(-1)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def load_json_snapshot(path):
    path = Path(path)
    before = sha256_file(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("JSON changed while it was being read: {}".format(path))
    return value, before


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite immutable output: {}".format(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("Stale temporary output exists: {}".format(temporary))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def workspace_path(relative):
    path = (WORKSPACE_ROOT / str(relative)).resolve()
    path.relative_to(WORKSPACE_ROOT.resolve())
    return path


def official_train_root(protocol):
    root = workspace_path(protocol["dataset"]["workspace_relative_train_root"])
    if root.name.lower() != "train":
        raise RuntimeError("Frozen dataset root is not the official train directory.")
    return root


def require_official_train_source(path, protocol, expected_name=None):
    path = Path(path).resolve()
    root = official_train_root(protocol).resolve()
    if path.parent != root:
        raise RuntimeError("Data source is outside the frozen train root: {}".format(path))
    if not re.fullmatch(r"train_\d{3}\.npz", path.name):
        raise RuntimeError("Only canonical train_###.npz sources are accepted.")
    if expected_name is not None and path.name != expected_name:
        raise RuntimeError("Train source name differs from the frozen record.")
    lowered = {part.lower() for part in path.parts}
    if "val" in lowered or "validation" in lowered or "test" in lowered:
        raise RuntimeError("Validation/test paths are forbidden.")
    return path


def _expect_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError("{} differs: {!r} != {!r}".format(label, actual, expected))


def source_groups(protocol):
    return protocol["dataset"]["source_groups"]


def source_index(protocol):
    result = {}
    for group, items in source_groups(protocol).items():
        for item in items:
            if item["name"] in result:
                raise RuntimeError("Frozen source groups overlap at {}".format(item["name"]))
            result[item["name"]] = {**item, "group": group}
    return result


def items_for_groups(protocol, group_names):
    output = []
    for name in group_names:
        if name not in source_groups(protocol):
            raise KeyError("Unknown frozen source group: {}".format(name))
        output.extend(source_groups(protocol)[name])
    return output


def held_items(protocol, fold):
    return items_for_groups(protocol, [fold["held_group"]])


def fit_items(protocol, fold):
    return items_for_groups(protocol, fold["fit_groups"])


def validate_protocol(protocol):
    _expect_equal(protocol.get("schema"), EXPECTED_SCHEMA, "protocol schema")
    _expect_equal(
        protocol.get("status"),
        "frozen_before_any_probe_formal_training_or_held_evaluation",
        "protocol status",
    )
    split = protocol["split_access"]
    _expect_equal(split["allowed_label_split"], "train", "allowed label split")
    _expect_equal(split["validation_or_test_read_allowed"], False, "val/test policy")
    _expect_equal(
        split["prior_persistence_formal_artifact_read_allowed"],
        False,
        "persistence formal policy",
    )
    amendment = protocol["audit_amendment"]
    _expect_equal(
        amendment["shared_parent_pretraining_exposure"],
        True,
        "shared-parent exposure disclosure",
    )
    _expect_equal(
        amendment["claim_scope"],
        "incremental_finetune_transfer_not_fold_clean_model_generalization",
        "claim scope",
    )

    defaults = protocol["history_audit"]["base_config_defaults"]
    expected_defaults = {
        "metric_aux_enabled": False,
        "metric_target_weight": 0.01,
        "metric_component_weight": 0.002,
        "metric_warmup_epochs": 5,
        "metric_spatial_cell_size": 3,
        "metric_min_cell_events": 2,
        "metric_component_ratio": 0.01,
        "metric_activation_threshold": 0.7,
        "metric_activation_temperature": 0.1,
    }
    _expect_equal(defaults, expected_defaults, "base metric defaults")
    candidate = protocol["training"]["candidate"]
    frozen_candidate = {
        "metric_aux_enabled": True,
        "metric_target_weight": 0.005,
        "metric_component_weight": 0.001,
        "metric_warmup_epochs": 1,
        "active_zero_based_epochs": [1, 2],
        "metric_spatial_cell_size": 3,
        "metric_min_cell_events": 2,
        "metric_component_ratio": 0.01,
        "metric_activation_threshold": 0.719,
        "metric_activation_temperature": 0.1,
    }
    for key, value in frozen_candidate.items():
        _expect_equal(candidate[key], value, "candidate {}".format(key))

    scope = protocol["training"]["training_scope_audit"]
    _expect_equal(scope["name"], "full", "training scope")
    _expect_equal(scope["trainable_state_tensor_count"], 89, "trainable tensors")
    _expect_equal(scope["trainable_parameter_count"], 1924716, "trainable parameters")
    _expect_equal(scope["frozen_parameter_count"], 0, "frozen parameters")
    _expect_equal(
        scope["trainable_name_shape_canonical_sha256"],
        "3bbe7100b5be460eeeea5218cccfb27d9ed697e449d5a3abe1babf918023f05e",
        "training-scope canonical SHA-256",
    )
    if any(scope["narrow_scope_switches"].values()):
        raise RuntimeError("Every narrow training-scope switch must be false.")

    index = source_index(protocol)
    _expect_equal(len(index), 11, "source union count")
    _expect_equal(protocol["dataset"]["source_union_count"], 11, "declared union count")
    all_names = set(index)
    held_occurrences = {name: 0 for name in all_names}
    held_sets = []
    formal_steps = 0
    for fold in protocol["dataset"]["folds"]:
        fit_names = {item["name"] for item in fit_items(protocol, fold)}
        current_held = {item["name"] for item in held_items(protocol, fold)}
        if fit_names & current_held or fit_names | current_held != all_names:
            raise RuntimeError("Fit/held partition is not a complete disjoint source union.")
        if len(fit_names) != int(fold["fit_video_count"]):
            raise RuntimeError("Fit-video count differs in {}".format(fold["fold_id"]))
        if len(current_held) != int(fold["held_video_count"]):
            raise RuntimeError("Held-video count differs in {}".format(fold["fold_id"]))
        expected_steps = (
            len(fit_names)
            * int(protocol["training"]["views_per_video"])
            * int(protocol["training"]["epochs"])
        )
        _expect_equal(
            expected_steps,
            int(fold["expected_optimizer_steps_per_run"]),
            "optimizer steps {}".format(fold["fold_id"]),
        )
        formal_steps += 2 * expected_steps
        for name in current_held:
            held_occurrences[name] += 1
        held_sets.append(current_held)
    if any(left & right for i, left in enumerate(held_sets) for right in held_sets[i + 1 :]):
        raise RuntimeError("Held groups are not pairwise disjoint.")
    if set().union(*held_sets) != all_names or set(held_occurrences.values()) != {1}:
        raise RuntimeError("Every source must be held exactly once.")
    _expect_equal(formal_steps, 1056, "total formal optimizer steps")
    _expect_equal(
        protocol["training"]["formal_optimizer_steps_total"],
        1056,
        "declared formal optimizer steps",
    )

    sampling = protocol["training"]["sampling_contract"]
    if "16 H2 views" not in sampling["historical_m23"]:
        raise RuntimeError("Historical M23 sampling disclosure is missing.")
    if "uniform 8" not in sampling["current_grouped_oof"]:
        raise RuntimeError("Current eight-view budget disclosure is missing.")
    if "not an exact replication" not in sampling["reuse_scope"]:
        raise RuntimeError("M23 reuse scope is overstated.")

    evaluation = protocol["evaluation"]
    _expect_equal(evaluation["split"], "train", "evaluation split")
    _expect_equal(evaluation["t32_allowed"], False, "T32 policy")
    _expect_equal(evaluation["prediction_threshold"], 0.719, "evaluation threshold")
    _expect_equal(
        evaluation["effective_c00_canonical_sha256"],
        "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413",
        "effective C00 SHA-256",
    )
    _expect_equal(
        protocol["promotion_gates"]["comparators"],
        ["paired_baseline_e3", "released_m20"],
        "promotion comparators",
    )
    _expect_equal(
        protocol["promotion_gates"][
            "against_each_comparator_each_fold_score_not_lower"
        ],
        True,
        "per-fold score gate",
    )
    return True


def load_protocol():
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Protocol SHA-256 {} differs from frozen {}".format(
                actual, EXPECTED_PROTOCOL_SHA256
            )
        )
    protocol, snapshot_sha = load_json_snapshot(PROTOCOL_PATH)
    _expect_equal(snapshot_sha, actual, "protocol snapshot SHA-256")
    validate_protocol(protocol)
    return protocol, actual


def runtime_fingerprint_cpu(protocol):
    marker = "METRIC_AUX_RUNTIME="
    code = (
        "import json,platform,sys; import numpy,torch; "
        "print('" + marker + "'+json.dumps({"
        "'python_version':platform.python_version(),"
        "'platform':platform.platform(),"
        "'numpy_version':numpy.__version__,"
        "'torch_version':torch.__version__,"
        "'cuda_initialized':torch.cuda.is_initialized()"
        "},sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(EVC_ROOT),
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError("CPU runtime fingerprint failed: {}".format(completed.stderr))
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("CPU runtime fingerprint did not emit one payload.")
    result = json.loads(lines[0][len(marker) :])
    expected = protocol["runtime_contract"]
    for key in ("python_version", "numpy_version", "torch_version", "platform"):
        _expect_equal(result[key], expected[key], "runtime {}".format(key))
    _expect_equal(result["cuda_initialized"], False, "CPU audit CUDA state")
    return result


def verify_assets(protocol):
    if EVC_ROOT.resolve() != (WORKSPACE_ROOT / "EVC-work").resolve():
        raise RuntimeError("Runner is not located at the frozen repository root.")
    verified_core = []
    for item in protocol["repository"]["core_files"]:
        path = (EVC_ROOT / item["path"]).resolve()
        path.relative_to(EVC_ROOT.resolve())
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError("Frozen core file mismatch: {}".format(path))
        verified_core.append({"path": str(path), "sha256": item["sha256"]})

    checkpoint = workspace_path(protocol["parent_checkpoint"]["workspace_relative_path"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != protocol["parent_checkpoint"]["sha256"]:
        raise RuntimeError("Released M20 checkpoint identity mismatch.")
    architecture_scope_audit = model_architecture_scope_audit_cpu(protocol)
    parent_state_shape_audit = checkpoint_name_shape_audit(protocol, checkpoint)
    history = protocol["history_audit"]["prior_enabled_summary"]
    history_path = workspace_path(history["workspace_relative_path"])
    if not history_path.is_file() or sha256_file(history_path) != history["sha256"]:
        raise RuntimeError("Frozen train-only M23 history record identity mismatch.")

    train_root = official_train_root(protocol)
    if not train_root.is_dir():
        raise NotADirectoryError("Official train root is missing: {}".format(train_root))
    verified_sources = []
    for name, item in sorted(source_index(protocol).items()):
        path = require_official_train_source(train_root / name, protocol, name)
        if not path.is_file():
            raise FileNotFoundError("Frozen train source is missing: {}".format(path))
        stat = path.stat()
        if int(stat.st_size) != int(item["size"]):
            raise RuntimeError("Train source size mismatch: {}".format(path))
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("Train source SHA-256 mismatch: {}".format(path))
        verified_sources.append(
            {
                "name": name,
                "group": item["group"],
                "path": str(path),
                "size": int(stat.st_size),
                "sha256": item["sha256"],
            }
        )
    return {
        "repository_root": str(EVC_ROOT),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": protocol["parent_checkpoint"]["sha256"],
        "parent_checkpoint_name_shape_audit": parent_state_shape_audit,
        "clean_cpu_architecture_scope_audit": architecture_scope_audit,
        "history_record": str(history_path),
        "history_record_sha256": history["sha256"],
        "core_files": verified_core,
        "sources": verified_sources,
        "runtime": runtime_fingerprint_cpu(protocol),
    }


def materialize_view(protocol, view_id, items):
    root = OUTPUT_ROOT / "data_views" / view_id
    train_root = root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    expected_names = {item["name"] for item in items}
    actual_existing = {path.name for path in train_root.glob("*.npz")}
    if actual_existing - expected_names:
        raise RuntimeError(
            "Data view contains unexpected sources: {}".format(
                sorted(actual_existing - expected_names)
            )
        )
    records = []
    official = official_train_root(protocol)
    for item in items:
        source = require_official_train_source(official / item["name"], protocol, item["name"])
        destination = train_root / item["name"]
        if not destination.exists():
            os.link(source, destination)
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise RuntimeError("Data view is not the expected hard link: {}".format(destination))
        if sha256_file(destination) != item["sha256"]:
            raise RuntimeError("Data-view SHA-256 mismatch: {}".format(destination))
        records.append(
            {
                "name": item["name"],
                "source": str(source),
                "destination": str(destination.resolve()),
                "sha256": item["sha256"],
                "samefile": True,
            }
        )
    actual = {path.name for path in train_root.glob("*.npz")}
    if actual != expected_names:
        raise RuntimeError("Materialized fit-view membership differs from protocol.")
    return {"root": str(root.resolve()), "records": records}


def materialize_views(protocol):
    views = {}
    for fold in protocol["dataset"]["folds"]:
        view_id = "{}_fit".format(fold["fold_id"])
        views[view_id] = materialize_view(protocol, view_id, fit_items(protocol, fold))
    probe_name = protocol["resource_probe"]["source"]
    probe_item = source_index(protocol).get(probe_name)
    if probe_item is None:
        raise RuntimeError("Probe source is outside the frozen source union.")
    views["probe"] = materialize_view(protocol, "probe", [probe_item])
    return views


def override_mapping(overrides):
    result = {}
    for value in overrides:
        key, separator, raw = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError("Invalid or duplicate config override: {}".format(value))
        result[key] = raw
    return result


def training_overrides(protocol, data_root, model_root, variant, epochs):
    if variant not in ("baseline", "metric_aux"):
        raise KeyError("Unknown training variant: {}".format(variant))
    training = protocol["training"]
    candidate = training["candidate"]
    checkpoint = workspace_path(protocol["parent_checkpoint"]["workspace_relative_path"])
    enabled = variant == "metric_aux"
    return [
        "DATA.root={}".format(Path(data_root).resolve().as_posix()),
        "TRAIN.seed={}".format(int(training["seed"])),
        "TRAIN.epochs={}".format(int(epochs)),
        "TRAIN.batch_size=1",
        "TRAIN.lr={:.8f}".format(float(training["learning_rate"])),
        "TRAIN.scheduler=cosine",
        "TRAIN.scheduler_min_lr={:.8f}".format(float(training["scheduler_min_lr"])),
        "TRAIN.checkpoint_interval=1",
        "TRAIN.model_save_root={}".format(Path(model_root).resolve().as_posix()),
        "TEMPORAL_MEMORY.temporal_memory_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_init_model_path={}".format(checkpoint.as_posix()),
        "TEMPORAL_MEMORY.temporal_memory_init_sequence_length_warm_start_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_sequence_length=16",
        "TEMPORAL_MEMORY.temporal_memory_train_min_event_count_exclusive=200000",
        "TEMPORAL_MEMORY.temporal_memory_train_views_per_video=8",
        "TEMPORAL_MEMORY.temporal_memory_positive_frame_probability=0.75",
        "TEMPORAL_MEMORY.temporal_memory_target_positive_loss_mass=0.20",
        "TEMPORAL_MEMORY.temporal_memory_max_positive_weight=16.0",
        "TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0",
        "TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0",
        "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true",
        "TEMPORAL_MEMORY.temporal_memory_attention_projection_only_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_density_bucket_boundaries=[]",
        "TEMPORAL_MEMORY.temporal_memory_density_bucket_views=[]",
        "TEMPORAL_MEMORY.temporal_memory_sparse_target_support_sampling_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.50",
        "TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled={}".format(
            "true" if enabled else "false"
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_target_weight={:.6f}".format(
            float(candidate["metric_target_weight"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_component_weight={:.6f}".format(
            float(candidate["metric_component_weight"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_warmup_epochs={}".format(
            int(candidate["metric_warmup_epochs"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_spatial_cell_size={}".format(
            int(candidate["metric_spatial_cell_size"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_min_cell_events={}".format(
            int(candidate["metric_min_cell_events"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_component_ratio={:.6f}".format(
            float(candidate["metric_component_ratio"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_activation_threshold={:.6f}".format(
            float(candidate["metric_activation_threshold"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_metric_activation_temperature={:.6f}".format(
            float(candidate["metric_activation_temperature"])
        ),
        "TEMPORAL_MEMORY.temporal_memory_target_coverage_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_cache_all_videos=false",
        "TEMPORAL_MEMORY.temporal_memory_cache_video_count=2",
        "TEMPORAL_MEMORY.temporal_memory_train_workers=0",
        "TEMPORAL_MEMORY.temporal_memory_confidence_only_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_freeze_base_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_head_only_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_dacc_v2_enabled=false",
        "TEMPORAL_MEMORY.temporal_memory_dacc_v2_only_enabled=false",
        "TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true",
        "TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false",
        "TEMPORAL_FRAME.temporal_frame_confidence_head_enabled=false",
    ]


def training_argv(protocol, data_root, model_root, variant, epochs):
    overrides = training_overrides(protocol, data_root, model_root, variant, epochs)
    argv = [
        str(EVC_ROOT / "train_temporal_memory.py"),
        "--config",
        str(EVC_ROOT / "configs" / "evisseg_evuav.yaml"),
        "--set",
        *overrides,
    ]
    return argv, overrides


def formal_specs(protocol, views):
    specs = []
    for fold in protocol["dataset"]["folds"]:
        view = views["{}_fit".format(fold["fold_id"])]
        for variant in protocol["training"]["paired_variants"]:
            run_id = "{}_{}".format(fold["fold_id"], variant)
            output = FORMAL_ROOT / run_id
            specs.append(
                {
                    "run_id": run_id,
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "fit_groups": list(fold["fit_groups"]),
                    "held_group": fold["held_group"],
                    "data_root": view["root"],
                    "expected_source_names": [item["name"] for item in fit_items(protocol, fold)],
                    "expected_videos": int(fold["fit_video_count"]),
                    "expected_sequences_per_epoch": int(fold["fit_video_count"])
                    * int(protocol["training"]["views_per_video"]),
                    "expected_optimizer_steps": int(fold["expected_optimizer_steps_per_run"]),
                    "epochs": int(protocol["training"]["epochs"]),
                    "output_root": str(output.resolve()),
                    "model_root": str((output / "model").resolve()),
                    "result_path": str((output / "runtime_result.json").resolve()),
                }
            )
    return specs


def probe_specs(protocol, views):
    result = []
    probe = protocol["resource_probe"]
    for variant in probe["paired_variants"]:
        output = OUTPUT_ROOT / "resource_probe" / variant
        result.append(
            {
                "run_id": "probe_{}".format(variant),
                "fold_id": "probe",
                "variant": variant,
                "fit_groups": ["probe"],
                "held_group": None,
                "data_root": views["probe"]["root"],
                "expected_source_names": [probe["source"]],
                "expected_videos": 1,
                "expected_sequences_per_epoch": int(probe["views_per_video"]),
                "expected_optimizer_steps": int(probe["expected_optimizer_steps_per_run"]),
                "epochs": int(probe["epochs"]),
                "output_root": str(output.resolve()),
                "model_root": str((output / "model").resolve()),
                "result_path": str((output / "training_result.json").resolve()),
            }
        )
    return result


def paired_diff(protocol, specs, commands):
    output = []
    for fold in protocol["dataset"]["folds"]:
        pair = {
            spec["variant"]: spec
            for spec in specs
            if spec["fold_id"] == fold["fold_id"]
        }
        if set(pair) != {"baseline", "metric_aux"}:
            raise RuntimeError("Formal pair is incomplete for {}".format(fold["fold_id"]))
        baseline = override_mapping(commands[pair["baseline"]["run_id"]]["overrides"])
        candidate = override_mapping(commands[pair["metric_aux"]["run_id"]]["overrides"])
        differences = {
            key: {"baseline": baseline.get(key), "metric_aux": candidate.get(key)}
            for key in sorted(set(baseline) | set(candidate))
            if baseline.get(key) != candidate.get(key)
        }
        expected = set(protocol["training"]["paired_difference_allowlist"])
        if set(differences) != expected:
            raise RuntimeError(
                "Paired differences escaped frozen allowlist: {}".format(sorted(differences))
            )
        output.append({"fold_id": fold["fold_id"], "differences": differences})
    return output


def resolve_training_config(protocol, spec):
    argv, overrides = training_argv(
        protocol,
        spec["data_root"],
        spec["model_root"],
        spec["variant"],
        spec["epochs"],
    )
    marker = "METRIC_AUX_RESOLVED="
    resolver = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
from configs.configs import cfg
r = cfg.resolved_config
payload = {
    "data_root": r["DATA"]["root"],
    "model_root": r["TRAIN"]["model_save_root"],
    "seed": r["TRAIN"]["seed"],
    "epochs": r["TRAIN"]["epochs"],
    "batch_size": r["TRAIN"]["batch_size"],
    "metric_enabled": r["TEMPORAL_MEMORY"]["temporal_memory_metric_aux_enabled"],
    "metric_target_weight": r["TEMPORAL_MEMORY"]["temporal_memory_metric_target_weight"],
    "metric_component_weight": r["TEMPORAL_MEMORY"]["temporal_memory_metric_component_weight"],
    "metric_warmup": r["TEMPORAL_MEMORY"]["temporal_memory_metric_warmup_epochs"],
    "metric_threshold": r["TEMPORAL_MEMORY"]["temporal_memory_metric_activation_threshold"],
    "views": r["TEMPORAL_MEMORY"]["temporal_memory_train_views_per_video"],
    "dense": r["TEMPORAL_MEMORY"]["temporal_memory_dense_sampling_enabled"],
    "sequence_length": r["TEMPORAL_MEMORY"]["temporal_memory_sequence_length"],
    "min_events": r["TEMPORAL_MEMORY"]["temporal_memory_train_min_event_count_exclusive"],
    "confidence_only": r["TEMPORAL_MEMORY"]["temporal_memory_confidence_only_enabled"],
    "freeze_base": r["TEMPORAL_MEMORY"]["temporal_memory_freeze_base_enabled"],
    "head_only": r["TEMPORAL_MEMORY"]["temporal_memory_head_only_enabled"],
    "dacc_v2_only": r["TEMPORAL_MEMORY"]["temporal_memory_dacc_v2_only_enabled"],
    "attention_projection_only": r["TEMPORAL_MEMORY"]["temporal_memory_attention_projection_only_enabled"],
    "config_overrides": cfg.config_overrides,
}
print("METRIC_AUX_RESOLVED=" + json.dumps(payload, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", resolver, str(EVC_ROOT), *argv[1:]],
        cwd=str(EVC_ROOT),
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Clean-process config resolution failed:\n{}\n{}".format(
                completed.stdout, completed.stderr
            )
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("Clean-process resolver emitted no unique payload.")
    resolved = json.loads(lines[0][len(marker) :])
    actual_sources = sorted(
        path.name for path in (Path(spec["data_root"]) / "train").glob("*.npz")
    )
    candidate = protocol["training"]["candidate"]
    expected_enabled = spec["variant"] == "metric_aux"
    checks = {
        "data_root": Path(resolved["data_root"]).resolve() == Path(spec["data_root"]).resolve(),
        "model_root": Path(resolved["model_root"]).resolve() == Path(spec["model_root"]).resolve(),
        "seed": int(resolved["seed"]) == int(protocol["training"]["seed"]),
        "epochs": int(resolved["epochs"]) == int(spec["epochs"]),
        "batch_size": int(resolved["batch_size"]) == 1,
        "metric_enabled": bool(resolved["metric_enabled"]) == expected_enabled,
        "metric_target_weight": float(resolved["metric_target_weight"]) == float(candidate["metric_target_weight"]),
        "metric_component_weight": float(resolved["metric_component_weight"]) == float(candidate["metric_component_weight"]),
        "metric_warmup": int(resolved["metric_warmup"]) == 1,
        "metric_threshold": float(resolved["metric_threshold"]) == 0.719,
        "views": int(resolved["views"]) == 8,
        "dense_disabled": resolved["dense"] is False,
        "sequence_length": int(resolved["sequence_length"]) == 16,
        "min_events": int(resolved["min_events"]) == 200000,
        "all_narrow_scopes_false": not any(
            bool(resolved[key])
            for key in (
                "confidence_only",
                "freeze_base",
                "head_only",
                "dacc_v2_only",
                "attention_projection_only",
            )
        ),
        "overrides_exact": resolved["config_overrides"] == overrides,
        "fit_source_membership": actual_sources == sorted(spec["expected_source_names"]),
        "optimizer_steps": len(actual_sources) * int(resolved["views"]) * int(resolved["epochs"])
        == int(spec["expected_optimizer_steps"]),
    }
    if not all(checks.values()):
        raise RuntimeError("Resolved-config preflight failed: {}".format(checks))
    return {"resolved": resolved, "checks": checks, "passed": True}


def command_audit_payload(protocol, protocol_sha256, assets, views):
    specs = formal_specs(protocol, views)
    commands = {}
    resolved = {}
    for spec in specs:
        argv, overrides = training_argv(
            protocol,
            spec["data_root"],
            spec["model_root"],
            spec["variant"],
            spec["epochs"],
        )
        commands[spec["run_id"]] = {
            "argv": [sys.executable, *argv],
            "overrides": overrides,
            "fold_id": spec["fold_id"],
            "variant": spec["variant"],
            "expected_source_names": spec["expected_source_names"],
            "expected_optimizer_steps": spec["expected_optimizer_steps"],
        }
        resolved[spec["run_id"]] = resolve_training_config(protocol, spec)
    probes = {}
    for spec in probe_specs(protocol, views):
        argv, overrides = training_argv(
            protocol,
            spec["data_root"],
            spec["model_root"],
            spec["variant"],
            spec["epochs"],
        )
        probes[spec["run_id"]] = {
            "argv": [sys.executable, *argv],
            "overrides": overrides,
            "expected_optimizer_steps": spec["expected_optimizer_steps"],
            "resolved_preflight": resolve_training_config(protocol, spec),
        }
    return {
        "schema": "ev-uav-metric-aux-h2-grouped-oof-command-audit-v1",
        "created_utc": utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "assets": assets,
        "data_views": views,
        "formal_commands": commands,
        "formal_resolved_preflights": resolved,
        "probe_commands": probes,
        "paired_diff_audit": paired_diff(protocol, specs, commands),
        "formal_optimizer_steps_total": sum(
            int(spec["expected_optimizer_steps"]) for spec in specs
        ),
        "gpu_or_cuda_initialized": False,
        "data_use_statement": (
            "Only frozen official train_088..train_098 file identities were read; "
            "no NPZ arrays, validation/test paths, T32 caches or persistence-formal artifacts were opened."
        ),
    }


def run_audit():
    protocol, protocol_sha = load_protocol()
    assets = verify_assets(protocol)
    views = materialize_views(protocol)
    payload = command_audit_payload(protocol, protocol_sha, assets, views)
    _expect_equal(payload["formal_optimizer_steps_total"], 1056, "audit optimizer steps")
    write_new_json(COMMAND_AUDIT_PATH, payload)
    print("command audit:", COMMAND_AUDIT_PATH)
    print("command audit sha256:", sha256_file(COMMAND_AUDIT_PATH))
    print("GPU not initialized; waiting for explicit root authorization.")
    return payload


INSTRUMENTED_TRAIN_BOOTSTRAP = r'''
import atexit
import json
import os
from pathlib import Path
import runpy
import sys

repo = Path(sys.argv[1]).resolve()
stats_path = Path(sys.argv[2]).resolve()
steps_per_epoch = int(sys.argv[3])
epochs = int(sys.argv[4])
warmup = int(sys.argv[5])
target_weight = float(sys.argv[6])
component_weight = float(sys.argv[7])
train_argv = list(sys.argv[8:])
sys.path.insert(0, str(repo))

import utils.component_hard_negative as metric_module

original_target = metric_module.target_frame_activation_loss
original_component = metric_module.component_hard_negative_loss
records = {
    "schema": "ev-uav-metric-aux-loss-instrumentation-v1",
    "steps_per_epoch": steps_per_epoch,
    "epochs": {
        str(epoch): {
            "target": {"calls": 0, "raw_loss_sum": 0.0, "group_count": 0, "missed_group_count": 0},
            "component": {"calls": 0, "raw_loss_sum": 0.0, "candidate_cell_count": 0, "hard_cell_count": 0},
        }
        for epoch in range(epochs)
    },
}
target_calls = 0
component_calls = 0


def target_wrapper(*args, **kwargs):
    global target_calls
    result = original_target(*args, **kwargs)
    epoch = warmup + target_calls // steps_per_epoch
    target_calls += 1
    bucket = records["epochs"].setdefault(str(epoch), {"target": {}, "component": {}})["target"]
    bucket["calls"] = int(bucket.get("calls", 0)) + 1
    bucket["raw_loss_sum"] = float(bucket.get("raw_loss_sum", 0.0)) + float(result[0].detach().item())
    bucket["group_count"] = int(bucket.get("group_count", 0)) + int(result[1])
    bucket["missed_group_count"] = int(bucket.get("missed_group_count", 0)) + int(result[2])
    return result


def component_wrapper(*args, **kwargs):
    global component_calls
    result = original_component(*args, **kwargs)
    epoch = warmup + component_calls // steps_per_epoch
    component_calls += 1
    bucket = records["epochs"].setdefault(str(epoch), {"target": {}, "component": {}})["component"]
    bucket["calls"] = int(bucket.get("calls", 0)) + 1
    bucket["raw_loss_sum"] = float(bucket.get("raw_loss_sum", 0.0)) + float(result[0].detach().item())
    bucket["candidate_cell_count"] = int(bucket.get("candidate_cell_count", 0)) + int(result[1])
    bucket["hard_cell_count"] = int(bucket.get("hard_cell_count", 0)) + int(result[2])
    return result


metric_module.target_frame_activation_loss = target_wrapper
metric_module.component_hard_negative_loss = component_wrapper


def write_stats():
    for epoch in range(epochs):
        bucket = records["epochs"][str(epoch)]
        for name, weight in (("target", target_weight), ("component", component_weight)):
            calls = int(bucket[name].get("calls", 0))
            raw_sum = float(bucket[name].get("raw_loss_sum", 0.0))
            bucket[name]["raw_loss_mean"] = raw_sum / max(calls, 1)
            bucket[name]["weighted_loss_mean"] = weight * bucket[name]["raw_loss_mean"]
    records["total_target_calls"] = target_calls
    records["total_component_calls"] = component_calls
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(records, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, stats_path)


atexit.register(write_stats)
sys.argv = train_argv
runpy.run_path(str(repo / "train_temporal_memory.py"), run_name="__main__")
'''


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the CPU audit before any GPU command.")
    payload, digest = load_json_snapshot(COMMAND_AUDIT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-h2-grouped-oof-command-audit-v1":
        raise RuntimeError("Command-audit schema mismatch.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Command audit does not bind the frozen protocol.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Runner changed after the command audit.")
    if payload.get("gpu_or_cuda_initialized") is not False:
        raise RuntimeError("Command audit did not remain CPU-only.")
    return payload, digest


def require_gpu_authorization(authorized):
    if not authorized:
        raise PermissionError(
            "GPU work is coordination-gated. Re-run only after root authorization with {}.".format(
                GPU_AUTHORIZATION_FLAG
            )
        )


def require_idle_gpu():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=str(EVC_ROOT),
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi compute-process preflight failed closed.")
    snapshot = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    pid_names = {}
    if os.name == "nt":
        tasklist = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if tasklist.returncode != 0:
            raise RuntimeError("tasklist PID/name audit failed closed.")
        for row in csv.reader(tasklist.stdout.splitlines()):
            if len(row) < 2:
                continue
            try:
                pid_names[int(row[1])] = row[0]
            except ValueError:
                continue
    python_processes = []
    for line in snapshot:
        columns = [value.strip() for value in line.split(",")]
        if not columns:
            continue
        try:
            pid = int(columns[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        process_name = pid_names.get(pid, columns[1] if len(columns) > 1 else "")
        basename = process_name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if basename not in ("python.exe", "pythonw.exe", "python", "pythonw"):
            continue
        python_processes.append(
            {"pid": pid, "process_name": process_name, "nvidia_smi_record": line}
        )
    if python_processes:
        raise RuntimeError(
            "GPU has another Python compute process; refusing concurrent work: {}".format(
                python_processes
            )
        )
    return {
        "nvidia_smi_process_snapshot": snapshot,
        "other_python_compute_processes": python_processes,
        "wddm_graphics_processes_ignored": True,
        "idle_for_python_training": True,
    }


def source_hash_snapshot(protocol, names):
    index = source_index(protocol)
    root = official_train_root(protocol)
    output = {}
    for name in names:
        item = index[name]
        path = require_official_train_source(root / name, protocol, name)
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError("Train source changed: {}".format(path))
        output[name] = actual
    return output


def load_torch_checkpoint(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def recursive_bitwise_equal(left, right):
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(recursive_bitwise_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(recursive_bitwise_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return type(left) is type(right) and left == right


def model_state_bitwise_equal(left_checkpoint, right_checkpoint):
    return recursive_bitwise_equal(
        left_checkpoint["model_state_dict"], right_checkpoint["model_state_dict"]
    )


def checkpoint_scope_audit(protocol, checkpoint_path):
    checkpoint = load_torch_checkpoint(checkpoint_path)
    name_shape_audit = checkpoint_name_shape_audit(
        protocol, checkpoint_path, loaded_checkpoint=checkpoint
    )
    scope = checkpoint.get("provenance", {}).get("training_scope", {})
    frozen = protocol["training"]["training_scope_audit"]
    optimizer = checkpoint.get("optimizer_state_dict", {})
    optimizer_tensor_count = sum(
        len(group.get("params", [])) for group in optimizer.get("param_groups", [])
    )
    checks = {
        "scope_name": scope.get("name") == "all",
        "trainable_parameter_count": int(scope.get("trainable_parameter_count", -1))
        == int(frozen["trainable_parameter_count"]),
        "frozen_parameter_count": int(scope.get("frozen_parameter_count", -1))
        == int(frozen["frozen_parameter_count"]),
        "optimizer_parameter_tensor_count": optimizer_tensor_count
        == int(frozen["trainable_state_tensor_count"]),
    }
    if not all(checks.values()):
        raise RuntimeError("Full-scope checkpoint audit failed: {}".format(checks))
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "scope": scope,
        "optimizer_parameter_tensor_count": optimizer_tensor_count,
        "checks": checks,
        "name_shape_audit": name_shape_audit,
        "passed": True,
    }


def checkpoint_name_shape_audit(protocol, checkpoint_path, loaded_checkpoint=None):
    checkpoint = (
        loaded_checkpoint
        if loaded_checkpoint is not None
        else load_torch_checkpoint(checkpoint_path)
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint is missing model_state_dict.")
    records = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(state.items())
    ]
    digest = canonical_sha256(records)
    frozen = protocol["training"]["training_scope_audit"]
    checks = {
        "state_tensor_count": len(records) == int(frozen["trainable_state_tensor_count"]),
        "state_parameter_count": sum(record["numel"] for record in records)
        == int(frozen["trainable_parameter_count"]),
        "name_shape_canonical_sha256": digest
        == frozen["trainable_name_shape_canonical_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError("Checkpoint name/shape canonical audit failed: {}".format(checks))
    return {
        "state_tensor_count": len(records),
        "state_parameter_count": sum(record["numel"] for record in records),
        "name_shape_canonical_sha256": digest,
        "checks": checks,
        "passed": True,
    }


def model_architecture_scope_audit_cpu(protocol):
    import torch
    from model.temporal_memory_net import BidirectionalTemporalMemoryNet

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before clean CPU architecture audit.")
    model = BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=16,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=False,
        confidence_head_enabled=False,
        temporal_attention_enabled=True,
    ).cpu()
    items = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0])
        if parameter.requires_grad
    ]
    digest = canonical_sha256(items)
    frozen = protocol["training"]["training_scope_audit"]
    checks = {
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "trainable_tensor_count": len(items) == int(frozen["trainable_state_tensor_count"]),
        "trainable_parameter_count": sum(item["numel"] for item in items)
        == int(frozen["trainable_parameter_count"]),
        "canonical_sha256": digest == frozen["trainable_name_shape_canonical_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError("Clean CPU architecture scope audit failed: {}".format(checks))
    return {
        "trainable_tensor_count": len(items),
        "trainable_parameter_count": sum(item["numel"] for item in items),
        "canonical_sha256": digest,
        "checks": checks,
        "passed": True,
    }


def validate_auxiliary_stats(protocol, stats, variant, epochs, steps_per_epoch):
    candidate = protocol["training"]["candidate"]
    checks = {}
    for epoch in range(epochs):
        bucket = stats["epochs"][str(epoch)]
        target = bucket["target"]
        component = bucket["component"]
        if variant == "baseline" or epoch < int(candidate["metric_warmup_epochs"]):
            checks["epoch{}_target_zero".format(epoch)] = (
                int(target["calls"]) == 0
                and float(target["raw_loss_sum"]) == 0.0
                and float(target["weighted_loss_mean"]) == 0.0
            )
            checks["epoch{}_component_zero".format(epoch)] = (
                int(component["calls"]) == 0
                and float(component["raw_loss_sum"]) == 0.0
                and float(component["weighted_loss_mean"]) == 0.0
            )
        else:
            checks["epoch{}_target_calls".format(epoch)] = int(target["calls"]) == steps_per_epoch
            checks["epoch{}_component_calls".format(epoch)] = int(component["calls"]) == steps_per_epoch
            checks["epoch{}_target_loss_positive".format(epoch)] = (
                math.isfinite(float(target["raw_loss_mean"]))
                and float(target["raw_loss_mean"]) > 0.0
                and math.isfinite(float(target["weighted_loss_mean"]))
                and float(target["weighted_loss_mean"]) > 0.0
            )
            checks["epoch{}_component_loss_positive".format(epoch)] = (
                math.isfinite(float(component["raw_loss_mean"]))
                and float(component["raw_loss_mean"]) > 0.0
                and math.isfinite(float(component["weighted_loss_mean"]))
                and float(component["weighted_loss_mean"]) > 0.0
            )
            checks["epoch{}_target_groups_positive".format(epoch)] = int(target["group_count"]) > 0
            checks["epoch{}_candidate_cells_positive".format(epoch)] = int(component["candidate_cell_count"]) > 0
            checks["epoch{}_hard_cells_positive".format(epoch)] = int(component["hard_cell_count"]) > 0
    if not all(checks.values()):
        raise RuntimeError("Auxiliary-loss instrumentation failed: {}".format(checks))
    return {"checks": checks, "passed": True}


def _training_run_directory(model_root):
    candidates = sorted((Path(model_root) / "runs").glob("*"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise RuntimeError("Expected exactly one generated training run directory.")
    return candidates[0]


def run_training_spec(protocol, spec):
    gpu_idle_preflight = require_idle_gpu()
    output_root = Path(spec["output_root"])
    result_path = Path(spec["result_path"])
    model_root = Path(spec["model_root"])
    if output_root.exists() or result_path.exists() or model_root.exists():
        raise FileExistsError("Refusing to overwrite training attempt: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=False)
    preflight = resolve_training_config(protocol, spec)
    before_sources = source_hash_snapshot(protocol, spec["expected_source_names"])
    before_core = {
        item["path"]: sha256_file(EVC_ROOT / item["path"])
        for item in protocol["repository"]["core_files"]
    }
    argv, overrides = training_argv(
        protocol,
        spec["data_root"],
        spec["model_root"],
        spec["variant"],
        spec["epochs"],
    )
    stats_path = output_root / "auxiliary_loss_instrumentation.json"
    log_path = output_root / "training.log"
    command = [
        sys.executable,
        "-c",
        INSTRUMENTED_TRAIN_BOOTSTRAP,
        str(EVC_ROOT),
        str(stats_path),
        str(spec["expected_sequences_per_epoch"]),
        str(spec["epochs"]),
        str(protocol["training"]["candidate"]["metric_warmup_epochs"]),
        str(protocol["training"]["candidate"]["metric_target_weight"]),
        str(protocol["training"]["candidate"]["metric_component_weight"]),
        *argv,
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_stream:
        completed = subprocess.run(
            command,
            cwd=str(EVC_ROOT),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Training subprocess failed for {}; inspect {}".format(spec["run_id"], log_path)
        )
    gpu_idle_postflight = require_idle_gpu()
    if not stats_path.is_file():
        raise RuntimeError("Training instrumentation output is missing.")
    stats, stats_sha = load_json_snapshot(stats_path)
    stats_audit = validate_auxiliary_stats(
        protocol,
        stats,
        spec["variant"],
        int(spec["epochs"]),
        int(spec["expected_sequences_per_epoch"]),
    )
    run_dir = _training_run_directory(model_root)
    summary_path = run_dir / "run_summary.json"
    summary, summary_sha = load_json_snapshot(summary_path)
    if summary.get("config_overrides") != overrides:
        raise RuntimeError("Training summary overrides differ from frozen command.")
    if int(summary["sampling"]["video_count"]) != int(spec["expected_videos"]):
        raise RuntimeError("Training summary video count differs.")
    if int(summary["sampling"]["sequence_count"]) != int(spec["expected_sequences_per_epoch"]):
        raise RuntimeError("Training summary sequence count differs.")

    checkpoints = {}
    for epoch in range(1, int(spec["epochs"]) + 1):
        path = run_dir / "epoch_{:03d}_seed{}.pt".format(
            epoch, protocol["training"]["seed"]
        )
        if not path.is_file():
            raise FileNotFoundError("Required epoch checkpoint is missing: {}".format(path))
        checkpoints["e{}".format(epoch)] = checkpoint_scope_audit(protocol, path)
    after_sources = source_hash_snapshot(protocol, spec["expected_source_names"])
    after_core = {
        item["path"]: sha256_file(EVC_ROOT / item["path"])
        for item in protocol["repository"]["core_files"]
    }
    if before_sources != after_sources or before_core != after_core:
        raise RuntimeError("Frozen train inputs or core code changed during training.")
    payload = {
        "schema": "ev-uav-metric-aux-training-result-v1",
        "created_utc": utc_now(),
        "status": "completed",
        "run_id": spec["run_id"],
        "fold_id": spec["fold_id"],
        "variant": spec["variant"],
        "fit_groups": spec["fit_groups"],
        "held_group": spec["held_group"],
        "expected_source_names": spec["expected_source_names"],
        "expected_optimizer_steps": int(spec["expected_optimizer_steps"]),
        "command": command,
        "training_argv": [sys.executable, *argv],
        "overrides": overrides,
        "resolved_preflight": preflight,
        "run_directory": str(run_dir.resolve()),
        "run_summary": str(summary_path.resolve()),
        "run_summary_sha256": summary_sha,
        "training_log": str(log_path.resolve()),
        "training_log_sha256": sha256_file(log_path),
        "auxiliary_loss_instrumentation": str(stats_path.resolve()),
        "auxiliary_loss_instrumentation_sha256": stats_sha,
        "auxiliary_loss_stats": stats,
        "auxiliary_loss_audit": stats_audit,
        "checkpoints": checkpoints,
        "input_source_sha256_before_after_equal": True,
        "core_sha256_before_after_equal": True,
        "elapsed_seconds": time.time() - started,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
        "gpu_idle_preflight": gpu_idle_preflight,
        "gpu_idle_postflight": gpu_idle_postflight,
    }
    write_new_json(result_path, payload)
    return payload


def load_training_result(spec):
    path = Path(spec["result_path"])
    payload, digest = load_json_snapshot(path)
    if payload.get("schema") != "ev-uav-metric-aux-training-result-v1":
        raise RuntimeError("Training-result schema mismatch: {}".format(path))
    if payload.get("status") != "completed" or payload.get("run_id") != spec["run_id"]:
        raise RuntimeError("Training result is not a completed matching run.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Training result protocol identity mismatch.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Training result runner identity mismatch.")
    for checkpoint in payload["checkpoints"].values():
        path_value = Path(checkpoint["path"])
        if not path_value.is_file() or sha256_file(path_value) != checkpoint["sha256"]:
            raise RuntimeError("Training checkpoint identity mismatch: {}".format(path_value))
    return payload, digest


def compare_pair_checkpoints(baseline_result, candidate_result):
    baseline_e1 = load_torch_checkpoint(baseline_result["checkpoints"]["e1"]["path"])
    candidate_e1 = load_torch_checkpoint(candidate_result["checkpoints"]["e1"]["path"])
    baseline_e2 = load_torch_checkpoint(baseline_result["checkpoints"]["e2"]["path"])
    candidate_e2 = load_torch_checkpoint(candidate_result["checkpoints"]["e2"]["path"])
    model_equal = model_state_bitwise_equal(baseline_e1, candidate_e1)
    state_sections = {
        key: recursive_bitwise_equal(baseline_e1[key], candidate_e1[key])
        for key in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
        )
    }
    e2_diverged = not model_state_bitwise_equal(baseline_e2, candidate_e2)
    checks = {
        "e1_model_state_bitwise_equal": model_equal,
        "e1_optimizer_state_bitwise_equal": state_sections["optimizer_state_dict"],
        "e1_scheduler_state_bitwise_equal": state_sections["scheduler_state_dict"],
        "e1_rng_state_bitwise_equal": state_sections["rng_state"],
        "e2_model_state_diverged": e2_diverged,
    }
    if not all(checks.values()):
        raise RuntimeError("Paired checkpoint identity/divergence audit failed: {}".format(checks))
    return {"checks": checks, "passed": True}


def synthetic_metric_gradient_probe(protocol, device_name="cpu"):
    import torch
    from utils.component_hard_negative import (
        component_hard_negative_loss,
        target_frame_activation_loss,
    )

    device = torch.device(device_name)
    logits = torch.nn.Parameter(torch.zeros(10, device=device, dtype=torch.float32))
    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], device=device, dtype=torch.float32)
    target_ids = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], device=device, dtype=torch.int64)
    locations = torch.tensor(
        [
            [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 1, 3],
            [0, 21, 21, 1], [0, 21, 21, 2], [0, 21, 21, 3],
            [0, 30, 30, 1], [0, 30, 30, 2], [0, 30, 30, 3], [0, 30, 30, 4],
        ],
        device=device,
        dtype=torch.int64,
    )
    candidate = protocol["training"]["candidate"]
    scores = torch.sigmoid(logits)
    target_loss, target_groups, missed_groups = target_frame_activation_loss(
        scores,
        labels,
        target_ids,
        locations,
        50,
        candidate["metric_activation_threshold"],
        candidate["metric_activation_temperature"],
    )
    component_loss, candidate_cells, hard_cells = component_hard_negative_loss(
        scores,
        labels,
        locations,
        candidate["metric_spatial_cell_size"],
        50,
        candidate["metric_min_cell_events"],
        candidate["metric_component_ratio"],
        candidate["metric_activation_threshold"],
        candidate["metric_activation_temperature"],
    )
    target_gradient = torch.autograd.grad(target_loss, logits, retain_graph=True)[0]
    component_gradient = torch.autograd.grad(component_loss, logits, retain_graph=True)[0]
    combined = (
        candidate["metric_target_weight"] * target_loss
        + candidate["metric_component_weight"] * component_loss
    )
    combined_gradient = torch.autograd.grad(combined, logits)[0]
    target_norm = float(torch.linalg.vector_norm(target_gradient).detach().cpu().item())
    component_norm = float(torch.linalg.vector_norm(component_gradient).detach().cpu().item())
    combined_norm = float(torch.linalg.vector_norm(combined_gradient).detach().cpu().item())
    before = logits.detach().clone()
    optimizer = torch.optim.SGD([logits], lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    logits.grad = combined_gradient.detach().clone()
    optimizer.step()
    update_norm = float(torch.linalg.vector_norm(logits.detach() - before).cpu().item())
    checks = {
        "target_loss_finite_positive": math.isfinite(float(target_loss.detach().cpu().item()))
        and float(target_loss.detach().cpu().item()) > 0.0,
        "component_loss_finite_positive": math.isfinite(float(component_loss.detach().cpu().item()))
        and float(component_loss.detach().cpu().item()) > 0.0,
        "target_group_count_positive": int(target_groups) > 0,
        "candidate_cell_count_positive": int(candidate_cells) > 0,
        "hard_cell_count_positive": int(hard_cells) > 0,
        "target_gradient_norm_finite_positive": math.isfinite(target_norm) and target_norm > 0.0,
        "component_gradient_norm_finite_positive": math.isfinite(component_norm) and component_norm > 0.0,
        "combined_gradient_norm_finite_positive": math.isfinite(combined_norm) and combined_norm > 0.0,
        "fresh_zero_state_optimizer_update_nonzero": math.isfinite(update_norm) and update_norm > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError("Synthetic metric-gradient probe failed: {}".format(checks))
    return {
        "device": str(device),
        "target_loss": float(target_loss.detach().cpu().item()),
        "component_loss": float(component_loss.detach().cpu().item()),
        "target_group_count": int(target_groups),
        "missed_target_group_count": int(missed_groups),
        "candidate_cell_count": int(candidate_cells),
        "hard_cell_count": int(hard_cells),
        "target_gradient_norm": target_norm,
        "component_gradient_norm": component_norm,
        "combined_gradient_norm": combined_norm,
        "fresh_optimizer_update_norm": update_norm,
        "checks": checks,
        "passed": True,
    }


def real_batch_metric_gradient_probe(protocol, candidate_result, probe_data_root):
    import numpy as np
    import torch

    sys.path.insert(0, str(EVC_ROOT))
    from dataset.temporal_memory import TemporalMemoryTrainDataset
    from utils.component_hard_negative import (
        component_hard_negative_loss,
        target_frame_activation_loss,
    )
    from utils.temporal_memory_inference import load_temporal_memory_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the real-batch gradient probe.")
    device = torch.device("cuda:0")
    torch.manual_seed(int(protocol["training"]["seed"]))
    torch.cuda.manual_seed_all(int(protocol["training"]["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint_path = Path(candidate_result["checkpoints"]["e1"]["path"])
    if sha256_file(checkpoint_path) != candidate_result["checkpoints"]["e1"]["sha256"]:
        raise RuntimeError("Candidate E1 checkpoint changed before real-batch probe.")
    source_name = protocol["resource_probe"]["source"]
    source_path = (Path(probe_data_root) / "train" / source_name).resolve()
    official_source = require_official_train_source(
        official_train_root(protocol) / source_name, protocol, source_name
    )
    if source_path.parent != (Path(probe_data_root) / "train").resolve():
        raise RuntimeError("Probe data-view source escaped its isolated train root.")
    if not source_path.is_file() or not os.path.samefile(source_path, official_source):
        raise RuntimeError("Probe data-view source is not the frozen official hard link.")
    source_sha_before = sha256_file(source_path)
    if source_sha_before != source_index(protocol)[source_name]["sha256"]:
        raise RuntimeError("Real-batch probe source identity mismatch.")

    dataset = TemporalMemoryTrainDataset(
        root=Path(probe_data_root) / "train",
        whole_t=8000,
        temporal_bin_size=50,
        context_bins=5,
        sequence_length=16,
        width=128,
        height=128,
        views_per_video=8,
        positive_frame_probability=0.75,
        random_seed=int(protocol["training"]["seed"]),
        log_count_clip=4.0,
        cache_all_videos=False,
        cache_video_count=2,
        dense_sampling_enabled=False,
        dense_event_count_cutoff=200000,
        dense_view_multiplier=2,
        density_bucket_boundaries=[],
        density_bucket_views=[],
        min_event_count_exclusive=200000,
        sparse_target_support_sampling_enabled=False,
        sparse_target_support_max_events=3,
        sparse_target_support_probability=0.75,
    )
    if len(dataset) != 8:
        raise RuntimeError("Real-batch probe dataset must contain eight views.")
    dataset.set_epoch(1)
    model, _ = load_temporal_memory_model(
        checkpoint_path,
        device,
        context_bins=5,
        width=16,
        sequence_length=16,
    )
    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable_parameters) != 89 or sum(p.numel() for p in trainable_parameters) != 1924716:
        raise RuntimeError("Real-batch probe model does not match frozen full scope.")
    candidate = protocol["training"]["candidate"]
    chosen = None
    for view_index in range(len(dataset)):
        sample = dataset[view_index]
        frames = torch.from_numpy(sample["frames"]).float().to(device).unsqueeze(0)
        time_indices = torch.from_numpy(sample["event_time_indices"]).long().to(device)
        event_x = torch.from_numpy(sample["event_x"]).long().to(device)
        event_y = torch.from_numpy(sample["event_y"]).long().to(device)
        labels = torch.from_numpy(sample["labels"]).float().to(device)
        target_ids = torch.from_numpy(sample["target_ids"]).long().to(device)
        logit_maps = model(frames).squeeze(0)
        event_logits = logit_maps[time_indices, 0, event_y, event_x]
        scores = torch.sigmoid(event_logits)
        locations = torch.stack(
            (
                torch.zeros_like(event_x),
                event_x,
                event_y,
                time_indices * 50 + 1,
            ),
            dim=1,
        )
        target_loss, target_groups, missed_groups = target_frame_activation_loss(
            scores,
            labels,
            target_ids,
            locations,
            50,
            candidate["metric_activation_threshold"],
            candidate["metric_activation_temperature"],
        )
        component_loss, candidate_cells, hard_cells = component_hard_negative_loss(
            scores,
            labels,
            locations,
            candidate["metric_spatial_cell_size"],
            50,
            candidate["metric_min_cell_events"],
            candidate["metric_component_ratio"],
            candidate["metric_activation_threshold"],
            candidate["metric_activation_temperature"],
        )
        if int(target_groups) > 0 and int(candidate_cells) > 0 and int(hard_cells) > 0:
            chosen = {
                "view_index": view_index,
                "event_count": int(labels.numel()),
                "target_loss": target_loss,
                "component_loss": component_loss,
                "target_groups": int(target_groups),
                "missed_groups": int(missed_groups),
                "candidate_cells": int(candidate_cells),
                "hard_cells": int(hard_cells),
            }
            break
        del frames, time_indices, event_x, event_y, labels, target_ids, logit_maps, event_logits
        torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("No deterministic epoch-1 probe view exercised both auxiliary losses.")

    target_gradients = torch.autograd.grad(
        chosen["target_loss"], trainable_parameters, retain_graph=True, allow_unused=True
    )
    component_gradients = torch.autograd.grad(
        chosen["component_loss"], trainable_parameters, retain_graph=True, allow_unused=True
    )

    def gradient_norm(gradients):
        squared = torch.zeros((), device=device)
        tensor_count = 0
        finite = True
        for gradient in gradients:
            if gradient is None:
                continue
            tensor_count += 1
            finite = finite and bool(torch.isfinite(gradient).all().item())
            squared = squared + torch.sum(gradient.detach() * gradient.detach())
        return float(torch.sqrt(squared).cpu().item()), tensor_count, finite

    target_norm, target_grad_tensors, target_finite = gradient_norm(target_gradients)
    component_norm, component_grad_tensors, component_finite = gradient_norm(component_gradients)
    before = [parameter.detach().clone() for parameter in trainable_parameters]
    optimizer = torch.optim.SGD(trainable_parameters, lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    combined = (
        float(candidate["metric_target_weight"]) * chosen["target_loss"]
        + float(candidate["metric_component_weight"]) * chosen["component_loss"]
    )
    combined.backward()
    optimizer.step()
    update_squared = 0.0
    update_tensor_count = 0
    for prior, parameter in zip(before, trainable_parameters):
        difference = parameter.detach() - prior
        if torch.count_nonzero(difference).item() > 0:
            update_tensor_count += 1
        update_squared += float(torch.sum(difference * difference).cpu().item())
    update_norm = math.sqrt(update_squared)
    source_sha_after = sha256_file(source_path)
    checks = {
        "source_before_after_equal": source_sha_before == source_sha_after,
        "target_loss_finite_positive": math.isfinite(float(chosen["target_loss"].detach().cpu().item()))
        and float(chosen["target_loss"].detach().cpu().item()) > 0.0,
        "component_loss_finite_positive": math.isfinite(float(chosen["component_loss"].detach().cpu().item()))
        and float(chosen["component_loss"].detach().cpu().item()) > 0.0,
        "target_groups_positive": chosen["target_groups"] > 0,
        "candidate_cells_positive": chosen["candidate_cells"] > 0,
        "hard_cells_positive": chosen["hard_cells"] > 0,
        "target_parameter_gradient_finite_positive": target_finite and target_norm > 0.0,
        "component_parameter_gradient_finite_positive": component_finite and component_norm > 0.0,
        "target_gradient_reaches_parameters": target_grad_tensors > 0,
        "component_gradient_reaches_parameters": component_grad_tensors > 0,
        "fresh_zero_state_optimizer_update_nonzero": update_norm > 0.0 and update_tensor_count > 0,
    }
    if not all(checks.values()):
        raise RuntimeError("Real-batch metric-gradient probe failed: {}".format(checks))
    torch.cuda.synchronize()
    return {
        "source_name": source_name,
        "source_sha256": source_sha_before,
        "epoch": 1,
        "view_index": chosen["view_index"],
        "event_count": chosen["event_count"],
        "target_loss": float(chosen["target_loss"].detach().cpu().item()),
        "component_loss": float(chosen["component_loss"].detach().cpu().item()),
        "target_group_count": chosen["target_groups"],
        "missed_target_group_count": chosen["missed_groups"],
        "candidate_cell_count": chosen["candidate_cells"],
        "hard_cell_count": chosen["hard_cells"],
        "target_parameter_gradient_norm": target_norm,
        "target_parameter_gradient_tensor_count": target_grad_tensors,
        "component_parameter_gradient_norm": component_norm,
        "component_parameter_gradient_tensor_count": component_grad_tensors,
        "fresh_optimizer_update_norm": update_norm,
        "fresh_optimizer_updated_tensor_count": update_tensor_count,
        "checks": checks,
        "passed": True,
    }


def run_probe(authorized=False):
    require_gpu_authorization(authorized)
    require_idle_gpu()
    protocol, _ = load_protocol()
    command_audit, command_audit_sha = load_command_audit()
    views = command_audit["data_views"]
    if PROBE_RESULT_PATH.exists():
        raise FileExistsError("Refusing to overwrite completed probe receipt.")
    results = {}
    for spec in probe_specs(protocol, views):
        print("starting fresh paired probe:", spec["run_id"], flush=True)
        results[spec["variant"]] = run_training_spec(protocol, spec)
    pair = compare_pair_checkpoints(results["baseline"], results["metric_aux"])
    synthetic = synthetic_metric_gradient_probe(protocol, device_name="cpu")
    real_batch = real_batch_metric_gradient_probe(
        protocol, results["metric_aux"], views["probe"]["root"]
    )
    candidate_stats = results["metric_aux"]["auxiliary_loss_stats"]["epochs"]
    checks = {
        "candidate_epoch0_target_exact_zero": float(candidate_stats["0"]["target"]["raw_loss_sum"]) == 0.0,
        "candidate_epoch0_component_exact_zero": float(candidate_stats["0"]["component"]["raw_loss_sum"]) == 0.0,
        "candidate_epoch1_target_positive": float(candidate_stats["1"]["target"]["raw_loss_mean"]) > 0.0,
        "candidate_epoch1_component_positive": float(candidate_stats["1"]["component"]["raw_loss_mean"]) > 0.0,
        "candidate_epoch1_target_groups_positive": int(candidate_stats["1"]["target"]["group_count"]) > 0,
        "candidate_epoch1_candidate_cells_positive": int(candidate_stats["1"]["component"]["candidate_cell_count"]) > 0,
        "candidate_epoch1_hard_cells_positive": int(candidate_stats["1"]["component"]["hard_cell_count"]) > 0,
        "paired_checkpoint_audit": pair["passed"],
        "synthetic_autograd_and_fresh_optimizer": synthetic["passed"],
        "real_epoch1_batch_parameter_gradients_and_fresh_optimizer": real_batch["passed"],
    }
    payload = {
        "schema": "ev-uav-metric-aux-resource-probe-result-v1",
        "created_utc": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed": all(checks.values()),
        "paired_training_results": {
            key: {
                "path": value["result_path"] if "result_path" in value else str(
                    Path(next(spec["result_path"] for spec in probe_specs(protocol, views) if spec["variant"] == key))
                ),
                "checkpoint_sha256": value["checkpoints"]["e2"]["sha256"],
            }
            for key, value in results.items()
        },
        "paired_checkpoint_audit": pair,
        "synthetic_metric_gradient_probe": synthetic,
        "real_epoch1_batch_metric_gradient_probe": real_batch,
        "command_audit_sha256": command_audit_sha,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_parent_pretraining_exposure": True,
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
    }
    write_new_json(PROBE_RESULT_PATH, payload)
    if not payload["passed"]:
        raise RuntimeError("Resource probe failed: {}".format(checks))
    print("resource probe passed:", PROBE_RESULT_PATH)
    return payload


def require_probe_passed():
    payload, digest = load_json_snapshot(PROBE_RESULT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-resource-probe-result-v1":
        raise RuntimeError("Resource-probe schema mismatch.")
    if payload.get("passed") is not True or not all(payload.get("checks", {}).values()):
        raise RuntimeError("Resource probe has not passed every frozen gate.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Resource probe protocol identity mismatch.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Resource probe runner identity mismatch.")
    return payload, digest


def formal_pair_audit(protocol, specs, write=True):
    fold_records = []
    for fold in protocol["dataset"]["folds"]:
        pair_specs = {
            spec["variant"]: spec
            for spec in specs
            if spec["fold_id"] == fold["fold_id"]
        }
        if set(pair_specs) != {"baseline", "metric_aux"}:
            raise RuntimeError("Formal pair specification is incomplete.")
        baseline, baseline_sha = load_training_result(pair_specs["baseline"])
        candidate, candidate_sha = load_training_result(pair_specs["metric_aux"])
        pair_check = compare_pair_checkpoints(baseline, candidate)
        candidate_stats = candidate["auxiliary_loss_stats"]["epochs"]
        active_epochs = []
        for epoch in range(int(protocol["training"]["epochs"])):
            target_calls = int(candidate_stats[str(epoch)]["target"]["calls"])
            component_calls = int(candidate_stats[str(epoch)]["component"]["calls"])
            if target_calls > 0 or component_calls > 0:
                active_epochs.append(epoch)
        checks = {
            "paired_checkpoint_gate": pair_check["passed"],
            "candidate_active_epochs_exact": active_epochs
            == protocol["training"]["candidate"]["active_zero_based_epochs"],
            "baseline_aux_audit_passed": baseline["auxiliary_loss_audit"]["passed"],
            "candidate_aux_audit_passed": candidate["auxiliary_loss_audit"]["passed"],
            "fit_members_exact": sorted(baseline["expected_source_names"])
            == sorted(candidate["expected_source_names"])
            == sorted(pair_specs["baseline"]["expected_source_names"]),
        }
        if not all(checks.values()):
            raise RuntimeError("Formal paired audit failed: {}".format(checks))
        fold_records.append(
            {
                "fold_id": fold["fold_id"],
                "baseline_result_sha256": baseline_sha,
                "candidate_result_sha256": candidate_sha,
                "pair_checkpoint_audit": pair_check,
                "candidate_active_zero_based_epochs": active_epochs,
                "candidate_auxiliary_loss_stats": candidate_stats,
                "checks": checks,
                "passed": True,
            }
        )
    payload = {
        "schema": "ev-uav-metric-aux-formal-pair-audit-v1",
        "created_utc": utc_now(),
        "fold_records": fold_records,
        "checks": {
            "all_three_pairs_present": len(fold_records) == 3,
            "all_pairs_passed": all(record["passed"] for record in fold_records),
        },
        "passed": len(fold_records) == 3 and all(record["passed"] for record in fold_records),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_parent_pretraining_exposure": True,
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
    }
    if write:
        write_new_json(PAIR_AUDIT_PATH, payload)
    return payload


def require_formal_pair_audit():
    payload, digest = load_json_snapshot(PAIR_AUDIT_PATH)
    if payload.get("schema") != "ev-uav-metric-aux-formal-pair-audit-v1":
        raise RuntimeError("Formal pair-audit schema mismatch.")
    if payload.get("passed") is not True or not all(payload["checks"].values()):
        raise RuntimeError("Formal pair audit has not passed.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Formal pair audit protocol identity mismatch.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Formal pair audit runner identity mismatch.")
    return payload, digest


def run_formal_training(run_id=None, authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    specs = formal_specs(protocol, command_audit["data_views"])
    if run_id is not None:
        matches = [spec for spec in specs if spec["run_id"] == run_id]
        if len(matches) != 1:
            raise KeyError("Unknown formal run id: {}".format(run_id))
        return [run_training_spec(protocol, matches[0])]

    results = []
    for spec in specs:
        result_path = Path(spec["result_path"])
        if result_path.is_file():
            result, _ = load_training_result(spec)
            print("retaining completed formal training:", spec["run_id"], flush=True)
            results.append(result)
            continue
        print("launching fresh formal process:", spec["run_id"], flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--run-id",
                spec["run_id"],
                GPU_AUTHORIZATION_FLAG,
            ],
            cwd=str(EVC_ROOT),
            check=True,
        )
        result, _ = load_training_result(spec)
        results.append(result)
    if PAIR_AUDIT_PATH.exists():
        raise FileExistsError("Refusing to overwrite formal pair audit.")
    pair_payload = formal_pair_audit(protocol, specs, write=True)
    if not pair_payload["passed"]:
        raise RuntimeError("Formal paired training did not pass frozen audits.")
    return results


def evaluation_specs(protocol, training_specs):
    by_fold_variant = {
        (spec["fold_id"], spec["variant"]): spec for spec in training_specs
    }
    output = []
    for fold in protocol["dataset"]["folds"]:
        held_names = [item["name"] for item in held_items(protocol, fold)]
        for variant in ("released_m20", "baseline", "metric_aux"):
            eval_id = "{}_{}".format(fold["fold_id"], variant)
            if variant == "released_m20":
                checkpoint = workspace_path(
                    protocol["parent_checkpoint"]["workspace_relative_path"]
                )
                checkpoint_sha = protocol["parent_checkpoint"]["sha256"]
                training_result_path = None
            else:
                train_spec = by_fold_variant[(fold["fold_id"], variant)]
                training_result, _ = load_training_result(train_spec)
                checkpoint = Path(training_result["checkpoints"]["e3"]["path"])
                checkpoint_sha = training_result["checkpoints"]["e3"]["sha256"]
                training_result_path = train_spec["result_path"]
            result_path = EVALUATION_ROOT / eval_id / "evaluation.json"
            output.append(
                {
                    "eval_id": eval_id,
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "held_group": fold["held_group"],
                    "held_source_names": held_names,
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "training_result_path": training_result_path,
                    "result_path": str(result_path.resolve()),
                }
            )
    return output


def evaluate_spec(protocol, spec):
    import numpy as np
    import torch

    sys.path.insert(0, str(EVC_ROOT))
    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay
    from crossfit_component_reranker import (
        SufficientCounts,
        metrics_from_counts,
        sufficient_counts_for_video,
    )
    from train_component_reranker import _load_train_source
    from utils.postprocess import ChallengePostprocessor
    from utils.temporal_memory_inference import (
        load_temporal_memory_model,
        predict_temporal_memory_scores,
        temporal_frame_video_from_sample,
    )

    gpu_idle_preflight = require_idle_gpu()
    result_path = Path(spec["result_path"])
    if result_path.exists() or result_path.parent.exists():
        raise FileExistsError("Refusing to overwrite held-train evaluation: {}".format(result_path))
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise RuntimeError("Evaluation checkpoint identity mismatch.")
    overrides = list(protocol["evaluation"]["fixed_config_overrides"])
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml", overrides
    )
    c00_contract = component_crossfit.validate_c00_config(
        cfg, float(protocol["evaluation"]["prediction_threshold"])
    )
    actual_c00_sha = component_crossfit.sha256_json(c00_contract)
    if actual_c00_sha != protocol["evaluation"]["effective_c00_canonical_sha256"]:
        raise RuntimeError("Effective C00 canonical SHA-256 mismatch.")
    if int(cfg.temporal_memory_sequence_length) != 16:
        raise RuntimeError("T32 or non-T16 evaluation is forbidden.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for held-train full-stream inference.")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint_payload = load_temporal_memory_model(
        checkpoint,
        device,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        cfg.temporal_memory_sequence_length,
    )
    checkpoint_metadata = {
        "epoch": int(checkpoint_payload.get("epoch", -1)),
        "next_epoch": int(checkpoint_payload.get("next_epoch", -1)),
        "temporal_memory": checkpoint_payload.get("temporal_memory", {}),
        "training_scope": checkpoint_payload.get("provenance", {}).get(
            "training_scope", {}
        ),
    }
    prediction_threshold = float(protocol["evaluation"]["prediction_threshold"])
    index = source_index(protocol)
    records = []
    pooled = SufficientCounts()
    started = time.time()
    for position, name in enumerate(spec["held_source_names"], start=1):
        frozen = index[name]
        source_path = require_official_train_source(
            official_train_root(protocol) / name, protocol, name
        )
        before_sha = sha256_file(source_path)
        if before_sha != frozen["sha256"]:
            raise RuntimeError("Held train source identity mismatch before load.")
        sample, labels, target_ids = _load_train_source(source_path)
        after_load_sha = sha256_file(source_path)
        if after_load_sha != before_sha:
            raise RuntimeError("Held train source changed while loading.")
        frame_video = temporal_frame_video_from_sample(
            sample, cfg.temporal_memory_bin_size, cfg.whole_t
        )
        with torch.no_grad():
            raw_tensor = predict_temporal_memory_scores(
                model,
                frame_video,
                device,
                cfg.temporal_memory_context_bins,
                cfg.res[0],
                cfg.res[1],
                cfg.temporal_memory_inference_batch_size,
                cfg.temporal_memory_log_count_clip,
            ).reshape(-1).detach().cpu().to(torch.float32)
        if int(raw_tensor.numel()) != int(len(labels)):
            raise RuntimeError("Full-stream prediction length differs from held source.")
        locations = np.column_stack(
            (
                np.zeros(len(sample["ev_loc"]), dtype=np.int64),
                sample["ev_loc"].astype(np.int64, copy=False),
            )
        )
        location_tensor = torch.from_numpy(locations).to(torch.int64).contiguous()
        processor = ChallengePostprocessor.from_cfg(
            cfg, prediction_threshold, event_count=int(len(labels))
        )
        processed_tensor, postprocess_stats = processor.apply(
            raw_tensor.clone(), location_tensor
        )
        counts = sufficient_counts_for_video(
            processed_tensor.numpy(),
            labels,
            target_ids,
            locations,
            prediction_threshold=prediction_threshold,
        )
        pooled = pooled + counts
        records.append(
            {
                "source_name": name,
                "source_sha256": frozen["sha256"],
                "event_count": int(len(labels)),
                "raw_scores_float32_sha256": sha256_float32(raw_tensor.numpy()),
                "processed_scores_float32_sha256": sha256_float32(
                    processed_tensor.numpy()
                ),
                "postprocess_stats": asdict(postprocess_stats),
                "counts": counts.to_dict(),
                "metrics": metrics_from_counts(counts),
            }
        )
        final_sha = sha256_file(source_path)
        if final_sha != before_sha:
            raise RuntimeError("Held train source changed during evaluation.")
        print(
            "held train {}/{}: {}".format(
                position, len(spec["held_source_names"]), name
            ),
            flush=True,
        )
        del sample, labels, target_ids, frame_video, raw_tensor, processed_tensor
        torch.cuda.empty_cache()
    if [record["source_name"] for record in records] != spec["held_source_names"]:
        raise RuntimeError("Held-record order differs from the frozen fold.")
    torch.cuda.synchronize()
    gpu_python_process_postflight = require_idle_gpu()
    payload = {
        "schema": "ev-uav-metric-aux-held-train-evaluation-v1",
        "created_utc": utc_now(),
        "eval_id": spec["eval_id"],
        "fold_id": spec["fold_id"],
        "variant": spec["variant"],
        "held_group": spec["held_group"],
        "dataset_split": "train",
        "held_stream": "complete_full_stream_t160",
        "t32_read_or_combined": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_metadata": checkpoint_metadata,
        "prediction_threshold": prediction_threshold,
        "config_overrides": overrides,
        "effective_c00_contract": c00_contract,
        "effective_c00_canonical_sha256": actual_c00_sha,
        "records": records,
        "pooled_counts": pooled.to_dict(),
        "pooled_metrics": metrics_from_counts(pooled),
        "elapsed_seconds": time.time() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_parent_pretraining_exposure": True,
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
        "gpu_idle_preflight": gpu_idle_preflight,
        "gpu_python_process_postflight": gpu_python_process_postflight,
    }
    write_new_json(result_path, payload)
    return payload


def load_evaluation_result(spec):
    path = Path(spec["result_path"])
    payload, digest = load_json_snapshot(path)
    if payload.get("schema") != "ev-uav-metric-aux-held-train-evaluation-v1":
        raise RuntimeError("Held-train evaluation schema mismatch.")
    if payload.get("eval_id") != spec["eval_id"] or payload.get("dataset_split") != "train":
        raise RuntimeError("Held-train evaluation identity/split mismatch.")
    if payload.get("t32_read_or_combined") is not False:
        raise RuntimeError("T32 was read or combined.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Evaluation protocol identity mismatch.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Evaluation runner identity mismatch.")
    if [record["source_name"] for record in payload["records"]] != spec["held_source_names"]:
        raise RuntimeError("Evaluation held sources differ from protocol.")
    return payload, digest


def run_formal_evaluation(eval_id=None, authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    require_formal_pair_audit()
    training_specs = formal_specs(protocol, command_audit["data_views"])
    specs = evaluation_specs(protocol, training_specs)
    if eval_id is not None:
        matches = [spec for spec in specs if spec["eval_id"] == eval_id]
        if len(matches) != 1:
            raise KeyError("Unknown evaluation id: {}".format(eval_id))
        return [evaluate_spec(protocol, matches[0])]
    results = []
    for spec in specs:
        if Path(spec["result_path"]).is_file():
            result, _ = load_evaluation_result(spec)
            print("retaining completed held-train evaluation:", spec["eval_id"])
            results.append(result)
            continue
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "evaluate",
                "--eval-id",
                spec["eval_id"],
                GPU_AUTHORIZATION_FLAG,
            ],
            cwd=str(EVC_ROOT),
            check=True,
        )
        result, _ = load_evaluation_result(spec)
        results.append(result)
    return results


COUNT_FIELDS = (
    "true_positive_events",
    "false_positive_events",
    "false_negative_events",
    "correct_objects",
    "object_count",
    "false_components",
    "frame_count",
    "event_count",
)


def add_count_dicts(values):
    output = {key: 0 for key in COUNT_FIELDS}
    for value in values:
        if set(value) != set(COUNT_FIELDS):
            raise RuntimeError("Sufficient-count fields differ from frozen evaluator.")
        for key in COUNT_FIELDS:
            output[key] += int(value[key])
    return output


def population_invariants(counts):
    return {
        "positive_events": int(counts["true_positive_events"])
        + int(counts["false_negative_events"]),
        "object_count": int(counts["object_count"]),
        "frame_count": int(counts["frame_count"]),
        "event_count": int(counts["event_count"]),
    }


def metric_delta(candidate, comparator):
    return {
        key: float(candidate[key]) - float(comparator[key])
        for key in ("score", "pd", "fa", "iou", "acc", "score_fa")
    }


def comparator_gates(candidate_counts, candidate_metrics, comparator_counts, comparator_metrics, pooled):
    invariants_equal = population_invariants(candidate_counts) == population_invariants(
        comparator_counts
    )
    if pooled:
        checks = {
            "score_delta_at_least_0p0002": float(candidate_metrics["score"])
            - float(comparator_metrics["score"])
            >= 0.0002,
            "false_positive_events_strictly_lower": int(
                candidate_counts["false_positive_events"]
            )
            < int(comparator_counts["false_positive_events"]),
            "false_components_strictly_lower": int(candidate_counts["false_components"])
            < int(comparator_counts["false_components"]),
            "population_invariants_equal": invariants_equal,
        }
    else:
        checks = {
            "true_positive_events_not_lower": int(candidate_counts["true_positive_events"])
            >= int(comparator_counts["true_positive_events"]),
            "correct_objects_not_lower": int(candidate_counts["correct_objects"])
            >= int(comparator_counts["correct_objects"]),
            "pd_not_lower": float(candidate_metrics["pd"]) >= float(comparator_metrics["pd"]),
            "iou_not_lower": float(candidate_metrics["iou"]) >= float(comparator_metrics["iou"]),
            "fa_not_higher": float(candidate_metrics["fa"]) <= float(comparator_metrics["fa"]),
            "score_not_lower": float(candidate_metrics["score"])
            >= float(comparator_metrics["score"]),
            "population_invariants_equal": invariants_equal,
        }
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_report_gates(protocol, fold_results, pooled):
    if len(fold_results) != 3:
        raise RuntimeError("Promotion gates require exactly three held folds.")
    expected_folds = {fold["fold_id"] for fold in protocol["dataset"]["folds"]}
    if {fold["fold_id"] for fold in fold_results} != expected_folds:
        raise RuntimeError("Promotion fold identities differ from protocol.")
    comparator_names = ("baseline", "released_m20")
    fold_checks = {}
    for fold in fold_results:
        fold_checks[fold["fold_id"]] = {}
        for comparator in comparator_names:
            fold_checks[fold["fold_id"]][comparator] = comparator_gates(
                fold["metric_aux"]["counts"],
                fold["metric_aux"]["metrics"],
                fold[comparator]["counts"],
                fold[comparator]["metrics"],
                pooled=False,
            )
    pooled_checks = {
        comparator: comparator_gates(
            pooled["metric_aux"]["counts"],
            pooled["metric_aux"]["metrics"],
            pooled[comparator]["counts"],
            pooled[comparator]["metrics"],
            pooled=True,
        )
        for comparator in comparator_names
    }
    all_fold_passed = all(
        result["passed"]
        for fold in fold_checks.values()
        for result in fold.values()
    )
    all_pooled_passed = all(result["passed"] for result in pooled_checks.values())
    return {
        "fold_checks": fold_checks,
        "pooled_checks": pooled_checks,
        "checks": {
            "every_fold_against_both_comparators": all_fold_passed,
            "pooled_against_both_comparators": all_pooled_passed,
            "single_candidate_no_grid": protocol["training"]["no_parameter_grid"] is True,
            "shared_parent_claim_limited": protocol["audit_amendment"]["claim_scope"]
            == "incremental_finetune_transfer_not_fold_clean_model_generalization",
        },
        "passed": all_fold_passed
        and all_pooled_passed
        and protocol["training"]["no_parameter_grid"] is True,
    }


def build_report(protocol, protocol_sha, evaluation_specs_value):
    import torch

    sys.path.insert(0, str(EVC_ROOT))
    from crossfit_component_reranker import SufficientCounts, metrics_from_counts

    if torch.cuda.is_initialized():
        raise RuntimeError("Report command must start in a CUDA-uninitialized process.")
    evaluations = {}
    evaluation_artifacts = {}
    for spec in evaluation_specs_value:
        payload, digest = load_evaluation_result(spec)
        evaluations[(spec["fold_id"], spec["variant"])] = payload
        evaluation_artifacts[spec["eval_id"]] = {
            "path": spec["result_path"],
            "sha256": digest,
        }
    folds = []
    pooled_counts_by_variant = {
        variant: [] for variant in ("released_m20", "baseline", "metric_aux")
    }
    held_seen_by_variant = {
        variant: [] for variant in ("released_m20", "baseline", "metric_aux")
    }
    for frozen_fold in protocol["dataset"]["folds"]:
        fold_id = frozen_fold["fold_id"]
        expected_names = [item["name"] for item in held_items(protocol, frozen_fold)]
        result = {"fold_id": fold_id, "held_group": frozen_fold["held_group"]}
        for variant in ("released_m20", "baseline", "metric_aux"):
            payload = evaluations[(fold_id, variant)]
            actual_names = [record["source_name"] for record in payload["records"]]
            if actual_names != expected_names:
                raise RuntimeError("Held evaluation sources differ from frozen fold.")
            held_seen_by_variant[variant].extend(actual_names)
            counts = payload["pooled_counts"]
            metrics = payload["pooled_metrics"]
            pooled_counts_by_variant[variant].append(counts)
            result[variant] = {"counts": counts, "metrics": metrics}
        result["delta_vs_baseline"] = metric_delta(
            result["metric_aux"]["metrics"], result["baseline"]["metrics"]
        )
        result["delta_vs_released_m20"] = metric_delta(
            result["metric_aux"]["metrics"], result["released_m20"]["metrics"]
        )
        folds.append(result)

    frozen_union = set(source_index(protocol))
    held_pool_checks = {}
    for variant, names in held_seen_by_variant.items():
        held_pool_checks[variant] = (
            len(names) == 11 and len(set(names)) == 11 and set(names) == frozen_union
        )
    if not all(held_pool_checks.values()):
        raise RuntimeError("Pooled records are not the exact held-only source union.")

    pooled = {}
    for variant in ("released_m20", "baseline", "metric_aux"):
        counts_dict = add_count_dicts(pooled_counts_by_variant[variant])
        counts_object = SufficientCounts(**counts_dict)
        pooled[variant] = {
            "counts": counts_dict,
            "metrics": metrics_from_counts(counts_object),
        }
    pooled["metric_aux"]["delta_vs_baseline"] = metric_delta(
        pooled["metric_aux"]["metrics"], pooled["baseline"]["metrics"]
    )
    pooled["metric_aux"]["delta_vs_released_m20"] = metric_delta(
        pooled["metric_aux"]["metrics"], pooled["released_m20"]["metrics"]
    )
    gates = evaluate_report_gates(protocol, folds, pooled)
    command_audit, command_audit_sha = load_command_audit()
    pair_audit, pair_audit_sha = require_formal_pair_audit()
    payload = {
        "schema": "ev-uav-metric-aux-h2-grouped-oof-report-v1",
        "created_utc": utc_now(),
        "status": "passed" if gates["passed"] else "failed",
        "passed": gates["passed"],
        "evidence_class": protocol["evidence_class"],
        "shared_parent_pretraining_exposure": True,
        "claim_scope": protocol["audit_amendment"]["claim_scope"],
        "claim_limitation": (
            "Released M20 was pretrained with exposure to all eleven H2 sources. "
            "These folds estimate only incremental E3 metric-aux fine-tune transfer "
            "to sources omitted from that fold's E3 fine-tune; they are not fold-clean "
            "generalization estimates for the complete model history."
        ),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "protocol_revision_history": protocol["revision_history"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "command_audit_sha256": command_audit_sha,
        "formal_pair_audit_sha256": pair_audit_sha,
        "fold_results": folds,
        "pooled": pooled,
        "held_only_pool_checks": held_pool_checks,
        "promotion_gates": gates,
        "evaluation_artifacts": evaluation_artifacts,
        "t32_read_or_combined": False,
        "decision": (
            "eligible_for_separate_validation_protocol_only"
            if gates["passed"]
            else protocol["promotion_gates"]["failure_action"]
        ),
        "no_default_submission_or_validation_change": True,
        "data_use_statement": (
            "Only frozen train_088..train_098 labels were used for E3 incremental-transfer "
            "evaluation. No validation/test file, T32 cache, persistence formal artifact, "
            "leaderboard result or platform submission was read."
        ),
    }
    write_new_json(REPORT_PATH, payload)
    return payload


def run_report():
    protocol, protocol_sha = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    require_formal_pair_audit()
    training_specs = formal_specs(protocol, command_audit["data_views"])
    specs = evaluation_specs(protocol, training_specs)
    payload = build_report(protocol, protocol_sha, specs)
    print("grouped OOF report:", REPORT_PATH)
    print("promotion passed:", payload["passed"])
    return payload


def run_formal_pair_audit_command():
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    specs = formal_specs(protocol, command_audit["data_views"])
    payload = formal_pair_audit(protocol, specs, write=True)
    print("formal pair audit:", PAIR_AUDIT_PATH)
    print("passed:", payload["passed"])
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="CPU-only asset, source-view and command audit.")
    probe = subparsers.add_parser("probe", help="Run the paired two-epoch GPU probe.")
    probe.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    train = subparsers.add_parser("train", help="Run all or one formal paired E3 training.")
    train.add_argument("--run-id", default=None)
    train.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser(
        "audit-training", help="CPU-audit all completed formal training pairs."
    )
    evaluate = subparsers.add_parser(
        "evaluate", help="Run all or one held-train full-stream evaluation."
    )
    evaluate.add_argument("--eval-id", default=None)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("report", help="Pool held-only counts and apply frozen gates.")
    all_after_probe = subparsers.add_parser(
        "all-after-probe", help="Run formal training, evaluation and report after a passed probe."
    )
    all_after_probe.add_argument(
        GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit()
    if args.command == "probe":
        return run_probe(args.authorized)
    if args.command == "train":
        return run_formal_training(args.run_id, args.authorized)
    if args.command == "audit-training":
        return run_formal_pair_audit_command()
    if args.command == "evaluate":
        return run_formal_evaluation(args.eval_id, args.authorized)
    if args.command == "report":
        return run_report()
    if args.command == "all-after-probe":
        require_gpu_authorization(args.authorized)
        run_formal_training(authorized=True)
        run_formal_evaluation(authorized=True)
        return run_report()
    raise RuntimeError("Unsupported command: {}".format(args.command))


if __name__ == "__main__":
    main()

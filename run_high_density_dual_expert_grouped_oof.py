"""Train-only grouped held-block runner for the high-density dual expert.

No command in this runner reads validation or test data.  ``audit`` is CPU
only.  GPU commands require an explicit coordination flag and a completed
resource probe.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
PROTOCOL_PATH = (
    EVC_ROOT
    / "protocols"
    / "high_density_dual_expert_grouped_oof_science_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "a4f3ffaaf5a887fade2f47e08d32d5a716b31d2251bdbb0abcba9ae9b4d47d42"
)
OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_high_density_dual_expert_grouped_oof_v1"
)
VIEW_ROOT = OUTPUT_ROOT / "fit_views"
TRAIN_ROOT = OUTPUT_ROOT / "paired_training"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
PROBE_PATH = OUTPUT_ROOT / "resource_probe.json"
PAIR_AUDIT_PATH = OUTPUT_ROOT / "formal_pair_audit.json"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--root-authorized-gpu"

ARMS = ("baseline", "candidate")
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_float32(array):
    import numpy as np

    value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_new_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("Refusing stale temporary output {}".format(temporary))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_snapshot(path):
    path = Path(path)
    before = sha256_file(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("JSON changed during read: {}".format(path))
    return value, before


def workspace_path(relative):
    value = (WORKSPACE_ROOT / relative).resolve()
    value.relative_to(WORKSPACE_ROOT.resolve())
    return value


def load_protocol():
    protocol, digest = load_json_snapshot(PROTOCOL_PATH)
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Science protocol SHA-256 differs from frozen constant.")
    if protocol.get("schema") != "ev-uav-high-density-dual-expert-grouped-oof-science-v1":
        raise RuntimeError("Science protocol schema mismatch.")
    if protocol.get("status") != "frozen_before_any_gpu_probe_training_or_held_evaluation":
        raise RuntimeError("Science protocol was not frozen before GPU work.")
    validate_protocol(protocol)
    return protocol, digest


def source_groups(protocol):
    return protocol["dataset"]["source_groups"]


def source_index(protocol):
    output = {}
    for group, items in source_groups(protocol).items():
        for item in items:
            if item["name"] in output:
                raise RuntimeError("Duplicate source in protocol: {}".format(item["name"]))
            output[item["name"]] = {**item, "group": group}
    return output


def items_for_groups(protocol, names):
    output = []
    for name in names:
        output.extend(source_groups(protocol)[name])
    return output


def fit_items(protocol, fold):
    return items_for_groups(protocol, fold["fit_groups"])


def held_items(protocol, fold):
    return list(source_groups(protocol)[fold["held_group"]])


def observable_route(event_count, polarity_minority_fraction):
    event_count = int(event_count)
    minority = float(polarity_minority_fraction)
    if not math.isfinite(minority) or not 0.0 <= minority <= 0.5:
        raise ValueError("polarity_minority_fraction must be finite and in [0,.5].")
    if event_count <= 200000:
        return "released_m20"
    return "h1" if minority < 0.20 else "h2"


def mode_for_arm(protocol, domain, arm):
    if arm == "baseline":
        return protocol["paired_design"]["baseline_mode"]
    if arm == "candidate":
        return protocol["paired_design"]["candidate_mode_by_domain"][domain]
    raise KeyError("Unknown arm: {}".format(arm))


def validate_protocol(protocol):
    if protocol["split_access"]["validation_read_allowed"] is not False:
        raise RuntimeError("Validation access must be forbidden.")
    if protocol["split_access"]["test_read_allowed"] is not False:
        raise RuntimeError("Test access must be forbidden.")
    if protocol["claim_limitations"]["candidate_count"] != 1:
        raise RuntimeError("Exactly one architecture candidate is allowed.")
    if protocol["claim_limitations"]["architecture_or_hyperparameter_grid_allowed"] is not False:
        raise RuntimeError("Architecture/hyperparameter grids must be forbidden.")
    architecture = protocol["architecture"]
    if (
        architecture["trainable_state_tensor_count"] != 14
        or architecture["trainable_parameter_count"] != 1712
        or architecture["trainable_name_shape_canonical_sha256"]
        != "0488383ced93885bdeaaf32f0bc3e28b90e5ba14d0ca8952dd37aa8de1beb855"
    ):
        raise RuntimeError("Frozen expert parameter scope mismatch.")
    training = protocol["training"]
    if not (
        training["seed"] == 49
        and training["epochs"] == 2
        and training["views_per_video"] == 4
        and training["sequence_length"] == 16
        and float(training["learning_rate"]) == 0.0005
    ):
        raise RuntimeError("Frozen training hyperparameters differ.")
    folds = protocol["dataset"]["folds"]
    if len(folds) != 5:
        raise RuntimeError("Expected exactly two H1 and three H2 folds.")
    if sum(fold["domain"] == "h1" for fold in folds) != 2:
        raise RuntimeError("Expected exactly two H1 folds.")
    if sum(fold["domain"] == "h2" for fold in folds) != 3:
        raise RuntimeError("Expected exactly three H2 folds.")
    seen_by_domain = {"h1": [], "h2": []}
    total_steps = 0
    for fold in folds:
        fit = fit_items(protocol, fold)
        held = held_items(protocol, fold)
        fit_names = {item["name"] for item in fit}
        held_names = {item["name"] for item in held}
        if fit_names & held_names:
            raise RuntimeError("Fit/held leakage in {}".format(fold["fold_id"]))
        if len(fit) != fold["fit_video_count"] or len(held) != fold["held_video_count"]:
            raise RuntimeError("Fold video count mismatch in {}".format(fold["fold_id"]))
        expected_steps = len(fit) * training["views_per_video"] * training["epochs"]
        if expected_steps != fold["optimizer_steps_per_arm"]:
            raise RuntimeError("Optimizer-step arithmetic mismatch.")
        total_steps += expected_steps * 2
        seen_by_domain[fold["domain"]].extend(sorted(held_names))
    index = source_index(protocol)
    for domain in ("h1", "h2"):
        expected = sorted(name for name, item in index.items() if item["group"].startswith(domain))
        if sorted(seen_by_domain[domain]) != expected or len(set(seen_by_domain[domain])) != len(expected):
            raise RuntimeError("Held blocks do not partition domain {}.".format(domain))
    if total_steps != training["expected_optimizer_steps_total"] or total_steps != 416:
        raise RuntimeError("Total paired optimizer-step budget mismatch.")
    probe = protocol["resource_probe"]
    if set(probe["domains"]) != {"h1", "h2"}:
        raise RuntimeError("Resource probe must cover exactly H1 and H2.")
    if not (
        probe["domains"]["h1"]["source"] == "train_045.npz"
        and probe["domains"]["h1"]["steps_per_arm"] == 4
        and probe["domains"]["h2"]["source"] == "train_096.npz"
        and probe["domains"]["h2"]["steps_per_arm"] == 2
        and probe["total_optimizer_steps"] == 12
    ):
        raise RuntimeError("Frozen dual-domain resource probe differs.")
    for item in index.values():
        routed = observable_route(item["event_count"], item["polarity_minority_fraction"])
        expected = "h1" if item["group"].startswith("h1") else "h2"
        if routed != expected:
            raise RuntimeError("Input-only route differs from frozen train domain.")
    gates = protocol["promotion_gates"]
    if float(gates["h2_held_pooled_score_delta_minimum_against_each_comparator"]) != 0.02:
        raise RuntimeError("H2 material-improvement gate must remain +0.02.")


def official_train_root(protocol):
    path = workspace_path(protocol["dataset"]["workspace_relative_train_root"])
    if not path.is_dir() or path.name.lower() != "train":
        raise RuntimeError("Official train root is missing or misnamed.")
    return path


def verify_assets(protocol):
    assets = {}
    for item in protocol["repository"]["bound_code"]:
        path = (EVC_ROOT / item["path"]).resolve()
        path.relative_to(EVC_ROOT)
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError("Bound code changed: {}".format(path))
        assets[item["path"]] = {"path": str(path), "sha256": actual}
    parent = workspace_path(protocol["parent_checkpoint"]["workspace_relative_path"])
    if sha256_file(parent) != protocol["parent_checkpoint"]["sha256"]:
        raise RuntimeError("Released M20 checkpoint identity mismatch.")
    assets["released_m20"] = {"path": str(parent), "sha256": sha256_file(parent)}
    cache_root = workspace_path(protocol["released_m20_train_cache"]["workspace_relative_path"])
    manifest = cache_root / "manifest.json"
    if sha256_file(manifest) != protocol["released_m20_train_cache"]["manifest_sha256"]:
        raise RuntimeError("Released M20 train-cache manifest mismatch.")
    assets["released_m20_train_cache"] = {
        "path": str(cache_root),
        "manifest_sha256": sha256_file(manifest),
    }
    train_root = official_train_root(protocol)
    for name, item in source_index(protocol).items():
        path = train_root / name
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise RuntimeError("Train source size/path mismatch: {}".format(name))
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError("Train source SHA mismatch: {}".format(name))
        assets[name] = {"path": str(path), "sha256": actual, "size": path.stat().st_size}
    return assets


def _verify_or_create_view(path, items, source_root):
    path = Path(path)
    expected = {item["name"]: item for item in items}
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        for name, item in expected.items():
            source = source_root / name
            destination = path / name
            os.link(str(source), str(destination))
    actual_names = sorted(item.name for item in path.glob("*.npz"))
    if actual_names != sorted(expected):
        raise RuntimeError("Materialized fit-view membership mismatch: {}".format(path))
    records = []
    for name in actual_names:
        destination = path / name
        if sha256_file(destination) != expected[name]["sha256"]:
            raise RuntimeError("Materialized fit-view SHA mismatch: {}".format(destination))
        records.append(
            {
                "name": name,
                "path": str(destination.resolve()),
                "sha256": expected[name]["sha256"],
                "same_file": os.path.samefile(str(source_root / name), str(destination)),
            }
        )
        if not records[-1]["same_file"]:
            raise RuntimeError("Fit view must be an immutable hard link.")
    return {"root": str(path.resolve()), "records": records}


def materialize_views(protocol):
    root = official_train_root(protocol)
    views = {}
    for fold in protocol["dataset"]["folds"]:
        views[fold["fold_id"]] = _verify_or_create_view(
            VIEW_ROOT / fold["fold_id"], fit_items(protocol, fold), root
        )
    for domain, probe in protocol["resource_probe"]["domains"].items():
        probe_item = source_index(protocol)[probe["source"]]
        view_id = "resource_probe_{}".format(domain)
        views[view_id] = _verify_or_create_view(
            VIEW_ROOT / view_id, [probe_item], root
        )
    return views


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def cpu_model_audit(protocol):
    import torch

    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay
    from model.high_density_polarity_expert import (
        HighDensityPolarityExpertMemoryNet,
        build_expert_model_from_m20,
        configure_expert_only_training,
    )
    from utils.temporal_memory_inference import load_temporal_memory_model

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit must start before CUDA initialization.")
    parent_path = workspace_path(protocol["parent_checkpoint"]["workspace_relative_path"])
    modes = ("activity_control", "h1_saturation", "h2_polarity")
    models = []
    states = []
    for mode in modes:
        torch.manual_seed(protocol["training"]["seed"])
        model, _ = build_expert_model_from_m20(parent_path, mode, device="cpu")
        models.append(model.eval())
        states.append(model.state_dict())
    if any(set(states[0]) != set(state) for state in states[1:]):
        raise RuntimeError("Paired model state keys differ.")
    for name in states[0]:
        if not all(torch.equal(states[0][name], state[name]) for state in states[1:]):
            raise RuntimeError("Paired initial tensor differs: {}".format(name))
    base, _ = load_temporal_memory_model(parent_path, "cpu", 5, 16, 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4917)
    frames = torch.rand((1, 2, 10, 32, 32), generator=generator)
    with torch.no_grad():
        parent_output = base(frames)
        outputs = [
            model(
                frames,
                expert_frames=(frames * 0.5 if mode == "h1_saturation" else None),
            )
            for mode, model in zip(modes, models)
        ]
    if not all(torch.equal(parent_output, output) for output in outputs):
        raise RuntimeError("Zero-initialized expert is not bitwise M20 identity.")
    trainable_names = configure_expert_only_training(models[0])
    items = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in sorted(models[0].named_parameters())
        if parameter.requires_grad
    ]
    architecture = protocol["architecture"]
    if (
        len(items) != architecture["trainable_state_tensor_count"]
        or sum(item["numel"] for item in items) != architecture["trainable_parameter_count"]
        or canonical_sha256(items) != architecture["trainable_name_shape_canonical_sha256"]
    ):
        raise RuntimeError("Actual expert-only scope differs from protocol.")
    expected_trainable = tuple(
        sorted(name for name, _ in models[0].named_parameters() if name.startswith("high_density_expert."))
    )
    if trainable_names != expected_trainable:
        raise RuntimeError("Trainable-name set differs from expert module.")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU model audit initialized CUDA.")
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml",
        list(protocol["evaluation"]["fixed_config_overrides"]),
    )
    c00 = component_crossfit.validate_c00_config(
        cfg, float(protocol["evaluation"]["prediction_threshold"])
    )
    c00_sha = component_crossfit.sha256_json(c00)
    if c00_sha != protocol["evaluation"]["effective_c00_canonical_sha256"]:
        raise RuntimeError("CPU preflight effective C00 SHA mismatch.")
    return {
        "modes": list(modes),
        "paired_initial_state_sha256": tensor_state_sha256(states[0]),
        "all_mode_states_bitwise_equal": True,
        "all_mode_outputs_bitwise_equal_to_m20": True,
        "trainable_tensor_count": len(items),
        "trainable_parameter_count": sum(item["numel"] for item in items),
        "trainable_name_shape_canonical_sha256": canonical_sha256(items),
        "effective_c00_canonical_sha256": c00_sha,
        "cuda_initialized": False,
    }


def formal_specs(protocol, views):
    output = []
    for fold in protocol["dataset"]["folds"]:
        for arm in ARMS:
            run_id = "{}__{}".format(fold["fold_id"], arm)
            run_root = TRAIN_ROOT / run_id
            output.append(
                {
                    "run_id": run_id,
                    "fold_id": fold["fold_id"],
                    "domain": fold["domain"],
                    "arm": arm,
                    "input_mode": mode_for_arm(protocol, fold["domain"], arm),
                    "fit_names": [item["name"] for item in fit_items(protocol, fold)],
                    "held_names": [item["name"] for item in held_items(protocol, fold)],
                    "view_root": views[fold["fold_id"]]["root"],
                    "expected_optimizer_steps": fold["optimizer_steps_per_arm"],
                    "root": str(run_root.resolve()),
                    "result_path": str((run_root / "runtime_result.json").resolve()),
                    "checkpoint_path": str((run_root / protocol["training"]["selection_checkpoint"]).resolve()),
                }
            )
    return output


def build_command_audit(protocol, protocol_sha, assets, views, model_audit):
    specs = formal_specs(protocol, views)
    commands = []
    for spec in specs:
        commands.append(
            {
                **spec,
                "command": [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "train",
                    "--run-id",
                    spec["run_id"],
                    GPU_AUTHORIZATION_FLAG,
                ],
            }
        )
    probe_specs = []
    for domain, frozen in protocol["resource_probe"]["domains"].items():
        for arm, mode_key in (
            ("baseline", "baseline_mode"),
            ("candidate", "candidate_mode"),
        ):
            probe_specs.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "source": frozen["source"],
                    "input_mode": frozen[mode_key],
                    "optimizer_steps": frozen["steps_per_arm"],
                    "view_root": views["resource_probe_{}".format(domain)]["root"],
                }
            )
    if sum(item["optimizer_steps"] for item in probe_specs) != protocol["resource_probe"]["total_optimizer_steps"]:
        raise RuntimeError("Resource-probe optimizer-step budget mismatch.")
    return {
        "schema": "ev-uav-high-density-dual-expert-command-audit-v1",
        "created_utc": utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "assets": assets,
        "views": views,
        "cpu_model_audit": model_audit,
        "formal_specs": commands,
        "resource_probe_specs": probe_specs,
        "paired_step_total": sum(spec["expected_optimizer_steps"] for spec in specs),
        "gpu_or_cuda_initialized": False,
        "forbidden_split_reads": ["validation", "test"],
    }


def run_audit():
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Audit must start before CUDA initialization.")
    protocol, protocol_sha = load_protocol()
    assets = verify_assets(protocol)
    views = materialize_views(protocol)
    model_audit = cpu_model_audit(protocol)
    payload = build_command_audit(protocol, protocol_sha, assets, views, model_audit)
    if payload["paired_step_total"] != protocol["training"]["expected_optimizer_steps_total"]:
        raise RuntimeError("Command audit optimizer-step total differs.")
    if torch.cuda.is_initialized():
        raise RuntimeError("Audit initialized CUDA.")
    write_new_json(COMMAND_AUDIT_PATH, payload)
    return payload


def load_command_audit():
    payload, digest = load_json_snapshot(COMMAND_AUDIT_PATH)
    if payload.get("schema") != "ev-uav-high-density-dual-expert-command-audit-v1":
        raise RuntimeError("Command audit schema mismatch.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Command audit protocol mismatch.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Runner changed after command audit.")
    if payload.get("gpu_or_cuda_initialized") is not False:
        raise RuntimeError("Command audit was not CPU-only.")
    return payload, digest


def require_gpu_authorization(authorized):
    if not authorized:
        raise PermissionError(
            "GPU work is coordination-gated; root must authorize {}.".format(
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
        raise RuntimeError("nvidia-smi process audit failed closed.")
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
    active = []
    for line in snapshot:
        columns = [part.strip() for part in line.split(",")]
        try:
            pid = int(columns[0])
        except (ValueError, IndexError):
            continue
        if pid == os.getpid():
            continue
        name = pid_names.get(pid, columns[1] if len(columns) > 1 else "")
        basename = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if basename in ("python", "python.exe", "pythonw", "pythonw.exe"):
            active.append({"pid": pid, "name": name, "record": line})
    if active:
        raise RuntimeError("Another Python process owns the GPU: {}".format(active))
    return {"snapshot": snapshot, "other_python_compute_processes": active, "passed": True}


def setup_seed(seed):
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ParallelClipTrainDataset:
    """Add a clip-8 frame stack without changing the frozen clip-4 dataset."""

    def __init__(self, base, enabled):
        self.base = base
        self.enabled = bool(enabled)
        self.file_paths = base.file_paths

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __getitem__(self, index):
        import numpy as np

        from dataset.temporal_frame import build_temporal_context_frame
        from dataset.temporal_memory import temporal_sequence_start

        sample = self.base[index]
        if not self.enabled:
            return sample
        index = int(index)
        video_index = int(
            np.searchsorted(self.base.view_offsets, index, side="right") - 1
        )
        view_index = int(index - self.base.view_offsets[video_index])
        video = self.base._video(video_index)
        center_bin = self.base._sample_center_bin(video_index, view_index, video)
        start_bin = temporal_sequence_start(
            center_bin,
            len(video.event_indices_by_bin),
            self.base.sequence_length,
        )
        expert_frames = np.stack(
            [
                build_temporal_context_frame(
                    video,
                    temporal_bin,
                    self.base.context_bins,
                    self.base.width,
                    self.base.height,
                    8.0,
                )
                for temporal_bin in range(
                    start_bin, start_bin + self.base.sequence_length
                )
            ],
            axis=0,
        )
        reconstructed_clip4 = np.stack(
            [
                build_temporal_context_frame(
                    video,
                    temporal_bin,
                    self.base.context_bins,
                    self.base.width,
                    self.base.height,
                    4.0,
                )
                for temporal_bin in range(
                    start_bin, start_bin + self.base.sequence_length
                )
            ],
            axis=0,
        )
        if not np.array_equal(sample["frames"], reconstructed_clip4):
            raise RuntimeError(
                "Parallel clip-8 sample is not aligned with released clip-4 frames."
            )
        saturated = reconstructed_clip4 >= 1.0
        recovered = saturated & (expert_frames > 0.0) & (expert_frames < 1.0)
        parallel_clip_audit = {
            "clip4_reconstruction_bitwise_equal": True,
            "shape_aligned": expert_frames.shape == reconstructed_clip4.shape,
            "clip4_nonzero_cells": int(np.count_nonzero(reconstructed_clip4)),
            "clip8_nonzero_cells": int(np.count_nonzero(expert_frames)),
            "clip4_saturated_cells": int(np.count_nonzero(saturated)),
            "clip8_recovered_dynamic_cells": int(np.count_nonzero(recovered)),
            "clip4_max": float(reconstructed_clip4.max()),
            "clip8_max": float(expert_frames.max()),
            "start_bin": int(start_bin),
            "sequence_length": int(self.base.sequence_length),
        }
        if not (
            parallel_clip_audit["shape_aligned"]
            and parallel_clip_audit["clip4_nonzero_cells"] > 0
            and parallel_clip_audit["clip8_nonzero_cells"] > 0
        ):
            raise RuntimeError(
                "Parallel clip-8 frames are empty or misaligned."
            )
        output = dict(sample)
        output["expert_frames"] = expert_frames
        output["parallel_clip_audit"] = parallel_clip_audit
        return output


def parallel_clip_collate(samples):
    import torch

    from dataset.temporal_memory import temporal_memory_collate

    output = temporal_memory_collate(samples)
    if "expert_frames" in samples[0]:
        output["expert_frames"] = torch.from_numpy(
            samples[0]["expert_frames"]
        ).float()
        output["parallel_clip_audit"] = dict(samples[0]["parallel_clip_audit"])
    return output


def _save_checkpoint_new(path, payload):
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite checkpoint {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("Stale checkpoint temporary exists.")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def _expert_training(
    protocol,
    spec,
    authorized,
    max_steps=None,
    epochs_override=None,
    result_path_override=None,
    checkpoint_path_override=None,
):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from dataset.temporal_memory import TemporalMemoryTrainDataset
    from model.high_density_polarity_expert import (
        build_expert_model_from_m20,
        configure_expert_only_training,
    )
    from utils.temporal_frame_loss import frame_balanced_event_bce

    require_gpu_authorization(authorized)
    gpu_preflight = require_idle_gpu()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    result_path = Path(result_path_override or spec["result_path"])
    checkpoint_path = Path(checkpoint_path_override or spec["checkpoint_path"])
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Refusing to overwrite training artifacts.")
    training = protocol["training"]
    epochs = int(epochs_override or training["epochs"])
    setup_seed(training["seed"])
    parent_path = workspace_path(protocol["parent_checkpoint"]["workspace_relative_path"])
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, parent_checkpoint = build_expert_model_from_m20(
        parent_path, spec["input_mode"], device=device
    )
    initial_state_sha = tensor_state_sha256(model.state_dict())
    trainable_names = configure_expert_only_training(model)
    named_trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if tuple(sorted(named_trainable)) != trainable_names:
        raise RuntimeError("Named expert trainable scope differs after configuration.")
    initial_expert_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_trainable.items()
    }
    gradient_seen = {name: False for name in named_trainable}
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"name": "high_density_expert", "params": trainable_parameters}],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training["scheduler_min_lr"]),
    )
    base_dataset = TemporalMemoryTrainDataset(
        root=Path(spec["view_root"]),
        whole_t=training["whole_t"],
        temporal_bin_size=training["temporal_bin_size"],
        context_bins=training["context_bins"],
        sequence_length=training["sequence_length"],
        width=training["resolution"][0],
        height=training["resolution"][1],
        views_per_video=training["views_per_video"],
        positive_frame_probability=training["positive_frame_probability"],
        random_seed=training["seed"],
        log_count_clip=4.0,
        cache_all_videos=False,
        cache_video_count=training["cache_video_count"],
        dense_sampling_enabled=False,
    )
    dataset = ParallelClipTrainDataset(
        base_dataset, enabled=(spec["domain"] == "h1")
    )
    actual_names = [path.name for path in dataset.file_paths]
    if actual_names != spec["fit_names"]:
        raise RuntimeError("Training dataset membership/order differs from command audit.")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=parallel_clip_collate,
        pin_memory=True,
    )
    started = time.time()
    steps = 0
    epoch_records = []
    gradient_nonzero_steps = 0
    parallel_clip_audits = []
    last_checkpoint = None
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.eval()
        model.high_density_expert.train()
        loss_sum = 0.0
        grad_norm_sum = 0.0
        epoch_steps = 0
        for batch in dataloader:
            frames = batch["frames"].to(device, non_blocking=True).unsqueeze(0)
            expert_frames = None
            if "expert_frames" in batch:
                expert_frames = batch["expert_frames"].to(
                    device, non_blocking=True
                ).unsqueeze(0)
                if len(parallel_clip_audits) < 4:
                    parallel_clip_audits.append(dict(batch["parallel_clip_audit"]))
            time_indices = batch["event_time_indices"].to(device, non_blocking=True)
            event_x = batch["event_x"].to(device, non_blocking=True)
            event_y = batch["event_y"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logit_maps = model(frames, expert_frames=expert_frames).squeeze(0)
            logits = logit_maps[time_indices, 0, event_y, event_x]
            loss, diagnostics = frame_balanced_event_bce(
                logits,
                labels,
                time_indices,
                target_positive_loss_mass=training["target_positive_loss_mass"],
                max_positive_weight=training["max_positive_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite expert training loss.")
            loss.backward()
            for name, parameter in named_trainable.items():
                gradient = parameter.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    continue
                if float(torch.linalg.vector_norm(gradient.detach()).item()) > 0.0:
                    gradient_seen[name] = True
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=float(training["gradient_clip_norm"])
            )
            grad_value = float(grad_norm.detach().item())
            if math.isfinite(grad_value) and grad_value > 0.0:
                gradient_nonzero_steps += 1
            optimizer.step()
            steps += 1
            epoch_steps += 1
            loss_sum += float(loss.detach().item())
            grad_norm_sum += grad_value
            if max_steps is not None and steps >= int(max_steps):
                break
        epoch_records.append(
            {
                "epoch": epoch,
                "steps": epoch_steps,
                "mean_loss": loss_sum / max(epoch_steps, 1),
                "mean_preclip_gradient_norm": grad_norm_sum / max(epoch_steps, 1),
            }
        )
        scheduler.step()
        if max_steps is None:
            last_checkpoint = checkpoint_path.parent / "epoch_{:03d}_seed49.pt".format(epoch)
            checkpoint_payload = {
                "checkpoint_format_version": 2,
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "temporal_memory": dict(parent_checkpoint.get("temporal_memory", {})),
                "high_density_expert": {
                    "schema": "ev-uav-high-density-dual-expert-v1",
                    "input_mode": spec["input_mode"],
                    "domain": spec["domain"],
                    "insertion_point": "level1",
                    "hidden_channels": 16,
                },
                "provenance": {
                    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                    "runner_sha256": sha256_file(Path(__file__).resolve()),
                    "released_m20_sha256": protocol["parent_checkpoint"]["sha256"],
                    "run_id": spec["run_id"],
                    "fit_names": spec["fit_names"],
                    "held_names": spec["held_names"],
                    "training_scope": {
                        "name": "high_density_expert_only",
                        "trainable_state_tensor_count": 14,
                        "trainable_parameter_count": 1712,
                        "trainable_names": list(trainable_names),
                        "inherited_m20_bitwise_frozen": True,
                    },
                },
            }
            _save_checkpoint_new(last_checkpoint, checkpoint_payload)
        if max_steps is not None and steps >= int(max_steps):
            break
    if max_steps is None and steps != int(spec["expected_optimizer_steps"]):
        raise RuntimeError("Formal optimizer-step count differs from protocol.")
    if max_steps is not None and steps != int(max_steps):
        raise RuntimeError("Probe optimizer-step count differs from protocol.")
    parent_state = parent_checkpoint["model_state_dict"]
    trained_state = model.state_dict()
    updated = {
        name: not torch.equal(
            parameter.detach().cpu(), initial_expert_state[name]
        )
        for name, parameter in named_trainable.items()
    }
    frozen_equal = all(torch.equal(trained_state[name].detach().cpu(), tensor.detach().cpu()) for name, tensor in parent_state.items())
    if not frozen_equal:
        raise RuntimeError("Inherited M20 tensor changed during expert-only training.")
    expert_changed = any(updated.values())
    if gradient_nonzero_steps <= 0:
        raise RuntimeError("Expert training produced no finite nonzero gradient.")
    torch.cuda.synchronize()
    result = {
        "schema": "ev-uav-high-density-dual-expert-training-result-v1",
        "created_utc": utc_now(),
        "status": "completed",
        "run_id": spec["run_id"],
        "fold_id": spec["fold_id"],
        "domain": spec["domain"],
        "arm": spec["arm"],
        "input_mode": spec["input_mode"],
        "fit_names": spec["fit_names"],
        "held_names": spec["held_names"],
        "optimizer_steps": steps,
        "gradient_nonzero_steps": gradient_nonzero_steps,
        "trainable_gradient_seen": gradient_seen,
        "trainable_updated": updated,
        "all_trainable_tensors_saw_nonzero_finite_gradient": all(
            gradient_seen.values()
        ),
        "all_trainable_tensors_updated": all(updated.values()),
        "parallel_clip_audits": parallel_clip_audits,
        "epoch_records": epoch_records,
        "initial_model_state_sha256": initial_state_sha,
        "final_model_state_sha256": tensor_state_sha256(trained_state),
        "inherited_m20_bitwise_frozen": frozen_equal,
        "expert_changed": bool(expert_changed),
        "selection_checkpoint": None if last_checkpoint is None else str(last_checkpoint.resolve()),
        "selection_checkpoint_sha256": None if last_checkpoint is None else sha256_file(last_checkpoint),
        "elapsed_seconds": time.time() - started,
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_preflight": gpu_preflight,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_new_json(result_path, result)
    del model, optimizer, scheduler, dataloader, dataset
    torch.cuda.empty_cache()
    return result


def _spec_by_id(command_audit, run_id):
    matches = [spec for spec in command_audit["formal_specs"] if spec["run_id"] == run_id]
    if len(matches) != 1:
        raise KeyError("Unknown formal run id: {}".format(run_id))
    spec = dict(matches[0])
    spec.pop("command", None)
    return spec


def load_training_result(spec):
    payload, digest = load_json_snapshot(spec["result_path"])
    if payload.get("schema") != "ev-uav-high-density-dual-expert-training-result-v1":
        raise RuntimeError("Training result schema mismatch.")
    if payload.get("run_id") != spec["run_id"] or payload.get("status") != "completed":
        raise RuntimeError("Training result identity/status mismatch.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Training result protocol mismatch.")
    if payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Training result runner mismatch.")
    checkpoint = Path(payload["selection_checkpoint"])
    if sha256_file(checkpoint) != payload["selection_checkpoint_sha256"]:
        raise RuntimeError("Training selection checkpoint SHA mismatch.")
    return payload, digest


def run_probe(authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, command_audit_sha = load_command_audit()
    if PROBE_PATH.exists():
        raise FileExistsError("Refusing to overwrite resource probe.")
    results = {}
    for domain, frozen in protocol["resource_probe"]["domains"].items():
        base_spec = {
            "fold_id": "resource_probe_{}".format(domain),
            "domain": domain,
            "fit_names": [frozen["source"]],
            "held_names": [],
            "view_root": command_audit["views"][
                "resource_probe_{}".format(domain)
            ]["root"],
            "expected_optimizer_steps": frozen["steps_per_arm"],
        }
        for arm, mode_key in (
            ("baseline", "baseline_mode"),
            ("candidate", "candidate_mode"),
        ):
            result_id = "{}__{}".format(domain, arm)
            root = OUTPUT_ROOT / "resource_probe" / domain / arm
            spec = {
                **base_spec,
                "run_id": "resource_probe__{}".format(result_id),
                "arm": arm,
                "input_mode": frozen[mode_key],
                "result_path": str((root / "runtime_result.json").resolve()),
                "checkpoint_path": str((root / "unused.pt").resolve()),
            }
            results[result_id] = _expert_training(
                protocol,
                spec,
                authorized=True,
                max_steps=frozen["steps_per_arm"],
                epochs_override=1,
                result_path_override=spec["result_path"],
                checkpoint_path_override=spec["checkpoint_path"],
            )
    h1_results = [results["h1__baseline"], results["h1__candidate"]]
    h1_clip_audits = [
        audit
        for result in h1_results
        for audit in result["parallel_clip_audits"]
    ]
    checks = {
        "paired_initial_model_state_equal_in_each_domain": all(
            results["{}__baseline".format(domain)]["initial_model_state_sha256"]
            == results["{}__candidate".format(domain)]["initial_model_state_sha256"]
            for domain in ("h1", "h2")
        ),
        "all_inherited_m20_frozen": all(result["inherited_m20_bitwise_frozen"] for result in results.values()),
        "all_runs_have_nonzero_gradients": all(result["gradient_nonzero_steps"] > 0 for result in results.values()),
        "h1_all_14_tensors_saw_nonzero_finite_gradient": all(
            result["all_trainable_tensors_saw_nonzero_finite_gradient"]
            for result in h1_results
        ),
        "h1_all_14_tensors_updated": all(
            result["all_trainable_tensors_updated"] for result in h1_results
        ),
        "h1_clip4_clip8_shape_time_alignment": bool(h1_clip_audits)
        and all(
            audit["clip4_reconstruction_bitwise_equal"]
            and audit["shape_aligned"]
            and audit["sequence_length"] == 16
            for audit in h1_clip_audits
        ),
        "h1_clip4_and_clip8_nonzero": bool(h1_clip_audits)
        and all(
            audit["clip4_nonzero_cells"] > 0
            and audit["clip8_nonzero_cells"] > 0
            for audit in h1_clip_audits
        ),
        "h1_clip4_saturation_visible_in_clip8": sum(
            audit["clip4_saturated_cells"] for audit in h1_clip_audits
        )
        > 0
        and sum(
            audit["clip8_recovered_dynamic_cells"] for audit in h1_clip_audits
        )
        > 0,
        "total_steps_exact": sum(result["optimizer_steps"] for result in results.values()) == protocol["resource_probe"]["total_optimizer_steps"],
    }
    payload = {
        "schema": "ev-uav-high-density-dual-expert-resource-probe-v1",
        "created_utc": utc_now(),
        "passed": all(checks.values()),
        "checks": checks,
        "results": results,
        "new_formal_training_steps": 0,
        "command_audit_sha256": command_audit_sha,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_new_json(PROBE_PATH, payload)
    if not payload["passed"]:
        raise RuntimeError("Resource probe failed.")
    return payload


def require_probe_passed():
    payload, digest = load_json_snapshot(PROBE_PATH)
    if payload.get("schema") != "ev-uav-high-density-dual-expert-resource-probe-v1" or payload.get("passed") is not True:
        raise RuntimeError("Resource probe is absent or failed.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256 or payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Resource probe evidence identity mismatch.")
    return payload, digest


def run_train(run_id, authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    spec = _spec_by_id(command_audit, run_id)
    return _expert_training(protocol, spec, authorized=True)


def run_train_all(authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    results = []
    for frozen in command_audit["formal_specs"]:
        spec = dict(frozen)
        spec.pop("command", None)
        if Path(spec["result_path"]).is_file():
            result, _ = load_training_result(spec)
            results.append(result)
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "train", "--run-id", spec["run_id"], GPU_AUTHORIZATION_FLAG],
            cwd=str(EVC_ROOT),
            check=True,
        )
        result, _ = load_training_result(spec)
        results.append(result)
    return formal_pair_audit(protocol, command_audit)


def _expert_delta_l2(left_checkpoint, right_checkpoint):
    import torch

    left = torch.load(left_checkpoint, map_location="cpu")["model_state_dict"]
    right = torch.load(right_checkpoint, map_location="cpu")["model_state_dict"]
    total = 0.0
    for name in sorted(left):
        if name.startswith("high_density_expert."):
            delta = left[name].to(torch.float64) - right[name].to(torch.float64)
            total += float(torch.sum(delta * delta).item())
        elif not torch.equal(left[name], right[name]):
            raise RuntimeError("Paired inherited M20 state differs: {}".format(name))
    return math.sqrt(total)


def formal_pair_audit(protocol=None, command_audit=None):
    if protocol is None:
        protocol, _ = load_protocol()
    if command_audit is None:
        command_audit, _ = load_command_audit()
    if PAIR_AUDIT_PATH.exists():
        raise FileExistsError("Refusing to overwrite formal pair audit.")
    results = {}
    result_hashes = {}
    for frozen in command_audit["formal_specs"]:
        spec = dict(frozen)
        spec.pop("command", None)
        payload, digest = load_training_result(spec)
        results[spec["run_id"]] = payload
        result_hashes[spec["run_id"]] = digest
    folds = []
    for fold in protocol["dataset"]["folds"]:
        baseline = results["{}__baseline".format(fold["fold_id"])]
        candidate = results["{}__candidate".format(fold["fold_id"])]
        checks = {
            "initial_model_state_bitwise_equal": baseline["initial_model_state_sha256"] == candidate["initial_model_state_sha256"],
            "same_fit_and_held_membership": baseline["fit_names"] == candidate["fit_names"] and baseline["held_names"] == candidate["held_names"],
            "same_optimizer_steps": baseline["optimizer_steps"] == candidate["optimizer_steps"] == fold["optimizer_steps_per_arm"],
            "both_inherited_m20_frozen": baseline["inherited_m20_bitwise_frozen"] and candidate["inherited_m20_bitwise_frozen"],
        }
        task_l2 = _expert_delta_l2(baseline["selection_checkpoint"], candidate["selection_checkpoint"])
        checks["representation_treatment_diverged"] = math.isfinite(task_l2) and task_l2 > 0.0
        folds.append({"fold_id": fold["fold_id"], "expert_task_l2": task_l2, "checks": checks, "passed": all(checks.values())})
    payload = {
        "schema": "ev-uav-high-density-dual-expert-formal-pair-audit-v1",
        "created_utc": utc_now(),
        "passed": all(fold["passed"] for fold in folds),
        "folds": folds,
        "training_result_sha256": result_hashes,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_new_json(PAIR_AUDIT_PATH, payload)
    if not payload["passed"]:
        raise RuntimeError("Formal paired training audit failed.")
    return payload


def require_pair_audit():
    payload, digest = load_json_snapshot(PAIR_AUDIT_PATH)
    if payload.get("schema") != "ev-uav-high-density-dual-expert-formal-pair-audit-v1" or payload.get("passed") is not True:
        raise RuntimeError("Formal pair audit is missing or failed.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256 or payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Formal pair audit identity mismatch.")
    return payload, digest


def evaluation_specs(protocol, command_audit):
    by_id = {spec["run_id"]: spec for spec in command_audit["formal_specs"]}
    output = []
    for fold in protocol["dataset"]["folds"]:
        for arm in ARMS:
            train_spec = dict(by_id["{}__{}".format(fold["fold_id"], arm)])
            train_spec.pop("command", None)
            result, _ = load_training_result(train_spec)
            eval_id = "{}__{}".format(fold["fold_id"], arm)
            output.append(
                {
                    "eval_id": eval_id,
                    "fold_id": fold["fold_id"],
                    "domain": fold["domain"],
                    "arm": arm,
                    "input_mode": train_spec["input_mode"],
                    "held_names": train_spec["held_names"],
                    "checkpoint": result["selection_checkpoint"],
                    "checkpoint_sha256": result["selection_checkpoint_sha256"],
                    "result_path": str((EVALUATION_ROOT / eval_id / "evaluation.json").resolve()),
                }
            )
    return output


def _load_trained_expert_model(checkpoint_path, device):
    import torch

    from model.high_density_polarity_expert import HighDensityPolarityExpertMemoryNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("high_density_expert", {})
    mode = metadata.get("input_mode")
    model = HighDensityPolarityExpertMemoryNet(
        input_channels=10,
        width=16,
        temporal_attention_enabled=True,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=False,
        confidence_head_enabled=False,
        expert_input_mode=mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def predict_expert_full_stream_scores(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
):
    """Full-T160 inference with an optional parallel clip-8 H1 expert stack."""
    import numpy as np
    import torch

    from dataset.temporal_frame import build_temporal_context_frame

    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count <= 0 or int(inference_batch_size) <= 0:
        raise ValueError("Invalid full-stream inference dimensions.")
    use_clip8 = model.expert_input_mode == "h1_saturation"

    def frame_batch(temporal_bins, clip):
        frames = np.stack(
            [
                build_temporal_context_frame(
                    video,
                    temporal_bin,
                    int(context_bins),
                    int(width),
                    int(height),
                    float(clip),
                )
                for temporal_bin in temporal_bins
            ],
            axis=0,
        )
        return torch.from_numpy(frames).float().to(device)

    bottlenecks = []
    with torch.no_grad():
        for start in range(0, temporal_bin_count, int(inference_batch_size)):
            bins = list(
                range(
                    start,
                    min(start + int(inference_batch_size), temporal_bin_count),
                )
            )
            frames = frame_batch(bins, log_count_clip)
            expert_frames = frame_batch(bins, 8.0) if use_clip8 else None
            bottlenecks.append(
                model.encode_bottleneck(frames, expert_frames=expert_frames)
            )
        residuals = model.temporal_residual(torch.cat(bottlenecks, dim=0))
    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, temporal_bin_count, int(inference_batch_size)):
            bins = list(
                range(
                    start,
                    min(start + int(inference_batch_size), temporal_bin_count),
                )
            )
            frames = frame_batch(bins, log_count_clip)
            expert_frames = frame_batch(bins, 8.0) if use_clip8 else None
            decoded = model.decode_with_residual(
                frames,
                residuals[start : start + len(bins)],
                expert_frames=expert_frames,
            )
            probabilities = torch.sigmoid(decoded).squeeze(1).cpu().numpy()
            for local_index, temporal_bin in enumerate(bins):
                event_indices = video.event_indices_by_bin[temporal_bin]
                if event_indices.size == 0:
                    continue
                locations = video.locations[event_indices]
                scores[event_indices] = probabilities[
                    local_index, locations[:, 1], locations[:, 0]
                ]
    if not np.isfinite(scores).all():
        raise RuntimeError("Expert full-stream inference produced non-finite scores.")
    return torch.from_numpy(scores)


def evaluate_spec(protocol, spec, authorized=False):
    import numpy as np
    import torch

    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay
    from crossfit_component_reranker import SufficientCounts, metrics_from_counts, sufficient_counts_for_video
    from train_component_reranker import _load_train_source
    from utils.postprocess import ChallengePostprocessor
    from utils.temporal_frame_inference import temporal_frame_video_from_sample

    require_gpu_authorization(authorized)
    gpu_preflight = require_idle_gpu()
    result_path = Path(spec["result_path"])
    if result_path.exists() or result_path.parent.exists():
        raise FileExistsError("Refusing to overwrite held-train evaluation.")
    checkpoint = Path(spec["checkpoint"])
    if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise RuntimeError("Evaluation checkpoint SHA mismatch.")
    cfg = replay.load_flat_config(
        EVC_ROOT / "configs" / "evisseg_evuav.yaml",
        list(protocol["evaluation"]["fixed_config_overrides"]),
    )
    threshold = float(protocol["evaluation"]["prediction_threshold"])
    c00 = component_crossfit.validate_c00_config(cfg, threshold)
    if component_crossfit.sha256_json(c00) != protocol["evaluation"]["effective_c00_canonical_sha256"]:
        raise RuntimeError("Effective C00 contract mismatch.")
    if int(cfg.temporal_memory_sequence_length) != 16:
        raise RuntimeError("Only frozen T16 full-stream inference is allowed.")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    model, checkpoint_payload = _load_trained_expert_model(checkpoint, device)
    records = []
    pooled = SufficientCounts()
    index = source_index(protocol)
    root = official_train_root(protocol)
    started = time.time()
    for name in spec["held_names"]:
        item = index[name]
        source = root / name
        before = sha256_file(source)
        if before != item["sha256"]:
            raise RuntimeError("Held train source identity mismatch.")
        sample, labels, target_ids = _load_train_source(source)
        if sha256_file(source) != before:
            raise RuntimeError("Held train source changed during load.")
        video = temporal_frame_video_from_sample(sample, cfg.temporal_memory_bin_size, cfg.whole_t)
        raw = predict_expert_full_stream_scores(
            model,
            video,
            device,
            cfg.temporal_memory_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
        ).reshape(-1).detach().cpu().to(torch.float32)
        if int(raw.numel()) != len(labels):
            raise RuntimeError("Held full-stream score length mismatch.")
        locations = np.column_stack((np.zeros(len(labels), dtype=np.int64), sample["ev_loc"].astype(np.int64, copy=False)))
        location_tensor = torch.from_numpy(locations).to(torch.int64).contiguous()
        processor = ChallengePostprocessor.from_cfg(cfg, threshold, event_count=len(labels))
        processed, stats = processor.apply(raw.clone(), location_tensor)
        counts = sufficient_counts_for_video(processed.numpy(), labels, target_ids, locations, prediction_threshold=threshold)
        pooled = pooled + counts
        records.append(
            {
                "source_name": name,
                "source_sha256": item["sha256"],
                "event_count": len(labels),
                "observable_route": observable_route(item["event_count"], item["polarity_minority_fraction"]),
                "raw_scores_float32_sha256": sha256_float32(raw.numpy()),
                "processed_scores_float32_sha256": sha256_float32(processed.numpy()),
                "postprocess_stats": asdict(stats),
                "counts": counts.to_dict(),
                "metrics": metrics_from_counts(counts),
            }
        )
        if sha256_file(source) != before:
            raise RuntimeError("Held train source changed during inference.")
        del sample, labels, target_ids, video, raw, processed
        torch.cuda.empty_cache()
    if [record["source_name"] for record in records] != spec["held_names"]:
        raise RuntimeError("Held record order mismatch.")
    payload = {
        "schema": "ev-uav-high-density-dual-expert-held-train-evaluation-v1",
        "created_utc": utc_now(),
        "eval_id": spec["eval_id"],
        "fold_id": spec["fold_id"],
        "domain": spec["domain"],
        "arm": spec["arm"],
        "input_mode": spec["input_mode"],
        "dataset_split": "train",
        "held_stream": "complete_full_stream_t160",
        "t32_read_or_combined": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "records": records,
        "pooled_counts": pooled.to_dict(),
        "pooled_metrics": metrics_from_counts(pooled),
        "effective_c00_contract": c00,
        "effective_c00_canonical_sha256": component_crossfit.sha256_json(c00),
        "elapsed_seconds": time.time() - started,
        "gpu_preflight": gpu_preflight,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_new_json(result_path, payload)
    return payload


def load_evaluation_result(spec):
    payload, digest = load_json_snapshot(spec["result_path"])
    if payload.get("schema") != "ev-uav-high-density-dual-expert-held-train-evaluation-v1":
        raise RuntimeError("Held evaluation schema mismatch.")
    if payload.get("eval_id") != spec["eval_id"] or payload.get("dataset_split") != "train":
        raise RuntimeError("Held evaluation identity/split mismatch.")
    if payload.get("t32_read_or_combined") is not False:
        raise RuntimeError("T32 was read or combined.")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256 or payload.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Held evaluation evidence identity mismatch.")
    if [record["source_name"] for record in payload["records"]] != spec["held_names"]:
        raise RuntimeError("Held evaluation source membership mismatch.")
    return payload, digest


def run_evaluate(eval_id, authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    require_pair_audit()
    specs = evaluation_specs(protocol, command_audit)
    matches = [spec for spec in specs if spec["eval_id"] == eval_id]
    if len(matches) != 1:
        raise KeyError("Unknown evaluation id: {}".format(eval_id))
    return evaluate_spec(protocol, matches[0], authorized=True)


def run_evaluate_all(authorized=False):
    require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    command_audit, _ = load_command_audit()
    require_probe_passed()
    require_pair_audit()
    results = []
    for spec in evaluation_specs(protocol, command_audit):
        if Path(spec["result_path"]).is_file():
            result, _ = load_evaluation_result(spec)
            results.append(result)
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "evaluate", "--eval-id", spec["eval_id"], GPU_AUTHORIZATION_FLAG],
            cwd=str(EVC_ROOT),
            check=True,
        )
        result, _ = load_evaluation_result(spec)
        results.append(result)
    return results


def add_count_dicts(values):
    output = {key: 0 for key in COUNT_FIELDS}
    for value in values:
        if set(value) != set(COUNT_FIELDS):
            raise RuntimeError("Sufficient-count fields differ from evaluator.")
        for key in COUNT_FIELDS:
            output[key] += int(value[key])
    return output


def population_invariants(counts):
    return {
        "positive_events": int(counts["true_positive_events"]) + int(counts["false_negative_events"]),
        "object_count": int(counts["object_count"]),
        "frame_count": int(counts["frame_count"]),
        "event_count": int(counts["event_count"]),
    }


def _released_m20_records(protocol):
    import numpy as np
    import torch

    import crossfit_component_reranker as component_crossfit
    import replay_temporal_memory_validation as replay
    from crossfit_component_reranker import metrics_from_counts, sufficient_counts_for_video
    from train_component_reranker import _load_cache_record, load_train_cache
    from utils.postprocess import ChallengePostprocessor

    cache_root = workspace_path(protocol["released_m20_train_cache"]["workspace_relative_path"])
    loaded_root, manifest_path, manifest_sha, manifest = load_train_cache(cache_root)
    if manifest_sha != protocol["released_m20_train_cache"]["manifest_sha256"]:
        raise RuntimeError("Released M20 cache manifest changed.")
    if manifest.get("base_checkpoint_sha256") != protocol["parent_checkpoint"]["sha256"]:
        raise RuntimeError("Released cache parent checkpoint mismatch.")
    cfg = replay.load_flat_config(EVC_ROOT / "configs" / "evisseg_evuav.yaml", list(protocol["evaluation"]["fixed_config_overrides"]))
    threshold = float(protocol["evaluation"]["prediction_threshold"])
    c00 = component_crossfit.validate_c00_config(cfg, threshold)
    if component_crossfit.sha256_json(c00) != protocol["evaluation"]["effective_c00_canonical_sha256"]:
        raise RuntimeError("Released cache C00 contract mismatch.")
    metadata_by_name = {item["source_name"]: item for item in manifest["records"]}
    records = {}
    for name, frozen in source_index(protocol).items():
        metadata = metadata_by_name[name]
        if metadata.get("source_sha256") != frozen["sha256"]:
            raise RuntimeError("Released cache source identity mismatch.")
        values = _load_cache_record(loaded_root, metadata)
        scores = np.ascontiguousarray(values["scores"], dtype=np.float32).reshape(-1)
        locs3 = np.ascontiguousarray(values["locs"], dtype=np.int64)
        locations = np.column_stack((np.zeros(scores.size, dtype=np.int64), locs3))
        raw = torch.from_numpy(scores).to(torch.float32)
        processor = ChallengePostprocessor.from_cfg(cfg, threshold, event_count=scores.size)
        processed, stats = processor.apply(raw.clone(), torch.from_numpy(locations).to(torch.int64))
        counts = sufficient_counts_for_video(
            processed.numpy(),
            values["labels"].reshape(-1),
            values["target_ids"].reshape(-1),
            locations,
            prediction_threshold=threshold,
        )
        records[name] = {
            "source_name": name,
            "source_sha256": frozen["sha256"],
            "raw_scores_float32_sha256": sha256_float32(scores),
            "processed_scores_float32_sha256": sha256_float32(processed.numpy()),
            "postprocess_stats": asdict(stats),
            "counts": counts.to_dict(),
            "metrics": metrics_from_counts(counts),
        }
    return records, {"path": str(manifest_path), "sha256": manifest_sha}


def _metrics_from_dict(counts):
    from crossfit_component_reranker import SufficientCounts, metrics_from_counts

    return metrics_from_counts(SufficientCounts(**counts))


def _metric_delta(candidate, comparator):
    return {key: float(candidate[key]) - float(comparator[key]) for key in ("score", "pd", "fa", "iou", "acc", "score_fa")}


def _fold_safety(candidate, comparator):
    checks = {
        "score_not_lower": float(candidate["metrics"]["score"]) >= float(comparator["metrics"]["score"]),
        "pd_not_lower": float(candidate["metrics"]["pd"]) >= float(comparator["metrics"]["pd"]),
        "iou_not_lower": float(candidate["metrics"]["iou"]) >= float(comparator["metrics"]["iou"]),
        "fa_not_higher": float(candidate["metrics"]["fa"]) <= float(comparator["metrics"]["fa"]),
        "correct_objects_not_lower": int(candidate["counts"]["correct_objects"]) >= int(comparator["counts"]["correct_objects"]),
        "population_invariants_equal": population_invariants(candidate["counts"]) == population_invariants(comparator["counts"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _pooled_gate(protocol, domain, candidate, comparator):
    threshold = (
        protocol["promotion_gates"]["h2_held_pooled_score_delta_minimum_against_each_comparator"]
        if domain == "h2"
        else protocol["promotion_gates"]["h1_held_pooled_score_delta_minimum_against_each_comparator"]
        if domain == "h1"
        else protocol["promotion_gates"]["combined_held_pooled_score_delta_minimum_against_each_comparator"]
    )
    checks = {
        "score_delta_minimum": float(candidate["metrics"]["score"]) - float(comparator["metrics"]["score"]) >= float(threshold),
        "pd_not_lower": float(candidate["metrics"]["pd"]) >= float(comparator["metrics"]["pd"]),
        "iou_not_lower": float(candidate["metrics"]["iou"]) >= float(comparator["metrics"]["iou"]),
        "fa_not_higher": float(candidate["metrics"]["fa"]) <= float(comparator["metrics"]["fa"]),
        "correct_objects_not_lower": int(candidate["counts"]["correct_objects"]) >= int(comparator["counts"]["correct_objects"]),
        "false_positive_events_strictly_lower": int(candidate["counts"]["false_positive_events"]) < int(comparator["counts"]["false_positive_events"]),
        "false_components_strictly_lower": int(candidate["counts"]["false_components"]) < int(comparator["counts"]["false_components"]),
        "population_invariants_equal": population_invariants(candidate["counts"]) == population_invariants(comparator["counts"]),
    }
    return {"required_score_delta": float(threshold), "checks": checks, "passed": all(checks.values())}


def run_report():
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Report must start before CUDA initialization.")
    protocol, protocol_sha = load_protocol()
    command_audit, command_audit_sha = load_command_audit()
    probe, probe_sha = require_probe_passed()
    pair, pair_sha = require_pair_audit()
    specs = evaluation_specs(protocol, command_audit)
    evaluations = {}
    artifacts = {}
    for spec in specs:
        payload, digest = load_evaluation_result(spec)
        evaluations[(spec["fold_id"], spec["arm"])] = payload
        artifacts[spec["eval_id"]] = {"path": spec["result_path"], "sha256": digest}
    released_records, cache_evidence = _released_m20_records(protocol)
    fold_results = []
    pooled_by_domain = {
        domain: {variant: [] for variant in ("released_m20", "baseline", "candidate")}
        for domain in ("h1", "h2", "combined")
    }
    held_seen = {variant: [] for variant in ("released_m20", "baseline", "candidate")}
    for fold in protocol["dataset"]["folds"]:
        fold_id = fold["fold_id"]
        domain = fold["domain"]
        expected_names = [item["name"] for item in held_items(protocol, fold)]
        result = {"fold_id": fold_id, "domain": domain, "held_names": expected_names}
        for arm in ARMS:
            payload = evaluations[(fold_id, arm)]
            counts = payload["pooled_counts"]
            result[arm] = {"counts": counts, "metrics": payload["pooled_metrics"]}
            pooled_by_domain[domain][arm].append(counts)
            pooled_by_domain["combined"][arm].append(counts)
            held_seen[arm].extend(expected_names)
        released_counts = add_count_dicts([released_records[name]["counts"] for name in expected_names])
        result["released_m20"] = {"counts": released_counts, "metrics": _metrics_from_dict(released_counts)}
        pooled_by_domain[domain]["released_m20"].append(released_counts)
        pooled_by_domain["combined"]["released_m20"].append(released_counts)
        held_seen["released_m20"].extend(expected_names)
        result["candidate"]["delta_vs_baseline"] = _metric_delta(result["candidate"]["metrics"], result["baseline"]["metrics"])
        result["candidate"]["delta_vs_released_m20"] = _metric_delta(result["candidate"]["metrics"], result["released_m20"]["metrics"])
        result["gates"] = {
            "vs_baseline": _fold_safety(result["candidate"], result["baseline"]),
            "vs_released_m20": _fold_safety(result["candidate"], result["released_m20"]),
        }
        fold_results.append(result)
    pooled = {}
    pooled_gates = {}
    for domain in ("h1", "h2", "combined"):
        pooled[domain] = {}
        for variant in ("released_m20", "baseline", "candidate"):
            counts = add_count_dicts(pooled_by_domain[domain][variant])
            pooled[domain][variant] = {"counts": counts, "metrics": _metrics_from_dict(counts)}
        pooled[domain]["candidate"]["delta_vs_baseline"] = _metric_delta(pooled[domain]["candidate"]["metrics"], pooled[domain]["baseline"]["metrics"])
        pooled[domain]["candidate"]["delta_vs_released_m20"] = _metric_delta(pooled[domain]["candidate"]["metrics"], pooled[domain]["released_m20"]["metrics"])
        pooled_gates[domain] = {
            "vs_baseline": _pooled_gate(protocol, domain, pooled[domain]["candidate"], pooled[domain]["baseline"]),
            "vs_released_m20": _pooled_gate(protocol, domain, pooled[domain]["candidate"], pooled[domain]["released_m20"]),
        }
    all_names = sorted(source_index(protocol))
    held_checks = {
        variant: sorted(names) == all_names and len(names) == len(set(names)) == len(all_names)
        for variant, names in held_seen.items()
    }
    passed = (
        all(check["passed"] for fold in fold_results for check in fold["gates"].values())
        and all(check["passed"] for domain in pooled_gates.values() for check in domain.values())
        and all(held_checks.values())
    )
    payload = {
        "schema": "ev-uav-high-density-dual-expert-grouped-oof-report-v1",
        "created_utc": utc_now(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "evidence_class": protocol["evidence_class"],
        "claim_limitations": protocol["claim_limitations"],
        "fold_results": fold_results,
        "pooled": pooled,
        "pooled_gates": pooled_gates,
        "held_only_pool_checks": held_checks,
        "evaluation_artifacts": artifacts,
        "released_m20_cache_evidence": cache_evidence,
        "command_audit_sha256": command_audit_sha,
        "resource_probe_sha256": probe_sha,
        "formal_pair_audit_sha256": pair_sha,
        "protocol_sha256": protocol_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "t32_read_or_combined": False,
        "no_validation_test_submission_or_default_change": True,
        "decision": "eligible_for_separate_all15_refit_protocol" if passed else protocol["promotion_gates"]["failure_action"],
    }
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU report initialized CUDA.")
    write_new_json(REPORT_PATH, payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Run immutable CPU-only protocol/preflight audit.")
    probe = subparsers.add_parser("probe", help="Run the four-step paired GPU resource probe.")
    probe.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    train = subparsers.add_parser("train", help="Train exactly one frozen fold/arm.")
    train.add_argument("--run-id", required=True)
    train.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    train_all = subparsers.add_parser("train-all", help="Train all ten frozen fold/arms.")
    train_all.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    pair = subparsers.add_parser("pair-audit", help="Audit all ten completed train runs on CPU.")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate exactly one held-train fold/arm.")
    evaluate.add_argument("--eval-id", required=True)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    evaluate_all = subparsers.add_parser("evaluate-all", help="Evaluate all ten held-train fold/arms.")
    evaluate_all.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    subparsers.add_parser("report", help="Build the CPU-only dual-anchor grouped report.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        payload = run_audit()
    elif args.command == "probe":
        payload = run_probe(args.authorized)
    elif args.command == "train":
        payload = run_train(args.run_id, args.authorized)
    elif args.command == "train-all":
        payload = run_train_all(args.authorized)
    elif args.command == "pair-audit":
        payload = formal_pair_audit()
    elif args.command == "evaluate":
        payload = run_evaluate(args.eval_id, args.authorized)
    elif args.command == "evaluate-all":
        payload = run_evaluate_all(args.authorized)
    elif args.command == "report":
        payload = run_report()
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

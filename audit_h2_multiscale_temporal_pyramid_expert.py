"""CPU-only protocol/API/identity audit for the H2 temporal pyramid expert."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from model.h2_multiscale_temporal_pyramid_expert import (
    OBSERVATION_CHANNELS,
    TEMPORAL_SCALES,
    FrozenM20MultiScalePyramidAdapter,
    audit_released_m20_feature_api,
    pyramid_expert_parameter_count,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROTOCOL_PATH = ROOT / "protocols" / "h2_multiscale_temporal_pyramid_expert_science_v1.json"
M20_PATH = ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
OUTPUT_PATH = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_multiscale_temporal_pyramid_expert_v1"
    / "cpu_audit"
    / "report.json"
)
EXPECTED_M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_released_m20_cpu(payload):
    metadata = payload.get("temporal_memory", {})
    required = {
        "temporal_bin_size": 50,
        "context_bins": 5,
        "width": 16,
        "sequence_length": 16,
        "log_count_clip": 4.0,
        "density_calibration_enabled": True,
        "confidence_head_enabled": False,
        "temporal_attention_enabled": True,
    }
    for name, expected in required.items():
        actual = metadata.get(name)
        if isinstance(expected, float):
            matches = float(actual) == expected
        elif isinstance(expected, bool):
            matches = bool(actual) is expected
        else:
            matches = int(actual) == expected
        if not matches:
            raise RuntimeError("released M20 metadata changed for {}".format(name))
    model = BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=16,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=False,
        confidence_head_enabled=False,
        temporal_attention_enabled=True,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, metadata


def validate_protocol(protocol):
    if protocol.get("schema") != "ev-uav-frozen-m20-h2-multiscale-temporal-pyramid-expert-science-v1":
        raise RuntimeError("unexpected pyramid science protocol schema")
    if protocol["status"] != "frozen_before_any_multiscale_pyramid_GPU_or_held_prediction":
        raise RuntimeError("pyramid protocol is not frozen before GPU")
    architecture = protocol["architecture"]
    if architecture["temporal_scales_bins"] != list(TEMPORAL_SCALES):
        raise RuntimeError("pyramid temporal scales changed")
    if architecture["observation_channels"] != OBSERVATION_CHANNELS:
        raise RuntimeError("pyramid observation width changed")
    if architecture["architecture_scale_or_model_grid_allowed"] is not False:
        raise RuntimeError("pyramid protocol must forbid architecture grids")
    if protocol["eight_step_probe"]["GPU_authorized"] is not False:
        raise RuntimeError("CPU audit requires an unauthorized GPU probe")
    if protocol["science_scope"]["validation_read_allowed"] is not False:
        raise RuntimeError("validation access must remain forbidden")
    if protocol["science_scope"]["test_read_allowed"] is not False:
        raise RuntimeError("test access must remain forbidden")


def analytic_budget(parameter_count):
    low_height = int(math.ceil(260 / 8))
    low_width = int(math.ceil(346 / 8))
    observation_elements = 160 * OBSERVATION_CHANNELS * low_height * low_width
    summary_elements = 160 * len(TEMPORAL_SCALES) * (2 * OBSERVATION_CHANNELS) * low_height * low_width
    return {
        "low_resolution_shape": [160, OBSERVATION_CHANNELS, low_height, low_width],
        "observation_fp32_MiB": observation_elements * 4 / (1024.0 ** 2),
        "four_scale_summary_fp16_CPU_MiB": summary_elements * 2 / (1024.0 ** 2),
        "expert_parameter_fp32_MiB": parameter_count * 4 / (1024.0 ** 2),
        "expert_parameter_gradient_Adam_fp32_upper_MiB": parameter_count * 16 / (1024.0 ** 2),
        "estimated_expert_activation_increment_MiB": 250.0,
        "conservative_total_peak_CUDA_GiB": 3.5,
        "measurement_class": "analytic_only_no_GPU_run",
    }


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit refuses an initialized CUDA context")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    validate_protocol(protocol)
    if sha256_file(M20_PATH) != EXPECTED_M20_SHA256:
        raise RuntimeError("released M20 checkpoint SHA-256 changed")
    payload = load_checkpoint(M20_PATH)
    released, metadata = build_released_m20_cpu(payload)
    before = state_sha256(released.state_dict())
    feature_api = audit_released_m20_feature_api(released, context_bins=5)
    wrapper = FrozenM20MultiScalePyramidAdapter(released, context_bins=5)
    wrapper.train()
    after = state_sha256(released.state_dict())
    if before != after or any(parameter.requires_grad for parameter in released.parameters()):
        raise RuntimeError("pyramid wrapper changed or unfroze released M20")
    parameter_count = pyramid_expert_parameter_count(wrapper)
    if parameter_count != int(protocol["architecture"]["trainable_parameter_count"]):
        raise RuntimeError("pyramid trainable parameter count changed")

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(ROOT / "tests"),
        "-p",
        "test_h2_multiscale_temporal_pyramid_expert.py",
        "-v",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("pyramid CPU tests failed\n" + completed.stdout + completed.stderr)
    report = {
        "schema": "ev-uav-h2-multiscale-temporal-pyramid-cpu-audit-v1",
        "created_utc": utc_now(),
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "released_m20": str(M20_PATH.resolve()),
        "released_m20_sha256": sha256_file(M20_PATH),
        "released_m20_metadata": metadata,
        "released_m20_state_sha256_before": before,
        "released_m20_state_sha256_after": after,
        "released_m20_bitwise_unchanged": before == after,
        "feature_API": feature_api,
        "expert_parameter_count": parameter_count,
        "model_sha256": sha256_file(ROOT / "model" / "h2_multiscale_temporal_pyramid_expert.py"),
        "loss_sha256": sha256_file(ROOT / "utils" / "h2_multiscale_pyramid_loss.py"),
        "tests_sha256": sha256_file(ROOT / "tests" / "test_h2_multiscale_temporal_pyramid_expert.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "cpu_tests": (completed.stdout + completed.stderr).strip(),
        "analytic_eight_step_GPU_budget": analytic_budget(parameter_count),
        "probe_optimizer_steps": 8,
        "GPU_authorized": False,
        "GPU_used": False,
        "train_source_array_read": False,
        "held_source_array_read": False,
        "validation_or_test_read": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "decision": "stop_before_GPU_and_request_root_authorization",
    }
    output = Path(args.output)
    if output.exists() or output.parent.exists():
        raise FileExistsError("refusing to overwrite pyramid CPU audit")
    output.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "protocol_sha256": report["protocol_sha256"],
                "expert_parameter_count": parameter_count,
                "conservative_total_peak_CUDA_GiB": report["analytic_eight_step_GPU_budget"]["conservative_total_peak_CUDA_GiB"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(OUTPUT_PATH))
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))

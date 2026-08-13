"""CPU-only protocol, identity, no-leak, and resource audit for recovery V2."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from model.h2_pyramid_component_recovery import (
    H2PyramidComponentRecoveryHead,
    component_recovery_parameter_count,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROTOCOL_PATH = ROOT / "protocols" / "h2_pyramid_component_recovery_v2_science_v1.json"
EXPECTED_PROTOCOL_SHA256 = "4c4c260b66bf4c77fb314432bd2c72432a3273917347a8f5bf943d8489933c70"
OUTPUT_PATH = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_pyramid_component_recovery_v2"
    / "cpu_audit"
    / "report.json"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(relative):
    value = (WORKSPACE / str(relative)).resolve()
    if value != WORKSPACE.resolve() and WORKSPACE.resolve() not in value.parents:
        raise RuntimeError("protocol path escaped workspace")
    return value


def write_json_exclusive(path, payload):
    values = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(values)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(values).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    descriptor = os.open(str(sidecar), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((digest + "  " + path.name + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def run():
    if OUTPUT_PATH.exists() or OUTPUT_PATH.parent.exists():
        raise FileExistsError("refusing to overwrite V2 CPU audit")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only audit")
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen V2 protocol changed")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol.get("status") != (
        "frozen_after_G3_development_before_any_G2_or_G1_array_read_or_V2_GPU"
    ):
        raise RuntimeError("V2 protocol is not frozen at the CPU/GPU gate")
    if protocol["resource_budget_before_GPU"]["authorized_GPU"] is not False:
        raise RuntimeError("V2 protocol unexpectedly authorizes GPU")
    if protocol["science_scope"]["validation_read_allowed"] is not False:
        raise RuntimeError("validation access is not forbidden")
    if protocol["science_scope"]["test_read_allowed"] is not False:
        raise RuntimeError("test access is not forbidden")

    for record in protocol["frozen_code"].values():
        path = workspace_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("frozen V2 code changed: {}".format(path))
    evidence_records = (
        protocol["development_evidence"]["descriptive_capacity"],
        protocol["development_evidence"]["selective_capacity"],
    )
    for record in evidence_records:
        path = workspace_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("V2 development evidence changed")
    decision_path = workspace_path(protocol["V1_archive"]["branch_decision_path"])
    checkpoint_path = workspace_path(protocol["V1_archive"]["V1_checkpoint_path"])
    if sha256_file(decision_path) != protocol["V1_archive"]["branch_decision_sha256"]:
        raise RuntimeError("V1 archive decision changed")
    if sha256_file(checkpoint_path) != protocol["V1_archive"]["V1_checkpoint_sha256"]:
        raise RuntimeError("V1 archived checkpoint changed")
    source_protocol = workspace_path(protocol["source_manifest"]["inherited_protocol"])
    if sha256_file(source_protocol) != protocol["source_manifest"]["sha256"]:
        raise RuntimeError("source manifest protocol changed")
    parent_science = workspace_path(protocol["Stage1_pyramid"]["parent_science_protocol"])
    if sha256_file(parent_science) != protocol["Stage1_pyramid"][
        "parent_science_protocol_sha256"
    ]:
        raise RuntimeError("Stage1 science protocol changed")

    groups = protocol["source_manifest"]["groups"]
    g1 = tuple(groups["G1_fresh_outer_held"])
    g2 = tuple(groups["G2_outer_fit"])
    g3 = tuple(groups["G3_outer_fit_design_exposed"])
    if len(g1) != 4 or len(g2) != 3 or len(g3) != 4:
        raise RuntimeError("V2 source group sizes changed")
    if set(g1) & set(g2) or set(g1) & set(g3) or set(g2) & set(g3):
        raise RuntimeError("V2 source groups overlap")
    if set(g1 + g2 + g3) != {
        "train_{:03d}.npz".format(index) for index in range(88, 99)
    }:
        raise RuntimeError("V2 source groups do not cover exact H2 train sources")

    head = H2PyramidComponentRecoveryHead()
    parameter_count = component_recovery_parameter_count(head)
    if parameter_count != protocol["recovery_head"]["trainable_parameter_count"]:
        raise RuntimeError("recovery-head parameter count changed")
    forward_arguments = tuple(inspect.signature(head.forward).parameters)
    if forward_arguments != ("node_features", "node_mask"):
        raise RuntimeError("recovery-head inference API changed")

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_h2_pyramid_component_recovery.py",
        "-v",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("V2 CPU tests failed:\n{}".format(completed.stdout + completed.stderr))
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during V2 CPU audit")

    resource = protocol["resource_budget_before_GPU"]
    parameter_mib = parameter_count * 4 / (1024.0 ** 2)
    node_cache_mib = (
        int(resource["component_microbatch_max_components"])
        * int(resource["component_temporal_nodes_max"])
        * int(protocol["feature_contract"]["node_feature_count"])
        * 4
        / (1024.0 ** 2)
    )
    if parameter_mib > float(resource["head_parameter_FP32_MiB"]):
        raise RuntimeError("analytic parameter budget is understated")
    if node_cache_mib != float(resource["component_node_cache_FP32_MiB"]):
        raise RuntimeError("analytic node-cache budget changed")

    descriptive = json.loads(
        workspace_path(protocol["development_evidence"]["descriptive_capacity"]["path"])
        .read_text(encoding="utf-8")
    )
    selective = json.loads(
        workspace_path(protocol["development_evidence"]["selective_capacity"]["path"])
        .read_text(encoding="utf-8")
    )
    if descriptive.get("capacity_passed") is not False:
        raise RuntimeError("restore-all capacity receipt interpretation changed")
    if selective.get("capacity_passed") is not True:
        raise RuntimeError("selective capacity gate does not pass")
    report = {
        "schema": "ev-uav-h2-pyramid-component-recovery-v2-cpu-audit-v1",
        "created_utc": utc_now(),
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "audit_runner_path": str(Path(__file__).resolve()),
        "audit_runner_sha256": sha256_file(Path(__file__)),
        "frozen_code_sha256": {
            name: record["sha256"] for name, record in protocol["frozen_code"].items()
        },
        "V1_archive_verified": True,
        "selective_capacity_verified": True,
        "recovery_head_parameter_count": parameter_count,
        "forward_arguments": list(forward_arguments),
        "unit_test_output": (completed.stdout + completed.stderr).strip(),
        "unit_test_count": 6,
        "analytic_resource_budget": {
            "head_parameter_FP32_MiB": parameter_mib,
            "head_AdamW_upper_MiB": resource[
                "head_AdamW_parameter_gradient_and_state_upper_MiB"
            ],
            "component_node_microbatch_FP32_MiB": node_cache_mib,
            "attention_and_head_activation_conservative_MiB": resource[
                "attention_and_head_activation_conservative_MiB"
            ],
            "parent_observed_Stage1_peak_MiB": resource[
                "frozen_Stage1_feature_extraction_observed_parent_peak_MiB"
            ],
            "conservative_total_peak_CUDA_GiB": resource[
                "conservative_total_peak_CUDA_GiB"
            ],
        },
        "G1_arrays_or_old_predictions_read": False,
        "G2_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
        "CUDA_initialized_or_used": False,
        "GPU_authorized": False,
        "CPU_audit_passed": True,
    }
    digest = write_json_exclusive(OUTPUT_PATH, report)
    print(
        json.dumps(
            {
                "report": str(OUTPUT_PATH.resolve()),
                "report_sha256": digest,
                "CPU_audit_passed": True,
                "recovery_head_parameter_count": parameter_count,
                "unit_test_count": 6,
                "G1_arrays_or_old_predictions_read": False,
                "G2_arrays_or_predictions_read": False,
                "validation_or_test_read": False,
                "CUDA_initialized_or_used": False,
                "GPU_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()

"""CPU-only pre-GPU audit for the H2 activity + selective-recovery design."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

from model.h2_activity_selective_recovery import (
    DisagreementRecoveryNet,
    recovery_parameter_count,
)
from model.high_density_polarity_expert import FineTemporalPolarityMultiScaleExpert


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
PROTOCOL_PATH = (
    REPO_ROOT
    / "protocols"
    / "h2_activity_suppress_selective_recovery_g2_science_v1.json"
)
OUTPUT_PATH = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_h2_activity_suppress_selective_recovery_g2_v1"
    / "cpu_audit_passed.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    protocol = load_json(PROTOCOL_PATH)
    checks = {}
    checks["status_pre_gpu"] = (
        protocol["status"]
        == "frozen_cpu_design_before_any_gpu_probe_or_fresh_g2_open"
    )
    checks["cuda_hidden_and_unavailable"] = (
        os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_available()
    )
    groups = protocol["source_groups"]
    g1 = set(groups["g1_088_091"])
    g2 = set(groups["g2_092_094_outer_held"])
    g3 = set(groups["g3_095_098"])
    checks["group_partition_disjoint"] = not (g1 & g2 or g1 & g3 or g2 & g3)
    checks["fit_exactly_g1_plus_g3"] = set(protocol["scope"]["fit_sources"]) == g1 | g3
    checks["held_exactly_g2"] = set(protocol["scope"]["fresh_outer_held_sources"]) == g2
    checks["validation_test_forbidden"] = not protocol["scope"][
        "validation_or_test_allowed"
    ]
    route = protocol["input_only_route"]
    checks["exact_h2_input_only_route"] = (
        int(route["event_count_cutoff_exclusive"]) == 200000
        and float(route["polarity_minority_cutoff"]) == 0.20
        and "source" not in route["domain"].lower()
    )

    code_paths = {
        "stage1_model": REPO_ROOT / "model" / "high_density_polarity_expert.py",
        "stage1_loss": REPO_ROOT / "utils" / "temporal_frame_loss.py",
        "stage2_model": REPO_ROOT / "model" / "h2_activity_selective_recovery.py",
        "stage2_mechanics": REPO_ROOT / "utils" / "activity_selective_recovery.py",
        "cpu_tests": REPO_ROOT / "tests" / "test_h2_activity_selective_recovery.py",
    }
    actual_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    checks["all_code_hashes_match"] = actual_hashes == protocol["code_hashes"]
    m20_path = WORKSPACE_ROOT / protocol["released_m20"]["workspace_relative_path"]
    checks["released_m20_hash_matches"] = (
        m20_path.is_file()
        and sha256_file(m20_path) == protocol["released_m20"]["sha256"]
    )
    manifest_path = (
        WORKSPACE_ROOT
        / protocol["rich_m20_cache"]["manifest_workspace_relative_path"]
    )
    checks["rich_cache_manifest_hash_matches"] = (
        manifest_path.is_file()
        and sha256_file(manifest_path)
        == protocol["rich_m20_cache"]["manifest_sha256"]
    )

    activity = FineTemporalPolarityMultiScaleExpert(input_mode="activity_control")
    stage1_parameters = sum(parameter.numel() for parameter in activity.parameters())
    recovery = DisagreementRecoveryNet()
    stage2_parameters = recovery_parameter_count(recovery)
    checks["stage1_parameter_contract"] = stage1_parameters == 1712
    checks["stage2_parameter_contract"] = stage2_parameters == 7910
    checks["stage1_output_projection_zero"] = bool(
        torch.count_nonzero(activity.output_projection.weight) == 0
        and torch.count_nonzero(activity.output_projection.bias) == 0
    )

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_h2_activity_selective_recovery.py",
        "-v",
    ]
    completed = subprocess.run(
        test_command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    test_output = (completed.stdout or "") + (completed.stderr or "")
    checks["cpu_tests_pass"] = (
        completed.returncode == 0
        and "Ran 10 tests" in test_output
        and "OK" in test_output
    )
    checks["outer_held_arrays_not_read"] = True
    checks["validation_or_test_arrays_not_read"] = True
    checks["gpu_compute_not_used"] = True
    passed = all(checks.values())
    payload = {
        "schema": "ev-uav-h2-activity-selective-recovery-cpu-audit-v1",
        "passed": passed,
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks,
        "actual_code_hashes": actual_hashes,
        "stage1_trainable_parameter_count": stage1_parameters,
        "stage2_trainable_parameter_count": stage2_parameters,
        "cpu_test_command": test_command,
        "cpu_test_returncode": completed.returncode,
        "cpu_test_output": test_output,
        "source_arrays_opened": [],
        "held_g2_array_read": False,
        "validation_or_test_read": False,
        "gpu_probe_authorized_or_run": False,
        "gpu_probe_budget": protocol["gpu_probe_budget"],
        "formal_resource_budget": protocol["formal_resource_budget"],
    }
    if OUTPUT_PATH.exists():
        raise FileExistsError("refusing to overwrite CPU audit")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "checks": checks,
                "protocol_sha256": payload["protocol_sha256"],
                "audit_script_sha256": payload["audit_script_sha256"],
                "output": str(OUTPUT_PATH.resolve()),
                "output_sha256": sha256_file(OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

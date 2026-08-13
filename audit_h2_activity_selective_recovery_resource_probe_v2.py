"""CPU-only preflight for the activity+recovery resource probe v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

import run_h2_activity_selective_recovery_resource_probe_v2 as runner
from run_high_density_dual_expert_grouped_oof import tensor_state_sha256


RECEIPT_PATH = (
    runner.WORKSPACE_ROOT
    / "experiments"
    / "20260811_h2_activity_suppress_selective_recovery_g2_v1"
    / "cpu_resource_probe_v2_audit_passed.json"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_audit():
    protocol = runner.load_protocol()
    checkpoint = torch.load(runner.OLD_ACTIVITY_CHECKPOINT, map_location="cpu")
    provenance = checkpoint["provenance"]
    training_scope = provenance["training_scope"]
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_h2_activity_selective_recovery*.py",
        "-v",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = subprocess.run(
        test_command,
        cwd=runner.EVC_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    test_text = completed.stdout + completed.stderr
    checks = {
        "protocol_hash_matches": runner.sha256_file(runner.PROTOCOL_PATH)
        == runner.EXPECTED_PROTOCOL_SHA256,
        "helper_hash_matches": runner.sha256_file(runner.HELPER_RUNNER_PATH)
        == runner.EXPECTED_HELPER_RUNNER_SHA256,
        "base_science_hash_matches": runner.sha256_file(
            runner.BASE_SCIENCE_PROTOCOL_PATH
        )
        == runner.EXPECTED_BASE_SCIENCE_PROTOCOL_SHA256,
        "resource_only": protocol["role"]
        == "resource_and_mechanical_evidence_only",
        "not_formal_evidence": not protocol["formal_scientific_evidence"],
        "formal_blocked": protocol["base_science_protocol"][
            "formal_remains_blocked"
        ],
        "embedded_gpu_authorization_false": not protocol["execution"][
            "gpu_authorized"
        ],
        "fixed_source_train089": protocol["scope_firewall"][
            "source_arrays_allowed"
        ]
        == ["train_089.npz"],
        "held_g2_forbidden": not protocol["scope_firewall"][
            "held_g2_train_092_094_allowed"
        ],
        "validation_test_forbidden": not protocol["scope_firewall"][
            "validation_or_test_allowed"
        ],
        "stage1_zero_steps": protocol["resource_budget"][
            "stage1_optimizer_steps"
        ]
        == 0,
        "stage2_eight_steps": protocol["resource_budget"][
            "stage2_optimizer_steps"
        ]
        == 8,
        "checkpoint_hash_matches": runner.sha256_file(
            runner.OLD_ACTIVITY_CHECKPOINT
        )
        == protocol["frozen_old_activity_checkpoint"]["sha256"],
        "runtime_hash_matches": runner.sha256_file(runner.OLD_ACTIVITY_RUNTIME)
        == protocol["frozen_old_activity_checkpoint"]["runtime_result_sha256"],
        "checkpoint_activity_control": checkpoint["high_density_expert"][
            "input_mode"
        ]
        == "activity_control",
        "checkpoint_state_hash_matches": tensor_state_sha256(
            checkpoint["model_state_dict"]
        )
        == protocol["frozen_old_activity_checkpoint"][
            "source_final_model_state_sha256"
        ],
        "train089_old_held": "train_089.npz" in provenance["held_names"],
        "sealed_g2_old_fit_and_therefore_nonformal": all(
            name in provenance["fit_names"]
            for name in ("train_092.npz", "train_093.npz", "train_094.npz")
        )
        and not protocol["frozen_old_activity_checkpoint"][
            "formal_weight_initialization_allowed"
        ],
        "old_m20_marked_bitwise_frozen": training_scope[
            "inherited_m20_bitwise_frozen"
        ],
        "stage2_parameter_contract": protocol["stage2_resource_training"][
            "architecture_parameter_count"
        ]
        == 7910,
        "no_existing_gpu_output": not runner.OUTPUT_ROOT.exists(),
        "cuda_hidden_for_audit": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        and not torch.cuda.is_available(),
        "cpu_tests_passed": completed.returncode == 0
        and "Ran 18 tests" in test_text
        and "OK" in test_text,
    }
    failure_chain = protocol["prior_probe_failure_chain"]
    old_root = (
        runner.WORKSPACE_ROOT
        / "experiments"
        / "20260811_h2_activity_suppress_selective_recovery_g2_v1"
        / "resource_probe"
    )
    chain_files = {
        "failure": (
            old_root / "probe_failure.json",
            failure_chain["failure_receipt_sha256"],
        ),
        "postmortem": (
            old_root / "probe_postmortem_cpu.json",
            failure_chain["postmortem_sha256"],
        ),
        "stage1_checkpoint": (
            old_root / "stage1_train095_step8.pt",
            failure_chain["stage1_checkpoint_sha256"],
        ),
        "input": (
            old_root / "immutable_probe_input.npz",
            failure_chain["immutable_input_sha256"],
        ),
        "labels": (
            old_root / "immutable_fit_only_probe_labels.npz",
            failure_chain["immutable_fit_labels_sha256"],
        ),
    }
    for name, (path, expected) in chain_files.items():
        checks["old_failure_chain_{}_matches".format(name)] = (
            path.is_file() and runner.sha256_file(path) == expected
        )
    if not all(checks.values()):
        raise RuntimeError(
            "CPU audit failed: {}".format(
                [name for name, passed in checks.items() if not passed]
            )
        )
    return {
        "schema": "ev-uav-h2-activity-selective-recovery-resource-cpu-audit-v2",
        "created_utc": utc_now(),
        "status": "passed_awaiting_separate_gpu_authorization",
        "protocol_path": str(runner.PROTOCOL_PATH.resolve()),
        "protocol_sha256": runner.sha256_file(runner.PROTOCOL_PATH),
        "runner_path": str((runner.EVC_ROOT / "run_h2_activity_selective_recovery_resource_probe_v2.py").resolve()),
        "runner_sha256": runner.sha256_file(
            runner.EVC_ROOT / "run_h2_activity_selective_recovery_resource_probe_v2.py"
        ),
        "helper_runner_sha256": runner.sha256_file(runner.HELPER_RUNNER_PATH),
        "old_activity_checkpoint_sha256": runner.sha256_file(
            runner.OLD_ACTIVITY_CHECKPOINT
        ),
        "old_failure_chain": {
            name: {"path": str(path.resolve()), "sha256": expected}
            for name, (path, expected) in chain_files.items()
        },
        "checks": checks,
        "check_count": len(checks),
        "test_command": test_command,
        "test_summary": "18/18 passed",
        "source_arrays_opened": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "formal_started": False,
        "gpu_probe_started": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    result = run_audit()
    if args.write_receipt:
        runner.write_json_exclusive(RECEIPT_PATH, result)
        result["receipt_path"] = str(RECEIPT_PATH.resolve())
        result["receipt_sha256"] = runner.sha256_file(RECEIPT_PATH)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

"""CPU-only static and protocol audit for the fresh-G2 formal runner."""

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

import run_h2_activity_selective_recovery_fresh_g2_formal as runner


RECEIPT_PATH = runner.EXPERIMENT_ROOT / "cpu_audit_passed_v2.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_audit():
    protocol = runner.load_protocol()
    runner_path = Path(runner.__file__).resolve()
    runner_text = runner_path.read_text(encoding="utf-8")
    compile_command = [sys.executable, "-m", "py_compile", str(runner_path)]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    compile_result = subprocess.run(
        compile_command,
        cwd=runner.EVC_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
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
    test_result = subprocess.run(
        test_command,
        cwd=runner.EVC_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    test_text = test_result.stdout + test_result.stderr
    fit_sources = set(protocol["scope"]["inner_fit_sources"])
    held_sources = set(protocol["scope"]["sealed_outer_held_sources"])
    source_union = set().union(
        *[set(values) for values in protocol["source_groups"].values()]
    )
    forbidden = runner.forbidden_hashes(protocol)
    verify_index = runner_text.index("receipt, strategy, stage1_path, stage2_path = verify_training_chain(protocol)")
    held_mkdir_index = runner_text.index("HELD_OUTPUT.mkdir(parents=True, exist_ok=False)")
    open_receipt_index = runner_text.index("write_json_exclusive(\n        held_open_receipt")
    held_extract_index = runner_text.index("record = extract_source_label_free(", open_receipt_index)
    checks = {
        "protocol_hash_matches": runner.sha256_file(runner.PROTOCOL_PATH)
        == runner.EXPECTED_PROTOCOL_SHA256,
        "protocol_status_frozen": protocol["status"]
        == "frozen_cpu_implementation_awaiting_priority_and_gpu_authorization",
        "base_science_hash_matches": runner.sha256_file(
            runner.EVC_ROOT
            / "protocols"
            / "h2_activity_suppress_selective_recovery_g2_science_v1.json"
        )
        == protocol["base_science_protocol"]["sha256"],
        "runner_compiles": compile_result.returncode == 0,
        "all_activity_recovery_cpu_tests_pass": test_result.returncode == 0
        and "Ran 30 tests" in test_text
        and "OK" in test_text,
        "cuda_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        and not torch.cuda.is_available(),
        "fit_has_eight_sources": len(fit_sources) == 8,
        "sealed_g2_exact": held_sources
        == {"train_092.npz", "train_093.npz", "train_094.npz"},
        "fit_and_held_disjoint": not fit_sources & held_sources,
        "manifest_has_exact_eleven_sources": set(protocol["sources"])
        == source_union
        and len(source_union) == 11,
        "validation_test_forbidden": not protocol["scope"][
            "validation_or_test_allowed"
        ],
        "exact_h2_route": protocol["input_only_route"][
            "event_count_cutoff_exclusive"
        ]
        == 200000
        and protocol["input_only_route"]["polarity_minority_cutoff"] == 0.2,
        "stage1_g1_predicts_only_g3": protocol["stage1_true_oof"][
            "fit_g1_predict_only_g3"
        ],
        "stage1_g3_predicts_only_g1": protocol["stage1_true_oof"][
            "fit_g3_predict_only_g1"
        ],
        "own_group_stage1_features_forbidden": not protocol["stage1_true_oof"][
            "own_group_activity_feature_generation_allowed"
        ],
        "stage2_crossfit_both_directions": protocol["stage2_nested_oof"][
            "recovery_fit_g1_oof_predict_g3_oof"
        ]
        and protocol["stage2_nested_oof"][
            "recovery_fit_g3_oof_predict_g1_oof"
        ],
        "final_stage2_true_oof_only": protocol["final_fit_after_inner_pass"][
            "stage2"
        ].endswith("true-OOF disagreement features only"),
        "inner_score_and_recovery_gates": protocol["inner_cutoff_and_gate"][
            "each_group"
        ]["score_strictly_above_released_m20"]
        and protocol["inner_cutoff_and_gate"]["each_group"][
            "tp_or_co_strictly_above_stage1_activity"
        ]
        and protocol["inner_cutoff_and_gate"]["pooled"][
            "score_gain_at_least"
        ]
        == 0.01,
        "absolute_pd_gate_exact": protocol["inner_cutoff_and_gate"][
            "each_group"
        ]["absolute_pd_delta_at_most"]
        == 0.005,
        "breakpoints_not_grid": "every unique" in protocol[
            "inner_cutoff_and_gate"
        ]["candidate_cutoffs"],
        "old_resource_three_hashes_forbidden": forbidden
        == {
            "a73a037f09aadb478540aa40e50f9bbfbec73afb315b0fa54fca0b897b2a5b82",
            "ce5e223655abad5038073e9de165f726767793907c271d372e2faa550a80d9df",
            "874212511c09f034d20e3c14d7e28ecc2db39a1490b3b3ec861a1f607008deb1",
        },
        "forbidden_hash_literals_not_in_runner": not any(
            digest in runner_text for digest in forbidden
        ),
        "two_gpu_subcommands": protocol["execution"]["train_subcommand"]
        == "train-and-freeze"
        and protocol["execution"]["held_subcommand"]
        == "evaluate-held-g2-once",
        "hash_chain_before_held_directory": verify_index < held_mkdir_index,
        "held_open_receipt_before_first_extract": open_receipt_index
        < held_extract_index,
        "held_retry_forbidden": not protocol["outer_held_once"][
            "retry_or_reopen_after_any_result_or_failure"
        ],
        "gpu_authorization_not_embedded": not protocol["execution"][
            "gpu_authorized"
        ]
        and not protocol["execution"]["priority_authorized"],
        "pyramid_priority_wait_required": protocol["execution"][
            "wait_for_pyramid_recovery_capacity_result_before_priority"
        ],
        "fit_output_absent": not runner.FIT_OUTPUT.exists(),
        "held_output_absent": not runner.HELD_OUTPUT.exists(),
        "hard_peak_budget_3600": protocol["formal_step_and_resource_budget"][
            "hard_peak_cuda_mib"
        ]
        == 3600,
        "stage1_step_budget_exact": protocol["formal_step_and_resource_budget"][
            "inner_stage1_optimizer_steps_total"
        ]
        == 64
        and protocol["formal_step_and_resource_budget"][
            "final_union_stage1_optimizer_steps_if_inner_passes"
        ]
        == 64,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "formal CPU audit failed: {}".format(
                [name for name, passed in checks.items() if not passed]
            )
        )
    return {
        "schema": "ev-uav-activity-selective-recovery-fresh-g2-cpu-audit-v1",
        "created_utc": utc_now(),
        "status": "ready_awaiting_pyramid_priority_and_gpu_authorization",
        "protocol_path": str(runner.PROTOCOL_PATH.resolve()),
        "protocol_sha256": runner.sha256_file(runner.PROTOCOL_PATH),
        "runner_path": str(runner_path),
        "runner_sha256": runner.sha256_file(runner_path),
        "formal_helper_sha256": runner.sha256_file(
            runner.EVC_ROOT / "utils" / "activity_selective_recovery_formal.py"
        ),
        "checks": checks,
        "check_count": len(checks),
        "test_summary": "30/30 passed",
        "commands": {
            "train_after_separate_authorization": [
                sys.executable,
                str(runner_path),
                "train-and-freeze",
                runner.TRAIN_AUTH_FLAG,
            ],
            "held_only_after_inner_pass_and_separate_authorization": [
                sys.executable,
                str(runner_path),
                "evaluate-held-g2-once",
                runner.HELD_AUTH_FLAG,
            ],
        },
        "step_budget": protocol["formal_step_and_resource_budget"],
        "source_arrays_opened": [],
        "gpu_started": False,
        "formal_started": False,
        "held_g2_opened": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
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

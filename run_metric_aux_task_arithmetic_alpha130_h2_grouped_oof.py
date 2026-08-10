"""Fixed alpha=1.30, train-only H2 task-arithmetic follow-up.

The only new scored artifacts exposed by this runner are three formal
held-train inferences.  Released-M20 and alpha=1 sufficient-count artifacts
are immutable comparators and are never re-inferred.  No training, alpha
grid, threshold search, validation, or test command exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EVC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVC_ROOT.parent
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = EVC_ROOT / "tests" / "test_run_metric_aux_task_arithmetic_alpha130_h2_grouped_oof.py"

_PRIVATE_NAME = "_metric_aux_task_arithmetic_h2_grouped_oof_for_alpha130"
_PREVIOUS_PRIVATE = sys.modules.get(_PRIVATE_NAME)
_V5_PATH = EVC_ROOT / "run_metric_aux_task_arithmetic_h2_grouped_oof.py"
_V5_SPEC = importlib.util.spec_from_file_location(_PRIVATE_NAME, _V5_PATH)
if _V5_SPEC is None or _V5_SPEC.loader is None:
    raise ImportError("Unable to create a private alpha=1 grouped-OOF module.")
v5 = importlib.util.module_from_spec(_V5_SPEC)
sys.modules[_PRIVATE_NAME] = v5
try:
    _V5_SPEC.loader.exec_module(v5)
finally:
    if _PREVIOUS_PRIVATE is None:
        sys.modules.pop(_PRIVATE_NAME, None)
    else:
        sys.modules[_PRIVATE_NAME] = _PREVIOUS_PRIVATE

core = v5.core
_V4_LOAD_PROTOCOL = v5._V4_LOAD_PROTOCOL
_BASE_EVALUATE_SPEC = v5._BASE_EVALUATE_SPEC
_BASE_LOAD_EVALUATION_RESULT = v5._BASE_LOAD_EVALUATION_RESULT
_BASE_VERIFY_ASSETS = v5._BASE_VERIFY_ASSETS
_VERIFY_INFERENCE_HELPER = v5._verify_inference_helper
_STRICT_LOAD = v5._strict_model_state_load_cpu
_MODEL_STATE_SHA = v5.model_state_canonical_sha256

PROTOCOL_PATH = EVC_ROOT / "protocols" / "metric_aux_task_arithmetic_alpha130_h2_grouped_oof_science_v1.json"
EXPECTED_PROTOCOL_SHA256 = "50a20e36f08c6683f816458ba41b8357678e51c01913aa77484d9fa035df5406"
EXPECTED_SCHEMA = "ev-uav-temporal-memory-metric-aux-task-arithmetic-alpha130-h2-grouped-oof-v1"
ALPHA = 1.30
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "20260811_metric_aux_task_arithmetic_alpha130_h2_grouped_oof_v1"
COMMAND_AUDIT_PATH = OUTPUT_ROOT / "command_audit.json"
SYNTHESIS_ROOT = OUTPUT_ROOT / "synthesis"
SYNTHESIS_MANIFEST_PATH = SYNTHESIS_ROOT / "synthesis_manifest.json"
EVALUATION_ROOT = OUTPUT_ROOT / "held_train_evaluation"
REPORT_PATH = OUTPUT_ROOT / "grouped_oof_report.json"
GPU_AUTHORIZATION_FLAG = "--authorized-by-root"

V5_PROTOCOL_SHA256 = "4586e1e20a501a4b5219c2d5953a6666c1fc0d4ac08da2a7b29d6cea70938266"
V5_RUNNER_SHA256 = "26dc7b53402afa9681add0448ef9e87c5dc297c6361066512d241bfbe8142e86"
V5_TEST_SHA256 = "af980ff1649b5ab8816caa0babd5731a6271ad4b928b82988a4a73ade195de1d"
V5_AUDIT_SHA256 = "4792962006d1ca1a08f9fa9409e23074acf177849454c7d8c5429227603fa265"
V5_MANIFEST_SHA256 = "c87738e0d9e52945b2e95c05b06749898718482e3ddda65293540e175c701876"
V5_REPORT_SHA256 = "df4d8d6bf123e75ae1be7f9f53ef03442daf50f9683042fdfd0343356ce5110e"
M20_SHA256 = "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
EFFECTIVE_C00_SHA256 = "5dd5fa3c3bfa6f510eebdee0cd098fde9932015fcb819eb9c1f54cd4794f6413"
V4_PROTOCOL_SHA256 = v5.V4_PROTOCOL_SHA256
V4_RUNNER_SHA256 = v5.V4_RUNNER_SHA256

_V5_EVIDENCE_HASHES = {
    "protocol": V5_PROTOCOL_SHA256,
    "runner": V5_RUNNER_SHA256,
    "tests": V5_TEST_SHA256,
    "command_audit": V5_AUDIT_SHA256,
    "synthesis_manifest": V5_MANIFEST_SHA256,
    "report": V5_REPORT_SHA256,
}
_ALL11_EVIDENCE_HASHES = {
    "protocol": "e6493681b4265620966fb1a6ea400de69a3926ab71c9a0c493f82824922cbe92",
    "runner": "c5703b194ee51985593a919c66c72c9ca9f48f02bf1d6820c4745fda2cf4b43a",
    "tests": "b133b35309a45938c19481fa934fc5c2813e327f739fd32e5db6b4754d07e2dc",
    "command_audit": "882954a4671d29bdbfff99ad12293ed6be12408e761ef45bfeb9cc4f88a4b821",
    "pair_audit": "c3c53857a451f9f9df2ea9caa0df4f023fe4c78647009594b054a861c3f44c19",
    "alpha1_checkpoint": "614999c09f82ec1911620ee35dae3f1f6362cb92d59a82e2e539e9b2ad2432ee",
    "synthesis_manifest": "7c23283720b046760463d91a4e68137803bcb448425d5ed88961b72f2a23caa1",
}


def _expect(actual, expected, label):
    core._expect_equal(actual, expected, label)


def _close(actual, expected, label, rel=2e-12, abs_tol=2e-15):
    if not math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol):
        raise RuntimeError("{} differs: {} != {}".format(label, actual, expected))


def _require_bound_file(record, expected_sha256, label):
    path = core.workspace_path(record["workspace_relative_path"])
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    digest = core.sha256_file(path)
    if digest != expected_sha256 or digest != record["sha256"]:
        raise RuntimeError("{} SHA-256 differs from the frozen contract.".format(label))
    return path


def _load_json_record(record, expected_sha256, label):
    path = _require_bound_file(record, expected_sha256, label)
    payload, digest = core.load_json_snapshot(path)
    _expect(digest, expected_sha256, "{} snapshot".format(label))
    return payload, digest, path


def _validate_overlay(overlay, digest):
    _expect(digest, EXPECTED_PROTOCOL_SHA256, "alpha130 protocol SHA-256")
    _expect(overlay.get("schema"), EXPECTED_SCHEMA, "alpha130 schema")
    _expect(
        overlay.get("status"),
        "frozen_after_alpha1_train_only_oof_and_all11_geometry_before_any_alpha130_checkpoint_or_inference",
        "alpha130 status",
    )
    candidate = overlay["fixed_candidate"]
    _expect(candidate.get("alpha"), ALPHA, "fixed alpha")
    _expect(candidate.get("candidate_count"), 1, "candidate count")
    _expect(candidate.get("new_training_optimizer_steps"), 0, "training steps")
    _expect(candidate.get("new_formal_candidate_inference_count"), 3, "inference count")
    for key in (
        "alpha_grid_allowed",
        "module_mask_or_projection_allowed",
        "threshold_search_allowed",
        "weight_or_c00_search_allowed",
        "cross_fold_vector_merge_allowed",
    ):
        _expect(candidate.get(key), False, "candidate {}".format(key))
    _expect(overlay["held_train_inference_contract"]["dataset_split"], "train", "split")
    _expect(overlay["held_train_inference_contract"]["prediction_threshold"], 0.719, "threshold")
    _expect(
        overlay["held_train_inference_contract"]["effective_c00_canonical_sha256"],
        EFFECTIVE_C00_SHA256,
        "C00 identity",
    )
    _expect(overlay["cli_contract"]["allowed_commands"], ["audit", "synthesize", "evaluate", "report", "all-evaluate-report"], "CLI")
    for key in (
        "validation_or_test_read_allowed",
        "current_failed_validation_report_read_allowed",
        "persistence_formal_artifact_read_allowed",
        "new_training_allowed",
        "platform_submission_allowed",
    ):
        _expect(overlay.get(key), False, key)
    for name, expected in _V5_EVIDENCE_HASHES.items():
        _require_bound_file(overlay["alpha1_train_only_evidence"][name], expected, "alpha1 {}".format(name))
    for anchor, records in overlay["anchor_sufficient_count_artifacts"].items():
        for fold_id, record in records.items():
            _require_bound_file(record, record["sha256"], "{} {} count artifact".format(anchor, fold_id))
    for name, expected in _ALL11_EVIDENCE_HASHES.items():
        _require_bound_file(overlay["all11_geometry_evidence"][name], expected, "all11 {}".format(name))
    inputs = overlay["input_checkpoint_contract"]
    _require_bound_file(inputs["released_m20"], M20_SHA256, "released M20")
    for fold_id, pair in inputs["fold_pairs"].items():
        for arm in ("baseline_e3", "metric_aux_e3"):
            _require_bound_file(pair[arm], pair[arm]["sha256"], "{} {}".format(fold_id, arm))
    for arm in ("baseline_e3", "metric_aux_e3"):
        record = inputs["all11_pair"][arm]
        _require_bound_file(record, record["sha256"], "all11 {}".format(arm))


def load_protocol():
    actual = core.sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Alpha130 protocol SHA-256 {} differs from frozen {}.".format(actual, EXPECTED_PROTOCOL_SHA256))
    overlay, snapshot_sha = core.load_json_snapshot(PROTOCOL_PATH)
    _validate_overlay(overlay, snapshot_sha)

    # Load only the pre-evaluation V4 scientific definition here.  Existing
    # alpha=1/M20 sufficient counts are deferred until report construction.
    effective_v4, inherited_sha = _V4_LOAD_PROTOCOL()
    _expect(inherited_sha, V4_PROTOCOL_SHA256, "inherited V4 protocol")
    _expect(effective_v4["evaluation"]["prediction_threshold"], 0.719, "inherited threshold")
    _expect(effective_v4["evaluation"]["effective_c00_canonical_sha256"], EFFECTIVE_C00_SHA256, "inherited C00")
    effective = copy.deepcopy(effective_v4)
    effective["status"] = overlay["status"]
    effective["experiment_id"] = overlay["experiment_id"]
    effective["evidence_class"] = overlay["evidence_class"]
    effective["audit_amendment"]["claim_scope"] = overlay["claim_scope"]
    effective["alpha130_contract"] = overlay
    effective["outputs"]["workspace_relative_directory"] = overlay["output_contract"]["workspace_relative_directory"]
    return effective, actual


def _checkpoint_path(record):
    return core.workspace_path(record["workspace_relative_path"])


def synthesis_specs(protocol):
    contract = protocol["alpha130_contract"]["input_checkpoint_contract"]
    output = []
    for fold in protocol["dataset"]["folds"]:
        fold_id = fold["fold_id"]
        pair = contract["fold_pairs"][fold_id]
        output.append(
            {
                "fold_id": fold_id,
                "held_group": fold["held_group"],
                "parent": str(_checkpoint_path(contract["released_m20"]).resolve()),
                "parent_sha256": contract["released_m20"]["sha256"],
                "baseline": str(_checkpoint_path(pair["baseline_e3"]).resolve()),
                "baseline_sha256": pair["baseline_e3"]["sha256"],
                "metric_aux": str(_checkpoint_path(pair["metric_aux_e3"]).resolve()),
                "metric_aux_sha256": pair["metric_aux_e3"]["sha256"],
                "output": str((SYNTHESIS_ROOT / fold_id / "isolated_metric_aux_alpha130.pt").resolve()),
            }
        )
    _expect([item["fold_id"] for item in output], ["hold_g1", "hold_g2", "hold_g3"], "fold order")
    return output


def _load_checkpoint(path):
    return core.load_torch_checkpoint(Path(path))


def _state(payload):
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model_state_dict mapping.")
    return state


def _name_shape_canonical_sha256(state):
    items = [
        {"name": name, "shape": list(value.shape), "numel": int(value.numel()), "dtype": str(value.dtype)}
        for name, value in sorted(state.items())
    ]
    raw = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), items


def synthesize_state_dict(parent, baseline, metric_aux, alpha=ALPHA):
    import torch

    if list(parent) != list(baseline) or list(parent) != list(metric_aux):
        raise RuntimeError("Task-arithmetic state-dict key order differs.")
    result = {}
    for name in parent:
        p, b, a = parent[name], baseline[name], metric_aux[name]
        if p.shape != b.shape or p.shape != a.shape or p.dtype != b.dtype or p.dtype != a.dtype or p.dtype != torch.float32:
            raise RuntimeError("Task-arithmetic tensor metadata differs: {}".format(name))
        value = (p.detach().cpu().to(torch.float64) + float(alpha) * (a.detach().cpu().to(torch.float64) - b.detach().cpu().to(torch.float64))).to(p.dtype).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Task-arithmetic output is non-finite: {}".format(name))
        result[name] = value
    return result


def _state_equal(left, right):
    import torch

    return list(left) == list(right) and all(
        left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and torch.equal(left[name], right[name])
        for name in left
    )


def _independent_formula_equal(parent, baseline, metric_aux, candidate, alpha):
    import torch

    for name in parent:
        expected = torch.add(
            parent[name].detach().cpu().to(torch.float64),
            torch.sub(metric_aux[name].detach().cpu().to(torch.float64), baseline[name].detach().cpu().to(torch.float64)),
            alpha=float(alpha),
        ).to(parent[name].dtype).contiguous()
        if not torch.equal(expected, candidate[name]):
            return False
    return True


def _delta(left, right):
    import torch

    if list(left) != list(right):
        raise RuntimeError("Model-state key order differs.")
    squared = base_squared = 0.0
    maximum = 0.0
    changed_tensors = changed_elements = 0
    vectors = {}
    module_squared = {}
    finite = True
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            raise RuntimeError("Model-state metadata differs: {}".format(name))
        value = (left[name].detach().cpu().to(torch.float64) - right[name].detach().cpu().to(torch.float64)).reshape(-1)
        vectors[name] = value
        square = float(torch.sum(value * value))
        squared += square
        base = right[name].detach().cpu().to(torch.float64)
        base_squared += float(torch.sum(base * base))
        maximum = max(maximum, float(torch.max(torch.abs(value))) if value.numel() else 0.0)
        count = int(torch.count_nonzero(value))
        changed_tensors += int(count > 0)
        changed_elements += count
        finite = finite and bool(torch.isfinite(value).all())
        module = name.split(".")[0]
        module_squared[module] = module_squared.get(module, 0.0) + square
    norm = math.sqrt(squared)
    return {
        "l2": norm,
        "max_abs": maximum,
        "relative_l2": norm / math.sqrt(base_squared),
        "base_l2": math.sqrt(base_squared),
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "finite": finite,
        "module_energy_share": {name: value / squared for name, value in module_squared.items()} if squared else {},
        "vectors": vectors,
    }


def _cosine(left, right):
    import torch

    if list(left) != list(right):
        raise RuntimeError("Task-vector key order differs.")
    dot = left_sq = right_sq = 0.0
    for name in left:
        dot += float(torch.sum(left[name] * right[name]))
        left_sq += float(torch.sum(left[name] * left[name]))
        right_sq += float(torch.sum(right[name] * right[name]))
    if left_sq <= 0.0 or right_sq <= 0.0:
        raise RuntimeError("Cannot compute cosine for a zero vector.")
    return dot / math.sqrt(left_sq * right_sq)


def _public_stats(stats):
    return {name: value for name, value in stats.items() if name != "vectors"}


def _validate_state_contract(state, contract, label):
    digest, items = _name_shape_canonical_sha256(state)
    _expect(digest, contract["name_shape_canonical_sha256"], "{} name/shape".format(label))
    _expect(len(items), contract["ordered_key_count"], "{} keys".format(label))
    _expect(sum(item["numel"] for item in items), contract["element_count"], "{} elements".format(label))
    if any(item["dtype"] != contract["dtype_each"] for item in items):
        raise RuntimeError("{} contains a non-float32 tensor.".format(label))


def _all11_geometry(protocol, parent_state, fold_vectors):
    contract = protocol["alpha130_contract"]
    inputs = contract["input_checkpoint_contract"]["all11_pair"]
    baseline = _state(_load_checkpoint(_checkpoint_path(inputs["baseline_e3"])))
    aux = _state(_load_checkpoint(_checkpoint_path(inputs["metric_aux_e3"])))
    task = _delta(aux, baseline)
    candidate = synthesize_state_dict(parent_state, baseline, aux, ALPHA)
    applied = _delta(candidate, parent_state)
    evidence = contract["all11_geometry_evidence"]
    _close(task["l2"], evidence["alpha1_task_l2"], "all11 raw task L2")
    _close(applied["l2"], evidence["alpha130_applied_l2"], "all11 applied L2")
    _close(applied["l2"] / math.sqrt(264.0), evidence["alpha130_step_normalized_l2"], "all11 step-normalized")
    _close(applied["relative_l2"], evidence["alpha130_task_over_m20"], "all11 task/M20")
    _close(applied["max_abs"], evidence["alpha130_max_abs"], "all11 max abs")
    _expect(applied["changed_tensor_count"], evidence["changed_tensor_count"], "all11 changed tensors")
    cosines = {fold_id: _cosine(task["vectors"], vector) for fold_id, vector in fold_vectors.items()}
    for fold_id, expected in evidence["cosine_with_fold_tasks"].items():
        _close(cosines[fold_id], expected, "all11 {} cosine".format(fold_id))
    for name, expected in evidence["module_energy_share"].items():
        _close(task["module_energy_share"][name], expected, "all11 {} module share".format(name))
    gates = contract["geometry_gates"]
    lower, upper = gates["alpha130_step_normalized_l2_inclusive_bounds"]
    checks = {
        "step_normalized_in_bounds": float(lower) <= applied["l2"] / math.sqrt(264.0) <= float(upper),
        "task_over_m20_below_cap": applied["relative_l2"] <= float(gates["alpha130_task_over_m20_maximum"]),
        "max_abs_below_cap": applied["max_abs"] <= float(gates["alpha130_max_abs_maximum"]),
        "changed_tensor_count_exact": applied["changed_tensor_count"] == int(gates["changed_tensor_count_exact"]),
        "cosine_floor_each_fold": all(value >= float(gates["cosine_with_each_fold_task_minimum"]) for value in cosines.values()),
        "module_shares_in_bounds": set(task["module_energy_share"]) == set(gates["module_energy_share_bounds"]) and all(
            float(bounds[0]) <= task["module_energy_share"][name] <= float(bounds[1])
            for name, bounds in gates["module_energy_share_bounds"].items()
        ),
        "finite_nonzero": task["finite"] and applied["finite"] and task["l2"] > 0.0 and applied["l2"] > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError("All11 alpha130 geometry gate failed: {}".format(checks))
    return {
        "raw_task": _public_stats(task),
        "applied_task": _public_stats(applied),
        "step_normalized_applied_l2": applied["l2"] / math.sqrt(264.0),
        "cosines_with_fold_tasks": cosines,
        "checks": checks,
        "passed": True,
    }


def task_arithmetic_preflight(protocol):
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before alpha130 CPU preflight.")
    contract = protocol["alpha130_contract"]["input_checkpoint_contract"]["model_state_dict"]
    records = []
    fold_vectors = {}
    parent_state = None
    for spec in synthesis_specs(protocol):
        for key in ("parent", "baseline", "metric_aux"):
            _expect(core.sha256_file(Path(spec[key])), spec["{}_sha256".format(key)], "{} {} SHA".format(spec["fold_id"], key))
        payloads = [_load_checkpoint(spec[key]) for key in ("parent", "baseline", "metric_aux")]
        states = [_state(payload) for payload in payloads]
        for label, state in zip(("parent", "baseline", "metric_aux"), states):
            _validate_state_contract(state, contract, "{} {}".format(spec["fold_id"], label))
        parent_state = states[0]
        alpha_zero = synthesize_state_dict(*states, alpha=0.0)
        candidate = synthesize_state_dict(*states, alpha=ALPHA)
        if not _state_equal(alpha_zero, states[0]):
            raise RuntimeError("Alpha=0 parent identity failed: {}".format(spec["fold_id"]))
        if not _independent_formula_equal(*states, candidate, ALPHA):
            raise RuntimeError("Alpha=1.30 independent formula failed: {}".format(spec["fold_id"]))
        raw = _delta(states[2], states[1])
        applied = _delta(candidate, states[0])
        fold_vectors[spec["fold_id"]] = raw["vectors"]
        records.append(
            {
                "fold_id": spec["fold_id"],
                "input_model_state_sha256": {name: _MODEL_STATE_SHA(state) for name, state in zip(("parent", "baseline", "metric_aux"), states)},
                "candidate_model_state_sha256": _MODEL_STATE_SHA(candidate),
                "alpha_zero_parent_identity": True,
                "alpha130_independent_formula_bitwise": True,
                "raw_task": _public_stats(raw),
                "applied_task": _public_stats(applied),
                "strict_model_load_cpu": _STRICT_LOAD(candidate),
            }
        )
    all11 = _all11_geometry(protocol, parent_state, fold_vectors)
    return {"records": records, "all11_geometry": all11, "cuda_not_initialized": not torch.cuda.is_initialized(), "passed": True}


def _write_json_exclusive(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite JSON evidence: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=".{}-".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except Exception:
        if path.exists() and path.samefile(temporary):
            path.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _atomic_torch_save_exclusive(payload, destination):
    import torch

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite synthesis output: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".{}-".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _nested_equal(left, right):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def command_audit_payload(protocol, protocol_sha, assets, preflight):
    specs = synthesis_specs(protocol)
    runner_sha = core.sha256_file(RUNNER_PATH)
    if not TEST_PATH.is_file():
        raise FileNotFoundError("Alpha130 CPU test file is missing: {}".format(TEST_PATH))
    return {
        "schema": "ev-uav-metric-aux-task-arithmetic-alpha130-command-audit-v1",
        "created_utc": core.utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": runner_sha,
        "tests_path": str(TEST_PATH),
        "tests_sha256": core.sha256_file(TEST_PATH),
        "assets": assets,
        "task_arithmetic_preflight": preflight,
        "synthesis_specs": specs,
        "synthesis_command": [sys.executable, str(RUNNER_PATH), "synthesize"],
        "evaluation_commands": {
            "{}_alpha130".format(spec["fold_id"]): [
                sys.executable,
                str(RUNNER_PATH),
                "evaluate",
                "--eval-id",
                "{}_alpha130".format(spec["fold_id"]),
                GPU_AUTHORIZATION_FLAG,
            ]
            for spec in specs
        },
        "report_command": [sys.executable, str(RUNNER_PATH), "report"],
        "allowed_cli_commands": ["audit", "synthesize", "evaluate", "report", "all-evaluate-report"],
        "alpha": ALPHA,
        "candidate_count": 1,
        "new_training_optimizer_steps": 0,
        "new_candidate_evaluation_count": 3,
        "released_m20_anchor_reinference": False,
        "alpha1_anchor_reinference": False,
        "alpha_threshold_c00_or_module_grid": False,
        "estimated_gpu_elapsed_seconds_for_three_formal_inferences": [60, 120],
        "existing_anchor_content_load_deferred_until_report": True,
        "gpu_or_cuda_initialized": False,
        "data_use_statement": (
            "CPU audit read only frozen science definitions and train-derived checkpoints; existing train-only anchor files were hash-bound but their counts are deferred to report. "
            "No validation/test dataset, failed validation report, persistence artifact, or GPU was read or used."
        ),
    }


def run_audit():
    protocol, protocol_sha = load_protocol()
    assets = _BASE_VERIFY_ASSETS(protocol)
    preflight = task_arithmetic_preflight(protocol)
    payload = command_audit_payload(protocol, protocol_sha, assets, preflight)
    _write_json_exclusive(COMMAND_AUDIT_PATH, payload)
    print("alpha130 immutable CPU audit:", COMMAND_AUDIT_PATH)
    print("command audit sha256:", core.sha256_file(COMMAND_AUDIT_PATH))
    print("GPU remains unauthorized; synthesis is CPU-only.")
    return payload


def load_command_audit():
    if not COMMAND_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the immutable alpha130 CPU audit first.")
    payload, digest = core.load_json_snapshot(COMMAND_AUDIT_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-alpha130-command-audit-v1", "audit schema")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "audit protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "audit runner")
    _expect(payload.get("tests_sha256"), core.sha256_file(TEST_PATH), "audit tests")
    _expect(payload.get("alpha"), ALPHA, "audit alpha")
    _expect(payload.get("candidate_count"), 1, "audit candidate count")
    _expect(payload.get("new_training_optimizer_steps"), 0, "audit training steps")
    _expect(payload.get("new_candidate_evaluation_count"), 3, "audit inference count")
    _expect(payload.get("released_m20_anchor_reinference"), False, "M20 reinference")
    _expect(payload.get("alpha1_anchor_reinference"), False, "alpha1 reinference")
    _expect(payload.get("gpu_or_cuda_initialized"), False, "audit CUDA")
    _expect(payload.get("task_arithmetic_preflight", {}).get("passed"), True, "audit preflight")
    return payload, digest


def run_synthesis():
    import torch

    protocol, _ = load_protocol()
    _, audit_sha = load_command_audit()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before alpha130 CPU synthesis.")
    specs = synthesis_specs(protocol)
    if SYNTHESIS_MANIFEST_PATH.exists() or any(Path(spec["output"]).exists() for spec in specs):
        raise FileExistsError("Refusing to overwrite alpha130 synthesis evidence.")
    records = []
    for spec in specs:
        before = {key: core.sha256_file(Path(spec[key])) for key in ("parent", "baseline", "metric_aux")}
        for key, digest in before.items():
            _expect(digest, spec["{}_sha256".format(key)], "{} {} before".format(spec["fold_id"], key))
        payloads = [_load_checkpoint(spec[key]) for key in ("parent", "baseline", "metric_aux")]
        states = [_state(payload) for payload in payloads]
        candidate = synthesize_state_dict(*states, alpha=ALPHA)
        alpha_zero = synthesize_state_dict(*states, alpha=0.0)
        if not _state_equal(alpha_zero, states[0]):
            raise RuntimeError("Alpha=0 identity failed during synthesis.")
        if not _independent_formula_equal(*states, candidate, ALPHA):
            raise RuntimeError("Alpha=1.30 independent formula failed during synthesis.")
        output_payload = {
            "model_state_dict": candidate,
            "temporal_memory": copy.deepcopy(payloads[0]["temporal_memory"]),
            "provenance": {
                "artifact_kind": "inference_only_metric_aux_task_arithmetic_alpha130",
                "fold_id": spec["fold_id"],
                "formula": "released_m20 + 1.30 * (metric_aux_e3 - baseline_e3)",
                "alpha": ALPHA,
                "arithmetic_dtype": "torch.float64_then_single_cast_to_parent_torch.float32",
                "input_checkpoint_sha256": before,
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": core.sha256_file(RUNNER_PATH),
                "command_audit_sha256": audit_sha,
                "new_training_optimizer_steps": 0,
                "optimizer_scheduler_rng_copied": False,
            },
        }
        _atomic_torch_save_exclusive(output_payload, spec["output"])
        reloaded = _load_checkpoint(spec["output"])
        if set(reloaded) != {"model_state_dict", "temporal_memory", "provenance"}:
            raise RuntimeError("Inference-only checkpoint has unexpected top-level keys.")
        if not _state_equal(_state(reloaded), candidate):
            raise RuntimeError("Reloaded alpha130 model state differs.")
        if not _nested_equal(reloaded["temporal_memory"], payloads[0]["temporal_memory"]):
            raise RuntimeError("Released-M20 temporal-memory metadata was not preserved.")
        after = {key: core.sha256_file(Path(spec[key])) for key in ("parent", "baseline", "metric_aux")}
        if before != after:
            raise RuntimeError("A synthesis input changed.")
        records.append(
            {
                "fold_id": spec["fold_id"],
                "output_path": spec["output"],
                "output_sha256": core.sha256_file(Path(spec["output"])),
                "model_state_canonical_sha256": _MODEL_STATE_SHA(candidate),
                "input_checkpoint_sha256": before,
                "input_before_after_equal": True,
                "alpha_zero_parent_identity": True,
                "alpha130_independent_formula_bitwise": True,
                "strict_model_load_cpu": _STRICT_LOAD(candidate),
                "strict_reload_bitwise": True,
                "temporal_memory_metadata_source": "released_m20",
                "top_level_keys_exact": ["model_state_dict", "provenance", "temporal_memory"],
                "optimizer_scheduler_rng_copied": False,
                "applied_task": _public_stats(_delta(candidate, states[0])),
            }
        )
    manifest = {
        "schema": "ev-uav-metric-aux-task-arithmetic-alpha130-synthesis-manifest-v1",
        "created_utc": core.utc_now(),
        "status": "completed",
        "passed": True,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "tests_sha256": core.sha256_file(TEST_PATH),
        "command_audit_sha256": audit_sha,
        "alpha": ALPHA,
        "records": records,
        "new_training_optimizer_steps": 0,
        "evaluation_or_score_run": False,
        "anchor_content_read": False,
        "validation_or_test_read": False,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    _write_json_exclusive(SYNTHESIS_MANIFEST_PATH, manifest)
    print("alpha130 synthesis manifest:", SYNTHESIS_MANIFEST_PATH)
    print("manifest sha256:", core.sha256_file(SYNTHESIS_MANIFEST_PATH))
    return manifest


def load_synthesis_manifest(verify_formula=True):
    import torch

    payload, digest = core.load_json_snapshot(SYNTHESIS_MANIFEST_PATH)
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-alpha130-synthesis-manifest-v1", "manifest schema")
    _expect(payload.get("passed"), True, "manifest passed")
    _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "manifest protocol")
    _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "manifest runner")
    _expect(payload.get("tests_sha256"), core.sha256_file(TEST_PATH), "manifest tests")
    _expect(payload.get("alpha"), ALPHA, "manifest alpha")
    _, audit_sha = load_command_audit()
    _expect(payload.get("command_audit_sha256"), audit_sha, "manifest audit")
    by_fold = {record["fold_id"]: record for record in payload.get("records", [])}
    _expect(set(by_fold), {"hold_g1", "hold_g2", "hold_g3"}, "manifest folds")
    protocol, _ = load_protocol()
    for spec in synthesis_specs(protocol):
        record = by_fold[spec["fold_id"]]
        output_path = Path(record["output_path"])
        _expect(output_path.resolve(), Path(spec["output"]).resolve(), "manifest output path")
        _expect(core.sha256_file(output_path), record["output_sha256"], "manifest output SHA")
        if verify_formula:
            states = [_state(_load_checkpoint(spec[key])) for key in ("parent", "baseline", "metric_aux")]
            checkpoint = _load_checkpoint(output_path)
            _expect(set(checkpoint), {"model_state_dict", "temporal_memory", "provenance"}, "checkpoint top-level keys")
            if not _independent_formula_equal(*states, _state(checkpoint), ALPHA):
                raise RuntimeError("Manifest checkpoint formula differs.")
            _expect(checkpoint["provenance"]["input_checkpoint_sha256"], record["input_checkpoint_sha256"], "checkpoint inputs")
            _expect(_MODEL_STATE_SHA(_state(checkpoint)), record["model_state_canonical_sha256"], "checkpoint model-state SHA")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU synthesis verification.")
    return payload, digest


def evaluation_specs(protocol, manifest):
    by_fold = {record["fold_id"]: record for record in manifest["records"]}
    output = []
    for fold in protocol["dataset"]["folds"]:
        fold_id = fold["fold_id"]
        record = by_fold[fold_id]
        output.append(
            {
                "eval_id": "{}_alpha130".format(fold_id),
                "fold_id": fold_id,
                "variant": "isolated_metric_aux_alpha130",
                "held_group": fold["held_group"],
                "held_source_names": [item["name"] for item in core.held_items(protocol, fold)],
                "checkpoint": str(Path(record["output_path"]).resolve()),
                "checkpoint_sha256": record["output_sha256"],
                "training_result_path": None,
                "result_path": str((EVALUATION_ROOT / "{}_alpha130".format(fold_id) / "evaluation.json").resolve()),
            }
        )
    _expect([item["eval_id"] for item in output], ["hold_g1_alpha130", "hold_g2_alpha130", "hold_g3_alpha130"], "evaluation plan")
    return output


def _immutable_inference_asset_hashes(spec, memory_inference, helper):
    helper_module = sys.modules[helper.__module__]
    paths = {
        "checkpoint": Path(spec["checkpoint"]).resolve(),
        "runner": RUNNER_PATH,
        "protocol": PROTOCOL_PATH,
        "memory_inference": Path(memory_inference.__file__).resolve(),
        "frame_inference": Path(helper_module.__file__).resolve(),
        "config": (EVC_ROOT / "configs" / "evisseg_evuav.yaml").resolve(),
    }
    return {name: {"path": str(path), "sha256": core.sha256_file(path)} for name, path in paths.items()}


def _alpha130_recovery_record(manifest_sha, before_after):
    return {
        "mode": "fixed_metric_aux_task_arithmetic_alpha130_v1",
        "formula": "released_m20 + 1.30 * (metric_aux_e3 - baseline_e3)",
        "alpha": ALPHA,
        "synthesis_manifest_sha256": manifest_sha,
        "alpha1_report_sha256": V5_REPORT_SHA256,
        "new_training_optimizer_steps": 0,
        "adaptive_train_only_not_independent_oof": True,
        "released_m20_anchor_reinference": False,
        "alpha1_anchor_reinference": False,
        "immutable_inference_assets_before_after": before_after,
    }


def evaluate_spec(protocol, spec, manifest_sha):
    memory_inference, helper = _VERIFY_INFERENCE_HELPER(protocol)
    original_writer = core.write_new_json
    before = _immutable_inference_asset_hashes(spec, memory_inference, helper)

    def guarded_writer(path, payload):
        _expect(Path(path).resolve(), Path(spec["result_path"]).resolve(), "candidate output path")
        _expect(payload.get("schema"), "ev-uav-metric-aux-held-train-evaluation-v1", "candidate schema")
        _expect(payload.get("eval_id"), spec["eval_id"], "candidate eval ID")
        _expect(payload.get("fold_id"), spec["fold_id"], "candidate fold")
        _expect(payload.get("variant"), spec["variant"], "candidate variant")
        _expect(payload.get("dataset_split"), "train", "candidate split")
        _expect(payload.get("protocol_sha256"), EXPECTED_PROTOCOL_SHA256, "candidate protocol")
        _expect(payload.get("runner_sha256"), core.sha256_file(RUNNER_PATH), "candidate runner")
        _expect(payload.get("checkpoint_sha256"), spec["checkpoint_sha256"], "candidate checkpoint")
        _expect(payload.get("prediction_threshold"), 0.719, "candidate threshold")
        _expect(payload.get("effective_c00_canonical_sha256"), EFFECTIVE_C00_SHA256, "candidate C00")
        _expect(payload.get("t32_read_or_combined"), False, "candidate T32")
        after = _immutable_inference_asset_hashes(spec, memory_inference, helper)
        if before != after:
            raise RuntimeError("An immutable inference asset changed during evaluation.")
        payload["alpha130_recovery"] = _alpha130_recovery_record(
            manifest_sha,
            {"before": before, "after": after, "equal": True},
        )
        return original_writer(path, payload)

    if hasattr(memory_inference, "temporal_frame_video_from_sample"):
        raise RuntimeError("Alpha130 helper injection target was not clean.")
    setattr(memory_inference, "temporal_frame_video_from_sample", helper)
    core.write_new_json = guarded_writer
    try:
        payload = _BASE_EVALUATE_SPEC(protocol, spec)
    finally:
        core.write_new_json = original_writer
        if hasattr(memory_inference, "temporal_frame_video_from_sample"):
            delattr(memory_inference, "temporal_frame_video_from_sample")
    if hasattr(memory_inference, "temporal_frame_video_from_sample"):
        raise RuntimeError("Alpha130 helper injection survived finally.")
    _expect(payload.get("alpha130_recovery", {}).get("alpha"), ALPHA, "candidate recovery alpha")
    return payload


def _count_tools():
    from crossfit_component_reranker import SufficientCounts, metrics_from_counts

    return SufficientCounts, metrics_from_counts


def _validate_count_dict(counts, label):
    if set(counts) != set(core.COUNT_FIELDS):
        raise RuntimeError("{} count fields differ.".format(label))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise RuntimeError("{} contains a non-integer or negative count.".format(label))


def _validate_evaluation_payload(payload, spec, protocol, expected_protocol_sha, expected_runner_sha, expected_checkpoint_sha, recovery_kind):
    SufficientCounts, metrics_from_counts = _count_tools()
    expected = {
        "schema": "ev-uav-metric-aux-held-train-evaluation-v1",
        "eval_id": spec["eval_id"],
        "fold_id": spec["fold_id"],
        "variant": spec["variant"],
        "held_group": spec["held_group"],
        "dataset_split": "train",
        "held_stream": "complete_full_stream_t160",
        "t32_read_or_combined": False,
        "checkpoint_sha256": expected_checkpoint_sha,
        "prediction_threshold": 0.719,
        "effective_c00_canonical_sha256": EFFECTIVE_C00_SHA256,
        "protocol_sha256": expected_protocol_sha,
        "runner_sha256": expected_runner_sha,
    }
    for key, value in expected.items():
        _expect(payload.get(key), value, "{} {}".format(spec["eval_id"], key))
    if spec.get("checkpoint") is not None:
        _expect(Path(payload.get("checkpoint", "")).resolve(), Path(spec["checkpoint"]).resolve(), "{} checkpoint path".format(spec["eval_id"]))
    _expect(payload.get("config_overrides"), protocol["evaluation"]["fixed_config_overrides"], "{} overrides".format(spec["eval_id"]))
    names = [record["source_name"] for record in payload["records"]]
    _expect(names, spec["held_source_names"], "{} source order".format(spec["eval_id"]))
    source_contract = core.source_index(protocol)
    record_counts = []
    for record in payload["records"]:
        source = source_contract[record["source_name"]]
        _expect(record.get("source_sha256"), source["sha256"], "record source SHA")
        _validate_count_dict(record["counts"], "{} {}".format(spec["eval_id"], record["source_name"]))
        _expect(record.get("event_count"), record["counts"]["event_count"], "record event count")
        if "event_count" in source:
            _expect(record["counts"]["event_count"], int(source["event_count"]), "frozen event count")
        if "positive_event_count" in source:
            _expect(
                record["counts"]["true_positive_events"] + record["counts"]["false_negative_events"],
                int(source["positive_event_count"]),
                "frozen positive count",
            )
        _expect(metrics_from_counts(SufficientCounts(**record["counts"])), record["metrics"], "record metrics")
        record_counts.append(record["counts"])
    _validate_count_dict(payload["pooled_counts"], "{} pooled".format(spec["eval_id"]))
    _expect(core.add_count_dicts(record_counts), payload["pooled_counts"], "{} record pooling".format(spec["eval_id"]))
    _expect(metrics_from_counts(SufficientCounts(**payload["pooled_counts"])), payload["pooled_metrics"], "{} pooled metrics".format(spec["eval_id"]))
    if recovery_kind == "candidate":
        recovery = payload.get("alpha130_recovery", {})
        _expect(recovery.get("mode"), "fixed_metric_aux_task_arithmetic_alpha130_v1", "candidate recovery mode")
        _expect(recovery.get("alpha"), ALPHA, "candidate recovery alpha")
        _expect(recovery.get("immutable_inference_assets_before_after", {}).get("equal"), True, "candidate asset stability")
    elif recovery_kind == "alpha1":
        recovery = payload.get("task_arithmetic_recovery", {})
        _expect(recovery.get("mode"), "adaptive_metric_aux_task_arithmetic_alpha1_v1", "alpha1 recovery mode")
        _expect(recovery.get("alpha"), 1.0, "alpha1 recovery alpha")
        _expect(recovery.get("synthesis_manifest_sha256"), V5_MANIFEST_SHA256, "alpha1 manifest")
    elif recovery_kind == "m20":
        _expect(payload.get("evaluation_recovery_v4", {}).get("mode"), "evaluation_only_import_route_recovery_v4", "M20 recovery")
    return payload


def load_evaluation_result(spec, manifest_sha):
    payload, digest = _BASE_LOAD_EVALUATION_RESULT(spec)
    protocol, _ = load_protocol()
    _validate_evaluation_payload(
        payload,
        spec,
        protocol,
        EXPECTED_PROTOCOL_SHA256,
        core.sha256_file(RUNNER_PATH),
        spec["checkpoint_sha256"],
        "candidate",
    )
    _expect(payload["alpha130_recovery"]["synthesis_manifest_sha256"], manifest_sha, "candidate manifest")
    memory_inference, helper = _VERIFY_INFERENCE_HELPER(protocol)
    current_assets = _immutable_inference_asset_hashes(spec, memory_inference, helper)
    _expect(payload["alpha130_recovery"]["immutable_inference_assets_before_after"]["after"], current_assets, "candidate current asset identity")
    return payload, digest


def run_evaluation(eval_id=None, authorized=False):
    core.require_gpu_authorization(authorized)
    protocol, _ = load_protocol()
    load_command_audit()
    manifest, manifest_sha = load_synthesis_manifest(verify_formula=True)
    specs = evaluation_specs(protocol, manifest)
    if eval_id is not None:
        matches = [spec for spec in specs if spec["eval_id"] == eval_id]
        if len(matches) != 1:
            raise KeyError("Unknown alpha130 evaluation ID: {}".format(eval_id))
        return [evaluate_spec(protocol, matches[0], manifest_sha)]
    results = []
    for spec in specs:
        if Path(spec["result_path"]).is_file():
            payload, _ = load_evaluation_result(spec, manifest_sha)
            results.append(payload)
            continue
        subprocess.run(
            [sys.executable, str(RUNNER_PATH), "evaluate", "--eval-id", spec["eval_id"], GPU_AUTHORIZATION_FLAG],
            cwd=str(EVC_ROOT),
            check=True,
        )
        payload, _ = load_evaluation_result(spec, manifest_sha)
        results.append(payload)
    return results


def _anchor_spec(protocol, fold_id, anchor, alpha1_manifest=None):
    fold = next(item for item in protocol["dataset"]["folds"] if item["fold_id"] == fold_id)
    names = [item["name"] for item in core.held_items(protocol, fold)]
    if anchor == "released_m20":
        parent = protocol["alpha130_contract"]["input_checkpoint_contract"]["released_m20"]
        return {
            "eval_id": "{}_released_m20".format(fold_id),
            "fold_id": fold_id,
            "variant": "released_m20",
            "held_group": fold["held_group"],
            "held_source_names": names,
            "checkpoint": str(_checkpoint_path(parent).resolve()),
            "checkpoint_sha256": M20_SHA256,
        }
    record = next(item for item in alpha1_manifest["records"] if item["fold_id"] == fold_id)
    return {
        "eval_id": "{}_isolated_metric_aux".format(fold_id),
        "fold_id": fold_id,
        "variant": "isolated_metric_aux",
        "held_group": fold["held_group"],
        "held_source_names": names,
        "checkpoint": str(Path(record["output_path"]).resolve()),
        "checkpoint_sha256": record["output_sha256"],
    }


def _load_alpha1_manifest(protocol):
    record = protocol["alpha130_contract"]["alpha1_train_only_evidence"]["synthesis_manifest"]
    payload, digest, _ = _load_json_record(record, V5_MANIFEST_SHA256, "alpha1 synthesis manifest")
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-synthesis-manifest-v1", "alpha1 manifest schema")
    _expect(payload.get("protocol_sha256"), V5_PROTOCOL_SHA256, "alpha1 manifest protocol")
    _expect(payload.get("runner_sha256"), V5_RUNNER_SHA256, "alpha1 manifest runner")
    _expect(payload.get("passed"), True, "alpha1 manifest passed")
    return payload, digest


def load_anchor_evaluation(protocol, fold_id, anchor, alpha1_manifest):
    record = protocol["alpha130_contract"]["anchor_sufficient_count_artifacts"][anchor][fold_id]
    payload, digest, _ = _load_json_record(record, record["sha256"], "{} {} anchor".format(anchor, fold_id))
    spec = _anchor_spec(protocol, fold_id, anchor, alpha1_manifest)
    if anchor == "released_m20":
        identities = (V4_PROTOCOL_SHA256, V4_RUNNER_SHA256, M20_SHA256, "m20")
    else:
        identities = (V5_PROTOCOL_SHA256, V5_RUNNER_SHA256, spec["checkpoint_sha256"], "alpha1")
    _validate_evaluation_payload(payload, spec, protocol, *identities)
    return payload, digest


def _false_reduction(candidate, anchor):
    return (
        int(candidate["false_positive_events"]) < int(anchor["false_positive_events"])
        or int(candidate["false_components"]) < int(anchor["false_components"])
    )


def _official_not_worse(candidate_counts, candidate_metrics, anchor_counts, anchor_metrics):
    return {
        "score_not_lower": float(candidate_metrics["score"]) >= float(anchor_metrics["score"]),
        "pd_not_lower": float(candidate_metrics["pd"]) >= float(anchor_metrics["pd"]),
        "iou_not_lower": float(candidate_metrics["iou"]) >= float(anchor_metrics["iou"]),
        "fa_not_higher": float(candidate_metrics["fa"]) <= float(anchor_metrics["fa"]),
        "correct_objects_not_lower": int(candidate_counts["correct_objects"]) >= int(anchor_counts["correct_objects"]),
        "population_invariants_equal": core.population_invariants(candidate_counts) == core.population_invariants(anchor_counts),
    }


def dual_anchor_gate(candidate_counts, candidate_metrics, m20_counts, m20_metrics, alpha1_counts, alpha1_metrics, pooled):
    vs_m20 = _official_not_worse(candidate_counts, candidate_metrics, m20_counts, m20_metrics)
    vs_alpha1 = _official_not_worse(candidate_counts, candidate_metrics, alpha1_counts, alpha1_metrics)
    vs_m20["false_positive_events_or_false_components_strictly_lower"] = _false_reduction(candidate_counts, m20_counts)
    vs_alpha1["raw_true_positive_events_not_lower"] = int(candidate_counts["true_positive_events"]) >= int(alpha1_counts["true_positive_events"])
    if pooled:
        vs_m20["score_delta_at_least_0p00032"] = float(candidate_metrics["score"]) - float(m20_metrics["score"]) >= 0.00032
        vs_alpha1["score_delta_at_least_0p00005"] = float(candidate_metrics["score"]) - float(alpha1_metrics["score"]) >= 0.00005
        vs_alpha1["false_positive_events_or_false_components_strictly_lower"] = _false_reduction(candidate_counts, alpha1_counts)
    return {
        "against_released_m20": {"checks": vs_m20, "passed": all(vs_m20.values())},
        "against_alpha1": {"checks": vs_alpha1, "passed": all(vs_alpha1.values())},
        "passed": all(vs_m20.values()) and all(vs_alpha1.values()),
    }


def _count_delta(candidate, anchor):
    return {name: int(candidate[name]) - int(anchor[name]) for name in core.COUNT_FIELDS}


def _validate_alpha1_report(protocol):
    record = protocol["alpha130_contract"]["alpha1_train_only_evidence"]["report"]
    payload, digest, _ = _load_json_record(record, V5_REPORT_SHA256, "alpha1 train-only report")
    _expect(payload.get("schema"), "ev-uav-metric-aux-task-arithmetic-h2-grouped-oof-report-v1", "alpha1 report schema")
    _expect(payload.get("status"), "passed", "alpha1 report status")
    _expect(payload.get("passed"), True, "alpha1 report passed")
    _expect(payload.get("protocol_sha256"), V5_PROTOCOL_SHA256, "alpha1 report protocol")
    _expect(payload.get("runner_sha256"), V5_RUNNER_SHA256, "alpha1 report runner")
    _expect(payload.get("synthesis_manifest_sha256"), V5_MANIFEST_SHA256, "alpha1 report manifest")
    _expect(payload.get("released_m20_anchor_reinference"), False, "alpha1 M20 reinference")
    evidence = protocol["alpha130_contract"]["alpha1_train_only_evidence"]
    by_fold = {item["fold_id"]: item for item in payload["fold_results"]}
    for fold_id, expected in evidence["observed_fold_score_delta"].items():
        _close(by_fold[fold_id]["metric_delta"]["score"], expected, "{} observed alpha1 gain".format(fold_id))
    _close(payload["pooled"]["isolated_metric_aux"]["metric_delta"]["score"], evidence["observed_pooled_score_delta"], "pooled observed alpha1 gain")
    for name, expected in evidence["observed_pooled_count_delta"].items():
        _expect(payload["pooled"]["isolated_metric_aux"]["count_delta"][name], expected, "pooled alpha1 {} delta".format(name))
    return payload, digest


def build_report(protocol, protocol_sha, manifest, manifest_sha, specs):
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Alpha130 report must start CUDA-uninitialized.")
    SufficientCounts, metrics_from_counts = _count_tools()
    alpha1_report, alpha1_report_sha = _validate_alpha1_report(protocol)
    alpha1_manifest, alpha1_manifest_sha = _load_alpha1_manifest(protocol)
    _expect(alpha1_manifest_sha, V5_MANIFEST_SHA256, "alpha1 manifest binding")
    fold_results = []
    fold_gates = {}
    pooled_inputs = {"alpha130": [], "alpha1": [], "released_m20": []}
    artifact_bindings = {"alpha130": {}, "alpha1_reused": {}, "released_m20_reused": {}}
    seen = {"alpha130": [], "alpha1": [], "released_m20": []}
    for spec in specs:
        candidate, candidate_sha = load_evaluation_result(spec, manifest_sha)
        alpha1, alpha1_sha = load_anchor_evaluation(protocol, spec["fold_id"], "alpha1", alpha1_manifest)
        m20, m20_sha = load_anchor_evaluation(protocol, spec["fold_id"], "released_m20", alpha1_manifest)
        payloads = {"alpha130": candidate, "alpha1": alpha1, "released_m20": m20}
        for name, payload in payloads.items():
            pooled_inputs[name].append(payload["pooled_counts"])
            seen[name].extend(record["source_name"] for record in payload["records"])
        gate = dual_anchor_gate(
            candidate["pooled_counts"],
            candidate["pooled_metrics"],
            m20["pooled_counts"],
            m20["pooled_metrics"],
            alpha1["pooled_counts"],
            alpha1["pooled_metrics"],
            pooled=False,
        )
        fold_gates[spec["fold_id"]] = gate
        fold_results.append(
            {
                "fold_id": spec["fold_id"],
                "held_group": spec["held_group"],
                "released_m20": {"counts": m20["pooled_counts"], "metrics": m20["pooled_metrics"]},
                "alpha1": {"counts": alpha1["pooled_counts"], "metrics": alpha1["pooled_metrics"]},
                "alpha130": {"counts": candidate["pooled_counts"], "metrics": candidate["pooled_metrics"]},
                "delta_vs_released_m20": {
                    "metrics": core.metric_delta(candidate["pooled_metrics"], m20["pooled_metrics"]),
                    "counts": _count_delta(candidate["pooled_counts"], m20["pooled_counts"]),
                },
                "delta_vs_alpha1": {
                    "metrics": core.metric_delta(candidate["pooled_metrics"], alpha1["pooled_metrics"]),
                    "counts": _count_delta(candidate["pooled_counts"], alpha1["pooled_counts"]),
                },
                "dual_anchor_gate": gate,
            }
        )
        artifact_bindings["alpha130"][spec["eval_id"]] = {"path": spec["result_path"], "sha256": candidate_sha}
        artifact_bindings["alpha1_reused"][spec["fold_id"]] = {
            "path": protocol["alpha130_contract"]["anchor_sufficient_count_artifacts"]["alpha1"][spec["fold_id"]]["workspace_relative_path"],
            "sha256": alpha1_sha,
        }
        artifact_bindings["released_m20_reused"][spec["fold_id"]] = {
            "path": protocol["alpha130_contract"]["anchor_sufficient_count_artifacts"]["released_m20"][spec["fold_id"]]["workspace_relative_path"],
            "sha256": m20_sha,
        }
    expected_union = [item["name"] for fold in protocol["dataset"]["folds"] for item in core.held_items(protocol, fold)]
    _expect(len(expected_union), 11, "held source count")
    for name, actual in seen.items():
        _expect(actual, expected_union, "{} exact 11-source union".format(name))
    pooled_counts = {name: core.add_count_dicts(values) for name, values in pooled_inputs.items()}
    pooled_metrics = {name: metrics_from_counts(SufficientCounts(**counts)) for name, counts in pooled_counts.items()}
    pooled_gate = dual_anchor_gate(
        pooled_counts["alpha130"],
        pooled_metrics["alpha130"],
        pooled_counts["released_m20"],
        pooled_metrics["released_m20"],
        pooled_counts["alpha1"],
        pooled_metrics["alpha1"],
        pooled=True,
    )
    gates = {
        "folds": fold_gates,
        "pooled": pooled_gate,
        "checks": {
            "every_fold_passes_both_anchors": all(item["passed"] for item in fold_gates.values()),
            "pooled_passes_both_anchors_and_overshoot": pooled_gate["passed"],
            "alpha_exactly_1p30": ALPHA == 1.30,
            "one_candidate_no_grid": True,
            "eleven_train_sources_exactly_once_per_variant": all(values == expected_union for values in seen.values()),
            "anchor_reinference_false": True,
        },
    }
    gates["passed"] = all(gates["checks"].values())
    _, command_audit_sha = load_command_audit()
    payload = {
        "schema": "ev-uav-metric-aux-task-arithmetic-alpha130-h2-grouped-oof-report-v1",
        "created_utc": core.utc_now(),
        "status": "passed" if gates["passed"] else "failed",
        "passed": gates["passed"],
        "evidence_class": protocol["evidence_class"],
        "claim_scope": protocol["alpha130_contract"]["claim_scope"],
        "adaptive_selection_disclosure": protocol["alpha130_contract"]["adaptive_selection_disclosure"],
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "tests_sha256": core.sha256_file(TEST_PATH),
        "command_audit_sha256": command_audit_sha,
        "synthesis_manifest_sha256": manifest_sha,
        "synthesis_records": manifest["records"],
        "alpha1_train_only_report_sha256": alpha1_report_sha,
        "alpha1_synthesis_manifest_sha256": alpha1_manifest_sha,
        "fold_results": fold_results,
        "pooled": {
            "released_m20": {"counts": pooled_counts["released_m20"], "metrics": pooled_metrics["released_m20"]},
            "alpha1": {"counts": pooled_counts["alpha1"], "metrics": pooled_metrics["alpha1"]},
            "alpha130": {"counts": pooled_counts["alpha130"], "metrics": pooled_metrics["alpha130"]},
            "delta_vs_released_m20": {
                "metrics": core.metric_delta(pooled_metrics["alpha130"], pooled_metrics["released_m20"]),
                "counts": _count_delta(pooled_counts["alpha130"], pooled_counts["released_m20"]),
            },
            "delta_vs_alpha1": {
                "metrics": core.metric_delta(pooled_metrics["alpha130"], pooled_metrics["alpha1"]),
                "counts": _count_delta(pooled_counts["alpha130"], pooled_counts["alpha1"]),
            },
        },
        "dual_anchor_overshoot_gates": gates,
        "artifact_bindings": artifact_bindings,
        "released_m20_anchor_reinference": False,
        "alpha1_anchor_reinference": False,
        "new_training_optimizer_steps": 0,
        "t32_read_or_combined": False,
        "decision": (
            "eligible_for_separate_all11_alpha130_synthesis_before_any_validation"
            if gates["passed"]
            else protocol["alpha130_contract"]["adaptive_selection_disclosure"]["failure_action"]
        ),
        "all11_alpha130_checkpoint_synthesized_by_this_runner": False,
        "no_default_submission_or_validation_change": True,
    }
    _write_json_exclusive(REPORT_PATH, payload)
    return payload


def run_report():
    protocol, protocol_sha = load_protocol()
    load_command_audit()
    manifest, manifest_sha = load_synthesis_manifest(verify_formula=True)
    payload = build_report(protocol, protocol_sha, manifest, manifest_sha, evaluation_specs(protocol, manifest))
    print("alpha130 grouped-OOF report:", REPORT_PATH)
    print("promotion passed:", payload["passed"])
    return payload


def _patch_core_for_alpha130():
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.EXPECTED_PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.COMMAND_AUDIT_PATH = COMMAND_AUDIT_PATH
    core.EVALUATION_ROOT = EVALUATION_ROOT
    core.REPORT_PATH = REPORT_PATH
    core.__file__ = str(RUNNER_PATH)
    core.load_protocol = load_protocol
    core.load_command_audit = load_command_audit
    core.write_new_json = _write_json_exclusive


_patch_core_for_alpha130()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    commands.add_parser("synthesize")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--eval-id", default=None)
    evaluate.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    commands.add_parser("report")
    all_eval = commands.add_parser("all-evaluate-report")
    all_eval.add_argument(GPU_AUTHORIZATION_FLAG, action="store_true", dest="authorized")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit()
    if args.command == "synthesize":
        return run_synthesis()
    if args.command == "evaluate":
        return run_evaluation(args.eval_id, args.authorized)
    if args.command == "report":
        return run_report()
    if args.command == "all-evaluate-report":
        core.require_gpu_authorization(args.authorized)
        run_evaluation(authorized=True)
        return run_report()
    raise RuntimeError("Unsupported alpha130 command: {}".format(args.command))


if __name__ == "__main__":
    main()

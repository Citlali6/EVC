import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset.temporal_frame import temporal_frame_video_from_events
from utils.h2_pyramid_recovery_v2_inference import (
    EFFECTIVE_C00_SHA256,
    FINAL_REFIT_PROTOCOL_SHA256,
    H2PyramidRecoveryV2Inference,
    OUTER_DECISION_SHA256,
    PREDICTION_THRESHOLD,
    Stage1ComponentPayload,
    _component_node_geometry,
    apply_atomic_stage2,
    load_final_package_payload,
    load_frozen_checkpoint_payloads,
    sha256_file,
    state_sha256,
    use_h2_pyramid_recovery_v2,
)


def _h2_polarities(event_count=200005):
    values = np.zeros(event_count, dtype=np.float32)
    values[: event_count // 5] = 1.0
    return values


def test_route_boundaries_are_strict_count_and_inclusive_minority():
    balanced_boundary = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), 100000)
    assert not use_h2_pyramid_recovery_v2(200000, balanced_boundary)

    polarities = _h2_polarities()
    assert polarities.sum() / polarities.size == pytest.approx(0.20)
    assert use_h2_pyramid_recovery_v2(polarities.size, polarities)

    below = polarities.copy()
    below[np.flatnonzero(below > 0.5)[-1]] = 0.0
    assert not use_h2_pyramid_recovery_v2(below.size, below)


def test_non_h2_returns_same_m20_object_without_calling_stage1():
    calls = []

    def forbidden_stage1(*_args):
        calls.append(True)
        raise AssertionError("non-H2 must not execute Stage1")

    wrapper = H2PyramidRecoveryV2Inference(
        forbidden_stage1,
        lambda _nodes: (_ for _ in ()).throw(
            AssertionError("non-H2 must not execute the recovery head")
        ),
        0.7,
    )
    scores = np.asarray([-0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    before = scores.view(np.uint32).copy()
    result = wrapper.apply(
        scores,
        video=None,
        polarities=np.zeros(scores.size, dtype=np.float32),
        locations4=None,
    )
    assert result.scores is scores
    assert np.array_equal(result.scores.view(np.uint32), before)
    assert result.receipt.non_h2_m20_object_identity
    assert not result.receipt.routed_to_h2
    assert result.receipt.second_c00_applied is False
    assert calls == []


def test_h2_restores_only_selected_complete_components_to_m20_bits():
    polarities = _h2_polarities()
    event_count = polarities.size
    m20 = np.linspace(0.01, 0.99, event_count, dtype=np.float32)
    stage1 = m20.copy()
    stage1[:6] = np.asarray([0.11, 0.12, 0.13, 0.14, 0.15, 0.16], dtype=np.float32)
    components = (
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
        np.asarray([4, 5], dtype=np.int64),
    )
    nodes = tuple(np.zeros((index + 1, 96), dtype=np.float32) for index in range(3))
    stage1_calls = []

    def stage1_executor(video, locations4):
        stage1_calls.append((video, locations4))
        return Stage1ComponentPayload(m20.copy(), stage1, components, nodes)

    wrapper = H2PyramidRecoveryV2Inference(
        stage1_executor,
        lambda received: np.asarray([0.9, 0.6, 0.8], dtype=np.float64),
        0.7,
    )
    sentinel_video = object()
    sentinel_locations = object()
    result = wrapper.apply(m20, sentinel_video, polarities, sentinel_locations)

    expected = stage1.copy()
    expected[components[0]] = m20[components[0]]
    expected[components[2]] = m20[components[2]]
    assert np.array_equal(result.scores.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(
        result.scores[components[1]].view(np.uint32),
        stage1[components[1]].view(np.uint32),
    )
    assert np.array_equal(result.scores[6:].view(np.uint32), stage1[6:].view(np.uint32))
    assert stage1_calls == [(sentinel_video, sentinel_locations)]
    assert result.receipt.routed_to_h2
    assert result.receipt.restored_component_count == 2
    assert result.receipt.restored_event_count == 4
    assert result.receipt.internal_m20_post_bitwise_verified
    assert result.receipt.whole_components_only
    assert result.receipt.second_c00_applied is False


def test_internal_m20_check_is_bitwise_not_numeric():
    polarities = _h2_polarities()
    m20 = np.full(polarities.size, 0.5, dtype=np.float32)
    internal = m20.copy()
    m20[0] = np.float32(-0.0)
    internal[0] = np.float32(0.0)
    payload = Stage1ComponentPayload(internal, m20.copy(), (), ())
    wrapper = H2PyramidRecoveryV2Inference(lambda *_args: payload, lambda _: [], 0.7)
    with pytest.raises(RuntimeError, match="differ bitwise"):
        wrapper.apply(m20, object(), polarities, object())


def test_atomic_helper_has_no_partial_or_outside_edits():
    m20 = np.asarray([0.8, 0.9, 0.7, 0.95, 0.6], dtype=np.float32)
    stage1 = np.asarray([0.1, 0.2, 0.7, 0.3, 0.6], dtype=np.float32)
    components = (np.asarray([0, 1]), np.asarray([3]))
    output, decisions = apply_atomic_stage2(
        stage1, m20, components, np.asarray([0.7, 0.699]), 0.7
    )
    assert decisions.tolist() == [True, False]
    expected = np.asarray([0.8, 0.9, 0.7, 0.3, 0.6], dtype=np.float32)
    assert np.array_equal(output.view(np.uint32), expected.view(np.uint32))


def test_geometry_matches_formal_96_feature_tail_without_source_fields():
    locations = np.asarray([[10, 20, 0], [12, 24, 50]], dtype=np.int64)
    video = temporal_frame_video_from_events(
        name="ignored_provenance",
        locations=locations,
        polarities=np.asarray([0.0, 1.0], dtype=np.float32),
        temporal_bin_size=50,
        whole_t=8000,
    )
    stage1 = np.asarray([PREDICTION_THRESHOLD, 0.1], dtype=np.float32)
    nodes = _component_node_geometry(np.asarray([0, 1]), video, stage1)
    assert len(nodes) == 2
    assert nodes[0][2].shape == (10,)
    assert nodes[1][2][7] == pytest.approx(2.0 / 346.0)
    assert nodes[1][2][8] == pytest.approx(4.0 / 260.0)
    assert nodes[0][2][9] == 1.0
    assert nodes[1][2][9] == 0.0

    inference_parameters = set(
        inspect.signature(H2PyramidRecoveryV2Inference.apply).parameters
    )
    forbidden = {"source", "name", "path", "hash", "index", "label", "target_id"}
    assert not inference_parameters.intersection(forbidden)


def test_promoted_checkpoint_receipts_load_on_cpu_without_cuda_initialization():
    assert not torch.cuda.is_initialized()
    payloads = load_frozen_checkpoint_payloads()
    assert payloads.recovery_cutoff == pytest.approx(0.6556717753410339)
    assert state_sha256(payloads.stage1_state_dict)
    assert state_sha256(payloads.recovery_state_dict)
    assert payloads.effective_c00_sha256 == EFFECTIVE_C00_SHA256
    assert not torch.cuda.is_initialized()


def test_final_package_loader_binds_wrapper_protocol_states_and_sidecar(tmp_path):
    stage1_state = {"weight": torch.arange(4, dtype=torch.float32)}
    recovery_state = {"weight": torch.arange(3, dtype=torch.float32)}
    dependency = Path(__file__).resolve().parents[1] / "model" / "h2_pyramid_component_recovery.py"
    wrapper_path = (
        Path(__file__).resolve().parents[1]
        / "utils"
        / "h2_pyramid_recovery_v2_inference.py"
    )
    package = {
        "schema": "ev-uav-h2-pyramid-recovery-v2-all11-final-package-v1",
        "protocol_sha256": FINAL_REFIT_PROTOCOL_SHA256,
        "runner_sha256": "1" * 64,
        "inference_wrapper_sha256": sha256_file(wrapper_path),
        "fit_sources": ["train_{:03d}.npz".format(value) for value in range(88, 99)],
        "execution_dependency_sha256": {
            "model/h2_pyramid_component_recovery.py": sha256_file(dependency)
        },
        "released_M20_checkpoint_sha256": (
            "4b8b2b19ea9d913ee4e52cb21ae52bf945b2b0f3cefd5cb5ab6f64d51bf49849"
        ),
        "released_m20_state_sha256": (
            "feb234e530688a11865e0d49b58a9f54806f69ea63a9de3e01e8b9f714a6113d"
        ),
        "outer_decision_sha256": OUTER_DECISION_SHA256,
        "fresh_Stage1_state_sha256": state_sha256(stage1_state),
        "fresh_Stage1_state_dict": stage1_state,
        "fresh_final_recovery_head_state_sha256": state_sha256(recovery_state),
        "fresh_final_recovery_head_state_dict": recovery_state,
        "recovery_cutoff": 0.71,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "effective_C00_sha256": EFFECTIVE_C00_SHA256,
        "route": {
            "event_count_cutoff_exclusive": 200000,
            "polarity_minority_cutoff_inclusive": 0.20,
            "non_H2_behavior": "bitwise_released_M20_identity",
        },
        "fresh_initialization": {
            "Stage1": True,
            "OOF_heads": True,
            "final_head": True,
        },
        "promoted_outer_weights_or_cutoff_reused": False,
        "optimizer_state_included": False,
        "validation_or_test_read": False,
    }
    package_path = tmp_path / "final.pt"
    torch.save(package, package_path)
    package_hash = sha256_file(package_path)
    Path(str(package_path) + ".sha256").write_text(
        "{}  {}\n".format(package_hash, package_path.name), encoding="ascii"
    )
    loaded = load_final_package_payload(package_path)
    assert loaded.recovery_cutoff == 0.71
    assert loaded.effective_c00_sha256 == EFFECTIVE_C00_SHA256
    assert not torch.cuda.is_initialized()

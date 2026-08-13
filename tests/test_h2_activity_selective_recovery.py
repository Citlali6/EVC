import unittest

import numpy as np
import torch
from torch.nn import functional

from model.h2_activity_selective_recovery import (
    DisagreementRecoveryNet,
    RECOVERY_PATCH_CHANNELS,
    TRAJECTORY_FEATURES,
    recovery_parameter_count,
    recovery_sequence_collate,
)
from model.high_density_polarity_expert import FineTemporalPolarityMultiScaleExpert
from utils.activity_selective_recovery import (
    atomic_recover_or_identity,
    disagreement_trajectory_context,
    extract_disagreement_components,
    marginal_recovery_targets,
    negative_reference_conformal_confidence,
    unique_recovery_cutoffs,
)
from utils.atomic_component_deletion import use_h2_atomic_deletion


THRESHOLD = 0.719


def synthetic_stream():
    locations = np.asarray(
        [
            [0, 10, 10, 0],
            [0, 11, 10, 50],
            [0, 30, 30, 0],
            [0, 31, 30, 50],
            [0, 50, 50, 0],
            [0, 70, 70, 0],
        ],
        dtype=np.int64,
    )
    m20 = np.asarray([0.80, 0.82, 0.90, 0.91, 0.10, 0.05], dtype=np.float32)
    activity = np.asarray([0.79, 0.20, 0.15, 0.10, 0.95, 0.05], dtype=np.float32)
    return m20, activity, locations


def test_stage1_activity_control_is_1712_parameter_zero_residual():
    torch.manual_seed(3)
    expert = FineTemporalPolarityMultiScaleExpert(input_mode="activity_control")
    assert sum(parameter.numel() for parameter in expert.parameters()) == 1712
    frames = torch.randn(2, 10, 16, 18)
    paired = expert.paired_input_features(frames)
    negative = frames[:, 0::2]
    positive = frames[:, 1::2]
    expected_activity = 0.5 * (negative + positive)
    assert torch.equal(paired, torch.cat((expected_activity,) * 3, dim=1))
    assert torch.count_nonzero(expert(frames)) == 0


def test_disagreement_units_are_complete_m20_components_not_activity_components():
    m20, activity, locations = synthetic_stream()
    batch = extract_disagreement_components(m20, activity, locations, THRESHOLD)
    assert batch.m20_component_count == 2
    assert batch.activity_component_count == 2
    assert batch.m20_component_ids.tolist() == [0, 1]
    assert [indices.tolist() for indices in batch.event_indices] == [[0, 1], [2, 3]]
    assert batch.missing_event_counts.tolist() == [1, 2]
    assert batch.activity_supported_event_counts.tolist() == [1, 0]


def test_atomic_recovery_copies_whole_m20_component_and_preserves_activity_elsewhere():
    m20, activity, locations = synthetic_stream()
    components = extract_disagreement_components(
        m20, activity, locations, THRESHOLD
    ).event_indices
    candidate, receipt = atomic_recover_or_identity(
        m20,
        activity,
        components,
        np.asarray([0.9, 0.1]),
        cutoff=0.5,
        prediction_threshold=THRESHOLD,
        enabled=True,
    )
    assert receipt.enabled
    assert receipt.recovered_component_count == 1
    assert receipt.recovered_event_count == 2
    assert receipt.complete_components_only
    assert receipt.activity_outside_recovery_bitwise_equal
    assert receipt.recovered_m20_scores_bitwise_equal
    assert np.array_equal(candidate[[0, 1]], m20[[0, 1]])
    assert np.array_equal(candidate[[2, 3, 4, 5]], activity[[2, 3, 4, 5]])
    assert candidate[4] == activity[4]  # Activity-only component is never removed.


def test_invalid_partial_or_overlapping_recovery_fails_closed_to_activity_identity():
    m20, activity, _ = synthetic_stream()
    invalid = (np.asarray([0, 1]), np.asarray([1, 2]))
    candidate, receipt = atomic_recover_or_identity(
        m20,
        activity,
        invalid,
        np.asarray([1.0, 1.0]),
        cutoff=0.5,
        prediction_threshold=THRESHOLD,
        enabled=True,
    )
    assert not receipt.enabled
    assert receipt.fallback_reason == "atomic_integrity_failure"
    assert not receipt.complete_components_only
    assert np.array_equal(candidate, activity)


def test_disabled_or_bad_confidence_count_is_exact_activity_identity():
    m20, activity, locations = synthetic_stream()
    components = extract_disagreement_components(
        m20, activity, locations, THRESHOLD
    ).event_indices
    for confidences, enabled, reason in (
        (np.asarray([1.0, 1.0]), False, "identity_policy"),
        (np.asarray([1.0]), True, "component_confidence_count_mismatch"),
    ):
        candidate, receipt = atomic_recover_or_identity(
            m20,
            activity,
            components,
            confidences,
            cutoff=0.5,
            prediction_threshold=THRESHOLD,
            enabled=enabled,
        )
        assert np.array_equal(candidate, activity)
        assert receipt.fallback_reason == reason


def test_fit_only_marginal_class_uses_strict_official_score_gain():
    m20, activity, locations = synthetic_stream()
    components = extract_disagreement_components(
        m20, activity, locations, THRESHOLD
    ).event_indices

    def score_fn(scores):
        predicted = np.asarray(scores) >= THRESHOLD
        # The first recovery is useful; the second incurs a larger synthetic cost.
        return float(np.count_nonzero(predicted) - 3 * int(predicted[2]))

    targets, deltas = marginal_recovery_targets(
        m20, activity, components, THRESHOLD, score_fn
    )
    assert targets.tolist() == [1, 0]
    assert deltas.tolist() == [1.0, -1.0]


def test_negative_reference_conformal_scale_is_conservative_on_ties():
    scores = np.asarray([0.1, 0.4, 0.5, 0.9])
    negatives = np.asarray([0.2, 0.4, 0.4, 0.8])
    confidence = negative_reference_conformal_confidence(scores, negatives)
    assert np.array_equal(confidence, np.asarray([0.2, 0.4, 0.8, 1.0]))
    cutoffs = unique_recovery_cutoffs(confidence)
    assert cutoffs[0] > confidence.max()
    assert np.array_equal(cutoffs[1:], np.asarray([1.0, 0.8, 0.4, 0.2]))
    try:
        negative_reference_conformal_confidence(scores, np.empty(0))
    except ValueError:
        pass
    else:
        raise AssertionError("empty negative reference did not fail closed")


def test_long_trajectory_context_is_query_aligned_finite_and_input_only():
    m20, activity, locations = synthetic_stream()
    component = np.asarray([0, 1], dtype=np.int64)
    queries, context = disagreement_trajectory_context(
        component,
        locations,
        m20,
        activity,
        THRESHOLD,
        patch_radius=7,
    )
    assert [query.temporal_bin for query in queries] == [0, 1]
    assert context.shape == (2, TRAJECTORY_FEATURES)
    assert np.isfinite(context).all()
    assert context[:, 0].tolist() == [-1.0, 1.0]
    assert context[:, 6].tolist() == [1.0, 0.0]
    assert context[1, 4] > 0.0


def test_recovery_head_variable_sequences_have_all_branch_gradients():
    torch.manual_seed(11)
    model = DisagreementRecoveryNet()
    assert recovery_parameter_count(model) == 7910
    items = []
    for length, target in ((3, 1.0), (2, 0.0), (4, 1.0)):
        items.append(
            {
                "patches": torch.randn(length, RECOVERY_PATCH_CHANNELS, 15, 15),
                "trajectory": torch.randn(length, TRAJECTORY_FEATURES),
                "target": target,
                "weight": 1.0,
            }
        )
    batch = recovery_sequence_collate(items)
    logits, embeddings, attention = model(
        batch["patches"],
        batch["trajectory"],
        batch["lengths"],
        return_embedding=True,
    )
    assert logits.shape == (3,)
    assert embeddings.shape == (3, 64)
    assert attention.shape == (3, 4)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)
    loss = functional.binary_cross_entropy_with_logits(
        logits, batch["targets"], weight=batch["weights"]
    )
    loss.backward()
    for prefix in (
        "semantic_stem.",
        "context_stem.",
        "spatial_fusion.",
        "trajectory_encoder.",
        "temporal.",
        "temporal_attention.",
        "classifier.",
    ):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_h2_route_remains_input_only_and_exact():
    balanced = np.asarray([0, 1] * 100001, dtype=np.uint8)
    saturated = np.zeros(200002, dtype=np.uint8)
    saturated[:1000] = 1
    assert use_h2_atomic_deletion(len(balanced), balanced)
    assert not use_h2_atomic_deletion(len(saturated), saturated)
    assert not use_h2_atomic_deletion(200000, np.asarray([0, 1] * 100000))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import torch

from model.h2_multiscale_temporal_pyramid_expert import (
    OBSERVATION_CHANNELS,
    TEMPORAL_SCALES,
    FrozenM20MultiScalePyramidAdapter,
    MultiScaleTemporalPyramidHead,
    audit_released_m20_feature_api,
    downsample_frozen_observations,
    fixed_multiscale_temporal_moments,
    pyramid_expert_parameter_count,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.h2_multiscale_pyramid_loss import (
    PyramidDualState,
    multiscale_pyramid_constrained_loss,
    validate_pyramid_step_diagnostics,
)


def dense_inputs(time=8, height=16, width=24):
    generator = torch.Generator().manual_seed(41)
    decoder = torch.randn(1, time, 16, height, width, generator=generator)
    base = torch.randn(1, time, 1, height, width, generator=generator)
    centre = torch.rand(1, time, 3, height, width, generator=generator)
    observations = downsample_frozen_observations(decoder, base, centre)
    summaries = fixed_multiscale_temporal_moments(observations)
    return decoder, base, centre, observations, summaries


class MultiScalePyramidHeadTests(unittest.TestCase):
    def test_full_stream_summary_is_global_and_local_scales_differ(self):
        values = torch.arange(160, dtype=torch.float32).reshape(1, 160, 1, 1, 1)
        summaries = fixed_multiscale_temporal_moments(values)
        self.assertEqual(len(summaries), 4)
        global_mean = summaries[-1][:, :, 0]
        self.assertTrue(torch.equal(global_mean, torch.full_like(global_mean, 79.5)))
        self.assertFalse(torch.equal(summaries[0][:, :, 0], global_mean))
        self.assertTrue(torch.all(summaries[-1][:, :, 1] > 0))

    def test_zero_initialization_is_exact_dense_identity(self):
        decoder, base, centre, _, summaries = dense_inputs()
        head = MultiScaleTemporalPyramidHead()
        parts = head(decoder, base, centre, summaries, return_parts=True)
        self.assertTrue(torch.equal(parts.refined_logits, base))
        self.assertTrue(torch.count_nonzero(parts.correction) == 0)
        expected = torch.full_like(parts.mixture_weights, 1.0 / len(TEMPORAL_SCALES))
        self.assertTrue(torch.equal(parts.mixture_weights, expected))
        self.assertLess(pyramid_expert_parameter_count(head), 50000)

    def test_dynamic_loss_reaches_long_context_after_zero_output_step(self):
        decoder, base, centre, _, summaries = dense_inputs()
        head = MultiScaleTemporalPyramidHead()
        optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)

        def one_step():
            optimizer.zero_grad(set_to_none=True)
            parts = head(decoder, base, centre, summaries, return_parts=True)
            # One aligned event per selected dense cell.
            refined = parts.refined_logits[0, :, 0, 0, 0]
            baseline = base[0, :, 0, 0, 0]
            labels = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0], dtype=torch.float32)
            target_ids = torch.tensor([1, 0, 1, 0, 2, 0, 0, 0])
            times = torch.arange(8)
            components = (torch.tensor([1, 3]), torch.tensor([5, 6, 7]))
            dual = PyramidDualState()
            loss, recall, suppression, diagnostics = multiscale_pyramid_constrained_loss(
                refined,
                baseline,
                labels,
                target_ids,
                times,
                components,
                dual,
            )
            loss.backward()
            return parts, loss, recall, suppression, diagnostics

        first, loss, recall, suppression, diagnostics = one_step()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(recall), 0.0)
        self.assertAlmostEqual(float(suppression), 1.0, places=5)
        self.assertAlmostEqual(diagnostics["classification_normalized"], 1.0, places=5)
        self.assertGreater(float(head.output_projection.weight.grad.abs().sum()), 0.0)
        optimizer.step()
        _, second_loss, _, _, _ = one_step()
        self.assertTrue(torch.isfinite(second_loss))
        self.assertIsNotNone(head.scale_encoder.input_projection.weight.grad)
        self.assertGreater(
            float(head.scale_encoder.input_projection.weight.grad.abs().sum()), 0.0
        )

    def test_forward_contract_has_no_truth_or_source_argument(self):
        parameters = set(inspect.signature(MultiScaleTemporalPyramidHead.forward).parameters)
        forbidden = {"source", "source_name", "path", "hash", "fold", "labels", "target_ids"}
        self.assertFalse(parameters & forbidden)
        self.assertEqual(OBSERVATION_CHANNELS, 20)


class FrozenM20FeatureApiTests(unittest.TestCase):
    def test_api_audit_and_wrapper_freeze(self):
        released = BidirectionalTemporalMemoryNet(
            input_channels=10,
            width=16,
            density_calibration_enabled=True,
            temporal_attention_enabled=True,
        )
        audit = audit_released_m20_feature_api(released, context_bins=5)
        self.assertEqual(audit["decoder0_channels"], 16)
        wrapper = FrozenM20MultiScalePyramidAdapter(released, context_bins=5)
        wrapper.train()
        self.assertFalse(wrapper.released_m20.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in released.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in wrapper.expert.parameters()))

    def test_step_diagnostic_validator(self):
        records = []
        for step in range(1, 9):
            records.append(
                {
                    "step": step,
                    "loss": 1.0,
                    "gradient_norm": 2.0,
                    "classification_normalized": 1.0,
                    "target_time_recall_violation": 0.0,
                    "hard_negative_suppression_violation": 1.0,
                    "dual_target_time_recall_after": 1.0,
                    "dual_hard_negative_suppression_after": 2.0,
                    "mixture_entropy": 1.0,
                    "correction_abs_mean": 0.0,
                }
            )
        self.assertTrue(validate_pyramid_step_diagnostics(records, 8))


if __name__ == "__main__":
    unittest.main()

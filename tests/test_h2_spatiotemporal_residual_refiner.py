import copy
import unittest

import torch

from model.h2_spatiotemporal_residual_refiner import (
    FrozenM20ResidualRefiner,
    SpatiotemporalResidualHead,
    refiner_parameter_count,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


def _base_model():
    torch.manual_seed(7)
    model = BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=4,
        temporal_attention_enabled=False,
        density_calibration_enabled=False,
    )
    model.eval()
    return model


class ResidualRefinerTests(unittest.TestCase):
    def test_head_zero_initialization_is_exact_identity(self):
        torch.manual_seed(11)
        head = SpatiotemporalResidualHead(decoder_channels=4, hidden_channels=6)
        decoder = torch.randn(2, 3, 4, 8, 10)
        logits = torch.randn(2, 3, 1, 8, 10)
        centre = torch.randn(2, 3, 3, 8, 10)
        residual = head(decoder, logits, centre)
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        self.assertTrue(torch.equal(logits + residual, logits))

    def test_wrapper_matches_frozen_base_bitwise_at_initialization(self):
        base = _base_model()
        wrapper = FrozenM20ResidualRefiner(
            base, context_bins=5, hidden_channels=6,
        ).eval()
        frames = torch.randn(1, 3, 10, 16, 24)
        with torch.no_grad():
            expected = base(frames)
            actual, returned_base = wrapper(frames, return_base_logits=True)
        self.assertTrue(torch.equal(returned_base, expected))
        self.assertTrue(torch.equal(actual, expected))

    def test_decode_with_residual_matches_released_interface_at_init(self):
        base = _base_model()
        wrapper = FrozenM20ResidualRefiner(
            base, context_bins=5, hidden_channels=6,
        ).eval()
        frames = torch.randn(4, 10, 16, 24)
        with torch.no_grad():
            bottlenecks = base.encode_bottleneck(frames)
            residual = base.temporal_residual(bottlenecks)
            expected = base.decode_with_residual(frames, residual)
            actual = wrapper.decode_with_residual(frames, residual)
        self.assertTrue(torch.equal(actual, expected))

    def test_optimizer_step_changes_only_refiner_state(self):
        base = _base_model()
        wrapper = FrozenM20ResidualRefiner(
            base, context_bins=5, hidden_channels=6,
        ).train()
        frozen_before = copy.deepcopy(wrapper.frozen_state_dict())
        optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=1e-3)
        frames = torch.randn(1, 3, 10, 16, 24)
        output = wrapper(frames)
        loss = output.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertTrue(all(
            torch.equal(frozen_before[name], value)
            for name, value in wrapper.frozen_state_dict().items()
        ))
        self.assertGreater(
            torch.count_nonzero(wrapper.refiner.output_projection.weight).item(), 0,
        )

    def test_refiner_has_no_label_source_or_route_arguments(self):
        base = _base_model()
        wrapper = FrozenM20ResidualRefiner(base, context_bins=5, hidden_channels=6)
        self.assertLess(refiner_parameter_count(wrapper), 10_000)
        forbidden = ('label', 'target', 'source', 'filename', 'fold', 'path', 'hash')
        state_names = tuple(wrapper.refiner.state_dict())
        self.assertFalse(any(
            token in name.lower() for name in state_names for token in forbidden
        ))

    def test_invalid_centre_input_shape_is_rejected(self):
        head = SpatiotemporalResidualHead(decoder_channels=4, hidden_channels=6)
        with self.assertRaisesRegex(ValueError, 'centre_inputs'):
            head(
                torch.randn(1, 2, 4, 8, 8),
                torch.randn(1, 2, 1, 8, 8),
                torch.randn(1, 2, 2, 8, 8),
            )


if __name__ == '__main__':
    unittest.main()

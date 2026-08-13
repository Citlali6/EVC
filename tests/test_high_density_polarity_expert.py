import inspect
import unittest

import torch

from model.high_density_polarity_expert import (
    FineTemporalPolarityMultiScaleExpert,
    HighDensityPolarityExpertMemoryNet,
    configure_expert_only_training,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


def _base_and_h1_model():
    torch.manual_seed(812)
    base = BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=4,
        temporal_attention_enabled=False,
        density_calibration_enabled=False,
    ).eval()
    torch.manual_seed(913)
    model = HighDensityPolarityExpertMemoryNet(
        input_channels=10,
        width=4,
        temporal_attention_enabled=False,
        density_calibration_enabled=False,
        expert_input_mode="h1_saturation",
        expert_hidden_channels=4,
    ).eval()
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    expected_missing = {
        name for name in model.state_dict() if name.startswith("high_density_expert.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise AssertionError("Synthetic M20 migration did not isolate the expert state.")
    if incompatible.unexpected_keys:
        raise AssertionError("Synthetic M20 migration produced unexpected state.")
    return base, model


class H1SaturationExpertTests(unittest.TestCase):
    def test_clip4_core_is_bitwise_identity_at_zero_initialization(self):
        base, model = _base_and_h1_model()
        generator = torch.Generator(device="cpu").manual_seed(101)
        clip4 = torch.rand((1, 3, 10, 16, 24), generator=generator)
        clip8_a = torch.rand((1, 3, 10, 16, 24), generator=generator)
        clip8_b = 1.0 - clip8_a
        with torch.no_grad():
            expected = base(clip4)
            actual_a = model(clip4, expert_frames=clip8_a)
            actual_b = model(clip4, expert_frames=clip8_b)
        self.assertTrue(torch.equal(actual_a, expected))
        self.assertTrue(torch.equal(actual_b, expected))

    def test_clip8_peak_bank_adds_information_only_after_clip4_saturation(self):
        control = FineTemporalPolarityMultiScaleExpert(
            input_channels=10,
            output_channels=8,
            hidden_channels=4,
            input_mode="activity_control",
        )
        h1 = FineTemporalPolarityMultiScaleExpert(
            input_channels=10,
            output_channels=8,
            hidden_channels=4,
            input_mode="h1_saturation",
        )
        clip4 = torch.zeros((1, 10, 1, 2))
        clip8 = torch.zeros_like(clip4)
        # One-polarity log-counts of 2 and 6.  The released input clips the
        # latter at four, while the parallel stack preserves it up to eight.
        clip4[0, 0, 0] = torch.tensor((2.0 / 4.0, 4.0 / 4.0))
        clip8[0, 0, 0] = torch.tensor((2.0 / 8.0, 6.0 / 8.0))
        control_features = control.paired_input_features(clip4)
        h1_features = h1.paired_input_features(clip4, expert_frames=clip8)
        control_second_bank = control_features[:, 5:10]
        h1_peak_bank = h1_features[:, 5:10]
        self.assertEqual(
            h1_peak_bank[0, 0, 0, 0].item(),
            control_second_bank[0, 0, 0, 0].item(),
        )
        self.assertGreater(
            h1_peak_bank[0, 0, 0, 1].item(),
            control_second_bank[0, 0, 0, 1].item(),
        )

    def test_h1_mode_requires_an_aligned_parallel_stack(self):
        expert = FineTemporalPolarityMultiScaleExpert(
            input_channels=10,
            output_channels=8,
            hidden_channels=4,
            input_mode="h1_saturation",
        )
        frames = torch.zeros((2, 10, 8, 8))
        with self.assertRaisesRegex(ValueError, "parallel clip-8"):
            expert.paired_input_features(frames)
        with self.assertRaisesRegex(ValueError, "must match"):
            expert.paired_input_features(
                frames,
                expert_frames=torch.zeros((2, 10, 8, 7)),
            )

    def test_optimizer_scope_cannot_update_released_m20(self):
        base, model = _base_and_h1_model()
        base_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("high_density_expert.")
        }
        trainable_names = configure_expert_only_training(model)
        clip4 = torch.rand((1, 2, 10, 16, 24))
        clip8 = torch.rand_like(clip4)
        output = model(clip4, expert_frames=clip8)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(trainable_names)
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith("high_density_expert.")
        ))
        self.assertGreater(
            model.high_density_expert.output_projection.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertTrue(all(
            torch.equal(base_before[name], value)
            for name, value in model.state_dict().items()
            if name in base_before
        ))
        # Keep the local variable live so accidental aliasing with the model's
        # inherited state would be caught by the equality assertions above.
        self.assertIsNotNone(base)

    def test_inference_signature_and_state_have_no_label_or_source_channel(self):
        _, model = _base_and_h1_model()
        forbidden = ("label", "target", "source", "filename", "fold", "path", "hash")
        arguments = tuple(inspect.signature(model.forward).parameters)
        state_names = tuple(model.high_density_expert.state_dict())
        self.assertFalse(any(
            token in value.lower()
            for value in arguments + state_names
            for token in forbidden
        ))


if __name__ == "__main__":
    unittest.main()

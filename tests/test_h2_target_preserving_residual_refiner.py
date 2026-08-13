import copy
import unittest

import torch

from model.h2_target_preserving_residual_refiner import (
    FrozenM20TargetPreservingRefiner,
    TargetPreservingResidualHead,
    target_preserving_parameter_count,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.target_preserving_residual import (
    TargetRetentionDualState,
    complete_input_polarity_minority_fraction,
    input_only_routed_scores,
    target_preserving_event_loss,
    target_retention_constraints,
    use_h2_residual_refiner,
    validate_all_step_diagnostics,
)


def _base_model():
    torch.manual_seed(19)
    return BidirectionalTemporalMemoryNet(
        input_channels=10,
        width=4,
        density_calibration_enabled=False,
        temporal_attention_enabled=False,
    ).eval()


class TargetPreservingResidualTests(unittest.TestCase):
    def test_zero_gates_give_exact_identity(self):
        torch.manual_seed(23)
        head = TargetPreservingResidualHead(
            decoder_channels=4, hidden_channels=6,
        )
        decoder = torch.randn(1, 3, 4, 8, 10)
        logits = torch.randn(1, 3, 1, 8, 10)
        centre = torch.randn(1, 3, 3, 8, 10)
        parts = head(decoder, logits, centre, return_parts=True)
        self.assertEqual(torch.count_nonzero(parts['residual']).item(), 0)
        self.assertEqual(torch.count_nonzero(parts['protection']).item(), 0)
        self.assertEqual(torch.count_nonzero(parts['suppression']).item(), 0)
        self.assertTrue(torch.equal(logits + parts['residual'], logits))

    def test_branch_signs_are_structural_after_positive_gates(self):
        torch.manual_seed(29)
        head = TargetPreservingResidualHead(
            decoder_channels=4, hidden_channels=6,
        )
        with torch.no_grad():
            head.protection_raw_gate.fill_(0.2)
            head.suppression_raw_gate.fill_(0.3)
        parts = head(
            torch.randn(1, 2, 4, 6, 8),
            torch.randn(1, 2, 1, 6, 8),
            torch.randn(1, 2, 3, 6, 8),
            return_parts=True,
        )
        self.assertTrue(torch.all(parts['protection'] >= 0))
        self.assertTrue(torch.all(parts['suppression'] >= 0))
        self.assertTrue(torch.equal(
            parts['residual'], parts['protection'] - parts['suppression']
        ))

    def test_gate_projection_is_fail_closed(self):
        head = TargetPreservingResidualHead(
            decoder_channels=4, hidden_channels=6,
        )
        with torch.no_grad():
            head.protection_raw_gate.fill_(-0.1)
            head.suppression_raw_gate.fill_(-0.2)
        with self.assertRaisesRegex(RuntimeError, 'projected'):
            head.gate_values()
        head.project_gates_()
        protection, suppression = head.gate_values()
        self.assertEqual(float(protection), 0.0)
        self.assertEqual(float(suppression), 0.0)

    def test_both_zero_gates_have_finite_nonzero_first_step_gradients(self):
        torch.manual_seed(31)
        head = TargetPreservingResidualHead(
            decoder_channels=4, hidden_channels=6,
        )
        decoder = torch.randn(1, 2, 4, 4, 5)
        base_maps = torch.randn(1, 2, 1, 4, 5)
        centre = torch.rand(1, 2, 3, 4, 5)
        parts = head(decoder, base_maps, centre, return_parts=True)
        refined = (base_maps + parts['residual']).reshape(-1)
        base = base_maps.detach().reshape(-1)
        labels = (torch.arange(refined.numel()) % 3 == 0).float()
        target_ids = torch.where(
            labels > 0.5,
            torch.arange(refined.numel()) % 4 + 1,
            torch.zeros(refined.numel(), dtype=torch.long),
        )
        times = torch.arange(refined.numel()) // (4 * 5)
        loss, _, _, _ = target_preserving_event_loss(
            refined,
            base,
            labels,
            target_ids,
            times,
            TargetRetentionDualState(),
        )
        loss.backward()
        for gate in (
            head.protection_raw_gate,
            head.suppression_raw_gate,
        ):
            self.assertIsNotNone(gate.grad)
            self.assertTrue(torch.isfinite(gate.grad))
            self.assertGreater(abs(float(gate.grad)), 0.0)

    def test_real_wrapper_matches_base_at_initialization(self):
        base = _base_model()
        wrapper = FrozenM20TargetPreservingRefiner(
            base, context_bins=5, hidden_channels=6,
        ).eval()
        frames = torch.randn(1, 3, 10, 16, 24)
        with torch.no_grad():
            expected = base(frames)
            actual = wrapper(frames)
        self.assertTrue(torch.equal(actual, expected))

    def test_optimizer_step_and_projection_leave_m20_unchanged(self):
        base = _base_model()
        wrapper = FrozenM20TargetPreservingRefiner(
            base, context_bins=5, hidden_channels=6,
        ).train()
        frozen = copy.deepcopy(wrapper.frozen_state_dict())
        optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=1e-3)
        frames = torch.randn(1, 3, 10, 16, 24)
        refined, base_logits = wrapper(frames, return_base_logits=True)
        loss = (refined - 0.5).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wrapper.project_gates_()
        wrapper.refiner.gate_values()
        self.assertTrue(all(
            torch.equal(frozen[name], value)
            for name, value in wrapper.frozen_state_dict().items()
        ))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

    def test_constraints_are_zero_at_identity_and_positive_on_deficit(self):
        base = torch.tensor((2.0, 1.5, -1.0, 0.2))
        labels = torch.tensor((1.0, 1.0, 0.0, 1.0))
        target_ids = torch.tensor((1, 1, 0, 2))
        times = torch.tensor((0, 0, 0, 1))
        event, group, count = target_retention_constraints(
            base.clone(), base, labels, target_ids, times,
        )
        self.assertEqual(float(event), 0.0)
        self.assertEqual(float(group), 0.0)
        self.assertEqual(count, 2)

        refined = base.clone()
        refined[0] -= 0.4
        refined[1] -= 0.6
        event, group, count = target_retention_constraints(
            refined, base, labels, target_ids, times,
        )
        self.assertGreater(float(event), 0.0)
        self.assertGreater(float(group), 0.0)
        self.assertEqual(count, 2)

    def test_dual_ascent_is_monotone_and_event_loss_has_gradient(self):
        dual = TargetRetentionDualState()
        dual.update(torch.tensor(0.2), torch.tensor(0.3))
        self.assertAlmostEqual(dual.positive_event, 1.2, places=6)
        self.assertAlmostEqual(dual.target_group, 1.3, places=6)
        refined = torch.tensor((0.8, -0.2, 0.4), requires_grad=True)
        base = torch.tensor((1.0, 0.0, 0.6))
        labels = torch.tensor((1.0, 0.0, 1.0))
        target_ids = torch.tensor((1, 0, 2))
        times = torch.tensor((0, 0, 1))
        loss, event, group, diagnostics = target_preserving_event_loss(
            refined, base, labels, target_ids, times, dual,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(refined.grad).all())
        self.assertGreater(float(event), 0.0)
        self.assertGreater(float(group), 0.0)
        self.assertGreater(diagnostics['retention_term'], 0.0)

    def test_parameter_budget_and_state_names_are_source_free(self):
        wrapper = FrozenM20TargetPreservingRefiner(
            _base_model(), context_bins=5, hidden_channels=6,
        )
        self.assertLess(target_preserving_parameter_count(wrapper), 10_000)
        forbidden = ('label', 'target_id', 'source', 'filename', 'fold', 'path', 'hash')
        self.assertFalse(any(
            token in name.lower()
            for name in wrapper.refiner.state_dict()
            for token in forbidden
        ))

    def test_complete_input_route_excludes_h1_and_preserves_identity(self):
        h2_polarities = torch.tensor([0.0] * 40 + [1.0] * 60).numpy()
        h1_polarities = torch.tensor([0.0] * 10 + [1.0] * 90).numpy()
        self.assertAlmostEqual(
            complete_input_polarity_minority_fraction(h2_polarities), 0.4,
        )
        self.assertTrue(use_h2_residual_refiner(300000, h2_polarities.repeat(3000)))
        self.assertFalse(use_h2_residual_refiner(300000, h1_polarities.repeat(3000)))
        self.assertFalse(use_h2_residual_refiner(100, h2_polarities))
        base = torch.arange(100, dtype=torch.float32)
        candidate = base + 1.0
        routed = input_only_routed_scores(base, candidate, h1_polarities)
        self.assertTrue(torch.equal(routed, base))

    def test_all_step_diagnostics_must_be_complete_and_contiguous(self):
        record = {
            'step': 1,
            'loss': 0.5,
            'gradient_norm': 0.2,
            'positive_event_constraint': 0.0,
            'target_group_constraint': 0.0,
            'protection_gate_after': 0.01,
            'suppression_gate_after': 0.02,
            'dual_positive_event_after': 1.0,
            'dual_target_group_after': 1.0,
        }
        self.assertTrue(validate_all_step_diagnostics([record], 1))
        with self.assertRaisesRegex(RuntimeError, 'count'):
            validate_all_step_diagnostics([], 1)
        broken = dict(record)
        broken.pop('gradient_norm')
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            validate_all_step_diagnostics([broken], 1)


if __name__ == '__main__':
    unittest.main()

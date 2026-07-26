import unittest

import torch

from utils.checkpoint import (
    load_state_dict_with_optional_compatibility,
    load_state_dict_with_p2b_compatibility,
)


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class P2bModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.density_conv = torch.nn.Conv1d(1, 1, 1, bias=False)
        self.density_gate = torch.nn.Linear(1, 4)


class P11Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.p11_activity_projection = torch.nn.Conv1d(1, 2, 1, bias=False)


class P12Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.p12_density_projection = torch.nn.Conv1d(1, 2, 1, bias=False)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_baseline_state_initializes_only_p2b_auxiliaries(self):
        baseline = BaseModel()
        p2b_model = P2bModel()
        initial_gate = p2b_model.density_gate.weight.detach().clone()

        initialized_neutral_p2b = load_state_dict_with_p2b_compatibility(
            p2b_model,
            baseline.state_dict(),
            p2b_enabled=True,
        )

        self.assertTrue(initialized_neutral_p2b)
        self.assertTrue(torch.equal(p2b_model.linear.weight, baseline.linear.weight))
        self.assertTrue(torch.equal(p2b_model.density_gate.weight, initial_gate))

    def test_unrelated_missing_key_is_rejected(self):
        baseline = BaseModel()
        p2b_model = P2bModel()
        incomplete_state = baseline.state_dict()
        del incomplete_state['linear.bias']

        with self.assertRaises(RuntimeError):
            load_state_dict_with_p2b_compatibility(
                p2b_model,
                incomplete_state,
                p2b_enabled=True,
            )

    def test_baseline_model_keeps_strict_loading(self):
        baseline = BaseModel()
        p2b_model = P2bModel()

        with self.assertRaises(RuntimeError):
            load_state_dict_with_p2b_compatibility(
                baseline,
                p2b_model.state_dict(),
                p2b_enabled=False,
            )

    def test_baseline_state_initializes_only_p11_activity_projection(self):
        baseline = BaseModel()
        p11_model = P11Model()
        initial_projection = p11_model.p11_activity_projection.weight.detach().clone()

        missing_keys = load_state_dict_with_optional_compatibility(
            p11_model,
            baseline.state_dict(),
            p11_enabled=True,
        )

        self.assertEqual(missing_keys, ('p11_activity_projection.weight',))
        self.assertTrue(torch.equal(p11_model.linear.weight, baseline.linear.weight))
        self.assertTrue(
            torch.equal(
                p11_model.p11_activity_projection.weight,
                initial_projection,
            )
        )

    def test_baseline_state_initializes_only_p12_density_projection(self):
        baseline = BaseModel()
        p12_model = P12Model()
        initial_projection = p12_model.p12_density_projection.weight.detach().clone()

        missing_keys = load_state_dict_with_optional_compatibility(
            p12_model,
            baseline.state_dict(),
            p12_enabled=True,
        )

        self.assertEqual(missing_keys, ('p12_density_projection.weight',))
        self.assertTrue(torch.equal(p12_model.linear.weight, baseline.linear.weight))
        self.assertTrue(
            torch.equal(
                p12_model.p12_density_projection.weight,
                initial_projection,
            )
        )


if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from utils.global_attention import apply_batchwise_global_attention


def uniform_value_attention():
    """Build an attention layer whose output is each sequence mean."""
    attention = torch.nn.MultiheadAttention(
        embed_dim=1,
        num_heads=1,
        bias=False,
        dropout=0.0,
    )
    with torch.no_grad():
        # Q/K=0 yields uniform attention; V and output projections are identity.
        attention.in_proj_weight.copy_(torch.tensor([[0.0], [0.0], [1.0]]))
        attention.out_proj.weight.fill_(1.0)
    return attention


class GlobalAttentionTests(unittest.TestCase):
    def test_attends_across_tokens_in_one_sparse_sample(self):
        attention = uniform_value_attention()
        features = torch.tensor([[1.0], [3.0]])

        output = apply_batchwise_global_attention(
            attention,
            features,
            torch.tensor([0, 0]),
        )

        torch.testing.assert_close(output, torch.tensor([[2.0], [2.0]]))

    def test_never_mixes_tokens_from_different_sparse_samples(self):
        attention = uniform_value_attention()
        features = torch.tensor([[1.0], [3.0], [10.0], [14.0]])

        output = apply_batchwise_global_attention(
            attention,
            features,
            torch.tensor([0, 0, 1, 1]),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([[2.0], [2.0], [12.0], [12.0]]),
        )

    def test_rejects_misaligned_batch_indices(self):
        with self.assertRaisesRegex(ValueError, 'batch_indices'):
            apply_batchwise_global_attention(
                uniform_value_attention(),
                torch.zeros((2, 1)),
                torch.tensor([0]),
            )


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
    random_partition_indices,
    subset_inference_sample,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


class InferenceChunkTests(unittest.TestCase):
    def test_random_partition_is_complete_and_deterministic(self):
        first = random_partition_indices(11, 4, 37)
        second = random_partition_indices(11, 4, 37)

        self.assertEqual(len(first), 3)
        self.assertEqual([len(chunk) for chunk in first], [4, 4, 3])
        np.testing.assert_array_equal(
            np.concatenate(first),
            np.concatenate(second),
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate(first)),
            np.arange(11),
        )

    def test_config_requires_unique_seed_list(self):
        config = InferenceChunkConfig(
            enabled=True,
            event_count_cutoff=100,
            chunk_size=100,
            random_seeds=[13, 37],
        )
        self.assertTrue(config.should_partition(101))
        self.assertFalse(config.should_partition(100))
        self.assertEqual(config.random_seeds, (13, 37))

        with self.assertRaises(ValueError):
            InferenceChunkConfig(random_seeds=[37, 37])

    @unittest.skipUnless(HAS_TORCH, "requires the project's PyTorch runtime")
    def test_subset_and_evaluation_batch_preserve_source_order(self):
        sample = {
            "ev_loc": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "evs_norm": np.array([[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, -1.0], [0.7, 0.8, 0.9, 1.0]]),
            "seg_label": np.array([0.0, 1.0, 0.0]),
            "idx": np.array([10, 11, 12]),
        }
        subset = subset_inference_sample(sample, np.array([2, 0]))
        batch = evaluation_batch_from_sample(sample)

        np.testing.assert_array_equal(subset["idx"], np.array([12, 10]))
        np.testing.assert_array_equal(
            batch["locs"].numpy(),
            np.array([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]]),
        )
        np.testing.assert_array_equal(batch["idx_label"], sample["idx"])


if __name__ == "__main__":
    unittest.main()

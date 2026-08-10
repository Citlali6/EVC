"""CPU-only tests for the train-only persistent-pixel OOF audit."""

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import crossfit_persistent_pixel_prior as prior_oof  # noqa: E402


class PersistentPixelPriorTest(unittest.TestCase):
    def test_fold_plan_is_disjoint_and_covers_every_high_source_once(self):
        held = []
        for fold in prior_oof.FOLD_PLAN:
            names = tuple(fold["held_names"])
            self.assertTrue(names)
            self.assertTrue(
                all(prior_oof._domain_for_name(name) == fold["domain"] for name in names)
            )
            held.extend(names)
        self.assertEqual(len(held), len(set(held)))
        self.assertEqual(set(held), set(prior_oof.HIGH_NAMES))

    def test_observable_domain_route_uses_no_labels(self):
        self.assertEqual(prior_oof._observable_domain(0.019), "h1")
        self.assertEqual(prior_oof._observable_domain(0.199), "h1")
        self.assertEqual(prior_oof._observable_domain(0.200), "h2")
        self.assertEqual(prior_oof._observable_domain(0.466), "h2")

    def test_longest_active_runs_handles_gaps_and_multiple_pixels(self):
        bins = prior_oof.TEMPORAL_BIN_COUNT
        pairs = np.asarray(
            [
                2 * bins + 0,
                2 * bins + 1,
                2 * bins + 2,
                2 * bins + 4,
                7 * bins + 3,
                7 * bins + 5,
                7 * bins + 6,
            ],
            dtype=np.int64,
        )
        longest = prior_oof._longest_active_runs(pairs, pixel_count=10)
        self.assertEqual(int(longest[2]), 3)
        self.assertEqual(int(longest[7]), 2)
        self.assertEqual(int(longest[0]), 0)

    def test_derive_prior_reads_only_input_fields_and_matches_locations(self):
        locations = np.asarray(
            [
                [1, 2, 0],
                [1, 2, 1],
                [1, 2, 50],
                [1, 2, 100],
                [1, 2, 200],
                [3, 4, 0],
            ],
            dtype=np.int64,
        )
        dtype = np.dtype(
            [
                ("x", "<i2"),
                ("y", "<i2"),
                ("t", "<f8"),
                ("p", "i1"),
                ("label", "i1"),
                ("name", "i1"),
            ]
        )
        events = np.zeros(len(locations), dtype=dtype)
        events["x"] = locations[:, 0]
        events["y"] = locations[:, 1]
        events["t"] = locations[:, 2]
        events["p"] = 1
        # Deliberately populate labels; prior derivation must be unchanged because
        # it reads only ev_loc and the p field.
        events["label"] = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_000.npz"
            np.savez(path, ev_loc=locations, ev=events)
            prior = prior_oof.derive_pixel_prior(path, locations)

        pixel = 2 * prior_oof.WIDTH + 1
        self.assertEqual(prior.summary["event_count"], 6)
        self.assertEqual(prior.summary["unique_pixel_count"], 2)
        self.assertEqual(prior.summary["observable_domain"], "h1")
        self.assertAlmostEqual(float(prior.log_events[pixel]), np.log1p(5))
        self.assertAlmostEqual(float(prior.active_fraction[pixel]), 4 / 160)
        self.assertAlmostEqual(float(prior.longest_run_fraction[pixel]), 3 / 160)
        self.assertAlmostEqual(float(prior.collision_fraction[pixel]), 1 / 5)
        self.assertAlmostEqual(float(prior.log_max_bin_events[pixel]), np.log1p(2))
        self.assertAlmostEqual(float(prior.polarity_dominance[pixel]), 1.0)

        features = prior_oof.component_persistence_features(
            prior, (np.asarray([0, 1], dtype=np.int64),)
        )
        self.assertEqual(features.shape, (1, len(prior_oof.PERSISTENCE_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())

    def test_generic_logistic_fit_accepts_persistence_width_and_video_weights(self):
        features = np.asarray(
            [[-2.0, 0.0], [-1.0, 0.2], [1.0, 0.8], [2.0, 1.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        weights = np.full(4, 0.25, dtype=np.float64)
        fitted = prior_oof._fit_balanced_logistic(
            features, labels, weights, positive_weight=8.0
        )
        probabilities = prior_oof.component_crossfit._predict_probabilities(
            features, fitted
        )
        self.assertEqual(fitted["coefficients"].shape, (2,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertGreater(float(probabilities[-1]), float(probabilities[0]))

    def test_path_guard_rejects_validation_and_test_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            cache = root / "cache"
            train.mkdir()
            cache.mkdir()
            resolved_train, resolved_cache = prior_oof._validate_train_paths(train, cache)
            self.assertEqual(resolved_train, train.resolve())
            self.assertEqual(resolved_cache, cache.resolve())

            val_root = root / "val" / "train"
            val_root.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden split"):
                prior_oof._validate_train_paths(val_root, cache)

            test_cache = root / "test" / "cache"
            test_cache.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden split"):
                prior_oof._validate_train_paths(train, test_cache)


if __name__ == "__main__":
    unittest.main()

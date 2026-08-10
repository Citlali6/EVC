import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_temporal_memory_input_route_train import (
    OFFICIAL_TRAIN_NAMES,
    discover_official_train_sources,
    read_input_statistics,
)


class TrainRouteAuditGuardTests(unittest.TestCase):
    def test_rejects_root_not_named_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            root.mkdir()

            with self.assertRaisesRegex(ValueError, "named train"):
                discover_official_train_sources(root)

    def test_rejects_train_nested_under_validation_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation" / "train"
            root.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "forbidden split"):
                discover_official_train_sources(root)

    def test_requires_complete_canonical_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            root.mkdir()
            (root / OFFICIAL_TRAIN_NAMES[0]).touch()

            with self.assertRaisesRegex(ValueError, "population mismatch"):
                discover_official_train_sources(root)

    def test_rejects_non_train_filename_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val_000.npz"
            path.touch()

            with self.assertRaisesRegex(ValueError, "non-train"):
                read_input_statistics(path)

    def test_input_reader_routes_without_label_or_target_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_000.npz"
            event_count = 10
            evs_norm = np.zeros((event_count, 4), dtype=np.float32)
            evs_norm[:, 3] = np.asarray([0, 1] * 5)
            locations = np.column_stack(
                (
                    np.arange(event_count) % 3,
                    np.arange(event_count) % 2,
                    np.arange(event_count),
                )
            )
            np.savez(path, evs_norm=evs_norm, ev_loc=locations)

            record = read_input_statistics(path)

            self.assertEqual(record["event_count"], event_count)
            self.assertEqual(record["checkpoint_role"], "m10")
            self.assertEqual(record["mode"], "full_stream")
            self.assertEqual(record["polarity_minority_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()

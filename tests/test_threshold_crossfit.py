import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_threshold_crossfit import fold_for_file


class ThresholdCrossfitTests(unittest.TestCase):
    def test_fold_rule_is_filename_suffix_mod_five(self):
        self.assertEqual(fold_for_file("val_000.npz"), 0)
        self.assertEqual(fold_for_file("val_019.npz"), 4)
        self.assertEqual(fold_for_file("val_023.npz"), 3)

    def test_fold_rule_rejects_noncanonical_names(self):
        for value in ("val_24.npz", "train_000.npz", "val_024.npz"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fold_for_file(value)


if __name__ == "__main__":
    unittest.main()

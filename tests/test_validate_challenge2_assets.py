"""Synthetic regression tests for validate_challenge2_assets.py."""

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from validate_challenge2_assets import main, validate_assets


RAW_DTYPE = np.dtype(
    [
        ("x", np.int32),
        ("y", np.int32),
        ("t", np.float64),
        ("p", np.int8),
        ("label", np.int8),
        ("name", np.int32),
    ]
)


def make_arrays():
    raw = np.array(
        [
            (0, 0, 0.0, 0, 0, 0),
            (345, 259, 1.25, 1, 1, 4),
            (123, 45, 2.5, 0, 1, 4),
        ],
        dtype=RAW_DTYPE,
    )
    ev_loc = np.column_stack((raw["x"], raw["y"], raw["t"]))
    evs_norm = np.column_stack(
        (
            raw["x"] / 346.0,
            raw["y"] / 260.0,
            raw["t"] / 2.5,
            raw["p"],
            raw["label"],
            raw["name"],
        )
    ).astype(np.float32)
    return raw, evs_norm, ev_loc


def write_npz(path, arrays=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw, evs_norm, ev_loc = arrays or make_arrays()
    np.savez(path, ev=raw, evs_norm=evs_norm, ev_loc=ev_loc)


def valid_submission_text():
    raw, _, _ = make_arrays()
    output = io.StringIO()
    for row in raw:
        output.write(
            "{} {} {:.9f} {} {}\n".format(
                row["x"], row["y"], row["t"], row["p"], row["label"]
            )
        )
    return output.getvalue()


class AssetValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "数据集"
        write_npz(self.root / "train" / "train_000.npz")
        write_npz(self.root / "val" / "val_000.npz")

    def tearDown(self):
        self.temp_dir.cleanup()

    def validate(self, submission=None):
        return validate_assets(
            self.root,
            submission=submission,
            expected_train=1,
            expected_val=1,
        )

    def test_valid_dataset_directory_and_flat_zip(self):
        directory = Path(self.temp_dir.name) / "submission"
        directory.mkdir()
        (directory / "val_000.txt").write_text(
            valid_submission_text(), encoding="utf-8"
        )
        directory_result = self.validate(directory)
        self.assertTrue(directory_result["ok"], directory_result["errors"])
        self.assertEqual(directory_result["submission"]["rows"], 3)

        archive_path = Path(self.temp_dir.name) / "submission.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("val_000.txt", valid_submission_text())
        archive_result = self.validate(archive_path)
        self.assertTrue(archive_result["ok"], archive_result["errors"])
        self.assertEqual(archive_result["submission"]["kind"], "zip")

    def test_dataset_schema_values_and_counts_are_checked(self):
        raw, evs_norm, ev_loc = make_arrays()
        evs_norm[0, 4] = 0.5
        evs_norm[1, 5] = -1
        ev_loc[2, 0] = 346
        np.savez(
            self.root / "train" / "train_000.npz",
            ev=raw[:2],
            evs_norm=evs_norm,
            ev_loc=ev_loc,
        )
        result = self.validate()
        self.assertFalse(result["ok"])
        codes = {issue["code"] for issue in result["errors"]}
        self.assertIn("event_count_mismatch", codes)
        self.assertIn("coordinate_bounds", codes)
        self.assertIn("label_values", codes)
        self.assertIn("target_id_values", codes)

    def test_submission_rows_source_fields_and_labels_are_checked(self):
        directory = Path(self.temp_dir.name) / "submission"
        directory.mkdir()
        lines = valid_submission_text().splitlines()
        fields = lines[0].split()
        fields[0] = "1"
        fields[4] = "2"
        (directory / "val_000.txt").write_text(
            " ".join(fields) + "\n" + lines[1] + "\n", encoding="utf-8"
        )
        result = self.validate(directory)
        self.assertFalse(result["ok"])
        codes = {issue["code"] for issue in result["errors"]}
        self.assertIn("submission_source_mismatch", codes)
        self.assertIn("submission_label", codes)
        self.assertIn("submission_row_count", codes)

    def test_zip_must_be_flat(self):
        archive_path = Path(self.temp_dir.name) / "nested.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("nested/val_000.txt", valid_submission_text())
        result = self.validate(archive_path)
        self.assertFalse(result["ok"])
        codes = {issue["code"] for issue in result["errors"]}
        self.assertIn("zip_not_flat", codes)
        self.assertIn("submission_missing", codes)

    def test_json_cli_and_nonzero_failure_exit(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    str(self.root),
                    "--expected-train",
                    "1",
                    "--expected-val",
                    "1",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        json_output = stdout.getvalue()
        self.assertTrue(json_output.isascii())
        self.assertTrue(json.loads(json_output)["ok"])

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    str(self.root),
                    "--expected-train",
                    "99",
                    "--expected-val",
                    "24",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the strict Challenge 2 submission scorer."""

import contextlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import score_challenge2_submission as scorer_module
from score_challenge2_submission import (
    CANONICAL_STEMS,
    SubmissionScoreError,
    canonical_val_manifest_sha256,
    score_submission,
    validate_json_output_target,
    write_json_atomic,
)


RAW_DTYPE = np.dtype(
    [
        ("x", np.int16),
        ("y", np.int16),
        ("t", np.float64),
        ("p", np.int8),
        ("label", np.int8),
        ("name", np.int8),
    ]
)


def _video_arrays():
    # Raw/TXT timestamps intentionally have a very different range from ev_loc.
    # A false component therefore makes Fa detect accidental use of TXT t.
    ev = np.array(
        [
            (10, 20, 0.0009999999992942321, 0, 1, 1),
            (11, 21, 5001.0000000003, 1, 0, 0),
        ],
        dtype=RAW_DTYPE,
    )
    ev_loc = np.array([[10, 20, 1], [11, 21, 51]], dtype=np.int64)
    evs_norm = np.array(
        [
            [10 / 346.0, 20 / 260.0, 0.0, 0.0, 1.0, 1.0],
            [11 / 346.0, 21 / 260.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return ev, evs_norm, ev_loc


def _prediction_text(second_label="1"):
    return "10 20 0.001000000 0 1\n11 21 5001.000000000 1 {}\n".format(
        second_label
    )


class ChallengeSubmissionScorerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.val_root = self.root / "dataset" / "val"
        self.submission_dir = self.root / "submission"
        self.val_root.mkdir(parents=True)
        self.submission_dir.mkdir()
        ev, evs_norm, ev_loc = _video_arrays()
        for stem in CANONICAL_STEMS:
            np.savez(
                self.val_root / (stem + ".npz"),
                ev=ev,
                evs_norm=evs_norm,
                ev_loc=ev_loc,
            )
            (self.submission_dir / (stem + ".txt")).write_text(
                _prediction_text(), encoding="ascii"
            )

    def tearDown(self):
        self.temporary.cleanup()

    def score(self, submission=None):
        return score_submission(
            self.val_root,
            submission or self.submission_dir,
            allow_unofficial_dataset=True,
        )

    def test_directory_scores_with_npz_ev_loc_timestamps(self):
        report = score_submission(
            self.root / "dataset",
            self.submission_dir,
            allow_unofficial_dataset=True,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["counts"]["videos"], 24)
        self.assertEqual(report["counts"]["events"], 48)
        self.assertEqual(report["counts"]["evaluator_frames"], 24)
        self.assertEqual(report["counts"]["evaluator_false_components"], 24)
        self.assertEqual(report["counts"]["event_true_positives"], 24)
        self.assertEqual(report["counts"]["event_false_positives"], 24)
        self.assertEqual(report["counts"]["event_false_negatives"], 0)
        self.assertAlmostEqual(report["metrics"]["iou"], 0.5)
        self.assertAlmostEqual(report["metrics"]["acc"], 1.0)
        self.assertAlmostEqual(report["metrics"]["pd"], 1.0)
        self.assertAlmostEqual(report["metrics"]["fa"], 1.0 / (346 * 260))
        self.assertEqual(
            report["evaluator"]["roc_timestamp_source"],
            "NPZ ev_loc[:, 2] (not TXT column 3)",
        )
        self.assertEqual(
            report["dataset"]["verification_mode"], "UNOFFICIAL-UNLOCKED"
        )
        self.assertFalse(report["dataset"]["matches_official_sha256"])
        self.assertIn("not certified", report["dataset"]["warning"])
        tp = report["counts"]["event_true_positives"]
        fp = report["counts"]["event_false_positives"]
        fn = report["counts"]["event_false_negatives"]
        self.assertAlmostEqual(report["metrics"]["iou"], tp / (tp + fp + fn))
        self.assertAlmostEqual(report["metrics"]["acc"], tp / (tp + fn))
        self.assertEqual(report["evaluator"]["resolution"], {"width": 346, "height": 260})
        self.assertEqual(report["evaluator"]["score_fa_scale"], 10000.0)
        self.assertEqual(
            report["evaluator"]["score_weights"],
            {"pd": 0.4, "score_fa": 0.3, "iou": 0.2, "acc": 0.1},
        )
        self.assertEqual(len(report["evaluator"]["class_source_sha256"]), 64)
        self.assertEqual(len(report["evaluator"]["metrics_helper_source_sha256"]), 64)
        self.assertIsNotNone(report["provenance"]["opencv"])
        self.assertIsNotNone(report["provenance"]["pandas"])

    def test_flat_zip_scores_and_nested_zip_is_rejected(self):
        archive_path = self.root / "submission.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for stem in CANONICAL_STEMS:
                archive.writestr(stem + ".txt", _prediction_text())
        report = self.score(archive_path)
        self.assertEqual(report["submission"]["kind"], "zip")
        self.assertIn("canonical_payload_sha256", report["submission"])

        nested_path = self.root / "nested.zip"
        with zipfile.ZipFile(nested_path, "w") as archive:
            for stem in CANONICAL_STEMS:
                archive.writestr("nested/" + stem + ".txt", _prediction_text())
        with self.assertRaisesRegex(SubmissionScoreError, "archive root"):
            self.score(nested_path)

    def test_source_mismatch_and_non_integer_label_are_rejected(self):
        mismatch = self.submission_dir / "val_000.txt"
        mismatch.write_text(
            "999 20 0.001000000 0 1\n11 21 5001.000000000 1 1\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(SubmissionScoreError, "differs from source ev"):
            self.score()

        mismatch.write_text(_prediction_text(second_label="1.0"), encoding="ascii")
        with self.assertRaisesRegex(SubmissionScoreError, "integer 0 or 1"):
            self.score()

    def test_json_write_is_atomic_and_refuses_overwrite_by_default(self):
        report = self.score()
        output = self.root / "audit" / "score.json"
        written = write_json_atomic(output, report)
        self.assertEqual(written, output.resolve())
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "passed")
        with self.assertRaises(FileExistsError):
            write_json_atomic(output, report)

    def _assert_cli_rejects_alias_before_scoring(self, output, submission=None):
        stderr = io.StringIO()
        argv = [
            "--val-root",
            str(self.val_root),
            "--submission",
            str(submission or self.submission_dir),
            "--json-out",
            str(output),
            "--overwrite-json",
        ]
        with mock.patch.object(scorer_module, "score_submission") as score_mock:
            with contextlib.redirect_stderr(stderr):
                return_code = scorer_module.main(argv)
        self.assertEqual(return_code, 2)
        score_mock.assert_not_called()
        self.assertIn("aliases protected scoring input", stderr.getvalue())

    def test_json_output_cannot_alias_submission_txt(self):
        alias = self.root / "txt-alias.json"
        os.link(self.submission_dir / "val_000.txt", alias)
        self._assert_cli_rejects_alias_before_scoring(alias)

    def test_json_output_cannot_alias_submission_zip(self):
        archive_path = self.root / "submission.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for stem in CANONICAL_STEMS:
                archive.writestr(stem + ".txt", _prediction_text())
        alias = self.root / "zip-alias.json"
        os.link(archive_path, alias)
        self._assert_cli_rejects_alias_before_scoring(alias, archive_path)

    def test_json_output_cannot_alias_validation_npz(self):
        alias = self.root / "npz-alias.json"
        os.link(self.val_root / "val_000.npz", alias)
        self._assert_cli_rejects_alias_before_scoring(alias)

    def test_json_output_requires_json_suffix_and_normal_target_is_allowed(self):
        with self.assertRaisesRegex(SubmissionScoreError, "\.json extension"):
            validate_json_output_target(
                self.val_root,
                self.submission_dir,
                self.root / "score.txt",
            )
        normal = self.root / "audit" / "score.json"
        self.assertEqual(
            validate_json_output_target(
                self.val_root, self.submission_dir, normal
            ),
            normal.resolve(),
        )

    def test_dataset_content_manifest_is_locked_by_default(self):
        with self.assertRaisesRegex(
            SubmissionScoreError, "dataset manifest SHA-256 mismatch"
        ):
            score_submission(self.val_root, self.submission_dir)

        _, original_manifest, _ = canonical_val_manifest_sha256(self.val_root)
        ev, evs_norm, ev_loc = _video_arrays()
        ev_loc = ev_loc.copy()
        ev_loc[0, 2] = 2
        np.savez(
            self.val_root / "val_000.npz",
            ev=ev,
            evs_norm=evs_norm,
            ev_loc=ev_loc,
        )
        _, changed_manifest, _ = canonical_val_manifest_sha256(self.val_root)
        self.assertNotEqual(original_manifest, changed_manifest)
        with self.assertRaisesRegex(
            SubmissionScoreError, "dataset manifest SHA-256 mismatch"
        ):
            score_submission(self.val_root, self.submission_dir)

    def test_npz_changed_after_load_is_rejected(self):
        original_loader = scorer_module._load_validation_video
        changed = False

        def load_then_replace(path):
            nonlocal changed
            arrays = original_loader(path)
            if not changed:
                ev, evs_norm, ev_loc = _video_arrays()
                ev_loc = ev_loc.copy()
                ev_loc[0, 2] = 2
                np.savez(path, ev=ev, evs_norm=evs_norm, ev_loc=ev_loc)
                changed = True
            return arrays

        with mock.patch.object(
            scorer_module, "_load_validation_video", side_effect=load_then_replace
        ):
            with self.assertRaisesRegex(
                SubmissionScoreError, "changed while being scored"
            ):
                self.score()

    def test_source_row_and_zip_total_byte_limits_are_enforced(self):
        oversized = self.submission_dir / "val_000.txt"
        oversized.write_bytes(
            b"0" * (2 * scorer_module.MAX_SUBMISSION_BYTES_PER_SOURCE_ROW + 1)
        )
        with self.assertRaisesRegex(
            SubmissionScoreError, "source-row-derived .* byte limit"
        ):
            self.score()

        archive_path = self.root / "submission.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for stem in CANONICAL_STEMS:
                archive.writestr(stem + ".txt", _prediction_text())
        with mock.patch.object(scorer_module, "MAX_TOTAL_SUBMISSION_BYTES", 100):
            with self.assertRaisesRegex(
                SubmissionScoreError, "uncompressed payload exceeds"
            ):
                self.score(archive_path)


if __name__ == "__main__":
    unittest.main()

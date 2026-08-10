"""Strictly validate and score an EV-UAV Challenge 2 submission offline.

The scorer accepts either a directory containing the 24 canonical validation
TXT files or a flat ZIP containing those files at its root.  Submission rows
are treated as predictions only after their first four columns have been
verified against the corresponding validation ``ev`` array.

This module deliberately reuses the project's unchanged Challenge 2 evaluator.
In particular, Pd/Fa frame assignment uses the integer timestamp in
``ev_loc[:, 2]`` just like the validation dataloader; the floating-point TXT
timestamp is only checked for provenance and is never used for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from utils.challenge_eval import (
    SCORE_FA_SCALE,
    add_batch_to_evaluator,
    evaluate_challenge_metrics,
)
from utils.eval import evalute


CANONICAL_VIDEO_COUNT = 24
CANONICAL_STEMS = tuple(
    "val_{:03d}".format(index) for index in range(CANONICAL_VIDEO_COUNT)
)
CANONICAL_TXT_NAMES = tuple(stem + ".txt" for stem in CANONICAL_STEMS)
CANONICAL_NPZ_NAMES = tuple(stem + ".npz" for stem in CANONICAL_STEMS)
REQUIRED_NPZ_KEYS = frozenset(("ev", "evs_norm", "ev_loc"))
REQUIRED_EV_FIELDS = ("x", "y", "t", "p", "label", "name")
PREDICTION_THRESHOLD = 0.9
PD_DETECTION_INTERVAL = 50
CORRECT_THRESHOLD = 0.0001
# ``submit_challenge2.save_prediction`` serializes t with ``%.9f``.  Accept no
# more than one half-unit at that published precision (plus a tiny float parse
# allowance), while keeping integer x/y/p comparisons exact.
TIMESTAMP_SERIALIZATION_ATOL = 5.1e-10
OFFICIAL_VAL_MANIFEST_SHA256 = (
    "d780da17e69446b988b1b5fae7954855d5ce66a32aa7b9581eeb3e4a0563f83f"
)
MAX_SUBMISSION_MEMBER_BYTES = 256 * 1024 * 1024
# The two audited canonical prediction sets are 38,668,635 bytes in total and
# at most 28 bytes per row.  These conservative limits leave ample formatting
# headroom while preventing ``read + splitlines + loadtxt(float64)`` expansion
# from becoming a practical memory-denial vector on the 16 GB workstation.
MAX_SUBMISSION_BYTES_PER_SOURCE_ROW = 64
OFFICIAL_CANONICAL_SUBMISSION_BYTES = 38_668_635
MAX_TOTAL_SUBMISSION_BYTES = 4 * OFFICIAL_CANONICAL_SUBMISSION_BYTES
HASH_BLOCK_BYTES = 1024 * 1024


class SubmissionScoreError(ValueError):
    """Raised when an input cannot be scored without changing its meaning."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _manifest_sha256(entries: Iterable[Tuple[str, str]]) -> str:
    """Hash canonical ``(name, content_sha256)`` entries deterministically."""

    digest = hashlib.sha256()
    for name, content_sha256 in entries:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(content_sha256))
    return digest.hexdigest()


def _resolve_val_root(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser().resolve()
    direct_names = {candidate.name for candidate in path.glob("*.npz")}
    if set(CANONICAL_NPZ_NAMES).issubset(direct_names):
        val_root = path
    elif (path / "val").is_dir():
        val_root = (path / "val").resolve()
    else:
        raise SubmissionScoreError(
            "validation root must contain val_000.npz ... val_023.npz "
            "directly or in a val/ child: {}".format(path)
        )

    actual_names = {
        candidate.name
        for candidate in val_root.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".npz"
    }
    expected_names = set(CANONICAL_NPZ_NAMES)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise SubmissionScoreError(
            "validation split must contain exactly the 24 canonical NPZ files; "
            "missing={}, extra={}".format(missing, extra)
        )
    return val_root


def canonical_val_manifest_sha256(
    val_root: Union[str, Path],
) -> Tuple[Path, str, Dict[str, str]]:
    """Hash all 24 canonical validation NPZ payloads in canonical order."""

    resolved_val_root = _resolve_val_root(val_root)
    member_hashes = {
        name: _sha256_file(resolved_val_root / name)
        for name in CANONICAL_NPZ_NAMES
    }
    manifest_hash = _manifest_sha256(
        (name, member_hashes[name]) for name in CANONICAL_NPZ_NAMES
    )
    return resolved_val_root, manifest_hash, member_hashes


def _paths_alias(output_path: Path, protected_path: Path) -> bool:
    """Return whether two paths normalize to, or identify, the same file.

    ``samefile`` catches hard-link aliases that normalization cannot.  When
    both paths exist, inability to establish their identity is treated as an
    unsafe condition instead of silently allowing a possible overwrite.
    """

    output_path = Path(output_path).expanduser().resolve()
    protected_path = Path(protected_path).expanduser().resolve()
    if os.path.normcase(str(output_path)) == os.path.normcase(str(protected_path)):
        return True
    if output_path.exists() and protected_path.exists():
        try:
            return os.path.samefile(str(output_path), str(protected_path))
        except OSError as exc:
            raise SubmissionScoreError(
                "cannot safely compare JSON output {} with protected input {}: {}".format(
                    output_path, protected_path, exc
                )
            ) from exc
    return False


def validate_json_output_target(
    val_root: Union[str, Path],
    submission_path: Union[str, Path],
    json_output_path: Union[str, Path],
) -> Path:
    """Reject JSON output paths that could overwrite any scoring input."""

    output_path = Path(json_output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise SubmissionScoreError(
            "JSON output path must have a .json extension: {}".format(output_path)
        )
    if output_path.is_dir():
        raise SubmissionScoreError(
            "JSON output path is an existing directory: {}".format(output_path)
        )

    resolved_val_root = _resolve_val_root(val_root)
    resolved_submission = Path(submission_path).expanduser().resolve()
    protected_paths = [
        resolved_val_root / npz_name for npz_name in CANONICAL_NPZ_NAMES
    ]
    if resolved_submission.is_dir():
        protected_paths.extend(
            resolved_submission / txt_name for txt_name in CANONICAL_TXT_NAMES
        )
    else:
        # Invalid/nonexistent submission paths are diagnosed by the normal
        # reader later.  Protect the supplied path itself in every file case.
        protected_paths.append(resolved_submission)

    for protected_path in protected_paths:
        if _paths_alias(output_path, protected_path):
            raise SubmissionScoreError(
                "JSON output aliases protected scoring input {}; choose a new "
                ".json file".format(protected_path)
            )
    return output_path


class _SubmissionReader:
    """Read canonical TXT payloads without extracting archives."""

    def __init__(self, submission_path: Union[str, Path]):
        self.path = Path(submission_path).expanduser().resolve()
        self.kind = ""
        self._archive: Optional[zipfile.ZipFile] = None
        self._zip_infos: Dict[str, zipfile.ZipInfo] = {}

    def __enter__(self) -> "_SubmissionReader":
        expected = set(CANONICAL_TXT_NAMES)
        if self.path.is_dir():
            self.kind = "directory"
            direct_txt = {
                candidate.name
                for candidate in self.path.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".txt"
            }
            nested_txt = [
                candidate
                for candidate in self.path.rglob("*.txt")
                if candidate.parent != self.path
            ]
            if nested_txt:
                raise SubmissionScoreError(
                    "submission TXT files must be direct children; nested files: {}".format(
                        [str(path.relative_to(self.path)) for path in nested_txt]
                    )
                )
            self._check_names(direct_txt, expected)
            return self

        if not self.path.is_file() or self.path.suffix.lower() != ".zip":
            raise SubmissionScoreError(
                "submission must be a TXT directory or .zip file: {}".format(
                    self.path
                )
            )

        self.kind = "zip"
        try:
            self._archive = zipfile.ZipFile(str(self.path), "r")
            infos = self._archive.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise SubmissionScoreError(
                "cannot open submission ZIP {}: {}".format(self.path, exc)
            ) from exc

        try:
            names = []
            total_uncompressed = 0
            for info in infos:
                if info.is_dir() or "/" in info.filename or "\\" in info.filename:
                    raise SubmissionScoreError(
                        "ZIP entries must be files at the archive root: {!r}".format(
                            info.filename
                        )
                    )
                if not info.filename.endswith(".txt"):
                    raise SubmissionScoreError(
                        "ZIP may contain only canonical TXT files: {!r}".format(
                            info.filename
                        )
                    )
                if info.flag_bits & 0x1:
                    raise SubmissionScoreError(
                        "encrypted ZIP members are not supported: {!r}".format(
                            info.filename
                        )
                    )
                if info.file_size > MAX_SUBMISSION_MEMBER_BYTES:
                    raise SubmissionScoreError(
                        "ZIP member exceeds the {} byte safety limit: {!r}".format(
                            MAX_SUBMISSION_MEMBER_BYTES, info.filename
                        )
                    )
                names.append(info.filename)
                total_uncompressed += info.file_size
                if info.filename in self._zip_infos:
                    raise SubmissionScoreError(
                        "duplicate ZIP member: {!r}".format(info.filename)
                    )
                self._zip_infos[info.filename] = info
            if total_uncompressed > MAX_TOTAL_SUBMISSION_BYTES:
                raise SubmissionScoreError(
                    "ZIP uncompressed payload exceeds the {} byte canonical "
                    "safety limit".format(MAX_TOTAL_SUBMISSION_BYTES)
                )
            self._check_names(set(names), expected)
        except Exception:
            self._archive.close()
            self._archive = None
            raise
        return self

    @staticmethod
    def _check_names(actual: set, expected: set) -> None:
        if actual != expected:
            raise SubmissionScoreError(
                "submission must contain exactly val_000.txt ... val_023.txt; "
                "missing={}, extra={}".format(
                    sorted(expected - actual), sorted(actual - expected)
                )
            )

    def read(self, name: str, source_event_count: int) -> bytes:
        if name not in CANONICAL_TXT_NAMES:
            raise SubmissionScoreError("non-canonical submission member: {}".format(name))
        if source_event_count <= 0:
            raise SubmissionScoreError(
                "source event count must be positive for {}".format(name)
            )
        dynamic_limit = min(
            MAX_SUBMISSION_MEMBER_BYTES,
            source_event_count * MAX_SUBMISSION_BYTES_PER_SOURCE_ROW,
        )
        if self.kind == "directory":
            path = self.path / name
            if path.is_symlink():
                raise SubmissionScoreError(
                    "submission TXT files may not be symbolic links: {}".format(path)
                )
            size = path.stat().st_size
            if size > dynamic_limit:
                raise SubmissionScoreError(
                    "submission TXT exceeds the source-row-derived {} byte "
                    "limit ({} rows x {} bytes): {}".format(
                        dynamic_limit,
                        source_event_count,
                        MAX_SUBMISSION_BYTES_PER_SOURCE_ROW,
                        path,
                    )
                )
            return path.read_bytes()
        assert self._archive is not None
        info = self._zip_infos[name]
        if info.file_size > dynamic_limit:
            raise SubmissionScoreError(
                "ZIP member {!r} exceeds the source-row-derived {} byte limit "
                "({} rows x {} bytes)".format(
                    name,
                    dynamic_limit,
                    source_event_count,
                    MAX_SUBMISSION_BYTES_PER_SOURCE_ROW,
                )
            )
        try:
            return self._archive.read(info)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SubmissionScoreError(
                "cannot read ZIP member {!r}: {}".format(name, exc)
            ) from exc

    def submission_hashes(self, member_hashes: Iterable[Tuple[str, str]]) -> dict:
        payload_hash = _manifest_sha256(member_hashes)
        if self.kind == "zip":
            return {
                "sha256": _sha256_file(self.path),
                "sha256_scheme": "raw-zip-bytes",
                "canonical_payload_sha256": payload_hash,
                "canonical_payload_sha256_scheme": (
                    "sha256(name_utf8 + NUL + member_sha256_bytes), canonical order"
                ),
            }
        return {
            "sha256": payload_hash,
            "sha256_scheme": (
                "sha256(name_utf8 + NUL + member_sha256_bytes), canonical order"
            ),
        }

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None


def _load_validation_video(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            keys = set(archive.files)
            if not REQUIRED_NPZ_KEYS.issubset(keys):
                raise SubmissionScoreError(
                    "{} is missing NPZ keys: {}".format(
                        path, sorted(REQUIRED_NPZ_KEYS - keys)
                    )
                )
            ev = np.asarray(archive["ev"])
            evs_norm = np.asarray(archive["evs_norm"])
            ev_loc = np.asarray(archive["ev_loc"])
    except SubmissionScoreError:
        raise
    except Exception as exc:
        raise SubmissionScoreError("cannot read {}: {}".format(path, exc)) from exc

    names = ev.dtype.names
    if ev.ndim != 1 or names is None:
        raise SubmissionScoreError("{}: ev must be a 1-D structured array".format(path))
    missing_fields = [field for field in REQUIRED_EV_FIELDS if field not in names]
    if missing_fields:
        raise SubmissionScoreError(
            "{}: ev is missing fields {}".format(path, missing_fields)
        )
    if ev_loc.ndim != 2 or ev_loc.shape[1] != 3:
        raise SubmissionScoreError("{}: ev_loc must have shape (N, 3)".format(path))
    if evs_norm.ndim != 2 or evs_norm.shape[1] < 6:
        raise SubmissionScoreError(
            "{}: evs_norm must have at least six columns".format(path)
        )
    event_count = len(ev)
    if event_count == 0:
        raise SubmissionScoreError("{}: validation video is empty".format(path))
    if len(evs_norm) != event_count or len(ev_loc) != event_count:
        raise SubmissionScoreError(
            "{}: ev, evs_norm, and ev_loc event counts differ".format(path)
        )
    if not np.isfinite(evs_norm).all() or not np.isfinite(ev_loc).all():
        raise SubmissionScoreError("{}: validation arrays contain NaN/Inf".format(path))
    for field in REQUIRED_EV_FIELDS:
        if not np.issubdtype(ev[field].dtype, np.number):
            raise SubmissionScoreError(
                "{}: ev field {!r} is not numeric".format(path, field)
            )
        if not np.isfinite(ev[field]).all():
            raise SubmissionScoreError(
                "{}: ev field {!r} contains NaN/Inf".format(path, field)
            )
    if not np.array_equal(ev_loc[:, 0], ev["x"]) or not np.array_equal(
        ev_loc[:, 1], ev["y"]
    ):
        raise SubmissionScoreError("{}: ev_loc x/y differ from ev x/y".format(path))
    if not np.logical_or(ev["label"] == 0, ev["label"] == 1).all():
        raise SubmissionScoreError("{}: ground-truth labels are not binary".format(path))
    if np.any(ev["name"] < 0) or not np.equal(
        ev["name"], np.floor(ev["name"])
    ).all():
        raise SubmissionScoreError("{}: target IDs must be non-negative integers".format(path))
    if not np.array_equal(evs_norm[:, 4], ev["label"]):
        raise SubmissionScoreError("{}: evs_norm labels differ from ev labels".format(path))
    if not np.array_equal(evs_norm[:, 5], ev["name"]):
        raise SubmissionScoreError("{}: evs_norm target IDs differ from ev names".format(path))
    return ev, evs_norm, ev_loc


def _is_integer_token(token: bytes) -> bool:
    if not token:
        return False
    if token[:1] in (b"+", b"-"):
        token = token[1:]
    return bool(token) and token.isdigit()


def _parse_submission_payload(
    payload: bytes, display_name: str, source_ev: np.ndarray
) -> np.ndarray:
    if not payload:
        raise SubmissionScoreError("{} is empty".format(display_name))
    if len(payload) > MAX_SUBMISSION_MEMBER_BYTES:
        raise SubmissionScoreError(
            "{} exceeds the {} byte safety limit".format(
                display_name, MAX_SUBMISSION_MEMBER_BYTES
            )
        )

    lines = payload.splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        fields = raw_line.strip().split()
        if len(fields) != 5:
            raise SubmissionScoreError(
                "{} line {} must contain exactly five columns".format(
                    display_name, line_number
                )
            )
        if not all(_is_integer_token(fields[index]) for index in (0, 1, 3)):
            raise SubmissionScoreError(
                "{} line {} x, y, and p must use integer tokens".format(
                    display_name, line_number
                )
            )
        if fields[4] not in (b"0", b"1"):
            raise SubmissionScoreError(
                "{} line {} label must be the integer 0 or 1".format(
                    display_name, line_number
                )
            )

    try:
        values = np.loadtxt(io.BytesIO(payload), dtype=np.float64, ndmin=2)
    except (UnicodeError, ValueError) as exc:
        raise SubmissionScoreError(
            "{} contains invalid numeric text: {}".format(display_name, exc)
        ) from exc
    expected_shape = (len(source_ev), 5)
    if values.shape != expected_shape:
        raise SubmissionScoreError(
            "{} must have shape {}, found {}".format(
                display_name, expected_shape, values.shape
            )
        )
    if not np.isfinite(values).all():
        raise SubmissionScoreError("{} contains NaN or infinity".format(display_name))

    source_columns = ("x", "y", "t", "p")
    for column_index, field in enumerate(source_columns):
        actual = values[:, column_index]
        expected = source_ev[field]
        if field == "t":
            matches = np.isclose(
                actual,
                expected,
                rtol=0.0,
                atol=TIMESTAMP_SERIALIZATION_ATOL,
            )
        else:
            matches = actual == expected
        if not matches.all():
            mismatch = int(np.flatnonzero(~matches)[0])
            raise SubmissionScoreError(
                "{} row {} column {!r} differs from source ev: {} != {}".format(
                    display_name,
                    mismatch + 1,
                    field,
                    actual[mismatch],
                    expected[mismatch],
                )
            )
    return values[:, 4].astype(np.float32, copy=False)


def _evaluation_batch(
    ground_truth: np.ndarray, target_ids: np.ndarray, ev_loc: np.ndarray
) -> dict:
    batch_column = np.zeros((len(ev_loc), 1), dtype=np.int64)
    locations = np.concatenate(
        (batch_column, np.asarray(ev_loc, dtype=np.int64)), axis=1
    )
    return {
        "seg_label": torch.from_numpy(
            np.asarray(ground_truth, dtype=np.float32)
        ),
        "idx_label": np.asarray(target_ids),
        "locs": torch.from_numpy(locations).long().contiguous(),
    }


def score_submission(
    val_root: Union[str, Path],
    submission_path: Union[str, Path],
    allow_unofficial_dataset: bool = False,
) -> dict:
    """Validate and score one canonical validation submission."""

    (
        resolved_val_root,
        dataset_manifest_sha256,
        source_hashes,
    ) = canonical_val_manifest_sha256(val_root)
    expected_manifest_sha256 = (
        None if allow_unofficial_dataset else OFFICIAL_VAL_MANIFEST_SHA256
    )
    if not allow_unofficial_dataset:
        if dataset_manifest_sha256 != OFFICIAL_VAL_MANIFEST_SHA256:
            raise SubmissionScoreError(
                "validation dataset manifest SHA-256 mismatch: expected {}, found {}. "
                "Refusing to report this as an official validation score.".format(
                    OFFICIAL_VAL_MANIFEST_SHA256, dataset_manifest_sha256
                )
            )
    evaluator_cfg = SimpleNamespace(
        roc=True,
        pd_detT=PD_DETECTION_INTERVAL,
        correct_thresh=CORRECT_THRESHOLD,
    )
    evaluator = evalute(evaluator_cfg)
    sample_number = 0
    total_events = 0
    predicted_positives = 0
    ground_truth_positives = 0
    event_true_positives = 0
    event_false_positives = 0
    event_false_negatives = 0
    file_reports = []
    submission_member_hashes = []

    with _SubmissionReader(submission_path) as submission:
        for stem, txt_name, npz_name in zip(
            CANONICAL_STEMS, CANONICAL_TXT_NAMES, CANONICAL_NPZ_NAMES
        ):
            npz_path = resolved_val_root / npz_name
            source_sha256 = source_hashes[npz_name]
            source_ev, _evs_norm, ev_loc = _load_validation_video(npz_path)
            post_load_source_sha256 = _sha256_file(npz_path)
            if post_load_source_sha256 != source_sha256:
                raise SubmissionScoreError(
                    "validation NPZ changed while being scored: {} (locked {}, "
                    "post-load {}). Refusing a mixed-provenance report.".format(
                        npz_path, source_sha256, post_load_source_sha256
                    )
                )

            payload = submission.read(txt_name, len(source_ev))
            prediction_sha256 = _sha256_bytes(payload)
            submission_member_hashes.append((txt_name, prediction_sha256))
            predictions = _parse_submission_payload(
                payload,
                "{}!{}".format(submission.path, txt_name),
                source_ev,
            )

            batch = _evaluation_batch(
                source_ev["label"], source_ev["name"], ev_loc
            )
            sample_number = add_batch_to_evaluator(
                evaluator,
                batch,
                torch.from_numpy(predictions),
                sample_number,
                PREDICTION_THRESHOLD,
                collect_roc=True,
            )

            event_count = int(len(source_ev))
            prediction_positive_count = int(np.count_nonzero(predictions == 1))
            ground_truth_positive_count = int(
                np.count_nonzero(source_ev["label"] == 1)
            )
            ground_truth = source_ev["label"]
            event_true_positives += int(
                np.count_nonzero((predictions == 1) & (ground_truth == 1))
            )
            event_false_positives += int(
                np.count_nonzero((predictions == 1) & (ground_truth == 0))
            )
            event_false_negatives += int(
                np.count_nonzero((predictions == 0) & (ground_truth == 1))
            )
            total_events += event_count
            predicted_positives += prediction_positive_count
            ground_truth_positives += ground_truth_positive_count
            file_reports.append(
                {
                    "stem": stem,
                    "events": event_count,
                    "predicted_positive_events": prediction_positive_count,
                    "ground_truth_positive_events": ground_truth_positive_count,
                    "txt_sha256": prediction_sha256,
                    "npz_sha256": source_sha256,
                }
            )

        metrics = evaluate_challenge_metrics(evaluator, PREDICTION_THRESHOLD)
        submission_hashes = submission.submission_hashes(submission_member_hashes)
        submission_kind = submission.kind
        resolved_submission_path = submission.path

    script_path = Path(__file__).resolve()
    challenge_eval_path = Path(sys.modules["utils.challenge_eval"].__file__).resolve()
    evaluator_path = Path(sys.modules["utils.eval"].__file__).resolve()
    matches_official_manifest = (
        dataset_manifest_sha256 == OFFICIAL_VAL_MANIFEST_SHA256
    )
    if allow_unofficial_dataset:
        dataset_verification_mode = "UNOFFICIAL-UNLOCKED"
        dataset_warning = (
            "Dataset identity was not enforced; metrics are not certified as "
            "official-validation results."
        )
    else:
        dataset_verification_mode = "OFFICIAL-MANIFEST-LOCKED"
        dataset_warning = None

    report = {
        "schema_version": 1,
        "status": "passed",
        "task": "EV-UAV Challenge 2 validation",
        "metrics": metrics.to_dict(),
        "counts": {
            "videos": CANONICAL_VIDEO_COUNT,
            "events": total_events,
            "predicted_positive_events": predicted_positives,
            "ground_truth_positive_events": ground_truth_positives,
            "event_true_positives": event_true_positives,
            "event_false_positives": event_false_positives,
            "event_false_negatives": event_false_negatives,
            "evaluator_objects": int(evaluator.obj_num),
            "evaluator_detected_objects": int(evaluator.correct_num),
            "evaluator_false_components": int(evaluator.false_num),
            "evaluator_frames": int(evaluator.frame_num),
        },
        "submission": {
            "path": str(resolved_submission_path),
            "kind": submission_kind,
            **submission_hashes,
        },
        "dataset": {
            "val_root": str(resolved_val_root),
            "sha256": dataset_manifest_sha256,
            "sha256_scheme": (
                "sha256(name_utf8 + NUL + member_sha256_bytes), canonical order"
            ),
            "verification_mode": dataset_verification_mode,
            "expected_sha256": expected_manifest_sha256,
            "official_sha256": OFFICIAL_VAL_MANIFEST_SHA256,
            "matches_official_sha256": matches_official_manifest,
            "warning": dataset_warning,
        },
        "evaluator": {
            "class": "utils.eval.evalute",
            "class_source": str(evaluator_path),
            "class_source_sha256": _sha256_file(evaluator_path),
            "metrics_helper": "utils.challenge_eval.evaluate_challenge_metrics",
            "metrics_helper_source": str(challenge_eval_path),
            "metrics_helper_source_sha256": _sha256_file(challenge_eval_path),
            "resolution": {"width": 346, "height": 260},
            "prediction_threshold": PREDICTION_THRESHOLD,
            "pd_detection_interval": PD_DETECTION_INTERVAL,
            "correct_threshold": CORRECT_THRESHOLD,
            "score_fa_scale": SCORE_FA_SCALE,
            "score_weights": {
                "pd": 0.4,
                "score_fa": 0.3,
                "iou": 0.2,
                "acc": 0.1,
            },
            "prediction_source": "TXT column 5",
            "ground_truth_label_source": "NPZ ev['label']",
            "target_id_source": "NPZ ev['name']",
            "roc_timestamp_source": "NPZ ev_loc[:, 2] (not TXT column 3)",
        },
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(script_path),
            "script_sha256": _sha256_file(script_path),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": getattr(sys.modules.get("cv2"), "__version__", None),
            "pandas": getattr(sys.modules.get("pandas"), "__version__", None),
        },
        "files": file_reports,
    }
    return report


def write_json_atomic(
    path: Union[str, Path], report: dict, overwrite: bool = False
) -> Path:
    """Atomically publish UTF-8 JSON, refusing overwrite by default."""

    try:
        dataset_val_root = report["dataset"]["val_root"]
        submission_path = report["submission"]["path"]
    except (KeyError, TypeError) as exc:
        raise SubmissionScoreError(
            "score report lacks dataset/submission provenance needed for safe output"
        ) from exc
    destination = validate_json_output_target(
        dataset_val_root, submission_path, path
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite existing JSON: {}".format(destination))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(str(temporary_path), str(destination))
        else:
            try:
                os.link(str(temporary_path), str(destination))
                temporary_path.unlink()
            except FileExistsError:
                raise FileExistsError(
                    "refusing to overwrite existing JSON: {}".format(destination)
                )
            except OSError:
                if destination.exists():
                    raise FileExistsError(
                        "refusing to overwrite existing JSON: {}".format(destination)
                    )
                os.rename(str(temporary_path), str(destination))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate and offline-score a canonical 24-video EV-UAV "
            "Challenge 2 validation submission."
        )
    )
    parser.add_argument(
        "--val-root",
        required=True,
        type=Path,
        help="official val/ directory or dataset root containing val/",
    )
    parser.add_argument(
        "--submission",
        required=True,
        type=Path,
        help="directory containing 24 TXT files or a flat ZIP",
    )
    parser.add_argument(
        "--allow-unofficial-dataset",
        action="store_true",
        help=(
            "DANGEROUS: score without dataset identity enforcement and mark the "
            "report UNOFFICIAL-UNLOCKED"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional atomic .json output; stdout is always emitted",
    )
    parser.add_argument(
        "--overwrite-json",
        action="store_true",
        help="explicitly allow replacing an existing --json-out file",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.overwrite_json and args.json_out is None:
        parser.error("--overwrite-json requires --json-out")
    try:
        if args.json_out is not None:
            # This preflight intentionally happens before model-free scoring so
            # even --overwrite-json can never target a scoring input.
            validate_json_output_target(
                args.val_root, args.submission, args.json_out
            )
        report = score_submission(
            args.val_root,
            args.submission,
            allow_unofficial_dataset=args.allow_unofficial_dataset,
        )
        if args.json_out is not None:
            write_json_atomic(args.json_out, report, overwrite=args.overwrite_json)
    except (FileExistsError, SubmissionScoreError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate EV-UAV Challenge 2 dataset and submission assets.

The validator intentionally depends only on NumPy and the Python standard
library.  Dataset archives are opened one at a time and submission text is
checked line by line so validation does not cache the whole dataset in RAM.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np


REQUIRED_NPZ_KEYS = frozenset(("ev", "evs_norm", "ev_loc"))
REQUIRED_RAW_FIELDS = ("x", "y", "t", "p")
TARGET_ID_FIELDS = ("name", "idx", "target_id", "id")
DEFAULT_WIDTH = 346
DEFAULT_HEIGHT = 260
CHECK_CHUNK_SIZE = 1_000_000


@dataclass
class IssueCollector:
    """Collect bounded issue details while retaining exact issue counts."""

    max_issues: int = 50
    errors: List[dict] = field(default_factory=list)
    warnings: List[dict] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    def error(self, code: str, path: Union[str, Path], message: str) -> None:
        self.error_count += 1
        if len(self.errors) < self.max_issues:
            self.errors.append(
                {"code": code, "path": str(path), "message": str(message)}
            )

    def warning(self, code: str, path: Union[str, Path], message: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < self.max_issues:
            self.warnings.append(
                {"code": code, "path": str(path), "message": str(message)}
            )


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    stem: str
    event_count: Optional[int]
    valid: bool


def _iter_chunks(values: np.ndarray) -> Iterable[np.ndarray]:
    values = np.asarray(values)
    for start in range(0, values.shape[0], CHECK_CHUNK_SIZE):
        yield values[start : start + CHECK_CHUNK_SIZE]


def _is_numeric(values: np.ndarray) -> bool:
    return bool(
        np.issubdtype(values.dtype, np.number)
        and not np.issubdtype(values.dtype, np.complexfloating)
    )


def _check_finite(
    values: np.ndarray,
    collector: IssueCollector,
    path: Path,
    field_name: str,
) -> bool:
    values = np.asarray(values)
    if not _is_numeric(values):
        collector.error(
            "non_numeric",
            path,
            "{} must contain real numeric values (found {}).".format(
                field_name, values.dtype
            ),
        )
        return False
    for chunk in _iter_chunks(values.reshape(-1)):
        if not np.isfinite(chunk).all():
            collector.error(
                "non_finite",
                path,
                "{} contains NaN or infinity.".format(field_name),
            )
            return False
    return True


def _check_bounds(
    values: np.ndarray,
    lower: float,
    upper_exclusive: float,
    collector: IssueCollector,
    path: Path,
    field_name: str,
) -> None:
    values = np.asarray(values).reshape(-1)
    if not _check_finite(values, collector, path, field_name):
        return
    for chunk in _iter_chunks(values):
        if np.any(chunk < lower) or np.any(chunk >= upper_exclusive):
            collector.error(
                "coordinate_bounds",
                path,
                "{} must be in [{}, {}).".format(
                    field_name, lower, upper_exclusive
                ),
            )
            return


def _check_binary(
    values: np.ndarray,
    collector: IssueCollector,
    path: Path,
    field_name: str,
    code: str,
) -> None:
    values = np.asarray(values).reshape(-1)
    if not _check_finite(values, collector, path, field_name):
        return
    for chunk in _iter_chunks(values):
        if not np.logical_or(chunk == 0, chunk == 1).all():
            collector.error(
                code,
                path,
                "{} must contain only 0 or 1.".format(field_name),
            )
            return


def _check_nonnegative_integers(
    values: np.ndarray,
    collector: IssueCollector,
    path: Path,
    field_name: str,
) -> None:
    values = np.asarray(values).reshape(-1)
    if not _check_finite(values, collector, path, field_name):
        return
    for chunk in _iter_chunks(values):
        if np.any(chunk < 0) or not np.equal(chunk, np.floor(chunk)).all():
            collector.error(
                "target_id_values",
                path,
                "{} must contain non-negative integers.".format(field_name),
            )
            return


def _array_event_count(
    array: np.ndarray,
    collector: IssueCollector,
    path: Path,
    key: str,
) -> Optional[int]:
    if array.ndim == 0:
        collector.error(
            "array_shape", path, "{} must have an event dimension.".format(key)
        )
        return None
    return int(array.shape[0])


def _validate_raw_events(
    raw_events: np.ndarray,
    collector: IssueCollector,
    path: Path,
    width: int,
    height: int,
) -> Optional[int]:
    event_count = _array_event_count(raw_events, collector, path, "ev")
    names = raw_events.dtype.names
    if raw_events.ndim != 1 or names is None:
        collector.error(
            "ev_schema",
            path,
            "ev must be a one-dimensional structured array.",
        )
        return event_count

    missing_fields = [name for name in REQUIRED_RAW_FIELDS if name not in names]
    if missing_fields:
        collector.error(
            "ev_fields",
            path,
            "ev is missing fields: {}.".format(", ".join(missing_fields)),
        )
        return event_count

    _check_bounds(raw_events["x"], 0, width, collector, path, "ev.x")
    _check_bounds(raw_events["y"], 0, height, collector, path, "ev.y")
    if _check_finite(raw_events["t"], collector, path, "ev.t"):
        for chunk in _iter_chunks(np.asarray(raw_events["t"]).reshape(-1)):
            if np.any(chunk < 0):
                collector.error(
                    "timestamp_values", path, "ev.t must be non-negative."
                )
                break
    _check_binary(
        raw_events["p"], collector, path, "ev.p", "polarity_values"
    )
    if "label" in names:
        _check_binary(
            raw_events["label"],
            collector,
            path,
            "ev.label",
            "label_values",
        )
    target_field = next((name for name in TARGET_ID_FIELDS if name in names), None)
    if target_field is not None:
        _check_nonnegative_integers(
            raw_events[target_field],
            collector,
            path,
            "ev.{}".format(target_field),
        )
    return event_count


def _validate_normalized_events(
    normalized: np.ndarray,
    collector: IssueCollector,
    path: Path,
) -> Optional[int]:
    event_count = _array_event_count(
        normalized, collector, path, "evs_norm"
    )
    if normalized.ndim != 2 or normalized.shape[1] < 6:
        collector.error(
            "evs_norm_shape",
            path,
            "evs_norm must have shape [N, 6+] (found {}).".format(
                tuple(normalized.shape)
            ),
        )
        return event_count
    if _check_finite(normalized, collector, path, "evs_norm"):
        _check_binary(
            normalized[:, 3],
            collector,
            path,
            "evs_norm[:, 3] (polarity)",
            "polarity_values",
        )
        _check_binary(
            normalized[:, 4],
            collector,
            path,
            "evs_norm[:, 4] (label)",
            "label_values",
        )
        _check_nonnegative_integers(
            normalized[:, 5],
            collector,
            path,
            "evs_norm[:, 5] (target id)",
        )
    return event_count


def _validate_locations(
    locations: np.ndarray,
    collector: IssueCollector,
    path: Path,
    width: int,
    height: int,
) -> Optional[int]:
    event_count = _array_event_count(locations, collector, path, "ev_loc")
    if locations.ndim != 2 or locations.shape[1] < 3:
        collector.error(
            "ev_loc_shape",
            path,
            "ev_loc must have shape [N, 3+] (found {}).".format(
                tuple(locations.shape)
            ),
        )
        return event_count
    _check_bounds(locations[:, 0], 0, width, collector, path, "ev_loc[:, 0]")
    _check_bounds(locations[:, 1], 0, height, collector, path, "ev_loc[:, 1]")
    timestamps = locations[:, 2]
    if _check_finite(timestamps, collector, path, "ev_loc[:, 2]"):
        for chunk in _iter_chunks(np.asarray(timestamps).reshape(-1)):
            if np.any(chunk < 0):
                collector.error(
                    "timestamp_values",
                    path,
                    "ev_loc[:, 2] must be non-negative.",
                )
                break
    return event_count


def validate_npz_file(
    path: Path,
    collector: IssueCollector,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> VideoMetadata:
    """Validate one archive while retaining at most one event array at once."""
    path = Path(path)
    errors_before = collector.error_count
    counts: Dict[str, Optional[int]] = {}
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            missing_keys = sorted(REQUIRED_NPZ_KEYS.difference(archive.files))
            if missing_keys:
                collector.error(
                    "npz_keys",
                    path,
                    "missing required keys: {}.".format(", ".join(missing_keys)),
                )

            if "ev" in archive.files:
                raw_events = archive["ev"]
                counts["ev"] = _validate_raw_events(
                    raw_events, collector, path, width, height
                )
                del raw_events

            if "evs_norm" in archive.files:
                normalized = archive["evs_norm"]
                counts["evs_norm"] = _validate_normalized_events(
                    normalized, collector, path
                )
                del normalized

            if "ev_loc" in archive.files:
                locations = archive["ev_loc"]
                counts["ev_loc"] = _validate_locations(
                    locations, collector, path, width, height
                )
                del locations
    except Exception as exc:
        collector.error("npz_open", path, "cannot read NPZ: {}".format(exc))

    known_counts = {key: value for key, value in counts.items() if value is not None}
    if len(set(known_counts.values())) > 1:
        collector.error(
            "event_count_mismatch",
            path,
            "event counts differ: {}.".format(
                ", ".join(
                    "{}={}".format(key, value)
                    for key, value in sorted(known_counts.items())
                )
            ),
        )
    event_count = known_counts.get("ev")
    if event_count is None and known_counts:
        event_count = next(iter(known_counts.values()))
    if event_count == 0:
        collector.error("empty_video", path, "video contains no events.")

    return VideoMetadata(
        path=path,
        stem=path.stem,
        event_count=event_count,
        valid=collector.error_count == errors_before,
    )


def _npz_files(split_dir: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".npz"
        ),
        key=lambda path: path.name.casefold(),
    )


def validate_split(
    dataset_root: Path,
    split_name: str,
    expected_count: int,
    collector: IssueCollector,
    width: int,
    height: int,
) -> Tuple[dict, List[VideoMetadata]]:
    split_dir = dataset_root / split_name
    if not split_dir.is_dir():
        collector.error(
            "split_missing", split_dir, "dataset split directory does not exist."
        )
        return (
            {
                "path": str(split_dir),
                "expected_files": expected_count,
                "files": 0,
                "valid_files": 0,
                "events": 0,
            },
            [],
        )

    files = _npz_files(split_dir)
    if len(files) != expected_count:
        collector.error(
            "split_count",
            split_dir,
            "expected {} NPZ files, found {}.".format(expected_count, len(files)),
        )

    folded_stems: Dict[str, Path] = {}
    for path in files:
        folded = path.stem.casefold()
        if folded in folded_stems:
            collector.error(
                "duplicate_stem",
                path,
                "duplicate case-insensitive stem also used by {}.".format(
                    folded_stems[folded].name
                ),
            )
        else:
            folded_stems[folded] = path

    metadata = [
        validate_npz_file(path, collector, width=width, height=height)
        for path in files
    ]
    return (
        {
            "path": str(split_dir),
            "expected_files": expected_count,
            "files": len(files),
            "valid_files": sum(video.valid for video in metadata),
            "events": sum(
                video.event_count or 0 for video in metadata
            ),
        },
        metadata,
    )


def _report_once(
    reported_codes: set,
    collector: IssueCollector,
    code: str,
    path: str,
    message: str,
) -> None:
    if code not in reported_codes:
        collector.error(code, path, message)
        reported_codes.add(code)


def _validate_submission_stream(
    stream: TextIO,
    display_path: str,
    source_path: Path,
    collector: IssueCollector,
    value_atol: float,
    value_rtol: float,
) -> int:
    reported_codes: set = set()
    row_count = 0
    try:
        with np.load(str(source_path), allow_pickle=False) as archive:
            if "ev" not in archive.files:
                collector.error(
                    "submission_source",
                    source_path,
                    "source NPZ has no ev array.",
                )
                return 0
            source_events = archive["ev"]
            names = source_events.dtype.names
            if source_events.ndim != 1 or names is None or any(
                field not in names for field in REQUIRED_RAW_FIELDS
            ):
                collector.error(
                    "submission_source",
                    source_path,
                    "source ev must expose x, y, t, p fields.",
                )
                return 0

            source_count = int(source_events.shape[0])
            source_columns = tuple(source_events[field] for field in REQUIRED_RAW_FIELDS)
            for line_number, raw_line in enumerate(stream, start=1):
                row_index = row_count
                row_count += 1
                fields = raw_line.strip().split()
                if len(fields) != 5:
                    _report_once(
                        reported_codes,
                        collector,
                        "submission_columns",
                        display_path,
                        "line {} must contain exactly five columns.".format(
                            line_number
                        ),
                    )
                    continue
                try:
                    values = [float(field) for field in fields]
                except ValueError:
                    _report_once(
                        reported_codes,
                        collector,
                        "submission_numeric",
                        display_path,
                        "line {} contains a non-numeric field.".format(line_number),
                    )
                    continue
                if not all(math.isfinite(value) for value in values):
                    _report_once(
                        reported_codes,
                        collector,
                        "submission_non_finite",
                        display_path,
                        "line {} contains NaN or infinity.".format(line_number),
                    )
                    continue
                if values[4] not in (0.0, 1.0):
                    _report_once(
                        reported_codes,
                        collector,
                        "submission_label",
                        display_path,
                        "line {} label must be 0 or 1.".format(line_number),
                    )
                if row_index < source_count:
                    source_values = (
                        float(source_columns[0][row_index]),
                        float(source_columns[1][row_index]),
                        float(source_columns[2][row_index]),
                        float(source_columns[3][row_index]),
                    )
                    if any(
                        not math.isclose(
                            actual,
                            expected,
                            rel_tol=value_rtol,
                            abs_tol=value_atol,
                        )
                        for actual, expected in zip(values[:4], source_values)
                    ):
                        _report_once(
                            reported_codes,
                            collector,
                            "submission_source_mismatch",
                            display_path,
                            "line {} first four fields differ from source ev.".format(
                                line_number
                            ),
                        )

            if row_count != source_count:
                collector.error(
                    "submission_row_count",
                    display_path,
                    "expected {} rows, found {}.".format(source_count, row_count),
                )
    except UnicodeError as exc:
        collector.error(
            "submission_encoding",
            display_path,
            "text is not valid UTF-8: {}".format(exc),
        )
    except Exception as exc:
        collector.error(
            "submission_read",
            display_path,
            "cannot validate submission text: {}".format(exc),
        )
    return row_count


def _expected_submission_files(
    val_videos: Sequence[VideoMetadata], collector: IssueCollector
) -> Dict[str, VideoMetadata]:
    expected: Dict[str, VideoMetadata] = {}
    for video in val_videos:
        name = video.stem + ".txt"
        if name in expected:
            collector.error(
                "submission_name_collision",
                video.path,
                "multiple validation files map to {}.".format(name),
            )
        else:
            expected[name] = video
    return expected


def validate_submission(
    submission_path: Path,
    val_videos: Sequence[VideoMetadata],
    collector: IssueCollector,
    value_atol: float,
    value_rtol: float,
) -> dict:
    submission_path = Path(submission_path)
    expected = _expected_submission_files(val_videos, collector)
    validated_files = 0
    total_rows = 0

    if submission_path.is_dir():
        kind = "directory"
        entries = {
            path.name: path
            for path in submission_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        actual_names = set(entries)
        expected_names = set(expected)
        for name in sorted(expected_names - actual_names):
            collector.error(
                "submission_missing", submission_path, "missing {}.".format(name)
            )
        for name in sorted(actual_names - expected_names):
            collector.error(
                "submission_extra", entries[name], "unexpected submission file."
            )
        for name in sorted(actual_names & expected_names):
            try:
                with entries[name].open(
                    "r", encoding="utf-8-sig", newline=None
                ) as stream:
                    total_rows += _validate_submission_stream(
                        stream,
                        str(entries[name]),
                        expected[name].path,
                        collector,
                        value_atol,
                        value_rtol,
                    )
            except OSError as exc:
                collector.error(
                    "submission_read",
                    entries[name],
                    "cannot open submission text: {}".format(exc),
                )
            validated_files += 1
        file_count = len(entries)
    elif submission_path.is_file() and submission_path.suffix.lower() == ".zip":
        kind = "zip"
        file_count = 0
        try:
            with zipfile.ZipFile(str(submission_path), "r") as archive:
                infos = archive.infolist()
                flat_infos: Dict[str, zipfile.ZipInfo] = {}
                for info in infos:
                    if info.is_dir() or "/" in info.filename or "\\" in info.filename:
                        collector.error(
                            "zip_not_flat",
                            "{}!{}".format(submission_path, info.filename),
                            "ZIP entries must be files at the archive root.",
                        )
                        continue
                    if not info.filename.lower().endswith(".txt"):
                        collector.error(
                            "zip_non_txt",
                            "{}!{}".format(submission_path, info.filename),
                            "ZIP may contain only TXT files.",
                        )
                        continue
                    file_count += 1
                    if info.filename in flat_infos:
                        collector.error(
                            "zip_duplicate",
                            "{}!{}".format(submission_path, info.filename),
                            "duplicate ZIP member name.",
                        )
                    else:
                        flat_infos[info.filename] = info

                actual_names = set(flat_infos)
                expected_names = set(expected)
                for name in sorted(expected_names - actual_names):
                    collector.error(
                        "submission_missing",
                        submission_path,
                        "missing {}.".format(name),
                    )
                for name in sorted(actual_names - expected_names):
                    collector.error(
                        "submission_extra",
                        "{}!{}".format(submission_path, name),
                        "unexpected submission file.",
                    )
                for name in sorted(actual_names & expected_names):
                    with archive.open(flat_infos[name], "r") as binary_stream:
                        with io.TextIOWrapper(
                            binary_stream, encoding="utf-8-sig", newline=None
                        ) as stream:
                            total_rows += _validate_submission_stream(
                                stream,
                                "{}!{}".format(submission_path, name),
                                expected[name].path,
                                collector,
                                value_atol,
                                value_rtol,
                            )
                    validated_files += 1
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            collector.error(
                "zip_open", submission_path, "cannot read ZIP: {}".format(exc)
            )
    else:
        kind = "unknown"
        file_count = 0
        collector.error(
            "submission_path",
            submission_path,
            "submission must be a directory or a .zip file.",
        )

    return {
        "path": str(submission_path),
        "kind": kind,
        "expected_files": len(expected),
        "files": file_count,
        "validated_files": validated_files,
        "rows": total_rows,
    }


def validate_assets(
    dataset_root: Union[str, Path],
    submission: Optional[Union[str, Path]] = None,
    expected_train: int = 99,
    expected_val: int = 24,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    value_atol: float = 1e-6,
    value_rtol: float = 1e-9,
    max_issues: int = 50,
) -> dict:
    """Validate a dataset root and, optionally, a submission directory/ZIP."""
    if expected_train < 0 or expected_val < 0:
        raise ValueError("expected split counts must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if value_atol < 0 or value_rtol < 0:
        raise ValueError("comparison tolerances must be non-negative")
    if max_issues <= 0:
        raise ValueError("max_issues must be positive")

    dataset_root = Path(dataset_root).expanduser().resolve()
    collector = IssueCollector(max_issues=max_issues)
    if not dataset_root.is_dir():
        collector.error(
            "dataset_root", dataset_root, "dataset root directory does not exist."
        )

    split_results = {}
    split_videos = {}
    for split_name, expected_count in (
        ("train", expected_train),
        ("val", expected_val),
    ):
        split_result, videos = validate_split(
            dataset_root,
            split_name,
            expected_count,
            collector,
            width,
            height,
        )
        split_results[split_name] = split_result
        split_videos[split_name] = videos

    submission_result = None
    if submission is not None:
        submission_result = validate_submission(
            Path(submission).expanduser().resolve(),
            split_videos["val"],
            collector,
            value_atol,
            value_rtol,
        )

    return {
        "ok": collector.error_count == 0,
        "dataset_root": str(dataset_root),
        "dimensions": {"width": width, "height": height},
        "splits": split_results,
        "submission": submission_result,
        "error_count": collector.error_count,
        "warning_count": collector.warning_count,
        "errors": collector.errors,
        "warnings": collector.warnings,
        "issues_truncated": (
            collector.error_count > len(collector.errors)
            or collector.warning_count > len(collector.warnings)
        ),
    }


def format_human(result: dict) -> str:
    status = "PASS" if result["ok"] else "FAIL"
    lines = [
        "{} EV-UAV Challenge 2 asset validation (errors={}, warnings={})".format(
            status, result["error_count"], result["warning_count"]
        ),
        "dataset: {}".format(result["dataset_root"]),
    ]
    for split_name in ("train", "val"):
        split = result["splits"][split_name]
        lines.append(
            "{}: files={}/{} valid={} events={}".format(
                split_name,
                split["files"],
                split["expected_files"],
                split["valid_files"],
                split["events"],
            )
        )
    submission = result.get("submission")
    if submission is not None:
        lines.append(
            "submission: kind={} files={}/{} validated={} rows={}".format(
                submission["kind"],
                submission["files"],
                submission["expected_files"],
                submission["validated_files"],
                submission["rows"],
            )
        )
    for issue in result["errors"]:
        lines.append(
            "ERROR [{}] {}: {}".format(
                issue["code"], issue["path"], issue["message"]
            )
        )
    for issue in result["warnings"]:
        lines.append(
            "WARN  [{}] {}: {}".format(
                issue["code"], issue["path"], issue["message"]
            )
        )
    omitted = (
        result["error_count"]
        + result["warning_count"]
        - len(result["errors"])
        - len(result["warnings"])
    )
    if omitted > 0:
        lines.append("... {} additional issue(s) omitted".format(omitted))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate EV-UAV Challenge 2 train/val NPZ assets and an optional "
            "submission directory or flat ZIP."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="directory containing train/ and val/")
    parser.add_argument(
        "--submission",
        type=Path,
        help="optional prediction directory or ZIP to validate",
    )
    parser.add_argument("--expected-train", type=int, default=99)
    parser.add_argument("--expected-val", type=int, default=24)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--value-atol",
        type=float,
        default=1e-6,
        help="absolute tolerance for submission x/y/t/p comparison",
    )
    parser.add_argument(
        "--value-rtol",
        type=float,
        default=1e-9,
        help="relative tolerance for submission x/y/t/p comparison",
    )
    parser.add_argument(
        "--max-issues", type=int, default=50, help="maximum issue details to print"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit one JSON object instead of human text"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = validate_assets(
            dataset_root=args.dataset_root,
            submission=args.submission,
            expected_train=args.expected_train,
            expected_val=args.expected_val,
            width=args.width,
            height=args.height,
            value_atol=args.value_atol,
            value_rtol=args.value_rtol,
            max_issues=args.max_issues,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        # ASCII escapes avoid Windows console/``conda run`` encoding failures
        # when the dataset path contains Chinese characters.
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    else:
        print(format_human(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

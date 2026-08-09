"""Evaluate temporal-memory checkpoints with the frozen Challenge 2 baseline.

This module is deliberately independent from the training code.  It launches
``test2.py`` in a fresh Python process for every distinct checkpoint hash,
captures the complete output, and stores enough provenance to reproduce or
audit each score.  It never imports PyTorch and therefore remains cheap to use
as an experiment orchestrator.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_SCRIPT = PROJECT_ROOT / "test2.py"
MANIFEST_NAME = "checkpoint_evaluations.json"
CSV_NAME = "checkpoint_evaluations.csv"
SCHEMA_VERSION = 2
EVALUATOR_VERSION = "2.0.0"
FROZEN_SETTINGS_VERSION = "m20-golden-2026-08-10-v1"
EXPECTED_VAL_FILENAMES = tuple("val_{:03d}.npz".format(index) for index in range(24))

METRIC_NAMES = ("IoU", "Acc", "Pd", "Fa", "Score_Fa", "Score")
METRIC_KEYS = {
    "IoU": "iou",
    "Acc": "acc",
    "Pd": "pd",
    "Fa": "fa",
    "Score_Fa": "score_fa",
    "Score": "score",
}
_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_METRIC_PATTERN = re.compile(
    r"^\s*(IoU|Acc|Pd|Fa|Score_Fa|Score):\s*(" + _FLOAT_PATTERN + r")\s*$",
    re.MULTILINE,
)

# These scalar settings are the published M20/M10 routing and post-processing
# baseline.  Paths are inserted separately by ``build_test2_command``.
FROZEN_M20_SETTINGS = (
    "TEST.eval=true",
    "TEST.roc=true",
    "TEST.prediction_threshold=0.719",
    "TEMPORAL_FRAME.temporal_frame_enabled=false",
    "TEMPORAL_MEMORY.temporal_memory_enabled=true",
    "TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000",
    "TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true",
    "TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0",
    "TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8",
    "POSTPROCESS.p0_enabled=true",
    "POSTPROCESS.p0_spatial_radius=2",
    "POSTPROCESS.p0_temporal_bin_size=50",
    "POSTPROCESS.p0_temporal_radius_bins=1",
    "POSTPROCESS.p0_min_cluster_events=3",
    "POSTPROCESS.p0_min_duration_bins=5",
    "POSTPROCESS.p0c_high_confidence_recovery_enabled=true",
    "POSTPROCESS.p0c_retain_min_score=0.95",
    "POSTPROCESS.p0b_enabled=false",
    "POSTPROCESS.p18_score_track_recovery_enabled=true",
    "POSTPROCESS.p18_event_count_cutoff=1",
    "POSTPROCESS.p18_max_event_count=35000",
    "POSTPROCESS.p18_candidate_floor=0.53",
    "POSTPROCESS.p18_spatial_radius=5",
    "POSTPROCESS.p18_temporal_bin_size=50",
    "POSTPROCESS.p18_max_link_distance=8.0",
    "POSTPROCESS.p18_max_gap_bins=1",
    "POSTPROCESS.p18_min_track_bins=4",
    "POSTPROCESS.p18_restore_mode=best",
    "POSTPROCESS.p6_density_threshold_enabled=true",
    "POSTPROCESS.p6_event_count_cutoff=30000",
    "POSTPROCESS.p6_low_density_threshold=0.718",
    "POSTPROCESS.p6_high_density_threshold=0.719",
)

DYNAMIC_M20_SETTING_KEYS = (
    "DATA.root",
    "TEMPORAL_MEMORY.temporal_memory_model_path",
    "TEMPORAL_MEMORY.temporal_memory_secondary_model_path",
)


class EvaluationError(RuntimeError):
    """Raised after a failed evaluation has been persisted to the manifest."""


class MetricParseError(ValueError):
    """Raised when ``test2.py`` does not emit all six official metrics."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def parse_metrics(output: str) -> Dict[str, float]:
    """Parse the six final metrics printed by ``test2.py``.

    The last occurrence wins so that an incidental earlier summary cannot
    shadow the final Challenge 2 block.  Missing or non-finite values are a
    hard error; an exit code of zero alone is not considered a valid run.
    """

    parsed: Dict[str, float] = {}
    for name, raw_value in _METRIC_PATTERN.findall(output):
        parsed[METRIC_KEYS[name]] = float(raw_value)

    missing = [METRIC_KEYS[name] for name in METRIC_NAMES if METRIC_KEYS[name] not in parsed]
    if missing:
        raise MetricParseError(
            "test2.py output is missing official metrics: {}".format(", ".join(missing))
        )
    non_finite = [key for key, value in parsed.items() if not math.isfinite(value)]
    if non_finite:
        raise MetricParseError(
            "test2.py emitted non-finite metrics: {}".format(", ".join(non_finite))
        )
    return parsed


def _yaml_string(value: Path) -> str:
    """Return a YAML-safe scalar embedded in a ``SECTION.KEY=value`` arg."""

    return json.dumps(value.resolve().as_posix(), ensure_ascii=False)


def build_test2_command(
    checkpoint: Path,
    m10_checkpoint: Path,
    data_root: Path,
    config: Path,
    *,
    python_executable: str | None = None,
    test_script: Path | None = None,
) -> List[str]:
    """Build the exact list-form subprocess command for one M20 candidate."""

    executable = python_executable or sys.executable
    script = (test_script or TEST_SCRIPT).resolve()
    path_settings = (
        "DATA.root={}".format(_yaml_string(data_root)),
        "TEMPORAL_MEMORY.temporal_memory_model_path={}".format(
            _yaml_string(checkpoint)
        ),
        "TEMPORAL_MEMORY.temporal_memory_secondary_model_path={}".format(
            _yaml_string(m10_checkpoint)
        ),
    )
    return [
        executable,
        str(script),
        "--config",
        str(config.resolve()),
        "--set",
        *path_settings,
        *FROZEN_M20_SETTINGS,
    ]


def expand_checkpoint_specs(specs: Sequence[str]) -> List[Path]:
    """Expand an explicit list and/or Windows-compatible glob patterns."""

    resolved: List[Path] = []
    seen: Set[str] = set()
    for original_spec in specs:
        spec = os.path.expandvars(os.path.expanduser(original_spec))
        if glob.has_magic(spec):
            matches = sorted(glob.glob(spec, recursive=True))
            if not matches:
                raise FileNotFoundError(
                    "Checkpoint pattern matched no files: {}".format(original_spec)
                )
        else:
            matches = [spec]

        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                raise FileNotFoundError("Checkpoint not found: {}".format(path))
            canonical = os.path.normcase(str(path))
            if canonical not in seen:
                seen.add(canonical)
                resolved.append(path)

    if not resolved:
        raise ValueError("At least one checkpoint is required.")
    return resolved


def new_manifest() -> MutableMapping[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "runs": [],
    }


def load_manifest(path: Path) -> MutableMapping[str, object]:
    if not path.exists():
        return new_manifest()
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
        raise ValueError("Invalid evaluation manifest: {}".format(path))
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        # Schema 1 skipped solely by candidate hash.  Keep its audit records,
        # but upgrade without assigning identities so none can be reused
        # unsafely under a different M10/config/data/tool context.
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["migrated_from_schema_version"] = 1
    elif schema_version != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported manifest schema {} in {}".format(
                schema_version, path
            )
        )
    return manifest


def successful_evaluation_identities(manifest: Mapping[str, object]) -> Set[str]:
    successful: Set[str] = set()
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Manifest runs must be a list.")
    for run in runs:
        if not isinstance(run, dict):
            continue
        digest = run.get("evaluation_identity_sha256")
        metrics = run.get("metrics")
        if (
            run.get("status") == "success"
            and run.get("exit_code") == 0
            and isinstance(digest, str)
            and digest
            and isinstance(metrics, dict)
            and all(key in metrics for key in METRIC_KEYS.values())
        ):
            successful.add(digest.lower())
    return successful


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


CSV_FIELDS = (
    "status",
    "evaluation_identity_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "m10_checkpoint_path",
    "m10_checkpoint_sha256",
    "data_identity_sha256",
    "evaluator_version",
    "evaluator_sha256",
    "test2_sha256",
    "frozen_settings_version",
    "frozen_settings_sha256",
    "parent_git_sha",
    "parent_git_branch",
    "parent_git_dirty",
    "parent_config_path",
    "parent_config_sha256",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_seconds",
    "exit_code",
    "iou",
    "acc",
    "pd",
    "fa",
    "score_fa",
    "score",
    "log_path",
    "error",
)


def _flatten_csv_record(record: Mapping[str, object]) -> Dict[str, object]:
    row = {field: record.get(field, "") for field in CSV_FIELDS}
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key in METRIC_KEYS.values():
            row[key] = metrics.get(key, "")
    return row


def write_csv(path: Path, runs: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow(_flatten_csv_record(run))
    os.replace(temporary, path)


def persist_manifest(
    manifest: MutableMapping[str, object], manifest_path: Path, csv_path: Path
) -> None:
    manifest["updated_at_utc"] = utc_now()
    _write_json_atomic(manifest_path, manifest)
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Manifest runs must be a list.")
    write_csv(csv_path, runs)


def git_provenance(project_root: Path) -> Dict[str, object]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "git {} failed (exit {}): {}".format(
                    " ".join(arguments), completed.returncode, completed.stderr.strip()
                )
            )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1")
    tracked_diff = git("diff", "--binary", "HEAD", "--", ".")
    source_state_sha256 = hashlib.sha256(
        (status + "\n" + tracked_diff).encode("utf-8")
    ).hexdigest()
    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current") or "(detached)",
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "source_state_sha256": source_state_sha256,
    }


def validation_data_identity(data_root: Path) -> Dict[str, object]:
    """Validate and fingerprint the exact official 24-file validation set."""

    val_dir = data_root / "val"
    if not val_dir.is_dir():
        raise NotADirectoryError("Validation directory not found: {}".format(val_dir))

    files = sorted(path for path in val_dir.glob("*.npz") if path.is_file())
    names = [path.name for path in files]
    expected = list(EXPECTED_VAL_FILENAMES)
    if names != expected:
        missing = [name for name in expected if name not in names]
        unexpected = [name for name in names if name not in expected]
        details = []
        if missing:
            details.append("missing={}".format(",".join(missing)))
        if unexpected:
            details.append("unexpected={}".format(",".join(unexpected)))
        raise ValueError(
            "Validation set must contain exactly the 24 expected val_000.npz through "
            "val_023.npz files; found {} ({})".format(
                len(files), "; ".join(details) or "filename/order mismatch"
            )
        )

    entries = []
    for path in files:
        size = path.stat().st_size
        if size <= 0:
            raise ValueError("Validation file is empty: {}".format(path))
        entries.append({"name": path.name, "size_bytes": size})

    identity = {
        "method": "ordered-val-filename-size-v1",
        "file_count": len(entries),
        "files": entries,
    }
    identity["sha256"] = sha256_json(identity)
    official_manifest = data_root / "official_google_drive_manifest.json"
    identity["official_manifest_sha256"] = (
        sha256_file(official_manifest) if official_manifest.is_file() else None
    )
    return identity


def evaluation_tool_identity() -> Dict[str, str]:
    frozen_payload = {
        "version": FROZEN_SETTINGS_VERSION,
        "dynamic_setting_keys": list(DYNAMIC_M20_SETTING_KEYS),
        "fixed_settings": list(FROZEN_M20_SETTINGS),
    }
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "test2_sha256": sha256_file(TEST_SCRIPT),
        "frozen_settings_version": FROZEN_SETTINGS_VERSION,
        "frozen_settings_sha256": sha256_json(frozen_payload),
    }


def build_evaluation_identity(
    checkpoint_sha256: str,
    m10_sha256: str,
    config_sha256: str,
    data_identity_sha256: str,
    tool_identity: Mapping[str, str],
    parent_git_sha: str,
    source_state_sha256: str,
) -> Dict[str, object]:
    """Build the complete context identity used for safe resume decisions."""

    components = {
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "m10_checkpoint_sha256": m10_sha256.lower(),
        "config_sha256": config_sha256.lower(),
        "data_identity_sha256": data_identity_sha256.lower(),
        "evaluator_version": tool_identity["evaluator_version"],
        "evaluator_sha256": tool_identity["evaluator_sha256"].lower(),
        "test2_sha256": tool_identity["test2_sha256"].lower(),
        "frozen_settings_version": tool_identity["frozen_settings_version"],
        "frozen_settings_sha256": tool_identity["frozen_settings_sha256"].lower(),
        "parent_git_sha": parent_git_sha.lower(),
        "source_state_sha256": source_state_sha256.lower(),
    }
    return {"sha256": sha256_json(components), "components": components}


def snapshot_config(config: Path, output_dir: Path) -> Dict[str, str]:
    digest = sha256_file(config)
    snapshot_dir = output_dir / "config_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / "{}-{}{}".format(config.stem, digest[:12], config.suffix)
    if not snapshot.exists():
        shutil.copyfile(config, snapshot)
    return {
        "path": str(config),
        "sha256": digest,
        "snapshot_path": str(snapshot),
    }


def _write_run_log(
    path: Path,
    command: Sequence[str],
    stdout: str,
    stderr: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("COMMAND (JSON list)\n")
        stream.write(json.dumps(list(command), ensure_ascii=False, indent=2))
        stream.write("\n\nSTDOUT\n")
        stream.write(stdout)
        if stdout and not stdout.endswith("\n"):
            stream.write("\n")
        stream.write("\nSTDERR\n")
        stream.write(stderr)
        if stderr and not stderr.endswith("\n"):
            stream.write("\n")


def evaluate_checkpoint(
    checkpoint: Path,
    checkpoint_sha256: str,
    evaluation_identity: Mapping[str, object],
    m10_checkpoint: Path,
    m10_sha256: str,
    data_root: Path,
    data_identity: Mapping[str, object],
    config: Path,
    config_provenance: Mapping[str, str],
    tool_identity: Mapping[str, str],
    git_info: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    command = build_test2_command(checkpoint, m10_checkpoint, data_root, config)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint.stem).strip("._")
    safe_stem = safe_stem or "checkpoint"
    started_at = utc_now()
    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    log_path = log_dir / "{}-{}-{}.log".format(
        safe_stem, checkpoint_sha256[:12], run_token
    )
    start = time.perf_counter()
    stdout = ""
    stderr = ""
    exit_code = None
    invocation_error = None
    try:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except OSError as exc:
        invocation_error = "{}: {}".format(type(exc).__name__, exc)
        stderr = invocation_error

    elapsed = time.perf_counter() - start
    finished_at = utc_now()
    _write_run_log(log_path, command, stdout, stderr)

    metrics = None
    parse_error = None
    if exit_code == 0:
        try:
            metrics = parse_metrics(stdout + "\n" + stderr)
        except MetricParseError as exc:
            parse_error = str(exc)

    error = invocation_error or parse_error
    if exit_code not in (None, 0):
        error = "test2.py exited with code {}".format(exit_code)
    status = "success" if exit_code == 0 and metrics is not None else "failed"

    return {
        "status": status,
        "evaluation_identity_sha256": evaluation_identity["sha256"],
        "evaluation_identity_components": evaluation_identity["components"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "m10_checkpoint_path": str(m10_checkpoint),
        "m10_checkpoint_sha256": m10_sha256,
        "data_root": str(data_root),
        "data_identity_sha256": data_identity["sha256"],
        "data_identity": data_identity,
        "evaluator_version": tool_identity["evaluator_version"],
        "evaluator_sha256": tool_identity["evaluator_sha256"],
        "test2_sha256": tool_identity["test2_sha256"],
        "frozen_settings_version": tool_identity["frozen_settings_version"],
        "frozen_settings_sha256": tool_identity["frozen_settings_sha256"],
        "parent_git_sha": git_info["sha"],
        "parent_git_branch": git_info["branch"],
        "parent_git_dirty": git_info["dirty"],
        "parent_git_status_porcelain": git_info["status_porcelain"],
        "source_state_sha256": git_info["source_state_sha256"],
        "parent_config_path": config_provenance["path"],
        "parent_config_sha256": config_provenance["sha256"],
        "parent_config_snapshot_path": config_provenance["snapshot_path"],
        "python_executable": sys.executable,
        "command": command,
        "frozen_m20_settings": list(FROZEN_M20_SETTINGS),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": round(elapsed, 6),
        "exit_code": exit_code,
        "metrics": metrics,
        "log_path": str(log_path),
        "error": error,
    }


def validate_inputs(m10_checkpoint: Path, data_root: Path, config: Path) -> None:
    if not TEST_SCRIPT.is_file():
        raise FileNotFoundError("Validation entrypoint not found: {}".format(TEST_SCRIPT))
    if not m10_checkpoint.is_file():
        raise FileNotFoundError("M10 checkpoint not found: {}".format(m10_checkpoint))
    if not config.is_file():
        raise FileNotFoundError("Config not found: {}".format(config))
    if not data_root.is_dir():
        raise NotADirectoryError("Data root not found: {}".format(data_root))
    # Exact validation filenames are checked by ``validation_data_identity``.


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    m10_checkpoint = Path(args.m10_checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    config = Path(args.config).resolve()
    validate_inputs(m10_checkpoint, data_root, config)
    checkpoints = expand_checkpoint_specs(args.checkpoints)
    data_info = validation_data_identity(data_root)

    manifest_path = output_dir / MANIFEST_NAME
    csv_path = output_dir / CSV_NAME
    manifest = load_manifest(manifest_path)
    successful = successful_evaluation_identities(manifest)
    git_info = git_provenance(PROJECT_ROOT)
    config_info = snapshot_config(config, output_dir)
    m10_sha256 = sha256_file(m10_checkpoint)
    tool_info = evaluation_tool_identity()

    runs = manifest["runs"]
    assert isinstance(runs, list)
    evaluated = 0
    skipped = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        digest = sha256_file(checkpoint)
        evaluation_identity = build_evaluation_identity(
            digest,
            m10_sha256,
            config_info["sha256"],
            data_info["sha256"],
            tool_info,
            str(git_info["sha"]),
            str(git_info["source_state_sha256"]),
        )
        identity_digest = str(evaluation_identity["sha256"])
        if identity_digest.lower() in successful:
            skipped += 1
            print(
                "[{}/{}] skip successful evaluation identity {} "
                "(checkpoint {}): {}".format(
                    index,
                    len(checkpoints),
                    identity_digest[:12],
                    digest[:12],
                    checkpoint,
                ),
                flush=True,
            )
            continue

        print(
            "[{}/{}] evaluating {} ({})".format(
                index, len(checkpoints), checkpoint, digest[:12]
            ),
            flush=True,
        )
        record = evaluate_checkpoint(
            checkpoint,
            digest,
            evaluation_identity,
            m10_checkpoint,
            m10_sha256,
            data_root,
            data_info,
            config,
            config_info,
            tool_info,
            git_info,
            output_dir,
        )
        runs.append(record)
        persist_manifest(manifest, manifest_path, csv_path)
        evaluated += 1

        if record["status"] != "success":
            raise EvaluationError(
                "Evaluation failed for {}: {}. Full log: {}".format(
                    checkpoint, record.get("error") or "unknown error", record["log_path"]
                )
            )
        successful.add(identity_digest.lower())
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        print(
            "  Score={:.10f} Pd={:.10f} Fa={:.10e} IoU={:.10f}".format(
                metrics["score"], metrics["pd"], metrics["fa"], metrics["iou"]
            ),
            flush=True,
        )

    # Create both files on an all-skipped first invocation as well.
    persist_manifest(manifest, manifest_path, csv_path)
    print(
        "Evaluation complete: evaluated={}, skipped={}, manifest={}".format(
            evaluated, skipped, manifest_path
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more M20-style checkpoints with the frozen M20/M10 "
            "Challenge 2 validation settings."
        )
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="PATH_OR_GLOB",
        help="checkpoint paths and/or glob patterns (expanded by this script on Windows)",
    )
    parser.add_argument(
        "--m10-checkpoint", required=True, help="frozen low-density M10 checkpoint"
    )
    parser.add_argument("--data-root", required=True, help="Challenge 2 dataset root")
    parser.add_argument("--config", required=True, help="base YAML config for test2.py")
    parser.add_argument(
        "--output-dir", required=True, help="directory for logs, JSON, CSV, and config snapshots"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (EvaluationError, FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

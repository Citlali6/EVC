"""CPU-only, label-free audit of the frozen temporal input route on train.

The command accepts only the canonical ``train`` directory and requires the
complete ``train_000.npz`` ... ``train_098.npz`` population.  It reads event
locations only for a row-count integrity check and consumes only normalized
polarity (``evs_norm[:, 3]``) for the H1/H2 decision.  Label and target-id
columns are never indexed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from utils.temporal_memory_input_router import (
    EXPECTED_TEMPORAL_BIN_COUNT,
    HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE,
    LOW_DENSITY_MAX_EVENT_COUNT,
    POLARITY_MINORITY_CUTOFF,
    route_policy_definition,
    route_policy_sha256,
    select_temporal_memory_input_route,
)


SCHEMA = "ev-uav-temporal-input-route-train-audit-v1"
TRAIN_FILE_PATTERN = re.compile(r"^train_[0-9]{3}\.npz$")
OFFICIAL_TRAIN_NAMES = tuple("train_{:03d}.npz".format(i) for i in range(99))
EXPECTED_HIGH_NAMES = tuple(
    ["train_{:03d}.npz".format(i) for i in range(44, 48)]
    + ["train_{:03d}.npz".format(i) for i in range(88, 99)]
)
FORBIDDEN_SPLIT_TOKENS = {"val", "validation", "test"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_tokens(path):
    tokens = set()
    for part in Path(path).parts:
        tokens.update(token for token in re.split(r"[^a-z0-9]+", part.lower()) if token)
    return tokens


def discover_official_train_sources(train_root):
    """Resolve and validate the exact official train source population."""

    train_root = Path(train_root).resolve()
    if train_root.name.lower() != "train":
        raise ValueError("--train-root must be the canonical directory named train.")
    ancestor_tokens = _path_tokens(train_root.parent)
    if ancestor_tokens & FORBIDDEN_SPLIT_TOKENS:
        raise ValueError("Train root is nested below a forbidden split path.")
    if not train_root.is_dir():
        raise NotADirectoryError(train_root)
    paths = sorted(train_root.glob("*.npz"))
    names = tuple(path.name for path in paths)
    invalid = [name for name in names if not TRAIN_FILE_PATTERN.fullmatch(name)]
    if invalid:
        raise ValueError(
            "Train-only guard rejected non-canonical sources: {}".format(
                ", ".join(invalid[:5])
            )
        )
    if names != OFFICIAL_TRAIN_NAMES:
        missing = sorted(set(OFFICIAL_TRAIN_NAMES).difference(names))
        extra = sorted(set(names).difference(OFFICIAL_TRAIN_NAMES))
        raise ValueError(
            "Official train population mismatch (missing={}, extra={}).".format(
                missing[:5],
                extra[:5],
            )
        )
    return train_root, tuple(paths)


def read_input_statistics(path):
    """Read only input columns needed by the label-free route audit."""

    path = Path(path).resolve()
    if not TRAIN_FILE_PATTERN.fullmatch(path.name):
        raise ValueError("Refusing non-train source: {}".format(path))
    with np.load(path, allow_pickle=False) as payload:
        if "evs_norm" not in payload.files or "ev_loc" not in payload.files:
            raise ValueError("Train source lacks evs_norm/ev_loc: {}".format(path))
        event_input = np.asarray(payload["evs_norm"])
        locations = np.asarray(payload["ev_loc"])
        if event_input.ndim != 2 or event_input.shape[1] < 4:
            raise ValueError("evs_norm must have at least four input columns.")
        if locations.ndim != 2 or locations.shape[1] < 3:
            raise ValueError("ev_loc must have at least three columns.")
        if event_input.shape[0] != locations.shape[0]:
            raise ValueError("evs_norm/ev_loc event counts disagree.")
        # Deliberately do not index evs_norm[:, 4] or evs_norm[:, 5].
        polarities = np.asarray(event_input[:, 3], dtype=np.float64).copy()
    decision = select_temporal_memory_input_route(
        polarities,
        EXPECTED_TEMPORAL_BIN_COUNT,
    )
    return {
        "source_name": path.name,
        "source_path": str(path),
        "source_size_bytes": path.stat().st_size,
        "source_sha256": sha256_file(path),
        **decision.to_metadata(),
        "would_use_t32_without_density_gate": bool(
            decision.event_count <= HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
            and decision.polarity_minority_fraction >= POLARITY_MINORITY_CUTOFF
        ),
    }


def _atomic_json(path, payload):
    path = Path(path).resolve()
    if path.exists() or Path(str(path) + ".sha256").exists():
        raise FileExistsError("Refusing to overwrite audit output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text("{}  {}\n".format(digest, path.name), encoding="ascii")
    return digest, sidecar


def run(args):
    train_root, paths = discover_official_train_sources(args.train_root)
    records = []
    for index, path in enumerate(paths, start=1):
        record = read_input_statistics(path)
        records.append(record)
        print(
            "[{}/{}] {} n={} fraction={:.6f} route={}/{}".format(
                index,
                len(paths),
                path.name,
                record["event_count"],
                record["polarity_minority_fraction"],
                record["checkpoint_role"],
                record["mode"],
            ),
            flush=True,
        )

    high_names = tuple(
        record["source_name"]
        for record in records
        if record["event_count"] > HIGH_DENSITY_MIN_EVENT_COUNT_EXCLUSIVE
    )
    if high_names != EXPECTED_HIGH_NAMES:
        raise RuntimeError(
            "The frozen >200k gate no longer selects the 15 audited H1/H2 sources."
        )
    gap_names = [
        record["source_name"]
        for record in records
        if 200_000 < record["event_count"] <= 250_000
    ]
    if gap_names:
        raise RuntimeError(
            "Unexpected train sources appeared in the 200k--250k evidence gap: {}".format(
                gap_names
            )
        )
    route_counts = Counter(
        "{}/{}".format(record["checkpoint_role"], record["mode"])
        for record in records
    )
    protected = [
        record
        for record in records
        if record["would_use_t32_without_density_gate"]
    ]
    project_root = Path(__file__).resolve().parent
    code_paths = {
        "audit_temporal_memory_input_route_train.py": Path(__file__).resolve(),
        "utils/temporal_memory_input_router.py": project_root
        / "utils"
        / "temporal_memory_input_router.py",
        "utils/temporal_memory_windowed_inference.py": project_root
        / "utils"
        / "temporal_memory_windowed_inference.py",
        "crossfit_persistent_pixel_prior.py": project_root
        / "crossfit_persistent_pixel_prior.py",
    }
    missing_code = [name for name, path in code_paths.items() if not path.is_file()]
    if missing_code:
        raise FileNotFoundError("Required candidate code is missing: {}".format(missing_code))
    payload = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "evidence_class": "complete_official_train_input_only_route_audit",
        "split_access": {
            "dataset_split": "train",
            "validation_or_test_read": False,
            "consumed": ["ev_loc row count", "evs_norm[:,3] polarity"],
            "not_consumed": ["evs_norm[:,4] labels", "evs_norm[:,5] target ids"],
        },
        "route_independence": {
            "labels_used": False,
            "source_name_used": False,
            "source_name_role": "manifest and post-route audit only",
        },
        "policy": route_policy_definition(),
        "policy_sha256": route_policy_sha256(),
        "population": {
            "train_root": str(train_root),
            "video_count": len(records),
            "event_count": int(sum(record["event_count"] for record in records)),
            "event_count_gt_30000": int(
                sum(record["event_count"] > LOW_DENSITY_MAX_EVENT_COUNT for record in records)
            ),
            "event_count_gt_200000": len(high_names),
            "event_count_200001_to_250000": len(gap_names),
            "gt_200000_names": list(high_names),
            "gt_200000_matches_existing_15_source_evidence": True,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "density_gate_protection": {
            "sources_that_polarity_only_would_send_to_t32": len(protected),
            "protected_source_names": [record["source_name"] for record in protected],
            "protected_routes": {
                record["source_name"]: "{}/{}".format(
                    record["checkpoint_role"], record["mode"]
                )
                for record in protected
            },
            "unassessed_below_200k_sources_sent_to_t32": 0,
        },
        "persistence_second_stage": {
            "enabled": False,
            "reason": (
                "The current grouped OOF persistence audit used M20 full-stream "
                "scores; interaction with routed H2 T32 scores is not yet audited."
            ),
        },
        "records": records,
        "provenance": {
            "code_paths": {name: str(path) for name, path in code_paths.items()},
            "code_sha256": {
                name: sha256_file(path) for name, path in code_paths.items()
            },
        },
    }
    digest, sidecar = _atomic_json(args.output, payload)
    print("report:", Path(args.output).resolve())
    print("report_sha256:", digest)
    print("sha256_sidecar:", sidecar)
    print("route_counts:", dict(sorted(route_counts.items())))
    print("protected_below_200k:", len(protected))
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

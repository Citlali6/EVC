"""CPU-only train-only diagnostic: C00 removal recovery capacity + feature separability.

Phase 1 diagnostic on the frozen low-domain route (M10/full-stream/0.718).
Two questions, both answered without reading val/test:

A) How many target frames are removed by the C00 chain despite raw score >=
   threshold, and what would conservative "restore" rules gain on train?

B) On the exact C00-related topologies, are label-free features able to
   separate (a) surviving pure-FP components from target components, and
   (b) removed target components from removed pure-FP components?

Two partitions, each over events with score >= threshold:
  - DELETE partition: components of FINAL scores (exact post-C00 predicted
    events).  Identical to run_low_domain_component_oracle.py, so the
    delete-all-pure-FP rule must reproduce its +0.0035973761 as a ground check.
  - RESTORE partition: components of RAW scores whose events are not already
    predicted (final < threshold).  Restoring sets final = raw for the whole
    component (events with raw >= threshold become predicted again).

Labels/target IDs are attached only after the label-free partitions exist.
No validation/test paths, no model inference, no CUDA, no source-identity
features in the feature set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import run_temporal_memory_input_route_train as routed
import run_low_domain_component_oracle as oracle
from dataset.temporal_frame import load_temporal_frame_video
from utils.atomic_component_deletion import extract_atomic_components
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXPERIMENT_ROOT = WORKSPACE / "experiments" / "20260812_low_c00_recovery_separability_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "train_only_recovery_and_separability.json"

WIDTH, HEIGHT = 346, 260
BIN = 50
FEATURE_NAMES = [
    "log_video_events", "log_component_events", "score_max", "score_mean",
    "score_min", "score_std", "score_margin_max", "log_unique_cells",
    "bbox_diagonal", "duration_bins", "max_events_per_bin",
    "displacement_per_bin", "max_gap_bins", "t_span",
]


FAMILY_MAP = {}

def family_for(source_name):
    """Lazy family lookup: dynamic map first, handover low families as fallback."""
    fam = FAMILY_MAP.get(source_name)
    if fam is None:
        fam = oracle.family_for_name(source_name)
    return fam



def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_exclusive(path: Path, payload) -> str:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = oracle.sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        "{}  {}\n".format(digest, path.name), encoding="ascii"
    )
    return digest


def component_features(video, indices, thr):
    """Label-free features for one atomic component (any partition)."""
    idx = np.asarray(indices, dtype=np.int64)
    scores = video.raw_scores[idx].astype(np.float64)
    locs = video.locations4[idx]
    x = locs[:, 1].astype(np.int64)
    y = locs[:, 2].astype(np.int64)
    t = locs[:, 3].astype(np.int64)
    bins = t // BIN
    unique_bins = np.unique(bins)
    n_events = idx.size
    cells = np.unique(np.stack([x, y], axis=1), axis=0)
    max_raw = float(scores.max())
    displacement = 0.0
    if unique_bins.size >= 2:
        first = int(np.argmin(bins))
        last = int(np.argmax(bins))
        displacement = float(
            math.hypot(x[last] - x[first], y[last] - y[first])
            / max(1, int(unique_bins[-1] - unique_bins[0]))
        )
    max_gap = 0
    if unique_bins.size >= 2:
        max_gap = int(np.max(np.diff(unique_bins)))
    per_bin_counts = (
        np.bincount(bins - unique_bins.min(), minlength=unique_bins.size)
        if unique_bins.size else np.zeros(1)
    )
    return {
        "log_video_events": float(math.log1p(video.event_count)),
        "log_component_events": float(math.log1p(n_events)),
        "score_max": max_raw,
        "score_mean": float(scores.mean()),
        "score_min": float(scores.min()),
        "score_std": float(scores.std()),
        "score_margin_max": max_raw - thr,
        "log_unique_cells": float(math.log1p(cells.shape[0])),
        "bbox_diagonal": float(math.hypot(x.max() - x.min(), y.max() - y.min())),
        "duration_bins": int(unique_bins.size),
        "max_events_per_bin": int(per_bin_counts.max()),
        "displacement_per_bin": displacement,
        "max_gap_bins": max_gap,
        "t_span": float(int(t.max() - t.min())),
    }


def build_family_map(records):
    """Contiguous train-index runs per domain -> family names.

    For the low domain this reproduces the handover's frozen families
    (015-027, 033-039, 048-058, 066, 075-087); middle/high get the same
    contiguous-run rule so no source is ever split or reshuffled.
    """
    mapping = {}
    by_domain = {}
    for record in records:
        by_domain.setdefault(record["decision"]["domain"], []).append(record)
    for domain in sorted(by_domain):
        dom_records = sorted(
            by_domain[domain],
            key=lambda r: int(r["source_name"].split("_")[1].split(".")[0]),
        )
        runs = []
        current = [dom_records[0]["source_name"]]
        last_num = int(dom_records[0]["source_name"].split("_")[1].split(".")[0])
        for record in dom_records[1:]:
            num = int(record["source_name"].split("_")[1].split(".")[0])
            if num == last_num + 1:
                current.append(record["source_name"])
            else:
                runs.append(tuple(current))
                current = [record["source_name"]]
            last_num = num
        runs.append(tuple(current))
        for run_index, run in enumerate(runs):
            for name in run:
                mapping[name] = "{}_f{}".format(domain, run_index)
    return mapping


def prepare_video_any(metadata):
    """Generalized version of oracle.prepare_video for any domain.

    - The baseline/candidate bitwise identity check follows the manifest flag
      (H2 sources legitimately differ: candidate is window T32, baseline is
      full-stream M20; the golden route uses the full-stream baseline).
    - The C00 chain runs at the per-video threshold from the manifest
      (oracle.p0_and_p18 hardcodes the low threshold 0.718).
    """
    source_name = str(metadata["source_name"])
    source_path = oracle.TRAIN_ROOT / source_name
    if not source_path.is_file() or oracle.sha256_file(source_path) != metadata["source_sha256"]:
        raise RuntimeError("raw train source changed: {}".format(source_name))
    cache_path = (oracle.CACHE_DIR / metadata["record"]).resolve()
    if not cache_path.is_file() or oracle.sha256_file(cache_path) != metadata["record_sha256"]:
        raise RuntimeError("score cache changed: {}".format(source_name))
    with np.load(cache_path, allow_pickle=False) as archive:
        if set(archive.files) != {"baseline_scores", "candidate_scores"}:
            raise RuntimeError("cache arrays differ: {}".format(source_name))
        baseline_scores = np.asarray(archive["baseline_scores"], dtype=np.float32).copy()
        candidate_scores = np.asarray(archive["candidate_scores"], dtype=np.float32).copy()
    if metadata.get("bitwise_equal_to_baseline") is True:
        if not np.array_equal(baseline_scores.view(np.uint32), candidate_scores.view(np.uint32)):
            raise RuntimeError("cache not bitwise identity: {}".format(source_name))

    with np.load(source_path, allow_pickle=False) as archive:
        locations3 = np.asarray(archive["ev_loc"], dtype=np.int64).copy()
    if locations3.shape != (int(metadata["event_count"]), 3):
        raise RuntimeError("input locations differ from cache event count: {}".format(source_name))
    locations4 = np.column_stack((np.zeros(locations3.shape[0], dtype=np.int64), locations3))
    thr = float(metadata["decision"]["prediction_threshold"])
    cfg = routed.c00_config()
    full_processor = ChallengePostprocessor.from_cfg(cfg, thr, event_count=int(metadata["event_count"]))
    final_scores, stats = full_processor.apply(
        torch.from_numpy(np.asarray(baseline_scores, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(locations4, dtype=np.int64).copy()),
    )
    final_scores = final_scores.numpy().astype(np.float32, copy=True)

    video = load_temporal_frame_video(source_path, BIN, 8000)
    if not np.array_equal(video.locations, locations3):
        raise RuntimeError("canonical truth loader locations differ: {}".format(source_name))
    result = oracle.PreparedLowVideo(
        source_name=source_name,
        family=family_for(source_name),
        source_sha256=str(metadata["source_sha256"]),
        cache_sha256=str(metadata["record_sha256"]),
        event_count=int(metadata["event_count"]),
        raw_scores=baseline_scores,
        p0_scores=final_scores,
        final_scores=final_scores,
        locations4=locations4,
        labels=video.labels.astype(np.uint8, copy=True),
        target_ids=video.target_ids.copy(),
        components=(),
        pure_fp=np.zeros(0, dtype=bool),
        baseline=oracle.Counts(),
    )
    result.threshold = thr
    result.baseline = official_counts_thr(final_scores, result)
    result.family = family_for(source_name)
    return result


def build_partitions(video):
    """Return (delete_rows, restore_rows, delete_indices, restore_indices).

    delete: components of events with final >= thr (post-C00 predicted set).
    restore: components of events with raw >= thr that are not predicted.
    """
    thr = float(getattr(video, "threshold", oracle.LOW_THRESHOLD))
    del_comps = extract_atomic_components(
        video.final_scores, video.locations4, thr,
        spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
    ).event_indices
    restore_comps = extract_atomic_components(
        video.raw_scores, video.locations4, thr,
        spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
    ).event_indices
    predicted = video.final_scores >= thr
    delete_rows, restore_rows = [], []
    delete_indices, restore_indices = [], []
    for cindex, indices in enumerate(del_comps):
        idx = np.asarray(indices, dtype=np.int64)
        has_target = bool(np.any(video.labels[idx] > 0))
        delete_rows.append({
            "component_index": cindex,
            "has_target": has_target,
            "pure_fp": (not has_target),
            "final_max": float(np.max(video.final_scores[idx])),
            "raw_max": float(np.max(video.raw_scores[idx])),
            **component_features(video, idx, thr),
        })
        delete_indices.append(idx)
    restore_counter = 0
    for indices in restore_comps:
        idx = np.asarray(indices, dtype=np.int64)
        if np.any(predicted[idx]):
            continue  # already predicted; nothing to restore
        has_target = bool(np.any(video.labels[idx] > 0))
        restore_rows.append({
            "component_index": restore_counter,
            "has_target": has_target,
            "pure_fp": (not has_target),
            "final_max": float(np.max(video.final_scores[idx])),
            "raw_max": float(np.max(video.raw_scores[idx])),
            **component_features(video, idx, thr),
        })
        restore_indices.append(idx)
        restore_counter += 1
    return delete_rows, restore_rows, delete_indices, restore_indices


def apply_actions(video, delete_indices, restore_indices, delete_mask, restore_mask):
    """Return modified final scores after deleting/restoring whole components."""
    output = video.final_scores.copy()
    assigned = np.zeros(output.size, dtype=bool)
    for component_index, indices in enumerate(delete_indices):
        if np.any(assigned[indices]):
            raise RuntimeError("delete components overlap")
        assigned[indices] = True
        if np.asarray(delete_mask, dtype=bool)[component_index]:
            output[indices] = np.float32(0.0)
    for component_index, indices in enumerate(restore_indices):
        if np.any(assigned[indices]):
            raise RuntimeError("restore components overlap with delete set")
        assigned[indices] = True
        if np.asarray(restore_mask, dtype=bool)[component_index]:
            output[indices] = video.raw_scores[indices].astype(np.float32, copy=True)
    if not np.array_equal(output[~assigned].view(np.uint32), video.final_scores[~assigned].view(np.uint32)):
        raise RuntimeError("non-component scores changed")
    return output


def official_counts_thr(scores, video):
    thr = float(getattr(video, "threshold", oracle.LOW_THRESHOLD))
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(video.labels, dtype=np.uint8).reshape(-1)
    ids = np.asarray(video.target_ids).reshape(-1)
    locations = np.asarray(video.locations4, dtype=np.int64)
    evaluator = evalute(type("Config", (), {"roc": True, "pd_detT": 50, "correct_thresh": 0.0001})())
    evaluator.roc_update(
        torch.from_numpy(locations[:, 3].copy()),
        torch.from_numpy(values.copy()),
        ids,
        torch.from_numpy(truth.astype(np.float32, copy=False)),
        torch.from_numpy(locations.copy()),
        thresh=float(thr),
    )
    predicted = values >= thr
    positive = truth > 0
    return oracle.Counts(
        true_positive_events=int(np.count_nonzero(predicted & positive)),
        false_positive_events=int(np.count_nonzero(predicted & ~positive)),
        false_negative_events=int(np.count_nonzero(~predicted & positive)),
        correct_target_frames=int(evaluator.correct_num),
        target_frames=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
        event_count=int(values.size),
    )


def frame_level_misses(video):
    """Per-(target,frame) miss classification using official window semantics.

    Also returns the official evaluator counts on final scores as a ground
    check for the frame bookkeeping.
    """
    thr = float(getattr(video, "threshold", oracle.LOW_THRESHOLD))
    t = video.locations4[:, 3]
    frame_num = int((float(t.max()) - float(t.min())) / BIN)
    stats = {}
    for event_index in range(video.labels.size):
        if video.labels[event_index] <= 0:
            continue
        gid = int(video.target_ids[event_index])
        ts = float(t[event_index])
        i = int(ts // BIN)
        if not (i * BIN < ts < (i + 1) * BIN) or i > frame_num:
            continue
        frame = stats.setdefault((gid, i), {"target_events": 0, "detected_events": 0, "max_raw": 0.0})
        frame["target_events"] += 1
        frame["max_raw"] = max(frame["max_raw"], float(video.raw_scores[event_index]))
        if video.final_scores[event_index] >= thr:
            frame["detected_events"] += 1
    misses = {"below_threshold": 0, "removed_by_c00": 0, "partial": 0}
    restore_candidate_frames = 0
    for (gid, i), frame in stats.items():
        detected = frame["detected_events"] / frame["target_events"] >= 0.0001
        if detected:
            continue
        if frame["max_raw"] < thr:
            misses["below_threshold"] += 1
        elif frame["detected_events"] > 0:
            misses["partial"] += 1
        else:
            misses["removed_by_c00"] += 1
            restore_candidate_frames += 1
    counts = official_counts_thr(video.final_scores, video)
    return misses, restore_candidate_frames, counts


def replay_rules(videos, partitions, rule_specs):
    results = {}
    for rule_name, del_fn, res_fn in rule_specs:
        per_source = {}
        for video in videos:
            delete_rows, restore_rows, delete_indices, restore_indices = partitions[video.source_name]
            del_mask = del_fn(video, delete_rows)
            res_mask = res_fn(video, restore_rows)
            scores = apply_actions(video, delete_indices, restore_indices, del_mask, res_mask)
            per_source[video.source_name] = official_counts_thr(scores, video)
        baseline = oracle.sum_counts(video.baseline for video in videos)
        candidate = oracle.sum_counts(per_source.values())
        results[rule_name] = {
            "count_delta": oracle.count_delta(candidate, baseline),
            "metric_delta": oracle.metric_delta(candidate, baseline),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domains", default="low", help="low | middle | high | all")
    args = parser.parse_args()

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only diagnostic refuses an initialized CUDA process")

    manifest, _ = oracle.validate_inputs()
    records = manifest["records"]
    global FAMILY_MAP
    FAMILY_MAP = build_family_map(records)
    domain_scope = args.domains
    if domain_scope == "low":
        records = [r for r in records if r["decision"]["domain"] == "low"]
    elif domain_scope == "middle":
        records = [r for r in records if r["decision"]["domain"] == "middle"]
    elif domain_scope == "high":
        records = [r for r in records if r["decision"]["domain"] in ("h1", "h2")]
    elif domain_scope in ("h1", "h2"):
        records = [r for r in records if r["decision"]["domain"] == domain_scope]
    elif domain_scope != "all":
        raise ValueError("unknown --domains: {}".format(domain_scope))

    videos = []
    partitions = {}
    for index, metadata in enumerate(records, start=1):
        video = prepare_video_any(metadata)
        delete_rows, restore_rows, delete_indices, restore_indices = build_partitions(video)
        partitions[video.source_name] = (delete_rows, restore_rows, delete_indices, restore_indices)
        videos.append(video)
        print("prepared {}/{} {}: del {} ({} pure-FP) restore {} ({} target)".format(
            index, len(records), video.source_name,
            len(delete_rows), sum(1 for r in delete_rows if r["pure_fp"]),
            len(restore_rows), sum(1 for r in restore_rows if r["has_target"]),
        ), flush=True)

    # ---- Part A: frame-level miss classification + evaluator ground check ----
    miss_summary = {"below_threshold": 0, "removed_by_c00": 0, "partial": 0}
    restore_candidate_frames = 0
    frame_checks = {}
    for video in videos:
        misses, cand_frames, counts = frame_level_misses(video)
        for key in miss_summary:
            miss_summary[key] += misses[key]
        restore_candidate_frames += cand_frames
        frame_checks[video.source_name] = {
            "misses": misses,
            "restore_candidate_frames": cand_frames,
            "my_obj_num": counts.target_frames,
            "my_correct_num": counts.correct_target_frames,
            "oracle_obj_num": video.baseline.target_frames,
            "oracle_correct_num": video.baseline.correct_target_frames,
            "frame_check_ok": (
                counts.target_frames == video.baseline.target_frames
                and counts.correct_target_frames == video.baseline.correct_target_frames
            ),
        }

    # ---- Rule replays ----
    def delete_all_pure_fp(video, delete_rows):
        return np.array([r["pure_fp"] for r in delete_rows], dtype=bool)

    def none(video, rows):
        return np.zeros(len(rows), dtype=bool)

    def restore_all_targets(video, restore_rows):
        return np.array([r["has_target"] for r in restore_rows], dtype=bool)

    def restore_conservative(video, restore_rows):
        mask = np.zeros(len(restore_rows), dtype=bool)
        for r in restore_rows:
            if r["has_target"] and r["duration_bins"] >= 2 and r["log_component_events"] >= math.log1p(3):
                mask[r["component_index"]] = True
        return mask

    rules = replay_rules(videos, partitions, [
        ("delete_all_pure_fp", delete_all_pure_fp, none),
        ("restore_all_target_components", none, restore_all_targets),
        ("restore_conservative_target_3ev_2bin", none, restore_conservative),
        ("joined_delete_pure_fp_restore_targets", delete_all_pure_fp, restore_all_targets),
    ])

    # ---- Part B: feature separability (per-family AUC) ----
    fam_rows = {}
    for video in videos:
        delete_rows, restore_rows, _, _ = partitions[video.source_name]
        fam_rows.setdefault(video.family, {"delete": [], "restore": []})
        fam_rows[video.family]["delete"].extend(delete_rows)
        fam_rows[video.family]["restore"].extend(restore_rows)

    def auc_for(rows, positive_key):
        from sklearn.metrics import roc_auc_score
        out = {}
        target = np.array([1 if r[positive_key] else 0 for r in rows], dtype=np.int64)
        for fname in FEATURE_NAMES:
            values = np.array([r[fname] for r in rows], dtype=np.float64)
            if len(np.unique(target)) < 2 or values.size == 0:
                out[fname] = None
                continue
            out[fname] = float(roc_auc_score(target, values))
        return out

    separability = {}
    for fam in sorted(fam_rows):
        d_rows = fam_rows[fam]["delete"]
        r_rows = fam_rows[fam]["restore"]
        separability[fam] = {
            "delete_components": len(d_rows),
            "delete_pure_fp": sum(1 for r in d_rows if r["pure_fp"]),
            "restore_components": len(r_rows),
            "restore_with_target": sum(1 for r in r_rows if r["has_target"]),
            "auc_pure_fp_vs_target_delete": auc_for(d_rows, "pure_fp"),
            "auc_target_vs_fp_restore": auc_for(r_rows, "has_target"),
        }

    payload = {
        "schema": "ev-uav-low-c00-recovery-and-separability-train-only-v1",
        "created_utc": utc_now(),
        "dataset_split": "train",
        "domains": domain_scope,
        "validation_or_test_read": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "recovery_capacity": {
            "misses_by_kind": miss_summary,
            "restore_candidate_frames": restore_candidate_frames,
            "per_source_frame_checks": frame_checks,
        },
        "rule_replays": rules,
        "separability_by_family": separability,
        "feature_names": FEATURE_NAMES,
        "inputs": {
            "manifest_path": str(oracle.MANIFEST_PATH),
            "manifest_sha256": oracle.EXPECTED_MANIFEST_SHA256,
            "runner_sha256": oracle.sha256_file(Path(__file__)),
        },
    }
    digest = write_json_exclusive(args.output, payload)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "sha256": digest,
        "misses": miss_summary,
        "restore_candidate_frames": restore_candidate_frames,
        "frame_checks_all_ok": all(v["frame_check_ok"] for v in frame_checks.values()),
        "rules": {k: v["metric_delta"] for k, v in rules.items()},
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

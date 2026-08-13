"""Train-only oracle coverage check for repair-then-delete.

This deliberately stops before model selection.  It recreates the frozen
positive-support V2 near-miss from F1--F4 OOF, generates P18 proposals after
the proposed deletions using observable scores/locations only, and checks
with train labels afterwards whether every lost positive event/target group
has a repair proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_allsize_deletion_head_oof as base
import run_positive_support_distance_guard_f5_smoke as deletion
from train_component_reranker import _load_cache_record
from utils import positive_support_guard as support_features
from utils.contextual_track_recovery import extract_contextual_track_edge_candidates
from utils.track_edge_recovery import FROZEN_TOPOLOGY


EVIDENCE_CHOICE = "relative_anchor"
MODE = "nearest"
FAMILY_AGGREGATION = "maximum"
ANCHOR_VETO = "none"
CUTOFF_RULE = "minimum"


def official_group(video, event_index):
    timestamp = int(video.locations[int(event_index), 3])
    if timestamp <= 0 or timestamp % FROZEN_TOPOLOGY.temporal_bin_size == 0:
        return None
    target_id = int(video.target_ids[int(event_index)])
    if target_id <= 0:
        return None
    return (timestamp // FROZEN_TOPOLOGY.temporal_bin_size, target_id)


def run(args):
    cache = Path(args.cache_dir).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    base.FEATURE_NAMES = support_features.FEATURE_NAMES
    base.extract_allsize_components = support_features.extract_positive_support_components
    manifest, _cfg, videos = base.prepare_videos(cache, c00_protocol)
    metadata = {record["source_name"]: record for record in manifest["records"]}

    evidence_names = support_features.EVIDENCE_SETS[EVIDENCE_CHOICE]
    models = deletion._fit_family_models(videos, evidence_names, MODE)
    supports = {}
    endpoints = {}
    for held_family in deletion.FIT_FAMILIES:
        held = [video for video in videos if video.group == held_family]
        for video in held:
            supports[video.source_name] = deletion._support(
                models, held_family, video, FAMILY_AGGREGATION, ANCHOR_VETO
            )
        family_positive = np.concatenate(
            [supports[video.source_name][video.component_labels > 0] for video in held]
        )
        endpoints[held_family] = float(family_positive.min())

    rows = []
    total_lost_events = 0
    covered_lost_events = 0
    lost_groups = set()
    covered_groups = set()
    total_proposals = 0
    positive_proposals = 0
    fold_records = []
    for held_family in deletion.FIT_FAMILIES:
        cutoff = deletion._cutoff(
            [value for family, value in endpoints.items() if family != held_family],
            CUTOFF_RULE,
        )
        held_videos = [item for item in videos if item.group == held_family]
        fold_baseline = base._sum_counts(video.baseline_counts for video in held_videos)
        fold_candidates = []
        for video in held_videos:
            fold_candidates.append(
                base._candidate_counts(video, supports[video.source_name], cutoff)
            )
            removed_positions = np.flatnonzero(supports[video.source_name] < cutoff)
            if removed_positions.size == 0:
                continue
            removed_indices = tuple(
                np.asarray(video.event_indices[int(position)], dtype=np.int64)
                for position in removed_positions
            )
            flat_removed = np.concatenate(removed_indices)
            lost = flat_removed[video.labels[flat_removed] > 0]
            deleted_scores = video.scores.copy()
            deleted_scores[flat_removed] = np.float32(0.0)
            record = _load_cache_record(cache, metadata[video.source_name])
            raw = record["scores"].reshape(-1).astype(np.float32, copy=False)
            proposals = extract_contextual_track_edge_candidates(
                raw,
                deleted_scores,
                video.locations,
                video.event_count,
                FROZEN_TOPOLOGY,
            )
            proposal_indices = np.asarray(
                [candidate.event_index for candidate in proposals], dtype=np.int64
            )
            proposal_positive = (
                proposal_indices[video.labels[proposal_indices] > 0]
                if proposal_indices.size
                else np.empty(0, dtype=np.int64)
            )
            proposal_groups = {
                group
                for index in proposal_positive.tolist()
                for group in (official_group(video, index),)
                if group is not None
            }
            source_lost_groups = {
                group
                for index in lost.tolist()
                for group in (official_group(video, index),)
                if group is not None
            }
            group_remaining_support = {}
            event_frames = np.floor_divide(video.locations[:, 3], 50)
            event_valid = (
                (video.locations[:, 3] > 0)
                & (np.mod(video.locations[:, 3], 50) != 0)
            )
            for group in source_lost_groups:
                frame, target_id = group
                group_mask = (
                    event_valid
                    & (event_frames == frame)
                    & (video.target_ids == target_id)
                    & (deleted_scores >= base.EFFECTIVE_THRESHOLD)
                )
                group_remaining_support[str(group)] = int(group_mask.sum())
            directly_covered = set(lost.tolist()) & set(proposal_indices.tolist())
            total_lost_events += int(lost.size)
            covered_lost_events += len(directly_covered)
            lost_groups.update((video.source_name, group) for group in source_lost_groups)
            covered_groups.update(
                (video.source_name, group)
                for group in source_lost_groups & proposal_groups
            )
            total_proposals += len(proposals)
            positive_proposals += int(proposal_positive.size)
            rows.append(
                {
                    "source_name": video.source_name,
                    "held_family": held_family,
                    "learned_cutoff": cutoff,
                    "removed_component_count": int(removed_positions.size),
                    "removed_event_count": int(flat_removed.size),
                    "removed_false_event_count": int(
                        np.sum(video.labels[flat_removed] == 0)
                    ),
                    "lost_positive_event_indices": lost.tolist(),
                    "lost_target_groups": [list(group) for group in sorted(source_lost_groups)],
                    "proposal_count_after_delete": len(proposals),
                    "positive_proposal_count_after_delete": int(proposal_positive.size),
                    "directly_covered_lost_event_indices": sorted(directly_covered),
                    "covered_lost_target_groups": [
                        list(group) for group in sorted(source_lost_groups & proposal_groups)
                    ],
                    "remaining_predicted_support_by_lost_group": group_remaining_support,
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

        fold_candidate = base._sum_counts(fold_candidates)
        fold_baseline_metrics = base.crossfit.metrics_from_counts(fold_baseline)
        fold_candidate_metrics = base.crossfit.metrics_from_counts(fold_candidate)
        fold_records.append(
            {
                "held_family": held_family,
                "baseline": {"counts": fold_baseline.to_dict(), "metrics": fold_baseline_metrics},
                "candidate": {"counts": fold_candidate.to_dict(), "metrics": fold_candidate_metrics},
                "score_delta": fold_candidate_metrics["score"] - fold_baseline_metrics["score"],
                "gates": {
                    "score_not_lower": fold_candidate_metrics["score"] >= fold_baseline_metrics["score"],
                    "iou_not_lower": fold_candidate_metrics["iou"] >= fold_baseline_metrics["iou"],
                    "pd_not_lower": fold_candidate_metrics["pd"] >= fold_baseline_metrics["pd"],
                    "fa_not_higher": fold_candidate_metrics["fa"] <= fold_baseline_metrics["fa"],
                    "correct_objects_not_lower": fold_candidate.correct_objects >= fold_baseline.correct_objects,
                    "tp_loss_within_0_05_percent": (
                        fold_baseline.true_positive_events - fold_candidate.true_positive_events
                        <= 0.0005 * fold_baseline.true_positive_events
                    ),
                },
            }
        )

    report = {
        "schema": "ev-uav-repair-then-delete-coverage-train-f1-f4-v1",
        "dataset_split": "train",
        "no_validation_test_or_gpu_access": True,
        "deletion_configuration": {
            "evidence_choice": EVIDENCE_CHOICE,
            "mode": MODE,
            "family_aggregation": FAMILY_AGGREGATION,
            "anchor_veto": ANCHOR_VETO,
            "cutoff_rule": CUTOFF_RULE,
            "endpoints": endpoints,
        },
        "summary": {
            "lost_positive_events": total_lost_events,
            "directly_covered_lost_events": covered_lost_events,
            "lost_target_groups": len(lost_groups),
            "covered_lost_target_groups": len(covered_groups),
            "proposal_count_after_delete": total_proposals,
            "positive_proposal_count_after_delete": positive_proposals,
            "coverage_sufficient": bool(
                total_lost_events > 0
                and covered_lost_events == total_lost_events
                and len(covered_groups) == len(lost_groups)
            ),
        },
        "official_f1_f4_folds": fold_records,
        "affected_sources": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["summary"]["coverage_sufficient"] else 2


def parser():
    root = Path(__file__).resolve().parent
    experiments = root.parent / "experiments"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--cache-dir",
        default=str(
            experiments
            / "20260810_component_reranker_crosssource_v1"
            / "train_cache_gt30000"
        ),
    )
    result.add_argument(
        "--c00-protocol",
        default=str(
            experiments
            / "20260810_component_reranker_crosssource_v1"
            / "crossfit_protocol.json"
        ),
    )
    result.add_argument(
        "--output",
        default=str(
            experiments
            / "20260811_repair_then_delete_joint_f5_smoke_v1"
            / "coverage_report.json"
        ),
    )
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))

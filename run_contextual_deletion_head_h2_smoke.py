"""Frozen train-only F5 smoke for contextual all-size deletion features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import run_allsize_deletion_head_oof as base
from utils import contextual_deletion_head as contextual


EXPECTED_SCIENCE_PROTOCOL_SHA256 = "b5549303eff1120d4fea929626cdd7cbcc0f008a2491d71cb5d5ecf6bbc778eb"
EXPECTED_CONTEXTUAL_MODULE_SHA256 = "60395f627256d58c87b9b07c2771ee638631558c2f278820769d770a25e95c3d"
FAMILIES = ("logistic_l2", "histgb_shallow", "extratrees_shallow")
HELD_GROUP = "block_088_098"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_record(counts):
    return {"counts": counts.to_dict(), "metrics": base.crossfit.metrics_from_counts(counts)}


def safety(candidate, baseline):
    cm = base.crossfit.metrics_from_counts(candidate)
    bm = base.crossfit.metrics_from_counts(baseline)
    return {
        "score_not_lower": cm["score"] >= bm["score"],
        "iou_not_lower": cm["iou"] >= bm["iou"],
        "pd_not_lower": cm["pd"] >= bm["pd"],
        "fa_not_higher": cm["fa"] <= bm["fa"],
        "correct_objects_not_lower": candidate.correct_objects >= baseline.correct_objects,
        "tp_not_lower": candidate.true_positive_events >= baseline.true_positive_events,
    }


def run(args):
    root = Path(__file__).resolve().parent
    science = Path(args.science_protocol).resolve()
    output = Path(args.output_report).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(science) != EXPECTED_SCIENCE_PROTOCOL_SHA256:
        raise ValueError("contextual science protocol changed.")
    if sha256_file(root / "utils" / "contextual_deletion_head.py") != EXPECTED_CONTEXTUAL_MODULE_SHA256:
        raise ValueError("contextual feature module changed after science freeze.")
    protocol = json.loads(science.read_text(encoding="utf-8"))
    if protocol["scope"]["dataset_split"] != "train" or protocol["scope"]["validation_allowed"] or protocol["scope"]["test_allowed"]:
        raise ValueError("science scope is not train-only.")

    base.FEATURE_NAMES = contextual.FEATURE_NAMES
    base.extract_allsize_components = contextual.extract_allsize_components
    cache = Path(args.cache_dir).resolve()
    c00_protocol = Path(args.c00_protocol).resolve()
    manifest, _cfg, videos = base.prepare_videos(cache, c00_protocol)
    fit = [video for video in videos if video.group != HELD_GROUP]
    held = [video for video in videos if video.group == HELD_GROUP]
    baseline = base._sum_counts(video.baseline_counts for video in held)
    candidates = []
    for family in FAMILIES:
        model = base._fit_model(family, fit)
        thresholds = base._fit_derived_thresholds(model, fit)
        held_labels = np.concatenate([video.component_labels for video in held])
        held_probabilities = np.concatenate([base._probabilities(model, video) for video in held])
        separability = {
            "component_roc_auc": float(roc_auc_score(held_labels, held_probabilities)),
            "component_average_precision": float(average_precision_score(held_labels, held_probabilities)),
            "held_component_count": int(held_labels.size),
            "held_positive_component_count": int(held_labels.sum()),
        }
        for risk_rule, threshold in thresholds.items():
            per_source = []
            counts = []
            for video in held:
                candidate = base._candidate_counts(
                    video, base._probabilities(model, video), threshold
                )
                counts.append(candidate)
                source_delta = base.crossfit.metrics_from_counts(candidate)["score"] - base.crossfit.metrics_from_counts(video.baseline_counts)["score"]
                per_source.append({"source_name": video.source_name, "score_delta": source_delta})
            candidate = base._sum_counts(counts)
            gates = safety(candidate, baseline)
            candidates.append(
                {
                    "family": family,
                    "risk_rule": risk_rule,
                    "fit_derived_keep_threshold": threshold,
                    "separability": separability,
                    "baseline": metrics_record(baseline),
                    "candidate": metrics_record(candidate),
                    "score_delta": base.crossfit.metrics_from_counts(candidate)["score"] - base.crossfit.metrics_from_counts(baseline)["score"],
                    "gates": gates,
                    "safety_passed": all(gates.values()),
                    "worst_source": min(per_source, key=lambda item: item["score_delta"]),
                }
            )
    nonidentity_signal = [
        item for item in candidates
        if item["risk_rule"] != "identity" and item["safety_passed"] and item["score_delta"] > 0.0
    ]
    report = {
        "schema": "ev-uav-contextual-allsize-deletion-head-f5-smoke-v2",
        "dataset_split": "train",
        "evidence_class": "single_frozen_family_separability_smoke_not_model_selection",
        "held_group": HELD_GROUP,
        "fit_groups": [group for group in base.SOURCE_GROUPS if group != HELD_GROUP],
        "feature_count": len(contextual.FEATURE_NAMES),
        "feature_names": list(contextual.FEATURE_NAMES),
        "source_name_or_index_is_feature": False,
        "inputs": {
            "science_protocol_path": str(science),
            "science_protocol_sha256": sha256_file(science),
            "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
            "c00_protocol_sha256": sha256_file(c00_protocol),
            "contextual_module_sha256": sha256_file(root / "utils" / "contextual_deletion_head.py"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_count": manifest["selected_video_count"],
            "event_count": manifest["selected_event_count"],
        },
        "candidates": candidates,
        "nonidentity_safe_signal": bool(nonidentity_signal),
        "best_safe_signal": max(nonidentity_signal, key=lambda item: item["score_delta"]) if nonidentity_signal else None,
        "decision": "proceed_to_full_nested_oof" if nonidentity_signal else "stop_contextual_v2",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output_report": str(output), "decision": report["decision"], "best_safe_signal": None if report["best_safe_signal"] is None else {key: report["best_safe_signal"][key] for key in ("family", "risk_rule", "score_delta", "fit_derived_keep_threshold")}}, indent=2))
    return 0


def parser():
    root = Path(__file__).resolve().parent
    experiments = root.parent / "experiments"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--science-protocol", default=str(root / "protocols" / "contextual_allsize_deletion_head_science_v2.json"))
    result.add_argument("--cache-dir", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "train_cache_gt30000"))
    result.add_argument("--c00-protocol", default=str(experiments / "20260810_component_reranker_crosssource_v1" / "crossfit_protocol.json"))
    result.add_argument("--output-report", default=str(experiments / "20260811_contextual_deletion_head_v2_smoke" / "h2_smoke_report.json"))
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))

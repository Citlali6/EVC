"""CPU-only G3 disagreement capacity for whole-component pyramid recovery V2."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import crossfit_component_reranker as crossfit
from utils.atomic_component_deletion import extract_atomic_components
from utils.h2_pyramid_component_recovery import restore_whole_components_bitwise


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXPERIMENT_ROOT = (
    WORKSPACE / "experiments" / "20260811_h2_multiscale_temporal_pyramid_expert_v1"
)
PAIRED_PATH = EXPERIMENT_ROOT / "held_train_evaluation" / "hold_g3" / "paired_evaluation.json"
EXPECTED_PAIRED_SHA256 = "1faef18185288ce27e88510b562e22f76397190b5165db4ef7613fdf34016817"
SOURCE_PROTOCOL_PATH = ROOT / "protocols" / "h2_spatiotemporal_residual_refiner_oof_science_v1.json"
EXPECTED_SOURCE_PROTOCOL_SHA256 = "7edec461f2ccc8047156f08c57389319a5defd59d0afcea69cbfcf32e81d2207"
TRAIN_ROOT = (WORKSPACE / "datasets" / "EV-UAV-Challenge2" / "train").resolve()
G3_SOURCES = tuple("train_{:03d}.npz".format(value) for value in range(95, 99))
THRESHOLD = 0.719
OUTPUT_PATH = (
    WORKSPACE
    / "experiments"
    / "20260811_h2_pyramid_component_recovery_v2"
    / "cpu_disagreement_capacity"
    / "report.json"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_exclusive(path, payload):
    values = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(values)
        stream.flush()
        os.fsync(stream.fileno())
    result = hashlib.sha256(values).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    descriptor = os.open(str(sidecar), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((result + "  " + path.name + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    return result


def challenge_report(counts):
    metrics = crossfit.metrics_from_counts(counts)
    return {
        "Score": float(metrics["score"]),
        "IoU": float(metrics["iou"]),
        "Pd": float(metrics["pd"]),
        "Fa": float(metrics["fa"]),
        "TP": int(counts.true_positive_events),
        "FP": int(counts.false_positive_events),
        "CO": int(counts.correct_objects),
        "FC": int(counts.false_components),
    }


def report_delta(reference, candidate):
    output = {}
    for key in reference:
        value = candidate[key] - reference[key]
        output[key] = float(value) if key in {"Score", "IoU", "Pd", "Fa"} else int(value)
    return output


def component_feature_vector(
    indices,
    locations4,
    polarities,
    base_raw,
    pyramid_raw,
    base_post,
    pyramid_post,
):
    indices = np.asarray(indices, dtype=np.int64)
    xyz = locations4[indices, 1:4].astype(np.float64, copy=False)
    bins = np.floor_divide(xyz[:, 2].astype(np.int64), 50)
    unique_bins = np.unique(bins)
    centroids = np.stack([xyz[bins == value, :2].mean(axis=0) for value in unique_bins])
    if centroids.shape[0] > 1:
        steps = np.sqrt(np.sum(np.diff(centroids, axis=0) ** 2, axis=1))
        displacement = float(np.linalg.norm(centroids[-1] - centroids[0]))
        path_length = float(steps.sum())
        mean_step = float(steps.mean())
        max_step = float(steps.max())
        straightness = displacement / max(path_length, np.finfo(np.float64).eps)
    else:
        mean_step = max_step = straightness = 0.0
    delta_raw = pyramid_raw[indices].astype(np.float64) - base_raw[indices].astype(np.float64)
    lost = (base_post[indices] >= np.float32(THRESHOLD)) & (
        pyramid_post[indices] < np.float32(THRESHOLD)
    )

    def moments(values):
        values = np.asarray(values, dtype=np.float64)
        return (values.mean(), values.std(), values.min(), values.max())

    features = [
        np.log1p(indices.size),
        float(unique_bins.size),
        float(unique_bins[-1] - unique_bins[0] + 1),
        float(np.ptp(xyz[:, 0]) + 1.0),
        float(np.ptp(xyz[:, 1]) + 1.0),
        float(xyz[:, 0].mean() / 346.0),
        float(xyz[:, 1].mean() / 260.0),
        float(xyz[:, 2].mean() / 8000.0),
        mean_step / 346.0,
        max_step / 346.0,
        straightness,
        float(np.mean(polarities[indices] > 0.5)),
        float(np.mean(lost)),
    ]
    for values in (base_raw[indices], pyramid_raw[indices], delta_raw, base_post[indices], pyramid_post[indices]):
        features.extend(moments(values))
    return np.asarray(features, dtype=np.float64)


def grouped_logistic_auc(features, labels, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    groups = np.asarray(groups)
    predictions = np.full(labels.size, np.nan, dtype=np.float64)
    fold_records = []
    for held_group in np.unique(groups):
        held = groups == held_group
        fit = ~held
        if np.unique(labels[fit]).size != 2 or np.unique(labels[held]).size != 2:
            fold_records.append(
                {
                    "held_group": str(held_group),
                    "auc": None,
                    "reason": "one_class_in_fit_or_held",
                    "held_component_count": int(np.sum(held)),
                }
            )
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=73,
            ),
        )
        model.fit(features[fit], labels[fit])
        predictions[held] = model.predict_proba(features[held])[:, 1]
        fold_records.append(
            {
                "held_group": str(held_group),
                "auc": float(roc_auc_score(labels[held], predictions[held])),
                "held_component_count": int(np.sum(held)),
                "held_target_component_count": int(np.sum(labels[held])),
            }
        )
    valid = np.isfinite(predictions)
    pooled_auc = (
        float(roc_auc_score(labels[valid], predictions[valid]))
        if np.any(valid) and np.unique(labels[valid]).size == 2
        else None
    )
    return {
        "method": "fixed_C1_balanced_logistic_leave_one_G3_source_out_diagnostic_only",
        "feature_count": int(features.shape[1]),
        "component_count": int(features.shape[0]),
        "target_component_count": int(labels.sum()),
        "folds": fold_records,
        "pooled_auc": pooled_auc,
        "cutoff_selected": False,
        "formal_evidence": False,
    }


def run():
    if OUTPUT_PATH.exists() or OUTPUT_PATH.parent.exists():
        raise FileExistsError("refusing to overwrite V2 CPU disagreement capacity")
    if sha256_file(PAIRED_PATH) != EXPECTED_PAIRED_SHA256:
        raise RuntimeError("frozen Stage1 paired evaluation changed")
    if sha256_file(SOURCE_PROTOCOL_PATH) != EXPECTED_SOURCE_PROTOCOL_SHA256:
        raise RuntimeError("frozen H2 source manifest changed")
    paired = read_json(PAIRED_PATH)
    if paired.get("promoted") is not False:
        raise RuntimeError("Stage1 branch is not archived")
    if paired.get("validation_or_test_read") is not False:
        raise RuntimeError("Stage1 evidence is contaminated")
    manifest = read_json(SOURCE_PROTOCOL_PATH)["h2_sources"]
    records_by_name = {value["source_name"]: value for value in paired["records"]}
    pooled_base = crossfit.SufficientCounts()
    pooled_stage1 = crossfit.SufficientCounts()
    pooled_oracle = crossfit.SufficientCounts()
    source_records = []
    all_features = []
    all_labels = []
    all_groups = []

    for source_name in G3_SOURCES:
        record = records_by_name[source_name]
        artifact_path = Path(record["score_artifact_path"])
        if sha256_file(artifact_path) != record["score_artifact_sha256"]:
            raise RuntimeError("paired score artifact changed: {}".format(source_name))
        with np.load(artifact_path, allow_pickle=False) as archive:
            forbidden = {"labels", "target_ids", "source_index"} & set(archive.files)
            if forbidden:
                raise RuntimeError("paired score artifact contains forbidden truth/provenance")
            locations4 = np.asarray(archive["locations4"], dtype=np.int64)
            base_raw = np.asarray(archive["base_raw_scores"], dtype=np.float32)
            pyramid_raw = np.asarray(archive["candidate_raw_scores"], dtype=np.float32)
            base_post = np.asarray(archive["base_post_C00_scores"], dtype=np.float32)
            pyramid_post = np.asarray(archive["candidate_post_C00_scores"], dtype=np.float32)
        source_path = (TRAIN_ROOT / source_name).resolve()
        if source_path.parent != TRAIN_ROOT:
            raise RuntimeError("G3 source escaped official train root")
        if sha256_file(source_path) != manifest[source_name]["sha256"]:
            raise RuntimeError("G3 fit/development source changed")
        with np.load(source_path, allow_pickle=False) as archive:
            events = np.asarray(archive["evs_norm"])
            source_locations3 = np.asarray(archive["ev_loc"], dtype=np.int64)
        labels = events[:, 4].astype(np.uint8, copy=True)
        target_ids = events[:, 5].astype(np.int64, copy=True)
        polarities = events[:, 3].astype(np.float32, copy=True)
        expected_locations4 = np.column_stack(
            (np.zeros(len(events), dtype=np.int64), source_locations3)
        )
        if not np.array_equal(locations4, expected_locations4):
            raise RuntimeError("G3 score artifact/source event alignment changed")
        base_counts = crossfit.sufficient_counts_for_video(
            base_post, labels, target_ids, locations4, THRESHOLD
        )
        stage1_counts = crossfit.sufficient_counts_for_video(
            pyramid_post, labels, target_ids, locations4, THRESHOLD
        )
        if base_counts.to_dict() != record["base_counts"]:
            raise RuntimeError("saved M20 metrics do not reproduce")
        if stage1_counts.to_dict() != record["candidate_counts"]:
            raise RuntimeError("saved pyramid metrics do not reproduce")

        components = extract_atomic_components(
            base_post,
            locations4,
            THRESHOLD,
            spatial_radius=2,
            temporal_bin_size=50,
            temporal_radius_bins=1,
        ).event_indices
        affected = []
        target_bearing = []
        pure_fp = []
        lost_target_events = 0
        lost_false_events = 0
        lost_target_ids = set()
        for component_index, indices in enumerate(components):
            lost = pyramid_post[indices] < np.float32(THRESHOLD)
            if not np.any(lost):
                continue
            affected.append(component_index)
            positives = labels[indices] > 0
            if np.any(positives):
                target_bearing.append(component_index)
            else:
                pure_fp.append(component_index)
            lost_indices = indices[lost]
            lost_positive = labels[lost_indices] > 0
            lost_target_events += int(np.sum(lost_positive))
            lost_false_events += int(np.sum(~lost_positive))
            lost_target_ids.update(
                int(value) for value in target_ids[lost_indices[lost_positive]] if int(value) > 0
            )
            all_features.append(
                component_feature_vector(
                    indices,
                    locations4,
                    polarities,
                    base_raw,
                    pyramid_raw,
                    base_post,
                    pyramid_post,
                )
            )
            all_labels.append(int(np.any(positives)))
            all_groups.append(source_name)

        affected_components = tuple(components[index] for index in affected)
        target_set = set(target_bearing)
        oracle_decisions = np.asarray(
            [component_index in target_set for component_index in affected],
            dtype=np.bool_,
        )
        oracle_scores = restore_whole_components_bitwise(
            pyramid_post,
            base_post,
            affected_components,
            oracle_decisions,
        )
        oracle_counts = crossfit.sufficient_counts_for_video(
            oracle_scores, labels, target_ids, locations4, THRESHOLD
        )
        base_report = challenge_report(base_counts)
        stage1_report = challenge_report(stage1_counts)
        oracle_report = challenge_report(oracle_counts)
        source_records.append(
            {
                "source_name": source_name,
                "M20_component_count": len(components),
                "threshold_disagreement_component_count": len(affected),
                "target_bearing_disagreement_component_count": len(target_bearing),
                "pure_FP_disagreement_component_count": len(pure_fp),
                "lost_target_event_count": lost_target_events,
                "lost_false_positive_event_count": lost_false_events,
                "lost_target_id_count": len(lost_target_ids),
                "M20": base_report,
                "Stage1_pyramid": stage1_report,
                "whole_component_truth_oracle": oracle_report,
                "Stage1_delta_vs_M20": report_delta(base_report, stage1_report),
                "oracle_delta_vs_M20": report_delta(base_report, oracle_report),
                "oracle_recovery_vs_Stage1": report_delta(stage1_report, oracle_report),
                "oracle_action": (
                    "restore_every_target-bearing_disagreement_M20_component_bitwise; "
                    "keep_every_pure-FP_disagreement_component_at_pyramid"
                ),
            }
        )
        pooled_base = pooled_base + base_counts
        pooled_stage1 = pooled_stage1 + stage1_counts
        pooled_oracle = pooled_oracle + oracle_counts

    pooled_base_report = challenge_report(pooled_base)
    pooled_stage1_report = challenge_report(pooled_stage1)
    pooled_oracle_report = challenge_report(pooled_oracle)
    separability = grouped_logistic_auc(all_features, all_labels, all_groups)
    pooled_capacity = {
        "M20": pooled_base_report,
        "Stage1_pyramid": pooled_stage1_report,
        "whole_component_truth_oracle": pooled_oracle_report,
        "Stage1_delta_vs_M20": report_delta(pooled_base_report, pooled_stage1_report),
        "oracle_delta_vs_M20": report_delta(pooled_base_report, pooled_oracle_report),
        "oracle_recovery_vs_Stage1": report_delta(pooled_stage1_report, pooled_oracle_report),
    }
    capacity_gate = {
        "oracle_score_gain_vs_M20_at_least_0_02": (
            pooled_oracle_report["Score"] - pooled_base_report["Score"] >= 0.02
        ),
        "oracle_TP_not_below_Stage1": pooled_oracle_report["TP"] >= pooled_stage1_report["TP"],
        "oracle_CO_not_below_Stage1": pooled_oracle_report["CO"] >= pooled_stage1_report["CO"],
        "oracle_strictly_recovers_TP_or_CO": (
            pooled_oracle_report["TP"] > pooled_stage1_report["TP"]
            or pooled_oracle_report["CO"] > pooled_stage1_report["CO"]
        ),
    }
    payload = {
        "schema": "ev-uav-h2-pyramid-whole-component-recovery-v2-cpu-capacity-v1",
        "created_utc": utc_now(),
        "evidence_class": "G3_development_only_not_fresh_outer_held_evidence",
        "Stage1_paired_evaluation_path": str(PAIRED_PATH.resolve()),
        "Stage1_paired_evaluation_sha256": EXPECTED_PAIRED_SHA256,
        "source_manifest_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "sources_read": list(G3_SOURCES),
        "G1_arrays_or_predictions_read": False,
        "G2_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
        "GPU_used": False,
        "prediction_threshold_changed": False,
        "component_topology": {
            "owner": "M20_post_C00_positive_components",
            "spatial_radius": 2,
            "temporal_bin_size": 50,
            "temporal_radius_bins": 1,
        },
        "source_records": source_records,
        "pooled_capacity": pooled_capacity,
        "diagnostic_component_separability": separability,
        "capacity_gate": capacity_gate,
        "capacity_passed": all(capacity_gate.values()),
    }
    digest = write_json_exclusive(OUTPUT_PATH, payload)
    print(
        json.dumps(
            {
                "report": str(OUTPUT_PATH.resolve()),
                "report_sha256": digest,
                "pooled_capacity": pooled_capacity,
                "separability": separability,
                "capacity_gate": capacity_gate,
                "capacity_passed": payload["capacity_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()

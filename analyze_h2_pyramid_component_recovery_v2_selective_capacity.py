"""Exact single/CO-critical whole-component recovery capacity on G3 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_h2_pyramid_component_recovery_v2_capacity as base
import crossfit_component_reranker as crossfit
from utils.atomic_component_deletion import extract_atomic_components
from utils.h2_pyramid_component_recovery import restore_whole_components_bitwise


OUTPUT_PATH = base.OUTPUT_PATH.with_name("selective_report.json")
EXPECTED_PARENT_REPORT_SHA256 = "b2e77504eacd64780ed23b9bad7cfba3762245bd4a390baba9a73863e278237c"


def counts_with_delta(reference, changed, original):
    values = {}
    for name in reference.__dataclass_fields__:
        values[name] = int(
            getattr(reference, name) + getattr(changed, name) - getattr(original, name)
        )
    return crossfit.SufficientCounts(**values)


def evaluate_source(scores, record):
    return crossfit.sufficient_counts_for_video(
        scores,
        record["labels"],
        record["target_ids"],
        record["locations4"],
        base.THRESHOLD,
    )


def load_records():
    if base.sha256_file(base.OUTPUT_PATH) != EXPECTED_PARENT_REPORT_SHA256:
        raise RuntimeError("parent capacity receipt changed")
    paired = base.read_json(base.PAIRED_PATH)
    manifest = base.read_json(base.SOURCE_PROTOCOL_PATH)["h2_sources"]
    records_by_name = {value["source_name"]: value for value in paired["records"]}
    output = {}
    for source_name in base.G3_SOURCES:
        metric_record = records_by_name[source_name]
        artifact_path = Path(metric_record["score_artifact_path"])
        if base.sha256_file(artifact_path) != metric_record["score_artifact_sha256"]:
            raise RuntimeError("paired score artifact changed")
        with np.load(artifact_path, allow_pickle=False) as archive:
            locations4 = np.asarray(archive["locations4"], dtype=np.int64)
            base_post = np.asarray(archive["base_post_C00_scores"], dtype=np.float32)
            pyramid_post = np.asarray(archive["candidate_post_C00_scores"], dtype=np.float32)
        source_path = base.TRAIN_ROOT / source_name
        if base.sha256_file(source_path) != manifest[source_name]["sha256"]:
            raise RuntimeError("G3 source changed")
        with np.load(source_path, allow_pickle=False) as archive:
            events = np.asarray(archive["evs_norm"])
        labels = events[:, 4].astype(np.uint8, copy=True)
        target_ids = events[:, 5].astype(np.int64, copy=True)
        components = extract_atomic_components(
            base_post,
            locations4,
            base.THRESHOLD,
            spatial_radius=2,
            temporal_bin_size=50,
            temporal_radius_bins=1,
        ).event_indices
        affected = []
        for component_index, indices in enumerate(components):
            if np.any(pyramid_post[indices] < np.float32(base.THRESHOLD)):
                affected.append(component_index)
        affected_components = tuple(components[index] for index in affected)
        stage1_counts = evaluate_source(
            pyramid_post,
            {
                "labels": labels,
                "target_ids": target_ids,
                "locations4": locations4,
            },
        )
        m20_counts = evaluate_source(
            base_post,
            {
                "labels": labels,
                "target_ids": target_ids,
                "locations4": locations4,
            },
        )
        output[source_name] = {
            "locations4": locations4,
            "labels": labels,
            "target_ids": target_ids,
            "base_post": base_post,
            "pyramid_post": pyramid_post,
            "components": affected_components,
            "component_ids": affected,
            "stage1_counts": stage1_counts,
            "m20_counts": m20_counts,
        }
    return output


def pooled_counts(records, field):
    result = crossfit.SufficientCounts()
    for value in records.values():
        result = result + value[field]
    return result


def run():
    if OUTPUT_PATH.exists():
        raise FileExistsError("refusing to overwrite selective capacity report")
    records = load_records()
    pooled_m20 = pooled_counts(records, "m20_counts")
    pooled_stage1 = pooled_counts(records, "stage1_counts")
    m20_report = base.challenge_report(pooled_m20)
    stage1_report = base.challenge_report(pooled_stage1)
    component_records = []
    co_critical = {name: [] for name in records}

    for source_name, record in records.items():
        for local_index, (original_id, indices) in enumerate(
            zip(record["component_ids"], record["components"])
        ):
            positives = record["labels"][indices] > 0
            if not np.any(positives):
                continue
            decisions = np.zeros(len(record["components"]), dtype=np.bool_)
            decisions[local_index] = True
            scores = restore_whole_components_bitwise(
                record["pyramid_post"],
                record["base_post"],
                record["components"],
                decisions,
            )
            changed = evaluate_source(scores, record)
            pooled = counts_with_delta(
                pooled_stage1, changed, record["stage1_counts"]
            )
            changed_report = base.challenge_report(changed)
            pooled_report = base.challenge_report(pooled)
            source_stage1_report = base.challenge_report(record["stage1_counts"])
            lost = record["pyramid_post"][indices] < np.float32(base.THRESHOLD)
            lost_indices = indices[lost]
            entry = {
                "source_name_for_audit_only_not_feature": source_name,
                "component_local_id_for_audit_only_not_feature": int(original_id),
                "component_event_count": int(len(indices)),
                "lost_event_count": int(np.sum(lost)),
                "lost_target_event_count": int(
                    np.sum(record["labels"][lost_indices] > 0)
                ),
                "lost_false_event_count": int(
                    np.sum(record["labels"][lost_indices] == 0)
                ),
                "source_recovery_vs_Stage1": base.report_delta(
                    source_stage1_report, changed_report
                ),
                "pooled_after_single_restore": pooled_report,
                "pooled_delta_vs_M20": base.report_delta(m20_report, pooled_report),
                "pooled_delta_vs_Stage1": base.report_delta(stage1_report, pooled_report),
                "pooled_score_floor_0_02_passed": (
                    pooled_report["Score"] - m20_report["Score"] >= 0.02
                ),
            }
            component_records.append(entry)
            if changed.correct_objects > record["stage1_counts"].correct_objects:
                co_critical[source_name].append(local_index)

    feasible_single = [
        value
        for value in component_records
        if value["pooled_score_floor_0_02_passed"]
        and (
            value["pooled_delta_vs_Stage1"]["TP"] > 0
            or value["pooled_delta_vs_Stage1"]["CO"] > 0
        )
        and value["pooled_delta_vs_Stage1"]["TP"] >= 0
        and value["pooled_delta_vs_Stage1"]["CO"] >= 0
    ]
    best_single = max(
        feasible_single,
        key=lambda value: (
            value["pooled_delta_vs_Stage1"]["CO"],
            value["pooled_delta_vs_Stage1"]["TP"],
            value["pooled_after_single_restore"]["Score"],
        ),
    ) if feasible_single else None

    pooled_co_critical = crossfit.SufficientCounts()
    co_critical_source_reports = []
    for source_name, record in records.items():
        decisions = np.zeros(len(record["components"]), dtype=np.bool_)
        decisions[co_critical[source_name]] = True
        scores = restore_whole_components_bitwise(
            record["pyramid_post"],
            record["base_post"],
            record["components"],
            decisions,
        )
        counts = evaluate_source(scores, record)
        pooled_co_critical = pooled_co_critical + counts
        co_critical_source_reports.append(
            {
                "source_name": source_name,
                "selected_CO_critical_component_count": len(co_critical[source_name]),
                "Stage2": base.challenge_report(counts),
                "recovery_vs_Stage1": base.report_delta(
                    base.challenge_report(record["stage1_counts"]),
                    base.challenge_report(counts),
                ),
            }
        )
    co_critical_report = base.challenge_report(pooled_co_critical)
    capacity_gate = {
        "at_least_one_exact_single_component_is_feasible": bool(feasible_single),
        "best_single_score_gain_vs_M20_at_least_0_02": bool(
            best_single is not None
            and best_single["pooled_delta_vs_M20"]["Score"] >= 0.02
        ),
        "best_single_recovers_TP_or_CO": bool(
            best_single is not None
            and (
                best_single["pooled_delta_vs_Stage1"]["TP"] > 0
                or best_single["pooled_delta_vs_Stage1"]["CO"] > 0
            )
        ),
    }
    parent = base.read_json(base.OUTPUT_PATH)
    payload = {
        "schema": "ev-uav-h2-pyramid-selective-whole-component-recovery-v2-capacity-v1",
        "created_utc": base.utc_now(),
        "evidence_class": "G3_development_only_exact_component_actions",
        "parent_capacity_path": str(base.OUTPUT_PATH.resolve()),
        "parent_capacity_sha256": EXPECTED_PARENT_REPORT_SHA256,
        "M20": m20_report,
        "Stage1_pyramid": stage1_report,
        "target_bearing_component_count": len(component_records),
        "exact_single_component_records": component_records,
        "feasible_single_component_count": len(feasible_single),
        "best_single_risk_controlled_recovery": best_single,
        "CO_critical_union": {
            "source_records": co_critical_source_reports,
            "Stage2": co_critical_report,
            "delta_vs_M20": base.report_delta(m20_report, co_critical_report),
            "recovery_vs_Stage1": base.report_delta(stage1_report, co_critical_report),
        },
        "diagnostic_separability": parent["diagnostic_component_separability"],
        "capacity_gate": capacity_gate,
        "capacity_passed": all(capacity_gate.values()),
        "G1_arrays_or_predictions_read": False,
        "G2_arrays_or_predictions_read": False,
        "validation_or_test_read": False,
        "GPU_used": False,
    }
    digest = base.write_json_exclusive(OUTPUT_PATH, payload)
    print(
        json.dumps(
            {
                "report": str(OUTPUT_PATH.resolve()),
                "report_sha256": digest,
                "target_bearing_component_count": len(component_records),
                "feasible_single_component_count": len(feasible_single),
                "best_single": best_single,
                "CO_critical_union": payload["CO_critical_union"],
                "capacity_gate": capacity_gate,
                "capacity_passed": payload["capacity_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()

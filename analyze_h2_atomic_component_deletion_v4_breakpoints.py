"""CPU-only fit-OOF breakpoint audit for the frozen V3 component scorer.

This script never opens the fresh-G2 held sources.  It replays complete-component
deletions on the already persisted G1/G3 nested-OOF probabilities and evaluates
every distinct probability action boundary with the unchanged challenge metric.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from crossfit_component_reranker import (
    SufficientCounts,
    metrics_from_counts,
    sufficient_counts_for_video,
)
from utils.atomic_component_deletion import (
    atomic_delete_or_identity,
    pure_false_positive_targets,
)


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
EXPERIMENT_ROOT = (
    WORKSPACE_ROOT
    / "experiments"
    / "20260811_h2_atomic_component_deletion_g2_v3"
)
FORMAL_ROOT = EXPERIMENT_ROOT / "formal_training" / "hold_g2"
V3_PROTOCOL_PATH = (
    REPO_ROOT / "protocols" / "h2_atomic_component_deletion_g2_science_v3.json"
)
EXPECTED_V3_PROTOCOL_SHA256 = (
    "533ecc90b9d96dc72c5f32a45641c9918905b4d2a46392dd7670afed01e55294"
)
TRAINING_RESULT_PATH = FORMAL_ROOT / "training_result.json"
EXPECTED_TRAINING_RESULT_SHA256 = (
    "5ada4d2dbebe3d81a9462ef9e60add562c87ea7ce9a9b84742fe633625e9f88c"
)
FIT_GROUP_IDS = ("g1_088_091", "g3_095_098")
FORBIDDEN_SOURCE_NAMES = frozenset(
    ("train_092.npz", "train_093.npz", "train_094.npz")
)
POOLED_GAIN_GATE = 0.005
OUTPUT_PATH = EXPERIMENT_ROOT / "v4_cpu_breakpoint_diagnostic" / "result.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_exclusive(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sum_counts(values) -> SufficientCounts:
    total = SufficientCounts()
    for value in values:
        total = total + value
    return total


def metric_record(counts: SufficientCounts):
    return {"counts": counts.to_dict(), "metrics": metrics_from_counts(counts)}


def metric_delta(candidate: SufficientCounts, baseline: SufficientCounts):
    candidate_metrics = metrics_from_counts(candidate)
    baseline_metrics = metrics_from_counts(baseline)
    return {
        key: float(candidate_metrics[key] - baseline_metrics[key])
        for key in baseline_metrics
    }


def count_delta(candidate: SufficientCounts, baseline: SufficientCounts):
    return {
        key: int(getattr(candidate, key) - getattr(baseline, key))
        for key in baseline.__dataclass_fields__
    }


def components_from_artifact(archive) -> tuple[np.ndarray, ...]:
    offsets = np.asarray(archive["component_offsets"], dtype=np.int64)
    flattened = np.asarray(archive["component_event_indices"], dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 1 or offsets[0] != 0:
        raise RuntimeError("invalid component offsets")
    if offsets[-1] != flattened.size or np.any(np.diff(offsets) <= 0):
        raise RuntimeError("invalid component event partition")
    return tuple(
        flattened[offsets[index] : offsets[index + 1]].copy()
        for index in range(offsets.size - 1)
    )


def source_baseline_record(training_result, group_id, source_name):
    records = training_result["inner_oof_calibration"]["group_diagnostics"][
        group_id
    ]["sources"]
    matches = [record for record in records if record["source_name"] == source_name]
    if len(matches) != 1:
        raise RuntimeError("training-result baseline record is missing or duplicate")
    return matches[0]["base"]


def load_fit_source(
    source_name,
    group_id,
    protocol,
    training_result,
    manifest_path,
    manifest,
):
    if source_name in FORBIDDEN_SOURCE_NAMES:
        raise RuntimeError("fresh-G2 held source access attempted")
    fit_path = Path(
        training_result["immutable_fit_input_artifacts"][source_name]["path"]
    )
    score_path = Path(
        training_result["immutable_inner_oof_score_artifacts"][source_name]["path"]
    )
    for path, expected in (
        (
            fit_path,
            training_result["immutable_fit_input_artifacts"][source_name]["sha256"],
        ),
        (
            score_path,
            training_result["immutable_inner_oof_score_artifacts"][source_name][
                "sha256"
            ],
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError("frozen V3 artifact changed")

    manifest_matches = [
        record for record in manifest["records"] if record["source_name"] == source_name
    ]
    if len(manifest_matches) != 1:
        raise RuntimeError("fit cache record is missing or duplicate")
    cache_metadata = manifest_matches[0]
    cache_path = manifest_path.parent / cache_metadata["record"]
    if not cache_path.is_file() or sha256_file(cache_path) != cache_metadata["record_sha256"]:
        raise RuntimeError("fit cache record changed")

    with np.load(fit_path, allow_pickle=False) as fit_archive:
        if str(fit_archive["artifact_schema"]) != "ev-uav-h2-atomic-component-input-artifact-v3":
            raise RuntimeError("unexpected fit-input artifact schema")
        base_scores = np.asarray(fit_archive["c00_post_scores"], dtype=np.float32).copy()
        reference_scores = np.asarray(
            fit_archive["rich_cache_reference_scores"], dtype=np.float32
        ).copy()
        locations = np.asarray(fit_archive["locations"], dtype=np.int64).copy()
        event_component_ids = np.asarray(
            fit_archive["event_component_ids"], dtype=np.int32
        ).copy()
        components = components_from_artifact(fit_archive)

    with np.load(score_path, allow_pickle=False) as score_archive:
        if str(score_archive["artifact_schema"]) != "ev-uav-h2-atomic-component-score-artifact-v3":
            raise RuntimeError("unexpected OOF-score artifact schema")
        probabilities = np.asarray(
            score_archive["consensus_pure_fp_probability"], dtype=np.float64
        ).copy()
        persisted_targets = np.asarray(
            score_archive["fit_only_pure_fp_targets"], dtype=np.uint8
        ).copy()
        model_group_ids = [str(value) for value in score_archive["model_group_ids"]]

    with np.load(cache_path, allow_pickle=False) as cache_archive:
        cache_scores = np.asarray(cache_archive["scores"], dtype=np.float32).copy()
        cache_locations = np.asarray(cache_archive["locs"], dtype=np.int64).copy()
        labels = np.asarray(cache_archive["labels"], dtype=np.uint8).copy()
        target_ids = np.asarray(cache_archive["target_ids"]).copy()

    n_events = base_scores.size
    if not (
        reference_scores.size
        == cache_scores.size
        == labels.size
        == target_ids.size
        == locations.shape[0]
        == cache_locations.shape[0]
        == n_events
    ):
        raise RuntimeError("fit-only record lengths differ")
    if locations.shape[1] != 4 or cache_locations.shape[1] != 3:
        raise RuntimeError("fit-only location schema differs")
    if not np.array_equal(locations[:, 1:4], cache_locations):
        raise RuntimeError("fit-input and cache locations differ")
    if not np.array_equal(reference_scores, cache_scores):
        raise RuntimeError("fit-input and cache reference scores differ")
    if probabilities.size != len(components):
        raise RuntimeError("component probability count differs")
    recomputed_targets = pure_false_positive_targets(components, labels)
    if not np.array_equal(recomputed_targets, persisted_targets):
        raise RuntimeError("persisted fit-only component classes differ")
    expected_oof_model = FIT_GROUP_IDS[1] if group_id == FIT_GROUP_IDS[0] else FIT_GROUP_IDS[0]
    if model_group_ids != [expected_oof_model]:
        raise RuntimeError("nested grouped-OOF scorer provenance differs")

    reconstructed_ids = np.full(n_events, -1, dtype=np.int32)
    for component_id, indices in enumerate(components):
        if np.any(reconstructed_ids[indices] != -1):
            raise RuntimeError("component event indices overlap")
        reconstructed_ids[indices] = component_id
    if not np.array_equal(reconstructed_ids, event_component_ids):
        raise RuntimeError("event/component map differs")

    threshold = float(protocol["baseline"]["prediction_threshold"])
    baseline = sufficient_counts_for_video(
        base_scores, labels, target_ids, locations, threshold
    )
    expected_baseline = source_baseline_record(training_result, group_id, source_name)
    if baseline.to_dict() != expected_baseline["counts"]:
        raise RuntimeError("official baseline counts do not reproduce")
    reproduced_metrics = metrics_from_counts(baseline)
    for key, value in expected_baseline["metrics"].items():
        if abs(reproduced_metrics[key] - float(value)) > 1e-15:
            raise RuntimeError("official baseline metrics do not reproduce")

    return {
        "source_name": source_name,
        "group_id": group_id,
        "base_scores": base_scores,
        "locations": locations,
        "labels": labels,
        "target_ids": target_ids,
        "components": components,
        "probabilities": probabilities,
        "pure_fp_targets": persisted_targets,
        "prediction_threshold": threshold,
        "baseline": baseline,
        "fit_input_sha256": sha256_file(fit_path),
        "oof_score_sha256": sha256_file(score_path),
        "cache_record_sha256": sha256_file(cache_path),
        "state_cache": {},
    }


def evaluate_source_at_cutoff(source, cutoff):
    deleted = source["probabilities"] >= float(cutoff)
    signature = deleted.tobytes()
    cached = source["state_cache"].get(signature)
    if cached is not None:
        return cached
    candidate_scores, receipt = atomic_delete_or_identity(
        source["base_scores"],
        source["components"],
        source["probabilities"],
        float(cutoff),
        enabled=True,
    )
    if not (
        receipt.enabled
        and receipt.complete_components_only
        and receipt.retained_scores_bitwise_equal
        and receipt.deleted_component_count == int(np.count_nonzero(deleted))
    ):
        raise RuntimeError("atomic deletion integrity failed")
    counts = sufficient_counts_for_video(
        candidate_scores,
        source["labels"],
        source["target_ids"],
        source["locations"],
        source["prediction_threshold"],
    )
    cached = {"counts": counts, "receipt": asdict(receipt)}
    source["state_cache"][signature] = cached
    return cached


def diagnostic_record(counts, baseline):
    return {
        "baseline": metric_record(baseline),
        "candidate": metric_record(counts),
        "metric_delta": metric_delta(counts, baseline),
        "count_delta": count_delta(counts, baseline),
    }


def main():
    if sha256_file(V3_PROTOCOL_PATH) != EXPECTED_V3_PROTOCOL_SHA256:
        raise RuntimeError("frozen V3 science protocol changed")
    if sha256_file(TRAINING_RESULT_PATH) != EXPECTED_TRAINING_RESULT_SHA256:
        raise RuntimeError("frozen V3 training result changed")
    protocol = load_json(V3_PROTOCOL_PATH)
    training_result = load_json(TRAINING_RESULT_PATH)
    if (
        training_result["g2_held_array_read"]
        or training_result["validation_or_test_read"]
        or training_result["deletion_enabled"]
    ):
        raise RuntimeError("V4 expects the stopped identity V3 fit result")

    group_sources = {
        group_id: tuple(protocol["source_groups"][group_id]["sources"])
        for group_id in FIT_GROUP_IDS
    }
    allowed_names = frozenset(name for names in group_sources.values() for name in names)
    if allowed_names & FORBIDDEN_SOURCE_NAMES or len(allowed_names) != 8:
        raise RuntimeError("fit-only source allowlist differs")

    manifest_path = (
        WORKSPACE_ROOT
        / protocol["rich_m20_cache"]["manifest_workspace_relative_path"]
    )
    if sha256_file(manifest_path) != protocol["rich_m20_cache"]["manifest_sha256"]:
        raise RuntimeError("rich-cache manifest changed")
    manifest = load_json(manifest_path)

    sources = []
    for group_id in FIT_GROUP_IDS:
        for source_name in group_sources[group_id]:
            sources.append(
                load_fit_source(
                    source_name,
                    group_id,
                    protocol,
                    training_result,
                    manifest_path,
                    manifest,
                )
            )

    all_probabilities = np.concatenate([source["probabilities"] for source in sources])
    if not np.isfinite(all_probabilities).all() or all_probabilities.size != 153:
        raise RuntimeError("frozen inner-OOF probability population differs")
    breakpoints = np.unique(all_probabilities)[::-1]
    tied_probability_count = int(all_probabilities.size - breakpoints.size)

    group_baselines = {
        group_id: sum_counts(
            source["baseline"] for source in sources if source["group_id"] == group_id
        )
        for group_id in FIT_GROUP_IDS
    }
    pooled_baseline = sum_counts(source["baseline"] for source in sources)

    rows = []
    identity_cutoff = float(np.nextafter(np.max(all_probabilities), np.inf))
    candidate_cutoffs = np.concatenate(
        (np.asarray([identity_cutoff], dtype=np.float64), breakpoints)
    )
    for index, cutoff in enumerate(candidate_cutoffs):
        state_by_source = {
            source["source_name"]: evaluate_source_at_cutoff(source, float(cutoff))
            for source in sources
        }
        group_candidates = {
            group_id: sum_counts(
                state_by_source[source["source_name"]]["counts"]
                for source in sources
                if source["group_id"] == group_id
            )
            for group_id in FIT_GROUP_IDS
        }
        pooled_candidate = sum_counts(group_candidates.values())
        group_deltas = {
            group_id: metric_delta(group_candidates[group_id], group_baselines[group_id])[
                "score"
            ]
            for group_id in FIT_GROUP_IDS
        }
        pooled_delta = metric_delta(pooled_candidate, pooled_baseline)["score"]
        deleted_components = {
            source["source_name"]: state_by_source[source["source_name"]]["receipt"][
                "deleted_component_count"
            ]
            for source in sources
        }
        deleted_events = {
            source["source_name"]: state_by_source[source["source_name"]]["receipt"][
                "deleted_event_count"
            ]
            for source in sources
        }
        both_positive = all(value > 0.0 for value in group_deltas.values())
        rows.append(
            {
                "breakpoint_index": index,
                "cutoff": float(cutoff),
                "cutoff_hex": float(cutoff).hex(),
                "cutoff_source": (
                    "identity_nextafter_global_max"
                    if index == 0
                    else "unique_frozen_inner_oof_probability"
                ),
                "deleted_component_count": int(sum(deleted_components.values())),
                "deleted_event_count": int(sum(deleted_events.values())),
                "deleted_components_by_source": deleted_components,
                "deleted_events_by_source": deleted_events,
                "group_score_delta": group_deltas,
                "minimum_group_score_delta": float(min(group_deltas.values())),
                "pooled_score_delta": float(pooled_delta),
                "both_groups_score_strictly_positive": bool(both_positive),
                "pooled_gain_gate_passed": bool(pooled_delta >= POOLED_GAIN_GATE),
                "groups": {
                    group_id: diagnostic_record(
                        group_candidates[group_id], group_baselines[group_id]
                    )
                    for group_id in FIT_GROUP_IDS
                },
                "pooled": diagnostic_record(pooled_candidate, pooled_baseline),
            }
        )

    positive_rows = [row for row in rows if row["both_groups_score_strictly_positive"]]
    pareto_rows = []
    for row in positive_rows:
        dominated = any(
            (
                other["minimum_group_score_delta"]
                >= row["minimum_group_score_delta"]
                and other["pooled_score_delta"] >= row["pooled_score_delta"]
                and (
                    other["minimum_group_score_delta"]
                    > row["minimum_group_score_delta"]
                    or other["pooled_score_delta"] > row["pooled_score_delta"]
                )
            )
            for other in positive_rows
        )
        if not dominated:
            pareto_rows.append(row)
    eligible = [
        row for row in pareto_rows if row["pooled_score_delta"] >= POOLED_GAIN_GATE
    ]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["minimum_group_score_delta"],
                row["pooled_score_delta"],
                row["cutoff"],
            ),
        )
        if eligible
        else None
    )

    compact_selected = None
    if selected is not None:
        compact_selected = {
            key: value
            for key, value in selected.items()
            if key not in ("groups", "pooled")
        }
        compact_selected["groups"] = selected["groups"]
        compact_selected["pooled"] = selected["pooled"]

    payload = {
        "schema": "ev-uav-h2-atomic-component-v4-fit-oof-breakpoint-diagnostic-v1",
        "compute": "CPU-only exact official metric replay",
        "held_g2_array_read": False,
        "validation_or_test_read": False,
        "model_retrained": False,
        "model_or_probability_modified": False,
        "v3_protocol_path": str(V3_PROTOCOL_PATH),
        "v3_protocol_sha256": sha256_file(V3_PROTOCOL_PATH),
        "v3_training_result_path": str(TRAINING_RESULT_PATH),
        "v3_training_result_sha256": sha256_file(TRAINING_RESULT_PATH),
        "rich_cache_manifest_path": str(manifest_path),
        "rich_cache_manifest_sha256": sha256_file(manifest_path),
        "fit_groups": list(FIT_GROUP_IDS),
        "fit_sources": [source["source_name"] for source in sources],
        "fit_source_artifacts": {
            source["source_name"]: {
                "fit_input_sha256": source["fit_input_sha256"],
                "oof_score_sha256": source["oof_score_sha256"],
                "cache_record_sha256": source["cache_record_sha256"],
                "component_count": len(source["components"]),
                "target_bearing_component_count": int(
                    np.count_nonzero(source["pure_fp_targets"] == 0)
                ),
                "pure_fp_component_count": int(
                    np.count_nonzero(source["pure_fp_targets"] == 1)
                ),
            }
            for source in sources
        },
        "breakpoint_definition": (
            "identity nextafter(global maximum,+inf), then every distinct frozen "
            "inner-OOF component probability with delete iff probability>=cutoff"
        ),
        "unique_probability_breakpoint_count": int(breakpoints.size),
        "component_probability_count": int(all_probabilities.size),
        "tied_probability_count": tied_probability_count,
        "tie_policy": "all components at one identical float64 probability act together",
        "evaluated_action_count": len(rows),
        "selection_rule": {
            "required": (
                "both complete fit groups have Score strictly above their own M20 "
                "baseline and pooled Score gain is at least 0.005"
            ),
            "pareto_axes": ["minimum_group_score_delta", "pooled_score_delta"],
            "deterministic_order": (
                "maximize minimum group Score delta, then pooled Score delta, "
                "then choose the higher/more-conservative cutoff"
            ),
            "numeric_grid_or_manual_cutoff": False,
        },
        "baseline": {
            "groups": {
                group_id: metric_record(group_baselines[group_id])
                for group_id in FIT_GROUP_IDS
            },
            "pooled": metric_record(pooled_baseline),
        },
        "both_positive_breakpoint_count": len(positive_rows),
        "pareto_breakpoint_indices": [row["breakpoint_index"] for row in pareto_rows],
        "eligible_breakpoint_count": len(eligible),
        "selected": compact_selected,
        "promotion_candidate_exists": selected is not None,
        "eliminate_frozen_atomic_network": selected is None,
        "all_breakpoints": rows,
    }
    write_json_exclusive(OUTPUT_PATH, payload)
    print(
        json.dumps(
            {
                "output_path": str(OUTPUT_PATH),
                "output_sha256": sha256_file(OUTPUT_PATH),
                "unique_probability_breakpoint_count": int(breakpoints.size),
                "both_positive_breakpoint_count": len(positive_rows),
                "pareto_breakpoint_indices": payload["pareto_breakpoint_indices"],
                "eligible_breakpoint_count": len(eligible),
                "promotion_candidate_exists": selected is not None,
                "selected": compact_selected,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

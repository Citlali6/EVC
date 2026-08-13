"""One-shot frozen Val24 replay of the combined candidate route.

Candidate (all parameters frozen from train-only evidence):
  low    (event_count <= 30000):                M10 @ 0.718, unchanged
  middle (30000 < count <= 200000):             M20 @ 0.719, unchanged
  h1     (count > 200000, polarity_min < 0.20): M20 @ 0.740 + band delete
  h2     (count > 200000, polarity_min >= 0.20):M20 @ 0.719 + band delete
  band delete: complete final-score components with
    4 <= bbox_diagonal <= 14 px AND duration_bins <= 6 AND events <= 12

Routing uses only observable event_count and polarity minority fraction
(pre-registered criteria from the frozen T32 protocol).  The baseline identity
replay must reproduce the golden counts exactly before the candidate is
accepted.  Exactly one execution; results are written to a fresh report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import run_temporal_memory_input_route_train as routed
from utils.atomic_component_deletion import extract_atomic_components
from utils.challenge_eval import challenge_score
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


ROOT = Path(__file__).resolve().parent
GOLDEN_REPORT = ROOT.parent / "results" / "submission_m20_golden" / "offline_score_report.json"
M10_CACHE = ROOT.parent / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"
M20_CACHE = ROOT.parent / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
VAL_ROOT = ROOT.parent / "datasets" / "EV-UAV-Challenge2" / "val"
DEFAULT_OUT = ROOT.parent / "experiments" / "20260812_combined_candidate_val24" / "frozen_validation_report.json"

H1_THRESHOLD = 0.740
BAND = dict(bbox_lo=4.0, bbox_hi=14.0, dur=6, ev=12)
WIDTH, HEIGHT = 346, 260


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_counts(scores, labels, ids, locations4, thr):
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    ids = np.asarray(ids).reshape(-1)
    locations = np.asarray(locations4, dtype=np.int64)
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
    return {
        "true_positive_events": int(np.count_nonzero(predicted & positive)),
        "false_positive_events": int(np.count_nonzero(predicted & ~positive)),
        "false_negative_events": int(np.count_nonzero(~predicted & positive)),
        "correct_target_frames": int(evaluator.correct_num),
        "target_frames": int(evaluator.obj_num),
        "false_components": int(evaluator.false_num),
        "frame_count": int(evaluator.frame_num),
        "event_count": int(values.size),
    }


def metrics_from_counts(c):
    positives = c["true_positive_events"] + c["false_negative_events"]
    union = positives + c["false_positive_events"]
    denominator = c["frame_count"] * WIDTH * HEIGHT
    iou = float(np.float32(c["true_positive_events"]) / np.float32(union))
    acc = float(np.float32(c["true_positive_events"]) / np.float32(positives))
    pd = c["correct_target_frames"] / c["target_frames"]
    fa = c["false_components"] / denominator
    score_fa, score = challenge_score(iou, acc, pd, fa)
    return {
        "iou": iou, "acc": acc, "pd": pd, "fa": fa,
        "score_fa": score_fa, "score": score,
    }


def sum_counts(values):
    values = list(values)
    keys = list(values[0].keys())
    return {key: int(sum(v[key] for v in values)) for key in keys}


def band_delete(final_scores, locations4, thr):
    comps = extract_atomic_components(
        final_scores, locations4, thr,
        spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
    ).event_indices
    output = final_scores.copy()
    for indices in comps:
        idx = np.asarray(indices, dtype=np.int64)
        x = locations4[idx, 1]
        y = locations4[idx, 2]
        t = locations4[idx, 3]
        bins = np.unique(t // 50)
        bbox = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
        if (BAND["bbox_lo"] <= bbox <= BAND["bbox_hi"]
                and len(bins) <= BAND["dur"]
                and idx.size <= BAND["ev"]):
            output[idx] = np.float32(0.0)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run", action="store_true", help="execute the replay")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(out_path)
    if args.preflight:
        protocol = {
            "schema": "ev-uav-combined-candidate-val24-preflight-v1",
            "h1_threshold": H1_THRESHOLD,
            "band": BAND,
            "m10_cache_sha256": sha256_file(M10_CACHE),
            "m20_cache_sha256": sha256_file(M20_CACHE),
            "golden_report_sha256": sha256_file(GOLDEN_REPORT),
            "runner_sha256": sha256_file(Path(__file__)),
            "val_reads": 0,
        }
        preflight_path = out_path.with_name("frozen_validation_preflight.json")
        preflight_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.run:
        print("use --preflight or --run", flush=True)
        return

    # ---- load caches ----
    m10_cache = torch.load(M10_CACHE, map_location="cpu", weights_only=False)
    m20_cache = torch.load(M20_CACHE, map_location="cpu", weights_only=False)
    recs10 = {r["file_name"]: r for r in m10_cache["records"]}
    recs20 = {r["file_name"]: r for r in m20_cache["records"]}

    cfg = routed.c00_config()
    videos = []
    for file_name in sorted(recs10):
        r10 = recs10[file_name]
        r20 = recs20[file_name]
        assert r10["event_count"] == r20["event_count"] == r10["scores"].numel()
        source_path = VAL_ROOT / file_name
        with np.load(source_path, allow_pickle=False) as archive:
            evs_norm = np.asarray(archive["evs_norm"])
            ev_loc = np.asarray(archive["ev_loc"], dtype=np.int64)
        assert ev_loc.shape[0] == r10["event_count"]
        polarity_min = float(min(
            np.mean(evs_norm[:, 3] > 0.5), 1.0 - np.mean(evs_norm[:, 3] > 0.5)))
        event_count = int(r10["event_count"])
        if event_count <= 30000:
            route = "low"
            scores = r10["scores"].numpy().astype(np.float32, copy=True)
            thr = 0.718
            band = False
        elif event_count <= 200000:
            route = "middle"
            scores = r20["scores"].numpy().astype(np.float32, copy=True)
            thr = 0.719
            band = False
        elif polarity_min < 0.20:
            route = "h1"
            scores = r20["scores"].numpy().astype(np.float32, copy=True)
            thr = H1_THRESHOLD
            band = True
        else:
            route = "h2"
            scores = r20["scores"].numpy().astype(np.float32, copy=True)
            thr = 0.719
            band = True
        locations4 = np.column_stack((np.zeros(event_count, dtype=np.int64), ev_loc))
        processor = ChallengePostprocessor.from_cfg(cfg, thr, event_count=event_count)
        final, _ = processor.apply(
            torch.from_numpy(scores.copy()),
            torch.from_numpy(locations4.copy()),
        )
        final = final.numpy().astype(np.float32, copy=True)
        if band:
            final = band_delete(final, locations4, thr)
        videos.append({
            "file_name": file_name,
            "route": route,
            "threshold": thr,
            "band": band,
            "labels": r10["seg_label"].numpy().astype(np.uint8, copy=True),
            "ids": np.asarray(r10["idx_label"]).copy(),
            "locations4": locations4,
            "final": final,
        })

    # identity sanity: baseline chain must reproduce golden exactly
    def chain(final_scores, v, thr):
        return official_counts(
            np.asarray(final_scores, dtype=np.float32).copy(),
            v["labels"], v["ids"], v["locations4"], thr,
        )

    golden = json.load(open(GOLDEN_REPORT, encoding="utf-8"))
    golden_metrics = golden["metrics"]
    golden_counts = golden["counts"]

    per_source = {}
    for v in videos:
        per_source[v["file_name"]] = official_counts(
            v["final"], v["labels"], v["ids"], v["locations4"], v["threshold"])

    pooled = sum_counts(per_source.values())
    metrics = metrics_from_counts(pooled)

    # identity replay at routed baseline thresholds (no candidate changes)
    identity_per = {}
    for v in videos:
        base_thr = 0.718 if v["route"] == "low" else 0.719
        if v["route"] == "low":
            scores = recs10[v["file_name"]]["scores"].numpy().astype(np.float32, copy=True)
        else:
            scores = recs20[v["file_name"]]["scores"].numpy().astype(np.float32, copy=True)
        processor = ChallengePostprocessor.from_cfg(cfg, base_thr, event_count=v["locations4"].shape[0])
        final, _ = processor.apply(
            torch.from_numpy(scores.copy()),
            torch.from_numpy(v["locations4"].copy()),
        )
        identity_per[v["file_name"]] = chain(
            final.numpy().astype(np.float32, copy=True), v, base_thr)
    identity_pooled = sum_counts(identity_per.values())
    identity_ok = (
        identity_pooled["true_positive_events"] == golden_counts["event_true_positives"]
        and identity_pooled["false_positive_events"] == golden_counts["event_false_positives"]
        and identity_pooled["false_negative_events"] == golden_counts["event_false_negatives"]
        and identity_pooled["correct_target_frames"] == golden_counts["evaluator_detected_objects"]
        and identity_pooled["target_frames"] == golden_counts["evaluator_objects"]
        and identity_pooled["false_components"] == golden_counts["evaluator_false_components"]
        and identity_pooled["frame_count"] == golden_counts["evaluator_frames"]
    )

    report = {
        "schema": "ev-uav-combined-candidate-val24-frozen-replay-v1",
        "h1_threshold": H1_THRESHOLD,
        "band": BAND,
        "routes": {v["file_name"]: v["route"] for v in videos},
        "identity_replay_matches_golden": identity_ok,
        "identity_pooled_counts": identity_pooled,
        "pooled_counts": pooled,
        "metrics": metrics,
        "golden_metrics": golden_metrics,
        "delta": {
            key: float(metrics[key] - golden_metrics[key])
            for key in ("iou", "acc", "pd", "fa", "score_fa", "score")
        },
        "inputs": {
            "m10_cache_sha256": sha256_file(M10_CACHE),
            "m20_cache_sha256": sha256_file(M20_CACHE),
            "golden_report_sha256": sha256_file(GOLDEN_REPORT),
        },
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "identity_ok": identity_ok,
        "metrics": metrics,
        "golden": golden_metrics,
        "delta": report["delta"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""CPU-only train-only evaluation of the combined candidate route.

Candidate:
  low    -> M10 @ 0.718, unchanged
  middle -> M20 @ 0.719, unchanged
  h1     -> M20 @ 0.740 + band delete (bbox 4-14 px, duration <= 6, events <= 12)
  h2     -> M20 @ 0.719 + band delete

h1/h2 routing uses only observable event_count and polarity minority fraction
(the same pre-registered criteria as the frozen T32 protocol).  Everything is
evaluated against the routed golden baseline via the official evaluator, with
per-family deltas.

No validation/test reads, no CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import evaluate_cross_model_blend as blend_mod
import run_low_domain_component_oracle as oracle
from utils.postprocess import ChallengePostprocessor


DEFAULT_OUT = blend_mod.CROSS_ROOT / "combined_candidate.json"
H1_THRESHOLD = 0.740
BAND = dict(bbox_lo=4.0, bbox_hi=14.0, dur=6, ev=12)


def make_band_mask(src, final_scores, thr):
    """Delete components of final scores with bbox/duration/event band."""
    from utils.atomic_component_deletion import extract_atomic_components
    comps = extract_atomic_components(
        final_scores, src["locations4"], thr,
        spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
    ).event_indices
    output = final_scores.copy()
    assigned = np.zeros(output.size, dtype=bool)
    for indices in comps:
        idx = np.asarray(indices, dtype=np.int64)
        if np.any(assigned[idx]):
            raise RuntimeError("overlap")
        assigned[idx] = True
        x = src["locations4"][idx, 1]
        y = src["locations4"][idx, 2]
        t = src["locations4"][idx, 3]
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
    parser.add_argument("--h1-threshold", type=float, default=H1_THRESHOLD)
    parser.add_argument("--band-delete", action="store_true", default=True)
    args = parser.parse_args()

    manifest, _ = oracle.validate_inputs()
    records = manifest["records"]
    import diagnose_low_c00_recovery_and_separability as diag
    diag.FAMILY_MAP = diag.build_family_map(records)

    from dataset.temporal_frame import load_temporal_frame_video
    train_root = oracle.TRAIN_ROOT
    cfg = blend_mod.routed.c00_config()

    sources = []
    for metadata in records:
        name = metadata["source_name"]
        cross_path = blend_mod.CROSS_ROOT / name
        if not cross_path.exists():
            continue
        with np.load(cross_path, allow_pickle=False) as archive:
            m10 = np.asarray(archive["m10_scores"], dtype=np.float32)
            m20 = np.asarray(archive["m20_scores"], dtype=np.float32)
        with np.load(train_root / name, allow_pickle=False) as archive:
            locations3 = np.asarray(archive["ev_loc"], dtype=np.int64).copy()
        video = load_temporal_frame_video(train_root / name, 50, 8000)
        sources.append({
            "name": name,
            "domain": metadata["decision"]["domain"],
            "family": diag.FAMILY_MAP.get(name, name),
            "baseline_thr": float(metadata["decision"]["prediction_threshold"]),
            "candidate_thr": float(args.h1_threshold) if metadata["decision"]["domain"] == "h1" else float(metadata["decision"]["prediction_threshold"]),
            "m10": m10,
            "m20": m20,
            "locations4": np.column_stack((np.zeros(locations3.shape[0], dtype=np.int64), locations3)),
            "labels": video.labels.astype(np.uint8, copy=True),
            "ids": video.target_ids.copy(),
            "event_count": int(metadata["event_count"]),
        })
    print("loaded {} sources".format(len(sources)), flush=True)

    def run_chain(src, model, thr, band_delete):
        scores = src["m10"] if model == "m10" else src["m20"]
        processor = ChallengePostprocessor.from_cfg(cfg, thr, event_count=src["event_count"])
        final, _ = processor.apply(
            torch.from_numpy(scores.copy()),
            torch.from_numpy(src["locations4"].copy()),
        )
        final = final.numpy().astype(np.float32, copy=True)
        if band_delete:
            final = make_band_mask(src, final, thr)
        return blend_mod.official_counts_thr(
            final, src["labels"], src["ids"], src["locations4"], thr,
        )

    def baseline_counts(src):
        model = "m10" if src["domain"] == "low" else "m20"
        return run_chain(src, model, src["baseline_thr"], False)

    def candidate_counts(src):
        model = "m10" if src["domain"] == "low" else "m20"
        band = args.band_delete and src["domain"] in ("h1", "h2")
        return run_chain(src, model, src["candidate_thr"], band)

    per_source = {}
    for src in sources:
        per_source[src["name"]] = {
            "baseline": baseline_counts(src),
            "candidate": candidate_counts(src),
        }

    base = oracle.sum_counts(v["baseline"] for v in per_source.values())
    cand = oracle.sum_counts(v["candidate"] for v in per_source.values())
    pooled_cd = oracle.count_delta(cand, base)
    pooled_md = oracle.metric_delta(cand, base)

    per_family = {}
    per_domain = {}
    for name, v in per_source.items():
        src = next(s for s in sources if s["name"] == name)
        for container, key in ((per_family, src["family"]), (per_domain, src["domain"])):
            entry = container.setdefault(key, {"base": oracle.Counts(), "cand": oracle.Counts()})
            entry["base"] += v["baseline"]
            entry["cand"] += v["candidate"]

    fam_report = {
        fam: {
            "count_delta": oracle.count_delta(v["cand"], v["base"]),
            "metric_delta": oracle.metric_delta(v["cand"], v["base"]),
        }
        for fam, v in per_family.items()
    }
    dom_report = {
        dom: {
            "count_delta": oracle.count_delta(v["cand"], v["base"]),
            "metric_delta": oracle.metric_delta(v["cand"], v["base"]),
        }
        for dom, v in per_domain.items()
    }

    gates = {
        "pooled_score_delta": pooled_md["score"],
        "pooled_score_positive": pooled_md["score"] > 0.0,
        "every_family_score_not_lower": all(
            fam_report[f]["metric_delta"]["score"] >= 0.0 for f in fam_report),
        "pooled_correct_frames_not_lower": pooled_cd["correct_target_frames"] >= 0,
        "pooled_pd_not_lower": pooled_md["pd"] >= 0.0,
        "pooled_tp_not_lower": pooled_cd["true_positive_events"] >= 0,
        "all_passed": (
            pooled_md["score"] > 0.0
            and all(fam_report[f]["metric_delta"]["score"] >= 0.0 for f in fam_report)
            and pooled_cd["correct_target_frames"] >= 0
            and pooled_md["pd"] >= 0.0
        ),
    }
    print(json.dumps({
        "pooled_delta": pooled_md,
        "count_delta": pooled_cd,
        "per_family": {k: v["metric_delta"]["score"] for k, v in fam_report.items()},
        "per_domain": {k: v["metric_delta"]["score"] for k, v in dom_report.items()},
        "gates": gates,
    }, ensure_ascii=False, indent=2), flush=True)

    payload = {
        "schema": "ev-uav-combined-candidate-train-only-v1",
        "dataset_split": "train",
        "validation_or_test_read": False,
        "h1_threshold": args.h1_threshold,
        "band_delete": args.band_delete,
        "band": BAND,
        "pooled": {"count_delta": pooled_cd, "metric_delta": pooled_md},
        "per_family": fam_report,
        "per_domain": dom_report,
        "gates": gates,
    }
    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

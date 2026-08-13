"""CPU-only train-only threshold-lowering sweep for low/middle domains.

The low domain's missed frames are mostly below-threshold targets; low FP
density makes a threshold reduction cheap.  Sweep low (M10) and middle (M20)
thresholds below the routed values with the frozen C00 chain + official
evaluator, per-family gates.  No validation reads, no CUDA.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import evaluate_cross_model_blend as blend_mod
import run_low_domain_component_oracle as oracle
from utils.postprocess import ChallengePostprocessor


DEFAULT_OUT = blend_mod.CROSS_ROOT / "threshold_lower_sweep.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--low-thresholds", default="0.700,0.705,0.710,0.715")
    parser.add_argument("--middle-thresholds", default="0.705,0.710,0.715")
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
        from dataset.temporal_frame import load_temporal_frame_video
        video = load_temporal_frame_video(train_root / name, 50, 8000)
        sources.append({
            "name": name,
            "domain": metadata["decision"]["domain"],
            "family": diag.FAMILY_MAP.get(name, name),
            "baseline_thr": float(metadata["decision"]["prediction_threshold"]),
            "m10": m10,
            "m20": m20,
            "locations4": np.column_stack((np.zeros(locations3.shape[0], dtype=np.int64), locations3)),
            "labels": video.labels.astype(np.uint8, copy=True),
            "ids": video.target_ids.copy(),
            "event_count": int(metadata["event_count"]),
        })
    print("loaded {} sources".format(len(sources)), flush=True)

    def counts_for(src, thr):
        scores = src["m10"] if src["domain"] == "low" else src["m20"]
        processor = ChallengePostprocessor.from_cfg(cfg, thr, event_count=src["event_count"])
        final, _ = processor.apply(
            torch.from_numpy(scores.copy()),
            torch.from_numpy(src["locations4"].copy()),
        )
        return blend_mod.official_counts_thr(
            final.numpy().astype(np.float32, copy=True),
            src["labels"], src["ids"], src["locations4"], thr,
        )

    def baseline_counts(src):
        return counts_for(src, src["baseline_thr"])

    results = {}
    configs = []
    for domain, thr in itertools.chain(
        [("low", float(t)) for t in args.low_thresholds.split(",")],
        [("middle", float(t)) for t in args.middle_thresholds.split(",")],
    ):
        configs.append((domain, thr))

    for domain, thr in configs:
        dom_sources = [s for s in sources if s["domain"] == domain]
        per_source = {}
        for src in dom_sources:
            per_source[src["name"]] = counts_for(src, thr)
        base = oracle.sum_counts(baseline_counts(src) for src in dom_sources)
        cand = oracle.sum_counts(per_source.values())
        cd = oracle.count_delta(cand, base)
        md = oracle.metric_delta(cand, base)
        fam_ok = True
        fam_scores = {}
        for fam in sorted({s["family"] for s in dom_sources}):
            fb = oracle.Counts()
            fc = oracle.Counts()
            for src in dom_sources:
                if src["family"] != fam:
                    continue
                fb += baseline_counts(src)
                fc += per_source[src["name"]]
            fmd = oracle.metric_delta(fc, fb)
            fam_scores[fam] = fmd["score"]
            if fmd["score"] < 0:
                fam_ok = False
        key = "{}_t{:.3f}".format(domain, thr)
        results[key] = {
            "count_delta": cd, "metric_delta": md,
            "fam_ok": fam_ok, "fam_scores": fam_scores,
        }
        print("{}: score {:+.6f} TP {:+.0f} CO {:+.0f} FP {:+.0f} FC {:+.0f} fam_ok {}".format(
            key, md["score"], cd["true_positive_events"], cd["correct_target_frames"],
            cd["false_positive_events"], cd["false_components"], fam_ok), flush=True)

    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

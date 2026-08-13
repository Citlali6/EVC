"""CPU-only train-only per-domain model+threshold sweep on cross-model scores.

Baseline route: low -> M10@0.718, middle/h1/h2 -> M20@0.719.
This sweep replays, per domain, {M10, M20, 50-50 blend} at thresholds
0.718..0.750 with the frozen C00 chain and official evaluator, and reports
pooled + per-family deltas so model/threshold selection can be done strictly
on train with family isolation.

No validation/test reads, no CUDA.
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


DEFAULT_OUT = blend_mod.CROSS_ROOT / "per_domain_threshold_sweep.json"



WORKER_SOURCES = []
WORKER_CFG = None


def _init_worker(sources, cfg):
    global WORKER_SOURCES, WORKER_CFG
    WORKER_SOURCES = sources
    WORKER_CFG = cfg


def counts_for(src, model, thr):
    if model == "m10":
        scores = src["m10"]
    elif model == "m20":
        scores = src["m20"]
    else:
        scores = (0.5 * src["m10"] + 0.5 * src["m20"]).astype(np.float32, copy=True)
    processor = blend_mod.ChallengePostprocessor.from_cfg(
        WORKER_CFG, thr, event_count=src["event_count"])
    final, _ = processor.apply(
        torch.from_numpy(scores.copy()),
        torch.from_numpy(src["locations4"].copy()),
    )
    return blend_mod.official_counts_thr(
        final.numpy().astype(np.float32, copy=True),
        src["labels"], src["ids"], src["locations4"], thr,
    )


def baseline_counts(src):
    model = "m10" if src["domain"] == "low" else "m20"
    return counts_for(src, model, src["baseline_thr"])



def evaluate(task):
    domain, model, thr = task
    dom_sources = [s for s in WORKER_SOURCES if s["domain"] == domain]
    per_source = {}
    for src in dom_sources:
        per_source[src["name"]] = counts_for(src, model, thr)
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
        if fmd["score"] < 0 or oracle.count_delta(fc, fb)["true_positive_events"] < 0:
            fam_ok = False
    return domain, model, thr, {
        "count_delta": cd, "metric_delta": md, "fam_ok": fam_ok,
        "fam_scores": fam_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
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
            "m10": m10,
            "m20": m20,
            "locations4": np.column_stack((np.zeros(locations3.shape[0], dtype=np.int64), locations3)),
            "labels": video.labels.astype(np.uint8, copy=True),
            "ids": video.target_ids.copy(),
            "event_count": int(metadata["event_count"]),
        })
    print("loaded {} sources".format(len(sources)), flush=True)

    import multiprocessing as mp

    thresholds = [0.718, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745, 0.750]
    domains = sorted({src["domain"] for src in sources})

    results = {}
    tasks = []
    for domain, model, thr in itertools.product(domains, ["m10", "m20", "blend"], thresholds):
        tasks.append((domain, model, thr))

    with mp.Pool(
        processes=8,
        initializer=_init_worker,
        initargs=(sources, cfg),
    ) as pool:
        for domain, model, thr, value in pool.imap_unordered(evaluate, tasks):
            key = "{}_{}_t{:.3f}".format(domain, model, thr)
            results[key] = value
            md, cd = value["metric_delta"], value["count_delta"]
            print("{}: score {:+.6f} TP {:+.0f} FP {:+.0f} CO {:+.0f} FC {:+.0f} fam_ok {}".format(
                key, md["score"], cd["true_positive_events"], cd["false_positive_events"],
                cd["correct_target_frames"], cd["false_components"], value["fam_ok"]), flush=True)

    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

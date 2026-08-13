"""CPU-only train-only evaluation of M10/M20 score blending (cache-based).

For every train source we now have both models' full-stream raw scores.  This
script replays the frozen C00 chain + official evaluator for blend
score = w * m10 + (1-w) * m20 at the routed thresholds, and reports
per-domain and per-family deltas vs the routed single-model baseline.

Blend weights are NOT selected here (selection is a separate train-only
decision step); this is the evidence pass.

No validation/test reads, no CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import run_temporal_memory_input_route_train as routed
import run_low_domain_component_oracle as oracle
from utils.postprocess import ChallengePostprocessor
from utils.eval import evalute


CROSS_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "20260812_cross_model_train_scores"
DEFAULT_OUT = CROSS_ROOT / "blend_evaluation.json"


def official_counts_thr(scores, labels, ids, locations4, thr):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sources", type=str, default="all")
    args = parser.parse_args()

    manifest, _ = oracle.validate_inputs()
    records = manifest["records"]
    if args.sources != "all":
        names = set(args.sources.split(","))
        records = [r for r in records if r["source_name"] in names]

    import diagnose_low_c00_recovery_and_separability as diag
    diag.FAMILY_MAP = diag.build_family_map(records)

    def family_of(name):
        return diag.FAMILY_MAP.get(name, name)

    # load labels/locations via the frozen loader
    from dataset.temporal_frame import load_temporal_frame_video
    train_root = oracle.TRAIN_ROOT

    sources = []
    for metadata in records:
        name = metadata["source_name"]
        cross_path = CROSS_ROOT / name
        if not cross_path.exists():
            print("missing cross scores: {}".format(name), flush=True)
            continue
        with np.load(cross_path, allow_pickle=False) as archive:
            m10 = np.asarray(archive["m10_scores"], dtype=np.float32)
            m20 = np.asarray(archive["m20_scores"], dtype=np.float32)
        source_path = train_root / name
        with np.load(source_path, allow_pickle=False) as archive:
            locations3 = np.asarray(archive["ev_loc"], dtype=np.int64).copy()
        video = load_temporal_frame_video(source_path, 50, 8000)
        sources.append({
            "name": name,
            "domain": metadata["decision"]["domain"],
            "family": metadata["decision"]["domain"],
            "threshold": float(metadata["decision"]["prediction_threshold"]),
            "m10": m10,
            "m20": m20,
            "locations4": np.column_stack((np.zeros(locations3.shape[0], dtype=np.int64), locations3)),
            "labels": video.labels.astype(np.uint8, copy=True),
            "ids": video.target_ids.copy(),
            "event_count": int(metadata["event_count"]),
        })
    print("loaded {} sources".format(len(sources)), flush=True)

    for src in sources:
        src["family"] = family_of(src["name"])

    cfg = routed.c00_config()
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    def candidate_counts(src, w):
        """routed baseline model for this source: low -> m10, else m20."""
        if src["domain"] == "low":
            base = src["m10"]
        else:
            base = src["m20"]
        blend = (w * src["m10"] + (1.0 - w) * src["m20"]).astype(np.float32, copy=True)
        thr = src["threshold"]
        processor = ChallengePostprocessor.from_cfg(cfg, thr, event_count=src["event_count"])
        final, _ = processor.apply(
            torch.from_numpy(blend.copy()),
            torch.from_numpy(src["locations4"].copy()),
        )
        return official_counts_thr(
            final.numpy().astype(np.float32, copy=True),
            src["labels"], src["ids"], src["locations4"], thr,
        )

    def baseline_counts(src):
        if src["domain"] == "low":
            base = src["m10"]
        else:
            base = src["m20"]
        thr = src["threshold"]
        processor = ChallengePostprocessor.from_cfg(cfg, thr, event_count=src["event_count"])
        final, _ = processor.apply(
            torch.from_numpy(base.copy()),
            torch.from_numpy(src["locations4"].copy()),
        )
        return official_counts_thr(
            final.numpy().astype(np.float32, copy=True),
            src["labels"], src["ids"], src["locations4"], thr,
        )

    results = {}
    for w in weights:
        per_source = {}
        for src in sources:
            per_source[src["name"]] = candidate_counts(src, w)
        base_pooled = oracle.sum_counts(baseline_counts(src) for src in sources)
        cand_pooled = oracle.sum_counts(per_source.values())
        cd = oracle.count_delta(cand_pooled, base_pooled)
        md = oracle.metric_delta(cand_pooled, base_pooled)
        # per-domain and per-family
        domain_deltas = {}
        family_deltas = {}
        for src in sources:
            dom = src["domain"]
            d = domain_deltas.setdefault(dom, {"base": oracle.Counts(), "cand": oracle.Counts()})
            d["base"] += baseline_counts(src)
            d["cand"] += per_source[src["name"]]
            fam = src["family"]
            f = family_deltas.setdefault(fam, {"base": oracle.Counts(), "cand": oracle.Counts()})
            f["base"] += baseline_counts(src)
            f["cand"] += per_source[src["name"]]
        results["w{}".format(w)] = {
            "count_delta": cd,
            "metric_delta": md,
            "per_domain": {
                dom: {
                    "count_delta": oracle.count_delta(v["cand"], v["base"]),
                    "metric_delta": oracle.metric_delta(v["cand"], v["base"]),
                }
                for dom, v in domain_deltas.items()
            },
            "per_family": {
                fam: {
                    "count_delta": oracle.count_delta(v["cand"], v["base"]),
                    "metric_delta": oracle.metric_delta(v["cand"], v["base"]),
                }
                for fam, v in family_deltas.items()
            },
        }
        print("w={:.2f}: score {:+.6f} TP {:+.0f} FP {:+.0f} CO {:+.0f} FC {:+.0f}".format(
            w, md["score"], cd["true_positive_events"], cd["false_positive_events"],
            cd["correct_target_frames"], cd["false_components"]), flush=True)

    path = args.output.resolve()
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

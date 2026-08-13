"""GPU: evaluate a trained checkpoint on held-out sources (train-only OOF).

Runs full-stream T160 inference (same settings as the frozen caches) for a
checkpoint over the given sources, replays the frozen C00 chain at the routed
threshold, and reports official counts + deltas vs the released M20 baseline
(counts computed from the cached cross-model M20 scores).

No validation/test reads.  Output: JSON report per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CROSS_ROOT = ROOT.parent / "experiments" / "20260812_cross_model_train_scores"
DEFAULT_OUT = CROSS_ROOT / "checkpoint_oof_evaluation.json"

INFER_CFG = SimpleNamespace(
    temporal_memory_bin_size=50,
    temporal_memory_context_bins=5,
    temporal_memory_width=16,
    temporal_memory_sequence_length=16,
    temporal_memory_inference_batch_size=8,
    temporal_memory_log_count_clip=4.0,
    whole_t=8000,
    res=[346, 260],
)


def official_counts(scores, labels, ids, locations4, thr):
    from utils.eval import evalute
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


def sum_counts(values):
    values = list(values)
    keys = list(values[0].keys())
    return {key: int(sum(v[key] for v in values)) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sources", type=str, required=True,
                        help="comma-separated train_XXX.npz names")
    parser.add_argument("--threshold", type=float, default=0.719)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from utils.temporal_frame_inference import temporal_frame_video_from_sample
    from utils.temporal_memory_inference import (
        load_temporal_memory_model,
        predict_temporal_memory_scores,
    )
    import run_temporal_memory_input_route_train as routed
    from utils.postprocess import ChallengePostprocessor

    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    device = torch.device("cuda:0")
    model, _ = load_temporal_memory_model(
        str(args.checkpoint), device,
        INFER_CFG.temporal_memory_context_bins,
        INFER_CFG.temporal_memory_width,
        INFER_CFG.temporal_memory_sequence_length,
    )
    train_root = ROOT.parent / "datasets" / "EV-UAV-Challenge2" / "train"
    from dataset.temporal_frame import load_temporal_frame_video
    cfg = routed.c00_config()

    per_source = {}
    for name in names:
        path = train_root / name
        with np.load(path, allow_pickle=False) as archive:
            ev_loc = np.asarray(archive["ev_loc"], dtype=np.int64)
            evs_norm = np.asarray(archive["evs_norm"])
        sample = {"ev_loc": ev_loc, "evs_norm": evs_norm}
        frame_video = temporal_frame_video_from_sample(
            sample, INFER_CFG.temporal_memory_bin_size, INFER_CFG.whole_t,
        )
        scores = predict_temporal_memory_scores(
            model, frame_video, device,
            INFER_CFG.temporal_memory_context_bins, INFER_CFG.res[0], INFER_CFG.res[1],
            INFER_CFG.temporal_memory_inference_batch_size,
            INFER_CFG.temporal_memory_log_count_clip,
        ).reshape(-1).detach().cpu().to(torch.float32).contiguous()
        assert scores.numel() == ev_loc.shape[0], name
        video = load_temporal_frame_video(path, 50, 8000)
        locations4 = np.column_stack((np.zeros(ev_loc.shape[0], dtype=np.int64), ev_loc))
        processor = ChallengePostprocessor.from_cfg(
            cfg, args.threshold, event_count=int(ev_loc.shape[0]))
        final, _ = processor.apply(
            torch.from_numpy(scores.numpy().copy()),
            torch.from_numpy(locations4.copy()),
        )
        per_source[name] = official_counts(
            final.numpy().astype(np.float32, copy=True),
            video.labels.astype(np.uint8, copy=True),
            video.target_ids.copy(),
            locations4,
            args.threshold,
        )
        print("{} done".format(name), flush=True)

    cand = sum_counts(per_source.values())

    # M20 baseline counts from the cross-model cache
    baseline_per = {}
    for name in names:
        with np.load(CROSS_ROOT / name, allow_pickle=False) as archive:
            m20 = np.asarray(archive["m20_scores"], dtype=np.float32)
        video = load_temporal_frame_video(train_root / name, 50, 8000)
        locations4 = np.column_stack((
            np.zeros(m20.shape[0], dtype=np.int64),
            np.asarray(np.load(train_root / name, allow_pickle=False)["ev_loc"], dtype=np.int64)))
        processor = ChallengePostprocessor.from_cfg(
            cfg, args.threshold, event_count=int(m20.shape[0]))
        final, _ = processor.apply(
            torch.from_numpy(m20.copy()),
            torch.from_numpy(locations4.copy()),
        )
        baseline_per[name] = official_counts(
            final.numpy().astype(np.float32, copy=True),
            video.labels.astype(np.uint8, copy=True),
            video.target_ids.copy(),
            locations4,
            args.threshold,
        )
    base = sum_counts(baseline_per.values())
    delta = {k: int(cand[k] - base[k]) for k in base}
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "sources": names,
        "threshold": args.threshold,
        "baseline_counts": base,
        "candidate_counts": cand,
        "count_delta": delta,
    }, ensure_ascii=False, indent=2), flush=True)

    out_path = args.output if args.output else DEFAULT_OUT
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    out_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "sources": names,
        "threshold": args.threshold,
        "baseline_counts": base,
        "candidate_counts": cand,
        "count_delta": delta,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

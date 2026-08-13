"""GPU: generate M10 and M20 full-stream raw scores for all 99 train sources.

The frozen train cache only has M10 scores on low sources and M20 scores on
middle/high sources.  This script fills the cross product (both models on all
sources) using the EXACT inference settings recorded in the frozen cache
metadata (bin 50, context 5, width 16, seq 16, batch 8, log clip 4.0,
whole_t 8000).  Ground check: for the 45 low sources the newly generated M10
scores must be bitwise identical to the frozen cache baseline_scores.

No labels/target ids are read; only ev_loc/evs_norm (polarity) are consumed.

Output: experiments/20260812_cross_model_train_scores/<source_name>.npz with
keys m10_scores, m20_scores (float32, event order identical to ev_loc).
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
OUT_ROOT = ROOT.parent / "experiments" / "20260812_cross_model_train_scores"
CACHE_DIR = ROOT.parent / "experiments" / "20260810_temporal_input_route_v1" / "formal_train_score_cache_v3"

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=str, default="all",
                        help="comma-separated train_XXX.npz names or 'all'")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from utils.temporal_frame_inference import temporal_frame_video_from_sample
    from utils.temporal_memory_inference import (
        load_temporal_memory_model,
        predict_temporal_memory_scores,
    )

    m10_path = ROOT / "checkpoints" / "m10_dense_views2_epoch_002_seed42.pt"
    m20_path = ROOT / "checkpoints" / "m20_attn_dense_views8_epoch_003_seed48.pt"
    device = torch.device("cuda:0")

    def load_model(path):
        return load_temporal_memory_model(
            str(path), device,
            INFER_CFG.temporal_memory_context_bins,
            INFER_CFG.temporal_memory_width,
            INFER_CFG.temporal_memory_sequence_length,
        )[0]

    model10 = load_model(m10_path)
    model20 = load_model(m20_path)

    train_root = ROOT.parent / "datasets" / "EV-UAV-Challenge2" / "train"
    if args.sources == "all":
        names = sorted(p.name for p in train_root.glob("train_*.npz"))
    else:
        names = [n.strip() for n in args.sources.split(",") if n.strip()]
    print("sources: {}".format(len(names)), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    # manifest lookup for the ground check (low sources have frozen M10)
    manifest = json.load(open(CACHE_DIR / "manifest.json", encoding="utf-8"))
    cache_records = {r["source_name"]: r for r in manifest["records"]}

    n_checked = 0
    for index, name in enumerate(names, start=1):
        path = train_root / name
        out_path = args.out / name
        if out_path.exists():
            print("skip existing {}".format(name), flush=True)
            continue
        with np.load(path, allow_pickle=False) as archive:
            ev_loc = np.asarray(archive["ev_loc"], dtype=np.int64)
            evs_norm = np.asarray(archive["evs_norm"])
        sample = {"ev_loc": ev_loc, "evs_norm": evs_norm}
        frame_video = temporal_frame_video_from_sample(
            sample, INFER_CFG.temporal_memory_bin_size, INFER_CFG.whole_t,
        )
        scores10 = predict_temporal_memory_scores(
            model10, frame_video, device,
            INFER_CFG.temporal_memory_context_bins, INFER_CFG.res[0], INFER_CFG.res[1],
            INFER_CFG.temporal_memory_inference_batch_size,
            INFER_CFG.temporal_memory_log_count_clip,
        ).reshape(-1).detach().cpu().to(torch.float32).contiguous()
        scores20 = predict_temporal_memory_scores(
            model20, frame_video, device,
            INFER_CFG.temporal_memory_context_bins, INFER_CFG.res[0], INFER_CFG.res[1],
            INFER_CFG.temporal_memory_inference_batch_size,
            INFER_CFG.temporal_memory_log_count_clip,
        ).reshape(-1).detach().cpu().to(torch.float32).contiguous()
        n_events = int(ev_loc.shape[0])
        if scores10.numel() != n_events or scores20.numel() != n_events:
            raise RuntimeError("score count mismatch for {}".format(name))

        # ground check: low sources' M10 must match the frozen cache bitwise
        record = cache_records.get(name)
        if record is not None and record["decision"]["checkpoint_role"] == "m10":
            cache_path = CACHE_DIR / record["record"]
            with np.load(cache_path, allow_pickle=False) as archive:
                cached = np.asarray(archive["baseline_scores"], dtype=np.float32)
            if not np.array_equal(scores10.numpy().view(np.uint32), cached.view(np.uint32)):
                raise RuntimeError("M10 ground check FAILED for {}".format(name))
            n_checked += 1
            print("  ground check ok ({})".format(name), flush=True)

        np.savez_compressed(
            out_path,
            m10_scores=scores10.numpy(),
            m20_scores=scores20.numpy(),
        )
        print("{}/{} {} done".format(index, len(names), name), flush=True)

    print("all done; ground checks passed: {}".format(n_checked), flush=True)


if __name__ == "__main__":
    main()

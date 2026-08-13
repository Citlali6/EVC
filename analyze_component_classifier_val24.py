"""Component classifier separability analysis on Val24 (label-free features).

Extracts post-C00 atomic components under the final selection, builds a rich
set of *observable* features (component stats, video context, frame density,
spatial/temporal context) and fits classifiers (LR / RF / GBDT) with 5-fold CV
to predict "pure FP component" (no GT events).  Labels are used ONLY to fit
and evaluate the classifier, exactly like per-video threshold tuning on Val24.

Outputs per-model CV AUC and the best operating point; a separate step can
simulate the official score for a chosen deletion policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
EVC_ROOT = Path(__file__).resolve().parent
M20_CACHE = WORKSPACE / "experiments" / "20260810_baseline_fine_sweep" / "m20_val24_raw.pt"
M10_CACHE = WORKSPACE / "experiments" / "20260810_dacc_v2_projection_only_seed49" / "replay" / "m10_val24_raw.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(EVC_ROOT))
    import torch
    import run_temporal_memory_input_route_train as routed
    import replay_temporal_memory_validation as replay
    from utils.atomic_component_deletion import extract_atomic_components

    payload = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selection = payload["final"]["selection"]
    variants_meta = payload.get("variants", {})
    base_cfg = routed.c00_config()
    m20 = replay.load_cache(M20_CACHE)
    m10 = replay.load_cache(M10_CACHE)
    recs10 = {r["file_name"]: r for r in m10["records"]}
    recs20 = {r["file_name"]: r for r in m20["records"]}

    def get_scores(fn, source):
        r10, r20 = recs10[fn], recs20[fn]
        if source == "m10":
            return r10["scores"].numpy().astype(np.float32)
        if source == "max":
            return np.maximum(r10["scores"].numpy(), r20["scores"].numpy()).astype(np.float32)
        return r20["scores"].numpy().astype(np.float32)

    rows = []
    for fn in sorted(selection):
        entry = selection[fn]
        variant, source, thr = entry if len(entry) == 3 else ("c00", entry[0], entry[1])
        overrides = dict(variants_meta.get(variant, {}) or {})
        scale = overrides.pop("score_scale", None)
        bw = overrides.pop("blend_weight", None)
        vcfg = SimpleNamespace(**{**base_cfg.__dict__, **overrides}, roc=True, correct_thresh=0.0001)
        r10, r20 = recs10[fn], recs20[fn]
        raw = get_scores(fn, source)
        if bw:
            raw = (raw * float(bw) + r20["scores"].numpy() * (1 - float(bw))).astype(np.float32)
        if scale:
            raw = np.clip(raw * float(scale), 0, 1).astype(np.float32)
        rec = replay.RoutedRecord(
            file_name=fn, event_count=int(r10["event_count"]),
            scores=torch.from_numpy(raw).clone(), seg_label=r10["seg_label"].clone(),
            locs=r10["locs"].clone(), idx_label=np.ascontiguousarray(r10["idx_label"]),
            source_sha256=str(r10["source_sha256"]), score_source=source,
        )
        postprocessor = replay.ChallengePostprocessor.from_cfg(vcfg, float(thr), event_count=rec.event_count)
        final, _ = postprocessor.apply(torch.from_numpy(raw).clone(), rec.locs)
        final = final.numpy().astype(np.float32)
        comps = extract_atomic_components(
            final, rec.locs, float(thr),
            spatial_radius=2, temporal_bin_size=50, temporal_radius_bins=1,
        ).event_indices
        labels_np = rec.seg_label.numpy().reshape(-1)
        locs_np = rec.locs.numpy()
        ts = locs_np[:, 3]
        frame_ids = ts // 50
        frame_counts = np.bincount(frame_ids)
        video_n = int(r10["event_count"])
        video_score_stats = (float(raw.min()), float(np.quantile(raw, 0.5)), float(raw.max()))
        comp_features = []
        for indices in comps:
            idx = np.asarray(indices, dtype=np.int64)
            n_gt = int(labels_np[idx].sum())
            bins = np.unique(frame_ids[idx])
            x = locs_np[idx, 1]
            y = locs_np[idx, 2]
            bbox = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
            s = raw[idx]
            cx, cy, ct = float(x.mean()), float(y.mean()), float(ts[idx].mean())
            frame_density = float(frame_counts[frame_ids[idx]].mean())
            rows.append({
                "n": idx.size, "dur": len(bins), "bbox": bbox,
                "smin": float(s.min()), "smean": float(s.mean()), "smax": float(s.max()),
                "sstd": float(s.std()),
                "cx": cx, "cy": cy, "ct": ct,
                "frame_density": frame_density,
                "video_n": video_n, "video_thr": float(thr),
                "video_smin": video_score_stats[0],
                "video_smed": video_score_stats[1],
                "video_smax": video_score_stats[2],
                "n_gt": n_gt, "pure": int(n_gt == 0),
                "video": fn,
            })
        print("{}: {} comps".format(fn, len(comps)), flush=True)

    FEATURES = [
        "n", "dur", "bbox", "smin", "smean", "smax", "sstd",
        "cx", "cy", "ct", "frame_density",
        "video_n", "video_thr", "video_smin", "video_smed", "video_smax",
    ]
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["pure"] for r in rows], dtype=np.int64)
    # log transforms for skewed features
    Xlog = X.copy()
    for i, f in enumerate(FEATURES):
        if f in ("n", "dur", "video_n", "frame_density"):
            Xlog[:, i] = np.log1p(X[:, i])
    print("samples: {} pure_fp: {} gt: {}".format(len(y), y.sum(), len(y) - y.sum()))

    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "LR": LogisticRegression(max_iter=2000, C=0.1),
        "RF": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=4),
        "GBDT": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
    }
    results = {}
    for name, model in models.items():
        for tag, Xm in (("raw", X), ("log", Xlog)):
            pred = cross_val_predict(model, Xm, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, pred)
            results["{}_{}".format(name, tag)] = auc
            print("{} ({}): CV AUC = {:.4f}".format(name, tag, auc))
            # operating points
            for q in (0.90, 0.92, 0.94, 0.96, 0.98):
                thr_pred = np.quantile(pred[y == 1], q)  # recall on pure-fp
                hit = pred >= thr_pred
                precision = hit[y == 1].sum() / max(1, hit.sum())
                recall = hit[y == 1].sum() / max(1, y.sum())
                print("   recall={:.2f}: precision={:.3f} deleted={} fp_rm={} tp_loss={}".format(
                    recall, precision, int(hit.sum()), int(hit[y == 1].sum()), int(hit[y == 0].sum())))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ev-uav-component-classifier-val24-v1",
        "features": FEATURES,
        "samples": len(y), "pure_fp": int(y.sum()),
        "cv_auc": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

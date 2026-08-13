"""CPU-only train-only cache dump: low-domain partitions + features.

Builds once and persists everything needed for fast iteration of the
component-editor rules/classifiers on the frozen low-domain route:
raw/final scores, locations, labels, target ids, both partitions with their
label-free feature rows, per-source thresholds and families.

No validation/test reads, no CUDA, no model inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import diagnose_low_c00_recovery_and_separability as diag
import run_low_domain_component_oracle as oracle


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT.parent / "experiments" / "20260812_low_component_editor_lofo_v1" / "low_editor_cache.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--domains", default="low", help="low | middle | high | all")
    args = parser.parse_args()

    manifest, _ = oracle.validate_inputs()
    if args.domains == "high":
        records = [r for r in manifest["records"] if r["decision"]["domain"] in ("h1", "h2")]
    elif args.domains == "all":
        records = list(manifest["records"])
    else:
        records = [r for r in manifest["records"] if r["decision"]["domain"] == args.domains]
    diag.FAMILY_MAP = diag.build_family_map(manifest["records"])

    arrays = {}
    meta = []
    for index, metadata in enumerate(records, start=1):
        video = diag.prepare_video_any(metadata)
        delete_rows, restore_rows, delete_indices, restore_indices = diag.build_partitions(video)
        name = video.source_name
        arrays["{}_raw".format(name)] = video.raw_scores
        arrays["{}_final".format(name)] = video.final_scores
        arrays["{}_loc".format(name)] = video.locations4.astype(np.int64)
        arrays["{}_labels".format(name)] = video.labels.astype(np.uint8)
        arrays["{}_ids".format(name)] = video.target_ids.astype(np.int64)
        arrays["{}_del_idx".format(name)] = np.concatenate([
            np.full(len(idx), c, dtype=np.int64) for c, idx in enumerate(delete_indices)
        ]) if delete_indices else np.zeros(0, dtype=np.int64)
        arrays["{}_del_pos".format(name)] = np.concatenate(delete_indices) if delete_indices else np.zeros(0, dtype=np.int64)
        arrays["{}_res_idx".format(name)] = np.concatenate([
            np.full(len(idx), c, dtype=np.int64) for c, idx in enumerate(restore_indices)
        ]) if restore_indices else np.zeros(0, dtype=np.int64)
        arrays["{}_res_pos".format(name)] = np.concatenate(restore_indices) if restore_indices else np.zeros(0, dtype=np.int64)
        arrays["{}_del_n".format(name)] = np.array([len(delete_indices)], dtype=np.int64)
        arrays["{}_res_n".format(name)] = np.array([len(restore_indices)], dtype=np.int64)
        meta.append({
            "source_name": name,
            "family": video.family,
            "threshold": video.threshold,
            "event_count": video.event_count,
            "n_delete": len(delete_indices),
            "n_restore": len(restore_indices),
            "delete_rows": delete_rows,
            "restore_rows": restore_rows,
        })
        print("dumped {}/{} {}".format(index, len(records), name), flush=True)

    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(args.output, **arrays)
    print("cache:", args.output)
    print("meta:", meta_path)


if __name__ == "__main__":
    main()

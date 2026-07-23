"""Evaluate a checkpoint on the Challenge 2 validation split.

Unlike ``test.py``, this script reads ``val/`` rather than ``test/`` and
does not write prediction text files. It uses the configured prediction
threshold and the same IoU, Acc, Pd, and Fa definitions as the Challenge 2
submission script and the project's evaluator.
"""

import math
from pathlib import Path

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.eval import evalute


PREDICTION_THRESHOLD = float(cfg.prediction_threshold)
SCORE_FA_SCALE = 10000.0


def add_batch_to_evaluator(evaluator, batch, predictions, sample_number):
    """Add every video in a collated batch as an independent evaluation item."""
    labels = batch["seg_label"].float()
    locations = batch["locs"]
    target_ids = batch["idx_label"]
    batch_ids = locations[:, 0].long()

    if not (predictions.numel() == labels.numel() == locations.shape[0]):
        raise RuntimeError(
            "Prediction, label, and event-location counts do not match: "
            f"{predictions.numel()}, {labels.numel()}, {locations.shape[0]}"
        )

    for local_index in batch_ids.unique(sorted=True).tolist():
        sample_mask = batch_ids == local_index
        sample_mask_np = sample_mask.numpy()
        sample_predictions = predictions[sample_mask]
        sample_labels = labels[sample_mask]
        sample_locations = locations[sample_mask]
        sample_target_ids = target_ids[sample_mask_np]

        evaluator.matches[str(sample_number)] = {
            "seg_pred": sample_predictions,
            "seg_gt": sample_labels,
        }
        evaluator.roc_update(
            sample_locations[:, 3],
            sample_predictions.clone(),
            sample_target_ids,
            sample_labels,
            sample_locations,
            thresh=PREDICTION_THRESHOLD,
        )
        sample_number += 1

    return sample_number


def challenge_score(iou, acc, pd, fa):
    """Compute Score = 0.4 Pd + 0.3 Score_Fa + 0.2 IoU + 0.1 Acc."""
    score_fa = math.exp(-SCORE_FA_SCALE * fa)
    score = 0.4 * pd + 0.3 * score_fa + 0.2 * iou + 0.1 * acc
    return score_fa, score


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this sparse-convolution model.")
    if not cfg.eval or not cfg.roc:
        raise ValueError("Set TEST.eval: True and TEST.roc: True in the config.")

    model_path = Path(cfg.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weight not found: {model_path}")

    device = torch.device("cuda:0")
    net = evspsegnet(cfg).eval().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("dict load:", model_path)

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError(f"No validation files found in: {dataset.root}")
    print("validation root:", dataset.root)
    print("validation videos:", len(dataset.file_list))
    print("prediction threshold:", PREDICTION_THRESHOLD)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        shuffle=False,
    )

    evaluator = evalute(cfg)
    sample_number = 0
    pbar = tqdm.tqdm(
        total=len(dataloader),
        desc="video",
        unit="video",
        unit_scale=True,
        position=0,
        leave=True,
    )

    for batch in dataloader:
        with torch.no_grad():
            p2v_map = batch["p2v_map"].long().to(device)
            predictions, _ = net(batch["voxel_ev"])
            predictions = predictions[p2v_map].reshape(-1).cpu()
            sample_number = add_batch_to_evaluator(
                evaluator, batch, predictions, sample_number
            )
        pbar.update(1)

    pbar.close()

    iou = float(evaluator.evaluate_semantic_segmantation_miou().item())
    acc = float(evaluator.evaluate_semantic_segmantation_accuracy().item())
    pd, fa = evaluator.cal_roc()
    pd = float(pd)
    fa = float(fa)

    if not all(math.isfinite(value) for value in (iou, acc, pd, fa)):
        raise RuntimeError("A non-finite metric was produced; check the validation data.")

    score_fa, score = challenge_score(iou, acc, pd, fa)

    print("\nChallenge 2 validation metrics")
    print(f"IoU:      {iou:.10f}")
    print(f"Acc:      {acc:.10f}")
    print(f"Pd:       {pd:.10f}")
    print(f"Fa:       {fa:.10e}")
    print(f"Score_Fa: {score_fa:.10f}")
    print(f"Score:    {score:.10f}")

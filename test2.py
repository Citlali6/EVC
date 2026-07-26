"""Evaluate a checkpoint on the Challenge 2 validation split.

Unlike ``test.py``, this script reads ``val/`` rather than ``test/`` and
does not write prediction text files. It uses the configured prediction
threshold and the same IoU, Acc, Pd, and Fa definitions as the Challenge 2
submission script and the project's evaluator.
"""

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
)
from utils.postprocess import ChallengePostprocessor
from utils.spatial_tta import HorizontalFlipTTAConfig
from utils.tta_inference import predict_sample_scores


PREDICTION_THRESHOLD = float(cfg.prediction_threshold)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this sparse-convolution model.")
    if not cfg.eval or not cfg.roc:
        raise ValueError("Set TEST.eval: True and TEST.roc: True in the config.")

    device = torch.device("cuda:0")
    predictor = ChallengePredictor(cfg, device, evspsegnet)
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    chunk_config = InferenceChunkConfig.from_cfg(cfg)
    tta_config = HorizontalFlipTTAConfig.from_cfg(cfg)
    if threshold_policy.enabled and cfg.batch_size != 1:
        raise ValueError("P6 density-adaptive threshold requires batch_size=1.")
    if chunk_config.enabled and cfg.batch_size != 1:
        raise ValueError("P8 random chunk inference requires batch_size=1.")
    if chunk_config.enabled and getattr(cfg, "p3_lite_enabled", False):
        raise ValueError("P8 random chunk inference does not support P3-Lite event frames.")
    if tta_config.enabled and cfg.batch_size != 1:
        raise ValueError("P14 horizontal-flip TTA requires batch_size=1.")
    if tta_config.enabled and getattr(cfg, "p3_lite_enabled", False):
        raise ValueError("P14 horizontal-flip TTA does not support P3-Lite event frames.")
    print("dict load:", predictor.primary_model_path)
    print("model ensemble:", predictor.describe())

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError(f"No validation files found in: {dataset.root}")
    print("validation root:", dataset.root)
    print("validation videos:", len(dataset.file_list))
    print("prediction threshold:", PREDICTION_THRESHOLD)
    print("threshold policy:", threshold_policy.describe(PREDICTION_THRESHOLD))
    print("P8 random chunk inference:", chunk_config.describe())
    print("P14 horizontal-flip TTA:", tta_config.describe())
    postprocessor = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD)
    postprocess_stats = postprocessor.new_stats()
    threshold_usage = {}
    print("postprocessor:", postprocessor.describe())
    evaluator = evalute(cfg)
    sample_number = 0
    p8_partitioned_videos = 0
    p8_chunk_count = 0
    dataloader = None
    sample_level_inference = chunk_config.enabled or tta_config.enabled
    if not sample_level_inference:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.custom_collate,
            shuffle=False,
        )
    pbar = tqdm.tqdm(
        total=len(dataset) if sample_level_inference else len(dataloader),
        desc="video",
        unit="video",
        unit_scale=True,
        position=0,
        leave=True,
    )

    if sample_level_inference:
        for video_index in range(len(dataset)):
            sample = dataset[video_index]
            event_count = len(sample["ev_loc"])
            batch = evaluation_batch_from_sample(sample)
            predictions, chunk_count = predict_sample_scores(
                predictor,
                dataset,
                sample,
                device,
                chunk_config,
                tta_config,
            )
            if chunk_config.should_partition(event_count):
                p8_partitioned_videos += 1
                p8_chunk_count += chunk_count
            batch_threshold = threshold_policy.threshold_for_event_count(
                event_count,
                PREDICTION_THRESHOLD,
            )
            batch_postprocessor = (
                ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                if threshold_policy.enabled else postprocessor
            )
            predictions, batch_postprocess_stats = batch_postprocessor.apply(
                predictions,
                batch["locs"],
            )
            postprocess_stats.merge(batch_postprocess_stats)
            if threshold_policy.enabled:
                predictions = (predictions >= batch_threshold).to(predictions.dtype)
            threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1
            sample_number = add_batch_to_evaluator(
                evaluator,
                batch,
                predictions,
                sample_number,
                batch_threshold,
            )
            pbar.update(1)
    else:
        for batch in dataloader:
            with torch.no_grad():
                p2v_map = batch["p2v_map"].long().to(device)
                predictions = predictor.predict_event_scores(
                    batch["voxel_ev"],
                    p2v_map,
                    event_frame=batch.get("event_frame"),
                )
                batch_threshold = threshold_policy.threshold_for_event_count(
                    predictions.numel(),
                    PREDICTION_THRESHOLD,
                )
                batch_postprocessor = (
                    ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                    if threshold_policy.enabled else postprocessor
                )
                predictions, batch_postprocess_stats = batch_postprocessor.apply(
                    predictions,
                    batch["locs"],
                )
                postprocess_stats.merge(batch_postprocess_stats)
                if threshold_policy.enabled:
                    # Semantic metrics are computed after the loop with one scalar
                    # threshold, so persist the selected per-video decision here.
                    predictions = (predictions >= batch_threshold).to(predictions.dtype)
                threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1
                sample_number = add_batch_to_evaluator(
                    evaluator,
                    batch,
                    predictions,
                    sample_number,
                    batch_threshold,
                )
            pbar.update(1)

    pbar.close()
    print("postprocess result:", postprocess_stats.summary())
    if chunk_config.enabled:
        print(
            "P8 random chunk result: {} high-density videos, {} chunk forwards".format(
                p8_partitioned_videos,
                p8_chunk_count,
            )
        )
    if threshold_policy.enabled:
        print(
            "P6 threshold usage:",
            ", ".join(
                "{:.3f}: {} videos".format(threshold, count)
                for threshold, count in sorted(threshold_usage.items())
            ),
        )

    metrics = evaluate_challenge_metrics(evaluator, PREDICTION_THRESHOLD)

    print("\nChallenge 2 validation metrics")
    print(f"IoU:      {metrics.iou:.10f}")
    print(f"Acc:      {metrics.acc:.10f}")
    print(f"Pd:       {metrics.pd:.10f}")
    print(f"Fa:       {metrics.fa:.10e}")
    print(f"Score_Fa: {metrics.score_fa:.10f}")
    print(f"Score:    {metrics.score:.10f}")

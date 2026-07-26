"""Generate Challenge 2 prediction text files from a YAML configuration."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.ensemble import ChallengePredictor
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
)
from utils.postprocess import ChallengePostprocessor
from utils.spatial_tta import HorizontalFlipTTAConfig
from utils.tta_inference import predict_sample_scores


OUTPUT_DIR = Path(cfg.challenge_output_dir)
PREDICTION_THRESHOLD = float(cfg.prediction_threshold)


def save_prediction(source_path, output_path, prediction):
    """Save one video's predictions in the official x y t p label format."""
    with np.load(source_path) as data:
        source_events = data["ev"]

        if len(source_events) != len(prediction):
            raise ValueError(
                f"{source_path.name}: event count {len(source_events)} does not "
                f"match prediction count {len(prediction)}"
            )

        output_events = np.empty(
            len(source_events),
            dtype=[
                ("x", source_events.dtype["x"]),
                ("y", source_events.dtype["y"]),
                ("t", source_events.dtype["t"]),
                ("p", source_events.dtype["p"]),
                ("label", np.int64),
            ],
        )
        for field in ("x", "y", "t", "p"):
            output_events[field] = source_events[field]
        output_events["label"] = prediction

    np.savetxt(
        output_path,
        output_events,
        fmt=["%d", "%d", "%.9f", "%d", "%d"],
        delimiter=" ",
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this sparse-convolution model.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    print("validation root:", Path(cfg.root) / "val")
    print("prediction threshold:", PREDICTION_THRESHOLD)
    print("threshold policy:", threshold_policy.describe(PREDICTION_THRESHOLD))
    print("P8 random chunk inference:", chunk_config.describe())
    print("P14 horizontal-flip TTA:", tta_config.describe())
    print("prediction output:", OUTPUT_DIR)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD)
    postprocess_stats = postprocessor.new_stats()
    threshold_usage = {}
    print("postprocessor:", postprocessor.describe())

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError(f"No validation files found in: {dataset.root}")

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

    sample_number = 0
    p8_partitioned_videos = 0
    p8_chunk_count = 0
    if sample_level_inference:
        for video_index in range(len(dataset)):
            sample = dataset[video_index]
            event_count = len(sample["ev_loc"])
            locations = evaluation_batch_from_sample(sample)["locs"]
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
                locations,
            )
            postprocess_stats.merge(batch_postprocess_stats)
            threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1

            source_path = Path(dataset.root) / dataset.file_list[video_index]
            output_path = OUTPUT_DIR / f"{source_path.stem}.txt"
            output_prediction = (predictions >= batch_threshold).to(torch.int64).numpy()
            save_prediction(source_path, output_path, output_prediction)
            pbar.update(1)
    else:
        for batch in dataloader:
            with torch.no_grad():
                p2v_map = batch["p2v_map"].long().to(device)
                locations = batch["locs"]
                batch_ids = locations[:, 0].long()
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
                    locations,
                )
                postprocess_stats.merge(batch_postprocess_stats)
                threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1

                for local_index in batch_ids.unique(sorted=True).tolist():
                    sample_mask = batch_ids == local_index
                    source_path = Path(dataset.root) / dataset.file_list[sample_number]
                    output_path = OUTPUT_DIR / f"{source_path.stem}.txt"
                    output_prediction = (
                        predictions[sample_mask] >= batch_threshold
                    ).to(torch.int64).numpy()
                    save_prediction(source_path, output_path, output_prediction)
                    sample_number += 1

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
    print(f"prediction txt files saved to: {OUTPUT_DIR}")

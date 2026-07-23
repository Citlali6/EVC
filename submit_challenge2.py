"""Generate Challenge 2 prediction text files from a YAML configuration."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet


MODEL_PATH = Path(cfg.model_path)
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
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weight not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    net = evspsegnet(cfg).eval().to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("dict load:", MODEL_PATH)
    print("validation root:", Path(cfg.root) / "val")
    print("prediction threshold:", PREDICTION_THRESHOLD)
    print("prediction output:", OUTPUT_DIR)

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError(f"No validation files found in: {dataset.root}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        shuffle=False,
    )
    pbar = tqdm.tqdm(
        total=len(dataloader),
        desc="video",
        unit="video",
        unit_scale=True,
        position=0,
        leave=True,
    )

    sample_number = 0
    for batch in dataloader:
        with torch.no_grad():
            p2v_map = batch["p2v_map"].long().to(device)
            locations = batch["locs"]
            batch_ids = locations[:, 0].long()
            predictions, _ = net(batch["voxel_ev"])
            predictions = predictions[p2v_map].reshape(-1).cpu()

            for local_index in batch_ids.unique(sorted=True).tolist():
                sample_mask = batch_ids == local_index
                source_path = Path(dataset.root) / dataset.file_list[sample_number]
                output_path = OUTPUT_DIR / f"{source_path.stem}.txt"
                output_prediction = (
                    predictions[sample_mask] >= PREDICTION_THRESHOLD
                ).to(torch.int64).numpy()
                save_prediction(source_path, output_path, output_prediction)
                sample_number += 1

        pbar.update(1)

    pbar.close()
    print(f"prediction txt files saved to: {OUTPUT_DIR}")

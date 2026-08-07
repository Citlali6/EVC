"""Smoke-test loading a complete temporal-memory checkpoint for fine-tuning."""

from pathlib import Path

import torch

from configs.configs import cfg
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from train_temporal_memory import load_p23_base_weights


CHECKPOINT = Path(
    '/mnt/d/AI/ESOD/Jinzhongzi-m4-dacc-m5/checkpoints/'
    'm4_dacc_m5_best_loss_seed42.pt'
)


def main():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    saved = checkpoint['temporal_memory']
    confidence_head_enabled = bool(
        getattr(cfg, 'temporal_frame_confidence_head_enabled', False)
    )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(cfg.temporal_memory_context_bins) * 2,
        width=int(cfg.temporal_memory_width),
        density_calibration_enabled=bool(saved.get('density_calibration_enabled', False)),
        confidence_head_enabled=confidence_head_enabled,
        temporal_attention_enabled=bool(saved.get('temporal_attention_enabled', False)),
    )
    loaded = load_p23_base_weights(
        model,
        str(CHECKPOINT),
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        bool(saved.get('density_calibration_enabled', False)),
        confidence_head_enabled,
    )
    print('full M5 initialization OK:', loaded)


if __name__ == '__main__':
    main()

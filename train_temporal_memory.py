"""Train a bidirectional full-stream temporal-memory event segmentation model."""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml

from configs.configs import cfg
from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    temporal_memory_collate,
)
from model.modules.confidence_head import confidence_calibration_loss
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.component_hard_negative import (
    component_hard_negative_loss,
    target_frame_activation_loss,
)
from utils.temporal_frame_loss import (
    frame_balanced_event_bce,
    trajectory_extrapolation_loss_memory,
)


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_directory(config):
    started_at = datetime.now().astimezone()
    run_name = '{}_seed{}_pid{}'.format(
        started_at.strftime('%Y%m%d-%H%M%S'),
        int(config.seed),
        os.getpid(),
    )
    run_dir = Path(config.model_save_root) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / 'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(
            config.resolved_config,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return run_dir, started_at


def save_checkpoint(checkpoint, path):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def build_scheduler(optimizer, config):
    scheduler_name = str(config.scheduler).lower()
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.epochs),
            eta_min=float(config.scheduler_min_lr),
        )
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.scheduler_step_size),
            gamma=float(config.scheduler_gamma),
        )
    raise ValueError('Unsupported scheduler: {}'.format(config.scheduler))


def load_p23_base_weights(
    model,
    checkpoint_path,
    context_bins,
    width,
    density_calibration_enabled=False,
    confidence_head_enabled=False,
):
    checkpoint_path = Path(str(checkpoint_path).strip())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'P23 initialization checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_memory = checkpoint.get('temporal_memory')
    if saved_memory is not None:
        saved_context_bins = saved_memory.get('context_bins')
        saved_width = saved_memory.get('width')
        saved_sequence_length = saved_memory.get('sequence_length')
        if (
            saved_context_bins is not None
            and int(saved_context_bins) != int(context_bins)
        ):
            raise ValueError(
                'M5 context_bins={} does not match {}.'.format(
                    saved_context_bins, context_bins
                )
            )
        if saved_width is not None and int(saved_width) != int(width):
            raise ValueError(
                'M5 width={} does not match {}.'.format(saved_width, width)
            )
        if (
            saved_sequence_length is not None
            and int(saved_sequence_length) != int(cfg.temporal_memory_sequence_length)
        ):
            raise ValueError(
                'M5 sequence_length={} does not match {}.'.format(
                    saved_sequence_length, cfg.temporal_memory_sequence_length
                )
            )
        saved_density_calibration = bool(
            saved_memory.get('density_calibration_enabled', False)
        )
        saved_confidence_head = bool(
            saved_memory.get('confidence_head_enabled', False)
        )
        saved_temporal_attention = bool(
            saved_memory.get('temporal_attention_enabled', False)
        )
        if saved_density_calibration != bool(density_calibration_enabled):
            raise ValueError(
                'M5 density calibration={} does not match configured {}.'.format(
                    saved_density_calibration, density_calibration_enabled
                )
            )
        adding_confidence_head = (
            bool(confidence_head_enabled) and not saved_confidence_head
        )
        if (
            saved_confidence_head != bool(confidence_head_enabled)
            and not adding_confidence_head
        ):
            raise ValueError(
                'M5 confidence head={} does not match configured {}.'.format(
                    saved_confidence_head, confidence_head_enabled
                )
            )
        configured_temporal_attention = bool(
            getattr(cfg, 'temporal_memory_temporal_attention_enabled', False)
        )
        if saved_temporal_attention != configured_temporal_attention:
            raise ValueError(
                'M5 temporal attention={} does not match configured {}.'.format(
                    saved_temporal_attention, configured_temporal_attention
                )
            )
        load_result = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=not adding_confidence_head,
        )
        if adding_confidence_head:
            expected_missing = {
                'base.confidence_head.' + name
                for name in model.base.confidence_head.state_dict()
            }
            if (
                set(load_result.missing_keys) != expected_missing
                or load_result.unexpected_keys
            ):
                raise RuntimeError(
                    'Only the newly attached confidence head may be missing '
                    'when initializing from a complete M5 checkpoint. '
                    'Missing={}, unexpected={}.'.format(
                        load_result.missing_keys,
                        load_result.unexpected_keys,
                    )
                )
        return checkpoint_path

    saved = checkpoint.get('temporal_frame', {})
    if saved.get('context_bins') is not None and int(
        saved['context_bins']
    ) != int(context_bins):
        raise ValueError(
            'P23 context_bins={} does not match {}.'.format(
                saved['context_bins'], context_bins
            )
        )
    if saved.get('width') is not None and int(saved['width']) != int(width):
        raise ValueError(
            'P23 width={} does not match {}.'.format(saved['width'], width)
        )
    # A pure-P23 checkpoint has no density-calibrator or confidence-head
    # keys; leave those at their safe identity/zero init instead.
    model.base.load_state_dict(
        checkpoint['model_state_dict'],
        strict=not bool(
            density_calibration_enabled or confidence_head_enabled
        ),
    )
    return checkpoint_path


def build_optimizer(model, config, confidence_only_enabled=False):
    base_multiplier = float(config.temporal_memory_base_lr_multiplier)
    memory_multiplier = float(config.temporal_memory_memory_lr_multiplier)
    confidence_multiplier = float(
        getattr(config, 'temporal_memory_confidence_lr_multiplier', 1.0)
    )
    if (
        base_multiplier <= 0.0
        or memory_multiplier <= 0.0
        or confidence_multiplier <= 0.0
    ):
        raise ValueError('Temporal-memory learning-rate multipliers must be positive.')
    confidence_parameters = []
    if model.confidence_head_enabled:
        confidence_parameters = list(model.base.confidence_head.parameters())
    confidence_parameter_ids = {id(parameter) for parameter in confidence_parameters}
    if confidence_only_enabled:
        if not confidence_parameters:
            raise ValueError(
                'Confidence-only mode requires the confidence head to be enabled.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'confidence',
                    'params': confidence_parameters,
                    'lr': float(config.lr) * confidence_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    base_parameters = [
        parameter
        for parameter in model.base.parameters()
        if id(parameter) not in confidence_parameter_ids
    ]
    memory_parameters = list(model.forward_memory.parameters())
    memory_parameters += list(model.backward_memory.parameters())
    memory_parameters += list(model.memory_projection.parameters())
    parameter_groups = [
        {
            'name': 'base',
            'params': base_parameters,
            'lr': float(config.lr) * base_multiplier,
        },
    ]
    if confidence_parameters:
        parameter_groups.append(
            {
                'name': 'confidence',
                'params': confidence_parameters,
                'lr': float(config.lr) * confidence_multiplier,
            }
        )
    parameter_groups.append(
        {
            'name': 'memory',
            'params': memory_parameters,
            'lr': float(config.lr) * memory_multiplier,
        }
    )
    return optim.AdamW(parameter_groups, weight_decay=1e-4)


def memory_config_summary(config):
    return (
        'enabled (bin_size={}, context_bins={}, width={}, sequence_length={}, '
        'views_per_video={}, positive_frame_probability={}, '
        'target_positive_loss_mass={}, max_positive_weight={}, '
        'base_lr_multiplier={}, memory_lr_multiplier={}, '
        'confidence_lr_multiplier={}, '
        'attention_enabled={})'
    ).format(
        config.temporal_memory_bin_size,
        config.temporal_memory_context_bins,
        config.temporal_memory_width,
        config.temporal_memory_sequence_length,
        config.temporal_memory_train_views_per_video,
        config.temporal_memory_positive_frame_probability,
        config.temporal_memory_target_positive_loss_mass,
        config.temporal_memory_max_positive_weight,
        config.temporal_memory_base_lr_multiplier,
        config.temporal_memory_memory_lr_multiplier,
        getattr(config, 'temporal_memory_confidence_lr_multiplier', 1.0),
        bool(getattr(config, 'temporal_memory_temporal_attention_enabled', False)),
    )


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-memory training.')
    if not bool(cfg.temporal_memory_enabled):
        raise ValueError('Set TEMPORAL_MEMORY.temporal_memory_enabled=true.')
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError('TEMPORAL_MEMORY.context_bins must be odd.')
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError('TEMPORAL_MEMORY.sequence_length must exceed one.')
    if int(cfg.temporal_memory_train_workers) != 0 and bool(
        cfg.temporal_memory_cache_all_videos
    ):
        raise ValueError(
            'Use TEMPORAL_MEMORY.train_workers=0 when cache_all_videos=true.'
        )
    if int(cfg.epochs) <= 0:
        raise ValueError('TRAIN.epochs must be positive.')

    setup_seed(cfg.seed)
    device = torch.device('cuda:0')
    run_dir, started_at = create_run_directory(cfg)
    dataset = TemporalMemoryTrainDataset(
        root=Path(cfg.root) / 'train',
        whole_t=cfg.whole_t,
        temporal_bin_size=cfg.temporal_memory_bin_size,
        context_bins=cfg.temporal_memory_context_bins,
        sequence_length=cfg.temporal_memory_sequence_length,
        width=cfg.res[0],
        height=cfg.res[1],
        views_per_video=cfg.temporal_memory_train_views_per_video,
        positive_frame_probability=cfg.temporal_memory_positive_frame_probability,
        random_seed=cfg.seed,
        log_count_clip=cfg.temporal_memory_log_count_clip,
        cache_all_videos=cfg.temporal_memory_cache_all_videos,
        cache_video_count=cfg.temporal_memory_cache_video_count,
        dense_sampling_enabled=getattr(
            cfg,
            'temporal_memory_dense_sampling_enabled',
            False,
        ),
        dense_event_count_cutoff=getattr(
            cfg,
            'temporal_memory_dense_event_count_cutoff',
            200000,
        ),
        dense_view_multiplier=getattr(
            cfg,
            'temporal_memory_dense_view_multiplier',
            2,
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.temporal_memory_train_workers),
        collate_fn=temporal_memory_collate,
        pin_memory=True,
    )
    density_calibration_enabled = bool(
        getattr(cfg, 'temporal_frame_density_calibration_enabled', False)
    )
    trajectory_enabled = bool(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_enabled', False)
    )
    trajectory_weight = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_weight', 0.05)
    )
    trajectory_margin = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_margin_logit', 1.0)
    )
    trajectory_min_points = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_min_points', 3)
    )
    trajectory_warmup_epochs = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_warmup_epochs', 3)
    )
    if trajectory_enabled:
        if trajectory_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_weight must be positive.'
            )
        if trajectory_min_points < 2:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_min_points must be at least 2.'
            )
        if trajectory_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_warmup_epochs must be non-negative.'
            )
    metric_aux_enabled = bool(
        getattr(cfg, 'temporal_memory_metric_aux_enabled', False)
    )
    metric_target_weight = float(
        getattr(cfg, 'temporal_memory_metric_target_weight', 0.01)
    )
    metric_component_weight = float(
        getattr(cfg, 'temporal_memory_metric_component_weight', 0.002)
    )
    metric_warmup_epochs = int(
        getattr(cfg, 'temporal_memory_metric_warmup_epochs', 5)
    )
    metric_spatial_cell_size = int(
        getattr(cfg, 'temporal_memory_metric_spatial_cell_size', 3)
    )
    metric_min_cell_events = int(
        getattr(cfg, 'temporal_memory_metric_min_cell_events', 2)
    )
    metric_component_ratio = float(
        getattr(cfg, 'temporal_memory_metric_component_ratio', 0.01)
    )
    metric_activation_threshold = float(
        getattr(cfg, 'temporal_memory_metric_activation_threshold', 0.70)
    )
    metric_activation_temperature = float(
        getattr(cfg, 'temporal_memory_metric_activation_temperature', 0.10)
    )
    if metric_aux_enabled:
        if metric_target_weight < 0.0 or metric_component_weight < 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric target/component weights must be non-negative.'
            )
        if metric_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_warmup_epochs must be non-negative.'
            )
        if metric_spatial_cell_size <= 0 or metric_min_cell_events <= 0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric cell size and minimum events must be positive.'
            )
        if not 0.0 < metric_component_ratio <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_component_ratio must be in (0, 1].'
            )
        if not 0.0 < metric_activation_threshold < 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_activation_threshold must be in (0, 1).'
            )
        if metric_activation_temperature <= 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_activation_temperature must be positive.'
            )
    checkpoint_interval = int(getattr(cfg, 'checkpoint_interval', 0))
    if checkpoint_interval < 0:
        raise ValueError('TRAIN.checkpoint_interval must be non-negative.')
    confidence_head_enabled = bool(
        getattr(cfg, 'temporal_frame_confidence_head_enabled', False)
    )
    confidence_only_enabled = bool(
        getattr(cfg, 'temporal_memory_confidence_only_enabled', False)
    )
    if confidence_only_enabled and not confidence_head_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.confidence_only_enabled requires '
            'TEMPORAL_FRAME.confidence_head_enabled=true.'
        )
    confidence_calibration_weight = float(
        getattr(cfg, 'temporal_frame_confidence_calibration_weight', 0.1)
    )
    if confidence_head_enabled and confidence_calibration_weight <= 0.0:
        raise ValueError(
            'TEMPORAL_FRAME.confidence_calibration_weight must be positive.'
        )
    temporal_attention_enabled = bool(
        getattr(cfg, 'temporal_memory_temporal_attention_enabled', False)
    )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(cfg.temporal_memory_context_bins) * 2,
        width=int(cfg.temporal_memory_width),
        density_calibration_enabled=density_calibration_enabled,
        confidence_head_enabled=confidence_head_enabled,
        temporal_attention_enabled=temporal_attention_enabled,
    ).to(device)
    initialized_from = load_p23_base_weights(
        model,
        cfg.temporal_memory_init_model_path,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        density_calibration_enabled=density_calibration_enabled,
        confidence_head_enabled=confidence_head_enabled,
    )
    if confidence_only_enabled:
        confidence_parameter_ids = {
            id(parameter) for parameter in model.base.confidence_head.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in confidence_parameter_ids
    optimizer = build_optimizer(model, cfg, confidence_only_enabled)
    scheduler = build_scheduler(optimizer, cfg)

    print('random seed:{}'.format(cfg.seed))
    print('run directory:', run_dir)
    print('config overrides:', ', '.join(cfg.config_overrides) or '(none)')
    print('temporal-memory model:', memory_config_summary(cfg))
    print('training videos:', len(dataset.file_paths))
    print('training sequences per epoch:', len(dataset))
    print(
        'dense sequence sampling: enabled={}, cutoff={}, multiplier={}, '
        'dense_videos={}, extra_views={}'.format(
            dataset.dense_sampling_enabled,
            dataset.dense_event_count_cutoff,
            dataset.dense_view_multiplier,
            dataset.dense_video_count,
            dataset.extra_dense_views,
        )
    )
    if 'temporal_memory' in torch.load(initialized_from, map_location='cpu'):
        print('initialized full temporal-memory weights from:', initialized_from)
    else:
        print('initialized P23 base weights from:', initialized_from)
    print('learning-rate scheduler:', cfg.scheduler)
    if confidence_only_enabled:
        print('confidence calibration mode: backbone and memory frozen')

    best_loss = float('inf')
    best_epoch = None
    for epoch in range(int(cfg.epochs)):
        dataset.set_epoch(epoch)
        if confidence_only_enabled:
            # Keep the released M5 representation deterministic and train only
            # the newly attached head.
            model.eval()
            model.base.confidence_head.train()
        else:
            model.train()
        loss_sum = 0.0
        positive_fraction_sum = 0.0
        positive_weight_sum = 0.0
        trajectory_loss_sum = 0.0
        confidence_loss_sum = 0.0
        metric_target_loss_sum = 0.0
        metric_component_loss_sum = 0.0
        batch_count = 0
        pbar = tqdm.tqdm(
            dataloader,
            desc='Epoch: {}'.format(epoch),
            unit='Sequence',
            position=0,
            leave=True,
        )
        for batch in pbar:
            frames = batch['frames'].to(device, non_blocking=True).unsqueeze(0)
            event_time_indices = batch['event_time_indices'].to(
                device,
                non_blocking=True,
            )
            event_y = batch['event_y'].to(device, non_blocking=True)
            event_x = batch['event_x'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            model_output = model(frames)
            if confidence_head_enabled:
                logit_maps, confidence_logit_maps = model_output
                logit_maps = logit_maps.squeeze(0)
                confidence_logit_maps = confidence_logit_maps.squeeze(0)
            else:
                logit_maps = model_output.squeeze(0)
            event_logits = logit_maps[
                event_time_indices,
                0,
                event_y,
                event_x,
            ]
            loss, diagnostics = frame_balanced_event_bce(
                event_logits,
                labels,
                event_time_indices,
                target_positive_loss_mass=(
                    cfg.temporal_memory_target_positive_loss_mass
                ),
                max_positive_weight=cfg.temporal_memory_max_positive_weight,
            )
            confidence_loss = event_logits.sum() * 0.0
            if confidence_head_enabled:
                event_confidence_logits = confidence_logit_maps[
                    event_time_indices,
                    0,
                    event_y,
                    event_x,
                ]
                confidence_loss = confidence_calibration_loss(
                    event_confidence_logits,
                    event_logits,
                    labels,
                    hard_target=True,
                )
                loss = loss + confidence_calibration_weight * confidence_loss
            metric_target_loss = event_logits.sum() * 0.0
            metric_component_loss = event_logits.sum() * 0.0
            if metric_aux_enabled and epoch >= metric_warmup_epochs:
                event_scores = torch.sigmoid(event_logits)
                event_locations = torch.stack(
                    (
                        torch.zeros_like(event_x),
                        event_x,
                        event_y,
                        event_time_indices * int(cfg.temporal_memory_bin_size) + 1,
                    ),
                    dim=1,
                )
                if metric_target_weight > 0.0:
                    metric_target_loss, _, _ = target_frame_activation_loss(
                        event_scores,
                        labels,
                        target_ids,
                        event_locations,
                        int(cfg.temporal_memory_bin_size),
                        metric_activation_threshold,
                        metric_activation_temperature,
                    )
                    loss = loss + metric_target_weight * metric_target_loss
                if metric_component_weight > 0.0:
                    metric_component_loss, _, _ = component_hard_negative_loss(
                        event_scores,
                        labels,
                        event_locations,
                        metric_spatial_cell_size,
                        int(cfg.temporal_memory_bin_size),
                        metric_min_cell_events,
                        metric_component_ratio,
                        metric_activation_threshold,
                        metric_activation_temperature,
                    )
                    loss = loss + metric_component_weight * metric_component_loss
            trajectory_loss = event_logits.sum() * 0.0
            if trajectory_enabled and epoch >= trajectory_warmup_epochs:
                trajectory_loss, trajectory_stats = (
                    trajectory_extrapolation_loss_memory(
                        logit_maps,
                        event_time_indices,
                        event_x,
                        event_y,
                        labels,
                        target_ids,
                        min_known_points=trajectory_min_points,
                        margin_logit=trajectory_margin,
                    )
                )
                loss = loss + trajectory_weight * trajectory_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.detach().item())
            positive_fraction_sum += diagnostics['positive_fraction']
            positive_weight_sum += diagnostics['mean_positive_weight']
            trajectory_loss_sum += float(trajectory_loss.detach().item())
            confidence_loss_sum += float(confidence_loss.detach().item())
            metric_target_loss_sum += float(metric_target_loss.detach().item())
            metric_component_loss_sum += float(metric_component_loss.detach().item())
            batch_count += 1
            pbar.set_postfix(
                loss='{:.5f}'.format(loss_sum / batch_count),
                pos='{:.4f}'.format(positive_fraction_sum / batch_count),
                pos_w='{:.2f}'.format(positive_weight_sum / batch_count),
                traj='{:.5f}'.format(trajectory_loss_sum / batch_count),
                conf='{:.5f}'.format(confidence_loss_sum / batch_count),
                metric_pd='{:.5f}'.format(metric_target_loss_sum / batch_count),
                metric_fa='{:.5f}'.format(metric_component_loss_sum / batch_count),
            )
        pbar.close()
        scheduler.step()

        epoch_loss = loss_sum / max(batch_count, 1)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'temporal_memory': {
                'temporal_bin_size': int(cfg.temporal_memory_bin_size),
                'context_bins': int(cfg.temporal_memory_context_bins),
                'width': int(cfg.temporal_memory_width),
                'sequence_length': int(cfg.temporal_memory_sequence_length),
                'log_count_clip': float(cfg.temporal_memory_log_count_clip),
                'density_calibration_enabled': bool(
                    getattr(
                        cfg,
                        'temporal_frame_density_calibration_enabled',
                        False,
                    )
                ),
                'trajectory_extrapolation_enabled': trajectory_enabled,
                'confidence_head_enabled': confidence_head_enabled,
                'confidence_only_enabled': confidence_only_enabled,
                'temporal_attention_enabled': temporal_attention_enabled,
            },
        }
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint,
                run_dir / 'best_loss_seed{}.pt'.format(cfg.seed),
            )
        save_checkpoint(
            checkpoint, run_dir / 'last_seed{}.pt'.format(cfg.seed)
        )
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint,
                run_dir / 'epoch_{:03d}_seed{}.pt'.format(epoch + 1, cfg.seed),
            )
        learning_rates = ', '.join(
            'lr_{}={:.8f}'.format(group['name'], group['lr'])
            for group in optimizer.param_groups
        )
        print(
            'epoch {}: loss={:.6f}, {}, best_loss={:.6f}'.format(
                epoch,
                epoch_loss,
                learning_rates,
                best_loss,
            )
        )

    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'seed': int(cfg.seed),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'best_loss_checkpoint': str(
            run_dir / 'best_loss_seed{}.pt'.format(cfg.seed)
        ),
        'last_checkpoint': str(run_dir / 'last_seed{}.pt'.format(cfg.seed)),
        'config_overrides': list(cfg.config_overrides),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print('best loss checkpoint:', summary['best_loss_checkpoint'])
    print('last checkpoint:', summary['last_checkpoint'])

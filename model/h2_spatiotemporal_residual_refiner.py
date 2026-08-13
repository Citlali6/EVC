"""Frozen-M20 wrapper with a tiny zero-initialized logit residual head.

The refiner is deliberately separate from :mod:`model.temporal_memory_net`.
It consumes only inference-time tensors (M20 decoder features, M20 logits and
the centre-bin polarity/activity images) and never sees a source name, fold
identifier, label, target id or absolute time coordinate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiotemporalResidualHead(nn.Module):
    """A compact spatial/temporal residual head with exact identity at init."""

    def __init__(self, decoder_channels=16, hidden_channels=16):
        super().__init__()
        decoder_channels = int(decoder_channels)
        hidden_channels = int(hidden_channels)
        if decoder_channels <= 0 or hidden_channels <= 0:
            raise ValueError('decoder_channels and hidden_channels must be positive.')

        # decoder0 + base logit + centre negative/positive/activity images.
        input_channels = decoder_channels + 4
        self.decoder_channels = decoder_channels
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.input_projection = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, bias=True,
        )
        self.spatial_depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )
        self.spatial_mixing = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=1, bias=True,
        )
        self.temporal_depthwise = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 1, 1),
            padding=0,
            groups=hidden_channels,
            bias=True,
        )
        self.temporal_mixing = nn.Conv3d(
            hidden_channels, hidden_channels, kernel_size=1, bias=True,
        )
        self.output_projection = nn.Conv3d(
            hidden_channels, 1, kernel_size=1, bias=True,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, decoder_features, base_logits, centre_inputs):
        """Return residual logits for tensors shaped ``[B,T,C,H,W]``."""
        if decoder_features.ndim != 5:
            raise ValueError('decoder_features must have shape [B, T, C, H, W].')
        if base_logits.ndim != 5 or base_logits.shape[2] != 1:
            raise ValueError('base_logits must have shape [B, T, 1, H, W].')
        if centre_inputs.ndim != 5 or centre_inputs.shape[2] != 3:
            raise ValueError('centre_inputs must have shape [B, T, 3, H, W].')
        if decoder_features.shape[2] != self.decoder_channels:
            raise ValueError('Unexpected decoder feature channel count.')
        common = (
            decoder_features.shape[0],
            decoder_features.shape[1],
            decoder_features.shape[3],
            decoder_features.shape[4],
        )
        for name, tensor in (
            ('base_logits', base_logits),
            ('centre_inputs', centre_inputs),
        ):
            actual = (
                tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4]
            )
            if actual != common:
                raise ValueError('{} does not align with decoder_features.'.format(name))

        batch_size, sequence_length = decoder_features.shape[:2]
        merged = torch.cat((decoder_features, base_logits, centre_inputs), dim=2)
        flat = merged.reshape(
            batch_size * sequence_length,
            merged.shape[2],
            merged.shape[3],
            merged.shape[4],
        )
        spatial = F.silu(self.input_projection(flat), inplace=False)
        spatial = F.silu(
            self.spatial_mixing(self.spatial_depthwise(spatial)),
            inplace=False,
        )
        temporal = spatial.reshape(
            batch_size,
            sequence_length,
            self.hidden_channels,
            spatial.shape[2],
            spatial.shape[3],
        ).permute(0, 2, 1, 3, 4)
        # A one-bin halo is the complete temporal receptive field. Replication
        # makes standalone sequences deterministic; full-stream inference can
        # supply a real halo and crop it in the dedicated runner.
        temporal = F.pad(temporal, (0, 0, 0, 0, 1, 1), mode='replicate')
        temporal = F.silu(
            self.temporal_mixing(self.temporal_depthwise(temporal)),
            inplace=False,
        )
        residual = self.output_projection(temporal)
        return residual.permute(0, 2, 1, 3, 4)


class FrozenM20ResidualRefiner(nn.Module):
    """Wrap released M20 while keeping every inherited tensor frozen."""

    def __init__(self, released_m20, context_bins=5, hidden_channels=16):
        super().__init__()
        context_bins = int(context_bins)
        if context_bins < 1 or context_bins % 2 == 0:
            raise ValueError('context_bins must be a positive odd integer.')
        if int(released_m20.input_channels) != context_bins * 2:
            raise ValueError('M20 input channels do not match context_bins.')
        if bool(getattr(released_m20, 'confidence_head_enabled', False)):
            raise ValueError('The released-M20 residual wrapper expects no confidence head.')

        self.released_m20 = released_m20
        self.context_bins = context_bins
        decoder_channels = int(released_m20.base.head.in_channels)
        self.refiner = SpatiotemporalResidualHead(
            decoder_channels=decoder_channels,
            hidden_channels=hidden_channels,
        )
        for parameter in self.released_m20.parameters():
            parameter.requires_grad_(False)
        self.released_m20.eval()

    @property
    def input_channels(self):
        return self.released_m20.input_channels

    @property
    def confidence_head_enabled(self):
        return False

    def train(self, mode=True):
        super().train(mode)
        # ``super().train`` recurses into children; immediately restore M20's
        # inference behaviour so its running state can never drift.
        self.released_m20.eval()
        return self

    def trainable_parameters(self):
        return tuple(self.refiner.parameters())

    def frozen_state_dict(self):
        return self.released_m20.state_dict()

    def encode_bottleneck(self, frames):
        with torch.no_grad():
            return self.released_m20.encode_bottleneck(frames)

    def temporal_residual(self, bottlenecks):
        with torch.no_grad():
            return self.released_m20.temporal_residual(bottlenecks)

    def _decode_frozen_features(self, frames, memory_residual):
        """Reproduce M20 decoding and expose decoder0 without core edits."""
        with torch.no_grad():
            level0, level1, level2, bottleneck = self.released_m20._encode(frames)
            if memory_residual.shape != bottleneck.shape:
                raise ValueError('Temporal residual does not match bottleneck shape.')
            base = self.released_m20.base
            decoded2 = base.decoder2(bottleneck + memory_residual, level2)
            decoded1 = base.decoder1(decoded2, level1)
            decoded0 = base.decoder0(decoded1, level0)
            if base.density_calibration_enabled:
                decoded0 = base.density_calibrator(
                    decoded0, frames[:, :self.input_channels],
                )
            base_logits = base.head(decoded0)
        return decoded0.detach(), base_logits.detach()

    def _centre_inputs(self, frames):
        centre_start = (self.context_bins // 2) * 2
        negative = frames[:, centre_start:centre_start + 1]
        positive = frames[:, centre_start + 1:centre_start + 2]
        activity = negative + positive
        return torch.cat((negative, positive, activity), dim=1)

    def refine_decoded_sequence(self, frames, decoder_features, base_logits):
        """Refine aligned flattened frame tensors from one temporal sequence."""
        if frames.ndim != 4 or decoder_features.ndim != 4 or base_logits.ndim != 4:
            raise ValueError('frames, decoder_features and base_logits must be 4-D.')
        sequence_length = frames.shape[0]
        centre_inputs = self._centre_inputs(frames)
        residual = self.refiner(
            decoder_features.unsqueeze(0),
            base_logits.unsqueeze(0),
            centre_inputs.unsqueeze(0),
        ).squeeze(0)
        if residual.shape[0] != sequence_length:
            raise RuntimeError('Refiner changed the temporal sequence length.')
        return base_logits + residual

    def decode_with_residual(
        self,
        frames,
        residual,
        return_confidence_logits=False,
    ):
        if return_confidence_logits:
            raise ValueError('The released M20 checkpoint has no confidence head.')
        decoder_features, base_logits = self._decode_frozen_features(frames, residual)
        return self.refine_decoded_sequence(frames, decoder_features, base_logits)

    def frozen_forward_parts(self, frames):
        """Return detached M20 decoder features/logits for ``[B,T,C,H,W]``."""
        if frames.ndim != 5:
            raise ValueError('frames must have shape [B, T, C, H, W].')
        batch_size, sequence_length = frames.shape[:2]
        flat = frames.reshape(
            batch_size * sequence_length,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        with torch.no_grad():
            level0, level1, level2, bottleneck = self.released_m20._encode(flat)
            shaped = bottleneck.reshape(
                batch_size,
                sequence_length,
                bottleneck.shape[1],
                bottleneck.shape[2],
                bottleneck.shape[3],
            )
            memory = self.released_m20._memory_residual(shaped).reshape_as(bottleneck)
            base = self.released_m20.base
            decoded2 = base.decoder2(bottleneck + memory, level2)
            decoded1 = base.decoder1(decoded2, level1)
            decoded0 = base.decoder0(decoded1, level0)
            if base.density_calibration_enabled:
                decoded0 = base.density_calibrator(
                    decoded0, flat[:, :self.input_channels],
                )
            base_logits = base.head(decoded0)
        decoder_features = decoded0.detach().reshape(
            batch_size,
            sequence_length,
            decoded0.shape[1],
            decoded0.shape[2],
            decoded0.shape[3],
        )
        base_logits = base_logits.detach().reshape(
            batch_size,
            sequence_length,
            1,
            base_logits.shape[2],
            base_logits.shape[3],
        )
        return decoder_features, base_logits

    def forward(self, frames, return_base_logits=False):
        decoder_features, base_logits = self.frozen_forward_parts(frames)
        flat_frames = frames.reshape(
            frames.shape[0] * frames.shape[1],
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        centre_inputs = self._centre_inputs(flat_frames).reshape(
            frames.shape[0],
            frames.shape[1],
            3,
            frames.shape[3],
            frames.shape[4],
        )
        residual = self.refiner(decoder_features, base_logits, centre_inputs)
        refined = base_logits + residual
        if return_base_logits:
            return refined, base_logits
        return refined


def refiner_parameter_count(model):
    """Return the number of trainable residual-head scalars."""
    return sum(parameter.numel() for parameter in model.refiner.parameters())


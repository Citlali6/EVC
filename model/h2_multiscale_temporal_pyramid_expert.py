"""Long-context multi-scale pyramid expert on frozen released-M20 features.

Unlike the earlier three-bin residual heads, this module first constructs a
spatially aligned, low-resolution summary of the complete temporal stream.
Fixed 16/32/64/full-stream moments are encoded by one shared scale encoder and
mixed adaptively before a zero-initialized dense projection.  The released
M20 remains frozen and can be decoded in streaming chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


DECODER_CHANNELS = 16
CENTRE_INPUT_CHANNELS = 3
OBSERVATION_CHANNELS = DECODER_CHANNELS + 1 + CENTRE_INPUT_CHANNELS
TEMPORAL_SCALES = (16, 32, 64, 160)
SPATIAL_DOWNSAMPLE = 8
CONTEXT_CHANNELS = 16


@dataclass(frozen=True)
class PyramidExpertOutput:
    refined_logits: torch.Tensor
    correction: torch.Tensor
    mixture_weights: torch.Tensor
    low_resolution_context: torch.Tensor


def _validate_aligned_dense(decoder_features, base_logits, centre_inputs):
    if decoder_features.ndim != 5 or decoder_features.shape[2] != DECODER_CHANNELS:
        raise ValueError("decoder_features must be [B,T,16,H,W]")
    if base_logits.ndim != 5 or base_logits.shape[2] != 1:
        raise ValueError("base_logits must be [B,T,1,H,W]")
    if centre_inputs.ndim != 5 or centre_inputs.shape[2] != CENTRE_INPUT_CHANNELS:
        raise ValueError("centre_inputs must be [B,T,3,H,W]")
    common = (
        decoder_features.shape[0],
        decoder_features.shape[1],
        decoder_features.shape[3],
        decoder_features.shape[4],
    )
    for name, tensor in (("base_logits", base_logits), ("centre_inputs", centre_inputs)):
        actual = (tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4])
        if actual != common:
            raise ValueError("{} does not align with decoder features".format(name))


def downsample_frozen_observations(
    decoder_features,
    base_logits,
    centre_inputs,
    spatial_downsample=SPATIAL_DOWNSAMPLE,
):
    """Stateless spatial pooling suitable for a first streaming decode pass."""

    _validate_aligned_dense(decoder_features, base_logits, centre_inputs)
    spatial_downsample = int(spatial_downsample)
    if spatial_downsample <= 0:
        raise ValueError("spatial_downsample must be positive")
    merged = torch.cat((decoder_features, base_logits, centre_inputs), dim=2)
    batch, time, channels, height, width = merged.shape
    output_height = int(math.ceil(height / spatial_downsample))
    output_width = int(math.ceil(width / spatial_downsample))
    pooled = F.adaptive_avg_pool2d(
        merged.reshape(batch * time, channels, height, width),
        (output_height, output_width),
    )
    return pooled.reshape(batch, time, channels, output_height, output_width)


def fixed_multiscale_temporal_moments(observations, scales=TEMPORAL_SCALES):
    """Return mean/std maps for centered windows and one complete-stream scale.

    Window edge counts are normalized exactly.  A requested scale greater than
    or equal to the observed sequence length is the global full-stream summary,
    repeated at every time step.
    """

    if observations.ndim != 5 or observations.shape[1] <= 0:
        raise ValueError("observations must be non-empty [B,T,C,h,w]")
    scales = tuple(int(value) for value in scales)
    if not scales or any(value <= 0 for value in scales) or tuple(sorted(scales)) != scales:
        raise ValueError("temporal scales must be positive and increasing")
    work = observations.float()
    batch, time, channels, height, width = work.shape
    prefix = torch.cat(
        (work.new_zeros((batch, 1, channels, height, width)), work.cumsum(dim=1)),
        dim=1,
    )
    square_prefix = torch.cat(
        (
            work.new_zeros((batch, 1, channels, height, width)),
            work.square().cumsum(dim=1),
        ),
        dim=1,
    )
    position = torch.arange(time, device=work.device)
    summaries = []
    for scale in scales:
        if scale >= time:
            starts = torch.zeros(time, dtype=torch.long, device=work.device)
            stops = torch.full((time,), time, dtype=torch.long, device=work.device)
        else:
            left = scale // 2
            right = scale - left
            starts = (position - left).clamp(min=0)
            stops = (position + right).clamp(max=time)
        counts = (stops - starts).to(dtype=work.dtype).view(1, time, 1, 1, 1)
        means = (prefix[:, stops] - prefix[:, starts]) / counts
        second = (square_prefix[:, stops] - square_prefix[:, starts]) / counts
        standard_deviation = torch.sqrt((second - means.square()).clamp_min(0.0))
        summaries.append(torch.cat((means, standard_deviation), dim=2))
    return tuple(summaries)


class SharedScaleEncoder(nn.Module):
    def __init__(self, input_channels=2 * OBSERVATION_CHANNELS, width=CONTEXT_CHANNELS):
        super().__init__()
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.input_projection = nn.Conv2d(self.input_channels, self.width, 1)
        self.depthwise = nn.Conv2d(
            self.width, self.width, 3, padding=1, groups=self.width
        )
        self.output_projection = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.GroupNorm(4, self.width)

    def forward(self, values):
        if values.ndim != 4 or values.shape[1] != self.input_channels:
            raise ValueError("scale summary channel mismatch")
        values = F.silu(self.input_projection(values), inplace=False)
        values = self.output_projection(self.depthwise(values))
        return F.silu(self.norm(values), inplace=False)


class MultiScaleTemporalPyramidHead(nn.Module):
    """Shared-scale mixture with an exactly zero dense output at initialization."""

    def __init__(
        self,
        decoder_channels=DECODER_CHANNELS,
        context_channels=CONTEXT_CHANNELS,
        scales=TEMPORAL_SCALES,
        spatial_downsample=SPATIAL_DOWNSAMPLE,
    ):
        super().__init__()
        if int(decoder_channels) != DECODER_CHANNELS:
            raise ValueError("released M20 decoder0 must have 16 channels")
        self.context_channels = int(context_channels)
        self.scales = tuple(int(value) for value in scales)
        self.spatial_downsample = int(spatial_downsample)
        if self.context_channels <= 0 or self.context_channels % 4:
            raise ValueError("context_channels must be a positive multiple of four")
        self.scale_encoder = SharedScaleEncoder(
            2 * OBSERVATION_CHANNELS, self.context_channels
        )
        self.scale_tokens = nn.Parameter(
            torch.zeros(len(self.scales), self.context_channels)
        )
        nn.init.normal_(self.scale_tokens, mean=0.0, std=0.02)
        self.mixture_query = nn.Sequential(
            nn.Conv2d(OBSERVATION_CHANNELS, self.context_channels, 1),
            nn.SiLU(),
        )
        self.mixture_projection = nn.Conv2d(
            self.context_channels, len(self.scales), 1
        )
        nn.init.zeros_(self.mixture_projection.weight)
        nn.init.zeros_(self.mixture_projection.bias)
        self.context_refine = nn.Sequential(
            nn.Conv2d(
                self.context_channels,
                self.context_channels,
                3,
                padding=1,
                groups=self.context_channels,
            ),
            nn.Conv2d(self.context_channels, self.context_channels, 1),
            nn.SiLU(),
        )
        self.local_encoder = nn.Sequential(
            nn.Conv2d(OBSERVATION_CHANNELS, self.context_channels, 1),
            nn.SiLU(),
            nn.Conv2d(
                self.context_channels,
                self.context_channels,
                3,
                padding=1,
                groups=self.context_channels,
            ),
            nn.Conv2d(self.context_channels, self.context_channels, 1),
            nn.GroupNorm(4, self.context_channels),
            nn.SiLU(),
        )
        self.context_to_film = nn.Conv2d(
            self.context_channels, 2 * self.context_channels, 1
        )
        self.output_projection = nn.Conv2d(self.context_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        decoder_features,
        base_logits,
        centre_inputs,
        multiscale_summaries,
        return_parts=False,
    ):
        _validate_aligned_dense(decoder_features, base_logits, centre_inputs)
        summaries = tuple(multiscale_summaries)
        if len(summaries) != len(self.scales):
            raise ValueError("one temporal summary is required for every frozen scale")
        batch, time, _, height, width = decoder_features.shape
        observations = downsample_frozen_observations(
            decoder_features,
            base_logits,
            centre_inputs,
            self.spatial_downsample,
        )
        low_height, low_width = observations.shape[-2:]
        encoded_scales = []
        for scale_index, summary in enumerate(summaries):
            if summary.shape != (
                batch,
                time,
                2 * OBSERVATION_CHANNELS,
                low_height,
                low_width,
            ):
                raise ValueError("temporal summary shape mismatch")
            encoded = self.scale_encoder(
                summary.reshape(
                    batch * time,
                    summary.shape[2],
                    low_height,
                    low_width,
                )
            )
            encoded = encoded + self.scale_tokens[scale_index].view(
                1, self.context_channels, 1, 1
            )
            encoded_scales.append(encoded)
        stacked = torch.stack(encoded_scales, dim=1)
        query = self.mixture_query(
            observations.reshape(
                batch * time,
                OBSERVATION_CHANNELS,
                low_height,
                low_width,
            )
        )
        mixture_weights = torch.softmax(self.mixture_projection(query), dim=1)
        context = (stacked * mixture_weights[:, :, None]).sum(dim=1)
        context = self.context_refine(context)
        film = self.context_to_film(context)
        film = F.interpolate(
            film, size=(height, width), mode="bilinear", align_corners=False
        )
        gamma, beta = film.chunk(2, dim=1)
        dense_observation = torch.cat(
            (decoder_features, base_logits, centre_inputs), dim=2
        ).reshape(batch * time, OBSERVATION_CHANNELS, height, width)
        local = self.local_encoder(dense_observation)
        fused = local * (1.0 + torch.tanh(gamma)) + beta
        correction = self.output_projection(F.silu(fused, inplace=False)).reshape(
            batch, time, 1, height, width
        )
        refined = base_logits + correction
        if return_parts:
            return PyramidExpertOutput(
                refined_logits=refined,
                correction=correction,
                mixture_weights=mixture_weights.reshape(
                    batch, time, len(self.scales), low_height, low_width
                ),
                low_resolution_context=context.reshape(
                    batch, time, self.context_channels, low_height, low_width
                ),
            )
        return refined


class FrozenM20MultiScalePyramidAdapter(nn.Module):
    """Feature-only adapter that never updates or changes released M20."""

    def __init__(self, released_m20, context_bins=5):
        super().__init__()
        self.released_m20 = released_m20
        self.context_bins = int(context_bins)
        if self.context_bins < 1 or self.context_bins % 2 == 0:
            raise ValueError("context_bins must be a positive odd integer")
        audit_released_m20_feature_api(released_m20, self.context_bins)
        self.expert = MultiScaleTemporalPyramidHead()
        for parameter in self.released_m20.parameters():
            parameter.requires_grad_(False)
        self.released_m20.eval()

    def train(self, mode=True):
        super().train(mode)
        self.released_m20.eval()
        return self

    def trainable_parameters(self):
        return tuple(self.expert.parameters())

    def _centre_inputs(self, frames):
        start = (self.context_bins // 2) * 2
        negative = frames[:, start : start + 1]
        positive = frames[:, start + 1 : start + 2]
        return torch.cat((negative, positive, negative + positive), dim=1)

    def decode_frozen_features(self, frames, memory_residual):
        """Expose detached decoder0/logits through the audited M20 API."""

        with torch.no_grad():
            level0, level1, level2, bottleneck = self.released_m20._encode(frames)
            if memory_residual.shape != bottleneck.shape:
                raise ValueError("M20 memory residual does not align with bottleneck")
            base = self.released_m20.base
            decoded2 = base.decoder2(bottleneck + memory_residual, level2)
            decoded1 = base.decoder1(decoded2, level1)
            decoder0 = base.decoder0(decoded1, level0)
            if base.density_calibration_enabled:
                decoder0 = base.density_calibrator(
                    decoder0, frames[:, : self.released_m20.input_channels]
                )
            logits = base.head(decoder0)
        return decoder0.detach(), logits.detach(), self._centre_inputs(frames).detach()

    def observe_chunk(self, frames, memory_residual):
        decoder, logits, centre = self.decode_frozen_features(frames, memory_residual)
        observations = downsample_frozen_observations(
            decoder.unsqueeze(0), logits.unsqueeze(0), centre.unsqueeze(0)
        ).squeeze(0)
        return decoder, logits, centre, observations.detach()

    def build_full_stream_pyramid(self, observations):
        if observations.ndim != 4:
            raise ValueError("full-stream observations must be [T,C,h,w]")
        with torch.no_grad():
            return tuple(
                value.squeeze(0)
                for value in fixed_multiscale_temporal_moments(
                    observations.unsqueeze(0), self.expert.scales
                )
            )

    def refine_chunk(self, decoder, base_logits, centre, summaries):
        return self.expert(
            decoder.unsqueeze(0),
            base_logits.unsqueeze(0),
            centre.unsqueeze(0),
            tuple(value.unsqueeze(0) for value in summaries),
        ).squeeze(0)


def audit_released_m20_feature_api(model, context_bins=5):
    """Fail closed if the frozen feature interface no longer matches M20."""

    required_model = ("_encode", "encode_bottleneck", "temporal_residual", "base")
    missing = [name for name in required_model if not hasattr(model, name)]
    if missing:
        raise RuntimeError("released M20 feature API is missing {}".format(missing))
    required_base = ("decoder2", "decoder1", "decoder0", "head")
    missing_base = [name for name in required_base if not hasattr(model.base, name)]
    if missing_base:
        raise RuntimeError("released M20 decoder API is missing {}".format(missing_base))
    if int(model.input_channels) != int(context_bins) * 2:
        raise RuntimeError("released M20 input/context channel contract changed")
    if int(model.base.head.in_channels) != DECODER_CHANNELS:
        raise RuntimeError("released M20 decoder0 width changed")
    if bool(getattr(model, "confidence_head_enabled", False)):
        raise RuntimeError("pyramid adapter expects released M20 without confidence head")
    return {
        "input_channels": int(model.input_channels),
        "decoder0_channels": int(model.base.head.in_channels),
        "density_calibration_enabled": bool(model.base.density_calibration_enabled),
        "confidence_head_enabled": False,
        "feature_tensors": ["decoder0", "base_logit", "centre_negative", "centre_positive", "centre_activity"],
    }


def pyramid_expert_parameter_count(model):
    module = model.expert if hasattr(model, "expert") else model
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


__all__ = (
    "CENTRE_INPUT_CHANNELS",
    "CONTEXT_CHANNELS",
    "DECODER_CHANNELS",
    "OBSERVATION_CHANNELS",
    "PyramidExpertOutput",
    "SPATIAL_DOWNSAMPLE",
    "TEMPORAL_SCALES",
    "FrozenM20MultiScalePyramidAdapter",
    "MultiScaleTemporalPyramidHead",
    "audit_released_m20_feature_api",
    "downsample_frozen_observations",
    "fixed_multiscale_temporal_moments",
    "pyramid_expert_parameter_count",
)

"""Activity-first component scorer for atomic H2 deletion V3.

This network never produces an event or pixel score.  It maps a variable-length
sequence of recentered component patches to exactly one pure-false-positive
logit.  The caller remains solely responsible for the all-or-nothing component
action and for bitwise preservation of retained M20+C00 scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DECODER_CHANNELS = 16
# decoder0[16], base logit, negative polarity, positive polarity, activity,
# component query mask.
PATCH_CHANNELS = DECODER_CHANNELS + 5


class ActivityFirstComponentScorer(nn.Module):
    """Return one pure-FP logit per padded component-patch sequence."""

    def __init__(
        self,
        decoder_channels: int = DECODER_CHANNELS,
        activity_width: int = 16,
        semantic_width: int = 16,
        temporal_width: int = 32,
    ):
        super().__init__()
        decoder_channels = int(decoder_channels)
        activity_width = int(activity_width)
        semantic_width = int(semantic_width)
        temporal_width = int(temporal_width)
        if min(decoder_channels, activity_width, semantic_width, temporal_width) <= 0:
            raise ValueError("all component-scorer widths must be positive")
        if activity_width % 4 or semantic_width % 4:
            raise ValueError("spatial widths must be divisible by four")
        self.decoder_channels = decoder_channels
        self.patch_channels = decoder_channels + 5
        self.activity_width = activity_width
        self.semantic_width = semantic_width
        self.temporal_width = temporal_width

        # Activity/polarity is deliberately its own adapter: the motivating G1
        # evidence showed that activity-only ranking removed substantial noise.
        # Its output is an embedding only and can never attenuate an event.
        self.activity_adapter = nn.Sequential(
            nn.Conv2d(4, activity_width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, activity_width),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                activity_width,
                activity_width,
                kernel_size=3,
                padding=1,
                groups=activity_width,
                bias=False,
            ),
            nn.Conv2d(activity_width, activity_width, kernel_size=1, bias=False),
            nn.GroupNorm(4, activity_width),
            nn.SiLU(inplace=False),
        )
        # Frozen-M20 decoder/logit patches provide semantic context to the
        # component ranker.  They remain features, never replacement scores.
        self.semantic_adapter = nn.Sequential(
            nn.Conv2d(
                decoder_channels + 2,
                semantic_width,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(4, semantic_width),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                semantic_width,
                semantic_width,
                kernel_size=3,
                padding=1,
                groups=semantic_width,
                bias=False,
            ),
            nn.Conv2d(semantic_width, semantic_width, kernel_size=1, bias=False),
            nn.GroupNorm(4, semantic_width),
            nn.SiLU(inplace=False),
        )

        per_bin_width = 3 * (activity_width + semantic_width)
        self.bin_projection = nn.Sequential(
            nn.Linear(per_bin_width, temporal_width),
            nn.LayerNorm(temporal_width),
            nn.SiLU(inplace=False),
        )
        self.temporal = nn.GRU(
            input_size=temporal_width,
            hidden_size=temporal_width,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * temporal_width, temporal_width),
            nn.SiLU(inplace=False),
            nn.Linear(temporal_width, 1),
        )

    def _masked_and_context_pool(self, features, mask):
        mask = mask.to(dtype=features.dtype)
        masked = (features * mask).sum(dim=(-2, -1)) / mask.sum(
            dim=(-2, -1)
        ).clamp_min(1.0)
        mean = features.mean(dim=(-2, -1))
        maximum = torch.amax(features, dim=(-2, -1))
        return torch.cat((masked, mean, maximum), dim=1)

    def component_embeddings(self, patches, lengths):
        """Return activity-only and fused embeddings for immutable auditing."""
        if patches.ndim != 5:
            raise ValueError("patches must have shape [B,L,C,H,W]")
        if patches.shape[2] != self.patch_channels:
            raise ValueError(
                "patch channel count is {}, expected {}".format(
                    patches.shape[2], self.patch_channels
                )
            )
        lengths = torch.as_tensor(lengths, device=patches.device, dtype=torch.long)
        if lengths.ndim != 1 or lengths.numel() != patches.shape[0]:
            raise ValueError("lengths must have shape [B]")
        if bool(torch.any(lengths <= 0)) or bool(torch.any(lengths > patches.shape[1])):
            raise ValueError("component lengths must lie in [1,L]")
        if not bool(torch.isfinite(patches).all()):
            raise ValueError("component patches must be finite")

        batch, sequence, channels, height, width = patches.shape
        flat = patches.reshape(batch * sequence, channels, height, width)
        decoder = flat[:, : self.decoder_channels]
        base_logit = flat[:, self.decoder_channels : self.decoder_channels + 1]
        negative = flat[:, self.decoder_channels + 1 : self.decoder_channels + 2]
        positive = flat[:, self.decoder_channels + 2 : self.decoder_channels + 3]
        activity = flat[:, self.decoder_channels + 3 : self.decoder_channels + 4]
        component_mask = flat[:, self.decoder_channels + 4 : self.decoder_channels + 5]
        if bool(torch.any(component_mask < 0.0)) or bool(torch.any(component_mask > 1.0)):
            raise ValueError("component mask must lie in [0,1]")
        if bool(torch.any(component_mask.sum(dim=(-2, -1)) <= 0.0)):
            raise ValueError("every valid or padded patch must carry a nonempty mask")

        activity_features = self.activity_adapter(
            torch.cat((negative, positive, activity, component_mask), dim=1)
        )
        semantic_features = self.semantic_adapter(
            torch.cat((decoder, base_logit, component_mask), dim=1)
        )
        activity_per_bin = self._masked_and_context_pool(
            activity_features, component_mask
        )
        semantic_per_bin = self._masked_and_context_pool(
            semantic_features, component_mask
        )
        pooled = torch.cat((activity_per_bin, semantic_per_bin), dim=1)
        per_bin = self.bin_projection(pooled).reshape(batch, sequence, -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            per_bin,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal(packed)
        temporal_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence,
        )
        valid = (
            torch.arange(sequence, device=patches.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        valid_float = valid.unsqueeze(-1).to(temporal_output.dtype)
        temporal_mean = (temporal_output * valid_float).sum(dim=1) / lengths.unsqueeze(
            1
        ).to(temporal_output.dtype)
        floor = torch.finfo(temporal_output.dtype).min
        temporal_max = temporal_output.masked_fill(~valid.unsqueeze(-1), floor).amax(dim=1)
        fused_embedding = torch.cat((temporal_mean, temporal_max), dim=1)

        activity_per_bin = activity_per_bin.reshape(batch, sequence, -1)
        activity_mean = (activity_per_bin * valid_float).sum(dim=1) / lengths.unsqueeze(
            1
        ).to(activity_per_bin.dtype)
        activity_max = activity_per_bin.masked_fill(
            ~valid.unsqueeze(-1), torch.finfo(activity_per_bin.dtype).min
        ).amax(dim=1)
        activity_embedding = torch.cat((activity_mean, activity_max), dim=1)
        return activity_embedding, fused_embedding

    def forward(self, patches, lengths):
        """Score ``patches[B,L,C,H,W]`` with valid lengths ``[B]``."""

        _, fused_embedding = self.component_embeddings(patches, lengths)
        logits = self.classifier(fused_embedding)
        return logits.squeeze(1)


def balanced_component_bce(logits, pure_fp_targets, sample_weights):
    """Weighted BCE; all weights must be derived from the current fit group."""

    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(
        pure_fp_targets, device=logits.device, dtype=logits.dtype
    )
    weights = torch.as_tensor(sample_weights, device=logits.device, dtype=logits.dtype)
    if logits.ndim != 1 or targets.shape != logits.shape or weights.shape != logits.shape:
        raise ValueError("logits, targets and sample_weights must align as vectors")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("logits and sample weights must be finite")
    if bool(torch.any(weights <= 0.0)):
        raise ValueError("sample weights must be positive")
    return (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        * weights
    ).sum() / weights.sum()


def component_scorer_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = (
    "ActivityFirstComponentScorer",
    "DECODER_CHANNELS",
    "PATCH_CHANNELS",
    "balanced_component_bce",
    "component_scorer_parameter_count",
)

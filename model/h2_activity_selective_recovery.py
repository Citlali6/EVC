"""Small component recovery head for the H2 activity-suppression pipeline.

The network never emits event scores.  It ranks only complete M20 components
that are partly absent from the post-C00 activity output.  The caller alone may
apply one whole-component recovery action after fit-only calibration.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


M20_DECODER_CHANNELS = 16
ACTIVITY_DECODER_CHANNELS = 16
SEMANTIC_PATCH_CHANNELS = 35
CONTEXT_PATCH_CHANNELS = 22
RECOVERY_PATCH_CHANNELS = SEMANTIC_PATCH_CHANNELS + CONTEXT_PATCH_CHANNELS
TRAJECTORY_FEATURES = 8


def _groups(channels):
    channels = int(channels)
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class DisagreementRecoveryNet(nn.Module):
    """Rank recovery utility from semantic delta, activity/polarity, and track context."""

    def __init__(
        self,
        semantic_hidden=12,
        context_hidden=8,
        spatial_hidden=16,
        trajectory_hidden=8,
        recurrent_hidden=16,
    ):
        super().__init__()
        semantic_hidden = int(semantic_hidden)
        context_hidden = int(context_hidden)
        spatial_hidden = int(spatial_hidden)
        trajectory_hidden = int(trajectory_hidden)
        recurrent_hidden = int(recurrent_hidden)
        if min(
            semantic_hidden,
            context_hidden,
            spatial_hidden,
            trajectory_hidden,
            recurrent_hidden,
        ) <= 0:
            raise ValueError("all hidden dimensions must be positive")

        # Channels 0:35 contain M20/activity decoder0 maps plus their logits
        # and logit delta.  Channels 35:57 contain the ten raw polarity frames,
        # five activity maps, five signed-polarity maps, and two binary masks.
        self.semantic_stem = nn.Sequential(
            nn.Conv2d(
                SEMANTIC_PATCH_CHANNELS,
                semantic_hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(semantic_hidden), semantic_hidden),
            nn.SiLU(inplace=False),
        )
        self.context_stem = nn.Sequential(
            nn.Conv2d(
                CONTEXT_PATCH_CHANNELS,
                context_hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(context_hidden), context_hidden),
            nn.SiLU(inplace=False),
        )
        merged = semantic_hidden + context_hidden
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(merged, spatial_hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(spatial_hidden), spatial_hidden),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                spatial_hidden,
                spatial_hidden,
                kernel_size=3,
                padding=1,
                groups=spatial_hidden,
                bias=False,
            ),
            nn.GroupNorm(_groups(spatial_hidden), spatial_hidden),
            nn.SiLU(inplace=False),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(TRAJECTORY_FEATURES, trajectory_hidden),
            nn.LayerNorm(trajectory_hidden),
            nn.SiLU(inplace=False),
        )
        token_channels = spatial_hidden * 2 + trajectory_hidden
        self.temporal = nn.GRU(
            token_channels,
            recurrent_hidden,
            batch_first=True,
            bidirectional=True,
        )
        temporal_channels = recurrent_hidden * 2
        self.temporal_attention = nn.Linear(temporal_channels, 1)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_channels * 2, recurrent_hidden),
            nn.SiLU(inplace=False),
            nn.Linear(recurrent_hidden, 1),
        )

    def component_embeddings(self, patches, trajectory, lengths):
        if patches.ndim != 5:
            raise ValueError("patches must have shape [B,L,C,H,W]")
        if int(patches.shape[2]) != RECOVERY_PATCH_CHANNELS:
            raise ValueError("unexpected recovery patch channel count")
        if trajectory.ndim != 3 or trajectory.shape[:2] != patches.shape[:2]:
            raise ValueError("trajectory must align with [B,L]")
        if int(trajectory.shape[2]) != TRAJECTORY_FEATURES:
            raise ValueError("unexpected trajectory feature count")
        if lengths.ndim != 1 or int(lengths.shape[0]) != int(patches.shape[0]):
            raise ValueError("lengths must have shape [B]")
        if torch.any(lengths <= 0) or torch.any(lengths > patches.shape[1]):
            raise ValueError("sequence lengths are out of bounds")

        batch, sequence, channels, height, width = patches.shape
        flat = patches.reshape(batch * sequence, channels, height, width)
        semantic = self.semantic_stem(flat[:, :SEMANTIC_PATCH_CHANNELS])
        context = self.context_stem(flat[:, SEMANTIC_PATCH_CHANNELS:])
        spatial = self.spatial_fusion(torch.cat((semantic, context), dim=1))
        spatial_average = functional.adaptive_avg_pool2d(spatial, 1).flatten(1)
        spatial_maximum = functional.adaptive_max_pool2d(spatial, 1).flatten(1)
        spatial_token = torch.cat((spatial_average, spatial_maximum), dim=1).reshape(
            batch, sequence, -1
        )
        trajectory_token = self.trajectory_encoder(trajectory)
        tokens = torch.cat((spatial_token, trajectory_token), dim=2)

        packed = nn.utils.rnn.pack_padded_sequence(
            tokens,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal(packed)
        temporal, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence,
        )
        positions = torch.arange(sequence, device=lengths.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        attention_logits = self.temporal_attention(temporal).squeeze(2)
        attention_logits = attention_logits.masked_fill(~valid, float("-inf"))
        attention = torch.softmax(attention_logits, dim=1)
        attended = torch.sum(temporal * attention.unsqueeze(2), dim=1)
        maximum = temporal.masked_fill(~valid.unsqueeze(2), float("-inf")).amax(dim=1)
        embedding = torch.cat((attended, maximum), dim=1)
        if not torch.isfinite(embedding).all():
            raise RuntimeError("recovery embedding is non-finite")
        return embedding, attention

    def forward(self, patches, trajectory, lengths, return_embedding=False):
        embedding, attention = self.component_embeddings(patches, trajectory, lengths)
        logits = self.classifier(embedding).squeeze(1)
        if return_embedding:
            return logits, embedding, attention
        return logits


def recovery_sequence_collate(items):
    if not items:
        raise ValueError("component batch must not be empty")
    lengths = torch.as_tensor(
        [int(item["patches"].shape[0]) for item in items], dtype=torch.long
    )
    if torch.any(lengths <= 0):
        raise ValueError("component sequences must not be empty")
    maximum = int(lengths.max().item())
    first_patches = torch.as_tensor(items[0]["patches"], dtype=torch.float32)
    first_trajectory = torch.as_tensor(items[0]["trajectory"], dtype=torch.float32)
    if first_patches.ndim != 4 or first_patches.shape[1] != RECOVERY_PATCH_CHANNELS:
        raise ValueError("item patches must have shape [L,57,H,W]")
    if first_trajectory.ndim != 2 or first_trajectory.shape[1] != TRAJECTORY_FEATURES:
        raise ValueError("item trajectory must have shape [L,8]")
    patches = torch.zeros(
        len(items),
        maximum,
        RECOVERY_PATCH_CHANNELS,
        first_patches.shape[2],
        first_patches.shape[3],
        dtype=torch.float32,
    )
    trajectory = torch.zeros(
        len(items), maximum, TRAJECTORY_FEATURES, dtype=torch.float32
    )
    targets = torch.empty(len(items), dtype=torch.float32)
    weights = torch.empty(len(items), dtype=torch.float32)
    for index, item in enumerate(items):
        item_patches = torch.as_tensor(item["patches"], dtype=torch.float32)
        item_trajectory = torch.as_tensor(item["trajectory"], dtype=torch.float32)
        length = int(lengths[index].item())
        if (
            item_patches.shape
            != (length, RECOVERY_PATCH_CHANNELS, patches.shape[3], patches.shape[4])
            or item_trajectory.shape != (length, TRAJECTORY_FEATURES)
            or not torch.isfinite(item_patches).all()
            or not torch.isfinite(item_trajectory).all()
        ):
            raise ValueError("component item shapes or values differ")
        patches[index, :length] = item_patches
        trajectory[index, :length] = item_trajectory
        targets[index] = float(item.get("target", 0.0))
        weights[index] = float(item.get("weight", 1.0))
    if not torch.isfinite(targets).all() or not torch.isfinite(weights).all():
        raise ValueError("targets and weights must be finite")
    return {
        "patches": patches,
        "trajectory": trajectory,
        "lengths": lengths,
        "targets": targets,
        "weights": weights,
    }


def recovery_parameter_count(model=None):
    model = DisagreementRecoveryNet() if model is None else model
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = (
    "ACTIVITY_DECODER_CHANNELS",
    "CONTEXT_PATCH_CHANNELS",
    "DisagreementRecoveryNet",
    "M20_DECODER_CHANNELS",
    "RECOVERY_PATCH_CHANNELS",
    "SEMANTIC_PATCH_CHANNELS",
    "TRAJECTORY_FEATURES",
    "recovery_parameter_count",
    "recovery_sequence_collate",
)

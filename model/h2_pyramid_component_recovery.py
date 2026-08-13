"""Small source-agnostic component recovery head for the H2 pyramid stage."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


NODE_FEATURE_DIM = 96
HIDDEN_DIM = 32
ATTENTION_HEADS = 4


class H2PyramidComponentRecoveryHead(nn.Module):
    """Score one variable-length M20 component from input-derived node features.

    Each node represents one component/time-bin slice.  Relative time and
    geometry are already part of the 96-dimensional node vector, so the head
    needs no source identity, path, fold index, label, or target identifier.
    """

    def __init__(
        self,
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        attention_heads=ATTENTION_HEADS,
    ):
        super().__init__()
        node_feature_dim = int(node_feature_dim)
        hidden_dim = int(hidden_dim)
        attention_heads = int(attention_heads)
        if node_feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(node_feature_dim),
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, node_mask):
        if node_features.ndim != 3 or node_features.shape[-1] != self.node_feature_dim:
            raise ValueError("node_features must be [B,N,96]")
        if node_mask.ndim != 2 or node_mask.shape != node_features.shape[:2]:
            raise ValueError("node_mask must align with [B,N]")
        if node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be Boolean")
        if not bool(torch.all(node_mask.any(dim=1))):
            raise ValueError("every component needs at least one temporal node")
        values = self.input_projection(node_features)
        normalized = self.attention_norm(values)
        attended, _ = self.temporal_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~node_mask,
            need_weights=False,
        )
        values = values + attended
        values = values + self.feedforward(self.feedforward_norm(values))
        mask = node_mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1).to(values.dtype)
        mean_pool = (values * mask).sum(dim=1) / denominator
        negative_infinity = torch.finfo(values.dtype).min
        max_pool = values.masked_fill(~mask, negative_infinity).max(dim=1).values
        logits = self.output(torch.cat((mean_pool, max_pool), dim=1)).squeeze(1)
        if not torch.isfinite(logits).all():
            raise RuntimeError("component recovery logits are non-finite")
        return logits


def component_recovery_parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = (
    "ATTENTION_HEADS",
    "HIDDEN_DIM",
    "H2PyramidComponentRecoveryHead",
    "NODE_FEATURE_DIM",
    "component_recovery_parameter_count",
)

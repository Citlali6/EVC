"""Small edge-conditioned temporal graph expert for H2 false tracks.

This module has no dependency on source identity or truth-bearing metadata.  A
caller supplies an input-only :class:`TemporalTrackGraph` converted to tensors;
the expert returns pure-FP logits for every complete component and every
deterministic track.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils.h2_temporal_track_graph import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
)


NODE_WIDTH = len(NODE_FEATURE_NAMES)
EDGE_WIDTH = len(EDGE_FEATURE_NAMES)
TRACK_WIDTH = len(TRACK_FEATURE_NAMES)
HIDDEN_WIDTH = 48
MESSAGE_PASSING_LAYERS = 3


@dataclass(frozen=True)
class TemporalTrackGraphOutput:
    component_pure_fp_logits: torch.Tensor
    track_pure_fp_logits: torch.Tensor
    component_embeddings: torch.Tensor
    track_embeddings: torch.Tensor


class EdgeConditionedMessageLayer(nn.Module):
    """Degree-normalized, gated directed message passing without PyG."""

    def __init__(self, hidden_width=HIDDEN_WIDTH):
        super().__init__()
        hidden_width = int(hidden_width)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_width, 2 * hidden_width),
            nn.SiLU(),
            nn.Linear(2 * hidden_width, hidden_width),
        )
        self.norm = nn.LayerNorm(hidden_width)

    def forward(self, nodes, edge_index, edges):
        if nodes.ndim != 2 or edges.ndim != 2 or edge_index.ndim != 2:
            raise ValueError("graph tensors must be rank two")
        if edge_index.shape[0] != 2 or edge_index.shape[1] != edges.shape[0]:
            raise ValueError("edge index and edge feature rows differ")
        if edge_index.numel() == 0:
            return self.norm(nodes + self.update(torch.cat((nodes, torch.zeros_like(nodes)), dim=1)))
        source = edge_index[0].long()
        target = edge_index[1].long()
        if int(source.min()) < 0 or int(target.min()) < 0:
            raise ValueError("edge index must be non-negative")
        if int(source.max()) >= nodes.shape[0] or int(target.max()) >= nodes.shape[0]:
            raise ValueError("edge index exceeds node count")
        source_nodes = nodes[source]
        target_nodes = nodes[target]
        messages = self.message(torch.cat((source_nodes, edges), dim=1))
        gates = torch.sigmoid(
            self.gate(torch.cat((source_nodes, target_nodes, edges), dim=1))
        )
        weighted = messages * gates
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, target, weighted)
        normalizer = nodes.new_zeros((nodes.shape[0], 1))
        normalizer.index_add_(0, target, gates)
        aggregate = aggregate / normalizer.clamp_min(1.0)
        update = self.update(torch.cat((nodes, aggregate), dim=1))
        return self.norm(nodes + update)


def scatter_track_pool(nodes, component_to_track, track_count):
    """Differentiable mean/max pooling from components to complete tracks."""

    if nodes.ndim != 2 or component_to_track.ndim != 1:
        raise ValueError("nodes must be [N,C] and component_to_track [N]")
    if component_to_track.numel() != nodes.shape[0] or int(track_count) <= 0:
        raise ValueError("track assignment and node count differ")
    assignment = component_to_track.long()
    if int(assignment.min()) < 0 or int(assignment.max()) >= int(track_count):
        raise ValueError("track assignment is outside track count")
    total = nodes.new_zeros((int(track_count), nodes.shape[1]))
    total.index_add_(0, assignment, nodes)
    counts = nodes.new_zeros((int(track_count), 1))
    counts.index_add_(0, assignment, nodes.new_ones((nodes.shape[0], 1)))
    mean = total / counts.clamp_min(1.0)
    maximum = nodes.new_full((int(track_count), nodes.shape[1]), -torch.inf)
    maximum.scatter_reduce_(
        0,
        assignment[:, None].expand(-1, nodes.shape[1]),
        nodes,
        reduce="amax",
        include_self=True,
    )
    if not torch.isfinite(maximum).all():
        raise RuntimeError("every declared track must contain a component")
    return mean, maximum


class TemporalTrackGraphExpert(nn.Module):
    """Message-passing component encoder with a complete-track pooling head."""

    def __init__(
        self,
        node_width=NODE_WIDTH,
        edge_width=EDGE_WIDTH,
        track_width=TRACK_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        message_passing_layers=MESSAGE_PASSING_LAYERS,
    ):
        super().__init__()
        self.node_width = int(node_width)
        self.edge_width = int(edge_width)
        self.track_width = int(track_width)
        self.hidden_width = int(hidden_width)
        if min(self.node_width, self.edge_width, self.track_width, self.hidden_width) <= 0:
            raise ValueError("graph expert widths must be positive")
        if int(message_passing_layers) <= 0:
            raise ValueError("message_passing_layers must be positive")
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(self.node_width),
            nn.Linear(self.node_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.LayerNorm(self.hidden_width),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(self.edge_width),
            nn.Linear(self.edge_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
        )
        self.layers = nn.ModuleList(
            EdgeConditionedMessageLayer(self.hidden_width)
            for _ in range(int(message_passing_layers))
        )
        self.track_encoder = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_width + self.track_width),
            nn.Linear(2 * self.hidden_width + self.track_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.LayerNorm(self.hidden_width),
        )
        self.track_head = nn.Sequential(
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, 1),
        )
        self.component_head = nn.Sequential(
            nn.Linear(2 * self.hidden_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, 1),
        )

    def forward(
        self,
        node_features,
        edge_index,
        edge_features,
        component_to_track,
        track_features,
    ):
        if node_features.ndim != 2 or node_features.shape[1] != self.node_width:
            raise ValueError("node feature width mismatch")
        if edge_features.ndim != 2 or edge_features.shape[1] != self.edge_width:
            raise ValueError("edge feature width mismatch")
        if track_features.ndim != 2 or track_features.shape[1] != self.track_width:
            raise ValueError("track feature width mismatch")
        if node_features.shape[0] == 0 or track_features.shape[0] == 0:
            raise ValueError("graph expert requires non-empty nodes and tracks")
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edges)
        track_mean, track_maximum = scatter_track_pool(
            nodes, component_to_track, track_features.shape[0]
        )
        tracks = self.track_encoder(
            torch.cat((track_mean, track_maximum, track_features), dim=1)
        )
        track_logits = self.track_head(tracks).squeeze(1)
        component_logits = self.component_head(
            torch.cat((nodes, tracks[component_to_track.long()]), dim=1)
        ).squeeze(1)
        return TemporalTrackGraphOutput(
            component_pure_fp_logits=component_logits,
            track_pure_fp_logits=track_logits,
            component_embeddings=nodes,
            track_embeddings=tracks,
        )


def balanced_graph_bce(
    output,
    component_targets,
    track_targets,
    component_weights=None,
    track_weights=None,
):
    """Equal-mass pure-FP/target-bearing BCE for nodes and complete tracks."""

    component_targets = component_targets.to(
        device=output.component_pure_fp_logits.device,
        dtype=output.component_pure_fp_logits.dtype,
    ).reshape(-1)
    track_targets = track_targets.to(
        device=output.track_pure_fp_logits.device,
        dtype=output.track_pure_fp_logits.dtype,
    ).reshape(-1)
    if component_targets.shape != output.component_pure_fp_logits.shape:
        raise ValueError("component target shape mismatch")
    if track_targets.shape != output.track_pure_fp_logits.shape:
        raise ValueError("track target shape mismatch")

    def balanced(logits, targets, supplied_weights):
        if not torch.any(targets == 0) or not torch.any(targets == 1):
            raise ValueError("balanced graph BCE needs both classes")
        weights = torch.ones_like(targets)
        if supplied_weights is not None:
            weights = supplied_weights.to(device=logits.device, dtype=logits.dtype).reshape(-1)
            if weights.shape != targets.shape or torch.any(weights <= 0):
                raise ValueError("graph BCE weights must align and be positive")
        negative_mass = weights[targets == 0].sum()
        positive_mass = weights[targets == 1].sum()
        class_weights = weights.clone()
        class_weights[targets == 1] *= negative_mass / positive_mass.clamp_min(1e-12)
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (losses * class_weights).sum() / class_weights.sum().clamp_min(1e-12)

    component_loss = balanced(
        output.component_pure_fp_logits,
        component_targets,
        component_weights,
    )
    track_loss = balanced(output.track_pure_fp_logits, track_targets, track_weights)
    loss = 0.5 * (component_loss + track_loss)
    return {
        "loss": loss,
        "component_loss": component_loss,
        "track_loss": track_loss,
    }


def graph_expert_parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = (
    "EDGE_WIDTH",
    "HIDDEN_WIDTH",
    "MESSAGE_PASSING_LAYERS",
    "NODE_WIDTH",
    "TRACK_WIDTH",
    "EdgeConditionedMessageLayer",
    "TemporalTrackGraphExpert",
    "TemporalTrackGraphOutput",
    "balanced_graph_bce",
    "graph_expert_parameter_count",
    "scatter_track_pool",
)

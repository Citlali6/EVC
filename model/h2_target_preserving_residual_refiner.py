"""Target-preserving dual-head residual refinement on a frozen released M20.

V1 used one unconstrained signed output and learned a broad negative shift.  V2
separates non-negative protection and suppression magnitudes.  Scalar gates
start at exactly zero, preserving bitwise M20 identity, and are projected onto
the non-negative half-line after every optimizer step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.h2_spatiotemporal_residual_refiner import FrozenM20ResidualRefiner


class SpatiotemporalMagnitudeBranch(nn.Module):
    """Produce a strictly non-negative, input-dependent residual magnitude."""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        input_channels = int(input_channels)
        hidden_channels = int(hidden_channels)
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError('Branch channel counts must be positive.')
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
            groups=hidden_channels,
            bias=True,
        )
        self.temporal_mixing = nn.Conv3d(
            hidden_channels, hidden_channels, kernel_size=1, bias=True,
        )
        self.magnitude_projection = nn.Conv3d(
            hidden_channels, 1, kernel_size=1, bias=True,
        )

    def forward(self, inputs):
        if inputs.ndim != 5:
            raise ValueError('inputs must have shape [B, T, C, H, W].')
        batch_size, sequence_length = inputs.shape[:2]
        flat = inputs.reshape(
            batch_size * sequence_length,
            inputs.shape[2],
            inputs.shape[3],
            inputs.shape[4],
        )
        spatial = F.silu(self.input_projection(flat), inplace=False)
        spatial = F.silu(
            self.spatial_mixing(self.spatial_depthwise(spatial)),
            inplace=False,
        )
        temporal = spatial.reshape(
            batch_size,
            sequence_length,
            spatial.shape[1],
            spatial.shape[2],
            spatial.shape[3],
        ).permute(0, 2, 1, 3, 4)
        temporal = F.pad(temporal, (0, 0, 0, 0, 1, 1), mode='replicate')
        temporal = F.silu(
            self.temporal_mixing(self.temporal_depthwise(temporal)),
            inplace=False,
        )
        # Softplus makes the semantic meaning of both branches structural:
        # their magnitudes can never change sign.
        magnitude = F.softplus(self.magnitude_projection(temporal))
        return magnitude.permute(0, 2, 1, 3, 4)


class TargetPreservingResidualHead(nn.Module):
    """Independent protection/suppression branches with zero scalar gates."""

    def __init__(self, decoder_channels=16, hidden_channels=32):
        super().__init__()
        decoder_channels = int(decoder_channels)
        hidden_channels = int(hidden_channels)
        if decoder_channels <= 0 or hidden_channels <= 0:
            raise ValueError('decoder_channels and hidden_channels must be positive.')
        self.decoder_channels = decoder_channels
        self.hidden_channels = hidden_channels
        self.input_channels = decoder_channels + 4
        self.protection_branch = SpatiotemporalMagnitudeBranch(
            self.input_channels, hidden_channels,
        )
        self.suppression_branch = SpatiotemporalMagnitudeBranch(
            self.input_channels, hidden_channels,
        )
        # expm1(0) is exactly zero with derivative one. The runner projects
        # these raw gates to >=0 after every optimizer step, making protection
        # non-negative and suppression non-positive by construction.
        self.protection_raw_gate = nn.Parameter(torch.zeros(()))
        self.suppression_raw_gate = nn.Parameter(torch.zeros(()))

    def project_gates_(self):
        with torch.no_grad():
            self.protection_raw_gate.clamp_(min=0.0)
            self.suppression_raw_gate.clamp_(min=0.0)
        return self

    def gate_values(self):
        if bool(self.protection_raw_gate.detach() < 0) or bool(
            self.suppression_raw_gate.detach() < 0
        ):
            raise RuntimeError('Residual gates must be projected before forward.')
        return (
            torch.expm1(self.protection_raw_gate),
            torch.expm1(self.suppression_raw_gate),
        )

    def forward(
        self,
        decoder_features,
        base_logits,
        centre_inputs,
        return_parts=False,
    ):
        if decoder_features.ndim != 5:
            raise ValueError('decoder_features must have shape [B,T,C,H,W].')
        if base_logits.ndim != 5 or base_logits.shape[2] != 1:
            raise ValueError('base_logits must have shape [B,T,1,H,W].')
        if centre_inputs.ndim != 5 or centre_inputs.shape[2] != 3:
            raise ValueError('centre_inputs must have shape [B,T,3,H,W].')
        if decoder_features.shape[2] != self.decoder_channels:
            raise ValueError('Unexpected decoder feature channel count.')
        merged = torch.cat((decoder_features, base_logits, centre_inputs), dim=2)
        protection_magnitude = self.protection_branch(merged)
        suppression_magnitude = self.suppression_branch(merged)
        protection_gate, suppression_gate = self.gate_values()
        protection = protection_gate * protection_magnitude
        suppression = suppression_gate * suppression_magnitude
        residual = protection - suppression
        if return_parts:
            return {
                'residual': residual,
                'protection': protection,
                'suppression': suppression,
                'protection_magnitude': protection_magnitude,
                'suppression_magnitude': suppression_magnitude,
                'protection_gate': protection_gate,
                'suppression_gate': suppression_gate,
            }
        return residual


class FrozenM20TargetPreservingRefiner(FrozenM20ResidualRefiner):
    """V1 wrapper interface with the V2 dual-head residual substituted."""

    def __init__(self, released_m20, context_bins=5, hidden_channels=32):
        super().__init__(
            released_m20,
            context_bins=context_bins,
            hidden_channels=hidden_channels,
        )
        self.refiner = TargetPreservingResidualHead(
            decoder_channels=int(released_m20.base.head.in_channels),
            hidden_channels=int(hidden_channels),
        )

    def refine_with_parts(self, frames, decoder_features, base_logits):
        if frames.ndim != 4 or decoder_features.ndim != 4 or base_logits.ndim != 4:
            raise ValueError('frames, decoder_features and base_logits must be 4-D.')
        centre_inputs = self._centre_inputs(frames)
        parts = self.refiner(
            decoder_features.unsqueeze(0),
            base_logits.unsqueeze(0),
            centre_inputs.unsqueeze(0),
            return_parts=True,
        )
        parts = {
            key: value.squeeze(0) if value.ndim == 5 else value
            for key, value in parts.items()
        }
        parts['refined_logits'] = base_logits + parts['residual']
        return parts

    def project_gates_(self):
        self.refiner.project_gates_()
        return self


def target_preserving_parameter_count(model):
    return sum(parameter.numel() for parameter in model.refiner.parameters())


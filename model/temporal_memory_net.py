"""Bidirectional temporal memory on top of the P23 full-frame backbone."""

import torch
import torch.nn as nn

from model.temporal_frame_net import TemporalFrameNet


class ConvGRUCell(nn.Module):
    """A compact spatial ConvGRU cell used only at the U-Net bottleneck."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError('channels must be positive.')
        self.channels = channels
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, inputs, state=None):
        if inputs.ndim != 4:
            raise ValueError('inputs must have shape [B, C, H, W].')
        if inputs.shape[1] != self.channels:
            raise ValueError('Unexpected ConvGRU input channels.')
        if state is None:
            state = torch.zeros_like(inputs)
        if state.shape != inputs.shape:
            raise ValueError('ConvGRU state shape does not match inputs.')
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, state), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * state), dim=1))
        )
        return (1.0 - update) * state + update * candidate


class BidirectionalTemporalMemoryNet(nn.Module):
    """P23 U-Net with a zero-initialized bidirectional temporal residual.

    Every temporal step first receives the original P23 local context stack.
    A pair of ConvGRU cells then propagates low-resolution evidence forward
    and backward through a sequence.  The residual projection is initialized
    to zero, allowing a P23 checkpoint to be loaded without changing its
    initial predictions before memory training begins.
    """

    def __init__(self, input_channels, width=16):
        super().__init__()
        self.base = TemporalFrameNet(
            input_channels=int(input_channels),
            width=int(width),
        )
        bottleneck_channels = int(width) * 6
        self.forward_memory = ConvGRUCell(bottleneck_channels)
        self.backward_memory = ConvGRUCell(bottleneck_channels)
        self.memory_projection = nn.Conv2d(
            bottleneck_channels * 2,
            bottleneck_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)

    @property
    def input_channels(self):
        return self.base.input_channels

    def _encode(self, frames):
        if frames.ndim != 4:
            raise ValueError('frames must have shape [B, C, H, W].')
        if frames.shape[1] != self.input_channels:
            raise ValueError(
                'frames have {} channels, expected {}.'.format(
                    frames.shape[1], self.input_channels
                )
            )
        level0 = self.base.encoder0(frames)
        level1 = self.base.encoder1(level0)
        level2 = self.base.encoder2(level1)
        bottleneck = self.base.context(self.base.encoder3(level2))
        return level0, level1, level2, bottleneck

    def encode_bottleneck(self, frames):
        """Encode a frame batch for full-stream inference memory passes."""
        return self._encode(frames)[-1]

    def _memory_residual(self, bottlenecks):
        if bottlenecks.ndim != 5:
            raise ValueError('bottlenecks must have shape [B, T, C, H, W].')
        batch_size, sequence_length = bottlenecks.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')

        forward_states = []
        state = None
        for time_index in range(sequence_length):
            state = self.forward_memory(bottlenecks[:, time_index], state)
            forward_states.append(state)

        backward_states = [None] * sequence_length
        state = None
        for time_index in range(sequence_length - 1, -1, -1):
            state = self.backward_memory(bottlenecks[:, time_index], state)
            backward_states[time_index] = state

        memory_features = torch.cat(
            (
                torch.stack(forward_states, dim=1),
                torch.stack(backward_states, dim=1),
            ),
            dim=2,
        )
        flat_features = memory_features.reshape(
            batch_size * sequence_length,
            memory_features.shape[2],
            memory_features.shape[3],
            memory_features.shape[4],
        )
        projected = self.memory_projection(flat_features)
        return projected.reshape(
            batch_size,
            sequence_length,
            projected.shape[1],
            projected.shape[2],
            projected.shape[3],
        )

    def temporal_residual(self, bottlenecks):
        """Return one zero-initialized temporal residual per bottleneck map."""
        if bottlenecks.ndim == 4:
            return self._memory_residual(bottlenecks.unsqueeze(0)).squeeze(0)
        return self._memory_residual(bottlenecks)

    def _decode(self, level0, level1, level2, bottleneck):
        decoded2 = self.base.decoder2(bottleneck, level2)
        decoded1 = self.base.decoder1(decoded2, level1)
        decoded0 = self.base.decoder0(decoded1, level0)
        return self.base.head(decoded0)

    def decode_with_residual(self, frames, residual):
        """Decode a frame batch after a full-stream memory pass."""
        level0, level1, level2, bottleneck = self._encode(frames)
        if residual.shape != bottleneck.shape:
            raise ValueError('Temporal residual does not match bottleneck shape.')
        return self._decode(level0, level1, level2, bottleneck + residual)

    def forward(self, frames):
        """Predict logit maps for ``[B, T, C, H, W]`` temporal sequences."""
        if frames.ndim != 5:
            raise ValueError('frames must have shape [B, T, C, H, W].')
        batch_size, sequence_length = frames.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')
        flat_frames = frames.reshape(
            batch_size * sequence_length,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        level0, level1, level2, bottleneck = self._encode(flat_frames)
        bottleneck = bottleneck.reshape(
            batch_size,
            sequence_length,
            bottleneck.shape[1],
            bottleneck.shape[2],
            bottleneck.shape[3],
        )
        residual = self._memory_residual(bottleneck).reshape_as(
            bottleneck.reshape(
                batch_size * sequence_length,
                bottleneck.shape[2],
                bottleneck.shape[3],
                bottleneck.shape[4],
            )
        )
        logits = self._decode(
            level0,
            level1,
            level2,
            bottleneck.reshape(
                batch_size * sequence_length,
                bottleneck.shape[2],
                bottleneck.shape[3],
                bottleneck.shape[4],
            ) + residual,
        )
        return logits.reshape(
            batch_size,
            sequence_length,
            logits.shape[1],
            logits.shape[2],
            logits.shape[3],
        )

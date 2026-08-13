"""Zero-initialized high-density expert for the released M20 model.

The expert consumes the existing five 50-unit, polarity-separated context
bins.  It adds a small half-resolution residual to M20's first downsampled
feature map.  The final projection is initialized to zero, so adding the
expert is an exact functional identity before training.

Three input modes intentionally share the exact same parameterization:

``activity_control``
    All three five-channel banks contain polarity-summed activity.  This is
    the paired architectural/control fine-tune.

``h2_polarity``
    The extra banks contain signed and absolute signed activity.  This is the
    balanced-polarity H2 representation.

``h1_saturation``
    The extra banks contain peak-polarity log-count and its square from a
    parallel clip-8 frame stack.  M20 itself continues to receive its exact
    released clip-4 stack.  This is the strongly one-polarity H1 saturation
    representation.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from model.temporal_memory_net import BidirectionalTemporalMemoryNet


EXPERT_INPUT_MODES = ("activity_control", "h2_polarity", "h1_saturation")


def _group_count(channels):
    channels = int(channels)
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class FineTemporalPolarityMultiScaleExpert(nn.Module):
    """Small half-resolution residual expert with two spatial receptive fields."""

    def __init__(
        self,
        input_channels=10,
        output_channels=32,
        hidden_channels=16,
        input_mode="h2_polarity",
    ):
        super().__init__()
        input_channels = int(input_channels)
        output_channels = int(output_channels)
        hidden_channels = int(hidden_channels)
        if input_channels <= 0 or input_channels % 2:
            raise ValueError("input_channels must be a positive even integer.")
        if input_channels != 10:
            raise ValueError("The frozen expert contract requires ten M20 channels.")
        if output_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Expert channel counts must be positive.")
        if input_mode not in EXPERT_INPUT_MODES:
            raise ValueError("Unknown expert input mode: {}".format(input_mode))

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.input_mode = str(input_mode)
        self.temporal_bin_count = input_channels // 2

        # A 1x1 temporal mixer sees all five 50-unit bins at every pixel.
        self.derived_input_channels = self.temporal_bin_count * 3
        self.temporal_mixer = nn.Sequential(
            nn.Conv2d(
                self.derived_input_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=False),
        )
        # Depthwise branches keep the expert cheap while adding local and
        # wider (dilation=2) spatial context at half resolution.
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=False),
        )
        self.wide_branch = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=False),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=False),
        )
        self.output_projection = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def paired_input_features(self, frames, expert_frames=None):
        if frames.ndim != 4:
            raise ValueError("frames must have shape [B, 10, H, W].")
        if int(frames.shape[1]) != self.input_channels:
            raise ValueError(
                "frames have {} channels, expected {}.".format(
                    int(frames.shape[1]), self.input_channels
                )
            )
        negative = frames[:, 0::2]
        positive = frames[:, 1::2]
        activity = 0.5 * (negative + positive)
        if self.input_mode == "h2_polarity":
            signed = positive - negative
            second = signed
            third = signed.abs()
        elif self.input_mode == "h1_saturation":
            if expert_frames is None:
                raise ValueError(
                    "h1_saturation requires the parallel clip-8 expert frames."
                )
            if expert_frames.shape != frames.shape:
                raise ValueError("clip-8 expert frames must match clip-4 frames.")
            expert_negative = expert_frames[:, 0::2]
            expert_positive = expert_frames[:, 1::2]
            peak_log_count = torch.maximum(expert_negative, expert_positive)
            second = peak_log_count
            third = peak_log_count.square()
        else:
            second = activity
            third = activity
        return torch.cat((activity, second, third), dim=1)

    def forward(self, frames, expert_frames=None):
        features = self.paired_input_features(frames, expert_frames=expert_frames)
        # M20 encoder1 uses stride=2,padding=1.  For the fixed even 260x346
        # input, 2x2 average pooling has the identical 130x173 output size.
        features = functional.avg_pool2d(features, kernel_size=2, stride=2)
        mixed = self.temporal_mixer(features)
        fused = self.fusion(
            torch.cat((self.local_branch(mixed), self.wide_branch(mixed)), dim=1)
        )
        return self.output_projection(fused)


class HighDensityPolarityExpertMemoryNet(BidirectionalTemporalMemoryNet):
    """Released M20 plus a zero-initialized level-1 residual expert."""

    def __init__(
        self,
        input_channels=10,
        width=16,
        temporal_attention_enabled=True,
        density_calibration_enabled=True,
        density_calibration_v2_enabled=False,
        confidence_head_enabled=False,
        expert_input_mode="h2_polarity",
        expert_hidden_channels=16,
    ):
        super().__init__(
            input_channels=input_channels,
            width=width,
            temporal_attention_enabled=temporal_attention_enabled,
            density_calibration_enabled=density_calibration_enabled,
            density_calibration_v2_enabled=density_calibration_v2_enabled,
            confidence_head_enabled=confidence_head_enabled,
        )
        self.high_density_expert = FineTemporalPolarityMultiScaleExpert(
            input_channels=input_channels,
            output_channels=int(width) * 2,
            hidden_channels=expert_hidden_channels,
            input_mode=expert_input_mode,
        )

    @property
    def expert_input_mode(self):
        return self.high_density_expert.input_mode

    def _encode(self, frames, expert_frames=None):
        if frames.ndim != 4:
            raise ValueError("frames must have shape [B, C, H, W].")
        if int(frames.shape[1]) != self.input_channels:
            raise ValueError(
                "frames have {} channels, expected {}.".format(
                    int(frames.shape[1]), self.input_channels
                )
            )
        level0 = self.base.encoder0(frames)
        level1 = self.base.encoder1(level0)
        expert_residual = self.high_density_expert(
            frames, expert_frames=expert_frames
        )
        if expert_residual.shape != level1.shape:
            raise RuntimeError(
                "Expert residual shape {} differs from M20 level1 {}.".format(
                    tuple(expert_residual.shape), tuple(level1.shape)
                )
            )
        level1 = level1 + expert_residual
        level2 = self.base.encoder2(level1)
        bottleneck = self.base.context(self.base.encoder3(level2))
        return level0, level1, level2, bottleneck

    def encode_bottleneck(self, frames, expert_frames=None):
        """Encode clip-4 M20 frames plus an optional parallel expert stack."""
        return self._encode(frames, expert_frames=expert_frames)[-1]

    def decode_with_residual(
        self,
        frames,
        residual,
        return_confidence_logits=False,
        expert_frames=None,
    ):
        level0, level1, level2, bottleneck = self._encode(
            frames, expert_frames=expert_frames
        )
        if residual.shape != bottleneck.shape:
            raise ValueError("Temporal residual does not match bottleneck shape.")
        return self._decode(
            level0,
            level1,
            level2,
            bottleneck + residual,
            base_input=frames[:, :self.input_channels],
            return_confidence_logits=return_confidence_logits,
        )

    def forward(self, frames, expert_frames=None):
        """Predict a T16 sequence while leaving the released clip-4 path intact."""
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, C, H, W].")
        if expert_frames is not None and expert_frames.shape != frames.shape:
            raise ValueError("expert_frames must match frames.")
        batch_size, sequence_length = frames.shape[:2]
        if sequence_length <= 0:
            raise ValueError("Temporal sequence must not be empty.")
        flat_frames = frames.reshape(
            batch_size * sequence_length,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        flat_expert = None
        if expert_frames is not None:
            flat_expert = expert_frames.reshape_as(flat_frames)
        level0, level1, level2, bottleneck = self._encode(
            flat_frames, expert_frames=flat_expert
        )
        bottleneck = bottleneck.reshape(
            batch_size,
            sequence_length,
            bottleneck.shape[1],
            bottleneck.shape[2],
            bottleneck.shape[3],
        )
        residual = self._memory_residual(bottleneck).reshape(
            batch_size * sequence_length,
            bottleneck.shape[2],
            bottleneck.shape[3],
            bottleneck.shape[4],
        )
        decode_output = self._decode(
            level0,
            level1,
            level2,
            bottleneck.reshape_as(residual) + residual,
            base_input=flat_frames[:, :self.input_channels],
            return_confidence_logits=self.confidence_head_enabled,
        )
        if self.confidence_head_enabled:
            logits, confidence_logits = decode_output
            logits = logits.reshape(
                batch_size,
                sequence_length,
                logits.shape[1],
                logits.shape[2],
                logits.shape[3],
            )
            confidence_logits = confidence_logits.reshape_as(logits)
            return logits, confidence_logits
        return decode_output.reshape(
            batch_size,
            sequence_length,
            decode_output.shape[1],
            decode_output.shape[2],
            decode_output.shape[3],
        )


def expert_state_keys(model):
    return tuple(
        sorted(
            name
            for name in model.state_dict()
            if name.startswith("high_density_expert.")
        )
    )


def configure_expert_only_training(model):
    """Freeze the inherited M20 state and train exactly the expert tensors."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.high_density_expert.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.high_density_expert.train()
    names = tuple(
        sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    )
    expected = tuple(
        sorted(
            name
            for name, _ in model.named_parameters()
            if name.startswith("high_density_expert.")
        )
    )
    if names != expected:
        raise RuntimeError("Expert-only trainable-parameter scope mismatch.")
    return names


def load_m20_parent_into_expert(model, checkpoint_or_path):
    """Strictly migrate a released M20 state into an expert model.

    Only the newly introduced ``high_density_expert.*`` state may be absent.
    The function intentionally does not alter or randomize the expert state.
    """
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = torch.load(Path(checkpoint_or_path), map_location="cpu")
    else:
        checkpoint = checkpoint_or_path
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("M20 checkpoint is missing model_state_dict.")
    parent_state = checkpoint["model_state_dict"]
    incompatible = model.load_state_dict(parent_state, strict=False)
    expected_missing = set(expert_state_keys(model))
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "M20 migration missing-key set differs from the expert state: {}.".format(
                sorted(incompatible.missing_keys)
            )
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "M20 migration has unexpected keys: {}.".format(
                sorted(incompatible.unexpected_keys)
            )
        )
    loaded = model.state_dict()
    for name, tensor in parent_state.items():
        if name not in loaded or not torch.equal(loaded[name].detach().cpu(), tensor.detach().cpu()):
            raise RuntimeError("Inherited M20 tensor changed during migration: {}".format(name))
    return checkpoint


def build_expert_model_from_m20(
    checkpoint_or_path,
    input_mode,
    device="cpu",
    context_bins=5,
    width=16,
    sequence_length=16,
):
    """Instantiate the frozen T16/M20 architecture plus one paired expert."""
    if int(context_bins) != 5 or int(width) != 16 or int(sequence_length) != 16:
        raise ValueError("The v1 expert contract requires context=5,width=16,T16.")
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = torch.load(Path(checkpoint_or_path), map_location="cpu")
    else:
        checkpoint = checkpoint_or_path
    saved = checkpoint.get("temporal_memory", {})
    required = {
        "context_bins": 5,
        "width": 16,
        "sequence_length": 16,
    }
    for key, expected in required.items():
        if int(saved.get(key, -1)) != expected:
            raise ValueError("M20 metadata {} differs from {}.".format(key, expected))
    model = HighDensityPolarityExpertMemoryNet(
        input_channels=10,
        width=16,
        temporal_attention_enabled=bool(saved.get("temporal_attention_enabled", False)),
        density_calibration_enabled=bool(saved.get("density_calibration_enabled", False)),
        density_calibration_v2_enabled=(
            str(saved.get("density_calibration_version", "")) == "v2"
            or bool(saved.get("density_calibration_v2_enabled", False))
        ),
        confidence_head_enabled=bool(saved.get("confidence_head_enabled", False)),
        expert_input_mode=input_mode,
    )
    load_m20_parent_into_expert(model, checkpoint)
    model.to(device)
    return model, checkpoint

"""Optional weighted prediction ensemble for Challenge 2 inference.

The primary checkpoint always comes from ``TEST.model_path``.  A secondary
checkpoint is loaded only when the explicit ENSEMBLE switch is enabled.
Scores are averaged before the configured decision threshold and before P0,
so the validation and submission paths remain identical.
"""

from dataclasses import dataclass
from pathlib import Path

from utils.checkpoint import load_state_dict_with_optional_compatibility


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError("Expected a boolean value, got {!r}.".format(value))
    return bool(value)


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for a primary/secondary weighted score average."""

    enabled: bool = False
    secondary_model_path: str = ""
    primary_weight: float = 0.85

    def __post_init__(self):
        if not 0.0 <= self.primary_weight <= 1.0:
            raise ValueError("ensemble primary_weight must be in [0, 1].")
        if self.enabled and not self.secondary_model_path:
            raise ValueError(
                "ENSEMBLE.secondary_model_path is required when ENSEMBLE.enabled=true."
            )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, "ensemble_enabled", False)),
            secondary_model_path=str(
                getattr(cfg, "ensemble_secondary_model_path", "")
            ),
            primary_weight=float(getattr(cfg, "ensemble_primary_weight", 0.85)),
        )


def weighted_average(primary_scores, secondary_scores, primary_weight):
    """Return the event-wise weighted score average without thresholding."""
    if primary_scores.shape != secondary_scores.shape:
        raise ValueError(
            "Ensemble prediction shapes do not match: {} and {}.".format(
                tuple(primary_scores.shape), tuple(secondary_scores.shape)
            )
        )
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("primary_weight must be in [0, 1].")
    return (
        primary_scores * primary_weight
        + secondary_scores * (1.0 - primary_weight)
    )


class ChallengePredictor:
    """Load one or two compatible models and return CPU event-level scores."""

    def __init__(self, cfg, device, model_factory):
        self.cfg = cfg
        self.device = device
        self.config = EnsembleConfig.from_cfg(cfg)
        self.primary_model_path = Path(cfg.model_path)
        self.primary_net = self._load_model(self.primary_model_path, model_factory)
        self.secondary_model_path = None
        self.secondary_net = None

        if self.config.enabled:
            self.secondary_model_path = Path(self.config.secondary_model_path)
            self.secondary_net = self._load_model(
                self.secondary_model_path,
                model_factory,
            )

    def _load_model(self, model_path, model_factory):
        import torch

        if not model_path.is_file():
            raise FileNotFoundError("Model weight not found: {}".format(model_path))
        net = model_factory(self.cfg).eval().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        load_state_dict_with_optional_compatibility(
            net,
            state_dict,
            p2b_enabled=bool(getattr(net, 'p2b_density_gdsca_enabled', False)),
            p11_enabled=bool(getattr(net, 'p11_local_activity_enabled', False)),
            p12_enabled=bool(getattr(net, 'p12_local_density_enabled', False)),
        )
        return net

    def describe(self):
        if not self.config.enabled:
            return "disabled (single model)"
        return (
            "enabled (primary_weight={:.3f}, secondary_weight={:.3f}, "
            "secondary_model={})"
        ).format(
            self.config.primary_weight,
            1.0 - self.config.primary_weight,
            self.secondary_model_path,
        )

    @staticmethod
    def _predict_event_scores(net, voxel_events, p2v_map, event_frame=None):
        voxel_scores, _ = net(voxel_events, event_frame=event_frame)
        return voxel_scores[p2v_map].reshape(-1).detach().cpu()

    def predict_event_score_pair(self, voxel_events, p2v_map, event_frame=None):
        """Return primary and optional secondary scores in original event order.

        This is used by offline ensemble diagnostics.  Inference remains
        sequential so a two-checkpoint ensemble stays within the 4 GB GPU
        memory budget.
        """
        primary_scores = self._predict_event_scores(
            self.primary_net,
            voxel_events,
            p2v_map,
            event_frame,
        )
        if self.secondary_net is None:
            return primary_scores, None

        secondary_scores = self._predict_event_scores(
            self.secondary_net,
            voxel_events,
            p2v_map,
            event_frame,
        )
        return primary_scores, secondary_scores

    def predict_event_scores(self, voxel_events, p2v_map, event_frame=None):
        """Return the configured single-model or weighted ensemble scores."""
        primary_scores, secondary_scores = self.predict_event_score_pair(
            voxel_events,
            p2v_map,
            event_frame,
        )
        if secondary_scores is None:
            return primary_scores
        return weighted_average(
            primary_scores,
            secondary_scores,
            self.config.primary_weight,
        )

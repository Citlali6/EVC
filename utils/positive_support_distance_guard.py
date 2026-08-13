"""Continuous positive-only support scores with source-free observable features."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils.positive_support_guard import FEATURE_NAMES


def _weighted_quantile(values, weights, probability):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= float(cumulative[-1])
    return float(ordered[min(np.searchsorted(cumulative, probability, side="left"), ordered.size - 1)])


class RobustPositiveSupport:
    """Positive-population support with continuous tail extrapolation.

    ``mode`` is part of the frozen algorithm family, not a numeric data cutoff.
    Location, scale, nearest prototypes, and the eventual deletion cutoff are all
    learned from fit/OOF positives.
    """

    def __init__(self, evidence_names, mode):
        if mode not in {"lower_second", "robust_box", "robust_rms", "nearest"}:
            raise KeyError(mode)
        unknown = set(evidence_names) - set(FEATURE_NAMES)
        if unknown:
            raise KeyError(f"unknown evidence features: {sorted(unknown)}")
        self.evidence_names = tuple(evidence_names)
        self.feature_indices = tuple(FEATURE_NAMES.index(name) for name in evidence_names)
        self.mode = mode
        self.location = None
        self.scale = None
        self.neighbors = None

    def fit(self, video_features, positive_masks):
        matrices = []
        source_weights = []
        for features, mask in zip(video_features, positive_masks):
            matrix = np.asarray(features, dtype=np.float64)
            selected = matrix[np.asarray(mask, dtype=bool)][:, self.feature_indices]
            if selected.shape[0] == 0:
                continue
            matrices.append(selected)
            source_weights.append(np.full(selected.shape[0], 1.0 / selected.shape[0]))
        if not matrices:
            raise RuntimeError("positive support fit received no positive components.")
        matrix = np.concatenate(matrices, axis=0)
        weights = np.concatenate(source_weights)
        location = []
        scale = []
        for column in range(matrix.shape[1]):
            median = _weighted_quantile(matrix[:, column], weights, 0.5)
            lower = _weighted_quantile(matrix[:, column], weights, 0.25)
            upper = _weighted_quantile(matrix[:, column], weights, 0.75)
            width = upper - lower
            if width < 1e-12:
                deviations = np.abs(matrix[:, column] - median)
                width = _weighted_quantile(deviations, weights, 0.75)
            if width < 1e-12:
                width = 1.0
            location.append(median)
            scale.append(width)
        self.location = np.asarray(location, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.mode == "nearest":
            normalized = (matrix - self.location) / self.scale
            self.neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1)
            self.neighbors.fit(normalized)
        return self

    def predict_support(self, features):
        if self.location is None or self.scale is None:
            raise RuntimeError("positive support model is not fitted.")
        matrix = np.asarray(features, dtype=np.float64)[:, self.feature_indices]
        standardized = (matrix - self.location) / self.scale
        if self.mode == "lower_second":
            if standardized.shape[1] < 2:
                raise RuntimeError("lower-second support requires two evidence features.")
            return np.partition(standardized, -2, axis=1)[:, -2]
        if self.mode == "robust_box":
            return -np.max(np.abs(standardized), axis=1)
        if self.mode == "robust_rms":
            return -np.sqrt(np.mean(np.square(standardized), axis=1))
        distances = self.neighbors.kneighbors(standardized, return_distance=True)[0][:, 0]
        return -distances / np.sqrt(float(standardized.shape[1]))


__all__ = ("RobustPositiveSupport",)

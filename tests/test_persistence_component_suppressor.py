import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fit_and_replay_persistence_standalone_train as fit_runner
import fit_and_replay_persistence_standalone_train_v2 as fit_runner_v2
from utils.component_reranker import sha256_file
from utils.persistence_component_suppressor import (
    ARTIFACT_SCHEMA,
    DEFAULT_TOPOLOGY,
    FEATURE_NAMES,
    FEATURE_SEMANTICS_VERSION,
    PersistenceArtifact,
    PersistenceComponentSuppressor,
    component_persistence_features,
    derive_pixel_prior_from_arrays,
    observable_route,
)


def _artifact_payload(intercept=-2.0):
    width = len(FEATURE_NAMES)
    return {
        "schema": ARTIFACT_SCHEMA,
        "candidate_id": "persistence_pw08_kp050",
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "component_topology": DEFAULT_TOPOLOGY.to_dict(),
        "model": {
            "feature_mean": [0.0] * width,
            "feature_scale": [1.0] * width,
            "coefficients": [0.0] * width,
            "intercept": intercept,
            "positive_weight": 8.0,
            "keep_probability": 0.5,
        },
        "runtime_contract": {
            "t32_allowed": False,
            "prediction_threshold": 0.719,
            "event_count_cutoff_exclusive": 200000,
            "polarity_minority_cutoff": 0.2,
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_hash_bound_winner_is_frozen_before_final_fit(self):
        protocol, _, winner = fit_runner.validate_protocol_and_winner()
        self.assertEqual(winner["candidate_id"], "persistence_pw08_kp050")
        self.assertTrue(winner["conservative_gate_passed"])
        self.assertFalse(protocol["standalone_runtime"]["t32_allowed"])
        self.assertFalse(protocol["selection_disclosure"]["independent_held_claim_allowed"])

    def test_corrected_protocol_fits_only_h2_and_discloses_selection_bias(self):
        protocol, _, winner = fit_runner_v2.validate_protocol_and_winner()
        self.assertEqual(winner["candidate_id"], "persistence_pw08_kp050")
        self.assertEqual(
            tuple(protocol["population"]["fit_h2_sources"]), fit_runner_v2.H2_NAMES
        )
        self.assertEqual(
            tuple(protocol["population"]["identity_only_h1_sources"]),
            fit_runner_v2.H1_NAMES,
        )
        self.assertEqual(protocol["population"]["h1_fit_mass"], 0.0)
        self.assertEqual(protocol["selection_disclosure"]["candidate_grid_count"], 7)
        self.assertTrue(
            protocol["selection_disclosure"][
                "pooled_oof_delta_is_selection_affected_not_independent"
            ]
        )
        self.assertTrue(
            protocol["estimator_correction"][
                "superseded_outputs_must_not_enter_validation"
            ]
        )
        self.assertEqual(
            protocol["standalone_runtime"]["effective_c00_canonical_sha256"],
            fit_runner_v2.EXPECTED_EFFECTIVE_C00_SHA256,
        )
        self.assertEqual(len(protocol["standalone_runtime"]["stage_order"]), 6)
        self.assertEqual(protocol["final_fit"]["feature_dtype"], "float64")


class ArtifactTests(unittest.TestCase):
    def test_artifact_rejects_t32_contract_or_wrong_width(self):
        payload = _artifact_payload()
        payload["runtime_contract"]["t32_allowed"] = True
        with self.assertRaisesRegex(ValueError, "contract"):
            PersistenceArtifact.from_payload(payload)
        payload = _artifact_payload()
        payload["model"]["coefficients"] = [0.0]
        with self.assertRaisesRegex(ValueError, "width"):
            PersistenceArtifact.from_payload(payload)

    def test_artifact_rejects_changed_component_topology(self):
        payload = _artifact_payload()
        payload["component_topology"]["max_component_events"] = 4
        with self.assertRaisesRegex(ValueError, "topology"):
            PersistenceArtifact.from_payload(payload)

    def test_file_loader_checks_external_sha256(self):
        payload = _artifact_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            actual = sha256_file(path)
            loaded = PersistenceArtifact.load(path, actual)
            self.assertEqual(loaded.artifact_sha256, actual)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                PersistenceArtifact.load(path, "0" * 64)


class RouteAndRuntimeTests(unittest.TestCase):
    def test_route_boundaries_are_input_only(self):
        balanced_200k = np.arange(200000, dtype=np.uint8) % 2
        balanced_200001 = np.arange(200001, dtype=np.uint8) % 2
        self.assertFalse(observable_route(200000, balanced_200k)["eligible"])
        self.assertTrue(observable_route(200001, balanced_200001)["eligible"])
        h1 = np.zeros(200001, dtype=np.uint8)
        self.assertFalse(observable_route(200001, h1)["eligible"])

    def test_non_h2_returns_same_tensor_without_component_call(self):
        artifact = PersistenceArtifact.from_payload(_artifact_payload())
        suppressor = PersistenceComponentSuppressor(artifact)
        count = 12
        scores = torch.linspace(0.0, 1.0, count)
        locations = np.column_stack(
            (np.arange(count) % 10, np.arange(count) % 8, np.arange(count))
        )
        polarities = np.arange(count) % 2
        with mock.patch(
            "utils.persistence_component_suppressor.extract_persistence_components"
        ) as extract:
            output, stats = suppressor.apply(scores, locations, polarities)
        self.assertIs(output, scores)
        self.assertTrue(torch.equal(output, scores))
        self.assertFalse(stats.component_chain_called)
        extract.assert_not_called()

    def test_high_count_h1_returns_same_numpy_object(self):
        artifact = PersistenceArtifact.from_payload(_artifact_payload())
        suppressor = PersistenceComponentSuppressor(artifact)
        count = 200001
        scores = np.zeros(count, dtype=np.float32)
        locations = np.column_stack(
            (
                np.arange(count) % 346,
                np.arange(count) % 260,
                np.arange(count) % 8000,
            )
        )
        polarities = np.zeros(count, dtype=np.uint8)
        with mock.patch(
            "utils.persistence_component_suppressor.extract_persistence_components"
        ) as extract:
            output, stats = suppressor.apply(scores, locations, polarities)
        self.assertIs(output, scores)
        self.assertFalse(stats.component_chain_called)
        extract.assert_not_called()

    def test_h2_invokes_component_chain_and_suppresses_rejected_candidate(self):
        artifact = PersistenceArtifact.from_payload(_artifact_payload(intercept=-2.0))
        suppressor = PersistenceComponentSuppressor(artifact)
        count = 200001
        scores = np.zeros(count, dtype=np.float32)
        scores[0] = np.float32(0.9)
        locations = np.column_stack(
            (
                np.arange(count) % 346,
                np.arange(count) % 260,
                np.arange(count) % 8000,
            )
        )
        polarities = np.arange(count, dtype=np.uint8) % 2
        output, stats = suppressor.apply(scores, locations, polarities)
        self.assertIsNot(output, scores)
        self.assertTrue(stats.component_chain_called)
        self.assertEqual(stats.candidate_component_count, 1)
        self.assertEqual(stats.removed_candidate_components, 1)
        self.assertEqual(float(output[0]), 0.0)


class PriorTests(unittest.TestCase):
    def test_prior_is_finite_and_preserves_frozen_h1_h2_summary_semantics(self):
        locations = np.asarray(
            [[1, 1, 0], [1, 1, 50], [2, 1, 50], [2, 1, 100]], dtype=np.int64
        )
        polarities = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        prior = derive_pixel_prior_from_arrays(locations, polarities)
        self.assertEqual(prior.summary["observable_domain"], "h2")
        self.assertTrue(np.isfinite(prior.active_fraction).all())
        self.assertEqual(prior.summary["event_count"], 4)

    def test_feature_semantics_use_complete_160_bins_and_event_repetition(self):
        locations = np.asarray(
            [[1, 1, 0], [1, 1, 50], [1, 1, 50], [2, 1, 100]],
            dtype=np.int64,
        )
        polarities = np.asarray([1, 0, 0, 1], dtype=np.uint8)
        prior = derive_pixel_prior_from_arrays(locations, polarities)
        pixel_one = 1 * 346 + 1
        self.assertEqual(prior.log_events[pixel_one], np.log1p(3.0))
        self.assertEqual(prior.active_fraction[pixel_one], 2.0 / 160.0)
        self.assertEqual(prior.longest_run_fraction[pixel_one], 2.0 / 160.0)
        self.assertEqual(prior.collision_fraction[pixel_one], 1.0 - 2.0 / 3.0)
        self.assertEqual(prior.polarity_dominance[pixel_one], abs(2.0 / 3.0 - 1.0))
        features = component_persistence_features(
            prior, (np.asarray([0, 1, 2, 3], dtype=np.int64),)
        )
        expected_log_mean = (
            prior.log_events[pixel_one] * 3.0
            + prior.log_events[1 * 346 + 2]
        ) / 4.0
        self.assertEqual(features.dtype, np.float64)
        self.assertEqual(features.shape, (1, len(FEATURE_NAMES)))
        self.assertEqual(features[0, 0], expected_log_mean)
        self.assertEqual(features[0, 1], prior.log_events[pixel_one])


if __name__ == "__main__":
    unittest.main()

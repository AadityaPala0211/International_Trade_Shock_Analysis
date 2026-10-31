"""Focused tests for the revised WD-only structural and SIS workflow."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz

from network_structural_analysis.config import (
    REPRESENTATIONS, LEGACY_REPRESENTATIONS, FAMILIES,
    ACTIVE_SYNTHETIC_REALIZATIONS, STAGE1_ETA_LEVELS,
)
from network_structural_analysis.revised_structural import (
    MATRIX_FORMAT, _wd_from_flows, atomic_json, sha256_file,
    run_synthetic_structural, validate_structural_group,
)
from network_structural_analysis.revised_dynamics import (
    simulate_one_step, simulate_single_node_batch, spectral_threshold,
    validate_wd_matrix, _run_resumable_experiments,
)


class TestRevisedNetworkScope(unittest.TestCase):
    def test_active_workflow_selects_only_wd(self):
        self.assertEqual(REPRESENTATIONS, ("WD",))
        self.assertEqual(LEGACY_REPRESENTATIONS, ("UU", "UD", "WD"))
        self.assertEqual(FAMILIES, ("ER", "BA", "WS"))
        self.assertEqual(ACTIVE_SYNTHETIC_REALIZATIONS, 1)
        self.assertEqual(STAGE1_ETA_LEVELS, (0.9, 1.1))
        self.assertEqual(MATRIX_FORMAT, "WD_supplier_to_buyer_dependency_share")

    def test_empirical_builder_creates_only_weighted_supplier_to_buyer_matrix(self):
        Z = np.array([
            [10.0, 30.0, 0.0],
            [0.0, 20.0, 5.0],
            [0.0, 70.0, 15.0],
        ])
        B, purchases = _wd_from_flows(Z, threshold=0.2)
        self.assertEqual(B.shape, (3, 3))
        self.assertAlmostEqual(purchases[1], 120.0)
        self.assertAlmostEqual(B[0, 1], 0.25)
        self.assertEqual(B[1, 0], 0.0)
        self.assertEqual(B.diagonal().sum(), 0.0)
        self.assertTrue(np.all(B.data > 0))
        self.assertTrue(np.all(B.data <= 1.0))



    def test_small_synthetic_structural_pilot_generates_nine_wd_networks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "diagnostics").mkdir(parents=True, exist_ok=True)
            (root / "metadata").mkdir(parents=True, exist_ok=True)
            atomic_json(root / "diagnostics" / "real_structural_validation.json", {
                "status": "valid", "group": "real",
            })
            atomic_json(root / "metadata" / "empirical_wd_calibration.json", {
                "node_count": 30,
                "threshold": 0.001,
                "reference_mean_degree": 4.0,
                "reference_reciprocity": 0.2,
            })
            np.savez_compressed(
                root / "metadata" / "empirical_wd_calibration_samples.npz",
                empirical_share_samples=np.array([0.1, 0.2, 0.3]),
                empirical_dependency_total_samples=np.full(30, 0.5),
                empirical_indegree_samples=np.full(30, 2, dtype=int),
            )
            records = run_synthetic_structural(
                root, seed=42, path_sample_sources=5, betweenness_k=5,
            )
            self.assertEqual(len(records), 9)
            self.assertEqual(
                {record["network_id"] for record in records},
                {f"syn_{family}_{level}" for family in ("ER", "BA", "WS") for level in (1, 2, 3)},
            )
            self.assertTrue(all(record["matrix_format"] == MATRIX_FORMAT for record in records))
            self.assertTrue(all(record["realization_id"] == 1 for record in records))
            validation = validate_structural_group(root, "synthetic", expected_nodes=30)
            self.assertEqual(validation["status"], "valid")
            self.assertEqual(validation["observed_network_count"], 9)
            self.assertTrue((root / "manifests" / "synthetic_network_manifest.csv").is_file())


class TestRevisedSISModel(unittest.TestCase):
    def test_supplier_to_buyer_direction_and_model_a_update(self):
        B = sparse.csr_matrix(np.array([
            [0.0, 0.5],
            [0.0, 0.0],
        ]))
        next_state = simulate_one_step(B, np.array([1.0, 0.0]), beta=0.4, rho=0.2)
        # Supplier 0 recovers to 0.8; buyer 1 receives 0.4 * 0.5 = 0.2.
        self.assertTrue(np.allclose(next_state, [0.8, 0.2]))

    def test_threshold_is_network_specific_and_uses_spectral_radius(self):
        B = sparse.csr_matrix(np.array([
            [0.0, 0.5],
            [0.25, 0.0],
        ]))
        radius, beta_c = spectral_threshold(B, rho=0.2)
        expected_radius = np.sqrt(0.125)
        self.assertAlmostEqual(radius, expected_radius, places=10)
        self.assertAlmostEqual(beta_c, 0.2 / expected_radius, places=10)

    def test_wd_matrix_validation_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            validate_wd_matrix(sparse.csr_matrix([[0.0, 1.1], [0.0, 0.0]]))
        with self.assertRaises(ValueError):
            validate_wd_matrix(sparse.csr_matrix([[0.0, -0.1], [0.0, 0.0]]))
        with self.assertRaises(ValueError):
            validate_wd_matrix(sparse.csr_matrix([[0.1, 0.0], [0.0, 0.0]]))

    def test_single_node_metrics_known_decay_trajectory(self):
        B = sparse.csr_matrix((2, 2), dtype=float)
        result = simulate_single_node_batch(
            B, [0], beta=0.2, rho=0.25, max_steps=2,
            affected_threshold=0.01, extinction_tolerance=1e-12,
            convergence_tolerance=1e-12, consecutive_steps=2,
            long_run_window=2,
        ).iloc[0]
        # For a zero matrix the one shocked node decays: 1, .75, .5625.
        self.assertAlmostEqual(result["max_prevalence"], 0.5)
        self.assertEqual(result["time_to_peak"], 0)
        self.assertAlmostEqual(result["cumulative_prevalence"], 0.765625)
        self.assertEqual(result["max_affected_nodes"], 1)
        self.assertEqual(result["final_affected_nodes"], 1)
        self.assertAlmostEqual(result["long_run_prevalence"], 0.328125)
        self.assertFalse(bool(result["extinction_status"]))
        self.assertFalse(bool(result["converged"]))
        self.assertTrue(np.isnan(result["convergence_time"]))

    def test_batch_source_experiments_are_independent(self):
        B = sparse.csr_matrix(np.array([
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0],
        ]))
        batch = simulate_single_node_batch(
            B, [0, 1], beta=0.3, rho=0.2, max_steps=8,
            affected_threshold=0.01, extinction_tolerance=1e-10,
            convergence_tolerance=1e-10, consecutive_steps=2,
            long_run_window=3,
        )
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch["source_node_index"].tolist(), [0, 1])
        self.assertTrue(batch["max_prevalence"].between(0, 1).all())
        self.assertTrue(batch["long_run_prevalence"].between(0, 1).all())


class TestResumableBatches(unittest.TestCase):
    def test_completed_batches_resume_and_recover_atomic_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            B = sparse.csr_matrix((3, 3), dtype=float)
            matrix_path = root / "toy_wd.npz"
            save_npz(matrix_path, B)
            record = {
                "network_id": "toy_ER_1",
                "network_family": "ER",
                "network_configuration": "1",
                "year": None,
                "realization_id": 1,
                "node_count": 3,
                "matrix_sha256": sha256_file(matrix_path),
                "spectral_radius": 0.5,
                "beta_c": 0.4,
            }
            node_map = pd.DataFrame({
                "node_index": [0, 1, 2],
                "node_id": ["toy::node_0", "toy::node_1", "toy::node_2"],
                "node_label": ["node 0", "node 1", "node 2"],
            })
            output = root / "batches"
            first = _run_resumable_experiments(
                B, record, node_map, eta=0.9, beta=0.36, rho=0.2,
                output_dir=output, stage_name="census",
                batch_size=2, max_steps=3, long_run_window=2,
            )
            self.assertEqual(len(first), 3)
            self.assertEqual(first["source_node_id"].nunique(), 3)
            checkpoint_path = output / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            self.assertEqual(len(checkpoint["completed_batches"]), 2)

            # Simulate interruption after atomic part files were written but
            # before checkpoint metadata was persisted.
            checkpoint["completed_batches"] = []
            atomic_json(checkpoint_path, checkpoint)
            resumed = _run_resumable_experiments(
                B, record, node_map, eta=0.9, beta=0.36, rho=0.2,
                output_dir=output, stage_name="census",
                batch_size=2, max_steps=3, long_run_window=2,
            )
            self.assertEqual(len(resumed), 3)
            self.assertEqual(resumed["source_node_id"].nunique(), 3)
            recovered = json.loads(checkpoint_path.read_text())
            self.assertEqual(len(recovered["completed_batches"]), 2)
            self.assertTrue(all(x.get("recovered_after_atomic_write") for x in recovered["completed_batches"]))


if __name__ == "__main__":
    unittest.main()

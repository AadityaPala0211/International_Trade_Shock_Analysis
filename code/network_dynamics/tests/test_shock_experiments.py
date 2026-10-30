import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz

from network_dynamics.shock_experiments import (
    calculate_targeting_metrics, run_single_node_census,
    run_targeted_cohort_experiment, simulate_model_a,
    summarize_targeted_cohorts, centrality_impact_correlations,
)
from network_dynamics.run_shock_experiments import run as run_shock_experiments


class ShockExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.B = sparse.csr_matrix(np.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ]))
        self.labels = ["supplier", "processor", "downstream"]

    def test_supplier_to_buyer_exposure_and_pagerank_orientation(self):
        result = simulate_model_a(self.B, [1., 0., 0.], beta=.5, rho=.2, max_steps=3)
        self.assertGreater(result["exposure_mass"][1], 0)
        metrics = calculate_targeting_metrics(self.B, self.labels)
        self.assertGreater(metrics.loc[2, "pagerank_supplier_to_buyer"],
                           metrics.loc[0, "pagerank_supplier_to_buyer"])

    def test_cumulative_prevalence_uses_common_fixed_horizon(self):
        empty = sparse.csr_matrix((3, 3), dtype=float)
        scalar = simulate_model_a(
            empty, [1., 0., 0.], beta=0., rho=.2,
            max_steps=5, tolerance=.5)
        expected = 5 * (.8/3)
        self.assertTrue(scalar["converged"])
        self.assertEqual(scalar["steps"], 1)
        self.assertAlmostEqual(scalar["cumulative_prevalence_observed_until_stop"], .8/3)
        self.assertAlmostEqual(scalar["cumulative_prevalence"], expected)

        metrics = calculate_targeting_metrics(empty, self.labels)
        census, trajectories = run_single_node_census(
            empty, 0., .2, node_metrics=metrics, max_steps=5, tolerance=.5,
            store_trajectories=True, batch_size=2)
        self.assertTrue(census.converged.all())
        self.assertTrue(np.allclose(census.cumulative_prevalence, expected))
        lengths = trajectories.groupby("source_node_id").size()
        self.assertTrue((lengths == 6).all())

    def test_batched_census_enumerates_every_node_and_matches_scalar(self):
        metrics = calculate_targeting_metrics(self.B, self.labels)
        census, trajectories = run_single_node_census(
            self.B, .5, .2, node_metrics=metrics, max_steps=5,
            store_trajectories=True, batch_size=2)
        self.assertEqual(set(census.source_node_id), {0, 1, 2})
        self.assertEqual(census.source_node_id.nunique(), 3)
        self.assertEqual(set(trajectories.source_node_id), {0, 1, 2})
        for source in range(3):
            state = np.zeros(3)
            state[source] = 1
            scalar = simulate_model_a(self.B, state, .5, .2, max_steps=5)
            row = census[census.source_node_id == source].iloc[0]
            self.assertAlmostEqual(row.final_prevalence, scalar["final_prevalence"])
            self.assertAlmostEqual(row.cumulative_prevalence, scalar["cumulative_prevalence"])

    def test_targeted_groups_have_random_controls_and_correlations(self):
        base = calculate_targeting_metrics(self.B, self.labels)
        persisted = pd.DataFrame({
            "node_id": [0, 1, 2], "node": self.labels,
            "betweenness": [1., .5, 0.], "k_core": [1, 1, 0],
            "clustering": [0., 0., 0.],
        })
        metrics = calculate_targeting_metrics(self.B, self.labels, persisted)
        self.assertIn("betweenness", metrics.columns)
        runs = run_targeted_cohort_experiment(
            self.B, .5, .2, metrics, fraction=1/3, repeats=4,
            seed=17, max_steps=5, centralities=("supplier_out_degree",),
            initial_infection=.4)
        self.assertEqual(set(runs.cohort), {"top", "bottom", "random"})
        self.assertEqual(len(runs[runs.cohort == "random"]), 4)
        self.assertTrue(np.allclose(runs.initial_infection, .4))
        summary = summarize_targeted_cohorts(runs)
        self.assertIn("cumulative_prevalence_difference_vs_random", summary.columns)
        census, _ = run_single_node_census(self.B, .5, .2, node_metrics=base, max_steps=5)
        correlations = centrality_impact_correlations(census)
        self.assertGreater(len(correlations), 0)
        self.assertIn("p_value_fdr_bh", correlations.columns)

    def test_rejects_invalid_probability_and_bad_node_alignment(self):
        with self.assertRaises(ValueError):
            simulate_model_a(self.B, [1.1, 0., 0.], .5, .2)
        metrics = calculate_targeting_metrics(self.B, self.labels)
        metrics.loc[0, "node_id"] = 99
        with self.assertRaises(ValueError):
            run_single_node_census(self.B, .5, .2, node_metrics=metrics)

    def test_end_to_end_runner_uses_exact_node_index_and_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            (root / "metadata").mkdir(parents=True)
            pd.DataFrame({"node_id": [0, 1, 2], "node": self.labels}).to_csv(
                root / "metadata" / "node_index.csv", index=False)
            matrices = {
                "UU": self.B.maximum(self.B.T).tocsr(),
                "UD": self.B,
                "WD": self.B.astype(float) * 0.5,
            }
            manifest = []
            tracked_files = []
            for year in (2019, 2022):
                for representation, matrix in matrices.items():
                    folder = root / "empirical" / str(year) / representation
                    folder.mkdir(parents=True)
                    matrix_path = folder / "adjacency.npz"
                    save_npz(matrix_path, matrix)
                    metrics = pd.DataFrame({
                        "node_id": [0, 1, 2], "node": self.labels,
                        "in_degree": [0, 1, 1], "out_degree": [1, 1, 0],
                        "degree_undirected_projection": [1, 2, 1],
                        "in_strength": [0., 1., 1.], "out_strength": [1., 1., 0.],
                        "pagerank": [0.2, 0.3, 0.5], "betweenness": [0., 1., 0.],
                        "k_core": [1, 1, 1], "clustering": [0., 0., 0.],
                    })
                    metrics_path = folder / "node_metrics.csv.gz"
                    metrics.to_csv(metrics_path, index=False, compression="gzip")
                    tracked_files.extend([matrix_path, metrics_path])
                    manifest.append({
                        "network_id": f"real_{year}_{representation}",
                        "network_type": "real", "year": year,
                        "representation": representation, "n_nodes": 3,
                        "matrix_path": f"empirical/{year}/{representation}/adjacency.npz",
                    })
            pd.DataFrame(manifest).to_csv(root / "empirical_network_manifest.csv", index=False)

            def digest(file_path):
                return hashlib.sha256(file_path.read_bytes()).hexdigest()

            before = {str(file_path): digest(file_path) for file_path in tracked_files}
            output = root / "dynamics_node_attribution"
            run_shock_experiments([
                "--results-dir", str(root), "--out", str(output), "--scope", "real",
                "--rho", "0.2", "--beta-uu", "0.1", "--beta-ud", "0.1",
                "--beta-wd", "0.1", "--scenarios", "absolute",
                "--max-steps", "3", "--random-cohorts", "2",
                "--target-fraction", "0.33", "--census-batch-size", "2",
            ])
            result_manifest = pd.read_csv(output / "shock_experiment_manifest.csv")
            self.assertEqual(len(result_manifest), 6)
            self.assertTrue((result_manifest.single_node_sources_tested == 3).all())
            self.assertTrue((output / "top_20_single_node_shocks.csv.gz").is_file())
            first = pd.read_csv(
                output / "real_2019_UU" / "absolute" / "targeted_cohort_runs.csv.gz")
            paired = pd.read_csv(
                output / "real_2022_WD" / "absolute" / "targeted_cohort_runs.csv.gz")
            first_random = first[first.cohort == "random"].sort_values("repeat").seed_node_ids.tolist()
            paired_random = paired[paired.cohort == "random"].sort_values("repeat").seed_node_ids.tolist()
            self.assertEqual(first_random, paired_random)
            self.assertFalse((output / "empirical").exists())
            after = {str(file_path): digest(file_path) for file_path in tracked_files}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from network_dynamics.shock_experiments import (
    calculate_targeting_metrics, run_single_node_census,
    run_targeted_cohort_experiment, simulate_model_a,
    summarize_targeted_cohorts, centrality_impact_correlations,
)


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
            seed=17, max_steps=5, centralities=("supplier_out_degree",))
        self.assertEqual(set(runs.cohort), {"top", "bottom", "random"})
        self.assertEqual(len(runs[runs.cohort == "random"]), 4)
        summary = summarize_targeted_cohorts(runs)
        self.assertIn("cumulative_prevalence_difference_vs_random", summary.columns)
        census, _ = run_single_node_census(self.B, .5, .2, node_metrics=base, max_steps=5)
        correlations = centrality_impact_correlations(census)
        self.assertGreater(len(correlations), 0)

    def test_rejects_invalid_probability_and_bad_node_alignment(self):
        with self.assertRaises(ValueError):
            simulate_model_a(self.B, [1.1, 0., 0.], .5, .2)
        metrics = calculate_targeting_metrics(self.B, self.labels)
        metrics.loc[0, "node_id"] = 99
        with self.assertRaises(ValueError):
            run_single_node_census(self.B, .5, .2, node_metrics=metrics)


if __name__ == "__main__":
    unittest.main()

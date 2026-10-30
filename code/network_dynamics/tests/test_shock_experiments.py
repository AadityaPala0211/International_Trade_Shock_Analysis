import unittest
import numpy as np
from scipy import sparse

from network_dynamics.shock_experiments import (
    calculate_targeting_metrics, run_single_node_census,
    run_targeted_cohort_experiment, simulate_model_a,
    summarize_targeted_cohorts,
)


class ShockExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.B = sparse.csr_matrix(np.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ]))
        self.labels = ["supplier", "processor", "downstream"]

    def test_supplier_to_buyer_orientation(self):
        result = simulate_model_a(self.B, [1., 0., 0.], beta=.5, rho=.2, max_steps=3)
        self.assertGreater(result["exposure_mass"][1], 0)

    def test_census_enumerates_every_node_once(self):
        metrics = calculate_targeting_metrics(self.B, self.labels)
        census, trajectories = run_single_node_census(
            self.B, .5, .2, node_metrics=metrics, max_steps=5, store_trajectories=True)
        self.assertEqual(set(census.source_node_id), {0, 1, 2})
        self.assertEqual(census.source_node_id.nunique(), 3)
        self.assertEqual(set(trajectories.source_node_id), {0, 1, 2})

    def test_targeted_groups_have_random_controls(self):
        metrics = calculate_targeting_metrics(self.B, self.labels)
        runs = run_targeted_cohort_experiment(
            self.B, .5, .2, metrics, fraction=1/3, repeats=4,
            seed=17, max_steps=5, centralities=("supplier_out_degree",))
        self.assertEqual(set(runs.cohort), {"top", "bottom", "random"})
        self.assertEqual(len(runs[runs.cohort == "random"]), 4)
        summary = summarize_targeted_cohorts(runs)
        self.assertIn("cumulative_prevalence_difference_vs_random", summary.columns)

    def test_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            simulate_model_a(self.B, [1.1, 0., 0.], .5, .2)


if __name__ == "__main__":
    unittest.main()

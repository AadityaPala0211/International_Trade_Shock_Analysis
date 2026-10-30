import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
from network_structural_analysis.config import SyntheticConfig
from network_structural_analysis.network_builder import dependency_shares, make_representations
from network_structural_analysis.thresholding import diagnose_thresholds
from network_structural_analysis.synthetic import (
    generate_skeleton, _orientation_from_undirected,
    _calibrate_weighted_adjacency, run_synthetic_ensemble,
)
from network_structural_analysis.metrics import structural_analysis
from network_structural_analysis.data_loader import _find_contiguous_numeric_block
from network_structural_analysis.provenance import inspect_icio_readme, summarize_icio_provenance
from network_structural_analysis.dynamics import simulate_sis, critical_beta, build_seed_sets


class TestNetworkPipeline(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[10., 30., 0.], [0., 20., 5.], [0., 70., 15.]])

    def test_dependency_share_and_paired_representations(self):
        S, purchases = dependency_shares(self.Z)
        self.assertAlmostEqual(S[0, 1], 30 / 120)
        self.assertAlmostEqual(S[2, 1], 70 / 120)
        self.assertEqual(S[1, 1], 0)
        self.assertAlmostEqual(purchases[1], 120)
        reps = make_representations(S, .2)
        self.assertEqual(reps["UD"][0, 1], 1)
        self.assertAlmostEqual(reps["WD"][0, 1], .25)
        self.assertEqual(reps["UU"][1, 0], 1)
        self.assertEqual(reps["UD"][1, 0], 0)

    def test_thresholds_are_never_automatically_selected(self):
        diag, pooled, selection, shares = diagnose_thresholds(
            {2019: self.Z, 2022: self.Z}, [0, .1, .2])
        self.assertIsNone(selection["selected_threshold"])
        self.assertFalse(selection["automatic_threshold_selection"])
        self.assertEqual(len(diag), 6)
        self.assertIn("mean_undirected_degree", pooled.columns)
        self.assertIn("out_degree_q99", diag.columns)

    def test_threshold_manual_selection(self):
        _, _, selection, _ = diagnose_thresholds(
            {2019: self.Z, 2022: self.Z}, [0, .1, .2], selected_threshold=.1)
        self.assertEqual(selection["selected_threshold"], .1)

    def test_icio_block_loader_and_column_label_reordering(self):
        raw = pd.DataFrame([
            ["node", "B", "A", "C", "FD"],
            ["A", "10", "20", "30", "4"],
            ["B", "40", "50", "60", "4"],
            ["C", "70", "80", "90", "4"],
        ], dtype=str)
        Z, labels, bounds, shape = _find_contiguous_numeric_block(raw, 3)
        self.assertEqual(labels, ["A", "B", "C"])
        self.assertTrue(np.array_equal(Z[0], [20., 10., 30.]))
        bad = raw.copy()
        bad.iloc[0, 1] = "WRONG"
        with self.assertRaises(ValueError):
            _find_contiguous_numeric_block(bad, 3)

    def test_synthetic_configurations_and_weights(self):
        cfg = SyntheticConfig()
        self.assertEqual(cfg.mean_degree_multipliers, (.5, 1., 1.5))
        self.assertEqual(cfg.ba_multipliers, (.5, 1., 1.5))
        self.assertEqual(cfg.ws_rewire_probabilities, (.01, .1, .5))
        graph, params = generate_skeleton("ER", 4050, 8., .5, .1, 42)
        self.assertEqual(graph.number_of_nodes(), 4050)
        self.assertAlmostEqual(params["p"], 4. / 4049)
        graph, params = generate_skeleton("BA", 4050, 8., 1.5, .1, 42)
        self.assertEqual(params["m"], 6)
        graph, params = generate_skeleton("WS", 100, 12., 1., .1, 42)
        self.assertEqual(params["k"], 12)

        A = sparse.csr_matrix(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8))
        totals, indegrees = np.array([0., .2, .5]), np.array([0, 1, 2])
        W, sampled = _calibrate_weighted_adjacency(
            A, [.1, .2, .3], totals, 12, empirical_indegree_samples=indegrees)
        self.assertTrue(np.allclose(np.asarray(W.sum(axis=0)).ravel(), sampled))
        self.assertEqual(sampled[0], 0.)

    def test_orientation_reciprocity_calibration(self):
        graph = nx.gnp_random_graph(500, .04, seed=12)
        directed = _orientation_from_undirected(graph, .4, np.random.default_rng(1))
        self.assertLess(abs(nx.reciprocity(directed) - .4), .08)

    def test_synthetic_matrices_and_node_metrics_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SyntheticConfig(realizations=1, seed=8)
            summaries = run_synthetic_ensemble(
                30, 8., .3, np.array([.1, .2, .3]), realizations=1, seed=8, out_dir=tmp,
                empirical_dependency_total_samples=np.array([0., .2, .5, .8]),
                empirical_indegree_samples=np.array([0, 1, 2, 3]), config=cfg,
                path_sample_sources=10, betweenness_k=10)
            manifest = pd.read_csv(Path(tmp) / "synthetic_network_manifest.csv")
            self.assertEqual(len(manifest), 27)
            self.assertEqual(len(summaries), 27)
            self.assertTrue(all((Path(tmp) / p).exists() for p in manifest.matrix_path))
            self.assertTrue(all((Path(tmp) / p).exists() for p in manifest.node_metrics_path))

    def test_sis_update_threshold_and_seed_sets(self):
        B = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        result = simulate_sis(B, [1., 0.], beta=.5, rho=.2, max_steps=1)
        self.assertTrue(np.allclose(result["prevalence"], [.5, .65]))
        lam, beta_c = critical_beta(B, .2)
        self.assertAlmostEqual(lam, 1., places=5)
        self.assertAlmostEqual(beta_c, .2, places=5)
        seeds = build_seed_sets(4050, 5, 123)
        self.assertTrue(all(len(x) == 41 for x in seeds))

    def test_structural_metrics_match_networkx_on_toy_graph(self):
        A = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8))
        summary, nodes = structural_analysis(A, ["a", "b", "c"], "UD",
                                             seed=3, path_sample_sources=3, betweenness_k=3)
        graph = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
        self.assertAlmostEqual(nodes.loc[1, "betweenness"],
                               nx.betweenness_centrality(graph)[1], places=10)
        self.assertAlmostEqual(nodes.loc[1, "pagerank"], nx.pagerank(graph)[1], places=5)
        self.assertEqual(summary["strong_component_count"], 3)

    def test_icio_companion_workbook_provenance_audit(self):
        from openpyxl import Workbook
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "readme.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "OECD ICIO 2025 metadata"
            ws.append(["Edition", "2025"])
            wb.save(path)
            workbook = inspect_icio_readme(path)
            audit = summarize_icio_provenance(workbook, [{
                "source_path": "2019.csv", "source_sha256": "abc",
                "Z": np.zeros((2, 2)), "n_nodes": 2}])
            self.assertEqual(audit["local_companion_workbook_status"],
                             "companion_workbook_mentions_oecd_icio_2025")
            self.assertIn("cannot by itself prove", audit["what_is_not_independently_proven"])


if __name__ == "__main__":
    unittest.main()

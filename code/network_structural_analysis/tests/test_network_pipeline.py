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
    _calibrate_weighted_adjacency, run_synthetic_ensemble, empirical_calibration,
)
from network_structural_analysis.metrics import structural_analysis
from network_structural_analysis.data_loader import _find_contiguous_numeric_block
from network_structural_analysis.provenance import inspect_icio_readme, summarize_icio_provenance
from network_structural_analysis.dynamics import simulate_sis, critical_beta, build_seed_sets, _write_ensemble_summary, _save_trajectory_summaries_and_plots, _ci
from network_structural_analysis.ensemble_validation import validate_network_manifests


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

    def test_dependency_share_threshold_rejects_values_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            diagnose_thresholds({2019: self.Z, 2022: self.Z}, [1.1])
        with self.assertRaises(ValueError):
            make_representations(np.zeros((3, 3)), 1.1)

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
        self.assertEqual(cfg.realizations, 10)
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

    def test_empirical_reciprocity_is_pooled_by_arc_count(self):
        calibration = empirical_calibration([
            {"year": 2019, "representation": "UU", "mean_degree_undirected_projection": 10.0},
            {"year": 2022, "representation": "UU", "mean_degree_undirected_projection": 12.0},
            {"year": 2019, "representation": "UD", "reciprocity": 0.2, "edges": 100},
            {"year": 2022, "representation": "UD", "reciprocity": 0.8, "edges": 300},
        ])
        self.assertAlmostEqual(calibration["reference_mean_degree"], 11.0)
        self.assertAlmostEqual(calibration["reference_reciprocity"], 0.65)
        self.assertEqual(calibration["reciprocity_calibration_arc_counts"], {"2019": 100, "2022": 300})

    def test_orientation_reciprocity_calibration(self):
        graph = nx.gnp_random_graph(500, .04, seed=12)
        directed = _orientation_from_undirected(graph, .4, np.random.default_rng(1))
        self.assertLess(abs(nx.reciprocity(directed) - .4), .08)

    def test_synthetic_matrices_and_node_metrics_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A rerun with a smaller R must remove stale files from older attempts.
            stale_dir = Path(tmp) / "matrices" / "ER" / "config_1" / "realization_999"
            stale_dir.mkdir(parents=True)
            (stale_dir / "stale.npz").write_text("stale")
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
            self.assertFalse(stale_dir.exists())
            self.assertEqual(len(list((Path(tmp) / "matrices").rglob("*.npz"))), 27)

    def test_reduced_realization_manifests_validate_and_orphans_warn(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 4
            ud = sparse.csr_matrix(np.array([
                [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0]
            ], dtype=np.uint8))
            uu = ud.maximum(ud.T).tocsr()
            wd = ud.astype(float) * 0.2
            matrices = {"UU": uu, "UD": ud, "WD": wd}
            real_rows, synthetic_rows = [], []
            for year in (2019, 2022):
                for rep, matrix in matrices.items():
                    path = root / "empirical" / str(year) / rep / "adjacency.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    from scipy.sparse import save_npz
                    save_npz(path, matrix)
                    real_rows.append({"network_id": f"real_{year}_{rep}", "year": year,
                        "representation": rep, "n_nodes": n,
                        "matrix_path": path.relative_to(root).as_posix()})
            pd.DataFrame(real_rows).to_csv(root / "empirical_network_manifest.csv", index=False)
            for family in ("ER", "BA", "WS"):
                for level in (1, 2, 3):
                    for rep, matrix in matrices.items():
                        path = root / "synthetic" / "matrices" / family / f"config_{level}" / "realization_001" / f"{rep}.npz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        save_npz(path, matrix)
                        synthetic_rows.append({
                            "network_id": f"syn_{family}_{level}_001_{rep}",
                            "family": family, "configuration": level, "level": level,
                            "realization": 1, "representation": rep,
                            "matrix_path": path.relative_to(root / "synthetic").as_posix(),
                        })
            synthetic_root = root / "synthetic"
            pd.DataFrame(synthetic_rows).to_csv(
                synthetic_root / "synthetic_network_manifest.csv", index=False)
            (synthetic_root / "synthetic_experiment_design.json").write_text(json.dumps({
                "realizations_per_configuration": 1,
                "independent_skeletons": 9,
                "paired_representation_matrices": 27,
            }))
            metadata = root / "metadata"
            metadata.mkdir()
            (metadata / "synthetic_stage_complete.json").write_text(json.dumps({
                "realization_count_per_configuration": 1,
                "independent_skeletons": 9,
                "paired_representation_matrices": 27,
            }))
            # A stale, unmanifested matrix is warned about but never consumed.
            stale = synthetic_root / "matrices" / "ER" / "config_1" / "realization_999" / "stale.npz"
            stale.parent.mkdir(parents=True)
            save_npz(stale, ud)
            report = validate_network_manifests(root, scope="all", expected_nodes=n)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["empirical_matrix_count"], 6)
            self.assertEqual(report["synthetic_manifest_rows"], 27)
            self.assertEqual(report["synthetic_realizations_per_configuration"]["ER_level_1"], 1)
            self.assertEqual(report["orphan_matrix_files"], 1)
            self.assertTrue(any("one synthetic realization" in x for x in report["warnings"]))
            mismatch = validate_network_manifests(
                root, scope="synthetic", expected_nodes=n, expected_realizations=10)
            self.assertFalse(mismatch["ok"])
            self.assertTrue(any("expected-realizations requested 10" in x for x in mismatch["errors"]))

    def test_single_observation_does_not_claim_zero_width_confidence_interval(self):
        mean, sd, low, high, count = _ci([0.25])
        self.assertEqual(mean, 0.25)
        self.assertEqual(count, 1)
        self.assertTrue(np.isnan(sd))
        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))

    def test_sis_update_threshold_and_seed_sets(self):
        B = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        result = simulate_sis(B, [1., 0.], beta=.5, rho=.2, max_steps=1)
        self.assertTrue(np.allclose(result["prevalence"], [.5, .65]))
        lam, beta_c = critical_beta(B, .2)
        self.assertAlmostEqual(lam, 1., places=5)
        self.assertAlmostEqual(beta_c, .2, places=5)
        seeds = build_seed_sets(4050, 5, 123)
        self.assertTrue(all(len(x) == 41 for x in seeds))

    def test_synthetic_dynamics_summary_reports_multiple_graph_level_cis(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame({
                "network_id": ["g1", "g2"], "family": ["ER", "ER"],
                "configuration": [1, 1], "representation": ["UD", "UD"],
                "scenario": ["eta_1", "eta_1"], "eta": [1.0, 1.0],
                "peak_prevalence_mean_across_seed_sets": [0.2, 0.4],
                "time_to_peak_mean_across_seed_sets": [5.0, 7.0],
                "long_run_prevalence_mean_across_seed_sets": [0.1, 0.3],
                "cumulative_prevalence_mean_across_seed_sets": [3.0, 4.0],
                "final_prevalence_mean_across_seed_sets": [0.1, 0.3],
                "max_mean_exposure_mean_across_seed_sets": [0.4, 0.6],
                "extinction_fraction": [0.5, 0.0], "convergence_fraction": [0.5, 1.0],
            })
            path = Path(tmp) / "summary.csv"
            _write_ensemble_summary(frame, ["family", "configuration", "representation", "scenario", "eta"], path)
            summary = pd.read_csv(path)
            self.assertEqual(len(summary), 1)
            self.assertIn("peak_prevalence_ci95_low_across_graph_realizations", summary.columns)
            self.assertIn("long_run_prevalence_mean_across_graph_realizations", summary.columns)
            self.assertIn("extinction_fraction_mean_across_graph_realizations", summary.columns)

    def test_dynamics_trajectory_summaries_and_plots_for_real_and_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for year in (2019, 2022):
                for time in (0, 1):
                    rows.append({"network_id": f"real_{year}_UU", "network_type": "real",
                        "year": year, "family": None, "configuration": None, "realization": None,
                        "representation": "UU", "scenario": "absolute", "eta": None, "time": time,
                        "prevalence_mean": .2 + time * .1,
                        "prevalence_ci95_seed_low": .15 + time * .1,
                        "prevalence_ci95_seed_high": .25 + time * .1, "mean_exposure": .3})
            for config in (1, 2):
                for realization in (1, 2):
                    for time in (0, 1):
                        rows.append({"network_id": f"syn_ER_{config}_{realization}_UU",
                            "network_type": "synthetic", "year": None, "family": "ER",
                            "configuration": config, "realization": realization,
                            "representation": "UU", "scenario": "absolute", "eta": None, "time": time,
                            "prevalence_mean": .1 * config + .01 * realization + .05 * time,
                            "prevalence_ci95_seed_low": .1, "prevalence_ci95_seed_high": .2,
                            "mean_exposure": .2})
            trajectory = Path(tmp) / "trajectories.csv.gz"
            pd.DataFrame(rows).to_csv(trajectory, index=False, compression="gzip")
            output = Path(tmp) / "summaries"
            _save_trajectory_summaries_and_plots(trajectory, output)
            self.assertTrue((output / "real_trajectory_summary.csv.gz").exists())
            self.assertTrue((output / "synthetic_trajectory_summary.csv.gz").exists())
            self.assertTrue((output / "plots" / "real" / "trajectory_UU_absolute.png").exists())
            self.assertTrue((output / "plots" / "synthetic" / "trajectory_ER_UU_absolute.png").exists())

    def test_structural_metrics_match_networkx_on_toy_graph(self):
        A = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8))
        summary, nodes = structural_analysis(A, ["a", "b", "c"], "UD",
                                             seed=3, path_sample_sources=3, betweenness_k=3)
        graph = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
        self.assertAlmostEqual(nodes.loc[1, "betweenness"],
                               nx.betweenness_centrality(graph)[1], places=10)
        self.assertAlmostEqual(nodes.loc[1, "pagerank"], nx.pagerank(graph)[1], places=5)
        self.assertEqual(summary["strong_component_count"], 3)

    def test_undirected_betweenness_normalization_matches_networkx(self):
        A = sparse.csr_matrix(np.array([
            [0, 1, 0], [1, 0, 1], [0, 1, 0]
        ], dtype=np.uint8))
        _, nodes = structural_analysis(A, ["a", "b", "c"], "UU",
                                       seed=4, path_sample_sources=3, betweenness_k=3)
        graph = nx.from_scipy_sparse_array(A)
        expected = nx.betweenness_centrality(graph)
        self.assertTrue(np.allclose(nodes["betweenness"].to_numpy(),
                                    np.array([expected[0], expected[1], expected[2]])))

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

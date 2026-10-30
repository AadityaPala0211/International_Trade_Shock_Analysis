import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
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
from network_structural_analysis.dynamics import (
    simulate_sis, simulate_sis_ensemble, critical_beta, build_seed_sets,
    _write_ensemble_summary, _save_trajectory_summaries_and_plots,
    validate_network_records, run_dynamics_analysis,
)


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

    def _write_complete_dynamics_fixture(self, root):
        root = Path(root)
        ud = sparse.csr_matrix(np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ], dtype=np.uint8))
        uu = ud.maximum(ud.T).tocsr()
        coo = ud.tocoo()
        wd = sparse.csr_matrix(
            (np.array([0.2, 0.3, 0.4, 0.5]), (coo.row, coo.col)),
            shape=ud.shape,
        )
        matrices = {"UU": uu, "UD": ud, "WD": wd}
        records = []

        for year in (2019, 2022):
            for rep, matrix in matrices.items():
                path = root / "empirical" / str(year) / rep / "adjacency.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_npz(path, matrix)
                records.append({
                    "network_id": f"real_{year}_{rep}", "network_type": "real",
                    "year": year, "representation": rep, "matrix_path": str(path),
                })

        for family in ("ER", "BA", "WS"):
            for configuration in (1, 2, 3):
                realization = 1
                group_seeds = {
                    "graph_seed": 1000 + configuration,
                    "orientation_seed": 2000 + configuration,
                    "weight_seed": 3000 + configuration,
                }
                for rep, matrix in matrices.items():
                    path = (root / "synthetic" / "matrices" / family /
                            f"config_{configuration}" / "realization_001" / f"{rep}.npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_npz(path, matrix)
                    records.append({
                        "network_id": f"syn_{family}_{configuration}_001_{rep}",
                        "network_type": "synthetic", "family": family,
                        "configuration": configuration, "realization": realization,
                        "representation": rep, "matrix_path": str(path),
                        **group_seeds,
                    })
        return records, matrices

    def test_batched_sis_matches_individual_seed_simulations(self):
        B = sparse.csr_matrix(np.array([
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
        ]))
        initial_states = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])
        batched = simulate_sis_ensemble(
            B, initial_states, beta=.35, rho=.2, max_steps=30, tolerance=1e-7)
        individual = [
            simulate_sis(B, initial_states[:, j], beta=.35, rho=.2,
                         max_steps=30, tolerance=1e-7)
            for j in range(initial_states.shape[1])
        ]
        self.assertEqual(len(batched), len(individual))
        for actual, expected in zip(batched, individual):
            self.assertTrue(np.allclose(actual["prevalence"], expected["prevalence"]))
            self.assertTrue(np.allclose(actual["mean_exposure"], expected["mean_exposure"]))
            self.assertTrue(np.allclose(actual["total_exposure"], expected["total_exposure"]))
            self.assertEqual(actual["iterations"], expected["iterations"])
            self.assertEqual(actual["extinct"], expected["extinct"])
            self.assertEqual(actual["converged"], expected["converged"])

    def test_sis_directed_force_flows_from_supplier_to_buyer(self):
        B = sparse.csr_matrix(np.array([[0., 1.], [0., 0.]]))
        result = simulate_sis(B, [1., 0.], beta=1.0, rho=.25, max_steps=1)
        self.assertTrue(np.allclose(result["final_state"], [.75, 1.0]))
        self.assertAlmostEqual(result["prevalence"][-1], .875)

    def test_sis_manifest_precheck_validates_full_design_semantics_on_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, matrices = self._write_complete_dynamics_fixture(tmp)
            normalised, diagnostics, report = validate_network_records(
                records, rho=.2, expected_scope="all",
                require_full_design=True, expected_nodes=4,
            )
            self.assertEqual(len(normalised), 33)
            self.assertEqual(len(diagnostics), 33)
            self.assertEqual(report["real_network_count"], 6)
            self.assertEqual(report["synthetic_network_count"], 27)
            self.assertEqual(report["synthetic_skeleton_count"], 9)
            self.assertEqual(report["synthetic_configuration_count"], 9)
            self.assertEqual(set(report["beta_critical_by_representation"]), {"UU", "UD", "WD"})
            self.assertEqual(set(diagnostics["n_nodes"]), {4})

    def test_sis_precheck_rejects_unpaired_synthetic_weighted_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, matrices = self._write_complete_dynamics_fixture(tmp)
            target = next(
                r for r in records
                if r["network_type"] == "synthetic" and r["family"] == "ER"
                and r["configuration"] == 1 and r["representation"] == "WD"
            )
            corrupted = matrices["WD"].tolil()
            corrupted[0, 1] = 0.0
            save_npz(target["matrix_path"], corrupted.tocsr())
            with self.assertRaisesRegex(ValueError, "WD and UD must have identical"):
                validate_network_records(
                    records, rho=.2, expected_scope="all",
                    require_full_design=True, expected_nodes=4,
                )

    def test_full_dynamics_end_to_end_real_and_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, _ = self._write_complete_dynamics_fixture(tmp)
            output = Path(tmp) / "dynamics" / "all"
            network_results, seed_results = run_dynamics_analysis(
                records, output, seed=42, rho=.2,
                beta_by_rep={"UU": .05, "UD": .05, "WD": .2},
                max_steps=5, tolerance=1e-7,
                real_seed_sets=2, synthetic_seed_sets=2,
                infection_fraction=.25, expected_scope="all",
                require_full_design=True, expected_nodes=4, write_plots=False,
            )
            self.assertEqual(network_results["network_id"].nunique(), 33)
            self.assertEqual(network_results["scenario"].nunique(), 7)
            self.assertEqual(len(network_results), 33 * 7)
            self.assertEqual(len(seed_results), 33 * 7 * 2)
            self.assertTrue((output / "dynamics_input_precheck.csv").exists())
            self.assertTrue((output / "dynamics_input_validation.json").exists())
            self.assertTrue((output / "dynamics_seed_results.csv.gz").exists())
            self.assertTrue((output / "dynamics_network_results.csv.gz").exists())
            self.assertTrue((output / "dynamics_trajectories.csv.gz").exists())
            self.assertTrue((output / "initial_seed_sets.csv.gz").exists())
            self.assertTrue((output / "network_seed_group_assignments.csv").exists())
            self.assertTrue((output / "synthetic_dynamics_ensemble_summary.csv").exists())
            self.assertTrue((output / "real_dynamics_seedset_summary.csv").exists())
            assignments = pd.read_csv(output / "network_seed_group_assignments.csv")
            real_assignments = assignments[assignments.network_type == "real"]
            self.assertEqual(real_assignments["seed_group"].nunique(), 1)
            self.assertEqual(len(real_assignments), 6)
            synthetic_assignments = assignments[assignments.network_type == "synthetic"]
            self.assertEqual(synthetic_assignments["seed_group"].nunique(), 9)
            seed_file = pd.read_csv(output / "initial_seed_sets.csv.gz")
            self.assertEqual(len(seed_file[seed_file.network_type == "real"]), 2)
            self.assertEqual(len(seed_file[seed_file.network_type == "synthetic"]), 18)
            self.assertTrue(np.isfinite(network_results["peak_prevalence_mean_across_seed_sets"]).all())

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
                        "prevalence_ci95_seed_high": .25 + time * .1, "mean_exposure": .3,
                        "total_exposure": .6})
            for config in (1, 2):
                for realization in (1, 2):
                    for time in (0, 1):
                        rows.append({"network_id": f"syn_ER_{config}_{realization}_UU",
                            "network_type": "synthetic", "year": None, "family": "ER",
                            "configuration": config, "realization": realization,
                            "representation": "UU", "scenario": "absolute", "eta": None, "time": time,
                            "prevalence_mean": .1 * config + .01 * realization + .05 * time,
                            "prevalence_ci95_seed_low": .1, "prevalence_ci95_seed_high": .2,
                            "mean_exposure": .2, "total_exposure": .4})
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

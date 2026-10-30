import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from network_structural_analysis.network_builder import dependency_shares, make_representations
from network_structural_analysis.thresholding import diagnose_thresholds
from network_structural_analysis.synthetic import (
    generate_skeleton, _orientation_from_undirected, _calibrate_weighted_adjacency,
)
from network_structural_analysis.metrics import structural_analysis
from network_structural_analysis.data_loader import _find_contiguous_numeric_block
from network_structural_analysis.provenance import inspect_icio_readme

class TestNetworkPipeline(unittest.TestCase):
    def setUp(self):
        self.Z=np.array([[10.,30.,0.],[0.,20.,5.],[0.,70.,15.]])

    def test_dependency_share_orientation_and_self_use_denominator(self):
        S,p=dependency_shares(self.Z)
        self.assertAlmostEqual(S[0,1],30/120)
        self.assertAlmostEqual(S[2,1],70/120)
        self.assertEqual(S[1,1],0.0)
        self.assertAlmostEqual(p[1],120.)

    def test_three_representations(self):
        S,_=dependency_shares(self.Z); reps=make_representations(S,0.2)
        self.assertEqual(reps["UD"][0,1],1)
        self.assertEqual(reps["WD"][0,1],0.25)
        self.assertEqual(reps["UU"][1,0],1)
        self.assertEqual(reps["UD"][1,0],0)
        self.assertEqual(reps["WD"].diagonal().sum(),0)

    def test_threshold_diagnostics_returns_candidate(self):
        diag,pooled,rec,shares=diagnose_thresholds({2019:self.Z,2022:self.Z},[0,0.1,0.2])
        self.assertIn(rec["recommended_threshold"],[0,0.1,0.2])
        self.assertEqual(len(diag),6)
        self.assertIn("local_sensitivity_thresholds",rec)

    def test_structural_metric_outputs(self):
        S,_=dependency_shares(self.Z); reps=make_representations(S,0.1)
        sm,node=structural_analysis(reps["WD"],["A_X","B_Y","C_Z"],"WD",path_sample_sources=3,betweenness_k=3)
        self.assertEqual(sm["nodes"],3); self.assertEqual(len(node),3)
        self.assertGreaterEqual(sm["density"],0)

    def test_icio_loader_extracts_square_intermediate_block_before_final_demand(self):
        labels=[f"C{i}_S{i}" for i in range(5)]
        rows=[["node"]+labels+["FD_A","FD_B"]]
        for i,label in enumerate(labels):
            rows.append([label]+[str(i*5+j+1) for j in range(5)]+["10","20"])
        rows.extend([[f"FD_ROW_{i}"]+[str(100+i+j) for j in range(5)]+["10","20"] for i in range(2)])
        raw=pd.DataFrame(rows,dtype=str)
        Z,found,bounds,shape=_find_contiguous_numeric_block(raw,5)
        self.assertEqual(found,labels)
        self.assertEqual(Z.shape,(5,5))
        self.assertEqual(Z[4,4],25.0)
        self.assertEqual(bounds,(1,6,1,6))
        bad_rows=[rows[0].copy()]
        bad_rows[0][1],bad_rows[0][2]=bad_rows[0][2],bad_rows[0][1]
        bad_rows.extend(rows[1:])
        with self.assertRaises(ValueError):
            _find_contiguous_numeric_block(pd.DataFrame(bad_rows,dtype=str),5)

    def test_synthetic_graphs_at_target_n(self):
        for family in ["ER","BA","WS"]:
            g,p=generate_skeleton(family,4050,8.0,1.0,0.1,42)
            self.assertEqual(g.number_of_nodes(),4050)

    def test_synthetic_directed_reciprocity_calibration(self):
        import networkx as nx
        G=nx.gnp_random_graph(500,0.04,seed=12)
        D=_orientation_from_undirected(G,0.4,np.random.default_rng(1))
        self.assertLess(abs(nx.reciprocity(D)-0.4),0.08)

    def test_synthetic_weighted_buyer_totals_are_calibrated(self):
        A=sparse.csr_matrix(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.uint8))
        target=np.array([0.2,0.4,0.6]); shares=np.array([0.1,0.2,0.3])
        W=_calibrate_weighted_adjacency(A,shares,target,seed=12)
        sums=np.asarray(W.sum(axis=0)).ravel()
        rng=np.random.default_rng(12)
        rng.choice(shares,size=A.nnz,replace=True)
        expected=rng.choice(target,size=A.shape[1],replace=True)
        self.assertTrue(np.allclose(sums,expected))

    def test_sparse_metrics_match_networkx_on_known_graphs(self):
        import networkx as nx
        A=sparse.csr_matrix(np.array([[0,1,0],[0,0,1],[0,0,0]],dtype=np.uint8))
        summary,node=structural_analysis(A,["a","b","c"],"UD",seed=3,path_sample_sources=3,betweenness_k=3)
        G=nx.from_scipy_sparse_array(A,create_using=nx.DiGraph)
        expected=nx.betweenness_centrality(G)
        self.assertAlmostEqual(node.loc[1,"betweenness"],expected[1],places=10)
        self.assertAlmostEqual(node.loc[1,"pagerank"],nx.pagerank(G)[1],places=5)
        self.assertEqual(summary["strong_component_count"],3)
        B=sparse.csr_matrix(np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=np.uint8))
        _,nodes=structural_analysis(B,["a","b","c"],"UU",seed=3,path_sample_sources=3,betweenness_k=3)
        H=nx.from_scipy_sparse_array(B)
        self.assertTrue(np.allclose(nodes["clustering"],list(nx.clustering(H).values())))
        self.assertTrue(np.allclose(nodes["k_core"],list(nx.core_number(H).values())))

    def test_ws_configurations_keep_empirical_degree_fixed(self):
        for _ in (0.75,1.0,1.25):
            graph,params=generate_skeleton("WS",100,12.0,1.0,0.1,42)
            self.assertEqual(params["k"],12)

    def test_optional_icio_readme_provenance_extraction(self):
        import tempfile
        from openpyxl import Workbook
        with tempfile.TemporaryDirectory() as tmp:
            path=f"{tmp}/readme.xlsx"
            wb=Workbook(); ws=wb.active; ws.title="ICIO metadata"
            ws.append(["Edition","2022"]); ws.append(["Countries",81]); wb.save(path)
            info=inspect_icio_readme(path)
            self.assertTrue(info["available"])
            self.assertIn("ICIO metadata",info["sheet_names"])
            self.assertEqual(info["sheets"][0]["preview"][0][0],"Edition")

if __name__=="__main__":
    unittest.main()

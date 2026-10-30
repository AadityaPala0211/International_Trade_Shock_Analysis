import unittest
import numpy as np
import pandas as pd
from network_structural_analysis.network_builder import dependency_shares, make_representations
from network_structural_analysis.thresholding import diagnose_thresholds
from network_structural_analysis.synthetic import generate_skeleton, _orient
from network_structural_analysis.metrics import structural_analysis
from network_structural_analysis.data_loader import _find_contiguous_numeric_block

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

    def test_structural_metric_outputs(self):
        S,_=dependency_shares(self.Z); reps=make_representations(S,0.1)
        sm,node=structural_analysis(reps["WD"],["A_X","B_Y","C_Z"],"WD",path_sample_sources=3,betweenness_k=3)
        self.assertEqual(sm["nodes"],3); self.assertEqual(len(node),3)
        self.assertGreaterEqual(sm["density"],0)

    def test_loader_extracts_square_block_before_final_demand(self):
        labels=[f"C{i}_S{i}" for i in range(5)]
        rows=[["node"]+labels+["FD_A","FD_B"]]
        for i,label in enumerate(labels):
            rows.append([label]+[str(i*5+j+1) for j in range(5)]+["10","20"])
        raw=pd.DataFrame(rows,dtype=str)
        Z,found,bounds,shape=_find_contiguous_numeric_block(raw,5)
        self.assertEqual(found,labels)
        self.assertEqual(Z.shape,(5,5))
        self.assertEqual(Z[4,4],25.0)
        self.assertEqual(bounds,(1,6,1,6))

    def test_synthetic_graphs_at_target_n(self):
        for family in ["ER","BA","WS"]:
            g,p=generate_skeleton(family,4050,8.0,1.0,0.1,42)
            self.assertEqual(g.number_of_nodes(),4050)

    def test_directed_reciprocity_calibration(self):
        import networkx as nx
        G=nx.gnp_random_graph(500,0.04,seed=12)
        D=_orient(G,0.4,np.random.default_rng(1))
        observed=nx.reciprocity(D)
        self.assertLess(abs(observed-0.4),0.08)

if __name__=="__main__":
    unittest.main()

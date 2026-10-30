"""ER/BA/WS synthetic network ensemble with paired UU/UD/WD tracks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from .config import FAMILIES
from .metrics import structural_analysis

def empirical_calibration(empirical_summaries):
    avg_degree=float(np.mean([x["mean_degree"] for x in empirical_summaries]))
    vals=[x.get("reciprocity",np.nan) for x in empirical_summaries]
    mean_r=float(np.nanmean(vals))
    reciprocity=mean_r if np.isfinite(mean_r) else 0.2
    return {"reference_mean_degree":avg_degree,"reference_reciprocity":float(np.clip(reciprocity,0,1))}

def _orient(G,target_reciprocity,rng):
    D=nx.DiGraph(); D.add_nodes_from(G.nodes)
    # If q is the probability of making an undirected skeleton dyad mutual,
    # expected arc reciprocity is 2q/(1+q). Invert to calibrate the requested r.
    r=float(np.clip(target_reciprocity,0.0,1.0))
    mutual_probability=r/(2.0-r) if r < 2.0 else 1.0
    for u,v in G.edges:
        if rng.random()<mutual_probability:
            D.add_edge(u,v); D.add_edge(v,u)
        elif rng.random()<0.5:
            D.add_edge(u,v)
        else:
            D.add_edge(v,u)
    return D

def generate_skeleton(family,n,reference_mean_degree,multiplier,ws_p,seed):
    if family=="ER":
        target=max(1.0,reference_mean_degree*multiplier)
        p=float(np.clip(target/max(1,n-1),0,1))
        return nx.gnp_random_graph(n,p,seed=seed),{"p":p,"target_mean_degree":target}
    if family=="BA":
        m=max(1,min(n-1,int(round(reference_mean_degree*multiplier/2))))
        return nx.barabasi_albert_graph(n,m,seed=seed),{"m":m,"target_mean_degree_approx":2*m}
    if family=="WS":
        k=max(2,min(n-1,int(round(reference_mean_degree*multiplier))))
        if k%2: k-=1
        k=max(2,k)
        return nx.watts_strogatz_graph(n,k,ws_p,seed=seed),{"k":k,"rewire_probability":ws_p}
    raise ValueError(f"Unknown family: {family}")

def _weights(empirical_shares,size,rng):
    vals=np.asarray(empirical_shares,dtype=float)
    vals=vals[np.isfinite(vals)&(vals>0)]
    return rng.choice(vals,size=size,replace=True) if len(vals) else np.full(size,0.01)

def run_synthetic_ensemble(n,empirical_mean_degree,empirical_reciprocity,empirical_share_samples,
                           realizations=20,seed=20261030,out_dir=None):
    rng=np.random.default_rng(seed); out=Path(out_dir) if out_dir else None
    if out: out.mkdir(parents=True,exist_ok=True)
    records=[]
    config_id=0
    for family in FAMILIES:
        for level in range(3):
            config_id+=1
            multiplier=(0.75,1.0,1.25)[level]
            ws_p=(0.01,0.10,0.50)[level]
            for realization in range(realizations):
                graph_seed=int(rng.integers(0,2**31-1))
                G,params=generate_skeleton(family,n,empirical_mean_degree,multiplier,ws_p,graph_seed)
                D=_orient(G,empirical_reciprocity,np.random.default_rng(graph_seed+17))
                A_uu=nx.to_scipy_sparse_array(G,nodelist=range(n),format="csr",dtype=np.uint8)
                A_ud=nx.to_scipy_sparse_array(D,nodelist=range(n),format="csr",dtype=np.uint8)
                row,col=A_ud.nonzero()
                weights=_weights(empirical_share_samples,len(row),np.random.default_rng(graph_seed+31))
                A_wd=sparse.csr_matrix((weights,(row,col)),shape=(n,n))
                for rep,A in {"UU":A_uu,"UD":A_ud,"WD":A_wd}.items():
                    sm,node=structural_analysis(A,[f"node_{i}" for i in range(n)],rep,seed=graph_seed,
                        path_sample_sources=min(64,n),betweenness_k=min(64,n))
                    records.append({"family":family,"configuration":config_id,"level":level+1,
                        "realization":realization,"representation":rep,"seed":graph_seed,**params,**sm})
    summaries=pd.DataFrame(records)
    if out:
        summaries.to_csv(out/"synthetic_realization_summaries.csv",index=False)
        grouped=summaries.groupby(["family","configuration","representation"],as_index=False).agg(
            realizations=("realization","count"),mean_degree=("mean_degree","mean"),
            sd_degree=("mean_degree","std"),density=("density","mean"),
            clustering=("average_clustering_undirected_projection","mean"),
            giant_component_fraction=("giant_component_fraction","mean"),
            reciprocity=("reciprocity","mean"),
            spectral_radius=("spectral_radius_adjacency_or_dependency","mean"))
        grouped.to_csv(out/"synthetic_configuration_summaries.csv",index=False)
        (out/"synthetic_experiment_design.json").write_text(json.dumps({
            "families":list(FAMILIES),"configurations_per_family":3,
            "realizations_per_configuration":realizations,"skeletons":9*realizations,
            "paired_representation_instances":27*realizations,"node_count":n,
            "empirical_mean_degree_reference":empirical_mean_degree,
            "empirical_reciprocity_reference":empirical_reciprocity,
            "weight_provenance":"Synthetic resampling from empirical dependency-share distribution; not observed transactions."
        },indent=2))
    return summaries

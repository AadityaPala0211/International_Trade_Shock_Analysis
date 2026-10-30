"""Paired ER/BA/WS synthetic ensembles calibrated to empirical topology and weights."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from .config import FAMILIES, SyntheticConfig
from .metrics import structural_analysis


def empirical_calibration(empirical_summaries):
    uu=[x for x in empirical_summaries if x.get("representation")=="UU"]
    ud=[x for x in empirical_summaries if x.get("representation")=="UD"]
    avg_degree=float(np.mean([x["mean_degree"] for x in (uu or empirical_summaries)]))
    vals=[float(x.get("reciprocity",np.nan)) for x in ud]
    finite_reciprocity=[value for value in vals if np.isfinite(value)]
    mean_r=float(np.mean(finite_reciprocity)) if finite_reciprocity else 0.2
    reciprocity=float(np.clip(mean_r,0,1))
    return {"reference_mean_degree":avg_degree,"reference_reciprocity":reciprocity}


def _orientation_from_undirected(G, target_reciprocity, rng):
    D=nx.DiGraph(); D.add_nodes_from(G.nodes)
    # q mutual dyads imply expected arc reciprocity 2q/(1+q); invert.
    r=float(np.clip(target_reciprocity,0.0,1.0))
    mutual_probability=r/(2.0-r) if r < 2.0 else 1.0
    for u,v in G.edges:
        if rng.random() < mutual_probability:
            D.add_edge(u,v); D.add_edge(v,u)
        elif rng.random()<0.5:
            D.add_edge(u,v)
        else:
            D.add_edge(v,u)
    return D


def _sample_weights(empirical_shares, size, rng):
    vals=np.asarray(empirical_shares,dtype=float)
    vals=vals[np.isfinite(vals)&(vals>0)]
    if len(vals)==0: return np.full(size,0.01)
    return rng.choice(vals,size=size,replace=True)


def _calibrate_weighted_adjacency(A_ud, empirical_shares, dependency_total_samples, seed):
    """Sample edge shares then calibrate each buyer's incoming sum.

    Per-edge values are sampled from observed positive dependency shares. For
    each synthetic buyer, the total incoming dependency share is separately
    sampled from the empirical distribution of off-diagonal buyer dependency
    totals. These are synthetic dependency-like weights, not monetary flows.
    """
    row,col=A_ud.nonzero()
    rng=np.random.default_rng(seed)
    values=_sample_weights(empirical_shares,len(row),rng)
    targets=np.asarray(dependency_total_samples,dtype=float)
    targets=targets[np.isfinite(targets)]
    if len(targets)==0: targets=np.array([0.5])
    target_by_buyer=rng.choice(targets,size=A_ud.shape[1],replace=True)
    order=np.argsort(col,kind="stable")
    row=row[order]; col=col[order]; values=values[order]
    counts=np.bincount(col,minlength=A_ud.shape[1])
    starts=np.cumsum(counts)-counts
    for buyer in np.flatnonzero(counts):
        start=int(starts[buyer]); stop=start+int(counts[buyer])
        total=values[start:stop].sum()
        if total>0:
            values[start:stop] *= max(0.0,float(target_by_buyer[buyer]))/total
    return sparse.csr_matrix((values,(row,col)),shape=A_ud.shape),target_by_buyer


def generate_skeleton(family, n, reference_mean_degree, multiplier, ws_p, seed):
    if family=="ER":
        target=max(1.0,reference_mean_degree*multiplier)
        p=float(np.clip(target/max(1,n-1),0,1))
        return nx.gnp_random_graph(n,p,seed=seed),{"p":p,"target_mean_degree":target}
    if family=="BA":
        m=max(1,min(n-1,int(round(reference_mean_degree*multiplier/2))))
        return nx.barabasi_albert_graph(n,m,seed=seed),{"m":m,"target_mean_degree_approx":2*m}
    if family=="WS":
        # WS sensitivity is rewiring probability; keep neighbourhood degree
        # fixed at the nearest even empirical mean degree across configurations.
        k=max(2,min(n-1,int(round(reference_mean_degree))))
        if k%2: k-=1
        k=max(2,k)
        return nx.watts_strogatz_graph(n,k,ws_p,seed=seed),{"k":k,"rewire_probability":ws_p}
    raise ValueError(f"Unknown family: {family}")


def _adjacency(G, n):
    return nx.to_scipy_sparse_array(G,nodelist=range(n),format="csr",dtype=np.uint8)


def run_synthetic_ensemble(n, empirical_mean_degree, empirical_reciprocity,
                           empirical_share_samples, realizations=20, seed=20261030,
                           out_dir=None, empirical_dependency_total_samples=None,
                           config: SyntheticConfig | None=None):
    """Generate 9 configurations x realizations skeletons and paired UU/UD/WD tracks."""
    cfg=config or SyntheticConfig(realizations=realizations,seed=seed)
    rng=np.random.default_rng(seed)
    out=Path(out_dir) if out_dir else None
    if out: out.mkdir(parents=True,exist_ok=True)
    all_summaries=[]; config_id=0
    for family in FAMILIES:
        for level in range(3):
            config_id+=1
            multiplier=(cfg.mean_degree_multipliers[level] if family=="ER"
                        else cfg.ba_multipliers[level] if family=="BA" else 1.0)
            ws_p=cfg.ws_rewire_probabilities[level]
            for rep in range(realizations):
                graph_seed=int(rng.integers(0,2**31-1))
                G,params=generate_skeleton(family,n,empirical_mean_degree,multiplier,ws_p,graph_seed)
                # One skeleton and one directed orientation are reused for paired tracks.
                D=_orientation_from_undirected(G,empirical_reciprocity,np.random.default_rng(graph_seed+17))
                A_uu=_adjacency(G,n); A_ud=_adjacency(D,n)
                total_samples=(empirical_dependency_total_samples if empirical_dependency_total_samples is not None
                               else np.asarray([0.5]))
                A_wd,target_totals=_calibrate_weighted_adjacency(
                    A_ud,empirical_share_samples,total_samples,graph_seed+31)
                tracks={"UU":A_uu,"UD":A_ud,"WD":A_wd}
                for rep_name,A in tracks.items():
                    sm,_=structural_analysis(A,[f"node_{i}" for i in range(n)],rep_name,
                        seed=graph_seed,path_sample_sources=min(64,n),betweenness_k=min(64,n),
                        include_node_metrics=False)
                    row_meta={"family":family,"configuration":config_id,"level":level+1,
                        "realization":rep,"representation":rep_name,"seed":graph_seed,
                        **params,**sm}
                    all_summaries.append(row_meta)
    summaries=pd.DataFrame(all_summaries)
    if out:
        summaries.to_csv(out/"synthetic_realization_summaries.csv",index=False)
        by=summaries.groupby(["family","configuration","representation"],as_index=False).agg(
            realizations=("realization","count"),mean_degree=("mean_degree","mean"),
            sd_degree=("mean_degree","std"),density=("density","mean"),
            clustering=("average_clustering_undirected_projection","mean"),
            giant_component_fraction=("giant_component_fraction","mean"),
            reciprocity=("reciprocity","mean"),
            spectral_radius=("spectral_radius_adjacency_or_dependency","mean"))
        by.to_csv(out/"synthetic_configuration_summaries.csv",index=False)
        (out/"synthetic_experiment_design.json").write_text(json.dumps({
            "families":list(FAMILIES),"configurations_per_family":3,
            "realizations_per_configuration":realizations,
            "skeletons":9*realizations,"paired_representation_instances":27*realizations,
            "node_count":n,"empirical_mean_degree_reference":empirical_mean_degree,
            "empirical_reciprocity_reference":empirical_reciprocity,
            "er_mean_degree_multipliers":list(cfg.mean_degree_multipliers),
            "ba_attachment_multipliers":list(cfg.ba_multipliers),
            "ws_rewire_probabilities":list(cfg.ws_rewire_probabilities),
            "ws_neighbourhood_degree":"fixed at nearest even empirical mean degree",
            "weight_provenance":"Synthetic edge dependency shares resampled from empirical threshold-retained positive shares and calibrated so each buyer's incoming sum follows the empirical threshold-retained dependency-total distribution; not observed transactions."
        },indent=2))
    return summaries

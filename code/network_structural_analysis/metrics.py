"""Network-level and node-level structural metrics."""
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigs

def _safe_spectral_radius(A):
    A=sparse.csr_matrix(A,dtype=float)
    if A.shape[0] <= 2:
        vals=np.linalg.eigvals(A.toarray())
        return float(np.max(np.abs(vals))) if len(vals) else 0.0
    try:
        return float(abs(eigs(A,k=1,which="LM",return_eigenvectors=False,tol=1e-5,maxiter=5000)[0]))
    except Exception:
        x=np.ones(A.shape[0]); x/=np.linalg.norm(x)
        for _ in range(500):
            y=A@x; norm=np.linalg.norm(y)
            if norm==0: return 0.0
            y/=norm
            if np.linalg.norm(y-x)<1e-9: x=y; break
            x=y
        return float(np.linalg.norm(A@x))

def _degree_cv(values):
    v=np.asarray(values,dtype=float); m=v.mean()
    return float(v.std()/m) if m else 0.0

def graph_from_sparse(A,directed,weighted=False):
    A=sparse.coo_matrix(A)
    G=nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    if weighted:
        G.add_weighted_edges_from((int(i),int(j),float(w)) for i,j,w in zip(A.row,A.col,A.data) if i!=j and w>0)
    else:
        G.add_edges_from((int(i),int(j)) for i,j in zip(A.row,A.col) if i!=j)
    return G

def structural_analysis(A,labels,representation,seed=20261030,path_sample_sources=256,betweenness_k=256):
    n=len(labels); A=sparse.csr_matrix(A)
    directed=representation in ("UD","WD"); weighted=representation=="WD"
    if A.shape!=(n,n): raise ValueError("Matrix shape and node labels disagree")
    G=graph_from_sparse(A,directed,weighted)
    m=G.number_of_edges()
    if directed:
        indeg=np.asarray(A.getnnz(axis=0)).ravel()
        outdeg=np.asarray(A.getnnz(axis=1)).ravel()
        in_strength=np.asarray(A.sum(axis=0)).ravel()
        out_strength=np.asarray(A.sum(axis=1)).ravel()
        degree=np.asarray((A+A.T).astype(bool).sum(axis=1)).ravel()
        reciprocity=nx.reciprocity(G) if m else 0.0
        U=G.to_undirected()
        clustering=nx.average_clustering(U)
        kcore=nx.core_number(U) if U.number_of_edges() else {i:0 for i in U.nodes}
        pagerank=nx.pagerank(G,weight="weight" if weighted else None,max_iter=500,tol=1e-9)
    else:
        deg=np.asarray(A.getnnz(axis=1)).ravel()
        indeg=outdeg=deg
        in_strength=out_strength=np.asarray(A.sum(axis=1)).ravel()
        degree=deg; reciprocity=float("nan"); U=G
        clustering=nx.average_clustering(G)
        kcore=nx.core_number(G) if G.number_of_edges() else {i:0 for i in G.nodes}
        pagerank=nx.pagerank(G,weight=None,max_iter=500,tol=1e-9)
    components=list(nx.connected_components(U))
    giant=max((len(c) for c in components),default=0)
    rng=np.random.default_rng(seed); nodes=np.arange(n)
    sample=nodes if n<=path_sample_sources else np.sort(rng.choice(nodes,path_sample_sources,replace=False))
    path_lengths=[]
    for source in sample:
        path_lengths.extend(v for k,v in nx.single_source_shortest_path_length(U,int(source)).items() if k!=source)
    avg_path=float(np.mean(path_lengths)) if path_lengths else float("nan")
    path_coverage=len(path_lengths)/(max(1,len(sample))*max(1,n-1))
    k=min(betweenness_k,n)
    bet=nx.betweenness_centrality(G,k=k,normalized=True,weight=None,seed=seed) if n>500 else nx.betweenness_centrality(G,normalized=True,weight=None)
    local=nx.clustering(U)
    node_df=pd.DataFrame({
      "node_id":np.arange(n),"node":labels,
      "in_degree":np.asarray(A.getnnz(axis=0)).ravel() if directed else degree,
      "out_degree":np.asarray(A.getnnz(axis=1)).ravel() if directed else degree,
      "degree_undirected_projection":degree,"in_strength":in_strength,"out_strength":out_strength,
      "pagerank":pd.Series(pagerank),"betweenness":pd.Series(bet),
      "k_core":pd.Series(kcore),"clustering":pd.Series(local)})
    summary={"representation":representation,"nodes":n,"edges":m,"density":nx.density(G),
      "mean_degree":float(np.mean(degree)),"degree_cv":_degree_cv(degree),
      "mean_in_degree":float(np.mean(indeg)),"mean_out_degree":float(np.mean(outdeg)),
      "mean_in_strength":float(np.mean(in_strength)),"mean_out_strength":float(np.mean(out_strength)),
      "total_weight":float(A.sum()) if weighted else float(m),
      "average_clustering_undirected_projection":float(clustering),
      "weak_component_count":len(components),"giant_component_nodes":int(giant),
      "giant_component_fraction":float(giant/n if n else 0),
      "average_shortest_path_sampled":avg_path,"path_sample_sources":len(sample),
      "reachable_pair_sample_coverage":float(path_coverage),
      "betweenness_method":"sampled_sources" if n>500 else "exact","betweenness_k":int(k),
      "reciprocity":float(reciprocity),"spectral_radius_adjacency_or_dependency":_safe_spectral_radius(A),
      "spectral_radius_definition":"WD dependency-share matrix; UU/UD binary adjacency matrix",
      "directed":directed,"weighted":weighted}
    return summary,node_df

def country_sector_summary(node_metrics):
    out=node_metrics.copy()
    split=out["node"].astype(str).str.replace(" ","_",regex=False).str.split("_",n=1,expand=True)
    out["country_code"]=split[0]
    out["industry_code"]=split[1] if split.shape[1]>1 else "UNKNOWN"
    cols=[c for c in out.select_dtypes(include=[np.number]).columns if c!="node_id"]
    return (out.groupby("country_code")[cols].mean(numeric_only=True).reset_index(),
            out.groupby("industry_code")[cols].mean(numeric_only=True).reset_index())

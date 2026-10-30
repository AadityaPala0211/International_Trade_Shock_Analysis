"""Sparse-first graph metrics designed for the full 4,050-node ICIO networks."""
from __future__ import annotations
from collections import deque
import heapq
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigs


def _safe_spectral_radius(A):
    A=sparse.csr_matrix(A,dtype=float)
    if A.shape[0] <= 2:
        vals=np.linalg.eigvals(A.toarray())
        return float(np.max(np.abs(vals))) if len(vals) else 0.0
    try:
        val=eigs(A,k=1,which="LM",return_eigenvectors=False,tol=1e-5,maxiter=5000)[0]
        return float(abs(val))
    except Exception:
        x=np.ones(A.shape[0]); x/=np.linalg.norm(x)
        for _ in range(1000):
            y=A@x; norm=np.linalg.norm(y)
            if norm==0: return 0.0
            y/=norm
            if min(np.linalg.norm(y-x),np.linalg.norm(y+x))<1e-9:
                x=y; break
            x=y
        return float(np.linalg.norm(A@x))


def _degree_cv(values):
    v=np.asarray(values,dtype=float); mean=v.mean()
    return float(v.std()/mean) if mean else 0.0


def _pagerank_sparse(A, alpha=0.85, tol=1e-10, max_iter=1000):
    """PageRank via sparse power iteration; A is source-row, destination-column."""
    A=sparse.csr_matrix(A,dtype=float); n=A.shape[0]
    if n==0: return np.array([],dtype=float)
    out=np.asarray(A.sum(axis=1)).ravel()
    inv=np.divide(1.0,out,out=np.zeros_like(out),where=out>0)
    transition=sparse.diags(inv,format="csr")@A
    rank=np.full(n,1.0/n); dangling=(out==0)
    for _ in range(max_iter):
        new=alpha*(transition.T@rank + rank[dangling].sum()/n)+(1-alpha)/n
        if np.abs(new-rank).sum()<tol:
            rank=new; break
        rank=new
    total=rank.sum()
    return rank/total if total else np.full(n,1.0/n)


def _core_numbers_sparse(B):
    """Matula-Beck style core decomposition on an undirected binary CSR matrix."""
    B=sparse.csr_matrix(B,dtype=np.uint8)
    degree=np.asarray(B.getnnz(axis=1)).ravel().astype(np.int64)
    core=degree.copy()
    heap=[(int(degree[v]),int(v)) for v in range(B.shape[0])]
    heapq.heapify(heap); indptr=B.indptr; indices=B.indices
    while heap:
        d,v=heapq.heappop(heap)
        if d!=core[v]: continue
        for u in indices[indptr[v]:indptr[v+1]]:
            if core[u]>d:
                core[u]-=1
                heapq.heappush(heap,(int(core[u]),int(u)))
    return core


def _local_clustering_sparse(B):
    """Local clustering on undirected binary B using common-neighbour counts."""
    B=sparse.csr_matrix(B,dtype=np.int32)
    degree=np.asarray(B.getnnz(axis=1)).ravel().astype(np.float64)
    if B.nnz==0: return np.zeros(B.shape[0],dtype=float)
    two_step=B@B
    common=np.asarray(B.multiply(two_step).sum(axis=1)).ravel().astype(np.float64)
    denominator=degree*(degree-1.0)
    return np.divide(common,denominator,out=np.zeros_like(common),where=denominator>0)


def _sampled_betweenness(B, k, seed, directed=True):
    """Unweighted Brandes betweenness approximation using k sampled sources."""
    B=sparse.csr_matrix(B,dtype=np.uint8); n=B.shape[0]
    if n<3 or B.nnz==0 or k<=0: return np.zeros(n,dtype=float)
    k=min(int(k),n)
    sources=np.arange(n) if k==n else np.sort(np.random.default_rng(seed).choice(n,k,replace=False))
    indptr,indices=B.indptr,B.indices
    centrality=np.zeros(n,dtype=np.float64)
    for source in sources:
        predecessors=[[] for _ in range(n)]
        sigma=np.zeros(n,dtype=np.float64); sigma[source]=1.0
        distance=np.full(n,-1,dtype=np.int32); distance[source]=0
        queue=deque([int(source)]); stack=[]
        while queue:
            v=queue.popleft(); stack.append(v); next_distance=distance[v]+1
            for w in indices[indptr[v]:indptr[v+1]]:
                if distance[w]<0:
                    distance[w]=next_distance; queue.append(int(w))
                if distance[w]==next_distance:
                    sigma[w]+=sigma[v]; predecessors[w].append(v)
        dependency=np.zeros(n,dtype=np.float64)
        while stack:
            w=stack.pop()
            if sigma[w]>0:
                factor=(1.0+dependency[w])/sigma[w]
                for v in predecessors[w]:
                    dependency[v]+=sigma[v]*factor
            if w!=source: centrality[w]+=dependency[w]
    if n>2:
        # Brandes accumulates both endpoint orders for undirected graphs;
        # divide by two before applying the directed/undirected shared scale.
        if not directed:
            centrality *= 0.5
        centrality *= (n/float(k))/((n-1)*(n-2))
    return centrality


def structural_analysis(A, labels, representation: str, seed: int=20261030,
                        path_sample_sources: int=256, betweenness_k: int=64,
                        include_node_metrics: bool=True):
    """Compute network and node metrics without materialising Python edge objects.

    For UD/WD, path distances and betweenness retain direction. Weak components,
    k-core and clustering are computed on the undirected projection. WD PageRank
    uses dependency shares as edge weights; all path metrics use binary edges.
    """
    n=len(labels); A=sparse.csr_matrix(A)
    directed=representation in ("UD","WD"); weighted=representation=="WD"
    if A.shape!=(n,n): raise ValueError("Matrix shape and node labels disagree")
    A.eliminate_zeros()
    A_bin=(A!=0).astype(np.uint8).tocsr()
    if directed:
        indeg=np.asarray(A_bin.getnnz(axis=0)).ravel()
        outdeg=np.asarray(A_bin.getnnz(axis=1)).ravel()
        in_strength=np.asarray(A.sum(axis=0)).ravel()
        out_strength=np.asarray(A.sum(axis=1)).ravel()
        degree=np.asarray(A_bin.maximum(A_bin.T).getnnz(axis=1)).ravel()
        mutual=A_bin.multiply(A_bin.T).nnz
        reciprocity=float(mutual/A_bin.nnz) if A_bin.nnz else 0.0
        strong_count,strong_labels=connected_components(A_bin,directed=True,connection="strong",return_labels=True)
        strong_sizes=np.bincount(strong_labels) if n else np.array([],dtype=int)
        giant_strong=int(strong_sizes.max()) if len(strong_sizes) else 0
        B=A_bin.maximum(A_bin.T).tocsr()
        path_graph=A_bin; path_is_directed=True
        pagerank=_pagerank_sparse(A if weighted else A_bin) if include_node_metrics else None
        edge_count=int(A_bin.nnz); density=edge_count/(n*(n-1)) if n>1 else 0.0
        path_definition="directed_reachable_pairs"
    else:
        B=A_bin; degree=np.asarray(B.getnnz(axis=1)).ravel()
        indeg=outdeg=degree
        in_strength=out_strength=np.asarray(A.sum(axis=1)).ravel()
        reciprocity=float("nan")
        strong_count=float("nan"); giant_strong=float("nan")
        path_graph=B; path_is_directed=False
        pagerank=_pagerank_sparse(A_bin) if include_node_metrics else None
        edge_count=int(B.nnz//2); density=edge_count/(n*(n-1)/2) if n>1 else 0.0
        path_definition="undirected_connected_pairs"

    component_count,component_labels=connected_components(B,directed=False,return_labels=True)
    component_sizes=np.bincount(component_labels) if n else np.array([],dtype=int)
    giant=int(component_sizes.max()) if len(component_sizes) else 0
    kcore=_core_numbers_sparse(B)
    local_clustering=_local_clustering_sparse(B)
    average_clustering=float(local_clustering.mean()) if n else 0.0

    rng=np.random.default_rng(seed+1009)
    path_k=min(path_sample_sources,n)
    path_sources=np.arange(n) if path_k==n else np.sort(rng.choice(n,path_k,replace=False))
    if path_k and n:
        distances=shortest_path(path_graph,directed=path_is_directed,unweighted=True,
                                indices=path_sources,method="D")
        reachable=(distances>0)&np.isfinite(distances)
        path_lengths=distances[reachable]
        avg_path=float(path_lengths.mean()) if path_lengths.size else float("nan")
        path_coverage=float(path_lengths.size/(path_k*max(1,n-1)))
    else:
        avg_path=float("nan"); path_coverage=0.0

    if include_node_metrics:
        betweenness=_sampled_betweenness(A_bin if directed else B,min(betweenness_k,n),seed+2027,directed=directed)
        node_df=pd.DataFrame({
            "node_id":np.arange(n),"node":labels,
            "in_degree":indeg,"out_degree":outdeg,
            "degree_undirected_projection":degree,
            "in_strength":in_strength,"out_strength":out_strength,
            "pagerank":pagerank,"betweenness":betweenness,
            "k_core":kcore,"clustering":local_clustering})
    else:
        node_df=None

    summary={
        "representation":representation,"nodes":n,"edges":edge_count,"density":float(density),
        "mean_degree":float(np.mean(degree)) if n else 0.0,
        "mean_degree_undirected_projection":float(np.mean(degree)) if n else 0.0,
        "degree_cv":_degree_cv(degree),"in_degree_cv":_degree_cv(indeg),
        "out_degree_cv":_degree_cv(outdeg),
        "mean_in_degree":float(np.mean(indeg)) if n else 0.0,
        "mean_out_degree":float(np.mean(outdeg)) if n else 0.0,
        "mean_in_strength":float(np.mean(in_strength)) if n else 0.0,
        "mean_out_strength":float(np.mean(out_strength)) if n else 0.0,
        "mean_k_core":float(np.mean(kcore)) if n else 0.0,
        "max_k_core":int(np.max(kcore)) if n else 0,
        "total_weight":float(A.sum()) if weighted else float(edge_count),
        "average_clustering_undirected_projection":average_clustering,
        "weak_component_count":int(component_count),"giant_component_nodes":giant,
        "giant_component_fraction":float(giant/n) if n else 0.0,
        "strong_component_count":strong_count,
        "giant_strong_component_fraction":float(giant_strong/n) if directed and n else (0.0 if directed else float("nan")),
        "average_shortest_path_sampled":avg_path,"path_sample_sources":int(path_k),
        "reachable_pair_sample_coverage":path_coverage,
        "path_metric_definition":path_definition,
        "betweenness_method":"sampled_sources" if n>betweenness_k else "exact_sources",
        "betweenness_k":int(min(betweenness_k,n)) if include_node_metrics else 0,
        "reciprocity":reciprocity,
        "spectral_radius_adjacency_or_dependency":_safe_spectral_radius(A),
        "spectral_radius_definition":"WD dependency-share matrix; UU/UD binary adjacency matrix",
        "directed":directed,"weighted":weighted}
    return summary,node_df


def country_sector_summary(node_metrics: pd.DataFrame):
    """Aggregate node metrics by country and industry tokens in ICIO labels."""
    out=node_metrics.copy()
    split=out["node"].astype(str).str.replace(" ","_",regex=False).str.split("_",n=1,expand=True)
    out["country_code"]=split[0]
    out["industry_code"]=split[1] if split.shape[1]>1 else "UNKNOWN"
    cols=[c for c in out.select_dtypes(include=[np.number]).columns if c!="node_id"]
    return (out.groupby("country_code")[cols].mean(numeric_only=True).reset_index(),
            out.groupby("industry_code")[cols].mean(numeric_only=True).reset_index())

"""Node-level shock attribution and centrality-targeted SIS experiments.

Separate from network_structural_analysis: this module consumes existing sparse
matrices and never mutates structural files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs


def _csr(matrix):
    B = sparse.csr_matrix(matrix, dtype=float)
    B.sum_duplicates()
    B.eliminate_zeros()
    if B.shape[0] != B.shape[1] or B.shape[0] == 0:
        raise ValueError("B must be a non-empty square matrix.")
    if not np.isfinite(B.data).all() or np.any(B.data < 0):
        raise ValueError("B must contain finite, nonnegative weights.")
    return B


def spectral_radius(B):
    B = _csr(B)
    n = B.shape[0]
    if n <= 2:
        return float(np.max(np.abs(np.linalg.eigvals(B.toarray()))))
    try:
        return float(abs(eigs(B, k=1, which="LM", return_eigenvectors=False,
                              tol=1e-7, maxiter=5000)[0]))
    except Exception:
        x = np.full(n, 1.0 / np.sqrt(n))
        for _ in range(3000):
            y = B @ x
            norm = np.linalg.norm(y)
            if norm == 0:
                return 0.0
            y /= norm
            if min(np.linalg.norm(y-x), np.linalg.norm(y+x)) < 1e-10:
                x = y
                break
            x = y
        return float(np.linalg.norm(B @ x))


def simulate_model_a(B, initial_state, beta, rho, max_steps=300,
                     tolerance=1e-8, extinction_threshold=1e-8):
    """Deterministic Model A; B[i,j] means supplier i -> buyer j."""
    B = _csr(B)
    beta, rho = float(beta), float(rho)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1].")
    if int(max_steps) < 1 or float(tolerance) <= 0 or extinction_threshold < 0:
        raise ValueError("Invalid max_steps, tolerance, or extinction_threshold.")
    d = np.asarray(initial_state, dtype=float).copy()
    if d.shape != (B.shape[0],) or not np.isfinite(d).all():
        raise ValueError("initial_state must be finite and match the network.")
    if np.any((d < 0) | (d > 1)):
        raise ValueError("initial_state must lie in [0,1].")
    prevalence = [float(d.mean())]
    exposure_mass = [float((B.T @ d).sum())]
    converged = False
    extinct = prevalence[0] <= extinction_threshold
    for step in range(1, int(max_steps)+1):
        exposure = np.asarray(B.T @ d).ravel()
        next_d = np.clip((1-rho)*d + (1-d)*beta*exposure, 0.0, 1.0)
        delta = float(np.max(np.abs(next_d-d)))
        d = next_d
        prevalence.append(float(d.mean()))
        exposure_mass.append(float((B.T @ d).sum()))
        if prevalence[-1] <= extinction_threshold:
            extinct = True
            break
        if delta <= tolerance:
            converged = True
            break
    p = np.asarray(prevalence)
    peak = int(np.argmax(p))
    return {
        "prevalence": p,
        "exposure_mass": np.asarray(exposure_mass),
        "peak_prevalence": float(p[peak]),
        "peak_step": peak,
        "final_prevalence": float(p[-1]),
        "cumulative_prevalence": float(p[1:].sum()),
        "extinction_time": int(len(p)-1) if extinct else np.nan,
        "steps": int(len(p)-1),
        "converged": bool(converged),
        "extinct": bool(extinct),
    }


def _stable_rank(values):
    return np.lexsort((np.arange(len(values)), -np.asarray(values, dtype=float)))


def calculate_targeting_metrics(B, labels: Iterable[str] | None = None):
    """Direction-aware metrics: supplier reach and buyer dependence."""
    B = _csr(B)
    n = B.shape[0]
    A = (B != 0).astype(np.int8).tocsr()
    out_degree = np.asarray(A.sum(axis=1)).ravel().astype(float)
    in_degree = np.asarray(A.sum(axis=0)).ravel().astype(float)
    out_strength = np.asarray(B.sum(axis=1)).ravel()
    in_strength = np.asarray(B.sum(axis=0)).ravel()
    undirected_degree = np.asarray(A.maximum(A.T).sum(axis=1)).ravel().astype(float)

    # PageRank follows supplier -> buyer orientation.
    P = B.T.tocsr()
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum), where=rowsum > 0)
    transition = sparse.diags(inv) @ P
    rank = np.full(n, 1.0/n)
    dangling = rowsum == 0
    for _ in range(1000):
        updated = 0.85*(transition.T @ rank + rank[dangling].sum()/n) + 0.15/n
        if np.abs(updated-rank).sum() < 1e-12:
            rank = updated
            break
        rank = updated
    if rank.sum():
        rank /= rank.sum()

    try:
        if n <= 2:
            vals, vecs = np.linalg.eig(B.T.toarray())
            vec = np.real(vecs[:, int(np.argmax(np.abs(vals)))])
        else:
            _, vecs = eigs(B.T, k=1, which="LM", maxiter=5000, tol=1e-7)
            vec = np.real(vecs[:, 0])
        eigenvector = np.abs(vec)
        if eigenvector.sum():
            eigenvector /= eigenvector.sum()
    except Exception:
        eigenvector = np.zeros(n)
    labels = [str(i) for i in range(n)] if labels is None else list(map(str, labels))
    if len(labels) != n:
        raise ValueError("labels length does not match B.")
    return pd.DataFrame({
        "node_id": np.arange(n, dtype=int), "node": labels,
        "supplier_out_degree": out_degree, "buyer_in_degree": in_degree,
        "undirected_projection_degree": undirected_degree,
        "supplier_out_strength": out_strength, "buyer_in_strength": in_strength,
        "pagerank_supplier_to_buyer": rank,
        "eigenvector_supplier_to_buyer": eigenvector,
    })


def run_single_node_census(B, beta, rho, labels=None, node_metrics=None,
                           max_steps=300, tolerance=1e-8,
                           initial_infection=1.0, store_trajectories=False):
    """Run one deterministic SIS trajectory for every possible single-node source."""
    B = _csr(B)
    n = B.shape[0]
    if not 0 < float(initial_infection) <= 1:
        raise ValueError("initial_infection must be in (0,1].")
    metrics = calculate_targeting_metrics(B, labels) if node_metrics is None else node_metrics.copy()
    if len(metrics) != n or "node_id" not in metrics:
        raise ValueError("node_metrics must have one row per node and node_id.")
    metrics = metrics.sort_values("node_id").reset_index(drop=True)
    lam = spectral_radius(B)
    beta_c = float(rho/lam) if lam > 0 else np.inf
    rows, trajectories = [], []
    for source in range(n):
        state = np.zeros(n)
        state[source] = float(initial_infection)
        result = simulate_model_a(B, state, beta, rho, max_steps, tolerance)
        row = {
            "source_node_id": source, "source_node": str(metrics.loc[source, "node"]),
            "beta": float(beta), "rho": float(rho),
            "beta_critical": beta_c,
            "source_out_degree": float(metrics.loc[source, "supplier_out_degree"]),
            "source_in_degree": float(metrics.loc[source, "buyer_in_degree"]),
            "source_out_strength": float(metrics.loc[source, "supplier_out_strength"]),
            "source_in_strength": float(metrics.loc[source, "buyer_in_strength"]),
            "source_pagerank": float(metrics.loc[source, "pagerank_supplier_to_buyer"]),
            "source_eigenvector": float(metrics.loc[source, "eigenvector_supplier_to_buyer"]),
            "peak_prevalence": result["peak_prevalence"],
            "peak_step": result["peak_step"],
            "final_prevalence": result["final_prevalence"],
            "cumulative_prevalence": result["cumulative_prevalence"],
            "extinction_time": result["extinction_time"],
            "steps": result["steps"], "converged": result["converged"],
            "extinct": result["extinct"],
        }
        rows.append(row)
        if store_trajectories:
            trajectories.extend({
                "source_node_id": source, "source_node": row["source_node"],
                "step": t, "prevalence": float(p),
                "exposure_mass": float(result["exposure_mass"][t]),
            } for t, p in enumerate(result["prevalence"]))
    census = pd.DataFrame(rows)
    census["impact_rank_peak"] = census["peak_prevalence"].rank(method="min", ascending=False).astype(int)
    census["impact_rank_cumulative"] = census["cumulative_prevalence"].rank(method="min", ascending=False).astype(int)
    census["impact_rank_final"] = census["final_prevalence"].rank(method="min", ascending=False).astype(int)
    census = census.sort_values(["impact_rank_cumulative", "source_node_id"]).reset_index(drop=True)
    return census, (pd.DataFrame(trajectories) if store_trajectories else None)


DEFAULT_CENTRALITIES = (
    "supplier_out_degree", "buyer_in_degree", "undirected_projection_degree",
    "supplier_out_strength", "buyer_in_strength",
    "pagerank_supplier_to_buyer", "eigenvector_supplier_to_buyer",
)


def run_targeted_cohort_experiment(B, beta, rho, node_metrics, fraction=0.01,
                                   repeats=20, seed=20261030, max_steps=300,
                                   tolerance=1e-8, centralities=DEFAULT_CENTRALITIES):
    """Top/bottom centrality cohorts compared with matched-size random cohorts."""
    B = _csr(B)
    n = B.shape[0]
    k = max(1, int(np.ceil(float(fraction)*n)))
    if k > n or int(repeats) < 1:
        raise ValueError("Invalid fraction or repeats.")
    metrics = node_metrics.sort_values("node_id").reset_index(drop=True)
    if len(metrics) != n:
        raise ValueError("node_metrics must have one row per network node.")
    rng = np.random.default_rng(seed)
    random_sets = [np.sort(rng.choice(n, k, replace=False)) for _ in range(int(repeats))]
    rows = []

    def evaluate(cohort, centrality, repeat_id, node_ids):
        state = np.zeros(n)
        state[np.asarray(node_ids, dtype=int)] = 1.0
        outcome = simulate_model_a(B, state, beta, rho, max_steps, tolerance)
        rows.append({
            "cohort": cohort, "centrality": centrality, "repeat": repeat_id,
            "seed_count": k, "seed_nodes": "|".join(map(str, node_ids)),
            "peak_prevalence": outcome["peak_prevalence"],
            "peak_step": outcome["peak_step"],
            "final_prevalence": outcome["final_prevalence"],
            "cumulative_prevalence": outcome["cumulative_prevalence"],
            "extinction_time": outcome["extinction_time"],
            "steps": outcome["steps"], "converged": outcome["converged"],
            "extinct": outcome["extinct"],
        })

    for centrality in centralities:
        if centrality not in metrics:
            raise ValueError(f"Missing centrality column: {centrality}")
        order = _stable_rank(metrics[centrality].to_numpy(dtype=float))
        evaluate("top", centrality, 0, order[:k])
        evaluate("bottom", centrality, 0, order[-k:][::-1])
        for repeat_id, sample in enumerate(random_sets, start=1):
            evaluate("random", centrality, repeat_id, sample)
    return pd.DataFrame(rows)


def summarize_targeted_cohorts(experiment):
    metrics = ("peak_prevalence", "final_prevalence", "cumulative_prevalence")
    rows = []
    for (centrality, cohort), group in experiment.groupby(["centrality", "cohort"], sort=True):
        row = {"centrality": centrality, "cohort": cohort, "n_runs": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(np.nanmin(values))
            row[f"{metric}_max"] = float(np.nanmax(values))
        rows.append(row)
    summary = pd.DataFrame(rows)
    random = summary[summary.cohort == "random"].set_index("centrality")
    for metric in metrics:
        summary[f"{metric}_difference_vs_random"] = np.nan
        for idx in summary.index[summary.cohort.isin(["top", "bottom"])]:
            cent = summary.loc[idx, "centrality"]
            summary.loc[idx, f"{metric}_difference_vs_random"] = (
                summary.loc[idx, f"{metric}_mean"] - random.loc[cent, f"{metric}_mean"]
            )
    return summary


def write_experiment_outputs(out_dir, census=None, trajectories=None, targeted=None,
                             targeted_summary=None, metadata=None):
    """Persist additive outputs below a dedicated dynamics output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if census is not None:
        census.to_csv(out/"single_node_shock_census.csv.gz", index=False, compression="gzip")
    if trajectories is not None:
        trajectories.to_csv(out/"single_node_shock_trajectories.csv.gz", index=False, compression="gzip")
    if targeted is not None:
        targeted.to_csv(out/"targeted_cohort_runs.csv.gz", index=False, compression="gzip")
    if targeted_summary is not None:
        targeted_summary.to_csv(out/"targeted_cohort_summary.csv", index=False)
    if metadata is not None:
        (out/"experiment_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n",
            encoding="utf-8")

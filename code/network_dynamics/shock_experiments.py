"""Node-level shock attribution and centrality-targeted SIS experiments.

This package is separate from network_structural_analysis. It consumes
persisted matrices/metrics, never mutates structural outputs, and preserves
the exact node ordering supplied by the structural pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr


def _csr(matrix):
    B = sparse.csr_matrix(matrix, dtype=float)
    B.sum_duplicates()
    B.eliminate_zeros()
    if B.shape[0] != B.shape[1] or B.shape[0] == 0:
        raise ValueError("B must be a non-empty square matrix.")
    if not np.isfinite(B.data).all() or np.any(B.data < 0):
        raise ValueError("B must contain finite, nonnegative weights.")
    if np.any(B.diagonal() != 0):
        raise ValueError("B must not contain self-loops.")
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
    """Deterministic Model A; B[i,j] denotes supplier i -> buyer j."""
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


DEFAULT_CENTRALITIES = (
    "supplier_out_degree", "buyer_in_degree", "undirected_projection_degree",
    "supplier_out_strength", "buyer_in_strength",
    "pagerank_supplier_to_buyer", "eigenvector_supplier_to_buyer",
)


def calculate_targeting_metrics(B, labels: Iterable[str] | None = None,
                                 structural_metrics: pd.DataFrame | None = None):
    """Calculate propagation-aware scores and append persisted structural scores.

    PageRank follows supplier -> buyer orientation. k-core, clustering and
    betweenness are copied from the persisted structural analysis (not
    recomputed or altered here), with node ID and label alignment checked.
    """
    B = _csr(B)
    n = B.shape[0]
    A = (B != 0).astype(np.int8).tocsr()
    out_degree = np.asarray(A.sum(axis=1)).ravel().astype(float)
    in_degree = np.asarray(A.sum(axis=0)).ravel().astype(float)
    out_strength = np.asarray(B.sum(axis=1)).ravel()
    in_strength = np.asarray(B.sum(axis=0)).ravel()
    undirected_degree = np.asarray(A.maximum(A.T).sum(axis=1)).ravel().astype(float)

    # Row-stochastic transition matrix on the actual propagation orientation.
    row_sum = np.asarray(B.sum(axis=1)).ravel()
    inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    transition = sparse.diags(inv) @ B
    rank = np.full(n, 1.0/n)
    dangling = row_sum == 0
    for _ in range(1000):
        updated = 0.85*(transition.T @ rank + rank[dangling].sum()/n) + 0.15/n
        if np.abs(updated-rank).sum() < 1e-12:
            rank = updated
            break
        rank = updated
    if rank.sum():
        rank /= rank.sum()

    # Standard directed eigenvector centrality: a node is important when
    # connected to important upstream nodes (left eigenvector of B).
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
    result = pd.DataFrame({
        "node_id": np.arange(n, dtype=int), "node": labels,
        "supplier_out_degree": out_degree, "buyer_in_degree": in_degree,
        "undirected_projection_degree": undirected_degree,
        "supplier_out_strength": out_strength, "buyer_in_strength": in_strength,
        "pagerank_supplier_to_buyer": rank,
        "eigenvector_supplier_to_buyer": eigenvector,
    })
    if structural_metrics is not None:
        structural = structural_metrics.copy()
        if len(structural) != n or "node_id" not in structural:
            raise ValueError("Persisted structural metrics must contain exactly one row per node.")
        structural = structural.sort_values("node_id").reset_index(drop=True)
        if not np.array_equal(structural["node_id"].to_numpy(dtype=int), np.arange(n)):
            raise ValueError("Persisted structural node IDs must be contiguous 0..N-1.")
        if "node" in structural and structural["node"].astype(str).tolist() != labels:
            raise ValueError("Persisted structural metrics do not match the supplied node index.")
        for column in ("betweenness", "k_core", "clustering"):
            if column in structural:
                result[column] = pd.to_numeric(structural[column], errors="raise").to_numpy()
    return result


def run_single_node_census(B, beta, rho, labels=None, node_metrics=None,
                           max_steps=300, tolerance=1e-8,
                           initial_infection=1.0, store_trajectories=False,
                           batch_size=64):
    """Test every node once as the sole initially affected node.

    Source nodes are evaluated in bounded batches using sparse-matrix/dense
    matrix products. This is numerically the same deterministic Model A update
    as one-source-at-a-time runs, but avoids repeating sparse traversals.
    """
    B = _csr(B)
    n = B.shape[0]
    beta, rho = float(beta), float(rho)
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("beta must be >=0 and rho must be in (0,1].")
    if int(max_steps) < 1 or float(tolerance) <= 0:
        raise ValueError("max_steps and tolerance must be positive.")
    if not 0 < float(initial_infection) <= 1:
        raise ValueError("initial_infection must be in (0,1].")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    metrics = calculate_targeting_metrics(B, labels) if node_metrics is None else node_metrics.copy()
    if len(metrics) != n or "node_id" not in metrics or "node" not in metrics:
        raise ValueError("node_metrics must contain one row per node, with node_id and node.")
    metrics = metrics.sort_values("node_id").reset_index(drop=True)
    if not np.array_equal(metrics.node_id.to_numpy(dtype=int), np.arange(n)):
        raise ValueError("node_metrics IDs must be contiguous 0..N-1.")
    lam = spectral_radius(B)
    beta_c = float(rho/lam) if lam > 0 else np.inf
    rows, trajectory_rows = [], []

    for start in range(0, n, batch_size):
        stop = min(start+batch_size, n)
        cols = stop-start
        D = np.zeros((n, cols), dtype=float)
        D[np.arange(start, stop), np.arange(cols)] = float(initial_infection)
        active = np.ones(cols, dtype=bool)
        extinct = np.zeros(cols, dtype=bool)
        converged = np.zeros(cols, dtype=bool)
        steps = np.full(cols, int(max_steps), dtype=int)
        p0 = D.mean(axis=0)
        peak = p0.copy()
        peak_step = np.zeros(cols, dtype=int)
        cumulative = np.zeros(cols, dtype=float)
        prevalence_history = [[float(p0[j])] for j in range(cols)] if store_trajectories else None
        exposure0 = np.asarray(B.T @ D).sum(axis=0)
        exposure_history = [[float(exposure0[j])] for j in range(cols)] if store_trajectories else None

        for step in range(1, int(max_steps)+1):
            if not active.any():
                break
            old = D
            exposure = B.T @ old
            candidate = np.clip((1-rho)*old + (1-old)*beta*exposure, 0.0, 1.0)
            candidate[:, ~active] = old[:, ~active]
            delta = np.max(np.abs(candidate-old), axis=0)
            D = candidate
            prevalence = D.mean(axis=0)
            exposure_mass = np.asarray(B.T @ D).sum(axis=0)
            for j in np.flatnonzero(active):
                p = float(prevalence[j])
                cumulative[j] += p
                if p > peak[j]:
                    peak[j] = p
                    peak_step[j] = step
                if store_trajectories:
                    prevalence_history[j].append(p)
                    exposure_history[j].append(float(exposure_mass[j]))
                if p <= 1e-8:
                    extinct[j] = True
                    steps[j] = step
                    active[j] = False
                elif float(delta[j]) <= float(tolerance):
                    converged[j] = True
                    steps[j] = step
                    active[j] = False

        for j, source in enumerate(range(start, stop)):
            final = float(D[:, j].mean())
            row = {
                "source_node_id": source, "source_node": str(metrics.loc[source, "node"]),
                "initial_infection": float(initial_infection), "beta": beta, "rho": rho,
                "beta_critical": beta_c, "lambda_max": lam,
                "peak_prevalence": float(peak[j]), "peak_step": int(peak_step[j]),
                "final_prevalence": final, "cumulative_prevalence": float(cumulative[j]),
                "extinction_time": int(steps[j]) if extinct[j] else np.nan,
                "steps": int(steps[j]), "converged": bool(converged[j]),
                "extinct": bool(extinct[j]),
            }
            for col in metrics.columns:
                if col not in ("node_id", "node"):
                    row[f"source_{col}"] = metrics.loc[source, col]
            rows.append(row)
            if store_trajectories:
                trajectory_rows.extend({
                    "source_node_id": source, "source_node": row["source_node"],
                    "step": t, "prevalence": value,
                    "exposure_mass": exposure_history[j][t],
                } for t, value in enumerate(prevalence_history[j]))

    census = pd.DataFrame(rows)
    census["impact_rank_peak"] = census["peak_prevalence"].rank(method="min", ascending=False).astype(int)
    census["impact_rank_cumulative"] = census["cumulative_prevalence"].rank(method="min", ascending=False).astype(int)
    census["impact_rank_final"] = census["final_prevalence"].rank(method="min", ascending=False).astype(int)
    census = census.sort_values(["impact_rank_cumulative", "source_node_id"]).reset_index(drop=True)
    return census, (pd.DataFrame(trajectory_rows) if store_trajectories else None)


def run_targeted_cohort_experiment(B, beta, rho, node_metrics, fraction=0.01,
                                   repeats=20, seed=20261030, max_steps=300,
                                   tolerance=1e-8, centralities=DEFAULT_CENTRALITIES,
                                   initial_infection=1.0):
    """Compare top/bottom centrality cohorts against matched-size random cohorts."""
    B = _csr(B)
    n = B.shape[0]
    fraction = float(fraction)
    k = max(1, int(np.ceil(fraction*n)))
    if not 0 < fraction <= 0.5 or int(repeats) < 1:
        raise ValueError("fraction must be in (0, 0.5] and repeats must be positive.")
    initial_infection = float(initial_infection)
    if not np.isfinite(initial_infection) or not 0 < initial_infection <= 1:
        raise ValueError("initial_infection must be in (0, 1].")
    metrics = node_metrics.sort_values("node_id").reset_index(drop=True)
    if len(metrics) != n:
        raise ValueError("node_metrics must have one row per network node.")
    rng = np.random.default_rng(seed)
    random_sets = [np.sort(rng.choice(n, k, replace=False)) for _ in range(int(repeats))]
    rows = []

    def evaluate(cohort, centrality, repeat_id, node_ids):
        state = np.zeros(n)
        state[np.asarray(node_ids, dtype=int)] = initial_infection
        outcome = simulate_model_a(B, state, beta, rho, max_steps, tolerance)
        rows.append({
            "cohort": cohort, "centrality": centrality, "repeat": repeat_id,
            "seed_count": k, "initial_infection": initial_infection,
            "seed_node_ids": "|".join(map(str, node_ids)),
            "seed_nodes": "|".join(metrics.loc[np.asarray(node_ids, dtype=int), "node"].astype(str)),
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
        order = np.lexsort((np.arange(n), -metrics[centrality].to_numpy(dtype=float)))
        evaluate("top", centrality, 0, order[:k])
        evaluate("bottom", centrality, 0, order[-k:][::-1])
        for repeat_id, sample in enumerate(random_sets, start=1):
            evaluate("random", centrality, repeat_id, sample)
    return pd.DataFrame(rows)


def summarize_targeted_cohorts(experiment):
    outcomes = ("peak_prevalence", "final_prevalence", "cumulative_prevalence")
    rows = []
    for (centrality, cohort), group in experiment.groupby(["centrality", "cohort"], sort=True):
        row = {"centrality": centrality, "cohort": cohort, "n_runs": len(group)}
        for metric in outcomes:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(np.nanmin(values))
            row[f"{metric}_max"] = float(np.nanmax(values))
        rows.append(row)
    summary = pd.DataFrame(rows)
    random = summary[summary.cohort == "random"].set_index("centrality")
    for metric in outcomes:
        summary[f"{metric}_difference_vs_random"] = np.nan
        for idx in summary.index[summary.cohort.isin(["top", "bottom"])]:
            centrality = summary.loc[idx, "centrality"]
            summary.loc[idx, f"{metric}_difference_vs_random"] = (
                summary.loc[idx, f"{metric}_mean"] -
                random.loc[centrality, f"{metric}_mean"]
            )
    return summary


def centrality_impact_correlations(census):
    """Spearman association between source centrality and SIS-impact measures."""
    impact = ("peak_prevalence", "cumulative_prevalence", "final_prevalence")
    excluded = {"source_node_id", "source_node", "initial_infection", "beta",
                "rho", "beta_critical", "lambda_max", "peak_step", "steps",
                "extinction_time", "converged", "extinct", "impact_rank_peak",
                "impact_rank_cumulative", "impact_rank_final"}
    candidate = [c for c in census.columns if c.startswith("source_") and c not in excluded]
    rows = []
    for centrality in candidate:
        x = pd.to_numeric(census[centrality], errors="coerce").to_numpy(dtype=float)
        for outcome in impact:
            y = pd.to_numeric(census[outcome], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3 and np.unique(x[valid]).size > 1 and np.unique(y[valid]).size > 1:
                rho_s, pvalue = spearmanr(x[valid], y[valid])
            else:
                rho_s, pvalue = np.nan, np.nan
            rows.append({
                "centrality": centrality.removeprefix("source_"),
                "impact_metric": outcome, "n_nodes": int(valid.sum()),
                "spearman_rho": float(rho_s) if np.isfinite(rho_s) else np.nan,
                "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
            })
    return pd.DataFrame(rows)


def write_experiment_outputs(out_dir, census=None, trajectories=None, targeted=None,
                             targeted_summary=None, correlations=None, metadata=None):
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
    if correlations is not None:
        correlations.to_csv(out/"centrality_impact_correlations.csv", index=False)
    if metadata is not None:
        (out/"experiment_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n",
            encoding="utf-8")

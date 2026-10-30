"""Paired, persisted ER/BA/WS ensembles calibrated to empirical networks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.stats import t as student_t
from scipy.sparse import save_npz
from .config import FAMILIES, SyntheticConfig
from .metrics import structural_analysis


def empirical_calibration(empirical_summaries):
    uu = [x for x in empirical_summaries if x.get("representation") == "UU"]
    ud = [x for x in empirical_summaries if x.get("representation") == "UD"]
    if not uu or not ud:
        raise ValueError("Empirical calibration needs both UU and UD summaries for 2019 and 2022.")
    mean_degree = float(np.mean([x["mean_degree_undirected_projection"] for x in uu]))
    valid_ud = [
        x for x in ud
        if np.isfinite(float(x.get("reciprocity", np.nan))) and int(x.get("edges", 0)) > 0
    ]
    if not valid_ud:
        raise ValueError("Could not derive empirical directed reciprocity.")
    total_arcs = sum(int(x["edges"]) for x in valid_ud)
    mutual_arcs = sum(float(x["reciprocity"]) * int(x["edges"]) for x in valid_ud)
    pooled_reciprocity = mutual_arcs / total_arcs if total_arcs else 0.0
    return {
        "reference_mean_degree": mean_degree,
        "reference_reciprocity": float(np.clip(pooled_reciprocity, 0.0, 1.0)),
        "reciprocity_pooling": "total_mutual_directed_arcs / total_directed_arcs across empirical years",
        "reciprocity_calibration_arc_counts": {
            str(x.get("year", "unknown")): int(x["edges"]) for x in valid_ud
        },
        "calibration_years": [2019, 2022],
    }


def _orientation_from_undirected(G, target_reciprocity, rng):
    """Orient each skeleton edge once, with expected arc-level reciprocity r."""
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes)
    r = float(np.clip(target_reciprocity, 0.0, 1.0))
    mutual_probability = r / (2.0 - r) if r < 1.0 else 1.0
    for u, v in G.edges:
        if rng.random() < mutual_probability:
            D.add_edge(u, v)
            D.add_edge(v, u)
        elif rng.random() < 0.5:
            D.add_edge(u, v)
        else:
            D.add_edge(v, u)
    return D


def _sample_weights(empirical_shares, size, rng):
    values = np.asarray(empirical_shares, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values) and size:
        raise ValueError("No positive empirical dependency-share weights available for synthetic WD.")
    return rng.choice(values, size=size, replace=True) if size else np.array([], dtype=float)


def _calibrate_weighted_adjacency(A_ud, empirical_shares, dependency_total_samples, seed,
                                  empirical_indegree_samples=None):
    """Sample edge shares and match buyer totals conditionally on synthetic in-degree.

    For a buyer with synthetic in-degree d, its target incoming weight is sampled
    from empirical buyers with the same retained in-degree. If that exact degree
    is absent, the nearest observed positive degree is used and documented by
    the returned target vector. Targets are synthetic dependency strengths, not
    monetary flows.
    """
    A_ud = sparse.csr_matrix(A_ud, dtype=np.uint8)
    row, col = A_ud.nonzero()
    rng = np.random.default_rng(seed)
    values = _sample_weights(empirical_shares, len(row), rng)
    targets = np.asarray(dependency_total_samples, dtype=float).ravel()
    targets = np.where(np.isfinite(targets) & (targets >= 0), targets, 0.0)
    if not len(targets):
        raise ValueError("Empirical buyer dependency totals are empty.")

    if empirical_indegree_samples is None:
        empirical_degree = None
    else:
        empirical_degree = np.asarray(empirical_indegree_samples, dtype=int).ravel()
        if len(empirical_degree) != len(targets):
            raise ValueError("Empirical in-degree and buyer-total samples must have equal lengths.")

    counts = np.bincount(col, minlength=A_ud.shape[1])
    target_by_buyer = np.zeros(A_ud.shape[1], dtype=float)
    for buyer, degree in enumerate(counts):
        if degree == 0:
            continue
        if empirical_degree is None:
            pool = np.flatnonzero(targets > 0)
        else:
            pool = np.flatnonzero((empirical_degree == degree) & (targets > 0))
            if not len(pool):
                positive_pool = np.flatnonzero((empirical_degree > 0) & (targets > 0))
                if not len(positive_pool):
                    raise ValueError("No positive empirical dependency totals for buyers with incoming edges.")
                nearest = np.min(np.abs(empirical_degree[positive_pool] - degree))
                pool = positive_pool[np.abs(empirical_degree[positive_pool] - degree) == nearest]
        if not len(pool):
            raise ValueError("No positive empirical buyer dependency-total samples.")
        target_by_buyer[buyer] = float(targets[rng.choice(pool)])

    order = np.argsort(col, kind="stable")
    row, col, values = row[order], col[order], values[order]
    starts = np.cumsum(counts) - counts
    for buyer in np.flatnonzero(counts):
        start = int(starts[buyer])
        stop = start + int(counts[buyer])
        total = float(values[start:stop].sum())
        if total > 0:
            values[start:stop] *= target_by_buyer[buyer] / total
    weighted = sparse.csr_matrix((values, (row, col)), shape=A_ud.shape)
    weighted.eliminate_zeros()
    return weighted, target_by_buyer


def generate_skeleton(family, n, reference_mean_degree, multiplier, ws_p, seed):
    n = int(n)
    reference_mean_degree = float(reference_mean_degree)
    if n < 3 or reference_mean_degree <= 0:
        raise ValueError("Synthetic graph size must be >=3 and reference degree must be positive.")
    if family == "ER":
        target = reference_mean_degree * float(multiplier)
        p = float(np.clip(target / (n - 1), 0.0, 1.0))
        graph = nx.gnp_random_graph(n, p, seed=int(seed), directed=False)
        return graph, {"target_mean_degree": target, "p": p}
    if family == "BA":
        target = reference_mean_degree * float(multiplier)
        m = max(1, min(n - 1, int(round(target / 2.0))))
        graph = nx.barabasi_albert_graph(n, m, seed=int(seed))
        return graph, {"target_mean_degree": target, "m": m, "expected_mean_degree_approx": 2.0 * m}
    if family == "WS":
        k = max(2, int(round(reference_mean_degree / 2.0) * 2))
        k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
        k = max(2, k)
        graph = nx.watts_strogatz_graph(n, k, float(ws_p), seed=int(seed))
        return graph, {"k": k, "rewire_probability": float(ws_p),
                       "reference_mean_degree": reference_mean_degree}
    raise ValueError(f"Unknown synthetic family: {family}")


def _adjacency(G, n):
    return sparse.csr_matrix(nx.to_scipy_sparse_array(
        G, nodelist=range(n), format="csr", dtype=np.uint8
    ))


def _write_degree_distribution(node_metrics, path, representation):
    column = "degree_undirected_projection"
    if representation in ("UD", "WD"):
        values = node_metrics[column].to_numpy(dtype=int)
    else:
        values = node_metrics[column].to_numpy(dtype=int)
    unique, counts = np.unique(values, return_counts=True)
    total = int(counts.sum())
    pd.DataFrame({"degree": unique, "node_count": counts,
                  "fraction_of_nodes": counts / total if total else counts}).to_csv(path, index=False)


def _add_distribution_summary(row, node_metrics):
    for column, prefix in (
        ("in_degree", "in_degree"),
        ("out_degree", "out_degree"),
        ("degree_undirected_projection", "undirected_degree"),
        ("in_strength", "in_strength"),
        ("out_strength", "out_strength"),
    ):
        values = node_metrics[column].to_numpy(dtype=float)
        q = np.quantile(values, [0, .25, .5, .75, .9, .99, 1])
        for label, value in zip(("min", "q25", "median", "q75", "q90", "q99", "max"), q):
            row[f"{prefix}_{label}"] = float(value)


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x)))


def run_synthetic_ensemble(n, empirical_mean_degree, empirical_reciprocity,
                           empirical_share_samples, realizations=20, seed=20261030,
                           out_dir=None, empirical_dependency_total_samples=None,
                           empirical_indegree_samples=None,
                           config: SyntheticConfig | None = None,
                           path_sample_sources=64, betweenness_k=64):
    """Create 9 configurations x R skeletons and persist all three paired matrices."""
    cfg = config or SyntheticConfig(realizations=realizations, seed=seed)
    realizations = int(realizations if config is None else cfg.realizations)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("matrices", "node_metrics", "degree_distributions", "skeleton_metadata"):
            (out / name).mkdir(parents=True, exist_ok=True)

    if empirical_dependency_total_samples is None:
        raise ValueError("Empirical dependency-total samples are required to calibrate synthetic WD.")
    rng = np.random.default_rng(seed)
    summary_rows, manifest_rows = [], []
    config_id = 0

    for family in FAMILIES:
        for level in range(3):
            config_id += 1
            if family in ("ER", "BA"):
                multiplier = cfg.mean_degree_multipliers[level] if family == "ER" else cfg.ba_multipliers[level]
            else:
                multiplier = 1.0
            ws_p = cfg.ws_rewire_probabilities[level]
            for realization in range(1, realizations + 1):
                graph_seed = int(rng.integers(0, 2**31 - 1))
                orientation_seed = int(rng.integers(0, 2**31 - 1))
                weight_seed = int(rng.integers(0, 2**31 - 1))
                G, params = generate_skeleton(family, n, empirical_mean_degree, multiplier, ws_p, graph_seed)
                D = _orientation_from_undirected(G, empirical_reciprocity,
                                                 np.random.default_rng(orientation_seed))
                A_uu = _adjacency(G, n)
                A_ud = _adjacency(D, n)
                A_wd, target_totals = _calibrate_weighted_adjacency(
                    A_ud, empirical_share_samples, empirical_dependency_total_samples,
                    weight_seed, empirical_indegree_samples
                )
                tracks = {"UU": A_uu, "UD": A_ud, "WD": A_wd}
                relative_dir = Path("matrices") / family / f"config_{level + 1}" / f"realization_{realization:03d}"
                meta_dir = Path("skeleton_metadata") / family / f"config_{level + 1}"
                realization_metadata = {
                    "family": family, "configuration": config_id, "level": level + 1,
                    "realization": realization, "node_count": int(n),
                    "graph_seed": graph_seed, "orientation_seed": orientation_seed,
                    "weight_seed": weight_seed, "empirical_reciprocity_target": float(empirical_reciprocity),
                    "params": params,
                    "paired_design": "One undirected skeleton is transformed into UU, UD and WD.",
                    "weight_calibration": "Empirical retained buyer totals sampled conditionally on synthetic buyer in-degree, with nearest-degree fallback only if an exact degree is absent.",
                    "synthetic_weights_are_monetary_flows": False,
                }
                if out:
                    _save_json(out / meta_dir / f"realization_{realization:03d}.json", realization_metadata)

                for representation, matrix in tracks.items():
                    if out:
                        matrix_path = out / relative_dir / f"{representation}.npz"
                        matrix_path.parent.mkdir(parents=True, exist_ok=True)
                        save_npz(matrix_path, matrix, compressed=True)
                        node_dir = out / "node_metrics" / family / f"config_{level + 1}" / f"realization_{realization:03d}"
                        node_dir.mkdir(parents=True, exist_ok=True)
                        dist_dir = out / "degree_distributions" / family / f"config_{level + 1}" / f"realization_{realization:03d}"
                        dist_dir.mkdir(parents=True, exist_ok=True)
                    labels = [f"node_{i}" for i in range(n)]
                    sm, node = structural_analysis(
                        matrix, labels, representation, seed=graph_seed,
                        path_sample_sources=min(int(path_sample_sources), n),
                        betweenness_k=min(int(betweenness_k), n),
                        include_node_metrics=True,
                    )
                    network_id = f"syn_{family}_{level + 1}_{realization:03d}_{representation}"
                    row = {
                        "network_id": network_id, "network_type": "synthetic",
                        "family": family, "configuration": config_id, "level": level + 1,
                        "realization": realization, "representation": representation,
                        "graph_seed": graph_seed, "orientation_seed": orientation_seed,
                        "weight_seed": weight_seed, **params, **sm,
                    }
                    _add_distribution_summary(row, node)
                    summary_rows.append(row)
                    if out:
                        node_path = node_dir / f"{representation}.csv.gz"
                        node.to_csv(node_path, index=False, compression="gzip")
                        degree_path = dist_dir / f"{representation}.csv"
                        _write_degree_distribution(node, degree_path, representation)
                        manifest_rows.append({
                            **{k: row[k] for k in ("network_id", "network_type", "family", "configuration",
                                "level", "realization", "representation", "graph_seed", "orientation_seed",
                                "weight_seed")},
                            "matrix_path": str((relative_dir / f"{representation}.npz").as_posix()),
                            "node_metrics_path": str((Path("node_metrics") / family / f"config_{level + 1}" /
                                f"realization_{realization:03d}" / f"{representation}.csv.gz").as_posix()),
                            "degree_distribution_path": str((Path("degree_distributions") / family /
                                f"config_{level + 1}" / f"realization_{realization:03d}" /
                                f"{representation}.csv").as_posix()),
                        })
                if out:
                    # Buyer target sums are retained for post-generation auditability.
                    pd.DataFrame({
                        "buyer_node_id": np.arange(n),
                        "synthetic_in_degree": np.asarray(A_ud.getnnz(axis=0)).ravel(),
                        "target_incoming_dependency_total": target_totals,
                        "actual_incoming_dependency_total": np.asarray(A_wd.sum(axis=0)).ravel(),
                    }).to_csv(out / relative_dir / "WD_buyer_weight_calibration.csv.gz",
                              index=False, compression="gzip")

    summaries = pd.DataFrame(summary_rows)
    if out:
        summaries.to_csv(out / "synthetic_realization_summaries.csv.gz", index=False, compression="gzip")
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(out / "synthetic_network_manifest.csv", index=False)
        numeric_metrics = ["mean_degree", "density", "average_clustering_undirected_projection",
                           "giant_component_fraction", "reciprocity",
                           "spectral_radius_adjacency_or_dependency"]
        grouped = summaries.groupby(["family", "configuration", "representation"], as_index=False)
        aggregates = []
        for keys, group in grouped:
            row = dict(zip(("family", "configuration", "representation"), keys))
            row["realizations"] = int(group["realization"].nunique())
            for metric in numeric_metrics:
                values = group[metric].dropna().to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
                half = float(student_t.ppf(.975, len(values) - 1) * se) if len(values) > 1 else np.nan
                row[f"{metric}_ci95_low"] = float(values.mean() - half) if len(values) > 1 else np.nan
                row[f"{metric}_ci95_high"] = float(values.mean() + half) if len(values) > 1 else np.nan
            aggregates.append(row)
        pd.DataFrame(aggregates).to_csv(out / "synthetic_configuration_summaries.csv", index=False)
        _save_json(out / "synthetic_experiment_design.json", {
            "families": list(FAMILIES), "configurations_per_family": 3,
            "realizations_per_configuration": realizations,
            "independent_skeletons": 9 * realizations,
            "paired_representation_matrices": 27 * realizations,
            "node_count": int(n), "reference_mean_degree": float(empirical_mean_degree),
            "empirical_reciprocity_target": float(empirical_reciprocity),
            "er_expected_degree_multipliers": list(cfg.mean_degree_multipliers),
            "ba_target_degree_multipliers": list(cfg.ba_multipliers),
            "ws_rewire_probabilities": list(cfg.ws_rewire_probabilities),
            "ws_degree": "Fixed at nearest even integer to the empirical reference mean UU degree.",
            "weight_calibration": "Empirical nonzero retained dependency shares; per-buyer retained totals sampled conditionally on synthetic in-degree with nearest-degree fallback.",
            "paired_seed_policy_for_dynamics": "Five initial seed sets per skeleton; same seed sets are reused across UU, UD and WD.",
            "matrices_persisted": bool(out),
        })
    return summaries

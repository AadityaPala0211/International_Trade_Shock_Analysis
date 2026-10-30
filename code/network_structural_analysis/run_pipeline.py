"""CLI entry point for network generation and structural analysis only."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import networkx
import matplotlib
from scipy import sparse
from .config import PipelineConfig, YEARS, REPRESENTATIONS
from .data_loader import load_icio
from .provenance import inspect_icio_readme
from .network_builder import make_representations
from .thresholding import diagnose_thresholds
from .metrics import structural_analysis, country_sector_summary
from .synthetic import empirical_calibration, run_synthetic_ensemble
from .compare import compare_node_metrics, top_node_table
from .visualization import (
    plot_degree_distribution, plot_strength_distribution,
    plot_metric_comparison, plot_synthetic_comparison,
)


def _json_write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer): return int(x)
        return str(x)
    Path(path).write_text(json.dumps(obj, indent=2, default=default))


def _save_sparse(path, A):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from scipy.sparse import save_npz
    save_npz(path, A)


def _save_empirical(year, reps, labels, base_out, seed, path_sample_sources, betweenness_k):
    summaries = []
    metrics_by_rep = {}
    for rep, A in reps.items():
        target = base_out / "empirical" / str(year) / rep
        target.mkdir(parents=True, exist_ok=True)
        _save_sparse(target / "adjacency.npz", A)
        sm, node = structural_analysis(
            A, labels, rep, seed=seed,
            path_sample_sources=path_sample_sources,
            betweenness_k=betweenness_k, include_node_metrics=True,
        )
        sm["year"] = year
        summaries.append(sm)
        metrics_by_rep[rep] = node
        node.to_csv(target / "node_metrics.csv.gz", index=False, compression="gzip")
        # Store undirected edges once; retain supplier->buyer orientation for UD/WD.
        if rep == "UU":
            coo = sparse.triu(A, k=1, format="coo")
            edge = pd.DataFrame({
                "node_u_id": coo.row, "node_v_id": coo.col,
                "node_u": np.asarray(labels, dtype=object)[coo.row],
                "node_v": np.asarray(labels, dtype=object)[coo.col],
                "weight": coo.data,
            })
        else:
            coo = A.tocoo()
            edge = pd.DataFrame({
                "supplier_node_id": coo.row, "buyer_node_id": coo.col,
                "supplier": np.asarray(labels, dtype=object)[coo.row],
                "buyer": np.asarray(labels, dtype=object)[coo.col],
                "weight": coo.data,
            })
        edge.to_csv(target / "edge_list.csv.gz", index=False, compression="gzip")
        country, sector = country_sector_summary(node)
        country.to_csv(target / "country_summary.csv", index=False)
        sector.to_csv(target / "sector_summary.csv", index=False)
        plot_degree_distribution(node, f"{year} {rep}: degree distribution", target / "degree_distribution.png")
        if rep == "WD":
            plot_strength_distribution(node, f"{year} WD: dependency strength", target / "strength_distribution.png")
    return summaries, metrics_by_rep


def _inspect(data_dir, expected_nodes):
    info = {}
    for year in YEARS:
        path = Path(data_dir) / f"{year}_SML.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_csv(path, header=None, nrows=4, dtype=str)
        info[str(year)] = {
            "path": str(path), "bytes": path.stat().st_size,
            "first_rows_preview": raw.fillna("").iloc[:3, :8].values.tolist(),
        }
    print(json.dumps({"expected_nodes": expected_nodes, "files": info}, indent=2))


def run(args=None):
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results_network_analysis")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--expected-nodes", type=int, default=cfg.expected_nodes)
    parser.add_argument("--realizations", type=int, default=cfg.synthetic.realizations)
    parser.add_argument("--seed", type=int, default=cfg.synthetic.seed)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--path-sample-sources", type=int, default=cfg.threshold.path_sample_sources)
    parser.add_argument("--betweenness-k", type=int, default=cfg.threshold.betweenness_k)
    parser.add_argument("--tau-candidates", default=",".join(str(x) for x in cfg.threshold.candidates))
    ns = parser.parse_args(args)

    data_dir = Path(ns.data_dir)
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    for subdir in ("metadata", "comparisons", "figures", "threshold_review", "empirical", "synthetic"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    if ns.inspect_only:
        _inspect(data_dir, ns.expected_nodes)
        return

    # Both editions are loaded before threshold selection so one common tau is used.
    loaded = {
        year: load_icio(data_dir / f"{year}_SML.csv", expected_nodes=ns.expected_nodes)
        for year in YEARS
    }

    # Align 2022 to 2019 explicitly by label; reject different node universes.
    if loaded[2019]["labels"] != loaded[2022]["labels"]:
        labels_2019 = loaded[2019]["labels"]
        labels_2022 = loaded[2022]["labels"]
        if set(labels_2019) != set(labels_2022):
            missing = sorted(set(labels_2019) - set(labels_2022))[:10]
            extra = sorted(set(labels_2022) - set(labels_2019))[:10]
            raise ValueError(f"2019/2022 node universes differ. Missing in 2022={missing}; extra in 2022={extra}")
        position = {label: i for i, label in enumerate(labels_2022)}
        order = np.asarray([position[label] for label in labels_2019], dtype=int)
        loaded[2022]["Z"] = loaded[2022]["Z"][np.ix_(order, order)]
        loaded[2022]["labels"] = labels_2019
        loaded[2022]["reordered_to_2019_label_order"] = True

    Zs = {year: loaded[year]["Z"] for year in YEARS}
    candidates = tuple(float(x) for x in ns.tau_candidates.split(","))
    diagnostics, pooled, recommendation, shares_by_year = diagnose_thresholds(
        Zs, candidates, out / "threshold_review",
        min_flow_coverage=cfg.threshold.recommended_min_flow_coverage,
        max_density=cfg.threshold.recommended_max_density,
        local_sensitivity_factor=cfg.threshold.local_sensitivity_factor,
    )
    tau = float(ns.threshold if ns.threshold is not None else recommendation["recommended_threshold"])

    _json_write(out / "metadata" / "icio_readme_metadata.json",
                inspect_icio_readme(data_dir / "ReadMe_ICIO_small.xlsx"))
    pd.DataFrame({
        "node_id": np.arange(len(loaded[2019]["labels"])),
        "node": loaded[2019]["labels"],
    }).to_csv(out / "metadata" / "node_index.csv", index=False)
    _json_write(out / "metadata" / "run_metadata.json", {
        "years": list(YEARS), "node_count": ns.expected_nodes, "threshold": tau,
        "threshold_recommendation": recommendation, "threshold_overridden": ns.threshold is not None,
        "seed": ns.seed, "realizations_per_synthetic_configuration": ns.realizations,
        "analysis_config": cfg.to_dict(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "networkx": networkx.__version__, "matplotlib": matplotlib.__version__,
        "inputs": {
            str(year): {
                "path": loaded[year]["source_path"], "sha256": loaded[year]["source_sha256"],
                "shape": list(loaded[year]["Z"].shape), "raw_shape": list(loaded[year]["raw_shape"]),
                "block_bounds_zero_based": loaded[year]["block_bounds_zero_based"],
                "total_intermediate_flow": loaded[year]["total_intermediate_flow"],
                "self_flow": loaded[year]["self_flow"],
                "reordered_to_2019_label_order": loaded[year].get("reordered_to_2019_label_order", False),
            } for year in YEARS
        },
        "node_label_sha256": hashlib.sha256("\\n".join(loaded[2019]["labels"]).encode()).hexdigest(),
        "interpretation": "Dependency-share thresholding; no dynamics implemented.",
    })

    if ns.threshold_only:
        print(json.dumps({
            "recommended_threshold": tau, "rationale": recommendation["rationale"], "out": str(out),
        }, indent=2))
        return

    summaries = []
    metrics_by_year_rep = {}
    if not ns.synthetic_only:
        for year in YEARS:
            reps = make_representations(shares_by_year[year], tau)
            year_summaries, year_metrics = _save_empirical(
                year, reps, loaded[year]["labels"], out, ns.seed,
                ns.path_sample_sources, ns.betweenness_k,
            )
            summaries.extend(year_summaries)
            for rep, node in year_metrics.items():
                metrics_by_year_rep[(year, rep)] = node

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out / "empirical_network_summaries.csv", index=False)
        for metric in ("density", "mean_degree", "average_clustering_undirected_projection",
                       "giant_component_fraction", "spectral_radius_adjacency_or_dependency", "reciprocity"):
            plot_metric_comparison(summary_df, out / "figures" / f"empirical_{metric}_comparison.png", metric)
        compare_node_metrics(metrics_by_year_rep).to_csv(
            out / "comparisons" / "node_metric_year_comparison.csv", index=False)
        for year in YEARS:
            for rep in REPRESENTATIONS:
                node = metrics_by_year_rep[(year, rep)]
                for metric in ("in_degree", "out_degree", "degree_undirected_projection",
                               "out_strength", "in_strength", "pagerank", "betweenness", "k_core"):
                    top_node_table(node, metric, 25).to_csv(
                        out / "comparisons" / f"top_{metric}_{year}_{rep}.csv", index=False)

    if not ns.skip_synthetic:
        # Synthetic weight calibration uses only retained empirical dependency
        # shares and per-buyer retained off-diagonal dependency totals.
        positive_samples = []
        dependency_total_samples = []
        for year in YEARS:
            S = shares_by_year[year]
            retained = S[(S >= tau) & (S > 0)]
            if len(retained) > cfg.synthetic.weight_distribution_sample_cap:
                rng = np.random.default_rng(ns.seed + year)
                retained = rng.choice(retained, cfg.synthetic.weight_distribution_sample_cap, replace=False)
            positive_samples.append(retained)
            dependency_total_samples.append(np.where((S >= tau) & (S > 0), S, 0.0).sum(axis=0))
        pooled_weights = np.concatenate(positive_samples)
        pooled_buyer_totals = np.concatenate(dependency_total_samples)

        empirical_summaries = summaries
        if ns.synthetic_only:
            for year in YEARS:
                reps = make_representations(shares_by_year[year], tau)
                for rep in ("UU", "UD"):
                    sm, _ = structural_analysis(
                        reps[rep], loaded[year]["labels"], rep, seed=ns.seed,
                        path_sample_sources=ns.path_sample_sources,
                        betweenness_k=ns.betweenness_k, include_node_metrics=False,
                    )
                    empirical_summaries.append(sm)

        calibration = empirical_calibration(empirical_summaries)
        _json_write(out / "metadata" / "synthetic_calibration.json", calibration)
        synthetic = run_synthetic_ensemble(
            ns.expected_nodes, calibration["reference_mean_degree"],
            calibration["reference_reciprocity"], pooled_weights,
            realizations=ns.realizations, seed=ns.seed, out_dir=out / "synthetic",
            empirical_dependency_total_samples=pooled_buyer_totals, config=cfg.synthetic,
        )
        for metric in ("mean_degree", "density", "average_clustering_undirected_projection",
                       "giant_component_fraction", "reciprocity"):
            plot_synthetic_comparison(synthetic, out / "figures" / f"synthetic_{metric}.png", metric)

    print(json.dumps({"status": "completed", "threshold": tau, "nodes": ns.expected_nodes, "out": str(out)}, indent=2))


if __name__ == "__main__":
    run()

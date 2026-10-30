"""Staged ICIO network pipeline: threshold review, real, synthetic and SIS dynamics."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import networkx
import matplotlib
from scipy import sparse
from scipy.sparse import save_npz
from .config import PipelineConfig, YEARS, REPRESENTATIONS, ETA_LEVELS
from .data_loader import load_icio
from .provenance import inspect_icio_readme, summarize_icio_provenance
from .network_builder import make_representations
from .thresholding import diagnose_thresholds
from .metrics import structural_analysis, country_sector_summary
from .synthetic import empirical_calibration, run_synthetic_ensemble
from .compare import compare_node_metrics, top_node_table
from .visualization import (
    plot_degree_distribution, plot_strength_distribution,
    plot_metric_comparison, plot_synthetic_comparison,
)
from .dynamics import run_dynamics_analysis


def _json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(value):
        if isinstance(value, np.floating): return float(value)
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.ndarray): return value.tolist()
        return str(value)
    path.write_text(json.dumps(obj, indent=2, default=default))


def _load_and_align(data_dir, expected_nodes):
    loaded = {
        year: load_icio(Path(data_dir) / f"{year}_SML.csv", expected_nodes=expected_nodes)
        for year in YEARS
    }
    labels_2019, labels_2022 = loaded[2019]["labels"], loaded[2022]["labels"]
    if labels_2019 != labels_2022:
        if set(labels_2019) != set(labels_2022):
            missing = sorted(set(labels_2019) - set(labels_2022))[:10]
            extra = sorted(set(labels_2022) - set(labels_2019))[:10]
            raise ValueError(f"2019/2022 node universes differ. Missing in 2022={missing}; extra in 2022={extra}")
        position = {label: i for i, label in enumerate(labels_2022)}
        order = np.asarray([position[label] for label in labels_2019], dtype=int)
        loaded[2022]["Z"] = loaded[2022]["Z"][np.ix_(order, order)]
        loaded[2022]["labels"] = labels_2019
        loaded[2022]["reordered_to_2019_label_order"] = True
    else:
        loaded[2022]["reordered_to_2019_label_order"] = False
    return loaded


def _record_input_metadata(data_dir, loaded, out, cfg, threshold, seed, realizations):
    workbook_path = Path(data_dir) / "ReadMe_ICIO_small.xlsx"
    workbook_info = inspect_icio_readme(workbook_path)
    provenance = summarize_icio_provenance(workbook_info, [loaded[y] for y in YEARS])
    _json_write(out / "metadata" / "icio_readme_metadata.json", workbook_info)
    _json_write(out / "metadata" / "source_provenance_audit.json", provenance)
    labels = loaded[2019]["labels"]
    pd.DataFrame({"node_id": np.arange(len(labels)), "node": labels}).to_csv(
        out / "metadata" / "node_index.csv", index=False)
    metadata = {
        "years": list(YEARS), "node_count": len(labels), "threshold": threshold,
        "threshold_status": "analyst_selected" if threshold is not None else "not_selected",
        "seed": int(seed), "realizations_per_synthetic_configuration": int(realizations),
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
        "node_label_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest(),
        "source_provenance_audit": provenance,
        "interpretation": "Structural network analysis; dynamical analysis is a separate stage.",
    }
    _json_write(out / "metadata" / "run_metadata.json", metadata)


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


def _save_empirical(year, reps, labels, base_out, seed, path_sample_sources, betweenness_k):
    summaries, metrics_by_rep, manifest = [], {}, []
    for representation, matrix in reps.items():
        target = base_out / "empirical" / str(year) / representation
        target.mkdir(parents=True, exist_ok=True)
        save_npz(target / "adjacency.npz", sparse.csr_matrix(matrix), compressed=True)
        summary, node = structural_analysis(
            matrix, labels, representation, seed=seed,
            path_sample_sources=path_sample_sources,
            betweenness_k=betweenness_k, include_node_metrics=True,
        )
        summary["year"] = int(year)
        summaries.append(summary)
        metrics_by_rep[representation] = node
        node.to_csv(target / "node_metrics.csv.gz", index=False, compression="gzip")
        if representation == "UU":
            coo = sparse.triu(matrix, k=1, format="coo")
            edge = pd.DataFrame({
                "node_u_id": coo.row, "node_v_id": coo.col,
                "node_u": np.asarray(labels, dtype=object)[coo.row],
                "node_v": np.asarray(labels, dtype=object)[coo.col],
                "weight": coo.data,
            })
        else:
            coo = sparse.coo_matrix(matrix)
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
        plot_degree_distribution(node, f"{year} {representation}: degree distribution",
                                 target / "degree_distribution.png")
        if representation == "WD":
            plot_strength_distribution(node, f"{year} WD: dependency strength",
                                        target / "strength_distribution.png")
        manifest.append({
            "network_id": f"real_{year}_{representation}", "network_type": "real",
            "year": int(year), "representation": representation, "n_nodes": len(labels),
            "matrix_path": str((Path("empirical") / str(year) / representation / "adjacency.npz").as_posix()),
        })
    return summaries, metrics_by_rep, manifest


def _threshold_stage(data_dir, out, expected_nodes, cfg, tau_candidates):
    loaded = _load_and_align(data_dir, expected_nodes)
    Zs = {year: loaded[year]["Z"] for year in YEARS}
    diagnostics, pooled, selection, _ = diagnose_thresholds(
        Zs, tau_candidates, out / "threshold_review",
    )
    _record_input_metadata(data_dir, loaded, out, cfg, None,
                           cfg.synthetic.seed, cfg.synthetic.realizations)
    print(json.dumps({
        "status": "threshold_diagnostics_ready",
        "threshold_selected": False,
        "next_step": "Inspect threshold_review figures and CSVs, then run --stage real --threshold <chosen_tau>.",
        "candidate_count": len(tau_candidates), "out": str(out),
    }, indent=2))


def _real_stage(data_dir, out, threshold, expected_nodes, cfg, seed,
                realizations, path_sample_sources, betweenness_k, tau_candidates):
    if threshold is None:
        raise ValueError("The real stage requires an explicit --threshold selected after threshold review.")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("--threshold must be finite and nonnegative.")
    loaded = _load_and_align(data_dir, expected_nodes)
    Zs = {year: loaded[year]["Z"] for year in YEARS}
    diagnostics, pooled, selection, shares_by_year = diagnose_thresholds(
        Zs, tau_candidates, out / "threshold_review",
        selected_threshold=threshold, local_sensitivity_factor=cfg.threshold.local_sensitivity_factor,
    )
    _record_input_metadata(data_dir, loaded, out, cfg, threshold, seed, realizations)
    (out / "empirical").mkdir(parents=True, exist_ok=True)

    summaries, metrics_by_year_rep, manifest = [], {}, []
    for year in YEARS:
        reps = make_representations(shares_by_year[year], threshold)
        year_summaries, year_metrics, year_manifest = _save_empirical(
            year, reps, loaded[year]["labels"], out, seed,
            path_sample_sources, betweenness_k,
        )
        summaries.extend(year_summaries)
        manifest.extend(year_manifest)
        for representation, node in year_metrics.items():
            metrics_by_year_rep[(year, representation)] = node

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "empirical_network_summaries.csv", index=False)
    pd.DataFrame(manifest).to_csv(out / "empirical_network_manifest.csv", index=False)
    for metric in ("density", "mean_degree", "average_clustering_undirected_projection",
                   "giant_component_fraction", "spectral_radius_adjacency_or_dependency", "reciprocity"):
        plot_metric_comparison(summary_df, out / "figures" / f"empirical_{metric}_comparison.png", metric)
    compare_node_metrics(metrics_by_year_rep).to_csv(
        out / "comparisons" / "node_metric_year_comparison.csv", index=False)
    for year in YEARS:
        for representation in REPRESENTATIONS:
            node = metrics_by_year_rep[(year, representation)]
            for metric in ("in_degree", "out_degree", "degree_undirected_projection",
                           "out_strength", "in_strength", "pagerank", "betweenness", "k_core"):
                top_node_table(node, metric, 25).to_csv(
                    out / "comparisons" / f"top_{metric}_{year}_{representation}.csv", index=False)

    # Persist the exact empirical samples used later to calibrate synthetic WD.
    weights, buyer_totals, buyer_indegrees = [], [], []
    per_year_cap = max(1, cfg.synthetic.weight_distribution_sample_cap // len(YEARS))
    for year in YEARS:
        S = shares_by_year[year]
        mask = (S >= threshold) & (S > 0)
        np.fill_diagonal(mask, False)
        retained = S[mask]
        if len(retained) > per_year_cap:
            rng = np.random.default_rng(seed + year)
            retained = rng.choice(retained, per_year_cap, replace=False)
        weights.append(np.asarray(retained, dtype=float))
        buyer_totals.append(np.where(mask, S, 0.0).sum(axis=0))
        buyer_indegrees.append(mask.sum(axis=0).astype(int))
    pooled_weights = np.concatenate(weights)
    pooled_buyer_totals = np.concatenate(buyer_totals)
    pooled_buyer_indegrees = np.concatenate(buyer_indegrees)
    if not len(pooled_weights):
        raise ValueError("The selected threshold retained no positive empirical WD edges.")
    calibration = empirical_calibration(summaries)
    calibration.update({
        "threshold": threshold, "node_count": expected_nodes,
        "weight_sample_count": int(len(pooled_weights)),
        "buyer_total_sample_count": int(len(pooled_buyer_totals)),
        "buyer_indegree_sample_count": int(len(pooled_buyer_indegrees)),
        "weight_distribution_source": "positive dependency shares retained at the selected common threshold in 2019 and 2022",
        "buyer_total_conditioning": "retained buyer dependency totals paired with each buyer's retained in-degree",
        "synthetic_weights_are_monetary_flows": False,
    })
    np.savez_compressed(
        out / "metadata" / "synthetic_calibration_samples.npz",
        empirical_share_samples=pooled_weights,
        empirical_dependency_total_samples=pooled_buyer_totals,
        empirical_indegree_samples=pooled_buyer_indegrees,
    )
    _json_write(out / "metadata" / "synthetic_calibration.json", calibration)
    _json_write(out / "metadata" / "real_stage_complete.json", {
        "threshold": threshold, "empirical_network_matrices": len(manifest),
        "expected_empirical_matrices": 6, "synthetic_calibration_saved": True,
        "synthetic_stage_run": False,
    })
    print(json.dumps({
        "status": "real_networks_completed", "threshold": threshold,
        "empirical_matrices": len(manifest),
        "next_step": "Review empirical outputs. Run --stage synthetic only after approving the real-data results.",
        "out": str(out),
    }, indent=2))


def _synthetic_stage(out, cfg, realizations, seed, path_sample_sources, betweenness_k):
    calibration_path = out / "metadata" / "synthetic_calibration.json"
    samples_path = out / "metadata" / "synthetic_calibration_samples.npz"
    if not calibration_path.exists() or not samples_path.exists():
        raise FileNotFoundError("Run --stage real first; the saved empirical calibration files are missing.")
    calibration = json.loads(calibration_path.read_text())
    samples = np.load(samples_path)
    synthetic_cfg = replace(cfg.synthetic, realizations=int(realizations), seed=int(seed))
    synthetic = run_synthetic_ensemble(
        int(calibration["node_count"]),
        float(calibration["reference_mean_degree"]),
        float(calibration["reference_reciprocity"]),
        samples["empirical_share_samples"],
        realizations=realizations, seed=seed, out_dir=out / "synthetic",
        empirical_dependency_total_samples=samples["empirical_dependency_total_samples"],
        empirical_indegree_samples=samples["empirical_indegree_samples"],
        config=synthetic_cfg,
        path_sample_sources=path_sample_sources, betweenness_k=betweenness_k,
    )
    for metric in ("mean_degree", "density", "average_clustering_undirected_projection",
                   "giant_component_fraction", "reciprocity"):
        plot_synthetic_comparison(synthetic, out / "figures" / f"synthetic_{metric}.png", metric)
    _json_write(out / "metadata" / "synthetic_stage_complete.json", {
        "realization_count_per_configuration": int(realizations),
        "independent_skeletons": 9 * int(realizations),
        "paired_representation_matrices": 27 * int(realizations),
        "matrices_persisted": True,
        "sis_stage_run": False,
    })
    print(json.dumps({
        "status": "synthetic_networks_completed",
        "independent_skeletons": 9 * int(realizations),
        "paired_representation_matrices": 27 * int(realizations),
        "out": str(out / "synthetic"),
        "next_step": "Run --stage dynamics --scope real or synthetic when ready to analyse shock propagation.",
    }, indent=2))


def _load_manifest_records(path, base_dir, expected_type):
    if not path.exists():
        raise FileNotFoundError(f"Required network manifest not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Network manifest is empty: {path}")
    records = frame.to_dict(orient="records")
    for record in records:
        record["base_dir"] = str(base_dir)
        record["network_type"] = expected_type
        record["n_nodes"] = int(record.get("n_nodes", 0) or 0)
    return records


def _dynamics_stage(out, scope, seed, rho, beta_by_rep, max_steps, tolerance,
                    real_seed_sets, synthetic_seed_sets, infection_fraction):
    records = []
    if scope in ("real", "all"):
        records.extend(_load_manifest_records(
            out / "empirical_network_manifest.csv", out, "real"))
    if scope in ("synthetic", "all"):
        records.extend(_load_manifest_records(
            out / "synthetic" / "synthetic_network_manifest.csv",
            out / "synthetic", "synthetic"))
    if not records:
        raise ValueError("No matrices were found for the requested dynamics scope.")
    destination = out / "dynamics" / scope
    network_results, seed_results = run_dynamics_analysis(
        records, destination, seed=seed, rho=rho,
        beta_by_rep=beta_by_rep, max_steps=max_steps, tolerance=tolerance,
        real_seed_sets=real_seed_sets, synthetic_seed_sets=synthetic_seed_sets,
        infection_fraction=infection_fraction,
    )
    print(json.dumps({
        "status": "dynamics_completed", "scope": scope,
        "networks_analysed": int(network_results["network_id"].nunique()),
        "network_scenario_rows": int(len(network_results)),
        "seed_scenario_rows": int(len(seed_results)),
        "out": str(destination),
    }, indent=2))


def run(args=None):
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "threshold", "real", "synthetic", "dynamics"),
                        default="threshold")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results_network_analysis")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Analyst-selected common dependency-share threshold; required for --stage real.")
    parser.add_argument("--expected-nodes", type=int, default=cfg.expected_nodes)
    parser.add_argument("--realizations", type=int, default=cfg.synthetic.realizations)
    parser.add_argument("--seed", type=int, default=cfg.synthetic.seed)
    parser.add_argument("--path-sample-sources", type=int, default=cfg.threshold.path_sample_sources)
    parser.add_argument("--betweenness-k", type=int, default=cfg.threshold.betweenness_k)
    parser.add_argument("--tau-candidates", default=",".join(str(x) for x in cfg.threshold.candidates))
    parser.add_argument("--scope", choices=("real", "synthetic", "all"), default="real")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--beta-uu", type=float, default=None)
    parser.add_argument("--beta-ud", type=float, default=None)
    parser.add_argument("--beta-wd", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--real-seed-sets", type=int, default=20)
    parser.add_argument("--synthetic-seed-sets", type=int, default=5)
    parser.add_argument("--infection-fraction", type=float, default=0.01)
    ns = parser.parse_args(args)

    data_dir = Path(ns.data_dir)
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("metadata", "comparisons", "figures", "threshold_review"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if ns.stage == "inspect":
        _inspect(data_dir, ns.expected_nodes)
        return
    tau_candidates = tuple(float(x.strip()) for x in ns.tau_candidates.split(",") if x.strip())
    if ns.stage == "threshold":
        _threshold_stage(data_dir, out, ns.expected_nodes, cfg, tau_candidates)
    elif ns.stage == "real":
        _real_stage(data_dir, out, ns.threshold, ns.expected_nodes, cfg, ns.seed,
                    ns.realizations, ns.path_sample_sources, ns.betweenness_k, tau_candidates)
    elif ns.stage == "synthetic":
        _synthetic_stage(out, cfg, ns.realizations, ns.seed,
                         ns.path_sample_sources, ns.betweenness_k)
    elif ns.stage == "dynamics":
        required = {"rho": ns.rho, "beta-uu": ns.beta_uu,
                    "beta-ud": ns.beta_ud, "beta-wd": ns.beta_wd}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("For --stage dynamics, provide --rho, --beta-uu, --beta-ud and --beta-wd.")
        if ns.max_steps < 1 or ns.tolerance <= 0:
            parser.error("--max-steps must be positive and --tolerance must be positive.")
        if not 0 < ns.infection_fraction <= 1:
            parser.error("--infection-fraction must be in (0, 1].")
        _dynamics_stage(
            out, ns.scope, ns.seed, ns.rho,
            {"UU": ns.beta_uu, "UD": ns.beta_ud, "WD": ns.beta_wd},
            ns.max_steps, ns.tolerance, ns.real_seed_sets,
            ns.synthetic_seed_sets, ns.infection_fraction,
        )


if __name__ == "__main__":
    run()

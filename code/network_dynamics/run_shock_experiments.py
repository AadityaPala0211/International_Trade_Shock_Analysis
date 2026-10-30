"""Command-line runner for additive node-level SIS shock attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from .shock_experiments import (
    DEFAULT_CENTRALITIES,
    calculate_targeting_metrics,
    centrality_impact_correlations,
    run_single_node_census,
    run_targeted_cohort_experiment,
    spectral_radius,
    summarize_targeted_cohorts,
    write_experiment_outputs,
)

ETA_LEVELS = (0.5, 0.9, 1.0, 1.1, 1.5, 2.0)
DEFAULT_SCENARIOS = ("absolute", "eta_0.5", "eta_0.9", "eta_1",
                     "eta_1.1", "eta_1.5", "eta_2")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if np.isnan(x):
            return None
        if np.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _read_network_records(results_root, scope, synthetic_realizations):
    records = []
    if scope in ("real", "all"):
        manifest = results_root / "empirical_network_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing empirical manifest: {manifest}")
        frame = pd.read_csv(manifest)
        required = {"network_id", "year", "representation", "matrix_path"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Empirical manifest missing columns: {sorted(required-set(frame.columns))}")
        for row in frame.to_dict("records"):
            records.append({
                **row, "network_type": "real",
                "matrix_path": results_root / str(row["matrix_path"]),
                "node_metrics_path": (
                    results_root / "empirical" / str(int(row["year"])) /
                    str(row["representation"]) / "node_metrics.csv.gz"
                ),
            })
        years = {int(x["year"]) for x in records if x["network_type"] == "real"}
        reps = {str(x["representation"]) for x in records if x["network_type"] == "real"}
        pairs = {(int(x["year"]), str(x["representation"])) for x in records if x["network_type"] == "real"}
        expected = {(year, rep) for year in (2019, 2022) for rep in ("UU", "UD", "WD")}
        if years != {2019, 2022} or reps != {"UU", "UD", "WD"} or pairs != expected:
            raise ValueError(f"Empirical manifest must contain six year/representation pairs; found {sorted(pairs)}.")

    if scope in ("synthetic", "all"):
        manifest = results_root / "synthetic" / "synthetic_network_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing synthetic manifest: {manifest}")
        frame = pd.read_csv(manifest)
        required = {"network_id", "family", "configuration", "realization",
                    "representation", "matrix_path", "node_metrics_path"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Synthetic manifest missing columns: {sorted(required-set(frame.columns))}")
        if synthetic_realizations > 0:
            frame = frame[pd.to_numeric(frame["realization"], errors="raise") <= synthetic_realizations]
        for row in frame.to_dict("records"):
            records.append({
                **row, "network_type": "synthetic",
                "matrix_path": results_root / "synthetic" / str(row["matrix_path"]),
                "node_metrics_path": results_root / "synthetic" / str(row["node_metrics_path"]),
            })
        if frame.empty:
            raise ValueError("Synthetic manifest selection is empty.")
        skeletons = {}
        for row in frame.to_dict("records"):
            key = (str(row["family"]), str(row["configuration"]), str(row["realization"]))
            skeletons.setdefault(key, set()).add(str(row["representation"]))
        incomplete = [key for key, reps in skeletons.items() if reps != {"UU", "UD", "WD"}]
        if incomplete:
            raise ValueError(f"Synthetic selection contains incomplete paired skeletons: {incomplete[:5]}")
        expected_configs = {(family, config) for family in ("ER", "BA", "WS") for config in ("1", "2", "3")}
        observed_configs = {(str(row["family"]), str(row["configuration"])) for row in frame.to_dict("records")}
        if observed_configs != expected_configs:
            raise ValueError(f"Synthetic manifest must contain all nine ER/BA/WS configurations; missing={sorted(expected_configs-observed_configs)}.")
        realization_sets = {}
        for family, config, realization in skeletons:
            realization_sets.setdefault((family, config), set()).add(str(realization))
        sets = [frozenset(values) for values in realization_sets.values()]
        if len(sets) != 9 or len(set(sets)) != 1:
            raise ValueError("All nine synthetic configurations must have identical realization IDs.")
        observed_ids = next(iter(sets))
        expected_ids = (
            frozenset(str(i) for i in range(1, synthetic_realizations+1))
            if synthetic_realizations > 0
            else frozenset(str(i) for i in range(1, len(observed_ids)+1))
        )
        if observed_ids != expected_ids:
            raise ValueError(f"Synthetic realization IDs must be {sorted(expected_ids)}; found {sorted(observed_ids)}.")
    if not records:
        raise ValueError("No network matrices selected.")
    return records


def _validate_paired_support(records):
    """Recheck the paired UU/UD/WD construction before any shock simulations."""
    groups = {}
    for record in records:
        if record["network_type"] == "real":
            key = ("real", int(record["year"]))
        else:
            key = ("synthetic", str(record["family"]), str(record["configuration"]),
                   str(record["realization"]))
        groups.setdefault(key, {})[str(record["representation"]).upper()] = record
    for key, triplet in groups.items():
        if set(triplet) != {"UU", "UD", "WD"}:
            raise ValueError(f"{key}: requires exactly one UU, UD and WD matrix.")
        uu = (load_npz(triplet["UU"]["matrix_path"]) != 0).astype(np.int8).tocsr()
        ud = (load_npz(triplet["UD"]["matrix_path"]) != 0).astype(np.int8).tocsr()
        wd = (load_npz(triplet["WD"]["matrix_path"]) != 0).astype(np.int8).tocsr()
        if uu.shape != ud.shape or ud.shape != wd.shape:
            raise ValueError(f"{key}: paired matrices have different dimensions.")
        difference = ud - wd
        difference.eliminate_zeros()
        if difference.nnz:
            raise ValueError(f"{key}: WD and UD must have identical directed support.")
        expected_uu = ud.maximum(ud.T).tocsr()
        difference = uu - expected_uu
        difference.eliminate_zeros()
        if difference.nnz:
            raise ValueError(f"{key}: UU must equal the undirected union of UD and UD.T.")


def _scenario_parameters(name, representation, beta_by_rep, beta_c):
    if name == "absolute":
        return float(beta_by_rep[representation]), None
    if not name.startswith("eta_"):
        raise ValueError(f"Unsupported scenario: {name}")
    try:
        eta = float(name[4:])
    except ValueError as exc:
        raise ValueError(f"Invalid eta scenario: {name}") from exc
    if not any(np.isclose(eta, allowed) for allowed in ETA_LEVELS):
        raise ValueError(f"eta must be one of {ETA_LEVELS}; got {eta}.")
    return (float(eta*beta_c) if np.isfinite(beta_c) else 0.0), eta


def run(args=None):
    parser = argparse.ArgumentParser(
        description="Run exhaustive node-level and centrality-targeted Model A SIS experiments."
    )
    parser.add_argument("--results-dir", default="results_network_analysis",
                        help="Existing structural-analysis outputs (read-only).")
    parser.add_argument("--out", default="results_network_analysis/dynamics_node_attribution",
                        help="Dedicated output folder separate from structural-analysis folders.")
    parser.add_argument("--scope", choices=("real", "synthetic", "all"), default="real")
    parser.add_argument("--synthetic-realizations", type=int, default=1,
                        help="Per configuration; 1 is a pilot, 0 selects every realization.")
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--beta-uu", type=float, required=True)
    parser.add_argument("--beta-ud", type=float, required=True)
    parser.add_argument("--beta-wd", type=float, required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--initial-infection", type=float, default=1.0)
    parser.add_argument("--target-fraction", type=float, default=0.01)
    parser.add_argument("--random-cohorts", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--census-batch-size", type=int, default=64)
    parser.add_argument("--store-trajectories", action="store_true",
                        help="Store every single-source trajectory (large output).")
    parser.add_argument("--seed", type=int, default=20261030)
    ns = parser.parse_args(args)

    if ns.synthetic_realizations < 0:
        parser.error("--synthetic-realizations must be >=0 (0 means all).")
    if not 0 < ns.rho <= 1:
        parser.error("--rho must be in (0,1].")
    if min(ns.beta_uu, ns.beta_ud, ns.beta_wd) < 0:
        parser.error("Absolute beta values must be nonnegative.")
    if not 0 < ns.initial_infection <= 1:
        parser.error("--initial-infection must be in (0,1].")
    if not 0 < ns.target_fraction <= 0.5:
        parser.error("--target-fraction must be in (0,0.5].")
    if ns.random_cohorts < 1 or ns.max_steps < 1 or ns.tolerance <= 0 or ns.census_batch_size < 1:
        parser.error("random cohorts, max steps, tolerance and batch size must be positive.")

    results_root = Path(ns.results_dir).resolve()
    out_root = Path(ns.out).resolve()
    protected = [results_root / name for name in (
        "empirical", "synthetic", "metadata", "comparisons", "figures",
        "threshold_review", "dynamics",
    )]
    if out_root == results_root or out_root in results_root.parents:
        parser.error("--out cannot equal results-dir or be an ancestor of results-dir.")
    if any(out_root == folder or folder in out_root.parents for folder in protected):
        parser.error("--out cannot be inside or replace structural-analysis outputs; use a separate dynamics_node_attribution folder.")

    node_index_path = results_root / "metadata" / "node_index.csv"
    if not node_index_path.is_file():
        parser.error(f"Missing exact node index: {node_index_path}")
    node_index = pd.read_csv(node_index_path)
    if not {"node_id", "node"}.issubset(node_index.columns):
        parser.error("node_index.csv must contain node_id,node.")
    node_index = node_index.sort_values("node_id").reset_index(drop=True)
    if not np.array_equal(node_index.node_id.to_numpy(dtype=int), np.arange(len(node_index))):
        parser.error("node_index.csv IDs must be unique and contiguous from 0 to N-1.")
    labels = node_index.node.astype(str).tolist()
    if len(set(labels)) != len(labels):
        parser.error("node_index.csv contains duplicate node labels.")

    try:
        records = _read_network_records(results_root, ns.scope, ns.synthetic_realizations)
        _validate_paired_support(records)
        beta_by_rep = {"UU": ns.beta_uu, "UD": ns.beta_ud, "WD": ns.beta_wd}
        scenarios = list(dict.fromkeys(ns.scenarios))
        if not scenarios:
            raise ValueError("At least one scenario is required.")
        if any(s != "absolute" and not s.startswith("eta_") for s in scenarios):
            raise ValueError("Scenarios must be 'absolute' or eta_0.5, eta_0.9, eta_1, eta_1.1, eta_1.5, eta_2.")
        out_root.mkdir(parents=True, exist_ok=True)
        index_rows, all_top_rows, run_inputs = [], [], []

        for number, record in enumerate(records, start=1):
            matrix_path = Path(record["matrix_path"])
            metrics_path = Path(record["node_metrics_path"])
            if not matrix_path.is_file():
                raise FileNotFoundError(matrix_path)
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            B = load_npz(matrix_path).tocsr().astype(float)
            n = B.shape[0]
            if B.shape != (len(labels), len(labels)):
                raise ValueError(f"{record['network_id']}: matrix dimensions do not match node_index.csv.")
            if not np.isfinite(B.data).all() or np.any(B.data < 0) or np.any(B.diagonal() != 0):
                raise ValueError(f"{record['network_id']}: matrix has invalid weights or self-loops.")
            rep = str(record["representation"]).upper()
            if rep not in ("UU", "UD", "WD"):
                raise ValueError(f"Unsupported representation: {rep}")
            if rep in ("UU", "UD") and B.nnz and not np.allclose(B.data, 1.0):
                raise ValueError(f"{record['network_id']}: {rep} must be binary.")
            if rep == "WD" and B.nnz and np.any(B.data > 1.0+1e-9):
                raise ValueError(f"{record['network_id']}: WD weights must be <=1.")

            structural_metrics = pd.read_csv(metrics_path)
            if len(structural_metrics) != n or "node_id" not in structural_metrics:
                raise ValueError(f"{record['network_id']}: node metrics do not match matrix dimensions.")
            structural_metrics = structural_metrics.sort_values("node_id").reset_index(drop=True)
            if "node" in structural_metrics and structural_metrics.node.astype(str).tolist() != labels:
                raise ValueError(f"{record['network_id']}: node-metric order differs from metadata/node_index.csv.")
            node_metrics = calculate_targeting_metrics(B, labels, structural_metrics)
            extra = tuple(c for c in ("betweenness", "k_core", "clustering") if c in node_metrics)
            centralities = DEFAULT_CENTRALITIES + extra
            lambda_max = spectral_radius(B)
            beta_c = float(ns.rho/lambda_max) if lambda_max > 0 else np.inf
            matrix_sha, metrics_sha = _sha256(matrix_path), _sha256(metrics_path)
            network_key = str(record["network_id"])
            node_out = out_root / network_key
            node_out.mkdir(parents=True, exist_ok=True)
            node_metrics.to_csv(node_out/"targeting_metrics.csv.gz", index=False, compression="gzip")

            for scenario in scenarios:
                beta, eta = _scenario_parameters(scenario, rep, beta_by_rep, beta_c)
                scenario_out = node_out / scenario
                census, trajectories = run_single_node_census(
                    B, beta, ns.rho, labels, node_metrics,
                    max_steps=ns.max_steps, tolerance=ns.tolerance,
                    initial_infection=ns.initial_infection,
                    store_trajectories=ns.store_trajectories,
                    batch_size=ns.census_batch_size,
                )
                targeted = run_targeted_cohort_experiment(
                    B, beta, ns.rho, node_metrics, fraction=ns.target_fraction,
                    repeats=ns.random_cohorts,
                    seed=ns.seed + sum(ord(ch) for ch in network_key),
                    max_steps=ns.max_steps, tolerance=ns.tolerance,
                    centralities=centralities,
                    initial_infection=ns.initial_infection,
                )
                targeted_summary = summarize_targeted_cohorts(targeted)
                correlations = centrality_impact_correlations(census)
                common = {
                    "network_id": network_key, "network_type": record["network_type"],
                    "year": record.get("year"), "family": record.get("family"),
                    "configuration": record.get("configuration"),
                    "realization": record.get("realization"),
                    "representation": rep, "scenario": scenario, "eta": eta,
                    "beta": beta, "beta_critical": beta_c, "lambda_max": lambda_max,
                    "rho": ns.rho,
                }
                for frame in (census, targeted, targeted_summary, correlations):
                    for key, value in common.items():
                        frame[key] = value
                metadata = {
                    **common, "model": "Deterministic mass-action/bilinear SIS (Model A; NIMFA-style)",
                    "equation": "d_j(t+1)=clip((1-rho)d_j+(1-d_j)beta*sum_i B_ij*d_i); e=B.T@d",
                    "orientation": "B[i,j] is supplier i -> buyer j; force on j is (B.T@d)[j].",
                    "initial_condition_census": "Exactly one source node at initial-infection; all others zero.",
                    "initial_infection": ns.initial_infection,
                    "census_node_count": n, "census_batch_size": ns.census_batch_size,
                    "target_fraction": ns.target_fraction,
                    "target_seed_count": int(np.ceil(ns.target_fraction*n)),
                    "random_cohorts": ns.random_cohorts, "centralities": list(centralities),
                    "max_steps": ns.max_steps, "tolerance": ns.tolerance,
                    "matrix_path": str(matrix_path), "matrix_sha256": matrix_sha,
                    "node_metrics_path": str(metrics_path), "node_metrics_sha256": metrics_sha,
                    "node_index_path": str(node_index_path),
                    "node_index_sha256": _sha256(node_index_path),
                    "synthetic_scope_note": (
                        "Only selected synthetic realization IDs were included."
                        if record["network_type"] == "synthetic" else None
                    ),
                }
                write_experiment_outputs(
                    scenario_out, census=census, trajectories=trajectories,
                    targeted=targeted, targeted_summary=targeted_summary,
                    correlations=correlations, metadata=_safe_json(metadata),
                )
                top = census.sort_values("cumulative_prevalence", ascending=False).head(20).copy()
                top["network_id"], top["representation"], top["scenario"] = network_key, rep, scenario
                all_top_rows.append(top)
                index_rows.append({
                    **common, "node_count": n, "single_node_sources_tested": len(census),
                    "targeted_centralities": len(centralities), "targeted_runs": len(targeted),
                    "matrix_sha256": matrix_sha,
                    "output_dir": str(scenario_out.relative_to(out_root)),
                })
                print(f"[{number}/{len(records)}] {network_key} {scenario}: census={len(census)}, targeted={len(targeted)}")

            run_inputs.append({
                "network_id": network_key, "matrix_path": str(matrix_path),
                "matrix_sha256": matrix_sha, "node_metrics_path": str(metrics_path),
                "node_metrics_sha256": metrics_sha,
            })

        pd.DataFrame(index_rows).to_csv(out_root/"shock_experiment_manifest.csv", index=False)
        if all_top_rows:
            pd.concat(all_top_rows, ignore_index=True).to_csv(
                out_root/"top_20_single_node_shocks.csv.gz", index=False, compression="gzip")
        summary = {
            "model": "One deterministic mass-action/bilinear SIS Model A",
            "scope": ns.scope, "network_count": len(records), "scenarios": scenarios,
            "rho": ns.rho, "absolute_beta_by_representation": beta_by_rep,
            "synthetic_realizations_per_configuration": ns.synthetic_realizations,
            "initial_infection": ns.initial_infection, "target_fraction": ns.target_fraction,
            "random_cohorts": ns.random_cohorts, "max_steps": ns.max_steps,
            "tolerance": ns.tolerance, "census_batch_size": ns.census_batch_size,
            "store_trajectories": ns.store_trajectories,
            "node_index_path": str(node_index_path),
            "node_index_sha256": _sha256(node_index_path), "inputs": run_inputs,
            "interpretation": (
                "Source identity enumeration and initial-cohort sampling; not stochastic SIS paths, "
                "causal effects, or calibrated monetary losses."
            ),
            "structural_analysis_files_modified": False,
        }
        (out_root/"shock_experiment_run_metadata.json").write_text(
            json.dumps(_safe_json(summary), indent=2, allow_nan=False)+"\n",
            encoding="utf-8",
        )
        print(f"Completed. Outputs: {out_root}")
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run()

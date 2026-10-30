"""Single deterministic mass-action SIS/NIMFA-style shock model."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t
from scipy.sparse import load_npz
from .config import ETA_LEVELS, REPRESENTATIONS
from .metrics import _safe_spectral_radius


def simulate_sis(B, initial_infected, beta, rho, max_steps=300,
                 tolerance=1e-8, extinction_threshold=1e-8):
    """Simulate d(t+1)=clip((1-rho)d + (1-d) beta B.T d, 0, 1)."""
    B = sparse.csr_matrix(B, dtype=float)
    if B.shape[0] != B.shape[1]:
        raise ValueError("SIS matrix B must be square.")
    beta, rho = float(beta), float(rho)
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative.")
    if rho < 0 or rho > 1 or not np.isfinite(rho):
        raise ValueError("rho must be in [0, 1].")
    d = np.asarray(initial_infected, dtype=float).copy()
    if d.shape != (B.shape[0],) or not np.isfinite(d).all():
        raise ValueError("initial_infected must be a finite vector matching B.")
    d = np.clip(d, 0.0, 1.0)
    prevalence = [float(d.mean()) if len(d) else 0.0]
    exposure = [float(np.mean(B.T @ d)) if len(d) else 0.0]
    converged, extinct, extinction_time = False, False, None
    for step in range(1, int(max_steps) + 1):
        force = np.asarray(B.T @ d).ravel()
        next_d = np.clip((1.0 - rho) * d + (1.0 - d) * beta * force, 0.0, 1.0)
        prevalence.append(float(next_d.mean()) if len(next_d) else 0.0)
        exposure.append(float(np.mean(np.asarray(B.T @ next_d).ravel())) if len(next_d) else 0.0)
        delta = float(np.max(np.abs(next_d - d))) if len(d) else 0.0
        d = next_d
        is_extinct = (float(np.max(d)) <= extinction_threshold) if len(d) else True
        if is_extinct:
            extinct = True
            extinction_time = step
            break
        if delta <= tolerance:
            converged = True
            break
    return {
        "prevalence": np.asarray(prevalence, dtype=float),
        "mean_exposure": np.asarray(exposure, dtype=float),
        "converged": bool(converged), "extinct": bool(extinct),
        "extinction_time": extinction_time,
        "iterations": len(prevalence) - 1,
        "final_state": d,
    }


def critical_beta(B, rho):
    """beta_c = rho / lambda_max(B); inf when no propagation edge exists."""
    lam = max(0.0, float(_safe_spectral_radius(sparse.csr_matrix(B, dtype=float))))
    beta_c = float(rho / lam) if lam > 0 else float("inf")
    return lam, beta_c


def build_seed_sets(n_nodes, count, seed, infection_fraction=0.01):
    """Generate independent sets of ceil(1% N) initially infected nodes."""
    n_nodes, count = int(n_nodes), int(count)
    if n_nodes < 1 or count < 1:
        raise ValueError("n_nodes and count must be positive.")
    if not 0 < float(infection_fraction) <= 1:
        raise ValueError("infection_fraction must be in (0, 1].")
    n_infected = max(1, int(np.ceil(float(infection_fraction) * n_nodes)))
    rng = np.random.default_rng(int(seed))
    return [np.sort(rng.choice(n_nodes, size=n_infected, replace=False)).astype(int)
            for _ in range(count)]


def _ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half, n


def _pad_trace(values, length):
    if len(values) >= length:
        return values[:length]
    return np.pad(values, (0, length - len(values)), mode="constant",
                  constant_values=float(values[-1]))


def _seed_outcomes(trace, network_id, scenario, seed_set_id, eta,
                   beta, beta_c, lambda_max, representation, network_type):
    p = trace["prevalence"]
    peak_idx = int(np.argmax(p))
    tail_start = max(0, int(np.floor(len(p) * .9)))
    long_run = float(np.mean(p[tail_start:]))
    return {
        "network_id": network_id, "network_type": network_type,
        "representation": representation, "scenario": scenario,
        "eta": eta, "beta": float(beta), "beta_critical": float(beta_c),
        "lambda_max": float(lambda_max), "seed_set_id": int(seed_set_id),
        "peak_prevalence": float(p[peak_idx]), "time_to_peak": peak_idx,
        "long_run_prevalence": long_run,
        "cumulative_prevalence": float(np.trapz(p, dx=1.0)),
        "extinction_time": trace["extinction_time"],
        "extinct": trace["extinct"], "converged": trace["converged"],
        "iterations": trace["iterations"], "final_prevalence": float(p[-1]),
        "max_prevalence": float(np.max(p)),
        "max_mean_exposure": float(np.max(trace["mean_exposure"])),
    }


def _scenario_list(B, representation, rho, beta_by_rep):
    lambda_max, beta_c = critical_beta(B, rho)
    scenarios = [("absolute", float(beta_by_rep[representation]), None, lambda_max, beta_c)]
    for eta in ETA_LEVELS:
        beta = float(eta * beta_c) if np.isfinite(beta_c) else 0.0
        scenarios.append((f"eta_{eta:g}", beta, float(eta), lambda_max, beta_c))
    return scenarios


def run_dynamics_analysis(network_records, out_dir, seed=20261030, rho=0.2,
                          beta_by_rep=None, max_steps=300, tolerance=1e-8,
                          real_seed_sets=20, synthetic_seed_sets=5,
                          infection_fraction=0.01):
    """Run Model A on every matrix in the selected real/synthetic manifest.

    Real initial node sets are matched across years and representations.
    Synthetic initial node sets are generated once per skeleton and reused
    across the three paired UU/UD/WD tracks.
    """
    if beta_by_rep is None or (set(REPRESENTATIONS) - set(beta_by_rep)):
        raise ValueError("Absolute beta values are required for UU, UD and WD.")
    if not 0 <= float(rho) <= 1:
        raise ValueError("rho must be in [0, 1].")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_path = out / "dynamics_trajectories.csv.gz"
    seed_rows, network_rows, seed_manifest = [], [], []
    real_seed_cache = None
    synthetic_seed_cache = {}

    with gzip.open(traj_path, "wt", encoding="utf-8", newline="") as trajectory_file:
        write_header = True
        for record in network_records:
            network_id = record["network_id"]
            representation = record["representation"]
            network_type = record["network_type"]
            matrix_path = Path(record["matrix_path"])
            if not matrix_path.is_absolute():
                matrix_path = Path(record.get("base_dir", ".")) / matrix_path
            B = load_npz(matrix_path).tocsr().astype(float)
            n = B.shape[0]
            if network_type == "real":
                if real_seed_cache is None:
                    real_seed_cache = build_seed_sets(
                        n, real_seed_sets, int(seed) + 101, infection_fraction)
                seed_sets = real_seed_cache
                seed_group = "real_common_matched_across_years_and_representations"
            else:
                skeleton_id = f"{record['family']}_{record['configuration']}_{record['realization']}"
                if skeleton_id not in synthetic_seed_cache:
                    synthetic_seed_cache[skeleton_id] = build_seed_sets(
                        n, synthetic_seed_sets,
                        int(seed) + int(record.get("graph_seed", 0)) + 9001,
                        infection_fraction
                    )
                seed_sets = synthetic_seed_cache[skeleton_id]
                seed_group = skeleton_id

            for seed_set_id, node_ids in enumerate(seed_sets):
                seed_manifest.append({
                    "network_id": network_id, "network_type": network_type,
                    "seed_group": seed_group, "seed_set_id": seed_set_id,
                    "initial_node_ids": ";".join(str(int(x)) for x in node_ids),
                })

            for scenario, beta, eta, lambda_max, beta_c in _scenario_list(
                    B, representation, rho, beta_by_rep):
                traces, outcomes = [], []
                for seed_set_id, node_ids in enumerate(seed_sets):
                    initial = np.zeros(n, dtype=float)
                    initial[node_ids] = 1.0
                    trace = simulate_sis(B, initial, beta, rho, max_steps=max_steps,
                                         tolerance=tolerance)
                    traces.append(trace)
                    outcomes.append(_seed_outcomes(
                        trace, network_id, scenario, seed_set_id, eta,
                        beta, beta_c, lambda_max, representation, network_type
                    ))

                for outcome in outcomes:
                    seed_rows.append({**record, **outcome, "rho": float(rho),
                                      "seed_group": seed_group})
                metric_names = ("peak_prevalence", "time_to_peak", "long_run_prevalence",
                                "cumulative_prevalence", "final_prevalence", "max_mean_exposure")
                aggregate = {
                    "network_id": network_id, "network_type": network_type,
                    "representation": representation, "scenario": scenario, "eta": eta,
                    "beta": float(beta), "beta_critical": float(beta_c),
                    "lambda_max": float(lambda_max), "rho": float(rho),
                    "seed_sets": len(outcomes), "seed_group": seed_group,
                    "extinction_fraction": float(np.mean([x["extinct"] for x in outcomes])),
                    "convergence_fraction": float(np.mean([x["converged"] for x in outcomes])),
                    "mean_iterations": float(np.mean([x["iterations"] for x in outcomes])),
                }
                for metric in metric_names:
                    mean, sd, low, high, _ = _ci([x[metric] for x in outcomes])
                    aggregate.update({
                        f"{metric}_mean_across_seed_sets": mean,
                        f"{metric}_sd_across_seed_sets": sd,
                        f"{metric}_ci95_seed_low": low,
                        f"{metric}_ci95_seed_high": high,
                    })
                network_rows.append({**record, **aggregate})

                length = int(max_steps) + 1
                prevalence = np.vstack([_pad_trace(x["prevalence"], length) for x in traces])
                exposure = np.vstack([_pad_trace(x["mean_exposure"], length) for x in traces])
                p_mean = prevalence.mean(axis=0)
                p_sd = prevalence.std(axis=0, ddof=1) if len(seed_sets) > 1 else np.zeros(length)
                e_mean = exposure.mean(axis=0)
                critical = (student_t.ppf(.975, len(seed_sets) - 1) / np.sqrt(len(seed_sets))
                            if len(seed_sets) > 1 else 0.0)
                frame = pd.DataFrame({
                    "network_id": network_id, "network_type": network_type,
                    "representation": representation, "scenario": scenario, "eta": eta,
                    "beta": float(beta), "beta_critical": float(beta_c), "rho": float(rho),
                    "time": np.arange(length), "prevalence_mean": p_mean,
                    "prevalence_sd_across_seed_sets": p_sd,
                    "prevalence_ci95_seed_low": np.clip(p_mean - critical * p_sd, 0, 1),
                    "prevalence_ci95_seed_high": np.clip(p_mean + critical * p_sd, 0, 1),
                    "mean_exposure": e_mean,
                })
                frame.to_csv(trajectory_file, index=False, header=write_header)
                write_header = False

    seed_results = pd.DataFrame(seed_rows)
    network_results = pd.DataFrame(network_rows)
    seed_results.to_csv(out / "dynamics_seed_results.csv.gz", index=False, compression="gzip")
    network_results.to_csv(out / "dynamics_network_results.csv.gz", index=False, compression="gzip")
    pd.DataFrame(seed_manifest).drop_duplicates(
        subset=["seed_group", "seed_set_id"]
    ).to_csv(out / "initial_seed_sets.csv.gz", index=False, compression="gzip")

    if not network_results.empty:
        metric = "peak_prevalence_mean_across_seed_sets"
        synthetic = network_results[network_results.network_type == "synthetic"]
        if not synthetic.empty:
            group_cols = ["family", "configuration", "representation", "scenario", "eta"]
            _write_ensemble_summary(synthetic, group_cols, metric,
                                    out / "synthetic_dynamics_ensemble_summary.csv")
        real = network_results[network_results.network_type == "real"]
        if not real.empty:
            real.to_csv(out / "real_dynamics_seedset_summary.csv", index=False)

    metadata = {
        "model": "Alternate Model A: deterministic mass-action/bilinear SIS (NIMFA-style)",
        "equation": "d_j(t+1)=clip_[0,1]((1-rho)*d_j(t)+(1-d_j(t))*beta*sum_i B_ij*d_i(t)); e(t)=B.T@d(t)",
        "orientation": "B[i,j] is supplier i to buyer j; force on buyer j is (B.T @ d)[j].",
        "representations": {
            "UU": "binary undirected adjacency", "UD": "binary directed supplier-to-buyer adjacency",
            "WD": "threshold-retained dependency-share matrix"
        },
        "rho": float(rho),
        "absolute_beta_by_representation": {k: float(v) for k, v in beta_by_rep.items()},
        "eta_levels": list(ETA_LEVELS), "beta_critical": "rho / lambda_max(B)",
        "infection_fraction": float(infection_fraction), "real_seed_sets": int(real_seed_sets),
        "synthetic_seed_sets_per_skeleton": int(synthetic_seed_sets),
        "max_steps": int(max_steps), "tolerance": float(tolerance),
        "uncertainty": "Seed-set uncertainty is reported within each network; synthetic confidence intervals are computed across independent graph realizations.",
        "other_dynamical_models": "None. Only this SIS equation is implemented.",
    }
    (out / "dynamics_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return network_results, seed_results


def _write_ensemble_summary(frame, group_cols, metric, output_path):
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        mean, sd, low, high, count = _ci(group[metric].to_numpy(dtype=float))
        row.update({
            "independent_synthetic_realizations": int(count),
            "mean_across_graph_realization_means": mean,
            "sd_across_graph_realization_means": sd,
            "ci95_low_across_graph_realizations": low,
            "ci95_high_across_graph_realizations": high,
            "within_network_seed_variability_is_separately_reported": True,
        })
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)

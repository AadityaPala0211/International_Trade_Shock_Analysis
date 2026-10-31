"""Batched, resumable WD-only individual-source SIS experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from .revised_structural import (
    ACTIVE_ETA_STAGE1, ACTIVE_ETA_STAGE2, ACTIVE_RHO, MATRIX_FORMAT,
    atomic_json, git_provenance, sha256_file,
)
from .metrics import _safe_spectral_radius


METRIC_COLUMNS = (
    "max_prevalence", "cumulative_prevalence", "time_to_peak",
    "max_affected_nodes", "final_affected_nodes", "long_run_prevalence",
    "extinction_status", "convergence_time",
)
STRUCTURAL_COLUMNS = (
    "degree", "in_degree", "out_degree", "weighted_in_strength",
    "weighted_out_strength", "pagerank", "betweenness", "k_core", "clustering",
)
DYNAMIC_COLUMNS = (
    "cumulative_prevalence", "max_prevalence", "max_affected_nodes",
    "long_run_prevalence",
)


def _safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def _json_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip" if str(path).endswith(".gz") else None)
    os.replace(tmp, path)


def spectral_threshold(matrix, rho=ACTIVE_RHO):
    rho = float(rho)
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1].")
    radius = float(_safe_spectral_radius(sparse.csr_matrix(matrix, dtype=float)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"WD spectral radius must be finite and positive; got {radius}.")
    beta_c = rho / radius
    if not np.isfinite(beta_c) or beta_c <= 0:
        raise ValueError("Computed beta_c is invalid.")
    return radius, beta_c


def validate_wd_matrix(matrix, network_id=None):
    B = sparse.csr_matrix(matrix, dtype=np.float64)
    B.sum_duplicates()
    B.eliminate_zeros()
    label = str(network_id or "network")
    if B.shape[0] != B.shape[1] or B.shape[0] == 0:
        raise ValueError(f"{label}: WD matrix must be nonempty and square.")
    if not np.isfinite(B.data).all() or np.any(B.data <= 0) or np.any(B.data > 1 + 1e-9):
        raise ValueError(f"{label}: WD dependency shares must be finite and in (0,1].")
    if np.any(B.diagonal() != 0):
        raise ValueError(f"{label}: WD diagonal must be zero.")
    return B


def simulate_one_step(B, state, beta, rho):
    """One Alternate Model A step; B[i,j] is supplier i -> buyer j."""
    B = sparse.csr_matrix(B, dtype=np.float64)
    d = np.asarray(state, dtype=np.float64)
    if d.ndim != 1 or d.size != B.shape[0]:
        raise ValueError("state vector must match the WD matrix.")
    exposure = np.asarray(B.T @ d).ravel()
    return np.clip((1.0 - rho) * d + (1.0 - d) * beta * exposure, 0.0, 1.0)


def simulate_single_node_batch(
    B, source_indices, beta, rho=ACTIVE_RHO, max_steps=300,
    affected_threshold=0.01, extinction_tolerance=1e-8,
    convergence_tolerance=1e-8, consecutive_steps=5, long_run_window=30,
):
    """Simulate independent one-time unit shocks, one per column of D."""
    B = validate_wd_matrix(B)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    n, batch_size = B.shape[0], len(source_indices)
    if batch_size == 0:
        return pd.DataFrame()
    if np.any(source_indices < 0) or np.any(source_indices >= n):
        raise ValueError("Source-node index outside network.")
    beta = float(beta); rho = float(rho)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1].")
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if not 0 <= affected_threshold <= 1:
        raise ValueError("affected_threshold must be in [0,1].")
    if extinction_tolerance < 0 or convergence_tolerance <= 0:
        raise ValueError("Invalid extinction or convergence tolerance.")
    if consecutive_steps < 1 or long_run_window < 1:
        raise ValueError("consecutive_steps and long_run_window must be positive.")

    D = np.zeros((n, batch_size), dtype=np.float64)
    D[source_indices, np.arange(batch_size)] = 1.0
    prevalence = np.empty((max_steps + 1, batch_size), dtype=np.float64)
    affected_counts = np.empty((max_steps + 1, batch_size), dtype=np.int32)
    prevalence[0] = D.mean(axis=0)
    affected_counts[0] = np.count_nonzero(D >= affected_threshold, axis=0)

    convergence_counts = np.zeros(batch_size, dtype=np.int32)
    extinction_counts = np.zeros(batch_size, dtype=np.int32)
    convergence_time = np.full(batch_size, -1, dtype=np.int32)
    extinction_time = np.full(batch_size, -1, dtype=np.int32)
    converged = np.zeros(batch_size, dtype=bool)
    extinct = np.zeros(batch_size, dtype=bool)

    for step in range(1, max_steps + 1):
        exposure = np.asarray(B.T @ D)
        next_D = np.clip(
            (1.0 - rho) * D + (1.0 - D) * beta * exposure,
            0.0, 1.0,
        )
        delta = np.max(np.abs(next_D - D), axis=0)
        convergence_counts = np.where(delta < convergence_tolerance, convergence_counts + 1, 0)
        extinction_now = np.max(next_D, axis=0) < extinction_tolerance
        extinction_counts = np.where(extinction_now, extinction_counts + 1, 0)
        just_converged = (convergence_counts >= consecutive_steps) & ~converged
        just_extinct = (extinction_counts >= consecutive_steps) & ~extinct
        convergence_time[just_converged] = step
        extinction_time[just_extinct] = step
        converged |= just_converged
        extinct |= just_extinct
        D = next_D
        prevalence[step] = D.mean(axis=0)
        affected_counts[step] = np.count_nonzero(D >= affected_threshold, axis=0)

    peak = prevalence.max(axis=0)
    peak_time = prevalence.argmax(axis=0)  # argmax returns earliest tied maximum
    cumulative = np.sum(0.5 * (prevalence[:-1] + prevalence[1:]), axis=0)
    window = min(int(long_run_window), max_steps + 1)
    long_run = prevalence[-window:].mean(axis=0)
    final_affected = affected_counts[-1]
    rows = pd.DataFrame({
        "source_node_index": source_indices,
        "max_prevalence": peak,
        "cumulative_prevalence": cumulative,
        "time_to_peak": peak_time,
        "max_affected_nodes": affected_counts.max(axis=0),
        "final_affected_nodes": final_affected,
        "long_run_prevalence": long_run,
        "extinction_status": extinct,
        "extinction_time": np.where(extinct, extinction_time, np.nan),
        "convergence_time": np.where(converged, convergence_time, np.nan),
        "converged": converged,
        "simulation_horizon": max_steps,
        "affected_node_threshold": affected_threshold,
        "extinction_tolerance": extinction_tolerance,
        "convergence_tolerance": convergence_tolerance,
        "consecutive_steps": consecutive_steps,
        "long_run_window": window,
        "integration_method": "trapezoidal_rule",
    })
    return rows


def _read_network_records(structural_out: Path):
    real_path = structural_out / "manifests" / "real_network_manifest.csv"
    synthetic_path = structural_out / "manifests" / "synthetic_network_manifest.csv"
    if not real_path.is_file() or not synthetic_path.is_file():
        raise FileNotFoundError("Both real and synthetic structural manifests are required.")
    records = pd.concat([pd.read_csv(real_path), pd.read_csv(synthetic_path)], ignore_index=True)
    expected_ids = {"real_2019", "real_2022"} | {
        f"syn_{family}_{configuration}"
        for family in ("ER", "BA", "WS") for configuration in (1, 2, 3)
    }
    observed_ids = set(records["network_id"].astype(str))
    if len(records) != 11 or records["network_id"].nunique() != 11 or observed_ids != expected_ids:
        raise ValueError(
            f"Expected exactly the 11 approved WD network IDs; got {len(records)} rows, "
            f"missing={sorted(expected_ids-observed_ids)}, extra={sorted(observed_ids-expected_ids)}."
        )
    synthetic = records[records["network_family"].isin(["ER", "BA", "WS"])]
    if len(synthetic) != 9 or not synthetic["realization_id"].astype(str).eq("1").all():
        raise ValueError("Exactly one realization (ID 1) per synthetic configuration is required.")
    for family in ("ER", "BA", "WS"):
        family_rows = synthetic[synthetic["network_family"] == family]
        if set(family_rows["network_configuration"].astype(str)) != {"1", "2", "3"}:
            raise ValueError(f"{family} synthetic configurations must be exactly 1, 2 and 3.")
    for record in records.to_dict("records"):
        if record["matrix_format"] != MATRIX_FORMAT:
            raise ValueError(f"{record['network_id']}: non-WD matrix in structural manifest.")
        record["matrix_path_abs"] = structural_out / record["matrix_path"]
        record["metrics_path_abs"] = structural_out / record["node_metrics_path"]
        record["index_path_abs"] = structural_out / record["node_index_path"]
        yield record


def _load_network_matrix(record):
    from scipy.sparse import load_npz
    path = Path(record["matrix_path_abs"])
    if sha256_file(path) != str(record["matrix_sha256"]):
        raise ValueError(f"{record['network_id']}: matrix checksum differs from structural manifest.")
    B = validate_wd_matrix(load_npz(path), record["network_id"])
    if B.shape[0] != int(record["node_count"]):
        raise ValueError(f"{record['network_id']}: node count differs from manifest.")
    return B


def validate_dynamics_preflight(structural_out: str | Path, rho=ACTIVE_RHO):
    structural_out = Path(structural_out)
    completion_path = structural_out / "diagnostics" / "structural_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError("All-11-network structural completeness report is required before SIS.")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "valid" or completion.get("network_count") != 11:
        raise ValueError("Structural completeness report is not valid for all 11 networks.")
    records = list(_read_network_records(structural_out))
    diagnostics = []
    for record in records:
        B = _load_network_matrix(record)
        if B.shape[0] != 4050:
            raise ValueError(f"{record['network_id']}: expected full 4050-node resolution, got {B.shape[0]}.")
        radius, beta_c = spectral_threshold(B, rho)
        diagnostics.append({
            "network_id": record["network_id"],
            "network_family": record["network_family"],
            "network_configuration": record["network_configuration"],
            "year": record.get("year"),
            "realization_id": record.get("realization_id"),
            "N": B.shape[0], "edges": B.nnz, "density": B.nnz/(B.shape[0]*(B.shape[0]-1)),
            "rho": float(rho), "spectral_radius": radius, "beta_c": beta_c,
            "matrix_format": MATRIX_FORMAT,
            "matrix_sha256": record["matrix_sha256"],
        })
    if len(diagnostics) != 11:
        raise ValueError("Preflight did not validate exactly 11 WD matrices.")
    return records, pd.DataFrame(diagnostics)


def _load_node_map(record):
    index = pd.read_csv(record["index_path_abs"])
    index["node_index"] = index["node_index"].astype(int)
    if index["node_index"].duplicated().any() or len(index) != int(record["node_count"]):
        raise ValueError(f"{record['network_id']}: invalid node index mapping.")
    if index["node_id"].astype(str).duplicated().any():
        raise ValueError(f"{record['network_id']}: duplicate stable node IDs in node index.")
    if not np.array_equal(np.sort(index["node_index"].to_numpy(dtype=int)), np.arange(int(record["node_count"]))):
        raise ValueError(f"{record['network_id']}: node index must be a complete 0..N-1 mapping.")
    return index.sort_values("node_index").reset_index(drop=True)


def _run_resumable_experiments(
    B, record, node_map, eta, beta, rho, output_dir, stage_name,
    source_indices=None, batch_size=128, max_steps=300,
    affected_threshold=0.01, extinction_tolerance=1e-8,
    convergence_tolerance=1e-8, consecutive_steps=5, long_run_window=30,
):
    output_dir = Path(output_dir)
    network_id = str(record["network_id"])
    N = B.shape[0]
    sources = np.arange(N, dtype=np.int64) if source_indices is None else np.asarray(source_indices, dtype=np.int64)
    if len(np.unique(sources)) != len(sources):
        raise ValueError(f"{network_id}: source list contains duplicates.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    config = {
        "stage": stage_name, "network_id": network_id, "matrix_sha256": record["matrix_sha256"],
        "node_count": N, "source_indices_sha256": hashlib.sha256(sources.tobytes()).hexdigest(),
        "eta": float(eta), "beta": float(beta), "rho": float(rho),
        "batch_size": int(batch_size), "max_steps": int(max_steps),
        "affected_threshold": float(affected_threshold),
        "extinction_tolerance": float(extinction_tolerance),
        "convergence_tolerance": float(convergence_tolerance),
        "consecutive_steps": int(consecutive_steps), "long_run_window": int(long_run_window),
        "matrix_format": MATRIX_FORMAT,
        "git": git_provenance(),
    }
    fingerprint = _json_fingerprint(config)
    state_path = output_dir / "checkpoint.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError(f"{network_id}/{eta}: checkpoint settings/input changed; choose a fresh output directory.")
    else:
        state = {"fingerprint": fingerprint, "configuration": config, "completed_batches": []}
        atomic_json(state_path, state)

    completed = {item["batch_id"]: item for item in state["completed_batches"]}
    parts_dir = output_dir / "batches"
    parts_dir.mkdir(parents=True, exist_ok=True)
    all_ranges = []
    for start in range(0, len(sources), batch_size):
        stop = min(start + batch_size, len(sources))
        all_ranges.append((start, stop))
        batch_id = f"{start:06d}_{stop:06d}"
        part_path = parts_dir / f"batch_{batch_id}.csv.gz"
        if batch_id in completed:
            if not part_path.is_file() or sha256_file(part_path) != completed[batch_id]["sha256"]:
                raise ValueError(f"{network_id}/{eta}: completed batch {batch_id} is missing or corrupted.")
            continue
        source_batch = sources[start:stop]
        if part_path.exists():
            # Recover a crash after atomic part write but before checkpoint update.
            orphan = pd.read_csv(part_path)
            expected_source_ids = [str(node_map.iloc[int(i)]["node_id"]) for i in source_batch]
            valid_orphan = (
                len(orphan) == len(source_batch)
                and orphan["source_node_id"].astype(str).tolist() == expected_source_ids
                and orphan["network_id"].astype(str).eq(network_id).all()
                and np.allclose(orphan["eta"].to_numpy(dtype=float), float(eta))
                and not orphan["source_node_id"].astype(str).duplicated().any()
            )
            if not valid_orphan:
                raise ValueError(f"{network_id}/{eta}: orphan batch {batch_id} failed validation; not adopting it.")
            state["completed_batches"].append({
                "batch_id": batch_id, "source_count": len(source_batch),
                "sha256": sha256_file(part_path), "recovered_after_atomic_write": True,
            })
            atomic_json(state_path, state)
            continue
        results = simulate_single_node_batch(
            B, source_batch, beta, rho, max_steps,
            affected_threshold, extinction_tolerance,
            convergence_tolerance, consecutive_steps, long_run_window,
        )
        results.insert(0, "network_id", network_id)
        results.insert(1, "network_family", record["network_family"])
        results.insert(2, "network_configuration", record["network_configuration"])
        results.insert(3, "year", record.get("year"))
        results.insert(4, "realization_id", record.get("realization_id"))
        results.insert(5, "N", N)
        results.insert(6, "rho", rho)
        results.insert(7, "spectral_radius", float(record["spectral_radius"]))
        results.insert(8, "beta_c", float(record["beta_c"]))
        results.insert(9, "eta", float(eta))
        results.insert(10, "beta", float(beta))
        results["source_node_id"] = [node_map.iloc[int(i)]["node_id"] for i in results["source_node_index"]]
        results["source_node_label"] = [node_map.iloc[int(i)]["node_label"] for i in results["source_node_index"]]
        results["node_index"] = results["source_node_index"]
        _atomic_csv(results, part_path)
        state["completed_batches"].append({
            "batch_id": batch_id, "source_count": len(source_batch), "sha256": sha256_file(part_path),
        })
        atomic_json(state_path, state)

    ledger_ids = [item["batch_id"] for item in state["completed_batches"]]
    if len(ledger_ids) != len(set(ledger_ids)):
        raise RuntimeError("Checkpoint contains duplicate completed-batch entries.")
    for start, stop in all_ranges:
        batch_id = f"{start:06d}_{stop:06d}"
        if batch_id not in set(ledger_ids):
            raise RuntimeError(f"Missing completed batch {batch_id}.")
    paths = [parts_dir / f"batch_{start:06d}_{stop:06d}.csv.gz" for start, stop in all_ranges]
    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected_ids = [str(node_map.iloc[int(i)]["node_id"]) for i in sources]
    if result["source_node_id"].astype(str).duplicated().any():
        raise ValueError(f"{network_id}/{eta}: duplicate source-node records in completed results.")
    if set(result["source_node_id"].astype(str)) != set(expected_ids):
        raise ValueError(f"{network_id}/{eta}: missing or extra source-node records.")
    _atomic_csv(result, output_dir / "source_node_metrics.csv.gz")
    return result


def _describe_metrics(frame, keys):
    rows = []
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        numeric = values.dropna().to_numpy(dtype=float)
        row = dict(zip(keys, [frame.iloc[0][key] for key in keys]))
        row.update({"metric": metric, "count": int(len(numeric))})
        if len(numeric):
            row.update({
                "mean": float(np.mean(numeric)), "median": float(np.median(numeric)),
                "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
                "min": float(np.min(numeric)), "max": float(np.max(numeric)),
                "q10": float(np.quantile(numeric, .1)), "q25": float(np.quantile(numeric, .25)),
                "q75": float(np.quantile(numeric, .75)), "q90": float(np.quantile(numeric, .9)),
            })
        else:
            row.update({k: np.nan for k in ("mean", "median", "std", "min", "max", "q10", "q25", "q75", "q90")})
        rows.append(row)
    return rows


def _write_rankings(frame, folder: Path, key_columns):
    folder.mkdir(parents=True, exist_ok=True)
    rank_metrics = (
        "cumulative_prevalence", "max_prevalence",
        "max_affected_nodes", "long_run_prevalence",
    )
    for metric in rank_metrics:
        ranked = frame.sort_values([metric, "source_node_id"], ascending=[False, True], kind="stable")
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.to_csv(folder / f"ranking_by_{metric}.csv", index=False)



def _plot_census_distributions(frame: pd.DataFrame, census_out: Path):
    """Create per-network/eta distributions for the four main disruption metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = census_out / "figures" / "metric_distributions"
    target.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("max_prevalence", "Maximum prevalence"),
        ("cumulative_prevalence", "Cumulative prevalence"),
        ("max_affected_nodes", "Maximum affected nodes"),
        ("long_run_prevalence", "Long-run prevalence"),
    )
    for (network_id, eta), group in frame.groupby(["network_id", "eta"], sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for axis, (column, label) in zip(axes.ravel(), metrics):
            axis.hist(pd.to_numeric(group[column], errors="coerce").dropna(), bins=40)
            axis.set_title(label)
            axis.set_xlabel(label)
            axis.set_ylabel("Source-node experiments")
        fig.suptitle(f"{network_id} | eta={eta}: single-node shock impact")
        fig.tight_layout()
        fig.savefig(target / f"{network_id}_eta_{_safe_name(eta)}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def _plot_spearman_heatmaps(correlation_frame: pd.DataFrame, target: Path):
    """Plot within-network structural-to-dynamic Spearman coefficient matrices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target.mkdir(parents=True, exist_ok=True)
    for (network_id, eta), group in correlation_frame.groupby(["network_id", "eta"], sort=True):
        pivot = group.pivot(index="structural_metric", columns="dynamic_metric", values="spearman_rho")
        if pivot.empty:
            continue
        fig, axis = plt.subplots(figsize=(8, max(4, 0.38 * len(pivot))))
        image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        axis.set_title(f"{network_id} | eta={eta}: Spearman rank correlation")
        fig.colorbar(image, ax=axis, label="Spearman rho")
        fig.tight_layout()
        fig.savefig(target / f"{network_id}_eta_{_safe_name(eta)}_spearman.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def _plot_targeted_comparison(frame: pd.DataFrame, targeted_out: Path):
    """Plot targeted-vs-census outcome agreement for the two shared eta levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = targeted_out / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for eta in ACTIVE_ETA_STAGE1:
        subset = frame[np.isclose(frame["eta_targeted"], eta)]
        if subset.empty:
            continue
        fig, axis = plt.subplots(figsize=(6.5, 6))
        axis.scatter(
            subset["cumulative_prevalence_census"],
            subset["cumulative_prevalence_targeted"],
            alpha=0.6,
        )
        maximum = float(max(
            subset["cumulative_prevalence_census"].max(),
            subset["cumulative_prevalence_targeted"].max(),
        ))
        axis.plot([0, maximum], [0, maximum], linestyle="--")
        axis.set_xlabel("Stage 1 cumulative prevalence")
        axis.set_ylabel("Stage 2 cumulative prevalence")
        axis.set_title(f"Targeted vs census individual shocks | eta={eta}")
        fig.tight_layout()
        fig.savefig(target / f"targeted_vs_census_eta_{_safe_name(eta)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_single_node_census(
    structural_out: str | Path, census_out: str | Path,
    rho=ACTIVE_RHO, eta_levels=ACTIVE_ETA_STAGE1, batch_size=128,
    max_steps=300, affected_threshold=0.01, extinction_tolerance=1e-8,
    convergence_tolerance=1e-8, consecutive_steps=5, long_run_window=30,
):
    structural_out, census_out = Path(structural_out), Path(census_out)
    records, spectral = validate_dynamics_preflight(structural_out, rho)
    (census_out / "config").mkdir(parents=True, exist_ok=True)
    if tuple(float(x) for x in eta_levels) != ACTIVE_ETA_STAGE1:
        raise ValueError(f"Stage 1 eta must be exactly {ACTIVE_ETA_STAGE1}.")
    params = {
        "run_id": "census_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ"),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": "Alternate Model A: deterministic mass-action bilinear SIS",
        "matrix_format": MATRIX_FORMAT, "rho": float(rho), "eta_levels": list(eta_levels),
        "beta_definition": "beta = eta * rho / spectral_radius(B)",
        "max_steps": int(max_steps), "affected_node_threshold": float(affected_threshold),
        "extinction_tolerance": float(extinction_tolerance),
        "convergence_tolerance": float(convergence_tolerance),
        "consecutive_steps": int(consecutive_steps), "long_run_window": int(long_run_window),
        "integration_method": "trapezoidal_rule", "batch_size": int(batch_size),
        "pulse_shock": "exactly one source starts at disruption 1; source then recovers normally",
        "expected_network_count": 11, "stage1_eta": [0.9, 1.1],
        "git": git_provenance(),
    }
    atomic_json(census_out / "config" / "census_config.json", params)
    spectral.to_csv(census_out / "config" / "network_thresholds.csv", index=False)
    all_rows, diagnostics = [], []
    expected_experiments = 0
    for record in records:
        B = _load_network_matrix(record)
        node_map = _load_node_map(record)
        expected_experiments += B.shape[0] * len(eta_levels)
        record["spectral_radius"] = float(spectral.loc[spectral.network_id == record["network_id"], "spectral_radius"].iloc[0])
        record["beta_c"] = float(spectral.loc[spectral.network_id == record["network_id"], "beta_c"].iloc[0])
        network_rows = []
        for eta in eta_levels:
            beta = float(eta * record["beta_c"])
            eta_dir = census_out / "raw_metrics" / str(record["network_id"]) / f"eta_{_safe_name(eta)}"
            frame = _run_resumable_experiments(
                B, record, node_map, eta, beta, rho, eta_dir, "census",
                batch_size=batch_size, max_steps=max_steps,
                affected_threshold=affected_threshold,
                extinction_tolerance=extinction_tolerance,
                convergence_tolerance=convergence_tolerance,
                consecutive_steps=consecutive_steps, long_run_window=long_run_window,
            )
            network_rows.append(frame)
            summary_rows = _describe_metrics(frame, ["network_id", "network_family", "network_configuration", "year", "eta"])
            summary_path = (
                census_out / "network_summaries" / str(record["network_id"]) /
                f"eta_{_safe_name(eta)}_descriptive_statistics.csv"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            _write_rankings(
                frame, census_out / "rankings" / str(record["network_id"]) / f"eta_{_safe_name(eta)}",
                ["network_id", "eta"],
            )
        combined = pd.concat(network_rows, ignore_index=True)
        all_rows.append(combined)
        diagnostics.append({
            "network_id": record["network_id"], "expected_source_nodes": B.shape[0],
            "eta_count": len(eta_levels), "expected_experiments": B.shape[0] * len(eta_levels),
            "completed_experiments": len(combined),
            "duplicate_experiments": int(combined.duplicated(["source_node_id", "eta"]).sum()),
            "missing_source_eta": int(B.shape[0] * len(eta_levels) - len(combined)),
            "failed_experiments": 0,
            "status": "complete" if len(combined) == B.shape[0] * len(eta_levels) and not combined.duplicated(["source_node_id", "eta"]).any() else "incomplete",
        })
    result = pd.concat(all_rows, ignore_index=True)
    _plot_census_distributions(result, census_out)
    key = ["network_id", "source_node_id", "eta"]
    duplicates = int(result.duplicated(key).sum())
    completed = int(len(result))
    missing = int(expected_experiments - completed)
    report = {
        "status": "complete" if completed == expected_experiments and missing == 0 and duplicates == 0 else "incomplete",
        "network_count_expected": 11, "network_count_processed": int(result["network_id"].nunique()),
        "node_count_by_network": {str(r["network_id"]): int(r["node_count"]) for r in records},
        "eta_levels": list(eta_levels),
        "experiments_expected": int(expected_experiments),
        "experiments_completed": completed,
        "experiments_missing": missing, "duplicate_records": duplicates,
        "failed_experiments": 0,
        "completion_by_eta": {
            str(eta): int(result.loc[result.eta == eta].shape[0]) for eta in eta_levels
        },
        "converged_experiments": int(result["converged"].sum()),
        "extinct_experiments": int(result["extinction_status"].sum()),
        "parameter_settings": params,
        "output": str(census_out),
        "git": git_provenance(),
    }
    _atomic_csv(result, census_out / "raw_metrics" / "single_node_census_metrics.csv.gz")
    (census_out / "diagnostics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(census_out / "diagnostics" / "network_completion.csv", index=False)
    atomic_json(census_out / "diagnostics" / "census_validation.json", report)
    if report["status"] == "complete":
        family_summary = result.groupby(["network_family", "eta"], dropna=False)[
            list(METRIC_COLUMNS)
        ].agg(["mean", "median", "std", "min", "max"]).reset_index()
        family_summary.to_csv(census_out / "network_summaries" / "cross_network_family_summary.csv", index=False)
        configuration_summary = result.groupby(
            ["network_family", "network_configuration", "eta"], dropna=False
        )[list(METRIC_COLUMNS)].agg(["mean", "median", "std", "min", "max"]).reset_index()
        configuration_summary.to_csv(
            census_out / "network_summaries" / "cross_network_configuration_summary.csv", index=False
        )
    return report


def validate_census(census_out: str | Path, structural_out: str | Path):
    census_out = Path(census_out)
    records = list(_read_network_records(Path(structural_out)))
    result_path = census_out / "raw_metrics" / "single_node_census_metrics.csv.gz"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    frame = pd.read_csv(result_path)
    expected = sum(int(record["node_count"]) for record in records) * len(ACTIVE_ETA_STAGE1)
    duplicates = int(frame.duplicated(["network_id", "source_node_id", "eta"]).sum())
    missing = expected - len(frame)
    observed = set(frame["network_id"].astype(str))
    expected_ids = {str(record["network_id"]) for record in records}
    errors = []
    if observed != expected_ids: errors.append("network ID set mismatch")
    if set(frame["eta"].unique()) != set(ACTIVE_ETA_STAGE1): errors.append("eta set mismatch")
    if duplicates: errors.append(f"{duplicates} duplicate source/eta records")
    if missing: errors.append(f"{missing} missing records")
    for record in records:
        part = frame[frame.network_id == record["network_id"]]
        count = int(record["node_count"])
        for eta in ACTIVE_ETA_STAGE1:
            sub = part[part.eta == eta]
            if len(sub) != count or sub.source_node_id.astype(str).nunique() != count:
                errors.append(f"{record['network_id']}, eta={eta}: expected {count} unique nodes, got {len(sub)}")
    report = {
        "status": "valid" if not errors else "incomplete",
        "expected_network_count": 11, "observed_network_count": int(frame.network_id.nunique()),
        "expected_experiments": expected, "completed_experiments": int(len(frame)),
        "missing_records": int(missing), "duplicate_records": duplicates, "errors": errors,
    }
    atomic_json(census_out / "diagnostics" / "census_validation.json", report)
    if errors:
        raise ValueError("Census completeness validation failed: " + "; ".join(errors[:10]))
    return report


def structural_dynamic_comparison(structural_out: str | Path, census_out: str | Path):
    structural_out, census_out = Path(structural_out), Path(census_out)
    census_validation = json.loads((census_out / "diagnostics" / "census_validation.json").read_text())
    if census_validation.get("status") != "valid":
        raise ValueError("Structural–dynamic comparison requires a complete validated census.")
    structural_frames = []
    for record in _read_network_records(structural_out):
        node = pd.read_csv(record["metrics_path_abs"])
        structural_frames.append(node)
    structural = pd.concat(structural_frames, ignore_index=True)
    dynamic = pd.read_csv(census_out / "raw_metrics" / "single_node_census_metrics.csv.gz")
    # Join only on stable network and node identifiers, never on row ordering.
    joined = dynamic.merge(
        structural, left_on=["network_id", "source_node_id"], right_on=["network_id", "node_id"],
        how="left", validate="many_to_one", suffixes=("", "_structural"),
    )
    if joined["node_index"].isna().any():
        raise ValueError("Some dynamics sources could not be joined to their structural node rows.")
    corr_rows = []
    for (network_id, eta), group in joined.groupby(["network_id", "eta"]):
        for structural_metric in STRUCTURAL_COLUMNS:
            if structural_metric not in group:
                continue
            for dynamic_metric in DYNAMIC_COLUMNS:
                rho, pvalue = spearmanr(group[structural_metric], group[dynamic_metric], nan_policy="omit")
                corr_rows.append({
                    "network_id": network_id, "network_family": group.iloc[0]["network_family"],
                    "network_configuration": group.iloc[0]["network_configuration"],
                    "year": group.iloc[0]["year"], "eta": eta,
                    "structural_metric": structural_metric, "dynamic_metric": dynamic_metric,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                    "node_count": int(group[[structural_metric, dynamic_metric]].dropna().shape[0]),
                })
    target = census_out / "structural_dynamic_comparisons"
    target.mkdir(parents=True, exist_ok=True)
    _atomic_csv(joined, target / "node_level_structural_dynamic_join.csv.gz")
    correlation_frame = pd.DataFrame(corr_rows)
    correlation_frame.to_csv(target / "spearman_rank_correlations.csv", index=False)
    _plot_spearman_heatmaps(correlation_frame, target / "figures")
    # Compare the two empirical years only for common real country-industry labels.
    real = joined[joined.network_family == "OECD_ICIO"]
    year_rows = []
    for eta in ACTIVE_ETA_STAGE1:
        a = real[(real.year == 2019) & np.isclose(real.eta, eta)].set_index("source_node_id")
        b = real[(real.year == 2022) & np.isclose(real.eta, eta)].set_index("source_node_id")
        if len(a) and len(b):
            common = a.index.intersection(b.index)
            for metric in (*STRUCTURAL_COLUMNS, *DYNAMIC_COLUMNS):
                if metric in a and metric in b:
                    rho, pvalue = spearmanr(a.loc[common, metric], b.loc[common, metric], nan_policy="omit")
                    year_rows.append({
                        "eta": eta, "metric": metric, "common_nodes": len(common),
                        "spearman_2019_2022": float(rho) if np.isfinite(rho) else np.nan,
                        "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                    })
    year_frame = pd.DataFrame(year_rows)
    year_frame.to_csv(target / "real_2019_2022_rank_stability.csv", index=False)
    if not year_frame.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(year_frame["metric"], year_frame["spearman_2019_2022"])
        axis.set_ylim(-1, 1)
        axis.set_ylabel("Spearman rho")
        axis.set_title("2019 vs 2022 real-network rank stability")
        axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(target / "figures" / "real_2019_2022_rank_stability.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    return {"joined_rows": len(joined), "correlation_rows": len(corr_rows), "real_year_comparisons": len(year_rows)}


def select_targeted_candidates(structural_out: str | Path, census_out: str | Path,
                               high_percentile=0.90, low_percentile=0.10):
    if not 0 <= low_percentile < high_percentile <= 1:
        raise ValueError("Candidate percentile cutoffs must satisfy 0 <= low < high <= 1.")
    structural_out, census_out = Path(structural_out), Path(census_out)
    validation = json.loads((census_out / "diagnostics" / "census_validation.json").read_text())
    if validation.get("status") != "valid":
        raise ValueError("Candidate selection requires a complete Stage 1 census.")
    joined = pd.read_csv(census_out / "structural_dynamic_comparisons" / "node_level_structural_dynamic_join.csv.gz")
    primary = joined[np.isclose(joined["eta"], 1.1)].copy()
    structural_metrics = [m for m in STRUCTURAL_COLUMNS if m in primary]
    if not structural_metrics:
        raise ValueError("No approved structural metrics are available for candidate selection.")
    dynamic_primary = "cumulative_prevalence"
    for metric in structural_metrics:
        primary[f"{metric}_percentile"] = primary.groupby("network_id")[metric].rank(pct=True, method="average")
    for metric in DYNAMIC_COLUMNS:
        primary[f"{metric}_percentile"] = primary.groupby("network_id")[metric].rank(pct=True, method="average")
    primary["structural_percentile"] = primary[[f"{m}_percentile" for m in structural_metrics]].mean(axis=1)
    primary["dynamic_percentile"] = primary[f"{dynamic_primary}_percentile"]
    candidates = {}
    def add(idx, category, reason):
        key = (str(primary.loc[idx, "network_id"]), str(primary.loc[idx, "source_node_id"]))
        if key not in candidates:
            candidates[key] = primary.loc[idx].to_dict()
            candidates[key]["candidate_categories"] = []
            candidates[key]["selection_reasons"] = []
        if category not in candidates[key]["candidate_categories"]:
            candidates[key]["candidate_categories"].append(category)
        if reason not in candidates[key]["selection_reasons"]:
            candidates[key]["selection_reasons"].append(reason)

    for metric in structural_metrics:
        pcol = f"{metric}_percentile"
        for idx in primary.index[primary[pcol] >= high_percentile]:
            add(idx, "A_structurally_important", f"{metric} percentile >= {high_percentile:.2f}")
    for dynamic_metric in DYNAMIC_COLUMNS:
        dynamic_pct = f"{dynamic_metric}_percentile"
        for idx in primary.index[primary[dynamic_pct] >= high_percentile]:
            add(
                idx, "B_dynamically_important",
                f"{dynamic_metric} percentile >= {high_percentile:.2f} at eta=1.1",
            )
    c1 = (primary.structural_percentile >= high_percentile) & (primary.dynamic_percentile <= low_percentile)
    for idx in primary.index[c1]:
        add(idx, "C1_structural_high_dynamic_low",
            f"structural composite >= {high_percentile:.2f}, dynamic percentile <= {low_percentile:.2f}")
    c2 = (primary.structural_percentile <= low_percentile) & (primary.dynamic_percentile >= high_percentile)
    for idx in primary.index[c2]:
        add(idx, "C2_structural_low_dynamic_high",
            f"structural composite <= {low_percentile:.2f}, dynamic percentile >= {high_percentile:.2f}")

    candidate_frame = pd.DataFrame([
        {**row,
         "candidate_categories": ";".join(row["candidate_categories"]),
         "selection_reasons": "; ".join(row["selection_reasons"])}
        for row in candidates.values()
    ])
    output = census_out / "targeted_candidates" / "targeted_candidate_nodes.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(output, index=False)
    report = {
        "status": "valid",
        "primary_eta": 1.1,
        "high_percentile": high_percentile, "low_percentile": low_percentile,
        "candidate_count": int(len(candidate_frame)),
        "category_counts": {
            name: int(candidate_frame.candidate_categories.str.contains(name, regex=False).sum())
            for name in ("A_structurally_important", "B_dynamically_important",
                         "C1_structural_high_dynamic_low", "C2_structural_low_dynamic_high")
        },
        "candidate_path": str(output),
        "rule": "within-network percentile ranks; structural composite is mean of available structural metric percentiles; dynamic primary is cumulative prevalence at eta=1.1",
    }
    atomic_json(census_out / "targeted_candidates" / "candidate_selection_manifest.json", report)
    return report


def run_targeted_individual_sis(
    structural_out: str | Path, census_out: str | Path, targeted_out: str | Path,
    rho=ACTIVE_RHO, eta_levels=ACTIVE_ETA_STAGE2, batch_size=128, max_steps=300,
    affected_threshold=0.01, extinction_tolerance=1e-8,
    convergence_tolerance=1e-8, consecutive_steps=5, long_run_window=30,
):
    structural_out, census_out, targeted_out = Path(structural_out), Path(census_out), Path(targeted_out)
    validate_dynamics_preflight(structural_out, rho)
    if tuple(float(x) for x in eta_levels) != ACTIVE_ETA_STAGE2:
        raise ValueError(f"Stage 2 eta must be exactly {ACTIVE_ETA_STAGE2}.")
    census_validation = validate_census(census_out, structural_out)
    if census_validation.get("status") != "valid":
        raise ValueError("Stage 2 requires a validated Stage 1 census.")
    selection = json.loads((census_out / "targeted_candidates" / "candidate_selection_manifest.json").read_text())
    if selection.get("status") != "valid":
        raise ValueError("Targeted candidate-selection output is not valid.")
    candidates = pd.read_csv(census_out / "targeted_candidates" / "targeted_candidate_nodes.csv")
    all_records, diagnostics = [], []
    for record in _read_network_records(structural_out):
        selected = candidates[candidates.network_id == record["network_id"]]
        if selected.empty:
            continue
        B = _load_network_matrix(record)
        node_map = _load_node_map(record)
        index_of = {str(row.node_id): int(row.node_index) for row in node_map.itertuples()}
        source_indices = np.asarray([index_of[str(x)] for x in selected.source_node_id], dtype=np.int64)
        spectral, beta_c = spectral_threshold(B, rho)
        record["spectral_radius"] = spectral
        record["beta_c"] = beta_c
        rows = []
        for eta in eta_levels:
            beta = float(eta * beta_c)
            eta_dir = targeted_out / "raw_metrics" / str(record["network_id"]) / f"eta_{_safe_name(eta)}"
            frame = _run_resumable_experiments(
                B, record, node_map, eta, beta, rho, eta_dir, "targeted",
                source_indices=source_indices, batch_size=batch_size, max_steps=max_steps,
                affected_threshold=affected_threshold, extinction_tolerance=extinction_tolerance,
                convergence_tolerance=convergence_tolerance,
                consecutive_steps=consecutive_steps, long_run_window=long_run_window,
            )
            frame = frame.merge(
                selected[["source_node_id", "candidate_categories", "selection_reasons",
                          "structural_percentile", "dynamic_percentile"]],
                on="source_node_id", how="left", validate="one_to_one",
            )
            rows.append(frame)
        combined = pd.concat(rows, ignore_index=True)
        all_records.append(combined)
        diagnostics.append({
            "network_id": record["network_id"], "selected_candidate_nodes": len(selected),
            "eta_levels": len(eta_levels), "expected_experiments": len(selected) * len(eta_levels),
            "completed_experiments": len(combined), "status": "complete" if len(combined) == len(selected)*len(eta_levels) else "incomplete",
        })
    if not all_records:
        raise ValueError("No targeted candidate nodes were selected.")
    results = pd.concat(all_records, ignore_index=True)
    _atomic_csv(results, targeted_out / "raw_metrics" / "targeted_individual_node_metrics.csv.gz")
    (targeted_out / "diagnostics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(targeted_out / "diagnostics" / "network_completion.csv", index=False)
    stage1 = pd.read_csv(census_out / "raw_metrics" / "single_node_census_metrics.csv.gz")
    comparison = results.merge(
        stage1, on=["network_id", "source_node_id", "eta"],
        how="left", suffixes=("_targeted", "_census"), validate="one_to_one",
    )
    for metric in METRIC_COLUMNS:
        comparison[f"{metric}_difference"] = comparison[f"{metric}_targeted"] - comparison[f"{metric}_census"]
    _atomic_csv(comparison, targeted_out / "comparisons" / "targeted_vs_census.csv.gz")
    _plot_targeted_comparison(comparison, targeted_out)
    report = {
        "run_id": "targeted_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ"),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "complete" if all(x["status"] == "complete" for x in diagnostics) else "incomplete",
        "candidate_count": int(results[["network_id", "source_node_id"]].drop_duplicates().shape[0]),
        "network_count": int(results.network_id.nunique()),
        "eta_levels": list(eta_levels),
        "experiments_expected": int(sum(x["expected_experiments"] for x in diagnostics)),
        "experiments_completed": int(len(results)),
        "simultaneous_shocks": False,
        "source_initialization": "one individual node per experiment, all other nodes start at zero",
        "model": "Alternate Model A",
        "rho": float(rho),
        "simulation_horizon": int(max_steps),
        "affected_node_threshold": float(affected_threshold),
        "extinction_tolerance": float(extinction_tolerance),
        "convergence_tolerance": float(convergence_tolerance),
        "consecutive_steps": int(consecutive_steps),
        "long_run_window": int(long_run_window),
        "batch_size": int(batch_size),
        "matrix_format": MATRIX_FORMAT,
        "git": git_provenance(),
    }
    atomic_json(targeted_out / "diagnostics" / "targeted_validation.json", report)
    return report

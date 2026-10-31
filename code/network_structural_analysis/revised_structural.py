"""WD-only revised structural workflow for the 11-network design."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse import save_npz

from .config import PipelineConfig
from .data_loader import load_icio
from .metrics import structural_analysis
from .synthetic import (
    _adjacency,
    _calibrate_weighted_adjacency,
    _orientation_from_undirected,
    generate_skeleton,
)

REAL_IDS = ("real_2019", "real_2022")
FAMILIES = ("ER", "BA", "WS")
CONFIGURATIONS = (1, 2, 3)
NETWORK_COUNT = 11
MATRIX_FORMAT = "WD_supplier_to_buyer_dependency_share"
ACTIVE_RHO = 0.2
ACTIVE_ETA_STAGE1 = (0.9, 1.1)
ACTIVE_ETA_STAGE2 = (0.5, 0.9, 1.0, 1.1, 1.5, 2.0)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def atomic_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance():
    import subprocess
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], check=True, capture_output=True, text=True
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return {"git_branch": branch or "unknown", "git_commit": commit or "unknown"}
    except Exception:
        return {"git_branch": os.environ.get("GIT_BRANCH", "unknown"),
                "git_commit": os.environ.get("GIT_COMMIT", "unknown")}


def load_icio_pair(data_dir: str | Path, expected_nodes: int = 4050):
    """Load both ICIO years and align 2022 by the stable 2019 activity labels."""
    data_dir = Path(data_dir)
    loaded = {
        year: load_icio(data_dir / f"{year}_SML.csv", expected_nodes=expected_nodes)
        for year in (2019, 2022)
    }
    labels_19 = list(loaded[2019]["labels"])
    labels_22 = list(loaded[2022]["labels"])
    if set(labels_19) != set(labels_22):
        missing = sorted(set(labels_19) - set(labels_22))[:20]
        extra = sorted(set(labels_22) - set(labels_19))[:20]
        raise ValueError(f"ICIO country-industry universes differ. Missing in 2022={missing}; extra={extra}")
    if labels_19 != labels_22:
        positions = {label: i for i, label in enumerate(labels_22)}
        order = np.asarray([positions[label] for label in labels_19], dtype=np.int64)
        loaded[2022]["Z"] = loaded[2022]["Z"][np.ix_(order, order)]
        loaded[2022]["labels"] = labels_19
        loaded[2022]["reordered_to_2019_label_order"] = True
    else:
        loaded[2022]["reordered_to_2019_label_order"] = False
    loaded[2019]["reordered_to_2019_label_order"] = False
    return loaded


def _wd_from_flows(Z: np.ndarray, threshold: float):
    """Build only the retained WD matrix; never materialise UU or UD matrices."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The common dependency-share threshold must be finite and in [0, 1].")
    Z = np.asarray(Z, dtype=np.float64)
    purchases = Z.sum(axis=0)
    shares = np.divide(Z, purchases[None, :], out=np.zeros_like(Z), where=purchases[None, :] > 0)
    np.fill_diagonal(shares, 0.0)
    row, col = np.nonzero((shares >= threshold) & (shares > 0))
    matrix = sparse.csr_matrix((shares[row, col], (row, col)), shape=Z.shape, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, purchases


def _matrix_summary(matrix, family, configuration, year=None):
    n = int(matrix.shape[0])
    support = (matrix != 0).astype(np.uint8).tocsr()
    out_degree = np.asarray(support.getnnz(axis=1)).ravel()
    in_degree = np.asarray(support.getnnz(axis=0)).ravel()
    out_strength = np.asarray(matrix.sum(axis=1)).ravel()
    in_strength = np.asarray(matrix.sum(axis=0)).ravel()
    mutual = support.multiply(support.T).nnz
    return {
        "network_family": family,
        "network_configuration": configuration,
        "year": year,
        "node_count": n,
        "edge_count": int(support.nnz),
        "density": float(support.nnz / (n * (n - 1))) if n > 1 else 0.0,
        "mean_in_degree": float(in_degree.mean()),
        "mean_out_degree": float(out_degree.mean()),
        "mean_in_strength": float(in_strength.mean()),
        "mean_out_strength": float(out_strength.mean()),
        "weighted_in_strength_total": float(matrix.sum()),
        "reciprocity": float(mutual / support.nnz) if support.nnz else 0.0,
        "minimum_nonzero_weight": float(matrix.data.min()) if matrix.nnz else 0.0,
        "maximum_weight": float(matrix.data.max()) if matrix.nnz else 0.0,
    }



def _plot_structural_node_strengths(node_frame: pd.DataFrame, out: Path, network_id: str):
    """Save auditable weighted in/out-strength distribution plots for a network."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = out / "figures" / "structural"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{network_id}_weighted_strengths.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    out_values = node_frame["weighted_out_strength"].to_numpy(dtype=float)
    out_values = out_values[np.isfinite(out_values)]
    axes[0].hist(out_values, bins=min(50, max(1, np.unique(out_values).size)))
    axes[0].set_title("Weighted out-strength")
    axes[0].set_xlabel("Total outgoing dependency share")
    axes[0].set_ylabel("Country-industry activities")
    in_values = node_frame["weighted_in_strength"].to_numpy(dtype=float)
    in_values = in_values[np.isfinite(in_values)]
    axes[1].hist(in_values, bins=min(50, max(1, np.unique(in_values).size)))
    axes[1].set_title("Weighted in-strength")
    axes[1].set_xlabel("Total incoming dependency share")
    axes[1].set_ylabel("Country-industry activities")
    fig.suptitle(f"{network_id}: WD structural strength distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(out).as_posix()


def _store_network(out: Path, network_id: str, group: str, family: str,
                   configuration: str, matrix, labels, year=None, realization_id=None,
                   threshold=None, seed=20261030, path_sample_sources=256, betweenness_k=64,
                   extra_metadata=None):
    """Persist one WD matrix and its node-level structural table."""
    network_dir = out / group / network_id
    network_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = network_dir / "adjacency_wd.npz"
    node_path = network_dir / "node_structural_metrics.csv.gz"
    index_path = network_dir / "node_index.csv"
    edge_path = network_dir / "edge_list.csv.gz"
    metadata_path = network_dir / "network_metadata.json"
    for path in (matrix_path, node_path, index_path, edge_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing revised result: {path}")
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    save_npz(matrix_path, matrix, compressed=True)
    summary, node = structural_analysis(
        matrix, labels, "WD", seed=int(seed),
        path_sample_sources=int(path_sample_sources),
        betweenness_k=int(betweenness_k),
        include_node_metrics=True,
    )
    node = node.rename(columns={"node": "node_label", "in_strength": "weighted_in_strength",
                                "out_strength": "weighted_out_strength"})
    node["node_index"] = node["node_id"].astype(np.int64)
    if year is not None:
        node["node_id"] = node["node_label"].astype(str)
    else:
        node["node_id"] = [f"{network_id}::node_{i:05d}" for i in node["node_index"]]
    node["network_id"] = network_id
    node["network_family"] = family
    node["network_configuration"] = configuration
    node["year"] = year
    node["realization_id"] = realization_id
    node["degree"] = node["degree_undirected_projection"]
    node.to_csv(node_path, index=False, compression="gzip")
    node[["node_id", "node_index", "node_label"]].to_csv(index_path, index=False)
    coo = matrix.tocoo()
    edge = pd.DataFrame({
        "supplier_node_id": node["node_id"].to_numpy(dtype=object)[coo.row],
        "buyer_node_id": node["node_id"].to_numpy(dtype=object)[coo.col],
        "supplier_node_index": coo.row,
        "buyer_node_index": coo.col,
        "dependency_share": coo.data,
    })
    edge.to_csv(edge_path, index=False, compression="gzip")
    structural_figure_path = _plot_structural_node_strengths(node, out, network_id)
    network_summary = _matrix_summary(matrix, family, configuration, year)
    network_summary.update(summary)
    network_summary["structural_figure_path"] = structural_figure_path
    network_summary.update({
        "network_id": network_id,
        "network_family": family,
        "network_configuration": configuration,
        "year": year,
        "realization_id": realization_id,
        "matrix_format": MATRIX_FORMAT,
        "threshold": threshold,
        "matrix_path": matrix_path.relative_to(out).as_posix(),
        "node_metrics_path": node_path.relative_to(out).as_posix(),
        "node_index_path": index_path.relative_to(out).as_posix(),
        "edge_list_path": edge_path.relative_to(out).as_posix(),
    })
    metadata = {
        "network_id": network_id,
        "network_family": family,
        "network_configuration": configuration,
        "year": year,
        "realization_id": realization_id,
        "matrix_format": MATRIX_FORMAT,
        "node_count": int(matrix.shape[0]),
        "edge_count": int(matrix.nnz),
        "threshold": threshold,
        "matrix_sha256": sha256_file(matrix_path),
        "node_metrics_sha256": sha256_file(node_path),
        "node_index_sha256": sha256_file(index_path),
        "git": git_provenance(),
        "extra": extra_metadata or {},
    }
    atomic_json(metadata_path, metadata)
    record = {
        "network_id": network_id,
        "network_family": family,
        "network_configuration": configuration,
        "year": year,
        "realization_id": realization_id,
        "matrix_format": MATRIX_FORMAT,
        "matrix_path": matrix_path.relative_to(out).as_posix(),
        "node_metrics_path": node_path.relative_to(out).as_posix(),
        "node_index_path": index_path.relative_to(out).as_posix(),
        "edge_list_path": edge_path.relative_to(out).as_posix(),
        "network_metadata_path": metadata_path.relative_to(out).as_posix(),
        "node_count": int(matrix.shape[0]),
        "edge_count": int(matrix.nnz),
        "density": network_summary["density"],
        "threshold": threshold,
        "matrix_sha256": metadata["matrix_sha256"],
        "node_metrics_sha256": metadata["node_metrics_sha256"],
        "node_index_sha256": metadata["node_index_sha256"],
    }
    return record, network_summary


def _write_run_manifest(out: Path, stage: str, records, config: dict):
    payload = {
        "run_id": f"{stage}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "stage": stage,
        "status": "complete",
        "matrix_format": MATRIX_FORMAT,
        "network_count": len(records),
        "networks": [r["network_id"] for r in records],
        "configuration": config,
        "git": git_provenance(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
    }
    atomic_json(out / "manifests" / f"{stage}_run_manifest.json", payload)


def run_real_structural(data_dir: str | Path, out_dir: str | Path, threshold: float,
                        expected_nodes: int = 4050, seed: int = 20261030,
                        path_sample_sources: int = 256, betweenness_k: int = 64,
                        legacy_calibration: str | Path = "results_network_analysis/metadata/synthetic_calibration.json"):
    """Generate WD-only structural outputs for OECD ICIO 2019 and 2022."""
    out = Path(out_dir)
    for folder in ("manifests", "summaries", "metadata", "diagnostics"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if (out / "manifests" / "real_network_manifest.csv").exists():
        raise FileExistsError("Real revised structural manifest already exists; outputs are never overwritten.")
    loaded = load_icio_pair(data_dir, expected_nodes)
    records, summaries, calibrations = [], [], []
    for year in (2019, 2022):
        matrix, purchases = _wd_from_flows(loaded[year]["Z"], float(threshold))
        labels = list(loaded[year]["labels"])
        network_id = f"real_{year}"
        record, summary = _store_network(
            out, network_id, "real_networks", "OECD_ICIO", str(year), matrix, labels,
            year=year, threshold=float(threshold), seed=seed,
            path_sample_sources=path_sample_sources, betweenness_k=betweenness_k,
            extra_metadata={
                "source_path": loaded[year]["source_path"],
                "source_sha256": loaded[year]["source_sha256"],
                "raw_shape": loaded[year]["raw_shape"],
                "block_bounds_zero_based": loaded[year]["block_bounds_zero_based"],
                "reordered_to_2019_label_order": loaded[year].get("reordered_to_2019_label_order", False),
                "total_intermediate_flow": loaded[year]["total_intermediate_flow"],
                "self_flow": loaded[year]["self_flow"],
            },
        )
        record.update({
            "source_csv_sha256": loaded[year]["source_sha256"],
            "node_label_order_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest(),
        })
        records.append(record)
        summaries.append(summary)
        support = (matrix != 0).astype(np.uint8).tocsr()
        calibrations.append({
            "weights": matrix.data.copy(),
            "buyer_totals": np.asarray(matrix.sum(axis=0)).ravel(),
            "buyer_indegree": np.asarray(support.getnnz(axis=0)).ravel(),
            "support": support,
        })
    if records[0]["node_label_order_sha256"] != records[1]["node_label_order_sha256"]:
        raise ValueError("2019 and 2022 country-industry node order hashes differ after alignment.")

    legacy_path = Path(legacy_calibration)
    if not legacy_path.is_file():
        raise FileNotFoundError(
            f"Approved synthetic reference-degree calibration not found: {legacy_path}. "
            "It is needed to preserve the established ER/BA/WS parameter values."
        )
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    if int(legacy.get("node_count", -1)) != expected_nodes or not np.isclose(float(legacy.get("threshold", -1)), threshold):
        raise ValueError("Legacy synthetic calibration node count/threshold does not match this WD run.")
    reference_mean_degree = float(legacy["reference_mean_degree"])
    if not np.isfinite(reference_mean_degree) or reference_mean_degree <= 0:
        raise ValueError("Stored synthetic reference mean degree is invalid.")
    total_arcs = sum(int(item["support"].nnz) for item in calibrations)
    mutual_arcs = sum(int(item["support"].multiply(item["support"].T).nnz) for item in calibrations)
    reference_reciprocity = float(mutual_arcs / total_arcs) if total_arcs else 0.0
    weights = np.concatenate([x["weights"] for x in calibrations])
    buyer_totals = np.concatenate([x["buyer_totals"] for x in calibrations])
    buyer_indegree = np.concatenate([x["buyer_indegree"] for x in calibrations])
    if weights.size == 0 or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("WD calibration requires positive finite retained dependency shares.")
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    calibration_samples = out / "metadata" / "empirical_wd_calibration_samples.npz"
    if calibration_samples.exists():
        raise FileExistsError(calibration_samples)
    np.savez_compressed(
        calibration_samples,
        empirical_share_samples=weights,
        empirical_dependency_total_samples=buyer_totals,
        empirical_indegree_samples=buyer_indegree,
    )
    calibration = {
        "node_count": int(expected_nodes),
        "threshold": float(threshold),
        "reference_mean_degree": reference_mean_degree,
        "reference_mean_degree_source": str(legacy_path),
        "reference_reciprocity": reference_reciprocity,
        "reference_reciprocity_source": "arc-weighted reciprocity calculated from the two selected WD supports",
        "weight_sample_count": int(weights.size),
        "buyer_total_sample_count": int(buyer_totals.size),
        "buyer_indegree_sample_count": int(buyer_indegree.size),
        "weight_distribution_source": "positive WD dependency shares retained at the common threshold in 2019 and 2022",
        "buyer_total_conditioning": "retained buyer WD totals paired with each buyer's retained in-degree",
        "synthetic_weights_are_monetary_flows": False,
    }
    atomic_json(out / "metadata" / "empirical_wd_calibration.json", calibration)
    pd.DataFrame(summaries).to_csv(out / "summaries" / "real_network_summaries.csv", index=False)
    pd.DataFrame(records).to_csv(out / "manifests" / "real_network_manifest.csv", index=False)
    _write_run_manifest(out, "real_structural", records, {
        "expected_network_count": 2, "threshold": float(threshold),
        "expected_nodes": int(expected_nodes), "matrix_format": MATRIX_FORMAT,
    })
    atomic_json(out / "diagnostics" / "real_structural_stage.json", {
        "status": "outputs_generated_validation_pending",
        "network_count": 2,
        "network_ids": [r["network_id"] for r in records],
        "node_count_each": int(expected_nodes),
        "threshold": float(threshold),
        "structural_analysis_only": True,
        "synthetic_stage_started": False,
        "dynamics_started": False,
    })
    return records


def run_synthetic_structural(out_dir: str | Path, seed: int = 20261030,
                             path_sample_sources: int = 256, betweenness_k: int = 64):
    """Generate exactly nine full-resolution synthetic WD networks, one each."""
    out = Path(out_dir)
    for folder in ("manifests", "summaries", "metadata", "diagnostics"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    real_validation_path = out / "diagnostics" / "real_structural_validation.json"
    if not real_validation_path.is_file():
        raise FileNotFoundError("Successful real structural validation is required before synthetic generation.")
    real_validation = json.loads(real_validation_path.read_text(encoding="utf-8"))
    if real_validation.get("status") != "valid" or real_validation.get("group") != "real":
        raise ValueError("Synthetic generation is gated on a successful real structural validation report.")
    manifest_path = out / "manifests" / "synthetic_network_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError("Synthetic revised structural manifest already exists; outputs are never overwritten.")
    calibration_path = out / "metadata" / "empirical_wd_calibration.json"
    samples_path = out / "metadata" / "empirical_wd_calibration_samples.npz"
    if not calibration_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError("Run and validate real structural analysis before synthetic generation.")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    samples = np.load(samples_path)
    n = int(calibration["node_count"])
    reference_mean_degree = float(calibration["reference_mean_degree"])
    reference_reciprocity = float(calibration["reference_reciprocity"])
    rng = np.random.default_rng(int(seed))
    cfg = PipelineConfig().synthetic
    records, summaries = [], []
    for family in FAMILIES:
        for level in range(3):
            config_number = level + 1
            multiplier = cfg.mean_degree_multipliers[level] if family == "ER" else (
                cfg.ba_multipliers[level] if family == "BA" else 1.0
            )
            ws_p = cfg.ws_rewire_probabilities[level]
            graph_seed = int(rng.integers(0, 2**31 - 1))
            orientation_seed = int(rng.integers(0, 2**31 - 1))
            weight_seed = int(rng.integers(0, 2**31 - 1))
            graph, params = generate_skeleton(
                family, n, reference_mean_degree, multiplier, ws_p, graph_seed
            )
            directed = _orientation_from_undirected(
                graph, reference_reciprocity, np.random.default_rng(orientation_seed)
            )
            directed_support = _adjacency(directed, n)
            wd, target_totals = _calibrate_weighted_adjacency(
                directed_support,
                samples["empirical_share_samples"],
                samples["empirical_dependency_total_samples"],
                weight_seed,
                samples["empirical_indegree_samples"],
            )
            network_id = f"syn_{family}_{config_number}"
            labels = [f"{network_id}::node_{i:05d}" for i in range(n)]
            record, summary = _store_network(
                out, network_id, "synthetic_networks", family, str(config_number), wd, labels,
                realization_id=1, threshold=float(calibration["threshold"]), seed=seed,
                path_sample_sources=path_sample_sources, betweenness_k=betweenness_k,
                extra_metadata={
                    "realization_count": 1,
                    "graph_seed": graph_seed,
                    "orientation_seed": orientation_seed,
                    "weight_seed": weight_seed,
                    "generation_parameters": params,
                    "requested_multiplier": multiplier,
                    "ws_rewire_probability": ws_p,
                    "empirical_reference_mean_degree": reference_mean_degree,
                    "empirical_reference_reciprocity": reference_reciprocity,
                    "buyer_target_dependency_totals": target_totals.tolist(),
                    "temporary_support_used_for_weighting_only": True,
                    "persisted_network_matrices": ["WD"],
                },
            )
            record.update({
                "realization_count": 1,
                "graph_seed": graph_seed,
                "orientation_seed": orientation_seed,
                "weight_seed": weight_seed,
            })
            records.append(record)
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(out / "summaries" / "synthetic_network_summaries.csv", index=False)
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    _write_run_manifest(out, "synthetic_structural", records, {
        "expected_network_count": 9,
        "families": list(FAMILIES),
        "configurations_per_family": 3,
        "realizations_per_configuration": 1,
        "reference_mean_degree": reference_mean_degree,
        "reference_reciprocity": reference_reciprocity,
        "matrix_format": MATRIX_FORMAT,
    })
    atomic_json(out / "diagnostics" / "synthetic_structural_stage.json", {
        "status": "outputs_generated_validation_pending",
        "network_count": 9,
        "network_ids": [r["network_id"] for r in records],
        "realizations_per_configuration": 1,
        "dynamics_started": False,
    })
    return records


def _load_manifest(out: Path, filename: str):
    path = out / "manifests" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required structural manifest is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Structural manifest is empty: {path}")
    return frame.to_dict(orient="records")


def _validate_one_record(out: Path, record: dict, expected_nodes: int):
    errors = []
    if record.get("matrix_format") != MATRIX_FORMAT:
        errors.append("incorrect matrix_format")
    matrix_path = out / str(record["matrix_path"])
    metrics_path = out / str(record["node_metrics_path"])
    index_path = out / str(record["node_index_path"])
    if not matrix_path.is_file():
        errors.append(f"missing matrix {matrix_path}")
        return errors, None
    if not metrics_path.is_file():
        errors.append(f"missing node metrics {metrics_path}")
        return errors, None
    if not index_path.is_file():
        errors.append(f"missing node index {index_path}")
        return errors, None
    if sha256_file(matrix_path) != str(record.get("matrix_sha256")):
        errors.append("matrix checksum mismatch")
    if record.get("node_metrics_sha256") and sha256_file(metrics_path) != str(record["node_metrics_sha256"]):
        errors.append("node structural metrics checksum mismatch")
    if record.get("node_index_sha256") and sha256_file(index_path) != str(record["node_index_sha256"]):
        errors.append("node index checksum mismatch")
    matrix = sparse.csr_matrix(__import__("scipy.sparse", fromlist=["load_npz"]).load_npz(matrix_path))
    if matrix.shape != (expected_nodes, expected_nodes):
        errors.append(f"matrix shape is {matrix.shape}, expected {(expected_nodes, expected_nodes)}")
    if int(matrix.nnz) != int(record["edge_count"]):
        errors.append(f"matrix edge count {matrix.nnz} does not match manifest edge_count {record['edge_count']}")
    if matrix.nnz and (not np.isfinite(matrix.data).all() or np.any(matrix.data <= 0) or np.any(matrix.data > 1 + 1e-9)):
        errors.append("WD weights must be finite and in (0, 1]")
    if matrix.diagonal().any():
        errors.append("matrix diagonal must be zero")
    metrics = pd.read_csv(metrics_path)
    index = pd.read_csv(index_path)
    if len(metrics) != expected_nodes or len(index) != expected_nodes:
        errors.append(f"node-level row count mismatch: metrics={len(metrics)}, index={len(index)}")
    required = {"network_id", "node_id", "node_label", "node_index", "out_degree",
                "in_degree", "weighted_out_strength", "weighted_in_strength",
                "pagerank", "betweenness", "k_core"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        errors.append(f"node metric columns missing: {missing}")
    if metrics["node_id"].duplicated().any():
        errors.append("duplicate node_id in structural metrics")
    if index["node_id"].duplicated().any():
        errors.append("duplicate node_id in node index")
    if set(metrics["node_id"].astype(str)) != set(index["node_id"].astype(str)):
        errors.append("node IDs differ between structural metrics and node index")
    if not np.array_equal(np.sort(index["node_index"].to_numpy(dtype=int)), np.arange(expected_nodes)):
        errors.append("node index must be a complete 0..N-1 mapping")
    numeric = [c for c in metrics.select_dtypes(include=[np.number]).columns]
    if numeric and not np.isfinite(metrics[numeric].to_numpy(dtype=float)).all():
        errors.append("non-finite numerical structural metric")
    if not metrics["network_id"].astype(str).eq(str(record["network_id"])).all():
        errors.append("network_id mismatch in node metrics")
    return errors, metrics


def validate_structural_group(out_dir: str | Path, group: str, expected_nodes: int = 4050):
    out = Path(out_dir)
    if group == "real":
        records = _load_manifest(out, "real_network_manifest.csv")
        expected_ids = set(REAL_IDS)
    elif group == "synthetic":
        records = _load_manifest(out, "synthetic_network_manifest.csv")
        expected_ids = {f"syn_{family}_{config}" for family in FAMILIES for config in CONFIGURATIONS}
    else:
        raise ValueError("group must be real or synthetic")
    observed_ids = [str(r["network_id"]) for r in records]
    errors = []
    if len(records) != len(expected_ids) or set(observed_ids) != expected_ids or len(set(observed_ids)) != len(observed_ids):
        errors.append(f"network scope mismatch; expected={sorted(expected_ids)}, observed={observed_ids}")
    metrics_by_id = {}
    summaries = []
    for record in records:
        item_errors, metrics = _validate_one_record(out, record, expected_nodes)
        if item_errors:
            errors.extend([f"{record['network_id']}: {message}" for message in item_errors])
        if metrics is not None:
            metrics_by_id[str(record["network_id"])] = metrics
        summaries.append({
            "network_id": record["network_id"],
            "network_family": record["network_family"],
            "network_configuration": record["network_configuration"],
            "year": record.get("year"),
            "node_count": record["node_count"],
            "edge_count": record["edge_count"],
            "validation_status": "failed" if item_errors else "valid",
            "errors": " | ".join(item_errors),
        })
    if group == "real" and set(metrics_by_id) == set(REAL_IDS):
        labels_19 = set(metrics_by_id["real_2019"]["node_label"].astype(str))
        labels_22 = set(metrics_by_id["real_2022"]["node_label"].astype(str))
        if labels_19 != labels_22:
            errors.append("2019/2022 node labels do not align")
    if group == "synthetic":
        counts = {}
        for record in records:
            key = (str(record["network_family"]), str(record["network_configuration"]))
            counts[key] = counts.get(key, 0) + int(record.get("realization_count", 1))
            if int(record.get("realization_id", 1)) != 1:
                errors.append(f"{record['network_id']}: only realization 1 is allowed")
        if len(counts) != 9 or any(value != 1 for value in counts.values()):
            errors.append(f"each of nine synthetic configurations must have exactly one realization: {counts}")
    report = {
        "status": "valid" if not errors else "failed",
        "group": group,
        "expected_network_count": len(expected_ids),
        "observed_network_count": len(records),
        "expected_nodes_per_network": int(expected_nodes),
        "network_ids": observed_ids,
        "errors": errors,
        "networks": summaries,
        "matrix_format": MATRIX_FORMAT,
    }
    atomic_json(out / "diagnostics" / f"{group}_structural_validation.json", report)
    if errors:
        raise ValueError(f"{group} structural validation failed: " + "; ".join(errors[:10]))
    return report


def validate_structural_all(out_dir: str | Path, expected_nodes: int = 4050):
    out = Path(out_dir)
    real_report = validate_structural_group(out, "real", expected_nodes)
    synthetic_report = validate_structural_group(out, "synthetic", expected_nodes)
    all_records = _load_manifest(out, "real_network_manifest.csv") + _load_manifest(out, "synthetic_network_manifest.csv")
    ids = [str(r["network_id"]) for r in all_records]
    errors = []
    if len(all_records) != NETWORK_COUNT or len(set(ids)) != NETWORK_COUNT:
        errors.append(f"expected exactly 11 unique network IDs; found {len(all_records)} rows and {len(set(ids))} IDs")
    if sum(r["network_family"] == "OECD_ICIO" for r in all_records) != 2:
        errors.append("expected exactly two OECD ICIO networks")
    for family in FAMILIES:
        family_records = [r for r in all_records if r["network_family"] == family]
        if len(family_records) != 3 or {str(r["network_configuration"]) for r in family_records} != {"1", "2", "3"}:
            errors.append(f"{family} must contain configurations 1, 2, 3 exactly once")
    if errors:
        raise ValueError("All-network structural validation failed: " + "; ".join(errors))
    report = {
        "status": "valid",
        "network_count": NETWORK_COUNT,
        "real_network_count": 2,
        "synthetic_network_count": 9,
        "families": {family: 3 for family in FAMILIES},
        "real_validation": real_report["status"],
        "synthetic_validation": synthetic_report["status"],
        "expected_nodes_per_network": int(expected_nodes),
        "realizations_per_synthetic_configuration": 1,
        "matrix_format": MATRIX_FORMAT,
        "network_ids": ids,
        "manifest_paths": [
            "manifests/real_network_manifest.csv",
            "manifests/synthetic_network_manifest.csv",
        ],
        "validated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git": git_provenance(),
        "dynamics_gate": "open",
    }
    atomic_json(out / "diagnostics" / "structural_complete.json", report)
    pd.DataFrame(all_records).to_csv(out / "manifests" / "all_11_network_manifest.csv", index=False)
    return report

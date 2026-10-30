"""Validated deterministic mass-action SIS/NIMFA-style shock propagation.

This module runs one and only one model on persisted UU, UD and WD matrices.
It does not build or modify structural networks; it reads their manifests and
sparse .npz files and writes all outputs beneath the requested dynamics folder.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from scipy.stats import t as student_t

from .config import ETA_LEVELS, REPRESENTATIONS
from .metrics import _safe_spectral_radius


def _as_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _resolve_matrix_path(record):
    path = Path(str(record["matrix_path"]))
    if not path.is_absolute():
        path = Path(record.get("base_dir", ".")) / path
    return path


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_support_matrix(record):
    matrix = sparse.csr_matrix(load_npz(_resolve_matrix_path(record)))
    matrix.eliminate_zeros()
    return (matrix != 0).astype(np.uint8).tocsr()


def _validate_paired_support(group_records, group_label):
    by_representation = {r["representation"]: r for r in group_records}
    if set(by_representation) != set(REPRESENTATIONS):
        raise ValueError(f"{group_label}: UU, UD and WD representations are all required.")
    uu = _load_support_matrix(by_representation["UU"])
    ud = _load_support_matrix(by_representation["UD"])
    wd = _load_support_matrix(by_representation["WD"])
    directed_weighted_difference = ud.astype(np.int8) - wd.astype(np.int8)
    directed_weighted_difference.eliminate_zeros()
    if directed_weighted_difference.nnz:
        raise ValueError(f"{group_label}: WD and UD must have identical directed edge support.")
    expected_uu = ud.maximum(ud.T).tocsr()
    undirected_difference = uu.astype(np.int8) - expected_uu.astype(np.int8)
    undirected_difference.eliminate_zeros()
    if undirected_difference.nnz:
        raise ValueError(f"{group_label}: UU must equal the undirected union of UD edges.")


def validate_network_records(network_records, rho=0.2, expected_scope=None,
                             require_full_design=True, expected_nodes=None,
                             expected_synthetic_realizations=None):
    """Validate manifests and all sparse matrices before running any SIS steps.

    Checks file existence, unique IDs, matrix shape, nonnegative finite values,
    zero diagonals, representation semantics, node-universe consistency,
    the six real year/representation matrices, and paired synthetic UU/UD/WD
    matrices. With require_full_design=True, all nine ER/BA/WS configurations
    must be present with the same positive realization count.
    """
    rho = _as_float(rho, "rho")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]; positive recovery is required for a meaningful beta_c.")
    if expected_scope not in (None, "real", "synthetic", "all"):
        raise ValueError("expected_scope must be real, synthetic, all or None.")
    if not network_records:
        raise ValueError("No network records were supplied for SIS validation.")

    normalized, diagnostics = [], []
    seen_ids = set()
    node_counts = set()

    for index, original in enumerate(network_records):
        record = dict(original)
        required = ("network_id", "network_type", "representation", "matrix_path")
        missing = [key for key in required if record.get(key) is None or str(record.get(key)).strip() == ""]
        if missing:
            raise ValueError(f"Manifest row {index + 1} is missing required fields: {missing}")
        network_id = str(record["network_id"])
        if network_id in seen_ids:
            raise ValueError(f"Duplicate network_id in dynamics manifest: {network_id}")
        seen_ids.add(network_id)

        network_type = str(record["network_type"]).lower()
        representation = str(record["representation"]).upper()
        if network_type not in ("real", "synthetic"):
            raise ValueError(f"{network_id}: network_type must be real or synthetic.")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"{network_id}: unsupported representation {representation}.")
        path = _resolve_matrix_path(record)
        if not path.is_file():
            raise FileNotFoundError(f"{network_id}: matrix file not found: {path}")

        matrix = sparse.csr_matrix(load_npz(path), dtype=float)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise ValueError(f"{network_id}: matrix must be non-empty and square; got {matrix.shape}.")
        n_nodes = int(matrix.shape[0])
        node_counts.add(n_nodes)
        if expected_nodes is not None and n_nodes != int(expected_nodes):
            raise ValueError(
                f"{network_id}: expected {int(expected_nodes)} nodes but matrix has {n_nodes}."
            )
        if not np.isfinite(matrix.data).all():
            raise ValueError(f"{network_id}: matrix contains NaN or infinite weights.")
        if np.any(matrix.data < 0):
            raise ValueError(f"{network_id}: matrix contains negative weights.")
        if np.any(matrix.diagonal() != 0):
            raise ValueError(f"{network_id}: self-loops are not allowed in the SIS network matrix.")

        if representation in ("UU", "UD"):
            if matrix.nnz and not np.allclose(matrix.data, 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(f"{network_id}: {representation} must be binary.")
        elif representation == "WD":
            if matrix.nnz and np.any(matrix.data > 1.0 + 1e-9):
                raise ValueError(f"{network_id}: WD dependency-share weights must be <= 1.")

        if representation == "UU":
            difference = matrix - matrix.T
            difference.eliminate_zeros()
            if difference.nnz:
                raise ValueError(f"{network_id}: UU matrix must be symmetric.")

        lambda_max = max(0.0, float(_safe_spectral_radius(matrix)))
        if not np.isfinite(lambda_max):
            raise ValueError(f"{network_id}: spectral radius could not be computed as a finite value.")
        beta_c = float(rho / lambda_max) if lambda_max > 0 else float("inf")
        edge_count = int(matrix.nnz // 2) if representation == "UU" else int(matrix.nnz)
        record.update({
            "network_id": network_id,
            "network_type": network_type,
            "representation": representation,
            "matrix_path": str(path),
            "n_nodes": n_nodes,
            "base_dir": str(record.get("base_dir", ".")),
            "_lambda_max": lambda_max,
            "_beta_critical": beta_c,
        })
        normalized.append(record)
        diagnostics.append({
            "network_id": network_id,
            "network_type": network_type,
            "year": record.get("year"),
            "family": record.get("family"),
            "configuration": record.get("configuration"),
            "realization": record.get("realization"),
            "representation": representation,
            "n_nodes": n_nodes,
            "edges": edge_count,
            "matrix_nnz": int(matrix.nnz),
            "minimum_nonzero_weight": float(matrix.data.min()) if matrix.nnz else 0.0,
            "maximum_weight": float(matrix.data.max()) if matrix.nnz else 0.0,
            "lambda_max": lambda_max,
            "rho": rho,
            "beta_critical": beta_c,
            "normalized_threshold_defined": bool(lambda_max > 0.0),
            "matrix_path": str(path),
            "matrix_sha256": _sha256_file(path),
        })

    if len(node_counts) != 1:
        raise ValueError(f"All SIS matrices must use the same node order and size; found sizes {sorted(node_counts)}.")

    types_present = {r["network_type"] for r in normalized}
    expected_types = {
        "real": {"real"},
        "synthetic": {"synthetic"},
        "all": {"real", "synthetic"},
    }
    if expected_scope is not None and types_present != expected_types[expected_scope]:
        raise ValueError(
            f"Requested scope '{expected_scope}' requires network types "
            f"{sorted(expected_types[expected_scope])}; manifest contains {sorted(types_present)}."
        )

    real_records = [r for r in normalized if r["network_type"] == "real"]
    if real_records:
        real_pairs = set()
        for record in real_records:
            try:
                year = int(record["year"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{record['network_id']}: real manifest rows require year 2019 or 2022.")
            if year not in (2019, 2022):
                raise ValueError(f"{record['network_id']}: unexpected real-network year {year}.")
            pair = (year, record["representation"])
            if pair in real_pairs:
                raise ValueError(f"Duplicate real year/representation pair: {pair}.")
            real_pairs.add(pair)
        if require_full_design:
            expected_pairs = {(year, rep) for year in (2019, 2022) for rep in REPRESENTATIONS}
            if real_pairs != expected_pairs:
                missing = sorted(expected_pairs - real_pairs)
                extra = sorted(real_pairs - expected_pairs)
                raise ValueError(f"Real manifest is incomplete. Missing={missing}; extra={extra}.")
        for year in (2019, 2022):
            year_group = [r for r in real_records if int(r["year"]) == year]
            if year_group:
                _validate_paired_support(year_group, f"Real ICIO {year}")

    synthetic_records = [r for r in normalized if r["network_type"] == "synthetic"]
    if synthetic_records:
        skeleton_groups = {}
        for record in synthetic_records:
            required = ("family", "configuration", "realization", "graph_seed",
                        "orientation_seed", "weight_seed")
            missing = [key for key in required if record.get(key) is None or pd.isna(record.get(key))]
            if missing:
                raise ValueError(f"{record['network_id']}: synthetic manifest missing {missing}.")
            family = str(record["family"]).upper()
            if family not in ("ER", "BA", "WS"):
                raise ValueError(f"{record['network_id']}: unknown synthetic family {family}.")
            key = (family, str(record["configuration"]), str(record["realization"]))
            skeleton_groups.setdefault(key, []).append(record)

        for skeleton_id, group in skeleton_groups.items():
            reps = [r["representation"] for r in group]
            if len(reps) != 3 or set(reps) != set(REPRESENTATIONS):
                raise ValueError(
                    f"Synthetic skeleton {skeleton_id} must have exactly one UU, UD and WD matrix; got {reps}."
                )
            for seed_name in ("graph_seed", "orientation_seed", "weight_seed"):
                values = [str(r.get(seed_name)) for r in group if r.get(seed_name) is not None and not pd.isna(r.get(seed_name))]
                if values and (len(values) != 3 or len(set(values)) != 1):
                    raise ValueError(f"Synthetic skeleton {skeleton_id} has inconsistent {seed_name} values.")
            _validate_paired_support(group, f"Synthetic skeleton {skeleton_id}")

        if require_full_design:
            configs_by_family = {}
            realizations_by_config = {}
            for family, configuration, realization in skeleton_groups:
                configs_by_family.setdefault(family, set()).add(configuration)
                realizations_by_config.setdefault((family, configuration), set()).add(realization)
            if set(configs_by_family) != {"ER", "BA", "WS"}:
                raise ValueError(f"Synthetic manifest must contain ER, BA and WS; got {sorted(configs_by_family)}.")
            if any(len(configs_by_family[family]) != 3 for family in ("ER", "BA", "WS")):
                raise ValueError("Synthetic manifest must contain exactly three configurations for each ER/BA/WS family.")
            counts = {key: len(values) for key, values in realizations_by_config.items()}
            realization_sets = [frozenset(values) for values in realizations_by_config.values()]
            if len(set(counts.values())) != 1 or not counts or min(counts.values()) < 1:
                raise ValueError(f"Every synthetic configuration must contain the same positive number of realizations; got {counts}.")
            if len(set(realization_sets)) != 1:
                raise ValueError("Every synthetic configuration must use the same realization IDs (normally 1..R).")
            config_id_sets = {
                family: frozenset(configs_by_family[family])
                for family in ("ER", "BA", "WS")
            }
            if any(values != frozenset(("1", "2", "3")) for values in config_id_sets.values()):
                raise ValueError(
                    "Synthetic configurations must be named 1, 2 and 3 within each ER/BA/WS family; "
                    f"got {config_id_sets}."
                )
            if expected_synthetic_realizations is not None:
                expected_count = int(expected_synthetic_realizations)
                if expected_count < 1:
                    raise ValueError("expected_synthetic_realizations must be positive.")
                observed_counts = set(counts.values())
                if observed_counts != {expected_count}:
                    raise ValueError(
                        f"Expected {expected_count} realizations in every synthetic configuration; "
                        f"observed counts are {counts}."
                    )
                expected_ids = frozenset(str(i) for i in range(1, expected_count + 1))
                if any(values != expected_ids for values in realizations_by_config.values()):
                    raise ValueError(
                        f"Each synthetic configuration must contain realization IDs 1..{expected_count}."
                    )

    diagnostics_df = pd.DataFrame(diagnostics)
    report = {
        "status": "valid",
        "expected_scope": expected_scope,
        "full_design_required": bool(require_full_design),
        "network_count": int(len(normalized)),
        "real_network_count": int(len(real_records)),
        "synthetic_network_count": int(len(synthetic_records)),
        "node_count": int(next(iter(node_counts))),
        "representations": list(REPRESENTATIONS),
        "eta_levels": list(ETA_LEVELS),
        "real_networks": len(real_records) == 6 if real_records else None,
        "synthetic_skeleton_count": len(skeleton_groups) if synthetic_records else 0,
        "expected_synthetic_realizations_per_configuration": (
            int(expected_synthetic_realizations)
            if expected_synthetic_realizations is not None and synthetic_records else None
        ),
        "synthetic_configuration_count": (
            len({(str(r.get("family")), str(r.get("configuration"))) for r in synthetic_records})
            if synthetic_records else 0
        ),
        "rho_for_beta_critical": rho,
        "normalized_threshold_undefined_network_ids": [
            row["network_id"] for row in diagnostics if row["lambda_max"] <= 0.0
        ],
        "beta_critical_by_representation": {},
        "interpretation": (
            "This is an input-integrity and spectral-threshold precheck; it does not simulate SIS dynamics."
        ),
    }
    for representation in REPRESENTATIONS:
        values = diagnostics_df.loc[
            diagnostics_df["representation"] == representation, "beta_critical"
        ].to_numpy(dtype=float)
        if len(values):
            finite = values[np.isfinite(values)]
            report["beta_critical_by_representation"][representation] = {
                "minimum": float(finite.min()) if len(finite) else None,
                "median": float(np.median(finite)) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "infinite_count_no_propagation": int(np.isinf(values).sum()),
            }
    return normalized, diagnostics_df, report


def simulate_sis(B, initial_infected, beta, rho, max_steps=300,
                 tolerance=1e-8, extinction_threshold=1e-8):
    """Simulate the agreed discrete-time bilinear SIS/NIMFA-style equation.

    B[i,j] represents the supplier i -> buyer j influence. Hence the force
    applied to buyer j is (B.T @ d)[j]. The same update is used for UU, UD,
    and WD; only the supplied matrix differs.
    """
    B = sparse.csr_matrix(B, dtype=float)
    B.sum_duplicates()
    B.eliminate_zeros()
    if B.shape[0] != B.shape[1] or B.shape[0] < 1:
        raise ValueError("SIS matrix B must be non-empty and square.")
    if not np.isfinite(B.data).all() or np.any(B.data < 0):
        raise ValueError("SIS matrix B must contain finite nonnegative weights.")
    beta = _as_float(beta, "beta")
    rho = _as_float(rho, "rho")
    tolerance = _as_float(tolerance, "tolerance")
    extinction_threshold = _as_float(extinction_threshold, "extinction_threshold")
    max_steps = int(max_steps)
    if beta < 0:
        raise ValueError("beta must be nonnegative.")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]; positive recovery is required for a meaningful beta_c.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if extinction_threshold < 0:
        raise ValueError("extinction_threshold must be nonnegative.")

    d = np.asarray(initial_infected, dtype=float).copy()
    if d.shape != (B.shape[0],) or not np.isfinite(d).all():
        raise ValueError("initial_infected must be a finite vector matching B.")
    d = np.clip(d, 0.0, 1.0)
    BT = B.T.tocsr()
    prevalence = [float(d.mean())]
    exposure = np.asarray(BT @ d).ravel()
    mean_exposure = [float(exposure.mean())]
    total_exposure = [float(exposure.sum())]
    converged, extinct, extinction_time = False, False, None

    for step in range(1, max_steps + 1):
        force = np.asarray(BT @ d).ravel()
        next_d = np.clip((1.0 - rho) * d + (1.0 - d) * beta * force, 0.0, 1.0)
        exposure = np.asarray(BT @ next_d).ravel()
        prevalence.append(float(next_d.mean()))
        mean_exposure.append(float(exposure.mean()))
        total_exposure.append(float(exposure.sum()))
        delta = float(np.max(np.abs(next_d - d)))
        d = next_d
        if float(np.max(d)) <= extinction_threshold:
            extinct = True
            extinction_time = step
            break
        if delta <= tolerance:
            converged = True
            break

    return {
        "prevalence": np.asarray(prevalence, dtype=float),
        "mean_exposure": np.asarray(mean_exposure, dtype=float),
        "total_exposure": np.asarray(total_exposure, dtype=float),
        "converged": bool(converged),
        "extinct": bool(extinct),
        "extinction_time": extinction_time,
        "iterations": len(prevalence) - 1,
        "final_state": d,
    }


def simulate_sis_ensemble(B, initial_states, beta, rho, max_steps=300,
                         tolerance=1e-8, extinction_threshold=1e-8):
    """Vectorized SIS simulation for multiple initial seed sets on one matrix.

    Columns of initial_states are separate simulations. The sparse matrix is
    traversed with sparse-dense products, while each seed set retains its own
    convergence/extinction time and exact trajectory length.
    """
    B = sparse.csr_matrix(B, dtype=float)
    B.sum_duplicates()
    B.eliminate_zeros()
    if B.shape[0] != B.shape[1] or B.shape[0] < 1:
        raise ValueError("SIS matrix B must be non-empty and square.")
    if not np.isfinite(B.data).all() or np.any(B.data < 0):
        raise ValueError("SIS matrix B must contain finite nonnegative weights.")
    beta = _as_float(beta, "beta")
    rho = _as_float(rho, "rho")
    tolerance = _as_float(tolerance, "tolerance")
    extinction_threshold = _as_float(extinction_threshold, "extinction_threshold")
    max_steps = int(max_steps)
    if beta < 0:
        raise ValueError("beta must be nonnegative.")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1].")
    if max_steps < 1 or tolerance <= 0 or extinction_threshold < 0:
        raise ValueError("max_steps/tolerance/extinction_threshold are outside their valid ranges.")

    states = np.asarray(initial_states, dtype=float).copy()
    if states.ndim != 2 or states.shape[0] != B.shape[0] or states.shape[1] < 1:
        raise ValueError("initial_states must have shape (number_of_nodes, number_of_seed_sets).")
    if not np.isfinite(states).all():
        raise ValueError("initial_states must contain only finite values.")
    states = np.clip(states, 0.0, 1.0)
    seed_count = states.shape[1]
    BT = B.T.tocsr()

    prevalence_traces = [[float(states[:, j].mean())] for j in range(seed_count)]
    exposure_initial = np.asarray(BT @ states)
    mean_exposure_traces = [
        [float(exposure_initial[:, j].mean())] for j in range(seed_count)
    ]
    total_exposure_traces = [
        [float(exposure_initial[:, j].sum())] for j in range(seed_count)
    ]

    active = np.ones(seed_count, dtype=bool)
    extinct = np.zeros(seed_count, dtype=bool)
    converged = np.zeros(seed_count, dtype=bool)
    extinction_times = [None] * seed_count

    for step in range(1, max_steps + 1):
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            break

        current = states[:, active_indices]
        force = np.asarray(BT @ current)
        next_states = np.clip(
            (1.0 - rho) * current + (1.0 - current) * beta * force,
            0.0, 1.0,
        )
        delta = np.max(np.abs(next_states - current), axis=0)
        states[:, active_indices] = next_states
        next_exposure = np.asarray(BT @ next_states)

        for local_index, seed_index in enumerate(active_indices):
            state = next_states[:, local_index]
            prevalence_traces[seed_index].append(float(state.mean()))
            mean_exposure_traces[seed_index].append(float(next_exposure[:, local_index].mean()))
            total_exposure_traces[seed_index].append(float(next_exposure[:, local_index].sum()))

        is_extinct = np.max(next_states, axis=0) <= extinction_threshold
        is_converged = (delta <= tolerance) & ~is_extinct
        for local_index, seed_index in enumerate(active_indices):
            if is_extinct[local_index]:
                extinct[seed_index] = True
                extinction_times[seed_index] = step
            elif is_converged[local_index]:
                converged[seed_index] = True
        active[active_indices[is_extinct | is_converged]] = False

    results = []
    for j in range(seed_count):
        results.append({
            "prevalence": np.asarray(prevalence_traces[j], dtype=float),
            "mean_exposure": np.asarray(mean_exposure_traces[j], dtype=float),
            "total_exposure": np.asarray(total_exposure_traces[j], dtype=float),
            "converged": bool(converged[j]),
            "extinct": bool(extinct[j]),
            "extinction_time": extinction_times[j],
            "iterations": len(prevalence_traces[j]) - 1,
            "final_state": states[:, j].copy(),
        })
    return results


def critical_beta(B, rho):
    """Return lambda_max(B) and beta_c = rho / lambda_max(B)."""
    rho = _as_float(rho, "rho")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]; positive recovery is required for a meaningful beta_c.")
    B = sparse.csr_matrix(B, dtype=float)
    if not np.isfinite(B.data).all() or np.any(B.data < 0):
        raise ValueError("B must contain finite nonnegative weights.")
    lambda_max = max(0.0, float(_safe_spectral_radius(B)))
    beta_c = float(rho / lambda_max) if lambda_max > 0 else float("inf")
    return lambda_max, beta_c


def build_seed_sets(n_nodes, count, seed, infection_fraction=0.01):
    """Generate independent initial infected-node sets of ceil(fraction*N)."""
    n_nodes, count = int(n_nodes), int(count)
    infection_fraction = _as_float(infection_fraction, "infection_fraction")
    if n_nodes < 1 or count < 1:
        raise ValueError("n_nodes and count must be positive.")
    if not 0.0 < infection_fraction <= 1.0:
        raise ValueError("infection_fraction must be in (0, 1].")
    n_infected = max(1, int(np.ceil(infection_fraction * n_nodes)))
    rng = np.random.default_rng(int(seed))
    return [
        np.sort(rng.choice(n_nodes, size=n_infected, replace=False)).astype(int)
        for _ in range(count)
    ]


def _ci(values, bounds=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    low, high = mean - half, mean + half
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return mean, sd, low, high, n


def _pad_trace(values, length):
    if len(values) >= length:
        return values[:length]
    return np.pad(values, (0, length - len(values)), mode="constant",
                  constant_values=float(values[-1]))


def _seed_outcomes(trace, network_id, scenario, seed_set_id, eta,
                   beta, beta_c, lambda_max, representation, network_type):
    p = trace["prevalence"]
    peak_idx = int(np.argmax(p))
    tail_start = max(0, int(np.floor(len(p) * 0.9)))
    return {
        "network_id": network_id,
        "network_type": network_type,
        "representation": representation,
        "scenario": scenario,
        "eta": eta,
        "beta": float(beta),
        "beta_critical": float(beta_c),
        "lambda_max": float(lambda_max),
        "seed_set_id": int(seed_set_id),
        "peak_prevalence": float(p[peak_idx]),
        "time_to_peak": int(peak_idx),
        "long_run_prevalence": float(np.mean(p[tail_start:])),
        "cumulative_prevalence": float(np.sum(p)),
        "extinction_time": trace["extinction_time"],
        "extinct": bool(trace["extinct"]),
        "converged": bool(trace["converged"]),
        "iterations": int(trace["iterations"]),
        "final_prevalence": float(p[-1]),
        "max_prevalence": float(np.max(p)),
        "max_mean_exposure": float(np.max(trace["mean_exposure"])),
        "max_total_exposure": float(np.max(trace["total_exposure"])),
        "normalized_threshold_defined": bool(lambda_max > 0.0),
    }


def _scenario_list(B, representation, rho, beta_by_rep, lambda_max=None, beta_c=None):
    if lambda_max is None or beta_c is None:
        lambda_max, beta_c = critical_beta(B, rho)
    scenarios = [("absolute", float(beta_by_rep[representation]), None, lambda_max, beta_c)]
    for eta in ETA_LEVELS:
        beta = float(eta * beta_c) if np.isfinite(beta_c) else 0.0
        scenarios.append((f"eta_{eta:g}", beta, float(eta), lambda_max, beta_c))
    return scenarios


def run_dynamics_analysis(network_records, out_dir, seed=20261030, rho=0.2,
                          beta_by_rep=None, max_steps=300, tolerance=1e-8,
                          real_seed_sets=20, synthetic_seed_sets=5,
                          infection_fraction=0.01, expected_scope=None,
                          require_full_design=True, expected_nodes=None,
                          write_plots=True, expected_synthetic_realizations=None):
    """Run Alternate Model A on every validated matrix in the selected scope."""
    rho = _as_float(rho, "rho")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]; positive recovery is required for a meaningful beta_c.")
    if beta_by_rep is None or (set(REPRESENTATIONS) - set(beta_by_rep)):
        raise ValueError("Absolute beta values are required for UU, UD and WD.")
    beta_by_rep = {rep: _as_float(beta_by_rep[rep], f"beta_{rep.lower()}") for rep in REPRESENTATIONS}
    if any(value < 0 for value in beta_by_rep.values()):
        raise ValueError("Absolute beta values must be nonnegative.")
    max_steps = int(max_steps)
    tolerance = _as_float(tolerance, "tolerance")
    if max_steps < 1 or tolerance <= 0:
        raise ValueError("max_steps must be positive and tolerance must be positive.")
    if int(real_seed_sets) < 1 or int(synthetic_seed_sets) < 1:
        raise ValueError("real_seed_sets and synthetic_seed_sets must be positive.")
    infection_fraction = _as_float(infection_fraction, "infection_fraction")
    if not 0.0 < infection_fraction <= 1.0:
        raise ValueError("infection_fraction must be in (0, 1].")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, precheck, validation_report = validate_network_records(
        network_records, rho=rho, expected_scope=expected_scope,
        require_full_design=require_full_design, expected_nodes=expected_nodes,
        expected_synthetic_realizations=expected_synthetic_realizations,
    )
    precheck.to_csv(out / "dynamics_input_precheck.csv", index=False)
    (out / "dynamics_input_validation.json").write_text(
        json.dumps(_json_safe(validation_report), indent=2, allow_nan=False)
    )

    trajectory_path = out / "dynamics_trajectories.csv.gz"
    seed_rows, network_rows = [], []
    seed_set_rows, network_seed_rows = [], []
    seed_cache = {}
    seed_metadata_written = set()

    with gzip.open(trajectory_path, "wt", encoding="utf-8", newline="") as trajectory_file:
        write_header = True
        for record in records:
            network_id = record["network_id"]
            representation = record["representation"]
            network_type = record["network_type"]
            B = sparse.csr_matrix(load_npz(record["matrix_path"]), dtype=float)
            n = B.shape[0]
            if network_type == "real":
                seed_group = "real_common_matched_across_years_and_representations"
                if seed_group not in seed_cache:
                    seed_cache[seed_group] = build_seed_sets(
                        n, real_seed_sets, int(seed) + 101, infection_fraction)
            else:
                skeleton_id = (
                    f"{record['family']}_{record['configuration']}_{record['realization']}"
                )
                seed_group = skeleton_id
                if seed_group not in seed_cache:
                    graph_seed = int(record["graph_seed"])
                    seed_cache[seed_group] = build_seed_sets(
                        n, synthetic_seed_sets,
                        int(seed) + graph_seed + 9001,
                        infection_fraction,
                    )
            seed_sets = seed_cache[seed_group]
            network_seed_rows.append({
                "network_id": network_id,
                "network_type": network_type,
                "seed_group": seed_group,
                "seed_set_count": len(seed_sets),
            })
            if seed_group not in seed_metadata_written:
                for seed_set_id, node_ids in enumerate(seed_sets):
                    seed_set_rows.append({
                        "network_type": network_type,
                        "seed_group": seed_group,
                        "seed_set_id": seed_set_id,
                        "initial_infected_count": len(node_ids),
                        "initial_infected_fraction": len(node_ids) / n,
                        "initial_node_ids": ";".join(str(int(x)) for x in node_ids),
                    })
                seed_metadata_written.add(seed_group)

            lambda_max = float(record["_lambda_max"])
            beta_c = float(record["_beta_critical"])
            scenarios = _scenario_list(
                B, representation, rho, beta_by_rep,
                lambda_max=lambda_max, beta_c=beta_c,
            )
            initial_states = np.zeros((n, len(seed_sets)), dtype=float)
            for seed_set_id, node_ids in enumerate(seed_sets):
                initial_states[node_ids, seed_set_id] = 1.0

            for scenario, beta, eta, scenario_lambda, scenario_beta_c in scenarios:
                if not np.isclose(lambda_max, scenario_lambda, rtol=1e-9, atol=1e-12):
                    raise RuntimeError(f"{network_id}: spectral-radius mismatch between validation and simulation.")
                traces = simulate_sis_ensemble(
                    B, initial_states, beta, rho,
                    max_steps=max_steps, tolerance=tolerance,
                )
                outcomes = [
                    _seed_outcomes(
                        trace, network_id, scenario, seed_set_id, eta,
                        beta, beta_c, lambda_max, representation, network_type,
                    )
                    for seed_set_id, trace in enumerate(traces)
                ]

                for outcome in outcomes:
                    seed_rows.append({
                        **{k: v for k, v in record.items() if not k.startswith("_") and k != "base_dir"},
                        **outcome,
                        "rho": rho,
                        "seed_group": seed_group,
                        "initial_infected_count": len(seed_sets[0]),
                        "initial_infected_fraction": len(seed_sets[0]) / n,
                    })

                metric_bounds = {
                    "peak_prevalence": (0.0, 1.0),
                    "long_run_prevalence": (0.0, 1.0),
                    "final_prevalence": (0.0, 1.0),
                    "extinction_fraction": (0.0, 1.0),
                    "convergence_fraction": (0.0, 1.0),
                }
                metric_names = (
                    "peak_prevalence", "time_to_peak", "long_run_prevalence",
                    "cumulative_prevalence", "final_prevalence", "max_mean_exposure",
                    "max_total_exposure",
                )
                aggregate = {
                    "network_id": network_id,
                    "network_type": network_type,
                    "representation": representation,
                    "scenario": scenario,
                    "eta": eta,
                    "beta": float(beta),
                    "beta_critical": beta_c,
                    "lambda_max": lambda_max,
                    "rho": rho,
                    "seed_sets": len(outcomes),
                    "seed_group": seed_group,
                    "normalized_threshold_defined": bool(lambda_max > 0.0),
                    "extinction_fraction": float(np.mean([x["extinct"] for x in outcomes])),
                    "convergence_fraction": float(np.mean([x["converged"] for x in outcomes])),
                    "mean_iterations": float(np.mean([x["iterations"] for x in outcomes])),
                }
                for metric in metric_names:
                    mean, sd, low, high, count = _ci(
                        [x[metric] for x in outcomes], bounds=metric_bounds.get(metric)
                    )
                    aggregate.update({
                        f"{metric}_mean_across_seed_sets": mean,
                        f"{metric}_sd_across_seed_sets": sd,
                        f"{metric}_ci95_seed_low": low,
                        f"{metric}_ci95_seed_high": high,
                    })
                for metric, bounds in metric_bounds.items():
                    if metric in ("extinction_fraction", "convergence_fraction"):
                        mean, sd, low, high, _ = _ci(
                            [float(x["extinct"] if metric == "extinction_fraction" else x["converged"])
                             for x in outcomes],
                            bounds=bounds,
                        )
                        aggregate.update({
                            f"{metric}_mean_across_seed_sets": mean,
                            f"{metric}_sd_across_seed_sets": sd,
                            f"{metric}_ci95_seed_low": low,
                            f"{metric}_ci95_seed_high": high,
                        })
                network_rows.append({
                    **{k: v for k, v in record.items() if not k.startswith("_") and k != "base_dir"},
                    **aggregate,
                })

                trace_length = max_steps + 1
                prevalence = np.vstack([_pad_trace(x["prevalence"], trace_length) for x in traces])
                mean_exposure = np.vstack([_pad_trace(x["mean_exposure"], trace_length) for x in traces])
                total_exposure = np.vstack([_pad_trace(x["total_exposure"], trace_length) for x in traces])
                p_mean = prevalence.mean(axis=0)
                p_sd = prevalence.std(axis=0, ddof=1) if len(seed_sets) > 1 else np.zeros(trace_length)
                e_mean = mean_exposure.mean(axis=0)
                et_mean = total_exposure.mean(axis=0)
                critical = (
                    student_t.ppf(.975, len(seed_sets) - 1) / np.sqrt(len(seed_sets))
                    if len(seed_sets) > 1 else 0.0
                )
                trajectory = pd.DataFrame({
                    "network_id": network_id,
                    "network_type": network_type,
                    "year": record.get("year"),
                    "family": record.get("family"),
                    "configuration": record.get("configuration"),
                    "realization": record.get("realization"),
                    "representation": representation,
                    "scenario": scenario,
                    "eta": eta,
                    "beta": float(beta),
                    "beta_critical": beta_c,
                    "lambda_max": lambda_max,
                    "rho": rho,
                    "time": np.arange(trace_length),
                    "prevalence_mean": p_mean,
                    "prevalence_sd_across_seed_sets": p_sd,
                    "prevalence_ci95_seed_low": np.clip(p_mean - critical * p_sd, 0.0, 1.0),
                    "prevalence_ci95_seed_high": np.clip(p_mean + critical * p_sd, 0.0, 1.0),
                    "mean_exposure": e_mean,
                    "total_exposure": et_mean,
                })
                trajectory.to_csv(trajectory_file, index=False, header=write_header)
                write_header = False

    seed_results = pd.DataFrame(seed_rows)
    network_results = pd.DataFrame(network_rows)
    seed_results.to_csv(out / "dynamics_seed_results.csv.gz", index=False, compression="gzip")
    network_results.to_csv(out / "dynamics_network_results.csv.gz", index=False, compression="gzip")
    pd.DataFrame(seed_set_rows).to_csv(
        out / "initial_seed_sets.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(network_seed_rows).to_csv(
        out / "network_seed_group_assignments.csv", index=False
    )

    trajectory_summaries_dir = out / "trajectory_summaries"
    if write_plots:
        _save_trajectory_summaries_and_plots(trajectory_path, trajectory_summaries_dir)
    else:
        _save_trajectory_summaries(trajectory_path, trajectory_summaries_dir)

    synthetic = network_results[network_results.network_type == "synthetic"]
    if not synthetic.empty:
        _write_ensemble_summary(
            synthetic,
            ["family", "configuration", "representation", "scenario", "eta"],
            out / "synthetic_dynamics_ensemble_summary.csv",
        )
    real = network_results[network_results.network_type == "real"]
    if not real.empty:
        real.to_csv(out / "real_dynamics_seedset_summary.csv", index=False)

    run_metadata = {
        "model": "Alternate Model A: deterministic mass-action/bilinear SIS (NIMFA-style)",
        "model_count": 1,
        "equation": (
            "d_j(t+1)=clip_[0,1]((1-rho)*d_j(t)+(1-d_j(t))*beta*sum_i B_ij*d_i(t)); "
            "e(t)=B.T@d(t)"
        ),
        "orientation": "B[i,j] is supplier i -> buyer j; force on buyer j is (B.T @ d)[j].",
        "representations": {
            "UU": "binary undirected adjacency A",
            "UD": "binary directed supplier-to-buyer adjacency A",
            "WD": "threshold-retained dependency-share matrix P",
        },
        "rho": rho,
        "absolute_beta_by_representation": beta_by_rep,
        "eta_levels": list(ETA_LEVELS),
        "beta_critical": "rho / lambda_max(B)",
        "initial_infected_fraction": infection_fraction,
        "real_seed_sets": int(real_seed_sets),
        "synthetic_seed_sets_per_skeleton": int(synthetic_seed_sets),
        "max_steps": max_steps,
        "tolerance": tolerance,
        "random_seed": int(seed),
        "network_count": int(len(records)),
        "input_validation": validation_report,
        "outputs": [
            "dynamics_input_precheck.csv",
            "dynamics_input_validation.json",
            "dynamics_seed_results.csv.gz",
            "dynamics_network_results.csv.gz",
            "dynamics_trajectories.csv.gz",
            "initial_seed_sets.csv.gz",
            "network_seed_group_assignments.csv",
            "real_dynamics_seedset_summary.csv (when real matrices are included)",
            "synthetic_dynamics_ensemble_summary.csv (when synthetic matrices are included)",
            "trajectory_summaries/",
        ],
        "uncertainty": (
            "Seed-set uncertainty is computed within each network. Synthetic graph-to-graph "
            "confidence intervals use the per-network seed-set means across independent realizations."
        ),
        "batched_simulation": "All initial seed sets for a given network/scenario are updated in one sparse-dense batch; seed-specific stopping times are preserved.",
        "other_dynamical_models": "None. Only Alternate Model A is implemented.",
        "interpretation_limit": (
            "This is an abstract deterministic propagation comparison, not a calibrated "
            "production-loss or outage-probability forecast."
        ),
    }
    (out / "dynamics_run_metadata.json").write_text(
        json.dumps(_json_safe(run_metadata), indent=2, allow_nan=False)
    )
    return network_results, seed_results


def _save_trajectory_summaries(trajectory_path, out_dir):
    """Chunked aggregation without loading the million-row trajectory file at once."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_parts, synthetic_parts = [], []
    chunks = pd.read_csv(trajectory_path, compression="gzip", chunksize=100_000)
    for chunk in chunks:
        real = chunk[chunk["network_type"] == "real"]
        if not real.empty:
            keys = ["year", "representation", "scenario", "eta", "time"]
            part = real.groupby(keys, dropna=False, as_index=False).agg(
                rows=("network_id", "count"),
                prevalence_sum=("prevalence_mean", "sum"),
                seed_low_sum=("prevalence_ci95_seed_low", "sum"),
                seed_high_sum=("prevalence_ci95_seed_high", "sum"),
                mean_exposure_sum=("mean_exposure", "sum"),
                total_exposure_sum=("total_exposure", "sum"),
            )
            real_parts.append(part)
        synthetic = chunk[chunk["network_type"] == "synthetic"]
        if not synthetic.empty:
            keys = ["family", "configuration", "representation", "scenario", "eta", "time"]
            synthetic = synthetic.copy()
            synthetic["prevalence_squared"] = synthetic["prevalence_mean"] ** 2
            part = synthetic.groupby(keys, dropna=False, as_index=False).agg(
                realization_count=("network_id", "nunique"),
                prevalence_sum=("prevalence_mean", "sum"),
                prevalence_squared_sum=("prevalence_squared", "sum"),
                mean_exposure_sum=("mean_exposure", "sum"),
                total_exposure_sum=("total_exposure", "sum"),
            )
            synthetic_parts.append(part)

    if real_parts:
        keys = ["year", "representation", "scenario", "eta", "time"]
        real_summary = pd.concat(real_parts, ignore_index=True).groupby(
            keys, dropna=False, as_index=False
        ).sum(numeric_only=True)
        count = real_summary["rows"].to_numpy(dtype=float)
        for source, target in (
            ("prevalence_sum", "prevalence_mean"),
            ("seed_low_sum", "prevalence_ci95_seed_low"),
            ("seed_high_sum", "prevalence_ci95_seed_high"),
            ("mean_exposure_sum", "mean_exposure"),
            ("total_exposure_sum", "total_exposure"),
        ):
            real_summary[target] = real_summary[source] / count
        real_summary.to_csv(out / "real_trajectory_summary.csv.gz",
                            index=False, compression="gzip")

    if synthetic_parts:
        keys = ["family", "configuration", "representation", "scenario", "eta", "time"]
        synthetic_summary = pd.concat(synthetic_parts, ignore_index=True).groupby(
            keys, dropna=False, as_index=False
        ).sum(numeric_only=True)
        count = synthetic_summary["realization_count"].to_numpy(dtype=float)
        mean = synthetic_summary["prevalence_sum"].to_numpy(dtype=float) / count
        sum_sq = synthetic_summary["prevalence_squared_sum"].to_numpy(dtype=float)
        variance = np.maximum((sum_sq - count * mean ** 2) / np.maximum(count - 1, 1), 0.0)
        sd = np.sqrt(variance)
        se = np.divide(sd, np.sqrt(count), out=np.zeros_like(sd), where=count > 0)
        critical = np.array([
            student_t.ppf(.975, int(n) - 1) if n > 1 else 0.0 for n in count
        ])
        half = critical * se
        synthetic_summary["prevalence_mean"] = mean
        synthetic_summary["prevalence_sd_across_realizations"] = sd
        synthetic_summary["prevalence_ci95_graph_low"] = np.clip(mean - half, 0.0, 1.0)
        synthetic_summary["prevalence_ci95_graph_high"] = np.clip(mean + half, 0.0, 1.0)
        synthetic_summary["mean_exposure"] = synthetic_summary["mean_exposure_sum"] / count
        synthetic_summary["total_exposure"] = synthetic_summary["total_exposure_sum"] / count
        synthetic_summary.to_csv(out / "synthetic_trajectory_summary.csv.gz",
                                 index=False, compression="gzip")


def _save_trajectory_summaries_and_plots(trajectory_path, out_dir):
    """Save seed-set/graph uncertainty trajectory tables and plots."""
    out = Path(out_dir)
    (out / "plots" / "real").mkdir(parents=True, exist_ok=True)
    (out / "plots" / "synthetic").mkdir(parents=True, exist_ok=True)
    _save_trajectory_summaries(trajectory_path, out)

    real_summary_path = out / "real_trajectory_summary.csv.gz"
    if real_summary_path.exists():
        real_summary = pd.read_csv(real_summary_path)
        for (representation, scenario), group in real_summary.groupby(["representation", "scenario"]):
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for year, year_group in group.groupby("year"):
                year_group = year_group.sort_values("time")
                x = year_group["time"].to_numpy()
                mean = year_group["prevalence_mean"].to_numpy()
                low = year_group["prevalence_ci95_seed_low"].to_numpy()
                high = year_group["prevalence_ci95_seed_high"].to_numpy()
                ax.plot(x, mean, label=str(int(year)))
                ax.fill_between(x, low, high, alpha=.15)
            ax.set_xlabel("Discrete time step")
            ax.set_ylabel("Mean prevalence")
            ax.set_ylim(0, 1)
            ax.set_title(f"Real ICIO SIS trajectory: {representation}, {scenario}")
            ax.legend(title="ICIO year")
            ax.grid(alpha=.25)
            fig.tight_layout()
            safe_scenario = str(scenario).replace(".", "p")
            fig.savefig(out / "plots" / "real" /
                        f"trajectory_{representation}_{safe_scenario}.png", dpi=180)
            plt.close(fig)

    synthetic_summary_path = out / "synthetic_trajectory_summary.csv.gz"
    if synthetic_summary_path.exists():
        synthetic_summary = pd.read_csv(synthetic_summary_path)
        for (family, representation, scenario), subset in synthetic_summary.groupby(
                ["family", "representation", "scenario"]):
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for configuration, config_group in subset.groupby("configuration"):
                config_group = config_group.sort_values("time")
                x = config_group["time"].to_numpy()
                mean = config_group["prevalence_mean"].to_numpy()
                low = config_group["prevalence_ci95_graph_low"].to_numpy()
                high = config_group["prevalence_ci95_graph_high"].to_numpy()
                ax.plot(x, mean, label=f"Configuration {int(configuration)}")
                ax.fill_between(x, low, high, alpha=.15)
            ax.set_xlabel("Discrete time step")
            ax.set_ylabel("Mean prevalence")
            ax.set_ylim(0, 1)
            ax.set_title(f"Synthetic SIS trajectory: {family}, {representation}, {scenario}")
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            safe_scenario = str(scenario).replace(".", "p")
            fig.savefig(out / "plots" / "synthetic" /
                        f"trajectory_{family}_{representation}_{safe_scenario}.png", dpi=180)
            plt.close(fig)


def _write_ensemble_summary(frame, group_cols, output_path):
    """Compute graph-to-graph CIs for all principal SIS metrics."""
    metric_columns = {
        "peak_prevalence": ("peak_prevalence_mean_across_seed_sets", (0.0, 1.0)),
        "time_to_peak": ("time_to_peak_mean_across_seed_sets", None),
        "long_run_prevalence": ("long_run_prevalence_mean_across_seed_sets", (0.0, 1.0)),
        "cumulative_prevalence": ("cumulative_prevalence_mean_across_seed_sets", None),
        "final_prevalence": ("final_prevalence_mean_across_seed_sets", (0.0, 1.0)),
        "max_mean_exposure": ("max_mean_exposure_mean_across_seed_sets", None),
        "max_total_exposure": ("max_total_exposure_mean_across_seed_sets", None),
        "extinction_fraction": ("extinction_fraction", (0.0, 1.0)),
        "convergence_fraction": ("convergence_fraction", (0.0, 1.0)),
    }
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["independent_synthetic_realizations"] = int(group["network_id"].nunique())
        for output_name, (source_column, bounds) in metric_columns.items():
            if source_column not in group:
                continue
            if output_name in ("extinction_fraction", "convergence_fraction"):
                values = (
                    group["extinction_fraction"] if output_name == "extinction_fraction"
                    else group["convergence_fraction"]
                ).to_numpy(dtype=float)
            else:
                values = group[source_column].to_numpy(dtype=float)
            mean, sd, low, high, count = _ci(values, bounds=bounds)
            row[f"{output_name}_mean_across_graph_realizations"] = mean
            row[f"{output_name}_sd_across_graph_realizations"] = sd
            row[f"{output_name}_ci95_low_across_graph_realizations"] = low
            row[f"{output_name}_ci95_high_across_graph_realizations"] = high
            row[f"{output_name}_valid_graph_count"] = int(count)
        row["uncertainty_note"] = (
            "Within-network seed-set uncertainty is reported in dynamics_network_results.csv.gz; "
            "these intervals describe variation across independent graph realizations."
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)

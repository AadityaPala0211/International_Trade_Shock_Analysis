"""Manifest-driven validation for empirical and reduced-size synthetic ensembles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz

from .config import FAMILIES, LEGACY_REPRESENTATIONS as REPRESENTATIONS, YEARS


def _load_manifest(path, base_dir, expected_type, required_columns, errors):
    if not path.is_file():
        errors.append(f"Missing {expected_type} manifest: {path}")
        return []
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Could not read {expected_type} manifest {path}: {exc}")
        return []
    if frame.empty:
        errors.append(f"Empty {expected_type} manifest: {path}")
        return []
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        errors.append(f"{expected_type} manifest missing columns: {missing}")
        return []
    if frame["network_id"].duplicated().any():
        errors.append(f"{expected_type} manifest contains duplicate network_id values.")
    records = []
    for row in frame.to_dict(orient="records"):
        record = dict(row)
        record["network_type"] = expected_type
        record["base_dir"] = str(base_dir)
        records.append(record)
    return records


def _matrix_path(record):
    path = Path(str(record["matrix_path"]))
    if not path.is_absolute():
        path = Path(record["base_dir"]) / path
    return path.resolve()


def _validate_one_matrix(record, expected_nodes):
    path = _matrix_path(record)
    if not path.is_file():
        raise FileNotFoundError(path)
    matrix = load_npz(path).tocsr()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: {path} shape={matrix.shape}")
    if expected_nodes is not None and matrix.shape != (int(expected_nodes), int(expected_nodes)):
        raise ValueError(
            f"Unexpected matrix shape for {path}: {matrix.shape}; "
            f"expected {(int(expected_nodes), int(expected_nodes))}"
        )
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Non-finite matrix entries: {path}")
    if np.any(matrix.data < 0):
        raise ValueError(f"Negative matrix entries: {path}")
    if np.any(matrix.diagonal() != 0):
        raise ValueError(f"Self-loops found: {path}")
    representation = str(record["representation"])
    if representation in ("UU", "UD") and matrix.nnz:
        if not np.allclose(matrix.data, 1.0):
            raise ValueError(f"{representation} matrix is not binary: {path}")
    if representation == "UU" and (matrix != matrix.T).nnz:
        raise ValueError(f"UU matrix is not symmetric: {path}")
    if representation == "WD" and matrix.nnz:
        if np.any(matrix.data <= 0) or np.any(matrix.data > 1.0 + 1e-8):
            raise ValueError(f"WD entries must be positive dependency shares <= 1: {path}")
    return matrix, {
        "network_id": record["network_id"],
        "network_type": record["network_type"],
        "family": record.get("family"),
        "year": record.get("year"),
        "configuration": record.get("configuration"),
        "level": record.get("level"),
        "realization": record.get("realization"),
        "representation": representation,
        "n_nodes": int(matrix.shape[0]),
        "edges_or_nnz": int(matrix.nnz),
        "matrix_path": str(path),
        "status": "ok",
    }


def _validate_paired_group(group_key, matrices):
    missing = sorted(set(REPRESENTATIONS) - set(matrices))
    if missing:
        raise ValueError(f"{group_key}: missing representations {missing}")
    uu, ud, wd = matrices["UU"], matrices["UD"], matrices["WD"]
    expected_uu = ud.maximum(ud.T).tocsr()
    if (uu != expected_uu).nnz:
        raise ValueError(f"{group_key}: UU is not the undirected union of UD and UD.T")
    wd_support = wd.copy().tocsr()
    wd_support.data = np.ones(wd_support.nnz, dtype=np.uint8)
    wd_support.eliminate_zeros()
    if (wd_support != ud).nnz:
        raise ValueError(f"{group_key}: WD and UD do not have identical directed support")


def validate_network_manifests(out_dir, scope="all", expected_nodes=4050,
                               expected_realizations=None):
    """Validate manifest completeness, matrix contents, paired supports and stale files.

    Any positive realization count is accepted. Counts below two are valid for
    pilot runs but cannot support graph-to-graph variance or confidence intervals.
    Unmanifested NPZ files are reported as stale/orphaned warnings; dynamics only
    consumes paths explicitly listed in the current manifest.
    """
    if scope not in ("real", "synthetic", "all"):
        raise ValueError("scope must be real, synthetic or all")
    if expected_realizations is not None and int(expected_realizations) < 1:
        raise ValueError("expected_realizations must be a positive integer")

    out = Path(out_dir).resolve()
    errors, warnings, matrix_checks = [], [], []
    records_by_type = {"real": [], "synthetic": []}

    if scope in ("real", "all"):
        real = _load_manifest(
            out / "empirical_network_manifest.csv", out, "real",
            ("network_id", "year", "representation", "n_nodes", "matrix_path"), errors,
        )
        records_by_type["real"] = real
        keys = [(int(r["year"]), str(r["representation"])) for r in real]
        required_keys = [(year, rep) for year in YEARS for rep in REPRESENTATIONS]
        if len(real) != 6 or set(keys) != set(required_keys) or len(set(keys)) != len(keys):
            errors.append(
                "Empirical manifest must contain exactly one matrix for each "
                "2019/2022 x UU/UD/WD combination."
            )

    synthetic_realizations = {}
    synthetic_rows = 0
    orphan_count = 0
    if scope in ("synthetic", "all"):
        synthetic_root = out / "synthetic"
        synthetic = _load_manifest(
            synthetic_root / "synthetic_network_manifest.csv", synthetic_root, "synthetic",
            ("network_id", "family", "level", "configuration", "realization",
             "representation", "matrix_path"), errors,
        )
        records_by_type["synthetic"] = synthetic
        synthetic_rows = len(synthetic)
        config_keys = {(family, level) for family in FAMILIES for level in (1, 2, 3)}
        groups = {}
        seen_keys = set()
        for record in synthetic:
            try:
                family = str(record["family"])
                level = int(record["level"])
                realization = int(record["realization"])
                representation = str(record["representation"])
            except (TypeError, ValueError):
                errors.append(f"Invalid synthetic manifest identifiers in {record.get('network_id')}")
                continue
            key = (family, level, realization, representation)
            if key in seen_keys:
                errors.append(f"Duplicate synthetic manifest key: {key}")
            seen_keys.add(key)
            groups.setdefault((family, level, realization), {})[representation] = record
        present_configs = {(family, level) for family, level, _ in groups}
        if present_configs != config_keys:
            errors.append(
                "Synthetic manifest must contain all nine family/level configurations; "
                f"missing={sorted(config_keys - present_configs)}, extra={sorted(present_configs - config_keys)}"
            )

        counts = []
        for family, level in sorted(config_keys):
            realization_ids = sorted(
                realization for f, l, realization in groups if f == family and l == level
            )
            count = len(realization_ids)
            counts.append(count)
            if realization_ids != list(range(1, count + 1)):
                errors.append(
                    f"{family} level {level}: realization IDs must be contiguous from 1; "
                    f"found {realization_ids[:20]}"
                )
            for realization in realization_ids:
                reps = groups[(family, level, realization)]
                if set(reps) != set(REPRESENTATIONS):
                    errors.append(
                        f"{family} level {level} realization {realization}: "
                        f"expected UU/UD/WD exactly once, found {sorted(reps)}"
                    )
        nonzero_counts = [count for count in counts if count > 0]
        if len(nonzero_counts) == 9 and len(set(nonzero_counts)) == 1:
            actual_r = nonzero_counts[0]
            synthetic_realizations = {f"{family}_level_{level}": actual_r
                                      for family, level in sorted(config_keys)}
            if expected_realizations is not None and actual_r != int(expected_realizations):
                errors.append(
                    f"Synthetic manifest has {actual_r} realizations/configuration; "
                    f"--expected-realizations requested {int(expected_realizations)}."
                )
            if actual_r == 1:
                warnings.append(
                    "Only one synthetic realization/configuration: valid as a pilot, "
                    "but between-graph variance and graph-level confidence intervals are not estimable."
                )
            elif actual_r < 5:
                warnings.append(
                    f"Only {actual_r} realizations/configuration: exploratory ensemble; "
                    "graph-level confidence intervals will be imprecise."
                )
            elif actual_r < 10:
                warnings.append(
                    f"{actual_r} realizations/configuration: usable for exploratory comparisons, "
                    "but less precise than the recommended 10."
                )
            elif actual_r < 20:
                warnings.append(
                    f"{actual_r} realizations/configuration: practical reduced ensemble; "
                    "uncertainty is larger than with the planned 20."
                )
        else:
            errors.append(
                "All nine synthetic configurations must contain the same positive realization count; "
                f"counts={counts}"
            )

        for metadata_name, count_key in (
            ("synthetic_experiment_design.json", "realizations_per_configuration"),
            ("../metadata/synthetic_stage_complete.json", "realization_count_per_configuration"),
        ):
            metadata_path = synthetic_root / metadata_name
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    metadata_r = int(metadata[count_key])
                    if len(nonzero_counts) == 9 and metadata_r != nonzero_counts[0]:
                        errors.append(
                            f"{metadata_path.name} records {metadata_r} realizations, "
                            f"but the manifest records {nonzero_counts[0]}."
                        )
                except Exception as exc:
                    errors.append(f"Could not validate {metadata_path}: {exc}")

        tracked_paths = {_matrix_path(r) for r in synthetic}
        matrix_root = synthetic_root / "matrices"
        if matrix_root.is_dir():
            actual_paths = {path.resolve() for path in matrix_root.rglob("*.npz")}
            orphans = sorted(actual_paths - tracked_paths)
            orphan_count = len(orphans)
            if orphan_count:
                warnings.append(
                    f"Found {orphan_count} unmanifested synthetic NPZ files. They are ignored by SIS; "
                    "rerunning --stage synthetic now cleans these stale files before generation."
                )

    # Validate each matrix, then reload only one paired triplet at a time for
    # support checks. This bounds peak memory independently of ensemble size.
    paired_records = {}
    for network_type, records in records_by_type.items():
        if (network_type == "real" and scope not in ("real", "all")) or (
            network_type == "synthetic" and scope not in ("synthetic", "all")
        ):
            continue
        for record in records:
            try:
                matrix, check = _validate_one_matrix(record, expected_nodes)
                matrix_checks.append(check)
                del matrix
                if network_type == "real":
                    group_key = ("real", int(record["year"]))
                else:
                    group_key = ("synthetic", str(record["family"]),
                                 int(record["level"]), int(record["realization"]))
                paired_records.setdefault(group_key, {})[str(record["representation"])] = record
            except Exception as exc:
                errors.append(f"{record.get('network_id')}: {exc}")

    for group_key, group_records in paired_records.items():
        try:
            missing = sorted(set(REPRESENTATIONS) - set(group_records))
            if missing:
                raise ValueError(f"{group_key}: missing representations {missing}")
            matrices = {
                representation: load_npz(_matrix_path(record)).tocsr()
                for representation, record in group_records.items()
            }
            _validate_paired_group(group_key, matrices)
            del matrices
        except Exception as exc:
            errors.append(str(exc))

    report = {
        "ok": not errors,
        "scope": scope,
        "expected_nodes": int(expected_nodes) if expected_nodes is not None else None,
        "empirical_matrix_count": len(records_by_type["real"]),
        "synthetic_manifest_rows": int(synthetic_rows),
        "synthetic_realizations_per_configuration": synthetic_realizations,
        "independent_synthetic_skeletons": int(sum(synthetic_realizations.values())),
        "expected_synthetic_matrices_from_manifest": int(synthetic_rows),
        "verified_matrix_count": len(matrix_checks),
        "orphan_matrix_files": int(orphan_count),
        "matrix_checks": matrix_checks,
        "warnings": warnings,
        "errors": errors,
    }
    return report

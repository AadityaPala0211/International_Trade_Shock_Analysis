"""Robust loading and validation of OECD ICIO CSV exports.

Z[i,j] is the intermediate flow from supplier i to buyer j. ICIO exports may
contain multi-row headers; the loader detects a numeric square block and requires
the expected 4,050 labelled nodes instead of silently guessing its orientation.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

class ICIOFormatError(ValueError):
    pass

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _clean_label(x):
    return "" if pd.isna(x) else str(x).strip()

def _read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, dtype=str, low_memory=False)

def _is_numeric_token(x) -> bool:
    if pd.isna(x) or str(x).strip() == "":
        return False
    try:
        return bool(np.isfinite(float(str(x).replace(",", ""))))
    except (TypeError, ValueError):
        return False

def _find_contiguous_numeric_block(raw: pd.DataFrame, expected_nodes: int | None):
    nrows, ncols = raw.shape
    row_runs = []
    min_run = max(20, (expected_nodes or 100) // 3)
    for i in range(nrows):
        flags = np.fromiter((_is_numeric_token(v) for v in raw.iloc[i].tolist()), bool, count=ncols)
        idx = np.flatnonzero(flags)
        if len(idx) < min_run:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
        ends = np.r_[starts[1:], len(idx)]
        best = max(zip(starts, ends), key=lambda z: z[1]-z[0])
        run = idx[best[0]:best[1]]
        if len(run) >= min_run:
            row_runs.append((i, int(run[0]), int(run[-1])+1, len(run)))
    by_interval = {}
    for row, c0, c1, count in row_runs:
        by_interval.setdefault((c0, c1), []).append(row)
    candidates = []
    for (c0, c1), rows in by_interval.items():
        rows = sorted(rows)
        groups=[]; cur=[]
        for r in rows:
            if cur and r != cur[-1]+1:
                groups.append(cur); cur=[]
            cur.append(r)
        if cur:
            groups.append(cur)
        for g in groups:
            if len(g) == c1-c0 and (expected_nodes is None or len(g) == expected_nodes):
                candidates.append((g[0], g[-1]+1, c0, c1))
    if len(candidates) != 1:
        raise ICIOFormatError(
            f"Expected one {expected_nodes or 'square'} labelled numeric block; found {len(candidates)} exact candidates. "
            "This CSV layout requires an explicit parser adjustment; no block was guessed."
        )
    r0,r1,c0,c1=candidates[0]
    if c0 == 0:
        raise ICIOFormatError("No column available for country-industry row labels.")
    row_labels = [_clean_label(x) for x in raw.iloc[r0:r1, c0-1].tolist()]
    if len(row_labels) != len(set(row_labels)) or any(not x for x in row_labels):
        raise ICIOFormatError("Row labels are empty or duplicated in the detected block.")
    mat = raw.iloc[r0:r1, c0:c1].apply(lambda col: pd.to_numeric(col.str.replace(",", "", regex=False), errors="coerce"))
    if mat.isna().any().any():
        raise ICIOFormatError(f"Detected transaction block contains {int(mat.isna().sum().sum()):,} non-numeric cells.")
    return mat.to_numpy(dtype=np.float64), row_labels, (r0,r1,c0,c1), raw.shape

def load_icio(path: str | Path, expected_nodes: int | None = 4050, allow_expected_mismatch: bool = False):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw=_read_raw(path)
    Z, labels, bounds, raw_shape = _find_contiguous_numeric_block(raw, expected_nodes if not allow_expected_mismatch else None)
    if expected_nodes is not None and not allow_expected_mismatch and Z.shape != (expected_nodes, expected_nodes):
        raise ICIOFormatError(f"Expected {expected_nodes} nodes; extracted {Z.shape}.")
    if not np.isfinite(Z).all():
        raise ICIOFormatError("Non-finite flows in ICIO block.")
    if (Z < -1e-10).any():
        raise ICIOFormatError("Negative intermediate flow detected; review source conventions.")
    Z[Z < 0] = 0.0
    if len(set(labels)) != len(labels):
        raise ICIOFormatError("Node labels are not unique.")
    return {"Z": Z, "labels": labels, "source_path": str(path),
            "source_sha256": sha256_file(path), "raw_shape": raw_shape,
            "block_bounds_zero_based": bounds, "n_nodes": len(labels),
            "total_intermediate_flow": float(Z.sum()), "self_flow": float(np.trace(Z))}

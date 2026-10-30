"""Robust loading and validation of OECD ICIO CSV exports."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


class ICIOFormatError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_label(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, dtype=str, low_memory=False)


def _is_numeric_token(value) -> bool:
    if pd.isna(value) or str(value).strip() == "":
        return False
    try:
        number = float(str(value).replace(",", ""))
        return bool(np.isfinite(number))
    except (TypeError, ValueError):
        return False


def _find_contiguous_numeric_block(raw: pd.DataFrame, expected_nodes: int):
    """Extract the leading N x N intermediate block and align its column labels.

    Rows denote suppliers and columns denote buyers. When the row-label and
    column-label sets match but their orders differ, columns are reordered to
    the row-label order before returning the matrix.
    """
    nrows, ncols = raw.shape
    N = int(expected_nodes)
    if N <= 0:
        raise ValueError("expected_nodes must be positive")
    min_run = max(2, min(N, max(20, N // 3)))
    row_runs = []
    for i in range(nrows):
        flags = np.fromiter((_is_numeric_token(v) for v in raw.iloc[i].tolist()),
                            bool, count=ncols)
        idx = np.flatnonzero(flags)
        if len(idx) < min_run:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
        ends = np.r_[starts[1:], len(idx)]
        runs = [idx[a:b] for a, b in zip(starts, ends) if b - a >= min_run]
        if runs:
            run = min(runs, key=lambda x: int(x[0]))
            row_runs.append((i, int(run[0]), int(run[-1]) + 1, len(run)))
    if not row_runs:
        raise ICIOFormatError("Could not identify numeric transaction rows. Check the CSV edition/layout.")

    by_start = {}
    for row, c0, c1, count in row_runs:
        by_start.setdefault(c0, []).append(row)
    candidates = []
    for c0, rows in by_start.items():
        rows = sorted(rows)
        groups, current = [], []
        for row in rows:
            if current and row != current[-1] + 1:
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)
        for group in groups:
            if len(group) < N or c0 + N > ncols or c0 == 0:
                continue
            r0 = group[0]
            r1 = r0 + N
            row_labels = [_clean_label(x) for x in raw.iloc[r0:r1, c0 - 1].tolist()]
            if len(row_labels) != N or any(not x for x in row_labels) or len(set(row_labels)) != N:
                continue
            header_labels = [_clean_label(x) for x in raw.iloc[r0 - 1, c0:c0 + N].tolist()] if r0 > 0 else []
            header_nonempty = len(header_labels) == N and all(header_labels)
            header_unique = header_nonempty and len(set(header_labels)) == N
            if header_unique and set(header_labels) != set(row_labels):
                continue
            exact_match = header_unique and header_labels == row_labels
            score = 2 if exact_match else (1 if header_unique else 0)
            candidates.append((score, r0, r1, c0, c0 + N, row_labels, header_labels))

    if not candidates:
        raise ICIOFormatError(
            f"Could not identify a {N} x {N} numeric intermediate block with {N} unique row labels. "
            "The CSV layout may require an edition-specific parser."
        )
    best_score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_score]
    if len(best) != 1:
        raise ICIOFormatError(f"Found {len(best)} equally plausible ICIO numeric blocks; parsing is ambiguous.")

    _, r0, r1, c0, c1, row_labels, header_labels = best[0]
    block = raw.iloc[r0:r1, c0:c1].apply(
        lambda col: pd.to_numeric(col.str.replace(",", "", regex=False), errors="coerce")
    )
    if block.isna().any().any():
        count = int(block.isna().sum().sum())
        raise ICIOFormatError(f"Detected transaction block contains {count:,} non-numeric cells.")
    if len(header_labels) == N and all(header_labels) and len(set(header_labels)) == N:
        col_position = {label: index for index, label in enumerate(header_labels)}
        if set(col_position) == set(row_labels):
            order = [col_position[label] for label in row_labels]
            block = block.iloc[:, order]
    return block.to_numpy(dtype=np.float64), row_labels, (r0, r1, c0, c1), raw.shape


def load_icio(path: str | Path, expected_nodes: int = 4050):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _read_raw(path)
    Z, labels, bounds, raw_shape = _find_contiguous_numeric_block(raw, expected_nodes)
    if Z.shape != (expected_nodes, expected_nodes):
        raise ICIOFormatError(f"Expected {expected_nodes} nodes; extracted {Z.shape}.")
    if not np.isfinite(Z).all():
        raise ICIOFormatError("Non-finite flows in ICIO block.")
    if (Z < -1e-10).any():
        raise ICIOFormatError("Negative intermediate flow detected; review source conventions.")
    Z[Z < 0] = 0.0
    if len(set(labels)) != len(labels):
        raise ICIOFormatError("Node labels are not unique.")
    return {
        "Z": Z, "labels": labels, "source_path": str(path),
        "source_sha256": sha256_file(path), "raw_shape": raw_shape,
        "block_bounds_zero_based": bounds, "n_nodes": len(labels),
        "total_intermediate_flow": float(Z.sum()), "self_flow": float(np.trace(Z)),
    }

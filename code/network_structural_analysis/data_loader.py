"""Robust loading and validation of OECD ICIO CSV exports.

Z[i,j] is the intermediate flow from supplier i to buyer j. ICIO exports can
append final-demand columns and accounting rows; this parser extracts the
leading N x N intermediate transaction block after label and shape checks.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

class ICIOFormatError(ValueError):
    pass

def sha256_file(path: Path,chunk_size: int=8*1024*1024) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(chunk_size),b""):
            h.update(chunk)
    return h.hexdigest()

def _clean_label(x):
    return "" if pd.isna(x) else str(x).strip()

def _read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path,header=None,dtype=str,low_memory=False)

def _is_numeric_token(x) -> bool:
    if pd.isna(x) or str(x).strip()=="":
        return False
    try:
        return bool(np.isfinite(float(str(x).replace(",",""))))
    except (TypeError,ValueError):
        return False

def _find_contiguous_numeric_block(raw: pd.DataFrame,expected_nodes: int):
    """Extract the leading N×N intermediate-input block from an ICIO CSV.

    ICIO exports may append final-demand columns to the right and rows below
    the intermediate block. Locate contiguous rows with at least N numeric
    cells starting at a common left edge, then select their leading N×N block.
    Header/row-label agreement disambiguates the selected block where possible.
    """
    nrows,ncols=raw.shape; N=int(expected_nodes)
    if N<=0: raise ValueError("expected_nodes must be positive")
    min_run=max(2,min(N,max(20,N//3)))
    row_runs=[]
    for i in range(nrows):
        flags=np.fromiter((_is_numeric_token(v) for v in raw.iloc[i].tolist()),bool,count=ncols)
        idx=np.flatnonzero(flags)
        if len(idx)<min_run: continue
        starts=np.r_[0,np.flatnonzero(np.diff(idx)>1)+1]
        ends=np.r_[starts[1:],len(idx)]
        runs=[idx[a:b] for a,b in zip(starts,ends) if b-a>=min_run]
        if runs:
            run=min(runs,key=lambda x:int(x[0]))
            row_runs.append((i,int(run[0]),int(run[-1])+1,len(run)))
    if not row_runs:
        raise ICIOFormatError("Could not identify numeric transaction rows. Check the CSV edition/layout.")
    by_start={}
    for row,c0,c1,count in row_runs:
        by_start.setdefault(c0,[]).append(row)
    candidates=[]
    for c0,rows in by_start.items():
        rows=sorted(rows); groups=[]; current=[]
        for r in rows:
            if current and r!=current[-1]+1:
                groups.append(current); current=[]
            current.append(r)
        if current: groups.append(current)
        for group in groups:
            if len(group)<N or c0+N>ncols: continue
            candidates.append((group[0],group[0]+N,c0,c0+N))
    scored=[]
    for r0,r1,c0,c1 in candidates:
        row_labels=[_clean_label(x) for x in raw.iloc[r0:r1,c0-1].tolist()] if c0>0 else []
        if len(row_labels)!=N or any(not x for x in row_labels) or len(set(row_labels))!=N:
            continue
        header_labels=[_clean_label(x) for x in raw.iloc[r0-1,c0:c1].tolist()] if r0>0 else []
        exact=(len(header_labels)==N and header_labels==row_labels)
        populated=(len(header_labels)==N and all(header_labels))
        score=2 if exact else (0 if not populated else -2)
        scored.append((score,r0,r1,c0,c1,row_labels))
    if not scored:
        raise ICIOFormatError(f"Could not identify a {N}x{N} numeric block with {N} unique row labels.")
    best_score=max(x[0] for x in scored)
    best=[x for x in scored if x[0]==best_score]
    if len(best)!=1:
        raise ICIOFormatError(f"Found {len(best)} equally plausible ICIO numeric blocks; parsing is ambiguous.")
    _,r0,r1,c0,c1,row_labels=best[0]
    if c0==0: raise ICIOFormatError("No column available for country-industry row labels.")
    mat=raw.iloc[r0:r1,c0:c1].apply(lambda col:pd.to_numeric(col.str.replace(",","",regex=False),errors="coerce"))
    if mat.isna().any().any():
        raise ICIOFormatError(f"Detected transaction block contains {int(mat.isna().sum().sum()):,} non-numeric cells.")
    return mat.to_numpy(dtype=np.float64),row_labels,(r0,r1,c0,c1),raw.shape

def load_icio(path: str|Path,expected_nodes: int=4050):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    raw=_read_raw(path)
    Z,labels,bounds,raw_shape=_find_contiguous_numeric_block(raw,expected_nodes)
    if Z.shape!=(expected_nodes,expected_nodes):
        raise ICIOFormatError(f"Expected {expected_nodes} nodes; extracted {Z.shape}.")
    if not np.isfinite(Z).all(): raise ICIOFormatError("Non-finite flows in ICIO block.")
    if (Z < -1e-10).any(): raise ICIOFormatError("Negative intermediate flow detected; review source conventions.")
    Z[Z<0]=0.0
    if len(set(labels))!=len(labels): raise ICIOFormatError("Node labels are not unique.")
    return {"Z":Z,"labels":labels,"source_path":str(path),"source_sha256":sha256_file(path),
            "raw_shape":raw_shape,"block_bounds_zero_based":bounds,"n_nodes":len(labels),
            "total_intermediate_flow":float(Z.sum()),"self_flow":float(np.trace(Z))}

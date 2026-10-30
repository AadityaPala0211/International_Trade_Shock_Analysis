"""Dependency normalisation and paired empirical network representations."""
from __future__ import annotations
import numpy as np
from scipy import sparse

def dependency_shares(Z: np.ndarray):
    """S[i,j] = Z[i,j] / total intermediate purchases of buyer j.

    Denominator includes self-use; off-diagonal graph edge candidates exclude it.
    """
    Z=np.asarray(Z,dtype=np.float64)
    if Z.ndim!=2 or Z.shape[0]!=Z.shape[1]:
        raise ValueError("Z must be square")
    if not np.isfinite(Z).all() or (Z<0).any():
        raise ValueError("Z must be finite and nonnegative")
    purchases=Z.sum(axis=0)
    S=np.divide(Z,purchases[None,:],out=np.zeros_like(Z),where=purchases[None,:]>0)
    np.fill_diagonal(S,0.0)
    return S,purchases

def make_representations(S: np.ndarray,tau: float):
    """Create UU, UD and WD with supplier rows -> buyer columns."""
    if not np.isfinite(float(tau)) or not 0 <= float(tau) <= 1:
        raise ValueError("dependency-share threshold tau must be finite and in [0, 1]")
    mask=(S>=tau)&(S>0)
    np.fill_diagonal(mask,False)
    row,col=np.nonzero(mask)
    shape=S.shape
    ud=sparse.csr_matrix((np.ones(len(row),dtype=np.uint8),(row,col)),shape=shape)
    wd=sparse.csr_matrix((S[row,col],(row,col)),shape=shape)
    uu=ud.maximum(ud.T).tocsr()
    uu.setdiag(0); uu.eliminate_zeros()
    return {"UU":uu,"UD":ud,"WD":wd}

def raw_flow_representations(Z: np.ndarray,tau: float):
    S,purchases=dependency_shares(Z)
    return make_representations(S,tau),S,purchases

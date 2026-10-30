"""Dependency normalisation and paired empirical network representations."""
import numpy as np
from scipy import sparse

def dependency_shares(Z: np.ndarray):
    """S[i,j] = Z[i,j] / buyer j's total intermediate purchases.

    Purchases retain diagonal self-use in the denominator; propagation graph
    numerators have no diagonal self-links.
    """
    Z=np.asarray(Z,dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError("Z must be square")
    if not np.isfinite(Z).all() or (Z < 0).any():
        raise ValueError("Z must be finite and nonnegative")
    purchases=Z.sum(axis=0)
    S=np.divide(Z,purchases[None,:],out=np.zeros_like(Z),where=purchases[None,:]>0)
    np.fill_diagonal(S,0.0)
    return S,purchases

def make_representations(S: np.ndarray, tau: float):
    """Create UU, UD, WD using a common dependency-share cutoff.

    UD/WD direction is supplier row -> buyer column. UU is the undirected
    projection of support in either direction.
    """
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    mask=(S >= tau) & (S > 0)
    np.fill_diagonal(mask,False)
    wd=sparse.csr_matrix(np.where(mask,S,0.0))
    ud=sparse.csr_matrix(mask.astype(np.uint8))
    uu=sparse.csr_matrix((mask | mask.T).astype(np.uint8))
    uu.setdiag(0); uu.eliminate_zeros()
    return {"UU":uu,"UD":ud,"WD":wd}

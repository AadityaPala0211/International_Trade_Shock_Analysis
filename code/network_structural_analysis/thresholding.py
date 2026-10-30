"""Pooled threshold diagnostics and an inspectable common-tau heuristic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from .network_builder import dependency_shares

def _quick_topology(mask):
    A=sparse.csr_matrix(mask.astype(np.uint8)); n=A.shape[0]; m=A.nnz
    density=m/(n*(n-1)) if n>1 else 0.0
    mutual=A.multiply(A.T).nnz
    reciprocity=mutual/m if m else 0.0
    U=A.maximum(A.T)
    nc,lab=connected_components(U,directed=False,return_labels=True)
    sizes=np.bincount(lab) if n else np.array([])
    giant=sizes.max()/n if len(sizes) else 0.0
    out=np.asarray(A.sum(axis=1)).ravel(); mean=out.mean()
    cv=out.std()/mean if mean else 0.0
    return density,reciprocity,giant,cv

def diagnose_thresholds(year_matrices: dict[int,np.ndarray], candidates, out_dir=None):
    records=[]; shares_by_year={}
    for year,Z in year_matrices.items():
        S,purchases=dependency_shares(Z); shares_by_year[year]=S
        offdiag_total=float(Z.sum()-np.trace(Z))
        positive_count=max(1,int((S>0).sum()))
        for tau in candidates:
            mask=(S>=tau)&(S>0); np.fill_diagonal(mask,False)
            retained_edges=int(mask.sum())
            retained_flow=float(np.where(mask,Z,0.0).sum())
            flow_cov=retained_flow/offdiag_total if offdiag_total>0 else 0.0
            density,recip,giant,cv=_quick_topology(mask)
            records.append({"year":year,"tau":float(tau),"edges":retained_edges,
                "edge_retention":retained_edges/positive_count,
                "offdiag_flow_coverage":flow_cov,"density":density,"reciprocity":recip,
                "giant_component_fraction":giant,"out_degree_cv":cv})
    df=pd.DataFrame(records)
    pooled=df.groupby("tau",as_index=False).agg(
        mean_flow_coverage=("offdiag_flow_coverage","mean"),
        min_flow_coverage=("offdiag_flow_coverage","min"),
        mean_density=("density","mean"),mean_giant_component=("giant_component_fraction","mean"),
        mean_reciprocity=("reciprocity","mean"),mean_edges=("edges","mean"))
    eligible=pooled[(pooled.min_flow_coverage>=0.90)&(pooled.mean_density<=0.20)]
    if len(eligible):
        connected=eligible[eligible.mean_giant_component>=0.90]
        pool=connected if len(connected) else eligible
        recommendation=float(pool.tau.max())
        rationale=("Largest tested threshold meeting >=90% flow coverage in both years, "
                    "<=20% mean density and, when available, >=90% mean giant-component coverage.")
    else:
        target=pooled.copy()
        target["score"]=(target.mean_flow_coverage-0.95).abs()+np.maximum(0,target.mean_density-0.20)*2
        recommendation=float(target.sort_values(["score","tau"]).iloc[0].tau)
        rationale="No candidate met all preferred constraints; selected closest to 95% mean flow coverage with density penalty."
    summary={"recommended_threshold":recommendation,"rationale":rationale,
             "candidate_thresholds":[float(x) for x in candidates],"is_unique_optimum":False,
             "warning":"Transparent sparsification heuristic, not a mathematically unique or economically observed threshold."}
    if out_dir:
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
        df.to_csv(out/"threshold_diagnostics.csv",index=False)
        pooled.to_csv(out/"threshold_pooled_summary.csv",index=False)
        (out/"threshold_recommendation.json").write_text(json.dumps(summary,indent=2))
        _plot_thresholds(df,pooled,out)
    return df,pooled,summary,shares_by_year

def _plot_thresholds(df,pooled,out):
    for col,ylabel,filename in [
      ("offdiag_flow_coverage","Retained off-diagonal monetary-flow coverage","threshold_flow_coverage.png"),
      ("density","Directed graph density","threshold_density.png"),
      ("giant_component_fraction","Giant weak-component fraction","threshold_giant_component.png"),
      ("reciprocity","Directed reciprocity","threshold_reciprocity.png")]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for year,g in df.groupby("year"):
            ax.plot(g.tau.replace(0,1e-10),g[col],marker="o",label=str(year))
        ax.set_xscale("log"); ax.set_xlabel("Dependency-share threshold τ"); ax.set_ylabel(ylabel)
        ax.grid(True,alpha=.25); ax.legend(title="ICIO edition"); fig.tight_layout()
        fig.savefig(out/filename,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(pooled.mean_density,pooled.mean_flow_coverage,marker="o")
    ax.set_xlabel("Mean directed density"); ax.set_ylabel("Mean retained flow coverage")
    ax.grid(True,alpha=.25); fig.tight_layout()
    fig.savefig(out/"threshold_density_coverage_tradeoff.png",dpi=180); plt.close(fig)

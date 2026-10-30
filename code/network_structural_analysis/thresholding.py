"""Pooled dependency threshold diagnostics and local sensitivity checks."""
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
    _,labels=connected_components(U,directed=False,return_labels=True)
    sizes=np.bincount(labels) if n else np.array([])
    giant=sizes.max()/n if len(sizes) else 0.0
    out=np.asarray(A.sum(axis=1)).ravel(); mean=out.mean()
    cv=out.std()/mean if mean else 0.0
    return density,reciprocity,giant,cv

def diagnose_thresholds(year_matrices,candidates,out_dir=None,min_flow_coverage=0.90,
                        max_density=0.20,min_giant_component=0.90,
                        local_sensitivity_factor=3.0):
    """Evaluate a common tau over all years and return a transparent recommendation.

    The score is deliberately a heuristic, not a claimed unique mathematical
    optimum. The complete diagnostic table and nearby-tau sensitivity are saved.
    """
    records=[]; shares_by_year={}
    for year,Z in year_matrices.items():
        S,purchases=dependency_shares(Z); shares_by_year[year]=S
        offdiag_total=float(Z.sum()-np.trace(Z))
        positive_count=max(1,int((S>0).sum()-np.count_nonzero(np.diag(S))))
        for tau in candidates:
            mask=(S>=tau)&(S>0); np.fill_diagonal(mask,False)
            edges=int(mask.sum())
            retained_flow=float(np.sum(Z,where=mask))
            coverage=retained_flow/offdiag_total if offdiag_total>0 else 0.0
            density,reciprocity,giant,cv=_quick_topology(mask)
            records.append({"year":year,"tau":float(tau),"edges":edges,
                "edge_retention":edges/positive_count,"offdiag_flow_coverage":coverage,
                "density":density,"reciprocity":reciprocity,
                "giant_component_fraction":giant,"out_degree_cv":cv})
    diagnostics=pd.DataFrame(records)
    pooled=diagnostics.groupby("tau",as_index=False).agg(
        mean_flow_coverage=("offdiag_flow_coverage","mean"),
        min_flow_coverage=("offdiag_flow_coverage","min"),
        mean_density=("density","mean"),
        mean_giant_component=("giant_component_fraction","mean"),
        mean_reciprocity=("reciprocity","mean"),mean_edges=("edges","mean"))
    eligible=pooled[(pooled.min_flow_coverage>=min_flow_coverage)&
                    (pooled.mean_density<=max_density)]
    if len(eligible):
        connected=eligible[eligible.mean_giant_component>=min_giant_component]
        pool=connected if len(connected) else eligible
        recommendation=float(pool.tau.max())
        rationale=(f"Largest tested threshold meeting >= {min_flow_coverage:.0%} flow coverage in both years, "
                   f"<= {max_density:.0%} mean density and, when available, >= {min_giant_component:.0%} "
                   "mean giant-component coverage.")
    else:
        target=pooled.copy()
        target_coverage=min(1.0,min_flow_coverage+0.05)
        target["score"]=(target.mean_flow_coverage-target_coverage).abs()+np.maximum(
            0,target.mean_density-max_density)*2
        recommendation=float(target.sort_values(["score","tau"]).iloc[0].tau)
        rationale=(f"No candidate met all preferred constraints; selected the candidate nearest "
                   f"{target_coverage:.0%} mean flow coverage with a density penalty. Analyst review required.")
    if recommendation>0:
        local_candidates=sorted(set([max(0.0,recommendation/local_sensitivity_factor),
            recommendation,min(1.0,recommendation*local_sensitivity_factor)]))
    else:
        positive=sorted(float(x) for x in candidates if float(x)>0)
        local_candidates=[0.0]+positive[:2]
    local_records=[]
    for year,Z in year_matrices.items():
        S=shares_by_year[year]; offdiag_total=float(Z.sum()-np.trace(Z))
        positive_count=max(1,int((S>0).sum()-np.count_nonzero(np.diag(S))))
        for tau in local_candidates:
            mask=(S>=tau)&(S>0); np.fill_diagonal(mask,False)
            edges=int(mask.sum()); flow=float(np.sum(Z,where=mask))
            density,reciprocity,giant,cv=_quick_topology(mask)
            local_records.append({"year":year,"tau":tau,"edges":edges,
                "edge_retention":edges/positive_count,
                "offdiag_flow_coverage":flow/offdiag_total if offdiag_total else 0.0,
                "density":density,"reciprocity":reciprocity,
                "giant_component_fraction":giant,"out_degree_cv":cv})
    local_df=pd.DataFrame(local_records)
    recommendation_info={"recommended_threshold":recommendation,"rationale":rationale,
        "candidate_thresholds":[float(x) for x in candidates],
        "local_sensitivity_thresholds":local_candidates,"is_unique_optimum":False,
        "constraints":{"min_flow_coverage":min_flow_coverage,"max_mean_density":max_density,
                       "min_mean_giant_component":min_giant_component},
        "warning":"A transparent sparsification heuristic, not a mathematically unique or economically observed threshold."}
    if out_dir:
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
        diagnostics.to_csv(out/"threshold_diagnostics.csv",index=False)
        pooled.to_csv(out/"threshold_pooled_summary.csv",index=False)
        local_df.to_csv(out/"threshold_local_sensitivity.csv",index=False)
        (out/"threshold_recommendation.json").write_text(json.dumps(recommendation_info,indent=2))
        _plot_thresholds(diagnostics,pooled,out)
    return diagnostics,pooled,recommendation_info,shares_by_year

def _plot_thresholds(df,pooled,out):
    series=[
        ("offdiag_flow_coverage","Retained off-diagonal monetary-flow coverage","threshold_flow_coverage.png"),
        ("density","Directed graph density","threshold_density.png"),
        ("giant_component_fraction","Giant weak-component fraction","threshold_giant_component.png"),
        ("reciprocity","Directed reciprocity","threshold_reciprocity.png")]
    for col,ylabel,filename in series:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for year,g in df.groupby("year"):
            ax.plot(g.tau.replace(0,1e-10),g[col],marker="o",label=str(year))
        ax.set_xscale("log"); ax.set_xlabel("Dependency-share threshold τ")
        ax.set_ylabel(ylabel); ax.grid(True,alpha=.25); ax.legend(title="ICIO edition")
        fig.tight_layout(); fig.savefig(out/filename,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(pooled.mean_density,pooled.mean_flow_coverage,marker="o")
    for _,row in pooled.iterrows():
        ax.annotate(f"τ={row.tau:g}",(row.mean_density,row.mean_flow_coverage),fontsize=7)
    ax.set_xlabel("Mean directed density"); ax.set_ylabel("Mean retained flow coverage")
    ax.grid(True,alpha=.25); fig.tight_layout()
    fig.savefig(out/"threshold_density_coverage_tradeoff.png",dpi=180); plt.close(fig)

"""CLI entry point for network generation and structural analysis only."""
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, networkx, matplotlib
from .config import YEARS, REPRESENTATIONS
from .data_loader import load_icio
from .network_builder import make_representations
from .thresholding import diagnose_thresholds
from .metrics import structural_analysis, country_sector_summary
from .synthetic import empirical_calibration, run_synthetic_ensemble
from .compare import compare_node_metrics, top_node_table
from .visualization import plot_degree_distribution, plot_strength_distribution, plot_metric_comparison, plot_synthetic_comparison

def _json_write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    def default(x):
        if isinstance(x,np.floating): return float(x)
        if isinstance(x,np.integer): return int(x)
        return str(x)
    Path(path).write_text(json.dumps(obj,indent=2,default=default))

def _save_empirical(year,reps,labels,base_out,seed,path_sample_sources,betweenness_k):
    summaries=[]; metrics_by_rep={}
    for rep,A in reps.items():
        target=base_out/"empirical"/str(year)/rep; target.mkdir(parents=True,exist_ok=True)
        from scipy.sparse import save_npz
        save_npz(target/"adjacency.npz",A)
        sm,node=structural_analysis(A,labels,rep,seed,path_sample_sources,betweenness_k)
        sm["year"]=year; summaries.append(sm); metrics_by_rep[rep]=node
        node.to_csv(target/"node_metrics.csv.gz",index=False,compression="gzip")
        coo=A.tocoo()
        edge=pd.DataFrame({"supplier_node_id":coo.row,"buyer_node_id":coo.col,
            "supplier":np.asarray(labels,dtype=object)[coo.row],
            "buyer":np.asarray(labels,dtype=object)[coo.col],"weight":coo.data})
        edge.to_csv(target/"edge_list.csv.gz",index=False,compression="gzip")
        country,sector=country_sector_summary(node)
        country.to_csv(target/"country_summary.csv",index=False)
        sector.to_csv(target/"sector_summary.csv",index=False)
        plot_degree_distribution(node,f"{year} {rep}: degree distribution",target/"degree_distribution.png")
        if rep=="WD": plot_strength_distribution(node,f"{year} WD: dependency strength",target/"strength_distribution.png")
    return summaries,metrics_by_rep

def _inspect(data_dir,expected_nodes):
    info={}
    for year in YEARS:
        p=Path(data_dir)/f"{year}_SML.csv"
        if not p.exists(): raise FileNotFoundError(p)
        raw=pd.read_csv(p,header=None,nrows=4,dtype=str)
        info[str(year)]={"path":str(p),"bytes":p.stat().st_size,
            "first_rows_preview":raw.fillna("").iloc[:3,:8].values.tolist()}
    print(json.dumps({"expected_nodes":expected_nodes,"files":info},indent=2))

def run(args=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",default="data"); ap.add_argument("--out",default="results_network_analysis")
    ap.add_argument("--threshold",type=float,default=None); ap.add_argument("--threshold-only",action="store_true")
    ap.add_argument("--inspect-only",action="store_true"); ap.add_argument("--expected-nodes",type=int,default=4050)
    ap.add_argument("--realizations",type=int,default=20); ap.add_argument("--seed",type=int,default=20261030)
    ap.add_argument("--skip-synthetic",action="store_true"); ap.add_argument("--synthetic-only",action="store_true")
    ap.add_argument("--path-sample-sources",type=int,default=256); ap.add_argument("--betweenness-k",type=int,default=256)
    ap.add_argument("--tau-candidates",default="0,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2")
    ns=ap.parse_args(args); data_dir=Path(ns.data_dir); out=Path(ns.out); out.mkdir(parents=True,exist_ok=True)
    if ns.inspect_only: _inspect(data_dir,ns.expected_nodes); return
    loaded={year:load_icio(data_dir/f"{year}_SML.csv",expected_nodes=ns.expected_nodes) for year in YEARS}
    if loaded[2019]["labels"] != loaded[2022]["labels"]:
        raise ValueError("2019/2022 node label order differs; align explicitly rather than comparing by position.")
    Zs={y:loaded[y]["Z"] for y in YEARS}
    candidates=tuple(float(x) for x in ns.tau_candidates.split(","))
    diag,pooled,recommendation,shares_by_year=diagnose_thresholds(Zs,candidates,out/"threshold_review")
    tau=float(ns.threshold if ns.threshold is not None else recommendation["recommended_threshold"])
    _json_write(out/"metadata"/"run_metadata.json",{
        "years":list(YEARS),"node_count":ns.expected_nodes,"threshold":tau,
        "threshold_recommendation":recommendation,"threshold_overridden":ns.threshold is not None,
        "seed":ns.seed,"realizations_per_synthetic_configuration":ns.realizations,
        "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
        "scipy":scipy.__version__,"networkx":networkx.__version__,"matplotlib":matplotlib.__version__,
        "inputs":{str(y):{"path":loaded[y]["source_path"],"sha256":loaded[y]["source_sha256"],
            "shape":list(loaded[y]["Z"].shape),"raw_shape":list(loaded[y]["raw_shape"]),
            "block_bounds_zero_based":loaded[y]["block_bounds_zero_based"],
            "total_intermediate_flow":loaded[y]["total_intermediate_flow"],
            "self_flow":loaded[y]["self_flow"]} for y in YEARS},
        "node_label_sha256":hashlib.sha256("\n".join(loaded[2019]["labels"]).encode()).hexdigest(),
        "interpretation":"Dependency-share thresholding; no dynamics implemented."})
    if ns.threshold_only:
        print(json.dumps({"recommended_threshold":tau,"rationale":recommendation["rationale"],"out":str(out)},indent=2)); return
    summaries=[]; metrics_by_year_rep={}
    if not ns.synthetic_only:
        for year in YEARS:
            reps=make_representations(shares_by_year[year],tau)
            sy,ny=_save_empirical(year,reps,loaded[year]["labels"],out,ns.seed,ns.path_sample_sources,ns.betweenness_k)
            summaries.extend(sy)
            for rep,node in ny.items(): metrics_by_year_rep[(year,rep)]=node
        summary_df=pd.DataFrame(summaries); summary_df.to_csv(out/"empirical_network_summaries.csv",index=False)
        (out/"figures").mkdir(parents=True,exist_ok=True)
        plot_metric_comparison(summary_df,out/"figures"/"empirical_density_comparison.png","density")
        compare_node_metrics(metrics_by_year_rep).to_csv(out/"comparisons"/"node_metric_year_comparison.csv",index=False)
        for year in YEARS:
            for rep in REPRESENTATIONS:
                df=metrics_by_year_rep[(year,rep)]
                for metric in ["out_strength","in_strength","pagerank","betweenness","k_core"]:
                    top_node_table(df,metric,25).to_csv(out/"comparisons"/f"top_{metric}_{year}_{rep}.csv",index=False)
    if not ns.skip_synthetic:
        pos=[]
        for y in YEARS:
            vals=shares_by_year[y][shares_by_year[y]>0]
            if len(vals)>2_000_000:
                rng=np.random.default_rng(ns.seed+y); vals=rng.choice(vals,2_000_000,replace=False)
            pos.append(vals)
        pooled_weights=np.concatenate(pos)
        empirical=[]
        if summaries:
            empirical=[x for x in summaries if x["representation"]=="UU"]
        else:
            for y in YEARS:
                reps=make_representations(shares_by_year[y],tau)
                sm,_=structural_analysis(reps["UU"],loaded[y]["labels"],"UU",seed=ns.seed,
                    path_sample_sources=ns.path_sample_sources,betweenness_k=ns.betweenness_k)
                empirical.append(sm)
        calibration=empirical_calibration(empirical)
        _json_write(out/"metadata"/"synthetic_calibration.json",calibration)
        syn=run_synthetic_ensemble(ns.expected_nodes,calibration["reference_mean_degree"],
            calibration["reference_reciprocity"],pooled_weights,ns.realizations,ns.seed,out/"synthetic")
        plot_synthetic_comparison(syn,out/"figures"/"synthetic_mean_degree.png","mean_degree")
    print(json.dumps({"status":"completed","threshold":tau,"nodes":ns.expected_nodes,"out":str(out)},indent=2))

if __name__=="__main__": run()

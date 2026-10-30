"""2019/2022 node metric comparisons."""
import pandas as pd
from scipy.stats import spearmanr

def compare_node_metrics(metrics_by_year_rep):
    rows=[]
    for rep in ("UU","UD","WD"):
        a=metrics_by_year_rep[(2019,rep)].set_index("node")
        b=metrics_by_year_rep[(2022,rep)].set_index("node")
        common=a.index.intersection(b.index)
        for metric in ["in_degree","out_degree","degree_undirected_projection","in_strength","out_strength","pagerank","betweenness","k_core"]:
            if metric not in a or metric not in b: continue
            x=a.loc[common,metric].astype(float); y=b.loc[common,metric].astype(float)
            rho=float(spearmanr(x,y).statistic) if len(common)>2 else float("nan")
            rows.append({"representation":rep,"metric":metric,"nodes_compared":len(common),
                "spearman_2019_vs_2022":rho,"mean_2019":float(x.mean()),"mean_2022":float(y.mean()),
                "median_2019":float(x.median()),"median_2022":float(y.median()),
                "mean_absolute_change":float((y-x).abs().mean())})
    return pd.DataFrame(rows)

def top_node_table(metrics,metric,top_n=25,largest=True):
    return metrics.sort_values(metric,ascending=not largest).head(top_n).copy()

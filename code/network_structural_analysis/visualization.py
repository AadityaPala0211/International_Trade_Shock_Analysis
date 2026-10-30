"""Publication-ready exploratory figures for structural analysis."""
import numpy as np
import matplotlib.pyplot as plt

def plot_degree_distribution(node_df,title,path,col="degree_undirected_projection"):
    fig,ax=plt.subplots(figsize=(6.5,4.2)); vals=node_df[col].to_numpy()
    bins=np.arange(0,max(1,int(vals.max()))+2)-0.5
    ax.hist(vals,bins=bins,density=True)
    ax.set_xlabel(col.replace("_"," ").title()); ax.set_ylabel("Probability")
    ax.set_title(title); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def plot_strength_distribution(node_df,title,path):
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    for col in ("in_strength","out_strength"):
        vals=node_df[col].to_numpy(); vals=vals[vals>0]
        if len(vals): ax.hist(np.log10(vals),bins=45,alpha=.55,label=col.replace("_"," "))
    ax.set_xlabel("log10 dependency strength"); ax.set_ylabel("Node count")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def plot_metric_comparison(summary_df,out_path,metric="density"):
    fig,ax=plt.subplots(figsize=(7,4.5))
    for rep,g in summary_df.groupby("representation"):
        ax.plot(g["year"],g[metric],marker="o",label=rep)
    ax.set_xlabel("ICIO year"); ax.set_ylabel(metric.replace("_"," ").title())
    ax.legend(title="Representation"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out_path,dpi=180); plt.close(fig)

def plot_synthetic_comparison(df,out_path,metric="mean_degree"):
    data=df.groupby(["family","configuration","representation"],as_index=False)[metric].mean()
    configs=(data[["family","configuration"]].drop_duplicates()
             .sort_values(["family","configuration"]).reset_index(drop=True))
    labels=[f"{row.family}-{int(row.configuration)}" for row in configs.itertuples()]
    reps=list(data.representation.unique()); x=np.arange(len(configs)); width=.24
    fig,ax=plt.subplots(figsize=(10,4.8))
    for idx,rep in enumerate(reps):
        subset=data[data.representation==rep].set_index(["family","configuration"])
        vals=[subset.loc[(row.family,row.configuration),metric] for row in configs.itertuples()]
        ax.bar(x+(idx-(len(reps)-1)/2)*width,vals,width,label=rep)
    ax.set_xticks(x,labels); ax.set_xlabel("Synthetic family and configuration")
    ax.set_ylabel(metric.replace("_"," ").title()); ax.set_title(f"Synthetic network comparison: {metric}")
    ax.legend(title="Representation"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(out_path,dpi=180); plt.close(fig)

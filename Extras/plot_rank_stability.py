"""Visualize how consistently node rankings are preserved from 2019 to 2022."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results_network_analysis" / "comparisons" / "node_metric_year_comparison.csv"
OUT = Path(__file__).resolve().parent / "figures"
METRICS = ["out_degree", "in_degree", "pagerank", "betweenness", "k_core"]
LABELS = ["Out-degree", "In-degree", "PageRank", "Betweenness", "K-core"]


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved comparison file not found: {SUMMARY}")
    data = pd.read_csv(SUMMARY)
    pivot = data[data["metric"].isin(METRICS)].pivot(
        index="metric", columns="representation", values="spearman_2019_vs_2022"
    ).reindex(METRICS)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(METRICS))
    width = 0.24
    for i, representation in enumerate(("UU", "UD", "WD")):
        offset = (i - 1) * width
        ax.bar([v + offset for v in x], pivot[representation].to_numpy(),
               width=width, label=representation)
    ax.set_xticks(list(x), LABELS)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Spearman rank correlation (2019 vs 2022)")
    ax.set_title("Persistence of country-industry node rankings")
    ax.legend(title="Network")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    path = OUT / "node_rank_stability.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

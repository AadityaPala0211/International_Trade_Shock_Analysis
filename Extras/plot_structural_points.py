"""Point-only comparison of the 2019 and 2022 empirical network summaries."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results_network_analysis" / "empirical_network_summaries.csv"
OUT = Path(__file__).resolve().parent / "figures"


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved summary not found: {SUMMARY}")
    data = pd.read_csv(SUMMARY)
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    specs = [
        ("average_clustering_undirected_projection", "Average clustering coefficient", False),
        ("mean_degree_undirected_projection", "Average degree (undirected projection)", False),
        ("giant_component_fraction", "Largest component (% of nodes)", True),
    ]

    # These three statistics are computed on the same undirected projection
    # for UU, UD and WD. Plot UU once to avoid three overlapping duplicates.
    years = [2019, 2022]
    for ax, (column, title, as_percent) in zip(axes.flat[:3], specs):
        points = (data[data["representation"] == "UU"]
                  .set_index("year").reindex(years))
        values = points[column].astype(float)
        ax.scatter(years, values, s=90, zorder=3)
        for year, value in zip(years, values):
            label = f"{value:.1%}" if as_percent else f"{value:.3f}"
            ax.annotate(label, (year, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_xticks(years)
        ax.set_xlim(2018.5, 2022.5)
        ax.grid(axis="y", alpha=0.25)
        if as_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Density is representation-specific, so retain the three point series.
    ax = axes.flat[3]
    offsets = {"UU": -0.10, "UD": 0.0, "WD": 0.10}
    for representation in ("UU", "UD", "WD"):
        part = data[data["representation"] == representation].set_index("year").reindex(years)
        xs = [year + offsets[representation] for year in years]
        ys = part["density"].astype(float).to_numpy()
        ax.scatter(xs, ys, s=75, label=representation, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2%}", (x, y), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("Network density by representation")
    ax.set_xticks(years)
    ax.set_xlim(2018.5, 2022.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="Network")
    ax.grid(axis="y", alpha=0.25)

    for ax in axes.flat:
        ax.set_xlabel("ICIO year")
    fig.suptitle("Empirical network structure: 2019 versus 2022", y=1.01, fontsize=14)
    fig.tight_layout()
    path = OUT / "structural_metrics_points.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

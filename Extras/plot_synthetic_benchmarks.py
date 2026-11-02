"""Map the saved synthetic configurations by mean degree and clustering."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results_network_analysis" / "synthetic" / "synthetic_configuration_summaries.csv"
OUT = Path(__file__).resolve().parent / "figures"
FAMILIES = ("ER", "BA", "WS")


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved synthetic summary not found: {SUMMARY}")
    data = pd.read_csv(SUMMARY)
    # UU gives the undirected graph statistics once per configuration.
    data = data[data["representation"] == "UU"].copy()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for family in FAMILIES:
        part = data[data["family"] == family]
        ax.scatter(part["mean_degree_mean"],
                   part["average_clustering_undirected_projection_mean"],
                   s=80, label=family, zorder=3)
        for row in part.itertuples():
            ax.annotate(f"{family}-{int(row.configuration)}",
                        (row.mean_degree_mean,
                         row.average_clustering_undirected_projection_mean),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Display uncertainty only when the saved output has multiple independent graphs.
    if (data["realizations"] > 1).all():
        x = data["mean_degree_mean"].to_numpy()
        y = data["average_clustering_undirected_projection_mean"].to_numpy()
        xlo = x - data["mean_degree_ci95_low"].to_numpy()
        xhi = data["mean_degree_ci95_high"].to_numpy() - x
        ylo = y - data["average_clustering_undirected_projection_ci95_low"].to_numpy()
        yhi = data["average_clustering_undirected_projection_ci95_high"].to_numpy() - y
        ax.errorbar(x, y, xerr=[xlo, xhi], yerr=[ylo, yhi],
                    fmt="none", ecolor="0.4", alpha=0.45, capsize=2)

    ax.set_xlabel("Mean degree")
    ax.set_ylabel("Average clustering coefficient")
    ax.set_title("Synthetic network design space")
    ax.legend(title="Network family")
    ax.grid(alpha=0.22)
    if int(data["realizations"].min()) == 1:
        ax.text(0.02, 0.02, "Saved outputs contain one graph per configuration; ensemble CIs are unavailable.",
                transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    path = OUT / "synthetic_degree_clustering_map.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

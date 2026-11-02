"""Compare 2019 and 2022 out-degree for the 12 biggest 2022 UD suppliers."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_network_analysis" / "empirical"
OUT = Path(__file__).resolve().parent / "figures"
TOP_N = 12


def main():
    path19 = RESULTS / "2019" / "UD" / "node_metrics.csv.gz"
    path22 = RESULTS / "2022" / "UD" / "node_metrics.csv.gz"
    for path in (path19, path22):
        if not path.exists():
            raise FileNotFoundError(f"Saved node metrics not found: {path}")

    m19 = pd.read_csv(path19, compression="gzip").set_index("node")
    m22 = pd.read_csv(path22, compression="gzip").set_index("node")
    nodes = m22.nlargest(TOP_N, "out_degree").index.tolist()
    values19 = m19.reindex(nodes)["out_degree"]
    values22 = m22.reindex(nodes)["out_degree"]
    if values19.isna().any() or values22.isna().any():
        raise ValueError("The saved 2019/2022 node universes do not contain the same labels.")

    OUT.mkdir(parents=True, exist_ok=True)
    y = list(range(len(nodes)))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([v - height / 2 for v in y], values19, height=height, label="2019")
    ax.barh([v + height / 2 for v in y], values22, height=height, label="2022")
    ax.set_yticks(y, nodes)
    ax.invert_yaxis()
    ax.set_xlabel("Out-degree (number of buyer links)")
    ax.set_title(f"Top {TOP_N} supplier nodes in the 2022 UD network")
    ax.legend()
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    path = OUT / "top_2022_supplier_hubs_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

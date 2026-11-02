"""Final prevalence versus normalized transmission, with the eta=1 threshold."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (ROOT / "results_network_analysis" / "dynamics" / "all" /
           "real_dynamics_seedset_summary.csv")
OUT = Path(__file__).resolve().parent / "figures"


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved SIS summary not found: {SUMMARY}")
    data = pd.read_csv(SUMMARY)
    data = data[data["eta"].notna()].copy()
    data["eta"] = pd.to_numeric(data["eta"], errors="raise")
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, representation in zip(axes, ("UU", "UD", "WD")):
        part = data[data["representation"] == representation]
        for year in (2019, 2022):
            curve = part[part["year"] == year].sort_values("eta")
            if curve.empty:
                continue
            x = curve["eta"].to_numpy(dtype=float)
            y = curve["final_prevalence_mean_across_seed_sets"].to_numpy(dtype=float)
            low = curve["final_prevalence_ci95_seed_low"].to_numpy(dtype=float)
            high = curve["final_prevalence_ci95_seed_high"].to_numpy(dtype=float)
            yerr = np.vstack((np.maximum(y - low, 0), np.maximum(high - y, 0)))
            ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=1.7,
                        markersize=5, capsize=2.5, label=str(year))
        ax.axvline(1.0, color="0.25", linestyle=":", linewidth=1.3)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-10, top=1)
        ax.set_title(representation)
        ax.set_xlabel("Normalized transmission (eta)")
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("Final prevalence at simulation endpoint (log scale)")
    axes[-1].legend(title="ICIO year")
    fig.suptitle("SIS response across the transmission threshold", y=1.02)
    fig.tight_layout()
    path = OUT / "sis_final_prevalence_vs_eta.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

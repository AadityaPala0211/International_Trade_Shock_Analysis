"""Plot real-network prevalence over model time at one fixed eta.

Uses the pre-aggregated real_trajectory_summary.csv.gz; does not rerun SIS.
Edit ETA below to choose a scenario already present in the saved results.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (ROOT / "results_network_analysis" / "dynamics" / "all" /
           "trajectory_summaries" / "real_trajectory_summary.csv.gz")
OUT = Path(__file__).resolve().parent / "figures"
ETA = 1.5  # Available scenarios: 0.5, 0.9, 1.0, 1.1, 1.5, 2.0
Y_SCALE = "log"  # Use "linear" to switch back to a 0..1 vertical axis


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved trajectory summary not found: {SUMMARY}")
    data = pd.read_csv(SUMMARY, compression="gzip")
    scenario = f"eta_{ETA:g}"
    data = data[(data["scenario"] == scenario) & np.isclose(data["eta"], ETA)]
    if data.empty:
        raise ValueError(f"No saved trajectory for eta={ETA:g} in {SUMMARY}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, representation in zip(axes, ("UU", "UD", "WD")):
        part = data[data["representation"] == representation]
        for year in (2019, 2022):
            curve = part[part["year"] == year].sort_values("time")
            if curve.empty:
                continue
            x = curve["time"].to_numpy()
            y = curve["prevalence_mean"].to_numpy()
            low = curve["prevalence_ci95_seed_low"].to_numpy()
            high = curve["prevalence_ci95_seed_high"].to_numpy()
            line, = ax.plot(x, y, linewidth=2, label=str(year))
            ax.fill_between(x, low, high, color=line.get_color(), alpha=0.16)
        ax.set_title(representation)
        ax.set_xlabel("Discrete model time step (t)")
        if Y_SCALE == "log":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-10, top=1)
        else:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("Mean prevalence")
    axes[-1].legend(title="ICIO year")
    fig.suptitle(f"Real-network SIS trajectories at fixed eta = {ETA:g}", y=1.02)
    fig.tight_layout()
    tag = str(ETA).replace(".", "p")
    path = OUT / f"sis_trajectory_eta_{tag}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print("X-axis is the model's discrete time step, not a calendar date.")


if __name__ == "__main__":
    main()

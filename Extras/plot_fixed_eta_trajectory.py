"""Single-axis SIS trajectory plot for subcritical, critical and supercritical eta.

Uses the saved real_trajectory_summary.csv.gz only. No SIS run is performed.
Change REPRESENTATION to UU, UD or WD to select the network view.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (ROOT / "results_network_analysis" / "dynamics" / "all" /
           "trajectory_summaries" / "real_trajectory_summary.csv.gz")
OUT = Path(__file__).resolve().parent / "figures"

# One representative eta nearest the threshold in each regime.
# The full saved grid is 0.5, 0.9, 1.0, 1.1, 1.5 and 2.0.
ETA_CASES = [
    ("Below threshold: eta < 1", 0.9),
    ("At threshold: eta = 1", 1.0),
    ("Above threshold: eta > 1", 1.1),
]
REPRESENTATION = "UD"  # Change to "UU" or "WD" as needed.
YEARS = (2019, 2022)
Y_MIN_FALLBACK = 1e-12
BAND_ALPHA = 0.055
SMOOTH_POINTS_PER_STEP = 3


def _scenario_name(eta):
    return f"eta_{eta:g}"


def _positive_floor(frame):
    candidates = []
    for column in ("prevalence_mean", "prevalence_ci95_seed_low"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            candidates.extend(values[np.isfinite(values) & (values > 0)].tolist())
    return max(Y_MIN_FALLBACK, min(candidates) * 0.5) if candidates else Y_MIN_FALLBACK


def main():
    if REPRESENTATION not in {"UU", "UD", "WD"}:
        raise ValueError("REPRESENTATION must be one of: UU, UD, WD.")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved trajectory summary not found: {SUMMARY}")

    data = pd.read_csv(SUMMARY, compression="gzip")
    data = data[data["representation"] == REPRESENTATION].copy()
    if data.empty:
        raise ValueError(f"No saved trajectories for representation {REPRESENTATION}.")

    OUT.mkdir(parents=True, exist_ok=True)
    floor = _positive_floor(data)
    colors = plt.get_cmap("tab10").colors
    year_styles = {2019: "-", 2022: "--"}
    plotted = []
    legend_handles = []
    max_time = 0

    fig, ax = plt.subplots(figsize=(10.8, 6.1))

    for case_index, (case_label, eta) in enumerate(ETA_CASES):
        color = colors[case_index]
        scenario = _scenario_name(eta)
        scenario_data = data[
            (data["scenario"] == scenario) &
            np.isclose(pd.to_numeric(data["eta"], errors="coerce"), eta)
        ]

        for year in YEARS:
            curve = (scenario_data[scenario_data["year"] == year]
                     .sort_values("time").copy())
            if curve.empty:
                raise ValueError(
                    f"Missing saved data for {REPRESENTATION}, year={year}, eta={eta:g}."
                )

            x = pd.to_numeric(curve["time"], errors="raise").to_numpy(dtype=float)
            y = pd.to_numeric(curve["prevalence_mean"], errors="raise").to_numpy(dtype=float)
            low = pd.to_numeric(curve["prevalence_ci95_seed_low"], errors="coerce").to_numpy(dtype=float)
            high = pd.to_numeric(curve["prevalence_ci95_seed_high"], errors="coerce").to_numpy(dtype=float)

            if (len(x) < 2 or np.any(np.diff(x) <= 0) or
                    not np.isfinite(y).all() or np.any(y <= 0)):
                raise ValueError(
                    f"Invalid or non-positive prevalence series for {year}, eta={eta:g}."
                )

            # PCHIP is only a display interpolation between saved integer time
            # steps; it does not change the discrete-time SIS calculations.
            x_smooth = np.linspace(x.min(), x.max(),
                                   max(len(x) * SMOOTH_POINTS_PER_STEP, 500))
            y_smooth = np.maximum(PchipInterpolator(x, y)(x_smooth), floor)

            line_style = year_styles[year]
            line, = ax.plot(
                x_smooth, y_smooth,
                color=color, linestyle=line_style, linewidth=2.8,
                solid_capstyle="round", antialiased=True,
                label=f"{case_label} ({year})",
                zorder=3 + case_index,
            )

            valid_band = np.isfinite(low) & np.isfinite(high)
            if valid_band.all():
                low_smooth = PchipInterpolator(x, np.maximum(low, floor))(x_smooth)
                high_smooth = PchipInterpolator(x, np.maximum(high, floor))(x_smooth)
                low_smooth = np.maximum(low_smooth, floor)
                high_smooth = np.maximum(high_smooth, low_smooth)
                ax.fill_between(
                    x_smooth, low_smooth, high_smooth,
                    color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1,
                )

            max_time = max(max_time, int(x.max()))
            plotted.append((case_label, eta, year))

            legend_handles.append(Line2D(
                [0], [0], color=color, linestyle=line_style, linewidth=2.8,
                label=f"{'Below' if eta < 1 else 'At' if eta == 1 else 'Above'} threshold "
                      f"(eta={eta:g}), {year}",
            ))

    ax.set_yscale("log")
    ax.set_ylim(bottom=floor, top=1.0)
    ax.set_xlim(left=0, right=max_time)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.set_xlabel("Discrete model time step (t)", fontsize=11)
    ax.set_ylabel("Mean prevalence (log scale)", fontsize=11)
    ax.set_title(
        f"{REPRESENTATION} network: SIS dynamics below, at and above threshold",
        fontsize=13, pad=14,
    )
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.margins(x=0.01)

    fig.legend(
        handles=legend_handles, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 0.91), frameon=False, fontsize=8.5,
        handlelength=2.8, columnspacing=1.4,
    )
    fig.text(
        0.5, 0.965,
        "Colors denote transmission regime; solid = 2019, dashed = 2022",
        ha="center", va="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.79)

    path = OUT / f"sis_trajectory_threshold_cases_{REPRESENTATION}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {path}")
    print(f"Representation: {REPRESENTATION}")
    print("Cases: eta=0.9 (<1), eta=1.0 (=1), eta=1.1 (>1)")
    print("2019 uses solid lines; 2022 uses dashed lines.")
    print("The x-axis is discrete model time, not calendar time.")
    print("PCHIP only smooths the visual connection between saved time steps.")


if __name__ == "__main__":
    main()

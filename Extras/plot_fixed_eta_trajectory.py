"""Single-axis SIS trajectory plot for subcritical, critical and supercritical eta.

The main axis uses a linear prevalence scale so the growth/flattening shape is
readable. An inset zooms into eta<1 and eta=1 so their small trajectories are
not hidden by the larger supercritical response.

Reads the existing real_trajectory_summary.csv.gz; does not rerun SIS.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (ROOT / "results_network_analysis" / "dynamics" / "all" /
           "trajectory_summaries" / "real_trajectory_summary.csv.gz")
OUT = Path(__file__).resolve().parent / "figures"

# One representative eta nearest the threshold in each regime.
ETA_CASES = [
    ("Below threshold", 0.9),
    ("At threshold", 1.0),
    ("Above threshold", 1.1),
]
REPRESENTATION = "UD"  # Change to "UU" or "WD" as needed.
YEARS = (2019, 2022)
SMOOTH_POINTS_PER_STEP = 3
BAND_ALPHA = 0.07


def _scenario_name(eta):
    return f"eta_{eta:g}"


def _get_curve(data, eta, year):
    scenario = _scenario_name(eta)
    curve = (data[
        (data["scenario"] == scenario) &
        np.isclose(pd.to_numeric(data["eta"], errors="coerce"), eta) &
        (pd.to_numeric(data["year"], errors="coerce") == year)
    ].sort_values("time").copy())

    if curve.empty:
        raise ValueError(
            f"Missing saved data for {REPRESENTATION}, year={year}, eta={eta:g}."
        )

    x = pd.to_numeric(curve["time"], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(curve["prevalence_mean"], errors="raise").to_numpy(dtype=float)
    low = pd.to_numeric(curve["prevalence_ci95_seed_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(curve["prevalence_ci95_seed_high"], errors="coerce").to_numpy(dtype=float)

    if (len(x) < 2 or np.any(np.diff(x) <= 0) or
            not np.isfinite(y).all() or np.any(y < 0)):
        raise ValueError(f"Invalid trajectory for {year}, eta={eta:g}.")

    # Interpolation only smooths the display between saved integer time steps.
    x_smooth = np.linspace(x.min(), x.max(),
                           max(len(x) * SMOOTH_POINTS_PER_STEP, 500))
    y_smooth = np.clip(PchipInterpolator(x, y)(x_smooth), 0.0, 1.0)
    low_smooth = high_smooth = None

    if np.isfinite(low).all() and np.isfinite(high).all():
        low_smooth = np.clip(PchipInterpolator(x, low)(x_smooth), 0.0, 1.0)
        high_smooth = np.clip(PchipInterpolator(x, high)(x_smooth), 0.0, 1.0)
        low_smooth = np.minimum(low_smooth, y_smooth)
        high_smooth = np.maximum(high_smooth, y_smooth)

    return x, y, x_smooth, y_smooth, low_smooth, high_smooth


def _draw_curve(ax, series, color, linestyle, linewidth, alpha_band=0.0, label=None):
    _, _, x, y, low, high = series
    line, = ax.plot(
        x, y, color=color, linestyle=linestyle, linewidth=linewidth,
        solid_capstyle="round", antialiased=True, label=label, zorder=3,
    )
    if low is not None and high is not None and alpha_band > 0:
        ax.fill_between(x, low, high, color=color, alpha=alpha_band,
                        linewidth=0, zorder=1)
    return line


def main():
    if REPRESENTATION not in {"UU", "UD", "WD"}:
        raise ValueError("REPRESENTATION must be one of: UU, UD, WD.")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Saved trajectory summary not found: {SUMMARY}")

    data = pd.read_csv(SUMMARY, compression="gzip")
    data = data[data["representation"] == REPRESENTATION].copy()
    if data.empty:
        raise ValueError(f"No saved trajectories for representation {REPRESENTATION}.")

    colors = plt.get_cmap("tab10").colors
    year_styles = {2019: "-", 2022: "--"}
    series = {}
    max_time = 0
    max_prevalence = 0.0

    # Load only the six time series needed for the selected representation.
    for case_index, (_, eta) in enumerate(ETA_CASES):
        for year in YEARS:
            item = _get_curve(data, eta, year)
            series[(eta, year)] = item
            max_time = max(max_time, int(item[0].max()))
            max_prevalence = max(max_prevalence, float(item[3].max()))

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    handles = []

    # Main panel: all three regimes and both ICIO years on a linear scale.
    for case_index, (case_label, eta) in enumerate(ETA_CASES):
        color = colors[case_index]
        for year in YEARS:
            line = _draw_curve(
                ax, series[(eta, year)], color, year_styles[year],
                linewidth=2.8, alpha_band=BAND_ALPHA,
                label=f"{case_label} (eta={eta:g}), {year}",
            )
            handles.append(Line2D(
                [0], [0], color=color, linestyle=year_styles[year],
                linewidth=2.8,
                label=f"{case_label} (eta={eta:g}), {year}",
            ))

    y_top = min(1.0, max_prevalence * 1.13 + 0.001)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, y_top)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Discrete model time step (t)", fontsize=11)
    ax.set_ylabel("Mean prevalence", fontsize=11)
    ax.set_title(
        f"{REPRESENTATION} network: SIS response below, at and above threshold",
        fontsize=13, pad=12,
    )
    ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)

    # Inset: focus on the two low-prevalence regimes across the full horizon.
    # This keeps their differences visible without compressing the main axis.
    inset = ax.inset_axes([0.57, 0.13, 0.37, 0.38])
    inset_top = max(
        item[3].max()
        for (eta, _), item in series.items() if eta <= 1.0
    ) * 1.12
    for eta, _ in ETA_CASES[:2]:
        case_index = next(i for i, (_, value) in enumerate(ETA_CASES) if value == eta)
        color = colors[case_index]
        for year in YEARS:
            _draw_curve(
                inset, series[(eta, year)], color, year_styles[year],
                linewidth=1.9, alpha_band=0.04,
            )

    inset.set_xlim(0, max_time)
    inset.set_ylim(0, max(inset_top, 0.001))
    inset.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    inset.xaxis.set_major_locator(MultipleLocator(100))
    inset.set_title("Zoom: below and at threshold", fontsize=8.5, pad=5)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, alpha=0.20, linewidth=0.6)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)

    fig.legend(
        handles=handles, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=8.5,
        handlelength=2.8, columnspacing=1.5,
    )
    fig.text(
        0.5, 0.975,
        "Colors denote transmission regime; solid = 2019, dashed = 2022",
        ha="center", va="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.80)

    path = OUT / f"sis_trajectory_threshold_cases_{REPRESENTATION}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {path}")
    print(f"Representation: {REPRESENTATION}")
    print("Cases: eta=0.9 (<1), eta=1.0 (=1), eta=1.1 (>1)")
    print("Main y-axis is linear; inset zooms into eta=0.9 and eta=1.0.")
    print("2019 uses solid lines; 2022 uses dashed lines.")
    print("The x-axis is discrete model time, not calendar time.")
    print("PCHIP only smooths the visual connection between saved time steps.")


if __name__ == "__main__":
    main()

# Extras: quick presentation figures

These scripts create presentation-ready figures from **results already stored** in
`results_network_analysis/`. They only read CSV/CSV.GZ summaries and saved node
metrics. They do **not** load or regenerate the large ICIO source files, rebuild
networks, load the adjacency NPZ matrices, or rerun SIS.

## Run from the repository root

Install the project's existing plotting dependencies once, if needed:

```bash
python -m pip install -r code/network_structural_analysis/requirements.txt
```

Run every figure:

```bash
python Extras/plot_all.py
```

Or run one figure at a time:

```bash
python Extras/plot_structural_points.py
python Extras/plot_fixed_eta_trajectory.py
python Extras/plot_sis_response_vs_eta.py
python Extras/plot_rank_stability.py
python Extras/plot_synthetic_benchmarks.py
python Extras/plot_top_hubs.py
```

All PNGs are written to `Extras/figures/` at 300 dpi. Existing results are not
modified.

## Figures and input files

| Script | What it plots | Saved input |
|---|---|---|
| `plot_structural_points.py` | Four point-only panels: clustering, undirected-projection mean degree, largest-component fraction, and density by UU/UD/WD | `empirical_network_summaries.csv` |
| `plot_fixed_eta_trajectory.py` | One graph with representative eta cases below (0.9), at (1.0), and above (1.1) the threshold; solid/dashed lines compare 2019/2022, with a linear main axis and a zoom inset for the lower trajectories | `dynamics/all/trajectory_summaries/real_trajectory_summary.csv.gz` |
| `plot_sis_response_vs_eta.py` | Final prevalence versus eta, with eta=1 threshold and seed-set error bars | `dynamics/all/real_dynamics_seedset_summary.csv` |
| `plot_rank_stability.py` | Spearman rank persistence of node metrics from 2019 to 2022 | `comparisons/node_metric_year_comparison.csv` |
| `plot_synthetic_benchmarks.py` | Synthetic configuration design space: mean degree versus clustering | `synthetic/synthetic_configuration_summaries.csv` |
| `plot_top_hubs.py` | 2019/2022 out-degree comparison for the top 12 2022 UD supplier nodes | Saved empirical UD node-metric CSV.GZ files |

## Two important interpretation notes

1. In the structural summary, clustering, mean degree on the undirected
   projection, and largest-component fraction are identical for UU, UD and WD.
   They are therefore drawn once as two year-points, rather than as duplicate
   representation series. Density is representation-specific, so it has one
   point series for each of UU, UD and WD. **No points are connected by lines**
   in the structural comparison.

2. In the fixed-eta figure, the horizontal axis is the SIS model's discrete
   time step (t), not a calendar timestamp. Prevalence is measured at each
   time step; the final prevalence is the last value at the simulation endpoint.
   The default is (eta=1.5). Change the `ETA` constant near the top of
   `plot_fixed_eta_trajectory.py` to 0.5, 0.9, 1.0, 1.1 or 2.0 to plot another
   saved scenario. This selection only filters the existing trajectory summary.

The current saved synthetic experiment has one graph realization per
configuration. The synthetic design-space plot therefore shows configuration
points without ensemble confidence intervals; the script will show saved
intervals automatically if a future experiment contains multiple realizations.

The fixed-ηη trajectory plot now shows three threshold regimes on one graph: `eta=0.9`, `eta=1.0`, and `eta=1.1`. It uses a linear main y-axis for an intuitive prevalence comparison and an inset to magnify the below-threshold and critical trajectories. Change `REPRESENTATION` in the script to `UU`, `UD`, or `WD`; the six series are still read from the stored trajectory summary only.

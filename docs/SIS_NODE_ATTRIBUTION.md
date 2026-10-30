# SIS node-level shock attribution add-on

This is a separate package from `network_structural_analysis`. It reads
persisted sparse matrices, exact node-index labels and precomputed structural
metrics. It does not rewrite structural matrices, manifests, metrics or code.

## Method: unchanged Model A, extended initial-condition design

All experiments use the deterministic mass-action SIS / NIMFA-style Model A:

```
d_j(t+1) = clip_[0,1]((1-rho)d_j(t) + (1-d_j(t))*beta*sum_i B_ij*d_i(t))
e(t) = B.T @ d(t)
```

`B[i,j]` means supplier `i` influences buyer `j`. Infection/recovery
events are not stochastic within a trajectory. The new experiments change
which node(s) start affected, not the contagion equation.

## Added experiments

1. **Exhaustive single-node shock census.** Every one of the N nodes is tested
   exactly once as the sole initially affected activity. For N=4,050 this is
   4,050 trajectories per network and beta scenario. Outputs rank initiating
   nodes by peak, cumulative and final prevalence.
2. **Centrality-targeted cohorts.** Top and bottom 1% (ceil(0.01*N)) are seeded
   separately for each available centrality, alongside 20 independently
   drawn random cohorts of identical size. The random cohort draws are reused
   across centrality and beta-scenario comparisons within each network.
3. **Centrality-impact association.** Spearman rank correlations quantify
   associations between each source metric and the three SIS impact measures.
   These are descriptive diagnostics, not causal tests. P-values are also
   adjusted with Benjamini-Hochberg false-discovery-rate control within each
   network/scenario centrality-by-impact test family.
4. **Existing random multi-node experiment.** The 20 real and five-per-
   synthetic-skeleton random seed sets in the original dynamics pipeline stay
   unchanged. They estimate response to random initial multi-node disruptions
   and complement, rather than replace, the new targeted experiments.

### Centralities

The extension calculates supplier out-degree/out-strength, buyer in-degree/
in-strength, undirected projection degree, supplier-to-buyer PageRank and
supplier-to-buyer eigenvector centrality. It also reuses the existing
structural-analysis betweenness, k-core and clustering columns when present.
The top/bottom cohorts are structural targeting hypotheses; the scores never
enter the SIS state transition itself.

## Runnable command

Install the existing project's dependencies, then from repository root:

```bash
# Linux/macOS
export PYTHONPATH=code
python -m network_dynamics.run_shock_experiments \
  --results-dir results_network_analysis \
  --out results_network_analysis/dynamics_node_attribution \
  --scope real --rho 0.2 \
  --beta-uu YOUR_VALIDATED_BETA_UU \
  --beta-ud YOUR_VALIDATED_BETA_UD \
  --beta-wd YOUR_VALIDATED_BETA_WD
```

Replace the three beta placeholders with the absolute values selected from
`dynamics/validation_all/dynamics_input_precheck.csv`; do not copy example
values without checking their beta-critical ranges. The absolute-beta scenario
holds one beta constant within each representation across networks. The six
additional eta scenarios use each matrix's beta-critical value, eta times
beta_c, with eta in {0.5, 0.9, 1.0, 1.1, 1.5, 2.0}.

The default empirical scope is all six matrices (2019/2022 × UU/UD/WD) and all
seven scenarios (absolute plus six eta values). This is 170,100 single-source
trajectories (6 × 7 × 4,050). The one-source census is computed in bounded
batches of 64 source nodes to reduce repeated sparse-matrix traversal and keep
memory usage predictable. Targeted cohorts default to ten available
centralities × (top + bottom + 20 random controls) for each network/scenario.

Synthetic networks are deliberately opt-in because a full 270-matrix, seven-
scenario node census would require 7,654,500 individual source trajectories
at ten realizations/configuration. To include a pilot with one paired
realization per configuration:

```bash
python -m network_dynamics.run_shock_experiments \
  --results-dir results_network_analysis \
  --out results_network_analysis/dynamics_node_attribution_synthetic_pilot \
  --scope synthetic --synthetic-realizations 1 --rho 0.2 \
  --beta-uu YOUR_VALIDATED_BETA_UU \
  --beta-ud YOUR_VALIDATED_BETA_UD \
  --beta-wd YOUR_VALIDATED_BETA_WD
```

Use `--synthetic-realizations 0` to include every realization. Use
`--scenarios absolute eta_1` for a smaller sensitivity run, or
`--random-cohorts 50` for more precise random-group comparison. Keep the
output under a separate `dynamics_node_attribution` directory. The runner
checks node-index and metric alignment and writes SHA-256 checksums of inputs.

Optional controls: `--initial-infection` (default 1.0),
`--target-fraction` (default 0.01), `--random-cohorts` (default 20),
`--max-steps` (300), `--tolerance` (1e-8), `--census-batch-size` (64),
`--store-trajectories` (off by default; creates much larger files).

## Outputs

- One directory per network and scenario, containing
  `single_node_shock_census.csv.gz` (one row per source, including fixed-horizon\n  cumulative prevalence, observed-until-stop area and plateau extension), optional
  `single_node_shock_trajectories.csv.gz`, `targeting_metrics.csv.gz`,
  `targeted_cohort_runs.csv.gz`, `targeted_cohort_summary.csv`,
  `centrality_impact_correlations.csv` and `experiment_metadata.json`.
- `shock_experiment_manifest.csv`: scenario inventory, source count, target
  cohort run count, beta/threshold values, input checksum and output directory.
- `top_20_single_node_shocks.csv.gz`: convenience extract of the top 20
  cumulative-impact sources for every network/scenario.
- `shock_experiment_run_metadata.json`: global parameter and provenance audit.

## Interpretation safeguards

A single-node initial state of 1 means a unit shock to one node's model state,
not a calibrated economic output loss, supplier outage probability or
percentage production loss. For N=4,050 the starting prevalence is 1/N.
One single-node shock can die out even when a 1%-many-node seed grows. Report
initial infection, beta, rho, beta_c/eta and horizon with every result.

A node census enumerates source identity and is not a stochastic Monte Carlo
over SIS infection/recovery events. Random-cohort variation is uncertainty
from the location of initial seeds, not stochastic transmission uncertainty.
Centrality-impact correlations and targeted-group differences are descriptive
network-model results, not causal effects. Synthetic experiments remain null-
model comparisons, not replicas of actual production or trade flows.

## Tests

```bash
PYTHONPATH=code python -m unittest discover -s code/network_dynamics/tests -v
```

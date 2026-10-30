# International Trade Shock Analysis — Network & Structural Analysis

This repository stage constructs and analyses the **2019 and 2022 OECD Inter-Country Input-Output (ICIO) production networks** and produces paired synthetic ER, BA and WS benchmark networks. It deliberately does **not** implement dynamical shock propagation.

## Research design

For each annual ICIO table, the package extracts the intermediate transaction block and constructs three representations using a common dependency-share threshold τ:

1. **UU** — unweighted undirected projection (an edge exists if either directed dependency passes the threshold).
2. **UD** — unweighted directed supplier-to-buyer graph.
3. **WD** — weighted directed dependency graph, retaining the dependency share as edge weight.

The supplier/buyer convention is fixed throughout: \`Z[i, j]\` is the intermediate flow from supplier node \`i\` to buyer node \`j\`. With buyer intermediate purchases as denominator, the dependency share is \`S[i, j] = Z[i, j] / sum_k Z[k, j]\`. The denominator retains domestic/self-use flows; the graph numerator excludes diagonal self-loops. This normalization is used consistently as the network edge weight.

The threshold is a **network sparsification parameter**, not an economically observed cutoff. The pipeline first writes threshold diagnostics and a candidate recommendation. The analyst should inspect the diagnostics before treating the recommendation as final. All thresholded outputs record the exact value used.

Synthetic network experiment: generate 3 ER, 3 BA and 3 WS configurations, each with 20 independent graph realizations (180 skeletons). ER mean degree and BA attachment parameter are varied around the empirical calibration target (0.75×, 1.00× and 1.25×); WS holds its nearest even empirical degree fixed and varies rewiring probability (0.01, 0.10, 0.50). Each skeleton is reused to make UU, UD and WD tracks, yielding 540 paired representation instances. Directed orientation is calibrated to empirical arc reciprocity. WD synthetic edge weights are drawn from the threshold-retained empirical dependency-share distribution and rescaled per buyer to match the distribution of retained per-buyer dependency totals. Network size is the empirical ICIO node count (expected 4,050, validated at run time). Synthetic weights are **not real trade or monetary flows**.

## Install

\`\`\`bash
python -m pip install -r code/network_structural_analysis/requirements.txt
\`\`\`

## Data contract

Place the original OECD data files in \`data/\`:

- \`2019_SML.csv\`
- \`2022_SML.csv\`
- optionally \`ReadMe_ICIO_small.xlsx\` (metadata/provenance reference)

The loader supports ICIO-style CSV layouts with row labels and multi-row/multi-column headers. Since editions and CSV export conventions can differ, it identifies and validates the leading square intermediate block from labels and the expected \`N = 4050\` detailed nodes. It can handle appended final-demand columns and accounting rows; it fails loudly rather than guessing if the layout is ambiguous. You can inspect \`--help\` and \`--inspect-only\` output before a full run.

## Recommended two-stage execution

### 1. Inspect input files and compute threshold diagnostics only

\`\`\`bash
PYTHONPATH=code python -m network_structural_analysis.run_pipeline \\
  --data-dir data --out results_network_analysis \\
  --threshold-only
\`\`\`

Inspect \`threshold_diagnostics.csv\`, \`threshold_local_sensitivity.csv\`, \`threshold_recommendation.json\` and the diagnostic figures. The candidate threshold is a transparent heuristic combining flow coverage, edge retention, density and topology. It is not a unique optimum; review the curves and select or override τ before the full run.

### 2. Build empirical networks and run structural analysis + synthetic ensemble

\`\`\`bash
PYTHONPATH=code python -m network_structural_analysis.run_pipeline \\
  --data-dir data --out results_network_analysis \\
  --realizations 20
\`\`\`

Pass \`--threshold 0.0001\` (or another reviewed value) to override the recommendation. The same threshold is applied to both years for comparability. The pipeline saves the diagnostics even when an override is supplied.

### Useful options

- \`--threshold-only\`: stop after threshold diagnostics.
- \`--threshold FLOAT\`: use an explicit common threshold.
- \`--realizations 20\`: independent synthetic realizations per configuration.
- \`--seed 20261030\`: reproducible random seed.
- \`--synthetic-only\`: skip empirical structural metric export after calibration.
- \`--skip-synthetic\`: empirical networks only.
- \`--inspect-only\`: preview the input files and dimensions.

## Outputs

\`results_network_analysis/\` contains:

- \`metadata/\`: input checksums, source details, companion ICIO workbook provenance preview, canonical node index, chosen threshold and run configuration.
- \`threshold_review/\`: edge-retention / flow-coverage / topology diagnostics and figures.
- \`empirical/{2019,2022}/{UU,UD,WD}/\`: adjacency matrices (\`.npz\`), edge lists, node metrics, country/industry aggregates and plots.
- \`comparisons/\`: year-over-year node and network comparisons, rank correlations and top-node tables.
- \`synthetic/\`: configuration metadata and realization-level / configuration-level summaries (the \`family\` field identifies ER, BA and WS).
- \`figures/\`: degree distributions, strength distributions and network-level empirical/synthetic comparisons.

The \`metadata/node_index.csv\` file records the canonical node-id to ICIO-label mapping. Sparse-first metrics avoid materialising millions of Python edge objects. The code writes sparse matrices and compressed CSVs where possible. Network-level diagnostics also report strong-component summaries for directed graphs and separate in/out degree heterogeneity. Do not commit generated results or duplicate source CSV files unless intentionally required.

See [\`docs/METHODOLOGY.md\`](docs/METHODOLOGY.md) for equations, threshold heuristic, synthetic calibration, metric definitions and interpretation limits.

## Structural metrics

Network-level metrics include node/edge counts, representation-specific density, mean degree, degree heterogeneity (coefficient of variation), weak/strong component summaries, clustering, sampled characteristic path length, directed reciprocity, weighted in/out strength and spectral-radius diagnostics. Node-level outputs include in/out degree and strength, PageRank, betweenness (reproducible sampled Brandes approximation; default 64 source nodes at N=4,050), k-core (on the undirected projection) and local clustering.

Path and betweenness measures are calculated on the **thresholded binary graph** and retain direction for UD/WD. Weak components, clustering and k-core use the undirected projection; strong-component summaries are also reported for UD/WD. Weighted strength, PageRank and weighted spectral radius are calculated from dependency-share weights. The threshold diagnostics include local sensitivity around the candidate value so conclusions are not tied to one sparsification choice.

## Interpretation cautions

- A country-industry node is a production activity in a particular economy, not a firm.
- A dependency share measures the proportion of a buyer's intermediate purchases attributed to a supplier; it is not an outage probability.
- The UU projection discards direction; UD retains direction; WD retains dependency magnitude.
- The synthetic networks are null-model comparisons, not forecasts or replicas of actual trade.
- PageRank, betweenness and k-core depend on the thresholded topology; weighted strengths preserve magnitude.
- Mean shortest-path estimates are sampled; output metadata records the sampling method and seed.
- Spectral radius is not interchangeable between adjacency and dependency matrices; outputs label which matrix was used.
- Dynamical shock propagation is outside the scope of this stage and will be developed separately.

## Tests

\`\`\`bash
PYTHONPATH=code python -m unittest discover -s code/network_structural_analysis/tests -v
python -m compileall -q code/network_structural_analysis
\`\`\`

## Reproducibility

The pipeline stores the seed, package versions, input file SHA-256 checksums, node-order checksum, threshold, configuration and companion workbook metadata in JSON. Each synthetic skeleton is generated once and transformed into the three paired tracks, avoiding independent sampling across representations.

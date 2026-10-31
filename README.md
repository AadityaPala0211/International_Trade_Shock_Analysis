# International Trade Shock Analysis — WD-only 11-network workflow

This branch introduces the revised structural-analysis and individual-node SIS research workflow. The active command-line interface uses **one weighted directed (WD) matrix per network** and contains exactly 11 selected networks. Historical results under \`results_network_analysis/\` are retained and are not inputs to the new run except for the approved synthetic degree calibration described below.

## Active design

| Group | Networks | Realizations |
|---|---:|---:|
| OECD ICIO 2019 | 1 | 1 |
| OECD ICIO 2022 | 1 | 1 |
| ER configurations 1, 2, 3 | 3 | 1 each |
| BA configurations 1, 2, 3 | 3 | 1 each |
| WS configurations 1, 2, 3 | 3 | 1 each |
| **Total** | **11** | **11** |

All networks retain the detailed 4,050 country-industry node universe. The active manifest and result-generation path contain only WD. No UU or UD matrices are written by the revised pipeline. Old UU/UD/WD result files and the old results directory are not deleted or overwritten.

## Retained matrix and economic meaning

Let \(Z_{ij}\) be the intermediate input transaction from supplier \(i\) to buyer \(j\). The buyer's total intermediate purchases are \(p_j=\sum_k Z_{kj}\), including self-use. The WD dependency-share matrix is:

\[
P_{ij}=\frac{Z_{ij}}{p_j},\qquad p_j>0.
\]

Self-edges are excluded from the network matrix, but self-use remains in the denominator. A common, manually reviewed threshold is applied to both empirical years; the previously reviewed threshold recorded in the existing project metadata is **\(\tau=0.001\)**. The retained WD edge weights are not renormalized after thresholding.

\[
B_{ij}=P_{ij}\quad\text{if }P_{ij}\geq\tau\text{ and }i\ne j;\qquad 0\text{ otherwise}.
\]

Rows are suppliers and columns are buyers. Consequently, outgoing degree/strength describe a supplier's direct downstream connections and total retained dependency shares supplied to buyers. Incoming degree/strength describe the supplier links and retained dependency shares received by a buyer. Betweenness uses directed paths with binary edge lengths; PageRank uses dependency-share weights; k-core and clustering use the undirected projection for those metrics only. The SIS model itself always consumes the same persisted WD matrix.

## Approved parameter preservation

The active structural pipeline retains the established parameter settings:

- Expected node count: 4,050.
- Synthetic ER expected-degree multipliers: 0.5, 1.0, 1.5.
- Synthetic BA target-degree multipliers: 0.5, 1.0, 1.5.
- Synthetic WS rewiring probabilities: 0.01, 0.10, 0.50.
- One fixed synthetic realization per configuration.
- Recovery parameter: \(\rho=0.2\), retained from the current approved SIS run metadata.
- Stage 1 transmission ratios: \(\eta\in\{0.9,1.1\}\).
- Stage 2 transmission ratios: \(\eta\in\{0.5,0.9,1.0,1.1,1.5,2.0\}\).
- SIS horizon: 300 updates by default.
- Affected-node threshold: 0.01.
- Extinction tolerance: \(10^{-8}\), sustained for 5 consecutive steps.
- Convergence tolerance: \(10^{-8}\), sustained for 5 consecutive steps.
- Final observation window: 30 time points.
- Census batch size: 128 source nodes (configurable).

The existing approved synthetic reference mean degree is read from \`results_network_analysis/metadata/synthetic_calibration.json\`; it is **not re-estimated from new UU matrices**. This preserves the previously agreed ER/BA/WS parameter values while keeping the new active workflow WD-only. Directed reciprocity is calculated from the two WD supports. Synthetic WD weights are sampled/calibrated from the retained empirical WD dependency-share samples. Synthetic matrices are benchmarks, not observed economic flows.

## Repository installation and tests

From the repository root:

\`\`\`bash
python -m pip install -r code/network_structural_analysis/requirements.txt
\`\`\`

Windows PowerShell:

\`\`\`powershell
$env:PYTHONPATH = "code"
\`\`\`

macOS/Linux:

\`\`\`bash
export PYTHONPATH=code
\`\`\`

Run tests:

\`\`\`bash
python -m unittest discover -s code/network_structural_analysis/tests -v
python -m compileall -q code/network_structural_analysis
\`\`\`

## Exact research execution sequence

Run these commands one at a time and inspect each validation report before proceeding. No stage automatically starts the next stage.

### 1. Preflight input files and node mapping

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage preflight --data-dir data --expected-nodes 4050
\`\`\`

This reads and validates both ICIO files, checks that each has 4,050 unique labels, and aligns 2022 to the 2019 label order. It does not construct structural metrics or start dynamics.

### 2. Real-network structural analysis only

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage structural --scope real --data-dir data --out results/revised_structural_analysis --threshold 0.001 --expected-nodes 4050 --seed 20261030
\`\`\`

This stage creates only \`real_2019\` and \`real_2022\` WD networks. It writes the sparse WD matrix, directed edge list, node mapping, node-level structural table, per-network metadata and real manifest. It does not generate synthetic networks or run SIS.

Validate the real results:

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage validate-structural --scope real --out results/revised_structural_analysis --expected-nodes 4050
\`\`\`

The validator checks both expected IDs, matrix checksums, dimensions, WD weight bounds, zero diagonal, node-level row counts, unique/stable IDs, complete node indices and finite metrics. Inspect \`results/revised_structural_analysis/diagnostics/real_structural_validation.json\`.

### 3. Synthetic structural analysis only

Run only after the real structural validation passed:

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage structural --scope synthetic --out results/revised_structural_analysis --expected-nodes 4050 --seed 20261030
\`\`\`

This generates exactly one realization for each ER/BA/WS configuration and persists only its WD matrix. It does not rerun real structural analysis or start SIS.

Validate the nine synthetic results:

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage validate-structural --scope synthetic --out results/revised_structural_analysis --expected-nodes 4050
\`\`\`

Inspect \`diagnostics/synthetic_structural_validation.json\`.

### 4. Validate all 11 structural networks

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage validate-structural --scope all --out results/revised_structural_analysis --expected-nodes 4050
\`\`\`

This writes \`diagnostics/structural_complete.json\` and \`manifests/all_11_network_manifest.csv\`. The all-network validator must report 2 real + 9 synthetic networks, exactly 3 each of ER/BA/WS, 4,050 nodes each, one realization per synthetic configuration, and WD-only matrix format.

### 5. Dynamics preflight (no simulations)

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage dynamics-preflight --out results/revised_structural_analysis --census-out results/single_node_shock_census --rho 0.2
\`\`\`

This stage is gated by the successful all-11 structural-completeness report. It verifies the same WD matrix checksums and node counts, calculates the spectral radius and \(\beta_c=\rho/\lambda_{\max}(B)\) separately for every network, and writes \`results/single_node_shock_census/diagnostics/dynamics_preflight.csv\`. A zero/non-finite spectral radius is an explicit error; no normalized-\(\eta\) run proceeds with an undefined threshold.

### 6. Exhaustive Stage 1 single-node shock census

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage census --out results/revised_structural_analysis --census-out results/single_node_shock_census --rho 0.2 --batch-size 128 --max-steps 300 --affected-threshold 0.01 --extinction-tolerance 1e-8 --convergence-tolerance 1e-8 --consecutive-steps 5 --long-run-window 30
\`\`\`

For each network, each node is shocked independently once at disruption 1; all other nodes start at zero, and the source recovers normally. The active equation is:

\[
d_j(t+1)=\operatorname{clip}_{[0,1]}\left[(1-\rho)d_j(t)+(1-d_j(t))\beta\sum_i B_{ij}d_i(t)\right],
\quad e(t)=B^\top d(t).
\]

Each network uses its own \(\beta_c=\rho/\lambda_{\max}(B)\) and \(\beta=\eta\beta_c\), with \(\eta=0.9\) and \(1.1\). For 11 networks of 4,050 nodes this is 89,100 independent trajectories. The actual count is calculated from the validated manifests.

The code uses sparse \(B^\top D\) multiplication over batches of independent source experiments. It does not keep all trajectory histories in memory. Every batch is written atomically, SHA-256 recorded in a checkpoint, and validated before being skipped on resume. A fully written batch whose checkpoint update was interrupted is recovered only after validating its source IDs and run condition. Full histories are not stored for every census node.

The eight output metrics are:

1. Maximum prevalence and its earliest time step.
2. Cumulative prevalence, using a documented trapezoidal integration rule over the common horizon.
3. Time to peak (earliest tied maximum).
4. Maximum affected nodes (\(d_j\geq0.01\), source included).
5. Final affected nodes at the common final horizon.
6. Long-run prevalence over the final 30 time points.
7. Extinction status, requiring max node disruption below \(10^{-8}\) for 5 consecutive steps.
8. Convergence time, requiring max absolute state change below \(10^{-8}\) for 5 consecutive steps; missing means not converged within the horizon.

Primary outputs include \`raw_metrics/single_node_census_metrics.csv.gz\`, per-network/eta metric summaries, separate rankings for cumulative prevalence, maximum prevalence, maximum affected nodes and long-run prevalence, batch checkpoints and diagnostics.

Validate completeness before comparisons:

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage census-validate --out results/revised_structural_analysis --census-out results/single_node_shock_census
\`\`\`

The census is not complete unless all expected source-node/eta combinations are present exactly once.

### 7. Structural–dynamic comparison and candidate selection

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage compare-select --out results/revised_structural_analysis --census-out results/single_node_shock_census --high-percentile 0.90 --low-percentile 0.10
\`\`\`

This joins node tables only on \`network_id\` and stable \`node_id\`, calculates Spearman correlations between all available structural metrics and the four impact measures, and compares 2019/2022 rank stability using matched country-industry labels. Correlations are descriptive and do not establish causality.

Candidate categories are:
- A: high structural percentile on one or more available structural measures.
- B: high cumulative-prevalence percentile at \(\eta=1.1\).
- C1: high structural composite and low dynamical percentile.
- C2: low structural composite and high dynamical percentile.

Percentiles are calculated within each network. The structural composite is the mean of available structural metric percentiles. Candidate reasons are retained together instead of duplicating nodes.

### 8. Stage 2 targeted individual-node SIS

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage targeted --out results/revised_structural_analysis --census-out results/single_node_shock_census --targeted-out results/targeted_individual_node_sis --rho 0.2 --batch-size 128 --max-steps 300
\`\`\`

Each candidate node is tested individually, with all other nodes starting at zero and the source recovering normally. Stage 2 uses the full approved \(\eta\) grid \(\{0.5,0.9,1.0,1.1,1.5,2.0\}\). It does not rerun the exhaustive census and does not create simultaneous targeted-node shocks or top-K group shocks. Results are compared with Stage 1 wherever the same \(\eta\) was tested.

Validate targeted results:

\`\`\`bash
python -m network_structural_analysis.run_pipeline --stage targeted-validate --targeted-out results/targeted_individual_node_sis
\`\`\`

## New output layout

\`\`\`text
results/
  revised_structural_analysis/
    real_networks/real_2019/
    real_networks/real_2022/
    synthetic_networks/syn_ER_1/ ... syn_WS_3/
    summaries/
    diagnostics/
    manifests/
    metadata/
  single_node_shock_census/
    config/
    raw_metrics/
    network_summaries/
    rankings/
    structural_dynamic_comparisons/
    targeted_candidates/
    diagnostics/
  targeted_individual_node_sis/
    raw_metrics/
    comparisons/
    diagnostics/
\`\`\`

The structural manifest includes one row per selected network, the exact WD matrix path and checksum, the node mapping and metric table paths, node/edge counts and metadata. Run manifests record the active Git branch/commit when locally available, run timestamps, library versions and scientific parameter settings.

## Legacy preservation

Historical outputs and old multi-representation helper modules are retained for reproducibility and compatibility. They are not used by the active CLI. The active configuration exposes \`REPRESENTATIONS=("WD",)\`; legacy helper code imports the explicitly named \`LEGACY_REPRESENTATIONS\`. Do not use the old \`results_network_analysis/dynamics\` outputs as revised census results.

## Scientific limitations

- A country-industry node is an economic production activity, not a firm.
- A dependency share measures a buyer's intermediate-purchase share associated with a supplier; it is not an outage probability.
- The SIS process is an abstract comparative propagation model, not a calibrated production-loss model.
- One realization per synthetic configuration does not characterize variability across the random graph-generating process. Do not calculate or claim between-realization confidence intervals.
- Synthetic network node IDs are configuration-specific and cannot be interpreted as the same economic activity across configurations.
- Spearman associations and node rankings do not establish causal economic effects.
- Historical differences between 2019 and 2022 should not be attributed to a specific event without independent evidence.

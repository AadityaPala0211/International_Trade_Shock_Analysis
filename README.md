# International Trade Shock Analysis — ICIO Networks

This repository constructs the 2019 and 2022 OECD Inter-Country Input-Output (ICIO) production networks, builds paired synthetic ER/BA/WS benchmarks, computes structural properties, and provides one deterministic SIS-style shock-propagation model.

The pipeline is staged deliberately. **Run threshold diagnostics first, inspect them, run the real networks, inspect those outputs, and only then generate the synthetic ensemble.** The real stage never launches synthetic generation automatically.

## Research design

The detailed country-industry node universe is expected to contain **4,050 nodes**. For each empirical year and each synthetic realization, the package creates three representations:

- **UU:** unweighted undirected projection.
- **UD:** unweighted directed supplier-to-buyer network.
- **WD:** weighted directed dependency network.

The transaction convention is fixed: \(Z_{ij}\) is the intermediate flow from supplier \(i\) to buyer \(j\). For buyer \(j\), dependency share is \(P_{ij}=Z_{ij}/\sum_k Z_{kj}\). Self-use remains in the denominator, while diagonal graph edges are excluded. The same manually selected dependency-share threshold \(\tau\) is applied to both empirical years. WD weights are not renormalized after thresholding.

Synthetic design: 3 ER configurations, 3 BA configurations and 3 WS configurations. Each configuration has 20 independent realizations by default (180 undirected skeletons). Each skeleton is transformed into paired UU, UD and WD matrices (540 synthetic matrices). Including the six empirical matrices gives **546 matrices**. Synthetic WD edge weights are dependency-strength-like values calibrated from the empirical networks; they are not real trade flows.

## Install

From the repository root:

```bash
python -m pip install -r code/network_structural_analysis/requirements.txt
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "code"
```

macOS/Linux:

```bash
export PYTHONPATH=code
```

## Required run sequence

### 1. Inspect the source files (optional)

```bash
python -m network_structural_analysis.run_pipeline --stage inspect --data-dir data
```

### 2. Generate threshold diagnostics only

```bash
python -m network_structural_analysis.run_pipeline --stage threshold --data-dir data --out results_network_analysis
```

Inspect the files in `results_network_analysis/threshold_review/` before choosing \(\tau\):

- `threshold_diagnostics.csv` and `threshold_pooled_summary.csv`
- Flow-coverage, density, mean-degree and degree-heterogeneity curves.
- Directed reciprocity and giant weak-component curves.
- `threshold_density_coverage_tradeoff.png`

The diagnostics include edge retention, off-diagonal monetary-flow coverage, directed density, in/out/undirected-projection mean degrees, degree coefficients of variation, degree-distribution quantiles, reciprocity and giant weak-component fraction. The code **does not select a threshold automatically**. Select a defensible value by reviewing the trade-off and any elbow in the curves. It is a sparsification parameter, not an observed economic cutoff.

### 3. Build and analyse the real 2019/2022 networks only

Replace `YOUR_REVIEWED_THRESHOLD` with the value you selected from step 2:

```bash
python -m network_structural_analysis.run_pipeline --stage real --data-dir data --out results_network_analysis --threshold YOUR_REVIEWED_THRESHOLD
```

This stage creates the six empirical matrices (2019/2022 × UU/UD/WD), edge lists, node-level metrics, country/industry summaries, network-level summaries, comparison tables and figures. It also saves the exact empirical weight and buyer-total samples used to calibrate synthetic WD networks.

**This stage does not generate any synthetic graph.** Review the real outputs and the selected threshold before continuing.

### 4. Generate the synthetic networks after approving the real results

```bash
python -m network_structural_analysis.run_pipeline --stage synthetic --out results_network_analysis --realizations 20 --seed 20261030
```

The synthetic stage uses the empirical calibration files saved by step 3; it does not reload or reconstruct the real networks. It creates 180 independent skeletons and persists all 540 paired representation matrices. It also saves node metrics and degree-frequency distributions for each representation of each realization.

The experiment parameters follow the plan:

- ER expected degree multipliers: **0.5×, 1.0×, 1.5×** the mean empirical UU degree; \(p=k_{target}/(N-1)\).
- BA target degree multipliers: **0.5×, 1.0×, 1.5×**; \(m\) is the nearest valid integer to \(k_{target}/2\).
- WS: fixed nearest even \(k\) to the empirical reference degree, with rewiring probabilities **0.01, 0.10, 0.50**.
- Directed orientation is calibrated to pooled empirical UD reciprocity.
- WD edge weights are sampled from empirical retained dependency shares. Buyer incoming totals are sampled conditionally on synthetic buyer in-degree; nearest-degree fallback is used only when the exact in-degree is absent.

### 5. Validate the persisted matrices before the SIS run

Do this only after the structural stages have finished and the empirical/synthetic manifests and matrices are present. This is a preflight; it **does not run SIS** and it does not write to the structural output folders.

For real networks only:

```bash
python -m network_structural_analysis.run_pipeline --stage dynamics-validate --scope real --out results_network_analysis --rho 0.2
```

For the final full experiment, after both empirical and synthetic networks are present:

```bash
python -m network_structural_analysis.run_pipeline --stage dynamics-validate --scope all --out results_network_analysis --rho 0.2
```

The full preflight checks all 546 matrices (six empirical and 540 synthetic), dimensions, finite/nonnegative weights, no diagonal links, binary UU/UD, WD weights in [0,1], symmetric UU matrices, and the paired support rules: WD must have the same directed support as UD, and UU must equal the undirected union of UD. It also verifies the six empirical year/representation combinations and the nine synthetic configurations with paired UU/UD/WD representations and common realization IDs/seeds. By default it requires exactly 20 realizations (IDs 1–20) in every synthetic configuration. The `--realizations` option can be set to a different positive count only when the structural synthetic stage was intentionally run with that count.

Review:

- `results_network_analysis/dynamics/validation_all/dynamics_input_precheck.csv`: one row per matrix, including edge count, checksum, largest eigenvalue and \(\beta_c\).
- `results_network_analysis/dynamics/validation_all/dynamics_input_validation.json`: the overall validation report and \(\beta_c\) ranges by representation.

Use the actual \(\beta_c\) values to choose the absolute \(\beta\) values. Do not copy the illustrative beta values below without checking their scale. For the final comparison across all 546 matrices, choose one \(\rho\) and one absolute \(\beta\) per representation, then hold those three betas fixed across every network. In particular, WD's weighted dependency-share scale differs from binary UU/UD.

### 6. Run the single SIS Model A

For the complete final comparison, run all matrices together after the full preflight passes:

```bash
python -m network_structural_analysis.run_pipeline --stage dynamics --scope all --out results_network_analysis --rho 0.2 --beta-uu 0.1 --beta-ud 0.1 --beta-wd 0.1
```

The beta values shown here are **illustrative syntax only**. Replace them with the values you chose after examining the preflight table. Keep the same values when running the full scope.

For an optional empirical-only trial before synthetic matrices exist:

```bash
python -m network_structural_analysis.run_pipeline --stage dynamics --scope real --out results_network_analysis --rho 0.2 --beta-uu 0.1 --beta-ud 0.1 --beta-wd 0.1
```

When the synthetic matrices exist, you can also preflight or run them separately using `--scope synthetic`. For a scientifically consistent comparison of the real and synthetic networks, however, prefer one `--scope all` run with common absolute beta values.

The single implemented model is:

\[
d_j(t+1)=\operatorname{clip}_{[0,1]}\left((1-\rho)d_j(t)
 +(1-d_j(t))\beta\sum_i B_{ij}d_i(t)\right),
\qquad e(t)=B^T d(t).
\]

Here \(B=A\) for UU and UD, and \(B=P\) for WD. The critical transmission is \(\beta_c=\rho/\lambda_{max}(B)\). The package runs one absolute-strength scenario and six normalized scenarios with \(\eta=\beta/\beta_c\) at \(\eta\in\{0.5,0.9,1.0,1.1,1.5,2.0\}\). It uses 20 real initial seed sets matched across 2019/2022 and all three representations, and five initial seed sets per synthetic skeleton reused across its paired UU/UD/WD matrices. The default initial infected fraction is 1% (41 nodes for 4,050 nodes).

Controls include `--max-steps`, `--tolerance`, `--real-seed-sets`, `--synthetic-seed-sets`, `--infection-fraction`, and `--seed`. \(\rho\) must be positive and at most one. Absolute beta values are constant within each representation across the matrices in a run; do not silently retune them by graph.

The dynamics run writes only beneath `results_network_analysis/dynamics/{real,synthetic,all}/`. It does not overwrite empirical or synthetic structural matrices, node metrics, edge lists or structural summaries.

## Output structure

`results_network_analysis/` includes:

- `threshold_review/`: threshold diagnostics and figures.
- `metadata/`: run metadata, input SHA-256 checksums, node index, workbook preview, conservative source-provenance audit and saved synthetic calibration samples.
- `empirical/{2019,2022}/{UU,UD,WD}/`: sparse `adjacency.npz`, edge list, node metrics, country/industry summaries and distributions.
- `empirical_network_manifest.csv`: exact paths to all six empirical matrices.
- `comparisons/` and `figures/`: year-to-year comparisons and structural plots.
- `synthetic/matrices/{ER,BA,WS}/config_*/realization_*/{UU,UD,WD}.npz`: all 540 synthetic matrices.
- `synthetic/node_metrics/` and `synthetic/degree_distributions/`: per-matrix node metrics plus separate in-degree, out-degree and undirected-projection degree distributions.
- `synthetic/synthetic_network_manifest.csv`: matrix paths and paired realization seeds.
- `dynamics/validation_{real,synthetic,all}/`: preflight matrix validation, checksums and per-matrix spectral thresholds; no SIS simulation is run by this stage.
- `dynamics/{real,synthetic,all}/`: seed-level and network-level Model A summaries, per-network trajectories, seed-set/graph-level confidence intervals, aggregated trajectory tables, and plots under `trajectory_summaries/plots/`.

Do not commit generated results or duplicate source CSV files unless intentionally required.

## Source provenance note

The official [OECD ICIO dataset page](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html) lists a 2025 regular ICIO release covering 1995–2022. The pipeline inspects the companion `ReadMe_ICIO_small.xlsx`, records the local CSV checksums and label universe, and writes `metadata/source_provenance_audit.json`. A companion workbook is not conclusive, file-specific proof of which release each CSV came from. Preserve the original OECD archive/download reference or source manifest for the 2019 and 2022 files.

## Structural measures

Network-level outputs include nodes, edges, density, degree statistics, in/out degree heterogeneity, weak and strong components, clustering, sampled shortest-path distance, reciprocity, weighted strengths and spectral radius. Node-level outputs include in/out degree and strength, PageRank, betweenness, k-core and clustering. Paths and betweenness are source-sampled for large matrices; sampling settings are recorded.

## SIS metrics and uncertainty

Model outputs include prevalence trajectories, mean exposure, peak prevalence, time-to-peak, long-run prevalence, cumulative prevalence, extinction time, convergence, \(\lambda_{max}\) and \(\beta_c\). Seed-set variation is retained within each network. Synthetic graph-to-graph summaries and 95% confidence intervals are computed across independent realizations, separately from within-network seed-set uncertainty.

## Tests

```bash
PYTHONPATH=code python -m unittest discover -s code/network_structural_analysis/tests -v
python -m compileall -q code/network_structural_analysis
```

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the detailed definitions and assumptions.

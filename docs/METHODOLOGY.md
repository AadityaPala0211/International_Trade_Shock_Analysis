# Network construction and structural-analysis methodology

## 1. Scope

This stage is a structural study of the OECD Inter-Country Input-Output (ICIO) production network for 2019 and 2022. It stops before any temporal shock propagation or dynamics. The unit of analysis is a country-industry production activity, not a firm.

The intended detailed node universe contains 4,050 country-industry nodes. The loader validates the requested node count and does not silently aggregate to broad sectors.

## 2. ICIO transaction block and direction

Let \`Z\` be the square intermediate-use block. \`Z[i, j]\` is the intermediate transaction from supplier node \`i\` to buyer node \`j\`; graph direction is therefore \`i -> j\`. The package extracts the leading 4,050 × 4,050 intermediate block, with supplier labels in the corresponding row-label column. Final-demand columns and non-intermediate rows outside this block are not graph nodes.

The code stores row/column block coordinates and input SHA-256 checksums. The companion \`ReadMe_ICIO_small.xlsx\`, when present, is inspected into a bounded metadata preview and is not treated as transaction data.

## 3. Dependency-share normalization

For buyer \`j\`, let total intermediate purchases be \`p[j] = sum_k Z[k, j]\`. The off-diagonal dependency share is \`S[i, j] = Z[i, j] / p[j]\` for \`i != j\`, and \`S[j, j] = 0\` in the graph. Self-use remains in \`p[j]\`. This gives each supplier contribution a buyer-specific scale. A share is not a probability of supplier failure and is not a monetary flow.

The total retained share entering buyer \`j\` in a thresholded WD graph can be below one because of self-use and edges removed by the sparsification threshold.

## 4. Three paired representations

A single common dependency-share threshold \`tau\` is applied to both years:

- **UU:** one undirected edge \`{i,j}\` exists when either \`S[i,j] >= tau\` or \`S[j,i] >= tau\`. Each undirected edge is stored once in the exported edge list.
- **UD:** a binary directed edge \`i -> j\` exists when \`S[i,j] >= tau\`.
- **WD:** the same directed support as UD, with edge weight equal to \`S[i,j]\`.

Diagonal self-links are always excluded. Adjacency matrices retain all 4,050 nodes, including nodes isolated by the selected threshold.

## 5. Threshold diagnostics

\`tau\` controls sparsification; it is not an economically observed cutoff. A coarse candidate grid is evaluated separately for 2019 and 2022. For each candidate and year, diagnostics include:

- retained off-diagonal monetary-flow coverage;
- retained edge fraction;
- directed density and reciprocity;
- giant weak-component fraction;
- out-degree coefficient of variation.

The common candidate recommendation is deliberately heuristic. It first looks for thresholds preserving at least 90% of off-diagonal flow in both years while keeping mean directed density at or below 20%; among those, it prefers a threshold with a large weak giant component. If no candidate meets those conditions, it selects the candidate closest to 95% flow coverage with a density penalty. The rationale and a local sensitivity table around the selected value are written to disk. A user-supplied threshold overrides the recommendation without deleting diagnostics.

Inspect the curves before accepting a candidate. The recommendation is not a unique mathematical optimum.

## 6. Synthetic benchmark ensemble

The calibration target for undirected skeletons is the mean degree of the thresholded empirical UU networks averaged across years. Three configurations per family are generated:

- **ER:** target mean degree at 0.75×, 1.00× and 1.25× the empirical reference; \`p = k_target / (N - 1)\`.
- **BA:** attachment parameter \`m\` is the rounded half of the corresponding target mean degree, bounded to the valid range.
- **WS:** nearest even neighbourhood degree to the empirical reference is held fixed; rewiring probabilities are 0.01, 0.10 and 0.50.

Twenty graph realizations are generated per configuration by default, giving 180 independent skeletons. Each skeleton is reused to form three paired representation tracks (UU, UD and WD), giving 540 track instances. The same skeleton is reused so comparisons across representations are paired rather than based on unrelated random graphs.

To create UD/WD tracks, each undirected skeleton edge is oriented once. Direction is made reciprocal with a probability calibrated so expected arc-level reciprocity matches empirical directed UD reciprocity. WD assigns positive synthetic edge weights sampled from the threshold-retained empirical dependency-share distribution, then rescales incoming weights buyer-by-buyer so their sum follows the empirical distribution of threshold-retained buyer dependency totals. These weights are synthetic dependency-like values, not OECD monetary transactions.

## 7. Structural measures and definitions

Network-level measures include node/edge counts, representation-specific density, degree heterogeneity, weighted in/out strength, clustering on the undirected projection, weak and strong component statistics, sampled shortest-path distances, directed reciprocity and spectral radius.

Node-level measures include in/out degree, undirected-projection degree, in/out strength, PageRank, betweenness, k-core and local clustering. WD PageRank uses dependency shares as edge weights. Betweenness is estimated by deterministic sampled Brandes traversals for large networks. K-core and clustering use the undirected projection.

For 4,050 nodes, betweenness is an approximation based on a reproducible source sample (default 64). Increase \`--betweenness-k\` to improve sampling precision at greater computational cost. Average path length is also source-sampled (default 256) and reports the fraction of sampled ordered pairs reachable. Directed paths retain direction; weak components and clustering explicitly use the undirected projection.

Sparse-first implementations are used for PageRank, components, shortest paths, k-core, clustering and Brandes traversal to avoid materialising millions of Python edge objects.

## 8. 2019–2022 comparisons

Node metrics are aligned by country-industry label. If the two CSVs contain the same label set in different orders, the 2022 matrix is explicitly reordered to the 2019 label order; different node universes fail the run. The pipeline computes Spearman rank correlations and mean absolute changes for comparable node-level statistics, plus top-node tables for strengths and centrality.

Country and industry summaries are grouped from label tokens split at the first underscore. The original node label is preserved in the node-level table; review inferred \`country_code\` and \`industry_code\` fields if the source-label convention changes.

## 9. Interpretation limits

- The annual tables describe input-output structure; they do not observe propagation speed or recovery.
- A thresholded edge indicates material dependency under the chosen cutoff, not a known outage channel.
- Structural centrality is a graph statistic, not a direct measure of real-world systemic loss.
- Synthetic ER/BA/WS networks are null-model references, not forecasts.
- Synthetic weights do not recreate actual transactions, production capacity or substitution.
- This package does not estimate shock dynamics, actual output losses, resilience interventions or causal effects.

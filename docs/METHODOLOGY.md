# Network construction, structural analysis and SIS shock propagation

## 1. Scope and experimental units

The empirical data are the OECD Inter-Country Input-Output (ICIO) tables for 2019 and 2022. The intended detailed production-activity node universe is 4,050 country-industry nodes. Each node is an economic activity, not a firm. The pipeline preserves the detailed node universe and rejects unexpected dimensions or incompatible node sets.

The design comprises six empirical matrices (two years × three representations) and nine synthetic configurations (three each for ER, BA and WS). Twenty independent synthetic realizations are generated per configuration by default. One undirected skeleton is reused to create all three representations. Therefore the final structural design contains 180 skeletons, 540 synthetic representation-specific matrices and 546 matrices overall including empirical networks.

## 2. ICIO transaction block and direction

Let \(Z\) be the square intermediate-use block. \(Z_{ij}\) is the intermediate transaction from supplier node \(i\) to buyer node \(j\); direction is therefore \(i\rightarrow j\). The package extracts the detailed square intermediate block and validates its unique labels. If the header's column labels are a permutation of the row labels, columns are reordered into row-label order. For the two years, a label-set mismatch causes the pipeline to stop; if their labels appear in a different order, both axes of the later-year matrix are explicitly reordered to the canonical 2019 order.

The companion `ReadMe_ICIO_small.xlsx` is inspected as documentation rather than transaction data. CSV SHA-256 hashes, dimensions, node-order hashes and block coordinates are written to metadata. The OECD's official ICIO dataset page lists a 2025 regular ICIO release for 1995–2022. The local provenance audit distinguishes this official release context from direct file-level proof: a companion readme does not independently prove which archive each individual CSV came from. Original download/archive references should be retained.

## 3. Dependency-share normalization

For buyer \(j\), total intermediate purchases are:

\[
p_j=\sum_k Z_{kj}.
\]

The supplier dependency share is:

\[
P_{ij}=\frac{Z_{ij}}{p_j},\qquad p_j>0.
\]

The diagonal is set to zero in the graph representation, but self-use remains in \(p_j\). A retained dependency share is not a supplier failure probability and is not a monetary flow. After thresholding, incoming retained WD weights are not renormalized; their sum may be below one due to self-use and removed dependencies.

## 4. Three paired empirical representations and threshold

A common dependency-share threshold \(\tau\) is used for both years:

- **UU:** undirected binary edge \(\{i,j\}\) exists when either \(P_{ij}\geq\tau\) or \(P_{ji}\geq\tau\).
- **UD:** binary directed edge \(i\rightarrow j\) exists when \(P_{ij}\geq\tau\).
- **WD:** the same directed support as UD, weighted by \(P_{ij}\).

Diagonal self-links are excluded. All 4,050 nodes are retained, including isolates.

Thresholding is a network sparsification choice, not an observed economic cutoff. The threshold stage evaluates all candidate values in both years. Diagnostics include retained off-diagonal monetary-flow coverage, edge retention, directed density, mean in/out/undirected-projection degree, degree coefficients of variation, degree-distribution quantiles, reciprocity and giant weak-component fraction. The pipeline plots curves and the flow-coverage/density trade-off.

**No automatic threshold is selected.** The analyst reviews those diagnostics and any elbow in the curves, chooses a common \(\tau\), and supplies it to the real-network stage. The chosen threshold and nearby sensitivity values are then recorded. The same \(\tau\) is used for all real years and representations.

## 5. Structural analysis

Network metrics include node/edge counts, representation-specific density, mean degree, degree heterogeneity, in/out degree and strength, clustering on the undirected projection, weak/strong component summaries, sampled average path length, directed reciprocity and spectral radius.

Node metrics include in/out degree, undirected-projection degree, in/out strength, PageRank, betweenness, k-core and local clustering. For directed networks, path distances and betweenness retain direction; weak components, clustering and k-core use the undirected projection. Betweenness and characteristic path length use reproducible source samples for large graphs. Per-node outputs are saved for each empirical matrix and each synthetic realization/representation.

## 6. Synthetic benchmark design

Let \(k_{ref}\) be the mean of the 2019 and 2022 empirical UU mean degrees.

### ER

The three target expected degrees are \(0.5k_{ref}\), \(k_{ref}\) and \(1.5k_{ref}\). For target \(k_{target}\):

\[
p=\frac{k_{target}}{N-1}.
\]

### BA

The BA attachment parameter is derived from the same three target degrees:

\[
m=\operatorname{round}\left(\frac{k_{target}}{2}\right),
\]

bounded to the valid interval for a BA graph. The requested target degree and actual \(2m\) approximation are recorded.

### WS

The neighbourhood degree \(k\) is the nearest valid even integer to \(k_{ref}\), fixed across configurations. Rewiring probabilities are 0.01, 0.10 and 0.50.

### Paired directed/weighted transforms

For each undirected skeleton, every edge is independently oriented; some dyads are made mutual. The mutual-dyad probability is calibrated so the expected arc-level directed reciprocity matches pooled empirical UD reciprocity. The same orientation is used to make the UD and WD representations.

WD synthetic edge weights are sampled from positive empirical dependency shares retained at the selected threshold. For each synthetic buyer, its target incoming dependency total is sampled from empirical buyers with the same retained in-degree. If that exact degree has no observed samples, the nearest observed positive in-degree is used. Sampled edge values are rescaled buyer-by-buyer to match the sampled target total. This preserves the target buyer-total distribution conditional on in-degree as closely as the empirical sample permits. These values are synthetic dependency strengths, not observed monetary trade flows.

All 540 sparse matrices are saved. Each skeleton has a graph seed, orientation seed, weight seed and parameter record. Every matrix has a manifest entry and saved node-level metrics and degree-frequency table.

## 7. Single dynamical model: Alternate Model A

Only one dynamical model is implemented: the deterministic mass-action/bilinear SIS (NIMFA-style) update:

\[
d_j(t+1)=\operatorname{clip}_{[0,1]}\left[
(1-\rho)d_j(t)+(1-d_j(t))\beta\sum_i B_{ij}d_i(t)
\right],
\qquad e(t)=B^T d(t).
\]

Here \(d_j(t)\in[0,1]\) is the infection/shock state of production node \(j\); \(e(t)\) is the vector of incoming exposure. The supplier-to-buyer orientation means the force on buyer \(j\) is \((B^T d)_j\). The same equation is used for every one of the 546 matrices:

- UU: \(B=A_{UU}\), binary undirected adjacency.
- UD: \(B=A_{UD}\), binary directed supplier-to-buyer adjacency.
- WD: \(B=P_{WD}\), threshold-retained weighted dependency-share matrix.

The critical transmission is:

\[
\beta_c=\frac{\rho}{\lambda_{max}(B)}.
\]

The two experiments are:

1. **Absolute transmission:** a common \(\rho\) and one fixed absolute \(\beta\) per representation, held constant across all networks of that representation. Edge-scale differences between binary and weighted matrices must be acknowledged when interpreting this comparison.
2. **Threshold-normalized transmission:** \(\eta=\beta/\beta_c\), with \(\eta\in\{0.5,0.9,1.0,1.1,1.5,2.0\}\).

The initial infected fraction is 1% (rounded up to the nearest node). Twenty independently sampled seed sets are used for the real networks and are matched across 2019/2022; the same sets are also paired across the three real representations. Five independent seed sets are generated per synthetic skeleton and reused unchanged across UU/UD/WD for that realization.

Outputs include prevalence trajectories, mean exposure, peak prevalence, time-to-peak, long-run prevalence, cumulative prevalence, extinction time, convergence, spectral radius and critical transmission. Seed-set uncertainty is reported within each network. Synthetic ensemble confidence intervals are then calculated across the 20 independent graph realizations, keeping the graph-to-graph and within-network seed-set variation separate.

No alternative SIS variants, signed model, Leontief model or additional dynamical hierarchy are included.

## 8. 2019–2022 and synthetic comparisons

Node-level empirical metrics are aligned by country-industry label. The package computes Spearman rank correlations and mean absolute changes for comparable node metrics and saves top-node tables. Synthetic comparisons retain family, parameter configuration, realization ID and representation, allowing ER/BA/WS comparisons and paired UU/UD/WD comparisons on the same skeleton.

Country and industry summaries currently group node labels by tokens split at the first underscore. The original labels remain intact in node-level files; the inferred country/industry tokens should be reviewed if the ICIO label convention differs.

## 9. Interpretation limits

- A country-industry node is a production activity, not a firm.
- A dependency share measures the fraction of a buyer's intermediate purchases attributed to a supplier; it is not an outage probability.
- UU removes direction; UD preserves direction without magnitude; WD preserves direction and threshold-retained dependency magnitude.
- ER/BA/WS networks are null-model benchmarks, not forecasts or replicas of actual trade.
- Synthetic weights do not recreate actual transactions, production capacity or substitution.
- The SIS process is an abstract comparative propagation model, not a calibrated production-loss model. Its absolute outcomes depend on the chosen \(\beta\), \(\rho\), time horizon and threshold.
- Centrality, spectral radius and SIS outcomes are network-model quantities; they do not by themselves establish causal economic losses.

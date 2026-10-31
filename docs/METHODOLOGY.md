# Revised methodology: WD-only structure and individual-source SIS

## 1. Research scope and selected networks

The revised active analysis contains exactly 11 full-resolution networks:

- OECD ICIO 2019 and OECD ICIO 2022 (two empirical networks).
- ER configurations 1–3 (three synthetic networks).
- BA configurations 1–3 (three synthetic networks).
- WS configurations 1–3 (three synthetic networks).

Each network contains the full 4,050 country-industry node universe. There is exactly one random realization per synthetic parameter configuration. The one realization is a single graph sample, not an estimate of graph-generation variability.

The active processing format is WD only. Each selected network has one persisted sparse matrix, one node-level structural result set and one corresponding SIS analysis. UU and UD are not persisted or selected by the active CLI. Historical result files are preserved outside the revised result folders.

## 2. ICIO transaction orientation and WD matrix construction

Let \(Z\) be the detailed square intermediate transaction block. \(Z_{ij}\) denotes the input supplied by production activity \(i\) to buyer activity \(j\). Thus matrix row \(i\) is the supplier and column \(j\) is the buyer, with direction \(i\rightarrow j\).

Buyer \(j\)'s total intermediate purchases are:

\[
p_j=\sum_k Z_{kj}.
\]

The dependency share is:

\[
P_{ij}=\frac{Z_{ij}}{p_j},\quad p_j>0.
\]

The denominator includes self-use. Self-links are excluded from the graph after the share is computed. The common empirical threshold is the previously reviewed value \(\tau=0.001\), recorded in the existing project calibration metadata. The retained matrix is:

\[
B_{ij}=
\begin{cases}
P_{ij},&P_{ij}\geq\tau,\ i\ne j,\ P_{ij}>0,\\
0,&\text{otherwise}.
\end{cases}
\]

No post-threshold renormalization is applied. The threshold is a sparsification choice, not an observed economic cutoff. A retained dependency share is not a supplier failure probability or a monetary flow.

The 2022 node labels are aligned to the canonical 2019 label order. The loader rejects a different country-industry label universe or an unexpected matrix size. Real node identifiers are the original stable ICIO activity labels; the row index is also saved separately.

## 3. Synthetic WD networks

Three existing configurations per family are retained. ER target expected degrees and BA target-degree multipliers are \(0.5,1.0,1.5\) times the approved reference mean degree; ER uses \(p=k_{target}/(N-1)\), and BA uses \(m=\operatorname{round}(k_{target}/2)\) subject to graph constraints. WS uses the previously established fixed even \(k\) and rewiring probabilities \(0.01,0.10,0.50\).

To preserve the already approved degree-calibration values without building UU matrices in the active workflow, the reference mean degree is read from the prior approved synthetic calibration metadata. Directed edge orientation is generated on a temporary skeleton; only a WD matrix is saved. The empirical WD directed support determines the arc-level reciprocity target. Positive retained empirical WD dependency shares are sampled for synthetic edges. Synthetic buyer incoming dependency totals are sampled conditional on synthetic buyer in-degree, with nearest observed positive in-degree used only when an exact in-degree is absent. Synthetic weights are rescaled per buyer to match the sampled buyer total. These are synthetic dependency strengths, not real trade flows.

The synthetic identifiers include family, configuration and node index. IDs are stable within a synthetic configuration and are intentionally not assumed to represent matching economic activities across different random configurations.

## 4. Structural metrics

For each network, the implementation saves node/edge counts, density, in-degree, out-degree, weighted in-strength, weighted out-strength, PageRank, betweenness, k-core and clustering. It also retains the other network-level statistics returned by the existing validated WD-compatible structural metric implementation.

Direction is preserved for degree, strengths, directed shortest paths and betweenness. The supplier-to-buyer interpretation is:
- Out-degree: number of direct buyer activities supplied by the node.
- Out-strength: total retained dependency-share weight of its outgoing supplier-to-buyer links.
- In-degree: number of supplier activities linked into the buyer.
- In-strength: sum of retained dependency shares entering the buyer.

PageRank uses the WD weights. Betweenness uses directed paths and binary edge lengths. k-core and local clustering are calculated on the undirected projection for those measures only; this does not create an additional active network matrix or SIS representation. Metrics based on sampling use deterministic seeds and record their sampling configuration.

## 5. Structural workflow and completeness gates

Real structural analysis and synthetic structural analysis are separate commands:

1. Generate the two real WD networks.
2. Validate both real network result sets, including node labels, checksums and metric rows.
3. Generate nine synthetic WD networks (one realization per configuration).
4. Validate all nine synthetic networks.
5. Validate all 11 networks together.

The synthetic stage requires a valid real structural validation report and the empirical WD samples used to calibrate its edge weights. The all-network structural validation writes a completion marker. SIS preflight requires this marker to be valid. No structural stage automatically starts the next stage.

Validation checks network identity, family/configuration/year, matrix format, node count, edge count, finite positive weights bounded by one, zero diagonal, checksums, expected node-level records, unique IDs, complete node-index mapping, duplicate/missing data and non-finite metric values.

## 6. Alternate Model A: mass-action bilinear SIS

Only the approved deterministic SIS/NIMFA-inspired disruption model is active:

\[
d_j(t+1)=\operatorname{clip}_{[0,1]}
\left[(1-\rho)d_j(t)+(1-d_j(t))\beta e_j(t)\right],
\qquad e(t)=\sum_i B_{ij}d_i(t).
\]

In vector notation:

\[
e(t)=B^\top d(t).
\]

Here \(d_j(t)\in[0,1]\) is the disruption state of activity \(j\), \(\rho\) is the recovery parameter and \(\beta\) is the transmission parameter. Matrix \(B\) is always the same WD supplier-to-buyer dependency-share matrix used for structural analysis. An edge \(B_{ij}\) exposes buyer \(j\) to a disruption at supplier \(i\).

The active recovery parameter is retained as \(\rho=0.2\), as recorded in the current approved SIS run metadata. The network-specific spectral threshold is:

\[
\beta_c=\frac{\rho}{\lambda_{\max}(B)},\qquad\beta=\eta\beta_c.
\]

Each network has its own spectral radius, \(\beta_c\) and \(\beta\). A non-finite or zero spectral radius is rejected for the normalized-\(\eta\) analysis.

## 7. Stage 1: exhaustive one-node pulse census

For every node \(s\) in every network, initialize one independent source experiment:

\[
d_i(0)=
\begin{cases}
1,&i=s,\\
0,&i\ne s.
\end{cases}
\]

Only that source starts fully disrupted. The source is not clamped or reinjected; it recovers through the same update equation as all other nodes. Every source experiment is independent. The stage uses exactly \(\eta=0.9\) and \(\eta=1.1\).

For a batch of \(b\) source experiments:

\[
D(t)\in\mathbb{R}^{N\times b},\qquad
E(t)=B^\top D(t),
\]

\[
D(t+1)=\operatorname{clip}_{[0,1]}
\left[(1-\rho)D(t)+(1-D(t))\odot\beta E(t)\right].
\]

Each column is an independent trajectory. Sparse matrix multiplication is used; the implementation stores only the current state and compact per-time prevalence/affected-count arrays for the current batch. It does not keep all node histories for all source experiments.

For 11 networks with 4,050 nodes each, the expected census count is \(11\times4050\times2=89,100\) independent trajectories. Actual counts come from the manifests.

### Eight dynamical metrics

Network prevalence is:

\[
I(t)=\frac{1}{N}\sum_jd_j(t).
\]

1. **Maximum prevalence:** \(I_{max}=\max_t I(t)\). Earliest time of a tied maximum is retained.
2. **Cumulative prevalence:** \(\int_0^T I(t)dt\), evaluated with the trapezoidal rule and unit time spacing.
3. **Time to peak:** earliest discrete time step where maximum prevalence occurs.
4. **Maximum affected nodes:** largest count at any time with \(d_j(t)\geq0.01\). The source counts if it meets the threshold.
5. **Final affected nodes:** count at the common final horizon \(T\) with \(d_j(T)\geq0.01\).
6. **Long-run prevalence:** mean prevalence over the final 30 observed time points.
7. **Extinction status:** true only when \(\max_jd_j(t)<10^{-8}\) for 5 consecutive steps. Otherwise false by the end of the horizon.
8. **Convergence time:** first step satisfying \(\max_j|d_j(t+1)-d_j(t)|<10^{-8}\) for 5 consecutive steps. It remains missing if convergence is not reached within the maximum horizon.

The common defaults are a 300-step horizon, affected threshold 0.01, extinction and convergence tolerances \(10^{-8}\), 5 consecutive steps, and a final window of 30 time points. Settings are recorded in run metadata. All metrics are kept separate.

### Checkpointing and reproducibility

Source-node batches are written to temporary files and atomically renamed. The checkpoint records each fully written batch, source count and SHA-256. A batch is skipped only if its configuration fingerprint matches and its output checksum validates. If a process stops after the atomic batch file write but before its checkpoint entry is saved, the next run adopts the orphan only after checking its ordered source IDs, network and eta. A corrupted or mismatched file is never marked complete. Source-node/eta keys are checked for duplicates and omissions before census completion is recorded. Full trajectory histories are not retained for all source nodes.

## 8. Structural–dynamic comparison and candidate selection

After census validation, structural and dynamic node data are joined by \((network\_id,node\_id)\); row order is never used. Spearman rank correlations are computed separately by network and eta between every available structural metric and each of cumulative prevalence, maximum prevalence, maximum affected nodes and long-run prevalence.

For real networks, 2019 and 2022 ranking stability is calculated only for matching country-industry labels. For synthetic configurations, analysis is performed within each configuration; anonymous synthetic node IDs are not matched across configurations. Correlation does not imply causality.

Candidate selection uses within-network percentiles at \(\eta=1.1\):
- **A — structural:** node is at or above the high percentile in at least one available structural metric.
- **B — dynamic:** node is at or above the high percentile in cumulative prevalence.
- **C1 — structural high, dynamic low:** mean available structural percentile is high and cumulative-prevalence percentile is low.
- **C2 — structural low, dynamic high:** mean available structural percentile is low and cumulative-prevalence percentile is high.

Default cutoffs are 90th percentile and 10th percentile. Candidate rows retain structural/dynamic values, percentiles, categories and all applicable selection reasons.

## 9. Stage 2: targeted individual-node SIS

Each selected candidate is tested in its own reset experiment using the same one-time pulse shock. Only one source is initialized to one; all other nodes start at zero. The source is allowed to recover normally.

The approved Stage 2 threshold-normalized grid is \(\eta\in\{0.5,0.9,1.0,1.1,1.5,2.0\}\). All eight metrics use the same definitions and settings as Stage 1. Stage 2 uses validated candidate outputs and does not rerun the exhaustive census. Simultaneous candidate shocks, top-K group shocks and simultaneous random-set comparisons are not implemented.

For eta levels also used in Stage 1 (0.9 and 1.1), targeted results are joined back to the census for direct consistency comparison. The remaining Stage 2 scenarios are retained as additional individual-node experiments.

## 10. Cross-network comparisons and limitations

The nine synthetic configurations are compared individually and by family. Since each configuration has one realization, no between-realization confidence interval or graph-generation variability estimate is reported. The results characterize the selected realization only.

The OECD ICIO 2019 and 2022 networks are compared using stable country-industry IDs. Changes in rankings, source shock impacts and structural-dynamic correlations are descriptive. They are not attributed to specific historical events without independent evidence.

The SIS process is an abstract propagation model over a WD dependency network. It is not a calibrated production-loss model. Its outcomes depend on \(\rho\), matrix threshold, horizon and normalized transmission level.

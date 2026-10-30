"""Threshold diagnostics. No threshold is selected automatically."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from .network_builder import dependency_shares


def _quantiles(values, prefix):
    values = np.asarray(values, dtype=float)
    q = np.quantile(values, [0, .25, .5, .75, .90, .99, 1]) if values.size else np.zeros(7)
    names = ("min", "q25", "median", "q75", "q90", "q99", "max")
    return {f"{prefix}_{name}": float(value) for name, value in zip(names, q)}


def _quick_topology(mask):
    """Compute threshold diagnostics without creating a NetworkX graph."""
    A = sparse.csr_matrix(mask.astype(np.uint8))
    A.eliminate_zeros()
    n, m = A.shape[0], A.nnz
    density = m / (n * (n - 1)) if n > 1 else 0.0
    mutual = A.multiply(A.T).nnz
    reciprocity = mutual / m if m else 0.0
    U = A.maximum(A.T).tocsr()
    U.setdiag(0)
    U.eliminate_zeros()
    _, labels = connected_components(U, directed=False, return_labels=True)
    sizes = np.bincount(labels) if n else np.array([])
    giant = sizes.max() / n if len(sizes) else 0.0
    out_degree = np.asarray(A.getnnz(axis=1)).ravel()
    in_degree = np.asarray(A.getnnz(axis=0)).ravel()
    undirected_degree = np.asarray(U.getnnz(axis=1)).ravel()

    def cv(x):
        mean = float(np.mean(x)) if len(x) else 0.0
        return float(np.std(x) / mean) if mean else 0.0

    result = {
        "directed_density": float(density),
        "reciprocity": float(reciprocity),
        "giant_component_fraction": float(giant),
        "mean_out_degree": float(out_degree.mean()) if n else 0.0,
        "mean_in_degree": float(in_degree.mean()) if n else 0.0,
        "mean_undirected_degree": float(undirected_degree.mean()) if n else 0.0,
        "out_degree_cv": cv(out_degree),
        "in_degree_cv": cv(in_degree),
        "undirected_degree_cv": cv(undirected_degree),
    }
    result.update(_quantiles(out_degree, "out_degree"))
    result.update(_quantiles(in_degree, "in_degree"))
    result.update(_quantiles(undirected_degree, "undirected_degree"))
    return result


def diagnose_thresholds(year_matrices, candidates, out_dir=None,
                        selected_threshold=None, local_sensitivity_factor=3.0):
    """Evaluate every candidate for both years; return diagnostics and shares.

    The diagnostic candidate grid is not an automatic selection rule. The analyst
    must inspect the trade-off/degree plots and provide a common threshold in the
    subsequent real-network stage.
    """
    candidates = tuple(sorted(set(float(x) for x in candidates)))
    if not candidates or any((not np.isfinite(x) or x < 0) for x in candidates):
        raise ValueError("Threshold candidates must be finite, nonnegative and non-empty.")

    records, shares_by_year = [], {}
    for year, Z in year_matrices.items():
        S, purchases = dependency_shares(Z)
        shares_by_year[year] = S
        offdiag_total = float(Z.sum() - np.trace(Z))
        positive_count = max(1, int(np.count_nonzero(S > 0)))
        for tau in candidates:
            mask = (S >= tau) & (S > 0)
            np.fill_diagonal(mask, False)
            edges = int(mask.sum())
            retained_flow = float(np.sum(Z, where=mask))
            topology = _quick_topology(mask)
            records.append({
                "year": int(year), "tau": tau, "directed_edges": edges,
                "positive_offdiagonal_edges": positive_count,
                "edge_retention": edges / positive_count,
                "offdiag_flow_coverage": retained_flow / offdiag_total if offdiag_total else 0.0,
                **topology,
            })

    diagnostics = pd.DataFrame(records)
    pooled = diagnostics.groupby("tau", as_index=False).agg(
        mean_flow_coverage=("offdiag_flow_coverage", "mean"),
        min_flow_coverage=("offdiag_flow_coverage", "min"),
        mean_edge_retention=("edge_retention", "mean"),
        mean_density=("directed_density", "mean"),
        mean_out_degree=("mean_out_degree", "mean"),
        mean_in_degree=("mean_in_degree", "mean"),
        mean_undirected_degree=("mean_undirected_degree", "mean"),
        mean_giant_component=("giant_component_fraction", "mean"),
        mean_reciprocity=("reciprocity", "mean"),
        mean_out_degree_cv=("out_degree_cv", "mean"),
        mean_in_degree_cv=("in_degree_cv", "mean"),
        mean_undirected_degree_cv=("undirected_degree_cv", "mean"),
        mean_edges=("directed_edges", "mean"),
    )

    if selected_threshold is None:
        sensitivity = []
        selection_note = ("No threshold selected. Review the diagnostics and figures, then rerun "
                          "the real stage with --threshold set to the analyst-selected common value.")
    else:
        selected_threshold = float(selected_threshold)
        if not np.isfinite(selected_threshold) or selected_threshold < 0:
            raise ValueError("selected_threshold must be finite and nonnegative.")
        local = sorted(set((max(0.0, selected_threshold / local_sensitivity_factor),
                            selected_threshold,
                            min(1.0, selected_threshold * local_sensitivity_factor))))
        for year, Z in year_matrices.items():
            S = shares_by_year[year]
            offdiag_total = float(Z.sum() - np.trace(Z))
            positive_count = max(1, int(np.count_nonzero(S > 0)))
            for tau in local:
                mask = (S >= tau) & (S > 0)
                np.fill_diagonal(mask, False)
                retained_flow = float(np.sum(Z, where=mask))
                sensitivity.append({
                    "year": int(year), "tau": tau,
                    "directed_edges": int(mask.sum()),
                    "edge_retention": int(mask.sum()) / positive_count,
                    "offdiag_flow_coverage": retained_flow / offdiag_total if offdiag_total else 0.0,
                    **_quick_topology(mask),
                })
        selection_note = f"Analyst-selected common threshold: {selected_threshold:g}."

    sensitivity_df = pd.DataFrame(sensitivity)
    selection = {
        "selected_threshold": selected_threshold,
        "selection_method": "manual_review_of_threshold_diagnostics",
        "automatic_threshold_selection": False,
        "candidate_thresholds": list(candidates),
        "analyst_instruction": selection_note,
        "warning": ("Threshold is a sparsification parameter, not an economically observed cutoff. "
                    "Do not treat a candidate or curve elbow as an automatic optimum."),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(out / "threshold_diagnostics.csv", index=False)
        pooled.to_csv(out / "threshold_pooled_summary.csv", index=False)
        if not sensitivity_df.empty:
            sensitivity_df.to_csv(out / "threshold_local_sensitivity.csv", index=False)
        (out / "threshold_selection.json").write_text(json.dumps(selection, indent=2))
        _plot_thresholds(diagnostics, pooled, out)

    return diagnostics, pooled, selection, shares_by_year


def _plot_thresholds(df, pooled, out):
    series = [
        ("offdiag_flow_coverage", "Retained off-diagonal flow coverage", "threshold_flow_coverage.png"),
        ("directed_density", "Directed graph density", "threshold_density.png"),
        ("mean_out_degree", "Mean out-degree", "threshold_mean_out_degree.png"),
        ("mean_in_degree", "Mean in-degree", "threshold_mean_in_degree.png"),
        ("mean_undirected_degree", "Mean undirected-projection degree", "threshold_mean_undirected_degree.png"),
        ("giant_component_fraction", "Giant weak-component fraction", "threshold_giant_component.png"),
        ("reciprocity", "Directed reciprocity", "threshold_reciprocity.png"),
        ("out_degree_cv", "Out-degree coefficient of variation", "threshold_out_degree_cv.png"),
        ("in_degree_cv", "In-degree coefficient of variation", "threshold_in_degree_cv.png"),
        ("undirected_degree_cv", "Undirected-degree coefficient of variation", "threshold_undirected_degree_cv.png"),
    ]
    for col, ylabel, filename in series:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for year, group in df.groupby("year"):
            x = group.tau.replace(0, 1e-12)
            ax.plot(x, group[col], marker="o", label=str(year))
        ax.set_xscale("log")
        ax.set_xlabel("Dependency-share threshold τ")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=.25)
        ax.legend(title="ICIO year")
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for year, group in df.groupby("year"):
        ax.plot(group["directed_density"], group["offdiag_flow_coverage"],
                marker="o", label=str(year))
        for _, row in group.iterrows():
            ax.annotate(f"{row.tau:g}", (row.directed_density, row.offdiag_flow_coverage), fontsize=6)
    ax.set_xlabel("Directed density")
    ax.set_ylabel("Retained off-diagonal flow coverage")
    ax.legend(title="ICIO year")
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "threshold_density_coverage_tradeoff.png", dpi=180)
    plt.close(fig)

"""CLI for the revised WD-only 11-network research workflow.

The previous UU/UD/WD multi-representation pipeline is not the active CLI.
This entry point only reads/writes the revised WD network design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import EXPECTED_NODES
from .revised_structural import (
    ACTIVE_RHO, MATRIX_FORMAT, atomic_json, load_icio_pair,
    run_real_structural, run_synthetic_structural,
    validate_structural_group, validate_structural_all, git_provenance,
)
from .revised_dynamics import (
    ACTIVE_ETA_STAGE1, ACTIVE_ETA_STAGE2,
    run_single_node_census, validate_census,
    validate_dynamics_preflight, structural_dynamic_comparison,
    select_targeted_candidates, run_targeted_individual_sis,
)


def _preflight(data_dir: Path, expected_nodes: int):
    loaded = load_icio_pair(data_dir, expected_nodes)
    rows = []
    for year in (2019, 2022):
        item = loaded[year]
        labels = item["labels"]
        rows.append({
            "network_id": f"real_{year}",
            "year": year,
            "node_count": len(labels),
            "unique_node_ids": len(set(labels)),
            "node_label_order_sha256": __import__("hashlib").sha256("\n".join(labels).encode()).hexdigest(),
            "source_sha256": item["source_sha256"],
            "reordered_to_2019_label_order": item.get("reordered_to_2019_label_order", False),
        })
    if any(row["node_count"] != expected_nodes for row in rows):
        raise ValueError(f"ICIO node count does not equal expected full resolution {expected_nodes}: {rows}")
    if rows[0]["node_label_order_sha256"] != rows[1]["node_label_order_sha256"]:
        raise ValueError("2019 and 2022 node label order hashes differ.")
    return rows


def _write_dynamics_preflight(structural_out: Path, census_out: Path, rho: float):
    records, diagnostics = validate_dynamics_preflight(structural_out, rho)
    target = census_out / "diagnostics"
    target.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(target / "dynamics_preflight.csv", index=False)
    clean_threshold_rows = diagnostics.astype(object).where(pd.notna(diagnostics), None).to_dict(orient="records")
    report = {
        "run_id": "dynamics_preflight_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ"),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "valid",
        "network_count": len(records),
        "network_ids": [r["network_id"] for r in records],
        "matrix_format": MATRIX_FORMAT,
        "rho": float(rho),
        "eta_stage1": list(ACTIVE_ETA_STAGE1),
        "eta_stage2": list(ACTIVE_ETA_STAGE2),
        "spectral_thresholds": clean_threshold_rows,
        "dynamics_started": False,
        "git": git_provenance(),
    }
    atomic_json(target / "dynamics_preflight.json", report)
    return report


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight", "structural", "validate-structural",
            "dynamics-preflight", "census", "census-validate",
            "compare-select", "targeted", "targeted-validate",
        ),
    )
    parser.add_argument("--scope", choices=("real", "synthetic", "all"), default="real")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results/revised_structural_analysis")
    parser.add_argument("--census-out", default="results/single_node_shock_census")
    parser.add_argument("--targeted-out", default="results/targeted_individual_node_sis")
    parser.add_argument("--expected-nodes", type=int, default=EXPECTED_NODES)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--legacy-calibration", default="results_network_analysis/metadata/synthetic_calibration.json")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--rho", type=float, default=ACTIVE_RHO)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--affected-threshold", type=float, default=0.01)
    parser.add_argument("--extinction-tolerance", type=float, default=1e-8)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-8)
    parser.add_argument("--consecutive-steps", type=int, default=5)
    parser.add_argument("--long-run-window", type=int, default=30)
    parser.add_argument("--high-percentile", type=float, default=0.90)
    parser.add_argument("--low-percentile", type=float, default=0.10)
    parser.add_argument("--path-sample-sources", type=int, default=256)
    parser.add_argument("--betweenness-k", type=int, default=64)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    structural_out = Path(args.out)
    census_out = Path(args.census_out)
    targeted_out = Path(args.targeted_out)

    if args.stage == "preflight":
        rows = _preflight(data_dir, args.expected_nodes)
        print(json.dumps({"status": "valid", "node_count": args.expected_nodes,
                          "real_networks": rows, "network_design": {
                              "real": 2, "ER": 3, "BA": 3, "WS": 3, "total": 11,
                              "realizations_per_synthetic_configuration": 1,
                              "matrix_format": MATRIX_FORMAT,
                          }}, indent=2))
        return

    if args.stage == "structural":
        if args.scope == "real":
            if args.threshold is None:
                parser.error("--stage structural --scope real requires --threshold (use the previously reviewed 0.001 unless deliberately changed).")
            records = run_real_structural(
                data_dir, structural_out, args.threshold, args.expected_nodes,
                args.seed, args.path_sample_sources, args.betweenness_k,
                args.legacy_calibration,
            )
        elif args.scope == "synthetic":
            records = run_synthetic_structural(
                structural_out, args.seed, args.path_sample_sources, args.betweenness_k
            )
        else:
            parser.error("Structural generation must be run separately with --scope real, then --scope synthetic.")
        print(json.dumps({"status": "structural_outputs_generated_validation_pending",
                          "scope": args.scope, "network_count": len(records),
                          "network_ids": [r["network_id"] for r in records],
                          "next_step": "Run --stage validate-structural with the same scope."}, indent=2))
        return

    if args.stage == "validate-structural":
        if args.scope == "all":
            report = validate_structural_all(structural_out, args.expected_nodes)
        else:
            report = validate_structural_group(structural_out, args.scope, args.expected_nodes)
        print(json.dumps(report, indent=2))
        return

    if args.stage == "dynamics-preflight":
        report = _write_dynamics_preflight(structural_out, census_out, args.rho)
        print(json.dumps({
            "status": report["status"], "networks_validated": report["network_count"],
            "rho": args.rho, "threshold_table": str(census_out / "diagnostics" / "dynamics_preflight.csv"),
            "dynamics_started": False,
        }, indent=2))
        return

    simulation_options = {
        "rho": args.rho,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "affected_threshold": args.affected_threshold,
        "extinction_tolerance": args.extinction_tolerance,
        "convergence_tolerance": args.convergence_tolerance,
        "consecutive_steps": args.consecutive_steps,
        "long_run_window": args.long_run_window,
    }

    if args.stage == "census":
        report = run_single_node_census(structural_out, census_out, **simulation_options)
        print(json.dumps(report, indent=2))
        return
    if args.stage == "census-validate":
        report = validate_census(census_out, structural_out)
        print(json.dumps(report, indent=2))
        return
    if args.stage == "compare-select":
        comparison = structural_dynamic_comparison(structural_out, census_out)
        candidates = select_targeted_candidates(
            structural_out, census_out, args.high_percentile, args.low_percentile
        )
        print(json.dumps({"comparison": comparison, "candidate_selection": candidates}, indent=2))
        return
    if args.stage == "targeted":
        report = run_targeted_individual_sis(
            structural_out, census_out, targeted_out, **simulation_options
        )
        print(json.dumps(report, indent=2))
        return
    if args.stage == "targeted-validate":
        report_path = targeted_out / "diagnostics" / "targeted_validation.json"
        if not report_path.is_file():
            raise FileNotFoundError("Targeted SIS validation report not found.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("simultaneous_shocks") is not False:
            raise ValueError("Targeted SIS validation failed.")
        result_path = targeted_out / "raw_metrics" / "targeted_individual_node_metrics.csv.gz"
        frame = pd.read_csv(result_path)
        expected = int(report["experiments_expected"])
        duplicate_count = int(frame.duplicated(["network_id", "source_node_id", "eta"]).sum())
        errors = []
        if len(frame) != expected: errors.append(f"expected {expected} records, got {len(frame)}")
        if duplicate_count: errors.append(f"{duplicate_count} duplicate records")
        report["duplicate_records"] = duplicate_count
        report["errors"] = errors
        report["status"] = "complete" if not errors else "incomplete"
        atomic_json(report_path, report)
        if errors: raise ValueError("; ".join(errors))
        print(json.dumps(report, indent=2))
        return


if __name__ == "__main__":
    run()

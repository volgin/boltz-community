"""Aggregate OpenStructure evaluation results for Boltz predictions.

Reads eval JSONs produced by run_boltz_eval.py, computes per-target and
aggregate metrics, and outputs a summary CSV. Optionally compares against
a baseline CSV to show regressions/improvements.

Usage:
    python scripts/eval/aggregate_boltz_eval.py \
        predictions/ evals/ \
        --output results.csv

    # Compare against baseline
    python scripts/eval/aggregate_boltz_eval.py \
        predictions/ evals/ \
        --baseline baseline_results.csv \
        --output results.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_polymer_eval(eval_path: Path) -> dict | None:
    """Load polymer evaluation JSON from OpenStructure."""
    if not eval_path.exists():
        return None
    with eval_path.open() as f:
        data = json.load(f)
    result = {}
    for metric in ["lddt", "bb_lddt", "tm_score", "rmsd"]:
        if metric in data:
            result[metric] = data[metric]
    if "dockq" in data and data["dockq"] is not None:
        valid_dockq = [v for v in data["dockq"] if v is not None]
        if valid_dockq:
            result["dockq_mean"] = np.mean(valid_dockq)
            result["dockq_>0.23"] = np.mean([float(v > 0.23) for v in valid_dockq])
            result["dockq_>0.49"] = np.mean([float(v > 0.49) for v in valid_dockq])
            result["n_interfaces"] = len(valid_dockq)
    return result


def load_ligand_eval(eval_path: Path) -> dict | None:
    """Load ligand evaluation JSON from OpenStructure."""
    if not eval_path.exists():
        return None
    with eval_path.open() as f:
        data = json.load(f)
    result = {}
    if "lddt_pli" in data:
        scores = [x["score"] for x in data["lddt_pli"]["assigned_scores"]]
        n_unassigned = len(data["lddt_pli"].get("model_ligand_unassigned_reason", {}))
        scores.extend([0.0] * n_unassigned)
        if scores:
            result["lddt_pli"] = np.mean(scores)
            result["n_ligands"] = len(scores)
    if "rmsd" in data:
        rmsds = [x["score"] for x in data["rmsd"]["assigned_scores"]]
        n_unassigned = len(data["rmsd"].get("model_ligand_unassigned_reason", {}))
        rmsds.extend([100.0] * n_unassigned)
        if rmsds:
            result["rmsd<2"] = np.mean([float(r < 2.0) for r in rmsds])
            result["rmsd<5"] = np.mean([float(r < 5.0) for r in rmsds])
    return result


def load_confidence(pred_dir: Path, target_name: str, model_idx: int) -> float | None:
    """Load Boltz confidence score for ranking."""
    conf_path = pred_dir / target_name / f"confidence_{target_name}_model_{model_idx}.json"
    if not conf_path.exists():
        # Try lowercase
        conf_path = (
            pred_dir / target_name.lower()
            / f"confidence_{target_name.lower()}_model_{model_idx}.json"
        )
    if not conf_path.exists():
        return None
    with conf_path.open() as f:
        data = json.load(f)
    return data.get("confidence_score")


def aggregate_target(
    target_name: str,
    eval_dir: Path,
    pred_dir: Path,
    num_samples: int,
) -> dict | None:
    """Aggregate metrics for a single target across all samples."""
    metrics_per_sample = {}
    confidences = {}

    for model_idx in range(num_samples):
        eval_name = f"{target_name.lower()}_model_{model_idx}"

        # Polymer metrics
        polymer = load_polymer_eval(eval_dir / f"{eval_name}.json")
        if polymer:
            for k, v in polymer.items():
                metrics_per_sample.setdefault(k, {})[model_idx] = v

        # Ligand metrics
        ligand = load_ligand_eval(eval_dir / f"{eval_name}_ligand.json")
        if ligand:
            for k, v in ligand.items():
                metrics_per_sample.setdefault(k, {})[model_idx] = v

        # Confidence
        conf = load_confidence(pred_dir, target_name, model_idx)
        if conf is not None:
            confidences[model_idx] = conf

    if not metrics_per_sample:
        return None

    # Determine top-1 model (highest confidence)
    top1_idx = max(confidences, key=confidences.get) if confidences else 0

    result = {"target": target_name}
    for metric, values in metrics_per_sample.items():
        if metric.startswith("n_"):
            continue
        all_vals = list(values.values())
        is_lower_better = metric in ("rmsd",)

        result[f"{metric}_oracle"] = min(all_vals) if is_lower_better else max(all_vals)
        result[f"{metric}_top1"] = values.get(top1_idx, np.nan)
        result[f"{metric}_avg"] = np.mean(all_vals)

    return result


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics with bootstrap 95% CI."""
    metric_cols = [c for c in df.columns if c != "target"]
    rows = []
    for col in metric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        mean = values.mean()
        # Bootstrap 95% CI
        n = len(values)
        boot_means = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            sample = values.iloc[rng.integers(0, n, size=n)]
            boot_means.append(sample.mean())
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        rows.append({
            "metric": col,
            "mean": round(mean, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n_targets": len(values),
        })
    return pd.DataFrame(rows)


def compare_with_baseline(current: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare current results against a baseline, showing deltas."""
    merged = current.merge(baseline, on="metric", suffixes=("", "_baseline"))
    merged["delta"] = merged["mean"] - merged["mean_baseline"]
    merged["delta_pct"] = (merged["delta"] / merged["mean_baseline"].abs()) * 100
    merged["delta_pct"] = merged["delta_pct"].round(2)

    # Flag regressions (metrics where lower is worse, except rmsd where lower is better)
    def is_regression(row):
        if "rmsd" in row["metric"] and "<" not in row["metric"]:
            return row["delta"] > 0.01
        return row["delta"] < -0.01

    merged["regression"] = merged.apply(is_regression, axis=1)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Boltz structural evaluation results"
    )
    parser.add_argument("predictions", type=Path,
                        help="Directory with Boltz predictions (for confidence files)")
    parser.add_argument("evals", type=Path,
                        help="Directory with OpenStructure eval JSONs")
    parser.add_argument("--output", type=Path, default=Path("results.csv"),
                        help="Output CSV for per-target results")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of diffusion samples per target")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline summary CSV for comparison")
    args = parser.parse_args()

    # Discover targets from eval directory
    eval_files = list(args.evals.glob("*_model_0.json"))
    target_names = sorted(set(
        f.stem.replace("_model_0", "").replace("_ligand", "")
        for f in eval_files
        if "_ligand" not in f.stem
    ))

    print(f"Found {len(target_names)} evaluated targets")

    # Aggregate per-target metrics
    rows = []
    for target in target_names:
        result = aggregate_target(target, args.evals, args.predictions, args.num_samples)
        if result:
            rows.append(result)

    if not rows:
        print("No results found!")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Per-target results saved to {args.output}")

    # Summary statistics
    summary = compute_summary(df)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSummary ({len(df)} targets):")
    print(summary.to_string(index=False))

    # Compare against baseline
    if args.baseline:
        baseline_summary = pd.read_csv(args.baseline)
        comparison = compare_with_baseline(summary, baseline_summary)
        comp_path = args.output.with_name(args.output.stem + "_comparison.csv")
        comparison.to_csv(comp_path, index=False)

        print(f"\nComparison with baseline:")
        regressions = comparison[comparison["regression"]]
        if regressions.empty:
            print("  No regressions detected!")
        else:
            print("  REGRESSIONS:")
            for _, row in regressions.iterrows():
                print(f"    {row['metric']}: {row['mean_baseline']:.4f} -> {row['mean']:.4f} "
                      f"({row['delta_pct']:+.2f}%)")

        improvements = comparison[
            (~comparison["regression"]) & (comparison["delta"].abs() > 0.01)
        ]
        if not improvements.empty:
            print("  Improvements:")
            for _, row in improvements.iterrows():
                print(f"    {row['metric']}: {row['mean_baseline']:.4f} -> {row['mean']:.4f} "
                      f"({row['delta_pct']:+.2f}%)")


if __name__ == "__main__":
    main()

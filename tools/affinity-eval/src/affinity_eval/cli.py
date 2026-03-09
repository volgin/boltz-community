"""CLI for affinity prediction evaluation.

Usage:
    # Evaluate Boltz predictions against ground truth
    affinity-eval boltz predictions/ --ground-truth labels.csv

    # Evaluate from a generic CSV
    affinity-eval csv predictions.csv --id-col name --pred-col pIC50 --exp-col experimental

    # Compare two models
    affinity-eval compare model_a.csv model_b.csv --names "Boltz-2" "Our Model"
"""

import json
import sys

import click
import numpy as np
import pandas as pd

from affinity_eval.metrics import evaluate
from affinity_eval.parsers import (
    merge_predictions_with_ground_truth,
    parse_boltz_predictions,
    parse_csv_predictions,
)


@click.group()
def main():
    """Model-agnostic binding affinity prediction evaluation."""


@main.command()
@click.argument("predictions_dir", type=click.Path(exists=True))
@click.option("--ground-truth", "-g", required=True, type=click.Path(exists=True),
              help="CSV with ground truth: id, experimental, [target_id], [is_active]")
@click.option("--value-field", default="affinity_pred_value",
              help="Boltz JSON field to use as prediction")
@click.option("--gt-id-col", default="id", help="Ground truth ID column name")
@click.option("--gt-exp-col", default="experimental", help="Ground truth experimental column")
@click.option("--gt-target-col", default=None, help="Ground truth target ID column")
@click.option("--gt-active-col", default=None, help="Ground truth activity label column")
@click.option("--output", "-o", default=None, help="Output JSON file for results")
@click.option("--no-ci", is_flag=True, help="Skip bootstrap CI computation")
def boltz(predictions_dir, ground_truth, value_field, gt_id_col, gt_exp_col,
          gt_target_col, gt_active_col, output, no_ci):
    """Evaluate Boltz affinity predictions."""
    predictions = parse_boltz_predictions(predictions_dir, value_field=value_field)
    gt = pd.read_csv(ground_truth)

    merged = merge_predictions_with_ground_truth(
        predictions, gt,
        gt_id_col=gt_id_col, gt_exp_col=gt_exp_col,
        gt_target_col=gt_target_col, gt_active_col=gt_active_col,
    )

    _run_evaluation(merged, compute_ci=not no_ci, output=output)


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--id-col", default="id", help="Column name for sample identifiers")
@click.option("--pred-col", default="predicted", help="Column name for predicted values")
@click.option("--exp-col", default="experimental", help="Column name for experimental values")
@click.option("--target-col", default=None, help="Column name for target identifiers")
@click.option("--active-col", default=None, help="Column name for activity labels")
@click.option("--output", "-o", default=None, help="Output JSON file for results")
@click.option("--no-ci", is_flag=True, help="Skip bootstrap CI computation")
def csv(csv_path, id_col, pred_col, exp_col, target_col, active_col, output, no_ci):
    """Evaluate predictions from a CSV file."""
    df = parse_csv_predictions(
        csv_path,
        id_col=id_col, pred_col=pred_col, exp_col=exp_col,
        target_col=target_col, active_col=active_col,
    )
    _run_evaluation(df, compute_ci=not no_ci, output=output)


@main.command()
@click.argument("csv_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--names", "-n", multiple=True, help="Model names (one per CSV)")
@click.option("--id-col", default="id")
@click.option("--pred-col", default="predicted")
@click.option("--exp-col", default="experimental")
@click.option("--target-col", default=None)
@click.option("--output", "-o", default=None)
@click.option("--no-ci", is_flag=True)
def compare(csv_paths, names, id_col, pred_col, exp_col, target_col, output, no_ci):
    """Compare predictions from multiple models."""
    if names and len(names) != len(csv_paths):
        click.echo("Error: number of --names must match number of CSV files", err=True)
        sys.exit(1)

    if not names:
        names = [f"Model {i+1}" for i in range(len(csv_paths))]

    results = {}
    for name, path in zip(names, csv_paths):
        click.echo(f"\n{'=' * 60}")
        click.echo(f"  {name}")
        click.echo(f"{'=' * 60}")
        df = parse_csv_predictions(
            path, id_col=id_col, pred_col=pred_col, exp_col=exp_col,
            target_col=target_col,
        )
        result = _run_evaluation(df, compute_ci=not no_ci, output=None)
        results[name] = result.to_dict()

    # Summary comparison table
    click.echo(f"\n{'=' * 60}")
    click.echo("  Comparison Summary")
    click.echo(f"{'=' * 60}")

    metrics_to_compare = ["pearson_r", "spearman_rho", "rmse", "per_target_pearson_r"]
    header = f"{'Metric':<25}" + "".join(f"{n:>15}" for n in names)
    click.echo(header)
    click.echo("-" * len(header))
    for metric in metrics_to_compare:
        values = []
        for name in names:
            v = results[name].get(metric)
            values.append(f"{v:.4f}" if v is not None else "N/A")
        click.echo(f"{metric:<25}" + "".join(f"{v:>15}" for v in values))

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output}")


def _run_evaluation(df, compute_ci=True, output=None):
    """Run evaluation on a merged DataFrame and print results."""
    predicted = df["predicted"].values
    experimental = df["experimental"].values
    target_ids = df["target_id"].values if "target_id" in df.columns else None
    is_active = df["is_active"].values if "is_active" in df.columns else None

    result = evaluate(
        predicted=predicted,
        experimental=experimental,
        target_ids=target_ids,
        is_active=is_active,
        compute_ci=compute_ci,
    )

    click.echo(f"\nResults:")
    click.echo(result.summary())

    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        click.echo(f"\nResults saved to {output}")

    return result


if __name__ == "__main__":
    main()

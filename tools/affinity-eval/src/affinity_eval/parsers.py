"""Parse affinity predictions from various model output formats.

Supports:
- Boltz-2 affinity JSON output
- Generic CSV files
- Custom formats via simple adapter functions
"""

import json
from pathlib import Path

import pandas as pd


def parse_boltz_predictions(
    predictions_dir: str | Path,
    value_field: str = "affinity_pred_value",
) -> pd.DataFrame:
    """Parse Boltz affinity predictions from output directory.

    Boltz-2 writes affinity results as JSON files:
    predictions/{record_id}/affinity_{record_id}.json

    Each JSON contains:
    - affinity_pred_value: log10(IC50) in uM
    - affinity_probability_binary: P(binder) in [0, 1]
    - affinity_pred_value1, affinity_probability_binary1: ensemble model 1
    - affinity_pred_value2, affinity_probability_binary2: ensemble model 2

    Parameters
    ----------
    predictions_dir : path
        Boltz output predictions directory.
    value_field : str
        Which field to use as the predicted value.
        Options: "affinity_pred_value", "affinity_probability_binary",
        "affinity_pred_value1", "affinity_pred_value2".

    Returns
    -------
    DataFrame with columns: id, predicted, and all raw affinity fields.
    """
    predictions_dir = Path(predictions_dir)
    records = []

    for subdir in sorted(predictions_dir.iterdir()):
        if not subdir.is_dir():
            continue
        affinity_file = subdir / f"affinity_{subdir.name}.json"
        if not affinity_file.exists():
            continue
        with affinity_file.open() as f:
            data = json.load(f)
        record = {"id": subdir.name}
        record.update(data)
        record["predicted"] = data.get(value_field)
        records.append(record)

    if not records:
        msg = f"No affinity JSON files found in {predictions_dir}"
        raise FileNotFoundError(msg)

    return pd.DataFrame(records)


def parse_csv_predictions(
    csv_path: str | Path,
    id_col: str = "id",
    pred_col: str = "predicted",
    exp_col: str | None = None,
    target_col: str | None = None,
    active_col: str | None = None,
) -> pd.DataFrame:
    """Parse predictions from a generic CSV file.

    Parameters
    ----------
    csv_path : path
        CSV file path.
    id_col : str
        Column name for sample identifiers.
    pred_col : str
        Column name for predicted values.
    exp_col : str, optional
        Column name for experimental values (if in same file).
    target_col : str, optional
        Column name for target identifiers (for per-target metrics).
    active_col : str, optional
        Column name for activity labels (for enrichment factor).

    Returns
    -------
    DataFrame with standardized column names.
    """
    df = pd.read_csv(csv_path)

    result = pd.DataFrame()
    result["id"] = df[id_col]
    result["predicted"] = df[pred_col].astype(float)

    if exp_col and exp_col in df.columns:
        result["experimental"] = df[exp_col].astype(float)
    if target_col and target_col in df.columns:
        result["target_id"] = df[target_col]
    if active_col and active_col in df.columns:
        result["is_active"] = df[active_col].astype(bool)

    return result


def merge_predictions_with_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    pred_id_col: str = "id",
    gt_id_col: str = "id",
    gt_exp_col: str = "experimental",
    gt_target_col: str | None = "target_id",
    gt_active_col: str | None = None,
) -> pd.DataFrame:
    """Merge prediction DataFrame with ground truth DataFrame.

    Parameters
    ----------
    predictions : DataFrame
        Must have 'id' and 'predicted' columns.
    ground_truth : DataFrame
        Must have id column and experimental value column.

    Returns
    -------
    Merged DataFrame with predicted, experimental, and optional metadata.
    """
    gt = ground_truth.rename(columns={gt_id_col: "id", gt_exp_col: "experimental"})
    if gt_target_col and gt_target_col in ground_truth.columns:
        gt = gt.rename(columns={gt_target_col: "target_id"})
    if gt_active_col and gt_active_col in ground_truth.columns:
        gt = gt.rename(columns={gt_active_col: "is_active"})

    merged = predictions.merge(gt, on="id", how="inner", suffixes=("", "_gt"))

    n_pred = len(predictions)
    n_gt = len(ground_truth)
    n_matched = len(merged)

    if n_matched < n_pred:
        print(f"Warning: {n_pred - n_matched}/{n_pred} predictions had no ground truth match")
    if n_matched < n_gt:
        print(f"Warning: {n_gt - n_matched}/{n_gt} ground truth entries had no prediction")

    return merged

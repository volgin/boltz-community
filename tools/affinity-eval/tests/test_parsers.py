"""Tests for prediction parsers."""

import json
import os
import tempfile

import pandas as pd
import pytest

from affinity_eval.parsers import (
    merge_predictions_with_ground_truth,
    parse_boltz_predictions,
    parse_csv_predictions,
)


class TestParseBoltzPredictions:
    def test_parse_affinity_jsons(self, tmp_path):
        # Create mock Boltz output structure
        for i in range(3):
            record_dir = tmp_path / f"complex_{i}"
            record_dir.mkdir()
            affinity = {
                "affinity_pred_value": -6.5 + i * 0.5,
                "affinity_probability_binary": 0.8 - i * 0.1,
                "affinity_pred_value1": -6.4 + i * 0.5,
                "affinity_probability_binary1": 0.75 - i * 0.1,
                "affinity_pred_value2": -6.6 + i * 0.5,
                "affinity_probability_binary2": 0.85 - i * 0.1,
            }
            with (record_dir / f"affinity_complex_{i}.json").open("w") as f:
                json.dump(affinity, f)

        df = parse_boltz_predictions(tmp_path)
        assert len(df) == 3
        assert "predicted" in df.columns
        assert "id" in df.columns
        assert df["predicted"].iloc[0] == pytest.approx(-6.5)

    def test_custom_value_field(self, tmp_path):
        record_dir = tmp_path / "test"
        record_dir.mkdir()
        with (record_dir / "affinity_test.json").open("w") as f:
            json.dump({
                "affinity_pred_value": -6.5,
                "affinity_probability_binary": 0.8,
            }, f)

        df = parse_boltz_predictions(tmp_path, value_field="affinity_probability_binary")
        assert df["predicted"].iloc[0] == pytest.approx(0.8)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_boltz_predictions(tmp_path)


class TestParseCSVPredictions:
    def test_basic_csv(self, tmp_path):
        csv_path = tmp_path / "predictions.csv"
        pd.DataFrame({
            "id": ["a", "b", "c"],
            "predicted": [1.0, 2.0, 3.0],
            "experimental": [1.1, 2.1, 3.1],
        }).to_csv(csv_path, index=False)

        df = parse_csv_predictions(csv_path, exp_col="experimental")
        assert len(df) == 3
        assert "experimental" in df.columns

    def test_custom_columns(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({
            "compound": ["x", "y"],
            "pIC50": [6.5, 7.2],
            "exp_pIC50": [6.3, 7.5],
            "protein": ["CDK2", "TYK2"],
        }).to_csv(csv_path, index=False)

        df = parse_csv_predictions(
            csv_path, id_col="compound", pred_col="pIC50",
            exp_col="exp_pIC50", target_col="protein",
        )
        assert len(df) == 2
        assert df["target_id"].tolist() == ["CDK2", "TYK2"]


class TestMergePredictions:
    def test_perfect_merge(self):
        predictions = pd.DataFrame({
            "id": ["a", "b", "c"],
            "predicted": [1.0, 2.0, 3.0],
        })
        gt = pd.DataFrame({
            "id": ["a", "b", "c"],
            "experimental": [1.1, 2.1, 3.1],
        })
        merged = merge_predictions_with_ground_truth(predictions, gt)
        assert len(merged) == 3
        assert "predicted" in merged.columns
        assert "experimental" in merged.columns

    def test_partial_merge(self):
        predictions = pd.DataFrame({
            "id": ["a", "b", "c"],
            "predicted": [1.0, 2.0, 3.0],
        })
        gt = pd.DataFrame({
            "id": ["a", "c"],
            "experimental": [1.1, 3.1],
        })
        merged = merge_predictions_with_ground_truth(predictions, gt)
        assert len(merged) == 2

"""Tests for affinity evaluation metrics."""

import numpy as np
import pytest

from affinity_eval.metrics import (
    bootstrap_ci,
    enrichment_factor,
    evaluate,
    mae,
    pearson_r,
    per_target_pearson,
    rmse,
    spearman_rho,
)


class TestPearsonR:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, p = pearson_r(x, x)
        assert r == pytest.approx(1.0)
        assert p < 0.01

    def test_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = pearson_r(x, -x)
        assert r == pytest.approx(-1.0)

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        r, _ = pearson_r(x, y)
        assert abs(r) < 0.1

    def test_too_few_samples(self):
        r, p = pearson_r(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert np.isnan(r)


class TestSpearmanRho:
    def test_perfect_monotonic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rho, p = spearman_rho(x, y)
        assert rho == pytest.approx(1.0)

    def test_nonlinear_monotonic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # y = x^2
        rho, _ = spearman_rho(x, y)
        assert rho == pytest.approx(1.0)


class TestRMSE:
    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert rmse(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        exp = np.array([2.0, 3.0, 4.0])
        assert rmse(pred, exp) == pytest.approx(1.0)


class TestMAE:
    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mae(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        exp = np.array([2.0, 4.0, 6.0])
        assert mae(pred, exp) == pytest.approx(2.0)


class TestEnrichmentFactor:
    def test_perfect_enrichment(self):
        # All actives are top-scored
        predicted = np.array([10, 9, 8, 7, 1, 2, 3, 4, 5, 6], dtype=float)
        is_active = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        ef = enrichment_factor(predicted, is_active, fraction=0.1)
        # Top 10% = 1 sample, which is active. 4/10 are active.
        # Expected: 1 / (4 * 0.1) = 2.5
        assert ef == pytest.approx(2.5)

    def test_random_enrichment(self):
        rng = np.random.default_rng(42)
        n = 10000
        predicted = rng.standard_normal(n)
        is_active = rng.random(n) < 0.1
        ef = enrichment_factor(predicted, is_active, fraction=0.05)
        # Should be close to 1.0 for random
        assert 0.5 < ef < 2.0

    def test_no_actives(self):
        predicted = np.array([1.0, 2.0, 3.0])
        is_active = np.array([False, False, False])
        assert enrichment_factor(predicted, is_active, 0.5) == 0.0


class TestPerTargetPearson:
    def test_single_target(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exp = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        targets = np.array(["A", "A", "A", "A", "A"])
        mean_r, n = per_target_pearson(pred, exp, targets)
        assert n == 1
        assert mean_r > 0.99

    def test_two_targets(self):
        pred = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
        exp = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
        targets = np.array(["A"] * 5 + ["B"] * 5)
        mean_r, n = per_target_pearson(pred, exp, targets)
        assert n == 2
        assert mean_r == pytest.approx(1.0)

    def test_min_compounds_filter(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Target B has only 2 compounds, below min_compounds=3
        targets = np.array(["A", "A", "A", "B", "B"])
        mean_r, n = per_target_pearson(pred, exp, targets, min_compounds=3)
        assert n == 1  # Only target A included


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        pred = rng.standard_normal(100)
        exp = pred + rng.standard_normal(100) * 0.1
        r, _ = pearson_r(pred, exp)
        ci_low, ci_high = bootstrap_ci(
            pred, exp, lambda p, e: pearson_r(p, e)[0], n_bootstrap=500,
        )
        assert ci_low <= r <= ci_high

    def test_ci_width_decreases_with_more_data(self):
        rng = np.random.default_rng(42)
        widths = []
        for n in [20, 100, 500]:
            pred = rng.standard_normal(n)
            exp = pred + rng.standard_normal(n) * 0.5
            ci_low, ci_high = bootstrap_ci(
                pred, exp, lambda p, e: pearson_r(p, e)[0], n_bootstrap=200,
            )
            widths.append(ci_high - ci_low)
        # CI should get narrower with more data
        assert widths[0] > widths[1] > widths[2]


class TestEvaluate:
    def test_basic_evaluation(self):
        pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        exp = np.array([1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.1, 8.2, 8.8, 10.1])
        result = evaluate(pred, exp, compute_ci=False)
        assert result.pearson_r > 0.99
        assert result.spearman_rho > 0.99
        assert result.rmse < 0.5
        assert result.n_samples == 10

    def test_with_targets(self):
        pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        targets = np.array(["A"] * 5 + ["B"] * 5)
        result = evaluate(pred, exp, target_ids=targets, compute_ci=False)
        assert result.per_target_pearson_r == pytest.approx(1.0)
        assert result.per_target_pearson_n_targets == 2

    def test_with_activity_labels(self):
        pred = np.array([10, 9, 8, 7, 1, 2, 3, 4, 5, 6], dtype=float)
        exp = np.array([10, 9, 8, 7, 1, 2, 3, 4, 5, 6], dtype=float)
        is_active = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = evaluate(pred, exp, is_active=is_active, compute_ci=False)
        assert result.enrichment_factor_1pct is not None
        assert result.enrichment_factor_5pct is not None

    def test_nan_handling(self):
        pred = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        exp = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = evaluate(pred, exp, compute_ci=False)
        assert result.n_samples == 3  # Two NaN pairs removed

    def test_summary_format(self):
        pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        exp = pred + 0.1
        result = evaluate(pred, exp, compute_ci=False)
        summary = result.summary()
        assert "Pearson R:" in summary
        assert "RMSE:" in summary

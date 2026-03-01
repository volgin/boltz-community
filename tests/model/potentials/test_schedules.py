"""Tests for boltz.model.potentials.schedules — ExponentialInterpolation, PiecewiseStepFunction."""

import math

import pytest

from boltz.model.potentials.schedules import ExponentialInterpolation, PiecewiseStepFunction


class TestExponentialInterpolation:
    """Tests for ExponentialInterpolation."""

    def test_endpoints(self):
        """Schedule at t=0 → start, t=1 → end."""
        sched = ExponentialInterpolation(start=1.0, end=5.0, alpha=-2.0)
        assert sched.compute(0.0) == pytest.approx(1.0, abs=1e-6)
        assert sched.compute(1.0) == pytest.approx(5.0, abs=1e-6)

    def test_linear_case(self):
        """alpha=0 → linear interpolation."""
        sched = ExponentialInterpolation(start=0.0, end=10.0, alpha=0.0)
        assert sched.compute(0.0) == pytest.approx(0.0, abs=1e-6)
        assert sched.compute(0.5) == pytest.approx(5.0, abs=1e-6)
        assert sched.compute(1.0) == pytest.approx(10.0, abs=1e-6)

    def test_monotonicity(self):
        """Schedule is monotonic between endpoints."""
        sched = ExponentialInterpolation(start=2.0, end=8.0, alpha=3.0)
        values = [sched.compute(t / 10.0) for t in range(11)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-10

    def test_negative_alpha(self):
        """Negative alpha → schedule curves differently than positive."""
        sched_pos = ExponentialInterpolation(start=0.0, end=10.0, alpha=2.0)
        sched_neg = ExponentialInterpolation(start=0.0, end=10.0, alpha=-2.0)

        # At t=0.5, they should differ (same endpoints but different curves)
        mid_pos = sched_pos.compute(0.5)
        mid_neg = sched_neg.compute(0.5)
        assert mid_pos != pytest.approx(mid_neg, abs=0.1)


class TestPiecewiseStepFunction:
    """Tests for PiecewiseStepFunction."""

    def test_step_regions(self):
        """Values in each region are correct."""
        sched = PiecewiseStepFunction(thresholds=[0.3, 0.7], values=[1.0, 2.0, 3.0])

        assert sched.compute(0.0) == 1.0
        assert sched.compute(0.2) == 1.0
        assert sched.compute(0.5) == 2.0
        assert sched.compute(0.8) == 3.0
        assert sched.compute(1.0) == 3.0

    def test_at_threshold(self):
        """At threshold boundary, value transitions to next region."""
        sched = PiecewiseStepFunction(thresholds=[0.5], values=[10.0, 20.0])

        assert sched.compute(0.4) == 10.0
        assert sched.compute(0.5) == 10.0  # t > threshold, not >=
        assert sched.compute(0.6) == 20.0

    def test_single_threshold(self):
        """Single threshold splits into two regions."""
        sched = PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])

        assert sched.compute(0.0) == 0.125
        assert sched.compute(0.3) == 0.125
        assert sched.compute(0.5) == 0.0
        assert sched.compute(1.0) == 0.0

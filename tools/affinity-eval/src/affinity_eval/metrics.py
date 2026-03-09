"""Core metrics for binding affinity prediction evaluation.

All metrics operate on paired arrays of predicted and experimental values.
Supports per-target evaluation (Pearson R averaged across targets) as used
in the Boltz-2 paper, as well as global (pooled) evaluation.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class EvalResult:
    """Result of an affinity evaluation."""

    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    rmse: float
    mae: float
    n_samples: int
    # Per-target averages (if target grouping provided)
    per_target_pearson_r: float | None = None
    per_target_pearson_n_targets: int | None = None
    # Enrichment
    enrichment_factor_1pct: float | None = None
    enrichment_factor_5pct: float | None = None
    # Bootstrap CIs
    pearson_r_ci: tuple[float, float] | None = None
    per_target_pearson_r_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"  n_samples:    {self.n_samples}",
            f"  Pearson R:    {self.pearson_r:.4f} (p={self.pearson_p:.2e})"
            + (f"  95% CI: [{self.pearson_r_ci[0]:.4f}, {self.pearson_r_ci[1]:.4f}]"
               if self.pearson_r_ci else ""),
            f"  Spearman rho: {self.spearman_rho:.4f} (p={self.spearman_p:.2e})",
            f"  RMSE:         {self.rmse:.4f}",
            f"  MAE:          {self.mae:.4f}",
        ]
        if self.per_target_pearson_r is not None:
            line = (
                f"  Per-target R: {self.per_target_pearson_r:.4f} "
                f"(n={self.per_target_pearson_n_targets} targets)"
            )
            if self.per_target_pearson_r_ci:
                line += (
                    f"  95% CI: [{self.per_target_pearson_r_ci[0]:.4f}, "
                    f"{self.per_target_pearson_r_ci[1]:.4f}]"
                )
            lines.append(line)
        if self.enrichment_factor_1pct is not None:
            lines.append(f"  EF@1%:        {self.enrichment_factor_1pct:.2f}")
        if self.enrichment_factor_5pct is not None:
            lines.append(f"  EF@5%:        {self.enrichment_factor_5pct:.2f}")
        return "\n".join(lines)


def pearson_r(predicted: np.ndarray, experimental: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    if len(predicted) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(predicted, experimental)
    return float(r), float(p)


def spearman_rho(predicted: np.ndarray, experimental: np.ndarray) -> tuple[float, float]:
    """Compute Spearman rank correlation and p-value."""
    if len(predicted) < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(predicted, experimental)
    return float(rho), float(p)


def rmse(predicted: np.ndarray, experimental: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((predicted - experimental) ** 2)))


def mae(predicted: np.ndarray, experimental: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(predicted - experimental)))


def enrichment_factor(
    predicted: np.ndarray,
    is_active: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Compute enrichment factor at a given fraction.

    Measures how many true actives are found in the top-scoring fraction
    of predictions, relative to random selection.

    Parameters
    ----------
    predicted : array
        Predicted scores (higher = more likely active).
    is_active : array
        Boolean array indicating true actives.
    fraction : float
        Top fraction to consider (e.g. 0.01 for top 1%).

    Returns
    -------
    float
        Enrichment factor. EF=1.0 means no better than random.
    """
    n = len(predicted)
    n_top = max(1, int(n * fraction))
    n_actives = is_active.sum()
    if n_actives == 0 or n == 0:
        return 0.0

    # Sort by predicted score (descending)
    top_indices = np.argsort(predicted)[::-1][:n_top]
    actives_in_top = is_active[top_indices].sum()

    expected = n_actives * fraction
    return float(actives_in_top / expected) if expected > 0 else 0.0


def per_target_pearson(
    predicted: np.ndarray,
    experimental: np.ndarray,
    target_ids: np.ndarray,
    min_compounds: int = 3,
) -> tuple[float, int]:
    """Compute Pearson R averaged across targets.

    This is the primary metric used in the Boltz-2 paper: compute Pearson R
    for each target independently, then average across targets.

    Parameters
    ----------
    predicted, experimental : arrays
        Predicted and experimental values.
    target_ids : array
        Target identifier for each sample (used for grouping).
    min_compounds : int
        Minimum compounds per target to include (default 3).

    Returns
    -------
    mean_r : float
        Mean Pearson R across targets.
    n_targets : int
        Number of targets with enough compounds.
    """
    unique_targets = np.unique(target_ids)
    rs = []
    for target in unique_targets:
        mask = target_ids == target
        if mask.sum() < min_compounds:
            continue
        r, _ = pearson_r(predicted[mask], experimental[mask])
        if not np.isnan(r):
            rs.append(r)
    if not rs:
        return np.nan, 0
    return float(np.mean(rs)), len(rs)


def bootstrap_ci(
    predicted: np.ndarray,
    experimental: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    target_ids: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    predicted, experimental : arrays
        Predicted and experimental values.
    metric_fn : callable
        Function(predicted, experimental) -> float.
        For per-target metrics, should accept target_ids kwarg.
    n_bootstrap : int
        Number of bootstrap iterations.
    alpha : float
        Significance level (0.05 for 95% CI).
    seed : int
        Random seed for reproducibility.
    target_ids : array, optional
        If provided, bootstrap resamples targets (not individual samples).

    Returns
    -------
    ci_lower, ci_upper : float
        Lower and upper bounds of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    boot_values = []

    if target_ids is not None:
        # Bootstrap over targets (preserving within-target structure)
        unique_targets = np.unique(target_ids)
        n_targets = len(unique_targets)
        for _ in range(n_bootstrap):
            sampled_targets = rng.choice(unique_targets, size=n_targets, replace=True)
            indices = np.concatenate([
                np.where(target_ids == t)[0] for t in sampled_targets
            ])
            val = metric_fn(predicted[indices], experimental[indices])
            if not np.isnan(val):
                boot_values.append(val)
    else:
        n = len(predicted)
        for _ in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            val = metric_fn(predicted[indices], experimental[indices])
            if not np.isnan(val):
                boot_values.append(val)

    if not boot_values:
        return np.nan, np.nan

    return (
        float(np.percentile(boot_values, 100 * alpha / 2)),
        float(np.percentile(boot_values, 100 * (1 - alpha / 2))),
    )


def evaluate(
    predicted: np.ndarray,
    experimental: np.ndarray,
    target_ids: np.ndarray | None = None,
    is_active: np.ndarray | None = None,
    compute_ci: bool = True,
    min_compounds_per_target: int = 3,
) -> EvalResult:
    """Run full evaluation suite.

    Parameters
    ----------
    predicted : array of float
        Predicted affinity values.
    experimental : array of float
        Experimental affinity values.
    target_ids : array of str, optional
        Target identifiers for per-target averaging.
    is_active : array of bool, optional
        Activity labels for enrichment factor computation.
    compute_ci : bool
        Whether to compute bootstrap 95% CIs (slower).
    min_compounds_per_target : int
        Minimum compounds per target for per-target Pearson R.

    Returns
    -------
    EvalResult
        Comprehensive evaluation results.
    """
    predicted = np.asarray(predicted, dtype=float)
    experimental = np.asarray(experimental, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(predicted) | np.isnan(experimental))
    predicted = predicted[valid]
    experimental = experimental[valid]
    if target_ids is not None:
        target_ids = np.asarray(target_ids)[valid]
    if is_active is not None:
        is_active = np.asarray(is_active, dtype=bool)[valid]

    # Global metrics
    r, r_p = pearson_r(predicted, experimental)
    rho, rho_p = spearman_rho(predicted, experimental)

    result = EvalResult(
        pearson_r=r,
        pearson_p=r_p,
        spearman_rho=rho,
        spearman_p=rho_p,
        rmse=rmse(predicted, experimental),
        mae=mae(predicted, experimental),
        n_samples=len(predicted),
    )

    # Per-target metrics
    if target_ids is not None:
        mean_r, n_targets = per_target_pearson(
            predicted, experimental, target_ids, min_compounds_per_target,
        )
        result.per_target_pearson_r = mean_r
        result.per_target_pearson_n_targets = n_targets

    # Enrichment factors
    if is_active is not None:
        result.enrichment_factor_1pct = enrichment_factor(predicted, is_active, 0.01)
        result.enrichment_factor_5pct = enrichment_factor(predicted, is_active, 0.05)

    # Bootstrap CIs
    if compute_ci and len(predicted) >= 10:
        result.pearson_r_ci = bootstrap_ci(
            predicted, experimental,
            lambda p, e: pearson_r(p, e)[0],
        )
        if target_ids is not None and result.per_target_pearson_n_targets:
            result.per_target_pearson_r_ci = bootstrap_ci(
                predicted, experimental,
                lambda p, e: pearson_r(p, e)[0],
                target_ids=target_ids,
            )

    return result

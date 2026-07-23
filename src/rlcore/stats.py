"""Aggregate statistics for multi-seed algorithm comparison.

This module is the platform's *statistics adapter* (ROADMAP M7): the
protocol follows Agarwal et al. (NeurIPS 2021), "Deep Reinforcement
Learning at the Edge of the Statistical Precipice", but the implementation
is native NumPy rather than a dependency on the archived ``rliable``
repository. Every estimator is exercised by hand-computed unit tests, so a
future swap to another backend has a fixed, tested contract to satisfy.

Conventions: a "score matrix" is ``[n_seeds]`` (one task) of final scores;
cross-task extensions are deliberately deferred until the platform runs
multi-task suites. All resampling randomness flows through an explicit
NumPy generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def interquartile_mean(scores: np.ndarray) -> float:
    """Mean of the middle 50% of scores (IQM).

    Drops the bottom and top 25% (by count, using the ceil/floor split that
    keeps at least half the data) and averages the rest — more robust than
    the mean, more statistically efficient than the median.

    Raises:
        ValueError: If ``scores`` is empty or not 1-D.
    """
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(f"scores must be a non-empty 1-D array, got shape {scores.shape}.")
    ordered = np.sort(scores)
    quarter = scores.size // 4
    trimmed = ordered[quarter : scores.size - quarter]
    return float(trimmed.mean())


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A point estimate with a percentile-bootstrap confidence interval.

    Attributes:
        point: The statistic on the full sample.
        low: Lower CI bound.
        high: Upper CI bound.
        confidence: Nominal coverage (e.g. 0.95).
    """

    point: float
    low: float
    high: float
    confidence: float


def bootstrap_interval(
    scores: np.ndarray,
    *,
    statistic: str = "iqm",
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    rng: np.random.Generator,
) -> BootstrapInterval:
    """Percentile-bootstrap CI for a robust statistic over seeds.

    Small-sample caveat, stated plainly: with the platform's typical 3-5
    seeds, bootstrap intervals are rough and tend toward under-coverage;
    they are reported as honest uncertainty indicators, not precision
    guarantees.

    Args:
        scores: ``[n_seeds]`` final scores.
        statistic: ``"iqm"``, ``"mean"``, or ``"median"``.
        confidence: Nominal coverage in (0, 1).
        n_resamples: Bootstrap resamples.
        rng: Explicit NumPy generator (determinism is part of the contract).

    Raises:
        ValueError: On unknown statistic or invalid confidence.
    """
    estimators: dict[str, Callable[[np.ndarray], float]] = {
        "iqm": interquartile_mean,
        "mean": lambda s: float(np.mean(s)),
        "median": lambda s: float(np.median(s)),
    }
    if statistic not in estimators:
        raise ValueError(f"Unknown statistic {statistic!r}; expected one of {sorted(estimators)}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    estimator = estimators[statistic]
    point = estimator(scores)
    resampled = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(scores, size=scores.size, replace=True)
        resampled[i] = estimator(sample)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled, [alpha, 1.0 - alpha])
    return BootstrapInterval(point=point, low=float(low), high=float(high), confidence=confidence)


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray) -> float:
    """P(X > Y) over all seed pairs, ties counted half (Mann-Whitney core).

    0.5 means no evidence either way; values near 1 mean X's runs beat Y's
    runs across seed pairings. A descriptive statistic here — no
    significance claim is attached at these sample sizes.

    Raises:
        ValueError: If either sample is empty.
    """
    if scores_x.size == 0 or scores_y.size == 0:
        raise ValueError("Both samples must be non-empty.")
    x = scores_x.reshape(-1, 1)
    y = scores_y.reshape(1, -1)
    wins = (x > y).sum() + 0.5 * (x == y).sum()
    return float(wins / (scores_x.size * scores_y.size))


def performance_profile(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Fraction of runs whose score is at least each threshold.

    The (one-task) run-score distribution of Agarwal et al.: monotonically
    non-increasing in the threshold; plotting several algorithms' profiles
    on shared thresholds shows *where* in the score range one dominates.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if scores.size == 0:
        raise ValueError("scores must be non-empty.")
    profile = (scores.reshape(-1, 1) >= thresholds.reshape(1, -1)).mean(axis=0)
    return np.asarray(profile)


def summarize(scores: np.ndarray, *, rng: np.random.Generator) -> dict[str, float]:
    """Standard summary block: mean, median, IQM with bootstrap CI, min/max."""
    interval = bootstrap_interval(scores, statistic="iqm", rng=rng)
    return {
        "n": float(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "iqm": interval.point,
        "iqm_ci_low": interval.low,
        "iqm_ci_high": interval.high,
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }

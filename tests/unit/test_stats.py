"""Hand-computed tests for the statistics adapter."""

from __future__ import annotations

import numpy as np
import pytest

from rlcore.stats import (
    bootstrap_interval,
    interquartile_mean,
    performance_profile,
    probability_of_improvement,
    summarize,
)


class TestIqm:
    def test_hand_computed_eight_values(self) -> None:
        # Sorted: 0..7; drop bottom 2 and top 2 -> mean(2,3,4,5) = 3.5
        scores = np.array([7.0, 0.0, 3.0, 5.0, 1.0, 6.0, 2.0, 4.0])
        assert interquartile_mean(scores) == pytest.approx(3.5)

    def test_small_samples_keep_middle(self) -> None:
        # n=5: quarter=1 -> mean of middle three.
        assert interquartile_mean(np.array([10.0, 1.0, 2.0, 3.0, -10.0])) == pytest.approx(2.0)
        # n<4: quarter=0 -> plain mean.
        assert interquartile_mean(np.array([1.0, 2.0])) == pytest.approx(1.5)

    def test_robust_to_one_outlier(self) -> None:
        clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        spiked = clean.copy()
        spiked[7] = 1e9
        assert interquartile_mean(spiked) == interquartile_mean(clean)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            interquartile_mean(np.array([]))


class TestBootstrap:
    def test_deterministic_given_rng(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = bootstrap_interval(scores, rng=np.random.default_rng(7))
        b = bootstrap_interval(scores, rng=np.random.default_rng(7))
        assert (a.low, a.high) == (b.low, b.high)

    def test_interval_brackets_point_and_orders(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = bootstrap_interval(scores, rng=np.random.default_rng(0))
        assert ci.low <= ci.point <= ci.high

    def test_degenerate_sample_collapses(self) -> None:
        ci = bootstrap_interval(np.full(5, 2.0), rng=np.random.default_rng(0))
        assert ci.low == ci.point == ci.high == 2.0

    def test_statistics_and_validation(self) -> None:
        scores = np.array([1.0, 2.0, 3.0])
        mean_ci = bootstrap_interval(scores, statistic="mean", rng=np.random.default_rng(0))
        assert mean_ci.point == pytest.approx(2.0)
        with pytest.raises(ValueError, match="Unknown statistic"):
            bootstrap_interval(scores, statistic="mode", rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_interval(scores, confidence=1.5, rng=np.random.default_rng(0))


class TestProbabilityOfImprovement:
    def test_hand_computed(self) -> None:
        x = np.array([3.0, 5.0])
        y = np.array([2.0, 4.0])
        # pairs: (3>2)=1, (3>4)=0, (5>2)=1, (5>4)=1 -> 3/4
        assert probability_of_improvement(x, y) == pytest.approx(0.75)

    def test_ties_count_half(self) -> None:
        assert probability_of_improvement(np.array([1.0]), np.array([1.0])) == 0.5

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=6), rng.normal(size=5)
        assert probability_of_improvement(x, y) + probability_of_improvement(y, x) == pytest.approx(
            1.0
        )


class TestPerformanceProfile:
    def test_hand_computed_and_monotone(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        thresholds = np.array([0.0, 2.5, 5.0])
        profile = performance_profile(scores, thresholds)
        assert profile.tolist() == [1.0, 0.5, 0.0]
        assert (np.diff(profile) <= 0).all()


class TestSummarize:
    def test_keys_and_values(self) -> None:
        summary = summarize(np.array([1.0, 2.0, 3.0]), rng=np.random.default_rng(0))
        assert summary["n"] == 3
        assert summary["mean"] == pytest.approx(2.0)
        assert summary["median"] == pytest.approx(2.0)
        assert summary["min"] == 1.0 and summary["max"] == 3.0
        assert summary["iqm_ci_low"] <= summary["iqm"] <= summary["iqm_ci_high"]

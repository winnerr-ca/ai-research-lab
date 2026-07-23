"""Tutorial 3: compare two configurations the platform's way.

Run it:

    uv run python examples/03_compare_with_statistics.py
    uv run python examples/03_compare_with_statistics.py --updates 2 --seeds 2

The rules this tutorial demonstrates:

1. Equal, fixed budgets (no early stopping) for anything compared.
2. Multiple seeds per configuration — single-seed comparisons are noise.
3. Robust statistics with uncertainty (IQM + bootstrap CI), and
   probability of improvement instead of a bare "A beats B".
4. Small-sample honesty: with a handful of seeds the output is an
   *indication*, never a significance claim — say so in your reports.
"""

from __future__ import annotations

import argparse

import numpy as np

from rlcore.agents.ppo.train import PpoConfig, train
from rlcore.stats import bootstrap_interval, probability_of_improvement


def final_scores(updates: int, seeds: int, *, normalize: bool) -> np.ndarray:
    """Final deterministic eval means for one configuration across seeds."""
    scores = []
    for seed in range(seeds):
        result = train(
            PpoConfig(
                seed=seed,
                total_updates=updates,
                normalize_advantages=normalize,
                eval_every=0,
                stop_return=None,
            )
        )
        scores.append(result.final_eval.mean_return)
    return np.asarray(scores)


def main() -> None:
    """Compare advantage normalization on/off at an equal tiny budget."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)  # explicit RNG: bootstrap is reproducible
    on = final_scores(args.updates, args.seeds, normalize=True)
    off = final_scores(args.updates, args.seeds, normalize=False)

    for label, scores in (("normalize on ", on), ("normalize off", off)):
        interval = bootstrap_interval(scores, statistic="iqm", rng=rng)
        print(
            f"{label}: scores {np.round(scores, 1).tolist()} | "
            f"IQM {interval.point:.1f} "
            f"[{interval.low:.1f}, {interval.high:.1f}] (95% bootstrap)"
        )
    p = probability_of_improvement(on, off)
    print(f"P(normalize-on run > normalize-off run) = {p:.2f}")
    print(
        f"note: n={args.seeds} seeds at a tiny budget — this demonstrates the "
        "protocol, and is deliberately NOT evidence about normalization."
    )


if __name__ == "__main__":
    main()

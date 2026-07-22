"""Slow stochastic canary: REINFORCE learns CartPole at all.

Excluded from the default (fast, deterministic) test run via the ``slow``
marker; run with ``pytest -m slow``. The threshold is deliberately generous —
this is a learns-at-all check, not the multi-seed validation experiment
(see ``benchmarks/m1_reinforce_cartpole.py`` for that).
"""

from __future__ import annotations

import pytest

from rlcore.agents.reinforce.train import ReinforceConfig, train


@pytest.mark.slow
def test_reinforce_learns_cartpole() -> None:
    config = ReinforceConfig(
        seed=1,
        updates=200,
        episodes_per_update=10,
        eval_every=10,
        eval_episodes=10,
        stop_return=195.0,
    )
    result = train(config)
    assert result.final_eval.mean_return >= 195.0, (
        f"final greedy eval {result.final_eval.mean_return:.1f} < 195 after "
        f"{result.total_episodes} episodes ({result.total_env_steps} steps)"
    )

"""Slow stochastic canary: PPO learns CartPole at all.

Excluded from the default (fast, deterministic) run via the ``slow``
marker; the multi-seed validation experiment lives in
``benchmarks/m2_ppo_cartpole.py``.
"""

from __future__ import annotations

import pytest

from rlcore.agents.ppo.train import PpoConfig, train


@pytest.mark.slow
def test_ppo_learns_cartpole() -> None:
    config = PpoConfig(
        seed=1,
        total_updates=25,
        eval_every=5,
        eval_episodes=10,
        stop_return=195.0,
    )
    result = train(config)
    assert result.final_eval.mean_return >= 195.0, (
        f"final greedy eval {result.final_eval.mean_return:.1f} < 195 after "
        f"{result.total_env_steps} env steps"
    )

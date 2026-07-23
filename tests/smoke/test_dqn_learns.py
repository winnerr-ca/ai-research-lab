"""Slow stochastic canary: DQN learns CartPole at all."""

from __future__ import annotations

import pytest

from rlcore.agents.dqn.train import DqnConfig, train


@pytest.mark.slow
def test_dqn_learns_cartpole() -> None:
    config = DqnConfig(
        seed=1,
        total_steps=30_000,
        eval_every=2_500,
        eval_episodes=10,
        stop_return=195.0,
    )
    result = train(config)
    assert result.final_eval.mean_return >= 195.0, (
        f"final greedy eval {result.final_eval.mean_return:.1f} < 195 after "
        f"{result.total_env_steps} env steps"
    )

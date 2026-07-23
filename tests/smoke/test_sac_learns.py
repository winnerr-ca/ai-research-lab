"""Slow stochastic canary: SAC learns Pendulum at all."""

from __future__ import annotations

import pytest

from rlcore.agents.sac.train import SacConfig, train


@pytest.mark.slow
def test_sac_learns_pendulum() -> None:
    config = SacConfig(
        seed=1,
        total_steps=15_000,
        eval_every=2_000,
        eval_episodes=5,
        stop_return=-300.0,
    )
    result = train(config)
    assert result.final_eval.mean_return >= -300.0, (
        f"final deterministic eval {result.final_eval.mean_return:.1f} < -300 after "
        f"{result.total_env_steps} env steps"
    )

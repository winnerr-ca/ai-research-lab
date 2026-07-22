"""Tests for the REINFORCE update step and seed-controlled training."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.reinforce.returns import discounted_returns
from rlcore.agents.reinforce.rollout import collect_episode
from rlcore.agents.reinforce.train import ReinforceConfig, reinforce_update, train
from rlcore.testing import ScriptedEnv

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.types import Batch

GAMMA = 0.9


def make_policy(seed: int = 0) -> CategoricalMlpPolicy:
    torch.manual_seed(seed)
    return CategoricalMlpPolicy(obs_dim=2, n_actions=2, hidden_sizes=(8,))


def make_episodes(policy: CategoricalMlpPolicy) -> list[Batch]:
    generator = torch.Generator().manual_seed(0)
    return [
        collect_episode(ScriptedEnv([1.0, 2.0, 3.0]), policy, generator=generator),
        collect_episode(ScriptedEnv([1.0, 1.0], terminate=False), policy, generator=generator),
    ]


def tiny_config(seed: int) -> ReinforceConfig:
    return ReinforceConfig(
        seed=seed,
        updates=2,
        episodes_per_update=2,
        eval_every=0,
        eval_episodes=2,
        stop_return=None,
    )


class TestReinforceUpdate:
    def test_parameters_change(self) -> None:
        policy = make_policy()
        before = [p.detach().clone() for p in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        reinforce_update(
            policy, optimizer, make_episodes(policy), gamma=GAMMA, normalize_returns=True
        )
        assert any(
            not torch.equal(b, p.detach()) for b, p in zip(before, policy.parameters(), strict=True)
        )

    def test_zero_lr_leaves_parameters_unchanged(self) -> None:
        policy = make_policy()
        before = [p.detach().clone() for p in policy.parameters()]
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
        reinforce_update(
            policy, optimizer, make_episodes(policy), gamma=GAMMA, normalize_returns=True
        )
        for b, p in zip(before, policy.parameters(), strict=True):
            assert torch.equal(b, p.detach())

    def test_metrics_present_and_finite(self) -> None:
        policy = make_policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        metrics = reinforce_update(
            policy, optimizer, make_episodes(policy), gamma=GAMMA, normalize_returns=True
        )
        expected = {
            "loss",
            "entropy",
            "grad_norm",
            "return_mean",
            "return_std",
            "train_return_mean",
        }
        assert expected <= set(metrics)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    def test_returns_do_not_leak_across_episodes(self) -> None:
        """The update's gradient must match a manual per-episode-returns loss.

        Episodes have different lengths and termination modes; discounting
        must restart at each episode boundary.
        """
        policy = make_policy()
        episodes = make_episodes(policy)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)  # keep params fixed
        reinforce_update(policy, optimizer, episodes, gamma=GAMMA, normalize_returns=False)
        update_grads = [p.grad.detach().clone() for p in policy.parameters() if p.grad is not None]

        policy.zero_grad()
        expected_returns = torch.cat([discounted_returns(ep.reward, GAMMA) for ep in episodes])
        obs = torch.cat([ep.obs for ep in episodes])
        actions = torch.cat([ep.action for ep in episodes])
        manual_loss = -(policy.action_log_prob(obs, actions) * expected_returns).mean()
        manual_loss.backward()  # type: ignore[no-untyped-call]
        manual_grads = [p.grad.detach().clone() for p in policy.parameters() if p.grad is not None]

        for got, expected in zip(update_grads, manual_grads, strict=True):
            assert torch.allclose(got, expected, atol=1e-7)

    def test_rejects_empty_batch(self) -> None:
        policy = make_policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        with pytest.raises(ValueError, match="at least one episode"):
            reinforce_update(policy, optimizer, [], gamma=GAMMA, normalize_returns=True)


class TestSeedControlledTraining:
    def test_same_seed_reproduces_parameters(self) -> None:
        first = train(tiny_config(seed=7))
        second = train(tiny_config(seed=7))
        for key, value in first.policy.state_dict().items():
            assert torch.equal(value, second.policy.state_dict()[key]), key

    def test_different_seeds_diverge(self) -> None:
        first = train(tiny_config(seed=7))
        second = train(tiny_config(seed=8))
        assert any(
            not torch.equal(value, second.policy.state_dict()[key])
            for key, value in first.policy.state_dict().items()
        )


class TestTrainOutputs:
    def test_history_and_counters(self) -> None:
        result = train(tiny_config(seed=3))
        assert len(result.history) == 2
        assert result.total_episodes == 4
        assert result.total_env_steps == int(result.history[-1]["env_steps"])
        assert not result.stopped_early
        assert len(result.final_eval.episode_returns) == 2

    def test_writes_run_directory(self, tmp_path: Path) -> None:
        result = train(tiny_config(seed=3), out_dir=tmp_path)
        assert (tmp_path / "config.yaml").exists()
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == len(result.history)
        payload = json.loads((tmp_path / "result.json").read_text())
        assert payload["total_episodes"] == result.total_episodes
        assert payload["final_eval_return_mean"] == result.final_eval.mean_return

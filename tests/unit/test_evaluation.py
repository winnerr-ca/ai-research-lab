"""Tests for the shared evaluation module (M3 extraction)."""

from __future__ import annotations

import copy

import pytest
import torch

from rlcore._testing import ScriptedEnv
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.evaluation import EvalStats, evaluate_policy


def make_policy() -> CategoricalMlpPolicy:
    torch.manual_seed(0)
    return CategoricalMlpPolicy(obs_dim=2, n_actions=2, hidden_sizes=(8,))


def make_model() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(obs_dim=2, n_actions=2, hidden_sizes=(8,))


class TestEvaluatePolicy:
    def test_returns_and_lengths_with_reinforce_actor(self) -> None:
        policy = make_policy()
        stats = evaluate_policy(
            ScriptedEnv([1.0, 2.0, 3.0]),
            lambda obs: policy.act(obs, deterministic=True),
            episodes=3,
        )
        assert stats.episode_returns == (6.0, 6.0, 6.0)
        assert stats.episode_lengths == (3, 3, 3)

    def test_returns_with_ppo_actor(self) -> None:
        model = make_model()
        stats = evaluate_policy(
            ScriptedEnv([1.0, 2.0]),
            lambda obs: model.act(obs, deterministic=True)[0],
            episodes=2,
        )
        assert stats.episode_returns == (3.0, 3.0)

    def test_does_not_modify_parameters(self) -> None:
        policy = make_policy()
        before = copy.deepcopy(policy.state_dict())
        evaluate_policy(
            ScriptedEnv([1.0, 1.0]), lambda obs: policy.act(obs, deterministic=True), episodes=2
        )
        for key, value in policy.state_dict().items():
            assert torch.equal(value, before[key])

    def test_greedy_closure_consumes_no_rng(self) -> None:
        policy = make_policy()
        generator = torch.Generator().manual_seed(0)
        state_before = generator.get_state()
        evaluate_policy(
            ScriptedEnv([1.0, 1.0]),
            lambda obs: policy.act(obs, generator=generator, deterministic=True),
            episodes=2,
        )
        assert torch.equal(generator.get_state(), state_before)

    def test_stochastic_closure_is_generator_controlled(self) -> None:
        policy = make_policy()

        def run(seed: int) -> tuple[float, ...]:
            generator = torch.Generator().manual_seed(seed)
            return evaluate_policy(
                ScriptedEnv([1.0] * 6),
                lambda obs: policy.act(obs, generator=generator, deterministic=False),
                episodes=2,
            ).episode_returns

        assert run(5) == run(5)

    def test_rejects_zero_episodes(self) -> None:
        policy = make_policy()
        with pytest.raises(ValueError, match=">= 1"):
            evaluate_policy(
                ScriptedEnv([1.0]), lambda obs: policy.act(obs, deterministic=True), episodes=0
            )


class TestEvalStats:
    def test_aggregates(self) -> None:
        stats = EvalStats(episode_returns=(1.0, 3.0), episode_lengths=(1, 3))
        assert stats.mean_return == 2.0
        assert stats.std_return == 1.0
        assert stats.min_return == 1.0
        assert stats.max_return == 3.0

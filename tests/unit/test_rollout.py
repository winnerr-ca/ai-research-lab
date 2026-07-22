"""Tests for episode collection and evaluation on scripted environments."""

from __future__ import annotations

import copy

import pytest
import torch

from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.reinforce.rollout import EvalStats, collect_episode, evaluate
from rlcore.testing import ScriptedEnv


def make_policy() -> CategoricalMlpPolicy:
    torch.manual_seed(0)
    return CategoricalMlpPolicy(obs_dim=2, n_actions=2, hidden_sizes=(8,))


class TestCollectEpisode:
    def test_contents_and_dtypes(self) -> None:
        env = ScriptedEnv([1.0, 2.0, 3.0])
        batch = collect_episode(env, make_policy())
        assert set(batch.keys()) == {"obs", "action", "reward", "terminated", "truncated"}
        assert len(batch) == 3
        assert batch.obs.dtype == torch.float32
        assert batch.action.dtype == torch.int64
        assert batch.reward.dtype == torch.float32
        assert torch.equal(batch.reward, torch.tensor([1.0, 2.0, 3.0]))

    def test_stores_pre_action_observations(self) -> None:
        # ScriptedEnv's observation at step t is filled with t: the batch must
        # hold the observation the action was computed FROM, not the successor.
        batch = collect_episode(ScriptedEnv([1.0, 1.0, 1.0]), make_policy())
        assert torch.equal(batch.obs[:, 0], torch.tensor([0.0, 1.0, 2.0]))

    def test_termination_flags(self) -> None:
        batch = collect_episode(ScriptedEnv([1.0, 1.0, 1.0], terminate=True), make_policy())
        assert torch.equal(batch.terminated, torch.tensor([False, False, True]))
        assert not bool(batch.truncated.any())

    def test_truncation_flags(self) -> None:
        batch = collect_episode(ScriptedEnv([1.0, 1.0, 1.0], terminate=False), make_policy())
        assert torch.equal(batch.truncated, torch.tensor([False, False, True]))
        assert not bool(batch.terminated.any())

    def test_actions_reach_the_env(self) -> None:
        env = ScriptedEnv([1.0, 1.0])
        batch = collect_episode(env, make_policy())
        assert env.received_actions == batch.action.tolist()

    def test_deterministic_collection_repeats(self) -> None:
        policy = make_policy()
        first = collect_episode(ScriptedEnv([1.0] * 5), policy, deterministic=True)
        second = collect_episode(ScriptedEnv([1.0] * 5), policy, deterministic=True)
        assert torch.equal(first.action, second.action)

    def test_sampled_collection_is_generator_controlled(self) -> None:
        policy = make_policy()
        first = collect_episode(
            ScriptedEnv([1.0] * 8), policy, generator=torch.Generator().manual_seed(5)
        )
        second = collect_episode(
            ScriptedEnv([1.0] * 8), policy, generator=torch.Generator().manual_seed(5)
        )
        assert torch.equal(first.action, second.action)


class TestEvaluate:
    def test_returns_and_lengths(self) -> None:
        stats = evaluate(ScriptedEnv([1.0, 2.0, 3.0]), make_policy(), episodes=3)
        assert stats.episode_returns == (6.0, 6.0, 6.0)
        assert stats.episode_lengths == (3, 3, 3)

    def test_does_not_modify_policy(self) -> None:
        policy = make_policy()
        before = copy.deepcopy(policy.state_dict())
        evaluate(ScriptedEnv([1.0, 1.0]), policy, episodes=2)
        for key, value in policy.state_dict().items():
            assert torch.equal(value, before[key])

    def test_greedy_eval_consumes_no_rng(self) -> None:
        generator = torch.Generator().manual_seed(0)
        state_before = generator.get_state()
        evaluate(ScriptedEnv([1.0, 1.0]), make_policy(), episodes=2, generator=generator)
        assert torch.equal(generator.get_state(), state_before)

    def test_rejects_zero_episodes(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            evaluate(ScriptedEnv([1.0]), make_policy(), episodes=0)


class TestEvalStats:
    def test_aggregates(self) -> None:
        stats = EvalStats(episode_returns=(1.0, 3.0), episode_lengths=(1, 3))
        assert stats.mean_return == 2.0
        assert stats.std_return == 1.0
        assert stats.min_return == 1.0
        assert stats.max_return == 3.0

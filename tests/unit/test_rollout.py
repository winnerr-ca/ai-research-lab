"""Tests for REINFORCE episode collection on scripted environments."""

from __future__ import annotations

import torch

from rlcore._testing import ScriptedEnv
from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.reinforce.rollout import collect_episode


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

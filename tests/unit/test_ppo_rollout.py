"""Tests for the PPO rollout collector: bootstrap contract and persistence."""

from __future__ import annotations

import copy

import pytest
import torch

from rlcore._testing import ScriptedEnv
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.ppo.rollout import RolloutCollector, evaluate


def make_model() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(obs_dim=2, n_actions=2, hidden_sizes=(8,))


def scripted_obs(step: int) -> torch.Tensor:
    return torch.full((1, 2), float(step))


class TestCollect:
    def test_fields_and_dtypes(self) -> None:
        collector = RolloutCollector(ScriptedEnv([1.0] * 10))
        rollout = collector.collect(make_model(), 4, generator=torch.Generator().manual_seed(0))
        batch = rollout.batch
        expected_keys = {
            "obs",
            "action",
            "reward",
            "terminated",
            "truncated",
            "done",
            "logprob",
            "value",
            "next_value",
        }
        assert set(batch.keys()) == expected_keys
        assert len(batch) == 4
        assert batch.logprob.dtype == torch.float32
        assert batch.next_value.dtype == torch.float32

    def test_next_values_without_boundary(self) -> None:
        """Non-boundary steps carry V(s_{t+1}); the horizon end bootstraps.

        The tail bootstrap is the value of the observation the next rollout
        will start from.
        """
        model = make_model()
        collector = RolloutCollector(ScriptedEnv([1.0] * 10))
        rollout = collector.collect(model, 3, generator=torch.Generator().manual_seed(0))
        batch = rollout.batch
        assert not bool(batch.done.any())
        assert torch.allclose(batch.next_value[:-1], batch.value[1:], atol=1e-6)
        with torch.no_grad():
            tail = model.value(scripted_obs(3)).item()
        assert batch.next_value[-1].item() == pytest.approx(tail, abs=1e-6)

    def test_terminal_next_value_is_zero(self) -> None:
        collector = RolloutCollector(ScriptedEnv([1.0, 2.0, 3.0], terminate=True))
        rollout = collector.collect(make_model(), 5, generator=torch.Generator().manual_seed(0))
        batch = rollout.batch
        assert bool(batch.terminated[2])
        assert batch.next_value[2].item() == 0.0
        assert rollout.completed_returns == (6.0,)
        # After the boundary, collection continues in a fresh episode.
        assert batch.obs[3, 0].item() == 0.0

    def test_truncation_next_value_bootstraps_final_obs(self) -> None:
        model = make_model()
        collector = RolloutCollector(ScriptedEnv([1.0, 2.0, 3.0], terminate=False))
        rollout = collector.collect(model, 3, generator=torch.Generator().manual_seed(0))
        batch = rollout.batch
        assert bool(batch.truncated[2]) and not bool(batch.terminated[2])
        with torch.no_grad():
            final_value = model.value(scripted_obs(3)).item()
        assert batch.next_value[2].item() == pytest.approx(final_value, abs=1e-6)
        assert batch.next_value[2].item() != 0.0

    def test_state_persists_across_collects(self) -> None:
        """A 4-step episode split across two 2-step rollouts continues."""
        collector = RolloutCollector(ScriptedEnv([1.0] * 4, terminate=True))
        model = make_model()
        generator = torch.Generator().manual_seed(0)
        first = collector.collect(model, 2, generator=generator)
        second = collector.collect(model, 2, generator=generator)
        assert torch.equal(first.batch.obs[:, 0], torch.tensor([0.0, 1.0]))
        assert torch.equal(second.batch.obs[:, 0], torch.tensor([2.0, 3.0]))
        assert not bool(first.batch.done.any())
        assert bool(second.batch.done[1])
        assert second.completed_returns == (4.0,)

    def test_behavior_logprobs_match_recomputation(self) -> None:
        model = make_model()
        collector = RolloutCollector(ScriptedEnv([1.0] * 6))
        batch = collector.collect(model, 6, generator=torch.Generator().manual_seed(0)).batch
        with torch.no_grad():
            log_prob, _, value = model.evaluate_actions(batch.obs, batch.action)
        assert torch.allclose(batch.logprob, log_prob, atol=1e-6)
        assert torch.allclose(batch.value, value, atol=1e-6)

    def test_rejects_zero_steps(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RolloutCollector(ScriptedEnv([1.0])).collect(make_model(), 0)


class TestEvaluate:
    def test_returns_and_no_side_effects(self) -> None:
        model = make_model()
        before = copy.deepcopy(model.state_dict())
        stats = evaluate(ScriptedEnv([1.0, 2.0, 3.0]), model, episodes=3)
        assert stats.episode_returns == (6.0, 6.0, 6.0)
        assert stats.episode_lengths == (3, 3, 3)
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key])

    def test_greedy_eval_consumes_no_rng(self) -> None:
        generator = torch.Generator().manual_seed(0)
        state_before = generator.get_state()
        evaluate(ScriptedEnv([1.0, 1.0]), make_model(), episodes=2, generator=generator)
        assert torch.equal(generator.get_state(), state_before)

    def test_rejects_zero_episodes(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            evaluate(ScriptedEnv([1.0]), make_model(), episodes=0)

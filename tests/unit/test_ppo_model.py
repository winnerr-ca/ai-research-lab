"""Tests for the actor-critic model: shapes, consistency, determinism."""

from __future__ import annotations

import math

import torch

from rlcore.agents.ppo.model import ActorCritic


def make_model(obs_dim: int = 4, n_actions: int = 3) -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(obs_dim, n_actions, hidden_sizes=(8,))


class TestShapes:
    def test_logits_and_value(self) -> None:
        model = make_model()
        obs = torch.randn(5, 4)
        assert model.logits(obs).shape == (5, 3)
        assert model.value(obs).shape == (5,)

    def test_evaluate_actions(self) -> None:
        model = make_model()
        obs = torch.randn(5, 4)
        actions = torch.tensor([0, 1, 2, 0, 1])
        log_prob, entropy, value = model.evaluate_actions(obs, actions)
        assert log_prob.shape == entropy.shape == value.shape == (5,)
        assert log_prob.requires_grad and entropy.requires_grad and value.requires_grad

    def test_act_shapes_dtypes_no_grad(self) -> None:
        model = make_model()
        action, log_prob, value = model.act(torch.randn(6, 4))
        assert action.shape == log_prob.shape == value.shape == (6,)
        assert action.dtype == torch.int64
        assert not log_prob.requires_grad and not value.requires_grad


class TestConsistency:
    def test_log_prob_matches_log_softmax_gather(self) -> None:
        model = make_model()
        obs = torch.randn(5, 4)
        actions = torch.tensor([2, 0, 1, 2, 0])
        log_prob, _, _ = model.evaluate_actions(obs, actions)
        expected = model.logits(obs).log_softmax(dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        assert torch.allclose(log_prob, expected, atol=1e-7)

    def test_entropy_of_uniform_logits_is_log_n(self) -> None:
        model = make_model()
        with torch.no_grad():  # force uniform logits through a zeroed final layer
            final = model.policy_net[-1]
            assert isinstance(final, torch.nn.Linear)
            final.weight.zero_()
            final.bias.zero_()
        _, entropy, _ = model.evaluate_actions(torch.randn(4, 4), torch.zeros(4, dtype=torch.int64))
        assert torch.allclose(entropy, torch.full((4,), math.log(3.0)), atol=1e-6)

    def test_act_log_prob_matches_evaluate_actions(self) -> None:
        model = make_model()
        obs = torch.randn(8, 4)
        action, log_prob, value = model.act(obs, generator=torch.Generator().manual_seed(1))
        eval_log_prob, _, eval_value = model.evaluate_actions(obs, action)
        assert torch.allclose(log_prob, eval_log_prob.detach(), atol=1e-6)
        assert torch.allclose(value, eval_value.detach(), atol=1e-6)


class TestDeterminism:
    def test_deterministic_is_argmax(self) -> None:
        model = make_model()
        obs = torch.randn(6, 4)
        action, _, _ = model.act(obs, deterministic=True)
        assert torch.equal(action, model.logits(obs).argmax(dim=-1))

    def test_sampling_is_generator_controlled(self) -> None:
        model = make_model()
        obs = torch.randn(64, 4)
        first, _, _ = model.act(obs, generator=torch.Generator().manual_seed(3))
        second, _, _ = model.act(obs, generator=torch.Generator().manual_seed(3))
        assert torch.equal(first, second)

    def test_separate_networks_no_shared_parameters(self) -> None:
        model = make_model()
        policy_params = {id(p) for p in model.policy_net.parameters()}
        value_params = {id(p) for p in model.value_net.parameters()}
        assert policy_params.isdisjoint(value_params)

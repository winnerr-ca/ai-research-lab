"""Tests for the categorical MLP policy: shapes, sampling, exact gradients."""

from __future__ import annotations

import math

import torch

from rlcore.agents.reinforce.policy import CategoricalMlpPolicy


def make_policy(obs_dim: int = 4, n_actions: int = 3) -> CategoricalMlpPolicy:
    torch.manual_seed(0)
    return CategoricalMlpPolicy(obs_dim, n_actions, hidden_sizes=(8,))


class TestShapes:
    def test_logits_shape(self) -> None:
        assert make_policy().forward(torch.randn(5, 4)).shape == (5, 3)

    def test_log_probs_normalized(self) -> None:
        log_probs = make_policy().log_probs(torch.randn(5, 4))
        assert log_probs.shape == (5, 3)
        assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_action_log_prob_shape(self) -> None:
        policy = make_policy()
        actions = torch.tensor([0, 2, 1, 0, 2])
        assert policy.action_log_prob(torch.randn(5, 4), actions).shape == (5,)

    def test_entropy_bounds(self) -> None:
        entropy = make_policy().entropy(torch.randn(5, 4))
        assert entropy.shape == (5,)
        assert bool((entropy >= 0).all())
        assert bool((entropy <= math.log(3) + 1e-6).all())

    def test_act_shape_and_dtype(self) -> None:
        actions = make_policy().act(torch.randn(5, 4))
        assert actions.shape == (5,)
        assert actions.dtype == torch.int64

    def test_empty_hidden_sizes_gives_single_linear(self) -> None:
        policy = CategoricalMlpPolicy(4, 2, hidden_sizes=())
        assert sum(1 for _ in policy.net) == 1


class TestActing:
    def test_deterministic_is_argmax(self) -> None:
        policy = make_policy()
        obs = torch.randn(6, 4)
        assert torch.equal(policy.act(obs, deterministic=True), policy.forward(obs).argmax(dim=-1))

    def test_sampling_is_seed_controlled(self) -> None:
        policy = make_policy()
        obs = torch.randn(64, 4)
        first = policy.act(obs, generator=torch.Generator().manual_seed(3))
        second = policy.act(obs, generator=torch.Generator().manual_seed(3))
        assert torch.equal(first, second)

    def test_act_builds_no_graph(self) -> None:
        assert not make_policy().act(torch.randn(2, 4)).requires_grad


class TestGradients:
    def test_gradients_flow_to_all_parameters(self) -> None:
        policy = make_policy()
        loss = -policy.action_log_prob(torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])).mean()
        loss.backward()  # type: ignore[no-untyped-call]
        for name, param in policy.named_parameters():
            assert param.grad is not None, name
            assert bool(param.grad.abs().sum() > 0), name

    def test_gradient_matches_analytic_formula(self) -> None:
        """For a linear policy, autograd must match the softmax score function.

        With logits z = Wx + b and loss L = -G * log softmax(z)[a]:
        dL/dz = -G * (onehot(a) - softmax(z)), dL/dW = dL/dz x^T, dL/db = dL/dz.
        """
        torch.manual_seed(1)
        policy = CategoricalMlpPolicy(obs_dim=3, n_actions=4, hidden_sizes=())
        linear = policy.net[0]
        assert isinstance(linear, torch.nn.Linear)

        x = torch.randn(1, 3)
        action = torch.tensor([2])
        g = 1.7
        loss = -(policy.action_log_prob(x, action) * g).mean()
        loss.backward()  # type: ignore[no-untyped-call]

        with torch.no_grad():
            probs = policy.forward(x).softmax(dim=-1).squeeze(0)
            onehot = torch.zeros(4)
            onehot[2] = 1.0
            dlogits = -g * (onehot - probs)
            expected_weight_grad = torch.outer(dlogits, x.squeeze(0))
        assert linear.weight.grad is not None
        assert linear.bias.grad is not None
        assert torch.allclose(linear.weight.grad, expected_weight_grad, atol=1e-6)
        assert torch.allclose(linear.bias.grad, dlogits, atol=1e-6)

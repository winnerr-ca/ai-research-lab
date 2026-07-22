"""Exact tests for GAE: limits, boundaries, and bootstrap semantics."""

from __future__ import annotations

import pytest
import torch

from rlcore.agents.ppo.gae import compute_gae


def reference_bootstrapped_returns(
    rewards: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Discounted returns with bootstrap, restarted at every boundary."""
    horizon = rewards.shape[0]
    returns = torch.empty_like(rewards)
    for t in range(horizon - 1, -1, -1):
        if bool(dones[t]) or t == horizon - 1:
            returns[t] = rewards[t] + gamma * next_values[t]
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    horizon = 12
    rewards = torch.rand(horizon, generator=generator)
    values = torch.rand(horizon, generator=generator)
    dones = torch.zeros(horizon, dtype=torch.bool)
    dones[4] = True  # true terminal mid-rollout
    next_values = torch.empty(horizon)
    for t in range(horizon):
        if bool(dones[t]):
            next_values[t] = 0.0
        elif t + 1 < horizon:
            next_values[t] = values[t + 1]
        else:
            next_values[t] = 0.37  # tail bootstrap
    return rewards, values, next_values, dones


class TestLimits:
    def test_lambda_zero_is_td_residual(self) -> None:
        rewards, values, next_values, dones = make_inputs()
        advantages, _ = compute_gae(rewards, values, next_values, dones, gamma=0.9, lam=0.0)
        deltas = rewards + 0.9 * next_values - values
        assert torch.allclose(advantages, deltas, atol=1e-6)

    def test_lambda_one_recovers_bootstrapped_returns(self) -> None:
        rewards, values, next_values, dones = make_inputs()
        advantages, value_targets = compute_gae(
            rewards, values, next_values, dones, gamma=0.9, lam=1.0
        )
        expected = reference_bootstrapped_returns(rewards, next_values, dones, gamma=0.9)
        assert torch.allclose(advantages + values, expected, atol=1e-5)
        assert torch.allclose(value_targets, expected, atol=1e-5)

    def test_value_targets_are_advantages_plus_values(self) -> None:
        rewards, values, next_values, dones = make_inputs()
        advantages, value_targets = compute_gae(
            rewards, values, next_values, dones, gamma=0.99, lam=0.95
        )
        assert torch.allclose(value_targets, advantages + values, atol=1e-6)


class TestBoundaries:
    def test_hand_computed_three_step_case(self) -> None:
        # gamma=0.5, lam=0.5; V=[1, 2, 3]; r=[1, 1, 1]; terminal at t=2.
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([1.0, 2.0, 3.0])
        next_values = torch.tensor([2.0, 3.0, 0.0])
        dones = torch.tensor([False, False, True])
        advantages, _ = compute_gae(rewards, values, next_values, dones, gamma=0.5, lam=0.5)
        # delta = [1+1-1, 1+1.5-2, 1+0-3] = [1, 0.5, -2]
        # A_2 = -2; A_1 = 0.5 + 0.25*(-2) = 0; A_0 = 1 + 0.25*0 = 1
        assert torch.allclose(advantages, torch.tensor([1.0, 0.0, -2.0]), atol=1e-6)

    def test_recursion_never_crosses_boundaries(self) -> None:
        rewards, values, next_values, dones = make_inputs()
        advantages, _ = compute_gae(rewards, values, next_values, dones, gamma=0.9, lam=0.95)
        # Perturbing pre-boundary rewards must not change post-boundary advantages.
        perturbed = rewards.clone()
        perturbed[:5] += 100.0
        perturbed_adv, _ = compute_gae(perturbed, values, next_values, dones, gamma=0.9, lam=0.95)
        assert torch.allclose(advantages[5:], perturbed_adv[5:], atol=1e-6)
        assert not torch.allclose(advantages[:5], perturbed_adv[:5])

    def test_truncation_bootstraps_termination_does_not(self) -> None:
        """The terminal/truncated boundary advantage differs by gamma * V(s_final).

        Same rewards and values; only the boundary's bootstrap value changes.
        """
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        final_value = 2.0
        dones = torch.tensor([False, False, True])
        terminal_next = torch.tensor([0.5, 0.5, 0.0])
        truncated_next = torch.tensor([0.5, 0.5, final_value])
        adv_term, _ = compute_gae(rewards, values, terminal_next, dones, gamma=0.9, lam=0.0)
        adv_trunc, _ = compute_gae(rewards, values, truncated_next, dones, gamma=0.9, lam=0.0)
        assert torch.allclose(
            adv_trunc[2] - adv_term[2], torch.tensor(0.9 * final_value), atol=1e-6
        )
        assert torch.allclose(adv_trunc[:2], adv_term[:2], atol=1e-6)


class TestValidation:
    def test_rejects_grad_inputs(self) -> None:
        rewards, values, next_values, dones = make_inputs()
        with pytest.raises(ValueError, match="require grad"):
            compute_gae(rewards, values.requires_grad_(), next_values, dones, gamma=0.9, lam=0.9)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_gae(
                torch.ones(3),
                torch.ones(4),
                torch.ones(3),
                torch.zeros(3, dtype=torch.bool),
                gamma=0.9,
                lam=0.9,
            )

    def test_rejects_non_bool_dones(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            compute_gae(
                torch.ones(3), torch.ones(3), torch.ones(3), torch.zeros(3), gamma=0.9, lam=0.9
            )

    @pytest.mark.parametrize(("gamma", "lam"), [(-0.1, 0.5), (1.1, 0.5), (0.9, -0.1), (0.9, 1.1)])
    def test_rejects_out_of_range(self, gamma: float, lam: float) -> None:
        with pytest.raises(ValueError, match="must be in"):
            compute_gae(
                torch.ones(3),
                torch.ones(3),
                torch.ones(3),
                torch.zeros(3, dtype=torch.bool),
                gamma=gamma,
                lam=lam,
            )

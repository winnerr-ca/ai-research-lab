"""Exact tests for discounted reward-to-go."""

from __future__ import annotations

import pytest
import torch

from rlcore.agents.reinforce.returns import discounted_returns


class TestValues:
    def test_hand_computed_case(self) -> None:
        returns = discounted_returns(torch.tensor([1.0, 2.0, 3.0]), gamma=0.5)
        # G_2 = 3; G_1 = 2 + 0.5*3 = 3.5; G_0 = 1 + 0.5*3.5 = 2.75
        assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))

    def test_gamma_one_is_reverse_cumsum(self) -> None:
        rewards = torch.tensor([1.0, 2.0, 3.0])
        returns = discounted_returns(rewards, gamma=1.0)
        assert torch.allclose(returns, torch.tensor([6.0, 5.0, 3.0]))

    def test_gamma_zero_is_rewards(self) -> None:
        rewards = torch.tensor([1.0, -2.0, 3.0])
        assert torch.equal(discounted_returns(rewards, gamma=0.0), rewards)

    def test_single_step(self) -> None:
        assert torch.equal(discounted_returns(torch.tensor([7.0]), gamma=0.9), torch.tensor([7.0]))

    def test_matches_closed_form_for_constant_reward(self) -> None:
        gamma, horizon = 0.95, 50
        returns = discounted_returns(torch.ones(horizon), gamma=gamma)
        expected = torch.tensor(
            [(1 - gamma ** (horizon - t)) / (1 - gamma) for t in range(horizon)]
        )
        assert torch.allclose(returns, expected, atol=1e-5)

    def test_trailing_return_is_last_reward(self) -> None:
        # Nothing is bootstrapped after the final step (see the truncation
        # bias note in the module and docs/algorithms/reinforce.md).
        returns = discounted_returns(torch.tensor([1.0, 2.0, 5.0]), gamma=0.99)
        assert returns[-1].item() == 5.0

    def test_dtype_preserved(self) -> None:
        assert (
            discounted_returns(torch.ones(3, dtype=torch.float64), gamma=0.9).dtype == torch.float64
        )


class TestValidation:
    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            discounted_returns(torch.ones(3, 2), gamma=0.9)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            discounted_returns(torch.ones(0), gamma=0.9)

    def test_rejects_integer_rewards(self) -> None:
        with pytest.raises(ValueError, match="floating-point"):
            discounted_returns(torch.ones(3, dtype=torch.int64), gamma=0.9)

    @pytest.mark.parametrize("gamma", [-0.1, 1.1])
    def test_rejects_gamma_out_of_range(self, gamma: float) -> None:
        with pytest.raises(ValueError, match="gamma"):
            discounted_returns(torch.ones(3), gamma=gamma)

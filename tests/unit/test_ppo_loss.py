"""Controlled reference calculations for PPO loss components."""

from __future__ import annotations

import math

import pytest
import torch

from rlcore.agents.ppo.loss import approx_kl_divergence, clipped_surrogate_loss, value_mse_loss


def log_probs_with_ratios(ratios: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (new, old) log-prob tensors with exactly the given ratios."""
    old = torch.full((len(ratios),), -1.0)
    new = old + torch.log(torch.tensor(ratios))
    return new, old


class TestClippedSurrogate:
    def test_unclipped_region_matches_plain_surrogate(self) -> None:
        new, old = log_probs_with_ratios([1.05, 0.9, 1.15])
        advantages = torch.tensor([1.0, -2.0, 0.5])
        loss, clip_fraction = clipped_surrogate_loss(new, old, advantages, clip_range=0.2)
        expected = -((new - old).exp() * advantages).mean()
        assert torch.allclose(loss, expected, atol=1e-6)
        assert clip_fraction.item() == 0.0

    def test_clipped_cases_hand_computed(self) -> None:
        # ratio 2.0, A=+1: min(2.0, 1.2) = 1.2  (clip caps the gain)
        # ratio 0.5, A=-1: min(-0.5, -0.8) = -0.8  (pessimistic branch)
        new, old = log_probs_with_ratios([2.0, 0.5])
        advantages = torch.tensor([1.0, -1.0])
        loss, clip_fraction = clipped_surrogate_loss(new, old, advantages, clip_range=0.2)
        assert torch.allclose(loss, torch.tensor(-(1.2 + (-0.8)) / 2.0), atol=1e-6)
        assert clip_fraction.item() == 1.0

    def test_identical_policies(self) -> None:
        new, old = log_probs_with_ratios([1.0, 1.0, 1.0])
        advantages = torch.tensor([1.0, 2.0, 3.0])
        loss, clip_fraction = clipped_surrogate_loss(new, old, advantages, clip_range=0.2)
        assert torch.allclose(loss, -advantages.mean(), atol=1e-6)
        assert clip_fraction.item() == 0.0

    def test_clipped_gradient_is_zero_beyond_range(self) -> None:
        """When the clipped branch is active, d loss / d new_log_prob = 0."""
        old = torch.tensor([-1.0])
        new = (old + math.log(2.0)).requires_grad_()
        advantages = torch.tensor([1.0])
        loss, _ = clipped_surrogate_loss(new, old.detach(), advantages, clip_range=0.2)
        loss.backward()  # type: ignore[no-untyped-call]
        assert new.grad is not None
        assert torch.allclose(new.grad, torch.zeros(1), atol=1e-8)

    def test_rejects_grad_constants(self) -> None:
        new, old = log_probs_with_ratios([1.0])
        with pytest.raises(ValueError, match="not require grad"):
            clipped_surrogate_loss(new, old.requires_grad_(), torch.ones(1), clip_range=0.2)

    def test_rejects_bad_clip_range(self) -> None:
        new, old = log_probs_with_ratios([1.0])
        with pytest.raises(ValueError, match="clip_range"):
            clipped_surrogate_loss(new, old, torch.ones(1), clip_range=0.0)


class TestValueLoss:
    def test_hand_computed_mse(self) -> None:
        loss = value_mse_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 4.0]))
        assert torch.allclose(loss, torch.tensor((1.0 + 4.0) / 2.0), atol=1e-6)

    def test_rejects_grad_targets(self) -> None:
        with pytest.raises(ValueError, match="not require grad"):
            value_mse_loss(torch.ones(2), torch.ones(2, requires_grad=True))


class TestApproxKl:
    def test_zero_for_identical_policies(self) -> None:
        new, old = log_probs_with_ratios([1.0, 1.0])
        assert approx_kl_divergence(new, old).item() == 0.0

    def test_exact_for_known_ratio(self) -> None:
        new, old = log_probs_with_ratios([2.0])
        assert torch.allclose(
            approx_kl_divergence(new, old), torch.tensor(1.0 - math.log(2.0)), atol=1e-6
        )

    def test_non_negative_on_random_inputs(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for _ in range(10):
            new = torch.randn(64, generator=generator)
            old = torch.randn(64, generator=generator)
            assert approx_kl_divergence(new, old).item() >= 0.0

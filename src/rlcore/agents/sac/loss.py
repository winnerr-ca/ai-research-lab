"""SAC loss components and Polyak averaging as pure, testable functions.

Conventions follow Haarnoja et al. (2018), the two-Q variant with automatic
entropy tuning (Haarnoja et al., 2019, "Soft Actor-Critic Algorithms and
Applications"). Termination semantics follow the platform contract: the
soft bootstrap is suppressed exactly where ``terminated`` is true, and
truncated transitions bootstrap from their stored final observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812  (torch's own convention)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn


def critic_targets(
    rewards: Tensor,
    terminated: Tensor,
    next_q_min: Tensor,
    next_log_prob: Tensor,
    *,
    gamma: float,
    alpha: float,
) -> Tensor:
    """Soft TD targets ``r + gamma (1 - term) (min Q'(s',a') - alpha log pi(a'|s'))``.

    Args:
        rewards: ``[B]`` rewards.
        terminated: ``[B]`` bool MDP-true endings (bootstrap suppressed).
        next_q_min: ``[B]`` element-wise minimum of the two *target* critics
            at the next state and a fresh policy action (no grad).
        next_log_prob: ``[B]`` log-probability of that action (no grad).
        gamma: Discount in ``[0, 1]``.
        alpha: Entropy temperature, >= 0.

    Raises:
        ValueError: On grad-carrying inputs, non-bool ``terminated``, or
            out-of-range ``gamma``/``alpha``.
    """
    if terminated.dtype != torch.bool:
        raise ValueError(f"terminated must be bool, got {terminated.dtype}.")
    if next_q_min.requires_grad or next_log_prob.requires_grad:
        raise ValueError("target inputs must not require grad; compute them under no_grad.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")
    soft_value = next_q_min - alpha * next_log_prob
    return rewards + gamma * (~terminated).to(rewards.dtype) * soft_value


def critic_loss(q1: Tensor, q2: Tensor, targets: Tensor) -> Tensor:
    """Sum of MSE losses of both critics against the shared soft targets.

    Raises:
        ValueError: If ``targets`` requires grad.
    """
    if targets.requires_grad:
        raise ValueError("targets must not require grad; they are data, not graph nodes.")
    return F.mse_loss(q1, targets) + F.mse_loss(q2, targets)


def actor_loss(q_min: Tensor, log_prob: Tensor, *, alpha: float) -> Tensor:
    """Policy objective ``E[alpha log pi(a|s) - min Q(s, a)]`` (minimized).

    ``q_min`` and ``log_prob`` must be differentiable through the
    reparameterized action — that is the pathwise gradient SAC relies on.
    """
    return (alpha * log_prob - q_min).mean()


def alpha_loss(log_alpha: Tensor, log_prob: Tensor, *, target_entropy: float) -> Tensor:
    """Temperature objective ``E[-log_alpha (log pi + target_entropy)]``.

    Gradient intuition (tested): when the policy's entropy is *below*
    target (``log_prob + target_entropy > 0`` on average), the gradient
    pushes ``log_alpha`` up — more entropy pressure — and vice versa.

    Args:
        log_alpha: Scalar learnable log-temperature (requires grad).
        log_prob: ``[B]`` sampled log-probabilities (treated as data).
        target_entropy: Target policy entropy (``-act_dim`` by default).
    """
    return -(log_alpha * (log_prob.detach() + target_entropy)).mean()


def polyak_update(source: nn.Module, target: nn.Module, *, tau: float) -> None:
    """In-place soft update ``theta_target <- tau theta + (1 - tau) theta_target``.

    Args:
        source: Online network.
        target: Target network (mutated).
        tau: Interpolation factor in ``(0, 1]``; ``1`` copies outright.

    Raises:
        ValueError: If ``tau`` is outside ``(0, 1]``.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}.")
    with torch.no_grad():
        source_params: Iterable[Tensor] = source.parameters()
        for param, target_param in zip(source_params, target.parameters(), strict=True):
            target_param.mul_(1.0 - tau).add_(param, alpha=tau)

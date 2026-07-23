"""DQN loss components and schedules as pure, testable functions.

The TD target implements the platform's termination contract: bootstrap is
suppressed exactly where ``terminated`` is true (an MDP-true ending), and
truncated transitions bootstrap naturally because their stored ``next_obs``
is the episode's true final observation (see :mod:`rlcore.replay`).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812  (torch's own convention)


def linear_epsilon(step: int, *, start: float, final: float, decay_steps: int) -> float:
    """Linearly anneal exploration from ``start`` to ``final``.

    Constant at ``final`` after ``decay_steps``.

    Raises:
        ValueError: If ``decay_steps < 1``.
    """
    if decay_steps < 1:
        raise ValueError(f"decay_steps must be >= 1, got {decay_steps}.")
    fraction = min(max(step, 0) / decay_steps, 1.0)
    return start + fraction * (final - start)


def select_next_values(
    target_q_next: Tensor, online_q_next: Tensor | None, *, double: bool
) -> Tensor:
    """Choose the next-state value estimate for the TD target.

    Standard DQN: ``max_a Q_target(s', a)`` — the same network both selects
    and evaluates, which is the source of the documented overestimation
    bias. Double DQN (van Hasselt et al., 2016): the *online* network
    selects ``a* = argmax_a Q_online(s', a)`` and the target network
    evaluates ``Q_target(s', a*)``, decoupling selection from evaluation.

    Args:
        target_q_next: ``[B, A]`` target-network Q-values at ``s'`` (no
            grad).
        online_q_next: ``[B, A]`` online-network Q-values at ``s'`` (no
            grad); required when ``double`` is true.
        double: Use Double-DQN action selection.

    Returns:
        ``[B]`` next-state values.

    Raises:
        ValueError: If grad-carrying inputs are passed, or ``double`` is set
            without ``online_q_next``.
    """
    if target_q_next.requires_grad:
        raise ValueError("target_q_next must not require grad.")
    if not double:
        return target_q_next.max(dim=-1).values
    if online_q_next is None:
        raise ValueError("Double DQN requires online_q_next for action selection.")
    if online_q_next.requires_grad:
        raise ValueError("online_q_next must not require grad.")
    best_actions = online_q_next.argmax(dim=-1, keepdim=True)
    return target_q_next.gather(1, best_actions).squeeze(1)


def td_targets(rewards: Tensor, terminated: Tensor, next_values: Tensor, *, gamma: float) -> Tensor:
    """One-step TD targets ``r + gamma * (1 - terminated) * V(s')``.

    ``terminated`` suppresses bootstrap only at MDP-true endings; truncated
    transitions carry ``terminated=False`` and bootstrap from their stored
    final observation's value — the stated continuing-task objective.

    Args:
        rewards: ``[B]`` float rewards.
        terminated: ``[B]`` bool, MDP-true episode endings.
        next_values: ``[B]`` next-state value estimates (no grad).
        gamma: Discount in ``[0, 1]``.

    Raises:
        ValueError: On dtype violations, grad-carrying ``next_values``, or
            out-of-range ``gamma``.
    """
    if terminated.dtype != torch.bool:
        raise ValueError(f"terminated must be bool, got {terminated.dtype}.")
    if next_values.requires_grad:
        raise ValueError("next_values must not require grad; compute them under no_grad.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    return rewards + gamma * (~terminated).to(rewards.dtype) * next_values


def q_huber_loss(q_taken: Tensor, targets: Tensor) -> Tensor:
    """Huber (smooth-L1) loss between taken-action Q-values and TD targets.

    Huber rather than MSE follows the DQN lineage (error clipping in Mnih
    et al., 2015, is equivalent to Huber on the TD error) and bounds the
    gradient of outlier TD errors.

    Raises:
        ValueError: If ``targets`` requires grad.
    """
    if targets.requires_grad:
        raise ValueError("targets must not require grad; they are data, not graph nodes.")
    return F.smooth_l1_loss(q_taken, targets)

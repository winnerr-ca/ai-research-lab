"""Discounted reward-to-go for a single episode.

Lives inside the REINFORCE package for now; shared return/advantage
transforms are extracted to ``rlcore.data`` at the M3 consolidation pass.
"""

from __future__ import annotations

import torch
from torch import Tensor


def discounted_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Compute discounted reward-to-go ``G_t = sum_{k>=t} gamma^(k-t) r_k``.

    Operates on the rewards of ONE episode; callers with several episodes
    compute returns per episode and concatenate, so no discounting leaks
    across episode boundaries.

    The trailing return is ``G_{T-1} = r_{T-1}``: nothing is bootstrapped
    after the last step. For *truncated* (time-limited) episodes this
    underestimates the true return — vanilla REINFORCE has no value function
    to bootstrap from; see ``docs/algorithms/reinforce.md`` for the bias
    discussion.

    Args:
        rewards: Floating-point rewards of one episode, shape ``[T]``, T >= 1.
        gamma: Discount factor in ``[0, 1]``.

    Returns:
        Reward-to-go of shape ``[T]``, same dtype as ``rewards``.

    Raises:
        ValueError: If ``rewards`` is not 1-D floating-point and non-empty,
            or ``gamma`` is out of range.
    """
    if rewards.ndim != 1 or rewards.shape[0] == 0:
        raise ValueError(
            f"rewards must be a non-empty 1-D tensor, got shape {tuple(rewards.shape)}."
        )
    if not rewards.is_floating_point():
        raise ValueError(f"rewards must be floating-point, got {rewards.dtype}.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")

    returns = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype)
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns

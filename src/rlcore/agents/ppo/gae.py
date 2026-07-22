"""Generalized Advantage Estimation (Schulman et al., ICLR 2016).

The estimator is the exponentially weighted average of k-step advantage
estimates, computed by the standard backward recursion over TD residuals:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

Termination/truncation semantics are made explicit through the
``next_values`` contract below, implementing the task-objective choice
stated in ``docs/algorithms/reinforce.md`` §5: truncation bootstraps, true
termination does not, and the advantage recursion never crosses an episode
boundary of either kind.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE advantages and value targets for one rollout.

    The ``next_values`` contract (set by the collector) encodes bootstrap
    semantics per step ``t``:

    - **true terminal** (``terminated``): ``next_values[t] = 0`` — no reward
      exists past an absorbing state;
    - **truncated** (external time limit): ``next_values[t] = V(s_final)``,
      the value of the episode's final observation — the return past the
      cutoff is estimated, per the stated task objective;
    - otherwise: ``next_values[t] = V(s_{t+1})`` (for the rollout's last
      step, the value of the observation the next rollout will start from).

    ``dones[t]`` must be ``True`` at *both* kinds of episode boundary: the
    recursion ``A_t = delta_t + gamma*lam*(1-done_t)*A_{t+1}`` may never
    chain across a reset, because step ``t+1`` then belongs to a different
    episode.

    Limiting identities (tested):

    - ``lam=0``: ``A_t = delta_t`` (one-step TD residual);
    - ``lam=1``: ``A_t + V(s_t)`` equals the discounted bootstrapped return
      of the remaining segment.

    Args:
        rewards: ``[T]`` float rewards.
        values: ``[T]`` float ``V(s_t)`` estimates (no grad).
        next_values: ``[T]`` float bootstrap values per the contract above
            (no grad).
        dones: ``[T]`` bool, ``True`` where the episode ended at step ``t``.
        gamma: Discount factor in ``[0, 1]``.
        lam: GAE lambda in ``[0, 1]``.

    Returns:
        ``(advantages, value_targets)``, both ``[T]``; value targets are
        ``advantages + values``.

    Raises:
        ValueError: On shape/dtype violations, out-of-range ``gamma``/``lam``,
            or inputs that require grad (GAE inputs are data, not graph
            nodes — a silent gradient path here is a classic bug).
    """
    tensors = {"rewards": rewards, "values": values, "next_values": next_values}
    for name, tensor in tensors.items():
        if tensor.ndim != 1 or tensor.shape != rewards.shape:
            raise ValueError(f"{name} must be 1-D of equal length, got {tuple(tensor.shape)}.")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating-point, got {tensor.dtype}.")
        if tensor.requires_grad:
            raise ValueError(f"{name} must not require grad; detach collector outputs.")
    if dones.shape != rewards.shape or dones.dtype != torch.bool:
        raise ValueError(f"dones must be bool of shape {tuple(rewards.shape)}.")
    if rewards.shape[0] == 0:
        raise ValueError("rollout must contain at least one step.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}.")

    horizon = rewards.shape[0]
    advantages = torch.empty_like(rewards)
    advantage = torch.zeros((), dtype=rewards.dtype)
    not_done = 1.0 - dones.to(rewards.dtype)
    deltas = rewards + gamma * next_values - values
    for t in range(horizon - 1, -1, -1):
        advantage = deltas[t] + gamma * lam * not_done[t] * advantage
        advantages[t] = advantage
    return advantages, advantages + values

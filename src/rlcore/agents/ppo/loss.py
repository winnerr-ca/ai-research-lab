"""PPO loss components as pure, independently testable functions.

Conventions follow Schulman et al. (2017) and match Stable-Baselines3's
composition — ``policy_loss + vf_coef * value_loss - ent_coef * entropy`` —
so reference comparisons stay honest where they are made.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812  (torch's own convention)


def _require_constant(name: str, tensor: Tensor) -> None:
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require grad; it is data, not a graph node.")


def clipped_surrogate_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    *,
    clip_range: float,
) -> tuple[Tensor, Tensor]:
    """Clipped importance-sampled policy surrogate (PPO's central objective).

    With ratio ``rho = exp(new - old)``, the per-sample objective is
    ``min(rho * A, clip(rho, 1-eps, 1+eps) * A)`` — a pessimistic bound that
    removes the incentive to move ``rho`` beyond the clip range in the
    direction the advantage favors.

    Args:
        new_log_prob: ``[B]`` log-probabilities under the current policy
            (differentiable).
        old_log_prob: ``[B]`` behavior log-probabilities (constant data).
        advantages: ``[B]`` advantage estimates (constant data; normalize
            before calling if desired).
        clip_range: The clip epsilon, > 0.

    Returns:
        ``(loss, clip_fraction)``: the scalar loss (negated mean objective)
        and the fraction of samples whose ratio lies outside the clip range.

    Raises:
        ValueError: If constants require grad or ``clip_range <= 0``.
    """
    _require_constant("old_log_prob", old_log_prob)
    _require_constant("advantages", advantages)
    if clip_range <= 0.0:
        raise ValueError(f"clip_range must be > 0, got {clip_range}.")
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    return loss, clip_fraction


def value_mse_loss(values: Tensor, value_targets: Tensor) -> Tensor:
    """Mean-squared error between value predictions and GAE value targets.

    Args:
        values: ``[B]`` current value predictions (differentiable).
        value_targets: ``[B]`` targets ``A + V_old`` (constant data).

    Raises:
        ValueError: If ``value_targets`` requires grad.
    """
    _require_constant("value_targets", value_targets)
    return F.mse_loss(values, value_targets)


def explained_variance(values: Tensor, value_targets: Tensor) -> float:
    """Diagnostic: how much of the targets' variance the critic explains.

    ``1 - Var[target - prediction] / Var[target]`` (population variance):
    1 is a perfect fit, 0 matches a constant mean predictor, negative is
    worse than that. Purely observational — never part of any loss.

    Returns:
        The scalar diagnostic; ``nan`` when the targets have zero variance
        (the ratio is undefined there).
    """
    with torch.no_grad():
        target_var = float(value_targets.var(correction=0).item())
        if target_var == 0.0:
            return float("nan")
        residual_var = float((value_targets - values).var(correction=0).item())
        return 1.0 - residual_var / target_var


def approx_kl_divergence(new_log_prob: Tensor, old_log_prob: Tensor) -> Tensor:
    """Monitor-grade approximation of ``KL(old || new)`` from sampled actions.

    Uses the low-variance, non-negative ``k3`` estimator
    ``E[(rho - 1) - log rho]`` (Schulman, "Approximating KL Divergence").
    Intended for monitoring and optional epoch early-stopping, not as a
    trained objective; computed under ``no_grad``.
    """
    with torch.no_grad():
        log_ratio = new_log_prob - old_log_prob
        return (log_ratio.exp() - 1.0 - log_ratio).mean()

"""SAC networks: squashed-Gaussian actor and continuous-action critics.

The actor's log-probability implements the exact tanh change of variables.
For u ~ N(mu, sigma) and a = tanh(u), scaled affinely to the env bounds:

    log pi(a_env | s) = log N(u; mu, sigma)
                        - sum_i log(1 - tanh(u_i)^2)      (tanh Jacobian)
                        - sum_i log(scale_i)              (affine Jacobian)

with the numerically stable identity
``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))`` used instead of the
naive form, which underflows for |u| beyond ~20. A unit test checks the
whole computation against torch.distributions' independent
TransformedDistribution implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812  (torch's own convention)

from rlcore.nets import build_mlp

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def tanh_squash_log_jacobian(pre_tanh: Tensor) -> Tensor:
    """``log(1 - tanh(u)^2)`` per element, numerically stable for large |u|."""
    return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))


class SquashedGaussianActor(nn.Module):
    """Gaussian policy with tanh squashing and affine scaling to env bounds.

    Args:
        obs_dim: Flat observation size.
        act_dim: Action-vector size.
        action_low: Per-dimension lower bounds, shape ``[act_dim]``.
        action_high: Per-dimension upper bounds, shape ``[act_dim]``.
        hidden_sizes: Trunk hidden widths.

    Raises:
        ValueError: If any bound pair is not strictly ordered and finite.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        """Build trunk and mean/log-std heads; register bound buffers."""
        super().__init__()
        if not bool(torch.isfinite(action_low).all() and torch.isfinite(action_high).all()):
            raise ValueError("SAC requires finite action bounds.")
        if not bool((action_high > action_low).all()):
            raise ValueError("action_high must exceed action_low in every dimension.")
        self.net = build_mlp(obs_dim, hidden_sizes, 2 * act_dim, final_gain=0.01)
        self.act_dim = act_dim
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self.action_scale: Tensor
        self.action_bias: Tensor

    def _mu_log_std(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        out: Tensor = self.net(obs)
        mu, log_std = out.chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(
        self, obs: Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Reparameterized sample with exact log-probability.

        Args:
            obs: Observations, ``[batch, obs_dim]``.
            generator: Explicit RNG stream for the standard-normal draw.

        Returns:
            ``(action, log_prob)`` — env-scaled actions ``[batch, act_dim]``
            (differentiable through the reparameterization) and per-sample
            log-probabilities ``[batch]`` under the squashed, scaled
            density.
        """
        mu, log_std = self._mu_log_std(obs)
        std = log_std.exp()
        noise_device = generator.device if generator is not None else mu.device
        noise = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=noise_device).to(
            mu.device
        )
        pre_tanh = mu + std * noise
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        gaussian_log_prob = (
            -0.5 * ((pre_tanh - mu) / std) ** 2 - log_std - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_prob = (
            gaussian_log_prob
            - tanh_squash_log_jacobian(pre_tanh).sum(dim=-1)
            - torch.log(self.action_scale).sum()
        )
        return action, log_prob

    @torch.no_grad()
    def deterministic_action(self, obs: Tensor) -> Tensor:
        """Mean action (tanh of mu), env-scaled; used for greedy evaluation."""
        mu, _ = self._mu_log_std(obs)
        return torch.tanh(mu) * self.action_scale + self.action_bias


class ContinuousQNetwork(nn.Module):
    """Q(s, a) over concatenated observation and action vectors."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int] = (256, 256)
    ) -> None:
        """Build the network."""
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, hidden_sizes, 1, final_gain=1.0)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Return Q-values of shape ``[batch]``."""
        out: Tensor = self.net(torch.cat([obs, action], dim=-1))
        return out.squeeze(-1)

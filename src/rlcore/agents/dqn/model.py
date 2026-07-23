"""Q-network and epsilon-greedy action selection for DQN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from rlcore.nets import build_mlp

if TYPE_CHECKING:
    from collections.abc import Sequence


class QNetwork(nn.Module):
    """MLP mapping flat observations to per-action Q-values.

    Output-layer orthogonal gain 1.0 (value scale), hidden gain sqrt(2) —
    see :func:`rlcore.nets.build_mlp`.

    Args:
        obs_dim: Size of the flat observation vector.
        n_actions: Number of discrete actions.
        hidden_sizes: Hidden-layer widths.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (128, 128)
    ) -> None:
        """Build the network."""
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, n_actions, final_gain=1.0)
        self.n_actions = n_actions

    def forward(self, obs: Tensor) -> Tensor:
        """Return Q-values of shape ``[batch, n_actions]``."""
        out: Tensor = self.net(obs)
        return out

    @torch.no_grad()
    def greedy_action(self, obs: Tensor) -> Tensor:
        """Return argmax-Q actions of shape ``[batch]`` (int64), no graph."""
        return self.forward(obs).argmax(dim=-1)


@torch.no_grad()
def epsilon_greedy_action(
    q_net: QNetwork,
    obs: Tensor,
    *,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Epsilon-greedy selection over a batch of observations.

    With probability ``epsilon`` (per row) a uniform random action is taken;
    otherwise the greedy action. All randomness flows through ``generator``.

    Args:
        q_net: The online Q-network.
        obs: Observations, ``[batch, obs_dim]``.
        epsilon: Exploration probability in ``[0, 1]``.
        generator: Explicit RNG stream.

    Returns:
        Actions, ``[batch]`` int64.

    Raises:
        ValueError: If ``epsilon`` is outside ``[0, 1]``.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
    greedy = q_net.greedy_action(obs)
    if epsilon == 0.0:
        return greedy
    batch = obs.shape[0]
    explore = torch.rand(batch, generator=generator) < epsilon
    random_actions = torch.randint(0, q_net.n_actions, (batch,), generator=generator)
    return torch.where(explore, random_actions, greedy)

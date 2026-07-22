"""Concrete categorical MLP policy for discrete action spaces.

Every operation is kept visible — explicit ``log_softmax``, ``gather``, and
``multinomial`` — rather than routed through ``torch.distributions`` or a
shared network library; those arrive at M3 by extraction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence


class CategoricalMlpPolicy(nn.Module):
    """An MLP mapping flat observations to action logits.

    Hidden layers use tanh activations. Weights use orthogonal initialization
    with gain sqrt(2) for hidden layers and 0.01 for the output layer, the
    empirically supported default for on-policy policy gradients
    (Engstrom et al., ICLR 2020; Huang et al., ICLR Blog Track 2022).

    Args:
        obs_dim: Size of the flat observation vector.
        n_actions: Number of discrete actions.
        hidden_sizes: Widths of the hidden layers; empty means a single
            linear layer from observations to logits.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (64, 64)
    ) -> None:
        """Build the network and apply orthogonal initialization."""
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.Tanh()]
            in_dim = size
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, obs: Tensor) -> Tensor:
        """Return action logits of shape ``[batch, n_actions]``."""
        logits: Tensor = self.net(obs)
        return logits

    def log_probs(self, obs: Tensor) -> Tensor:
        """Return log pi(. | obs) of shape ``[batch, n_actions]``."""
        return self.forward(obs).log_softmax(dim=-1)

    def action_log_prob(self, obs: Tensor, action: Tensor) -> Tensor:
        """Return log pi(action | obs) of shape ``[batch]``, differentiable.

        Args:
            obs: Observations, ``[batch, obs_dim]``.
            action: Taken actions, ``[batch]`` int64.
        """
        return self.log_probs(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    def entropy(self, obs: Tensor) -> Tensor:
        """Return the policy entropy per observation, shape ``[batch]``."""
        log_probs = self.log_probs(obs)
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> Tensor:
        """Select actions for a batch of observations, shape ``[batch]`` int64.

        Args:
            obs: Observations, ``[batch, obs_dim]``.
            generator: Explicit RNG stream for sampling; falls back to the
                global torch generator when ``None``.
            deterministic: Take the argmax action instead of sampling.
        """
        logits = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

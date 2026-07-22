"""Concrete categorical MLP policy for discrete action spaces.

Network construction and log-prob/entropy math are shared via
:mod:`rlcore.nets` (M3 extraction). Action *sampling* stays local: this
policy samples from ``softmax(logits)`` while PPO samples from
``exp(log_softmax(logits))`` — equal in exact arithmetic but not bitwise,
so unifying them would alter seeded RNG-draw streams (see the audit,
``docs/design/m3-consolidation-audit.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from rlcore.nets import build_mlp, entropy_from_log_probs, gather_log_prob

if TYPE_CHECKING:
    from collections.abc import Sequence


class CategoricalMlpPolicy(nn.Module):
    """An MLP mapping flat observations to action logits.

    Tanh hidden layers; orthogonal initialization with gain sqrt(2) hidden
    and 0.01 output (see :func:`rlcore.nets.build_mlp` for provenance).

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
        self.net = build_mlp(obs_dim, hidden_sizes, n_actions, final_gain=0.01)

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
        return gather_log_prob(self.log_probs(obs), action)

    def entropy(self, obs: Tensor) -> Tensor:
        """Return the policy entropy per observation, shape ``[batch]``."""
        return entropy_from_log_probs(self.log_probs(obs))

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

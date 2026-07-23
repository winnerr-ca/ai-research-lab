"""Actor-critic model: separate policy and value MLPs over flat observations.

Separate networks (no shared torso) keep the two losses from interfering
through shared features — the simplest correct default, and the one whose
behavior is easiest to reason about. As in M1, every operation stays
explicit (``log_softmax``/``gather``/``multinomial``); the shared network
library arrives at M3 by extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from rlcore.nets import build_mlp, entropy_from_log_probs, gather_log_prob

if TYPE_CHECKING:
    from collections.abc import Sequence


class ActorCritic(nn.Module):
    """Categorical policy head and state-value head as separate MLPs.

    Initialization follows the on-policy defaults with empirical support
    (Engstrom et al., ICLR 2020; Huang et al., ICLR Blog Track 2022):
    orthogonal, hidden gain sqrt(2), policy output gain 0.01 (near-uniform
    initial policy), value output gain 1.0.

    Args:
        obs_dim: Size of the flat observation vector.
        n_actions: Number of discrete actions.
        hidden_sizes: Hidden-layer widths used by both networks.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (64, 64)
    ) -> None:
        """Build policy and value networks with orthogonal initialization."""
        super().__init__()
        self.policy_net = build_mlp(obs_dim, hidden_sizes, n_actions, final_gain=0.01)
        self.value_net = build_mlp(obs_dim, hidden_sizes, 1, final_gain=1.0)

    def logits(self, obs: Tensor) -> Tensor:
        """Return action logits of shape ``[batch, n_actions]``."""
        out: Tensor = self.policy_net(obs)
        return out

    def value(self, obs: Tensor) -> Tensor:
        """Return state-value estimates of shape ``[batch]``."""
        out: Tensor = self.value_net(obs)
        return out.squeeze(-1)

    def evaluate_actions(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return differentiable ``(log_prob, entropy, value)``, each ``[batch]``.

        Args:
            obs: Observations, ``[batch, obs_dim]``.
            action: Taken actions, ``[batch]`` int64.
        """
        log_probs = self.logits(obs).log_softmax(dim=-1)
        log_prob = gather_log_prob(log_probs, action)
        entropy = entropy_from_log_probs(log_probs)
        return log_prob, entropy, self.value(obs)

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select actions without building a graph.

        Args:
            obs: Observations, ``[batch, obs_dim]``.
            generator: Explicit RNG stream for sampling; global torch RNG
                when ``None``.
            deterministic: Take the argmax action instead of sampling.

        Returns:
            ``(action, log_prob, value)`` with shapes ``[batch]`` — the
            behavior log-probability and value estimate recorded for later
            (importance-ratio and GAE) use.
        """
        log_probs = self.logits(obs).log_softmax(dim=-1)
        if deterministic:
            action = log_probs.argmax(dim=-1)
        else:
            action = (
                torch.multinomial(log_probs.exp().cpu(), num_samples=1, generator=generator)
                .squeeze(-1)
                .to(log_probs.device)
            )
        return action, gather_log_prob(log_probs, action), self.value(obs)

"""Small shared network helpers, extracted at M3 from REINFORCE and PPO.

Scope is deliberately narrow (see ``docs/design/m3-consolidation-audit.md``):
the MLP builder and the two categorical log-prob/entropy helpers whose
operation sequences were byte-for-byte identical in both algorithms. Action
*sampling* is intentionally NOT unified: REINFORCE samples from
``softmax(logits)`` and PPO from ``exp(log_softmax(logits))`` — equal in
exact arithmetic but not bitwise in floats, so unifying them would silently
alter the algorithms' RNG-draw streams and break run reproduction. That
unification waits for a milestone with a scheduled re-validation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_mlp(
    in_dim: int, hidden_sizes: Sequence[int], out_dim: int, *, final_gain: float
) -> nn.Sequential:
    """Build a tanh MLP with orthogonal initialization.

    Hidden layers use gain sqrt(2), the output layer ``final_gain`` — the
    empirically supported on-policy defaults (Engstrom et al., ICLR 2020;
    Huang et al., ICLR Blog Track 2022): 0.01 for policy heads (near-uniform
    initial policy), 1.0 for value heads. Layer construction and
    initialization order are part of this function's contract: they fix the
    global-RNG draw sequence, and reordering them would silently change
    seeded network initializations platform-wide.

    Args:
        in_dim: Input feature size.
        hidden_sizes: Hidden-layer widths; empty means a single linear layer.
        out_dim: Output size.
        final_gain: Orthogonal gain for the output layer.
    """
    layers: list[nn.Module] = []
    for size in hidden_sizes:
        layers += [nn.Linear(in_dim, size), nn.Tanh()]
        in_dim = size
    layers.append(nn.Linear(in_dim, out_dim))
    linear_layers = [m for m in layers if isinstance(m, nn.Linear)]
    for layer in linear_layers[:-1]:
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(linear_layers[-1].weight, gain=final_gain)
    nn.init.zeros_(linear_layers[-1].bias)
    return nn.Sequential(*layers)


def gather_log_prob(log_probs: Tensor, action: Tensor) -> Tensor:
    """Select ``log pi(action | s)`` from full log-probabilities.

    Args:
        log_probs: ``[batch, n_actions]`` log-probabilities (from
            ``log_softmax``).
        action: ``[batch]`` int64 actions.

    Returns:
        ``[batch]`` selected log-probabilities; differentiable if the input
        is.
    """
    return log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


def entropy_from_log_probs(log_probs: Tensor) -> Tensor:
    """Categorical entropy per row of ``[batch, n_actions]`` log-probabilities.

    Returns:
        ``[batch]`` entropies; differentiable if the input is.
    """
    return -(log_probs.exp() * log_probs).sum(dim=-1)

"""Policy evaluation, extracted at M3 from near-identical REINFORCE/PPO code.

The two algorithms' actors have different ``act`` signatures (REINFORCE
returns an action tensor, PPO's actor-critic returns a triple), so the
extraction point is a **callable**, not a policy interface: the caller wraps
its model, determinism choice, and RNG stream into ``select_action``, and
this module owns only the episode loop and statistics. This is deliberately
the smallest interface that removes the duplication (see
``docs/design/m3-consolidation-audit.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class EvalStats:
    """Aggregate statistics over evaluation episodes.

    Attributes:
        episode_returns: Undiscounted return of each evaluation episode.
        episode_lengths: Length (steps) of each evaluation episode.
    """

    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]

    @property
    def mean_return(self) -> float:
        """Mean undiscounted episode return."""
        return sum(self.episode_returns) / len(self.episode_returns)

    @property
    def std_return(self) -> float:
        """Population standard deviation of episode returns."""
        mean = self.mean_return
        return math.sqrt(
            sum((r - mean) ** 2 for r in self.episode_returns) / len(self.episode_returns)
        )

    @property
    def min_return(self) -> float:
        """Smallest episode return."""
        return min(self.episode_returns)

    @property
    def max_return(self) -> float:
        """Largest episode return."""
        return max(self.episode_returns)


def evaluate_policy(
    env: gym.Env[Any, Any],
    select_action: Callable[[Tensor], Tensor],
    *,
    episodes: int,
) -> EvalStats:
    """Run full evaluation episodes and report undiscounted returns.

    Parameters are only read, never written; use a dedicated env instance so
    evaluation never perturbs training env RNG streams. Greedy vs.
    stochastic behavior (and any RNG stream) lives inside ``select_action``::

        # greedy REINFORCE                      # stochastic PPO
        lambda obs: policy.act(obs,             lambda obs: model.act(obs,
            deterministic=True)                     generator=g)[0]

    Args:
        env: Dedicated evaluation env.
        select_action: Maps a ``[1, obs_dim]`` float32 observation to a
            ``[1]`` int64 action.
        episodes: Number of episodes, >= 1.

    Raises:
        ValueError: If ``episodes < 1``.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}.")
    returns: list[float] = []
    lengths: list[int] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_return, length = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs, reward, terminated, truncated, _ = env.step(int(select_action(obs_t).item()))
            episode_return += float(reward)
            length += 1
        returns.append(episode_return)
        lengths.append(length)
    return EvalStats(episode_returns=tuple(returns), episode_lengths=tuple(lengths))

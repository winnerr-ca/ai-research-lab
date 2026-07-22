"""Single-environment episode collection and evaluation.

Deliberately minimal: one Gymnasium env, sequential episodes, no
vectorization and no collector abstraction (excluded from M1 by design;
extracted at M3). Collection runs the policy under ``no_grad`` and stores
observations/actions; log-probabilities are recomputed differentiably at
update time from the same (not-yet-updated) network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from rlcore.types import Batch

if TYPE_CHECKING:
    import gymnasium as gym

    from rlcore.agents.reinforce.policy import CategoricalMlpPolicy


def collect_episode(
    env: gym.Env[Any, Any],
    policy: CategoricalMlpPolicy,
    *,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
) -> Batch:
    """Run one full episode and return its transitions as a :class:`Batch`.

    The episode ends when the env reports either ``terminated`` (an MDP-true
    absorbing state) or ``truncated`` (a time-limit cutoff); both flags are
    recorded per step so downstream code can distinguish them.

    The env is reset at the start of each call *without* a seed: seed the
    env's RNG once at setup (``env.reset(seed=...)``) and let subsequent
    resets continue that stream, per Gymnasium semantics.

    Args:
        env: A Gymnasium env with flat float observations and discrete
            actions.
        policy: The policy to act with (queried under ``no_grad``).
        generator: Explicit RNG stream for action sampling.
        deterministic: Act greedily instead of sampling (evaluation mode).

    Returns:
        A batch with per-step fields ``obs`` (float32), ``action`` (int64),
        ``reward`` (float32), ``terminated`` and ``truncated`` (bool). All
        flags are ``False`` except at most the last step's.
    """
    obs_steps: list[torch.Tensor] = []
    actions: list[int] = []
    rewards: list[float] = []
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        action = int(
            policy.act(obs_t.unsqueeze(0), generator=generator, deterministic=deterministic).item()
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_steps.append(obs_t)
        actions.append(action)
        rewards.append(float(reward))

    length = len(rewards)
    terminated_flags = torch.zeros(length, dtype=torch.bool)
    truncated_flags = torch.zeros(length, dtype=torch.bool)
    terminated_flags[-1] = terminated
    truncated_flags[-1] = truncated
    return Batch(
        obs=torch.stack(obs_steps),
        action=torch.as_tensor(actions, dtype=torch.int64),
        reward=torch.as_tensor(rewards, dtype=torch.float32),
        terminated=terminated_flags,
        truncated=truncated_flags,
    )


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


def evaluate(
    env: gym.Env[Any, Any],
    policy: CategoricalMlpPolicy,
    *,
    episodes: int,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
) -> EvalStats:
    """Run evaluation episodes and report undiscounted returns.

    Evaluation only reads the policy — parameters are untouched — and should
    use a dedicated env instance (with its own seed stream) so it never
    perturbs the training env's RNG. Greedy (``deterministic=True``) and
    stochastic evaluation answer different questions; the M1 validation
    report shows both.

    Args:
        env: Dedicated evaluation env.
        policy: Policy to evaluate.
        episodes: Number of episodes, >= 1.
        deterministic: Greedy actions (no RNG consumed) if ``True``.
        generator: RNG stream for stochastic evaluation.

    Raises:
        ValueError: If ``episodes < 1``.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}.")
    collected = [
        collect_episode(env, policy, generator=generator, deterministic=deterministic)
        for _ in range(episodes)
    ]
    return EvalStats(
        episode_returns=tuple(float(ep.reward.sum().item()) for ep in collected),
        episode_lengths=tuple(len(ep) for ep in collected),
    )

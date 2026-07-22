"""Single-environment episode collection.

Deliberately minimal: one Gymnasium env, sequential episodes, no
vectorization and no collector abstraction (the M3 audit kept episodic
collection local; evaluation moved to :mod:`rlcore.evaluation`). Collection
runs the policy under ``no_grad`` and stores observations/actions;
log-probabilities are recomputed differentiably at update time from the
same (not-yet-updated) network.
"""

from __future__ import annotations

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

"""Numerical parity: our GAE vs. Stable-Baselines3's, on identical inputs.

Scope, chosen deliberately: SB3's ``RolloutBuffer.compute_returns_and_advantage``
is an isolated, callable implementation of the same recursion, so comparing
on identical synthetic inputs is a genuine like-for-like test. The
comparison is restricted to termination-only sequences because SB3's buffer
never represents truncation (SB3 folds the truncation bootstrap into
rewards upstream of the buffer); our truncation handling is therefore
verified against hand-computed references in ``tests/unit/test_ppo_gae.py``,
not against SB3.

Input mapping between the two conventions:

- SB3 stores ``episode_start[t]`` (= done at t-1); ours stores ``done[t]``.
- SB3 derives next-state values internally from ``values[t+1]`` and the
  final ``last_values``; ours takes an explicit ``next_values`` array.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlcore.agents.ppo.gae import compute_gae

sb3_buffers = pytest.importorskip("stable_baselines3.common.buffers")

GAMMA = 0.98
LAM = 0.92


def build_case(
    horizon: int, done_steps: list[int], seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Random rewards/values with terminations at ``done_steps``."""
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.rand(horizon, generator=generator)
    values = torch.rand(horizon, generator=generator)
    last_value = float(torch.rand(1, generator=generator).item())
    dones = torch.zeros(horizon, dtype=torch.bool)
    dones[done_steps] = True
    next_values = torch.empty(horizon)
    for t in range(horizon):
        if bool(dones[t]):
            next_values[t] = 0.0
        elif t + 1 < horizon:
            next_values[t] = values[t + 1]
        else:
            next_values[t] = last_value
    return rewards, values, next_values, dones, last_value


def sb3_advantages(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run SB3's buffer computation on the equivalent inputs."""
    import gymnasium as gym

    horizon = len(rewards)
    buffer = sb3_buffers.RolloutBuffer(
        buffer_size=horizon,
        observation_space=gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Discrete(2),
        device="cpu",
        gamma=GAMMA,
        gae_lambda=LAM,
        n_envs=1,
    )
    obs = np.zeros((1, 1), dtype=np.float32)
    for t in range(horizon):
        episode_start = np.array([bool(dones[t - 1])]) if t > 0 else np.array([False])
        buffer.add(
            obs,
            np.array([[0]]),
            np.array([float(rewards[t])]),
            episode_start,
            torch.tensor([float(values[t])]),
            torch.tensor([0.0]),
        )
    buffer.compute_returns_and_advantage(
        last_values=torch.tensor([last_value]), dones=np.array([bool(dones[-1])])
    )
    return buffer.advantages.flatten(), buffer.returns.flatten()


@pytest.mark.parametrize(
    ("horizon", "done_steps", "seed"),
    [
        (16, [], 0),  # single ongoing segment, tail bootstrap
        (16, [5, 11], 1),  # two mid-rollout terminations
        (16, [15], 2),  # termination exactly at the horizon end
        (8, [0], 3),  # termination at the first step
    ],
)
def test_gae_matches_sb3(horizon: int, done_steps: list[int], seed: int) -> None:
    rewards, values, next_values, dones, last_value = build_case(horizon, done_steps, seed)
    ours_adv, ours_ret = compute_gae(rewards, values, next_values, dones, gamma=GAMMA, lam=LAM)
    theirs_adv, theirs_ret = sb3_advantages(rewards, values, dones, last_value)
    assert np.allclose(ours_adv.numpy(), theirs_adv, atol=1e-5)
    assert np.allclose(ours_ret.numpy(), theirs_ret, atol=1e-5)

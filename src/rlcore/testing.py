"""Deterministic test-support components.

Exact tests need environments whose rewards, lengths, and termination mode
are known in advance. :class:`ScriptedEnv` provides that; it is part of the
package (rather than the test tree) so later milestones' tests — and
downstream users testing their own components — can reuse it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


class ScriptedEnv(gym.Env[NDArray[np.float32], int]):
    """A deterministic env that plays out a fixed reward sequence.

    Actions are recorded but do not influence anything. The observation at
    step ``t`` is a float32 vector of shape ``(obs_dim,)`` filled with ``t``,
    so tests can verify exactly which observation was stored with which step.
    The episode ends after ``len(rewards)`` steps, flagged as *terminated*
    when ``terminate=True`` and as *truncated* otherwise — letting tests pin
    down termination/truncation handling exactly.

    Attributes:
        received_actions: Every action passed to :meth:`step` since
            construction, for asserting what a collector sent.
    """

    def __init__(
        self,
        rewards: Sequence[float],
        *,
        terminate: bool = True,
        obs_dim: int = 2,
        n_actions: int = 2,
    ) -> None:
        """Configure the scripted episode.

        Args:
            rewards: Reward emitted at each step; also fixes episode length.
            terminate: End the episode as ``terminated`` (vs. ``truncated``).
            obs_dim: Observation vector size.
            n_actions: Size of the (ignored) discrete action space.

        Raises:
            ValueError: If ``rewards`` is empty.
        """
        if len(rewards) == 0:
            raise ValueError("ScriptedEnv needs at least one reward (one step).")
        super().__init__()
        self._rewards = [float(r) for r in rewards]
        self._terminate = terminate
        self._step_index = 0
        self.received_actions: list[int] = []
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

    def _obs(self) -> NDArray[np.float32]:
        shape = self.observation_space.shape
        assert shape is not None
        return np.full(shape, float(self._step_index), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Restart the scripted episode; ``seed`` is accepted and irrelevant."""
        super().reset(seed=seed)
        self._step_index = 0
        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance one step, emitting the scripted reward for this step."""
        self.received_actions.append(int(action))
        reward = self._rewards[self._step_index]
        self._step_index += 1
        done = self._step_index >= len(self._rewards)
        terminated = done and self._terminate
        truncated = done and not self._terminate
        return self._obs(), reward, terminated, truncated, {}

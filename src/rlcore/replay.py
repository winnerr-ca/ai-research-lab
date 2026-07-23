"""Uniform experience replay for off-policy algorithms.

Introduced at M5 with DQN as its first consumer and SAC (M6) as the
scheduled second — the rule of two, satisfied by schedule. Transitions are
stored in preallocated NumPy ring buffers; sampling is uniform and driven
by an explicit torch generator so replay order is part of the platform's
seed-controlled state.

Termination semantics (the platform contract, ARCHITECTURE §5): a stored
transition's ``next_obs`` is always the *true successor observation* —
at truncation that is the episode's final observation (bootstrap from it,
per the stated continuing-task objective), and ``terminated`` marks only
MDP-true endings (bootstrap suppressed). Collectors never store a
transition that crosses a reset: after any boundary, the next stored
transition starts from the reset observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlcore.types import Batch


class ReplayBuffer:
    """A uniform ring buffer of 1-step transitions.

    Args:
        capacity: Maximum stored transitions; the oldest are overwritten.
        obs_shape: Observation shape (flat vectors today).
        action_shape: Action shape; ``()`` for scalar discrete actions.
        action_dtype: ``np.int64`` for discrete, ``np.float32`` for
            continuous.

    Raises:
        ValueError: If ``capacity < 1``.
    """

    def __init__(
        self,
        capacity: int,
        *,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        action_dtype: type = np.int64,
    ) -> None:
        """Preallocate storage."""
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        self.capacity = capacity
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._action: np.ndarray = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.bool_)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int | float | np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition, overwriting the oldest at capacity.

        ``next_obs`` must be the true successor observation (the final
        observation at truncation, never a reset observation); ``terminated``
        must be ``True`` only for MDP-true endings.
        """
        self._obs[self._pos] = obs
        self._action[self._pos] = action
        self._reward[self._pos] = reward
        self._next_obs[self._pos] = next_obs
        self._terminated[self._pos] = terminated
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> Batch:
        """Sample a uniform batch of transitions.

        Args:
            batch_size: Transitions to draw (with replacement), >= 1.
            generator: Explicit RNG stream; sampling is deterministic given
                the generator state and buffer contents.

        Returns:
            A :class:`Batch` with fields ``obs``, ``action``, ``reward``,
            ``next_obs``, ``terminated``.

        Raises:
            ValueError: If the buffer is empty or ``batch_size < 1``.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = torch.randint(0, self._size, (batch_size,), generator=generator).numpy()
        return Batch(
            obs=torch.as_tensor(self._obs[indices]),
            action=torch.as_tensor(self._action[indices]),
            reward=torch.as_tensor(self._reward[indices]),
            next_obs=torch.as_tensor(self._next_obs[indices]),
            terminated=torch.as_tensor(self._terminated[indices]),
        )

    def state(self) -> dict[str, Any]:
        """Snapshot the full buffer contents for checkpointing."""
        return {
            "obs": self._obs.copy(),
            "action": self._action.copy(),
            "reward": self._reward.copy(),
            "next_obs": self._next_obs.copy(),
            "terminated": self._terminated.copy(),
            "pos": self._pos,
            "size": self._size,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by :meth:`state`.

        Raises:
            ValueError: If the snapshot's capacity differs from this
                buffer's.
        """
        if state["obs"].shape[0] != self.capacity:
            raise ValueError(
                f"Snapshot capacity {state['obs'].shape[0]} != buffer capacity {self.capacity}."
            )
        self._obs[:] = state["obs"]
        self._action[:] = state["action"]
        self._reward[:] = state["reward"]
        self._next_obs[:] = state["next_obs"]
        self._terminated[:] = state["terminated"]
        self._pos = int(state["pos"])
        self._size = int(state["size"])

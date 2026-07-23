"""Vectorized fixed-horizon rollout collection for PPO.

Uses Gymnasium's ``SyncVectorEnv`` in ``SAME_STEP`` autoreset mode: when an
episode ends at step *t*, ``step()`` returns the **reset** observation for
that env while the episode's true **final** observation arrives in
``info["final_obs"]``. That is precisely what the platform's bootstrap
contract needs:

- terminated  -> ``next_value = 0``
- truncated   -> ``next_value = V(final_obs)`` (from ``info["final_obs"]``)
- otherwise   -> ``next_value = V(s_{t+1})``

and the advantage recursion breaks at both boundary kinds because the
next stored row belongs to a reset episode. GAE is computed **per env
column** before flattening, so no temporal chain ever crosses environments
or resets.

Deterministic child seeding: training env *i* of *n* receives child seed
``10 + i`` of the run's root seed (documented extension of the layout in
:mod:`rlcore.envs`; children 0-2 remain the single-env layout).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from rlcore.types import Batch

if TYPE_CHECKING:
    from rlcore.agents.ppo.model import ActorCritic
    from rlcore.utils.seeding import Rng

VECTOR_ENV_CHILD_SEED_BASE = 10


def make_vector_train_env(env_id: str, n_envs: int, rng: Rng) -> SyncVectorEnv:
    """Create a SAME_STEP synchronous vector env with deterministic seeding.

    Args:
        env_id: Gymnasium env id (flat Box obs, Discrete actions — same
            requirements as the single-env PPO path).
        n_envs: Number of parallel envs, >= 2.
        rng: The run's seed bundle; env *i* consumes child seed ``10 + i``.

    Raises:
        ValueError: If ``n_envs < 2`` (use the single-env collector).
    """
    if n_envs < 2:
        raise ValueError(f"n_envs must be >= 2 for the vector collector, got {n_envs}.")
    # functools.partial (not a lambda) keeps the env fns picklable, which is
    # what makes exact vector-mode checkpoint resume possible (pickle
    # fidelity across autoresets is covered by tests).
    env = SyncVectorEnv(
        [functools.partial(gym.make, env_id) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env.reset(seed=[rng.child_seed(VECTOR_ENV_CHILD_SEED_BASE + i) for i in range(n_envs)])
    return env


@dataclass(frozen=True, slots=True)
class VectorRollout:
    """One vectorized fixed-horizon collection result.

    Attributes:
        batch: Flattened ``[n_steps * n_envs]`` transition fields (env-major
            within each step; flattening order is column-consistent with
            the per-env GAE computed by the trainer).
        per_env: Per-env-column views used for GAE: each field is
            ``[n_steps, n_envs]``.
        completed_returns: Undiscounted returns of episodes finished during
            this rollout (all envs pooled; for logging).
    """

    batch: Batch
    per_env: dict[str, torch.Tensor]
    completed_returns: tuple[float, ...]


class VectorRolloutCollector:
    """Steps a SAME_STEP vector env with the current model, statefully.

    The vector env must be created by :func:`make_vector_train_env` (or
    seeded equivalently) before the first collect.
    """

    def __init__(self, env: SyncVectorEnv) -> None:
        """Wrap ``env``; the first :meth:`collect` performs the reset."""
        self._env = env
        self._obs: np.ndarray | None = None
        self._episode_returns = np.zeros(env.num_envs, dtype=np.float64)

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return int(self._env.num_envs)

    def state(self) -> dict[str, Any]:
        """Snapshot for checkpointing: current obs and running returns."""
        return {
            "obs": None if self._obs is None else self._obs.copy(),
            "episode_returns": self._episode_returns.copy(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by :meth:`state`."""
        self._obs = None if state["obs"] is None else np.asarray(state["obs"])
        self._episode_returns = np.asarray(state["episode_returns"], dtype=np.float64)

    def collect(
        self,
        model: ActorCritic,
        n_steps: int,
        *,
        generator: torch.Generator | None = None,
    ) -> VectorRollout:
        """Collect ``n_steps`` transitions from every env (``n_steps * n_envs`` total).

        Actions for all envs are sampled in one batched forward pass per
        step. Behavior log-probabilities and value estimates are recorded;
        boundary bootstrap values follow the module-level contract.

        Raises:
            ValueError: If ``n_steps < 1``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if self._obs is None:
            reset_out: tuple[Any, dict[str, Any]] = self._env.reset()
            self._obs = np.asarray(reset_out[0])

        n_envs = self.n_envs
        # Rollout storage stays on CPU; the model may live on another device.
        model_device = next(model.parameters()).device
        obs_steps = torch.zeros((n_steps, n_envs, self._obs.shape[-1]), dtype=torch.float32)
        actions = torch.zeros((n_steps, n_envs), dtype=torch.int64)
        rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        terminateds = torch.zeros((n_steps, n_envs), dtype=torch.bool)
        truncateds = torch.zeros((n_steps, n_envs), dtype=torch.bool)
        logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        values = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        # Bootstrap values known at collection time (boundaries); the rest
        # are filled from values[t+1] / the tail below.
        next_values = torch.zeros((n_steps, n_envs), dtype=torch.float32)
        boundary = torch.zeros((n_steps, n_envs), dtype=torch.bool)
        completed: list[float] = []

        for t in range(n_steps):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=model_device)
            action_t, logprob_t, value_t = model.act(obs_t, generator=generator)
            step_out: tuple[Any, Any, Any, Any, dict[str, Any]] = self._env.step(
                action_t.cpu().numpy()
            )
            next_obs, reward, terminated, truncated, info = step_out
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            obs_steps[t] = obs_t.cpu()
            actions[t] = action_t.cpu()
            rewards[t] = torch.as_tensor(reward, dtype=torch.float32)
            terminateds[t] = torch.as_tensor(terminated)
            truncateds[t] = torch.as_tensor(truncated)
            logprobs[t] = logprob_t.cpu()
            values[t] = value_t.cpu()

            done = terminated | truncated
            if done.any():
                boundary[t] = torch.as_tensor(done)
                final_obs = info["final_obs"]
                truncated_indices = np.flatnonzero(truncated)
                if truncated_indices.size:
                    finals = torch.as_tensor(
                        np.stack([final_obs[i] for i in truncated_indices]),
                        dtype=torch.float32,
                        device=model_device,
                    )
                    with torch.no_grad():
                        final_values = model.value(finals)
                    next_values[t, torch.as_tensor(truncated_indices)] = final_values.cpu()
                # terminated columns keep next_value = 0.
                self._episode_returns += reward
                for i in np.flatnonzero(done):
                    completed.append(float(self._episode_returns[i]))
                    self._episode_returns[i] = 0.0
            else:
                self._episode_returns += reward
            self._obs = np.asarray(next_obs)

        # Non-boundary steps: V(s_{t+1}); the horizon tail: V(current obs).
        with torch.no_grad():
            tail_values = model.value(
                torch.as_tensor(self._obs, dtype=torch.float32, device=model_device)
            ).cpu()
        for t in range(n_steps):
            following = values[t + 1] if t + 1 < n_steps else tail_values
            next_values[t] = torch.where(boundary[t], next_values[t], following)

        per_env = {
            "reward": rewards,
            "value": values,
            "next_value": next_values,
            "done": terminateds | truncateds,
        }
        batch = Batch(
            obs=obs_steps.reshape(n_steps * n_envs, -1),
            action=actions.reshape(-1),
            reward=rewards.reshape(-1),
            terminated=terminateds.reshape(-1),
            truncated=truncateds.reshape(-1),
            done=(terminateds | truncateds).reshape(-1),
            logprob=logprobs.reshape(-1),
            value=values.reshape(-1),
            next_value=next_values.reshape(-1),
        )
        return VectorRollout(batch=batch, per_env=per_env, completed_returns=tuple(completed))

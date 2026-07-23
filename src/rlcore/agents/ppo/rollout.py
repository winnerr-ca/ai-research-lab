"""Fixed-horizon rollout collection and evaluation for PPO.

PPO collects a fixed number of env steps per update, crossing episode
boundaries; the collector is stateful so partial episodes continue into the
next rollout. It is deliberately PPO-specific and single-env: the M3 audit
kept fixed-horizon collection local (it shares no stable common structure
with REINFORCE's episodic collection). Evaluation moved to
:mod:`rlcore.evaluation` at M3.

The collector also owns the ``next_values`` bootstrap contract consumed by
:func:`rlcore.agents.ppo.gae.compute_gae`: 0 at true terminals, the value of
the episode's final observation at truncations, and the value of the
successor state otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from rlcore.types import Batch

if TYPE_CHECKING:
    import gymnasium as gym

    from rlcore.agents.ppo.model import ActorCritic


@dataclass(frozen=True, slots=True)
class Rollout:
    """One fixed-horizon collection result.

    Attributes:
        batch: Per-step fields ``obs`` (float32), ``action`` (int64),
            ``reward`` (float32), ``terminated``/``truncated``/``done``
            (bool), ``logprob``/``value``/``next_value`` (float32).
        completed_returns: Undiscounted returns of episodes that *finished*
            during this rollout (for logging; may be empty).
    """

    batch: Batch
    completed_returns: tuple[float, ...]


class RolloutCollector:
    """Steps one env with the current model, keeping state across rollouts.

    The env must be seeded by the caller (``env.reset(seed=...)``) before the
    first collect; the collector performs its own unseeded resets thereafter,
    continuing the env's RNG stream per Gymnasium semantics.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Wrap ``env``; the first :meth:`collect` call resets it."""
        self._env = env
        self._obs: Any = None
        self._episode_return = 0.0

    def state(self) -> dict[str, Any]:
        """Snapshot for checkpointing: current obs and running episode return."""
        return {"obs": self._obs, "episode_return": self._episode_return}

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by :meth:`state`.

        Pair the snapshot with the env it was captured alongside.
        """
        self._obs = state["obs"]
        self._episode_return = float(state["episode_return"])

    def collect(
        self,
        model: ActorCritic,
        n_steps: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Rollout:
        """Collect exactly ``n_steps`` transitions under the current policy.

        Actions are sampled (never greedy — PPO's ratios assume the behavior
        policy is the stochastic current policy); behavior log-probabilities
        and value estimates are recorded per step for the update.

        Args:
            model: The current actor-critic (queried under ``no_grad``).
            n_steps: Rollout horizon, >= 1.
            generator: Explicit RNG stream for action sampling.

        Raises:
            ValueError: If ``n_steps < 1``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if self._obs is None:
            self._obs, _ = self._env.reset()

        obs_steps: list[torch.Tensor] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        logprobs: list[float] = []
        values: list[float] = []
        # None marks "successor value not known yet"; filled after the loop.
        next_values: list[float | None] = []
        completed: list[float] = []

        for _ in range(n_steps):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32)
            action_t, logprob_t, value_t = model.act(obs_t.unsqueeze(0), generator=generator)
            action = int(action_t.item())
            next_obs, reward, terminated, truncated, _ = self._env.step(action)

            obs_steps.append(obs_t)
            actions.append(action)
            rewards.append(float(reward))
            terminateds.append(terminated)
            truncateds.append(truncated)
            logprobs.append(float(logprob_t.item()))
            values.append(float(value_t.item()))
            self._episode_return += float(reward)

            if terminated:
                next_values.append(0.0)
            elif truncated:
                final_obs = torch.as_tensor(next_obs, dtype=torch.float32)
                next_values.append(float(model.value(final_obs.unsqueeze(0)).item()))
            else:
                next_values.append(None)

            if terminated or truncated:
                completed.append(self._episode_return)
                self._episode_return = 0.0
                self._obs, _ = self._env.reset()
            else:
                self._obs = next_obs

        # Fill successor values for non-boundary steps: V(s_{t+1}) is the
        # recorded value of step t+1, except at the horizon end, where it is
        # the value of the observation the next rollout will start from.
        with torch.no_grad():
            tail_value = float(
                model.value(torch.as_tensor(self._obs, dtype=torch.float32).unsqueeze(0)).item()
            )
        filled = [
            value if value is not None else (values[t + 1] if t + 1 < n_steps else tail_value)
            for t, value in enumerate(next_values)
        ]

        terminated_t = torch.tensor(terminateds, dtype=torch.bool)
        truncated_t = torch.tensor(truncateds, dtype=torch.bool)
        batch = Batch(
            obs=torch.stack(obs_steps),
            action=torch.as_tensor(actions, dtype=torch.int64),
            reward=torch.as_tensor(rewards, dtype=torch.float32),
            terminated=terminated_t,
            truncated=truncated_t,
            done=terminated_t | truncated_t,
            logprob=torch.as_tensor(logprobs, dtype=torch.float32),
            value=torch.as_tensor(values, dtype=torch.float32),
            next_value=torch.as_tensor(filled, dtype=torch.float32),
        )
        return Rollout(batch=batch, completed_returns=tuple(completed))

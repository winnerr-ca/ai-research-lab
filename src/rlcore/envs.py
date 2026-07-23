"""Environment creation and the platform's seed-stream layout.

Extracted at M3 from identical blocks in the REINFORCE and PPO trainers.
Centralizing this is less about the ~20 saved lines than about the
**seeding protocol**: every algorithm derives the same child-seed layout
from the root seed (child 0 → training env, child 1 → training action
space, child 2 → evaluation env), which keeps cross-algorithm comparisons
on an identical environment randomness footing. Changing this layout
changes every seeded run on the platform; treat it as a compatibility
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from rlcore.utils.seeding import Rng


@dataclass(frozen=True, slots=True)
class EnvPair:
    """A seeded training/evaluation env pair with validated spaces.

    Attributes:
        train_env: The training env (seeded with child 0; action space with
            child 1).
        eval_env: A separate evaluation env (seeded with child 2) so
            evaluation never perturbs training RNG streams.
        obs_dim: Flat observation size.
        n_actions: Number of discrete actions.
    """

    train_env: gym.Env[Any, Any]
    eval_env: gym.Env[Any, Any]
    obs_dim: int
    n_actions: int

    def close(self) -> None:
        """Close both environments."""
        self.train_env.close()
        self.eval_env.close()


def env_pair_from_envs(train_env: gym.Env[Any, Any], eval_env: gym.Env[Any, Any]) -> EnvPair:
    """Build a validated :class:`EnvPair` from existing env instances.

    Used when envs come from somewhere other than ``gym.make`` — most
    importantly checkpoint restoration, where unpickled envs carry their
    own RNG and physical state and must NOT be re-seeded.

    Raises:
        ValueError: If the envs' spaces do not match the requirements.
    """
    obs_space = train_env.observation_space
    act_space = train_env.action_space
    name = getattr(getattr(train_env, "spec", None), "id", type(train_env).__name__)
    if (
        not isinstance(obs_space, spaces.Box)
        or obs_space.shape is None
        or len(obs_space.shape) != 1
    ):
        raise ValueError(f"{name}: requires a flat Box observation space, got {obs_space}.")
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError(f"{name}: requires a Discrete action space, got {act_space}.")
    return EnvPair(
        train_env=train_env,
        eval_env=eval_env,
        obs_dim=int(obs_space.shape[0]),
        n_actions=int(act_space.n),
    )


def make_discrete_env_pair(env_id: str, rng: Rng) -> EnvPair:
    """Create and seed a train/eval env pair for a flat-obs discrete task.

    Args:
        env_id: Gymnasium env id; must have a 1-D ``Box`` observation space
            and a ``Discrete`` action space.
        rng: The run's seed bundle; child seeds 0/1/2 are consumed per the
            module-level layout.

    Raises:
        ValueError: If the env's spaces do not match the requirements.
    """
    train_env: gym.Env[Any, Any] = gym.make(env_id)
    eval_env: gym.Env[Any, Any] = gym.make(env_id)
    train_env.reset(seed=rng.child_seed(0))
    train_env.action_space.seed(rng.child_seed(1))
    eval_env.reset(seed=rng.child_seed(2))
    return env_pair_from_envs(train_env, eval_env)


@dataclass(frozen=True, slots=True)
class ContinuousEnvPair:
    """A seeded train/eval env pair for continuous-action tasks.

    Same child-seed layout as :class:`EnvPair` (0 -> train env, 1 -> train
    action space, 2 -> eval env).

    Attributes:
        train_env: The training env.
        eval_env: A separate evaluation env.
        obs_dim: Flat observation size.
        act_dim: Action-vector size.
        action_low: Per-dimension lower bounds (finite).
        action_high: Per-dimension upper bounds (finite).
    """

    train_env: gym.Env[Any, Any]
    eval_env: gym.Env[Any, Any]
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    def close(self) -> None:
        """Close both environments."""
        self.train_env.close()
        self.eval_env.close()


def continuous_pair_from_envs(
    train_env: gym.Env[Any, Any], eval_env: gym.Env[Any, Any]
) -> ContinuousEnvPair:
    """Build a validated :class:`ContinuousEnvPair` from existing envs.

    Raises:
        ValueError: If observation/action spaces are not 1-D Box, or action
            bounds are not finite.
    """
    obs_space = train_env.observation_space
    act_space = train_env.action_space
    name = getattr(getattr(train_env, "spec", None), "id", type(train_env).__name__)
    if (
        not isinstance(obs_space, spaces.Box)
        or obs_space.shape is None
        or len(obs_space.shape) != 1
    ):
        raise ValueError(f"{name}: requires a flat Box observation space, got {obs_space}.")
    if (
        not isinstance(act_space, spaces.Box)
        or act_space.shape is None
        or len(act_space.shape) != 1
    ):
        raise ValueError(f"{name}: requires a 1-D Box action space, got {act_space}.")
    low = np.asarray(act_space.low, dtype=np.float32)
    high = np.asarray(act_space.high, dtype=np.float32)
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError(f"{name}: requires finite action bounds, got low={low} high={high}.")
    return ContinuousEnvPair(
        train_env=train_env,
        eval_env=eval_env,
        obs_dim=int(obs_space.shape[0]),
        act_dim=int(act_space.shape[0]),
        action_low=low,
        action_high=high,
    )


def make_continuous_env_pair(env_id: str, rng: Rng) -> ContinuousEnvPair:
    """Create and seed a train/eval env pair for a continuous-action task.

    Args:
        env_id: Gymnasium env id; 1-D Box observation and action spaces
            with finite action bounds.
        rng: The run's seed bundle; child seeds 0/1/2 per the module-level
            layout.

    Raises:
        ValueError: If the env's spaces do not match the requirements.
    """
    train_env: gym.Env[Any, Any] = gym.make(env_id)
    eval_env: gym.Env[Any, Any] = gym.make(env_id)
    train_env.reset(seed=rng.child_seed(0))
    train_env.action_space.seed(rng.child_seed(1))
    eval_env.reset(seed=rng.child_seed(2))
    return continuous_pair_from_envs(train_env, eval_env)

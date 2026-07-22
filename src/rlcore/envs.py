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

    obs_space = train_env.observation_space
    act_space = train_env.action_space
    if (
        not isinstance(obs_space, spaces.Box)
        or obs_space.shape is None
        or len(obs_space.shape) != 1
    ):
        raise ValueError(f"{env_id}: requires a flat Box observation space, got {obs_space}.")
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError(f"{env_id}: requires a Discrete action space, got {act_space}.")
    return EnvPair(
        train_env=train_env,
        eval_env=eval_env,
        obs_dim=int(obs_space.shape[0]),
        n_actions=int(act_space.n),
    )

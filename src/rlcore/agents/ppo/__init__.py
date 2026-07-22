"""PPO (clipped-surrogate policy optimization) — the M2 vertical slice.

Concrete by design, like REINFORCE (M1): actor-critic model, fixed-horizon
rollouts, GAE with explicit termination/truncation bootstrapping, clipped
surrogate objective, minibatch epochs, and approximate-KL monitoring — with
plumbing kept local to this package. Shared abstractions are extracted at
M3 from this code and REINFORCE's. Derivation and analysis:
``docs/algorithms/ppo.md``.
"""

from rlcore.agents.ppo.gae import compute_gae
from rlcore.agents.ppo.loss import (
    approx_kl_divergence,
    clipped_surrogate_loss,
    explained_variance,
    value_mse_loss,
)
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.ppo.rollout import Rollout, RolloutCollector
from rlcore.agents.ppo.train import PpoConfig, TrainResult, ppo_update, train

__all__ = [
    "ActorCritic",
    "PpoConfig",
    "Rollout",
    "RolloutCollector",
    "TrainResult",
    "approx_kl_divergence",
    "clipped_surrogate_loss",
    "compute_gae",
    "explained_variance",
    "ppo_update",
    "train",
    "value_mse_loss",
]

"""SAC (Soft Actor-Critic) — the M6 vertical slice for continuous actions.

Concrete by design: squashed-Gaussian actor with the exact change-of-
variables log-probability, twin Q-networks with Polyak-averaged targets,
entropy regularization with automatic temperature tuning (fixed-alpha mode
configurable), reusing the shared replay buffer. Derivation and analysis:
``docs/algorithms/sac.md``.
"""

from rlcore.agents.sac.loss import (
    actor_loss,
    alpha_loss,
    critic_loss,
    critic_targets,
    polyak_update,
)
from rlcore.agents.sac.model import (
    ContinuousQNetwork,
    SquashedGaussianActor,
    tanh_squash_log_jacobian,
)
from rlcore.agents.sac.train import SacConfig, TrainResult, sac_update, train

__all__ = [
    "ContinuousQNetwork",
    "SacConfig",
    "SquashedGaussianActor",
    "TrainResult",
    "actor_loss",
    "alpha_loss",
    "critic_loss",
    "critic_targets",
    "polyak_update",
    "sac_update",
    "tanh_squash_log_jacobian",
    "train",
]

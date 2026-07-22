"""REINFORCE (vanilla Monte-Carlo policy gradient) — the M1 vertical slice.

Deliberately concrete: this package is the smallest end-to-end exercise of
the platform (collection -> returns -> policy gradient -> update -> eval),
with plumbing inlined for inspectability. Shared abstractions are extracted
at M3, from this code and PPO's. Derivation and analysis:
``docs/algorithms/reinforce.md``.
"""

from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.reinforce.returns import discounted_returns
from rlcore.agents.reinforce.rollout import collect_episode
from rlcore.agents.reinforce.train import ReinforceConfig, TrainResult, reinforce_update, train

__all__ = [
    "CategoricalMlpPolicy",
    "ReinforceConfig",
    "TrainResult",
    "collect_episode",
    "discounted_returns",
    "reinforce_update",
    "train",
]

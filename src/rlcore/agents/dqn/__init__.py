"""DQN (deep Q-learning) — the M5 vertical slice, with Double DQN optional.

Concrete by design like the on-policy algorithms: Q-network, target
network, epsilon-greedy exploration, uniform replay, warm-up, minibatch TD
updates, configurable update/synchronization frequencies — with the core
math in small visible functions. Derivation and analysis:
``docs/algorithms/dqn.md``.
"""

from rlcore.agents.dqn.loss import linear_epsilon, q_huber_loss, select_next_values, td_targets
from rlcore.agents.dqn.model import QNetwork, epsilon_greedy_action
from rlcore.agents.dqn.train import DqnConfig, TrainResult, dqn_update, train

__all__ = [
    "DqnConfig",
    "QNetwork",
    "TrainResult",
    "dqn_update",
    "epsilon_greedy_action",
    "linear_epsilon",
    "q_huber_loss",
    "select_next_values",
    "td_targets",
    "train",
]

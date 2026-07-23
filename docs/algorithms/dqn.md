# DQN (deep Q-learning), with Double DQN

**Implementation:** `src/rlcore/agents/dqn/`
**References:** Watkins & Dayan (1992), "Q-learning"; Mnih et al. (Nature
2015), "Human-level control through deep reinforcement learning";
van Hasselt et al. (AAAI 2016), "Deep Reinforcement Learning with Double
Q-learning"; Sutton & Barto (2018), ch. 6 and 11.

## 1. Q-learning

Q-learning learns the optimal action-value function via the Bellman
optimality backup: from a transition (s, a, r, s'),

    Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

It is **off-policy**: the backup uses the *max* over next actions, not the
action the behavior policy took — so data from any sufficiently exploratory
policy (here, epsilon-greedy) trains estimates of the *optimal* values. In
the tabular case, with standard step-size conditions and persistent
exploration, Q-learning converges to Q\* (Watkins & Dayan, 1992).

## 2. What function approximation breaks, and DQN's two stabilizers

With a neural network Q_θ, the loss on a sampled minibatch is

    L(θ) = Huber( Q_θ(s, a),  y ),    y = r + γ (1 − terminated) · max_a' Q_θ⁻(s', a')

Bootstrapping + function approximation + off-policy data is the "deadly
triad" (Sutton & Barto, ch. 11): each ingredient is fine alone, together
they can diverge. DQN's two mitigations, both implemented here:

1. **Experience replay** (`rlcore.replay`): sampling minibatches uniformly
   from a large ring buffer breaks the temporal correlation of consecutive
   transitions and reuses data (off-policy learning makes the reuse valid).
2. **Target network** Q_θ⁻: the bootstrap target uses a periodically
   synchronized *copy* of the network, so the regression target does not
   move with every gradient step. No gradient ever flows through it
   (enforced and tested).

The Huber loss follows the DQN lineage — the Nature paper's TD-error
clipping is equivalent to Huber — bounding the gradient of outlier errors.

## 3. Termination vs. truncation (the platform contract)

The target multiplies the bootstrap by ``(1 − terminated)``:

- **terminated** (MDP-true ending): no future exists; bootstrap suppressed.
- **truncated** (external time limit): under the stated indefinite-horizon
  objective (`reinforce.md` §5), the return continues past the cutoff. The
  replay buffer stores the episode's *true final observation* as
  ``next_obs`` (raw envs do not autoreset), with ``terminated = False`` —
  so the target bootstraps from the final observation exactly as required.
- Collectors never store a transition crossing a reset, so no temporal
  chain ever spans an episode boundary.

All three behaviors are unit-tested with hand-computed targets.

## 4. Exploration

Epsilon-greedy with linear annealing (`eps_start → eps_final` over
`eps_fraction · total_steps`), preceded by a uniform-random warm-up that
fills the buffer before the first update. All exploration randomness draws
from the run's explicit torch generator, keeping runs seed-reproducible.

## 5. Overestimation and Double DQN

The max operator both *selects* and *evaluates* the next action with the
same (noisy) estimator, biasing targets upward — E[max_a Q̂] ≥ max_a E[Q̂]
by Jensen. Double DQN (van Hasselt et al., 2016) decouples the roles: the
**online** network selects ``a* = argmax_a Q_θ(s', a)``, the **target**
network evaluates ``Q_θ⁻(s', a*)``. Implemented as the ``double_q`` config
flag; with it off, the computation is exactly standard DQN. A unit test
pins the defining property that double targets never exceed standard
max-targets on the same batch.

## 6. Weaknesses

- **Discrete actions only** — the max over actions is an enumeration.
- **No convergence guarantee** under function approximation; divergence is
  configuration-dependent (target sync too infrequent/frequent, lr too
  high).
- **Overestimation** (mitigated, not eliminated, by Double DQN).
- **Uniform replay** ignores transition informativeness (prioritized
  replay is a deliberate non-goal at M5).
- Sensitive to reward scale (Huber helps; no reward clipping is applied
  here — classic-control scales don't need it, and clipping would change
  the objective).

## 7. Comparison with alternatives

| Algorithm | Relation |
|---|---|
| **PPO/REINFORCE** | On-policy policy gradients: fresh data per update, direct policy over any action space; DQN reuses data heavily but is discrete-only and learns values, not a policy object. |
| **Double DQN** | Same machinery, decoupled selection/evaluation in the target. |
| **SAC (M6)** | Off-policy like DQN (shares the replay buffer), continuous actions, entropy-regularized; soft value targets replace the hard max. |

## 8. Open problems this algorithm exposes

The deadly triad's precise boundary (when exactly does semi-gradient TD
diverge?); principled target-network schedules; exploration beyond
epsilon-greedy; replay prioritization without bias.

## 9. Validation for this implementation

Multi-seed end-to-end implementation validation on CartPole-v1 and
Acrobot-v1, plus a 2-seed Double-DQN configuration check
(`benchmarks/results/m5_dqn.*`). Hand-computed TD targets, terminal
suppression, truncation bootstrap, target isolation/synchronization,
epsilon schedule, replay overwrite/determinism/restore, update-equivalence,
and continuous-vs-resume (in-process and fresh-process) tests in
`tests/unit/test_dqn.py` and `tests/unit/test_replay.py`.

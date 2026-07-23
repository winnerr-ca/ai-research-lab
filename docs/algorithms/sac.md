# SAC (Soft Actor-Critic)

**Implementation:** `src/rlcore/agents/sac/`
**References:** Haarnoja et al. (ICML 2018), "Soft Actor-Critic: Off-Policy
Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor";
Haarnoja et al. (2019), "Soft Actor-Critic Algorithms and Applications"
(automatic temperature tuning); Fujimoto et al. (ICML 2018), "Addressing
Function Approximation Error" (the twin-Q/clipped-double-Q idea SAC
adopts).

## 1. Maximum-entropy RL

SAC maximizes return *plus* policy entropy:

    J(pi) = E [ Σ_t γ^t ( r_t + α H(pi(·|s_t)) ) ]

The temperature α trades reward against entropy. Consequences: exploration
is part of the objective rather than an external epsilon; the optimal
policy is stochastic (a Boltzmann distribution over soft Q-values); and
learning is off-policy, so the shared replay buffer applies.

## 2. Soft policy evaluation (critics)

The soft Bellman target for a transition (s, a, r, s'), with a' freshly
sampled from the current policy at s':

    y = r + γ (1 − terminated) [ min(Q'_1(s',a'), Q'_2(s',a')) − α log π(a'|s') ]

- **Twin Q + min** (Fujimoto et al., 2018): two independent critics, target
  takes the element-wise minimum — countering the positive bias that a
  single maximizing critic accumulates.
- **Polyak-averaged targets**: θ' ← τθ + (1−τ)θ' each update (τ = 0.005) —
  the continuous analog of DQN's periodic hard sync.
- **Termination contract**: bootstrap suppressed exactly at MDP-true
  endings; truncated transitions bootstrap from their stored final
  observation (`rlcore.replay`), per the platform's stated
  indefinite-horizon objective.

Both critics regress on the shared target with MSE.

## 3. Reparameterized actor with tanh squashing

The policy is a Gaussian in pre-squash space, pushed through tanh and
scaled to the env bounds:

    u ~ N(μ_θ(s), σ_θ(s)),   a = tanh(u) · scale + bias

Sampling is **reparameterized** (a = f_θ(s, ε), ε ~ N(0, I)), giving the
low-variance pathwise gradient of

    J_π(θ) = E[ α log π_θ(a|s) − min(Q_1, Q_2)(s, a) ]     (minimized)

**Exact change of variables** (the classic silent-bug site, tested against
torch.distributions' independent implementation):

    log π(a|s) = log N(u; μ, σ) − Σ_i log(1 − tanh(u_i)²) − Σ_i log(scale_i)

computed with the stable identity
``log(1 − tanh(u)²) = 2(log 2 − u − softplus(−2u))`` — the naive form
underflows to log(0) for |u| ≳ 20, exactly where a confident policy
operates; finiteness near the bounds is unit-tested.

## 4. Automatic temperature tuning

Fixed α is brittle across reward scales. The constrained formulation
(Haarnoja et al., 2019) targets a fixed entropy level H̄ (heuristic:
−act_dim) and adapts α by minimizing

    J(α) = E[ −log α · (log π(a|s) + H̄) ]

so α rises when entropy is below target and falls above it (the gradient
direction is unit-tested). `auto_alpha=False` freezes α at `fixed_alpha`
for the ablation-friendly fixed mode.

## 5. Weaknesses

- **Hyperparameter sensitivity is reduced, not gone** — τ, lr, and network
  width still matter; reward scale still leaks in through the critics.
- **No discrete-action support** in this form (the squashed Gaussian is
  continuous; discrete SAC variants exist but are out of scope).
- **Bounded-action assumption**: finite Box bounds are required and
  enforced at construction.
- **Compute per env step** is the highest on the platform (three networks
  plus targets update every step).

## 6. Comparison with alternatives

| Algorithm | Relation |
|---|---|
| **DQN (M5)** | Shares replay + target-network machinery; DQN's hard max over discrete actions becomes a soft, sampled continuous update. |
| **PPO (M2)** | On-policy with fresh data per update; SAC reuses replay data and typically needs far fewer env steps on continuous control, at higher per-step compute. |
| **TD3** | The deterministic-policy sibling: twin critics + target policy smoothing instead of entropy; SAC's stochastic actor subsumes exploration. |

## 7. Open problems this algorithm exposes

Principled target-entropy selection (the −act_dim heuristic is a
heuristic); interaction between entropy regularization and value
overestimation; replay staleness for off-policy actors.

## 8. Validation for this implementation

Multi-seed end-to-end implementation validation on Pendulum-v1
(`benchmarks/results/m6_sac.*`) — MuJoCo-scale claims are explicitly out of
scope on this hardware. Unit tests: action-bound enforcement, analytic
log-prob vs. torch.distributions, finiteness near bounds, twin-min
targets, Polyak averaging (hand-computed + SB3 parity), actor/critic/alpha
losses (hand-computed, gradient-direction), fixed-alpha behavior,
termination/truncation targets, replay restore, resume-equals-continuous,
evaluation side-effect freedom.

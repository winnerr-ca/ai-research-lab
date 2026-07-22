# PPO (Proximal Policy Optimization, clipped surrogate)

**Implementation:** `src/rlcore/agents/ppo/`
**References:** Schulman et al. (2017), "Proximal Policy Optimization
Algorithms"; Schulman et al. (2016), "High-Dimensional Continuous Control
Using Generalized Advantage Estimation" (GAE); Engstrom et al. (ICLR 2020),
"Implementation Matters"; Huang et al. (ICLR Blog Track 2022), "The 37
Implementation Details of PPO".

## 1. The problem PPO solves

REINFORCE (M1) takes exactly one gradient step per batch of on-policy data,
then discards it — and it has no defense against a single destructive
update. Both problems share a cause: the policy-gradient estimator is only
valid *at* the behavior policy. Reusing data or stepping far requires
correcting for, and controlling, the distribution shift.

## 2. Importance-sampled surrogate

For data collected under the old policy π_old, the performance of a new
policy π_θ can be estimated with importance ratios
ρ_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t):

    L^IS(θ) = E_t [ ρ_t(θ) · Â_t ]

whose gradient at θ = θ_old equals the policy gradient. The estimate
degrades as π_θ leaves π_old's neighborhood (the ratios' variance grows and
the first-order surrogate stops tracking true performance). TRPO constrains
the step with an explicit KL trust region; PPO replaces the constraint with
a clipped objective.

## 3. The clipped objective

    L^CLIP(θ) = E_t [ min( ρ_t Â_t,  clip(ρ_t, 1−ε, 1+ε) Â_t ) ]

Mechanics (implemented in `loss.py`, tested case by case):

- If the update would *improve* the surrogate beyond the clip range
  (ρ > 1+ε with Â > 0, or ρ < 1−ε with Â < 0), the clipped term is active
  and its gradient with respect to ρ is zero — the incentive to move
  further vanishes.
- The `min` makes the bound *pessimistic*: the unclipped term is kept
  whenever it is worse, so the objective is a lower bound on the
  importance-sampled surrogate and never rewards clipping-induced optimism.

Clipping is a heuristic proxy for a trust region, not an equivalent of one:
the ratio can drift outside the clip range (gradients from other samples
move shared parameters), which is why we also *monitor* the KL (§6).

## 4. GAE: the advantage estimator

The k-step advantage estimators interpolate between the one-step TD residual
(low variance, value-function bias) and Monte-Carlo advantages (unbiased
given the true value function, high variance). GAE takes their exponentially
weighted average, computable by the backward recursion

    δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
    Â_t = δ_t + γλ·(1 − done_t)·Â_{t+1}

with λ the bias/variance dial. Limits (both are enforced as tests):
λ = 0 gives Â_t = δ_t; λ = 1 telescopes so that Â_t + V(s_t) is the
discounted bootstrapped return of the remaining segment.

**Termination vs. truncation** implements the task-objective choice stated
in `reinforce.md` §5 (CartPole as an indefinite-horizon task under an
external time limit): at a *true terminal*, V(s_{t+1}) := 0; at a
*truncation*, V(s_{t+1}) := V(s_final) — the value function estimates the
return the time limit cut off; and the recursion breaks (`done_t`) at both
kinds of boundary, because the next stored step belongs to a new episode.
The collector owns this contract (`rollout.py`); GAE just consumes it.

## 5. Value loss, entropy, and the composed objective

The value network regresses on targets Â_t + V_old(s_t) (equivalently, the
λ-return) with an MSE loss. An optional entropy bonus discourages premature
determinism (default coefficient 0 for CartPole, following SB3). The
composed minibatch loss matches SB3's convention:

    L = L^policy + c_vf · L^value − c_ent · H[π_θ]

**Advantage normalization** (per minibatch, default on) carries the same
framing as M1's return standardization: a *configurable implementation
heuristic* — same-batch statistics can introduce finite-batch bias, and the
std division rescales the effective step size. It is not part of the PPO
derivation.

## 6. Approximate-KL monitoring

Multiple epochs over one rollout are exactly the regime where π_θ can drift
from π_old. We monitor E[KL(π_old‖π_θ)] per minibatch with the non-negative,
low-variance k3 estimator (ρ − 1) − log ρ (Schulman, "Approximating KL
Divergence"), and optionally stop the remaining epochs of an iteration when
it exceeds 1.5 × `target_kl` (SB3's convention; disabled by default). The
monitor is diagnostic — persistent KL growth with degrading returns is the
classic signature of a destructive-update regime.

## 7. Weaknesses and failure modes

- **Still on-policy:** every rollout is used for a handful of epochs, then
  discarded; sample efficiency remains far below replay-based methods.
- **Clipping is not a trust region:** ratios can leave the clip range
  without penalty; pathologies documented by Engstrom et al. show much of
  PPO's practical performance rides on implementation details
  (normalization, initialization, learning-rate schedules) rather than the
  clipping alone.
- **Hyperparameter coupling:** n_steps, minibatch size, epochs, clip range,
  and learning rate interact; defaults transfer poorly across task families.
- **Single-env collection here (M2):** wall-clock inefficient; vectorized
  collection is an M3 concern.

## 8. Comparison with alternatives

| Algorithm | Relation |
|---|---|
| **REINFORCE (M1)** | PPO's ancestor: one step per batch, no baseline, no ratio control. PPO adds a learned baseline (critic), GAE, data reuse via clipped IS ratios. |
| **TRPO** | The explicit-constraint version: theoretically cleaner monotonic-improvement motivation, second-order machinery; PPO trades that for first-order simplicity. |
| **A2C** | PPO with one epoch and no clipping (approximately); cheaper per update, less data reuse. |
| **SAC/DQN (off-policy)** | Replay-based sample efficiency at the cost of different stability machinery; arrive at M5. |

## 9. Open problems this algorithm exposes

What the clipped objective actually optimizes (the surrogate's relationship
to true policy improvement is loose); principled step-size/trust-region
selection without TRPO's cost; the role of implementation details vs.
algorithm (Engstrom et al.); variance-aware advantage normalization with
guarantees rather than heuristics.

## 10. Validation for this implementation

- Exact unit tests: GAE limits (λ ∈ {0, 1}), termination vs. truncation
  bootstrapping, clipped/unclipped loss cases, value loss, minibatch
  accounting, finite gradients, seed-controlled training.
- Isolated reference comparison: GAE against SB3's
  `RolloutBuffer.compute_returns_and_advantage` on identical synthetic
  inputs, restricted to termination-only sequences where the two
  computations are genuinely equivalent (SB3's buffer never sees truncation;
  SB3 folds truncation bootstrap into rewards upstream).
- End-to-end implementation validation: multi-seed CartPole-v1 run with
  recorded protocol (`benchmarks/results/`).

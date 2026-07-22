# REINFORCE (Monte-Carlo policy gradient)

**Implementation:** `src/rlcore/agents/reinforce/`
**Reference:** Williams (1992), "Simple statistical gradient-following
algorithms for connectionist reinforcement learning"; Sutton & Barto (2018),
ch. 13.

## 1. Setting and objective

Episodic MDP with initial-state distribution ρ, dynamics P, reward r, and a
stochastic policy π_θ(a|s) differentiable in θ. A trajectory
τ = (s₀, a₀, r₀, …, s_{T−1}, a_{T−1}, r_{T−1}) has probability

    p_θ(τ) = ρ(s₀) · Π_t π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t)

and we maximize the expected discounted return

    J(θ) = E_{τ~p_θ} [ Σ_t γ^t r_t ].

## 2. The likelihood-ratio ("log-derivative") trick

The gradient of an expectation over a θ-dependent distribution:

    ∇_θ J = ∇_θ ∫ p_θ(τ) R(τ) dτ
          = ∫ p_θ(τ) ∇_θ log p_θ(τ) R(τ) dτ        (since ∇p = p ∇log p)
          = E_τ [ ∇_θ log p_θ(τ) · R(τ) ].

Crucially, log p_θ(τ) decomposes as log ρ(s₀) + Σ_t log π_θ(a_t|s_t)
+ Σ_t log P(s_{t+1}|s_t, a_t), and only the policy terms depend on θ:

    ∇_θ log p_θ(τ) = Σ_t ∇_θ log π_θ(a_t | s_t).

**This is why REINFORCE works without a model:** the unknown dynamics and
initial-state terms vanish from the gradient. We can follow the gradient of
expected return using only samples and the policy's own log-probabilities.

## 3. Variance reduction I: reward-to-go (causality)

Using the full return R(τ) for every step is wasteful: the action at time t
cannot influence rewards earned *before* t. Formally, for k < t,
E[∇log π_θ(a_t|s_t) · r_k] = 0 (condition on the history up to s_t; the inner
expectation of the score is zero). Dropping those terms leaves the
**reward-to-go** estimator used in this implementation:

    ∇_θ J ≈ (1/N) Σ_episodes Σ_t ∇_θ log π_θ(a_t|s_t) · G_t,
    G_t = Σ_{k≥t} γ^{k−t} r_k          (rlcore.agents.reinforce.returns)

Same expectation, strictly lower (or equal) variance.

## 4. Variance reduction II: baselines and our standardization

Subtracting any action-independent baseline b(s_t) from G_t leaves the
gradient unbiased: E[∇log π_θ(a_t|s_t) · b(s_t)] = b · E[score] = 0.

This implementation uses **batch return standardization**
(`normalize_returns=True`): within each update batch, returns are shifted by
their mean and divided by their standard deviation. Two honest caveats:

- The subtracted mean is a *constant* baseline estimated from the same batch
  it is applied to, which introduces a small O(1/N) correlation bias — in
  practice negligible against the variance it removes.
- Dividing by the standard deviation is not a baseline at all; it rescales
  the gradient, acting as an adaptive step size that makes a single learning
  rate workable across training stages.

A learned state-value baseline (actor-critic) is the principled next step and
arrives with PPO at M2.

## 5. Two documented biases in the standard estimator

1. **Dropped γ^t weighting.** The exact gradient of the discounted objective
   carries a γ^t factor per step (∇J = E[Σ_t γ^t G_t ∇log π]). Like nearly
   all practical implementations, we drop it — the resulting update is not
   the gradient of J but of a related "average-ish" objective (Thomas, ICML
   2014; Nota & Thomas, AAMAS 2020). Standard practice, stated openly.
2. **Truncation.** `G_{T−1} = r_{T−1}` assumes the episode is over. When the
   env *truncates* (CartPole-v1's 500-step time limit), the true return
   continues past the cutoff, so late-episode reward-to-go is
   underestimated. Vanilla REINFORCE has no value function to bootstrap
   from, so this bias is inherent to the algorithm on time-limited tasks
   (Pardo et al., ICML 2018). The collector records `terminated` and
   `truncated` separately precisely so bootstrapping algorithms (M2+) can do
   this correctly.

## 6. Why high variance, and other weaknesses

The estimator uses a single Monte-Carlo rollout of an entire episode per
sample: every source of randomness (policy, dynamics, episode length)
accumulates into G_t. Consequences and further weaknesses:

- **Sample inefficiency:** strictly on-policy; every gradient step discards
  its data. Full episodes must finish before any update.
- **Step-size sensitivity:** no trust region or clipping; one large update
  can collapse the policy (motivates TRPO/PPO).
- **Credit assignment:** reward-to-go is the only mechanism; no bootstrapping
  or eligibility-style attribution within the episode.

## 7. Comparison with alternatives

| Algorithm | Relation |
|---|---|
| **DQN** | Off-policy, value-based, replay buffer; far more sample-efficient on discrete tasks but no direct policy over continuous actions and its own stability machinery (target nets). |
| **Actor-critic (A2C)** | Adds a learned value baseline and bootstrapped targets: less variance, some bias, updates before episode end. |
| **PPO** | Adds GAE, a clipped surrogate allowing multiple epochs per batch, and minibatching — the practical workhorse; REINFORCE is its ancestor with every safeguard removed. |

## 8. Open problems this algorithm exposes

Variance reduction beyond baselines (action-dependent baselines and their
debated benefits — cf. Tucker et al., ICML 2018), principled handling of the
discount/γ^t mismatch, credit assignment over long horizons, and step-size
selection without trust regions — each visible in miniature here, each still
active research.

## 9. Validation experiments for this implementation

- **Multi-seed learning on CartPole-v1** (the M1 validation report,
  `benchmarks/results/`): seeds, config, and per-seed outcomes recorded.
- Worthwhile ablations the platform can run later: `normalize_returns`
  on/off (variance reduction's practical value), γ sweep (bias/variance of
  reward-to-go), batch size (episodes_per_update) vs. gradient noise.

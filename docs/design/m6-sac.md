# M6: SAC and continuous-action infrastructure

**Status:** Implemented

## Objective, and why

A correct Soft Actor-Critic implementation — the platform's first
continuous-action algorithm — introducing the tanh-squashed Gaussian
policy with the exact change-of-variables log-probability, twin Q
networks with min-of-two soft targets, Polyak target averaging, and
automatic temperature tuning (with a fixed-alpha configuration mode).
SAC is the scheduled second consumer of `rlcore.replay` from M5, which
is what justified extracting the buffer there (rule of two, satisfied on
schedule rather than speculatively).

## Deliverables / non-goals / acceptance

Delivered: `SquashedGaussianActor` with numerically stable tanh
log-Jacobian `2(log 2 − u − softplus(−2u))` and registered
action-scale/bias buffers; `ContinuousQNetwork` twin critics;
pure-function critic targets, critic/actor/alpha losses, and
`polyak_update`; step-based trainer reusing the M5 replay buffer;
continuous env-pair construction with finite-bounds validation and the
same child-seed layout as the discrete path; run-dir/tracker/CLI/
checkpoint integration with exact resume (including `log_alpha` and its
optimizer). Non-goals held: no MuJoCo claims, no distributional or
ensemble critics, no CrossQ/TQC variants. Acceptance: required test
list green (23 SAC tests), canonical gate green, multi-seed validation
recorded.

## Correctness anchors

- The squashed log-probability is cross-checked against
  `torch.distributions.TransformedDistribution(Normal, TanhTransform)`
  to tight tolerance — an independent implementation of the same
  change of variables.
- `polyak_update` is checked for parity against Stable-Baselines3's
  `polyak_update` (a genuinely equivalent isolated function, unlike the
  DQN TD path at M5).
- The alpha loss's gradient direction is tested: entropy below target
  pushes alpha up, above target pushes it down.
- Critic targets use `(1 − terminated)` exactly as in DQN; the replay
  buffer's true-successor-observation semantics carry over unchanged,
  so truncation bootstraps and termination suppresses.

## Platform bug exposed and fixed

SAC's `target_entropy: float = nan` sentinel ("derive −act_dim at
setup") broke `check_resume_config`, which compared config fields with
`==` — and `nan != nan`. Resume of any SAC run with the default
sentinel would have been rejected as a config mismatch. Fixed in
`rlcore.checkpoints` with NaN-aware field comparison, plus a regression
test. This is the concrete value of the fresh-process resume test
requirement: the bug surfaced before any user hit it.

## Validation results (`benchmarks/results/m6_sac.{md,json}`)

Stated criterion (docs/reproductions/repro-sac.md): ≥4/5 auto-alpha
seeds at or above −180 mean deterministic return on the fresh final
evaluation.

- **Auto-alpha, 5 seeds:** seeds 0/1/3/4 reached −146.1 / −129.7 /
  −156.5 / −155.7. **Seed 2 missed at −205.7 ± 104.6** — its periodic
  evaluation crossed the −180 early-stop bar but the fresh final
  evaluation landed below it (the early-stop-selection caveat from M1,
  again visible in the data; Pendulum eval std is ~100, so single-seed
  eval means are noisy). 4/5 → criterion met, with the miss recorded.
- **Fixed alpha 0.2 (2-seed configuration check):** −136.0 and −129.1.
  Configuration works; no auto-vs-fixed performance claim is made or
  supportable from n=2.
- All runs early-stopped at 4k–8k of the 40k-step budget.

## Self-review

- MuJoCo-scale validation is explicitly out of scope on this hardware;
  the report says so rather than implying broader evidence.
- Deterministic evaluation of a stochastic-policy algorithm
  (`tanh(mu)`) is the standard SAC protocol; the independent stochastic
  evaluation is also recorded so the gap is visible.
- Known limitation: one gradient step per env step on CPU bounds
  throughput (~75–90 env steps/s); M8 measures rather than guesses.

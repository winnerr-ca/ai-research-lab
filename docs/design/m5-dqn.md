# M5: DQN and off-policy infrastructure

**Status:** Implemented

## Objective, and why

A correct DQN baseline for discrete-action environments — the platform's
first off-policy algorithm — introducing the shared replay buffer
(`rlcore.replay`; SAC at M6 is its scheduled second consumer, satisfying
the rule of two), target networks, epsilon-greedy exploration with warm-up,
and a step-based training loop. Double DQN ships as a clearly configurable
extension (`double_q`) validated after the baseline.

## Deliverables / non-goals / acceptance

Delivered: replay buffer with explicit-RNG sampling and full save/restore;
Q/target networks; pure-function TD targets, next-value selection, epsilon
schedule, Huber loss; configurable train/target-sync frequencies; gradient
clipping; run-dir/tracker/CLI/checkpoint integration; exact resume incl.
replay state. Non-goals held: no prioritized replay, dueling,
distributional RL, or Rainbow. Acceptance: required test list green (33
tests), canonical gate green, multi-seed validation recorded.

## Termination/truncation handling

The buffer stores the *true successor observation* per transition (raw
envs never autoreset, so at truncation this is the episode's final
observation) with `terminated` marking only MDP-true endings; TD targets
multiply the bootstrap by `(1 − terminated)`. No stored transition crosses
a reset. All three behaviors are unit-tested with hand-computed targets.

## Validation results (`benchmarks/results/m5_dqn.{md,json}`)

Five seeds per env, default config (Acrobot: `stop_return=-100`):

- **CartPole-v1:** 4/5 seeds reach ≥475 greedy (three at 500.0 exactly);
  **seed 1 failed** — plateaued at 96.3 within the 60k-step budget. This is
  reported, not hidden: single-seed DQN instability on CartPole is a
  documented phenomenon, and it is exactly why the platform's comparison
  protocol demands multiple seeds.
- **Acrobot-v1:** 5/5 seeds reach the −100 early-stop bar (final fresh
  evals −74.3 to −152.5; seed 4's fresh final eval was substantially worse
  than its triggering eval — the early-stop-selection caveat from M1
  again, visible in the data).
- **Double DQN (2-seed configuration check):** seed 0 solved, seed 1 did
  not within budget (138.8). With n=2 this is a *configuration works*
  check; no standard-vs-Double performance claim is made or supportable
  from it.

## Self-review

- The `-dirty` protection and clean-SHA workflow held (report stamps the
  M5 code commit).
- SB3 was not used for DQN parity: SB3's TD-target computation is not
  callable in isolation (it lives inside `DQN.train()`), so no genuinely
  equivalent isolated comparison exists; hand-computed references and the
  update-equivalence test carry the correctness argument instead. SB3
  parity returns in M6 where `polyak_update` *is* an isolated equivalent.
- Known limitation: per-step Python loop bounds throughput (~360 env
  steps/s with updates on this CPU); M8 addresses measurement and
  vectorization.

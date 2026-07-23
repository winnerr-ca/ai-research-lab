# Reproduction package: DQN (and Double DQN)

**Label: implementation validation.** No Atari results are claimed — the
Nature DQN's experimental domain was not run. Validated: the algorithm as
specified (replay, target network, epsilon-greedy, Huber TD loss) learns
two classic-control tasks across seeds, with the defining mechanisms
pinned by unit tests.

## Papers and targets

- **Citations:** Mnih et al. (Nature 2015), *Human-level control through
  deep reinforcement learning*; van Hasselt, Guez, Silver (AAAI 2016),
  *Deep Reinforcement Learning with Double Q-learning*.
- **Target claims (papers):** (1) replay + target networks stabilize deep
  Q-learning; (2) decoupling action selection from evaluation reduces
  overestimation.
- **Targets actually tested here:** (1) the implementation with both
  stabilizers learns CartPole-v1 and Acrobot-v1 across seeds; (2) the
  Double-DQN computation has the defining property (double targets never
  exceed standard max-targets on identical batches — unit test) and the
  configuration trains (2-seed check). Overestimation *reduction in
  practice* is NOT tested — that would need the papers' scale.

## Protocol

- 5 seeds per env, `DqnConfig` defaults (buffer 50k, batch 64, lr 1e-3,
  target sync 500 steps, ε 1.0→0.05 over 20% of 60k steps, Huber loss,
  grad clip 10); Acrobot early-stop at −100, CartPole at 475.
- Results: `benchmarks/results/m5_dqn.{md,json}`.

## Implementation correspondence and known deviations

| Aspect | Nature DQN | This implementation |
|---|---|---|
| Uniform replay, warm-up | ✓ (1M buffer, Atari) | ✓ (50k, classic control) |
| Target network (hard sync) | ✓ | ✓ (isolation + sync unit-tested) |
| Error clipping | clip TD error (≈ Huber) | Huber (`smooth_l1_loss`) |
| ε-greedy anneal | 1.0→0.1 over 1M frames | 1.0→0.05 over 20% of budget |
| Reward clipping, frame stacking, Nature CNN | Atari-specific | not applicable (MLP, classic control) |
| Double DQN | separate paper | `double_q` flag, exact standard DQN when off |

## Commands and resources

```sh
uv run python benchmarks/m5_dqn.py --seeds 0 1 2 3 4
```

CPU-only; ~25 minutes total on the recorded hardware.

## Results and analysis

CartPole: 4/5 seeds ≥ 475 greedy; **seed 1 failed** (96.3 at budget) —
reported, consistent with documented single-seed DQN variance. Acrobot:
5/5 reached −100. Double DQN: 1/2 seeds solved within budget (an n=2
configuration check; no comparative claim).

## Limitations

No Atari; no overestimation measurement; no prioritized replay/dueling;
budget-bounded failures visible in the raw data. Nothing here reproduces
or contradicts the papers' Atari results — not attempted.

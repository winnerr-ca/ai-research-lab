# Reproduction package: PPO

**Label: implementation validation.** This package does NOT claim a paper
reproduction — the paper's experimental domains (MuJoCo locomotion, Atari)
were not run. What is validated is that this implementation of the
published algorithm learns reliably under the published default
hyperparameters on accessible tasks, with loss-level parity against a
reference implementation where genuinely equivalent.

## Paper and target

- **Citation:** Schulman, Wolski, Dhariwal, Radford, Klimov (2017),
  *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- **Target claim (paper):** the clipped surrogate objective with multiple
  minibatch epochs per batch attains strong performance across continuous
  and discrete control with a single hyperparameter setting.
- **Target actually tested here:** the algorithm as specified (clipped
  surrogate, GAE(λ), minibatch epochs, advantage normalization) learns
  CartPole-scale discrete control reliably across seeds with
  Stable-Baselines3-default hyperparameters, and its GAE computation is
  numerically identical to SB3's.

## Protocol

- 5 seeds (0–4), SB3-default hyperparameters (`PpoConfig` defaults:
  n_steps 2048, minibatch 64, 10 epochs, γ 0.99, λ 0.95, clip 0.2,
  lr 3e-4, vf_coef 0.5, max_grad_norm 0.5), CartPole-v1.
- Evaluation: periodic 20-episode greedy eval on a child-seeded separate
  env; early stop at 475; fresh final greedy evaluation; independent
  stochastic evaluation (documented in the report).
- Results: `benchmarks/results/m2_ppo_cartpole.{md,json}` (protocol
  section included in the report).

## Implementation correspondence and known deviations

| Aspect | Paper / SB3 | This implementation |
|---|---|---|
| Clipped surrogate, min-with-unclipped | ✓ | ✓ (unit-tested per branch) |
| GAE(λ) | ✓ | ✓ — numerically identical to SB3's `RolloutBuffer` on termination-only sequences (parity test) |
| Truncation bootstrap | SB3 folds into rewards | explicit `next_values` contract (hand-verified; equivalent semantics) |
| Advantage normalization | per minibatch (SB3) | per minibatch (documented as heuristic) |
| Value clipping | optional in SB3 (off by default) | not implemented (deliberate omission) |
| LR/clip annealing | used in paper's Atari runs | not implemented (constant, as SB3 defaults) |
| Vectorized envs | 8 actors (paper) | single env (M2); vectorized collection added at M8 |

## Commands and resources

```sh
uv run python benchmarks/m2_ppo_cartpole.py --seeds 0 1 2 3 4
```

CPU-only; ~2 minutes total on the recorded hardware (see report's
environment block).

## Results and analysis

All 5 seeds reached the 475 early-stop bar in 10–20k env steps; final
greedy evals 481.4–500.0 (raw JSON in the results file). Consistent with
the algorithm functioning as specified at this scale. Seed 2 showed the
weakest stochastic policy at the earliest stop — an early-stopping
selection effect discussed in the report, not an algorithm property.

## Limitations

Single small discrete task; no MuJoCo/Atari; no annealing; n=5 seeds. None
of the paper's headline numbers are reproduced or contradicted by this
package — they were not attempted.

# Reproduction package: SAC

**Label: implementation validation.** No MuJoCo results are claimed
(MuJoCo was not run on this hardware). Validated: the algorithm as
specified — twin critics, Polyak targets, reparameterized tanh-Gaussian
actor with exact change-of-variables log-probability, automatic
temperature tuning — learns Pendulum-v1 across seeds, with the
correctness-critical math pinned by tests against independent
implementations.

## Papers and targets

- **Citations:** Haarnoja, Zhou, Abbeel, Levine (ICML 2018), *Soft
  Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic
  Actor*; Haarnoja et al. (2019), *Soft Actor-Critic Algorithms and
  Applications* (automatic temperature).
- **Target claims (papers):** maximum-entropy off-policy actor-critic is
  sample-efficient and stable on continuous control; the constrained
  formulation removes per-task temperature tuning.
- **Targets actually tested here:** the implementation learns a standard
  continuous-control task across seeds with the papers' default
  machinery; the squashed log-probability equals torch.distributions'
  independent computation; Polyak averaging equals SB3's; the temperature
  update moves α in the direction the constrained objective prescribes.

## Protocol

- 5 seeds, `SacConfig` defaults (twin 256×256 critics, τ 0.005, lr 3e-4,
  batch 256, target entropy −act_dim, auto-α), Pendulum-v1, early stop at
  −180 within 40k steps; plus a 2-seed fixed-α=0.2 configuration check.
- Results: `benchmarks/results/m6_sac.{md,json}`.

## Implementation correspondence and known deviations

| Aspect | Paper / SB3 | This implementation |
|---|---|---|
| Twin Q, min target | ✓ | ✓ (hand-computed test) |
| Polyak τ | ✓ | ✓ (SB3 parity test) |
| Tanh change of variables | ✓ | ✓ (stable identity; torch.distributions cross-check) |
| Auto temperature | ✓ | ✓ (gradient-direction test); fixed-α mode configurable |
| Updates per step | 1 (SB3 default) | 1 |
| Multiple gradient steps / UTD > 1 | used in some paper configs | not implemented |

## Commands and resources

```sh
uv run python benchmarks/m6_sac.py --seeds 0 1 2 3 4
```

CPU-only; ~20 minutes total on the recorded hardware.

## Results and analysis

See `benchmarks/results/m6_sac.md` for the per-seed table (the report is
generated from the raw JSON; consult it rather than this file for
numbers). Analysis criterion, stated in advance: implementation validation
passes if ≥ 4/5 auto-α seeds reach the −180 deterministic-eval bar within
budget.

## Limitations

Single low-dimensional task; no MuJoCo; UTD 1 only; n=5 seeds; fixed-α
section is n=2. The papers' benchmark numbers are neither reproduced nor
contradicted — not attempted.

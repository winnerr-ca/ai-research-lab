# M2 end-to-end implementation validation: PPO on CartPole-v1

Seeds: [0, 1, 2, 3, 4] — one full training run each, default `PpoConfig` (Stable-Baselines3-default hyperparameters), greedy + stochastic final evaluation (20 episodes each).

**Scope of this result.** This is an end-to-end *implementation*
validation: evidence that the implementation trains successfully on
this task across seeds. It is not evidence of algorithmic
superiority, and no performance claim beyond this task, this
configuration, and the environment recorded below is intended.

## Evaluation protocol

- **Seed streams:** the root seed derives independent child seeds
  via `SeedSequence` — child 0 seeds the training env, child 1 the
  training action space, child 2 the *separate* evaluation env.
  Action sampling and minibatch shuffling draw from the bundle's
  explicit torch generator.
- **Evaluation frequency:** a greedy (20-episode)
  evaluation every 5 updates on the dedicated eval
  env; its RNG stream continues across evaluations (episodes are
  fresh draws from the stream, not re-seeded per evaluation).
- **Stopping rule:** training stops early when a periodic greedy
  evaluation's mean return reaches 475.0, with a hard
  budget of 75 updates (153600 env steps) otherwise.
- **Final evaluation:** after training ends (early stop or budget),
  a *fresh* greedy 20-episode evaluation runs on
  the same eval env's continuing stream — the greedy numbers in the
  table are this re-evaluation, not the evaluation that triggered
  the stop.
- **Stochastic evaluation:** a separate env and sampling generator,
  both seeded `20000 + seed`, independent of
  all training streams.

## Configuration

```json
{
  "env_id": "CartPole-v1",
  "seed": "per-run (see table)",
  "total_updates": 75,
  "n_steps": 2048,
  "minibatch_size": 64,
  "n_epochs": 10,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "lr": 0.0003,
  "ent_coef": 0.0,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "normalize_advantages": true,
  "target_kl": null,
  "hidden_sizes": [
    64,
    64
  ],
  "eval_every": 5,
  "eval_episodes": 20,
  "stop_return": 475.0
}
```

## Environment

- rlcore: `0.0.1`
- python: `3.11.15`
- torch: `2.13.0+cu130`
- gymnasium: `1.3.0`
- platform: `Linux-6.18.5-x86_64-with-glibc2.39`
- commit: `b697fc27ccd0101fced44d864e3ef8c000185c47-dirty`

## Per-seed results

| Seed | Env steps | Wall time (s) | Stopped early | Greedy eval (mean±std) | Stochastic eval (mean±std) |
|---|---|---|---|---|---|
| 0 | 20480 | 14.1 | yes | 500.0 ± 0.0 | 450.5 ± 66.2 |
| 1 | 20480 | 13.4 | yes | 500.0 ± 0.0 | 490.9 ± 39.7 |
| 2 | 10240 | 7.6 | yes | 481.4 ± 36.2 | 244.8 ± 105.7 |
| 3 | 20480 | 12.9 | yes | 500.0 ± 0.0 | 482.9 ± 68.0 |
| 4 | 20480 | 13.9 | yes | 500.0 ± 0.0 | 483.4 ± 34.9 |

## Aggregate (greedy eval means across seeds)

- mean: 496.3
- median: 500.0
- min / max: 481.4 / 500.0

Small-sample caveat: these are per-seed point estimates over 5 seeds; the rliable aggregate-metrics protocol (IQM, bootstrap CIs) arrives with the M4 evaluation engine.

Reproducibility scope: these numbers are expected to reproduce only under the pinned software versions recorded above, on CPU, on comparable hardware; no broader reproducibility claim is intended.

## Reproduce

```sh
uv run python benchmarks/m2_ppo_cartpole.py --seeds 0 1 2 3 4
```

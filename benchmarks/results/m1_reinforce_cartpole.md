# M1 end-to-end implementation validation: REINFORCE on CartPole-v1

Seeds: [0, 1, 2, 3, 4] — one full training run each, default `ReinforceConfig`, greedy + stochastic final evaluation (20 episodes each).

**Scope of this result.** This is an end-to-end *implementation*
validation: evidence that the implementation trains successfully on
this task across seeds. It is not evidence of algorithmic
superiority, and no performance claim beyond this task, this
configuration, and the environment recorded below is intended.

## Evaluation protocol

- **Seed streams:** the root seed derives independent child seeds
  via `SeedSequence` — child 0 seeds the training env, child 1 the
  training action space, child 2 the *separate* evaluation env.
  Action sampling during training draws from the bundle's explicit
  torch generator.
- **Evaluation frequency:** a greedy (20-episode)
  evaluation every 20 updates on the dedicated eval
  env; its RNG stream continues across evaluations (episodes are
  fresh draws from the stream, not re-seeded per evaluation).
- **Stopping rule:** training stops early when a periodic greedy
  evaluation's mean return reaches 475.0, with a hard
  budget of 300 updates otherwise.
- **Final evaluation:** after training ends (early stop or budget),
  a *fresh* greedy 20-episode evaluation runs on
  the same eval env's continuing stream — the greedy numbers in the
  table are this re-evaluation, not the evaluation that triggered
  the stop.
- **Stochastic evaluation:** a separate env and sampling generator,
  both seeded `10000 + seed`, independent of
  all training streams.

## Configuration

```json
{
  "env_id": "CartPole-v1",
  "seed": "per-run (see table)",
  "updates": 300,
  "episodes_per_update": 10,
  "gamma": 0.99,
  "lr": 0.005,
  "hidden_sizes": [
    64,
    64
  ],
  "normalize_returns": true,
  "eval_every": 20,
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
- commit: `40776403e353f564e60a50ed0115dcc53825a79c`

## Per-seed results

| Seed | Episodes | Env steps | Wall time (s) | Stopped early | Greedy eval (mean±std) | Stochastic eval (mean±std) |
|---|---|---|---|---|---|---|
| 0 | 400 | 100960 | 28.4 | yes | 500.0 ± 0.0 | 495.3 ± 20.5 |
| 1 | 400 | 95415 | 25.4 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 2 | 400 | 98012 | 26.8 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 3 | 400 | 73466 | 19.3 | yes | 492.9 ± 16.7 | 500.0 ± 0.0 |
| 4 | 200 | 17410 | 6.5 | yes | 500.0 ± 0.0 | 383.6 ± 125.0 |

## Aggregate (greedy eval means across seeds)

- mean: 498.6
- median: 500.0
- min / max: 492.9 / 500.0

Small-sample caveat: these are per-seed point estimates over 5 seeds; the rliable aggregate-metrics protocol (IQM, bootstrap CIs) arrives with the M4 evaluation engine.

Reproducibility scope: these numbers are expected to reproduce only under the pinned software versions recorded above, on CPU, on comparable hardware; no broader reproducibility claim is intended.

## Reproduce

```sh
uv run python benchmarks/m1_reinforce_cartpole.py --seeds 0 1 2 3 4
```

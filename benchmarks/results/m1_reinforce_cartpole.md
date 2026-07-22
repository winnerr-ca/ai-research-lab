# M1 validation: REINFORCE on CartPole-v1 (multi-seed)

Seeds: [0, 1, 2, 3, 4] — one full training run each, default `ReinforceConfig`, greedy + stochastic final evaluation (20 episodes each).

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
- commit: `de3580288b7a202977b7a71ff5150c314da3dc03`

## Per-seed results

| Seed | Episodes | Env steps | Wall time (s) | Stopped early | Greedy eval (mean±std) | Stochastic eval (mean±std) |
|---|---|---|---|---|---|---|
| 0 | 400 | 100960 | 17.6 | yes | 500.0 ± 0.0 | 495.3 ± 20.5 |
| 1 | 400 | 95415 | 15.7 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 2 | 400 | 98012 | 16.7 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 3 | 400 | 73466 | 12.2 | yes | 492.9 ± 16.7 | 500.0 ± 0.0 |
| 4 | 200 | 17410 | 4.1 | yes | 500.0 ± 0.0 | 383.6 ± 125.0 |

## Aggregate (greedy eval means across seeds)

- mean: 498.6
- median: 500.0
- min / max: 492.9 / 500.0

Small-sample caveat: these are per-seed point estimates over 5 seeds; the rliable aggregate-metrics protocol (IQM, bootstrap CIs) arrives with the M4 evaluation engine.

## Reproduce

```sh
uv run python benchmarks/m1_reinforce_cartpole.py --seeds 0 1 2 3 4
```

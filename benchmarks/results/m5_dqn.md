# M5 end-to-end implementation validation: DQN

**Scope of this result.** End-to-end *implementation* validation:
evidence that the implementation trains successfully across seeds and
two discrete-action environments. Not evidence of algorithmic
superiority; no cross-algorithm or standard-vs-Double comparison claim
is made (budgets and sample sizes here do not support one).

Evaluation protocol matches the M1/M2 reports (child-seeded train/
eval envs, periodic greedy evaluation with early stop, fresh final
greedy evaluation, independent eps=0.05 evaluation seeded `30000 + seed`). Full configs in the JSON
record; Acrobot uses stop_return=-100, all else default.

## Environment

- rlcore: `0.0.1`
- python: `3.11.15`
- torch: `2.13.0+cu130`
- gymnasium: `1.3.0`
- numpy: `2.4.6`
- hydra: `1.3.4`
- platform: `Linux-6.18.5-x86_64-with-glibc2.39`
- machine: `x86_64`
- device: `cpu`
- cuda_available: `False`
- commit: `13046579b3aa4a39f52c97d7fa87164b4110f017`
- git_dirty: `False`

## Standard DQN — CartPole-v1

| Seed | Env steps | Wall time (s) | Stopped early | Greedy eval (mean±std) | eps=0.05 eval (mean±std) |
|---|---|---|---|---|---|
| 0 | 22500 | 60.0 | yes | 495.2 ± 15.0 | 498.2 ± 7.8 |
| 1 | 60000 | 156.6 | no | 96.3 ± 22.0 | 75.2 ± 37.5 |
| 2 | 60000 | 168.6 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 3 | 20000 | 55.5 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 4 | 40000 | 110.5 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |

Aggregate greedy means: mean 418.3, median 500.0, min/max 96.3/500.0.

## Standard DQN — Acrobot-v1

| Seed | Env steps | Wall time (s) | Stopped early | Greedy eval (mean±std) | eps=0.05 eval (mean±std) |
|---|---|---|---|---|---|
| 0 | 42500 | 147.1 | yes | -85.3 ± 14.8 | -89.7 ± 10.3 |
| 1 | 55000 | 188.7 | yes | -78.9 ± 9.7 | -90.8 ± 21.9 |
| 2 | 35000 | 127.3 | yes | -99.0 ± 28.2 | -100.8 ± 29.7 |
| 3 | 42500 | 158.3 | yes | -74.3 ± 12.2 | -90.2 ± 23.5 |
| 4 | 45000 | 154.5 | yes | -152.5 ± 146.3 | -107.4 ± 29.1 |

Aggregate greedy means: mean -98.0, median -85.3, min/max -152.5/-74.3.

## Double DQN — CartPole-v1 (configuration-extension check, 2 seeds)

| Seed | Env steps | Wall time (s) | Stopped early | Greedy eval (mean±std) | eps=0.05 eval (mean±std) |
|---|---|---|---|---|---|
| 0 | 35000 | 99.7 | yes | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 1 | 60000 | 174.7 | no | 138.8 ± 18.9 | 99.3 ± 50.3 |

Aggregate greedy means: mean 319.4, median 319.4, min/max 138.8/500.0.

Reproducibility scope: pinned software versions above, CPU, comparable hardware only.

## Reproduce

```sh
uv run python benchmarks/m5_dqn.py --seeds 0 1 2 3 4
```

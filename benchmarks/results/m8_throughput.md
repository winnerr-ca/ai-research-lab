# M8 throughput and memory profile

All numbers measured on the hardware recorded below (CPU; see
`m8_throughput.json` for raw data). CartPole-v1 with default network
sizes; vectorized numbers use Gymnasium `SyncVectorEnv` (SAME_STEP
autoreset). No speedup is claimed beyond these measurements.

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
- commit: `3d65e36d5c67167e2cf79c9c14d116269a780c5b`
- git_dirty: `False`
- torch threads: 4

## Measurements (env steps / second)

| Measurement | steps/s |
|---|---|
| Raw `env.step` loop | 115509.7 |
| PPO collection, n_envs=1 | 2964.9 |
| PPO collection, n_envs=2 | 2167.7 |
| PPO collection, n_envs=4 | 5357.7 |
| PPO collection, n_envs=8 | 8694.0 |
| PPO full training, n_envs=1 | 779.5 |
| PPO full training, n_envs=4 | 635.2 |
| DQN full training (update every step) | 331.4 |

Peak RSS: 674.2 MB.

## Reading

- Collection throughput improves with batching only past a break-even
  point on this hardware: n_envs=2 (2167.7 steps/s) is *slower* than
  n_envs=1 (2964.9), while n_envs=4/8 reach 1.8x/2.9x. Env stepping
  itself stays sequential in `SyncVectorEnv`.
- **Vectorization did not accelerate full PPO training at this scale**:
  779.5 steps/s single-env vs 635.2 at n_envs=4 — the optimization
  phase dominates and the vector path adds per-column bookkeeping.
  n_envs is a scaling primitive for env-bound workloads, not a
  recommended CartPole speedup, and this report says so rather than
  claiming one.
- At classic-control scale the workload is Python/env-bound, not
  matmul-bound — the measured profile, not an assumption, is the
  reason no distributed execution layer is added at M8, and why GPU
  execution is plumbed but not expected to pay off at these sizes.

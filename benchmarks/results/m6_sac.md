# M6 end-to-end implementation validation: SAC on Pendulum-v1

**Scope of this result.** End-to-end *implementation* validation on
a single accessible continuous-control task. Not evidence of
algorithmic superiority; no MuJoCo results are claimed (MuJoCo-scale
validation is out of scope on this hardware). The fixed-alpha section
is a configuration-extension check, not a tuned comparison.

Evaluation protocol matches earlier reports (child-seeded train/eval
envs, periodic deterministic evaluation with early stop, fresh final
deterministic evaluation, independent stochastic evaluation seeded
`40000 + seed`). Full configs in the JSON record.

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
- commit: `1bdd52a4e29e26025b510cb7388f598cc94566eb`
- git_dirty: `False`

## SAC (automatic temperature) — Pendulum-v1

| Seed | Env steps | Wall time (s) | Stopped early | Deterministic eval (mean±std) | Stochastic eval (mean±std) | Final alpha |
|---|---|---|---|---|---|---|
| 0 | 6000 | 81.3 | yes | -146.1 ± 45.4 | -135.5 ± 66.1 | 0.297 |
| 1 | 8000 | 110.0 | yes | -129.7 ± 92.6 | -163.5 ± 82.6 | 0.178 |
| 2 | 6000 | 76.8 | yes | -205.7 ± 104.6 | -189.7 ± 92.5 | 0.292 |
| 3 | 6000 | 70.5 | yes | -156.5 ± 107.3 | -178.4 ± 107.3 | 0.288 |
| 4 | 6000 | 66.6 | yes | -155.7 ± 92.1 | -163.6 ± 64.6 | 0.295 |

Aggregate deterministic means: mean -158.8, median -155.7, min/max -205.7/-129.7.

## SAC (fixed alpha = 0.2) — Pendulum-v1 (configuration check, 2 seeds)

| Seed | Env steps | Wall time (s) | Stopped early | Deterministic eval (mean±std) | Stochastic eval (mean±std) | Final alpha |
|---|---|---|---|---|---|---|
| 0 | 4000 | 36.6 | yes | -136.0 ± 99.7 | -205.5 ± 103.6 | 0.200 |
| 1 | 8000 | 103.0 | yes | -129.1 ± 91.0 | -170.5 ± 97.8 | 0.200 |

Aggregate deterministic means: mean -132.6, median -132.6, min/max -136.0/-129.1.

Reproducibility scope: pinned software versions above, CPU, comparable hardware only.

## Reproduce

```sh
uv run python benchmarks/m6_sac.py --seeds 0 1 2 3 4
```

# M8: Vectorized collection, device support, containers, profiling

**Status:** Implemented

## Objective, and why

Make the platform runnable at larger scale without changing its
correctness story: parallel environment collection for PPO, a device
knob (CPU default, CUDA supported with safe fallback), verified CPU
containerization, and a *measured* throughput/memory profile that
justifies what was — and was not — built.

## Deliverables / non-goals / acceptance

Delivered: `rlcore.agents.ppo.vector_rollout` (Gymnasium `SyncVectorEnv`
in `SAME_STEP` autoreset mode, exact per-env boundary handling,
per-column GAE, deterministic child seeding `10 + i`, picklable env
factories for exact resume); `PpoConfig.n_envs`; `rlcore.utils.device`
with `device` config fields on PPO/DQN/SAC; CPU `Dockerfile` (built and
run here — see `benchmarks/results/m8_docker.md`) and documented
unverified `Dockerfile.cuda`; `docs/DOCKER.md` + `docs/CLOUD.md`
(interruption recovery included); `benchmarks/m8_throughput.py` with
recorded results. Non-goals held: no distributed framework, no async
vector envs, no GPU performance claims (no GPU was available to
measure).

Acceptance: vector boundary tests exact on scripted envs; vector
fresh-process resume test; device fallback tests; canonical gate green;
throughput profile recorded from a clean commit.

## The boundary contract under SAME_STEP autoreset

Empirically verified against Gymnasium 1.3 before design: when env *i*
finishes at step *t*, `step()` returns that env's **reset** observation
while the true final observation arrives in `info["final_obs"][i]`.
The collector therefore:

- stores `next_value = 0` for terminated columns,
- stores `next_value = V(final_obs)` for truncated columns,
- fills non-boundary rows with `V(s_{t+1})` and the horizon tail with
  `V(current obs)`,
- computes GAE **per env column** with `done` breaking the recursion,
  so no temporal chain crosses an env boundary or a reset.

Flattening is step-major/env-minor and tested to match the per-column
view element-for-element.

## Devices, honestly

`resolve_device` maps `auto`/`cpu`/`cuda[:N]`; a CUDA request without
CUDA degrades to CPU with a logged warning (queued cloud jobs should
degrade, not die). CPU remains the tested reproducibility bar; no GPU
existed in the development environment, so the CUDA path is exercised
only through its CPU-fallback branch and is documented as such.

## Self-review

- The measured profile (see `m8_throughput.md`) shows collection
  throughput at classic-control scale is Python/env-bound; that
  measurement is the recorded reason no distributed layer was built.
- Vectorized PPO is the only consumer of vector collection; DQN/SAC
  keep single-env collection (their per-step update loop is the
  bottleneck, not the env), per the rule of extracting only for real
  consumers.
- `SyncVectorEnv` pickling for exact resume was verified empirically
  (200-step fidelity across autoresets) before being relied on; the
  env factories use `functools.partial`, not lambdas, for exactly this
  reason.

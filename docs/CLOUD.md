# Running rlcore on cloud machines

rlcore has no cloud-provider integration and deliberately ships none at
v1: a training run is one process reading a config and writing one run
directory, which is exactly the shape every batch service (AWS Batch,
GCP Batch, Slurm, plain VMs) already knows how to schedule. This page
documents the three things that matter when you do that.

## 1. Use the containers

Build the CPU image or CUDA image (both documented; neither could be
built in the development environment — see `docs/DOCKER.md`) and run it with your scheduler's volume mount for
`outputs/`. All state a run needs to survive preemption lives in its run
directory.

## 2. Devices, honestly

- `algo.device=cpu` (default): the platform's tested reproducibility
  bar. All recorded validations and benchmarks in this repository were
  produced on CPU with pinned versions.
- `algo.device=cuda`: supported code path with safe CPU fallback when
  CUDA is absent. PyTorch does not promise bitwise-identical results
  across devices or driver versions, so treat cross-device comparisons
  accordingly: compare *statistics across seeds*, never single-seed
  curves, and record the device in every report (`run.json` does this
  automatically).
- At classic-control scale the measured profile
  (`benchmarks/results/m8_throughput.md`) shows the workload is
  Python/environment-bound, not matmul-bound — a GPU is not expected to
  pay off below roughly MuJoCo/pixel scale. Measure before paying.

## 3. Preemption and resume

Set `algo.checkpoint_every` and resubmit the identical command plus
`resume_from=<run_dir>/checkpoint.pt` on restart. The resume contract
(exact continuation, config-mismatch rejection, budget extension as the
only allowed change) is enforced by `rlcore.checkpoints` and covered by
fresh-process tests per algorithm.

## What is deliberately absent

No distributed training, no multi-node orchestration, no cloud SDK
wrappers. The M8 throughput profile is the recorded justification: at
the environment scales this platform validates on, a second process
buys more than a second GPU, and either is a plain scheduler concern.
If a future workload measures differently, that measurement — not a
framework fashion — should drive the addition (ROADMAP, deferred list).

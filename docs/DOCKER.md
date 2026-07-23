# Running rlcore in Docker

Two images are defined at the repository root:

| File | Base | Status |
|---|---|---|
| `Dockerfile` | `python:3.11-slim` (CPU) | **Documented, not yet verified** — the development environment's network policy denies all container-registry blob downloads (Docker Hub, ECR Public, and GHCR were each attempted; see `benchmarks/results/m8_docker.md` for the recorded denials). Build it on a host with registry access. |
| `Dockerfile.cuda` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | **Documented, not verified** — the development environment has no GPU, so this build follows the standard pattern but has not been executed. Validate on a GPU host before relying on it. |

## CPU image

```sh
docker build -t rlcore:cpu .
docker run --rm -v "$(pwd)/outputs:/app/outputs" rlcore:cpu \
    algo=ppo algo.total_updates=5 out_dir=outputs/docker-demo
```

The entrypoint is `rlcore-train` (Hydra CLI); any config override works
exactly as outside the container. Mount `./outputs` to keep run
directories, checkpoints, and plots on the host.

### Interruption recovery

Checkpoints are ordinary files in the mounted run directory, so a killed
container resumes the same way a killed process does:

```sh
docker run --rm -v "$(pwd)/outputs:/app/outputs" rlcore:cpu \
    algo=dqn algo.checkpoint_every=5000 out_dir=outputs/dqn-run
# ... container dies (spot reclaim, OOM kill, ^C) ...
docker run --rm -v "$(pwd)/outputs:/app/outputs" rlcore:cpu \
    algo=dqn algo.checkpoint_every=5000 out_dir=outputs/dqn-run \
    resume_from=outputs/dqn-run/checkpoint.pt
```

Resume is exact (same seeds, same replay contents, fresh-process-tested
at the algorithm level); config changes other than extending the budget
are rejected, so a stale checkpoint cannot silently train with different
hyperparameters.

## CUDA image (unverified build)

```sh
docker build -f Dockerfile.cuda -t rlcore:cuda .
docker run --rm --gpus all -v "$(pwd)/outputs:/app/outputs" \
    rlcore:cuda algo=sac algo.device=cuda
```

`algo.device=cuda` falls back to CPU with a logged warning if the
container sees no GPU (`rlcore.utils.device`), so a mis-scheduled job
degrades instead of dying. See `docs/CLOUD.md` for the scope of the
platform's determinism guarantee off-CPU.

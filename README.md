# rlcore — Reinforcement Learning Research Platform

A reproducibility-first platform for reinforcement learning research:
algorithms implemented from scratch with visible math, trained on
standard environments, compared with honest statistics, and recorded so
every number traces back to a commit, a config, and a seed.

**Version 1.0.0** — Python 3.11 · PyTorch · Gymnasium · Apache-2.0

## What's inside

| | |
|---|---|
| **Algorithms** | REINFORCE, PPO (single + vectorized envs), DQN (+ Double DQN), SAC (auto/fixed temperature) — each with pure unit-tested loss functions, exact terminated/truncated handling, deterministic seeding, and exact checkpoint resume |
| **Experimentation** | Run directories (config/metrics/result/model/plots), Hydra CLI, local tracking always + optional W&B/MLflow, manifest-driven benchmarks with fixed budgets and robust statistics (IQM, bootstrap CIs, probability of improvement) |
| **Scale-out** | Vectorized collection, `device=cpu/cuda/auto` with safe fallback, CPU/CUDA Dockerfiles (documented, build-blocked in the dev environment), measured throughput profile |
| **Research layer** | Typed versioned research entities (papers → claims → hypotheses → plans → runs → analyses → conclusions → reports), source-grounded citations, five human approval gates, provider-neutral LLM interface (offline mock included) |

## Install

```sh
git clone <this-repo> && cd ai-research-lab
uv sync                  # core
uv sync --all-extras     # + trackers, research layer, SB3 parity, dev tools
```

## Train something

```sh
uv run rlcore-train algo=ppo                          # PPO on CartPole-v1
uv run rlcore-train algo=sac algo.seed=3              # SAC on Pendulum-v1
uv run rlcore-train algo=dqn algo.double_q=true       # Double DQN
uv run rlcore-train algo=ppo algo.n_envs=4            # vectorized collection
```

Every run writes a self-describing directory under `outputs/`:
`run.json` (identity + full environment metadata), `config.yaml`,
`metrics.jsonl`, `result.json`, `final_model.pt`, `summary.md`, and a
learning-curve plot. Evaluate a finished run with
`uv run rlcore-evaluate <run_dir>`.

## Compare things fairly

```sh
uv run rlcore-benchmark benchmarks/manifests/m7-manifest.json
```

Manifests fix budgets (early stopping is rejected by the schema), runs
execute in isolated subprocesses with timeouts, failures are recorded
as results, and interrupted benchmarks resume. Reports aggregate with
IQM, bootstrap CIs, and probability of improvement — with small-sample
caveats stated, not hidden.

## Learn the platform

Start with the tutorials in [`examples/`](examples/) (each one is
tested in CI):

1. `01_train_and_evaluate.py` — trains an agent, tours the run directory
2. `02_resume_training.py` — proves resume-from-checkpoint is *exact*
3. `03_compare_with_statistics.py` — the multi-seed comparison protocol
4. `04_research_workflow.py` — grounded claims, real runs, approval gates

## Documents

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design principles, system architecture, testing strategy |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestone history and deferred work |
| [`docs/algorithms/`](docs/algorithms/) | Per-algorithm math, derivations, and caveats |
| [`docs/design/`](docs/design/) | Per-milestone design records (decisions + self-reviews) |
| [`docs/reproductions/`](docs/reproductions/) | Honestly-labeled reproduction packages |
| [`docs/research.md`](docs/research.md) | The research layer and its approval gates |
| [`docs/DOCKER.md`](docs/DOCKER.md) / [`docs/CLOUD.md`](docs/CLOUD.md) | Containers, devices, preemption recovery |
| [`benchmarks/results/`](benchmarks/results/) | Recorded validations — raw JSON + protocol |

## Principles (the short version)

- **Truncation is not termination.** No bootstrap at true termination;
  bootstrap from the final observation at truncation; no temporal chain
  crosses either boundary. Unit-tested with hand-computed cases.
- **Numbers come from runs.** Configs, seeds, versions, commit, device,
  and protocol are recorded with every result; failed seeds are
  reported, not dropped.
- **Claims match evidence.** Multi-seed, equal budgets, robust
  statistics — or it's an engineering observation, clearly labeled.
- **Math stays visible.** Losses and estimators are pure functions in
  each algorithm's directory; shared code is extracted only after
  multiple algorithms genuinely need it.
- **Humans hold the gates.** Expensive, paid, published, destructive,
  and hypothesis-to-conclusion actions all require an explicit recorded
  human approval.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All project spaces follow the
[Code of Conduct](CODE_OF_CONDUCT.md); security policy in
[SECURITY.md](SECURITY.md). Cite via [CITATION.cff](CITATION.cff).

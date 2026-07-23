# rlcore — Reinforcement Learning Research Platform

A modular, open-source platform for reinforcement learning research: implement
algorithms from scratch, train them on standard environments, compare them
fairly, and reproduce published results.

> **Status: Phase 0 (scaffolding) in progress.** The architecture and roadmap
> are approved — start with the documents below.

The platform has two connected layers: an **execution layer** (environments,
agents, training, buffers, checkpoints, tracking, statistical comparison —
milestones 0–7) and a **research intelligence layer** built on top of it
(paper index, research memory, hypothesis and experiment-plan registry,
evidence-linked reports, LLM-assisted analysis — milestones 8–9). See
`docs/ARCHITECTURE.md` §1.1 for how the layers connect.

## Documents

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design principles, prior-art review, system architecture, component specifications, repository layout, testing strategy, risks |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones, phase gates, acceptance criteria, and the detailed plan for Phase 0 |

## Approach

Existing RL codebases resolve the tension between single-file clarity and
reusable infrastructure in different ways, each well suited to its own goals.
This project aims at a specific point in that space: **legible algorithm
code on top of shared, tested infrastructure**, with reproducibility
(seeds, configs, complete checkpoints), numerical verification against
reference implementations, and multi-seed statistical evaluation treated as
core features. Shared abstractions are extracted from working algorithm
implementations rather than designed up front.

## Usage

```sh
# Train (algorithm is a Hydra config group; any field is overridable)
rlcore-train algo=ppo
rlcore-train algo=reinforce algo.seed=3 algo.lr=0.005

# Re-evaluate a finished run from its run directory
rlcore-evaluate outputs/<date>/<time>/ --episodes 50

# Checkpoint every 10 updates; resume later with a larger budget
rlcore-train algo=ppo algo.checkpoint_every=10
rlcore-train algo=ppo algo.total_updates=150 resume_from=<run_dir>/checkpoint.pt
```

Every run writes a self-describing local run directory (identity, resolved
config, git/version/device provenance, JSONL metrics, final model,
summary + learning-curve plot) — no external account required. See
`docs/design/m4a-local-tracking.md` for the layout.

## Planned stack

Python · PyTorch · Gymnasium · Hydra · Weights & Biases (adapter-based, MLflow
compatible) · Docker · Ruff · mypy · pytest · Stable-Baselines3 (verification
only, never as the primary implementation).

## Development process

Work proceeds in reviewed phases. Each phase states its objective, designs
before coding, ships tested and type-checked code, and passes a review gate
before the next phase begins. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

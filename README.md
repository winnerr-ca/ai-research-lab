# rlcore — Reinforcement Learning Research Platform

A modular, open-source platform for reinforcement learning research: implement
algorithms from scratch, train them on standard environments, compare them
fairly, and reproduce published results.

> **Status: design phase.** The overall vision is approved and the design is
> in revision. No implementation code exists yet — start with the documents
> below.

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

## Planned stack

Python · PyTorch · Gymnasium · Hydra · Weights & Biases (adapter-based, MLflow
compatible) · Docker · Ruff · mypy · pytest · Stable-Baselines3 (verification
only, never as the primary implementation).

## Development process

Work proceeds in reviewed phases. Each phase states its objective, designs
before coding, ships tested and type-checked code, and passes a review gate
before the next phase begins. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

# rlab — Reinforcement Learning Research Platform

A modular, open-source platform for reinforcement learning research: implement
algorithms from scratch, train them on standard environments, compare them
fairly, and reproduce published results — with the engineering rigor of a
production codebase.

> **Status: design phase.** No implementation code exists yet. The
> architecture is under review — start with the documents below.

## Documents

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design principles, prior-art analysis, system architecture, component specifications, repository layout, testing strategy, risks |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones, phase gates, acceptance criteria, and the detailed plan for Phase 0 |

## Vision

Most RL codebases force a choice: single-file clarity (CleanRL) or reusable
infrastructure (SB3, RLlib) — readable *or* extensible, rarely both. This
platform aims for a third point: **algorithms that read top-to-bottom, on top
of infrastructure that is factored out once and trusted**, with verification
against reference implementations as a first-class test category.

## Planned stack

Python · PyTorch · Gymnasium · Hydra · Weights & Biases (adapter-based, MLflow
compatible) · Docker · Ruff · mypy · pytest · Stable-Baselines3 (verification
only, never as the primary implementation).

## Development process

Work proceeds in reviewed phases. Each phase states its objective, designs
before coding, ships production-quality code with tests, and passes a review
gate before the next phase begins. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

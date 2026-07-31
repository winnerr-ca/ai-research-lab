# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Lab dashboard** (`rlcore-lab`): a localhost, dependency-free web
  interface for testing research on the platform — run browsing over
  standard run directories, learning-curve overlays from
  `metrics.jsonl`, grouped comparisons via `rlcore.stats` (fixed
  bootstrap seed; small-sample/unequal-budget/repeated-seed caveats
  stated; excluded runs listed with reasons), subprocess-isolated run
  launching with request validation and stderr capture, automatic
  discovery of algorithm packages under `rlcore.agents` (user-added
  variants included), and a read-only research-store listing. Gated
  research actions remain unreachable over HTTP. Documented in
  `docs/lab.md`.

## [1.0.0] - 2026-07-23

First stable release. Everything below was built and validated across
milestones M0–M10; per-milestone design records live in `docs/design/`
and recorded validations in `benchmarks/results/`.

### Added

- **Algorithms** (each with visible math in pure, unit-tested loss
  functions, exact terminated/truncated handling, deterministic
  seeding, checkpoint/exact-resume, run directories, and multi-seed
  validation records): REINFORCE, PPO (single and vectorized
  collection), DQN (+ Double DQN configuration), SAC (auto and fixed
  temperature).
- **Infrastructure**: typed immutable `Batch`; seed bundle with
  documented child-seed layout; replay buffer with full save/restore;
  GAE with per-boundary bootstrap contract; evaluation protocol with
  independent eval seeding; atomic checkpoint envelope with
  config-mismatch rejection; standard run directories (config, metrics,
  result, model, summary, plots); Hydra CLI (`rlcore-train`,
  `rlcore-evaluate`).
- **Experiment tracking**: local JSONL always; optional W&B (offline
  default) and MLflow (local sqlite default) adapters — no credentials
  ever required.
- **Benchmarking**: manifest-driven multi-algorithm/env/seed execution
  with fixed budgets, per-run subprocess isolation and timeouts,
  failed/timeout recording, resumability (`rlcore-benchmark`);
  statistics module (IQM, percentile-bootstrap CIs, probability of
  improvement, performance profiles) with hand-computed test anchors;
  honestly-labeled reproduction packages for PPO/DQN/SAC.
- **Scale-out**: Gymnasium `SyncVectorEnv` collection (SAME_STEP
  autoreset) with exact per-env boundary handling and per-column GAE;
  `device` config with safe CUDA→CPU fallback; measured throughput and
  memory profile; CPU and CUDA Dockerfiles (documented; container
  builds were blocked by the development environment's network
  policy — see benchmarks/results/m8_docker.md);
  interruption-recovery documentation.
- **Research layer**: 14 typed, versioned entities with provenance and
  human-review state; JSON store with full version history;
  Markdown/text/PDF ingestion preserving sources; deterministic lexical
  retrieval with section/page references; grounding rules (claims cite
  sources or runs; conclusions require run-grounded analyses);
  provider-neutral LLM interface (deterministic mock + Anthropic
  adapter, keys from environment only); five human approval gates
  (expensive experiments, paid services, publication, destructive
  deletion, hypothesis→conclusion promotion); `rlcore-research` CLI.
- **Quality**: strict mypy across `src`/`tests`/`benchmarks`; Ruff
  lint+format; fast deterministic test suite plus opt-in slow learning
  canaries; SB3 numerical parity tests where a genuinely equivalent
  isolated reference exists; CI running the full gate plus build and
  install-smoke jobs.

### Security

- Checkpoint loading documented as a pickle trust boundary
  (`SECURITY.md`).

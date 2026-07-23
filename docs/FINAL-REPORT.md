# rlcore Version 1.0 — Final Report

**Date:** 2026-07-23 · **Branch:** `claude/rl-research-platform-rlh70h` · **Version:** 1.0.0 · **License:** Apache-2.0

## Executive summary (plain language)

rlcore is a working reinforcement-learning research platform built from
an empty repository in ten gated milestones. It trains four algorithms
(REINFORCE, PPO, DQN, SAC) with mathematics kept visible and unit-tested
against hand-computed cases; it records every run so any number traces
back to a commit, a config, a seed, and pinned software versions; it can
stop a run at any point and resume it *exactly*, even in a fresh
process; it compares algorithms only under fixed, equal budgets with
robust statistics that state their own uncertainty; and it carries a
research-notebook layer where claims must cite sources or real runs and
five kinds of consequential action (expensive experiments, paid LLM
calls, publication, deletion, promoting a hypothesis to a conclusion)
always require a recorded human approval.

The project's defining habit is that failures are part of the record:
failed seeds are published next to successful ones, a benchmark cell
that crashes is a first-class result, and the two things this
environment could not verify (container image builds; GPU execution)
are documented as blocked with evidence rather than claimed as done.

## Milestone commit IDs

| Milestone | Implementation | Validation/records |
|---|---|---|
| M0–M4C (foundation → trackers) | `9ba358b` … `c838236` | in-history records |
| M5 DQN | `1304657` | `5724f4a` |
| M6 SAC | `1bdd52a` | `c008f2f` |
| M7 Benchmarks | `708de42` | `507da13` |
| M8 Scale-out | `3d65e36` | `97ff73e` |
| M9 Research layer | `e46dabe` | `adad3f0` |
| M10 Release prep | `138fd36` | this commit (final records) |

## Final architecture

Two layers (docs/ARCHITECTURE.md):

- **Execution layer** — `rlcore.types.Batch` (immutable tensor
  container); `rlcore.utils.seeding` (root seed → documented child-seed
  layout: 0 train env, 1 action space, 2 eval env, 10+i vector envs);
  `rlcore.envs` (discrete + continuous env pairs); per-algorithm
  packages under `rlcore.agents.*` each holding pure loss/estimator
  functions, model, and trainer; `rlcore.replay` (shared by DQN and
  SAC — extracted on the rule of two); `rlcore.checkpoints` (versioned
  envelope, atomic writes, config-mismatch rejection, NaN-aware field
  comparison); `rlcore.experiments`/`reporting`/`tracking` (run
  directories, summaries, optional W&B/MLflow mirrors);
  `rlcore.evaluation` (independent child-seeded evaluation);
  `rlcore.stats`/`benchmark`/`benchreport` (fixed-budget manifests,
  subprocess isolation, robust statistics); `rlcore.utils.device`
  (cpu/cuda/auto with safe fallback).
- **Research layer** — `rlcore.research`: 14 typed versioned entities,
  version-history store, source-preserving ingestion, lexical
  retrieval with section/page citations, provider-neutral LLM
  interface (deterministic mock + Anthropic adapter), grounding rules,
  five human approval gates.

## Repository structure

```
src/rlcore/            # 46 Python modules (agents/, research/, utils/, core)
tests/                 # 36 test files: unit/ (fast, deterministic) + parity/ + slow canaries
benchmarks/            # validation drivers, manifests/, results/ (committed records)
docs/                  # ARCHITECTURE, ROADMAP, algorithms/, design/ (10 records),
                       # reproductions/, research.md, DOCKER.md, CLOUD.md
examples/              # 4 tested tutorials
.github/               # CI (gate + build/install-smoke + docs), templates
Dockerfile[.cuda], mkdocs.yml, CHANGELOG.md, CITATION.cff, CONTRIBUTING.md,
CODE_OF_CONDUCT.md, SECURITY.md, LICENSE (Apache-2.0)
```

## Implemented algorithms

REINFORCE (return standardization as a documented configurable
heuristic); PPO (GAE with exact boundary contract, clipped surrogate,
KL monitoring/early epoch stop, advantage normalization heuristic,
single-env and SyncVectorEnv collection); DQN (replay, target network,
epsilon-greedy warm-up, Huber TD, Double DQN configuration); SAC
(tanh-Gaussian actor with exact change-of-variables log-probability
cross-checked against torch.distributions, twin critics, Polyak
averaging verified against SB3's isolated equivalent, automatic and
fixed temperature). All: terminated/truncated distinction enforced and
hand-tested; deterministic seeding; exact resume incl. fresh-process
tests; standard run directories.

## Test counts and commands

- **380 tests**: 376 fast/deterministic (default) + 4 slow learning
  canaries (`-m slow`), all green at `138fd36`; strict mypy over 93
  files (`src`, `tests`, `benchmarks`, `examples`).
- Canonical gate: `uv sync --all-extras && uv run ruff check . && uv
  run ruff format --check . && uv run mypy && uv run pytest`
- Slow suite: `uv run pytest -m slow`

## Validation environments, seeds, raw results

| Record (benchmarks/results/) | What it holds |
|---|---|
| `m1_reinforce_cartpole.*`, `m2_ppo_cartpole.*` | 5-seed CartPole validations (M1/M2 protocol) |
| `m5_dqn.*` | DQN CartPole seeds 0–4 (**4/5 solved; seed 1 failed at 96.3 — reported**), Acrobot 5/5, Double DQN 2-seed config check (1/2) |
| `m6_sac.*` | SAC Pendulum auto-α seeds 0–4 (**4/5 ≥ −180; seed 2 missed at −205.7 — reported**), fixed-α 2-seed check |
| `m7_benchmark.*` + `plots/` | 12/12 fixed-budget cells (REINFORCE/PPO/DQN CartPole ×3, SAC Pendulum ×3), IQM/CI/P(improvement) |
| `m8_throughput.*` | measured CPU profile (incl. the honest negative: vectorization is **not** a full-training speedup at this scale) |
| `m8_docker.md` | recorded container-registry denials (see Docker verification) |
| `m9_research_demo.*` | research-layer end-to-end demo, real 2×2 ablation, gates provably enforced |
| `v1_demonstration.*` | the 20-item final demonstration record |

## Benchmark summaries

M7 (`m7_benchmark.md`, commit `708de42`, all cells ok): CartPole —
PPO 500.0/500.0/500.0, REINFORCE 500.0/500.0/500.0, DQN 126.4–177.1 at
its halved 30k budget; Pendulum — SAC IQM −170.4 [−189.4, −155.0].
Budgets are per-algorithm as declared; no cross-algorithm superiority
claims are made and the report says so.

## Reproduction status

`docs/reproductions/repro-{ppo,dqn,sac}.md` are labeled
**implementation validation** — expected behavior reached on accessible
classic-control environments under this repository's recorded protocol.
No package claims partial/approximate/full *paper* reproduction,
because no published protocol (environment suite, budget, evaluation)
was matched; those labels are defined and reserved.

## Checkpoint and resume guarantees

Exact continuation on pinned-version CPU: model/optimizer/RNG-stream/
env-state (pickled)/collector/replay/temperature state all restored;
resume rejects any config change except extending the budget field and
`checkpoint_every`; atomic writes; per-algorithm 2N ≡ N+save+fresh-
process-resume+N tests (histories and weights bit-equal). Loading a
checkpoint is a documented pickle trust boundary (SECURITY.md).

## Research-intelligence demonstration

`m9_research_demo.*` and items 12–17 of `v1_demonstration.*`: real
document ingestion with SHA-256-preserved source and section refs;
retrieval-grounded claims; a real 2-seed × 2-variant PPO ablation at
equal fixed budgets with run-grounded observations/analyses; the
conclude and publish gates raising without approval and succeeding
with a recorded one; analysis explicitly covering a failed run; and
the recorded conclusion confined to the pipeline mechanism (n=2 at 3
updates supports no performance claim).

## Docker verification

**Not verified — blocked, with evidence.** The Docker daemon runs in
the development environment, but its egress policy denies every
container-registry blob CDN (Docker Hub, ECR Public, GHCR — each
attempted; denials recorded in `benchmarks/results/m8_docker.md`).
Both Dockerfiles are documented, standard, and unverified here; build
on any registry-connected host (`docker build -t rlcore:cpu .`). The
CUDA image additionally awaits a GPU host.

## Installation and usage

```sh
uv sync --all-extras                                  # dev setup
uv run pytest                                         # fast suite
uv run rlcore-train algo=ppo                          # train (Hydra overrides)
uv run rlcore-train algo=dqn algo.double_q=true
uv run rlcore-train algo=ppo algo.n_envs=4 algo.device=auto
uv run rlcore-train algo=sac resume_from=<run>/checkpoint.pt
uv run rlcore-evaluate <run_dir>                      # independent evaluation
uv run rlcore-benchmark benchmarks/manifests/m7-manifest.json
uv run rlcore-research ingest docs/algorithms/sac.md  # research layer
uv build                                              # wheel + sdist
uv run --extra docs mkdocs build --strict             # docs site
```

## Known limitations

1. Validation is classic-control scale; no MuJoCo/Atari results exist
   and none are claimed.
2. Reproducibility guarantees are scoped to pinned versions on CPU;
   CUDA support is code-complete with safe fallback but **never
   executed on a GPU** — all cross-device behavior is untested.
3. Container builds unverified here (network policy; evidence
   recorded).
4. Statistics at 3–5 seeds are honest uncertainty indicators, not
   significance tests, and reports say so.
5. Retrieval is lexical, not semantic; scanned PDFs without text
   layers yield no sections.
6. The research store is single-process (no locking).
7. Bootstrap intervals tend toward under-coverage at these sample
   sizes (documented in `rlcore.stats`).
8. Checkpoints are pickle archives — load only trusted files.
9. Pre-M8 checkpoints do not resume under the M8+ config schema
   (config-mismatch rejection; by design, documented here).
10. Old-style vector-env resume requires the same `n_envs`.

## Deferred to Version 2.0

MuJoCo/Atari validation and true paper reproductions; GPU-verified
CUDA path and measured GPU profiles; async/distributed collection *if
profiling on larger workloads justifies it*; prioritized replay,
dueling/distributional DQN, TD3, model-based methods; semantic
retrieval layered over the citation contract; multi-task normalized
benchmark scores; research-store locking/multi-user; docs hosting;
action-sampling unification (still deliberately un-unified).

## Requires explicit user authorization (not performed)

PyPI publication; public GitHub release/tag; enabling paid LLM
providers anywhere; any cloud spend; force-pushes or history rewrites.

## Release readiness

The codebase is **release-ready**: version 1.0.0, gate green (380
tests, strict types), wheel builds and installs cleanly with working
entry points, docs build strict, community files complete, CI covers
gate + packaging + docs. It is **not released**: publication is a
human decision this project's own publication gate exists to protect.
Recommended pre-release human steps: verify the Docker build on a
networked host, run the CUDA path on a GPU host, and replace the
placeholder author metadata in `CITATION.cff`.

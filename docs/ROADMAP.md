# Roadmap

**Status:** Revision 2 — vision approved; incorporating review feedback
**Companion:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

**Changes from revision 1 (review feedback):** milestones reordered so the
first algorithms are built *before* the abstraction system — shared
infrastructure is now extracted from working algorithm code at a dedicated
consolidation milestone, instead of being built speculatively up front.
Phase 0 is cut to the minimum that later phases genuinely depend on. GAE
test references corrected. Package renamed to `rlcore`.

Milestones are gated by acceptance criteria, not calendar dates. Each phase
ends with a review; the next phase does not start until the review passes.

## Ordering rationale

Two rules drive the sequence:

1. **Working algorithms before abstractions.** An interface designed before
   its consumers exist is a guess. We build REINFORCE and PPO first — with
   plumbing inlined where that is clearer — then extract the environment
   layer, collector, network library, and training engine from two working
   implementations at a consolidation milestone. The off-policy pass repeats
   this: the replay buffer is built when DQN needs it, not before.
2. **Verification travels with the algorithm.** Every algorithm milestone
   includes its SB3 loss-parity tests, invariant tests, and multi-seed
   learning verification as part of the milestone, not as a later phase.

## Milestone map

| # | Milestone | Delivers | Depends on |
|---|---|---|---|
| 0 | **Minimal scaffolding** | Package skeleton, lint/type/test toolchain, CI, `Batch` v1, central seeding | — |
| 1 | **Vertical slice: REINFORCE** | End-to-end training on CartPole: minimal env setup, small policy net, simple collector, train script with minimal Hydra config; plumbing deliberately inline | 0 |
| 2 | **PPO + verification** | GAE (with corrected invariant tests), clipped objective, minibatch epochs; SB3 loss parity; multi-seed classic-control verification | 1 |
| 3 | **Consolidation audit** | Duplication audit over the two working algorithms (`docs/design/m3-consolidation-audit.md`); extractions the audit justified: env creation + seed layout, evaluation, run outputs/metadata, small network helpers. Collectors and training loops stayed local (audited: no stable common structure); action-sampling unification deferred (bitwise RNG preservation); wrappers/vectorization/normalization deferred to the milestone that needs them | 2 |
| 4A | **Experiment config & local tracking** | Compact Hydra config with a unified `rlcore-train` entry point and an `rlcore-evaluate` counterpart; unique run identity; resolved config, git commit + dirty state, package/device/OS metadata; standard run-directory structure (run record, JSONL metrics, final-model artifact, summary + learning-curve plot); fully local, no external account | 3 |
| 4B | **Checkpoint & resume** | Checkpoint contract for the *existing* REINFORCE/PPO state only: model, optimizer, progress counters, RNG states, config, run identity, collector/episode state; atomic writes, schema versioning, clear validation errors; gate test: continuous 2N ≡ N + save + fresh-process restore + N on pinned CPU; env-state boundary documented | 4A |
| 4C | **Optional tracker adapters** | W&B and MLflow behind a `Tracker` protocol as optional dependencies; offline/local backends by default; credentials never required for normal operation; the same train command verified to work local-only | 4B |
| 5 | **DQN + off-policy infrastructure** | Replay buffer (shared module; SAC is the scheduled second consumer), Q/target networks, epsilon-greedy with warm-up, minibatch TD updates with configurable frequencies, Double DQN as a config extension, checkpoint incl. replay state, multi-seed validation on two discrete envs | 4C |
| 6 | **SAC + continuous-action infrastructure** | Tanh-squashed Gaussian actor with exact change-of-variables log-prob, twin Q + Polyak targets, automatic entropy tuning (fixed-alpha mode configurable), replay reuse, checkpoint/resume, multi-seed continuous-control validation | 5 |
| 7 | **Benchmark & reproduction harness** | Manifest-driven multi-algo/env/seed runner with fixed budgets, failure/timeout recording, and resumability; native aggregate statistics (IQM, bootstrap CIs, probability of improvement, performance profiles) behind a small tested adapter; documented reproduction packages with honest labels | 6 |
| 8 | **Performance, vectorization, Docker, cloud readiness** | Vector-env collection where compatible (correct boundaries, final-obs handling, deterministic child seeding), device resolution with safe CUDA fallback, measured throughput/memory profiles, CPU Docker image + documented CUDA build, interruption-recovery and cloud guidance | 7 |
| 9 | **Research-intelligence layer** | Typed, versioned research entities with provenance and review states; source-preserving ingestion and grounded retrieval; provider-neutral LLM interface with deterministic mock; citation-enforced reports; approval gates; plan↔run linking with missing/failed-run detection | 8 |
| 10 | **Documentation & v1.0 release preparation** | Complete documentation set, tested tutorials, community/release files, CI extended to build/install/docs checks, final end-to-end demonstration | 9 |

Notes:

- **Why REINFORCE first (M1):** it is the smallest end-to-end integration
  test of the entire pipeline — collect → returns → policy gradient →
  update — with the fewest moving parts, and it has real pedagogical value.
  PPO (M2) then adds GAE, clipping, and minibatch epochs on a pipeline that
  demonstrably works.
- **Minimal config/logging arrives with M1** (one Hydra config, console
  logging, a plain run directory), because a training script needs them.
  M4 is where experiment infrastructure becomes *complete* — trackers, the
  full checkpoint contract, rliable evaluation — not where it begins.
- **M2 verification uses plain scripts** for multi-seed stats in
  `benchmarks/`; reusable aggregate-evaluation tooling is deferred to M6
  (see the rliable note below).
- **Research mode applies from M1 onward:** every algorithm ships with its
  mathematical derivation, known weaknesses, comparison to alternatives, and
  an implementation-detail review against the literature (e.g. the PPO
  detail catalogue of Huang et al., ICLR Blog Track 2022).
- **M8–M9 are the researcher layer** (ARCHITECTURE §1.1): M0–M7 build the
  execution layer — the experimental engine — and M8–M9 add the research
  intelligence layer on top of it. They sit last not because they matter
  least, but because their design should be extracted from real research
  workflows run on the finished engine, same as the code abstractions. Two
  seams are grown early on their behalf: the run registry (every run
  attributable and citable, from M1's first run directory onward) and typed
  research artifacts at M8. Scope note, stated honestly: LLM-assisted
  components support literature work, critique, and analysis under strict
  citation discipline (claims must link to run IDs or references); hypothesis
  generation and gap identification remain human-led with tool support.

---

## Phase 0 (Milestone 0): Minimal scaffolding

### 1. Objective

Stand up the smallest repository skeleton that later phases genuinely depend
on: the installable `rlcore` package, the lint/type/test toolchain with CI,
the `Batch` type (v1), and central seeding.

### 2. Why it matters

- Tooling is cheap on commit one and expensive to retrofit. Ruff, mypy, and
  pytest in CI make "every file tested and type-hinted" something the repo
  *enforces* rather than something we intend.
- `Batch` and seeding are imported by every later module; their contracts are
  the only design work worth doing before algorithm code exists.
- Everything else scaffolding-ish (Docker, pre-commit, config tree, run
  directories, entry points) is deferred to the milestone that needs it —
  per review feedback, Phase 0 carries no speculative deliverables.

### 3. Deliverables

| Deliverable | Content |
|---|---|
| `pyproject.toml` | `uv`-managed; hatchling build; runtime deps: torch, gymnasium, numpy; extras: `dev` (ruff, mypy, pytest), `parity` (stable-baselines3, test-only) |
| `src/rlcore/` | `types.py` — `Batch` v1 (construction/validation, attribute access, `.to(device)`, `len`, slicing, minibatch iteration) and `PolicyOutput`; `utils/seeding.py` — one entry point fanning out to torch/numpy/`random`, returning seeded generators for env use |
| `tests/` | Unit tests for `Batch` (including error cases: mismatched lengths, bad keys); seeding determinism test (same seed → identical draws twice); `slow` marker registered for later smoke tests |
| `.github/workflows/ci.yml` | Single job: `ruff check` + `ruff format --check` → `mypy` → `pytest`; Python 3.11, CPU-only torch wheel, uv cache |

Explicitly **cut from Phase 0** (deferred to the milestone that needs them):
Docker (M6), pre-commit config (optional, any time), Hydra config tree and
entry point (M1), JSONL/run-directory machinery (M1), tracker integration
(M4), multi-version CI matrix (when a second Python version matters).

### 4. Risks specific to this phase

- **Over-building `Batch` before real usage** — mitigation: v1 implements
  only what its own tests and M1 need; M2/M3 reviews are the scheduled
  points to extend it against real collector usage.
- **Tooling bikeshedding** — mitigation: toolchain choices were ratified in
  the revision-1 review (ARCHITECTURE §10); changes require an ADR.
- **CI flakiness from torch install times** — mitigation: uv lockfile + CI
  cache; CPU-only wheel.

### 5. Acceptance criteria (review gate)

1. Fresh clone → `uv sync --all-extras && ruff check . && mypy && pytest`
   passes locally and in CI.
2. `import rlcore` works from the installed package.
3. Seeding determinism test passes: identical seeds → identical torch/numpy
   draws and identical `Batch` contents across two seeded contexts.
4. No deliverable outside the table above (the review checks for scope creep
   as well as for gaps).

---

## Later-phase sketches (bind at their own reviews)

- **M2 verification protocol:** SB3 PPO loss parity on synthetic batches
  (exact); GAE invariant tests — λ=0 reduces to the one-step TD residual,
  λ=1 recovers the discounted Monte-Carlo return *minus the value baseline*
  (equivalently, advantage + value reproduces the MC return), with
  truncation bootstrapping covered; then CartPole-v1 / Acrobot-v1 /
  LunarLander-v2 runs across ≥5 seeds compared against SB3 under identical
  eval protocols (rliable CIs).
- **M4B checkpoint gate:** the resume-equals-continuous test from
  ARCHITECTURE §6.9 — 2N continuous vs. N + save + restore-in-fresh-process
  + N, identical CPU weights, scoped to the state REINFORCE and PPO
  actually have (no replay buffers, target networks, or SAC temperature
  until the algorithms that own them exist).
- **rliable deferral:** aggregate-metrics tooling waits for M6, when
  multiple algorithms and environments make aggregate statistics
  meaningful; because rliable's upstream repository is archived, any use
  goes behind a small, tested adapter with a pinned dependency.
- **Action sampling stays un-unified through M4** (audit entry 3): the
  REINFORCE and PPO sampling paths are preserved verbatim unless a genuine
  consumer requires a shared implementation *and* a full revalidation is
  scheduled.
- **M6 reproduction targets (proposal):** PPO on MuJoCo locomotion vs.
  published reference results; DQN on a 3–5 game Atari subset vs. published
  scores. Full-suite Atari is out of scope until compute is budgeted.
- **M7 scale path:** vectorized-env throughput benchmarks first; distributed
  collection only if profiling shows collection-bound training — scale
  decisions driven by measurements.

## Execution log (as built)

Design records live in `docs/design/`; recorded validations in
`benchmarks/results/`. Where an executed milestone deviated from a note
above (e.g. the MuJoCo/Atari reproduction *proposals*), the design record
states what was actually done and why.

| # | Status | Record | Validation |
|---|---|---|---|
| 0–2 | done | (pre-consolidation; see git history) | `m1_reinforce_cartpole.*`, `m2_ppo_cartpole.*` |
| 3 | done | `m3-consolidation-audit.md` | reproduction re-confirmed in-milestone |
| 4A–4C | done | `m4a-*.md`, `m4b-*.md`, `m4c-*.md` | gate tests in `tests/unit/` |
| 5 | done | `m5-dqn.md` | `m5_dqn.*` (CartPole 4/5 seeds, Acrobot 5/5; failures reported) |
| 6 | done | `m6-sac.md` | `m6_sac.*` (Pendulum 4/5 auto-α seeds; miss reported) |
| 7 | done | `m7-benchmarks.md` | `m7_benchmark.*` (12/12 cells ok, clean commit) |
| 8 | done | `m8-scale-out.md` | `m8_throughput.*` (measured; vectorization honestly not a speedup at this scale), `m8_docker.md` (container build blocked by env network policy — recorded) |
| 9 | done | `m9-research-layer.md` | `m9_research_demo.*` (real 2x2 ablation, gates demonstrably enforced, MockProvider only) |

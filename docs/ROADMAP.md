# Roadmap

**Status:** Proposal — awaiting review
**Companion:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

Milestones are gated by acceptance criteria, not calendar dates. Each phase
ends with a review; the next phase does not start until the review passes.
Ordering rationale: **infrastructure before algorithms, data path before
losses** — because most RL bugs live in data handling, and because every
algorithm milestone then inherits tested plumbing instead of re-litigating it.

## Milestone map

| # | Milestone | Delivers | Depends on |
|---|---|---|---|
| 0 | **Scaffolding & rails** | Repo skeleton, tooling, CI, core types, seeding | — |
| 1 | **Environment layer** | Factory, wrappers, vectorization seam, normalization | 0 |
| 2 | **Data path** | `Batch` finalized, rollout collector, replay buffer, transforms (GAE, n-step) | 1 |
| 3 | **Network library** | Torsos, heads, distributions, init schemes | 0 |
| 4 | **On-policy track: REINFORCE → PPO** | First algorithms end-to-end, SB3 parity tests, CartPole/classic-control verification | 2, 3 |
| 5 | **Experiment infrastructure** | Full Hydra tree, tracker adapters (W&B/MLflow/offline), checkpoint manager + resume, eval engine with rliable metrics | 4 |
| 6 | **Off-policy track: DQN → SAC** | Replay-buffer maturity (n-step, then prioritized), target networks, continuous control | 5 |
| 7 | **Benchmark & reproduction harness** | Standard eval suites, multi-seed benchmark runs vs. published results, committed results tables, Optuna sweeps | 6 |
| 8 | **Scale & research module** | CUDA image, multi-GPU/distributed collection seams exercised, ablation harness, model registry maturity | 7 |

Notes on ordering:

- **Milestone 3 can proceed in parallel with 1–2** (depends only on 0); it is
  sequenced here for review simplicity, but if we want throughput it is the
  natural concurrent track.
- **Minimal config/logging arrives in Milestone 0** (Hydra entry point,
  console + JSONL logging) so Milestone 4 can train real runs; Milestone 5 is
  where experiment infrastructure becomes *complete* (trackers, resume,
  rliable eval), not where it begins.
- **Why REINFORCE → PPO before DQN (Milestone 4):** REINFORCE is ~100 lines
  on top of the finished data path and validates the entire on-policy
  pipeline (collector → returns → policy gradient → update) with the fewest
  moving parts — it is the cheapest possible end-to-end integration test that
  also has pedagogical value. PPO then adds GAE, clipping, and minibatch
  epochs on proven plumbing, and PPO is the field's workhorse baseline, which
  makes it the highest-value first "real" algorithm. DQN first would force
  building replay maturity and target-network machinery before any end-to-end
  signal exists.
- **Research mode applies from Milestone 4 onward:** every algorithm ships
  with its mathematical derivation, known weaknesses, comparison to
  alternatives, and the implementation-detail checklist from the literature
  (e.g. the 37 PPO details catalogued by Huang et al., ICLR Blog Track 2022).

---

## Phase 0 (Milestone 0): Scaffolding & rails

### 1. Objective

Stand up the repository skeleton, engineering toolchain, CI, and the small
set of core primitives everything else imports: the `Batch` type (initial
version), central seeding, device utilities, and a minimal Hydra entry point
with console/JSONL logging.

### 2. Why it matters

- Tooling is **cheap on commit one and expensive to retrofit**. Ruff, mypy,
  pytest markers, and CI set the quality bar structurally — "every file
  tested and type-hinted" becomes something the repo *enforces* rather than
  something we intend.
- `Batch` and seeding are imported by every later module; getting their
  contracts right now prevents platform-wide churn later.
- A working `uv sync && ruff check && mypy && pytest` loop from day one means
  every subsequent phase review is mechanical: green CI + code review, no
  environment archaeology.

### 3. Architecture (what exactly gets built)

| Deliverable | Content |
|---|---|
| `pyproject.toml` | `uv`-managed; hatchling build; deps: torch, gymnasium, hydra-core, numpy; extras: `test` (pytest, SB3 for parity), `track` (wandb, mlflow), `dev` (ruff, mypy, pre-commit) |
| `src/rlab/` package | `types.py` (`Batch` v1, `PolicyOutput`, key constants), `utils/seeding.py` (single entry point fanning out to torch/numpy/random/env seeds), `utils/device.py`, `utils/log.py` (console + JSONL) |
| `configs/` | Root `config.yaml` + skeleton groups (`algo/`, `env/`, `train/`, `track/`) with a no-op smoke config |
| `tests/` | Unit tests for `Batch` (construction, `.to()`, slicing, minibatch iteration, error cases); determinism test for seeding (two seeded contexts produce identical torch/numpy draws); layout for `invariants/`, `parity/`, `smoke/` with markers registered |
| `.github/workflows/ci.yml` | lint (ruff check + format check) → type-check (mypy) → tests (CPU); Python 3.11 and 3.12 matrix |
| `.pre-commit-config.yaml` | ruff, ruff-format, mypy, standard hygiene hooks |
| `docker/Dockerfile` | CPU dev image reproducing CI exactly |
| `docs/design/` | ADR-0001 recording the decisions ratified in this review (package name, license, tracker, Batch strategy) |

Explicitly **out of scope** for Phase 0: environments, networks, collectors,
buffers, any algorithm. (Rule of two: `Batch` is built now only because
Milestones 2–4 are concrete scheduled consumers.)

### 4. Risks specific to this phase

- **Tooling bikeshedding** — mitigations: defaults are proposed in
  ARCHITECTURE §10 and ratified once in the review; changes afterward
  require an ADR.
- **Over-building `Batch` before real usage** — mitigation: v1 implements
  only what Phase 0 tests need plus the documented canonical keys; the
  Milestone 2 review is the scheduled point to extend it against real
  collector/buffer usage.
- **CI flakiness from torch install times** — mitigation: uv with lockfile +
  CI cache; CPU-only torch wheel in CI.

### 5. Acceptance criteria (review gate)

1. Fresh clone → `uv sync --all-extras && ruff check . && mypy && pytest`
   passes locally and in CI.
2. `import rlab` works from the installed package; `python -m rlab` (or the
   Hydra entry point) runs the no-op smoke config and writes a run directory
   with resolved config + JSONL log.
3. Seeding determinism test passes: identical seeds → identical draws and
   identical `Batch` contents across two process-local contexts.
4. Docker image builds and runs the same check suite green.
5. ADR-0001 committed.

---

## Later-phase sketches (bind at their own reviews)

- **Milestone 4 verification protocol:** loss-level parity with SB3 PPO on
  synthetic batches (exact), then CartPole-v1 / Acrobot-v1 / LunarLander-v2
  learning runs across ≥5 seeds compared against SB3 under identical eval
  protocols (statistical, rliable CIs).
- **Milestone 7 reproduction targets (proposal):** PPO on MuJoCo locomotion
  vs. Huang et al. reference results; DQN on a 3–5 game Atari subset vs.
  published scores. Full-suite Atari is out of scope until compute is
  budgeted.
- **Milestone 8 scale path:** vectorized-env throughput benchmarks first;
  distributed collection only if profiling shows collection-bound training —
  scale decisions driven by measurements, not architecture astronautics.

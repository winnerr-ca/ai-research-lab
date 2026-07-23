# M7: Benchmark system, statistics, reproduction packages

**Status:** Implemented

## Objective, and why

Fair multi-algorithm comparison as *infrastructure* rather than ad-hoc
scripts: manifests declare (algorithm, environment, seed) cells with
fixed budgets; execution is isolated, timeout-bounded, and resumable;
failures are first-class results; aggregation uses robust statistics
with stated uncertainty. Plus reproduction packages that say exactly
what they are — and are not.

## Deliverables / non-goals / acceptance

Delivered: `rlcore.stats` (IQM, percentile-bootstrap CI, probability of
improvement, performance profiles — native NumPy against hand-computed
anchors; the tested contract a future backend swap must satisfy, chosen
over depending on the archived `rliable`), `rlcore.benchmark`
(manifest schema that *rejects* `stop_return` overrides — fixed budgets
are enforced, not encouraged; one subprocess per run with a real
wall-clock timeout; `status.json` per cell making interrupted
benchmarks resumable; ok/failed/timeout/cached census),
`rlcore.benchreport` (aggregation + Markdown report + curves/profiles),
the `rlcore-benchmark` CLI, `benchmarks/manifests/m7-manifest.json`,
and `docs/reproductions/repro-{ppo,dqn,sac}.md`.

Non-goals held: no cross-task normalized scores (single-task suites so
far), no hyperparameter search, no significance-test veneer at n=3.

Acceptance: 20 unit tests (statistics anchors, manifest validation,
failure/timeout/resume behavior on scripted subprocesses), canonical
gate green, and the manifest executed with results recorded.

## Honest labels for "reproduction"

Each reproduction package states its level explicitly:

- **Implementation validation** (what all three currently are): the
  algorithm implementation reaches expected behavior on accessible
  environments under this repo's own protocol.
- *Partial / approximate / full reproduction* labels are defined and
  reserved for when a published paper's environment suite, budget, and
  evaluation protocol are actually matched. No package claims these
  today, because none matches a published protocol.

## Metadata-integrity incident (recorded on purpose)

The first execution of the M7 manifest was launched *before* the M7
code commit existed, so every run stamped `git_dirty=True` — the exact
failure mode the platform's clean-SHA discipline exists to catch. The
pass was discarded and re-executed from the clean commit; the recorded
results (`benchmarks/results/m7_benchmark.md`) come entirely from the
clean pass. Cost: ~35 minutes of compute; benefit: every recorded run
is attributable to an exact commit.

## Validation results (`benchmarks/results/m7_benchmark.{md,json}`)

See the report for the full census and per-cell statistics. Summary:
- **Census: 12/12 cells ok** (no failures, no timeouts) from clean commit
  `708de42`.
- CartPole-v1 final deterministic evals: PPO 500.0 on all 3 seeds
  (15 updates = 30,720 steps); REINFORCE 500.0 on all 3 seeds (50
  updates); DQN 126.4-177.1 at its 30,000-step budget — consistent with
  M5's observation that DQN needs roughly the full 60k default budget on
  CartPole. Budgets are per-algorithm as declared in the manifest, so
  cross-algorithm rows are budget-context engineering numbers, not
  superiority claims.
- Pendulum-v1 SAC: -155.0 to -189.4 across 3 seeds
  (IQM -170.4 [-189.4, -155.0]).
- Resumability is covered by unit tests on scripted subprocess runs; the
  real benchmark completed in one pass, so cache-skip behavior was not
  additionally exercised at full scale.

## Self-review

- Statistics on n=3 seeds are honest indicators only, and the report
  template says so in its scope block.
- Discrete-action algorithms ran CartPole while SAC ran Pendulum;
  cross-environment rows are explicitly marked non-comparable.
- The subprocess-per-run design costs ~2s of Python startup per cell —
  negligible against run durations here, and it is what makes timeouts
  and isolation real.

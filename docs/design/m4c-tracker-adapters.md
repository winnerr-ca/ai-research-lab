# M4C: Optional tracker adapters

**Status:** Implemented

## Objectives

W&B and MLflow adapters behind the small `Tracker` protocol
(`start` / `log_metrics` / `log_summary` / `finish`), added only after
local tracking (M4A) was complete. The local run directory remains the
source of record; trackers are strictly additive mirrors driven by the
trainers, so the same `rlcore-train` command works identically with
`track=none` (default), `track=wandb`, or `track=mlflow`.

## Credential policy (verified by tests and end-to-end runs)

- `track=none` performs **no optional imports at all** (tested via
  `sys.modules`).
- `track=wandb` defaults to **offline mode** (respecting a user-set
  `WANDB_MODE`); no account or login is ever required — users who later
  want the hosted dashboard run `wandb sync` themselves.
- `track=mlflow` defaults to a **local SQLite store**
  (`sqlite:///mlflow.db`, MLflow's recommended local backend now that its
  legacy `./mlruns` file store is in maintenance mode); no server or
  credential involved.
- Both packages live in the optional `track` extra; missing packages fail
  at tracker construction — before any training time is spent — with an
  actionable install message.

## Non-goals

Dashboards, sweep orchestration, artifact upload (models/checkpoints to
tracker backends), tracker-side run comparison — all later-milestone
material if a real need emerges.

## Risks and mitigations

- **Backend API drift** (both libraries move fast) — the adapters touch a
  deliberately tiny API surface (init/log/summary/finish) and are covered
  by real-backend tests that will catch breakage on dependency bumps; the
  MLflow file-store deprecation was caught exactly this way during
  implementation and resolved by moving to the recommended SQLite default.
- **Tracker faults corrupting runs** — trackers never sit between the
  trainer and the local run directory; local artifacts are written
  regardless.

## Tests and validation

- Protocol wiring via a recording test double: trainers call `start` once
  with the platform run id, `log_metrics` once per iteration with the
  exact history entries, `log_summary` and `finish` once.
- `make_tracker("none")` returns `None` without importing wandb/mlflow;
  unknown kinds rejected.
- Real-backend tests (skipped if the extra isn't installed): W&B offline
  run directory created and MLflow SQLite store written, both without
  credentials; NaN metrics filtered for MLflow.
- End-to-end: the identical training command executed with all three
  `track` values, producing identical training results (same seed, same
  final evaluation) — trackers observed, never perturbed.

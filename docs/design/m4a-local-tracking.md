# M4A: Experiment configuration and local tracking

**Status:** Implemented

## Objectives

- One user-facing training command — `rlcore-train algo=<name>
  [algo.key=value ...]` — with the algorithm as a Hydra config group, and a
  matching `rlcore-evaluate <run_dir>` that re-evaluates any finished run.
- Every run gets a unique identity and a complete, local, self-describing
  run directory: no external service, account, or credential is involved at
  any point.
- Provenance is recorded automatically: resolved config, git commit and
  dirty-tree state, Python/PyTorch/Gymnasium/NumPy/Hydra versions, device
  and OS/machine metadata.

## Run-directory layout (v1)

```
run_dir/
├── run.json          # run_id, created_at, algo/env/seed, full metadata
├── config.yaml       # resolved algorithm config
├── metrics.jsonl     # one JSON object per training iteration
├── result.json       # final summary payload (algorithm-specific fields)
├── final_model.pt    # final weights: versioned header + state_dict
├── summary.md        # human-readable summary
├── evaluations.jsonl # appended by rlcore-evaluate (post-hoc evaluations)
└── plots/learning_curve.png
```

`final_model.pt` is a **final artifact** (for evaluation and sharing), not
a resumable training checkpoint — resumable checkpoints are M4B's separate,
richer contract. The header (`format`/`version`/`algo`) makes loaders fail
loudly on wrong file kinds or newer versions instead of part-loading.

Run identity: `{UTC timestamp}-{algo}-{env}-s{seed}-{6 hex}` — sortable,
greppable, unique.

## Non-goals

- External trackers (M4C), resumable checkpoints (M4B), rliable aggregate
  statistics (M6), sweep orchestration, dashboards, a run registry/index.
- No `configs/` YAML tree: with two algorithms, structured configs
  registered in code (`ConfigStore`) are the compact form; YAML files
  appear when composition across groups (env suites, sweep presets) has a
  real consumer.

## Risks and mitigations

- **Run-dir schema churn** — every file that needs parsing carries either a
  versioned header (`final_model.pt`) or is line-delimited JSON that
  tolerates added fields; the layout is documented here as v1.
- **Matplotlib dependency weight** — accepted (the stack always planned
  local visualization); imported lazily so `import rlcore` stays light and
  headless machines work (Agg backend forced).
- **Hydra group dispatch complexity** — kept to one dataclass +
  `isinstance` dispatch; verified by compose-API tests.

## Tests and validation

- Unit: run-id format/uniqueness; metadata keys; the full run-directory
  layout (identity round-trip, final-model header validation and
  algo-mismatch rejection, summary content, PNG magic bytes); Hydra
  compose dispatch for both algorithms with overrides; `evaluate_run`
  determinism given an eval seed, stochastic mode, and its
  `evaluations.jsonl` record.
- End-to-end: `rlcore-train algo=ppo ...` (tiny run) produced the complete
  run directory and `rlcore-evaluate` re-evaluated it from disk.
- Training behavior is untouched: artifact writing happens after training
  and consumes no RNG (`new_run_id` uses `uuid4`/`os.urandom`, not seeded
  streams), and the M1/M2 validation scripts call `train()` without
  `out_dir`, so committed validation numbers are unaffected.

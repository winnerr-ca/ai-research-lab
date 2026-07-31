# The lab dashboard (`rlcore-lab`)

A localhost web interface for testing your research on the platform:
browse and compare recorded runs, launch new training runs (including
your own algorithm packages), watch launched jobs, and list
research-store entities.

```sh
uv run rlcore-lab                 # http://127.0.0.1:8765
uv run rlcore-lab --port 9000 --runs-root outputs --store research-store
```

Everything is local. There is no account, no credential, and no
external service; the server binds to loopback by default because the
launch endpoint starts training subprocesses.

## What it shows, and where the numbers come from

The dashboard adds no numbers of its own:

- **Runs** are any directories holding a `run.json` under `--runs-root`
  (default `outputs/`) — the standard run directories written by
  `rlcore-train`, the benchmark runner, or dashboard launches.
- **Learning curves** are drawn straight from each run's
  `metrics.jsonl`; the metric key is selectable from whatever the run
  recorded.
- **Comparison** groups selected runs by (algorithm, environment),
  scores each run by its recorded `final_eval_return_mean`, and
  aggregates with `rlcore.stats.summarize` (IQM, bootstrap CI) under a
  fixed bootstrap seed, so the same selection always reports the same
  interval. Caveats are stated, not hidden: small samples (n < 5),
  unequal budgets within a group, and repeated seeds are each flagged,
  and cross-group numbers are shown side by side without ranking. Runs
  that cannot contribute a score are listed as *excluded* with a
  reason — never silently dropped.

## Launching runs (yours included)

The launch form lists every algorithm package discovered under
`rlcore.agents` — discovery, not a hard-coded list. A package qualifies
when its `train` module holds exactly one `*Config` dataclass and a
`train` callable, which the four built-ins satisfy; add your own
variant package with the same shape and it appears in the dashboard
without touching lab code.

Launches follow the benchmark runner's isolation pattern: one
subprocess per run, stderr captured into the run directory, failures
recorded (with the stderr tail) rather than swallowed. Every launched
run writes a standard run directory plus `lab-launch.json` recording
the exact request, so a dashboard run is as attributable as a CLI run.

Requests are validated before anything executes: unknown algorithm
names and unknown config-field override keys are rejected with HTTP
400, and run-directory paths from the client are confined to the runs
root.

## What it deliberately cannot do

Consistent with the research layer's approval gates (`docs/research.md`)
and the read-mostly `rlcore-research` CLI: no HTTP endpoint can
publish, delete, or conclude research entities, or enable a paid LLM
provider. The research panel is a listing only. Gated actions require
constructing an explicit `Approval` in code, exactly as before.

## API summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | the dashboard page |
| `/api/runs` | GET | run summaries under the runs root |
| `/api/metrics?dir=<rel>` | GET | one run's `metrics.jsonl` |
| `/api/compare?dirs=a|b|c` | GET | grouped statistics + caveats |
| `/api/algos` | GET | discovered algorithms and config fields |
| `/api/launch` | POST | `{"algo", "overrides"}` → job snapshot |
| `/api/jobs` | GET | launched-job statuses (this server session) |
| `/api/research` | GET | research-store entity ids by kind |

## Limitations

- Job state lives in the server process; after a restart, finished runs
  remain visible as run directories but the jobs table starts empty.
- The comparison uses final evaluation returns only; for full
  fixed-budget protocol comparisons (subprocess timeouts, failure
  cells, reports), use `rlcore-benchmark` with a manifest.
- Single user, single machine by design — like the research store, no
  locking or multi-user support.

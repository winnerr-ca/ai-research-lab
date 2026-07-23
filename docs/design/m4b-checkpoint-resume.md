# M4B: Checkpoint and resume

**Status:** Implemented

## Objectives

Resumable training checkpoints scoped to the state REINFORCE and PPO
*actually have*, with the gate test as the definition of correctness:
**continuous 2N training ≡ N + save + fresh-process restore + N**, verified
bit-for-bit (identical final weights *and* identical post-resume metric
entries) for both algorithms in the pinned CPU environment.

## Checkpoint contents (schema v1)

Envelope: `format` / `version` / `algo` / `payload`, validated on load with
distinct errors (missing file, unreadable/corrupted, wrong format, newer
schema version, wrong algorithm) — never a silent partial load. Payload:

| Item | REINFORCE | PPO |
|---|---|---|
| Model / optimizer state dicts | ✓ | ✓ |
| Progress counters (update, env steps, episodes) | ✓ | ✓ (no episode counter) |
| Metric history (for continued `metrics.jsonl`) | ✓ | ✓ |
| Full RNG state: bundle streams + torch/numpy/python globals | ✓ | ✓ |
| Resolved config + run identity (`run_id` survives resume) | ✓ | ✓ |
| Collector state (current obs, running episode return) | n/a | ✓ |
| Environments (train + eval), pickled | ✓ | ✓ |

Writes are atomic (temp file → fsync → rename); a crash mid-write leaves
the previous checkpoint intact.

Resume validation: the requested config must equal the checkpoint's in
every field except the *operational* ones — the budget
(`updates`/`total_updates`; extending it is the core resume use case) and
`checkpoint_every` (scheduling only). Any other difference is refused with
the differing fields listed.

## The environment-state boundary (documented)

Exact mid-episode continuation requires the simulator's internal state,
which Gymnasium exposes no generic API for. v1 resolves this by
**pickling the environments** into the checkpoint: classic-control envs
(CartPole included) pickle cleanly, carrying both their RNG streams and
their physical state — which is why even PPO's fixed-horizon resume, where
the env is usually mid-episode at a checkpoint, is exact. Environments
that cannot be pickled fail **at save time** with a clear error stating
the boundary; for them, exact resume is out of scope in v1. This is a
documented limitation, not a silent degradation.

**Trust boundary:** checkpoints contain pickled objects, so loading uses
full unpickling (`weights_only=False`). Load only checkpoints produced by
you or your own run infrastructure — the standard rule for pickle files.

## Non-goals (explicitly excluded)

Replay-buffer state, target networks, SAC temperature, or any other
future-algorithm state (added when those algorithms exist, with their own
schema bump); retention policies and best-checkpoint registries (benchmark
milestone); checkpoint upload/artifact tracking (M4C+).

## Risks and mitigations

- **Pickled-env fragility across versions** — the checkpoint records the
  resolved config and package versions travel in the run record; loading is
  documented as same-pinned-environment, matching the platform's stated
  reproducibility scope.
- **Config drift on resume** — refused loudly by `check_resume_config`.
- **Schema evolution** — versioned envelope; newer-version files are
  rejected with an upgrade message rather than part-loaded.

## Tests

Envelope roundtrip and atomicity (no temp residue); each validation error
distinctly; env pickle roundtrip and the unpicklable-env save-time error;
full RNG-state roundtrip across all six streams; operational-vs-fixed
config fields on resume; `checkpoint_every` requiring `out_dir`;
in-process and **fresh-subprocess** resume-equals-continuous for both
algorithms; `run_id` preservation across resume. 203 fast tests green.

## Usage

```sh
rlcore-train algo=ppo algo.checkpoint_every=10
rlcore-train algo=ppo algo.total_updates=150 resume_from=<run_dir>/checkpoint.pt
```

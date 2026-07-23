## What this changes

<!-- One concern per PR. Link the issue if there is one. -->

## Why

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` passes locally
- [ ] New/changed behavior is covered by tests (mathematical anchors for algorithm math; boundary cases for terminated/truncated handling)
- [ ] Config fields, CLI flags, and user-visible behavior are documented under `docs/`
- [ ] Any reported numbers come from recorded runs (config + seed + versions + commit), with failed seeds included
- [ ] No secrets, keys, or credentials anywhere in the diff

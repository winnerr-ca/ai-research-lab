# Contributing to rlcore

Thank you for considering a contribution. This project holds a small
number of rules tightly; everything else is ordinary open-source
practice.

## Ground rules (the non-negotiables)

1. **No fabricated results.** Every number in a report, docstring, or PR
   description must come from a recorded run (config, seed, versions,
   commit, device). Failed seeds are reported, not dropped.
2. **Termination vs truncation is sacred.** No bootstrap at true
   termination; bootstrap from the final observation at truncation; no
   temporal chain crosses either boundary. Changes touching this need
   hand-computed test cases.
3. **Claims match evidence.** Multi-seed statistics on equal budgets or
   it's an engineering observation, not a performance claim.
4. **Algorithm math stays visible.** Losses and estimators live in
   pure, unit-tested functions in the algorithm's own directory. Shared
   components are extracted only when multiple algorithms already need
   them (rule of two), via a consolidation audit.
5. **No secrets in the repository.** API keys come from environment
   variables, always optional, never logged or committed.

## Development setup

```sh
uv sync --all-extras
uv run pytest            # fast, deterministic suite
```

The full gate that CI (and every PR) must pass:

```sh
uv run ruff check . && uv run ruff format --check . \
  && uv run mypy && uv run pytest
```

`mypy` runs strict on `src`, `tests`, and `benchmarks`. Slow learning
canaries (`pytest -m slow`) are excluded from the default run and
exercised on demand.

## What a good PR looks like

- One concern per PR; algorithm changes come with their required test
  list (mathematical anchors, boundary handling, determinism, resume).
- New config fields are documented in the config dataclass docstring
  and covered by a test.
- Anything user-visible updates the relevant page under `docs/`.
- Validation results (if any) go in `benchmarks/results/` with raw JSON
  plus a Markdown summary stating the protocol, and are produced from a
  clean commit.

## Reporting bugs / proposing features

Use the issue templates. For anything touching the scientific-correctness
rules above, include a minimal reproduction (seed + config + versions).

## Code of conduct

All project spaces are covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the project's license.

# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no |

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
["Report a vulnerability"](../../security/advisories/new) form for this
repository rather than opening a public issue. You can expect an
acknowledgement within a week. Please include a minimal reproduction
where possible.

## Scope notes specific to this project

- **Checkpoint loading is a trust boundary.** `rlcore` checkpoints are
  `torch.save` archives loaded with `weights_only=False`, which executes
  pickle deserialization. Load checkpoints only from sources you trust —
  the same rule as for any pickle file. This is documented behavior, not
  a vulnerability, but reports of ways the platform could make it safer
  without breaking exact resume are welcome.
- **No network services.** The platform runs local processes and writes
  local files; optional tracker/LLM integrations (`wandb`, `mlflow`,
  `anthropic`) only activate when explicitly configured, and API keys are
  read from environment variables and never persisted by rlcore.

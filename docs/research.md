# The research layer

`rlcore.research` is an auditable research notebook built on the
execution layer's ground truth (run directories). It stores typed,
versioned entities; every statement in it is grounded, and five actions
always require an explicit human approval.

## Quick start (offline, free)

```python
from pathlib import Path
from rlcore.research import (
    Approval, MockProvider, ResearchStore, ResearchWorkflow,
    ingest_path, load_documents, retrieve,
)

store = ResearchStore(Path("research-store"))
workflow = ResearchWorkflow(store, MockProvider())

# Ingest a source and make a claim that cites it (section-level ref).
ingest_path(Path("docs/algorithms/ppo.md"), store)
hits = retrieve("clipped surrogate objective", load_documents(store))
claim = workflow.claim_from_sections("PPO clips the ratio.", hits)

# Ground an observation in a real run directory.
ref = workflow.register_run(Path("outputs/my-run"))
obs = workflow.observe("Solved in 20k steps.", (ref,))
```

Or from the shell: `rlcore-research ingest docs/algorithms/ppo.md`,
`rlcore-research search "advantage estimation"`, `rlcore-research list`.

## Entities

Paper, Citation, Claim, ResearchQuestion, Hypothesis, ExperimentPlan,
Baseline, Ablation, RunReference, Observation, Analysis, Conclusion,
Limitation, ResearchReport — all share: stable ID, UTC timestamps,
integer version (full history kept by the store), provenance
(`human` / `llm:<provider>` / `ingest:<path>`), typed links, and
`review_state` (`draft`/`approved`/`rejected`).

Grounding is enforced at construction: a `Claim` without a source
section or run ID, an `Observation` without runs, or a `Conclusion`
without analyses is unrepresentable.

## The five human gates

1. **Expensive experiments** — executing a plan with
   `cost_class != "small"` needs an `Approval("run-experiment", ...)`.
2. **Paid services** — any call through a provider with
   `is_paid=True` needs `Approval("paid-llm", ...)`. The bundled
   `MockProvider` is free and deterministic; tests and demos use it
   exclusively.
3. **Publication** — `publish_report` needs `Approval("publish", ...)`;
   the store independently refuses a published-but-unapproved report.
4. **Destructive deletion** — `store.delete` needs an
   `Approval("delete", ...)` naming the entity.
5. **Hypotheses are not conclusions** — `conclude()` requires at least
   one Analysis that references real runs, plus
   `Approval("conclude", ...)`. LLM text can never be promoted to a
   conclusion on its own.

## Using Claude for drafting (optional, paid)

```python
from rlcore.research import AnthropicProvider, ResearchWorkflow, Approval

provider = AnthropicProvider()  # reads ANTHROPIC_API_KEY from the env
workflow = ResearchWorkflow(store, provider)
analysis = workflow.summarize_evidence(
    "What does the doc say about advantage estimation?",
    hits,
    approval=Approval(action="paid-llm", subject=provider.name,
                      granted_by="you@example.org"),
)
```

Requires `uv sync --extra research` (installs `anthropic` and `pypdf`).
Keys are read from the environment at construction, held in memory, and
never written to the store, logs, or reports. **Never commit keys.**
LLM-drafted entities carry `llm:*` provenance and stay `draft` until a
human reviews them.

## Retrieval scope, honestly

Retrieval is deterministic lexical term-overlap with stable
tie-breaking, returning section/page references for citation. It is
not semantic search and is labeled accordingly; scanned PDFs without a
text layer yield no sections.

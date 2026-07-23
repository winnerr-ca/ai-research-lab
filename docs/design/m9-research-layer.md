# M9: Research-intelligence layer

**Status:** Implemented

## Objective, and why

The layer the platform was pointed at from Phase 0: turn training runs
and source documents into an auditable research record. The execution
layer produces ground truth (run directories); M9 adds the typed memory
on top — papers, claims, hypotheses, plans, observations, analyses,
conclusions, reports — with two invariants that hold everywhere:
**grounding** (nothing citable exists without a source section or a run
ID behind it) and **human gates** (nothing expensive, paid, published,
destructive, or epistemically escalating happens on an LLM's say-so).

## Deliverables / non-goals / acceptance

Delivered: `rlcore.research` with 14 entity kinds (stable IDs, UTC
timestamps, integer versions, provenance strings, typed links,
review states; JSON round-trip tested for all 14); a filesystem store
with full version history and atomic writes; Markdown/text/PDF
ingestion that preserves raw sources with SHA-256 and yields
section/page-referenced units; deterministic lexical retrieval (labeled
as such — it grounds citations, it does not rank by meaning);
a provider-neutral `CompletionProvider` protocol with a deterministic
`MockProvider` and an `AnthropicProvider` (official SDK, lazy import,
key strictly from `ANTHROPIC_API_KEY`, never persisted); workflow
operations enforcing grounding; the five approval gates; a read-mostly
`rlcore-research` CLI. Non-goals held: no embeddings/vector DB, no
autonomous agent loop, no background LLM calls, no paid call anywhere
in tests or demos.

Acceptance: the gate/grounding test list green, canonical gate green,
and a recorded end-to-end demonstration on real artifacts
(`benchmarks/results/m9_research_demo.{md,json}`).

## The five gates, and where each is enforced

| Gate | Enforced in |
|---|---|
| Expensive experiments | `ResearchWorkflow.check_plan_executable` (cost_class ≠ small) |
| Paid services | `ResearchWorkflow._complete` (provider.is_paid) |
| Publication | `publish_report` **and** independently in `ResearchStore.save` (a published-but-unapproved report is unpersistable) |
| Destructive deletion | `ResearchStore.delete` (approval must name the entity) |
| Hypothesis → conclusion | `conclude` (requires run-grounded Analysis + approval) |

An `Approval` is data recorded with grantor/note/timestamp — a human
act. LLM output cannot mint one, and LLM-produced entities are always
born `review_state="draft"`.

## Design choices

- **Entities as frozen dataclasses** with a hand-rolled, bounded JSON
  decoder (the field-type vocabulary is small and closed) rather than a
  serialization framework — smallest correct thing, fully typed.
- **Version history in the store, immutability in the objects**:
  `revise()` is the only way to produce version *n+1*; the store rejects
  version skips and keeps every prior version.
- **Retrieval is lexical on purpose.** An embedding index would add a
  model dependency and non-determinism to the *grounding* path; term
  overlap with stable tie-breaks is auditable and sufficient for
  section-level citation. Semantic search can layer on later without
  touching the citation contract.
- **The demo ablation is real but tiny**, and its recorded conclusion is
  strictly about the pipeline mechanism — the record itself demonstrates
  the no-overclaiming rule (n=2 seeds, 3 updates: no performance claim).

## Self-review

- PDF extraction quality depends on `pypdf`'s text layer; scanned PDFs
  yield empty sections. Recorded as a Limitation-shaped caveat in the
  docs rather than papered over.
- The store is single-process (no locking); concurrent writers could
  interleave version checks. Acceptable for a local research notebook,
  listed in deferred work.
- `AnthropicProvider` is intentionally untested against the live API in
  CI (that would be a paid call — gate 2 applies to the project itself);
  its key handling and shape are unit-tested, and the integration seam
  is one function.

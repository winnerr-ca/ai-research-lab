"""Research-intelligence layer: typed entities, grounded retrieval, gates.

Everything in this package obeys two invariants (ARCHITECTURE §8):

1. **Grounding** — claims cite sources or runs; observations cite runs;
   conclusions cite analyses; LLM output is always born ``draft``.
2. **Human gates** — expensive experiments, paid LLM calls, publication,
   destructive deletion, and hypothesis→conclusion promotion each require
   an explicit recorded human :class:`~rlcore.research.workflows.Approval`.
"""

from rlcore.research.entities import (
    Analysis,
    Baseline,
    Citation,
    Claim,
    Conclusion,
    Entity,
    ExperimentPlan,
    Hypothesis,
    Limitation,
    Observation,
    Paper,
    ResearchQuestion,
    ResearchReport,
    RunReference,
    SourceRef,
)
from rlcore.research.ingest import ingest_path, load_documents, retrieve
from rlcore.research.llm import AnthropicProvider, CompletionProvider, MockProvider
from rlcore.research.store import ResearchStore
from rlcore.research.workflows import Approval, ApprovalRequiredError, ResearchWorkflow

__all__ = [
    "Analysis",
    "AnthropicProvider",
    "Approval",
    "ApprovalRequiredError",
    "Baseline",
    "Citation",
    "Claim",
    "CompletionProvider",
    "Conclusion",
    "Entity",
    "ExperimentPlan",
    "Hypothesis",
    "Limitation",
    "MockProvider",
    "Observation",
    "Paper",
    "ResearchQuestion",
    "ResearchReport",
    "ResearchStore",
    "ResearchWorkflow",
    "RunReference",
    "SourceRef",
    "ingest_path",
    "load_documents",
    "retrieve",
]

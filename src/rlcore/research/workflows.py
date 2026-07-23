"""Research workflows: grounding enforcement and human approval gates.

The five mandated gates (ARCHITECTURE §8), each enforced here or in the
store, never bypassable by an LLM output:

1. **Expensive experiments** — executing an :class:`ExperimentPlan` whose
   ``cost_class`` is not ``small`` requires an :class:`Approval`.
2. **Paid services** — calling a provider with ``is_paid=True`` requires
   an :class:`Approval` for the ``paid-llm`` action.
3. **Publication** — a :class:`ResearchReport` becomes ``published`` only
   through :meth:`ResearchWorkflow.publish_report` with an approval; the
   store independently refuses unapproved published reports.
4. **Destructive deletion** — :meth:`ResearchStore.delete` requires an
   approval naming the entity.
5. **Hypotheses are not conclusions** — :meth:`ResearchWorkflow.conclude`
   refuses to produce a Conclusion unless it is grounded in an Analysis
   over real runs *and* a human approves; the result is still born
   ``draft`` if approval covers execution but review is pending.

An :class:`Approval` is a *human* act recorded as data. Nothing in this
package fabricates one; CLI and tests construct them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlcore.research.entities import (
    Analysis,
    Claim,
    Conclusion,
    ExperimentPlan,
    Observation,
    ResearchReport,
    RunReference,
    create,
    revise,
    utc_now,
)

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.research.ingest import RetrievedSection
    from rlcore.research.llm import CompletionProvider
    from rlcore.research.store import ResearchStore


class ApprovalRequiredError(PermissionError):
    """Raised when a gated action is attempted without human approval."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Approval:
    """A recorded human decision authorizing one gated action.

    Attributes:
        action: One of ``run-experiment``, ``paid-llm``, ``publish``,
            ``delete``, ``conclude``.
        subject: The entity ID (or provider name) the approval covers.
        granted_by: Human identifier (name/email); never an LLM.
        note: Free-text rationale.
        granted_at: UTC timestamp.
    """

    action: str
    subject: str
    granted_by: str
    note: str = ""
    granted_at: str = ""

    def __post_init__(self) -> None:
        """Stamp the grant time if absent."""
        if not self.granted_at:
            object.__setattr__(self, "granted_at", utc_now())


def _require(approval: Approval | None, action: str, subject: str) -> Approval:
    if approval is None or approval.action != action or approval.subject != subject:
        raise ApprovalRequiredError(
            f"Action {action!r} on {subject!r} requires an explicit human "
            f"Approval(action={action!r}, subject={subject!r}); got {approval!r}."
        )
    return approval


class ResearchWorkflow:
    """Grounded research operations over a store and an LLM provider."""

    def __init__(self, store: ResearchStore, provider: CompletionProvider) -> None:
        """Bind the workflow to a store and a completion provider."""
        self._store = store
        self._provider = provider

    @property
    def provider_provenance(self) -> str:
        """Provenance string for entities produced by the bound provider."""
        return f"llm:{self._provider.name}"

    def _complete(self, prompt: str, *, system: str, approval: Approval | None) -> str:
        if self._provider.is_paid:
            _require(approval, "paid-llm", self._provider.name)
        return self._provider.complete(prompt, system=system)

    def register_run(self, run_dir: Path) -> RunReference:
        """Record an existing rlcore run directory as a RunReference.

        Reads ``run.json`` and ``result.json`` — the reference points at
        real recorded results; it cannot be created from thin air.

        Raises:
            FileNotFoundError: If the run directory lacks its records.
        """
        import json

        run_info = json.loads((run_dir / "run.json").read_text())
        result = json.loads((run_dir / "result.json").read_text())
        summary = tuple(
            (key, str(value))
            for key, value in sorted(result.items())
            if isinstance(value, int | float | str | bool)
        )
        reference = create(
            RunReference,
            run_id=str(run_info["run_id"]),
            run_dir=str(run_dir),
            algo=str(run_info.get("algo", "")),
            env_id=str(run_info.get("env_id", "")),
            seed=int(run_info["seed"]) if "seed" in run_info else None,
            commit=str(run_info.get("metadata", {}).get("commit", "")),
            summary=summary,
            provenance=f"ingest:{run_dir}",
        )
        self._store.save(reference)
        return reference

    def claim_from_sections(self, text: str, sections: tuple[RetrievedSection, ...]) -> Claim:
        """Create a human claim grounded in retrieved sections.

        Raises:
            ValueError: If ``sections`` is empty (a Claim must cite).
        """
        refs = tuple(hit.section.source_ref() for hit in sections)
        claim = create(Claim, text=text, source_refs=refs, provenance="human")
        self._store.save(claim)
        return claim

    def observe(self, text: str, runs: tuple[RunReference, ...]) -> Observation:
        """Record an observation grounded in registered runs."""
        observation = create(
            Observation,
            text=text,
            run_ids=tuple(run.run_id for run in runs),
            links=tuple(run.entity_id for run in runs),
            provenance="human",
        )
        self._store.save(observation)
        return observation

    def check_plan_executable(
        self, plan: ExperimentPlan, *, approval: Approval | None = None
    ) -> None:
        """Gate 1: expensive experiment plans need human approval.

        Small plans pass silently; moderate/large raise
        :class:`ApprovalRequiredError` unless a matching approval is supplied.
        """
        if plan.cost_class != "small":
            _require(approval, "run-experiment", plan.entity_id)

    def summarize_evidence(
        self,
        question: str,
        sections: tuple[RetrievedSection, ...],
        *,
        approval: Approval | None = None,
    ) -> Analysis:
        """LLM-draft an analysis over retrieved sections, with citations.

        The prompt contains only the retrieved text (source-grounded); the
        result is an Analysis with ``llm:*`` provenance in ``draft`` state
        — human review is required before it can be approved. Paid
        providers require the paid-services approval (gate 2).
        """
        cited = "\n\n".join(
            f"[{hit.section.document_id} / {hit.section.ref}]\n{hit.section.text}"
            for hit in sections
        )
        system = (
            "You summarize ONLY from the provided sections. Cite each point "
            "as [document_id / section]. If the sections do not answer the "
            "question, say so plainly. Never invent results."
        )
        answer = self._complete(
            f"Question: {question}\n\nSections:\n{cited}", system=system, approval=approval
        )
        analysis = create(
            Analysis,
            text=answer,
            method=f"lexical retrieval + {self._provider.name} summary",
            provenance=self.provider_provenance,
        )
        self._store.save(analysis)
        return analysis

    def analyze_runs(
        self,
        text: str,
        method: str,
        observations: tuple[Observation, ...],
    ) -> Analysis:
        """Record a human analysis over run-grounded observations."""
        run_ids = tuple(dict.fromkeys(rid for obs in observations for rid in obs.run_ids))
        analysis = create(
            Analysis,
            text=text,
            method=method,
            observation_ids=tuple(obs.entity_id for obs in observations),
            run_ids=run_ids,
            provenance="human",
        )
        self._store.save(analysis)
        return analysis

    def conclude(
        self,
        text: str,
        analyses: tuple[Analysis, ...],
        *,
        approval: Approval | None = None,
    ) -> Conclusion:
        """Gate 5: a conclusion needs run-grounded analysis + human approval.

        Raises:
            ValueError: If no analysis is provided, or none of the analyses
                reference real runs (a hypothesis with no experimental
                evidence cannot become a conclusion).
            ApprovalRequiredError: Without a human ``conclude`` approval.
        """
        if not analyses:
            raise ValueError("A conclusion requires at least one Analysis.")
        if not any(analysis.run_ids for analysis in analyses):
            raise ValueError(
                "Refusing to conclude: none of the supplied analyses reference "
                "runs. Hypotheses do not become conclusions without "
                "experimental evidence (ARCHITECTURE §8)."
            )
        _require(approval, "conclude", analyses[0].entity_id)
        conclusion = create(
            Conclusion,
            text=text,
            analysis_ids=tuple(a.entity_id for a in analyses),
            links=tuple(a.entity_id for a in analyses),
            provenance="human",
        )
        self._store.save(conclusion)
        return conclusion

    def draft_report(
        self,
        title: str,
        entity_ids: tuple[str, ...],
        *,
        approval: Approval | None = None,
    ) -> ResearchReport:
        """Assemble a draft report over stored entities.

        Structured skeleton plus a provider-drafted narrative paragraph
        (gate 2 applies for paid providers). The body lists every included
        entity with its provenance and review state — the reader always
        sees what is human-approved and what is an unreviewed draft.
        """
        rows = []
        for entity_id in entity_ids:
            entity = self._store.get(entity_id)
            rows.append(
                f"- `{entity.entity_id}` ({entity.kind}, v{entity.version}, "
                f"{entity.review_state}, provenance: {entity.provenance})"
            )
        narrative = self._complete(
            "Write a short neutral narrative paragraph for a research report "
            f"titled {title!r} covering these entities:\n" + "\n".join(rows),
            system="Neutral tone. No claims beyond the listed entities.",
            approval=approval,
        )
        body = "\n".join(
            [
                f"# {title}",
                "",
                narrative,
                "",
                "## Included entities (provenance and review state)",
                "",
                *rows,
                "",
            ]
        )
        report = create(
            ResearchReport,
            title=title,
            body_markdown=body,
            entity_ids=entity_ids,
            provenance=self.provider_provenance,
        )
        self._store.save(report)
        return report

    def publish_report(
        self, report: ResearchReport, *, approval: Approval | None = None
    ) -> ResearchReport:
        """Gate 3: publication requires explicit human approval.

        The approved report is re-saved as the next version with
        ``review_state="approved"`` and ``published=True``.
        """
        checked = _require(approval, "publish", report.entity_id)
        published = revise(
            report,
            published=True,
            review_state="approved",
            provenance=f"{report.provenance}; approved by {checked.granted_by}",
        )
        self._store.save(published)
        return published

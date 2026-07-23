"""Unit tests for workflows: grounding enforcement and all five gates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from rlcore.research.entities import Analysis, ExperimentPlan, create
from rlcore.research.ingest import ingest_path, load_documents, retrieve
from rlcore.research.llm import MockProvider
from rlcore.research.store import ResearchStore
from rlcore.research.workflows import Approval, ApprovalRequiredError, ResearchWorkflow

if TYPE_CHECKING:
    from pathlib import Path


class PaidMockProvider(MockProvider):
    """MockProvider that reports itself as paid, for gate tests only."""

    @property
    def is_paid(self) -> bool:
        return True


@pytest.fixture
def store(tmp_path: Path) -> ResearchStore:
    return ResearchStore(tmp_path / "store")


@pytest.fixture
def workflow(store: ResearchStore) -> ResearchWorkflow:
    return ResearchWorkflow(store, MockProvider())


def _write_run_dir(tmp_path: Path, run_id: str = "run-abc") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "algo": "ppo",
                "env_id": "CartPole-v1",
                "seed": 0,
                "metadata": {"commit": "abc123"},
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps({"final_eval_return_mean": 500.0, "total_env_steps": 2048})
    )
    return run_dir


def test_register_run_reads_real_records(
    workflow: ResearchWorkflow, store: ResearchStore, tmp_path: Path
) -> None:
    run_dir = _write_run_dir(tmp_path)
    reference = workflow.register_run(run_dir)
    assert reference.run_id == "run-abc"
    assert reference.algo == "ppo"
    assert reference.commit == "abc123"
    assert ("final_eval_return_mean", "500.0") in reference.summary
    assert store.get(reference.entity_id) == reference


def test_register_run_missing_records_fails(workflow: ResearchWorkflow, tmp_path: Path) -> None:
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        workflow.register_run(empty)


def test_claim_from_sections_carries_citations(
    workflow: ResearchWorkflow, store: ResearchStore, tmp_path: Path
) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# GAE\n\nAdvantage estimation reduces variance.\n")
    ingest_path(path, store)
    hits = retrieve("advantage", load_documents(store))
    claim = workflow.claim_from_sections("GAE reduces variance.", hits)
    assert claim.source_refs
    assert claim.source_refs[0].section == "GAE"


def test_claim_without_sections_fails(workflow: ResearchWorkflow) -> None:
    with pytest.raises(ValueError, match="cite"):
        workflow.claim_from_sections("ungrounded", ())


def test_observation_links_runs(workflow: ResearchWorkflow, tmp_path: Path) -> None:
    reference = workflow.register_run(_write_run_dir(tmp_path))
    observation = workflow.observe("Solved at 500.", (reference,))
    assert observation.run_ids == ("run-abc",)
    assert observation.links == (reference.entity_id,)


def test_gate_expensive_plan_requires_approval(workflow: ResearchWorkflow) -> None:
    small = create(ExperimentPlan, hypothesis_id="h", description="local", cost_class="small")
    workflow.check_plan_executable(small)  # no approval needed
    big = create(ExperimentPlan, hypothesis_id="h", description="sweep", cost_class="large")
    with pytest.raises(ApprovalRequiredError):
        workflow.check_plan_executable(big)
    approval = Approval(action="run-experiment", subject=big.entity_id, granted_by="human")
    workflow.check_plan_executable(big, approval=approval)


def test_gate_paid_provider_requires_approval(store: ResearchStore) -> None:
    paid = ResearchWorkflow(store, PaidMockProvider())
    with pytest.raises(ApprovalRequiredError):
        paid.summarize_evidence("q", ())
    approval = Approval(action="paid-llm", subject="mock", granted_by="human")
    analysis = paid.summarize_evidence("q", (), approval=approval)
    assert analysis.provenance == "llm:mock"


def test_free_provider_needs_no_approval(workflow: ResearchWorkflow) -> None:
    analysis = workflow.summarize_evidence("q", ())
    assert analysis.review_state == "draft"
    assert analysis.provenance == "llm:mock"


def test_llm_analysis_is_born_draft_with_llm_provenance(
    workflow: ResearchWorkflow, store: ResearchStore, tmp_path: Path
) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# GAE\n\nAdvantage estimation reduces variance.\n")
    ingest_path(path, store)
    hits = retrieve("variance", load_documents(store))
    analysis = workflow.summarize_evidence("what reduces variance?", hits)
    assert analysis.review_state == "draft"
    assert analysis.provenance.startswith("llm:")
    assert "lexical retrieval" in analysis.method


def test_gate_conclusion_requires_run_grounded_analysis(
    workflow: ResearchWorkflow,
) -> None:
    ungrounded = create(Analysis, text="thoughts")
    with pytest.raises(ValueError, match="Refusing to conclude"):
        workflow.conclude("done", (ungrounded,), approval=None)
    with pytest.raises(ValueError, match="at least one Analysis"):
        workflow.conclude("done", ())


def test_gate_conclusion_requires_human_approval(
    workflow: ResearchWorkflow, tmp_path: Path
) -> None:
    reference = workflow.register_run(_write_run_dir(tmp_path))
    observation = workflow.observe("Solved.", (reference,))
    analysis = workflow.analyze_runs("Consistent solve.", "eyeball", (observation,))
    assert analysis.run_ids == ("run-abc",)
    with pytest.raises(ApprovalRequiredError):
        workflow.conclude("It works.", (analysis,))
    approval = Approval(action="conclude", subject=analysis.entity_id, granted_by="human")
    conclusion = workflow.conclude("It works.", (analysis,), approval=approval)
    assert conclusion.analysis_ids == (analysis.entity_id,)


def test_gate_publication_requires_approval(
    workflow: ResearchWorkflow, store: ResearchStore, tmp_path: Path
) -> None:
    reference = workflow.register_run(_write_run_dir(tmp_path))
    report = workflow.draft_report("Demo", (reference.entity_id,))
    assert report.published is False
    assert reference.entity_id in report.body_markdown
    with pytest.raises(ApprovalRequiredError):
        workflow.publish_report(report)
    approval = Approval(action="publish", subject=report.entity_id, granted_by="human")
    published = workflow.publish_report(report, approval=approval)
    assert published.published is True
    assert published.review_state == "approved"
    assert published.version == report.version + 1
    assert "approved by human" in published.provenance


def test_scripted_mock_provider_is_consumed_in_order(store: ResearchStore) -> None:
    provider = MockProvider(responses=("first", "second"))
    workflow = ResearchWorkflow(store, provider)
    assert workflow.summarize_evidence("a", ()).text == "first"
    assert workflow.summarize_evidence("b", ()).text == "second"
    with pytest.raises(RuntimeError, match="no scripted responses"):
        workflow.summarize_evidence("c", ())


def test_mock_digest_echo_is_deterministic() -> None:
    provider = MockProvider()
    assert provider.complete("same prompt") == provider.complete("same prompt")
    assert provider.complete("same prompt") != provider.complete("other prompt")


def test_anthropic_provider_refuses_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlcore.research.llm import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_anthropic_provider_is_paid_and_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlcore.research.llm import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    provider = AnthropicProvider()
    assert provider.is_paid is True
    assert provider.name == "anthropic:claude-opus-4-8"

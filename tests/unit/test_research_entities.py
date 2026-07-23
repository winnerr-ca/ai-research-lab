"""Unit tests for research entities: schema, versioning, serialization."""

from __future__ import annotations

import dataclasses

import pytest

from rlcore.research.entities import (
    ENTITY_KINDS,
    Analysis,
    Claim,
    Conclusion,
    Entity,
    ExperimentPlan,
    Observation,
    ResearchQuestion,
    SourceRef,
    create,
    entity_from_json,
    revise,
    to_json_dict,
)

REQUIRED_FIELD_SAMPLES: dict[str, dict[str, object]] = {
    "paper": {"title": "Sample"},
    "citation": {"paper_id": "paper-x", "text": "see §3"},
    "claim": {"text": "grounded", "run_ids": ("run-1",)},
    "researchquestion": {"text": "does X help?"},
    "hypothesis": {"text": "X helps"},
    "experimentplan": {"hypothesis_id": "hypothesis-x", "description": "test X"},
    "baseline": {"name": "default"},
    "ablation": {"plan_id": "experimentplan-x", "component": "X"},
    "runreference": {"run_id": "run-1", "run_dir": "/tmp/run"},
    "observation": {"text": "saw Y", "run_ids": ("run-1",)},
    "analysis": {"text": "Y because Z"},
    "conclusion": {"text": "Z", "analysis_ids": ("analysis-x",)},
    "limitation": {"text": "only CartPole"},
    "researchreport": {"title": "R", "body_markdown": "# R"},
}


def test_all_fourteen_kinds_registered() -> None:
    assert len(ENTITY_KINDS) == 14
    assert set(REQUIRED_FIELD_SAMPLES) == set(ENTITY_KINDS)


@pytest.mark.parametrize("kind", sorted(ENTITY_KINDS))
def test_create_roundtrips_through_json(kind: str) -> None:
    entity = create(ENTITY_KINDS[kind], **REQUIRED_FIELD_SAMPLES[kind])
    payload = to_json_dict(entity)
    assert payload["kind"] == kind
    restored = entity_from_json(payload)
    assert restored == entity


def test_create_stamps_spine_fields() -> None:
    question = create(ResearchQuestion, text="q")
    assert question.entity_id.startswith("researchquestion-")
    assert question.version == 1
    assert question.review_state == "draft"
    assert question.created_at == question.updated_at


def test_entities_are_immutable() -> None:
    question = create(ResearchQuestion, text="q")
    with pytest.raises(dataclasses.FrozenInstanceError):
        question.text = "changed"  # type: ignore[misc]


def test_revise_bumps_version_and_keeps_id() -> None:
    question = create(ResearchQuestion, text="q")
    revised = revise(question, text="q, sharper")
    assert revised.entity_id == question.entity_id
    assert revised.version == 2
    assert revised.created_at == question.created_at
    assert revised.text == "q, sharper"


def test_revise_refuses_locked_fields() -> None:
    question = create(ResearchQuestion, text="q")
    with pytest.raises(ValueError, match="version"):
        revise(question, version=7)


def test_invalid_review_state_rejected() -> None:
    with pytest.raises(ValueError, match="review_state"):
        create(ResearchQuestion, text="q", review_state="maybe")


def test_claim_requires_grounding() -> None:
    with pytest.raises(ValueError, match="cite"):
        create(Claim, text="ungrounded")


def test_claim_accepts_source_refs() -> None:
    ref = SourceRef(document_id="doc-1", section="Derivation")
    claim = create(Claim, text="grounded", source_refs=(ref,))
    restored = entity_from_json(to_json_dict(claim))
    assert isinstance(restored, Claim)
    assert restored.source_refs == (ref,)


def test_observation_requires_runs() -> None:
    with pytest.raises(ValueError, match="run ID"):
        create(Observation, text="saw nothing")


def test_conclusion_requires_analysis() -> None:
    with pytest.raises(ValueError, match="Analysis"):
        create(Conclusion, text="done")


def test_experiment_plan_validates_cost_class() -> None:
    with pytest.raises(ValueError, match="cost_class"):
        create(ExperimentPlan, hypothesis_id="h", description="d", cost_class="huge")


def test_unknown_kind_rejected_on_load() -> None:
    with pytest.raises(ValueError, match="Unknown entity kind"):
        entity_from_json({"kind": "poem", "entity_id": "poem-1"})


def test_optional_int_field_roundtrips_none_and_value() -> None:
    from rlcore.research.entities import RunReference

    with_seed = create(RunReference, run_id="r", run_dir="/x", seed=3)
    without = create(RunReference, run_id="r", run_dir="/x")
    assert entity_from_json(to_json_dict(with_seed)) == with_seed
    restored = entity_from_json(to_json_dict(without))
    assert isinstance(restored, RunReference)
    assert restored.seed is None


def test_kind_property_matches_registry() -> None:
    analysis = create(Analysis, text="a")
    assert analysis.kind == "analysis"
    assert isinstance(analysis, Entity)

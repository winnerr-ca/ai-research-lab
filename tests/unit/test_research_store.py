"""Unit tests for the research store: history, conflicts, deletion gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rlcore.research.entities import ResearchQuestion, ResearchReport, create, revise
from rlcore.research.store import ResearchStore
from rlcore.research.workflows import Approval

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> ResearchStore:
    return ResearchStore(tmp_path / "store")


def test_save_get_roundtrip(store: ResearchStore) -> None:
    question = create(ResearchQuestion, text="q")
    store.save(question)
    assert store.get(question.entity_id) == question


def test_update_appends_history(store: ResearchStore) -> None:
    question = create(ResearchQuestion, text="v1 text")
    store.save(question)
    revised = revise(question, text="v2 text")
    store.save(revised)
    assert store.get(question.entity_id) == revised
    history = store.history(question.entity_id)
    assert len(history) == 1
    assert history[0] == question


def test_version_conflict_rejected(store: ResearchStore) -> None:
    question = create(ResearchQuestion, text="q")
    store.save(question)
    with pytest.raises(ValueError, match="Version conflict"):
        store.save(question)  # same version again
    stale_jump = revise(revise(question, text="a"), text="b")
    with pytest.raises(ValueError, match="Version conflict"):
        store.save(stale_jump)  # v3 while store has v1


def test_new_entity_must_be_version_one(store: ResearchStore) -> None:
    question = revise(create(ResearchQuestion, text="q"), text="r")
    with pytest.raises(ValueError, match="version 1"):
        store.save(question)


def test_get_missing_raises_keyerror(store: ResearchStore) -> None:
    with pytest.raises(KeyError):
        store.get("researchquestion-missing")


def test_list_ids_filters_by_kind(store: ResearchStore) -> None:
    question = create(ResearchQuestion, text="q")
    store.save(question)
    assert store.list_ids("researchquestion") == (question.entity_id,)
    assert store.list_ids("paper") == ()
    assert question.entity_id in store.list_ids()


def test_delete_requires_matching_approval(store: ResearchStore) -> None:
    question = create(ResearchQuestion, text="q")
    store.save(question)
    wrong = Approval(action="publish", subject=question.entity_id, granted_by="human")
    with pytest.raises(PermissionError, match="delete"):
        store.delete(question.entity_id, approval=wrong)
    mismatched = Approval(action="delete", subject="other-id", granted_by="human")
    with pytest.raises(PermissionError, match="delete"):
        store.delete(question.entity_id, approval=mismatched)
    good = Approval(action="delete", subject=question.entity_id, granted_by="human")
    store.delete(question.entity_id, approval=good)
    with pytest.raises(KeyError):
        store.get(question.entity_id)


def test_published_report_requires_approved_state(store: ResearchStore) -> None:
    report = create(ResearchReport, title="R", body_markdown="# R", published=True)
    with pytest.raises(ValueError, match="human-approved"):
        store.save(report)

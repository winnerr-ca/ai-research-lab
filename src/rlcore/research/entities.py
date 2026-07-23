"""Typed, versioned research entities (ROADMAP M9).

Every entity carries the same audit spine: a stable ID (fixed for the
entity's whole life; versions change, the ID does not), UTC creation and
update timestamps, an integer version starting at 1, a provenance string
(``"human"``, ``"llm:<provider>:<model>"``, or ``"ingest:<path>"``), typed
links to other entities by ID, and a human-review state. LLM-produced
content is *always* born ``draft`` — nothing generated becomes
``approved`` without an explicit human action (ARCHITECTURE §8).

Grounding rules enforced at construction time:

- A :class:`Claim` must cite at least one source section or run ID.
- An :class:`Observation` must reference at least one run.
- A :class:`Conclusion` must link at least one :class:`Analysis`.

The store (:mod:`rlcore.research.store`) persists these as JSON with full
version history; serialization helpers live here so the entity schema and
its wire format stay in one file.
"""

from __future__ import annotations

import dataclasses
import types
import typing
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

REVIEW_STATES = ("draft", "approved", "rejected")

E = TypeVar("E", bound="Entity")


def utc_now() -> str:
    """Current UTC time in ISO-8601 (the timestamp format of every entity)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_entity_id(kind: str) -> str:
    """Fresh stable ID: ``<kind>-<12 hex chars>``."""
    return f"{kind}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceRef:
    """A pointer into an ingested document: section/page granularity.

    Attributes:
        document_id: ID of the ingested :class:`~rlcore.research.ingest.SourceDocument`.
        section: Section reference (Markdown heading, ``"page 3"``, or
            ``"lines 40-79"`` depending on the source kind).
        detail: Optional finer locator (e.g. a quoted phrase).
    """

    document_id: str
    section: str
    detail: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class Entity:
    """Common audit spine shared by every research entity."""

    entity_id: str
    created_at: str
    updated_at: str
    version: int = 1
    provenance: str = "human"
    links: tuple[str, ...] = ()
    review_state: str = "draft"

    def __post_init__(self) -> None:
        """Validate the spine fields."""
        if self.review_state not in REVIEW_STATES:
            raise ValueError(
                f"review_state must be one of {REVIEW_STATES}, got {self.review_state!r}."
            )
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}.")

    @property
    def kind(self) -> str:
        """Entity kind (lower-case class name)."""
        return type(self).__name__.lower()


@dataclass(frozen=True, slots=True, kw_only=True)
class Paper(Entity):
    """A source publication (usually created by ingestion)."""

    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    source_document_id: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation(Entity):
    """A specific citable location inside a paper."""

    paper_id: str
    text: str
    ref: SourceRef | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Claim(Entity):
    """A statement that MUST be grounded in a source or a run.

    Raises:
        ValueError: If neither ``source_refs`` nor ``run_ids`` is provided —
            ungrounded claims are not representable.
    """

    text: str
    source_refs: tuple[SourceRef, ...] = ()
    run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce grounding on top of the spine validation."""
        Entity.__post_init__(self)
        if not self.source_refs and not self.run_ids:
            raise ValueError(
                "A Claim must cite at least one source section or run ID; "
                "ungrounded claims are not representable (ARCHITECTURE §8)."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchQuestion(Entity):
    """An open question motivating experiments."""

    text: str
    motivation: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class Hypothesis(Entity):
    """A falsifiable prediction attached to a question.

    A hypothesis is never a conclusion: promotion requires an Analysis over
    real runs plus explicit human approval (see
    :meth:`rlcore.research.workflows.ResearchWorkflow.conclude`).
    """

    text: str
    question_id: str = ""
    testable_prediction: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentPlan(Entity):
    """A concrete plan to test a hypothesis.

    Attributes:
        cost_class: ``"small"`` (local, minutes), ``"moderate"``, or
            ``"large"``; anything above small requires human approval to
            execute (workflow gate).
    """

    hypothesis_id: str
    description: str
    algo: str = ""
    env_id: str = ""
    seeds: tuple[int, ...] = ()
    overrides: tuple[tuple[str, str], ...] = ()
    cost_class: str = "small"

    def __post_init__(self) -> None:
        """Validate the cost class."""
        Entity.__post_init__(self)
        if self.cost_class not in ("small", "moderate", "large"):
            raise ValueError(f"cost_class must be small/moderate/large, got {self.cost_class!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Baseline(Entity):
    """A reference configuration/result other runs are compared against."""

    name: str
    description: str = ""
    run_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Ablation(Entity):
    """A component-removal/variation study derived from a plan."""

    plan_id: str
    component: str
    variants: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class RunReference(Entity):
    """A pointer to a concrete rlcore run directory (the ground truth).

    Attributes:
        run_id: The run's ID from ``run.json``.
        run_dir: Path of the run directory at registration time.
        summary: Flat metric summary copied from ``result.json`` (values
            stringified for schema stability).
    """

    run_id: str
    run_dir: str
    algo: str = ""
    env_id: str = ""
    seed: int | None = None
    commit: str = ""
    summary: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation(Entity):
    """A factual note about what runs showed; must reference runs.

    Raises:
        ValueError: If ``run_ids`` is empty.
    """

    text: str
    run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce run grounding on top of the spine validation."""
        Entity.__post_init__(self)
        if not self.run_ids:
            raise ValueError("An Observation must reference at least one run ID.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Analysis(Entity):
    """Interpretation of observations, with the method stated."""

    text: str
    method: str = ""
    observation_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Conclusion(Entity):
    """A settled statement; must be backed by at least one Analysis.

    Raises:
        ValueError: If ``analysis_ids`` is empty.
    """

    text: str
    analysis_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce analysis grounding on top of the spine validation."""
        Entity.__post_init__(self)
        if not self.analysis_ids:
            raise ValueError("A Conclusion must link at least one Analysis.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Limitation(Entity):
    """An explicit scope limit attached to other entities."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchReport(Entity):
    """An assembled report over stored entities.

    ``published`` may only become True through the publication approval
    gate; the store refuses to persist a published report whose
    ``review_state`` is not ``approved``.
    """

    title: str
    body_markdown: str
    entity_ids: tuple[str, ...] = ()
    published: bool = False


ENTITY_KINDS: dict[str, type[Entity]] = {
    cls.__name__.lower(): cls
    for cls in (
        Paper,
        Citation,
        Claim,
        ResearchQuestion,
        Hypothesis,
        ExperimentPlan,
        Baseline,
        Ablation,
        RunReference,
        Observation,
        Analysis,
        Conclusion,
        Limitation,
        ResearchReport,
    )
}


def create(kind_cls: type[E], **fields: Any) -> E:  # noqa: ANN401
    """Construct a new entity with a fresh ID and timestamps.

    Args:
        kind_cls: The entity class.
        **fields: Entity-specific fields (plus optional ``provenance``,
            ``links``, ``review_state``, and ``entity_id`` override for
            deterministic tests).
    """
    now = utc_now()
    fields.setdefault("entity_id", new_entity_id(kind_cls.__name__.lower()))
    fields.setdefault("created_at", now)
    fields.setdefault("updated_at", now)
    return kind_cls(**fields)


def revise(entity: E, **changes: Any) -> E:  # noqa: ANN401
    """Produce the next version of an entity (same ID, version + 1).

    ``entity_id``, ``created_at``, and ``version`` cannot be overridden;
    ``updated_at`` is stamped automatically.
    """
    for locked in ("entity_id", "created_at", "version"):
        if locked in changes:
            raise ValueError(f"{locked} cannot be changed by revise().")
    return dataclasses.replace(entity, version=entity.version + 1, updated_at=utc_now(), **changes)


def to_json_dict(entity: Entity) -> dict[str, Any]:
    """Serialize an entity to a JSON-compatible dict (with its kind)."""
    payload = dataclasses.asdict(entity)
    payload["kind"] = entity.kind
    return payload


def _decode_value(annotation: Any, value: Any) -> Any:  # noqa: ANN401
    """Decode one JSON value against a (resolved) type annotation."""
    origin = typing.get_origin(annotation)
    if annotation is SourceRef:
        return SourceRef(**value)
    if origin is tuple:
        item_type = typing.get_args(annotation)[0]
        return tuple(_decode_value(item_type, item) for item in value)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _decode_value(non_none[0], value)
    return value


def entity_from_json(payload: dict[str, Any]) -> Entity:
    """Deserialize an entity dict written by :func:`to_json_dict`.

    Raises:
        ValueError: On an unknown ``kind``.
    """
    data = dict(payload)
    kind = data.pop("kind", None)
    if kind not in ENTITY_KINDS:
        raise ValueError(f"Unknown entity kind {kind!r}; expected one of {sorted(ENTITY_KINDS)}.")
    cls = ENTITY_KINDS[kind]
    hints = typing.get_type_hints(cls)
    kwargs = {
        f.name: _decode_value(hints[f.name], data[f.name])
        for f in dataclasses.fields(cls)
        if f.name in data
    }
    return cls(**kwargs)


__all__ = [
    "ENTITY_KINDS",
    "REVIEW_STATES",
    "Ablation",
    "Analysis",
    "Baseline",
    "Citation",
    "Claim",
    "Conclusion",
    "Entity",
    "ExperimentPlan",
    "Hypothesis",
    "Limitation",
    "Observation",
    "Paper",
    "ResearchQuestion",
    "ResearchReport",
    "RunReference",
    "SourceRef",
    "create",
    "entity_from_json",
    "new_entity_id",
    "revise",
    "to_json_dict",
    "utc_now",
]

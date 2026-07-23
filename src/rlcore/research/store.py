"""JSON-file persistence for research entities, with version history.

Layout under the store root::

    entities/<kind>/<entity_id>.json   # {"current": {...}, "history": [...]}
    sources/<document_id><ext>         # raw ingested bytes (see ingest)

Every update appends the previous version to ``history`` — versions are
never overwritten in place, so the audit trail survives. Writes are
atomic (tmp + rename), matching the platform's checkpoint discipline.

Destructive deletion is gated: :meth:`ResearchStore.delete` requires an
explicit human :class:`~rlcore.research.workflows.Approval` for the
``delete`` action (ARCHITECTURE §8 — the store will not remove recorded
research without a human decision on record).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from rlcore.research.entities import Entity, entity_from_json, to_json_dict

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.research.workflows import Approval


class ResearchStore:
    """Filesystem-backed entity store with per-entity version history."""

    def __init__(self, root: Path) -> None:
        """Open (creating if needed) a store rooted at ``root``."""
        self._root = root
        (root / "entities").mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """The store's root directory."""
        return self._root

    def _path_for(self, entity_id: str) -> Path:
        kind = entity_id.rsplit("-", 1)[0]
        return self._root / "entities" / kind / f"{entity_id}.json"

    def save(self, entity: Entity) -> None:
        """Persist a new entity (version 1) or the next version of one.

        Raises:
            ValueError: If the version is not exactly current + 1 for an
                existing entity, or not 1 for a new one — the caller must
                go through :func:`rlcore.research.entities.revise`.
            ValueError: If a published report is saved without approval.
        """
        from rlcore.research.entities import ResearchReport

        if (
            isinstance(entity, ResearchReport)
            and entity.published
            and entity.review_state != "approved"
        ):
            raise ValueError(
                "A published ResearchReport must be human-approved; "
                "publication is an explicit approval gate."
            )
        path = self._path_for(entity.entity_id)
        if path.exists():
            doc = json.loads(path.read_text())
            current_version = int(doc["current"]["version"])
            if entity.version != current_version + 1:
                raise ValueError(
                    f"Version conflict for {entity.entity_id}: store has "
                    f"v{current_version}, save() got v{entity.version} "
                    f"(expected v{current_version + 1}; use revise())."
                )
            doc["history"].append(doc["current"])
            doc["current"] = to_json_dict(entity)
        else:
            if entity.version != 1:
                raise ValueError(
                    f"New entity {entity.entity_id} must be saved at version 1, "
                    f"got v{entity.version}."
                )
            doc = {"current": to_json_dict(entity), "history": []}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        os.replace(tmp, path)

    def get(self, entity_id: str) -> Entity:
        """Load the current version of an entity.

        Raises:
            KeyError: If the entity does not exist.
        """
        path = self._path_for(entity_id)
        if not path.exists():
            raise KeyError(f"No entity {entity_id!r} in store {self._root}.")
        doc = json.loads(path.read_text())
        return entity_from_json(doc["current"])

    def history(self, entity_id: str) -> tuple[Entity, ...]:
        """All prior versions of an entity, oldest first (excludes current)."""
        path = self._path_for(entity_id)
        if not path.exists():
            raise KeyError(f"No entity {entity_id!r} in store {self._root}.")
        doc = json.loads(path.read_text())
        return tuple(entity_from_json(item) for item in doc["history"])

    def list_ids(self, kind: str | None = None) -> tuple[str, ...]:
        """Sorted entity IDs, optionally restricted to one kind."""
        base = self._root / "entities"
        kinds = [kind] if kind is not None else sorted(p.name for p in base.iterdir())
        ids: list[str] = []
        for k in kinds:
            kind_dir = base / k
            if kind_dir.is_dir():
                ids.extend(sorted(p.stem for p in kind_dir.glob("*.json")))
        return tuple(ids)

    def delete(self, entity_id: str, *, approval: Approval) -> None:
        """Destructively remove an entity and its whole history.

        Requires a human approval record for the ``delete`` action naming
        this entity ID; this is the destructive-deletion gate.

        Raises:
            PermissionError: If the approval does not cover this deletion.
            KeyError: If the entity does not exist.
        """
        if approval.action != "delete" or approval.subject != entity_id:
            raise PermissionError(
                f"Deletion of {entity_id!r} requires a human approval record "
                f"with action='delete' and subject={entity_id!r}; got "
                f"action={approval.action!r}, subject={approval.subject!r}."
            )
        path = self._path_for(entity_id)
        if not path.exists():
            raise KeyError(f"No entity {entity_id!r} in store {self._root}.")
        path.unlink()

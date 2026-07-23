"""``rlcore-research`` command-line interface.

Read-mostly operations plus ingestion; anything gated (paid providers,
publication, deletion, conclusions) is deliberately *not* reachable from
this CLI — gates require constructing an explicit Approval in code, so a
shell one-liner can never publish or delete research records.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rlcore.research.ingest import ingest_path, load_documents, retrieve
from rlcore.research.store import ResearchStore

logger = logging.getLogger(__name__)


def research_main() -> None:
    """Argparse entry point for ``rlcore-research``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="rlcore research store operations.")
    parser.add_argument(
        "--store", type=Path, default=Path("research-store"), help="store root directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a .md/.txt/.pdf document")
    p_ingest.add_argument("path", type=Path)

    sub.add_parser("docs", help="list ingested documents")

    p_search = sub.add_parser("search", help="lexical retrieval with section refs")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=3)

    p_list = sub.add_parser("list", help="list stored entities")
    p_list.add_argument("--kind", default=None)

    p_show = sub.add_parser("show", help="show one entity (current version)")
    p_show.add_argument("entity_id")

    args = parser.parse_args()
    store = ResearchStore(args.store)

    if args.command == "ingest":
        document = ingest_path(args.path, store)
        logger.info(
            "ingested %s as %s (%d sections, sha256 %s)",
            args.path,
            document.document_id,
            len(document.sections),
            document.sha256[:12],
        )
    elif args.command == "docs":
        for document in load_documents(store):
            logger.info(
                "%s  %s  (%s, %d sections)",
                document.document_id,
                document.path,
                document.kind,
                len(document.sections),
            )
    elif args.command == "search":
        hits = retrieve(args.query, load_documents(store), top_k=args.top_k)
        if not hits:
            logger.info("no matching sections")
        for hit in hits:
            logger.info(
                "%.3f  [%s / %s]  %s",
                hit.score,
                hit.section.document_id,
                hit.section.ref,
                hit.section.text[:120].replace("\n", " "),
            )
    elif args.command == "list":
        for entity_id in store.list_ids(args.kind):
            logger.info("%s", entity_id)
    elif args.command == "show":
        entity = store.get(args.entity_id)
        logger.info(
            "%s (%s v%d, %s, provenance: %s)",
            entity.entity_id,
            entity.kind,
            entity.version,
            entity.review_state,
            entity.provenance,
        )
        for name, value in sorted(vars_of(entity).items()):
            logger.info("  %s: %s", name, value)


def vars_of(entity: object) -> dict[str, object]:
    """Dataclass fields as a dict (slots-safe; ``vars()`` fails on slots)."""
    import dataclasses

    return {f.name: getattr(entity, f.name) for f in dataclasses.fields(entity)}  # type: ignore[arg-type]

"""Document ingestion and source-grounded retrieval (ROADMAP M9).

Ingestion preserves the source: the raw bytes are copied into the store
(``sources/<document_id><ext>``) with a SHA-256 recorded, then split into
sections whose references survive into every downstream citation:

- **Markdown**: split at headings; each section's ref is its heading text
  and line range.
- **Plain text**: fixed-size line chunks; ref is the line range.
- **PDF**: one section per page via ``pypdf`` (optional ``research``
  extra); ref is ``page N``.

Retrieval is deliberately lexical and deterministic — term-overlap
scoring with stable tie-breaking, no embeddings and no network. It is
labeled as such; it grounds citations, it does not rank by meaning.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from rlcore.research.entities import SourceRef, new_entity_id, utc_now

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.research.store import ResearchStore

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TEXT_CHUNK_LINES = 40
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True, kw_only=True)
class Section:
    """One retrievable unit of an ingested document.

    Attributes:
        document_id: Owning document.
        ref: Human-readable locator (heading, line range, or page).
        start_line: 1-based start line (0 for PDF pages).
        text: The section's verbatim text.
    """

    document_id: str
    ref: str
    start_line: int
    text: str

    def source_ref(self, detail: str = "") -> SourceRef:
        """The citation pointer for this section."""
        return SourceRef(document_id=self.document_id, section=self.ref, detail=detail)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceDocument:
    """An ingested document with preserved raw source and sections."""

    document_id: str
    path: str
    kind: str
    sha256: str
    ingested_at: str
    sections: tuple[Section, ...]


def _sections_from_markdown(document_id: str, text: str) -> tuple[Section, ...]:
    sections: list[Section] = []
    current_ref = "(preamble)"
    current_start = 1
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                Section(
                    document_id=document_id,
                    ref=current_ref,
                    start_line=current_start,
                    text=body,
                )
            )

    for lineno, line in enumerate(text.splitlines(), start=1):
        match = _HEADING.match(line)
        if match:
            flush()
            current_ref = match.group(2).strip()
            current_start = lineno
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return tuple(sections)


def _sections_from_text(document_id: str, text: str) -> tuple[Section, ...]:
    lines = text.splitlines()
    sections = []
    for start in range(0, len(lines), _TEXT_CHUNK_LINES):
        chunk = lines[start : start + _TEXT_CHUNK_LINES]
        body = "\n".join(chunk).strip()
        if body:
            end = min(start + _TEXT_CHUNK_LINES, len(lines))
            sections.append(
                Section(
                    document_id=document_id,
                    ref=f"lines {start + 1}-{end}",
                    start_line=start + 1,
                    text=body,
                )
            )
    return tuple(sections)


def _sections_from_pdf(document_id: str, path: Path) -> tuple[Section, ...]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "PDF ingestion requires pypdf; install the 'research' extra (uv sync --extra research)."
        ) from exc
    reader = PdfReader(str(path))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        body = (page.extract_text() or "").strip()
        if body:
            sections.append(
                Section(
                    document_id=document_id,
                    ref=f"page {page_number}",
                    start_line=0,
                    text=body,
                )
            )
    return tuple(sections)


def ingest_path(path: Path, store: ResearchStore) -> SourceDocument:
    """Ingest a Markdown/text/PDF file, preserving its raw source.

    The raw bytes are copied to ``<store>/sources/<document_id><ext>``,
    the parsed document (with sections) to
    ``<store>/sources/<document_id>.doc.json``.

    Raises:
        ValueError: For unsupported file extensions.
        FileNotFoundError: If ``path`` does not exist.
    """
    suffix = path.suffix.lower()
    kinds = {".md": "markdown", ".markdown": "markdown", ".txt": "text", ".pdf": "pdf"}
    if suffix not in kinds:
        raise ValueError(f"Unsupported source type {suffix!r}; expected .md, .txt, or .pdf.")
    raw = path.read_bytes()
    document_id = new_entity_id("doc")
    kind = kinds[suffix]
    sources_dir = store.root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    preserved = sources_dir / f"{document_id}{suffix}"
    preserved.write_bytes(raw)
    if kind == "markdown":
        sections = _sections_from_markdown(document_id, raw.decode("utf-8"))
    elif kind == "text":
        sections = _sections_from_text(document_id, raw.decode("utf-8"))
    else:
        sections = _sections_from_pdf(document_id, preserved)

    document = SourceDocument(
        document_id=document_id,
        path=str(path),
        kind=kind,
        sha256=hashlib.sha256(raw).hexdigest(),
        ingested_at=utc_now(),
        sections=sections,
    )
    (sources_dir / f"{document_id}.doc.json").write_text(json.dumps(asdict(document), indent=2))
    return document


def load_documents(store: ResearchStore) -> tuple[SourceDocument, ...]:
    """Load every ingested document in a store (sorted by ID)."""
    sources_dir = store.root / "sources"
    if not sources_dir.is_dir():
        return ()
    documents = []
    for doc_path in sorted(sources_dir.glob("*.doc.json")):
        payload = json.loads(doc_path.read_text())
        payload["sections"] = tuple(Section(**s) for s in payload["sections"])
        documents.append(SourceDocument(**payload))
    return tuple(documents)


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedSection:
    """One retrieval hit: a section plus its lexical score."""

    section: Section
    score: float


def retrieve(
    query: str, documents: tuple[SourceDocument, ...], *, top_k: int = 3
) -> tuple[RetrievedSection, ...]:
    """Deterministic lexical retrieval over ingested sections.

    Scoring: count of query-term occurrences in the section, normalized by
    the square root of the section's token count (long sections don't win
    by length alone). Ties break on (document_id, ref) so results are
    stable across runs. Sections with zero overlap are never returned.
    """
    query_terms = _TOKEN.findall(query.lower())
    if not query_terms:
        return ()
    scored: list[RetrievedSection] = []
    for document in documents:
        for section in document.sections:
            tokens = _TOKEN.findall(section.text.lower())
            if not tokens:
                continue
            hits = sum(tokens.count(term) for term in set(query_terms))
            if hits:
                scored.append(RetrievedSection(section=section, score=hits / len(tokens) ** 0.5))
    scored.sort(key=lambda r: (-r.score, r.section.document_id, r.section.ref))
    return tuple(scored[:top_k])

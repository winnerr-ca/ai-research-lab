"""Unit tests for ingestion (markdown/text/pdf) and lexical retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rlcore.research.ingest import ingest_path, load_documents, retrieve
from rlcore.research.store import ResearchStore

if TYPE_CHECKING:
    from pathlib import Path


def _minimal_pdf(text: str = "Advantage estimation") -> bytes:
    """A valid single-page PDF with a real xref table (no writer dependency)."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(content) + 1, content),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


MARKDOWN = """# Title

Intro paragraph about policy gradients.

## Derivation

The advantage function reduces variance.

## Tests

Hand-computed GAE cases.
"""


@pytest.fixture
def store(tmp_path: Path) -> ResearchStore:
    return ResearchStore(tmp_path / "store")


def test_markdown_sections_carry_heading_refs(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(MARKDOWN)
    document = ingest_path(path, store)
    assert document.kind == "markdown"
    refs = [section.ref for section in document.sections]
    assert refs == ["Title", "Derivation", "Tests"]
    derivation = document.sections[1]
    assert "advantage function" in derivation.text
    assert derivation.start_line == 5
    source_ref = derivation.source_ref()
    assert source_ref.document_id == document.document_id
    assert source_ref.section == "Derivation"


def test_text_chunks_carry_line_ranges(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("\n".join(f"line {i}" for i in range(1, 101)))
    document = ingest_path(path, store)
    assert document.kind == "text"
    assert [section.ref for section in document.sections] == [
        "lines 1-40",
        "lines 41-80",
        "lines 81-100",
    ]


def test_pdf_sections_carry_page_refs(store: ResearchStore, tmp_path: Path) -> None:
    pytest.importorskip("pypdf", reason="PDF ingestion needs the 'research' extra")
    path = tmp_path / "paper.pdf"
    path.write_bytes(_minimal_pdf())
    document = ingest_path(path, store)
    assert document.kind == "pdf"
    assert len(document.sections) == 1
    assert document.sections[0].ref == "page 1"
    assert "Advantage estimation" in document.sections[0].text


def test_raw_source_preserved_with_sha(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(MARKDOWN)
    document = ingest_path(path, store)
    preserved = store.root / "sources" / f"{document.document_id}.md"
    assert preserved.read_text() == MARKDOWN
    assert len(document.sha256) == 64


def test_unsupported_extension_rejected(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n")
    with pytest.raises(ValueError, match="Unsupported source type"):
        ingest_path(path, store)


def test_load_documents_roundtrips(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(MARKDOWN)
    document = ingest_path(path, store)
    loaded = load_documents(store)
    assert loaded == (document,)


def test_retrieve_ranks_matching_section_first(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(MARKDOWN)
    ingest_path(path, store)
    hits = retrieve("advantage variance", load_documents(store), top_k=2)
    assert hits
    assert hits[0].section.ref == "Derivation"
    assert hits[0].score > 0


def test_retrieve_is_deterministic_and_bounded(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(MARKDOWN)
    ingest_path(path, store)
    documents = load_documents(store)
    first = retrieve("policy", documents, top_k=1)
    second = retrieve("policy", documents, top_k=1)
    assert first == second
    assert len(first) <= 1


def test_retrieve_empty_query_and_no_overlap(store: ResearchStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(MARKDOWN)
    ingest_path(path, store)
    documents = load_documents(store)
    assert retrieve("", documents) == ()
    assert retrieve("zebra quantum", documents) == ()

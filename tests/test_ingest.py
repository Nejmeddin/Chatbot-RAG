"""Tests for document loading and chunking."""

from __future__ import annotations

import pytest

from rag.ingest import (
    Chunk,
    Document,
    build_corpus,
    chunk_documents,
    load_documents,
)


def test_chunking_preserves_provenance():
    document = Document(
        text="alpha " * 400, source_file="report.pdf", file_type="pdf", page=7
    )
    chunks = chunk_documents([document], chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(chunk.source_file == "report.pdf" for chunk in chunks)
    assert all(chunk.page == 7 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_respect_size_limit():
    document = Document(text="word " * 500, source_file="a.txt", file_type="txt")
    chunks = chunk_documents([document], chunk_size=100, chunk_overlap=20)
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_content_hash_is_stable_and_distinguishes_sources():
    a = Chunk(text="same text", source_file="one.txt", file_type="txt", chunk_index=0)
    same = Chunk(text="same text", source_file="one.txt", file_type="txt", chunk_index=0)
    other = Chunk(text="same text", source_file="two.txt", file_type="txt", chunk_index=0)

    # Identical chunks hash alike, which is what makes ingestion idempotent.
    assert a.content_hash == same.content_hash
    # The same sentence in a different file stays distinct, so both get cited.
    assert a.content_hash != other.content_hash


def test_citation_includes_page_only_for_paged_documents():
    paged = Chunk("t", source_file="r.pdf", file_type="pdf", chunk_index=0, page=3)
    flat = Chunk("t", source_file="notes.txt", file_type="txt", chunk_index=0)
    assert paged.citation == "r.pdf p.3"
    assert flat.citation == "notes.txt"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_documents([], chunk_size=100, chunk_overlap=100)


def test_missing_data_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_non_utf8_text_file_still_loads(tmp_path):
    # Corpora from older Windows tooling are often cp1252, not UTF-8.
    (tmp_path / "latin.txt").write_bytes("caf\xe9 pr\xe9sentation".encode("cp1252"))
    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert "café" in documents[0].text


def test_unsupported_and_empty_files_are_skipped(tmp_path):
    (tmp_path / "notes.txt").write_text("real content", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "sheet.xlsx").write_bytes(b"\x00binary")
    (tmp_path / "Thumbs.db").write_bytes(b"\x00")

    documents = load_documents(tmp_path)
    assert [document.source_file for document in documents] == ["notes.txt"]


def test_build_corpus_end_to_end(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.txt").write_text("sentence. " * 80, encoding="utf-8")

    chunks = build_corpus(tmp_path, chunk_size=120, chunk_overlap=20)

    assert chunks
    assert all(chunk.source_file == "deep.txt" for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)

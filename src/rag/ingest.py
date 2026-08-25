"""Load documents from disk and split them into embeddable chunks.

Every chunk carries its provenance (source file, page, position). That
metadata survives all the way into PostgreSQL, which is what lets the chatbot
cite its sources instead of asserting answers from nowhere.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".txt")

# Corpora produced by older Windows tooling are rarely UTF-8. Try the most
# likely encodings in order rather than failing the whole file.
TEXT_ENCODINGS = ("utf-8", "cp1252", "latin-1")


@dataclass(frozen=True)
class Document:
    """One page of a PDF, or one whole text file."""

    text: str
    source_file: str
    file_type: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    """A slice of a document, sized for embedding."""

    text: str
    source_file: str
    file_type: str
    chunk_index: int
    page: int | None = None

    @property
    def content_hash(self) -> str:
        """Stable identity for this chunk, used to make ingestion idempotent.

        Keyed on provenance *and* text, so re-ingesting an unchanged corpus
        inserts nothing, while the same sentence appearing in two different
        files is still stored (and cited) twice.
        """
        payload = f"{self.source_file}|{self.page}|{self.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def citation(self) -> str:
        """Short human-readable source label, e.g. ``report.pdf p.4``."""
        return f"{self.source_file} p.{self.page}" if self.page else self.source_file


def _read_text_file(path: Path) -> str:
    """Read a text file, trying a few encodings before giving up."""
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Last resort: never let one unreadable file abort the whole corpus.
    logger.warning("Falling back to lossy decoding for %s", path.name)
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader  # imported lazily to keep module import cheap

    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # scanned/image-only page, nothing to embed
        documents.append(
            Document(text=text, source_file=path.name, file_type="pdf", page=page_number)
        )
    return documents


def _load_txt(path: Path) -> list[Document]:
    text = _read_text_file(path).strip()
    if not text:
        return []
    return [Document(text=text, source_file=path.name, file_type="txt")]


def load_documents(data_dir: Path | str) -> list[Document]:
    """Recursively load every supported file under ``data_dir``."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    logger.info("Found %d file(s) to load in %s", len(paths), data_dir)

    documents: list[Document] = []
    for path in paths:
        loader = _load_pdf if path.suffix.lower() == ".pdf" else _load_txt
        try:
            loaded = loader(path)
        except Exception:
            # One malformed PDF should not cost us the other 90 files.
            logger.exception("Skipping unreadable file: %s", path.name)
            continue
        documents.extend(loaded)
        logger.debug("Loaded %d document(s) from %s", len(loaded), path.name)

    logger.info("Loaded %d document(s) total", len(documents))
    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Chunk]:
    """Split documents into overlapping chunks, preserving their provenance."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Chunk] = []
    for document in documents:
        for index, piece in enumerate(splitter.split_text(document.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source_file=document.source_file,
                    file_type=document.file_type,
                    chunk_index=index,
                    page=document.page,
                )
            )

    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def build_corpus(
    data_dir: Path | str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Chunk]:
    """Convenience wrapper: load every file and chunk it in one call."""
    return chunk_documents(load_documents(data_dir), chunk_size, chunk_overlap)

"""PostgreSQL persistence for chunks, their metadata and their embeddings.

Embeddings live in a ``DOUBLE PRECISION[]`` column rather than a ``pgvector``
column, so the project runs on a stock PostgreSQL install with no extension to
compile. See the README for the trade-off this implies at scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg import sql

from .embeddings import l2_normalize
from .ingest import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredCorpus:
    """Everything needed to rank chunks against a query, held in memory."""

    ids: list[int]
    texts: list[str]
    sources: list[str]
    pages: list[int | None]
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalised

    def __len__(self) -> int:
        return len(self.ids)


class DocumentStore:
    """Create the schema, write chunks, and read the corpus back."""

    def __init__(self, dsn: str, table_name: str = "documents") -> None:
        self.dsn = dsn
        self.table_name = table_name
        # Identifiers cannot be passed as query parameters, so compose them
        # through psycopg's SQL builder rather than formatting them into text.
        self._table = sql.Identifier(table_name)

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn)

    def create_schema(self) -> None:
        """Create the table if absent, and add metadata columns if missing.

        The ALTER statements let a database created by the original prototype
        (which stored only text and embedding) be upgraded in place, instead of
        being dropped and rebuilt.
        """
        create = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                id              SERIAL PRIMARY KEY,
                corpus_text     TEXT NOT NULL,
                embedding_float DOUBLE PRECISION[]
            )
            """
        ).format(table=self._table)

        migrations = [
            sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source_file TEXT"),
            sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS file_type TEXT"),
            sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS page INTEGER"),
            sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS chunk_index INTEGER"),
            sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS content_hash TEXT"),
            sql.SQL(
                "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS created_at "
                "TIMESTAMPTZ DEFAULT now()"
            ),
        ]

        # The unique index is what makes ingestion idempotent: re-running it over
        # an unchanged corpus inserts nothing instead of duplicating every chunk.
        index = sql.SQL(
            "CREATE UNIQUE INDEX IF NOT EXISTS {index} ON {table} (content_hash)"
        ).format(
            index=sql.Identifier(f"{self.table_name}_content_hash_key"),
            table=self._table,
        )

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(create)
            for statement in migrations:
                cur.execute(statement.format(table=self._table))
            cur.execute(index)
            conn.commit()
        logger.info("Schema ready on table %r", self.table_name)

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """Insert chunks with their vectors; return how many were newly added.

        Chunks already stored (same source, page and text) are skipped, so this
        is safe to run repeatedly.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if not chunks:
            return 0

        statement = sql.SQL(
            """
            INSERT INTO {table}
                (corpus_text, embedding_float, source_file, file_type, page,
                 chunk_index, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            """
        ).format(table=self._table)

        rows = [
            (
                chunk.text,
                embedding.tolist(),
                chunk.source_file,
                chunk.file_type,
                chunk.page,
                chunk.chunk_index,
                chunk.content_hash,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        before = self.count()
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(statement, rows)
            conn.commit()
        inserted = self.count() - before

        logger.info(
            "Inserted %d new chunk(s); %d already present",
            inserted,
            len(rows) - inserted,
        )
        return inserted

    def fetch_corpus(self) -> StoredCorpus:
        """Load every embedded chunk into memory, ready for ranking."""
        statement = sql.SQL(
            """
            SELECT id, corpus_text, source_file, page, embedding_float
            FROM {table}
            WHERE embedding_float IS NOT NULL
            ORDER BY id
            """
        ).format(table=self._table)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(statement)
            rows = cur.fetchall()

        if not rows:
            logger.warning("No embedded documents found in table %r", self.table_name)
            return StoredCorpus([], [], [], [], np.empty((0, 0), dtype=np.float32))

        ids: list[int] = []
        texts: list[str] = []
        sources: list[str] = []
        pages: list[int | None] = []
        vectors: list[np.ndarray] = []

        for doc_id, text, source, page, embedding in rows:
            ids.append(doc_id)
            texts.append(text)
            sources.append(source or "unknown source")
            pages.append(page)
            vectors.append(np.asarray(embedding, dtype=np.float32))

        # Normalise on read as well as on write: rows written by the original
        # prototype were stored un-normalised, and this keeps them comparable.
        matrix = l2_normalize(np.vstack(vectors))
        logger.info("Loaded %d embedded chunk(s) from PostgreSQL", len(ids))
        return StoredCorpus(ids, texts, sources, pages, matrix)

    def count(self) -> int:
        """Number of rows currently stored."""
        statement = sql.SQL("SELECT count(*) FROM {table}").format(table=self._table)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(statement)
            return cur.fetchone()[0]

    def clear(self) -> None:
        """Delete every row. Used by ``ingest --reset``."""
        statement = sql.SQL("TRUNCATE {table} RESTART IDENTITY").format(
            table=self._table
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(statement)
            conn.commit()
        logger.info("Cleared table %r", self.table_name)

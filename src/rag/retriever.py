"""Similarity search over the stored corpus.

Because every vector is L2-normalised, cosine similarity reduces to a dot
product, and the whole ranking step is one ``(n_chunks, dim) @ (dim,)`` matmul.

Scaling note: this loads the full corpus into memory and scores every chunk, so
retrieval is O(n_chunks) per query. That is the honest cost of storing vectors
in a plain array column instead of ``pgvector``. It is comfortably fast for the
few thousand chunks this project targets; past roughly 10k chunks the right move
is a pgvector column with an HNSW index, which turns the scan into a sublinear
index lookup. The corpus is cached after the first query so the database is not
re-read on every question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingManager, l2_normalize
from .store import DocumentStore, StoredCorpus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit, with enough provenance to cite it."""

    id: int
    text: str
    source_file: str
    page: int | None
    score: float
    rank: int

    @property
    def citation(self) -> str:
        """Short source label, e.g. ``report.pdf p.4``."""
        return f"{self.source_file} p.{self.page}" if self.page else self.source_file


class Retriever:
    """Finds the chunks most similar to a question."""

    def __init__(
        self,
        store: DocumentStore,
        embedder: EmbeddingManager,
        cache_corpus: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.cache_corpus = cache_corpus
        self._corpus: StoredCorpus | None = None

    @property
    def corpus(self) -> StoredCorpus:
        """The stored corpus, read from PostgreSQL at most once when cached."""
        if self._corpus is None or not self.cache_corpus:
            corpus = self.store.fetch_corpus()
            if not self.cache_corpus:
                return corpus
            self._corpus = corpus
        return self._corpus

    def refresh(self) -> None:
        """Drop the cached corpus so the next query re-reads the database."""
        self._corpus = None

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        corpus = self.corpus
        if len(corpus) == 0:
            logger.warning("Corpus is empty - run `ingest` before asking questions")
            return []

        query_vector = l2_normalize(self.embedder.encode_one(query))[0]
        if query_vector.shape[0] != corpus.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding has dimension {query_vector.shape[0]} but the "
                f"stored corpus has {corpus.embeddings.shape[1]}. The corpus was "
                "probably built with a different EMBEDDING_MODEL - re-run "
                "`ingest --reset`."
            )

        # Unit vectors, so this dot product *is* the cosine similarity.
        scores = corpus.embeddings @ query_vector

        # argpartition finds the top_k without fully sorting all n scores.
        k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        hits = []
        for rank, index in enumerate(top_indices, start=1):
            score = float(scores[index])
            if score < score_threshold:
                continue
            hits.append(
                RetrievedChunk(
                    id=corpus.ids[index],
                    text=corpus.texts[index],
                    source_file=corpus.sources[index],
                    page=corpus.pages[index],
                    score=score,
                    rank=rank,
                )
            )

        logger.info("Retrieved %d chunk(s) for %r", len(hits), query)
        return hits

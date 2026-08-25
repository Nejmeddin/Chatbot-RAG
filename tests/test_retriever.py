"""Tests for similarity ranking.

A stub embedder maps a handful of words to fixed vectors, so retrieval logic is
tested without downloading a transformer model or running PostgreSQL.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import l2_normalize
from rag.retriever import Retriever
from rag.store import StoredCorpus

VECTORS = {
    "cat": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "kitten": np.array([0.9, 0.1, 0.0], dtype=np.float32),
    "engine": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "galaxy": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


class StubEmbedder:
    """Deterministic stand-in for EmbeddingManager."""

    def encode_one(self, text: str) -> np.ndarray:
        return VECTORS[text]

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        return np.vstack([VECTORS[text] for text in texts])


class StubStore:
    """Returns a fixed corpus instead of querying PostgreSQL."""

    def __init__(self, corpus: StoredCorpus) -> None:
        self.corpus = corpus
        self.fetch_count = 0

    def fetch_corpus(self) -> StoredCorpus:
        self.fetch_count += 1
        return self.corpus


def make_corpus() -> StoredCorpus:
    return StoredCorpus(
        ids=[1, 2, 3],
        texts=["about cats", "about engines", "about galaxies"],
        sources=["animals.pdf", "machines.txt", "space.pdf"],
        pages=[4, None, 12],
        embeddings=l2_normalize(
            np.vstack([VECTORS["cat"], VECTORS["engine"], VECTORS["galaxy"]])
        ),
    )


def make_retriever(corpus: StoredCorpus | None = None) -> tuple[Retriever, StubStore]:
    store = StubStore(corpus if corpus is not None else make_corpus())
    return Retriever(store=store, embedder=StubEmbedder()), store


def test_most_similar_chunk_ranks_first():
    retriever, _ = make_retriever()
    hits = retriever.retrieve("kitten", top_k=3)

    assert hits[0].text == "about cats"
    assert hits[0].rank == 1
    # Scores must come back in descending order.
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )


def test_top_k_limits_results():
    retriever, _ = make_retriever()
    assert len(retriever.retrieve("cat", top_k=2)) == 2


def test_top_k_larger_than_corpus_is_safe():
    retriever, _ = make_retriever()
    assert len(retriever.retrieve("cat", top_k=99)) == 3


def test_score_threshold_filters_weak_matches():
    retriever, _ = make_retriever()
    hits = retriever.retrieve("cat", top_k=3, score_threshold=0.5)

    # Only the cat chunk clears the bar; engine and galaxy are orthogonal.
    assert len(hits) == 1
    assert hits[0].text == "about cats"


def test_retrieved_chunks_carry_citations():
    retriever, _ = make_retriever()
    hits = retriever.retrieve("cat", top_k=3)
    citations = {hit.citation for hit in hits}
    assert "animals.pdf p.4" in citations
    assert "machines.txt" in citations  # no page for a flat text file


def test_empty_corpus_returns_no_hits():
    empty = StoredCorpus([], [], [], [], np.empty((0, 0), dtype=np.float32))
    retriever, _ = make_retriever(empty)
    assert retriever.retrieve("cat", top_k=5) == []


def test_corpus_is_cached_between_queries():
    retriever, store = make_retriever()
    retriever.retrieve("cat")
    retriever.retrieve("engine")
    assert store.fetch_count == 1

    retriever.refresh()
    retriever.retrieve("cat")
    assert store.fetch_count == 2


def test_dimension_mismatch_explains_the_fix():
    corpus = StoredCorpus(
        ids=[1],
        texts=["stored with another model"],
        sources=["old.pdf"],
        pages=[1],
        embeddings=l2_normalize(np.ones((1, 8), dtype=np.float32)),
    )
    retriever, _ = make_retriever(corpus)

    with pytest.raises(ValueError, match="ingest --reset"):
        retriever.retrieve("cat")


@pytest.mark.parametrize("bad_query", ["", "   "])
def test_empty_query_is_rejected(bad_query):
    retriever, _ = make_retriever()
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve(bad_query)


def test_top_k_must_be_positive():
    retriever, _ = make_retriever()
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("cat", top_k=0)

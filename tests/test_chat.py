"""Tests for prompt construction and answer formatting."""

from __future__ import annotations

import numpy as np

from rag.chat import (
    NO_CONTEXT_MESSAGE,
    Answer,
    RagChatbot,
    build_context,
    build_user_prompt,
)
from rag.embeddings import l2_normalize
from rag.retriever import RetrievedChunk, Retriever
from rag.store import StoredCorpus


def make_chunk(rank: int, source: str, page: int | None, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=rank,
        text=f"content {rank}",
        source_file=source,
        page=page,
        score=score,
        rank=rank,
    )


class StubLLM:
    """Records the messages it was given and returns a canned reply."""

    def __init__(self, reply: str = "The answer is 42 [1].") -> None:
        self.reply = reply
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": self.reply})()


class EmptyStore:
    def fetch_corpus(self) -> StoredCorpus:
        return StoredCorpus([], [], [], [], np.empty((0, 0), dtype=np.float32))


class OneDocStore:
    def fetch_corpus(self) -> StoredCorpus:
        return StoredCorpus(
            ids=[1],
            texts=["the meaning of life"],
            sources=["guide.pdf"],
            pages=[7],
            embeddings=l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32)),
        )


class StubEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_context_is_numbered_and_labelled():
    chunks = [make_chunk(1, "a.pdf", 2, 0.9), make_chunk(2, "b.txt", None, 0.8)]
    context = build_context(chunks)

    assert "[1] (source: a.pdf p.2)" in context
    assert "[2] (source: b.txt)" in context
    assert "content 1" in context


def test_user_prompt_carries_question_and_context():
    prompt = build_user_prompt("What is X?", [make_chunk(1, "a.pdf", 1, 0.9)])
    assert "What is X?" in prompt
    assert "content 1" in prompt
    assert "citing your sources" in prompt


def test_answer_citations_are_deduplicated_in_order():
    answer = Answer(
        text="ok",
        sources=[
            make_chunk(1, "a.pdf", 2, 0.9),
            make_chunk(2, "b.txt", None, 0.8),
            make_chunk(3, "a.pdf", 2, 0.7),  # same page as the first hit
        ],
    )
    assert answer.citations == ["a.pdf p.2", "b.txt"]


def test_formatted_answer_lists_sources_with_scores():
    answer = Answer(text="Some answer.", sources=[make_chunk(1, "a.pdf", 2, 0.912)])
    rendered = answer.format()

    assert "Some answer." in rendered
    assert "Sources:" in rendered
    assert "[1] a.pdf p.2" in rendered
    assert "0.912" in rendered


def test_answer_without_sources_formats_as_plain_text():
    assert Answer(text="No idea.").format() == "No idea."


def test_empty_corpus_short_circuits_before_calling_the_llm():
    llm = StubLLM()
    retriever = Retriever(store=EmptyStore(), embedder=StubEmbedder())
    chatbot = RagChatbot(retriever=retriever, llm=llm)

    answer = chatbot.ask("anything?")

    assert answer.text == NO_CONTEXT_MESSAGE
    assert answer.sources == []
    # The LLM must not be billed for a question we cannot ground.
    assert llm.messages is None


def test_ask_grounds_the_prompt_and_returns_sources():
    llm = StubLLM()
    retriever = Retriever(store=OneDocStore(), embedder=StubEmbedder())
    chatbot = RagChatbot(retriever=retriever, llm=llm)

    answer = chatbot.ask("what is the meaning of life?")

    assert answer.text == "The answer is 42 [1]."
    assert answer.citations == ["guide.pdf p.7"]

    system, user = llm.messages
    assert "ONLY from the numbered context" in system.content
    assert "the meaning of life" in user.content

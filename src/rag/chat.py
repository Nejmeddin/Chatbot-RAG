"""Answer generation: retrieved context in, grounded answer out.

The context block is numbered, and the system prompt requires the model to mark
each claim with the number of the chunk it came from. Those markers are what
turn a plausible-sounding answer into one a reader can verify against the
source documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise assistant answering questions about a \
document corpus.

Rules:
- Answer ONLY from the numbered context below. Never use outside knowledge.
- Mark every factual claim with the number of the source it came from, like [1] \
or [2][3].
- If the context does not contain the answer, say so plainly instead of \
guessing.
- Answer in the same language as the question.
- Be concise: a few sentences unless the question needs more."""

NO_CONTEXT_MESSAGE = (
    "I could not find anything relevant in the indexed documents to answer that."
)


@dataclass(frozen=True)
class Answer:
    """A generated answer together with the chunks it was grounded in."""

    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        """De-duplicated source labels, in the order they were retrieved."""
        seen: dict[str, None] = {}
        for chunk in self.sources:
            seen.setdefault(chunk.citation, None)
        return list(seen)

    def format(self) -> str:
        """Render the answer with a source list, for terminal output."""
        if not self.sources:
            return self.text
        lines = [self.text, "", "Sources:"]
        lines += [
            f"  [{i}] {chunk.citation}  (similarity {chunk.score:.3f})"
            for i, chunk in enumerate(self.sources, start=1)
        ]
        return "\n".join(lines)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    return "\n\n".join(
        f"[{i}] (source: {chunk.citation})\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user turn: the numbered context, then the question."""
    return (
        f"Context:\n{build_context(chunks)}\n\n"
        f"Question: {query}\n\n"
        "Answer, citing your sources by number:"
    )


def build_llm(api_key: str, model: str, temperature: float = 0.2):
    """Construct the Groq chat model. Imported lazily to keep startup fast."""
    from langchain_groq import ChatGroq

    return ChatGroq(api_key=api_key, model=model, temperature=temperature)


class RagChatbot:
    """Ties retrieval and generation together."""

    def __init__(self, retriever: Retriever, llm, top_k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def ask(
        self, query: str, top_k: int | None = None, score_threshold: float = 0.0
    ) -> Answer:
        """Retrieve context for ``query`` and generate a grounded answer."""
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = self.retriever.retrieve(
            query, top_k=top_k or self.top_k, score_threshold=score_threshold
        )
        if not chunks:
            return Answer(text=NO_CONTEXT_MESSAGE, sources=[])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(query, chunks)),
        ]
        response = self.llm.invoke(messages)
        return Answer(text=response.content.strip(), sources=chunks)

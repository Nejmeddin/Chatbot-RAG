"""Factory helpers that wire the components together from configuration.

Both the CLI and the Streamlit app build their objects here, so there is one
place where the pipeline is assembled.
"""

from __future__ import annotations

import logging

from .chat import RagChatbot, build_llm
from .config import Settings
from .config import settings as default_settings
from .embeddings import EmbeddingManager
from .retriever import Retriever
from .store import DocumentStore

logger = logging.getLogger(__name__)


def build_store(settings: Settings | None = None) -> DocumentStore:
    settings = settings or default_settings
    return DocumentStore(dsn=settings.dsn, table_name=settings.table_name)


def build_embedder(settings: Settings | None = None) -> EmbeddingManager:
    settings = settings or default_settings
    return EmbeddingManager(model_name=settings.embedding_model)


def build_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or default_settings
    return Retriever(store=build_store(settings), embedder=build_embedder(settings))


def build_chatbot(settings: Settings | None = None) -> RagChatbot:
    """Assemble a ready-to-use chatbot. Requires GROQ_API_KEY to be set."""
    settings = settings or default_settings
    llm = build_llm(
        api_key=settings.require_groq_key(),
        model=settings.llm_model,
    )
    return RagChatbot(
        retriever=build_retriever(settings), llm=llm, top_k=settings.top_k
    )

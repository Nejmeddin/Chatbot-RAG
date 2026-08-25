"""Streamlit chat interface for the RAG chatbot.

Run with:  streamlit run app.py

The heavy objects (embedding model, database corpus, LLM client) are built once
and cached, so only the first question pays the model-loading cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.config import settings  # noqa: E402
from rag.pipeline import build_chatbot  # noqa: E402

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Loading embedding model and corpus...")
def get_chatbot():
    """Build the pipeline once per session."""
    return build_chatbot()


st.title("💬 RAG Chatbot")
st.caption(
    f"Answers grounded in your local corpus · {settings.embedding_model} "
    f"embeddings · {settings.llm_model}"
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, settings.top_k)
    threshold = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.05)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    chatbot = get_chatbot()
except Exception as exc:
    st.error(f"Could not start the chatbot: {exc}")
    st.info(
        "Check that PostgreSQL is running, that `.env` is filled in, and that "
        "you have run `python -m rag ingest`."
    )
    st.stop()

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] {source['citation']}** · {source['score']:.3f}")
                    st.caption(source["preview"])

if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the corpus..."):
            try:
                answer = chatbot.ask(question, top_k=top_k, score_threshold=threshold)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        st.markdown(answer.text)

        sources = [
            {
                "citation": chunk.citation,
                "score": chunk.score,
                "preview": chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
            }
            for chunk in answer.sources
        ]
        if sources:
            with st.expander("Sources"):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] {source['citation']}** · {source['score']:.3f}")
                    st.caption(source["preview"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": sources}
    )

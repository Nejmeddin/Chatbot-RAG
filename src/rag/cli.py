"""Command-line interface.

    python -m rag ingest [--reset]   build the index from the data/ folder
    python -m rag ask "question"     ask one question and print the answer
    python -m rag chat               interactive question/answer loop
    python -m rag stats              show what is currently indexed
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import settings


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    """Load, chunk, embed and store every document under the data directory."""
    from .ingest import build_corpus
    from .pipeline import build_embedder, build_store

    store = build_store()
    store.create_schema()

    if args.reset:
        store.clear()

    chunks = build_corpus(
        settings.data_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        print(f"No .pdf or .txt files found under {settings.data_dir}")
        return 1

    print(f"Prepared {len(chunks)} chunks from {settings.data_dir}")

    embedder = build_embedder()
    embeddings = embedder.encode([chunk.text for chunk in chunks])

    inserted = store.add_chunks(chunks, embeddings)
    skipped = len(chunks) - inserted
    print(f"Inserted {inserted} new chunks ({skipped} already indexed).")
    print(f"Total rows in '{settings.table_name}': {store.count()}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a single question."""
    from .pipeline import build_chatbot

    chatbot = build_chatbot()
    answer = chatbot.ask(args.question, top_k=args.top_k)
    print()
    print(answer.format())
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive loop. Ctrl-C or 'exit' to quit."""
    from .pipeline import build_chatbot

    chatbot = build_chatbot()
    print(f"RAG chatbot ready ({settings.llm_model}). Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            return 0

        try:
            answer = chatbot.ask(question, top_k=args.top_k)
        except Exception as exc:  # keep the session alive on a bad question
            print(f"\nError: {exc}\n")
            continue

        print(f"\nBot: {answer.format()}\n")


def cmd_stats(args: argparse.Namespace) -> int:
    """Report what is currently indexed."""
    from .pipeline import build_store

    store = build_store()
    corpus = store.fetch_corpus()
    print(f"Database : {settings.pg_db} @ {settings.pg_host}:{settings.pg_port}")
    print(f"Table    : {settings.table_name}")
    print(f"Chunks   : {len(corpus)}")
    if len(corpus):
        print(f"Dimension: {corpus.embeddings.shape[1]}")
        print(f"Sources  : {len(set(corpus.sources))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG chatbot over a local document corpus, backed by PostgreSQL.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show progress logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="index the documents in data/")
    ingest.add_argument(
        "--reset", action="store_true", help="empty the table before indexing"
    )
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="ask a single question")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument("--top-k", type=int, default=settings.top_k)
    ask.set_defaults(func=cmd_ask)

    chat = subparsers.add_parser("chat", help="interactive chat session")
    chat.add_argument("--top-k", type=int, default=settings.top_k)
    chat.set_defaults(func=cmd_chat)

    stats = subparsers.add_parser("stats", help="show index statistics")
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

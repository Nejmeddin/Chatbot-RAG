"""Runtime configuration, read from the environment.

Nothing in this module hard-codes a credential. Values come from a local
``.env`` file (see ``.env.example``) or from real environment variables, which
keeps secrets out of the repository and lets the same code run against a
different database without an edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env once, at import time. override=False means a real environment
# variable (CI, Docker, systemd) always wins over the local file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to a development default."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {os.getenv(name)!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to know, resolved from the environment."""

    # --- PostgreSQL -------------------------------------------------------
    pg_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5432))
    pg_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "rag_chatbot"))
    pg_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "postgres"))
    pg_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    table_name: str = field(default_factory=lambda: _env("POSTGRES_TABLE", "documents"))
    # Without this, libpq waits on an unreachable host for minutes. Failing in
    # seconds with a clear message is far more useful than an apparent hang.
    pg_connect_timeout: int = field(
        default_factory=lambda: _env_int("POSTGRES_CONNECT_TIMEOUT", 10)
    )

    # --- Models -----------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "llama-3.3-70b-versatile")
    )

    # --- Pipeline tuning --------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / _env("DATA_DIR", "data")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))

    @property
    def dsn(self) -> str:
        """A libpq connection string, with values escaped by psycopg."""
        return make_conninfo(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.pg_db,
            user=self.pg_user,
            password=self.pg_password or None,
            connect_timeout=self.pg_connect_timeout,
        )

    def require_groq_key(self) -> str:
        """Return the Groq API key, or explain how to set it."""
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "key from https://console.groq.com/keys"
            )
        return self.groq_api_key


settings = Settings()

"""Local sentence-transformer embeddings.

The model is loaded lazily so that importing this module (for tests, or for the
CLI's ``--help``) does not pull PyTorch into memory.

Vectors are L2-normalised on the way out. That is a deliberate choice: for unit
vectors the cosine similarity is exactly the dot product, so retrieval becomes a
single matrix multiply and the project needs no similarity library at all.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against division by zero for empty or degenerate vectors.
    norms[norms == 0] = 1.0
    return vectors / norms


class EmbeddingManager:
    """Turns text into normalised vectors with a local Hugging Face model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """The underlying SentenceTransformer, loaded on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded (dimension=%d)", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts into a ``(len(texts), dim)`` float32 matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        logger.info("Embedding %d text(s) ...", len(texts))
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single string into a 1-D vector."""
        return self.encode([text])[0]

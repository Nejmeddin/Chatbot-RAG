"""A Retrieval-Augmented Generation chatbot over a local document corpus.

Embeddings are computed locally with sentence-transformers and stored in
PostgreSQL as ``DOUBLE PRECISION[]`` columns, so the project runs end to end
on a single laptop with no paid API for the retrieval half of the pipeline.
"""

__version__ = "0.1.0"

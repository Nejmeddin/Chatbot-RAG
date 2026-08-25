"""Tests for vector normalisation.

Normalisation is what lets retrieval use a plain dot product as cosine
similarity, so it is worth pinning down.
"""

from __future__ import annotations

import numpy as np

from rag.embeddings import l2_normalize


def test_rows_become_unit_vectors():
    normalised = l2_normalize(np.array([[3.0, 4.0], [1.0, 1.0]]))
    assert np.allclose(np.linalg.norm(normalised, axis=1), 1.0)


def test_direction_is_preserved():
    normalised = l2_normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(normalised, [[0.6, 0.8]])


def test_zero_vector_does_not_divide_by_zero():
    normalised = l2_normalize(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(normalised))
    assert np.allclose(normalised, [[0.0, 0.0]])


def test_one_dimensional_input_is_promoted_to_a_row():
    normalised = l2_normalize(np.array([3.0, 4.0]))
    assert normalised.shape == (1, 2)


def test_output_is_float32():
    assert l2_normalize(np.array([[1.0, 2.0]], dtype=np.float64)).dtype == np.float32


def test_dot_product_of_normalised_vectors_equals_cosine_similarity():
    a = np.array([[2.0, 1.0, 0.0]])
    b = np.array([[1.0, 3.0, 2.0]])

    expected = float(
        np.dot(a[0], b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0]))
    )
    actual = float(l2_normalize(a)[0] @ l2_normalize(b)[0])

    assert actual == np.float32(expected)

"""Verifier unit tests (no ARC data required)."""

from __future__ import annotations

from packages.eval.verifier import (
    exact_match,
    pixel_accuracy,
    shape_accuracy,
    validate_grid,
)


def test_exact_match_same():
    g = [[0, 1], [2, 3]]
    assert exact_match(g, [[0, 1], [2, 3]]) is True


def test_exact_match_different():
    assert exact_match([[0, 1]], [[0, 2]]) is False


def test_pixel_accuracy_partial():
    assert pixel_accuracy([[0, 1, 2, 3]], [[0, 1, 9, 9]]) == 0.5


def test_pixel_accuracy_shape_mismatch_is_zero():
    assert pixel_accuracy([[0, 1]], [[0], [1]]) == 0.0


def test_shape_accuracy():
    assert shape_accuracy([[0, 0]], [[1, 1]]) is True
    assert shape_accuracy([[0]], [[0, 0]]) is False


def test_validate_grid():
    assert validate_grid([[0, 9]]) is True
    assert validate_grid([[0, 10]]) is False
    assert validate_grid([]) is False
    assert validate_grid([[0, 1], [0]]) is False

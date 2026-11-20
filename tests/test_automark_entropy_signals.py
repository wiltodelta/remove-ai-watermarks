"""Entropy inputs cannot silently degrade to missing or zero author weights."""

import math
from typing import Any

import numpy as np
import pytest

from scripts import benchmark_automark_reference as adapter


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_valid_entropy_shape_sentinel_and_uniform_bound(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    adapter.validate_entropies([0, 1, 9], np.array([0, 2, math.log(10)], dtype=dtype), 10)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "shape",
        "nan",
        "infinity",
        "negative",
        "zero_tail",
        "zero_fallback",
        "first_nonzero",
        "above_uniform",
        "integer",
        "boolean",
        "complex",
        "string",
        "short_tokens",
        "negative_token",
        "outside_token",
        "fractional_token",
        "boolean_token",
        "boolean_vocab",
        "fractional_vocab",
        "small_vocab",
    ],
)
def test_invalid_or_degenerate_entropy_signals_fail(fault: str) -> None:
    tokens: list[Any] = [0, 1, 9]
    values: Any = np.array([0.0, 2.0, 1.5])
    size: Any = 10
    if fault == "missing":
        values = None
    elif fault == "shape":
        values = values[:, None]
    elif fault == "nan":
        values[1] = np.nan
    elif fault == "infinity":
        values[1] = np.inf
    elif fault == "negative":
        values[1] = -1
    elif fault == "zero_tail":
        values[2] = 0
    elif fault == "zero_fallback":
        values[:] = 0
    elif fault == "first_nonzero":
        values[0] = 1
    elif fault == "above_uniform":
        values[1] = math.log(size) + 0.01
    elif fault == "integer":
        values = np.array([0, 2, 1])
    elif fault == "boolean":
        values = values.astype(bool)
    elif fault == "complex":
        values = values.astype(complex)
    elif fault == "string":
        values = values.astype(str)
    elif fault == "short_tokens":
        tokens, values = [0], np.array([0.0])
    elif fault == "negative_token":
        tokens[1] = -1
    elif fault == "outside_token":
        tokens[1] = size
    elif fault == "fractional_token":
        tokens[1] = 1.5
    elif fault == "boolean_token":
        tokens[1] = True
    elif fault == "boolean_vocab":
        size = True
    elif fault == "fractional_vocab":
        size = 10.5
    elif fault == "small_vocab":
        size = 1
    with pytest.raises(ValueError, match="entropy"):
        adapter.validate_entropies(tokens, values, size)

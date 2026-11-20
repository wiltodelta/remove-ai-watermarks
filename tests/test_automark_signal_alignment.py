"""Reject position-shifted or internally inconsistent teacher-forced signals."""

import numpy as np
import pytest

from scripts.benchmark_automark_reference import validate_auxiliary


def _signals():
    tokens = [0, 1, 9]
    lp = np.array([0.0, -2.0, -9.0])
    ids = np.tile(np.arange(8), (3, 1)).astype(float)
    rows = np.tile(-np.arange(1, 9, dtype=float), (3, 1))
    ids[0] = rows[0] = 0
    return tokens, lp, ids, rows, 10


def test_consistent_teacher_forced_rows_pass():
    validate_auxiliary(*_signals())


@pytest.mark.parametrize(
    "fault",
    [
        "first_target",
        "first_ids",
        "first_rows",
        "zero_target",
        "zero_row",
        "duplicate_id",
        "unsorted_row",
        "target_disagrees",
        "absent_target_too_likely",
        "fractional_token",
        "negative_token",
        "outside_token",
        "boolean_token",
        "small_vocabulary",
        "boolean_vocabulary",
        "fractional_vocabulary",
    ],
)
def test_inconsistent_auxiliary_signals_fail(fault):
    tokens, lp, ids, rows, vocab = _signals()
    if fault == "first_target":
        lp[0] = -1
    elif fault == "first_ids":
        ids[0, 0] = 1
    elif fault == "first_rows":
        rows[0, 0] = -1
    elif fault == "zero_target":
        tokens[2] = 0
        lp[2] = 0
        rows[2, 0] = 0
    elif fault == "zero_row":
        rows[2] = 0
    elif fault == "duplicate_id":
        ids[1, 2] = ids[1, 0]
    elif fault == "unsorted_row":
        rows[2, [1, 2]] = rows[2, [2, 1]]
    elif fault == "target_disagrees":
        lp[1] = -3
    elif fault == "absent_target_too_likely":
        lp[2] = -1
    elif fault == "fractional_token":
        tokens[1] = 1.5
        lp[1] = -9
    elif fault == "negative_token":
        tokens[1] = -1
    elif fault == "outside_token":
        tokens[1] = vocab
    elif fault == "boolean_token":
        tokens[1] = True
    elif fault == "small_vocabulary":
        vocab = 7
    elif fault == "boolean_vocabulary":
        vocab = True
    elif fault == "fractional_vocabulary":
        vocab = 10.5
    with pytest.raises(ValueError, match="signal"):
        validate_auxiliary(tokens, lp, ids, rows, vocab)


def test_clamped_tail_ties_and_float32_rounding_pass():
    tokens, lp, ids, rows, vocab = _signals()
    rows[2, 5:] = -30
    lp[2] = -30
    lp[1] += 1e-6
    validate_auxiliary(tokens, lp, ids, rows, vocab)

"""Paper-level tests for the SAMark response statistic."""

import math

import pytest

from remove_ai_watermarks._internal.samark import score_responses


def test_samark_response_statistic_matches_hand_computed_value() -> None:
    score = score_responses([[0.1], [0.1], [-0.1]])

    assert score.flags == (1,)
    assert score.count == 3
    assert score.z == pytest.approx(math.sqrt(3 / 8))


def test_samark_sentence_permutation_preserves_score() -> None:
    rows = [[0.11, -0.04], [0.08, -0.10], [-0.02, 0.06]]

    score = score_responses(rows)

    assert score.flags == (1, -1)
    assert score.count == 6
    assert score_responses(rows[::-1]) == score


def test_samark_exact_tie_uses_zero_flag() -> None:
    score = score_responses([[0.25, 0.01], [-0.25, 0.08], [0.0, -0.04]])

    assert score.flags == (0, 1)
    assert score.z is not None


def test_samark_identical_alignments_are_unscorable() -> None:
    score = score_responses([[0.03, -0.03]] * 3)

    assert score.flags == (1, -1)
    assert score.z is None


@pytest.mark.parametrize("bad", [[], [[]], [[0.0], [0.0, 0.0]], [[math.nan]], [[1.01]]])
def test_samark_rejects_invalid_response_matrices(bad: list[list[float]]) -> None:
    with pytest.raises(ValueError, match=r"Responses|Cosine"):
        score_responses(bad)


@pytest.mark.parametrize("sharpness", [0, -1, math.inf, math.nan])
def test_samark_rejects_invalid_sharpness(sharpness: float) -> None:
    with pytest.raises(ValueError, match="Sharpness"):
        score_responses([[0.1]], sharpness)

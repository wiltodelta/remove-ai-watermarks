"""Conformance checks for Google DeepMind's open Threefry-based reference."""

import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks._internal.synthid_threefry import threefry2x32


@pytest.mark.parametrize(
    ("key", "counter", "expected"),
    [
        ((0, 0), (0, 0), (0x6B200159, 0x99BA4EFE)),
        ((0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x1CB996FC, 0xBB002BE7)),
        ((0x13198A2E, 0x03707344), (0x243F6A88, 0x85A308D3), (0xC4923A9C, 0x483DF7A0)),
    ],
)
def test_threefry_known_answer_vectors(key, counter, expected):
    assert threefry2x32(key, *counter) == expected


@pytest.mark.parametrize(
    ("ids", "key", "ngram_len", "depth", "history", "eos", "ones", "positions", "p_value"),
    [
        ([7, 44, 3, 98, 22, 5, 5, 17, 43, 2, 19, 17], 654, 5, 30, 1024, (), 121, 8, 0.47427519708532034),
        ([2, 3, 9, 2, 3, 9, 2, 3, 9, 2, 3, 1], 655, 3, 65, 2, (1,), 264, 9, 0.9917942399307822),
        (
            [77, 3, 1, 644, 49, 23, 5, 9, 2, 62, 19, 4],
            (1 << 47) + 32768,
            2,
            129,
            5,
            (2, 0),
            457,
            7,
            0.36966082083054025,
        ),
        ([6, 7, 8, 9, 10, 300, 1, 2, 3], 0, 3, 64, 2, (6,), 0, 0, 1.0),
    ],
)
def test_frequentist_scoring_matches_open_reference(ids, key, ngram_len, depth, history, eos, ones, positions, p_value):
    result = raiw.score_synthid_threefry_text(
        ids, key, ngram_len=ngram_len, depth=depth, context_history_size=history, eos_token_ids=eos, alpha=0.5
    )
    assert isinstance(result, raiw.SynthIDThreefryScore)
    assert result.num_tokens == positions
    assert result.num_ones == ones
    assert result.num_trials == positions * depth
    assert result.p_value == pytest.approx(p_value, abs=1e-13)
    assert result.detected is (p_value <= 0.5)


def test_repeated_contexts_and_short_text_are_handled():
    repeated = raiw.score_synthid_threefry_text([1, 2, 3, 1, 2, 9, 4], 654, ngram_len=3)
    assert repeated.num_tokens == 4
    empty = raiw.score_synthid_threefry_text([1, 2], 654)
    assert empty.p_value == 1
    assert empty.num_tokens == 0
    assert not empty.detected

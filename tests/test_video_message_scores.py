"""Known-message scope, streaming completeness, and numerical guards."""

from dataclasses import asdict

import numpy as np
import pytest

from remove_ai_watermarks._internal.video_message_scores import MessageAccumulator, reference_bits

MESSAGE = reference_bits("a5" * 32)
SIGN = np.asarray(MESSAGE, dtype=np.float64) * 2 - 1


def inspect(values, expected=MESSAGE, *, batch=8):
    accumulator = MessageAccumulator(expected)
    for start in range(0, len(values), batch):
        accumulator.update(values[start : start + batch])
    return accumulator.finish()


@pytest.mark.parametrize("reference", ["", "a" * 63, "g" * 64, "a " * 32])
def test_reference_hex_is_exact_and_binary(reference):
    with pytest.raises(ValueError, match="64 hexadecimal"):
        reference_bits(reference)


def test_reference_case_and_bit_order():
    assert reference_bits("A5" * 32) == MESSAGE
    assert MESSAGE[:8] == (1, 0, 1, 0, 0, 1, 0, 1)


@pytest.mark.parametrize("fault", ["nan", "empty", "width", "rank", "overflow"])
def test_invalid_update_cannot_finalize_an_earlier_valid_prefix(fault):
    accumulator = MessageAccumulator(MESSAGE)
    accumulator.update(np.tile(-SIGN, (12, 1)))
    values = np.tile(SIGN, (13, 1))
    if fault == "nan":
        values[0, 0] = np.nan
    elif fault == "empty":
        values = values[:0]
    elif fault == "width":
        values = values[:, :-1]
    elif fault == "rank":
        values = values[:, None]
    else:
        values *= 1e300
    with pytest.raises(RuntimeError):
        accumulator.update(values)
    with pytest.raises(RuntimeError, match="unfinalized"):
        accumulator.finish()


def test_foreign_reference_is_independent():
    foreign = reference_bits("69" * 32)
    values = np.tile(np.asarray(foreign) * 2 - 1, (13, 1))
    assert inspect(values, foreign).target_matched
    assert not inspect(values).target_matched


def test_weighted_whole_readout_is_preserved():
    values = np.tile(-SIGN, (46, 1))
    for index in range(6):
        values[40 + index] = SIGN * 5
        values[40 + index, index * 40 : (index + 1) * 40] *= -1
    report = inspect(values)
    assert report.whole_clip == {"avg": 0, "squared_avg": 1, "l1norm_avg": 1, "l2norm_avg": 1}
    assert report.per_frame_max < 0.9
    assert report.target_matched


def test_single_surviving_frame_prevents_a_negative():
    values = np.tile(-SIGN, (24, 1))
    values[1] = SIGN
    report = inspect(values)
    assert max(report.whole_clip.values()) == 0
    assert all(max(window.scores.values()) == 0 for window in report.windows)
    assert report.per_frame_max == 1
    assert report.target_matched


def test_context_completed_tail_can_be_the_only_surviving_scope():
    values = np.tile(SIGN, (13, 1))
    values[0] = -SIGN * 100
    for index in range(1, 13):
        values[index, (np.arange(40) + (index - 1) * 19) % 256] *= -1
    report = inspect(values)
    assert max(report.whole_clip.values()) == 0
    assert report.per_frame_max < 0.9
    assert max(report.windows[0].scores.values()) == 0
    assert report.windows[-1].start_frame == 1
    assert report.windows[-1].end_frame == 13
    assert min(report.windows[-1].scores.values()) == 1
    assert report.target_matched


def test_realizable_threshold_neighbors():
    below = np.tile(SIGN, (13, 1))
    below[:, :26] *= -1
    above = np.tile(SIGN, (13, 1))
    above[:, :25] *= -1
    assert inspect(below).max_bit_accuracy == 230 / 256
    assert not inspect(below).target_matched
    assert inspect(above).max_bit_accuracy == 231 / 256
    assert inspect(above).target_matched


@pytest.mark.parametrize("frames", [1, 12, 13, 25, 64])
def test_batch_boundaries_do_not_change_scope_or_scores(frames):
    values = np.random.default_rng(18421).normal(size=(frames, 256))
    reference = asdict(inspect(values, batch=frames))
    for batch in [1, 5, 8, 32]:
        assert asdict(inspect(values, batch=batch)) == reference


def test_empty_or_already_finished_stream_cannot_be_published():
    accumulator = MessageAccumulator(MESSAGE)
    with pytest.raises(RuntimeError, match="unfinalized"):
        accumulator.finish()
    accumulator.update(SIGN[None])
    assert accumulator.finish().target_matched
    with pytest.raises(RuntimeError, match="already finalized"):
        accumulator.update(SIGN[None])
    with pytest.raises(RuntimeError, match="unfinalized"):
        accumulator.finish()


def test_thirty_six_frames_produce_twenty_five_sliding_windows_and_catch_boundary_crossing():
    values = np.tile(-SIGN * 2.0, (36, 1))
    for offset in range(12):
        row = SIGN.copy()
        flipped = (np.arange(40) + offset * 21) % 256
        row[flipped] *= -1.0
        values[6 + offset] = row

    report = inspect(values, batch=7)
    assert [(w.start_frame, w.end_frame) for w in report.windows] == [(i, i + 12) for i in range(25)]
    assert report.per_frame_max == 216 / 256 < 0.9
    assert max(report.whole_clip.values()) == 0.0
    disjoint = [w for w in report.windows if w.start_frame in {0, 12, 24}]
    assert len(disjoint) == 3
    assert all(max(w.scores.values()) == 0.0 for w in disjoint)
    crossing = next(w for w in report.windows if (w.start_frame, w.end_frame) == (6, 18))
    assert min(crossing.scores.values()) == 1.0
    assert report.target_matched is True

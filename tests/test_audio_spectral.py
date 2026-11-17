"""Physical filter behavior and stream boundary preservation."""

from __future__ import annotations

import numpy as np
import pytest


def _tones(rate=48000, seconds=3.1):
    t = np.arange(int(rate * seconds)) / rate
    return np.column_stack(
        (
            0.2 * np.sin(2 * np.pi * 500 * t) + 0.1 * np.sin(2 * np.pi * 1250 * t),
            0.2 * np.sin(2 * np.pi * 3000 * t) + 0.1 * np.sin(2 * np.pi * 1250 * t),
        )
    ).astype(np.float32)


def test_research_band_is_suppressed_without_moving_other_tones():
    from remove_ai_watermarks._internal.audio_spectral import filter_audio_chunks

    rate = 48000
    source = _tones(rate)
    output = np.concatenate(list(filter_audio_chunks([source], rate)))
    assert output.shape == source.shape
    # Use the steady-state part, away from reflected file endpoints.
    section = output[rate:-rate]
    t = np.arange(len(section)) / rate

    def amplitude(channel, frequency):
        return abs(2 * np.mean(section[:, channel] * np.exp(-2j * np.pi * frequency * t)))

    for channel, kept in [(0, 500), (1, 3000)]:
        assert amplitude(channel, 1250) < 0.0001
        assert amplitude(channel, kept) == pytest.approx(0.2, abs=0.002)


def test_chunked_filter_matches_whole_clip_and_keeps_short_tail():
    from remove_ai_watermarks._internal.audio_spectral import filter_audio_chunks

    source = _tones(seconds=3.0003)
    whole = np.concatenate(list(filter_audio_chunks([source], 48000)))
    chunks = [source[start : start + 24000] for start in range(0, len(source), 24000)]
    delivered = list(filter_audio_chunks(chunks, 48000))
    assert [len(c) for c in delivered] == [len(c) for c in chunks]
    np.testing.assert_allclose(np.concatenate(delivered), whole, rtol=0, atol=1e-6)


def test_single_sample_is_retained_and_finite():
    from remove_ai_watermarks._internal.audio_spectral import filter_audio_chunks

    source = np.array([[0.1, -0.1]], dtype=np.float32)
    output = np.concatenate(list(filter_audio_chunks([source], 48000)))
    assert output.shape == source.shape
    assert np.isfinite(output).all()


def test_processor_failure_closes_input_generator():
    from remove_ai_watermarks._internal.audio_spectral import filter_audio_chunks

    closed = []

    def source():
        try:
            yield np.zeros((10000, 2), dtype=np.float32)
            raise RuntimeError("upstream read failed")
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match="upstream read failed"):
        list(filter_audio_chunks(source(), 48000))
    assert closed == [True]


def test_caller_stop_closes_the_pending_input_iterator():
    from remove_ai_watermarks._internal.audio_spectral import filter_audio_chunks

    closed = []

    def source():
        try:
            for _ in range(3):
                yield np.zeros((10000, 2), dtype=np.float32)
        finally:
            closed.append(True)

    input_iterator = source()
    filtered = filter_audio_chunks(input_iterator, 16000)
    next(filtered)
    filtered.close()
    assert closed == [True]

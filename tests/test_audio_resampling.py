"""Clock, FIR, source-tail and boundary tests, independent of model downloads."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import correlate, firwin

from remove_ai_watermarks._internal import audio_resampling as src


def _roundtrip(value, rate=48000):
    clock = src.RationalClock(len(value), rate)
    return src.inverse(src.unpad_model(src.pad_model(src.forward(value, clock), clock), clock), clock)


def _direct_fir(value, up, down):
    maximum = max(up, down)
    half = 10 * maximum
    kernel = firwin(2 * half + 1, 1 / maximum, window=("kaiser", 5.0)).astype(np.float32) * up
    result = np.zeros((len(value) * up + down - 1) // down, dtype=np.float64)
    indices = np.arange(len(value))
    for index in range(len(result)):
        taps = half + index * down - indices * up
        valid = (taps >= 0) & (taps < len(kernel))
        result[index] = np.dot(value[valid].astype(np.float64), kernel[taps[valid]].astype(np.float64))
    return result


def test_all_residues_short_and_long_geometry():
    for frames in [*range(1, 321), *range(480000, 480160)]:
        clock = src.RationalClock(frames, 48000)
        assert clock.model_frames == (frames * 147 + 159) // 160
        assert clock.inverse_frames == (clock.model_frames * 160 + 146) // 147
        assert Fraction(clock.model_frames - 1, 44100) < clock.duration <= Fraction(clock.model_frames, 44100)
        assert all(Fraction(k, 48000) >= clock.duration for k in range(frames, clock.inverse_frames))
        if frames <= 320:
            value = np.arange(frames, dtype=np.float32) / frames
            assert len(src.forward(value, clock)) == clock.model_frames
            assert _roundtrip(value).shape == value.shape
    clock = src.RationalClock(480240, 48000)
    assert (clock.model_frames, clock.inverse_frames, clock.duration) == (441221, 480241, Fraction(2001, 200))
    assert clock.record()["excluded_inverse_indices"] == [480240, 480241]


@pytest.mark.parametrize("frames", [1, 2, 13, 159, 160, 161, 511, 557, 558, 559, 1024, 1200])
def test_direct_fir_reference_and_edge_impulses(frames):
    for position in sorted({0, frames - 1, min(frames - 1, 557)}):
        value = np.zeros(frames, dtype=np.float32)
        value[position] = 1
        clock = src.RationalClock(frames, 48000)
        model = src.forward(value, clock)
        np.testing.assert_allclose(model, _direct_fir(value, 147, 160), rtol=2e-6, atol=2e-6)
        result = src.inverse(model, clock)
        np.testing.assert_allclose(result, _direct_fir(model, 160, 147)[:frames], rtol=2e-6, atol=2e-6)
        assert abs(int(np.argmax(np.abs(result))) - position) <= 1


def test_last_source_sample_reaches_output_tail():
    value = np.zeros(160, dtype=np.float32)
    value[-1] = 1
    clock = src.RationalClock(len(value), 48000)
    assert np.max(np.abs(src.forward(value, clock)[-16:])) > 0.05
    assert abs(_roundtrip(value)[-1]) > 0.1


@pytest.mark.parametrize("frames", [1, 2, 557, 558, 559, 1114, 1115, 1116])
def test_reflection_context_and_identity_unpad(frames):
    clock = src.RationalClock(frames, 48000)
    value = np.random.default_rng(12).normal(size=(frames, 2)).astype(np.float32)
    model = src.forward(value, clock)
    padded = src.pad_model(model, clock)
    assert len(padded) >= 1024
    assert len(padded) % 512 == 0
    expected = np.pad(model, ((0, clock.padded_frames - len(model)), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(padded, expected)
    np.testing.assert_array_equal(src.unpad_model(padded, clock), model)
    if clock.model_frames == 1:
        np.testing.assert_array_equal(padded, np.repeat(model, len(padded), axis=0))
    with pytest.raises(ValueError, match="frames"):
        src.unpad_model(padded[:-1], clock)


def test_stereo_separation_and_native_channel_order():
    for channel, position in [(0, 100), (1, 1500)]:
        value = np.zeros((1600, 2), dtype=np.float32)
        value[position, channel] = 1
        result = _roundtrip(value)
        assert np.all(result[:, 1 - channel] == 0)
        assert int(np.argmax(result[:, channel])) == position


def test_inband_lag_and_expected_high_frequency_filtering():
    time = np.arange(48000) / 48000
    value = sum(
        amplitude * np.sin(2 * np.pi * frequency * time)
        for frequency, amplitude in [(997, 0.3), (4003, 0.2), (11987, 0.1)]
    ).astype(np.float32)
    result = _roundtrip(value)
    interior = slice(256, -256)
    correlation = correlate(result[interior], value[interior], mode="full", method="fft")
    assert int(np.argmax(correlation)) - (len(value[interior]) - 1) == 0
    assert np.sqrt(np.mean((result[interior] - value[interior]) ** 2)) / 0.3 < 0.02
    high = np.sin(2 * np.pi * 23500 * time).astype(np.float32)
    filtered = _roundtrip(high)
    assert np.sqrt(np.mean(filtered[interior] ** 2) / np.mean(high[interior] ** 2)) < 0.25


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000, 96000])
def test_other_native_rates_exact_count(rate):
    value = np.random.default_rng(6).normal(size=(rate // 20 + 1, 2)).astype(np.float32)
    result = _roundtrip(value, rate)
    assert result.shape == value.shape
    assert result.dtype == np.float32
    if rate == 44100:
        np.testing.assert_array_equal(result, value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("native_frames", 0), ("native_rate", 0), ("native_rate", True), ("native_rate", 48000.0), ("hop", -1)],
)
def test_invalid_clocks(field, value):
    arguments = {"native_frames": 4, "native_rate": 48000, field: value}
    with pytest.raises(ValueError, match="positive integer"):
        src.RationalClock(**arguments)


def test_missing_optional_dependencies(monkeypatch):
    monkeypatch.setattr(src, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires numpy and scipy"):
        src.forward(np.zeros(4, dtype=np.float32), src.RationalClock(4, 48000))

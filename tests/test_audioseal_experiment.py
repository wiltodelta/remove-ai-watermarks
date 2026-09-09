"""Tests for the pure helpers of the AudioSeal experiment script.

The oracle-dependent stages (embedding, detection, ffmpeg arms) stay
development-only and are exercised by the recorded local runs; these tests
cover the deterministic helpers that must not drift.
"""

from __future__ import annotations

import shutil
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audioseal_experiment import (  # noqa: E402
    CARRIERS,
    SAMPLE_RATE,
    SPEECH_VOICES,
    assert_distinct_voices,
    message_bits,
    moving_background,
    partial_noise_attack,
    snr_db,
    sora_like_mark,
    stamped_frame,
    synthesize_speech,
    tone_stack_carrier,
    wav_pcm16_bytes,
)


class TestCarriers:
    def test_carriers_are_deterministic_per_seed(self) -> None:
        n = 4000
        first = tone_stack_carrier(n, np.random.default_rng(11))
        second = tone_stack_carrier(n, np.random.default_rng(11))
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_produce_different_carriers(self) -> None:
        n = 4000
        first = tone_stack_carrier(n, np.random.default_rng(11))
        second = tone_stack_carrier(n, np.random.default_rng(12))
        assert not np.array_equal(first, second)

    def test_registered_carriers_return_expected_lengths(self) -> None:
        n = 1000
        for name, factory in CARRIERS.items():
            samples = np.asarray(factory(n, np.random.default_rng(3)), dtype=np.float32)
            assert samples.shape == (n,), name
            assert np.isfinite(samples).all(), name


class TestMessage:
    def test_message_is_stable_and_binary(self) -> None:
        bits = message_bits()
        assert bits == message_bits()
        assert len(bits) == 16
        assert set(bits) <= {0, 1}


class TestWavWriter:
    def test_wav_bytes_round_trip_through_wave_module(self, tmp_path: Path) -> None:
        samples = np.linspace(-0.9, 0.9, 256, dtype=np.float32)
        path = tmp_path / "out.wav"
        path.write_bytes(wav_pcm16_bytes(samples, SAMPLE_RATE))
        with wave.open(str(path)) as reader:
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            assert reader.getframerate() == SAMPLE_RATE
            decoded = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        np.testing.assert_allclose(decoded.astype(np.float32) / 32767.0, samples, atol=1 / 32767.0)

    def test_wav_header_declares_exact_payload_length(self) -> None:
        samples = np.zeros(10, dtype=np.float32)
        data = wav_pcm16_bytes(samples, SAMPLE_RATE)
        assert data[:4] == b"RIFF"
        assert struct.unpack("<I", data[4:8])[0] == 36 + 20
        assert data[36:40] == b"data"
        assert struct.unpack("<I", data[40:44])[0] == 20


class TestSnr:
    def test_zero_residual_is_infinite_headroom(self) -> None:
        signal = np.ones(100, dtype=np.float32)
        assert snr_db(signal, np.zeros(100, dtype=np.float32)) > 100.0

    def test_known_ratio(self) -> None:
        signal = np.ones(1000, dtype=np.float32)
        residual = np.full(1000, 0.1, dtype=np.float32)
        assert snr_db(signal, residual) == pytest.approx(20.0, abs=0.01)


class TestVideoFrames:
    def test_mark_stamps_only_its_region(self) -> None:
        clean = moving_background(5)
        mark, mark_x, mark_y = sora_like_mark()
        stamped = stamped_frame(5, mark, mark_x, mark_y)
        height, width = mark.shape
        outside = (
            slice(0, mark_y),
            slice(0, mark_x),
        )
        np.testing.assert_array_equal(stamped[outside], clean[outside])
        assert not np.array_equal(
            stamped[mark_y : mark_y + height, mark_x : mark_x + width],
            clean[mark_y : mark_y + height, mark_x : mark_x + width],
        )

    def test_mark_geometry_matches_the_test_clip_layout(self) -> None:
        mark, mark_x, mark_y = sora_like_mark()
        assert mark.shape == (44, 124)
        assert mark_x + 124 <= 840
        assert mark_y + 44 <= 480


class TestSpeechCarriers:
    def test_assert_distinct_voices_rejects_aliased_bytes(self) -> None:
        with pytest.raises(SystemExit, match="byte-identical"):
            assert_distinct_voices({"Samantha": b"abc", "Alex": b"abc"})

    def test_assert_distinct_voices_accepts_distinct_bytes(self) -> None:
        assert_distinct_voices({"Samantha": b"abc", "Daniel": b"abd"})

    def test_partial_noise_attack_touches_only_its_window(self) -> None:
        samples = np.ones(4 * SAMPLE_RATE, dtype=np.float32)
        attacked = partial_noise_attack(samples, 1.0, 2.0)

        np.testing.assert_array_equal(attacked[:SAMPLE_RATE], samples[:SAMPLE_RATE])
        np.testing.assert_array_equal(attacked[2 * SAMPLE_RATE :], samples[2 * SAMPLE_RATE :])
        assert not np.array_equal(attacked[SAMPLE_RATE : 2 * SAMPLE_RATE], samples[SAMPLE_RATE : 2 * SAMPLE_RATE])

    @pytest.mark.skipif(shutil.which("say") is None, reason="requires the macOS say tool")
    def test_system_speech_is_deterministic_per_voice(self, tmp_path: Path) -> None:
        first = synthesize_speech("Samantha", tmp_path)
        (tmp_path / "again").mkdir()
        synthesize_speech("Samantha", tmp_path / "again")

        assert first.read_bytes() == (tmp_path / "again" / "speech-Samantha.aiff").read_bytes()

    @pytest.mark.skipif(shutil.which("say") is None, reason="requires the macOS say tool")
    def test_configured_voices_render_distinct_bytes(self, tmp_path: Path) -> None:
        rendered = {voice: synthesize_speech(voice, tmp_path).read_bytes() for voice in SPEECH_VOICES}

        assert_distinct_voices(rendered)

"""Audio I/O invariants exercised through real lossless files and ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")


def test_surround_layout_survives_raw_sample_roundtrip(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio, read_audio, write_audio

    source, output = tmp_path / "side.wav", tmp_path / "processed.wav"
    samples = np.tile(np.arange(1, 7, dtype=np.float32) / 10, (1600, 1))
    subprocess.run(  # noqa: S603 - Synthetic samples and task-owned paths.
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "6",
            "-channel_layout",
            "5.1(side)",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_f32le",
            str(source),
        ],
        input=samples.astype("<f4").tobytes(),
        check=True,
    )
    info = probe_audio(source)
    write_audio(info, output, read_audio(source, info), expected_frames=len(samples))
    payload = json.loads(
        subprocess.check_output([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(output)])  # noqa: S603 - Synthetic output only.
    )
    assert payload["streams"][0]["channel_layout"] == "5.1(side)"
    np.testing.assert_array_equal(np.concatenate(list(read_audio(output, probe_audio(output)))), samples)


def test_bitexact_wav_is_byte_stable_across_chunking_and_decode_encode(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio, read_audio, write_audio

    source, samples = _stereo_source(tmp_path)
    info = probe_audio(source)
    first, second, third = (tmp_path / name for name in ("first.wav", "second.wav", "third.wav"))
    write_audio(info, first, [samples], bitexact=True)
    write_audio(info, second, [samples[:1], samples[1:2000], samples[2000:]], bitexact=True)
    write_audio(probe_audio(first), third, read_audio(first, probe_audio(first)), bitexact=True)
    assert first.read_bytes() == second.read_bytes() == third.read_bytes()
    np.testing.assert_array_equal(np.concatenate(list(read_audio(first, probe_audio(first)))), samples)
    assert b"Lavf" not in first.read_bytes()
    assert b"JUNK" not in first.read_bytes()


def test_bitexact_wav_rejects_oversized_view_before_scanning_or_writing_samples(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal.audio_io import probe_audio, write_audio

    source, _ = _stereo_source(tmp_path)
    info = probe_audio(source)
    huge = np.broadcast_to(np.zeros((), dtype="<f4"), (1 << 30, 2))
    original = np.isfinite

    def finite(samples):
        if samples.shape == huge.shape:
            pytest.fail("RIFF size guard must precede full-array scanning")
        return original(samples)

    monkeypatch.setattr(np, "isfinite", finite)
    output = tmp_path / "bounded.wav"
    with pytest.raises(ValueError, match="short RIFF size bound"):
        write_audio(info, output, [huge], bitexact=True)
    assert not output.exists()


def _stereo_source(tmp_path: Path) -> tuple[Path, np.ndarray]:
    source = tmp_path / "stereo.wav"
    values = np.column_stack((np.arange(2501, dtype=np.int16) - 1250, 2400 - np.arange(2501, dtype=np.int16)))
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(values.astype("<i2").tobytes())
    return source, values.astype(np.float32) / 32768


def test_decoding_keeps_channels_rate_and_nonmultiple_tail(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio, read_audio

    source, expected = _stereo_source(tmp_path)
    info = probe_audio(source)
    assert (info.channels, info.sample_rate) == (2, 48000)
    chunks = list(read_audio(source, info, chunk_frames=127))
    assert len(chunks[-1]) == 88
    np.testing.assert_array_equal(np.concatenate(chunks), expected)


@pytest.mark.parametrize("suffix", [".wav", ".flac"])
def test_lossless_encoding_preserves_audio_geometry_and_samples(tmp_path, suffix):
    from remove_ai_watermarks._internal.audio_io import probe_audio, read_audio, write_audio

    source, expected = _stereo_source(tmp_path)
    info = probe_audio(source)
    output = tmp_path / ("output" + suffix)
    assert write_audio(info, output, read_audio(source, info, chunk_frames=127), expected_frames=2501) == 2501
    delivered = probe_audio(output)
    assert (delivered.channels, delivered.sample_rate) == (2, 48000)
    np.testing.assert_array_equal(np.concatenate(list(read_audio(output, delivered))), expected)


@pytest.mark.parametrize("fault", ["generator", "nan", "wrong_channels", "wrong_length"])
def test_late_failure_keeps_existing_output_and_cleans_staging(tmp_path, fault):
    from remove_ai_watermarks._internal.audio_io import probe_audio, write_audio

    source, expected = _stereo_source(tmp_path)
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")
    before = set(tmp_path.iterdir())

    def chunks():
        yield expected[:100]
        if fault == "generator":
            raise RuntimeError("source failed after its first chunk")
        if fault == "nan":
            yield np.full((10, 2), np.nan, dtype=np.float32)
        elif fault == "wrong_channels":
            yield expected[100:, :1]
        elif fault == "wrong_length":
            yield expected[100:-1]

    with pytest.raises((ValueError, RuntimeError)):
        write_audio(probe_audio(source), output, chunks(), expected_frames=2501)
    assert output.read_bytes() == b"existing output"
    assert set(tmp_path.iterdir()) == before


def test_no_audio_stream_is_an_input_error(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio

    source = tmp_path / "not-audio.bin"
    source.write_bytes(b"not a media container")
    with pytest.raises(ValueError, match="Cannot read audio"):
        probe_audio(source)


def test_encoder_failure_reports_stderr_without_replacing_output(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal import audio_io

    source, _ = _stereo_source(tmp_path)
    encoder = tmp_path / "broken-ffmpeg"
    encoder.write_text("#!/bin/sh\necho 'synthetic encoder failure' >&2\nexit 17\n")
    encoder.chmod(0o700)
    originalrequire_tool = audio_io.require_tool
    monkeypatch.setattr(
        audio_io, "require_tool", lambda name: str(encoder) if name == "ffmpeg" else originalrequire_tool(name)
    )
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")
    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        audio_io.write_audio(audio_io.probe_audio(source), output, [np.zeros((100000, 2), dtype=np.float32)])
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".output-*"))


def test_rejected_chunk_closes_the_supplied_generator(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio, write_audio

    source, _ = _stereo_source(tmp_path)
    closed = []

    def frames():
        try:
            yield np.full((2, 2), np.nan, dtype=np.float32)
            yield np.zeros((10, 2), dtype=np.float32)
        finally:
            closed.append(True)

    iterator = frames()
    with pytest.raises(ValueError, match="finite"):
        write_audio(probe_audio(source), tmp_path / "out.wav", iterator)
    assert closed == [True]


def test_failure_after_encoder_consumes_audio_preserves_existing_file(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio, write_audio

    source, _ = _stereo_source(tmp_path)
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")

    def chunks():
        # Exceed the pipe and encoder buffers before the upstream failure.
        yield np.zeros((256000, 2), dtype=np.float32)
        raise RuntimeError("late upstream failure")

    with pytest.raises(RuntimeError, match="late upstream failure"):
        write_audio(probe_audio(source), output, chunks())
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".output-*"))


def test_audio_io_falls_back_to_bundled_video_tools_without_system_ffmpeg(monkeypatch):
    from remove_ai_watermarks import video_tools
    from remove_ai_watermarks._internal import audio_io

    monkeypatch.setattr(audio_io.shutil, "which", lambda _name: None)
    monkeypatch.setattr(video_tools, "bundled_video_tools_available", lambda: True)
    monkeypatch.setattr(video_tools, "find_video_tool", lambda name: f"/bundled/{name}")
    assert audio_io.is_available() is True
    assert audio_io.require_tool("ffmpeg") == "/bundled/ffmpeg"
    assert audio_io.require_tool("ffprobe") == "/bundled/ffprobe"


@pytest.mark.parametrize("suffix", [".mp3", ".m4a", ".ogg", ".opus"])
def test_lossy_audio_containers_roundtrip_and_measure_fidelity(tmp_path, suffix):
    import soundfile as sf

    from remove_ai_watermarks._internal.audio_io import probe_audio, write_audio
    from remove_ai_watermarks._internal.audio_quality import audio_quality_passed, measure_fidelity

    sr = 24000
    t = np.arange(sr * 2 + 137, dtype=np.float32) / sr
    left = 0.35 * np.sin(2.0 * np.pi * 330.0 * t)
    right = 0.30 * np.sin(2.0 * np.pi * 440.0 * t + 0.4)
    stereo = np.column_stack((left, right)).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, stereo, sr, subtype="FLOAT")

    info = probe_audio(source)
    output = tmp_path / f"delivered{suffix}"
    written = write_audio(info, output, [stereo[:20000], stereo[20000:]])
    assert written == len(stereo)
    assert output.is_file()

    out_info = probe_audio(output)
    assert out_info.channels == 2
    assert out_info.sample_rate == (48000 if suffix in {".ogg", ".opus"} else sr)

    fidelity = measure_fidelity(source, info, output)
    assert fidelity.frames == len(stereo)
    assert fidelity.snr_db is not None
    assert fidelity.snr_db > 14.0
    assert audio_quality_passed(fidelity, lossy=True) is True


def test_mp3_rejects_unsupported_sample_rate(tmp_path):
    import soundfile as sf

    from remove_ai_watermarks._internal.audio_io import probe_audio, write_audio

    samples = np.zeros((19000, 1), dtype=np.float32)
    source = tmp_path / "odd.wav"
    sf.write(source, samples, 19000, subtype="FLOAT")
    output = tmp_path / "odd.mp3"
    with pytest.raises(ValueError, match="MP3 output does not support 19000 Hz"):
        write_audio(probe_audio(source), output, [samples])
    assert not output.exists()

"""Tests for v0.72.0 unified container/format support, DAC default HF preservation, and pruned experimental backends."""

from __future__ import annotations

import shutil
import wave
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _write_wav(path: Path, *, sample_rate: int = 44100, seconds: float = 0.5, channels: int = 1) -> Path:
    import math
    import struct

    n_frames = int(sample_rate * seconds)
    samples = []
    for i in range(n_frames):
        val = int(12000 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        for _ in range(channels):
            samples.append(val)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


@pytest.mark.parametrize("method", ["bigvgan", "mimi", "focalcodec"])
def test_removed_experimental_audio_methods_are_rejected(tmp_path: Path, method: str) -> None:
    from remove_ai_watermarks import audio

    source = _write_wav(tmp_path / "src.wav", sample_rate=16000)
    with pytest.raises(ValueError, match="Audio method must be"):
        audio.remove_audio_invisible(source, tmp_path / "out.wav", method=method, force=True, verify=False)  # type: ignore[arg-type]


def test_removed_silentcipher16_scheme_is_rejected(tmp_path: Path) -> None:
    from remove_ai_watermarks import audio

    source = _write_wav(tmp_path / "src.wav", sample_rate=16000)
    with pytest.raises(ValueError, match="Audio scheme must be"):
        audio.identify_audio(source, scheme="silentcipher16")  # type: ignore[arg-type]


@pytest.mark.parametrize("suffix", [".aac", ".oga", ".mka"])
def test_unified_audio_output_formats_encode_and_probe(tmp_path: Path, suffix: str) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe required")

    from remove_ai_watermarks import audio

    source = _write_wav(tmp_path / "src.wav", sample_rate=44100, seconds=0.4, channels=2)
    output = tmp_path / f"clean{suffix}"
    result = audio.remove_audio_invisible(source, output, method="spectral", force=True, verify=False)
    assert result.output == output
    assert output.is_file()
    assert result.after is not None
    assert result.after.stream.channels == 2
    assert result.frames_processed > 0


def test_dac_defaults_preserve_hf_14khz_on_44k_and_leaves_16k_none(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe required")

    from remove_ai_watermarks import audio

    seen: list[tuple[int, float | None]] = []

    def fake_dac(samples, sample_rate, *, local_files_only=False, device="cpu", preserve_hf_hz=None):
        seen.append((sample_rate, preserve_hf_hz))
        return samples

    monkeypatch.setattr(audio.audio_dac, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_dac, "resynthesize", fake_dac)

    src_44k = _write_wav(tmp_path / "src44.wav", sample_rate=44100, seconds=0.3, channels=1)
    audio.remove_audio_invisible(src_44k, tmp_path / "out44_default.wav", method="dac", force=True, verify=False)
    audio.remove_audio_invisible(
        src_44k, tmp_path / "out44_custom.wav", method="dac", preserve_hf_hz=12000.0, force=True, verify=False
    )
    src_16k = _write_wav(tmp_path / "src16.wav", sample_rate=16000, seconds=0.3, channels=1)
    audio.remove_audio_invisible(src_16k, tmp_path / "out16_default.wav", method="dac", force=True, verify=False)

    assert seen == [(44100, 14000.0), (44100, 12000.0), (16000, None)]


def test_unified_video_container_constants_match_all_pipelines() -> None:
    from remove_ai_watermarks._internal.constants import (
        AUDIO_EXTENSIONS,
        AUDIO_OUTPUT_EXTENSIONS,
        WAN_SUPPORTED_EXTENSIONS,
    )
    from remove_ai_watermarks.video import _REGENERATED_VIDEO_EXTENSIONS, VIDEO_EXTENSIONS

    assert WAN_SUPPORTED_EXTENSIONS == VIDEO_EXTENSIONS == _REGENERATED_VIDEO_EXTENSIONS
    assert AUDIO_OUTPUT_EXTENSIONS == AUDIO_EXTENSIONS
    for ext in (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv"):
        assert ext in WAN_SUPPORTED_EXTENSIONS
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".mka"):
        assert ext in AUDIO_OUTPUT_EXTENSIONS


def test_top_level_identify_audio_and_all_text_cli_dispatch(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    wav_src = _write_wav(tmp_path / "clip.wav", sample_rate=16000, seconds=0.2)
    if shutil.which("ffprobe") is not None:
        res_audio = CliRunner().invoke(main, ["identify", str(wav_src), "--no-visible", "--json"])
        assert res_audio.exit_code == 0, res_audio.output

    md_src = tmp_path / "note.md"
    md_src.write_text("# Title\n\nPlain text without watermarks.\n", encoding="utf-8")
    md_out = tmp_path / "note_clean.md"
    res_text = CliRunner().invoke(main, ["all", str(md_src), "-o", str(md_out)])
    assert res_text.exit_code in (0, 2), res_text.output


def test_m4v_10bit_hdr_raw_video_command_uses_mp4_muxer(tmp_path: Path) -> None:
    from remove_ai_watermarks.video_encoding import VideoEncodeProfile, raw_video_command

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required")
    prof = VideoEncodeProfile(pixel_format="yuv420p10le", component_depth=10)
    cmd = raw_video_command(tmp_path / "out.m4v", width=64, height=64, fps=24.0, crf=18, profile=prof)
    assert cmd[-3:-1] == ["-f", "mp4"]


def test_resolve_preset_preserves_explicit_cv2_backend() -> None:
    from remove_ai_watermarks.cli import _resolve_preset

    _eff_force, eff_backend, _lossless = _resolve_preset("balanced", force=False, backend="cv2")
    assert eff_backend == "cv2"


def test_truncated_aac_output_is_rejected_by_measure_fidelity(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe required")

    from remove_ai_watermarks._internal import audio_io, audio_quality

    src = _write_wav(tmp_path / "long.wav", sample_rate=44100, seconds=1.0, channels=1)
    short = _write_wav(tmp_path / "short.wav", sample_rate=44100, seconds=0.2, channels=1)
    src_info = audio_io.probe_audio(src)
    short_info = audio_io.probe_audio(short)
    trunc_aac = tmp_path / "trunc.aac"
    audio_io.write_audio(short_info, trunc_aac, audio_io.read_audio(short, short_info))
    with pytest.raises(RuntimeError, match="decoded sample-frame count"):
        audio_quality.measure_fidelity(src, src_info, trunc_aac)

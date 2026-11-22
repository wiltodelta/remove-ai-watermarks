"""Tests for Wave 7: 10-bit HEVC Seedance, clean profile names, and video/audio auto-pipeline."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from remove_ai_watermarks import audio, video, video_encoding, video_seedance
from remove_ai_watermarks._internal import audio_dac, audio_io
from remove_ai_watermarks.video_tools import require_video_tool

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_seedance_accepts_and_preserves_10bit_hevc_color_and_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seedance natively processes 10-bit HEVC MP4 without 8-bit downgrade or color-tag loss."""
    ffmpeg = require_video_tool("ffmpeg")
    src = tmp_path / "seedance_10bit.mp4"
    out = tmp_path / "seedance_10bit_clean.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=24:duration=0.25",
            "-c:v",
            "libx265",
            "-tag:v",
            "hvc1",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-x265-params",
            "log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt709",
            str(src),
        ],
        check=True,
    )

    def fake_reconstruct(path: Path, _runtime: Any, size: tuple[int, int], *, high_depth: bool = False) -> Any:
        import cv2

        for frame in video_seedance._frames(path, high_depth=high_depth):
            u8 = (frame >> 8).astype(np.uint8) if frame.dtype == np.uint16 else frame
            yield cv2.resize(u8, size, interpolation=cv2.INTER_AREA)

    monkeypatch.setattr(video_seedance, "is_available", lambda: True)
    monkeypatch.setattr(
        video_seedance,
        "load_video_vae_runtime",
        lambda **_kw: object(),
    )
    monkeypatch.setattr(video_seedance, "_reconstruct", fake_reconstruct)

    metrics = video_seedance.regenerate_seedance_experimental(src, out, device="cpu", quality="delivery")
    assert out.exists()
    assert metrics.frames == 6
    assert metrics.psnr_db > 30.0

    out_profile = video_encoding.probe_video_encode_profile(out)
    assert out_profile.pixel_format == "yuv420p10le"
    assert out_profile.component_depth == 10
    assert out_profile.color_primaries == "bt709"
    assert out_profile.color_transfer == "bt709"
    assert out_profile.color_space == "bt709"


def test_clean_video_profile_aliases_and_labels() -> None:
    """Clean profile names (seedance, wan21-synthid, wan21-seedance) validate and normalize."""
    from remove_ai_watermarks.cli import _video_profile_label

    for name, canonical in (
        ("seedance", "seedance-experimental"),
        ("wan21-synthid", "wan21-synthid-experimental"),
        ("wan21-seedance", "wan21-seedance-experimental"),
    ):
        video._validate_video_profile(name, "delivery", enabled=True)
        assert video._normalize_video_profile(name) == canonical
        assert "experimental" not in _video_profile_label(name).lower()


def test_video_provenance_auto_trigger_respects_clean_and_ai_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include_invisible='auto' triggers on AI metadata/visible marks and skips clean clips."""
    from remove_ai_watermarks import video_invisible

    assert video.video_has_invisible_provenance({"synthid_watermark": "Google SynthID"}, None) is True
    assert video.video_has_invisible_provenance({}, "seedance") is True
    assert video.video_has_invisible_provenance({}, None) is False

    # Clean clip with include_invisible='auto' must not call remove_video_invisible.
    src = tmp_path / "clean.mp4"
    out = tmp_path / "clean_out.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    monkeypatch.setattr(video, "_require_video_runtime", lambda: None)
    monkeypatch.setattr(video_invisible, "is_available", lambda: True)
    monkeypatch.setattr(
        video,
        "remove_video_visible",
        lambda s, o, **kw: video.VideoVisibleResult(
            source=s,
            output=None,
            mark="auto",
            total_frames=1,
            detected_frames=0,
            removed_frames=0,
            remaining_metadata={},
        ),
    )
    monkeypatch.setattr(
        video,
        "remove_video_metadata",
        lambda s, o, **kw: (
            o.write_bytes(s.read_bytes()),
            video.VideoMetadataResult(source=s, output=o, detected={}, remaining={}),
        )[1],
    )
    inv_calls: list[str] = []
    monkeypatch.setattr(
        video,
        "remove_video_invisible",
        lambda s, o, **kw: inv_calls.append(str(kw.get("profile"))),
    )

    res = video.remove_video_all(
        src,
        out,
        include_invisible="auto",
        trim_bumper="auto",
        _detected_metadata={},
    )
    assert inv_calls == []
    assert res.invisible_removed is False


def test_audio_provenance_auto_scrub_triggers_only_for_known_ai_vendors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should_auto_scrub_audio triggers for Google/ElevenLabs/Cartesia/OpenAI metadata when DAC is available."""
    stream = audio_io.AudioStreamInfo(
        index=0,
        sample_rate=48000,
        channels=2,
        codec="pcm_s16le",
        start_time=Fraction(0),
        duration=Fraction(2),
        channel_layout="stereo",
    )
    monkeypatch.setattr(audio_dac, "is_available", lambda: True)

    google_rep = audio.AudioProvenanceReport(
        tmp_path / "g.wav",
        "wav",
        stream,
        {"claim_generator": "Google Veo", "synthid_watermark": "Google SynthID"},
        "not_detected",
        (),
        (),
    )
    eleven_rep = audio.AudioProvenanceReport(
        tmp_path / "e.wav", "wav", stream, {"claim_generator": "ElevenLabs"}, "not_detected", (), ()
    )
    cartesia_rep = audio.AudioProvenanceReport(
        tmp_path / "c.wav", "wav", stream, {"ai_tool": "Cartesia Sonic"}, "not_detected", (), ()
    )
    clean_rep = audio.AudioProvenanceReport(tmp_path / "clean.wav", "wav", stream, {}, "not_detected", (), ())

    assert audio.should_auto_scrub_audio(google_rep, method="auto", explicit_vendor=None) is True
    assert audio.should_auto_scrub_audio(eleven_rep, method="auto", explicit_vendor=None) is True
    assert audio.should_auto_scrub_audio(cartesia_rep, method="auto", explicit_vendor=None) is False
    assert audio.should_auto_scrub_audio(clean_rep, method="auto", explicit_vendor=None) is False

    monkeypatch.setattr(audio_dac, "is_available", lambda: False)
    assert audio.should_auto_scrub_audio(google_rep, method="auto", explicit_vendor=None) is False

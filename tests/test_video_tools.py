"""Video installations resolve and execute both tools without a system FFmpeg."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from remove_ai_watermarks import video_tools


def test_base_install_does_not_bootstrap_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: None)
    monkeypatch.setattr(video_tools, "bundled_video_tools_available", lambda: False)
    video_tools._bundled_tools.cache_clear()
    try:
        assert video_tools.find_video_tool("ffprobe") is None
        with pytest.raises(RuntimeError, match=r"install remove-ai-watermarks\[video\]"):
            video_tools.require_video_tool("ffmpeg")
    finally:
        video_tools._bundled_tools.cache_clear()


def test_complete_system_pair_does_not_extract_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_tools.shutil, "which", lambda name: f"/system/{name}")
    monkeypatch.setattr(video_tools, "_bundled_tools", lambda: pytest.fail("unnecessary extraction"))
    assert video_tools.require_video_tool("ffmpeg") == "/system/ffmpeg"
    assert video_tools.require_video_tool("ffprobe") == "/system/ffprobe"


def test_partial_system_pair_uses_both_bundled_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_tools.shutil, "which", lambda name: "/system/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(video_tools, "_bundled_tools", lambda: ("/bundle/ffmpeg", "/bundle/ffprobe"))
    assert video_tools.require_video_tool("ffmpeg") == "/bundle/ffmpeg"
    assert video_tools.require_video_tool("ffprobe") == "/bundle/ffprobe"


def test_broken_bundle_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_tools.shutil, "which", lambda _name: None)

    def fail() -> None:
        raise OSError("unwritable cache")

    monkeypatch.setattr(video_tools, "_bundled_tools", fail)
    with pytest.raises(RuntimeError, match="reinstall remove-ai-watermarks"):
        video_tools.require_video_tool("ffmpeg")


@pytest.mark.skipif(not video_tools.bundled_video_tools_available(), reason="video extra not installed")
def test_offline_bundle_encodes_probes_muxes_and_strips_without_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from remove_ai_watermarks import metadata, video_encoding, video_seedance

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(video_tools.Path, "home", lambda: tmp_path)
    video_tools._bundled_tools.cache_clear()
    try:
        ffmpeg = video_tools.require_video_tool("ffmpeg")
        ffprobe = video_tools.require_video_tool("ffprobe")
        assert Path(ffmpeg).is_relative_to(tmp_path / ".cache")
        assert Path(ffprobe).is_relative_to(tmp_path / ".cache")
        assert os.environ["PATH"] == str(tmp_path)
        assert capsys.readouterr().out == ""

        source = tmp_path / "source.mp4"
        subprocess.run(  # noqa: S603 - Resolved executable and synthetic fixtures.
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=32x24:rate=4:duration=1",
                "-f",
                "lavfi",
                "-i",
                "sine=duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        profile = video_encoding.probe_video_encode_profile(source)
        assert profile.time_base is not None
        assert len(video_encoding.probe_video_timestamps(source)) == 4
        encoded = tmp_path / "encoded.mp4"
        command = video_encoding.raw_video_command(encoded, width=32, height=24, fps=4, crf=18, profile=profile)
        subprocess.run(  # noqa: S603 - Shared encoder command and synthetic frames.
            command, input=bytes(32 * 24 * 3 * 4), check=True, capture_output=True
        )
        muxed = tmp_path / "muxed.mp4"
        video_encoding.mux_encoded_video(encoded, source, muxed, strip_metadata=True)
        delivered = tmp_path / "delivered.mp4"
        video_seedance._encode_delivery(muxed, source, delivered)
        assert len(video_encoding.probe_video_timestamps(delivered)) == 4
        info = json.loads(
            subprocess.check_output(  # noqa: S603 - Resolved probe and synthetic fixtures.
                [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(delivered)],
                text=True,
            )
        )
        assert {stream["codec_type"] for stream in info["streams"]} == {"video", "audio"}

        tagged = tmp_path / "tagged.mkv"
        subprocess.run(  # noqa: S603 - Resolved executable and synthetic fixtures.
            [ffmpeg, "-v", "error", "-i", str(source), "-c", "copy", "-metadata", "AIGC=1", str(tagged)],
            check=True,
            capture_output=True,
        )
        cleaned = metadata._strip_with_ffmpeg(tagged, tmp_path / "cleaned.mkv")
        tags = subprocess.check_output(  # noqa: S603 - Resolved probe and synthetic fixtures.
            [ffprobe, "-v", "error", "-show_entries", "format_tags", "-of", "json", str(cleaned)],
            text=True,
        )
        assert "AIGC" not in tags
    finally:
        video_tools._bundled_tools.cache_clear()

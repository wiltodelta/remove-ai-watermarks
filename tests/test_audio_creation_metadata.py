"""Native Suno creation comments, independent of C2PA and waveform verification."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from remove_ai_watermarks import metadata
from remove_ai_watermarks._internal import audio_io
from remove_ai_watermarks._internal.audio_quality import measure_fidelity

COMMENT = "made with suno; created=2026-08-02T12:34:56.789Z; id=11111111-2222-4333-8444-555555555555"


def _source(tmp_path, suffix, *, comment=COMMENT, video=False):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    source = tmp_path / ("source" + suffix)
    command = [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=0.25:sample_rate=48000"]
    if video:
        command.extend(["-f", "lavfi", "-i", "color=black:size=32x32:rate=12:duration=0.25", "-c:v", "libx264"])
    command.extend(["-metadata", "comment=" + comment, "-metadata", "title=Ordinary title"])
    if suffix in {".m4a", ".mp4"}:
        command.extend(["-c:a", "aac"])
    command.append(str(source))
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603 - Local synthetic fixture, no shell.
    return source


@pytest.mark.parametrize("suffix", [".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".opus"])
def test_native_comment_is_detected_and_removed_without_changing_audio(tmp_path, suffix):
    source = _source(tmp_path, suffix)
    assert metadata.get_ai_metadata(source)["audio_generator"] == "Suno"
    assert metadata.has_ai_metadata(source)
    output = tmp_path / ("clean" + suffix)
    metadata.strip_and_verify(source, output)
    assert metadata.get_ai_metadata(output) == {}
    assert not metadata.has_ai_metadata(output)
    fidelity = measure_fidelity(source, audio_io.probe_audio(source), output)
    assert fidelity.max_abs_error == 0
    probe = subprocess.run(  # noqa: S603 - Fixed local probe.
        [shutil.which("ffprobe"), "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "made with suno" not in probe.stdout.lower()


def test_studio_comment_is_a_creation_declaration(tmp_path):
    source = _source(tmp_path, ".wav", comment="made with suno studio; created=2026-08-02T12:34:56Z")
    assert metadata.get_ai_metadata(source)["audio_generator"] == "Suno Studio"


@pytest.mark.parametrize(
    "comment",
    [
        "I compared this recording to Suno",
        "not made with suno",
        "made with suno",
        COMMENT + "; unrelated caption",
        COMMENT.replace("2026-08-02", "2026-02-30"),
        COMMENT.replace("11111111-2222-4333-8444-555555555555", "not-a-uuid"),
        COMMENT.replace("11111111-2222-4333-8444-555555555555", "1" * 36),
    ],
)
def test_mentions_and_malformed_creation_comments_are_not_evidence(tmp_path, comment):
    source = _source(tmp_path, ".wav", comment=comment)
    assert "audio_generator" not in metadata.get_ai_metadata(source)


def test_audio_declaration_does_not_attribute_video_pixels(tmp_path):
    source = _source(tmp_path, ".mp4", video=True)
    assert "audio_generator" not in metadata.get_ai_metadata(source)
    output = tmp_path / "preserved.mp4"
    metadata.remove_ai_metadata(source, output)
    assert source.read_bytes() == output.read_bytes()


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-le", "utf-16-be"])
def test_metadata_comment_encodings_share_one_strict_parser(encoding):
    from remove_ai_watermarks._internal.audio_creation import creation_tool

    assert creation_tool(COMMENT.encode(encoding)) == "Suno"


def test_native_tag_cache_tracks_source_changes(tmp_path):
    source = _source(tmp_path, ".wav")
    assert metadata.get_ai_metadata(source)["audio_generator"] == "Suno"
    source = _source(tmp_path, ".wav", comment="An ordinary recording")
    assert "audio_generator" not in metadata.get_ai_metadata(source)


def test_unrelated_comment_text_does_not_reach_logs(tmp_path, caplog):
    import logging

    source = _source(tmp_path, ".wav", comment="PRIVATE_COMMENT_TEST_SENTINEL")
    caplog.set_level(logging.INFO)
    assert "audio_generator" not in metadata.get_ai_metadata(source)
    assert "PRIVATE_COMMENT_TEST_SENTINEL" not in caplog.text


def test_isobmff_comment_patch_is_surgical_and_counted(tmp_path):
    from remove_ai_watermarks._internal.isobmff import strip_isobmff_media_file

    source = _source(tmp_path, ".m4a")
    output = tmp_path / "patched.m4a"
    before = source.read_bytes()
    start = before.index(COMMENT.encode())
    boxes, tags = strip_isobmff_media_file(source, output, strip_audio_comments=True)
    assert boxes == 0
    assert tags == 1
    after = output.read_bytes()
    assert len(before) == len(after)
    assert all(start <= i < start + len(COMMENT) for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b)


def test_creation_text_inside_waveform_is_not_metadata(tmp_path):
    import wave

    source = tmp_path / "payload.wav"
    data = COMMENT.encode() * 16
    data += b"\0" * (len(data) % 2)
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(data)
    assert "audio_generator" not in metadata.get_ai_metadata(source)


def test_attached_cover_does_not_hide_the_audio_declaration(tmp_path):
    from PIL import Image

    source = _source(tmp_path, ".mp3")
    cover = tmp_path / "cover.jpg"
    Image.new("RGB", (32, 32), "blue").save(cover)
    output = tmp_path / "with-cover.mp3"
    subprocess.run(  # noqa: S603 - Fixed local fixture mux, no shell.
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(cover),
            "-map",
            "0:a",
            "-map",
            "1:v",
            "-c",
            "copy",
            "-disposition:v",
            "attached_pic",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert metadata.get_ai_metadata(output)["audio_generator"] == "Suno"


@pytest.mark.parametrize("kind", ["video", "subtitle"])
def test_auxiliary_stream_comment_does_not_attribute_audio(tmp_path, monkeypatch, kind):
    import json

    from remove_ai_watermarks._internal import audio_creation

    source = tmp_path / "auxiliary.wav"
    source.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    document = {
        "streams": [
            {"codec_type": "audio"},
            {"codec_type": kind, "disposition": {"attached_pic": 1}, "tags": {"comment": COMMENT}},
        ]
    }

    def probe(command, *, stdout, **_kwargs):
        stdout.write(json.dumps(document).encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_creation.shutil, "which", lambda _: "/synthetic/ffprobe")
    monkeypatch.setattr(audio_creation.subprocess, "run", probe)
    assert audio_creation.audio_creation_tool(source) is None


def test_audio_creation_falls_back_to_bundled_ffprobe(tmp_path, monkeypatch):
    import json

    from remove_ai_watermarks import video_tools
    from remove_ai_watermarks._internal import audio_creation

    source = tmp_path / "suno.wav"
    source.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    audio_creation._probe.cache_clear()
    document = {"format": {"tags": {"comment": COMMENT}}, "streams": [{"codec_type": "audio"}]}

    def probe(command, *, stdout, **_kwargs):
        assert command[0] == "/bundled/ffprobe"
        stdout.write(json.dumps(document).encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_creation.shutil, "which", lambda _: None)
    monkeypatch.setattr(video_tools, "find_video_tool", lambda name: "/bundled/ffprobe" if name == "ffprobe" else None)
    monkeypatch.setattr(audio_creation.subprocess, "run", probe)
    assert audio_creation.audio_creation_tool(source) == "Suno"

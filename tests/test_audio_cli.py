"""Audio CLI reports and exit codes, including unknown and failed verification."""

from __future__ import annotations

import json
import shutil
import wave

import numpy as np
import pytest
from click.testing import CliRunner

from remove_ai_watermarks.cli import main

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")


def _source(tmp_path):
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    return path


def test_metadata_only_json_never_claims_a_clean_audio_verdict(tmp_path):
    source = _source(tmp_path)
    result = CliRunner().invoke(main, ["audio", "identify", str(source), "--no-invisible", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["watermark_status"] == "unverified"
    assert report["stream"]["sample_rate"] == 16000
    assert len(report["sha256"]) == 64


def test_no_compatible_source_signal_exits_two_without_output(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal import audioseal

    monkeypatch.setattr(audioseal, "is_available", lambda: False)
    source = _source(tmp_path)
    output = tmp_path / "output.wav"
    result = CliRunner().invoke(main, ["audio", "invisible", str(source), "-o", str(output), "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["skipped"] is True
    assert not output.exists()


def test_forced_unverified_processing_has_a_structured_result(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "output.wav"
    result = CliRunner().invoke(
        main, ["audio", "invisible", str(source), "-o", str(output), "--force", "--no-verify", "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["verified_removed"] is None
    assert report["frames_processed"] == 16000
    assert report["after"]["watermark_status"] == "unverified"
    assert output.is_file()


def test_bigvgan_cli_routes_device_and_retains_unverified_report(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []
    monkeypatch.setattr(audio.audio_vocoder, "is_available", lambda: True)

    def process(samples, rate, *, device, local_files_only):
        calls.append((samples.shape, rate, device, local_files_only))
        return samples

    monkeypatch.setattr(audio.audio_vocoder, "resynthesize", process)
    result = CliRunner().invoke(
        main,
        [
            "audio",
            "invisible",
            str(_source(tmp_path)),
            "--method",
            "bigvgan",
            "--vocoder-device",
            "mps",
            "--force",
            "--no-verify",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [((16000, 1), 16000, "mps", False)]
    report = json.loads(result.output)
    assert report["method"] == "bigvgan"
    assert report["vocoder_device"] == "mps"
    assert report["verified_removed"] is None


def test_failed_verification_exits_one_and_preserves_existing_output(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal import audioseal

    monkeypatch.setattr(audioseal, "is_available", lambda: True)
    monkeypatch.setattr(audioseal, "load_pinned_detector", object)
    monkeypatch.setattr(audioseal, "read", lambda *_: audioseal.AudioSealReading(1.0, 1.0, (0, 1) * 8))
    from remove_ai_watermarks._internal import audio_codec

    monkeypatch.setattr(audio_codec, "is_available", lambda: False)
    source = _source(tmp_path)
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")
    result = CliRunner().invoke(main, ["audio", "invisible", str(source), "-o", str(output), "--json"])
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert report["after"]["watermark_status"] == "detected"
    assert "not published" in report["error"]
    assert output.read_bytes() == b"existing output"


def test_missing_audio_runtime_has_an_install_hint(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal import audio_io

    monkeypatch.setattr(audio_io, "is_available", lambda: False)
    source = _source(tmp_path)
    output = tmp_path / "output.wav"
    result = CliRunner().invoke(main, ["audio", "invisible", str(source), "-o", str(output), "--force", "--no-verify"])
    assert result.exit_code == 1
    assert "remove-ai-watermarks[audio]" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


def test_codec_method_reports_its_missing_extra(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal import audio_codec

    monkeypatch.setattr(audio_codec, "is_available", lambda: False)
    output = tmp_path / "codec.wav"
    result = CliRunner().invoke(
        main,
        [
            "audio",
            "invisible",
            str(_source(tmp_path)),
            "-o",
            str(output),
            "--method",
            "encodec",
            "--force",
            "--no-verify",
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "remove-ai-watermarks[audio-codec]" in json.loads(result.output)["error"]
    assert "Traceback" not in result.output
    assert not output.exists()


@pytest.mark.parametrize("scheme", ["silentcipher", "silentcipher16"])
def test_silentcipher_selection_keeps_a_short_source_unverified(tmp_path, monkeypatch, scheme):
    from remove_ai_watermarks._internal import silentcipher, silentcipher16

    backend = silentcipher16 if scheme == "silentcipher16" else silentcipher
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "load_model", object)
    result = CliRunner().invoke(main, ["audio", "identify", str(_source(tmp_path)), "--scheme", scheme, "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["watermark_status"] == "unverified"
    assert report["detections"][0]["scheme"] == scheme
    if scheme == "silentcipher16":
        assert report["detections"][0]["decoder_sample_rate"] == 16000
    assert "three seconds" in report["detections"][0]["whole_clip_error"]


def test_mimi_cli_routes_device_and_retains_unverified_report(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)

    def process(samples, rate, *, local_files_only):
        calls.append((samples.shape, rate, "cpu", local_files_only))
        return samples

    monkeypatch.setattr(audio.audio_mimi, "resynthesize", process)
    result = CliRunner().invoke(
        main,
        [
            "audio",
            "invisible",
            str(_source(tmp_path)),
            "--method",
            "mimi",
            "--vocoder-device",
            "cpu",
            "--force",
            "--no-verify",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [((16000, 1), 16000, "cpu", False)]
    report = json.loads(result.output)
    assert report["method"] == "mimi"
    assert report["vocoder_device"] == "cpu"
    assert report["verified_removed"] is None


def test_focalcodec_cli_routes_device_and_retains_unverified_report(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []
    monkeypatch.setattr(audio.audio_focalcodec, "is_available", lambda: True)

    def process(samples, rate, *, local_files_only):
        calls.append((samples.shape, rate, "cpu", local_files_only))
        return samples

    monkeypatch.setattr(audio.audio_focalcodec, "resynthesize", process)
    result = CliRunner().invoke(
        main,
        [
            "audio",
            "invisible",
            str(_source(tmp_path)),
            "--method",
            "focalcodec",
            "--vocoder-device",
            "cpu",
            "--force",
            "--no-verify",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [((16000, 1), 16000, "cpu", False)]
    report = json.loads(result.output)
    assert report["method"] == "focalcodec"
    assert report["vocoder_device"] == "cpu"
    assert report["verified_removed"] is None


def test_mimi_choices_align_cli_shared_catalog_and_legacy_mcp():
    from remove_ai_watermarks import integration, mcp_server

    operations = integration.operation_catalog()
    for name in ["cli_audio_invisible", "cli_video_audio"]:
        operation = next(o for o in operations if o["id"] == name)
        method = next(p for p in operation["parameters"] if p["name"] == "method")
        assert "mimi" in method["schema"]["enum"]
        assert "focalcodec" in method["schema"]["enum"]
        assert "experimental" in method["schema"]["description"]
    for name in ["audio_invisible", "video_audio"]:
        tool = next(t for t in mcp_server.TOOLS if t["name"] == name)
        assert "mimi" in tool["inputSchema"]["properties"]["method"]["enum"]
        assert "focalcodec" in tool["inputSchema"]["properties"]["method"]["enum"]


def test_audio_metadata_all_and_batch_cli_commands(tmp_path, monkeypatch):
    import subprocess

    from remove_ai_watermarks import audio
    from remove_ai_watermarks._internal import audio_io

    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)
    monkeypatch.setattr(audio.audioseal, "available", lambda: False)

    suno_comment = "Made with Suno; created=2026-05-01T12:00:00Z; id=12345678-1234-5678-9abc-123456789abc"
    suno_mp3 = tmp_path / "clip.mp3"
    subprocess.run(  # noqa: S603 - Fixed local fixture encoder, no shell.
        [
            audio_io.require_tool("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=24000:duration=1.0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-metadata",
            f"comment={suno_comment}",
            str(suno_mp3),
        ],
        check=True,
    )

    runner = CliRunner()
    chk = runner.invoke(main, ["audio", "metadata", "--check", "--json", str(suno_mp3)])
    assert chk.exit_code == 0, chk.output
    assert json.loads(chk.output)["has_ai_metadata"] is True

    rem = runner.invoke(main, ["audio", "metadata", "--remove", "--json", str(suno_mp3)])
    assert rem.exit_code == 0, rem.output
    rem_payload = json.loads(rem.output)
    assert rem_payload["remaining"] == {}
    assert (tmp_path / "clip_clean.mp3").is_file()

    all_out = tmp_path / "clip_all.mp3"
    all_cmd = runner.invoke(main, ["audio", "all", "-o", str(all_out), "--json", str(suno_mp3)])
    assert all_cmd.exit_code == 0, all_cmd.output
    assert json.loads(all_cmd.output)["invisible_applied"] is False
    assert all_out.is_file()

    empty_dir = tmp_path / "empty_audio"
    empty_dir.mkdir()
    empty_res = runner.invoke(main, ["audio", "batch", str(empty_dir)])
    assert empty_res.exit_code == 2

    batch_dir = tmp_path / "batch_audio"
    batch_dir.mkdir()
    (batch_dir / "track.mp3").write_bytes(suno_mp3.read_bytes())
    batch_res = runner.invoke(main, ["audio", "batch", "--json", str(batch_dir)])
    assert batch_res.exit_code == 0, batch_res.output
    batch_payload = json.loads(batch_res.output)
    assert len(batch_payload["items"]) == 1
    assert batch_payload["items"][0]["changed"] is True

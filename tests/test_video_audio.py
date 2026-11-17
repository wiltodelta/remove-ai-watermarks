"""Audio-only video processing preserves encoded visual packets and timing."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from tests.test_audio import _source, _spectral_detector

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")


def _video(
    tmp_path,
    *,
    audio_codec="pcm_f32le",
    shifted=False,
    original_disposition=True,
    container=".mkv",
    gap=False,
    selected_codec=None,
    selected_offset=0.5,
    first_offset=0.25,
):
    audio = _source(tmp_path, frames=24031)
    source = tmp_path / ("source" + container)
    command = [
        shutil.which("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x64:rate=12:duration=1",
        "-itsoffset",
        str(first_offset),
        "-i",
        str(audio),
        "-itsoffset",
        str(selected_offset),
        "-i",
        str(audio),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-c:a",
        audio_codec,
    ]
    if selected_codec:
        command.extend(["-c:a:1", selected_codec])
    if shifted:
        command.extend(["-output_ts_offset", "2"])
    if gap:
        command.extend(["-filter:a:0", "asetpts=PTS+gte(T\\,0.7)*0.25/TB"])
    command.extend(
        [
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=rus",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default+original" if original_disposition else "default",
        ]
    )
    subprocess.run([*command, str(source)], check=True)  # noqa: S603 - Fixed local fixture encoder, no shell.
    return source


def test_video_audio_verifies_actual_mux_and_keeps_other_tracks(tmp_path, monkeypatch):
    from remove_ai_watermarks.video_audio import remove_video_audio

    _spectral_detector(monkeypatch)
    source, output = _video(tmp_path), tmp_path / "processed.mkv"
    original = source.read_bytes()
    result = remove_video_audio(source, output, stream=1)
    assert result.verified_removed is True
    assert result.frames_processed == 24031
    assert result.unchanged_streams == (0, 1)
    assert result.before.stream.index == result.after.stream.index == 2
    assert abs(result.source_audio_start - result.output_audio_start) <= 0.001
    assert source.read_bytes() == original
    assert result.after.source == output
    from remove_ai_watermarks.video_audio import _streams

    tracks = _streams(output)
    assert [s.get("tags", {}).get("language") for s in tracks if s["codec_type"] == "audio"] == ["eng", "rus"]
    assert tracks[2]["disposition"]["default"] == tracks[2]["disposition"]["original"] == 1


def test_bigvgan_selected_track_routes_device_and_checks_final_mux(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio
    from remove_ai_watermarks.video_audio import remove_video_audio

    calls = []
    monkeypatch.setattr(audio.audio_vocoder, "is_available", lambda: True)

    def process(samples, rate, *, device, local_files_only):
        calls.append((samples.shape, rate, device, local_files_only))
        return samples * 0.5

    monkeypatch.setattr(audio.audio_vocoder, "resynthesize", process)
    result = remove_video_audio(
        _video(tmp_path),
        tmp_path / "vocoder.mkv",
        stream=1,
        method="bigvgan",
        vocoder_device="mps",
        force=True,
        verify=False,
    )
    assert calls == [((24031, 2), 16000, "mps", False)]
    assert result.method == "bigvgan"
    assert result.vocoder_device == "mps"
    assert result.verified_removed is None
    assert result.unchanged_streams == (0, 1)
    assert result.fidelity.frames == 24031
    assert result.fidelity.snr_db == pytest.approx(6.0206, abs=0.001)


def test_resistant_video_audio_preserves_existing_output(tmp_path, monkeypatch):
    from remove_ai_watermarks.audio import AudioVerificationIncomplete
    from remove_ai_watermarks.video_audio import remove_video_audio

    _spectral_detector(monkeypatch, resistant=True)
    output = tmp_path / "processed.mkv"
    output.write_bytes(b"existing output")
    with pytest.raises(AudioVerificationIncomplete):
        remove_video_audio(_video(tmp_path), output)
    assert output.read_bytes() == b"existing output"


def test_audio_only_source_is_not_a_video(tmp_path):
    from remove_ai_watermarks.video_audio import remove_video_audio

    with pytest.raises(ValueError, match="video stream"):
        remove_video_audio(_source(tmp_path), tmp_path / "output.mkv", force=True, verify=False)


@pytest.mark.parametrize(
    ("codec", "suffix", "shifted"),
    [
        ("pcm_f32le", ".mov", False),
        ("aac", ".mkv", False),
        ("pcm_f32le", ".mkv", True),
    ],
)
def test_video_audio_preserves_decoded_timeline_across_muxes(tmp_path, codec, suffix, shifted):
    from remove_ai_watermarks.video_audio import remove_video_audio

    source = _video(
        tmp_path, audio_codec=codec, shifted=shifted, original_disposition=suffix != ".mov", container=suffix
    )
    result = remove_video_audio(source, tmp_path / ("processed" + suffix), force=True, verify=False)
    assert result.frames_processed == result.fidelity.frames
    assert result.frames_processed >= 24031
    assert abs(result.source_audio_start - result.output_audio_start) <= 0.001
    if shifted:
        assert result.source_audio_start > 2


def test_conversion_that_changes_stream_disposition_is_rejected(tmp_path):
    from remove_ai_watermarks.video_audio import remove_video_audio

    source = _video(tmp_path, original_disposition=False)
    output = tmp_path / "processed.mov"
    output.write_bytes(b"existing output")
    with pytest.raises(RuntimeError, match="disposition"):
        remove_video_audio(source, output, force=True, verify=False)
    assert output.read_bytes() == b"existing output"


def test_final_mux_verdict_is_checked_independently(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio
    from remove_ai_watermarks.video_audio import remove_video_audio

    _spectral_detector(monkeypatch)
    real = audio.identify_audio

    def identify(path, **kwargs):
        report = real(path, **kwargs)
        if path.suffix == ".mkv" and path.name.startswith(".processed-"):
            return replace(report, watermark_status="detected")
        return report

    monkeypatch.setattr(audio, "identify_audio", identify)
    output = tmp_path / "processed.mkv"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        remove_video_audio(_video(tmp_path), output)
    assert output.read_bytes() == b"existing output"


def test_discontinuous_source_audio_is_rejected(tmp_path):
    from remove_ai_watermarks.video_audio import remove_video_audio

    source = _video(tmp_path, gap=True)
    output = tmp_path / "processed.mkv"
    with pytest.raises(ValueError, match="Discontinuous"):
        remove_video_audio(source, output, force=True, verify=False)
    assert not output.exists()


def test_video_audio_cli_reports_selected_track(tmp_path):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source, output = _video(tmp_path), tmp_path / "processed.mkv"
    result = CliRunner().invoke(
        main, ["video", "audio", str(source), "-o", str(output), "--stream", "1", "--force", "--no-verify", "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["verified_removed"] is None
    assert report["unchanged_streams"] == [0, 1]
    assert report["after"]["stream"]["index"] == 2
    assert output.is_file()


@pytest.mark.parametrize("fault", ["video", "offset"])
def test_corrupted_mux_is_rejected_atomically(tmp_path, monkeypatch, fault):
    from remove_ai_watermarks import video_audio

    _spectral_detector(monkeypatch)
    source = _video(tmp_path)
    output = tmp_path / "processed.mkv"
    output.write_bytes(b"existing output")
    popen = subprocess.Popen

    def corrupt(command, **kwargs):
        command = list(command)
        if "-itsoffset" in command:
            if fault == "video":
                command[-1:-1] = ["-c:v", "libx264", "-crf", "35"]
            else:
                command[command.index("-itsoffset") + 1] = "0.75"
        return popen(command, **kwargs)

    monkeypatch.setattr(video_audio.subprocess, "Popen", corrupt)
    with pytest.raises(RuntimeError, match=r"packets changed|audio timeline"):
        video_audio.remove_video_audio(source, output, force=True, verify=False)
    assert output.read_bytes() == b"existing output"


def test_mimi_selected_track_routes_device_and_checks_final_mux(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio
    from remove_ai_watermarks.video_audio import remove_video_audio

    calls = []
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)

    def process(samples, rate, *, local_files_only):
        calls.append((samples.shape, rate, "cpu", local_files_only))
        return samples * 0.5

    monkeypatch.setattr(audio.audio_mimi, "resynthesize", process)
    result = remove_video_audio(
        _video(tmp_path),
        tmp_path / "vocoder.mkv",
        stream=1,
        method="mimi",
        vocoder_device="cpu",
        force=True,
        verify=False,
    )
    assert calls == [((24031, 2), 16000, "cpu", False)]
    assert result.method == "mimi"
    assert result.vocoder_device == "cpu"
    assert result.verified_removed is None
    assert result.unchanged_streams == (0, 1)
    assert result.fidelity.frames == 24031
    assert result.fidelity.snr_db == pytest.approx(6.0206, abs=0.001)


def test_focalcodec_selected_track_routes_device_and_checks_final_mux(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio
    from remove_ai_watermarks.video_audio import remove_video_audio

    calls = []
    monkeypatch.setattr(audio.audio_focalcodec, "is_available", lambda: True)

    def process(samples, rate, *, local_files_only):
        calls.append((samples.shape, rate, local_files_only))
        return samples * 0.5

    monkeypatch.setattr(audio.audio_focalcodec, "resynthesize", process)
    result = remove_video_audio(
        _video(tmp_path),
        tmp_path / "focal.mkv",
        stream=1,
        method="focalcodec",
        vocoder_device="cpu",
        force=True,
        verify=False,
    )
    assert calls == [((24031, 2), 16000, False)]
    assert result.method == "focalcodec"
    assert result.vocoder_device == "cpu"
    assert result.verified_removed is None
    assert result.unchanged_streams == (0, 1)
    assert result.fidelity.frames == 24031


def test_mimi_final_mux_positive_rejected_after_negative_intermediate(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio
    from remove_ai_watermarks.video_audio import remove_video_audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda x, *a, **k: x * 0.01)
    real = audio.identify_audio
    checked = []

    def identify(path, **kwargs):
        report = real(path, **kwargs)
        if path.suffix == ".mkv" and path.name.startswith(".processed-"):
            checked.append(report.watermark_status)
            return replace(report, watermark_status="detected")
        return report

    monkeypatch.setattr(audio, "identify_audio", identify)
    output = tmp_path / "processed.mkv"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        remove_video_audio(_video(tmp_path), output, method="mimi", stream=1)
    assert checked == ["not_detected"]
    assert output.read_bytes() == b"existing output"


def test_mp4_aac_preserves_odd_sample_count_and_other_tracks(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda x, *a, **k: x * 0.01)
    source = _video(
        tmp_path,
        audio_codec="aac",
        selected_codec="pcm_f32le",
        container=".mov",
        original_disposition=False,
        selected_offset=0,
        first_offset=0,
    )
    popen = subprocess.Popen
    mux_commands = []

    def capture(command, **kwargs):
        if "-itsoffset" in command:
            mux_commands.append(command)
        return popen(command, **kwargs)

    monkeypatch.setattr(video_audio.subprocess, "Popen", capture)
    output = tmp_path / "processed.mp4"
    result = video_audio.remove_video_audio(source, output, stream=1, method="mimi")
    assert result.verified_removed is True
    assert result.frames_processed == result.fidelity.frames == 24031
    assert result.source_audio_start == result.output_audio_start
    assert result.source_audio_start == 0
    assert result.after.stream.codec == "aac"
    assert result.after.stream.sample_rate == 16000
    assert result.after.stream.channels == 2
    assert result.unchanged_streams == (0, 1)
    assert "lossy AAC" in result.caveats[-1]
    assert len(mux_commands) == 1
    assert mux_commands[0][mux_commands[0].index("-c:2") + 1] == "aac"
    assert mux_commands[0][mux_commands[0].index("-b:2") + 1] == "192k"
    trimming = list(video_audio._audio_packet_trimming(source, 1))
    assert trimming[0][0] == 1024
    assert trimming[-1][1] > 0
    assert trimming == list(video_audio._audio_packet_trimming(output, 1))


@pytest.mark.parametrize("status", ["detected", "unverified"])
def test_mp4_rechecks_actual_AAC_after_negative_intermediate(tmp_path, monkeypatch, status):
    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    real = audio.identify_audio
    checked = []

    def identify(path, **kwargs):
        report = real(path, **kwargs)
        if path.suffix == ".mp4" and path.name.startswith(".processed-"):
            checked.append(report.stream.codec)
            return replace(report, watermark_status=status)
        return report

    monkeypatch.setattr(audio, "identify_audio", identify)
    source = _video(
        tmp_path,
        audio_codec="aac",
        selected_codec="pcm_f32le",
        container=".mov",
        original_disposition=False,
        selected_offset=0,
    )
    output = tmp_path / "processed.mp4"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        video_audio.remove_video_audio(source, output, stream=1, method="spectral")
    assert checked == ["aac"]
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("fault", ["count", "offset", "priming", "tail"])
def test_mp4_rejects_changed_sample_count_timing_or_copied_AAC_trim(tmp_path, monkeypatch, fault):
    from remove_ai_watermarks import video_audio

    source = _video(
        tmp_path,
        audio_codec="aac",
        selected_codec="pcm_f32le",
        container=".mov",
        original_disposition=False,
        selected_offset=0,
    )
    output = tmp_path / "processed.mp4"
    output.write_bytes(b"existing output")
    if fault in {"count", "offset"}:
        original = video_audio._timeline

        def timeline(path, *args):
            start, count = original(path, *args)
            if path.suffix == ".mp4":
                return (start + 1 if fault == "offset" else start), (count + 1 if fault == "count" else count)
            return start, count

        monkeypatch.setattr(video_audio, "_timeline", timeline)
    else:
        original = video_audio._audio_packet_trimming

        def trimming(path, index):
            packets = list(original(path, index))
            for ordinal, (skip, discard) in enumerate(packets):
                if path.suffix == ".mp4":
                    skip += fault == "priming" and ordinal == 0
                    discard += fault == "tail" and ordinal == len(packets) - 1
                yield skip, discard

        monkeypatch.setattr(video_audio, "_audio_packet_trimming", trimming)
    with pytest.raises(RuntimeError, match=r"audio timeline|priming or tail"):
        video_audio.remove_video_audio(source, output, stream=1, force=True, verify=False, method="spectral")
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("shifted", [False, True])
def test_mp4_keeps_nonzero_offset_or_rejects_changed_AAC_priming(tmp_path, shifted):
    from remove_ai_watermarks import video_audio

    source = _video(
        tmp_path,
        audio_codec="aac",
        selected_codec="pcm_f32le",
        container=".mov",
        original_disposition=False,
        shifted=shifted,
    )
    output = tmp_path / "processed.mp4"
    output.write_bytes(b"existing output")
    failure_message = None
    try:
        result = video_audio.remove_video_audio(source, output, stream=1, method="spectral", force=True, verify=False)
    except RuntimeError as exc:
        failure_message = str(exc)
    if failure_message is not None:
        assert failure_message == "Mux changed the selected audio timeline"
        assert output.read_bytes() == b"existing output"
    else:
        assert result.frames_processed == result.fidelity.frames == 24031
        assert result.source_audio_start == result.output_audio_start == (2.5 if shifted else 0.5)


def test_auto_falls_back_to_combined_when_mp4_aac_relapses_after_negative_wav(tmp_path, monkeypatch):
    from pathlib import Path

    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.audio_codec, "is_available", lambda: True)
    codec_calls = []

    def resynthesize(chunks, sample_rate, *, bandwidth_kbps, stereo_mode="independent"):
        codec_calls.append((sample_rate, bandwidth_kbps))
        for chunk in chunks:
            yield chunk * 0.5

    monkeypatch.setattr(audio.audio_codec, "resynthesize_chunks", resynthesize)
    source = _video(
        tmp_path,
        audio_codec="aac",
        selected_codec="pcm_f32le",
        container=".mov",
        original_disposition=False,
        selected_offset=0,
        first_offset=0,
    )
    real_identify = audio.identify_audio
    inspections = []
    mp4_attempts = 0

    def identify(path, **kwargs):
        nonlocal mp4_attempts
        resolved = Path(path)
        inspections.append(resolved)
        report = real_identify(path, **kwargs)
        if resolved.suffix == ".mp4" and resolved.name.startswith(".processed-"):
            mp4_attempts += 1
            if mp4_attempts == 1:
                return replace(report, watermark_status="detected")
        return report

    monkeypatch.setattr(audio, "identify_audio", identify)
    output = tmp_path / "processed.mp4"
    output.write_bytes(b"existing output")
    result = video_audio.remove_video_audio(source, output, stream=1, method="auto")
    assert mp4_attempts == 2
    assert codec_calls == [(16000, 24.0)]
    assert inspections.count(source.resolve()) == 1
    assert result.method == "combined"
    assert result.verified_removed is True
    assert result.quality_passed is True


def test_video_audio_quality_gate_rejects_silent_erasure_unless_forced(tmp_path, monkeypatch):
    import numpy as np

    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda samples, *a, **k: np.zeros_like(samples))
    source = _video(tmp_path)
    output = tmp_path / "processed.mkv"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete, match="quality gate"):
        video_audio.remove_video_audio(source, output, stream=1, method="mimi")
    assert output.read_bytes() == b"existing output"

    forced = video_audio.remove_video_audio(source, output, stream=1, method="mimi", force=True)
    assert forced.verified_removed is True
    assert forced.quality_passed is False


def test_remove_video_audio_all_streams_cleans_multiple_tracks_and_cli_all_streams(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from remove_ai_watermarks import audio, video_audio
    from remove_ai_watermarks.cli import main

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)

    source = _video(tmp_path, first_offset=0, selected_offset=0)
    output = tmp_path / "all_cleaned.mkv"
    res = video_audio.remove_video_audio_all_streams(source, output, scheme="auto")
    assert res.skipped is False
    assert res.processed_ordinals == (0, 1)
    assert res.unchanged_streams == (0,)
    assert res.verified_removed is True
    assert res.quality_passed is True
    assert len(res.streams) == 2
    assert all(s.verified_removed is True for s in res.streams)
    assert audio.identify_audio(output, stream=0, scheme="auto").watermark_status == "not_detected"
    assert audio.identify_audio(output, stream=1, scheme="auto").watermark_status == "not_detected"

    # Re-running on the already-cleaned output skips cleanly with exit code 2 via CLI --all-streams
    cli_out = tmp_path / "second_pass.mkv"
    skip_run = CliRunner().invoke(
        main,
        ["video", "audio", "--all-streams", "--scheme", "auto", "-o", str(cli_out), "--json", str(output)],
    )
    assert skip_run.exit_code == 2, skip_run.output
    assert json.loads(skip_run.output)["skipped"] is True
    assert not cli_out.exists()

    pos_out = tmp_path / "cli_cleaned.mkv"
    pos_run = CliRunner().invoke(
        main,
        ["video", "audio", "--all-streams", "--scheme", "auto", "-o", str(pos_out), "--json", str(source)],
    )
    assert pos_run.exit_code == 0, pos_run.output
    payload = json.loads(pos_run.output)
    assert payload["processed_ordinals"] == [0, 1]
    assert payload["verified_removed"] is True
    assert pos_out.is_file()

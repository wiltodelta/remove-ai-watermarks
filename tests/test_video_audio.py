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
    assert mux_commands[0][mux_commands[0].index("-b:2") + 1] == "256k"
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


def test_remove_video_audio_all_streams_preserves_existing_destination_on_late_failure(tmp_path, monkeypatch):
    from pathlib import Path

    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)

    source = _video(tmp_path, first_offset=0, selected_offset=0)
    existing = tmp_path / "existing.mkv"
    sentinel = b"EXISTING-DESTINATION-BYTES-DO-NOT-CLOBBER"
    existing.write_bytes(sentinel)
    fresh = tmp_path / "fresh.mkv"

    # 1. Multi-track late quality-gate failure after all inner remux steps succeed
    orig_quality = video_audio.audio_quality_passed
    calls = 0

    def fail_outer_quality(fidelity, *, lossy):
        nonlocal calls
        calls += 1
        # Inner remove_video_audio runs 2 times (once per track); fail on the outer aggregate check
        if calls >= 3:
            return False
        return orig_quality(fidelity, lossy=lossy)

    monkeypatch.setattr(video_audio, "audio_quality_passed", fail_outer_quality)
    with pytest.raises(audio.AudioVerificationIncomplete):
        video_audio.remove_video_audio_all_streams(source, existing, scheme="auto")
    assert existing.read_bytes() == sentinel

    calls = 0
    with pytest.raises(audio.AudioVerificationIncomplete):
        video_audio.remove_video_audio_all_streams(source, fresh, scheme="auto")
    assert not fresh.exists()

    # 2. Single-track late source-hash mutation check must not leave a published replacement behind
    monkeypatch.setattr(video_audio, "audio_quality_passed", orig_quality)
    orig_identify = audio.identify_audio

    def identify_only_track0(src, *, check_invisible=True, stream=0, scheme="audioseal"):
        rep = orig_identify(src, check_invisible=check_invisible, stream=stream, scheme=scheme)
        if Path(src).resolve() == source.resolve() and stream == 1:
            return replace(
                rep,
                watermark_status="not_detected",
                detections=tuple(replace(d, status="not_detected") for d in rep.detections),
            )
        return rep

    monkeypatch.setattr(audio, "identify_audio", identify_only_track0)
    orig_sha = video_audio.sha256_file
    sha_calls = 0

    def mutated_on_second_check(p):
        nonlocal sha_calls
        if Path(p).resolve() == source.resolve():
            sha_calls += 1
            if sha_calls >= 2:
                return "0" * 64
        return orig_sha(p)

    monkeypatch.setattr(video_audio, "sha256_file", mutated_on_second_check)
    with pytest.raises(RuntimeError, match="changed during"):
        video_audio.remove_video_audio_all_streams(source, existing, scheme="auto")
    assert existing.read_bytes() == sentinel


def test_remove_video_audio_all_streams_unverified_unprocessed_track_blocks_aggregate_verified(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)

    source = _video(tmp_path, first_offset=0, selected_offset=0)
    output = tmp_path / "partial_unverified.mkv"
    orig_identify = audio.identify_audio

    def identify_with_unverified_track1(src, *, check_invisible=True, stream=0, scheme="audioseal"):
        rep = orig_identify(src, check_invisible=check_invisible, stream=stream, scheme=scheme)
        if stream == 1:
            return replace(
                rep,
                watermark_status="unverified",
                detections=tuple(replace(d, status="unverified", whole_clip_error="boom") for d in rep.detections),
            )
        return rep

    monkeypatch.setattr(audio, "identify_audio", identify_with_unverified_track1)
    res = video_audio.remove_video_audio_all_streams(source, output, scheme="auto")
    assert res.processed_ordinals == (0,)
    assert res.verified_removed is None
    assert res.streams[0].verified_removed is True
    assert res.streams[1].skipped is True
    for item in res.streams:
        assert item.source == source.resolve()
        assert item.output == output
        assert item.before.source == source.resolve()
        if item.after is not None:
            assert item.after.source == output


def test_preflight_untagged_stereo_post_fidelity_hash_and_selected_stream_ordinals(tmp_path, monkeypatch):
    from pathlib import Path

    from remove_ai_watermarks import audio, video_audio

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)

    # 1. Forced preflight accepts untagged 2-channel PCM stereo in MKV (where raw channel_layout is omitted)
    source = _video(tmp_path, first_offset=0, selected_offset=0)
    monkeypatch.setattr(audio.audio_dac, "is_available", lambda: True)
    orig_inventory = video_audio.stream_inventory
    monkeypatch.setattr(
        video_audio,
        "stream_inventory",
        lambda p: [{k: v for k, v in s.items() if k != "channel_layout"} for s in orig_inventory(p)],
    )
    video_audio.preflight_video_audio_workload(source, force=True, method="dac", scheme="auto")
    monkeypatch.setattr(video_audio, "stream_inventory", orig_inventory)

    # 2. Multi-track source mutation during final measure_fidelity must abort BEFORE publishing destination
    existing = tmp_path / "sentinel.mkv"
    existing.write_bytes(b"SENTINEL-MUST-SURVIVE-FINAL-FIDELITY-SOURCE-MUTATION")
    orig_measure = video_audio.measure_fidelity
    orig_sha = video_audio.sha256_file
    mutated = False

    def measure_then_mutate(src, info, out, *, stream=0):
        nonlocal mutated
        res_fid = orig_measure(src, info, out, stream=stream)
        mutated = True
        return res_fid

    monkeypatch.setattr(video_audio, "measure_fidelity", measure_then_mutate)
    monkeypatch.setattr(
        video_audio,
        "sha256_file",
        lambda p: "f" * 64 if mutated and Path(p).resolve() == source.resolve() else orig_sha(p),
    )
    with pytest.raises(RuntimeError, match="changed during"):
        video_audio.remove_video_audio_all_streams(source, existing, scheme="auto")
    assert existing.read_bytes() == b"SENTINEL-MUST-SURVIVE-FINAL-FIDELITY-SOURCE-MUTATION"
    monkeypatch.setattr(video_audio, "measure_fidelity", orig_measure)
    monkeypatch.setattr(video_audio, "sha256_file", orig_sha)

    # 3. Explicit streams=(0,) with allow_unverified=True processes ONLY track 0 and leaves track 1 untouched
    monkeypatch.setattr(audio.audioseal, "read", lambda _m, _s: audio.audioseal.AudioSealReading(0.0, 0.0, (0,) * 16))
    selective_out = tmp_path / "selective_track0.mkv"
    res_sel = video_audio.remove_video_audio_all_streams(
        source,
        selective_out,
        scheme="auto",
        method="spectral",
        allow_unverified=True,
        streams=(0,),
    )
    assert res_sel.processed_ordinals == (0,)
    assert res_sel.streams[0].skipped is False
    assert res_sel.streams[1].skipped is True


def test_video_audio_all_streams_preserves_prestrip_vendor_metadata(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio, video_audio

    source = _video(tmp_path, container=".mp4", audio_codec="aac", first_offset=0, selected_offset=0)
    monkeypatch.setattr(audio.audioseal, "is_available", lambda: False)
    monkeypatch.setattr(audio.wavmark, "is_available", lambda: False)
    monkeypatch.setattr(audio.perth, "is_available", lambda: False)
    monkeypatch.setattr(audio.metadata, "get_ai_metadata", lambda _p: {})
    monkeypatch.setattr(audio.audio_dac, "is_available", lambda: True)
    dac_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        audio.audio_dac,
        "resynthesize",
        lambda samples, *_a, **_k: dac_calls.append(samples.shape) or (samples * 0.9),
    )
    monkeypatch.setattr(
        audio.audio_spectral,
        "filter_audio_chunks",
        lambda *_a, **_k: pytest.fail("spectral must not run when pre-strip Google SynthID metadata is supplied"),
    )

    out = tmp_path / "clean_veo_stage.mp4"
    res = video_audio.remove_video_audio_all_streams(
        source,
        out,
        verify=False,
        method="auto",
        scheme="auto",
        allow_unverified=True,
        _metadata_markers={
            "issuer": "Google LLC",
            "synthid_watermark": "present according to Google LLC provenance",
        },
    )
    assert res.skipped is False
    assert [s.method for s in res.streams] == ["dac", "dac"]
    assert len(dac_calls) == 2

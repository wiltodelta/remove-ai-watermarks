"""Public audio routing, source validation, and fail-closed publication."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")


def _source(tmp_path, *, frames=16000, rate=16000):
    path = tmp_path / "source.wav"
    t = np.arange(frames) / rate
    left = 0.2 * np.sin(2 * np.pi * 500 * t) + 0.1 * np.sin(2 * np.pi * 1250 * t)
    right = 0.2 * np.sin(2 * np.pi * 3000 * t) + 0.1 * np.sin(2 * np.pi * 1250 * t)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.column_stack((left, right)) * 32767).astype("<i2").tobytes())
    return path


def _spectral_detector(monkeypatch, *, resistant=False):
    from remove_ai_watermarks._internal import audioseal

    monkeypatch.setattr(audioseal, "is_available", lambda: True)
    monkeypatch.setattr(audioseal, "load_pinned_detector", object)

    def read(_model, samples):
        t = np.arange(len(samples)) / 16000
        amplitude = abs(2 * np.mean(samples * np.exp(-2j * np.pi * 1250 * t)))
        found = resistant or amplitude > 0.01
        return audioseal.AudioSealReading(float(found), float(found), (0, 1) * 8)

    monkeypatch.setattr(audioseal, "read", read)
    if resistant:
        from remove_ai_watermarks._internal import audio_codec

        monkeypatch.setattr(audio_codec, "is_available", lambda: False)


def test_missing_optional_decoder_reports_unverified(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audioseal, "is_available", lambda: False)
    report = audio.identify_audio(_source(tmp_path))
    assert report.watermark_status == "unverified"
    assert report.detections == ()
    assert report.stream.channels == 2


def test_bigvgan_delivery_routes_full_track_and_preserves_unverified_scope(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []

    def process(samples, sample_rate, *, device, local_files_only):
        calls.append((samples.shape, sample_rate, device, local_files_only))
        return samples * 0.5

    monkeypatch.setattr(audio.audio_vocoder, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_vocoder, "resynthesize", process)
    result = audio.remove_audio_invisible(
        _source(tmp_path, frames=16001),
        tmp_path / "vocoder.wav",
        method="bigvgan",
        vocoder_device="mps",
        force=True,
        verify=False,
    )
    assert calls == [((16001, 2), 16000, "mps", False)]
    assert result.method == "bigvgan"
    assert result.bandwidth_kbps is None
    assert result.verified_removed is None
    assert result.fidelity.frames == 16001
    assert result.fidelity.snr_db == pytest.approx(6.0206, abs=0.001)

    assert b"Lavf" not in result.output.read_bytes()


def test_bigvgan_still_positive_does_not_replace_output(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch, resistant=True)
    monkeypatch.setattr(audio.audio_vocoder, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_vocoder, "resynthesize", lambda samples, *a, **kw: samples * 0.5)
    output = tmp_path / "protected.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        audio.remove_audio_invisible(_source(tmp_path), output, method="bigvgan")
    assert output.read_bytes() == b"previous output"


@pytest.mark.parametrize("channels", [1, 2])
def test_bigvgan_full_track_limit_stops_and_closes_before_model(monkeypatch, channels):
    from remove_ai_watermarks import audio

    closed = []
    monkeypatch.setattr(audio.audio_vocoder, "resynthesize", lambda *a, **kw: pytest.fail("model must not run"))

    def chunks():
        try:
            yield np.zeros((299, channels), dtype=np.float32)
            yield np.zeros((2, channels), dtype=np.float32)
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="30 seconds"):
        list(audio._vocode_full_track(chunks(), 10, "cpu"))
    assert closed == [True]


def test_bigvgan_rejects_two_channels_that_are_not_a_stereo_pair(tmp_path, monkeypatch):
    from dataclasses import replace

    from remove_ai_watermarks import audio

    source = _source(tmp_path)
    before = audio.identify_audio(source, check_invisible=False)
    before = replace(before, stream=replace(before.stream, channel_layout="2 channels (FC+LFE)"))
    monkeypatch.setattr(audio, "identify_audio", lambda *a, **kw: before)
    monkeypatch.setattr(audio.audio_vocoder, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_vocoder, "resynthesize", lambda *a, **kw: pytest.fail("model must not run"))
    output = tmp_path / "protected.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="mono or stereo channel layout"):
        audio.remove_audio_invisible(source, output, method="bigvgan", force=True, verify=False)
    assert output.read_bytes() == b"previous output"


def test_bigvgan_missing_base_stack_names_the_complete_processing_extra(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audio_io, "is_available", lambda: False)
    with pytest.raises(ImportError, match="audio-vocoder"):
        audio.remove_audio_invisible(
            _source(tmp_path), tmp_path / "out.wav", method="bigvgan", force=True, verify=False
        )


def test_bigvgan_missing_stack_preserves_existing_output(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audio_vocoder, "is_available", lambda: False)
    output = tmp_path / "protected.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(ImportError, match="audio-vocoder"):
        audio.remove_audio_invisible(_source(tmp_path), output, method="bigvgan", force=True, verify=False)
    assert output.read_bytes() == b"previous output"


def test_codec_method_uses_requested_bandwidth_and_preserves_delivery(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []

    def process(chunks, sample_rate, *, bandwidth_kbps, stereo_mode="independent"):
        calls.append((sample_rate, bandwidth_kbps))
        for chunk in chunks:
            yield chunk * 0.5

    monkeypatch.setattr(audio.audio_codec, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_codec, "resynthesize_chunks", process)
    result = audio.remove_audio_invisible(
        _source(tmp_path), tmp_path / "processed.wav", force=True, verify=False, method="encodec", bandwidth_kbps=12
    )
    assert calls == [(16000, 12)]
    assert result.method == "encodec"
    assert result.bandwidth_kbps == 12
    assert result.fidelity.frames == 16000
    assert result.fidelity.snr_db == pytest.approx(6.0206, abs=0.001)


@pytest.mark.parametrize(
    ("scheme", "rate", "seconds"), [("wavmark", 16000, 1), ("perth", 16000, 1), ("silentcipher", 44100, 3)]
)
def test_framed_model_routing_checks_the_complete_short_tail(tmp_path, monkeypatch, scheme, rate, seconds):
    from remove_ai_watermarks import audio

    lengths = []

    def read(model, samples, **kwargs):
        if scheme == "silentcipher":
            assert kwargs == {"sample_rate": rate}
        lengths.append(len(samples))
        if len(samples) < seconds * rate:
            raise ValueError("Incomplete model window")
        t = np.arange(len(samples)) / rate
        found = abs(2 * np.mean(samples * np.exp(-2j * np.pi * 1250 * t))) > 0.01
        if scheme == "silentcipher":
            return audio.silentcipher.SilentCipherReading(
                (1, 2, 3, 4, 5) if found else None, float(found), 0 if found else None, 205, "cpu"
            )
        return (
            audio.perth.PerthReading(float(found))
            if scheme == "perth"
            else audio.wavmark.WavMarkReading((0,) * 16 if found else None, (0,) if found else ())
        )

    monkeypatch.setattr(getattr(audio, scheme), "is_available", lambda: True)
    monkeypatch.setattr(getattr(audio, scheme), "load_model", object)
    monkeypatch.setattr(getattr(audio, scheme), "read", read)
    if scheme == "wavmark":
        negative = audio.wavmark.WavMarkSignatureReading((0,), (0,) * 32, "64656661", 18, 1.0)
        monkeypatch.setattr(audio.wavmark, "read_openvoice_default", lambda *_: (negative, (negative,)))
    result = audio.remove_audio_invisible(
        _source(tmp_path, frames=seconds * rate + 31, rate=rate),
        tmp_path / "processed.wav",
        scheme=scheme,
        method="spectral",
    )
    assert result.verified_removed is True
    assert result.verification_scope == (scheme,)
    assert min(lengths) == seconds * rate
    assert result.after.detections[0].windows[-1].start_frame == 31
    assert result.after.detections[0].windows[-1].end_frame == seconds * rate + 31


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_silentcipher_reports_model_clock_separately_from_source_clock(tmp_path, monkeypatch, rate):
    from remove_ai_watermarks import audio

    calls = []

    def read(_model, samples, *, sample_rate):
        calls.append((len(samples), sample_rate))
        return audio.silentcipher.SilentCipherReading(None, 0.0, None, 205, "cpu")

    monkeypatch.setattr(audio.silentcipher, "is_available", lambda: True)
    monkeypatch.setattr(audio.silentcipher, "load_model", object)
    monkeypatch.setattr(audio.silentcipher, "read", read)
    report = audio.identify_audio(_source(tmp_path, frames=3 * rate + 31, rate=rate), scheme="silentcipher")

    assert report.stream.sample_rate == rate
    assert report.watermark_status == "not_detected"
    assert len(report.detections) == 3
    for detection in report.detections:
        assert detection.decoder_sample_rate == 44100
        assert detection.windows[-1].start_frame == 31
        assert detection.windows[-1].end_frame == 3 * rate + 31
    assert {sample_rate for _, sample_rate in calls} == {rate}


def test_all_schemes_cannot_be_negative_with_an_unavailable_decoder(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.wavmark, "is_available", lambda: False)
    monkeypatch.setattr(audio.perth, "is_available", lambda: False)
    monkeypatch.setattr(audio.silentcipher, "is_available", lambda: False)
    monkeypatch.setattr(audio.silentcipher16, "is_available", lambda: False)
    output = tmp_path / "processed.wav"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete) as caught:
        audio.remove_audio_invisible(_source(tmp_path), output, scheme="all", method="spectral")
    assert caught.value.after.watermark_status == "unverified"
    assert [d.scheme for d in caught.value.after.detections] == [
        "audioseal",
        "audioseal",
        "audioseal",
        "wavmark",
        "perth",
        "silentcipher",
        "silentcipher16",
    ]
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_silentcipher16_delivery_checks_native_windows_and_reports_ternary_symbols(tmp_path, monkeypatch, rate):
    from remove_ai_watermarks import audio
    from remove_ai_watermarks._internal import silentcipher16

    lengths = []

    def read(_model, samples, *, sample_rate):
        assert sample_rate == rate
        lengths.append(len(samples))
        assert len(samples) >= 3 * rate
        t = np.arange(len(samples)) / rate
        found = abs(2 * np.mean(samples * np.exp(-2j * np.pi * 1250 * t))) > 0.01
        return silentcipher16.SilentCipher16Reading(
            (0, 1, 2, 0, 2, 1, 0) if found else None, float(found), 0, 103, "cpu"
        )

    monkeypatch.setattr(silentcipher16, "is_available", lambda: True)
    monkeypatch.setattr(silentcipher16, "load_model", object)
    monkeypatch.setattr(silentcipher16, "read", read)
    result = audio.remove_audio_invisible(
        _source(tmp_path, frames=3 * rate + 31, rate=rate),
        tmp_path / "processed.wav",
        scheme="silentcipher16",
        method="spectral",
    )
    assert result.verified_removed is True
    assert result.verification_scope == ("silentcipher16",)
    assert result.before.detections[0].whole_clip.payload_symbols == (0, 1, 2, 0, 2, 1, 0)
    assert result.before.detections[0].whole_clip.framing == "terminated8"
    assert result.after.stream.sample_rate == rate
    assert result.fidelity.frames == 3 * rate + 31
    for detection in result.after.detections:
        assert detection.whole_clip.framing == "terminated8"
        assert detection.scheme == "silentcipher16"
        assert detection.decoder_sample_rate == 16000
        assert detection.windows[-1].start_frame == 31
        assert detection.windows[-1].end_frame == 3 * rate + 31
    assert min(lengths) == 3 * rate


def test_all_requires_silentcipher16_even_when_every_other_receiver_is_negative(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    original_scan = audio._scan_audio

    def scan(source, info, scheme):
        if scheme == "audioseal":
            return original_scan(source, info, scheme)
        reading = {
            "wavmark": audio.wavmark.WavMarkReading(None, ()),
            "perth": audio.perth.PerthReading(0.0),
            "silentcipher": audio.silentcipher.SilentCipherReading(None, 0.4, None, 205, "cpu"),
        }[scheme]
        window = audio.AudioWindowReading(0, 3 * info.sample_rate, reading)
        return (audio.AudioDetection(scheme, 0, "not_detected", reading, (window,)),), ()

    monkeypatch.setattr(audio, "_scan_audio", scan)
    for backend in [audio.wavmark, audio.perth, audio.silentcipher]:
        monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(audio.silentcipher16, "is_available", lambda: False)
    output = tmp_path / "processed.wav"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete) as caught:
        audio.remove_audio_invisible(_source(tmp_path, frames=48000), output, scheme="all", method="spectral")
    assert caught.value.after.watermark_status == "unverified"
    assert caught.value.after.detections[-1].scheme == "silentcipher16"
    assert caught.value.after.detections[-1].status == "unverified"
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("force", [False, True])
def test_auto_falls_back_without_publishing_failed_candidates(tmp_path, monkeypatch, force):
    from remove_ai_watermarks import audio

    source, output = _source(tmp_path), tmp_path / "processed.wav"
    output.write_bytes(b"existing output")
    monkeypatch.setattr(audio.audioseal, "is_available", lambda: True)
    monkeypatch.setattr(audio.audioseal, "load_pinned_detector", object)
    monkeypatch.setattr(
        audio.audioseal,
        "read",
        lambda model, samples: audio.audioseal.AudioSealReading(
            float(np.mean(samples**2) > 0.001), float(np.mean(samples**2) > 0.001), (0,) * 16
        ),
    )
    monkeypatch.setattr(audio.audio_codec, "is_available", lambda: True)
    calls, inspections = [], []
    inspect = audio.identify_audio

    def identify(path, **kwargs):
        inspections.append(Path(path))
        return inspect(path, **kwargs)

    def resynthesize(chunks, sample_rate, *, bandwidth_kbps, stereo_mode="independent"):
        assert output.read_bytes() == b"existing output"
        calls.append((sample_rate, bandwidth_kbps))
        for chunk in chunks:
            yield chunk * 0.1

    monkeypatch.setattr(audio, "identify_audio", identify)
    monkeypatch.setattr(audio.audio_codec, "resynthesize_chunks", resynthesize)
    result = audio.remove_audio_invisible(source, output, force=force)
    assert result.verified_removed is True
    assert result.method == "combined"
    assert calls == [(16000, 24)]
    assert inspections.count(source) == 1


def test_default_without_a_source_positive_writes_nothing(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audioseal, "is_available", lambda: False)
    output = tmp_path / "output.wav"
    result = audio.remove_audio_invisible(_source(tmp_path), output)
    assert result.skipped is True
    assert result.output is None
    assert not output.exists()


def test_forced_processing_keeps_geometry_and_original_file(tmp_path):
    from remove_ai_watermarks._internal.audio_io import probe_audio, read_audio
    from remove_ai_watermarks.audio import remove_audio_invisible

    source = _source(tmp_path, frames=16031)
    original = source.read_bytes()
    output = tmp_path / "output.wav"
    result = remove_audio_invisible(source, output, force=True, verify=False)
    assert result.output == output
    assert result.frames_processed == 16031
    assert result.verified_removed is None
    delivered = probe_audio(output)
    assert (delivered.channels, delivered.sample_rate) == (2, 16000)
    samples = np.concatenate(list(read_audio(output, delivered)))
    assert samples.shape == (16031, 2)
    assert source.read_bytes() == original
    section = samples[1600:-1600, 0]
    t = np.arange(len(section)) / 16000
    assert abs(2 * np.mean(section * np.exp(-2j * np.pi * 1250 * t))) < 0.001
    assert abs(2 * np.mean(section * np.exp(-2j * np.pi * 500 * t))) == pytest.approx(0.2, abs=0.003)


def test_processing_cannot_replace_the_source(tmp_path):
    from remove_ai_watermarks.audio import remove_audio_invisible

    source = _source(tmp_path)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="differ from the source"):
        remove_audio_invisible(source, source, force=True, verify=False)
    assert source.read_bytes() == original


def test_verification_reads_the_delivered_candidate(tmp_path, monkeypatch):
    from remove_ai_watermarks.audio import remove_audio_invisible

    _spectral_detector(monkeypatch)
    output = tmp_path / "output.wav"
    result = remove_audio_invisible(_source(tmp_path), output)
    assert result.before.watermark_status == "detected"
    assert result.after.watermark_status == "not_detected"
    assert result.verified_removed is True
    assert output.is_file()


def test_resistant_watermark_does_not_replace_existing_output(tmp_path, monkeypatch):
    from remove_ai_watermarks.audio import AudioVerificationIncomplete, remove_audio_invisible

    _spectral_detector(monkeypatch, resistant=True)
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")
    with pytest.raises(AudioVerificationIncomplete):
        remove_audio_invisible(_source(tmp_path), output)
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".output-*"))


def test_force_retains_a_failed_candidate_without_calling_it_removed(tmp_path, monkeypatch):
    from remove_ai_watermarks.audio import remove_audio_invisible

    _spectral_detector(monkeypatch, resistant=True)
    output = tmp_path / "output.wav"
    result = remove_audio_invisible(_source(tmp_path), output, force=True, method="spectral")
    assert result.output == output
    assert result.verified_removed is False
    assert result.after.watermark_status == "detected"


def test_truncated_audio_is_not_a_no_signal_skip(tmp_path):
    from remove_ai_watermarks.audio import remove_audio_invisible

    source = _source(tmp_path)
    source.write_bytes(source.read_bytes()[:-4])
    output = tmp_path / "output.wav"
    with pytest.raises(ValueError, match="cannot read"):
        remove_audio_invisible(source, output, force=True, verify=False)
    assert not output.exists()


def test_disabling_verification_requires_explicit_force(tmp_path):
    from remove_ai_watermarks.audio import remove_audio_invisible

    with pytest.raises(ValueError, match="force"):
        remove_audio_invisible(_source(tmp_path), tmp_path / "output.wav", verify=False)


def test_source_change_during_processing_prevents_publication(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    source = _source(tmp_path)
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")
    original_filter = audio.audio_spectral.filter_audio_chunks

    def changing_input(chunks, rate):
        for frames in original_filter(chunks, rate):
            data = bytearray(source.read_bytes())
            data[-1] ^= 1
            source.write_bytes(data)
            yield frames

    monkeypatch.setattr(audio.audio_spectral, "filter_audio_chunks", changing_input)
    with pytest.raises(RuntimeError, match="source changed"):
        audio.remove_audio_invisible(source, output)
    assert output.read_bytes() == b"existing output"


def test_aiff_input_is_accepted_by_its_audio_probe(tmp_path):
    import subprocess

    from remove_ai_watermarks.audio import identify_audio

    source = _source(tmp_path)
    aiff = tmp_path / "input.aiff"
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    subprocess.run([ffmpeg, "-v", "error", "-i", str(source), str(aiff)], check=True)  # noqa: S603
    report = identify_audio(aiff, check_invisible=False)
    assert report.stream.format_name == "aiff"
    assert report.stream.channels == 2


def test_short_silentcipher_source_stays_unverified_and_preserves_output(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.silentcipher, "is_available", lambda: True)
    monkeypatch.setattr(audio.silentcipher, "load_model", object)
    source = _source(tmp_path, frames=32000)
    report = audio.identify_audio(source, scheme="silentcipher")
    assert report.watermark_status == "unverified"
    assert all(d.whole_clip_error for d in report.detections)
    assert all(w.error for d in report.detections for w in d.windows)
    output = tmp_path / "existing.wav"
    output.write_bytes(b"existing")
    result = audio.remove_audio_invisible(source, output, scheme="silentcipher")
    assert result.skipped is True
    assert result.verified_removed is None
    assert output.read_bytes() == b"existing"


def test_mimi_delivery_routes_full_track_and_preserves_unverified_scope(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []

    def process(samples, sample_rate, *, local_files_only):
        calls.append((samples.shape, sample_rate, "cpu", local_files_only))
        return samples * 0.5

    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", process)
    result = audio.remove_audio_invisible(
        _source(tmp_path, frames=16001),
        tmp_path / "vocoder.wav",
        method="mimi",
        vocoder_device="cpu",
        force=True,
        verify=False,
    )
    assert calls == [((16001, 2), 16000, "cpu", False)]
    assert result.method == "mimi"
    assert result.bandwidth_kbps is None
    assert result.verified_removed is None
    assert result.fidelity.frames == 16001
    assert result.fidelity.snr_db == pytest.approx(6.0206, abs=0.001)

    assert b"Lavf" not in result.output.read_bytes()


def test_mimi_still_positive_does_not_replace_output(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch, resistant=True)
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda samples, *a, **kw: samples * 0.5)
    output = tmp_path / "protected.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        audio.remove_audio_invisible(_source(tmp_path), output, method="mimi")
    assert output.read_bytes() == b"previous output"


def test_mimi_rejects_two_channels_that_are_not_a_stereo_pair(tmp_path, monkeypatch):
    from dataclasses import replace

    from remove_ai_watermarks import audio

    source = _source(tmp_path)
    before = audio.identify_audio(source, check_invisible=False)
    before = replace(before, stream=replace(before.stream, channel_layout="2 channels (FC+LFE)"))
    monkeypatch.setattr(audio, "identify_audio", lambda *a, **kw: before)
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda *a, **kw: pytest.fail("model must not run"))
    output = tmp_path / "protected.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="mono or stereo channel layout"):
        audio.remove_audio_invisible(source, output, method="mimi", force=True, verify=False)
    assert output.read_bytes() == b"previous output"


def test_mimi_missing_base_stack_names_the_complete_processing_extra(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audio_io, "is_available", lambda: False)
    with pytest.raises(ImportError, match="audio-codec"):
        audio.remove_audio_invisible(_source(tmp_path), tmp_path / "out.wav", method="mimi", force=True, verify=False)


def test_mimi_missing_stack_preserves_existing_output(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: False)
    output = tmp_path / "protected.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(ImportError, match="audio-codec"):
        audio.remove_audio_invisible(_source(tmp_path), output, method="mimi", force=True, verify=False)
    assert output.read_bytes() == b"previous output"


@pytest.mark.parametrize("limit", ["seconds", "values"])
def test_mimi_full_track_limit_closes_before_model(monkeypatch, limit):
    from remove_ai_watermarks import audio

    closed = []
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda *a, **k: pytest.fail("model must not run"))
    if limit == "values":
        monkeypatch.setattr(audio, "_MAX_WHOLE_VALUES", 4)

    def chunks():
        try:
            yield np.zeros((301 if limit == "seconds" else 3, 2), dtype=np.float32)
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="30 seconds and 16000000"):
        list(audio._mimi_full_track(chunks(), 10))
    assert closed == [True]


def test_mimi_is_explicit_cpu_only_and_not_in_auto(tmp_path):
    from remove_ai_watermarks import audio

    assert all("mimi" not in profiles for profiles in audio._AUTO_PROFILES.values())
    assert all("focalcodec" not in profiles for profiles in audio._AUTO_PROFILES.values())
    with pytest.raises(ValueError, match="requires method=bigvgan"):
        audio.remove_audio_invisible(_source(tmp_path), tmp_path / "out.wav", method="mimi", vocoder_device="mps")
    with pytest.raises(ValueError, match="requires method=bigvgan"):
        audio.remove_audio_invisible(_source(tmp_path), tmp_path / "out.wav", method="focalcodec", vocoder_device="mps")


def test_focalcodec_delivery_and_still_positive_safeguard(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    calls = []

    def process(samples, sample_rate, *, local_files_only):
        calls.append((samples.shape, sample_rate, local_files_only))
        return samples * 0.5

    monkeypatch.setattr(audio.audio_focalcodec, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_focalcodec, "resynthesize", process)
    result = audio.remove_audio_invisible(
        _source(tmp_path, frames=16001),
        tmp_path / "focal.wav",
        method="focalcodec",
        vocoder_device="cpu",
        force=True,
        verify=False,
    )
    assert calls == [((16001, 2), 16000, False)]
    assert result.method == "focalcodec"
    assert result.bandwidth_kbps is None
    assert result.verified_removed is None
    assert result.fidelity.frames == 16001
    assert b"Lavf" not in result.output.read_bytes()

    _spectral_detector(monkeypatch, resistant=True)
    protected = tmp_path / "protected_focal.wav"
    protected.write_bytes(b"previous output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        audio.remove_audio_invisible(_source(tmp_path), protected, method="focalcodec")
    assert protected.read_bytes() == b"previous output"


def test_focalcodec_missing_stack_and_channel_layout_guards(tmp_path, monkeypatch):
    from dataclasses import replace

    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audio_focalcodec, "is_available", lambda: False)
    with pytest.raises(ImportError, match="audio-codec"):
        audio.remove_audio_invisible(
            _source(tmp_path), tmp_path / "out.wav", method="focalcodec", force=True, verify=False
        )

    source = _source(tmp_path)
    before = audio.identify_audio(source, check_invisible=False)
    before = replace(before, stream=replace(before.stream, channel_layout="2 channels (FC+LFE)"))
    monkeypatch.setattr(audio, "identify_audio", lambda *a, **kw: before)
    monkeypatch.setattr(audio.audio_focalcodec, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_focalcodec, "resynthesize", lambda *a, **kw: pytest.fail("model must not run"))
    with pytest.raises(ValueError, match="FocalCodec requires a mono or stereo channel layout"):
        audio.remove_audio_invisible(source, tmp_path / "out.wav", method="focalcodec", force=True, verify=False)


def test_mimi_source_change_rejected_after_resynthesis(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    source = _source(tmp_path)
    output = tmp_path / "protected.wav"
    output.write_bytes(b"existing output")
    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)

    def process(samples, *a, **kw):
        data = bytearray(source.read_bytes())
        data[-1] ^= 1
        source.write_bytes(data)
        return samples

    monkeypatch.setattr(audio.audio_mimi, "resynthesize", process)
    with pytest.raises(RuntimeError, match="source changed"):
        audio.remove_audio_invisible(source, output, method="mimi", force=True, verify=False)
    assert output.read_bytes() == b"existing output"


def test_mimi_flac_uses_separate_non_bitexact_encoding(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda samples, *a, **k: samples)
    real = audio.audio_io.write_audio
    calls = []

    def write(info, output, chunks, **kwargs):
        calls.append((output.suffix, kwargs["bitexact"]))
        return real(info, output, chunks, **kwargs)

    monkeypatch.setattr(audio.audio_io, "write_audio", write)
    result = audio.remove_audio_invisible(
        _source(tmp_path), tmp_path / "mimi.flac", method="mimi", force=True, verify=False
    )
    assert calls == [(".flac", False)]
    assert result.frames_processed == 16000
    assert result.verified_removed is None
    assert result.after.stream.codec == "flac"


def test_long_audio_exceeding_thirty_seconds_verifies_across_bounded_segments(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    source = _source(tmp_path, frames=31 * 16000, rate=16000)
    output = tmp_path / "long_clean.wav"
    result = audio.remove_audio_invisible(source, output, method="spectral")
    assert result.verified_removed is True
    assert result.quality_passed is True
    assert result.after is not None
    assert result.after.watermark_status == "not_detected"
    for detection in result.after.detections:
        assert len(detection.whole_segments) == 2
        assert detection.whole_segments[0].start_frame == 0
        assert detection.whole_segments[0].end_frame == 30 * 16000
        assert detection.whole_segments[1].start_frame == 29 * 16000
        assert detection.whole_segments[1].end_frame == 31 * 16000

    # A multi-second mark straddling the 30 s boundary must be caught by the overlapping segment read.
    def boundary_only_read(_model, samples):
        # Individual 1 s windows are length 16000; only segment reads (> 16000) spanning [29 s, 31 s) see both halves.
        found = len(samples) == 2 * 16000
        return audio.audioseal.AudioSealReading(float(found), float(found), (0, 1) * 8)

    monkeypatch.setattr(audio.audioseal, "read", boundary_only_read)
    boundary_report = audio.identify_audio(output)
    assert boundary_report.watermark_status == "detected"


def test_audio_quality_gate_rejects_silent_erasure_and_clipping_unless_forced(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    source = _source(tmp_path)
    protected = tmp_path / "protected.wav"
    protected.write_bytes(b"existing output")

    monkeypatch.setattr(audio.audio_mimi, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_mimi, "resynthesize", lambda samples, *a, **k: np.zeros_like(samples))
    with pytest.raises(audio.AudioVerificationIncomplete, match="quality gate"):
        audio.remove_audio_invisible(source, protected, method="mimi")
    assert protected.read_bytes() == b"existing output"

    forced_silent = audio.remove_audio_invisible(source, tmp_path / "forced_silent.wav", method="mimi", force=True)
    assert forced_silent.verified_removed is True
    assert forced_silent.quality_passed is False

    # Runaway gain / full-scale overload on float32 WAV is also rejected unless forced.
    from dataclasses import replace

    float_source = tmp_path / "float_source.wav"
    src_info = audio.audio_io.probe_audio(source)
    src_pcm = np.concatenate(list(audio.audio_io.read_audio(source, src_info)), axis=0)
    audio.audio_io.write_audio(
        replace(src_info, codec="pcm_f32le"),
        float_source,
        [src_pcm],
        expected_frames=src_pcm.shape[0],
    )
    monkeypatch.setattr(
        audio.audio_mimi,
        "resynthesize",
        lambda samples, rate, *a, **k: next(audio.audio_spectral.filter_audio_chunks(iter([samples]), rate)) * 8.0,
    )
    with pytest.raises(audio.AudioVerificationIncomplete, match="quality gate"):
        audio.remove_audio_invisible(float_source, protected, method="mimi")
    assert protected.read_bytes() == b"existing output"


def test_missing_decoder_under_all_names_install_extra(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)
    monkeypatch.setattr(audio.perth, "is_available", lambda: False)
    with pytest.raises(audio.AudioVerificationIncomplete, match=r"remove-ai-watermarks\[audio-perth\]"):
        audio.remove_audio_invisible(_source(tmp_path), tmp_path / "out.wav", scheme="all", method="spectral")


def test_scheme_auto_scans_installed_decoders_and_selects_adaptive_minimal_profile(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)

    # Unlike scheme="all" (which marks missing optional decoders as unverified),
    # scheme="auto" checks only installed compatible decoders and adaptively selects "spectral"
    # (never loading EnCodec) when only AudioSeal/Perth detected a mark.
    monkeypatch.setattr(
        audio.audio_codec,
        "resynthesize_chunks",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("EnCodec must not run when spectral suffices")),
    )
    source = _source(tmp_path)
    report = audio.identify_audio(source, scheme="auto")
    assert report.watermark_status == "detected"
    assert {d.scheme for d in report.detections} == {"audioseal"}

    out_m4a = tmp_path / "clean.m4a"
    res = audio.remove_audio_invisible(source, out_m4a, scheme="auto", method="auto")
    assert res.skipped is False
    assert res.verified_removed is True
    assert res.method == "spectral"
    assert res.quality_passed is True
    assert out_m4a.is_file()


def test_stereo_encodec_routes_mid_side_mode(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.audio_codec, "is_available", lambda: True)
    seen_modes = []

    def fake_chunks(chunks, sample_rate, *, bandwidth_kbps=24.0, stereo_mode="independent"):
        seen_modes.append((sample_rate, bandwidth_kbps, stereo_mode))
        yield from audio.audio_spectral.filter_audio_chunks(chunks, sample_rate)

    monkeypatch.setattr(audio.audio_codec, "resynthesize_chunks", fake_chunks)
    stereo_src = _source(tmp_path)

    res = audio.remove_audio_invisible(stereo_src, tmp_path / "stereo_out.mp3", method="encodec")
    assert seen_modes == [(16000, 24.0, "ms")]
    assert res.verified_removed is True
    assert res.quality_passed is True


def test_audio_metadata_all_and_batch_suites(tmp_path, monkeypatch):
    import subprocess

    from remove_ai_watermarks import audio
    from remove_ai_watermarks._internal import audio_io

    _spectral_detector(monkeypatch)
    for backend in (audio.wavmark, audio.perth, audio.silentcipher, audio.silentcipher16):
        monkeypatch.setattr(backend, "is_available", lambda: False)

    marked_wav = _source(tmp_path)
    suno_comment = "Made with Suno; created=2026-05-01T12:00:00Z; id=12345678-1234-5678-9abc-123456789abc"
    suno_mp3 = tmp_path / "suno.mp3"
    subprocess.run(  # noqa: S603 - Fixed local fixture encoder, no shell.
        [
            audio_io.require_tool("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=24000:duration=1.5",
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

    meta_report = audio.inspect_audio_metadata(suno_mp3)
    assert meta_report.has_ai_metadata is True
    assert meta_report.markers.get("audio_generator") == "Suno"

    meta_res = audio.remove_audio_metadata(suno_mp3)
    assert meta_res.output == tmp_path / "suno_clean.mp3"
    assert meta_res.detected.get("audio_generator") == "Suno"
    assert meta_res.remaining == {}
    assert audio.inspect_audio_metadata(meta_res.output).has_ai_metadata is False

    # remove_audio_all with include_invisible=True on clean audio + AI metadata falls back to lossless metadata strip
    all_meta_only = audio.remove_audio_all(suno_mp3, tmp_path / "suno_all.mp3", include_invisible=True)
    assert all_meta_only.invisible_applied is False
    assert all_meta_only.remaining_metadata == {}
    assert all_meta_only.output.is_file()

    # remove_audio_all with include_invisible=True on watermarked audio applies invisible removal
    all_inv = audio.remove_audio_all(marked_wav, tmp_path / "marked_all.wav", include_invisible=True)
    assert all_inv.invisible_applied is True
    assert all_inv.invisible_result is not None
    assert all_inv.invisible_result.verified_removed is True
    assert all_inv.remaining_metadata == {}

    # remove_audio_batch across a directory with both files
    batch_in = tmp_path / "batch_in"
    batch_in.mkdir()
    (batch_in / "a_suno.mp3").write_bytes(suno_mp3.read_bytes())
    (batch_in / "b_marked.wav").write_bytes(marked_wav.read_bytes())
    summary = audio.remove_audio_batch(batch_in, include_invisible=True)
    assert summary.processed == 2
    assert summary.failed == 0
    assert summary.invisible_applied == 1
    assert summary.output_directory == tmp_path / "batch_in_clean"
    assert (summary.output_directory / "a_suno.mp3").is_file()
    assert (summary.output_directory / "b_marked.wav").is_file()


def test_auto_scheme_skips_experimental_silentcipher_and_short_clip_wavmark_unverified(tmp_path, monkeypatch):
    from remove_ai_watermarks import audio

    _spectral_detector(monkeypatch)
    monkeypatch.setattr(audio.wavmark, "is_available", lambda: True)
    monkeypatch.setattr(audio.wavmark, "load_model", object)
    monkeypatch.setattr(audio.perth, "is_available", lambda: False)
    monkeypatch.setattr(audio.silentcipher, "is_available", lambda: True)
    monkeypatch.setattr(audio.silentcipher16, "is_available", lambda: True)
    monkeypatch.setattr(
        audio.silentcipher,
        "load_model",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("experimental silentcipher must not run under auto")),
    )
    monkeypatch.setattr(
        audio.silentcipher16,
        "load_model",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("experimental silentcipher16 must not run under auto")),
    )

    short_wav = _source(tmp_path, frames=12800)  # 0.8s < 1.0s WavMark minimum window
    res = audio.remove_audio_invisible(short_wav, tmp_path / "short_clean.wav", scheme="auto")
    assert res.skipped is False
    assert res.verified_removed is True
    assert res.after is not None
    assert res.after.watermark_status == "not_detected"
    assert any("WavMark" in c or "one second" in c or "16000" in c for c in res.after.caveats)

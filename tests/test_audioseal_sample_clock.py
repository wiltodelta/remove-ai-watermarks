"""AudioSeal receivers retain marks embedded on a native sample clock."""

from __future__ import annotations

import shutil
import wave

import numpy as np
import pytest

from remove_ai_watermarks import audio
from remove_ai_watermarks._internal import audioseal

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")


def _carrier(tmp_path, rate):
    path = tmp_path / "native-clock.wav"
    samples = 0.2 * np.sin(2 * np.pi * np.arange(rate + 768) / 24)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


def _receiver(monkeypatch, *, fail_native=False):
    monkeypatch.setattr(audioseal, "is_available", lambda: True)
    monkeypatch.setattr(audioseal, "load_pinned_detector", object)

    def read(_model, samples):
        # A sample-clock code moves when resampled, even if physical pitch stays fixed.
        peak = np.argmax(abs(np.fft.rfft(samples))) / len(samples)
        matched = abs(peak - 1 / 24) < 0.001
        if matched and fail_native:
            raise RuntimeError("native receiver failed")
        return audioseal.AudioSealReading(float(matched), float(matched), (0,) * 16)

    monkeypatch.setattr(audioseal, "read", read)


@pytest.mark.parametrize("rate", [24000, 48000])
def test_native_clock_mark_survives_the_receiver_routing(tmp_path, monkeypatch, rate):
    _receiver(monkeypatch)
    report = audio.identify_audio(_carrier(tmp_path, rate))
    assert report.watermark_status == "detected"
    checks = {reading.decoder_sample_rate: reading for reading in report.detections}
    assert checks[16000].status == "not_detected"
    assert checks[rate].status == "detected"
    assert all(window.reading.detected for window in checks[rate].windows)
    assert checks[rate].windows[-1].end_frame == rate + 768


def test_native_receiver_failure_is_not_a_complete_negative(tmp_path, monkeypatch):
    _receiver(monkeypatch, fail_native=True)
    report = audio.identify_audio(_carrier(tmp_path, 24000))
    assert report.watermark_status == "unverified"
    checks = {reading.decoder_sample_rate: reading for reading in report.detections}
    assert checks[16000].status == "not_detected"
    assert checks[24000].status == "unverified"
    assert "native receiver failed" in checks[24000].whole_clip_error


def test_standard_clock_is_not_checked_twice(tmp_path, monkeypatch):
    _receiver(monkeypatch)
    report = audio.identify_audio(_carrier(tmp_path, 16000))
    assert report.watermark_status == "detected"
    assert len(report.detections) == 1
    assert report.detections[0].decoder_sample_rate == 16000


def test_publication_requires_the_native_receiver_to_be_negative(tmp_path, monkeypatch):
    _receiver(monkeypatch)
    source = _carrier(tmp_path, 24000)
    output = tmp_path / "existing.wav"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete) as caught:
        audio.remove_audio_invisible(source, output, method="spectral")
    assert caught.value.after.watermark_status == "detected"
    assert output.read_bytes() == b"existing output"


def test_short_tail_reuses_context_in_both_audioseal_receivers(tmp_path, monkeypatch):
    _receiver(monkeypatch)

    def short_input_false_positive(_model, samples):
        found = len(samples) < 1000
        return audioseal.AudioSealReading(float(found), float(found), (0,) * 16)

    monkeypatch.setattr(audioseal, "read", short_input_false_positive)
    report = audio.identify_audio(_carrier(tmp_path, 48000))
    assert report.watermark_status == "not_detected"
    for detection in report.detections:
        tail = detection.windows[-1]
        assert tail.end_frame == 48000 + 768
        assert tail.end_frame - tail.start_frame == 48000


def test_uncalibrated_native_clock_is_not_added_automatically(tmp_path, monkeypatch):
    _receiver(monkeypatch)
    report = audio.identify_audio(_carrier(tmp_path, 96000))
    assert report.watermark_status == "not_detected"
    assert {detection.decoder_sample_rate for detection in report.detections} == {16000}
    assert any("24 kHz and 48 kHz" in caveat for caveat in report.caveats)

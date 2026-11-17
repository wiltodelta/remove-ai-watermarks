"""The existing WavMark route also checks native OpenVoice default framing."""

from __future__ import annotations

import json
import shutil
import wave

import numpy as np
import pytest
from click.testing import CliRunner

from remove_ai_watermarks import audio
from remove_ai_watermarks.cli import main

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")


def _source(tmp_path):
    rate = 44100
    t = np.arange(2 * rate + 37) / rate
    frames = np.stack((0.2 * np.sin(2 * np.pi * 1250 * t), 0.1 * np.sin(2 * np.pi * 250 * t)), axis=1)
    file = tmp_path / "native-openvoice.wav"
    with wave.open(str(file), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((frames * 32767).astype("<i2").tobytes())
    return file


def _backend(monkeypatch, *, invalid=False, resistant=False):
    torch = pytest.importorskip("torch")

    class Decoder:
        def parameters(self):
            yield torch.zeros(())

        def decode(self, samples):
            if invalid:
                return torch.full((len(samples), 32), float("nan"))
            values = samples.cpu().numpy()
            t = np.arange(16000) / 44100
            amplitudes = abs(2 * np.mean(values * np.exp(-2j * np.pi * 1250 * t), axis=1))
            bits = np.unpackbits(np.frombuffer(b"default ", dtype=np.uint8)).reshape(2, 32)
            return torch.tensor(
                np.stack(
                    [
                        bits[index] if resistant or amplitude > 0.01 else np.zeros(32)
                        for index, amplitude in enumerate(amplitudes)
                    ]
                ),
                dtype=torch.float32,
            )

    model = Decoder()
    monkeypatch.setattr(audio.wavmark, "is_available", lambda: True)
    monkeypatch.setattr(audio.wavmark, "load_model", lambda: model)
    monkeypatch.setattr(audio.wavmark, "read", lambda *_: audio.wavmark.WavMarkReading(None, ()))


def test_existing_wavmark_route_checks_native_framing_and_each_channel(tmp_path, monkeypatch):
    _backend(monkeypatch)
    report = audio.identify_audio(_source(tmp_path), scheme="wavmark")
    assert report.watermark_status == "detected"
    native = [check for check in report.detections if check.decoder_sample_rate == 44100]
    assert [(check.channel, check.status) for check in native] == [
        (0, "detected"),
        (1, "not_detected"),
        ("mix", "detected"),
    ]
    assert native[0].whole_clip.framing == "openvoice-default"
    assert [(window.start_frame, window.end_frame) for window in native[0].windows] == [(0, 16000), (32000, 48000)]


def test_native_signature_can_gate_verified_processing(tmp_path, monkeypatch):
    _backend(monkeypatch)
    result = audio.remove_audio_invisible(
        _source(tmp_path), tmp_path / "processed.wav", scheme="wavmark", method="spectral"
    )
    assert result.verified_removed is True
    assert not result.skipped
    assert result.after.watermark_status == "not_detected"
    assert result.verification_scope == ("wavmark",)
    assert result.frames_processed == 88237


def test_remaining_signature_preserves_an_existing_output(tmp_path, monkeypatch):
    _backend(monkeypatch, resistant=True)
    source = _source(tmp_path)
    output = tmp_path / "existing.wav"
    output.write_bytes(b"existing output")
    with pytest.raises(audio.AudioVerificationIncomplete):
        audio.remove_audio_invisible(source, output, scheme="wavmark", method="spectral")
    assert output.read_bytes() == b"existing output"


def test_invalid_raw_values_stay_unverified(tmp_path, monkeypatch):
    _backend(monkeypatch, invalid=True)
    report = audio.identify_audio(_source(tmp_path), scheme="wavmark")
    assert report.watermark_status == "unverified"
    assert any(check.whole_clip_error for check in report.detections)


def test_cli_json_distinguishes_the_native_signature(tmp_path, monkeypatch):
    _backend(monkeypatch)
    result = CliRunner().invoke(main, ["audio", "identify", str(_source(tmp_path)), "--scheme", "wavmark", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["watermark_status"] == "detected"
    assert any(check["whole_clip"].get("framing") == "openvoice-default" for check in report["detections"])


def test_native_prefix_reader_is_closed_before_model_inference(tmp_path, monkeypatch):
    _backend(monkeypatch)
    source = _source(tmp_path)
    info = audio.audio_io.probe_audio(source)
    state = {"closed": False}
    readers = []

    def read_audio(*_args, **_kwargs):
        def chunks():
            try:
                yield np.zeros((48000, 2), dtype=np.float32)
                pytest.fail("Read beyond the bounded OpenVoice prefix")
            finally:
                state["closed"] = True

        reader = chunks()
        readers.append(reader)
        return reader

    negative = audio.wavmark.WavMarkSignatureReading((0,), (0,) * 32, "64656661", 18, 1.0)

    def read_signature(*_args):
        assert state["closed"]
        return negative, (negative,)

    monkeypatch.setattr(audio.audio_io, "read_audio", read_audio)
    monkeypatch.setattr(audio.wavmark, "read_openvoice_default", read_signature)
    checks, _ = audio._scan_openvoice_default(source, info)
    assert all(check.status == "not_detected" for check in checks)
    assert state["closed"]

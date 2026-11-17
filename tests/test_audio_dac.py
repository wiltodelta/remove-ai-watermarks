"""Pinned 44.1 kHz DAC loading, raw M/S stereo, and CLI/API routing without downloading weights."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks import audio
from remove_ai_watermarks._internal import audio_dac as dac
from remove_ai_watermarks._internal import audio_resampling as src


@pytest.fixture(autouse=True)
def clear_cache():
    dac.load_pinned_model.cache_clear()
    yield
    dac.load_pinned_model.cache_clear()


def test_import_is_optional_stack_pure():
    root = Path(__file__).resolve().parents[1] / "src"
    script = f"import sys; sys.path.insert(0, {str(root)!r}); import remove_ai_watermarks._internal.audio_dac; "
    script += (
        "assert not {'torch','numpy','scipy','transformers','huggingface_hub','safetensors'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)  # noqa: S603 - fixed local interpreter and test code


def test_missing_dependencies(monkeypatch):
    monkeypatch.setattr(dac.importlib.util, "find_spec", lambda name: None)
    assert not dac.is_available()
    with pytest.raises(ImportError):
        dac.load_pinned_model()
    with pytest.raises(ImportError):
        dac.resynthesize(np.zeros((64, 1), dtype=np.float32), 44100)


def _fake_dac(monkeypatch, *, fault: str | None = None, gain_scale: float = 1.0):
    torch = pytest.importorskip("torch")
    encode_calls = []

    class FakeDacModel:
        def encode(self, value, *, n_quantizers):
            encode_calls.append((value.detach().cpu().numpy().copy(), n_quantizers))
            assert n_quantizers == dac.FIXED_QUANTIZERS == 3
            return SimpleNamespace(audio_codes=value.clone())

        def decode(self, *, audio_codes):
            out = audio_codes.clone() * gain_scale
            if fault == "short":
                out = out[..., :-8]
            elif fault == "nan":
                out[:] = float("nan")
            elif fault == "extra":
                out = torch.cat((out, torch.zeros_like(out[..., :17])), dim=-1)
            return SimpleNamespace(audio_values=out)

    monkeypatch.setattr(dac, "is_available", lambda: True)
    monkeypatch.setattr(dac, "load_pinned_model", lambda **kw: FakeDacModel())
    return encode_calls


@pytest.mark.parametrize(("mode", "channels"), [("ms", 2), ("lr", 2), ("ms", 1)])
def test_identity_ms_and_rms_restoration_preserve_native_grid(monkeypatch, mode, channels):
    encode_calls = _fake_dac(monkeypatch, fault="extra", gain_scale=0.6)
    rng = np.random.default_rng(7)
    x = (0.3 * rng.standard_normal((1024, 2))).astype(np.float32)[:, :channels]
    before = x.copy()
    y = dac.resynthesize(x, 44100, stereo_mode=mode)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(y, x, atol=1e-5, rtol=1e-5)
    assert len(encode_calls) == channels


def test_silent_side_stem_skips_codec_noise_on_dual_mono_stereo(monkeypatch):
    encode_calls = _fake_dac(monkeypatch)
    mono = (0.25 * np.sin(np.linspace(0.0, 20.0, 2048, dtype=np.float32))).reshape(-1, 1)
    dual_mono = np.repeat(mono, 2, axis=1)
    y = dac.resynthesize(dual_mono, 44100, stereo_mode="ms")
    assert len(encode_calls) == 1
    np.testing.assert_allclose(y[:, 0], y[:, 1], atol=1e-7)
    np.testing.assert_allclose(y, dual_mono, atol=1e-5)


@pytest.mark.parametrize(("n", "rate"), [(48001, 48000), (16003, 16000), (44101, 44100)])
def test_rational_clock_restores_exact_native_sample_count(monkeypatch, n, rate):
    encode_calls = _fake_dac(monkeypatch)
    t = np.arange(n, dtype=np.float32) / float(rate)
    x = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).reshape(-1, 1)
    clock = src.RationalClock(n, rate, model_rate=44100, hop=512, minimum_model_frames=512)
    y = dac.resynthesize(x, rate)
    assert encode_calls[0][0].shape == (1, 1, clock.padded_frames)
    assert y.shape == (n, 1)
    assert np.isfinite(y).all()


@pytest.mark.parametrize("fault", ["short", "nan"])
def test_model_faults_raise_runtime_error(monkeypatch, fault):
    _fake_dac(monkeypatch, fault=fault)
    x = np.full((1024, 1), 0.2, dtype=np.float32)
    with pytest.raises(RuntimeError, match="DAC returned"):
        dac.resynthesize(x, 44100)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros((0, 1), dtype=np.float32),
        np.zeros((8, 3), dtype=np.float32),
        np.zeros((8, 1), dtype=np.int16),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
    ],
)
def test_invalid_inputs_rejected(monkeypatch, x):
    _fake_dac(monkeypatch)
    with pytest.raises(ValueError, match="DAC"):
        dac.resynthesize(x, 44100)


def _loader(tmp_path, monkeypatch, *, bad_config: bool = False, loading: dict | None = None):
    torch = pytest.importorskip("torch")
    paths = {"config.json": tmp_path / "config.json", dac.CHECKPOINT_FILENAME: tmp_path / dac.CHECKPOINT_FILENAME}
    for p in paths.values():
        p.write_bytes(p.name.encode())
    monkeypatch.setattr(dac, "CONFIG_SHA256", hashlib.sha256(paths["config.json"].read_bytes()).hexdigest())
    monkeypatch.setattr(
        dac, "CHECKPOINT_SHA256", hashlib.sha256(paths[dac.CHECKPOINT_FILENAME].read_bytes()).hexdigest()
    )
    monkeypatch.setattr(dac, "is_available", lambda: True)
    config = SimpleNamespace(
        model_type="dac",
        sampling_rate=44100,
        n_codebooks=12 if bad_config else 9,
        codebook_size=1024,
        hop_length=512,
    )
    calls = []

    def download(**kw):
        assert kw["repo_id"] == dac.MODEL_REPOSITORY
        assert kw["revision"] == dac.MODEL_REVISION
        return str(paths[kw["filename"]])

    class Model:
        def __init__(self):
            self.config = config

        def to(self, device):
            assert device == "cpu"
            return self

        def train(self, mode):
            assert mode is False
            return self

    def load(path, **kw):
        calls.append(kw)
        assert kw["use_safetensors"] is True
        assert kw["trust_remote_code"] is False
        assert kw["local_files_only"] is True
        assert kw["output_loading_info"] is True
        assert kw["dtype"] is torch.float32
        return Model(), loading or {}

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            DacConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
            DacModel=SimpleNamespace(from_pretrained=load),
        ),
    )
    return paths, calls


def test_safe_builtin_cpu_loading_and_hash_guard(tmp_path, monkeypatch):
    paths, calls = _loader(tmp_path, monkeypatch)
    assert dac.load_pinned_model().config.n_codebooks == 9
    assert len(calls) == 1

    dac.load_pinned_model.cache_clear()
    paths[dac.CHECKPOINT_FILENAME].write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        dac.load_pinned_model()


def test_wrong_config_and_incomplete_load_rejected(tmp_path, monkeypatch):
    _, calls = _loader(tmp_path, monkeypatch, bad_config=True)
    with pytest.raises(RuntimeError, match="configuration"):
        dac.load_pinned_model()
    assert not calls

    dac.load_pinned_model.cache_clear()
    _loader(tmp_path, monkeypatch, loading={"missing_keys": ["decoder.weight"]})
    with pytest.raises(RuntimeError, match="did not load completely"):
        dac.load_pinned_model()


def test_cli_and_api_wire_dac_method(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "in.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        t = np.arange(4410, dtype=np.float32) / 44100.0
        pcm = np.clip(np.rint(8000.0 * np.sin(2.0 * np.pi * 440.0 * t)), -32768, 32767).astype("<i2")
        wav.writeframes(np.column_stack((pcm, pcm)).tobytes())

    assert all("dac" not in profiles for profiles in audio._AUTO_PROFILES.values())
    with pytest.raises(ValueError, match="method=bigvgan"):
        audio.remove_audio_invisible(source, tmp_path / "out.wav", method="dac", vocoder_device="mps")

    seen = []

    def fake_resynth(samples, sample_rate, *, local_files_only=False):
        seen.append((samples.shape, sample_rate, local_files_only))
        return samples

    monkeypatch.setattr(audio.audio_dac, "is_available", lambda: True)
    monkeypatch.setattr(audio.audio_dac, "resynthesize", fake_resynth)
    out = tmp_path / "out.wav"
    res = CliRunner().invoke(
        main,
        [
            "audio",
            "invisible",
            str(source),
            "-o",
            str(out),
            "--method",
            "dac",
            "--force",
            "--no-verify",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    report = json.loads(res.output)
    assert report["method"] == "dac"
    assert report["vocoder_device"] == "cpu"
    assert report["verified_removed"] is None
    assert seen == [((4410, 2), 44100, False)]
    assert out.is_file()


@pytest.mark.parametrize("method", ["dac", "focalcodec"])
def test_long_audio_streams_via_raised_cosine_overlap_windows_preserving_exact_frames(monkeypatch, method):
    sr = 1000
    total_frames = 47503  # 47.503 s > 30 s whole-track limit
    timeline = np.linspace(-0.4, 0.4, total_frames, dtype=np.float32)
    full = np.column_stack((timeline, -0.5 * timeline))
    window_shapes: list[tuple[int, ...]] = []

    def fake_resynth(samples, sample_rate, *, local_files_only=False):
        assert sample_rate == sr
        assert local_files_only is False
        window_shapes.append(samples.shape)
        assert samples.shape[0] <= 22 * sr
        return samples.copy()

    if method == "dac":
        monkeypatch.setattr(audio.audio_dac, "resynthesize", fake_resynth)
        runner = audio._dac_full_track
    else:
        monkeypatch.setattr(audio.audio_focalcodec, "resynthesize_conditioned", fake_resynth)
        runner = audio._focalcodec_full_track

    closed: list[bool] = []

    def irregular_chunks():
        try:
            idx = 0
            for step in (3333, 9101, 18000, 7777, 9292):
                yield full[idx : idx + step]
                idx += step
        finally:
            closed.append(True)

    delivered = np.concatenate(list(runner(irregular_chunks(), sr)), axis=0)
    assert closed == [True]
    assert len(window_shapes) == 3
    assert delivered.shape == full.shape
    np.testing.assert_allclose(delivered, full, atol=1e-6)


def test_stream_neural_track_crossfades_window_offsets_monotonic():
    sr = 100
    total_frames = 3500  # 35 s > 30 s -> triggers 20 s (2000) + 2 s (200) windows
    calls = 0

    def step_offset_transform(samples):
        nonlocal calls
        calls += 1
        return np.full_like(samples, float(calls), dtype=np.float32)

    def gen():
        yield np.zeros((total_frames, 1), dtype=np.float32)

    out = np.concatenate(list(audio._stream_neural_track(gen(), sr, "DAC", step_offset_transform)), axis=0)[:, 0]
    assert calls == 2
    assert out.shape == (total_frames,)
    np.testing.assert_allclose(out[:2000], 1.0, atol=1e-6)
    overlap = out[2000:2200]
    assert np.all((overlap > 1.0) & (overlap < 2.0))
    assert np.all(np.diff(overlap) > 0.0)
    np.testing.assert_allclose(out[2200:], 2.0, atol=1e-6)

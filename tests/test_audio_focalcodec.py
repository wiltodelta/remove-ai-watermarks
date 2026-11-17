"""Pinned FocalCodec loading, architecture, and native clock geometry tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import _focalcodec_network as network
from remove_ai_watermarks._internal import audio_focalcodec as focal
from remove_ai_watermarks._internal import audio_resampling as src


@pytest.fixture(autouse=True)
def clear_cache():
    focal.load_pinned_model.cache_clear()
    yield
    focal.load_pinned_model.cache_clear()


def test_import_is_optional_stack_pure():
    root = Path(__file__).resolve().parents[1] / "src"
    script = (
        f"import sys; sys.path.insert(0, {str(root)!r}); "
        "import remove_ai_watermarks._internal._focalcodec_network; "
        "import remove_ai_watermarks._internal.audio_focalcodec; "
        "assert not {'torch','numpy','scipy','huggingface_hub','safetensors'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)  # noqa: S603 - fixed local interpreter and test code


def test_missing_dependencies(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert not focal.is_available()
    with pytest.raises(ImportError, match="FocalCodec requires"):
        focal.load_pinned_model()
    with pytest.raises(ImportError, match="FocalCodec requires"):
        focal.resynthesize(np.zeros((3, 1)), 24000)


def _fake(monkeypatch, fault=None):
    torch = pytest.importorskip("torch")
    calls = []

    class Model:
        def resynthesize_features(self, context_wave):
            calls.append(context_wave.numpy().copy())
            c = focal.WAVLM_CONTEXT_EACH
            assert np.all(context_wave[:, :c].numpy() == 0.0)
            assert np.all(context_wave[:, -c:].numpy() == 0.0)
            core = context_wave[:, c:-c].clone()
            if fault == "short":
                core = core[:, :-1]
            if fault == "long":
                core = torch.cat((core, core[:, :1]), dim=-1)
            if fault == "nan":
                core[:] = float("nan")
            return core

    monkeypatch.setattr(focal, "is_available", lambda: True)
    monkeypatch.setattr(focal, "load_pinned_model", lambda **kw: Model())
    return calls


@pytest.mark.parametrize(("mode", "channels"), [("ms", 2), ("lr", 2), ("ms", 1)])
def test_identity_keeps_raw_gain_channel_mapping_and_symmetric_context(monkeypatch, mode, channels):
    calls = _fake(monkeypatch)
    x = np.tile(np.array([[1.5, -0.5], [-0.75, 0.25], [0.5, 0.5], [0.0, 0.25]], dtype=np.float32), (80, 1))[
        :, :channels
    ]
    before = x.copy()
    y = focal.resynthesize(x, 16000, stereo_mode=mode)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(x, before)
    assert len(calls) == channels
    expected = (
        x if mode == "lr" or channels == 1 else np.column_stack(((x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2))
    )
    c = focal.WAVLM_CONTEXT_EACH
    for ch in range(channels):
        assert calls[ch].shape == (1, 320 + 2 * c)
        np.testing.assert_array_equal(calls[ch][0, c:-c], expected[:, ch])


@pytest.mark.parametrize(("n", "rate"), [(480241, 48000), (4411, 44100), (17, 16000), (1, 48000)])
def test_fractional_clock_pads_hop_frames_and_returns_exact_native_grid(monkeypatch, n, rate):
    calls = _fake(monkeypatch)
    x = np.zeros((n, 1), dtype=np.float32)
    x[0] = 0.25
    x[-1] = 0.5
    clock = src.RationalClock(n, rate, model_rate=16000, hop=320, minimum_model_frames=320)
    y = focal.resynthesize(x, rate)
    assert calls[0].shape == (1, clock.padded_frames + 2 * focal.WAVLM_CONTEXT_EACH)
    np.testing.assert_array_equal(y, src.inverse(src.forward(x, clock), clock))
    assert y.shape == x.shape


@pytest.mark.parametrize("fault", ["short", "long", "nan"])
def test_model_faults_are_not_repaired(monkeypatch, fault):
    _fake(monkeypatch, fault)
    with pytest.raises(RuntimeError, match="FocalCodec returned"):
        focal.resynthesize(np.ones((320, 2), np.float32), 16000)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(3),
        np.zeros((0, 1)),
        np.zeros((3, 3)),
        np.zeros((3, 1), dtype=int),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_input_geometry_and_values_rejected(monkeypatch, x):
    _fake(monkeypatch)
    with pytest.raises(ValueError, match="FocalCodec"):
        focal.resynthesize(x, 24000)


@pytest.mark.parametrize("rate", [0, -1, True, 24000.0])
def test_invalid_native_rate(monkeypatch, rate):
    _fake(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        focal.resynthesize(np.ones((3, 1)), rate)


def test_unknown_stereo_mode(monkeypatch):
    _fake(monkeypatch)
    with pytest.raises(ValueError, match="stereo mode"):
        focal.resynthesize(np.ones((3, 2)), 24000, stereo_mode="normalize")


def _loader(
    tmp_path,
    monkeypatch,
    *,
    bad_config=False,
    key_count=focal.EXPECTED_STATE_KEY_COUNT,
    missing=(),
    unexpected=(),
    split_dirs=False,
):
    pytest.importorskip("torch")
    cfg_dir = tmp_path / ("a" if split_dirs else "snap")
    ckpt_dir = tmp_path / ("b" if split_dirs else "snap")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_payload = dict(focal._EXPECTED_CONFIG)
    if bad_config:
        config_payload = {**config_payload, "quantizer_config": {"codebook_size": 1024}}
    paths = {
        "config.json": cfg_dir / "config.json",
        focal.CHECKPOINT_FILENAME: ckpt_dir / focal.CHECKPOINT_FILENAME,
    }
    paths["config.json"].write_text(json.dumps(config_payload), encoding="utf-8")
    paths[focal.CHECKPOINT_FILENAME].write_bytes(b"safetensors-placeholder")
    monkeypatch.setattr(focal, "CONFIG_SHA256", hashlib.sha256(paths["config.json"].read_bytes()).hexdigest())
    monkeypatch.setattr(
        focal, "CHECKPOINT_SHA256", hashlib.sha256(paths[focal.CHECKPOINT_FILENAME].read_bytes()).hexdigest()
    )
    monkeypatch.setattr(focal, "is_available", lambda: True)
    built = []

    def download(**kw):
        assert kw["repo_id"] == focal.MODEL_REPOSITORY
        assert kw["revision"] == focal.MODEL_REVISION
        assert kw["local_files_only"] is True
        return str(paths[kw["filename"]])

    class Model:
        def load_state_dict(self, state, *, strict):
            assert strict is True
            built.append(len(state))
            return SimpleNamespace(missing_keys=list(missing), unexpected_keys=list(unexpected))

        def to(self, device):
            assert device == "cpu"
            return self

        def float(self):
            return self

        def train(self, mode):
            assert mode is False
            return self

        def requires_grad_(self, flag):
            assert flag is False
            return self

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        SimpleNamespace(load_file=lambda p, device: {f"k{i}": i for i in range(key_count)}),
    )
    monkeypatch.setattr(focal, "build_focalcodec", lambda cfg: Model())
    return paths, built


def test_safe_cpu_safetensors_loading(tmp_path, monkeypatch):
    _, built = _loader(tmp_path, monkeypatch)
    focal.load_pinned_model()
    assert built == [focal.EXPECTED_STATE_KEY_COUNT]


@pytest.mark.parametrize("filename", ["config.json", focal.CHECKPOINT_FILENAME])
def test_bad_hash_stops_before_model_load(tmp_path, monkeypatch, filename):
    paths, built = _loader(tmp_path, monkeypatch)
    paths[filename].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        focal.load_pinned_model()
    assert not built


def test_mismatched_snapshot_parent_rejected(tmp_path, monkeypatch):
    _, built = _loader(tmp_path, monkeypatch, split_dirs=True)
    with pytest.raises(RuntimeError, match="one snapshot directory"):
        focal.load_pinned_model()
    assert not built


def test_wrong_profile_stops_before_model_load(tmp_path, monkeypatch):
    _, built = _loader(tmp_path, monkeypatch, bad_config=True)
    with pytest.raises(RuntimeError, match="configuration"):
        focal.load_pinned_model()
    assert not built


def test_wrong_state_key_count_rejected(tmp_path, monkeypatch):
    _, built = _loader(tmp_path, monkeypatch, key_count=354)
    with pytest.raises(RuntimeError, match="355 state keys"):
        focal.load_pinned_model()
    assert not built


@pytest.mark.parametrize(("missing", "unexpected"), [(["enc.w"], []), ([], ["extra.w"])])
def test_incomplete_state_load_rejected(tmp_path, monkeypatch, missing, unexpected):
    _loader(tmp_path, monkeypatch, missing=missing, unexpected=unexpected)
    with pytest.raises(RuntimeError, match="did not load completely"):
        focal.load_pinned_model()


def test_build_focalcodec_architecture_forward_and_validation():
    torch = pytest.importorskip("torch")
    bad = {**focal._EXPECTED_CONFIG, "decoder_name": "UnknownVocoder"}
    with pytest.raises(ValueError, match="Only the pinned"):
        network.build_focalcodec(bad)

    tiny_cfg = {
        "encoder_name": "WavLM",
        "encoder_config": {
            "hidden_dims": [16, 16],
            "kernel_sizes": [10, 3],
            "strides": [5, 2],
            "num_layers": 1,
            "dim": 32,
            "ffn_dim": 64,
            "num_heads": 4,
            "num_buckets": 32,
            "max_distance": 64,
            "dropout": 0.0,
            "conv_pos": 8,
            "conv_pos_groups": 4,
        },
        "compressor_name": "FocalEncoder",
        "compressor_config": {
            "input_dim": 32,
            "output_dim": 4,
            "hidden_dims": [32, 16],
            "downscale_factors": [1, 1],
            "focal_window": 3,
            "focal_level": 2,
            "focal_factor": 2,
            "dropout": 0.0,
            "use_post_norm": False,
            "use_layerscale": False,
            "layerscale_init": 0.0001,
            "normalize_modulator": False,
        },
        "quantizer_name": "BinarySphericalQuantizer",
        "quantizer_config": {"codebook_size": 16},
        "decompressor_name": "FocalDecoder",
        "decompressor_config": {
            "input_dim": 4,
            "output_dim": 32,
            "hidden_dims": [16, 32],
            "upscale_factors": [1, 1],
            "focal_window": 3,
            "focal_level": 2,
            "focal_factor": 2,
            "dropout": 0.0,
            "use_post_norm": False,
            "use_layerscale": False,
            "layerscale_init": 0.0001,
            "normalize_modulator": False,
        },
        "decoder_name": "Vocos",
        "decoder_config": {
            "input_channels": 32,
            "num_layers": 1,
            "dim": 16,
            "ffn_dim": 32,
            "kernel_size": 3,
            "padding": 1,
            "layerscale_init": None,
            "n_fft": 32,
            "hop_length": 10,
        },
    }
    model = network.build_focalcodec(tiny_cfg).train(False)
    # Receptive field of [k=10,s=5] -> [k=3,s=2] is 10 + 5*(3-1) = 20, total stride 10;
    # 30 core samples + (5, 5) symmetric context = 40 -> 3 frames -> 30 output samples.
    with torch.inference_mode():
        wave = torch.linspace(-0.25, 0.25, 40, dtype=torch.float32)[None]
        out = model.resynthesize_features(wave)
    assert tuple(out.shape) == (1, 30)
    assert torch.isfinite(out).all()


def test_pinned_architecture_exact_355_state_keys():
    pytest.importorskip("torch")
    model = network.build_focalcodec(focal._EXPECTED_CONFIG)
    assert len(model.state_dict()) == focal.EXPECTED_STATE_KEY_COUNT


def test_cached_pinned_checkpoint_end_to_end_when_present():
    if not focal.is_available():
        pytest.skip("Optional FocalCodec dependencies unavailable")
    try:
        focal.load_pinned_model(local_files_only=True)
    except Exception as exc:  # pragma: no cover - CI lacks cached model weights
        if "LocalEntryNotFoundError" in type(exc).__name__ or "Cannot find the requested files" in str(exc):
            pytest.skip("Pinned FocalCodec snapshot not cached locally")
        raise
    t = np.linspace(0.0, 0.2, 4800, endpoint=False, dtype=np.float32)
    wave = (0.15 * np.sin(2 * np.pi * 220.0 * t))[:, None]
    out = focal.resynthesize(wave, 24000, local_files_only=True)
    assert out.shape == (4800, 1)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_resynthesize_conditioned_preconditions_midband_and_restores_subband_envelopes(monkeypatch):
    monkeypatch.setattr(focal, "is_available", lambda: True)
    seen_inputs = []

    def fake_resynthesize(samples, sample_rate, *, stereo_mode="ms", local_files_only=True):
        assert stereo_mode == "ms"
        seen_inputs.append((samples.copy(), sample_rate, local_files_only))
        # Simulate Vocos 50 Hz codec attenuating 300 Hz speech band by 0.5x and generating slight pause hiss.
        out = samples * 0.5
        out[-2400:] += 0.02
        return out

    monkeypatch.setattr(focal, "resynthesize", fake_resynthesize)
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    carrier = np.zeros((sr, 2), dtype=np.float32)
    # Active speech-band carrier in first 0.8s + 1250 Hz Perth-band tone; silence in final 0.2s.
    carrier[:19200, 0] = 0.25 * np.sin(2 * np.pi * 300.0 * t[:19200]) + 0.10 * np.sin(2 * np.pi * 1250.0 * t[:19200])
    carrier[:19200, 1] = 0.20 * np.sin(2 * np.pi * 300.0 * t[:19200])

    delivered = focal.resynthesize_conditioned(carrier, sr, local_files_only=False)
    assert delivered.shape == (sr, 2)
    assert delivered.dtype == np.float32
    assert np.isfinite(delivered).all()
    assert len(seen_inputs) == 1
    pre_input, seen_sr, seen_local = seen_inputs[0]
    assert seen_sr == sr
    assert seen_local is False
    # 1250 Hz mark band must be strongly notched before reaching FocalCodec encoder.
    spec_src = np.abs(np.fft.rfft(carrier[:19200, 0]))
    spec_pre = np.abs(np.fft.rfft(pre_input[:19200, 0]))
    freqs = np.fft.rfftfreq(19200, d=1.0 / sr)
    idx_1250 = int(np.argmin(np.abs(freqs - 1250.0)))
    assert spec_pre[idx_1250] < spec_src[idx_1250] * 0.05
    # Active 300 Hz speech band energy is restored close to source despite 0.5x codec attenuation,
    # while trailing silence hiss (0.02) is suppressed toward zero.
    rms_src_active = float(np.sqrt(np.mean(carrier[2400:16800, 1] ** 2)))
    rms_out_active = float(np.sqrt(np.mean(delivered[2400:16800, 1] ** 2)))
    assert abs(20.0 * np.log10(rms_out_active / rms_src_active)) < 0.5
    assert float(np.sqrt(np.mean(delivered[-1200:, 0] ** 2))) < 0.001

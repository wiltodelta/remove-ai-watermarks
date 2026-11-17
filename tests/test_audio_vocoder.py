"""Pinned loading and full-track semantics with in-memory models, no downloads."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import audio_resampling as src
from remove_ai_watermarks._internal import audio_vocoder as vocoder


def _identity_model(monkeypatch, fault=None):
    torch = pytest.importorskip("torch")
    calls = []

    def model(value):
        calls.append(value.clone())
        result = value.unsqueeze(1)
        if fault == "short":
            return result[..., :-1]
        if fault == "nonfinite":
            return result * float("nan")
        return result

    monkeypatch.setattr(vocoder, "is_available", lambda: True)
    monkeypatch.setattr(vocoder, "load_pinned_model", lambda *a, **kw: model)
    monkeypatch.setattr(vocoder, "mel_spectrogram", lambda a: torch.from_numpy(a).unsqueeze(0))
    return calls


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("frames", [1, 2, 160, 1115, 480240])
def test_identity_model_retains_native_grid_and_raw_mid_side(monkeypatch, frames, channels):
    calls = _identity_model(monkeypatch)
    value = np.random.default_rng(31).normal(size=(frames, channels)).astype(np.float32) * 2
    result = vocoder.resynthesize(value, 48000)
    clock = src.RationalClock(frames, 48000)
    reference = src.inverse(src.forward(value, clock), clock)
    np.testing.assert_allclose(result, reference, rtol=3e-5, atol=3e-6)
    assert result.shape == value.shape
    assert result.dtype == np.float32
    assert len(calls) == channels
    if channels == 2:
        mid = (value[:, 0] + value[:, 1]) / 2
        side = (value[:, 0] - value[:, 1]) / 2
        for call, expected in zip(calls, (mid, side), strict=True):
            np.testing.assert_array_equal(call.numpy()[0], src.pad_model(src.forward(expected, clock), clock))
    if frames > 100:
        assert np.max(np.abs(result)) > 1  # No additional clipping or normalization.


@pytest.mark.parametrize("fault", ["short", "nonfinite"])
def test_invalid_model_output_rejected(monkeypatch, fault):
    _identity_model(monkeypatch, fault)
    with pytest.raises(RuntimeError, match="BigVGAN returned"):
        vocoder.resynthesize(np.ones((1024, 2), dtype=np.float32), 48000)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(5),
        np.zeros((0, 1)),
        np.zeros((5, 3)),
        np.zeros((5, 1), dtype=np.int16),
        np.full((5, 1), np.nan),
        np.full((5, 1), np.inf),
    ],
)
def test_bad_source_fails_before_model_load(monkeypatch, value):
    monkeypatch.setattr(vocoder, "load_pinned_model", lambda *a, **kw: pytest.fail("Loaded invalid source"))
    with pytest.raises(ValueError, match="BigVGAN"):
        vocoder.resynthesize(value, 48000)


def test_import_does_not_load_optional_runtimes():
    root = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from remove_ai_watermarks._internal import audio_vocoder, audio_resampling, _bigvgan_network
assert not any(name in sys.modules for name in ('numpy', 'scipy', 'torch', 'librosa', 'huggingface_hub'))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert completed.returncode == 0, completed.stderr


def test_missing_dependencies_fail_before_import(monkeypatch):
    vocoder.load_pinned_model.cache_clear()
    monkeypatch.setattr(vocoder.importlib.util, "find_spec", lambda name: None)
    assert not vocoder.is_available()
    with pytest.raises(RuntimeError, match="requires numpy"):
        vocoder.load_pinned_model()


def _loading_fixture(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    vocoder.load_pinned_model.cache_clear()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"fixture": True}))
    checkpoint = tmp_path / vocoder.CHECKPOINT_FILENAME
    torch.save({"generator": {"weight": torch.ones(1, 1)}}, checkpoint)
    monkeypatch.setattr(vocoder, "CONFIG_SHA256", hashlib.sha256(config.read_bytes()).hexdigest())
    monkeypatch.setattr(vocoder, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    requests = []

    def download(**kwargs):
        requests.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(vocoder, "is_available", lambda: True)
    return torch, config, checkpoint, requests


@pytest.mark.parametrize("artifact", ["config", "checkpoint"])
def test_wrong_digest_rejected_before_deserialization(tmp_path, monkeypatch, artifact):
    torch, config, checkpoint, _ = _loading_fixture(tmp_path, monkeypatch)
    (config if artifact == "config" else checkpoint).write_bytes(b"wrong bytes")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Deserialized wrong digest"))
    monkeypatch.setattr(vocoder, "build_generator", lambda *a: pytest.fail("Built before checking both hashes"))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        vocoder.load_pinned_model()


def test_incomplete_checkpoint_fails_before_weight_norm_removal(tmp_path, monkeypatch):
    torch, _, _, _ = _loading_fixture(tmp_path, monkeypatch)
    model = torch.nn.Linear(1, 1)  # Missing bias in the saved state.
    model.remove_weight_norm = lambda: pytest.fail("Removed weight norm before strict state check")
    monkeypatch.setattr(vocoder, "build_generator", lambda config: model)
    with pytest.raises(RuntimeError, match="Missing key"):
        vocoder.load_pinned_model()


def test_weights_only_strict_loading_then_weight_norm_removal(tmp_path, monkeypatch):
    torch, _, _, requests = _loading_fixture(tmp_path, monkeypatch)
    events = []
    original_load = torch.load

    def load(*args, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        events.append("weights_only")
        return original_load(*args, **kwargs)

    class Model(torch.nn.Linear):
        def load_state_dict(self, state, strict=True):
            assert strict is True
            events.append("strict")
            return super().load_state_dict(state, strict=strict)

        def remove_weight_norm(self):
            events.append("remove")

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(vocoder, "build_generator", lambda config: Model(1, 1, bias=False))
    model = vocoder.load_pinned_model()
    assert events == ["weights_only", "strict", "remove"]
    assert not model.training
    assert not model.weight.requires_grad
    assert model.weight.dtype == torch.float32
    assert all(r["revision"] == vocoder.MODEL_REVISION and r["local_files_only"] is True for r in requests)
    vocoder.load_pinned_model.cache_clear()


@pytest.mark.parametrize("device", ["auto", "gpu", "cpu:1"])
def test_unknown_device_is_rejected(device):
    with pytest.raises(ValueError, match="device"):
        vocoder.load_pinned_model(device)

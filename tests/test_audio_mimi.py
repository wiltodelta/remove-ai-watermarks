"""Pinned Mimi loading and native geometry, without downloading model weights."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import audio_mimi as mimi
from remove_ai_watermarks._internal import audio_resampling as src


@pytest.fixture(autouse=True)
def clear_cache():
    mimi.load_pinned_model.cache_clear()
    yield
    mimi.load_pinned_model.cache_clear()


def test_import_is_optional_stack_pure():
    root = Path(__file__).resolve().parents[1] / "src"
    script = f"import sys; sys.path.insert(0, {str(root)!r}); import remove_ai_watermarks._internal.audio_mimi; "
    script += (
        "assert not {'torch','numpy','scipy','transformers','huggingface_hub','safetensors'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)  # noqa: S603 - fixed local interpreter and test code


def test_missing_dependencies(monkeypatch):
    monkeypatch.setattr(mimi.importlib.util, "find_spec", lambda name: None)
    assert not mimi.is_available()
    with pytest.raises(ImportError):
        mimi.load_pinned_model()
    with pytest.raises(ImportError):
        mimi.resynthesize(np.zeros((3, 1)), 24000)


def _fake(monkeypatch, fault=None):
    torch = pytest.importorskip("torch")
    calls = []

    class Model:
        def __call__(self, value, *, padding_mask):
            calls.append(value.numpy().copy())
            assert padding_mask.shape == value.shape
            assert padding_mask.all()
            result = value.clone()
            if fault == "short":
                result = result[..., :-1]
            if fault == "long":
                result = torch.cat((result, result[..., :1]), dim=-1)
            if fault == "channels":
                result = result[:0]
            if fault == "nan":
                result[:] = float("nan")
            return SimpleNamespace(audio_values=result)

    monkeypatch.setattr(mimi, "is_available", lambda: True)
    monkeypatch.setattr(mimi, "load_pinned_model", lambda **kw: Model())
    return calls


@pytest.mark.parametrize(("mode", "channels"), [("ms", 2), ("lr", 2), ("ms", 1)])
def test_identity_keeps_raw_gain_and_channel_mapping(monkeypatch, mode, channels):
    calls = _fake(monkeypatch)
    x = np.array([[1.5, -0.5], [-0.75, 0.25], [0.5, 0.5]], dtype=np.float32)[:, :channels]
    before = x.copy()
    y = mimi.resynthesize(x, 24000, stereo_mode=mode)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(x, before)
    expected = (
        x if mode == "lr" or channels == 1 else np.column_stack(((x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2))
    )
    np.testing.assert_array_equal(calls[0][:, 0, :].T, expected)


@pytest.mark.parametrize(("n", "rate"), [(480241, 48000), (4411, 44100), (17, 16000), (1, 48000)])
def test_fractional_clock_no_external_model_padding(monkeypatch, n, rate):
    calls = _fake(monkeypatch)
    x = np.zeros((n, 1), dtype=np.float32)
    x[0] = 0.25
    x[-1] = 0.5
    clock = src.RationalClock(n, rate, 24000, 1, 1)
    y = mimi.resynthesize(x, rate)
    assert calls[0].shape == (1, 1, clock.model_frames)
    np.testing.assert_array_equal(y, src.inverse(src.forward(x, clock), clock))
    assert y.shape == x.shape


@pytest.mark.parametrize("fault", ["short", "long", "channels", "nan"])
def test_model_faults_are_not_repaired(monkeypatch, fault):
    _fake(monkeypatch, fault)
    with pytest.raises(RuntimeError, match="Mimi returned"):
        mimi.resynthesize(np.ones((9, 2), np.float32), 24000)


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
    with pytest.raises(ValueError, match="Mimi"):
        mimi.resynthesize(x, 24000)


@pytest.mark.parametrize("rate", [0, -1, True, 24000.0])
def test_invalid_native_rate(monkeypatch, rate):
    _fake(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        mimi.resynthesize(np.ones((3, 1)), rate)


def test_unknown_stereo_mode(monkeypatch):
    _fake(monkeypatch)
    with pytest.raises(ValueError, match="stereo mode"):
        mimi.resynthesize(np.ones((3, 2)), 24000, stereo_mode="normalize")


def _loader(tmp_path, monkeypatch, *, bad_config=False, loading=None):
    torch = pytest.importorskip("torch")
    paths = {"config.json": tmp_path / "config.json", mimi.CHECKPOINT_FILENAME: tmp_path / mimi.CHECKPOINT_FILENAME}
    for p in paths.values():
        p.write_bytes(p.name.encode())
    monkeypatch.setattr(mimi, "CONFIG_SHA256", hashlib.sha256(paths["config.json"].read_bytes()).hexdigest())
    monkeypatch.setattr(
        mimi, "CHECKPOINT_SHA256", hashlib.sha256(paths[mimi.CHECKPOINT_FILENAME].read_bytes()).hexdigest()
    )
    monkeypatch.setattr(mimi, "is_available", lambda: True)
    config = SimpleNamespace(
        model_type="mimi",
        sampling_rate=24000,
        audio_channels=1,
        num_quantizers=8 if bad_config else 32,
        use_cache=False,
        normalize=False,
        use_causal_conv=True,
        trim_right_ratio=1.0,
    )
    calls = []

    def download(**kw):
        assert kw["repo_id"] == mimi.MODEL_REPOSITORY
        assert kw["revision"] == mimi.MODEL_REVISION
        assert kw["local_files_only"] is True
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
            MimiConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
            MimiModel=SimpleNamespace(from_pretrained=load),
        ),
    )
    return paths, calls


def test_safe_builtin_cpu_loading(tmp_path, monkeypatch):
    _, calls = _loader(tmp_path, monkeypatch)
    assert mimi.load_pinned_model().config.num_quantizers == 32
    assert len(calls) == 1


@pytest.mark.parametrize("filename", ["config.json", mimi.CHECKPOINT_FILENAME])
def test_bad_hash_stops_before_model_load(tmp_path, monkeypatch, filename):
    paths, calls = _loader(tmp_path, monkeypatch)
    paths[filename].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        mimi.load_pinned_model()
    assert not calls


def test_wrong_profile_stops_before_model_load(tmp_path, monkeypatch):
    _, calls = _loader(tmp_path, monkeypatch, bad_config=True)
    with pytest.raises(RuntimeError, match="configuration"):
        mimi.load_pinned_model()
    assert not calls


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_load_rejected(tmp_path, monkeypatch, key):
    _loader(tmp_path, monkeypatch, loading={key: ["fault"]})
    with pytest.raises(RuntimeError, match="did not load completely"):
        mimi.load_pinned_model()

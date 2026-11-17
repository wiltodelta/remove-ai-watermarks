"""Codec geometry and failures, using an in-memory model without downloads."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import audio_codec as codec


def _model(monkeypatch, fault=None):
    torch = pytest.importorskip("torch")

    class Model:
        def __call__(self, samples, *, bandwidth):
            assert bandwidth == 24
            assert samples.shape[0:2] == (2, 1)
            if fault == "short":
                return SimpleNamespace(audio_values=samples[..., :-1])
            if fault == "channels":
                return SimpleNamespace(audio_values=samples[:1])
            if fault == "nonfinite":
                return SimpleNamespace(audio_values=samples * float("nan"))
            return SimpleNamespace(audio_values=torch.cat((samples * 0.5, torch.zeros((2, 1, 5))), dim=-1))

    monkeypatch.setattr(codec, "is_available", lambda: True)
    monkeypatch.setattr(codec, "load_pinned_model", Model)


def test_resynthesis_keeps_independent_channels_and_trims_only_padding(monkeypatch):
    _model(monkeypatch)
    samples = np.arange(34, dtype=np.float32).reshape(17, 2) / 100
    delivered = codec.resynthesize(samples, 24000, 24)
    np.testing.assert_array_equal(delivered, samples * 0.5)


@pytest.mark.parametrize("fault", ["short", "channels", "nonfinite"])
def test_bad_model_output_cannot_be_published(monkeypatch, fault):
    _model(monkeypatch, fault)
    with pytest.raises(RuntimeError, match="EnCodec returned"):
        codec.resynthesize(np.ones((17, 2), dtype=np.float32), 24000, 24)


def test_streaming_context_keeps_channel_order_and_exact_tail(monkeypatch):
    calls = []

    def process(samples, rate, bandwidth, *, stereo_mode="independent"):
        calls.append(len(samples))
        return samples * 0.5

    monkeypatch.setattr(codec, "resynthesize", process)
    samples = np.arange(20 * 16000 * 2 + 2, dtype=np.float32).reshape(-1, 2)
    chunks = (samples[start : start + 128000] for start in range(0, len(samples), 128000))
    result = np.concatenate(list(codec.resynthesize_chunks(chunks, 16000)))
    np.testing.assert_array_equal(result, samples * 0.5)
    assert calls == [144000, 160000, 80001]


def test_wrong_checkpoint_is_rejected_before_loading(tmp_path, monkeypatch):
    codec.load_pinned_model.cache_clear()
    path = tmp_path / "checkpoints" / codec.CHECKPOINT_FILENAME
    path.parent.mkdir()
    path.write_bytes(b"incorrect checkpoint")
    monkeypatch.setattr(codec, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda **kw: str(path)))
    with pytest.raises(RuntimeError, match="checkpoint digest mismatch"):
        codec.load_pinned_model()


def test_partial_model_loading_is_rejected(tmp_path, monkeypatch):
    codec.load_pinned_model.cache_clear()
    config, weights = tmp_path / "config.json", tmp_path / "model.safetensors"
    config.write_bytes(b"{}")
    weights.write_bytes(b"test weights")
    monkeypatch.setattr(codec, "is_available", lambda: True)
    monkeypatch.setattr(codec, "CONFIG_SHA256", hashlib.sha256(config.read_bytes()).hexdigest())
    monkeypatch.setattr(codec, "CHECKPOINT_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    model = SimpleNamespace()
    model.to = lambda device: model
    model.eval = lambda: model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kw: str(config if kw["filename"] == "config.json" else weights)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            EncodecModel=SimpleNamespace(from_pretrained=lambda *a, **kw: (model, {"missing_keys": ["decoder.weight"]}))
        ),
    )
    with pytest.raises(RuntimeError, match="did not load completely"):
        codec.load_pinned_model()


def test_stereo_ms_mode_preserves_mid_side_channel_coherence(monkeypatch):
    pytest.importorskip("torch")

    seen_inputs = []

    class AsymmetricModel:
        def __call__(self, samples, *, bandwidth):
            assert bandwidth == 24
            seen_inputs.append(samples.detach().cpu().numpy().copy())
            out = samples.clone()
            # Slot 0 and slot 1 experience slightly different quantizer bias/gain.
            out[0] = out[0] * 0.96 + 0.015
            out[1] = out[1] * 0.88 - 0.015
            return SimpleNamespace(audio_values=out)

    monkeypatch.setattr(codec, "is_available", lambda: True)
    monkeypatch.setattr(codec, "load_pinned_model", AsymmetricModel)

    t = np.linspace(0.0, 1.0, 2400, endpoint=False, dtype=np.float32)
    center = 0.35 * np.sin(2.0 * np.pi * 220.0 * t)
    stereo = np.column_stack((center, center))  # Identical L/R phantom-center signal

    indep = codec.resynthesize(stereo, 24000, 24, stereo_mode="independent")
    ms = codec.resynthesize(stereo, 24000, 24, stereo_mode="ms")

    # Independent mode introduces L/R gain+DC mismatch on a pure center signal,
    # whereas M/S encodes Side=0 in slot 1 so constant slot-1 DC cancels or stays uniform,
    # and slot 0 receives exact Mid=(L+R)/2 and slot 1 receives exact Side=0.
    np.testing.assert_allclose(seen_inputs[1][0, 0], center, atol=1e-6)
    np.testing.assert_allclose(seen_inputs[1][1, 0], np.zeros_like(center), atol=1e-6)
    indep_lr_rms = float(np.sqrt(np.mean((indep[:, 0] - indep[:, 1]) ** 2)))
    ms_lr_rms = float(np.sqrt(np.mean((ms[:, 0] - ms[:, 1]) ** 2)))
    assert ms_lr_rms < indep_lr_rms

    forwarded_modes = []

    def spy_resynthesize(samples, rate, bandwidth, *, stereo_mode="independent"):
        forwarded_modes.append(stereo_mode)
        return samples

    monkeypatch.setattr(codec, "resynthesize", spy_resynthesize)
    out = np.concatenate(list(codec.resynthesize_chunks(iter([stereo]), 24000, stereo_mode="ms")))
    assert forwarded_modes == ["ms"]
    np.testing.assert_array_equal(out, stereo)

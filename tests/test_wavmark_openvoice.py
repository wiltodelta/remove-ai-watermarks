"""Known OpenVoice default framing, distinct from WavMark's sync wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks._internal import wavmark


def _bits():
    return np.unpackbits(np.frombuffer(b"default ", dtype=np.uint8)).reshape(2, 32)


class _Decoder:
    def __init__(self, values):
        pytest.importorskip("torch")
        self.values = values
        self.seen = None

    def parameters(self):
        import torch

        yield torch.zeros(())

    def decode(self, samples):
        import torch

        self.seen = samples.cpu().numpy().copy()
        return torch.tensor(self.values, dtype=torch.float32)


@pytest.mark.parametrize(("length", "count"), [(16000, 1), (47999, 1), (48000, 2), (64000, 2)])
def test_native_offsets_and_final_complete_fragment(length, count):
    samples = np.arange(length, dtype=np.float32) / length
    model = _Decoder(_bits()[:count])
    whole, windows = wavmark.read_openvoice_default(model, samples)
    assert whole.detected
    assert whole.framing == "openvoice-default"
    assert whole.reference_hex == b"default "[: 4 * count].hex()
    assert whole.matching_bits == 32 * count
    assert whole.start_samples == (0, 32000)[:count]
    assert len(windows) == count
    np.testing.assert_array_equal(model.seen[0], samples[:16000])
    if count == 2:
        np.testing.assert_array_equal(model.seen[1], samples[32000:48000])
    assert all(window.detected for window in windows)


def test_one_matching_fragment_cannot_be_hidden_by_the_other():
    bits = _bits().copy()
    bits[0] = 1 - bits[0]
    whole, windows = wavmark.read_openvoice_default(_Decoder(bits), np.zeros(48000))
    assert whole.detected
    assert not windows[0].detected
    assert windows[1].detected


def test_distributed_residual_correlation_is_not_a_negative():
    bits = _bits().copy()
    bits[:, :5] = 1 - bits[:, :5]
    whole, windows = wavmark.read_openvoice_default(_Decoder(bits), np.zeros(48000))
    assert whole.matching_bits == 54
    assert whole.detected
    assert not any(window.detected for window in windows)


@pytest.mark.parametrize(("matches", "detected"), [(29, False), (30, True)])
def test_single_fragment_correlation_boundary(matches, detected):
    bits = _bits()[:1].copy()
    bits[0, matches:] = 1 - bits[0, matches:]
    whole, _ = wavmark.read_openvoice_default(_Decoder(bits), np.zeros(16000))
    assert whole.detected is detected


@pytest.mark.parametrize("samples", [np.zeros(15999), np.zeros((16000, 2)), np.full(16000, np.nan)])
def test_invalid_native_samples_are_not_negative(samples):
    with pytest.raises(ValueError, match="native mono"):
        wavmark.read_openvoice_default(_Decoder(_bits()), samples)


@pytest.mark.parametrize("values", [np.zeros((1, 32)), np.zeros((2, 31)), np.full((2, 32), np.nan)])
def test_invalid_model_output_is_not_negative(values):
    with pytest.raises(RuntimeError, match="raw signature"):
        wavmark.read_openvoice_default(_Decoder(values), np.zeros(48000))

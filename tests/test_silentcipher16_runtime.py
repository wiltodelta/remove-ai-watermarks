"""Receiver guards for the experimental terminated eight-symbol profile."""

from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import silentcipher16


def _raw(payload=(0, 1, 2, 0, 2, 1, 0), cycles=4):
    symbols = [value + 1 for value in payload] + [0]
    return np.eye(4, dtype=np.float32)[np.tile(symbols, cycles)].T


@pytest.mark.parametrize("payload", [(0,) * 7, (2,) * 7, (0, 1, 2, 0, 2, 1, 0)])
def test_symbols_are_canonical_and_never_truncated_into_bytes(payload):
    for offset in range(8):
        found, agreement = silentcipher16._decode_sequence(np.roll(_raw(payload), offset, axis=1))
        assert found == payload
        assert agreement == 1


@pytest.mark.parametrize("symbol", range(4))
def test_constant_class_bias_is_not_a_valid_frame(symbol):
    raw = np.eye(4, dtype=np.float32)[np.full(32, symbol)].T
    assert silentcipher16._decode_sequence(raw)[0] is None


def test_multiple_terminators_are_not_a_valid_frame():
    labels = _raw().argmax(axis=0)
    labels[::8] = 0
    assert silentcipher16._decode_sequence(np.eye(4, dtype=np.float32)[labels].T)[0] is None


@pytest.mark.parametrize("fault", ["nan", "shape", "short"])
def test_bad_neural_values_and_incomplete_repetitions_remain_unverified(fault):
    raw = _raw()
    if fault == "nan":
        raw[0, 0] = np.nan
    elif fault == "shape":
        raw = raw[:3]
    else:
        raw = raw[:, :23]
    with pytest.raises(RuntimeError):
        silentcipher16._decode_sequence(raw)


def _model(monkeypatch, outputs, hop=30):
    torch = pytest.importorskip("torch")
    from remove_ai_watermarks._internal import _silentcipher_network

    iterator = iter(outputs)
    monkeypatch.setattr(silentcipher16, "HOP_LENGTH", hop)
    monkeypatch.setattr(_silentcipher_network, "magnitude", lambda samples, window, **kwargs: samples)
    return silentcipher16.base.DecoderModel(lambda _: torch.from_numpy(next(iterator))[None, None], None, "cpu")


def test_invalid_strong_phase_cannot_hide_a_valid_foreign_message(monkeypatch):
    bias = np.eye(4, dtype=np.float32)[np.ones(32, dtype=int)].T
    expected = (2, 1, 0, 2, 1, 0, 2)
    model = _model(monkeypatch, [bias, _raw(expected), bias])
    result = silentcipher16.read(model, np.ones(48000))
    assert result.payload_symbols == expected
    assert result.phase_offset == 10
    assert result.phases_checked == 2


def test_negative_requires_every_phase_in_the_grid(monkeypatch):
    bias = np.eye(4, dtype=np.float32)[np.ones(32, dtype=int)].T
    result = silentcipher16.read(_model(monkeypatch, [bias] * 3), np.ones(48000))
    assert not result.detected
    assert result.phases_checked == 3
    assert result.symbol_consistency == 1


@pytest.mark.parametrize(("errors", "detected"), [(8, True), (9, False)])
def test_agreement_threshold_is_applied_to_valid_framing(monkeypatch, errors, detected):
    labels = _raw(cycles=10).argmax(axis=0)
    indexes = [*(np.arange(8) * 9), 8]
    for index in indexes[:errors]:
        labels[index] = (labels[index] + 1) % 4
    raw = np.eye(4, dtype=np.float32)[labels].T
    result = silentcipher16.read(_model(monkeypatch, [raw], hop=10), np.ones(48000))
    assert result.detected is detected
    assert result.symbol_consistency == pytest.approx(1 - errors / 80)


def test_numeric_failure_never_becomes_a_cpu_negative(monkeypatch):
    def fail(*args):
        raise silentcipher16.base.ModelCheckError("invalid numerics")

    monkeypatch.setattr(silentcipher16, "_read_native", fail)
    monkeypatch.setattr(silentcipher16, "load_model", lambda *_: pytest.fail("Unexpected CPU fallback"))
    with pytest.raises(RuntimeError, match="invalid numerics"):
        silentcipher16.read(SimpleNamespace(device="mps"), np.ones(48000))


def test_device_failure_retries_cpu_and_reports_actual_device(monkeypatch):
    calls = []

    def read(model, samples):
        calls.append(model.device)
        if model.device == "mps":
            raise RuntimeError("unsupported device operation")
        return silentcipher16.SilentCipher16Reading(None, 0.4, None, 103, "cpu")

    monkeypatch.setattr(silentcipher16, "_read_native", read)
    monkeypatch.setattr(silentcipher16, "load_model", lambda device: SimpleNamespace(device=device))
    result = silentcipher16.read(SimpleNamespace(device="mps"), np.ones(48000))
    assert result.device == "cpu"
    assert calls == ["mps", "cpu"]


def test_short_or_silent_input_is_not_absence():
    model = SimpleNamespace(device="cpu")
    with pytest.raises(ValueError, match="three seconds"):
        silentcipher16.read(model, np.ones(47999))
    with pytest.raises(RuntimeError, match="silent"):
        silentcipher16.read(model, np.zeros(48000))

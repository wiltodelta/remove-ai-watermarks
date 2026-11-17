"""Synthetic receiver guards, independent of checkpoint and GPU availability."""

import numpy as np
import pytest

from remove_ai_watermarks._internal.silentcipher import _decode_sequence as parse


def raw_message(payload=(1, 2, 3, 4, 5), cycles=4):
    bits = "".join(format(n, "08b") for n in payload)
    letters = [int(bits[i : i + 2], 2) + 1 for i in range(0, 40, 2)] + [0]
    return np.eye(5, dtype=np.float32)[np.tile(letters, cycles)].T


@pytest.mark.parametrize("payload", [(1, 2, 3, 4, 5), (5, 4, 3, 2, 1), (0, 0, 0, 0, 0), (255, 255, 255, 255, 255)])
def test_any_valid_payload_is_recovered(payload):
    for shift in [0, 1, 7, 20]:
        recovered, confidence = parse(np.roll(raw_message(payload), shift, axis=1))
        assert recovered == payload
        assert confidence == 1


@pytest.mark.parametrize("fault", ["nan", "shape", "short"])
def test_invalid_numerics_or_incomplete_repetitions_are_not_absence(fault):
    raw = raw_message()
    if fault == "nan":
        raw[0, 0] = np.nan
    if fault == "shape":
        raw = raw[:4]
    if fault == "short":
        raw = raw[:, :42]
    with pytest.raises((RuntimeError, ValueError)):
        parse(raw)


@pytest.mark.parametrize("symbol", [0, 1, 2, 3, 4])
def test_constant_class_bias_cannot_become_a_payload(symbol):
    raw = np.eye(5, dtype=np.float32)[np.full(84, symbol)].T
    assert parse(raw)[0] is None


def _fake_model(monkeypatch, outputs):
    torch = pytest.importorskip("torch")
    from remove_ai_watermarks._internal import _silentcipher_network, silentcipher

    iterator = iter(outputs)
    monkeypatch.setattr(_silentcipher_network, "magnitude", lambda samples, window: samples)
    monkeypatch.setattr(silentcipher, "HOP_LENGTH", 30)
    return silentcipher.DecoderModel(lambda samples: torch.from_numpy(next(iterator))[None, None], None, "cpu")


def test_invalid_high_consistency_phase_does_not_hide_a_foreign_payload(monkeypatch):
    from remove_ai_watermarks._internal import silentcipher

    bias = np.eye(5, dtype=np.float32)[np.zeros(84, dtype=int)].T
    model = _fake_model(monkeypatch, [bias, raw_message((5, 4, 3, 2, 1)), bias])
    result = silentcipher.read(model, np.ones(silentcipher.MIN_READ_SAMPLES))
    assert result.detected
    assert result.payload == (5, 4, 3, 2, 1)
    assert result.symbol_consistency == 1
    assert result.phase_offset == 10
    assert result.phases_checked == 2


def test_agreement_boundary_and_complete_negative_scope(monkeypatch):
    from remove_ai_watermarks._internal import silentcipher

    labels = raw_message((0, 0, 0, 0, 0), cycles=10).argmax(axis=0)
    labels[:20] = 4
    labels[21] = 4
    damaged = np.eye(5, dtype=np.float32)[labels].T
    model = _fake_model(monkeypatch, [damaged])
    assert silentcipher.read(model, np.ones(silentcipher.MIN_READ_SAMPLES)).detected
    labels[:41] = 3
    weak = np.eye(5, dtype=np.float32)[labels].T
    model = _fake_model(monkeypatch, [weak] * 3)
    result = silentcipher.read(model, np.ones(silentcipher.MIN_READ_SAMPLES))
    assert not result.detected
    assert result.phases_checked == 3


@pytest.mark.parametrize("fault", ["short", "nan", "channels"])
def test_bad_input_is_not_sent_to_the_neural_model(fault):
    from remove_ai_watermarks._internal import silentcipher

    samples = np.ones(silentcipher.MIN_READ_SAMPLES)
    if fault == "short":
        samples = samples[:-1]
    elif fault == "nan":
        samples[0] = np.nan
    else:
        samples = samples[:, None]
    with pytest.raises(ValueError, match="finite mono"):
        silentcipher.read(None, samples)


def test_invalid_numerics_do_not_trigger_a_cpu_retry(monkeypatch):
    from types import SimpleNamespace

    from remove_ai_watermarks._internal import silentcipher

    def fail(*args):
        raise silentcipher.ModelCheckError("Invalid neural values")

    monkeypatch.setattr(silentcipher, "_read_native", fail)
    monkeypatch.setattr(
        silentcipher, "load_model", lambda device: pytest.fail("Numerical failure is not a device failure")
    )
    with pytest.raises(RuntimeError, match="Invalid neural"):
        silentcipher.read(SimpleNamespace(device="mps"), np.ones(silentcipher.MIN_READ_SAMPLES))


def test_device_failure_retries_cpu_and_reports_the_actual_device(monkeypatch):
    from types import SimpleNamespace

    from remove_ai_watermarks._internal import silentcipher

    calls = []

    def read(model, values):
        calls.append(model.device)
        if model.device == "mps":
            raise RuntimeError("Unsupported operation")
        return silentcipher.SilentCipherReading(None, 0.4, None, 205, "cpu")

    monkeypatch.setattr(silentcipher, "_read_native", read)
    monkeypatch.setattr(silentcipher, "load_model", lambda device: SimpleNamespace(device=device))
    result = silentcipher.read(SimpleNamespace(device="mps"), np.ones(silentcipher.MIN_READ_SAMPLES))
    assert result.device == "cpu"
    assert calls == ["mps", "cpu"]


@pytest.mark.parametrize("corrupt", ["dec_m_0.ckpt", "hparams.yaml"])
def test_all_model_files_are_verified_before_unpickling(tmp_path, monkeypatch, corrupt):
    import hashlib
    from types import SimpleNamespace

    from remove_ai_watermarks._internal import silentcipher

    files = {name: hashlib.sha256(b"trusted").hexdigest() for name in silentcipher.MODEL_FILES}
    monkeypatch.setattr(silentcipher, "MODEL_FILES", files)
    for name in files:
        (tmp_path / name.rsplit("/", 1)[-1]).write_bytes(b"tampered" if name.endswith(corrupt) else b"trusted")
    hub = SimpleNamespace(hf_hub_download=lambda **kwargs: str(tmp_path / kwargs["filename"].rsplit("/", 1)[-1]))
    torch = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("Unpickling happened before validation"))
    monkeypatch.setattr(silentcipher, "import_module", lambda name: hub if name == "huggingface_hub" else torch)
    monkeypatch.setattr(silentcipher, "is_available", lambda: True)
    silentcipher.load_model.cache_clear()
    with pytest.raises(RuntimeError, match="pinned digest"):
        silentcipher.load_model("cpu")
    silentcipher.load_model.cache_clear()


@pytest.mark.parametrize("fault", [None, "duplicate", "non_string", "not_dict"])
def test_checkpoint_loading_is_safe_strict_and_preserves_random_state(monkeypatch, fault):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from remove_ai_watermarks._internal import _silentcipher_network, silentcipher

    state = {"module.weight": object()}
    if fault == "duplicate":
        state["weight"] = object()
    elif fault == "non_string":
        state[3] = object()
    elif fault == "not_dict":
        state = []
    calls = []

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return state

    def fork(*, devices):
        assert devices == []
        calls.append("fork")
        return nullcontext()

    def load_state(values, *, strict):
        assert strict is True
        assert set(values) == {"weight"}
        calls.append("load")

    decoder = SimpleNamespace(load_state_dict=load_state)
    decoder.to = lambda device: decoder
    torch = SimpleNamespace(
        load=load,
        random=SimpleNamespace(fork_rng=fork),
        hann_window=lambda *args, **kwargs: None,
        float32="float32",
    )
    monkeypatch.setattr(silentcipher, "is_available", lambda: True)
    monkeypatch.setattr(silentcipher, "_download_files", lambda: {"44_1_khz/73999_iteration/dec_m_0.ckpt": "pinned"})
    monkeypatch.setattr(silentcipher, "import_module", lambda name: torch)

    def build(*, message_dim, message_band_size):
        assert (message_dim, message_band_size) == (5, 1024)
        return decoder

    monkeypatch.setattr(_silentcipher_network, "build_decoder", build)
    silentcipher.load_model.cache_clear()
    try:
        if fault:
            with pytest.raises(RuntimeError, match=r"parameter|dictionary"):
                silentcipher.load_model("cpu")
            assert not calls
        else:
            assert silentcipher.load_model("cpu").decoder is decoder
            assert calls == ["fork", "load"]
    finally:
        silentcipher.load_model.cache_clear()


def test_negative_reports_strong_invalid_patterns_without_accepting_them(monkeypatch):
    from remove_ai_watermarks._internal import silentcipher

    bias = np.eye(5, dtype=np.float32)[np.zeros(84, dtype=int)].T
    model = _fake_model(monkeypatch, [bias] * 3)
    result = silentcipher.read(model, np.ones(silentcipher.MIN_READ_SAMPLES))
    assert not result.detected
    assert result.symbol_consistency == 1
    assert result.phases_checked == 3

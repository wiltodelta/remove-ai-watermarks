"""Public Perth package trust gates and named presence readings."""

from __future__ import annotations

import hashlib
import importlib.machinery
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import perth


@pytest.mark.parametrize("fault", ["source", "checkpoint", "config", "extra-checkpoint"])
def test_untrusted_package_never_reaches_the_model(tmp_path, monkeypatch, fault):
    package = tmp_path / "perth"
    directory = package / "perth_net/pretrained/implicit"
    directory.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("raise AssertionError('untrusted code executed')\n")
    digest = hashlib.sha256()
    digest.update(b"__init__.py\0")
    digest.update(hashlib.sha256(source.read_bytes()).digest())
    monkeypatch.setattr(perth, "SOURCE_TREE_SHA256", digest.hexdigest())
    files = {name: hashlib.sha256(b"fixture").hexdigest() for name in perth._MODEL_FILES}
    monkeypatch.setattr(perth, "_MODEL_FILES", files)
    for name in files:
        (directory / name).write_bytes(b"fixture")
    monkeypatch.setattr(
        perth.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(source)),
    )
    monkeypatch.setattr(perth.importlib.metadata, "version", lambda name: "1.0.1")
    if fault == "source":
        source.write_text("raise AssertionError('modified source')\n")
    elif fault == "extra-checkpoint":
        (directory / "perth_net_999999.pth.tar").write_bytes(b"fixture")
    else:
        name = "hparams.yaml" if fault == "config" else "perth_net_250000.pth.tar"
        (directory / name).write_bytes(b"modified")
    monkeypatch.setattr(perth, "import_module", lambda name: pytest.fail("Model imported before trust checks"))
    perth.load_model.cache_clear()
    with pytest.raises(RuntimeError, match="pinned audio-perth"):
        perth.load_model()
    perth.load_model.cache_clear()


@pytest.mark.parametrize("samples", [np.zeros((16000, 2)), np.zeros(15999), np.full(16000, np.nan)])
def test_incompatible_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="finite mono"):
        perth.read(object(), samples)


@pytest.mark.parametrize("score", [[], [0, 1], [np.nan], [np.inf]])
def test_invalid_model_confidence_is_not_a_negative(score):
    pytest.importorskip("torch")
    model = _model(score)
    with pytest.raises(RuntimeError, match="invalid presence"):
        perth.read(model, np.zeros(16000))


@pytest.mark.parametrize(
    ("score", "expected"), [(-0.5, False), (0.0, False), (0.5, False), (0.50001, True), (1.0, True), (1.5, True)]
)
def test_presence_threshold_matches_the_public_model(score, expected):
    pytest.importorskip("torch")
    model = _model([score])
    reading = perth.read(model, np.zeros(16000))
    assert reading.detected is expected
    if score < 0:
        assert reading.confidence == 0
    elif score > 1:
        assert reading.confidence == 1


def _model(score):
    import torch

    return SimpleNamespace(
        hp=SimpleNamespace(sample_rate=16000),
        device="cpu",
        ap=SimpleNamespace(signal_to_magphase=lambda signal: (signal, None)),
        decoder=lambda signal: torch.tensor([score], dtype=torch.float32),
    )

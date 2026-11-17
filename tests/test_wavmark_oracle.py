"""Research import compatibility and pinned model loading without downloads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import wavmark

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import wavmark_oracle as oracle  # noqa: E402


def test_foreign_message_is_still_detected():
    reading = oracle.WavMarkReading((0,) * 16, (0,))
    assert reading.detected is True
    assert reading.payload != oracle.MESSAGE


@pytest.mark.parametrize("samples", [np.zeros(15999), np.full(32000, np.nan), np.zeros((32000, 2))])
def test_incompatible_input_is_not_a_negative(monkeypatch, samples):
    monkeypatch.setitem(sys.modules, "wavmark", SimpleNamespace())
    with pytest.raises(ValueError, match="finite mono samples"):
        oracle.read(object(), samples)


def test_checkpoint_is_checked_before_model_loading(tmp_path, monkeypatch):
    wavmark.load_model.cache_clear()
    loaded = []
    module = SimpleNamespace(load_model=lambda path: loaded.append(path))
    monkeypatch.setitem(sys.modules, "wavmark", module)
    path = tmp_path / "invalid.pkl"
    path.write_bytes(b"invalid checkpoint")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda **kw: str(path)))
    monkeypatch.setattr(wavmark, "import_failure", lambda: None)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        oracle.load_model()
    assert loaded == []

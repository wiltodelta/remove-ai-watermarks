"""Complete public-model window coverage and raw-output validation."""

from __future__ import annotations

import importlib.machinery
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from remove_ai_watermarks._internal import wavmark

PATTERN = (1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0)


def _model(monkeypatch, positive_at=0, *, nonfinite=False):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "wavmark", SimpleNamespace(wm_add_util=SimpleNamespace(fix_pattern=PATTERN)))
    seen = []

    class Model:
        def parameters(self):
            yield torch.zeros(1)

        def decode(self, batch):
            starts = batch[:, 0].cpu().numpy()
            seen.extend(int(x) for x in starts)
            result = np.zeros((len(batch), 32), dtype=np.float32)
            for i, start in enumerate(starts):
                if start == positive_at:
                    result[i, :16] = PATTERN
            if nonfinite:
                result[0, 0] = np.nan
            return torch.from_numpy(result)

    return Model(), seen


@pytest.mark.parametrize("length", [16000, 20000, 20001])
def test_final_valid_window_is_always_checked(monkeypatch, length):
    end = length - 16000
    model, seen = _model(monkeypatch, positive_at=end)
    reading = wavmark.read(model, np.arange(length, dtype=np.float32))
    assert reading.detected is True
    assert reading.payload == (0,) * 16
    assert reading.start_samples == (end,)
    assert seen[-1] == end
    assert (end // 800) * 800 in seen


def test_nonfinite_raw_output_is_not_converted_to_negative_bits(monkeypatch):
    model, _ = _model(monkeypatch, nonfinite=True)
    with pytest.raises(RuntimeError, match="invalid raw message"):
        wavmark.read(model, np.zeros(16000, dtype=np.float32))


def test_untrusted_package_source_is_rejected_without_import(tmp_path, monkeypatch):
    package = tmp_path / "wavmark"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("raise AssertionError('must never execute')\n")
    monkeypatch.setattr(
        wavmark.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec("wavmark", loader=None, origin=str(source)),
    )
    monkeypatch.setattr(wavmark.importlib.metadata, "version", lambda name: "0.0.3")
    failure = wavmark.import_failure()
    assert isinstance(failure, RuntimeError)
    assert "source differs" in str(failure)

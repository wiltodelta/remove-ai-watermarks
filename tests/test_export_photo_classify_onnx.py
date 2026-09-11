"""Vision-only photo-classifier export helpers, without the real checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _exporter() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_photo_classify_onnx.py"
    spec = importlib.util.spec_from_file_location("export_photo_classify_onnx", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vision_embedding_drops_the_text_tower_and_normalizes() -> None:
    torch = pytest.importorskip("torch")

    class FakeVision(torch.nn.Module):
        def forward(self, *, pixel_values: Any) -> Any:
            return SimpleNamespace(pooler_output=pixel_values)

    clip = SimpleNamespace(
        vision_model=FakeVision(),
        visual_projection=torch.nn.Identity(),
        text_model=object(),
    )
    model = _exporter().VisionEmbedding(SimpleNamespace(clip=clip))

    output = model(torch.tensor([[3.0, 4.0]]))

    assert set(dict(model.named_children())) == {"vision_model", "visual_projection"}
    assert torch.equal(output, torch.tensor([[0.6, 0.8]]))

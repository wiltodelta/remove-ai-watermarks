"""Path resolution for Hub photo-classify publish, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _publish() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_photo_classify_hf.py"
    spec = importlib.util.spec_from_file_location("publish_photo_classify_hf", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hub_id_is_raiw_models() -> None:
    assert _publish().HUB_REPO == "wiltodelta/raiw-models"


def test_weight_paths_accepts_run1_layout(tmp_path: Path) -> None:
    (tmp_path / "clip-l-ft.pt").write_bytes(b"clip")
    (tmp_path / "probe-weights-clip-l-ft.npz").write_bytes(b"probe")
    run1 = tmp_path / "run1"
    run1.mkdir()
    (run1 / "detector.pt").write_bytes(b"det")
    (run1 / "provider.pt").write_bytes(b"prov")
    found = _publish().weight_paths(tmp_path)
    assert found["detector.pt"] == run1 / "detector.pt"
    assert found["clip-l-ft.pt"] == tmp_path / "clip-l-ft.pt"


def test_weight_paths_requires_every_file(tmp_path: Path) -> None:
    (tmp_path / "clip-l-ft.pt").write_bytes(b"clip")
    with pytest.raises(SystemExit, match=r"missing probe-weights-clip-l-ft\.npz"):
        _publish().weight_paths(tmp_path)


def test_stage_card_copies_tracked_files(tmp_path: Path) -> None:
    dest = tmp_path / "stage"
    _publish().stage_card(dest)
    assert (dest / "README.md").is_file()
    assert (dest / "operating-point.json").is_file()
    text = (dest / "operating-point.json").read_text()
    assert "clip-l-ft.pt" in text

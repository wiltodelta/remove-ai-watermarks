"""Path resolution and SHA-256 checks for Hub photo-classify publish, no network."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import publish_photo_classify_hf as publish  # noqa: E402


def test_hub_ids_match_the_runtime_loader() -> None:
    # The workflow runs this script with --no-project, so it cannot import the package constants.
    from remove_ai_watermarks.classify import (
        PROVIDER_FILE,
        PROVIDER_SHA256,
        RECEIPT_GATE_FILE,
        RECEIPT_GATE_STABLE_FILE,
        WEIGHTS_REPO,
    )

    assert publish.HUB_REPO == WEIGHTS_REPO
    assert publish.PROVIDER_FILE == PROVIDER_FILE
    assert publish.PROVIDER_SHA256 == PROVIDER_SHA256
    assert publish.GATE_FILE == RECEIPT_GATE_FILE
    assert publish.GATE_STABLE_FILE == RECEIPT_GATE_STABLE_FILE
    assert publish.GATE_SOURCE.is_file()


def test_publish_freeze_includes_both_clip_runtimes() -> None:
    from remove_ai_watermarks.classify import _WEIGHT_FILES, CLIP_ONNX_FILE, WEIGHTS_ALLOW_PATTERNS

    assert publish.CLIP_FILE in publish.WEIGHT_FILES
    assert publish.CLIP_ONNX_FILE in publish.WEIGHT_FILES
    assert set(publish.WEIGHT_FILES) == {*_WEIGHT_FILES, CLIP_ONNX_FILE}
    assert set(publish.WEIGHT_FILES) <= set(WEIGHTS_ALLOW_PATTERNS)
    assert "provider.pt" in publish.RETIRED_FILES


def test_weight_paths_accepts_run1_layout(tmp_path: Path) -> None:
    (tmp_path / "clip-l-ft.pt").write_bytes(b"clip")
    (tmp_path / "clip-l-ft-vision-fp32.onnx").write_bytes(b"onnx")
    (tmp_path / "probe-weights-clip-l-ft.npz").write_bytes(b"probe")
    (tmp_path / "multi-provider-source-v6.npz").write_bytes(b"source-v6")
    run1 = tmp_path / "run1"
    run1.mkdir()
    (run1 / "detector.pt").write_bytes(b"det")
    found = publish.weight_paths(tmp_path)
    assert found["detector.pt"] == run1 / "detector.pt"
    assert found["clip-l-ft.pt"] == tmp_path / "clip-l-ft.pt"
    assert found["multi-provider-source-v6.npz"] == tmp_path / "multi-provider-source-v6.npz"


def test_stage_weights_rejects_mismatched_provider_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for name in publish.WEIGHT_FILES:
        (src / name).write_bytes(f"payload-{name}".encode())
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit, match=r"SHA-256 mismatch for multi-provider-source-v6\.npz"):
        publish.stage_weights(dest, src)

    good_sha = hashlib.sha256(b"payload-multi-provider-source-v6.npz").hexdigest()
    monkeypatch.setattr(publish, "PROVIDER_SHA256", good_sha)
    publish.stage_weights(dest, src)
    assert (dest / "multi-provider-source-v6.npz").is_file()


def test_weight_paths_requires_every_file(tmp_path: Path) -> None:
    (tmp_path / "clip-l-ft.pt").write_bytes(b"clip")
    (tmp_path / "clip-l-ft-vision-fp32.onnx").write_bytes(b"onnx")
    with pytest.raises(SystemExit, match=r"missing probe-weights-clip-l-ft\.npz"):
        publish.weight_paths(tmp_path)


def test_stage_card_copies_tracked_files(tmp_path: Path) -> None:
    dest = tmp_path / "stage"
    publish.stage_card(dest)
    assert (dest / "README.md").is_file()
    assert (dest / "operating-point.json").is_file()
    text = (dest / "operating-point.json").read_text()
    assert "clip-l-ft.pt" in text
    assert "multi-provider-source-v6.npz" in text

"""Local staging checks for the source-classifier Hub publisher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import publish_source_classify_hf as publish  # noqa: E402


def test_hub_ids_match_the_runtime_loader() -> None:
    # The workflow runs this script with --no-project, so it cannot import the package constants.
    from remove_ai_watermarks.source_classify import MODEL_FILE, WEIGHTS_REPO

    assert publish.HUB_REPO == WEIGHTS_REPO
    assert publish.MODEL_FILE == MODEL_FILE


def test_stage_release_copies_only_public_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "multi-provider-source-v4.npz").write_bytes(b"weights")
    destination = tmp_path / "stage"
    publish.stage_release(destination, source)
    assert sorted(path.name for path in destination.iterdir()) == sorted(publish.PUBLIC_FILES)


def test_stage_release_requires_the_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"multi-provider-source-v4\.npz"):
        publish.stage_release(tmp_path / "stage", tmp_path)


def test_card_sha_matches_the_metrics_manifest() -> None:
    metrics = json.loads((publish.CARD_DIR / "metrics.json").read_text())
    card = (publish.CARD_DIR / "README.md").read_text()
    assert metrics["artifact_sha256"] in card

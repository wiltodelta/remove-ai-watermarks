from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_routed_detector as detector
import synthid_routed_expert_bank as bank


def test_detect_path_routes_one_scored_record(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fixture")

    def score_path(path: Path) -> dict[str, object]:
        assert path == image_path
        return {
            "id": "a" * 64,
            "path": str(path),
            "width": 1024,
            "height": 1024,
            "observations": [
                {
                    "name": bank.synthid_detector.DETECTOR_ID,
                    "supported": True,
                    "score": 0.5,
                },
                {
                    "name": bank.synthid_detector.REGISTERED_DETECTOR_ID,
                    "supported": True,
                    "score": 0.0,
                },
                {
                    "name": bank.synthid_detector.LARGE_DETECTOR_ID,
                    "supported": False,
                    "score": None,
                },
            ],
        }

    monkeypatch.setattr(detector, "score_path", score_path)

    result = detector.detect_path(image_path)

    assert result["result"]["verdict"] == "abstain"
    assert result["result"]["reason"] == "fixed_only_ambiguous"


def test_cli_writes_combined_hash_pinned_report(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    report_path = tmp_path / "report.json"
    image_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        detector,
        "detect_path",
        lambda path: {
            "id": "b" * 64,
            "path": str(path),
            "width": 1024,
            "height": 1024,
            "observations": [],
            "result": {"verdict": "detected", "reason": "registered_threshold_crossed"},
        },
    )

    result = CliRunner().invoke(
        detector.main,
        [str(image_path), "--report-out", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {"detected": 1, "abstain": 0}
    assert report["records"][0]["id"] == "b" * 64

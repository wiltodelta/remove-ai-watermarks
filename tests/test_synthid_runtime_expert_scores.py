from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_runtime_expert_scores as scorer


def test_unsupported_geometry_emits_no_synthetic_scores() -> None:
    observations = scorer.score_pixels(np.zeros((64, 64, 3), dtype=np.uint8))

    assert observations == [
        {"name": scorer.FIXED_EXPERT_NAME, "supported": False, "score": None},
        {"name": scorer.REGISTERED_EXPERT_NAME, "supported": False, "score": None},
        {"name": scorer.LARGE_EXPERT_NAME, "supported": False, "score": None},
    ]


def test_supported_image_scores_each_expert_once(monkeypatch) -> None:
    calls = {"fixed": 0, "registered": 0}

    def detect(path, *, image, register_scale=False):
        branch = "registered" if register_scale else "fixed"
        calls[branch] += 1
        return scorer.synthid_detector.SynthIDDetection(
            status="detected",
            width=1024,
            height=1024,
            score=1.5 if register_scale else 0.25,
            threshold=1.0 if register_scale else 0.17,
        )

    monkeypatch.setattr(scorer.synthid_detector, "detect_synthid", detect)

    observations = scorer.score_pixels(np.zeros((1024, 1024, 3), dtype=np.uint8))

    assert observations == [
        {"name": scorer.FIXED_EXPERT_NAME, "supported": True, "score": 0.25},
        {"name": scorer.REGISTERED_EXPERT_NAME, "supported": True, "score": 1.5},
        {"name": scorer.LARGE_EXPERT_NAME, "supported": False, "score": None},
    ]
    assert calls == {"fixed": 1, "registered": 1}


def test_pixels_must_be_rgb_uint8() -> None:
    with np.testing.assert_raises_regex(ValueError, "RGB uint8"):
        scorer.score_pixels(np.zeros((64, 64, 3), dtype=np.float32))


def test_cli_writes_hash_pinned_observation_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    report_path = tmp_path / "scores.json"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(image_path)

    result = CliRunner().invoke(scorer.main, [str(image_path), "--report-out", str(report_path)])

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["experts"] == [
        scorer.FIXED_EXPERT_NAME,
        scorer.REGISTERED_EXPERT_NAME,
        scorer.LARGE_EXPERT_NAME,
    ]
    assert len(report["records"][0]["id"]) == 64
    assert report["records"][0]["width"] == 64
    assert all(not observation["supported"] for observation in report["records"][0]["observations"])


def test_large_default_is_not_mislabeled_as_fixed(monkeypatch) -> None:
    def detect(path, *, image, register_scale=False):
        detector_id = scorer.REGISTERED_EXPERT_NAME if register_scale else scorer.LARGE_EXPERT_NAME
        return scorer.synthid_detector.SynthIDDetection(
            status="unsupported" if register_scale else "detected",
            width=4096,
            height=4096,
            score=None if register_scale else 1.2,
            threshold=1.0,
            detector=detector_id,
        )

    monkeypatch.setattr(scorer.synthid_detector, "detect_synthid", detect)

    observations = scorer.score_pixels(np.zeros((4096, 4096, 3), dtype=np.uint8))

    assert observations == [
        {"name": scorer.FIXED_EXPERT_NAME, "supported": False, "score": None},
        {"name": scorer.REGISTERED_EXPERT_NAME, "supported": False, "score": None},
        {"name": scorer.LARGE_EXPERT_NAME, "supported": True, "score": 1.2},
    ]

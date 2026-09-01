from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_ensemble_detector as detector
from synthid_color_space_probe import ColorPhaseScore


def _config(tmp_path: Path) -> detector.EnsembleConfig:
    return detector.EnsembleConfig(
        width=64,
        height=64,
        rgb_model_path=tmp_path / "rgb.npz",
        rgb_model_sha256="0" * 64,
        rgb_evidence_threshold=0.3,
        rgb_active_threshold=0.5,
        hsv_model_path=tmp_path / "hsv.npz",
        hsv_model_sha256="1" * 64,
        hsv_sv_evidence_threshold=0.3,
        hsv_active_threshold=0.5,
    )


def _score(
    *, color_space: str, evidence: float, active: float, channels: tuple[float, float, float]
) -> ColorPhaseScore:
    return ColorPhaseScore(
        path="fixture.png",
        color_space=color_space,
        phase_score=0.8,
        active_weight_fraction=active,
        evidence_score=evidence,
        channel_evidence=channels,
        selected_peak_counts=(80, 88, 88),
        peak_count=256,
    )


def test_positive_requires_both_branches_and_support(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rgb = _score(color_space="rgb", evidence=0.4, active=0.7, channels=(0.1, 0.1, 0.2))
    hsv = _score(color_space="hsv", evidence=0.45, active=0.8, channels=(0.05, 0.2, 0.2))

    verdict = detector.classify_scores(Path("fixture.png"), rgb, hsv, config)

    assert verdict.verdict == "positive"
    assert verdict.reason == "ensemble_pass"


def test_low_support_abstains_even_when_scores_pass(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rgb = _score(color_space="rgb", evidence=0.4, active=0.49, channels=(0.1, 0.1, 0.2))
    hsv = _score(color_space="hsv", evidence=0.45, active=0.8, channels=(0.05, 0.2, 0.2))

    verdict = detector.classify_scores(Path("fixture.png"), rgb, hsv, config)

    assert verdict.verdict == "abstain"
    assert verdict.reason == "insufficient_support"


def test_branch_disagreement_abstains(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rgb = _score(color_space="rgb", evidence=0.4, active=0.8, channels=(0.1, 0.1, 0.2))
    hsv = _score(color_space="hsv", evidence=0.2, active=0.8, channels=(0.02, 0.1, 0.08))

    verdict = detector.classify_scores(Path("fixture.png"), rgb, hsv, config)

    assert verdict.verdict == "abstain"
    assert verdict.reason == "branch_disagreement"


def test_unsupported_geometry_abstains_without_scoring(tmp_path: Path) -> None:
    config = _config(tmp_path)
    image_path = tmp_path / "small.png"
    Image.new("RGB", (32, 32)).save(image_path)

    verdict = detector.detect_image(image_path, config, None, None)  # type: ignore[arg-type]

    assert verdict.verdict == "abstain"
    assert verdict.reason == "unsupported_geometry"
    assert verdict.rgb_evidence is None

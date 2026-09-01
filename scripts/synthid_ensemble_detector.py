"""Run a positive-only exact-geometry SynthID research detector.

The detector requires independently frozen RGB and HSV phase models. It emits
``positive`` only when both branches and both support gates pass; every other
case is ``abstain``. It never claims that SynthID is absent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from PIL import Image
from synthid_color_space_probe import ColorPhaseModel, ColorPhaseScore, load_model, score_image
from synthid_research_manifest import artifact_sha256

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleConfig:
    """Frozen paths, hashes, geometry, and positive thresholds."""

    width: int
    height: int
    rgb_model_path: Path
    rgb_model_sha256: str
    rgb_evidence_threshold: float
    rgb_active_threshold: float
    hsv_model_path: Path
    hsv_model_sha256: str
    hsv_sv_evidence_threshold: float
    hsv_active_threshold: float


@dataclass(frozen=True)
class EnsembleVerdict:
    """Positive-only verdict with the evidence needed to audit it."""

    path: str
    verdict: str
    reason: str
    rgb_evidence: float | None
    rgb_active_support: float | None
    hsv_sv_evidence: float | None
    hsv_active_support: float | None


def load_config(path: Path) -> EnsembleConfig:
    """Load a frozen epoch manifest and verify both model artifacts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("verdict_scope") != "positive-only exact-geometry research detector":
        raise ValueError("config is not a positive-only exact-geometry detector")
    rgb = payload["rgb_model"]
    hsv = payload["hsv_model"]
    geometry = payload["geometry"]
    config = EnsembleConfig(
        width=int(geometry["width"]),
        height=int(geometry["height"]),
        rgb_model_path=Path(rgb["path"]),
        rgb_model_sha256=str(rgb["sha256"]),
        rgb_evidence_threshold=float(rgb["evidence_threshold"]),
        rgb_active_threshold=float(rgb["active_support_threshold"]),
        hsv_model_path=Path(hsv["path"]),
        hsv_model_sha256=str(hsv["sha256"]),
        hsv_sv_evidence_threshold=float(hsv["sv_evidence_threshold"]),
        hsv_active_threshold=float(hsv["active_support_threshold"]),
    )
    if config.width < 64 or config.height < 64:
        raise ValueError("invalid detector geometry")
    for model_path, expected_hash in (
        (config.rgb_model_path, config.rgb_model_sha256),
        (config.hsv_model_path, config.hsv_model_sha256),
    ):
        if not model_path.is_file():
            raise ValueError(f"model artifact does not exist: {model_path}")
        if artifact_sha256(model_path) != expected_hash:
            raise ValueError(f"model artifact hash mismatch: {model_path}")
    return config


def load_models(config: EnsembleConfig) -> tuple[ColorPhaseModel, ColorPhaseModel]:
    """Load and cross-check the RGB and HSV models in CONFIG."""
    rgb_model = load_model(config.rgb_model_path)
    hsv_model = load_model(config.hsv_model_path)
    if rgb_model.color_space != "rgb" or hsv_model.color_space != "hsv":
        raise ValueError("detector requires one RGB model and one HSV model")
    expected_geometry = (config.height, config.width)
    if (rgb_model.height, rgb_model.width) != expected_geometry:
        raise ValueError("RGB model geometry does not match config")
    if (hsv_model.height, hsv_model.width) != expected_geometry:
        raise ValueError("HSV model geometry does not match config")
    return rgb_model, hsv_model


def classify_scores(
    path: Path,
    rgb_score: ColorPhaseScore,
    hsv_score: ColorPhaseScore,
    config: EnsembleConfig,
) -> EnsembleVerdict:
    """Apply CONFIG's positive-only rule to precomputed branch scores."""
    hsv_sv_evidence = float(sum(hsv_score.channel_evidence[1:]))
    rgb_support = rgb_score.active_weight_fraction >= config.rgb_active_threshold
    hsv_support = hsv_score.active_weight_fraction >= config.hsv_active_threshold
    rgb_pass = rgb_score.evidence_score >= config.rgb_evidence_threshold
    hsv_pass = hsv_sv_evidence >= config.hsv_sv_evidence_threshold
    if rgb_support and hsv_support and rgb_pass and hsv_pass:
        verdict = "positive"
        reason = "ensemble_pass"
    elif not rgb_support or not hsv_support:
        verdict = "abstain"
        reason = "insufficient_support"
    elif rgb_pass != hsv_pass:
        verdict = "abstain"
        reason = "branch_disagreement"
    else:
        verdict = "abstain"
        reason = "below_positive_threshold"
    return EnsembleVerdict(
        path=str(path),
        verdict=verdict,
        reason=reason,
        rgb_evidence=rgb_score.evidence_score,
        rgb_active_support=rgb_score.active_weight_fraction,
        hsv_sv_evidence=hsv_sv_evidence,
        hsv_active_support=hsv_score.active_weight_fraction,
    )


def detect_image(
    path: Path,
    config: EnsembleConfig,
    rgb_model: ColorPhaseModel,
    hsv_model: ColorPhaseModel,
) -> EnsembleVerdict:
    """Evaluate PATH or abstain when its geometry is unsupported."""
    with Image.open(path) as image:
        if image.size != (config.width, config.height):
            return EnsembleVerdict(
                path=str(path),
                verdict="abstain",
                reason="unsupported_geometry",
                rgb_evidence=None,
                rgb_active_support=None,
                hsv_sv_evidence=None,
                hsv_active_support=None,
            )
    return classify_scores(
        path,
        score_image(path, rgb_model),
        score_image(path, hsv_model),
        config,
    )


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(config_path: Path, images: tuple[Path, ...], report_out: Path) -> None:
    """Score IMAGES with the frozen positive-only detector CONFIG_PATH."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(config_path)
    rgb_model, hsv_model = load_models(config)
    verdicts = [detect_image(image, config, rgb_model, hsv_model) for image in images]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "positive_count": sum(verdict.verdict == "positive" for verdict in verdicts),
                "abstain_count": sum(verdict.verdict == "abstain" for verdict in verdicts),
                "verdicts": [asdict(verdict) for verdict in verdicts],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d positive-only detector verdicts: %s", len(verdicts), report_out)


if __name__ == "__main__":
    main()

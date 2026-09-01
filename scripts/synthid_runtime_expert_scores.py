"""Export fixed, large, and scale-registered SynthID observations for images.

The output is an input manifest for ``synthid_conformal_cascade.py``. All
experts consume decoded RGB pixels only. Unsupported geometry is recorded
explicitly and never represented by a synthetic score.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import click
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthid_pixel_attack import load_rgb  # noqa: E402
from synthid_research_manifest import artifact_sha256  # noqa: E402
from synthid_runtime import synthid_detector  # noqa: E402

log = logging.getLogger(__name__)

FIXED_EXPERT_NAME = synthid_detector.DETECTOR_ID
REGISTERED_EXPERT_NAME = synthid_detector.REGISTERED_DETECTOR_ID
LARGE_EXPERT_NAME = synthid_detector.LARGE_DETECTOR_ID


class ExpertScore(TypedDict):
    """One JSON-safe runtime expert observation."""

    name: str
    supported: bool
    score: float | None


class ScoredImage(TypedDict):
    """One hash-pinned image with every runtime expert observation."""

    id: str
    path: str
    width: int
    height: int
    observations: list[ExpertScore]


def _observation(name: str, supported: bool, score: float | None) -> ExpertScore:
    return {"name": name, "supported": supported, "score": score}


def score_pixels(pixels: NDArray[np.uint8]) -> list[ExpertScore]:
    """Return explicit fixed, registered, and large observations for RGB PIXELS."""
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("pixels must be an RGB uint8 array")
    bgr_pixels = np.ascontiguousarray(pixels[:, :, ::-1])
    native = synthid_detector.detect_synthid("decoded-image", image=bgr_pixels, register_scale=False)
    registered = synthid_detector.detect_synthid("decoded-image", image=bgr_pixels, register_scale=True)
    fixed = _observation(FIXED_EXPERT_NAME, False, None)
    large = _observation(LARGE_EXPERT_NAME, False, None)
    native_observation = _observation(native.detector, native.status != "unsupported", native.score)
    if native.detector == FIXED_EXPERT_NAME:
        fixed = native_observation
    elif native.detector == LARGE_EXPERT_NAME:
        large = native_observation
    else:
        raise RuntimeError(f"unexpected default SynthID expert: {native.detector}")
    return [
        fixed,
        _observation(REGISTERED_EXPERT_NAME, registered.status != "unsupported", registered.score),
        large,
    ]


def score_path(path: Path) -> ScoredImage:
    """Decode PATH once and return one hash-pinned observation record."""
    pixels = load_rgb(path)
    height, width = pixels.shape[:2]
    return {
        "id": artifact_sha256(path),
        "path": str(path),
        "width": width,
        "height": height,
        "observations": score_pixels(pixels),
    }


@click.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(images: tuple[Path, ...], report_out: Path) -> None:
    """Score IMAGES with every shipped pixel expert."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = [score_path(path) for path in images]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experts": [FIXED_EXPERT_NAME, REGISTERED_EXPERT_NAME, LARGE_EXPERT_NAME],
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d three-expert score records: %s", len(records), report_out)


if __name__ == "__main__":
    main()

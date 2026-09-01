"""Build deterministic, pixel-only SynthID attack candidates and controls.

The generated variants use quantization, resampling, and a smooth sub-pixel
warp. No generative model or image synthesis stage is involved. The command
also emits a norm-matched random-noise control so an oracle change cannot be
attributed to pixel distance alone.

This is a research harness. A candidate is successful only when the matching
provider oracle changes from detected to not detected while the crop-only
control remains detected.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import cv2
import numpy as np
from invisible_quality_audit import _ssim
from PIL import Image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FidelityMeasurement:
    """Paired pixel metrics for one attack candidate."""

    name: str
    path: str
    width: int
    height: int
    psnr_db: float
    ssim: float
    changed_pixel_fraction: float
    residual_rms: float
    residual_max: float


def load_rgb(path: Path) -> np.ndarray:
    """Load PATH as uint8 RGB pixels."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def crop_visible_badge(pixels: np.ndarray, margin: int) -> np.ndarray:
    """Remove the bottom and right margins that contain the visible badge."""
    height, width = pixels.shape[:2]
    if margin < 0 or margin >= min(height, width):
        raise ValueError("crop margin must be nonnegative and smaller than the image")
    if margin == 0:
        return pixels.copy()
    return pixels[: height - margin, : width - margin].copy()


def quantize(pixels: np.ndarray, step: int) -> np.ndarray:
    """Round RGB samples to the nearest multiple of STEP."""
    if step < 2 or step > 64:
        raise ValueError("quantization step must be between 2 and 64")
    values = np.rint(pixels.astype(np.float64) / step) * step
    return np.clip(values, 0, 255).astype(np.uint8)


def smooth_warp(pixels: np.ndarray, *, amplitude: float, sigma: float, seed: int) -> np.ndarray:
    """Apply a deterministic smooth sub-pixel displacement field."""
    if amplitude < 0.0 or sigma <= 0.0:
        raise ValueError("warp amplitude must be nonnegative and sigma positive")
    height, width = pixels.shape[:2]
    rng = np.random.default_rng(seed)
    fields: list[np.ndarray] = []
    for _ in range(2):
        noise = rng.normal(size=(height, width)).astype(np.float32)
        field = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
        field_std = float(np.std(field))
        fields.append(np.zeros_like(field) if field_std == 0.0 else field * (amplitude / field_std))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        pixels,
        xx + fields[0],
        yy + fields[1],
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def resize_squeeze(pixels: np.ndarray, factor: float) -> np.ndarray:
    """Downsample and restore the original geometry without synthesis."""
    if not 0.5 <= factor < 1.0:
        raise ValueError("resize factor must be in [0.5, 1.0)")
    height, width = pixels.shape[:2]
    reduced = cv2.resize(
        pixels,
        (max(1, round(width * factor)), max(1, round(height * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LANCZOS4)


def jpeg_round_trip(pixels: np.ndarray, quality: int) -> np.ndarray:
    """Apply one in-memory JPEG encode/decode while returning RGB pixels."""
    if quality < 1 or quality > 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def norm_matched_noise(reference: np.ndarray, target: np.ndarray, *, seed: int) -> np.ndarray:
    """Return random RGB noise with approximately TARGET's residual RMS."""
    target_residual = target.astype(np.float64) - reference.astype(np.float64)
    target_rms = float(np.sqrt(np.mean(np.square(target_residual))))
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=reference.shape)
    noise *= target_rms / (float(np.sqrt(np.mean(np.square(noise)))) + 1e-12)
    return np.clip(np.rint(reference.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def measure(reference: np.ndarray, candidate: np.ndarray, *, name: str, path: Path) -> FidelityMeasurement:
    """Measure paired fidelity between equal-shaped RGB arrays."""
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    residual = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(residual)))
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    return FidelityMeasurement(
        name=name,
        path=str(path),
        width=int(reference.shape[1]),
        height=int(reference.shape[0]),
        psnr_db=psnr,
        ssim=float(_ssim(reference_gray, candidate_gray)),
        changed_pixel_fraction=float(np.mean(np.any(residual != 0.0, axis=2))),
        residual_rms=float(math.sqrt(mse)),
        residual_max=float(np.max(np.abs(residual))),
    )


def build_candidates(source: np.ndarray) -> dict[str, np.ndarray]:
    """Build the preregistered attack batch from cropped SOURCE pixels."""
    candidates: dict[str, np.ndarray] = {
        "control-crop": source.copy(),
        "quantize-2": quantize(source, 2),
        "quantize-4": quantize(source, 4),
        "quantize-8": quantize(source, 8),
        "warp-035": smooth_warp(source, amplitude=0.35, sigma=48.0, seed=20260809),
    }
    combo = smooth_warp(source, amplitude=0.55, sigma=48.0, seed=20260810)
    combo = resize_squeeze(combo, 0.96)
    combo = quantize(combo, 4)
    combo = jpeg_round_trip(combo, 96)
    candidates["combo-mild"] = combo
    candidates["sham-combo-rms"] = norm_matched_noise(source, combo, seed=20260811)
    return candidates


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--crop-margin", type=click.IntRange(min=0), default=160, show_default=True)
def main(source: Path, output_dir: Path, crop_margin: int) -> None:
    """Write a frozen pixel-only attack batch for SOURCE into OUTPUT_DIR."""
    reference = crop_visible_badge(load_rgb(source), crop_margin)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[FidelityMeasurement] = []
    for name, pixels in build_candidates(reference).items():
        path = output_dir / f"{name}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
        measurements.append(measure(reference, pixels, name=name, path=path))
    report_path = output_dir / "fidelity.json"
    payload = {
        "source": str(source),
        "crop_margin": crop_margin,
        "variants": [asdict(row) for row in measurements],
    }
    report_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d candidates and fidelity report: %s", len(measurements), report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

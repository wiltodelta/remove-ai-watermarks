from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_pixel_attack as attack


def _fixture() -> np.ndarray:
    yy, xx = np.mgrid[:128, :160]
    channels = [
        80 + xx * 0.4 + yy * 0.2,
        60 + xx * 0.3 + yy * 0.5,
        40 + xx * 0.6 + yy * 0.1,
    ]
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


def test_quantization_is_bounded_and_deterministic() -> None:
    pixels = _fixture()

    first = attack.quantize(pixels, 4)
    second = attack.quantize(pixels, 4)

    assert np.array_equal(first, second)
    assert np.max(np.abs(first.astype(int) - pixels.astype(int))) <= 2


def test_smooth_warp_preserves_geometry_and_is_deterministic() -> None:
    pixels = _fixture()

    first = attack.smooth_warp(pixels, amplitude=0.35, sigma=8.0, seed=17)
    second = attack.smooth_warp(pixels, amplitude=0.35, sigma=8.0, seed=17)

    assert first.shape == pixels.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first, pixels)


def test_norm_matched_control_has_similar_rms(tmp_path: Path) -> None:
    pixels = _fixture()
    target = attack.quantize(pixels, 8)

    sham = attack.norm_matched_noise(pixels, target, seed=23)
    target_metrics = attack.measure(pixels, target, name="target", path=tmp_path / "target.png")
    sham_metrics = attack.measure(pixels, sham, name="sham", path=tmp_path / "sham.png")

    assert sham_metrics.residual_rms == pytest.approx(target_metrics.residual_rms, rel=0.1)


def test_crop_visible_badge() -> None:
    pixels = _fixture()

    cropped = attack.crop_visible_badge(pixels, 16)

    assert cropped.shape == (112, 144, 3)

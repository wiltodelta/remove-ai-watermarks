"""124-d freeze extractor: shape and the undersize abstain, no downloads."""

from __future__ import annotations

import numpy as np

from remove_ai_watermarks._internal.forensic_124d import FEATURE_WIDTH, PATCH, image_features


def test_a_full_patch_returns_124_floats() -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (PATCH, PATCH, 3), dtype=np.uint8)
    vector = image_features(pixels)
    assert vector is not None
    assert vector.shape == (FEATURE_WIDTH,)
    assert vector.dtype == np.float64


def test_a_larger_image_still_returns_124_floats() -> None:
    pixels = np.random.default_rng(1).integers(0, 256, (512, 640, 3), dtype=np.uint8)
    vector = image_features(pixels)
    assert vector is not None
    assert vector.shape == (FEATURE_WIDTH,)


def test_an_undersized_image_returns_none() -> None:
    pixels = np.zeros((PATCH - 1, PATCH, 3), dtype=np.uint8)
    assert image_features(pixels) is None


def test_non_rgb_returns_none() -> None:
    pixels = np.zeros((PATCH, PATCH), dtype=np.uint8)
    assert image_features(pixels) is None

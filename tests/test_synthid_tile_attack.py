from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_tile_attack as attack
from synthid_periodic_tile import fold_residual_template


def test_modulo_folding_recovers_repeated_high_frequency_tile() -> None:
    tile = np.fromfunction(lambda y, x, channel: ((x + y + channel) % 2) * 2.0 - 1.0, (8, 16, 3))
    pixels = 100.0 + np.tile(tile, (8, 4, 1))

    estimated = fold_residual_template(
        pixels,
        tile_height=8,
        tile_width=16,
        denoise_sigma=1.0,
    )

    correlation = np.corrcoef(tile.ravel(), estimated.ravel())[0, 1]
    assert correlation > 0.99


def test_subtraction_reduces_repeated_tile_energy() -> None:
    tile = np.fromfunction(lambda y, x, channel: ((x + y + channel) % 2) * 2.0 - 1.0, (8, 16, 3))
    pixels = np.clip(np.rint(100.0 + 4.0 * np.tile(tile, (8, 4, 1))), 0, 255).astype(np.uint8)
    template = fold_residual_template(
        pixels,
        tile_height=8,
        tile_width=16,
        denoise_sigma=1.0,
    )

    result = attack.subtract_tiled_template(pixels, template, strength=1.0)
    before = np.std(pixels.astype(np.float64) - np.mean(pixels, axis=(0, 1), keepdims=True))
    after = np.std(result.astype(np.float64) - np.mean(result, axis=(0, 1), keepdims=True))

    assert after < before


def test_folding_accepts_nondivisible_geometry() -> None:
    pixels = np.zeros((63, 64, 3), dtype=np.uint8)

    folded = fold_residual_template(
        pixels,
        tile_height=8,
        tile_width=16,
        denoise_sigma=1.0,
    )

    assert folded.shape == (8, 16, 3)
    assert np.count_nonzero(folded) == 0


def test_subtraction_accepts_nondivisible_geometry() -> None:
    pixels = np.full((5, 7, 3), 100, dtype=np.uint8)
    template = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)

    result = attack.subtract_tiled_template(pixels, template, strength=1.0)

    expected_template = np.tile(template, (3, 3, 1))[:5, :7]
    expected = np.rint(100.0 - expected_template).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
